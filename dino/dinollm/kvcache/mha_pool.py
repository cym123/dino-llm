from __future__ import annotations

import torch
# 获取张量并行（多卡）信息
from dinollm.distributed import get_tp_info
# 工具函数：均匀除法（分给多张卡）
from dinollm.utils import div_even

# 继承基类（规则手册）
from .base import BaseKVCachePool


# ======================================================
# MHAKVCache = 标准注意力的 KV 缓存实现
# 作用：一次性开辟一大块GPU显存，给所有层、所有token存 K 和 V
# ======================================================
class MHAKVCache(BaseKVCachePool):


    def __init__(
        self,
        num_kv_heads: int,    # 模型总共有多少个 KV 头
        num_layers: int,     # 模型多少层
        head_dim: int,       # 每个头的维度
        num_pages: int,      # 一共开多少个页（显存块）
        page_size: int,      # 每个页能存多少个 token
        dtype: torch.dtype,  # 精度：fp16 / bf16
        device: torch.device,# 设备：GPU
    ) -> None:
        # --------------------------
        # 1. 处理多卡并行（TP = Tensor Parallel）
        # 如果是多张卡，每张卡只需要存一部分 KV 头
        # --------------------------
        tp_info = get_tp_info()
        # 总KV头 / 卡数 → 每张卡存多少个头
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)

        # --------------------------
        # 2. 【核心】一次性开辟巨大的空显存！
        # 形状：
        # 2 → [K缓存, V缓存]
        # num_layers → 每一层
        # num_pages → 多少个页
        # page_size → 每页多少token
        # local_kv_heads → 每张卡的头数
        # head_dim → 头维度
        # --------------------------
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        # --------------------------
        # 3. 把 K 和 V 分开指向同一块显存
        # 不用额外占空间，只是“引用”
        # --------------------------
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]  # K 缓存
        self._v_buffer = self._kv_buffer[1]  # V 缓存

        self._device = device

        # --------------------------
        # 4. 展平后的形状：[总token数, 头数, 头维度]
        # 给 store_cache 内核使用
        # --------------------------
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    # --------------------------
    # 获取第 index 层的 K 缓存
    # --------------------------
    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    # --------------------------
    # 获取第 index 层的 V 缓存
    # --------------------------
    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    # --------------------------
    # 【核心】把新计算的 K、V 存入缓存
    # 调用 CUDA 高速内核 store_cache
    # --------------------------
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from dinollm.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,  # 存在哪个位置
            k=k,  # 新 K
            v=v,  # 新 V
        )

    # --------------------------
    # 只读属性
    # --------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
