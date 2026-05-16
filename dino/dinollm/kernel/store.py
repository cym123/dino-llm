from __future__ import annotations

import functools
from typing import TYPE_CHECKING

# 工具：编译CUDA内核、加载内核、构建参数
from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

# 默认内核配置：128线程，满负载运行，不使用PDL
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


# 缓存编译好的 CUDA 存储内核（只编译一次）
@functools.cache
def _jit_store_module(
    element_size: int,        # 每个元素的字节大小
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    # 生成编译参数
    args = make_cpp_args(element_size, *config)
    # 编译 store.cu 得到高速CUDA内核
    return load_jit(
        "store",          # 内核名字
        *args,            # 编译参数
        cuda_files=["store.cu"],  # 要编译的CUDA文件
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],  # 运行入口
    )


# 【核心函数】把 K、V 高速存入 KV Cache
# 作用：把刚算出来的 k 和 v，存到 k_cache 和 v_cache 里
def store_cache(
    k_cache: torch.Tensor,   # K 缓存（大显存池）
    v_cache: torch.Tensor,   # V 缓存（大显存池）
    indices: torch.Tensor,   # 要存在缓存的哪个位置
    k: torch.Tensor,         # 刚计算出来的新 K
    v: torch.Tensor,         # 刚计算出来的新 V
) -> None:
    # 把 KV Cache 展平成 [num_tokens, hidden_dim]
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)

    # 计算每个 token 的 K+V 占多少字节（用于内存对齐）
    element_size = k_cache.shape[1] * k_cache.element_size()

    # 加载编译好的高速CUDA存储内核
    module = _jit_store_module(element_size)

    # 调用CUDA内核：把 k、v 存入对应 indices 位置
    module.launch(k_cache, v_cache, indices, k, v)
