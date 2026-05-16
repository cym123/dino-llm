from __future__ import annotations

from dataclasses import dataclass
# 缓存属性：只计算一次，后续直接用（加速）
from functools import cached_property
# 类型检查：仅给编辑器提示用，运行时不执行
from typing import TYPE_CHECKING, List

import torch
# 分布式信息：rank / size（你之前看过的那个类）
from dinollm.distributed import DistributedInfo
# 工具函数：缓存加载 HuggingFace 模型配置
from dinollm.utils import cached_load_hf_config

# 仅用于类型检查，避免运行时循环导入
if TYPE_CHECKING:
    from dinollm.models import ModelConfig


@dataclass(frozen=True)  # frozen=True = 实例创建后不可修改（安全）
class EngineConfig:
    """
    【引擎核心配置类】
    整个 LLM 推理引擎的所有配置都存在这里：模型路径、精度、并行、内存、KV缓存、优化开关等
    只读配置，创建后不能修改，避免运行时乱改导致崩溃
    """
    # -------------------------- 基础必选配置 --------------------------
    model_path: str                      # 模型路径（本地文件夹 / HuggingFace 模型名）
    tp_info: DistributedInfo             # 张量并行信息（rank / size）
    dtype: torch.dtype                   # 推理精度（float16 / bfloat16 / int8 等）

    # -------------------------- 性能 & 调度配置 --------------------------
    max_running_req: int = 256           # 最大同时运行的请求数（控制并发）
    attention_backend: str = "auto"       # Attention 计算后端（flash/flashinfer/auto）
    moe_backend: str = "auto"            # MoE 模型后端（自动选择最优）

    # -------------------------- CUDA Graph 加速配置 --------------------------
    cuda_graph_bs: List[int] | None = None  # 启用 CUDA Graph 的 batch size 列表
    cuda_graph_max_bs: int | None = None    # CUDA Graph 最大 batch size

    # -------------------------- Paged Attention 配置 --------------------------
    page_size: int = 1                   # KV Cache 块大小（PagedAttention）
    memory_ratio: float = 0.9             # GPU 显存占用比例（最多用90%显存）
    num_page_override: int | None = None  # 手动覆盖 KV Cache 块数量（高级）

    # -------------------------- 分布式 & 超时 --------------------------
    distributed_timeout: float = 60.0     # 分布式通信超时时间（秒）

    # -------------------------- 调试 & 开发 --------------------------
    use_dummy_weight: bool = False        # 是否使用假权重（测试用，不加载真实权重）
    use_pynccl: bool = True               # 是否启用 PyNCCL 高速通信

    # -------------------------- 序列长度 --------------------------
    max_seq_len_override: int | None = None  # 手动覆盖模型最大长度（高级）

    # ==================== 缓存属性（只算一次，加速访问） ====================
    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:

        from dinollm.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    # ==================== 计算属性（动态获取，不存储） ====================
    @property
    def max_seq_len(self) -> int:

        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
