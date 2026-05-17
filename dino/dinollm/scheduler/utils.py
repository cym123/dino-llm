from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from dinollm.core import SamplingParams  # 采样参数（max_tokens/temp等）
    from .prefill import ChunkedReq         # 分块预填充的请求（长文本分块）


# ======================================================
# PendingReq：【还没被调度运行的请求】
# 作用：保存用户请求的所有信息，等待被调度器执行
# ======================================================
@dataclass
class PendingReq:
    uid: int                  # 请求唯一ID
    input_ids: torch.Tensor   # 输入的token序列（CPU张量）
    sampling_params: SamplingParams  # 生成参数
    chunked_req: ChunkedReq | None = None  # 长文本分块预填充对象

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


# ======================================================
# ScheduleResult：【调度结果】
# 调度器决定：这批请求可以跑了，返回这个结果
# ======================================================
@dataclass
class ScheduleResult:
    reqs: List[PendingReq]        # 本次要跑的请求列表
    output_indices: List[torch.Tensor]  # 每个请求要输出的位置（KV缓存里的位置）