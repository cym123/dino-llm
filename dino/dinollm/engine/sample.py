from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from dinollm.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from dinollm.core import Batch



@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None  # 温度值
    top_k: torch.Tensor | None = None  # top_k
    top_p: torch.CudaTensor | None = None  # top_p



def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


# --------------------------
# 真正的采样实现
# 输入：logits（模型原始输出）
# 输出：下一个 token
# --------------------------
def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    # 调用 FlashInfer 高性能采样
    import flashinfer.sampling as sampling

    # 1. 对 logits 做 softmax，同时应用温度系数
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())

    # 2. 根据参数选择采样方式
    if top_k is None and top_p is None:
        # 普通随机采样
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        # 只使用 top_k
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        # 只使用 top_p
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    # 同时使用 top_k + top_p
    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


# --------------------------
# 采样器主类：给一批请求生成 token
# --------------------------
@dataclass
class Sampler:
    device: torch.device        # 运行设备：GPU
    vocab_size: int             # 词表大小（如 32000）

    # --------------------------
    # 准备一批请求的采样参数
    # 把每个请求的 temperature / top_k / top_p 传到 GPU
    # --------------------------
    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        # 拿到这批请求的所有采样参数
        params = [r.sampling_params for r in batch.reqs]

        # 如果全部是 greedy（直接选概率最大），直接返回空温度
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        # 下限值，防止除0
        MIN_P = MIN_T = 1e-6

        # 收集每个请求的参数
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]

        # 传到 GPU
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None

        # 只有参数不一样时，才传到GPU（优化）
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)

        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    # --------------------------
    # 【核心】执行采样：logits → token
    # --------------------------
    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            # 如果是 greedy：直接选概率最大的词
            if args.temperatures is None:
                return torch.argmax(logits, dim=-1)

            # 否则：调用 FlashInfer 采样
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
