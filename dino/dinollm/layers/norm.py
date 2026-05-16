from typing import Tuple

import torch

from .base import BaseOP
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()

        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = nn.Parameter(torch.empty(size))
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = nn.Parameter(torch.empty(size))
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
