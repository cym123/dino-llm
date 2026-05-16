from __future__ import annotations

# 抽象基类相关：用于定义必须实现的接口
from abc import ABC, abstractmethod
# 数据类：简化类定义，不用写__init__
from dataclasses import dataclass
# 类型检查相关：仅给编辑器提示用，运行时不执行
from typing import TYPE_CHECKING, List

import torch
# PyTorch 分布式通信库
import torch.distributed as dist

# 仅用于类型检查，避免运行时循环导入
if TYPE_CHECKING:
    from dinollm.distributed import DistributedInfo
    from dinollm.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    """
    分布式通信的**抽象基类**
    规定所有通信实现必须实现 all_reduce 和 all_gather 两个方法
    作用：统一接口，方便切换不同的通信后端
    """
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """多卡数据归约求和，所有卡得到相同结果"""
        ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """收集所有卡的数据并拼接，所有卡得到完整结果"""
        ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """
    基于 PyTorch 原生分布式的通信实现
    优点：兼容性好
    缺点：速度不如 NCCL 定制版
    """
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # 获取张量并行的总卡数
        tp_size = dist.get_world_size()
        # 单卡无需通信，直接返回
        if tp_size == 1:
            return x
        # 执行多卡求和归约
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        # 单卡无需收集，直接返回
        if tp_size == 1:
            return x
        # 构造输出形状：第一维扩大 tp_size 倍
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        # 创建空的输出张量
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        # 执行 all_gather，把所有卡的数据收集到 out
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """
    基于 PyNCCL 的高性能分布式通信实现
    NVIDIA 官方 GPU 高速通信，比 PyTorch 原生更快
    """
    # NCCL 通信器实例
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # 调用 NCCL 执行求和归约
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        # 导入获取 TP 信息的工具
        from .info import get_tp_info

        # 获取总卡数
        world_size = get_tp_info().size
        # 构造输出形状
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        # 创建输出张量
        result = x.new_empty(output_shape)
        # NCCL 执行 all_gather
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """
    分布式通信统一入口（核心类）
    插件式架构：默认用 torch 原生，可追加 NCCL 插件切换高速模式
    永远使用最后添加的插件（优先级最高）
    """
    # 插件列表：默认加载 PyTorch 原生通信实现
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # 调用最后一个插件的 all_reduce 实现
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        # 调用最后一个插件的 all_gather 实现
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    启用 PyNCCL 高性能通信
    会把 NCCL 实现追加到插件列表，自动切换为高速模式
    """
    # 单卡不需要 NCCL，直接返回
    if tp_info.size == 1:
        return
    # 延迟导入，避免循环依赖
    from dinollm.kernel import init_pynccl

    # 初始化 NCCL 通信器
    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    # 把 NCCL 实现加入插件列表（自动成为最新、最高优先级）
    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    销毁所有分布式通信插件
    用于释放资源、重置状态
    """
    DistributedCommunicator.plugins = []
