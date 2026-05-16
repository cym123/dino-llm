from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

# 加载提前编译好的 CUDA 代码
from dinollm.core import ENV

from .utils import load_aot

# --------------------------
# 类型检查阶段（给编辑器看，不运行）
# --------------------------
if TYPE_CHECKING:
    from abc import abstractmethod

    import torch


    class PyNCCLCommunicator:
        """
        定义 PyNCCL 通信器必须有的接口
        只是规范，不实现
        """
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...

        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...

        @abstractmethod
        def get_buffer(self) -> int: ...

# 运行时，把它变成任意类型
else:
    PyNCCLCommunicator = Any


# --------------------------
# 缓存加载：编译好的 NCCL 底层模块（.so 文件）
# 只加载一次，全局复用
# --------------------------
@functools.cache
def _load_nccl_module():
    # 加载编译好的 pynccl CUDA 模块
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


# --------------------------
# 缓存获取：NCCL 包装类
# 对接 TVM FFI，让 Python 调用 C++/CUDA 代码
# --------------------------
@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("dinollm.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


# --------------------------
# 【核心函数】初始化 PyNCCL 高速通信
# --------------------------
def init_pynccl(
    *,
    tp_rank: int,            # 当前是第几张卡
    tp_size: int,            # 总共有几张卡
    tp_cpu_group: torch.distributed.ProcessGroup,  # CPU 通信组（用来握手）
    max_size_bytes: int = 0, # 最大通信缓冲区大小
) -> PyNCCLCommunicator:
    import torch

    # 限制缓冲区最大大小，防止爆显存
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    # 1. 加载底层 CUDA 模块
    module = _load_nccl_module()
    # 2. 获取包装类（Python ↔ C++ 桥梁）
    cls = _get_pynccl_wrapper_cls()

    # --------------------------
    # 关键步骤：所有卡获取同一个 NCCL ID
    # 必须让所有 GPU 知道同一个 ID，才能互相通信
    # --------------------------
    if tp_rank == 0:
        # 主卡生成唯一的 NCCL UID
        id_list = [module.create_nccl_uid()]
        # 广播给所有其他卡
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        # 其他卡接收主卡发来的 NCCL ID
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    # 拿到全局统一的 NCCL ID
    nccl_id = id_list[0]
    assert nccl_id is not None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # 创建真正的 PyNCCL 通信器（C++ 对象）
    # 这个对象可以直接调用 all_reduce / all_gather
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
