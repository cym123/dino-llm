from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

# 工具：内核配置、编译CUDA、构建编译参数
from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

# 默认索引内核配置：128线程， occupancy=1，不使用PDL
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


# 缓存编译好的CUDA索引内核
# 相同参数只编译一次，超级快
@functools.cache
def _jit_index_module(
    element_size: int,        # 每个元素的字节大小
    *,
    num_splits: int = 1,      # 切分份数（加速内存访问）
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    # 构建编译参数
    args = make_cpp_args(element_size, num_splits, *config)
    # 编译 index.cu 得到可调用的CUDA模块
    return load_jit(
        "index",          # 模块名字
        *args,            # 编译参数
        cuda_files=["index.cu"],  # 要编译的CUDA文件
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],  # 运行入口
    )


# 【核心函数】高速索引：根据 indices 从 weights 里取数据
def indexing(
    weights: torch.Tensor,     # 大权重表（例如：词表、专家权重）
    indices: torch.Tensor,     # 要取哪些行的序号
    *,
    output: torch.Tensor | None = None,  # 输出张量
    vocab_range: Tuple[int, int] | None = None,  # 词表范围（可选）
) -> torch.Tensor:
    # 如果没给输出，自动创建输出
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    # 计算每个“条目”占多少字节
    element_size = weights.shape[1] * weights.element_size()

    # 根据大小自动选择切分数，让内存访问更快
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1

    # 拿到编译好的高速CUDA索引内核
    module = _jit_index_module(element_size, num_splits=num_splits)

    # 调用CUDA内核，执行索引
    module.launch(weights, indices, output, vocab_range)

    return output
