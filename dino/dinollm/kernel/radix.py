from __future__ import annotations

import functools
from typing import TYPE_CHECKING

# 加载提前编译好的 C++ 模块
from .utils import load_aot

# 只给编辑器看的类型提示
if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


# 缓存加载 C++ 模块（只加载一次，全局复用）
@functools.cache
def _load_radix_module() -> Module:
    # 加载编译好的 radix.cpp
    return load_aot("radix", cpp_files=["radix.cpp"])


# 【核心函数】快速比较两个 1D int 张量是否完全一样
def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """
    比较两个 1维 int 型 CPU 张量是否完全相等
    返回：
        0 = 完全一样
        1 = 不一样
    """
    # 调用 C++ 实现的超快比较函数
    return _load_radix_module().fast_compare_key(x, y)
