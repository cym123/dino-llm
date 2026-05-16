from __future__ import annotations

import functools
from typing import TYPE_CHECKING

# 加载提前编译好的 C++ 模块
from .utils import load_aot

# 类型检查（只给编辑器看，运行时不执行）
if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


# 缓存加载 C++ 模块
# 只加载一次，全局复用，不用重复加载
@functools.cache
def _load_test_tensor_module() -> Module:
    # 加载编译好的 tensor.cpp
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


# 【核心函数】C++ 加速：比较两个张量是否“形状 + 数据”完全一样
def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    """
    输入：两个 torch 张量 x, y
    输出：0 = 完全一样；1 = 不一样
    作用：引擎调试、验证计算正确性、单元测试
    """
    # 调用底层 C++ 函数做高速比较
    return _load_test_tensor_module().test(x, y)
