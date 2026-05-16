# 固定写法：让Python类型提示更友好，照抄就行
from __future__ import annotations

# functools.cache：给函数加缓存，运行一次就记住结果，不用重复算
import functools
# 类型提示：Tuple = 元组（比如 (9,0) 这种格式）
from typing import Tuple


# 缓存函数：只执行一次，后面直接返回结果
@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    # 导入 PyTorch
    import torch
    import torch.version

    # 如果：没有GPU / CUDA不可用 / 没有CUDA版本 → 返回 None
    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    
    # 获取显卡的算力版本（比如 RTX 4090 = (8,9)，H100 = (9,0)）
    return torch.cuda.get_device_capability()


# 通用判断：当前显卡 >= 指定的算力版本吗？
def is_arch_supported(major: int, minor: int = 0) -> bool:
    # 获取当前显卡算力
    arch = _get_torch_cuda_version()
    
    # 没有GPU → 不支持
    if arch is None:
        return False
    
    # 比较：当前显卡版本 >= 要求的版本
    return arch >= (major, minor)


# 判断：是否支持 算力9.0（H100 / H200 专用）
def is_sm90_supported() -> bool:
    return is_arch_supported(9, 0)


# 判断：是否支持 算力10.0（下一代显卡，Blackwell 架构）
def is_sm100_supported() -> bool:
    return is_arch_supported(10, 0)