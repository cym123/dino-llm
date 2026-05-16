# 固定写法：让类型提示更兼容
from __future__ import annotations

# 装饰器工具
import functools
# 上下文管理器工具（with 语句用）
from contextlib import contextmanager
# 类型检查标记
from typing import TYPE_CHECKING

# 仅给编辑器看的类型提示
if TYPE_CHECKING:
    import torch


# ==========================================
# 1. 临时切换 PyTorch 默认精度（超级重要）
# 使用方式：with torch_dtype(torch.bfloat16):
# ==========================================
@contextmanager  # 变成 with 语句能用的函数
def torch_dtype(dtype: torch.dtype):
    import torch  # 用到时才导入，加快启动速度

    # 保存【原来的】默认精度
    old_dtype = torch.get_default_dtype()
    # 设置【新的】精度
    torch.set_default_dtype(dtype)

    try:
        yield  # 执行 with 里面的代码
    finally:
        # 无论是否报错，都【恢复原来的精度】
        torch.set_default_dtype(old_dtype)


# ==========================================
# 2. 性能分析装饰器：给模型层打标记（测速/调试用）
# ==========================================
def nvtx_annotate(name: str, layer_id_field: str | None = None):
    # NVTX：NVIDIA 性能分析工具标记
    import torch.cuda.nvtx as nvtx

    # 装饰器
    def decorator(fn):
        # 保留原函数信息（不破坏函数名、文档）
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            # 如果需要显示层ID（比如 layer_1、layer_2）
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))

            # 给这段代码加性能标记
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper
    return decorator