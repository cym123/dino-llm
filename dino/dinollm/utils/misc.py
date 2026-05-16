# 固定写法：让类型提示更兼容，不用管原理
from __future__ import annotations


# ==========================
# 1. 装饰器：只有直接运行脚本才执行函数
# ==========================
def call_if_main(name: str = "__main__", discard: bool | None = None):
    """
    装饰器作用：
    当文件被【直接运行】时，自动执行函数
    当文件被【导入】时，不执行
    """
    # 如果不是主程序（是被别人import导入）
    if name != "__main__":
        # 默认不执行函数
        discard = False if discard is None else discard
        if discard:
            # 返回空函数，啥也不干
            return lambda _: None
        else:
            # 返回原函数，不执行
            return lambda f: f
    # 如果是主程序（直接运行这个文件）
    else:
        discard = True if discard is None else discard
        if discard:
            # 执行函数，执行完就扔掉，不保存返回值
            return lambda f: (f() or True) and None
        else:
            # 执行函数，保留函数本身
            return lambda f: (f() and None) or f


# ==========================
# 2. 安全除法（大模型专用：切分头/张量）
# ==========================
def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """
    整数除法，要求 a 必须能被 b 整除（没有余数）
    专门用于：模型的头数、张量切分
    """
    # 特殊情况：允许复制（b比a大，且能整除）
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    # 强制检查：a 必须能被 b 整除，否则报错
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    # 返回整除结果
    return a // b


# ==========================
# 3. 向上取整除法
# ==========================
def div_ceil(a: int, b: int) -> int:
    """
    除法后【向上取整】
    例子：5 / 2 = 2.5 → 返回 3
    公式：(a + b - 1) // b
    """
    return (a + b - 1) // b


# ==========================
# 4. 向上对齐到 b 的整数倍
# ==========================
def align_ceil(a: int, b: int) -> int:
    """
    把数字 a 向上对齐到 b 的倍数
    例子：a=5, b=2 → 找 ≥5 的最小2的倍数 → 6
    """
    return div_ceil(a, b) * b


# ==========================
# 5. 向下对齐到 b 的整数倍
# ==========================
def align_down(a: int, b: int) -> int:
    """
    把数字 a 向下对齐到 b 的倍数
    例子：a=5, b=2 → 找 ≤5 的最大2的倍数 → 4
    """
    return (a // b) * b


# ==========================
# 6. 空值标记类：表示“未设置”
# ==========================
class Unset:
    """空类，用来标记一个值【没有被设置过】"""
    pass


# 全局唯一的“未设置”标记
UNSET = Unset()