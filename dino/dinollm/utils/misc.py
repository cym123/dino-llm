# 固定写法：让类型提示更兼容，不用管原理
from __future__ import annotations



def call_if_main(name: str = "__main__", discard: bool | None = None):

    if name != "__main__":

        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f



def div_even(a: int, b: int, allow_replicate: bool = False) -> int:

    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b



def div_ceil(a: int, b: int) -> int:

    return (a + b - 1) // b



def align_ceil(a: int, b: int) -> int:

    return div_ceil(a, b) * b



def align_down(a: int, b: int) -> int:

    return (a // b) * b



class Unset:
    pass


UNSET = Unset()