from __future__ import annotations

from dataclasses import dataclass

#frozen=True 创建后只读无法修改
@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    #当前进程编号（比如第 0 号 GPU、第 1 号 GPU）
    rank: int
    #总共有多少个进程（比如一共 4 张 GPU）
    size: int

    #检查数据是否正确 造好对象后自动运行
    def __post_init__(self):
        assert 0 <= self.rank < self.size

    #判断是不是 “主进程 / 主卡”
    def is_primary(self) -> bool:
        return self.rank == 0


#创建一个全局变量，存分布式信息 全局唯一的 “分布式信息存储器”
_TP_INFO: DistributedInfo | None = None


#设置分布式信息（只能设一次）
def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


#获取分布式信息（没设置就报错）
def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


#尝试获取分布式信息（不报错）
def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


#规定别人导入时能用哪些东西
__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
