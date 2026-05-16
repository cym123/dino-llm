from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# 序列化工具：把对象转字典 / 字典转对象（用于传输）
from .utils import deserialize_type, serialize_type


# =========================================================
# 【前端消息基类】
# 作用：所有“后端 → 前端”的消息都继承这个类
# 负责：消息的序列化（发出去）、反序列化（收回来）
# =========================================================
@dataclass
class BaseFrontendMsg:
    # 静态方法：把消息对象 → 字典（用于网络/进程传输）
    @staticmethod
    def encoder(msg: "BaseFrontendMsg") -> Dict:
        return serialize_type(msg)

    # 静态方法：把字典 → 消息对象
    @staticmethod
    def decoder(json: Dict) -> "BaseFrontendMsg":
        return deserialize_type(globals(), json)


# =========================================================
# 批量前端消息
# 作用：一次返回多个结果给前端，提高效率
# =========================================================
@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    # 存放多个子消息（比如多个 UserReply）
    data: List[BaseFrontendMsg]


# =========================================================
# 【最重要】用户回复消息
# 后端 → 前端 发送生成结果时，用的就是这个类
# =========================================================
@dataclass
class UserReply(BaseFrontendMsg):
    uid: int                  # 对应请求的编号（和 UserMsg 里的 uid 对应）
    incremental_output: str   # 本次生成的【增量文本】（流式输出用）
    finished: bool           # 标记：这个请求是否生成完毕（True=结束）