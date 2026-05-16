from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# 采样参数（生成时的配置：max_tokens、temperature等）
from dinollm.core import SamplingParams

# 序列化工具：把消息对象 ↔ 字典（用于进程间通信）
from .utils import deserialize_type, serialize_type


# =========================================================
# 【分词器消息基类】
# 作用：所有和“分词/解码”相关的消息，都继承这个类
# 负责：序列化（发消息）、反序列化（收消息）
# =========================================================
@dataclass
class BaseTokenizerMsg:
    # 把消息对象 → 字典（用于发送/传输）
    @staticmethod
    def encoder(msg: "BaseTokenizerMsg") -> Dict:
        return serialize_type(msg)

    # 把字典 → 恢复成消息对象（用于接收）
    @staticmethod
    def decoder(json: Dict) -> "BaseTokenizerMsg":
        return deserialize_type(globals(), json)


# =========================================================
# 批量分词消息
# 作用：一次打包多条消息发给分词器，提高效率
# =========================================================
@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]  # 存放多条子消息


# =========================================================
# 【解码消息】模型生成了一个token → 转回文字
# 方向：Scheduler → Tokenizer 进程
# 作用：告诉分词器：把这个token转成文字
# =========================================================
@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int          # 对应哪个用户请求
    next_token: int   # 模型刚生成出来的 token id
    finished: bool    # 这个请求是否生成完毕（结束标记）


# =========================================================
# 【编码消息】用户输入文字 → 转成token
# 方向：Frontend → Tokenizer 进程
# 作用：把用户输入的字符串 → token ids
# =========================================================
@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int                      # 请求编号
    text: str | List[Dict[str, str]]  # 用户输入的文字
    sampling_params: SamplingParams   # 生成参数


# =========================================================
# 终止消息：取消某个请求的分词/解码
# =========================================================
@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int  # 要取消的请求ID