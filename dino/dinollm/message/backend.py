from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
# 采样参数：生成时用的 temperature、max_tokens、top_k 等
from dinollm.core import SamplingParams

# 序列化工具：把对象 → JSON；把 JSON → 对象
from .utils import deserialize_type, serialize_type


# =========================================================
# 【所有后端消息的基类】
# 作用：定义所有消息必须支持 序列化(encoder) / 反序列化(decoder)
# 所有消息（UserMsg/ExitMsg 等）都继承它
# =========================================================
@dataclass
class BaseBackendMsg:
    # 把当前消息对象 → 字典（用于网络传输/保存）
    def encoder(self) -> Dict:
        return serialize_type(self)

    # 把字典 → 恢复成消息对象
    @staticmethod
    def decoder(json: Dict) -> "BaseBackendMsg":
        return deserialize_type(globals(), json)


# =========================================================
# 批量消息：把多个消息打包成一个发送
# =========================================================
@dataclass
class BatchBackendMsg(BaseBackendMsg):
    # 里面装一堆子消息（比如多个 UserMsg）
    data: List[BaseBackendMsg]


# =========================================================
# 退出消息：告诉后端“停止运行、退出服务”
# =========================================================
@dataclass
class ExitMsg(BaseBackendMsg):
    pass


# =========================================================
# 【最核心】用户请求消息
# 前端 → 后端 发请求时，用的就是这个消息
# =========================================================
@dataclass
class UserMsg(BaseBackendMsg):
    uid: int                  # 请求唯一编号（0、1、2、3...）
    input_ids: torch.Tensor   # 输入的 token 序列（CPU 一维 int32）
    sampling_params: SamplingParams  # 生成参数（怎么采样、生成多长）


# =========================================================
# 终止请求消息：让后端停止某个 uid 的生成
# =========================================================
@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int  # 要终止的请求 ID