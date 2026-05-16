from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
# 采样参数：控制生成策略（temperature、max_tokens等）
from dinollm.core import SamplingParams
# 分布式信息：记录卡数、当前卡号
from dinollm.distributed import DistributedInfo
# 消息定义：用户请求消息、模型返回的生成消息
from dinollm.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
# 调度器父类：负责模型推理、KV缓存、批处理
from dinollm.scheduler import Scheduler, SchedulerConfig


# 自定义异常：当所有请求都生成完毕时抛出，用于退出循环
class RequestAllFinished(Exception):
    pass


# 请求状态类：记录每个请求的 编号、输入token、输出token
@dataclass
class RequestStatus:
    uid: int              # 请求唯一ID（0、1、2...自增）
    input_ids: List[int]  # 用户输入的token ids
    output_ids: List[int] # 模型生成的token ids（动态追加）


# LLM 类：对外接口 + 调度器封装，用户直接调用这个类生成文本
class LLM(Scheduler):
    # 初始化：加载模型、初始化调度器
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        # 构造调度器配置
        config = SchedulerConfig(
            model_path=model_path,        # 模型路径
            tp_info=DistributedInfo(0, 1),# 张量并行：0号卡，共1张卡
            dtype=dtype,                  # 精度：bf16/fp16
            offline_mode=True,            # 离线推理模式
            **kwargs
        )
        # 初始化父类 Scheduler（真正加载模型、KV缓存、分词器）
        super().__init__(config)

        # 待处理请求队列：保存 (prompt字符串/token, 采样参数)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        # 请求状态映射：uid → RequestStatus，记录每个请求的输入输出
        self.status_map: Dict[int, RequestStatus] = {}
        # 请求ID计数器：每来一个新请求 +1
        self.counter = 0

    # 工具函数：把输入（字符串 或 token列表）转成模型需要的tensor
    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            # 如果是字符串，用tokenizer编码成token
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            # 如果已经是token，直接转成tensor
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    # 离线模式：从等待队列中取出请求，包装成消息交给调度器
    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # 如果阻塞模式且没有待处理请求，抛出结束异常
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()

        # 要返回给调度器的消息列表
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0

        # 遍历等待队列，按预填充预算（prefill_budget）取请求
        for tokens_or_prompt, sampling_params in self.pending_requests:
            # 超过单次预填充预算就停止，避免OOM
            if sum_input_len >= self.prefill_budget:
                break

            # 把输入编码成token
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)

            # 分配请求UID
            uid, added = self.counter + added, added + 1
            # 构造用户请求消息，发给调度器
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))

            # 记录这个请求的初始状态
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt,
                output_ids=[],  # 初始输出为空
            )

        # 更新计数器
        self.counter += added
        # 移除已经处理的请求
        self.pending_requests = self.pending_requests[added:]
        return results

    # 接收模型生成结果，把新token追加到对应请求的output_ids
    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            # 找到对应请求
            status = self.status_map[msg.uid]
            # 如果请求未结束，且不是结束符，就追加token
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    # 对外接口：用户直接调用 generate([prompt1, prompt2]) 生成结果
    def generate(
        self,
        prompts: List[str] | List[List[int]],       # 输入prompt列表
        sampling_params: List[SamplingParams] | SamplingParams, # 采样参数
    ) -> List[Dict[str, str | List[int]]]:

        # 重置状态
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0

        # 如果只传了一个采样参数，给所有prompt使用同一个
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        # 把所有prompt加入等待队列
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))

        # 运行调度器，直到所有请求完成
        try:
            self.run_forever()
        except RequestAllFinished:
            pass

        # 收集结果，解码成文本返回
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            # token → 文本
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({
                "text": output_text,
                "token_ids": status.output_ids
            })

        return results
