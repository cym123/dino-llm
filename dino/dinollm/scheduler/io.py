from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from dinollm.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from dinollm.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


# =========================================================
# SchedulerIOMixin
# 调度器的【网络/进程通信模块】
# 作用：
# 1. 接收用户请求
# 2. 发送生成结果
# 3. 多卡（Tensor Parallel）广播消息
# 4. 对接 Tokenizer 进程
# =========================================================
class SchedulerIOMixin:

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group

        # ======================
        # 离线模式：不启动网络
        # 直接本地函数调用
        # ======================
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return

        # ======================
        # 主卡（rank 0）初始化
        # 1. 接收用户请求
        # 2. 发送结果给分词器
        # ======================
        if tp_info.is_primary():
            # 从前端/用户进程拉取消息
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            # 发送结果到分词器进程
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        # ======================
        # 选择消息收发方式
        # 单卡 / 多卡
        # ======================
        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0

        if tp_info.size > 1:
            # 多卡：主卡（rank0）
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
                # 广播消息给其他卡
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            # 多卡：从卡（rank1+）
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                # 从主卡接收广播
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        # 绑定最终使用的收发函数
        self.receive_msg = recv
        self.send_result = send

    # ======================
    # 空闲时运行（子类实现）
    # ======================
    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    # ======================
    # 离线模式接口（子类实现）
    # ======================
    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        raise NotImplementedError("should be implemented")

    # ======================
    # 多卡同步：CPU 端 barrier
    # ======================
    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    # ======================
    # 单卡：接收消息
    # ======================
    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    # ======================
    # 多卡主卡：接收 + 广播给所有从卡
    # ======================
    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # 广播消息数量给所有从卡
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        # 广播每条消息
        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    # ======================
    # 多卡从卡：从主卡接收消息
    # ======================
    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # 同步消息数量
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        # 接收所有广播消息
        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    # ======================
    # 主卡：回复结果给分词器
    # ======================
    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))

    # ======================
    # 从卡：不发送结果（只由主卡发）
    # ======================
    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        _ = reply