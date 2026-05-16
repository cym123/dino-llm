from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from dinollm.core import Batch, Req


# =========================================================
# DecodeManager：解码阶段管理器
# 作用：专门管理【推理生成阶段（decode phase）】的请求
# 也就是：已经完成预填充，现在一步一步生成 token 的阶段
# =========================================================
@dataclass
class DecodeManager:
    # 一页存多少个 token（和 PagedAttention 保持一致）
    page_size: int

    # 正在 decode 阶段运行的请求集合
    running_reqs: Set[Req] = field(default_factory=set)

    # ------------------------------
    # 过滤请求：只保留“可以进入decode”的请求
    # ------------------------------
    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        # 合并新请求 → 只保留 can_decode = True 的
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    # 从运行队列中移除一个请求（结束/取消时用）
    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    # 根据 uid 取消一个请求
    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    # ------------------------------
    # 【调度核心】计算当前正在占用的 token 数量
    # 用于控制并发，防止 OOM
    # ------------------------------
    @property
    def inflight_tokens(self) -> int:
        # 预留空间：每个正在运行的请求，预留 1 页的空间（防止扩容）
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)
        # 总占用 = 所有请求剩余生成长度 + 预留页空间
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    # ------------------------------
    # 调度下一个 batch：把所有可运行的 decode 请求打包成一批
    # ------------------------------
    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        # 打包成一个 decode 批次
        return Batch(reqs=list(self.running_reqs), phase="decode")

    # 是否有可运行的请求
    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
