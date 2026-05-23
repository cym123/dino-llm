from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from dinollm.core import Batch, Req
from dinollm.utils import init_logger
import time

from .utils import PendingReq

if TYPE_CHECKING:
    from dinollm.kvcache import BaseCacheHandle
    from dinollm.message import UserMsg
    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)



class ChunkedReq(Req):
    # 分块请求不参与采样
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    # 标记：禁止进入 decode 阶段
    @property
    def can_decode(self) -> bool:
        return False


# =========================================================
# PrefillAdder：预取请求添加器
# 作用：**给新请求分配资源、匹配缓存、加入调度**
# 核心：判断能不能加新请求、加多少、资源够不够
# =========================================================
@dataclass
class PrefillAdder:
    token_budget: int          # 本次 prefill 最大 token 数
    reserved_size: int         # 为 decode 阶段预留的显存
    cache_manager: CacheManager  # 缓存管理器
    table_manager: TableManager  # 页表管理器

    # ------------------------------
    # 尝试给一个请求分配资源（页表 + 缓存）
    # ------------------------------
    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        # 1. 没有空闲槽位 → 拒绝
        if self.table_manager.available_size == 0:
            return None

        # 2. 匹配前缀缓存，获取已缓存长度
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len

        print(f"cached_len : {cached_len}")

        # 3. 计算需要的显存
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # 4. 显存不够 → 拒绝
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None

        # 5. 锁定缓存，防止被淘汰
        self.cache_manager.lock(handle)

        # 6. 再次检查显存（防止中间被其他请求占用）
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        # 7. 分配页表槽位
        table_idx = self.table_manager.allocate()

        # 8. 如果有缓存，复制缓存的 token 和页表
        if cached_len > 0:
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        # 9. 返回分配好的资源
        return handle, table_idx

    # ------------------------------
    # 正式添加一个请求到 prefill 批次
    # ------------------------------
    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        # 剩余未填充长度
        remain_len = pending_req.input_len - cached_len
        # 本次最多填 chunk_size 个 token
        chunk_size = min(self.token_budget, remain_len)
        # 是否需要分块（一次填不完）
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req

        # 扣除预算
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len

        # 复制本次要 prefill 的 token 到 GPU
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)

        # 创建请求对象返回
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    # ------------------------------
    # 对外接口：尝试添加一个请求
    # ------------------------------
    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        # 预算耗尽
        if self.token_budget <= 0:
            return None

        # 如果是分块请求 → 继续填
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # 普通请求 → 尝试分配资源
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


# =========================================================
# PrefillManager：预取管理器
# 作用：**管理所有等待 prefill 的请求**
# 决定：哪些请求可以进模型、分块、资源检查
# =========================================================
@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)  # 等待队列

    # 添加一个用户请求到等待队列
    def add_one_req(self, req: UserMsg) -> None:
            # 携带优先级，无则默认0
        prio = getattr(req, "priority", 0)
        pending = PendingReq(
            uid=req.uid,
            input_ids=req.input_ids,
            sampling_params=req.sampling_params,
            priority=prio,
            arrive_time=time.time()
        )
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    # ------------------------------
    # 调度下一个 prefill 批次（核心）
    # ------------------------------
    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        # 没有等待请求
        if len(self.pending_list) == 0:
            return None

        # 创建请求添加器
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )

        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []

        # 遍历等待列表，尽可能添加请求
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                # 如果是分块请求 → 保留到下一轮
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # 资源不足，停止添加

        # 没有可添加的请求
        if len(reqs) == 0:
            return None

        # 更新等待列表：分块请求优先
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    # 取消请求
    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    # 是否有可运行请求
    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
