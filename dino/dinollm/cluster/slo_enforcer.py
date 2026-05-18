from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List
import time

# 直接使用你定义的 WorkerMetrics
from .load_predictor import WorkerMetrics


class WorkerSLOState(Enum):
    NORMAL = 0       # 正常：可全量调度
    WARNING = 1      # 警告：轻微超标
    RESTRICTED = 2   # 受限：仅允许粘性会话
    BANNED = 3       # 封禁：完全不调度


@dataclass
class SLOConfig:
    # 延迟 SLO
    max_ttft_ms: float = 4000.0
    warn_ttft_ms: float = 2000.0

    max_tpot_ms: float = 100.0
    warn_tpot_ms: float = 40.0

    # 队列 SLO
    max_total_queue: int = 128
    warn_total_queue: int = 64

    max_wait_prefill_queue: int = 64
    warn_wait_prefill_queue: int = 32

    # KV 缓存 SLO
    min_kv_free_gb: float = 3.0
    warn_kv_free_gb: float = 6.0

    # 封禁策略
    ban_cooldown_sec: int = 30
    ban_breach_threshold: int = 2
    restrict_breach_threshold: int = 1
    restrict_warn_threshold: int = 3


class SLOEnforcer:
    def __init__(self, config: Optional[SLOConfig] = None):
        self.cfg = config or SLOConfig()
        self.ban_until: Dict[str, float] = {}

    def check_worker(self, metrics: WorkerMetrics) -> WorkerSLOState:
        """检查节点是否健康，返回 SLO 状态"""
        worker_id = metrics.worker_id
        now = time.time()

        # 封禁冷却检查
        if worker_id in self.ban_until:
            if now < self.ban_until[worker_id]:
                return WorkerSLOState.BANNED
            del self.ban_until[worker_id]

        cfg = self.cfg
        warn_count = 0
        breach_count = 0

        # ------------------------------
        # TTFT
        # ------------------------------
        if metrics.ttft_ms >= cfg.max_ttft_ms:
            breach_count += 1
        elif metrics.ttft_ms >= cfg.warn_ttft_ms:
            warn_count += 1

        # ------------------------------
        # TPOT
        # ------------------------------
        if metrics.tpot_ms >= cfg.max_tpot_ms:
            breach_count += 1
        elif metrics.tpot_ms >= cfg.warn_tpot_ms:
            warn_count += 1

        # ------------------------------
        # 总队列
        # ------------------------------
        total_q = metrics.prefill_queue_len + metrics.decode_queue_len
        if total_q >= cfg.max_total_queue:
            breach_count += 1
        elif total_q >= cfg.warn_total_queue:
            warn_count += 1

        # ------------------------------
        # 等待预填队列
        # ------------------------------
        wpq = metrics.wait_prefill_queue_len
        if wpq >= cfg.max_wait_prefill_queue:
            breach_count += 1
        elif wpq >= cfg.warn_wait_prefill_queue:
            warn_count += 1

        # ------------------------------
        # KV 空闲显存
        # ------------------------------
        kv_free = metrics.kv_cache_free_gb
        if kv_free <= cfg.min_kv_free_gb:
            breach_count += 1
        elif kv_free <= cfg.warn_kv_free_gb:
            warn_count += 1

        # ------------------------------
        # 状态判定
        # ------------------------------
        if breach_count >= cfg.ban_breach_threshold:
            self.ban_until[worker_id] = now + cfg.ban_cooldown_sec
            return WorkerSLOState.BANNED

        if breach_count >= cfg.restrict_breach_threshold or warn_count >= cfg.restrict_warn_threshold:
            return WorkerSLOState.RESTRICTED

        if warn_count >= 1:
            return WorkerSLOState.WARNING

        return WorkerSLOState.NORMAL

    def get_available_workers(self, metrics_list: List[WorkerMetrics]) -> List[WorkerMetrics]:
        """返回【未封禁】节点（可用于调度）"""
        return [m for m in metrics_list if self.check_worker(m) != WorkerSLOState.BANNED]

    def get_healthy_workers(self, metrics_list: List[WorkerMetrics]) -> List[WorkerMetrics]:
        """返回【健康节点】：NORMAL + WARNING（最推荐用于调度）"""
        allowed = {WorkerSLOState.NORMAL, WorkerSLOState.WARNING}
        return [m for m in metrics_list if self.check_worker(m) in allowed]