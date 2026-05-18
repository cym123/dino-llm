from __future__ import annotations
import time
from typing import Dict, Set, Optional
from .load_predictor import WorkerMetrics

class PredictiveEarlyRejector:
    """
    预测性早期拒绝器
    功能：
    1. 节点高负载提前拒绝
    2. 全局 QPS 限流
    3. 用户级 QPS 限流
    4. 【用户级】Token 配额保护（每个用户独立额度）
    5. 黑名单拦截
    6. 白名单跳过所有限流
    """
    def __init__(
        self,
        max_global_qps: int = 100,
        max_user_qps: int = 10,
        max_user_token_quota: int = 100000000,  # 现在是：每个用户最大token
        high_load_score_threshold: float = 1
    ):
        # 配置
        self.high_load_score_threshold = high_load_score_threshold
        self.max_global_qps = max_global_qps
        self.max_user_qps = max_user_qps
        self.max_user_token_quota = max_user_token_quota  # 改名：用户级token限额

        # 状态
        self.blacklist: Set[str] = set()
        self.whitelist: Set[str] = set()

        # ========================
        # 关键改动：用户维度 token 计数
        # ========================
        self.user_token_used: Dict[str, int] = {}  # user_id → used_tokens

        # QPS 统计
        self.global_qps_tracker: Dict[int, int] = {}
        self.user_qps_tracker: Dict[str, Dict[int, int]] = {}

    # -------------------------------------------------------------------------
    # 黑白名单
    # -------------------------------------------------------------------------
    def add_to_blacklist(self, user_id: str):
        self.blacklist.add(user_id)

    def add_to_whitelist(self, user_id: str):
        self.whitelist.add(user_id)

    def remove_blacklist(self, user_id: str):
        self.blacklist.discard(user_id)

    def remove_whitelist(self, user_id: str):
        self.whitelist.discard(user_id)

    # -------------------------------------------------------------------------
    # 核心：提前拒绝检查
    # -------------------------------------------------------------------------
    def should_reject(
        self,
        user_id: Optional[str] = None,
        token_count: int = 0,
        worker_load_score: float = 0.0
    ) -> tuple[bool, str]:
        """
        return: (是否拒绝, 拒绝原因)
        """
        # 白名单直接放行
        if user_id and user_id in self.whitelist:
            return False, "whitelist_skip_all"

        # 黑名单直接拒绝
        if user_id and user_id in self.blacklist:
            return True, "blacklisted_user"

        # 节点高负载
        if worker_load_score >= self.high_load_score_threshold:
            return True, f"worker_high_load_score={worker_load_score:.2f}"

        # ========================
        # 关键改动：用户级 Token 配额检查
        # ========================
        if user_id and token_count > 0:
            used = self.user_token_used.get(user_id, 0)
            if used + token_count > self.max_user_token_quota:
                return True, "user_token_quota_exceeded"

        # 全局 QPS
        now = int(time.time())
        global_cnt = self.global_qps_tracker.get(now, 0)
        if global_cnt >= self.max_global_qps:
            return True, "global_qps_limit"
        self.global_qps_tracker[now] = global_cnt + 1

        # 用户 QPS
        if user_id:
            user_tracker = self.user_qps_tracker.setdefault(user_id, {})
            user_cnt = user_tracker.get(now, 0)
            if user_cnt >= self.max_user_qps:
                return True, "user_qps_limit"
            user_tracker[now] = user_cnt + 1

        # ========================
        # 关键改动：增加用户级 token 计数
        # ========================
        if user_id and token_count > 0:
            self.user_token_used[user_id] = self.user_token_used.get(user_id, 0) + token_count

        return False, "passed"

    # -------------------------------------------------------------------------
    # 清理过期数据
    # -------------------------------------------------------------------------
    def clean_expired_qps(self):
        now = int(time.time())
        # 清理全局QPS
        self.global_qps_tracker = {
            t: c for t, c in self.global_qps_tracker.items() if t >= now - 2
        }
        # 清理用户QPS
        for uid in self.user_qps_tracker:
            self.user_qps_tracker[uid] = {
                t: c for t, c in self.user_qps_tracker[uid].items() if t >= now - 2
            }