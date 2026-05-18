from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict
from .load_predictor import WorkerMetrics, WorkerLoadPredictor

from dinollm.utils import init_logger



logger = init_logger(__name__)


class RoutePolicy(Enum):
    """路由调度策略（顺序已调整）"""
    USER_STICKY = 0     # 用户粘性会话调度（第0位）
    LOAD_BEST = 1       # 负载最优调度（第1位）
    ROUND_ROBIN = 2     # 平滑轮询（第2位）


class RouteStrategyManager:
    """统一路由策略管理器（精简生产版）"""
    def __init__(self, load_predictor: WorkerLoadPredictor):
        self._predictor = load_predictor

        # 用户会话粘性映射：user_id → worker_id
        self.user_sticky_map: Dict[str, str] = {}

        # 轮询游标（保证平滑均匀）
        self._round_robin_cursor = 0

    def bind_user_worker(self, user_id: str, worker_id: str) -> None:
        """绑定用户到节点，保持会话 KV 缓存复用"""
        self.user_sticky_map[user_id] = worker_id

    def unbind_user_worker(self, user_id: str) -> None:
        """解除用户绑定"""
        self.user_sticky_map.pop(user_id, None)

    def get_sticky_bind_worker(self, user_id: str) -> Optional[str]:
        """获取用户已绑定节点"""
        return self.user_sticky_map.get(user_id)

    def select_worker(
        self,
        policy: RoutePolicy,
        worker_metrics_list: List[WorkerMetrics],
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        统一调度入口
        :param policy: 0/1/2 三选一
        :param worker_metrics_list: 健康节点列表
        :param user_id: 粘性路由必传
        :return: 最优节点ID
        """
        if not worker_metrics_list:
            return None

        # 可用节点 ID 列表
        worker_ids = [m.worker_id for m in worker_metrics_list]
        available = {m.worker_id: m for m in worker_metrics_list}

        # --------------------------
        # 策略 0：用户粘性会话（默认第一位）
        # --------------------------
        if policy == RoutePolicy.USER_STICKY:
            if user_id:
                bound_wid = self.get_sticky_bind_worker(user_id)
                if bound_wid and bound_wid in available:
                    logger.info(f"用户 {user_id} 粘性绑定节点 {bound_wid} 可用，继续使用")
                    return bound_wid

            # 无绑定/节点不可用 → 自动降级负载最优
            return self._predictor.pick_best_worker(worker_metrics_list)

        # --------------------------
        # 策略 1：负载最优
        # --------------------------
        elif policy == RoutePolicy.LOAD_BEST:
            return self._predictor.pick_best_worker(worker_metrics_list)

        # --------------------------
        # 策略 2：平滑轮询（均匀分发）
        # --------------------------
        elif policy == RoutePolicy.ROUND_ROBIN:
            idx = self._round_robin_cursor % len(worker_ids)
            selected = worker_ids[idx]
            self._round_robin_cursor += 1
            return selected

        # --------------------------
        # 兜底：负载最优
        # --------------------------
        return self._predictor.pick_best_worker(worker_metrics_list)
    

