from __future__ import annotations
from dataclasses import dataclass
from typing import List

from dinollm.utils import init_logger



logger = init_logger(__name__)


@dataclass
class WorkerMetrics:
    """LLM推理节点7维核心指标"""
    worker_id: str                  # 节点唯一标识
    gpu_flops_total: float          # GPU峰值算力 TFLOPS
    kv_cache_free_gb: float         # 空闲KV缓存显存 GB
    prefill_queue_len: int          # 正在执行prefill队列数
    decode_queue_len: int           # 正在执行decode队列数
    wait_prefill_queue_len: int     # 排队等待prefill任务数
    ttft_ms: float                  # 首token耗时 ms
    tpot_ms: float                  # 单token生成耗时 ms


class WorkerLoadPredictor:
    """7维纯GPU推理负载调度打分器
    分数越低 = 节点越空闲 = 优先调度
    """
    def __init__(self):
        # 7维新权重 总和=1.0
        self.W_PREFILL: float = 0.15
        self.W_DECODE: float = 0.16
        self.W_WAIT_PREFILL: float = 0.10
        self.W_TTFT: float = 0.12
        self.W_TPOT: float = 0.07
        self.W_GPU_POWER: float = 0.18
        self.W_KV_FREE: float = 0.22

        # 归一化上限阈值
        self.MAX_QUEUE: int = 256
        self.MAX_TTFT: float = 4000.0
        self.MAX_TPOT: float = 200.0
        self.MAX_GPU_FLOPS: float = 200.0
        self.MAX_KV_FREE_GB: float = 40.0

    def _norm_forward(self, val: float, max_val: float) -> float:
        """正向归一：数值越大分数越高（队列、延迟）"""
        return min(val / max_val, 1.0)

    def _norm_reverse(self, val: float, max_val: float) -> float:
        """反向归一：数值越大分数越低（算力、空闲显存）"""
        return 1.0 - min(val / max_val, 1.0)

    def calc_node_score(self, metrics: WorkerMetrics) -> float:
        """计算单节点0~1综合负载分数"""
        # 正向劣化指标
        n_prefill = self._norm_forward(metrics.prefill_queue_len, self.MAX_QUEUE)
        n_decode = self._norm_forward(metrics.decode_queue_len, self.MAX_QUEUE)
        n_wait_pre = self._norm_forward(metrics.wait_prefill_queue_len, self.MAX_QUEUE)
        n_ttft = self._norm_forward(metrics.ttft_ms, self.MAX_TTFT)
        n_tpot = self._norm_forward(metrics.tpot_ms, self.MAX_TPOT)

        # 反向优质指标
        n_gpu = self._norm_reverse(metrics.gpu_flops_total, self.MAX_GPU_FLOPS)
        n_kv_free = self._norm_reverse(metrics.kv_cache_free_gb, self.MAX_KV_FREE_GB)

        # 加权求和
        total_score = (
            n_prefill * self.W_PREFILL
            + n_decode * self.W_DECODE
            + n_wait_pre * self.W_WAIT_PREFILL
            + n_ttft * self.W_TTFT
            + n_tpot * self.W_TPOT
            + n_gpu * self.W_GPU_POWER
            + n_kv_free * self.W_KV_FREE
        )

        return max(round(total_score, 6), 0.0)

    def pick_best_worker(self, worker_list: List[WorkerMetrics]) -> str:
        """选出最优推理节点ID"""
        if not worker_list:
            raise RuntimeError("无可用推理工作节点")
        worker_list.sort(key=self.calc_node_score)
        

        logger.info(f"选定最优节点: {worker_list[0].worker_id} | score={self.calc_node_score(worker_list[0])}")
        return worker_list[0].worker_id

    def get_all_score_map(self, worker_list: List[WorkerMetrics]) -> dict[str, float]:
        """获取所有节点分数，用于日志监控"""
        return {w.worker_id: self.calc_node_score(w) for w in worker_list}
    

