from __future__ import annotations
# 解决导入报错（必须放最顶部）
import sys
import os

import grpc
import asyncio

# 导入 gRPC 自动生成的文件
from dinollm.proto import worker_metrics_pb2
from dinollm.proto import worker_metrics_pb2_grpc

from dinollm.cluster.load_predictor import WorkerMetrics

# ==========================
# gRPC 服务：接收 worker 上报
# ==========================
class RouterGrpcService(worker_metrics_pb2_grpc.WorkerMetricServiceServicer):
    def __init__(self, router_state):
        self.state = router_state

    async def ReportMetrics(self, request, context):
        worker_id = request.worker_id
        
        print("📥 gRPC 收到上报：")
        print(f"  worker_id               : {worker_id}")
        print(f"  gpu_flops_total         : {request.gpu_flops_total}")
        print(f"  kv_cache_free_gb        : {request.kv_cache_free_gb:.2f} GB")
        print(f"  prefill_queue_len       : {request.prefill_queue_len}")
        print(f"  decode_queue_len        : {request.decode_queue_len}")
        print(f"  wait_prefill_queue_len  : {request.wait_prefill_queue_len}")
        print(f"  ttft_ms                 : {request.ttft_ms:.1f} ms")
        print(f"  tpot_ms                 : {request.tpot_ms:.2f} ms")


        # 把数据存到你的 state 里
        if worker_id in self.state.all_workers:
            self.state.worker_realtime_metrics[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                gpu_flops_total=request.gpu_flops_total,
                kv_cache_free_gb=request.kv_cache_free_gb,
                prefill_queue_len=request.prefill_queue_len,
                decode_queue_len=request.decode_queue_len,
                wait_prefill_queue_len=request.wait_prefill_queue_len,
                ttft_ms=request.ttft_ms,
                tpot_ms=request.tpot_ms
            )

        return worker_metrics_pb2.HealthResponse(code=200, msg="ok")

# ==========================
# 启动 gRPC 服务
# ==========================
async def start_grpc_server(router_state):
    try:
        server = grpc.aio.server()
        servicer = RouterGrpcService(router_state)
        worker_metrics_pb2_grpc.add_WorkerMetricServiceServicer_to_server(servicer, server)

        server.add_insecure_port("0.0.0.0:50051")
        
        print("gRPC 开始服务启动 0.0.0.0:50051")
        
        await server.start()
        
        # ==========================================
        # 你要的打印 一定在这里执行！
        # ==========================================
        print("gRPC 成功服务启动 0.0.0.0:50051")

        # 不阻塞启动，只后台等待
        await server.wait_for_termination()
    except Exception as e:
        print("❌ gRPC 启动失败:", e)