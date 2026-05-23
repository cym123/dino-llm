from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch

from dinollm.core import Batch, Req
from dinollm.core import ENV
from dinollm.message import (
    AbortBackendMsg,        # 取消请求消息
    BaseBackendMsg,         # 后端消息基类
    BatchBackendMsg,        # 批量消息包裹
    DetokenizeMsg,          # 解码返回消息
    ExitMsg,                # 退出服务消息
    UserMsg,                # 用户请求消息
)
from dinollm.utils import init_logger, load_tokenizer

from .cache import CacheManager       # KV缓存+前缀缓存管理器
from .config import SchedulerConfig   # 调度器配置
from .decode import DecodeManager     # Decode解码阶段请求管理器
from .io import SchedulerIOMixin      # 调度器网络通信混入类
from .prefill import ChunkedReq,PrefillManager   # Prefill预填充请求管理器
from .table import TableManager       # 页表槽位管理器

if TYPE_CHECKING:
    from dinollm.engine import BatchSamplingArgs, ForwardOutput
    
# 自动生成的 gRPC 文件
# gRPC 自动生成
from dinollm.proto import worker_metrics_pb2
from dinollm.proto import worker_metrics_pb2_grpc

# 你自己的 dataclass
from dinollm.cluster.load_predictor import WorkerMetrics

# ============================
# gRPC 上报客户端（集成到 Scheduler）
# ============================
import grpc
import asyncio
import threading

logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]

# Router gRPC 地址
ROUTER_GRPC_ADDR = "127.0.0.1:50051"

# =============================================================================
# 🔥 终极通用版：传入 WorkerMetrics 对象，无任何硬编码
# =============================================================================
# 后台上报协程
async def _report_worker_metrics_async(worker_id: str, get_metrics_func):
    async with grpc.aio.insecure_channel(ROUTER_GRPC_ADDR) as channel:
        stub = worker_metrics_pb2_grpc.WorkerMetricServiceStub(channel)
        logger.info(f"✅ gRPC 上报客户端已启动，目标 Router: {ROUTER_GRPC_ADDR}")

        while True:
            try:
                # 获取当前实时指标
                m = get_metrics_func()
                req = worker_metrics_pb2.WorkerMetricsRequest(
                    worker_id=worker_id,
                    gpu_flops_total=m.gpu_flops_total,
                    kv_cache_free_gb=m.kv_cache_free_gb,
                    prefill_queue_len=m.prefill_queue_len,
                    decode_queue_len=m.decode_queue_len,
                    wait_prefill_queue_len=m.wait_prefill_queue_len,
                    ttft_ms=m.ttft_ms,
                    tpot_ms=m.tpot_ms,
                )
                logger.debug(f"📤 正在上报指标 | worker={worker_id}")
                res = await stub.ReportMetrics(req)
                logger.info(f"📥 上报成功 | worker={worker_id}, code={res.code}, msg={res.msg}")
            except Exception as e:
                pass
                # logger.error(f"❌ 上报失败 | worker={worker_id}")
                # logger.error(f"❌ 上报失败 | worker={worker_id}, 错误={str(e)}")
            await asyncio.sleep(5)

# 启动后台上报线程（不阻塞推理）
def start_worker_metrics_report(worker_id: str, get_metrics_func):
    def run():
        asyncio.run(_report_worker_metrics_async(worker_id, get_metrics_func))
    th = threading.Thread(target=run, daemon=True)
    th.start()
    logger.info(f"🚀 后台指标上报线程已启动 | worker={worker_id}")



class ForwardInput(NamedTuple):
    batch: Batch               # 当前要推理的请求批次
    sample_args: BatchSamplingArgs  # 批次采样配置（温度、topk等）
    input_tuple: Indice2D      # 输入寻址二元组：(请求行号张量, 位置张量)
    write_tuple: Indice2D      # 输出写入寻址二元组


# 类型别名：一次推理完整数据 = 输入结构体 + 模型前向输出结果
ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


# 调度器主类，继承网络通信混入类，直接拥有收发消息能力
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from dinollm.engine import Engine

        self.engine = Engine(config)

        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)


        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)

        self.cache_manager = CacheManager(
            self.engine.cache_per_page,
            self.engine.num_pages,    # 总页数
            config.page_size,         # 每页token数
            self.engine.page_table,   # 全局页表
            config.cache_type         # 缓存类型radix/naive
        )

        self.decode_manager = DecodeManager(config.page_size)

        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager
        )

        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.schedule_strategy = config.schedule_strategy  # fcfs / priority
        
        self.worker_id = f"{config.server_host}:{config.server_port}" 
        def get_metrics():
            
            
            return WorkerMetrics(
                worker_id=self.worker_id,
                gpu_flops_total=180.0,                          # 你可以从engine拿
                kv_cache_free_gb=self.cache_manager.free_pages_gb,  # 从cache拿
                prefill_queue_len=0,
                decode_queue_len=len(self.decode_manager.running_reqs),
                wait_prefill_queue_len=len(self.prefill_manager.pending_list),
                ttft_ms=200,
                tpot_ms=50,
            )
        start_worker_metrics_report(self.worker_id, get_metrics)

        

        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        # 判定是否需要阻塞等待新消息
        # 有上批数据待处理 / 有prefill可跑 / 有decode可跑 → 不阻塞
        blocking = not (
            last_data is not None
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )

        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None

        if forward_input is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    # 普通非重叠主循环：串行 收消息→调度→推理→处理结果，无双流并行
    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))
        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()

        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()

        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue

                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())

                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id

                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)


    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    # 批次预处理：补全批次、分配KV页、构造位置索引、准备注意力元数据
    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # 对批次做padding补齐长度
        self.engine.graph_runner.pad_batch(batch)
        # 为批次请求分配未占用的KV显存页
        self.cache_manager.allocate_paged(batch.reqs)
        # 构造每个token的position位置编码
        batch.positions = _make_positions(batch, self.device)
        # 构造输入寻址二元组
        input_mapping = _make_input_tuple(batch, self.device)
        # 构造输出写入寻址二元组
        write_mapping = _make_write_tuple(batch, self.device)
        # 填入批次对应的页表物理位置
        batch.out_loc = self.engine.page_table[input_mapping]
        # 注意力后端准备批次元数据（PagedAttention所需）
        self.engine.attn_backend.prepare_metadata(batch)
        # 封装成ForwardInput返回
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    # 调度决策：优先调度prefill批次，没有再调度decode批次
    def _schedule_next_batch(self) -> ForwardInput | None:
        
        pending_queue = self.prefill_manager.pending_list
        # 根据策略重排等待队列
        if pending_queue:
            if self.schedule_strategy == "priority":
                # 排序权重：有分块请求 > 优先级 > 到达时间
                pending_queue.sort(
                key=lambda x: (0 if x.chunked_req is not None else 1, -x.priority, x.arrive_time)
                )
        # fcfs无需改动，原生入队顺序
        
        # 先尝试拿prefill批次，拿不到再拿decode批次
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        # 有批次就预处理封装，无则返回None
        return self._prepare_batch(batch) if batch else None

    # 执行模型前向推理：填token池、调用引擎前向、回填新生成token、过滤可解码请求
    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        # 从全局token_pool取出当前批次input_ids送入模型
        batch.input_ids = self.token_pool[input_mapping]
        # 引擎执行前向推理+采样，得到新token
        forward_output = self.engine.forward_batch(batch, sample_args)
        # 把GPU侧新生成token写回token_pool
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # 过滤更新可进入decode的请求集合
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


# 工具函数：构造批次所有token的position id张量
def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    # 计算整个批次需要的总token数量
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    # 页锁内存申请CPU张量，方便异步拷贝
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    # 逐个请求生成position：从已缓存长度到当前长度连续递增
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    # 异步拷贝到GPU返回
    return indices_host.to(device, non_blocking=True)

# 工具函数：构造输入用的(请求行号, 位置索引)二元组
def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    # 每个请求所有token都填充当前请求的table_idx行号
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    # 行号张量 + 位置张量 组成二元组返回
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)

# 工具函数：构造输出写入用的(请求行号, 写入位置)二元组
def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # 收集每个请求的table_idx
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    # 可解码取device_len，不可解码填-1标记
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    # 异步拷到GPU返回二元组
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
