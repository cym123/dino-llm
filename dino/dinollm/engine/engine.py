from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from dinollm.attention import create_attention_backend
from dinollm.core import Batch, Context, Req, set_global_ctx
from dinollm.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from dinollm.kvcache import create_kvcache_pool
from dinollm.layers import set_rope_device
from dinollm.models import create_model, load_weight
from dinollm.moe import create_moe_backend
from dinollm.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler


logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:

    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # ======================= 初始化多卡通信 =======================
        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"加载模型前空闲显存: {mem_GB(init_free_memory)}")

        # ======================= 模型初始化 ========================
        # 设置RoPE设备
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV Cache 初始化 =======================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 虚拟页
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page Table 初始化 ========================
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= 注意力 & MoE 后端初始化 ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        # 如果是MoE模型，初始化MoE后端
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= 采样器初始化 ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"初始化后空闲显存: {mem_GB(post_free_memory)}")

        # ======================= CUDA Graph 初始化 ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,
            cache_handle=None,
        )
        # 虚拟请求指向虚拟页
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
        # CUDA Graph 运行器（极致加速推理）
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
   
        if config.tp_info.size == 1 or config.use_pynccl:
            # 初始化CPU通信组（gloo后端）
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            # 计算最大通信字节数
            max_bytes = config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            # 启用PyNCCL高速通信
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            # 传统NCCL通信初始化
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """加载模型权重：支持假权重（测试）和真实权重"""
        if config.use_dummy_weight:
            # 假权重：随机初始化，快速测试
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # 真实权重：从模型路径加载
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
    
        new_free_memory = self._sync_get_memory()[1]
        # 计算每个KV块占用的显存
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        # 手动指定页数优先级最高
        num_pages = config.num_page_override
        if num_pages is None:
            # 模型占用的显存
            model_memory = old_free_memory - new_free_memory
            # 可用于KV Cache的显存
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            # 计算页数
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "显存不足，无法分配KV缓存"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"分配KV缓存token数: {num_tokens}, 大小: {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
      
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        # 多卡同步获取最小/最大空闲显存
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        # 显存不均衡直接报错
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error("多卡显存不均衡！")
            raise RuntimeError("多卡显存不均衡")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
   
        # 必须使用引擎内部的CUDA流
        assert torch.cuda.current_stream() == self.stream
        # 进入批次上下文
        with self.ctx.forward_batch(batch):
            # 优先使用CUDA Graph加速
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                # 普通前向传播
                logits = self.model.forward()

        # 标记请求完成一步生成
        for req in batch.reqs:
            req.complete_one()

        # 采样下一个token
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 异步拷贝到CPU
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        # 记录拷贝完成事件
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self):
        """关闭引擎：销毁CUDA Graph、通信组、释放资源"""
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):

    def override(attr: str, value: Any):
        object.__setattr__(config, attr, value)


    if config.attention_backend == "auto":
        backend = "fa,fi" if is_sm90_supported() else "fi"
        override("attention_backend", backend)
        logger.info_rank0(f"自动选择attention后端: {config.attention_backend}")


    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"自动选择MoE后端: {config.moe_backend}")