from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from dinollm.distributed import DistributedInfo
from dinollm.scheduler import SchedulerConfig
from dinollm.utils import init_logger


# ====================== 服务配置类 ======================
@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    """
    服务启动的全部配置
    继承自 SchedulerConfig（调度器配置）
    增加：API服务、Tokenizer、ZMQ通信地址
    """
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    # 分词器进程数量
    num_tokenizer: int = 0
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        """是否共享 Tokenizer/DeTokenizer(同一进程)"""
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        """前端 ↔ 分词器 通信地址"""
        return "ipc:///tmp/dinollm_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        """Tokenizer 通信地址"""
        # 共享模式 → 和 detokenizer 用同一个地址
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/dinollm_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        """是否由 tokenizer 创建地址（而非后端）"""
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """是否由后端创建 detokenizer 连接"""
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        """是否由前端创建 tokenizer 连接"""
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        """多卡分布式通信地址"""
        return f"tcp://127.0.0.1:{self.server_port + 1}"


# ====================== 命令行参数解析 ======================
def parse_args(args: List[str]) -> ServerArgs:
    """
    解析命令行参数 → 构造 ServerArgs
    返回：(配置对象, 是否运行shell模式)
    """
    from dinollm.attention import validate_attn_backend
    from dinollm.kvcache import SUPPORTED_CACHE_MANAGER
    from dinollm.moe import SUPPORTED_MOE_BACKENDS

    # 定义参数解析器
    parser = argparse.ArgumentParser(description="dinollm Server Arguments")

    # 模型路径
    parser.add_argument(
        "--model-path", "--model",
        type=str, required=True,
        help="模型路径",
    )

    # 数据类型
    parser.add_argument(
        "--dtype",
        type=str, default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型权重数据类型",
    )

    # 张量并行大小（GPU数量）
    parser.add_argument(
        "--tensor-parallel-size", "--tp-size",
        type=int, default=1,
        help="GPU 张量并行数量",
    )

    # 最大同时运行请求数
    parser.add_argument(
        "--max-running-requests",
        type=int, dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="最大同时运行请求数",
    )

    # 强制覆盖最大序列长度
    parser.add_argument(
        "--max-seq-len-override",
        type=int, default=ServerArgs.max_seq_len_override,
        help="强制覆盖模型最大序列长度",
    )

    # KV缓存使用的显存比例
    parser.add_argument(
        "--memory-ratio",
        type=float, default=ServerArgs.memory_ratio,
        help="KV缓存使用的GPU显存比例",
    )

    # 是否使用假权重（测试用）
    parser.add_argument(
        "--dummy-weight",
        action="store_true", dest="use_dummy_weight",
        help="使用假权重，仅用于测试",
    )

    # 是否禁用 PyNCCL
    parser.add_argument(
        "--disable-pynccl",
        action="store_false", dest="use_pynccl",
        help="禁用PyNCCL",
    )

    # API 服务监听HOST
    parser.add_argument(
        "--host",
        type=str, dest="server_host",
        default=ServerArgs.server_host,
        help="API服务监听地址",
    )

    # API 服务端口
    parser.add_argument(
        "--port",
        type=int, dest="server_port",
        default=ServerArgs.server_port,
        help="API服务端口",
    )

    # CUDA Graph 最大batch
    parser.add_argument(
        "--cuda-graph-max-bs", "--graph",
        type=int, default=ServerArgs.cuda_graph_max_bs,
        help="CUDA Graph 最大批处理大小",
    )

    # 分词器进程数
    parser.add_argument(
        "--num-tokenizer", "--tokenizer-count",
        type=int, default=ServerArgs.num_tokenizer,
        help="分词器进程数，0表示共享进程",
    )

    # 预填充块大小（分块prefill）
    parser.add_argument(
        "--max-prefill-length", "--max-extend-length",
        type=int, dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="分块预填充的块大小",
    )

    # KV缓存页数
    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int, default=ServerArgs.num_page_override,
        help="KV缓存页数",
    )

    # 分页大小
    parser.add_argument(
        "--page-size",
        type=int, default=ServerArgs.page_size,
        help="分页大小",
    )

    # 注意力后端
    parser.add_argument(
        "--attention-backend", "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="注意力后端",
    )

    # 模型来源：huggingface / modelscope
    parser.add_argument(
        "--model-source",
        type=str, default="modelscope",
        choices=["huggingface", "modelscope"],
        help="模型来源",
    )

    # 缓存类型
    parser.add_argument(
        "--cache-type",
        type=str, default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="KV缓存管理策略",
    )

    # MoE 后端
    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="MoE 后端",
    )



    # ====================== 解析参数 ======================
    kwargs = parser.parse_args(args).__dict__.copy()



    # 展开 ~ 路径
    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    # 如果是 modelscope 模型，自动下载
    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download
            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # 自动推断 dtype
    if (dtype_str := kwargs["dtype"]) == "auto":
        from dinollm.utils import cached_load_hf_config
        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    # 字符串 → torch.dtype
    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str

    # 构造分布式信息
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    # 构造最终配置
    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result