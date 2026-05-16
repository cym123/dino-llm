from __future__ import annotations
# 允许类注解前向引用，不用写字符串引号，Python3.7+语法兼容

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias
# 导入类型注解工具：列表、命名元组、无返回、集合、元组、类型别名

import torch
# 导入PyTorch张量、CUDA流、分布式、推理模式等核心能力

# 底层核心基础类
from dinollm.core import Batch, Req
# 全局环境变量
from dinollm.core import ENV
# 所有前后端通信消息结构体
from dinollm.message import (
    AbortBackendMsg,        # 取消请求消息
    BaseBackendMsg,         # 后端消息基类
    BatchBackendMsg,        # 批量消息包裹
    DetokenizeMsg,          # 解码返回消息
    ExitMsg,                # 退出服务消息
    UserMsg,                # 用户请求消息
)
# 日志工具、加载分词器工具
from dinollm.utils import init_logger, load_tokenizer

# 同目录下自定义模块导入
from .cache import CacheManager       # KV缓存+前缀缓存管理器
from .config import SchedulerConfig   # 调度器配置
from .decode import DecodeManager     # Decode解码阶段请求管理器
from .io import SchedulerIOMixin      # 调度器网络通信混入类
from .prefill import ChunkedReq,PrefillManager   # Prefill预填充请求管理器
from .table import TableManager       # 页表槽位管理器

# 仅类型检查时导入，运行时不加载，避免循环导入
if TYPE_CHECKING:
    from dinollm.engine import BatchSamplingArgs, ForwardOutput
    # 批次采样参数、模型前向输出结构

# 初始化日志器，绑定当前模块名
logger = init_logger(__name__)

# 定义类型别名：二元组(张量, 张量)，统一指代页表行列索引
Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# 命名元组：封装一次前向推理需要的**所有输入数据**
# 把批次、采样参数、输入索引、写入索引打包成一个结构体
class ForwardInput(NamedTuple):
    batch: Batch               # 当前要推理的请求批次
    sample_args: BatchSamplingArgs  # 批次采样配置（温度、topk等）
    input_tuple: Indice2D      # 输入寻址二元组：(请求行号张量, 位置张量)
    write_tuple: Indice2D      # 输出写入寻址二元组


# 类型别名：一次推理完整数据 = 输入结构体 + 模型前向输出结果
ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


# 调度器主类，继承网络通信混入类，直接拥有收发消息能力
class Scheduler(SchedulerIOMixin):
    # 构造函数：初始化整个调度器所有组件、流、管理器、通信
    def __init__(self, config: SchedulerConfig):
        # 延迟导入引擎，避免顶层循环导入
        from dinollm.engine import Engine

        # 初始化底层推理引擎：加载模型、初始化KV页表、分布式组、注意力后端
        self.engine = Engine(config)

        # 获取设备类型（cuda）
        self.device = self.engine.device
        # 创建独立CUDA流：调度逻辑跑在独立流，和模型推理流并行重叠
        self.stream = torch.cuda.Stream(device=self.device)
        # 获取引擎专属CUDA流上下文管理器
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        # 将当前代码执行流切到调度专属流
        torch.cuda.set_stream(self.stream)

        # 初始化页表管理器：管理并发请求在page_table中的行槽位分配/回收
        # 主要记录token的在kvcache中的位置
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)

        # 初始化缓存管理器：管理KV显存页分配、释放、Radix前缀缓存匹配/淘汰
        self.cache_manager = CacheManager(
            self.engine.num_pages,    # 总页数
            config.page_size,         # 每页token数
            self.engine.page_table,   # 全局页表
            config.cache_type         # 缓存类型radix/naive
        )

        # 初始化解码管理器：管理所有进入decode生成阶段的请求
        self.decode_manager = DecodeManager(config.page_size)

        # 初始化预填充管理器：管理等待prefill的请求队列、分块prefill、批次组装
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager
        )

        # 存放已经跑完结束的请求集合，用于去重防重复释放资源
        self.finished_reqs: Set[Req] = set()
        # 加载分词器，用于获取eos_id、文本编解码
        self.tokenizer = load_tokenizer(config.model_path)
        # 结束符token id，判断生成是否终止
        self.eos_token_id = self.tokenizer.eos_token_id
        # 引用全局token池：存储所有请求的input_ids，和page_table同shape
        self.token_pool = self.table_manager.token_pool
        # 单次prefill最大token预算，控制批次大小
        self.prefill_budget = config.max_extend_tokens

        # 调用父类混入类构造，初始化Zmq通信、多卡广播、收发函数绑定
        super().__init__(config, self.engine.tp_cpu_group)

    # 调度器空闲时回调：打印空闲日志、校验缓存完整性防内存泄漏
    def run_when_idle(self) -> None:
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    # 重叠调度主循环：CPU调度和GPU推理双流重叠，隐藏CPU开销、拉高GPU利用率
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        # 判定是否需要阻塞等待新消息
        # 有上批数据待处理 / 有prefill可跑 / 有decode可跑 → 不阻塞
        blocking = not (
            last_data is not None
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )

        # 接收消息并逐个处理
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 调度生成下一个可执行批次（prefill优先）
        forward_input = self._schedule_next_batch()
        ongoing_data = None

        # 如果凑到可执行批次，切到引擎流执行前向推理
        if forward_input is not None:
            with self.engine_stream_ctx:
                # 引擎流等待调度流做完准备，保证依赖有序
                self.engine.stream.wait_stream(self.stream)
                # 封装 当前输入 + 前向推理结果
                ongoing_data = (forward_input, self._forward(forward_input))

        # 处理上一轮推理结果：解析token、返回前端、释放资源、缓存写入
        self._process_last_data(last_data)
        # 返回当前批次数据，下一轮循环当做last_data处理
        return ongoing_data

    # 普通非重叠主循环：串行 收消息→调度→推理→处理结果，无双流并行
    def normal_loop(self) -> None:
        # 无待跑请求才阻塞等消息
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        # 接收并处理消息
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 调度下一批
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        # 执行推理
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))
        # 处理结果
        self._process_last_data(ongoing_data)

    # 永久运行入口：推理模式下死循环跑调度
    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # 环境变量关闭重叠调度 → 走普通循环
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        # 开启重叠调度 → 死循环跑overlap_loop
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    # 关闭调度器：同步CUDA、多卡屏障、关闭引擎资源
    def shutdown(self) -> None:
        # 等待所有CUDA操作完成
        torch.cuda.synchronize(self.device)
        # 多卡CPU端同步屏障
        self.sync_all_ranks()
        # 关闭引擎、释放模型权重、缓存资源
        self.engine.shutdown()

    # 处理上一轮推理输出结果：解析token、构造返回消息、资源回收、缓存落盘
    def _process_last_data(self, last_data: ForwardData | None) -> None:
        # 无历史数据直接返回
        if last_data is None:
            return

        # 解包：批次 + 前向输出(新token、拷贝完成事件)
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        # 等待GPU->CPU拷贝完成
        copy_done.synchronize()

        # 存放要发给分词器的解码消息
        reply: List[DetokenizeMsg] = []
        # 本轮新结束的请求集合
        new_finished_reqs: Set[Req] = set()

        # 延迟释放上下文：批量回收页，减少频繁显存操作
        with self.cache_manager.lazy_free_region():
            # 遍历批次里每个请求
            for i, req in enumerate(batch.reqs):
                # 分块预填充请求，不参与解码生成，跳过
                if isinstance(req, ChunkedReq):
                    continue

                # 取出当前生成的单个token
                next_token = next_tokens_cpu[i]
                # 把新token追加到请求的host侧序列
                req.append_host(next_token.unsqueeze(0))
                # 转成整型token id
                next_token = int(next_token.item())

                # 判定请求是否可继续解码，不可则标记结束
                finished = not req.can_decode
                # 不忽略eos且遇到结束符，也标记结束
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id

                # 构造返回给分词器的消息：uid、新token、是否结束
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # 请求结束且未标记过完成
                if finished and req not in self.finished_reqs:
                    # 从解码管理器移除
                    self.decode_manager.remove_req(req)
                    # 释放请求占用的页表槽位、KV缓存
                    self._free_req_resources(req)
                    # 加入本轮完成集合
                    new_finished_reqs.add(req)
                # 若是prefill阶段且未结束，把前缀写入Radix缓存做复用
                elif batch.is_prefill:
                    self.cache_manager.cache_req(req, finished=False)

        # 更新已完成请求集合
        self.finished_reqs = new_finished_reqs
        # 把批量解码消息发往分词器进程
        self.send_result(reply)

    # 单条消息处理入口：分发不同消息类型逻辑
    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        # 批量包裹消息：解包逐个递归处理
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        # 退出消息：抛键盘中断终止服务
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        # 用户新请求消息
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            # 获取输入长度、模型最大支持序列长度
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            # 可生成的最大token数 = 总长 - 输入长
            max_output_len = max_seq_len - input_len
            # 输入超长，直接丢弃请求
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            # 用户设置的max_tokens超过可生成上限，强制截断
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # 把用户请求加入prefill等待队列
            self.prefill_manager.add_one_req(msg)
        # 取消请求消息
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            # 先从prefill队列找，再从decode队列找
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            # 找到则释放资源
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        # 未知消息类型，直接抛异常
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    # 释放请求全部资源：页表槽位回收、KV缓存标记释放
    def _free_req_resources(self, req: Req) -> None:
        # 归还table_manager的行槽位
        self.table_manager.free(req.table_idx)
        # 缓存管理器标记请求结束，回收KV页、清理前缀缓存引用
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
