from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from dinollm.distributed import DistributedInfo
from dinollm.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:

    import torch
    from dinollm.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)

        scheduler.sync_all_ranks()


        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

 
        if args.silent_output:
            logging.disable(logging.INFO)

        try:

            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # 换行，让^C显示更干净
                logger.info("Scheduler exiting gracefully...")
            # 关闭调度器，释放资源
            scheduler.shutdown()


def launch_server() -> None:


    from .api import run_api_server
    from .args import parse_args

    server_args = parse_args(sys.argv[1:])
    logger = init_logger(__name__, "startup")

    def start_subprocess() -> None:

        from dinollm.tokenizer import tokenize_worker

        # CUDA多进程必须使用spawn方式启动
        mp.set_start_method("spawn", force=True)

        # TP并行大小 = 使用的GPU数量
        world_size = server_args.tp_info.size
        # 用于等待子进程启动完成的消息队列
        ack_queue: mp.Queue[str] = mp.Queue()

        # ========== 1. 为每个GPU启动一个Scheduler进程 ==========
        for i in range(world_size):
            # 复制参数并设置当前rank信息
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            # 创建并启动调度器进程
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"dinollm-TP{i}-scheduler",
            ).start()

        # 启动的分词器进程数量
        num_tokenizers = server_args.num_tokenizer

        # ========== 2. 启动1个 DeTokenizer 进程（token → 文本） ==========
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="dinollm-detokenizer-0",
        ).start()

        # ========== 3. 启动 N 个 Tokenizer 进程（文本 → token） ==========
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"dinollm-tokenizer-{i}",
            ).start()


        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    run_api_server(server_args, start_subprocess)


if __name__ == "__main__":
    launch_server()
