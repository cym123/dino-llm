from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import os


from dinollm.core import SamplingParams

from dinollm.core import ENV

from dinollm.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg, 
    TokenizeMsg,
    UserReply,
)

from dinollm.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger


from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter


from pydantic import BaseModel, Field
from starlette.background import BackgroundTask


from .args import ServerArgs


logger = init_logger(__name__, "FrontendAPI")


_GLOBAL_STATE = None



def get_global_state() -> FrontendManager:
    
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE



def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:

    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]



class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = True
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "dina-llm"
    root: str


class ModelList(BaseModel):
    """模型列表返回"""
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


# ====================== 前端管理器（核心） ======================

@dataclass
class FrontendManager:

    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]   # 发送 → 分词器
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]    # 接收 ← 分词器
    uid_counter: int = 0                                  # 用户 UID 自增
    initialized: bool = False                             # 是否已启动监听
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)  # 结果缓存 key 是用户ID
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)  # 完成通知

    # 来了一个新用户
    def new_user(self) -> int:
        """分配新用户 UID"""
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    # 后台监听 拉取分词器返回的数据 填入 对应的字典ack_map event_map
    async def listen(self):
        """后台监听分词器返回的结果"""
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                # 把当前用户的结果放入对应 UID 队列
                self.ack_map[msg.uid].append(msg)
                # 并且通知：有新数据
                self.event_map[msg.uid].set()

    #启动监听你 只需要启动一次 根据 initialized判断
    def _create_listener_once(self):
        """只启动一次监听协程"""
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    # 判断是否启动监听 没有启动 打开监听 发送数据到分词器 异步
    async def send_one(self, msg: BaseTokenizerMsg):
        """发送一条消息到分词器"""
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    # 为每个用户启动一个监听 判断是否来了数据 来了数据 读取 然后yield 出去
# 【核心函数】根据用户ID，等待模型一段一段生成结果，流式返回
    async def wait_for_ack(self, uid: int):
        """等待并迭代返回结果（流式）"""
        # 拿到这个用户专属的“等待事件”（每个人一个，互不干扰）
        event = self.event_map[uid]

        # 无限循环：一直等，直到模型回答结束
        while True:
            # 等待模型生成内容（不卡死程序）
            await event.wait()
            # 模型生成了一段，清空等待状态，准备等下一段
            event.clear()

            # 取出这个用户刚生成好的所有内容片段
            pending = self.ack_map[uid]
            # 取出后清空缓存，准备接收下一批内容
            self.ack_map[uid] = []
            # 定义变量，用来保存最后一条结果
            ack = None
            
            # 循环把每一段内容返回给调用者
            for ack in pending:
                # 流式返回：来一段，返回一段
                yield ack
            
            # 如果最后一条内容标记为“已结束”
            if ack and ack.finished:
                # 退出循环，不再等待
                break

        # 【对话完成】清理这个用户的所有数据，释放内存
        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        """简单流式生成：返回 data: xxx"""
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        """OpenAI 兼容流式格式"""
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # 发送结束块
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        """包装流式生成，支持客户端断开自动取消请求"""
        try:
            async for chunk in generator:
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        """取消用户请求"""
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        """关闭队列"""
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()



@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()
        
WORKER_ID = os.getenv("WORKER_ID", "worker-0")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8001"))


# 初始化 FastAPI
app = FastAPI(title="dinollm API Server", version="0.0.1", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "worker_id": WORKER_ID, "port": WORKER_PORT}


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """简单生成接口"""
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    """OpenAI 兼容对话接口"""
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
        media_type="text/event-stream",
    )


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])






# ====================== 启动 API 服务 ======================

def run_api_server(config: ServerArgs, start_backend: Callable[[], None]) -> None:
    """
    启动前端 API 服务
    - 初始化 ZMQ
    - 启动后端进程
    - 启动 FastAPI
    """
    global _GLOBAL_STATE



    # http服务的IP
    host = config.server_host
    # http服务的端口
    port = config.server_port

    # 初始化全局前端管理器
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # 启动后端（调度器 + 分词器）
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")

    # 启动 API 或 Shell
    uvicorn.run(app, host=host, port=port)
