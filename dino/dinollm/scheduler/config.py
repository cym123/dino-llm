from __future__ import annotations

from dataclasses import dataclass, field

# 继承底层 Engine 的配置（模型、设备、精度、并行、内存等基础配置）
from dinollm.engine import EngineConfig


# 工具函数：生成一个 进程PID 后缀
# 作用：让每个进程的通信地址唯一，不冲突
def _get_pid_suffix() -> str:
    import os
    # 返回类似 ".pid=12345" 的字符串
    return f".pid={os.getpid()}"


# ======================================================
# Scheduler 配置类（调度器核心配置）
# 继承 EngineConfig → 拥有模型加载、设备、并行等全部基础配置
# ======================================================
@dataclass(frozen=True)  # frozen = 不可修改，保证配置安全
class SchedulerConfig(EngineConfig):
    # 1. 每次推理最大能“扩展/生成”多少个 token
    max_extend_tokens: int = 8192

    # 2. KV 缓存前缀复用类型
    # "radix" = 基数树前缀缓存（SGLang 核心，速度最快）
    # "naive" = 简单缓存
    cache_type: str = "radix"

    # 3. 是否离线模式
    # True = 直接本地运行，不启动网络服务
    # False = 启动 ZeroMQ 网络服务，支持外部请求调用
    offline_mode: bool = False

    # 4. 唯一后缀：自动用进程PID生成，防止多进程冲突
    # field(default_factory=...) → 动态创建值
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    # --------------------------------------------------
    # ZeroMQ 网络通信地址（进程间 / 网络通信）
    # --------------------------------------------------

    # 后端服务地址（接收用户请求）
    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/dinollm_0" + self._unique_suffix

    # 解码服务地址（token → text）
    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/dinollm_1" + self._unique_suffix

    # 调度器广播地址
    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/dinollm_2" + self._unique_suffix

    # --------------------------------------------------
    # 派生配置（直接用已有配置计算，不用手动传）
    # --------------------------------------------------

    # 最大前向推理长度 = 最大扩展token数
    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    # 是否让后端创建解码连接
    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
