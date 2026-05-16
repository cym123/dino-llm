# 让Python支持未来的语法特性
from __future__ import annotations

# 类型检查相关：只给编辑器看
from typing import TYPE_CHECKING, Protocol

# 注册表工具：用来管理各种缓存类
from dinollm.utils import Registry

# 仅在类型检查时导入（运行时不导入，防止循环引用）
if TYPE_CHECKING:
    import torch
    from dinollm.models import ModelConfig


# 导入缓存相关的**基类（基础模板）**
# 基类 = 规定所有缓存必须实现什么功能，不能乱实现
from .base import (
    BaseCacheHandle,    # 缓存句柄（代表一个缓存块）
    BaseKVCachePool,    # KV缓存池（管理所有显存块）
    BasePrefixCache,    # 前缀缓存基类（核心缓存）
    MatchResult,        # 缓存匹配结果
    SizeInfo,           # 大小信息
)


# --------------------------
# Protocol = 定义“函数长什么样”
# 这里规定：创建缓存的函数必须接收 device，返回 BasePrefixCache
# --------------------------
class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


# --------------------------
# 注册表：存放所有支持的缓存类型
# 比如 naive、radix
# --------------------------
SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


# --------------------------
# 【函数1】创建 KV 缓存池（管理显存）
# 作用：给模型创建一大块显存，用来存 K、V
# --------------------------
def create_kvcache_pool(
    model_config: ModelConfig,  # 模型配置（多少层、多少头）
    num_pages: int,             # 要创建多少个页（显存块）
    page_size: int,             # 每页多大
    dtype: torch.dtype,         # 精度 fp16/bf16
    device: torch.device,       # 设备 GPU
) -> BaseKVCachePool:

    # 目前只支持 MHA 普通注意力缓存
    from .mha_pool import MHAKVCache

    # 创建并返回 KV 缓存池
    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


# --------------------------
# 注册第一种缓存：naive（简单版缓存）
# --------------------------
@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


# --------------------------
# 注册第二种缓存：radix（高性能前缀缓存）
# --------------------------
@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


# --------------------------
# 【函数2】创建前缀缓存（根据名字选择类型）
# 你传入 "naive" 或 "radix"
# 它就返回对应的缓存对象
# --------------------------
def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


# --------------------------
# 导出给外部使用的函数
# --------------------------
__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
