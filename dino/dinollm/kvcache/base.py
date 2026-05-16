from __future__ import annotations

# ABC = 抽象基类，用来定规则
# abstractmethod = 必须实现的方法，不写就报错
from abc import ABC, abstractmethod

# 数据类，简化结构体
from dataclasses import dataclass

# 命名元组，轻量数据包
from typing import NamedTuple

# 张量
import torch


# =========================================================
# 1. KV 缓存池基类（管理一大块显存）
# 作用：规定“显存管理器”必须有哪些功能
# =========================================================
class BaseKVCachePool(ABC):
    """
    KV 缓存的基类
    定义所有 KV 缓存必须实现的接口
    """

    # 获取第 layer_id 层的 K 缓存
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    # 获取第 layer_id 层的 V 缓存
    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    # 把新的 K、V 存到缓存里
    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    # 设备：GPU
    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    # 精度：fp16/bf16
    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    # 多少层
    @property
    @abstractmethod
    def num_layers(self) -> int: ...


# =========================================================
# 2. 缓存句柄（代表一段被缓存的前缀）
# 你可以把它理解为“缓存的门票”
# =========================================================
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int  # 已经缓存了多长的前缀

    # 拿到缓存里的下标位置
    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


# =========================================================
# 3. 缓存大小信息
# 记录：能释放多少、保护多少
# =========================================================
class SizeInfo(NamedTuple):
    evictable_size: int  # 可以释放的缓存长度
    protected_size: int  # 受保护、不能释放的长度

    # 总缓存长度
    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


# =========================================================
# 4. 插入缓存的结果
# =========================================================
class InsertResult(NamedTuple):
    cached_len: int  # 插入前已经缓存的长度
    handle: BaseCacheHandle  # 缓存的门票


# =========================================================
# 5. 匹配前缀的结果
# =========================================================
class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle  # 缓存门票


# =========================================================
# 6. 前缀缓存基类（最核心！缓存管理器总规则）
# 作用：规定“缓存系统”必须能做什么
# =========================================================
class BasePrefixCache(ABC):

    # 锁定/解锁缓存
    # 锁定 = 不能被删除
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    # 匹配前缀
    # 查一下：这个请求之前有没有缓存过？
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        pass

    # 把新的前缀插入缓存
    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        pass

    # 释放缓存空间
    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        pass

    # 清空缓存
    @abstractmethod
    def reset(self) -> None:
        pass

    # 获取缓存大小信息
    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        pass

    # 检查缓存是否损坏
    @abstractmethod
    def check_integrity(self) -> None:
        pass