from __future__ import annotations


from abc import ABC, abstractmethod

from dataclasses import dataclass

from typing import NamedTuple

import torch



class BaseKVCachePool(ABC):

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...


    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...


    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...


    @property
    @abstractmethod
    def device(self) -> torch.device: ...


    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...


    @property
    @abstractmethod
    def num_layers(self) -> int: ...



@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...



class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int


    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size



class InsertResult(NamedTuple):
    cached_len: int
    handle: BaseCacheHandle



class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle



class BasePrefixCache(ABC):


    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass


    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        pass


    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        pass


    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        pass


    @abstractmethod
    def reset(self) -> None:
        pass


    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        pass

    @abstractmethod
    def check_integrity(self) -> None:
        pass