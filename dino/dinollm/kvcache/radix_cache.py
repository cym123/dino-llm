from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from dinollm.core import get_global_ctx
from dinollm.utils import align_down

# 导入缓存基类（规则）
from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

# 定义类型：key 生成函数
KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


# ======================================================
# 【核心数据结构】基数树（前缀树）节点
# 每个节点存一段 token 序列 + 对应的 KV 缓存下标
# ======================================================
class RadixTreeNode:
    counter: int = 0  # 给每个节点唯一编号

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn          # 生成子节点 key 的函数
        self.children: Dict[Any, RadixTreeNode] = {}  # 子节点
        self._parent: RadixTreeNode | None = None     # 父节点
        self.ref_count: int = 0       # 引用计数（=0 可删除）
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()  # 时间戳（LRU 用）

        # 下面三个字段存真正的内容
        self._key: torch.Tensor       # token 序列（前缀）
        self._value: torch.Tensor     # 对应的 KV 下标
        self._length: int             # 长度

    # 设置节点保存的前缀 + 缓存下标
    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    # 设置父节点
    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    # 【关键】比较当前节点的前缀和输入，返回匹配长度
    """
    例子：当前节点 N1 key：[1,2,3,4]输入新来 [1,2,3,4,9,10]fast_compare_key 逐个比，
          返回从头连续匹配了多少个这里返回 4。
    作用：判断新来请求和当前节点前缀能对上多长。
    """
    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from dinollm.kernel import fast_compare_key
        return fast_compare_key(self._key, input_ids)

    # 分裂节点（前缀只匹配一部分时）
    """
    假设：
    原来节点 X 存：[1,2,3,4,5,6,7,8] 长度 8
    新来请求只匹配到前 4 个 [1,2,3,4]，后面对不上
    调用 split_at(pos=4)
    分裂过程：
    新建一个新节点 NewNode
    NewNode 存前半段：[1,2,3,4]
    原来的节点 X 砍成后半段：[5,6,7,8]
    父子关系改成：
    """
    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        # 新建节点，存前半部分
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        # 自己存后半部分
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    # 比较时间戳（用于 LRU 淘汰）
    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


# ======================================================
# 缓存句柄（拿到它 = 拿到一段缓存的使用权）
# ======================================================
@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode  # 指向树节点

    # 把从根到当前节点的所有下标拼起来 = 完整缓存位置
    """
    匹配到路径：root → N1 (1,2,3,4) → N2(5,6,7,8)
    从 N2 往上遍历
    收集每个节点的 value（KV 下标）
    反转拼接，得到整条前缀对应的完整 KV 缓存位置
    Handle 就是一个 “门票”，拿着它就能取出已经缓存好的 KV 位置
    """
    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


# ======================================================
# 【整个文件最重要】Radix 前缀缓存实现
# 作用：共享重复前缀的 KV 缓存
# ======================================================
class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size  # 页大小
        self.key_fn = _get_key_fn(self.page_size)    # 生成 key
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0    # 可释放缓存长度
        self.protected_size = 0    # 受保护缓存长度
        self.root_node = RadixTreeNode(self.key_fn)  # 根节点
        self.root_node.ref_count = 1  # 根永远不删

    # 锁定/解锁缓存（锁定 = 不能被释放）
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            # 解锁：引用计数减少
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            # 加锁：引用计数增加
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    # 【核心API】匹配前缀：查有没有缓存
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    # 【核心API】插入前缀：把新请求存入缓存
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)

        if prefix_len < insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node

        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    # 【核心API】释放缓存（LRU 淘汰最老的）
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            node = heapq.heappop(leave_nodes)
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length

            parent = node.parent
            del parent.children[self.key_fn(node._key)]

            # 如果父节点也变成叶子，也加入淘汰队列
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError()

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(self.evictable_size, self.protected_size)

    def check_integrity(self) -> None:
        pass

    # 收集所有叶子节点（用于淘汰）
    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes = [self.root_node]
        leave_nodes = []
        while nodes:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                nodes.extend(node.children.values())
        return leave_nodes

    # 【最核心】树遍历：查找最长匹配前缀
    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        total_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < total_len:
            # 找子节点
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len

            node = child
            # 计算匹配长度
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # 没完全匹配 → 分裂节点
            if match_len != node.length:
                node = node.split_at(match_len)
                return node, prefix_len

            # 更新访问时间（LRU）
            node.timestamp = tic

        return node, prefix_len


# 根据页大小生成 key
def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
