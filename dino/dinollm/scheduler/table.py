import torch


# =========================================================
# TableManager：表格管理器
# 作用：管理【请求在页表里的槽位】
# 也就是：给每个正在运行的请求，分配一个 table_idx
# =========================================================
class TableManager:
    def __init__(
        self,
        max_running_reqs: int,  # 最多同时跑多少个请求
        page_table: torch.Tensor  # 页表（全局KV缓存位置表）
    ) -> None:
        self._max_running_reqs = max_running_reqs

        # 空闲槽位列表：[0,1,2,3,...,max_running_reqs-1]
        # 每个数字代表一个“请求槽位”
        self._free_slots = list(range(max_running_reqs))

        # 全局页表：记录每个请求的每个token存在KV缓存的哪个位置
        # shape: [max_running_reqs, max_pages_per_req]
        self.page_table = page_table

        # token池：存储每个请求的输入token
        # 形状和页表完全一样
        # 作用：快速读取请求的token，用于前缀匹配
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    # 可用槽位数量（还能同时跑多少请求）
    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    # 分配一个槽位：来一个新请求，拿走一个编号
    def allocate(self) -> int:
        return self._free_slots.pop()

    # 释放槽位：请求结束，把编号还回去
    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
