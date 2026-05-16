from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


# ==========================
# 工具函数：递归序列化任意数据
# 作用：把 字典/列表/张量/对象 变成可传输的 JSON 格式
# ==========================
def _serialize_any(value: Any) -> Any:
    # 如果是字典 → 递归把每个 value 都序列化
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}

    # 如果是列表/元组 → 逐个元素序列化
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)

    # 如果是基础类型（int/float/str/bool/None/bytes）→ 直接返回
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value

    # 其他情况（自定义对象）→ 调用对象的 serialize_type
    else:
        return serialize_type(value)


# ==========================
# 核心：把一个对象 → 可传输的字典
# 被所有消息类（BaseBackendMsg / BaseFrontendMsg 等）调用
# ==========================
def serialize_type(self) -> Dict:
    # 最终输出的字典
    serialized = {}

    # ======================
    # 特殊处理：torch.Tensor
    # ======================
    if isinstance(self, torch.Tensor):
        # 目前只支持 1D tensor（input_ids 都是 1D）
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"      # 标记类型：Tensor
        serialized["buffer"] = self.numpy().tobytes()  # 张量 → 字节
        serialized["dtype"] = str(self.dtype)  # 保存数据类型
        return serialized

    # ======================
    # 普通对象（UserMsg / DetokenizeMsg 等）
    # ======================
    serialized["__type__"] = self.__class__.__name__  # 记录类名

    # 遍历对象所有成员变量（比如 uid, input_ids, finished 等）
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)

    return serialized


# ==========================
# 工具函数：递归反序列化
# 作用：把字典 → 恢复成原来的对象
# ==========================
def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    # 如果是字典
    if isinstance(data, dict):
        # 如果带 __type__ 标记 → 是对象，需要反序列化
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        # 普通字典 → 递归恢复每个 value
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}

    # 列表/元组 → 逐个恢复
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)

    # 基础类型 → 直接返回
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data

    # 不支持的类型
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


# ==========================
# 核心：把字典 → 恢复成对象
# ==========================
def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    # 读取类型标记（UserMsg / Tensor / DetokenizeMsg 等）
    type_name = data["__type__"]

    # ======================
    # 特殊恢复：Tensor
    # ======================
    if type_name == "Tensor":
        buffer = data["buffer"]                # 字节数据
        dtype_str = data["dtype"].replace("torch.", "")  # 类型：int32 / bf16
        np_dtype = getattr(np, dtype_str)      # 转成 numpy 类型
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)  # 字节 → numpy
        return torch.from_numpy(np_tensor.copy())         # numpy → torch

    # ======================
    # 恢复普通对象
    # ======================
    cls = cls_map[type_name]       # 根据类型名找到类（如 UserMsg）
    kwargs = {}                   # 构造参数

    # 遍历所有字段，恢复成员变量
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)

    # 创建对象并返回  cls(**kwargs)
    return cls(**kwargs)
