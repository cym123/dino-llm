# 缓存工具：让函数只运行一次，提速
import functools

# 处理JSON格式（读取配置文件）
import json

# 操作系统工具：判断文件夹、路径等
import os

# 类型提示：Any = 任意类型
from typing import Any

# 从 HuggingFace 云端下载模型
from huggingface_hub import hf_hub_download, snapshot_download

# 进度条工具（这里用来关闭进度条）
from tqdm.asyncio import tqdm

#  Transformers 库：自动加载模型配置、分词器
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase
)


# ==========================
# 1. 关闭下载进度条的类
# ==========================
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        # 删掉名字参数
        kwargs.pop("name", None)
        # 关键：disable=True = 关闭进度条
        kwargs["disable"] = True
        # 调用父类初始化
        super().__init__(*args, **kwargs)


# ==========================
# 2. 加载分词器（文字 ↔ 数字）
# ==========================
def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    # 自动加载模型对应的分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 如果分词器没有对话模板（Mistral模型常见）
    if not getattr(tokenizer, "chat_template", None):
        try:
            # 从云端下载 chat_template.json 对话模板文件
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            # 读取并设置对话模板
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            # 下载失败就算了，不报错
            pass

    # 返回可用的分词器
    return tokenizer


# ==========================
# 3. 带缓存的：加载模型配置文件
# ==========================
@functools.cache  # 缓存：只加载一次
def _load_hf_config(model_path: str) -> Any:
    # 自动读取模型的 config.json
    return AutoConfig.from_pretrained(model_path)


# ==========================
# 4. 安全读取模型配置
# ==========================
def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    # 从缓存拿配置
    config = _load_hf_config(model_path)
    # 复制一份新配置返回，防止原配置被意外修改
    return type(config)(**config.to_dict())


# ==========================
# 5. 下载模型权重（.safetensors 文件）
# ==========================
def download_hf_weight(model_path: str) -> str:
    # 如果 model_path 是本地文件夹 → 直接返回
    if os.path.isdir(model_path):
        return model_path

    try:
        # 从 HuggingFace 云端下载模型
        return snapshot_download(
            model_path,                # 模型名字
            allow_patterns=["*.safetensors"],  # 只下权重文件
            tqdm_class=DisabledTqdm,   # 关闭进度条
        )
    except Exception as e:
        # 既不是本地路径，也不是有效模型 → 报错
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )