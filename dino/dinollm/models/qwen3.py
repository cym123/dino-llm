from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from dinollm.core import get_global_ctx
from dinollm.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from dinollm.utils import nvtx_annotate,div_ceil

from dinollm.distributed import DistributedCommunicator, get_tp_info

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

import torch.nn as nn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # self.embed_tokens = VocabParallelEmbedding(
        #     num_embeddings=config.vocab_size,
        #     embedding_dim=config.hidden_size,
        # )
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,


        print("num_embeddings 类型:", type(num_embeddings))
        print("num_embeddings 值:", num_embeddings)
        print("num_embeddings[0] 类型:", type(num_embeddings[0]))
        print("num_embeddings[0] 值:", num_embeddings[0])
        self.embed_tokens = nn.Embedding(num_embeddings[0], embedding_dim[0])

        self._comm = DistributedCommunicator()

        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        # x = self._comm.all_reduce(x) if self.tp_size > 1 else x
        residual: torch.Tensor | None = None
        for layer in self.layers:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3ForCausalLM"]
