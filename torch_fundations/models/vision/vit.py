from dataclasses import dataclass, field
from torch.nn import (
    Module,
    ModuleList,
    InstanceNorm2d,
    Linear,
    ReLU,
    Conv2d,
    Identity,
    Sequential,
    Parameter,
    Embedding,
    Dropout,
    LayerNorm,
)
import torch
from torch import Tensor
from typing import Any, List, Tuple, Union, Optional, Literal, Callable, Type, Dict
import torch.nn.functional as F

from ...meta import Available


class Zeros(Module):
    def forward(*args, **kwargs):
        return 0


class FeedForward(Module):
    def __init__(
        self,
        hidden_size: int,
        feedforward_size: int,
        activation: Callable | Module = F.gelu,
        norm: Type[Module] = LayerNorm,
    ):
        super().__init__()
        self.activation = activation
        self.norm = norm(hidden_size)
        self.proj1 = Linear(hidden_size, feedforward_size)
        self.proj2 = Linear(feedforward_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden_states = self.activation(self.proj1(inputs))
        hidden_states = self.proj2(hidden_states) + inputs
        results = self.norm(hidden_states)
        return results


class MultiheadAttention(Module):
    def __init__(
        self,
        hidden_size: int,
        feedforward_size: int,
        num_heads: int,
        activation: Module | Callable = F.gelu,
        norm: Type[Module] = LayerNorm,
        dropout: float = 0.1,
        query_proj: bool = True,
        key_proj: bool = True,
        value_proj: bool = True,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        attention_residual_connection: bool = True,
        attention_residual_connection_type: Literal["linear", "identity"] = "linear",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.activation = activation
        self.query_proj = (
            Linear(hidden_size, hidden_size, bias=query_bias)
            if query_proj
            else Identity()
        )
        self.key_proj = (
            Linear(hidden_size, hidden_size, bias=key_bias) if key_proj else Identity()
        )
        self.value_proj = (
            Linear(hidden_size, hidden_size, bias=value_bias)
            if value_proj
            else Identity()
        )
        self.attn_res_conn = (
            Linear(hidden_size, hidden_size)
            if attention_residual_connection
            and attention_residual_connection_type == "linear"
            else Identity()
            if attention_residual_connection
            and attention_residual_connection_type == "identity"
            else Zeros()
        )
        self.post_attention_proj = Linear(hidden_size, hidden_size)
        self.post_attention_norm = norm(hidden_size)
        self.dropout = Dropout(dropout)
        self.feedforward = FeedForward(hidden_size, feedforward_size, activation, norm)
        self.norm = norm(hidden_size)

    def split_heads(self, inputs: Tensor) -> Tensor:
        results = (
            inputs.contiguous()
            .view(inputs.shape[0], inputs.shape[1], self.num_heads, -1)
            .transpose(-2, -3)
        )
        return results

    def merge_heads(self, inputs: Tensor) -> Tensor:
        results = inputs.transpose(-2, -3).flatten(-2)
        return results

    def forward(
        self,
        priors: Tensor,
        conditions: Tensor | None = None,
        attention_masks: Tensor | None = None,
    ) -> Tensor:
        conditions = priors if conditions is None else conditions
        queries = self.query_proj(priors)
        keys = self.key_proj(conditions)
        values = self.value_proj(conditions)
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)
        attn_values = F.scaled_dot_product_attention(
            queries, keys, values, attention_masks
        )
        attn_values = self.merge_heads(attn_values)
        attn_values = self.post_attention_proj(attn_values) + self.attn_res_conn(priors)
        attn_values = self.post_attention_norm(attn_values)

        attn_values = self.dropout(attn_values)
        results = self.feedforward(attn_values)
        return results


class PatchEmbedding2d(Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        norm: Type[Module] = LayerNorm,
    ):
        super().__init__()
        self.conv = Conv2d(in_channels, hidden_size, patch_size, patch_size)
        self.norm = norm(hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden_states = self.conv(inputs).flatten(-2).transpose(-2, -1)
        results = self.norm(hidden_states)
        return results


class RelativePositionEmbedding(Module):
    def __init__(self, hidden_size: int, norm: Type[Module] = LayerNorm):
        super().__init__()
        self.proj = Linear(1, hidden_size)
        self.norm = norm(hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        position_ids = torch.linspace(
            -1, 1, inputs.shape[-2], device=inputs.device, dtype=inputs.dtype
        ).view(1, -1, 1)
        position_embeds = self.proj(position_ids)
        position_embeds = position_embeds.expand_as(inputs) + inputs
        position_embeds = self.norm(position_embeds)
        return position_embeds


class AbsolutePositionEmbedding(Module):
    def __init__(
        self, hidden_size: int, max_length: int, norm: Type[Module] = LayerNorm
    ):
        super().__init__()
        self.position_ids = Parameter(
            torch.arange(max_length).view(1, -1), requires_grad=False
        )
        self.position_embeds = Embedding(max_length, hidden_size)
        self.norm = norm(hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        position_embeds = self.position_embeds(
            self.position_ids[:, : inputs.shape[-2]]
        ).expand_as(inputs)
        hidden_states = inputs + position_embeds
        hidden_states = self.norm(hidden_states)
        return position_embeds


class Transformer(Module):
    def __init__(
        self,
        hidden_size: int,
        feedforward_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: Module | Callable = F.gelu,
        norm: Type[Module] = LayerNorm,
        query_proj: bool = True,
        key_proj: bool = True,
        value_proj: bool = True,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        attention_residual_connection: bool = True,
        attention_residual_connection_type: Literal["linear", "identity"] = "linear",
    ):
        super().__init__()
        self.layers = ModuleList(
            [
                MultiheadAttention(
                    hidden_size,
                    feedforward_size,
                    num_heads,
                    activation,
                    norm,
                    dropout,
                    query_proj,
                    key_proj,
                    value_proj,
                    query_bias,
                    key_bias,
                    value_bias,
                    attention_residual_connection,
                    attention_residual_connection_type,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        priors: Tensor,
        condition: Tensor | None = None,
        attention_masks: Tensor | None = None,
    ) -> Tensor:
        hidden_states = priors
        for layer in self.layers:
            hidden_states = layer(hidden_states, condition, attention_masks)
        return hidden_states


class VisionTransformer(Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        hidden_size: int = 768,
        feedforward_size: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        activation: Module | Callable = F.gelu,
        norm: Type[Module] = LayerNorm,
        query_proj: bool = True,
        key_proj: bool = True,
        value_proj: bool = True,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        attention_residual_connection: bool = True,
        attention_residual_connection_type: Literal["linear", "identity"] = "linear",
        position_embedding_type: Literal["absolute", "relative"] = "relative",
        position_embedding_max_length: int = 10000,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding2d(
            in_channels, hidden_size, patch_size, norm
        )
        self.position_embedding = (
            AbsolutePositionEmbedding(hidden_size, position_embedding_max_length, norm)
            if position_embedding_type == "absolute"
            else RelativePositionEmbedding(hidden_size, norm)
        )
        self.transformer = Transformer(
            hidden_size,
            feedforward_size,
            num_heads,
            num_layers,
            dropout,
            activation,
            norm,
            query_proj,
            key_proj,
            value_proj,
            query_bias,
            key_bias,
            value_bias,
            attention_residual_connection,
            attention_residual_connection_type,
        )

    def forward(
        self,
        priors: Tensor,
        conditions: Tensor | None = None,
        attention_masks: Tensor | None = None,
    ) -> Tensor:
        hidden_states = self.patch_embedding(priors)
        hidden_states = self.position_embedding(hidden_states)
        hidden_states = self.transformer(hidden_states, conditions, attention_masks)
        return hidden_states


class VisionTransformerConfig(metaclass=Available, target=VisionTransformer):
    ...
