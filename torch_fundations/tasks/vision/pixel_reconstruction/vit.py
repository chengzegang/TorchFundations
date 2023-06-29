from abc import ABCMeta
from typing import Callable, Literal, Tuple, Type

from ....meta import Available
from ....models.vision import VisionTransformer as Backbone
from torch.nn import Module, Linear, Conv2d, LayerNorm
from torch import Tensor
import torch.nn.functional as F


class VisionTransformerPixelHead(Module):
    def __init__(
        self,
        out_channels: int = 3,
        image_size: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.norm = LayerNorm(hidden_size)
        self.proj = Linear(hidden_size, out_channels * patch_size**2)
        self.conv = Conv2d(out_channels, out_channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.norm(inputs)
        inputs = self.proj(inputs).transpose(-1, -2)
        inputs = F.fold(
            inputs, self.image_size, self.patch_size, stride=self.patch_size
        )
        results = self.conv(inputs)
        return results


class VisionTransformer(Module):
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
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
        self.backbone = Backbone(
            in_channels,
            patch_size,
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
            position_embedding_type,
            position_embedding_max_length,
        )
        self.head = VisionTransformerPixelHead(
            out_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden_states = self.backbone(inputs)
        results = self.head(hidden_states)
        return results


class VisionTransformerAvailable(
    metaclass=Available, task="pixel_reconstruction", model=VisionTransformer
):
    ...
