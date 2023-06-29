import torch
from torch.nn import (
    Conv2d,
    ModuleList,
    Sequential,
    Module,
    UpsamplingNearest2d,
    Identity,
    InstanceNorm2d,
)
from torch import Tensor
from typing import Any, List, Tuple, Union, Optional, Literal, Callable, Type, Dict
import torch.nn.functional as F

from ...meta import Available


class ScaleConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 1,
        scale_direction: Literal["up", "down", "none"] = "up",
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        if scale_direction == "down":
            kernel_size = scale_factor * 2 - scale_factor % 2
            stride = scale_factor
            padding = kernel_size // 2
            self.conv: Module = Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode=padding_mode,
            )
        elif scale_direction == "up":
            self.conv = Sequential(
                UpsamplingNearest2d(scale_factor),
                Conv2d(in_channels, out_channels, 1),
            )
        elif scale_direction == "none":
            self.conv = Conv2d(in_channels, out_channels, 1)
        else:
            raise ValueError('scale_direction must be "up", "down" or "none"')

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ResidualConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_channels: int = 0,
        padding_mode: str = "zeros",
        scale_factor: int = 1,
        scale_direction: Literal["up", "down", "none"] = "none",
        activation: Callable | Module = F.relu,
        norm: Type[Module] = InstanceNorm2d,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
        )
        self.conv3 = ScaleConv2d(
            out_channels + condition_channels,
            out_channels,
            scale_factor,
            scale_direction,
        )
        self.res_connect = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = norm(out_channels + condition_channels)
        self.activation = activation

    def forward(self, priors: Tensor, conditions: Tensor | None = None) -> Tensor:
        hidden_states = self.activation(self.conv1(priors))
        hidden_states = self.activation(self.conv2(hidden_states))
        if conditions is not None:
            hidden_states = torch.cat([hidden_states, conditions], dim=1)
        hidden_states = self.norm(self.conv3(hidden_states) + self.res_connect(priors))
        results = self.activation(hidden_states)
        return results


class UNetEncoder(Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        channel_multiply_factor: int = 2,
        padding_mode: str = "zeros",
        activation: Callable | Module = F.relu,
        norm: Type[Module] = InstanceNorm2d,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiply_factor**num_layers
        self.layers = ModuleList(
            [
                ResidualConv2d(
                    in_channels * channel_multiply_factor**i,
                    in_channels * channel_multiply_factor ** (i + 1),
                    padding_mode=padding_mode,
                    activation=activation,
                    norm=norm,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, priors: Tensor) -> Tensor:
        hidden_states = priors
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class UNetDecoder(Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        channel_multiply_factor: int = 2,
        padding_mode: str = "zeros",
        activation: Callable | Module = F.relu,
        norm: Type[Module] = InstanceNorm2d,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // channel_multiply_factor**num_layers
        self.layers = ModuleList(
            [
                ResidualConv2d(
                    in_channels // channel_multiply_factor**i,
                    in_channels // channel_multiply_factor ** (i + 1),
                    in_channels // channel_multiply_factor**i,
                    padding_mode=padding_mode,
                    activation=activation,
                    norm=norm,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, priors: Tensor) -> Tensor:
        hidden_states = priors
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Proxy:
    #  avoid circular reference
    def __init__(self, obj: Any):
        self.obj = obj

    def __getattr__(self, name: str) -> Any:
        return getattr(self.obj, name)

    def __repr__(self) -> str:
        return f"Proxy({self.obj.__class__.__name__})"


class ProxyConnect(Module):
    def __init__(self, server: Module, client: Module, key: str = "conditions"):
        super().__init__()
        self.server = Proxy(server)
        self.client = Proxy(client)
        self.key = key
        self._cache_values: Tensor | None = None
        server.register_forward_hook(self._cache_forward_hook)
        client.register_forward_pre_hook(self._fetch_proxy_value_hook, with_kwargs=True)
        if not hasattr(client, "proxy_connects"):
            setattr(client, "proxy_connects", ModuleList())
            client.register_module("proxy_connects", getattr(client, "proxy_connects"))
        getattr(client, "proxy_connects").append(self)

    def _cache_forward_hook(self, module: Module, arg: Any, result: Any) -> None:
        self._cache_values = result

    def _fetch_proxy_value_hook(
        self, module: Module, arg: Any, kwargs: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        kwargs[self.key] = self._cache_values
        return arg, kwargs

    def __repr__(self) -> str:
        return (
            f"ProxyConnect(server={self.server}, client={self.client}, key={self.key})"
        )


class UNet(Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        channel_multiply_factor: int = 2,
        padding_mode: str = "zeros",
        activation: Callable | Module = F.relu,
        norm: Type[Module] = InstanceNorm2d,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = UNetEncoder(
            in_channels,
            num_layers,
            channel_multiply_factor,
            padding_mode,
            activation,
            norm,
        )
        self.decoder = UNetDecoder(
            self.encoder.out_channels,
            num_layers,
            channel_multiply_factor,
            padding_mode,
            activation,
            norm,
        )
        self.out_channels = self.decoder.out_channels

        for encoder_layer, decoder_layer in zip(
            self.encoder.layers[::-1], self.decoder.layers
        ):
            ProxyConnect(encoder_layer, decoder_layer)

    def forward(self, priors: Tensor) -> Tensor:
        hidden_states = self.encoder(priors)
        results = self.decoder(hidden_states)
        return results
