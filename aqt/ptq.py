from typing import Literal

import torch
from torch import Tensor, nn
from typing import Dict

from .quantizer import Quantizer
from .utils.common import cleanup_memory
from ..utils.logger import logger


class PostTrainingQuantization:
    def __init__(
        self,
        layers: Dict[str, nn.Module],
        quant_type: Literal["minmax", "ssz"] = "minmax",
        quant_bit: int = 8,
        quant_sym: bool = True,
        group_size: int = 0,
    ) -> None:
        self.layers = layers
        self.quant_type = quant_type
        self.quant_bit = quant_bit
        self.quant_sym = quant_sym
        self.group_size = group_size
        if self.group_size != 0:
            assert self.group_size % 32 == 0, f"group_size {self.group_size} must be divisible by 32"

        first_layer = next(iter(layers.values()))
        self.dtype = first_layer.weight.data.dtype
        self.device = first_layer.weight.data.device

        self.orig_weight: Dict[str, Tensor] = {}
        self.quantizer = Quantizer(
            self.quant_type, self.quant_bit, self.quant_sym, self.group_size
        )

    def quantize_dequantize(self, weight: Tensor) -> Tensor:
        quant_weight, scale, zero_point = self.quantizer.quantize_weight(weight=weight)
        dequant_weight = self.quantizer.dequantize_weight(
            quant_weight=quant_weight,
            scale=scale,
            zero_point=zero_point,
            dtype=self.dtype,
            device=self.device,
        )
        return dequant_weight

    @torch.no_grad()
    def run(self) -> None:
        logger.info("Post-training quantization started.")
        for name, layer in self.layers.items():
            self.orig_weight[name] = layer.weight.data.to("cpu", non_blocking=True, copy=True)
            layer.weight.data = self.quantize_dequantize(layer.weight.data)
        cleanup_memory()

    def free(self) -> None:
        logger.info("Post-training quantization completed.")
        for name, layer in self.layers.items():
            layer.weight.data = self.orig_weight[name].to(self.device, non_blocking=True)
        del self.quantizer, self.orig_weight
        cleanup_memory()
