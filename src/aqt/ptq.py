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
        group_size: int = 0,
    ) -> None:
        self.layers = layers
        self.quant_type = quant_type
        self.quant_bit = quant_bit
        self.group_size = group_size
        
        if self.group_size != 0:
            raise NotImplementedError("Currently per-group quantization is not supported.")

        first_layer = next(iter(layers.values()))
        self.dtype = first_layer.weight.dtype
        self.device = first_layer.weight.device

        self.orig_weight: Dict[str, Tensor] = {}
        self.quantizer = Quantizer(self.quant_type, self.quant_bit)

    @torch.no_grad()
    def run(self) -> None:
        logger.info("Post-training quantization started.")
        for name, layer in self.layers.items():
            self.orig_weight[name] = layer.weight.to("cpu", copy=True)
            fq_weight = self.quantizer.fake_quantize_weight(layer.weight)
            layer.weight.copy_(fq_weight)
            del fq_weight
        cleanup_memory()

    @torch.no_grad()
    def free(self) -> None:
        logger.info("Restoring original weights and cleaning up...")
        for name, layer in self.layers.items():
            layer.weight.copy_(self.orig_weight[name])
        del self.orig_weight, self.quantizer
        cleanup_memory()
        logger.info("Post-training quantization completed and memory freed.")
