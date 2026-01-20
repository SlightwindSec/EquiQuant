from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn

from utils.quantizer import Quantizer
from utils.model import get_linear_layers_classes


class PostTrainingQuantization(ABC):
    """Base class for post-training quantization methods."""

    def __init__(
        self,
        layer: nn.Module,
        quantizer: Optional[Quantizer] = None,
        context_length: int = 2048,
        group_size: int = 0,
    ) -> None:
        self.layer = layer
        self.quantizer = quantizer
        self.device = self.layer.weight.device
        self.group_size = group_size

        if self.group_size != 0:
            assert self.group_size % 32 == 0

        self.columns = self.layer.weight.data.shape[-1]
        self.inps = torch.zeros(
            (context_length, self.columns), dtype=torch.float32, device=self.device
        )
        self.bias = self.layer.bias

        self.add_batch = self._add_batch
        self.samples_num = 0
        self.context_length = context_length

    def free(self) -> None:
        self.inps = None
        self.quantizer = None
        torch.cuda.empty_cache()

    def _add_batch(self, inp: Tensor, _: Tensor) -> None:
        raise NotImplementedError

    def post_batch(self) -> None:
        torch.cuda.empty_cache()

    @abstractmethod
    def run(self, transform_weights: bool = True) -> None:
        raise NotImplementedError
