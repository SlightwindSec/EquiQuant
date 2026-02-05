from typing import List, Literal

import time
import torch
from torch import Tensor, nn

from aqt.quantizer import Quantizer
from aqt.sensitivity import get_sensitivity_metric, get_linear_output


class PostTrainingQuantization:

    def __init__(
        self,
        layer: nn.Module,
        quant_type: Literal["minmax", "ssz"] = "minmax",
        quant_bits: int = 8,
        quant_sym: bool = True,
        context_length: int = 2048,
        group_size: int = 0,
        sensitivity_metric: str = None,
    ) -> None:
        self.layer = layer
        self.quant_type = quant_type
        self.quant_bits = quant_bits
        self.quant_sym = quant_sym
        self.device = self.layer.weight.device
        self.group_size = group_size

        if self.group_size != 0:
            assert self.group_size % 32 == 0

        self.columns = self.layer.weight.data.shape[-1]
        self.inps = torch.zeros(
            (context_length, self.columns), dtype=torch.float32, device=self.device
        )
        self.bias = self.layer.bias if hasattr(self.layer, 'bias') and self.layer.bias is not None else None
        self.samples_num = 0
        self.context_length = context_length

        self.sensitivity_metric = get_sensitivity_metric(sensitivity_metric)

        self.orig_weight = None
        self.dequant_weight = None

        self.quantizer = Quantizer(self.quant_type, self.quant_bits, self.quant_sym, self.group_size)

    def add_batch(self, inp: Tensor, out: Tensor) -> None:
        inp = inp[0]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_samples_num = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = nn.functional.pad(inp, (0, 0, self.context_length - inp.shape[0], 0))
        self.samples_num += batch_samples_num
        self.inps.add_(inp)

    def post_batch(self) -> None:
        if self.samples_num > 0:
            self.inps /= self.samples_num
        torch.npu.empty_cache()

    def quantize_dequantize(self) -> None:
        self.orig_weight = self.layer.weight.data.clone()

        quant_weight, scale, zero_point = self.quantizer.quantize_weight(
            weight=self.orig_weight,
            dtype=self.orig_weight.dtype
        )

        self.dequant_weight = self.quantizer.dequantize_weight(
            quant_weight=quant_weight,
            scale=scale,
            zero_point=zero_point,
        )

        self.dequant_weight = self.dequant_weight.to(self.layer.weight.device).to(self.layer.weight.dtype)

    def run(self, transform_weights: bool = False) -> List[float]:
        tick = time.time()

        self.quantize_dequantize()

        dtype = self.layer.weight.data.dtype
        self.inps = self.inps.to(dtype)
        non_zero_tokens_mask = self.inps.sum(1) != 0.0
        self.inps = self.inps[non_zero_tokens_mask]
        self.orig_weight = self.orig_weight.to(dtype)

        output = get_linear_output(
            inp=self.inps,
            weight=self.orig_weight,
            bias=self.bias
        )
        losses = [
            self.sensitivity_metric(
                inp=self.inps,
                output=output,
                weight=self.orig_weight,
                quant_weight=self.dequant_weight,
                bias=self.bias,
            ).item()
        ]

        if not transform_weights:
            self.layer.weight.data = self.orig_weight.clone()

        return losses

    def free(self) -> None:
        self.inps = None
        self.quantizer = None
        self.orig_weight = None
        self.dequant_weight = None
        torch.npu.empty_cache()
