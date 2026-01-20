import time
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from utils.ptq_method import PostTrainingQuantization
from utils.quantizer import Quantizer
from utils.sensitivity import get_sensitivity_metric, get_linear_output


class MinMax(PostTrainingQuantization):
    def __init__(
        self,
        layer: nn.Module,
        quantizer: Quantizer,
        sq_scales: Optional[Tensor] = None,
        group_size: int = 0,
        context_length: int = 2048,
        sensitivity_metric: List[str] = None,
    ) -> None:
        """
        Min Max Quantization
        """
        super().__init__(
            layer=layer,
            quantizer=quantizer,
            context_length=context_length,
            group_size=group_size,
        )

        self.sq_scales = sq_scales
        self.sensitivity_metric = list(map(get_sensitivity_metric, sensitivity_metric))

    def _add_batch(self, inp: Tensor, out: Tensor) -> None:
        inp = inp[0]

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)  # TODO: think about case with BS > 1

        batch_samples_num = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = nn.functional.pad(inp, (0, 0, self.context_length - inp.shape[0], 0))

        if self.sq_scales is not None:
            # In activations quantization scenario, if smooth scales have been
            # computed, weights were divided by them and there were no
            # layernorm fusing, we need to multiply inputs by smooth quant scales
            # in order to support the equivalence of computations
            inp = inp / self.sq_scales.to(inp).view(-1)

        self.samples_num += batch_samples_num
        self.inps.add_(inp)

    def post_batch(self) -> None:
        self.inps /= self.samples_num
        torch.cuda.empty_cache()

    def run(self, transform_weights: bool = True) -> float:
        tick = time.time()

        weights = self.layer.weight.data.clone().float()

        quant_weights, losses = self._run(weights=weights)

        if transform_weights:
            self.layer.weight.data = quant_weights.reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype
            )

        print(f"Quantization took {time.time() - tick:.4f}s")
        if losses is not None and not transform_weights:
            print(f"Mean error: {losses}")

        return losses

    def _run(self, weights: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.quantizer.ready():
            self.quantizer.find_params(weights, weight=True)

        return self._run_minmax(weights=weights)

    def _run_minmax(self, weights: Tensor) -> Tuple[Tensor, Optional[List[float]]]:
        scales = []
        zeros = []

        quantizer = self.quantizer
        quant_weights = torch.zeros_like(weights)

        group_size = self.columns if self.group_size == 0 else self.group_size

        for i in range(0, self.columns, group_size):
            w = weights[:, i : i + group_size]

            quantizer.find_params(
                weights[:, i : i + group_size],
                weight=True,
            )

            scales.append(quantizer.scale)
            zeros.append(quantizer.zero)

            q = quantizer.quantize(x=w)
            quant_weights[:, i : i + group_size] = q

        # Note: when computing metrics we need to cast dequant weights
        # back into original dtype. Moreover, for expert layers we
        # pad inps to the context_length, so we have to drop zero tokens
        # before computing sensitivity scores.
        dtype = self.layer.weight.data.dtype
        self.inps = self.inps.to(dtype)
        non_zero_tokens_mask = self.inps.sum(1) != 0.0
        self.inps = self.inps[non_zero_tokens_mask]
        weights = weights.to(dtype)

        output = get_linear_output(inp=self.inps, weight=weights, bias=self.bias)
        losses = [
            metric(
                inp=self.inps,
                output=output,
                weight=weights,
                quant_weight=quant_weights,
                bias=self.bias,
            ).item()
            for metric in self.sensitivity_metric
        ]

        if scales is not None:
            self.quantizer.scale = torch.stack(scales, dim=1)
            self.quantizer.zero = torch.stack(zeros, dim=1)

        return quant_weights, losses
