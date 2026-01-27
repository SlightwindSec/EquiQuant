import inspect
import time
from typing import List

import torch
from torch import Tensor, nn

try:
    from msmodelslim.pytorch.llm_ptq.accelerate_adapter.hook_adapter import (
        PrepareWeight,
    )
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator
    from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import LinearQuantizer
    from msmodelslim.pytorch.lowbit.atomic_power_outlier import (
        quant_one_weight_by_outliers as quant_one_weight_by_outliers_low_bit,
    )
    from msmodelslim.pytorch.lowbit.quant_modules import (
        LinearQuantizer as LowBitLinearQuantizer,
    )
except ImportError:
    PrepareWeight = None
    Calibrator = None
    LinearQuantizer = None
    quant_one_weight_by_outliers_low_bit = None
    LowBitLinearQuantizer = None

from aqt.ptq_method import PostTrainingQuantization
from aqt.sensitivity import get_sensitivity_metric, get_linear_output


class ModelslimQuantization(PostTrainingQuantization):
    def __init__(
        self,
        layer: nn.Module,
        calibrator: Calibrator,
        group_size: int = 0,
        context_length: int = 2048,
        sensitivity_metric: List[str] = None,
    ) -> None:
        assert calibrator is not None

        super().__init__(
            layer=layer,
            quantizer=None,
            context_length=context_length,
            group_size=group_size,
        )

        self.sensitivity_metric = list(map(get_sensitivity_metric, sensitivity_metric))

    def _add_batch(self, inp: Tensor, out: Tensor) -> None:
        inp = inp[0]

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)  # TODO: think about case with BS > 1

        batch_samples_num = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = nn.functional.pad(inp, (0, 0, self.context_length - inp.shape[0], 0))

        self.samples_num += batch_samples_num
        self.inps.add_(inp)

    def post_batch(self) -> None:
        self.inps /= self.samples_num
        torch.cuda.empty_cache()

    def run(self, transform_weights: bool = False) -> float:
        tick = time.time()
        orig_weight = self.layer.weight.data.clone()
        with PrepareWeight(self.layer):
            if isinstance(self.layer, LinearQuantizer):
                # quant was disabled when PTQ class was created
                self.layer.quant_weight.is_enable = True
                self.layer.quant_input.is_enable = True

                dequant_weight = self.layer.quant_weight(self.layer.weight)

                # disable quant again
                self.layer.quant_weight.is_enable = False
                self.layer.quant_input.is_enable = False
                self.layer.quant_weight.has_init_quant_para = False
                self.layer.quant_input.has_init_quant_para = False
            elif isinstance(self.layer, LowBitLinearQuantizer):
                # TODO: don't know how to turn off quantization when run inference in
                # lowbit=True scenario
                raise NotImplementedError
                self.layer.weight_quant_flag = False

                if not self.layer.cfg.is_stage_quant:
                    self.layer.fp_weight = self.layer.weight.cpu().clone()

                kwargs = {
                    "powerquant": self.layer.cfg.nonuniform,
                    "fraction": self.layer.cfg.fraction,
                    "num_bits": self.layer.cfg.w_bit,
                    "isolate_outlier_amax": False,
                    "per_channel": not self.layer.cfg.mm_tensor,
                    "use_cuda": True if self.layer.cfg.dev_type == "gpu" else False,
                    "use_sigma": self.layer.cfg.use_sigma,
                    "sigma_factor": self.layer.cfg.sigma_factor,
                    "open_outlier": self.layer.cfg.open_outlier,
                    "group_size": self.layer.cfg.group_size,
                    "w_sym": self.layer.cfg.w_sym,
                    "use_hqq": self.layer.cfg.hqq,
                }
                if (
                    "progressive"
                    in inspect.signature(
                        quant_one_weight_by_outliers_low_bit
                    ).parameters
                ):
                    kwargs["progressive"] = self.layer.cfg.is_stage_quant

                dequant_weight, scale_w, _, offset_w = (
                    quant_one_weight_by_outliers_low_bit(self.layer.weight, **kwargs)
                )
                is_scale_w_list = isinstance(scale_w, list) and len(scale_w) == 2
                is_offset_w_list = isinstance(offset_w, list) and len(offset_w) == 2
                if is_scale_w_list and is_offset_w_list:
                    self.layer.quant_weight.weight_scale = scale_w[0].cpu()
                    self.layer.quant_weight.weight_offset = offset_w[0].cpu()
                    self.layer.quant_weight.weight_scale_second = scale_w[1].cpu()
                    self.layer.quant_weight.weight_offset_second = offset_w[1].cpu()
                else:
                    self.layer.quant_weight.weight_scale = scale_w.cpu()
                    self.layer.quant_weight.weight_offset = offset_w.cpu()
                self.layer.has_init_quant_para = True
                if not self.layer.cfg.is_stage_quant:
                    self.layer.weight[:] = dequant_weight[:]

                self.layer.weight_quant_flag = True

        # Note: when computing metrics we need to cast dequant weights
        # back into original dtype. Moreover, for expert layers we
        # pad inps to the context_length, so we have to drop zero tokens
        # before computing sensitivity scores.
        dtype = self.layer.weight.data.dtype
        self.inps = self.inps.to(dtype)
        non_zero_tokens_mask = self.inps.sum(1) != 0.0
        self.inps = self.inps[non_zero_tokens_mask]
        orig_weight = orig_weight.to(dtype)

        output = get_linear_output(inp=self.inps, weight=orig_weight, bias=self.bias)
        losses = [
            metric(
                inp=self.inps,
                output=output,
                weight=orig_weight,
                quant_weight=dequant_weight,
                bias=self.bias,
            ).item()
            for metric in self.sensitivity_metric
        ]

        print(f"Quantization took {time.time() - tick:.4f}s")
        if losses is not None and not transform_weights:
            print(f"Mean error: {losses}")

        return losses
