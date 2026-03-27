import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal
from aqt.utils.common import cleanup_memory


class Quantizer:
    def __init__(
        self,
        quant_type: Literal["minmax", "ssz"] = "minmax",
        quant_bits: int = 8,
        quant_sym: bool = True,
        group_size: int = 0,
        percentile: float = 0.999,
    ):
        self.quant_type = quant_type.lower()
        self.quant_bits = quant_bits
        self.quant_sym = quant_sym
        self.percentile = percentile
        self.group_size = group_size

        if self.group_size != 0:
            raise NotImplementedError("Currently not support per_group.")

        self.n_levels = 2**quant_bits

        if quant_sym:
            self.qmin = -(2 ** (quant_bits - 1))
            self.qmax = (2 ** (quant_bits - 1)) - 1
            self.dtype = torch.int8
        else:
            self.qmin = 0
            self.qmax = self.n_levels - 1
            self.dtype = torch.uint8

        self._scale: Optional[torch.Tensor] = None
        self._zero_point: Optional[torch.Tensor] = None

    @torch.no_grad()
    def find_params(
        self,
        weight: torch.Tensor,
        is_weight: bool = True,
    ) -> None:
        if not is_weight:
            raise NotImplementedError("Only weight quantization is supported now!")
        if weight.dim() != 2:
            raise ValueError(f"Weight tensor must be 2D, got shape {weight.shape}")

        if self.quant_type == "minmax":
            self._find_params_minmax(weight)
        elif self.quant_type == "ssz":
            self._find_params_ssz(weight)
        else:
            raise ValueError(f"Unsupported quant_type: {self.quant_type}")

    def _find_params_minmax(self, weight: torch.Tensor) -> None:
        w_min = weight.min(dim=1, keepdim=True)[0]
        w_max = weight.max(dim=1, keepdim=True)[0]

        self._calculate_scale_zp(w_min, w_max, weight.device)

    def _find_params_ssz(self, weight: torch.Tensor) -> None:
        if self.quant_sym:
            w_abs_max = torch.quantile(weight.abs(), self.percentile, dim=1, keepdim=True)
            w_min = -w_abs_max
            w_max = w_abs_max
        else:
            lower_percentile = 1.0 - self.percentile
            w_min = torch.quantile(weight, lower_percentile, dim=1, keepdim=True)
            w_max = torch.quantile(weight, self.percentile, dim=1, keepdim=True)

        self._calculate_scale_zp(w_min, w_max, weight.device)

    def _calculate_scale_zp(self, w_min: torch.Tensor, w_max: torch.Tensor, device: torch.device) -> None:
        out_features = w_min.shape[0]

        if self.quant_sym:
            w_abs_max = torch.max(w_min.abs(), w_max.abs())
            self._scale = w_abs_max / (2 ** (self.quant_bits - 1) - 1)
            self._zero_point = torch.zeros(
                out_features, 1, dtype=torch.float32, device=device
            )
        else:
            self._scale = (w_max - w_min) / (self.qmax - self.qmin)
            self._scale = torch.clamp(self._scale, min=1e-8)
            self._zero_point = torch.round(self.qmin - w_min / self._scale)

        self._scale = torch.clamp(self._scale, min=1e-8)

    @torch.no_grad()
    def quantize_weight(
        self,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.find_params(weight, is_weight=True)

        if self._scale is None or self._zero_point is None:
            raise RuntimeError("Quantization parameters not found.")

        scale_float = self._scale.to(weight.dtype)
        zp_float = self._zero_point.to(weight.dtype)
        
        quant_weight_float = torch.round(weight / scale_float) + zp_float
        quant_weight_float = torch.clamp(quant_weight_float, self.qmin, self.qmax)
        quant_weight = quant_weight_float.to(self.dtype)

        return quant_weight, self._scale.float(), self._zero_point.float()

    @torch.no_grad()
    def dequantize_weight(
        self,
        quant_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        quant_weight = quant_weight.float().to(device)
        scale = scale.to(device)
        zero_point = zero_point.to(device)
        dequant_weight = ((quant_weight - zero_point) * scale).to(dtype)

        del quant_weight, scale, zero_point
        return dequant_weight
