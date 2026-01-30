import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal


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
            raise NotImplementedError(f"Currently not support per_group.")

        self.n_levels = 2 ** quant_bits
        if quant_sym:
            self.qmin = -(2 ** (quant_bits - 1))
            self.qmax = (2 ** (quant_bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = self.n_levels - 1
        
        self._scale: Optional[torch.Tensor] = None
        self._zero_point: Optional[torch.Tensor] = None
    
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
        out_features = weight.shape[0]

        w_min = weight.min(dim=1, keepdim=True)[0]
        w_max = weight.max(dim=1, keepdim=True)[0]

        if self.quant_sym:
            w_abs_max = torch.max(w_min.abs(), w_max.abs())
            self._scale = w_abs_max / (2 ** (self.quant_bits - 1) - 1)
            self._zero_point = torch.zeros(out_features, 1, dtype=torch.int32, device=weight.device)
        else:
            self._scale = (w_max - w_min) / (self.qmax - self.qmin)
            self._zero_point = torch.round(self.qmin - w_min / self._scale).to(torch.int32)
        
        self._scale = torch.clamp(self._scale, min=1e-8)
    
    def _find_params_ssz(self, weight: torch.Tensor) -> None:
        w_abs_max = torch.quantile(weight.abs(), self.percentile, dim=1, keepdim=True)
        
        if self.quant_sym:
            self._scale = w_abs_max / (2 ** (self.quant_bits - 1) - 1)
            self._zero_point = torch.zeros_like(w_abs_max, dtype=torch.int32)
        else:
            w_min = -w_abs_max
            w_max = w_abs_max
            self._scale = (w_max - w_min) / (self.qmax - self.qmin)
            self._zero_point = torch.round(self.qmin - w_min / self._scale).to(torch.int32)
        
        self._scale = torch.clamp(self._scale, min=1e-8)
    
    def quantize_weight(
        self,
        weight: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dtype is None:
            dtype = weight.dtype

        self.find_params(weight, is_weight=True)

        if self._scale is None or self._zero_point is None:
            raise RuntimeError("Quantization parameters not found. Call find_params first.")
        
        quant_weight = torch.clamp(
            torch.round(weight / self._scale) + self._zero_point.to(self._scale.dtype),
            self.qmin,
            self.qmax
        ).to(torch.int8)
        
        return quant_weight, self._scale.to(dtype), self._zero_point
    
    def dequantize_weight(
        self,
        quant_weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        dequant_weight = (quant_weight.to(scale.dtype) - zero_point.to(scale.dtype)) * scale
        return dequant_weight
    
    def reset(self) -> None:
        self._scale = None
        self._zero_point = None
