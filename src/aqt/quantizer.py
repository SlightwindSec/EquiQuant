import torch
from typing import Tuple, Literal


class Quantizer:
    def __init__(
        self,
        quant_type: Literal["minmax", "ssz"] = "minmax",
        quant_bits: int = 8,
        percentile: float = 0.999,
    ):
        self.quant_type = quant_type.lower()
        self.quant_bits = quant_bits
        self.percentile = percentile

        if not (2 <= quant_bits <= 8):
            raise ValueError(f"quant_bits must be between 2 and 8, got {quant_bits}")

        self.qmax = (1 << (quant_bits - 1)) - 1
        self.qmin = -(1 << (quant_bits - 1))
        
        self.dtype = torch.int8

    @torch.no_grad()
    def find_params(self, weight: torch.Tensor) -> None:
        if weight.dim() != 2:
            raise ValueError(f"Weight tensor must be 2D, got shape {weight.shape}")

        if self.quant_type == "minmax":
            w_min = weight.min(dim=1, keepdim=True)[0]
            w_max = weight.max(dim=1, keepdim=True)[0]
            
        elif self.quant_type == "ssz":
            w_min = torch.quantile(weight.float(), 1 - self.percentile, dim=1, keepdim=True).to(weight.dtype)
            w_max = torch.quantile(weight.float(), self.percentile, dim=1, keepdim=True).to(weight.dtype)
            
        else:
            raise ValueError(f"Unsupported quant_type: {self.quant_type}")

        w_min = torch.clamp(w_min, max=0.0)
        w_max = torch.clamp(w_max, min=0.0)

        divisor = float(self.qmax - self.qmin)
        scale = (w_max - w_min) / divisor
        scale = torch.clamp(scale, min=1e-8)

        zp = torch.round(self.qmin - w_min / scale)
        zp = torch.clamp(zp, self.qmin, self.qmax)

        return scale, zp.to(torch.int8)

    @torch.no_grad()
    def quantize_weight(
        self,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale, zp = self.find_params(weight)

        if scale is None or zp is None:
            raise RuntimeError("Parameters not found.")

        scale = scale.to(weight.dtype)
        zp = zp.to(weight.dtype)

        quant_weight = torch.round(weight / scale) + zp
        quant_weight = torch.clamp(quant_weight, self.qmin, self.qmax)
        
        return quant_weight.to(self.dtype), scale, zp

    @staticmethod
    @torch.no_grad()
    def dequantize_weight(
        quant_weight: torch.Tensor,
        scale: torch.Tensor,
        zp: torch.Tensor,
        out_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        device = quant_weight.device

        qw_float = quant_weight.to(out_dtype)
        zp_float = zp.to(device=device, dtype=out_dtype)
        s_float = scale.to(device=device, dtype=out_dtype)

        dequant_weight = (qw_float - zp_float) * s_float

        del quant_weight, scale, zp
        return dequant_weight

    @torch.no_grad()
    def fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        scale, zp = self.find_params(weight)

        if scale is None or zp is None:
            raise RuntimeError("Parameters not found.")

        scale = scale.to(weight.dtype)
        zp = zp.to(weight.dtype)

        quantized = torch.round(weight / scale) + zp
        quantized = torch.clamp(quantized, self.qmin, self.qmax)

        fake_quantized = (quantized - zp) * scale

        del quantized, scale, zp
        return fake_quantized
    

def pack_4bit_to_8bit(unpacked_w: torch.Tensor) -> torch.Tensor:
    if unpacked_w.shape[1] % 2 != 0:
        raise ValueError("In_features (dim=1) must be divisible by 2 for 4-bit packing.")

    left_half = unpacked_w[:, 0::2]
    right_half = unpacked_w[:, 1::2]

    left_half_uint8 = left_half.to(torch.uint8) & 0x0F
    right_half_uint8 = right_half.to(torch.uint8) & 0x0F

    packed_w = (left_half_uint8 << 4) | right_half_uint8
    
    return packed_w

def unpack_8bit_to_4bit(packed_w: torch.Tensor) -> torch.Tensor:
    M, N_half = packed_w.shape
    unpacked_w = torch.empty((M, N_half * 2), dtype=torch.int8, device=packed_w.device)

    unpacked_w[:, 0::2] = (packed_w.to(torch.int8) >> 4)
    unpacked_w[:, 1::2] = ((packed_w << 4).to(torch.int8) >> 4)
    
    return unpacked_w
