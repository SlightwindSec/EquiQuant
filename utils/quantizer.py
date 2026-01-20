import torch
from torch import Tensor, nn

SCALE_SEARCH_ITER_NUM = 30
SCALE_SEARCH_CONVERGE_THRESHOLD = 1e-10
SCALE_SEARCH_MIN_SCALE = 1e-5


def quantize_qfna(x: Tensor, scale: Tensor, zero: Tensor, maxq: Tensor) -> Tensor:
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero

    q = torch.clamp(
        torch.round(x / scale) + zero, torch.tensor(0, device=x.device), maxq
    )

    return scale * (q - zero)


quantize = quantize_qfna


def quantize_qfnb(x: Tensor, scale: Tensor, maxq: Tensor) -> Tensor:
    q = x / scale
    q = torch.clamp(
        torch.round(((q + 1) / 2) * maxq), torch.tensor(0, device=x.device), maxq
    )
    q = (q / maxq) * 2 - 1
    return q * scale


def quantize(x: Tensor, scale: Tensor, zero: Tensor, maxq: Tensor) -> Tensor:
    q = torch.clamp(
        torch.round(x / scale) + zero, torch.tensor(0, device=x.device), maxq
    )
    return q


class Quantizer(nn.Module):  # pylint: disable=abstract-method
    maxq: Tensor
    scale: Tensor
    zero: Tensor

    def __init__(
        self,
        shape: int = 1,
        bits: int = 3,
        qfn: str = "a",
        perchannel: bool = False,
        sym: bool = True,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        trits: bool = False,
    ) -> None:
        super().__init__()

        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

        self.maxq = torch.tensor(2**bits - 1)
        if trits:
            self.maxq = torch.tensor(-1)

        self.qfn = qfn
        self.perchannel = perchannel
        self.sym = sym
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x: Tensor, weight: bool = False) -> None:
        if self.qfn in ["a", "c"]:
            self.find_params_qfna(x, weight=weight)
        elif self.qfn == "b":
            self.find_params_qfnb(x)
        elif self.qfn == "ssz":
            self.find_params_ssz(x, weight=weight)
        else:
            raise ValueError("Unknown quantization function name.")

    def find_params_qfna(self, x: Tensor, weight: bool = False) -> None:
        device = x.device
        self.maxq = self.maxq.to(device)

        shape = list(x.shape)
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=device)
        x_min = torch.minimum(x.min(1)[0], tmp)
        x_max = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            x_max = torch.maximum(torch.abs(x_min), x_max)
            tmp = x_min < 0
            if torch.any(tmp):
                x_min[tmp] = -x_max[tmp]

        tmp = (x_min == 0) & (x_max == 0)
        x_min[tmp] = -1
        x_max[tmp] = 1

        if self.maxq < 0:
            self.scale = x_max
            self.zero = x_min
        else:
            self.scale = (x_max - x_min) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq.item() + 1) / 2)
            else:
                self.zero = torch.round(-x_min / self.scale)

        if not self.perchannel:
            if weight:
                tmp2 = shape[0]
            else:
                tmp2 = shape[1] if len(shape) != 3 else shape[2]

            self.scale = self.scale.repeat(tmp2)
            self.zero = self.zero.repeat(tmp2)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return

        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def find_params_qfnb(self, x: Tensor) -> None:
        dev = x.device
        self.maxq = self.maxq.to(dev)
        self.scale = 2.4 * x.square().mean().sqrt() + 1e-16

    def find_params_ssz(self, x: Tensor, weight=False):
        self.find_params_qfna(x, weight=weight)
        weight = x
        best_scale = self.scale
        best_zero = self.zero

        if len(best_scale.shape) == 1:
            best_scale = best_scale.unsqueeze(1)
            best_zero = best_zero.unsqueeze(1)

        best_quant_weight = quantize(
            x=weight, scale=self.scale, zero=self.zero, maxq=self.maxq
        )
        best_dequant_weight = quantize_qfna(
            x=weight, scale=self.scale, zero=self.zero, maxq=self.maxq
        )

        best_mse = torch.mean(
            torch.pow(torch.abs((weight - best_dequant_weight)), 2), dim=1, keepdim=True
        )
        quant_weight = best_quant_weight

        for _ in range(SCALE_SEARCH_ITER_NUM):
            quant_weight_tensor = quant_weight.to(weight.dtype)
            if self.sym:
                current_scale = torch.sum(
                    weight * quant_weight_tensor, dim=1, keepdim=True
                ) / torch.sum(
                    quant_weight_tensor * quant_weight_tensor, dim=1, keepdim=True
                ).clamp(
                    min=SCALE_SEARCH_MIN_SCALE
                )
                self.scale = current_scale
                current_zero = self.zero
                quant_weight = quantize(
                    x=weight, scale=self.scale, zero=self.zero, maxq=self.maxq
                )
            else:
                quant_weight_minus_zero = quant_weight_tensor - self.zero
                current_scale = torch.sum(
                    weight * quant_weight_minus_zero, dim=1, keepdim=True
                ) / torch.sum(
                    quant_weight_minus_zero * quant_weight_minus_zero,
                    dim=1,
                    keepdim=True,
                ).clamp(
                    min=SCALE_SEARCH_MIN_SCALE
                )
                current_zero = torch.sum(
                    quant_weight_tensor * current_scale - weight, dim=1, keepdim=True
                ) / (weight.shape[0] * current_scale)

                self.scale = current_scale
                self.zero = current_zero
                quant_weight = quantize(
                    x=weight, scale=self.scale, zero=self.zero, maxq=self.maxq
                )
            current_dequant_weight = quantize_qfna(
                x=weight, scale=self.scale, zero=self.zero, maxq=self.maxq
            )
            current_mse = torch.mean(
                torch.pow(torch.abs((weight - current_dequant_weight)), 2),
                dim=1,
                keepdim=True,
            )
            mask1 = (
                torch.abs(best_mse - current_mse) / best_mse.clamp(min=1e-4)
                < SCALE_SEARCH_CONVERGE_THRESHOLD
            )
            mask2 = torch.abs(best_mse - current_mse) < SCALE_SEARCH_CONVERGE_THRESHOLD
            if (
                torch.sum(
                    torch.logical_and(
                        torch.logical_not(mask1), torch.logical_not(mask2)
                    )
                )
                == 0
            ):
                break

            mask = (current_mse < best_mse).to(torch.int32)
            if mask.sum() == 0:
                mask = (current_mse - 1e-5 < best_mse).to(torch.int32)
            best_mse = best_mse * (1 - mask) + current_mse * mask
            best_scale = best_scale * (1 - mask) + current_scale * mask
            if self.sym:
                best_zero = self.zero
            else:
                best_zero = best_zero * (1 - mask) + current_zero * mask

            best_quant_weight = best_quant_weight * (1 - mask) + quant_weight * mask
            quant_weight = best_quant_weight

        self.scale = best_scale
        self.zero = best_zero

    def quantize(self, x: Tensor) -> Tensor:
        assert self.ready()

        if self.qfn == "a" or self.qfn == "ssz":
            return quantize_qfna(x=x, scale=self.scale, zero=self.zero, maxq=self.maxq)
        elif self.qfn == "b":
            self.scale = 2.4 * x.square().mean().sqrt() + 1e-16
            return quantize_qfnb(x=x, scale=self.scale, maxq=self.maxq)
        else:
            raise NotImplementedError

    def ready(self) -> Tensor:
        return torch.all(self.scale != torch.tensor(0))
