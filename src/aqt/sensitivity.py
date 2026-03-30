from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from ..utils.logger import logger


def get_sensitivity_metric(name: str) -> Callable:
    if name in SUPPORTED_SENSITIVITY_METRICS:
        return SUPPORTED_SENSITIVITY_METRICS[name]
    raise NotImplementedError(
        "Currently, only the following sensitivity metrics are supported:"
        f"{list(SUPPORTED_SENSITIVITY_METRICS.keys())}"
    )


def mse(
    y_true: torch.Tensor,
    y_fake: torch.Tensor,
) -> float:
    mse_loss = F.mse_loss(y_fake, y_true, reduction="mean").item()
    return mse_loss


def cosine_sim(
    y_true: torch.Tensor,
    y_fake: torch.Tensor,
) -> float:
    cosine_sim = F.cosine_similarity(y_true, y_fake, dim=-1)
    cosine_sim_loss = 1.0 - cosine_sim.mean().item()
    return cosine_sim_loss


def relative_l2(
    y_true: torch.Tensor,
    y_fake: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    diff_norm = torch.norm(y_true - y_fake, p="fro")
    true_norm = torch.norm(y_true, p="fro")
    relative_l2 = (diff_norm / (true_norm + eps)).item()
    return relative_l2


def snr(
    y_true: torch.Tensor,
    y_fake: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    noise_power = torch.pow(torch.norm(y_true - y_fake, p="fro"), 2)
    signal_power = torch.pow(torch.norm(y_true, p="fro"), 2)
    snr = (10 * torch.log10(signal_power / (noise_power + eps))).item()
    return snr


def kl_div(
    y_true: torch.Tensor,
    y_fake: torch.Tensor,
) -> float:
    log_probs_fake = F.log_softmax(y_fake, dim=-1)
    probs_true = F.softmax(y_true, dim=-1)
    kl_div = F.kl_div(log_probs_fake, probs_true, reduction="batchmean").item()
    return kl_div


SUPPORTED_SENSITIVITY_METRICS = {
    "cosine": cosine_sim,
    "mse": mse,
    "relative_l2": relative_l2,
    "snr_db": snr,
    "kl_div": kl_div,
}


def calculate_losses(
    y_true: torch.Tensor, y_fake: torch.Tensor, metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    assert y_true.shape == y_fake.shape, "y_true and y_fake must have the same shape"

    if metrics is None:
        metrics = list(SUPPORTED_SENSITIVITY_METRICS.keys())
        logger.warning(
            f"No metrics specified, defaulting to all supported metrics: {metrics}"
        )

    y_true = y_true.float()
    y_fake = y_fake.float()

    if y_true.dim() == 3:
        D = y_true.shape[-1]
        y_true = y_true.view(-1, D)
        y_fake = y_fake.view(-1, D)

    results = {}
    for metric_name in metrics:
        metric_func = get_sensitivity_metric(metric_name)
        try:
            result = metric_func(y_true, y_fake)
            results[metric_name] = result
        except Exception as e:
            logger.warning(f"计算 {metric_name} 时出错: {e}")
            results[metric_name] = float("nan")
    return results


def get_layer_sensitivity_group_mapping(num_experts: int | None = None) -> Dict[str, List[str]]:
    return {
        "self_attn.qkv_a_proj": ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"],
        "self_attn.q_b_proj": ["self_attn.q_b_proj"],
        # "self_attn.k_b_proj": ["self_attn.k_b_proj"],
        "self_attn.qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "self_attn.o_proj": ["self_attn.o_proj"],
        # "linear_attn.in_proj_qkvz": ["linear_attn.in_proj_qkvz"],
        # "linear_attn.in_proj_ba": ["linear_attn.in_proj_ba"],        
        "linear_attn.in_proj_qkvz": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"],
        "linear_attn.in_proj_ba": ["linear_attn.in_proj_b", "linear_attn.in_proj_a"],
        "linear_attn.out_proj": ["linear_attn.out_proj"],
        "mlp.experts": [
            f"mlp.experts.{expert_idx}.{layer_type}"
            for expert_idx in range(num_experts)
            for layer_type in ["gate_proj", "up_proj", "down_proj"]
        ] if num_experts is not None else ["mlp.experts"],
        "mlp.shared_expert.gate_up_proj": [
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
        ],
        "mlp.shared_expert.down_proj": ["mlp.shared_expert.down_proj"],
        "mlp.shared_experts.gate_up_proj": [
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
        ],
        "mlp.shared_experts.down_proj": ["mlp.shared_experts.down_proj"],
        "mlp.gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
        "mlp.down_proj": ["mlp.down_proj"],
    }


def get_subset_layer_names(
    subset_name: str, layers_mapping: Dict[str, List[str]]
) -> List[str]:
    for layer_type, mapping in layers_mapping.items():
        if layer_type in subset_name:
            return [
                subset_name.replace(layer_type, subset_layer)
                for subset_layer in mapping
            ]
