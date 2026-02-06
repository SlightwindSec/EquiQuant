from __future__ import annotations

import os
import warnings
from collections import defaultdict
from os.path import join as pjoin
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import Tensor, nn

from aqt.utils.quant_config_manager import QuantLayerConfigManager
from utils.logger import logger


SensitivityScoresT = Dict[str, Dict[int, Dict[str, float]]]


def get_sensitivity_metric(name: str) -> Callable:
    if name in SUPPORTED_SENSITIVITY_METRICS:
        return SUPPORTED_SENSITIVITY_METRICS[name]
    raise NotImplementedError(
        "Currently, only the following sensitivity metrics are supported:"
        f"{list(SUPPORTED_SENSITIVITY_METRICS.keys())}"
    )


def MSE_loss(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    quant_out = get_linear_output(inp, quant_weight, bias)
    MSE_loss = torch.norm(output - quant_out, 2) / output.numel()
    return MSE_loss


def KLD_loss(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    quant_out = get_linear_output(inp, quant_weight, bias)
    P_out = torch.log_softmax(output, dim=1)
    P_quant_out = torch.log_softmax(quant_out, dim=1)
    KLD_loss = F.kl_div(P_quant_out, P_out, reduction="batchmean", log_target=True)
    return KLD_loss


def entropy_loss(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    P_out = torch.softmax(output, dim=1) + 1e-12
    entropy_loss = (-P_out * P_out.log()).sum(1).mean()
    return entropy_loss


def SQNR_loss(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    SQNR_loss = 10 * torch.log((((weight - quant_weight) ** 2 / weight**2).sum()).sum())
    SQNR_loss /= weight.numel()
    return SQNR_loss


def SQNR_out_loss(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    quant_out = get_linear_output(inp, quant_weight, bias)
    SQNR_loss = 10 * torch.log(
        1 + ((output - quant_out) ** 2 / (output**2).sum()).sum()
    )
    SQNR_loss /= output.numel()
    return SQNR_loss


def cosine_similarity_loss(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    quant_out = get_linear_output(inp, quant_weight, bias)
    cosine_sim = 1 - F.cosine_similarity(output, quant_out, dim=1)
    return cosine_sim.mean()


def MRE_loss(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    quant_out = get_linear_output(inp, quant_weight, bias)
    MRE_score = torch.mean((output - quant_out).abs() / (output.abs() + 1e-9))
    return MRE_score


def HIGGS_loss(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    quant_out = get_linear_output(inp, quant_weight, bias)
    higgs_score = torch.norm(output - quant_out) ** 2 / (torch.norm(output) ** 2 + 1e-9)
    return higgs_score / output.numel()


def weight_range(
    inp: Tensor,
    output: Tensor,
    weight: Tensor,
    quant_weight: Tensor,
    bias: Optional[Tensor],
) -> Tensor:
    return weight.max() - weight.min()


SUPPORTED_SENSITIVITY_METRICS = {
    "mse": MSE_loss,
    "kld": KLD_loss,
    "entropy": entropy_loss,
    "sqnr": SQNR_loss,
    "sqnr_out": SQNR_out_loss,
    "mre": MRE_loss,
    "cosine": cosine_similarity_loss,
    "higgs": HIGGS_loss,
    "range": weight_range,
}


def get_linear_output(inp: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
    if bias is not None:
        bias = bias.to(inp)

    return F.linear(inp, weight, bias)


def get_layer_sensitivity_group_mapping(experts_num: int = 0) -> Dict[str, List[str]]:
    return {
        "qkv_a_proj": ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"],
        "qkv_b_proj": ["self_attn.q_b_proj", "self_attn.kv_b_proj"],
        "qkv_proj": ["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj"],
        "o_proj": ["self_attn.o_proj"],
        ".experts": [
            f"mlp.experts.{expert_idx}.{layer_type}"
            for expert_idx in range(experts_num)
            for layer_type in ["up_proj", "gate_proj", "down_proj"]
        ],
        "shared_experts.up_gate_proj": [
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.gate_proj",
        ],
        "shared_experts.down_proj": ["mlp.shared_experts.down_proj"],
        "up_gate_proj": ["mlp.up_proj", "mlp.gate_proj"],
        "down_proj": ["mlp.down_proj"],
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


def analyze_sensitivity_scores(
    sensitivity_scores: SensitivityScoresT,
    score_name: str,
    save_dir: Union[str, os.PathLike],
    experts_num: int = 0,
) -> None:
    layers_mapping = get_layer_sensitivity_group_mapping(experts_num)

    for bits in [4, 8]:
        bit_lines = []
        fig, ax = plt.subplots()
        layers_set = set()
        for subset_name, layer_names in layers_mapping.items():
            if len(layer_names) == 0:
                continue

            layer_name = layer_names[0]
            line = []
            for name, bit_mapping in sensitivity_scores.items():
                if (layer_name in name) and (name not in layers_set):
                    score = bit_mapping[bits][score_name]
                    line.append(score)

                    layer_type = name.replace(layer_name, subset_name)
                    subset_names = get_subset_layer_names(layer_type, layers_mapping)
                    for sub_name in subset_names:
                        layers_set.add(sub_name)

            if len(line) == 0:
                continue
            bit_lines.append((line, name))
            plt.plot(line, marker="o")
            plt.yscale("log")
            plt.title(f"{score_name}, {subset_name} {bits} bits")
            plt.xlabel("Layer Num")
            plt.ylabel("Sensitivity Score")
            save_path = pjoin(save_dir, f"{score_name}_{subset_name}_{bits}_bits.png")
            plt.savefig(save_path)
            plt.close()

            ax.plot(line, marker="o", label=subset_name)

        ax.set_yscale("log")
        ax.set_title(f"{score_name}, {bits} bits")
        ax.set_xlabel("Layer Num")
        ax.set_ylabel("Sensitivity Score")
        ax.legend()
        fig.savefig(pjoin(save_dir, f"{score_name}_{bits}_bits.png"))
        plt.close()


def show_diff_between_bits(
    sensitivity_scores: SensitivityScoresT,
    score_name: str,
    save_dir: Optional[os.PathLike] = None,
    experts_num: int = 0,
) -> None:
    layers_mapping = get_layer_sensitivity_group_mapping(experts_num)

    diff_lines = []
    fig, ax = plt.subplots()
    layers_set = set()
    for subset_name, layer_names in layers_mapping.items():
        if len(layer_names) == 0:
            continue

        layer_name = layer_names[0]
        line = []
        for name, bit_mapping in sensitivity_scores.items():
            if layer_name in name and name not in layers_set:
                score_diff = bit_mapping["ratio"][score_name]
                line.append(score_diff)

                layer_type = name.replace(layer_name, subset_name)
                subset_names = get_subset_layer_names(layer_type, layers_mapping)
                for sub_name in subset_names:
                    layers_set.add(sub_name)

        if not line:
            continue

        diff_lines.append((line, name))
        plt.plot(line, marker="o")
        plt.yscale("log")
        plt.title(f"{score_name}, {subset_name} 4-8 bits diff")
        plt.xlabel("Layer Num")
        plt.ylabel("Sensitivity Score Difference")
        save_path = pjoin(save_dir, f"{score_name}_{subset_name}_diff.png")
        plt.savefig(save_path)
        plt.close()

        ax.plot(line, marker="o", label=subset_name)

    ax.set_yscale("log")
    ax.set_title(f"{score_name}, 4-8 bits diff")
    ax.set_xlabel("Layer Num")
    ax.set_ylabel("Sensitivity Score Difference")
    ax.legend()
    fig.savefig(pjoin(save_dir, f"{score_name}_diff.png"))
    plt.close()
