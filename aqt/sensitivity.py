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

from aqt.utils.lp_solver import LPS, get_constraints_for_search
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
    # TODO: in comparison with quantization, linear layers group processing,
    # sensitivity scores currently should match vllm-ascend shared layers rules,
    # so we have to comply with them by aggregating sensitivity scores for such
    # groups. Refactor this logic later?
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


MEGABYTE_SIZE = 1024**2
BYTES_PER_4BIT_PARAM = 0.5


def update_quant_layer_cfg_greedy(
    sensitivity_scores: Dict[str, Dict[int, Any]],
    model: nn.Module,
    quant_layer_cfg_mngr: QuantLayerConfigManager,
    score_name: str,
    ckpt_size_budget_mb: int = 500,
) -> None:
    experts_num = getattr(model.config, "num_experts", 0)
    layers_mapping = get_layer_sensitivity_group_mapping(experts_num)
    layer_score_info = []
    seen_layers = set()
    for name, bit_mapping in sensitivity_scores.items():
        for layer_subset, layer_names in layers_mapping.items():
            if not layer_names:
                continue

            if layer_names[0] in name and name not in seen_layers:
                score = bit_mapping["ratio"][score_name]
                layer_type = name.replace(layer_names[0], layer_subset)
                layer_score_info.append((score, layer_type))

                subset_names = get_subset_layer_names(
                    subset_name=layer_type, layers_mapping=layers_mapping
                )
                for subset_name in subset_names:
                    seen_layers.add(subset_name)

    layer_score_info.sort(reverse=True)

    curr_ckpt_diff = 0
    layer_num = 0

    # TODO: we consider only 4 vs 8 bit case here. Extend with fp16/bf16 later
    ckpt_size_budget_mb = ckpt_size_budget_mb * MEGABYTE_SIZE
    while curr_ckpt_diff < ckpt_size_budget_mb and layer_num < len(layer_score_info):
        subset_name = layer_score_info[layer_num][1]

        weight_size = 0
        layer_names = get_subset_layer_names(subset_name, layers_mapping)
        for layer_name in layer_names:
            weight_size += sensitivity_scores[layer_name]["size"] * BYTES_PER_4BIT_PARAM

        if curr_ckpt_diff + weight_size <= ckpt_size_budget_mb:
            curr_ckpt_diff += weight_size
            for layer_name in layer_names:
                quant_layer_cfg_mngr.cfg[layer_name].weight_bits = 8

        layer_num += 1

    for _, subset_name in layer_score_info[layer_num:]:
        layer_names = get_subset_layer_names(subset_name, layers_mapping)
        for layer_name in layer_names:
            quant_layer_cfg_mngr.cfg[layer_name].weight_bits = 4


def update_quant_layer_cfg_lp(
    sensitivity_scores: Dict[str, Dict[int, Any]],
    model: nn.Module,
    quant_layer_cfg_mngr: QuantLayerConfigManager,
    score_name: str,
    ckpt_size_budget_mb: int = 500,
) -> None:
    # TODO: greedy strategy is prefered for now. Revisit LP solution later.
    best_scores = run_quant_schema_search(
        model=model,
        sensitivity_scores=sensitivity_scores,
        score_name=score_name,
        ckpt_size_budget_mb=ckpt_size_budget_mb,
    )

    for layer_name, bits in best_scores.items():
        quant_layer_cfg_mngr.cfg[layer_name].weight_bits = bits


def run_quant_schema_search(
    model: nn.Module,
    sensitivity_scores: Dict[str, Dict[str, Any]],
    score_name: str,
    ckpt_size_budget_mb: float,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    # Adapted from TensorRT code
    experts_num = getattr(model.config, "num_experts", 0)
    layers_mapping = get_layer_sensitivity_group_mapping(experts_num)
    candidate_stats = defaultdict(dict)

    total_weight_size = 0
    for layer_name in sensitivity_scores:
        total_weight_size += sensitivity_scores["size"]

    weight_size_after_compression = (
        total_weight_size * BYTES_PER_4BIT_PARAM + ckpt_size_budget_mb * MEGABYTE_SIZE
    )

    seen_layers = set()
    for name, mapping in layers_mapping.items():
        if not mapping:
            continue

        for layer_name, _ in sensitivity_scores.items():
            if mapping[0] in layer_name and layer_name not in seen_layers:
                layer_type = layer_name.replace(mapping[0], name)
                subset_names = get_subset_layer_names(layer_type, layers_mapping)
                for subset_name in subset_names:
                    seen_layers.add(subset_name)

                formats, scores, costs = [], [], []
                for recipe in [4, 8]:
                    formats.append(recipe)
                    cost = 0
                    score = 0
                    for subset_layer in mapping:
                        subset_name = layer_name.replace(mapping[0], subset_layer)
                        weight = sensitivity_scores[subset_layer]["size"]
                        cost += weight.numel() * recipe / 8  # TODO: refactor
                        score += sensitivity_scores[subset_name][recipe][score_name]

                    scores.append(score)
                    costs.append(cost)

                candidate_stats[layer_type]["formats"] = formats
                candidate_stats[layer_type]["scores"] = scores
                candidate_stats[layer_type]["costs"] = costs

    for lower_bound in [None, 0.99, 0.90]:
        # The LP solver for auto_quantize sometimes fails to find a solution if
        # a lower bound is not specified. I dont know why this happens.
        # As a workaround, lets specify a lower bound for the weight compression
        # if previous search without lower bound fails.
        constraints, constraint_name = get_constraints_for_search(
            weight_size_after_compression, lower_bound
        )

        lps = LPS(
            name="AutoQuantize",
            constraints=constraints,
            constraints_to_candidate_costs={
                constraint_name: [
                    candidate_stat["costs"]
                    for candidate_stat in candidate_stats.values()
                ]
            },
            candidate_scores=[
                candidate_stat["scores"] for candidate_stat in candidate_stats.values()
            ],
            objective_type="minimize",
            verbose=verbose,
        )
        selections, status = lps()
        if status == "Optimal":
            break

    best = {}
    if status != "Optimal":
        warnings.warn(
            "AutoQuantize FAILED to find a solution! "
            "The searched model might not meet all constraints."
        )
        best["is_satisfied"] = False
    else:
        best["is_satisfied"] = True

    best_recipe = {}
    best_constraints, best_scores = 0, 0
    for name, selected_idx in zip(candidate_stats.keys(), selections):
        best_recipe_for_name = candidate_stats[name]["formats"][selected_idx]
        best_constraints += candidate_stats[name]["costs"][selected_idx]
        best_scores += candidate_stats[name]["scores"][selected_idx]
        layer_names = get_subset_layer_names(name, layers_mapping)
        for layer_name in layer_names:
            best_recipe[layer_name] = best_recipe_for_name
            if verbose:
                logger.info(
                    f"AutoQuantize best recipe for {layer_name}: "
                    f"{best_recipe[layer_name]}"
                )

    effective_bits_from_search = (best_constraints / total_weight_size) * 16
    if verbose:
        logger.info(
            "AutoQuantize effective bits from search: "
            f"{effective_bits_from_search: .2f}"
        )

    return best_recipe


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
