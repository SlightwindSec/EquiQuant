from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from ..utils.logger import logger

import json
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


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


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_sensitivity_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_layer_info(layer_name):
    if "self_attn.qkv_proj" in layer_name:
        return int(layer_name.split(".")[2]), "attn_qkv_proj"
    elif "self_attn.o_proj" in layer_name:
        return int(layer_name.split(".")[2]), "attn_o_proj"
    elif "mlp.experts" in layer_name:
        return int(layer_name.split(".")[2]), "mlp_experts"
    else:
        return -1, "other"


def get_save_dir(json_path):
    save_dir = Path(json_path).parent / "plots"
    save_dir.mkdir(exist_ok=True)
    return save_dir


# ------------------------------------------------------------------------------
# 1. 绘制 Gold Score（分模块）
# ------------------------------------------------------------------------------
def plot_gold_by_module(json_path, total_layers):
    data = load_sensitivity_data(json_path)
    save_dir = get_save_dir(json_path)
    records = []

    for name, info in data.items():
        layer_idx, module = parse_layer_info(name)
        if module == "other":
            continue
        records.append({
            "layer": layer_idx,
            "module": module,
            "gold": info.get("gold", 0)
        })

    df = pd.DataFrame(records)
    for module in ["attn_qkv_proj", "attn_o_proj", "mlp_experts"]:
        sub = df[df["module"] == module]
        if sub.empty:
            continue

        plt.figure(figsize=(12, 5))
        sns.barplot(x="layer", y="gold", data=sub, color="#ff9999", width=0.6)
        plt.title(f"Gold Score - {module}", fontsize=14)
        plt.xlabel("Layer")
        plt.ylabel("Gold (Higher = More Sensitive)")
        plt.tight_layout()
        plt.savefig(save_dir / f"gold_{module}.png", dpi=300)
        plt.close()
    print("✅ Gold 分数绘制完成")


# ------------------------------------------------------------------------------
# 2. 8-bit 所有指标 → 动态检测！有什么画什么
# ------------------------------------------------------------------------------
def plot_8bit_all_metrics_dynamic(json_path, total_layers):
    data = load_sensitivity_data(json_path)
    save_dir = get_save_dir(json_path)

    # 自动收集所有出现过的 metrics（mse 一定有）
    metric_set = set()
    for name, info in data.items():
        if "8-bit" in info and "metrics" in info["8-bit"]:
            metric_set.update(info["8-bit"]["metrics"].keys())
    metric_list = sorted(list(metric_set))
    print(f"📊 检测到 8-bit 指标：{metric_list}")

    records = []
    for name, info in data.items():
        layer_idx, module = parse_layer_info(name)
        if module == "other":
            continue
        metrics = info.get("8-bit", {}).get("metrics", {})
        row = {"layer": layer_idx, "module": module}
        for m in metric_list:
            row[m] = metrics.get(m, np.nan)
        records.append(row)

    df = pd.DataFrame(records)
    for module in ["attn_qkv_proj", "attn_o_proj", "mlp_experts"]:
        sub = df[df["module"] == module]
        if sub.empty:
            continue

        for metric in metric_list:
            plt.figure(figsize=(12, 5))
            sns.barplot(x="layer", y=metric, data=sub, width=0.6, palette="viridis")
            plt.title(f"8-bit {metric.upper()} - {module}", fontsize=14)
            plt.xlabel("Layer")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(save_dir / f"8bit_{metric}_{module}.png", dpi=300)
            plt.close()
    print("✅ 8-bit 动态指标绘制完成")


# ------------------------------------------------------------------------------
# 3. 专家层 4bit vs 8bit → 同样动态检测！有什么画什么
# ------------------------------------------------------------------------------
def plot_expert_4vs8_dynamic(json_path, total_layers):
    data = load_sensitivity_data(json_path)
    save_dir = get_save_dir(json_path)
    records = []
    metric_set = set()

    # 先收集所有出现过的指标
    for name, info in data.items():
        if "mlp.experts" not in name:
            continue
        if "4-bit" in info and "metrics" in info["4-bit"]:
            metric_set.update(info["4-bit"]["metrics"].keys())
        if "8-bit" in info and "metrics" in info["8-bit"]:
            metric_set.update(info["8-bit"]["metrics"].keys())

    metric_list = sorted(list(metric_set))
    if not metric_list:
        print("⚠️ 未检测到专家层指标")
        return
    print(f"📊 检测到专家层指标：{metric_list}")

    # 构建数据
    for name, info in data.items():
        if "mlp.experts" not in name:
            continue
        layer_idx, _ = parse_layer_info(name)
        m4 = info.get("4-bit", {}).get("metrics", {})
        m8 = info.get("8-bit", {}).get("metrics", {})
        for metric in metric_list:
            records.append({
                "layer": layer_idx,
                "metric": metric,
                "4bit": m4.get(metric, np.nan),
                "8bit": m8.get(metric, np.nan),
            })

    df = pd.DataFrame(records)
    for metric in metric_list:
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(12, 5))
        xs = np.arange(len(sub))
        width = 0.35
        plt.bar(xs - width/2, sub["4bit"], width, label="4bit")
        plt.bar(xs + width/2, sub["8bit"], width, label="8bit")
        plt.title(f"MLP Experts 4bit vs 8bit - {metric}", fontsize=14)
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.xticks(xs, sub["layer"])
        plt.legend()

        # 除了 snr_db 其他都用对数坐标更清晰
        if metric not in ["snr_db"]:
            plt.yscale("log")

        plt.tight_layout()
        plt.savefig(save_dir / f"expert_4vs8_{metric}.png", dpi=300)
        plt.close()
    print("✅ 专家层 4bit vs 8bit 动态绘制完成")


# ------------------------------------------------------------------------------
# 一键绘制所有
# ------------------------------------------------------------------------------
def plot_all_visuals(json_path, total_layers):
    print("\n======================================")
    print("📊 开始生成敏感度可视化图表（动态版）")
    print("======================================")
    plot_gold_by_module(json_path, total_layers)
    plot_8bit_all_metrics_dynamic(json_path, total_layers)
    plot_expert_4vs8_dynamic(json_path, total_layers)
    print("\n🎉 所有图表已生成完毕！")
