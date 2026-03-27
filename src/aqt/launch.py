import json
import argparse
from ..utils.logger import logger
import os

from .utils.quant_config_manager import (
    QuantLayerConfigManager,
    compress_hybrid_quant_schema,
)


def update_quant_layer_cfg(
    quant_layer_cfg_mngr: QuantLayerConfigManager,
    ckpt_size_budget_mb: int,
    sensitivity_scores: dict,
) -> None:
    layer_score_info = []
    for name, mapping in sensitivity_scores.items():
        score = mapping.get("gold", 0.0)
        layer_score_info.append((score, name))
    layer_score_info.sort(reverse=True, key=lambda x: x[0])

    curr_ckpt_diff = 0
    layer_num = 0
    while curr_ckpt_diff < ckpt_size_budget_mb and layer_num < len(layer_score_info):
        subset_name = layer_score_info[layer_num][1]
        layer_num += 1

        subset_data = sensitivity_scores.get(subset_name, {})
        if "mlp.experts" in subset_name:
            weight_size = subset_data.get("4-bit", {}).get("size", 0)
        else:
            weight_size = subset_data.get("8-bit", {}).get("size", 0)

        if curr_ckpt_diff + weight_size <= ckpt_size_budget_mb:
            curr_ckpt_diff += weight_size
            if subset_name not in quant_layer_cfg_mngr.cfg:
                continue
            if "mlp.experts" in subset_name:
                quant_layer_cfg_mngr.cfg[subset_name].weight_bits = 8
            else:
                quant_layer_cfg_mngr.cfg[subset_name].weight_bits = 16
                quant_layer_cfg_mngr.cfg[subset_name].act_bits = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True, type=str)
    parser.add_argument("--ckpt-size-budget-mb", required=True, type=int)
    parser.add_argument("--hybrid_quant_schema_path", required=True, type=str)
    parser.add_argument("--hybrid_quant_schema_re_path", required=True, type=str)
    parser.add_argument("--sensitivity_scores_save_path", required=True, type=str)
    parser.add_argument("--is-mm", action="store_true")

    args = parser.parse_args()

    with open(args.sensitivity_scores_save_path, "r", encoding="utf-8") as f:
        sensitivity_scores: dict = json.load(f)

    config_path = os.path.join(args.model_name_or_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config, dict) and "text_config" in config:
        config = config["text_config"]
    num_experts = config.get("num_experts", config.get("n_routed_experts", 0))
    num_layers = config.get("num_hidden_layers")

    quant_layer_cfg_mngr = QuantLayerConfigManager(layer_names=sensitivity_scores.keys(), num_experts=num_experts, num_layers=num_layers)

    update_quant_layer_cfg(
        quant_layer_cfg_mngr=quant_layer_cfg_mngr,
        ckpt_size_budget_mb=args.ckpt_size_budget_mb,
        sensitivity_scores=sensitivity_scores,
    )

    layers_quant_mapping = quant_layer_cfg_mngr._create_quant_layers_mapping()

    hybrid_quant_schema, hybrid_quant_schema_re = compress_hybrid_quant_schema(layers_quant_mapping, is_mm=args.is_mm)

    logger.info("Saving hybrid quant config...")
    with open(args.hybrid_quant_schema_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_quant_schema, f, indent=4)
    with open(args.hybrid_quant_schema_re_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_quant_schema_re, f, indent=4)


if __name__ == "__main__":
    main()
