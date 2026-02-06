import json
import argparse
from utils.logger import logger

import torch
import torch_npu
from torch import nn

from transformers import AutoModelForCausalLM

from aqt.utils.quant_config_manager import (
    QuantLayerConfigManager,
    compress_hybrid_quant_schema,
)
from aqt.sensitivity import (
    get_layer_sensitivity_group_mapping,
    get_subset_layer_names,
)


MEGABYTE_SIZE = 1024**2


def update_quant_layer_cfg(
    model: nn.Module,
    quant_layer_cfg_mngr: QuantLayerConfigManager,
    score_name: str,
    ckpt_size_budget_mb: int,
    sensitivity_scores_save_path: str,
) -> None:
    with open(sensitivity_scores_save_path) as f:
        sensitivity_scores = json.load(f)
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
    bit_mapping_cfg = {"lower": 4, "upper": 8, "bytes_per_param": 0.5}

    ckpt_size_budget_mb = ckpt_size_budget_mb * MEGABYTE_SIZE
    skipped = []
    while curr_ckpt_diff < ckpt_size_budget_mb and layer_num < len(layer_score_info):
        subset_name = layer_score_info[layer_num][1]
        layer_num += 1
        if ".experts" not in subset_name:
            continue

        weight_size = 0
        layer_names = get_subset_layer_names(subset_name, layers_mapping)
        for layer_name in layer_names:
            n_elements = sensitivity_scores[layer_name]["size"]
            weight_size += n_elements * bit_mapping_cfg["bytes_per_param"]

        if curr_ckpt_diff + weight_size <= ckpt_size_budget_mb:
            curr_ckpt_diff += weight_size
            for layer_name in layer_names:
                quant_layer_cfg_mngr.cfg[layer_name].weight_bits = bit_mapping_cfg[
                    "upper"
                ]
        else:
            skipped.append(subset_name)

    for subset_name in skipped:
        layer_names = get_subset_layer_names(subset_name, layers_mapping)
        for layer_name in layer_names:
            quant_layer_cfg_mngr.cfg[layer_name].weight_bits = bit_mapping_cfg["lower"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name-or-path', required=True, type=str)
    parser.add_argument('--sensitivity-metric', required=True, type=str)
    parser.add_argument('--ckpt-size-budget-mb', required=True, type=int)
    parser.add_argument('--hybrid_quant_schema_path', required=True, type=str)
    parser.add_argument('--hybrid_quant_schema_re_path', required=True, type=str)
    parser.add_argument('--last_hybrid_quant_schema_path', required=True, type=str)
    parser.add_argument('--sensitivity_scores_save_path', required=True, type=str)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True,
    )

    quant_layer_cfg_mngr = QuantLayerConfigManager(model=model, last_hybrid_quant_schema_path=args.last_hybrid_quant_schema_path)

    if args.last_hybrid_quant_schema_path != "":
        logger.info(f"last_hybrid_quant_schema_path: {args.last_hybrid_quant_schema_path}")
        update_quant_layer_cfg(
            model=model,
            quant_layer_cfg_mngr=quant_layer_cfg_mngr,
            score_name=args.sensitivity_metric,
            ckpt_size_budget_mb=args.ckpt_size_budget_mb,
            sensitivity_scores_save_path=args.sensitivity_scores_save_path,
        )

    layers_quant_mapping = quant_layer_cfg_mngr._create_quant_layers_mapping(
        overwrite_act_to_8bit=False
    )
    hybrid_quant_schema, hybrid_quant_schema_re = compress_hybrid_quant_schema(
        cfg=layers_quant_mapping, experts_num=quant_layer_cfg_mngr.experts_num, layers_num=quant_layer_cfg_mngr.layers_num
    )
    
    logger.info("Saving hybrid quant config...")
    with open(args.hybrid_quant_schema_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_quant_schema, f, indent=4)
    with open(args.hybrid_quant_schema_re_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_quant_schema, f, indent=4)


if __name__ == "__main__":
    main()
