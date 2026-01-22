import argparse
import json
from argparse import Namespace
from collections import defaultdict
from typing import Any, Dict, Union

import yaml
from msmodelslim.model.deepseek_v3_2.model import ModelArgs
from msmodelslim.model.deepseek_v3_2.model_adapter import DeepSeekV32ModelAdapter
from torch import nn
from transformers import AutoModelForCausalLM

from utils.sensitivity import get_layer_sensitivity_group_mapping, get_subset_layer_names
from utils.quant_config_manager import (
    TRANSFORMER_LAYER_PATTERNS,
    QuantLayerConfig,
    QuantLayerConfigManager,
    compress_hybrid_quant_schema,
)


class FlowStyleList(list):
    pass


def flow_style_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(FlowStyleList, flow_style_representer)


MEGABYTE_SIZE = 1024**2
BYTES_PER_4BIT_PARAM = 0.5
BYTES_PER_8BIT_PARAM = 1

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()

    # default
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument(
        "--is-deepseek-v3_2",
        action="store_true",
        help="Whether the model is in the DeepSeek-V3.2 family.",
    )
    parser.add_argument(
        "--weight-quant-bits",
        type=int,
        default=4,
        choices=[4, 8, 16],
        help=(
            "The number of bits to use for weight quantization. "
            "Use 16 for evaluating base model."
        ),
    )
    parser.add_argument(
        "--act-quant-bits",
        type=int,
        default=16,
        choices=[4, 8, 16],
        help=(
            "The number of bits to use for act quantization. "
            "Use 16 for evaluating base model."
        ),
    )
    parser.add_argument(
        "--quant-group-size",
        type=int,
        default=0,
        help="Groups size in per-group scenario.",
    )
    parser.add_argument(
        "--hybrid-quant",
        action="store_true",
        help="Whether to apply hybrid quantization",
    )
    parser.add_argument(
        "--sensitivity-metric",
        default=None,
        help="Metric to use for computing sensitivity scores",
    )
    parser.add_argument(
        "--ckpt-size-budget-mb",
        type=int,
        default=2500,
        help="Checkpoint size budget for hybrid quantization.",
    )
    parser.add_argument(
        "--sensitivity-scores-path",
        type=str,
        help="Path to the sensitivity scores file.",
    )
    parser.add_argument(
        "--template-path", type=str, help="Path to Modeslim template file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to Modeslim .yaml file with prepared hybrid configuration.",
    )

    args = parser.parse_args()

    return args

def prepare_modelslim_config() -> None:
    args = parse_args()
    
    if args.is_deepseek_v3_2:
        model_type = "DeepSeek-V3.2-Exp"
        adapter = DeepSeekV32ModelAdapter(
            model_path=args.model_name_or_path, model_type=model_type
        )
        model = adapter.init_model(device="cpu")
        model.config = adapter._load_config()
        model.config.num_experts = model.config.n_routed_experts
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
            local_files_only=True,
        )
    quant_layer_cfg_mngr = QuantLayerConfigManager(args=args, model=model)

    with open(args.sensitivity_scores_path) as f:
        sensitivity_scores = json.load(f)

    update_quant_layer_cfg_modelsim(
        sensitivity_scores=sensitivity_scores,
        model=model,
        args=args,
        quant_layer_cfg_mngr=quant_layer_cfg_mngr,
        score_name=args.sensitivity_metric.split(".")[0],
        ckpt_size_budget_mb=args.ckpt_size_budget_mb,
    )
    layers_quant_mapping = quant_layer_cfg_mngr._create_quant_layers_mapping(
        overwrite_act_to_8bit=False
    )
    hybrid_quant_schema = compress_hybrid_quant_schema(
        cfg=layers_quant_mapping, experts_num=quant_layer_cfg_mngr.experts_num
    )
    fill_modelslim_yaml_template(
        hybrid_quant_schema=hybrid_quant_schema,
        template_path=args.template_path,
        output_path=args.output_path,
    )


def update_quant_layer_cfg_modelsim(
    sensitivity_scores: Dict[str, Dict[int, Any]],
    model: nn.Module,
    args: Namespace,
    quant_layer_cfg_mngr: QuantLayerConfigManager,
    score_name: str,
    ckpt_size_budget_mb: int = 500,
) -> None:
    experts_num = getattr(model.config, "num_experts", 0)
    layers_mapping = get_layer_sensitivity_group_mapping(experts_num)
    layer_score_info = []
    seen_layers = set()
    for name, bit_mapping in sensitivity_scores.items():
        if name not in quant_layer_cfg_mngr.cfg:
            quant_layer_cfg_mngr.cfg[name] = QuantLayerConfig(
                weight_bits=args.weight_quant_bits,
                act_bits=args.act_quant_bits,
                group_size=args.quant_group_size,
            )

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
    skipped = []
    while curr_ckpt_diff < ckpt_size_budget_mb and layer_num < len(layer_score_info):
        subset_name = layer_score_info[layer_num][1]
        bit_mapping_cfg = _get_lower_upper_bit_type(subset_name)

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

        layer_num += 1

    for subset_name in skipped:
        layer_names = get_subset_layer_names(subset_name, layers_mapping)
        bit_mapping_cfg = _get_lower_upper_bit_type(subset_name)
        for layer_name in layer_names:
            quant_layer_cfg_mngr.cfg[layer_name].weight_bits = bit_mapping_cfg["lower"]


def _get_lower_upper_bit_type(subset_name: str) -> Dict[str, Union[int, float]]:
    if "experts" in subset_name:
        return {"lower": 4, "upper": 8, "bytes_per_param": 0.5}
    else:
        return {"lower": 8, "upper": 16, "bytes_per_param": 1}


def fill_modelslim_yaml_template(
    hybrid_quant_schema: Dict[str, str],
    template_path: str,
    output_path: str,
) -> None:
    w4a8_cfg = defaultdict(FlowStyleList)
    w8a8_cfg = defaultdict(FlowStyleList)

    for pattern, quant_schema in hybrid_quant_schema.items():
        if pattern in TRANSFORMER_LAYER_PATTERNS:
            if quant_schema.startswith("w4"):
                w4a8_cfg["include"].append(pattern)
            elif quant_schema.startswith("w8"):
                w8a8_cfg["include"].append(pattern)
        else:
            if quant_schema.startswith("w4"):
                w4a8_cfg["include"].append(pattern)
            elif quant_schema.startswith("w8"):
                w4a8_cfg["exclude"].append(pattern)
                w8a8_cfg["include"].append(pattern)
            else:
                w8a8_cfg[""].append(pattern)

    with open(template_path) as f:
        yaml_template_dict = yaml.safe_load(f)

    anchor_mapping = {}
    for process in yaml_template_dict["spec"]["process"]:
        if process["type"] == "group":
            for config in process["configs"]:
                if config["qconfig"]["weight"]["dtype"] == "int4":
                    config.update(w4a8_cfg)
                    anchor_mapping["default_w8a8_dynamic"] = id(config)
                elif config["qconfig"]["weight"]["dtype"] == "int8":
                    config.update(w8a8_cfg)
                    anchor_mapping["default_w4a8_dynamic"] = id(config)

    with open(output_path, "w") as f:
        yaml.dump(yaml_template_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    prepare_modelslim_config()