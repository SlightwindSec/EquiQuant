import fnmatch
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
from ...utils.logger import logger

from ..sensitivity import (
    get_layer_sensitivity_group_mapping,
    get_subset_layer_names,
)


@dataclass
class QuantLayerConfig:
    weight_bits: int
    act_bits: int
    group_size: int


class QuantLayerConfigManager:
    def __init__(
        self,
        layer_names: List[str], 
        num_experts: int,
        num_layers: int,
    ) -> None:
        # 补充 skip 的 layer name
        self.skips = {"embed_tokens", "mlp.gate.", "lm_head", "mlp.shared_expert_gate", "model.norm", "indexer"}
        self.layer_names = layer_names
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.cfg: Dict[str, QuantLayerConfig] = self._process_hybrid_quant_config()

    def _process_hybrid_quant_config(self) -> Dict[str, QuantLayerConfig]:
        cfg = {}
        for name in self.layer_names:
            self.update_hybrid_quant_config(name=name, cfg=cfg)
        return cfg

    def update_hybrid_quant_config(
        self,
        name: str,
        cfg: Dict[str, QuantLayerConfig],
    ) -> None:
        if any([i in name for i in self.skips]):
            weight_bits = 16
            act_bits = 16
            group_size = 0
        else:
            weight_bits = 4 if "mlp.experts" in name else 8
            act_bits = 8
            group_size = 0

        cfg[name] = QuantLayerConfig(
            weight_bits=weight_bits,
            act_bits=act_bits,
            group_size=group_size,
        )

        self._validate_group_size_for_layer(name=name, quant_layer_cfg=cfg[name])

    def _create_quant_layers_mapping(
        self, overwrite_act_to_8bit: bool = False
    ) -> Dict[str, List[str]]:
        layers_quant_mapping = {}
        for layer_name, layer_cfg in self.cfg.items():
            if layer_cfg.weight_bits == 16 and layer_cfg.act_bits == 16:
                layers_quant_mapping[layer_name] = "float"
            elif layer_cfg.weight_bits == 8 and layer_cfg.act_bits == 16:
                if overwrite_act_to_8bit:
                    layers_quant_mapping[layer_name] = "w8a8_dynamic"
                else:
                    layers_quant_mapping[layer_name] = "w8a16"
            elif layer_cfg.weight_bits == 4 and layer_cfg.act_bits == 16:
                if overwrite_act_to_8bit:
                    layers_quant_mapping[layer_name] = "w4a8_dynamic"
                else:
                    layers_quant_mapping[layer_name] = "w4a16"
            elif layer_cfg.weight_bits == 8 and layer_cfg.act_bits == 8:
                layers_quant_mapping[layer_name] = "w8a8_dynamic"
            elif layer_cfg.weight_bits == 4 and layer_cfg.act_bits == 8:
                if layer_cfg.group_size == 0:
                    layers_quant_mapping[layer_name] = "w4a8_dynamic_perchannel"
                else:
                    layers_quant_mapping[layer_name] = "w4a8_dynamic_pergroup"
            elif layer_cfg.weight_bits == 4 and layer_cfg.act_bits == 4:
                layers_quant_mapping[layer_name] = "w4a4_flatquant_dynamic"
            else:
                raise NotImplementedError

        return layers_quant_mapping

    def _validate_group_size_for_layer(
        self,
        name: str,
        quant_layer_cfg: QuantLayerConfig,
    ) -> None:
        weight_bits = quant_layer_cfg.weight_bits
        act_bits = quant_layer_cfg.act_bits
        group_size = quant_layer_cfg.group_size

        if weight_bits == 8 and act_bits == 8:
            if group_size != 0:
                logger.info(
                    "WARNING! Currently, there is no per-group support for w8a8 kernel. "
                    f"Use group_size = 0 for layer '{name}' instead of {group_size}."
                )
            quant_layer_cfg.group_size = 0


def _get_re_format(layer_names: List[str]) -> List[str]:
    if not layer_names:
        return []

    numbers = []
    first_name = layer_names[0]
    pattern = r"(.*?layers\.)(\d+)(\..*)"
    match = re.match(pattern, first_name)
    
    if not match:
        return []
    
    prefix = match.group(1)
    suffix = match.group(3)

    for name in layer_names:
        num_match = re.match(pattern, name)
        if num_match:
            numbers.append(num_match.group(2))

    num_str = "|".join(numbers)
    result = f"re:.*{prefix}({num_str}){suffix}.*"

    if "mlp.experts" in result:
        results = [result + proj for proj in [".gate_proj.*", ".up_proj.*", ".down_proj.*"]]
    else:
        results = [result]
    return results


def _sort_mapping(mapping: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, List[str]]]:
    sorted_mapping = {}
    for outer_key, inner_dict in mapping.items():
        list_sorted = {k: sorted(v) for k, v in inner_dict.items()}
        sorted_pairs = sorted(list_sorted.items(), key=lambda x: len(x[1]))
        sorted_inner = dict(sorted_pairs)
        sorted_mapping[outer_key] = sorted_inner
    return sorted_mapping


def compress_hybrid_quant_schema(cfg: Dict[str, str], is_mm: bool = False) -> Tuple[Dict[str, str], Dict[str, str]]:
    mapping = {}
    layer_mapping = get_layer_sensitivity_group_mapping()
    for layer_name, quant_schema in cfg.items():
        subset_names = get_subset_layer_names(layer_name, layer_mapping)
        for subset_name in subset_names:
            for pattern in TRANSFORMER_LAYER_PATTERNS:
                pattern_ = pattern.replace("model.layers", "model.language_model.layers") if is_mm else pattern
                if fnmatch.fnmatchcase(subset_name, pattern_):
                    if pattern_ == "*":
                        continue
                    if pattern_ not in mapping:
                        mapping[pattern_] = {}
                    if quant_schema not in mapping[pattern_]:
                        mapping[pattern_][quant_schema] = []
                    mapping[pattern_][quant_schema].append(subset_name)
                    break
            else:
                raise ValueError(f"Layer {subset_name} not found in any pattern")

    mapping = _sort_mapping(mapping)
    quant_schemas_all = set(cfg.values())

    output = {k: {"include": [], "exclude": []} for k in quant_schemas_all}
    for pattern, pattern_mapping in mapping.items():
        if not pattern_mapping:
            continue
        elif len(pattern_mapping) == 1:
            quant_schema = list(pattern_mapping.keys())[0]
            output[quant_schema]["include"].append("*" + pattern + "*")
        else:
            quant_schemas = list(pattern_mapping.keys())
            for quant_schema in quant_schemas[:-1]:
                for layer_name in pattern_mapping[quant_schema]:
                    output[quant_schema]["include"].append("*" + layer_name + "*")
                    output[quant_schemas[-1]]["exclude"].append("*" + layer_name + "*")
            output[quant_schemas[-1]]["include"].append("*" + pattern + "*")

    output_re = {k: [] for k in quant_schemas_all}
    for quant_schema in quant_schemas_all:
        for pattern in mapping.keys():
            if quant_schema in mapping[pattern]:
                for re_names in _get_re_format(mapping[pattern][quant_schema]):
                    output_re[quant_schema].append(re_names)
    
    if "float" in quant_schemas_all:
        del output["float"]
        del output_re["float"]
    return output, output_re


TRANSFORMER_LAYER_PATTERNS = [
    # self attn
    "model.layers.*.self_attn.q_proj",
    "model.layers.*.self_attn.q_a_proj",
    "model.layers.*.self_attn.q_b_proj",
    "model.layers.*.self_attn.k_proj",
    "model.layers.*.self_attn.v_proj",
    "model.layers.*.self_attn.kv_a_proj_with_mqa",
    # "model.layers.*.self_attn.kv_b_proj",
    "model.layers.*.self_attn.o_proj",
    # linear attn
    "model.layers.*.linear_attn.in_proj_qkvz",
    "model.layers.*.linear_attn.in_proj_ba",
    "model.layers.*.linear_attn.in_proj_qkv",
    "model.layers.*.linear_attn.in_proj_z",
    "model.layers.*.linear_attn.in_proj_b",
    "model.layers.*.linear_attn.in_proj_a",
    "model.layers.*.linear_attn.out_proj",

    # mlp
    "model.layers.*.mlp.shared_expert.up_proj",
    "model.layers.*.mlp.shared_expert.gate_proj",
    "model.layers.*.mlp.shared_expert.down_proj",
    "model.layers.*.mlp.shared_experts.up_proj",
    "model.layers.*.mlp.shared_experts.gate_proj",
    "model.layers.*.mlp.shared_experts.down_proj",
    "model.layers.*.mlp.experts",
    # "model.layers.*.mlp.experts.*.up_proj",
    # "model.layers.*.mlp.experts.*.gate_proj",
    # "model.layers.*.mlp.experts.*.down_proj",
    "model.layers.*.mlp.up_proj",
    "model.layers.*.mlp.gate_proj",
    "model.layers.*.mlp.down_proj",
    # rest
    "*",
]