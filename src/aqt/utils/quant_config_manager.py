import fnmatch
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
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
    act_scope: str = "per_token"


class QuantLayerConfigManager:
    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        sensitivity_scores: Dict[str, Any] = None,
        layer_configs: Dict[str, Any] = None,
    ) -> None:
        # 补充 skip 的 layer name
        self.skips = {"embed_tokens", "mlp.gate.", "lm_head", "mlp.shared_expert_gate", "model.norm", "indexer"}
        self.layer_names = sensitivity_scores.keys()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.sensitivity_scores = sensitivity_scores
        self.layer_configs = layer_configs or {}
        self.cfg: Dict[str, QuantLayerConfig] = self._init_layer_quant_configs()

    def _init_layer_quant_configs(self) -> Dict[str, QuantLayerConfig]:
        """
        Initialize the quantization configuration for all layers.
        """
        cfg = {}
        for name in self.layer_names:
            cfg[name] = self._determine_layer_config(name)
        return cfg

    def _determine_layer_config(self, name: str) -> QuantLayerConfig:
        """
        Determine the specific weight/activation configuration for a single layer 
        based on priority: Skips > Existing State > Sensitivity Heuristic > Default.
        """
        # 1. Skip layers (Fixed Float16/BFloat16)
        if any([i in name for i in self.skips]):
            return QuantLayerConfig(weight_bits=16, act_bits=16, group_size=0, act_scope="per_token")
        
        # 2. Stateful Upgrade (Loading existing configuration from a previous trial)
        if name in self.layer_configs:
            state = self.layer_configs[name]
            w_bits = state["weight_bits"]
            return QuantLayerConfig(
                weight_bits=w_bits,
                act_bits=8 if w_bits in (4, 8) else 16,
                group_size=0,
                act_scope=state["act_scope"]
            )

        # 3. Efficiency Heuristic (First Trial: Choosing between W4 and W8 based on MSE/Size)
        if name in self.sensitivity_scores:
            data = self.sensitivity_scores[name]
            score4 = data.get("score4")
            score8 = data.get("score8")
            # TODO: how to select the initial config
            if score4 < score8:
                return QuantLayerConfig(weight_bits=4, act_bits=8, group_size=0, act_scope="per_token")
            else:
                return QuantLayerConfig(weight_bits=8, act_bits=8, group_size=0, act_scope="per_token")
            
        # 4. Global Default (Standard W8 Dynamic)
        return QuantLayerConfig(weight_bits=8, act_bits=8, group_size=0, act_scope="per_token")

    def create_quant_layers_mapping(self) -> Dict[str, str]:
        layers_quant_mapping = {}
        for layer_name, layer_cfg in self.cfg.items():
            if layer_cfg.weight_bits == 16:
                layers_quant_mapping[layer_name] = "float"
            elif layer_cfg.weight_bits == 8:
                if layer_cfg.act_scope == "per_token":
                    layers_quant_mapping[layer_name] = "w8a8_dynamic"
                else:
                    layers_quant_mapping[layer_name] = "w8a8_default"
            elif layer_cfg.weight_bits == 4:
                layers_quant_mapping[layer_name] = "w4a8_dynamic"
            else:
                raise NotImplementedError

        return layers_quant_mapping


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