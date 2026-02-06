import fnmatch
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
from utils.logger import logger

from torch import nn


@dataclass
class QuantLayerConfig:
    weight_bits: int
    act_bits: int
    group_size: int


class QuantLayerConfigManager:
    def __init__(
        self,
        model: nn.Module,
        last_hybrid_quant_schema_path: str,
    ) -> None:
        # "*mlp.gate" is skipped in MoE architecture, carefull with
        # "gate_proj" in self attention, don't add "*" at the end of the pattern
        self.skip_layers = ["*embed_tokens", "*mlp.gate", "*lm_head", "*indexer*"]
        self.experts_num = getattr(model.config, "num_experts", 0)
        self.layers_num = len(model.model.layers)
        self.last_hybrid_quant_schema_path = last_hybrid_quant_schema_path
        self.cfg: Dict[str, QuantLayerConfig] = self._process_hybrid_quant_config(model=model)

    def _process_hybrid_quant_config(
        self,
        model: nn.Module,
    ) -> Dict[str, QuantLayerConfig]:
        cfg = {}
        for name, module in model.named_modules():
            self.update_hybrid_quant_config(
                name=name, module=module, cfg=cfg
            )

        return cfg

    def update_hybrid_quant_config(
        self,
        name: str,
        module: nn.Module,
        cfg: Dict[str, QuantLayerConfig],
    ) -> None:
        pattern_cfg = self._load_hybrid_quant_config()

        if not isinstance(module, nn.Linear) or name in cfg:
            return

        for pattern, rule in pattern_cfg.items():
            if fnmatch.fnmatchcase(name=name, pat=pattern):
                cfg[name] = _extract_quant_layer_cfg(
                    rule=rule, name=name, default_group_size=0
                )
                break
        else:
            if self._check_skip_layer(name):
                weight_bits = 16
                act_bits = 16
                group_size = 0
            else:
                weight_bits = 4 if ".mlp.experts." in name else 8
                act_bits = 8
                group_size = 0

            cfg[name] = QuantLayerConfig(
                weight_bits=weight_bits,
                act_bits=act_bits,
                group_size=group_size,
            )

        _validate_group_size_for_layer(name=name, quant_layer_cfg=cfg[name])

    def _check_skip_layer(self, name: str) -> bool:
        return any(
            fnmatch.fnmatchcase(name=name, pat=pattern) for pattern in self.skip_layers
        )

    def get_group_size(self, name: str) -> int:
        return self.cfg[name].group_size
    
    def save_hybrid_quant_cfg(
        self, save_path: str, overwrite_act_to_8bit: bool = False
    ) -> None:
        layers_quant_mapping = self._create_quant_layers_mapping(overwrite_act_to_8bit)
        output = compress_hybrid_quant_schema(
            cfg=layers_quant_mapping, experts_num=self.experts_num, layers_num=self.layers_num,
        )
        with open(save_path, "w", encoding="utf-8") as f:
            logger.info("Saving hybrid quant config...")
            json.dump(output, f, indent=4)

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

    def _load_hybrid_quant_config(self) -> Dict[str, str]:
        config = {}
        if self.last_hybrid_quant_schema_path != "":
            with open(self.last_hybrid_quant_schema_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        return config


def _extract_quant_layer_cfg(
    rule: str,
    name: str,
    default_group_size: int
) -> QuantLayerConfig:
    rule = rule.lower()
    if rule == "float":
        return QuantLayerConfig(
            weight_bits=16,
            act_bits=16,
            group_size=0,
        )        

    weight_match = re.search(r"w\d*", rule)
    if weight_match:
        weight_bits = int(weight_match.group(0)[1:])
    else:
        raise ValueError(
            f"Can't extract weight bits from layer '{name}' "
            f"having following rule: '{rule}'"
        )

    act_match = re.search(r"a\d*", rule)
    if act_match:
        act_bits = int(act_match.group(0)[1:])
    else:
        raise ValueError(
            f"Can't extract weight bits from layer '{name}' "
            f"having following rule: '{rule}'"
        )

    group_size_match = re.search(r"gs\d*", rule)
    group_size = (
        int(group_size_match.group(0)[2:]) if group_size_match else default_group_size
    )

    return QuantLayerConfig(
        weight_bits=weight_bits,
        act_bits=act_bits,
        group_size=group_size,
    )


def _validate_group_size_for_layer(
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


def _get_hybrid_quant_schema_re(original_dict: Dict[str, str], total_layers: int) -> Dict[str, str]:
    layer_pattern = re.compile(r'^(model\.layers\.)([\d\*]+)(\..*)$')

    group_dict: Dict[Tuple[str, str], Dict[str, List[int] | str | None]] = {}
    for layer_path, q_type in original_dict.items():
        match = layer_pattern.match(layer_path)
        if not match:
            if ("other",) not in group_dict:
                group_dict[("other",)] = {"paths": {layer_path: q_type}}
            else:
                group_dict[("other",)]["paths"][layer_path] = q_type
            continue
        
        prefix, layer_part, suffix = match.groups()
        group_key = (prefix, suffix)

        if group_key not in group_dict:
            group_dict[group_key] = {"nums": [], "wildcard_qtype": None}

        if layer_part == "*":
            group_dict[group_key]["wildcard_qtype"] = q_type
        elif layer_part.isdigit():
            group_dict[group_key]["nums"].append(int(layer_part))

    result = {}
    for group_key, group_info in group_dict.items():
        if group_key == ("other",):
            for path, q_type in group_info["paths"].items():
                reg_path = f"re:{path}" if not path.startswith("re:") else path
                result[reg_path] = q_type
            continue
        
        prefix, suffix = group_key
        nums: List[int] = sorted(list(set(group_info["nums"])))
        wildcard_qtype: str | None = group_info["wildcard_qtype"]

        if nums:
            num_str = "|".join(map(str, nums))
            merged_num_path = f"{prefix}({num_str}){suffix}"
            reg_num_path = f"re:{merged_num_path}" if not merged_num_path.startswith("re:") else merged_num_path

            result[reg_num_path] = original_dict[f"{prefix}{nums[0]}{suffix}"]

        if wildcard_qtype is not None:
            all_layers = list(range(total_layers + 1))
            excluded_layers = nums
            remaining_layers = [x for x in all_layers if x not in excluded_layers]

            if not remaining_layers:
                continue
            remaining_str = "|".join(map(str, remaining_layers))
            merged_wildcard_path = f"{prefix}({remaining_str}){suffix}"
            reg_wildcard_path = f"re:{merged_wildcard_path}" if not merged_wildcard_path.startswith("re:") else merged_wildcard_path
            result[reg_wildcard_path] = wildcard_qtype

    return _expand_mlp_experts(result)


def _expand_mlp_experts(merged_dict: Dict[str, str]) -> Dict[str, str]:
    final_dict = {}
    mlp_suffixes = [".gate_proj", ".up_proj", ".down_proj"]
    mlp_pattern = re.compile(r're:.*mlp\.experts\.\*$')

    for pattern, quant_schema in merged_dict.items():
        if mlp_pattern.match(pattern):
            for suffix in mlp_suffixes:
                new_pattern = pattern + suffix
                final_dict[new_pattern] = quant_schema
        else:
            final_dict[pattern] = quant_schema

    return final_dict


def compress_hybrid_quant_schema(
    cfg: Dict[str, str],
    experts_num: int,
    layers_num: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    # extract layers mapping for pattern and quant_schema inside it
    mapping = {pattern: defaultdict(list) for pattern in TRANSFORMER_LAYER_PATTERNS}
    for layer_name, quant_schema in cfg.items():
        for pattern in TRANSFORMER_LAYER_PATTERNS:
            if fnmatch.fnmatchcase(layer_name, pattern):
                if pattern != "*":
                    mapping[pattern][quant_schema].append(layer_name)
                break
        else:
            raise ValueError

    # iterate through found mapping and substitute layer names with pattern if possible
    output = {}
    for pattern, pattern_mapping in mapping.items():
        if not pattern_mapping:
            continue
        elif len(pattern_mapping) == 1:
            output[pattern] = list(pattern_mapping.keys())[0]
        else:
            pattern_mapping_stats = sorted(
                (
                    (quant_schema, len(layers))
                    for quant_schema, layers in pattern_mapping.items()
                ),
                key=lambda x: x[1],
            )
            pattern_sorted_quant_schemas = [x[0] for x in pattern_mapping_stats]
            for quant_schema in pattern_sorted_quant_schemas[:-1]:
                if experts_num > 0 and "experts" in pattern:
                    # for experts we need additional "compression" step
                    # so it won't write quant schema for every expert of mlp layer
                    for k, v in _compress_expert_pattern_layers(
                        layers=pattern_mapping[quant_schema],
                        pattern=pattern,
                        quant_schema=quant_schema,
                        experts_num=experts_num,
                    ).items():
                        output[k] = v
                else:
                    for layer_name in pattern_mapping[quant_schema]:
                        output[layer_name] = quant_schema

            output[pattern] = pattern_sorted_quant_schemas[-1]

    output = _postprocess_expert_layers(output)
    output_re = _get_hybrid_quant_schema_re(output, layers_num)

    return output, output_re


TRANSFORMER_LAYER_PATTERNS = [
    # self attn
    "model.layers.*.self_attn.q_proj",
    "model.layers.*.self_attn.q_a_proj",
    "model.layers.*.self_attn.q_b_proj",
    "model.layers.*.self_attn.k_proj",
    "model.layers.*.self_attn.v_proj",
    "model.layers.*.self_attn.kv_a_proj_with_mqa",
    "model.layers.*.self_attn.kv_b_proj",
    "model.layers.*.self_attn.o_proj",
    "model.layers.*.input_layernorm",
    "model.layers.*.post_attention_layernorm",
    # mlp
    "model.layers.*.mlp.shared_expert.up_proj",
    "model.layers.*.mlp.shared_expert.gate_proj",
    "model.layers.*.mlp.shared_expert.down_proj",
    # "model.layers.*.mlp.experts.*",
    "model.layers.*.mlp.experts.*.up_proj",
    "model.layers.*.mlp.experts.*.gate_proj",
    "model.layers.*.mlp.experts.*.down_proj",
    "model.layers.*.mlp.up_proj",
    "model.layers.*.mlp.gate_proj",
    "model.layers.*.mlp.down_proj",
    # rest
    "*",
]


def _compress_expert_pattern_layers(
    layers: List[str],
    pattern: str,
    quant_schema: str,
    experts_num: int,
) -> Dict[str, str]:
    output = {}
    experts_stats = defaultdict(int)
    for layer_name in layers:
        layer_idx = int(layer_name.split("layers.")[-1].split(".")[0])

        experts_stats[layer_idx] += 1

    for layer_idx, v in experts_stats.items():
        if v != experts_num:
            raise ValueError(
                "All experts projections of the same type should be"
                f"quantized in the same quant schema, but there is {experts_num}"
                f"and {v} experts quantized in same schema for {layer_idx} mlp layer"
            )
        else:
            output[pattern.replace("layers.*", f"layers.{layer_idx}")] = quant_schema

    return output


def _postprocess_expert_layers(hybrid_quant_schema: Dict[str, str]) -> Dict[str, str]:
    updated_schema = hybrid_quant_schema.copy()
    for layer_name, quant_schema in hybrid_quant_schema.items():
        if "experts" in layer_name and "up_proj" in layer_name:
            neighbor_layers = [
                layer_name.replace("up_proj", neighbor_layer_name)
                for neighbor_layer_name in ["up_proj", "gate_proj", "down_proj"]
            ]
            schema_set = set()
            for neighbor_layer in neighbor_layers:
                schema_set.add(hybrid_quant_schema.get(neighbor_layer))

            if len(schema_set) == 1:
                for name in neighbor_layers:
                    updated_schema.pop(name)
                updated_schema[layer_name.replace(".up_proj", "")] = quant_schema

    return updated_schema
