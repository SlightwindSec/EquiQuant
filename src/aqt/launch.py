import json
import argparse
from ..utils.logger import logger
import os

from .utils.quant_config_manager import (
    QuantLayerConfigManager,
    compress_hybrid_quant_schema,
)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True, type=str)
    parser.add_argument("--layer-configs-path", type=str, default=None)
    parser.add_argument("--initial-configs-rules-path", type=str, default=None)
    parser.add_argument("--hybrid-quant-schema-path", required=True, type=str)
    parser.add_argument("--hybrid-quant-schema-re-path", required=True, type=str)
    parser.add_argument("--sensitivity-scores-save-path", required=True, type=str)
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

    # 1. Load last layer configs if provided
    layer_configs = None
    if args.layer_configs_path and os.path.exists(args.layer_configs_path):
        logger.info(f"Loading stateful layer configurations from {args.layer_configs_path}")
        with open(args.layer_configs_path, 'r') as f:
            layer_configs = json.load(f)

    # 2. Load initial config rules if provided
    initial_configs_rules = None
    if args.initial_configs_rules_path and os.path.exists(args.initial_configs_rules_path):
        logger.info(f"Loading initial configuration rules from {args.initial_configs_rules_path}")
        with open(args.initial_configs_rules_path, 'r') as f:
            initial_configs_rules = json.load(f)

    # 3. Initialize Manager (Initialization logic is now inside __init__)
    quant_layer_cfg_mgr = QuantLayerConfigManager(
        num_experts=num_experts, 
        num_layers=num_layers,
        sensitivity_scores=sensitivity_scores,
        layer_configs=layer_configs,
        initial_configs_rules=initial_configs_rules
    )

    layers_quant_mapping = quant_layer_cfg_mgr.create_quant_layers_mapping()
    hybrid_quant_schema, hybrid_quant_schema_re = compress_hybrid_quant_schema(layers_quant_mapping, is_mm=args.is_mm)

    logger.info("Saving hybrid quant config...")
    with open(args.hybrid_quant_schema_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_quant_schema, f, indent=4)
    with open(args.hybrid_quant_schema_re_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_quant_schema_re, f, indent=4)
    
    # Also save the current layer configs state for the engine to persist
    if args.layer_configs_path:
        current_state = {
            name: {"weight_bits": cfg.weight_bits, "act_scope": cfg.act_scope}
            for name, cfg in quant_layer_cfg_mgr.cfg.items()
        }
        with open(args.layer_configs_path, "w") as f:
            json.dump(current_state, f, indent=4)


if __name__ == "__main__":
    main()
