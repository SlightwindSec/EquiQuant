import json
import argparse
from argparse import Namespace
from typing import Dict, Any
from ..utils.logger import logger

import torch
import torch_npu
from torch import Tensor, nn

from transformers import AutoModelForCausalLM

from .utils.model import (
    catch_model_cache,
    find_layers,
)
from .ptq import PostTrainingQuantization
from .utils.data import prepare_calibration_samples
from .utils.common import seed_everything, cleanup_memory, calculate_weight_size
from .moe_utils import NEED_CONVERT_MOE, CONVERT_MOE_FUNC
from .sensitivity import (
    get_layer_sensitivity_group_mapping,
    calculate_losses,
)


def _compute_sensitivity_scores(
    model: nn.Module, args: Namespace, input_ids: Tensor, adapter: Any = None
) -> None:
    logger.info("Computing sensitivity scores...")
    # ========================================================================
    if adapter is None:
        use_cache = model.config.use_cache
        model.config.use_cache = False
    layers = model.model.layers
    logger.info(f"This model gets {len(layers)} layers.")

    logger.info("Gathering inputs from Embeddings...")
    inps, model_cache = catch_model_cache(model=model, input_ids=input_ids)
    if args.is_deepseek_v32:
        inps = (inps[0], None)
    logger.info(f"inps shape: {inps[0].shape}")
    logger.info("Inputs gathered.")

    if hasattr(model.config, "num_experts"):
        num_experts = model.config.num_experts
    elif hasattr(model.config, "n_routed_experts"):
        num_experts = model.config.n_routed_experts
    else:
        num_experts = 0
    num_layers = model.config.num_hidden_layers
    prefix = "model.language_model.layers" if args.is_mm else "model.layers"

    sensitivity_scores = {}
    sensitivity_metrics = args.sensitivity_metrics.split(",")
    logger.info(f"Computing sensitivity scores for {sensitivity_metrics}...")

    layer_iter = adapter.generate_decoder_layer(model) if adapter else enumerate(layers)

    for layer_idx, layer in layer_iter:
        layer_idx_str = str(layer_idx)
        logger.info(f"Processing layer {layer_idx_str}.")

        layer.npu()

        full = find_layers(layer)
        logger.info(f"This layer gets {len(full)} linears.")
        layer_groups = get_layer_sensitivity_group_mapping(num_experts)
        subset = {}
        for group_name, sub_names in layer_groups.items():
            matched_subs = {n: full[n] for n in sub_names if n in full}
            if matched_subs:
                subset[group_name] = matched_subs
        for key, value in subset.items():
            logger.info(f"{key}: {len(value)}")

        if not subset:
            layer.cpu()
            del layer
            cleanup_memory()
            continue

        # r1: List[Tensor] = [None]

        # def _make_hook():
        #     def hook(module, input):
        #         t = input[0] if isinstance(input, tuple) else input
        #         r1[0] = t.detach().cpu()

        #     return hook

        # handles = [
        #     layer.post_attention_layernorm.register_forward_pre_hook(_make_hook())
        # ]

        inps_npu = tuple(t.npu() if t is not None else t for t in inps) 
        with torch.no_grad():
            layer_out = layer(*inps_npu, **model_cache)
        if args.is_deepseek_v32:
            outs = layer_out[0] + layer_out[1]
        else:
            outs = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        outs = outs.detach().cpu()

        # r1 = r1[0]

        # for h in handles:
        #     h.remove()

        for name, linears in subset.items():

            layer_name = f"{prefix}.{layer_idx_str}.{name}"
            logger.info(f"Processing {layer_name}")
            size = calculate_weight_size(linears)
            results = {"size": size * 2}
            quant_bits = [4, 8] if "mlp.experts" in name else [8]
            for quant_bit in quant_bits:
                logger.info(f"quantizing to {quant_bit}-bit...")
                ptq = PostTrainingQuantization(
                    layers=linears,
                    quant_type=args.quant_type,
                    quant_bit=quant_bit,
                    quant_sym=True,
                    group_size=0,
                )
                # 浮点权重下 cpu，linears权重变为伪量化权重，上 npu
                ptq.run()
                logger.info("computing losses...")
                part_results = {"size": size * quant_bit / 8}
                # if "mlp" in name:
                #     with torch.no_grad():
                #         r2_fake = layer.mlp(layer.post_attention_layernorm(r1.npu()))
                #     r2_fake = r2_fake[0] if isinstance(r2_fake, tuple) else r2_fake
                #     r2_fake = r2_fake.detach().cpu() + r1
                #     part_results["metrics"] = calculate_losses(
                #         y_true=outs,
                #         y_fake=r2_fake,
                #         metrics=sensitivity_metrics,
                #     )
                # else:
                    # with torch.no_grad():
                    #     r1_fake = layer.self_attn(
                    #         layer.input_layernorm(inps_npu[0]), **model_cache
                    #     )
                    # r1_fake = r1_fake[0] if isinstance(r1_fake, tuple) else r1_fake
                    # r1_fake = r1_fake.detach().cpu() + inps[0]
                    # part_results["metrics"] = calculate_losses(
                    #     y_true=r1,
                    #     y_fake=r1_fake,
                    #     metrics=sensitivity_metrics,
                    # )
                with torch.no_grad():
                    fake = layer(*inps_npu, **model_cache)
                if args.is_deepseek_v32:
                    fake = fake[0] + fake[1]
                else:
                    fake = fake[0] if isinstance(fake, tuple) else fake
                fake = fake.detach().cpu()
                part_results["metrics"] = calculate_losses(
                    y_true=outs,
                    y_fake=fake,
                    metrics=sensitivity_metrics,
                )
                results[f"{quant_bit}-bit"] = part_results

                # 删除量化权重，linears权重恢复为浮点权重
                ptq.free()
                del ptq, fake

            if "mlp.experts" in name:
                results["gold"] = (
                    (
                        results["4-bit"]["metrics"]["mse"]
                        - results["8-bit"]["metrics"]["mse"]
                    )
                    * 1e8
                    / (results["8-bit"]["size"] - results["4-bit"]["size"])
                )
            else:
                results["gold"] = (
                    results["8-bit"]["metrics"]["mse"]
                    * 1e8
                    / (results["size"] - results["8-bit"]["size"])
                )

            logger.info(f"{layer_name} = {results}")

            sensitivity_scores[layer_name] = results

        inps = layer_out
        layer.cpu()
        # del r1, handles
        del outs, inps_npu, layer_out
        cleanup_memory()

    if adapter is None:
        model.config.use_cache = use_cache
    del inps, model_cache
    cleanup_memory()
    # ========================================================================
    logger.info("Sensitivity scores have been computed.")

    with open(args.sensitivity_scores_save_path, "w") as f:
        json.dump(sensitivity_scores, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--quant-data-path", required=True, type=str)
    parser.add_argument("--quant-data-save-path", required=True, type=str)
    parser.add_argument("--quant-samples-num", required=True, type=int)
    parser.add_argument("--quant-context-length", required=True, type=int)
    parser.add_argument("--quant-type", required=True, type=str)
    parser.add_argument("--sensitivity-metrics", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--sensitivity_scores_save_path", required=True, type=str)
    parser.add_argument("--is-mm", action="store_true")
    parser.add_argument("--is-deepseek-v32", action="store_true")

    args = parser.parse_args()

    if args.is_deepseek_v32:
        from msmodelslim.model.deepseek_v3_2.model_adapter import DeepSeekV32ModelAdapter

        model_type = "deepseek_v32"
        adapter = DeepSeekV32ModelAdapter(
            model_path=args.model_name_or_path,
            model_type="DeepSeek-V3.2",
        )
        model = adapter.init_model(device="cpu")
        model.config = adapter._load_config()
        model.dtype = torch.bfloat16
        model.device = torch.device("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu",
            local_files_only=True,
        )
        model_type = model.config.model_type
        adapter = None

    logger.info(f"model_type = {model_type}")
    if NEED_CONVERT_MOE[model_type]:
        CONVERT_MOE_FUNC[model_type](model.model)
    
    model.eval()

    seed_everything(args.seed)
    input_ids = prepare_calibration_samples(args=args)
    logger.info("prepare calibration samples successfully!")

    _compute_sensitivity_scores(
        model=model,
        args=args,
        input_ids=input_ids,
        adapter=adapter,
    )


if __name__ == "__main__":
    main()
