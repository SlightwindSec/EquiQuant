import copy
import gc
import json
import argparse
from argparse import Namespace
from collections import defaultdict
from itertools import chain
from typing import Dict
from utils.logger import logger

import torch
import torch_npu
from torch import Tensor, nn

from transformers import AutoModelForCausalLM

from aqt.utils.model import (
    catch_model_cache,
    find_layers,
)
from aqt.ptq import PostTrainingQuantization
from aqt.utils.data import prepare_calibration_samples
from aqt.utils.common import seed_everything
from aqt.sensitivity import (
    analyze_sensitivity_scores,
    get_layer_sensitivity_group_mapping,
    show_diff_between_bits,
)


def cleanup_memory():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()


def _compute_sensitivity_scores(
    model: nn.Module, args: Namespace, calibration_samples: Tensor
) -> None:
    logger.info("Computing sensitivity scores...")
    # ========================================================================
    samples_num = len(calibration_samples)
    model.eval()
    model.cpu()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    logger.info(f"This model gets {len(layers)} layers.")

    logger.info("Gathering inputs from Embeddings...")
    model.model.embed_tokens = model.model.embed_tokens.npu()
    model.model.norm = model.model.norm.npu()
    inps, model_cache = catch_model_cache(
        model=model, layers=layers, calibration_samples=calibration_samples
    )
    outs = [None] * samples_num
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    cleanup_memory()
    logger.info("Inputs gathered.")

    experts_num = getattr(model.config, "num_experts", 0)
    sensitivity_scores = defaultdict(dict)

    for layer_idx, layer in enumerate(layers):
        layer_idx_str = str(layer_idx)
        logger.info(f"Processing layer {layer_idx_str}.")

        layer.npu()

        full = find_layers(layer)
        logger.info(f"This layer gets {len(full)} linears.")
        layer_groups = get_layer_sensitivity_group_mapping(experts_num).values()
        subset = {n: full[n] for n in chain.from_iterable(layer_groups) if n in full}

        if not subset:
            layer.cpu()
            del layer
            cleanup_memory()
            continue

        quant_bits_list = [4, 8]
        ptq: Dict[str, PostTrainingQuantization] = {}
        for quant_bits in quant_bits_list:
            logger.info(f"Quantizing to {quant_bits} bits...")
            for name, linear_layer in subset.items():
                layer_name = f"model.layers.{layer_idx_str}.{name}"
                ptq[name] = PostTrainingQuantization(
                    layer=linear_layer,
                    quant_type=args.quant_type,
                    quant_bits=quant_bits,
                    quant_sym=True,
                    context_length=args.quant_context_length,
                    group_size=0,
                    sensitivity_metric=args.sensitivity_metric,
                )

            def add_batch(name_: str):
                def tmp(_, inp, out):
                    ptq[name_].add_batch(inp, out)

                return tmp

            handles = [m.register_forward_hook(add_batch(n)) for n, m in subset.items()]

            for j in range(samples_num):
                res = layer(
                    *inps[j],
                    **{
                        k: v[j] if isinstance(v, list) else v
                        for k, v in model_cache.items()
                    },
                )
                outs[j] = res if isinstance(res, tuple) else (res,)
            for h in handles:
                h.remove()
            for name in subset:
                ptq[name].post_batch()

            for name, module in subset.items():
                layer_name = f"model.layers.{layer_idx_str}.{name}"
                losses = ptq[name].run(transform_weights=False)
                sensitivity_scores[layer_name][quant_bits] = {
                    args.sensitivity_metric: losses[0]
                }
                ptq[name].free()

        del ptq
        cleanup_memory()

        for layer_group in layer_groups:
            subset = {n: full[n] for n in layer_group if n in full}
            if not subset:
                continue
            subset_scores = [-torch.inf]
            for name in subset:
                layer_name = f"model.layers.{layer_idx_str}.{name}"
                ratio_scores = [
                    sensitivity_scores[layer_name][4][args.sensitivity_metric]
                    / (
                        sensitivity_scores[layer_name][8][args.sensitivity_metric]
                        + 1e-9
                    )
                ]
                subset_scores = [max(subset_scores[0], ratio_scores[0])]

            for name, module in subset.items():
                layer_name = f"model.layers.{layer_idx_str}.{name}"
                sensitivity_scores[layer_name]["ratio"] = {
                    args.sensitivity_metric: subset_scores[0]
                }
                sensitivity_scores[layer_name]["size"] = module.weight.numel()

        inps, outs = outs, inps
        layer.cpu()
        del layer
        cleanup_memory()

    del inps, outs, model_cache
    cleanup_memory()
    model.config.use_cache = use_cache
    # ========================================================================

    logger.info("Sensitivity scores have been computed.")

    # sensitivity scores analysis
    analyze_sensitivity_scores(
        sensitivity_scores=sensitivity_scores,
        score_name=args.sensitivity_metric,
        save_dir=args.save_dir,
        experts_num=experts_num,
    )
    show_diff_between_bits(
        sensitivity_scores=sensitivity_scores,
        score_name=args.sensitivity_metric,
        save_dir=args.save_dir,
        experts_num=experts_num,
    )

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
    parser.add_argument("--sensitivity-metric", required=True, type=str)
    parser.add_argument("--save-dir", required=True, type=str)
    parser.add_argument("--sensitivity_scores_save_path", required=True, type=str)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True,
    )

    seed_everything(args.seed)
    calibration_samples = prepare_calibration_samples(args=args)
    logger.info("prepare calibration samples successfully!")

    with torch.no_grad():
        _compute_sensitivity_scores(
            model=model,
            args=args,
            calibration_samples=calibration_samples,
        )


if __name__ == "__main__":
    main()
