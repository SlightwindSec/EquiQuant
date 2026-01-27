import os
import copy
import gc
import json
import yaml
import argparse
from argparse import Namespace
from collections import defaultdict
from itertools import chain
from os.path import join as pjoin
from typing import Any, Dict, List, Optional, Union
from utils.logger import logger

import torch
import torch_npu
from torch import Tensor, nn

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import (
        Quantizer as MSQuantizer,
    )
from transformers import AutoModelForCausalLM

from aqt.utils.model import (
    catch_model_cache,
    find_layers,
    get_linear_layers_classes,
)
from aqt.minmax import MinMax
from aqt.modelslim import ModelslimQuantization
from aqt.ptq_method import PostTrainingQuantization
from aqt.quantizer import Quantizer
from aqt.utils.quant_config_manager import (
    QuantLayerConfig,
    QuantLayerConfigManager,
    compress_hybrid_quant_schema,
)
from aqt.arguments import parse_args

from aqt.utils.data import prepare_calibration_samples
from aqt.utils.common import seed_everything
from aqt.sensitivity import (
    analyze_sensitivity_scores,
    get_layer_sensitivity_group_mapping,
    update_quant_layer_cfg_greedy,
    update_quant_layer_cfg_lp,
    show_diff_between_bits,
    get_subset_layer_names,
)


MEGABYTE_SIZE = 1024**2
BYTES_PER_4BIT_PARAM = 0.5
BYTES_PER_8BIT_PARAM = 1


def _compute_sensitivity_scores(
    model: nn.Module,
    args: Namespace,
    quant_layer_cfg_mngr: QuantLayerConfigManager,
    ms_calibrator: Optional[Calibrator],
    calibration_samples: Tensor,
) -> Dict[str, Dict]:
    # Note: This function is directly inherited from our quantization pipeline
    # where we usually use GPTQ as a weight quantizer. Once we decide to use
    # data-free only PTQ techniques to compute sensitivity scores, this code
    # can be significantly simplified and refactored.
    logger.info("Computing sensitivity scores...")

    samples_num = len(calibration_samples)
    model.eval()
    model.cpu()

    # TODO: add support for other types of architectures
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # collect inputs
    print(f"Gathering inputs...")
    model.model.embed_tokens = model.model.embed_tokens.npu()
    model.model.norm = model.model.norm.npu()
    inps, model_cache = catch_model_cache(
        model=model,
        layers=layers,
        calibration_samples=calibration_samples,
    )
    print(f"{len(inps)}, {inps[0][0].shape}")
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()

    torch.npu.empty_cache()
    torch.npu.synchronize()

    experts_num = getattr(model.config, "num_experts", 0)

    # quantization
    sensitivity_scores = defaultdict(dict)
    outs = copy.deepcopy(inps)
    if args.sensitivity_metric is not None:
        sensitivity_metrics = args.sensitivity_metric.split(",")
    else:
        sensitivity_metrics = []

    if ms_calibrator is not None:
        for name, module in model.model.named_modules():
            if f"model.{name}" in ms_calibrator.cfg.disable_names:
                continue

            if isinstance(module, get_linear_layers_classes()):
                module.quant_weight.is_enable = False
                module.quant_input.is_enable = False

    layer_iterator = enumerate(layers)

    for layer_idx, layer in layer_iterator:
        layer_idx = str(layer_idx).replace("model.layers.", "")
        print(f"Processing layer {layer_idx}")

        if ms_calibrator is not None and not isinstance(layer, MSQuantizer):
            ms_calibrator.quantize_model(layer)

        layer.npu()

        full = find_layers(layer)
        layer_groups = get_layer_sensitivity_group_mapping(experts_num).values()
        # Note: For AQT, we compute sensitivity scores independently,
        # so there's no need to consider groups of layers inside transformer
        # block like qkv or up_gate separately. However, we need this mapping
        # to skip layers that shouldn't be quantized by model architecture design.
        subset = {n: full[n] for n in chain.from_iterable(layer_groups) if n in full}
        if not subset:
            continue

        ptq: Dict[str, PostTrainingQuantization] = {}
        quant_bits_list = [4, 8]  # TODO: refactor this for loop later
        for quant_bits in quant_bits_list:
            for name, linear_layer in subset.items():
                layer_name = f"model.layers.{layer_idx}.{name}"
                quant_layer_cfg_mngr.update_hybrid_quant_config(
                    args=args,
                    module=linear_layer,
                    name=layer_name,
                    cfg=quant_layer_cfg_mngr.cfg,
                )
                ptq[name] = _get_ptq(
                    linear_layer=linear_layer,
                    layer_name=layer_name,
                    args=args,
                    quant_layer_cfg_mngr=quant_layer_cfg_mngr,
                    quant_bits=quant_bits,
                    sensitivity_metrics=sensitivity_metrics,
                    ms_calibrator=ms_calibrator,
                )

            def add_batch(name_: str):
                def tmp(_, inp, out):
                    ptq[name_].add_batch(inp, out)

                return tmp

            handles = []
            for name, module in subset.items():
                handles.append(module.register_forward_hook(add_batch(name)))
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
                print(f"Layer {layer_idx}: {name}")
                print(f"Quantizing to {quant_bits} bits")
                # In activations quantization scenario, if smooth scales have been
                # computed but not fused with layernorm, weights weren't transformed.
                # We need to do it here for quantization...
                layer_name = f"model.layers.{layer_idx}.{name}"

                losses = ptq[name].run(transform_weights=False)
                sensitivity_scores[layer_name][quant_bits] = {
                    metric: losses[i] for i, metric in enumerate(sensitivity_metrics)
                }

                ptq[name].free()

        # Note: we use ratio scores now for auto quant schema search,
        # which should be computed after computing absolute scores
        for layer_group in layer_groups:
            subset = {n: full[n] for n in layer_group if n in full}

            subset_scores = [-torch.inf for _ in range(len(sensitivity_metrics))]
            for name in subset:
                layer_name = f"model.layers.{layer_idx}.{name}"
                ratio_scores = [
                    (
                        sensitivity_scores[layer_name][4][metric]
                        / (sensitivity_scores[layer_name][8][metric] + 1e-9)
                    )
                    for metric in sensitivity_metrics
                ]
                subset_scores = [
                    max(subset_scores[i], ratio_scores[i])
                    for i in range(len(subset_scores))
                ]

            for name, module in subset.items():
                layer_name = f"model.layers.{layer_idx}.{name}"
                sensitivity_scores[layer_name]["ratio"] = {
                    metric: subset_scores[i]
                    for i, metric in enumerate(sensitivity_metrics)
                }
                sensitivity_scores[layer_name]["size"] = module.weight.numel()

        inps, outs = outs, inps

        layer = layer.cpu()
        del layer
        torch.npu.empty_cache()
        torch.npu.synchronize()

    del inps, outs
    del model_cache
    gc.collect()
    torch.npu.empty_cache()

    model.config.use_cache = use_cache

    logger.info("Sensitivity scores have been computed.")

    # sensitivity scores analysis
    if args.sensitivity_metric is not None:
        for metric in sensitivity_metrics:
            save_dir = os.path.join(args.save_dir, args.quant_type, metric)
            os.makedirs(save_dir, exist_ok=True)
            analyze_sensitivity_scores(
                sensitivity_scores=sensitivity_scores,
                score_name=metric,
                save_dir=save_dir,
                experts_num=experts_num,
            )
            show_diff_between_bits(
                sensitivity_scores=sensitivity_scores,
                score_name=metric,
                save_dir=save_dir,
                experts_num=experts_num,
            )

        scores_save_path = os.path.join(
            save_dir, f"{args.sensitivity_metric}_scores.json"
        )
        with open(scores_save_path, "w") as f:
            json.dump(sensitivity_scores, f, indent=4)

    return sensitivity_scores

def _get_ptq(
    linear_layer: nn.Module,
    layer_name: str,
    args: Namespace,
    quant_layer_cfg_mngr: QuantLayerConfigManager,
    quant_bits: int,
    sensitivity_metrics: List[str],
    ms_calibrator: Optional[Calibrator] = None,
) -> PostTrainingQuantization:
    if args.quant_type in ("minmax", "ssz"):
        return MinMax(
            layer=linear_layer,
            quantizer=Quantizer(
                bits=quant_bits,
                qfn="ssz" if args.quant_type == "ssz" else "a",
                perchannel=True,
                sym=args.quant_sym,
            ),
            group_size=quant_layer_cfg_mngr.get_group_size(layer_name),
            context_length=args.quant_context_length,
            sensitivity_metric=sensitivity_metrics,
        )
    elif args.quant_type == "modelslim":
        # disable quant until PTQ class "run" method call
        linear_layer.quant_weight.bit = quant_bits
        linear_layer.cfg.w_bit = quant_bits
        return ModelslimQuantization(
            layer=linear_layer,
            calibrator=ms_calibrator,
            group_size=quant_layer_cfg_mngr.get_group_size(layer_name),
            context_length=args.quant_context_length,
            sensitivity_metric=sensitivity_metrics,
        )
    else:
        raise NotImplementedError


def update_quant_layer_cfg(
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
        # if name not in quant_layer_cfg_mngr.cfg:
        #     quant_layer_cfg_mngr.cfg[name] = QuantLayerConfig(
        #         weight_bits=args.weight_quant_bits,
        #         act_bits=args.act_quant_bits,
        #         group_size=args.quant_group_size,
        #     )

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

    # TODO: we consider only 4 vs 8 bit case here. Extend with fp16/bf16 later
    ckpt_size_budget_mb = ckpt_size_budget_mb * MEGABYTE_SIZE
    skipped = []
    while curr_ckpt_diff < ckpt_size_budget_mb and layer_num < len(layer_score_info):
        subset_name = layer_score_info[layer_num][1]
        layer_num += 1
        if "self_attn." in subset_name:
            continue

        # bit_mapping_cfg = _get_lower_upper_bit_type(subset_name)

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
        # bit_mapping_cfg = _get_lower_upper_bit_type(subset_name)
        for layer_name in layer_names:
            quant_layer_cfg_mngr.cfg[layer_name].weight_bits = bit_mapping_cfg["lower"]


def _get_lower_upper_bit_type(subset_name: str) -> Dict[str, Union[int, float]]:
    return {"lower": 4, "upper": 8, "bytes_per_param": 0.5}


def main() -> None:
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True,
    )
    count = 0
    for name, module in model.named_modules():
        if count < 5:
            logger.info(f"name: {name}, module: {module}")
        count += 1

    seed_everything(args.seed)
    calibration_samples = prepare_calibration_samples(args=args)
    logger.info("prepare calibration samples successfully!")

    disable_names = []
    # for ids in range(model.config.num_hidden_layers):
    #     disable_names.append("model.layers." + str(ids) + ".mlp.gate")

    assert (
        args.quant_type == "modelslim"
        and args.weight_quant_bits == 4
        and args.act_quant_bits == 16
    )
    ms_quant_cfg = QuantConfig(
        a_bit=16,
        w_bit=4,
        disable_names=disable_names,
        dev_type="cpu",
        pr=1.0,
        w_sym=args.quant_sym,
        mm_tensor=False,
        is_lowbit=False,
    )
    ms_calibrator = Calibrator(
        model=model,
        cfg=ms_quant_cfg,
        calib_data=None,
        disable_level="L0",
        mix_cfg=None,
    )
    
    quant_layer_cfg_mngr = QuantLayerConfigManager(args=args, model=model)

    with torch.no_grad():
        seed_everything(args.seed)
        sensitivity_scores = _compute_sensitivity_scores(
            model=model,
            quant_layer_cfg_mngr=quant_layer_cfg_mngr,
            ms_calibrator=ms_calibrator,
            args=args,
            calibration_samples=calibration_samples,
        )
        # if args.quant_cfg_updater == "greedy":
        #     quant_cfg_updater_method = update_quant_layer_cfg_greedy
        # elif args.quant_cfg_updater == "lp":
        #     quant_cfg_updater_method = update_quant_layer_cfg_lp
        # else:
        #     raise NotImplementedError

        # metric = args.sensitivity_metric.split(",")[0]
        # quant_cfg_updater_method(
        #     sensitivity_scores=sensitivity_scores,
        #     model=model,
        #     quant_layer_cfg_mngr=quant_layer_cfg_mngr,
        #     score_name=metric,
        #     ckpt_size_budget_mb=args.ckpt_size_budget_mb,
        # )
        # quant_layer_cfg_mngr.save_hybrid_quant_cfg(
        #     pjoin(args.save_dir, "hybrid_quant_config.json")
        # )

    update_quant_layer_cfg(
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
    
    with open(pjoin(args.save_dir, "hybrid_quant_schema.json"), "w", encoding="utf-8") as f:
        print("Saving hybrid quant config...")
        json.dump(hybrid_quant_schema, f, indent=4)


if __name__ == "__main__":
    main()
