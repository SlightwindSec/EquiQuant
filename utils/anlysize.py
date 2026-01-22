import os
import copy
import gc
import json
import argparse
from argparse import Namespace
from collections import defaultdict
from itertools import chain
from os.path import join as pjoin
from typing import Any, Dict, List, Optional

import torch
import torch_npu
from torch import Tensor, nn

from msmodelslim.model.deepseek_v3_2.model import ModelArgs
from msmodelslim.model.deepseek_v3_2.model_adapter import DeepSeekV32ModelAdapter
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import (
        Quantizer as MSQuantizer,
    )
from transformers import AutoModelForCausalLM

from utils.model import (
    catch_model_cache,
    find_layers,
    get_linear_layers_classes,
)
from utils.minmax import MinMax
from utils.modelslim import ModelslimQuantization
from utils.ptq_method import PostTrainingQuantization
from utils.quantizer import Quantizer
from utils.quant_config_manager import QuantLayerConfigManager
from utils.arguments import parse_args

from utils.data import prepare_calibration_samples
from utils.common import seed_everything
from utils.sensitivity import (
    analyze_sensitivity_scores,
    get_layer_sensitivity_group_mapping,
    show_diff_between_bits,
)


def _compute_sensitivity_scores(
    model: nn.Module,
    args: Namespace,
    quant_layer_cfg_mngr: QuantLayerConfigManager,
    ms_calibrator: Optional[Calibrator],
    calibration_samples: Tensor,
    sq_scales: Dict[str, Tensor],
    adapter: Optional[DeepSeekV32ModelAdapter],
) -> Dict[str, Dict]:
    # Note: This function is directly inherited from our quantization pipeline
    # where we usually use GPTQ as a weight quantizer. Once we decide to use
    # data-free only PTQ techniques to compute sensitivity scores, this code
    # can be significantly simplified and refactored.
    print("Computing sensitivity scores...")

    samples_num = len(calibration_samples)
    model.eval()
    model.cpu()

    # TODO: add support for other types of architectures
    if not args.is_deepseek_v3_2:
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

    if args.is_deepseek_v3_2:
        layer_iterator = adapter.generate_decoder_layer(model)
    else:
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
                    sq_scales=sq_scales,
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
                sq_scale = sq_scales.get(layer_name, None)
                if sq_scale is not None:
                    print("Multiplying the weights with smooth scales...")
                    module.weight.data.mul_(sq_scale.to(module.weight).view(1, -1))

                losses = ptq[name].run(transform_weights=False)
                sensitivity_scores[layer_name][quant_bits] = {
                    metric: losses[i] for i, metric in enumerate(sensitivity_metrics)
                }

                # ...and revert this transformation back
                if sq_scale is not None:
                    print("Reverting smooth scales on dequantized weights...")
                    module.weight.data.div_(sq_scale.to(module.weight).view(1, -1))

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

    if not args.is_deepseek_v3_2:
        model.config.use_cache = use_cache

    print("Sensitivity scores have been computed.")

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
    sq_scales: Dict[str, Tensor],
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
            sq_scales=sq_scales.get(layer_name, None),
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

def main() -> None:
    args = parse_args()
    print(args.__dict__)

    if args.is_deepseek_v3_2:
        model_type = "DeepSeek-V3.2-Exp"
        adapter = DeepSeekV32ModelAdapter(model_path=args.model_name_or_path, model_type=model_type)
        model = adapter.init_model(device="cpu")
        model.config = adapter._load_config()
        model.config.num_experts = model.config.n_routed_experts
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
        adapter = None

    seed_everything(args.seed)
    calibration_samples = prepare_calibration_samples(args=args)

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
    
    quant_layer_cfg_mngr = QuantLayerConfigManager(args=args,model=model)

    with torch.no_grad():
        sq_scales: Dict[str, Tensor] = {}
        seed_everything(args.seed)
        sensitivity_scores = _compute_sensitivity_scores(
                model=model,
                quant_layer_cfg_mngr=quant_layer_cfg_mngr,
                ms_calibrator=ms_calibrator,
                args=args,
                calibration_samples=calibration_samples,
                sq_scales=sq_scales,
                adapter=adapter,
            )

if __name__ == "__main__":
    main()