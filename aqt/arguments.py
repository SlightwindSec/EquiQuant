# TODO: add missing args, clean paths afterwards

import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()

    # default
    parser.add_argument("--model-name-or-path", type=str)
    parser.add_argument("--model-save-path", type=str)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Run model in bfloat16 mode.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
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
        "--quant-data-path",
        type=str,
        help="Path to calibration samples.",
    )
    parser.add_argument(
        "--quant-data-save-path",
        type=str,
        help="Path to save processed calibration samples.",
    )
    parser.add_argument(
        "--quant-samples-num",
        type=int,
        default=128,
        help="Number of calibration data samples.",
    )
    parser.add_argument(
        "--quant-context-length",
        type=int,
        default=4096,
        help="Context length of calibration samples.",
    )
    parser.add_argument(
        "--quant-group-size",
        type=int,
        default=0,
        help="Groups size in per-group scenario.",
    )
    parser.add_argument(
        "--eval-ppl",
        action="store_true",
        help="Whether to evaluate perplexity after quantization.",
    )

    # quant hyperparams
    parser.add_argument(
        "--quant-type",
        type=str,
        choices=["minmax", "ssz", "modelslim"],
        default="minmax",
        help="Quantization Method.",
    )
    parser.add_argument(
        "--quant-sym",
        action="store_true",
        default=False,
        help="Whether to perform symmetric quantization.",
    )
    parser.add_argument(
        "--fuse-with-ln",
        action="store_true",
        help="Whether to fuse smoothquant scales with layernorm.",
    )
    parser.add_argument(
        "--disable-smoothquant",
        action="store_true",
        help="Turn off smooth quant scales completely.",
    )
    parser.add_argument(
        "--no-smoothquant-experts",
        action="store_true",
        help=(
            "Turn off smooth quant scales for regular expert layers."
        ),
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="Smooth quant alpha.",
    )
    parser.add_argument(
        "--quant-perc-damp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--quant-block-size",
        type=int,
        default=128,
        help="Block size to use for quantization..",
    )
    parser.add_argument(
        "--quant-act-order",
        action="store_true",
        help="Whether to apply the activation order GPTQ heuristic.",
    )

    # hybrid quant params
    parser.add_argument(
        "--hybrid-quant",
        action="store_true",
        help="Whether to apply hybrid quantization",
    )

    # automatic quantization args
    parser.add_argument(
        "--sensitivity-metric",
        default=None,
        help="Metric to use for computing sensitivity scores",
    )
    parser.add_argument(
        "--quant-cfg-updater",
        type=str,
        choices=["greedy", "lp"],
        default="greedy",
        help="Method to update hybrid config based on sensitivity scores",
    )
    parser.add_argument(
        "--compute-sensitivity-scores-only",
        action="store_true",
        default=False,
        help="Disable final quantization based on sensitivity scores.",
    )
    parser.add_argument(
        "--ckpt-size-budget-mb",
        type=int,
        default=2500,
        help="Checkpoint size budget for hybrid quantization.",
    )
    parser.add_argument("--save-dir", default="results")

    args = parser.parse_args()

    return args
