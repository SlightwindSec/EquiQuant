from ..utils.common import is_transformers_ge


NEED_CONVERT_MOE = {
    "qwen3_5_moe": True,
    "qwen3_5_moe_text": True,
    "qwen3_moe": is_transformers_ge("5.0.0"),
    "deepseek_v32": is_transformers_ge("5.0.0"),
}

CONVERT_MOE_FUNC = {
    "qwen3_5_moe": None,
    "qwen3_5_moe_text": None,
    "qwen3_moe": None,
    "deepseek_v32": None,
}

if is_transformers_ge("5.2.0"):
    from .convert_qwen3_5_moe import convert_qwen3_5_moe
    CONVERT_MOE_FUNC["qwen3_5_moe"] = convert_qwen3_5_moe
    CONVERT_MOE_FUNC["qwen3_5_moe_text"] = convert_qwen3_5_moe

if is_transformers_ge("5.0.0"):
    from .convert_qwen3_moe import convert_qwen3_moe
    CONVERT_MOE_FUNC["qwen3_moe"] = convert_qwen3_moe