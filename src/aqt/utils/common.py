import gc
import os
import random
from pathlib import Path
from packaging import version

import numpy as np
import torch
from ...utils.logger import logger


def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def seed_everything(seed: int = 42, torch_deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=torch_deterministic)


def cleanup_memory():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()


def write_log(content: str, log_file: str="tmp/log.log") -> None:
    try:
        # 获取日志文件所在目录，不存在则创建
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
        log_content = f"{content}\n"
    
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_content)
    
    except Exception as e:
        print(f"写入日志失败: {str(e)}")


def print_memory_usage():
    allocated = torch.npu.memory_allocated() / 1024 / 1024 / 1024
    cache = torch.npu.memory_reserved() / 1024 / 1024 / 1024
    logger.info(f"Allocated: {allocated:.2f} GB, Cached: {cache:.2f} GB")


def is_transformers_ge(ve: str = "5.0.0") -> bool:
    try:
        import transformers
        return version.parse(transformers.__version__) >= version.parse(ve)
    except ImportError:
        logger.error("transformers not installed")
        return False
