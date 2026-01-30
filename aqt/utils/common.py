import os
import random
from pathlib import Path

import numpy as np
import torch


def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def seed_everything(seed: int = 42, torch_deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.use_deterministic_algorithms(mode=torch_deterministic)
