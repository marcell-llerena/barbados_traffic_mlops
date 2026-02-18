import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU and CUDA).

    Ensures reproducible behavior across runs. When deterministic is True,
    enables PyTorch deterministic algorithms and CUDNN deterministic mode,
    and sets CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED. Deterministic mode
    may reduce performance.

    Args:
        seed: Value to seed all RNGs. Defaults to 0.
        deterministic: If True, enable full deterministic behavior in PyTorch
            and set environment variables for reproducibility.
    """
    # https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
