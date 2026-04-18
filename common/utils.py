import os
import json
import pickle
import numpy as np
from typing import Dict, Any, Tuple
import random
import torch


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: Dict[str, Any], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def ensure_dir(directory: str):
    os.makedirs(directory, exist_ok=True)


def get_feature_bounds(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X.min(axis=0), X.max(axis=0)


def clip_to_bounds(X: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray) -> np.ndarray:
    return np.clip(X, min_bounds, max_bounds)


def print_section_header(title: str, width: int = 80) -> None:
    print(f"\n{'=' * width}")
    print(title)
    print(f"{'=' * width}")


def print_dict_block(label: str, data: Dict[str, Any], key_width: int = 30) -> None:
    print(f"\n{label}")
    for key, value in data.items():
        print(f"  {key:{key_width}s}: {value}")