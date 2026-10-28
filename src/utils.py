"""Small shared helpers."""
import json
import os
import random

import numpy as np


def seed_everything(seed: int = 42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def file_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024


def state_dict_size_mb(model, tmp="/tmp/_size_probe.pt"):
    import torch
    torch.save(model.state_dict(), tmp)
    size = file_size_mb(tmp)
    os.remove(tmp)
    return size


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
