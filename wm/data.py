"""Load collected trajectories onto the GPU and sample training batches there.

The whole dataset is a few GB of uint8, so we keep it resident on the device and
index random batches directly — no DataLoader, no worker processes, no copies.
"""

from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parents[1] / "data"


def load(layout="four_rooms", split="train", device="cuda"):
    """Dict of tensors: obs (N, T+1, H, W, 3) uint8, actions (N, T, 2),
    pos (N, T+1, 2), collided (N, T). See scripts/collect_data.py."""
    with np.load(DATA / f"{layout}_{split}.npz") as f:
        return {k: torch.from_numpy(f[k]).to(device) for k in f.files}


def to_float(obs):
    """uint8 (..., H, W, 3) -> float (..., 3, H, W) in [0, 1], the layout conv/ViT layers expect."""
    return obs.movedim(-1, -3).float().div_(255)


def sample_clips(d, batch_size, clip_len=1, generator=None):
    """Random clips of `clip_len` consecutive frames.

    Returns obs (B, L, 3, H, W) float, actions (B, L-1, 2), pos (B, L, 2),
    collided (B, L-1). actions[:, t] takes obs[:, t] to obs[:, t+1].
    clip_len=1 gives single frames (Module 1); longer clips are for Modules 3-4.
    """
    N, T1 = d["pos"].shape[:2]
    dev = d["pos"].device
    i = torch.randint(N, (batch_size,), device=dev, generator=generator)
    t0 = torch.randint(T1 - clip_len + 1, (batch_size,), device=dev, generator=generator)
    t = t0[:, None] + torch.arange(clip_len, device=dev)  # (B, L) frame indices
    i = i[:, None]
    return {
        "obs": to_float(d["obs"][i, t]),
        "actions": d["actions"][i, t[:, :-1]],
        "pos": d["pos"][i, t],
        "collided": d["collided"][i, t[:, :-1]],
    }
