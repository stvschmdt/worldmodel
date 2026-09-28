"""Collect random-policy trajectories from PointMaze-lite and save them to data/.

One file per (layout, split):  data/{layout}_{split}.npz  with arrays
  obs       (N, T+1, 64, 64, 3) uint8    frames; obs[:, t+1] is the result of actions[:, t]
  actions   (N, T, 2)           float32  in [-1, 1]
  pos       (N, T+1, 2)         float32  true agent (x, y), the probe target
  collided  (N, T)              bool     did actions[:, t] hit a wall
plus data/meta.json recording how the data was made.

We store whole trajectories, not (obs_t, a_t, obs_t+1) tuples: tuples store every
frame twice, and trajectories let later modules cut clips of any length.

    python scripts/collect_data.py                 # all layouts, default sizes
    python scripts/collect_data.py --layouts four_rooms --n-train 500
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wm.envs.pointmaze import LAYOUTS, PointMaze, PointMazeConfig, collect_trajectory

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figures"
SPLITS = ("train", "val")


def collect(layout, split, n, T, seed):
    # One seed per (layout, split) so each file is reproducible on its own, and
    # train and val are independent draws of start positions and actions.
    ss = np.random.SeedSequence([seed, list(LAYOUTS).index(layout), SPLITS.index(split)])
    env = PointMaze(PointMazeConfig(layout=layout), seed=ss)
    out = {"obs": [], "actions": [], "pos": [], "collided": []}
    for _ in range(n):
        for k, v in zip(out, collect_trajectory(env, T)):
            out[k].append(v)
    return {k: np.stack(v) for k, v in out.items()}


def summarize(name, d):
    gb = sum(v.nbytes for v in d.values()) / 1e9
    step = np.linalg.norm(np.diff(d["pos"], axis=1), axis=-1)
    print(f"  {name:20s} {d['obs'].shape[0]:5d} traj  {gb:5.2f} GB  "
          f"collided {d['collided'].mean():5.1%}  "
          f"mean |step| {step.mean():.4f}  stuck (|step|<1e-4) {(step < 1e-4).mean():5.1%}")


def sanity_figure(layout, d, path):
    """Random frames with the *stored* pos drawn on top (checks obs/pos alignment),
    plus where the training data goes."""
    rng = np.random.default_rng(0)
    N, T1 = d["pos"].shape[:2]
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.15, 2.2])
    for j in range(8):
        i, t = rng.integers(N), rng.integers(T1)
        ax = fig.add_subplot(gs[j // 4, j % 4])
        ax.imshow(d["obs"][i, t], extent=(0, 1, 1, 0), interpolation="nearest")
        ax.plot(*d["pos"][i, t], "+", color="black", ms=9, mew=1.2)
        ax.set_title(f"traj {i}, t={t}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(gs[:, 5])
    p = d["pos"].reshape(-1, 2)
    h = ax.hist2d(p[:, 0], p[:, 1], bins=64, range=[[0, 1], [0, 1]], cmap="Blues")
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(f"{layout} train: visited positions ({len(p):,} frames)", fontsize=9)
    fig.colorbar(h[3], ax=ax, fraction=0.046, label="frames per bin")
    fig.suptitle("Dataset check — black + is the stored pos; it should sit on the red agent")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=list(LAYOUTS))
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--T", type=int, default=64, help="steps per trajectory (T+1 frames)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    for layout in args.layouts:
        for split, n in (("train", args.n_train), ("val", args.n_val)):
            t0 = time.time()
            d = collect(layout, split, n, args.T, args.seed)
            np.savez(DATA / f"{layout}_{split}.npz", **d)  # uncompressed: loads fast
            summarize(f"{layout}_{split}", d)
            print(f"  {'':20s} ({time.time() - t0:.1f}s)")
            if split == "train":
                sanity_figure(layout, d, FIGS / f"06_dataset_{layout}.png")

    cfg = PointMazeConfig()
    meta = {**vars(args), "radius": cfg.radius, "max_step": cfg.max_step, "size": cfg.size,
            "policy": "random_actions(smooth=0.8, scale=0.6)"}
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
