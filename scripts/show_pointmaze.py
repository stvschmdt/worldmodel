"""Visual tour of PointMaze-lite: what the world model actually sees.

Writes to figures/:
  01_layouts.png        - the three layouts at model resolution (64x64)
  02_trajectory.png     - a random-policy trajectory as a filmstrip, with actions
  03_model_view.png     - one frame as a ViT sees it: patches, tokens, a JEPA mask
  04_coverage.png       - where 200 random trajectories go (data coverage)
  05_rollout.gif        - an animated trajectory (upscaled)
"""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from wm.envs.pointmaze import LAYOUTS, PointMaze, PointMazeConfig, collect_trajectory

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
PATCH = 8  # 64x64 image -> 8x8 grid of 8x8-pixel patches = 64 tokens


def upscale(img, k=6):
    return np.kron(img, np.ones((k, k, 1), dtype=img.dtype))


def fig_layouts():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    for ax, name in zip(axes, LAYOUTS):
        env = PointMaze(PointMazeConfig(layout=name), seed=1)
        env.reset(goal=env.sample_free())
        ax.imshow(env.render(), interpolation="nearest")
        ax.set_title(f"{name}  (64x64)")
        ax.axis("off")
    fig.suptitle("PointMaze-lite layouts — red = agent, blue ring = goal", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT / "01_layouts.png", dpi=130)


def fig_trajectory(env, obs, actions, pos):
    idx = np.linspace(0, len(actions) - 1, 10).astype(int)
    fig, axes = plt.subplots(2, 10, figsize=(16, 3.9), gridspec_kw={"height_ratios": [1, 0.35]})
    for j, t in enumerate(idx):
        ax = axes[0, j]
        ax.imshow(obs[t], interpolation="nearest")
        # arrow: the action about to be taken, in pixel units
        x, y = pos[t] * 64 - 0.5
        a = actions[t] * env.cfg.max_step * 64 * 3  # exaggerated x3 for visibility
        ax.annotate("", xy=(x + a[0], y + a[1]), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
        ax.set_title(f"t={t}", fontsize=9)
        ax.axis("off")
        axes[1, j].bar(["ax", "ay"], actions[t], color=["#555", "#999"])
        axes[1, j].set_ylim(-1, 1)
        axes[1, j].axhline(0, color="k", lw=0.5)
        axes[1, j].tick_params(labelsize=7)
    fig.suptitle("A random-policy trajectory: (obs_t, a_t) → obs_t+1   (arrows = action, exaggerated ×3)")
    fig.tight_layout()
    fig.savefig(OUT / "02_trajectory.png", dpi=130)


def sample_jepa_mask(rng, grid=8, n_targets=4):
    """I-JEPA-style masks on the patch grid: a few rectangular target blocks,
    and a context block that excludes them. Returns (context, [targets])."""
    targets = []
    for _ in range(n_targets):
        h, w = rng.integers(2, 4, size=2)
        r, c = rng.integers(0, grid - h + 1), rng.integers(0, grid - w + 1)
        m = np.zeros((grid, grid), bool)
        m[r:r + h, c:c + w] = True
        targets.append(m)
    context = np.ones((grid, grid), bool)
    for m in targets:
        context &= ~m
    return context, targets


def fig_model_view(frame, rng):
    g = frame.shape[0] // PATCH
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))

    axes[0].imshow(frame, interpolation="nearest")
    axes[0].set_title("raw pixels: 64×64×3\n(12,288 numbers)")

    axes[1].imshow(frame, interpolation="nearest")
    for k in range(1, g):
        axes[1].axhline(k * PATCH - 0.5, color="white", lw=0.8)
        axes[1].axvline(k * PATCH - 0.5, color="white", lw=0.8)
    for i in range(g):
        for j in range(g):
            axes[1].text(j * PATCH + 3.5, i * PATCH + 3.5, str(i * g + j), fontsize=6,
                         ha="center", va="center", color="white", alpha=0.8)
    axes[1].set_title(f"ViT view: {g}×{g} = {g*g} patches\n(each 8×8×3 → one token)")

    context, targets = sample_jepa_mask(rng, g)
    overlay = frame.astype(float) / 255
    ctx_img = overlay.copy()
    ctx_up = np.kron(context, np.ones((PATCH, PATCH)))
    ctx_img[ctx_up == 0] *= 0.15
    axes[2].imshow(ctx_img, interpolation="nearest")
    axes[2].set_title("context encoder input\n(target blocks hidden)")

    colors = plt.cm.tab10(np.arange(len(targets)))
    tgt_img = overlay * 0.35
    for m, col in zip(targets, colors):
        up = np.kron(m, np.ones((PATCH, PATCH))).astype(bool)
        tgt_img[up] = 0.5 * overlay[up] + 0.5 * col[:3]
    axes[3].imshow(tgt_img, interpolation="nearest")
    axes[3].set_title("targets: predictor must guess the\nEMBEDDINGS of these blocks, not pixels")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "03_model_view.png", dpi=130)


def fig_coverage(env, n=200, T=100):
    all_pos = np.concatenate([collect_trajectory(env, T)[2] for _ in range(n)])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    axes[0].imshow(env.render(pos=np.array([-1.0, -1.0])), extent=(0, 1, 1, 0))
    for _ in range(12):
        p = collect_trajectory(env, T)[2]
        axes[0].plot(p[:, 0], p[:, 1], lw=1, alpha=0.8)
    axes[0].set_title("12 random trajectories (T=100)")
    axes[1].hist2d(all_pos[:, 0], all_pos[:, 1], bins=48, range=[[0, 1], [0, 1]], cmap="magma", norm=LogNorm())
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")
    axes[1].set_title(f"state coverage, log scale: {n} trajectories\n(random policy piles up in corners)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "04_coverage.png", dpi=130)


def gif_rollout(obs):
    frames = [upscale(o, 5) for o in obs]
    imageio.mimsave(OUT / "05_rollout.gif", frames, duration=0.06, loop=0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    env = PointMaze(PointMazeConfig(layout="four_rooms"), seed=3)
    obs, actions, pos = collect_trajectory(env, T=120)
    print(f"obs {obs.shape} {obs.dtype}, actions {actions.shape}, pos {pos.shape}")

    fig_layouts()
    fig_trajectory(env, obs, actions, pos)
    fig_model_view(obs[40], rng)
    fig_coverage(PointMaze(PointMazeConfig(layout="four_rooms"), seed=7))
    gif_rollout(obs)
    print("wrote", sorted(p.name for p in OUT.iterdir()))
