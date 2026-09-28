"""Watch a stored episode as an annotated GIF (open it in VS Code to play it).

    python scripts/watch_episode.py                        # random four_rooms train episode
    python scripts/watch_episode.py --index 42 --fps 5
    python scripts/watch_episode.py --layout two_rooms --pick farthest
    python scripts/watch_episode.py --pick bumpiest -n 3    # three episodes

--pick: random | bumpiest (most wall hits) | farthest (largest start-to-end distance)
Writes runs/episodes/{layout}_{split}_{index}.gif (runs/ is gitignored).
"""

import argparse
from pathlib import Path

import numpy as np

from wm.viz import episode_frames, save_gif

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="four_rooms")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, nargs="*", help="episode indices (overrides --pick)")
    ap.add_argument("--pick", default="random", choices=["random", "bumpiest", "farthest"])
    ap.add_argument("-n", type=int, default=1, help="how many episodes to pick")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=int, default=6)
    args = ap.parse_args()

    with np.load(ROOT / "data" / f"{args.layout}_{args.split}.npz") as f:
        actions, pos, collided = f["actions"], f["pos"], f["collided"]
        if args.index:
            idx = args.index
        elif args.pick == "random":
            idx = np.random.default_rng().choice(len(pos), args.n, replace=False).tolist()
        else:
            score = collided.sum(1) if args.pick == "bumpiest" else np.linalg.norm(pos[:, -1] - pos[:, 0], axis=1)
            idx = np.argsort(-score)[: args.n].tolist()
        obs = f["obs"][idx]  # only the chosen episodes stay in memory

    out = ROOT / "runs" / "episodes"
    out.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(idx):
        frames = episode_frames(obs[k], actions[i], pos[i], collided[i], scale=args.scale,
                                title=f"{args.layout}/{args.split} #{i}")
        path = out / f"{args.layout}_{args.split}_{i}.gif"
        save_gif(frames, path, fps=args.fps)
        print(f"{path.relative_to(ROOT)}   bumps {collided[i].sum()}/{len(collided[i])}  "
              f"start→end {np.linalg.norm(pos[i, -1] - pos[i, 0]):.2f}")


if __name__ == "__main__":
    main()
