# worldmodel — JEPA world models short course

This repo is a hands-on course: the user is learning JEPA-based world models by
building one from scratch with Claude, focusing on **training techniques** and
**attaching different heads for different tasks**. The syllabus and progress
checkboxes live in `CURRICULUM.md` — read it at the start of a session and tick
boxes as milestones are completed.

## How we work
- Teaching first. Each session: short concept explanation → the user writes the
  core piece (loss, masking, predictor, etc.) with Claude scaffolding, debugging
  and plotting → one experiment → one figure.
- Don't write the core learning pieces for the user unless asked; scaffolding,
  data plumbing, logging and plotting code are fine to write.
- Show, don't just tell: make figures (saved to `figures/`) so the user can build
  intuition from images. View figures yourself before presenting them.
- Keep models and data small (64×64 images) so runs take minutes and sweeps are cheap.

## Environment
- Machine: DGX Spark `spark-dc2e` (NVIDIA GB10, sm_121, ~119 GB unified memory,
  aarch64, Ubuntu 24.04). The user edits from a Mac (8 GB RAM) via VS Code
  Remote-SSH — never run training on the Mac.
- Python: `conda activate wm` (Python 3.12, PyTorch 2.14+cu130, bf16 ≈ 99 TFLOPS).
  The env sets `PYTHONNOUSERSITE=1` so `~/.local` packages don't leak in.
  Don't install into the `base` or `ml` envs; those belong to other projects.
- Package installed editable (`pip install -e .`), import as `wm.*`.
- `data/`, `runs/`, `checkpoints/` are gitignored.

## Layout
- `wm/envs/pointmaze.py` — PointMaze-lite: agent disk in the unit square, walls as
  rectangles (`open`, `two_rooms`, `four_rooms`), actions in [-1,1]² scaled by
  `max_step`, wall sliding on collision, 64×64 RGB render with anti-aliased agent
  (encodes sub-pixel position). `collect_trajectory` uses smoothed random actions.
- `scripts/show_pointmaze.py` — visual tour → `figures/01..05`.
- `scripts/collect_data.py` — random trajectories → `data/{layout}_{split}.npz`
  (2000 train / 200 val × T=64 per layout, ~1 min) + `figures/06_dataset_*.png`.
- `wm/data.py` — `load()` puts a split on the GPU; `sample_clips(d, B, L)` samples
  batches there (L=1 single frames; L>1 clips with the L-1 actions between them).
- `wm/viz.py` — `episode_frames()` / `save_gif()`: annotated episode GIFs (trace,
  action arrow, red border on wall hits); reuse for model rollouts.
- `scripts/watch_episode.py` — stored episode → `runs/episodes/*.gif`
  (`--index`, `--pick random|bumpiest|farthest`, `-n`, `--fps`).

## Status (2026-09-28)
Module 0 in progress. Done: env, project skeleton, PointMaze-lite, visual tour,
dataset + loader. Next: walk the user through the data script, TensorBoard with
port forwarding, then Module 1 (I-JEPA on still frames + linear (x, y) probe).
Known data property: random policy over-visits corners and under-visits doorways
(see `figures/04_coverage.png`) — relevant in Modules 6–7. ~50% of four_rooms
steps collide (random walks hug walls), ~12% are fully stuck.
The user also wants to learn real-world training-pipeline practices — see the
running track in CURRICULUM.md; teach each where it first earns its keep.
Activate the env in scripts with `eval "$(conda shell.bash hook)" && conda activate wm`
(miniforge3).

Git: repo-local identity (stvschmdt / quantdata@gmail.com); remote `origin` =
github.com:stvschmdt/worldmodel (SSH). Push at the end of each session.
