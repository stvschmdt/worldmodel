# JEPA World Models — a hands-on short course

**Capstone goal:** an action-conditioned JEPA world model, learned from pixels,
with a set of task heads, that plans an agent's path through a 2D maze.

**Running environment:** *PointMaze-lite* (`wm/envs/pointmaze.py`) — a dot moving
through rooms, rendered at 64×64. Small enough for fast iteration, rich enough to
test representations, dynamics, and planning. We always know the true state
(x, y), so we can always check what the model has learned.

**Hardware:** DGX Spark (`spark-dc2e`, GB10, 128 GB unified memory), conda env `wm`
(Python 3.12, PyTorch 2.14 + CUDA 13.0). Edit from the Mac via VS Code Remote-SSH.

**How each session works:** 5-minute concept, then you write the core piece
(loss / masking / predictor) while Claude scaffolds, debugs and plots. Every
session ends with one experiment and one figure.

---

## Module 0 — Setup and the mental model
- [x] Conda env `wm` on the Spark (PyTorch + CUDA verified on GB10)
- [x] Project skeleton + git repo at `~/proj/worldmodel`
- [x] PointMaze-lite env + visual tour (`python scripts/show_pointmaze.py`)
- [x] VS Code Remote-SSH into the Spark; Claude Code running there
- [x] Dataset collection script: random trajectories `(obs_t, a_t, obs_t+1, pos_t)` saved to disk
  (`scripts/collect_data.py` → `data/{layout}_{split}.npz`; GPU-resident loader `wm/data.py`)
- [ ] TensorBoard with port forwarding
- **Concepts:** generative vs. joint-embedding world models; why predict in latent
  space; context encoder / target encoder / predictor.
- **Checkpoint:** explain why a naive JEPA collapses to a constant.

## Module 1 — First JEPA on still images (I-JEPA style)
- **Build:** small ViT encoder (8×8 patches → 64 tokens), predictor, block masking
  (context + target blocks), L2 / smooth-L1 loss in embedding space.
- **Technique:** EMA target encoder + stop-gradient.
- **Experiments:** (1) remove EMA → watch collapse (embedding std, effective rank);
  (2) EMA momentum sweep + schedules; (3) masking: random vs. block, mask ratio.
- **Head #1:** frozen **linear probe** → agent (x, y). Our representation-quality meter.
- **Checkpoint:** probe R² > 0.9, and a collapse plot.

## Module 2 — Anti-collapse, the other way
- **Build:** single-encoder regularizers: **VICReg**, **SIGReg** (LeJEPA).
- **Experiments:** EMA vs. VICReg vs. SIGReg on probe accuracy, stability,
  hyperparameter sensitivity; loss weights; embedding dimension.
- **Checkpoint:** comparison table + your opinion.

## Module 3 — Adding time (V-JEPA style)
- **Build:** clip inputs; predictor forecasts future-frame embeddings;
  spatiotemporal (tubelet) masking.
- **Heads:** **velocity probe**; **pixel decoder** on frozen features (visualization only).
- **Checkpoint:** decoded rollouts that look plausible for a few steps.

## Module 4 — Action-conditioned world model (the core)
- **Build:** predictor `f(z_t, a_t) → ẑ_{t+1}`; action injection via concat vs.
  FiLM vs. AdaLN.
- **Techniques:** single-step vs. multi-step rollout loss; teacher forcing vs.
  self-feeding; rollout-length curriculum / scheduled sampling; joint encoder
  training vs. frozen encoder; frame-stack vs. recurrent vs. transformer history.
- **Experiments:** error vs. rollout horizon (1 → 50) per variant.
- **Checkpoint:** 20-step latent rollouts that decode to correct agent positions.

## Module 5 — A zoo of heads
| Head | Task | Teaches |
|---|---|---|
| State probe | z → (x, y) | representation quality |
| Inverse dynamics | (z_t, z_t+1) → a_t | controllability in latent space |
| Goal distance | z → distance-to-goal | value-like signals |
| Collision | z → p(hit wall) | classification, class imbalance |
| Pixel decoder | z → image | interpretability |

- **Techniques:** frozen vs. fine-tuned vs. multi-task joint training with the
  JEPA loss; loss balancing (fixed, uncertainty weighting, GradNorm); do
  auxiliary heads help or hurt the world model?
- **Checkpoint:** table of head effects on probe and rollout metrics.

## Module 6 — Planning in latent space
- **Build:** encode current obs + goal image; plan action sequences with
  **CEM / MPPI**, then **gradient-based** planning through the predictor;
  closed-loop MPC.
- **Experiments:** success vs. horizon, vs. world-model variant, with/without the
  goal-distance head as cost.
- **Checkpoint:** agent reaches image-specified goals across rooms.

## Module 7 — Data and generalization
- **Experiments:** random vs. exploratory vs. expert data; train on some layouts,
  test on unseen ones; add distractors (moving background noise).
- **Build:** pixel-reconstruction world model baseline to compare against.
- **Checkpoint:** evidence for or against JEPA's claim that it ignores
  unpredictable detail.

## Module 8 — Capstone (pick one)
- **A.** Real benchmark: DINO-WM-style frozen DINOv2 / V-JEPA 2 encoder on PushT or MuJoCo.
- **B.** Hierarchical JEPA: a slower latent level for long-horizon planning.
- **C.** Policy learned purely in imagination (Dreamer-style, JEPA latent space).
- **D.** Write-up of your ablation findings.

## Running track — real-world training pipelines
Woven through the modules: each practice is introduced where it first earns its
keep, in the small version, alongside what it looks like at scale.

| Where | Practice | At our scale → at real scale |
|---|---|---|
| M0 | Reproducible data: seeds per shard, `meta.json`, sanity figures | `.npz` in RAM → sharded WebDataset / Parquet / zarr, streamed |
| M0 | Data loading | whole dataset on GPU → DataLoader workers, pinned memory, prefetch |
| M1 | Config system + run directories (config, git hash, seed saved per run) | dataclass → Hydra / YAML configs |
| M1 | Logging and metrics: loss, grad norm, LR, throughput, embedding stats | TensorBoard → W&B / MLflow |
| M1 | Checkpoint + resume (model, optimizer, EMA, RNG state, step) | local files → fault-tolerant, preemptible jobs |
| M1 | Optimization hygiene: AdamW, warmup + cosine, grad clipping, weight decay rules | same, plus µP / LR scaling rules |
| M1 | Speed: bf16 autocast, `torch.compile`, measuring samples/s and MFU | same, plus `torch.profiler` bottleneck hunts |
| M2 | Sweeps and ablation discipline: several seeds, error bars, one change at a time | loops → sweep schedulers |
| M4 | Eval harness separate from training; fixed val sets; regression checks | same, run by CI on checkpoints |
| M5 | Multi-task loss balancing; monitoring per-head gradients | same |
| M8 | Scaling out: DDP / FSDP, gradient accumulation (concepts; one GPU here) | multi-node clusters |

---

## Reading list (in order of need)
1. VICReg — Bardes, Ponce, LeCun (2021)
2. I-JEPA — Assran et al. (2023)
3. V-JEPA — Bardes et al. (2024)
4. DINO-WM — Zhou et al. (2024)
5. PLDM / *Learning from Reward-Free Offline Data* — Sobal et al. (2025)
6. V-JEPA 2 (2025) — especially the action-conditioned V-JEPA 2-AC part
7. LeJEPA / SIGReg — Balestriero & LeCun (2025)
