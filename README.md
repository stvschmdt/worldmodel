# worldmodel

JEPA world models, built from scratch. See [CURRICULUM.md](CURRICULUM.md).

## Setup — DGX Spark (training)

```bash
conda activate wm           # Python 3.12, PyTorch 2.14 + CUDA 13.0
pip install -e ".[train]"
python scripts/show_pointmaze.py   # visual tour -> figures/
python scripts/collect_data.py     # datasets -> data/ (~1 min, 5 GB)
```

## Setup — Mac (env, data, viewing; no training)

The Mac has 8 GB of RAM, so it gets a light env without PyTorch, and makes its
own small dataset instead of copying the Spark's 5 GB.

```bash
# once: install Miniforge if you don't have conda  (brew install miniforge)
conda create -n wm python=3.12 -y
conda activate wm
git clone git@github.com:stvschmdt/worldmodel.git && cd worldmodel
pip install -e .

python scripts/collect_data.py --n-train 200 --n-val 50   # ~5 s, ~0.6 GB
python scripts/watch_episode.py --pick farthest            # -> runs/episodes/*.gif
open runs/episodes/*.gif
```

Datasets are seeded, so the Mac's episode #i is the same as the Spark's #i
(for i < 200).
