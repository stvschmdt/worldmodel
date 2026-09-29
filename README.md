# worldmodel

JEPA world models, built from scratch. See [CURRICULUM.md](CURRICULUM.md).

## Setup (DGX Spark)

```bash
conda activate wm           # Python 3.12, PyTorch 2.14 + CUDA 13.0
pip install -e ".[train]"
python scripts/show_pointmaze.py   # visual tour -> figures/
python scripts/collect_data.py     # datasets -> data/ (~1 min, 5 GB)
```

Everything runs on the Spark; the Mac is only the VS Code Remote-SSH client.
