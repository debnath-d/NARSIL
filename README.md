# Efficiently Solving the TSP with Non-Autoregressive Self Improvement Learning

This repository contains the implementation code for our IJCNN 2026 paper "Efficiently Solving the TSP with Non-Autoregressive Self Improvement Learning".

# Install python environment

1. Install `uv`. Official documentation: https://docs.astral.sh/uv/getting-started/installation/
1. Install python 3.13:
    ```bash
    uv python install 3.13
    ```
1. Install virtual environment using `uv`:
    ```bash
    uv sync
    ```
    
# Configurations

Configuration files are given in the `config` directory in json format.

# Train

To train the model on multiple GPUs using [torchrun (Elastic Launch)](https://docs.pytorch.org/docs/stable/elastic/run.html):

```bash
uv run torchrun --standalone --nproc-per-node=auto train.py
```

To train the model using on single GPU:

```bash
uv run train.py
```

# Test

Test results in the paper are reported on single GPU:

```bash
uv run test.py
```