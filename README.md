# Predicting Drug-Drug Interactions Using Graph Neural Networks

**ECE 57000 — Artificial Intelligence | Course Project**  
**Author:** Oulaiphone  
**Purdue University, Spring 2026**

---

## Project Overview

This project implements and compares two Graph Neural Network (GNN) architectures — **GraphSAGE** and **Graph Attention Networks (GAT)** — for predicting drug-drug interactions (DDIs) using the [OGB ogbl-ddi](https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi) benchmark.

### Key Results

| Method | Val Hits@20 | Test Hits@20 | Training Time |
|---|---|---|---|
| GraphSAGE (CP1 baseline) | 12.26% | — | — |
| **GraphSAGE (tuned)** | **18.54%** | **12.99%** | ~1.8 hr |
| GAT (4 heads) | 3.13% | 0.54% | ~21 hr |

---

## Repository Structure

```
.
├── train_ddi.py          # Main training script (models, training, evaluation)
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── results/              # Output directory (created at runtime)
    ├── loss_curves.png
    ├── hits20_progression.png
    └── results.json
```

---

## Code Authorship

### Written by me (Oulaiphone):
- **`train_ddi.py`** — Full file, including:
  - `DDI_GAT` class (lines 55–70): GAT encoder implementation
  - `run_experiment()` function (lines 120–185): Fair comparison framework with Xavier re-initialization
  - `plot_training_curves()` function (lines 192–225): Visualization utilities
  - `print_comparison_table()` function (lines 228–235): Results formatting
  - `main()` function (lines 242–340): CLI argument parsing and experiment orchestration
  - All docstrings, comments, and argument parsing

### Adapted from OGB examples:
- **`DDI_GraphSAGE` class** (lines 33–48): Adapted from [OGB ogbl-ddi example](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/ddi). Modified to parameterize dropout and add docstrings.
- **`train()` function** (lines 78–100): Adapted from OGB example training loop. Modified for batched training with configurable batch size.
- **`evaluate()` function** (lines 103–120): Adapted from OGB example evaluation. Restructured for reusability with the experiment runner.

### External libraries used:
- PyTorch, PyTorch Geometric, OGB (for data loading and evaluation)
- matplotlib (for plotting)

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, you may need to install with specific CUDA version. For CPU-only:

```bash
pip install torch-geometric
```

For GPU (CUDA 11.8):

```bash
pip install torch-geometric --extra-index-url https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## Running the Project

### Quick Start (both models, default settings)

```bash
python train_ddi.py
```

This will:
1. Automatically download the OGB ogbl-ddi dataset (~5MB, first run only)
2. Train GraphSAGE for 500 epochs
3. Train GAT for 500 epochs
4. Save plots and results to `results/`

### Run Only GraphSAGE (faster, ~1.8 hours on CPU)

```bash
python train_ddi.py --models graphsage
```

### Run Only GAT (~21 hours on CPU)

```bash
python train_ddi.py --models gat
```

### Custom Hyperparameters

```bash
python train_ddi.py \
    --hidden_dim 512 \
    --lr 0.005 \
    --epochs 500 \
    --batch_size 65536 \
    --dropout 0.3 \
    --eval_every 50 \
    --seed 42 \
    --output_dir results
```

### Reproduce CP1 Baseline

```bash
python train_ddi.py --models graphsage --hidden_dim 256 --lr 0.001 --epochs 200
```

---

## Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--hidden_dim` | 512 | Embedding and hidden layer dimension |
| `--lr` | 0.005 | Learning rate for Adam optimizer |
| `--epochs` | 500 | Number of training epochs |
| `--batch_size` | 65536 | Edges per training batch |
| `--dropout` | 0.3 | Dropout rate |
| `--eval_every` | 50 | Evaluate Hits@20 every N epochs |
| `--models` | both | Which model(s): graphsage, gat, or both |
| `--output_dir` | results | Directory for output files |
| `--seed` | 42 | Random seed for reproducibility |

---

## Dataset

The **OGB ogbl-ddi** dataset is automatically downloaded on first run.

- **Nodes:** 4,267 drugs
- **Edges:** 1,334,889 known drug-drug interactions
- **Task:** Link prediction (predict unseen DDIs)
- **Metric:** Hits@20 (fraction of true positives in top 20 predictions)
- **Split:** Standard OGB train/validation/test split

---

## Expected Output

After training completes, the `results/` directory will contain:

- `loss_curves.png` — Training loss over epochs for all models
- `hits20_progression.png` — Validation Hits@20 over training
- `results.json` — Numerical results summary

Console output will show a comparison table:

```
======================================================================
Method                    Val Hits@20  Test Hits@20   Time (hr)
----------------------------------------------------------------------
GraphSAGE (tuned)              18.54%        12.99%       1.80
GAT (4 heads)                   3.13%         0.54%      21.00
======================================================================
```

**Note:** Results may vary slightly due to random initialization and negative sampling. The relative ordering (GraphSAGE >> GAT) is consistent across seeds.

---

## LLM Acknowledgment

Claude (Anthropic) was used to assist with:
- Drafting and editing the term paper
- Formatting code documentation and README

All experimental design, model implementation, training, and analysis were conducted by the author.
