# Deep Learning for Perception – Assignment No. 1
### Building, Breaking and Fixing a Neural Network


## Overview

This repository contains the notebook for Assignment No. 1, which builds a
feedforward neural network on the Fashion-MNIST dataset end to end: implementing
backpropagation from scratch, studying activations/loss functions/optimisers,
deliberately overfitting the model, and then repairing it using regularisation
and hyperparameter tuning.

## Repository Contents

| File | Description |
|---|---|
| `DL_ASS01_23F_0572_0672.ipynb` | Main notebook containing all 7 parts of the assignment, with visible outputs |
| `DLP_Assignment01_Results_Summary.docx` | One-page results summary (final test score, configuration, most effective change) |
| `README.md` | This file |

## Dataset

- **Name:** Fashion-MNIST
- **Source:** [Kaggle – zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- **Files used:** `fashion-mnist_train.csv`, `fashion-mnist_test.csv`

## Environment

- **Platform:** [Kaggle Notebooks](https://www.kaggle.com/)
- **Accelerator:** GPU T4 x2
- **Language:** Python 3

### Dependencies

```
numpy
pandas
scikit-learn
matplotlib
seaborn
torch
torchvision
Pillow
```

Install locally with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision Pillow
```

## Reproducing the Results

1. **Get the data.** Download the Fashion-MNIST CSV files from the Kaggle
   dataset link above.
2. **Set the data path.** The notebook reads the CSVs from
   `/kaggle/input/datasets/zalando-research/fashionmnist/`. If running outside
   Kaggle, update this path to wherever the CSVs are stored locally, or add
   the dataset to a new Kaggle notebook as an input.
3. **Run on Kaggle with a GPU T4 x2 accelerator** (or any CUDA-capable GPU
   locally) for reasonable runtime, especially for Part 5 (deep, oversized
   network) and Part 7 (12 configs × 5-fold cross-validation search).
4. **Run all cells in order, top to bottom.** Later parts reuse variables,
   models and helper functions defined earlier in the notebook.
5. **Random seed.** A fixed seed (`SEED = 42`) is set once near the top of
   the notebook and used for the train/validation split and for all
   subsequent random operations, so results should match on re-run.

## Notebook Structure

| Part | Section | What it does |
|---|---|---|
| Setup | Environment Setup | Load, normalise and split Fashion-MNIST (80/20 train/val), report class distribution |
| 1 | Backpropagation From Scratch | Two-layer MLP in NumPy only; gradient check against PyTorch |
| 2 | Baseline Model and Activation Study | Two-hidden-layer network with sigmoid, tanh, ReLU, leaky ReLU; vanishing-gradient and dead-ReLU analysis |
| 3 | Loss Functions | Cross-entropy vs. MSE for classification; separate tabular regression task |
| 4 | Optimiser Comparison | SGD, SGD+momentum, RMSProp, Adam — shared and tuned learning rates |
| 5 | Forcing Overfitting | Oversized network on 2,000 samples to create a large generalisation gap |
| 6 | Regularisation Study | L2, L1, dropout, batch norm, early stopping, data augmentation, more data |
| 7 | Hyperparameter Tuning | Random search with 5-fold cross-validation, final retraining and test evaluation |

## Final Result

- **Test accuracy:** 88.95% (macro precision 0.8881, macro recall 0.8895, macro F1 0.8881)
- **Improvement over Part 2 baseline:** +2.01 percentage points (86.94% → 88.95%)
- **Final configuration:** hidden width 512, dropout 0.5, learning rate 0.001 (Adam), selected via random search + 5-fold cross-validation, retrained on the full training set
- **Most effective single change:** Dropout (rate 0.6) reduced the generalisation gap from 14.55 to 1.54 percentage points on the overfitted model from Part 5


## Group Members

- 23F-0572
- 23F-0672
