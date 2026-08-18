# KNN-OCR

A **public learning project** for building OCR-style systems from scratch. I'm iterating on this over time — starting with handwritten digit recognition on MNIST, then moving toward stronger models and fuller OCR pipelines.

Milestones so far are implemented from scratch (NumPy only for the faster paths) and kept side by side rather than replacing each other:

1. **k-nearest neighbors** — IDX loading, flattened pixels, distances, majority vote  
2. **Multilayer perceptron** — sigmoid layers, MSE loss, backprop, SGD  

This is not a finished product. Expect the layout, models, and docs to keep changing. [NEXT_STEPS.md](NEXT_STEPS.md) is the living roadmap.

## Project layout

```
KNN-OCR/
├── MNIST_data/               # Shared MNIST IDX files
├── KNN_MNIST/
│   ├── ocr.py                # Pure-Python kNN
│   └── ocr_efficient.py      # Faster kNN (NumPy + squared distance + top-k heap)
├── MLP_MNIST/
│   └── MLP_OCR.py            # From-scratch MLP (sigmoid + MSE + backprop)
├── NEXT_STEPS.md
├── README.md
└── .gitignore
```

## Requirements

- Python 3.10+
- NumPy (for `ocr_efficient.py` and `MLP_OCR.py`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
```

## Data

Place the standard MNIST IDX files under `MNIST_data/`:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Usage

Run from the **repo root** (paths in the scripts assume that):

```bash
# Pure Python kNN (slow on full MNIST)
python KNN_MNIST/ocr.py

# Faster kNN
python KNN_MNIST/ocr_efficient.py

# From-scratch MLP
python MLP_MNIST/MLP_OCR.py
```

kNN scripts print test accuracy after classifying with `k=5` by default. The MLP trains 10 epochs of SGD on the full training set, then prints test accuracy.

For quicker experiments, pass a sample limit into `read_images` / `read_labels`.

## Current results

Full MNIST train (~60k) and full test set (10k), raw flattened pixels:

| Model | Accuracy | Notes |
|-------|----------|--------|
| kNN (`k=5`, Euclidean) | **~96.4%** | Current strongest baseline |
| MLP (`784→100→100→10`, sigmoid, MSE) | **94.42%** | Trains in a few minutes |

The MLP is in the right ballpark for this architecture and loss. A CNN (or even softmax + cross-entropy) is the usual way to push past kNN on pixels.

## What's next?

See [NEXT_STEPS.md](NEXT_STEPS.md) for the planned learning path and backlog for this project.
