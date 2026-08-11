# KNN-OCR

A **public learning project** for building OCR-style systems from scratch. I'm iterating on this over time — starting with handwritten digit recognition on MNIST, then moving toward stronger models and fuller OCR pipelines.

The first milestone is a from-scratch **k-nearest neighbors (kNN)** classifier: reading the MNIST IDX format, turning images into feature vectors, computing distances, and voting on a label. Later work (preprocessing, neural nets, harder datasets, document OCR) will be added alongside this baseline rather than replacing it.

This is not a finished product. Expect the layout, models, and docs to keep changing. [NEXT_STEPS.md](NEXT_STEPS.md) is the living roadmap.

## Project layout

```
KNN-OCR/
├── KNN_MNIST/
│   ├── data/                 # MNIST IDX files
│   ├── ocr.py                # Pure-Python kNN
│   └── ocr_efficient.py      # Faster kNN (NumPy + squared distance + top-k heap)
├── NEXT_STEPS.md             # Living roadmap for future iterations
├── README.md
└── .gitignore
```

## Requirements

- Python 3.10+
- NumPy (for `ocr_efficient.py` only)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
```

## Data

Place the standard MNIST IDX files under `KNN_MNIST/data/`:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Usage

Run from the **repo root** (paths in the scripts assume that):

```bash
# Pure Python baseline (slow on full MNIST)
python KNN_MNIST/ocr.py

# Faster version
python KNN_MNIST/ocr_efficient.py
```

Both scripts load the training set, classify the test set with kNN (`k=5` by default), and print accuracy.

For quicker experiments, pass a sample limit into `read_images` / `read_labels` inside `main()`.

## Current results (kNN milestone)

On full MNIST train (~60k) and the full test set (10k), the efficient kNN setup reached about **96.4%** accuracy — a normal result for raw-pixel Euclidean kNN, and the baseline future models in this repo should beat.

## What's next?

See [NEXT_STEPS.md](NEXT_STEPS.md) for the planned learning path and backlog for this project.
