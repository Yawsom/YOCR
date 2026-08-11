# Next Steps Forward

> Roadmap drafted with [Cursor](https://cursor.com) (AI coding assistant).  
> Credit: Cursor / Grok — August 2026.  
> This file is updated as milestones land — it stays ahead of the repo, not frozen in place.

This is a **public learning repo**. I'm building OCR-related systems from scratch to learn the stack properly. The kNN MNIST classifier (~96% on raw pixels) is **milestone one**: an honest baseline that stays in the repo while newer approaches are added.

The sections below are the planned path for this project. If you're reading along, treat them as the backlog — not promises with dates. Order may change; completed work will be reflected in the tree and called out here over time.

---

## 1. Keep kNN as the benchmark

The kNN implementation is not throwaway code. New models and preprocessing should be compared against the same evaluation protocol:

- same data split
- same accuracy metric
- preferably a confusion matrix, not just a single number

If a fancier model can't beat ~96% on clean MNIST digits, training or evaluation is probably wrong.

---

## 2. Better features before fancier models

Raw 28×28 pixels are brittle (shift, thickness, slant). Planned cheap wins:

- center / normalize each digit
- light deskew
- downsample or PCA to shrink dimensionality

These should help kNN and whatever comes next.

---

## 3. Trainable classifiers

Intended order:

1. **Softmax / logistic regression** on flattened pixels — fast intro to loss + gradients  
2. **Small MLP** (2–3 fully connected layers) — first neural net without convolution  
3. **SVM** (optional) — historically strong on MNIST; useful contrast to neural nets  

Hyperparameters get tuned on a **validation** split. The test set is for a final report, not shopping for scores.

---

## 4. CNNs — the leap for vision OCR

Digits and characters are spatial. A small ConvNet (a few conv + pool blocks → dense → 10-way softmax) is the next major architecture milestone.

On MNIST, a modest CNN should reach the **high 99%s** if trained cleanly. Goals for that stage:

- why local receptive fields beat flat vectors
- overfitting / regularization / data augmentation
- reading training curves

---

## 5. Project hygiene

As the repo grows:

- train / val / test splits  
- tune on val, report test once  
- log loss + accuracy  
- confusion matrices (common MNIST confusions: `4`/`9`, `3`/`5`, etc.)  
- reproducible runs (seed, config, dependency versions)

New approaches should get their own folders (e.g. `MLP_MNIST/`, `CNN_MNIST/`) rather than overwriting the kNN baseline.

---

## 6. Beyond MNIST

MNIST is cleaned and centered. Harder, more OCR-like data planned later:

- **EMNIST** — letters as well as digits  
- **SVHN** — digits in the wild  
- personal scans — crop characters, then classify  

Full document OCR is a pipeline, not one classifier:

**detect → crop / line / word → recognize → post-process** (lexicon / language model)

---

## 7. Practical OCR later

Further out, when the focus shifts to real documents:

- study an existing engine (Tesseract, PaddleOCR) by using and dissecting it  
- build pieces: text detection + CRNN / Transformer recognizer  
- language-model / dictionary post-correction

---

## Short path

```
preprocess → MLP on MNIST → small CNN → error analysis
        → EMNIST or real scans → detection + recognition pipeline
```

Milestone one is done: a measured baseline built from scratch. Everything after that is deliberate iteration — beat the baseline, then raise the difficulty.
