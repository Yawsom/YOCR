# Next Steps Forward

> Roadmap drafted with [Cursor](https://cursor.com) (AI coding assistant).  
> Credit: Cursor / Grok — August 2026.  
> This file is updated as milestones land — it stays ahead of the repo, not frozen in place.

This is a **public learning repo**. I'm building OCR-related systems from scratch to learn the stack properly.

**Done so far**

- **Milestone 1 — kNN:** raw-pixel Euclidean kNN on MNIST, ~**96.4%** test accuracy. Stays in the repo as the benchmark.  
- **Milestone 2 — MLP:** from-scratch network (`784→100→100→10`, sigmoid, MSE, backprop, SGD), **94.42%** test accuracy.

The MLP is a real trained model and much faster than kNN at inference, but it does not yet beat the kNN baseline — expected for MSE + sigmoid on 10-way digits. Next work should either close that gap (better loss / training) or jump to models that can.

The sections below are the planned path. If you're reading along, treat them as the backlog — not promises with dates.

---

## 1. Keep earlier models as benchmarks

kNN and the MLP are not throwaway code. New models and preprocessing should be compared against the same evaluation protocol:

- same data split
- same accuracy metric
- preferably a confusion matrix, not just a single number

kNN is still the number to beat (~96.4%). If a fancier model lands far below that on clean MNIST, training or evaluation is probably wrong.

---

## 2. Better features / training for the models we have

Raw 28×28 pixels are brittle (shift, thickness, slant). Cheap wins that would help both kNN and the MLP:

- center / normalize each digit more carefully  
- light deskew  
- downsample or PCA to shrink dimensionality  

MLP-specific upgrades before (or alongside) a CNN:

- **softmax + cross-entropy** instead of sigmoid + MSE  
- a validation split, learning-rate / epoch sweeps  
- print loss and accuracy each epoch  

---

## 3. Trainable classifiers (remaining)

1. ~~**Small MLP**~~ — done (sigmoid + MSE)  
2. **Softmax / logistic regression** — still useful as a linear baseline, or as the MLP’s output layer  
3. **SVM** (optional) — historically strong on MNIST; useful contrast to neural nets  

Hyperparameters get tuned on a **validation** split. The test set is for a final report, not shopping for scores.

---

## 4. CNNs — the leap for vision OCR

This is the next major architecture milestone. Digits and characters are spatial; a small ConvNet (a few conv + pool blocks → dense → 10-way softmax) should beat both kNN and the MLP on MNIST, often into the **high 99%s** if trained cleanly.

Goals for that stage:

- why local receptive fields beat flat vectors  
- overfitting / regularization / data augmentation  
- reading training curves  

New code should live in its own folder (e.g. `CNN_MNIST/`).

---

## 5. Project hygiene

As the repo grows:

- train / val / test splits  
- tune on val, report test once  
- log loss + accuracy  
- confusion matrices (common MNIST confusions: `4`/`9`, `3`/`5`, etc.)  
- reproducible runs (seed, config, dependency versions)

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
kNN baseline ✓ → MLP ✓ → (softmax/CE or preprocess) → small CNN → error analysis
        → EMNIST or real scans → detection + recognition pipeline
```

Two measured from-scratch models are in the tree. Next is to beat kNN on purpose, then raise the difficulty.
