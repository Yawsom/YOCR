# Next Steps Forward

> Roadmap drafted with [Cursor](https://cursor.com) (AI coding assistant).  
> Credit: Cursor / Grok — September 2026.  
> This file is updated as milestones land — it stays ahead of the repo, not frozen in place.

This is a **public learning repo**. I'm building OCR-related systems from scratch to learn the stack properly.

**Done so far**

- **Milestone 1 — kNN:** raw-pixel Euclidean kNN on MNIST, ~**96.4%** test accuracy. Stays in the repo as the benchmark.  
- **Milestone 2 — MLP:** from-scratch network (`784→100→100→10`, sigmoid, MSE, backprop, SGD), **94.42%** test accuracy.

The MLP is paused. Softmax + cross-entropy will come back after the RNN, not before. The MLP is a real trained model and much faster than kNN at inference, but it does not yet beat the kNN baseline — expected for MSE + sigmoid on 10-way digits.

**Next up — Milestone 3: a from-scratch RNN on sequential MNIST.** Flattened MNIST is not a sequence. Reading each image **column by column** is: 28 timesteps of 28-pixel vectors, then classify. Same files, same accuracy metric, same train/test split. That scan direction is also how later CRNN OCR treats width as time.

The sections below are the planned path. If you're reading along, treat them as the backlog — not promises with dates.

---

## 1. Keep earlier models as benchmarks

kNN and the MLP are not throwaway code. New models should be compared against the same evaluation protocol:

- same data split
- same accuracy metric
- preferably a confusion matrix, not just a single number

kNN is still the number to beat (~96.4%). A vanilla RNN on sequential MNIST is not expected to crush that on the first try — if it lands far below the MLP (~94%), training or BPTT is probably wrong.

---

## 2. Next: from-scratch RNN (sequential MNIST)

New code should live in its own folder (e.g. `RNN_MNIST/`). NumPy only, same spirit as the MLP: explicit forward pass, cached activations, backprop.

**Task.** Many-to-one classification. Each MNIST image is a sequence of **28 columns**, each a 28-dim vector. The last hidden state maps to 10 digit scores. Reuse the existing IDX files under `MNIST_data/`.

**First implementation (the milestone):**

- a synthetic sequence task first (parity, delayed copy, or adding short digit strings) so BPTT can be checked on data where the answer is known and `T` is small
- vanilla RNN cell: `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)` (sigmoid is fine to start if that matches the MLP, then switch)
- unroll over `T=28`, backprop through time
- last hidden state → linear output layer → 10-way prediction
- SGD, same train / test split as the MLP, report test accuracy

Keep the first version small: one hidden size (~64–128), no LSTM yet, no softmax + CE yet. The goal is a working recurrence and a number on the same test set.

---

## 3. After the RNN trains: flesh it out

Once the vanilla cell is learning, the model is not done. In roughly this order:

**Training / evaluation (same hygiene the MLP still needs)**

- softmax + cross-entropy instead of sigmoid + MSE (do this here, then reuse it when returning to the MLP)
- a validation split; tune hidden size / learning rate / epochs on val, report test once
- print loss and accuracy each epoch
- confusion matrix (expect the usual `4`/`9`, `3`/`5` confusions if the sequence model is actually reading shape)

**RNN-specific upgrades**

- **vanishing gradients:** compare column-wise (`T=28`) vs pixel-by-pixel (`T=784`). Vanilla RNNs should struggle on the long version; that is the point
- **LSTM or GRU** on the same sequential MNIST loader — same split, same metric, so the gate vs vanilla gap is measurable
- gradient clipping, better init, maybe a bidirectional pass (left-to-right and right-to-left columns)

**Optional sequence tasks (not MNIST accuracy)**

- tiny Shakespeare / next-character prediction — a real language sequence, different metric (next-char accuracy / samples)
- generated adding (`"035+047"` → `"082"`) if seq2seq is the next learning goal

None of these replace sequential MNIST as the OCR-adjacent benchmark. They prove the recurrence generalizes past digit columns.

---

## 4. Better features / paused MLP work

Raw 28×28 pixels are brittle (shift, thickness, slant). Cheap wins that would help kNN, the MLP, and the RNN:

- center / normalize each digit more carefully
- light deskew
- downsample or PCA to shrink dimensionality

**Paused — return after the RNN is in place:**

- MLP **softmax + cross-entropy** (share the loss with the RNN output head)
- MLP validation split, learning-rate / epoch sweeps, epoch logging

---

## 5. Trainable classifiers (remaining)

1. ~~**Small MLP**~~ — done (sigmoid + MSE); softmax + CE still pending  
2. **RNN** — next (vanilla, sequential MNIST); LSTM / GRU after it trains  
3. **Softmax / logistic regression** — still useful as a linear baseline, or as the output layer on the MLP and RNN  
4. **SVM** (optional) — historically strong on MNIST; useful contrast to neural nets  

Hyperparameters get tuned on a **validation** split. The test set is for a final report, not shopping for scores.

---

## 6. CNNs — the leap for vision OCR

After the RNN is a real baseline, digits and characters are still spatial. A small ConvNet (a few conv + pool blocks → dense → 10-way softmax) should beat kNN, the MLP, and a vanilla RNN on MNIST, often into the **high 99%s** if trained cleanly.

Goals for that stage:

- why local receptive fields beat both flat vectors and a scanline RNN
- overfitting / regularization / data augmentation
- reading training curves

New code should live in its own folder (e.g. `CNN_MNIST/`).

The longer OCR path is **CRNN**: conv features over the image width, then the RNN over that sequence. Sequential MNIST is the stripped-down version of that idea.

---

## 7. Project hygiene

As the repo grows:

- train / val / test splits
- tune on val, report test once
- log loss + accuracy
- confusion matrices (common MNIST confusions: `4`/`9`, `3`/`5`, etc.)
- reproducible runs (seed, config, dependency versions)

---

## 8. Beyond MNIST

MNIST is cleaned and centered. Harder, more OCR-like data planned later:

- **EMNIST** — letters as well as digits
- **SVHN** — digits in the wild
- personal scans — crop characters, then classify

Full document OCR is a pipeline, not one classifier:

**detect → crop / line / word → recognize → post-process** (lexicon / language model)

---

## 9. Practical OCR later

Further out, when the focus shifts to real documents:

- study an existing engine (Tesseract, PaddleOCR) by using and dissecting it
- build pieces: text detection + CRNN / Transformer recognizer
- language-model / dictionary post-correction

---

## Short path

```
kNN baseline ✓ → MLP ✓ → vanilla RNN on sequential MNIST
        → softmax/CE + val logging → LSTM/GRU (and T=28 vs T=784)
        → (MLP softmax/CE) → small CNN → CRNN
        → EMNIST or real scans → detection + recognition pipeline
```

Two measured from-scratch models are in the tree. Next is a working RNN on column-wise MNIST, then make that recurrence actually robust.
