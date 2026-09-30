# %% [markdown]
# # Sequence Models for Text: GRU and LSTM in PyTorch
#
# **Why this matters.** Bag-of-words models treat a review as an unordered pile of words, so "dog bites man" and "man bites dog" are identical. Recurrent networks (RNNs) read the words **one at a time, in order**,
# carrying a running memory. They were the workhorse of NLP before transformers, they still appear in production (small, fast, streaming models), and understanding the hidden state is the stepping stone to attention.
# The habit worth building here is the one from every module so far: compare against the simple baseline, then construct a test that shows *when* the extra machinery is necessary.
#
# **You will learn**
# - turn text into padded integer batches with a vocabulary and lengths, and use `pack_padded_sequence` so padding is ignored
# - build a GRU / LSTM text classifier in PyTorch and predict its tensor shapes and parameter counts
# - compare it honestly against TF-IDF + logistic regression on the same split
# - design an order-sensitive test where bag-of-words *cannot* work, and see what each model does with it
#
# **Prerequisites:** [Bag of words, TF-IDF and classifiers](02_bag_of_words_tfidf_and_classifiers.ipynb), [Word embeddings](03_word_embeddings_from_scratch.ipynb), and the PyTorch training loop from
# [PyTorch: nn.Module and training loops](../14_PyTorch/02_nn_module_and_training_loop.ipynb).
#
# > Data used: `datasets/reviews.csv` (synthetic, templated; `docs/DATASETS.md`). Sentiment on these reviews is easy for any model, which is exactly why we add a constructed order-dependent task in section 5.

# %%
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)
reviews = pd.read_csv(Path("../datasets/reviews.csv"))
classes = ["negative", "neutral", "positive"]
y_all = reviews["sentiment"].map({c: i for i, c in enumerate(classes)}).to_numpy()
groups = pd.factorize(reviews["review_text"])[0]
tv_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(reviews, groups=groups))
tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(tv_idx, groups=groups[tv_idx]))
tr_idx, va_idx = tv_idx[tr_idx], tv_idx[va_idx]  # indices into the full table, no text shared between splits
print("train / validation / test rows:", len(tr_idx), len(va_idx), len(te_idx))

# %% [markdown]
# ## 1. From text to padded integer batches
# A network needs equal-length tensors, but reviews have 6 to 19 words. The standard recipe: map each word to an integer id from a vocabulary built on **training text only** (id 0 = padding, id 1 = unknown word),
# pad every sequence with zeros to a common length, and keep the **true length** of each sequence so the model can ignore the padding.

# %%
def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


counts = Counter(w for t in reviews["review_text"].iloc[tr_idx] for w in tokenize(t))
itos = ["<pad>", "<unk>"] + [w for w, c in counts.most_common()]
stoi = {w: i for i, w in enumerate(itos)}


def encode(text, max_len):
    ids = [stoi.get(w, 1) for w in tokenize(text)][:max_len]
    length = len(ids)
    return ids + [0] * (max_len - length), length


def make_tensors(texts, max_len):
    enc = [encode(t, max_len) for t in texts]
    return torch.tensor([e[0] for e in enc]), torch.tensor([e[1] for e in enc])


MAX_LEN = 24
ids, lens = make_tensors(reviews["review_text"].iloc[:3], MAX_LEN)
print("vocabulary size:", len(itos))
print("ids shape:", tuple(ids.shape), "| lengths:", lens.tolist())
print(ids[0].tolist())

# %% [markdown]
# ## 2. The model: embedding, recurrent layer, linear head
# **Intuition.** The recurrent layer keeps a **hidden state** `h`, a vector that summarises everything read so far. At each word it updates `h` from the previous `h` and the new word's embedding. After the last real word,
# `h` is a fixed-size summary of the whole text, and a linear layer turns it into class scores.
#
# - A **GRU** (gated recurrent unit) uses two learned *gates*: an update gate decides how much of the old state to keep, a reset gate decides how much of it to consult when forming the new one.
# - An **LSTM** keeps a separate *cell state* with three gates (input, forget, output). It has more parameters; in practice GRU and LSTM perform similarly on small tasks.
# - Why gates? A plain RNN multiplies by the same matrix at every step, so gradients shrink (or explode) over long sequences. Gates give the gradient a more direct path, which helps, but very long-range dependencies remain hard;
#   attention (next notebook) attacks that problem directly.

# %%
class TextRNN(nn.Module):
    def __init__(self, vocab_size, n_classes=3, emb=32, hidden=32, cell="gru"):
        super().__init__()
        self.cell = cell
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell](emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, ids, lengths):
        x = self.emb(ids)  # (batch, time, emb)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)  # padding is skipped
        _, h = self.rnn(packed)  # h: (layers, batch, hidden); for an LSTM it is the pair (h, c)
        if self.cell == "lstm":
            h = h[0]
        return self.head(h[-1])  # (batch, n_classes)


gru = TextRNN(len(itos), cell="gru")
lstm = TextRNN(len(itos), cell="lstm")
print("GRU  parameters:", sum(p.numel() for p in gru.parameters()))
print("LSTM parameters:", sum(p.numel() for p in lstm.parameters()))
logits = gru(ids, lens)
print("logits shape for a batch of 3:", tuple(logits.shape))

# %% [markdown]
# Shapes: the embedding turns `(3, 24)` ids into `(3, 24, 32)` vectors; the GRU returns its final hidden state `(1, 3, 32)`; `h[-1]` is `(3, 32)`; the head gives `(3, 3)`.
# **Packing** means the GRU really stops at each sequence's true length: without it the model would keep "reading" padding zeros and its summary would depend on how much padding a batch happened to have.
# Parameter counts (check them yourself in Exercise 3): a GRU has 3 sets of weights, an LSTM has 4, so the LSTM is larger at the same hidden size.

# %% [markdown]
# ## 3. Training loop
# The standard PyTorch loop: forward, loss, `zero_grad`, backward, step; evaluate in `eval()` mode under `no_grad`. We keep the model with the best validation loss.

# %%
def train_model(model, Xtr, ltr, ytr, Xva, lva, yva, epochs=8, lr=5e-3, batch=64, seed=42):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr, ltr, torch.as_tensor(ytr)), batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    best, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        for xb, lb, yb in loader:
            opt.zero_grad()
            lossf(model(xb, lb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = lossf(model(Xva, lva), torch.as_tensor(yva)).item()
        history.append(val_loss)
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history


def predict(model, X, lens):
    model.eval()
    with torch.no_grad():
        return model(X, lens).argmax(dim=1).numpy()


Xtr, ltr = make_tensors(reviews["review_text"].iloc[tr_idx], MAX_LEN)
Xva, lva = make_tensors(reviews["review_text"].iloc[va_idx], MAX_LEN)
Xte, lte = make_tensors(reviews["review_text"].iloc[te_idx], MAX_LEN)
ytr, yva, yte = y_all[tr_idx], y_all[va_idx], y_all[te_idx]

results = {}
for cell in ["gru", "lstm"]:
    torch.manual_seed(42)
    model = TextRNN(len(itos), cell=cell)
    t0 = time.perf_counter()
    hist = train_model(model, Xtr, ltr, ytr, Xva, lva, yva)
    pred = predict(model, Xte, lte)
    results[cell] = model
    print(f"{cell.upper():5s} trained in {time.perf_counter() - t0:4.1f}s | best val loss {min(hist):.4f} | test accuracy {accuracy_score(yte, pred):.3f}")

# %% [markdown]
# ## 4. The baseline: TF-IDF + logistic regression
# Same rows, same split, one line of model:

# %%
tfidf = TfidfVectorizer()
lr_model = LogisticRegression(max_iter=1000).fit(tfidf.fit_transform(reviews["review_text"].iloc[tr_idx]), ytr)
pred_lr = lr_model.predict(tfidf.transform(reviews["review_text"].iloc[te_idx]))
table = {"TF-IDF + logistic regression": pred_lr}
for cell, m in results.items():
    table[f"{cell.upper()} (PyTorch)"] = predict(m, Xte, lte)
pd.DataFrame({k: {"test accuracy": accuracy_score(yte, p), "test macro-F1": f1_score(yte, p, average="macro")} for k, p in table.items()}).T.round(3)

# %% [markdown]
# *What just happened?* On this easy, templated task the order-blind baseline and the recurrent networks reach (nearly) the same score. So **the RNN has not earned its complexity on sentiment here**: it is slower to train,
# has more knobs, and adds nothing the baseline lacks. This is the ordinary outcome when the signal lives in individual words. The interesting question is: *is there a task where order matters, and does the RNN then win?*

# %% [markdown]
# ## 5. A test where order is the signal
# We construct one. Take two reviews with **opposite** sentiment, join them as `"<review A> However, <review B>"`, and let the label be the sentiment of the **first** review. Both classes contain exactly the same kinds of words,
# so a bag of words cannot tell which opinion came first. To avoid leakage, the training pairs are built only from training rows, and the test pairs only from test rows (no text appears on both sides).

# %%
rng = np.random.default_rng(42)
texts_all = reviews["review_text"].to_numpy()
is_pos, is_neg = y_all == 2, y_all == 0


def make_pairs(idx, n):
    pos, neg = idx[is_pos[idx]], idx[is_neg[idx]]
    first_pos = rng.random(n) < 0.5
    a = np.where(first_pos, rng.choice(pos, n), rng.choice(neg, n))
    b = np.where(first_pos, rng.choice(neg, n), rng.choice(pos, n))
    text = [f"{texts_all[i]} However, {texts_all[j]}" for i, j in zip(a, b)]
    return text, first_pos.astype(int)  # label 1 = the FIRST review is positive


pair_tr, py_tr = make_pairs(tr_idx, 2000)
pair_va, py_va = make_pairs(va_idx, 600)
pair_te, py_te = make_pairs(te_idx, 800)
print(pair_te[0], "->", "first review positive" if py_te[0] else "first review negative")
print("share positive-first in test:", round(py_te.mean(), 3), "(chance level is 0.5)")

# %%
PAIR_LEN = 48
P_tr, PL_tr = make_tensors(pair_tr, PAIR_LEN)
P_va, PL_va = make_tensors(pair_va, PAIR_LEN)
P_te, PL_te = make_tensors(pair_te, PAIR_LEN)

rows = {}
for name, ngram in [("TF-IDF unigrams + LR", (1, 1)), ("TF-IDF uni+bigrams + LR", (1, 2))]:
    v = TfidfVectorizer(ngram_range=ngram)
    m = LogisticRegression(max_iter=2000).fit(v.fit_transform(pair_tr), py_tr)
    rows[name] = accuracy_score(py_te, m.predict(v.transform(pair_te)))
for cell in ["gru", "lstm"]:
    torch.manual_seed(42)
    m = TextRNN(len(itos), n_classes=2, cell=cell)
    train_model(m, P_tr, PL_tr, py_tr, P_va, PL_va, py_va, epochs=10)
    rows[f"{cell.upper()} (PyTorch)"] = accuracy_score(py_te, predict(m, P_te, PL_te))
pd.Series(rows, name="test accuracy").round(3).to_frame()

# %% [markdown]
# Read the first row: the unigram model is at about **chance** (0.5), because the two classes contain the same words. That is the bag-of-words limit made concrete. The recurrent models, which read in order,
# solve the task. But look at the second row: the unigram + **bigram** model also scores near-perfectly in our run, and the reason is instructive rather than embarrassing. The word pair at the joint
# ("...early However") exposes the ending of review A, and the ending of a review is strongly tied to its own sentiment in this data, so a *local* n-gram is enough to recover the answer. So this test separates
# "no order information" from "some", but it does **not** separate the GRU from a bigram model. Take the general lesson: **n-grams are a cheap, strong way to add local order and must be tried before a neural model.**
# A test that needs the GRU would have to hide the answer from every short window (for example, a label that depends on a cue at the start of the text and a cue at the end together). Design that test
# when you think a sequence model is justified; do not assume it is. (Exact numbers depend on the seed; rerun with another before believing a small difference.)

# %% [markdown]
# ## Common mistakes
#
# 1. **Not packing or masking padding.** The hidden state then absorbs padding, and predictions change with batch composition. Use `pack_padded_sequence` (or an attention mask).
# 2. **Reading `output[:, -1]` of a padded batch** as the last step: for shorter sequences that position is padding. Use the final hidden state from a packed input, or index by true length.
# 3. **Forgetting `model.eval()` / `torch.no_grad()`** at prediction time (matters once you add dropout).
# 4. **Building the vocabulary on all text**, including the test set.
# 5. **`batch_first` confusion.** With `batch_first=False` (the default) inputs must be `(time, batch, features)`. We pass `batch_first=True` and `(batch, time, features)`.
# 6. **Declaring the RNN better from one seed.** Repeat with a few seeds; differences below the spread are noise.
# 7. **Using an RNN when a baseline already saturates the task.** Always run TF-IDF + logistic regression first.

# %%
# Demonstration of mistake 1: the same text with different amounts of padding gives different answers WITHOUT packing
emb_layer, rnn_layer = results["gru"].emb, results["gru"].rnn
sample = "Great quality and it arrived early."
out = {}
for L in (12, 30):
    ids_, n_ = make_tensors([sample], L)
    with torch.no_grad():
        x = emb_layer(ids_)
        naive = rnn_layer(x)[1][-1]  # final state after reading ALL positions, padding included
        packed = rnn_layer(pack_padded_sequence(x, n_, batch_first=True, enforce_sorted=False))[1][-1]
    out[L] = (naive, packed)
print("hidden-state change when padding grows from 12 to 30 positions:")
print(f"  without packing: {float((out[12][0] - out[30][0]).abs().max()):.4f}")
print(f"  with packing   : {float((out[12][1] - out[30][1]).abs().max()):.2e}")

# %% [markdown]
# ## Interview corner
#
# **1. What problem do LSTM and GRU solve compared with a plain RNN?**
# <details><summary>Answer</summary>
# In a plain RNN the gradient passes through the same weight matrix at every step, so it tends to vanish or explode over long sequences, and the model cannot learn long-range dependencies. Gating lets the network
# decide what to keep and what to overwrite, giving gradients a more direct path (especially the LSTM cell state). It mitigates, but does not eliminate, the long-range problem.
# </details>
#
# **2. Why did transformers largely replace RNNs for text?**
# <details><summary>Answer</summary>
# RNNs process tokens sequentially, so training cannot be parallelised across time and information has to squeeze through one fixed-size state. Self-attention lets every token look at every other token directly,
# trains in parallel on GPUs, and scales better with data. RNNs remain useful for small, streaming or low-latency models.
# </details>
#
# **3. How do you handle variable-length sequences in a batch?**
# <details><summary>Answer</summary>
# Pad to a common length and either pack the batch (<code>pack_padded_sequence</code>) or pass a mask so the model ignores padding; sort or bucket by length to reduce wasted computation. Never let padding influence
# the representation or the loss.
# </details>
#
# **4. When would you pick TF-IDF + logistic regression over an RNN?**
# <details><summary>Answer</summary>
# When the signal is in individual words or short phrases, data is small, latency and explainability matter, and the baseline is already near the ceiling. Try n-grams for cheap local order. Move to sequence models or transformers
# when order, long-range context or transfer from pretraining really matters.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Predict the shapes
# A batch of 4 sequences, each 7 tokens long, goes through `nn.Embedding(50, 8)` and then `nn.GRU(8, 5, batch_first=True)`. Without running them, store in `out_shape` the shape (tuple) of the GRU's per-step output
# and in `h_shape` the shape of its final hidden state. The check builds the layers and verifies.

# %% tags=["starter"]
out_shape = ...
h_shape = ...

# %%
_e, _g = nn.Embedding(50, 8), nn.GRU(8, 5, batch_first=True)
_out, _h = _g(_e(torch.zeros(4, 7, dtype=torch.long)))
assert tuple(out_shape) == tuple(_out.shape), f"per-step output shape is {tuple(_out.shape)}"
assert tuple(h_shape) == tuple(_h.shape), f"final hidden state shape is {tuple(_h.shape)}"
print("Exercise 1 passed:", tuple(out_shape), tuple(h_shape))

# %% tags=["solution"]
out_shape = (4, 7, 5)  # (batch, time, hidden)
h_shape = (1, 4, 5)  # (layers, batch, hidden)

# %% [markdown]
# ### Exercise 2: Parameter counts by formula
# A recurrent layer with input size `n_in` and hidden size `n_h` has, per gate block, a weight matrix `n_h x n_in`, a recurrent matrix `n_h x n_h`, and two bias vectors of length `n_h` (PyTorch keeps two).
# A GRU has 3 such blocks, an LSTM has 4. For `n_in = 8`, `n_h = 5` store the counts in `gru_params` and `lstm_params`, computed by hand. The check counts the real layers.

# %% tags=["starter"]
gru_params = ...
lstm_params = ...

# %%
assert gru_params == sum(p.numel() for p in nn.GRU(8, 5).parameters()), "GRU: 3 x (n_h*n_in + n_h*n_h + 2*n_h)"
assert lstm_params == sum(p.numel() for p in nn.LSTM(8, 5).parameters()), "LSTM: 4 x (n_h*n_in + n_h*n_h + 2*n_h)"
print("Exercise 2 passed:", gru_params, lstm_params)

# %% tags=["solution"]
block = 5 * 8 + 5 * 5 + 2 * 5
gru_params = 3 * block
lstm_params = 4 * block

# %% [markdown]
# ### Exercise 3: Does padding change the prediction?
# Use the trained GRU `results["gru"]` and the sentence `"Terrible quality and it broke after a week."`. Encode it with `make_tensors` at `max_len=20` and at `max_len=40`, run the model on both and store in `pad_diff` the largest absolute
# difference between the two logit vectors. It should be at floating-point noise level because the model packs its input. Store in `padding_safe` whether `pad_diff < 1e-5`.

# %% tags=["starter"]
pad_diff = ...
padding_safe = ...

# %%
_m = results["gru"]
_m.eval()
_s = "Terrible quality and it broke after a week."
with torch.no_grad():
    _a = _m(*make_tensors([_s], 20))
    _b = _m(*make_tensors([_s], 40))
_exp = float((_a - _b).abs().max())
assert abs(pad_diff - _exp) < 1e-9, f"expected {_exp:.3e}"
assert padding_safe == (_exp < 1e-5) and padding_safe, "with packing, padding must not change the output"
print(f"Exercise 3 passed (difference {pad_diff:.1e})")

# %% tags=["solution"]
_m = results["gru"]
_m.eval()
_s = "Terrible quality and it broke after a week."
with torch.no_grad():
    _a = _m(*make_tensors([_s], 20))
    _b = _m(*make_tensors([_s], 40))
pad_diff = float((_a - _b).abs().max())
padding_safe = bool(pad_diff < 1e-5)

# %% [markdown]
# ### Exercise 4: Same words, different order
# Write `swap_halves(pair_text)` for texts of the form `"<A> However, <B>"`: return `"<B> However, <A>"`. Apply it to every text in `pair_te` and store in `acc_swapped_gru` the accuracy of the **pair-trained GRU** against the
# *flipped* labels (`1 - py_te`). If the model truly reads order, swapping the halves should flip its prediction, so this accuracy should be about as high as the original. You need the model trained in section 5; train it again
# here with `torch.manual_seed(42)`, `TextRNN(len(itos), n_classes=2, cell="gru")`, `train_model(..., epochs=10)` exactly as in the notebook and keep it in `pair_gru`.

# %% tags=["starter"]
def swap_halves(pair_text):
    ...


pair_gru = ...
acc_swapped_gru = ...

# %%
assert swap_halves("Good one. However, Bad one.") == "Bad one. However, Good one.", "swap the text before and after ' However, '"
_swapped = [swap_halves(t) for t in pair_te]
assert all(swap_halves(s) == t for s, t in zip(_swapped, pair_te)), "swapping twice must give the original"
_X, _L = make_tensors(_swapped, PAIR_LEN)
_exp = accuracy_score(1 - py_te, predict(pair_gru, _X, _L))
assert abs(acc_swapped_gru - _exp) < 1e-9, f"expected {_exp:.3f}"
print(f"Exercise 4 passed (accuracy against flipped labels: {acc_swapped_gru:.3f})")

# %% tags=["solution"]
def swap_halves(pair_text):
    first, second = pair_text.split(" However, ")
    return f"{second} However, {first}"


torch.manual_seed(42)
pair_gru = TextRNN(len(itos), n_classes=2, cell="gru")
train_model(pair_gru, P_tr, PL_tr, py_tr, P_va, PL_va, py_va, epochs=10)
_Xs, _Ls = make_tensors([swap_halves(t) for t in pair_te], PAIR_LEN)
acc_swapped_gru = float(accuracy_score(1 - py_te, predict(pair_gru, _Xs, _Ls)))

# %% [markdown]
# ### Exercise 5 (open-ended): Report the honest comparison
# Write five sentences as a message to a teammate who wants "to use an LSTM because it is more advanced": what your experiments showed on plain sentiment, what they showed on the order task, what the unigram versus bigram results
# teach, and what evidence (on real data, several seeds) would make you adopt a recurrent or transformer model.

# %% [markdown]
# ## Recap
# - An RNN reads tokens in order and carries a hidden state; GRU and LSTM add gates to keep useful memory and ease gradient flow. Shapes: embedding `(B, T, E)`, final hidden `(layers, B, H)`, logits `(B, classes)`.
# - Pad to a common length and **pack** (or mask) so padding never reaches the representation; build the vocabulary on training text only.
# - On word-level signals, TF-IDF + logistic regression matches the recurrent models at a fraction of the cost. On a constructed order-dependent task, unigram bag-of-words drops to chance, which shows what it cannot do.
# - Try n-grams before neural models for cheap local order, and repeat experiments over seeds before crowning a winner.
#
# **Go deeper:** the PyTorch tutorial "NLP From Scratch" and the `torch.nn.GRU` / `torch.nn.LSTM` documentation (pytorch.org); Christopher Olah's blog post "Understanding LSTM Networks" (colah.github.io); Cho et al. (2014) for the GRU.
#
# **Next:** [Attention and transformers from scratch](05_attention_and_transformers_from_scratch.ipynb)
