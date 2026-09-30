# %% [markdown]
# # Text Classification with Keras (and an Honest TF-IDF Baseline)
#
# **Why this matters.** "Tag every incoming review or support ticket as positive, negative or neutral" is one of the most common first NLP projects at work.
# The tempting move is to reach for a neural network. The professional move is to build a TF-IDF + logistic regression baseline first (ten lines, trains in
# milliseconds), and only keep the neural model if it earns its complexity. This notebook builds both with the same split and compares them fairly, and it
# shows how an **embedding** turns words into trainable vectors.
#
# **You will learn**
# - turn raw strings into integer sequences with `TextVectorization` and put that step *inside* the model so it accepts plain text
# - build an embedding + pooling classifier and read its summary
# - split text data without leaking duplicates, and see how much a naive split flatters the score
# - compare against TF-IDF + logistic regression, and run the error analysis that tells you what to do next
# - explain what "Keras 3 is multi-backend" means and how `KERAS_BACKEND` works
#
# **Prerequisites:** [Sequential and Functional API](01_sequential_and_functional_api.ipynb), [Callbacks, regularization and tuning](02_callbacks_regularization_and_tuning.ipynb).
#
# > Data used: `datasets/reviews.csv` (4,000 synthetic product reviews; `docs/DATASETS.md`). **Read this caveat first:** the reviews are assembled from a small set of
# > templates, so there are only a few hundred *distinct* texts and the label is almost fully determined by the opening phrase. Scores here are far more optimistic than on real reviews. The techniques transfer; the numbers do not.

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import re
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

reviews = pd.read_csv(Path("../datasets/reviews.csv"))
print(reviews.shape)
print(reviews["sentiment"].value_counts(normalize=True).round(3).to_dict())
reviews.sample(4, random_state=1)[["review_text", "sentiment"]]

# %% [markdown]
# ## 1. Look at the text before modelling
# Two quick checks that save hours later: how long are the texts, and how many are *unique*?

# %%
n_words = reviews["review_text"].str.split().str.len()
print(f"words per review: min {n_words.min()}, median {int(n_words.median())}, max {n_words.max()}")
print(f"distinct texts: {reviews['review_text'].nunique()} of {len(reviews)} rows")
reviews["review_text"].value_counts().head(3)

# %% [markdown]
# Only a fraction of the rows are distinct: many reviews are **exact copies**. On real data this happens too (copy-pasted reviews, bot traffic, repeated tickets).
# It matters because a normal random split puts a text in the training set and its identical twin in the test set, and the model is then "tested" on something it memorised.
# Let us measure how much that inflates a score with a quick linear baseline:

# %%
def tfidf_lr(train_text, train_y, test_text, test_y):
    vec = TfidfVectorizer(lowercase=True)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(train_text), train_y)
    pred = clf.predict(vec.transform(test_text))
    return accuracy_score(test_y, pred), pred, vec, clf


text, label = reviews["review_text"].to_numpy(dtype=object), reviews["sentiment"]

# Naive split: random rows
i_tr, i_te = train_test_split(np.arange(len(reviews)), test_size=0.2, random_state=42, stratify=label)
acc_naive, *_ = tfidf_lr(text[i_tr], label.iloc[i_tr], text[i_te], label.iloc[i_te])
overlap = pd.Series(text[i_te]).isin(set(text[i_tr])).mean()
print(f"naive row split: {overlap:.0%} of test texts also appear verbatim in training; accuracy {acc_naive:.3f}")

# %% [markdown]
# Now split by **group**: every copy of a text goes to the same side. `GroupShuffleSplit` does this, given a group id per row. We carve out validation the same way.

# %%
groups = pd.factorize(reviews["review_text"])[0]  # same text -> same integer id


def group_split(idx, size, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    a, b = next(gss.split(idx, groups=groups[idx]))
    return idx[a], idx[b]


all_idx = np.arange(len(reviews))
tmp_idx, te_idx = group_split(all_idx, 0.2, 42)
tr_idx, va_idx = group_split(tmp_idx, 0.2, 42)
classes = ["negative", "neutral", "positive"]
to_int = {c: i for i, c in enumerate(classes)}
y_all = reviews["sentiment"].map(to_int).to_numpy()
print("rows  train/val/test:", len(tr_idx), len(va_idx), len(te_idx))
print("texts shared between train and test:", len(set(text[tr_idx]) & set(text[te_idx])))

# %% [markdown]
# *What just happened?* The naive split let most test rows be seen copies; the group split guarantees the test set holds texts the model has never seen.
# Use the group split from here on. (Because the templates share words across texts, the test set is still *easier* than real-world data.)

# %% [markdown]
# ## 2. The baseline to beat
# TF-IDF represents each text as a sparse vector of word weights (frequent-in-this-text and rare-overall words weigh more; notebook 17_NLP/02 covers it in depth).
# Logistic regression then learns one weight per word per class.

# %%
acc_tfidf_val, _, vec, lr = tfidf_lr(text[tr_idx], y_all[tr_idx], text[va_idx], y_all[va_idx])
pred_tfidf_te = lr.predict(vec.transform(text[te_idx]))
print(f"TF-IDF + logistic regression: validation accuracy {acc_tfidf_val:.3f}, test accuracy {accuracy_score(y_all[te_idx], pred_tfidf_te):.3f}")
print("vocabulary size:", len(vec.vocabulary_))

# %% [markdown]
# ## 3. `TextVectorization`: strings to integers
# **Intuition.** A network multiplies numbers, so text must become numbers. `TextVectorization` (1) lowercases and strips punctuation, (2) splits on whitespace,
# (3) looks each word up in a **vocabulary** learned from the training text, and (4) pads or truncates to a fixed length. Id 0 is padding, id 1 is `[UNK]`
# (a word not seen when the vocabulary was built). `adapt()` builds the vocabulary: call it on **training text only**, as with any fitted preprocessing.

# %%
MAX_LEN = 24  # the longest review has 19 words; leave some headroom
vectorize = keras.layers.TextVectorization(max_tokens=None, output_mode="int", output_sequence_length=MAX_LEN)
vectorize.adapt(text[tr_idx])
vocab = vectorize.get_vocabulary()
print("vocabulary size (incl. padding and [UNK]):", len(vocab))
print("first ids:", list(enumerate(vocab[:8])))
ex = np.array(["Great quality, but the screen cracked!", "Completely unseen words here"], dtype=object)
print(vectorize(ex).numpy())

# %% [markdown]
# Read the second row: words the training text never contained map to id 1 (`[UNK]`), and the zeros at the end are padding. The punctuation and capital letters in the first
# text vanished before lookup. This is also why a bag-of-words model cannot handle a word it has never seen, a limitation that subword tokenisers (17_NLP/01) address.

# %% [markdown]
# ## 4. Embedding + pooling classifier
# **Intuition.** An `Embedding(vocab_size, 16)` layer is a lookup table with one trainable 16-number vector per word id. Instead of hand-designing features, the training
# process moves each word's vector so words that signal the same sentiment end up useful together. `GlobalAveragePooling1D` then averages the vectors of the words in a
# review into one 16-number summary (ignoring padding when `mask_zero=True`), and a `Dense` layer with softmax turns that into three class probabilities.
#
# Putting `vectorize` *inside* the model means the saved model accepts raw strings: no separate preprocessing code to forget in production.

# %%
def build_text_model(embed_dim=16, dropout=0.3):
    keras.utils.set_random_seed(42)
    m = keras.Sequential([
        keras.layers.Input(shape=(), dtype="string"),
        vectorize,
        keras.layers.Embedding(len(vocab), embed_dim, mask_zero=True),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(3, activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(1e-2), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


model = build_text_model()
model.summary()

# %% [markdown]
# Check the parameters: the embedding has `vocab_size x 16` weights and the final layer `16 x 3 + 3`. The `TextVectorization` layer has no trainable weights (its vocabulary is fixed data).

# %%
stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
hist = model.fit(text[tr_idx], y_all[tr_idx], validation_data=(text[va_idx], y_all[va_idx]), epochs=60, batch_size=32, callbacks=[stop], verbose=0)
print(f"stopped after {len(hist.epoch)} epochs; best validation loss {min(hist.history['val_loss']):.3f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(hist.history["loss"], label="train loss")
ax.plot(hist.history["val_loss"], label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy (lower is better)")
ax.set_title("Embedding + pooling classifier: training curves")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. The fair comparison
# Same train/validation/test rows for both. We report accuracy and **macro-F1** (the unweighted mean of per-class F1, so the rarer neutral class counts as much as the common one).

# %%
proba_te = model.predict(text[te_idx], verbose=0)
pred_nn_te = proba_te.argmax(axis=1)
y_te = y_all[te_idx]
comp = pd.DataFrame({
    "TF-IDF + logistic regression": {"test accuracy": accuracy_score(y_te, pred_tfidf_te), "test macro-F1": f1_score(y_te, pred_tfidf_te, average="macro")},
    "Embedding + pooling (Keras)": {"test accuracy": accuracy_score(y_te, pred_nn_te), "test macro-F1": f1_score(y_te, pred_nn_te, average="macro")},
}).T.round(3)
print(f"test set: {len(y_te)} rows, {len(set(text[te_idx]))} distinct texts")
comp

# %% [markdown]
# *What just happened?* Two very different models land close together. That is the typical result on small, simple text tasks: the linear baseline is hard to beat, trains
# instantly and is easy to explain (you can list the words with the biggest weights). The neural model's advantages (learned representations, reusing pretrained embeddings,
# sharing structure with other tasks) matter more when there is more data, richer language, or word order that carries meaning. With only a few hundred distinct texts
# here, **differences in the third decimal mean nothing**: one test text can move the score.

# %% [markdown]
# ### What does pooling throw away? Word order
# Averaging vectors is **order-blind**: "not good" and "good not" give the same result. Verify it instead of trusting the claim: shuffle the words in each test review and compare predictions.

# %%
rng = np.random.default_rng(42)
shuffled = np.array([" ".join(rng.permutation(t.split())) for t in text[te_idx]], dtype=object)
p_shuf = model.predict(shuffled, verbose=0)
print("example:", text[te_idx][0], "->", shuffled[0])
print("largest change in any predicted probability:", float(np.abs(p_shuf - proba_te).max()))

# %% [markdown]
# The difference is at floating-point rounding level. So this model is a bag-of-words model with *learned, dense* word features. Sequence models (17_NLP/04) and
# transformers (17_NLP/05) are what make word order matter.

# %% [markdown]
# ## 6. Error analysis (when there are no errors)
# Look at the section 5 table: both models are at or very near 100%, and the confusion matrix is empty off the diagonal. **A perfect score is a finding about the data, not a
# triumph.** The generator writes each review from an opening phrase that belongs to exactly one sentiment ("Great quality", "Terrible quality", "Average product", ...), so a
# model only has to recognise the opener. Two checks make that visible: which words carry the weight, and what happens on text that does not look like the templates.

# %%
cm = confusion_matrix(y_te, pred_nn_te)
print(pd.DataFrame(cm, index=[f"true {c}" for c in classes], columns=[f"pred {c}" for c in classes]))
wrong = np.flatnonzero(pred_nn_te != y_te)
print(f"neural model wrong on {len(wrong)} test rows, TF-IDF model wrong on {(pred_tfidf_te != y_te).sum()}")

inv_vocab = np.array(vec.get_feature_names_out())
for k, c in enumerate(classes):
    top = inv_vocab[np.argsort(-lr.coef_[k])[:5]]
    print(f"words pushing the TF-IDF model towards {c:8s}:", list(top))

# %% [markdown]
# The top words are the template opener words, exactly as predicted. Now stress-test with reviews written in ordinary language. No labels are "official" here: use your own judgement
# about what a sensible model should say.

# %%
probe = np.array([
    "The screen is gorgeous but the battery life is poor",
    "Not good at all",
    "It is fine",
    "Absolutely love this",
    "Terrible quality",
    "Arrived on time, works as described",
], dtype=object)
p_nn = model.predict(probe, verbose=0)
p_lr = lr.predict_proba(vec.transform(probe))
pd.DataFrame({
    "text": probe,
    "neural model": [f"{classes[i]} ({p_nn[r, i]:.0%})" for r, i in enumerate(p_nn.argmax(axis=1))],
    "TF-IDF + LR": [f"{classes[i]} ({p_lr[r, i]:.0%})" for r, i in enumerate(p_lr.argmax(axis=1))],
})

# %% [markdown]
# Read each row and ask *why*. Some answers look right for the wrong reason: "Not good at all" comes out negative because the word "not" appears in template negatives ("Would not recommend"),
# and `good` was never in the training vocabulary (see the check in the next cell), not because the model understands negation. "Arrived on time, works as described" is called positive because
# of the word "works" from "Works perfectly". The mixed review in the first row is forced into a single class with very high confidence. **A classifier always answers**; its confidence
# is meaningful only on data that resembles its training data. That is why "100% on the test set" never replaces testing on text from the real source.

# %%
print({w: w in set(vocab) for w in ["good", "not", "fine", "terrible", "love", "gorgeous"]})

# %% [markdown]
# ## 7. Keras 3 is multi-backend (explanation only)
# Since version 3, Keras is no longer tied to TensorFlow. The same Keras code (`keras.layers`, `compile`, `fit`) can run on **TensorFlow**, **JAX** or **PyTorch**.
# - You pick the backend with the environment variable `KERAS_BACKEND` (`tensorflow`, `jax` or `torch`), set **before** `import keras`, or by editing the `"backend"` entry in `~/.keras/keras.json`.
# - `keras.backend.backend()` tells you which one is active (this environment prints `tensorflow`; the first cell of notebook 01 showed it).
# - To write code that works on any backend, use `keras.ops` (for example `keras.ops.matmul`) instead of calling `tf.*` or `torch.*` inside layers and losses.
# - Some layers depend on backend-specific machinery. String layers such as `TextVectorization` are built on TensorFlow string operations; with another backend, the documented
#   pattern is to do text preprocessing in a `tf.data` pipeline and feed integers to the model. Check the Keras documentation for the version you install.
# - Models saved as `.keras` files are backend-portable in principle; custom layers that used backend-specific calls are the usual exception.
#
# Nothing in this course changes the backend: every notebook here runs with TensorFlow, and the checker pins `KERAS_BACKEND=tensorflow`.

# %% [markdown]
# ## Common mistakes
#
# 1. **Random split on duplicated text.** Identical texts on both sides inflate the test score (section 1 measured it). Split by group, or dedupe first.
# 2. **Fitting the vocabulary or TF-IDF on all the data** before splitting. `adapt` / `fit` on training text only.
# 3. **Skipping the baseline.** Without the TF-IDF number you cannot say whether the neural model is any good.
# 4. **Forgetting `mask_zero=True`** with padded sequences: the padding zeros get averaged in and the summary depends on the review *length*.
# 5. **Passing a `<U` NumPy string array** where an `object` array or `tf.constant` is expected, which can raise dtype errors in some versions; use `dtype=object` as we did.
# 6. **A sequence length far too short** silently truncates long texts. Check the length distribution (section 1) before choosing `output_sequence_length`.
# 7. **Reading accuracy on an imbalanced label** as success: a model that always says "positive" already scores about half here (see the cell below). Report macro-F1 and the confusion matrix.

# %%
# Demonstration of mistake 7: the majority-class baseline
majority = np.bincount(y_all[tr_idx]).argmax()
print(f"always predicting '{classes[majority]}': accuracy {(y_te == majority).mean():.3f}, macro-F1 {f1_score(y_te, np.full_like(y_te, majority), average='macro'):.3f}")

# %% [markdown]
# ## Interview corner
#
# **1. What is an embedding layer and why use one instead of one-hot vectors?**
# <details><summary>Answer</summary>
# A trainable lookup table mapping each token id to a dense vector. One-hot vectors are huge, sparse and treat every word as equally different from every other; dense embeddings are
# small, learned for the task (or pretrained), and let similar words share statistical strength.
# </details>
#
# **2. Your text model scores 99% on the test set. What do you check before celebrating?**
# <details><summary>Answer</summary>
# Leakage (duplicates or near-duplicates across splits, preprocessing fitted on all data), class balance (compare with the majority baseline), whether the data resembles production text,
# and a simple baseline like TF-IDF + logistic regression. Near-perfect scores on small text datasets are more often a data problem than a modelling triumph.
# </details>
#
# **3. When would you choose TF-IDF + logistic regression over a neural network?**
# <details><summary>Answer</summary>
# Small to medium data, need for speed, explainability and cheap retraining, or when the baseline is already close. Move to neural models when word order or context matters, when you can use
# pretrained representations, or when there is enough data to learn from.
# </details>
#
# **4. What does putting `TextVectorization` inside the model buy you?**
# <details><summary>Answer</summary>
# The exported model takes raw strings, so training-time and serving-time preprocessing cannot drift apart. The cost is that the model is tied to the vocabulary it was adapted on and (for string ops) to TensorFlow.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: How big is the vocabulary?
# Create a new `TextVectorization()` layer (defaults: no length limit, so the output is ragged-free only if you set `output_sequence_length`; you only need the vocabulary),
# adapt it on the **training** text `text[tr_idx]`, and store it in `tv_full`. Store in `n_distinct_words` the number of distinct words it learned (not counting padding and `[UNK]`).
# The check computes the same number independently by lowercasing and splitting on non-letters.

# %% tags=["starter"]
tv_full = ...
n_distinct_words = ...

# %%
expected = len({w for t in text[tr_idx] for w in re.findall(r"[a-z']+", t.lower())})
assert len(tv_full.get_vocabulary()) == n_distinct_words + 2, "the vocabulary includes padding and [UNK], so subtract 2"
assert n_distinct_words == expected, f"expected {expected} distinct words, got {n_distinct_words}"
print(f"Exercise 1 passed ({n_distinct_words} distinct training words)")

# %% tags=["solution"]
tv_full = keras.layers.TextVectorization()
tv_full.adapt(text[tr_idx])
n_distinct_words = len(tv_full.get_vocabulary()) - 2

# %% [markdown]
# ### Exercise 2: How much duplication is in the test rows?
# Using a naive split `train_test_split(reviews, test_size=0.3, random_state=7)` (note: different from section 1), store in `dup_fraction` the fraction of **test rows**
# whose `review_text` also appears in the training rows. Then store in `n_unique_test` the number of distinct texts in the test rows.

# %% tags=["starter"]
dup_fraction = ...
n_unique_test = ...

# %%
a, b = train_test_split(reviews, test_size=0.3, random_state=7)
train_set = set(a["review_text"])
exp_frac = sum(t in train_set for t in b["review_text"]) / len(b)
assert abs(dup_fraction - exp_frac) < 1e-9, f"expected {exp_frac:.4f}"
assert n_unique_test == b["review_text"].nunique(), "count distinct texts among the test rows"
print(f"Exercise 2 passed ({dup_fraction:.1%} of test rows are seen texts)")

# %% tags=["solution"]
tr_df, te_df = train_test_split(reviews, test_size=0.3, random_state=7)
seen = set(tr_df["review_text"])
dup_fraction = float(te_df["review_text"].isin(seen).mean())
n_unique_test = int(te_df["review_text"].nunique())

# %% [markdown]
# ### Exercise 3: Which class does the neural model confuse most?
# From the confusion matrix `cm` (rows = true, columns = predicted, classes in the order `classes`), find the **most common wrong pairing** (off-diagonal cell with the largest count;
# take the first in row-major order on ties). Store the true class name in `conf_true` and the predicted class name in `conf_pred`.
# If there are no mistakes at all (all off-diagonal cells are 0), store `None` in both.

# %% tags=["starter"]
conf_true = ...
conf_pred = ...

# %%
off = cm.copy()
np.fill_diagonal(off, 0)
if off.max() == 0:
    assert conf_true is None and conf_pred is None, "there are no mistakes, so both should be None"
else:
    i, j = np.unravel_index(off.argmax(), off.shape)
    assert (conf_true, conf_pred) == (classes[i], classes[j]), f"expected {classes[i]} -> {classes[j]}"
print("Exercise 3 passed:", conf_true, "->", conf_pred)

# %% tags=["solution"]
off = cm.copy()
np.fill_diagonal(off, 0)
if off.max() == 0:
    conf_true = conf_pred = None
else:
    i, j = np.unravel_index(off.argmax(), off.shape)
    conf_true, conf_pred = classes[i], classes[j]

# %% [markdown]
# ### Exercise 4: Does a bigger embedding help?
# Train `build_text_model(embed_dim=d)` for `d` in `[2, 16, 64]` with the same early-stopping recipe as section 4 (monitor `val_loss`, patience 5, up to 60 epochs, batch size 32).
# Store the **validation accuracy** of each in a dict `emb_acc` keyed by `d`, and store in `smallest_good` the smallest `d` whose validation accuracy is within 0.01 of the best one.
# *Hint: `model.evaluate(text[va_idx], y_all[va_idx], verbose=0)[1]` gives accuracy.*

# %% tags=["starter"]
emb_acc = ...
smallest_good = ...

# %%
assert sorted(emb_acc) == [2, 16, 64], "one entry per embedding size"
best_acc = max(emb_acc.values())
assert smallest_good == min(d for d, a in emb_acc.items() if a >= best_acc - 0.01), "smallest size within 0.01 of the best validation accuracy"
assert all(0.4 < a <= 1.0 for a in emb_acc.values()), "accuracies should be plausible (better than guessing)"
print("Exercise 4 passed:", {d: round(a, 3) for d, a in emb_acc.items()}, "-> smallest good:", smallest_good)

# %% tags=["solution"]
emb_acc = {}
for d in [2, 16, 64]:
    m = build_text_model(embed_dim=d)
    m.fit(text[tr_idx], y_all[tr_idx], validation_data=(text[va_idx], y_all[va_idx]), epochs=60, batch_size=32,
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)], verbose=0)
    emb_acc[d] = float(m.evaluate(text[va_idx], y_all[va_idx], verbose=0)[1])
best_acc = max(emb_acc.values())
smallest_good = min(d for d, a in emb_acc.items() if a >= best_acc - 0.01)

# %% [markdown]
# ### Exercise 5 (open-ended): Break your own model
# Write ten reviews in your own words (mix clear, mixed and sarcastic ones) and run them through the neural model and the TF-IDF model. Which ones fail, and why (unknown words, negation, mixed sentiment)?
# Then write three sentences: what data would you collect to fix the worst failure, which model family would use word order (see 17_NLP/04 and 05), and why a perfect score on this synthetic dataset does not show the model is ready for real customer reviews.

# %% [markdown]
# ## Recap
# - Build the TF-IDF + logistic regression baseline first; it is often within noise of a small neural model and far cheaper.
# - `TextVectorization` inside the model makes it accept raw strings; adapt it on training text only. Id 0 is padding, id 1 is `[UNK]`.
# - Embedding + average pooling is a learned but order-blind bag of words (verified by shuffling the words).
# - Split by text (group) when duplicates exist, otherwise the test score is inflated; report macro-F1 and compare with the majority-class baseline.
# - Keras 3 runs on TensorFlow, JAX or PyTorch, chosen with `KERAS_BACKEND` before importing Keras; use `keras.ops` for backend-neutral code.
#
# **Go deeper:** the Keras guides "Text classification from scratch" and "Using pre-trained word embeddings" (keras.io), and the scikit-learn user guide section "Working with text data".
#
# **Next module:** [NLP: text preprocessing and tokenization](../17_NLP/01_text_preprocessing_and_tokenization.ipynb)
