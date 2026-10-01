# %% [markdown]
# # Word Embeddings from Scratch: Co-occurrence, PPMI and SVD
#
# **Why this matters.** A bag-of-words model treats "refund" and "reimbursement" as unrelated columns. **Word embeddings** give every word a short dense vector so that words used in similar ways end up close together.
# They are the bridge from counting words to neural language models: the input layer of almost every modern text model is an embedding table. Here you build embeddings the classical way, with **no neural network at all**:
# count which words appear near each other, turn the counts into an association score, and compress with linear algebra. Once you have done it by hand, `word2vec`, GloVe and transformer embeddings stop being magic.
#
# **You will learn**
# - build a word-by-word co-occurrence matrix with a sliding window
# - convert counts into **PPMI** (positive pointwise mutual information) and explain why raw counts are a poor feature
# - compress with truncated SVD into dense vectors and find nearest neighbours by cosine similarity
# - test the vectors honestly: what they capture, and the failure modes (antonyms, tiny corpora, polysemy)
#
# **Prerequisites:** [Bag of words, TF-IDF and classifiers](02_bag_of_words_tfidf_and_classifiers.ipynb); linear algebra basics from [Math for Data Science](../02_Math_for_Data_Science/) (matrix, dot product, SVD idea).
#
# > Data used: the distinct texts in `datasets/reviews.csv` plus the ten policy documents in `datasets/knowledge_base/` (synthetic, `docs/DATASETS.md`). The corpus is **tiny and templated** (a few thousand words),
# > whereas real embeddings are trained on billions. Expect sensible but limited results; we will point out exactly where they break.

# %%
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

DATA = Path("../datasets")
reviews = pd.read_csv(DATA / "reviews.csv")


def tokens_of(sentence):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())


review_sents = [tokens_of(t) for t in reviews["review_text"].drop_duplicates()]
kb_sents = []
for f in sorted((DATA / "knowledge_base").glob("*.md")):
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if sent.strip():
                kb_sents.append(tokens_of(sent))
sentences = review_sents + kb_sents
print(f"{len(review_sents)} distinct review texts + {len(kb_sents)} knowledge-base sentences; {sum(map(len, sentences))} tokens")
print("example:", kb_sents[0])

# %% [markdown]
# ## 1. The distributional idea
# > "You shall know a word by the company it keeps." (J. R. Firth, 1957)
#
# If "refund" and "reimbursement" appear near the same neighbouring words ("issued", "payment", "days"), they probably mean similar things. So we represent a word by **the counts of the words around it**.
# Step one is the vocabulary. Words seen only once have too few contexts to describe, so normally we would keep words that occur at least twice. The policy documents are so small that most of their words occur once,
# so we keep every policy word too, knowing their vectors will be unreliable (we will see that):

# %%
freq = Counter(w for s in sentences for w in s)
kb_words = {w for s in kb_sents for w in s}
vocab = sorted(w for w, c in freq.items() if c >= 2 or w in kb_words)
index = {w: i for i, w in enumerate(vocab)}
print(f"vocabulary: {len(vocab)} words (dropped {len(freq) - len(vocab)} one-off review words; kept {len(kb_words)} policy words)")
print("most frequent:", freq.most_common(8))

# %% [markdown]
# ## 2. Co-occurrence counts with a sliding window
# For each word, look at up to `window` words on each side within the same sentence and add one to the count for every (word, context) pair. The result is a square matrix `C` where `C[i, j]` says how often
# word `j` appeared near word `i`. It is symmetric because "near" is symmetric.

# %%
def cooccurrence(sentences, index, window=2):
    C = np.zeros((len(index), len(index)))
    for sent in sentences:
        ids = [index[w] for w in sent if w in index]  # out-of-vocabulary words are dropped, which brings neighbours closer
        for pos, i in enumerate(ids):
            for j in ids[max(0, pos - window) : pos] + ids[pos + 1 : pos + 1 + window]:
                C[i, j] += 1
    return C


C = cooccurrence(sentences, index, window=2)
print("matrix shape:", C.shape, "| symmetric:", bool(np.allclose(C, C.T)), "| non-zero cells:", f"{(C > 0).mean():.1%}")
for a, b in [("great", "quality"), ("terrible", "quality"), ("great", "terrible"), ("refund", "days")]:
    print(f"C[{a!r}, {b!r}] = {int(C[index[a], index[b]])}")

# %% [markdown]
# *What just happened?* "great" and "terrible" never sit next to each other, but both sit next to "quality", which we will see matters. Raw counts have a problem, though: the most frequent words ("and", "the", "it")
# co-occur with everything, so they dominate every row and say little about meaning.

# %% [markdown]
# ## 3. PPMI: keep the surprising associations
# **Pointwise mutual information** asks: do words `w` and `c` appear together *more often than chance would predict* if they were independent?
#
# `PMI(w, c) = log( P(w, c) / (P(w) * P(c)) )`
#
# A large positive value means a real association; zero means independent; negative means they avoid each other. Because negative values are mostly noise from small counts, we clip them at zero: **PPMI**.
# We also raise the context probabilities to the power 0.75, a standard smoothing that stops very rare contexts from getting inflated scores.

# %%
def ppmi_matrix(C, alpha=0.75):
    total = C.sum()
    p_wc = C / total
    p_w = C.sum(axis=1, keepdims=True) / total
    c_alpha = C.sum(axis=0) ** alpha
    p_c = (c_alpha / c_alpha.sum())[None, :]
    with np.errstate(divide="ignore"):
        pmi = np.log(p_wc / (p_w * p_c))
    return np.where(np.isfinite(pmi) & (pmi > 0), pmi, 0.0)  # zero counts give -inf; clip negatives to 0


P = ppmi_matrix(C)
row = pd.Series(P[index["support"]], index=vocab).sort_values(ascending=False)
print("contexts most associated with 'support' (PPMI):")
print(row.head(6).round(2).to_string())
raw = pd.Series(C[index["support"]], index=vocab).sort_values(ascending=False)
print("\ncontexts with the highest RAW count for 'support':", list(raw.head(6).index))

# %% [markdown]
# Compare the two lists: the raw ranking is topped by "and", a word that co-occurs with nearly everything, while PPMI puts "helpful", "replied", "never" and "customer" first, the words that really belong with "support"
# in these reviews. Re-weighting by chance is the reason PPMI-based vectors work far better than raw counts. (Also note that "great" has only two non-zero contexts in this corpus; it appears almost only in "Great quality and ...".)

# %% [markdown]
# ## 4. Compress with truncated SVD
# Each PPMI row is a sparse vector with one entry per vocabulary word. Singular value decomposition factorises `P = U S V^T`; keeping only the `k` largest singular values gives the best rank-`k` approximation of `P`,
# and the rows of `U[:, :k] * sqrt(S[:k])` are **dense k-dimensional word vectors**. Words with similar context profiles get similar vectors, even if they never co-occur directly.

# %%
def embed(P, k=20):
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    E = U[:, :k] * np.sqrt(S[:k])
    return E, S


E, S = embed(P, k=20)
print("embedding matrix:", E.shape, "(one 20-number vector per word)")
energy = np.cumsum(S**2) / np.sum(S**2)
print("share of the matrix 'energy' kept with 20 of", len(S), "dimensions:", f"{energy[19]:.1%}")

# %%
def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def neighbours(word, E, k=6):
    sims = unit(E) @ unit(E[index[word]][None, :])[0]
    order = np.argsort(-sims)
    return [(vocab[i], round(float(sims[i]), 2)) for i in order if vocab[i] != word][:k]


for w in ["great", "terrible", "refund", "shipping", "electronics", "support"]:
    print(f"{w:12s}", neighbours(w, E, 5))

# %% [markdown]
# **Cosine similarity** is the cosine of the angle between two vectors: 1 means the same direction, 0 unrelated, negative opposite. It ignores vector length, which mostly reflects word frequency.
#
# Read the neighbours critically, word by word. "support" gets sensible neighbours ("customer", "helpful", "replied", "emailing"): it occurs 224 times in varied contexts. "electronics" picks up policy words
# ("carry", "refurbished", "full"). But "refund" is seen only once or twice, and its neighbours ("testing", "week", "broke") are accidents of one sentence: **rare words get unreliable vectors**. And "great" and "terrible"
# have cosine similarity 1.0 with each other and with "excellent": each appears in essentially one context pattern ("great quality and ..."), so their vectors coincide. Check the next cell.

# %%
pairs = [("great", "excellent"), ("great", "terrible"), ("great", "refund")]
for a, b in pairs:
    print(f"cosine({a}, {b}) = {cosine(E[index[a]], E[index[b]]):.2f}")

# %% [markdown]
# "great" and "terrible" are **opposites**, yet their vectors are as close as "great" and "excellent", because they are used in the same places ("great quality" / "terrible quality"). Distributional vectors capture
# *usage*, not *polarity*. This is the best-known limitation of embeddings: antonyms are neighbours. (Much larger real-world embeddings show the same effect in a milder form; it is one reason sentiment models are fine-tuned on labelled data.)

# %% [markdown]
# ### A picture
# Project the 20-dimensional vectors of the 40 most frequent words onto their first two dimensions (this throws information away; treat the picture as a hint, not a proof).

# %%
top = [w for w, _ in freq.most_common(200) if w in index][:40]
xy = E[[index[w] for w in top], :2]
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(xy[:, 0], xy[:, 1], s=10)
for w, (x, y) in zip(top, xy):
    ax.annotate(w, (x, y), fontsize=8, xytext=(2, 2), textcoords="offset points")
ax.set_xlabel("embedding dimension 1")
ax.set_ylabel("embedding dimension 2")
ax.set_title("40 most frequent words in the first two SVD dimensions")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Do the vectors carry information? A downstream test
# A nicer test than eyeballing neighbours: use them as features. Represent each review as the **average of its word vectors** and train a logistic regression to predict sentiment. Compare different numbers of
# dimensions `k` and the plain bag-of-words baseline on the same split (by distinct text, so no duplicate leaks).

# %%
texts = reviews["review_text"]
groups = pd.factorize(texts)[0]
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(texts, groups=groups))
y = reviews["sentiment"]
doc_tokens = [tokens_of(t) for t in texts]


def doc_matrix(E):
    out = np.zeros((len(doc_tokens), E.shape[1]))
    for n, toks in enumerate(doc_tokens):
        ids = [index[w] for w in toks if w in index]
        if ids:
            out[n] = E[ids].mean(axis=0)
    return out


from sklearn.feature_extraction.text import CountVectorizer

rows = []
for k in [2, 5, 10, 20, 40]:
    Ek, _ = embed(P, k=k)
    D = doc_matrix(Ek)
    clf = LogisticRegression(max_iter=2000).fit(D[tr], y.iloc[tr])
    rows.append({"features": f"averaged embeddings, k={k}", "test accuracy": clf.score(D[te], y.iloc[te])})
bow = CountVectorizer()
clf = LogisticRegression(max_iter=2000).fit(bow.fit_transform(texts.iloc[tr]), y.iloc[tr])
rows.append({"features": "bag of words (for reference)", "test accuracy": clf.score(bow.transform(texts.iloc[te]), y.iloc[te])})
rows.append({"features": "majority class (for reference)", "test accuracy": float((y.iloc[te] == y.iloc[tr].mode()[0]).mean())})
pd.DataFrame(rows).set_index("features").round(3)

# %% [markdown]
# Read this table against the two reference rows. Averaged embeddings beat the majority class clearly, and accuracy improves with the number of dimensions and then flattens, so the vectors do carry sentiment-relevant
# information. But they are **well below bag-of-words**. Two reasons you can now explain: each review averages a few informative words with many filler words, diluting the signal, and the antonym problem means that "great" and
# "terrible" have nearly the same vector, so the average cannot tell the classes apart. On this data, embeddings are the wrong tool, and bag-of-words is the right one. Embeddings pay off when the corpus is large
# enough for vectors to encode real meaning, when labelled data is scarce (so you borrow knowledge from unlabelled text), or when you need similarity search rather than classification.

# %% [markdown]
# ## 6. Limits you should remember
# - **Corpus size and domain.** These vectors know only what this small corpus shows. Words seen once were dropped; a word's neighbours can be dominated by one template. Real embeddings need vast text.
# - **Antonyms are neighbours** (shown above): usage similarity is not meaning similarity.
# - **One vector per word (no polysemy).** "bank" of a river and "bank" account share a single vector. Contextual models (transformers, notebook 05) produce a different vector per occurrence.
# - **Static.** The vectors do not change with the sentence; negation ("not great") is not handled by averaging.
# - **Bias.** Embeddings inherit the statistical associations of their training text, including social biases. Audit them before using them in decisions about people.
#
# **How this connects to neural embeddings.** `word2vec` (skip-gram with negative sampling) trains a small network to predict context words; it has been shown to be implicitly factorising a shifted PMI matrix,
# which is essentially what you just did with an explicit SVD. The embedding layer in Keras/PyTorch is the same lookup table, just trained for your task instead of derived from counts.

# %% [markdown]
# ## Common mistakes
#
# 1. **Using raw counts for similarity.** Frequent words dominate; apply PPMI (or at least TF-IDF-like weighting) first.
# 2. **Comparing vectors with Euclidean distance** when length tracks frequency; use cosine similarity.
# 3. **Judging embeddings by a few handpicked neighbours.** Also run a downstream test (classifier, retrieval) and look at failures, such as antonyms.
# 4. **Training on too little text** and then reading meaning into the geometry.
# 5. **Forgetting to drop or handle rare words** (sparse, noisy rows) and out-of-vocabulary words at inference time.
# 6. **Averaging word vectors and expecting word order or negation to survive.** It cannot.
# 7. **Taking more dimensions as always better**: beyond a point you fit noise, and a bigger `k` costs memory for every word.

# %%
# Demonstration of mistake 1: raw-count neighbours vs PPMI neighbours for "great"
def neighbours_from(M, word, k=6):
    sims = unit(M) @ unit(M[index[word]][None, :])[0]
    return [vocab[i] for i in np.argsort(-sims) if vocab[i] != word][:k]


print("from raw counts:", neighbours_from(C, "great"))
print("from PPMI+SVD  :", neighbours_from(E, "great"))

# %% [markdown]
# ## Interview corner
#
# **1. How does word2vec (or GloVe) produce embeddings, in one paragraph?**
# <details><summary>Answer</summary>
# Both use the distributional idea: words in similar contexts get similar vectors. word2vec trains a shallow network to predict context words from a word (or the reverse) with negative sampling; GloVe fits vectors to the log
# co-occurrence counts. Both are closely related to factorising a (shifted) PMI matrix, which is what an explicit PPMI + SVD does.
# </details>
#
# **2. Why PPMI rather than raw co-occurrence counts?**
# <details><summary>Answer</summary>
# Raw counts are dominated by frequent words that co-occur with everything. PMI compares the joint frequency with what independence would predict, so it highlights informative associations; clipping negatives (PPMI)
# removes noisy negative values from sparse counts.
# </details>
#
# **3. What are the main limitations of static word embeddings?**
# <details><summary>Answer</summary>
# One vector per word regardless of context (polysemy), antonyms appear close because they share contexts, out-of-vocabulary words have no vector, and they absorb biases from the training text.
# Contextual embeddings from transformers address the first and third problems in part.
# </details>
#
# **4. How do you evaluate embeddings?**
# <details><summary>Answer</summary>
# Intrinsic checks (nearest neighbours, similarity benchmarks, analogies) are quick but shallow; extrinsic evaluation on the downstream task (classification, retrieval quality) is what matters for a project. Always look at failure cases.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Count the training pairs
# Write `window_pairs(tokens, window)` returning the list of ordered `(word, context)` pairs for one sentence: every token paired with each other token at most `window` positions away on either side.
# Then store in `n_pairs_kb` the total number of pairs produced over all knowledge-base sentences `kb_sents` with `window=3`. The check uses a closed-form count (positions within the window) for the same number.

# %% tags=["starter"]
def window_pairs(tokens, window):
    ...


n_pairs_kb = ...

# %%
assert sorted(window_pairs(["a", "b", "c"], 1)) == sorted([("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")]), "window=1 on a b c"
assert window_pairs(["x"], 2) == [], "a one-word sentence has no pairs"
expected = sum(min(i + 3, len(s) - 1) - max(i - 3, 0) for s in kb_sents for i in range(len(s)))
assert n_pairs_kb == expected, f"expected {expected} pairs"
print(f"Exercise 1 passed ({n_pairs_kb} ordered pairs in the knowledge base)")

# %% tags=["solution"]
def window_pairs(tokens, window):
    pairs = []
    for i, w in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j != i:
                pairs.append((w, tokens[j]))
    return pairs


n_pairs_kb = sum(len(window_pairs(s, 3)) for s in kb_sents)

# %% [markdown]
# ### Exercise 2: PMI by hand
# From the count matrix `C` (window 2) compute, **without calling `ppmi_matrix`**, the plain PMI (natural log, no 0.75 smoothing) of the pair `("great", "quality")`:
# `log( (C[w,c] / T) / ((rowsum_w / T) * (colsum_c / T)) )` with `T = C.sum()`. Store it in `pmi_great_quality`. Then store in `is_associated` whether it is positive (they occur together more than chance).

# %% tags=["starter"]
pmi_great_quality = ...
is_associated = ...

# %%
T = C.sum()
i, j = index["great"], index["quality"]
expected = np.log((C[i, j] / T) / ((C[i].sum() / T) * (C[:, j].sum() / T)))
assert abs(pmi_great_quality - expected) < 1e-9, f"expected PMI {expected:.4f}"
assert is_associated == (expected > 0), "positive PMI means above-chance co-occurrence"
assert np.allclose(ppmi_matrix(C, alpha=1.0)[i, j], max(expected, 0)), "with alpha=1 the library function must agree with your hand computation"
print(f"Exercise 2 passed (PMI = {pmi_great_quality:.3f})")

# %% tags=["solution"]
T = C.sum()
i, j = index["great"], index["quality"]
pmi_great_quality = float(np.log((C[i, j] / T) / ((C[i].sum() / T) * (C[:, j].sum() / T))))
is_associated = bool(pmi_great_quality > 0)

# %% [markdown]
# ### Exercise 3: Nearest neighbours of a business word
# Using the 20-dimensional embedding `E`, store in `support_nn` the list of the **3 words most similar (by cosine) to `"support"`**, excluding the word itself, most similar first.
# Do it with your own NumPy code (not `neighbours`). The check recomputes via a different route.

# %% tags=["starter"]
support_nn = ...

# %%
vecs = E / np.linalg.norm(E, axis=1, keepdims=True)
sims = {w: float(vecs[index["support"]] @ vecs[index[w]]) for w in vocab if w != "support"}
expected = sorted(sims, key=sims.get, reverse=True)[:3]
assert list(support_nn) == expected, f"expected {expected}, got {list(support_nn)}"
print("Exercise 3 passed:", support_nn)

# %% tags=["solution"]
Eu = E / np.linalg.norm(E, axis=1, keepdims=True)
scores = Eu @ Eu[index["support"]]
support_nn = [vocab[i] for i in np.argsort(-scores) if vocab[i] != "support"][:3]

# %% [markdown]
# ### Exercise 4: How many dimensions do you need?
# `S` holds the singular values of the PPMI matrix. Store in `k90` the smallest number of dimensions whose squared singular values add up to at least 90% of the total (`sum(S**2)`).
# Then store in `k90_share` the share of the vocabulary size that `k90` is (`k90 / len(vocab)`). Is the PPMI matrix highly compressible, or does it need many dimensions? (The answer is in the number.)

# %% tags=["starter"]
k90 = ...
k90_share = ...

# %%
cum = np.cumsum(S**2) / np.sum(S**2)
exp_k = int(np.searchsorted(cum, 0.90) + 1)
assert k90 == exp_k, f"expected {exp_k}"
assert abs(k90_share - exp_k / len(vocab)) < 1e-12, "share of the vocabulary size"
assert 1 <= k90 <= len(vocab)
print(f"Exercise 4 passed ({k90} of {len(vocab)} dimensions = {k90_share:.0%} capture 90% of the energy)")

# %% tags=["solution"]
cum = np.cumsum(S**2) / np.sum(S**2)
k90 = int(np.argmax(cum >= 0.90)) + 1
k90_share = k90 / len(vocab)

# %% [markdown]
# ### Exercise 5 (open-ended): Break it
# Try to find a pair of words where the vectors disagree with your intuition (an antonym pair that is close, two synonyms that are far, or a word whose neighbours make no sense). Explain in two or three sentences
# *why the corpus caused it* (frequency? template structure? a word seen only a few times?) and what data you would add to fix it. Then change `window` from 2 to 5 in the `cooccurrence` call, rebuild `P` and `E`,
# and describe how the neighbours of `"great"` change. (Short windows capture syntax-like similarity, long windows more topical similarity.)

# %% [markdown]
# ## Recap
# - A word is described by the company it keeps: count co-occurrences in a window, giving a word-by-word matrix.
# - Raw counts are dominated by frequent words; **PPMI** keeps the above-chance associations. SVD compresses PPMI rows into dense vectors, and cosine similarity finds neighbours.
# - Validate with a downstream task as well as neighbours. Expect antonyms to be close, rare words to be poor, and no sense of word order or polysemy.
# - Neural embeddings (word2vec, the embedding layer, transformer inputs) are the same lookup-table idea, trained instead of counted.
#
# **Go deeper:** Jurafsky and Martin, *Speech and Language Processing* (free online draft), chapter on vector semantics and embeddings; Levy and Goldberg (2014), "Neural Word Embedding as Implicit Matrix Factorization"; the original word2vec paper by Mikolov et al. (2013).
#
# **Next:** [Sequence models for text](04_sequence_models_for_text.ipynb)
