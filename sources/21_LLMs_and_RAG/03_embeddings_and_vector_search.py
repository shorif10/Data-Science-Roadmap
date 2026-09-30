# %% [markdown]
# # Embeddings and Vector Search
#
# **Why this matters.** A support bot cannot paste the whole company handbook into every prompt. It has to *find* the two or
# three passages that answer this question. Keyword search (`grep`) finds "refund" but misses "get my money back". Embeddings
# turn text into vectors so that "similar meaning" becomes "nearby in space", and a vector index makes that search fast. This is
# the retrieval half of RAG (next notebook), and it is also how semantic search, deduplication and recommendations work.
#
# **You will learn**
# - build TF-IDF vectors and a hashing-based embedding from scratch, and compare them with `hit@k`
# - compute cosine similarity correctly (normalisation, zero vectors)
# - write a small exact vector index with NumPy, persist it, and time it
# - explain approximate nearest-neighbour (ANN) search by implementing random-hyperplane hashing and measuring its recall
# - say what real embedding models add, and what you would change to use one
#
# **Prerequisites:** [How LLMs work](01_how_llms_work.ipynb), [NumPy arrays](../04_NumPy/) and a little linear algebra
# (dot product, vector length).
#
# > Data used: the ten knowledge-base documents in `datasets/knowledge_base/` (synthetic company policies), split into one passage
# > per line. Real embedding models are **not** run here (they need a model download or an API); everything runs offline.

# %%
import hashlib
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

KB_DIR = Path("../datasets/knowledge_base")
docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*.md"))}

passages = []
for name, text in docs.items():
    lines = text.strip().splitlines()
    title = lines[0].lstrip("# ").strip()
    for i, line in enumerate(lines[1:], start=1):
        passages.append({"id": f"{name}#{i}", "doc": name, "text": f"{title}. {line}"})
print(len(passages), "passages from", len(docs), "documents")
print(passages[3])

# %% [markdown]
# ## 1. The problem: keywords are not meaning
#
# A user asks *"How do I get my money back?"* The knowledge base says *"Refunds are issued to the original payment method..."*.
# They share **no** content word. Exact keyword matching cannot connect them; you would need to anticipate every paraphrase.
# An **embedding** maps each text to a vector such that texts with related meaning have vectors pointing in similar directions.
# Search then becomes: embed the query, find the nearest passage vectors.
#
# We will build three embedders of increasing "semantic" power, all with the same interface, and **measure** them on a small
# question set instead of trusting intuition. First the question set: 12 questions, each with the document that contains the answer.

# %%
QUESTIONS = [
    ("How many days do I have to send back a pair of headphones?", "returns_policy"),
    ("Does express delivery cost extra?", "shipping"),
    ("What cards can I pay with?", "payment"),
    ("Is a cracked phone screen from a drop covered?", "warranty"),
    ("How much is a premium subscription per year?", "premium_membership"),
    ("Can I delete my personal information?", "privacy"),
    ("Is there a phone line open on weekends?", "support_hours"),
    ("How many points do I get for spending a dollar?", "loyalty_points"),
    ("I changed my mind before it shipped, can I cancel the order?", "cancellations"),
    ("Is your packaging recyclable?", "sustainability"),
    ("Do you deliver to Canada?", "shipping"),
    ("When does my loyalty balance run out?", "loyalty_points"),
]
assert all(gold in docs for _, gold in QUESTIONS)
print(len(QUESTIONS), "questions written by hand; gold = the document that contains the answer")

# %% [markdown]
# ## 2. Embedder #1: TF-IDF
#
# **Bag of words** represents a text as a vector of word counts (one dimension per vocabulary word). Raw counts overweight
# common words like "the". **TF-IDF** rescales each word by how *rare* it is across passages:
# `idf(w) = ln((1 + n) / (1 + df(w))) + 1`, where `n` is the number of passages and `df` the number containing `w`.
# Then each vector is scaled to length 1 (**L2 normalisation**), so a long passage does not win just by being long.

# %%
TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text):
    return TOKEN.findall(text.lower())


def l2norm(x):
    """Scale rows (or a single vector) to unit length; zero vectors stay zero instead of becoming NaN."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


class TfidfEmbedder:
    def fit(self, texts):
        toks = [tokenize(t) for t in texts]
        self.vocab = {w: i for i, w in enumerate(sorted({w for t in toks for w in t}))}
        df = np.zeros(len(self.vocab))
        for t in toks:
            for w in set(t):
                df[self.vocab[w]] += 1
        self.idf = np.log((1 + len(texts)) / (1 + df)) + 1
        return self

    def embed(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            for w in tokenize(t):
                if w in self.vocab:              # words never seen in fit are ignored
                    X[r, self.vocab[w]] += 1
        return l2norm(X * self.idf)


texts = [p["text"] for p in passages]
tfidf = TfidfEmbedder().fit(texts)
P_tfidf = tfidf.embed(texts)
print("matrix:", P_tfidf.shape, "(passages x vocabulary words)")

# Trust but verify: does our implementation match scikit-learn's TfidfVectorizer?
sk = TfidfVectorizer().fit(texts)
assert list(sk.get_feature_names_out()) == sorted(tfidf.vocab), "same vocabulary"
assert np.allclose(sk.transform(texts).toarray(), P_tfidf), "same vectors"
print("matches sklearn TfidfVectorizer")

# %% [markdown]
# *What just happened?* We built a sparse-looking (mostly zeros) matrix with one row per passage and verified it against
# scikit-learn. Each dimension *is* a word, so this embedding is interpretable, and it can only ever match words that literally appear.
#
# ## 3. Cosine similarity
#
# Two vectors point in similar directions when the angle between them is small. **Cosine similarity** is `a . b / (|a| |b|)`.
# If both are already unit length, it is just the dot product, and scoring a query against *every* passage is one matrix-vector
# product.

# %%
def cosine_matrix(A, B):
    """All pairwise cosine similarities between rows of A and rows of B."""
    return l2norm(A) @ l2norm(B).T


q = tfidf.embed(["How many days do I have to send back a pair of headphones?"])[0]
scores = P_tfidf @ q
for i in np.argsort(-scores)[:3]:
    print(f"{scores[i]:.3f}  {passages[i]['id']:<22} {passages[i]['text'][:70]}")

# %% [markdown]
# Now an evaluation function. **hit@k** is the fraction of questions where at least one of the top-`k` retrieved passages comes
# from the gold document. It is the retrieval metric you will use throughout the RAG notebooks: if the right passage is not
# retrieved, no prompt can rescue the answer.

# %%
def hit_at_k(search, questions, k):
    """`search(question, k)` must return a list of document names (one per retrieved passage)."""
    hits = [gold in search(question, k) for question, gold in questions]
    return float(np.mean(hits))


def make_search(embedder, P):
    def search(question, k):
        s = P @ embedder.embed([question])[0]
        return [passages[i]["doc"] for i in np.argsort(-s)[:k]]
    return search


search_tfidf = make_search(tfidf, P_tfidf)
print({k: round(hit_at_k(search_tfidf, QUESTIONS, k), 2) for k in (1, 3, 5)})

# %%
misses = [(qq, gold, search_tfidf(qq, 1)[0]) for qq, gold in QUESTIONS if search_tfidf(qq, 1)[0] != gold]
for qq, gold, got in misses:
    print(f"Q: {qq}\n   gold={gold}  top-1 got={got}")

# %% [markdown]
# Two things to read off this. First, **hit@1 is lower than hit@3**: the right passage is often *near* the top but not first, so
# how many passages you retrieve (`k`) is a design parameter that trades recall against prompt length. Second, the misses show
# why word overlap is a blunt instrument: function words such as "do", "I" or "does" (we did not remove stop words) and
# unrelated shared words can outvote the one informative word. Real search engines add stop-word lists, stemming and field
# weighting; embeddings attack the problem differently.

# %% [markdown]
# ## 4. Embedder #2: hashing (with character n-grams)
#
# TF-IDF needs a vocabulary fitted on your data, and it breaks on unseen forms ("refunds" versus "refund"). The **hashing trick**
# skips the vocabulary: hash each feature to one of `dim` buckets (with a random sign so collisions tend to cancel). Using
# **character n-grams** as features makes the vectors robust to plurals, typos and word forms, because "refund" and "refunds" share
# most of their 3- and 4-grams. Use a stable hash (`hashlib`), never Python's built-in `hash()`, which is randomised per process
# for strings.

# %%
def _bucket(feature, dim):
    d = hashlib.md5(feature.encode("utf-8")).digest()
    return int.from_bytes(d[:4], "little") % dim, (1.0 if d[4] & 1 else -1.0)


class HashingEmbedder:
    def __init__(self, dim=512, mode="char", ngrams=(3, 4)):
        self.dim, self.mode, self.ngrams = dim, mode, ngrams

    def _features(self, text):
        words = tokenize(text)
        if self.mode == "word":
            return words
        feats = []
        for w in words:
            padded = f" {w} "
            for n in self.ngrams:
                feats += [padded[i:i + n] for i in range(len(padded) - n + 1)]
        return feats

    def embed(self, texts):
        X = np.zeros((len(texts), self.dim))
        for r, t in enumerate(texts):
            for f in self._features(t):
                j, sign = _bucket(f, self.dim)
                X[r, j] += sign
        return l2norm(X)


for mode in ("word", "char"):
    emb = HashingEmbedder(mode=mode)
    P = emb.embed(texts)
    s = make_search(emb, P)
    print(f"hashing/{mode:<4} hit@1={hit_at_k(s, QUESTIONS, 1):.2f}  hit@3={hit_at_k(s, QUESTIONS, 3):.2f}")

emb_c = HashingEmbedder(mode="char")
v_refund, v_refunds, v_warranty = emb_c.embed(["refund", "refunds", "warranty"])
print("cosine(refund, refunds)  =", round(float(v_refund @ v_refunds), 2))
print("cosine(refund, warranty) =", round(float(v_refund @ v_warranty), 2))

# %% [markdown]
# Character n-grams make morphological variants land close together, and they need no fitting step, so new documents and
# new words are handled with no retraining. They still know nothing about *meaning*: "money back" and "refund" share no
# letters. Note also the **collision** trade-off: with a small `dim`, unrelated features share buckets and add noise; a
# bigger `dim` costs memory. Let's measure that instead of guessing.

# %%
dims = [16, 32, 64, 128, 256, 512, 1024, 2048]
hit3 = []
for d in dims:
    emb = HashingEmbedder(dim=d)
    hit3.append(hit_at_k(make_search(emb, emb.embed(texts)), QUESTIONS, 3))
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(dims, hit3, marker="o")
ax.set_xscale("log", base=2)
ax.set_title("Hashing embedder: retrieval quality vs. number of buckets")
ax.set_xlabel("dimensions (hash buckets)")
ax.set_ylabel("hit@3 on 12 questions")
plt.show()
print(dict(zip(dims, [round(h, 2) for h in hit3])))

# %% [markdown]
# ## 5. Embedder #3: a learned low-dimensional space (LSA)
#
# Real embedding models are *trained* so that meaning-related texts end up close. The oldest classical version of that idea is
# **latent semantic analysis (LSA)**: take the TF-IDF matrix and compress it with a truncated SVD into a few dense dimensions.
# Words that appear in similar passages get similar coordinates, so a query can match a passage even without sharing a word,
# *provided the corpus shows they co-occur*. Our corpus has about 40 short passages, so expect noise: treat this as a
# demonstration of the mechanism, not of quality.

# %%
class LSAEmbedder:
    def __init__(self, n_components=12):
        self.n_components = n_components

    def fit(self, texts):
        self.tfidf = TfidfEmbedder().fit(texts)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42).fit(self.tfidf.embed(texts))
        return self

    def embed(self, texts):
        return l2norm(self.svd.transform(self.tfidf.embed(texts)))


rows = []
for nc in (4, 8, 12, 20, 30):
    lsa = LSAEmbedder(nc).fit(texts)
    s = make_search(lsa, lsa.embed(texts))
    rows.append((nc, hit_at_k(s, QUESTIONS, 1), hit_at_k(s, QUESTIONS, 3)))
print(f"{'dims':>5} {'hit@1':>6} {'hit@3':>6}")
for nc, h1, h3 in rows:
    print(f"{nc:>5} {h1:>6.2f} {h3:>6.2f}")

# %% [markdown]
# *What just happened?* Compressing to a handful of dense dimensions forces the model to keep the *shared structure* and
# discard word-specific detail. Too few dimensions blur unrelated topics together; enough dimensions approach plain TF-IDF.
# A real model (a neural network trained on huge amounts of text) plays the same game with far better data and a far better
# function, which is why it can connect "money back" with "refund".
#
# Here is a summary table of everything so far on the same questions:

# %%
lsa = LSAEmbedder(12).fit(texts)
embedders = {
    "tfidf": (tfidf, P_tfidf),
    "hash-word": (HashingEmbedder(mode="word"), None),
    "hash-char": (HashingEmbedder(mode="char"), None),
    "lsa-12": (lsa, lsa.embed(texts)),
}
summary = {}
for name, (emb, P) in embedders.items():
    P = emb.embed(texts) if P is None else P
    s = make_search(emb, P)
    summary[name] = {f"hit@{k}": round(hit_at_k(s, QUESTIONS, k), 2) for k in (1, 3, 5)}
for name, r in summary.items():
    print(f"{name:<10}", r)

# %% [markdown]
# Small question sets are noisy: with 12 questions, one question is 8 percentage points. Use these numbers to *compare* methods
# roughly and to learn the workflow, not as a benchmark. In a real project you would build a much larger labelled set from
# real user questions (notebook 05).

# %% [markdown]
# ## 6. A vector index from scratch
#
# A **vector index** stores vectors with their payloads (here: the passage) and answers "top-`k` most similar to this query".
# The exact version is a single matrix product, which is surprisingly fast for up to hundreds of thousands of vectors.

# %%
class VectorIndex:
    """Exact (brute-force) cosine-similarity index."""

    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.items = []

    def add(self, vectors, items):
        vectors = l2norm(vectors).astype(np.float32)
        assert vectors.shape[1] == self.dim and len(vectors) == len(items)
        self.vectors = np.vstack([self.vectors, vectors])
        self.items.extend(items)

    def search(self, query, k=3):
        q = l2norm(query).astype(np.float32)
        scores = self.vectors @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]           # k best without a full sort
        top = top[np.argsort(-scores[top])]                 # order just those k
        return [(self.items[i], float(scores[i])) for i in top]

    def save(self, path):
        np.savez(path, vectors=self.vectors)          # payloads are stored elsewhere (a database, a JSON file)

    @classmethod
    def load(cls, path, items):
        data = np.load(path)
        idx = cls(data["vectors"].shape[1])
        idx.vectors, idx.items = data["vectors"], items
        return idx


index = VectorIndex(dim=P_tfidf.shape[1])
index.add(P_tfidf, passages)
for item, score in index.search(tfidf.embed(["Do you deliver to Canada?"])[0], k=2):
    print(f"{score:.3f}  {item['id']:<20} {item['text'][:60]}")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "kb_index.npz"
    index.save(path)
    restored = VectorIndex.load(path, passages)
    same = restored.search(tfidf.embed(["Do you deliver to Canada?"])[0], k=2)[0][0]["id"]
print("restored index answers the same:", same == index.search(tfidf.embed(["Do you deliver to Canada?"])[0], k=1)[0][0]["id"])

# %% [markdown]
# Three design points that matter in real systems:
# - **Store unit vectors** and use dot product; it is the fastest form of cosine similarity.
# - **Never mix embedders.** Query and passages must come from the *same* model (and version). Changing the model means re-embedding everything.
# - **Keep the payload with the vector** (text, source, position) so you can cite where an answer came from.
#
# How fast is brute force? Let's measure on 100,000 random vectors instead of guessing.

# %%
rng = np.random.default_rng(42)
big = VectorIndex(dim=128)
big.add(rng.normal(size=(100_000, 128)), list(range(100_000)))
qv = rng.normal(size=128)
t0 = time.perf_counter()
for _ in range(20):
    big.search(qv, k=5)
ms = (time.perf_counter() - t0) / 20 * 1000
print(f"exact search over {len(big.items):,} vectors of dimension 128: {ms:.1f} ms per query on this machine")

# %% [markdown]
# ## 7. Approximate nearest neighbours (ANN): the idea
#
# Brute force costs time proportional to the number of vectors. At tens of millions of vectors, or when you need many queries
# per second, you trade a little **recall** (occasionally missing a true neighbour) for a big speed-up by comparing the query
# against a small *candidate set*. Production systems use structures such as graph indexes (HNSW) or clustering-based indexes
# (IVF); all follow the same recipe: organise vectors so that nearby ones are found together, then search only the nearby region.
#
# The simplest version to implement is **random-hyperplane hashing (LSH)**: pick random directions; the sign of the dot product with
# each direction is one bit; vectors with the same bit pattern share a bucket. Similar vectors agree on most bits. We test it on
# synthetic clustered data and measure recall against the exact answer.

# %%
rng = np.random.default_rng(0)
n, d = 20_000, 64
centers = rng.normal(size=(50, d)) * 3
data = l2norm(centers[rng.integers(0, 50, n)] + rng.normal(size=(n, d)))
queries = l2norm(data[rng.choice(n, 200, replace=False)] + 0.3 * rng.normal(size=(200, d)))


class LSHIndex:
    def __init__(self, data, n_bits=10, n_tables=8, seed=1):
        r = np.random.default_rng(seed)
        self.data = data
        self.planes = [r.normal(size=(data.shape[1], n_bits)) for _ in range(n_tables)]
        self.weights = 1 << np.arange(n_bits)
        self.tables = []
        for P in self.planes:
            codes = ((data @ P) > 0).astype(int) @ self.weights
            table = {}
            for i, c in enumerate(codes):
                table.setdefault(int(c), []).append(i)
            self.tables.append(table)

    def search(self, q, k=10):
        cand = set()
        for P, table in zip(self.planes, self.tables):
            cand.update(table.get(int(((q @ P) > 0).astype(int) @ self.weights), []))
        cand = np.fromiter(cand, dtype=int)
        if len(cand) == 0:
            return [], 0
        s = self.data[cand] @ q
        return list(cand[np.argsort(-s)[:k]]), len(cand)


def evaluate_lsh(n_bits, n_tables):
    lsh = LSHIndex(data, n_bits=n_bits, n_tables=n_tables)
    recalls, cand_frac = [], []
    for q in queries:
        exact = set(np.argsort(-(data @ q))[:10])
        approx, n_cand = lsh.search(q, 10)
        recalls.append(len(exact & set(approx)) / 10)
        cand_frac.append(n_cand / n)
    return float(np.mean(recalls)), float(np.mean(cand_frac))


sweep = {(b, t): evaluate_lsh(b, t) for b in (4, 6, 8, 10) for t in (2, 8, 16)}
print(f"{'bits':>4} {'tables':>6} {'recall@10':>10} {'% of data compared':>19}")
for (b, t), (rec, frac) in sweep.items():
    print(f"{b:>4} {t:>6} {rec:>10.2f} {frac:>19.1%}")

# %% [markdown]
# *What just happened?* Each row of the table is one point on the recall-versus-work curve: the index compares each query against only a fraction of
# the data and finds some share of the true top-10. More tables, or fewer bits per table, raises recall and also raises the candidate count; that recall-versus-work dial exists in
# every ANN index (in HNSW it is the search-breadth parameter). Notice that this plain LSH is a *weak* baseline on our data: high
# recall only appears in rows where it compares a large share of the vectors. Graph-based and cluster-based indexes are used in
# practice because they usually reach good recall while touching far less data; you would measure that on your own vectors.
# Whether the recall you get is acceptable is an application
# question: for RAG over a few thousand passages, exact search is simpler and you should just use it.

# %% [markdown]
# ## 8. Real embedding models (no-run)
#
# A real **embedding model** is a neural network trained (typically with contrastive objectives on huge text collections) to
# map text to a dense vector, commonly a few hundred to a few thousand dimensions, so that semantically related texts are close in cosine similarity.
# Compared with our embedders they handle paraphrase, synonyms and (for multilingual models) other languages. They also cost
# something: a model download plus CPU/GPU time when run locally, or an API call per text (latency, money, and your text
# leaving your network). They have context limits too, which is why documents are split into passages before embedding.
#
# The cells below are **not executed here**: the first needs a one-time model download (network), the second needs an
# embeddings API account. Anthropic's documentation describes which embedding providers it recommends; check it for the
# current list rather than relying on this notebook. Note how little of *your* code changes: everything above only needs an
# object with `.embed(list_of_texts) -> array`.

# %% tags=["no-run"]
# Option A: a local open model via the sentence-transformers package (pip install sentence-transformers; downloads weights once)
from sentence_transformers import SentenceTransformer


class LocalModelEmbedder:
    def __init__(self, name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)

    def embed(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True)


real = LocalModelEmbedder()
P_real = real.embed(texts)
print(hit_at_k(make_search(real, P_real), QUESTIONS, 3))   # same evaluation code as above

# %% tags=["no-run"]
# Option B: a hosted embeddings API (each provider has its own client and key; read the key from the environment, never hard-code it)
import os

api_key = os.environ["EMBEDDING_PROVIDER_API_KEY"]   # provider-specific; see the provider's documentation for the call itself

# %% [markdown]
# ## Common mistakes
#
# 1. **Comparing vectors from different embedding models** (or different versions), or forgetting to re-embed the corpus after switching.
# 2. **Cosine similarity without normalising**: raw dot products favour long vectors. Normalise once, at insert time.
# 3. **NaN from zero vectors** when a text has no known words: guard the division (see `l2norm`).
# 4. **Fitting TF-IDF on the test questions** (leakage), or refitting it per query so the IDF changes between requests.
# 5. **Embedding whole documents** when the answer is one sentence: the vector becomes a blurry average. Split into passages (next notebook).
# 6. **Python's `hash()` for feature hashing**: it is randomised per process, so your vectors change between runs. Use `hashlib`.
# 7. **Trusting a tiny evaluation.** 12 questions cannot tell 0.83 from 0.92.

# %%
# Demonstration of mistake 3
print("unit norm of an all-zero vector without the guard would be NaN; with the guard:", l2norm(np.zeros(3)))
unknown = tfidf.embed(["zzzz qqqq"])[0]
print("query with only unknown words -> all-zero vector:", bool((unknown == 0).all()), " scores:", np.unique(P_tfidf @ unknown))

# %% [markdown]
# ## Interview corner
#
# **1. Why use cosine similarity instead of Euclidean distance for text embeddings?**
# <details><summary>Answer</summary>
# Cosine compares direction and ignores length; embedding magnitude often reflects things like text length or term frequency rather than
# meaning. For unit-length vectors the two are equivalent for ranking (squared Euclidean distance is 2 - 2 cos), so the practical rule is:
# normalise once, then use whichever your index computes fastest, usually the dot product.
# </details>
#
# **2. Exact versus approximate nearest-neighbour search: when do you need ANN?**
# <details><summary>Answer</summary>
# Exact search costs time proportional to the number of vectors. It is simple, has perfect recall, and is fine up to hundreds of thousands
# of vectors (measure it). ANN (HNSW, IVF, LSH) trades some recall for sub-linear query time, needed at millions or billions of vectors or
# very high query rates. Always measure recall against exact search on your own data.
# </details>
#
# **3. TF-IDF versus embeddings for search: what is the difference and when does TF-IDF/BM25 still win?**
# <details><summary>Answer</summary>
# TF-IDF/BM25 match exact words, are interpretable, need no model and are strong for rare identifiers (error codes, product SKUs, names).
# Embeddings match meaning, handling paraphrase and synonyms, but can miss exact tokens. Hybrid search (combine both scores) is common.
# </details>
#
# **4. Why do you split documents into chunks before embedding?**
# <details><summary>Answer</summary>
# Models have input limits, one vector per long document blurs many topics together, and you want to hand the LLM only the relevant passage
# (fewer tokens, better focus, precise citations). Chunk size is a trade-off you tune with retrieval metrics.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Safe pairwise cosine similarity
# Write `cos_sim(A, B)` returning the matrix of cosine similarities between every row of `A` and every row of `B` (shape `len(A) x len(B)`),
# **without raising or returning NaN when a row is all zeros** (its similarity to everything should be 0). *Hint: `l2norm` above already
# handles the zero case.*

# %% tags=["starter"]
def cos_sim(A, B):
    ...


# %%
from sklearn.metrics.pairwise import cosine_similarity

A = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 0.5, 2.0]])
B = np.array([[3.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
got = cos_sim(A, B)
assert got.shape == (3, 2), f"expected shape (3, 2), got {got.shape}"
assert not np.isnan(got).any(), "zero rows must not produce NaN"
assert np.allclose(got[1], 0), "an all-zero vector has similarity 0 to everything"
assert np.allclose(got[[0, 2]], cosine_similarity(A[[0, 2]], B)), "must agree with scikit-learn on normal rows"
print("Exercise 1 passed")

# %% tags=["solution"]
def cos_sim(A, B):
    return l2norm(A) @ l2norm(B).T


# %% [markdown]
# ### Exercise 2: Your own hashing embedder
# Write `word_hash_embed(text, dim=128)`: hash each lowercase word (use `tokenize`) with `_bucket(word, dim)`, which returns `(index, sign)`,
# add `sign` to that position, and return the vector **L2-normalised**. It must be deterministic and return a zero vector for empty text.
# *Hint: this is `HashingEmbedder(mode="word")` for a single text.*

# %% tags=["starter"]
def word_hash_embed(text, dim=128):
    ...


# %%
v = word_hash_embed("Premium members get free returns", dim=128)
assert v.shape == (128,), "one vector of length dim"
assert abs(np.linalg.norm(v) - 1) < 1e-9, "must be unit length"
assert np.array_equal(v, word_hash_embed("Premium members get free returns", dim=128)), "must be deterministic"
assert np.array_equal(word_hash_embed("free RETURNS premium get members", 128), v), "word order and case must not matter"
assert not word_hash_embed("", 64).any(), "empty text -> zero vector"
sim_close = float(word_hash_embed("free returns for premium members") @ word_hash_embed("premium members get free returns"))
sim_far = float(word_hash_embed("customs fees for canada") @ word_hash_embed("premium members get free returns"))
assert sim_close > sim_far, "a text sharing words must be closer than an unrelated text"
print("Exercise 2 passed")

# %% tags=["solution"]
def word_hash_embed(text, dim=128):
    v = np.zeros(dim)
    for w in tokenize(text):
        j, sign = _bucket(w, dim)
        v[j] += sign
    return l2norm(v)


# %% [markdown]
# ### Exercise 3: Retrieval metric with a floor
# `hit_at_k` counts a hit if the gold document is anywhere in the top `k`. Write `mrr(search, questions, k=5)`, the **mean reciprocal rank**:
# for each question, `1 / rank` of the first passage from the gold document within the top `k` (rank starts at 1), or 0 if it is not there;
# return the mean over questions. MRR rewards putting the right passage *first*, which `hit@k` cannot see.

# %% tags=["starter"]
def mrr(search, questions, k=5):
    ...


# %%
perfect = lambda question, k: [dict(QUESTIONS)[question]] + ["x"] * (k - 1)
second = lambda question, k: ["x", dict(QUESTIONS)[question]] + ["x"] * (k - 2)
never = lambda question, k: ["x"] * k
assert abs(mrr(perfect, QUESTIONS) - 1.0) < 1e-9, "gold always first -> MRR 1"
assert abs(mrr(second, QUESTIONS) - 0.5) < 1e-9, "gold always second -> MRR 0.5"
assert mrr(never, QUESTIONS) == 0.0, "gold never retrieved -> MRR 0"
assert abs(mrr(second, QUESTIONS, k=1) - 0.0) < 1e-9, "outside the top k counts as a miss"
print(f"Exercise 3 passed (tf-idf MRR@5 on our questions: {mrr(search_tfidf, QUESTIONS):.2f})")

# %% tags=["solution"]
def mrr(search, questions, k=5):
    total = 0.0
    for question, gold in questions:
        ranked = search(question, k)[:k]
        if gold in ranked:
            total += 1.0 / (ranked.index(gold) + 1)
    return total / len(questions)


# %% [markdown]
# ### Exercise 4: Know when to say "nothing relevant"
# A retriever always returns *something*. Add a method to `VectorIndex` (define a subclass `ThresholdIndex(VectorIndex)`) called
# `search_min(query, k=3, min_score=0.2)` that returns only results whose cosine score is at least `min_score` (possibly an empty list).
# Then set `n_answerable` to the number of `QUESTIONS` for which the TF-IDF index returns at least one result with `min_score=0.25`
# and `n_junk` to the number of these off-topic queries that return at least one result: `["quantum physics lecture", "zzzz qqqq", "best pizza in Rome"]`.

# %% tags=["starter"]
class ThresholdIndex(VectorIndex):
    ...


n_answerable = ...
n_junk = ...

# %%
tidx = ThresholdIndex(dim=P_tfidf.shape[1])
tidx.add(P_tfidf, passages)
r_all = tidx.search_min(tfidf.embed(["Do you deliver to Canada?"])[0], k=3, min_score=0.0)
assert len(r_all) == 3, "min_score=0 keeps the top k"
assert tidx.search_min(tfidf.embed(["zzzz qqqq"])[0], k=3, min_score=0.01) == [], "no matching words -> empty list"
assert all(sc >= 0.25 for _, sc in tidx.search_min(tfidf.embed(["Do you deliver to Canada?"])[0], 5, 0.25))
exp_ans = sum(len(tidx.search(tfidf.embed([qq])[0], 3)) > 0 and tidx.search(tfidf.embed([qq])[0], 3)[0][1] >= 0.25 for qq, _ in QUESTIONS)
assert n_answerable == exp_ans, f"n_answerable should be {exp_ans}, got {n_answerable}"
exp_junk = sum(tidx.search(tfidf.embed([qq])[0], 3)[0][1] >= 0.25 for qq in ["quantum physics lecture", "zzzz qqqq", "best pizza in Rome"])
assert n_junk == exp_junk, f"n_junk should be {exp_junk}, got {n_junk}"
print(f"Exercise 4 passed ({n_answerable}/{len(QUESTIONS)} real questions above the threshold, {n_junk}/3 junk queries above it)")

# %% tags=["solution"]
class ThresholdIndex(VectorIndex):
    def search_min(self, query, k=3, min_score=0.2):
        return [(item, s) for item, s in self.search(query, k) if s >= min_score]


_t = ThresholdIndex(dim=P_tfidf.shape[1])
_t.add(P_tfidf, passages)
n_answerable = sum(bool(_t.search_min(tfidf.embed([qq])[0], 3, 0.25)) for qq, _ in QUESTIONS)
n_junk = sum(bool(_t.search_min(tfidf.embed([qq])[0], 3, 0.25)) for qq in ["quantum physics lecture", "zzzz qqqq", "best pizza in Rome"])


# %% [markdown]
# ### Exercise 5 (open-ended): Where does keyword-style retrieval fail?
# Write **five new questions** about the knowledge base that you think a keyword method will get wrong (paraphrases, synonyms, indirect wording).
# Run them through `search_tfidf` and the `hash-char` search. Which fail? Does character n-gram hashing rescue any? Write 3-4 sentences in a markdown
# cell about what kind of failure each embedder has, and what a real embedding model would need to do to fix the rest.

# %% [markdown]
# ## Recap
# - An embedding maps text to a vector so that "related" becomes "near"; retrieval is *embed the query, find nearest passages*.
# - TF-IDF = interpretable word-weight vectors; hashing = no vocabulary, robust with character n-grams; LSA = compressed dense space. All share `.embed(texts)`.
# - Normalise once, use dot product as cosine; guard zero vectors; never mix embedders between index and query.
# - `hit@k` and MRR measure retrieval on a labelled question set; small sets are noisy, so compare methods roughly.
# - An exact index is one matrix product and is enough for small and medium corpora; ANN trades recall for speed and you must measure that recall.
# - Real embedding models capture meaning far better but need a download or an API; the rest of your code stays the same.
#
# **Go deeper:** the scikit-learn user guide on text feature extraction and on TruncatedSVD/LSA (scikit-learn.org); the FAISS project
# documentation for production ANN indexes; the Anthropic documentation section on embeddings (docs.anthropic.com) for current provider guidance.
#
# **Next:** [Build RAG from scratch](04_build_rag_from_scratch.ipynb)
