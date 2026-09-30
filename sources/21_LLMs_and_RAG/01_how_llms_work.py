# %% [markdown]
# # How LLMs Work: Tokens, Context, Next-Token Probabilities, Sampling
#
# **Why this matters.** Every LLM feature you will ship (a support bot, a SQL assistant, an agent) sits on
# one mechanism: a model that reads a bounded sequence of tokens and produces a probability distribution over
# the *next* token. Most production surprises (truncated answers, a bill that doubles, an answer that changes
# between runs, a confident wrong fact) are direct consequences of that mechanism. If you understand it, you can
# predict those problems instead of debugging them blind.
#
# **You will learn**
# - split text into tokens with a byte-pair-encoding (BPE) tokenizer that you train yourself
# - build a tiny character-level language model and read its next-token distribution
# - control generation with temperature, top-k and top-p, and see the effect on real samples
# - explain what a context window is and why chat cost grows with conversation length
# - estimate cost and latency for a workload before you build it
#
# **Prerequisites:** [Python basics](../01_Python/) and [NumPy arrays](../04_NumPy/). No deep-learning knowledge needed.
#
# > Data used: `datasets/knowledge_base/*.md`, ten short synthetic company policy documents (about 3 KB of text in total).
# > That is a *toy* corpus on purpose: small enough that you can see every count. Real LLMs train on vastly more
# > text, and nothing in this notebook reproduces their quality. It reproduces their **mechanics**.

# %%
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

KB_DIR = Path("../datasets/knowledge_base")
docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*.md"))}
corpus = "\n\n".join(docs.values())
print(len(docs), "documents,", len(corpus), "characters")
print(corpus[:160])

# %% [markdown]
# ## 1. Tokens: the unit the model actually sees
#
# A model does not read characters or words. It reads **tokens**: chunks from a fixed vocabulary, typically
# whole common words, word pieces, and single bytes as a fallback. Common strings become one token; rare strings
# split into several. This is why "how many tokens is this?" has no answer without naming the tokenizer, and why
# non-English text or code can cost more than the same amount of English.
#
# Most modern tokenizers are built with **byte-pair encoding (BPE)**: start with single characters, repeatedly
# merge the most frequent adjacent pair into a new symbol, and stop after a fixed number of merges. You can
# implement the core idea in 25 lines.

# %%
PRETOKEN = re.compile(r"\s?\w+|\s?[^\w\s]|\s")  # word (with its leading space), punctuation, or whitespace


def train_bpe(text, n_merges):
    """Learn an ordered list of merges from `text`."""
    words = Counter(PRETOKEN.findall(text))
    seqs = {w: tuple(w) for w in words}          # each pre-token starts as a tuple of characters
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, seq in seqs.items():
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += words[w]
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))   # tie-break on the pair itself: deterministic
        merges.append(best)
        for w, seq in seqs.items():
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                    out.append(seq[i] + seq[i + 1]); i += 2
                else:
                    out.append(seq[i]); i += 1
            seqs[w] = tuple(out)
    return merges


def bpe_encode(text, merges):
    tokens = []
    for w in PRETOKEN.findall(text):
        seq = list(w)
        for a, b in merges:                      # apply merges in the order they were learned
            i = 0
            while i < len(seq) - 1:
                if seq[i] == a and seq[i + 1] == b:
                    seq[i:i + 2] = [a + b]
                else:
                    i += 1
        tokens.extend(seq)
    return tokens


merges = train_bpe(corpus, 300)
sentence = "Premium members get free return shipping."
print(merges[:8])
print(bpe_encode(sentence, merges))

# %% [markdown]
# *What just happened?* The first merges are the most frequent character pairs in the corpus. By merge 300,
# frequent words such as ` return` or ` shipping` have become single tokens, while a rare word is still spelled
# out in pieces. Let's measure how the number of merges (the "vocabulary size") changes the token count.

# %%
rows = []
for n in (0, 25, 100, 300):
    m = merges[:n]
    toks = bpe_encode(corpus, m)
    rows.append((n, len(toks), len(corpus) / len(toks)))
print(f"{'merges':>7} {'tokens':>8} {'chars/token':>12}")
for n, t, r in rows:
    print(f"{n:>7} {t:>8} {r:>12.2f}")

# %% [markdown]
# More merges means fewer tokens for the same text (each token carries more characters) but a bigger vocabulary
# to learn and store. Real vocabularies are tens of thousands of tokens, trained on far more text than ours, so
# their chars-per-token differs from this table. A common rule of thumb for English prose is "roughly 4
# characters per token", which is good enough for a *rough* budget and useless for an exact one. For exact counts
# of Claude models, use the token-counting endpoint of the API (shown in a no-run cell later).
#
# We will use that rough rule as a cheap estimator through the rest of the course:

# %%
def est_tokens(text: str) -> int:
    """Rough token estimate for English text (about 4 characters per token). Not exact."""
    return max(1, math.ceil(len(text) / 4))


print("KB estimate:", est_tokens(corpus), "tokens   our BPE(300):", len(bpe_encode(corpus, merges)))

# %% [markdown]
# ## 2. A language model is a next-token probability table
#
# Strip away the neural network and a language model is a function
# `context -> probability distribution over the next token`. The simplest version is a counting table:
# a **bigram model** looks at only the previous character and counts what followed it in the training text.
# An **n-gram** model looks at the previous `n-1` characters. We train it on the knowledge base.

# %%
BOS = "\x02"  # start-of-text padding


class CharNGram:
    """Character-level n-gram language model with add-k smoothing."""

    def __init__(self, order=2, k=0.05):
        self.order, self.k = order, k
        self.vocab = []

    def fit(self, text):
        self.vocab = sorted(set(text) | {BOS})
        self.idx = {c: i for i, c in enumerate(self.vocab)}
        padded = BOS * (self.order - 1) + text
        self._table = defaultdict(Counter)   # context string (any length up to order-1) -> Counter of next chars
        for i in range(self.order - 1, len(padded)):
            for n in range(self.order):       # register the context at every length so we can back off
                self._table[padded[i - n:i]][padded[i]] += 1
        return self

    def probs(self, ctx):
        """P(next char | recent context) as a NumPy vector over self.vocab.

        If the full context (order-1 characters) was never seen in training, back off to a shorter one.
        """
        padded = (BOS * (self.order - 1) + ctx)[-(self.order - 1):] if self.order > 1 else ""
        c = {}
        for cut in range(len(padded) + 1):           # try the longest context first, then shorter suffixes
            c = self.counts_for(padded[cut:])
            if c:
                break
        v = np.array([c.get(ch, 0) for ch in self.vocab], dtype=float) + self.k
        return v / v.sum()

    def counts_for(self, ctx):
        return self._table.get(ctx, {})


bigram = CharNGram(order=2).fit(corpus)
p = bigram.probs("t")
top = np.argsort(-p)[:6]
print("after 't':", [(bigram.vocab[i], round(float(p[i]), 3)) for i in top])

# %% [markdown]
# (Two engineering details: `k` is *add-k smoothing*, so no character ever has probability exactly zero; and if a
# long context was never seen in training, `probs` **backs off** to a shorter suffix of it.)
#
# The model says that after `t` the likely characters are `h`, `i`, `space`, and so on: English-looking statistics
# learned from 3 KB. Let's look at the whole distribution for one context.

# %%
fig, ax = plt.subplots(figsize=(8, 3))
order_idx = np.argsort(-p)[:15]
ax.bar([repr(bigram.vocab[i]) for i in order_idx], p[order_idx])
ax.set_title("Bigram model: P(next character | previous character = 't')")
ax.set_xlabel("next character")
ax.set_ylabel("probability")
plt.show()

# %% [markdown]
# *What just happened?* We turned raw counts into a probability distribution. An LLM does exactly this at every
# step, except: (1) the context is thousands of tokens, not one character, (2) the "table" is computed by a neural
# network instead of looked up, so it can handle contexts it has never seen, and (3) the units are tokens.
# **Generating text is just: get the distribution, pick one token, append it, repeat.**

# %% [markdown]
# ## 3. Sampling: turning a distribution into text
#
# Choosing the next token is a *policy*, separate from the model. Three common knobs:
#
# | Knob | What it does |
# |---|---|
# | **Temperature** `T` | Divide the log-probabilities by `T` before renormalising. `T<1` sharpens (more predictable), `T>1` flattens (more random), `T->0` is greedy (always the top token). |
# | **Top-k** | Keep only the `k` most likely tokens, renormalise. |
# | **Top-p** (nucleus) | Keep the smallest set of tokens whose cumulative probability reaches `p`, renormalise. |
#
# Implementing them makes the behaviour concrete. (Exercises 1 and 2 ask you to write them yourself; here is
# a reference version of top-k first.)

# %%
def apply_temperature(probs, T):
    logits = np.log(probs + 1e-12) / max(T, 1e-6)
    logits -= logits.max()                 # numerical stability
    e = np.exp(logits)
    return e / e.sum()


def top_k_filter(probs, k):
    keep = np.argsort(-probs)[:k]
    out = np.zeros_like(probs)
    out[keep] = probs[keep]
    return out / out.sum()


def entropy_bits(probs):
    q = probs[probs > 0]
    return float(-(q * np.log2(q)).sum())


for T in (0.2, 0.5, 1.0, 2.0):
    q = apply_temperature(p, T)
    print(f"T={T:<4} top prob={q.max():.3f}   entropy={entropy_bits(q):.2f} bits")

# %% [markdown]
# Entropy is a measure of "how many plausible options are there": low temperature concentrates the mass
# (low entropy), high temperature spreads it out. Now sample real text. We use a 5-gram model (4 characters of
# context) trained on the whole corpus and generate 160 characters at different temperatures.

# %%
def generate(model, prompt, n_chars, T=1.0, top_k=None, seed=0):
    rng = np.random.default_rng(seed)
    text = prompt
    for _ in range(n_chars):
        q = apply_temperature(model.probs(text), T)
        if top_k:
            q = top_k_filter(q, top_k)
        nxt = model.vocab[rng.choice(len(q), p=q)]
        if nxt == BOS:
            continue
        text += nxt
    return text


ng5 = CharNGram(order=5, k=0.01).fit(corpus)
for T in (0.1, 0.7, 1.5):
    out = generate(ng5, "Premium ", 150, T=T, seed=1).replace("\n", " / ")
    print(f"--- T={T}\n{out}\n")

# %% [markdown]
# Read the three samples. At low temperature the model tends to reproduce policy text almost word for word
# (it **memorised** a tiny corpus and mostly follows one path); at high temperature it drifts into invented
# words. Both effects are real properties of LLMs, at very different scales: low temperature is more repeatable
# but can get stuck in loops; high temperature is more varied but more likely to say something wrong.
#
# Memorisation deserves a measurement instead of an impression. How much of the generated text is copied
# verbatim from the corpus?

# %%
def verbatim_rate(text, source, n=12):
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return sum(g in source for g in grams) / len(grams)


for T in (0.1, 0.7, 1.5):
    samples = [generate(ng5, "Premium ", 300, T=T, seed=s) for s in range(20)]
    rate = float(np.mean([verbatim_rate(s, corpus) for s in samples]))
    print(f"T={T:<4} share of 12-character windows copied verbatim from the corpus: {rate:.0%}")

# %% [markdown]
# ## 4. Bigger context is not automatically better: held-out perplexity
#
# How good is a model? A standard measure is **perplexity** on text it has *not* seen: `exp` of the average
# negative log-probability of each true next token. Loosely, "how many choices the model is choosing between
# at each step"; lower is better. We hold out two documents, train on the rest, and vary the context length.

# %%
def perplexity(model, text):
    padded = BOS * (model.order - 1) + text
    nll = 0.0
    n = 0
    for i in range(model.order - 1, len(padded)):
        ctx = padded[i - model.order + 1:i]
        q = model.probs(ctx) if model.order > 1 else model.probs("")
        ch = padded[i]
        if ch not in model.idx:
            continue                            # character never seen in training: skip
        nll -= math.log(q[model.idx[ch]])
        n += 1
    return math.exp(nll / n)


held_out = ["support_hours", "warranty"]
train_text = "\n\n".join(t for k, t in docs.items() if k not in held_out)
test_text = "\n\n".join(docs[k] for k in held_out)

results = {}
for order in range(1, 7):
    m = CharNGram(order=order, k=0.05).fit(train_text)
    results[order] = (perplexity(m, train_text), perplexity(m, test_text))
print(f"{'order':>5} {'train ppl':>10} {'held-out ppl':>13}")
for o, (tr, te) in results.items():
    print(f"{o:>5} {tr:>10.2f} {te:>13.2f}")

# %%
orders = list(results)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(orders, [results[o][0] for o in orders], marker="o", label="train")
ax.plot(orders, [results[o][1] for o in orders], marker="o", label="held-out")
ax.set_title("More context memorises training text but stops helping on new text")
ax.set_xlabel("n-gram order (characters of context + 1)")
ax.set_ylabel("perplexity (lower is better)")
ax.legend()
plt.show()
best_order = min(results, key=lambda o: results[o][1])
print("best order on held-out text:", best_order)

# %% [markdown]
# *What just happened?* Training perplexity keeps falling as the context grows (the model memorises), but
# held-out perplexity bottoms out and then rises, because long contexts in new text were never seen in training.
# This is the same **generalisation vs memorisation** tension you know from ML; it is why bigger models need
# far more data, and why an LLM can recite text it saw often while still being unreliable on facts it saw rarely.
# Neural networks fix the "never seen this exact context" problem by mapping similar contexts to similar
# vectors, which is where attention and embeddings come in.

# %% [markdown]
# ## 5. What changes at scale, and what does not
#
# **What changes** when you go from our counting table to a large transformer model:
# - the context is a long token sequence, and a mechanism called **attention** lets every position look at
#   every earlier position and decide which ones matter;
# - the probabilities come from learned weights (billions of parameters) instead of counts, so the model
#   generalises to contexts it has never seen;
# - after pretraining on next-token prediction, models are further trained to follow instructions and to avoid
#   harmful output. That is why you can *ask* them things instead of only *continuing* text.
#
# **What does not change:** the output at each step is still a distribution over tokens; text is still produced
# one token at a time; your sampling policy is still separate from the model; the model still has no built-in
# access to facts after its training data, and no built-in way to verify what it says.
#
# Attention is small enough to write in NumPy. Each position produces a *query*, *key* and *value* vector;
# the weights are `softmax(Q K^T / sqrt(d))`, with a **causal mask** so a position cannot see the future.

# %%
def causal_attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)   # True above the diagonal = future positions
    scores = np.where(mask, -np.inf, scores)
    w = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V, w


rng = np.random.default_rng(42)
T_len, d = 5, 8
Q, K, V = (rng.normal(size=(T_len, d)) for _ in range(3))
out, weights = causal_attention(Q, K, V)
print(np.round(weights, 2))
assert np.allclose(weights.sum(axis=1), 1) and np.allclose(np.triu(weights, 1), 0)

# %% [markdown]
# *What just happened?* Row `i` says how much position `i` reads from each earlier position (each row sums
# to 1, and everything above the diagonal is zero). The output for a position is a weighted mix of the value
# vectors. Real models stack many such layers, use many heads in parallel, and learn Q, K, V from data; the
# principle is what you see here. Note that comparing every position with every other makes the cost grow
# roughly with the *square* of the sequence length for this basic form, which is one reason long contexts
# are expensive.

# %% [markdown]
# ## 6. The context window and why chat gets more expensive every turn
#
# The **context window** is the maximum number of tokens (input plus output) the model can consider in one
# request. The model has **no memory between requests**: a chat application resends the whole conversation
# every turn. So the input you pay for grows with every turn, and the total over a conversation grows roughly
# with the *square* of the number of turns.

# %%
def conversation_input_tokens(user_tokens, assistant_tokens, system_tokens=200):
    """Input tokens billed on each turn if the full history is resent every time."""
    history, per_turn = system_tokens, []
    for u, a in zip(user_tokens, assistant_tokens):
        history += u
        per_turn.append(history)     # this request carries system prompt + everything so far + new user message
        history += a
    return per_turn


per_turn = conversation_input_tokens([50] * 10, [150] * 10)
print(per_turn)
print("total input tokens over 10 turns:", sum(per_turn), " vs a single turn of the last size:", per_turn[-1])

# %% [markdown]
# When the history no longer fits, the application has to drop or compress something (oldest messages,
# a summary, retrieved snippets instead of full documents). That is a design decision you will make in the
# RAG and agent notebooks. Silent truncation is the failure to avoid: the model then answers confidently
# without information the user believes it has.

# %%
def fit_to_window(messages, budget, count=est_tokens):
    """Drop the oldest non-system messages until the estimated total fits the budget."""
    msgs = list(messages)
    while sum(count(m["content"]) for m in msgs) > budget and len(msgs) > 2:
        del msgs[1]          # keep msgs[0] (the system prompt), drop the oldest after it
    return msgs


chat = [{"role": "system", "content": "You are a support assistant."}] + [
    {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " + "word " * 40} for i in range(12)
]
trimmed = fit_to_window(chat, budget=200)
print(len(chat), "->", len(trimmed), "messages; first kept after system:", trimmed[1]["content"][:10])

# %% [markdown]
# ## 7. Cost and latency thinking
#
# LLM APIs bill **per token, with output tokens costing several times more than input tokens**, and latency has
# two parts: time until the first token, then a roughly constant generation speed per output token. That gives a
# simple back-of-envelope model. **The prices and speeds below are illustrative placeholders that you must replace
# with the current numbers from your provider's pricing page**; the point is the shape of the calculation.

# %%
PRICE_PER_MTOK = {"small": (1.0, 5.0), "large": (5.0, 25.0)}   # ILLUSTRATIVE (input $, output $) per 1M tokens


def request_cost(model, in_tokens, out_tokens):
    pin, pout = PRICE_PER_MTOK[model]
    return (in_tokens * pin + out_tokens * pout) / 1_000_000


def request_latency(out_tokens, ttft_s=0.6, tokens_per_s=60):   # ILLUSTRATIVE speed assumptions
    return ttft_s + out_tokens / tokens_per_s


# a RAG answer: 3 retrieved chunks + question + system prompt in, a short answer out
rag_in = est_tokens("x" * 4 * (3 * 120 + 40 + 200))
rag_out = 150
for model in PRICE_PER_MTOK:
    c = request_cost(model, rag_in, rag_out)
    print(f"{model:>5}: ${c:.5f} per request = ${c * 1000:,.2f} per 1,000 requests = ${c * 1e6:,.0f} per million requests")
print(f"latency estimate for {rag_out} output tokens: {request_latency(rag_out):.1f}s")

# %% [markdown]
# Units are where cost estimates go wrong: per request, per thousand and per million requests differ by
# factors of a thousand. Always name the unit.
#
# The levers you have, in the order most teams reach for them: send fewer input tokens (retrieve less, trim
# the system prompt), cap the output length, cache a repeated prefix if your provider supports it, use a smaller
# model for easy requests, and batch non-urgent work. Each has a quality trade-off, which is why evaluation
# (module notebook 05) comes before optimisation.

# %% [markdown]
# ## 8. Calling a real model (no-run)
#
# The cell below is **not executed here**: it needs the `anthropic` package, network access and an API key in
# the `ANTHROPIC_API_KEY` environment variable. It shows the same ideas against a real model: counting tokens
# exactly, and one request with a system prompt. Never paste a key into a notebook.
#
# A detail worth knowing: newer Claude models control randomness themselves and do not accept `temperature`,
# `top_p` or `top_k` (the API returns an error if you send non-default values). On those models you steer
# behaviour with the prompt instead. Some older or smaller models still accept sampling parameters.

# %% tags=["no-run"]
import anthropic

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from the environment

count = client.messages.count_tokens(
    model="claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Premium members get free return shipping."}],
)
print("exact input tokens:", count.input_tokens)

response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=300,
    system="You are a concise support assistant.",
    messages=[{"role": "user", "content": "In one sentence, what is a context window?"}],
)
print(response.content[0].text)
print(response.usage.input_tokens, response.usage.output_tokens)   # what you are billed for

# %% [markdown]
# ## Common mistakes
#
# 1. **Treating the model as a database.** It generates plausible text; it does not look facts up. Hallucination
#    (a fluent, confident, wrong answer) is a property of the mechanism, not a bug that a version bump removes.
#    Grounding answers in retrieved documents (notebook 04) reduces it but does not eliminate it.
# 2. **Assuming determinism.** Sampling makes outputs vary between runs, and even "greedy" decoding can differ
#    across hardware or versions. Design tests around properties (JSON is valid, the cited source is correct),
#    not exact strings.
# 3. **Counting characters or words instead of tokens** when budgeting context and cost.
# 4. **Forgetting output tokens cost more** and that the whole history is resent each turn.
# 5. **Forgetting the window is shared.** Input plus output must fit; a huge prompt leaves less room to answer.

# %%
# Demonstration of mistake 3: the same character count can be very different token counts
plain = "returns policy " * 20
rare = "xqzvkjw" * 43
print(len(plain), len(rare))
print(len(bpe_encode(plain, merges)), len(bpe_encode(rare, merges)))

# %% [markdown]
# ## Interview corner
#
# **1. What does temperature do, and what happens at 0?**
# <details><summary>Answer</summary>
# It rescales the log-probabilities before sampling: below 1 sharpens the distribution, above 1 flattens it. At (or near) 0
# sampling becomes greedy: always the most likely token, which is repeatable but can loop or be dull. Even then, exact
# reproducibility across runs is not guaranteed in a hosted service.
# </details>
#
# **2. Why does the cost of a chat conversation grow faster than linearly with the number of turns?**
# <details><summary>Answer</summary>
# The model is stateless, so each request resends the system prompt and the full history. Turn <code>t</code> carries about
# <code>t</code> exchanges of input, so the total over <code>T</code> turns grows roughly with <code>T^2</code>. Mitigations: summarise or trim history,
# retrieve only relevant context, and cache a stable prefix if supported.
# </details>
#
# **3. What is a hallucination and can you prevent it?**
# <details><summary>Answer</summary>
# A fluent output that is not supported by the source or by reality. The model optimises for plausible continuations, not truth.
# You reduce it with grounding (retrieval with citations), instructions to say "I don't know", validation of outputs against sources
# or schemas, and evaluation on cases that tempt the model to invent. You do not fully prevent it, so high-stakes flows need checks or a human.
# </details>
#
# **4. What is a token, and why does it matter to an engineer?**
# <details><summary>Answer</summary>
# A vocabulary unit (word piece) produced by the tokenizer. Context limits, price and speed are all measured in tokens, and different
# tokenizers give different counts for the same text, so use the provider's counting tool for exact budgets.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Temperature softmax
# The model produces raw scores (**logits**) instead of probabilities. Write `temp_softmax(logits, T)`: divide the logits by `T`,
# then apply a numerically stable softmax (subtract the maximum first). It must return a probability vector that sums to 1.
# *Hint: `np.exp(z - z.max())`, then divide by the sum.*

# %% tags=["starter"]
def temp_softmax(logits, T):
    ...


# %%
lg = np.array([2.0, 1.0, 0.1, -1.0])
base = temp_softmax(lg, 1.0)
ref = np.exp(lg) / np.exp(lg).sum()
assert np.allclose(base, ref), "T=1 must equal the ordinary softmax"
assert abs(temp_softmax(lg, 0.5).sum() - 1) < 1e-9, "must sum to 1"
assert temp_softmax(lg, 0.3).max() > base.max() > temp_softmax(lg, 3.0).max(), "lower T must sharpen, higher T must flatten"
assert np.all(np.isfinite(temp_softmax(np.array([1000.0, 999.0]), 1.0))), "subtract the max: large logits must not overflow"
print("Exercise 1 passed")

# %% tags=["solution"]
def temp_softmax(logits, T):
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# %% [markdown]
# ### Exercise 2: Nucleus (top-p) sampling filter
# Write `top_p_filter(probs, p)`: keep the smallest set of highest-probability tokens whose cumulative probability
# reaches `p`, zero out the rest, and renormalise so the result sums to 1. The output has the same length as the input.
# *Hint: sort descending, `np.cumsum`, find the first index where the cumulative sum is >= p, keep everything up to it.*

# %% tags=["starter"]
def top_p_filter(probs, p):
    ...


# %%
q = np.array([0.05, 0.5, 0.15, 0.3])          # deliberately unsorted
r = top_p_filter(q, 0.75)
assert r.shape == q.shape and abs(r.sum() - 1) < 1e-9, "same shape, sums to 1"
assert list(np.nonzero(r)[0]) == [1, 3], f"p=0.75 needs the top 2 tokens (0.5 + 0.3), got indices {list(np.nonzero(r)[0])}"
assert np.allclose(r[[1, 3]], np.array([0.5, 0.3]) / 0.8), "kept tokens must be renormalised"
assert list(np.nonzero(top_p_filter(q, 0.9))[0]) == [1, 2, 3], "p=0.9 needs three tokens"
assert np.count_nonzero(top_p_filter(q, 1.0)) == 4, "p=1.0 keeps everything"
assert np.count_nonzero(top_p_filter(q, 0.01)) == 1, "a tiny p still keeps the single best token"
print("Exercise 2 passed")

# %% tags=["solution"]
def top_p_filter(probs, p):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs)
    cum = np.cumsum(probs[order])
    k = min(int(np.searchsorted(cum, p - 1e-9)) + 1, len(probs))
    out = np.zeros_like(probs)
    out[order[:k]] = probs[order[:k]]
    return out / out.sum()


# %% [markdown]
# ### Exercise 3: What will this workload cost?
# A support assistant handles **20,000 conversations a day**. Each conversation has **6 turns**; in each turn the user writes 60 tokens
# and the assistant answers with 140 tokens; the system prompt is 300 tokens and the full history is resent every turn.
# Using `conversation_input_tokens` and `request_cost` from this notebook with the `"small"` price row, store the **total input tokens per
# conversation** in `in_per_conv`, the **output tokens per conversation** in `out_per_conv`, and the **cost per day in dollars** in `daily_cost`.
# *Hint: the output tokens are billed once per turn; input tokens follow the history growth. Remember `request_cost` takes token counts, not millions.*

# %% tags=["starter"]
in_per_conv = ...
out_per_conv = ...
daily_cost = ...

# %%
hist = 300
exp_in = 0
for _ in range(6):
    hist += 60
    exp_in += hist
    hist += 140
exp_out = 6 * 140
exp_cost = 20000 * (exp_in * PRICE_PER_MTOK["small"][0] + exp_out * PRICE_PER_MTOK["small"][1]) / 1e6
assert in_per_conv == exp_in, f"input tokens per conversation: expected {exp_in}, got {in_per_conv}"
assert out_per_conv == exp_out, f"output tokens per conversation: expected {exp_out}, got {out_per_conv}"
assert abs(daily_cost - exp_cost) < 1e-6, f"daily cost: expected {exp_cost:.2f}, got {daily_cost}"
print(f"Exercise 3 passed (input {in_per_conv} + output {out_per_conv} tokens per conversation, ${daily_cost:,.2f}/day at the illustrative prices)")

# %% tags=["solution"]
in_per_conv = sum(conversation_input_tokens([60] * 6, [140] * 6, system_tokens=300))
out_per_conv = 6 * 140
daily_cost = 20000 * request_cost("small", in_per_conv, out_per_conv)


# %% [markdown]
# ### Exercise 4: Which context length generalises best?
# Using the `results` dictionary computed above (order -> (train perplexity, held-out perplexity)), store in `ppl_gap` a dict mapping each order
# to `held_out - train` perplexity. Then store in `overfit_order` the order at which the gap is **largest**.
# *Hint: a dict comprehension over `results.items()`, then `max(..., key=...)`.*

# %% tags=["starter"]
ppl_gap = ...
overfit_order = ...

# %%
assert set(ppl_gap) == set(results), "one gap per order"
for o, (tr, te) in results.items():
    assert abs(ppl_gap[o] - (te - tr)) < 1e-9, f"gap for order {o} is held_out - train"
assert overfit_order == max(results, key=lambda o: results[o][1] - results[o][0]), "overfit_order should be the order with the biggest gap"
print(f"Exercise 4 passed (largest generalisation gap at order {overfit_order})")

# %% tags=["solution"]
ppl_gap = {o: te - tr for o, (tr, te) in results.items()}
overfit_order = max(ppl_gap, key=ppl_gap.get)


# %% [markdown]
# ### Exercise 5 (open-ended): Break your tiny model
# Train `CharNGram(order=3)` and `CharNGram(order=6)` on `corpus`, generate 200 characters from each at `T=0.7` with a prompt of your choice,
# and write 3-4 sentences in a markdown cell: which is more fluent, which is more repetitive or copied, and how does that connect to the
# `verbatim_rate` measurement and the perplexity table? Then try a prompt the corpus never contains (for example `"Quantum "`) and note what happens.

# %% [markdown]
# ## Recap
# - An LLM is a next-token probability function; generation repeats *distribution -> pick a token -> append*.
# - Text is split into tokens (BPE merges frequent pairs); limits, price and speed are all per token, and counts are tokenizer-specific.
# - Temperature, top-k and top-p are a sampling policy applied *outside* the model; low temperature is repeatable but can copy or loop.
# - More context helps a model up to a point, then it memorises; held-out data tells the truth.
# - Attention lets each position read earlier positions; scale changes the machinery, not the interface.
# - The context window is per request; chat resends history, so cost grows super-linearly and something must be trimmed or retrieved.
# - Cost and latency estimates are simple arithmetic: name your units and replace placeholder prices with current ones.
#
# **Go deeper:** the Anthropic API documentation (docs.anthropic.com), sections on models, tokens and the Messages API;
# the original transformer paper "Attention Is All You Need" (Vaswani et al., 2017) for the attention mechanism.
#
# **Next:** [Prompting and structured outputs](02_prompting_and_structured_outputs.ipynb)
