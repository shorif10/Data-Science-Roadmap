# %% [markdown]
# # Attention and Transformers from Scratch (NumPy)
#
# **Why this matters.** Every large language model, every modern translation, search-ranking and code-completion system is built from one mechanism: **attention**. If you can write scaled dot-product attention,
# multi-head attention, a causal mask and positional encodings in NumPy and check them against a reference implementation, you can read any transformer paper or codebase without treating it as a black box,
# and you will know why context length is expensive and why generation uses a causal mask.
#
# **You will learn**
# - explain attention as a soft lookup with queries, keys and values, and implement it with a numerically stable softmax
# - verify the properties that make it work: rows of weights sum to 1, the `1/sqrt(d)` scaling matters, a causal mask blocks the future
# - build multi-head attention and positional encodings, and check your code against PyTorch's `nn.MultiheadAttention`
# - assemble a tiny transformer block (attention + feed-forward + residual + layer norm) and run a forward pass
#
# **Prerequisites:** [Word embeddings from scratch](03_word_embeddings_from_scratch.ipynb), [Sequence models for text](04_sequence_models_for_text.ipynb), matrix multiplication ([NumPy](../04_NumPy/)).
#
# > Data used: none. The weights below are **random and untrained**, so the attention patterns have no linguistic meaning. The goal is to get the mechanics exactly right; training them is the job of the next notebook's pretrained models.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

rng = np.random.default_rng(42)
np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Attention as a soft dictionary lookup
# A Python dictionary maps an exact key to a value. **Attention** does a fuzzy lookup: a **query** vector is compared with every **key** vector (by dot product, a similarity score), the scores are turned into
# weights that sum to 1 (softmax), and the answer is the weighted average of the **value** vectors. Keys that match the query strongly contribute most.
#
# Tiny example with hand-made 2-D vectors. Two of the three keys point roughly where the query points:

# %%
query = np.array([1.0, 0.0])
keys = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 0.2]])
values = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
scores = keys @ query
print("scores (dot products):", scores)

# %% [markdown]
# A naive softmax, `exp(x) / sum(exp(x))`, overflows for large scores. The standard fix subtracts the maximum first, which does not change the result (the factor cancels) but keeps the exponents <= 0:

# %%
def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print("naive softmax of [1000, 1001, 1002]:", naive)
print("stable softmax:                     ", softmax(big))

w = softmax(scores)
print("\nattention weights:", w, "| sum =", w.sum())
print("output = weighted average of values:", w @ values)

# %% [markdown]
# *What just happened?* The query most resembles the first key, so the output is pulled towards the first value. The output is a **blend** of values: nothing is selected outright. That is what "soft" lookup means,
# and why the whole thing is differentiable and trainable.

# %% [markdown]
# ## 2. Scaled dot-product attention
# For many queries at once, stack them in a matrix `Q` (n_queries x d), the keys in `K` and values in `V`:
#
# `Attention(Q, K, V) = softmax( Q K^T / sqrt(d) ) V`
#
# Row `i` of the weight matrix says how much query `i` looks at each key. The division by `sqrt(d)` is not decoration; test it below.

# %%
def attention(Q, K, V, mask=None):
    """mask: boolean array broadcastable to (n_q, n_k); True = may attend, False = blocked."""
    d = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # blocked positions get ~zero weight after softmax
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


Q = rng.normal(size=(4, 8))
K = rng.normal(size=(5, 8))
V = rng.normal(size=(5, 3))
out, weights = attention(Q, K, V)
print("output shape:", out.shape, "| weights shape:", weights.shape)
print("row sums of the weights:", weights.sum(axis=1))
print("all rows sum to 1:", bool(np.allclose(weights.sum(axis=1), 1.0)))

# %% [markdown]
# ### Why divide by sqrt(d)?
# If the components of a query and a key are independent with variance 1, their dot product has variance `d`: it grows with the dimension. Large scores push softmax towards one-hot, where gradients vanish.
# Measure it instead of trusting the claim:

# %%
for d in [4, 64, 1024]:
    q = rng.normal(size=(2000, d))
    k = rng.normal(size=(2000, d))
    dots = (q * k).sum(axis=1)
    keys8 = rng.normal(size=(8, d))
    q1 = rng.normal(size=d)
    unscaled = softmax(keys8 @ q1).max()
    scaled = softmax(keys8 @ q1 / np.sqrt(d)).max()
    print(f"d={d:5d}: std of q.k = {dots.std():6.2f} (sqrt(d) = {np.sqrt(d):5.2f}) | largest of 8 weights: unscaled {unscaled:.2f}, scaled {scaled:.2f}")

# %% [markdown]
# The spread of the dot products tracks `sqrt(d)`. Without scaling, at large `d` one key takes almost all the weight (largest weight close to 1.0) and the model barely learns where to look; with scaling the weights stay spread out
# and the gradients healthy.

# %% [markdown]
# ## 3. Self-attention with learned projections
# In **self-attention** the queries, keys and values all come from the same sequence `X` (one row per token), each through its own learned matrix: `Q = X Wq`, `K = X Wk`, `V = X Wv`. The projections let the model ask
# different questions ("what am I looking for?") than the ones it answers with ("what do I offer?"). Here are six tokens with random embeddings and random projections:

# %%
tokens = ["the", "cat", "sat", "on", "the", "mat"]
d_model = 8
X = rng.normal(size=(len(tokens), d_model))
Wq, Wk, Wv = (rng.normal(size=(d_model, d_model)) / np.sqrt(d_model) for _ in range(3))
out, weights = attention(X @ Wq, X @ Wk, X @ Wv)

fig, ax = plt.subplots(figsize=(4.8, 4))
im = ax.imshow(weights, cmap="Blues", vmin=0)
ax.set_xticks(range(6), tokens)
ax.set_yticks(range(6), tokens)
ax.set_xlabel("key (attended to)")
ax.set_ylabel("query (attending)")
ax.set_title("Self-attention weights (random, untrained)")
fig.colorbar(im, ax=ax, label="weight")
plt.tight_layout()
plt.show()

# %% [markdown]
# Every token attends to every token, including itself. The pattern above means nothing, since the weights are random; in a trained model they become meaningful (for example a pronoun attending to its noun).

# %% [markdown]
# ## 4. Masks: padding and the causal mask
# Two reasons to block attention:
# - **Padding mask**: batches are padded to equal length; padded positions must get zero weight.
# - **Causal mask**: a language model predicts the next token, so position `i` may only look at positions `<= i`. Otherwise it would simply copy the answer from the future.
#
# The causal mask is a lower-triangular matrix of "allowed" positions:

# %%
n = len(tokens)
causal = np.tril(np.ones((n, n), dtype=bool))
out_c, w_c = attention(X @ Wq, X @ Wk, X @ Wv, mask=causal)
print("weights above the diagonal are zero:", bool(np.allclose(np.triu(w_c, k=1), 0)))
print("rows still sum to 1:", bool(np.allclose(w_c.sum(axis=1), 1)))
print(w_c[:3, :3])

# %% [markdown]
# Do not just believe the mask works: **perturb the future** and check that the past does not notice. Change the *last* token's embedding and compare outputs at the earlier positions, with and without the mask:

# %%
X2 = X.copy()
X2[-1] += 5.0  # change only the last token
same_causal = np.allclose(attention(X @ Wq, X @ Wk, X @ Wv, causal)[0][:-1], attention(X2 @ Wq, X2 @ Wk, X2 @ Wv, causal)[0][:-1])
same_full = np.allclose(attention(X @ Wq, X @ Wk, X @ Wv)[0][:-1], attention(X2 @ Wq, X2 @ Wk, X2 @ Wv)[0][:-1])
print("earlier outputs unchanged with the causal mask:   ", same_causal)
print("earlier outputs unchanged WITHOUT a mask (leaks): ", same_full)

# %% [markdown]
# ## 5. Multi-head attention
# One attention pattern has to serve every purpose. **Multi-head attention** runs `h` attentions in parallel, each in a smaller `d_model / h` subspace with its own projections, then concatenates the results and mixes them with
# an output matrix `Wo`. Different heads can specialise (in trained models, one might track the previous word, another a matching bracket). Implementation is mostly reshaping:

# %%
def split_heads(M, h):  # (n, d) -> (h, n, d/h)
    n, d = M.shape
    return M.reshape(n, h, d // h).transpose(1, 0, 2)


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    Qh, Kh, Vh = (split_heads(X @ W, n_heads) for W in (Wq, Wk, Wv))
    out, weights = attention(Qh, Kh, Vh, mask=mask)  # batched over heads: (h, n, n) weights
    concat = out.transpose(1, 0, 2).reshape(X.shape[0], -1)  # (h, n, d/h) -> (n, d)
    return concat @ Wo, weights


Wo = rng.normal(size=(d_model, d_model)) / np.sqrt(d_model)
mha_out, mha_w = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=2)
print("output:", mha_out.shape, "| per-head weights:", mha_w.shape)
print("each head's rows sum to 1:", bool(np.allclose(mha_w.sum(axis=-1), 1)))
single, _ = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=1)
print("with one head it equals plain attention followed by Wo:", bool(np.allclose(single, attention(X @ Wq, X @ Wk, X @ Wv)[0] @ Wo)))

# %% [markdown]
# ### Check against a reference implementation
# A from-scratch implementation is only trustworthy once it matches a trusted one. PyTorch's `nn.MultiheadAttention` stores the three projections stacked in `in_proj_weight` and applies them as `x @ W.T`.
# Copy our random weights into it and compare outputs and per-head weights:

# %%
mha = torch.nn.MultiheadAttention(d_model, num_heads=2, bias=False, batch_first=True)
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.tensor(np.concatenate([Wq.T, Wk.T, Wv.T], axis=0), dtype=torch.float32))
    mha.out_proj.weight.copy_(torch.tensor(Wo.T, dtype=torch.float32))
    x_t = torch.tensor(X, dtype=torch.float32)[None]
    ref_out, ref_w = mha(x_t, x_t, x_t, need_weights=True, average_attn_weights=False)
print("outputs match PyTorch:         ", bool(np.allclose(mha_out, ref_out[0].numpy(), atol=1e-5)))
print("per-head weights match PyTorch:", bool(np.allclose(mha_w, ref_w[0].numpy(), atol=1e-5)))

# %% [markdown]
# ## 6. Positional encoding: attention alone is blind to order
# Self-attention treats its input as a **set**: shuffle the tokens and you get the same outputs, shuffled the same way (it is *permutation equivariant*). Verify it:

# %%
perm = rng.permutation(n)
base, _ = multi_head_attention(X, Wq, Wk, Wv, Wo, 2)
shuffled, _ = multi_head_attention(X[perm], Wq, Wk, Wv, Wo, 2)
print("output of shuffled input == shuffled output:", bool(np.allclose(shuffled, base[perm])))

# %% [markdown]
# So "dog bites man" and "man bites dog" would look identical. The fix is to **add position information** to the embeddings. The original transformer used fixed sinusoids of different wavelengths:
#
# `PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(pos / 10000^(2i/d))`
#
# Each position gets a unique pattern; low dimensions oscillate quickly, high dimensions slowly (like the digits of a counter). Modern models often learn the positions or use rotary embeddings, but the purpose is the same.

# %%
def positional_encoding(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / np.power(10000.0, i / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


PE = positional_encoding(50, 16)
fig, ax = plt.subplots(figsize=(6, 3.6))
im = ax.imshow(PE, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
ax.set_xlabel("embedding dimension")
ax.set_ylabel("position in the sequence")
ax.set_title("Sinusoidal positional encoding (50 positions, 16 dims)")
fig.colorbar(im, ax=ax, label="value")
plt.tight_layout()
plt.show()

# %%
Xp = X + positional_encoding(n, d_model)
with_pos, _ = multi_head_attention(Xp, Wq, Wk, Wv, Wo, 2)
shuffled_tokens_with_pos, _ = multi_head_attention(X[perm] + positional_encoding(n, d_model), Wq, Wk, Wv, Wo, 2)
print("with positions added, shuffling tokens no longer just shuffles the output:", not np.allclose(shuffled_tokens_with_pos, with_pos[perm]))

# %% [markdown]
# ## 7. A tiny transformer block
# A transformer block wraps attention with three more ingredients:
# 1. **Feed-forward network** applied to each token independently: `relu(x W1 + b1) W2 + b2`, widening to `d_ff` (typically 4x) and back.
# 2. **Residual connections**: `x + sublayer(x)`, so information and gradients can skip the layer.
# 3. **Layer normalisation**: rescale each token vector to zero mean and unit variance (plus learned scale/shift), which stabilises training.
#
# We use the "pre-norm" arrangement (normalise before each sub-layer), common in current models. Real models use GELU-type activations and stack dozens of blocks; the structure is the same.

# %%
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def make_block(d_model, d_ff, rng):
    s = lambda *shape: rng.normal(size=shape) / np.sqrt(shape[0])
    return {"Wq": s(d_model, d_model), "Wk": s(d_model, d_model), "Wv": s(d_model, d_model), "Wo": s(d_model, d_model),
            "W1": s(d_model, d_ff), "b1": np.zeros(d_ff), "W2": s(d_ff, d_model), "b2": np.zeros(d_model),
            "g1": np.ones(d_model), "be1": np.zeros(d_model), "g2": np.ones(d_model), "be2": np.zeros(d_model)}


def transformer_block(x, p, n_heads, mask=None):
    h = layer_norm(x, p["g1"], p["be1"])
    attn_out, weights = multi_head_attention(h, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask)
    x = x + attn_out  # residual 1
    h = layer_norm(x, p["g2"], p["be2"])
    x = x + np.maximum(h @ p["W1"] + p["b1"], 0) @ p["W2"] + p["b2"]  # residual 2 around the feed-forward net
    return x, weights


block = make_block(d_model, 4 * d_model, rng)
y, w_blk = transformer_block(Xp, block, n_heads=2, mask=causal)
n_params = sum(v.size for v in block.values())
print("input", Xp.shape, "-> output", y.shape, "(same shape: blocks can be stacked)")
print("parameters in this block:", n_params)
print("layer_norm gives rows with mean 0 and variance 1:", bool(np.allclose(layer_norm(Xp, 1, 0).mean(axis=1), 0, atol=1e-9) and np.allclose(layer_norm(Xp, 1, 0).var(axis=1), 1, atol=1e-3)))

# %% [markdown]
# *What just happened?* A sequence of 6 vectors went in and 6 came out with the same shape, so the block can be stacked 12 or 96 times. A full language model adds an embedding table at the bottom and a linear layer + softmax
# over the vocabulary at the top, and is trained to predict the next token under the causal mask. Parameter count check: four `d x d` attention matrices plus two feed-forward matrices (`d x 4d` and `4d x d`) plus biases and norm parameters.

# %% [markdown]
# ### The cost of attention
# The weight matrix is `n x n`. Doubling the sequence length quadruples the memory and compute for the attention scores. That is the root of the "context length" problem and the reason for long-context tricks
# (sparse or sliding-window attention, FlashAttention-style kernels that avoid storing the full matrix):

# %%
for length in [512, 2048, 8192]:
    cells = length * length
    print(f"sequence length {length:5d}: {cells:>11,} attention scores per head per layer ({cells * 4 / 1e6:8.1f} MB in float32)")

# %% [markdown]
# ## Common mistakes
#
# 1. **Softmax over the wrong axis.** The weights for each query must sum to 1 over the *keys* (last axis). Check `weights.sum(axis=-1)`.
# 2. **Unstable softmax** without subtracting the maximum: `inf`/`nan` for large scores.
# 3. **Forgetting the `1/sqrt(d)` scaling**, then wondering why training is unstable or the attention is one-hot.
# 4. **Mask polarity bugs.** Decide whether `True` means "allowed" or "blocked" and stick with it; test with the perturbation check from section 4.
# 5. **Adding `-inf` masks to a row where everything is masked**, which gives `nan`. Keep a large negative number or ensure each row has at least one allowed key.
# 6. **Reshaping heads incorrectly**: `reshape` then `transpose` (as in `split_heads`) keeps each head's feature block contiguous; transposing before reshaping mixes heads. Verify against a reference.
# 7. **Forgetting positional information**, so the model is order-blind (permutation test in section 6).

# %%
# Demonstration of mistake 1: softmax over the wrong axis
scores = rng.normal(size=(3, 4))
print("axis=-1 (correct) row sums:", softmax(scores, axis=-1).sum(axis=1))
print("axis=0  (wrong)   row sums:", softmax(scores, axis=0).sum(axis=1))

# %% [markdown]
# ## Interview corner
#
# **1. Explain self-attention in two sentences, and write its formula.**
# <details><summary>Answer</summary>
# Each token produces a query, key and value; its new representation is the average of all tokens' values, weighted by how well its query matches their keys. Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.
# </details>
#
# **2. Why scale by sqrt(d_k)?**
# <details><summary>Answer</summary>
# Dot products of d_k-dimensional random vectors have variance proportional to d_k, so scores grow with dimension, softmax saturates towards one-hot and its gradients vanish. Dividing by sqrt(d_k) keeps the scores at unit scale.
# </details>
#
# **3. Why multiple heads? And what is the cost of attention in sequence length?**
# <details><summary>Answer</summary>
# Heads let the model attend to different relationships at once in different subspaces, at the same total cost as one full-width head. Attention compares every token with every token, so time and memory for the score matrix are O(n^2)
# in the sequence length n.
# </details>
#
# **4. What is the difference between an encoder block and a decoder (GPT-style) block, and why the causal mask?**
# <details><summary>Answer</summary>
# An encoder block (BERT-style) attends in both directions and is used to build representations for classification or retrieval. A decoder block uses a causal mask so each position sees only earlier tokens, which is what allows training on
# next-token prediction and generating text one token at a time. (Encoder-decoder models add cross-attention from decoder to encoder outputs.)
# </details>

# %% [markdown]
# ## Exercises
# Use the names given (prefixed `my_`) so your work does not overwrite the notebook's own functions.

# %% [markdown]
# ### Exercise 1: A stable softmax
# Write `my_softmax(x, axis=-1)` that subtracts the maximum along `axis` before exponentiating. The check verifies that rows sum to 1, that it survives huge inputs, and that it is shift-invariant.

# %% tags=["starter"]
def my_softmax(x, axis=-1):
    ...


# %%
_t = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
_s = my_softmax(_t)
assert np.all(np.isfinite(_s)), "huge inputs must not overflow: subtract the row maximum first"
assert np.allclose(_s.sum(axis=-1), 1.0), "each row must sum to 1"
assert np.allclose(_s[0], _s[1]), "softmax is invariant to adding a constant to every score"
assert np.allclose(my_softmax(_t, axis=0).sum(axis=0), 1.0), "axis argument must be respected"
print("Exercise 1 passed")

# %% tags=["solution"]
def my_softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# %% [markdown]
# ### Exercise 2: Attention, checked against a loop
# Write `my_attention(Q, K, V)` returning `(output, weights)` using matrix operations (no loops), with the `1/sqrt(d)` scaling. The check compares with a slow per-query loop that does the same thing one query at a time.

# %% tags=["starter"]
def my_attention(Q, K, V):
    ...


# %%
_r = np.random.default_rng(0)
_Q, _K, _V = _r.normal(size=(5, 6)), _r.normal(size=(7, 6)), _r.normal(size=(7, 4))
_o, _w = my_attention(_Q, _K, _V)
_loop_o = []
for _q in _Q:
    _sc = np.array([_q @ _k for _k in _K]) / np.sqrt(6)
    _e = np.exp(_sc - _sc.max())
    _wt = _e / _e.sum()
    _loop_o.append(sum(_wt[j] * _V[j] for j in range(7)))
assert _w.shape == (5, 7) and _o.shape == (5, 4), "weights are (n_queries, n_keys); output is (n_queries, d_v)"
assert np.allclose(_w.sum(axis=1), 1.0), "weights for each query must sum to 1"
assert np.allclose(_o, np.array(_loop_o)), "output differs from the loop reference"
print("Exercise 2 passed")

# %% tags=["solution"]
def my_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    z = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(z)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V, w


# %% [markdown]
# ### Exercise 3: Causal attention that cannot peek
# Write `my_causal_attention(Q, K, V)` (same as attention but position `i` may only use keys `<= i`). Then store in `future_leak` the **largest absolute change** in the output at position 0 when you add 10 to the **last row** of `K` and `V`
# (use the `_Q, _K, _V` arrays from the previous check cell). A correct causal implementation gives exactly zero.

# %% tags=["starter"]
def my_causal_attention(Q, K, V):
    ...


future_leak = ...

# %%
_r = np.random.default_rng(1)
_Q, _K, _V = _r.normal(size=(6, 4)), _r.normal(size=(6, 4)), _r.normal(size=(6, 3))
_o, _w = my_causal_attention(_Q, _K, _V)
assert np.allclose(np.triu(_w, k=1), 0), "weights above the diagonal must be zero"
assert np.allclose(_w.sum(axis=1), 1.0), "rows must still sum to 1"
assert np.allclose(_w[0], [1, 0, 0, 0, 0, 0]), "the first token can only attend to itself"
_K2, _V2 = _K.copy(), _V.copy()
_K2[-1] += 10
_V2[-1] += 10
assert np.allclose(_o[:-1], my_causal_attention(_Q, _K2, _V2)[0][:-1]), "changing the last token must not change earlier outputs"
assert future_leak == 0.0, f"expected exactly 0.0 leakage from the future, got {future_leak}"
print("Exercise 3 passed")

# %% tags=["solution"]
def my_causal_attention(Q, K, V):
    n_q, n_k = Q.shape[0], K.shape[0]
    mask = np.tril(np.ones((n_q, n_k), dtype=bool))
    scores = np.where(mask, Q @ K.T / np.sqrt(Q.shape[-1]), -1e9)
    z = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(z)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V, w


_Q3, _K3, _V3 = np.random.default_rng(0).normal(size=(7, 6)), np.random.default_rng(1).normal(size=(7, 6)), np.random.default_rng(2).normal(size=(7, 4))
_a = my_causal_attention(_Q3, _K3, _V3)[0]
_K4, _V4 = _K3.copy(), _V3.copy()
_K4[-1] += 10
_V4[-1] += 10
future_leak = float(np.abs(_a[0] - my_causal_attention(_Q3, _K4, _V4)[0][0]).max())

# %% [markdown]
# ### Exercise 4: Positional encoding
# Write `my_positional_encoding(n_pos, d)` for even `d`, following the sinusoidal formula. Then store in `pe_row0` the row for position 0 (it should alternate 0 and 1) and in `pe_distinct` whether all 100 positions of
# `my_positional_encoding(100, 32)` have **distinct** vectors (`True`/`False`).

# %% tags=["starter"]
def my_positional_encoding(n_pos, d):
    ...


pe_row0 = ...
pe_distinct = ...

# %%
_pe = my_positional_encoding(10, 8)
assert _pe.shape == (10, 8)
for _pos, _i in [(3, 0), (3, 1), (7, 2), (7, 3), (9, 6), (9, 7)]:
    _k = _i // 2
    _ang = _pos / 10000 ** (2 * _k / 8)
    assert abs(_pe[_pos, _i] - (np.sin(_ang) if _i % 2 == 0 else np.cos(_ang))) < 1e-9, f"wrong value at position {_pos}, dimension {_i}"
assert np.allclose(pe_row0, [0, 1] * 4), "position 0: sin(0)=0 and cos(0)=1 alternating"
_big = my_positional_encoding(100, 32)
assert pe_distinct == (len({tuple(np.round(r, 9)) for r in _big}) == 100), "distinct means no two positions share the same vector"
print("Exercise 4 passed (all positions distinct:", pe_distinct, ")")

# %% tags=["solution"]
def my_positional_encoding(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    k = np.arange(d // 2)[None, :]
    angle = pos / np.power(10000.0, 2 * k / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


pe_row0 = my_positional_encoding(1, 8)[0]
pe_distinct = len({tuple(np.round(r, 9)) for r in my_positional_encoding(100, 32)}) == 100

# %% [markdown]
# ### Exercise 5 (open-ended): Do the arithmetic on long context
# A model has 32 layers and 32 heads. Using the "cost of attention" cell as a template, estimate the memory needed to store all attention score matrices for one sequence of 32,000 tokens in float32 if nothing is recomputed
# or fused, and compare it with the memory of a typical GPU you know about. Then explain in three sentences why real systems do not materialise that matrix, and what trade-off they make instead.

# %% [markdown]
# ## Recap
# - Attention is a soft lookup: `softmax(QK^T / sqrt(d)) V`. Verify weights sum to 1, use a max-subtracted softmax, and keep the scaling (measured above: dot-product spread grows like `sqrt(d)`).
# - A causal mask makes position `i` ignore later tokens; prove it with the perturbation test rather than trusting the code.
# - Multi-head attention splits `d_model` into heads, attends in parallel and recombines with `Wo`; our implementation matches PyTorch's `nn.MultiheadAttention`.
# - Attention is order-blind, so positions are added (sinusoids here; learned or rotary in many modern models).
# - A transformer block is attention + feed-forward, each wrapped in a residual connection and layer norm; cost of attention is O(n^2) in sequence length.
#
# **Go deeper:** Vaswani et al. (2017), "Attention Is All You Need"; Jay Alammar's "The Illustrated Transformer"; the PyTorch documentation for `nn.MultiheadAttention` and `nn.TransformerEncoderLayer`; Andrej Karpathy's "Let's build GPT" video lecture.
#
# **Next:** [Hugging Face and pretrained models](06_hugging_face_and_pretrained_models.ipynb)
