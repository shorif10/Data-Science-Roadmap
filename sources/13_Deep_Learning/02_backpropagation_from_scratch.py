# %% [markdown]
# # Backpropagation from Scratch
#
# **Why this matters.** `loss.backward()` in PyTorch and `model.fit()` in Keras hide the one algorithm that makes deep
# learning possible. When a model produces `nan` gradients, trains suspiciously slowly, or you write a custom layer, you are
# debugging backpropagation whether you know it or not. Deriving it once by hand, and *verifying it with numerical
# gradient checking*, is the best insurance a practitioner can buy. Gradient checking is also a real technique
# used to test custom layers and loss functions.
#
# **You will learn**
# - apply the chain rule to a tiny computation graph and check it numerically
# - derive and implement the gradients of a two-layer network with a softmax + cross-entropy output
# - verify any gradient with a centred finite-difference check (and see it catch a planted bug)
# - train the NumPy network on the digits dataset with mini-batch gradient descent
#
# **Prerequisites:** [01 Neurons, layers and the forward pass](01_neurons_layers_and_the_forward_pass.ipynb) and
# [Calculus: derivatives and gradients](../02_Math_for_Data_Science/03_calculus_derivatives_and_gradients.ipynb)
# (chain rule, partial derivatives). The maths of gradient descent is in
# [Optimisation](../02_Math_for_Data_Science/04_gradient_descent_from_scratch.ipynb).
#
# > Data used: scikit-learn's bundled `digits` (1,797 real 8x8 handwritten digit images).

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=4, suppress=True)

# %% [markdown]
# ## 1. The chain rule on a tiny graph
#
# Take one neuron with one input: `z = w * x + b`, `a = sigmoid(z)`, `loss = (a - t)^2`. Training asks: *how does the loss change if I nudge `w`?*
# The loss depends on `w` only through a chain of steps, so multiply the local slopes:
#
# `d loss / d w = (d loss / d a) * (d a / d z) * (d z / d w) = 2(a - t) * a(1 - a) * x`
#
# **Backpropagation is just this, applied from the loss backwards, reusing the intermediate results.**

# %%
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


x, t = 1.5, 1.0
w, b = 0.4, -0.2

# forward pass (keep the intermediates: backward needs them)
z = w * x + b
a = sigmoid(z)
loss = (a - t) ** 2

# backward pass: multiply local derivatives from the loss back to w
dloss_da = 2 * (a - t)
da_dz = a * (1 - a)
dz_dw = x
dloss_dw = dloss_da * da_dz * dz_dw
print(f"analytic d loss/d w = {dloss_dw:.6f}")

# %% [markdown]
# How do we know that is right? **Numerical gradient checking**: nudge `w` up and down by a tiny `eps` and measure the slope,
# `(L(w + eps) - L(w - eps)) / (2 eps)`. The centred version is far more accurate than the one-sided one.

# %%
def loss_of_w(w_):
    return (sigmoid(w_ * x + b) - t) ** 2


eps = 1e-6
numeric = (loss_of_w(w + eps) - loss_of_w(w - eps)) / (2 * eps)
print(f"numeric  d loss/d w = {numeric:.6f}")
print("relative error:", abs(numeric - dloss_dw) / max(abs(numeric), abs(dloss_dw)))

# %% [markdown]
# *What just happened?* Two independent routes to the same slope agreed to about 9-10 significant digits. That is the pattern for the whole notebook:
# derive the gradient analytically (fast, exact), and confirm it against the slow-but-simple numerical estimate.

# %% [markdown]
# ## 2. Deriving the gradients of a 2-layer network
#
# Network: `Z1 = X W1 + b1`, `A1 = f(Z1)` (ReLU or tanh), `Z2 = A1 W2 + b2`, `P = softmax(Z2)`, `L = mean(-log P[i, y_i])`.
# `n` is the batch size. Going backwards:
#
# | Step | Gradient | Why |
# |---|---|---|
# | output | `dZ2 = (P - Y) / n` | the beautiful softmax + cross-entropy result: predicted probabilities minus one-hot labels (`Y`) |
# | layer 2 weights | `dW2 = A1.T @ dZ2` | each weight's gradient is (its input) x (error at its output), summed over the batch |
# | layer 2 bias | `db2 = dZ2.sum(axis=0)` | the bias sees the error directly |
# | hidden activations | `dA1 = dZ2 @ W2.T` | send the error back through the weights |
# | hidden pre-activations | `dZ1 = dA1 * f'(Z1)` | multiply by the local slope (ReLU: 1 if `Z1 > 0` else 0) |
# | layer 1 | `dW1 = X.T @ dZ1`, `db1 = dZ1.sum(axis=0)` | same as layer 2 |
#
# Every line has the same shape as its forward counterpart (a very useful sanity check: `dW1` must be shaped like `W1`).
# For deeper networks the pattern simply repeats: `dZ -> dW, db, dA_previous`.

# %%
def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """Fully-connected network: [input] -> hidden layers (ReLU or tanh) -> softmax."""

    def __init__(self, sizes, activation="relu", seed=42):
        rng = np.random.default_rng(seed)
        self.act = activation
        # He-style init: std = sqrt(2 / fan_in)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def _f(self, z):
        return relu(z) if self.act == "relu" else np.tanh(z)

    def _df(self, z):
        return (z > 0).astype(float) if self.act == "relu" else 1.0 - np.tanh(z) ** 2

    def forward(self, X):
        """Return probabilities and a cache of (input, pre-activation) per layer."""
        cache, a = [], X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = a @ W + b
            cache.append((a, z))
            a = self._f(z)
        z = a @ self.W[-1] + self.b[-1]
        cache.append((a, z))
        return softmax(z), cache

    def loss(self, X, y):
        p, _ = self.forward(X)
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))

    def backward(self, X, y):
        """Return the loss and the gradients (lists dW, db matching self.W, self.b)."""
        n = len(y)
        p, cache = self.forward(X)
        loss = float(-np.mean(np.log(p[np.arange(n), y] + 1e-12)))
        dz = p.copy()
        dz[np.arange(n), y] -= 1.0           # P - Y (one-hot subtracted in place)
        dz /= n                                # ... divided by the batch size
        dWs, dbs = [], []
        for layer in reversed(range(len(self.W))):
            a_in, z = cache[layer]
            if layer < len(self.W) - 1:        # hidden layers multiply by the activation slope
                dz = dz * self._df(z)
            dWs.append(a_in.T @ dz)
            dbs.append(dz.sum(axis=0))
            dz = dz @ self.W[layer].T          # pass the error to the previous layer's activations
        return loss, dWs[::-1], dbs[::-1]

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)


net = MLP([64, 32, 10])
print("W shapes:", [W.shape for W in net.W])

# %% [markdown]
# ## 3. Gradient checking the whole network
#
# We perturb individual parameters one at a time, recompute the loss, and compare the finite-difference slope with the analytic gradient.
# Checking *every* parameter is slow (thousands of loss evaluations), so we check a random sample. Use `float64`, a small network
# and a tiny batch, and compare with a **relative error**:
# `|g_num - g_ana| / max(|g_num|, |g_ana|, tiny)`. Rules of thumb: below 1e-6 is great, above 1e-3 means a bug.

# %%
digits = load_digits()
X_all = digits.data / 16.0
y_all = digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25, stratify=y_all, random_state=42)
print("train:", X_tr.shape, " test:", X_te.shape)


def gradient_check(model, X, y, grads_fn=None, n_samples=25, eps=1e-6, seed=0):
    """Return the worst relative error over a random sample of parameters."""
    rng = np.random.default_rng(seed)
    _, dWs, dbs = model.backward(X, y) if grads_fn is None else grads_fn(model, X, y)
    worst = 0.0
    for params, grads in [(model.W, dWs), (model.b, dbs)]:
        for P, G in zip(params, grads):
            flat_idx = rng.choice(P.size, size=min(n_samples, P.size), replace=False)
            for fi in flat_idx:
                idx = np.unravel_index(fi, P.shape)
                old = P[idx]
                P[idx] = old + eps
                lp = model.loss(X, y)
                P[idx] = old - eps
                lm = model.loss(X, y)
                P[idx] = old                      # always restore!
                num = (lp - lm) / (2 * eps)
                rel = abs(num - G[idx]) / max(abs(num), abs(G[idx]), 1e-8)
                worst = max(worst, rel)
    return worst


xb, yb = X_tr[:16], y_tr[:16]
for act in ["tanh", "relu"]:
    m = MLP([64, 20, 10], activation=act, seed=1)
    print(f"{act:>4}: worst relative error = {gradient_check(m, xb, yb):.2e}")

# %% [markdown]
# Both are far below the 1e-6 threshold, so the hand-derived backward pass is correct. (ReLU has a kink at 0 where the derivative is
# undefined, so on rare occasions a nudge that crosses the kink produces a false alarm; re-check with a different seed before panicking.)
#
# Now the reason this technique exists: **it catches bugs**. Here is a classic one, forgetting to divide by the batch size.

# %%
def buggy_backward(model, X, y):
    loss, dWs, dbs = model.backward(X, y)
    n = len(y)
    return loss, [g * n for g in dWs], [g * n for g in dbs]   # pretend we forgot the 1/n


m = MLP([64, 20, 10], seed=1)
print(f"correct gradient: worst relative error = {gradient_check(m, xb, yb):.2e}")
print(f"buggy gradient:   worst relative error = {gradient_check(m, xb, yb, grads_fn=buggy_backward):.2e}")

# %% [markdown]
# The buggy version is off by a factor of 16 (the batch size), so its relative error is near 1. A bug like this would *not* crash: the network would still
# "learn", just with an effective learning rate that changes with batch size. Silent bugs are the dangerous ones.

# %% [markdown]
# ## 4. Training with mini-batch gradient descent
#
# With correct gradients, training is a loop: shuffle, take a mini-batch, compute gradients, step downhill
# `param -= lr * grad`. One pass over the training set is an **epoch**.

# %%
def train(model, X, y, X_val, y_val, epochs=40, lr=0.05, batch_size=32, seed=42):
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        for i in range(0, len(X), batch_size):
            j = order[i:i + batch_size]
            _, dWs, dbs = model.backward(X[j], y[j])
            for P, G in zip(model.W + model.b, dWs + dbs):
                P -= lr * G                        # in-place update
        history["train_loss"].append(model.loss(X, y))
        history["val_loss"].append(model.loss(X_val, y_val))
        history["val_acc"].append(float(np.mean(model.predict(X_val) == y_val)))
    return history


net = MLP([64, 32, 10], seed=42)
print(f"before training: loss {net.loss(X_tr, y_tr):.3f}, test accuracy {np.mean(net.predict(X_te) == y_te):.3f}")
hist = train(net, X_tr, y_tr, X_te, y_te)
print(f"after  training: loss {hist['train_loss'][-1]:.3f}, test accuracy {hist['val_acc'][-1]:.3f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist["train_loss"], label="train")
ax[0].plot(hist["val_loss"], label="validation")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("cross-entropy loss")
ax[0].set_title("Loss falls as gradient descent runs")
ax[0].legend()
ax[1].plot(hist["val_acc"])
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("accuracy on held-out digits")
ax[1].set_title("Held-out accuracy")
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* A 2,410-parameter network went from a random guesser (its untrained accuracy is printed above) to high accuracy on unseen digits using nothing but the formulas in the table.
# Every framework does this same loop; what they add is automatic differentiation (so you never derive `backward` yourself),
# GPUs, and better optimisers (notebook 03).
#
# Let us look at what the trained hidden units learned by drawing their incoming weights as 8x8 images.

# %%
fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
for k, a_ in enumerate(axes.ravel()):
    a_.imshow(net.W[0][:, k].reshape(8, 8), cmap="RdBu")
    a_.set_xticks([])
    a_.set_yticks([])
fig.suptitle("Incoming weights of 16 hidden units (red/blue = positive/negative)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Individual units respond to stroke-like pixel patterns. You would not have programmed those detectors; gradient descent found them.

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting to divide by the batch size** (or dividing twice): the loss still goes down, so nothing crashes. Gradient checking catches it.
# 2. **Mutating the cache / using the wrong `z`:** the backward pass needs the *forward* values of the same batch. Recomputing with different data corrupts gradients.
# 3. **Shape bugs that broadcast silently:** `dz.sum(axis=0)` for the bias, not `.sum()`. A gradient with the wrong shape may still "work" through broadcasting.
# 4. **Gradient checking with a large `eps` or in `float32`:** rounding noise dominates. Use float64 and `eps` around 1e-6 (shown above).
# 5. **Leaving gradient checking on during training:** it costs two forward passes per parameter. Check once on a tiny batch, then turn it off.

# %%
# Demonstration of mistake 4: float32 makes the numerical gradient useless
def numeric_grad_f32(eps):
    xx, bb, tt = np.float32(1.5), np.float32(-0.2), np.float32(1.0)

    def L(w_):
        zz = np.float32(w_) * xx + bb
        aa = np.float32(1.0) / (np.float32(1.0) + np.exp(-zz))
        return (aa - tt) ** 2

    w0 = np.float32(0.4)
    return float((L(w0 + np.float32(eps)) - L(w0 - np.float32(eps))) / np.float32(2 * eps))


print(f"float32 numeric gradient with eps=1e-6: {numeric_grad_f32(1e-6):.4f}   (float64 analytic value: {dloss_dw:.4f})")

# %% [markdown]
# ## Interview corner
#
# **1. What is backpropagation, in one paragraph?**
# <details><summary>Answer</summary>
# An efficient application of the chain rule. A forward pass computes and stores every intermediate value; a backward pass then walks from the loss to the inputs,
# multiplying local derivatives and reusing results, so the gradient of the loss with respect to <em>all</em> parameters costs roughly one extra forward pass, instead of one evaluation per parameter.
# </details>
#
# **2. Why is the gradient of softmax + cross-entropy with respect to the logits just `P - Y`?**
# <details><summary>Answer</summary>
# The softmax Jacobian and the derivative of the log cancel neatly. Intuitively: each logit is pushed up in proportion to how much probability it is missing (true class)
# or how much it wrongly holds (other classes). This is also why frameworks fuse softmax and cross-entropy into one numerically stable op.
# </details>
#
# **3. How do you verify a custom gradient?**
# <details><summary>Answer</summary>
# Gradient checking: compare against centred finite differences <code>(L(w+eps) - L(w-eps)) / (2 eps)</code> in float64 on a tiny input, using relative error (below about 1e-6 is good, above 1e-3 suggests a bug).
# PyTorch ships <code>torch.autograd.gradcheck</code> for exactly this.
# </details>
#
# **4. Why do we need to store activations during the forward pass, and what does that cost?**
# <details><summary>Answer</summary>
# Local derivatives (like the ReLU mask or <code>A1.T</code> in <code>dW2</code>) need the forward values. Memory therefore grows with batch size and depth, which is why large models use
# tricks such as gradient checkpointing (recompute instead of store).
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A numerical gradient function
# Write `numerical_gradient(f, x, eps=1e-6)` for a scalar function `f` of a NumPy vector `x`. Return an array of the same shape with the centred
# finite-difference estimate for each coordinate. Do not modify the caller's `x` permanently.
# *Hint: copy `x`, loop over indices, add and subtract `eps`.*

# %% tags=["starter"]
def numerical_gradient(f, x, eps=1e-6):
    ...

# %%
xt = np.array([1.0, -2.0, 0.5])
g1 = numerical_gradient(lambda v: np.sum(v ** 2), xt)
assert np.allclose(g1, 2 * xt, atol=1e-6), f"d/dx sum(x^2) = 2x, got {g1}"
g2 = numerical_gradient(lambda v: np.sin(v[0]) * v[1] + np.exp(v[2]), xt)
expected = np.array([np.cos(xt[0]) * xt[1], np.sin(xt[0]), np.exp(xt[2])])
assert np.allclose(g2, expected, atol=1e-6), f"got {g2}, expected {expected}"
assert np.array_equal(xt, [1.0, -2.0, 0.5]), "do not leave x modified"
print("Exercise 1 passed")

# %% tags=["solution"]
def numerical_gradient(f, x, eps=1e-6):
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        old = x.flat[i]
        x.flat[i] = old + eps
        fp = f(x)
        x.flat[i] = old - eps
        fm = f(x)
        x.flat[i] = old
        grad.flat[i] = (fp - fm) / (2 * eps)
    return grad

# %% [markdown]
# ### Exercise 2: Backprop for logistic regression
# A logistic regression is a network with no hidden layer: `p = sigmoid(X @ w + b)`, loss = mean binary cross-entropy
# `-mean(y log p + (1 - y) log(1 - p))`. Implement `logreg_grads(w, b, X, y)` returning `(dw, db)`.
# *Hint: the same beautiful result as softmax applies: `d loss / d z = (p - y) / n`.*

# %% tags=["starter"]
def logreg_grads(w, b, X, y):
    ...

# %%
rng_t = np.random.default_rng(3)
Xl = rng_t.normal(size=(40, 3))
yl = (Xl[:, 0] - Xl[:, 1] + 0.3 * rng_t.normal(size=40) > 0).astype(float)
wl, bl = rng_t.normal(size=3), 0.1


def _bce(theta):
    p = sigmoid(Xl @ theta[:3] + theta[3])
    return -np.mean(yl * np.log(p) + (1 - yl) * np.log(1 - p))


ref = np.zeros(4)
for i in range(4):
    e = np.zeros(4); e[i] = 1e-6
    th = np.append(wl, bl)
    ref[i] = (_bce(th + e) - _bce(th - e)) / 2e-6
dw, db = logreg_grads(wl, bl, Xl, yl)
assert np.shape(dw) == (3,), "dw has the same shape as w"
assert np.allclose(np.append(dw, db), ref, atol=1e-6), f"analytic {np.append(dw, db)} vs numeric {ref}"
print("Exercise 2 passed")

# %% tags=["solution"]
def logreg_grads(w, b, X, y):
    p = sigmoid(X @ w + b)
    dz = (p - y) / len(y)
    return X.T @ dz, dz.sum()

# %% [markdown]
# ### Exercise 3: Catch the bug
# The function below claims to compute gradients for our `MLP` but it has a planted bug (it forgets the ReLU mask in the hidden layer).
# Use `gradient_check` (defined above) on a fresh `MLP([64, 20, 10], seed=1)` and the batch `xb, yb`, passing `grads_fn=sneaky_backward`.
# Store the worst relative error in `worst_err`, and store `True` in `looks_correct` if that error is below `1e-4`, else `False`.

# %% tags=["starter"]
def sneaky_backward(model, X, y):
    n = len(y)
    p, cache = model.forward(X)
    dz = p.copy()
    dz[np.arange(n), y] -= 1.0
    dz /= n
    dWs, dbs = [], []
    for layer in reversed(range(len(model.W))):
        a_in, z = cache[layer]
        dWs.append(a_in.T @ dz)          # BUG: the activation slope is never applied to hidden layers
        dbs.append(dz.sum(axis=0))
        dz = dz @ model.W[layer].T
    return 0.0, dWs[::-1], dbs[::-1]


worst_err = ...
looks_correct = ...

# %%
assert 0.01 < worst_err < 10, f"unexpected error size {worst_err}: pass grads_fn=sneaky_backward to gradient_check"
assert looks_correct is False, "a gradient with an error this large is wrong"
print(f"Exercise 3 passed (worst relative error {worst_err:.2f})")

# %% tags=["solution"]
def sneaky_backward(model, X, y):
    n = len(y)
    p, cache = model.forward(X)
    dz = p.copy()
    dz[np.arange(n), y] -= 1.0
    dz /= n
    dWs, dbs = [], []
    for layer in reversed(range(len(model.W))):
        a_in, z = cache[layer]
        dWs.append(a_in.T @ dz)
        dbs.append(dz.sum(axis=0))
        dz = dz @ model.W[layer].T
    return 0.0, dWs[::-1], dbs[::-1]


worst_err = gradient_check(MLP([64, 20, 10], seed=1), xb, yb, grads_fn=sneaky_backward)
looks_correct = bool(worst_err < 1e-4)

# %% [markdown]
# ### Exercise 4: Does a wider hidden layer help?
# Train `MLP([64, H, 10], seed=42)` with the `train` function (default settings) for `H` in `[4, 16, 64]`. Store the final held-out accuracy of each in a dict
# `acc_by_width` such as `{4: 0.83, ...}` (your numbers will differ). Store the width with the best accuracy in `best_width`.

# %% tags=["starter"]
acc_by_width = {}
best_width = ...

# %%
ref_acc = {}
for H in [4, 16, 64]:
    ref_acc[H] = train(MLP([64, H, 10], seed=42), X_tr, y_tr, X_te, y_te)["val_acc"][-1]
assert set(acc_by_width) == {4, 16, 64}, "the dict needs the keys 4, 16 and 64"
for H in ref_acc:
    assert abs(acc_by_width[H] - ref_acc[H]) < 1e-9, f"width {H}: expected {ref_acc[H]:.4f}, got {acc_by_width[H]:.4f} (use train(...) defaults and seed=42)"
assert best_width == max(ref_acc, key=ref_acc.get)
assert ref_acc[16] > ref_acc[4], "sanity: 16 hidden units should beat 4 on digits"
print("Exercise 4 passed:", {k: round(v, 3) for k, v in acc_by_width.items()})

# %% tags=["solution"]
acc_by_width = {}
for H in [4, 16, 64]:
    acc_by_width[H] = train(MLP([64, H, 10], seed=42), X_tr, y_tr, X_te, y_te)["val_acc"][-1]
best_width = max(acc_by_width, key=acc_by_width.get)

# %% [markdown]
# ### Exercise 5 (open-ended): Which digits does the network confuse?
# Compute the confusion matrix of the trained `net` on the test set (`sklearn.metrics.confusion_matrix`) and find the most confused pair of
# distinct digits. Write one sentence on whether that pair looks similar to a human. No assert: explore and report.

# %% [markdown]
# ## Recap
# - Backprop = chain rule from the loss backwards, reusing stored forward values.
# - For softmax + cross-entropy the output error is `(P - Y) / n`; for each dense layer: `dW = A_in.T @ dZ`, `db = dZ.sum(0)`, `dA_prev = dZ @ W.T`, then multiply by the activation slope.
# - Gradient checking (centred differences, float64, relative error) is how you trust a hand-written or custom gradient.
# - Training is shuffle, mini-batch, gradient, step, repeated for epochs; watch train and validation curves.
#
# **Go deeper:** the PyTorch documentation page "Automatic differentiation with `torch.autograd`" (pytorch.org/tutorials), and
# Stanford CS231n's lecture notes on backpropagation (cs231n.github.io).
#
# **Next:** [Training dynamics, optimizers and regularization](03_training_dynamics_optimizers_and_regularization.ipynb)
