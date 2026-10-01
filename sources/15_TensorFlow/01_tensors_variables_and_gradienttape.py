# %% [markdown]
# # Tensors, Variables and GradientTape
#
# **Why this matters.** Every neural network, in any framework, is "multiply some arrays, compare with the
# truth, and nudge the parameters in the direction that reduces the error". TensorFlow gives you the arrays
# (tensors), the parameters (variables) and an automatic-differentiation engine (`GradientTape`) that computes
# the nudge for you. When a Keras model misbehaves at work, this is the layer you drop down to in order to
# see what is really going on (exploding gradients, a frozen layer, a `None` gradient).
#
# **You will learn**
# - create and manipulate tensors, and move between tensors and NumPy arrays
# - use `tf.Variable` for trainable state and understand why tensors are immutable
# - compute gradients with `tf.GradientTape` and recognise the common ways it silently returns `None` (and the one that raises instead)
# - fit a linear regression by hand with gradient descent and verify it against a closed-form solution
#
# **Prerequisites:** [NumPy arrays](../04_NumPy/), [PyTorch autograd](../14_PyTorch/) (helpful, not required),
# and gradient descent from [Math for Data Science](../02_Math_for_Data_Science/).
#
# > Data used: `datasets/housing.csv`, a synthetic house-price table (see `docs/DATASETS.md`).

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's C++ info logs
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
print("TensorFlow", tf.__version__)
print("devices:", [d.device_type for d in tf.config.list_physical_devices()])

# %% [markdown]
# ## 1. Tensors: NumPy arrays with a dtype, a shape and (optionally) a GPU
#
# A **tensor** is an n-dimensional array. In TensorFlow 2 operations run **eagerly**: `a + b` executes
# immediately and gives you a value you can print, exactly like NumPy. That is why TF2 feels like normal Python.

# %%
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([10.0, 20.0])
print(a.shape, a.dtype)
print(a + b)  # broadcasting works just like NumPy: b is added to every row

# %% [markdown]
# *What just happened?* `tf.constant` built a 2x2 `float32` tensor. **Default dtype is float32**, not float64 like
# NumPy: neural networks are trained in 32-bit (or lower) because it is faster and rarely hurts accuracy.
# Mixing dtypes is an error in TF; NumPy would quietly upcast:

# %%
x_int = tf.constant([1, 2, 3])
try:
    x_int + tf.constant([0.5, 0.5, 0.5])
except (tf.errors.InvalidArgumentError, TypeError) as e:
    print(type(e).__name__, "-", str(e)[:100], "...")
print(tf.cast(x_int, tf.float32) + 0.5)  # be explicit

# %% [markdown]
# ### Tensors and NumPy
#
# `.numpy()` gives you an array; NumPy functions accept tensors. Use this to inspect values or to hand results to
# pandas/matplotlib. Tensors are **immutable**: you cannot write `a[0, 0] = 5`.

# %%
arr = a.numpy()
print(type(arr).__name__, arr.dtype)
print(np.mean(a))  # NumPy accepts a tensor and returns a NumPy scalar
try:
    a[0, 0] = 5.0
except TypeError as e:
    print("TypeError:", e)

# %% [markdown]
# Typical operations you will use daily. Shapes are the thing to watch: most bugs are shape bugs.

# %%
m = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
print("sum over axis 0:", tf.reduce_sum(m, axis=0).numpy())
print("mean of each row:", tf.reduce_mean(m, axis=1).numpy())
print("matmul (3x4) @ (4x2) ->", tf.matmul(m, tf.ones((4, 2))).shape)
print("slicing like NumPy:", m[1:, ::2].numpy().tolist())

# %% [markdown]
# ## 2. Variables: tensors that can change
#
# Model weights must be updated thousands of times, so TF has a mutable container, `tf.Variable`. Layers in Keras
# create these for you; here we make one by hand. `assign` and `assign_sub` change the value in place.

# %%
w = tf.Variable([1.0, 2.0], name="w")
w.assign_sub([0.5, 0.5])  # w = w - [0.5, 0.5], in place
print(w.numpy(), "| trainable:", w.trainable)

# %% [markdown]
# `trainable=True` (the default) tells `GradientTape` to watch this variable automatically. Note `w - 1` gives a
# plain *tensor*, not a variable: only `assign*` methods mutate state. Forgetting this is a classic bug
# (`w = w - lr * grad` rebinds the Python name to a constant tensor and training silently stops).

# %%
w2 = w - 1.0
print(type(w2).__name__, "vs", type(w).__name__)

# %% [markdown]
# ## 3. GradientTape: automatic differentiation
#
# **Intuition.** Picture a tape recorder. While the `with tf.GradientTape()` block is open, every operation on a
# watched variable is written to the tape. Afterwards you call `tape.gradient(loss, variables)` and TF plays the
# tape **backwards**, applying the chain rule at each step (reverse-mode autodiff, a.k.a. backpropagation).
#
# Start with something you can check by hand: for `y = x^2` the derivative is `2x`, so at `x = 3` it is 6.

# %%
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2
dy_dx = tape.gradient(y, x)
print("dy/dx at x=3:", dy_dx.numpy())

# %% [markdown]
# Several variables at once give a list of gradients in the same order. Try `f = x^2 * y + sin(y)`:
# analytically `df/dx = 2xy` and `df/dy = x^2 + cos(y)`.

# %%
x = tf.Variable(3.0)
y = tf.Variable(2.0)
with tf.GradientTape() as tape:
    f = x**2 * y + tf.sin(y)
gx, gy = tape.gradient(f, [x, y])
print("df/dx:", gx.numpy(), "expected", 2 * 3.0 * 2.0)
print("df/dy:", gy.numpy(), "expected", 3.0**2 + np.cos(2.0))

# %% [markdown]
# ### Compared with PyTorch
#
# If you know PyTorch autograd the ideas are identical, only the API differs. In PyTorch each tensor carries
# `requires_grad=True`, the graph is recorded as operations run, you call `loss.backward()` and gradients
# *accumulate* into `param.grad` (so you must zero them each step). In TensorFlow the recording is **scoped** to a
# `with GradientTape()` block, `tape.gradient(loss, params)` **returns** the gradients as values (nothing is
# accumulated, nothing to zero), and the tape is freed after one call unless `persistent=True`. TF's explicit
# scoping makes it easy to see exactly which computation you differentiate; PyTorch's implicit graph is a bit
# terser. Both are eager by default; TF additionally lets you compile functions into graphs (notebook 03).

# %% [markdown]
# ### Watching constants and using a persistent tape
# Only *trainable variables* are watched automatically. For a plain tensor call `tape.watch`. A tape can be used
# for one `gradient()` call; ask for `persistent=True` to reuse it (and `del` it afterwards).

# %%
c = tf.constant(3.0)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(c)
    y1 = c**2
    y2 = c**3
print(tape.gradient(y1, c).numpy(), tape.gradient(y2, c).numpy(), "(expected 6 and 27)")
del tape

# %% [markdown]
# Second derivatives come from nesting tapes: the outer tape records the inner gradient computation.
# For `y = x^3`, `y'' = 6x`, which is 12 at `x = 2`.

# %%
x = tf.Variable(2.0)
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = x**3
    dy = inner.gradient(y, x)  # computed inside the outer tape so it is recorded too
d2y = outer.gradient(dy, x)
print("y'' at 2:", d2y.numpy())

# %% [markdown]
# ## 4. Fitting a line by hand
#
# We will predict house `price` (in $1000s) from `area_sqft` with the model `price = w * area + b`, training
# `w` and `b` with gradient descent and our own loop: predict, compute mean squared error (MSE), get gradients,
# update. This is *exactly* what `model.fit` does for a one-layer network. We standardise both columns first
# (zero mean, unit variance) so a single learning rate works.

# %%
housing = pd.read_csv(Path("../datasets/housing.csv"))
h = housing.dropna(subset=["area_sqft", "price"])
lo, hi = h["price"].quantile([0.01, 0.99])
h = h[h["price"].between(lo, hi)]  # drop the planted absurd prices (see DATASETS.md)
print(len(housing), "rows ->", len(h), "after removing missing values and extreme prices")

xa = ((h["area_sqft"] - h["area_sqft"].mean()) / h["area_sqft"].std()).to_numpy(dtype="float32")
ya = ((h["price"] - h["price"].mean()) / h["price"].std()).to_numpy(dtype="float32")
X, Y = tf.constant(xa), tf.constant(ya)

# %%
w = tf.Variable(0.0)
b = tf.Variable(0.0)
lr = 0.1
losses = []
for step in range(100):
    with tf.GradientTape() as tape:
        pred = w * X + b
        loss = tf.reduce_mean((pred - Y) ** 2)  # MSE
    gw, gb = tape.gradient(loss, [w, b])
    w.assign_sub(lr * gw)  # in-place update keeps w a Variable
    b.assign_sub(lr * gb)
    losses.append(float(loss))
print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}   w={w.numpy():.3f} b={b.numpy():.4f}")

# %% [markdown]
# Verify with a method that does not use gradients at all: the closed-form least-squares solution.
# For one standardised feature the best slope is the correlation coefficient between the two columns.

# %%
w_ls, b_ls = np.polyfit(xa, ya, 1)
print(f"least squares: w={w_ls:.3f} b={b_ls:.4f}   correlation={np.corrcoef(xa, ya)[0, 1]:.3f}")
assert abs(w.numpy() - w_ls) < 0.01, "gradient descent should land on the least-squares slope"

# %% [markdown]
# *What just happened?* 100 steps of gradient descent recovered the analytic answer. That agreement is your
# test that the tape is differentiating what you think it is. Whenever you write a custom loss or layer, do
# this kind of sanity check against something simple.

# %% [markdown]
# ## Common mistakes
#
# 1. **Gradient is `None`.** The tape has nothing connecting `loss` to your variable. The usual causes are shown below.
# 2. **Leaving TensorFlow** inside the tape (`.numpy()`, `np.*` functions, Python `float(...)`) cuts the recording.
# 3. **Calling `tape.gradient` twice** on a non-persistent tape raises `RuntimeError`.
# 4. **Rebinding a Variable** (`w = w - lr*g`) instead of `w.assign_sub(...)`.
# 5. **Integer tensors have no gradient**: cast to float first.

# %%
v = tf.Variable(2.0)

# (a) computing the loss OUTSIDE the tape
y_out = v * 3.0
with tf.GradientTape() as tape:
    pass
print("outside tape ->", tape.gradient(y_out, v))

# (b) leaving TensorFlow inside the tape
with tf.GradientTape() as tape:
    y_np = tf.constant(v.numpy() * 3.0)  # numpy round trip: the link to v is gone
print("via numpy    ->", tape.gradient(y_np, v))

# (c) a constant that was never watched
k = tf.constant(2.0)
with tf.GradientTape() as tape:
    y_c = k * 3.0
print("unwatched    ->", tape.gradient(y_c, k))

# (d) second call on a non-persistent tape
with tf.GradientTape() as tape:
    y_ok = v * 3.0
tape.gradient(y_ok, v)
try:
    tape.gradient(y_ok, v)
except RuntimeError as e:
    print("second call  -> RuntimeError:", str(e)[:70], "...")

# %% [markdown]
# ## Interview corner
#
# **1. What does `tf.GradientTape` do, and why is it needed if TF2 runs eagerly?**
# <details><summary>Answer</summary>
# Eager execution only computes values. To train you also need derivatives of the loss with respect to the parameters.
# The tape records operations on watched tensors during the forward pass and replays them in reverse (reverse-mode autodiff)
# to obtain gradients efficiently, one backward pass no matter how many parameters there are.
# </details>
#
# **2. Why did `tape.gradient` return `None`?**
# <details><summary>Answer</summary>
# There is no recorded path from the target to the source: the computation happened outside the tape, went through NumPy or
# <code>.numpy()</code>, used a constant that was not <code>watch</code>ed, used integer dtypes, or the op has no defined gradient
# (for example <code>argmax</code>, rounding). Check the chain of ops between the variable and the loss.
# </details>
#
# **3. Tensor vs Variable?**
# <details><summary>Answer</summary>
# Tensors are immutable values. A Variable is mutable state (weights, optimizer slots, counters) that persists across steps,
# is trainable by default and is watched automatically by the tape.
# </details>
#
# **4. How do TensorFlow and PyTorch differ in computing gradients?**
# <details><summary>Answer</summary>
# PyTorch: <code>loss.backward()</code> accumulates into <code>.grad</code> and you zero it each step. TensorFlow:
# <code>tape.gradient(loss, vars)</code> returns fresh gradients from a tape scoped by a <code>with</code> block.
# Same underlying reverse-mode autodiff.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Centre a matrix with broadcasting
# Given the tensor `M` below, create `centered`: `M` minus its **column means** (so every column averages to 0),
# using TensorFlow ops only. *Hint: `tf.reduce_mean(M, axis=0)` has shape (3,) and broadcasts across rows.*

# %%
M = tf.constant([[1.0, 10.0, 100.0], [2.0, 20.0, 300.0], [3.0, 60.0, 200.0], [6.0, 50.0, 400.0]])

# %% tags=["starter"]
centered = ...

# %%
assert isinstance(centered, tf.Tensor), "centered should be a tf.Tensor"
expected = M.numpy() - M.numpy().mean(axis=0)
assert np.allclose(centered.numpy(), expected, atol=1e-5), "subtract the mean of each COLUMN (axis=0)"
assert np.allclose(centered.numpy().mean(axis=0), 0, atol=1e-5)
print("Exercise 1 passed")

# %% tags=["solution"]
centered = M - tf.reduce_mean(M, axis=0)

# %% [markdown]
# ### Exercise 2: Gradient of a function of two variables
# For `f(p, q) = p * q**2 + exp(p)` compute the gradient at `p = 1.5`, `q = -2.0`. Store the two partial derivatives
# in `dp` and `dq` (as Python floats). *Hint: create two `tf.Variable`s and use `tf.exp`. Derive the answer by hand
# first: `df/dp = q^2 + exp(p)`, `df/dq = 2pq`.*

# %% tags=["starter"]
dp = ...
dq = ...

# %%
p0, q0 = 1.5, -2.0
assert abs(dp - (q0**2 + np.exp(p0))) < 1e-3, f"df/dp should be q^2 + exp(p) = {q0**2 + np.exp(p0):.4f}"
assert abs(dq - (2 * p0 * q0)) < 1e-3, f"df/dq should be 2pq = {2 * p0 * q0}"
print("Exercise 2 passed")

# %% tags=["solution"]
p = tf.Variable(1.5)
q = tf.Variable(-2.0)
with tf.GradientTape() as tape:
    f = p * q**2 + tf.exp(p)
g_p, g_q = tape.gradient(f, [p, q])
dp, dq = float(g_p), float(g_q)

# %% [markdown]
# ### Exercise 3: Two features, by hand
# Extend section 4: predict standardised `price` from **two** standardised features, `area_sqft` and `quality_score`
# (drop rows where either is missing, and use the same price trimming). Train weights `w` (a `tf.Variable` of shape `(2,)`)
# and `b` with `GradientTape` for 300 steps at learning rate 0.1. Store the final weights as a NumPy array `w_final`.
# The check compares against `np.linalg.lstsq`. *Hint: build `F` with shape (n, 2); predict with `tf.linalg.matvec(F, w) + b`.*

# %%
h2 = housing.dropna(subset=["area_sqft", "quality_score", "price"])
lo2, hi2 = h2["price"].quantile([0.01, 0.99])
h2 = h2[h2["price"].between(lo2, hi2)]
z = lambda s: ((s - s.mean()) / s.std()).to_numpy(dtype="float32")
F = tf.constant(np.column_stack([z(h2["area_sqft"]), z(h2["quality_score"])]))
T = tf.constant(z(h2["price"]))
print(F.shape, T.shape)

# %% tags=["starter"]
w_final = ...

# %%
Fn, Tn = F.numpy(), T.numpy()
A = np.column_stack([Fn, np.ones(len(Fn), dtype="float32")])
sol = np.linalg.lstsq(A, Tn, rcond=None)[0]
w_final = np.asarray(w_final)
assert w_final.shape == (2,), "w_final should hold two weights"
assert np.allclose(w_final, sol[:2], atol=0.02), f"expected about {sol[:2].round(3)}, got {w_final.round(3)}; did you train long enough?"
print("Exercise 3 passed, weights (area, quality):", w_final.round(3))

# %% tags=["solution"]
w = tf.Variable(tf.zeros(2))
b = tf.Variable(0.0)
for _ in range(300):
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((tf.linalg.matvec(F, w) + b - T) ** 2)
    gw, gb = tape.gradient(loss, [w, b])
    w.assign_sub(0.1 * gw)
    b.assign_sub(0.1 * gb)
w_final = w.numpy()

# %% [markdown]
# ### Exercise 4 (open-ended): Break it on purpose
# Take the training loop from section 4 and change **one** thing: set `lr = 1.5`, or replace `assign_sub` by
# `w = w - lr * gw`. Observe what happens to the loss and to `w`, and write two sentences in a markdown cell
# explaining *why* (hint: what does each change do to the update step or to the tape's view of the variable?).

# %% [markdown]
# ## Recap
# - Tensors are immutable `float32`-by-default arrays with NumPy-like broadcasting; Variables hold mutable, trainable state.
# - `with tf.GradientTape() as tape:` records; `tape.gradient(loss, vars)` returns gradients (same order as `vars`).
# - `None` gradients mean the recording was cut: outside the tape, through NumPy, unwatched constants, or non-differentiable ops.
# - A hand-written loop (predict, loss, gradient, `assign_sub`) is what `model.fit` automates; verify it against a closed form.
#
# **Go deeper:** the TensorFlow guides "Introduction to gradients and automatic differentiation" and
# "Introduction to Variables" at tensorflow.org/guide.
#
# **Next:** [tf.data input pipelines](02_tf_data_input_pipelines.ipynb)
