# %% [markdown]
# # Custom Training Loop and `tf.function`
#
# **Why this matters.** `model.fit` covers 90% of training jobs. The other 10% is where you earn your salary:
# a loss that weighs mistakes unevenly, two networks trained together, gradient clipping on a specific layer,
# logging exactly the metric your stakeholder asked for. For those you write the loop yourself, and to make it
# fast you wrap the step in `tf.function`. This notebook shows how, and the retracing traps that quietly make a
# "compiled" function slower than plain Python.
#
# **You will learn**
# - what `tf.function` does (Python function -> TensorFlow graph) and measure the speed-up honestly
# - how tracing works, and how to provoke and fix **retracing**
# - why `print` behaves differently inside a graph and how to use `tf.print`
# - write a complete custom training loop with `GradientTape`, an optimizer and Keras metrics
#
# **Prerequisites:** [Tensors, Variables and GradientTape](01_tensors_variables_and_gradienttape.ipynb),
# [tf.data pipelines](02_tf_data_input_pipelines.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic telecom customers, see `docs/DATASETS.md`).

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

keras.utils.set_random_seed(42)  # seeds Python, NumPy and TensorFlow together
print("TF", tf.__version__, "| Keras", keras.__version__)

# %% [markdown]
# ## 1. Eager vs graph
#
# **Intuition.** Eager mode is a *live interpreter*: Python drives, and each TF op is dispatched one at a time.
# `tf.function` is a *compiler*: the first call **traces** your Python code once, recording the TF ops into a graph
# (a data-flow program). Later calls run that graph directly in C++ with no Python in the loop, and TensorFlow can
# fuse and optimise operations. The trade-off: tracing costs time, and your Python code only runs during tracing.
#
# Here is a function with many small ops, where Python dispatch overhead dominates:

# %%
rng = np.random.default_rng(42)
W = tf.constant(rng.normal(0, 0.1, (32, 32)).astype("float32"))
x0 = tf.constant(rng.normal(size=(64, 32)).astype("float32"))


def deep_stack(x):
    for _ in range(50):  # an ordinary Python loop: 50 x (matmul + add + tanh)
        x = tf.tanh(x @ W + 0.1)
    return x


graph_stack = tf.function(deep_stack)
print("same result?", np.allclose(deep_stack(x0).numpy(), graph_stack(x0).numpy(), atol=1e-5))


def avg_ms(fn, n=30):
    t0 = time.perf_counter()
    for _ in range(n):
        fn(x0)
    return (time.perf_counter() - t0) / n * 1000


print(f"eager: {avg_ms(deep_stack):6.2f} ms/call")
print(f"graph: {avg_ms(graph_stack):6.2f} ms/call   (first call already traced above)")

# %% [markdown]
# *What just happened?* Same maths, same result, but the graph version avoids 150 Python-to-TensorFlow dispatches per
# call. The gap is largest for many small ops (like this) and shrinks when each op is big (large matmuls on a
# CPU spend their time in the kernel anyway). Do not promise a fixed speed-up: measure your own step.

# %% [markdown]
# ## 2. Tracing: your Python runs only while tracing
#
# Put an ordinary `print` and a Python counter inside a `tf.function` and watch when they run. `tf.print` is
# the graph-safe version: it becomes an op in the graph and runs on every call.

# %%
trace_log = []


@tf.function
def double(x):
    trace_log.append("traced")  # Python side effect: runs only during tracing
    print("python print, x =", x)  # shows a symbolic tensor, not a value
    tf.print("tf.print, x =", x)  # runs every call, shows the real value
    return x * 2


double(tf.constant(1.0))
double(tf.constant(2.0))
double(tf.constant(3.0))
print("times the Python body executed:", len(trace_log))

# %% [markdown]
# Three calls, one trace. The `print` fired once and showed a *symbolic* tensor (no value exists at trace time);
# `tf.print` fired three times with real values. **Rule:** treat a `tf.function` body as a program to be *compiled*, not run:
# side effects (appending to a list, mutating a Python object, printing) happen at trace time only.
#
# AutoGraph also rewrites Python control flow that depends on **tensor values** into graph ops (`tf.cond`, `tf.while_loop`):

# %%
@tf.function
def clip_or_flip(x):
    if tf.reduce_sum(x) > 0:  # depends on a tensor VALUE -> becomes tf.cond in the graph
        return tf.minimum(x, 1.0)
    return -x


print(clip_or_flip(tf.constant([2.0, 0.5])).numpy(), clip_or_flip(tf.constant([-2.0, 0.5])).numpy())

# %% [markdown]
# ## 3. Retracing: the silent performance bug
#
# A trace is cached by the **signature** of the arguments: tensor `shape` and `dtype`, but for plain Python values
# (`int`, `float`, `str`) the *actual value*. A new signature means a fresh trace, which costs far more than the call itself.

# %%
@tf.function
def scale(x, factor):
    return x * factor


for f in [1, 2, 3, 4]:
    scale(tf.constant([1.0, 2.0]), f)  # Python ints: each new VALUE retraces
print("Python int factors  -> traces:", scale.experimental_get_tracing_count())


@tf.function
def scale_t(x, factor):
    return x * factor


for f in [1.0, 2.0, 3.0, 4.0]:
    scale_t(tf.constant([1.0, 2.0]), tf.constant(f))  # tensors: same dtype/shape -> reuse
print("tensor factors      -> traces:", scale_t.experimental_get_tracing_count())

# %% [markdown]
# Changing **shapes** also retraces (variable-length batches, the last small batch of an epoch). Two fixes:
# declare an `input_signature` with `None` for the flexible dimension, or pass `reduce_retracing=True` so TensorFlow
# generalises the shape after a couple of traces.

# %%
@tf.function
def total(x):
    return tf.reduce_sum(x)


for n in [2, 3, 4, 5, 6]:
    total(tf.ones(n))
print("varying length, no signature:", total.experimental_get_tracing_count(), "traces")


@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def total_sig(x):
    return tf.reduce_sum(x)


for n in [2, 3, 4, 5, 6]:
    total_sig(tf.ones(n))
print("with input_signature [None] :", total_sig.experimental_get_tracing_count(), "trace")

# %% [markdown]
# How much do retraces cost? Time a function that retraces every call against one that does not.

# %%
@tf.function
def add_const(x, k):
    return x + k


t0 = time.perf_counter()
for k in range(20):
    add_const(tf.constant(1.0), k)  # 20 different Python ints -> 20 traces
retrace_time = time.perf_counter() - t0

t0 = time.perf_counter()
for k in range(20):
    add_const(tf.constant(1.0), tf.constant(float(k)))  # tensors -> a couple of traces at most
tensor_time = time.perf_counter() - t0
print(f"20 calls, Python int arg: {retrace_time * 1000:.0f} ms | tensor arg: {tensor_time * 1000:.0f} ms")

# %% [markdown]
# TensorFlow also warns (`tf.function retracing` warning) when it sees this pattern; do not silence it, fix it.

# %% [markdown]
# ## 4. A custom training loop
#
# The loop has four repeating parts: **forward pass** under a tape, **loss**, **gradients**, **optimizer step**. We will
# train a small network on churn. First the data, prepared exactly as in the previous notebook.

# %%
churn = pd.read_csv(Path("../datasets/churn.csv"))
churn["contract"] = churn["contract"].str.lower()
churn["internet_service"] = churn["internet_service"].fillna("None")
churn["total_charges"] = churn["total_charges"].fillna(churn["monthly_charges"] * churn["tenure_months"])
for c in ["paperless_billing", "autopay"]:
    churn[c] = (churn[c] == "Yes").astype("int64")
X = pd.concat(
    [churn[["tenure_months", "monthly_charges", "total_charges", "support_calls", "paperless_billing", "autopay", "senior_citizen"]],
     pd.get_dummies(churn[["contract", "internet_service"]], dtype="float32")], axis=1).astype("float32")
y = churn["churned"].to_numpy("float32")

perm = np.random.default_rng(42).permutation(len(X))
tr, te = perm[:4000], perm[4000:]
mu, sd = X.iloc[tr].mean(), X.iloc[tr].std()
Xs = ((X - mu) / sd).to_numpy("float32")

train_ds = tf.data.Dataset.from_tensor_slices((Xs[tr], y[tr])).shuffle(4000, seed=42).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((Xs[te], y[te])).batch(256).prefetch(tf.data.AUTOTUNE)

# %% [markdown]
# Model, loss, optimizer and **metrics**. Keras metric objects keep running totals across batches (`update_state`),
# report with `result()` and are cleared with `reset_state()` at the start of every epoch.

# %%
def make_model():
    return keras.Sequential([
        keras.layers.Input((Xs.shape[1],)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),  # raw logit; the loss applies the sigmoid (more numerically stable)
    ])


model = make_model()
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.01)
train_loss, train_acc = keras.metrics.Mean(name="loss"), keras.metrics.BinaryAccuracy(name="acc", threshold=0.0)  # logits: 0 <-> p=0.5


@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        logits = model(xb, training=True)[:, 0]
        loss = loss_fn(yb, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply(grads, model.trainable_variables)
    train_loss.update_state(loss)
    train_acc.update_state(yb, logits)


# %%
for epoch in range(1, 6):
    train_loss.reset_state()
    train_acc.reset_state()
    for xb, yb in train_ds:
        train_step(xb, yb)
    print(f"epoch {epoch}: loss={train_loss.result():.4f} acc={train_acc.result():.4f}")
print("train_step traces so far:", train_step.experimental_get_tracing_count())

# %% [markdown]
# *What just happened?* The loop is `model.fit` with the lid off. Two details worth understanding:
# - `training=True` switches layers such as dropout on (they behave differently in training and inference).
# - `train_step` reports **3 traces**. The first call traces twice because the optimizer creates its own variables (Adam's
#   moment estimates) lazily during the first step, so TensorFlow traces again once they exist (a one-off cost that is normal). The
#   third trace is the final batch: 4000 % 128 = 32 rows, a different shape. Harmless here, but with many different
#   shapes it is the retracing trap from section 3. Fix it with `drop_remainder=True` or an `input_signature`.
#
# Now evaluate on held-out data, and compare with the majority-class baseline so the number means something.

# %%
@tf.function
def eval_step(xb, yb, acc_metric):
    logits = model(xb, training=False)[:, 0]
    acc_metric.update_state(yb, logits)


test_acc = keras.metrics.BinaryAccuracy(threshold=0.0)
for xb, yb in test_ds:
    eval_step(xb, yb, test_acc)
majority = max(y[te].mean(), 1 - y[te].mean())
print(f"test accuracy {test_acc.result():.3f} | always-predict-majority {majority:.3f}")

# %% [markdown]
# ### Customising the step: a weighted loss and gradient clipping
# Missing a churner is usually costlier than a false alarm. A custom loop makes it a few lines: weight the positive class and
# clip the global gradient norm for stability.

# %%
def weighted_bce(y_true, logits, pos_weight):
    # BCE with logits: -[ w*y*log(p) + (1-y)*log(1-p) ], via the stable TF helper
    per_example = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=logits, pos_weight=pos_weight)
    return tf.reduce_mean(per_example)


def train_weighted(pos_weight, epochs=5):
    keras.utils.set_random_seed(42)
    m = make_model()
    opt = keras.optimizers.Adam(0.01)

    @tf.function
    def step(xb, yb):
        with tf.GradientTape() as tape:
            loss = weighted_bce(yb, m(xb, training=True)[:, 0], pos_weight)
        grads = tape.gradient(loss, m.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)  # rescale if the total gradient norm exceeds 5
        opt.apply(grads, m.trainable_variables)

    for _ in range(epochs):
        for xb, yb in train_ds:
            step(xb, yb)
    p = m(Xs[te], training=False).numpy()[:, 0] > 0  # logit > 0  <=>  p > 0.5
    truth = y[te] == 1
    return (p & truth).sum() / truth.sum(), (p == truth).mean()


for pw in [1.0, 3.0]:
    recall, acc = train_weighted(pw)
    print(f"pos_weight={pw}: recall on churners={recall:.3f}, accuracy={acc:.3f}")

# %% [markdown]
# Up-weighting positives trades overall accuracy for recall of the rare-but-costly class. Whether that trade is right is a
# business question (what does a missed churner cost versus a wasted retention offer?), not a modelling one.

# %% [markdown]
# ## Common mistakes
#
# 1. **Python values as arguments** (`step(x, epoch)` with an `int`): retraces every new value. Pass tensors or keep them out of the function.
# 2. **Python side effects inside `tf.function`** (list `append`, `print`, incrementing a counter) run once at trace time. Use `tf.print` and `tf.Variable`.
# 3. **Creating `tf.Variable`s inside a `tf.function` on every call.** Variables may only be created on the first trace; build them outside (or in `__init__`).
# 4. **Forgetting `training=True`** when calling the model inside the tape, so dropout/batch-norm behave as at inference.
# 5. **Not resetting metrics each epoch**, so you report a running average over the whole run.
# 6. **`from_logits` mismatch**: a `Dense(1)` output with a loss that expects probabilities (or the reverse) trains badly with no error.
# 7. **Optimising a `tf.function` before measuring** it. For big matmuls or convolutions the gain is small.

# %%
# Demonstration of mistake 3
@tf.function
def bad_counter(x):
    v = tf.Variable(0.0)  # a new variable on every trace
    v.assign_add(x)
    return v


try:
    bad_counter(tf.constant(1.0))
    bad_counter(tf.constant([1.0, 2.0]))  # different shape -> retrace -> tries to create another variable
except ValueError as e:
    print("ValueError:", str(e).splitlines()[-1][:110], "...")

# %% [markdown]
# ## Interview corner
#
# **1. What does `tf.function` do and when does it retrace?**
# <details><summary>Answer</summary>
# It traces a Python function into a TensorFlow graph on first call and reuses that graph for later calls with the same input
# signature. It retraces for a new tensor shape/dtype, or for a new value of any Python-typed argument. Retraces are expensive,
# so pass tensors, use <code>input_signature</code>, or <code>reduce_retracing=True</code>.
# </details>
#
# **2. Why does `print` inside a `tf.function` only print once?**
# <details><summary>Answer</summary>
# Python code runs only while tracing. The graph contains TF ops, not your <code>print</code>. Use <code>tf.print</code>, which becomes a graph op.
# </details>
#
# **3. When would you write a custom training loop rather than <code>model.fit</code>?**
# <details><summary>Answer</summary>
# Non-standard losses or updates (multiple optimisers, GANs, RL, meta-learning), per-step gradient manipulation, custom
# logging or curriculum schedules, or research code. Otherwise prefer <code>fit</code> (callbacks, distribution and metrics come for free);
# in Keras 3 you can also override <code>train_step</code> and keep <code>fit</code>.
# </details>
#
# **4. What is <code>from_logits=True</code>?**
# <details><summary>Answer</summary>
# The model outputs raw scores and the loss applies sigmoid/softmax internally, in a fused, numerically stable form. It avoids
# saturated probabilities (log of 0) that cause NaNs or lost gradients.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Stop the retracing
# `rescale` below is called with batches of different lengths and a Python-float `alpha`. Write `rescale_fixed`: the same
# computation `alpha * x` as a `tf.function` with an `input_signature` (a float32 vector of any length, and a float32 scalar),
# so that after the calls below it has been traced **exactly once**. *Hint: `tf.TensorSpec(shape=[None], dtype=tf.float32)` and
# `tf.TensorSpec(shape=[], dtype=tf.float32)`; then pass `alpha` as a tensor (or a Python float, which is converted to match).*

# %%
@tf.function
def rescale(x, alpha):
    return alpha * x


calls = [(tf.ones(n), a) for n, a in [(3, 0.5), (4, 0.6), (5, 0.7)]]
for x_, a_ in calls:
    rescale(x_, a_)
print("original traces:", rescale.experimental_get_tracing_count())

# %% tags=["starter"]
rescale_fixed = ...

# %%
for x_, a_ in calls:
    out = rescale_fixed(x_, tf.constant(a_))
    assert np.allclose(out.numpy(), a_ * x_.numpy()), "the result should still be alpha * x"
assert rescale_fixed.experimental_get_tracing_count() == 1, f"expected 1 trace, got {rescale_fixed.experimental_get_tracing_count()}"
print("Exercise 1 passed")

# %% tags=["solution"]
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.float32)])
def rescale_fixed(x, alpha):
    return alpha * x


# %% [markdown]
# ### Exercise 2: Balanced accuracy from metric objects
# Accuracy hides the minority class. Using a trained network (`model` from section 4) compute on the test set
# `recall_pos` (recall of churners) and `recall_neg` (recall of non-churners), by looping over `test_ds` and updating two Keras
# metrics two `keras.metrics.Recall()` objects. The model outputs **logits**, so convert to probabilities with
# `tf.sigmoid` first (the default threshold is 0.5). Store `balanced = (recall_pos + recall_neg) / 2`.
# *Hint: negatives' recall is `Recall` computed on flipped labels: `1 - yb` and `1 - p`. Call `.result().numpy()` for a float.*

# %% tags=["starter"]
recall_pos = ...
recall_neg = ...
balanced = ...

# %%
p_te = tf.sigmoid(model(Xs[te], training=False)[:, 0]).numpy()
pred = p_te >= 0.5
truth = y[te] == 1
exp_pos = (pred & truth).sum() / truth.sum()
exp_neg = ((~pred) & (~truth)).sum() / (~truth).sum()
assert abs(recall_pos - exp_pos) < 1e-3, f"recall of churners should be about {exp_pos:.3f}, got {float(recall_pos):.3f}"
assert abs(recall_neg - exp_neg) < 1e-3, f"recall of non-churners should be about {exp_neg:.3f}, got {float(recall_neg):.3f}"
assert abs(balanced - (exp_pos + exp_neg) / 2) < 1e-3
print(f"Exercise 2 passed (recall+ {recall_pos:.3f}, recall- {recall_neg:.3f}, balanced {balanced:.3f})")

# %% tags=["solution"]
r_pos, r_neg = keras.metrics.Recall(), keras.metrics.Recall()
for xb, yb in test_ds:
    p = tf.sigmoid(model(xb, training=False)[:, 0])
    r_pos.update_state(yb, p)
    r_neg.update_state(1.0 - yb, 1.0 - p)  # flip roles: "positive" is now the non-churner
recall_pos = float(r_pos.result().numpy())
recall_neg = float(r_neg.result().numpy())
balanced = (recall_pos + recall_neg) / 2

# %% [markdown]
# ### Exercise 3: Your own loss
# Implement `huber(y_true, y_pred, delta=1.0)` in TensorFlow: mean over the batch of `0.5*e^2` when `|e| <= delta`, else
# `delta*(|e| - 0.5*delta)`, where `e = y_true - y_pred`. Huber loss is quadratic for small errors and linear for large ones, so
# outliers (like the absurd prices in `housing.csv`) do not dominate. The check compares with the hand-computed formula and
# confirms the gradient is finite. *Hint: `tf.where(tf.abs(e) <= delta, quadratic, linear)`.*

# %% tags=["starter"]
def huber(y_true, y_pred, delta=1.0):
    ...


# %%
yt = tf.constant([0.0, 0.0, 0.0, 0.0])
yp = tf.constant([0.5, -1.0, 3.0, -10.0])
e = np.abs(yt.numpy() - yp.numpy())
expected = np.where(e <= 1.0, 0.5 * e**2, 1.0 * (e - 0.5)).mean()
got = float(huber(yt, yp))
assert abs(got - expected) < 1e-5, f"expected {expected:.4f}, got {got:.4f}"
v = tf.Variable(yp)
with tf.GradientTape() as tape:
    l_ = huber(yt, v)
g = tape.gradient(l_, v)
assert g is not None and bool(tf.reduce_all(tf.math.is_finite(g))), "the gradient should exist and be finite"
assert float(tf.abs(g[3])) <= 0.25 + 1e-6, "for large errors the gradient magnitude is capped at delta / n"
print("Exercise 3 passed")

# %% tags=["solution"]
def huber(y_true, y_pred, delta=1.0):
    e = y_true - y_pred
    quadratic = 0.5 * tf.square(e)
    linear = delta * (tf.abs(e) - 0.5 * delta)
    return tf.reduce_mean(tf.where(tf.abs(e) <= delta, quadratic, linear))


# %% [markdown]
# ### Exercise 4 (open-ended): Retrace hunting
# Take `train_step` from section 4 and call it with a batch of size 128, then 100, then 77, then 50. Read
# `train_step.experimental_get_tracing_count()` before and after, then re-create it with an `input_signature` (or
# `reduce_retracing=True`) and compare. Write three sentences: what did TensorFlow do, why is the count what it is, and which
# of the two fixes would you use for a model whose batch size varies in production?

# %% [markdown]
# ## Recap
# - `tf.function` traces Python into a graph once per input signature; the graph then runs without Python overhead (fastest for many small ops; measure it).
# - Python side effects run only while tracing; use `tf.print` and `tf.Variable` for runtime behaviour.
# - Python-valued arguments and new tensor shapes cause retracing; pass tensors, use `input_signature` or `reduce_retracing=True`.
# - A custom loop = tape (forward + loss) -> `tape.gradient` -> `optimizer.apply`, with Keras metrics for bookkeeping (reset each epoch).
# - Customise the loss and the gradients freely, but compare against a simple baseline before believing a number.
#
# **Go deeper:** the TensorFlow guides "Better performance with tf.function" and "Writing a training loop from scratch" (Keras guide) at
# tensorflow.org/guide and keras.io.
#
# **Next:** [Saving, SavedModel and serving concepts](04_saving_savedmodel_and_serving_concepts.ipynb)
