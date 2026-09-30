# %% [markdown]
# # tf.data Input Pipelines
#
# **Why this matters.** On real projects the model is rarely the bottleneck: the data is. Reading files, decoding,
# shuffling and batching in plain Python leaves an expensive accelerator waiting. `tf.data` lets you describe an
# input pipeline declaratively (`map`, `shuffle`, `batch`, `prefetch`) and TensorFlow runs it efficiently, in
# parallel, overlapped with training. Even on a laptop CPU, knowing the order of these calls prevents subtle bugs
# such as batches that are never reshuffled.
#
# **You will learn**
# - build a `tf.data.Dataset` from arrays and from a dictionary of columns
# - chain `map`, `shuffle`, `batch`, `cache` and `prefetch`, and know why the order matters
# - measure honestly what `prefetch` and `num_parallel_calls` buy you (and when they do not)
# - write and read a small TFRecord file
#
# **Prerequisites:** [Tensors, Variables and GradientTape](01_tensors_variables_and_gradienttape.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic telecom customers, see `docs/DATASETS.md`) and scikit-learn's bundled
# > `digits` images (real handwritten digits, 8x8 pixels).

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_digits

tf.random.set_seed(42)
AUTOTUNE = tf.data.AUTOTUNE

# %% [markdown]
# ## 1. A Dataset is a lazy stream of elements
#
# **Intuition.** Think of a conveyor belt. `from_tensor_slices` puts your rows on the belt one at a time; each
# `map`/`batch`/`shuffle` is a station along the belt. Nothing moves until something *pulls* (a `for` loop or `model.fit`).

# %%
digits = load_digits()
imgs = digits.images.astype("float32")  # (1797, 8, 8), pixel values 0..16
labels = digits.target.astype("int32")
ds = tf.data.Dataset.from_tensor_slices((imgs, labels))
print(ds.element_spec)  # what ONE element looks like
for img, lab in ds.take(2):
    print(img.shape, int(lab))

# %% [markdown]
# *What just happened?* `from_tensor_slices` sliced along the first axis, so one element is an `(image, label)` pair.
# `element_spec` is the dataset's schema; look at it first whenever a pipeline shape error appears.
#
# Now the standard four stations. We scale pixels to 0..1 and add a channel axis (the CNN convention) in `map`.

# %%
def prep(img, lab):
    img = tf.expand_dims(img / 16.0, -1)  # (8, 8) -> (8, 8, 1), scaled to [0, 1]
    return img, lab


train_ds = (
    ds.shuffle(buffer_size=1797, seed=42)  # 1) mix rows (BEFORE batching)
    .map(prep)  # 2) per-element transform
    .batch(64)  # 3) group into batches
    .prefetch(AUTOTUNE)  # 4) prepare the next batch while the model works on the current one
)
xb, yb = next(iter(train_ds))
print(xb.shape, yb.shape, "min/max pixel:", float(tf.reduce_min(xb)), float(tf.reduce_max(xb)))
print("batches per epoch:", len(train_ds), "| 1797 / 64 =", 1797 / 64)

# %% [markdown]
# The last batch is smaller (1797 is not a multiple of 64). Use `.batch(64, drop_remainder=True)` when the model
# needs a fixed batch size (some compiled/TPU setups do).

# %% [markdown]
# ## 2. Order of stations matters
#
# | Station | Rule of thumb |
# |---|---|
# | `shuffle` | **before** `batch`. Shuffling after batching only reorders whole batches; the same rows stay together forever. |
# | `map` | before `batch` for per-row logic, after `batch` (vectorised) when the function is cheap and can handle a batch. |
# | `cache` | after expensive deterministic `map`s, **before** random ones (augmentation, shuffle) so randomness is not frozen. |
# | `prefetch` | last, always. |
#
# Let us prove the shuffle rule instead of taking it on trust. We tag each row with its index, run two epochs, and
# see whether the *composition* of the batches changes.

# %%
idx = tf.data.Dataset.range(12)


def batch_sets(pipeline, n_epochs=2):
    out = []
    for _ in range(n_epochs):
        out.append(sorted(tuple(sorted(b.numpy().tolist())) for b in pipeline))
    return out


good = idx.shuffle(12, seed=1).batch(4)  # shuffle rows, then batch
bad = idx.batch(4).shuffle(3, seed=1)  # batch, then shuffle the batches
g = batch_sets(good)
b_ = batch_sets(bad)
print("shuffle->batch: same batches every epoch?", g[0] == g[1])
print("batch->shuffle: same batches every epoch?", b_[0] == b_[1], b_[0])

# %% [markdown]
# With `batch` first, every epoch contains the same three groups `(0-3)`, `(4-7)`, `(8-11)`, only their order changes:
# the model sees correlated neighbours together every time. `shuffle` reshuffles every epoch by default
# (`reshuffle_each_iteration=True`), which is why the good pipeline differs between epochs.
#
# **`buffer_size`** is how many upcoming elements are held in memory to draw from. A buffer of 1 does no shuffling; a buffer
# at least as large as the dataset gives a perfect shuffle. For big data choose a large fraction of it, not the full size.

# %% [markdown]
# ## 3. Real-ish tabular data: churn as a dataset of (features, label)
#
# We build features with pandas (cleaning is a pandas job), then hand plain arrays to `tf.data`.

# %%
churn = pd.read_csv(Path("../datasets/churn.csv"))
churn["contract"] = churn["contract"].str.lower()  # fix the planted casing bug
churn["internet_service"] = churn["internet_service"].fillna("None")
churn["total_charges"] = churn["total_charges"].fillna(churn["monthly_charges"] * churn["tenure_months"])
for c in ["paperless_billing", "autopay"]:
    churn[c] = (churn[c] == "Yes").astype("int64")  # "Yes"/"No" text -> 1/0
num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
cat_cols = ["contract", "internet_service"]
X = pd.concat([churn[num_cols], churn[["paperless_billing", "autopay", "senior_citizen"]],
               pd.get_dummies(churn[cat_cols], dtype="float32")], axis=1).astype("float32")
y = churn["churned"].astype("float32")
print(X.shape, "positive rate:", round(float(y.mean()), 3))

# %%
n = len(X)
rng = np.random.default_rng(42)
perm = rng.permutation(n)
tr, te = perm[: int(0.8 * n)], perm[int(0.8 * n):]
mu, sd = X.iloc[tr].mean(), X.iloc[tr].std()  # statistics from TRAIN only (no leakage)
Xs = ((X - mu) / sd).to_numpy("float32")
ys = y.to_numpy("float32")

train = tf.data.Dataset.from_tensor_slices((Xs[tr], ys[tr])).shuffle(len(tr), seed=42).batch(128).prefetch(AUTOTUNE)
test = tf.data.Dataset.from_tensor_slices((Xs[te], ys[te])).batch(256).prefetch(AUTOTUNE)  # no shuffle for evaluation
print(train.element_spec)

# %% [markdown]
# A `Dataset` plugs straight into Keras (`model.fit(train)`). A single dense layer with a sigmoid is logistic
# regression; a couple of epochs is enough to see it beat the majority-class baseline. (Keras itself is the topic of the next module.)

# %%
import keras

keras.utils.set_random_seed(42)
model = keras.Sequential([keras.layers.Input((Xs.shape[1],)), keras.layers.Dense(1, activation="sigmoid")])
model.compile(optimizer=keras.optimizers.Adam(0.05), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, epochs=5, verbose=0)
loss, acc = model.evaluate(test, verbose=0)
baseline = max(ys[te].mean(), 1 - ys[te].mean())
print(f"test accuracy {acc:.3f}  vs  always-predict-majority {baseline:.3f}")

# %% [markdown]
# ## 4. Measuring: what do `prefetch` and `num_parallel_calls` really do?
#
# Claims about speed are only worth something if you measure. To make the effect visible on a tiny dataset we add
# a *deliberately slow* step to the producer (sleeping about 1 ms per element, simulating decoding a file or an image augmentation)
# and a slow consumer (a 60 ms "training step" per batch). `time.sleep` releases Python's lock, so parallel workers
# can genuinely overlap it.

# %%
def slow_decode(x):
    def _f(v):
        time.sleep(0.001)  # pretend this is expensive I/O or decoding
        return v

    return tf.py_function(_f, [x], tf.int64)


def run(pipeline, consume_ms=60):
    t0 = time.perf_counter()
    for _ in pipeline:
        time.sleep(consume_ms / 1000)  # pretend this is the training step
    return time.perf_counter() - t0


def no_defaults(dataset):
    """Switch off tf.data's automatic optimisations (incl. an automatically injected prefetch)."""
    opts = tf.data.Options()
    opts.experimental_optimization.apply_default_optimizations = False
    return dataset.with_options(opts)


base = tf.data.Dataset.range(320)
pipelines = {
    "sequential (defaults off)": no_defaults(base.map(slow_decode).batch(32)),
    "parallel map": no_defaults(base.map(slow_decode, num_parallel_calls=AUTOTUNE).batch(32)),
    "prefetch": no_defaults(base.map(slow_decode).batch(32).prefetch(AUTOTUNE)),
    "parallel + prefetch": no_defaults(base.map(slow_decode, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)),
    "sequential (TF defaults)": base.map(slow_decode).batch(32),
}
print(f"producer alone: {run(pipelines['sequential (defaults off)'], 0) * 1000:.0f} ms | consumer alone: {10 * 60} ms   (10 batches x 60 ms)")
run(pipelines["parallel + prefetch"])  # warm-up so the first timing does not pay start-up costs
times = {name: run(p) for name, p in pipelines.items()}
ref = times["sequential (defaults off)"]
for k, v in times.items():
    print(f"{k:26s} {v * 1000:7.0f} ms   ({ref / v:.2f}x vs sequential)")

# %% [markdown]
# *What just happened?* With no overlap, each epoch costs *producer time + consumer time*. `prefetch` runs the
# producer in the background while the consumer works, so the total moves towards *max(producer, consumer)*;
# `num_parallel_calls` shrinks the producer time itself. Your exact numbers depend on the machine, so read the ratios
# rather than the milliseconds, and expect some run-to-run noise.
#
# One surprise worth noticing: the last row, plain `map(...).batch(...)` with **TensorFlow's default settings**, is usually about as
# fast as the hand-tuned pipeline. `tf.data` applies automatic optimisations (it can inject a `prefetch` at the end of the
# pipeline and parallelise some maps by itself), so we switched them off to see the effect of each station. Still write
# `prefetch` explicitly: it documents intent and does not depend on optimiser heuristics.
#
# The **honest caveat**: when the per-element work is tiny (as in the digits `prep` above), a parallel `map` has scheduling
# overhead and may be no faster or even slower. Profile before you tune:

# %%
def timed_epochs(pipeline, epochs=3):
    t0 = time.perf_counter()
    for _ in range(epochs):
        for _ in pipeline:
            pass
    return (time.perf_counter() - t0) / epochs


cheap_seq = ds.map(prep).batch(64)
cheap_par = ds.map(prep, num_parallel_calls=AUTOTUNE).batch(64)
cheap_cache = ds.map(prep).cache().batch(64)
timed_epochs(cheap_cache, 1)  # first pass fills the cache
for name, p in {"cheap map": cheap_seq, "cheap parallel map": cheap_par, "cheap map + cache": cheap_cache}.items():
    print(f"{name:20s} {timed_epochs(p) * 1000:6.1f} ms / epoch")

# %% [markdown]
# For 1,797 tiny images the differences are small and can go either way. `cache()` helps most when the `map` is
# expensive and the data fits in memory (it stores post-`map` results after the first epoch).

# %% [markdown]
# ## 5. TFRecord: the on-disk format for big datasets
#
# When data does not fit in memory you read it from files. TFRecord is TensorFlow's binary record format: each record
# is a serialised `tf.train.Example` (a dict of typed lists). Here is a full round trip in a temporary directory.

# %%
def to_example(vec, label):
    feature = {
        "x": tf.train.Feature(float_list=tf.train.FloatList(value=vec.tolist())),
        "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


spec = {"x": tf.io.FixedLenFeature([Xs.shape[1]], tf.float32), "y": tf.io.FixedLenFeature([], tf.int64)}
with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp) / "churn_sample.tfrecord")
    with tf.io.TFRecordWriter(path) as writer:
        for row, lab in zip(Xs[:100], ys[:100]):
            writer.write(to_example(row, lab))
    rec = tf.data.TFRecordDataset(path).map(lambda s: tf.io.parse_single_example(s, spec)).batch(25)
    parsed = list(rec)  # materialise before the temp dir disappears
print(len(parsed), "batches; x shape", parsed[0]["x"].shape, "| first labels", parsed[0]["y"].numpy()[:5])
print("round trip exact?", np.allclose(parsed[0]["x"].numpy(), Xs[:25]))

# %% [markdown]
# ## Common mistakes
#
# 1. **`batch` before `shuffle`** (proved above): batches never change composition.
# 2. **Shuffling the validation/test set** or fitting normalisation statistics on all data. Shuffle only training data; compute
#    statistics on the training split (we did: `mu, sd`).
# 3. **Random augmentation placed before `cache()`**: the first epoch's random result is cached and replayed forever.
# 4. **`prefetch` not last**, or forgetting it entirely.
# 5. **Tuning blindly**: `num_parallel_calls` and `cache` are not free. Time an epoch before and after.
# 6. **Using `py_function` in production pipelines** for convenience: it holds Python's GIL and cannot be serialised into
#    a fast graph. Prefer native `tf.*` ops (we used it here only to simulate slowness).

# %% [markdown]
# ## Interview corner
#
# **1. Why call `prefetch` and where does it go in the pipeline?**
# <details><summary>Answer</summary>
# It overlaps data preparation for step N+1 with model execution for step N, hiding input latency. Put it last so it buffers
# fully prepared batches. <code>prefetch(tf.data.AUTOTUNE)</code> lets TensorFlow choose the buffer size.
# </details>
#
# **2. What does <code>shuffle(buffer_size)</code> do, and what happens if the buffer is small?**
# <details><summary>Answer</summary>
# It keeps a buffer of upcoming elements and emits a random one, refilling from the source. A buffer smaller than the dataset gives
# only a local shuffle, so if the file is sorted (say by label or date) batches stay biased. Use a big buffer or shuffle the source files too.
# </details>
#
# **3. When would you use <code>cache()</code>?**
# <details><summary>Answer</summary>
# When an expensive deterministic transformation (decode, resize, parse) is repeated every epoch and the result fits in memory or on
# local disk. Place it before random augmentation and after the deterministic work.
# </details>
#
# **4. Why use tf.data instead of a Python generator or NumPy slicing?**
# <details><summary>Answer</summary>
# The pipeline runs in TensorFlow's C++ runtime, can parallelise and prefetch outside the GIL, streams data larger than memory,
# and is the same code path on CPU, GPU, TPU and distributed setups. For small in-memory data NumPy slicing is fine.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A correct training pipeline
# Build `pipe` from `Xs[tr]`, `ys[tr]` (the churn training arrays above): shuffle with a buffer equal to the number of rows
# (`seed=42`), **then** batch with size 100, drop the incomplete last batch, and prefetch with `AUTOTUNE`.
# *Hint: `from_tensor_slices((a, b))` then chain the calls; `batch(..., drop_remainder=True)`.*

# %% tags=["starter"]
pipe = ...

# %%
n_batches = sum(1 for _ in pipe)
assert n_batches == len(tr) // 100, f"expected {len(tr) // 100} full batches, got {n_batches} (drop_remainder=True?)"
xb, yb = next(iter(pipe))
assert xb.shape == (100, Xs.shape[1]) and yb.shape == (100,), f"unexpected batch shapes {xb.shape}, {yb.shape}"
e1 = [b[1].numpy().tolist() for b in pipe]
e2 = [b[1].numpy().tolist() for b in pipe]
assert e1 != e2, "the data should be reshuffled every epoch: shuffle BEFORE batch"
print("Exercise 1 passed")

# %% tags=["solution"]
pipe = (
    tf.data.Dataset.from_tensor_slices((Xs[tr], ys[tr]))
    .shuffle(len(tr), seed=42)
    .batch(100, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

# %% [markdown]
# ### Exercise 2: Digits pipeline with a `map`
# Build `digit_ds` yielding batches of 32 from `imgs`/`labels` where each image is standardised **per image**:
# subtract that image's own mean and divide by its own standard deviation plus `1e-6` (shape stays `(8, 8)`).
# No shuffle. *Hint: inside `map`, `tf.reduce_mean(img)` and `tf.math.reduce_std(img)`; batch after the map.*

# %% tags=["starter"]
digit_ds = ...

# %%
xb, yb = next(iter(digit_ds))
assert xb.shape == (32, 8, 8), f"expected (32, 8, 8), got {xb.shape}"
means = tf.reduce_mean(xb, axis=[1, 2]).numpy()
assert np.allclose(means, 0, atol=1e-4), "every image should have mean 0"
img0 = imgs[0]
expect0 = (img0 - img0.mean()) / (img0.std() + 1e-6)
assert np.allclose(xb[0].numpy(), expect0, atol=1e-3), "the first image should match (img - mean) / (std + 1e-6)"
assert np.array_equal(yb.numpy(), labels[:32]), "do not shuffle: labels should follow the original order"
print("Exercise 2 passed")

# %% tags=["solution"]
def standardise(img, lab):
    return (img - tf.reduce_mean(img)) / (tf.math.reduce_std(img) + 1e-6), lab


digit_ds = tf.data.Dataset.from_tensor_slices((imgs, labels)).map(standardise).batch(32)

# %% [markdown]
# ### Exercise 3: Sliding windows over a sensor stream
# `tf.data` is also handy for sequences. From `datasets/sensors.csv` take the `temperature_c` column (fill missing values by
# forward-fill, then back-fill). Build `windows`: batches of size 8 whose elements are **overlapping windows of 24 consecutive
# hourly readings** (stride 1, no shuffle), from the first 200 readings only. Store the number of windows in `n_windows`.
# *Hint: `Dataset.from_tensor_slices(series).window(24, shift=1, drop_remainder=True)` gives nested datasets; flatten each with
# `.flat_map(lambda w: w.batch(24))`. Expected windows: 200 - 24 + 1.*

# %%
sensors = pd.read_csv(Path("../datasets/sensors.csv"))
temp = sensors["temperature_c"].ffill().bfill().to_numpy("float32")[:200]
print(temp.shape, "missing left:", int(np.isnan(temp).sum()))

# %% tags=["starter"]
windows = ...
n_windows = ...

# %%
first = next(iter(windows))
assert first.shape == (8, 24), f"batches should be (8, 24), got {first.shape}"
assert np.allclose(first[0].numpy(), temp[:24]) and np.allclose(first[1].numpy(), temp[1:25]), "windows must slide by 1 hour"
assert n_windows == 200 - 24 + 1, f"expected {200 - 24 + 1} windows, got {n_windows}"
print("Exercise 3 passed")

# %% tags=["solution"]
windows = (
    tf.data.Dataset.from_tensor_slices(temp)
    .window(24, shift=1, drop_remainder=True)
    .flat_map(lambda w: w.batch(24))
    .batch(8)
)
n_windows = sum(int(b.shape[0]) for b in windows)

# %% [markdown]
# ### Exercise 4 (open-ended): Does `cache()` pay off here?
# Make the digits `map` artificially expensive (for example call `tf.linalg.matmul` on a 200x200 random matrix inside `prep`),
# time an epoch with and without `.cache()` using `timed_epochs`, and write two sentences: what changed, and at what point
# would you stop worrying about input speed on your own project?

# %% [markdown]
# ## Recap
# - A `Dataset` is a lazy stream; `element_spec` is its schema. Build from arrays, files (TFRecord) or generators.
# - Standard order: `shuffle` (train only) -> `map` -> `batch` -> `prefetch(AUTOTUNE)`; `cache` after deterministic, before random work.
# - Datasets plug directly into `model.fit`; compute normalisation statistics on the training split only.
# - Parallelism and caching are measurements, not magic: they help when per-element work is heavy, and can cost more than they save when it is tiny.
#
# **Go deeper:** the TensorFlow guides "tf.data: Build TensorFlow input pipelines" and "Better performance with the tf.data API"
# at tensorflow.org/guide.
#
# **Next:** [Custom training loop and tf.function](03_custom_training_loop_and_tf_function.ipynb)
