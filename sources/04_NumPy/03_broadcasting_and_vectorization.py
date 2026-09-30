# %% [markdown]
# # Broadcasting and Vectorization
#
# **Why this matters.** "Replace the loop with array operations" is the single most common performance advice in
# data science, and *broadcasting* is the mechanism that makes it possible for arrays of different shapes.
# In a job you will standardise features, compute pairwise distances and apply per-column scaling; all are broadcasting.
#
# **You will learn**
# - state the two broadcasting rules and predict the result shape before running code
# - standardise a matrix and compute pairwise distances without loops
# - benchmark a Python loop against vectorised code **fairly**
# - recognise when vectorising is *not* the answer
#
# **Prerequisites:** [Indexing and masking](02_indexing_slicing_masking.ipynb).

# %%
import time

import numpy as np

# %% [markdown]
# ## 1. Broadcasting: making shapes compatible without copying
#
# Adding a scalar to an array is the simplest broadcast: the scalar is "stretched" to the array's shape.
# The general rule compares shapes **from the right**. Two dimensions are compatible if they are **equal or one of them is 1**;
# a missing dimension counts as 1. The result takes the larger size on each axis.

# %%
m = np.arange(6).reshape(2, 3)          # shape (2, 3)
row = np.array([10, 20, 30])            # shape (3,)   -> treated as (1, 3)
col = np.array([[100], [200]])          # shape (2, 1)

print((m + row).shape, "\n", m + row)
print((m + col).shape, "\n", m + col)

# %% [markdown]
# *What just happened?* `row` was reused for every row; `col` was reused for every column. No copy is made: NumPy just
# uses stride zero along the stretched axis. When the rule fails you get an explicit error, which is good:

# %%
try:
    m + np.array([1, 2])      # (2, 3) vs (2,): compares 3 with 2 from the right
except ValueError as e:
    print("ValueError:", e)

# %% [markdown]
# A helpful mental exercise: line the shapes up right-aligned and check every column.
#
# ```
# (2, 3)        (5, 1)        (5, 3)
# (   3)        (   4)        (2, 1)
# -----         -----         -----
# (2, 3) ok     (5, 4) ok     error: 5 vs 2 on axis 0
# ```

# %%
print((np.zeros((5, 1)) + np.zeros(4)).shape)     # (5, 1) with (4,) -> (5, 4): an outer-sum table

# %% [markdown]
# ## 2. Standardising features (z-scores)
#
# A machine-learning classic: subtract each column's mean and divide by its standard deviation. `X` is `(n_samples, n_features)`.
# The column means have shape `(n_features,)`, which broadcasts over the rows automatically.

# %%
rng = np.random.default_rng(42)
X = rng.normal(loc=[50, 0, 1000], scale=[5, 1, 200], size=(500, 3))   # 3 features on very different scales

mu = X.mean(axis=0)          # (3,)
sd = X.std(axis=0)           # (3,)
Z = (X - mu) / sd            # (500, 3) op (3,) -> (500, 3)

print("all column means ~ 0:", bool(np.abs(Z.mean(axis=0)).max() < 1e-12))
print("column stds:", Z.std(axis=0).round(6))

# %% [markdown]
# For row-wise operations you must keep the reduced axis with `keepdims=True` (or `[:, None]`), otherwise the shapes line up the wrong way.

# %%
row_sums = X.sum(axis=1, keepdims=True)         # (500, 1)
shares = X / row_sums                           # (500, 3) / (500, 1) -> works
print(shares.sum(axis=1)[:3])

try:
    X / X.sum(axis=1)                           # (500, 3) / (500,) : compares 3 with 500
except ValueError as e:
    print("ValueError:", e)

# %% [markdown]
# ## 3. Pairwise distances with no loops
#
# Given `P` of shape `(n, 2)`, the distance between every pair of points is a broadcast of `(n, 1, 2)` against `(1, n, 2)`.
# The intermediate array has shape `(n, n, 2)`, so this trades memory for speed; fine for thousands of points, not millions.

# %%
P = rng.random((5, 2))
diff = P[:, None, :] - P[None, :, :]            # (5, 1, 2) - (1, 5, 2) -> (5, 5, 2)
D = np.sqrt((diff ** 2).sum(axis=-1))           # (5, 5)
print(D.shape, "diagonal (distance to self):", np.diag(D))
print("symmetric:", np.allclose(D, D.T))

# cross-check one entry with plain Python
i, j = 1, 3
print(np.isclose(D[i, j], ((P[i, 0] - P[j, 0]) ** 2 + (P[i, 1] - P[j, 1]) ** 2) ** 0.5))

# %% [markdown]
# ## 4. Vectorization: a *fair* benchmark
#
# Claims like "NumPy is 100x faster" are only meaningful if the comparison is fair. Ours will:
# - compute **the same result** (we assert it),
# - build inputs **outside** the timed section,
# - take the **best of several repeats** (the minimum is the least noisy estimate of the true cost),
# - compare against a *reasonable* pure-Python loop, not a deliberately silly one.
#
# Task: sum of squares of one million numbers.

# %%
def best_of(fn, repeats=5):
    """Return the fastest wall-clock time of `repeats` calls, in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


data = np.random.default_rng(42).random(1_000_000)
data_list = data.tolist()                      # the fair input for the pure-Python version

def loop_version():
    total = 0.0
    for v in data_list:
        total += v * v
    return total

def numpy_version():
    return float(np.dot(data, data))

assert np.isclose(loop_version(), numpy_version()), "the two versions must agree"

t_loop = best_of(loop_version, repeats=3)
t_np = best_of(numpy_version, repeats=5)
print(f"python loop: {t_loop * 1000:8.2f} ms")
print(f"numpy      : {t_np * 1000:8.2f} ms")
print(f"speedup    : about {t_loop / t_np:.0f}x   (machine dependent; run it again)")

# %% [markdown]
# Two honest caveats. **(1)** The speedup differs by machine, array size and operation, so do not memorise a number.
# **(2)** If the timed NumPy version must first convert a Python list into an array, the advantage can vanish:
# the conversion itself walks every Python object. Vectorisation pays off when data *lives* in arrays.

# %%
def numpy_with_conversion():
    arr = np.array(data_list)
    return float((arr * arr).sum())

t_conv = best_of(numpy_with_conversion, repeats=3)
print(f"numpy incl. list->array conversion: {t_conv * 1000:.2f} ms  (vs {t_np * 1000:.2f} ms when data already is an array, and {t_loop * 1000:.2f} ms for the plain loop)")

# %% [markdown]
# ## 5. Ufuncs, `where` and when a loop is fine
#
# Functions like `np.sqrt`, `np.exp`, `np.maximum`, `np.abs` are *universal functions* (ufuncs): compiled, element-wise, broadcasting-aware.
# `np.where(cond, a, b)` is the vectorised `if/else`.

# %%
x = np.linspace(-3, 3, 7)
print(x)
print(np.where(x > 0, x, 0.0))        # ReLU, the activation function from neural nets
print(np.maximum(x, 0.0))             # the same thing as a single ufunc

# %% [markdown]
# **When not to vectorise.** If each step depends on the previous result (a recurrence such as `x[t] = 0.9 * x[t-1] + noise[t]`),
# you cannot express it as a single array operation. Options: keep the loop, use `scipy.signal.lfilter`, or use a compiled tool
# such as Numba. Also, huge intermediate arrays (like the `(n, n, 2)` above) can exhaust memory; sometimes chunking beats "pure" vectorisation.

# %% [markdown]
# ## Common mistakes
#
# 1. **Relying on a silent wrong broadcast.** A `(n,)` array combined with an `(n, 1)` array gives `(n, n)`, not an error. Always print shapes when in doubt.
# 2. **Reducing without `keepdims`** and then dividing: `(500, 3) / (500,)` fails, or worse, works for a square matrix but divides along the wrong axis.
# 3. **Benchmarking once, or including setup in the timing.** Use repeats and time only the work.
# 4. **Vectorising with `np.vectorize`.** It is a convenience wrapper that still loops in Python; it is not faster.
# 5. **Comparing with a strawman loop** and then trusting the headline number.

# %%
sq = np.arange(3)
print((sq[:, None] + sq).shape)          # (3, 1) + (3,) -> (3, 3): probably not what you wanted for two vectors
square = np.arange(9).reshape(3, 3)
print(square / square.sum(axis=1))       # runs without error but divides along the WRONG axis
print(square / square.sum(axis=1, keepdims=True))   # correct: each row divided by its own sum

# %% [markdown]
# ## Interview corner
#
# **1. State the broadcasting rules.**
# <details><summary>Answer</summary>
# Align shapes from the trailing axis. Missing leading axes count as size 1. On each axis the sizes must be equal or one of them must be 1;
# the result takes the larger size. Otherwise NumPy raises a <code>ValueError</code>.
# </details>
#
# **2. What is the shape of `A[:, None] * B[None, :]` for `A` of shape `(m,)` and `B` of shape `(n,)`?**
# <details><summary>Answer</summary>
# <code>(m, n)</code>: the outer product. This is the standard trick for building all-pairs tables.
# </details>
#
# **3. Why is vectorised code faster, and when might it not help?**
# <details><summary>Answer</summary>
# The loop runs in compiled code over a typed contiguous buffer, avoiding interpreter overhead per element. It does not help for inherently sequential
# recurrences, for tiny arrays (call overhead dominates), or when the intermediate arrays become too large for memory.
# </details>
#
# **4. Is `np.vectorize` a vectorisation?**
# <details><summary>Answer</summary>
# No; its documentation says it is provided for convenience, not performance. It is essentially a Python <code>for</code> loop.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Predict the shape
# For each pair below, store the resulting shape in the dictionary `shapes` (as a tuple), or the string `"error"` if broadcasting fails.
# Keys: `"a"` for `(4, 1) + (3,)`, `"b"` for `(2, 3) + (2,)`, `"c"` for `(5, 1, 3) * (4, 1)`.
# Work it out on paper first, then the check will run the real operation.

# %% tags=["starter"]
shapes = {"a": ..., "b": ..., "c": ...}

# %%
def real_shape(s1, s2):
    try:
        return (np.zeros(s1) + np.zeros(s2)).shape
    except ValueError:
        return "error"

assert shapes["a"] == real_shape((4, 1), (3,)), f"a: got {shapes['a']}"
assert shapes["b"] == real_shape((2, 3), (2,)), f"b: got {shapes['b']}"
assert shapes["c"] == real_shape((5, 1, 3), (4, 1)), f"c: got {shapes['c']}"
print("Exercise 1 passed")

# %% tags=["solution"]
shapes = {"a": (4, 3), "b": "error", "c": (5, 4, 3)}

# %% [markdown]
# ### Exercise 2: Min-max scale each column
# Scale every column of `X` (defined above, shape `(500, 3)`) to the range 0-1 with `(x - min) / (max - min)` per column.
# Store the result in `X01`. No loops.

# %% tags=["starter"]
X01 = ...

# %%
assert X01.shape == X.shape, "same shape as X"
assert np.allclose(X01.min(axis=0), 0) and np.allclose(X01.max(axis=0), 1), "every column should span exactly 0 to 1"
j = 1
assert np.isclose(X01[7, j], (X[7, j] - X[:, j].min()) / (X[:, j].max() - X[:, j].min())), "check the formula for one element"
print("Exercise 2 passed")

# %% tags=["solution"]
X01 = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# %% [markdown]
# ### Exercise 3: Nearest neighbour without a loop
# `pts` has 300 random 2-D points and `q` is a single query point of shape `(2,)`. Find `nearest_idx`: the index of the point closest
# to `q` (Euclidean), and `nearest_dist`: that distance.

# %%
pts = np.random.default_rng(42).random((300, 2))
q = np.array([0.5, 0.5])

# %% tags=["starter"]
nearest_idx = ...
nearest_dist = ...

# %%
brute = min(range(len(pts)), key=lambda i: ((pts[i, 0] - q[0]) ** 2 + (pts[i, 1] - q[1]) ** 2))
assert nearest_idx == brute, f"expected index {brute}"
assert np.isclose(nearest_dist, np.hypot(*(pts[brute] - q))), "distance to the nearest point"
print("Exercise 3 passed")

# %% tags=["solution"]
d = np.sqrt(((pts - q) ** 2).sum(axis=1))
nearest_idx = int(d.argmin())
nearest_dist = float(d[nearest_idx])

# %% [markdown]
# ### Exercise 4: Multiplication table and a fair timing
# (a) Build `table`: the 12x12 multiplication table (`table[i, j] == (i+1) * (j+1)`) using broadcasting only.
# (b) Store in `ratio` how many times slower a nested Python loop is than your NumPy version (`t_loop_table / t_np_table`),
# using the provided `best_of` helper. The check only requires `ratio` to be a positive number, since timing is machine dependent.

# %% tags=["starter"]
table = ...
ratio = ...

# %%
assert table.shape == (12, 12) and table[0, 0] == 1 and table[11, 11] == 144 and table[2, 4] == 15, "table[i, j] == (i+1)*(j+1)"
assert np.array_equal(table, np.array([[(i + 1) * (j + 1) for j in range(12)] for i in range(12)])), "compare with the loop version"
assert ratio > 0, "ratio must be a positive number"
print(f"Exercise 4 passed (loop was {ratio:.1f}x the NumPy time; tiny arrays show smaller gains than big ones)")

# %% tags=["solution"]
def make_np():
    return np.arange(1, 13)[:, None] * np.arange(1, 13)

def make_loop():
    return [[(i + 1) * (j + 1) for j in range(12)] for i in range(12)]

table = make_np()
ratio = best_of(make_loop, 20) / best_of(make_np, 20)

# %% [markdown]
# ### Exercise 5 (open-ended): Find the crossover
# Time the sum of squares (loop vs `np.dot`) for input sizes 10, 100, 10_000 and 1_000_000. At what size does NumPy start to win on your
# machine? Write two sentences in a markdown cell explaining why small arrays behave differently.

# %% [markdown]
# ## Recap
# - Broadcasting aligns shapes from the right; sizes must match or be 1; missing axes count as 1.
# - Use `keepdims=True` (or `[:, None]`) when a reduction result must broadcast back against the original.
# - Vectorised code moves the loop into compiled code; benchmark fairly (same result, setup excluded, best of N).
# - Some problems (recurrences, giant intermediates) do not vectorise well; that is fine.
#
# **Go deeper:** NumPy user guide "Broadcasting", and the `np.vectorize` docs note on performance (numpy.org/doc/stable).
#
# **Next:** [Linear algebra, random numbers and statistics](04_linear_algebra_random_and_statistics.ipynb)
