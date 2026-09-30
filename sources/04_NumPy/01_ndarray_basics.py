# %% [markdown]
# # NumPy ndarray Basics: Creation, dtype, shape and axes
#
# **Why this matters.** Every data-science library you will use (pandas, scikit-learn, PyTorch, TensorFlow)
# stores numbers in something that behaves like a NumPy array. If you understand `shape`, `dtype` and `axis`,
# most "mysterious" shape-mismatch errors and slow loops disappear. Reading a stack trace that says
# `(32, 10) vs (10, 32)` becomes routine instead of scary.
#
# **You will learn**
# - create arrays from lists, ranges, constants and random numbers
# - read and control `dtype`, `shape`, `ndim`, `size`, and `reshape` safely
# - reason about `axis` so that `sum(axis=0)` versus `sum(axis=1)` is never a guess
# - stack, split and combine arrays
#
# **Prerequisites:** [Python basics](../01_Python/) (lists, loops, functions).
#
# > Everything here runs offline; the only random numbers come from a seeded generator.

# %%
import numpy as np

print("NumPy", np.__version__)

# %% [markdown]
# ## 1. Why an array, not a list?
#
# A Python list holds *pointers to Python objects*. A NumPy array holds **one contiguous block of raw
# numbers, all of the same type**. That is what makes it compact and lets NumPy run loops in C.
# Think of a list as a shelf of labelled boxes and an array as a single tray of identical coins.

# %%
py_list = [1, 2, 3, 4]
arr = np.array(py_list)

print(py_list * 2)   # list: repetition
print(arr * 2)       # array: element-wise arithmetic
print(type(arr), arr.dtype, arr.shape)

# %% [markdown]
# *What just happened?* `*` means "repeat" for lists but "multiply every element" for arrays. Operators on
# arrays are **element-wise**; this is called *vectorised* arithmetic and it is the core idea of the library.

# %% [markdown]
# ## 2. Creating arrays
#
# | You want | Write |
# |---|---|
# | from existing data | `np.array([...])` |
# | a range with a step | `np.arange(0, 10, 2)` |
# | N evenly spaced points, **both ends included** | `np.linspace(0, 1, 5)` |
# | zeros / ones / a constant | `np.zeros(shape)`, `np.ones(shape)`, `np.full(shape, 7)` |
# | identity matrix | `np.eye(3)` |
# | random numbers | `rng = np.random.default_rng(42); rng.random(shape)` |

# %%
print(np.arange(0, 10, 2))
print(np.linspace(0, 1, 5))
print(np.zeros((2, 3)))
print(np.full((2, 2), 7))
print(np.eye(3, dtype=int))

# %%
rng = np.random.default_rng(42)           # the modern, seeded generator: same numbers every run
print(rng.random(3))
print(rng.integers(low=0, high=10, size=(2, 4)))   # high is EXCLUSIVE

# %% [markdown]
# `arange` uses a step and excludes the end (like `range`); `linspace` uses a *count* and includes the end.
# `arange` with float steps can produce surprising lengths because of rounding, so prefer `linspace` for floats.

# %%
print(len(np.arange(0, 1, 0.1)), len(np.arange(0, 1.0000001, 0.1)))   # sensitive to the endpoint
print(len(np.linspace(0, 1, 11)))                                       # always exactly what you asked for

# %% [markdown]
# ## 3. `dtype`: the type of every element
#
# Every array has exactly one `dtype`. Common ones: `int64`, `float64`, `bool`, `float32`, `uint8`.
# If you mix ints and floats NumPy **upcasts** to the type that can hold everything.

# %%
print(np.array([1, 2, 3]).dtype)
print(np.array([1, 2, 3.5]).dtype)        # upcast to float64
print(np.array([1, True, 3]).dtype)       # bool upcast to int
print(np.array([1, "a"]).dtype)           # ...and to text if it has to (a unicode string dtype)

# %%
x = np.array([1.9, -1.9, 2.5])
print(x.astype(int))            # astype TRUNCATES toward zero; it does not round
print(np.round(x).astype(int))  # round first if you meant rounding (2.5 -> 2: round half to even)

# %% [markdown]
# `astype` always returns a **new** array (a copy). Smaller dtypes save memory: an image pixel needs
# `uint8` (0-255), not `float64`. The catch: small integer types **overflow silently**; we will see this in
# the pitfalls notebook.

# %%
big = np.arange(1_000_000)
print(big.dtype, big.nbytes / 1e6, "MB")
print(big.astype(np.float32).nbytes / 1e6, "MB as float32")

# %% [markdown]
# ## 4. Shape, ndim, size and reshape
#
# `shape` is a tuple with the length of each axis. `ndim` is `len(shape)`. `size` is the product of the shape.

# %%
a = np.arange(12)
print(a.shape, a.ndim, a.size)

m = a.reshape(3, 4)          # 3 rows, 4 columns
print(m)
print(m.shape, m.ndim, m.size)

# %% [markdown]
# `reshape` never changes the data or its order, it only changes how the same numbers are *read*. The new
# shape must multiply to the same `size`. Use `-1` for "figure this axis out for me":

# %%
print(a.reshape(2, -1).shape, a.reshape(-1, 6).shape)
print(m.T.shape)                 # transpose swaps the axes
print(m.ravel())                 # flatten to 1-D

try:
    a.reshape(5, 3)
except ValueError as e:
    print("ValueError:", e)

# %% [markdown]
# A shape like `(3,)` (1-D) is **not** the same as `(3, 1)` (a column) or `(1, 3)` (a row). This distinction
# causes most broadcasting surprises (see the broadcasting notebook). `np.newaxis` (or `None`) adds an axis:

# %%
v = np.array([10, 20, 30])
print(v.shape, v[:, np.newaxis].shape, v[np.newaxis, :].shape)

# %% [markdown]
# ## 5. Axis: the idea that trips everyone up
#
# For a 2-D array, `axis=0` runs **down the rows** (you collapse the rows, leaving one value per column) and
# `axis=1` runs **across the columns** (one value per row). Rule of thumb: **the axis you name is the one that disappears.**

# %%
sales = np.array([[10, 20, 30],     # store A: Mon, Tue, Wed
                  [ 1,  2,  3]])    # store B: Mon, Tue, Wed
print("shape:", sales.shape)
print("sum(axis=0):", sales.sum(axis=0), "-> per day, across stores")
print("sum(axis=1):", sales.sum(axis=1), "-> per store, across days")
print("sum():      ", sales.sum(), "-> everything")

# %% [markdown]
# *What just happened?* `axis=0` (length 2) vanished for `sum(axis=0)`, leaving shape `(3,)`.
# For `axis=1` the length-3 axis vanished, leaving `(2,)`. Check the output shape against this rule every time.
# It generalises to 3-D: a batch of images `(N, H, W)` averaged over `axis=0` gives the "mean image" of shape `(H, W)`.

# %%
images = np.random.default_rng(42).random((5, 4, 4))   # 5 tiny 4x4 "images"
print(images.mean(axis=0).shape, images.mean(axis=(1, 2)).shape)   # mean image; brightness of each image
print(sales.sum(axis=0, keepdims=True).shape)                     # keepdims keeps the axis with length 1

# %% [markdown]
# ## 6. Combining and splitting
#
# `np.concatenate` joins along an existing axis; `np.stack` creates a **new** axis. `vstack`/`hstack` are shortcuts.

# %%
r1, r2 = np.array([1, 2, 3]), np.array([4, 5, 6])
print(np.concatenate([r1, r2]).shape)          # (6,)   joined
print(np.stack([r1, r2]).shape)                # (2, 3) new axis 0
print(np.stack([r1, r2], axis=1).shape)        # (3, 2) new axis 1
print(np.vstack([r1, r2]).tolist(), np.hstack([r1, r2]).tolist())

# %%
parts = np.split(np.arange(10), [3, 7])        # cut before index 3 and 7
print([p.tolist() for p in parts])

# %% [markdown]
# ## Common mistakes
#
# 1. **Thinking `axis=0` means "rows".** It means "collapse along the first axis". Verify with `.shape`.
# 2. **`np.array([1, 2, 3], [4, 5, 6])`** (missing the outer brackets): the second positional argument is `dtype`. Write `np.array([[1, 2, 3], [4, 5, 6]])`.
# 3. **Ragged lists.** Arrays must be rectangular; different-length rows raise an error in current NumPy.
# 4. **Expecting `astype(int)` to round.** It truncates.
# 5. **Using the legacy global `np.random.seed`.** Prefer `default_rng(seed)`: it is local, reproducible and what current code uses.

# %%
try:
    np.array([[1, 2, 3], [4, 5]])
except ValueError as e:
    print("ValueError:", str(e)[:80], "...")

# %% [markdown]
# ## Interview corner
#
# **1. Why is a NumPy array faster than a Python list for numeric work?**
# <details><summary>Answer</summary>
# It stores homogeneous values in one contiguous typed buffer, so operations run as compiled loops
# without per-element Python object overhead, and memory access is cache-friendly. The cost is that all elements share one dtype.
# </details>
#
# **2. What does `axis=1` mean in `a.sum(axis=1)` for a 2-D array?**
# <details><summary>Answer</summary>
# The second axis is collapsed: you get one value per row (summing across columns). The result has the axis you named removed from the shape.
# </details>
#
# **3. What is the difference between `reshape`, `ravel` and `flatten`?**
# <details><summary>Answer</summary>
# <code>reshape</code> re-reads the same data with a new shape (a view when possible). <code>ravel</code> flattens to 1-D and returns a view when it can;
# <code>flatten</code> always copies.
# </details>
#
# **4. `np.arange(0, 1, 0.1)` vs `np.linspace(0, 1, 11)`: what is the difference?**
# <details><summary>Answer</summary>
# <code>arange</code> takes a step and excludes the stop value; <code>linspace</code> takes a count and includes the endpoint by default.
# With float steps, <code>arange</code>'s length can be off by one due to rounding, so <code>linspace</code> is safer for floats.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution.

# %% [markdown]
# ### Exercise 1: Build a grid
# Create `grid`: an integer array of shape `(4, 5)` containing the numbers `1..20` in row-major order
# (first row is 1 to 5). Then store its column sums in `col_sums` and its row means in `row_means`.
# *Hint: `arange(1, 21)` and `reshape`.*

# %% tags=["starter"]
grid = ...
col_sums = ...
row_means = ...

# %%
assert grid.shape == (4, 5), f"grid should be (4, 5), got {grid.shape}"
assert grid[0].tolist() == [1, 2, 3, 4, 5] and grid[-1, -1] == 20, "fill row by row with 1..20"
assert col_sums.shape == (5,) and np.array_equal(col_sums, sum(grid[i] for i in range(4))), "col_sums: one value per column"
assert row_means.shape == (4,) and np.allclose(row_means, [np.mean(list(r)) for r in grid]), "row_means: one value per row"
print("Exercise 1 passed")

# %% tags=["solution"]
grid = np.arange(1, 21).reshape(4, 5)
col_sums = grid.sum(axis=0)
row_means = grid.mean(axis=1)

# %% [markdown]
# ### Exercise 2: Fix the dtype
# Sensor readings arrive as the array `raw` below (in tenths of a degree, stored wastefully as `float64`).
# Create `temps_c`: a `float32` array in **degrees Celsius** (divide by 10), and `saved_bytes`: how many bytes
# you save by storing `temps_c` (float32) instead of `raw` (float64).

# %%
raw = np.random.default_rng(42).integers(-50, 400, size=10_000).astype(np.float64)

# %% tags=["starter"]
temps_c = ...
saved_bytes = ...

# %%
assert temps_c.dtype == np.float32, f"temps_c must be float32, got {temps_c.dtype}"
assert np.allclose(temps_c, raw / 10, atol=1e-4), "divide the raw values by 10"
assert saved_bytes == raw.nbytes - temps_c.nbytes, "saved_bytes = raw.nbytes - temps_c.nbytes"
print(f"Exercise 2 passed (saved {saved_bytes} bytes)")

# %% tags=["solution"]
temps_c = (raw / 10).astype(np.float32)
saved_bytes = raw.nbytes - temps_c.nbytes

# %% [markdown]
# ### Exercise 3: Which axis?
# `daily` has shape `(7, 24)`: 7 days of 24 hourly power readings. Compute
# `peak_hour_per_day` (shape `(7,)`: the hour index 0-23 of the maximum for each day, `argmax`) and
# `avg_by_hour` (shape `(24,)`: the average across the 7 days for each hour).
# *Hint: which axis should disappear in each case?*

# %%
rng = np.random.default_rng(42)
daily = rng.normal(200, 20, size=(7, 24)) + 40 * np.exp(-((np.arange(24) - 18) ** 2) / 8)   # evening peak

# %% tags=["starter"]
peak_hour_per_day = ...
avg_by_hour = ...

# %%
assert peak_hour_per_day.shape == (7,), "one hour index per day"
assert all(daily[i, peak_hour_per_day[i]] == daily[i].max() for i in range(7)), "argmax along the hours axis"
assert avg_by_hour.shape == (24,), "one average per hour"
assert np.allclose(avg_by_hour, [np.mean([daily[d, h] for d in range(7)]) for h in range(24)]), "average over days"
print("Exercise 3 passed")

# %% tags=["solution"]
peak_hour_per_day = daily.argmax(axis=1)
avg_by_hour = daily.mean(axis=0)

# %% [markdown]
# ### Exercise 4 (open-ended): Shapes in the wild
# Load the scikit-learn digits data with `from sklearn.datasets import load_digits; d = load_digits()`. Print `d.data.shape`,
# `d.images.shape` and `d.target.shape`. In a markdown cell, explain in your own words how `data` and `images` relate
# (hint: `reshape`) and what each axis means.

# %% [markdown]
# ## Recap
# - An array is a typed, contiguous block; operators are element-wise.
# - `shape`, `dtype`, `ndim`, `size` describe it; `reshape` re-reads the same data (use `-1` to infer).
# - The axis you pass to a reduction is the one that disappears from the shape.
# - `concatenate` joins along an existing axis, `stack` adds a new one.
# - Use `np.random.default_rng(seed)`.
#
# **Go deeper:** the NumPy user guide "NumPy: the absolute basics for beginners" and "Array creation" (numpy.org/doc/stable/user).
#
# **Next:** [Indexing, slicing and masking](02_indexing_slicing_masking.ipynb)
