# %% [markdown]
# # Indexing, Slicing and Boolean Masking (and the View-vs-Copy Trap)
#
# **Why this matters.** Selecting the right elements is 80% of array work: "all readings above a threshold",
# "every second frame", "the rows for class 3". The bug that bites software engineers most is
# **views vs copies**: modifying a slice can silently modify the original array. You will learn to predict which one you get and prove it.
#
# **You will learn**
# - index and slice 1-D and 2-D arrays, including negative indices and steps
# - filter with boolean masks and combine conditions correctly
# - select with integer arrays ("fancy indexing")
# - tell a view from a copy using `np.shares_memory`
#
# **Prerequisites:** [ndarray basics](01_ndarray_basics.ipynb).

# %%
import numpy as np

a = np.arange(10, 20)
a

# %% [markdown]
# ## 1. Basic indexing and slicing
#
# Slices are `start:stop:step`; `stop` is excluded; negatives count from the end. It is the same rule as Python lists.

# %%
print(a[0], a[-1])
print(a[2:5])
print(a[::3])
print(a[::-1])        # reversed

# %% [markdown]
# For 2-D arrays use **one bracket with a comma**: `m[row, col]`. Not `m[row][col]` (that works but builds an intermediate array).

# %%
m = np.arange(20).reshape(4, 5)
print(m)
print("element (1, 2):", m[1, 2])
print("row 2:", m[2])              # missing trailing index means ':'
print("column 3:", m[:, 3])
print("sub-block rows 1-2, cols 1-3:\n", m[1:3, 1:4])

# %% [markdown]
# *What just happened?* Each comma-separated position indexes one axis. `:` means "everything on this axis".
# An integer index **removes** that axis (`m[2]` is 1-D); a slice **keeps** it.

# %%
print(m[2].shape, m[2:3].shape, m[:, 3].shape, m[:, 3:4].shape)

# %% [markdown]
# ## 2. Views vs copies: the trap
#
# A basic **slice** returns a **view**: a new array object looking at the *same memory*. Writing through the view changes the original.
# (This is deliberate: slicing a 4 GB array must not copy 4 GB.)

# %%
base = np.arange(6)
view = base[1:4]
view[0] = 99
print("base after writing to the slice:", base)
print("shares memory?", np.shares_memory(base, view))

# %% [markdown]
# If you want independence, ask for it with `.copy()`:

# %%
base = np.arange(6)
safe = base[1:4].copy()
safe[0] = 99
print("base:", base, "| shares memory?", np.shares_memory(base, safe))

# %% [markdown]
# Rule of thumb, and let us verify each row of it rather than trust it:
#
# | Operation | Result |
# |---|---|
# | basic slice `a[1:4]`, `a[:, 0]` | view |
# | `reshape`, `.T`, `ravel()` (when possible) | view |
# | boolean mask `a[a > 3]` | **copy** |
# | fancy (integer-array) indexing `a[[0, 2]]` | **copy** |
# | `astype`, `.copy()`, `flatten()` | **copy** |

# %%
x = np.arange(12)
checks = {
    "slice x[2:8]": x[2:8],
    "reshape": x.reshape(3, 4),
    "transpose": x.reshape(3, 4).T,
    "ravel of a reshape": x.reshape(3, 4).ravel(),
    "mask x[x > 3]": x[x > 3],
    "fancy x[[0, 2]]": x[[0, 2]],
    "flatten": x.reshape(3, 4).flatten(),
    "astype(float)": x.astype(float),
}
for name, result in checks.items():
    print(f"{name:20s} shares memory: {np.shares_memory(x, result)}")

# %% [markdown]
# A subtle consequence: assigning *through* an index expression works, but assigning to a **copy** does nothing to the original.
# This is the array version of chained-assignment bugs you will meet again in pandas.

# %%
x = np.arange(6)
x[x > 3] = 0            # OK: assignment to a masked selection writes into x
print(x)

y = np.arange(6)
tmp = y[y > 3]          # a copy
tmp[:] = 0              # modifies only the copy
print(y)

# %% [markdown]
# ## 3. Boolean masks
#
# A comparison on an array gives a boolean array of the same shape. Use it as an index to keep the `True` positions.
# Combine masks with `&` (and), `|` (or), `~` (not), **with parentheses**. Python's `and` / `or` do not work on arrays.

# %%
temps = np.array([12.5, 31.0, 28.4, 35.2, 19.8, 40.1, 25.0])
hot = temps > 30
print(hot)
print(temps[hot])
print("count hot:", hot.sum(), "| fraction hot:", hot.mean().round(3))   # True counts as 1

# %%
print(temps[(temps > 20) & (temps < 36)])
print(temps[~hot])

try:
    temps[(temps > 20) and (temps < 36)]
except ValueError as e:
    print("ValueError:", str(e)[:70], "...")

# %% [markdown]
# `mask.sum()` counts matches and `mask.mean()` gives the fraction: the two most-used one-liners in data work.
# `np.any(mask)` and `np.all(mask)` answer "is there at least one?" and "is every one?".
#
# Masks also work per axis on 2-D arrays. Select the **rows** of `m` whose first column is a multiple of 10:

# %%
print(m[m[:, 0] % 10 == 0])

# %% [markdown]
# ## 4. Fancy indexing (integer arrays)
#
# Pass a list or array of positions to pick and reorder elements. The result is always a copy.

# %%
scores = np.array([70, 95, 82, 60, 88])
order = np.argsort(scores)[::-1]          # positions from best to worst
print(order, scores[order])
print("top 2 positions:", order[:2])

# %%
print(m[[0, 3]])                     # rows 0 and 3
print(m[[0, 3], [1, 4]])             # PAIRS: elements (0,1) and (3,4), not a 2x2 block
print(m[np.ix_([0, 3], [1, 4])])     # np.ix_ builds the 2x2 block

# %% [markdown]
# `np.nonzero(mask)` (or `np.where(mask)` with one argument) converts a mask into positions, and `argmax`/`argmin` give the position of an extreme.

# %%
print(np.nonzero(temps > 30)[0], temps.argmax())

# %% [markdown]
# ## Common mistakes
#
# 1. **Modifying a slice and being surprised the original changed.** Use `.copy()` when you want independence.
# 2. **Forgetting parentheses** in `(a > 1) & (a < 5)`.
# 3. **Expecting `m[[0, 3], [1, 4]]` to be a block.** It pairs indices; use `np.ix_` for the outer product.
# 4. **Assigning to a masked *copy*:** `tmp = a[a > 3]; tmp[:] = 0` leaves `a` untouched; write `a[a > 3] = 0`.
# 5. **Out-of-range integer index** raises `IndexError`, but an out-of-range *slice* silently clips (`a[5:100]` is fine).

# %%
print(a[5:100].shape)          # slices clip silently
try:
    a[100]
except IndexError as e:
    print("IndexError:", e)

# %% [markdown]
# ## Interview corner
#
# **1. Does `b = a[2:5]; b[0] = 0` change `a`? What about `b = a[a > 3]`?**
# <details><summary>Answer</summary>
# The slice is a view, so yes, <code>a</code> changes. Boolean-mask and fancy indexing return copies, so <code>a</code> does not change.
# Assigning directly, <code>a[a &gt; 3] = 0</code>, does write into <code>a</code> because the assignment happens on the original through the index.
# </details>
#
# **2. How would you prove whether two arrays share memory?**
# <details><summary>Answer</summary>
# <code>np.shares_memory(a, b)</code> (exact) or the cheaper <code>np.may_share_memory</code>. Checking <code>b.base is a</code> also hints at a view but is not a complete test.
# </details>
#
# **3. Select every row of a 2-D array where any value is negative.**
# <details><summary>Answer</summary>
# <code>arr[(arr &lt; 0).any(axis=1)]</code>: build the element mask, reduce it along the columns axis to one boolean per row, and use that as a row index.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Slice a table
# `table` is a `(6, 6)` array. Create `inner` (the 4x4 block with the outer border removed) and `every_other_col`
# (all rows, columns 0, 2, 4). Both should be **views** of `table`.

# %%
table = np.arange(36).reshape(6, 6)

# %% tags=["starter"]
inner = ...
every_other_col = ...

# %%
assert inner.shape == (4, 4) and inner[0, 0] == table[1, 1] and inner[-1, -1] == table[4, 4], "inner: drop the first/last row and column"
assert every_other_col.shape == (6, 3) and every_other_col[:, 1].tolist() == table[:, 2].tolist(), "columns 0, 2, 4"
assert np.shares_memory(table, inner) and np.shares_memory(table, every_other_col), "use basic slices so they are views"
print("Exercise 1 passed")

# %% tags=["solution"]
inner = table[1:-1, 1:-1]
every_other_col = table[:, ::2]

# %% [markdown]
# ### Exercise 2: Clip the outliers safely
# `readings` contains sensor values where anything above 100 is a glitch. Without changing `readings`, create `cleaned`
# (a copy where glitches are set to `np.nan`) and `n_glitches` (how many there were).
# *Hint: `astype(float)` first, then assign through a mask.*

# %%
readings = np.random.default_rng(42).integers(0, 120, size=50)
readings_before = readings.copy()

# %% tags=["starter"]
cleaned = ...
n_glitches = ...

# %%
assert np.array_equal(readings, readings_before), "do not modify the original `readings`"
assert n_glitches == int((readings > 100).sum()), "count values strictly above 100"
assert np.isnan(cleaned).sum() == n_glitches and np.array_equal(cleaned[~np.isnan(cleaned)], readings[readings <= 100]), "only glitches become NaN"
print(f"Exercise 2 passed ({n_glitches} glitches)")

# %% tags=["solution"]
cleaned = readings.astype(float)
cleaned[cleaned > 100] = np.nan
n_glitches = int((readings > 100).sum())

# %% [markdown]
# ### Exercise 3: Top-k without sorting the world
# `scores` holds 1,000 model scores. Find `top5_idx` (positions of the 5 largest, largest first) and `above_mean_frac`
# (the fraction of scores strictly above the mean).
# *Hint: `argsort` then slice, or `argpartition` if you want to be fancy.*

# %%
scores = np.random.default_rng(42).normal(0, 1, size=1000)

# %% tags=["starter"]
top5_idx = ...
above_mean_frac = ...

# %%
assert len(top5_idx) == 5 and np.all(np.diff(scores[top5_idx]) <= 0), "5 positions, best first"
assert set(top5_idx) == set(np.argsort(scores)[-5:]), "these are not the 5 largest"
assert np.isclose(above_mean_frac, np.mean([s > scores.mean() for s in scores])), "fraction strictly above the mean"
print("Exercise 3 passed")

# %% tags=["solution"]
top5_idx = np.argsort(scores)[::-1][:5]
above_mean_frac = (scores > scores.mean()).mean()

# %% [markdown]
# ### Exercise 4: Rows that satisfy two conditions
# In `points` (shape `(200, 2)`, columns are x and y) select the rows inside the unit circle whose x is positive.
# Store them in `inside_right` and store the fraction of **all** points that satisfy both in `frac`.

# %%
points = np.random.default_rng(42).uniform(-1.5, 1.5, size=(200, 2))

# %% tags=["starter"]
inside_right = ...
frac = ...

# %%
mask = (points[:, 0] ** 2 + points[:, 1] ** 2 < 1) & (points[:, 0] > 0)
assert np.array_equal(inside_right, points[mask]), "keep rows with x^2 + y^2 < 1 and x > 0"
assert np.isclose(frac, mask.sum() / len(points)), "frac is relative to all 200 points"
print(f"Exercise 4 passed ({len(inside_right)} points)")

# %% tags=["solution"]
mask = (points[:, 0] ** 2 + points[:, 1] ** 2 < 1) & (points[:, 0] > 0)
inside_right = points[mask]
frac = mask.mean()

# %% [markdown]
# ### Exercise 5 (open-ended): Predict, then verify
# Write three slicing/indexing expressions on an array of your choice for which you **predict** whether the result shares memory with the original.
# Verify each with `np.shares_memory` and note in a markdown cell any prediction you got wrong.

# %% [markdown]
# ## Recap
# - `a[start:stop:step]`, one bracket with commas for N-D; integer indices drop an axis, slices keep it.
# - Slices, reshape and transpose give **views**; masks, fancy indexing and `astype` give **copies**. Prove it with `np.shares_memory`.
# - Masks: `&`, `|`, `~` with parentheses; `.sum()` counts and `.mean()` gives a fraction.
# - `argsort`, `argmax`, `nonzero` turn values into positions.
#
# **Go deeper:** NumPy user guide, "Indexing on ndarrays" and "Copies and views" (numpy.org/doc/stable/user).
#
# **Next:** [Broadcasting and vectorization](03_broadcasting_and_vectorization.ipynb)
