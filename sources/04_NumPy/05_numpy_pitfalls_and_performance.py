# %% [markdown]
# # NumPy Pitfalls and Performance
#
# **Why this matters.** NumPy bugs are rarely crashes. They are silent: an image whose bright pixels wrap to black,
# a cumulative total that drifts, a mean that is `nan` because one sensor dropped out. This notebook collects the
# failure modes that show up in real pipelines, each demonstrated so you recognise them in the wild.
#
# **You will learn**
# - predict and avoid integer overflow and unwanted upcasting
# - understand floating-point error and compare floats correctly
# - handle `NaN` with `nan*` functions and masks
# - use in-place operations, `out=`, and memory estimates sensibly
# - see what `strides` are and use `sliding_window_view` safely
#
# **Prerequisites:** [Linear algebra, random numbers and statistics](04_linear_algebra_random_and_statistics.ipynb).

# %%
import warnings

import numpy as np

# %% [markdown]
# ## 1. Integer overflow: fixed-width integers wrap around
#
# `uint8` holds 0-255. Arithmetic on **arrays** wraps around silently (modular arithmetic). This is how bright image pixels turn dark.

# %%
info = np.iinfo(np.uint8)
print("uint8 range:", info.min, "to", info.max)

pixels = np.array([100, 200, 250], dtype=np.uint8)
print("pixels + 10:", pixels + 10)                 # 250 + 10 = 260 wraps to 4
print("dtype stays:", (pixels + 10).dtype)

# %% [markdown]
# *What just happened?* `260 mod 256 = 4`. No error, no warning. Note that NumPy 2 **does** refuse a Python literal that cannot fit
# in the array's dtype (`pixels + 300` raises), but it will not protect you when the *result* of a valid operation overflows.
# Scalars are a bit friendlier and emit a `RuntimeWarning`:

# %%
try:
    pixels + 300
except OverflowError as e:
    print("OverflowError:", e)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    r = np.int32(2**31 - 1) + np.int32(1)
print("int32 scalar overflow ->", r, "| warning:", str(caught[0].message))

# %% [markdown]
# **Fix:** compute in a wider type, then cast back deliberately (with clipping if that is what you want).

# %%
wide = pixels.astype(np.int16) + 10
print("in int16:", wide, "| clipped back to uint8:", np.clip(wide, 0, 255).astype(np.uint8))

# %% [markdown]
# The same idea applies to sums: `np.sum` of a small integer array accumulates in a wider type by default (platform integer), which is why
# `uint8_array.sum()` is usually safe but `uint8_array.cumsum(dtype=np.uint8)` is not.

# %%
small = np.full(1000, 200, dtype=np.uint8)
print("sum:", small.sum(), "| sum forced into uint8:", small.sum(dtype=np.uint8))

# %% [markdown]
# ## 2. Floating-point precision
#
# Floats are binary approximations. `0.1 + 0.2 != 0.3`, and small errors *accumulate*. Never compare floats with `==`; use `np.isclose` / `np.allclose`.

# %%
print(0.1 + 0.2 == 0.3, "|", 0.1 + 0.2, "|", np.isclose(0.1 + 0.2, 0.3))

# %% [markdown]
# A `float32` has about 7 significant decimal digits. Above 2^24 = 16,777,216 it cannot even represent every integer:

# %%
print(np.float32(16_777_216) + np.float32(1) == np.float32(16_777_216))

# %% [markdown]
# Accumulation error is a real hazard. Adding `0.1` a million times in `float32` with a naive loop drifts far from the true 100,000,
# while `np.sum` uses **pairwise summation** and stays much closer:

# %%
acc = np.float32(0)
for _ in range(1_000_000):
    acc += np.float32(0.1)
vec = np.full(1_000_000, 0.1, dtype=np.float32).sum()
print(f"naive float32 loop: {acc:.1f} | np.sum float32: {vec:.1f} | exact: 100000.0")

# %% [markdown]
# Lesson: prefer library reductions over hand-written accumulation loops, and use `float64` (the default) unless memory demands `float32`
# (deep learning commonly uses `float32`, which is fine there because models are tolerant to small noise).

# %% [markdown]
# ## 3. NaN: contagious and never equal to itself
#
# `NaN` ("not a number") marks missing or invalid floats. Any arithmetic with it gives `NaN`, and `nan == nan` is `False`.
# Integers cannot hold NaN, so a column with a missing value becomes float.

# %%
temps = np.array([21.0, 22.5, np.nan, 20.0, np.nan, 23.5])
print("mean:", temps.mean(), "| nan == nan:", np.nan == np.nan)
print("nanmean:", np.nanmean(temps), "| nanmax:", np.nanmax(temps))
print("isnan mask:", np.isnan(temps))

# %% [markdown]
# Use `np.isnan` (never `== np.nan`), the `nan*` reductions (`nansum`, `nanmean`, `nanstd`, `nanpercentile`), or filter first.
# **Decide** what a missing value means before ignoring it: `nanmean` silently changes the sample size.

# %%
valid = temps[~np.isnan(temps)]
print(f"{len(valid)} of {len(temps)} valid | mean {valid.mean():.2f}")

filled = np.where(np.isnan(temps), np.nanmean(temps), temps)      # simple mean imputation
print(filled)

# %% [markdown]
# Also note `np.inf` (division by a float zero gives it, with a warning) and that comparisons with NaN are False, so `temps > 22` silently drops the NaN rows.

# %%
print(temps > 22)
# float division by zero gives inf/nan plus a RuntimeWarning; np.errstate controls the warning
with np.errstate(divide="ignore", invalid="ignore"):
    print(np.array([1.0, 0.0, -1.0]) / 0.0)

# %% [markdown]
# ## 4. Memory: know your array's footprint
#
# `nbytes = size * itemsize`. Before creating a big array, compute its size. A 50,000 x 50,000 `float64` matrix is about 20 GB, which
# will kill your kernel; you can see that without allocating it.

# %%
def gb(shape, dtype=np.float64):
    return np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize / 1e9

print(f"50k x 50k float64: {gb((50_000, 50_000)):.1f} GB | as float32: {gb((50_000, 50_000), np.float32):.1f} GB")
print(f"1080p RGB frame uint8: {gb((1080, 1920, 3), np.uint8) * 1000:.2f} MB | as float64: {gb((1080, 1920, 3)) * 1000:.1f} MB")

# %% [markdown]
# ## 5. In-place operations and `out=`
#
# `a = a + 1` allocates a new array; `a += 1` writes into the existing buffer (no allocation). In-place is faster and lighter on memory,
# but (1) it changes **every alias** of that array, and (2) it cannot change dtype: `+= 1.5` on an integer array raises.

# %%
a = np.arange(3)
alias = a
a += 1                       # in place: alias sees it
print("alias:", alias)
a = a + 1                    # rebinding: alias keeps the old buffer
print("a:", a, "| alias:", alias)

try:
    ints = np.arange(3)
    ints += 1.5
except TypeError as e:
    print(type(e).__name__, "-", str(e)[:75])

# %% [markdown]
# `out=` lets a ufunc write into a pre-allocated buffer, avoiding temporaries in hot loops:

# %%
x = np.random.default_rng(42).random(5)
buf = np.empty_like(x)
np.multiply(x, 2, out=buf)
np.add(buf, 1, out=buf)
print(np.allclose(buf, x * 2 + 1))

# %% [markdown]
# ## 6. Strides, contiguity and `sliding_window_view` (lightly)
#
# A NumPy array is a flat buffer plus **strides**: how many bytes to step to reach the next element along each axis.
# Transposing does not move data; it only swaps strides. That is why `.T` is free, and also why iterating a transposed array can be slower
# (non-contiguous memory access).

# %%
m = np.arange(12).reshape(3, 4)
print("strides:", m.strides, "| transposed:", m.T.strides, "| C-contiguous after .T?", m.T.flags["C_CONTIGUOUS"])

# %% [markdown]
# `sliding_window_view` builds overlapping windows **as a view** (no copying), perfect for moving averages:

# %%
from numpy.lib.stride_tricks import sliding_window_view

signal = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
windows = sliding_window_view(signal, window_shape=3)
print(windows)
print("moving average (window 3):", windows.mean(axis=1))
print("view of the original?", np.shares_memory(windows, signal), "| read-only:", not windows.flags.writeable)

# %% [markdown]
# Treat raw `as_strided` as an expert tool: a wrong shape reads memory outside the array with no error. `sliding_window_view` is the checked, read-only alternative.

# %% [markdown]
# ## Common mistakes
#
# 1. **Doing arithmetic on `uint8` images** and letting results wrap. Cast to a wider type first.
# 2. **Comparing floats with `==`.** Use `np.isclose` with a sensible tolerance.
# 3. **`x == np.nan`** to find missing values: always `False`. Use `np.isnan`.
# 4. **Silently using `nanmean`** and forgetting how many values were dropped.
# 5. **Aliasing and in-place updates.** `b = a; b += 1` changes `a`. Use `b = a.copy()`.
# 6. **Assuming `float32` is "the same but smaller".** It has ~7 digits of precision; accumulate in `float64`.
# 7. **Building a huge array to "see if it fits".** Compute `nbytes` first.

# %%
print(np.array([np.nan, 1.0]) == np.nan)      # all False: the classic bug

# %% [markdown]
# ## Interview corner
#
# **1. What happens when a `uint8` array holds 250 and you add 10?**
# <details><summary>Answer</summary>
# The array result wraps modulo 256 to 4 without an error. Cast to a wider dtype (or clip) before arithmetic if overflow is possible.
# </details>
#
# **2. Why is `0.1 + 0.2 != 0.3` and how do you compare floats?**
# <details><summary>Answer</summary>
# Neither 0.1 nor 0.2 is exactly representable in binary floating point, so the sum differs from 0.3 by a tiny amount.
# Compare with a tolerance: <code>np.isclose</code>, <code>np.allclose</code> or <code>math.isclose</code>.
# </details>
#
# **3. How do you compute the mean of an array with missing values, and what should you worry about?**
# <details><summary>Answer</summary>
# <code>np.nanmean</code> (or mask with <code>~np.isnan</code>). Worry about how many values are missing and whether the missingness is random; ignoring them
# changes the effective sample size and can bias the result.
# </details>
#
# **4. Difference between `a += 1` and `a = a + 1`?**
# <details><summary>Answer</summary>
# <code>+=</code> modifies the existing buffer in place (all aliases and views see it, dtype cannot change);
# <code>a = a + 1</code> allocates a new array and rebinds the name.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Brighten an image safely
# `img` is a `uint8` image. Create `brighter`: every pixel increased by 60 but **clipped at 255**, still dtype `uint8`.
# A naive `img + 60` would wrap bright pixels to dark ones.

# %%
img = np.random.default_rng(42).integers(0, 256, size=(8, 8), dtype=np.uint8)

# %% tags=["starter"]
brighter = ...

# %%
expected = np.minimum(img.astype(int) + 60, 255)
assert brighter.dtype == np.uint8, f"keep dtype uint8, got {brighter.dtype}"
assert np.array_equal(brighter, expected), "each pixel = min(pixel + 60, 255)"
assert (brighter >= img).all(), "no pixel may get darker"
print("Exercise 1 passed")

# %% tags=["solution"]
brighter = np.clip(img.astype(np.int16) + 60, 0, 255).astype(np.uint8)

# %% [markdown]
# ### Exercise 2: Missing sensor readings
# `power` has 200 readings with some `NaN`. Compute `n_missing`, `mean_valid` (mean ignoring NaN) and `power_filled`
# (NaN replaced by the **median** of valid values, not the mean).

# %%
power = np.random.default_rng(42).normal(210, 20, 200)
power[np.random.default_rng(1).choice(200, 17, replace=False)] = np.nan

# %% tags=["starter"]
n_missing = ...
mean_valid = ...
power_filled = ...

# %%
mask = np.isnan(power)
assert n_missing == int(mask.sum()), "count the NaNs"
assert np.isclose(mean_valid, power[~mask].sum() / (~mask).sum()), "mean of the non-NaN values"
assert not np.isnan(power_filled).any(), "no NaN should remain"
assert np.allclose(power_filled[mask], np.median(power[~mask])), "fill with the median of the valid values"
assert np.array_equal(power_filled[~mask], power[~mask]), "valid values must be unchanged"
print(f"Exercise 2 passed ({n_missing} missing)")

# %% tags=["solution"]
n_missing = int(np.isnan(power).sum())
mean_valid = np.nanmean(power)
power_filled = np.where(np.isnan(power), np.nanmedian(power), power)

# %% [markdown]
# ### Exercise 3: Float comparison
# Sum `0.1` ten times in a Python loop into `total`. Store in `equals_one` the result of the (naive) `total == 1.0`, and in
# `close_to_one` the result of a tolerance-based comparison. *Predict `equals_one` before you run it.*

# %% tags=["starter"]
total = 0.0
# accumulate here
equals_one = ...
close_to_one = ...

# %%
ref = 0.0
for _ in range(10):
    ref += 0.1
assert np.isclose(total, ref), "add 0.1 exactly ten times"
assert equals_one == (ref == 1.0), "equals_one is the plain == result"
assert close_to_one is True or close_to_one == np.True_, "use np.isclose(total, 1.0)"
print(f"Exercise 3 passed (total = {total!r}, == 1.0 is {equals_one})")

# %% tags=["solution"]
total = 0.0
for _ in range(10):
    total += 0.1
equals_one = bool(total == 1.0)
close_to_one = bool(np.isclose(total, 1.0))

# %% [markdown]
# ### Exercise 4: Moving average with windows
# For `signal` (length 100), compute `ma5`: the moving average with window 5 (only full windows, so length 96) using `sliding_window_view`.
# Also store `is_view`: `True` if `ma5`'s source windows share memory with `signal` (use `np.shares_memory` on the windows, not on the result).

# %%
signal = np.cumsum(np.random.default_rng(42).normal(0, 1, 100))

# %% tags=["starter"]
ma5 = ...
is_view = ...

# %%
loop = np.array([signal[i:i + 5].mean() for i in range(96)])
assert ma5.shape == (96,) and np.allclose(ma5, loop), "average each full window of 5 consecutive values"
assert is_view is True or is_view == np.True_, "sliding_window_view returns a view"
print("Exercise 4 passed")

# %% tags=["solution"]
windows = sliding_window_view(signal, 5)
ma5 = windows.mean(axis=1)
is_view = bool(np.shares_memory(windows, signal))

# %% [markdown]
# ### Exercise 5 (open-ended): How big is too big?
# Your laptop has 16 GB free. For a matrix of 1 million rows x 500 features, compute the size as `float64` and `float32`. Would you
# load it whole? Write two sentences on what you would do otherwise (chunking, `float32`, `np.memmap`, a database).

# %% [markdown]
# ## Recap
# - Fixed-width ints wrap silently on arrays; widen, then clip and cast back.
# - Floats are approximations: use `isclose`, prefer library reductions, and keep accumulators in `float64`.
# - NaN is contagious and unequal to itself: `isnan`, `nan*` functions, and decide what "missing" means.
# - `nbytes = size * itemsize`; compute before allocating. In-place ops save memory but affect every alias.
# - Strides explain free transposes; use `sliding_window_view` instead of raw `as_strided`.
#
# **Go deeper:** NumPy docs on "Data types" (including integer overflow notes) and "Handling of floating-point errors" (`np.errstate`); numpy.org/doc/stable.
#
# **Next:** [NumPy mini project: images and k-means](06_numpy_mini_project.ipynb)
