# %% [markdown]
# # Linear Algebra, Random Numbers and Statistics in NumPy
#
# **Why this matters.** A linear regression is `solve(X'X, X'y)`; a simulation is "draw 10,000 random samples and
# average"; a report is "mean, spread and percentiles by group". All three are a few NumPy calls, and knowing them means you can
# prototype an algorithm from a paper in minutes and check a library's answer against your own.
#
# **You will learn**
# - multiply matrices (`@`), solve linear systems and fit a least-squares line with `np.linalg`
# - draw reproducible random numbers with `default_rng` and simulate a simple process
# - compute reductions along an axis, percentiles, and handle ties in sorting
# - use `np.where` / `np.select` for vectorised conditional logic
#
# **Prerequisites:** [Broadcasting](03_broadcasting_and_vectorization.ipynb). Some math background helps: see
# [Math for Data Science](../02_Math_for_Data_Science/).

# %%
import numpy as np

# %% [markdown]
# ## 1. Matrix multiplication and `@`
#
# `*` is element-wise. `@` (or `np.matmul`) is the matrix product: `(m, k) @ (k, n) -> (m, n)`. The inner sizes must match.

# %%
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])   # swaps columns when multiplied on the right
print("A * B (element-wise):\n", A * B)
print("A @ B (matrix product):\n", A @ B)

x = np.array([1.0, 1.0])
print("A @ x =", A @ x)          # a matrix times a vector

# %% [markdown]
# *What just happened?* `A @ B` took rows of `A` dotted with columns of `B`; here it swapped the columns of `A`.
# In machine learning, `X @ w` computes the predictions of a linear model for **all rows at once**.

# %% [markdown]
# ## 2. `np.linalg`: solve, inverse, least squares
#
# To solve `A x = b`, use **`np.linalg.solve`**, not `inv(A) @ b`: it is faster and numerically more stable.

# %%
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
sol = np.linalg.solve(A, b)
print("solution:", sol, "| A @ sol == b ?", np.allclose(A @ sol, b))
print("det:", np.linalg.det(A), "| inverse @ A ~ identity:", np.allclose(np.linalg.inv(A) @ A, np.eye(2)))

# %% [markdown]
# **Least squares.** Fitting a line `y = w0 + w1 * x` to noisy data means finding `w` that minimises `||Xw - y||^2`.
# `np.linalg.lstsq` does that stably. We generate data from known parameters, so we can check the answer is close.

# %%
rng = np.random.default_rng(42)
x = rng.uniform(0, 10, size=200)
true_w0, true_w1 = 3.0, 2.0
y = true_w0 + true_w1 * x + rng.normal(0, 1.0, size=200)

X_design = np.column_stack([np.ones_like(x), x])       # a column of 1s gives the intercept
w, residuals, rank, sv = np.linalg.lstsq(X_design, y, rcond=None)
print("estimated [intercept, slope]:", w.round(3), "| true:", [true_w0, true_w1])

# the textbook "normal equations" give the same answer
w_normal = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
print("normal equations agree:", np.allclose(w, w_normal))

# %% [markdown]
# The normal equations are fine for teaching but get numerically fragile when columns are nearly collinear; `lstsq` (SVD-based) is the safer default.
# Eigen-decomposition and SVD are also one call away:

# %%
C = np.cov(np.column_stack([x, y]).T)                 # 2x2 covariance matrix of (x, y)
vals, vecs = np.linalg.eigh(C)                        # eigh: for symmetric matrices
print("eigenvalues (ascending):", vals.round(3))
print("share of variance on the main axis:", (vals[-1] / vals.sum()).round(3))

# %% [markdown]
# That last line is the core of PCA: the eigenvector with the largest eigenvalue is the direction of maximum variance.

# %% [markdown]
# ## 3. Random numbers with `default_rng`
#
# Create **one generator** with a seed and pass it around. Same seed, same numbers; different generators do not interfere.

# %%
rng = np.random.default_rng(42)
print(rng.random(3))
print(rng.normal(loc=100, scale=15, size=3).round(1))
print(rng.integers(1, 7, size=10))                     # ten dice rolls (upper bound exclusive)
print(rng.choice(["A", "B", "C"], size=5, p=[0.6, 0.3, 0.1]))

rng2 = np.random.default_rng(42)
print("same seed reproduces:", np.allclose(rng2.random(3), np.random.default_rng(42).random(3)))

# %% [markdown]
# **A tiny simulation.** How likely is at least one six in four dice rolls? Probability theory says `1 - (5/6)**4`.
# We simulate 100,000 games and compare: the simulation is a *check* on the theory (or the other way round).

# %%
rng = np.random.default_rng(42)
rolls = rng.integers(1, 7, size=(100_000, 4))          # each row is one game
at_least_one_six = (rolls == 6).any(axis=1)
print("simulated:", at_least_one_six.mean().round(4), "| exact:", round(1 - (5 / 6) ** 4, 4))

# %% [markdown]
# ## 4. Reductions, percentiles and `axis`

# %%
rng = np.random.default_rng(42)
latency = rng.lognormal(mean=3.0, sigma=0.5, size=(4, 1000))   # 4 servers x 1000 requests (ms); skewed like real latency

print("mean per server  :", latency.mean(axis=1).round(1))
print("median per server:", np.median(latency, axis=1).round(1))
print("p95 per server   :", np.percentile(latency, 95, axis=1).round(1))
print("std per server   :", latency.std(axis=1, ddof=1).round(1), "(ddof=1: sample std)")

# %% [markdown]
# Two details worth remembering: (1) for skewed data like latency the **mean is above the median**, so report percentiles;
# (2) `np.std` defaults to the **population** formula (`ddof=0`); for a sample use `ddof=1`. pandas' `.std()` defaults to `ddof=1`, a classic cross-library discrepancy.
#
# Correlation between two variables:

# %%
print(np.corrcoef(x, y)[0, 1].round(3))      # for the (x, y) line data from above

# %% [markdown]
# ## 5. Conditional logic: `np.where` and `np.select`
#
# `np.where(cond, if_true, if_false)` is a vectorised ternary. For **several** conditions use `np.select(condlist, choicelist, default)`;
# the **first** matching condition wins.

# %%
scores = np.array([95, 82, 67, 45, 78, 59])
grade = np.select([scores >= 90, scores >= 75, scores >= 60], ["A", "B", "C"], default="F")
print(list(zip(scores.tolist(), grade.tolist())))
print(np.where(scores >= 60, "pass", "fail"))

# %% [markdown]
# ## 6. Sorting and ranking

# %%
v = np.array([30, 10, 20, 10])
print(np.sort(v), np.argsort(v))                 # argsort: the positions that would sort v
print(np.argsort(v, kind="stable"))              # stable: ties keep original order (positions 1 and 3 here)
print(np.unique(v, return_counts=True))          # distinct values and counts

# %%
grid = np.array([[3, 1, 2], [9, 7, 8]])
print(np.sort(grid, axis=1))                     # sort within each row
print(np.cumsum(v), np.diff(v))                  # running total; differences between neighbours

# %% [markdown]
# ## Common mistakes
#
# 1. **`inv(A) @ b`** instead of `solve(A, b)`.
# 2. **`*` when you meant `@`.** For 2-D arrays `A * B` silently multiplies element-wise if shapes match.
# 3. **`np.std` without `ddof=1`** when you mean the sample standard deviation (and comparing to pandas without noticing the different default).
# 4. **Using the global `np.random.rand`/`seed`** in new code: state is shared and easy to break. Pass an `rng` object.
# 5. **Reporting only the mean** for skewed data; add median and percentiles.
# 6. **Ordering conditions wrongly in `np.select`.** Put the most specific condition first, because the first match wins.

# %%
print(np.std([1, 2, 3, 4], ddof=0).round(4), np.std([1, 2, 3, 4], ddof=1).round(4))

# %% [markdown]
# ## Interview corner
#
# **1. Why prefer `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b`?**
# <details><summary>Answer</summary>
# <code>solve</code> uses an LU factorisation without forming the inverse: it is cheaper and more accurate. Computing the inverse explicitly is rarely needed.
# </details>
#
# **2. What is the difference between `A * B` and `A @ B`?**
# <details><summary>Answer</summary>
# <code>*</code> is element-wise (with broadcasting); <code>@</code> is the matrix product, requiring matching inner dimensions.
# </details>
#
# **3. How do you make random results reproducible in modern NumPy?**
# <details><summary>Answer</summary>
# Create a generator with <code>rng = np.random.default_rng(seed)</code> and draw from it. Pass the generator to functions that need randomness rather than using global state.
# </details>
#
# **4. `np.std` and pandas `.std()` give different numbers for the same data. Why?**
# <details><summary>Answer</summary>
# NumPy defaults to <code>ddof=0</code> (population), pandas to <code>ddof=1</code> (sample). Set <code>ddof</code> explicitly.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Solve a system
# A shop sells bundles: 2 pens + 3 notebooks cost 12.5; 4 pens + 1 notebook cost 9.5. Set up `A` (2x2) and `b` (length 2) and store the
# prices `[pen, notebook]` in `prices` using `np.linalg.solve`.

# %% tags=["starter"]
A = ...
b = ...
prices = ...

# %%
assert np.allclose(A @ prices, b), "prices must satisfy both equations"
assert np.allclose(A, [[2, 3], [4, 1]]) and np.allclose(b, [12.5, 9.5]), "A holds the counts, b the totals"
assert np.allclose(prices, [1.6, 3.1]), f"got {prices} (check by hand: 2*1.6 + 3*3.1 = 12.5)"
print("Exercise 1 passed:", prices)

# %% tags=["solution"]
A = np.array([[2.0, 3.0], [4.0, 1.0]])
b = np.array([12.5, 9.5])
prices = np.linalg.solve(A, b)

# %% [markdown]
# ### Exercise 2: Fit a line and predict
# The arrays `hours` and `score` are study hours and exam scores. Fit `score = w0 + w1 * hours` by least squares. Store `w0`, `w1`
# and `pred_8h` (the predicted score for 8 hours).

# %%
_rng = np.random.default_rng(42)
hours = _rng.uniform(0, 10, 80)
score = 40 + 5 * hours + _rng.normal(0, 4, 80)

# %% tags=["starter"]
w0 = ...
w1 = ...
pred_8h = ...

# %%
slope, intercept = np.polyfit(hours, score, 1)          # independent method
assert np.isclose(w1, slope) and np.isclose(w0, intercept), "compare with np.polyfit(hours, score, 1)"
assert np.isclose(pred_8h, w0 + 8 * w1), "pred_8h = w0 + 8 * w1"
print(f"Exercise 2 passed: score = {w0:.1f} + {w1:.2f} * hours; 8h -> {pred_8h:.1f}")

# %% tags=["solution"]
Xd = np.column_stack([np.ones_like(hours), hours])
(w0, w1), *_ = np.linalg.lstsq(Xd, score, rcond=None)
pred_8h = w0 + 8 * w1

# %% [markdown]
# ### Exercise 3: Simulate the birthday problem
# What is the probability that in a group of 23 people at least two share a birthday (365 equally likely days, ignore leap years)?
# Simulate 20,000 groups with `default_rng(42)` and store the estimate in `p_sim`. Compute the exact value in `p_exact`
# (`1 - product of (365 - k) / 365 for k in 0..22`).
# *Hint: for each row, "at least one shared" means the number of unique values is less than 23; `np.sort` along axis 1 then `np.diff` helps.*

# %% tags=["starter"]
p_sim = ...
p_exact = ...

# %%
assert abs(p_exact - (1 - np.prod([(365 - k) / 365 for k in range(23)]))) < 1e-12, "check the exact formula"
assert abs(p_sim - p_exact) < 0.02, f"simulation {p_sim:.3f} should be within 0.02 of exact {p_exact:.3f}"
print(f"Exercise 3 passed: simulated {p_sim:.3f} vs exact {p_exact:.3f}")

# %% tags=["solution"]
rng = np.random.default_rng(42)
days = np.sort(rng.integers(0, 365, size=(20_000, 23)), axis=1)
p_sim = (np.diff(days, axis=1) == 0).any(axis=1).mean()
p_exact = 1 - np.prod((365 - np.arange(23)) / 365)

# %% [markdown]
# ### Exercise 4: Tiered pricing with `np.select`
# `qty` holds order quantities. Discount rules: quantity >= 100 gets 20% off, >= 50 gets 10% off, >= 10 gets 5% off, otherwise none.
# Create `discount` (an array of fractions, e.g. 0.1) and `total` = `qty * 2.0 * (1 - discount)` (unit price 2.0).

# %%
qty = np.array([1, 9, 10, 49, 50, 99, 100, 500])

# %% tags=["starter"]
discount = ...
total = ...

# %%
expected = np.array([0, 0, 0.05, 0.05, 0.10, 0.10, 0.20, 0.20])
assert np.allclose(discount, expected), f"discount should be {expected}, got {discount}"
assert np.allclose(total, qty * 2.0 * (1 - expected)), "total = qty * unit_price * (1 - discount)"
print("Exercise 4 passed")

# %% tags=["solution"]
discount = np.select([qty >= 100, qty >= 50, qty >= 10], [0.20, 0.10, 0.05], default=0.0)
total = qty * 2.0 * (1 - discount)

# %% [markdown]
# ### Exercise 5 (open-ended): Mean vs median
# Draw 10,000 samples from `rng.lognormal(3, 0.5)`. Compute mean, median, p95. In a markdown cell, state which single number you would
# report to a stakeholder asking "how long does a typical request take?" and why.

# %% [markdown]
# ## Recap
# - `@` for matrix products; `solve` and `lstsq` instead of explicit inverses; `eigh` for symmetric matrices.
# - `default_rng(seed)` gives reproducible, local randomness; simulations check your theory.
# - Reductions take an `axis`; `np.percentile` describes skewed data; mind `ddof`.
# - `np.where` for two branches, `np.select` for many (first match wins); `argsort` and `unique` for ordering and counting.
#
# **Go deeper:** NumPy reference sections "Linear algebra (numpy.linalg)" and "Random Generator" (numpy.org/doc/stable/reference).
#
# **Next:** [NumPy pitfalls and performance](05_numpy_pitfalls_and_performance.ipynb)
