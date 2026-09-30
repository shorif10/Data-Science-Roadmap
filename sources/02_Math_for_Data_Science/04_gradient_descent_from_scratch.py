# %% [markdown]
# # Gradient Descent from Scratch
#
# **Why this matters.** Linear regression has a formula, but almost every other model (logistic regression, neural
# networks, gradient boosting's inner loops) is fitted by *iterative* optimisation. Gradient descent is the engine.
# If you understand learning rate, convergence, divergence and feature scaling on a one-feature regression, you understand the
# knobs you will tune on a neural network. You will also be able to answer "why did my loss become `NaN`?".
#
# **You will learn**
# - fit `price = w * area + b` on real-looking data using only NumPy and a `for` loop
# - visualise the loss surface and the path descent takes across it
# - see what too-small and too-large learning rates do, and compute the largest stable one
# - explain why feature scaling speeds up training, and verify against the closed-form answer
#
# **Prerequisites:** [Calculus: derivatives and gradients](03_calculus_derivatives_and_gradients.ipynb).
#
# > Data used: `datasets/housing.csv` (synthetic house prices, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)
DATA = Path("../datasets")

# %% [markdown]
# ## 1. The problem: predict price from area
#
# The file has a few impossible prices (negative, or millions for a normal house). A proper cleaning workflow lives in the pandas module; here
# we apply one simple, explicit rule and keep going: prices between 10,000 and 1,000,000. We also express area in **thousands of sqft** and price in
# **thousands of dollars** so numbers are near 1. (Why that matters: section 6.)

# %%
raw = pd.read_csv(DATA / "housing.csv")
h = raw[(raw["price"] > 10_000) & (raw["price"] < 1_000_000)]
print(f"kept {len(h)} of {len(raw)} rows")

x = (h["area_sqft"] / 1000).to_numpy()      # thousand sqft
y = (h["price"] / 1000).to_numpy()          # thousand dollars

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, s=6, alpha=0.4)
ax.set_xlabel("area (thousand sqft)"); ax.set_ylabel("price (thousand $)"); ax.set_title("Price rises with area")
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 2. The answer we are aiming for: the closed form
#
# For linear regression the best `(w, b)` has a direct solution (least squares). We compute it first as the *ground truth* to check our descent against.

# %%
A = np.column_stack([x, np.ones_like(x)])          # design matrix: [area, 1]
(w_opt, b_opt), *_ = np.linalg.lstsq(A, y, rcond=None)
mse_opt = np.mean((A @ [w_opt, b_opt] - y) ** 2)
print(f"closed form: w = {w_opt:.3f} (k$ per 1000 sqft), b = {b_opt:.3f} (k$), MSE = {mse_opt:.1f}")

# %% [markdown]
# ## 3. The loss and its gradient
#
# Loss (mean squared error): `L(w, b) = mean((w*x + b - y)^2)`. Differentiating (you verified this formula in the previous notebook):
#
# - `dL/dw = 2 * mean((pred - y) * x)`
# - `dL/db = 2 * mean(pred - y)`
#
# The **loss surface** is a bowl over the `(w, b)` plane. Let us look at it.

# %%
def mse(w, b):
    return np.mean((w * x + b - y) ** 2)

def gradient(w, b):
    err = w * x + b - y
    return 2 * np.mean(err * x), 2 * np.mean(err)

ws, bs = np.meshgrid(np.linspace(0, 300, 120), np.linspace(-150, 150, 120))
Z = np.array([[mse(w_, b_) for w_ in ws[0]] for b_ in bs[:, 0]])

fig, ax = plt.subplots(figsize=(6, 4.5))
cs = ax.contour(ws, bs, Z, levels=25, cmap="viridis")
ax.plot(w_opt, b_opt, "r*", ms=14, label="closed-form optimum")
ax.set_xlabel("w (slope)"); ax.set_ylabel("b (intercept)"); ax.set_title("Loss surface of linear regression (contours of MSE)")
ax.legend()
plt.show()

# %% [markdown]
# The contours are elongated ellipses: a **long narrow valley**. `w` and `b` trade off (a steeper line needs a lower intercept), so the surface is
# much steeper in some directions than others. Keep this shape in mind; it is why learning rates are delicate.

# %% [markdown]
# ## 4. Gradient descent, in 8 lines
#
# Repeat: compute the gradient, take a small step *against* it. `theta <- theta - lr * grad`. The **learning rate** `lr` is the step size.

# %%
def gradient_descent(lr, steps, w0=0.0, b0=0.0):
    w, b = w0, b0
    path = [(w, b, mse(w, b))]
    for _ in range(steps):
        gw, gb = gradient(w, b)
        w, b = w - lr * gw, b - lr * gb
        path.append((w, b, mse(w, b)))
    return np.array(path)

path = gradient_descent(lr=0.05, steps=2000)
print(f"after 2000 steps: w = {path[-1, 0]:.3f}, b = {path[-1, 1]:.3f}, MSE = {path[-1, 2]:.1f}")
print(f"closed form     : w = {w_opt:.3f}, b = {b_opt:.3f}, MSE = {mse_opt:.1f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
axes[0].contour(ws, bs, Z, levels=25, cmap="viridis", alpha=0.6)
axes[0].plot(path[:, 0], path[:, 1], "o-", ms=2, color="tab:red", lw=1, label="descent path")
axes[0].plot(w_opt, b_opt, "k*", ms=12, label="optimum")
axes[0].set_xlabel("w"); axes[0].set_ylabel("b"); axes[0].set_title("Path across the loss surface"); axes[0].legend()
axes[1].plot(path[:, 2])
axes[1].set_yscale("log"); axes[1].set_xlabel("step"); axes[1].set_ylabel("MSE (k$^2), log scale")
axes[1].set_title("Loss falls fast, then creeps along the valley")
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
# *What just happened?* Starting at `(0, 0)`, the path drops steeply into the valley then slides along its floor towards the optimum. The
# slow crawl at the end is the elongated-valley effect. We got the same answer as the formula, without ever inverting a matrix.
# That matters when you have millions of parameters and no closed form.

# %% [markdown]
# ## 5. The learning rate: too small, just right, too large
#
# The step is `lr * gradient`. Too small: you crawl. Too large: you leap over the valley and the loss grows without bound (divergence).
# There is a hard threshold for this loss: for a quadratic, descent is stable only if `lr < 2 / lambda_max`, where `lambda_max` is the largest eigenvalue
# of the Hessian (matrix of second derivatives). For MSE the Hessian is `2 * mean([[x^2, x], [x, 1]])`, so we can compute the limit from data.

# %%
H = 2 * np.array([[np.mean(x**2), np.mean(x)], [np.mean(x), 1.0]])
lam = np.linalg.eigvalsh(H)
lr_max = 2 / lam.max()
print("Hessian eigenvalues:", lam, "-> condition number", round(lam.max() / lam.min(), 1))
print(f"largest stable learning rate: {lr_max:.3f}")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for lr, style in [(0.005, "tab:blue"), (0.05, "tab:green"), (0.9 * lr_max, "tab:orange"), (1.05 * lr_max, "tab:red")]:
    p = gradient_descent(lr, steps=60)
    ax.plot(np.minimum(p[:, 2], 1e6), color=style, label=f"lr = {lr:.3f}")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE (capped at 1e6)")
ax.set_title("Learning rate decides: crawl, converge, oscillate, explode")
ax.legend(); ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# Blue crawls, green makes steady progress, orange (10% below the limit) oscillates but the loss still falls, red (5% above it) explodes. The formula
# separates the two behaviours, and Exercise 3 confirms it by brute force: divergence is not random, it is arithmetic. In practice you cannot compute the Hessian of a neural
# network, so you find a good learning rate by trying a few values on a log scale (0.1, 0.03, 0.01, ...) and watching the loss curve.

# %% [markdown]
# ## 6. Why feature scaling matters
#
# Suppose we had left area in raw square feet (values about 1,500) instead of thousands. The curvature along `w` grows by the square of the scale,
# the valley gets far narrower, and the stable learning rate collapses. Measure it:

# %%
x_raw = h["area_sqft"].to_numpy()
H_raw = 2 * np.array([[np.mean(x_raw**2), np.mean(x_raw)], [np.mean(x_raw), 1.0]])
lam_raw = np.linalg.eigvalsh(H_raw)
print(f"raw sqft : condition number {lam_raw.max() / lam_raw.min():,.0f}, largest stable lr = {2 / lam_raw.max():.2e}")
print(f"k-sqft   : condition number {lam.max() / lam.min():,.0f}, largest stable lr = {lr_max:.2e}")

xs_ = (x_raw - x_raw.mean()) / x_raw.std()
H_std = 2 * np.array([[np.mean(xs_**2), np.mean(xs_)], [np.mean(xs_), 1.0]])
lam_std = np.linalg.eigvalsh(H_std)
print(f"standardised: condition number {lam_std.max() / lam_std.min():.1f}, largest stable lr = {2 / lam_std.max():.2f}")

# %% [markdown]
# *What just happened?* The **condition number** (ratio of largest to smallest curvature) measures how squashed the valley is. Raw square feet gives a
# huge one, so you need a tiny step to avoid divergence and then it takes forever to travel along the valley. Standardising (mean 0, std 1) makes the contours
# nearly circular and one learning rate suits every direction. The same reason applies to every neural network input.

# %% [markdown]
# ## 7. Stochastic and mini-batch gradient descent
#
# Computing the gradient over all rows is costly for big data. **Mini-batch** descent estimates it from a random subset each step. The path gets noisy but each
# step is cheap, and the noise averages out. This is what "SGD" means in deep learning libraries.

# %%
def minibatch_gd(lr, epochs, batch_size, seed=42):
    rng = np.random.default_rng(seed)
    w, b = 0.0, 0.0
    hist = []
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for i in range(0, len(x), batch_size):
            idx = order[i : i + batch_size]
            err = w * x[idx] + b - y[idx]
            w -= lr * 2 * np.mean(err * x[idx])
            b -= lr * 2 * np.mean(err)
        hist.append(mse(w, b))
    return w, b, hist

w_mb, b_mb, hist = minibatch_gd(lr=0.02, epochs=40, batch_size=64)
print(f"mini-batch: w = {w_mb:.2f}, b = {b_mb:.2f}, MSE = {hist[-1]:.1f} (optimum MSE {mse_opt:.1f})")

# %% [markdown]
# Finally, compare with scikit-learn, which solves the same problem by a direct method:

# %%
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"sklearn: w = {lr_model.coef_[0]:.3f}, b = {lr_model.intercept_:.3f}")
print(f"ours   : w = {path[-1, 0]:.3f}, b = {path[-1, 1]:.3f}")

# %% [markdown]
# ## Common mistakes
#
# 1. **Not scaling features**, then blaming the optimiser. Standardise inputs first.
# 2. **A learning rate that is too big**: the loss goes up or turns `inf`/`NaN`. Halve it until the loss decreases monotonically.
# 3. **Stopping too early** because the loss "looks flat" on a linear axis. Plot on a log axis, or track the gradient norm.
# 4. **Updating `w` and then computing the `b` gradient with the new `w`.** Compute both gradients from the same point, then update together.
# 5. **Judging on the training loss only.** Descent minimises training error; overfitting is a different matter (later modules).

# %%
p_bad = gradient_descent(lr=1.2 * lr_max, steps=30)
print("loss after 30 steps with lr = 1.2 x limit:", f"{p_bad[-1, 2]:.3g}", "(started at", f"{p_bad[0, 2]:.3g})")

# %% [markdown]
# ## Interview corner
#
# **1. What happens if the learning rate is too large or too small?**
# <details><summary>Answer</summary>
# Too small: painfully slow, may look stuck. Too large: overshoots the minimum, oscillates or diverges (loss increases, NaN). Practical fixes: try a log-spaced grid,
# use a schedule that decays, or an adaptive optimiser such as Adam.
# </details>
#
# **2. Batch, mini-batch and stochastic gradient descent: differences?**
# <details><summary>Answer</summary>
# Batch uses all data per step (exact gradient, expensive); stochastic uses one sample (very noisy, cheap); mini-batch uses a small subset (the practical compromise that also
# uses GPU parallelism well). The noise can help escape shallow regions, but you usually need a decaying learning rate to settle.
# </details>
#
# **3. Why does feature scaling speed up gradient descent?**
# <details><summary>Answer</summary>
# Unscaled features make the loss surface a narrow valley (a large condition number), so one learning rate cannot suit all directions. Scaling makes the curvature similar in every direction.
# </details>
#
# **4. Linear regression has a closed form. Why would you ever use gradient descent for it?**
# <details><summary>Answer</summary>
# The normal equations cost roughly cubic time in the number of features and need a matrix inverse or decomposition, which is infeasible or unstable for huge feature counts.
# Descent scales to big data, works with mini-batches and streaming, and generalises to models with no closed form.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Gradient for multiple features
# Write `mse_grad(w, b, X, y)` for a linear model `X @ w + b` with `X` of shape `(n, d)`. Return `(grad_w, grad_b)` where `grad_w` has shape `(d,)`.
# *Hint: with `err = X @ w + b - y`, `grad_w = 2/n * X.T @ err` and `grad_b = 2 * mean(err)`.*

# %% tags=["starter"]
def mse_grad(w, b, X, y):
    ...

# %%
r = np.random.default_rng(1)
Xt, yt, wt, bt = r.normal(size=(40, 3)), r.normal(size=40), r.normal(size=3), 0.7
f_ = lambda w_, b_: np.mean((Xt @ w_ + b_ - yt) ** 2)
gw, gb = mse_grad(wt, bt, Xt, yt)
num_w = np.array([(f_(wt + e, bt) - f_(wt - e, bt)) / 2e-6 for e in np.eye(3) * 1e-6])
num_b = (f_(wt, bt + 1e-6) - f_(wt, bt - 1e-6)) / 2e-6
assert np.shape(gw) == (3,), "grad_w should have shape (d,)"
assert np.allclose(gw, num_w, atol=1e-5) and abs(gb - num_b) < 1e-5, "compare with numerical gradient: check the factor 2/n and the transpose"
print("Exercise 1 passed")

# %% tags=["solution"]
def mse_grad(w, b, X, y):
    err = X @ w + b - y
    return 2 / len(y) * X.T @ err, 2 * np.mean(err)

# %% [markdown]
# ### Exercise 2: Fit with standardised area and convert back
# Standardise `x` (call it `z`, mean 0, std 1) and run gradient descent for `lr=0.3`, `steps=200` starting from `(0, 0)` on `z` instead of `x`
# (write your own loop with `mse_grad`, using `z.reshape(-1, 1)`). The fitted `(w_z, b_z)` are in standardised units.
# Store in `w_orig` and `b_orig` the equivalent slope and intercept in the original `x` units (thousand sqft).
# *Hint: `w*z + b = w*(x - mu)/sd + b`, so `w_orig = w/sd` and `b_orig = b - w*mu/sd`.*

# %% tags=["starter"]
z = (x - x.mean()) / x.std()
w_orig = ...
b_orig = ...

# %%
assert abs(w_orig - w_opt) < 0.05 and abs(b_orig - b_opt) < 0.05, f"expected about w={w_opt:.3f}, b={b_opt:.3f}; got w={w_orig:.3f}, b={b_orig:.3f}"
print(f"Exercise 2 passed: w={w_orig:.3f}, b={b_orig:.3f} vs closed form {w_opt:.3f}, {b_opt:.3f}")

# %% tags=["solution"]
z = (x - x.mean()) / x.std()
wz, bz = np.zeros(1), 0.0
for _ in range(200):
    gw_, gb_ = mse_grad(wz, bz, z.reshape(-1, 1), y)
    wz, bz = wz - 0.3 * gw_, bz - 0.3 * gb_
w_orig = float(wz[0] / x.std())
b_orig = float(bz - wz[0] * x.mean() / x.std())

# %% [markdown]
# ### Exercise 3: Find the divergence threshold empirically
# Without using the Hessian formula, find by bisection the largest learning rate (to within 0.01) for which `gradient_descent(lr, steps=200)` on the `x` data
# ends with a **lower loss than it started**. Search between `lo=0.01` and `hi=1.0`. Store the result in `lr_found`.
# *Hint: keep `lo` (converges) and `hi` (diverges); test the midpoint; move the matching end.*

# %% tags=["starter"]
lr_found = ...

# %%
def converges(lr):
    p = gradient_descent(lr, 200)
    return np.isfinite(p[-1, 2]) and p[-1, 2] < p[0, 2]

assert converges(lr_found), "the returned learning rate should itself converge"
assert not converges(lr_found + 0.03), "a rate 0.03 above your answer should already diverge, so your answer is not the largest"
assert abs(lr_found - lr_max) < 0.03, f"theory (2/lambda_max) says about {lr_max:.3f}"
print(f"Exercise 3 passed: found {lr_found:.3f}, theory {lr_max:.3f}")

# %% tags=["solution"]
def _conv(lr):
    with np.errstate(all="ignore"):
        p = gradient_descent(lr, 200)
    return np.isfinite(p[-1, 2]) and p[-1, 2] < p[0, 2]

lo, hi = 0.01, 1.0
while hi - lo > 0.005:
    mid = (lo + hi) / 2
    if _conv(mid):
        lo = mid
    else:
        hi = mid
lr_found = lo

# %% [markdown]
# ### Exercise 4 (open-ended): Descent on a different feature
# Repeat the fit with `distance_to_center_km` instead of area (drop rows with missing values first, standardise it, choose your own learning rate). Report the
# sign and size of the slope in dollars per km, and say in a markdown cell whether closer-to-centre means more expensive here and how sure you are (look at a scatter plot).

# %% [markdown]
# ## Recap
# - Gradient descent repeats `theta <- theta - lr * grad(theta)`; for MSE the gradient is a couple of NumPy lines.
# - The loss surface of linear regression is a bowl; elongated bowls make descent slow and the learning rate delicate.
# - There is a hard stability limit `lr < 2 / lambda_max`; beyond it the loss diverges. Otherwise choose `lr` by trying a log-spaced range.
# - Standardising features shrinks the condition number, which is the reason it speeds up training.
# - Mini-batches trade exactness for cheap, noisy steps; that is SGD.
#
# **Go deeper:** the scikit-learn user guide sections on `SGDRegressor` and "Stochastic Gradient Descent"; Sebastian Ruder's overview "An overview of gradient descent optimization algorithms" is a well-known reference.
#
# **Next:** [Loss functions and optimization](05_loss_functions_and_optimization.ipynb)
