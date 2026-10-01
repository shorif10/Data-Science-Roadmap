# %% [markdown]
# # Loss Functions and Optimization
#
# **Why this matters.** The loss function *is* the definition of "good" that your model chases. Pick squared error when your data
# has wild outliers and the model bends over backwards to fit them; pick accuracy as your training target and there is no gradient
# to follow. Choosing the loss, and understanding what the optimiser can and cannot do with it, is a large part of modelling judgment.
#
# **You will learn**
# - compare MSE, MAE and Huber losses and what statistic each one estimates (mean vs median)
# - compare log-loss, hinge loss and 0-1 loss for classification and why log-loss punishes confident mistakes
# - recognise convex vs non-convex losses, and watch descent get stuck in a local minimum
# - compare plain gradient descent with momentum and Adam on a badly scaled surface
#
# **Prerequisites:** [Gradient descent from scratch](04_gradient_descent_from_scratch.ipynb).
#
# > Data used: `datasets/housing.csv` (synthetic, see `docs/DATASETS.md`) and small toy functions.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)
DATA = Path("../datasets")

# %% [markdown]
# ## 1. Regression losses: what does each one "want"?
#
# Given residual `r = prediction - actual`:
#
# | Loss | Formula | Behaviour |
# |---|---|---|
# | MSE | `r^2` | punishes big errors heavily; smooth; sensitive to outliers |
# | MAE | `abs(r)` | every unit of error costs the same; robust; not differentiable at 0 |
# | Huber | `r^2/2` if `abs(r) <= d`, else `d*(abs(r) - d/2)` | quadratic near 0, linear in the tails: a compromise |

# %%
def mse_l(r): return r**2
def mae_l(r): return np.abs(r)
def huber_l(r, d=1.0):
    return np.where(np.abs(r) <= d, 0.5 * r**2, d * (np.abs(r) - 0.5 * d))

r = np.linspace(-4, 4, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(r, mse_l(r), label="MSE r^2"); ax.plot(r, mae_l(r), label="MAE |r|"); ax.plot(r, huber_l(r), label="Huber (d=1)")
ax.set_ylim(0, 10); ax.set_xlabel("residual (prediction - actual)"); ax.set_ylabel("loss")
ax.set_title("Shape of common regression losses"); ax.grid(alpha=0.3); ax.legend()
plt.show()

# %% [markdown]
# Here is the deep fact: **the constant that minimises MSE is the mean; the constant that minimises MAE is the median.** So choosing MSE means
# "predict the average", MAE means "predict the typical (median)". We can check on real prices. Take house prices including the weird ones.

# %%
prices = pd.read_csv(DATA / "housing.csv")["price"].to_numpy() / 1000   # thousand dollars
cands = np.linspace(prices.min(), 400, 4000)
mse_curve = [np.mean((c - prices) ** 2) for c in cands]
mae_curve = [np.mean(np.abs(c - prices)) for c in cands]

print(f"argmin MSE = {cands[np.argmin(mse_curve)]:.1f}   mean   = {prices.mean():.1f}")
print(f"argmin MAE = {cands[np.argmin(mae_curve)]:.1f}   median = {np.median(prices):.1f}")

# %% [markdown]
# Now add a handful of absurdly expensive houses. Watch which estimate moves:

# %%
with_outliers = np.append(prices, [3000.0] * 15)
print(f"mean   : {prices.mean():.1f} -> {with_outliers.mean():.1f}   (moved by {with_outliers.mean() - prices.mean():+.1f})")
print(f"median : {np.median(prices):.1f} -> {np.median(with_outliers):.1f}   (moved by {np.median(with_outliers) - np.median(prices):+.1f})")

# %% [markdown]
# *What just happened?* 15 bad rows out of 3,000 dragged the mean noticeably but barely touched the median. A model trained with MSE inherits the mean's
# sensitivity; one trained with MAE or Huber inherits the median's robustness. The price: MAE's gradient has constant magnitude, so it does not slow down as
# it approaches the optimum, and Huber has an extra hyperparameter `d`.

# %% [markdown]
# ## 2. Classification losses: from the margin view
#
# For a binary label `y` in {-1, +1} and a raw score `s` (positive means "predict +1"), the **margin** is `m = y * s`. Positive margin = correct,
# larger = more confident.
#
# - **0-1 loss**: 1 if `m <= 0` else 0. This is what accuracy measures. It is flat almost everywhere, so its gradient is useless for training.
# - **Hinge loss** `max(0, 1 - m)`: used by SVMs; zero once the margin exceeds 1.
# - **Log-loss** `log(1 + exp(-m))`: used by logistic regression; smooth and always pushes for a larger margin.

# %%
m = np.linspace(-3, 3, 400)
zero_one = (m <= 0).astype(float)
hinge = np.maximum(0, 1 - m)
logloss = np.log1p(np.exp(-m))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(m, zero_one, label="0-1 (accuracy)"); ax.plot(m, hinge, label="hinge"); ax.plot(m, logloss, label="log-loss")
ax.set_xlabel("margin  y * score"); ax.set_ylabel("loss"); ax.set_title("Smooth surrogate losses vs the 0-1 loss")
ax.grid(alpha=0.3); ax.legend()
plt.show()

# %% [markdown]
# Hinge and log-loss are smooth-ish **surrogates** for 0-1: they upper-bound it (log-loss does after dividing by `log 2`) and give a gradient we can follow.
#
# In probability terms, for predicted probability `p` of the true class, log-loss per sample is `-log(p)`. Being confidently wrong is catastrophic:

# %%
for p in [0.9, 0.6, 0.5, 0.1, 0.01, 0.001]:
    print(f"probability assigned to the true class = {p:<6}  ->  loss = {-np.log(p):.2f}")

# %% [markdown]
# Two classifiers can have the *same accuracy* and very different log-loss, because one of them is overconfident when wrong:

# %%
from sklearn.metrics import accuracy_score, log_loss

y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
p_calm = np.array([0.7, 0.7, 0.7, 0.4, 0.3, 0.3, 0.3, 0.6])       # one mistake each side of 0.5
p_brash = np.array([0.99, 0.99, 0.99, 0.01, 0.01, 0.01, 0.01, 0.99])  # same mistakes, extreme confidence
for name, p in [("calm", p_calm), ("brash", p_brash)]:
    print(f"{name:6s} accuracy = {accuracy_score(y_true, p > 0.5):.3f}   log-loss = {log_loss(y_true, p):.3f}")

# %% [markdown]
# ## 3. Convex vs non-convex: can descent get stuck?
#
# A function is **convex** if the straight line between any two points on its graph lies above the graph (a single bowl). For a convex loss every
# local minimum is the global minimum, so gradient descent (with a sane step) gets you there regardless of where you start. MSE and log-loss for linear models are convex. Neural network
# losses are not; they have many local minima and saddle points, yet still train well in practice (an active research topic, not a solved one).
#
# A non-convex toy: `f(x) = x^4 - 3x^2 + x` has two valleys. Where descent ends depends on where it starts:

# %%
f = lambda x: x**4 - 3 * x**2 + x
df = lambda x: 4 * x**3 - 6 * x + 1

def descend(x0, lr=0.01, steps=400):
    x = x0
    for _ in range(steps):
        x -= lr * df(x)
    return x

xs = np.linspace(-2.2, 2.2, 300)
starts = [-2.0, -0.3, 0.3, 2.0]
ends = [descend(s) for s in starts]
for s, e in zip(starts, ends):
    print(f"start {s:+.1f} -> ends at x = {e:+.3f}, f = {f(e):+.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xs, f(xs))
ax.scatter(starts, f(np.array(starts)), c="tab:green", zorder=3, label="starts")
ax.scatter(ends, f(np.array(ends)), c="tab:red", marker="*", s=150, zorder=3, label="where descent ended")
ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title("Non-convex loss: the starting point decides the valley")
ax.grid(alpha=0.3); ax.legend()
plt.show()

# %% [markdown]
# *What just happened?* Starts on the left reach the deeper (global) minimum; starts on the right get trapped in the shallower one. Common remedies: multiple random restarts,
# noisy mini-batch updates, momentum, and sensible initialisation. Convexity is a property worth checking before you promise someone a model "converges to the best solution".

# %% [markdown]
# ## 4. Momentum and Adam on a badly scaled surface
#
# Consider `f(x, y) = 0.5 * (x^2 + 25 y^2)`: steep in `y`, shallow in `x`, a miniature of the narrow valley from the last notebook. Plain descent must use a
# step small enough for the steep direction, so it crawls along the shallow one. **Momentum** keeps a running "velocity" `v <- beta*v + grad`,
# `theta <- theta - lr*v`: like a heavy ball, it accelerates along consistent directions and damps zig-zags. **Adam** additionally rescales each coordinate by a running
# estimate of its gradient magnitude.

# %%
def f2(p): return 0.5 * (p[0] ** 2 + 25 * p[1] ** 2)
def g2(p): return np.array([p[0], 25 * p[1]])

def plain(lr, p0, steps):
    p = np.array(p0, float); path = [p.copy()]
    for _ in range(steps):
        p = p - lr * g2(p); path.append(p.copy())
    return np.array(path)

def momentum(lr, beta, p0, steps):
    p = np.array(p0, float); v = np.zeros(2); path = [p.copy()]
    for _ in range(steps):
        v = beta * v + g2(p)
        p = p - lr * v; path.append(p.copy())
    return np.array(path)

def adam(lr, p0, steps, b1=0.9, b2=0.999, eps=1e-8):
    p = np.array(p0, float); m = np.zeros(2); s = np.zeros(2); path = [p.copy()]
    for t in range(1, steps + 1):
        g = g2(p)
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g**2
        p = p - lr * (m / (1 - b1**t)) / (np.sqrt(s / (1 - b2**t)) + eps)
        path.append(p.copy())
    return np.array(path)

start = (5.0, 1.0)
paths = {
    "plain GD (lr=0.07)": plain(0.07, start, 60),
    "momentum (lr=0.05, beta=0.8)": momentum(0.05, 0.8, start, 60),
    "Adam (lr=0.4)": adam(0.4, start, 60),
}

gx, gy = np.meshgrid(np.linspace(-1, 6, 200), np.linspace(-1.3, 1.3, 200))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].contour(gx, gy, 0.5 * (gx**2 + 25 * gy**2), levels=20, cmap="viridis", alpha=0.6)
for name, pth in paths.items():
    axes[0].plot(pth[:, 0], pth[:, 1], "o-", ms=2, lw=1, label=name)
    axes[1].plot([f2(q) for q in pth], label=name)
axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_title("Paths on an elongated bowl"); axes[0].legend(fontsize=8)
axes[1].set_yscale("log"); axes[1].set_xlabel("step"); axes[1].set_ylabel("f (log scale)"); axes[1].set_title("Loss per step"); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# %%
def steps_to(path_fn, tol=1e-6, maxsteps=5000):
    pth = path_fn(maxsteps)
    hit = np.where(np.array([f2(q) for q in pth]) < tol)[0]
    return int(hit[0]) if len(hit) else None

print("steps to reach f < 1e-6:")
print("  plain    :", steps_to(lambda n: plain(0.07, start, n)))
print("  momentum :", steps_to(lambda n: momentum(0.05, 0.8, start, n)))
print("  Adam     :", steps_to(lambda n: adam(0.4, start, n)))

# %% [markdown]
# *What just happened?* Plain descent zig-zags across the steep `y` direction while creeping along `x`. Momentum averages the zig-zag away and builds speed along `x`.
# Adam normalises each coordinate, so both directions progress at comparable rates, but with a constant learning rate it overshoots and jitters near the optimum.
# In this run momentum was the fastest and Adam was *not* faster than plain descent. The counts depend on the hand-picked settings and the tolerance, so do not read
# this as a ranking of optimisers: on many deep-learning problems Adam is the easy default, and a well-tuned SGD with momentum sometimes generalises better. Treat optimiser choice as something you test.

# %% [markdown]
# ## Common mistakes
#
# 1. **Using accuracy (or any non-differentiable metric) as the training loss.** Train on a smooth surrogate (log-loss), *evaluate* on the business metric.
# 2. **MSE on data with extreme outliers** without looking at them first. Consider Huber/MAE or fix the data.
# 3. **Clipping predicted probabilities incorrectly.** `log(0)` is `-inf`; libraries clip to a tiny epsilon, hand-written code must too.
# 4. **Comparing losses across different loss functions.** An MSE of 4 and an MAE of 4 are not comparable numbers.
# 5. **Assuming a smaller training loss means a better model.** Validation loss is what counts.

# %%
p = np.array([1.0, 0.0])
with np.errstate(divide="ignore"):
    print("naive log-loss with p=0 on the true class:", -np.log(p))
print("with clipping to [1e-15, 1-1e-15]:", -np.log(np.clip(p, 1e-15, 1 - 1e-15)))

# %% [markdown]
# ## Interview corner
#
# **1. When would you choose MAE or Huber over MSE?**
# <details><summary>Answer</summary>
# When the target has outliers or heavy tails you do not want to dominate the fit, or when the business cost of an error is linear. MSE is preferable when large errors are disproportionately costly or you want the conditional mean.
# </details>
#
# **2. Why not optimise accuracy directly?**
# <details><summary>Answer</summary>
# Accuracy is piecewise constant in the model parameters: its gradient is zero almost everywhere, so gradient methods get no signal. We optimise a smooth surrogate (log-loss, hinge) and report accuracy separately.
# </details>
#
# **3. What is a convex loss and why do we like it?**
# <details><summary>Answer</summary>
# Any line segment between two points on the graph lies above it, so every local minimum is global. Descent with a suitable step converges to the optimum from any start. Linear/logistic regression losses are convex; deep nets are not.
# </details>
#
# **4. What does momentum do?**
# <details><summary>Answer</summary>
# It accumulates an exponentially decaying sum of past gradients, which speeds progress along consistently signed directions and dampens oscillation across steep, narrow ones.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Implement log-loss safely
# Write `my_log_loss(y, p)` for binary labels `y` (0/1) and predicted probabilities `p` of class 1. Return the mean of `-(y*log(p) + (1-y)*log(1-p))`, clipping `p` to
# `[1e-15, 1 - 1e-15]` first.

# %% tags=["starter"]
def my_log_loss(y, p):
    ...

# %%
r_ = np.random.default_rng(3)
yy = r_.integers(0, 2, 200)
pp = np.clip(r_.random(200), 0, 1)
assert abs(my_log_loss(yy, pp) - log_loss(yy, pp)) < 1e-9, "should match sklearn.metrics.log_loss"
assert np.isfinite(my_log_loss(np.array([1, 0]), np.array([0.0, 1.0]))), "must not return inf for p in {0, 1}: clip first"
print("Exercise 1 passed")

# %% tags=["solution"]
def my_log_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

# %% [markdown]
# ### Exercise 2: Which constant does each loss pick?
# For the prices array (`prices`, thousand dollars), find by brute-force search over the grid `cands` the constant that minimises **Huber loss with d=50** (store in `best_huber`),
# and the constant minimising **MAE** (store in `best_mae`). Then store in `huber_between` whether `best_huber` lies between `best_mae` and `prices.mean()` (inclusive, allow grid error of 1).
# *Hint: use `huber_l(c - prices, 50).mean()` for each `c`.*

# %% tags=["starter"]
best_huber = ...
best_mae = ...
huber_between = ...

# %%
hub = [np.mean(huber_l(c - prices, 50.0)) for c in cands]
mae_ = [np.mean(np.abs(c - prices)) for c in cands]
assert abs(best_huber - cands[int(np.argmin(hub))]) < 1e-9, "best_huber should be the grid value with the smallest mean Huber loss"
assert abs(best_mae - cands[int(np.argmin(mae_))]) < 1e-9, "best_mae should be the grid value with the smallest mean absolute error"
lo, hi = sorted([best_mae, prices.mean()])
assert bool(huber_between) == bool(lo - 1 <= best_huber <= hi + 1), "compare best_huber with the interval [best_mae, mean]"
print(f"Exercise 2 passed: MAE->{best_mae:.1f}, Huber->{best_huber:.1f}, mean={prices.mean():.1f}  (between: {huber_between})")

# %% tags=["solution"]
best_huber = float(cands[int(np.argmin([np.mean(huber_l(c - prices, 50.0)) for c in cands]))])
best_mae = float(cands[int(np.argmin([np.mean(np.abs(c - prices)) for c in cands]))])
lo_, hi_ = sorted([best_mae, prices.mean()])
huber_between = bool(lo_ - 1 <= best_huber <= hi_ + 1)

# %% [markdown]
# ### Exercise 3: Code momentum yourself
# Write `steps_momentum(lr, beta)` that runs momentum descent on the elongated bowl (`f2`, `g2`, starting at `start`) and returns the number of steps needed until
# `f2 < 1e-6` (at most 5000; return 5000 if never). Use `v = beta * v + grad; p = p - lr * v`. Store `n_plain = steps_momentum(0.07, 0.0)` (beta 0 is plain descent) and
# `n_mom = steps_momentum(0.05, 0.8)`.

# %% tags=["starter"]
def steps_momentum(lr, beta):
    ...

n_plain = ...
n_mom = ...

# %%
assert steps_momentum(0.07, 0.0) == steps_to(lambda n: plain(0.07, start, n)), "with beta=0 your loop must equal plain descent"
assert n_plain == steps_momentum(0.07, 0.0) and n_mom == steps_momentum(0.05, 0.8)
assert n_mom < n_plain, f"momentum took {n_mom} steps vs plain {n_plain}; did you apply v = beta*v + grad?"
print(f"Exercise 3 passed: plain {n_plain} steps, momentum {n_mom} steps")

# %% tags=["solution"]
def steps_momentum(lr, beta):
    p = np.array(start, float)
    v = np.zeros(2)
    for i in range(5001):
        if f2(p) < 1e-6:
            return i
        v = beta * v + g2(p)
        p = p - lr * v
    return 5000

n_plain = steps_momentum(0.07, 0.0)
n_mom = steps_momentum(0.05, 0.8)

# %% [markdown]
# ### Exercise 4 (open-ended): Break the optimiser
# Keep beta = 0.8 and increase the momentum learning rate gradually (0.05, 0.08, 0.12, 0.15, 0.2, ...). At what value does it start to diverge on this bowl? Plain descent
# diverges above `2 / 25 = 0.08`. A known result for this momentum form on a quadratic is a limit of `2 * (1 + beta) / 25`. Does your experiment agree? Report in a markdown cell,
# and explain in one sentence why the *effective* step size (about `lr / (1 - beta)` in steady state) is nevertheless larger than plain descent's.

# %% [markdown]
# ## Recap
# - MSE estimates the mean and is outlier-sensitive; MAE estimates the median and is robust; Huber blends them.
# - Train on smooth surrogates (log-loss, hinge), report the real metric (accuracy, F1, revenue).
# - Log-loss punishes confident errors; always clip probabilities.
# - Convex losses have one basin; non-convex ones make the starting point and the optimiser matter.
# - Momentum and Adam help on badly scaled surfaces; none is universally best, so compare them on your problem.
#
# **Go deeper:** the PyTorch docs for `torch.optim` and the scikit-learn user guide "Metrics and scoring"; the scikit-learn docs for `HuberRegressor`.
#
# **Next:** [Probability and information theory basics](06_probability_and_information_theory_basics.ipynb)
