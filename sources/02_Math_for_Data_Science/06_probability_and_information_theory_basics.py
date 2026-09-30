# %% [markdown]
# # Probability and Information Theory Basics: Entropy, Cross-Entropy, KL
#
# **Why this matters.** The words "cross-entropy loss", "information gain" and "KL divergence" appear in logistic regression, decision trees, neural
# networks, t-SNE and variational models. They are all the same idea: *how surprised is a model by what actually happened?* Once you see that, log-loss stops
# being a formula to memorise and a decision tree's split criterion stops being a mystery.
#
# **You will learn**
# - measure surprise (`-log p`) and average it into **entropy**
# - compute cross-entropy and KL divergence and see that log-loss *is* cross-entropy
# - compute **information gain** for a decision-tree split on the churn data
# - connect minimising log-loss to maximum likelihood
#
# **Prerequisites:** [Loss functions and optimization](05_loss_functions_and_optimization.ipynb). The next module, statistics, goes deeper on probability itself.
#
# > Data used: `datasets/churn.csv` (synthetic telecom customers, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)
DATA = Path("../datasets")
churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn["contract"].str.lower()      # repair the planted casing inconsistency (details in the pandas module)

# %% [markdown]
# ## 1. Surprise: rare events carry more information
#
# If someone tells you "the sun rose today", you learn nothing. If they tell you "it snowed in the desert", you learn a lot. Define the **surprise** (information content)
# of an event with probability `p` as `-log2(p)` **bits**. Probability 1/2 -> 1 bit, 1/8 -> 3 bits, 1 -> 0 bits. Using `log2` gives bits; using the natural log gives *nats*
# (what most ML libraries use). Only the unit changes.

# %%
for p in [1.0, 0.5, 0.125, 0.01]:
    print(f"p = {p:<6}  surprise = {abs(np.log2(p)):.2f} bits")

# %% [markdown]
# ## 2. Entropy: average surprise
#
# **Entropy** `H(p) = -sum(p_i * log2(p_i))` is the *expected* surprise of a random outcome: how unpredictable the distribution is. It is highest for a uniform distribution
# and 0 when the outcome is certain. For a coin with heads-probability `p`:

# %%
def entropy(p, base=2):
    p = np.asarray(p, float)
    p = p[p > 0]                                    # convention: 0 * log 0 = 0
    return float(-(p * np.log(p)).sum() / np.log(base))

ps = np.linspace(0.001, 0.999, 300)
H = [entropy([q, 1 - q]) for q in ps]
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(ps, H)
ax.set_xlabel("probability of heads"); ax.set_ylabel("entropy (bits)"); ax.set_title("A fair coin is the most unpredictable")
ax.grid(alpha=0.3)
plt.show()
print("fair coin:", entropy([0.5, 0.5]), "bits;  99%-biased coin:", round(entropy([0.99, 0.01]), 3), "bits")

# %% [markdown]
# Now on data. How unpredictable is a customer's churn status, and their contract type?

# %%
p_churn = churn["churned"].value_counts(normalize=True).sort_index()
p_contract = churn["contract"].value_counts(normalize=True)
print("churn distribution:\n", p_churn.round(3).to_string())
print(f"entropy of churn label : {entropy(p_churn.to_numpy()):.3f} bits (max possible for 2 classes = 1)")
print(f"entropy of contract    : {entropy(p_contract.to_numpy()):.3f} bits (max for 3 classes = {np.log2(3):.3f})")

# %% [markdown]
# *What just happened?* Churn is about 37% / 63%, close to a fair coin, so the label carries nearly 1 bit of uncertainty per customer. A good model should reduce that uncertainty
# using features: this is exactly what a decision tree measures.

# %% [markdown]
# ## 3. Cross-entropy: surprise under the wrong model
#
# You believe outcomes follow `q` (your model) but they really follow `p` (reality). **Cross-entropy** `H(p, q) = -sum(p_i * log q_i)` is the average surprise you experience.
# It is never smaller than the true entropy `H(p)`; the gap is the price of being wrong.

# %%
def cross_entropy(p, q, base=2):
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = p > 0
    return float(-(p[mask] * np.log(q[mask])).sum() / np.log(base))

p_true = np.array([0.7, 0.2, 0.1])
for name, q in [("perfect model", p_true), ("vague (uniform)", np.ones(3) / 3), ("confidently wrong", np.array([0.05, 0.05, 0.9]))]:
    print(f"{name:18s} H(p,q) = {cross_entropy(p_true, q):.3f} bits   (H(p) = {entropy(p_true):.3f})")

# %% [markdown]
# ### Log-loss is cross-entropy
# For classification, the "true distribution" for one sample is one-hot (probability 1 on the actual class). Then `H(p, q) = -log q[true class]`, and averaging over samples is exactly
# **log-loss** (in nats with the natural log). Check it against scikit-learn:

# %%
from sklearn.metrics import log_loss

rng = np.random.default_rng(42)
y = rng.integers(0, 2, 500)
q1 = np.clip(0.5 * y + 0.25 + rng.normal(0, 0.15, 500), 0.02, 0.98)       # a mediocre model's P(class 1)

manual = np.mean([-np.log(q1[i] if y[i] == 1 else 1 - q1[i]) for i in range(len(y))])
print(f"manual cross-entropy (nats): {manual:.5f}   sklearn log_loss: {log_loss(y, q1):.5f}")

# %% [markdown]
# ## 4. KL divergence: the extra cost of the wrong model
#
# `KL(p || q) = H(p, q) - H(p) = sum(p_i * log(p_i / q_i))`: the *additional* bits you pay because you used `q` instead of the truth `p`. It is zero only when `q == p`,
# never negative, and **not symmetric** (`KL(p||q) != KL(q||p)`), so it is a divergence, not a distance.

# %%
def kl(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

p1 = np.array([0.7, 0.2, 0.1]); p2 = np.array([0.4, 0.4, 0.2])
print(f"KL(p1||p2) = {kl(p1, p2):.4f}   KL(p2||p1) = {kl(p2, p1):.4f}   KL(p1||p1) = {kl(p1, p1):.4f}")

from scipy.stats import entropy as scipy_entropy
print("scipy check:", round(scipy_entropy(p1, p2, base=2), 4), "(scipy.stats.entropy with two arguments is KL)")

# %% [markdown]
# **Why this matters for training.** When you minimise cross-entropy against fixed data, `H(p)` is a constant, so you are *also* minimising `KL(p || q)`: pushing your model's
# distribution towards the data's. That is why cross-entropy is the natural loss for classifiers.

# %% [markdown]
# ## 5. Maximum likelihood = minimum log-loss
#
# Suppose you flip a coin 20 times and see 14 heads. The **likelihood** of a heads-probability `p` is `p^14 * (1-p)^6`. Maximum likelihood picks the `p` that makes the data
# most probable. Taking `-log` turns the product into a sum and the maximum into a minimum: the negative log-likelihood is exactly the total log-loss.

# %%
heads, n = 14, 20
grid = np.linspace(0.01, 0.99, 981)
nll = -(heads * np.log(grid) + (n - heads) * np.log(1 - grid))       # negative log-likelihood = total log-loss
best = grid[np.argmin(nll)]
print(f"p minimising the negative log-likelihood: {best:.3f}   (heads / n = {heads / n:.3f})")

# %% [markdown]
# So fitting a model by minimising log-loss is the same as choosing the parameters under which the observed labels were most probable. The same logic gives MSE as the
# maximum-likelihood loss when errors are assumed to be Gaussian.

# %% [markdown]
# ## 6. Information gain: how a decision tree picks a split
#
# A tree wants a question ("is the contract month-to-month?") whose answer leaves the *children* less uncertain than the *parent*. Define
#
# `information gain = H(parent) - weighted average of H(children)`.
#
# Weights are the fraction of samples going to each child.

# %%
def split_gain(labels, mask):
    """Information gain (bits) of splitting `labels` (0/1 array) by boolean `mask`."""
    labels = np.asarray(labels); mask = np.asarray(mask, bool)
    def h(lab):
        return entropy(np.bincount(lab, minlength=2) / len(lab)) if len(lab) else 0.0
    w = mask.mean()
    return h(labels) - (w * h(labels[mask]) + (1 - w) * h(labels[~mask]))

yc = churn["churned"].to_numpy()
mtm = (churn["contract"] == "month-to-month").to_numpy()
print(f"P(churn | month-to-month) = {yc[mtm].mean():.3f},  P(churn | other) = {yc[~mtm].mean():.3f}")
print(f"H(parent) = {entropy(np.bincount(yc) / len(yc)):.4f} bits,  information gain = {split_gain(yc, mtm):.4f} bits")

# %% [markdown]
# Try several candidate questions and rank them:

# %%
candidates = {
    "contract is month-to-month": mtm,
    "support_calls >= 3": (churn["support_calls"] >= 3).to_numpy(),
    "tenure_months < 12": (churn["tenure_months"] < 12).to_numpy(),
    "senior_citizen == 1": (churn["senior_citizen"] == 1).to_numpy(),
    "autopay == Yes": (churn["autopay"] == "Yes").to_numpy(),
}
gains = pd.Series({k: split_gain(yc, v) for k, v in candidates.items()}).sort_values(ascending=False)
gains.round(4)

# %% [markdown]
# A tree greedily picks the top question, then repeats inside each child. Compare with scikit-learn's one-question tree (a "stump") using the entropy criterion; its root split should
# agree with our ranking among these candidates:

# %%
from sklearn.tree import DecisionTreeClassifier

Xc = pd.DataFrame({k: v.astype(int) for k, v in candidates.items()})
stump = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=42).fit(Xc, yc)
print("sklearn's chosen feature:", Xc.columns[stump.tree_.feature[0]])
print("our best candidate      :", gains.index[0])

# %% [markdown]
# Trees often use **Gini impurity** `1 - sum(p_i^2)` instead of entropy. It is cheaper (no logarithm) and in practice picks nearly the same splits; both are zero for a pure node and maximal for a 50/50 mix.

# %% [markdown]
# ## Common mistakes
#
# 1. **`log(0)`** in hand-written entropy/log-loss: clip probabilities or skip zero-probability terms.
# 2. **Mixing bits and nats** when comparing numbers from different sources (`log2` vs `ln`). They differ by the constant factor `ln 2`.
# 3. **Treating KL as symmetric or as a distance.** `KL(p||q)` and `KL(q||p)` differ, and KL is infinite if `q` gives zero probability where `p` does not.
# 4. **Reading a high information gain as "important feature" without care.** Gain favours features with many distinct values (an ID column splits perfectly). That is why trees need depth limits, pruning, or gain ratio.
# 5. **Confusing the probabilities you *predict* with the labels.** Cross-entropy takes probabilities; passing hard 0/1 predictions makes it explode on every mistake.

# %%
id_like = np.arange(len(yc))                           # a unique id per customer
mask_first_half = id_like < len(yc) // 2
print(f"gain of a meaningless 'first half of the file' split: {split_gain(yc, mask_first_half):.5f} bits (near 0: good)")
print("but splitting on a unique-per-row id would reach zero child entropy: gain =", round(entropy(np.bincount(yc) / len(yc)), 4), "bits (the maximum, and useless)")

# %% [markdown]
# ## Interview corner
#
# **1. Why is cross-entropy the standard loss for classification?**
# <details><summary>Answer</summary>
# It is the negative log-likelihood of the labels under the model, so minimising it is maximum likelihood; it equals entropy plus KL to the data distribution, so it pushes predicted probabilities towards the truth;
# and it is convex for logistic regression and has well-behaved gradients (`p - y` for a sigmoid/softmax output).
# </details>
#
# **2. What is KL divergence and is it a distance?**
# <details><summary>Answer</summary>
# `KL(p||q)` is the expected extra surprise from using `q` when the truth is `p`. It is non-negative and zero only when the distributions match, but asymmetric and it violates the triangle inequality, so it is not a metric.
# </details>
#
# **3. How does a decision tree choose a split?**
# <details><summary>Answer</summary>
# For each candidate feature/threshold it computes the impurity drop (entropy-based information gain or Gini decrease) from parent to size-weighted children and takes the largest, greedily, without looking ahead.
# </details>
#
# **4. Entropy of a fair six-sided die versus a fair coin?**
# <details><summary>Answer</summary>
# `log2(6)` is about 2.58 bits versus 1 bit: more equally likely outcomes means more uncertainty. Uniform over `k` outcomes has entropy `log2(k)`.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Entropy from raw counts
# Write `entropy_from_counts(counts)` returning the entropy in bits of a distribution given as counts (e.g. `[30, 70]`). Then store in `h_internet` the entropy of the
# `internet_service` column of `churn`, treating missing values as their own category (`value_counts(dropna=False)`).

# %% tags=["starter"]
def entropy_from_counts(counts):
    ...

h_internet = ...

# %%
assert abs(entropy_from_counts([1, 1]) - 1.0) < 1e-12, "a 50/50 split has 1 bit"
assert abs(entropy_from_counts([5, 0, 0]) - 0.0) < 1e-12, "a certain outcome has 0 bits (watch out for 0 * log 0)"
assert abs(entropy_from_counts([1, 1, 1, 1]) - 2.0) < 1e-12, "4 equally likely outcomes: 2 bits"
vc = churn["internet_service"].value_counts(dropna=False).to_numpy()
assert abs(h_internet - scipy_entropy(vc, base=2)) < 1e-9, "did you include the missing values as a category?"
print(f"Exercise 1 passed: H(internet_service) = {h_internet:.3f} bits")

# %% tags=["solution"]
def entropy_from_counts(counts):
    c = np.asarray(counts, float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())

h_internet = entropy_from_counts(churn["internet_service"].value_counts(dropna=False).to_numpy())

# %% [markdown]
# ### Exercise 2: Cross-entropy of a classifier vs the base rate
# Using `q1`/`y` from section 3: store in `ce_model` the mean log-loss (nats) of `q1`, and in `ce_baseline` the mean log-loss of a "know-nothing" model that predicts
# the training base rate `y.mean()` for every sample. Store `model_beats_baseline` as a bool. *Hint: `sklearn.metrics.log_loss` accepts a constant array.*

# %% tags=["starter"]
ce_model = ...
ce_baseline = ...
model_beats_baseline = ...

# %%
assert abs(ce_model - log_loss(y, q1)) < 1e-9, "ce_model = log_loss(y, q1)"
assert abs(ce_baseline - log_loss(y, np.full(len(y), y.mean()))) < 1e-9, "baseline predicts y.mean() for everyone"
assert bool(model_beats_baseline) == (ce_model < ce_baseline), "lower log-loss is better"
print(f"Exercise 2 passed: model {ce_model:.3f} vs baseline {ce_baseline:.3f}")

# %% tags=["solution"]
ce_model = float(log_loss(y, q1))
ce_baseline = float(log_loss(y, np.full(len(y), y.mean())))
model_beats_baseline = bool(ce_model < ce_baseline)

# %% [markdown]
# ### Exercise 3: Information gain of an internet-service split
# Compute the information gain (bits) for splitting churn by `internet_service == "Fiber"` (missing values go with the non-Fiber side) and store it in `gain_fiber`.
# Then store in `beats_autopay` whether it is larger than the gain of the `autopay == "Yes"` split. *Hint: reuse `split_gain`.*

# %% tags=["starter"]
gain_fiber = ...
beats_autopay = ...

# %%
mask_f = (churn["internet_service"] == "Fiber").to_numpy()
# independent computation with scipy
def _h(lab): return scipy_entropy(np.bincount(lab, minlength=2), base=2)
w_ = mask_f.mean()
expected = _h(yc) - w_ * _h(yc[mask_f]) - (1 - w_) * _h(yc[~mask_f])
assert abs(gain_fiber - expected) < 1e-9, f"expected {expected:.5f}; make sure missing values are on the non-Fiber side (use == 'Fiber', not != 'DSL')"
assert bool(beats_autopay) == (expected > split_gain(yc, (churn['autopay'] == 'Yes').to_numpy())), "compare with the autopay gain"
print(f"Exercise 3 passed: gain {gain_fiber:.4f} bits (autopay: {split_gain(yc, (churn['autopay'] == 'Yes').to_numpy()):.4f})")

# %% tags=["solution"]
mask_fiber = (churn["internet_service"] == "Fiber").to_numpy()
gain_fiber = split_gain(yc, mask_fiber)
beats_autopay = bool(gain_fiber > split_gain(yc, (churn["autopay"] == "Yes").to_numpy()))

# %% [markdown]
# ### Exercise 4 (open-ended): Best threshold for `tenure_months`
# Scan thresholds `t` from 1 to 72 for the question `tenure_months < t`, compute the information gain of each with `split_gain`, plot gain against `t`, and report the best `t` and its gain
# in a markdown cell. Then run `DecisionTreeClassifier(max_depth=1, criterion="entropy")` on `tenure_months` alone and compare its threshold with yours.

# %% [markdown]
# ## Recap
# - Surprise is `-log p`; entropy is average surprise; uniform distributions maximise it.
# - Cross-entropy is surprise under a wrong model; KL is the extra cost over the true entropy; log-loss is cross-entropy against one-hot labels.
# - Minimising log-loss is maximum likelihood.
# - Information gain (entropy drop from parent to weighted children) is how trees choose splits; beware features with many distinct values.
#
# **Go deeper:** scikit-learn user guide "Decision Trees" (mathematical formulation section) and `scipy.stats.entropy`. The classic text "Elements of Information Theory" by Cover and Thomas is the standard reference.
#
# **Next:** you have finished the math foundation. Continue with [Statistics: descriptive statistics](../03_Statistics_and_Probability/01_descriptive_statistics.ipynb).
