# %% [markdown]
# # Probability and Simulation
#
# **Why this matters.** Probability is how you reason about uncertain things: "what share of these customers will churn?", "if this fraud alert
# fires, how likely is it a real fraud?". Humans are notoriously bad at it (the medical-test example below fools many doctors). The fix is to
# *simulate*: when the algebra is confusing, generate a million fake cases and count. This notebook teaches the rules and gives you the
# Monte Carlo habit of checking every probabilistic claim by simulation.
#
# **You will learn**
# - interpret probability as a long-run frequency and see the law of large numbers
# - use complement, addition and multiplication rules, and test independence on real data
# - compute conditional probabilities and apply Bayes' rule correctly (and avoid base-rate neglect)
# - verify answers with Monte Carlo: birthday problem and Monty Hall
#
# **Prerequisites:** [Descriptive statistics](01_descriptive_statistics.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("../datasets")
rng = np.random.default_rng(42)
churn = pd.read_csv(DATA / "churn.csv")
churn["contract"] = churn["contract"].str.lower()

# %% [markdown]
# ## 1. Probability as a long-run frequency
#
# "The probability of heads is 0.5" means: flip many times and the fraction of heads settles near 0.5. This is the **law of large numbers**. Small samples wobble, large ones settle.

# %%
flips = rng.integers(0, 2, size=10_000)
running = np.cumsum(flips) / np.arange(1, len(flips) + 1)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(running)
ax.axhline(0.5, color="gray", ls="--")
ax.set_xscale("log"); ax.set_xlabel("number of flips (log scale)"); ax.set_ylabel("fraction of heads")
ax.set_title("The running average settles, slowly")
plt.show()
print("after 10 flips:", running[9], "| after 100:", running[99], "| after 10,000:", running[-1])

# %% [markdown]
# *What just happened?* Early on the fraction jumps around; by thousands of flips it is very close to 0.5. A related trap is the **gambler's fallacy**: the coin does not "owe" you tails after
# five heads. The average settles because new flips dilute old ones, not because the coin compensates.

# %% [markdown]
# ## 2. The basic rules, checked on real data
#
# An **event** is something that may or may not happen (a customer churns). `P(A)` is between 0 and 1.
#
# | Rule | Formula | Plain words |
# |---|---|---|
# | Complement | `P(not A) = 1 - P(A)` | it happens or it does not |
# | Addition | `P(A or B) = P(A) + P(B) - P(A and B)` | do not double-count the overlap |
# | Multiplication | `P(A and B) = P(A) * P(B given A)` | both happen |
# | Independence | `P(A and B) = P(A) * P(B)` | knowing A tells nothing about B |
#
# Here `A` = "customer churned" and `B` = "customer is on a month-to-month contract". Probabilities from data are just proportions.

# %%
A = churn["churned"] == 1
B = churn["contract"] == "month-to-month"

pA, pB, pAB = A.mean(), B.mean(), (A & B).mean()
print(f"P(A) = {pA:.3f}   P(B) = {pB:.3f}   P(A and B) = {pAB:.3f}")
print(f"P(A or B)   via rule: {pA + pB - pAB:.3f}   directly: {(A | B).mean():.3f}")
print(f"if independent, P(A and B) would be P(A)*P(B) = {pA * pB:.3f}  ->  they are NOT independent")

# %% [markdown]
# If churn and contract type were independent we would expect an overlap of `P(A) * P(B)`. The observed overlap is much larger, which says month-to-month customers churn more.
# (Whether a difference like this could be chance is the subject of the hypothesis-testing notebook.)

# %% [markdown]
# ## 3. Conditional probability
#
# `P(A | B)` ("probability of A *given* B") restricts attention to the world where `B` is true: `P(A | B) = P(A and B) / P(B)`. In data, that is just filtering, then averaging.

# %%
p_churn_given_mtm = pAB / pB
direct = churn.loc[B, "churned"].mean()
print(f"P(churn | month-to-month) = {p_churn_given_mtm:.3f}  (filter-and-average gives {direct:.3f})")
print(f"P(churn | other contract) = {churn.loc[~B, 'churned'].mean():.3f}")
print(f"P(month-to-month | churn) = {pAB / pA:.3f}   <- a DIFFERENT number")

# %% [markdown]
# `P(churn | month-to-month)` and `P(month-to-month | churn)` answer different questions. Confusing them (the "transposed conditional" or prosecutor's fallacy) is one of the most common reasoning errors;
# Bayes' rule is the tool that converts one into the other.

# %% [markdown]
# ## 4. Bayes' rule and base-rate neglect
#
# `P(A | B) = P(B | A) * P(A) / P(B)`.
#
# **The puzzle.** A disease affects 1% of people. A test catches 90% of sick people (sensitivity) and wrongly flags 5% of healthy people (false-positive rate). You test positive. What is the probability you are sick?
# Most people say "about 90%". Compute it:

# %%
prev, sens, fpr = 0.01, 0.90, 0.05
p_pos = sens * prev + fpr * (1 - prev)            # total probability of a positive test
ppv = sens * prev / p_pos                          # Bayes: P(sick | positive)
print(f"P(positive) = {p_pos:.4f}    P(sick | positive) = {ppv:.3f}")

# %% [markdown]
# Only about 15%. Think in **natural frequencies**: out of 10,000 people, 100 are sick and 90 of them test positive; 9,900 are healthy and 495 of them test positive. So among 585 positives, only 90 are sick.
# The mistake (**base-rate neglect**) is ignoring that healthy people vastly outnumber sick ones, so even a small false-positive rate produces more false alarms than true cases. Verify by simulation rather than trusting the algebra:

# %%
n = 1_000_000
sick = rng.random(n) < prev
test_pos = np.where(sick, rng.random(n) < sens, rng.random(n) < fpr)
print(f"simulated P(sick | positive) = {sick[test_pos].mean():.3f}   (formula: {ppv:.3f})")
print(f"positives: {test_pos.sum():,}, of which truly sick: {(sick & test_pos).sum():,}")

# %%
prevs = np.logspace(-3, np.log10(0.5), 100)
ppvs = sens * prevs / (sens * prevs + fpr * (1 - prevs))
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(prevs, ppvs)
ax.set_xscale("log"); ax.set_xlabel("disease prevalence (log scale)"); ax.set_ylabel("P(sick | positive test)")
ax.set_title("The same test is useless at low prevalence, useful at high")
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# *What just happened?* The test never changed, yet its meaning did: the answer depends on how common the condition is. The same logic applies to fraud alerts, spam filters and a "churn risk" flag when churners are rare in the list being scored.

# %% [markdown]
# ## 5. Monte Carlo: when the algebra is hard, count
#
# **Monte Carlo** = estimate a probability by simulating the random process many times and taking the fraction of hits. The error shrinks like `1/sqrt(N)`: 100x more runs buys 10x more accuracy.
#
# **Birthday problem.** In a room of `n` people, what is the chance two share a birthday (ignoring leap days)? Exact answer: `1 - (365/365)*(364/365)*...*((365-n+1)/365)`.

# %%
def birthday_exact(n):
    p_all_different = np.prod((365 - np.arange(n)) / 365)
    return 1 - p_all_different

def birthday_sim(n, trials=20_000):
    days = rng.integers(0, 365, size=(trials, n))
    # a row has a match if the number of distinct days is less than n
    has_match = np.array([len(set(row)) < n for row in days])
    return has_match.mean()

for n_people in [10, 23, 40]:
    print(f"n={n_people:2d}  exact = {birthday_exact(n_people):.3f}   simulated = {birthday_sim(n_people):.3f}")

# %% [markdown]
# With only 23 people the chance already passes 50%, far lower than intuition suggests, because you are comparing all `23*22/2 = 253` pairs, not one person against the rest.
#
# **Monty Hall.** Three doors, one car. You pick a door; the host (who knows where the car is) opens a *different* door showing a goat and offers a switch. Should you?

# %%
def monty(switch, trials=50_000):
    car = rng.integers(0, 3, trials)
    pick = rng.integers(0, 3, trials)
    wins = 0
    for c, p in zip(car, pick):
        opened = next(d for d in range(3) if d != p and d != c)     # host opens a goat door that is not your pick
        final = next(d for d in range(3) if d != p and d != opened) if switch else p
        wins += final == c
    return wins / trials

print(f"stay  : {monty(False):.3f}   (theory 1/3)")
print(f"switch: {monty(True):.3f}   (theory 2/3)")

# %% [markdown]
# Staying wins only when your first guess was right (1/3). Switching wins whenever your first guess was wrong (2/3), because the host's reveal concentrates the remaining probability on the other door.
# Many statisticians initially rejected this answer; the simulation is what settles arguments.

# %% [markdown]
# ### How accurate is a Monte Carlo estimate?
# Estimate `P(two dice sum to 7) = 6/36` with more and more trials:

# %%
truth = 6 / 36
print(f"{'trials':>10} {'estimate':>9} {'abs error':>10}")
for trials in [100, 10_000, 1_000_000]:
    d = rng.integers(1, 7, size=(trials, 2)).sum(axis=1)
    est = (d == 7).mean()
    print(f"{trials:10,d} {est:9.4f} {abs(est - truth):10.5f}")

# %% [markdown]
# ## Common mistakes
#
# 1. **Transposing the conditional**: `P(A|B)` is not `P(B|A)`. Most 'the test is 95% accurate, so I am 95% likely to be sick' errors are this.
# 2. **Ignoring the base rate** (prevalence) in Bayes problems.
# 3. **Assuming independence** without checking: `P(A and B) = P(A)P(B)` only holds for independent events.
# 4. **The gambler's fallacy**: independent trials have no memory.
# 5. **Simulating with too few trials** and trusting digits that are pure noise; quote the `1/sqrt(N)` scale of error.

# %%
# Demonstrating mistake 5: ten repeats of a 100-trial estimate of a 1/6 probability
est = [(rng.integers(1, 7, 100) == 6).mean() for _ in range(10)]
print("ten estimates of P(roll a six) using 100 rolls each:", np.round(est, 2), "(truth 0.167)")

# %% [markdown]
# ## Interview corner
#
# **1. A test is 99% accurate and you test positive for a rare disease (1 in 10,000). How worried should you be?**
# <details><summary>Answer</summary>
# Use Bayes with natural frequencies. In 1,000,000 people, about 100 are sick and about 99 are detected; of the 999,900 healthy, about 1% (roughly 10,000) are false positives. So only around 1% of positives are truly sick. The result depends on what "99% accurate" means (sensitivity vs specificity) and on the base rate.
# </details>
#
# **2. Explain the Monty Hall problem.**
# <details><summary>Answer</summary>
# Your first pick is right 1/3 of the time. The host's reveal does not change that, so the other door holds the remaining 2/3. Switching wins whenever the first guess was wrong. It relies on the host knowing where the car is and always opening a goat door.
# </details>
#
# **3. What does it mean for two events to be independent, and how would you check it in data?**
# <details><summary>Answer</summary>
# Knowing one does not change the probability of the other: P(A and B) = P(A)P(B), equivalently P(A|B) = P(A). In data compare the joint frequency with the product of the marginals, or test with a chi-square test of independence (covered later).
# </details>
#
# **4. Why use Monte Carlo simulation?**
# <details><summary>Answer</summary>
# When an exact calculation is hard or error-prone, simulating the process and counting gives an estimate with error that shrinks like 1/sqrt(N). It is also a great way to test analytic answers and to explore "what if" questions.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Conditional churn and lift
# Compute `p_senior`, the probability of churn **given** `senior_citizen == 1`, and `lift_senior = p_senior / P(churn)` (the overall churn rate). Store whether seniors churn more than average in `seniors_higher`.

# %% tags=["starter"]
p_senior = ...
lift_senior = ...
seniors_higher = ...

# %%
exp_p = churn.loc[churn["senior_citizen"] == 1, "churned"].mean()
assert abs(p_senior - exp_p) < 1e-9, "p_senior should be the churn rate among senior_citizen == 1"
assert abs(lift_senior - exp_p / churn["churned"].mean()) < 1e-9, "lift = conditional rate / overall rate"
assert bool(seniors_higher) == (exp_p > churn["churned"].mean())
print(f"Exercise 1 passed: P(churn|senior) = {p_senior:.3f}, lift = {lift_senior:.2f}")

# %% tags=["solution"]
p_senior = churn.loc[churn["senior_citizen"] == 1, "churned"].mean()
lift_senior = p_senior / churn["churned"].mean()
seniors_higher = bool(p_senior > churn["churned"].mean())

# %% [markdown]
# ### Exercise 2: A fraud alert
# A fraud model flags 2% of all transactions. Suppose 0.5% of transactions are actually fraud, the model catches 80% of the frauds (sensitivity) and has a false-positive rate `fpr`
# you must derive: the flagged share must come out to exactly 2%. Write `ppv_from(prev, sens, fpr)` (Bayes) and solve for `fpr_needed` algebraically, then compute `precision = ppv_from(0.005, 0.8, fpr_needed)`.
# *Hint: `P(flag) = sens*prev + fpr*(1-prev) = 0.02`.*

# %% tags=["starter"]
def ppv_from(prev, sens, fpr):
    ...

fpr_needed = ...
precision = ...

# %%
assert abs(ppv_from(0.01, 0.9, 0.05) - 0.9 * 0.01 / (0.9 * 0.01 + 0.05 * 0.99)) < 1e-12, "check your Bayes formula"
assert abs(0.8 * 0.005 + fpr_needed * 0.995 - 0.02) < 1e-9, "fpr_needed should make P(flag) equal 0.02"
# independent check by simulation
sim_n = 2_000_000
r_ = np.random.default_rng(7)
fraud = r_.random(sim_n) < 0.005
flag = np.where(fraud, r_.random(sim_n) < 0.8, r_.random(sim_n) < fpr_needed)
assert abs(flag.mean() - 0.02) < 0.001, "simulated flag rate should be about 2%"
assert abs(precision - fraud[flag].mean()) < 0.01, f"precision {precision:.3f} disagrees with simulation {fraud[flag].mean():.3f}"
print(f"Exercise 2 passed: fpr = {fpr_needed:.4f}, precision = {precision:.3f}")

# %% tags=["solution"]
def ppv_from(prev, sens, fpr):
    return sens * prev / (sens * prev + fpr * (1 - prev))

fpr_needed = (0.02 - 0.8 * 0.005) / (1 - 0.005)
precision = ppv_from(0.005, 0.8, fpr_needed)

# %% [markdown]
# ### Exercise 3: How many people for a 50% birthday match?
# Using `birthday_exact`, find the smallest `n_half` with probability at least 0.5. Then find `n_95`, the smallest group size with probability at least 0.95.

# %% tags=["starter"]
n_half = ...
n_95 = ...

# %%
probs = [birthday_exact(k) for k in range(1, 100)]
assert n_half == 1 + next(i for i, p in enumerate(probs) if p >= 0.5), "smallest n where P >= 0.5"
assert n_95 == 1 + next(i for i, p in enumerate(probs) if p >= 0.95), "smallest n where P >= 0.95"
assert abs(birthday_sim(n_half, 30_000) - birthday_exact(n_half)) < 0.02, "sanity check vs simulation"
print(f"Exercise 3 passed: {n_half} people for 50%, {n_95} for 95%")

# %% tags=["solution"]
n_half = next(k for k in range(1, 366) if birthday_exact(k) >= 0.5)
n_95 = next(k for k in range(1, 366) if birthday_exact(k) >= 0.95)

# %% [markdown]
# ### Exercise 4 (open-ended): A harder Monty Hall
# Change `monty` to 100 doors where the host opens 98 goat doors after your pick. Simulate stay vs switch and report the win probabilities. Predict the answer first with the reasoning
# "my first pick is right with probability 1/100", and write in a markdown cell whether the simulation confirms it. What if the host opens a random unpicked door and happens to reveal a goat? Does the answer change? (Try it.)

# %% [markdown]
# ## Recap
# - Probability is long-run frequency; the law of large numbers explains why samples settle, and the gambler's fallacy is misunderstanding it.
# - Use complement, addition (subtract the overlap), multiplication (conditional) and compare `P(A and B)` with `P(A)P(B)` to check independence.
# - `P(A|B)` is not `P(B|A)`; Bayes' rule converts between them and needs the base rate. Natural frequencies make it intuitive.
# - Monte Carlo estimates probabilities by counting simulated outcomes; its error shrinks like `1/sqrt(N)`. Use it to verify algebra.
#
# **Go deeper:** the "Probability" chapters of the free book "Think Bayes" by Allen Downey; NumPy's `numpy.random.Generator` documentation.
#
# **Next:** [Probability distributions](03_distributions.ipynb)
