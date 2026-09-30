# %% [markdown]
# # Capstone 1: Problem Framing and Success Criteria
#
# **Why this matters.** Most failed data science projects did not fail on the model; they answered a question
# nobody asked. A retention manager does not want "a model with 0.82 AUC". She wants to know *whom to call this
# week, how many calls that is, and what it is worth*. Turning that into a metric, a baseline and a definition of
# done is the first hour of the project, and it decides whether the last hour is a demo or a product.
#
# **You will learn**
# - translate a business request into a prediction target, a decision and a cost matrix
# - derive the decision threshold from costs instead of defaulting to 0.5
# - compute "do nothing" and "contact everyone" baselines before modelling
# - write a definition of done that a pipeline can check automatically
#
# **Prerequisites:** [Pandas](../05_Pandas/), [Machine Learning](../12_Machine_Learning/) and [MLOps](../20_MLOps/) basics.
#
# > **The capstone.** Over seven notebooks you will build a *Customer Churn Prediction Service*: clean data, a
# > leakage-safe model, a cost-based threshold, a tested API, a drift monitor, and the Docker / Airflow / DVC / CI
# > files around it. The real code lives in `src/churn/` (with tests in `tests/`); the notebooks teach it and make you
# > extend it. Data: `datasets/churn.csv`, synthetic, 5,000 telecom customers, `churned` = 1 if they left.

# %% [markdown]
# ## The request
#
# > *"We lose too many customers. Can you build us a churn model?"*
#
# That sentence hides four decisions you must pin down **before** touching data:
#
# | Question | Our answer for this project |
# |---|---|
# | What exactly is the prediction? | Will this customer cancel (`churned = 1`)? One row per customer. |
# | What decision does it drive? | Who gets a retention offer (a discount call) this month. |
# | What does a wrong decision cost? | See the cost matrix below. |
# | How will it be used? | A batch list for the retention team, plus an API a CRM can call per customer. |

# %%
import sys
import tempfile
from pathlib import Path

SRC = Path("../src").resolve()  # the package is not pip-installed: import it straight from src/
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn import train
from churn.config import load_config, resolve
from churn.data import TARGET, load_dataset

CFG = load_config("../configs/config.yaml")
df = load_dataset(resolve(CFG, CFG["paths"]["data"]))
print(f"{len(df):,} customers, churn rate {df[TARGET].mean():.1%}")

# %% [markdown]
# `load_dataset` is our own function (notebook 02 opens it up): it loads, cleans and **validates** the file, and
# raises a readable error if anything breaks the schema. From here on we work with the validated frame.

# %% [markdown]
# ## 1. A cost matrix in plain money
#
# A classifier makes four kinds of outcomes. Each has a price. We invent illustrative numbers (you would get real
# ones from finance) and, importantly, **write the derivation down**:
#
# - A retention offer costs **20** (a discount plus the agent's time).
# - A customer who leaves is worth **200** in future margin.
# - **40%** of churners who receive an offer stay.

# %%
offer_cost, churner_value, save_rate = 20.0, 200.0, 0.40

cost_matrix = {
    "tn": 0.0,                                          # loyal customer, we did nothing: free
    "fp": offer_cost,                                   # loyal customer, we paid for an offer they did not need
    "fn": churner_value,                                # churner, we did nothing: lose the customer
    "tp": offer_cost + (1 - save_rate) * churner_value  # churner, offer paid, but 60% still leave
}
print(cost_matrix)
assert cost_matrix == CFG["costs"], "configs/config.yaml should hold exactly this matrix"

# %% [markdown]
# *What just happened?* The four numbers in `configs/config.yaml` are not magic: they come from three business
# assumptions. When finance updates `save_rate`, you change one line and the whole pipeline (threshold, metrics,
# model card) follows. The `assert` confirms the notebook and the config agree.
#
# Note the **true positive is not free**: contacting a churner costs the offer, and 60% leave anyway. The model
# only pays off because losing the customer (200) is much worse than the expected cost of trying (140).

# %% [markdown]
# ## 2. The break-even probability
#
# Suppose the model says a customer leaves with probability `p`. Should we contact them?
# - Contact: expected cost `p * tp + (1 - p) * fp`
# - Do nothing: expected cost `p * fn + (1 - p) * tn`
#
# Contact when the first is lower. Solving for `p` gives a threshold that depends **only on the costs**:
#
# `threshold = (fp - tn) / ((fp - tn) + (fn - tp))`

# %%
def break_even(costs: dict) -> float:
    return (costs["fp"] - costs["tn"]) / ((costs["fp"] - costs["tn"]) + (costs["fn"] - costs["tp"]))


p_star = break_even(cost_matrix)
print(f"contact anyone with P(churn) above {p_star:.2f}  (not 0.50)")
assert p_star == train.theoretical_threshold(cost_matrix)  # the package implements the same formula

# %%
p = np.linspace(0, 1, 101)
contact = p * cost_matrix["tp"] + (1 - p) * cost_matrix["fp"]
ignore = p * cost_matrix["fn"] + (1 - p) * cost_matrix["tn"]
plt.figure(figsize=(6, 3.5))
plt.plot(p, contact, label="contact (offer)")
plt.plot(p, ignore, label="do nothing")
plt.axvline(p_star, color="grey", linestyle="--", label=f"break-even p = {p_star:.2f}")
plt.xlabel("predicted probability of churn")
plt.ylabel("expected cost per customer (currency units)")
plt.title("Who should get an offer?")
plt.legend()
plt.show()

# %% [markdown]
# Below the dashed line, doing nothing is cheaper; above it, contacting is. **This only holds if the model's
# probabilities are well calibrated** ("70% really means 7 in 10"). If they are not, the best threshold drifts away
# from the formula, which is why the training code chooses the threshold empirically on a validation set and
# reports the formula value next to it. Notebook 04 compares the two.

# %% [markdown]
# ## 3. Baselines: what must the model beat?
#
# A model is only good *relative to something simpler*. Before modelling, price three dumb policies on all the data.

# %%
y = df[TARGET].to_numpy()
policies = {
    "contact nobody": np.zeros(len(df), dtype=bool),
    "contact everybody": np.ones(len(df), dtype=bool),
    "contact month-to-month customers": (df["contract"] == "Month-to-month").to_numpy(),
}
rows = []
for name, flag in policies.items():
    cm = train.confusion(y, flag)
    rows.append({"policy": name, "flag_rate": flag.mean(),
                 "recall": cm["tp"] / (cm["tp"] + cm["fn"]),
                 "cost_per_customer": train.cost_per_customer(y, flag, cost_matrix)})
baselines = pd.DataFrame(rows).set_index("policy")
baselines.round(3)

# %% [markdown]
# Read this table carefully, it is the project's bar:
# - **Accuracy is useless here.** Predicting "nobody churns" is right for about `1 - churn rate` of customers
#   (computed below) yet catches zero churners.
# - A one-line business rule ("call the month-to-month customers") is already a serious competitor.
#   A model that cannot beat it does not justify its maintenance cost.

# %%
print(f"'nobody churns' accuracy: {1 - y.mean():.1%}")
best_rule = baselines["cost_per_customer"].idxmin()
print(f"cheapest dumb policy: {best_rule} at {baselines.loc[best_rule, 'cost_per_customer']:.2f} per customer")

# %% [markdown]
# ## 4. Metrics: what we report and why
#
# | Metric | Why we use it | Why it is not enough alone |
# |---|---|---|
# | **ROC AUC** | Threshold-free ranking quality; used to pick between models | Says nothing about cost or calibration |
# | **Average precision** | Focuses on the positive class | Depends on the churn rate |
# | **Cost per customer at the chosen threshold** | The business metric | Only as good as the cost matrix |
# | **Recall / precision at that threshold** | What the retention team experiences | Move with the threshold |
# | **Brier score + calibration curve** | Are the probabilities honest? | Needs enough data per bin |
# | **Subgroup metrics** (contract, senior citizen) | Is performance uneven across customers? | Small groups are noisy |
#
# Model *selection* uses validation AUC; the *decision* uses cost; the *final report* uses the test set exactly once.

# %% [markdown]
# ## 5. Definition of done
#
# "Done" must be checkable by a machine, otherwise it means "I got tired". Ours:
#
# 1. **Data:** a schema with explicit checks; bad input fails with a message that names the column.
# 2. **Model:** beats the best dumb policy on the **test** cost per customer, and its test AUC is above a floor.
# 3. **Honesty:** test set touched once; preprocessing fitted on train only (a test proves it).
# 4. **Service:** `/health`, `/predict`, `/predict/batch` with validation, versioned responses, tests.
# 5. **Operations:** drift report with thresholds, a release gate, a model card with limitations.
# 6. **Reproducible:** one config file, seeded, a single command trains everything.
#
# Items 2 and 5 are encoded in `configs/config.yaml` as the **release gate**:

# %%
print(CFG["gate"])

# %% [markdown]
# The floor (`min_test_auc`) is a regression tripwire, deliberately below what the model reaches, so a broken
# pipeline fails loudly without blocking normal retrains. We set it **after** measuring the baseline (notebook 06
# prints the numbers); setting it before you have seen any result would be guessing.

# %% [markdown]
# ## Common mistakes
#
# 1. **Optimising accuracy (or F1) because it is the default.** It ignores that a missed churner costs 10x a wasted offer.
# 2. **Threshold 0.5 by habit.** With these costs the break-even is 0.25; 0.5 silently throws away money.
# 3. **Cost numbers nobody can trace.** Keep the derivation next to the matrix, as above.
# 4. **No baseline.** an AUC figure means nothing until you know what a one-line business rule achieves.
# 5. **Forgetting the model is a decision aid.** The 40% save rate is an *assumption*; the model cannot know it.

# %% [markdown]
# ## Interview corner
#
# **1. The classes are imbalanced (37% churn). Do you resample?**
# <details><summary>Answer</summary>
# Not by default. Imbalance this mild does not break most models. Rank with AUC / average precision, then choose the
# decision threshold from costs. Resampling distorts probabilities, which ruins the cost-based threshold.
# </details>
#
# **2. How do you pick a classification threshold?**
# <details><summary>Answer</summary>
# From the business cost of each error type: theoretical break-even is (FP cost) / (FP cost + FN cost) for zero-cost
# true outcomes, and in general uses the cost differences. Confirm empirically on a validation set, because it assumes calibration.
# </details>
#
# **3. The model has 0.85 AUC. Is it good?**
# <details><summary>Answer</summary>
# Unknown until compared with a baseline on the business metric, checked for calibration and subgroup gaps,
# and validated on data that looks like production (ideally out of time).
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: New assumptions, new matrix
# Finance revises the numbers: an offer costs **30**, a churner is worth **300**, and **50%** of contacted churners stay.
# Build `new_costs` (keys `tn, fp, fn, tp`) the same way as above.

# %% tags=["starter"]
new_costs = ...

# %%
assert set(new_costs) == {"tn", "fp", "fn", "tp"}, "keys must be tn, fp, fn, tp"
assert new_costs["tn"] == 0 and new_costs["fp"] == 30 and new_costs["fn"] == 300, "check fp and fn"
assert new_costs["tp"] == 30 + (1 - 0.5) * 300, "tp = offer cost + (1 - save_rate) * churner_value"
print("Exercise 1 passed")

# %% tags=["solution"]
new_costs = {"tn": 0.0, "fp": 30.0, "fn": 300.0, "tp": 30.0 + (1 - 0.5) * 300.0}

# %% [markdown]
# ### Exercise 2: Break-even under the new costs
# Using `break_even`, store the new break-even probability in `new_p_star`. Then decide whether the retention team should now
# contact **more** or **fewer** customers than before, and store `"more"` or `"fewer"` in `contact_more_or_fewer`.
# *Hint: a lower threshold means more customers are flagged.*

# %% tags=["starter"]
new_p_star = ...
contact_more_or_fewer = ...

# %%
expected = (30 - 0) / ((30 - 0) + (300 - 180))
assert abs(new_p_star - expected) < 1e-9, f"expected {expected:.4f}"
assert contact_more_or_fewer == ("more" if expected < p_star else "fewer"), "compare new_p_star with p_star"
print(f"Exercise 2 passed (threshold {p_star:.2f} -> {new_p_star:.2f})")

# %% tags=["solution"]
new_p_star = break_even(new_costs)
contact_more_or_fewer = "more" if new_p_star < p_star else "fewer"

# %% [markdown]
# ### Exercise 3: Price a business rule
# The retention manager proposes: *"contact customers with **4 or more support calls**"*. Using the **original** `cost_matrix`,
# store its cost per customer over all of `df` in `rule_cost`. Then store in `beats_contact_nobody` whether it is cheaper than contacting nobody.
# *Hint: `train.cost_per_customer(y, flag, costs)`; `flag` is a boolean array.*

# %% tags=["starter"]
rule_cost = ...
beats_contact_nobody = ...

# %%
flag = (df["support_calls"] >= 4).to_numpy()
tp_ = int((flag & (y == 1)).sum()); fp_ = int((flag & (y == 0)).sum())
fn_ = int((~flag & (y == 1)).sum()); tn_ = int((~flag & (y == 0)).sum())
manual = (tp_ * 140 + fp_ * 20 + fn_ * 200 + tn_ * 0) / len(df)
assert abs(rule_cost - manual) < 1e-9, f"expected {manual:.3f}"
assert beats_contact_nobody == (manual < baselines.loc["contact nobody", "cost_per_customer"])
print(f"Exercise 3 passed (rule costs {rule_cost:.2f} per customer)")

# %% tags=["solution"]
rule_flag = (df["support_calls"] >= 4).to_numpy()
rule_cost = train.cost_per_customer(y, rule_flag, cost_matrix)
beats_contact_nobody = bool(rule_cost < baselines.loc["contact nobody", "cost_per_customer"])

# %% [markdown]
# ### Exercise 4 (open-ended): Write your own definition of done
# In a markdown cell, list **five** acceptance criteria a stakeholder could sign off for this project, each with a number or a yes/no test
# (for example "test cost per customer at least X% below the best rule"). Which of them can the gate in `config.yaml` check automatically,
# and which need a human?

# %% [markdown]
# ## Recap
# - Frame the problem as **decision + cost**: target, who acts, what each error costs, how it is consumed.
# - Four costs (TN, FP, FN, TP) from three written assumptions give a break-even probability; it is not 0.5.
# - Always price simple baselines (nobody, everybody, one business rule) before modelling.
# - A definition of done must be checkable; ours becomes tests, a release gate and a model card.
#
# **Go deeper:** the scikit-learn user guide section "Tuning the decision threshold for class prediction"
# (`TunedThresholdClassifierCV`), and Google's *Rules of Machine Learning* (rule 1: do not be afraid to launch without machine learning).
#
# **Next:** [Data understanding and cleaning](02_data_understanding_and_cleaning.ipynb)
