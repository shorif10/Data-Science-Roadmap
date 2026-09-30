# %% [markdown]
# # Capstone 2: Data Understanding and Cleaning
#
# **Why this matters.** The first version of a churn model is usually decided by data bugs, not by algorithms: a
# category spelled two ways, a "missing" value that is really an answer, a column that quietly encodes the future.
# In production the same bugs return every month with a new file. So we do two things: understand the data by hand
# once, then turn every finding into code (`src/churn/data.py`) that fails loudly.
#
# **You will learn**
# - read a CSV *without* letting pandas decide what "missing" means
# - separate missing-because-unknown from missing-because-none-of-them
# - run a cleaning function that reports what it changed, and a validator that lists every remaining problem
# - look for target leakage and for signal, before modelling
#
# **Prerequisites:** [Notebook 01](01_problem_framing_and_success_criteria.ipynb) and [pandas loading](../05_Pandas/01_loading_and_inspecting_data.ipynb).

# %%
import sys
from pathlib import Path

SRC = Path("../src").resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn import data
from churn.data import DataValidationError, clean_with_report, load_raw, validate

CSV = Path("../datasets/churn.csv")

# %% [markdown]
# ## 1. The trap in `read_csv`
#
# Load the file the naive way, and the way `data.load_raw` does it.

# %%
naive = pd.read_csv(CSV)
raw = load_raw(CSV)
print("naive  internet_service missing:", naive["internet_service"].isna().sum())
print("proper internet_service missing:", raw["internet_service"].isna().sum())
raw["internet_service"].value_counts(dropna=False)

# %% [markdown]
# The file contains the literal text `None` for customers with **no internet service**. By default pandas reads
# the string `"None"` as a missing value (it is in its default NA list), so the naive load merges two different groups:
# customers with no internet (an *answer*) and customers whose service is genuinely unknown (a *gap*).
# `load_raw` passes `keep_default_na=False, na_values=[""]`, so only an empty field counts as missing.
#
# > This is not a curiosity: with default parsing, `df.info()` in the earlier pandas notebook shows this column as
# > heavily "missing", yet most of those rows are the answer "no internet". Always ask *what does a blank / None / NA mean here?*

# %% [markdown]
# ## 2. Inconsistent labels

# %%
raw["contract"].value_counts()

# %% [markdown]
# Six spellings for three contracts. A model would treat `month-to-month` as a different customer type
# from `Month-to-month`, and a one-hot encoder would create six columns. `clean` maps each case-variant to the canonical label
# and strips whitespace. It never guesses: an unknown value like `Weekly` is left alone so that `validate` can report it.

# %% [markdown]
# ## 3. Run the real cleaner

# %%
cleaned, report = clean_with_report(raw, drop_duplicates=True)
print(report)
cleaned["contract"].value_counts()

# %% [markdown]
# `clean_with_report` returns the cleaned frame and a small dict of what changed: rows in/out, casing fixes per column,
# numbers that failed to parse, duplicates dropped. Logging this per run is how you notice, months later, that a new
# export suddenly has 4,000 casing fixes.
#
# Now the validator. It checks the whole schema in `data.SCHEMA`: required columns, allowed levels, numeric ranges, target in {0, 1}, unique ids.

# %%
validate(cleaned)  # returns None when the frame is valid, raises otherwise
print("schema OK")
pd.DataFrame([vars(c) for c in data.SCHEMA])

# %% [markdown]
# ### What a failure looks like
# Break the data on purpose. A good validator names **every** problem in one go, so the person fixing the file does not
# have to re-run five times.

# %%
broken = cleaned.copy()
broken.loc[0, "contract"] = "Weekly"
broken.loc[1, "tenure_months"] = -3
broken.loc[2, "autopay"] = "maybe"
try:
    validate(broken)
except DataValidationError as exc:
    print(exc)

# %% [markdown]
# ## 4. Missing values: what is missing, and does it matter?

# %%
missing = cleaned.isna().sum()
missing[missing > 0]

# %% [markdown]
# Two columns have gaps and we do **not** drop these rows (that would throw away data and bias the sample). Instead the model
# pipeline will impute *inside* the training fold and add a "was missing" flag. Before trusting that, check whether missingness carries signal:

# %%
churn = "churned"
unknown = cleaned["internet_service"].isna()
tc_missing = cleaned["total_charges"].isna()
summary = pd.DataFrame({
    "n": [unknown.sum(), tc_missing.sum()],
    "churn_rate_when_missing": [cleaned.loc[unknown, churn].mean(), cleaned.loc[tc_missing, churn].mean()],
    "churn_rate_otherwise": [cleaned.loc[~unknown, churn].mean(), cleaned.loc[~tc_missing, churn].mean()],
}, index=["internet_service", "total_charges"])
summary["std_error_of_missing_rate"] = np.sqrt(summary["churn_rate_when_missing"] * (1 - summary["churn_rate_when_missing"]) / summary["n"])
summary.round(3)

# %% [markdown]
# The gap between the two rates is well inside one to two standard errors, so **we cannot claim missingness predicts churn**. With so few
# rows, "no evidence" is not "proof of no effect". Keeping an explicit missing category for internet service and a missing-indicator
# for total charges is cheap insurance. In `features.py` these are the `"Unknown"` level and `add_indicator=True`.

# %% [markdown]
# A second look at `total_charges`: is it consistent with `tenure * monthly_charges`? A large discrepancy would flag a data-entry problem.

# %%
d = cleaned[(cleaned["tenure_months"] > 0) & cleaned["total_charges"].notna()]
ratio = d["total_charges"] / (d["tenure_months"] * d["monthly_charges"])
print(ratio.describe().round(3))
print("customers with tenure 0:", int((cleaned["tenure_months"] == 0).sum()),
      "-> their total_charges is about one month, not 0:", cleaned.loc[cleaned["tenure_months"] == 0, "total_charges"].median())

# %% [markdown]
# `total_charges` is almost exactly `tenure * monthly_charges` (the ratio printed above stays between 0.95 and 1.05), which means it adds little beyond
# those two columns and is nearly collinear with them. That matters for interpretation (notebook 05) and explains a result you will see in notebook 03.

# %% [markdown]
# ## 5. Where is the signal?
# Churn rate by the strongest-looking segments (target mean per group, as in the pandas notebook).

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
by_contract = cleaned.groupby("contract")[churn].mean().reindex(data.CONTRACT_LEVELS)
by_calls = cleaned.assign(calls=cleaned["support_calls"].clip(upper=6)).groupby("calls")[churn].mean()
by_tenure = cleaned.groupby(pd.cut(cleaned["tenure_months"], [-1, 6, 12, 24, 48, 72]), observed=True)[churn].mean()
for ax, s, title, xl in [(axes[0], by_contract, "by contract", "contract"),
                         (axes[1], by_calls, "by support calls (6 = 6 or more)", "support calls in period"),
                         (axes[2], by_tenure, "by tenure", "tenure (months)")]:
    ax.bar(range(len(s)), s.to_numpy())
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([str(i) for i in s.index], rotation=20)
    ax.set_title(title)
    ax.set_xlabel(xl)
axes[0].set_ylabel("share of customers who churned")
plt.tight_layout()
plt.show()

# %% [markdown]
# Month-to-month contracts, many support calls and short tenure all go with higher churn: a coherent story a retention manager would recognise.
# These are **associations**: contract type may be a *cause* of leaving, or a marker of customers who were never committed.

# %%
cleaned.groupby(cleaned["internet_service"].fillna("Unknown"))[churn].agg(["mean", "size"]).round(3)

# %% [markdown]
# ## 6. Leakage checklist (before any modelling)
#
# Target leakage means using information that would not exist when you make the prediction. For every column ask: *at the moment we score
# a customer, do we know this?*
#
# | Column | Known at scoring time? | Risk |
# |---|---|---|
# | `tenure_months`, `monthly_charges`, `contract`, `autopay` ... | yes, current state | low |
# | `total_charges` | yes, accumulated to date | **ask the data owner**: if the snapshot was taken *at the moment of cancellation*, churners' tenure/charges are truncated and the model learns "short history = churn" |
# | `support_calls` | count in the period | low, but check the window does not include the cancellation call itself |
# | `customer_id` | an identifier | must **never** be a feature (a sequential id can encode signup order) |
#
# We cannot verify the snapshot timing from a CSV; that is exactly the kind of open question you write in the project log and ask about.
# `customer_id` is excluded from `FEATURE_COLUMNS` on purpose.

# %%
print(data.FEATURE_COLUMNS)
assert data.ID_COLUMN not in data.FEATURE_COLUMNS and data.TARGET not in data.FEATURE_COLUMNS

# %% [markdown]
# ## Common mistakes
#
# 1. **Letting `read_csv` guess missingness.** `"None"`, `"NA"` and `"N/A"` are converted to NaN by default; sometimes they are answers.
# 2. **Fixing data by hand in a notebook.** A manual `df.loc[...] = ...` is invisible to the API at serving time. Put every fix in a function both paths call (`clean`).
# 3. **Dropping rows with missing values before splitting.** You change the population and leak information about which rows are hard.
# 4. **Imputing on the full dataset.** Fill values must be learned from train only (notebook 03).
# 5. **Deduplicating at serving time.** We drop duplicate rows in training tables only; two identical API requests are two customers.

# %% [markdown]
# ## Interview corner
#
# **1. How do you handle missing values?**
# <details><summary>Answer</summary>
# First find out why they are missing (unknown, not applicable, error) and whether missingness relates to the target. Then choose: keep as its own category,
# impute with a statistic learned on training data plus a missing flag, or a model-based imputer. Never impute before splitting.
# </details>
#
# **2. What is data leakage? Give a non-obvious example.**
# <details><summary>Answer</summary>
# Information available in training that will not be available at prediction time. Non-obvious: features computed on a snapshot taken after the outcome,
# a customer id that encodes time, scaling or imputing with statistics from the whole dataset, or duplicates that land in both train and test.
# </details>
#
# **3. Why validate input data in production and not only in training?**
# <details><summary>Answer</summary>
# Models fail silently: a wrong column order or a new category still returns a number. A schema check turns silent corruption into a clear error and is the first line of monitoring.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: "Unknown" is a group too
# Compute the churn rate for each `internet_service` level of `cleaned`, **including** the customers whose value is missing (call that level `"Unknown"`).
# Store the result as a `Series` named `internet_churn` indexed by level. *Hint: `fillna("Unknown")` before `groupby`.*

# %% tags=["starter"]
internet_churn = ...

# %%
levels = cleaned["internet_service"].fillna("Unknown")
for level in levels.unique():
    expected = cleaned.loc[levels == level, "churned"].mean()
    assert abs(internet_churn[level] - expected) < 1e-9, f"wrong churn rate for {level!r}"
assert set(internet_churn.index) == set(levels.unique()), "index should be DSL, Fiber, None, Unknown"
print("Exercise 1 passed")

# %% tags=["solution"]
internet_churn = cleaned.groupby(cleaned["internet_service"].fillna("Unknown"))["churned"].mean()

# %% [markdown]
# ### Exercise 2: Write a data-quality rule
# A finance colleague says `total_charges` should be within **3%** of `tenure_months * monthly_charges` for any customer with `tenure_months > 0`.
# Count the rows that **violate** this rule (ignore rows where `total_charges` is missing) and store the count in `n_suspicious`.

# %% tags=["starter"]
n_suspicious = ...

# %%
sub = cleaned[(cleaned["tenure_months"] > 0) & cleaned["total_charges"].notna()]
expected_ratio = sub["total_charges"].to_numpy() / (sub["tenure_months"].to_numpy() * sub["monthly_charges"].to_numpy())
expected = int((np.abs(expected_ratio - 1) > 0.03).sum())
assert n_suspicious == expected, f"expected {expected} violating rows, got {n_suspicious}"
print(f"Exercise 2 passed ({n_suspicious} rows violate the 3% rule)")

# %% tags=["solution"]
sub = cleaned[(cleaned["tenure_months"] > 0) & cleaned["total_charges"].notna()]
ratio_ = sub["total_charges"] / (sub["tenure_months"] * sub["monthly_charges"])
n_suspicious = int(((ratio_ - 1).abs() > 0.03).sum())

# %% [markdown]
# ### Exercise 3: Catch the corruption
# Make a copy `bad` of `cleaned`, set `contract` to `"Weekly"` in row 5 and `support_calls` to `-1` in row 6, call `validate(bad)` and catch the
# `DataValidationError`. Store the list of problems in `problems`. *Hint: the exception has a `.problems` attribute.*

# %% tags=["starter"]
problems = ...

# %%
assert isinstance(problems, list) and len(problems) == 2, f"expected exactly 2 problems, got {problems}"
joined = " ".join(problems)
assert "contract" in joined and "support_calls" in joined, "one problem per broken column"
print("Exercise 3 passed:", problems)

# %% tags=["solution"]
bad = cleaned.copy()
bad.loc[5, "contract"] = "Weekly"
bad.loc[6, "support_calls"] = -1
try:
    validate(bad)
    problems = []
except DataValidationError as exc:
    problems = exc.problems

# %% [markdown]
# ### Exercise 4 (open-ended): Extend the schema
# Open `src/churn/data.py`. Which **new rule** would you add to `SCHEMA` or `validate` after this notebook (for example a cross-column check, or a tighter
# `support_calls` maximum)? Write the rule and the reason in a markdown cell, then, as a stretch goal, implement it and add a test in `tests/test_data.py`.

# %% [markdown]
# ## Recap
# - `read_csv` defaults can rewrite your data: `"None"` is an answer here, an empty field is the gap.
# - `clean` fixes only what is safe and reports it; `validate` lists **all** remaining problems at once.
# - Missingness and segments were checked for signal, and columns for leakage, *before* modelling.
# - The same `clean` + `validate` run in training and in the API, so they cannot drift apart.
#
# **Go deeper:** the pandas user guide "Working with missing data" and the `read_csv` docs for `na_values` / `keep_default_na`;
# the scikit-learn documentation chapter "Common pitfalls and recommended practices" (data leakage).
#
# **Next:** [Features and a baseline model](03_features_and_baseline_model.ipynb)
