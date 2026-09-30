# %% [markdown]
# # Loading and Inspecting Data with pandas
#
# **Why this matters.** Before you build a chart or a model, you have to answer boring questions:
# how big is this table, what types are the columns, what is missing, and does anything look wrong?
# Analysts spend a large share of every project here, and the mistakes you catch now are the ones
# that would otherwise silently poison every later result.
#
# **You will learn**
# - load a CSV into a `DataFrame` and understand its structure (`shape`, `dtypes`, `info`)
# - read the first rows, a random sample, and summary statistics
# - find missing values and suspicious values before doing anything else
# - select columns and rows with `[]`, `.loc` and `.iloc`
#
# **Prerequisites:** [Python basics](../01_Python/) and [NumPy arrays](../04_NumPy/).
#
# > Data used: `datasets/churn.csv`, a synthetic telecom customer table (one row per customer,
# > `churned` = 1 if they left). See `docs/DATASETS.md`.

# %%
from pathlib import Path

import pandas as pd

DATA = Path("../datasets")
df = pd.read_csv(DATA / "churn.csv")
type(df)

# %% [markdown]
# `read_csv` returned a **DataFrame**: a table where every column is a `Series` (a labelled 1-D array)
# sharing one row index. If you know NumPy, think of it as a 2-D array that lets each column have its
# own type and gives every row and column a name.

# %% [markdown]
# ## 1. First look: size, columns, types

# %%
print("shape:", df.shape)  # (rows, columns)
df.head()

# %% [markdown]
# `head()` shows the top 5 rows. It answers *"did the file parse the way I expected?"*. Two more looks
# are worth the habit: a **random sample** (the top of a file is often not representative) and the **tail**.

# %%
df.sample(5, random_state=42)

# %%
df.info()

# %% [markdown]
# `info()` is the single most useful inspection call. Read it for three things:
# 1. **Non-Null Count** lower than the row count means missing values (`total_charges`, `internet_service`).
# 2. **Dtype**: numbers should be `int64`/`float64`. Text columns show as `str` in pandas 3
#    (older versions showed `object`).
# 3. **Memory usage** at the bottom, useful when files get large.

# %% [markdown]
# ## 2. Summary statistics

# %%
df.describe()

# %% [markdown]
# `describe()` covers numeric columns only by default. Look at `min` and `max` for anything impossible
# (negative tenure? a charge of 10 million?) and compare `mean` with `50%` (the median): a large gap hints at skew.
#
# For text columns, ask for the categories directly:

# %%
df["contract"].value_counts()

# %% [markdown]
# Notice `Month-to-month` and `month-to-month` are counted as different values. That is a **data quality
# bug**, planted on purpose. We will fix it in the cleaning notebook; here the goal is to *notice* it.
# `value_counts(dropna=False)` also counts missing values, which the default hides:

# %%
df["internet_service"].value_counts(dropna=False)

# %% [markdown]
# ## 3. Finding missing values

# %%
missing = df.isna().sum()
missing[missing > 0]

# %%
(df.isna().mean() * 100).round(2).sort_values(ascending=False).head(4)

# %% [markdown]
# `isna()` gives a True/False table the same shape as `df`. `.sum()` counts Trues per column;
# `.mean()` gives the *fraction* missing, which is easier to judge (3% missing is a different problem from 60%).

# %% [markdown]
# ## 4. Selecting data
#
# | You want | Write | Returns |
# |---|---|---|
# | one column | `df["tenure_months"]` | Series |
# | several columns | `df[["tenure_months", "churned"]]` | DataFrame |
# | rows by **label** | `df.loc[row_labels, col_labels]` | depends |
# | rows by **position** | `df.iloc[row_positions, col_positions]` | depends |
# | rows by **condition** | `df[df["tenure_months"] > 60]` | DataFrame |

# %%
df[["customer_id", "monthly_charges"]].head(3)

# %%
# .loc uses labels (here the default index labels happen to be 0..n-1); the end of a slice is INCLUDED
print(df.loc[0:2, "tenure_months"].tolist())
# .iloc uses positions; the end of a slice is EXCLUDED, like normal Python slicing
print(df.iloc[0:2, 3].tolist())

# %%
long_term = df[df["tenure_months"] > 60]
print(f"{len(long_term)} customers with more than 60 months of tenure")
long_term[["customer_id", "tenure_months", "contract"]].head()

# %% [markdown]
# Combine conditions with `&` (and), `|` (or) and `~` (not), and wrap **each** condition in parentheses:

# %%
risky = df[(df["contract"] == "Month-to-month") & (df["support_calls"] >= 4)]
print(f"{len(risky)} month-to-month customers with 4+ support calls")
print(f"their churn rate: {risky['churned'].mean():.1%}   overall: {df['churned'].mean():.1%}")

# %% [markdown]
# *What just happened?* We filtered rows with a boolean mask, then compared the churn rate of that segment
# with everyone else. This "filter, then compare with the baseline" move is the core of exploratory analysis.
# (The mean of a 0/1 column is the proportion of ones, a handy trick.)

# %% [markdown]
# ## Common mistakes
#
# 1. **`df[df.a > 1 & df.b < 5]` without parentheses**: `&` binds tighter than `>`, so this raises an error
#    (or worse, computes something else). Always write `(df.a > 1) & (df.b < 5)`.
# 2. **Using `and` / `or`** on Series: Python cannot decide truthiness of a whole Series. Use `&` / `|`.
# 3. **Trusting `describe()` alone**: it skips text columns and hides missing values. Combine it with `info()` and `value_counts()`.
# 4. **Assuming the top of the file is typical.** Use `sample()` too.

# %%
# Demonstration of mistake 2
try:
    df[(df["tenure_months"] > 60) and (df["churned"] == 1)]
except ValueError as e:
    print("ValueError:", str(e)[:90], "...")

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between `loc` and `iloc`?**
# <details><summary>Answer</summary>
# <code>loc</code> selects by label and includes the end of a slice; <code>iloc</code> selects by integer position and excludes the end.
# On a default 0..n-1 index they look similar, which hides bugs once the index changes (for example after filtering or sorting).
# </details>
#
# **2. How do you count missing values per column, and why look at the fraction too?**
# <details><summary>Answer</summary>
# <code>df.isna().sum()</code> for counts, <code>df.isna().mean()</code> for the fraction. The right handling depends on how much is missing
# (drop, impute, or flag) and whether missingness is random or related to the target.
# </details>
#
# **3. Why can `describe()` mislead you?**
# <details><summary>Answer</summary>
# It defaults to numeric columns, ignores NaNs in its counts, and summarises with a mean that outliers can distort. Check <code>info()</code>,
# categorical <code>value_counts()</code>, and plots as well.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden
# solution cell at the bottom of each exercise.

# %% [markdown]
# ### Exercise 1: How big and how dirty?
# Store the number of **columns** in `n_cols` and the number of **rows that have at least one missing value** in `n_rows_with_missing`.
# *Hint: `df.isna().any(axis=1)` flags each row.*

# %% tags=["starter"]
n_cols = ...
n_rows_with_missing = ...

# %%
assert n_cols == df.shape[1], "n_cols should be the number of columns (df.shape[1])"
expected = int(df.isna().any(axis=1).sum())
assert n_rows_with_missing == expected, f"expected {expected} rows with a missing value, got {n_rows_with_missing}"
print("Exercise 1 passed")

# %% tags=["solution"]
n_cols = df.shape[1]
n_rows_with_missing = int(df.isna().any(axis=1).sum())

# %% [markdown]
# ### Exercise 2: The longest-standing customers
# Create `top_tenure`: a DataFrame with only `customer_id`, `tenure_months` and `contract` for the **10 customers with the
# highest tenure** (any tie-break is fine). *Hint: `sort_values(..., ascending=False)` then `head`, or `nlargest`.*

# %% tags=["starter"]
top_tenure = ...

# %%
assert list(top_tenure.columns) == ["customer_id", "tenure_months", "contract"], "keep exactly those 3 columns, in that order"
assert len(top_tenure) == 10, "you need exactly 10 rows"
assert sorted(top_tenure["tenure_months"]) == sorted(df["tenure_months"].nlargest(10)), "these are not the 10 largest tenures"
print("Exercise 2 passed")

# %% tags=["solution"]
top_tenure = df.nlargest(10, "tenure_months")[["customer_id", "tenure_months", "contract"]]

# %% [markdown]
# ### Exercise 3: A risky segment
# What fraction of customers on **fiber internet** with **monthly charges above 70** churned? Store it (0 to 1) in `fiber_high_churn`.
# Then decide: is it higher or lower than the overall churn rate? Store `True` in `is_higher` if higher.

# %% tags=["starter"]
fiber_high_churn = ...
is_higher = ...

# %%
mask = (df["internet_service"] == "Fiber") & (df["monthly_charges"] > 70)
assert abs(fiber_high_churn - df.loc[mask, "churned"].mean()) < 1e-9, "check your filter: Fiber AND monthly_charges > 70"
assert is_higher == (fiber_high_churn > df["churned"].mean()), "compare against df['churned'].mean()"
print(f"Exercise 3 passed (segment churn {fiber_high_churn:.1%} vs overall {df['churned'].mean():.1%})")

# %% tags=["solution"]
mask = (df["internet_service"] == "Fiber") & (df["monthly_charges"] > 70)
fiber_high_churn = df.loc[mask, "churned"].mean()
is_higher = bool(fiber_high_churn > df["churned"].mean())

# %% [markdown]
# ### Exercise 4 (open-ended): Write a 3-line data-quality report
# Using only what you learned in this notebook, find **three** things about `churn.csv` you would want to fix or ask about
# before modelling (for example: missing values, inconsistent labels, suspicious ranges). Write them in a markdown cell.

# %% [markdown]
# ## Recap
# - `shape`, `head()`, `sample()`, `info()`, `describe()` and `value_counts()` are your standard first look.
# - `isna().sum()` / `.mean()` find missing data; check both count and fraction.
# - `df["col"]` is a Series, `df[["a", "b"]]` a DataFrame; `.loc` is by label (end included), `.iloc` by position (end excluded).
# - Filter with boolean masks, use `&`, `|`, `~`, and parenthesise every condition.
#
# **Go deeper:** the pandas user guide, sections "Indexing and selecting data" and "Working with missing data"
# (pandas.pydata.org/docs/user_guide).
#
# **Next:** [Cleaning data](02_cleaning_data.ipynb)
