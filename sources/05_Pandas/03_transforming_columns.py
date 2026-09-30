# %% [markdown]
# # Transforming Columns: New Features from Old
#
# **Why this matters.** Raw columns are rarely what an analysis or a model needs. You will constantly derive new ones: revenue from price
# times quantity, a "high spender" flag, the weekday of an order, a price band, the brand name inside a product title. This notebook is the
# toolbox for doing that *fast* (vectorised) and *readably*, and for knowing when `apply` is the wrong tool.
#
# **You will learn**
# - create columns with `assign` and vectorised arithmetic
# - recode values with `map` and `replace`; use `np.where` / `np.select` for rules
# - know when `apply` is acceptable and how much slower it is
# - use the `.str` and `.dt` accessors, the `category` dtype, and bin numbers with `cut` / `qcut`
#
# **Prerequisites:** [Cleaning data](02_cleaning_data.ipynb).
#
# > Data used: `churn.csv` plus the shop tables (`customers`, `products`, `orders`). Synthetic, see `docs/DATASETS.md`.

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")

# churn, loaded the careful way from the cleaning notebook: "None" stays a category, only blanks are missing
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
churn["contract"] = churn["contract"].str.strip().str.lower().map(
    {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}
)
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv")
print(churn.shape, customers.shape, products.shape, orders.shape)

# %% [markdown]
# ## 1. New columns: `df["new"] = ...` and `assign`
#
# Arithmetic between columns is **vectorised**: pandas applies it to every row at once, aligned by index label.
# `assign` returns a *new* DataFrame with the extra columns, which lets you chain steps without mutating the original.

# %%
churn["avg_monthly_billed"] = churn["total_charges"] / churn["tenure_months"]   # tenure 0 -> division by zero
print(churn["avg_monthly_billed"].replace([np.inf, -np.inf], np.nan).describe().round(2).to_string())
print("rows with inf:", int(np.isinf(churn["avg_monthly_billed"]).sum()))

# %% [markdown]
# Two lessons in one cell. (1) Dividing by zero in pandas gives `inf`/`NaN` rather than an error, so always check for it.
# (2) We just *mutated* `churn`. That is fine for a personal notebook; for pipelines prefer `assign`:

# %%
churn = churn.drop(columns="avg_monthly_billed")
churn2 = churn.assign(
    tenure_years=churn["tenure_months"] / 12,
    high_support=churn["support_calls"] >= 4,
    charge_vs_median=churn["monthly_charges"] - churn["monthly_charges"].median(),
)
print(churn2[["tenure_months", "tenure_years", "high_support", "charge_vs_median"]].head(3).round(2).to_string())
print("original has new columns?", "tenure_years" in churn.columns)

# %% [markdown]
# *What just happened?* `assign` built a new frame; the original `churn` is unchanged. Inside `assign` you can also use a `lambda` that receives
# the frame built so far, so later columns can use earlier ones: `df.assign(a=..., b=lambda d: d["a"] * 2)`.

# %% [markdown]
# ## 2. Recoding: `map`, `replace` and `astype`
#
# `Series.map(dict)` looks each value up; unmapped values become `NaN`. `replace(dict)` leaves unmapped values alone. Pick the one that matches how
# you want surprises to behave.

# %%
yes_no = {"Yes": 1, "No": 0}
paperless = churn["paperless_billing"].map(yes_no)
print(paperless.value_counts().to_string())
print("unmapped -> NaN:", int(paperless.isna().sum()))
print(pd.Series(["Yes", "No", "Maybe"]).map(yes_no).tolist(), "vs replace:", pd.Series(["Yes", "No", "Maybe"]).replace(yes_no).tolist())

# %% [markdown]
# ## 3. Rules: `np.where` and `np.select`
# `np.where(condition, a, b)` is a vectorised if/else. For several branches use `np.select(conditions, choices, default)`: **the first matching condition wins**.

# %%
churn = churn.assign(
    is_long_term=np.where(churn["tenure_months"] >= 24, "long", "short"),
    risk=np.select(
        [
            (churn["contract"] == "Month-to-month") & (churn["support_calls"] >= 4),
            (churn["contract"] == "Month-to-month") | (churn["support_calls"] >= 4),
        ],
        ["high", "medium"],
        default="low",
    ),
)
summary = churn.groupby("risk")["churned"].agg(rows="size", churn_rate="mean").round(3)
print(summary.to_string())

# %% [markdown]
# Because it was built from domain knowledge, our hand-made `risk` rule separates churn rates well. That is worth *testing*, not assuming:
# the table above is the test. (`groupby` gets its own notebook next.)

# %% [markdown]
# ## 4. `apply` vs vectorised: a fair timing
#
# `Series.apply(func)` calls a Python function once per element; `DataFrame.apply(func, axis=1)` once per **row** and is slower still.
# It is flexible but runs at Python speed. Rule: **if a vectorised expression exists, use it; reach for `apply` only when logic truly cannot be expressed on whole columns.**
# Same task three ways (a 10% discount on charges above 60), same result asserted, best of 3 runs:

# %%
def best_of(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out

def with_apply_rows():
    return churn.apply(lambda r: r["monthly_charges"] * 0.9 if r["monthly_charges"] > 60 else r["monthly_charges"], axis=1)

def with_apply_series():
    return churn["monthly_charges"].apply(lambda v: v * 0.9 if v > 60 else v)

def vectorised():
    return np.where(churn["monthly_charges"] > 60, churn["monthly_charges"] * 0.9, churn["monthly_charges"])

t_rows, r1 = best_of(with_apply_rows)
t_ser, r2 = best_of(with_apply_series)
t_vec, r3 = best_of(vectorised, repeats=10)
assert np.allclose(r1, r2) and np.allclose(r1, r3), "all three must agree"
print(f"apply(axis=1): {t_rows * 1000:7.1f} ms | Series.apply: {t_ser * 1000:6.1f} ms | vectorised: {t_vec * 1000:6.2f} ms")
print(f"vectorised is about {t_rows / t_vec:.0f}x faster than row-wise apply on {len(churn):,} rows (machine dependent)")

# %% [markdown]
# ## 5. Text with `.str`
# `.str` exposes vectorised string methods: `lower`, `strip`, `contains`, `startswith`, `split`, `extract` (regular-expression capture groups) and more.
# Missing text stays missing rather than raising an error.

# %%
print(products["product_name"].head(4).tolist())
parts = products["product_name"].str.rsplit(" ", n=1, expand=True)        # split from the right: "Yoga Mat Pro" -> ["Yoga Mat", "Pro"]
products = products.assign(item=parts[0], tier=parts[1])
print(products[["product_name", "item", "tier"]].iloc[[0, 1, 48]].to_string())
print(products["tier"].value_counts().to_string())

# %% [markdown]
# `rsplit(" ", n=1)` splits only at the **last** space, so multi-word items such as "Yoga Mat" survive. A plain `split()` would have cut them in half:
# choosing the right split is the entire skill. `str.extract` with a regular expression is often clearer when the pattern is fixed:

# %%
ids = churn["customer_id"].str.extract(r"C(\d+)")[0].astype(int)     # "C100000" -> 100000
print(ids.head(3).tolist(), "| pro items whose name contains 'Pro':", int(products["product_name"].str.contains("Pro").sum()))

# %% [markdown]
# ## 6. Dates with `.dt`
# Parse text into real datetimes once (`pd.to_datetime`); the `.dt` accessor then gives components with no string slicing.

# %%
customers = customers.assign(signup=pd.to_datetime(customers["signup_date"]))
customers = customers.assign(
    signup_year=customers["signup"].dt.year,
    signup_month=customers["signup"].dt.month,
    signup_weekday=customers["signup"].dt.day_name(),
    days_since_signup=(pd.Timestamp("2025-01-01") - customers["signup"]).dt.days,
)
print(customers[["signup", "signup_year", "signup_month", "signup_weekday", "days_since_signup"]].head(3).to_string())
print(customers["signup_weekday"].value_counts().head(3).to_string())

# %% [markdown]
# Subtracting two datetimes gives a `timedelta`; `.dt.days` turns it into a number. `dt.to_period("M")` maps each date to its month, a handy grouping key:

# %%
print(customers["signup"].dt.to_period("M").value_counts().sort_index().head(3).to_string())

# %% [markdown]
# ## 7. The `category` dtype
# A column with few distinct values (contract, country, channel) can be stored as `category`: integer codes plus a small lookup table.
# Benefits: much less memory, faster groupby, and an **ordering** if you declare one.

# %%
before = churn["contract"].memory_usage(deep=True)
contract_cat = churn["contract"].astype(pd.CategoricalDtype(["Month-to-month", "One year", "Two year"], ordered=True))
after = contract_cat.memory_usage(deep=True)
print(f"memory: {before / 1e3:.1f} KB as text -> {after / 1e3:.1f} KB as category")
print(contract_cat.cat.categories.tolist(), "| ordered:", contract_cat.cat.ordered)
print("customers on a contract longer than month-to-month:", int((contract_cat > "Month-to-month").sum()))

# %% [markdown]
# The ordered comparison works only because we declared the order. Catch: categories can surprise you: `groupby` on a category column shows **empty groups** for unused categories
# unless you pass `observed=True`, and assigning a value that is not among the categories raises an error.

# %% [markdown]
# ## 8. Binning numbers: `cut` and `qcut`
#
# - `pd.cut(x, bins=edges)` uses **your** boundaries. Intervals are `(left, right]` by default: the right edge is included.
# - `pd.qcut(x, q=4)` makes bins with (roughly) **equal counts**: quartiles.

# %%
edges = [0, 35, 65, 100]
band = pd.cut(churn["monthly_charges"], bins=edges, labels=["low", "mid", "high"])
print(band.value_counts().sort_index().to_string())

quart = pd.qcut(churn["tenure_months"], q=4)
print(quart.value_counts().sort_index().to_string())

# %% [markdown]
# `cut` gives unequal counts, `qcut` gives (nearly) equal ones. Then look at how churn varies across bins:

# %%
print(churn.groupby(quart, observed=True)["churned"].mean().round(3).to_string())

# %% [markdown]
# Choose `cut` when boundaries have business meaning (price bands, age groups) and `qcut` when you want groups of equal size (deciles of spend).
# `qcut` can fail when many values are identical (duplicate bin edges); pass `duplicates="drop"` and check what you got.

# %%
try:
    pd.qcut(churn["support_calls"], q=10)
except ValueError as e:
    print("ValueError:", str(e)[:70], "...")
print(pd.qcut(churn["support_calls"], q=10, duplicates="drop").nunique(), "distinct bins after duplicates='drop'")

# %% [markdown]
# ## Common mistakes
#
# 1. **`apply(axis=1)` for arithmetic** that is a one-line vectorised expression (slow, and harder to read).
# 2. **Forgetting `map` turns unknowns into `NaN`.** Check `isna().sum()` after mapping.
# 3. **Editing a slice and expecting the original to change** (or the reverse). In pandas 3, `df2 = df[cols]` behaves as an independent copy; use `df.loc[mask, "col"] = v` to change `df`.
# 4. **`np.select` with conditions in the wrong order**: the first true one wins. Put specific rules first.
# 5. **`cut` right-inclusive surprises**: with edges `[0, 35, 65]` a value of exactly 35 lands in the first bin. Use `right=False` if you want `[left, right)`.
# 6. **Dividing by zero** (tenure 0) silently produces `inf`.
# 7. **String dates.** Sorting `"12/01/2024"` as text does not sort by date; convert first.

# %%
print(pd.cut(pd.Series([35]), [0, 35, 65]).astype(str).tolist(), "vs right=False:", pd.cut(pd.Series([35]), [0, 35, 65], right=False).astype(str).tolist())

# %% [markdown]
# ## Interview corner
#
# **1. `apply` versus vectorised operations: when do you use each?**
# <details><summary>Answer</summary>
# Vectorised operations (arithmetic, <code>np.where</code>, <code>.str</code>, <code>.dt</code>) run in compiled code and are typically far faster and more readable. <code>apply</code> runs a Python function per element or row;
# use it only when the logic cannot be expressed on whole columns, and consider <code>numba</code> or restructuring the problem before accepting the cost.
# </details>
#
# **2. `map` versus `replace`?**
# <details><summary>Answer</summary>
# <code>map</code> substitutes via a dict/function and turns unmatched values into NaN; <code>replace</code> only changes matched values and leaves the rest. <code>map</code> is stricter, which can be a feature when you want unexpected values to surface.
# </details>
#
# **3. `pd.cut` versus `pd.qcut`?**
# <details><summary>Answer</summary>
# <code>cut</code> bins by fixed value ranges you choose (unequal counts); <code>qcut</code> bins by quantiles (roughly equal counts, unequal widths).
# </details>
#
# **4. Why use the `category` dtype?**
# <details><summary>Answer</summary>
# Low-cardinality text is stored once as a lookup table plus integer codes, saving memory and speeding up grouping and comparisons; it also supports ordered categories. The trade-offs are stricter assignment rules and empty groups with <code>groupby</code> unless <code>observed=True</code>.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Charge bands
# Add a column `band` to a copy of `churn` called `churn_b` using `pd.cut` on `monthly_charges` with edges `[0, 35, 65, 100]` and labels `["low", "mid", "high"]`.
# Then store in `band_churn` the churn rate (mean of `churned`) per band as a Series indexed by band, ordered low, mid, high.

# %% tags=["starter"]
churn_b = churn.copy()
band_churn = ...

# %%
assert "band" in churn_b.columns, "add the band column to churn_b"
manual = {b: churn.loc[(churn["monthly_charges"] > lo) & (churn["monthly_charges"] <= hi), "churned"].mean()
          for b, lo, hi in [("low", 0, 35), ("mid", 35, 65), ("high", 65, 100)]}
assert list(band_churn.index) == ["low", "mid", "high"], "order the index low, mid, high"
for b, v in manual.items():
    assert abs(band_churn[b] - v) < 1e-9, f"band {b}: expected {v:.4f}"
print("Exercise 1 passed:", band_churn.round(3).to_dict())

# %% tags=["solution"]
churn_b = churn.copy()
churn_b["band"] = pd.cut(churn_b["monthly_charges"], bins=[0, 35, 65, 100], labels=["low", "mid", "high"])
band_churn = churn_b.groupby("band", observed=True)["churned"].mean()

# %% [markdown]
# ### Exercise 2: Equal-sized spenders
# Split `churn` customers into **five** equal-count groups by `monthly_charges` (quintiles, labelled 1 to 5, where 5 is the highest charges) in a Series `quintile`.
# Then store `top_quintile_churn`, the churn rate of quintile 5, and `bottom_quintile_churn` for quintile 1.

# %% tags=["starter"]
quintile = ...
top_quintile_churn = ...
bottom_quintile_churn = ...

# %%
x = churn["monthly_charges"]
q20, q80 = x.quantile(0.2), x.quantile(0.8)          # qcut bins are (edge_i, edge_i+1], the first bin includes the minimum
assert sorted(quintile.astype(int).unique()) == [1, 2, 3, 4, 5], "labels 1..5"
sizes = quintile.astype(int).value_counts()
assert sizes.min() > 0.8 * len(churn) / 5 and sizes.max() < 1.2 * len(churn) / 5, "groups should be roughly equal in size (ties in the data make them not exact)"
assert abs(top_quintile_churn - churn.loc[x > q80, "churned"].mean()) < 1e-9, "top quintile churn rate"
assert abs(bottom_quintile_churn - churn.loc[x <= q20, "churned"].mean()) < 1e-9, "bottom quintile churn rate"
print(f"Exercise 2 passed (top {top_quintile_churn:.1%} vs bottom {bottom_quintile_churn:.1%})")

# %% tags=["solution"]
quintile = pd.qcut(churn["monthly_charges"], q=5, labels=[1, 2, 3, 4, 5])
top_quintile_churn = churn.loc[quintile == 5, "churned"].mean()
bottom_quintile_churn = churn.loc[quintile == 1, "churned"].mean()

# %% [markdown]
# ### Exercise 3: Product tiers and price premium
# Using the `products` frame (which already has `tier`), create `tier_price`: the average `unit_price` for each tier (Basic / Pro) as a Series.
# Then `pro_premium`: how many times more expensive Pro is on average than Basic (`pro_avg / basic_avg`). Do not hard-code numbers.

# %% tags=["starter"]
tier_price = ...
pro_premium = ...

# %%
b = products.loc[products["product_name"].str.endswith("Basic"), "unit_price"].mean()
p = products.loc[products["product_name"].str.endswith("Pro"), "unit_price"].mean()
assert set(tier_price.index) == {"Basic", "Pro"}, "index should be the two tiers"
assert abs(tier_price["Basic"] - b) < 1e-9 and abs(tier_price["Pro"] - p) < 1e-9, "average unit_price per tier"
assert abs(pro_premium - p / b) < 1e-9, "pro_premium = pro_avg / basic_avg"
print(f"Exercise 3 passed (Pro is {pro_premium:.2f}x Basic on average)")

# %% tags=["solution"]
tier_price = products.groupby("tier")["unit_price"].mean()
pro_premium = tier_price["Pro"] / tier_price["Basic"]

# %% [markdown]
# ### Exercise 4: Order calendar
# Convert `orders["order_date"]` to datetimes and build `weekday_counts`: the number of orders per weekday name, indexed by the names
# `Monday` ... `Sunday` **in calendar order** (not alphabetical, not by count). Also store `busiest_day`, the name of the weekday with the most orders.
# *Hint: `dt.day_name()`, `value_counts`, then `reindex` with the weekday names in order.*

# %% tags=["starter"]
weekday_counts = ...
busiest_day = ...

# %%
names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
exp = pd.to_datetime(orders["order_date"]).dt.dayofweek.value_counts().sort_index()
assert list(weekday_counts.index) == names, "index must be Monday..Sunday in order"
assert list(weekday_counts.values) == list(exp.values), "counts per weekday do not match"
assert busiest_day == names[int(exp.idxmax())], f"expected {names[int(exp.idxmax())]}"
print(f"Exercise 4 passed (busiest weekday: {busiest_day})")

# %% tags=["solution"]
odt = pd.to_datetime(orders["order_date"])
names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_counts = odt.dt.day_name().value_counts().reindex(names)
busiest_day = weekday_counts.idxmax()

# %% [markdown]
# ### Exercise 5 (open-ended): Design a risk score
# Extend the `risk` rule (or make your own with `np.select`) using `autopay`, `paperless_billing`, `senior_citizen` or `tenure_months`.
# Report the churn rate of each of your levels in a markdown cell. Does your version separate churners better than the one in the notebook? (Look at the spread between the lowest and the highest level.)

# %% [markdown]
# ## Recap
# - Vectorised column arithmetic, `assign` for chainable new columns, and watch for divide-by-zero.
# - `map` (strict) vs `replace` (lenient); `np.where` / `np.select` for rules (first match wins).
# - `apply(axis=1)` is the slow path: measure before you accept it.
# - `.str` (`rsplit`, `extract`, `contains`) and `.dt` (components, `to_period`, subtraction to `timedelta`) avoid manual string slicing.
# - `category` saves memory and gives ordering; `cut` = your edges, `qcut` = equal counts.
#
# **Go deeper:** pandas user guide "Working with text data", "Categorical data" and the API reference for `pandas.cut` and `pandas.qcut` (pandas.pydata.org/docs).
#
# **Next:** [GroupBy and aggregation](04_groupby_and_aggregation.ipynb)
