# %% [markdown]
# # GroupBy and Aggregation
#
# **Why this matters.** "Churn rate by contract", "revenue by category", "top three products per country", "each customer's spend relative to their segment":
# nearly every business question is *compute something per group*. `groupby` is the pandas answer, and learning its four verbs
# (`agg`, `transform`, `filter`, `apply`) covers most of what you will do with tables in a job or an interview.
#
# **You will learn**
# - explain **split-apply-combine** and translate any "per group" question into `groupby`
# - summarise with **named aggregation** (`agg(new_name=("col", "func"))`)
# - broadcast group statistics back onto rows with `transform`, and drop groups with `filter`
# - avoid the classic traps: mean of means, dropped `NaN` keys, empty categorical groups
#
# **Prerequisites:** [Transforming columns](03_transforming_columns.ipynb).
#
# > Data used: `churn.csv`, and the shop tables `customers`, `products`, `order_items` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
churn["contract"] = churn["contract"].str.strip().str.lower().map(
    {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}
)
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
order_items = pd.read_csv(DATA / "order_items.csv")

# %% [markdown]
# ## 1. Split, apply, combine
#
# Think of sorting a stack of customer cards into piles by contract type (**split**), computing the churn rate of each pile (**apply**), and writing the
# results on one sheet (**combine**). Here is that done by hand with a loop, and then with `groupby`:

# %%
by_hand = {c: churn.loc[churn["contract"] == c, "churned"].mean() for c in churn["contract"].unique()}
print(pd.Series(by_hand).round(3).to_string())
print()
print(churn.groupby("contract")["churned"].mean().round(3).to_string())

# %% [markdown]
# *What just happened?* Same numbers, one line, and `groupby` works for any number of groups. It also **sorts by the group key** by default (`sort=False` keeps first-seen order).
# `groupby` itself computes nothing until you ask for an aggregate; inspecting the object is a useful debugging tool:

# %%
g = churn.groupby("contract")
print(type(g).__name__, "| number of groups:", g.ngroups)
print(g.size().to_string())      # rows per group (counts NaN rows too), unlike .count() which counts non-null values per column

# %% [markdown]
# ## 2. Named aggregation: the readable way to summarise
#
# `agg` with **named aggregation** produces well-named columns in one step: `agg(output_name=("input_column", "function"))`. The function can be a string
# (`"mean"`, `"sum"`, `"size"`, `"nunique"`, `"median"`) or any callable.

# %%
summary = churn.groupby("contract").agg(
    customers=("customer_id", "size"),
    churn_rate=("churned", "mean"),
    avg_monthly=("monthly_charges", "mean"),
    median_tenure=("tenure_months", "median"),
    p90_calls=("support_calls", lambda s: s.quantile(0.9)),
)
summary.round(3)

# %% [markdown]
# Group by several keys by passing a list; the result has a **MultiIndex**. Use `as_index=False` (or `reset_index()`) when you want the keys back as ordinary columns,
# e.g. for merging or plotting:

# %%
two_keys = churn.groupby(["contract", "internet_service"], dropna=False, as_index=False).agg(
    customers=("customer_id", "size"), churn_rate=("churned", "mean")
)
print(two_keys.round(3).to_string())

# %% [markdown]
# Note `dropna=False`: by default `groupby` **silently drops rows whose key is missing**. Here `internet_service` has genuinely empty cells (see the cleaning notebook).
# Without the flag those customers would vanish from the report and the totals would not add up. A cheap safeguard: check that group sizes sum to the row count.

# %%
kept = churn.groupby("internet_service")["customer_id"].size().sum()
print("rows kept with default groupby:", kept, "of", len(churn), "| with dropna=False:", churn.groupby("internet_service", dropna=False).size().sum())

# %% [markdown]
# ## 3. `transform`: group statistics aligned back to rows
#
# `agg` **shrinks** (one row per group). `transform` returns a result the **same length as the input**, with each row receiving its group's value. Use it to compare
# each row with its group: share of group total, z-score within group, or fill missing values with the group median.

# %%
churn = churn.assign(
    contract_avg_charge=churn.groupby("contract")["monthly_charges"].transform("mean"),
)
churn = churn.assign(charge_vs_contract=churn["monthly_charges"] - churn["contract_avg_charge"])
print(churn[["contract", "monthly_charges", "contract_avg_charge", "charge_vs_contract"]].head(4).round(2).to_string())
print("mean of the deviation inside each group (should be ~0):")
print(churn.groupby("contract")["charge_vs_contract"].mean().abs().max() < 1e-9)

# %% [markdown]
# Within-group standardisation and group-wise imputation follow the same pattern:

# %%
z = churn.groupby("contract")["monthly_charges"].transform(lambda s: (s - s.mean()) / s.std())
print(z.groupby(churn["contract"]).agg(["mean", "std"]).round(3).to_string())

filled = churn["total_charges"].fillna(churn.groupby("contract")["total_charges"].transform("median"))
print("missing total_charges before:", int(churn["total_charges"].isna().sum()), "| after group-median fill:", int(filled.isna().sum()))

# %% [markdown]
# ## 4. `filter` and picking rows per group
#
# `filter` keeps or drops **whole groups** based on a condition on the group. Example: keep only the (contract, internet) combinations with at least 500 customers.

# %%
big = churn.groupby(["contract", "internet_service"], dropna=False).filter(lambda d: len(d) >= 500)
print(len(big), "rows in", big.groupby(["contract", "internet_service"], dropna=False).ngroups, "groups")

# %% [markdown]
# "Top N per group" is a very common interview question: sort, then take the first N rows of each group.

# %%
top2_tenure = (
    churn.sort_values("tenure_months", ascending=False)
    .groupby("contract")
    .head(2)[["contract", "customer_id", "tenure_months"]]
    .sort_values(["contract", "tenure_months"], ascending=[True, False])
)
print(top2_tenure.to_string())

# %% [markdown]
# ## 5. Shop data: revenue by category
#
# Revenue lives across two tables: `order_items` has the quantity, `products` has the price. We need to attach the price to each item first.
# That operation is a **merge**, and gets its own notebook; for now treat it as "look up each item's price".

# %%
items = order_items.merge(products, on="product_id", how="left")
items = items.assign(revenue=items["quantity"] * items["unit_price"])
print(items.shape, "| missing prices:", int(items["unit_price"].isna().sum()))

by_cat = items.groupby("category").agg(
    units=("quantity", "sum"),
    revenue=("revenue", "sum"),
    avg_line_value=("revenue", "mean"),
).sort_values("revenue", ascending=False)
by_cat["revenue_share"] = by_cat["revenue"] / by_cat["revenue"].sum()
by_cat.round(3)

# %% [markdown]
# Category revenue differs a lot from category *unit* volume because prices differ by category. Always look at both.
# (Total revenue here counts every order line regardless of order status; later notebooks decide how to treat cancelled and returned orders.)

# %% [markdown]
# ## 6. Traps
#
# **Mean of means.** The average of the group averages (or rates) is *not* the overall figure unless the groups are the same size:

# %%
group_means = churn.groupby("contract")["churned"].mean()          # a rate per group
print("mean of group churn rates:", round(group_means.mean(), 3), "| overall rate:", round(churn["churned"].mean(), 3))
weights = churn.groupby("contract").size()
print("weighted by group size:", round(float(np.average(group_means, weights=weights)), 3))

# %% [markdown]
# The unweighted average of the three churn rates is visibly different from the true overall rate; weighting by group size recovers it. Always carry the count next to the rate.
#
# **Categorical keys and unused categories.** With a `category` column, pandas 3 groups only the categories that occur (`observed=True` is now the default); older versions listed every category as an empty group. `observed=False` brings that behaviour back, which is useful when you need a complete table:

# %%
cat = churn.assign(contract=pd.Categorical(churn["contract"], ["Month-to-month", "One year", "Two year", "Three year"]))
print(cat.groupby("contract", observed=False).size().to_string())
print("-- observed=True --")
print(cat.groupby("contract", observed=True).size().to_string())

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting `dropna=False`** and losing rows with a missing key. Verify that group sizes sum to the row count.
# 2. **Averaging averages / rates** without weights.
# 3. **`count()` vs `size()`**: `count` ignores NaN per column; `size` counts rows.
# 4. **Using `apply` with a Python function** when a named aggregation or `transform("mean")` exists; the built-in string aggregations are optimised and much faster.
# 5. **Grouping by a float column** (rounding noise makes near-identical values separate groups). Bin it first with `cut`.
# 6. **Expecting a flat table** after grouping by two keys and then being confused by the MultiIndex: use `as_index=False` or `reset_index()`.
# 7. **Aggregating text columns with `mean`**. It raises an error in current pandas (it did not always). Select numeric columns explicitly.

# %%
try:
    churn.groupby("contract")[["internet_service"]].mean()
except (TypeError, ValueError) as e:
    print(type(e).__name__, "-", str(e)[:80])

# %% [markdown]
# ## Interview corner
#
# **1. Explain split-apply-combine and the difference between `agg`, `transform` and `filter`.**
# <details><summary>Answer</summary>
# Split the frame by key, apply a function to each piece, combine the results. <code>agg</code> reduces each group to one row; <code>transform</code> returns a result aligned to the original rows (same length);
# <code>filter</code> keeps or drops entire groups by a group-level condition.
# </details>
#
# **2. How do you get "top 3 products by revenue in each category"?**
# <details><summary>Answer</summary>
# Aggregate revenue by (category, product), sort by revenue descending, then <code>groupby("category").head(3)</code>. Alternatives: <code>groupby(...)["revenue"].rank(ascending=False) &lt;= 3</code>, or <code>nlargest</code> inside <code>apply</code> (slower).
# </details>
#
# **3. Why did some rows disappear after my `groupby`?**
# <details><summary>Answer</summary>
# By default groups whose key is <code>NaN</code> are dropped. Use <code>dropna=False</code> to keep them, and sanity-check that the group sizes sum to the number of input rows.
# </details>
#
# **4. `count` vs `size`?**
# <details><summary>Answer</summary>
# <code>size</code> is the number of rows in the group; <code>count</code> is the number of non-null values for each column separately.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Churn by internet service
# Build `svc`: a DataFrame indexed by `internet_service` (include the missing group) with columns `customers` (row count) and `churn_rate`.
# *Hint: named aggregation, and remember `dropna=False`.*

# %% tags=["starter"]
svc = ...

# %%
assert list(svc.columns) == ["customers", "churn_rate"], "columns must be customers and churn_rate"
assert svc["customers"].sum() == len(churn), "the missing-service customers must be included so the counts sum to all rows"
fiber = churn[churn["internet_service"] == "Fiber"]
assert svc.loc["Fiber", "customers"] == len(fiber) and abs(svc.loc["Fiber", "churn_rate"] - fiber["churned"].mean()) < 1e-9, "check the Fiber row"
assert svc.index.isna().sum() == 1, "there should be one row for the missing key"
print("Exercise 1 passed")
print(svc.round(3))

# %% tags=["solution"]
svc = churn.groupby("internet_service", dropna=False).agg(
    customers=("customer_id", "size"), churn_rate=("churned", "mean")
)

# %% [markdown]
# ### Exercise 2: Compare each customer with their contract peers
# Create a Series `calls_vs_contract` (same length as `churn`): each customer's `support_calls` minus the **average support calls of their contract group**.
# Then store in `n_above` how many customers have more calls than their group average.
# *Hint: `transform`.*

# %% tags=["starter"]
calls_vs_contract = ...
n_above = ...

# %%
manual_avg = churn["contract"].map({c: churn.loc[churn["contract"] == c, "support_calls"].mean() for c in churn["contract"].unique()})
assert len(calls_vs_contract) == len(churn), "must be the same length as churn (a transform, not an aggregate)"
assert np.allclose(calls_vs_contract, churn["support_calls"] - manual_avg), "support_calls minus the mean of the customer's own contract group"
assert n_above == int((calls_vs_contract > 0).sum()), "count strictly above the group average"
print(f"Exercise 2 passed ({n_above} customers above their group average)")

# %% tags=["solution"]
calls_vs_contract = churn["support_calls"] - churn.groupby("contract")["support_calls"].transform("mean")
n_above = int((calls_vs_contract > 0).sum())

# %% [markdown]
# ### Exercise 3: Premium share by country (with a minimum size)
# In `customers`, compute `premium_share`: for each country the fraction of customers with `is_premium == True`, **only for countries with at least 200 customers**,
# sorted from highest to lowest share. *Hint: `filter`, or compute size and mean together and then select.*

# %% tags=["starter"]
premium_share = ...

# %%
sizes = customers.groupby("country").size()
eligible = sizes[sizes >= 200].index
expected = customers[customers["country"].isin(eligible)].groupby("country")["is_premium"].mean().sort_values(ascending=False)
assert set(premium_share.index) == set(eligible), f"expected countries {sorted(eligible)}"
assert np.allclose(premium_share.sort_index(), expected.sort_index()), "share of is_premium per country"
assert list(premium_share.values) == sorted(premium_share.values, reverse=True), "sort descending"
print("Exercise 3 passed")
print(premium_share.round(3).to_string())

# %% tags=["solution"]
big_countries = customers.groupby("country").filter(lambda d: len(d) >= 200)
premium_share = big_countries.groupby("country")["is_premium"].mean().sort_values(ascending=False)

# %% [markdown]
# ### Exercise 4: Revenue by category, independently verified
# Using the `items` table built above, compute `revenue_by_category` (Series, index = category, values = total revenue, largest first).
# Then store `top_category`, the category with the highest revenue.

# %% tags=["starter"]
revenue_by_category = ...
top_category = ...

# %%
price = dict(zip(products["product_id"], products["unit_price"]))
cat_of = dict(zip(products["product_id"], products["category"]))
manual = {}
for pid, q in zip(order_items["product_id"], order_items["quantity"]):
    manual[cat_of[pid]] = manual.get(cat_of[pid], 0.0) + q * price[pid]
assert set(revenue_by_category.index) == set(manual), "one row per category"
for c, v in manual.items():
    assert abs(revenue_by_category[c] - v) < 1e-6, f"{c}: expected {v:.2f}"
assert list(revenue_by_category.values) == sorted(revenue_by_category.values, reverse=True), "sort largest first"
assert top_category == max(manual, key=manual.get), "top category"
print(f"Exercise 4 passed (top: {top_category})")

# %% tags=["solution"]
revenue_by_category = items.groupby("category")["revenue"].sum().sort_values(ascending=False)
top_category = revenue_by_category.idxmax()

# %% [markdown]
# ### Exercise 5 (open-ended): Which segment should retention target?
# Group `churn` by any two columns you think matter (for example `contract` and `senior_citizen`, or `autopay` and `support_calls` bucketed with `cut`).
# Show a table with customers, churn rate and the **expected number of churners** (`customers * churn_rate`). In a markdown cell, recommend one segment and say why you chose
# the *count* rather than only the *rate* to decide (hint: a 90% churn rate in a segment of 10 people matters less than 40% in a segment of 2,000).

# %% [markdown]
# ## Recap
# - `groupby` = split, apply, combine. Nothing is computed until you aggregate.
# - `agg(name=("col", "func"))` for readable summaries; `size` counts rows, `count` counts non-nulls.
# - `transform` keeps the original length (group statistics on every row); `filter` keeps or drops whole groups; `head(n)` after sorting gives top-N per group.
# - Traps: `NaN` keys are dropped unless `dropna=False`; never average rates without weights; for categorical keys, `observed=False` lists unused categories (pandas 3 hides them by default).
#
# **Go deeper:** pandas user guide "Group by: split-apply-combine" (pandas.pydata.org/docs/user_guide/groupby.html).
#
# **Next:** [Merge, join and concat](05_merge_join_concat.ipynb)
