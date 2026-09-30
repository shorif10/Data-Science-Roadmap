# %% [markdown]
# # EDA Project: The E-commerce Shop
#
# **Why this matters.** A hiring manager's favourite take-home is "here is a set of tables, tell me what is going on". Exploratory data analysis (EDA) is the loop of
# *ask a question, compute the answer, sanity-check it, decide what to ask next*. This notebook walks through a complete EDA of the shop database using everything from
# the pandas module (loading, cleaning, joining, grouping, reshaping, time) and ends with **you** writing five findings.
#
# **You will learn**
# - turn four raw tables into an analysis-ready table with **row-count checks** at every step
# - answer business questions: revenue trend and seasonality, category mix, premium vs non-premium, cohorts, top customers
# - separate *growth* from *seasonality*, and *frequency* from *basket size*
# - write findings that state a number, a comparison and a caveat
#
# **Prerequisites:** all previous pandas notebooks, especially [merge/join](05_merge_join_concat.ipynb) and [reshaping and windows](06_reshaping_and_window_functions.ipynb).
#
# > Data used: `customers`, `products`, `orders`, `order_items` (synthetic, `docs/DATASETS.md`). Planted facts exist (the docs mention some), but you should *earn* each conclusion from the data, not from the docs.
# > **Definitions used throughout:** revenue = `quantity * unit_price` on lines of **non-cancelled** orders (returned orders count as gross revenue). "Orders" means non-cancelled orders.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
order_items = pd.read_csv(DATA / "order_items.csv")

# %% [markdown]
# ## Step 0. Build the analysis table, with checks
# One row per order line, with product, order and customer attributes. Every join is validated and the row count asserted.

# %%
lines = (
    order_items.merge(products, on="product_id", how="left", validate="many_to_one")
    .merge(orders, on="order_id", how="left", validate="many_to_one")
    .merge(customers, on="customer_id", how="left", validate="many_to_one")
)
assert len(lines) == len(order_items), "joins must not change the number of lines"
assert lines[["unit_price", "status", "country", "signup_date"]].notna().all().all(), "unmatched keys"
lines = lines.assign(revenue=lines["quantity"] * lines["unit_price"], month=lines["order_date"].dt.to_period("M"))
sold = lines[lines["status"] != "cancelled"].copy()
print(f"lines: {len(lines):,} | non-cancelled lines: {len(sold):,} | customers: {len(customers):,} | orders: {len(orders):,}")

# %% [markdown]
# ## Step 1. Sanity checks before any insight
# Cheap checks that catch expensive mistakes: ranges, keys, impossible values, time logic.

# %%
print("order dates:", orders["order_date"].min().date(), "to", orders["order_date"].max().date())
print("signup dates:", customers["signup_date"].min().date(), "to", customers["signup_date"].max().date())
print("statuses:", orders["status"].value_counts().to_dict())
print("price range:", products["unit_price"].min(), "-", products["unit_price"].max(), "| cost above price:", int((products["unit_cost"] > products["unit_price"]).sum()))
print("quantity values:", sorted(order_items["quantity"].unique()))
print("duplicate order ids:", int(orders["order_id"].duplicated().sum()), "| duplicate customer ids:", int(customers["customer_id"].duplicated().sum()))

first = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id", validate="many_to_one")
print("orders placed before the customer's signup date:", int((first["order_date"] < first["signup_date"]).sum()))

# %% [markdown]
# Everything is consistent. Two structural facts matter later: orders only cover **June 2023 to May 2025** while signups start in **January 2023**, and some customers never ordered.
# Note them; they limit what the cohort analysis can say.

# %% [markdown]
# ## Step 2. Revenue trend and seasonality
# First, revenue and orders by month, plus average order value (AOV = revenue per order).

# %%
order_rev = sold.groupby(["order_id", "month"], as_index=False)["revenue"].sum()
monthly = order_rev.groupby("month").agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
monthly["aov"] = monthly["revenue"] / monthly["orders"]
print(monthly.round(0).astype(int).T.iloc[:, [0, 1, 5, 6, 11, 12, 17, 18, 23]].to_string())

# %%
fig, ax = plt.subplots(figsize=(9, 3.4))
(monthly["revenue"] / 1000).plot(ax=ax, marker="o", label="monthly revenue")
(monthly["revenue"].rolling(3).mean() / 1000).plot(ax=ax, lw=2, label="3-month rolling mean")
ax.set_title("Shop revenue by month (non-cancelled orders)")
ax.set_xlabel("month")
ax.set_ylabel("revenue (thousands, shop currency)")
ax.legend()
plt.show()

# %% [markdown]
# Revenue climbs steeply *and* jumps every November and December. Two effects are tangled: **growth** (the customer base grows, so everything grows) and **seasonality** (a holiday peak).
# To separate them, divide by the number of customers who had signed up by that month (a rough per-customer measure), then compare the same month in consecutive years.

# %%
signups = customers["signup_date"].dt.to_period("M").value_counts().sort_index().cumsum()
registered = signups.reindex(monthly.index, method="ffill")            # customers registered by each month
per_customer = (monthly["revenue"] / registered).rename("revenue_per_registered_customer")
print(per_customer.round(1).loc[["2023-06", "2023-11", "2023-12", "2024-06", "2024-11", "2024-12", "2025-05"]].to_string())
yoy = (monthly["revenue"] / monthly["revenue"].shift(12) - 1).dropna()
print(f"\nrevenue yoy growth: min {yoy.min():+.0%}, max {yoy.max():+.0%} | registered customers grew {registered.iloc[-1] / registered.iloc[0] - 1:+.0%} over the period")

# %% [markdown]
# Even per registered customer, November and December stand out against neighbouring months, so there is genuine seasonality *in addition to* growth. The year-over-year jump
# is much larger for early months: the shop was still ramping up when the data begin. A caveat for any forecast: we have only **two** Novembers, so we cannot claim to know the seasonal shape precisely.
#
# Let us quantify the seasonal effect for the last full year (June 2024 to May 2025): each month's revenue as a share of the average month.

# %%
last12 = monthly["revenue"].iloc[-12:]
season_index = (last12 / last12.mean()).round(2)
print(season_index.sort_values(ascending=False).head(4).to_string())

# %% [markdown]
# ## Step 3. Category mix
# Which categories bring revenue, which bring margin, and does the mix change over time?

# %%
cat = sold.assign(margin=sold["quantity"] * (sold["unit_price"] - sold["unit_cost"])).groupby("category").agg(
    revenue=("revenue", "sum"), margin=("margin", "sum"), units=("quantity", "sum")
)
cat["revenue_share"] = cat["revenue"] / cat["revenue"].sum()
cat["margin_pct"] = cat["margin"] / cat["revenue"]
cat.sort_values("revenue", ascending=False).round(3)

# %% [markdown]
# The category with the most revenue is not the one with the highest margin percentage: a discussion for a business meeting (which category deserves promotion budget?). Does the mix shift by quarter?

# %%
q = sold.assign(quarter=sold["order_date"].dt.to_period("Q")).pivot_table(index="quarter", columns="category", values="revenue", aggfunc="sum")
q_share = q.div(q.sum(axis=1), axis=0)
print((q_share * 100).round(1).to_string())
print("largest quarter-to-quarter change in any category share (percentage points):", round(float(q_share.diff().abs().max().max() * 100), 1))

# %% [markdown]
# The shares are quite stable across quarters, so the revenue growth is spread across all categories rather than driven by one. (Stable share also means the seasonal peak is *not* a category story.)

# %% [markdown]
# ## Step 4. Premium vs non-premium customers
# A fair comparison needs the **right denominator**: all customers in each group, including those who never ordered. Otherwise you compare only survivors.

# %%
cust = customers.assign(
    orders=customers["customer_id"].map(sold.groupby("customer_id")["order_id"].nunique()).fillna(0),
    revenue=customers["customer_id"].map(sold.groupby("customer_id")["revenue"].sum()).fillna(0.0),
)
premium = cust.groupby("is_premium").agg(
    customers=("customer_id", "size"),
    ever_ordered=("orders", lambda s: (s > 0).mean()),
    orders_per_customer=("orders", "mean"),
    revenue_per_customer=("revenue", "mean"),
)
premium["aov"] = cust.groupby("is_premium")["revenue"].sum() / cust.groupby("is_premium")["orders"].sum()
premium.round(2)

# %% [markdown]
# Premium customers buy **far more often** (orders per customer) and almost all of them order at least once, but the average order value is similar. So premium is a **frequency** effect, not a basket-size effect.
# A quick check that this is not just because premium customers have been around longer:

# %%
same_cohort = cust[cust["signup_date"].between("2023-06-01", "2023-12-31")]
print(same_cohort.groupby("is_premium")["orders"].agg(["size", "mean"]).round(2).to_string())
print("returns and cancellations by segment (share of all orders):")
seg = orders.merge(customers[["customer_id", "is_premium"]], on="customer_id")
print(pd.crosstab(seg["is_premium"], seg["status"], normalize="index").round(3).to_string())

# %% [markdown]
# Restricting to customers who signed up in the same months still shows the gap, so tenure alone does not explain it. The return and cancellation shares are close: premium is not driving refunds.
# (We did **not** test whether the difference is statistically significant; with these sample sizes the frequency gap is far outside noise, but for the small return-rate differences you would want a proper test. See the Statistics module.)

# %% [markdown]
# ## Step 5. Cohorts: do customers keep buying after signup?
# A **cohort** is a group of customers who signed up in the same month. The retention question: of the customers in a cohort, what share placed an order in month 0 (the signup month), month 1, month 2, ...?
#
# Two data limits to respect: orders only start in June 2023, so cohorts **before June 2023** are missing their first months; and later cohorts have had less time to be observed.
# We use cohorts from June 2023 to December 2024 and look at months 0 to 5, which every one of them can be observed for.

# %%
c0 = customers.assign(cohort=customers["signup_date"].dt.to_period("M"))
o = sold[["order_id", "customer_id", "order_date"]].drop_duplicates("order_id").merge(c0[["customer_id", "cohort"]], on="customer_id", validate="many_to_one")
o["k"] = (o["order_date"].dt.to_period("M") - o["cohort"]).apply(lambda d: d.n)      # months since signup
cohort_size = c0.groupby("cohort").size()
active = o.groupby(["cohort", "k"])["customer_id"].nunique().unstack(fill_value=0)
retention = active.div(cohort_size, axis=0)
window = retention.loc["2023-06":"2024-12", [0, 1, 2, 3, 4, 5]]
print(window.shape)
print((window * 100).round(0).astype(int).iloc[::3].to_string())

# %%
fig, ax = plt.subplots(figsize=(6, 4.2))
im = ax.imshow(window.to_numpy() * 100, aspect="auto", cmap="viridis")
ax.set_xticks(range(6), labels=[str(k) for k in window.columns])
ax.set_yticks(range(0, len(window), 3), labels=[str(p) for p in window.index[::3]])
ax.set_title("Share of each signup cohort ordering, by months since signup")
ax.set_xlabel("months since signup")
ax.set_ylabel("signup month (cohort)")
fig.colorbar(im, label="% of cohort with an order")
plt.show()

# %% [markdown]
# Reading a cohort heatmap: each row is a cohort, each column a month since signup. Rows that look alike mean behaviour is stable across cohorts; a drift down the rows would mean
# newer customers behave differently. The pattern here also includes the **November/December** effect: cohorts whose month k falls in the holiday season show a higher share ordering.
# That is a calendar effect layered on top of "months since signup", so compare cohorts carefully.

# %% [markdown]
# ## Step 6. Top customers and concentration
# "Who are our best customers, and how dependent are we on them?" The share of revenue from the top X% is a standard concentration measure.

# %%
by_cust = cust.sort_values("revenue", ascending=False)
top10 = by_cust.head(10)[["customer_id", "country", "is_premium", "orders", "revenue"]]
print(top10.round(0).to_string(index=False))
n10 = int(len(cust) * 0.10)
print(f"top 10 customers: {by_cust['revenue'].head(10).sum() / cust['revenue'].sum():.1%} of revenue | "
      f"top 10% of customers ({n10}): {by_cust['revenue'].head(n10).sum() / cust['revenue'].sum():.1%}")
print("premium share among the top 10%:", f"{by_cust['is_premium'].head(n10).mean():.0%}", "vs premium share overall:", f"{cust['is_premium'].mean():.0%}")

# %% [markdown]
# The business is **not** dangerously concentrated in a handful of accounts (the top 10 customers hold a small share), but the top tenth of customers carries a large slice and is heavily premium.
# A retention programme aimed at premium customers protects a lot of revenue.

# %% [markdown]
# ## Common mistakes in EDA
#
# 1. **Skipping sanity checks**, then building beautiful charts on broken joins.
# 2. **Wrong denominators**: comparing groups using only customers who ordered.
# 3. **Confusing growth with seasonality**: yoy comparisons and per-customer normalisation help; two data points per month never prove a seasonal shape.
# 4. **Cohort analysis that ignores the observation window** (truncated start, unobserved future months).
# 5. **Reporting a number without a comparison or a caveat.** "Premium AOV is 404" means nothing; "premium AOV is about the same as non-premium (404 vs 400)" is a finding.
# 6. **Treating a pattern as causal.** Premium customers order more: does premium *cause* it, or do heavy buyers choose premium? The data cannot say.
# 7. **Changing definitions midway** (cancelled orders in one table, out in another). Write definitions at the top.

# %% [markdown]
# ## Interview corner
#
# **1. You are handed an unfamiliar orders database. What do you check first?**
# <details><summary>Answer</summary>
# Grain and keys of each table (unique ids, relationships), row counts before and after joins, missing values, date ranges and time logic (orders before signup), impossible values, category spellings, and how statuses (cancelled/returned) should count in revenue.
# Only then compute metrics.
# </details>
#
# **2. How would you decide whether the November spike is seasonality or growth?**
# <details><summary>Answer</summary>
# Normalise for the size of the customer base (revenue per active or registered customer), compare the same month across years (year-over-year), and look at whether other months show a similar steady climb. With little history state the uncertainty.
# </details>
#
# **3. What is a cohort retention table and what are its pitfalls?**
# <details><summary>Answer</summary>
# Rows are signup cohorts, columns are months since signup, values are the share of the cohort active. Pitfalls: truncated observation windows (recent cohorts have fewer columns), calendar effects (holidays), small cohorts, and the definition of "active".
# </details>
#
# **4. Premium customers spend more overall. What else would you want to know before recommending a premium push?**
# <details><summary>Answer</summary>
# Whether the effect is causal (selection bias: heavy buyers may self-select into premium), margin and cost of the premium benefits, whether spend is from frequency or basket size, and results from an experiment or at least a matched comparison.
# </details>

# %% [markdown]
# ## Exercises
# Each exercise builds one of the key tables of the analysis. Use `sold`, `cust`, `customers`, `orders`. The final exercise asks for your findings and runs an automated shape check on the tables you built.

# %% [markdown]
# ### Exercise 1: Monthly table
# Create `monthly_tbl`: index = month (`Period`), columns exactly `revenue`, `orders` (distinct non-cancelled orders) and `aov` (`revenue / orders`), sorted by month.

# %% tags=["starter"]
monthly_tbl = ...

# %%
rev, ords = {}, {}
for oid, m, r in zip(sold["order_id"], sold["month"], sold["revenue"]):
    rev[m] = rev.get(m, 0.0) + r
seen = {}
for oid, m in zip(sold["order_id"], sold["month"]):
    seen.setdefault(m, set()).add(oid)
assert list(monthly_tbl.columns) == ["revenue", "orders", "aov"], "columns: revenue, orders, aov"
assert len(monthly_tbl) == len(rev) and monthly_tbl.index.is_monotonic_increasing, "one row per month, sorted"
for m in list(rev)[:6]:
    row = monthly_tbl.loc[m]
    assert abs(row["revenue"] - rev[m]) < 1e-6 and row["orders"] == len(seen[m]), f"month {m}"
    assert abs(row["aov"] - rev[m] / len(seen[m])) < 1e-6, f"aov for {m}"
print(f"Exercise 1 passed ({len(monthly_tbl)} months)")

# %% tags=["solution"]
monthly_tbl = sold.groupby("month").agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"))
monthly_tbl["aov"] = monthly_tbl["revenue"] / monthly_tbl["orders"]

# %% [markdown]
# ### Exercise 2: Premium summary
# Create `premium_tbl`: index `is_premium` (False, True) with columns `customers`, `orders_per_customer` and `revenue_per_customer`. The averages must use **all** customers in each group (including those who never ordered).
# Also store `freq_ratio`: premium `orders_per_customer` divided by non-premium.

# %% tags=["starter"]
premium_tbl = ...
freq_ratio = ...

# %%
n = {False: 0, True: 0}
tot_orders = {False: 0, True: 0}
tot_rev = {False: 0.0, True: 0.0}
prem = dict(zip(customers["customer_id"], customers["is_premium"]))
for cid, p in prem.items():
    n[p] += 1
ord_by = sold.groupby("customer_id")["order_id"].nunique()
rev_by = sold.groupby("customer_id")["revenue"].sum()
for cid, k in ord_by.items():
    tot_orders[prem[cid]] += k
for cid, v in rev_by.items():
    tot_rev[prem[cid]] += v
assert list(premium_tbl.columns) == ["customers", "orders_per_customer", "revenue_per_customer"], "column names"
assert list(premium_tbl.index) == [False, True], "index: False then True"
for p in (False, True):
    assert premium_tbl.loc[p, "customers"] == n[p], f"customer count for {p}"
    assert abs(premium_tbl.loc[p, "orders_per_customer"] - tot_orders[p] / n[p]) < 1e-9, f"orders per customer for {p} (denominator = all customers)"
    assert abs(premium_tbl.loc[p, "revenue_per_customer"] - tot_rev[p] / n[p]) < 1e-6, f"revenue per customer for {p}"
assert abs(freq_ratio - (tot_orders[True] / n[True]) / (tot_orders[False] / n[False])) < 1e-9, "freq_ratio = premium / non-premium orders per customer"
print(f"Exercise 2 passed (premium customers order {freq_ratio:.1f}x as often)")

# %% tags=["solution"]
premium_tbl = cust.groupby("is_premium").agg(
    customers=("customer_id", "size"),
    orders_per_customer=("orders", "mean"),
    revenue_per_customer=("revenue", "mean"),
)
freq_ratio = premium_tbl.loc[True, "orders_per_customer"] / premium_tbl.loc[False, "orders_per_customer"]

# %% [markdown]
# ### Exercise 3: Cohort table
# Create `cohort_tbl`: the retention table for signup cohorts **2023-06 to 2024-12** (rows, `Period` index) and months since signup **0 to 5** (columns 0..5): the share of the cohort that placed at least one non-cancelled order in that month.
# Expected shape: 19 rows by 6 columns. *Hint: reuse the approach of Step 5, but write it yourself.*

# %% tags=["starter"]
cohort_tbl = ...

# %%
assert cohort_tbl.shape == (19, 6), f"expected (19, 6), got {cohort_tbl.shape}"
assert list(cohort_tbl.columns) == [0, 1, 2, 3, 4, 5], "columns are months since signup 0..5"
assert str(cohort_tbl.index[0]) == "2023-06" and str(cohort_tbl.index[-1]) == "2024-12", "cohorts 2023-06 .. 2024-12"
assert cohort_tbl.min().min() >= 0 and cohort_tbl.max().max() <= 1, "shares must be between 0 and 1"
# independent check of one cell with plain python sets
coh, k = pd.Period("2024-03"), 2
members = set(customers.loc[customers["signup_date"].dt.to_period("M") == coh, "customer_id"])
tgt = coh + k
buyers = {c for c, d in zip(sold["customer_id"], sold["order_date"]) if c in members and d.to_period("M") == tgt}
assert abs(cohort_tbl.loc[coh, k] - len(buyers) / len(members)) < 1e-9, "2024-03 cohort, month 2"
print("Exercise 3 passed")

# %% tags=["solution"]
cohort_tbl = retention.loc["2023-06":"2024-12", [0, 1, 2, 3, 4, 5]]

# %% [markdown]
# ### Exercise 4: Top customers and concentration
# Create `top_customers`: a DataFrame of the **10** customers with the highest revenue with columns `customer_id` and `revenue`, highest first (reset index). Also store `top10pct_share`: the share of total revenue from the top 10% of customers (10% of *all* customers, i.e. 200).

# %% tags=["starter"]
top_customers = ...
top10pct_share = ...

# %%
rev_by = sold.groupby("customer_id")["revenue"].sum()
exp = rev_by.sort_values(ascending=False)
assert list(top_customers.columns) == ["customer_id", "revenue"] and len(top_customers) == 10, "10 rows, columns customer_id and revenue"
assert list(top_customers["customer_id"]) == list(exp.index[:10]), "the ten best customers, best first"
assert np.allclose(top_customers["revenue"], exp.iloc[:10].to_numpy()), "revenue values"
assert abs(top10pct_share - exp.iloc[:200].sum() / rev_by.sum()) < 1e-9, "top 200 customers' revenue / total revenue"
print(f"Exercise 4 passed (top 10% of customers = {top10pct_share:.1%} of revenue)")

# %% tags=["solution"]
ranked = cust.sort_values("revenue", ascending=False)
top_customers = ranked.head(10)[["customer_id", "revenue"]].reset_index(drop=True)
top10pct_share = ranked["revenue"].head(int(len(cust) * 0.10)).sum() / cust["revenue"].sum()

# %% [markdown]
# ### Exercise 5: Write five findings (and pass the shape check)
# Add **five** strings to the list `findings`. Each finding must contain (1) a number you computed, (2) a comparison (versus what?), and (3) a caveat or a next question.
# Example shape: *"Premium customers order about X times as often as non-premium (orders per customer), but AOV is similar, so ... (caveat: ...)."*
# Then run the check cell: it verifies that you wrote five findings of reasonable length **and** that your tables from Exercises 1-4 have the expected shapes. There is no automated way to grade insight:
# re-read each finding and ask "would a manager know what to do with this?"

# %% tags=["starter"]
findings = []
# findings.append("...")

# %%
assert len(findings) == 5, f"write exactly 5 findings, you have {len(findings)}"
assert all(isinstance(f, str) and len(f) >= 60 for f in findings), "each finding needs a number, a comparison and a caveat (at least ~60 characters)"
assert any(ch.isdigit() for f in findings for ch in f), "include numbers you computed"
# structural checks on the tables built in Exercises 1-4
assert monthly_tbl.shape == (24, 3), f"monthly_tbl should be (24, 3), got {monthly_tbl.shape}"
assert premium_tbl.shape == (2, 3), f"premium_tbl should be (2, 3), got {premium_tbl.shape}"
assert cohort_tbl.shape == (19, 6), f"cohort_tbl should be (19, 6), got {cohort_tbl.shape}"
assert top_customers.shape == (10, 2), f"top_customers should be (10, 2), got {top_customers.shape}"
assert abs(monthly_tbl["revenue"].sum() - sold["revenue"].sum()) < 1e-6, "monthly revenue must add up to total revenue"
print("Exercise 5 passed: five findings written and all key tables have the expected shape")
for i, f in enumerate(findings, 1):
    print(f"{i}. {f}")

# %% tags=["solution"]
peak = monthly_tbl["revenue"].iloc[-12:]
findings = [
    f"Revenue peaks in {peak.idxmax()}: {peak.max() / peak.mean():.1f}x the average month of the last full year (seasonality), but only two years of data exist, so the exact seasonal shape is uncertain.",
    f"Premium customers place {freq_ratio:.1f}x as many orders per customer as non-premium ones, yet their average order value is similar ({premium_tbl.shape[0]} segments compared): the gap is frequency, not basket size; causality is not established.",
    f"The top 10% of customers generate {top10pct_share:.0%} of revenue while the top 10 customers alone are a small share, so risk is spread across a loyal upper tier rather than a few accounts.",
    f"Month-0 ordering in signup cohorts averages {cohort_tbl[0].mean():.0%} versus {cohort_tbl[3].mean():.0%} in month 3: most customers who buy do so soon after signup; holiday months distort individual cohort rows.",
    f"Average order value moved between {monthly_tbl['aov'].min():.0f} and {monthly_tbl['aov'].max():.0f} across months, much less than order counts did, so growth comes from more orders rather than bigger ones; next question: is it new customers or repeat buying?",
]

# %% [markdown]
# ## Recap
# - Build one analysis table with validated joins and assert row counts; sanity-check ranges, keys and time logic before any insight.
# - Separate growth from seasonality (normalise per customer, compare year over year) and admit how little history you have.
# - Use the right denominator (all customers in a segment); distinguish frequency from basket size.
# - Cohort tables need care with observation windows and calendar effects.
# - A finding = number + comparison + caveat. Causality needs more than a table.
#
# **Go deeper:** pandas user guide "Cookbook" (pandas.pydata.org/docs/user_guide/cookbook.html); the Statistics module for significance testing; the Matplotlib and Seaborn modules for better charts.
#
# **You have finished the pandas module.** See the [module README](README.md) for a checklist and what to practise next.
#
# **Next:** [SQL for data science](../06_SQL/)
