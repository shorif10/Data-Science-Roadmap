# %% [markdown]
# # SQL Analytics Patterns
#
# **Why this matters.** In a real analytics or data-science job you rarely write a clever one-off query; you write the same handful of *patterns* again and again: retention cohorts,
# funnels, customer segmentation, "top N per group", year-over-year comparisons, streaks and gaps. Recognising the pattern (and having the skeleton in your fingers) is what makes you fast in an interview
# take-home and on the job.
#
# **You will learn**
# - build a **cohort retention** table (first-order month vs months since)
# - compute a **funnel** and step-to-step conversion, and be honest about what a snapshot funnel can and cannot say
# - segment customers with **RFM** (recency, frequency, monetary) using `NTILE`
# - do **top-N per group**, **year-over-year** comparisons, and **gaps-and-islands** (streaks)
#
# **Prerequisites:** [Window functions](05_window_functions.ipynb) and [Subqueries and CTEs](04_subqueries_and_ctes.ipynb).
#
# > Data used: `datasets/shop.db` (synthetic, read-only). Revenue = `quantity * unit_price` of non-cancelled orders. The dataset stores one current price per product, so this is "revenue at today's prices".

# %%
import sqlite3

import pandas as pd

pd.set_option("display.width", 140)
con = sqlite3.connect("file:../datasets/shop.db?mode=ro", uri=True)


def q(sql, params=None):
    return pd.read_sql(sql, con, params=params)


# A reusable CTE fragment (no user values are inserted into it, so building SQL text this way is safe; see notebook 09).
REV = """
line_rev AS (
    SELECT o.order_id, o.customer_id, o.order_date, o.channel, i.product_id, i.quantity * p.unit_price AS revenue
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    JOIN products p    ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled'
)"""

q("WITH " + REV + " SELECT COUNT(*) AS lines, ROUND(SUM(revenue)) AS total_revenue FROM line_rev")

# %% [markdown]
# ## 1. Cohort retention
#
# **Question:** of the customers who first bought in a given month (their **cohort**), what share came back 1, 2, 3 ... months later? Retention curves tell you whether the product sticks, and they are
# the most common "senior analyst" SQL interview problem.
#
# **Skeleton:** (1) map every order to a month number, (2) find each customer's first month, (3) join activity to the cohort and count distinct customers per `months_since`, (4) divide by cohort size.

# %%
cohort_sql = """
WITH act AS (
    SELECT DISTINCT customer_id,
           CAST(strftime('%Y', order_date) AS INTEGER) * 12 + CAST(strftime('%m', order_date) AS INTEGER) AS mi
    FROM orders WHERE status <> 'cancelled'
),
first AS (SELECT customer_id, MIN(mi) AS cohort_mi FROM act GROUP BY customer_id),
cohort_size AS (SELECT cohort_mi, COUNT(*) AS size FROM first GROUP BY cohort_mi)
SELECT printf('%04d-%02d', (f.cohort_mi - 1) / 12, (f.cohort_mi - 1) % 12 + 1) AS cohort,
       a.mi - f.cohort_mi                        AS months_since,
       COUNT(DISTINCT a.customer_id)             AS active,
       s.size                                    AS cohort_size,
       ROUND(100.0 * COUNT(DISTINCT a.customer_id) / s.size, 1) AS retention_pct
FROM first f
JOIN act a         ON a.customer_id = f.customer_id
JOIN cohort_size s ON s.cohort_mi = f.cohort_mi
GROUP BY f.cohort_mi, a.mi - f.cohort_mi
ORDER BY f.cohort_mi, months_since
"""
cohort = q(cohort_sql)
matrix = cohort.pivot(index="cohort", columns="months_since", values="retention_pct")
size = cohort.groupby("cohort").cohort_size.first().rename("size")
pd.concat([size, matrix.loc[:, 0:6]], axis=1).iloc[[0, 1, 5, 10, 17, 22]]

# %% [markdown]
# *What just happened?* Month 0 is always 100% (the cohort by definition ordered in month 0). Read each row left to right to see how many customers ordered again. Lower-triangular missing values are
# not zeros: those months have not happened yet (**right-censoring**), so never average them in as 0. Also look at `size`: the earliest cohorts contain few customers, so their percentages jump around. Quote the size next to a retention rate.
#
# Retention here counts "ordered in month k", not "ordered in every month up to k". That is one of several valid definitions; state yours. Another thing to notice: **re-ordering in a given month is seasonal** (a November cohort member may come back because it is Black Friday),
# so compare cohorts of the same calendar period when you can.

# %% [markdown]
# ## 2. Funnel by status
#
# A **funnel** counts how many entities reach each successive step. Our `orders.status` is the *current* state of each order: delivered, shipped, returned or cancelled. That is a snapshot, not an event log, so we
# define steps from end states. (A true funnel needs an event table with a row per step and timestamp; say so if asked.)
#
# Steps: **placed** (every order) -> **not cancelled** -> **dispatched** (shipped, delivered or returned) -> **delivered** (delivered or returned) -> **kept** (delivered and not returned).
# Note "shipped" orders are still in transit: they have not reached `delivered` yet, which lowers that step for recent orders.

# %%
funnel = q("""
SELECT channel,
       COUNT(*)                                                                AS placed,
       SUM(status <> 'cancelled')                                              AS not_cancelled,
       SUM(status IN ('shipped', 'delivered', 'returned'))                     AS dispatched,
       SUM(status IN ('delivered', 'returned'))                                AS delivered,
       SUM(status = 'delivered')                                               AS kept
FROM orders GROUP BY channel ORDER BY placed DESC
""")
funnel

# %%
steps = ["placed", "not_cancelled", "dispatched", "delivered", "kept"]
conv = funnel.set_index("channel")[steps]
step_conv = (conv.div(conv.shift(axis=1)).iloc[:, 1:] * 100).round(1)
overall = (conv[steps].iloc[:, -1] / conv["placed"] * 100).round(1).rename("end_to_end_pct")
pd.concat([step_conv, overall], axis=1)

# %% [markdown]
# Notice `dispatched` equals `not_cancelled` (a 100% step): this dataset has no status such as "paid but not yet shipped", so that step carries no information. A step that never drops is a sign your funnel definition is too coarse.
# Step conversion (each step / previous step) shows *where* orders leak; end-to-end (kept / placed) shows the overall yield. Look for the channel with the lowest number in the first columns before
# drawing conclusions, and remember the differences are only as solid as the group sizes.

# %% [markdown]
# ## 3. RFM segmentation
#
# **RFM** scores each customer on **R**ecency (days since last order), **F**requency (number of orders) and **M**onetary value (total spend), usually into quartiles 1 to 4, where 4 is best.
# It is a fast, explainable segmentation that marketing teams love. `NTILE(4)` does the bucketing. For Recency a *smaller* number of days is better, so we sort in the opposite direction.

# %%
rfm_sql = """
WITH """ + REV + """,
cust AS (
    SELECT customer_id,
           CAST(julianday((SELECT MAX(order_date) FROM orders)) - julianday(MAX(order_date)) AS INTEGER) AS recency_days,
           COUNT(DISTINCT order_id)                                                                      AS frequency,
           SUM(revenue)                                                                                  AS monetary
    FROM line_rev GROUP BY customer_id
),
scored AS (
    SELECT *,
           NTILE(4) OVER (ORDER BY recency_days DESC, customer_id) AS r,
           NTILE(4) OVER (ORDER BY frequency, customer_id)         AS f,
           NTILE(4) OVER (ORDER BY monetary, customer_id)          AS m
    FROM cust
)
SELECT *,
       CASE WHEN r >= 3 AND f >= 3 AND m >= 3 THEN 'champion'
            WHEN r <= 2 AND f >= 3             THEN 'at_risk_loyal'
            WHEN r >= 3 AND f <= 2             THEN 'recent_light'
            ELSE 'other' END AS segment
FROM scored
"""
rfm = q(rfm_sql)
seg = rfm.groupby("segment").agg(n=("customer_id", "size"), median_recency=("recency_days", "median"), median_orders=("frequency", "median"), median_spend=("monetary", "median")).round(0)
print(seg.to_string())
assert (seg.n > 0).all() and len(seg) == 4, "every segment should have customers; adjust the rules if not"

# %% [markdown]
# Read the medians to *name* each segment sensibly, and always check that no segment is empty or tiny before you ship a definition (an empty segment means your thresholds do not match your data).
# Caveats an experienced analyst states: quartile buckets are relative (they always make a "top 25%", even in a bad business); ties are split arbitrarily by `NTILE` (we added `customer_id` as a stable tiebreaker);
# and frequency and monetary are strongly correlated, so the three scores are less independent than they look.

# %% [markdown]
# ## 4. Top-N per group
# "Best three products by revenue in each category" = aggregate, then `ROW_NUMBER() OVER (PARTITION BY ...)`, then filter (you saw this in notebook 05).

# %%
q("""
WITH """ + REV + """,
prod AS (
    SELECT p.category, p.product_name, SUM(l.revenue) AS revenue
    FROM line_rev l JOIN products p ON p.product_id = l.product_id
    GROUP BY p.product_id
),
ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC, product_name) AS rn FROM prod)
SELECT category, rn, product_name, ROUND(revenue) AS revenue FROM ranked WHERE rn <= 2 ORDER BY category, rn
""")

# %% [markdown]
# ## 5. Year-over-year
# Comparing a month with the **same month last year** removes seasonality, which matters a lot for this shop (peaks in November and December). Growing "versus last month" in November would look fantastic every year for free.
# We join the monthly table to itself one year back (safer than `LAG(12)` because a missing month would shift `LAG`).

# %%
yoy = q("""
WITH """ + REV + """,
monthly AS (SELECT strftime('%Y-%m', order_date) AS month, SUM(revenue) AS revenue FROM line_rev GROUP BY month)
SELECT m.month, ROUND(m.revenue) AS revenue, ROUND(p.revenue) AS last_year,
       ROUND(100.0 * (m.revenue - p.revenue) / p.revenue, 1) AS yoy_pct
FROM monthly m
JOIN monthly p ON p.month = strftime('%Y-%m', date(m.month || '-01', '-1 year'))
ORDER BY m.month
""")
yoy.iloc[[0, 1, 5, 6, 7, 11]]

# %% [markdown]
# The join is an **inner** join on purpose: the first year of data has no prior year, so it has no YoY. A `LEFT JOIN` would keep those months with `NULL`, which is better if you will chart all months.
# Also notice the shop is young: part of any YoY change is growth of the customer base, not just repeat behaviour.

# %% [markdown]
# ## 6. Gaps and islands
# **Islands** are runs of consecutive values (days with orders in a row); **gaps** are the holes between them. The trick: if you number the distinct days 1, 2, 3, ... with `ROW_NUMBER()` and subtract that from the date,
# consecutive days all produce **the same difference**, which becomes the island's identifier.

# %%
islands = q("""
WITH days AS (SELECT DISTINCT order_date AS d FROM orders),
numbered AS (SELECT d, julianday(d) - ROW_NUMBER() OVER (ORDER BY d) AS grp FROM days)
SELECT MIN(d) AS start_date, MAX(d) AS end_date, COUNT(*) AS n_days
FROM numbered GROUP BY grp ORDER BY start_date
""")
print(len(islands), "islands;", "longest:")
islands.sort_values(["n_days", "start_date"], ascending=[False, True]).head(3)

# %% [markdown]
# Now the **gaps**: the quiet stretch between one island's end and the next island's start, via `LAG`.

# %%
q("""
WITH days AS (SELECT DISTINCT order_date AS d FROM orders),
numbered AS (SELECT d, julianday(d) - ROW_NUMBER() OVER (ORDER BY d) AS grp FROM days),
isl AS (SELECT MIN(d) AS start_date, MAX(d) AS end_date FROM numbered GROUP BY grp)
SELECT date(end_date, '+1 day') AS gap_start, date(next_start, '-1 day') AS gap_end,
       CAST(julianday(next_start) - julianday(end_date) - 1 AS INTEGER) AS gap_days
FROM (SELECT end_date, LEAD(start_date) OVER (ORDER BY start_date) AS next_start FROM isl)
WHERE next_start IS NOT NULL
ORDER BY gap_days DESC, gap_start LIMIT 3
""")

# %% [markdown]
# The same trick finds user login streaks, consecutive months of subscription, or sensor outages. The early months of the data (low volume) are where the gaps are, which tells you something about a growing shop.

# %% [markdown]
# ## Common mistakes
#
# 1. **Counting rows instead of distinct customers** in retention (an active customer with 3 orders in a month counts once).
# 2. **Treating not-yet-observed months as 0% retention** (right-censoring).
# 3. **Comparing months with the previous month** when the business is seasonal; use year-over-year.
# 4. **Funnel steps that are not nested** (a later step larger than an earlier one), which means your definitions overlap: check that counts decrease.
# 5. **RFM with empty or tiny segments** because thresholds were copied from a blog, not from your data.
# 6. **Gaps-and-islands on non-distinct values**: duplicates break the `ROW_NUMBER` arithmetic. `SELECT DISTINCT` first.
# 7. **`LAG(x, 12)` across missing months.**

# %%
# Mistake 4: sanity-check that the funnel is nested
assert (conv.diff(axis=1).iloc[:, 1:] <= 0).all().all(), "a later funnel step is bigger than the one before it!"
print("funnel steps are nested: OK")

# %% [markdown]
# ## Interview corner
#
# **1. How would you calculate month-1 retention in SQL?**
# <details><summary>Answer</summary>
# Find each customer's first active month (cohort), get the set of (customer, active month) pairs, join them, compute months since cohort, count distinct customers with months_since = 1 and divide by cohort size. Exclude cohorts whose month 1 has not happened yet.
# </details>
#
# **2. Explain gaps and islands.**
# <details><summary>Answer</summary>
# Number the ordered distinct values with <code>ROW_NUMBER()</code> and subtract the number from the value (date minus row number). Consecutive values yield the same difference, so <code>GROUP BY</code> that difference gives each run (island); gaps are the space between successive islands (use <code>LAG</code>/<code>LEAD</code>).
# </details>
#
# **3. What is RFM and what are its weaknesses?**
# <details><summary>Answer</summary>
# Scoring customers on recency, frequency and monetary value, usually quartiles, then combining scores into segments. It is simple and explainable, but relative (always produces a top group), ignores product/category and seasonality, and the three axes are correlated.
# </details>
#
# **4. How do you return the top 3 rows per group, and what about ties?**
# <details><summary>Answer</summary>
# <code>ROW_NUMBER()</code> (or <code>DENSE_RANK</code>/<code>RANK</code> to keep ties) partitioned by group and ordered by the metric with a tiebreaker, filtered in an outer query or with <code>QUALIFY</code>.
# </details>

# %% [markdown]
# ## Exercises

# %%
customers = q("SELECT * FROM customers")
products = q("SELECT * FROM products")
orders = q("SELECT * FROM orders")
items = q("SELECT * FROM order_items")
lr = (items.merge(orders, on="order_id").merge(products, on="product_id").query("status != 'cancelled'")
      .assign(revenue=lambda d: d.quantity * d.unit_price, month=lambda d: d.order_date.str[:7]))

# %% [markdown]
# ### Exercise 1: Best sellers per category
# For each category return the **top 3 products by revenue** (non-cancelled orders): `category`, `product_name`, `revenue` (rounded to 2 decimals), `rank_in_category` (1 = best).
# Order by `category`, `rank_in_category`. *Hint: aggregate by product first, then `ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC)`.*

# %% tags=["starter"]
ex1_sql = """
WITH ...
SELECT ...
"""

# %%
res = q(ex1_sql)
pr = lr.groupby(["category", "product_name"]).revenue.sum().reset_index().sort_values(["category", "revenue"], ascending=[True, False])
pr["rk"] = pr.groupby("category").cumcount() + 1
exp = pr[pr.rk <= 3].reset_index(drop=True)
assert list(res.columns) == ["category", "product_name", "revenue", "rank_in_category"], f"columns: {list(res.columns)}"
assert len(res) == len(exp), f"expected {len(exp)} rows (3 per category)"
assert res.product_name.tolist() == exp.product_name.tolist(), "wrong products or order"
assert ((res.revenue - exp.revenue.round(2)).abs() < 0.02).all() and res.rank_in_category.tolist() == exp.rk.tolist()
print("Exercise 1 passed")

# %% tags=["solution"]
ex1_sql = """
WITH line_rev AS (
    SELECT i.product_id, i.quantity * p.unit_price AS revenue
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    JOIN products p    ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled'
),
prod AS (
    SELECT p.category, p.product_name, SUM(l.revenue) AS revenue
    FROM line_rev l JOIN products p ON p.product_id = l.product_id
    GROUP BY p.product_id
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) AS rank_in_category FROM prod
)
SELECT category, product_name, ROUND(revenue, 2) AS revenue, rank_in_category
FROM ranked
WHERE rank_in_category <= 3
ORDER BY category, rank_in_category
"""

# %% [markdown]
# ### Exercise 2: Month-1 retention
# For every first-order cohort month (first **non-cancelled** order, `YYYY-MM`) return `cohort`, `cohort_size`, `retained_m1` (customers of that cohort who placed a non-cancelled order in the **calendar month right after** the cohort month)
# and `retention_pct` (1 decimal). **Exclude the last month in the data** (its month 1 is not observed yet). Order by cohort.
# *Hint: reuse the month-number trick `year * 12 + month`.*

# %% tags=["starter"]
ex2_sql = """
WITH ...
SELECT ...
"""

# %%
res = q(ex2_sql)
a = lr[["customer_id", "order_date"]].assign(p=pd.to_datetime(lr.order_date).dt.to_period("M")).drop_duplicates(["customer_id", "p"])
first_p = a.groupby("customer_id").p.min().rename("cohort")
a = a.join(first_p, on="customer_id")
last_month = a.p.max()
size = first_p.value_counts().sort_index()
ret = a[(a.p - a.cohort).apply(lambda x: x.n) == 1].groupby("cohort").customer_id.nunique().reindex(size.index, fill_value=0)
exp = pd.DataFrame({"cohort": size.index.astype(str), "size": size.values, "m1": ret.values})
exp = exp[exp.cohort < str(last_month)].reset_index(drop=True)
exp["pct"] = (100 * exp.m1 / exp["size"]).round(1)
assert list(res.columns) == ["cohort", "cohort_size", "retained_m1", "retention_pct"], f"columns: {list(res.columns)}"
assert res.cohort.tolist() == exp.cohort.tolist(), f"cohorts should run {exp.cohort.iloc[0]} to {exp.cohort.iloc[-1]}"
assert res.cohort_size.tolist() == exp["size"].tolist(), "cohort_size differs: count distinct customers whose FIRST order is in that month"
assert res.retained_m1.tolist() == exp.m1.tolist(), "retained_m1 differs"
assert ((res.retention_pct - exp.pct).abs() < 0.11).all()
print(f"Exercise 2 passed (average month-1 retention across cohorts {res.retention_pct.mean():.1f}%)")

# %% tags=["solution"]
ex2_sql = """
WITH act AS (
    SELECT DISTINCT customer_id,
           CAST(strftime('%Y', order_date) AS INTEGER) * 12 + CAST(strftime('%m', order_date) AS INTEGER) AS mi
    FROM orders WHERE status <> 'cancelled'
),
first AS (SELECT customer_id, MIN(mi) AS cohort_mi FROM act GROUP BY customer_id),
agg AS (
    SELECT f.cohort_mi,
           COUNT(*) AS cohort_size,
           SUM(EXISTS (SELECT 1 FROM act a WHERE a.customer_id = f.customer_id AND a.mi = f.cohort_mi + 1)) AS retained_m1
    FROM first f
    GROUP BY f.cohort_mi
)
SELECT printf('%04d-%02d', (cohort_mi - 1) / 12, (cohort_mi - 1) % 12 + 1) AS cohort,
       cohort_size, retained_m1,
       ROUND(100.0 * retained_m1 / cohort_size, 1) AS retention_pct
FROM agg
WHERE cohort_mi < (SELECT MAX(mi) FROM act)
ORDER BY cohort_mi
"""

# %% [markdown]
# ### Exercise 3: Year-over-year revenue
# For every month that has a same month one year earlier, return `month`, `revenue`, `last_year_revenue` (both rounded to 2 decimals) and `yoy_pct` (1 decimal), non-cancelled revenue, ordered by month.

# %% tags=["starter"]
ex3_sql = """
WITH ...
SELECT ...
"""

# %%
res = q(ex3_sql)
mm = lr.groupby("month").revenue.sum()
exp = pd.DataFrame({"month": mm.index, "rev": mm.values})
prev = mm.copy()
prev.index = [f"{int(m[:4]) + 1}-{m[5:]}" for m in prev.index]
exp = exp[exp.month.isin(prev.index)].reset_index(drop=True)
exp["ly"] = prev.loc[exp.month].values
exp["pct"] = (100 * (exp.rev - exp.ly) / exp.ly).round(1)
assert list(res.columns) == ["month", "revenue", "last_year_revenue", "yoy_pct"], f"columns: {list(res.columns)}"
assert res.month.tolist() == exp.month.tolist(), f"expected months {exp.month.iloc[0]} to {exp.month.iloc[-1]}"
assert ((res.revenue - exp.rev.round(2)).abs() < 0.02).all() and ((res.last_year_revenue - exp.ly.round(2)).abs() < 0.02).all()
assert ((res.yoy_pct - exp.pct).abs() < 0.11).all()
print(f"Exercise 3 passed (YoY ranges from {res.yoy_pct.min()}% to {res.yoy_pct.max()}%)")

# %% tags=["solution"]
ex3_sql = """
WITH line_rev AS (
    SELECT o.order_date, i.quantity * p.unit_price AS revenue
    FROM orders o
    JOIN order_items i ON i.order_id = o.order_id
    JOIN products p    ON p.product_id = i.product_id
    WHERE o.status <> 'cancelled'
),
monthly AS (SELECT strftime('%Y-%m', order_date) AS month, SUM(revenue) AS revenue FROM line_rev GROUP BY month)
SELECT m.month,
       ROUND(m.revenue, 2) AS revenue,
       ROUND(p.revenue, 2) AS last_year_revenue,
       ROUND(100.0 * (m.revenue - p.revenue) / p.revenue, 1) AS yoy_pct
FROM monthly m
JOIN monthly p ON p.month = strftime('%Y-%m', date(m.month || '-01', '-1 year'))
ORDER BY m.month
"""

# %% [markdown]
# ### Exercise 4: The longest order streak
# Using gaps-and-islands on the distinct `order_date`s of **all** orders, return one row: `start_date`, `end_date`, `n_days` for the **longest run of consecutive calendar days with at least one order**
# (if tied, the earliest run).

# %% tags=["starter"]
ex4_sql = """
WITH ...
SELECT ...
"""

# %%
res = q(ex4_sql)
d = pd.to_datetime(pd.Series(sorted(orders.order_date.unique())))
grp = (d.diff().dt.days != 1).cumsum()
runs = d.groupby(grp).agg(start="min", end="max", n="size").sort_values(["n", "start"], ascending=[False, True]).reset_index(drop=True)
best = runs.iloc[0]
assert list(res.columns) == ["start_date", "end_date", "n_days"] and len(res) == 1, "one row with start_date, end_date, n_days"
assert res.n_days[0] == best.n, f"longest streak is {best.n} days, got {res.n_days[0]}"
assert res.start_date[0] == best.start.strftime("%Y-%m-%d") and res.end_date[0] == best.end.strftime("%Y-%m-%d"), "wrong dates"
print(f"Exercise 4 passed: {res.n_days[0]} consecutive days from {res.start_date[0]}")

# %% tags=["solution"]
ex4_sql = """
WITH days AS (SELECT DISTINCT order_date AS d FROM orders),
numbered AS (SELECT d, julianday(d) - ROW_NUMBER() OVER (ORDER BY d) AS grp FROM days)
SELECT MIN(d) AS start_date, MAX(d) AS end_date, COUNT(*) AS n_days
FROM numbered
GROUP BY grp
ORDER BY n_days DESC, start_date
LIMIT 1
"""

# %% [markdown]
# ### Exercise 5 (open-ended): Design your own RFM segments
# Starting from `rfm` (the DataFrame from section 3, with scores `r`, `f`, `m`), define a segment rule of your own (for example "big spenders who have gone quiet") with pandas or SQL, check that **no segment is empty**,
# and report the size and median spend of each. In two sentences: what action would marketing take for each segment, and what would make you distrust the segmentation?

# %% [markdown]
# ## Recap
# - **Cohort retention**: first-active month per customer, months-since, count **distinct** customers, divide by cohort size; unobserved months are not zeros.
# - **Funnel**: nested step counts and step-to-step conversion; a status snapshot is not an event log.
# - **RFM**: recency/frequency/monetary quartiles with `NTILE` (stable tiebreakers), segments named from their medians, never empty.
# - **Top-N per group**: aggregate, `ROW_NUMBER` over a partition, filter in an outer query.
# - **Year-over-year** beats month-over-month when seasonal; join on the month one year back.
# - **Gaps and islands**: date minus `ROW_NUMBER` identifies runs.
#
# **Go deeper:** practise these on public datasets (see this module's README); the SQLite documentation on date and time functions (sqlite.org/lang_datefunc.html).
#
# **Next:** [SQL from Python](09_sql_from_python.ipynb)
