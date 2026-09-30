# %% [markdown]
# # Tableau Project: The "Customer & Revenue" Dashboard
#
# **Why this matters.** A dashboard is finished when a stakeholder can make a decision with it **and** trust its numbers. Most BI failures are not visual: two tabs disagree, "customers" means three things, a filter
# quietly changes a denominator. This project gives you a written spec, agreed KPI definitions, **ground-truth numbers computed locally** (to compare with what your Tableau workbook shows), the signatures of the usual mistakes, a
# story outline and a review rubric you can apply to your own work or a teammate's.
#
# **You will learn**
# - turn a stakeholder request into decisions, questions, KPI definitions and a page layout
# - generate **acceptance targets** for each KPI and chart and several filter states, and export them as CSV
# - diagnose a wrong number by matching it to a known mistake signature
# - structure a Tableau **story** (a sequence of captioned points) around one argument
# - review a dashboard against a rubric with a reconcile-to-ground-truth check
#
# **Prerequisites:** [Tableau concepts](01_tableau_concepts_and_data_prep.ipynb), [calculated fields and LODs](02_calculated_fields_and_lod_expressions.ipynb), [charts, dashboards and interactivity](03_charts_dashboards_and_interactivity.ipynb).
#
# > **Candour box (read this).** Tableau is a licensed GUI product and was **not run** while writing this module. Interface steps, calculation syntax and publishing behaviour are written from general knowledge of the product and are
# > tagged `no-run`: **verify them in the current Tableau documentation** (help.tableau.com). No pricing, licensing or limit claims are made. The **target numbers below are computed from the data by pandas and do run**. Your finished workbook must match them.
# > Data: synthetic shop tables (`docs/DATASETS.md`).

# %%
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
EXPORTS = Path("exports")
EXPORTS.mkdir(exist_ok=True)

customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

lines = (
    items.merge(orders, on="order_id").merge(products, on="product_id").merge(customers[["customer_id", "country", "is_premium"]], on="customer_id")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"], cost=lambda d: d["quantity"] * d["unit_cost"])
)
done = lines[lines["status"].isin(["delivered", "shipped"])].copy()                 # completed orders: the data source filter
done["month"] = done["order_date"].dt.to_period("M").dt.to_timestamp()
print(f"{len(lines):,} order lines, {len(done):,} completed; {orders['order_id'].nunique():,} orders, {done['order_id'].nunique():,} completed")

# %% [markdown]
# ## 1. The brief
#
# **Stakeholder:** VP of Customer Growth. **Decision to support:** whether next quarter's budget goes to *acquiring new customers* or to *bringing existing customers back*.
#
# **The dashboard must answer four questions** (a visual that answers none of them gets deleted):
# 1. How much revenue do we earn, from how many customers, at what order value and margin? (headline KPIs)
# 2. Is revenue coming from **new** customers or **returning** ones, month by month? (the decision)
# 3. Where does value concentrate: which countries and which customers? (targeting)
# 4. How common is repeat buying, and are premium customers different? (the lever)
#
# **Out of scope:** forecasting, individual customer drill-downs (privacy), profitability by product.
#
# ## 2. KPI definitions (write these on the dashboard)
#
# | KPI | Definition | Caveat |
# |---|---|---|
# | **Net Revenue** | sum of `quantity * unit_price` on completed lines (status delivered or shipped) | excludes cancelled and returned orders |
# | **Orders** | distinct completed orders | `COUNT` of lines is wrong (one order has several lines) |
# | **Active Customers** | distinct customers with at least one completed order in the filter | a distinct count: not additive across categories |
# | **AOV** | Net Revenue / Orders | not the average of line values |
# | **Revenue per Active Customer** | Net Revenue / Active Customers | |
# | **Gross Margin %** | (revenue - cost) / revenue | uses catalogue `unit_cost`; a simplification |
# | **Repeat Customer Rate** | share of active customers with 2 or more completed orders | depends on the period shown: a short window undercounts repeats |
# | **Premium Revenue Share** | revenue from premium customers / Net Revenue | |
# | **Problem-Order Rate** | cancelled plus returned orders / all orders | uses all statuses, unlike the others |

# %% tags=["no-run"]
TABLEAU_KPI_CALCS = """
// Data source filter: [Status] = 'delivered' OR [Status] = 'shipped'   (applies to everything except Problem-Order Rate: see note)
Net Revenue:          SUM([Quantity] * [Unit Price])
Orders:               COUNTD([Order ID])
Active Customers:     COUNTD([Customer ID])
AOV:                  [Net Revenue] / [Orders]
Margin %:             (SUM([Quantity] * [Unit Price]) - SUM([Quantity] * [Unit Cost])) / SUM([Quantity] * [Unit Price])

// customer-level helper fields (FIXED LODs, see notebook 02)
Customer Orders:      { FIXED [Customer ID] : COUNTD([Order ID]) }
Is Repeat:            IF [Customer Orders] >= 2 THEN 1 ELSE 0 END          // careful: a FIXED ignores dimension filters
Repeat Rate:          SUM([Is Repeat]) / COUNTD([Customer ID])             // wrong: SUM over lines; see the comment below

// Repeat Rate needs one row per customer: use a customer-grain aggregate instead, for example
Repeat Customers:     COUNTD(IF [Customer Orders] >= 2 THEN [Customer ID] END)
Repeat Rate (fixed):  [Repeat Customers] / [Active Customers]

// New vs Returning: compare each order's month with the customer's first order month
First Order Month:    DATETRUNC('month', { FIXED [Customer ID] : MIN([Order Date]) })
Customer Type:        IF DATETRUNC('month', [Order Date]) = [First Order Month] THEN 'New' ELSE 'Returning' END

// Problem-Order Rate needs ALL statuses, so it cannot sit behind the completed-only data source filter.
// Options: a second data source / relationship without the filter, or build the filter as a normal filter instead. Decide and document.
"""

# %% [markdown]
# Note the trap in the last comment: one KPI (problem orders) needs the very rows that the headline filter removes. In a real project you catch this in the **spec**, not after the build.

# %% [markdown]
# ## 3. Ground-truth KPIs
# Computed with a slightly different method per KPI (orders counted at order grain, customers from a per-customer table) so it is independent of how you build it in Tableau.

# %%
def compute_kpis(df, all_orders=None):
    """KPIs for a set of completed lines. `all_orders` (optional): all-status orders for the problem-order rate."""
    n_orders = df["order_id"].nunique()
    per_cust = df.groupby("customer_id")["order_id"].nunique()
    rev = df["revenue"].sum()
    out = {
        "Net Revenue": rev,
        "Orders": n_orders,
        "Active Customers": len(per_cust),
        "AOV": rev / n_orders if n_orders else np.nan,
        "Revenue per Active Customer": rev / len(per_cust) if len(per_cust) else np.nan,
        "Gross Margin %": (rev - df["cost"].sum()) / rev if rev else np.nan,
        "Repeat Customer Rate": (per_cust >= 2).mean() if len(per_cust) else np.nan,
        "Premium Revenue Share": df.loc[df["is_premium"], "revenue"].sum() / rev if rev else np.nan,
    }
    if all_orders is not None:
        out["Problem-Order Rate"] = all_orders["status"].isin(["cancelled", "returned"]).mean()
    return out

orders_c = orders.merge(customers[["customer_id", "country", "is_premium"]], on="customer_id")
kpis = pd.Series(compute_kpis(done, orders_c), name="value")
kpis.to_frame().assign(display=lambda d: [f"{v:,.2f}" if abs(v) > 10 else f"{v:.1%}" for v in d["value"]])

# %% [markdown]
# These are the numbers your headline cards must show (Net Revenue, Orders, Active Customers, AOV, and so on). Some sanity checks we can make *before* any tool is involved: margin between 0 and 1, rates between 0 and 1, AOV times orders equals revenue.

# %%
assert 0 < kpis["Gross Margin %"] < 1 and 0 <= kpis["Repeat Customer Rate"] <= 1 and 0 <= kpis["Premium Revenue Share"] <= 1 and 0 <= kpis["Problem-Order Rate"] <= 1
assert abs(kpis["AOV"] * kpis["Orders"] - kpis["Net Revenue"]) < 1e-6
assert abs(kpis["Revenue per Active Customer"] * kpis["Active Customers"] - kpis["Net Revenue"]) < 1e-6
print("KPI sanity checks passed")

# %% [markdown]
# ## 4. The decision: new vs returning revenue
# Question 2 needs a per-customer attribute (the first order month) attached to every order: a `FIXED` LOD in Tableau (notebook 02), `groupby(...).transform("min")` in pandas. Then classify each order's revenue by comparing its month with that first month.

# %%
done["first_month"] = done.groupby("customer_id")["month"].transform("min")                # { FIXED [Customer ID] : MIN(month) }
done["customer_type"] = np.where(done["month"] == done["first_month"], "New", "Returning")
nr = done.pivot_table(index="month", columns="customer_type", values="revenue", aggfunc="sum", fill_value=0.0)
monthly_total = done.groupby("month")["revenue"].sum()
assert np.allclose(nr.sum(axis=1), monthly_total), "new + returning must equal monthly revenue"
nr["returning_share"] = nr["Returning"] / (nr["New"] + nr["Returning"])
print(f"overall: New {nr['New'].sum() / monthly_total.sum():.1%} / Returning {nr['Returning'].sum() / monthly_total.sum():.1%} of revenue")
nr.round(2).iloc[[0, 1, 5, 11, 17, 23]]

# %% [markdown]
# *What just happened?* Early months are almost all "New" (there is no history yet), and the returning share grows as the customer base ages. That is a **data-history effect**, so a fair dashboard shows the trend, not a single all-time split, and its caption says
# that the first months are new by construction. Reading the decision off this chart: how does the returning share in the last 6 months compare with the first 6?

# %%
early, late = nr["returning_share"].iloc[:6].mean(), nr["returning_share"].iloc[-6:].mean()
print(f"mean monthly returning share: first 6 months {early:.1%}, last 6 months {late:.1%}")

# %% [markdown]
# ## 5. Where does value concentrate? (Question 3 and 4)

# %%
ltv = done.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
top_n = int(np.ceil(0.2 * len(ltv)))
top20_share = ltv.iloc[:top_n].sum() / ltv.sum()
country = done.groupby("country").agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"), customers=("customer_id", "nunique"))
country["aov"] = country["revenue"] / country["orders"]
country["rev_per_customer"] = country["revenue"] / country["customers"]
print(f"top 20% of customers ({top_n} of {len(ltv)}) generate {top20_share:.1%} of revenue")
country.sort_values("revenue", ascending=False).round(2)

# %% [markdown]
# Note two different readings of "country performance": total revenue (size) and revenue per customer (quality). A sorted bar of revenue with a label of revenue per customer tells both in one view. Premium vs non-premium:

# %%
prem = done.groupby("is_premium").agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique"), orders=("order_id", "nunique"))
prem["orders_per_customer"] = prem["orders"] / prem["customers"]
prem["rev_per_customer"] = prem["revenue"] / prem["customers"]
prem.round(2)

# %% [markdown]
# ## 6. Visual targets and acceptance files
# Every chart gets target numbers. We assemble them in one tidy table and export two CSVs (`exports/`): KPI targets and visual targets. After building the workbook, compare each mark with its row.

# %%
monthly_tbl = done.groupby("month").agg(revenue=("revenue", "sum"), orders=("order_id", "nunique"), customers=("customer_id", "nunique"))
def rows(visual, metric, series):
    return pd.DataFrame({"visual": visual, "key": [str(k) for k in series.index], "metric": metric, "value": series.astype(float).round(2).to_numpy()})

nr_long = nr[["New", "Returning"]].stack()
nr_long.index = [f"{t} {m:%Y-%m}" for m, t in nr_long.index]
vis = pd.concat([
    rows("Monthly revenue", "Net Revenue", monthly_tbl["revenue"].set_axis(monthly_tbl.index.strftime("%Y-%m"))),
    rows("New vs returning by month", "Net Revenue", nr_long),
    rows("Revenue by country", "Net Revenue", country["revenue"]),
    rows("Revenue per customer by country", "Revenue per Active Customer", country["rev_per_customer"]),
    rows("Revenue by category", "Net Revenue", done.groupby("category")["revenue"].sum()),
], ignore_index=True)

kpis.rename_axis("kpi").reset_index().to_csv(EXPORTS / "acceptance_targets_customer_revenue_kpis.csv", index=False)
vis.to_csv(EXPORTS / "acceptance_targets_customer_revenue_visuals.csv", index=False)
print(len(kpis), "KPI targets and", len(vis), "visual targets written to exports/")
print(vis.groupby("visual").size().to_dict())

# %% [markdown]
# ## 7. Filter states: the dashboard must be right when someone clicks
# Reconcile not only the unfiltered page but several states (a country selected, premium only, a category). For each, compute the KPIs, and check that **additive** measures partition.

# %%
states = {
    "All": done,
    "Country = US": done[done["country"] == "US"],
    "Premium only": done[done["is_premium"]],
    "Category = Books": done[done["category"] == "Books"],
    "Channel = store": done[done["channel"] == "store"],
}
state_tbl = pd.DataFrame({name: pd.Series(compute_kpis(df)) for name, df in states.items()}).T
state_tbl[["Net Revenue", "Orders", "Active Customers", "AOV", "Repeat Customer Rate"]].round(3)

# %%
parts = pd.DataFrame({flag: pd.Series(compute_kpis(g)) for flag, g in done.groupby("is_premium")}).T
print(f"premium + non-premium revenue = total: {np.isclose(parts['Net Revenue'].sum(), kpis['Net Revenue'])}; orders: {parts['Orders'].sum() == kpis['Orders']}; customers: {parts['Active Customers'].sum() == kpis['Active Customers']}")
cat_parts = pd.DataFrame({c: pd.Series(compute_kpis(g)) for c, g in done.groupby("category")}).T
print(f"sum of category 'Active Customers' = {cat_parts['Active Customers'].sum():,.0f} vs total {kpis['Active Customers']:,.0f}: distinct counts are not additive across categories")

# %% [markdown]
# Premium vs non-premium partitions everything (each customer is one or the other). Category does not: a card that adds category customers will be wrong. Put this knowledge on the dashboard as a footnote.

# %% [markdown]
# ## 8. Mistake signatures: if your number is X, you made mistake Y
# Compute the wrong numbers each classic mistake produces so you can recognise them in your workbook.

# %%
sig = {
    "Orders = COUNT of lines (not COUNTD)": (len(done), kpis["Orders"]),
    "Orders = distinct orders of ALL statuses": (orders["order_id"].nunique(), kpis["Orders"]),
    "Revenue includes cancelled and returned": (lines["revenue"].sum(), kpis["Net Revenue"]),
    "AOV = average of line revenue": (done["revenue"].mean(), kpis["AOV"]),
    "Customers = all customers in the table": (len(customers), kpis["Active Customers"]),
    "Customers = COUNT of lines": (len(done), kpis["Active Customers"]),
}
pd.DataFrame([{"mistake": k, "you see": round(a, 2), "correct": round(b, 2), "error": f"{a / b - 1:+.1%}"} for k, (a, b) in sig.items()])

# %% [markdown]
# Each mistake has a distinct fingerprint, so when a card is off you can often name the cause from the size and sign of the gap, then confirm in the data source.

# %% [markdown]
# ## 9. Privacy and publishing
# - **Minimise columns.** `customers.csv` has `first_name`: a dashboard about revenue does not need it. Remove it before the data source, especially if the workbook will be shared widely.
# - **Publishing target matters.** Tableau's free public-web edition publishes to the public internet (notebook 01): never use confidential data there. Sharing inside an organisation goes through server/cloud hosting, where permissions and scheduled extract refresh are configured (verify the current product names and options).
# - **Extract refresh**: decide how often the data changes and who is responsible, and state "data as of" on the dashboard.
# - **Row-level security**: if different viewers may see different countries, test it like any other filter state: each viewer's numbers should partition the total (section 7). The mechanism (user filters, entitlement tables) is product-specific: verify in the docs.

# %%
safe = customers.drop(columns=["first_name"])
assert "first_name" not in safe.columns
safe.to_csv(EXPORTS / "customers_dashboard_safe.csv", index=False, date_format="%Y-%m-%d")
print("dashboard-safe customer table columns:", list(safe.columns))

# %% [markdown]
# ## 10. The layout spec
#
# ```
# +--------------------------------------------------------------------------------+
# | Customer & Revenue   (period, filters shown, "data as of ...")                 |
# +----------+----------+----------+----------+----------+----------+--------------+
# | Net Rev  | Orders   | Active   | AOV      | Margin % | Repeat   | Premium share|
# |          |          | Customers|          |          | Rate     |              |
# +----------+----------+----------+----------+----------+----------+--------------+
# | New vs returning revenue by month (stacked bar)   | Revenue by country (sorted  |
# | caption: first months are 'new' by construction   | bar, label = per customer)  |
# +---------------------------------------------------+-----------------------------+
# | Revenue per customer: premium vs not  | Customer value concentration (Pareto)   |
# +---------------------------------------+-----------------------------------------+
# | Footnotes: definitions, completed orders only, distinct counts not additive      |
# +--------------------------------------------------------------------------------+
# ```
# **Interactions:** clicking a country filters the other views (a filter action; say so in a caption); a clearly labelled way to reset; no more than three quick filters (period, country, premium).

# %% tags=["no-run"]
TABLEAU_PROJECT_STEPS = """
1. Connect to exports/order_lines_flat.csv (notebook 01) or the four tables with relationships. Convert id fields to dimensions. Apply the completed-orders data source filter.
2. Create the calculated fields from section 2 (KPI calcs, FIXED helper fields). Verify names and LOD behaviour in the current docs.
3. Build one worksheet per visual in the layout; give each a descriptive name (not 'Sheet 7').
4. Build a KPI card sheet per KPI (or one sheet with a measure names/values layout: verify the approach in the docs).
5. Assemble the dashboard on a fixed size chosen for the target device; add the country bar as a filter for the other sheets; add captions and footnotes.
6. Compare every card and mark with exports/acceptance_targets_customer_revenue_*.csv for at least three filter states.
7. Publish only after the review rubric below scores well; decide the audience and permissions first.
// Written from general knowledge; none of this was executed.
"""

# %% [markdown]
# ## 11. The story: an argument in five points
# A Tableau **story** is an ordered set of points, each a sheet or dashboard with a caption. The same discipline as a good slide deck: one claim per point, evidence on the page, and a recommendation at the end.
# The captions below are built from the computed numbers (so they cannot drift from the data):

# %%
story = [
    ("1. The business today", f"{kpis['Net Revenue']:,.0f} net revenue from {kpis['Active Customers']:,.0f} customers across {done['order_id'].nunique():,} orders (AOV {kpis['AOV']:,.0f})."),
    ("2. Where revenue comes from", f"Returning customers' share of monthly revenue rose from {early:.0%} (first 6 months) to {late:.0%} (last 6 months); early months are new by construction."),
    ("3. Who matters most", f"The top 20% of customers produce {top20_share:.0%} of revenue; {country['revenue'].idxmax()} is the largest country with {country['revenue'].max() / country['revenue'].sum():.0%} of revenue."),
    ("4. The repeat lever", f"{kpis['Repeat Customer Rate']:.0%} of active customers ordered at least twice; premium customers place {prem.loc[True, 'orders_per_customer']:.1f} orders each vs {prem.loc[False, 'orders_per_customer']:.1f} for non-premium."),
    ("5. Recommendation and caveats", "Weigh retention spend against acquisition using the returning-share trend; caveat: synthetic data, completed orders only, repeat rate depends on the window."),
]
for title, caption in story:
    print(f"{title}\n   {caption}")

# %% [markdown]
# Whether the *recommendation* is right is a business judgement; the story's job is to make the evidence for it easy to inspect. Each caption states a number you can trace to a cell above.

# %% [markdown]
# ## 12. The review rubric
# Use it on your own dashboard and on colleagues' (be kind and specific). Score each criterion between 0 (missing) and 1 (fully met). The weights and the thresholds below are our own suggested starting point, not an industry standard: adjust them for your team.

# %%
RUBRIC = pd.DataFrame([
    ("Answers the stated questions (and nothing else)", 15, "Every visual maps to a question in the brief"),
    ("Numbers reconcile to ground truth in several filter states", 20, "KPI cards and marks match the acceptance CSVs, including after clicks"),
    ("Definitions documented on the dashboard", 10, "Completed-only, distinct counts, period, data-as-of"),
    ("Chart type fits the question", 10, "Compare=bar, trend=line, etc.; no pie for precise comparison"),
    ("Titles, labels, units", 10, "Titles state the finding; axes have units"),
    ("Colour and emphasis", 10, "One meaning per colour; consistent across sheets; not colour alone"),
    ("Interactions behave and are explained", 10, "Filter actions work, a reset exists, caption says what a click does"),
    ("Layout and reading order", 5, "KPIs first, then trend, then detail; fits the target size"),
    ("Simplicity and performance", 5, "Few filters and marks; loads in acceptable time"),
    ("Privacy and publishing", 5, "No unneeded personal fields; audience and permissions decided"),
], columns=["criterion", "weight", "evidence"])
assert RUBRIC["weight"].sum() == 100

def score_review(scores):
    """scores: {criterion: 0..1}. Returns (points, verdict). Missing criteria count as 0."""
    unknown = set(scores) - set(RUBRIC["criterion"])
    if unknown:
        raise KeyError(f"unknown criteria: {unknown}")
    points = sum(w * float(np.clip(scores.get(c, 0.0), 0, 1)) for c, w in zip(RUBRIC["criterion"], RUBRIC["weight"]))
    verdict = "ready to share" if points >= 85 else "fix before sharing" if points >= 60 else "rework"
    # a hard rule: numbers that do not reconcile block sharing regardless of the total
    if scores.get("Numbers reconcile to ground truth in several filter states", 0) < 0.8:
        verdict = "blocked: numbers do not reconcile"
    return round(points, 1), verdict

demo = {c: 1.0 for c in RUBRIC["criterion"]}
demo["Numbers reconcile to ground truth in several filter states"] = 0.5
print(score_review(demo))
RUBRIC

# %% [markdown]
# A dashboard can score 90 on everything else and still be **blocked** if its numbers do not reconcile: good looks never compensate for wrong numbers. That hard rule is deliberate.

# %% [markdown]
# ## Common mistakes
#
# 1. **No written definitions** ("customers" means different things on different tabs).
# 2. **Filtering away the rows a KPI needs** (problem orders, section 2).
# 3. **Headline numbers never reconciled** in any filter state.
# 4. **A single all-time "new vs returning" split** that is dominated by history length.
# 5. **Adding distinct counts** across categories.
# 6. **Personal fields left in** a widely shared data source.
# 7. **A story that is a slideshow** of unrelated charts instead of one argument.

# %%
# Mistake 4 in numbers: the all-time split depends on how long you have been collecting data
for months in (3, 12, 24):
    sub = nr.iloc[:months]
    print(f"first {months:2d} months: returning share of revenue = {sub['Returning'].sum() / (sub['New'].sum() + sub['Returning'].sum()):.1%}")

# %% [markdown]
# ## Interview corner
#
# **1. How do you make sure a KPI on your dashboard is correct?**
# <details><summary>Answer</summary>
# Write the definition, compute it independently (SQL or pandas at the right grain), compare in several filter states, check that additive measures partition, and keep a record of the targets. Investigate gaps by matching them to known mistakes (lines vs orders, statuses included, distinct counts).
# </details>
#
# **2. Your dashboard's "total customers" is smaller than the sum of customers across categories. What do you tell the stakeholder?**
# <details><summary>Answer</summary>
# Customers can buy in more than one category, so per-category distinct counts overlap; the total counts each customer once. Both are right, but they answer different questions, so label them and do not add the rows.
# </details>
#
# **3. How would you show whether growth comes from new or returning customers?**
# <details><summary>Answer</summary>
# Attach each customer's first order date with a FIXED LOD, classify each order as new (same month as first order) or returning, and plot revenue by type over time. Caveat the start of the series, where everyone is new by construction.
# </details>
#
# **4. What would make you refuse to publish a dashboard?**
# <details><summary>Answer</summary>
# Numbers that do not reconcile with an independent source, undefined KPIs, personal data that is not needed, or an audience and permissions that have not been decided.
# </details>

# %% [markdown]
# ## Exercises
# `done`, `kpis`, `compute_kpis`, `score_review` and `RUBRIC` are defined above.

# %% [markdown]
# ### Exercise 1: Do repeat customers carry the revenue?
# Store `repeat_rate` (share of active customers with 2 or more completed orders) and `repeat_revenue_share` (share of **revenue** coming from those repeat customers).
# Verify your answer with an **independent SQL query** on `../datasets/shop.db` and store `sql_repeat_customers` = the number of customers with at least 2 completed orders.

# %% tags=["starter"]
repeat_rate = ...
repeat_revenue_share = ...
sql_repeat_customers = ...

# %%
_oc = done.groupby("customer_id")["order_id"].nunique()
_rev = done.groupby("customer_id")["revenue"].sum()
_rep = _oc[_oc >= 2].index
assert abs(repeat_rate - len(_rep) / len(_oc)) < 1e-12
assert abs(repeat_revenue_share - _rev[_rep].sum() / _rev.sum()) < 1e-12
assert 0 < repeat_revenue_share <= 1 and repeat_revenue_share > repeat_rate, "repeat customers spend more, so their revenue share exceeds their headcount share"
assert int(sql_repeat_customers) == len(_rep), "SQL and pandas should agree on the number of repeat customers"
print(f"Exercise 1 passed: {repeat_rate:.1%} of customers, {repeat_revenue_share:.1%} of revenue")

# %% tags=["solution"]
_orders_per = done.groupby("customer_id")["order_id"].nunique()
_repeat_ids = _orders_per[_orders_per >= 2].index
repeat_rate = len(_repeat_ids) / len(_orders_per)
_rev_per = done.groupby("customer_id")["revenue"].sum()
repeat_revenue_share = _rev_per[_repeat_ids].sum() / _rev_per.sum()
_con = sqlite3.connect(DATA / "shop.db")
sql_repeat_customers = _con.execute("""
    SELECT COUNT(*) FROM (
        SELECT customer_id FROM orders WHERE status IN ('delivered', 'shipped') GROUP BY customer_id HAVING COUNT(DISTINCT order_id) >= 2
    )
""").fetchone()[0]
_con.close()

# %% [markdown]
# ### Exercise 2: Last month: new or returning?
# For the **last month** in the data, store `new_rev` and `returning_rev` (revenue from customers whose first completed order was that very month, and from everyone else) and `returning_share`.
# Make sure `new_rev + returning_rev` equals that month's total revenue.

# %% tags=["starter"]
new_rev = ...
returning_rev = ...
returning_share = ...

# %%
_last = done["month"].max()
_m = done[done["month"] == _last]
_first = done.groupby("customer_id")["month"].min()
_is_new = _m["customer_id"].map(_first) == _last
assert abs(new_rev - _m.loc[_is_new, "revenue"].sum()) < 1e-6
assert abs(returning_rev - _m.loc[~_is_new, "revenue"].sum()) < 1e-6
assert abs(new_rev + returning_rev - _m["revenue"].sum()) < 1e-6, "new + returning must equal the month's revenue"
assert abs(returning_share - returning_rev / _m["revenue"].sum()) < 1e-12
print(f"Exercise 2 passed: {_last:%b %Y}: new {new_rev:,.0f}, returning {returning_rev:,.0f} ({returning_share:.1%})")

# %% tags=["solution"]
_last_month = done["month"].max()
_cur = done[done["month"] == _last_month]
_fm = done.groupby("customer_id")["month"].min()
_new_mask = _cur["customer_id"].map(_fm) == _last_month
new_rev = _cur.loc[_new_mask, "revenue"].sum()
returning_rev = _cur.loc[~_new_mask, "revenue"].sum()
returning_share = returning_rev / (new_rev + returning_rev)

# %% [markdown]
# ### Exercise 3: A reconciler that catches mistakes
# Write `reconcile(actual, targets=kpis, tol=0.005)` that compares a dict of KPI values you read from your workbook with the ground truth and returns the **sorted list of KPI names whose relative difference exceeds `tol`** (0.5% by default; a KPI missing from `actual` counts as failing).
# It must return `[]` for the true values, catch `Orders` when you pass the line count, and ignore a difference of 0.001%.

# %% tags=["starter"]
def reconcile(actual, targets=kpis, tol=0.005):
    ...

# %%
_true = kpis.to_dict()
assert reconcile(_true) == [], "true values must pass"
_bad = dict(_true, Orders=len(done))
assert reconcile(_bad) == ["Orders"], "the line-count mistake must be caught (and only it)"
_tiny = dict(_true, **{"Net Revenue": _true["Net Revenue"] * 1.00001})
assert reconcile(_tiny) == [], "a 0.001% difference is within tolerance"
_missing = {k: v for k, v in _true.items() if k != "AOV"}
assert reconcile(_missing) == ["AOV"], "a missing KPI counts as failing"
_two = dict(_true, AOV=done["revenue"].mean(), **{"Net Revenue": lines["revenue"].sum()})
assert reconcile(_two) == ["AOV", "Net Revenue"], "sorted names of every failing KPI"
print("Exercise 3 passed")

# %% tags=["solution"]
def reconcile(actual, targets=kpis, tol=0.005):
    failing = []
    for name, truth in targets.items():
        if name not in actual:
            failing.append(name)
            continue
        if abs(actual[name] - truth) > tol * abs(truth):
            failing.append(name)
    return sorted(failing)

# %% [markdown]
# ### Exercise 4 (open-ended): Review a dashboard
# Take any dashboard you built (this project, or a screenshot of a public one) and score it with `score_review` (be honest, use 0 to 1 per criterion). Write three specific improvements ranked by impact, and name the **one** number on it
# that you would reconcile first and how you would compute it independently.

# %% [markdown]
# ## Recap
# - Start from the decision and the stakeholder's questions; write KPI definitions; decide what is out of scope.
# - Compute **acceptance targets** locally (KPIs, each chart, several filter states) and export them; reconcile before sharing.
# - Know the mistake fingerprints (lines vs orders, statuses, non-additive distinct counts) and the history-length effect in new vs returning splits.
# - A story is an argument in points with traceable numbers, not a slideshow.
# - Review with a rubric, and let non-reconciling numbers block publication.
#
# **Go deeper:** Tableau Help (help.tableau.com): data sources, LOD expressions, stories and publishing; Cole Nussbaumer Knaflic, *Storytelling with Data*; Stephen Few, *Information Dashboard Design*.
#
# **Try next with real public data:** Kaggle retail or e-commerce datasets and the UCI Machine Learning Repository (archive.ics.uci.edu). Rebuild this dashboard spec on one of them.
#
# **Next:** this completes the BI tools block. Continue with [Machine Learning](../12_Machine_Learning/).
