# %% [markdown]
# # Project: The "Sales Performance" Dashboard, and Publishing It
#
# **Why this matters.** A dashboard is finished when a stakeholder can trust it, not when it looks good. Trust comes from a written spec, agreed definitions, and an
# **acceptance checklist with target numbers** that you tested *before* sharing. This capstone gives you all three for the shop data: you build the report in Power BI Desktop, and this notebook
# tells you exactly which numbers every card, chart and slicer state must show, and how to diagnose the mistake when they do not.
#
# **You will learn**
# - turn a business request into questions, KPIs with precise definitions, and a page layout spec
# - generate ground-truth **acceptance targets** for every KPI under several slicer states, and export them to compare with your report
# - diagnose a wrong number by matching it to a known mistake signature
# - explain publishing, refresh, sharing and **row-level security** (RLS), and test the RLS logic before deploying it
#
# **Prerequisites:** all previous Power BI notebooks ([data model](01_power_bi_overview_and_data_model.ipynb), [DAX](03_dax_fundamentals.ipynb), [time intelligence](04_time_intelligence_and_visual_design.ipynb)).
#
# > **Candour box.** Power BI Desktop and the Service were **not run** here. Everything about their menus, sharing model, licences, refresh limits and security roles is described
# > **conceptually and must be verified in the current Microsoft documentation** (learn.microsoft.com/power-bi); no licence or pricing claims are made. The **target numbers below are computed
# > by pandas from the data and do run**: they are what your finished report must match.
# > Data: synthetic shop tables (`docs/DATASETS.md`).

# %%
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA = Path("../datasets")
EXPORTS = Path("exports")
EXPORTS.mkdir(exist_ok=True)

customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

sales = (
    items.merge(orders, on="order_id").merge(products, on="product_id")
    .merge(customers[["customer_id", "country", "is_premium"]], on="customer_id")
    .assign(year=lambda d: d["order_date"].dt.year, year_month=lambda d: d["order_date"].dt.strftime("%Y-%m"),
            amount=lambda d: d["quantity"] * d["unit_price"], cost=lambda d: d["quantity"] * d["unit_cost"])
)

# make sure the load files from notebook 01 exist (same content), so this notebook works standalone
fact = sales[["item_id", "order_id", "order_date", "customer_id", "product_id", "status", "channel", "quantity"]].sort_values("item_id")
if not (EXPORTS / "fact_sales.csv").exists():
    fact.to_csv(EXPORTS / "fact_sales.csv", index=False, date_format="%Y-%m-%d")
    customers.to_csv(EXPORTS / "dim_customer.csv", index=False, date_format="%Y-%m-%d")
    products.to_csv(EXPORTS / "dim_product.csv", index=False)
    print("wrote fact_sales, dim_customer, dim_product (dim_date comes from notebook 01)")
sales.shape

# %% [markdown]
# ## 1. The brief
#
# **Stakeholder:** Head of Sales. **Decision the dashboard supports:** where to spend next quarter's promotion budget.
#
# **Questions the page must answer** (if a visual does not answer one of these, delete it):
# 1. How much have we sold, and at what margin? (headline)
# 2. Is revenue growing? Is the seasonality what we expect? (trend, versus prior year)
# 3. Which categories and products carry the revenue? (composition)
# 4. Which channel (web, mobile, store) performs best on order value? (comparison)
# 5. How much of revenue comes from premium customers? (segment)
# 6. How much is lost to returns? (quality)
#
# **KPI definitions** (the contract: same as notebook 03):
#
# | KPI | Definition |
# |---|---|
# | Net Revenue | quantity x unit price for orders with status `delivered` or `shipped` |
# | Margin % | (Net Revenue - Net Cost) / Net Revenue, cost = quantity x unit cost |
# | Orders | distinct completed orders |
# | AOV | Net Revenue / Orders |
# | Premium Share | Net Revenue from premium customers / Net Revenue |
# | Return Rate | Sales Amount of `returned` orders / Sales Amount of all orders |

# %% [markdown]
# ## 2. Layout spec
#
# **Page 1: Overview** (16:9 canvas)
# ```
# +--------------------------------------------------------------+
# | Title: "Net revenue is seasonal and Electronics-led"  [Year] [Channel] [Category] slicers |
# +----------+----------+----------+----------+----------+-------+
# | Net Rev  | Margin % | Orders   | AOV      | Premium  | Return|   <- 6 KPI cards (+ vs prior year where available)
# +----------+----------+----------+----------+ Share    | Rate  |
# | Line: net revenue by month, with prior-year line       | Bar: net revenue by category (sorted) |
# +--------------------------------------------------------+---------------------------------------+
# | Bar: AOV by channel                | Table: Top 10 products by net revenue (drill-through to Page 3)|
# +------------------------------------+-----------------------------------------------------------+
# | footnote: "Net revenue excludes cancelled and returned orders. Data refreshed: <date>"          |
# ```
# **Page 2: Customers** (premium vs standard, country bar, new customers by signup month), **Page 3: Product detail** (drill-through target: units, revenue, margin by month for the chosen product).
# Interactions: slicers sync across pages 1-2; cross-filter on; drill-through from product.

# %% [markdown]
# ## 3. Acceptance targets: what your report must show
#
# Below, `kpis()` is an **independent pandas implementation** of the definitions. We evaluate it for several slicer states. In Power BI, set the same slicers and compare each card.
# Tolerance: currency to the cent (formatting may round), ratios to 4 decimals.

# %%
COMPLETED = ["delivered", "shipped"]


def kpis(df):
    d = df[df["status"].isin(COMPLETED)]
    rev, cost, n = d["amount"].sum(), d["cost"].sum(), d["order_id"].nunique()
    prem = d.loc[d["is_premium"], "amount"].sum()
    total = df["amount"].sum()
    ret = df.loc[df["status"] == "returned", "amount"].sum()
    nan = np.nan
    return {
        "net_revenue": rev,
        "margin_pct": (rev - cost) / rev if rev else nan,
        "orders": n,
        "aov": rev / n if n else nan,
        "premium_share": prem / rev if rev else nan,
        "return_rate": ret / total if total else nan,
    }


SCENARIOS = {
    "All data": lambda s: s,
    "Year = 2024": lambda s: s[s["year"] == 2024],
    "Year = 2025": lambda s: s[s["year"] == 2025],
    "Channel = web": lambda s: s[s["channel"] == "web"],
    "Category = Books": lambda s: s[s["category"] == "Books"],
    "2024 + mobile + Electronics": lambda s: s[(s["year"] == 2024) & (s["channel"] == "mobile") & (s["category"] == "Electronics")],
    "Country = US": lambda s: s[s["country"] == "US"],
}
targets = pd.DataFrame({name: kpis(f(sales)) for name, f in SCENARIOS.items()}).T
targets.assign(net_revenue=targets["net_revenue"].round(2), margin_pct=targets["margin_pct"].round(4), aov=targets["aov"].round(2),
               premium_share=targets["premium_share"].round(4), return_rate=targets["return_rate"].round(4))

# %% [markdown]
# *What just happened?* Seven slicer states, six KPIs each: **42 numbers** to verify. If all 42 match, your measures and relationships are right; if one fails, the scenario tells you where to look
# (a failure only under `Channel = web` points at a relationship or a measure that overrides a filter; a failure only in `Year = 2025` points at the date table or a YoY guard).
#
# Now the **visual-level targets** (what each chart's bars and points should be), plus invariants that must hold at any filter state:

# %%
done = sales[sales["status"].isin(COMPLETED)]
by_month = done.groupby("year_month")["amount"].sum()
by_category = done.groupby("category")["amount"].sum().sort_values(ascending=False)
aov_channel = done.groupby("channel").apply(lambda g: g["amount"].sum() / g["order_id"].nunique(), include_groups=False)
top10 = done.groupby("product_name")["amount"].sum().nlargest(10)

# invariants (a report that violates these has a modelling bug)
assert abs(by_month.sum() - targets.loc["All data", "net_revenue"]) < 1e-6, "months must sum to total"
assert abs(by_category.sum() - targets.loc["All data", "net_revenue"]) < 1e-6, "categories must sum to total"
print(f"months on the line chart: {len(by_month)}; categories: {len(by_category)}; top-10 products carry {top10.sum() / by_category.sum():.1%} of net revenue")
print("\nTop 10 products by net revenue")
top10.round(2).to_frame("net_revenue")

# %% [markdown]
# Write everything to `exports/` so you can open the targets next to your report (or load a visual's exported data and diff it with a script, as below).

# %%
long = [{"scenario": s, "kpi": k, "expected": v} for s, row in targets.iterrows() for k, v in row.items()]
pd.DataFrame(long).to_csv(EXPORTS / "acceptance_targets_kpis.csv", index=False)
visual_rows = (
    [{"visual": "revenue_by_month", "key": k, "expected": v} for k, v in by_month.items()]
    + [{"visual": "revenue_by_category", "key": k, "expected": v} for k, v in by_category.items()]
    + [{"visual": "aov_by_channel", "key": k, "expected": v} for k, v in aov_channel.items()]
    + [{"visual": "top10_products", "key": k, "expected": v} for k, v in top10.items()]
)
pd.DataFrame(visual_rows).to_csv(EXPORTS / "acceptance_targets_visuals.csv", index=False)
print(sorted(p.name for p in EXPORTS.iterdir()))


def compare(reported: dict, scenario: str, rel=1e-4, abs_cent=0.01):
    """Compare KPI values you read from your report with the target for a slicer state."""
    out = {}
    for k, want in targets.loc[scenario].items():
        got = reported.get(k)
        tol = abs_cent if k == "net_revenue" else rel * max(1.0, abs(want))
        out[k] = "not provided" if got is None else ("PASS" if abs(got - want) <= tol else f"FAIL (report {got:,.4f} vs target {want:,.4f})")
    return out


# a pretend report that has not excluded 'returned' orders from Net Revenue, everything else right
pretend = {"net_revenue": float(sales.loc[sales["status"] != "cancelled", "amount"].sum()), "orders": targets.loc["All data", "orders"]}
compare(pretend, "All data")

# %% [markdown]
# ## 4. Diagnosing a wrong number: mistake signatures
#
# Each common modelling or DAX mistake produces a **characteristic wrong value**. Compute them once and you can recognise them in the wild: match the number on your card to a row below.

# %%
d_all = sales
signatures = {
    "correct Net Revenue": done["amount"].sum(),
    "included cancelled + returned (used Sales Amount)": d_all["amount"].sum(),
    "excluded only cancelled": d_all.loc[d_all["status"] != "cancelled", "amount"].sum(),
    "excluded only returned": d_all.loc[d_all["status"] != "returned", "amount"].sum(),
    "only 'delivered' (forgot shipped)": d_all.loc[d_all["status"] == "delivered", "amount"].sum(),
    "SUM(qty) x AVERAGE(price) on completed": done["quantity"].sum() * done["unit_price"].mean(),
}
orders_sig = {
    "correct Orders": done["order_id"].nunique(),
    "counted lines instead of distinct orders": len(done),
    "counted all statuses": sales["order_id"].nunique(),
}
aov_sig = {
    "correct AOV": done["amount"].sum() / done["order_id"].nunique(),
    "revenue / lines (units of the wrong grain)": done["amount"].sum() / len(done),
    "simple average of channel AOVs": aov_channel.mean(),
}
pd.concat({"Net Revenue": pd.Series(signatures), "Orders": pd.Series(orders_sig), "AOV": pd.Series(aov_sig)}).round(2).to_frame("value")

# %% [markdown]
# *What just happened?* When a stakeholder says "your revenue is different from Finance", the first move is to compute these alternatives. Often the difference is a **definition** (which statuses count) rather than a bug,
# and the fix is a sentence in the footnote of the page: that is why the layout has one.

# %% [markdown]
# ## 5. Publishing and sharing (concepts; verify all details)
#
# | Step | What happens | Notes |
# |---|---|---|
# | **Publish** | Desktop sends the report and its semantic model to a **workspace** in the Service | A workspace is the shared container; choose one per team/project |
# | **Refresh** | The model reloads data on a schedule (for Import tables) | Cloud sources refresh directly; on-premises sources need a **gateway**. Check limits for your licence |
# | **Incremental refresh** | Only recent partitions reload, defined with `RangeStart`/`RangeEnd` parameters in Power Query | Big win when history is large and static |
# | **Share** | Give viewers access to the report/app, or distribute an **app** from the workspace | Separate *who edits* (workspace roles) from *who views* (app / report sharing) |
# | **Row-level security** | A role with a DAX filter limits rows a viewer can see | Applies to viewers; verify how it interacts with workspace roles in the docs |
# | **Certify / endorse** | Mark a semantic model as trusted | Reuse one certified model across reports instead of copies |
#
# **Why incremental refresh matters, in numbers.** In a daily refresh only the newest data changes. Compare the share of rows in the last 30 days with the whole table:

# %%
last_day = sales["order_date"].max()
recent = sales[sales["order_date"] > last_day - pd.Timedelta(days=30)]
print(f"rows in the last 30 days of data: {len(recent)} of {len(sales)} ({len(recent) / len(sales):.1%}) -> a full reload re-processes the other {1 - len(recent) / len(sales):.0%} for nothing")

# %% [markdown]
# **Row-level security, tested before you deploy it.** Suppose country managers may see only their own country. In Power BI you define a role on `Customer` with a filter such as `[country] = "US"` (or a *dynamic* role that
# looks up the signed-in user with `USERPRINCIPALNAME()` in a mapping table). Test the logic first: each role's total must be a **partition** of the overall total, and every country needs an owner or its revenue is invisible to everyone.

# %% tags=["no-run"]
RLS_DAX = """
-- role "US Managers": filter expression on table Customer
[country] = "US"

-- dynamic security: a mapping table Manager(email, country) related to Customer
[country] = LOOKUPVALUE ( Manager[country], Manager[email], USERPRINCIPALNAME () )
"""

# %%
rls = done.groupby("country")["amount"].sum().sort_values(ascending=False)
print(rls.round(2).to_dict())
assert abs(rls.sum() - done["amount"].sum()) < 1e-6, "roles must partition the total"
print(f"OK: {len(rls)} country roles sum to the overall total; countries with no manager would silently disappear from every manager's view")

# %% [markdown]
# ## 6. Pre-publish review checklist
#
# | Area | Check |
# |---|---|
# | Numbers | All 42 KPI targets and the visual targets match; invariants hold (months and categories sum to total) |
# | Definitions | Footnote states inclusion rules; KPI names match the business glossary |
# | Model | Star schema; unique dimension keys; no (Blank) categories; date table marked |
# | Design | Title states the message; KPIs top-left; sorted bars; consistent formats; alt text |
# | Interactivity | Slicers few and synced as intended; drill-through works; reset state is obvious |
# | Performance | No high-cardinality slicers; visuals load in a few seconds (check with the Performance Analyzer; verify name) |
# | Security | RLS tested with "View as"; no sensitive columns in the model; sharing scope reviewed |
# | Operations | Refresh scheduled and monitored; owner and contact named on the page; refresh date shown |

# %% [markdown]
# ## Common mistakes
#
# 1. **Building visuals before agreeing the definitions.** Write the KPI table first (as above).
# 2. **Publishing without a numeric acceptance test.** "It looks right" is not evidence.
# 3. **One giant page with everything.** Split by question; use drill-through for detail.
# 4. **Copies of the same model across reports**, each drifting. Share one certified model.
# 5. **Testing RLS as the author.** Authors with edit rights may see everything; test using the "view as role" facility or a test account.
# 6. **No refresh owner.** A stale dashboard is worse than none, because it is trusted.

# %%
# Demonstration of mistake 1: how much the answer depends on one definition
print(f"Net revenue under 'completed only': {done['amount'].sum():,.0f}; under 'everything not cancelled': {sales.loc[sales['status'] != 'cancelled', 'amount'].sum():,.0f}; "
      f"gap: {sales.loc[sales['status'] != 'cancelled', 'amount'].sum() / done['amount'].sum() - 1:+.1%}")

# %% [markdown]
# ## Interview corner
#
# **1. How do you make sure a dashboard's numbers are right before release?**
# <details><summary>Answer</summary>
# Agree written KPI definitions, compute the same figures independently (SQL or pandas) for several filter states, compare card by card with tolerances, check invariants (parts sum to totals), and review the model (keys, blanks, date table). Record the test in a checklist that stays with the report.
# </details>
#
# **2. Explain row-level security in Power BI.**
# <details><summary>Answer</summary>
# Roles hold DAX filter expressions on tables; when a user assigned to a role views the report, those filters are applied to the model so they see only permitted rows. Dynamic RLS maps the signed-in user (USERPRINCIPALNAME) to allowed rows through a lookup table. Test with "view as role", and remember that people with edit permissions on the workspace may not be restricted by it: verify the rules in the documentation.
# </details>
#
# **3. Import vs DirectQuery: how do you choose?**
# <details><summary>Answer</summary>
# Import is fast and supports full DAX but needs refresh and has size limits; DirectQuery gives fresher data and handles huge sources at the price of slower visuals and source load. Default to Import; use DirectQuery when data is too large or must be real-time, and consider composite or incremental-refresh options.
# </details>
#
# **4. A stakeholder says your revenue does not match Finance's number. What do you do?**
# <details><summary>Answer</summary>
# Establish definitions first: which statuses, which date (order vs ship), currency, tax, returns. Reproduce both numbers from the source with explicit filters, find the gap by reconciling category by category or status by status, then document the agreed definition on the page.
# </details>

# %% [markdown]
# ## Exercises
# Helpers: `sales` (with `amount`, `cost`, `year`), `kpis`, `targets`, `compare`, `done`, `COMPLETED`.

# %% [markdown]
# ### Exercise 1: A second route to a target
# Compute the KPIs for `Year = 2024` and `Channel = web` (both slicers applied) **with SQL on DuckDB** rather than pandas. Store `sql_result` as a dict with keys `net_revenue`, `orders`, `aov`. (Register `sales` with `con.register("sales", sales)`.)
# Then the checker compares with `kpis()`. Two independent implementations agreeing is how you trust a target.

# %% tags=["starter"]
con = duckdb.connect()
con.register("sales", sales)
sql_result = ...

# %%
_exp = kpis(sales[(sales["year"] == 2024) & (sales["channel"] == "web")])
assert abs(sql_result["net_revenue"] - _exp["net_revenue"]) < 1e-6, f"net_revenue: {sql_result['net_revenue']} vs {_exp['net_revenue']}"
assert sql_result["orders"] == _exp["orders"]
assert abs(sql_result["aov"] - _exp["aov"]) < 1e-9
print(f"Exercise 1 passed (2024 web: net revenue {_exp['net_revenue']:,.2f}, {_exp['orders']} orders)")

# %% tags=["solution"]
con = duckdb.connect()
con.register("sales", sales)
_r = con.sql("""
    SELECT SUM(amount) AS net_revenue, COUNT(DISTINCT order_id) AS orders
    FROM sales WHERE year = 2024 AND channel = 'web' AND status IN ('delivered', 'shipped')
""").fetchone()
sql_result = {"net_revenue": _r[0], "orders": _r[1], "aov": _r[0] / _r[1]}

# %% [markdown]
# ### Exercise 2: Which mistake is this?
# A colleague's card shows **Net Revenue = `reported_value`** (below). Using the signatures, decide which mistake produced it. Store the matching key from the `signatures` dict (exact text) in `diagnosis`.

# %% tags=["starter"]
reported_value = float(sales.loc[sales["status"] != "returned", "amount"].sum())
diagnosis = ...

# %%
_matches = [k for k, v in signatures.items() if abs(v - reported_value) < 0.005]
assert len(_matches) == 1, "the reported value should match exactly one signature"
assert diagnosis == _matches[0], f"expected {_matches[0]!r}, got {diagnosis!r}"
print("Exercise 2 passed:", diagnosis)

# %% tags=["solution"]
reported_value = float(sales.loc[sales["status"] != "returned", "amount"].sum())
diagnosis = next(k for k, v in signatures.items() if abs(v - reported_value) < 0.005)

# %% [markdown]
# ### Exercise 3: Test the security roles
# For row-level security by **channel** (roles `web`, `mobile`, `store`), compute `rls_by_channel`: a dict of net revenue per channel. Store `rls_ok` = whether the roles sum to the overall net revenue (within 1 cent),
# and `largest_role` = the channel with the most revenue. Then answer `store_share` = the store role's share of the total (0 to 1).

# %% tags=["starter"]
rls_by_channel = ...
rls_ok = ...
largest_role = ...
store_share = ...

# %%
_t = done.groupby("channel")["amount"].sum()
assert set(rls_by_channel) == set(_t.index)
assert all(abs(rls_by_channel[c] - _t[c]) < 1e-6 for c in _t.index)
assert rls_ok is True or rls_ok == True  # noqa: E712
assert largest_role == _t.idxmax()
assert abs(store_share - _t["store"] / _t.sum()) < 1e-12
print(f"Exercise 3 passed (largest role: {largest_role}; store share {store_share:.1%})")

# %% tags=["solution"]
rls_by_channel = done.groupby("channel")["amount"].sum().to_dict()
rls_ok = bool(abs(sum(rls_by_channel.values()) - done["amount"].sum()) < 0.01)
largest_role = max(rls_by_channel, key=rls_by_channel.get)
store_share = rls_by_channel["store"] / sum(rls_by_channel.values())

# %% [markdown]
# ### Exercise 4: How much would incremental refresh save?
# Suppose you refresh the **last 90 days** of data only. Using `last_day` (the newest order date), store `rows_90` (order lines with `order_date` within the last 90 days, i.e. `> last_day - 90 days`) and `saved_share` (the share of rows you *avoid* reloading, 0 to 1).

# %% tags=["starter"]
rows_90 = ...
saved_share = ...

# %%
_cut = last_day - pd.Timedelta(days=90)
_n = int((sales["order_date"] > _cut).sum())
assert rows_90 == _n, f"expected {_n}"
assert abs(saved_share - (1 - _n / len(sales))) < 1e-12
print(f"Exercise 4 passed ({rows_90} rows reloaded, {saved_share:.1%} avoided)")

# %% tags=["solution"]
rows_90 = int((sales["order_date"] > last_day - pd.Timedelta(days=90)).sum())
saved_share = 1 - rows_90 / len(sales)

# %% [markdown]
# ### Exercise 5 (open-ended): Acceptance criteria for a new page
# Your stakeholder asks for a **Customers** page: premium vs standard revenue, top 5 countries, and new customers by signup month. Write (a) the KPI definitions, (b) five acceptance checks with the target numbers
# computed above or with pandas here, (c) one invariant that must hold, and (d) the security implication if country managers use it.

# %% [markdown]
# ## Recap
# - Start from questions and precise KPI definitions; the layout follows (KPIs top-left, one message per page, definitions in a footnote).
# - Compute **acceptance targets** independently (pandas/SQL) for several slicer states and invariants; compare card by card. `exports/acceptance_targets_*.csv` are written for you.
# - Wrong numbers have signatures: recognise the definition mistakes (statuses, grain, average-of-averages) before hunting for bugs.
# - Publish to a workspace, schedule refresh (gateway for on-premises, incremental refresh for big history), separate editors from viewers, test row-level security as a partition of the total.
# - All UI/licensing specifics: verify in the current Microsoft documentation.
#
# **Go deeper:** Microsoft Learn's Power BI documentation (publish, refresh, security) and the free Microsoft Learn learning paths for the PL-300 data analyst certification (check the current exam page for its scope).
#
# **Next:** this completes the Power BI module. Continue with [Tableau concepts and data preparation](../11_Tableau/01_tableau_concepts_and_data_prep.ipynb).
