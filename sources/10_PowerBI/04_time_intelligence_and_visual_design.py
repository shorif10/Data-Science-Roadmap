# %% [markdown]
# # Time Intelligence and Visual Design
#
# **Why this matters.** "How are we doing versus last month, last year, and year to date?" is the first question every stakeholder asks, and time
# calculations are where dashboards most often show *plausible but wrong* numbers (a YTD line that keeps running into the future, a YoY that compares a full
# year with a half year). The second question is whether people can *read* your report in five seconds. You will learn to compute the expected time-based values
# in pandas so you can check your DAX, and the design rules that make a report usable.
#
# **You will learn**
# - what a date table must satisfy and why time-intelligence DAX depends on it
# - YTD, MoM, YoY and rolling-window logic, each computed in pandas to validate the DAX
# - three time-intelligence traps in real data: incomplete years, future months, role-playing dates
# - pick the right chart for the question, and apply report design rules (slicers, drill-down, drill-through)
#
# **Prerequisites:** [DAX fundamentals](03_dax_fundamentals.ipynb) (measures, `CALCULATE`, filter context).
#
# > **Candour box.** Power BI was **not run** here; DAX blocks are `no-run` and unexecuted. Verify function names and behaviour in the DAX reference (learn.microsoft.com/dax).
# > Interface names for visuals, drill and slicer options change between releases: check the current documentation. The pandas computations **do run** and are the numbers your visuals must match.
# > Data: synthetic shop tables (`docs/DATASETS.md`). Orders run from 2023-06 to 2025-05, which matters for the incomplete-year trap below.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

done = (
    items.merge(orders, on="order_id").merge(products, on="product_id")
    .query("status in ['delivered', 'shipped']")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"])
)
print("completed lines:", len(done), "| first order date:", orders["order_date"].min().date(), "| last:", orders["order_date"].max().date())

# %% [markdown]
# ## 1. The date table contract
#
# Time-intelligence functions (`TOTALYTD`, `DATEADD`, `SAMEPERIODLASTYEAR`, ...) move *along the date table*, not along your fact table. They assume the date table:
# 1. has **one row per day**, no gaps, **unique** dates;
# 2. covers **complete calendar years** (from Jan 1 of the first to Dec 31 of the last year of your data);
# 3. is **related** to the fact's date column (one-to-many, single direction) and **marked as the date table** in the model (verify the option name in the current UI);
# 4. is used for every axis, slicer and time function: never put the fact's raw date on the axis in a time-intelligence visual.
#
# We built and exported `dim_date` in notebook 01. Here we validate the contract programmatically:

# %%
def make_dim_date(start, end):
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    d["year"], d["month_number"] = d["date"].dt.year, d["date"].dt.month
    d["year_month"] = d["date"].dt.strftime("%Y-%m")
    return d


def check_date_table(dim, fact_dates):
    days = dim["date"].diff().dropna()
    return {
        "unique": bool(dim["date"].is_unique),
        "continuous (no gaps)": bool((days == pd.Timedelta(days=1)).all()),
        "starts Jan 1": bool(dim["date"].iloc[0].month == 1 and dim["date"].iloc[0].day == 1),
        "ends Dec 31": bool(dim["date"].iloc[-1].month == 12 and dim["date"].iloc[-1].day == 31),
        "covers all fact dates": bool(fact_dates.min() >= dim["date"].min() and fact_dates.max() <= dim["date"].max()),
    }


dim_date = make_dim_date("2023-01-01", "2025-12-31")
print(check_date_table(dim_date, orders["order_date"]))
print("a date table built from the first to last ORDER date only:", check_date_table(make_dim_date(orders["order_date"].min(), orders["order_date"].max()), orders["order_date"]))

# %% [markdown]
# The second table fails the "whole years" test, which is exactly the mistake of generating the calendar from `MIN/MAX` of the sales dates.

# %% [markdown]
# ## 2. The monthly series and its DAX
#
# One base measure, then time-shifted variants of it. Below, the *expected* series in pandas. (Net Revenue = quantity x unit price on `delivered`/`shipped` orders, as in notebook 03.)

# %%
months = pd.period_range("2023-06", "2025-05", freq="M")
rev = done.assign(m=done["order_date"].dt.to_period("M")).groupby("m")["revenue"].sum().reindex(months)
monthly = pd.DataFrame({"net_revenue": rev})
monthly["ytd"] = monthly.groupby(monthly.index.year)["net_revenue"].cumsum()                  # TOTALYTD, restarting each Jan
monthly["mom_pct"] = monthly["net_revenue"].pct_change()                                      # vs previous month
monthly["yoy_pct"] = monthly["net_revenue"] / monthly["net_revenue"].shift(12) - 1            # vs same month last year
monthly["rolling3"] = monthly["net_revenue"].rolling(3).sum()                                 # DATESINPERIOD(..., -3, MONTH)
monthly.round(3).iloc[[0, 1, 6, 7, 12, 18, 23]]

# %% tags=["no-run"]
DAX_TIME = """
Net Revenue YTD    = TOTALYTD ( [Net Revenue], 'Date'[date] )
Net Revenue PM     = CALCULATE ( [Net Revenue], DATEADD ( 'Date'[date], -1, MONTH ) )
Net Revenue MoM %  = DIVIDE ( [Net Revenue] - [Net Revenue PM], [Net Revenue PM] )
Net Revenue PY     = CALCULATE ( [Net Revenue], SAMEPERIODLASTYEAR ( 'Date'[date] ) )
Net Revenue YoY %  = DIVIDE ( [Net Revenue] - [Net Revenue PY], [Net Revenue PY] )
Net Revenue R3M    = CALCULATE ( [Net Revenue], DATESINPERIOD ( 'Date'[date], MAX ( 'Date'[date] ), -3, MONTH ) )
"""

# %% [markdown]
# *What just happened?* Each Power BI time measure is "the same measure, evaluated over a shifted set of dates". `ytd` is a cumulative sum that resets in January; `mom_pct` compares neighbours; `yoy_pct` compares with
# the value 12 rows back; `rolling3` sums a 3-month window. Two features of the output are **correct behaviour** that will surprise a beginner:
#
# - The first month has no MoM and the first **12** months have no YoY (**blank**, not zero), because the data starts in 2023-06. Do not "fix" blanks by replacing them with 0.
# - `ytd` for 2023 covers only Jun-Dec: comparing "2024 YTD" with "2023 YTD" *in the first half of the year* compares a full period to no data.

# %%
print("months with a YoY value:", int(monthly["yoy_pct"].notna().sum()), "of", len(monthly), "| first:", monthly["yoy_pct"].first_valid_index())
print(f"YoY in that first month: {monthly['yoy_pct'].dropna().iloc[0]:+.0%}  (a huge % because the prior-year base is tiny: the shop was ramping up; customers keep signing up through {customers['signup_date'].max().date()})")

# Independent check of YTD at Sep 2024: filter raw lines by date range instead of using cumsum
alt_ytd = done.loc[(done["order_date"] >= "2024-01-01") & (done["order_date"] <= "2024-09-30"), "revenue"].sum()
assert abs(alt_ytd - monthly.loc[pd.Period("2024-09"), "ytd"]) < 1e-6
assert abs(monthly.loc[pd.Period("2024-12"), "ytd"] - done.loc[done["order_date"].dt.year == 2024, "revenue"].sum()) < 1e-6
print(f"YTD at 2024-09 = {alt_ytd:,.2f} (matches cumulative sum); YTD at 2024-12 equals the 2024 total")

# %% [markdown]
# ### Trap 1: the YTD line that runs into the future
# Our date table extends to **2025-12-31** (whole years), but sales stop at 2025-05. `TOTALYTD` at Aug 2025 sums Jan-Aug 2025, which is the same as Jan-May: the line stays **flat** for months that have not happened,
# which reads as "growth stopped". The standard fix is to return BLANK for periods after the last sales date.

# %%
full = pd.period_range("2023-01", "2025-12", freq="M")
rev_full = rev.reindex(full)                                              # months without data stay NaN (BLANK)
ytd_naive = rev_full.fillna(0).groupby(rev_full.index.year).cumsum()      # what an unguarded YTD shows
last_sales_month = orders["order_date"].max().to_period("M")
ytd_guarded = ytd_naive.where(ytd_naive.index <= last_sales_month)        # DAX: IF ( MAX('Date'[date]) <= MAX(Sales[order_date]), [YTD] )
cmp = pd.DataFrame({"naive_ytd": ytd_naive, "guarded_ytd": ytd_guarded}).loc["2025-04":"2025-08"]
cmp.round(0)

# %% tags=["no-run"]
DAX_GUARD = """
Net Revenue YTD (guarded) =
    IF ( MAX ( 'Date'[date] ) <= MAX ( Sales[order_date] ),   -- MAX over the date table in context vs the last fact date (use ALL(Sales) if needed; verify)
         TOTALYTD ( [Net Revenue], 'Date'[date] ) )
"""

# %% [markdown]
# ### Trap 2: role-playing dates and `USERELATIONSHIP`
# Customers have a `signup_date`, also a date. The model can have **one active** relationship between `Customer` and `Date`... but the *orders* already use the active one from `Sales[order_date]`. You create a second, **inactive**
# relationship `Customer[signup_date] -> Date[date]` and activate it inside a measure. Expected values by signup month:

# %%
new_by_month = customers.assign(m=customers["signup_date"].dt.to_period("M")).groupby("m")["customer_id"].nunique()
print("months with signups:", len(new_by_month), "| total new customers:", int(new_by_month.sum()), "(= all customers:", len(customers), ")")
print("2024-03:", int(new_by_month.loc["2024-03"]), "| peak month:", new_by_month.idxmax(), int(new_by_month.max()))

# %% tags=["no-run"]
DAX_ROLE = """
New Customers =
    CALCULATE ( DISTINCTCOUNT ( Customer[customer_id] ), USERELATIONSHIP ( Customer[signup_date], 'Date'[date] ) )
"""

# %% [markdown]
# ### Trap 3: partial periods
# The last month in your data may be incomplete when you refresh mid-month; its MoM looks like a collapse. Options: show it with a "month to date" label, exclude the current month by a page filter, or compare with the
# same *day range* last month. Decide, and write the decision on the page.

# %%
last = orders["order_date"].max()
print("last order date:", last.date(), "| days of data in that month:", last.day, "of", last.days_in_month)

# %% [markdown]
# ## 3. Choosing the chart: match the question
#
# | Question | Visual | Why | Avoid |
# |---|---|---|---|
# | What is the value right now? | **Card / KPI** | One number, big | A gauge with fake precision |
# | How does it change over time? | **Line** (or column for few periods) | Slope encodes trend; time on the x-axis | Pie/donut over time |
# | Which category is largest? | **Bar (horizontal), sorted** | Length is compared accurately; long labels fit | Unsorted bars, 3D |
# | What is the share of a whole (few parts)? | **Stacked bar / 100% bar**; a donut only for 2-4 parts | Part-to-whole | Pie with many slices |
# | Two measures against each other? | **Scatter** | Reveals relationship and outliers | Dual-axis lines that imply a link |
# | Exact values needed? | **Table / matrix** | Lookup | Using it as the main story |
#
# Bars must start at **zero** (length encodes value); a truncated axis on bars exaggerates differences. Lines may use a non-zero baseline, but label it.
# The number of distinct values of a field constrains the chart. Count them:

# %%
cards = pd.Series({
    "category": products["category"].nunique(),
    "channel": orders["channel"].nunique(),
    "customer country": customers["country"].nunique(),
    "product": products["product_id"].nunique(),
    "year-month": len(months),
    "customer": len(customers),
}, name="distinct values")
cards.to_frame().assign(reasonable_visual=lambda d: pd.cut(d["distinct values"], [0, 5, 15, 30, 100, np.inf],
                        labels=["pie/donut or bar", "bar", "bar / line", "Top N bar + 'Other'", "search/table, not a chart"]).astype(str))

# %% [markdown]
# *What just happened?* The cardinality tells you the design: 5 categories can be bars or a donut, 60 products need a **Top N** filter (plus "Other" if the total matters), and 2,000 customers should never be a chart axis
# (they belong in a searchable table). The bins are a heuristic, not a rule.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
cur = monthly["net_revenue"]
axes[0].plot(cur.index.to_timestamp(), cur.values / 1000, marker="o", label="net revenue")
axes[0].plot(cur.index.to_timestamp()[12:], cur.values[:-12] / 1000, linestyle="--", label="same month, prior year")
axes[0].set(title="Net revenue by month (thousands, sales currency)", xlabel="Month", ylabel="Net revenue (thousands)")
axes[0].legend()
bycat = done.groupby("category")["revenue"].sum().sort_values()
axes[1].barh(bycat.index, bycat.values / 1000)
axes[1].set(title="Net revenue by category (sorted)", xlabel="Net revenue (thousands)", ylabel="Category")
plt.tight_layout()
plt.show()

# %% [markdown]
# Line for time with the prior-year overlay, sorted horizontal bars for categories, both zero-based where it matters and both with titled axes and units: the two most-used shapes on a business dashboard.

# %% [markdown]
# ## 4. Interactivity: slicers, cross-filtering, drill-down, drill-through
#
# | Feature | What it does | Use it for | Watch out |
# |---|---|---|---|
# | **Slicer** | A visible filter for the page/report (list, dropdown, date range) | The 2-3 dimensions people always filter by | High-cardinality slicers are slow and unusable; sync slicers only when meant |
# | **Cross-filtering / highlighting** | Clicking a bar filters/highlights other visuals on the page | Exploration | Can hide the fact that a filter is active: show it clearly |
# | **Hierarchy + drill-down** | Year > Quarter > Month (or Category > Product) inside one visual | Progressive detail | Needs a proper hierarchy on the axis field |
# | **Drill-through** | Right-click a value to jump to a detail page *carrying that filter* | Detail pages (e.g. one customer or category) | Discoverability: add a note or button |
# | **Tooltip pages** | A mini-visual on hover | Context without clutter | Keep them light |
# | **Bookmarks / buttons** | Saved states, navigation | Guided stories, view toggles | Over-engineering |
#
# Exact names and behaviours: verify in the current documentation.

# %% [markdown]
# ## 5. Report design rules that pay off
#
# 1. **One page, one message.** Title says the takeaway ("Revenue is up in Q4"), not the field name.
# 2. **KPIs top-left, detail lower/right.** People scan roughly in a Z; put the answer where eyes land first.
# 3. **Fewer visuals.** A common guideline is a handful (say 5 to 8) per page: each extra visual costs comprehension and refresh/render time.
# 4. **Consistent meaning of colour and formats.** One colour = one thing across pages; the same number format everywhere (thousands separators, % with the same decimals).
# 5. **Show the comparison.** A number with no benchmark (target, previous period) is not a KPI.
# 6. **Accessible by default.** Do not rely on colour alone; check contrast; add alt text and a logical tab order.
# 7. **Performance.** Avoid many high-cardinality visuals; keep measures simple; hide fields users should not use. Test with the Performance Analyzer (verify name).
# 8. **Annotate assumptions.** "Net revenue excludes cancelled and returned orders" should be on the page, not in your head.

# %% [markdown]
# ## Common mistakes
#
# 1. **No date table, or one built from min/max of sales** (breaks whole-year functions).
# 2. **Putting the fact table's date on the axis** while using time intelligence on the date table: measures return blanks or wrong values.
# 3. **YTD lines running flat into the future**, or YoY against an incomplete prior period. Guard and label.
# 4. **Replacing BLANK with 0** for missing comparisons, which fakes a -100% or a 0% change.
# 5. **Bar charts with a truncated axis; pies with 8 slices; unsorted bars.**
# 6. **A slicer for every field**, so nobody knows what is filtered.

# %%
# Demonstration of mistake 4: filling blanks with 0 turns "no data" into a fake collapse
fake = monthly["yoy_pct"].fillna(-1.0)
print("months that would show -100% growth after fillna(-1):", int((fake == -1.0).sum()), "- all of them have no prior-year data at all")

# %% [markdown]
# ## Interview corner
#
# **1. Why does Power BI need a separate date table?**
# <details><summary>Answer</summary>
# Time-intelligence functions shift along a continuous, unique, whole-year calendar. A fact table's dates have gaps and duplicates and do not provide attributes like month name or fiscal year. A dedicated date table, related to the fact and marked as the date table, provides both.
# </details>
#
# **2. How do you compute month-over-month and year-over-year growth?**
# <details><summary>Answer</summary>
# Define the base measure, define a "previous period" measure with <code>DATEADD</code> (or <code>SAMEPERIODLASTYEAR</code>), then <code>DIVIDE(current - previous, previous)</code>. DIVIDE returns blank when there is no prior period, which is the correct behaviour.
# </details>
#
# **3. A YTD line is flat for the last few months of the year although the year is not over. Why and how do you fix it?**
# <details><summary>Answer</summary>
# The date table extends beyond the last sales date, so YTD keeps returning the same cumulative total for future months. Return BLANK when the date in context is later than the last date with data (compare <code>MAX('Date'[date])</code> with the last fact date).
# </details>
#
# **4. When would you use USERELATIONSHIP?**
# <details><summary>Answer</summary>
# When a fact or dimension has several date columns (order date, ship date, signup date) that relate to the same date table: only one relationship can be active, so the others are created inactive and switched on inside specific measures.
# </details>

# %% [markdown]
# ## Exercises
# `monthly` (DataFrame indexed by month), `done`, `customers`, `orders`, `rev`, `dim_date`, `check_date_table` exist above.

# %% [markdown]
# ### Exercise 1: YTD and a rolling window
# Store `ytd_sep_2024` (net revenue from 2024-01-01 to 2024-09-30) and `r3m_dec_2024` (net revenue of Oct, Nov and Dec 2024 together). Compute them from the raw completed lines `done`, using date filters (not the `monthly` table),
# so the check is a real second route.

# %% tags=["starter"]
ytd_sep_2024 = ...
r3m_dec_2024 = ...

# %%
assert abs(ytd_sep_2024 - monthly.loc[pd.Period("2024-09"), "ytd"]) < 1e-6, "YTD at Sep 2024 differs from the cumulative series"
assert abs(r3m_dec_2024 - monthly.loc[pd.Period("2024-12"), "rolling3"]) < 1e-6, "rolling 3 months at Dec 2024 differs"
print(f"Exercise 1 passed (YTD Sep 2024 {ytd_sep_2024:,.0f}; Q4 2024 {r3m_dec_2024:,.0f})")

# %% tags=["solution"]
ytd_sep_2024 = done.loc[(done["order_date"] >= "2024-01-01") & (done["order_date"] <= "2024-09-30"), "revenue"].sum()
r3m_dec_2024 = done.loc[(done["order_date"] >= "2024-10-01") & (done["order_date"] <= "2024-12-31"), "revenue"].sum()

# %% [markdown]
# ### Exercise 2: Growth rates and blanks
# Store `mom_dec_2024` (Dec 2024 versus Nov 2024) and `yoy_dec_2024` (Dec 2024 versus Dec 2023) as fractions, e.g. 0.12 for +12%. Also store `first_yoy_month`: the first month (as a string like `"2024-06"`)
# for which a YoY value exists. Use only `rev` (the monthly revenue Series).

# %% tags=["starter"]
mom_dec_2024 = ...
yoy_dec_2024 = ...
first_yoy_month = ...

# %%
_p = pd.Period
assert abs(mom_dec_2024 - (rev[_p("2024-12")] / rev[_p("2024-11")] - 1)) < 1e-12, "MoM = Dec / Nov - 1"
assert abs(yoy_dec_2024 - (rev[_p("2024-12")] / rev[_p("2023-12")] - 1)) < 1e-12, "YoY = Dec 2024 / Dec 2023 - 1"
assert first_yoy_month == str(rev.index[12]), f"YoY needs 12 prior months, so it starts at {rev.index[12]}"
print(f"Exercise 2 passed (MoM {mom_dec_2024:+.1%}, YoY {yoy_dec_2024:+.1%}, first YoY month {first_yoy_month})")

# %% tags=["solution"]
mom_dec_2024 = rev[pd.Period("2024-12")] / rev[pd.Period("2024-11")] - 1
yoy_dec_2024 = rev[pd.Period("2024-12")] / rev[pd.Period("2023-12")] - 1
first_yoy_month = str(rev.index[12])

# %% [markdown]
# ### Exercise 3: New customers by signup month (role-playing date)
# Store `new_2024_03` (customers whose signup falls in March 2024) and `top3_signup_months` (list of the 3 months with the most signups, as `"YYYY-MM"` strings, in descending order of signups; ties broken by the earlier month).
# Also store `share_2024` = share of all customers who signed up in calendar year 2024.

# %% tags=["starter"]
new_2024_03 = ...
top3_signup_months = ...
share_2024 = ...

# %%
_s = customers["signup_date"]
assert new_2024_03 == int(((_s.dt.year == 2024) & (_s.dt.month == 3)).sum())
_cnt = _s.dt.strftime("%Y-%m").value_counts()
_exp = sorted(_cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
assert top3_signup_months == [k for k, _ in _exp], f"expected {[k for k, _ in _exp]}"
assert abs(share_2024 - (_s.dt.year == 2024).mean()) < 1e-12
print(f"Exercise 3 passed ({new_2024_03} signups in 2024-03; {share_2024:.1%} of customers signed up in 2024)")

# %% tags=["solution"]
_ym = customers["signup_date"].dt.strftime("%Y-%m")
new_2024_03 = int((_ym == "2024-03").sum())
_c = _ym.value_counts()
top3_signup_months = [k for k, _ in sorted(_c.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
share_2024 = float((customers["signup_date"].dt.year == 2024).mean())

# %% [markdown]
# ### Exercise 4 (open-ended): Sketch the page
# For the question "Are we growing, and where?", pick 4 to 5 visuals from the chart table, say which measure and axis each uses, where the KPI cards go, which two slicers you would add, and what the page title would say.
# Name one measure from this notebook that each visual depends on.

# %% [markdown]
# ## Recap
# - Time intelligence needs a **continuous, unique, whole-year date table**, related to the fact and used on every time axis.
# - YTD/MoM/YoY/rolling are one base measure over shifted date sets; blanks for missing comparisons are **correct**; guard future periods and label incomplete ones.
# - Role-playing dates use inactive relationships plus `USERELATIONSHIP`.
# - Choose charts by question and field cardinality: bars sorted and zero-based, lines for time, cards for the headline; Top N for long lists.
# - Design: one message per page, KPIs first, few visuals, consistent colour/format, accessible, performant, assumptions written on the page.
#
# **Go deeper:** Microsoft Learn, "Time intelligence functions (DAX)" and "Power BI report design tips" sections; SQLBI's free articles on date tables and time intelligence.
#
# **Next:** [Dashboard project and publishing](05_dashboard_project_and_publishing.ipynb)
