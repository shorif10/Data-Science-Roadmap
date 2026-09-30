# %% [markdown]
# # Snowflake Cost, Performance and Security Basics
#
# **Why this matters.** "We got a surprise bill" and "the analyst could see salary data" are the two stories that get a
# data platform attention for the wrong reasons. Both are preventable with a few habits: suspend warehouses quickly, size them for
# the workload, scan less data, cap spend with resource monitors, and give each role only the access it needs. This notebook gives
# you the reasoning tools, including a **cost calculator you can edit**.
#
# **You will learn**
# - compute what a warehouse costs from credits, size and run time, including the effect of auto-suspend
# - reason about bigger-vs-smaller warehouses and when scaling up stops paying
# - explain clustering in terms of data scanned (connecting back to pruning)
# - model roles and grants to apply least privilege, and simulate a resource monitor
#
# **Prerequisites:** [Architecture](01_snowflake_architecture_and_concepts.ipynb) and [loading](04_loading_data_stages_copy_into_and_formats.ipynb).
#
# > **Candour box, important for this notebook.** Everything about **prices, credit rates, minimum billing periods and edition features
# > is an ASSUMPTION** written into editable constants below. The author has not run Snowflake here and does not quote prices. Your real credit price depends on
# > edition, cloud, region and contract; check your account's usage views and the current pricing/docs pages. The *structure* of the calculation is what
# > you are learning; replace the numbers with your own. SQL in `no-run` cells was not executed: verify syntax in the docs.

# %%
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# %% [markdown]
# ## 1. The cost model in four lines
#
# | Cost | Driven by | Your lever |
# |---|---|---|
# | **Compute** (virtual warehouses) | credits per hour for the warehouse size x time it is running | auto-suspend, size, query efficiency, resource monitors |
# | **Storage** | average compressed data stored (tables, Time Travel history, clones that diverged) | retention settings, drop staging data, avoid needless copies |
# | **Cloud services** | metadata/optimiser work (often small, with an allowance; *verify the current rule*) | avoid thousands of tiny metadata-heavy statements |
# | **Data transfer** | moving data out to another region/cloud | keep consumers near the data |
#
# Compute is usually the lever you control most. A standard warehouse uses **credits per hour that double with each size step**
# (XS, S, M, L, ...). The table below is our **assumption to verify in the current docs**; it is exposed as a constant so you can change it.

# %%
CREDITS_PER_HOUR = {"XS": 1, "S": 2, "M": 4, "L": 8, "XL": 16, "2XL": 32}   # ASSUMPTION: verify in current docs
CREDIT_PRICE_USD = 3.00        # PLACEHOLDER, not a quote: use the price on YOUR contract
MIN_BILLED_SECONDS = 60        # ASSUMPTION: a minimum is billed each time a warehouse resumes; verify
STORAGE_USD_PER_TB_MONTH = 23  # PLACEHOLDER, not a quote: edit

pd.Series(CREDITS_PER_HOUR, name="credits/hour").to_frame().assign(usd_per_hour=lambda d: d["credits/hour"] * CREDIT_PRICE_USD)

# %% [markdown]
# ## 2. Auto-suspend: where idle money goes
#
# You are billed while the warehouse is **running**, including the idle minutes before it auto-suspends. Picture a BI warehouse serving 40 dashboard queries spread over a working day.
# Each query takes a few seconds. If auto-suspend is 1 hour, one query at 9:00 keeps the warehouse on until at least 10:00, so the warehouse is effectively on all day.
#
# Let us simulate this. `billed_seconds` merges query activity into **runs** (a run continues while a new query arrives before the idle timer expires) and applies the minimum billing per run.

# %%
def billed_seconds(queries, auto_suspend_s, min_billed=MIN_BILLED_SECONDS):
    """queries: list of (start_second, duration_seconds). Returns total billed seconds (concurrency ignored)."""
    total, run_start, run_end = 0.0, None, None
    for start, dur in sorted(queries):
        if run_start is not None and start <= run_end:          # still running: extend the idle timer
            run_end = max(run_end, start + dur + auto_suspend_s)
        else:                                                   # suspended: close previous run, resume
            if run_start is not None:
                total += max(min_billed, run_end - run_start)
            run_start, run_end = start, start + dur + auto_suspend_s
    if run_start is not None:
        total += max(min_billed, run_end - run_start)
    return total


def make_day(seed=42, n=40):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(9 * 3600, 17 * 3600, n))      # 09:00-17:00
    durs = rng.lognormal(mean=np.log(6), sigma=0.5, size=n)    # seconds, median about 6
    return list(zip(starts.tolist(), durs.tolist()))


day = make_day()
print(f"{len(day)} queries, total query time {sum(d for _, d in day):.0f} s")
rows = []
for label, suspend in [("1 min", 60), ("5 min", 300), ("10 min", 600), ("60 min", 3600)]:
    secs = billed_seconds(day, suspend)
    credits = secs / 3600 * CREDITS_PER_HOUR["M"]
    rows.append({"auto_suspend": label, "billed_hours": round(secs / 3600, 2), "credits_M": round(credits, 2), "usd": round(credits * CREDIT_PRICE_USD, 2)})
suspend_table = pd.DataFrame(rows)
suspend_table

# %% [markdown]
# *What just happened?* The same 40 queries cost very different amounts purely because of the idle timer. With a short timeout you pay mostly for actual query time plus short idle tails; with
# a long one you pay for the whole working day. The trade-off: each resume has a start-up cost and a **cold cache** (data cached on the warehouse's local disks is lost on suspend), so very aggressive suspension
# can slow the first query after a pause and cost extra minimums. The defensible default for ad-hoc/BI warehouses is a short timeout (often a minute or a few); for steady, back-to-back workloads a
# longer one may be cheaper. **Measure with your own workload.**

# %% [markdown]
# Edit-and-rerun section: change `n` (more or fewer queries) and see how the gap between timeouts shrinks as the workload gets busier (a warehouse that is always busy has little idle time to save).

# %%
for n in (5, 40, 400):
    d = make_day(n=n)
    costs = {s: round(billed_seconds(d, s) / 3600, 2) for s in (60, 600, 3600)}
    print(f"{n:4d} queries -> billed hours by auto-suspend seconds: {costs}")

# %% [markdown]
# ## 3. Bigger warehouse: faster, but is it cheaper?
#
# One size step up doubles the credits per hour. If it also halves the runtime, the **cost is unchanged and you get the answer sooner**. That is the best case (a query that parallelises perfectly).
# Real queries scale imperfectly and some do not scale at all (small data, a single-threaded step, a wait on a lock). Define `efficiency` as the fraction of the ideal 2x speed-up you actually get per
# step. *This is a model with a made-up efficiency you should measure by running your query at two sizes; it is not a Snowflake constant.*

# %%
SIZES = list(CREDITS_PER_HOUR)


def run_cost(base_seconds_xs, size, efficiency):
    k = SIZES.index(size)
    runtime = base_seconds_xs / (2 * efficiency) ** k
    credits = runtime / 3600 * CREDITS_PER_HOUR[size]
    return runtime, credits


for eff in (1.0, 0.85, 0.5):
    print(f"efficiency {eff:.2f}")
    print(pd.DataFrame([{"size": s, **dict(zip(["runtime_s", "credits"], np.round(run_cost(600, s, eff), 3)))} for s in SIZES]).to_string(index=False))

# %% [markdown]
# *What just happened?* At efficiency 1.0 every size costs the same credits and only the wait changes. At 0.5 (a query that barely benefits) each step up **doubles** the cost for no speed gain.
# So: scale up when a query is genuinely parallel and slow (and especially when it **spills to disk** for lack of memory, which the Query Profile shows), scale *out* (multi-cluster) for concurrency, and first try to scan less data.

# %% [markdown]
# ## 4. Scan less: clustering in cost terms
#
# From notebook 01: a filter prunes micro-partitions by min/max. In a large table whose physical order does not match your common filter, every query scans nearly everything. **Clustering keys** tell Snowflake to keep data
# organised by chosen columns (automatic clustering maintenance itself consumes credits, so use it only on large tables with stable, selective filters). Here is the scan fraction on our data under two layouts:

# %%
orders = pd.read_csv("../datasets/orders.csv", parse_dates=["order_date"])
items = pd.read_csv("../datasets/order_items.csv")
lines = items.merge(orders, on="order_id").sort_values(["order_date", "item_id"], ignore_index=True)


def scan_fraction(df, col, lo, hi, rows=400):
    parts = [df[col].iloc[i : i + rows] for i in range(0, len(df), rows)]
    hit = sum(1 for p in parts if p.min() <= hi and p.max() >= lo)
    return hit / len(parts)


print(f"filter on order_date (table ordered by date):       {scan_fraction(lines, 'order_date', pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-31')):.0%} of partitions")
print(f"filter on customer_id 500-520 (ordered by date):    {scan_fraction(lines, 'customer_id', 500, 520):.0%} of partitions")
print(f"filter on customer_id 500-520 (clustered by it):    {scan_fraction(lines.sort_values('customer_id'), 'customer_id', 500, 520):.0%} of partitions")

# %% tags=["no-run"]
PERF_SQL = """
-- warehouses: create small, suspend fast, start suspended
CREATE WAREHOUSE bi_wh WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE;
ALTER WAREHOUSE bi_wh SET WAREHOUSE_SIZE = 'MEDIUM';        -- resize is instant for new queries

-- clustering: only for big tables with stable selective filters; automatic clustering uses credits
ALTER TABLE order_lines CLUSTER BY (order_date);
SELECT SYSTEM$CLUSTERING_INFORMATION('order_lines', '(order_date)');

-- where did the credits go? (ACCOUNT_USAGE views lag by some time and need privileges; verify names and latency in docs)
SELECT warehouse_name, SUM(credits_used) FROM snowflake.account_usage.warehouse_metering_history
WHERE start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP()) GROUP BY 1 ORDER BY 2 DESC;
"""

# %% [markdown]
# Other performance habits worth knowing: select only the columns you need; filter early; beware join fan-out (check the row count after a join); use the **Query Profile** to see
# where time goes; repeated identical queries on unchanged data can be served from a result cache (conditions and lifetime: *verify in the docs*).

# %% [markdown]
# ## 5. Storage and Time Travel arithmetic
#
# History is not free: changed or deleted data stays stored for the retention period. A rough model for extra storage is
# `table size x fraction of data changed per day x retention days`. Edit the assumptions (they are illustrative).

# %%
def storage_monthly_usd(table_tb, daily_change_frac, retention_days, price=STORAGE_USD_PER_TB_MONTH):
    history_tb = table_tb * daily_change_frac * retention_days      # crude model
    return table_tb * price, history_tb * price


base, hist = storage_monthly_usd(table_tb=2.0, daily_change_frac=0.02, retention_days=1)
print(f"live data ${base:,.0f}/month, history at 1 day retention ${hist:,.2f}/month")
for days in (1, 7, 30, 90):
    print(f"retention {days:2d} days -> history ${storage_monthly_usd(2.0, 0.02, days)[1]:8,.2f}/month  (a table that churns 50% daily: ${storage_monthly_usd(2.0, 0.5, days)[1]:10,.2f})")

# %% [markdown]
# *What just happened?* For a slowly-changing table long retention is cheap; for a heavily rewritten table (think a staging table replaced every run) the same setting can cost more than the data itself.
# Practical habit: use a short retention (or transient tables, which have reduced history; verify current semantics) for scratch and staging data.

# %% [markdown]
# ## 6. Resource monitors: a spending cap
#
# A **resource monitor** tracks credits used by a set of warehouses (or the account) in a period and triggers actions at percentage thresholds, such as notify at 80% and suspend at 100%.
# Here is the logic simulated over a month of daily usage from our model.

# %%
def month_daily_credits(size="M", auto_suspend=600, days=30):
    return [billed_seconds(make_day(seed=s, n=int(np.random.default_rng(s).integers(10, 60))), auto_suspend) / 3600 * CREDITS_PER_HOUR[size] for s in range(days)]


daily = month_daily_credits()
cum = np.cumsum(daily)
QUOTA = round(cum[-1] * 0.9, 1)      # we deliberately set the quota below projected use so a trigger fires


def trigger_day(cum, quota, pct):
    hits = np.nonzero(cum >= quota * pct / 100)[0]
    return int(hits[0]) + 1 if len(hits) else None


print(f"projected month: {cum[-1]:.1f} credits; quota: {QUOTA} credits")
print("notify at 80% on day", trigger_day(cum, QUOTA, 80), "| suspend at 100% on day", trigger_day(cum, QUOTA, 100))

# %% tags=["no-run"]
MONITOR_SQL = """
-- needs a role with the privilege to create monitors (commonly ACCOUNTADMIN); verify in docs
CREATE RESOURCE MONITOR bi_monitor WITH CREDIT_QUOTA = 100 FREQUENCY = MONTHLY START_TIMESTAMP = IMMEDIATELY
  TRIGGERS ON 80 PERCENT DO NOTIFY
           ON 100 PERCENT DO SUSPEND;          -- suspend lets running queries finish; SUSPEND_IMMEDIATE cancels them
ALTER WAREHOUSE bi_wh SET RESOURCE_MONITOR = bi_monitor;
"""

# %% [markdown]
# A monitor is a **safety net, not a budget process**: someone must own the notification, and a hard suspend stops dashboards, so choose thresholds with the business in mind.

# %% [markdown]
# ## 7. Security: roles and least privilege
#
# Snowflake access control is **role-based**: privileges are granted to *roles*, roles are granted to *users* (and to other roles, forming a hierarchy in which the parent inherits the child's privileges).
# A user works under one *active* role at a time. Least privilege means: each role gets only what its job needs, and powerful roles are not used for daily work. Snowflake ships with system roles
# (`ACCOUNTADMIN`, `SECURITYADMIN`, `USERADMIN`, `SYSADMIN`, `PUBLIC`: *verify the exact responsibilities in the docs*). A common guidance is to keep `ACCOUNTADMIN` for a handful of trusted people, protected
# by MFA, and never use it to run pipelines.
#
# To practise the reasoning without an account, here is a **toy RBAC engine**: grants on objects, role inheritance, and a `can()` check. (The real system has far more: ownership, future grants, managed access schemas, masking policies and more.)

# %%
@dataclass
class ToyRBAC:
    grants: set = field(default_factory=set)        # (role, privilege, object_pattern)
    parents: dict = field(default_factory=dict)     # role -> set of roles it inherits from

    def grant(self, role, privilege, obj):
        self.grants.add((role, privilege, obj))

    def grant_role(self, child, parent):
        """parent role receives everything child has (like GRANT ROLE child TO ROLE parent)."""
        self.parents.setdefault(parent, set()).add(child)

    def _closure(self, role):
        seen, stack = set(), [role]
        while stack:
            r = stack.pop()
            if r not in seen:
                seen.add(r)
                stack.extend(self.parents.get(r, ()))
        return seen

    def can(self, role, privilege, obj):
        for r in self._closure(role):
            for g_role, g_priv, g_obj in self.grants:
                if g_role == r and g_priv == privilege and (g_obj == obj or (g_obj.endswith("*") and obj.startswith(g_obj[:-1]))):
                    return True
        return False


rbac = ToyRBAC()
rbac.grant("ANALYST", "SELECT", "table:SHOP.PUBLIC.*")
rbac.grant("ANALYST", "USAGE", "warehouse:BI_WH")
rbac.grant("LOADER", "INSERT", "table:SHOP.PUBLIC.*")
rbac.grant("LOADER", "USAGE", "warehouse:ETL_WH")
rbac.grant_role("ANALYST", "SENIOR_ANALYST")          # senior inherits analyst

tests = [
    ("ANALYST", "SELECT", "table:SHOP.PUBLIC.ORDERS"),
    ("ANALYST", "INSERT", "table:SHOP.PUBLIC.ORDERS"),
    ("ANALYST", "USAGE", "warehouse:ETL_WH"),
    ("SENIOR_ANALYST", "SELECT", "table:SHOP.PUBLIC.CUSTOMERS"),
    ("LOADER", "SELECT", "table:SHOP.PUBLIC.ORDERS"),
]
pd.DataFrame([(*t, rbac.can(*t)) for t in tests], columns=["role", "privilege", "object", "allowed"])

# %% [markdown]
# *What just happened?* Access is the union of what the role and everything it inherits was granted; anything not granted is denied. Notice the loader can write but **cannot read**, which is
# legitimate least privilege for a service account. The equivalent real SQL is below (*not executed, verify*).

# %% tags=["no-run"]
RBAC_SQL = """
CREATE ROLE analyst;
GRANT USAGE ON WAREHOUSE bi_wh TO ROLE analyst;
GRANT USAGE ON DATABASE shop TO ROLE analyst;
GRANT USAGE ON SCHEMA shop.public TO ROLE analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA shop.public TO ROLE analyst;
GRANT SELECT ON FUTURE TABLES IN SCHEMA shop.public TO ROLE analyst;   -- new tables too
GRANT ROLE analyst TO USER some_person;                                 -- roles go to users
-- never: GRANT ... TO PUBLIC for sensitive data; never run daily work as ACCOUNTADMIN
"""

# %% [markdown]
# Beyond roles: **masking policies** (hide column values such as emails from roles that should not see them), **row access policies**, **network policies**, **MFA/SSO**, and key-pair auth for service users.
# They are the next things to read about once grants make sense.

# %% [markdown]
# ## Common mistakes
#
# 1. **Long auto-suspend "so it is fast"**: you pay for idle hours; use caching and a short timeout, and measure.
# 2. **Sizing up to fix a query that scans too much**: fix the pruning first (filters, clustering, avoiding `SELECT *`).
# 3. **One shared huge warehouse for everything**: no isolation, no cost attribution. Separate warehouses per workload, and tag or name them.
# 4. **Doing daily work as `ACCOUNTADMIN`**, or granting broad privileges "temporarily" that stay forever.
# 5. **Long Time Travel retention on churn-heavy staging tables.**
# 6. **Resource monitor with only `NOTIFY` and nobody reading the notifications.**

# %%
# Demonstration of mistake 1: how many hours of a "business day" a long timeout keeps the warehouse on
d = make_day()
print(f"60-min timeout keeps it on for {billed_seconds(d, 3600) / 3600:.1f} h; 1-min timeout for {billed_seconds(d, 60) / 3600:.2f} h  (same {len(d)} queries)")

# %% [markdown]
# ## Interview corner
#
# **1. What drives Snowflake compute cost, and how do you control it?**
# <details><summary>Answer</summary>
# Warehouse size (credits per hour doubling per size) times running time, including idle time before suspend. Control with short auto-suspend, right-sizing per workload, separate warehouses for attribution,
# efficient queries (prune, select only needed columns), caching, and resource monitors as a safety net.
# </details>
#
# **2. Will a bigger warehouse make every query cheaper or faster?**
# <details><summary>Answer</summary>
# No. Each size step doubles the credit rate; it pays back only if runtime drops proportionally. Queries limited by data scanned, a serial step or small input gain little. It helps most when the query is large and parallel or spilling to
# disk. Measure at two sizes.
# </details>
#
# **3. Explain RBAC and least privilege in Snowflake.**
# <details><summary>Answer</summary>
# Privileges are granted to roles; roles are granted to users and to other roles (inheritance). Give each role the minimum objects and actions for its job (for example usage on one warehouse, select on one schema),
# use service roles for pipelines, reserve admin roles for a few people with MFA, and review grants regularly.
# </details>
#
# **4. What is a resource monitor and what are its limits?**
# <details><summary>Answer</summary>
# An object that tracks credit consumption for warehouses/account over a period and triggers notify or suspend actions at thresholds. It is a guardrail: it reacts to usage, does not optimise it, and a suspend affects users, so thresholds and ownership of alerts matter.
# </details>

# %% [markdown]
# ## Exercises
# `billed_seconds`, `make_day`, `CREDITS_PER_HOUR`, `SIZES`, `run_cost`, `ToyRBAC` and `rbac` are defined above.

# %% [markdown]
# ### Exercise 1: Reimplement the billing logic independently
# Write `billed_seconds_v2(queries, auto_suspend_s)` that gives the same answer as `billed_seconds` but by a **different method**: mark each second of a 24-hour day as "running" if it falls inside
# `[start, start + duration + auto_suspend)` of any query (use a NumPy boolean array, round start down and end up), count the maximal runs of running seconds, and bill at least 60 seconds per run.
# The checker accepts a difference of up to 2 seconds per run because of rounding. *Hint: `np.diff` on the padded boolean array finds run starts and ends.*

# %% tags=["starter"]
def billed_seconds_v2(queries, auto_suspend_s):
    ...

# %%
for seed in (1, 2, 3):
    q = make_day(seed=seed, n=25)
    a, b = billed_seconds(q, 600), billed_seconds_v2(q, 600)
    n_runs = len(q)   # upper bound on the number of runs
    assert abs(a - b) <= 2 * n_runs, f"seed {seed}: reference {a:.1f}s vs yours {b:.1f}s"
assert billed_seconds_v2([(1000.0, 5.0)], 0) == 60, "a single 5 second query is billed the 60 second minimum"
print("Exercise 1 passed")

# %% tags=["solution"]
def billed_seconds_v2(queries, auto_suspend_s):
    day = np.zeros(24 * 3600 + 10_000, dtype=bool)
    for start, dur in queries:
        day[int(np.floor(start)) : int(np.ceil(start + dur + auto_suspend_s))] = True
    padded = np.concatenate([[False], day, [False]]).astype(int)
    edges = np.diff(padded)
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    return float(sum(max(60, e - s) for s, e in zip(starts, ends)))

# %% [markdown]
# ### Exercise 2: Pick the size
# A query takes 1,200 seconds on an `XS` warehouse. Assuming `efficiency = 0.8` per size step (our model), find the **smallest size whose runtime is at most 120 seconds**. Store it in `best_size` and its
# credit cost in `best_credits`; also store `xs_credits` (cost of running it on XS). Then decide: `cheaper_on_bigger = True/False`.
# *Hint: loop over `SIZES` with `run_cost(1200, size, 0.8)`.*

# %% tags=["starter"]
best_size = ...
best_credits = ...
xs_credits = ...
cheaper_on_bigger = ...

# %%
_ok = [s for s in SIZES if run_cost(1200, s, 0.8)[0] <= 120]
assert _ok, "the model says no size reaches 120 s"
assert best_size == _ok[0], f"expected {_ok[0]}, got {best_size}"
assert abs(best_credits - run_cost(1200, _ok[0], 0.8)[1]) < 1e-12
assert abs(xs_credits - run_cost(1200, "XS", 0.8)[1]) < 1e-12
assert cheaper_on_bigger == (best_credits < xs_credits)
print(f"Exercise 2 passed: {best_size} takes {run_cost(1200, best_size, 0.8)[0]:.0f}s and costs {best_credits:.3f} credits vs {xs_credits:.3f} on XS")

# %% tags=["solution"]
best_size = next(s for s in SIZES if run_cost(1200, s, 0.8)[0] <= 120)
best_credits = run_cost(1200, best_size, 0.8)[1]
xs_credits = run_cost(1200, "XS", 0.8)[1]
cheaper_on_bigger = bool(best_credits < xs_credits)

# %% [markdown]
# ### Exercise 3: Least-privilege grants
# Create a **new** `ToyRBAC()` called `rbac2` and grant so that: `ANALYST` can `SELECT` any table in `table:SHOP.PUBLIC.` (use the wildcard pattern) and `USAGE` on `warehouse:BI_WH`;
# `ETL` can `INSERT` and `SELECT` on `table:SHOP.STAGING.*` and `USAGE` on `warehouse:ETL_WH`; `LEAD` inherits `ANALYST` via `grant_role`. Nobody gets anything else.

# %% tags=["starter"]
rbac2 = ToyRBAC()
# rbac2.grant(...)

# %%
checks = [
    (("ANALYST", "SELECT", "table:SHOP.PUBLIC.ORDERS"), True),
    (("ANALYST", "SELECT", "table:SHOP.STAGING.RAW"), False),
    (("ANALYST", "INSERT", "table:SHOP.PUBLIC.ORDERS"), False),
    (("ANALYST", "USAGE", "warehouse:BI_WH"), True),
    (("ANALYST", "USAGE", "warehouse:ETL_WH"), False),
    (("ETL", "INSERT", "table:SHOP.STAGING.RAW"), True),
    (("ETL", "SELECT", "table:SHOP.STAGING.RAW"), True),
    (("ETL", "SELECT", "table:SHOP.PUBLIC.ORDERS"), False),
    (("ETL", "USAGE", "warehouse:ETL_WH"), True),
    (("LEAD", "SELECT", "table:SHOP.PUBLIC.CUSTOMERS"), True),
    (("LEAD", "INSERT", "table:SHOP.STAGING.RAW"), False),
]
for args, want in checks:
    assert rbac2.can(*args) == want, f"{args}: expected {want}"
print("Exercise 3 passed")

# %% tags=["solution"]
rbac2 = ToyRBAC()
rbac2.grant("ANALYST", "SELECT", "table:SHOP.PUBLIC.*")
rbac2.grant("ANALYST", "USAGE", "warehouse:BI_WH")
rbac2.grant("ETL", "INSERT", "table:SHOP.STAGING.*")
rbac2.grant("ETL", "SELECT", "table:SHOP.STAGING.*")
rbac2.grant("ETL", "USAGE", "warehouse:ETL_WH")
rbac2.grant_role("ANALYST", "LEAD")

# %% [markdown]
# ### Exercise 4 (open-ended): Cost review
# Pick a workload (your own, or invent one: 500 queries a day, median 20 s, some at night). Using `make_day`-style simulation or your own arithmetic, estimate monthly compute credits for two warehouse
# configurations (size and auto-suspend). State your assumptions, the winner, and **one measurement you would take in a real account** to confirm it.

# %% [markdown]
# ## Recap
# - Compute cost = credits/hour (doubles per size step) x running time, **including idle time before suspend**; storage cost includes Time Travel history.
# - Short auto-suspend, right-sized and separate warehouses, and scanning less data are the main levers; a bigger warehouse pays off only if runtime shrinks proportionally.
# - Clustering helps large tables with stable selective filters, at a maintenance cost.
# - RBAC: grant privileges to roles, roles to users/roles; least privilege; keep admin roles rare. Resource monitors are guardrails, not budgets.
# - Every price and rate here is a labelled assumption: replace with your account's real numbers.
#
# **Go deeper:** docs.snowflake.com, sections on "Understanding compute cost", "Access control" and "Resource monitors".
#
# **Next:** this completes the Snowflake module. Continue with the [Power BI module](../10_PowerBI/01_power_bi_overview_and_data_model.ipynb).
