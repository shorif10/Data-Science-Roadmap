# %% [markdown]
# # The Data Engineering Landscape
#
# **Why this matters.** A model is only as good as the data that reaches it, and that data does not arrive
# by magic. Somebody has to pull it out of production databases and APIs, reshape it, store it somewhere
# analysts can query cheaply, and keep it fresh and correct. In interviews and on the job you will be asked
# to talk to the people who do this (and often to do a slice of it yourself), so you need the vocabulary
# *and* a feel for why the designs look the way they do.
#
# **You will learn**
# - tell ETL from ELT and batch from streaming, and pick one for a given requirement
# - explain OLTP vs OLAP and lake vs warehouse vs lakehouse in one sentence each
# - measure why column-oriented storage is cheaper for analytics than row-oriented storage
# - draw the line between what a data engineer owns and what a data scientist owns
#
# **Prerequisites:** [pandas basics](../05_Pandas/01_loading_and_inspecting_data.ipynb) and a little [SQL](../06_SQL/).
#
# > Data used: the synthetic shop tables (`datasets/orders.csv`, `order_items.csv`, `shop.db`). See `docs/DATASETS.md`.

# %%
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
orders = pd.read_csv(DATA / "orders.csv")
items = pd.read_csv(DATA / "order_items.csv")
print(orders.shape, items.shape)

# %% [markdown]
# ## 1. The big picture: where data flows
#
# ```
#  sources                 ingestion              storage                 serving / use
#  -------                 ---------              -------                 -------------
#  app databases (OLTP) -> batch extract   ->  raw / landing zone   ->   BI dashboards
#  event streams        -> stream consumer ->  cleaned / modelled    ->   ML features & training sets
#  third-party APIs     -> API pulls       ->  marts (warehouse)     ->   reverse ETL, APIs
#  files, logs          -> file drops      ->  (lake / lakehouse)
# ```
#
# Everything in this module (and the next two) is about making the middle of that picture reliable.

# %% [markdown]
# ## 2. ETL vs ELT
#
# Both move data from a source to an analytical store. The difference is **where the transformation happens**.
#
# | | ETL: Extract, Transform, Load | ELT: Extract, Load, Transform |
# |---|---|---|
# | Order | clean *before* loading | load raw first, clean *inside* the destination |
# | Transform runs in | a separate compute step (Python, Spark job) | the warehouse itself (SQL) |
# | Strength | only clean data lands; good when the destination is small or costly, or data must be masked first | raw history is kept, you can re-transform when logic changes; warehouses scale SQL well |
# | Weakness | changing the logic means re-extracting from the source | raw sensitive data sits in the warehouse; you need good SQL discipline |
#
# Modern cloud warehouses made compute cheap, which is why ELT became the default. Let's do both on the
# same task: *count delivered orders per channel*. The "warehouse" is an in-memory SQLite database.

# %%
def etl_delivered_by_channel(csv_path):
    """ETL: transform in Python, load only the small finished result."""
    df = pd.read_csv(csv_path)                                  # Extract
    out = (df[df["status"] == "delivered"]                      # Transform (in Python)
           .groupby("channel").size().rename("n_delivered").reset_index())
    con = sqlite3.connect(":memory:")
    out.to_sql("delivered_by_channel", con, index=False)        # Load (only the result)
    return con


def elt_delivered_by_channel(csv_path):
    """ELT: load the raw table untouched, then transform with SQL inside the destination."""
    df = pd.read_csv(csv_path)                                  # Extract
    con = sqlite3.connect(":memory:")
    df.to_sql("raw_orders", con, index=False)                   # Load (raw, as-is)
    con.execute("""
        CREATE TABLE delivered_by_channel AS                    -- Transform (SQL, in the warehouse)
        SELECT channel, COUNT(*) AS n_delivered
        FROM raw_orders WHERE status = 'delivered' GROUP BY channel
    """)
    return con


etl_con = etl_delivered_by_channel(DATA / "orders.csv")
elt_con = elt_delivered_by_channel(DATA / "orders.csv")
a = pd.read_sql("SELECT * FROM delivered_by_channel ORDER BY channel", etl_con)
b = pd.read_sql("SELECT * FROM delivered_by_channel ORDER BY channel", elt_con)
print(a.equals(b))
print("tables in ETL destination:", [r[0] for r in etl_con.execute("SELECT name FROM sqlite_master")])
print("tables in ELT destination:", [r[0] for r in elt_con.execute("SELECT name FROM sqlite_master")])

# %% [markdown]
# *What just happened?* Same answer, different footprint. The ELT destination still holds `raw_orders`, so
# tomorrow, when someone asks "what about returned orders?", you just write another SQL query. In the ETL
# destination the raw rows are gone, and you would have to go back to the source.

# %% [markdown]
# ## 3. Batch vs streaming
#
# * **Batch**: collect data for a period, process it all at once ("every night at 02:00 load yesterday's orders").
#   Simple, cheap, easy to re-run. Latency = the period.
# * **Streaming**: process each event (or tiny group of events) as it arrives. Latency of seconds, but you have
#   to handle out-of-order events, state, and failures continuously.
#
# Rule of thumb: **start with batch**; go streaming only when a business decision genuinely cannot wait
# (fraud checks, live inventory). Many "real-time" requests are satisfied by a batch every 5-15 minutes.
#
# A streaming job is conceptually a loop that keeps state. Here is the idea with a Python generator that
# "receives" orders one at a time and maintains a running count per channel:

# %%
def event_stream(df):
    for row in df.sort_values("order_id").itertuples(index=False):
        yield {"order_id": row.order_id, "channel": row.channel, "status": row.status}


state = {}                       # the streaming job's memory
for i, event in enumerate(event_stream(orders)):
    state[event["channel"]] = state.get(event["channel"], 0) + 1
    if i + 1 in (1000, 3000):    # peek at the state part-way, as a dashboard would
        print(f"after {i + 1} events:", dict(sorted(state.items())))

batch = orders.groupby("channel").size().to_dict()
print("final streaming state == batch result:", dict(sorted(state.items())) == dict(sorted(batch.items())))

# %% [markdown]
# *What just happened?* Streaming and batch computed the same totals; streaming just had a correct answer
# **at every moment** instead of once at the end. The price is that `state` must survive crashes (real systems
# checkpoint it) and events can arrive late or twice. Notebook 04 of the Pipelines module covers those problems.

# %% [markdown]
# ## 4. OLTP vs OLAP
#
# | | OLTP (transactional) | OLAP (analytical) |
# |---|---|---|
# | Typical query | "fetch order 4711", "insert this payment" | "revenue by month by country for 2 years" |
# | Rows touched | a handful | millions |
# | Columns touched | most of the row | a few columns |
# | Optimised for | many small reads/writes, strict consistency | few huge scans and aggregations |
# | Examples | PostgreSQL/MySQL behind an app, SQLite | Snowflake, BigQuery, Redshift, DuckDB, ClickHouse |
#
# Never run heavy analytics on the production OLTP database: a 10-minute scan can slow down real customers.
# That is the reason data gets *copied* into a separate analytical system at all.
#
# We can see the OLTP flavour with an index. A point lookup is fast with an index and a full scan without:

# %%
con = sqlite3.connect(":memory:")
big = pd.concat([items] * 20, ignore_index=True)       # ~190k rows so timing is visible
big["item_id"] = np.arange(len(big))
big.to_sql("order_items", con, index=False)


def time_lookup(n=300):
    t0 = time.perf_counter()
    for k in range(n):
        con.execute("SELECT * FROM order_items WHERE item_id = ?", (k * 7,)).fetchall()
    return (time.perf_counter() - t0) / n * 1e6


no_index_us = time_lookup()
con.execute("CREATE INDEX ix_item ON order_items(item_id)")
index_us = time_lookup()
plan = con.execute("EXPLAIN QUERY PLAN SELECT * FROM order_items WHERE item_id = 5").fetchall()[0][-1]
print(f"lookup without index: {no_index_us:8.1f} microseconds")
print(f"lookup with index:    {index_us:8.1f} microseconds  ({no_index_us / index_us:,.0f}x faster)")
print("plan:", plan)

# %% [markdown]
# *What just happened?* A B-tree index turns "scan everything" into "walk a few pages". OLTP databases lean on
# indexes and store each row's fields **together on disk** because a typical request wants the whole row.
# Analytical queries want the opposite, as we measure next.

# %% [markdown]
# ## 5. Row vs column storage, measured
#
# Imagine a table with 2 million rows and 8 numeric columns. The query is `AVG(col_0)`: one column only.
#
# * **Row store**: values of one row sit next to each other: `r0c0 r0c1 ... r0c7 r1c0 r1c1 ...`
# * **Column store**: values of one column sit next to each other: `c0r0 c0r1 c0r2 ...`
#
# NumPy lets us build both layouts of the same numbers: C order is row-major, Fortran order is column-major.

# %%
rng = np.random.default_rng(42)
n_rows, n_cols = 2_000_000, 8
row_store = rng.random((n_rows, n_cols))                 # C order: rows contiguous
col_store = np.asfortranarray(row_store)                 # same numbers, columns contiguous
print("row store  C-contiguous:", row_store.flags["C_CONTIGUOUS"], "| col store F-contiguous:", col_store.flags["F_CONTIGUOUS"])


def best_of(fn, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1000


t_row = best_of(lambda: row_store[:, 0].mean())
t_col = best_of(lambda: col_store[:, 0].mean())
print(f"mean of one column, row layout:    {t_row:6.2f} ms")
print(f"mean of one column, column layout: {t_col:6.2f} ms   ({t_row / t_col:.1f}x faster)")
print("same answer:", np.isclose(row_store[:, 0].mean(), col_store[:, 0].mean()))

# %% [markdown]
# *What just happened?* With the row layout the CPU has to pull in whole 64-byte cache lines that hold all 8 columns
# just to use one number from each. With the column layout it streams exactly the bytes it needs. The exact ratio depends on your
# machine, but the direction is dependable. Real column stores add two more wins that we measure in the next notebook:
# **compression** (a column of similar values compresses far better than a row of mixed types) and **column pruning**
# (never even read the columns the query does not mention).
#
# The flip side: to fetch one *whole row* from a column store you must stitch together 8 separate places. That is why
# OLTP systems are row-oriented and analytics systems are column-oriented.
#
# ### Bytes the query actually needs
#
# A quick way to reason about it: bytes that must be read.

# %%
bytes_needed_row_store = row_store.nbytes                # a scan must walk everything
bytes_needed_col_store = row_store[:, 0].nbytes          # only the touched column
print(f"row store reads ~{bytes_needed_row_store / 1e6:.0f} MB, column store reads ~{bytes_needed_col_store / 1e6:.0f} MB "
      f"for a 1-of-{n_cols}-column query")
del row_store, col_store

# %% [markdown]
# ## 6. Lake, warehouse, lakehouse
#
# | | Data lake | Data warehouse | Lakehouse |
# |---|---|---|---|
# | Holds | any files (CSV, JSON, Parquet, images, logs) in cheap object storage | modelled, typed tables | open files (usually Parquet) **plus** a table layer |
# | Schema | on **read** (decide when you query) | on **write** (enforced on load) | enforced, with evolution |
# | Good at | cheap raw history, ML on unstructured data | fast reliable SQL and BI | one copy of data for BI and ML |
# | Risk | becomes a "data swamp" without governance | rigid; costly for huge raw data | newer, more moving parts |
#
# The lakehouse idea is: keep the cheap open-file storage of a lake, and add transactions, schema enforcement
# and time travel on top (table formats such as Delta Lake, Apache Iceberg, Apache Hudi do this). In practice most companies run a mix,
# using a **medallion** pattern: *bronze* = raw as landed, *silver* = cleaned and conformed, *gold* = business-ready marts.

# %% [markdown]
# ## 7. Who owns what?
#
# | Stage | Data engineer | Data scientist / analyst |
# |---|---|---|
# | Source connectors, ingestion, scheduling | owns | consumer |
# | Storage, partitioning, cost, access control | owns | consumer |
# | Data quality checks on raw and modelled tables | owns the framework | writes checks for *their* datasets |
# | Modelling (star schemas, marts) | owns core models | co-owns domain marts |
# | Feature engineering for a model | supports (feature store, pipelines) | owns the logic |
# | Training, evaluation | supports infra | owns |
# | Serving and monitoring models | shared with ML/MLOps engineers | owns model quality |
#
# The lines blur in small teams. A data scientist who can write an idempotent pipeline and a data-quality check is far more
# valuable than one who can only hand over a notebook, which is why this module exists.

# %% [markdown]
# ## Common mistakes
#
# 1. **Choosing streaming because it sounds modern.** It multiplies operational complexity. Ask "who acts on this data, and how quickly?" first.
# 2. **Analytics on the OLTP database.** Big scans compete with customer traffic. Replicate or extract instead.
# 3. **Throwing away raw data** in an ETL step and later needing it. Keep a raw/bronze copy when storage is cheap.
# 4. **Believing "the lake is the warehouse".** Files without schema, ownership and tests become a swamp.
# 5. **Benchmarking storage layouts with tiny data.** Cache effects only show at scale; measure with realistic sizes.

# %% [markdown]
# ## Interview corner
#
# **1. ETL or ELT: how do you choose?**
# <details><summary>Answer</summary>
# Default to ELT when the destination is a scalable warehouse: land raw data, transform in SQL, keep history so logic can be re-run.
# Choose ETL when data must be masked or reduced before it may land (privacy rules), when the destination is small or expensive,
# or when the transformation is not expressible in SQL (for example, parsing images or calling a model).
# </details>
#
# **2. Why are analytical databases column-oriented?**
# <details><summary>Answer</summary>
# Analytical queries touch few columns of many rows. Column layout reads only those columns (pruning), compresses well because
# values in a column are similar, and lets the CPU process contiguous values in a tight loop. Row layout is better when you fetch or
# write whole rows, which is the OLTP pattern.
# </details>
#
# **3. What is a lakehouse and what problem does it solve?**
# <details><summary>Answer</summary>
# It puts warehouse-like features (ACID transactions, schema enforcement, time travel) on top of open file formats in object storage,
# so BI and ML can share one copy of the data instead of maintaining a lake and a separate warehouse.
# </details>
#
# **4. A stakeholder asks for "real-time" dashboards. What do you ask?**
# <details><summary>Answer</summary>
# What decision changes if the number is 1 minute old versus 15 minutes old versus 1 day old? Most requirements are met by micro-batch.
# True streaming is justified by an action that loses value in seconds (fraud, alerting), and it needs monitoring and replay strategies.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: ELT it yourself
# Using the `elt_con` connection (it has the table `raw_orders`), write a SQL query that returns, for each **status**,
# the number of orders, as a DataFrame `by_status` with columns `status` and `n`, sorted by `n` descending (ties by `status`).
# *Hint: `pd.read_sql(sql, elt_con)`; `GROUP BY status`.*

# %% tags=["starter"]
by_status = ...

# %%
expected = orders.groupby("status").size().rename("n").reset_index().sort_values(["n", "status"], ascending=[False, True]).reset_index(drop=True)
assert list(by_status.columns) == ["status", "n"], "columns must be status, n"
got = by_status.sort_values(["n", "status"], ascending=[False, True]).reset_index(drop=True)
assert got.equals(expected), "counts per status do not match"
assert by_status["n"].is_monotonic_decreasing, "sort by n descending"
print("Exercise 1 passed")

# %% tags=["solution"]
by_status = pd.read_sql("SELECT status, COUNT(*) AS n FROM raw_orders GROUP BY status ORDER BY n DESC, status", elt_con)

# %% [markdown]
# ### Exercise 2: A streaming window
# Streaming jobs often aggregate in **tumbling windows** (fixed, non-overlapping time buckets). Treat `orders` as an event stream ordered by
# `order_id`, and compute the number of orders in each window of **500 consecutive events**. Store a list of counts per window per
# channel as `windows`: a list of dicts `{channel: count}`, one dict per window (the last window may be smaller).
# Do it with a plain loop and a dict, the way a stream processor would, not with `groupby`.

# %% tags=["starter"]
windows = ...

# %%
ordered = orders.sort_values("order_id").reset_index(drop=True)
expected_windows = [g["channel"].value_counts().to_dict() for _, g in ordered.groupby(ordered.index // 500)]
assert len(windows) == len(expected_windows), f"expected {len(expected_windows)} windows"
assert all(dict(w) == e for w, e in zip(windows, expected_windows)), "counts in at least one window differ"
assert sum(sum(w.values()) for w in windows) == len(orders), "every event must be counted exactly once"
print("Exercise 2 passed:", len(windows), "windows")

# %% tags=["solution"]
windows = []
current = {}
for i, event in enumerate(event_stream(orders), start=1):
    current[event["channel"]] = current.get(event["channel"], 0) + 1
    if i % 500 == 0:
        windows.append(current)
        current = {}
if current:
    windows.append(current)

# %% [markdown]
# ### Exercise 3: Why does storage layout matter here?
# Using `orders` and `items` (in memory), estimate the **fraction of bytes** a column store must read to answer
# "total quantity per product" versus a row store that must read every column of `order_items`.
# Compute `fraction` = (bytes of the 2 needed columns `product_id` and `quantity`) / (bytes of all columns), using
# `items.memory_usage(index=False)`.

# %% tags=["starter"]
fraction = ...

# %%
mu = items.memory_usage(index=False)
assert abs(fraction - (mu[["product_id", "quantity"]].sum() / mu.sum())) < 1e-9, "use only the two needed columns in the numerator"
assert 0 < fraction < 1
print(f"Exercise 3 passed: a column store reads {fraction:.0%} of the bytes")

# %% tags=["solution"]
mu = items.memory_usage(index=False)
fraction = mu[["product_id", "quantity"]].sum() / mu.sum()

# %% [markdown]
# ### Exercise 4 (open-ended): Design on paper
# The shop wants (a) a fraud check on every payment within 2 seconds and (b) a weekly revenue report for finance.
# In a markdown cell, decide batch or streaming for each, ETL or ELT for the report, and name what you would land in bronze,
# silver and gold. Two sentences per choice, including one risk you accept.

# %% [markdown]
# ## Recap
# - **ETL** transforms before loading; **ELT** loads raw first and transforms in the warehouse (keeps history, re-runnable).
# - **Batch** first; streaming only when the decision cannot wait, and then plan for state, late and duplicate events.
# - **OLTP** = many small row operations (index-driven); **OLAP** = few huge column scans. Do not mix them on one system.
# - Column layout reads fewer bytes, compresses better and scans faster; we measured the scan difference.
# - Lake = cheap files, schema on read; warehouse = modelled tables, schema on write; lakehouse = open files plus a table layer.
#
# **Go deeper:** the DuckDB documentation (duckdb.org/docs) on its columnar engine, and the Apache Parquet documentation (parquet.apache.org).
#
# **Next:** [File formats: CSV, JSON, Parquet](02_file_formats_csv_json_parquet.ipynb)
