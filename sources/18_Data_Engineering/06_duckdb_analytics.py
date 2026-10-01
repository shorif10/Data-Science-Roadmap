# %% [markdown]
# # DuckDB for Analytics
#
# **Why this matters.** Between "a CSV opened in pandas" and "a cloud warehouse with a monthly bill" there is a huge middle ground: datasets from a few hundred MB to tens of GB that
# one laptop or one VM can handle **if** the tool is built for analytics. DuckDB is an in-process, columnar SQL engine (think "SQLite for analytics"): no server, one `pip install`, and it queries CSV and Parquet
# files where they sit. Data scientists use it to explore big files, to prototype warehouse SQL locally, and to test pipeline logic without a cluster.
#
# **You will learn**
# - query CSV, Parquet and in-memory DataFrames with SQL, without loading anything first
# - use window functions (running totals, ranks, month-over-month change) and `QUALIFY`
# - join the shop tables and answer business questions in a few lines
# - measure when DuckDB beats pandas, and when it does not
#
# **Prerequisites:** [File formats](02_file_formats_csv_json_parquet.ipynb), [Star schema](03_data_modeling_star_schema_and_normalization.ipynb) and SQL basics ([SQL module](../06_SQL/)).
#
# > Data used: `orders.csv`, `order_items.csv`, `products.csv`, `customers.csv` (synthetic, see `docs/DATASETS.md`), plus a generated 5-million-row table for the benchmarks.

# %%
import atexit
import shutil
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="duck_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
con = duckdb.connect()                                     # in-memory; pass a filename to persist
print("duckdb", duckdb.__version__)

# %% [markdown]
# ## 1. Query a CSV in place
# No `CREATE TABLE`, no import step: the file name **is** the table. DuckDB samples the file to detect the delimiter, header and column types.

# %%
con.sql("SELECT * FROM '../datasets/orders.csv' LIMIT 3").df()

# %%
con.sql("DESCRIBE SELECT * FROM '../datasets/orders.csv'").df()[["column_name", "column_type"]]

# %% [markdown]
# Unlike `pd.read_csv`, DuckDB inferred `order_date` as a real `DATE` straight away. And because it streams the file, aggregating a CSV bigger than RAM works too.

# %%
con.sql("""
SELECT channel, status, count(*) AS n
FROM '../datasets/orders.csv'
GROUP BY ALL          -- DuckDB shortcut: group by every non-aggregated column
ORDER BY channel, n DESC
""").df().head(6)

# %% [markdown]
# ## 2. Parquet, and DataFrames as tables
# `COPY ... TO` writes Parquet; a Parquet file is then queried the same way (and DuckDB pushes column selection and filters down into it, as in notebook 02).
# Any pandas DataFrame **in your Python scope** can be queried by its variable name:

# %%
con.execute(f"""
COPY (SELECT oi.item_id, o.order_id, o.order_date, o.status, o.channel, o.customer_id, oi.product_id, oi.quantity
      FROM '../datasets/order_items.csv' oi JOIN '../datasets/orders.csv' o USING (order_id))
TO '{(TMP / "lines.parquet").as_posix()}' (FORMAT parquet)
""")
lines_path = (TMP / "lines.parquet").as_posix()
print(con.sql(f"SELECT count(*), min(order_date), max(order_date) FROM '{lines_path}'").fetchall())

products = pd.read_csv(DATA / "products.csv")
con.sql("SELECT category, round(avg(unit_price), 2) AS avg_price, count(*) AS n FROM products GROUP BY 1 ORDER BY 2 DESC").df()

# %% [markdown]
# *What just happened?* `FROM products` found the pandas variable `products` (a "replacement scan") and read it without copying. Going the other way, `.df()` turns a result into a DataFrame.
# The pattern "SQL for the heavy lifting, pandas for the last mile (plots, models)" is very effective.

# %% [markdown]
# ## 3. Joins over the shop data
# Business question: **monthly delivered revenue and gross margin by product category.** Three tables, one query.

# %%
monthly = con.sql(f"""
SELECT date_trunc('month', l.order_date)::DATE AS month, p.category,
       round(sum(l.quantity * p.unit_price), 2) AS revenue,
       round(sum(l.quantity * (p.unit_price - p.unit_cost)) / sum(l.quantity * p.unit_price), 3) AS margin
FROM '{lines_path}' l JOIN products p USING (product_id)
WHERE l.status = 'delivered'
GROUP BY ALL ORDER BY month, category
""").df()
print(monthly.shape)
monthly.head(4)

# %%
# independent check in pandas (never trust one engine blindly when you are learning a new one)
lines_df = con.sql(f"SELECT * FROM '{lines_path}'").df().merge(products, on="product_id")
pd_check = (lines_df.query("status == 'delivered'").assign(rev=lambda d: d.quantity * d.unit_price)
            .groupby([lines_df["order_date"].dt.to_period("M").dt.to_timestamp(), "category"])["rev"].sum().round(2))
print("same revenue as pandas:", np.allclose(monthly["revenue"].to_numpy(), pd_check.to_numpy()))

# %% [markdown]
# ## 4. Window functions
# A **window function** computes a value for each row using a set of related rows, *without collapsing them* like `GROUP BY` does. The three ingredients:
# `PARTITION BY` (which group), `ORDER BY` (in what order), and an optional frame (which neighbouring rows). Start from total monthly revenue:

# %%
con.execute("CREATE OR REPLACE TABLE monthly_rev AS SELECT month, sum(revenue) AS revenue FROM monthly GROUP BY 1")
con.sql("""
SELECT month, revenue,
       round(sum(revenue) OVER (ORDER BY month), 2)                                            AS running_total,
       round(avg(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)   AS moving_avg_3m,
       round(100 * (revenue / lag(revenue) OVER (ORDER BY month) - 1), 1)                      AS mom_growth_pct,
       rank() OVER (ORDER BY revenue DESC)                                                     AS revenue_rank
FROM monthly_rev ORDER BY month
""").df().head(8)

# %% [markdown]
# *What just happened?* Every month kept its own row and gained four columns: a running total (default frame = start to current row), a 3-month moving average (explicit frame), the change versus the previous row via `lag`,
# and a rank. The first month has no previous row, so `mom_growth_pct` is NULL, not 0: window functions expose edge effects honestly.
#
# **Top-N per group** is the classic interview use. `ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC)` numbers products inside each category; DuckDB's `QUALIFY` filters on a window result directly
# (no subquery needed):

# %%
con.sql(f"""
SELECT p.category, p.product_name, round(sum(l.quantity * p.unit_price), 2) AS revenue,
       row_number() OVER (PARTITION BY p.category ORDER BY sum(l.quantity * p.unit_price) DESC) AS rank_in_category
FROM '{lines_path}' l JOIN products p USING (product_id)
WHERE l.status = 'delivered'
GROUP BY p.category, p.product_name
QUALIFY rank_in_category <= 2
ORDER BY category, rank_in_category
""").df()

# %% [markdown]
# Note the order of evaluation: `GROUP BY` runs first, **then** the window function sees the aggregated rows, and `QUALIFY` filters after that (like `HAVING` does for aggregates).
# `QUALIFY` is not in every database (Snowflake, BigQuery and DuckDB have it; PostgreSQL does not); elsewhere wrap the query in a subquery.

# %% [markdown]
# **Customer-level example: days between consecutive orders**, using `lag` partitioned by customer.

# %%
gaps = con.sql(f"""
SELECT customer_id, order_date, order_date - lag(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS days_since_prev
FROM (SELECT DISTINCT customer_id, order_id, order_date FROM '{lines_path}')
QUALIFY days_since_prev IS NOT NULL
""").df()
print(f"{len(gaps)} repeat orders; median gap {gaps['days_since_prev'].median():.0f} days")

# %% [markdown]
# ## 5. DuckDB vs pandas, measured
# Claims about speed are worth nothing until measured. We build a **5-million-row** table (order lines with a category, a customer and an amount), run the same three queries in both engines,
# and time them (best of 3) in **three setups**: pandas; DuckDB querying the pandas DataFrame directly; and DuckDB querying its **own table**
# (loaded once). Timings depend on your machine, so look at the *ratios*, and remember that this is one benchmark, not a law.

# %%
rng = np.random.default_rng(42)
n = 5_000_000
cats = np.array(["Books", "Electronics", "Fashion", "Home", "Sports", "Toys", "Garden", "Grocery"])
big = pd.DataFrame({"customer_id": rng.integers(0, 200_000, n),
                    "category": cats[rng.integers(0, len(cats), n)],
                    "amount": rng.gamma(2.0, 30.0, n).round(2)})
dept = pd.DataFrame({"category": cats, "department": ["Media", "Tech", "Style", "Home", "Outdoors", "Kids", "Outdoors", "Food"]})
big.to_parquet(TMP / "big.parquet", index=False)
print(f"{len(big):,} rows, {big.memory_usage(deep=True).sum() / 1e6:.0f} MB in pandas, {(TMP / 'big.parquet').stat().st_size / 1e6:.0f} MB on disk as Parquet")
t0 = time.perf_counter()
con.execute("CREATE TABLE big_t AS SELECT * FROM big")             # DuckDB's own columnar copy, loaded once
print(f"one-off cost of loading the DataFrame into a DuckDB table: {time.perf_counter() - t0:.2f}s")


def best_s(fn, repeat=3):
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return min(out)


def q1_pandas():   # filter + group by a low-cardinality key
    return big[big["amount"] > 50].groupby("category")["amount"].agg(["count", "sum", "mean"])

Q1 = "SELECT category, count(*), sum(amount), avg(amount) FROM {t} WHERE amount > 50 GROUP BY 1"
Q2 = "SELECT customer_id, sum(amount) s FROM {t} GROUP BY 1 ORDER BY s DESC LIMIT 10"
Q3 = "SELECT d.department, sum(b.amount) FROM {t} b JOIN dept d USING (category) GROUP BY 1"

def q2_pandas():   # group by a high-cardinality key + top 10
    return big.groupby("customer_id")["amount"].sum().nlargest(10)

def q3_pandas():   # join with a small dimension, then aggregate
    return big.merge(dept, on="category").groupby("department")["amount"].sum()


rows = []
for name, fp, sql in [("filter + group (8 groups)", q1_pandas, Q1), ("group by 200k keys + top 10", q2_pandas, Q2), ("join small dim + group", q3_pandas, Q3)]:
    tp = best_s(fp)
    t_df = best_s(lambda: con.sql(sql.format(t="big")).df())         # DuckDB reading the pandas DataFrame
    t_tb = best_s(lambda: con.sql(sql.format(t="big_t")).df())       # DuckDB reading its own table
    rows.append({"query": name, "pandas_s": round(tp, 3), "duckdb_on_dataframe_s": round(t_df, 3), "duckdb_own_table_s": round(t_tb, 3),
                 "table_vs_pandas": round(tp / t_tb, 1)})
pd.DataFrame(rows)

# %% [markdown]
# Sanity check first: do both engines agree? (A fast wrong answer is worthless.)

# %%
a = q2_pandas().reset_index().sort_values(["amount", "customer_id"], ascending=[False, True]).reset_index(drop=True)
b = con.sql(Q2.format(t="big_t")).df().sort_values(["s", "customer_id"], ascending=[False, True]).reset_index(drop=True)
print("top-10 customers identical:", a["customer_id"].tolist() == b["customer_id"].tolist() and np.allclose(a["amount"], b["s"]))

# %% [markdown]
# *What just happened?* Look at the middle column: **querying the pandas DataFrame directly can be *slower* than pandas itself.** DuckDB has to read and convert pandas data into its own
# representation on every single query (a likely contributor in pandas 3 is the text column, stored as a pandas `str` column; we did not profile which part dominates). Once the data lives in DuckDB's own table (or in Parquet, next), the conversion cost disappears and the columnar engine shows its strength. The lesson is not "DuckDB is slow on DataFrames":
# it is that **where the data lives matters as much as which engine you use**. Convert once, or skip pandas entirely by reading files.
#
# Now the more realistic case for big data: the data starts **on disk**. Compare "read the Parquet file into pandas, then aggregate" with "let DuckDB aggregate the file directly"
# (it reads only the two needed columns, in parallel, and never builds the whole DataFrame).

# %%
pq = (TMP / "big.parquet").as_posix()
t_pandas_disk = best_s(lambda: pd.read_parquet(pq).groupby("category")["amount"].sum())
t_duck_disk = best_s(lambda: con.sql(f"SELECT category, sum(amount) FROM '{pq}' GROUP BY 1").df())
print(f"from Parquet on disk: pandas {t_pandas_disk:.3f}s, DuckDB {t_duck_disk:.3f}s ({t_pandas_disk / t_duck_disk:.1f}x)")

# %% [markdown]
# ### When DuckDB does not win
# DuckDB is built for scans and aggregations over lots of rows. Its advantage shrinks or reverses when:
# 1. **The data is tiny.** Parsing SQL and planning the query is fixed overhead; pandas has almost none.
# 2. **You need row-by-row Python logic** (custom functions, iterative algorithms), which SQL cannot express well.
# 3. **You do many small point lookups or updates** (that is OLTP work; use SQLite/Postgres).
# 4. **Everything already fits in memory and the operation is trivial** (a single `.sum()` on one column).
# 5. **You have many concurrent writers.** DuckDB is single-writer per database file.
#
# Measure case 1: the same aggregation on 20,000 rows, repeated 200 times (typical interactive use). DuckDB gets the fair setup, its own table.

# %%
small = big.head(20_000).copy()

def small_pandas():
    return small[small["amount"] > 50].groupby("category")["amount"].agg(["count", "sum", "mean"])

con.execute("CREATE OR REPLACE TABLE small_t AS SELECT * FROM small")
reps = 200
tp = best_s(lambda: [small_pandas() for _ in range(reps)]) / reps * 1000
td = best_s(lambda: [con.sql(Q1.format(t="small_t")).df() for _ in range(reps)]) / reps * 1000
print(f"20,000 rows: pandas {tp:.2f} ms/query, DuckDB (own table) {td:.2f} ms/query")

# %% [markdown]
# Read that pair of numbers as *the fixed cost per query*, not as a verdict: at this size both are fast enough that it does not matter, and the choice should go to whichever makes your code clearer.
# A good rule: **for anything that fits comfortably in pandas and where you like the pandas API, use pandas; reach for DuckDB when data is large or lives in files, when the logic is naturally SQL (windows, joins), or when you are prototyping warehouse SQL.**
# They also cooperate: DuckDB can query a DataFrame and return a DataFrame, so you can mix them in one notebook.

# %% [markdown]
# ## Common mistakes
#
# 1. **Benchmarking on toy data** (or once, cold). Use realistic sizes and repeat the timing.
# 2. **Forgetting that `.df()` materialises the result**. A query returning 50 million rows into pandas defeats the point; aggregate in SQL first.
# 3. **Relying on the CSV sniffer for production.** For stable pipelines specify `read_csv('f.csv', types={...}, header=true)` so a weird first 20,000 rows cannot change your schema.
# 4. **Window functions before aggregation.** Order of evaluation is `WHERE`, `GROUP BY`, `HAVING`, window functions, `QUALIFY`, `ORDER BY`, `LIMIT`. Filtering on a window result in `WHERE` is an error.
# 5. **Comparing engines without checking they agree** (NULL handling, integer division and sort ties differ subtly).
# 6. **Treating DuckDB as a multi-user server.** It is an embedded, single-process engine (one writer per file).

# %% [markdown]
# ## Interview corner
#
# **1. What is DuckDB and when would you use it over pandas or a warehouse?**
# <details><summary>Answer</summary>
# An embedded columnar analytical SQL engine. Use it for larger-than-comfortable local data, files (CSV/Parquet) you want to query without loading, SQL-shaped logic (joins, windows), and local testing of warehouse SQL.
# Use pandas for small in-memory work with custom Python logic; use a warehouse when you need shared, governed, concurrent access at scale.
# </details>
#
# **2. What is the difference between `RANK`, `DENSE_RANK` and `ROW_NUMBER`?**
# <details><summary>Answer</summary>
# <code>ROW_NUMBER</code> gives unique consecutive numbers (ties broken arbitrarily), <code>RANK</code> gives ties the same number and skips the next numbers (1,1,3), <code>DENSE_RANK</code> gives ties the same number without gaps (1,1,2).
# </details>
#
# **3. Get the top 3 products by revenue in every category.**
# <details><summary>Answer</summary>
# Aggregate revenue per product, then <code>ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC)</code> and keep rows where it is at most 3 (with <code>QUALIFY</code>, or a subquery/CTE in databases without it).
# </details>
#
# **4. Why can a window function not appear in a `WHERE` clause?**
# <details><summary>Answer</summary>
# Logical evaluation order: <code>WHERE</code> filters rows before windows are computed. To filter on a window result use <code>QUALIFY</code> or wrap the query and filter in the outer one.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Query the CSV in place
# Write a query on `'../datasets/orders.csv'` (no pandas load) returning a DataFrame `active_by_channel` with columns `channel` and `customers`: the number of **distinct customers** who placed at least one **delivered** order in that channel,
# sorted by `customers` descending then `channel`.

# %% tags=["starter"]
active_by_channel = ...

# %%
o = pd.read_csv(DATA / "orders.csv")
exp = (o[o["status"] == "delivered"].groupby("channel")["customer_id"].nunique().rename("customers").reset_index()
       .sort_values(["customers", "channel"], ascending=[False, True]).reset_index(drop=True))
assert list(active_by_channel.columns) == ["channel", "customers"], "columns: channel, customers"
assert active_by_channel.reset_index(drop=True).astype({"customers": "int64"}).equals(exp), "results differ from the pandas calculation"
print("Exercise 1 passed")

# %% tags=["solution"]
active_by_channel = con.sql("""
SELECT channel, count(DISTINCT customer_id) AS customers FROM '../datasets/orders.csv'
WHERE status = 'delivered' GROUP BY channel ORDER BY customers DESC, channel
""").df()

# %% [markdown]
# ### Exercise 2: Best month per category
# Using the DataFrame `monthly` (columns `month, category, revenue, margin`) **as a table in SQL**, return `best_month`: one row per category with `category`, `month` and `revenue` for the month with the **highest revenue** in that category.
# Use a window function with `QUALIFY` (or a subquery). Sort by `category`.

# %% tags=["starter"]
best_month = ...

# %%
idx = monthly.groupby("category")["revenue"].idxmax()
exp = monthly.loc[idx, ["category", "month", "revenue"]].sort_values("category").reset_index(drop=True)
got = best_month[["category", "month", "revenue"]].sort_values("category").reset_index(drop=True)
assert list(best_month.columns)[:3] == ["category", "month", "revenue"], "columns: category, month, revenue"
assert got["category"].tolist() == exp["category"].tolist()
assert pd.to_datetime(got["month"]).tolist() == pd.to_datetime(exp["month"]).tolist(), "wrong month for at least one category"
assert np.allclose(got["revenue"], exp["revenue"])
print("Exercise 2 passed")

# %% tags=["solution"]
best_month = con.sql("""
SELECT category, month, revenue FROM monthly
QUALIFY row_number() OVER (PARTITION BY category ORDER BY revenue DESC) = 1
ORDER BY category
""").df()

# %% [markdown]
# ### Exercise 3: Year-over-year growth
# With `monthly_rev` (table: `month`, `revenue`), compute `yoy` = a DataFrame with columns `month` and `yoy_pct`: the percentage change in revenue versus the **same month 12 months earlier** (use `lag(revenue, 12)`), rounded to 1 decimal,
# keeping only months where it exists (not NULL). Order by month.

# %% tags=["starter"]
yoy = ...

# %%
m = con.sql("SELECT * FROM monthly_rev ORDER BY month").df()
m["yoy_pct"] = ((m["revenue"] / m["revenue"].shift(12) - 1) * 100).round(1)
exp = m.dropna(subset=["yoy_pct"])[["month", "yoy_pct"]].reset_index(drop=True)
assert list(yoy.columns) == ["month", "yoy_pct"], "columns: month, yoy_pct"
assert len(yoy) == len(exp) and len(exp) > 0, f"expected {len(exp)} rows (months with a value 12 months earlier)"
assert np.allclose(yoy["yoy_pct"].to_numpy(dtype=float), exp["yoy_pct"].to_numpy(dtype=float), atol=0.11), "values differ"
print(f"Exercise 3 passed: {len(yoy)} months, e.g. {yoy.iloc[0].to_dict()}")

# %% tags=["solution"]
yoy = con.sql("""
SELECT month, round(100 * (revenue / lag(revenue, 12) OVER (ORDER BY month) - 1), 1) AS yoy_pct
FROM monthly_rev QUALIFY yoy_pct IS NOT NULL ORDER BY month
""").df()

# %% [markdown]
# ### Exercise 4: Where is the crossover?
# For row counts `sizes = [1_000, 10_000, 100_000, 1_000_000]`, take the first N rows of `big`, time the pandas version and the DuckDB version of the `Q1` aggregation (DuckDB on its **own table**, created outside the timing; `best_s` with the default `repeat=3`) and store the results in a DataFrame
# `crossover` with columns `rows`, `pandas_ms`, `duckdb_ms`. There is no asserted winner: your machine decides. Then write in a markdown cell at what size, if any, DuckDB pulled ahead on your machine.
# *Hint: `CREATE OR REPLACE TABLE t AS SELECT * FROM df_name` copies a DataFrame into DuckDB.*

# %% tags=["starter"]
sizes = [1_000, 10_000, 100_000, 1_000_000]
crossover = ...

# %%
assert list(crossover.columns) == ["rows", "pandas_ms", "duckdb_ms"], "columns: rows, pandas_ms, duckdb_ms"
assert crossover["rows"].tolist() == sizes
assert (crossover[["pandas_ms", "duckdb_ms"]] > 0).all().all()
print(crossover.round(2).to_string(index=False))
print("Exercise 4 done")

# %% tags=["solution"]
sizes = [1_000, 10_000, 100_000, 1_000_000]
recs = []
for k in sizes:
    t = big.head(k)
    con.register("t_src", t)
    con.execute("CREATE OR REPLACE TABLE t AS SELECT * FROM t_src")
    con.unregister("t_src")
    tp = best_s(lambda: t[t["amount"] > 50].groupby("category")["amount"].agg(["count", "sum", "mean"])) * 1000
    td = best_s(lambda: con.sql(Q1.format(t="t")).df()) * 1000
    recs.append({"rows": k, "pandas_ms": tp, "duckdb_ms": td})
crossover = pd.DataFrame(recs)

# %% [markdown]
# ### Exercise 5 (open-ended): Your own question
# Pick a business question about the shop that needs a join **and** a window function (for example: "for each customer, what share of their lifetime revenue came from their first order?", or "which country's
# customers have the shortest gap between orders?"). Answer it in DuckDB SQL, check one number in pandas, and note in a markdown cell whether DuckDB or pandas made it easier to write.

# %% [markdown]
# ## Recap
# - DuckDB queries CSV, Parquet and DataFrames in place with SQL; it infers types (real `DATE`s) and streams data larger than memory.
# - Window functions add per-row context without collapsing rows: `sum() OVER`, `lag`, `rank`, `row_number`, frames like `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`; `QUALIFY` filters on them.
# - We measured: on multi-million-row data DuckDB is fast when it reads its own table or Parquet, but querying a pandas DataFrame pays a conversion cost on every query; on tiny data the difference is fixed overhead and decides nothing.
# - Verify engines against each other on a sample; aggregate in SQL before calling `.df()`.
#
# **Go deeper:** the DuckDB documentation (duckdb.org/docs), in particular the pages on window functions, `read_csv` options and the Parquet reader; the pandas user guide for the pandas side of the comparison.
#
# **Next:** [Pipeline design and idempotency](../19_Data_Pipelines/01_pipeline_design_and_idempotency.ipynb)
