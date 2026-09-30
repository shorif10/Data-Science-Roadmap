# %% [markdown]
# # Snowflake SQL and Dialect Differences
#
# **Why this matters.** If you already write SQL, 90% of Snowflake feels familiar. The other 10% is what gets asked in
# interviews and what breaks when you paste a query from Postgres or MySQL: `QUALIFY` to filter on window functions,
# the `VARIANT` type with `FLATTEN` for JSON, `MERGE` for upserts, and Time Travel clauses. This notebook teaches those,
# and lets you *run* everything that is portable on DuckDB so you can check your logic without a Snowflake account.
#
# **You will learn**
# - filter window-function results directly with `QUALIFY` (and know the older subquery equivalent)
# - read and flatten semi-structured JSON (`VARIANT`, `:` paths, `FLATTEN`) and reproduce the result locally
# - write an upsert with `MERGE` and avoid its duplicate-source trap
# - list the small dialect differences that most often break copy-pasted queries
#
# **Prerequisites:** [Snowflake architecture](01_snowflake_architecture_and_concepts.ipynb) and joins/window functions from the [SQL module](../06_SQL/).
#
# > **Candour reminder.** Cells tagged `no-run` contain Snowflake-only syntax that was written from the documentation and
# > **not executed** by the author (no Snowflake account was available). Runnable cells use **DuckDB**, which shares a lot of
# > syntax with Snowflake (`QUALIFY`, `MERGE`, `::` casts, `GROUP BY ALL`). *Verify Snowflake-only syntax in the current docs.*

# %%
import json
from pathlib import Path

import duckdb
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

con = duckdb.connect()  # in-memory
for name, df in [("customers", customers), ("products", products), ("orders", orders), ("order_items", items)]:
    con.register(name, df)
con.sql("SELECT (SELECT count(*) FROM orders) AS orders, (SELECT count(*) FROM order_items) AS items").df()

# %% [markdown]
# ## 1. QUALIFY: `HAVING`, but for window functions
#
# Order of evaluation in a query: `FROM/JOIN -> WHERE -> GROUP BY -> HAVING -> window functions -> QUALIFY -> ORDER BY -> LIMIT`.
# You cannot write `WHERE row_number() OVER (...) = 1`, because window functions are computed *after* `WHERE`. The classic workaround is a subquery or CTE.
# Snowflake (and DuckDB, BigQuery, Teradata) adds `QUALIFY`, which filters *after* the window step.
#
# Task: the **most recent order of each customer**.

# %%
subquery_version = con.sql("""
    SELECT customer_id, order_id, order_date FROM (
        SELECT customer_id, order_id, order_date,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC, order_id DESC) AS rn
        FROM orders
    ) WHERE rn = 1
    ORDER BY customer_id
""").df()

qualify_version = con.sql("""
    SELECT customer_id, order_id, order_date
    FROM orders
    QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC, order_id DESC) = 1
    ORDER BY customer_id
""").df()

print(len(qualify_version), "customers with at least one order")
assert subquery_version.equals(qualify_version)
qualify_version.head(3)

# %% [markdown]
# *What just happened?* Same result, one less nesting level. **Tie-breaking matters:** if two orders share a date and you only order by `order_date`, the
# "latest" one is arbitrary and can change between runs. Always add a unique column (here `order_id`) at the end of the window's `ORDER BY`.
#
# `QUALIFY` is also the idiomatic **de-duplication** tool: keep the newest row per key.
#
# Compare with pandas to prove the logic:

# %%
expected = orders.sort_values(["order_date", "order_id"]).groupby("customer_id").tail(1).sort_values("customer_id")
assert (expected["order_id"].to_numpy() == qualify_version["order_id"].to_numpy()).all()
print("matches the pandas groupby().tail(1) result")

# %% [markdown]
# ## 2. Semi-structured data: `VARIANT`, `:` paths and `FLATTEN`
#
# Real pipelines land raw JSON (events, API responses) before anyone models it. In Snowflake you can store JSON in a
# `VARIANT` column and query it with path syntax, without defining a schema first. Arrays are turned into rows with `FLATTEN`.
#
# We will fabricate a raw event table: one JSON document per order, with a nested customer and an array of lines.

# %%
first_orders = orders.head(400)
docs = []
for r in first_orders.itertuples():
    ls = items[items["order_id"] == r.order_id]
    docs.append(json.dumps({
        "order_id": int(r.order_id),
        "customer": {"id": int(r.customer_id)},
        "channel": r.channel,
        "lines": [{"product_id": int(p), "qty": int(q)} for p, q in zip(ls["product_id"], ls["quantity"])],
    }))
raw_events = pd.DataFrame({"payload": docs})
con.register("raw_events", raw_events)
print(raw_events["payload"].iloc[0])

# %% [markdown]
# **Snowflake syntax** (not executed here). `payload:customer.id::int` reads a path and casts it; `LATERAL FLATTEN` explodes the array, exposing each element as `value`:

# %% tags=["no-run"]
SNOWFLAKE_FLATTEN = """
-- table with a VARIANT column, loaded from JSON files (see notebook 04)
CREATE OR REPLACE TABLE raw_events (payload VARIANT);

SELECT
    payload:order_id::int         AS order_id,
    payload:customer.id::int      AS customer_id,
    payload:channel::string       AS channel,
    l.value:product_id::int       AS product_id,
    l.value:qty::int              AS qty
FROM raw_events,
     LATERAL FLATTEN(input => payload:lines) l;
-- Note: a missing key returns NULL, not an error. FLATTEN on an empty array yields no rows;
-- add OUTER => TRUE to keep such parents (check the docs for exact options).
"""

# %% [markdown]
# **Local equivalent on DuckDB** (runs). Different function names, same idea: extract paths, cast, explode the array.

# %%
flat = con.sql("""
    SELECT CAST(payload->>'$.order_id' AS INTEGER)       AS order_id,
           CAST(payload->>'$.customer.id' AS INTEGER)    AS customer_id,
           payload->>'$.channel'                         AS channel,
           l.product_id, l.qty
    FROM (
        SELECT payload,
               unnest(from_json(payload->'$.lines', '[{"product_id":"INTEGER","qty":"INTEGER"}]')) AS l
        FROM raw_events
    )
""").df()

# ground truth straight from the tables (no JSON involved)
truth = items[items["order_id"].isin(first_orders["order_id"])]
print(len(flat), "flattened line rows; source table has", len(truth))
assert len(flat) == len(truth) and flat["qty"].sum() == truth["quantity"].sum()
flat.head(3)

# %% [markdown]
# *What just happened?* One JSON document became several relational rows, and the totals match the original tables. Whenever you flatten, **assert the
# row count and a sum against a known source**. A wrong path or a JSON array that occasionally is an object instead of a list silently drops or multiplies rows.
#
# Trade-off: `VARIANT` is wonderfully flexible for landing data, but querying deep paths on every dashboard refresh is slower and harder to govern than
# a modelled table. The common pattern: land as VARIANT, then flatten into typed tables.

# %% [markdown]
# ## 3. MERGE: upsert in one statement
#
# `MERGE` compares a **target** table with a **source** by a key and does `UPDATE` for matches and `INSERT` for non-matches (optionally `DELETE`).
# Typical use: apply the daily changes file to a dimension table. Snowflake's syntax is the standard one; DuckDB supports it too, so we can run it.

# %%
changes = pd.DataFrame({
    "product_id": [1, 2, 9001],
    "product_name": ["Headphones Basic", "Headphones Pro", "New Gadget"],
    "category": ["Electronics", "Electronics", "Electronics"],
    "unit_price": [499.0, 749.0, 99.0],
    "unit_cost": [340.16, 521.0, 60.0],
})
con.register("changes", changes)
con.execute("CREATE OR REPLACE TABLE dim_products AS SELECT * FROM products")

con.execute("""
    MERGE INTO dim_products AS t
    USING changes AS s
      ON t.product_id = s.product_id
    WHEN MATCHED THEN UPDATE SET unit_price = s.unit_price, unit_cost = s.unit_cost
    WHEN NOT MATCHED THEN INSERT (product_id, product_name, category, unit_price, unit_cost)
                          VALUES (s.product_id, s.product_name, s.category, s.unit_price, s.unit_cost)
""")
print("rows before:", len(products), " after:", con.sql("SELECT count(*) FROM dim_products").fetchone()[0])
con.sql("SELECT product_id, product_name, unit_price FROM dim_products WHERE product_id IN (1, 2, 9001) ORDER BY 1").df()

# %% [markdown]
# *What just happened?* Two rows updated, one inserted, everything else untouched. **The trap:** if the source contains **two rows for the same key**, a target row
# matches multiple source rows and the result is ambiguous. In Snowflake this can either raise an error or pick one nondeterministically depending on a session
# setting (`ERROR_ON_NONDETERMINISTIC_MERGE`; *verify current default and behaviour in the docs*). The fix is to make the source unique first, and `QUALIFY` is the tool:

# %%
dup_source = con.sql("""
    SELECT * FROM (VALUES (1, 'a', 10.0, '2025-01-01'), (1, 'a', 12.0, '2025-01-02'), (2, 'b', 5.0, '2025-01-01'))
      AS s(product_id, name, price, updated_at)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY updated_at DESC) = 1
    ORDER BY product_id
""").df()
dup_source

# %% [markdown]
# ## 4. Time Travel and other Snowflake-only clauses (no-run)
#
# DuckDB has no Time Travel, so these are shown as reference. Behaviour and limits depend on your edition and retention setting: *verify in the current docs*.

# %% tags=["no-run"]
TIME_TRAVEL = """
SELECT * FROM orders AT(TIMESTAMP => '2025-05-01 09:00:00'::timestamp_ltz);   -- state at a moment
SELECT * FROM orders AT(OFFSET => -3600);                                     -- one hour ago (seconds)
SELECT * FROM orders BEFORE(STATEMENT => '<query_id>');                        -- just before a specific statement ran
CREATE TABLE orders_restored CLONE orders AT(OFFSET => -3600);                 -- clone from the past
UNDROP TABLE orders;                                                          -- restore a dropped table
"""

# %% [markdown]
# ## 5. Dialect cheat sheet (and a runnable portability test)
#
# | Topic | Snowflake | Postgres-style / notes |
# |---|---|---|
# | Unquoted identifiers | folded to **UPPER CASE** (`select a` -> column `A`); double quotes preserve case | Postgres folds to lower case |
# | Conditional | `IFF(cond, a, b)` (also `CASE`) | `CASE` only |
# | Null helpers | `NVL`, `ZEROIFNULL`, `COALESCE` | `COALESCE` |
# | Date arithmetic | `DATEADD(day, 3, d)`, `DATEDIFF(day, a, b)` | `d + INTERVAL '3 day'`, `b - a` |
# | Safe cast | `TRY_CAST(x AS INT)`, `TRY_TO_NUMBER` | `CAST` errors on bad input |
# | Cast shorthand | `x::int` | same |
# | String aggregation | `LISTAGG(x, ',')` | `STRING_AGG` |
# | Exclude columns | `SELECT * EXCLUDE (col)` | not in standard Postgres |
# | Group shortcut | `GROUP BY ALL` | newer Postgres lacks it |
# | JSON | `VARIANT`, `col:path`, `PARSE_JSON`, `FLATTEN` | `jsonb`, `->`, `->>`, `jsonb_array_elements` |
#
# Try a few on DuckDB to see which are portable. **A failure here is informative, not a bug**: it shows the function is Snowflake-specific
# (or at least not in DuckDB). This is how you find what to rewrite when moving code between engines.

# %%
probes = {
    "TRY_CAST": "SELECT TRY_CAST('abc' AS INTEGER)",
    "GROUP BY ALL": "SELECT a, count(*) FROM (SELECT 1 AS a) GROUP BY ALL",
    "SELECT * EXCLUDE": "SELECT * EXCLUDE (b) FROM (SELECT 1 AS a, 2 AS b)",
    "LISTAGG": "SELECT LISTAGG(x, ',') FROM (VALUES ('a'), ('b')) t(x)",
    "DATEDIFF": "SELECT DATEDIFF('day', DATE '2024-01-01', DATE '2024-03-01')",
    "IFF": "SELECT IFF(TRUE, 1, 2)",
    "DATEADD": "SELECT DATEADD(day, 3, DATE '2024-01-01')",
    "ZEROIFNULL": "SELECT ZEROIFNULL(NULL)",
}
results = {}
for k, q in probes.items():
    try:
        con.sql(q).fetchall()
        results[k] = "works on DuckDB"
    except duckdb.Error:
        results[k] = "Snowflake-specific (not in DuckDB): rewrite when porting"
pd.Series(results, name="portability")

# %% [markdown]
# ## Common mistakes
#
# 1. **Ordering by a non-unique column inside `ROW_NUMBER()`**: arbitrary winner, results change between runs. Add a tiebreaker.
# 2. **Trusting a path on `VARIANT` without checking for NULLs**: typos in a path return `NULL`, not an error. Count non-null after extracting.
# 3. **`MERGE` with duplicate keys in the source**: de-duplicate first (`QUALIFY`).
# 4. **Quoted mixed-case identifiers** (`"OrderDate"`) force you to quote them forever, because unquoted names are upper-cased.
# 5. **Confusing Time Travel with backup**: it is bounded by the retention setting.

# %%
# Demonstration of mistake 2: a wrong path silently gives NULLs, so always count non-nulls
wrong = con.sql("SELECT count(payload->>'$.customer.identifier') AS non_null, count(*) AS docs FROM raw_events").fetchone()
print("non-null values from a mistyped path:", wrong[0], "of", wrong[1], "documents")

# %% [markdown]
# ## Interview corner
#
# **1. What does `QUALIFY` do and why does it exist?**
# <details><summary>Answer</summary>
# It filters rows using the result of a window function, evaluated after window functions (like <code>HAVING</code> is after aggregation). It removes the need for a wrapping
# subquery/CTE, e.g. <code>QUALIFY ROW_NUMBER() OVER (PARTITION BY k ORDER BY ts DESC) = 1</code> to keep the latest row per key.
# </details>
#
# **2. How do you query JSON in Snowflake?**
# <details><summary>Answer</summary>
# Store it in a <code>VARIANT</code> column (via <code>PARSE_JSON</code> or a JSON file format on load), navigate with <code>col:path.subpath</code>, cast with <code>::type</code>, and turn arrays into rows with
# <code>LATERAL FLATTEN(input => col:array)</code>. Land raw as VARIANT, then model into typed tables for performance and governance.
# </details>
#
# **3. What can go wrong with MERGE?**
# <details><summary>Answer</summary>
# Multiple source rows matching one target row give ambiguous updates. De-duplicate the source with a window function and <code>QUALIFY</code>. Also make sure the match key is unique in the target and
# that NULL keys do not silently fail to match (<code>NULL = NULL</code> is not true).
# </details>
#
# **4. Why does an unquoted column name come back upper-case in Snowflake?**
# <details><summary>Answer</summary>
# Unquoted identifiers are stored and resolved in upper case, so <code>select a</code> and <code>select A</code> refer to the same column; a quoted lower-case name is a different identifier. Tools that
# expect lower-case (like pandas code) often need <code>df.columns = df.columns.str.lower()</code> after fetching.
# </details>

# %% [markdown]
# ## Exercises
# The DuckDB connection `con` has `customers, products, orders, order_items` and `raw_events` registered.

# %% [markdown]
# ### Exercise 1: Two best sellers per category
# For each product `category`, find the **2 products with the highest total revenue** (`quantity * unit_price` summed over all order lines). Store the result in `top2`
# as a DataFrame with columns `category, product_name, revenue, rk`, using `QUALIFY` (via DuckDB). *Hint: aggregate in a CTE, then `QUALIFY RANK() ... <= 2` over the aggregated rows.*

# %% tags=["starter"]
top2 = ...  # e.g. top2 = con.sql("""...""").df()

# %%
_rev = items.merge(products, on="product_id").assign(revenue=lambda d: d["quantity"] * d["unit_price"])
_p = _rev.groupby(["category", "product_name"], as_index=False)["revenue"].sum()
_exp = _p.sort_values(["category", "revenue"], ascending=[True, False]).groupby("category").head(2)
assert set(top2.columns) >= {"category", "product_name", "revenue", "rk"}, "need columns category, product_name, revenue, rk"
assert len(top2) == len(_exp) == 10, f"expected 10 rows (5 categories x 2), got {len(top2)}"
assert set(zip(top2["category"], top2["product_name"])) == set(zip(_exp["category"], _exp["product_name"])), "wrong products"
assert abs(top2["revenue"].sum() - _exp["revenue"].sum()) < 1e-6, "revenue totals differ"
print("Exercise 1 passed")

# %% tags=["solution"]
top2 = con.sql("""
    WITH p AS (
        SELECT pr.category, pr.product_name, SUM(oi.quantity * pr.unit_price) AS revenue
        FROM order_items oi JOIN products pr USING (product_id)
        GROUP BY pr.category, pr.product_name
    )
    SELECT category, product_name, revenue, RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS rk
    FROM p
    QUALIFY rk <= 2
    ORDER BY category, rk
""").df()

# %% [markdown]
# ### Exercise 2: Flatten and aggregate JSON
# Using only the `raw_events` table (JSON), compute the **total quantity per channel** as a DataFrame `qty_by_channel` with columns `channel, total_qty` (sorted by `channel`).
# *Hint: reuse the `unnest(from_json(...))` pattern in a subquery, then `GROUP BY`.* Snowflake's `LATERAL FLATTEN` version would follow the same shape.

# %% tags=["starter"]
qty_by_channel = ...

# %%
_o = orders.head(400).merge(items, on="order_id")
_e = _o.groupby("channel", as_index=False)["quantity"].sum().rename(columns={"quantity": "total_qty"}).sort_values("channel", ignore_index=True)
_got = qty_by_channel.reset_index(drop=True)[["channel", "total_qty"]].astype({"total_qty": "int64"})
assert list(_got.columns) == ["channel", "total_qty"]
assert _got.equals(_e.astype({"total_qty": "int64"})), f"expected\n{_e}\ngot\n{_got}"
print("Exercise 2 passed")

# %% tags=["solution"]
qty_by_channel = con.sql("""
    SELECT channel, SUM(l.qty) AS total_qty FROM (
        SELECT payload->>'$.channel' AS channel,
               unnest(from_json(payload->'$.lines', '[{"product_id":"INTEGER","qty":"INTEGER"}]')) AS l
        FROM raw_events
    ) GROUP BY channel ORDER BY channel
""").df()

# %% [markdown]
# ### Exercise 3: Write the upsert
# A price-change feed `feed` arrives (below). Write a `MERGE` statement as the string `merge_sql` that **updates `unit_price`** of existing products and **inserts** unknown ones into
# a table called `dim2` (created for you). The checker runs your SQL.

# %%
feed = pd.DataFrame({
    "product_id": [3, 4, 9002, 9003],
    "product_name": ["Keyboard Basic", "Keyboard Pro", "Widget A", "Widget B"],
    "category": ["Electronics", "Electronics", "Home", "Home"],
    "unit_price": [400.0, 450.0, 10.0, 20.0],
    "unit_cost": [240.59, 300.0, 5.0, 8.0],
})
con.register("feed", feed)

# %% tags=["starter"]
merge_sql = """
-- MERGE INTO dim2 ...
"""

# %%
con.execute("CREATE OR REPLACE TABLE dim2 AS SELECT * FROM products")
con.execute(merge_sql)
_n = con.sql("SELECT count(*) FROM dim2").fetchone()[0]
assert _n == len(products) + 2, f"expected {len(products) + 2} rows after 2 inserts, got {_n}"
_p3 = con.sql("SELECT unit_price FROM dim2 WHERE product_id = 3").fetchone()[0]
_p1 = con.sql("SELECT unit_price FROM dim2 WHERE product_id = 1").fetchone()[0]
assert _p3 == 400.0, "product 3 should be updated to 400.0"
assert abs(_p1 - float(products.loc[products['product_id'] == 1, 'unit_price'].iloc[0])) < 1e-9, "products not in the feed must not change"
print("Exercise 3 passed")

# %% tags=["solution"]
merge_sql = """
MERGE INTO dim2 AS t
USING feed AS s ON t.product_id = s.product_id
WHEN MATCHED THEN UPDATE SET unit_price = s.unit_price
WHEN NOT MATCHED THEN INSERT (product_id, product_name, category, unit_price, unit_cost)
                      VALUES (s.product_id, s.product_name, s.category, s.unit_price, s.unit_cost)
"""

# %% [markdown]
# ### Exercise 4 (open-ended): Port a query
# Take a query you have written for Postgres/MySQL that uses a subquery with `ROW_NUMBER()`, a date function, and a conditional. Rewrite it in Snowflake dialect
# (`QUALIFY`, `DATEADD`/`DATEDIFF`, `IFF`) in a markdown cell, then list one thing you would *verify in the Snowflake docs* before trusting it.

# %% [markdown]
# ## Recap
# - `QUALIFY` filters on window functions; always add a unique tiebreaker to the window's `ORDER BY`.
# - `VARIANT` + `:` paths + `LATERAL FLATTEN` turn JSON into rows; validate row counts and sums against a source after flattening.
# - `MERGE` = upsert; de-duplicate the source first.
# - Time Travel (`AT`/`BEFORE`, `UNDROP`, clone at a point in time) is bounded by retention.
# - Small dialect differences (`IFF`, `DATEADD`, upper-case identifiers, `TRY_CAST`) cause most copy-paste breakage.
#
# **Go deeper:** the "SQL command reference" and "Semi-structured data" sections of docs.snowflake.com; the DuckDB documentation (duckdb.org/docs) for the portable dialect.
#
# **Next:** [Python connector and Snowpark](03_python_connector_and_snowpark.ipynb)
