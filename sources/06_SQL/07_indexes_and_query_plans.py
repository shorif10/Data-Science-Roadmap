# %% [markdown]
# # Indexes and Query Plans
#
# **Why this matters.** A query that takes 50 ms on your laptop sample can take minutes on production tables with hundreds of millions of rows. Data scientists
# are often the ones who write the slow query (a dashboard that times out, a feature pipeline that misses its window). Knowing how to *read a query plan* and when an
# index helps turns "it is slow, ask the DBA" into a fix you can make and measure yourself.
#
# **You will learn**
# - read `EXPLAIN QUERY PLAN` and tell a full-table `SCAN` from an index `SEARCH`
# - create indexes (single, composite, covering) and **measure** the before/after
# - write **sargable** conditions that let the database use an index
# - explain what indexes cost (space and slower writes), so you do not index everything
#
# **Prerequisites:** [DDL, DML and constraints](06_ddl_dml_and_constraints.ipynb).
#
# > Data used: a temporary copy of `datasets/shop.db`, plus a larger synthetic `events` table generated with a fixed seed so differences are big enough to measure.
# > The tiny shop tables would all be "instant" and hide the effect. Everything is deleted at the end.

# %%
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

tmp = tempfile.TemporaryDirectory()
db_path = Path(tmp.name) / "perf.db"
shutil.copy("../datasets/shop.db", db_path)
con = sqlite3.connect(db_path, isolation_level=None)


def q(sql, params=()):
    return pd.read_sql(sql, con, params=params)


def plan(sql, params=()):
    """Return the query-plan lines SQLite would use (without running the query)."""
    return [row[3] for row in con.execute("EXPLAIN QUERY PLAN " + sql, params)]


def bench(sql, params_list, repeat=5):
    """Best-of-`repeat` wall time in milliseconds to run the query once for each parameter tuple."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        for p in params_list:
            con.execute(sql, p).fetchall()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


# a larger table: 300,000 events from 5,000 users
rng = np.random.default_rng(42)
n = 300_000
events = list(zip(
    range(1, n + 1),
    rng.integers(1, 5001, n).tolist(),
    rng.choice(["view", "click", "cart", "buy", "share"], n, p=[0.55, 0.25, 0.12, 0.06, 0.02]).tolist(),
    (pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 3600, n), unit="s")).strftime("%Y-%m-%d %H:%M:%S").tolist(),
    rng.gamma(2.0, 20.0, n).round(2).tolist(),
))
con.execute("CREATE TABLE events (event_id INTEGER PRIMARY KEY, user_id INTEGER, event_type TEXT, ts TEXT, amount REAL)")
con.execute("BEGIN")
con.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", events)
con.execute("COMMIT")
q("SELECT COUNT(*) AS n_events, COUNT(DISTINCT user_id) AS n_users FROM events")

# %% [markdown]
# ## 1. Reading a query plan
#
# The **query planner** decides *how* to answer your SQL: which table to read first, whether to use an index, in what order to join. You cannot see it in the results,
# but you can ask for it with `EXPLAIN QUERY PLAN` (Postgres/MySQL/Snowflake spell it `EXPLAIN`, and Postgres has `EXPLAIN ANALYZE` to also run and time it).

# %%
query = "SELECT COUNT(*) AS n, SUM(amount) AS total FROM events WHERE user_id = ?"
print(plan(query, (42,)))

# %% [markdown]
# `SCAN events` means **read every row** and test the condition on each: cost grows linearly with table size. It is the right answer when you want most of the table and the wrong answer when you want 60 rows out of 300,000.
# Now add an index on the filter column and ask again.

# %%
uids = [(u,) for u in range(1, 41)]                    # 40 different lookups, so we time more than noise
t_scan = bench(query, uids)
con.execute("CREATE INDEX idx_events_user ON events(user_id)")
print(plan(query, (42,)))
t_index = bench(query, uids)
print(f"40 lookups: scan {t_scan:.1f} ms  ->  index {t_index:.1f} ms   (about {t_scan / t_index:.0f}x faster)")
assert t_index < t_scan

# %% [markdown]
# *What just happened?* An index is a separate, sorted structure (a B-tree) from column values to row locations, like the index at the back of a book: jump to "user 42", read the
# matching row pointers, instead of reading the whole book. The plan changed to `SEARCH ... USING INDEX`. The exact speed-up depends on your machine and table size; the measured ratio above is yours.
# On a 10-row table an index would not matter, and on billions of rows it is the difference between milliseconds and minutes.

# %% [markdown]
# ## 2. Composite indexes: column order matters
#
# An index on `(a, b)` is sorted by `a` and, within equal `a`, by `b`, like a phone book sorted by last name then first name. It can serve:
# - `WHERE a = ?`
# - `WHERE a = ? AND b = ?` (or a range on `b`)
#
# but **not** `WHERE b = ?` alone (you cannot look up first names without the last name). Rule of thumb: put **equality** columns first, then the **range** column.

# %%
con.execute("CREATE INDEX idx_events_type_ts ON events(event_type, ts)")
both = "SELECT COUNT(*) FROM events WHERE event_type = 'buy' AND ts >= '2025-06-01' AND ts < '2025-07-01'"
only_second = "SELECT COUNT(*) FROM events WHERE ts >= '2025-06-01' AND ts < '2025-07-01'"
print("both columns   :", plan(both))
print("second col only:", plan(only_second))

# %% [markdown]
# The first query uses both columns of the index (`event_type=? AND ts>? AND ts<?`); the second cannot use it as a search, since it skipped the leading column. SQLite may
# still scan that index or the table; the point is that it does not get a *seek*.

# %% [markdown]
# ## 3. Covering indexes: never touch the table
#
# If an index contains **every column the query needs**, the database answers from the index alone and never reads the table rows. It is marked `COVERING INDEX`.

# %%
cover = "SELECT SUM(amount) FROM events WHERE user_id = ?"
print("before:", plan(cover, (42,)))
con.execute("CREATE INDEX idx_events_user_amount ON events(user_id, amount)")
print("after :", plan(cover, (42,)))

# %% [markdown]
# Covering indexes are fast and bigger; use them for a handful of hot queries, not everywhere.

# %% [markdown]
# ## 4. Sargable conditions: do not hide the column
#
# A condition is **sargable** ("Search ARGument ABLE") if the database can use an index for it. Wrapping the indexed column in a function usually breaks that: the index is sorted by `ts`,
# not by `strftime('%Y-%m', ts)`.

# %%
con.execute("CREATE INDEX idx_events_ts ON events(ts)")
slow = "SELECT COUNT(*) FROM events WHERE strftime('%Y-%m', ts) = '2025-06'"
fast = "SELECT COUNT(*) FROM events WHERE ts >= '2025-06-01' AND ts < '2025-07-01'"
print("function on column:", plan(slow))
print("range on column   :", plan(fast))
a, b = con.execute(slow).fetchone()[0], con.execute(fast).fetchone()[0]
print(f"same answer: {a == b} ({a} rows)   |  {bench(slow, [()], 3):.1f} ms vs {bench(fast, [()], 3):.1f} ms")

# %% [markdown]
# Same rows, different cost. Other sargability killers: `WHERE amount + 0 > 50`, `WHERE LOWER(name) = 'x'` (fix: store normalised, or use an expression index where supported), a **leading wildcard**
# `LIKE '%abc'` (an index can help `LIKE 'abc%'`, never `'%abc'`), and comparing a text column with a number (implicit casts).
# Half-open ranges (`>= start AND < next_start`) beat `BETWEEN` for timestamps because they never double-count the boundary.

# %% [markdown]
# ## 5. Indexes are not free
#
# Every `INSERT`/`UPDATE`/`DELETE` must also update every index on the table, and each index uses disk space. Measure it: insert the same 20,000 rows into a table with no indexes and one with four.

# %%
extra = [(n + 1 + i, int(u), "view", "2026-01-01 00:00:00", 1.0) for i, u in enumerate(rng.integers(1, 5001, 20000))]


def timed_insert(table):
    con.execute("BEGIN")
    t0 = time.perf_counter()
    con.executemany(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?)", extra)
    dt = time.perf_counter() - t0
    con.execute("ROLLBACK")          # leave the table unchanged
    return dt * 1000


con.execute("CREATE TABLE events_plain (event_id INTEGER PRIMARY KEY, user_id INTEGER, event_type TEXT, ts TEXT, amount REAL)")
t_plain = timed_insert("events_plain")
t_idx = timed_insert("events")
print(f"20,000 inserts: no extra indexes {t_plain:.0f} ms | table with 4 indexes {t_idx:.0f} ms | ratio {t_idx / t_plain:.1f}x")
print("index sizes are not shown here, but list them with PRAGMA index_list:", q("PRAGMA index_list(events)").name.tolist())

# %% [markdown]
# So indexes speed reads and slow writes. Analytics warehouses (Snowflake, BigQuery) mostly do not use user-defined indexes at all: they scan compressed columnar data in parallel and
# rely on partitioning/clustering instead. Row stores (Postgres, MySQL, SQL Server, SQLite) are where index design matters most.
#
# One more honest trade-off: an index on a **low-selectivity** column (few distinct values, like `event_type` with five values) is a poor deal when you fetch a large fraction of the table:
# hopping through the index to millions of scattered row locations can cost *more* than one sequential scan. Compare the measured times below.

# %%
low = "SELECT * FROM events WHERE event_type = 'view'"
print("plan with index available:", plan(low))
con.execute("DROP INDEX idx_events_type_ts")
t_noidx = bench(low, [()], 3)
con.execute("CREATE INDEX idx_events_type_ts ON events(event_type, ts)")
t_withidx = bench(low, [()], 3)
print(f"fetch a large share of rows: without the index {t_noidx:.0f} ms, with it {t_withidx:.0f} ms -> the index path took {t_withidx / t_noidx:.1f}x the time of a plain scan")

# %% [markdown]
# The planner also needs **statistics** to choose well; `ANALYZE` gathers them (`VACUUM ANALYZE` in Postgres, automatic in most warehouses).
# Let us give SQLite the facts and ask again about the low-selectivity query:

# %%
con.execute("ANALYZE")
print(q("SELECT tbl, idx, stat FROM sqlite_stat1 WHERE tbl = 'events' ORDER BY idx").head())
print("plan after ANALYZE:", plan(low))

# %% [markdown]
# The planner is a heuristic, not an oracle: even with statistics, SQLite may keep an index path that our measurement showed to be slower for this query. That is why the habit is
# **read the plan, then measure**; never assume "it uses an index" means "it is fast".

# %% [markdown]
# ## Common mistakes
#
# 1. **Indexing every column** "just in case": slower writes, wasted space, and the planner still only uses a few.
# 2. **Function on the indexed column** in `WHERE` (`strftime`, `LOWER`, `DATE(...)`), which defeats the index.
# 3. **Leading wildcard `LIKE '%text'`.**
# 4. **Composite index in the wrong order**: `(b, a)` does not serve a query that filters only on `a`.
# 5. **Judging by a small table.** Test on realistic row counts; plans on 100 rows tell you nothing about 100 million.
# 6. **Timing a single run.** Caches and noise dominate; repeat and take the best (or median).
# 7. **Adding an index for a query that returns most of the table.**

# %% [markdown]
# ## Interview corner
#
# **1. What is an index and how does it speed up a query?**
# <details><summary>Answer</summary>
# A sorted auxiliary structure (usually a B-tree) mapping column values to row locations, so the database can seek to matching rows instead of scanning the whole table. It costs disk space and slows writes.
# </details>
#
# **2. You have <code>WHERE a = 1 AND b &gt; 5</code>. Which composite index, and why that order?**
# <details><summary>Answer</summary>
# <code>(a, b)</code>: equality column first, range column second, so one seek finds <code>a = 1</code> and then a contiguous range of <code>b</code>. With <code>(b, a)</code> the database can only use the range on <code>b</code> and must then filter <code>a</code>.
# </details>
#
# **3. A query is slow. How do you investigate?**
# <details><summary>Answer</summary>
# Read the plan (<code>EXPLAIN</code>/<code>EXPLAIN ANALYZE</code>) for full scans, big joins and sorts; check that filters are sargable and statistics are fresh; consider an index or a rewrite (pre-aggregate, filter earlier, select fewer columns); re-measure on realistic data.
# </details>
#
# **4. When can an index hurt?**
# <details><summary>Answer</summary>
# Write-heavy tables (each write maintains every index), tiny tables, low-selectivity columns where most rows match anyway, and too many overlapping indexes that bloat storage and confuse maintenance.
# </details>

# %% [markdown]
# ## Exercises
# We work on the `orders` table of the temporary copy, which starts with no indexes. Each check inspects the **plan text and index definition**, not timing (timing is noisy).

# %%
def clean(*names):
    for nm in names:
        con.execute(f"DROP INDEX IF EXISTS {nm}")


print("indexes on orders at start:", q("PRAGMA index_list(orders)").name.tolist())

# %% [markdown]
# ### Exercise 1: Speed up a customer lookup
# Customer service pulls all orders for one customer all day. Write the statement that creates an index called `idx_orders_customer` on `orders(customer_id)` and store it in `ex1_sql`.
# *Hint: `CREATE INDEX name ON table(column)`.*

# %% tags=["starter"]
ex1_sql = """
...
"""

# %%
clean("idx_orders_customer")
lookup = "SELECT * FROM orders WHERE customer_id = ?"
before = plan(lookup, (7,))
assert any("SCAN" in line for line in before), f"expected a full scan before the index, got {before}"
con.execute(ex1_sql)
after = plan(lookup, (7,))
assert any("SEARCH" in line and "idx_orders_customer" in line for line in after), f"plan should SEARCH using your index, got {after}"
assert q(lookup, (7,)).shape[0] == pd.read_sql("SELECT * FROM orders WHERE customer_id = 7", con).shape[0]
print("Exercise 1 passed:", after)

# %% tags=["solution"]
ex1_sql = "CREATE INDEX idx_orders_customer ON orders(customer_id)"

# %% [markdown]
# ### Exercise 2: One index for two filters
# The ops dashboard runs `SELECT COUNT(*) FROM orders WHERE channel = ? AND order_date >= ? AND order_date < ?`. Create **one** index named `idx_orders_channel_date` so that the plan uses **both** `channel` and `order_date` in the search.
# Choose the column order deliberately. Store the statement in `ex2_sql`.

# %% tags=["starter"]
ex2_sql = """
...
"""

# %%
clean("idx_orders_channel_date")
con.execute(ex2_sql)
dash = "SELECT COUNT(*) FROM orders WHERE channel = ? AND order_date >= ? AND order_date < ?"
pl = plan(dash, ("web", "2024-11-01", "2024-12-01"))
cols = [r[2] for r in con.execute("PRAGMA index_info(idx_orders_channel_date)")]
assert cols == ["channel", "order_date"], f"equality column first, then the range column; got {cols}"
assert any("idx_orders_channel_date" in line and "channel=?" in line and "order_date>?" in line for line in pl), f"plan should use both columns: {pl}"
print("Exercise 2 passed:", pl)

# %% tags=["solution"]
ex2_sql = "CREATE INDEX idx_orders_channel_date ON orders(channel, order_date)"

# %% [markdown]
# ### Exercise 3: Make it sargable
# The query below counts orders in December 2024 but wraps the column in a function, so an index on `order_date` cannot be used. Rewrite it as a **range condition** returning the **same count** and store it in `ex3_sql`.
# ```sql
# SELECT COUNT(*) FROM orders WHERE strftime('%Y-%m', order_date) = '2024-12'
# ```

# %% tags=["starter"]
ex3_sql = """
SELECT ...
"""

# %%
clean("idx_orders_date")
con.execute("CREATE INDEX idx_orders_date ON orders(order_date)")
old = "SELECT COUNT(*) FROM orders WHERE strftime('%Y-%m', order_date) = '2024-12'"
expected = con.execute(old).fetchone()[0]
assert expected > 0
assert con.execute(ex3_sql).fetchone()[0] == expected, "your query must return the same count as the original"
pl = plan(ex3_sql)
assert any("idx_orders_date" in line and "SEARCH" in line for line in pl), f"the plan should SEARCH idx_orders_date, got {pl}"
assert "strftime" not in ex3_sql.lower(), "do not wrap order_date in a function"
print("Exercise 3 passed:", pl)

# %% tags=["solution"]
ex3_sql = "SELECT COUNT(*) FROM orders WHERE order_date >= '2024-12-01' AND order_date < '2025-01-01'"

# %% [markdown]
# ### Exercise 4: A covering index
# This query is run constantly: `SELECT status, COUNT(*) FROM orders WHERE customer_id = ? GROUP BY status`. Create an index named `idx_orders_customer_status` (replace the plain customer index from exercise 1 for this purpose) such that the plan says **COVERING INDEX**
# for this query, i.e. the table itself is never read. Store the statement in `ex4_sql`.

# %% tags=["starter"]
ex4_sql = """
...
"""

# %%
clean("idx_orders_customer_status", "idx_orders_customer")
con.execute(ex4_sql)
target = "SELECT status, COUNT(*) FROM orders WHERE customer_id = ? GROUP BY status"
pl = plan(target, (7,))
assert any("COVERING INDEX" in line and "idx_orders_customer_status" in line for line in pl), f"plan should use your index as a COVERING INDEX, got {pl}"
print("Exercise 4 passed:", pl)

# %% tags=["solution"]
ex4_sql = "CREATE INDEX idx_orders_customer_status ON orders(customer_id, status)"

# %% [markdown]
# ### Exercise 5 (open-ended): What would you index?
# Take the three queries you find most natural on `orders`, `order_items` and `products` (for example: "order lines of one order", "orders by status in a date range", "products by category").
# For each: write the query, run `plan(...)`, decide on an index, re-run `plan(...)`. Report which index you would **not** create even though it is tempting, and why (think: selectivity, write cost).
# Bonus: measure a 20,000-row insert into `orders_copy` with and without your indexes using the `timed_insert` idea from section 5.

# %% [markdown]
# ## Clean up

# %%
con.close()
tmp.cleanup()
print("temp database removed:", not db_path.exists())

# %% [markdown]
# ## Recap
# - `EXPLAIN QUERY PLAN` shows how the database will answer: `SCAN` reads everything, `SEARCH ... USING INDEX` seeks.
# - An index (sorted B-tree) makes selective lookups fast; measure before and after on realistic row counts.
# - Composite indexes: equality columns first, then the range column; the leading column must appear in the filter.
# - Keep conditions sargable: no functions on the indexed column, no leading-wildcard `LIKE`.
# - Indexes cost storage and slow writes; warehouses like Snowflake and BigQuery rely on partitioning/clustering instead.
#
# **Go deeper:** the SQLite documents "Query Planning" and "The SQLite Query Planner" (sqlite.org); PostgreSQL docs, "Indexes" and `EXPLAIN`; the book *Use The Index, Luke* (use-the-index-luke.com) is a well-known free resource on this topic.
#
# **Next:** [SQL analytics patterns](08_sql_analytics_patterns.ipynb)
