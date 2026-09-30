# %% [markdown]
# # Incremental Loads, Backfills and CDC
#
# **Why this matters.** A full reload of a 2-billion-row table every night stops being an option long before the table gets that big. Real pipelines load *only what changed*, and then have to answer the awkward questions:
# what about rows that were **updated** at the source, rows that were **deleted**, events that arrive **late or twice**, and the day last month that turned out to be wrong and must be **re-loaded**?
# This notebook builds each mechanism in SQLite so you can break it and watch it recover.
#
# **You will learn**
# - load incrementally with a **watermark**, and see why `>` vs `>=` and late commits can silently lose rows
# - apply changes with an idempotent **upsert (merge)** that also refuses to overwrite newer data with older data
# - capture inserts, updates and **deletes** with a trigger-based change log (a tiny **CDC**, change data capture)
# - **backfill** a date range one partition at a time, safely and repeatably
# - explain **at-least-once**, **at-most-once** and **effectively-once** delivery, and build the last one
#
# **Prerequisites:** [Pipeline design and idempotency](01_pipeline_design_and_idempotency.ipynb) and [APIs and ingestion](../18_Data_Engineering/05_apis_and_ingestion.ipynb).
#
# > Data used: the synthetic `orders.csv` (see `docs/DATASETS.md`), loaded into a SQLite file in a temp folder that is removed at exit.

# %%
import atexit
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="incremental_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

orders = pd.read_csv(DATA / "orders.csv")
rng = np.random.default_rng(42)
# A source system stamps every row with the moment it was last modified. We fake that stamp: order day + a random time of day.
secs = rng.integers(0, 86_400, len(orders))
orders["updated_at"] = (pd.to_datetime(orders["order_date"]) + pd.to_timedelta(secs, unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S")
print(orders.shape)
orders.head(3)

# %% [markdown]
# ## 1. The source and the target
# We use two SQLite databases: `source.db` (the operational system we do not control) and `warehouse.db` (our copy). A third small table, `etl_state`, lives in the warehouse and remembers how far we have read.

# %%
def connect(name):
    con = sqlite3.connect(TMP / name)
    con.row_factory = sqlite3.Row
    return con

src, wh = connect("source.db"), connect("warehouse.db")

src.execute("""CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT,
    status TEXT, channel TEXT, updated_at TEXT NOT NULL)""")
src.execute("CREATE INDEX ix_orders_updated ON orders(updated_at)")      # the watermark column must be indexed
cols = ["order_id", "customer_id", "order_date", "status", "channel", "updated_at"]
src.executemany(f"INSERT INTO orders VALUES ({','.join('?' * 6)})", orders[cols].itertuples(index=False, name=None))
src.commit()

wh.execute("""CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT,
    status TEXT, channel TEXT, updated_at TEXT NOT NULL)""")
wh.execute("CREATE TABLE etl_state (pipeline TEXT PRIMARY KEY, watermark TEXT NOT NULL)")
wh.commit()
print("source rows:", src.execute("SELECT COUNT(*) FROM orders").fetchone()[0])

# %% [markdown]
# ## 2. Watermark-based incremental load
# A **watermark** is "the largest `updated_at` I have already loaded". Each run asks the source for rows *newer than the watermark*, loads them, and moves the watermark forward.
# Two rules make it safe:
# 1. **Upsert, don't insert.** A row that was updated at the source arrives again with the same key; the target must replace it, not duplicate it.
# 2. **Move the watermark only after the load committed**, and do both in one transaction. If you advance it first and then crash, the batch is lost forever.

# %%
UPSERT = """
INSERT INTO orders (order_id, customer_id, order_date, status, channel, updated_at)
VALUES (:order_id, :customer_id, :order_date, :status, :channel, :updated_at)
ON CONFLICT(order_id) DO UPDATE SET
    customer_id = excluded.customer_id, order_date = excluded.order_date,
    status = excluded.status, channel = excluded.channel, updated_at = excluded.updated_at
"""

def get_watermark(con, pipeline="orders", default="1970-01-01 00:00:00"):
    row = con.execute("SELECT watermark FROM etl_state WHERE pipeline = ?", (pipeline,)).fetchone()
    return row["watermark"] if row else default

def incremental_load(src, wh, until=None, pipeline="orders"):
    """Load source rows with updated_at > watermark (and <= until). Returns (rows_loaded, new_watermark)."""
    wm = get_watermark(wh, pipeline)
    sql = "SELECT * FROM orders WHERE updated_at > ?" + (" AND updated_at <= ?" if until else "") + " ORDER BY updated_at"
    batch = [dict(r) for r in src.execute(sql, (wm, until) if until else (wm,))]
    if not batch:
        return 0, wm
    new_wm = batch[-1]["updated_at"]
    with wh:                                              # one transaction: data and watermark commit together, or neither does
        wh.executemany(UPSERT, batch)
        wh.execute("INSERT INTO etl_state VALUES (?, ?) ON CONFLICT(pipeline) DO UPDATE SET watermark = excluded.watermark", (pipeline, new_wm))
    return len(batch), new_wm

def count(con, table="orders"):
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

print("first load (only data up to 2024-12-31):", incremental_load(src, wh, until="2024-12-31 23:59:59"))
print("warehouse rows:", count(wh))
print("second run, nothing new:", incremental_load(src, wh, until="2024-12-31 23:59:59"))

# %% [markdown]
# *What just happened?* The first run loaded everything up to the cut-off and stored the watermark; the second run found nothing newer and did nothing. Now let the clock move on: the rest of the source rows "arrive", and a few existing orders change status.

# %%
# Simulate source activity on 2025-01-01: an old order is updated (shipped -> delivered) and its updated_at moves forward.
shipped_old = src.execute("SELECT order_id FROM orders WHERE status = 'shipped' AND order_date < '2024-12-31' LIMIT 3").fetchall()
print("orders updated at the source:", [r["order_id"] for r in shipped_old])
for r in shipped_old:
    src.execute("UPDATE orders SET status = 'delivered', updated_at = '2025-01-01 08:00:00' WHERE order_id = ?", (r["order_id"],))
src.commit()

loaded, wm = incremental_load(src, wh)
print(f"loaded {loaded} rows, new watermark {wm}")
print("warehouse rows:", count(wh), "| source rows:", count(src))
changed = wh.execute("SELECT order_id, status FROM orders WHERE order_id IN (%s)" % ",".join(str(r["order_id"]) for r in shipped_old)).fetchall()
print("updated rows in warehouse:", [(r["order_id"], r["status"]) for r in changed])

# %% [markdown]
# The three updated orders were **replaced**, not duplicated, because of the upsert. Always check equality of a cheap aggregate after a load:

# %%
def fingerprint(con):
    return con.execute("SELECT COUNT(*), COUNT(DISTINCT order_id), MIN(updated_at), MAX(updated_at) FROM orders").fetchone()[:]

assert fingerprint(src) == fingerprint(wh), "warehouse does not match source"
print("source and warehouse agree:", fingerprint(wh))

# %% [markdown]
# ## 3. Where watermarks lose data
# A watermark is only as good as the column behind it. Three classic holes, each reproducible:
#
# **Hole 1: ties at the boundary.** If you load with `updated_at > wm` and two rows share the exact watermark value but only one was visible when you ran, the second is skipped forever.
# **Hole 2: late commits.** A slow transaction at the source stamps `updated_at = 10:00:00` but commits at `10:00:07`. If your run at 10:00:05 already moved the watermark past 10:00:00, that row never appears.
# **Hole 3: the column is not maintained.** A bulk fix script that forgets to touch `updated_at` changes rows invisibly.
#
# Let us reproduce hole 2.

# %%
# Fresh tables for a clean demonstration
src.execute("DELETE FROM orders"); wh.execute("DELETE FROM orders"); wh.execute("DELETE FROM etl_state")
def put(con, oid, ts, status="shipped"):
    con.execute("INSERT INTO orders VALUES (?, 1, '2025-03-01', ?, 'web', ?) ON CONFLICT(order_id) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at",
                (oid, status, ts))
put(src, 1, "2025-03-01 10:00:01")
put(src, 2, "2025-03-01 10:00:09")      # committed first, but stamped later
src.commit()
print("run at 10:00:10 ->", incremental_load(src, wh))
put(src, 3, "2025-03-01 10:00:04")      # a slow transaction: stamped 10:00:04, became visible only after our run
src.commit()
print("next run        ->", incremental_load(src, wh))
print("source has", sorted(r[0] for r in src.execute("SELECT order_id FROM orders")), "| warehouse has", sorted(r[0] for r in wh.execute("SELECT order_id FROM orders")))

# %% [markdown]
# Order 3 is **lost**: its stamp (10:00:04) is older than our watermark (10:00:09). The standard cure is a **lookback window**: re-read everything from `watermark - safety_margin` on every run and rely on the idempotent upsert to absorb the overlap.
# The margin should exceed your longest transaction (and clock skew); it costs a little re-reading and buys correctness.

# %%
def incremental_load_safe(src, wh, lookback="-1 hour", pipeline="orders"):
    wm = get_watermark(wh, pipeline)
    batch = [dict(r) for r in src.execute(
        "SELECT * FROM orders WHERE updated_at > datetime(?, ?) ORDER BY updated_at", (wm, lookback))]
    if not batch:
        return 0, wm
    new_wm = max(wm, batch[-1]["updated_at"])             # never move the watermark backwards
    with wh:
        wh.executemany(UPSERT, batch)
        wh.execute("INSERT INTO etl_state VALUES (?, ?) ON CONFLICT(pipeline) DO UPDATE SET watermark = excluded.watermark", (pipeline, new_wm))
    return len(batch), new_wm

print("safe run ->", incremental_load_safe(src, wh), "(re-read 3 rows: 2 already loaded, 1 newly found)")
print("warehouse has", sorted(r[0] for r in wh.execute("SELECT order_id FROM orders")))
print("run again ->", incremental_load_safe(src, wh), "(same rows re-read, still no duplicates)")
print("rows in warehouse:", count(wh))

# %% [markdown]
# ## 4. Merge semantics: do not let old data overwrite new data
# With overlapping reads, retries and parallel workers, an **older version of a row can arrive after a newer one**. A blind upsert would roll the row back. Guard the update with the version column:

# %%
GUARDED_UPSERT = UPSERT.rstrip() + "\nWHERE excluded.updated_at > orders.updated_at"

wh.execute("DELETE FROM orders")
new_version = dict(order_id=10, customer_id=5, order_date="2025-03-02", status="delivered", channel="web", updated_at="2025-03-02 12:00:00")
old_version = dict(new_version, status="shipped", updated_at="2025-03-02 09:00:00")

for label, sql in [("blind upsert  ", UPSERT), ("guarded upsert", GUARDED_UPSERT)]:
    wh.execute("DELETE FROM orders")
    wh.execute(sql, new_version)
    wh.execute(sql, old_version)                          # the stale copy arrives last
    print(label, "->", dict(wh.execute("SELECT status, updated_at FROM orders WHERE order_id = 10").fetchone()))
wh.commit()

# %% [markdown]
# The blind upsert rolled the order back to `shipped`; the guarded one kept the newer `delivered`. The guard makes the merge **commutative**: applying the same set of versions in any order gives the same final state, which is exactly what you need in a system where order is not guaranteed.
#
# In a warehouse with `MERGE` (Snowflake, BigQuery, SQL Server, Postgres 15+) the same idea reads like this (shown, not run here):

# %% tags=["no-run"]
MERGE_SQL = """
MERGE INTO dw.orders AS t
USING staging.orders_batch AS s
  ON t.order_id = s.order_id
WHEN MATCHED AND s.updated_at > t.updated_at THEN
  UPDATE SET status = s.status, channel = s.channel, updated_at = s.updated_at
WHEN NOT MATCHED THEN
  INSERT (order_id, customer_id, order_date, status, channel, updated_at)
  VALUES (s.order_id, s.customer_id, s.order_date, s.status, s.channel, s.updated_at);
"""

# %% [markdown]
# Two practical notes: load the batch into a **staging table** first and merge from there (one set-based statement beats row-by-row upserts), and make sure the staging batch has **at most one row per key**, otherwise `MERGE` is ambiguous (most engines raise an error).

# %% [markdown]
# ## 5. Deletes and change data capture (CDC)
# A watermark only sees rows that *exist*. A deleted row simply vanishes, and your warehouse keeps it forever. Options:
#
# | Approach | How | Catches deletes? | Cost |
# |---|---|---|---|
# | Soft delete at source | `deleted_at` column, never `DELETE` | yes | needs source cooperation |
# | Full key comparison | compare all keys in source and target | yes | reads every key each run |
# | **Change log (CDC)** | record every insert/update/delete as an event | yes, plus history | needs a log: triggers or database log |
#
# **Log-based CDC** (Debezium, database-native logical replication) reads the database's own transaction log, so it adds no load to the tables and misses nothing. Below, a **trigger-based** version that you can run and inspect: triggers write one row to `change_log` per change.

# %%
src.executescript("""
DROP TABLE IF EXISTS change_log;
CREATE TABLE change_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,      -- total order of changes
    op TEXT NOT NULL,                           -- I, U or D
    order_id INTEGER NOT NULL,
    status TEXT, channel TEXT, customer_id INTEGER, order_date TEXT,
    changed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);
CREATE TRIGGER orders_ai AFTER INSERT ON orders BEGIN
  INSERT INTO change_log (op, order_id, status, channel, customer_id, order_date)
  VALUES ('I', NEW.order_id, NEW.status, NEW.channel, NEW.customer_id, NEW.order_date);
END;
CREATE TRIGGER orders_au AFTER UPDATE ON orders BEGIN
  INSERT INTO change_log (op, order_id, status, channel, customer_id, order_date)
  VALUES ('U', NEW.order_id, NEW.status, NEW.channel, NEW.customer_id, NEW.order_date);
END;
CREATE TRIGGER orders_ad AFTER DELETE ON orders BEGIN
  INSERT INTO change_log (op, order_id, status, channel, customer_id, order_date)
  VALUES ('D', OLD.order_id, OLD.status, OLD.channel, OLD.customer_id, OLD.order_date);
END;
""")

# Reset both sides to the same three rows, then make changes
src.execute("DELETE FROM orders"); src.execute("DELETE FROM change_log"); wh.execute("DELETE FROM orders"); wh.commit()
for oid in (1, 2, 3):
    put(src, oid, f"2025-03-01 10:00:0{oid}")
src.execute("UPDATE orders SET status = 'delivered', updated_at = '2025-03-01 11:00:00' WHERE order_id = 2")
src.execute("DELETE FROM orders WHERE order_id = 3")
put(src, 4, "2025-03-01 12:00:00")
src.commit()
pd.DataFrame([dict(r) for r in src.execute("SELECT seq, op, order_id, status FROM change_log")])

# %% [markdown]
# Every change is an event with a sequence number. A consumer remembers the last `seq` it applied (its **offset**) and replays the rest **in order**:

# %%
def apply_changes(src, wh, pipeline="orders_cdc"):
    offset = int(get_watermark(wh, pipeline, default="0"))
    events = [dict(r) for r in src.execute("SELECT * FROM change_log WHERE seq > ? ORDER BY seq", (offset,))]
    with wh:
        for e in events:
            if e["op"] == "D":
                wh.execute("DELETE FROM orders WHERE order_id = ?", (e["order_id"],))
            else:                                          # I and U are both "make the row look like this"
                wh.execute(UPSERT, {**e, "updated_at": e["changed_at"]})
        if events:
            wh.execute("INSERT INTO etl_state VALUES (?, ?) ON CONFLICT(pipeline) DO UPDATE SET watermark = excluded.watermark", (pipeline, str(events[-1]["seq"])))
    return len(events)

print("events applied:", apply_changes(src, wh))
print("warehouse:", [(r["order_id"], r["status"]) for r in wh.execute("SELECT order_id, status FROM orders ORDER BY order_id")])
print("source   :", [(r["order_id"], r["status"]) for r in src.execute("SELECT order_id, status FROM orders ORDER BY order_id")])
print("replay    :", apply_changes(src, wh), "events (offset remembered)")

# %% [markdown]
# *What just happened?* The delete of order 3 propagated, which the watermark approach could never do. Note the design: I and U share one code path ("upsert the new image"), D removes. Real CDC streams also carry the **before image** and the **commit timestamp**; the pattern is identical.
#
# A long log can be **compacted**: only the last event per key matters for the current state. In SQL that is a window function (the "latest row per key" idiom you will meet everywhere):

# %%
pd.read_sql("""
SELECT order_id, op, status FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY seq DESC) AS rn FROM change_log
) WHERE rn = 1 ORDER BY order_id""", src)

# %% [markdown]
# ## 6. Backfills by date partition
# A backfill re-computes history: a bug was fixed, a source was corrected, a new column was added. The safe pattern is **one partition at a time, delete-then-insert inside a transaction**, so each day is replaced atomically and re-running any day is harmless (this is the *partition overwrite* idea from the idempotency notebook).

# %%
wh.executescript("""
DROP TABLE IF EXISTS daily_orders;
CREATE TABLE daily_orders (order_date TEXT, channel TEXT, n_orders INTEGER, PRIMARY KEY (order_date, channel));
""")
src.execute("DELETE FROM orders")
src.executemany(f"INSERT INTO orders VALUES ({','.join('?' * 6)})", orders[cols].itertuples(index=False, name=None))
src.commit()

def build_partition(day):
    """Recompute one day of daily_orders from the source. Safe to run any number of times."""
    rows = src.execute("SELECT order_date, channel, COUNT(*) AS n FROM orders WHERE order_date = ? GROUP BY order_date, channel", (day,)).fetchall()
    with wh:
        wh.execute("DELETE FROM daily_orders WHERE order_date = ?", (day,))
        wh.executemany("INSERT INTO daily_orders VALUES (?, ?, ?)", [(r["order_date"], r["channel"], r["n"]) for r in rows])
    return len(rows)

def backfill(start, end):
    days = pd.date_range(start, end).strftime("%Y-%m-%d")
    return {d: build_partition(d) for d in days}

first = backfill("2025-05-01", "2025-05-30")
print("partitions built:", len(first), "| rows in table:", count(wh, "daily_orders"))
again = backfill("2025-05-01", "2025-05-30")                 # run the exact same backfill again
print("after re-running the backfill:", count(wh, "daily_orders"), "rows (unchanged)")
total = wh.execute("SELECT SUM(n_orders) FROM daily_orders").fetchone()[0]
expected = src.execute("SELECT COUNT(*) FROM orders WHERE order_date BETWEEN '2025-05-01' AND '2025-05-30'").fetchone()[0]
assert total == expected
print("orders counted:", total, "= source rows in range:", expected)

# %% [markdown]
# Backfill practice that saves weekends: run a **small range first** and compare with the source; limit concurrency so you do not hammer the source system; make each partition independent so a failure in one day does not block the others; and log per-partition row counts so you can spot a day that came out empty.

# %% [markdown]
# ## 7. Deduplication and delivery guarantees
# Message queues and APIs usually promise **at-least-once** delivery: you will get every event, *possibly more than once* (a retry after a lost acknowledgement). The alternatives:
#
# | Guarantee | Meaning | Failure mode |
# |---|---|---|
# | at-most-once | send once, never retry | events can be lost |
# | at-least-once | retry until acknowledged | events can be duplicated |
# | exactly-once (delivery) | each event delivered once | not achievable in general across a network; systems that claim it restrict the scope |
# | **effectively-once** | at-least-once delivery + an idempotent sink | you see each event's effect once |
#
# We simulate a consumer that writes a batch, crashes *before* saving its offset, and restarts.

# %%
wh.executescript("""
DROP TABLE IF EXISTS events_naive; DROP TABLE IF EXISTS events_idem; DROP TABLE IF EXISTS consumer_offsets;
CREATE TABLE events_naive (event_id INTEGER, amount REAL);                    -- no key: duplicates possible
CREATE TABLE events_idem  (event_id INTEGER PRIMARY KEY, amount REAL);        -- keyed: replays are absorbed
CREATE TABLE consumer_offsets (name TEXT PRIMARY KEY, next_offset INTEGER);
""")
stream = [(i, float(10 + i)) for i in range(1, 11)]         # the broker's log: offsets 0..9

class Crash(Exception):
    pass

def consume(sink, offset_name, batch=5, crash_after_write=False):
    row = wh.execute("SELECT next_offset FROM consumer_offsets WHERE name = ?", (offset_name,)).fetchone()
    start = row["next_offset"] if row else 0
    chunk = stream[start:start + batch]
    if not chunk:
        return 0
    if sink == "events_naive":
        wh.executemany("INSERT INTO events_naive VALUES (?, ?)", chunk)
    else:
        wh.executemany("INSERT INTO events_idem VALUES (?, ?) ON CONFLICT(event_id) DO UPDATE SET amount = excluded.amount", chunk)
    wh.commit()
    if crash_after_write:
        raise Crash("process died after writing but before saving the offset")
    wh.execute("INSERT INTO consumer_offsets VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET next_offset = excluded.next_offset", (offset_name, start + len(chunk)))
    wh.commit()
    return len(chunk)

for sink in ("events_naive", "events_idem"):
    try:
        consume(sink, sink, crash_after_write=True)          # first attempt crashes
    except Crash:
        pass
    while consume(sink, sink):                               # the restarted consumer drains the stream
        pass
    n, distinct = wh.execute(f"SELECT COUNT(*), COUNT(DISTINCT event_id) FROM {sink}").fetchone()[:]
    print(f"{sink:13s} rows={n:2d} distinct events={distinct}")

# %% [markdown]
# The crash left the offset un-saved, so the restart **replayed** the first batch. The keyed sink absorbed the replay (10 rows), the naive sink stored 5 events twice (15 rows).
# This is the whole trick behind "exactly-once" pipelines: **at-least-once delivery plus idempotent writes**. When the sink cannot be keyed (for example it is an external API call), store the offset *in the same transaction* as the write, as `incremental_load` did with the watermark.
#
# If you only have a duplicate-ridden table, dedupe by keeping the latest version per key:

# %%
dups = pd.DataFrame({"event_id": [1, 1, 2, 3, 3, 3], "amount": [10, 11, 20, 30, 31, 32], "received_at": pd.date_range("2025-01-01", periods=6, freq="h")})
latest = dups.sort_values("received_at").drop_duplicates("event_id", keep="last")
print(latest[["event_id", "amount"]].to_dict("records"))

# %% [markdown]
# ## Common mistakes
#
# 1. **Advancing the watermark before the load commits** (or in a separate transaction). A crash between the two loses a batch for good.
# 2. **`updated_at > watermark` with no lookback.** Late commits and equal timestamps slip through. Read an overlap and let the idempotent merge absorb it.
# 3. **Using `created_at` as the watermark** for a table whose rows get updated: new rows load, updates never do.
# 4. **Ignoring deletes.** Nothing in a watermark load ever removes a row; the warehouse quietly drifts from the source. Use soft deletes, key comparison or CDC.
# 5. **Blind upserts with parallel or overlapping loads**: an old version overwrites a newer one. Guard with the version column.
# 6. **Non-idempotent backfills** (plain `INSERT` of a recomputed day). Re-running doubles the day. Delete-then-insert the partition in one transaction.
# 7. **Backfilling all history in one giant statement**: one failure restarts everything and the locks/memory hurt. Go partition by partition.
# 8. **Trusting "exactly-once" marketing.** Ask *where* the guarantee ends. Your sink still needs to be idempotent.

# %% [markdown]
# ## Interview corner
#
# **1. How do you load a 500M-row table incrementally?**
# <details><summary>Answer</summary>
# Keep a watermark on a reliable, indexed "last modified" column (or a monotonically increasing id), read rows past it with a safety lookback, stage them, and MERGE into the target guarded by the version column. Commit data and watermark atomically.
# Add a periodic reconciliation (counts/checksums per partition against the source) because incremental loads drift silently, and a plan for deletes.
# </details>
#
# **2. What is CDC and when do you prefer log-based over query-based?**
# <details><summary>Answer</summary>
# Change data capture streams every insert, update and delete as events. Query-based CDC (polling with a watermark) is simple but misses deletes, intermediate states, and rows changed without a timestamp update.
# Log-based CDC reads the database's transaction log, catches all of these with very low load on the source, at the price of more infrastructure and permissions.
# </details>
#
# **3. Is exactly-once processing possible?**
# <details><summary>Answer</summary>
# Not as pure delivery over an unreliable network: a sender cannot distinguish a lost message from a lost acknowledgement. What you can get is exactly-once <em>effect</em>: at-least-once delivery combined with idempotent or transactional writes (keyed upserts, or writing results and offsets in one transaction).
# </details>
#
# **4. A day of data was wrong for two weeks. How do you fix it?**
# <details><summary>Answer</summary>
# Fix the cause first, then backfill the affected partitions with an idempotent partition overwrite, validating one day against the source before running the range, with bounded concurrency. Then re-run downstream models for the same range (or rely on the DAG to cascade) and tell consumers which dates changed.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A correct watermark query
# Write `rows_to_load(con, watermark, lookback_minutes)` returning the list of `order_id`s from the `orders` table in `con` whose `updated_at` is **strictly greater than `watermark - lookback_minutes`**, sorted ascending.
# *Hint: `datetime(?, ?)` in SQLite accepts a modifier like `'-30 minutes'`.*

# %%
ex = sqlite3.connect(":memory:")
ex.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, updated_at TEXT)")
ex.executemany("INSERT INTO orders VALUES (?, ?)", [(1, "2025-01-01 09:00:00"), (2, "2025-01-01 09:40:00"), (3, "2025-01-01 10:00:00"), (4, "2025-01-01 10:20:00")])

# %% tags=["starter"]
def rows_to_load(con, watermark, lookback_minutes):
    ...

# %%
assert rows_to_load(ex, "2025-01-01 10:00:00", 0) == [4], "with no lookback only rows strictly after the watermark qualify"
assert rows_to_load(ex, "2025-01-01 10:00:00", 30) == [2, 3, 4], "30 minutes of lookback re-reads 09:40 and 10:00 (and what is new)"
assert rows_to_load(ex, "2025-01-01 10:00:00", 20) == [3, 4], "09:40 is exactly on the boundary (10:00 minus 20 minutes) and the comparison is strict (>)"
print("Exercise 1 passed")

# %% tags=["solution"]
def rows_to_load(con, watermark, lookback_minutes):
    cur = con.execute(
        "SELECT order_id FROM orders WHERE updated_at > datetime(?, ?) ORDER BY order_id",
        (watermark, f"-{lookback_minutes} minutes"))
    return [r[0] for r in cur]

# %% [markdown]
# ### Exercise 2: Merge that ignores stale versions
# Write `merge_batch(con, batch)` that upserts a list of `(key, value, version)` tuples into table `kv(key PRIMARY KEY, value, version)`, updating an existing row **only if the incoming version is higher**. The same key may appear several times in one batch, in any order.

# %%
def fresh_kv():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT, version INTEGER)")
    return c

# %% tags=["starter"]
def merge_batch(con, batch):
    ...

# %%
import itertools
batch = [("a", "a1", 1), ("a", "a3", 3), ("a", "a2", 2), ("b", "b1", 1)]
for perm in itertools.permutations(batch):
    c = fresh_kv()
    merge_batch(c, list(perm))
    assert dict((k, v) for k, v, _ in c.execute("SELECT * FROM kv")) == {"a": "a3", "b": "b1"}, f"order-dependent result for {perm}"
c = fresh_kv()
merge_batch(c, [("a", "a3", 3)])
merge_batch(c, [("a", "a1", 1)])
assert c.execute("SELECT value FROM kv WHERE key = 'a'").fetchone()[0] == "a3", "a stale batch must not overwrite a newer row"
print("Exercise 2 passed: the merge gives the same answer in every arrival order")

# %% tags=["solution"]
def merge_batch(con, batch):
    with con:
        con.executemany(
            """INSERT INTO kv (key, value, version) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, version = excluded.version
               WHERE excluded.version > kv.version""", batch)

# %% [markdown]
# ### Exercise 3: An idempotent weekly backfill
# Extend the daily-partition pattern. Write `rebuild_month(month)` (for example `"2025-04"`) that recomputes `daily_orders` for **every day of that month that has at least one order**, using the functions above, and returns the total `n_orders` now stored for the month.
# Then call it twice and store the two results in `first_total` and `second_total`; they must be identical and equal to the count of source orders in that month.
# *Hint: `build_partition(day)` is already idempotent. You just need the list of days; `SELECT DISTINCT order_date ... WHERE order_date LIKE '2025-04%'`.*

# %% tags=["starter"]
def rebuild_month(month):
    ...

first_total = ...
second_total = ...

# %%
expected = src.execute("SELECT COUNT(*) FROM orders WHERE order_date LIKE '2025-04%'").fetchone()[0]
assert expected > 0
assert first_total == expected, f"expected {expected} orders for 2025-04, got {first_total}"
assert second_total == first_total, "re-running a backfill must not change the result"
stored = wh.execute("SELECT COUNT(*) FROM (SELECT order_date, channel FROM daily_orders WHERE order_date LIKE '2025-04%' GROUP BY 1, 2)").fetchone()[0]
assert stored == wh.execute("SELECT COUNT(*) FROM daily_orders WHERE order_date LIKE '2025-04%'").fetchone()[0], "duplicate (day, channel) rows found"
print("Exercise 3 passed:", first_total, second_total)

# %% tags=["solution"]
def rebuild_month(month):
    days = [r[0] for r in src.execute("SELECT DISTINCT order_date FROM orders WHERE order_date LIKE ? ORDER BY 1", (month + "%",))]
    for d in days:
        build_partition(d)
    return wh.execute("SELECT SUM(n_orders) FROM daily_orders WHERE order_date LIKE ?", (month + "%",)).fetchone()[0]

first_total = rebuild_month("2025-04")
second_total = rebuild_month("2025-04")

# %% [markdown]
# ### Exercise 4: Replay a change log into current state
# Given a list of CDC events (`seq`, `op`, `id`, `value`), write `current_state(events)` returning a dict `{id: value}` of rows that exist after applying all events in `seq` order. The list is **not sorted**, `U` may arrive for an id that was never inserted (treat it as an insert), and `D` removes the id (a later `I` can bring it back).

# %% tags=["starter"]
def current_state(events):
    ...

# %%
ev = [(4, "D", 2, None), (1, "I", 1, "a"), (3, "U", 1, "a2"), (2, "I", 2, "b"), (6, "U", 9, "z"), (5, "I", 2, "b2")]
assert current_state(ev) == {1: "a2", 2: "b2", 9: "z"}, "apply in seq order: 2 is deleted at seq 4 and re-inserted at seq 5"
assert current_state([]) == {}
assert current_state([(1, "I", 1, "x"), (2, "D", 1, None)]) == {}
print("Exercise 4 passed")

# %% tags=["solution"]
def current_state(events):
    state = {}
    for seq, op, key, value in sorted(events):
        if op == "D":
            state.pop(key, None)
        else:
            state[key] = value
    return state

# %% [markdown]
# ### Exercise 5 (open-ended): Choose your strategy
# Pick one table from a system you know (orders, users, events). In a markdown cell decide: which column is the watermark, how large must the lookback be and why, how do you learn about deletes, and how would you verify (one query) that the target still matches the source after a month of incremental loads?

# %% [markdown]
# ## Recap
# - An incremental load reads past a **watermark** and moves it only in the same transaction that commits the data.
# - Watermarks lose rows through ties and late commits: read a **lookback window** and let an **idempotent merge** absorb the overlap.
# - Guard merges with a version column so stale data can never overwrite newer data, in any arrival order.
# - Watermarks cannot see deletes; soft deletes, key comparison or **CDC** (change log with ordered events and a consumer offset) can.
# - **Backfill** one partition at a time with delete-then-insert in a transaction: repeatable and restartable.
# - Delivery is at-least-once in practice; **effectively-once = at-least-once + idempotent sink**.
#
# **Go deeper:** the SQLite documentation pages on "UPSERT" and "Triggers"; the Debezium documentation (debezium.io) for log-based CDC; Martin Kleppmann, *Designing Data-Intensive Applications*, chapters on stream processing and the end-to-end argument.
#
# **Next:** [Testing, monitoring and alerting pipelines](05_testing_monitoring_and_alerting_pipelines.ipynb)
