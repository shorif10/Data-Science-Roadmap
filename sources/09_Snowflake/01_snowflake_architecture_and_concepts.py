# %% [markdown]
# # Snowflake: Architecture and Core Concepts
#
# **Why this matters.** On a data team you will be told "just put it in Snowflake" long before anyone explains
# *why* it behaves differently from the Postgres or MySQL you know. The architecture explains the things that
# surprise newcomers: why a huge query can be made faster by clicking a size button, why an idle cluster still costs
# money until it suspends, why "copying" a 2 TB table takes a second, and why `SELECT *` on a wide table is an
# expensive habit. If you know the model you can predict cost and speed instead of guessing.
#
# **You will learn**
# - name the three layers (storage, compute, cloud services) and the problem each one solves
# - explain **micro-partitions** and **pruning**, and *measure* pruning with a toy model you run yourself
# - explain **Time Travel** and **zero-copy clone** by building a small copy-on-write table that behaves the same way
# - say honestly when Snowflake is the wrong tool
#
# **Prerequisites:** basic SQL ([SQL module](../06_SQL/)) and [pandas basics](../05_Pandas/01_loading_and_inspecting_data.ipynb).
#
# > **Candour box (read this).** Snowflake is a paid cloud service. **The author of this notebook did not run any
# > Snowflake code** while writing it. So this module is split in two kinds of cell:
# > 1. **Runnable local cells** (pandas / DuckDB / SQLite / pyarrow). These were executed and they compute the numbers
# >    you should expect to see in Snowflake, or a faithful toy of the concept.
# > 2. **Cells tagged `no-run`**: Snowflake-only syntax or code. They are written from the documented behaviour, but
# >    were **not executed**. UIs, defaults, editions and prices change: anything marked *verify in the current docs*
# >    means exactly that. Start at [docs.snowflake.com](https://docs.snowflake.com).
# >
# > Data used: the synthetic shop tables in `datasets/` (`docs/DATASETS.md`).

# %%
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

# one wide "fact" table: one row per order line
lines = (
    items.merge(orders, on="order_id")
    .merge(products, on="product_id")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"])
    [["item_id", "order_id", "customer_id", "order_date", "status", "channel", "product_id", "category", "quantity", "revenue"]]
    .sort_values(["order_date", "item_id"], ignore_index=True)
)
print(lines.shape)
lines.head(3)

# %% [markdown]
# ## 1. The problem Snowflake was built to solve
#
# A classic database server ties **storage and compute together**: the disks and the CPUs live in the same machine
# (or cluster). That causes three recurring pains:
#
# | Pain | Why it happens when storage and compute are coupled |
# |---|---|
# | The month-end report slows down everyone | Reports and dashboards fight for the same CPUs |
# | You cannot scale compute without scaling disks (or the reverse) | They are bought as one box |
# | Copying data for a dev or test environment takes hours and doubles storage | A copy is a physical copy |
#
# Snowflake's answer is to **separate** the pieces. Think of a library: the books (storage) sit in a huge shared
# archive, and each team sends its own reading-room staff (compute) to work on them. More staff does not need more books.

# %% [markdown]
# ## 2. The three layers
#
# | Layer | What it is | What it solves | Your mental model |
# |---|---|---|---|
# | **Storage** | Table data stored in compressed, columnar, immutable files in the cloud provider's object storage, organised as micro-partitions | Cheap, durable, shared by all compute | A shared archive of read-only files |
# | **Compute: virtual warehouses** | Named clusters of compute you size (XS, S, M, L, ...) and that start, suspend and resume independently | Workload isolation; scale compute without moving data | Separate reading rooms |
# | **Cloud services** | Metadata, query optimiser, security, transaction management | Knows *which files* a query needs; enforces access control | The librarian with the catalogue |
#
# Consequences you will feel in practice:
# - **Isolation.** The data-science team's warehouse and the BI team's warehouse read the same tables but do not slow each other.
# - **Elasticity.** Resize a warehouse (or let it suspend when idle) without reloading data.
# - **You pay for compute while a warehouse runs, and separately for storage.** The habit that follows: suspend
#   warehouses aggressively (see notebook 05).
#
# Snowflake runs on AWS, Azure or Google Cloud; you choose one when you create an account.
# Exact regions, editions and features differ: *verify in the current docs*.

# %% [markdown]
# ## 3. Micro-partitions and pruning: measure it
#
# Snowflake automatically cuts every table into many small, immutable **micro-partitions** (the docs describe them as
# roughly tens to hundreds of MB of uncompressed data each; *verify the current figure*). For each partition and each
# column it stores **metadata**, such as the minimum and maximum value. When you run
# `WHERE order_date BETWEEN '2024-03-01' AND '2024-03-31'`, the optimiser reads only the metadata, and **skips every
# partition whose min/max range cannot contain a match**. That is **pruning**. You do not create indexes; you get
# this for free, but how well it works depends on how your data was *physically ordered* when it was loaded.
#
# Let us build a toy to see this. We cut our order-line table into chunks of 400 rows, record min/max per chunk,
# and count how many chunks a filter has to read.

# %%
ROWS_PER_PART = 400


def make_partitions(df: pd.DataFrame, rows: int = ROWS_PER_PART) -> list[pd.DataFrame]:
    return [df.iloc[i : i + rows] for i in range(0, len(df), rows)]


def partitions_scanned(parts: list[pd.DataFrame], col: str, lo, hi) -> int:
    """A partition must be read iff its [min, max] range overlaps [lo, hi]."""
    return sum(1 for p in parts if p[col].min() <= hi and p[col].max() >= lo)


rng = np.random.default_rng(42)
ordered = lines                                   # loaded in date order (typical for an append-only feed)
shuffled = lines.sample(frac=1, random_state=42)  # loaded in random order

lo, hi = pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-31")
for name, layout in [("date-ordered load", ordered), ("random-order load", shuffled)]:
    parts = make_partitions(layout)
    hit = partitions_scanned(parts, "order_date", lo, hi)
    print(f"{name:18s}: {hit:2d} of {len(parts)} partitions scanned")

# %% [markdown]
# *What just happened?* The same filter touches a handful of partitions when the data arrived in date order, and
# essentially **all** partitions when it arrived randomly, because every random chunk contains some March row and some
# non-March row and its min/max range covers March. Fewer partitions read means less I/O, less warehouse time, and lower cost.
#
# Two more facts this toy makes obvious:
# - Pruning works on *ranges*, so it helps filters on the column the data is **ordered by** and does little for the others.
# - Snowflake's own **clustering keys** (notebook 05) are the tool to re-organise a big table around the columns you filter on.

# %%
# A filter on a column the data is NOT ordered by prunes badly, even in the date-ordered layout
parts = make_partitions(ordered)
print("customer_id 500..520 on a date-ordered table:", partitions_scanned(parts, "customer_id", 500, 520), "of", len(parts))

by_customer = make_partitions(lines.sort_values(["customer_id", "item_id"]))
print("same filter after re-ordering by customer_id:", partitions_scanned(by_customer, "customer_id", 500, 520), "of", len(by_customer))

# %% [markdown]
# Real Snowflake is more sophisticated than this toy (partitions are not fixed row counts, metadata is richer, and
# the optimiser has more tricks), but the **min/max pruning idea and the importance of physical order** are accurate and
# worth remembering.

# %% [markdown]
# ## 4. Time Travel and zero-copy clone: a copy-on-write toy
#
# Because micro-partitions are **immutable**, an `UPDATE` never edits a file. It writes **new** partitions for the rows
# that changed and records a new *version* of the table: a list of pointers to partitions. The old list still exists,
# and as long as the old partitions are retained you can read it. That is **Time Travel**.
#
# A **clone** is then just *copying the list of pointers*. No data moves, so it takes about a second regardless of size,
# and the clone costs nothing extra until one side changes. When the clone changes a row, only that partition is
# rewritten. This is **copy-on-write**.
#
# Below is a ~30-line toy that behaves this way. It is a teaching model, not Snowflake's implementation.

# %%
_ids = itertools.count(1)


class ToyTable:
    """Immutable partitions + a list of versions. Each version is a tuple of partition ids."""

    def __init__(self, parts=None, store=None):
        self.store = dict(store or {})                      # partition id -> DataFrame (shared *objects*)
        self.versions = [tuple(parts or [])]                # version 0 = current list

    @classmethod
    def from_df(cls, df, rows=ROWS_PER_PART):
        t = cls()
        for p in make_partitions(df, rows):
            pid = next(_ids)
            t.store[pid] = p
            t.versions[0] += (pid,)
        return t

    def update(self, mask_fn, col, value) -> int:
        """Rewrite only the partitions that contain matching rows. Returns #partitions rewritten."""
        new, rewritten = [], 0
        for pid in self.versions[-1]:
            part = self.store[pid]
            m = mask_fn(part)
            if m.any():
                part2 = part.copy()
                part2.loc[m, col] = value
                npid = next(_ids)
                self.store[npid] = part2
                new.append(npid)
                rewritten += 1
            else:
                new.append(pid)                             # untouched: same partition reused
        self.versions.append(tuple(new))
        return rewritten

    def read(self, version: int = -1) -> pd.DataFrame:
        return pd.concat([self.store[p] for p in self.versions[version]], ignore_index=True)

    def clone(self) -> "ToyTable":
        t = ToyTable(store=self.store)                      # copies the dict of pointers, not the DataFrames
        t.versions = [self.versions[-1]]
        return t


sales = ToyTable.from_df(lines)
print("partitions:", len(sales.versions[-1]), " rows:", len(sales.read()))

# %% [markdown]
# **Time Travel.** An analyst runs a careless update: every line with `quantity >= 3` gets `quantity = 0`.
# Then we go back to the version before the mistake.

# %%
before_total = sales.read()["quantity"].sum()
rewritten = sales.update(lambda d: d["quantity"] >= 3, "quantity", 0)
after_total = sales.read()["quantity"].sum()
restored_total = sales.read(version=-2)["quantity"].sum()   # the version before the update

print(f"partitions rewritten: {rewritten} of {len(sales.versions[-1])}")
print(f"total quantity  before: {before_total}   after the bad update: {after_total}   read from old version: {restored_total}")
assert restored_total == before_total

# %% [markdown]
# *What just happened?* Nothing was deleted. The bad update produced new partitions and a new version; the old version
# still pointed at the old partitions, so reading it "travelled back in time". Real Snowflake keeps old partitions only for the table's
# **Time Travel retention period** (the default is short, and a longer maximum is available on higher editions: *verify in
# the current docs*), after which they move to a further, non-queryable safety window called Fail-safe. Longer retention
# means you pay for more stored history.
#
# **Zero-copy clone.** Clone the table, change 1 value in the clone, and see how much physical data is shared.

# %%
prod = ToyTable.from_df(lines)
dev = prod.clone()
n_rewritten = dev.update(lambda d: d["item_id"] == int(lines["item_id"].iloc[0]), "quantity", 99)

prod_ids, dev_ids = set(prod.versions[-1]), set(dev.versions[-1])
shared = prod_ids & dev_ids
print(f"prod partitions: {len(prod_ids)}, dev partitions: {len(dev_ids)}, shared: {len(shared)}")
print(f"physical partitions stored for both tables: {len(prod_ids | dev_ids)} (a full copy would need {len(prod_ids) * 2})")
print("prod untouched:", (prod.read()["quantity"] == 99).sum() == 0, "| dev changed:", (dev.read()["quantity"] == 99).sum() == 1)

# %% [markdown]
# *What just happened?* The clone shared every partition except the one we changed. That is why cloning a production
# table into a dev sandbox is cheap and fast, and why experiments on a clone cannot damage the original.
#
# In Snowflake the syntax is a one-liner (**not executed here; verify in current docs**):

# %% tags=["no-run"]
# Snowflake SQL, not run in this notebook
CLONE_AND_TRAVEL = """
-- create a dev copy of a production table (metadata-only operation)
CREATE TABLE dev.public.order_lines CLONE prod.public.order_lines;

-- read the table as it was 30 minutes ago (within the retention period)
SELECT COUNT(*) FROM prod.public.order_lines AT(OFFSET => -60*30);

-- bring back a table dropped by mistake (within the retention period)
UNDROP TABLE prod.public.order_lines;
"""

# %% [markdown]
# ## 5. Virtual warehouses in one paragraph
#
# A virtual warehouse is the thing that *executes* queries. You pick a **size** (each step up roughly doubles the compute and
# the credits consumed per hour; see notebook 05 for a calculator, and verify the credit table in the current docs),
# and it can **auto-suspend** when idle and **auto-resume** when a query arrives. Bigger is not always faster: a
# query that cannot be parallelised will not speed up, it will just cost more per minute. Conversely, a query that is
# slow because it scans too much data is often better fixed by pruning than by renting more compute.
#
# Multi-cluster warehouses (for many concurrent users, on some editions) add more copies of the same size to absorb concurrency;
# they do **not** make one big query faster. Check edition availability in the docs.

# %% [markdown]
# ## 6. When NOT to use Snowflake
#
# | Situation | Why it is a poor fit | Consider instead |
# |---|---|---|
# | A web app doing thousands of tiny single-row reads/writes per second | Built for scans and aggregates, not low-latency point lookups. (Snowflake has added transactional-style table types; *verify* current status and limits) | PostgreSQL / MySQL / a key-value store |
# | Your whole dataset fits comfortably on a laptop | Cloud warehouse overhead and cost for no benefit | DuckDB, SQLite, Postgres, Parquet files |
# | A warehouse left running 24/7 for a light workload | Compute is billed while running | Auto-suspend, smaller size, or a cheaper engine |
# | Sub-second streaming analytics on event streams | Micro-batch/ingest latency is not sub-second in general *(verify current ingest options)* | A streaming/real-time OLAP engine |
# | Strict requirement to keep data on your own hardware | It is a managed cloud service | Self-hosted engines |
#
# None of this means Snowflake is bad. It means: **match the tool to the access pattern**.

# %% [markdown]
# ## Common mistakes
#
# 1. **Leaving a large warehouse running.** Suspended warehouses cost no compute credits; idle running ones do.
# 2. **Fixing a slow query by sizing up first.** Look at how much data it scans (pruning) before you spend more.
# 3. **`SELECT *` on wide tables.** Storage is columnar, so reading only the columns you need is cheaper. (The toy above ignored columns, but the real
#    engine reads only the referenced columns' data.)
# 4. **Assuming Time Travel is a backup strategy.** It is a short window, not a long-term archive or a DR plan.
# 5. **Expecting indexes.** There are none to create; physical order (clustering) and query shape drive speed.

# %%
# Demonstration of mistake 4 using the toy: old versions are only readable while retained
sales_demo = ToyTable.from_df(lines.head(1200))
sales_demo.update(lambda d: d["quantity"] >= 3, "quantity", 0)
RETENTION_VERSIONS = 1   # toy "retention": keep 1 prior version
sales_demo.versions = sales_demo.versions[-(RETENTION_VERSIONS + 1):]
sales_demo.update(lambda d: d["quantity"] == 1, "quantity", 7)
sales_demo.versions = sales_demo.versions[-(RETENTION_VERSIONS + 1):]
print("versions still readable:", len(sales_demo.versions), "-> the original load is gone for good")

# %% [markdown]
# ## Interview corner
#
# **1. What is different about Snowflake's architecture compared with a traditional warehouse?**
# <details><summary>Answer</summary>
# Storage and compute are decoupled. Data sits in shared cloud object storage as immutable, columnar micro-partitions;
# independent virtual warehouses provide compute; a cloud-services layer holds metadata and does optimisation and security.
# That gives workload isolation, elastic scaling and pay-for-what-you-run compute.
# </details>
#
# **2. What are micro-partitions and how do they make queries fast?**
# <details><summary>Answer</summary>
# Automatically created, immutable, compressed columnar files with per-column metadata (such as min/max). The optimiser uses the
# metadata to skip partitions that cannot match a filter (pruning). It works best when data is physically ordered by the filtered
# column, which is what clustering keys are for.
# </details>
#
# **3. Why is cloning a huge table nearly instant?**
# <details><summary>Answer</summary>
# Clone copies metadata (pointers to the existing partitions), not data. Storage is shared until either table changes, and then only
# the changed partitions are rewritten (copy-on-write).
# </details>
#
# **4. A query is slow. List what you check before making the warehouse bigger.**
# <details><summary>Answer</summary>
# How much data is scanned versus returned (pruning), whether filters use the clustered/ordered column, whether `SELECT *` is reading
# unneeded columns, whether a join explodes rows, spilling to disk (memory pressure, where a bigger size does help), and queueing due to concurrency.
# </details>

# %% [markdown]
# ## Exercises
#
# The toy helpers (`make_partitions`, `partitions_scanned`, `ToyTable`) and the DataFrames `lines`, `ordered`, `shuffled` are defined above.

# %% [markdown]
# ### Exercise 1: How much does physical order matter?
# Using partitions of `ROWS_PER_PART` rows, count how many partitions a query for **`order_date` in the whole of January 2025**
# must scan on the date-ordered layout (`scanned_ordered`) and on the shuffled layout (`scanned_shuffled`).
# *Hint: `partitions_scanned(make_partitions(layout), "order_date", lo, hi)`.*

# %% tags=["starter"]
lo, hi = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")
scanned_ordered = ...
scanned_shuffled = ...

# %%
def _expected(layout):
    return sum(
        1
        for i in range(0, len(layout), ROWS_PER_PART)
        if layout["order_date"].iloc[i : i + ROWS_PER_PART].min() <= hi
        and layout["order_date"].iloc[i : i + ROWS_PER_PART].max() >= lo
    )


assert scanned_ordered == _expected(ordered), f"ordered layout: expected {_expected(ordered)}, got {scanned_ordered}"
assert scanned_shuffled == _expected(shuffled), f"shuffled layout: expected {_expected(shuffled)}, got {scanned_shuffled}"
assert scanned_ordered < scanned_shuffled, "pruning should be better on the date-ordered layout"
print(f"Exercise 1 passed ({scanned_ordered} vs {scanned_shuffled} partitions)")

# %% tags=["solution"]
lo, hi = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")
scanned_ordered = partitions_scanned(make_partitions(ordered), "order_date", lo, hi)
scanned_shuffled = partitions_scanned(make_partitions(shuffled), "order_date", lo, hi)

# %% [markdown]
# ### Exercise 2: Re-cluster for a different access pattern
# Dashboards now mostly filter by **`category`** (for example `"Books"`), not by date. Re-order `lines` so that filtering on
# `category == "Books"` scans as few partitions as possible, store the re-ordered DataFrame in `by_category`, and store the number of
# partitions scanned for `category == "Books"` in `scanned_books`. (`partitions_scanned` compares with `<=`/`>=`, which works for strings too.)
# *Hint: sort by `category`. This is exactly what a clustering key does, conceptually.*

# %% tags=["starter"]
by_category = ...
scanned_books = ...

# %%
n_books_rows = int((lines["category"] == "Books").sum())
min_possible = int(np.ceil(n_books_rows / ROWS_PER_PART))
assert len(by_category) == len(lines), "keep every row"
assert scanned_books == partitions_scanned(make_partitions(by_category), "category", "Books", "Books"), "scanned_books must be measured on your by_category layout"
assert scanned_books <= min_possible + 1, f"a perfectly clustered layout needs about {min_possible} partitions (+1 for the boundary), you got {scanned_books}"
baseline = partitions_scanned(make_partitions(ordered), "category", "Books", "Books")
print(f"Exercise 2 passed ({scanned_books} partitions vs {baseline} on the date-ordered layout)")

# %% tags=["solution"]
by_category = lines.sort_values(["category", "item_id"], ignore_index=True)
scanned_books = partitions_scanned(make_partitions(by_category), "category", "Books", "Books")

# %% [markdown]
# ### Exercise 3: Clone, change, measure
# Create `prod = ToyTable.from_df(lines)` and `dev = prod.clone()`. In `dev`, set `quantity` to 5 for **all lines whose `channel` is `"store"`**.
# Store the number of partitions `dev` rewrote in `rewritten` and the number of partitions `prod` and `dev` still share in `shared_count`.
# *Hint: `update` returns the rewritten count; shared = intersection of the two tables' current partition-id sets.*

# %% tags=["starter"]
prod = ToyTable.from_df(lines)
dev = prod.clone()
rewritten = ...
shared_count = ...

# %%
n_parts = len(prod.versions[-1])
expected_rewritten = sum(1 for p in make_partitions(lines) if (p["channel"] == "store").any())
assert rewritten == expected_rewritten, f"expected {expected_rewritten} rewritten partitions, got {rewritten}"
assert shared_count == n_parts - expected_rewritten, "shared = total partitions minus the ones the clone rewrote"
assert (prod.read()["quantity"] == 5).sum() == ((lines["quantity"] == 5).sum()), "prod must be unchanged"
print(f"Exercise 3 passed (rewrote {rewritten} of {n_parts} partitions; sharing depends on how *localised* the change is)")

# %% tags=["solution"]
prod = ToyTable.from_df(lines)
dev = prod.clone()
rewritten = dev.update(lambda d: d["channel"] == "store", "quantity", 5)
shared_count = len(set(prod.versions[-1]) & set(dev.versions[-1]))

# %% [markdown]
# ### Exercise 4 (open-ended): Would you put it in Snowflake?
# Pick three data workloads from your current or a past job. For each, write one line: *Snowflake / not Snowflake / depends*, and **which property of the
# architecture** (decoupled compute, pruning, clone, Time Travel, or the cost model) drives your answer. Put it in a markdown cell.

# %% [markdown]
# ## Recap
# - Snowflake decouples **storage** (immutable columnar micro-partitions), **compute** (virtual warehouses) and **cloud services** (metadata, optimiser, security).
# - **Pruning** skips partitions by min/max metadata; it works best when physical order matches your filters (clustering).
# - **Time Travel** and **zero-copy clone** follow from immutability: a new version is a new list of pointers.
# - Compute is billed while a warehouse runs; suspend early, size for the workload, and fix scans before buying compute.
# - It is the wrong tool for high-rate single-row transactions and for data that easily fits on a laptop.
#
# **Go deeper:** Snowflake documentation at docs.snowflake.com (the "Key Concepts and Architecture" section).
#
# **Next:** [Snowflake SQL and dialect differences](02_snowflake_sql_and_dialect_differences.ipynb)
