# %% [markdown]
# # File Formats: CSV, JSON and Parquet
#
# **Why this matters.** The format you store data in decides how big it is, how fast it loads, whether
# your column types survive the round trip, and how much a query has to read. A team that keeps a few
# hundred gigabytes of history as CSV pays for it in every single job; a team that switches to Parquet often
# sees the same jobs get much cheaper. Choosing a format is one of the cheapest, highest-impact decisions in data engineering.
#
# **You will learn**
# - measure file size and read time for CSV, JSON Lines and Parquet on the same table
# - read a Parquet schema and its metadata, and load only the columns or rows you need
# - write a partitioned Parquet dataset and read it back with a filter
# - handle a schema change (a new column) without silently losing data
#
# **Prerequisites:** [The data engineering landscape](01_data_engineering_landscape.ipynb) (row vs column storage).
#
# > Data used: the synthetic shop tables (`orders.csv`, `order_items.csv`, `products.csv`). See `docs/DATASETS.md`.

# %%
import atexit
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="fmt_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)      # nothing is left behind, even if a cell fails

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")

# One wide "fact" table: an order line with its order and product attributes.
lines = (items.merge(orders, on="order_id").merge(products, on="product_id"))
lines["line_total"] = (lines["quantity"] * lines["unit_price"]).round(2)
print(lines.shape)
lines.head(3)

# %% [markdown]
# ## 1. Same table, three formats
#
# * **CSV**: plain text, one row per line. Universal, human-readable, but every value is text (no types), nothing is compressed,
#   and to read one column you must parse the whole file.
# * **JSON Lines** (`.jsonl`): one JSON object per line. Good for nested or evolving records and for APIs and logs; the column names
#   repeat on **every row**, so it is even bulkier than CSV.
# * **Parquet**: binary, **columnar**, compressed, with the schema and per-column statistics stored in the file. The default choice for analytics.
#
# We repeat the table 10 times so that timings are not dominated by noise (the content is otherwise unchanged).

# %%
big = pd.concat([lines] * 10, ignore_index=True)
print(f"{len(big):,} rows x {big.shape[1]} columns")

paths = {
    "csv": TMP / "lines.csv",
    "jsonl": TMP / "lines.jsonl",
    "parquet_snappy": TMP / "lines_snappy.parquet",
    "parquet_zstd": TMP / "lines_zstd.parquet",
}
big.to_csv(paths["csv"], index=False)
big.to_json(paths["jsonl"], orient="records", lines=True, date_format="iso")
big.to_parquet(paths["parquet_snappy"], compression="snappy", index=False)
big.to_parquet(paths["parquet_zstd"], compression="zstd", index=False)

sizes = pd.Series({k: p.stat().st_size / 1e6 for k, p in paths.items()}, name="size_mb").round(2)
sizes.to_frame().assign(vs_csv=lambda d: (d["size_mb"] / d.loc["csv", "size_mb"]).round(2))

# %% [markdown]
# *What just happened?* Parquet is a fraction of the CSV size for two reasons you can now name: it stores each column contiguously
# (so **dictionary and run-length encoding** work well on repetitive columns like `status` or `category`) and then applies a general
# compression codec on top. `snappy` is fast and modest; `zstd` usually compresses more at similar speed. The exact ratio depends on your data.
#
# ## 2. Read time
# Now the cost of reading everything back (best of 3 runs):

# %%
def best_ms(fn, repeat=3):
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return round(min(out), 1)


read_ms = {
    "csv": best_ms(lambda: pd.read_csv(paths["csv"])),
    "jsonl": best_ms(lambda: pd.read_json(paths["jsonl"], lines=True)),
    "parquet_snappy": best_ms(lambda: pd.read_parquet(paths["parquet_snappy"])),
    "parquet_zstd": best_ms(lambda: pd.read_parquet(paths["parquet_zstd"])),
}
pd.Series(read_ms, name="read_all_ms")

# %% [markdown]
# Parquet is typically much faster to load because there is no text parsing and no type guessing. Numbers vary by machine, so read
# them as *relative*, not absolute.
#
# ## 3. Types survive Parquet, not CSV
# A CSV round trip forgets that `order_date` was a date and that `quantity` was an integer.

# %%
csv_back = pd.read_csv(paths["csv"])
pq_back = pd.read_parquet(paths["parquet_snappy"])
cmp = pd.DataFrame({"original": big.dtypes.astype(str), "after CSV": csv_back.dtypes.astype(str), "after Parquet": pq_back.dtypes.astype(str)})
cmp.loc[["order_date", "quantity", "unit_price", "category"]]

# %% [markdown]
# The date came back from CSV as plain text (`str`) and you would have to re-parse it (and remember to, in every consumer).
# Parquet returned exactly what we wrote. This is not only convenience: type drift between producer and consumer is a classic source of silent bugs
# (`"007"` becoming the number `7`, for instance).

# %% [markdown]
# ## 4. Looking inside a Parquet file
# The schema, row groups and column statistics are all in the footer, readable **without loading the data**.

# %%
pf = pq.ParquetFile(paths["parquet_snappy"])
print(pf.schema_arrow)
print("row groups:", pf.num_row_groups, "| rows:", pf.metadata.num_rows)

# %%
col_idx = pf.schema_arrow.get_field_index("quantity")
stats = pf.metadata.row_group(0).column(col_idx).statistics
print("row group 0, column 'quantity': min =", stats.min, "max =", stats.max, "nulls =", stats.null_count)

# %% [markdown]
# Those min/max statistics per row group are what let a reader **skip** entire chunks of a file when you filter
# (`WHERE quantity > 100` can skip a row group whose max is 5). That trick is called **predicate pushdown**.
#
# ## 5. Column pruning and row filtering
# CSV must parse everything even if you want two columns. Parquet reads only the column chunks you ask for.

# %%
t_all = best_ms(lambda: pd.read_parquet(paths["parquet_snappy"]))
t_two = best_ms(lambda: pd.read_parquet(paths["parquet_snappy"], columns=["category", "line_total"]))
t_csv_two = best_ms(lambda: pd.read_csv(paths["csv"], usecols=["category", "line_total"]))
print(f"parquet, all {big.shape[1]} columns: {t_all:7.1f} ms")
print(f"parquet, 2 columns:  {t_two:7.1f} ms")
print(f"csv,     2 columns:  {t_csv_two:7.1f} ms   (still parses every line)")

# %%
only_returned = pd.read_parquet(paths["parquet_snappy"], filters=[("status", "==", "returned")])
print(len(only_returned), "returned lines; all statuses:", only_returned["status"].unique().tolist())
print("matches an in-memory filter:", len(only_returned) == int((big["status"] == "returned").sum()))

# %% [markdown]
# *What just happened?* `columns=` is **column pruning** and `filters=` is **predicate pushdown**. Both mean "do not read what you do not need",
# the same idea as the column-store bytes argument in the previous notebook, now on real files.

# %% [markdown]
# ## 6. Partitioned datasets
# Big tables are stored as **many files in folders named by a column value**, so a query on that column opens only the relevant folders
# (**partition pruning**). The classic choice is a date. Pick a partition column you often filter on and with **low cardinality**:
# thousands of tiny files is slower than a few hundred big ones.

# %%
part_root = TMP / "lines_by_month"
partitioned = lines.assign(order_month=lines["order_date"].dt.strftime("%Y-%m"))
partitioned.to_parquet(part_root, partition_cols=["order_month"], index=False)

folders = sorted(p.name for p in part_root.iterdir())
print(len(folders), "partitions, e.g.", folders[:3])
print("files in the first partition:", [f.name for f in (part_root / folders[0]).iterdir()])

# %%
dset = ds.dataset(part_root, format="parquet", partitioning="hive")
dec = dset.to_table(filter=ds.field("order_month") == "2023-12", columns=["order_id", "line_total"]).to_pandas()
print(f"December 2023: {len(dec)} lines, revenue {dec['line_total'].sum():,.2f}")
expected = lines.loc[lines["order_date"].dt.strftime("%Y-%m") == "2023-12", "line_total"].sum()
print("matches in-memory calculation:", abs(dec["line_total"].sum() - expected) < 1e-6)

# %% [markdown]
# The folder name `order_month=2023-12` *is* the data: the reader only opened that one folder (the partition column is reconstructed from the path).
# The same layout (`key=value/` folders, "Hive-style") is understood by Spark, DuckDB, Athena, BigQuery external tables and most other engines.

# %% [markdown]
# ## 7. Schema evolution: a new column appears
# Sources change. Suppose last week's file has 2 columns and this week's producer added a third. A careless reader loses the new column
# without any error:

# %%
evo = TMP / "evolve"
evo.mkdir()
pd.DataFrame({"order_id": [1, 2], "channel": ["web", "store"]}).to_parquet(evo / "week1.parquet", index=False)
pd.DataFrame({"order_id": [3], "channel": ["mobile"], "coupon": ["SAVE10"]}).to_parquet(evo / "week2.parquet", index=False)

naive = ds.dataset(evo, format="parquet")                   # schema is taken from the FIRST file
print("columns seen by the naive reader:", naive.schema.names)
print(naive.to_table().to_pandas())

# %%
unified = pa.unify_schemas([pq.read_schema(p) for p in sorted(evo.glob("*.parquet"))])
safe = ds.dataset(evo, format="parquet", schema=unified)
print("columns with an explicit unified schema:", safe.schema.names)
safe.to_table().to_pandas()

# %% [markdown]
# With an explicit unified schema, old files get `null` for the new column and nothing is dropped. What is **safe** to evolve?
#
# | Change | Safe for old readers? | Notes |
# |---|---|---|
# | add a nullable column | yes | old rows read as null |
# | drop a column | only if no consumer uses it | announce first, then drop |
# | rename a column | **no** | treat as add + backfill + drop |
# | widen a type (int32 to int64) | usually | check every consumer |
# | change meaning or unit (dollars to cents) | **no**, and silent | the worst kind; needs a versioned contract |
#
# A type change that cannot be cast fails loudly, which is what you want:

# %%
pd.DataFrame({"order_id": ["not-a-number"], "channel": ["web"]}).to_parquet(evo / "week3.parquet", index=False)
try:
    ds.dataset(evo, format="parquet", schema=unified).to_table()
except Exception as e:  # noqa: BLE001
    print(type(e).__name__, "->", str(e)[:100])
(evo / "week3.parquet").unlink()

# %% [markdown]
# ## Choosing a format
#
# | Situation | Use |
# |---|---|
# | Analytics tables, ML training data, anything queried by column | **Parquet** (partitioned if large) |
# | Exchanging a small file with a human or a spreadsheet | CSV |
# | Events, logs, API payloads, nested/evolving records | JSON Lines (convert to Parquet downstream) |
# | Need updates, deletes, time travel on files | a table format on Parquet (Delta Lake, Iceberg), see the lakehouse notes |

# %% [markdown]
# ## Common mistakes
#
# 1. **Treating CSV as a database.** No types, no schema, ambiguous quoting and encodings, and a dropped column is invisible. Fix: write Parquet; if CSV is forced on you, validate it on load (next notebooks).
# 2. **Partitioning by a high-cardinality column** (like `customer_id`): millions of tiny files, slow listing, slow queries. Partition by date or a low-cardinality category.
# 3. **Reading the whole file and filtering afterwards** in pandas. Use `columns=` and `filters=` so the reader skips data.
# 4. **Many tiny files** from a job that writes every minute. Compact them periodically into files of roughly hundreds of MB.
# 5. **Reading a dataset with the first file's schema** and silently losing new columns (shown above). Pass an explicit schema.

# %% [markdown]
# ## Interview corner
#
# **1. Why is Parquet faster than CSV for analytics?**
# <details><summary>Answer</summary>
# It is columnar (read only needed columns), compressed with per-column encodings, typed (no parsing or inference), and stores min/max statistics
# per row group so readers can skip data that cannot match a filter.
# </details>
#
# **2. What is partitioning and what can go wrong?**
# <details><summary>Answer</summary>
# Splitting a dataset into folders by column values so queries read only relevant folders. Too fine a partition key creates huge numbers of small files
# (slow metadata operations); a key nobody filters on gives no benefit; skewed keys create one giant partition.
# </details>
#
# **3. A producer adds a column to a Parquet feed. What breaks?**
# <details><summary>Answer</summary>
# Nothing loudly, if readers take the first file's schema: the new column is ignored. Readers that expect a fixed set of columns may fail;
# readers that merge schemas get nulls for older files. Use an explicit schema or a schema registry / data contract and test for drift.
# </details>
#
# **4. When would you still choose JSON or CSV?**
# <details><summary>Answer</summary>
# JSON for nested or schema-flexible event data and API responses, CSV for interchange with people and tools that only speak CSV. Land them raw, then
# convert to Parquet in the next layer.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: How much smaller is zstd Parquet?
# Compute `zstd_ratio` = size of `lines_zstd.parquet` divided by size of `lines.csv` (use `paths` and `Path.stat().st_size`).

# %% tags=["starter"]
zstd_ratio = ...

# %%
exp = paths["parquet_zstd"].stat().st_size / paths["csv"].stat().st_size
assert abs(zstd_ratio - exp) < 1e-9, "divide the zstd parquet size (bytes) by the csv size (bytes)"
assert zstd_ratio < 1, "parquet should be smaller than csv for this table"
print(f"Exercise 1 passed: zstd Parquet is {zstd_ratio:.1%} of the CSV size")

# %% tags=["solution"]
zstd_ratio = paths["parquet_zstd"].stat().st_size / paths["csv"].stat().st_size

# %% [markdown]
# ### Exercise 2: Read only what you need
# Read from `paths["parquet_snappy"]` **only** the columns `product_name` and `quantity`, **only** for rows where `channel == "web"`, into `web_qty`
# (a DataFrame). Then compute `web_units` = the total quantity of the web lines *in the file* (which contains the table 10 times over).
# *Hint: `pd.read_parquet(path, columns=[...], filters=[(col, op, value)])`.*

# %% tags=["starter"]
web_qty = ...
web_units = ...

# %%
assert list(web_qty.columns) == ["product_name", "quantity"], "read exactly those two columns"
exp_units = int(big.loc[big["channel"] == "web", "quantity"].sum())
assert len(web_qty) == int((big["channel"] == "web").sum()), "filter on channel == 'web' while reading"
assert web_units == exp_units, f"expected {exp_units}"
print("Exercise 2 passed")

# %% tags=["solution"]
web_qty = pd.read_parquet(paths["parquet_snappy"], columns=["product_name", "quantity"], filters=[("channel", "==", "web")])
web_units = int(web_qty["quantity"].sum())

# %% [markdown]
# ### Exercise 3: Partition pruning by hand
# Using the partitioned dataset `dset` (partitioned by `order_month`), compute `q4_revenue`: the total `line_total` for the three months
# `2023-10`, `2023-11` and `2023-12`. Read only `order_month` and `line_total`, and let the dataset filter use `isin`.
# *Hint: `ds.field("order_month").isin([...])`.*

# %% tags=["starter"]
q4_revenue = ...

# %%
months = ["2023-10", "2023-11", "2023-12"]
exp = lines.loc[lines["order_date"].dt.strftime("%Y-%m").isin(months), "line_total"].sum()
assert abs(q4_revenue - exp) < 1e-6, f"expected {exp:.2f}"
print(f"Exercise 3 passed: Q4 2023 revenue {q4_revenue:,.2f}")

# %% tags=["solution"]
tbl = dset.to_table(filter=ds.field("order_month").isin(["2023-10", "2023-11", "2023-12"]), columns=["order_month", "line_total"])
q4_revenue = float(tbl.to_pandas()["line_total"].sum())

# %% [markdown]
# ### Exercise 4: Type drift on CSV
# `csv_back["order_date"]` is text. Convert it back and store in `csv_dates_ok` the number of rows whose parsed date equals the original `big["order_date"]`
# (should be all of them). Then explain in one markdown line why you still prefer Parquet.

# %% tags=["starter"]
csv_dates_ok = ...

# %%
assert csv_dates_ok == len(big), "parse with pd.to_datetime and compare element-wise"
print("Exercise 4 passed")

# %% tags=["solution"]
csv_dates_ok = int((pd.to_datetime(csv_back["order_date"]) == big["order_date"]).sum())

# %% [markdown]
# ### Exercise 5 (open-ended): Pick a partition key
# `lines` has `channel`, `status`, `category`, `order_month`. For each, print how many distinct values it has and decide which you would partition a
# 5-year, 5-billion-row version by. Write your reasoning (cardinality, typical filters, skew) in a markdown cell.

# %% [markdown]
# ## Recap
# - CSV = simple but untyped and bulky; JSON Lines = flexible but bulkier still; Parquet = typed, columnar, compressed: the analytics default.
# - The Parquet footer holds schema and min/max stats; use `columns=` (pruning) and `filters=` (pushdown) to read less.
# - Partition big tables by a low-cardinality column you filter on (usually a date); avoid many tiny files.
# - Schema evolution: adding nullable columns is safe if readers use an explicit unified schema; renames and meaning changes are not.
#
# **Go deeper:** the Apache Arrow / PyArrow documentation on `pyarrow.parquet` and `pyarrow.dataset` (arrow.apache.org/docs/python) and the pandas I/O guide.
#
# **Next:** [Data modelling: normalisation and star schema](03_data_modeling_star_schema_and_normalization.ipynb)
