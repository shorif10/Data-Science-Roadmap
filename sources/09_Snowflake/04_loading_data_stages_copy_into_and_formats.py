# %% [markdown]
# # Loading Data: Stages, COPY INTO and File Formats
#
# **Why this matters.** Getting data *in* is where most warehouse projects lose days: a date that does not parse, a file with one extra
# column, the same file loaded twice, a 40 GB CSV in a single file that loads on one thread. Snowflake's bulk-load path is
# `files -> stage -> COPY INTO table`. Most of the pain is avoidable by **preparing and validating files before they leave your machine**, which is
# exactly what you can practise offline here with `pyarrow`.
#
# **You will learn**
# - explain stages, file formats and `COPY INTO`, and what `ON_ERROR` changes
# - choose between CSV, JSON and Parquet for a load, and see the difference in a measurement
# - write Parquet with `pyarrow`, split into sensibly sized files, and validate schema and row counts before upload
# - reproduce COPY's error-handling modes locally so you know what to expect
#
# **Prerequisites:** [Snowflake architecture](01_snowflake_architecture_and_concepts.ipynb), [Python connector](03_python_connector_and_snowpark.ipynb).
#
# > **Candour box.** Snowflake commands here (`CREATE STAGE`, `PUT`, `COPY INTO`, ...) are tagged `no-run` and were **not executed**:
# > written from the documented syntax; *verify options in the current docs* (docs.snowflake.com, "Data loading" section). The file
# > preparation, validation and the local COPY imitation **do run**, and the files it writes into `exports/` are ready for you to upload to a real trial account.

# %%
import csv
import math
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path("../datasets")
EXPORTS = Path("exports")
EXPORTS.mkdir(exist_ok=True)

orders = pd.read_csv(DATA / "orders.csv")
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")

lines = (
    items.merge(orders, on="order_id").merge(products, on="product_id")
    .assign(revenue=lambda d: (d["quantity"] * d["unit_price"]).round(2),
            order_date=lambda d: pd.to_datetime(d["order_date"]))
    [["item_id", "order_id", "customer_id", "order_date", "status", "channel", "product_id", "category", "quantity", "revenue"]]
    .sort_values("item_id", ignore_index=True)
)
print(lines.shape)
lines.dtypes

# %% [markdown]
# ## 1. The loading path in one picture
#
# ```
# your files --PUT / upload--> STAGE (a named storage location) --COPY INTO--> TABLE
#                                   |-- internal stage: storage managed by Snowflake
#                                   '-- external stage: your S3 / Azure Blob / GCS location (needs credentials or a storage integration)
# ```
#
# | Concept | What it is | Why it exists |
# |---|---|---|
# | **Stage** | A named pointer to a place where files sit. Internal (Snowflake-managed) or external (your cloud bucket) | Decouples "getting files there" from "loading them" |
# | **File format** | A reusable definition: `TYPE = CSV/JSON/PARQUET/...`, delimiter, header rows, null strings, compression | So the parsing rules live in one reviewed object, not in every command |
# | **COPY INTO table** | Bulk-loads staged files. Tracks which files it already loaded so re-running does not duplicate them (for a limited window; *verify the current period*) | Fast parallel load with built-in error handling |
# | **Snowpipe** | Continuous, event-driven loading of new files as they arrive | Near-real-time ingestion instead of scheduled batches |
#
# The right mental model: **COPY INTO is a batch bulk loader; `INSERT` row by row is the slow path** (fine for a handful of rows, wrong for a million).

# %% tags=["no-run"]
LOAD_SQL = """
-- 1. reusable parsing rules
CREATE OR REPLACE FILE FORMAT ff_csv
  TYPE = CSV  FIELD_DELIMITER = ','  SKIP_HEADER = 1  NULL_IF = ('', 'NULL')  FIELD_OPTIONALLY_ENCLOSED_BY = '"';
CREATE OR REPLACE FILE FORMAT ff_parquet TYPE = PARQUET;

-- 2. a named internal stage using that format
CREATE OR REPLACE STAGE shop_stage FILE_FORMAT = (FORMAT_NAME = ff_csv);

-- 3. upload local files (PUT runs from SnowSQL or a connector session, not from the web worksheet; give an absolute local path; verify in docs)
PUT file://exports/order_lines.csv @shop_stage AUTO_COMPRESS = TRUE;
LIST @shop_stage;

-- 4. target table with real types
CREATE OR REPLACE TABLE order_lines (
  item_id INT, order_id INT, customer_id INT, order_date DATE, status STRING, channel STRING,
  product_id INT, category STRING, quantity INT, revenue NUMBER(12,2)
);

-- 5. dry-run first: report errors without loading
COPY INTO order_lines FROM @shop_stage/order_lines.csv.gz VALIDATION_MODE = RETURN_ERRORS;

-- 6. load for real
COPY INTO order_lines FROM @shop_stage/order_lines.csv.gz ON_ERROR = ABORT_STATEMENT;

-- Parquet: match Parquet columns to table columns by name instead of position
COPY INTO order_lines FROM @shop_stage/order_lines.parquet
  FILE_FORMAT = (FORMAT_NAME = ff_parquet) MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- 7. verify: the row count must equal what you exported (see the local check below)
SELECT COUNT(*), SUM(revenue) FROM order_lines;
"""

# %% [markdown]
# ## 2. Prepare the files (these run)
#
# We export the same order-line table in three formats into `exports/`. Then we *measure* them rather than repeat folklore.

# %%
# CSV (dates as ISO strings)
csv_path = EXPORTS / "order_lines.csv"
lines.to_csv(csv_path, index=False, date_format="%Y-%m-%d")

# Parquet with real types: convert the timestamp to a pure DATE so it loads into a DATE column
tbl = pa.Table.from_pandas(lines, preserve_index=False)
tbl = tbl.set_column(tbl.schema.get_field_index("order_date"), "order_date", tbl["order_date"].cast(pa.date32()))
parquet_path = EXPORTS / "order_lines.parquet"
pq.write_table(tbl, parquet_path, compression="snappy")

# newline-delimited JSON (one document per line): loads into a VARIANT column
ndjson_path = EXPORTS / "order_lines.ndjson"
lines.assign(order_date=lines["order_date"].dt.strftime("%Y-%m-%d")).to_json(ndjson_path, orient="records", lines=True)

sizes = pd.Series({p.name: p.stat().st_size for p in [csv_path, parquet_path, ndjson_path]}, name="bytes")
sizes.to_frame().assign(kb=lambda d: (d["bytes"] / 1024).round(0), vs_csv=lambda d: (d["bytes"] / sizes["order_lines.csv"]).round(2))

# %% [markdown]
# *What just happened?* Same data, three sizes. Parquet is columnar and compressed (and carries the schema); newline JSON repeats every key on every row and is
# the largest. Exact ratios depend on the data (these are synthetic), so treat the numbers as an illustration, not a benchmark.
#
# | Format | Pros | Cons | Use it when |
# |---|---|---|---|
# | CSV | Universal, human-readable | No types (dates/numbers are text), quoting/escaping bugs, no schema | Interchange with legacy tools |
# | JSON / NDJSON | Nested and evolving structure; loads to `VARIANT` | Bulky, slower to parse, weak typing | API/event payloads |
# | Parquet | Typed, columnar, compressed, self-describing | Not human-readable; write with a library | Analytics loads and data-lake files: the default choice |

# %% [markdown]
# ## 3. Validate before you upload
#
# Cheap checks locally save a failed load and a re-upload. Three that catch most problems: **row count**, **schema against the target table**, and
# **null/duplicate key checks**. Record the expected values now; after `COPY INTO` you compare them with `SELECT COUNT(*), SUM(revenue)` in Snowflake.

# %%
meta = pq.ParquetFile(parquet_path).metadata
print("parquet rows:", meta.num_rows, "| row groups:", meta.num_row_groups, "| columns:", meta.num_columns)
print("csv rows    :", sum(1 for _ in open(csv_path, encoding="utf-8")) - 1, "(minus header)")
assert meta.num_rows == len(lines)

expected_target = {  # what the Snowflake table DDL above declares, expressed as Arrow types
    "item_id": pa.int64(), "order_id": pa.int64(), "customer_id": pa.int64(), "order_date": pa.date32(),
    "status": None, "channel": None, "product_id": pa.int64(), "category": None, "quantity": pa.int64(), "revenue": pa.float64(),
}  # None = any string type


def schema_problems(path, expected):
    """Names of columns that are missing or have an unexpected Arrow type."""
    schema = pq.read_schema(path)
    problems = []
    for name, typ in expected.items():
        if name not in schema.names:
            problems.append(name)
        elif typ is not None and schema.field(name).type != typ:
            problems.append(name)
        elif typ is None and not (pa.types.is_string(schema.field(name).type) or pa.types.is_large_string(schema.field(name).type)):
            problems.append(name)
    return problems


print("schema problems:", schema_problems(parquet_path, expected_target))
control = {"rows": len(lines), "sum_revenue": round(float(lines["revenue"].sum()), 2), "distinct_items": int(lines["item_id"].nunique())}
control

# %% [markdown]
# These three numbers are your **control totals**: after loading, `SELECT COUNT(*), SUM(revenue), COUNT(DISTINCT item_id) FROM order_lines` must return the same. If it does not, stop and investigate before
# anything downstream is built on it. (Snowflake stores `NUMBER(12,2)` exactly; floats may differ in the last digits, so compare with a tolerance.)

# %% [markdown]
# ## 4. File size and parallelism
#
# A warehouse loads **files in parallel**, one file per thread (roughly). One giant file is loaded by one thread; thousands of tiny files carry per-file overhead. The
# documentation recommends files in the range of roughly 100-250 MB compressed (*verify the current guidance*). Our data is tiny, so we imitate the *splitting logic* with a small row target:

# %%
def split_to_parquet(df, out_dir, rows_per_file):
    paths = []
    for i, start in enumerate(range(0, len(df), rows_per_file)):
        t = pa.Table.from_pandas(df.iloc[start : start + rows_per_file], preserve_index=False)
        p = Path(out_dir) / f"part-{i:04d}.parquet"
        pq.write_table(t, p, compression="snappy")
        paths.append(p)
    return paths


with tempfile.TemporaryDirectory() as tmp:
    parts = split_to_parquet(lines, tmp, rows_per_file=3000)
    counts = [pq.ParquetFile(p).metadata.num_rows for p in parts]
print("files:", len(counts), "rows per file:", counts, "total:", sum(counts))
assert sum(counts) == len(lines)

# %% [markdown]
# ## 5. What `ON_ERROR` changes: a local imitation
#
# Real data is dirty. `COPY INTO` lets you choose what happens when a row cannot be parsed: `ABORT_STATEMENT` (the documented default for `COPY INTO <table>`: load nothing),
# `CONTINUE` (skip bad rows, load the rest), `SKIP_FILE` (skip any file with an error), plus threshold variants such as `SKIP_FILE_<n>` (*verify current option list*). Below is a small
# local imitation, and a deliberately corrupted file. It is a toy of the *semantics*, not of Snowflake's parser.

# %%
class LoadError(Exception):
    pass


PARSERS = {
    "item_id": int, "order_id": int, "customer_id": int, "order_date": date.fromisoformat, "status": str, "channel": str,
    "product_id": int, "category": str, "quantity": int, "revenue": float,
}


def local_copy(path, on_error="ABORT_STATEMENT"):
    """Return (rows_loaded, errors). Imitates COPY INTO's handling of bad rows for one CSV file with a header."""
    good, errors = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # SKIP_HEADER = 1
        for n, row in enumerate(reader, start=2):
            try:
                if len(row) != len(PARSERS):
                    raise ValueError(f"expected {len(PARSERS)} fields, found {len(row)}")
                good.append([parse(v) for parse, v in zip(PARSERS.values(), row)])
            except ValueError as e:
                errors.append((n, str(e)))
    if errors and on_error == "ABORT_STATEMENT":
        raise LoadError(f"{len(errors)} error(s), first at line {errors[0][0]}: {errors[0][1]}. Nothing loaded.")
    if errors and on_error == "SKIP_FILE":
        return 0, errors
    return len(good), errors        # CONTINUE, or no errors at all


# build a dirty file in a temp dir: first 200 rows, then corrupt four of them
dirty_dir = Path(tempfile.mkdtemp())
dirty_path = dirty_dir / "dirty.csv"
rows = list(csv.reader(open(csv_path, newline="", encoding="utf-8")))[:201]   # header + 200 rows
CORRUPTED = {10: ("quantity", "two"), 50: ("order_date", "2024-13-45")}
for i, (col, bad) in CORRUPTED.items():
    rows[i][list(PARSERS).index(col)] = bad
rows[90] = rows[90][:-1]            # missing last field
rows[120] = rows[120] + ["oops"]    # extra field
CORRUPTED[90] = CORRUPTED[120] = ("shape", "")
with open(dirty_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

try:
    local_copy(dirty_path)
except LoadError as e:
    print("ABORT_STATEMENT ->", e)
loaded, errs = local_copy(dirty_path, "CONTINUE")
print("CONTINUE        -> loaded", loaded, "of 200 rows; error lines:", [n for n, _ in errs])
print("SKIP_FILE       -> loaded", local_copy(dirty_path, "SKIP_FILE")[0])
assert loaded == 200 - len(CORRUPTED) and len(errs) == len(CORRUPTED)

# %% [markdown]
# *What just happened?* Four bad rows: the default mode loaded nothing and told us the first bad line; `CONTINUE` loaded the other 196; `SKIP_FILE` dropped the whole file.
# **`CONTINUE` is dangerous when you do not check afterwards**: you silently have 196 of 200 rows. If you use it, you must reconcile row counts (Snowflake exposes load results via the
# `COPY INTO` output and history views; verify names in the docs) and route rejected rows somewhere a human looks.
#
# Clean up the temp file used above:

# %%
import shutil
shutil.rmtree(dirty_dir)
print("temp dir removed:", not dirty_dir.exists())

# %% [markdown]
# ## Common mistakes
#
# 1. **Loading CSV dates without checking the format**: `03/04/2025` is March 4th or April 3rd depending on the parser settings. Export ISO `YYYY-MM-DD`.
# 2. **Header row loaded as data** (or the first data row dropped): set `SKIP_HEADER` deliberately, and compare row counts.
# 3. **`ON_ERROR = CONTINUE` without reconciliation**: silent data loss.
# 4. **One huge file** (no parallelism) or **millions of tiny files** (overhead). Aim for the documented size range.
# 5. **Trusting float sums for money**: use `NUMBER(p,s)` in the table and compare control totals with a tolerance.
# 6. **Committing credentials in an external stage definition**: use a storage integration or your platform's secrets mechanism.

# %%
# Demonstration of mistake 1: same string, different meaning
s = "03/04/2025"
print(pd.to_datetime(s, format="%d/%m/%Y").date(), "vs", pd.to_datetime(s, format="%m/%d/%Y").date())

# %% [markdown]
# ## Interview corner
#
# **1. What is a stage and how is it different from a table?**
# <details><summary>Answer</summary>
# A stage is a named location holding files (internal to Snowflake, or an external cloud bucket). It is not queryable like a table by default; you load from it with <code>COPY INTO</code> (or query files directly with specific syntax).
# It separates "where files land" from "how and when they are loaded".
# </details>
#
# **2. Why is Parquet usually preferred over CSV for warehouse loads?**
# <details><summary>Answer</summary>
# It is typed and self-describing (no date/number parsing ambiguity), columnar and compressed (smaller, faster to read), and carries schema for column matching by name. CSV needs delimiter, quoting and null conventions agreed by both sides.
# </details>
#
# **3. How do you make sure a load did not lose or duplicate data?**
# <details><summary>Answer</summary>
# Record control totals at the source (row count, sums of key measures, distinct keys), run a validation/dry-run mode first, avoid silent-skip error modes or reconcile after them, and compare the same totals in the target table.
# Loading is tracked per file, so re-running the same load does not re-insert files it already loaded (within the tracking window).
# </details>
#
# **4. COPY INTO versus Snowpipe?**
# <details><summary>Answer</summary>
# <code>COPY INTO</code> is a batch command you (or a scheduler) run on a warehouse. Snowpipe loads new files continuously as they arrive in a stage, using Snowflake-managed compute, which suits low-latency ingestion. Batch is simpler and easier to reason about; choose Snowpipe when freshness matters.
# </details>

# %% [markdown]
# ## Exercises
# `lines` (DataFrame), `csv_path`, `parquet_path`, `schema_problems`, `split_to_parquet` and `local_copy` are defined above. `dirty_path` was deleted with its folder; the exercise recreates what it needs.

# %% [markdown]
# ### Exercise 1: Split for parallel loading
# Write `lines` to a temporary directory in files of **2,500 rows** using `split_to_parquet`, then store the **number of files** in `n_files` and the **row count of the last file** in `last_file_rows`
# (read it from Parquet metadata, not from arithmetic). Keep the directory only inside the `with` block.

# %% tags=["starter"]
with tempfile.TemporaryDirectory() as tmp:
    paths = split_to_parquet(lines, tmp, rows_per_file=2500)
    n_files = ...
    last_file_rows = ...

# %%
_q, _r = divmod(len(lines), 2500)
assert n_files == _q + (1 if _r else 0), f"expected {_q + (1 if _r else 0)} files"
assert last_file_rows == (_r or 2500), f"the last file should hold {_r or 2500} rows"
print("Exercise 1 passed")

# %% tags=["solution"]
with tempfile.TemporaryDirectory() as tmp:
    paths = split_to_parquet(lines, tmp, rows_per_file=2500)
    n_files = len(paths)
    last_file_rows = pq.ParquetFile(paths[-1]).metadata.num_rows

# %% [markdown]
# ### Exercise 2: Predict the load
# Build a dirty CSV in a temp directory: the first 300 data rows of `csv_path`, with the `quantity` value replaced by `"N/A"` in data rows **5, 77 and 250** (1-based, counting data rows after the header).
# Run `local_copy` with `CONTINUE` and store the rows loaded in `loaded_continue`; then with `SKIP_FILE`, storing in `loaded_skip`. Finally set `abort_raises` to `True` if the default mode raises `LoadError`.
# *Hint: `rows = list(csv.reader(open(csv_path)))[:301]`, quantity is column index `list(PARSERS).index("quantity")`.*

# %% tags=["starter"]
loaded_continue = ...
loaded_skip = ...
abort_raises = ...

# %%
assert loaded_continue == 300 - 3, f"3 bad rows out of 300 -> 297 loaded, got {loaded_continue}"
assert loaded_skip == 0, "SKIP_FILE loads nothing from a file that has any error"
assert abort_raises is True, "the default ABORT_STATEMENT mode should raise LoadError"
print("Exercise 2 passed")

# %% tags=["solution"]
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "ex2.csv"
    rows = list(csv.reader(open(csv_path, newline="", encoding="utf-8")))[:301]
    qi = list(PARSERS).index("quantity")
    for n in (5, 77, 250):
        rows[n][qi] = "N/A"
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    loaded_continue = local_copy(p, "CONTINUE")[0]
    loaded_skip = local_copy(p, "SKIP_FILE")[0]
    try:
        local_copy(p)
        abort_raises = False
    except LoadError:
        abort_raises = True

# %% [markdown]
# ### Exercise 3: Catch a schema drift before upload
# A colleague's export renames `revenue` to `net_revenue` and stores `quantity` as a string. Create `drifted` = a copy of the Arrow table `tbl` with those two changes, write it to a temp
# Parquet file, and store in `problems` the list returned by `schema_problems(that_file, expected_target)` (sorted).
# *Hint: `tbl.rename_columns([...])` and `.cast` on a column, or rebuild the table from pandas after `astype`.*

# %% tags=["starter"]
problems = ...

# %%
assert problems == ["quantity", "revenue"], f"expected ['quantity', 'revenue'], got {problems}"
print("Exercise 3 passed")

# %% tags=["solution"]
_df = lines.rename(columns={"revenue": "net_revenue"}).assign(quantity=lambda d: d["quantity"].astype(str))
_t = pa.Table.from_pandas(_df, preserve_index=False)
_t = _t.set_column(_t.schema.get_field_index("order_date"), "order_date", _t["order_date"].cast(pa.date32()))
with tempfile.TemporaryDirectory() as tmp:
    _p = Path(tmp) / "drifted.parquet"
    pq.write_table(_t, _p)
    problems = sorted(schema_problems(_p, expected_target))

# %% [markdown]
# ### Exercise 4 (open-ended): Design a staging layout
# Orders arrive daily from a source system. In a markdown cell propose: (a) the stage path convention (for example by date), (b) the file format and why, (c) how you would prevent loading the same day twice,
# (d) which control totals you would record, and (e) when you would move from scheduled `COPY INTO` to Snowpipe.

# %% [markdown]
# ## Recap
# - Path: files -> **stage** -> `COPY INTO` table, with a reusable **file format**. Prefer Parquet for typed analytic loads.
# - Validate locally first: row counts, schema vs target DDL, keys, and record **control totals** to reconcile after loading.
# - Size files for parallelism; do not load with row-by-row `INSERT`.
# - `ON_ERROR` decides what a bad row does; `CONTINUE` without reconciliation means silent data loss.
# - The files this notebook wrote to `exports/` are ready for a real trial account: upload them and check your totals against `control`.
#
# **Go deeper:** docs.snowflake.com, "Loading Data into Snowflake" (stages, COPY INTO, Snowpipe) and the pyarrow Parquet documentation (arrow.apache.org/docs/python).
#
# **Next:** [Cost, performance and security basics](05_cost_performance_and_security_basics.ipynb)
