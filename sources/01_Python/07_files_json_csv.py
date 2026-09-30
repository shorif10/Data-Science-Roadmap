# %% [markdown]
# # Files, JSON and CSV
#
# **Why this matters.** Data arrives as files and leaves as files: CSV exports, JSON from APIs, logs, configs. Reading them correctly (encoding, newlines, types) and writing them safely
# (no half-written output, no lost commas) is a daily task. pandas hides most of this, but when a file is odd or huge, or you are writing a small tool, you need the standard library underneath.
#
# **You will learn**
# - open files safely with `with`, explicit encodings and modes; use `pathlib` for paths
# - read and write CSV with `csv.reader/DictReader/DictWriter` and understand quoting
# - read and write JSON, know exactly which Python values survive the round trip, and serialise dates
# - write and read JSON Lines for big or streaming data
#
# **Prerequisites:** [Comprehensions and generators](06_comprehensions_and_generators.ipynb).
#
# > Data used: `datasets/orders.csv` and `datasets/products.csv` (synthetic; see `docs/DATASETS.md`). Everything we write goes to a temp folder that is deleted at the end.

# %%
import csv
import json
import tempfile
from datetime import date, datetime
from pathlib import Path

DATA = Path("../datasets")
tmp = tempfile.TemporaryDirectory()
work = Path(tmp.name)
print("scratch folder exists:", work.exists())

# %% [markdown]
# ## 1. Opening files
# `open(path, mode, encoding=...)` returns a file object. Use it inside a **`with` block**, which closes the file even if an error occurs (context managers are explained in a later notebook).
#
# | Mode | Meaning |
# |---|---|
# | `"r"` (default) | read text; error if missing |
# | `"w"` | write text, **truncating** an existing file |
# | `"a"` | append |
# | `"x"` | create, fail if it exists (a safe way not to overwrite) |
# | `"rb"`, `"wb"` | bytes instead of text |

# %%
notes = work / "notes.txt"
with open(notes, "w", encoding="utf-8") as f:
    f.write("first line\n")
    f.write("second line, with an accent: café\n")
    f.writelines(["third\n", "fourth\n"])

with open(notes, encoding="utf-8") as f:
    print(f.readline().rstrip())        # one line
    rest = f.read()                     # everything left
print(repr(rest))
print("file is closed after the with block:", f.closed)

# %% [markdown]
# Iterating over the file object reads **one line at a time**, so it works for files bigger than memory. `Path` objects have one-line helpers for small files:

# %%
with open(notes, encoding="utf-8") as f:
    n_chars = sum(len(line) for line in f)      # lazy, constant memory
print("characters:", n_chars)

print(notes.read_text(encoding="utf-8").splitlines()[:2])
(work / "greeting.txt").write_text("hello\n", encoding="utf-8")       # overwrites; returns the number of characters written
print(sorted(p.name for p in work.iterdir()))

# %% [markdown]
# Rule: **always pass `encoding="utf-8"`** for text files. Without it Python uses the platform default (often `cp1252` on Windows), and the same code will corrupt or crash on another machine.

# %%
cafe_bytes = "café".encode("utf-8")
(work / "latin.txt").write_bytes(cafe_bytes)
print("utf-8 read :", (work / "latin.txt").read_text(encoding="utf-8"))
print("latin-1 read:", (work / "latin.txt").read_text(encoding="latin-1"), "<- mojibake, no error")

# %% [markdown]
# ## 2. CSV
# CSV looks trivial and is not: fields may contain commas, quotes and newlines, so **do not** use `line.split(",")` on real data. The `csv` module handles the quoting rules.
# Always open CSV files with `newline=""` so the module controls line endings.

# %%
with open(DATA / "products.csv", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    first = next(reader)
print(header)
print(first, "<- every field is a str")

with open(DATA / "products.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))             # each row is a dict keyed by the header
print(rows[0]["product_name"], rows[0]["unit_price"], type(rows[0]["unit_price"]))

# %% [markdown]
# `DictReader` is usually the better default: you address columns by name, so a reordered column does not silently break your code. Now writing, including a value that contains a comma and a quote:

# %%
people = [
    {"name": "Smith, John", "quote": 'He said "hello"', "age": 41},
    {"name": "Ana", "quote": "line one\nline two", "age": 29},
]
out = work / "people.csv"
with open(out, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "quote", "age"])
    writer.writeheader()
    writer.writerows(people)
print(out.read_text(encoding="utf-8"))

with open(out, encoding="utf-8", newline="") as f:
    back = list(csv.DictReader(f))
print(back == [{k: str(v) for k, v in p.items()} for p in people], "<- round trip equal, but ages are strings now")

# %% [markdown]
# *What just happened?* The writer quoted fields containing commas, quotes or newlines and doubled the inner quotes; the reader undid it. Two things do **not** round-trip: types (everything comes back as `str`,
# so convert explicitly) and `None` (written as an empty string). `csv` has no notion of column types; pandas will infer them for you, and sometimes get them wrong (see the pandas notebooks).

# %% [markdown]
# ## 3. JSON
# JSON is the lingua franca of APIs and config files. `json.dumps(obj)` -> string, `json.loads(text)` -> object; `json.dump(obj, file)` / `json.load(file)` do the same with files.

# %%
record = {"order_id": 7, "customer": "Ana", "items": [{"sku": "A1", "qty": 2}], "paid": True, "coupon": None, "total": 59.9}
text = json.dumps(record, indent=2)
print(text)
print(json.loads(text) == record)

# %% [markdown]
# JSON has only objects, arrays, strings, numbers, `true/false`, `null`. So Python types map like this, and **some do not survive the round trip**:

# %%
tricky = {"tuple": (1, 2), "int_key": {1: "a"}}
round_tripped = json.loads(json.dumps(tricky))
print("tuple becomes list:", round_tripped["tuple"])
print("int keys become strings:", round_tripped["int_key"])

for bad in ({"when": date(2024, 1, 31)}, {"s": {1, 2}}):
    try:
        json.dumps(bad)
    except TypeError as e:
        print("TypeError:", e)

print(json.dumps({"x": float("nan")}), "<- NaN is not valid JSON; some parsers reject it")

# %% [markdown]
# For unsupported types give `dumps` a `default=` function that converts them, and parse them back yourself on load. ISO 8601 strings are the safe way to store dates:

# %%
def encode_extra(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__}")

event = {"when": datetime(2024, 1, 31, 9, 30), "day": date(2024, 1, 31), "tags": {"b", "a"}}
s = json.dumps(event, default=encode_extra)
print(s)
loaded = json.loads(s)
print(datetime.fromisoformat(loaded["when"]) == event["when"])

# %% [markdown]
# **Never** load data with `pickle` from a source you do not trust: unpickling can execute arbitrary code. JSON is data-only, which is why it is the safe interchange format.

# %% [markdown]
# ## 4. JSON Lines (one JSON object per line)
# A single big JSON array must be read completely before you can use it. **JSON Lines** (`.jsonl`) stores one object per line, so you can append, stream and process line by line: the standard shape for logs and ML datasets.

# %%
jl = work / "orders_sample.jsonl"
with open(DATA / "orders.csv", encoding="utf-8", newline="") as src, open(jl, "w", encoding="utf-8") as dst:
    for i, row in enumerate(csv.DictReader(src)):
        if i >= 1000:
            break
        row["order_id"] = int(row["order_id"])
        dst.write(json.dumps(row) + "\n")

with open(jl, encoding="utf-8") as f:
    first_two = [json.loads(next(f)) for _ in range(2)]
print(first_two)
with open(jl, encoding="utf-8") as f:
    n_returned = sum(1 for line in f if json.loads(line)["status"] == "returned")
print("returned among the first 1000:", n_returned)

# %% [markdown]
# ## 5. Safe writing
# If your program crashes halfway through writing `results.csv`, you are left with a truncated file that looks valid. The standard defence is to write to a **temporary file in the same folder and then rename it**;
# rename (`Path.replace`) is atomic on the same filesystem, so readers see either the old file or the complete new one.

# %%
def write_json_atomic(path, obj):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp_path.replace(path)                     # atomic swap

target = work / "result.json"
write_json_atomic(target, {"accuracy": 0.9})
write_json_atomic(target, {"accuracy": 0.95})
print(json.loads(target.read_text(encoding="utf-8")), "| leftovers:", [p.name for p in work.glob("*.tmp")])

# %% [markdown]
# ## Common mistakes
#
# 1. **No `encoding=`** on `open`: works on your laptop, garbles text elsewhere.
# 2. **`line.split(",")` on real CSV** breaks on quoted commas. Use `csv`.
# 3. **Forgetting `newline=""`** with `csv`: blank rows appear between records on Windows.
# 4. **Mode `"w"` on a file you meant to keep:** it truncates immediately, before you write anything. Use `"x"` or `"a"` when appropriate.
# 5. **Assuming JSON round-trips Python objects:** tuples become lists, keys become strings, dates and sets are errors.
# 6. **Reading a huge file with `.read()` / `readlines()`** instead of iterating.
# 7. **Not closing files:** use `with`.

# %%
# Mistake 2 in action
line = '"Smith, John","He said ""hello"""'
print("split  ->", line.split(","))
print("csv    ->", next(csv.reader([line])))

# %% [markdown]
# ## Interview corner
#
# **1. Why use <code>with open(...)</code>?**
# <details><summary>Answer</summary>
# The context manager guarantees the file is closed when the block exits, even on an exception, which prevents leaked file handles and unflushed writes.
# </details>
#
# **2. How do you read a file too large for memory?**
# <details><summary>Answer</summary>
# Iterate over the file object (line by line), or read fixed-size chunks, and keep only aggregates or write results as you go. For tabular data use <code>csv.reader</code> or pandas with <code>chunksize</code>.
# </details>
#
# **3. What does JSON not preserve from Python?**
# <details><summary>Answer</summary>
# Tuples (become lists), non-string dict keys (become strings), sets, dates, decimals, custom objects (errors unless you provide <code>default=</code>), and NaN/Infinity are non-standard. Dates are stored as ISO strings by convention.
# </details>
#
# **4. CSV versus JSON versus Parquet: when would you choose each?**
# <details><summary>Answer</summary>
# CSV: simple flat tables, universal but untyped. JSON: nested, self-describing records and APIs. Parquet: large analytical tables, columnar, typed and compressed, best for data pipelines and repeated queries.
# </details>

# %% [markdown]
# ## Exercises
# The temp folder `work` exists until the last cell of the notebook.

# %% [markdown]
# ### Exercise 1: Export a filtered CSV
# Write `delivered_store.csv` into `work` containing only the orders with status `delivered` and channel `store`, with all five original columns (header included), using `csv.DictWriter`.
# Store the number of rows you wrote (excluding the header) in `n_written`.

# %% tags=["starter"]
n_written = ...

# %%
with open(DATA / "orders.csv", encoding="utf-8", newline="") as f:
    src = list(csv.DictReader(f))
want_rows = [r for r in src if r["status"] == "delivered" and r["channel"] == "store"]
with open(work / "delivered_store.csv", encoding="utf-8", newline="") as f:
    got = list(csv.DictReader(f))
assert got == want_rows, "file content differs from the expected filtered rows (columns and order must match orders.csv)"
assert n_written == len(want_rows), f"n_written should be {len(want_rows)}"
print("Exercise 1 passed:", n_written, "rows written")

# %% tags=["solution"]
with open(DATA / "orders.csv", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    keep = [r for r in reader if r["status"] == "delivered" and r["channel"] == "store"]
with open(work / "delivered_store.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(keep)
n_written = len(keep)

# %% [markdown]
# ### Exercise 2: Round-trip an awkward record
# Write `save_record(path, record)` and `load_record(path)` using JSON so that this record survives **exactly** (dates as dates, sets as sets), given the fields `when` (a `datetime`), `tags` (a `set`
# of str) and `amount` (float). Hint: store the extras as ISO strings/sorted lists and convert back in `load_record`; you know the field names.

# %% tags=["starter"]
def save_record(path, record):
    ...

def load_record(path):
    ...

# %%
rec = {"when": datetime(2024, 5, 17, 13, 45, 10), "tags": {"vip", "web"}, "amount": 129.5}
p = work / "record.json"
save_record(p, rec)
assert json.loads(p.read_text(encoding="utf-8")), "the file must contain valid JSON"
assert load_record(p) == rec, f"round trip failed: {load_record(p)}"
assert isinstance(load_record(p)["when"], datetime) and isinstance(load_record(p)["tags"], set)
print("Exercise 2 passed")

# %% tags=["solution"]
def save_record(path, record):
    payload = {"when": record["when"].isoformat(), "tags": sorted(record["tags"]), "amount": record["amount"]}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")

def load_record(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"when": datetime.fromisoformat(raw["when"]), "tags": set(raw["tags"]), "amount": raw["amount"]}

# %% [markdown]
# ### Exercise 3: JSON Lines to totals
# Iterate over the file `jl` (created above, 1000 orders) **line by line** and store `channel_counts`, a dict of orders per channel among those 1000 records.

# %% tags=["starter"]
channel_counts = ...

# %%
want = {}
with open(DATA / "orders.csv", encoding="utf-8", newline="") as f:
    for i, r in enumerate(csv.DictReader(f)):
        if i >= 1000:
            break
        want[r["channel"]] = want.get(r["channel"], 0) + 1
assert channel_counts == want, f"expected {want}, got {channel_counts}"
assert sum(channel_counts.values()) == 1000
print("Exercise 3 passed:", channel_counts)

# %% tags=["solution"]
channel_counts = {}
with open(jl, encoding="utf-8") as f:
    for line in f:
        ch = json.loads(line)["channel"]
        channel_counts[ch] = channel_counts.get(ch, 0) + 1

# %% [markdown]
# ### Exercise 4: Commas in the data
# A junior wrote `bad_write(path, rows)` that joins fields with commas, corrupting names like `"Smith, John"`. Write `good_write(path, rows)` (rows are lists of values) with the `csv` module and check that
# reading with `csv.reader` gives back exactly the strings you wrote.

# %%
def bad_write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

# %% tags=["starter"]
def good_write(path, rows):
    ...

# %%
rows = [["id", "name", "note"], [1, "Smith, John", 'said "hi"'], [2, "Ana", "two\nlines"]]
bp, gp = work / "bad.csv", work / "good.csv"
bad_write(bp, rows)
good_write(gp, rows)
with open(bp, encoding="utf-8", newline="") as f:
    bad_back = list(csv.reader(f))
with open(gp, encoding="utf-8", newline="") as f:
    good_back = list(csv.reader(f))
assert bad_back != [[str(c) for c in r] for r in rows], "(sanity) the naive writer should corrupt the data"
assert good_back == [[str(c) for c in r] for r in rows], f"csv round trip failed: {good_back}"
print("Exercise 4 passed")

# %% tags=["solution"]
def good_write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)

# %% [markdown]
# ### Exercise 5 (open-ended): Products by category as JSON
# Read `products.csv` and write `products_by_category.json` into `work`: a JSON object mapping each category to a list of `{"name": ..., "price": <float>}` dicts, sorted by price descending.
# Load it back and print the most expensive product per category. Note which of your choices (float vs string prices, key order) another program reading the file might trip over.

# %%
tmp.cleanup()
print("scratch folder removed:", not work.exists())

# %% [markdown]
# ## Recap
# - `with open(path, mode, encoding="utf-8")`; iterate lines lazily; `pathlib` for paths and small reads/writes.
# - Use the `csv` module (`newline=""`), prefer `DictReader`/`DictWriter`; all CSV values come back as strings.
# - JSON round-trips only basic types: convert dates/sets explicitly (`default=`, ISO strings); tuples become lists, keys become strings.
# - JSON Lines for big or streaming records; write via a temp file + atomic rename to avoid half-written outputs.
# - Never unpickle untrusted data.
#
# **Go deeper:** the docs for `csv`, `json`, `pathlib` and *Reading and Writing Files* in the Python tutorial (docs.python.org/3).
#
# **Next:** [Errors and debugging](08_errors_and_debugging.ipynb)
