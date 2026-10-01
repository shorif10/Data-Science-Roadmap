# %% [markdown]
# # pandas Performance and Best Practices
#
# **Why this matters.** The notebook that ran in seconds on your laptop sample crawls (or runs out of memory) on the real table. Knowing *where* pandas is fast, where it is slow and
# how to make tables small is what separates a data analyst who waits from one who ships. This notebook also covers the pandas 3 behaviour that changes how you write code:
# **copy-on-write**, which is why old chained-assignment habits no longer work.
#
# **You will learn**
# - measure memory with `memory_usage(deep=True)` and shrink a table with the right dtypes
# - compare vectorised code, `apply`, `itertuples` and `iterrows` with a fair benchmark
# - write readable method chains with `assign`, `query` and `pipe`
# - compare CSV and Parquet (size, speed, dtype fidelity) and process a big file in chunks
# - explain copy-on-write and fix chained-assignment bugs
#
# **Prerequisites:** [Time series in pandas](07_time_series_in_pandas.ipynb).
#
# > Data used: the shop tables. To have something big enough to time, we build a 300,000-row table by resampling order lines with a fixed seed. It is only for benchmarking: the rows
# > are repeated, not new data.

# %%
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv")
order_items = pd.read_csv(DATA / "order_items.csv")

lines = (
    order_items.merge(products, on="product_id", validate="many_to_one")
    .merge(orders, on="order_id", validate="many_to_one")
    .merge(customers[["customer_id", "country"]], on="customer_id", validate="many_to_one")
)
big = lines.sample(n=300_000, replace=True, random_state=42).reset_index(drop=True)      # benchmarking table
print("rows:", len(lines), "->", len(big))


def best_of(fn, repeats=3):
    """Fastest wall-clock time (seconds) of `repeats` calls, plus the last result."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


# %% [markdown]
# ## 1. Memory: measure first
# `df.info()` gives an estimate; `memory_usage(deep=True)` counts the real size of text columns too. Look at which columns dominate.

# %%
mem = big.memory_usage(deep=True)
print((mem / 1e6).round(1).sort_values(ascending=False).to_string())
print(f"total: {mem.sum() / 1e6:.1f} MB")

# %% [markdown]
# Text columns dominate. Low-cardinality text (few distinct values repeated many times) is the ideal candidate for `category`, and integers rarely need 64 bits.

# %%
slim = big.assign(
    category=big["category"].astype("category"),
    status=big["status"].astype("category"),
    channel=big["channel"].astype("category"),
    country=big["country"].astype("category"),
    order_id=pd.to_numeric(big["order_id"], downcast="unsigned"),
    item_id=pd.to_numeric(big["item_id"], downcast="unsigned"),
    product_id=pd.to_numeric(big["product_id"], downcast="unsigned"),
    customer_id=pd.to_numeric(big["customer_id"], downcast="unsigned"),
    quantity=big["quantity"].astype("uint8"),
    unit_price=big["unit_price"].astype("float32"),
    unit_cost=big["unit_cost"].astype("float32"),
)
slim_mem = slim.memory_usage(deep=True).sum()
print(f"before: {mem.sum() / 1e6:.1f} MB | after: {slim_mem / 1e6:.1f} MB | saved: {1 - slim_mem / mem.sum():.0%}")
print(slim.dtypes.astype(str).to_dict())

# %% [markdown]
# *What just happened?* Category columns store each distinct string once plus small integer codes, and narrower numeric types shrink the numbers. Two costs to remember:
# (1) `float32` has about 7 significant digits: fine for a price, wrong for large money totals; (2) a category column with nearly unique values (an id or a free-text field) gets **bigger**, not smaller.
# The saving is only worth it if analysis results are unchanged: confirm.

# %%
rev_big = (big["quantity"] * big["unit_price"]).sum()
rev_slim = (slim["quantity"].astype("int64") * slim["unit_price"].astype("float64")).sum()
print(f"revenue big: {rev_big:,.2f} | slim: {rev_slim:,.2f} | relative difference: {abs(rev_big - rev_slim) / rev_big:.1e}")

# %% [markdown]
# Note the explicit casts in the check: arithmetic on narrow dtypes can wrap around (`uint8` times `uint8`) or lose precision (`float32` sums), the NumPy dtype traps from the NumPy module applying to pandas too.
# **Reading only what you need** is the cheapest saving of all: `pd.read_csv(path, usecols=[...], dtype={...})` never loads the columns you drop.

# %% [markdown]
# ## 2. Vectorised vs `apply` vs loops: a fair benchmark
# Task: revenue per line = `quantity * unit_price`. Same result asserted in each version, best of 3, inputs prepared outside the timer.
# Row loops are so slow we time them on a 20,000-row slice and scale the comparison per row.

# %%
sub = big.head(20_000)

def v_vector():
    return (sub["quantity"] * sub["unit_price"]).to_numpy()

def v_apply_rows():
    return sub.apply(lambda r: r["quantity"] * r["unit_price"], axis=1).to_numpy()

def v_itertuples():
    return np.array([row.quantity * row.unit_price for row in sub.itertuples()])

def v_iterrows():
    return np.array([r["quantity"] * r["unit_price"] for _, r in sub.iterrows()])

results = {}
for name, fn in [("vectorised", v_vector), ("apply(axis=1)", v_apply_rows), ("itertuples", v_itertuples), ("iterrows", v_iterrows)]:
    t, out = best_of(fn, repeats=3)
    results[name] = (t, out)
ref = results["vectorised"][1]
for name, (t, out) in results.items():
    assert np.allclose(out, ref), f"{name} disagrees"
base = results["vectorised"][0]
for name, (t, _) in results.items():
    print(f"{name:14s} {t * 1000:9.2f} ms   {t / base:8.0f}x the vectorised time")

# %% [markdown]
# Vectorised code wins by orders of magnitude. Among the slow options the ranking can change with the number of columns and the pandas version (compare `itertuples` and `apply` in your own run),
# but `iterrows` is reliably the worst because it builds a Series for every row. Guidelines: vectorise; if you must loop, prefer `itertuples` over `iterrows`; use `apply` only for logic you
# cannot express on whole columns. Another classic: **growing a DataFrame inside a loop** copies everything every time. Collect pieces in a list and `concat` once.

# %%
pieces = [pd.DataFrame({"i": [i], "sq": [i * i]}) for i in range(300)]

def grow_in_loop():
    df = pd.DataFrame({"i": [], "sq": []})
    for p in pieces:
        df = pd.concat([df, p], ignore_index=True)
    return df

def concat_once():
    return pd.concat(pieces, ignore_index=True)

t_loop, a = best_of(grow_in_loop, 3)
t_once, b = best_of(concat_once, 3)
assert a.shape == b.shape and (a["sq"].to_numpy() == b["sq"].to_numpy()).all()
print(f"concat in a loop: {t_loop * 1000:.1f} ms | concat once: {t_once * 1000:.1f} ms | ratio {t_loop / t_once:.0f}x for 300 pieces")

# %% [markdown]
# ## 3. Method chaining: readable pipelines
# A **method chain** applies one transformation after another without naming every intermediate frame. Combined with `assign`, `query`, `pipe` and `groupby`, it reads like a recipe
# and avoids leftover temporary variables. Parenthesise the chain so you can break lines and comment.

# %%
def check_rows(df, label):
    """A tiny debugging hook for .pipe(): print the shape and return the frame unchanged."""
    print(f"  [{label}] {df.shape}")
    return df

top_categories = (
    lines
    .pipe(check_rows, "all lines")
    .query("status != 'cancelled'")                       # boolean filter written as a string
    .pipe(check_rows, "without cancelled")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"],
            margin=lambda d: d["quantity"] * (d["unit_price"] - d["unit_cost"]))
    .groupby("category", as_index=False)
    .agg(revenue=("revenue", "sum"), margin=("margin", "sum"))
    .assign(margin_pct=lambda d: d["margin"] / d["revenue"] * 100)
    .sort_values("revenue", ascending=False)
    .round(1)
)
top_categories

# %% [markdown]
# `pipe(func, ...)` lets your own function sit inside the chain; the `check_rows` hook shows how to *watch* the row count without breaking the chain, which is exactly the join/filter discipline
# from earlier notebooks. Trade-off: very long chains are hard to debug; break a chain into named steps when it stops fitting on one screen.

# %% [markdown]
# ## 4. CSV vs Parquet
# CSV is text: slow to parse, larger, and it **forgets dtypes** (dates come back as text, categories as strings). Parquet is a compressed columnar binary format that keeps dtypes and lets you read a subset of columns.
# It needs `pyarrow` (installed here). We write into a temporary directory and clean up automatically.

# %%
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    src = slim.assign(order_date=pd.to_datetime(slim["order_date"]))
    t_wcsv, _ = best_of(lambda: src.to_csv(tmp / "lines.csv", index=False), 1)
    t_wpq, _ = best_of(lambda: src.to_parquet(tmp / "lines.parquet", index=False), 1)
    t_rcsv, from_csv = best_of(lambda: pd.read_csv(tmp / "lines.csv"), 1)
    t_rpq, from_pq = best_of(lambda: pd.read_parquet(tmp / "lines.parquet"), 1)
    t_rpq_cols, _ = best_of(lambda: pd.read_parquet(tmp / "lines.parquet", columns=["category", "quantity"]), 1)
    size_csv, size_pq = (tmp / "lines.csv").stat().st_size, (tmp / "lines.parquet").stat().st_size
    print(f"file size : csv {size_csv / 1e6:6.1f} MB | parquet {size_pq / 1e6:6.1f} MB")
    print(f"write time: csv {t_wcsv:6.2f} s  | parquet {t_wpq:6.2f} s")
    print(f"read time : csv {t_rcsv:6.2f} s  | parquet {t_rpq:6.2f} s | parquet, 2 columns only {t_rpq_cols:6.3f} s")
    print("dtype of order_date after round trip -> csv:", from_csv["order_date"].dtype, "| parquet:", from_pq["order_date"].dtype)
    print("dtype of category    after round trip -> csv:", from_csv["category"].dtype, "| parquet:", from_pq["category"].dtype)
print("temp directory removed:", not tmp.exists())

# %% [markdown]
# Results are machine dependent, but the pattern is general: Parquet is smaller, faster to read, keeps types, and supports column selection. Use CSV for interchange with humans and tools that require it, Parquet for
# data you will read again. **Chunked reading** handles files larger than memory: process a slice at a time and combine partial results.

# %%
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "big.csv"
    big[["category", "quantity", "unit_price"]].to_csv(path, index=False)
    total = 0.0
    n_chunks = 0
    for chunk in pd.read_csv(path, chunksize=100_000):
        total += (chunk["quantity"] * chunk["unit_price"]).sum()
        n_chunks += 1
print(f"{n_chunks} chunks, chunked total: {total:,.2f} | in-memory total: {rev_big:,.2f} | equal: {np.isclose(total, rev_big)}")

# %% [markdown]
# Chunking works for aggregations that can be combined (sums, counts, min/max), but not directly for a median or a join; for those use a database or DuckDB (see the SQL module).

# %% [markdown]
# ## 5. Copy-on-write (pandas 3): why chained assignment no longer works
#
# In pandas 3, **any result that looks like a new object behaves as an independent copy** (copy-on-write, "CoW"). Data is shared lazily for speed and copied only at the moment one of the objects is modified.
# The practical consequences:
#
# 1. Modifying a derived object (a column you took out, a filtered frame, a result of a method) **never changes the original**.
# 2. So the old "chained assignment" pattern `df[mask]["col"] = value` modifies a temporary and **does nothing** to `df`. pandas 3 warns you with `ChainedAssignmentError`.
# 3. The `SettingWithCopyWarning` of old pandas is gone; a defensive `.copy()` is rarely needed.
# 4. To modify a frame, assign to it directly, in a single step: `df.loc[mask, "col"] = value`.

# %%
df = pd.DataFrame({"name": ["a", "b", "c"], "score": [10, 200, 30]})

# (a) chained assignment: the intermediate df[df.score > 100] is a temporary copy
df[df["score"] > 100]["score"] = 100
print("after chained assignment:", df["score"].tolist(), "(unchanged: the bug)")

# (b) the correct, single-step assignment
df.loc[df["score"] > 100, "score"] = 100
print("after .loc assignment   :", df["score"].tolist())

# %% [markdown]
# Independence of derived objects, demonstrated:

# %%
original = pd.DataFrame({"x": [1, 2, 3]})
col = original["x"]
col.iloc[0] = 999                      # modifies `col` only; pandas copies on this write
print("original:", original["x"].tolist(), "| col:", col.tolist())

alias = original                       # a plain assignment of the SAME object: still one object
alias.loc[0, "x"] = -1
print("alias is the same object, so original sees it:", original["x"].tolist())

def double_scores(frame):
    frame = frame.assign(x=frame["x"] * 2)      # assign returns a new frame; the argument is untouched
    return frame

out = double_scores(original)
print("function result:", out["x"].tolist(), "| argument still:", original["x"].tolist())

# %% [markdown]
# So two things share memory *lazily* but behave as separate objects. You can still see the sharing with NumPy, until the first write:

# %%
a = pd.DataFrame({"x": np.arange(5)})
b = a[["x"]]                                              # column selection: a lazy copy
print("shares memory before any write:", np.shares_memory(a["x"].to_numpy(), b["x"].to_numpy()))
b.loc[0, "x"] = 100
print("after writing to b:", np.shares_memory(a["x"].to_numpy(), b["x"].to_numpy()), "| a:", a["x"].tolist(), "| b:", b["x"].tolist())

# %% [markdown]
# *What just happened?* The two frames shared a buffer until `b` was written, at which moment pandas copied. You get the speed of no-copy until you need independence, and correctness guaranteed by the rules above.
# One more habit CoW encourages: avoid `inplace=True` (it rarely saves memory and defeats chaining); assign the result: `df = df.drop(columns="x")`.

# %% [markdown]
# ## Common mistakes
#
# 1. **Optimising before measuring.** Profile (`best_of`, `%timeit`, `memory_usage(deep=True)`) first; most slowness is one or two operations.
# 2. **`iterrows` / `apply(axis=1)` for column arithmetic.**
# 3. **`category` on high-cardinality columns**, or `float32` for money totals.
# 4. **Growing a DataFrame in a loop** with `concat` or `append`-style code.
# 5. **Chained assignment** (`df[mask]["col"] = v`): silently ineffective in pandas 3.
# 6. **`inplace=True` everywhere.** It does not guarantee less memory and blocks method chaining.
# 7. **Assuming a filtered frame is a view** you can write through, or writing defensive `.copy()` calls everywhere out of old habit.
# 8. **Storing intermediate data as CSV** and then losing date and category dtypes.

# %% [markdown]
# ## Interview corner
#
# **1. How would you reduce the memory of a large DataFrame?**
# <details><summary>Answer</summary>
# Measure with <code>memory_usage(deep=True)</code>, then convert low-cardinality strings to <code>category</code>, downcast numerics (<code>pd.to_numeric(downcast=...)</code>, <code>float32</code> where precision allows),
# read only needed columns with <code>usecols</code>, and consider Parquet or chunked processing. Verify results are unchanged after the conversion.
# </details>
#
# **2. Why is `iterrows` slow and what do you use instead?**
# <details><summary>Answer</summary>
# It creates a Series object per row in Python. Use vectorised column operations; if a loop is unavoidable, <code>itertuples</code> is much cheaper; for custom logic consider <code>numba</code> or restructuring.
# </details>
#
# **3. What is copy-on-write in pandas 3 and what does it break?**
# <details><summary>Answer</summary>
# Derived objects behave as independent copies (data is shared lazily and copied on first write), so modifying a slice or column never mutates the parent. Code relying on chained assignment
# (<code>df[mask]["col"] = v</code>) stops working; use <code>df.loc[mask, "col"] = v</code>.
# </details>
#
# **4. CSV versus Parquet?**
# <details><summary>Answer</summary>
# CSV is human-readable, universal, row-oriented text with no types. Parquet is a compressed columnar binary format that preserves dtypes, is smaller and faster to read, and supports reading specific columns and predicate pushdown in some engines.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Put the table on a diet
# Create `diet` from `big`: convert `channel` and `status` to `category` and `quantity` to `uint8`, keep all other columns unchanged.
# Store `saved_pct`: the percentage by which `diet`'s `memory_usage(deep=True).sum()` is lower than `big`'s.

# %% tags=["starter"]
diet = ...
saved_pct = ...

# %%
assert isinstance(diet["channel"].dtype, pd.CategoricalDtype) and isinstance(diet["status"].dtype, pd.CategoricalDtype), "channel and status must be category"
assert diet["quantity"].dtype == np.uint8, "quantity must be uint8"
assert diet.shape == big.shape and (diet["quantity"].astype("int64") == big["quantity"]).all() and (diet["channel"].astype(str) == big["channel"]).all(), "values must not change"
exp = (1 - diet.memory_usage(deep=True).sum() / big.memory_usage(deep=True).sum()) * 100
assert abs(saved_pct - exp) < 1e-6 and saved_pct > 0, f"saved_pct should be {exp:.2f}"
print(f"Exercise 1 passed (saved {saved_pct:.1f}%)")

# %% tags=["solution"]
diet = big.assign(
    channel=big["channel"].astype("category"),
    status=big["status"].astype("category"),
    quantity=big["quantity"].astype("uint8"),
)
saved_pct = (1 - diet.memory_usage(deep=True).sum() / big.memory_usage(deep=True).sum()) * 100

# %% [markdown]
# ### Exercise 2: Vectorise the slow function
# The function `margin_row` computes the margin percentage of one line. Below, `slow_margin` applies it row by row. Produce `fast_margin` (a Series) with the same values, using only column arithmetic, and store `speedup`
# (slow time divided by fast time, from `best_of`). The check requires equal values and `speedup > 1`.

# %%
def margin_row(r):
    return (r["unit_price"] - r["unit_cost"]) / r["unit_price"] * 100

work = big.head(10_000)
t_slow, slow_margin = best_of(lambda: work.apply(margin_row, axis=1), repeats=2)

# %% tags=["starter"]
fast_margin = ...
speedup = ...

# %%
assert np.allclose(fast_margin.to_numpy(), slow_margin.to_numpy()), "values must equal the row-wise version"
assert speedup > 1, "the vectorised version should be faster"
print(f"Exercise 2 passed (about {speedup:.0f}x faster)")

# %% tags=["solution"]
def fast():
    return (work["unit_price"] - work["unit_cost"]) / work["unit_price"] * 100

t_fast, fast_margin = best_of(fast, repeats=5)
speedup = t_slow / t_fast

# %% [markdown]
# ### Exercise 3: A method chain
# Write one chain starting from `lines` that produces `country_rev`: revenue (`quantity * unit_price`) per `country`, using only **delivered** orders, as a Series sorted from highest to lowest.
# Use `query` (or a boolean filter), `assign`, `groupby`, `sort_values`. No temporary variables inside the chain.

# %% tags=["starter"]
country_rev = ...

# %%
d = lines[lines["status"] == "delivered"]
exp = (d["quantity"] * d["unit_price"]).groupby(d["country"]).sum().sort_values(ascending=False)
assert isinstance(country_rev, pd.Series), "country_rev must be a Series"
assert list(country_rev.index) == list(exp.index), "sorted from highest to lowest revenue"
assert np.allclose(country_rev.to_numpy(), exp.to_numpy()), "revenue per country (delivered only)"
print(f"Exercise 3 passed (top: {country_rev.index[0]})")

# %% tags=["solution"]
country_rev = (
    lines
    .query("status == 'delivered'")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"])
    .groupby("country")["revenue"].sum()
    .sort_values(ascending=False)
)

# %% [markdown]
# ### Exercise 4: Predict what copy-on-write does
# For each snippet, decide whether it changes the parent frame `p` and store `True` or `False` in the dict `changes`.
# `"a"`: `p[p["v"] > 1]["v"] = 0`  |  `"b"`: `p.loc[p["v"] > 1, "v"] = 0`  |  `"c"`: `s = p["v"]; s.iloc[0] = 0`  |  `"d"`: `q = p; q.loc[0, "v"] = 0`.
# Think first; the check runs the snippets for real.

# %% tags=["starter"]
changes = {"a": ..., "b": ..., "c": ..., "d": ...}

# %%
import warnings

def run(snippet):
    p = pd.DataFrame({"v": [1, 2, 3]})
    before = p["v"].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if snippet == "a":
            p[p["v"] > 1]["v"] = 0
        elif snippet == "b":
            p.loc[p["v"] > 1, "v"] = 0
        elif snippet == "c":
            s = p["v"]
            s.iloc[0] = 0
        elif snippet == "d":
            q = p
            q.loc[0, "v"] = 0
    return p["v"].tolist() != before

for k in "abcd":
    assert changes[k] is run(k) or changes[k] == run(k), f"snippet {k}: the parent {'does' if run(k) else 'does NOT'} change"
print("Exercise 4 passed")

# %% tags=["solution"]
changes = {"a": False, "b": True, "c": False, "d": True}

# %% [markdown]
# ### Exercise 5 (open-ended): Your file-format policy
# Using the timings above (or re-run them on `big`), write four sentences in a markdown cell: when would you store a table as CSV, when as Parquet, what dtype information each loses or keeps, and what you would do with a 20 GB
# CSV that does not fit in memory (name at least two options).

# %% [markdown]
# ## Recap
# - Measure first: `memory_usage(deep=True)`, timing helpers. Shrink with `category`, downcast numerics and `usecols`; verify results are unchanged.
# - Vectorise; `itertuples` beats `iterrows`; never grow a frame inside a loop.
# - Method chains with `assign`, `query`, `pipe` read like recipes; break them when they get long.
# - Parquet keeps dtypes and is smaller and faster than CSV; chunked reading handles big files for combinable aggregates.
# - pandas 3 copy-on-write: derived objects are independent; chained assignment does nothing; assign with `df.loc[mask, col] = value`.
#
# **Go deeper:** pandas user guide "Copy-on-Write (CoW)", "Scaling to large datasets" and "Enhancing performance" (pandas.pydata.org/docs/user_guide); Apache Arrow / Parquet documentation at arrow.apache.org.
#
# **Next:** [EDA project: the e-commerce shop](09_eda_project_ecommerce.ipynb)
