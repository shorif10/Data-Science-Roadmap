# %% [markdown]
# # Comprehensions and Generators
#
# **Why this matters.** Comprehensions are how idiomatic Python filters and transforms data in one readable line. Generators are how it processes files or streams **larger than memory**
# without loading them. Together they are the pure-Python ancestors of pandas operations and of every data pipeline: read lazily, transform, aggregate.
#
# **You will learn**
# - list, dict and set comprehensions, with filters, and where the `if` goes
# - generator expressions and generator functions (`yield`), and why they are lazy
# - the one-shot nature of generators (exhaustion) and how it bites
# - building a small lazy pipeline over a CSV file
#
# **Prerequisites:** [Functions](05_functions.ipynb).
#
# > Data used: `datasets/orders.csv` and `datasets/products.csv` (synthetic; see `docs/DATASETS.md`).

# %%
import csv
import sys
from itertools import islice
from pathlib import Path

DATA = Path("../datasets")
with open(DATA / "products.csv", encoding="utf-8", newline="") as f:
    products = list(csv.DictReader(f))
print(len(products), "products")

# %% [markdown]
# ## 1. List comprehensions
# A comprehension is a loop that builds a list, written as an expression: `[expression for item in iterable if condition]`.

# %%
# the loop version
expensive_names = []
for p in products:
    if float(p["unit_price"]) > 700:
        expensive_names.append(p["product_name"])

# the comprehension version: same result, one expression
expensive_names_c = [p["product_name"] for p in products if float(p["unit_price"]) > 700]
print(expensive_names == expensive_names_c, expensive_names_c)

# %% [markdown]
# Read it in this order: **`for` clause** (where items come from), **`if` clause** (which to keep), then the **leading expression** (what to produce).
# There are two different positions for `if`:
# - a **filter** goes at the end: `[x for x in data if x > 0]` (drops items)
# - a **conditional expression** goes in the front and needs an `else`: `[x if x > 0 else 0 for x in data]` (keeps every item, transforms some)

# %%
data = [3, -1, 4, -1, 5]
print([x for x in data if x > 0])            # filter: 3 items
print([x if x > 0 else 0 for x in data])     # transform: 5 items
print([(a, b) for a in "ab" for b in (1, 2)])  # nested loops: read left to right, like nested for statements

# %% [markdown]
# ## 2. Dict and set comprehensions
# The same syntax with `{}`. A dict comprehension has `key: value`.

# %%
price_of = {p["product_name"]: float(p["unit_price"]) for p in products}
categories = {p["category"] for p in products}
print(len(price_of), sorted(categories))
cheap_by_category = {}
for p in products:
    if float(p["unit_price"]) < 100:
        cheap_by_category.setdefault(p["category"], []).append(p["product_name"])
print({k: len(v) for k, v in cheap_by_category.items()})   # comprehension over an existing dict

# %% [markdown]
# Inverting a dict, or building a lookup table, are the classic uses:

# %%
id_to_name = {p["product_id"]: p["product_name"] for p in products}
name_to_id = {v: k for k, v in id_to_name.items()}
print(id_to_name["1"], "->", name_to_id["Headphones Basic"])

# %% [markdown]
# **When not to use a comprehension:** if you need several statements, side effects (printing, writing), or the expression no longer fits comfortably on one or two lines. A plain loop is clearer.
# Do not write `[print(x) for x in data]` to loop; the comprehension builds a useless list of `None`.

# %% [markdown]
# ## 3. Generators: values on demand
# A list comprehension builds **all** results in memory now. A **generator expression** (same syntax with `()`) produces them **one at a time, when asked**.

# %%
squares_list = [n * n for n in range(1_000_000)]
squares_gen = (n * n for n in range(1_000_000))
print("list      :", sys.getsizeof(squares_list), "bytes (just the pointer array)")
print("generator :", sys.getsizeof(squares_gen), "bytes, regardless of how many items")
print(next(squares_gen), next(squares_gen), next(squares_gen))
print("sum over the rest, without ever building a list:", sum(squares_gen))

# %% [markdown]
# *What just happened?* The generator stores only its current state (where it is in `range`), not the results. `sum(...)` pulled items through it one by one. When a function's only argument
# is a generator expression you may drop the extra parentheses: `sum(x * x for x in data)`, `any(...)`, `max(...)`.
#
# ### Generator functions: `yield`
# A function containing `yield` returns a generator. Execution **pauses** at each `yield` and resumes on the next request, keeping its local variables.

# %%
def countdown(n):
    print("  (starting)")
    while n > 0:
        yield n                 # hand a value out, freeze here
        n -= 1
    print("  (done)")

g = countdown(3)
print("nothing has run yet:", g)
print(next(g))
print(next(g))
print(list(g))                   # consumes the rest

# %% [markdown]
# ### Generators are one-shot
# Once a generator is exhausted it yields nothing, silently. This is the most common generator bug.

# %%
evens = (n for n in range(10) if n % 2 == 0)
print("first pass :", list(evens))
print("second pass:", list(evens), "<- empty, no error")

def mean_of(values):
    return sum(values) / len(values)

try:
    mean_of(n for n in range(5))            # a generator has no len()
except TypeError as e:
    print("TypeError:", e)

# %% [markdown]
# If you need to iterate twice, materialise once with `list(...)`, or write a function that creates a fresh generator each time it is called.

# %% [markdown]
# ## 4. A lazy pipeline over a file
# A file object is itself a lazy iterator of lines. Chain small generator functions and only one row is in memory at a time, however large the file.

# %%
def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)          # 'yield from' delegates to another iterable

def only(rows, **conditions):
    for row in rows:
        if all(row[k] == v for k, v in conditions.items()):
            yield row

def column(rows, name):
    for row in rows:
        yield row[name]

pipeline = column(only(read_rows(DATA / "orders.csv"), status="returned", channel="web"), "order_id")
print(pipeline)                              # nothing has been read yet
first_five = list(islice(pipeline, 5))       # islice takes the first n items lazily
print(first_five)

# %% [markdown]
# `yield from` forwards items from the inner iterable. Because everything is lazy, asking for only five results reads only as much of the file as needed. This pattern (read -> filter -> transform -> aggregate)
# is exactly how streaming ETL code and pandas' `chunksize=` reading work.

# %%
n_returned_web = sum(1 for _ in only(read_rows(DATA / "orders.csv"), status="returned", channel="web"))
print("returned web orders:", n_returned_web)

# %% [markdown]
# ## Common mistakes
#
# 1. **Using a comprehension for side effects** (`[print(x) for x in xs]`).
# 2. **Putting a filter `if` in front** (`[x for ... if ...]` vs `[x if ... else y for ...]`): SyntaxError or the wrong number of items.
# 3. **Consuming a generator twice** and getting an empty result the second time.
# 4. **Calling `len()` or indexing a generator**: it has neither; use `list()` first if you truly need them.
# 5. **Over-nesting comprehensions** until nobody can read them.
# 6. **Expecting exceptions at creation time:** errors inside a generator only appear when it is iterated.

# %%
def broken_gen():
    yield 1
    raise RuntimeError("boom, but only when we get here")

g = broken_gen()             # no error yet
print(next(g))
try:
    next(g)
except RuntimeError as e:
    print("RuntimeError:", e)

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between a list comprehension and a generator expression?**
# <details><summary>Answer</summary>
# The list comprehension builds the whole list in memory immediately; the generator expression is lazy, producing one item at a time with O(1) memory, and can be consumed only once.
# </details>
#
# **2. What does <code>yield</code> do?**
# <details><summary>Answer</summary>
# It turns the function into a generator function. Each <code>yield</code> hands a value to the caller and suspends the function, preserving local state until the next <code>next()</code> call.
# </details>
#
# **3. When would you not use a comprehension?**
# <details><summary>Answer</summary>
# When it needs side effects, several statements, complex conditions or deep nesting that hurts readability, or when you need error handling per item. Use a loop.
# </details>
#
# **4. How would you process a 20 GB CSV on a laptop with plain Python?**
# <details><summary>Answer</summary>
# Iterate lazily (file object, <code>csv.reader</code>, generator pipeline), keep only running aggregates or write results incrementally, and avoid building lists of all rows.
# With pandas, use <code>chunksize</code>; for heavier work use a columnar format or a database engine such as DuckDB.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A price lookup and a margin table
# Build `price_by_id`: a dict from `product_id` (string) to `unit_price` (float) for **only the Electronics products**, using a dict comprehension. Then build
# `margin_pct`: a dict from `product_name` to gross margin percentage `(price - cost) / price * 100`, rounded to 1 decimal, for **all** products.

# %% tags=["starter"]
price_by_id = ...
margin_pct = ...

# %%
assert price_by_id == {p["product_id"]: float(p["unit_price"]) for p in products if p["category"] == "Electronics"}
assert set(margin_pct) == {p["product_name"] for p in products}
p0 = products[0]
assert margin_pct[p0["product_name"]] == round((float(p0["unit_price"]) - float(p0["unit_cost"])) / float(p0["unit_price"]) * 100, 1)
print("Exercise 1 passed:", len(price_by_id), "electronics products")

# %% tags=["solution"]
price_by_id = {p["product_id"]: float(p["unit_price"]) for p in products if p["category"] == "Electronics"}
margin_pct = {p["product_name"]: round((float(p["unit_price"]) - float(p["unit_cost"])) / float(p["unit_price"]) * 100, 1) for p in products}

# %% [markdown]
# ### Exercise 2: Lazy counting of a status
# Using `read_rows` from above and a **generator expression** (no list), store `n_delivered_mobile`: the number of orders with status `delivered` and channel `mobile`.
# *Hint: `sum(1 for row in ... if ...)`.*

# %% tags=["starter"]
n_delivered_mobile = ...

# %%
with open(DATA / "orders.csv", encoding="utf-8", newline="") as f:
    want = sum(1 for r in csv.DictReader(f) if r["status"] == "delivered" and r["channel"] == "mobile")
assert n_delivered_mobile == want, f"expected {want}, got {n_delivered_mobile}"
print("Exercise 2 passed:", n_delivered_mobile)

# %% tags=["solution"]
n_delivered_mobile = sum(1 for row in read_rows(DATA / "orders.csv") if row["status"] == "delivered" and row["channel"] == "mobile")

# %% [markdown]
# ### Exercise 3: Write a generator function
# Write `chunks(iterable, size)` as a **generator function** that yields lists of `size` items (the last one may be shorter) without materialising the whole input.
# It must work on any iterable, including another generator. *Hint: `itertools.islice(it, size)` on an `iter(...)`, stop when the slice is empty.*

# %% tags=["starter"]
def chunks(iterable, size):
    ...

# %%
assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
assert list(chunks([], 3)) == []
assert list(chunks("abcdef", 3)) == [list("abc"), list("def")]
import types
assert isinstance(chunks(range(3), 2), types.GeneratorType), "chunks must be a generator function (use yield)"
pulled = []
def noisy():
    for i in range(10):
        pulled.append(i)
        yield i
first = next(chunks(noisy(), 4))
assert first == [0, 1, 2, 3] and len(pulled) == 4, f"chunks must be lazy: pulled {len(pulled)} items to produce the first chunk"
sizes = [len(c) for c in chunks(read_rows(DATA / "orders.csv"), 1000)]
assert sum(sizes) == len(list(read_rows(DATA / "orders.csv"))), "no rows may be lost"
print("Exercise 3 passed: chunk sizes", sizes)

# %% tags=["solution"]
def chunks(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch

# %% [markdown]
# ### Exercise 4: Repair the exhausted generator
# `describe(numbers)` below receives a generator and tries to use it twice, so `max - min` is wrong. Write `spread(numbers)` returning `max - min` that works correctly when `numbers` is a
# **generator or any one-shot iterable**. *Hint: what can you do once, at the top?*

# %%
def spread_buggy(numbers):
    return max(numbers) - min(numbers)          # the second pass sees an empty generator

try:
    spread_buggy(n for n in [5, 9, 2])
except ValueError as e:
    print("buggy version raises:", e)

# %% tags=["starter"]
def spread(numbers):
    ...

# %%
assert spread(n for n in [5, 9, 2]) == 7
assert spread([1, 1]) == 0
assert spread(iter((10, 4, 6))) == 6
print("Exercise 4 passed")

# %% tags=["solution"]
def spread(numbers):
    values = list(numbers)           # materialise once
    return max(values) - min(values)

# %% [markdown]
# ### Exercise 5 (open-ended): Stream, do not load
# Write a generator pipeline over `order_items.csv` that yields `(order_id, quantity)` as ints for rows with quantity 3 or more, and use it to report the ten order ids with the most such rows
# (you may keep a dict of counts). Explain in one sentence why this scales to a file 1000 times larger, and what part of your solution would not (hint: the dict).

# %% [markdown]
# ## Recap
# - `[expr for x in it if cond]`; dict/set comprehensions use `{}`; a filter `if` goes last, a conditional expression goes in front with `else`.
# - Generators (`(...)` or `yield`) are lazy, use constant memory, and can be consumed once; errors inside surface at iteration time.
# - `yield from`, `itertools.islice`, and small composable generator functions give you streaming pipelines.
# - Use loops when a comprehension would need side effects or hurt readability.
#
# **Go deeper:** the Python docs *Generator expressions* / *Yield expressions* and the `itertools` module (docs.python.org/3), and PEP 289 (generator expressions) at peps.python.org.
#
# **Next:** [Files, JSON and CSV](07_files_json_csv.ipynb)
