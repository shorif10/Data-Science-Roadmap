# %% [markdown]
# # Functions
#
# **Why this matters.** Every reusable piece of analysis (a cleaning step, a metric, a feature) should be a function with a clear signature: it can be tested, reused across notebooks and
# passed to `sorted`, `map`, `groupby` or pandas `apply`. Python functions have a few sharp edges that experienced developers still hit: **mutable default arguments**, **late-binding closures**
# and scope rules. This notebook shows each one failing, then fixes it.
#
# **You will learn**
# - positional, keyword, default, `*args`, `**kwargs` and keyword-only parameters
# - return values (including the implicit `None`), docstrings, and type hints
# - scope (LEGB), closures, `nonlocal`, and functions as first-class objects
# - the mutable-default and late-binding traps and how to avoid them
#
# **Prerequisites:** [Control flow](04_control_flow.ipynb).
#
# > Data used: `datasets/products.csv` (synthetic; see `docs/DATASETS.md`).

# %%
import csv
import statistics
from pathlib import Path

DATA = Path("../datasets")
with open(DATA / "products.csv", encoding="utf-8", newline="") as f:
    products = [{**r, "unit_price": float(r["unit_price"]), "unit_cost": float(r["unit_cost"])} for r in csv.DictReader(f)]
print(len(products), "products")

# %% [markdown]
# ## 1. Defining and calling
# `def name(parameters):` creates a function object and binds it to `name`. A function with no `return` returns `None`.

# %%
def margin(price, cost):
    """Gross margin as a fraction of price."""
    return (price - cost) / price

def report(price, cost):
    print(f"margin {margin(price, cost):.1%}")      # prints, but returns None

print(margin(100, 60))
result = report(100, 60)
print("report() returned:", result)

# %% [markdown]
# Return several values by returning a tuple (which the caller can unpack). Keep functions doing **one thing** and prefer returning values over printing, so callers decide what to do with them.

# %%
def price_stats(values):
    return min(values), max(values), statistics.mean(values)

lo, hi, avg = price_stats([p["unit_price"] for p in products])
print(f"{lo:.2f} .. {hi:.2f}, mean {avg:.2f}")

# %% [markdown]
# ## 2. Parameters in all their flavours

# %%
def describe(name, price, *tags, currency="USD", **extra):
    print(f"{name} {price} {currency} | tags={tags} | extra={extra}")

describe("Keyboard", 386.1)
describe("Keyboard", 386.1, "office", "usb", currency="EUR", warranty_years=2)

# %% [markdown]
# - `*tags` collects extra **positional** arguments into a tuple; `**extra` collects extra **keyword** arguments into a dict.
# - A parameter after `*` (here `currency`) is **keyword-only**: callers must write `currency="EUR"`. This is great for flags that would otherwise be a mystery `True`.
# - `/` marks positional-only parameters; you will see it in signatures such as `len(obj, /)`.
# - At the call site `*seq` and `**mapping` unpack a sequence or dict into arguments.

# %%
def clip(value, low=0, high=100, /, *, strict=False):
    if strict and not (low <= value <= high):
        raise ValueError(f"{value} outside [{low}, {high}]")
    return max(low, min(high, value))

print(clip(150), clip(-5, 0, 10), clip(5, 0, 10, strict=True))
args = (0, 10)
kwargs = {"strict": True}
print(clip(7, *args, **kwargs))

# %% [markdown]
# ## 3. The mutable default argument trap
# Default values are evaluated **once, when `def` runs**, not on each call. If the default is a mutable object, every call shares it.

# %%
def add_item(item, basket=[]):          # BUG: one list shared by every call
    basket.append(item)
    return basket

print(add_item("pen"))
print(add_item("book"))                 # surprise: the first item is still there
print("the default lives on the function:", add_item.__defaults__)

# %% [markdown]
# The fix is the `None` sentinel: create the fresh object *inside* the call.

# %%
def add_item_fixed(item, basket=None):
    if basket is None:
        basket = []
    basket.append(item)
    return basket

print(add_item_fixed("pen"), add_item_fixed("book"))

# %% [markdown]
# *What just happened?* `__defaults__` shows the one list that `def` created. Same rule for `{}`, `set()` and any object you construct in the signature. Immutable defaults
# (numbers, strings, `None`, tuples) are safe.

# %% [markdown]
# ## 4. Scope: LEGB
# Python looks a name up in this order: **L**ocal, **E**nclosing function, **G**lobal (module), **B**uilt-in. Assigning to a name inside a function makes it local *for the whole function*.

# %%
rate = 0.2

def apply_tax(price):
    return price * (1 + rate)           # reads the global; fine

def broken(price):
    total = price * (1 + rate)
    rate = 0.3                          # assignment makes 'rate' local to broken(), so the read above fails
    return total

print(apply_tax(100))
try:
    broken(100)
except UnboundLocalError as e:
    print("UnboundLocalError:", e)

# %% [markdown]
# Prefer passing values as parameters over reading module globals; use `global` and `nonlocal` only when you really must rebind an outer name.

# %% [markdown]
# ## 5. Functions are objects: closures and higher-order functions
# You can store a function in a variable, pass it as an argument, or return it. A function defined inside another and using its variables is a **closure**; it keeps those variables alive.

# %%
def make_multiplier(factor):
    def multiply(x):
        return x * factor               # 'factor' comes from the enclosing scope
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(10), triple(10), "| remembered:", double.__closure__[0].cell_contents)

def make_counter():
    count = 0
    def tick():
        nonlocal count                  # rebind the enclosing variable
        count += 1
        return count
    return tick

c1, c2 = make_counter(), make_counter()
print(c1(), c1(), c1(), "| independent counter:", c2())

# %%
cheapest = sorted(products, key=lambda p: p["unit_price"])[:3]        # lambda: a one-expression anonymous function
print([p["product_name"] for p in cheapest])
print(list(map(lambda p: round(margin(p["unit_price"], p["unit_cost"]), 2), products[:4])))

# %% [markdown]
# Use `lambda` only for tiny throwaway functions like sort keys; anything with a name or more than one expression should be a `def`.

# %% [markdown]
# ### The late-binding closure trap
# A closure captures the **variable**, not its value at the time. Functions created in a loop all see the loop variable's final value.

# %%
adders_bad = [lambda x: x + i for i in range(3)]
print([f(10) for f in adders_bad], "<- all use the final i = 2")

adders_good = [lambda x, i=i: x + i for i in range(3)]       # bind the current value as a default argument
print([f(10) for f in adders_good])

from functools import partial
adders_partial = [partial(lambda i, x: x + i, i) for i in range(3)]   # partial also freezes the value
print([f(10) for f in adders_partial])

# %% [markdown]
# ## 6. Docstrings and type hints
# A docstring documents behaviour (shown by `help`); **type hints** document expected types for readers and tools. Python does not enforce hints at run time (the testing notebook covers tools that do).

# %%
def normalise(values: list[float], *, ndigits: int = 3) -> list[float]:
    """Scale values to the 0-1 range.

    Raises ValueError if all values are equal (the range would be zero).
    """
    lo, hi = min(values), max(values)
    if hi == lo:
        raise ValueError("cannot normalise a constant list")
    return [round((v - lo) / (hi - lo), ndigits) for v in values]

print(normalise([10, 20, 15]))
print(normalise.__annotations__)

# %% [markdown]
# ## Common mistakes
#
# 1. **Mutable default arguments** (`def f(x, cache={})`): shared across calls. Use `None`.
# 2. **Forgetting `return`**: the function silently returns `None`, and the error appears far away (`TypeError: 'NoneType' ...`).
# 3. **Late binding in loops** (lambdas/closures created in a `for`): bind with a default argument or `functools.partial`.
# 4. **Mutating an argument** you were given (the caller's list changes). Copy first, or document it.
# 5. **Printing instead of returning**, which makes the function untestable and unreusable.
# 6. **Relying on globals** that change between notebook runs.

# %%
def sorted_prices_bad(values):
    return values.sort()                # returns None!

print(sorted_prices_bad([3, 1, 2]))

# %% [markdown]
# ## Interview corner
#
# **1. Why is <code>def f(x, items=[])</code> dangerous?**
# <details><summary>Answer</summary>
# The default list is created once when the function is defined and shared by all calls that omit the argument, so mutations accumulate. Use <code>items=None</code> and create the list inside.
# </details>
#
# **2. What is a closure? Give a use.**
# <details><summary>Answer</summary>
# A function that remembers variables from its enclosing scope after that scope has finished. Used for factories (<code>make_multiplier</code>), decorators, callbacks and simple private state.
# </details>
#
# **3. What do <code>*args</code> and <code>**kwargs</code> do?**
# <details><summary>Answer</summary>
# In a definition they collect extra positional arguments into a tuple and extra keyword arguments into a dict. In a call they unpack a sequence or mapping into arguments. Together they let wrappers forward any signature.
# </details>
#
# **4. Explain Python's argument passing model.**
# <details><summary>Answer</summary>
# Arguments are passed by object reference (assignment): the parameter name binds to the caller's object. Mutating a mutable argument affects the caller; rebinding the name inside the function does not.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A summary function with keyword-only options
# Write `summarize(values, *, ndigits=2)` returning a dict with keys `"n"`, `"mean"`, `"min"`, `"max"` (mean rounded to `ndigits`). It must raise `ValueError` for an empty list.
# Then store `catalogue_summary = summarize([p["unit_price"] for p in products])`.

# %% tags=["starter"]
def summarize(values, *, ndigits=2):
    ...

catalogue_summary = ...

# %%
vals = [p["unit_price"] for p in products]
assert summarize([1, 2, 3, 4]) == {"n": 4, "mean": 2.5, "min": 1, "max": 4}
assert summarize([1, 2, 2], ndigits=1)["mean"] == 1.7
try:
    summarize([])
    raise AssertionError("an empty list must raise ValueError")
except ValueError:
    pass
try:
    summarize([1, 2], 3)
    raise AssertionError("ndigits must be keyword-only")
except TypeError:
    pass
assert catalogue_summary == {"n": len(vals), "mean": round(sum(vals) / len(vals), 2), "min": min(vals), "max": max(vals)}
print("Exercise 1 passed:", catalogue_summary)

# %% tags=["solution"]
def summarize(values, *, ndigits=2):
    if not values:
        raise ValueError("values must not be empty")
    return {"n": len(values), "mean": round(sum(values) / len(values), ndigits), "min": min(values), "max": max(values)}

catalogue_summary = summarize([p["unit_price"] for p in products])

# %% [markdown]
# ### Exercise 2: Fix the shared-state bug
# `record_price(price, history=[])` below appends to a list shared between calls. Write `record_price_fixed(price, history=None)` that returns a **new** list each call when no history is given
# and appends to the given list otherwise.

# %%
def record_price(price, history=[]):
    history.append(price)
    return history

# %% tags=["starter"]
def record_price_fixed(price, history=None):
    ...

# %%
a = record_price_fixed(10)
b = record_price_fixed(20)
assert a == [10] and b == [20], f"calls must not share state, got {a} and {b}"
mine = [1]
out = record_price_fixed(2, mine)
assert out == [1, 2] and out is mine, "when a list is passed it should be extended in place and returned"
print("Exercise 2 passed")

# %% tags=["solution"]
def record_price_fixed(price, history=None):
    if history is None:
        history = []
    history.append(price)
    return history

# %% [markdown]
# ### Exercise 3: Closures that remember
# Write `make_price_filter(min_price)` returning a function `keep(product)` that is `True` when the product's `unit_price` is at least `min_price`.
# Then build `filters = {t: make_price_filter(t) for t in (100, 400, 700)}` and store `counts = {t: number of products kept by filters[t]}`.

# %% tags=["starter"]
def make_price_filter(min_price):
    ...

filters = {t: make_price_filter(t) for t in (100, 400, 700)}
counts = ...

# %%
want = {t: sum(p["unit_price"] >= t for p in products) for t in (100, 400, 700)}
assert counts == want, f"expected {want}, got {counts}"
assert len(set(want.values())) > 1, "the thresholds should produce different counts (sanity check on the data)"
print("Exercise 3 passed:", counts)

# %% tags=["solution"]
def make_price_filter(min_price):
    def keep(product):
        return product["unit_price"] >= min_price
    return keep

filters = {t: make_price_filter(t) for t in (100, 400, 700)}
counts = {t: sum(1 for p in products if filters[t](p)) for t in filters}

# %% [markdown]
# ### Exercise 4: Beat the late-binding trap
# Build `power_functions`, a list of 4 functions where `power_functions[k](x)` returns `x ** k` (for k = 0..3). Build it with a loop or comprehension over `range(4)`.
# A naive lambda would make all four compute `x ** 3`.

# %% tags=["starter"]
power_functions = ...

# %%
assert [f(2) for f in power_functions] == [1, 2, 4, 8], f"got {[f(2) for f in power_functions]}"
assert [f(3) for f in power_functions] == [1, 3, 9, 27]
print("Exercise 4 passed")

# %% tags=["solution"]
power_functions = [lambda x, k=k: x ** k for k in range(4)]

# %% [markdown]
# ### Exercise 5 (open-ended): Write a function you would reuse
# Pick a computation you did in an earlier notebook (a margin, a size bucket, a name cleaner). Turn it into a function with a docstring, at least one keyword-only option, and type hints.
# Write down two inputs you would test it with, including one edge case, and what it should do on invalid input.

# %% [markdown]
# ## Recap
# - Parameters: positional, keyword, defaults, `*args`, `**kwargs`, keyword-only (after `*`), positional-only (before `/`).
# - Defaults are evaluated once: never use a mutable default; use `None` and create inside.
# - Scope is LEGB; assigning inside a function makes the name local; closures keep enclosing variables (`nonlocal` to rebind).
# - Closures capture variables, not values: fix loops with default arguments or `partial`.
# - Return values instead of printing; document with docstrings and hints.
#
# **Go deeper:** the Python tutorial section *More on Defining Functions*, and the `functools` docs (`partial`, `lru_cache`, `wraps`) at docs.python.org/3.
#
# **Next:** [Comprehensions and generators](06_comprehensions_and_generators.ipynb)
