# %% [markdown]
# # Control Flow
#
# **Why this matters.** Branching and looping are how you encode business rules ("orders over 200 with a coupon ship free"), scan data for the first match, and retry flaky work.
# The syntax is small; the Pythonic idioms (`enumerate`, `zip`, `for ... else`, `match`, short-circuit `and`/`or`) are what make code short and correct instead of a translation from C.
#
# **You will learn**
# - `if / elif / else`, conditional expressions, chained comparisons, and short-circuit evaluation
# - `for` over any iterable, `enumerate`, `zip`, `range`, and when to prefer `while`
# - `break`, `continue`, and the little-known loop `else`
# - structural pattern matching with `match` (Python 3.10+) and the walrus operator `:=`
#
# **Prerequisites:** [Lists, tuples, sets and dicts](03_lists_tuples_sets_dicts.ipynb).
#
# > Data used: `datasets/orders.csv`, `order_items.csv`, `products.csv` (synthetic; see `docs/DATASETS.md`).

# %%
import csv
import math
from pathlib import Path

DATA = Path("../datasets")


def read_csv(name):
    with open(DATA / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


orders = read_csv("orders.csv")
items = read_csv("order_items.csv")
products = read_csv("products.csv")
print(len(orders), "orders,", len(items), "order lines,", len(products), "products")

# %% [markdown]
# ## 1. Branching
# Blocks are defined by indentation after a colon. Python evaluates conditions top to bottom and runs the **first** branch that is true.

# %%
def shipping_fee(order_total, is_premium):
    if is_premium or order_total >= 200:
        return 0.0
    elif order_total >= 50:
        return 4.99
    else:
        return 9.99

print([shipping_fee(t, p) for t, p in [(30, False), (80, False), (250, False), (30, True)]])

# %% [markdown]
# Three idioms to know:
# - **Conditional expression**: `value_if_true if condition else value_if_false`, for simple choices inside an expression.
# - **Chained comparisons**: `50 <= x < 200` means `50 <= x and x < 200`, evaluated once.
# - **Short-circuiting**: `a and b` stops at the first falsy operand, `a or b` at the first truthy one, and both return *that operand* (not necessarily a bool).

# %%
total = 120
label = "big" if total >= 100 else "small"
print(label, "|", 50 <= total < 200)

name = ""
print("display name:", name or "anonymous")        # 'or' returns the first truthy operand: a common default idiom
items_list = []
print(items_list and items_list[0])                 # 'and' stops before indexing an empty list: no IndexError

# %% [markdown]
# Careful with the `or` default trick when `0` or `""` is a valid value: `discount or 0.1` turns a legitimate `0` into `0.1`. Use `if discount is None` there.

# %% [markdown]
# ## 2. `for` loops iterate over *things*, not counters
# Any iterable works: lists, strings, dicts (keys), files (lines), generators. Avoid `for i in range(len(x))` unless you truly need the index; Python offers better tools.

# %%
statuses = ["delivered", "shipped", "returned"]

for s in statuses:                              # direct iteration
    print(s.upper(), end=" ")
print()

for i, s in enumerate(statuses, start=1):       # index + value
    print(i, s)

prices = [470.7, 712.94, 386.1]
names = ["Headphones Basic", "Headphones Pro", "Keyboard Basic"]
for name, price in zip(names, prices):          # walk two sequences together
    print(f"{name:<18}{price:>8.2f}")

# %% [markdown]
# `zip` stops at the **shortest** input silently; in Python 3.10+ `zip(a, b, strict=True)` raises if the lengths differ, which is what you want when a mismatch means corrupted data.

# %%
try:
    list(zip([1, 2, 3], ["a", "b"], strict=True))
except ValueError as e:
    print("ValueError:", e)

# %% [markdown]
# `range(start, stop, step)` produces integers with the stop **excluded** and uses constant memory. Looping over a dict: `for k in d`, `for v in d.values()`, `for k, v in d.items()`.

# %%
print(list(range(3)), list(range(2, 10, 3)), list(range(5, 0, -2)))
d = {"web": 2899, "mobile": 2271}
for channel, n in d.items():
    print(channel, n)

# %% [markdown]
# ## 3. `while`, `break`, `continue` and loop `else`
# Use `while` when you do not know the number of iterations in advance (convergence, retries, reading until a sentinel).

# %%
balance, months = 1000.0, 0
while balance < 2000:                # how long to double at 1% monthly interest?
    balance *= 1.01
    months += 1
print("months to double:", months, "| check by formula:", math.ceil(math.log(2) / math.log(1.01)))

# %% [markdown]
# `break` leaves the loop; `continue` skips to the next iteration. A loop can also have an **`else` block that runs only if the loop finished without hitting `break`**.
# It reads as "no break happened", and is the clean way to write "search, and if nothing found do X" without a flag variable.

# %%
def first_product_above(limit):
    for p in products:
        if float(p["unit_price"]) > limit:
            found = p["product_name"]
            break
    else:                                   # only when the loop never broke
        found = None
    return found

print(first_product_above(700), "|", first_product_above(10_000))

# %% [markdown]
# ## 4. `match`: branching on structure (Python 3.10+)
# `match` compares a value against patterns and can **destructure** it. It is not a C-style `switch`: patterns can bind names and check shapes.

# %%
def handle(event):
    match event:
        case {"type": "order", "status": "delivered"}:
            return "close ticket"
        case {"type": "order", "status": "returned" | "cancelled" as st}:
            return f"start refund review ({st})"
        case {"type": "order"}:
            return "wait"
        case (x, y):
            return f"pair {x},{y}"
        case [first, *rest]:
            return f"list starting {first} with {len(rest)} more"
        case _:
            return "unknown"

for e in [{"type": "order", "status": "delivered"}, {"type": "order", "status": "returned"},
          {"type": "order", "status": "shipped"}, (3, 4), [1, 2, 3], 42]:
    print(e, "->", handle(e))

# %% [markdown]
# *What just happened?* Patterns are tried in order; `|` means "either", `as` captures the matched value, `[first, *rest]` destructures sequences, `_` is the catch-all.
# For simple value-to-value lookups a **dict** is still the better tool; use `match` when the shape of the data drives the logic (parsed JSON events, AST-like structures).

# %% [markdown]
# ### The walrus operator `:=`
# Assigns and tests in one expression: handy in `while` loops and comprehensions that would otherwise compute a value twice.

# %%
data = [3, 8, 15, 4, 42]
big = [y for x in data if (y := x * 2) > 20]        # compute x*2 once, use it in the test and the result
print(big)

# %% [markdown]
# ## 5. Putting it together: scanning real data
# Question: how many orders in a row (a "streak") were delivered before the first non-delivered order, and what was the longest run of consecutive `delivered` orders overall?

# %%
longest = current = 0
first_problem = None
for i, o in enumerate(orders):
    if o["status"] == "delivered":
        current += 1
        longest = max(longest, current)
    else:
        if first_problem is None:
            first_problem = (i, o["order_id"], o["status"])
        current = 0
print("first non-delivered:", first_problem)
print("longest delivered streak:", longest)

# %% [markdown]
# One pass, constant memory, and two pieces of state (`current`, `longest`) updated by simple rules. Many data problems reduce to "walk once, keep a small state".

# %% [markdown]
# ## Common mistakes
#
# 1. **Modifying a list while looping over it** (see the previous notebook). Build a new list instead.
# 2. **Off-by-one with `range`:** the stop is excluded. `range(1, n)` misses `n`.
# 3. **`if x == 1 or 2:`** is always truthy (`2` is truthy). Write `x in (1, 2)`.
# 4. **`for ... else` misread as "if the loop had no iterations".** It runs whenever there was no `break`, including on an empty iterable.
# 5. **Using `or` for defaults when 0/empty is valid.** Compare with `None`.
# 6. **Infinite `while` loops:** make sure the loop variable changes and cap retries.

# %%
x = 5
print("x == 1 or 2 ->", bool(x == 1 or 2), "  <- wrong: always truthy")
print("x in (1, 2) ->", x in (1, 2))

for _ in []:
    pass
else:
    print("loop-else ran on an empty loop (no break happened)")

# %% [markdown]
# ## Interview corner
#
# **1. What does <code>for ... else</code> do?**
# <details><summary>Answer</summary>
# The <code>else</code> block runs when the loop ends normally, that is, without <code>break</code>. It is used for search loops: break when found, else handle "not found".
# </details>
#
# **2. What does <code>a or b</code> return?**
# <details><summary>Answer</summary>
# The first truthy operand, or the last operand if none is truthy. <code>and</code> returns the first falsy operand, or the last. They short-circuit, so the right side may never be evaluated.
# </details>
#
# **3. How do you iterate with both index and value, and over two lists together?**
# <details><summary>Answer</summary>
# <code>enumerate(seq, start=0)</code> and <code>zip(a, b)</code> (add <code>strict=True</code> to error on unequal lengths). Both are lazy iterators.
# </details>
#
# **4. Why is <code>if x == 1 or 2</code> a bug?**
# <details><summary>Answer</summary>
# It parses as <code>(x == 1) or 2</code>, and <code>2</code> is always truthy, so the condition is always true. Use <code>x in (1, 2)</code>.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Bucket the orders by size
# Write `size_label(units)` returning `"small"` for fewer than 3 units, `"medium"` for 3 to 5 units inclusive, and `"large"` for 6 or more. Then store `label_counts`: a dict of how many **orders**
# fall in each label, where an order's units are the sum of `quantity` over its rows in `order_items.csv`. *Hint: build a dict `order_id -> units` first.*

# %% tags=["starter"]
def size_label(units):
    ...

label_counts = ...

# %%
assert [size_label(u) for u in (1, 2, 3, 5, 6, 40)] == ["small", "small", "medium", "medium", "large", "large"], "check the boundaries 2/3 and 5/6"
units = {}
for r in items:
    units[r["order_id"]] = units.get(r["order_id"], 0) + int(r["quantity"])
want = {"small": 0, "medium": 0, "large": 0}
for u in units.values():
    want["small" if u < 3 else "medium" if u <= 5 else "large"] += 1
assert label_counts == want, f"expected {want}, got {label_counts}"
assert sum(label_counts.values()) == len(units)
print("Exercise 1 passed:", label_counts)

# %% tags=["solution"]
def size_label(units):
    if units < 3:
        return "small"
    elif units <= 5:
        return "medium"
    return "large"

units_per_order = {}
for r in items:
    units_per_order[r["order_id"]] = units_per_order.get(r["order_id"], 0) + int(r["quantity"])
label_counts = {"small": 0, "medium": 0, "large": 0}
for u in units_per_order.values():
    label_counts[size_label(u)] += 1

# %% [markdown]
# ### Exercise 2: Search with `for ... else`
# Write `first_order_id(status, channel)` returning the `order_id` (a string) of the **first** order in file order with that status and channel, or `None` if there is none.
# Use `for ... else` (or an early return). *Check both a real combination and one that does not exist.*

# %% tags=["starter"]
def first_order_id(status, channel):
    ...

# %%
def _expected(status, channel):
    return next((o["order_id"] for o in orders if o["status"] == status and o["channel"] == channel), None)

for st, ch in [("delivered", "web"), ("returned", "store"), ("cancelled", "mobile"), ("shipped", "fax"), ("unknown", "web")]:
    assert first_order_id(st, ch) == _expected(st, ch), f"wrong answer for ({st}, {ch})"
assert first_order_id("shipped", "fax") is None
print("Exercise 2 passed")

# %% tags=["solution"]
def first_order_id(status, channel):
    for o in orders:
        if o["status"] == status and o["channel"] == channel:
            return o["order_id"]
    return None

# %% [markdown]
# ### Exercise 3: Running totals with `enumerate`/`zip`
# Take the quantities of the **first 20 order lines**. Build `running` (a list of 20 ints) where `running[i]` is the sum of the first `i + 1` quantities, using a loop (no `itertools`).
# Also store `argmax_jump`: the 0-based position of the line with the **largest quantity** (first one if tied).

# %% tags=["starter"]
running = ...
argmax_jump = ...

# %%
from itertools import accumulate
q = [int(r["quantity"]) for r in items[:20]]
assert running == list(accumulate(q)), f"expected {list(accumulate(q))}, got {running}"
assert argmax_jump == q.index(max(q)), f"the largest quantity {max(q)} is at position {q.index(max(q))}"
print("Exercise 3 passed:", running[-5:], argmax_jump)

# %% tags=["solution"]
q = [int(r["quantity"]) for r in items[:20]]
running = []
total = 0
for value in q:
    total += value
    running.append(total)
best, argmax_jump = q[0], 0
for i, value in enumerate(q):
    if value > best:
        best, argmax_jump = value, i

# %% [markdown]
# ### Exercise 4: Rules with `match`
# Write `route(ticket)` where `ticket` is a dict. Return: `"fraud-team"` if `ticket["kind"] == "payment"` and `ticket["amount"] > 1000`; `"billing"` for any other payment;
# `"logistics"` if the kind is `"shipping"` or `"delivery"`; `"general"` for everything else (including dicts with no `kind`). Use `match` with mapping patterns and a guard (`case ... if ...`).

# %% tags=["starter"]
def route(ticket):
    ...

# %%
cases = [
    ({"kind": "payment", "amount": 5000}, "fraud-team"),
    ({"kind": "payment", "amount": 20}, "billing"),
    ({"kind": "shipping"}, "logistics"),
    ({"kind": "delivery", "order_id": 7}, "logistics"),
    ({"kind": "other"}, "general"),
    ({}, "general"),
]
for t, want in cases:
    assert route(t) == want, f"route({t}) -> {route(t)!r}, expected {want!r}"
print("Exercise 4 passed")

# %% tags=["solution"]
def route(ticket):
    match ticket:
        case {"kind": "payment", "amount": amount} if amount > 1000:
            return "fraud-team"
        case {"kind": "payment"}:
            return "billing"
        case {"kind": "shipping" | "delivery"}:
            return "logistics"
        case _:
            return "general"

# %% [markdown]
# ### Exercise 5 (open-ended): Find a pattern in the order stream
# Using one pass over `orders`, compute something with a small piece of state: for example the longest run of consecutive orders from the same channel, or the first date on which a `returned`
# order appears twice in a row. Print your finding and say in a sentence whether it is surprising or just what randomness would produce.

# %% [markdown]
# ## Recap
# - `if/elif/else`, conditional expressions and chained comparisons; `and`/`or` short-circuit and return operands.
# - Iterate directly over iterables; use `enumerate` and `zip` (`strict=True`) instead of index arithmetic; `range` excludes its stop.
# - `for ... else` runs when no `break` happened; `while` for unknown iteration counts.
# - `match` destructures data shapes; a dict is still better for plain lookups. `:=` avoids computing a value twice.
# - Many data tasks are "one pass with a small state".
#
# **Go deeper:** the Python tutorial chapter *More Control Flow Tools*, and PEP 636 (structural pattern matching tutorial) at peps.python.org.
#
# **Next:** [Functions](05_functions.ipynb)
