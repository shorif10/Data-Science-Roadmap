# %% [markdown]
# # Lists, Tuples, Sets and Dicts
#
# **Why this matters.** These four containers are the vocabulary of Python. Before you ever touch pandas you will use them to group rows, count categories, remove duplicates and
# look things up. Picking the right one is the difference between a 0.01 second and a 10 second job, and misunderstanding copies is behind most "my data changed by itself" bugs.
#
# **You will learn**
# - what each container is for, and the operations you use daily (indexing, slicing, sorting, membership)
# - why a `set` or `dict` lookup is fast and a `list` search is slow, measured on real data
# - shallow vs deep copies, and the `[[0] * 3] * 3` trap
# - set algebra for questions such as "who never ordered?"
#
# **Prerequisites:** [Numbers, strings and f-strings](02_numbers_strings_fstrings.ipynb).
#
# > Data used: `datasets/customers.csv` and `datasets/orders.csv` (synthetic; see `docs/DATASETS.md`).

# %%
import copy
import csv
import timeit
from pathlib import Path

DATA = Path("../datasets")


def read_csv(name):
    with open(DATA / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


customers = read_csv("customers.csv")      # list of dicts: one per customer
orders = read_csv("orders.csv")
print(len(customers), "customers,", len(orders), "orders")
print(customers[0])

# %% [markdown]
# Notice we already used a **list of dicts**: the standard in-memory shape for table-like data before pandas.
#
# | Container | Ordered | Mutable | Duplicates | Lookup by | Use it for |
# |---|---|---|---|---|---|
# | `list` | yes | yes | yes | position | a sequence you grow, sort, or iterate |
# | `tuple` | yes | no | yes | position | a fixed record; dict keys; returning several values |
# | `set` | no | yes | no | membership | uniqueness, fast `in`, set algebra |
# | `dict` | yes (insertion) | yes | keys unique | key | a mapping: counts, lookups, grouping |

# %% [markdown]
# ## 1. Lists
# Ordered, growable, any mix of types. Indexing starts at 0, negatives count from the end, slices copy.

# %%
ages = [int(c["age"]) for c in customers]
print(ages[:8], "...", ages[-3:])
print(ages[2:5], ages[::500])          # slice with a step
ages_copy = ages[:]                    # a slice is a (shallow) copy
ages_copy.sort()
print("original untouched:", ages[:3], "| sorted copy starts:", ages_copy[:3])

# %% [markdown]
# Mutating methods return `None` (a frequent bug is writing `x = x.sort()`); `sorted()` returns a new list:

# %%
nums = [3, 1, 2]
print("sort() returns:", nums.sort(), "->", nums)
print("sorted() returns:", sorted([3, 1, 2], reverse=True))
nums.append(9); nums.extend([7, 8]); nums.insert(0, 0)
print("before pop/remove:", nums)
print("pop():", nums.pop(), "| remove(9):", nums.remove(9), "->", nums)

# %% [markdown]
# Sort with a `key` function (called once per item), not by hand-writing comparisons. Python's sort is **stable**: items that compare equal keep their original order,
# so you can sort by several criteria by sorting repeatedly (least important key first).

# %%
oldest = sorted(customers, key=lambda c: int(c["age"]), reverse=True)[:3]
print([(c["first_name"], c["age"]) for c in oldest])

# %% [markdown]
# ## 2. Tuples
# Like a list that cannot change. Use tuples for **records** whose fields have different meanings and for anything that must be hashable.

# %%
point = (51.5, -0.12)
lat, lon = point
print(lat, lon)
single = (5,)                          # the comma makes the tuple, not the parentheses
print(type((5)), type(single))

visits = {("CA", 2024): 10, ("DE", 2024): 7}      # tuple as a composite key
print(visits[("CA", 2024)])

# %% [markdown]
# Immutability is shallow: a tuple containing a list cannot change *which* list it holds, but the list itself can still change.

# %%
t = ([1, 2], "x")
t[0].append(3)
print(t)

# %% [markdown]
# ## 3. Dicts
# Key -> value, backed by a hash table: lookup, insert and delete take roughly constant time regardless of size. Since Python 3.7 a dict remembers **insertion order**.

# %%
by_country = {}
for c in customers:
    by_country[c["country"]] = by_country.get(c["country"], 0) + 1     # get(key, default) avoids KeyError
print(dict(sorted(by_country.items(), key=lambda kv: kv[1], reverse=True)))

# %% [markdown]
# *What just happened?* `dict.get(k, 0)` returns 0 for a new key, so the counter needs no `if`. The same idea is packaged as `collections.Counter` and `defaultdict`
# (the stdlib essentials notebook). Useful dict tools:

# %%
d = {"a": 1, "b": 2}
print(d["a"], d.get("z"), d.get("z", 0))
print(list(d.keys()), list(d.values()), list(d.items()))
d["c"] = 3
d.update({"a": 100, "d": 4})
merged = d | {"e": 5}                       # 3.9+: new dict; right side wins on conflicts
print(merged)
print({k: v * 2 for k, v in d.items() if v > 3})        # dict comprehension
try:
    d["missing"]
except KeyError as e:
    print("KeyError:", e)

# %% [markdown]
# Index a list of records by a key when you need repeated lookups. This turns an O(n) scan per lookup into O(1):

# %%
cust_by_id = {c["customer_id"]: c for c in customers}
print(cust_by_id["4"])

# %% [markdown]
# ## 4. Sets
# Unordered collections of **unique, hashable** items. Two superpowers: removing duplicates and answering "is x in here?" fast.

# %%
countries = {c["country"] for c in customers}             # set comprehension
print(sorted(countries))
print(len(orders), "orders from", len({o["customer_id"] for o in orders}), "distinct customers")

# %% [markdown]
# Membership test speed: same question, list vs set, on real ids. We measure instead of asserting a number:

# %%
order_customer_list = [o["customer_id"] for o in orders]
order_customer_set = set(order_customer_list)
probe = "999999"                                            # an id that is NOT present: worst case for a list
t_list = timeit.timeit(lambda: probe in order_customer_list, number=200)
t_set = timeit.timeit(lambda: probe in order_customer_set, number=200)
print(f"list: {t_list:.4f}s  set: {t_set:.6f}s  -> the set is about {t_list / t_set:,.0f}x faster for this lookup")

# %% [markdown]
# The list must scan every element (O(n)); the set hashes the key and jumps to a bucket (O(1) on average). The exact ratio depends on your machine; the *shape* (grows with n) is the lesson.
#
# Set algebra answers business questions directly:

# %%
all_ids = {c["customer_id"] for c in customers}
ordering_ids = {o["customer_id"] for o in orders}
print("never ordered :", len(all_ids - ordering_ids))         # difference
print("ordered, but not in customers table:", len(ordering_ids - all_ids))
print("in both       :", len(all_ids & ordering_ids), "| in either:", len(all_ids | ordering_ids))
print("in exactly one:", len(all_ids ^ ordering_ids))

# %% [markdown]
# ## 5. Copies: shallow versus deep
# `copy()`, `list(x)` and `x[:]` copy only the **outer** container; the inner objects are shared. `copy.deepcopy` copies recursively.

# %%
cart = [["pen", 2], ["book", 1]]
shallow = cart.copy()
deep = copy.deepcopy(cart)
cart[0][1] = 99                     # change a nested list
print("original:", cart)
print("shallow :", shallow, "<- changed too: inner lists are shared")
print("deep    :", deep)

# %% [markdown]
# And the classic construction trap: multiplying a list of lists repeats the **same inner list**.

# %%
bad = [[0] * 3] * 3
bad[0][0] = 1
print("bad :", bad)                      # every row changed
good = [[0] * 3 for _ in range(3)]       # a comprehension creates a new row each time
good[0][0] = 1
print("good:", good)

# %% [markdown]
# ## Common mistakes
#
# 1. **`x = x.sort()`** or `x = x.append(1)`: these return `None`. Use `sorted(x)` or call the method without assigning.
# 2. **Mutating a list while iterating over it** skips elements. Iterate over a copy or build a new list with a comprehension.
# 3. **Assuming a shallow copy is independent** when elements are themselves lists/dicts.
# 4. **`[[0] * n] * m`** for matrices/grids.
# 5. **Using a list for membership tests in a loop** (`if x in big_list`) instead of a set.
# 6. **Unhashable keys:** lists and dicts cannot be dict keys or set members; use tuples.

# %%
# Mistake 2 demonstrated: deleting inside the loop skips items
vals = [1, 2, 2, 3, 4]
for v in vals:
    if v == 2:
        vals.remove(v)
print("after loop:", vals, "<- one of the 2s survived")
print("correct   :", [v for v in [1, 2, 2, 3, 4] if v != 2])

# %% [markdown]
# ## Interview corner
#
# **1. What is the time complexity of <code>x in some_list</code> versus <code>x in some_set</code>?**
# <details><summary>Answer</summary>
# O(n) for a list (linear scan) versus O(1) average for a set (hash lookup). Dicts behave like sets for their keys. The hash lookup can degrade with pathological collisions, but in practice it is constant.
# </details>
#
# **2. Why must dict keys be hashable, and can you use a list as a key?**
# <details><summary>Answer</summary>
# Lookup finds a bucket from <code>hash(key)</code>, so the hash must never change while the key is in the dict; mutable objects like lists could change. Use a tuple (of hashable items) instead.
# </details>
#
# **3. Explain shallow vs deep copy.**
# <details><summary>Answer</summary>
# A shallow copy creates a new outer container that references the same inner objects, so mutating an inner list is visible in both. <code>copy.deepcopy</code> recursively copies inner objects too (at higher cost).
# </details>
#
# **4. Are dicts ordered?**
# <details><summary>Answer</summary>
# Yes: since Python 3.7 a dict preserves insertion order as a language guarantee. Sets are not ordered, so never depend on set iteration order.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Customers per country
# Build `country_counts`, a dict mapping each country code in `customers` to its number of customers. Then store in `top3` the **list of the three `(country, count)` tuples with the
# highest counts**, largest first (ties: any order is accepted if the counts match). *Hint: `sorted(..., key=..., reverse=True)[:3]`.*

# %% tags=["starter"]
country_counts = ...
top3 = ...

# %%
truth = {}
for c in customers:
    truth[c["country"]] = truth.get(c["country"], 0) + 1
assert country_counts == truth, "country_counts differs from the expected per-country totals"
assert len(top3) == 3 and all(isinstance(t, tuple) for t in top3), "top3 must be a list of 3 tuples"
assert [n for _, n in top3] == sorted(truth.values(), reverse=True)[:3], "top3 must hold the 3 largest counts in descending order"
assert all(truth[k] == n for k, n in top3), "each tuple must be (country, its count)"
print("Exercise 1 passed:", top3)

# %% tags=["solution"]
country_counts = {}
for c in customers:
    country_counts[c["country"]] = country_counts.get(c["country"], 0) + 1
top3 = sorted(country_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]

# %% [markdown]
# ### Exercise 2: Who never ordered?
# Using sets, store in `never_ordered` the **sorted list of customer ids (as ints)** from `customers.csv` that have no row in `orders.csv`, and in `n_premium_never` how many of them are premium
# (`is_premium == "True"`). *Hint: difference of two sets; look the premium flag up with a dict keyed by id.*

# %% tags=["starter"]
never_ordered = ...
n_premium_never = ...

# %%
ordered = {int(o["customer_id"]) for o in orders}
want = sorted(int(c["customer_id"]) for c in customers if int(c["customer_id"]) not in ordered)
assert never_ordered == want, f"expected {len(want)} ids, got {len(never_ordered)}"
assert n_premium_never == sum(1 for c in customers if int(c["customer_id"]) in set(want) and c["is_premium"] == "True")
print(f"Exercise 2 passed: {len(never_ordered)} never ordered, {n_premium_never} of them premium")

# %% tags=["solution"]
all_ids = {int(c["customer_id"]) for c in customers}
ordered_ids = {int(o["customer_id"]) for o in orders}
never_ordered = sorted(all_ids - ordered_ids)
premium_by_id = {int(c["customer_id"]): c["is_premium"] == "True" for c in customers}
n_premium_never = sum(premium_by_id[i] for i in never_ordered)

# %% [markdown]
# ### Exercise 3: Deduplicate but keep the order
# `set(seq)` removes duplicates but loses order. Write `dedupe(seq)` returning a list with duplicates removed, **keeping the first occurrence of each item in its original position**. It should be O(n).
# *Hint: keep a `seen` set while looping; or `dict.fromkeys(seq)`.*

# %% tags=["starter"]
def dedupe(seq):
    ...

# %%
assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
assert dedupe([]) == []
assert dedupe("banana") == ["b", "a", "n"]
channels_in_order = dedupe(o["channel"] for o in orders)
assert sorted(channels_in_order) == sorted({o["channel"] for o in orders})
assert channels_in_order[0] == orders[0]["channel"], "the first channel seen must come first"
print("Exercise 3 passed:", channels_in_order)

# %% tags=["solution"]
def dedupe(seq):
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

# %% [markdown]
# ### Exercise 4: A grid that does not alias
# Build `grid`: a 4-row by 5-column grid of zeros as a list of lists, where changing one cell does not change any other row. Then set `grid[2][3] = 7`.

# %% tags=["starter"]
grid = ...

# %%
assert len(grid) == 4 and all(len(r) == 5 for r in grid), "grid must be 4 rows x 5 columns"
assert grid[2][3] == 7, "set grid[2][3] = 7"
assert sum(sum(r) for r in grid) == 7, "only ONE cell should have changed (rows must not share the same inner list)"
assert len({id(r) for r in grid}) == 4, "each row must be a distinct list object"
print("Exercise 4 passed")

# %% tags=["solution"]
grid = [[0] * 5 for _ in range(4)]
grid[2][3] = 7

# %% [markdown]
# ### Exercise 5 (open-ended): Index first, then look up
# Orders reference customers by id. Write a short loop that, for the first 5 orders, prints the order id, the customer's first name and country by looking the customer up in `cust_by_id`.
# Then explain in one sentence why building the dict first is better than searching the `customers` list inside the loop, and at what number of orders it starts to matter.

# %% [markdown]
# ## Recap
# - list: ordered/mutable; tuple: fixed record/hashable; set: unique + fast membership + algebra; dict: fast key lookup, insertion-ordered.
# - `sort()` returns `None`, `sorted()` returns a new list; `key=` does custom ordering and sorts are stable.
# - Lookup in a set/dict is O(1) on average, a list scan is O(n): index your data before looping.
# - `copy()`/slices are shallow; nested structures need `deepcopy`; `[[0]*n]*m` aliases rows.
# - Never mutate a list while iterating over it.
#
# **Go deeper:** the Python tutorial chapter *Data Structures*, and the `collections` docs (docs.python.org/3), which add `Counter`, `defaultdict` and `deque` (next notebooks).
#
# **Next:** [Control flow](04_control_flow.ipynb)
