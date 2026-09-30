# %% [markdown]
# # Standard Library Essentials
#
# **Why this matters.** "Batteries included" is real: `collections`, `itertools`, `pathlib`, `datetime`, `re` and `functools` solve counting, grouping, dates, text extraction and caching without installing anything.
# Knowing them keeps scripts short, and they show up constantly in interviews and in code you will inherit.
#
# **You will learn**
# - `collections`: `Counter`, `defaultdict`, `deque`, `namedtuple`
# - `itertools`: `chain`, `islice`, `groupby` (and its sorting trap), `product`, `combinations`, `accumulate`
# - `pathlib` for finding files; `datetime` for parsing, arithmetic and formatting dates
# - `re` for extracting structure from text; `functools` for `lru_cache` and `partial`
#
# **Prerequisites:** [Errors and debugging](08_errors_and_debugging.ipynb).
#
# > Data used: `datasets/orders.csv`, `order_items.csv` (synthetic; see `docs/DATASETS.md`).

# %%
import csv
import re
from collections import Counter, defaultdict, deque, namedtuple
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache, partial
from itertools import accumulate, chain, combinations, groupby, islice, product
from pathlib import Path

DATA = Path("../datasets")


def read_csv(name):
    with open(DATA / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


orders = read_csv("orders.csv")
items = read_csv("order_items.csv")

# %% [markdown]
# ## 1. `collections`
# **`Counter`**: a dict specialised for counting.

# %%
status_counts = Counter(o["status"] for o in orders)
print(status_counts)
print(status_counts.most_common(2))
print(Counter("mississippi").most_common(2), "| missing key ->", status_counts["lost"])
print(Counter(a=3) + Counter(a=1, b=2), "<- counters add")

# %% [markdown]
# **`defaultdict`**: a dict that creates a missing value with a factory, removing the `if key not in d` dance. Grouping is its main use.

# %%
orders_by_customer = defaultdict(list)
for o in orders:
    orders_by_customer[o["customer_id"]].append(o["order_id"])
biggest = max(orders_by_customer.items(), key=lambda kv: len(kv[1]))
print("most orders:", biggest[0], "with", len(biggest[1]))

# %% [markdown]
# A `defaultdict` **inserts** the key on a plain lookup, so merely reading a missing key grows the dict. Use `key in d` to test without inserting.
#
# **`deque`**: a double-ended queue with O(1) appends/pops at both ends (a list is O(n) at the front). `maxlen` makes a sliding window.
# **`namedtuple`**: a tuple with named fields, lightweight and immutable.

# %%
last3 = deque(maxlen=3)
for n in range(6):
    last3.append(n)
print(last3)

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p, p.x, p[1], p._asdict())

# %% [markdown]
# ## 2. `itertools`: building blocks for iteration
# All lazy, all composable.

# %%
print(list(chain([1, 2], "ab", (9,))))                          # concatenate iterables
print(list(islice(range(100), 5, 10, 2)))                       # slicing for any iterator
print(list(accumulate([1, 2, 3, 4])), list(accumulate([3, 1, 4], max)))   # running total / running max
print(list(product("ab", [1, 2])))                               # cartesian product
print(list(combinations("abc", 2)))                              # unordered pairs, no repeats

# %% [markdown]
# ### `groupby`: powerful, and with one famous trap
# `groupby(iterable, key)` groups **consecutive** items with the same key. If the data is not sorted by that key, the same key appears in several groups. `order_items.csv` happens to be sorted by `order_id`,
# so we can total quantities per order in one lazy pass:

# %%
def qty_per_order(rows):
    for order_id, group in groupby(rows, key=lambda r: r["order_id"]):
        yield order_id, sum(int(r["quantity"]) for r in group)

print(list(islice(qty_per_order(items), 4)))

# the trap: unsorted input gives repeated keys
letters = "aabbaa"
print([(k, len(list(g))) for k, g in groupby(letters)], "<- 'a' appears twice")
print([(k, len(list(g))) for k, g in groupby(sorted(letters))], "<- sort first")

# %% [markdown]
# ## 3. `pathlib`: files and folders as objects

# %%
csvs = sorted(p.name for p in DATA.glob("*.csv"))
print(csvs)
f = DATA / "orders.csv"
print(f.suffix, f.stem, f.parent.name, f.exists(), f.stat().st_size > 0)

# %% [markdown]
# ## 4. `datetime`: parse, compute, format
# Dates in CSV files are strings. Convert once, then use real date arithmetic.

# %%
d = date.fromisoformat(orders[0]["order_date"])             # ISO format YYYY-MM-DD
print(d, d.strftime("%A %d %B %Y"), "| weekday() (Mon=0):", d.weekday(), "| isocalendar week:", d.isocalendar().week)

dates = [date.fromisoformat(o["order_date"]) for o in orders]
span = max(dates) - min(dates)                               # subtracting dates gives a timedelta
print("first:", min(dates), "last:", max(dates), "span:", span.days, "days")
print(date(2024, 1, 31) + timedelta(days=30), "|", datetime.strptime("17/05/2024 13:45", "%d/%m/%Y %H:%M"))

# %% [markdown]
# Two traps. **Naive vs aware:** a `datetime` without a time zone is "naive"; you cannot compare it with an aware one, and mixing them silently yields wrong hours. Store UTC (aware) and convert for display.
# **Month arithmetic** is not a fixed number of days, so `timedelta(days=30)` is not "one month".

# %%
naive = datetime(2024, 5, 17, 12, 0)
aware = datetime(2024, 5, 17, 12, 0, tzinfo=timezone.utc)
try:
    naive < aware
except TypeError as e:
    print("TypeError:", e)
print(aware.astimezone(timezone(timedelta(hours=6))).isoformat())

# %% [markdown]
# ## 5. `re`: regular expressions
# A regex describes a text pattern. Raw strings (`r"..."`) avoid double backslashes. Key functions: `search` (first match anywhere), `match` (only at the start), `fullmatch`, `findall`, `sub`, and `compile` for reuse.

# %%
text = "Order 1042 shipped on 2024-03-05 to ravi@example.com; order 77 returned 2024-03-09."
print(re.findall(r"\d{4}-\d{2}-\d{2}", text))
print(re.search(r"[\w.]+@[\w.]+", text).group())
print(re.sub(r"\d{4}-\d{2}-\d{2}", "<date>", text)[:50])

# %% [markdown]
# **Named groups** make extraction readable, and `finditer` gives match objects:

# %%
pattern = re.compile(r"[Oo]rder (?P<id>\d+) (?P<action>shipped|returned)")
for m in pattern.finditer(text):
    print(m.group("id"), m.group("action"), m.span())

# %% [markdown]
# **Greedy vs lazy:** `.*` grabs as much as possible; `.*?` as little as possible.

# %%
html = "<b>one</b> and <b>two</b>"
print(re.findall(r"<b>.*</b>", html), "<- greedy swallowed everything between the outer tags")
print(re.findall(r"<b>(.*?)</b>", html), "<- lazy")

# %% [markdown]
# Regexes are brittle for nested or structured formats (HTML, JSON, CSV): use a real parser for those. Keep patterns short, named and tested.

# %% [markdown]
# ## 6. `functools`: caching and partial application
# `lru_cache` memoises a pure function (same arguments, same result). Arguments must be hashable.

# %%
calls = 0

@lru_cache(maxsize=None)
def fib(n):
    global calls
    calls += 1
    return n if n < 2 else fib(n - 1) + fib(n - 2)

print(fib(40), "computed with only", calls, "real calls;", fib.cache_info())

power_of_two = partial(pow, 2)              # partial: fix some arguments in advance
print([power_of_two(n) for n in range(5)])

# %% [markdown]
# Caches keep results alive (memory) and are wrong for functions with side effects or that read changing data. `maxsize` bounds the memory.

# %% [markdown]
# ## Common mistakes
#
# 1. **`groupby` on unsorted data.**
# 2. **Reading a `defaultdict` key** and accidentally creating it.
# 3. **Naive datetimes** compared or stored across time zones; using 30 days for "a month".
# 4. **`re.match` when you meant `re.search`:** `match` only anchors at the start of the string.
# 5. **Regex for nested formats** such as HTML.
# 6. **`lru_cache` on a function whose result depends on mutable state or files.**

# %%
print(re.match(r"\d+", "order 12"), "<- match fails: the string does not START with digits")
print(re.search(r"\d+", "order 12").group(), "<- search finds it")
dd = defaultdict(int)
_ = dd["typo"]
print("reading inserted a key:", dict(dd))

# %% [markdown]
# ## Interview corner
#
# **1. Counter vs defaultdict(int)?**
# <details><summary>Answer</summary>
# Both count, but <code>Counter</code> takes an iterable directly, returns 0 for missing keys without inserting them, and adds <code>most_common</code> and arithmetic. <code>defaultdict(int)</code> inserts keys on access and is more general (any factory).
# </details>
#
# **2. Why does <code>itertools.groupby</code> need sorted input?**
# <details><summary>Answer</summary>
# It only groups runs of adjacent equal keys in a single lazy pass; it never looks back. Unsorted data yields the same key in several groups. Sort by the key first (or use a dict / pandas <code>groupby</code>).
# </details>
#
# **3. <code>re.match</code> vs <code>re.search</code> vs <code>re.fullmatch</code>?**
# <details><summary>Answer</summary>
# <code>match</code> requires a match at the beginning, <code>search</code> finds the first match anywhere, <code>fullmatch</code> requires the entire string to match.
# </details>
#
# **4. When is <code>lru_cache</code> appropriate?**
# <details><summary>Answer</summary>
# For pure, deterministic functions with hashable arguments that are called repeatedly with the same inputs (recursive algorithms, expensive lookups). Not for functions with side effects or results that change over time.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Top customers
# Store in `top_customers` the list of the **three `(customer_id, order_count)` pairs** with the most orders (ids as strings), largest first. *Hint: `Counter(...).most_common(3)`.*

# %% tags=["starter"]
top_customers = ...

# %%
manual = {}
for o in orders:
    manual[o["customer_id"]] = manual.get(o["customer_id"], 0) + 1
want_counts = sorted(manual.values(), reverse=True)[:3]
assert [n for _, n in top_customers] == want_counts, f"counts should be {want_counts}"
assert all(manual[c] == n for c, n in top_customers), "each pair must be (customer_id, its order count)"
print("Exercise 1 passed:", top_customers)

# %% tags=["solution"]
top_customers = Counter(o["customer_id"] for o in orders).most_common(3)

# %% [markdown]
# ### Exercise 2: Orders per weekday
# Store `weekday_counts`: a dict mapping the weekday name (`"Monday"`, ...) to the number of orders placed on that weekday, using `date.fromisoformat` and `strftime("%A")`.

# %% tags=["starter"]
weekday_counts = ...

# %%
import calendar
want = {name: 0 for name in calendar.day_name}
for o in orders:
    y, m, d_ = map(int, o["order_date"].split("-"))
    want[calendar.day_name[calendar.weekday(y, m, d_)]] += 1
assert dict(weekday_counts) == want, f"expected {want}, got {dict(weekday_counts)}"
assert sum(weekday_counts.values()) == len(orders)
print("Exercise 2 passed:", dict(weekday_counts))

# %% tags=["solution"]
weekday_counts = Counter(date.fromisoformat(o["order_date"]).strftime("%A") for o in orders)

# %% [markdown]
# ### Exercise 3: The busiest month
# Store `orders_by_month`: a dict mapping `"YYYY-MM"` to order count, and `busiest_month`: the month with the most orders. *Hint: string slicing `order_date[:7]`, or `strftime("%Y-%m")`.*

# %% tags=["starter"]
orders_by_month = ...
busiest_month = ...

# %%
want = {}
for o in orders:
    want[o["order_date"][:7]] = want.get(o["order_date"][:7], 0) + 1
assert dict(orders_by_month) == want, "month counts differ"
assert want[busiest_month] == max(want.values()), "busiest_month is not the month with the most orders"
print("Exercise 3 passed:", busiest_month, orders_by_month[busiest_month])

# %% tags=["solution"]
orders_by_month = Counter(o["order_date"][:7] for o in orders)
busiest_month = orders_by_month.most_common(1)[0][0]

# %% [markdown]
# ### Exercise 4: Parse log lines with a regex
# Write `parse_log(line)` for lines like `"2024-03-05 14:02:11 ERROR payment-service: timeout after 30s"`. Return a dict with keys `date`, `time`, `level`, `service`, `message` (all strings), or `None` if the line does not match.
# Use one compiled regex with named groups. Level is one of INFO/WARNING/ERROR.

# %% tags=["starter"]
def parse_log(line):
    ...

# %%
r = parse_log("2024-03-05 14:02:11 ERROR payment-service: timeout after 30s")
assert r == {"date": "2024-03-05", "time": "14:02:11", "level": "ERROR", "service": "payment-service", "message": "timeout after 30s"}, r
assert parse_log("2024-03-05 14:02:12 INFO api: ok: 200")["message"] == "ok: 200"
assert parse_log("garbage line") is None
assert parse_log("2024-03-05 14:02:11 DEBUG x: y") is None, "only INFO/WARNING/ERROR are valid levels"
print("Exercise 4 passed")

# %% tags=["solution"]
LOG_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<level>INFO|WARNING|ERROR) (?P<service>[\w-]+): (?P<message>.*)")

def parse_log(line):
    m = LOG_RE.fullmatch(line)
    return m.groupdict() if m else None

# %% [markdown]
# ### Exercise 5 (open-ended): Cache a slow lookup
# Write a function that finds the number of orders a customer placed by scanning `orders` (slow, O(n)). Decorate it with `lru_cache`, call it 1000 times for random customers, and compare `cache_info()`
# hits and the total time with the undecorated version. Say when caching would give you *wrong* answers here.

# %% [markdown]
# ## Recap
# - `Counter`, `defaultdict`, `deque`, `namedtuple` cover counting, grouping, queues and light records.
# - `itertools` gives lazy building blocks; `groupby` needs input sorted by the key.
# - `pathlib` for paths; `datetime` for real date maths (prefer aware UTC datetimes); `timedelta` is days/seconds, not months.
# - `re`: `search` vs `match`, named groups, greedy vs lazy; do not parse nested formats with regex.
# - `lru_cache` memoises pure functions; `partial` pre-fills arguments.
#
# **Go deeper:** the docs for `collections`, `itertools` (including the Recipes section), `datetime`, `re` and the *Regular Expression HOWTO* (docs.python.org/3).
#
# **Next:** [OOP, classes and dataclasses](10_oop_classes_dataclasses.ipynb)
