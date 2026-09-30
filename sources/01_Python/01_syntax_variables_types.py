# %% [markdown]
# # Python Syntax, Variables and Types
#
# **Why this matters.** You already program, so the syntax takes minutes. What costs people days is Python's *object model*: names are
# labels attached to objects, some objects can change in place and some cannot, and `is` is not `==`. These rules explain the bugs in
# every later notebook (pandas "SettingWithCopy", shared default arguments, lists that change behind your back).
#
# **You will learn**
# - how names bind to objects (no "boxes"), and what `id`, `type`, `is` and `==` actually tell you
# - which built-in types are mutable, which are immutable, and why it matters for aliasing
# - truthiness, `None`, dynamic-but-strong typing, and unpacking
# - the naming and formatting conventions (PEP 8) a reviewer expects
#
# **Prerequisites:** [Setup and tools](../00_Setup_and_Tools/README.md). Experience in any other language is enough.
#
# > Data used: the first rows of `datasets/orders.csv` (synthetic; see `docs/DATASETS.md`).

# %% [markdown]
# ## 1. Names are labels, not boxes
#
# In C, `int x = 5;` reserves a box named `x` holding 5. In Python, `x = 5` creates (or finds) an **object** `5` and attaches the **name** `x` to it.
# Assignment never copies data; it binds a name.

# %%
a = [1, 2, 3]
b = a                      # b is a second label on the SAME list
b.append(4)
print("a:", a)
print("same object?", a is b, "| ids:", id(a) == id(b))

# %% [markdown]
# *What just happened?* `b = a` did not copy the list; both names point at one object, so mutating through `b` is visible through `a`.
# `id(x)` is an identity number for the object (in CPython, its memory address); `a is b` compares identities.
# To get an independent copy you must ask for one:

# %%
c = a.copy()               # or list(a), or a[:]
c.append(5)
print("a:", a, "| c:", c, "| same object?", a is c)

# %% [markdown]
# ### `==` versus `is`
# - `==` asks "do these two objects have equal **value**?" (calls `__eq__`).
# - `is` asks "are these the **very same object**?"
#
# Use `is` only for singletons: `x is None`, `x is True`. Never use it to compare numbers or strings: whether two equal numbers happen to be the same
# object is an implementation detail of CPython (it caches small integers and some strings), and can change between versions.

# %%
x = [1, 2]
y = [1, 2]
print("x == y:", x == y, "| x is y:", x is y)       # equal values, different objects

n = None
print("None check:", n is None)

# %% [markdown]
# ## 2. Everything has a type; types belong to objects, not names
# Python is **dynamically typed** (a name can be re-bound to any type) and **strongly typed** (it will not silently convert `"1" + 1`).

# %%
v = 10
print(type(v), v)
v = "ten"                   # legal: the name now points at a str
print(type(v), v)

try:
    "1" + 1
except TypeError as e:
    print("TypeError:", e)

print("explicit conversion works:", int("1") + 1, "|", "1" + str(1))

# %% [markdown]
# Core built-in types and whether the object can change after creation:
#
# | Type | Example | Mutable? | Notes |
# |---|---|---|---|
# | `int` | `42`, `1_000_000` | no | arbitrary size, no overflow |
# | `float` | `3.14` | no | 64-bit binary floating point |
# | `bool` | `True` | no | subclass of `int` |
# | `str` | `"hello"` | no | Unicode text |
# | `bytes` | `b"hi"` | no | raw bytes |
# | `NoneType` | `None` | no | "no value" singleton |
# | `tuple` | `(1, 2)` | no | fixed sequence |
# | `frozenset` | `frozenset({1})` | no | immutable set |
# | `list` | `[1, 2]` | **yes** | ordered, growable |
# | `dict` | `{"a": 1}` | **yes** | key -> value, insertion ordered |
# | `set` | `{1, 2}` | **yes** | unique items, unordered |
# | `bytearray` | `bytearray(b"x")` | **yes** | mutable bytes |
#
# **Immutable** means the object itself cannot change. "Changing" an int or a string creates a new object and re-binds the name:

# %%
s = "data"
before = id(s)
s += " science"              # builds a NEW string and re-binds s
print(s, "| same object as before?", id(s) == before)

lst = [1]
before = id(lst)
lst += [2]                   # extends the SAME list in place
print(lst, "| same object as before?", id(lst) == before)

# %% [markdown]
# Two consequences you will meet repeatedly: (1) mutable objects are the ones that can be **aliased** and surprise you, (2) only **hashable** (in practice:
# immutable) objects can be dict keys or set members.

# %%
try:
    {[1, 2]: "a list cannot be a key"}
except TypeError as e:
    print("TypeError:", e)
print({(1, 2): "a tuple can"})

# %% [markdown]
# ## 3. Numbers, booleans, None and truthiness
# Integers have no overflow; `bool` is an `int` in disguise, which is convenient for counting and a trap for type checks.

# %%
print(2 ** 100)
print(True + True, isinstance(True, int))
print(type(3 / 2), type(3 // 2), 7 % 3)

# %% [markdown]
# **Truthiness:** in `if`/`while`, these are *falsy*: `False`, `None`, `0`, `0.0`, `""`, `[]`, `()`, `{}`, `set()`. Everything else is truthy.
# This is why `if items:` means "if the list is non-empty". It also means `0` and `None` are both falsy even though they mean different things; use
# `is None` when the difference matters.

# %%
for value in [0, 1, "", "0", [], [0], None, 0.0]:
    print(f"{value!r:>5} -> {bool(value)}")

def describe_discount(discount):
    if discount is None:               # "not provided"
        return "no discount info"
    return f"discount={discount}"      # 0 is a valid discount, so 'if not discount' would be wrong here

print(describe_discount(None), "|", describe_discount(0))

# %% [markdown]
# ## 4. Unpacking and multiple assignment
# Unpacking is how Python code stays short and readable.

# %%
row = "2,392,2023-06-02,returned,mobile".split(",")
order_id, customer_id, order_date, status, channel = row
print(order_id, customer_id, order_date, status, channel)

first, *middle, last = [10, 20, 30, 40, 50]      # * collects the rest into a list
print(first, middle, last)

a, b = 1, 2
a, b = b, a                                       # swap without a temp variable
print(a, b)

# %% [markdown]
# *What just happened?* The right side is evaluated completely before any assignment, which is why the swap works. Values from `split` are all **strings**: `order_id`
# is `"2"`, not `2`. Forgetting this is a classic source of "why does my sort put 10 before 2?" bugs.

# %% [markdown]
# ## 5. Syntax and style you will be reviewed on
#
# - **Indentation is syntax.** Four spaces per level; blocks start after a `:`. Mixing tabs and spaces is an error.
# - **Naming (PEP 8):** `snake_case` for variables and functions, `CapWords` for classes, `UPPER_CASE` for constants (by convention only; Python has no `const`).
# - Names may not be keywords (`class`, `def`, `lambda`, ...). Do not **shadow built-ins** either: `list = [1, 2]` or `sum = 0` silently breaks later code.
# - `#` comments explain *why*, not *what*. A string as the first statement of a function is its **docstring** (shown by `help()`).
# - Line continuation: inside `()`, `[]`, `{}` lines continue freely; avoid backslashes.
# - Tools: `ruff` or `black` format and lint automatically; you should not debate style by hand.

# %%
total_revenue = 1_250.50          # good: descriptive snake_case, digit grouping with _
MAX_RETRIES = 3                   # constant by convention

def net_price(price, discount=0.0):
    """Return price after a fractional discount (0.1 = 10%)."""
    return price * (1 - discount)

help(net_price)

# %% [markdown]
# ## Common mistakes
#
# 1. **Comparing with `is`:** `if status is "delivered"` may work by accident and fail elsewhere (CPython even warns with a `SyntaxWarning`). Use `==`.
# 2. **Assuming assignment copies:** `backup = data` for a list or dict gives you two names for one object. Use `.copy()` (shallow) or `copy.deepcopy` (next notebooks).
# 3. **Shadowing built-ins:** a variable named `list`, `dict`, `sum`, `max` or `id` breaks later calls. Demonstrated below.
# 4. **Forgetting CSV values are strings:** `"10" < "9"` is `True`. Convert with `int()`/`float()` when you parse.
# 5. **`bool` surprises:** `isinstance(True, int)` is `True`, so check for `bool` *before* `int` when you classify values.

# %%
# Mistake 3: shadowing a built-in
numbers = [3, 1, 2]
sum = 0                          # oops: now 'sum' is an int
try:
    sum(numbers)
except TypeError as e:
    print("TypeError:", e)
del sum                          # removing the shadowing name restores the built-in
print("restored:", sum(numbers))

# Mistake 4
print("'10' < '9':", "10" < "9", "| 10 < 9:", 10 < 9)

# %% [markdown]
# ## Interview corner
#
# **1. Is Python pass-by-value or pass-by-reference?**
# <details><summary>Answer</summary>
# Neither, strictly: it is <em>pass-by-object-reference</em> (or "pass by assignment"). The parameter name is bound to the same object the caller passed.
# Mutating that object (for example <code>list.append</code>) is visible to the caller; re-binding the parameter name (<code>x = ...</code>) is not.
# </details>
#
# **2. What is the difference between <code>is</code> and <code>==</code>?**
# <details><summary>Answer</summary>
# <code>==</code> compares values via <code>__eq__</code>; <code>is</code> compares object identity. Use <code>is</code> for <code>None</code> and other singletons only.
# Small-integer and string caching makes <code>is</code> look like equality sometimes, which is an implementation detail.
# </details>
#
# **3. Mutable versus immutable types; name two of each and why it matters.**
# <details><summary>Answer</summary>
# Mutable: list, dict, set. Immutable: int, str, tuple, frozenset. Immutability makes objects safely hashable (dict keys, set members) and safe to share; mutability means aliases
# see each other's changes and mutable objects must never be default argument values.
# </details>
#
# **4. Python is dynamically typed. Is it weakly typed?**
# <details><summary>Answer</summary>
# No. Types are checked at run time and there are almost no implicit conversions between unrelated types (<code>"1" + 1</code> raises <code>TypeError</code>). "Dynamic" is about when types are
# checked; "strong" is about how strictly they are enforced.
# </details>

# %% [markdown]
# ## Exercises
# Write your answer in the starter cell, then run the check cell below it.

# %% [markdown]
# ### Exercise 1: Unpack a CSV row
# Read the **third line** (the second data row) of `../datasets/orders.csv`, split it on commas, and unpack it into
# `order_id` (an `int`), `customer_id` (an `int`), `order_date`, `status`, `channel` (strings). *Hint: `open(...)`, read lines, `int(...)`.*

# %%
from pathlib import Path
import csv

DATA = Path("../datasets")

# %% tags=["starter"]
order_id = customer_id = order_date = status = channel = ...

# %%
with open(DATA / "orders.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))
want = rows[2]
assert isinstance(order_id, int) and isinstance(customer_id, int), "order_id and customer_id must be ints"
assert [str(order_id), str(customer_id), order_date, status, channel] == want, f"expected {want}"
print("Exercise 1 passed:", order_id, customer_id, order_date, status, channel)

# %% tags=["solution"]
with open(DATA / "orders.csv", encoding="utf-8") as f:
    lines = f.read().splitlines()
oid, cid, order_date, status, channel = lines[2].split(",")
order_id, customer_id = int(oid), int(cid)

# %% [markdown]
# ### Exercise 2: Which types are mutable?
# Given the list `samples`, store in `mutable_types` the **sorted list of type names** (`type(x).__name__`) of the values that are mutable.
# *Hint: you cannot use `hash()` on mutable built-in containers; or simply reason from the table above.*

# %%
samples = [1, "s", [1], (1, 2), {1}, {"a": 1}, frozenset(), b"x", bytearray(b"x"), 3.0, None, True]

# %% tags=["starter"]
mutable_types = ...

# %%
def _hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False

expected = sorted(type(x).__name__ for x in samples if not _hashable(x))
assert mutable_types == expected, f"expected {expected}, got {mutable_types}"
print("Exercise 2 passed:", mutable_types)

# %% tags=["solution"]
mutable_types = sorted(type(x).__name__ for x in samples if type(x) in (list, set, dict, bytearray))

# %% [markdown]
# ### Exercise 3: A type labeller that handles the `bool` trap
# Write `kind(x)` returning one of the strings `"none"`, `"bool"`, `"int"`, `"float"`, `"str"`, or `"other"`.
# `True` must be reported as `"bool"`, not `"int"`.

# %% tags=["starter"]
def kind(x):
    ...

# %%
cases = [(None, "none"), (True, "bool"), (False, "bool"), (0, "int"), (7, "int"), (2.5, "float"), ("x", "str"), ([1], "other"), ((1,), "other")]
for value, want in cases:
    got = kind(value)
    assert got == want, f"kind({value!r}) returned {got!r}, expected {want!r}"
print("Exercise 3 passed")

# %% tags=["solution"]
def kind(x):
    if x is None:
        return "none"
    if isinstance(x, bool):      # must come before int: bool is a subclass of int
        return "bool"
    if isinstance(x, int):
        return "int"
    if isinstance(x, float):
        return "float"
    if isinstance(x, str):
        return "str"
    return "other"

# %% [markdown]
# ### Exercise 4: Do not mutate what you were given
# `add_tag(tags, tag)` below mutates the caller's list. Write `with_tag(tags, tag)` that returns a **new** list with `tag` appended and leaves the input untouched.

# %%
def add_tag(tags, tag):
    tags.append(tag)
    return tags

# %% tags=["starter"]
def with_tag(tags, tag):
    ...

# %%
original = ["new", "vip"]
result = with_tag(original, "churn-risk")
assert result == ["new", "vip", "churn-risk"], f"unexpected result {result}"
assert original == ["new", "vip"], "the input list was mutated"
assert result is not original, "you must return a new list"
print("Exercise 4 passed")

# %% tags=["solution"]
def with_tag(tags, tag):
    return [*tags, tag]

# %% [markdown]
# ### Exercise 5 (open-ended): Explain `0.1 + 0.2 == 0.3`
# Run `0.1 + 0.2 == 0.3` and `0.1 + 0.2`. In a markdown cell, explain in your own words why the result is what it is (hint: binary floating point),
# and write the one-line way to compare floats safely (look up `math.isclose`). The next notebook digs into this.

# %% [markdown]
# ## Recap
# - Names are labels bound to objects; assignment never copies. `.copy()` makes a shallow copy.
# - `==` compares values, `is` compares identity (use it for `None`). Python is dynamically **and** strongly typed.
# - list, dict, set are mutable; int, float, str, tuple, frozenset are immutable. Only hashable objects can be dict keys.
# - `bool` is an `int`; falsy values include `0`, `""`, `[]`, `None`; use `is None` when `0` is meaningful.
# - Unpacking (`a, *rest = ...`) and PEP 8 naming make code idiomatic; never shadow built-ins.
#
# **Go deeper:** the Python tutorial and the *Data model* chapter of the language reference (docs.python.org/3), and PEP 8 (peps.python.org/pep-0008).
#
# **Next:** [Numbers, strings and f-strings](02_numbers_strings_fstrings.ipynb)
