# %% [markdown]
# # Numbers, Strings and f-strings
#
# **Why this matters.** Two things go wrong in analysis code more than anything else: numbers that are *almost* right (floating point, rounding,
# money) and text that is *almost* clean (stray spaces, mixed case, wrong encoding). And the last step of nearly every analysis is turning
# numbers into readable text for a report, a log line or a dashboard label, which is what f-strings are for.
#
# **You will learn**
# - integer vs float behaviour: `//`, `%`, `divmod`, banker's rounding, why `0.1 + 0.2 != 0.3`, and when to use `Decimal`
# - strings as immutable sequences: slicing, the methods you use daily, `split`/`join`
# - f-string formatting for thousands separators, decimals, percentages, alignment and `=` debugging
# - bytes vs text and encodings, at the level you need to avoid `UnicodeDecodeError`
#
# **Prerequisites:** [Syntax, variables and types](01_syntax_variables_types.ipynb).
#
# > Data used: `datasets/products.csv` (synthetic product catalogue; see `docs/DATASETS.md`).

# %%
import csv
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

DATA = Path("../datasets")
with open(DATA / "products.csv", encoding="utf-8", newline="") as f:
    products = list(csv.DictReader(f))
print(len(products), "products; first:", products[0])

# %% [markdown]
# ## 1. Integers and division
# `int` has unlimited size. Division has three flavours, and the negative-number behaviour surprises people from C/Java:

# %%
print("7 / 2  =", 7 / 2, "   (true division: always float)")
print("7 // 2 =", 7 // 2, "   (floor division)")
print("-7 // 2 =", -7 // 2, "  (floors toward -infinity, NOT toward zero)")
print("-7 % 2 =", -7 % 2, "   (result takes the sign of the divisor, so a % b is never negative for b > 0)")
print("divmod(-7, 2) =", divmod(-7, 2), "  (the pair (a // b, a % b))")
print("int(-3.7) =", int(-3.7), "| math.floor(-3.7) =", math.floor(-3.7), "| round(-3.7) =", round(-3.7))

# %% [markdown]
# *What just happened?* Python guarantees `a == (a // b) * b + a % b`. Because `//` floors, `%` has the sign of the divisor, which is exactly what you want
# for things like "which weekday index is this?" (`day % 7`) or wrapping an index. `int()` **truncates** toward zero, `math.floor` floors: different for negatives.

# %% [markdown]
# ## 2. Floats: fast, approximate, and everywhere
# A `float` is a 64-bit **binary** fraction. Numbers such as 0.1 have no exact binary representation, like 1/3 has none in decimal.

# %%
print(0.1 + 0.2)
print(0.1 + 0.2 == 0.3)
print(f"{0.1:.20f}")           # the value actually stored for 0.1
print("difference:", 0.1 + 0.2 - 0.3)
print("isclose:", math.isclose(0.1 + 0.2, 0.3))

# %% [markdown]
# Never compare floats with `==` after arithmetic. Use `math.isclose(a, b)` (relative tolerance, default `1e-09`) or, in tests, `pytest.approx` / `numpy.isclose`.
# Errors also accumulate when magnitudes differ a lot:

# %%
print(1e16 + 1 == 1e16)                   # the +1 is lost: floats have ~15-16 significant digits
print(sum([0.1] * 10) == 1.0, "|", math.fsum([0.1] * 10) == 1.0)   # fsum tracks lost low-order bits

# %% [markdown]
# ### Rounding is not what school taught you
# Python's `round` uses **round-half-to-even** ("banker's rounding"): ties go to the even neighbour. This avoids a systematic upward bias in large sums.

# %%
print([round(x) for x in [0.5, 1.5, 2.5, 3.5]])
print(round(2.675, 2), "<- not 2.68, because the float 2.675 is really stored as", Decimal(2.675))

# %% [markdown]
# ### Money: use `Decimal` (or integer cents)
# `Decimal` does exact decimal arithmetic. **Construct it from strings**, not floats, or you import the float's error.

# %%
prices = [p["unit_price"] for p in products[:5]]          # strings straight from the CSV
print(prices)
float_total = sum(float(p) for p in prices)
dec_total = sum((Decimal(p) for p in prices), start=Decimal("0"))
print("float  total:", float_total)
print("Decimal total:", dec_total, "->", dec_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
print("Decimal(0.1) :", Decimal(0.1))
print("Decimal('0.1'):", Decimal("0.1"))

# %% [markdown]
# Trade-off: `Decimal` is exact but slower and does not mix with `float`. For large analyses (millions of rows) you normally keep floats, round only for display,
# and store money as integer cents when exactness matters (invoicing, ledgers). Knowing which regime you are in is the skill.

# %% [markdown]
# ## 3. Strings: immutable sequences of characters
# A `str` is a sequence of Unicode code points. You can index and slice it like a list, but you cannot modify it.

# %%
s = "  Headphones Pro  "
print(repr(s), len(s))
print(repr(s.strip()))
t = s.strip()
print(t[0], t[-1], t[:10], t[::-1])         # index, negative index, slice, reversed

try:
    t[0] = "h"
except TypeError as e:
    print("TypeError:", e)

# %% [markdown]
# Slices are `[start:stop:step]`, **stop is excluded**, and out-of-range slices never raise (they just clip). Every "modifying" method returns a **new** string:
# calling `t.upper()` and ignoring the result does nothing.

# %%
name = "Headphones Pro"
print(name.upper(), "|", name.lower(), "|", name.title())
print(name.replace("Pro", "Basic"), "|", name)                      # original untouched
print(name.split(" "), "|", "-".join(["a", "b", "c"]))
print(name.startswith("Head"), "Pro" in name, name.find("xyz"), name.count("o"))
print("  a  b   c ".split(), "|", "  a  b   c ".split(" "))         # split() collapses whitespace; split(" ") does not

# %% [markdown]
# Methods worth memorising: `strip/lstrip/rstrip`, `split/rsplit/splitlines`, `join`, `replace`, `startswith/endswith`, `find` (returns -1 if missing) vs `index` (raises),
# `casefold` (aggressive lower-casing for comparisons), `isdigit`, `zfill`, `partition`. **Build big strings with `"".join(parts)`**, not repeated `+=` in a loop
# (each `+=` may copy the whole string).

# %%
categories = sorted({p["category"] for p in products})
print(", ".join(categories))
print("Straße".lower() == "STRASSE".lower(), "|", "Straße".casefold() == "STRASSE".casefold())

# %% [markdown]
# ## 4. f-strings: formatting numbers for humans
# Prefix a string with `f`, put any expression in `{}`, and add a **format spec** after a colon: `{value:spec}`.
#
# | Spec | Meaning | Example (`x = 1234567.891`) | Result |
# |---|---|---|---|
# | `.2f` | 2 decimals | `{x:.2f}` | `1234567.89` |
# | `,` | thousands separator | `{x:,.0f}` | `1,234,568` |
# | `.1%` | percent | `{0.4271:.1%}` | `42.7%` |
# | `>10` / `<10` / `^10` | right / left / centre in width 10 | `{"ab":>10}` | `        ab` |
# | `010.3f` | zero-pad to width 10 | `{3.14159:010.3f}` | `000003.142` |
# | `e` | scientific | `{x:.2e}` | `1.23e+06` |
# | `!r` | use `repr()` | `{"a"!r}` | `'a'` |
# | `=` | print expression and value | `{x=}` | `x=1234567.891` |

# %%
x = 1234567.891
share = 0.4271
print(f"{x:.2f} | {x:,.0f} | {share:.1%} | {x:.2e}")
print(f"[{'ab':>10}] [{'ab':<10}] [{'ab':^10}] [{3.14159:010.3f}]")
print(f"{x=}", f"{2 + 3 = }")

# %% [markdown]
# A small formatted table, the everyday use. Width and precision can themselves come from variables with nested braces:

# %%
width = 18
print(f"{'product':<{width}}{'category':<14}{'price':>10}")
print("-" * (width + 14 + 10))
for p in products[:5]:
    print(f"{p['product_name']:<{width}}{p['category']:<14}{float(p['unit_price']):>10,.2f}")

# %% [markdown]
# *What just happened?* `<{width}` inserts the variable `width` into the spec. Values arrived as strings from the CSV, so the price needed `float()` before a numeric spec like `,.2f`
# would work (applying `.2f` to a `str` raises `ValueError`). Older styles (`"%.2f" % x` and `"{:.2f}".format(x)`) still work and you will meet them in existing code, but write f-strings in new code.
# One caution: do not build SQL or shell commands with f-strings from user input (injection); use parameters.

# %% [markdown]
# ## 5. Text vs bytes, and encodings
# Text (`str`) and bytes (`bytes`) are different types. **Encoding** turns text into bytes, **decoding** turns bytes into text. Always name the encoding when you
# read or write files (`encoding="utf-8"`); the default depends on the OS and is a common cause of "works on my machine".

# %%
word = "café"
data = word.encode("utf-8")
print(len(word), "characters ->", len(data), "bytes:", data)
print(data.decode("utf-8"))
try:
    data.decode("ascii")
except UnicodeDecodeError as e:
    print("UnicodeDecodeError:", str(e)[:60], "...")
print("wrong codec silently garbles:", data.decode("latin-1"))
print("raw string keeps backslashes:", r"C:\new\table", "| escapes:", "tab:[\t] newline count:", "a\nb".count("\n"))

# %% [markdown]
# `é` takes two bytes in UTF-8. Decoding with the wrong codec either raises (ASCII) or, worse, silently produces mojibake (`cafÃ©` with Latin-1). When data looks garbled, the fix
# is to find the right encoding, not to strip characters.

# %% [markdown]
# ## Common mistakes
#
# 1. **`==` on floats** after arithmetic. Use `math.isclose`.
# 2. **`Decimal(0.1)`** instead of `Decimal("0.1")`.
# 3. **Ignoring the return value** of string methods (`name.strip()` on its own line does nothing).
# 4. **Trusting `round()`** to give 2.5 -> 3, or `int()` to floor negatives.
# 5. **Building numbers with strings:** `"5" * 3` is `"555"`, and CSV numbers are strings until you convert them.
# 6. **Leaving the encoding to the default** when opening files.

# %%
print("5" * 3, "|", int("5") * 3)
print("  x ".strip() == "x", "| untouched original:", repr("  x "))

# %% [markdown]
# ## Interview corner
#
# **1. Why is <code>0.1 + 0.2 != 0.3</code> in Python, and how do you compare floats?**
# <details><summary>Answer</summary>
# Floats are binary approximations; 0.1, 0.2 and 0.3 are not exactly representable, so the sum differs from the literal 0.3 by a tiny amount (about 5.6e-17; print the difference to see it). Compare with a tolerance
# (<code>math.isclose</code>, <code>numpy.isclose</code>), and use <code>Decimal</code> or integer cents when you need exact decimal arithmetic such as money.
# </details>
#
# **2. What does <code>-7 // 2</code> return and why?**
# <details><summary>Answer</summary>
# <code>-4</code>. Floor division rounds toward negative infinity, and <code>%</code> is defined so that <code>a == (a // b) * b + a % b</code>, giving <code>-7 % 2 == 1</code>.
# </details>
#
# **3. Are strings mutable? What is the cost of building a string in a loop?**
# <details><summary>Answer</summary>
# Strings are immutable, so each modification creates a new string. Repeated <code>+=</code> can be quadratic in the worst case (CPython sometimes optimises it, but do not rely on it);
# collect parts in a list and <code>"".join(parts)</code> once.
# </details>
#
# **4. What is the difference between <code>str</code> and <code>bytes</code>?**
# <details><summary>Answer</summary>
# <code>str</code> is a sequence of Unicode characters; <code>bytes</code> is a sequence of integers 0-255. You convert with <code>encode</code>/<code>decode</code> using an explicit encoding
# (UTF-8 almost always). Mixing them raises <code>TypeError</code>; wrong decoding gives errors or mojibake.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Exact catalogue value
# Using `Decimal`, compute `catalogue_value`: the exact sum of `unit_price` over **all** products (strings from the CSV, no floats), quantized to 2 decimals with `ROUND_HALF_UP`.
# Also store `float_value` = the same sum computed with floats and `round(..., 2)`. *Hint: build each `Decimal` from the string.*

# %% tags=["starter"]
catalogue_value = ...
float_value = ...

# %%
exact = sum((Decimal(p["unit_price"]) for p in products), start=Decimal("0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
assert isinstance(catalogue_value, Decimal), "catalogue_value must be a Decimal"
assert catalogue_value == exact, f"expected {exact}, got {catalogue_value}"
assert isinstance(float_value, float) and math.isclose(float_value, float(exact), abs_tol=0.011), "float_value should be close to the exact total"
print(f"Exercise 1 passed: exact {catalogue_value} vs float {float_value}")

# %% tags=["solution"]
catalogue_value = sum((Decimal(p["unit_price"]) for p in products), start=Decimal("0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
float_value = round(sum(float(p["unit_price"]) for p in products), 2)

# %% [markdown]
# ### Exercise 2: A report line
# For the product with `product_id == "1"`, build `line` (a string) with **exactly** this layout using one f-string: the name left-aligned in 20 characters, then the price
# right-aligned in 10 characters with a thousands separator and 2 decimals, then the margin (price minus cost, divided by price) as a percentage with 1 decimal, right-aligned in 8.
# For example `f"{name:<20}{price:>10,.2f}{margin:>8.1%}"`-style, but you write it.

# %% tags=["starter"]
line = ...

# %%
p1 = next(p for p in products if p["product_id"] == "1")
price, cost = float(p1["unit_price"]), float(p1["unit_cost"])
expected = "{:<20}{:>10}{:>8}".format(p1["product_name"], format(price, ",.2f"), format((price - cost) / price, ".1%"))
assert line == expected, f"\nexpected {expected!r}\n     got {line!r}"
print("Exercise 2 passed:", repr(line))

# %% tags=["solution"]
p = next(p for p in products if p["product_id"] == "1")
price, cost = float(p["unit_price"]), float(p["unit_cost"])
line = f"{p['product_name']:<20}{price:>10,.2f}{(price - cost) / price:>8.1%}"

# %% [markdown]
# ### Exercise 3: Seconds to `HH:MM:SS`
# Write `to_hms(seconds)` returning a zero-padded string like `"01:01:01"` for 3661. Use `divmod`. It must work for values above 24 hours (`90000` -> `"25:00:00"`).

# %% tags=["starter"]
def to_hms(seconds):
    ...

# %%
assert to_hms(0) == "00:00:00"
assert to_hms(59) == "00:00:59"
assert to_hms(3661) == "01:01:01"
assert to_hms(86399) == "23:59:59"
assert to_hms(90000) == "25:00:00"
print("Exercise 3 passed")

# %% tags=["solution"]
def to_hms(seconds):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"

# %% [markdown]
# ### Exercise 4: Normalise messy names
# Write `clean_name(s)` that strips the ends, collapses any run of whitespace (including tabs) between words into one space, and title-cases each word.
# `"  frank   o'neil\t"` should become `"Frank O'Neil"`. *Hint: `split()` with no argument collapses whitespace; `str.title()` capitalises after an apostrophe too.*
# (Real names are harder than this; the goal is the method chain, and noticing `title()` has limits.)

# %% tags=["starter"]
def clean_name(s):
    ...

# %%
assert clean_name("  frank   o'neil\t") == "Frank O'Neil"
assert clean_name("Carla Diaz ") == "Carla Diaz"
assert clean_name("ANA\nMARIA  lopez") == "Ana Maria Lopez"
assert clean_name("") == ""
print("Exercise 4 passed")

# %% tags=["solution"]
def clean_name(s):
    return " ".join(s.split()).title()

# %% [markdown]
# ### Exercise 5 (open-ended): Price formatting study
# Print the five most expensive products as a table with a header, aligned columns, thousands separators and a "% of catalogue value" column with one decimal.
# Then write two sentences: which formatting choice made the table easiest to scan, and would you use `float` or `Decimal` for the percentages?

# %% [markdown]
# ## Recap
# - `//` floors, `%` takes the divisor's sign, `int()` truncates, `round()` is half-to-even.
# - Floats are binary approximations: compare with `math.isclose`; use `Decimal("...")` (from strings) or integer cents for money.
# - Strings are immutable; methods return new strings; build with `join`. `split()` and `split(" ")` differ.
# - f-string specs (`,`, `.2f`, `.1%`, `>10`, `=`) turn numbers into readable text; values from CSVs are strings until converted.
# - `str` and `bytes` are different; always pass `encoding="utf-8"` explicitly.
#
# **Go deeper:** the Python docs *Format Specification Mini-Language*, the `decimal` module docs, and the *Floating Point Arithmetic: Issues and Limitations* page of the tutorial (docs.python.org/3).
#
# **Next:** [Lists, tuples, sets and dicts](03_lists_tuples_sets_dicts.ipynb)
