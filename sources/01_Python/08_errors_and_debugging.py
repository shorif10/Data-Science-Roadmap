# %% [markdown]
# # Errors and Debugging
#
# **Why this matters.** Real data is dirty and real pipelines fail: a missing file, a blank cell, an API that times out. The skill that separates a script from a dependable tool is handling the failures you
# *expect* (and letting the ones you don't expect stop loudly), plus being able to read a traceback and find a bug methodically instead of by trial and error.
#
# **You will learn**
# - Python's exception model: `try / except / else / finally`, `raise`, custom exceptions, and exception chaining
# - how to read a traceback bottom-up and what the common exception types mean
# - `assert`, `warnings`, and `logging` and when each is appropriate
# - debugging with `breakpoint()` / `pdb` (shown here in a scripted, runnable way) and a repeatable debugging routine
#
# **Prerequisites:** [Functions](05_functions.ipynb).
#
# > Data used: small inline examples and `datasets/messy.csv` (synthetic, deliberately dirty; see `docs/DATASETS.md`).

# %%
import csv
import io
import logging
import pdb
import sys
import traceback
import warnings
from pathlib import Path

DATA = Path("../datasets")

# %% [markdown]
# ## 1. Exceptions: the basic shape
# When something goes wrong Python **raises** an exception object, which unwinds the call stack until some `except` clause handles it, or the program stops and prints a traceback.

# %%
def ratio(a, b):
    return a / b

try:
    ratio(1, 0)
except ZeroDivisionError as e:
    print("caught:", type(e).__name__, "-", e)

# %% [markdown]
# Full form, in the order they run:
#
# ```python
# try:
#     risky()               # code that might fail
# except SpecificError as e:
#     handle(e)             # runs only if that exception was raised
# else:
#     no_error_path()       # runs only if the try block raised nothing
# finally:
#     cleanup()             # ALWAYS runs: success, handled error, unhandled error, even return
# ```

# %%
def parse_age(text):
    try:
        age = int(text)
    except ValueError:
        print(f"  {text!r}: not a number")
        return None
    else:
        print(f"  {text!r}: parsed")          # only when int() worked
        return age
    finally:
        print("  (finally always runs)")

print(parse_age("42"))
print(parse_age("forty"))

# %% [markdown]
# *What just happened?* `else` keeps the code that should **not** be protected out of the `try` block, so an unrelated bug there is not swallowed by your `except`.
# `finally` is for cleanup that must happen regardless (closing, unlocking, deleting temp files).

# %% [markdown]
# ### Catch specific exceptions
# Exceptions form a hierarchy. `except Exception` catches nearly everything; a bare `except:` also catches `KeyboardInterrupt` and `SystemExit`, which you almost never want to swallow.
# Catch the narrowest type you can actually handle.

# %%
print(ZeroDivisionError.__mro__[:4])
print([c.__name__ for c in KeyError.__mro__])          # KeyError is a LookupError is an Exception

try:
    {}["missing"]
except LookupError as e:                                # catches KeyError and IndexError alike
    print("LookupError family caught:", type(e).__name__)

# %% [markdown]
# Errors you will see weekly, and what they usually mean:
#
# | Exception | Typical cause |
# |---|---|
# | `KeyError` | dict key / DataFrame column missing (typo, or different casing) |
# | `IndexError` | list position out of range (off-by-one, empty list) |
# | `ValueError` | right type, wrong value: `int("abc")` |
# | `TypeError` | wrong type: `"a" + 1`, calling `None`, wrong number of arguments |
# | `AttributeError` | `None.foo` (often a function that returned `None`), or a typo |
# | `FileNotFoundError` | wrong path or wrong working directory |
# | `ModuleNotFoundError` | package not installed in *this* environment (see Setup notebook 03) |
# | `UnicodeDecodeError` | reading a file with the wrong encoding |
# | `NameError` / `UnboundLocalError` | name not defined (run order in a notebook!) or assigned later in the function |

# %% [markdown]
# ## 2. Raising your own errors
# Validate inputs at the boundary and **fail early with a clear message**. Define a custom exception class for errors specific to your domain so callers can catch exactly those.

# %%
class DataQualityError(ValueError):
    """Raised when input data violates a rule we rely on."""

def load_salary(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise DataQualityError(f"salary {text!r} is not numeric") from exc      # chain: keep the original as __cause__
    if value < 0:
        raise DataQualityError(f"salary {value} is negative")
    return value

for raw in ["61000", "unknown", "-5"]:
    try:
        print(load_salary(raw))
    except DataQualityError as e:
        print("DataQualityError:", e, "| caused by:", repr(e.__cause__))

# %% [markdown]
# `raise ... from exc` records the underlying cause, so the traceback shows both ("The above exception was the direct cause of..."). Use a bare `raise` inside an `except` block to re-raise the current
# exception unchanged after logging it. Do **not** write `except Exception: pass`: it hides bugs and makes failures impossible to diagnose.

# %% [markdown]
# Try it on genuinely dirty data. The `salary` column of `messy.csv` contains numbers and junk:

# %%
with open(DATA / "messy.csv", encoding="utf-8", newline="") as f:
    messy = list(csv.DictReader(f))

good, bad = [], {}
for row in messy:
    try:
        good.append(load_salary(row["salary"]))
    except DataQualityError as e:
        bad[row["salary"]] = bad.get(row["salary"], 0) + 1
print(len(good), "usable salaries;", sum(bad.values()), "rejected; distinct bad values:", sorted(bad)[:6])

# %% [markdown]
# ## 3. Reading a traceback
# A traceback lists the call stack **oldest call first, the failing line last**, and the exception type and message at the very end. Read it **bottom-up**: what happened, where, and then how we got there.

# %%
def read_price(text):
    return float(text)

def read_row(row):
    return read_price(row["price"])

def read_table(rows):
    return [read_row(r) for r in rows]

try:
    read_table([{"price": "9.99"}, {"price": "N/A"}])
except ValueError:
    print(traceback.format_exc())

# %% [markdown]
# The last line says `ValueError: could not convert string to float: 'N/A'`. The frame just above it is `read_price`, where the error was raised: that is the **innermost** function and the first place to look.
# The frames above it show who called it with what. The bug is sometimes in the innermost frame (bad logic) and sometimes in a caller (passed the wrong thing); the data shown in the message tells you which.

# %% [markdown]
# ## 4. Assertions, warnings, logging
# - **`assert condition, message`**: states something that must be true if the code is correct (an internal invariant). It can be disabled with `python -O`, so never use it to validate user input or data; raise exceptions for that.
# - **`warnings.warn`**: "this works but you should know about a problem" (deprecations, suspicious inputs). Shown once per location by default.
# - **`logging`**: a record of what happened, with levels and timestamps, for programs that run unattended (see the Setup notebook on project structure).

# %%
def mean_age(ages):
    assert ages, "mean_age needs at least one age (a bug in the caller if empty)"
    if any(a > 120 for a in ages):
        warnings.warn("ages above 120 found; check the source", UserWarning, stacklevel=2)
    return sum(ages) / len(ages)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    print(mean_age([30, 40, 150]))
print("warnings captured:", [str(w.message) for w in caught])

log = logging.getLogger("demo")
stream = io.StringIO()
handler = logging.StreamHandler(stream)
handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
log.addHandler(handler)
log.setLevel(logging.INFO)
log.propagate = False
log.info("loaded %d rows", 640)
log.warning("dropped %d duplicate ids", 12)
log.error("salary column unusable in %d rows", 3)
print(stream.getvalue().strip())

# %% [markdown]
# ## 5. Debugging
# ### A routine that works
# 1. **Reproduce** reliably (same input, same error). If you cannot reproduce it, you cannot fix it.
# 2. **Read the error and the traceback** fully before changing anything.
# 3. **Shrink** the input to the smallest case that still fails (one row, three numbers).
# 4. **Form a hypothesis**, then test it: print or log the values involved, or inspect them in a debugger. Change one thing at a time.
# 5. **Bisect** when lost: find the last point where the data was still right by checking midway through the pipeline.
# 6. After fixing, **write a test** that would have caught it (next notebooks), so it stays fixed.
#
# ### `pdb`, the built-in debugger
# `breakpoint()` (or `import pdb; pdb.set_trace()`) stops execution and gives you a prompt where you can inspect variables and step through code. Main commands:
#
# | Command | Does |
# |---|---|
# | `p expr` / `pp expr` | print / pretty-print an expression |
# | `n` | next line (step over calls) |
# | `s` | step into a call |
# | `c` | continue until the next breakpoint |
# | `l` / `ll` | list source around here / the whole function |
# | `w` | where: show the call stack |
# | `u` / `d` | move up / down the stack |
# | `q` | quit |
#
# A notebook cannot show an interactive prompt in a static document, so here we **script** the debugger: we give it its keystrokes through a string instead of the keyboard. You would type the same commands.

# %%
def mean_price(prices):
    total = 0
    for p in prices:
        total += p
    return total / len(prices)

try:
    mean_price([])
except ZeroDivisionError:
    tb = sys.exc_info()[2]                                # the traceback object of the error we just caught

script = io.StringIO("w\np prices\np total\nq\n")         # the commands we would type
out = io.StringIO()
dbg = pdb.Pdb(stdin=script, stdout=out)
dbg.use_rawinput = False                                  # read commands from 'script', not the keyboard
dbg.reset()
dbg.interaction(None, tb)                                 # "post-mortem": open the debugger where the error happened
print("\n".join(line for line in out.getvalue().splitlines() if line.strip())[:900])

# %% [markdown]
# *What just happened?* Post-mortem debugging opens the debugger **at the failing frame**: `w` showed the stack, `p prices` shows an empty list, `p total` shows 0, and the cause is obvious: dividing by
# `len([])`. In a terminal you get the same by running `python -m pdb script.py` (it stops when the script crashes), and in Jupyter by running `%debug` in the cell after an exception.
# VS Code's debugger offers the same with clickable breakpoints and a variables panel, often the fastest route for most people.

# %% [markdown]
# ## Common mistakes
#
# 1. **`except:` / `except Exception: pass`** that swallows errors. Catch specific types, and log or re-raise.
# 2. **A huge `try` block** covering many lines, so you cannot tell which line failed. Keep it tight; use `else` for the rest.
# 3. **Using exceptions for validation of expected "bad" values** without a message. Include the offending value in the message.
# 4. **Fixing the symptom:** wrapping in `try/except` so the crash goes away while the data is still wrong. Find the cause.
# 5. **`assert` for input validation:** stripped under `python -O`. Raise `ValueError`.
# 6. **Losing the original traceback** by raising a new exception without `from` (or by `raise e` in some flows). Use `raise NewError(...) from exc`, or bare `raise`.

# %%
# Mistake 1: a swallowed error turns a bug into wrong numbers
def total_bad(rows):
    total = 0
    for r in rows:
        try:
            total += float(r)
        except Exception:
            pass                        # silently drops "1,5" and "N/A": the total is quietly wrong
    return total

print(total_bad(["10", "20", "1,5", "N/A"]), "<- looks fine, but two values vanished")

# %% [markdown]
# ## Interview corner
#
# **1. What runs in <code>try / except / else / finally</code> and in what order?**
# <details><summary>Answer</summary>
# <code>try</code> runs first; if it raises and an <code>except</code> matches, that handler runs; if it did not raise, <code>else</code> runs; <code>finally</code> always runs last, even when the function returns or the exception is unhandled.
# </details>
#
# **2. Why is a bare <code>except:</code> bad practice?**
# <details><summary>Answer</summary>
# It catches everything including <code>KeyboardInterrupt</code> and <code>SystemExit</code> and hides real bugs. Catch specific exceptions, and at most <code>except Exception</code> at a top-level boundary where you log and re-raise or exit.
# </details>
#
# **3. What is exception chaining?**
# <details><summary>Answer</summary>
# <code>raise NewError(...) from original</code> stores the original as <code>__cause__</code> and shows both in the traceback. It lets you translate low-level errors into domain errors without losing the root cause.
# </details>
#
# **4. How do you debug a failure you cannot reproduce locally?**
# <details><summary>Answer</summary>
# Add structured logging with context (inputs, ids, versions), capture the traceback and input that triggered it, reproduce with that exact input in a test, compare environments (Python and library versions, config, data), then fix and add a regression test.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A parser with a domain error
# Define the exception class `PriceError(ValueError)` and the function `parse_price(text)`. It accepts strings such as `"$12.50"`, `" 1,299.00 "` and `"7"`: strip spaces, remove a leading `$` and thousands commas, return a `float`.
# For anything that cannot be converted (like `"free"`) raise `PriceError` with a message containing the original text, **chained** from the underlying `ValueError`. Negative prices also raise `PriceError`.

# %% tags=["starter"]
class PriceError(ValueError):
    pass

def parse_price(text):
    ...

# %%
assert parse_price("$12.50") == 12.5 and parse_price(" 1,299.00 ") == 1299.0 and parse_price("7") == 7.0
for bad in ["free", "", "$-3", "-3"]:
    try:
        parse_price(bad)
        raise AssertionError(f"{bad!r} should raise PriceError")
    except PriceError as e:
        assert repr(bad) in str(e) or bad in str(e), f"the message should mention the input: {e}"
try:
    parse_price("free")
except PriceError as e:
    assert isinstance(e.__cause__, ValueError), "chain the original ValueError with 'raise ... from exc'"
assert issubclass(PriceError, ValueError)
print("Exercise 1 passed")

# %% tags=["solution"]
class PriceError(ValueError):
    pass

def parse_price(text):
    cleaned = text.strip().lstrip("$").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise PriceError(f"cannot parse price {text!r}") from exc
    if value < 0:
        raise PriceError(f"negative price {text!r}")
    return value

# %% [markdown]
# ### Exercise 2: Which function failed?
# `fail_chain()` below raises deep inside a call chain. Write `innermost_function(exc)` that returns the **name of the function where the exception was raised** (the last frame of the traceback).
# *Hint: `traceback.extract_tb(exc.__traceback__)` gives frames, each with `.name`.*

# %%
def level3():
    return [1, 2, 3][10]

def level2():
    return level3()

def level1():
    return level2()

# %% tags=["starter"]
def innermost_function(exc):
    ...

# %%
try:
    level1()
except IndexError as e:
    caught = e
assert innermost_function(caught) == "level3", f"got {innermost_function(caught)!r}"
try:
    read_table([{"price": "x"}])
except ValueError as e:
    assert innermost_function(e) == "read_price"
print("Exercise 2 passed")

# %% tags=["solution"]
def innermost_function(exc):
    return traceback.extract_tb(exc.__traceback__)[-1].name

# %% [markdown]
# ### Exercise 3: Retry on transient failures
# Write `call_with_retry(fn, attempts=3)`: call `fn()`; if it raises `ConnectionError` try again, up to `attempts` calls in total; return the first successful result. If every attempt fails, **re-raise the last `ConnectionError`**.
# Any other exception type must propagate immediately without retrying.

# %% tags=["starter"]
def call_with_retry(fn, attempts=3):
    ...

# %%
def make_flaky(fail_times, exc=ConnectionError):
    state = {"calls": 0}
    def fn():
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc(f"failure {state['calls']}")
        return "ok"
    return fn, state

fn, st = make_flaky(2)
assert call_with_retry(fn, attempts=3) == "ok" and st["calls"] == 3
fn, st = make_flaky(5)
try:
    call_with_retry(fn, attempts=3)
    raise AssertionError("should re-raise after the last attempt")
except ConnectionError as e:
    assert st["calls"] == 3 and "failure 3" in str(e), "re-raise the LAST error, after exactly 3 calls"
fn, st = make_flaky(1, exc=ValueError)
try:
    call_with_retry(fn, attempts=3)
    raise AssertionError("ValueError must propagate")
except ValueError:
    assert st["calls"] == 1, "do not retry errors other than ConnectionError"
print("Exercise 3 passed")

# %% tags=["solution"]
def call_with_retry(fn, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except ConnectionError:
            if attempt == attempts:
                raise

# %% [markdown]
# ### Exercise 4: Find and fix the bug
# `moving_average_buggy(values, window)` should return the average of each full window (so `len(values) - window + 1` values). Run it on `[1, 2, 3, 4]` with window 2, print what it returns, locate the bug using the method above,
# and write the corrected `moving_average`.

# %%
def moving_average_buggy(values, window):
    return [sum(values[i:i + window]) / window for i in range(len(values))]

print(moving_average_buggy([1, 2, 3, 4], 2))

# %% tags=["starter"]
def moving_average(values, window):
    ...

# %%
assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
assert moving_average([2, 4, 6], 3) == [4.0]
assert moving_average([1, 2], 3) == [], "window longer than the data: no full window"
print("Exercise 4 passed")

# %% tags=["solution"]
def moving_average(values, window):
    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]

# %% [markdown]
# ### Exercise 5 (open-ended): Postmortem of a silent failure
# Take `total_bad` from the Common mistakes section. Rewrite it so that unparseable values are **counted and logged** (using `logging`), and the function returns both the total and the number of rejected values.
# Then write three sentences: what would have happened to a dashboard built on the original, and how would you notice?

# %% [markdown]
# ## Recap
# - `try/except/else/finally`: catch specific exceptions, keep `try` blocks small, use `finally` for cleanup.
# - Raise clear errors early; use custom exception classes and `raise ... from exc` to keep the cause.
# - Read tracebacks bottom-up: the last frame is where it failed, the message says what.
# - `assert` for invariants, `warnings` for "works but suspicious", `logging` for unattended runs.
# - Debug by reproducing, shrinking, hypothesising and testing one change at a time; `breakpoint()`/`pdb`, `%debug` and the VS Code debugger let you inspect state instead of guessing.
#
# **Go deeper:** the Python tutorial chapter *Errors and Exceptions*, the `pdb`, `traceback` and `logging` docs (docs.python.org/3), and the *Logging HOWTO*.
#
# **Next:** [The standard library essentials](09_stdlib_essentials.ipynb)
