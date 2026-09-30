# %% [markdown]
# # Testing, Typing and pytest
#
# **Why this matters.** "It ran without errors" is not "it is correct". Data code fails silently: a join that drops rows, a percentage computed on the wrong base. Automated tests turn "I think it works" into evidence you can
# re-run after every change, and type hints make function contracts readable. In a job, untested code is code nobody dares to touch, and most teams run `pytest` in CI on every pull request.
#
# **You will learn**
# - type hints: the modern syntax, what they do (and do not do) at run time
# - runtime validation of data with Pydantic
# - writing tests with `pytest`: plain `assert`, `parametrize`, `pytest.raises`, `approx`, fixtures, `tmp_path`
# - running pytest on a real temp package from Python, and reading a failure
#
# **Prerequisites:** [Iterators, decorators and context managers](11_iterators_decorators_context_managers.ipynb).
#
# > Everything runs offline in a temp folder. We run `pytest` as a subprocess so you see exactly what you would see in a terminal.

# %%
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Callable, Literal, Protocol, TypedDict, get_type_hints

# %% [markdown]
# ## 1. Type hints
# Annotate parameters and return values. The syntax you need today (Python 3.10+):

# %%
def average(values: list[float], *, default: float | None = None) -> float | None:
    """Mean of values, or `default` if empty."""
    return sum(values) / len(values) if values else default

def apply_all(funcs: list[Callable[[float], float]], x: float) -> list[float]:
    return [f(x) for f in funcs]

class OrderRow(TypedDict):
    order_id: int
    status: Literal["delivered", "shipped", "cancelled", "returned"]

row: OrderRow = {"order_id": 1, "status": "delivered"}
print(get_type_hints(average))

# %% [markdown]
# Cheat sheet: `list[int]`, `dict[str, float]`, `tuple[int, str]`, `set[str]`, `X | None` (optional, older code writes `Optional[X]`), `Callable[[A], B]`, `Literal["a", "b"]`, `TypedDict` for dict-shaped records, `Protocol` for
# "anything with these methods" (structural typing), `Any` as the escape hatch.
#
# **The crucial fact: Python does not enforce hints when running.**

# %%
def double(x: int) -> int:
    return x * 2

print(double(21), double("ab"), "<- a str sails through an int annotation")

# %% [markdown]
# Hints are checked by **separate tools** (`mypy`, `pyright`, editors like VS Code/PyCharm) that read your code without running it and flag `double("ab")`. They are not installed in this environment, so we do not run them here;
# in a real project you add one to CI. Benefits: documentation that cannot drift as easily as comments, editor autocomplete, and bugs caught before runtime. Cost: some effort on dynamic code; adopt them gradually on function boundaries.
#
# ### Runtime validation with Pydantic
# When data comes from outside (JSON, CSV, API requests) you need checks *at run time*. **Pydantic** turns annotations into validation, which is why FastAPI uses it.

# %%
from pydantic import BaseModel, Field, ValidationError

class Order(BaseModel):
    order_id: int
    quantity: int = Field(gt=0)
    status: Literal["delivered", "shipped", "cancelled", "returned"]

print(Order(order_id="17", quantity=2, status="shipped"))          # "17" is coerced to int 17 (lax mode)
try:
    Order(order_id=1, quantity=0, status="lost")
except ValidationError as e:
    print(e.error_count(), "problems:", [(err["loc"][0], err["type"]) for err in e.errors()])

# %% [markdown]
# Pydantic reported **both** problems at once, with the field names. Note the coercion of `"17"`: convenient, but decide deliberately whether your boundary should be lenient or strict.

# %% [markdown]
# ## 2. Why test, and what makes a good test
# A test is a small program that calls your code with a known input and asserts the known correct output. Good tests are **fast, deterministic, independent, and specific** (when one fails you know what broke).
# Test the behaviour and the edge cases: empty input, zero, negative numbers, one element, the boundaries of your `if` statements, invalid input.
#
# ## 3. A real package with real tests
# We build a tiny project in a temp folder: a module `shop/pricing.py` and tests in `tests/`. `pytest` discovers files named `test_*.py` and functions named `test_*`.

# %%
tmp = tempfile.TemporaryDirectory()
root = Path(tmp.name)
(root / "shop").mkdir()
(root / "tests").mkdir()
(root / "shop" / "__init__.py").write_text("", encoding="utf-8")

pricing_ok = textwrap.dedent('''\
    def discount(price: float, pct: float) -> float:
        """Price after a percentage discount (pct in 0..100)."""
        if not 0 <= pct <= 100:
            raise ValueError(f"pct must be between 0 and 100, got {pct}")
        return price * (1 - pct / 100)


    def margin(price: float, cost: float) -> float:
        return (price - cost) / price
''')
(root / "shop" / "pricing.py").write_text(pricing_ok, encoding="utf-8")

test_src = textwrap.dedent('''\
    import pytest
    from shop.pricing import discount, margin


    def test_no_discount_keeps_price():
        assert discount(100, 0) == 100


    def test_ten_percent():
        assert discount(200, 10) == pytest.approx(180)


    @pytest.mark.parametrize("pct", [-1, 101])
    def test_invalid_percent_raises(pct):
        with pytest.raises(ValueError, match="pct must be between"):
            discount(100, pct)


    @pytest.mark.parametrize("price, cost, expected", [(100, 60, 0.4), (50, 50, 0.0), (80, 100, -0.25)])
    def test_margin(price, cost, expected):
        assert margin(price, cost) == pytest.approx(expected)
''')
(root / "tests" / "test_pricing.py").write_text(test_src, encoding="utf-8")


def run_pytest(cwd):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "--color=no"],
                       cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout


code, out = run_pytest(root)
print("exit code:", code)
print(out.strip())

# %% [markdown]
# *What just happened?* `pytest` found 4 test functions; two were **parametrized**, so they ran once per parameter set (`[-1, 101]` and three margin cases), giving more test cases than functions. The dots are passes; the exit code `0` is what CI checks.
# Things to notice in the test file:
# - Plain `assert`, no assertion methods to memorise; pytest rewrites them to give detailed failure messages.
# - `pytest.approx` for floats (never compare floats with `==` after arithmetic).
# - `pytest.raises(..., match=...)` asserts that the right exception with the right message is raised.
# - `parametrize` replaces copy-pasted tests with a table of cases.
#
# ### Reading a failure
# Now introduce a realistic bug: someone forgets to divide the percentage by 100.

# %%
(root / "shop" / "pricing.py").write_text(pricing_ok.replace("pct / 100", "pct"), encoding="utf-8")
code, out = run_pytest(root)
print("exit code:", code)
print("\n".join(out.strip().splitlines()[:25]))

# %% [markdown]
# Read it like a traceback: which test failed, the line of the assertion, the **actual vs expected values** (`assert -1800 == approx(180)`), and the summary line. Notice `test_no_discount_keeps_price` still
# **passed**, because a 0% discount hides the bug. That is why you test more than one input: a test suite is only as good as its cases. Restore the file:

# %%
(root / "shop" / "pricing.py").write_text(pricing_ok, encoding="utf-8")
print("after the fix, exit code:", run_pytest(root)[0])

# %% [markdown]
# ### Fixtures
# A **fixture** is reusable setup (test data, a temp file, a database connection) injected into tests by naming it as a parameter. `tmp_path` is built in: a fresh temporary folder per test.

# %%
(root / "tests" / "test_files.py").write_text(textwrap.dedent('''\
    import json
    import pytest


    @pytest.fixture
    def orders():
        return [{"id": 1, "status": "delivered"}, {"id": 2, "status": "returned"}]


    def test_roundtrip(tmp_path, orders):
        path = tmp_path / "orders.json"
        path.write_text(json.dumps(orders))
        assert json.loads(path.read_text()) == orders


    def test_count_returned(orders):
        assert sum(o["status"] == "returned" for o in orders) == 1
'''), encoding="utf-8")
code, out = run_pytest(root)
print("exit code:", code, "|", out.strip().splitlines()[-1])

# %% [markdown]
# Practical habits: one behaviour per test, names that describe it (`test_invalid_percent_raises`), no dependence between tests, no network or wall-clock time inside unit tests (pass data in, or fake it),
# and a fast suite you run constantly (`pytest -x` stops at the first failure; `pytest -k name` runs matching tests; `pytest path::test_name` runs one).
# **Test-driven development** means writing the failing test first; even if you do not follow it strictly, writing a test *for every bug you fix* keeps it fixed.

# %% [markdown]
# ## Common mistakes
#
# 1. **Believing hints are enforced.** They are not at run time; use a type checker and/or Pydantic.
# 2. **Comparing floats with `==`** in tests. Use `pytest.approx`.
# 3. **Testing only the happy path** (see the 0% discount that hid the bug).
# 4. **Tests that depend on each other or on order**, on today's date, on the network, or on files in your home folder.
# 5. **Asserting too much in one test**, so a failure does not say what broke.
# 6. **Not running tests automatically.** A suite nobody runs rots; run it in CI and before every commit.

# %% [markdown]
# ## Interview corner
#
# **1. Are Python type hints enforced?**
# <details><summary>Answer</summary>
# No, not by the interpreter. Tools such as mypy and pyright check them statically; libraries such as Pydantic use them for runtime validation. They are documentation plus tooling.
# </details>
#
# **2. How would you test a function that reads a CSV and computes a metric?**
# <details><summary>Answer</summary>
# Write the input to a temp file (<code>tmp_path</code>) or pass a small in-memory structure, assert the metric on a tiny input you can compute by hand, and add edge cases: empty file, missing column, malformed values, duplicates. Keep the computation separate from file I/O so it is easy to test.
# </details>
#
# **3. What are fixtures and parametrization in pytest?**
# <details><summary>Answer</summary>
# Fixtures provide reusable setup/teardown injected by parameter name; parametrization runs one test function against a table of inputs and expected outputs, reporting each case separately.
# </details>
#
# **4. How do you test code with randomness?**
# <details><summary>Answer</summary>
# Seed the generator (or inject it as a parameter), assert properties that hold for any seed (shape, bounds, reproducibility with the same seed), and use tolerances for statistical outcomes instead of exact values.
# </details>

# %% [markdown]
# ## Exercises
# The temp project `root` and the helper `run_pytest(cwd)` exist. Clean-up happens in the last cell.

# %% [markdown]
# ### Exercise 1: Write tests that catch a bug
# `shop/stats.py` (created below) should compute the average order value `total / n_orders` as a float, raising `ZeroDivisionError` when `n_orders == 0`. A buggy version uses integer division. Write `test_source`, the text of a pytest
# test file (import with `from shop.stats import average_order_value`) that **fails on the buggy version and passes on the correct version**. Include at least two tests.

# %%
good_stats = "def average_order_value(total, n_orders):\n    return total / n_orders\n"
bad_stats = "def average_order_value(total, n_orders):\n    return total // n_orders\n"

# %% tags=["starter"]
test_source = """
from shop.stats import average_order_value

# write your tests here
"""

# %%
def _run_with(stats_src):
    (root / "shop" / "stats.py").write_text(stats_src, encoding="utf-8")
    (root / "tests" / "test_stats_ex1.py").write_text(test_source, encoding="utf-8")
    return run_pytest(root)

code_bad, out_bad = _run_with(bad_stats)
code_good, out_good = _run_with(good_stats)
assert code_good == 0, f"your tests must PASS on the correct implementation:\n{out_good[-600:]}"
assert code_bad != 0, "your tests did not notice the integer-division bug: add a case where the result is not a whole number"
assert test_source.count("def test_") >= 2, "write at least two test functions"
(root / "tests" / "test_stats_ex1.py").unlink()
print("Exercise 1 passed: fails on the bug, passes on the fix")

# %% tags=["solution"]
test_source = """
import pytest
from shop.stats import average_order_value


def test_whole_number_average():
    assert average_order_value(100, 4) == 25


def test_fractional_average():
    assert average_order_value(10, 4) == pytest.approx(2.5)


def test_zero_orders_raises():
    with pytest.raises(ZeroDivisionError):
        average_order_value(10, 0)
"""

# %% [markdown]
# ### Exercise 2: Annotate a function
# Write `top_n(items, n=3)` with type hints `list[str]`, `int`, return `list[str]`. It returns the `n` most frequent strings, most frequent first (ties: alphabetical). Use `collections.Counter`.
# The check inspects the hints with `get_type_hints`.

# %% tags=["starter"]
def top_n(items, n=3):
    ...

# %%
hints = get_type_hints(top_n)
assert hints == {"items": list[str], "n": int, "return": list[str]}, f"annotations are {hints}"
assert top_n(["b", "a", "b", "c", "a", "b"], 2) == ["b", "a"]
assert top_n(["x", "y"], 5) == ["x", "y"], "ties alphabetical, n larger than distinct count"
assert top_n([]) == []
print("Exercise 2 passed")

# %% tags=["solution"]
from collections import Counter

def top_n(items: list[str], n: int = 3) -> list[str]:
    counts = Counter(items)
    ordered = sorted(counts, key=lambda k: (-counts[k], k))
    return ordered[:n]

# %% [markdown]
# ### Exercise 3: Validate incoming data with Pydantic
# Define `Customer(BaseModel)` with `customer_id: int`, `age: int` (between 18 and 120 inclusive), `country: str` (exactly 2 characters) and `is_premium: bool = False`.
# Store in `n_valid` how many rows of `../datasets/customers.csv` validate (pass the CSV row dict straight in; it will coerce strings), and in `n_invalid` how many do not.

# %% tags=["starter"]
class Customer(BaseModel):
    ...

n_valid = ...
n_invalid = ...

# %%
import csv
with open("../datasets/customers.csv", encoding="utf-8", newline="") as f:
    crows = list(csv.DictReader(f))
assert n_valid + n_invalid == len(crows)
want_valid = sum(1 for r in crows if 18 <= int(r["age"]) <= 120 and len(r["country"]) == 2)
assert n_valid == want_valid, f"expected {want_valid} valid rows, got {n_valid}"
assert Customer(customer_id="1", age="30", country="CA").is_premium is False
for bad in [dict(customer_id=1, age=17, country="CA"), dict(customer_id=1, age=30, country="CAN")]:
    try:
        Customer(**bad)
        raise AssertionError(f"{bad} should be invalid")
    except ValidationError:
        pass
print(f"Exercise 3 passed: {n_valid} valid, {n_invalid} invalid")

# %% tags=["solution"]
class Customer(BaseModel):
    customer_id: int
    age: int = Field(ge=18, le=120)
    country: str = Field(min_length=2, max_length=2)
    is_premium: bool = False

import csv as _csv
with open("../datasets/customers.csv", encoding="utf-8", newline="") as _f:
    _rows = list(_csv.DictReader(_f))
n_valid = n_invalid = 0
for _r in _rows:
    try:
        Customer(**_r)
        n_valid += 1
    except ValidationError:
        n_invalid += 1

# %% [markdown]
# ### Exercise 4 (open-ended): Test-drive a function
# Choose a small function from an earlier notebook (for example `parse_price`, `to_hms`, or `clean_name`). Write its pytest file in a string with at least four cases including edge cases and one invalid input, run it with `run_pytest`
# in the temp project, then deliberately break the function and confirm your tests notice. Write down which case caught the bug.

# %%
tmp.cleanup()
print("temp project removed:", not root.exists())

# %% [markdown]
# ## Recap
# - Type hints (`list[int]`, `X | None`, `Callable`, `TypedDict`, `Literal`) document contracts but are **not** enforced at run time; use mypy/pyright statically and Pydantic for runtime validation.
# - pytest: name files `test_*.py` and functions `test_*`; plain `assert`; `approx` for floats; `raises` for errors; `parametrize` for tables; fixtures and `tmp_path` for setup.
# - Test edge cases, not just the happy path: a bug can hide behind an input like a 0% discount.
# - Write a test for every bug you fix; run the suite automatically in CI.
#
# **Go deeper:** the pytest documentation (docs.pytest.org), the `typing` docs and the Pydantic documentation (docs.pydantic.dev).
#
# **Next:** [Python for data: a practice project](13_python_for_data_practice.ipynb)
