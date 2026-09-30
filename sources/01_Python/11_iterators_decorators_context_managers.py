# %% [markdown]
# # Iterators, Decorators and Context Managers
#
# **Why this matters.** These three protocols are how Python libraries give you clean APIs: `for` over a PyTorch `DataLoader` (iterator), `@app.get("/")` in FastAPI and `@pytest.fixture` (decorators), and
# `with open(...)`, `with torch.no_grad()`, `with mlflow.start_run()` (context managers). Understanding the mechanism lets you read that code and write your own timing, retry, caching and cleanup helpers.
#
# **You will learn**
# - the iterator protocol (`__iter__`/`__next__`), and iterable vs iterator
# - decorators: functions that wrap functions, `functools.wraps`, decorators with arguments
# - context managers: `__enter__`/`__exit__` and `contextlib.contextmanager`
# - practical recipes: timing, call counting, retrying, changing directory safely
#
# **Prerequisites:** [OOP, classes and dataclasses](10_oop_classes_dataclasses.ipynb) and [Functions](05_functions.ipynb) (closures).

# %%
import functools
import os
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path

# %% [markdown]
# ## 1. The iterator protocol
# - An **iterable** has `__iter__()` returning an iterator (lists, dicts, files, generators).
# - An **iterator** has `__next__()` returning the next item and raising `StopIteration` at the end, and its `__iter__()` returns itself.
#
# A `for` loop is just this, written by the language:

# %%
nums = [10, 20, 30]
it = iter(nums)                   # ask the iterable for an iterator
print(next(it), next(it), next(it))
try:
    next(it)
except StopIteration:
    print("StopIteration: the for loop catches this to know when to stop")

# the for loop, unrolled
it = iter(nums)
while True:
    try:
        x = next(it)
    except StopIteration:
        break
    print("got", x)

# %% [markdown]
# Write your own iterator as a class when you need custom state, though a generator function is usually shorter:

# %%
class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self                      # an iterator returns itself

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

print(list(Countdown(4)))
c = Countdown(3)
print(list(c), list(c), "<- an iterator is exhausted after one pass")

def countdown_gen(start):
    while start > 0:
        yield start
        start -= 1

print(list(countdown_gen(4)))

# %% [markdown]
# *What just happened?* `Countdown` is an iterator, so it is single-use. A **re-iterable** container (like `list`) returns a *fresh* iterator on each `iter()` call, which is why you can loop over a list twice.
# Two more useful forms: `iter(callable, sentinel)` calls the function until it returns the sentinel, and `enumerate`, `zip`, `map`, `filter` all return iterators.

# %%
import random
rng = random.Random(42)
rolls = list(iter(lambda: rng.randint(1, 6), 6))       # roll until a 6 appears
print("rolled until first six:", rolls)

# %% [markdown]
# ## 2. Decorators
# A **decorator** is a function that takes a function and returns a (usually wrapped) function. The `@name` line above a `def` is shorthand for `f = name(f)`.

# %%
def shout(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return str(result).upper()
    return wrapper

@shout
def greet(name):
    """Say hello."""
    return f"hello {name}"

print(greet("ana"))
print("name now:", greet.__name__, "| doc:", greet.__doc__)      # metadata of the original was lost

# %% [markdown]
# The wrapper replaced the function, so `greet.__name__` is now `"wrapper"` and the docstring is gone, which breaks help, debuggers and frameworks that inspect functions. Fix with **`functools.wraps`**:

# %%
def shout(func):
    @functools.wraps(func)                # copy __name__, __doc__, etc. from func onto wrapper
    def wrapper(*args, **kwargs):
        return str(func(*args, **kwargs)).upper()
    return wrapper

@shout
def greet(name):
    """Say hello."""
    return f"hello {name}"

print(greet("ana"), "|", greet.__name__, "|", greet.__doc__)

# %% [markdown]
# Two everyday decorators: **timing** and **memoising**.

# %%
def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            wrapper.last_seconds = time.perf_counter() - start     # store on the wrapper so callers can read it
    return wrapper

@timed
def slow_sum(n):
    return sum(range(n))

slow_sum(1_000_000)
print(f"last call took {slow_sum.last_seconds:.4f}s (varies by machine)")

def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    return wrapper

@memoize
def square(n):
    print("  computing", n)
    return n * n

print(square(4), square(4), len(square.cache))

# %% [markdown]
# ### Decorators with arguments
# `@retry(times=3)` means *call* `retry(times=3)` first, and it must **return the actual decorator**. That is three nested functions: factory, decorator, wrapper.

# %%
def retry(times=3, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == times:
                        raise
        return wrapper
    return decorator

attempts = {"n": 0}

@retry(times=3, exceptions=(ConnectionError,))
def flaky():
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ConnectionError("try again")
    return f"ok after {attempts['n']} attempts"

print(flaky())

# %% [markdown]
# Stacked decorators apply bottom-up (closest to the function first). Decorated code behaves the same as plain calls *plus* the extra behaviour, which is exactly why a decorator must forward `*args, **kwargs` and the return value.

# %% [markdown]
# ## 3. Context managers
# A **context manager** runs setup on entering a `with` block and guaranteed cleanup on leaving it, whether normally or by exception. The protocol is two methods:

# %%
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self                               # bound by "as t"

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        return False                              # False/None: do not swallow exceptions

with Timer() as t:
    total = sum(range(500_000))
print(f"block took {t.elapsed:.4f}s (varies)")

try:
    with Timer() as t2:
        raise ValueError("inside the block")
except ValueError:
    print("exception propagated; cleanup still ran, elapsed set:", hasattr(t2, "elapsed"))

# %% [markdown]
# `__exit__` receives the exception details; returning a truthy value **swallows** the exception (rarely what you want). For simple cases write a generator function with `@contextmanager`: code before `yield` is setup,
# after it is cleanup (put it in `finally` so it runs on errors).

# %%
@contextmanager
def working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield Path(path)
    finally:
        os.chdir(previous)                        # always restore

here = os.getcwd()
with tempfile.TemporaryDirectory() as d:
    with working_directory(d) as p:
        print("inside :", Path.cwd() == Path(d).resolve() or Path.cwd() == p)
    print("outside restored:", os.getcwd() == here)

with suppress(FileNotFoundError):                 # stdlib helper: ignore ONE specific expected exception
    Path("does-not-exist.txt").unlink()
print("suppress swallowed the FileNotFoundError")

# %% [markdown]
# Real uses: files, locks, database transactions (commit on success, rollback on error), temporary files, `torch.no_grad()`, timing blocks, changing settings temporarily. Whenever you write "do A, then always undo A", reach for a context manager.

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting `functools.wraps`**, which breaks introspection, docs and some frameworks.
# 2. **A wrapper that drops the return value or arguments:** always `return func(*args, **kwargs)`.
# 3. **Forgetting the parentheses** for a decorator factory (`@retry` instead of `@retry()`), which passes the function as `times`.
# 4. **Iterating an iterator twice** and getting nothing the second time.
# 5. **Cleanup in a `@contextmanager` not in `finally`:** skipped when the block raises.
# 6. **Returning `True` from `__exit__`** and silently swallowing every error.

# %%
try:
    @retry
    def oops():
        return 1
    oops()
except TypeError as e:
    print("forgot the parentheses ->", type(e).__name__)

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between an iterable and an iterator?**
# <details><summary>Answer</summary>
# An iterable can produce iterators via <code>__iter__</code>; an iterator maintains position and implements <code>__next__</code>, returning itself from <code>__iter__</code>. Lists are iterables, each <code>iter(list)</code> is a new iterator; iterators are single pass.
# </details>
#
# **2. What does a decorator do, and why <code>functools.wraps</code>?**
# <details><summary>Answer</summary>
# It takes a function and returns a replacement, typically adding behaviour (timing, caching, auth, retry). <code>wraps</code> copies the original's name, docstring and other metadata onto the wrapper so introspection and debugging still work.
# </details>
#
# **3. Explain <code>with</code> and context managers.**
# <details><summary>Answer</summary>
# <code>with</code> calls <code>__enter__</code> on entry and <code>__exit__</code> on exit, even when an exception is raised, guaranteeing cleanup (closing files, releasing locks, rolling back transactions). Use <code>contextlib.contextmanager</code> for generator-based ones.
# </details>
#
# **4. Write a decorator that counts calls.**
# <details><summary>Answer</summary>
# Wrap with <code>@functools.wraps</code>, increment <code>wrapper.calls += 1</code> on each call (initialise <code>wrapper.calls = 0</code>), and forward <code>*args, **kwargs</code> and the result. (That is Exercise 2.)
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A range iterator class
# Write class `Evens(limit)` that is an **iterator** yielding the even numbers `0, 2, 4, ...` strictly below `limit`, implemented with `__iter__` and `__next__` (no generator).

# %% tags=["starter"]
class Evens:
    ...

# %%
assert list(Evens(10)) == [0, 2, 4, 6, 8]
assert list(Evens(0)) == [] and list(Evens(1)) == [0]
e = Evens(5)
assert iter(e) is e, "an iterator returns itself from __iter__"
assert next(e) == 0
assert list(e) == [2, 4] and list(e) == [], "iterator is exhausted after one pass"
print("Exercise 1 passed")

# %% tags=["solution"]
class Evens:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration
        value = self.current
        self.current += 2
        return value

# %% [markdown]
# ### Exercise 2: Count the calls
# Write decorator `count_calls` so a decorated function has an attribute `.calls` that starts at 0 and increases by one per call. It must preserve the function's name and docstring and pass through arguments and return values.

# %% tags=["starter"]
def count_calls(func):
    ...

# %%
@count_calls
def add(a, b=0):
    """Add two numbers."""
    return a + b

assert add.calls == 0
assert add(1, 2) == 3 and add(5, b=1) == 6
assert add.calls == 2, f"expected 2 calls, got {add.calls}"
assert add.__name__ == "add" and add.__doc__ == "Add two numbers.", "use functools.wraps"
print("Exercise 2 passed")

# %% tags=["solution"]
def count_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

# %% [markdown]
# ### Exercise 3: A decorator with an argument
# Write `validate_positive(argname)`: a decorator factory. The decorated function must raise `ValueError` (message mentioning the argument name) when the keyword argument called `argname` is `<= 0`; otherwise call through.
# Apply it to `def discount_price(price, *, rate)` with `@validate_positive("rate")` returning `price * (1 - rate)`.

# %% tags=["starter"]
def validate_positive(argname):
    ...

@validate_positive("rate")
def discount_price(price, *, rate):
    return price * (1 - rate)

# %%
assert abs(discount_price(100, rate=0.2) - 80) < 1e-9
for bad in (0, -0.1):
    try:
        discount_price(100, rate=bad)
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "rate" in str(e)
assert discount_price.__name__ == "discount_price"
print("Exercise 3 passed")

# %% tags=["solution"]
def validate_positive(argname):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if argname in kwargs and kwargs[argname] <= 0:
                raise ValueError(f"{argname} must be positive, got {kwargs[argname]}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_positive("rate")
def discount_price(price, *, rate):
    return price * (1 - rate)

# %% [markdown]
# ### Exercise 4: A context manager that cleans up
# Write `temp_file(text)` as a `@contextmanager` that creates a temporary **file** containing `text`, yields its `Path`, and **deletes the file afterwards even if the block raises**.

# %% tags=["starter"]
@contextmanager
def temp_file(text):
    ...

# %%
with temp_file("hello") as path:
    assert path.read_text() == "hello" and path.exists()
    saved = path
assert not saved.exists(), "file must be deleted after the with block"
try:
    with temp_file("x") as path2:
        saved2 = path2
        raise RuntimeError("boom")
except RuntimeError:
    pass
assert not saved2.exists(), "file must be deleted even when the block raises"
print("Exercise 4 passed")

# %% tags=["solution"]
@contextmanager
def temp_file(text):
    fd, name = tempfile.mkstemp(suffix=".txt")
    path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        yield path
    finally:
        path.unlink(missing_ok=True)

# %% [markdown]
# ### Exercise 5 (open-ended): Design a decorator for your own pipeline
# Pick a repeated concern from your work (logging the shape of a DataFrame between steps, timing each step, caching a slow query to disk). Sketch the decorator: what it wraps, what state it keeps, what it must forward,
# and what could go wrong if the wrapped function has side effects. Implement it in a cell and test on a small function.

# %% [markdown]
# ## Recap
# - `for` = `iter()` + repeated `next()` until `StopIteration`; iterators are single-use, containers are re-iterable.
# - A decorator is `f = deco(f)`; always use `functools.wraps`, forward `*args, **kwargs` and the result; a decorator with arguments is a factory returning a decorator.
# - `with` guarantees cleanup via `__enter__`/`__exit__`; `@contextmanager` does it with `yield` and `try/finally`.
# - Timing, caching, retry, temp resources and directory changes are the everyday uses.
#
# **Go deeper:** the docs for `functools`, `contextlib`, and the *Iterator Types* section of the standard library reference (docs.python.org/3); PEP 318 (decorators).
#
# **Next:** [Testing, typing and pytest](12_testing_typing_pytest.ipynb)
