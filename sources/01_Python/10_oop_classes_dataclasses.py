# %% [markdown]
# # OOP: Classes and Dataclasses
#
# **Why this matters.** scikit-learn estimators, PyTorch modules, pandas DataFrames and Pydantic models are all classes. To read their source, extend them (a custom transformer, a `Dataset`) or model your own domain cleanly you need
# to know how classes, inheritance and special methods work. You also need judgment: many data tasks are better as plain functions, and `@dataclass` removes most class boilerplate.
#
# **You will learn**
# - define classes, `__init__`, methods, `self`, `__repr__`/`__str__`, properties
# - class attributes vs instance attributes (and the shared mutable trap)
# - inheritance, `super()`, and why composition is often better
# - special ("dunder") methods: `__len__`, `__iter__`, `__eq__`, `__lt__`
# - `@dataclass`: frozen, defaults, `field(default_factory=...)`, `__post_init__`
#
# **Prerequisites:** [The standard library essentials](09_stdlib_essentials.ipynb).
#
# > Data used: `datasets/products.csv` (synthetic; see `docs/DATASETS.md`).

# %%
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATA = Path("../datasets")
with open(DATA / "products.csv", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
print(rows[0])

# %% [markdown]
# ## 1. A first class
# A class bundles **data** (attributes) and **behaviour** (methods). `__init__` runs when you create an instance; `self` is the instance, passed automatically as the first argument.

# %%
class Product:
    def __init__(self, product_id, name, price, cost):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.cost = cost

    def margin(self):
        return (self.price - self.cost) / self.price

p = Product(1, "Headphones Basic", 470.7, 340.16)
print(p)                       # default repr is unhelpful
print(f"{p.margin():.1%}", p.name)

# %% [markdown]
# `print(p)` shows an address. Define **`__repr__`** (for developers: unambiguous, ideally `eval`-able-looking) and optionally **`__str__`** (for users). Also make the margin a **property**: computed like an
# attribute, accessed without parentheses.

# %%
class Product:
    def __init__(self, product_id, name, price, cost):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.cost = cost

    @property
    def margin(self):
        return (self.price - self.cost) / self.price

    def __repr__(self):
        return f"Product({self.product_id!r}, {self.name!r}, price={self.price})"

p = Product(1, "Headphones Basic", 470.7, 340.16)
print(repr(p), f"| margin {p.margin:.1%}")

# %% [markdown]
# *What just happened?* `@property` turns a method into a read-only computed attribute, so callers do not care whether a value is stored or calculated (and you can add validation later without changing callers).

# %% [markdown]
# ## 2. Class attributes vs instance attributes
# Attributes set in `__init__` belong to each instance. Attributes defined in the class body belong to the **class** and are shared by all instances. Shared is fine for constants; for mutable values it is a classic bug.

# %%
class BadBasket:
    items = []                 # ONE list shared by every instance

    def add(self, x):
        self.items.append(x)

a, b = BadBasket(), BadBasket()
a.add("pen")
print("b sees a's item:", b.items, "| same list:", a.items is b.items)

class Basket:
    TAX_RATE = 0.2             # class attribute as a constant: fine

    def __init__(self):
        self.items = []        # per-instance state belongs in __init__

    def add(self, x):
        self.items.append(x)

a, b = Basket(), Basket()
a.add("pen")
print("b is independent:", b.items, "| class constant:", Basket.TAX_RATE)

# %% [markdown]
# ## 3. Special methods: make your class feel built in
# Double-underscore ("dunder") methods let your objects work with `len`, `for`, `==`, `sorted`, `in`.

# %%
class Cart:
    def __init__(self, products=()):
        self._products = list(products)

    def add(self, product):
        self._products.append(product)

    def __len__(self):
        return len(self._products)

    def __iter__(self):
        return iter(self._products)

    def __getitem__(self, i):
        return self._products[i]

    def __contains__(self, product):
        return product in self._products

    def total(self):
        return sum(p.price for p in self)

cart = Cart([Product(1, "A", 10.0, 6.0), Product(2, "B", 25.5, 20.0)])
print(len(cart), [p.name for p in cart], cart[1].name, round(cart.total(), 2))

# %% [markdown]
# A leading underscore (`_products`) signals "internal, do not touch" by convention; Python does not enforce privacy. Equality and ordering work the same way via `__eq__` and `__lt__`, and
# **defining `__eq__` without `__hash__` makes the class unhashable**. This is easy to forget, and it is one reason dataclasses are so handy.

# %% [markdown]
# ## 4. Inheritance and `super()`
# A subclass reuses and specialises a parent. `super()` calls the parent's version.

# %%
class DiscountedProduct(Product):
    def __init__(self, product_id, name, price, cost, discount):
        super().__init__(product_id, name, price, cost)
        self.discount = discount

    @property
    def margin(self):                              # override
        sale_price = self.price * (1 - self.discount)
        return (sale_price - self.cost) / sale_price

base = Product(1, "Keyboard", 100.0, 60.0)
sale = DiscountedProduct(1, "Keyboard", 100.0, 60.0, discount=0.25)
print(f"{base.margin:.1%} -> {sale.margin:.1%}", "| isinstance:", isinstance(sale, Product), issubclass(DiscountedProduct, Product))

# %% [markdown]
# **Prefer composition to deep inheritance.** Inheritance says "is a"; it couples the child to the parent's internals. If the relationship is "has a" (a `Cart` has products, a `Model` has a `Scaler`), hold the object
# as an attribute instead. Real libraries mostly use shallow inheritance (subclass `BaseEstimator`, `nn.Module`) to plug into a framework; inside your own code, keep hierarchies flat.

# %% [markdown]
# ## 5. Dataclasses: the modern default for data holders
# `@dataclass` generates `__init__`, `__repr__` and `__eq__` from annotated fields.

# %%
@dataclass
class Item:
    product_id: int
    name: str
    price: float
    cost: float
    tags: list[str] = field(default_factory=list)       # mutable default requires default_factory

    @property
    def margin(self):
        return (self.price - self.cost) / self.price

item = Item(1, "Headphones Basic", 470.7, 340.16)
print(item)
print(item == Item(1, "Headphones Basic", 470.7, 340.16), "|", asdict(item)["name"], "|", f"{item.margin:.1%}")

try:
    @dataclass
    class Bad:
        tags: list = []
except ValueError as e:
    print("ValueError:", e)

# %% [markdown]
# Useful options:
# - `@dataclass(frozen=True)`: instances are immutable and hashable (usable as dict keys / set members).
# - `@dataclass(order=True)`: adds `<`, `>` comparing fields in order.
# - `__post_init__`: run validation or derived fields after the generated `__init__`.
# - Annotations are **not enforced** at run time (`Item("x", 1, 2, 3)` is accepted); Pydantic, used later in the roadmap, validates types.

# %%
@dataclass(frozen=True, order=True)
class Price:
    amount: float
    currency: str = "USD"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError(f"negative price: {self.amount}")

cheap, dear = Price(9.99), Price(19.5)
print(cheap < dear, {cheap: "can be a dict key"})
try:
    cheap.amount = 1
except Exception as e:
    print(type(e).__name__, "-", e)

# %% [markdown]
# Building objects from CSV rows, converting types in one place:

# %%
catalogue = [Item(int(r["product_id"]), r["product_name"], float(r["unit_price"]), float(r["unit_cost"])) for r in rows]
best = max(catalogue, key=lambda it: it.margin)
print(len(catalogue), "items; best margin:", best.name, f"{best.margin:.1%}")

# %% [markdown]
# ## Common mistakes
#
# 1. **Mutable class attributes** shared across instances (`items = []` in the class body).
# 2. **Forgetting `self`** in a method definition or call (`TypeError: takes 1 positional argument but 2 were given`).
# 3. **Mutable default in a dataclass field** without `default_factory` (Python refuses, as shown above).
# 4. **Defining `__eq__` and expecting hashing to work.**
# 5. **Overusing classes:** a class with one method and no state is a function.
# 6. **Deep inheritance trees** where a simple attribute would do.

# %%
class NoSelf:
    def greet():               # forgot self
        return "hi"

try:
    NoSelf().greet()
except TypeError as e:
    print("TypeError:", e)

class WithEq:
    def __init__(self, v): self.v = v
    def __eq__(self, other): return self.v == other.v

try:
    {WithEq(1)}
except TypeError as e:
    print("TypeError:", e)

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between <code>__str__</code> and <code>__repr__</code>?**
# <details><summary>Answer</summary>
# <code>__repr__</code> is for developers (unambiguous, used in the REPL and in containers); <code>__str__</code> is the friendly form used by <code>print</code>/<code>str()</code>. If only <code>__repr__</code> is defined, <code>str()</code> falls back to it.
# </details>
#
# **2. Class attribute vs instance attribute?**
# <details><summary>Answer</summary>
# Class attributes live on the class and are shared by all instances (and found by attribute lookup if the instance has none); instance attributes are set on <code>self</code> and are per object. Mutable class attributes are a common bug.
# </details>
#
# **3. Inheritance or composition?**
# <details><summary>Answer</summary>
# Use inheritance for a true "is a" relationship or to plug into a framework base class; use composition ("has a") otherwise. Composition keeps coupling low and is easier to change and test.
# </details>
#
# **4. Why use a dataclass?**
# <details><summary>Answer</summary>
# It generates <code>__init__</code>, <code>__repr__</code>, <code>__eq__</code> (and optionally ordering, hashing, immutability) from field annotations, cutting boilerplate and bugs for plain data holders.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: A `Product` dataclass with a computed margin
# Define a frozen dataclass `Prod` with fields `product_id: int`, `name: str`, `category: str`, `price: float`, `cost: float`, and a property `margin` (`(price - cost) / price`). Build `prods`, a list of `Prod` from all CSV `rows`, and store `top_category`:
# the category whose **average margin** is highest.

# %% tags=["starter"]
@dataclass(frozen=True)
class Prod:
    ...

prods = ...
top_category = ...

# %%
assert len(prods) == len(rows) and isinstance(prods[0], Prod)
p0 = prods[0]
assert p0.product_id == int(rows[0]["product_id"]) and abs(p0.margin - (p0.price - p0.cost) / p0.price) < 1e-12
try:
    p0.price = 1
    raise AssertionError("Prod must be frozen")
except Exception as e:
    assert "assign" in str(e).lower() or "frozen" in str(e).lower()
avg = {}
for r in rows:
    avg.setdefault(r["category"], []).append((float(r["unit_price"]) - float(r["unit_cost"])) / float(r["unit_price"]))
want = max(avg, key=lambda c: sum(avg[c]) / len(avg[c]))
assert top_category == want, f"expected {want}"
print("Exercise 1 passed:", top_category)

# %% tags=["solution"]
@dataclass(frozen=True)
class Prod:
    product_id: int
    name: str
    category: str
    price: float
    cost: float

    @property
    def margin(self):
        return (self.price - self.cost) / self.price

prods = [Prod(int(r["product_id"]), r["product_name"], r["category"], float(r["unit_price"]), float(r["unit_cost"])) for r in rows]
_by_cat = {}
for pr in prods:
    _by_cat.setdefault(pr.category, []).append(pr.margin)
top_category = max(_by_cat, key=lambda c: sum(_by_cat[c]) / len(_by_cat[c]))

# %% [markdown]
# ### Exercise 2: A container that behaves like a collection
# Write class `Catalogue` taking a list of `Prod`. It must support `len(c)`, iteration, `c["Books"]`-style lookup **by category** returning a list of the products in that category (empty list if none), and a method `value()` returning the sum of prices.
# Use `__len__`, `__iter__`, `__getitem__`.

# %% tags=["starter"]
class Catalogue:
    ...

# %%
cat = Catalogue(prods)
assert len(cat) == len(prods)
assert list(cat) == prods
assert cat["Books"] == [p for p in prods if p.category == "Books"] and cat["Toys"] == []
assert abs(cat.value() - sum(p.price for p in prods)) < 1e-9
print("Exercise 2 passed")

# %% tags=["solution"]
class Catalogue:
    def __init__(self, products):
        self._products = list(products)

    def __len__(self):
        return len(self._products)

    def __iter__(self):
        return iter(self._products)

    def __getitem__(self, category):
        return [p for p in self._products if p.category == category]

    def value(self):
        return sum(p.price for p in self._products)

# %% [markdown]
# ### Exercise 3: Inherit and override
# Write `Clearance(Prod)`-style behaviour without touching `Prod`: a subclass `Clearance` (a normal or frozen dataclass subclass is fine) adding a field `discount: float = 0.3` and overriding `margin` so it is computed on the **discounted** price.
# Create `c = Clearance(99, "Old Phone", "Electronics", 200.0, 150.0)` and check it.

# %% tags=["starter"]
@dataclass(frozen=True)
class Clearance(Prod):
    ...

c = ...

# %%
assert isinstance(c, Prod) and c.discount == 0.3
sale = 200.0 * (1 - 0.3)
assert abs(c.margin - (sale - 150.0) / sale) < 1e-12, "margin must use the discounted price"
assert c.margin < Prod(99, "x", "y", 200.0, 150.0).margin
print("Exercise 3 passed: margin", round(c.margin, 3))

# %% tags=["solution"]
@dataclass(frozen=True)
class Clearance(Prod):
    discount: float = 0.3

    @property
    def margin(self):
        sale = self.price * (1 - self.discount)
        return (sale - self.cost) / sale

c = Clearance(99, "Old Phone", "Electronics", 200.0, 150.0)

# %% [markdown]
# ### Exercise 4: Fix the shared state
# `Counter1` below keeps its counts in a class attribute, so two counters interfere. Write `Tally` with the same interface (`add(key)`, `get(key)` returning 0 for unknown keys) that keeps state per instance.

# %%
class Counter1:
    counts = {}
    def add(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
    def get(self, key):
        return self.counts.get(key, 0)

# %% tags=["starter"]
class Tally:
    ...

# %%
t1, t2 = Tally(), Tally()
t1.add("x"); t1.add("x"); t2.add("y")
assert t1.get("x") == 2 and t2.get("x") == 0 and t2.get("y") == 1 and t1.get("zzz") == 0
print("Exercise 4 passed")

# %% tags=["solution"]
class Tally:
    def __init__(self):
        self.counts = {}

    def add(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1

    def get(self, key):
        return self.counts.get(key, 0)

# %% [markdown]
# ### Exercise 5 (open-ended): Class or function?
# Look at your `Catalogue`. List which of its methods really need shared state and which could be plain functions taking a list of products. Decide what you would keep as a class in a real project and justify it in two sentences.

# %% [markdown]
# ## Recap
# - A class bundles state and behaviour; `__init__` sets per-instance state; `@property` exposes computed attributes; define `__repr__`.
# - Class attributes are shared: fine for constants, a bug for mutable state.
# - Dunder methods (`__len__`, `__iter__`, `__getitem__`, `__eq__`) integrate classes with the language; `__eq__` without `__hash__` removes hashability.
# - `super()` calls the parent; prefer composition and shallow hierarchies.
# - `@dataclass` (with `frozen`, `order`, `default_factory`, `__post_init__`) is the default for data holders; annotations are not enforced at run time.
#
# **Go deeper:** the Python tutorial chapter *Classes*, the `dataclasses` docs, and the *Data model* chapter of the language reference (docs.python.org/3).
#
# **Next:** [Iterators, decorators and context managers](11_iterators_decorators_context_managers.ipynb)
