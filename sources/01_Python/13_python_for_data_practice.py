# %% [markdown]
# # Python for Data: A Practice Project
#
# **Why this matters.** Before you reach for pandas, you should be able to answer a business question with nothing but the standard library. It proves you understand what a join, a group-by and an aggregation actually do,
# and it is exactly the skill tested in "no pandas allowed" screens. It also gives you a ground truth to check later tools against.
#
# **You will learn**
# - load related CSV tables, convert types, and validate them (row counts, missing references)
# - join tables with dictionaries, aggregate with `defaultdict`/`Counter`
# - answer: revenue by category, top customers, monthly trend, return rates
# - cross-check your result with an independent method (SQL on `shop.db`) before trusting it
#
# **Prerequisites:** all previous notebooks in this module (especially [stdlib essentials](09_stdlib_essentials.ipynb) and [comprehensions](06_comprehensions_and_generators.ipynb)).
#
# > **The scenario.** You are the first analyst at an online shop. The manager asks: *Which categories make the money? Who are our best customers? Is revenue seasonal? Which channel has the worst returns?*
# > Data: `datasets/orders.csv`, `order_items.csv`, `products.csv`, `customers.csv` (synthetic; see `docs/DATASETS.md`).
# >
# > **Business rule we adopt:** *revenue* counts only orders with status `delivered` or `shipped`. Cancelled and returned orders earn nothing. (Stating the rule explicitly is part of the job; a different rule changes every number.)

# %%
import csv
import sqlite3
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

DATA = Path("../datasets")


def read_csv(name):
    with open(DATA / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


orders = [{**r, "order_id": int(r["order_id"]), "customer_id": int(r["customer_id"]), "order_date": date.fromisoformat(r["order_date"])}
          for r in read_csv("orders.csv")]
items = [{"order_id": int(r["order_id"]), "product_id": int(r["product_id"]), "quantity": int(r["quantity"])} for r in read_csv("order_items.csv")]
products = {int(r["product_id"]): {"name": r["product_name"], "category": r["category"],
                                   "price": float(r["unit_price"]), "cost": float(r["unit_cost"])} for r in read_csv("products.csv")}
customers = {int(r["customer_id"]): {"name": r["first_name"], "country": r["country"], "premium": r["is_premium"] == "True"} for r in read_csv("customers.csv")}
print(len(orders), "orders |", len(items), "lines |", len(products), "products |", len(customers), "customers")

# %% [markdown]
# ## Step 1: Validate before you analyse
# Never start aggregating before you know the tables are consistent. Every order line should reference an existing order and product, and every order an existing customer.

# %%
order_by_id = {o["order_id"]: o for o in orders}
assert len(order_by_id) == len(orders), "duplicate order ids"
orphan_lines = sum(1 for it in items if it["order_id"] not in order_by_id)
unknown_products = sum(1 for it in items if it["product_id"] not in products)
unknown_customers = sum(1 for o in orders if o["customer_id"] not in customers)
orders_without_lines = len(set(order_by_id) - {it["order_id"] for it in items})
print("lines without an order:", orphan_lines, "| unknown products:", unknown_products, "| unknown customers:", unknown_customers)
print("orders with no lines  :", orders_without_lines)
print("date range:", min(o["order_date"] for o in orders), "->", max(o["order_date"] for o in orders))
print("status mix:", dict(Counter(o["status"] for o in orders)))

# %% [markdown]
# Everything joins cleanly: no orphan lines, no unknown products or customers, and every order has at least one line (the printed zeros prove it). Findings like these go in your notes: they are the difference between "I checked" and "I hope".

# %% [markdown]
# ## Step 2: The join
# A **join** matches rows from two tables on a key. With dictionaries, a lookup replaces the search: build `{key: row}` once, then look up per row (O(1) each). We compute each line's revenue and attach the order and product facts.

# %%
COUNTED = {"delivered", "shipped"}

lines = []
for it in items:
    order = order_by_id[it["order_id"]]
    prod = products[it["product_id"]]
    lines.append({
        "order_id": it["order_id"],
        "customer_id": order["customer_id"],
        "month": order["order_date"].strftime("%Y-%m"),
        "channel": order["channel"],
        "status": order["status"],
        "category": prod["category"],
        "product": prod["name"],
        "revenue": it["quantity"] * prod["price"],
        "gross_profit": it["quantity"] * (prod["price"] - prod["cost"]),
    })
counted_lines = [l for l in lines if l["status"] in COUNTED]
total_revenue = sum(l["revenue"] for l in counted_lines)
print(f"{len(counted_lines)} counted lines, total revenue {total_revenue:,.2f}")
print(lines[0])

# %% [markdown]
# ## Step 3: Revenue by category (group-by)
# A group-by is: split rows by a key, then reduce each group. `defaultdict(float)` is the smallest implementation.

# %%
by_category = defaultdict(float)
for l in counted_lines:
    by_category[l["category"]] += l["revenue"]

print(f"{'category':<14}{'revenue':>14}{'share':>8}")
for cat, rev in sorted(by_category.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{cat:<14}{rev:>14,.2f}{rev / total_revenue:>8.1%}")

# %% [markdown]
# *What just happened?* We summed line revenue per category and expressed each as a share of the total. The shares must add to 100%: a cheap sanity check we assert next.

# %%
assert abs(sum(by_category.values()) - total_revenue) < 1e-6
print("category revenues add up to the total")

# %% [markdown]
# ## Step 4: Top customers
# Revenue per customer requires the join through orders (line -> order -> customer). We already attached `customer_id` to each line.

# %%
by_customer = defaultdict(float)
for l in counted_lines:
    by_customer[l["customer_id"]] += l["revenue"]

top10 = sorted(by_customer.items(), key=lambda kv: kv[1], reverse=True)[:10]
for rank, (cid, rev) in enumerate(top10, 1):
    c = customers[cid]
    print(f"{rank:>2}. #{cid:<5}{c['name']:<8}{c['country']:<4}{'premium' if c['premium'] else '':<8}{rev:>12,.2f}")

top10_share = sum(rev for _, rev in top10) / total_revenue
print(f"top 10 customers = {top10_share:.1%} of revenue; {len(by_customer)} customers have any revenue")

# %% [markdown]
# Always report concentration next to a leaderboard: whether the top 10 are 2% or 30% of revenue changes what the business should do about them.

# %% [markdown]
# ## Step 5: Monthly trend
# We group by the `YYYY-MM` string (which also sorts chronologically) and draw a text bar chart.

# %%
by_month = defaultdict(float)
for l in counted_lines:
    by_month[l["month"]] += l["revenue"]

peak = max(by_month.values())
for month in sorted(by_month):
    bar = "#" * round(by_month[month] / peak * 40)
    print(f"{month}  {by_month[month]:>12,.0f}  {bar}")

top_months = sorted(by_month, key=by_month.get, reverse=True)[:3]
print("top 3 months:", sorted(top_months))

# %% [markdown]
# Read the chart before interpreting it: November and December stand out in both years, and 2024 is clearly above 2023. Ask: does it show seasonality (compare the same month in both years), a trend, or both? Two years of data is thin for strong claims, and the first and last months may be partial.

# %%
first_month, last_month = min(by_month), max(by_month)
days_in_first = {o["order_date"] for o in orders if o["order_date"].strftime("%Y-%m") == first_month}
print(first_month, "has order dates on", len(days_in_first), "distinct days;", last_month, "on",
      len({o["order_date"] for o in orders if o["order_date"].strftime("%Y-%m") == last_month}))

# %% [markdown]
# ## Step 6: Returns and cancellations by channel
# A rate needs a **numerator and a denominator** that you define explicitly: here, returned orders divided by all orders in that channel.

# %%
per_channel = defaultdict(Counter)
for o in orders:
    per_channel[o["channel"]][o["status"]] += 1

print(f"{'channel':<8}{'orders':>8}{'returned':>10}{'cancelled':>11}")
for ch, c in sorted(per_channel.items()):
    n = sum(c.values())
    print(f"{ch:<8}{n:>8}{c['returned'] / n:>10.1%}{c['cancelled'] / n:>11.1%}")

# %% [markdown]
# Differences of a percentage point or two between channels with few hundred orders can be noise; the statistics module of this roadmap shows how to judge that. Do not announce a finding from a table like this without asking "how big is the sample?".

# %% [markdown]
# ## Step 7: Trust but verify (an independent cross-check)
# The same numbers computed a completely different way: SQL against `shop.db` (SQLite, in the standard library). If two independent methods agree, you can trust the pipeline.

# %%
con = sqlite3.connect(DATA / "shop.db")
sql_total = con.execute("""
    SELECT SUM(oi.quantity * p.unit_price)
    FROM order_items oi
    JOIN orders o   ON o.order_id = oi.order_id
    JOIN products p ON p.product_id = oi.product_id
    WHERE o.status IN ('delivered', 'shipped')
""").fetchone()[0]
sql_by_cat = dict(con.execute("""
    SELECT p.category, SUM(oi.quantity * p.unit_price)
    FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id
    WHERE o.status IN ('delivered', 'shipped') GROUP BY p.category
""").fetchall())
con.close()

print(f"python {total_revenue:,.2f}  vs  sql {sql_total:,.2f}")
assert abs(total_revenue - sql_total) < 1e-4
assert all(abs(by_category[c] - sql_by_cat[c]) < 1e-4 for c in sql_by_cat) and set(by_category) == set(sql_by_cat)
print("pure-Python and SQL agree on the total and on every category")

# %% [markdown]
# ## Step 8: Package it as functions (and test it)
# A notebook full of loops is an exploration. The reusable version is a function with a clear contract that you can test.

# %%
def revenue_by(key, rows=None):
    """Sum line revenue (counted statuses only) grouped by the field `key`."""
    rows = counted_lines if rows is None else rows
    out = defaultdict(float)
    for l in rows:
        out[l[key]] += l["revenue"]
    return dict(out)


assert revenue_by("category") == dict(by_category)
assert set(revenue_by("channel")) == {"web", "mobile", "store"}
print({k: round(v) for k, v in revenue_by("channel").items()})

# %% [markdown]
# ## Common mistakes
#
# 1. **Not stating the business rule** (which statuses count as revenue) and then comparing numbers with someone who used another rule.
# 2. **Skipping validation**, then joining on keys that do not match, which silently drops or duplicates rows.
# 3. **Averaging averages** or rates of rates; aggregate the underlying numerators and denominators.
# 4. **Leaderboards without context:** report share of total and sample size.
# 5. **Trusting one implementation.** Cross-check important numbers a second way.
# 6. **Floating-point sums for money** in a real billing system (see the numbers notebook); for analysis, floats plus rounded display are usually acceptable.

# %% [markdown]
# ## Interview corner
#
# **1. Compute revenue per category without pandas. Outline the algorithm.**
# <details><summary>Answer</summary>
# Build a dict from product_id to category, iterate over order lines, add quantity times price into a dict keyed by category (defaultdict(float)), then sort by value. The key idea is a hash join through a dictionary, O(n) overall.
# </details>
#
# **2. How would you find the top 10 customers by revenue in a file too large for memory?**
# <details><summary>Answer</summary>
# Stream the file keeping a running total per customer if the number of customers fits in memory; otherwise aggregate in partitions or push the group-by to a database/columnar engine (DuckDB/SQL), then take the top N with a heap (<code>heapq.nlargest</code>).
# </details>
#
# **3. Your total differs from finance's by 3%. What do you check first?**
# <details><summary>Answer</summary>
# The definition (which statuses, gross vs net of returns, tax, currency, date cut-off and time zone), then join correctness (duplicates/dropped rows), then data freshness. Reconcile a small slice (one day, one product) line by line.
# </details>
#
# **4. Why did we compare against SQL?**
# <details><summary>Answer</summary>
# Two independent implementations rarely share the same bug, so agreement is strong evidence of correctness, and a mismatch tells you to investigate before anything reaches a report.
# </details>

# %% [markdown]
# ## Exercises
# The variables `orders`, `items`, `products`, `customers`, `lines`, `counted_lines` and `total_revenue` from above are available.

# %% [markdown]
# ### Exercise 1: Gross margin by category
# Using the counted lines, store `margin_by_category`: a dict from category to **gross profit divided by revenue** (both summed over counted lines of that category), rounded to 3 decimals.
# *Hint: `lines` already has `gross_profit` and `revenue`. Do not average per-line margins.*

# %% tags=["starter"]
margin_by_category = ...

# %%
rev, gp = defaultdict(float), defaultdict(float)
for it in items:
    o = order_by_id[it["order_id"]]
    if o["status"] in ("delivered", "shipped"):
        p = products[it["product_id"]]
        rev[p["category"]] += it["quantity"] * p["price"]
        gp[p["category"]] += it["quantity"] * (p["price"] - p["cost"])
want = {c: round(gp[c] / rev[c], 3) for c in rev}
assert margin_by_category == want, f"expected {want}, got {margin_by_category}"
print("Exercise 1 passed:", margin_by_category)

# %% tags=["solution"]
_rev, _gp = defaultdict(float), defaultdict(float)
for l in counted_lines:
    _rev[l["category"]] += l["revenue"]
    _gp[l["category"]] += l["gross_profit"]
margin_by_category = {c: round(_gp[c] / _rev[c], 3) for c in _rev}

# %% [markdown]
# ### Exercise 2: Return rate by channel, as data
# Store `return_rate`: a dict from channel to (returned orders / all orders of that channel), rounded to 4 decimals. Then `worst_channel` = the channel with the highest rate.

# %% tags=["starter"]
return_rate = ...
worst_channel = ...

# %%
n, ret = Counter(), Counter()
for o in orders:
    n[o["channel"]] += 1
    ret[o["channel"]] += o["status"] == "returned"
want = {ch: round(ret[ch] / n[ch], 4) for ch in n}
assert return_rate == want, f"expected {want}, got {return_rate}"
assert worst_channel == max(want, key=want.get)
print("Exercise 2 passed:", return_rate, "-> worst:", worst_channel)

# %% tags=["solution"]
_n, _r = Counter(), Counter()
for o in orders:
    _n[o["channel"]] += 1
    _r[o["channel"]] += o["status"] == "returned"
return_rate = {ch: round(_r[ch] / _n[ch], 4) for ch in _n}
worst_channel = max(return_rate, key=return_rate.get)

# %% [markdown]
# ### Exercise 3: Do premium customers order more?
# Compute the **average number of orders per customer** separately for premium and non-premium customers, **including customers with zero orders** in the denominator. Store `avg_orders = {"premium": ..., "standard": ...}`
# (floats, rounded to 2 decimals) and `ratio` = premium / standard, rounded to 2 decimals.

# %% tags=["starter"]
avg_orders = ...
ratio = ...

# %%
cnt = Counter(o["customer_id"] for o in orders)
prem = [cnt[c] for c, v in customers.items() if v["premium"]]
std = [cnt[c] for c, v in customers.items() if not v["premium"]]
want = {"premium": round(sum(prem) / len(prem), 2), "standard": round(sum(std) / len(std), 2)}
assert avg_orders == want, f"expected {want}, got {avg_orders}"
assert ratio == round(want["premium"] / want["standard"], 2)
print("Exercise 3 passed:", avg_orders, "ratio", ratio)

# %% tags=["solution"]
_cnt = Counter(o["customer_id"] for o in orders)
_p = [_cnt[c] for c, v in customers.items() if v["premium"]]
_s = [_cnt[c] for c, v in customers.items() if not v["premium"]]
avg_orders = {"premium": round(sum(_p) / len(_p), 2), "standard": round(sum(_s) / len(_s), 2)}
ratio = round(avg_orders["premium"] / avg_orders["standard"], 2)

# %% [markdown]
# ### Exercise 4 (open-ended): Your own finding
# Pick a question the manager did not ask: for example *which product has the highest return rate*, *does average order value differ by channel*, *which country's customers spend the most per customer*, or *which weekday is slowest*.
# Answer it with the tools above, state your definition of the metric (numerator, denominator, which statuses), validate it (a total that adds up, or a second method), and write three sentences a non-technical manager
# could act on, including one caveat about how sure you are.

# %% [markdown]
# ## Recap
# - State the business rule first; validate joins and counts before aggregating.
# - A join is a dictionary lookup; a group-by is a `defaultdict` accumulation; both are O(n).
# - Report shares and sample sizes next to rankings; define rates with explicit numerators and denominators.
# - Cross-check key numbers with an independent method (SQL here) and wrap working loops into tested functions.
# - You now have ground truth to compare against when you redo this in pandas.
#
# **Go deeper:** the Python docs for `csv`, `collections`, `heapq` and `sqlite3` (docs.python.org/3).
#
# **Next module:** [Math for data science](../02_Math_for_Data_Science/) (and the pandas notebooks, where you will redo this analysis in a few lines).
