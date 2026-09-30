# %% [markdown]
# # Merge, Join and Concat: Combining Tables Safely
#
# **Why this matters.** Real data lives in several tables (customers, orders, order lines, products). Almost every analysis starts by joining them,
# and joins are where analysts silently corrupt results: a many-to-many key duplicates rows and **revenue doubles**, a mismatched key drops customers and
# **counts shrink**, and nobody notices because no error is raised. The professional habit is simple: know the relationship of the keys, and *check the row counts
# before and after every join*.
#
# **You will learn**
# - pick the right join type (`inner`, `left`, `right`, `outer`) by asking what you want to keep
# - declare expectations with `validate=` and diagnose mismatches with `indicator=True`
# - predict and prevent row explosions from many-to-many keys
# - stack tables with `concat` and know when `join` on an index is convenient
#
# **Prerequisites:** [GroupBy and aggregation](04_groupby_and_aggregation.ipynb).
#
# > Data used: the shop tables `customers`, `orders`, `order_items`, `products`
# > (`customers 1--* orders 1--* order_items *--1 products`). Synthetic, see `docs/DATASETS.md`.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv")
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv")
order_items = pd.read_csv(DATA / "order_items.csv")
for name, t in [("customers", customers), ("products", products), ("orders", orders), ("order_items", order_items)]:
    print(f"{name:12s} {t.shape}")

# %% [markdown]
# ## 1. Join types on a toy example
# Draw the picture before you write code. Two tiny tables with **one key in common** and **one key on each side only**:

# %%
left = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["Ana", "Ben", "Chen"]})
right = pd.DataFrame({"customer_id": [2, 3, 4], "amount": [20, 30, 40]})

for how in ["inner", "left", "right", "outer"]:
    print(f"--- how='{how}'")
    print(left.merge(right, on="customer_id", how=how).to_string(index=False))

# %% [markdown]
# | `how` | Keeps | Use it when |
# |---|---|---|
# | `inner` | keys in **both** | you only care about matched rows |
# | `left` | every left row, matches from the right (else `NaN`) | left is your "main" table and you add attributes: **the default choice for enrichment** |
# | `right` | mirror of left | rarely; swap the tables and use `left` |
# | `outer` | keys in **either** | reconciling two lists |
#
# *What just happened?* Ben appears only on the left, Dan (id 4) only on the right. `left` keeps Ben with `NaN`; `inner` drops both; `outer` keeps everyone.
# When you add columns to a main table, a **left join preserves its row count** (if the right key is unique), which is why it is the safe default.

# %% [markdown]
# ## 2. One-to-many: orders and their lines
# `orders` has one row per order, `order_items` several per order. Joining the two multiplies each order by its number of lines: that is *expected* for one-to-many.
# Declare the expectation with `validate=`; pandas then **raises** if the keys do not have the relationship you claimed.

# %%
o_items = orders.merge(order_items, on="order_id", how="left", validate="one_to_many")
print("orders:", len(orders), "| order_items:", len(order_items), "| joined:", len(o_items))
print("orders with no items:", int(o_items["item_id"].isna().sum()))

# %% [markdown]
# The joined table has (about) as many rows as `order_items`, not `orders`: the grain (what one row means) changed from "order" to "order line".
# **Always know your grain.** Now add the price by joining each line to its product (many lines to one product: `many_to_one`), and check nothing was lost or duplicated:

# %%
lines = order_items.merge(products, on="product_id", how="left", validate="many_to_one")
assert len(lines) == len(order_items), "joining to a unique lookup must not change the row count"
assert lines["unit_price"].notna().all(), "every line should find its product"
lines = lines.assign(revenue=lines["quantity"] * lines["unit_price"])
print(lines[["order_id", "product_name", "quantity", "unit_price", "revenue"]].head(3).to_string(index=False))

# %% [markdown]
# Those two asserts are the cheapest insurance in data work. Write them after every join.
#
# Now the full chain: order lines with product, order and customer attributes. Different key names use `left_on` / `right_on`; overlapping column names get `suffixes`.

# %%
full = (
    lines.merge(orders, on="order_id", how="left", validate="many_to_one")
    .merge(customers, on="customer_id", how="left", validate="many_to_one", suffixes=("", "_cust"))
)
assert len(full) == len(order_items), "still one row per order line"
print(full.shape)
print(full[["order_id", "status", "channel", "country", "is_premium", "revenue"]].head(3).to_string(index=False))

# %% [markdown]
# ## 3. Finding what did not match: `indicator=True`
# `indicator=True` adds a `_merge` column saying whether each row came from `left_only`, `right_only` or `both`. This is the **anti-join**: "which customers never ordered?"

# %%
check = customers.merge(orders[["customer_id"]].drop_duplicates(), on="customer_id", how="left", indicator=True)
print(check["_merge"].value_counts().to_string())
never = check[check["_merge"] == "left_only"]
print("customers who never ordered:", len(never), "| share:", f"{len(never) / len(customers):.1%}")

# %% [markdown]
# Note `drop_duplicates()` on the right side: without it, a customer with 5 orders would appear 5 times. Also note the situation is normal here (customers can sign up and never buy);
# in a different context, unmatched keys could mean a data-quality bug. `indicator` lets you *see* the mismatch instead of guessing.

# %% [markdown]
# ## 4. Key mismatches: the silent killers
# **(a) Type mismatch.** The same customer id stored as text on one side and integer on the other. pandas refuses to merge them:

# %%
cust_text = customers.assign(customer_id=customers["customer_id"].astype(str))
try:
    orders.merge(cust_text, on="customer_id")
except ValueError as e:
    print("ValueError:", str(e)[:110], "...")

fixed = orders.merge(cust_text.astype({"customer_id": int}), on="customer_id", how="left")
print("after casting both sides to int:", len(fixed), "rows")

# %% [markdown]
# **(b) Formatting mismatch** (case, whitespace, leading zeros) does not raise: the rows simply fail to match. The symptom is a pile of `NaN` after a left join:

# %%
lookup = pd.DataFrame({"code": ["a1", "b2", "c3"], "label": ["Alpha", "Beta", "Gamma"]})
events = pd.DataFrame({"code": ["A1", " b2", "c3", "d4"]})
bad = events.merge(lookup, on="code", how="left")
print(bad["label"].isna().sum(), "of", len(bad), "unmatched (one of them, d4, is genuinely missing)")

events["code_clean"] = events["code"].str.strip().str.lower()
good = events.merge(lookup, left_on="code_clean", right_on="code", how="left", suffixes=("_raw", ""))
print(good[["code_raw", "label"]].to_string(index=False))

# %% [markdown]
# **Rule:** after a left join, count the `NaN`s in a column that should always match. If the number is not what you expect, stop and investigate the keys.

# %% [markdown]
# ## 5. Many-to-many: how rows explode
# If the key is **not unique on either side**, every left row pairs with every matching right row. Merging `orders` with itself on `customer_id`
# (a common mistake when trying to "get the customer's other orders") gives, for a customer with *n* orders, *n x n* rows.
# So the joined size is predictable **without doing the join**: the sum over customers of `n^2`.

# %%
n_per_customer = orders.groupby("customer_id").size()
predicted = int((n_per_customer ** 2).sum())
exploded = orders[["order_id", "customer_id"]].merge(orders[["order_id", "customer_id", "channel"]], on="customer_id", suffixes=("_a", "_b"))
print("orders:", len(orders), "| predicted rows:", predicted, "| actual rows:", len(exploded))
print("blow-up factor:", round(len(exploded) / len(orders), 1), "x")

# %% [markdown]
# Any sum computed after such a join is inflated by the same duplication. `validate=` catches this the moment it happens:

# %%
from pandas.errors import MergeError

try:
    orders[["order_id", "customer_id"]].merge(orders[["customer_id", "channel"]], on="customer_id", validate="one_to_one")
except MergeError as e:
    print("MergeError:", str(e)[:100])

# %% [markdown]
# The **fix** depends on what you actually want. To attach a customer-level number, aggregate first (to one row per customer) and *then* join:

# %%
per_customer = orders.groupby("customer_id").agg(n_orders=("order_id", "size"), last_order=("order_date", "max")).reset_index()
enriched = customers.merge(per_customer, on="customer_id", how="left", validate="one_to_one")
enriched["n_orders"] = enriched["n_orders"].fillna(0).astype(int)      # no match = zero orders (a deliberate, explained fill)
assert len(enriched) == len(customers)
print(enriched[["customer_id", "n_orders", "last_order"]].head(3).to_string(index=False))

# %% [markdown]
# ## 6. `concat`: stacking tables
# `pd.concat` stacks frames **vertically** (`axis=0`, more rows) or **horizontally** (`axis=1`, more columns aligned on the index).
# Use it to append files with the same columns, for example one CSV per month.

# %%
odt = pd.to_datetime(orders["order_date"])
part_2023 = orders[odt.dt.year == 2023]
part_2024 = orders[odt.dt.year == 2024]
part_2025 = orders[odt.dt.year == 2025]

stacked = pd.concat([part_2023, part_2024, part_2025], ignore_index=True)
print(len(part_2023), len(part_2024), len(part_2025), "->", len(stacked), "| same as original:", len(stacked) == len(orders))

tagged = pd.concat({"y2023": part_2023, "y2024": part_2024}, names=["source", None])
print(tagged.index.names, "|", tagged.shape)

# %% [markdown]
# `ignore_index=True` renumbers the rows (otherwise duplicate index labels survive, which surprises `loc`). `keys=` (or a dict) records where each row came from.
# If the frames have different columns, `concat` fills the gaps with `NaN` (`join="outer"`, the default) so **check the columns** you expect:

# %%
a = pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0]})
b = pd.DataFrame({"id": [3], "y": ["new"]})
print(pd.concat([a, b], ignore_index=True).to_string(index=False))

# %% [markdown]
# `DataFrame.join` is a shortcut for merging on the **index**. It is handy after `set_index`, but explicit `merge(on=...)` is easier to read and validate.

# %%
cust_idx = customers.set_index("customer_id")[["country", "is_premium"]]
j = orders.join(cust_idx, on="customer_id", how="left")     # `on=` looks up orders["customer_id"] in cust_idx's index
print(j.shape, "| unmatched:", int(j["country"].isna().sum()))

# %% [markdown]
# ## A checklist before you trust a join
#
# 1. What is the **grain** (one row = ?) of each table, and which columns are *unique* keys?
# 2. What relationship do you expect (`one_to_one`, `one_to_many`, `many_to_one`)? Pass it as `validate=`.
# 3. Row count before and after; is the change what the relationship predicts?
# 4. `NaN` count in columns from the right table after a left join.
# 5. A cross-check total: revenue from the joined table equals revenue computed another way.

# %% [markdown]
# ## Common mistakes
#
# 1. **Inner join by default.** `merge` defaults to `how="inner"`; unmatched rows vanish without warning. Choose deliberately, usually `left`.
# 2. **Joining on a non-unique key** and summing afterwards (inflated totals). Use `validate=`.
# 3. **Key dtype or formatting differences** (int vs str, case, whitespace).
# 4. **Filtering the right table's column after a left join** (`df[df["right_col"] == x]`), which silently turns it back into an inner join. Filter before joining, or keep the `NaN`s.
# 5. **Column name collisions** producing `_x` / `_y` suffixes that nobody remembers. Choose `suffixes` explicitly or select columns before merging.
# 6. **`concat` with `axis=1` on unaligned indexes**: rows align by index label, not by position; reset the index first if you meant position.

# %%
# Mistake 4 in action: a left join keeps unmatched rows, a filter on the right column then removes them
lj = customers.merge(per_customer, on="customer_id", how="left")
print("left join rows:", len(lj), "| after filtering n_orders >= 1:", len(lj[lj["n_orders"] >= 1]))

# %% [markdown]
# ## Interview corner
#
# **1. Difference between `merge`, `join` and `concat`?**
# <details><summary>Answer</summary>
# <code>merge</code> is a SQL-style join on columns (or indexes); <code>join</code> is a convenience for joining on the index; <code>concat</code> stacks frames along an axis without matching keys
# (it aligns on the other axis's labels).
# </details>
#
# **2. After a left join your row count went up. Why?**
# <details><summary>Answer</summary>
# The right table's key was not unique, so matching left rows were duplicated (one-to-many or many-to-many). Check <code>right[key].is_unique</code>, pass <code>validate="many_to_one"</code>, or aggregate the right table first.
# </details>
#
# **3. How do you find rows in A that have no match in B?**
# <details><summary>Answer</summary>
# Left-merge with <code>indicator=True</code> and keep <code>_merge == "left_only"</code> (an anti-join); or <code>A[~A["key"].isin(B["key"])]</code>.
# </details>
#
# **4. How can you predict the size of a many-to-many join without running it?**
# <details><summary>Answer</summary>
# For an inner join, sum over each key value of (count in left) x (count in right).
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Who never ordered?
# Using an anti-join, store in `never_ordered` a DataFrame of the customers with no orders, and in `premium_rate_never` the share of those customers that are premium.
# Compare with the premium rate among customers who did order, stored in `premium_rate_ordered`. *(A premium customer orders about 3x as often, so what would you expect?)*

# %% tags=["starter"]
never_ordered = ...
premium_rate_never = ...
premium_rate_ordered = ...

# %%
has_order = customers["customer_id"].isin(orders["customer_id"])
assert len(never_ordered) == int((~has_order).sum()), "customers whose id never appears in orders"
assert set(never_ordered["customer_id"]) == set(customers.loc[~has_order, "customer_id"]), "wrong customers"
assert abs(premium_rate_never - customers.loc[~has_order, "is_premium"].mean()) < 1e-9, "premium share among non-buyers"
assert abs(premium_rate_ordered - customers.loc[has_order, "is_premium"].mean()) < 1e-9, "premium share among buyers"
print(f"Exercise 1 passed (premium: {premium_rate_never:.1%} of non-buyers vs {premium_rate_ordered:.1%} of buyers)")

# %% tags=["solution"]
m = customers.merge(orders[["customer_id"]].drop_duplicates(), on="customer_id", how="left", indicator=True)
never_ordered = m[m["_merge"] == "left_only"].drop(columns="_merge")
premium_rate_never = never_ordered["is_premium"].mean()
premium_rate_ordered = m.loc[m["_merge"] == "both", "is_premium"].mean()

# %% [markdown]
# ### Exercise 2: Revenue per order
# Build `order_revenue`: a DataFrame with columns `order_id`, `status`, `revenue` (sum of `quantity * unit_price` over the order's lines), **one row per order in `orders`**
# (an order with no lines, if any, gets revenue 0). Store the total revenue of **delivered** orders in `delivered_revenue`.
# *Hint: aggregate the lines first, then left-join to `orders`.*

# %% tags=["starter"]
order_revenue = ...
delivered_revenue = ...

# %%
price = dict(zip(products["product_id"], products["unit_price"]))
manual = {}
for oid, pid, q in zip(order_items["order_id"], order_items["product_id"], order_items["quantity"]):
    manual[oid] = manual.get(oid, 0.0) + q * price[pid]
assert list(order_revenue.columns) == ["order_id", "status", "revenue"], "columns: order_id, status, revenue"
assert len(order_revenue) == len(orders) and order_revenue["order_id"].is_unique, "exactly one row per order"
assert order_revenue["revenue"].notna().all(), "orders without lines should have revenue 0, not NaN"
sample = order_revenue.sample(50, random_state=1)
assert all(abs(r - manual.get(o, 0.0)) < 1e-6 for o, r in zip(sample["order_id"], sample["revenue"])), "revenue per order differs from the independent calculation"
status_of = dict(zip(orders["order_id"], orders["status"]))
exp = sum(v for o, v in manual.items() if status_of[o] == "delivered")
assert abs(delivered_revenue - exp) < 1e-4, f"expected {exp:.2f}"
print(f"Exercise 2 passed (delivered revenue {delivered_revenue:,.0f})")

# %% tags=["solution"]
line_rev = order_items.merge(products[["product_id", "unit_price"]], on="product_id", how="left", validate="many_to_one")
line_rev = line_rev.assign(revenue=line_rev["quantity"] * line_rev["unit_price"])
per_order = line_rev.groupby("order_id", as_index=False)["revenue"].sum()
order_revenue = orders[["order_id", "status"]].merge(per_order, on="order_id", how="left", validate="one_to_one")
order_revenue["revenue"] = order_revenue["revenue"].fillna(0.0)
delivered_revenue = order_revenue.loc[order_revenue["status"] == "delivered", "revenue"].sum()

# %% [markdown]
# ### Exercise 3: Predict the explosion
# Consider joining `orders[["order_id", "channel", "customer_id"]]` to itself on `channel` (a low-cardinality key: only a few distinct values).
# **Without running the merge**, compute `predicted_rows` (sum over channels of `n_channel ** 2`). Then run the merge and store the actual row count in `actual_rows`.

# %% tags=["starter"]
predicted_rows = ...
actual_rows = ...

# %%
sub = orders[["order_id", "channel", "customer_id"]]
real = len(sub.merge(sub, on="channel"))
assert predicted_rows == int((orders["channel"].value_counts() ** 2).sum()), "sum over channels of count squared"
assert actual_rows == real == predicted_rows, "the prediction and the actual merge must match"
print(f"Exercise 3 passed: {predicted_rows:,} rows from a table of {len(orders):,} (that is {predicted_rows / len(orders):,.0f}x)")

# %% tags=["solution"]
sub = orders[["order_id", "channel", "customer_id"]]
predicted_rows = int((sub["channel"].value_counts() ** 2).sum())
actual_rows = len(sub.merge(sub, on="channel"))

# %% [markdown]
# ### Exercise 4: Append and verify
# Split `orders` into the rows with status `delivered` (`d`) and all the other rows (`rest`), then combine them back into `all_orders` with `concat`
# (fresh index 0..n-1). Also store `n_dupe_ids`, the number of duplicated `order_id` values in your result (it should be 0).

# %% tags=["starter"]
d = ...
rest = ...
all_orders = ...
n_dupe_ids = ...

# %%
assert (d["status"] == "delivered").all() and (rest["status"] != "delivered").all(), "split by status"
assert len(all_orders) == len(orders), "no rows lost or gained"
assert list(all_orders.index) == list(range(len(orders))), "use ignore_index=True"
assert n_dupe_ids == int(all_orders["order_id"].duplicated().sum()) == 0, "no order_id should appear twice"
assert set(all_orders["order_id"]) == set(orders["order_id"]), "same orders as the original"
print("Exercise 4 passed")

# %% tags=["solution"]
d = orders[orders["status"] == "delivered"]
rest = orders[orders["status"] != "delivered"]
all_orders = pd.concat([d, rest], ignore_index=True)
n_dupe_ids = int(all_orders["order_id"].duplicated().sum())

# %% [markdown]
# ### Exercise 5 (open-ended): Revenue by customer country
# Build the full chain (lines, products, orders, customers) and compute revenue per customer `country`. Report which country leads and by how much, and state in a markdown cell
# which order statuses you included or excluded and why (should cancelled and returned orders count as revenue?). Put your `validate=` and row-count checks in the code.

# %% [markdown]
# ## Recap
# - `left` join to enrich a main table; `inner` when you only want matches; know the grain of every table.
# - `validate=` states the expected key relationship; `indicator=True` reveals unmatched rows (anti-join).
# - Many-to-many joins multiply rows: predicted size = sum of (left count x right count) per key. Aggregate before joining.
# - Fix key dtype and formatting problems *before* the join; check `NaN`s after a left join.
# - `concat` stacks (use `ignore_index=True` and check the columns); `join` is merge on the index.
#
# **Go deeper:** pandas user guide "Merge, join, concatenate and compare" (pandas.pydata.org/docs/user_guide/merging.html).
#
# **Next:** [Reshaping and window functions](06_reshaping_and_window_functions.ipynb)
