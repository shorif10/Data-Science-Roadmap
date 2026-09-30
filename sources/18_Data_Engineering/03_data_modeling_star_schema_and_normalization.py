# %% [markdown]
# # Data Modelling: Normalisation and the Star Schema
#
# **Why this matters.** Two very different tables can hold the same facts. One is built so that
# an application can update a customer's address in exactly one place; the other is built so that an analyst can
# answer "revenue by region by month" with two joins and no surprises. Knowing when to use which, and how to
# convert between them, is the core of data modelling. It is also the answer to a very common interview question:
# "a customer changes country, what happens to last year's reports?"
#
# **You will learn**
# - spot redundancy in a flat table with functional dependencies, and normalise it to 3NF
# - keep data honest with primary and foreign keys (DuckDB enforces them)
# - build a star schema (one fact table, several dimensions) and query it
# - implement slowly changing dimensions, Type 1 (overwrite) vs Type 2 (keep history), in SQL
#
# **Prerequisites:** [File formats](02_file_formats_csv_json_parquet.ipynb) and basic SQL joins ([SQL module](../06_SQL/)).
#
# > Data used: `customers.csv`, `products.csv`, `orders.csv`, `order_items.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import duckdb
import pandas as pd

DATA = Path("../datasets")
customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(DATA / "products.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")

# %% [markdown]
# ## 1. The messy flat table
# Imagine the finance team hands you this: one giant export, one row per order line, with everything
# repeated on every row. (We also add a `region` for each country, which will matter in a minute.)

# %%
REGION = {"US": "North America", "CA": "North America", "UK": "Europe", "DE": "Europe", "FR": "Europe",
          "IN": "South Asia", "BD": "South Asia", "AU": "Oceania"}
flat = (items.merge(orders, on="order_id").merge(customers, on="customer_id").merge(products, on="product_id"))
flat["region"] = flat["country"].map(REGION)
flat = flat[["item_id", "order_id", "order_date", "status", "channel", "customer_id", "first_name", "country", "region",
             "signup_date", "age", "is_premium", "product_id", "product_name", "category", "unit_price", "unit_cost", "quantity"]]
print(flat.shape)
flat.head(3)

# %% [markdown]
# Ask: how many times is customer 1's first name and country stored? Once per order line they ever bought.
# That redundancy causes three classic **anomalies**:
#
# * **Update anomaly**: customer moves country; you must change *every* row. Miss one and the data disagrees with itself.
# * **Insert anomaly**: you cannot record a new product until someone orders it (there is no row to hold it).
# * **Delete anomaly**: deleting a customer's only order also deletes everything you knew about them.
#
# ## 2. Functional dependencies: the tool for finding redundancy
# `A -> B` means "knowing A tells you B exactly". If `customer_id -> country` holds, storing `country` per order line is redundant.
# Test it: for each value of A, B must have exactly one distinct value.

# %%
def fd_holds(df, determinant, dependent):
    """True if every value of `determinant` maps to exactly one value of `dependent`."""
    return bool((df.groupby(determinant)[dependent].nunique(dropna=False) <= 1).all())


for a, b in [("customer_id", "country"), ("country", "region"), ("product_id", "category"),
             ("category", "product_id"), ("order_id", "customer_id"), ("customer_id", "order_id")]:
    print(f"{a:12s} -> {b:12s}: {fd_holds(flat, a, b)}")

# %% [markdown]
# So `customer_id` determines the customer attributes, `product_id` the product attributes, `order_id` the order attributes.
# Notice `country -> region` also holds: `region` depends on `customer_id` only **through** `country`. That is a **transitive dependency**.
#
# **Normal forms in one paragraph each**
# * **1NF**: every cell holds one atomic value (no lists like `"web,store"`), and rows are identifiable.
# * **2NF**: no non-key column depends on *part* of a composite key. (Here the line key is `item_id` alone, so we pass; with the key `(order_id, product_id)`, `product_name` would violate it.)
# * **3NF**: no non-key column depends on another non-key column (no transitive dependencies). "Every attribute depends on the key, the whole key, and nothing but the key."
#
# ## 3. Normalise to 3NF, with keys enforced
# One table per "thing": `region_country`, `customer`, `product`, `orders`, `order_item`. DuckDB enforces primary and foreign keys, so the database itself refuses bad data.

# %%
con = duckdb.connect()   # in-memory database
con.execute("""
CREATE TABLE country (country VARCHAR PRIMARY KEY, region VARCHAR NOT NULL);
CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, first_name VARCHAR, country VARCHAR REFERENCES country(country),
                       signup_date DATE, age INTEGER, is_premium BOOLEAN);
CREATE TABLE product (product_id INTEGER PRIMARY KEY, product_name VARCHAR, category VARCHAR, unit_price DOUBLE, unit_cost DOUBLE);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customer(customer_id),
                     order_date DATE, status VARCHAR, channel VARCHAR);
CREATE TABLE order_item (item_id INTEGER PRIMARY KEY, order_id INTEGER REFERENCES orders(order_id),
                         product_id INTEGER REFERENCES product(product_id), quantity INTEGER);
""")
country_df = pd.DataFrame({"country": list(REGION), "region": list(REGION.values())})
for name, df in [("country", country_df), ("customer", customers), ("product", products), ("orders", orders), ("order_item", items)]:
    con.register("_src", df)                      # expose the DataFrame to SQL under a temporary name
    con.execute(f"INSERT INTO {name} SELECT * FROM _src")   # order matters: parents before children
    con.unregister("_src")
print(con.sql("SELECT table_name, estimated_size AS rows FROM duckdb_tables() ORDER BY 1").df().to_string(index=False))

# %% [markdown]
# Insert order matters: a child row cannot point at a parent that does not exist yet. Let's prove the database protects us:

# %%
try:
    con.execute("INSERT INTO order_item VALUES (999999, 999999, 1, 1)")   # order 999999 does not exist
except duckdb.Error as e:
    print(type(e).__name__, "->", str(e)[:110])

# %% [markdown]
# *What just happened?* The foreign key rejected an "orphan" row. This is **referential integrity**; in analytical stores (Snowflake, BigQuery) keys are often
# *not* enforced, so you must test for it yourself, which is the topic of notebook 04.
#
# Is the split lossless? Join everything back and compare with the flat table:

# %%
rebuilt = con.sql("""
SELECT oi.item_id, o.order_id, o.order_date, o.status, o.channel, c.customer_id, c.first_name, c.country, r.region,
       c.signup_date, c.age, c.is_premium, p.product_id, p.product_name, p.category, p.unit_price, p.unit_cost, oi.quantity
FROM order_item oi JOIN orders o USING (order_id) JOIN customer c USING (customer_id)
JOIN country r USING (country) JOIN product p USING (product_id)
""").df().sort_values("item_id").reset_index(drop=True)
same = rebuilt.astype(str).equals(flat.sort_values("item_id").reset_index(drop=True).astype(str))
print("joined-back table equals the flat table:", same)
print(f"flat table cells: {flat.size:,}  vs  normalised cells: {sum(con.sql(f'SELECT count(*) FROM {t}').fetchone()[0] * n for t, n in [('country', 2), ('customer', 6), ('product', 5), ('orders', 5), ('order_item', 4)]):,}")

# %% [markdown]
# Same information, roughly half the cells, and every fact now lives in **one** place, so an update is a one-row change. That is what normalisation buys:
# **write correctness** for applications (OLTP). What it costs: analysts need 4-5 joins for a simple question.
#
# ## 4. Star schema for analytics
# Analytics flips the priorities: reads dominate, so we accept some redundancy for simple, fast, understandable queries.
# A **star schema** has
# * one **fact table**: one row per business event at a declared **grain** (here: *one order line*), with numeric **measures** and foreign keys;
# * several **dimension tables**: descriptive attributes to slice by (who, what, when, where), each with a **surrogate key** (a meaningless integer the warehouse controls).
#
# Why surrogate keys? Source ids can be reused, collide across systems, or (as we will see with SCD2) need several versions of one business entity.
#
# First the date dimension: generated once, it lets you group by month, quarter or weekend without date arithmetic in every query.

# %%
con.execute("""
CREATE TABLE dim_date AS
SELECT CAST(strftime(d, '%Y%m%d') AS INTEGER) AS date_key, d AS date, year(d) AS year, month(d) AS month,
       strftime(d, '%Y-%m') AS year_month, quarter(d) AS quarter, isodow(d) IN (6, 7) AS is_weekend
FROM (SELECT CAST(x AS DATE) AS d FROM range(DATE '2023-06-01', DATE '2025-07-01', INTERVAL 1 DAY) t(x))
""")
con.sql("SELECT * FROM dim_date WHERE date IN ('2023-06-03', '2024-02-29') ORDER BY date").df()

# %% [markdown]
# Now the product and channel dimensions, and the customer dimension with **SCD2 columns** (`valid_from`, `valid_to`, `is_current`). At first each customer has exactly one
# version, valid "forever" (half-open interval `[valid_from, valid_to)`).

# %%
con.execute("""
CREATE TABLE dim_product AS
SELECT row_number() OVER (ORDER BY product_id) AS product_key, product_id, product_name, category, unit_price, unit_cost FROM product;

CREATE TABLE dim_channel AS
SELECT row_number() OVER (ORDER BY channel) AS channel_key, channel FROM (SELECT DISTINCT channel FROM orders);

CREATE TABLE dim_customer AS
SELECT row_number() OVER (ORDER BY c.customer_id) AS customer_key, c.customer_id, c.first_name, c.country, r.region, c.is_premium,
       DATE '1900-01-01' AS valid_from, DATE '9999-12-31' AS valid_to, TRUE AS is_current
FROM customer c JOIN country r USING (country);
""")
con.sql("SELECT * FROM dim_customer LIMIT 3").df()

# %% [markdown]
# The fact table stores keys and measures. Note the **as-of join** on the customer dimension: each order is tied to the customer *version that was valid on the order date*.
# (Right now every customer has one version, so this is trivial; we will make it matter in section 5.)

# %%
def load_fact_sales(con):
    con.execute("DROP TABLE IF EXISTS fact_sales")
    con.execute("""
    CREATE TABLE fact_sales AS
    SELECT oi.item_id, d.date_key, dc.customer_key, dp.product_key, ch.channel_key, o.order_id, o.status,
           oi.quantity, oi.quantity * p.unit_price AS revenue, oi.quantity * p.unit_cost AS cost
    FROM order_item oi
    JOIN orders o USING (order_id)
    JOIN product p USING (product_id)
    JOIN dim_date d ON d.date = o.order_date
    JOIN dim_customer dc ON dc.customer_id = o.customer_id AND o.order_date >= dc.valid_from AND o.order_date < dc.valid_to
    JOIN dim_product dp ON dp.product_id = oi.product_id
    JOIN dim_channel ch ON ch.channel = o.channel
    """)


load_fact_sales(con)
print(con.sql("SELECT count(*) AS n_fact_rows, count(DISTINCT item_id) AS n_distinct_items FROM fact_sales").df().to_string(index=False))

# %% [markdown]
# Grain check passed: as many fact rows as source order lines, one per `item_id`. A fan-out (a join that duplicates rows) would show up here as more fact rows than items;
# checking the grain after every load is a cheap habit that prevents inflated revenue numbers.
#
# Now the payoff, a business question with a flat, predictable query shape (fact joined to dimensions, group by attributes):

# %%
monthly = con.sql("""
SELECT d.year_month, p.category, round(sum(f.revenue), 2) AS revenue
FROM fact_sales f JOIN dim_date d USING (date_key) JOIN dim_product p USING (product_key)
WHERE f.status = 'delivered'
GROUP BY 1, 2 ORDER BY 1, 2
""").df()
print(monthly.head(4).to_string(index=False))

pandas_check = (flat.assign(ym=flat["order_date"].dt.strftime("%Y-%m"), rev=flat["quantity"] * flat["unit_price"])
                .query("status == 'delivered'").groupby(["ym", "category"])["rev"].sum().round(2))
print("matches the pandas result on the flat table:",
      monthly.set_index(["year_month", "category"])["revenue"].sub(pandas_check).abs().max() < 0.011)

# %% [markdown]
# **Star vs snowflake.** If dimensions are themselves normalised (product -> category -> department in separate tables) you get a *snowflake*. It saves a little space but
# adds joins; most teams keep dimensions denormalised ("star") because storage is cheap and simple queries are worth more.
#
# | | 3NF (normalised) | Star schema |
# |---|---|---|
# | Purpose | run the application, write correctness | analytics, read speed and clarity |
# | Redundancy | minimal | deliberate, in dimensions |
# | Typical query | many small joins for one entity | few joins, big aggregations |
# | Change a customer attribute | update one row | depends: Type 1 or Type 2 (below) |

# %% [markdown]
# ## 5. Slowly changing dimensions (SCD)
# Dimension attributes change slowly and unpredictably: a customer moves, a product is re-categorised. What should last year's report say?
#
# * **Type 1**: overwrite. Simple; history is lost, and old reports change if re-run.
# * **Type 2**: keep the old row (closed with an end date) and insert a new row (open). Reports reproduce as they were; costs a few extra rows and the as-of join.
# * (Type 3 keeps one "previous value" column; useful only for the last change. Rarely worth it.)
#
# Scenario: our busiest customer moves to a different country on **2024-06-01**. Who is that, and which country do they move to?

# %%
top_customer = int(con.sql("SELECT customer_id FROM orders GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT 1").fetchone()[0])
old_country = con.sql(f"SELECT country FROM customer WHERE customer_id = {top_customer}").fetchone()[0]
new_country = "DE" if old_country != "DE" else "FR"
CHANGE_DATE = "2024-06-01"
before_after = con.sql(f"""SELECT count(*) FILTER (WHERE order_date < DATE '{CHANGE_DATE}') AS orders_before,
                               count(*) FILTER (WHERE order_date >= DATE '{CHANGE_DATE}') AS orders_after
                        FROM orders WHERE customer_id = {top_customer}""").fetchone()
print(f"customer {top_customer}: {old_country} -> {new_country} on {CHANGE_DATE}; orders before/after: {before_after}")

# %% [markdown]
# **Type 1**: a single `UPDATE`. We do it on a *copy* so we can compare both approaches side by side.

# %%
con.execute("CREATE TABLE dim_customer_t1 AS SELECT * FROM dim_customer")
con.execute(f"""UPDATE dim_customer_t1 SET country = '{new_country}',
                region = (SELECT region FROM country WHERE country = '{new_country}')
                WHERE customer_id = {top_customer}""")
con.sql(f"SELECT customer_key, customer_id, country, region, valid_from, valid_to, is_current FROM dim_customer_t1 WHERE customer_id = {top_customer}").df()

# %% [markdown]
# One row, new country, and no trace of the old one. **Type 2** is a *close-and-insert*, which we wrap as a function that is safe to run twice (idempotent):
# if the attribute already has the new value it does nothing. Idempotency is the theme of notebook 19.01.

# %%
def scd2_change_country(con, customer_id, new_country, effective_date):
    """Type 2 change on dim_customer. Returns True if a new version was created."""
    cur = con.execute("SELECT country FROM dim_customer WHERE customer_id = ? AND is_current", [customer_id]).fetchone()
    if cur is None:
        raise ValueError(f"unknown customer {customer_id}")
    if cur[0] == new_country:
        return False                                            # already applied, nothing to do
    con.execute("BEGIN")
    con.execute("UPDATE dim_customer SET valid_to = CAST(? AS DATE), is_current = FALSE WHERE customer_id = ? AND is_current",
                [effective_date, customer_id])
    con.execute("""
        INSERT INTO dim_customer
        SELECT (SELECT max(customer_key) + 1 FROM dim_customer), c.customer_id, c.first_name, ?, r.region, c.is_premium,
               CAST(? AS DATE), DATE '9999-12-31', TRUE
        FROM customer c, country r WHERE c.customer_id = ? AND r.country = ?
    """, [new_country, effective_date, customer_id, new_country])
    con.execute("COMMIT")
    return True


print("first call created a version:", scd2_change_country(con, top_customer, new_country, CHANGE_DATE))
print("second call created a version:", scd2_change_country(con, top_customer, new_country, CHANGE_DATE))
con.sql(f"SELECT customer_key, customer_id, country, valid_from, valid_to, is_current FROM dim_customer WHERE customer_id = {top_customer} ORDER BY customer_key").df()

# %% [markdown]
# Two rows now: the old version is closed at the change date, the new one is open. Rebuild the fact table (the as-of join now picks the right version for each order date) and
# compare what each approach says about this customer's revenue by country:

# %%
load_fact_sales(con)
t2 = con.sql(f"""
SELECT dc.country, round(sum(f.revenue), 2) AS revenue_type2
FROM fact_sales f JOIN dim_customer dc USING (customer_key) WHERE dc.customer_id = {top_customer} GROUP BY 1 ORDER BY 1""").df()
t1 = con.sql(f"""
SELECT dc.country, round(sum(f.revenue), 2) AS revenue_type1
FROM fact_sales f JOIN dim_customer c2 USING (customer_key)
JOIN dim_customer_t1 dc ON dc.customer_id = c2.customer_id WHERE dc.customer_id = {top_customer} GROUP BY 1 ORDER BY 1""").df()
print(t2.to_string(index=False))
print(t1.to_string(index=False))

# %% [markdown]
# *What just happened?* Type 2 split this customer's revenue between the two countries according to **where they lived when they ordered**. Type 1 assigns all of it to the new country,
# so revenue attributed to the old country silently drops when you re-run last year's report. Neither is universally right: for a typo fix use Type 1; for anything the business
# might want to analyse historically (region, segment, price tier) use Type 2.
#
# One more Type 2 check worth automating: for each business key the versions must not overlap or leave gaps.

# %%
gaps = con.sql("""
SELECT customer_id FROM (
  SELECT customer_id, valid_to, lead(valid_from) OVER (PARTITION BY customer_id ORDER BY valid_from) AS next_from FROM dim_customer)
WHERE next_from IS NOT NULL AND next_from <> valid_to
""").df()
one_current = con.sql("SELECT count(*) FROM (SELECT customer_id FROM dim_customer WHERE is_current GROUP BY 1 HAVING count(*) <> 1)").fetchone()[0]
print("customers with gaps/overlaps:", len(gaps), "| customers without exactly one current row:", one_current)

# %% [markdown]
# ## Common mistakes
#
# 1. **Joining a fact table to an SCD2 dimension on the business key only.** You match *all* versions and multiply the rows (fan-out), inflating revenue. Join on the surrogate key or with the as-of date range.
# 2. **Not declaring the grain.** "One row per order" vs "per order line" changes every measure. Write it down and test it (count vs distinct keys).
# 3. **Storing derived per-row values that depend on non-key columns** (`region` beside `country`), which invites update anomalies. Put them in the table where their determinant is the key.
# 4. **Over-normalising the warehouse.** Ten joins to get a customer's region punishes every analyst. Denormalise dimensions.
# 5. **Using natural keys as warehouse keys.** They get reused, change format, or clash across source systems. Use surrogate keys and keep the natural key as an attribute.

# %% [markdown]
# ## Interview corner
#
# **1. What is 3NF and why does it matter?**
# <details><summary>Answer</summary>
# Every non-key attribute depends on the key, the whole key, and nothing but the key (no transitive dependencies). It removes redundancy, so an update touches one row and cannot leave the data
# contradicting itself. It is the default for transactional (OLTP) schemas.
# </details>
#
# **2. Explain a star schema and why analysts like it.**
# <details><summary>Answer</summary>
# One fact table at a declared grain holding measures and foreign keys, surrounded by denormalised dimension tables. Queries are always "fact join dimensions, group by attributes", which is easy to write,
# easy for BI tools to generate, and fast on columnar engines.
# </details>
#
# **3. A customer's country changes. How do you model it?**
# <details><summary>Answer</summary>
# It depends on whether history matters. Type 1 overwrites (simple, rewrites history). Type 2 closes the old row with an end date and inserts a new row with a new surrogate key and validity dates
# (reports reproduce as they were; facts join on the version valid at event time). I would default to Type 2 for attributes people segment by.
# </details>
#
# **4. What is a surrogate key and why use one?**
# <details><summary>Answer</summary>
# A meaningless, warehouse-generated integer key. It is stable when source keys change or collide, it lets one business entity have several SCD2 versions, and integer joins are compact.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Does the dependency hold?
# Use `fd_holds` (or your own check) on `flat` to answer: does `product_name -> product_id` hold? Does `category -> unit_price` hold? Store the booleans in `name_determines_id` and `category_determines_price`.
# Think about what each answer means for a normalised design (can you use `product_name` as a key?).

# %% tags=["starter"]
name_determines_id = ...
category_determines_price = ...

# %%
exp1 = flat[["product_name", "product_id"]].drop_duplicates().groupby("product_name").size().max() == 1
exp2 = flat[["category", "unit_price"]].drop_duplicates().groupby("category").size().max() == 1
assert name_determines_id == exp1, "for each product_name, how many distinct product_id values exist?"
assert category_determines_price == exp2, "for each category, how many distinct unit_price values exist?"
print("Exercise 1 passed:", name_determines_id, category_determines_price)

# %% tags=["solution"]
name_determines_id = fd_holds(flat, "product_name", "product_id")
category_determines_price = fd_holds(flat, "category", "unit_price")

# %% [markdown]
# ### Exercise 2: Revenue by region and year from the star schema
# Write SQL against `fact_sales`, `dim_date` and `dim_customer` (join fact to both dimensions on their surrogate keys) that returns a DataFrame `region_year` with columns
# `region`, `year`, `revenue` (rounded to 2 decimals) for **delivered** orders only. Because `dim_customer` is now Type 2, this reports revenue by the region *at time of order*.

# %% tags=["starter"]
region_year = ...

# %%
assert list(region_year.columns) == ["region", "year", "revenue"], "columns: region, year, revenue"
hist = (flat.merge(con.sql("SELECT customer_id, region, valid_from, valid_to FROM dim_customer").df(), on="customer_id", suffixes=("_flat", "_dim"))
        .query("order_date >= valid_from and order_date < valid_to and status == 'delivered'"))
hist = hist.assign(year=hist["order_date"].dt.year, rev=hist["quantity"] * hist["unit_price"])
exp = hist.groupby(["region_dim", "year"])["rev"].sum().round(2)
got = region_year.set_index(["region", "year"])["revenue"]
assert len(got) == len(exp), f"expected {len(exp)} region-year rows, got {len(got)}"
assert (got.sort_index().to_numpy() - exp.sort_index().to_numpy()).__abs__().max() < 0.011, "revenue differs from the independent pandas calculation"
print("Exercise 2 passed")

# %% tags=["solution"]
region_year = con.sql("""
SELECT dc.region, d.year, round(sum(f.revenue), 2) AS revenue
FROM fact_sales f JOIN dim_date d USING (date_key) JOIN dim_customer dc USING (customer_key)
WHERE f.status = 'delivered' GROUP BY 1, 2 ORDER BY 1, 2
""").df()

# %% [markdown]
# ### Exercise 3: As-was revenue for the moved customer
# For the customer `top_customer`, compute `rev_before` = revenue (quantity times unit price) of their order lines dated **before** `CHANGE_DATE`, and `rev_after` = on or after it,
# **from the star schema** (`fact_sales` joined to `dim_date`), any status. Then `rev_old_country_type2` = the revenue that Type 2 attributes to `old_country`.
# Both should be identical, which proves the SCD2 as-of join split the history at the right date.

# %% tags=["starter"]
rev_before = ...
rev_after = ...
rev_old_country_type2 = ...

# %%
lines_c = flat[flat["customer_id"] == top_customer].assign(rev=lambda d: d["quantity"] * d["unit_price"])
exp_before = lines_c.loc[lines_c["order_date"] < CHANGE_DATE, "rev"].sum()
exp_after = lines_c.loc[lines_c["order_date"] >= CHANGE_DATE, "rev"].sum()
assert abs(rev_before - exp_before) < 0.01, f"expected {exp_before:.2f} before the change date"
assert abs(rev_after - exp_after) < 0.01, f"expected {exp_after:.2f} on/after the change date"
assert abs(rev_old_country_type2 - rev_before) < 0.01, "type 2 should attribute exactly the pre-change revenue to the old country"
print("Exercise 3 passed")

# %% tags=["solution"]
r = con.sql(f"""
SELECT sum(f.revenue) FILTER (WHERE d.date <  DATE '{CHANGE_DATE}') AS b,
       sum(f.revenue) FILTER (WHERE d.date >= DATE '{CHANGE_DATE}') AS a
FROM fact_sales f JOIN dim_date d USING (date_key) JOIN dim_customer dc USING (customer_key)
WHERE dc.customer_id = {top_customer}""").fetchone()
rev_before, rev_after = float(r[0]), float(r[1])
rev_old_country_type2 = float(con.sql(f"""
SELECT sum(f.revenue) FROM fact_sales f JOIN dim_customer dc USING (customer_key)
WHERE dc.customer_id = {top_customer} AND dc.country = '{old_country}'""").fetchone()[0])

# %% [markdown]
# ### Exercise 4: Catch a fan-out
# Someone loads facts by joining `dim_customer` on `customer_id` alone (no as-of condition). Run that query to build `bad_fact` (a DataFrame of `item_id` and `revenue` columns
# is enough) and store in `inflated_by` how much **total revenue is overstated** compared with `fact_sales` (a number, currency units).
# *Hint: only the moved customer has two versions.*

# %% tags=["starter"]
inflated_by = ...

# %%
true_total = con.sql("SELECT sum(revenue) FROM fact_sales").fetchone()[0]
extra = flat[flat["customer_id"] == top_customer].eval("quantity * unit_price").sum()   # every line of that customer is counted twice
assert abs(inflated_by - extra) < 0.01, f"expected about {extra:.2f}"
print(f"Exercise 4 passed: the naive join overstates revenue by {inflated_by:,.2f} ({inflated_by / true_total:.2%})")

# %% tags=["solution"]
bad_fact = con.sql("""
SELECT oi.item_id, oi.quantity * p.unit_price AS revenue
FROM order_item oi JOIN orders o USING (order_id) JOIN product p USING (product_id)
JOIN dim_customer dc ON dc.customer_id = o.customer_id
""").df()
inflated_by = float(bad_fact["revenue"].sum() - con.sql("SELECT sum(revenue) FROM fact_sales").fetchone()[0])

# %% [markdown]
# ### Exercise 5 (open-ended): Model something new
# Sketch a star schema for `sensors.csv` (hourly temperature and power readings) or `ab_test.csv`. State the grain, the measures, and at least three dimensions, and say which
# dimension attributes you would track with Type 2 and why. Write it in a markdown cell.

# %% [markdown]
# ## Recap
# - **Functional dependencies** (`A -> B`) reveal redundancy; 3NF removes it so each fact lives in one place. Keys and foreign keys enforce integrity (DuckDB does; many warehouses do not).
# - A **star schema** = one fact table at a declared grain + denormalised dimensions with surrogate keys. Check the grain after loading.
# - **SCD Type 1** overwrites (history lost); **Type 2** closes and inserts versions with validity dates, and facts join on the version valid at event time.
# - Joining a fact to an SCD2 dimension on the business key alone multiplies rows; check for fan-out.
#
# **Go deeper:** the DuckDB SQL documentation (duckdb.org/docs); Kimball's dimensional modelling techniques (Ralph Kimball, *The Data Warehouse Toolkit*) for the full catalogue of SCD types and fact-table patterns.
#
# **Next:** [Data quality and validation](04_data_quality_and_validation.ipynb)
