# %% [markdown]
# # DDL, DML, Constraints and Transactions
#
# **Why this matters.** Analysts mostly read data, but sooner or later you create a staging table, backfill a column, fix bad rows, or load model output
# back into a database. Doing that safely is a skill: a forgotten `WHERE` on an `UPDATE` has ended careers. Constraints stop bad data at the door, and transactions make a
# multi-step change all-or-nothing. Data engineers will expect you to know this.
#
# **You will learn**
# - define tables with `CREATE TABLE` (types, `PRIMARY KEY`, `NOT NULL`, `UNIQUE`, `CHECK`, `DEFAULT`, `FOREIGN KEY`)
# - change data with `INSERT` (also `INSERT ... SELECT`), `UPDATE`, `DELETE` and upsert
# - see constraints reject bad rows, and know SQLite's quirks (foreign keys off by default, flexible types)
# - group statements into a **transaction** and `ROLLBACK` when something fails
#
# **Prerequisites:** [Window functions](05_window_functions.ipynb) (we reuse joins and aggregation).
#
# > Data used: a **temporary copy** of `datasets/shop.db`. The original is never modified, and the copy is deleted at the end of the notebook.
# > Terms: **DDL** = data definition language (`CREATE`, `ALTER`, `DROP`); **DML** = data manipulation language (`INSERT`, `UPDATE`, `DELETE`).

# %%
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

tmp = tempfile.TemporaryDirectory()
work_db = Path(tmp.name) / "shop_copy.db"
shutil.copy("../datasets/shop.db", work_db)

# isolation_level=None = autocommit: Python will not open transactions behind our back; we write BEGIN/COMMIT ourselves.
w = sqlite3.connect(work_db, isolation_level=None)
w.execute("PRAGMA foreign_keys = ON")   # SQLite ships with this OFF; more on that below

# the original, read-only, for comparisons in the exercise checks
ro = sqlite3.connect("file:../datasets/shop.db?mode=ro", uri=True)


def wq(sql, params=()):
    return pd.read_sql(sql, w, params=params)


wq("SELECT COUNT(*) AS n_customers FROM customers")

# %% [markdown]
# ## 1. CREATE TABLE and constraints
#
# A **constraint** is a rule the database enforces on every write, whoever writes (your notebook, a web app, a colleague's script). Rules in the database beat rules in
# application code because they cannot be bypassed.
#
# | Constraint | Means |
# |---|---|
# | `PRIMARY KEY` | unique + not null; identifies the row |
# | `NOT NULL` | a value is required |
# | `UNIQUE` | no duplicates in this column (NULLs excepted) |
# | `CHECK (expr)` | any condition that must be true |
# | `DEFAULT x` | value used when none is given |
# | `REFERENCES t(col)` | **foreign key**: value must exist in the parent table |
#
# We create a loyalty programme table: one account per customer, a tier from a fixed list, and non-negative points.

# %%
w.execute("""
CREATE TABLE loyalty_accounts (
    account_id  INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL UNIQUE REFERENCES customers(customer_id),
    tier        TEXT    NOT NULL CHECK (tier IN ('bronze', 'silver', 'gold')),
    points      INTEGER NOT NULL DEFAULT 0 CHECK (points >= 0),
    created_at  TEXT    NOT NULL DEFAULT (date('now'))
)
""")
wq("PRAGMA table_info(loyalty_accounts)")[["name", "type", "notnull", "dflt_value", "pk"]]

# %% [markdown]
# `INTEGER PRIMARY KEY` in SQLite auto-assigns ids (in Postgres you write `GENERATED ALWAYS AS IDENTITY` or `SERIAL`; in MySQL `AUTO_INCREMENT`).
# Now insert a first row. It fails, and the message is instructive:

# %%
try:
    w.execute("INSERT INTO loyalty_accounts (customer_id, tier) VALUES (1, 'bronze')")
except sqlite3.Error as e:
    print(type(e).__name__, "->", e)

# %% [markdown]
# A foreign key must point at a column that is a `PRIMARY KEY` or has a `UNIQUE` index in the parent. The shop tables were bulk-loaded from CSV **without any constraints** (check with
# `SELECT sql FROM sqlite_master`), so `customers.customer_id` is not declared unique. That is a real situation: data lakes and pandas `to_sql` often produce constraint-free tables. We add the missing uniqueness (which also *verifies* it, since the index creation fails if duplicates exist):

# %%
w.execute("CREATE UNIQUE INDEX ux_customers_id ON customers(customer_id)")
w.execute("INSERT INTO loyalty_accounts (customer_id, tier) VALUES (1, 'bronze')")
wq("SELECT * FROM loyalty_accounts")

# %% [markdown]
# Notice we did not give `account_id`, `points` or `created_at`: the primary key and the `DEFAULT`s filled them in.

# %% [markdown]
# ## 2. INSERT variants
# One row, many rows, or rows produced by a query (`INSERT ... SELECT`, the workhorse of ETL). Below, every premium customer except customer 1 gets a silver account
# with 10 points per order they placed.

# %%
w.execute("""
INSERT INTO loyalty_accounts (customer_id, tier, points)
SELECT c.customer_id, 'silver', 10 * COUNT(o.order_id)
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id
WHERE c.is_premium = 1 AND c.customer_id <> 1
GROUP BY c.customer_id
""")
w.executemany("INSERT INTO loyalty_accounts (customer_id, tier, points) VALUES (?, ?, ?)", [(2, "gold", 500), (3, "bronze", 20)])  # ? = parameters, see notebook 09
wq("SELECT tier, COUNT(*) AS n, SUM(points) AS points FROM loyalty_accounts GROUP BY tier ORDER BY tier")

# %% [markdown]
# Always list the target columns (`INSERT INTO t (a, b) ...`): without the list, the statement silently breaks when someone adds a column.

# %% [markdown]
# ## 3. Watching constraints work
# Each statement below violates a different rule; each is rejected and **nothing is written**.

# %%
bad_rows = {
    "duplicate customer (UNIQUE)":     "INSERT INTO loyalty_accounts (customer_id, tier) VALUES (1, 'gold')",
    "unknown tier (CHECK)":            "INSERT INTO loyalty_accounts (customer_id, tier) VALUES (4, 'platinum')",
    "negative points (CHECK)":         "INSERT INTO loyalty_accounts (customer_id, tier, points) VALUES (4, 'gold', -5)",
    "missing tier (NOT NULL)":         "INSERT INTO loyalty_accounts (customer_id) VALUES (4)",
    "no such customer (FOREIGN KEY)":  "INSERT INTO loyalty_accounts (customer_id, tier) VALUES (999999, 'gold')",
}
for label, sql in bad_rows.items():
    try:
        w.execute(sql)
        print(f"{label:34s} -> ACCEPTED (unexpected)")
    except sqlite3.IntegrityError as e:
        print(f"{label:34s} -> rejected: {e}")

# %% [markdown]
# `IntegrityError` is Python's name for a constraint violation, so application code can catch it specifically.
#
# **SQLite quirks worth knowing** (other engines are stricter):
# 1. Foreign keys are only enforced if you run `PRAGMA foreign_keys = ON` **on every connection**. Without it the last insert above would have been accepted.
# 2. Column types are only *hints* ("affinity"): you can store text in an `INTEGER` column. Declare a table `STRICT` to get real type checks.

# %%
m = sqlite3.connect(":memory:")
m.execute("CREATE TABLE loose (x INTEGER)")
m.execute("INSERT INTO loose VALUES ('not a number')")
print("loose table accepted:", m.execute("SELECT x, typeof(x) FROM loose").fetchall())
m.execute("CREATE TABLE tight (x INTEGER) STRICT")
try:
    m.execute("INSERT INTO tight VALUES ('not a number')")
except sqlite3.Error as e:
    print("STRICT table rejected:", e)
m.close()

# %% [markdown]
# ## 4. UPDATE and DELETE: the SELECT-first habit
#
# `UPDATE` and `DELETE` without `WHERE` hit **every row**. The safe routine: write the `WHERE` in a `SELECT` first, look at what it matches, then change the verb.

# %%
wq("SELECT account_id, customer_id, tier, points FROM loyalty_accounts WHERE tier = 'silver' AND points >= 100 ORDER BY points DESC LIMIT 5")

# %%
cur = w.execute("UPDATE loyalty_accounts SET tier = 'gold' WHERE tier = 'silver' AND points >= 100")
print("rows updated:", cur.rowcount)
wq("SELECT tier, COUNT(*) AS n FROM loyalty_accounts GROUP BY tier ORDER BY tier")

# %% [markdown]
# `cursor.rowcount` tells you how many rows a statement touched: assert on it in scripts (expect 1 row, got 0 or 5000 means something is wrong).
# `UPDATE ... SET a = a + 1` uses the current value; `DELETE FROM t WHERE ...` removes rows. `TRUNCATE` (Postgres/MySQL/Snowflake) empties a table quickly but
# does not exist in SQLite (use `DELETE FROM t`).

# %% [markdown]
# ## 5. Upsert: insert or update
# "Add points for customer 3, creating the account if needed" is a race-prone read-then-write in application code. SQL does it atomically with `ON CONFLICT`:

# %%
upsert = """
INSERT INTO loyalty_accounts (customer_id, tier, points) VALUES (?, 'bronze', ?)
ON CONFLICT (customer_id) DO UPDATE SET points = points + excluded.points
"""
new_id = int(wq("SELECT MIN(customer_id) AS c FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM loyalty_accounts)").c[0])
w.execute(upsert, (3, 15))       # account exists: 20 + 15
w.execute(upsert, (new_id, 15))  # no account yet: plain insert
wq("SELECT customer_id, tier, points FROM loyalty_accounts WHERE customer_id IN (?, ?)", (3, new_id))

# %% [markdown]
# `excluded` refers to the row you tried to insert. Spelling differs: Postgres and SQLite use `ON CONFLICT`, MySQL `ON DUPLICATE KEY UPDATE`, Snowflake/BigQuery/SQL Server `MERGE`.

# %% [markdown]
# ## 6. Transactions: all or nothing
#
# A **transaction** groups statements so they succeed or fail together (the *A* in ACID: atomicity). Classic example: move points from one account to another. If the second step fails after the first succeeded,
# points vanish. Here is what happens **without** a transaction when the credit fails:

# %%
def points(cid):
    return int(wq("SELECT points FROM loyalty_accounts WHERE customer_id = ?", (cid,)).points[0])


print("before:", points(2), points(3))
w.execute("UPDATE loyalty_accounts SET points = points - 100 WHERE customer_id = 2")       # succeeds and is auto-committed
try:
    w.execute("UPDATE loyalty_accounts SET points = points + 100, tier = 'diamond' WHERE customer_id = 3")  # violates CHECK
except sqlite3.IntegrityError as e:
    print("credit failed:", e)
print("after :", points(2), points(3), "  <- 100 points vanished")
w.execute("UPDATE loyalty_accounts SET points = points + 100 WHERE customer_id = 2")      # put them back by hand

# %% [markdown]
# Now the same failure inside a transaction: `BEGIN`, do the steps, `COMMIT` on success, `ROLLBACK` on any error.

# %%
def transfer(con, src, dst, pts, dst_tier=None):
    try:
        con.execute("BEGIN")
        con.execute("UPDATE loyalty_accounts SET points = points - ? WHERE customer_id = ?", (pts, src))
        con.execute("UPDATE loyalty_accounts SET points = points + ?, tier = COALESCE(?, tier) WHERE customer_id = ?", (pts, dst_tier, dst))
        con.execute("COMMIT")
        return True
    except sqlite3.Error as e:
        con.execute("ROLLBACK")
        print("  rolled back:", e)
        return False


print("before:", points(2), points(3))
ok = transfer(w, 2, 3, 100, dst_tier="diamond")
print("ok =", ok, "| after:", points(2), points(3), "  <- nothing changed")
ok = transfer(w, 2, 3, 100)
print("ok =", ok, "| after:", points(2), points(3), "  <- both steps applied")

# %% [markdown]
# *What just happened?* In the failing call the debit had already executed inside the transaction, and `ROLLBACK` undid it. The database went back to the exact state before `BEGIN`.
# Also handy: try a dangerous statement inside a transaction, inspect the damage, then roll back.

# %%
w.execute("BEGIN")
w.execute("DELETE FROM loyalty_accounts")             # oops, no WHERE
print("inside transaction:", wq("SELECT COUNT(*) AS n FROM loyalty_accounts").n[0], "accounts")
w.execute("ROLLBACK")
print("after rollback    :", wq("SELECT COUNT(*) AS n FROM loyalty_accounts").n[0], "accounts")

# %% [markdown]
# **Other DDL**, briefly: `ALTER TABLE t ADD COLUMN c TYPE` (adds a column; SQLite can also rename/drop columns in modern versions), `CREATE TABLE t2 AS SELECT ...` (materialise a query result),
# `DROP TABLE t` (gone, and often without an undo; in SQLite/Postgres DDL is transactional, in MySQL it is not).

# %%
w.execute("ALTER TABLE loyalty_accounts ADD COLUMN note TEXT")
w.execute("CREATE TABLE gold_members AS SELECT customer_id, points FROM loyalty_accounts WHERE tier = 'gold'")
print(wq("SELECT COUNT(*) AS n FROM gold_members").n[0], "gold members;", "columns of loyalty_accounts:", wq("PRAGMA table_info(loyalty_accounts)").name.tolist())
w.execute("DROP TABLE gold_members")

# %% [markdown]
# **Dialect notes:** auto ids (`INTEGER PRIMARY KEY` here, `SERIAL`/`IDENTITY` Postgres, `AUTO_INCREMENT` MySQL, `IDENTITY`/`AUTOINCREMENT` Snowflake); real `DATE`/`TIMESTAMP`/`BOOLEAN` types exist in Postgres, MySQL and Snowflake (SQLite stores text/integers);
# `ALTER COLUMN` type changes are limited in SQLite. Snowflake accepts primary/foreign key declarations **but does not enforce them** (except `NOT NULL`), so uniqueness must be checked by your pipeline, an important thing to know for warehouses.

# %% [markdown]
# ## Common mistakes
#
# 1. **`UPDATE`/`DELETE` without `WHERE`.** SELECT first; run inside a transaction; check `rowcount`.
# 2. **Assuming foreign keys are enforced** in SQLite (need the PRAGMA per connection) or in warehouses like Snowflake (declared only).
# 3. **No transaction around multi-step changes**, so a crash halfway leaves half-applied data.
# 4. **`INSERT` without a column list**, which breaks on schema change.
# 5. **Forgetting to `COMMIT`.** In Postgres/MySQL clients with autocommit off, your change is invisible to others and lost on disconnect. Python's `sqlite3` also needs `con.commit()` unless autocommit is on.
# 6. **Storing money in floating point.** Use integer cents or `DECIMAL` (SQLite has no true decimal type). Our products table uses `REAL` for simplicity.

# %% [markdown]
# ## Interview corner
#
# **1. What are the ACID properties?**
# <details><summary>Answer</summary>
# Atomicity (all or nothing), Consistency (constraints hold before and after), Isolation (concurrent transactions do not see each other's half-finished work), Durability (committed data survives a crash).
# </details>
#
# **2. <code>DELETE</code>, <code>TRUNCATE</code> and <code>DROP</code>?**
# <details><summary>Answer</summary>
# <code>DELETE</code> removes rows (optionally filtered, logged, can be rolled back). <code>TRUNCATE</code> quickly removes all rows of a table (no filter; transactional behaviour depends on the engine). <code>DROP</code> removes the table itself.
# </details>
#
# **3. How do you make sure an <code>UPDATE</code> only touches the rows you intend?**
# <details><summary>Answer</summary>
# Write the <code>WHERE</code> as a <code>SELECT</code> first and count rows, run the change inside a transaction, check the affected-row count, and roll back if it is unexpected.
# </details>
#
# **4. Primary key versus unique key?**
# <details><summary>Answer</summary>
# Both enforce uniqueness. A table has one primary key (not null, the row's identity) and can have several unique constraints; unique columns may allow NULLs (behaviour differs a bit between engines).
# </details>

# %% [markdown]
# ## Exercises
# These run against the temporary copy `w`. Each check resets what it needs so you can re-run cells freely.

# %% [markdown]
# ### Exercise 1: Design a returns log
# Write the `CREATE TABLE` for `returns_log` with: `return_id` integer primary key; `order_id` integer, required; `reason` text, required, only `'damaged'`, `'wrong_item'` or `'changed_mind'`;
# `refund_amount` real, required, never negative. Store the statement in `ex1_sql`.

# %% tags=["starter"]
ex1_sql = """
CREATE TABLE returns_log (
    ...
)
"""

# %%
w.execute("DROP TABLE IF EXISTS returns_log")
w.execute(ex1_sql)
info = wq("PRAGMA table_info(returns_log)").set_index("name")
assert set(info.index) == {"return_id", "order_id", "reason", "refund_amount"}, f"columns: {list(info.index)}"
assert info.loc["return_id", "pk"] == 1, "return_id must be the primary key"
assert info.loc["order_id", "notnull"] == 1 and info.loc["reason", "notnull"] == 1 and info.loc["refund_amount", "notnull"] == 1, "order_id, reason, refund_amount must be NOT NULL"
w.execute("INSERT INTO returns_log (order_id, reason, refund_amount) VALUES (2, 'damaged', 10.5)")
for bad in ["INSERT INTO returns_log (order_id, reason, refund_amount) VALUES (2, 'boredom', 5)",
            "INSERT INTO returns_log (order_id, reason, refund_amount) VALUES (2, 'damaged', -1)",
            "INSERT INTO returns_log (order_id, reason) VALUES (2, 'damaged')",
            "INSERT INTO returns_log (reason, refund_amount) VALUES ('damaged', 1)"]:
    try:
        w.execute(bad)
        raise AssertionError(f"should have been rejected: {bad}")
    except sqlite3.IntegrityError:
        pass
w.execute("DELETE FROM returns_log")
print("Exercise 1 passed")

# %% tags=["solution"]
ex1_sql = """
CREATE TABLE returns_log (
    return_id     INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    reason        TEXT    NOT NULL CHECK (reason IN ('damaged', 'wrong_item', 'changed_mind')),
    refund_amount REAL    NOT NULL CHECK (refund_amount >= 0)
)
"""

# %% [markdown]
# ### Exercise 2: Backfill with INSERT ... SELECT
# Fill `returns_log` with **one row per order whose status is `returned`**: `order_id`, reason `'changed_mind'`, and `refund_amount` = the order's total value (`quantity * unit_price` summed over its lines, rounded to 2 decimals).
# Store the statement in `ex2_sql`. *Hint: join `orders`, `order_items`, `products`; group by order; one row per order, not per line.*

# %% tags=["starter"]
ex2_sql = """
INSERT INTO returns_log (order_id, reason, refund_amount)
SELECT ...
"""

# %%
w.execute("DELETE FROM returns_log")
cur = w.execute(ex2_sql)
got = wq("SELECT order_id, reason, refund_amount FROM returns_log ORDER BY order_id")
o = pd.read_sql("SELECT * FROM orders", ro)
i = pd.read_sql("SELECT * FROM order_items", ro)
p = pd.read_sql("SELECT * FROM products", ro)
ret = o[o.status == "returned"].merge(i, on="order_id").merge(p, on="product_id")
exp = (ret.assign(v=ret.quantity * ret.unit_price).groupby("order_id").v.sum().round(2)).reset_index().sort_values("order_id").reset_index(drop=True)
assert len(exp) > 0
assert len(got) == len(exp), f"expected one row per returned order ({len(exp)}), got {len(got)}"
assert got.order_id.tolist() == exp.order_id.tolist(), "wrong set of orders"
assert (got.reason == "changed_mind").all(), "reason must be 'changed_mind'"
assert ((got.refund_amount - exp.v).abs() < 0.02).all(), "refund_amount should be the order total (sum over its lines)"
print(f"Exercise 2 passed ({len(got)} rows, total refunds {got.refund_amount.sum():,.2f})")

# %% tags=["solution"]
ex2_sql = """
INSERT INTO returns_log (order_id, reason, refund_amount)
SELECT o.order_id, 'changed_mind', ROUND(SUM(i.quantity * p.unit_price), 2)
FROM orders o
JOIN order_items i ON i.order_id = o.order_id
JOIN products p    ON p.product_id = i.product_id
WHERE o.status = 'returned'
GROUP BY o.order_id
"""

# %% [markdown]
# ### Exercise 3: A targeted UPDATE
# Raise the `unit_price` of every **Sports** product by 10% (rounded to 2 decimals) in the working copy, leaving other categories untouched. Store the statement in `ex3_sql`.
# Then, in the checks, we compare against the untouched original.

# %% tags=["starter"]
ex3_sql = """
UPDATE products
SET ...
WHERE ...
"""

# %%
orig = pd.read_sql("SELECT product_id, category, unit_price FROM products", ro)
w.executemany("UPDATE products SET unit_price = ? WHERE product_id = ?", list(zip(orig.unit_price, orig.product_id)))   # reset the copy so re-runs are safe
cur = w.execute(ex3_sql)
now = wq("SELECT product_id, category, unit_price FROM products").merge(orig, on="product_id", suffixes=("", "_orig"))
sports = now.category == "Sports"
assert cur.rowcount == int(sports.sum()), f"expected {int(sports.sum())} rows updated, got {cur.rowcount}: check the WHERE clause"
assert ((now.loc[sports, "unit_price"] - (now.loc[sports, "unit_price_orig"] * 1.1).round(2)).abs() < 0.006).all(), "Sports prices should be orig * 1.1 rounded to 2 decimals"
assert (now.loc[~sports, "unit_price"] == now.loc[~sports, "unit_price_orig"]).all(), "non-Sports products changed!"
print(f"Exercise 3 passed ({cur.rowcount} products updated)")

# %% tags=["solution"]
ex3_sql = """
UPDATE products
SET unit_price = ROUND(unit_price * 1.1, 2)
WHERE category = 'Sports'
"""

# %% [markdown]
# ### Exercise 4: An atomic transfer with an audit row
# Write `transfer_with_audit(con, src, dst, pts)` that, **in one transaction**, (1) subtracts `pts` from `src`'s points, (2) adds them to `dst`, and (3) inserts a row `(src, dst, pts)` into `points_ledger`
# (the check creates that table with `CHECK (pts > 0)`). On any `sqlite3.Error` it must roll back and return `False`; on success commit and return `True`.
# *Hint: the `transfer` function above is nearly what you need.*

# %% tags=["starter"]
def transfer_with_audit(con, src, dst, pts):
    ...

# %%
w.execute("DROP TABLE IF EXISTS points_ledger")
w.execute("CREATE TABLE points_ledger (src INTEGER NOT NULL, dst INTEGER NOT NULL, pts INTEGER NOT NULL CHECK (pts > 0))")
before = (points(2), points(3))
assert before[0] >= 50, "sanity: customer 2 needs at least 50 points"
assert transfer_with_audit(w, 2, 3, 50) is True
assert (points(2), points(3)) == (before[0] - 50, before[1] + 50), "a successful transfer should move 50 points"
assert wq("SELECT COUNT(*) AS n FROM points_ledger").n[0] == 1, "a successful transfer should write one ledger row"
mid = (points(2), points(3))
# overdraw: debit would make points negative -> CHECK fails on step 1
assert transfer_with_audit(w, 2, 3, mid[0] + 1) is False, "overdrawing must fail and return False"
assert (points(2), points(3)) == mid, "a failed transfer must leave both balances unchanged"
# ledger CHECK (pts > 0) fails on step 3, AFTER the two updates ran: must undo them
assert transfer_with_audit(w, 2, 3, 0) is False, "pts = 0 violates the ledger CHECK"
assert (points(2), points(3)) == mid, "steps 1 and 2 must be rolled back when step 3 fails"
assert wq("SELECT COUNT(*) AS n FROM points_ledger").n[0] == 1, "no ledger row for failed transfers"
print("Exercise 4 passed")

# %% tags=["solution"]
def transfer_with_audit(con, src, dst, pts):
    try:
        con.execute("BEGIN")
        con.execute("UPDATE loyalty_accounts SET points = points - ? WHERE customer_id = ?", (pts, src))
        con.execute("UPDATE loyalty_accounts SET points = points + ? WHERE customer_id = ?", (pts, dst))
        con.execute("INSERT INTO points_ledger (src, dst, pts) VALUES (?, ?, ?)", (src, dst, pts))
        con.execute("COMMIT")
        return True
    except sqlite3.Error:
        con.execute("ROLLBACK")
        return False

# %% [markdown]
# ### Exercise 5 (open-ended): Model a shipment
# Design (and create in `w`) a `shipments` table for the shop: which columns, which types, which constraints, which foreign key? Insert two valid rows and prove that two different invalid rows are rejected.
# Write a sentence on what you could **not** enforce with constraints alone (for example "a shipment date must be after the order date" needs another table's value).

# %% [markdown]
# ## Clean up
# We delete the temporary copy so nothing is left behind.

# %%
w.close()
ro.close()
tmp.cleanup()
print("temp copy removed:", not work_db.exists())

# %% [markdown]
# ## Recap
# - DDL defines structure (`CREATE`, `ALTER`, `DROP`); DML changes rows (`INSERT`, `UPDATE`, `DELETE`, upsert).
# - Constraints (`PRIMARY KEY`, `NOT NULL`, `UNIQUE`, `CHECK`, `FOREIGN KEY`, `DEFAULT`) enforce data quality at the database, for every writer.
# - SQLite quirks: enable `PRAGMA foreign_keys = ON` per connection; types are hints unless the table is `STRICT`. Warehouses like Snowflake do not enforce most constraints.
# - `SELECT` first, then `UPDATE`/`DELETE`; check `rowcount`.
# - Transactions make multi-step changes atomic: `BEGIN`, then `COMMIT` or `ROLLBACK` on error.
#
# **Go deeper:** SQLite docs on foreign keys (sqlite.org/foreignkeys.html), on transactions (sqlite.org/lang_transaction.html) and on upsert (sqlite.org/lang_upsert.html).
#
# **Next:** [Indexes and query plans](07_indexes_and_query_plans.ipynb)
