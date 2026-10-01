# %% [markdown]
# # Python Connector, Snowpark, and Testing Without an Account
#
# **Why this matters.** Real work happens in code, not the web UI: a scheduled Python job pulls yesterday's numbers,
# a notebook loads a feature table, a pipeline writes results back. The two mistakes that end careers here are
# **hard-coded credentials committed to Git** and **pipeline code that can only be tested against production**.
# This notebook shows the right connector/Snowpark patterns, and a *fake connection* so you can unit-test your
# pipeline code on a laptop with no Snowflake account.
#
# **You will learn**
# - load credentials from environment variables and fail with a clear message when they are missing
# - use the Python connector correctly: cursors, bound parameters, `fetch_pandas_all`, `write_pandas`
# - explain Snowpark's lazy DataFrame model (and see it in a toy that you run)
# - test pipeline functions against a local DuckDB "fake Snowflake" behind the same interface
#
# **Prerequisites:** [Snowflake SQL](02_snowflake_sql_and_dialect_differences.ipynb), plain Python functions and `pytest`-style assertions.
#
# > **Candour box.** `snowflake-connector-python` and `snowflake-snowpark-python` are **not installed** in this environment and there
# > was no account, so every cell that imports them is tagged `no-run` and was **not executed**. They are written from the
# > libraries' documented API; *verify parameter names and versions in the current docs* (docs.snowflake.com, "Developer" section)
# > before relying on them. The fake-connection code and the tests **are** executed here.

# %%
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

DATA = Path("../datasets")

# %% [markdown]
# ## 1. Credentials: environment variables, never source code
#
# Anything you type into a `.py` or `.ipynb` file ends up in Git history, in shared notebooks, and in screenshots. Read secrets from the
# **environment** (set by your shell, your orchestrator, or a secrets manager) and fail early if one is missing.
# Many organisations also use SSO, key-pair authentication or a secrets manager instead of passwords: *ask what yours requires*.
#
# The variable names below are **our own convention**; they are not mandated by Snowflake.

# %%
REQUIRED = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE"]


@dataclass(frozen=True)
class SnowflakeConfig:
    account: str
    user: str
    password: str = field(repr=False)          # repr=False: never printed in logs or tracebacks of the object
    warehouse: str
    database: str
    schema: str = "PUBLIC"
    role: str | None = None

    @classmethod
    def from_env(cls, env=os.environ) -> "SnowflakeConfig":
        missing = [k for k in REQUIRED if not env.get(k)]
        if missing:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
        return cls(
            account=env["SNOWFLAKE_ACCOUNT"], user=env["SNOWFLAKE_USER"], password=env["SNOWFLAKE_PASSWORD"],
            warehouse=env["SNOWFLAKE_WAREHOUSE"], database=env["SNOWFLAKE_DATABASE"],
            schema=env.get("SNOWFLAKE_SCHEMA", "PUBLIC"), role=env.get("SNOWFLAKE_ROLE"),
        )


# a fake environment dict: we never touch your real environment or put a secret in the notebook
fake_env = {"SNOWFLAKE_ACCOUNT": "demo_account", "SNOWFLAKE_USER": "demo_user", "SNOWFLAKE_PASSWORD": "placeholder",
            "SNOWFLAKE_WAREHOUSE": "DEV_WH", "SNOWFLAKE_DATABASE": "DEV_DB"}
cfg = SnowflakeConfig.from_env(fake_env)
print(cfg)  # the password field is not displayed

try:
    SnowflakeConfig.from_env({"SNOWFLAKE_USER": "x"})
except RuntimeError as e:
    print("RuntimeError:", e)

# %% [markdown]
# *What just happened?* A frozen dataclass holds settings; the secret is excluded from `repr`, and a missing variable fails **at start-up with the list
# of names**, instead of a cryptic authentication error 20 lines later. Also add `.env` files to `.gitignore` if you use them locally.

# %% [markdown]
# ## 2. The Python connector (not executed here)
#
# The mental model: a **connection** (authenticated session, uses a warehouse) creates **cursors**; a cursor runs SQL and fetches rows.
# Use context managers so connections close (an open session does not by itself keep a warehouse running, but leaks are untidy and can hit limits).

# %% tags=["no-run"]
# Requires: pip install "snowflake-connector-python[pandas]"   (verify extras name in the current docs)
import os
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

with snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],      # or key-pair / SSO authentication, per your organisation's policy
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
) as conn:
    with conn.cursor() as cur:
        # bound parameters (pyformat style by default): the driver sends values separately from the SQL text
        cur.execute(
            "SELECT channel, SUM(revenue) AS revenue FROM order_lines WHERE order_date >= %(start)s GROUP BY channel",
            {"start": "2025-01-01"},
        )
        df = cur.fetch_pandas_all()          # needs the pandas extra; column names come back UPPER CASE
        df.columns = df.columns.str.lower()

    # writing a DataFrame back: bulk-loads through a stage (see notebook 04). By default write_pandas quotes identifiers, so the
    # DataFrame's column names are used exactly as given (case-sensitive): upper-case them for ordinary unquoted-style columns (verify `quote_identifiers` in the docs).
    df.columns = df.columns.str.upper()
    ok, n_chunks, n_rows, _ = write_pandas(conn, df, table_name="CHANNEL_REVENUE", auto_create_table=True)

# %% [markdown]
# Things worth knowing about the real connector (verify details in the docs):
# - `SUM` over a `NUMBER` column typically arrives in Python as `decimal.Decimal`, and `DATE` as `datetime.date`. Convert types explicitly before doing arithmetic with floats.
# - Column names are upper case unless quoted at creation time.
# - Bind parameters for **values**, never for identifiers (table/column names): those you must validate against an allow-list yourself.
# - Pulling millions of rows into pandas defeats the purpose. Aggregate in SQL and fetch the result.

# %% [markdown]
# ## 3. Snowpark: lazy DataFrames that run inside Snowflake
#
# Snowpark gives a DataFrame API (Python, also Java/Scala) that looks a bit like PySpark. The key idea: **transformations do not fetch data.**
# `session.table(...).filter(...).group_by(...).agg(...)` only builds a query plan; it is translated to SQL and executed **inside Snowflake's warehouse** when you call an
# action such as `.collect()`, `.show()` or `.to_pandas()`. Your Python process never sees the raw rows.

# %% tags=["no-run"]
# Requires: pip install snowflake-snowpark-python   (verify current supported Python versions in the docs)
import os
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, sum as sum_

session = Session.builder.configs({
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    "database": os.environ["SNOWFLAKE_DATABASE"],
    "schema": os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
}).create()

channel_rev = (
    session.table("ORDER_LINES")
    .filter(col("STATUS").isin("delivered", "shipped"))
    .group_by("CHANNEL")
    .agg(sum_(col("REVENUE")).alias("REVENUE"))
)                                   # nothing has run yet
print(channel_rev.queries)          # inspect the generated SQL (check attribute name in the current docs)
pdf = channel_rev.to_pandas()       # NOW it executes, and only the small result comes back
session.close()

# %% [markdown]
# **See laziness by building it.** The toy below records operations, generates SQL text, and executes only on `collect`. It is a teaching model,
# not Snowpark's implementation, but the behaviour (no work until an action) is the same, and you can *count* the queries it sends.

# %%
class FakeCursor:
    """Mimics the parts of a Snowflake cursor we use, on top of DuckDB."""

    def __init__(self, con, log):
        self._con, self._log, self._rows, self.description = con, log, [], None

    def execute(self, sql, params=None):
        self._log.append(sql)
        native = re.sub(r"%\((\w+)\)s", r"$\1", sql)          # pyformat -> DuckDB named parameters
        rel = self._con.execute(native, params or {})
        self._rows = rel.fetchall()
        self.description = [(d[0].upper(), *d[1:]) for d in rel.description]   # Snowflake returns UPPER CASE names
        return self

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetch_pandas_all(self):
        return pd.DataFrame(self._rows, columns=[d[0] for d in self.description])

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeSnowflakeConnection:
    """Same interface as the connection our pipeline code uses; backed by an in-memory DuckDB."""

    def __init__(self, data_dir=DATA):
        self._con = duckdb.connect()
        for t in ["customers", "products", "orders", "order_items"]:
            self._con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{(Path(data_dir) / (t + '.csv')).as_posix()}')")
        self.queries: list[str] = []

    def cursor(self):
        return FakeCursor(self._con, self.queries)

    def close(self):
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LazyFrame:
    """Toy of Snowpark's lazy model: methods return new objects; only collect() touches the connection."""

    def __init__(self, table, wheres=(), group=(), aggs=()):
        self.table, self.wheres, self.group, self.aggs = table, tuple(wheres), tuple(group), tuple(aggs)

    def filter(self, cond):
        return LazyFrame(self.table, self.wheres + (cond,), self.group, self.aggs)

    def group_by(self, *cols):
        return LazyFrame(self.table, self.wheres, cols, self.aggs)

    def agg(self, *exprs):
        return LazyFrame(self.table, self.wheres, self.group, exprs)

    @property
    def sql(self):
        sel = ", ".join(list(self.group) + list(self.aggs)) or "*"
        q = f"SELECT {sel} FROM {self.table}"
        if self.wheres:
            q += " WHERE " + " AND ".join(f"({w})" for w in self.wheres)
        if self.group:
            q += " GROUP BY " + ", ".join(self.group)
        return q

    def collect(self, conn):
        with conn.cursor() as cur:
            return cur.execute(self.sql).fetch_pandas_all()


conn = FakeSnowflakeConnection()
plan = LazyFrame("orders").filter("status IN ('delivered','shipped')").group_by("channel").agg("COUNT(*) AS n_orders")
print("queries sent so far:", len(conn.queries))
print(plan.sql)
result = plan.collect(conn)
print("queries sent after collect:", len(conn.queries))
result

# %% [markdown]
# *What just happened?* Building the plan sent **zero** queries; `collect` sent **one**. Consequences with real Snowpark: you can compose transformations cheaply, Snowflake's optimiser sees
# the whole plan at once, and forgetting an action means nothing runs (a classic beginner confusion). Note also the column names came back **upper case** (`CHANNEL`, `N_ORDERS`), exactly as the real
# connector would return them.

# %% [markdown]
# ## 4. Pipeline code that can be tested: depend on an interface
#
# Write pipeline functions that **receive a connection** instead of creating one inside. In production you pass the real Snowflake connection; in tests you pass
# `FakeSnowflakeConnection`. Keep the SQL to the portable subset (joins, `SUM`, `GROUP BY`, `CASE`, bound parameters) so it runs on both. For anything Snowflake-specific
# (`QUALIFY` works in both; `FLATTEN` and Time Travel do not) you would test against a real dev account instead.

# %%
COMPLETED = "('delivered', 'shipped')"


def revenue_by_channel(conn) -> pd.DataFrame:
    """Revenue of completed orders (delivered or shipped) per sales channel."""
    sql = f"""
        SELECT o.channel AS channel, SUM(oi.quantity * p.unit_price) AS revenue
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN products p ON p.product_id = oi.product_id
        WHERE o.status IN {COMPLETED}
        GROUP BY o.channel
        ORDER BY o.channel
    """
    with conn.cursor() as cur:
        df = cur.execute(sql).fetch_pandas_all()
    df.columns = df.columns.str.lower()            # normalise: Snowflake returns UPPER CASE
    df["revenue"] = df["revenue"].astype(float)    # real connector may return Decimal
    return df


# a plain-pandas ground truth, computed independently of SQL
orders = pd.read_csv(DATA / "orders.csv")
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
truth = (
    items.merge(orders, on="order_id").merge(products, on="product_id")
    .query("status in ['delivered', 'shipped']")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"])
    .groupby("channel", as_index=False)["revenue"].sum()
)

got = revenue_by_channel(conn)
pd.testing.assert_frame_equal(got, truth, check_exact=False, rtol=1e-9)
got

# %% [markdown]
# *What just happened?* The function ran against the fake and matched an independent pandas computation. You have just written the equivalent of a unit test for warehouse code.
# What it does **not** prove: that Snowflake's types, permissions or performance behave identically. Keep a small integration test against a dev account for that.
#
# **Bind parameters, do not concatenate.** Here is why, on the fake:

# %%
def count_orders_unsafe(conn, channel):
    with conn.cursor() as cur:
        return cur.execute(f"SELECT COUNT(*) FROM orders WHERE channel = '{channel}'").fetchone()[0]


def count_orders(conn, channel):
    with conn.cursor() as cur:
        return cur.execute("SELECT COUNT(*) FROM orders WHERE channel = %(ch)s", {"ch": channel}).fetchone()[0]


evil = "web' OR '1'='1"
print("unsafe, hostile input :", count_orders_unsafe(conn, evil), "(every row leaked into the count)")
print("bound,  hostile input :", count_orders(conn, evil))
print("bound,  normal input  :", count_orders(conn, "web"))

# %% [markdown]
# ## 5. Choosing the connection at run time
#
# A small factory keeps the switch in one place: `USE_FAKE_SNOWFLAKE=1` gives the fake (CI, laptops), otherwise the real connector is imported lazily.

# %%
def get_connection(env=os.environ):
    if env.get("USE_FAKE_SNOWFLAKE") == "1":
        return FakeSnowflakeConnection()
    import snowflake.connector                      # imported lazily: not needed when using the fake

    cfg = SnowflakeConfig.from_env(env)
    return snowflake.connector.connect(account=cfg.account, user=cfg.user, password=cfg.password,
                                       warehouse=cfg.warehouse, database=cfg.database, schema=cfg.schema)


with get_connection({"USE_FAKE_SNOWFLAKE": "1"}) as c:
    print(revenue_by_channel(c).round(0).to_dict("records")[:2])

# %% [markdown]
# ## Common mistakes
#
# 1. **Credentials in code or notebooks.** Rotate them if this ever happens; deleting the commit is not enough.
# 2. **f-string SQL with user input** (shown above). Use bound parameters for values.
# 3. **Forgetting an action on a lazy Snowpark DataFrame**: no error, no result.
# 4. **`SELECT *` then filtering in pandas**: you pay to move data you throw away. Push filters and aggregates to SQL.
# 5. **Ignoring type differences** (`Decimal`, upper-case names) between the fake and the real service. Normalise them in one place, as `revenue_by_channel` does.
# 6. **Creating a new connection for every small query in a loop**: slow; reuse one.

# %% [markdown]
# ## Interview corner
#
# **1. How do you handle credentials in a Snowflake pipeline?**
# <details><summary>Answer</summary>
# Never in code. Read them from environment variables or a secrets manager injected at run time, prefer key-pair or SSO/OAuth where the organisation supports it, use a dedicated service user with a
# least-privilege role, and rotate on exposure.
# </details>
#
# **2. What is the difference between the Python connector and Snowpark?**
# <details><summary>Answer</summary>
# The connector is a DB-API-style driver: you send SQL text and fetch rows to the client. Snowpark is a lazy DataFrame API whose operations are compiled to SQL and run inside Snowflake, so
# data does not leave the warehouse until you ask for a result. Snowpark also lets you deploy Python code (UDFs, stored procedures) to run server-side.
# </details>
#
# **3. How would you test code that queries Snowflake in CI?**
# <details><summary>Answer</summary>
# Inject the connection. Unit-test against a local stand-in (DuckDB/SQLite) for portable SQL, and run a small integration suite against a dev account or a zero-copy clone for Snowflake-specific behaviour
# and types. Compare outputs to an independent computation.
# </details>
#
# **4. Why bind parameters instead of string formatting?**
# <details><summary>Answer</summary>
# Values travel separately from the SQL text, so input cannot change the query structure (SQL injection), and quoting/type conversion is handled by the driver. Identifiers cannot be bound; validate those against an allow-list.
# </details>

# %% [markdown]
# ## Exercises
# `conn` is a `FakeSnowflakeConnection`; `orders`, `items`, `products` are pandas DataFrames of the same data for ground truth.

# %% [markdown]
# ### Exercise 1: Fail fast on missing configuration
# Write `missing_vars(env)` that returns the **sorted list** of names from `REQUIRED` that are absent or empty in the dict `env`.
# *Hint: a list comprehension over `REQUIRED` with `env.get(name)`.*

# %% tags=["starter"]
def missing_vars(env):
    ...

# %%
assert missing_vars(fake_env) == [], "a complete environment has nothing missing"
assert missing_vars({}) == sorted(REQUIRED), "an empty environment misses everything, returned sorted"
assert missing_vars({**fake_env, "SNOWFLAKE_PASSWORD": ""}) == ["SNOWFLAKE_PASSWORD"], "empty string counts as missing"
partial = {k: "x" for k in REQUIRED if k not in ("SNOWFLAKE_USER", "SNOWFLAKE_ACCOUNT")}
assert missing_vars(partial) == ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"]
print("Exercise 1 passed")

# %% tags=["solution"]
def missing_vars(env):
    return sorted(name for name in REQUIRED if not env.get(name))

# %% [markdown]
# ### Exercise 2: A testable pipeline function
# Write `top_customers(conn, n)` returning a DataFrame with lower-case columns `customer_id, revenue` for the **n customers with the highest revenue** on completed orders
# (delivered or shipped), highest first. Use a single SQL query through `conn.cursor()`, break ties by `customer_id`, and convert `revenue` to float.

# %% tags=["starter"]
def top_customers(conn, n):
    ...

# %%
_truth = (
    items.merge(orders, on="order_id").merge(products, on="product_id")
    .query("status in ['delivered', 'shipped']")
    .assign(revenue=lambda d: d["quantity"] * d["unit_price"])
    .groupby("customer_id", as_index=False)["revenue"].sum()
    .sort_values(["revenue", "customer_id"], ascending=[False, True], ignore_index=True)
)
_got = top_customers(conn, 5)
assert list(_got.columns) == ["customer_id", "revenue"], f"columns must be lower-case customer_id, revenue; got {list(_got.columns)}"
assert len(_got) == 5
assert _got["customer_id"].tolist() == _truth["customer_id"].head(5).tolist(), "wrong customers or wrong order"
assert abs(_got["revenue"].iloc[0] - _truth["revenue"].iloc[0]) < 1e-6
print("Exercise 2 passed")

# %% tags=["solution"]
def top_customers(conn, n):
    sql = f"""
        SELECT o.customer_id AS customer_id, SUM(oi.quantity * p.unit_price) AS revenue
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        JOIN products p ON p.product_id = oi.product_id
        WHERE o.status IN {COMPLETED}
        GROUP BY o.customer_id
        ORDER BY revenue DESC, o.customer_id
        LIMIT {int(n)}
    """
    with conn.cursor() as cur:
        df = cur.execute(sql).fetch_pandas_all()
    df.columns = df.columns.str.lower()
    df["revenue"] = df["revenue"].astype(float)
    return df

# %% [markdown]
# ### Exercise 3: Bound parameters
# Write `orders_in_range(conn, channel, start, end)` returning the **number of orders** on `channel` with `start <= order_date <= end` (ISO date strings). Use bound
# parameters `%(name)s` for all three values. The checker also sends a hostile channel string.
# *Hint: `cur.execute(sql, {"ch": channel, "s": start, "e": end})`; `order_date` is a DATE column in the fake, compare with `CAST(%(s)s AS DATE)`.*

# %% tags=["starter"]
def orders_in_range(conn, channel, start, end):
    ...

# %%
_o = orders.assign(order_date=pd.to_datetime(orders["order_date"]))
_exp = int(((_o["channel"] == "mobile") & (_o["order_date"] >= "2024-01-01") & (_o["order_date"] <= "2024-06-30")).sum())
assert orders_in_range(conn, "mobile", "2024-01-01", "2024-06-30") == _exp, f"expected {_exp}"
assert orders_in_range(conn, "web' OR '1'='1", "2024-01-01", "2024-06-30") == 0, "hostile input must not match anything: use bound parameters"
print(f"Exercise 3 passed ({_exp} mobile orders in H1 2024)")

# %% tags=["solution"]
def orders_in_range(conn, channel, start, end):
    sql = """SELECT COUNT(*) FROM orders
             WHERE channel = %(ch)s AND order_date >= CAST(%(s)s AS DATE) AND order_date <= CAST(%(e)s AS DATE)"""
    with conn.cursor() as cur:
        return int(cur.execute(sql, {"ch": channel, "s": start, "e": end}).fetchone()[0])

# %% [markdown]
# ### Exercise 4 (open-ended): Design the seam
# Think of a pipeline you own (or would write): "read yesterday's orders, compute KPIs, write a summary table". In a markdown cell list (a) which function(s) take the connection as an argument,
# (b) which behaviour you would test on the fake, and (c) which behaviour needs a real dev account.

# %% [markdown]
# ## Recap
# - Credentials come from the environment (or a secrets manager), are excluded from `repr`, and missing ones fail fast with names.
# - Connector = send SQL, fetch rows; use context managers, bound parameters, and aggregate in SQL. Snowpark = lazy DataFrame executed inside Snowflake.
# - Inject the connection into functions; test against a DuckDB-backed fake for portable SQL and keep a small real-account integration test.
# - Normalise real-service quirks (UPPER CASE columns, `Decimal`) in one place.
#
# **Go deeper:** docs.snowflake.com, "Python Connector" and "Snowpark Developer Guide for Python" sections (check the current install instructions there).
#
# **Next:** [Loading data: stages, COPY INTO and file formats](04_loading_data_stages_copy_into_and_formats.ipynb)
