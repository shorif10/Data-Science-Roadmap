# %% [markdown]
# # A Data Agent with SQL and Knowledge-Base Tools
#
# **Why this matters.** "Ask a question of our data in plain English" is the most requested agent in every company, and the most dangerous one to build carelessly: you are letting a model write SQL that runs against
# a real database. The interesting engineering is not the prompt. It is the **tool**: a SQL executor that can only read, only one statement at a time, only the tables you allow, never returns a million rows, and stops a
# runaway query. This notebook builds that tool in layers, combines it with a knowledge-base search tool, and tests the agent the way you would test any service: scripted scenarios that assert the **tool trajectory** and
# the **numbers in the answer** against an independent pandas computation.
#
# **You will learn**
# - build a read-only SQL tool with layered defences: a validator, a read-only connection, a SQLite authorizer, a row/size cap and a timeout
# - return SQL errors to the model in a form it can act on, and watch a deterministic agent recover from one
# - assert an agent's trajectory (which tools, in what order, with which errors) and its numeric answers against pandas
# - show that hostile SQL fails at **each** layer even when an earlier layer is switched off
#
# **Prerequisites:** [Build an agent loop](01_build_an_agent_loop_from_scratch.ipynb) and [Agent safety](../22_Agentic_AI/04_agent_safety_guardrails_and_evaluation.ipynb). Basic SQL ([module 06](../06_SQL/)).
#
# > Data used: `datasets/shop.db` (synthetic SQLite: customers, products, orders, order_items), opened **read-only**; the notebook hashes the file before and after to prove it is untouched. The "model" is a **deterministic
# > mock** that maps question types to SQL templates (and can be told to make one realistic mistake). That is enough to exercise every tool and failure path; it does **not** measure how well a real model writes SQL (text-to-SQL
# > accuracy on messy real schemas is genuinely hard and must be evaluated on your own questions).

# %%
import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
import pandas as pd

DATA = Path("../datasets")
DB_FILE = (DATA / "shop.db").resolve()
DB_URI = DB_FILE.as_uri() + "?mode=ro"
KB = {p.stem: p.read_text(encoding="utf-8") for p in sorted((DATA / "knowledge_base").glob("*.md"))}
HASH_BEFORE = hashlib.sha256(DB_FILE.read_bytes()).hexdigest()
print("db hash (first 12):", HASH_BEFORE[:12], "| KB documents:", len(KB))

# %% [markdown]
# ## 1. The threat: what "SELECT-only" has to mean
#
# A model (or text it read) can produce SQL you did not intend. List what must be impossible through this tool:
#
# | Attack or accident | Example |
# |---|---|
# | write or DDL | `DELETE FROM orders`, `DROP TABLE customers` |
# | a second statement | `SELECT 1; DELETE FROM orders` |
# | a write hidden behind a read keyword | `WITH x AS (SELECT 1) DELETE FROM orders` |
# | reading what you did not expose | `SELECT * FROM sqlite_master`, another database via `ATTACH` |
# | engine controls | `PRAGMA ...`, `load_extension` |
# | unbounded output | `SELECT * FROM order_items` (9,689 rows) |
# | unbounded work | a three-way cross join that never finishes |
#
# No single check covers all of them, so we stack independent layers: **(1)** a validator that rejects bad SQL with a clear message, **(2)** a **read-only connection**, **(3)** a SQLite **authorizer** (an allow-list the engine
# itself consults for every operation), **(4)** a **row and size cap**, **(5)** a **timeout** via a progress handler. If layer 1 has a bug, layers 2 and 3 still refuse writes; if a query is legal but huge, layers 4 and 5 bound it.

# %% [markdown]
# ## 2. Layer 1: the validator
#
# Key trick: first replace the contents of every `'string literal'` with an empty literal, so that words and semicolons inside strings (`WHERE first_name = 'DROP TABLE x;'`) are not mistaken for SQL. Then check what remains.
# The validator is deliberately *strict and simple*: it rejects comments and quoted identifiers entirely (rarely needed by an agent, and a classic way to hide things), requires a single statement starting with `SELECT` or
# `WITH`, rejects forbidden words, and only allows known tables after `FROM` / `JOIN`. It will occasionally reject legal SQL (a column called `replace`, say). For a safety layer, false rejections are the right side to err on.

# %%
class UnsafeSQL(Exception):
    pass


ALLOWED_TABLES = {"customers", "products", "orders", "order_items"}
FORBIDDEN = set("insert update delete drop alter create attach detach pragma vacuum reindex analyze replace begin commit rollback savepoint release load_extension".split())


def strip_literals(sql):
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_sql(sql):
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQL("empty query")
    if len(sql) > 1500:
        raise UnsafeSQL("query too long (max 1500 characters)")
    body = strip_literals(sql)
    if "'" in body.replace("''", ""):
        raise UnsafeSQL("unterminated string literal")
    if "--" in body or "/*" in body:
        raise UnsafeSQL("comments are not allowed")
    if any(ch in body for ch in ('"', "`", "[")):
        raise UnsafeSQL("quoted identifiers are not allowed; use plain table and column names")
    body = body.strip().rstrip(";").strip()
    if ";" in body:
        raise UnsafeSQL("only one statement is allowed")
    low = body.lower()
    words = re.findall(r"[a-z_][a-z0-9_]*", low)
    if not words or words[0] not in ("select", "with"):
        raise UnsafeSQL("only SELECT queries are allowed (read-only tool)")
    if bad := sorted(FORBIDDEN & set(words)):
        raise UnsafeSQL(f"forbidden keyword: {bad[0]}")
    if any(w.startswith("sqlite_") for w in words):
        raise UnsafeSQL("system tables are not accessible")
    ctes = set(re.findall(r"([a-z_][a-z0-9_]*)\s*(?:\([^)]*\))?\s*as\s*\(", low))
    for table in re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", low):
        if table not in ALLOWED_TABLES and table not in ctes:
            raise UnsafeSQL(f"table not allowed: {table}. Allowed tables: {sorted(ALLOWED_TABLES)}")
    return sql.strip().rstrip(";").strip()


GOOD = ["SELECT COUNT(*) FROM customers", "select country, count(*) from customers group by country;", "SELECT * FROM customers WHERE first_name = 'DROP TABLE x; --' LIMIT 3",
        "WITH d AS (SELECT * FROM orders WHERE status = 'delivered') SELECT COUNT(*) FROM d"]
BAD = ["DELETE FROM orders", "SELECT 1; DELETE FROM orders", "WITH x AS (SELECT 1) DELETE FROM orders", "SELECT * FROM sqlite_master", "SELECT * FROM orders -- hi",
       "PRAGMA table_info(orders)", "SELECT * FROM secrets", "ATTACH DATABASE 'x.db' AS x", "SELECT * FROM \"orders\"", ""]
for sql in GOOD:
    validate_sql(sql)
print(f"{len(GOOD)} legitimate queries accepted")
for sql in BAD:
    try:
        validate_sql(sql)
        print("!! ACCEPTED:", sql)
    except UnsafeSQL as exc:
        print(f"rejected {sql[:42]!r:<46} -> {exc}")

# %% [markdown]
# ## 3. Layers 2 to 5: read-only connection, authorizer, cap, timeout
#
# - **Read-only connection** (`mode=ro` in the URI): the database file cannot be modified through this connection, whatever SQL arrives.
# - **Authorizer**: SQLite calls your function for every operation it is about to perform (select, read a column, call a function, insert, pragma, attach...) and you answer *allow* or *deny*. We allow only `SELECT`, function
#   calls, recursive CTEs, and **reads of the four allowed tables**; everything else, including any write and any read of `sqlite_master`, is denied by the engine itself.
# - **Cap**: fetch at most `max_rows + 1` rows; if the extra row exists, return `max_rows` and `truncated: true`, so the model knows the result is partial rather than assuming it is complete.
# - **Timeout**: a progress handler is called every N virtual-machine steps; returning non-zero aborts the query.
#
# Errors become `ToolError`s with messages the model can act on; for an unknown column we append the real schema.

# %%
class ToolError(Exception):
    """Expected failure; the message goes back to the model verbatim."""


def authorizer(action, arg1, arg2, dbname, source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 33)):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 in ALLOWED_TABLES:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def schema_summary():
    con = sqlite3.connect(DB_URI, uri=True)
    try:
        return "; ".join(f"{t}({', '.join(r[1] for r in con.execute(f'pragma table_info({t})'))})" for t in sorted(ALLOWED_TABLES))
    finally:
        con.close()


def fetch_limited(cur, max_rows):
    if cur.description is None:                      # a statement that returns no rows (only reachable when a layer is switched off)
        return {"columns": [], "rows": [], "row_count": 0, "truncated": False}
    rows = cur.fetchmany(max_rows + 1)
    truncated = len(rows) > max_rows
    return {"columns": [d[0] for d in cur.description], "rows": [list(r) for r in rows[:max_rows]], "row_count": min(len(rows), max_rows), "truncated": truncated}


def execute_sql(sql, max_rows=50, timeout_s=2.0, validate=True, use_authorizer=True):
    """Run one query through the layers. The `validate` and `use_authorizer` switches exist only so we can test each layer alone."""
    try:
        cleaned = validate_sql(sql) if validate else sql
    except UnsafeSQL as exc:
        raise ToolError(f"Rejected: {exc}") from exc
    con = sqlite3.connect(DB_URI, uri=True)                       # layer 2: read-only
    if use_authorizer:
        con.set_authorizer(authorizer)                             # layer 3
    deadline = time.monotonic() + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)      # layer 5
    try:
        return fetch_limited(con.execute(cleaned), max_rows)       # layer 4
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        if "interrupted" in msg:
            raise ToolError(f"Query exceeded {timeout_s}s and was stopped. Add filters or aggregate.") from exc
        if "no such column" in msg or "no such table" in msg:
            raise ToolError(f"SQL error: {msg}. Schema: {schema_summary()}") from exc
        raise ToolError(f"SQL error: {msg}") from exc
    except (sqlite3.DatabaseError, sqlite3.ProgrammingError) as exc:
        raise ToolError(f"SQL error: {exc}") from exc
    finally:
        con.close()


r = execute_sql("select status, count(*) as n from orders group by status order by n desc")
print(r)
print(execute_sql("select order_id from order_items", max_rows=3))
try:
    execute_sql("select name from customers")
except ToolError as exc:
    print("ToolError:", str(exc)[:160], "...")

# %% [markdown]
# Now the layers one at a time. Each attack is run **with the validator switched off** so the read-only connection and authorizer must do the work alone, then with everything on. The cross join demonstrates the timeout: it is
# perfectly legal SQL (allowed tables, SELECT only) and would run for a very long time.

# %%
attacks = {
    "delete rows": "DELETE FROM orders WHERE status = 'cancelled'",
    "drop table": "DROP TABLE customers",
    "insert row": "INSERT INTO customers VALUES (999999, 'x', 'US', '2024-01-01', 30, 0)",
    "read sqlite_master": "SELECT name FROM sqlite_master",
    "attach database": "ATTACH DATABASE ':memory:' AS scratch",
    "pragma": "PRAGMA writable_schema = 1",
    "two statements": "SELECT 1; DELETE FROM orders",
}
rows = []
for name, sql in attacks.items():
    outcome = {}
    for label, kwargs in (("validator off, authorizer off", dict(validate=False, use_authorizer=False)), ("validator off, authorizer on", dict(validate=False)), ("all layers", {})):
        try:
            execute_sql(sql, **kwargs)
            outcome[label] = "EXECUTED"
        except ToolError as exc:
            outcome[label] = "blocked"
    rows.append({"attack": name, **outcome})
layer_table = pd.DataFrame(rows)
print(layer_table.to_string(index=False))
assert (layer_table["all layers"] == "blocked").all() and (layer_table["validator off, authorizer on"] == "blocked").all()

t0 = time.perf_counter()
try:
    execute_sql("select count(*) from order_items a, order_items b, order_items c", timeout_s=0.5)
except ToolError as exc:
    print(f"\ncross join stopped after {time.perf_counter() - t0:.2f}s:", exc)
print("db untouched:", hashlib.sha256(DB_FILE.read_bytes()).hexdigest() == HASH_BEFORE)

# %% [markdown]
# *What just happened?* With the validator **and** authorizer off, the read-only connection alone still stopped every write (SQLite refuses to write a database opened `mode=ro`). Some attacks that are *not* writes (`sqlite_master`, `ATTACH`, `PRAGMA`) are
# blocked by the authorizer even when the validator is off, which is why the layers are independent. The "executed" cells in the first column show what you would have had with no layers: reads of hidden tables and engine
# commands. The two-statement attack is refused by the Python `sqlite3` driver itself (it executes one statement per call), another free layer you should not rely on alone. The legal-but-expensive cross join was cut off by the timeout. No single layer is trusted.
#
# Two things the layers do **not** do: they do not stop the model from asking a *wrong* question, and they do not prevent it from reading data it is allowed to read but the **user** should not see. Per-user row-level authorisation must
# be enforced in code around the tool (for example by pre-filtering by the authenticated customer id), never left to the model's prompt.

# %% [markdown]
# ## 4. The tools and the agent
#
# Three tools: `describe_schema` (so the model does not guess column names), `run_sql` (the guarded executor above, returning compact JSON), and `search_kb` (policy lookup). The loop is the compact one from notebook 01: dispatcher
# that never raises, `tool_use` / `tool_result` messages, step limit.

# %%
STOP = set("a an the i my me to of is it in on for do you can how what when was be and or with after have this that are does will much many".split())
words = lambda t: {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP}


def describe_schema():
    return schema_summary()


def run_sql(sql: str):
    return json.dumps(execute_sql(sql))


def search_kb(query: str):
    best = max(((len(words(query) & words(line)), name, line) for name, text in KB.items() for line in text.splitlines()[1:]), key=lambda x: x[0])
    return json.dumps({"document": best[1], "text": best[2]} if best[0] else {"document": None, "text": "no relevant passage found"})


def spec(name, description, **props):
    return {"name": name, "description": description, "input_schema": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()},
                                                                      "required": list(props), "additionalProperties": False}}


REGISTRY = {
    "describe_schema": (spec("describe_schema", "List the tables and columns of the shop database. Call this before writing SQL if unsure of column names."), describe_schema),
    "run_sql": (spec("run_sql", "Run ONE read-only SELECT query on the shop database (tables: customers, products, orders, order_items). Returns at most 50 rows as JSON with a truncated flag. "
                                "Aggregate in SQL instead of fetching raw rows.", sql="string"), run_sql),
    "search_kb": (spec("search_kb", "Search the company policy knowledge base (returns, shipping, warranty, loyalty points, payment). Use for policy questions, never for data questions.", query="string"), search_kb),
}


def run_tool(registry, call):
    name, args = call["name"], call["input"]
    out = lambda text, err=False: {"type": "tool_result", "tool_use_id": call["id"], "content": text[:3000], **({"is_error": True} if err else {})}
    if name not in registry:
        return out(f"Unknown tool {name!r}. Available: {sorted(registry)}", True)
    schema, fn = registry[name]
    if problems := [e.message for e in jsonschema.Draft202012Validator(schema["input_schema"]).iter_errors(args)]:
        return out("Invalid input: " + "; ".join(problems), True)
    try:
        return out(fn(**args))
    except ToolError as exc:
        return out(str(exc), True)
    except Exception as exc:  # noqa: BLE001
        return out(f"{type(exc).__name__}: {exc}", True)


blocks_of = lambda m: [{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else m["content"]


@dataclass
class Result:
    text: str
    status: str
    messages: list
    calls: list = field(default_factory=list)          # (tool name, input, is_error)


def run_agent(llm, registry, question, max_steps=6):
    schemas, messages, calls = [s for s, _ in registry.values()], [{"role": "user", "content": question}], []
    for _ in range(max_steps):
        resp = llm.complete(messages, tools=schemas)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return Result(resp.text, "done", messages, calls)
        results = []
        for c in resp.tool_calls:
            r = run_tool(registry, c)
            calls.append((c["name"], c["input"], bool(r.get("is_error"))))
            results.append(r)
        messages.append({"role": "user", "content": results})
    return Result("", "max_steps", messages, calls)


# %% [markdown]
# ## 5. A scripted "model" for a data agent
#
# `DataAgentLLM` recognises a handful of question types and writes the SQL for each from a template; it asks for every tool it needs in one turn, then turns the **tool results** into a sentence using the column names. Two switches
# let us test failure paths: `sloppy=True` makes it use a wrong column name the first time (a very common real mistake), and a request to *delete* data makes it emit a `DELETE` (what a naive or manipulated model might do).
# Its reliability is **not** the subject; the tools, loop and tests around it are.

# %%
def est_tokens(x):
    x = x if isinstance(x, str) else json.dumps(x)
    return max(1, math.ceil(len(x) / 4))


@dataclass
class LLMResponse:
    content: list
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=dict)

    @property
    def text(self):
        return "".join(b["text"] for b in self.content if b["type"] == "text")

    @property
    def tool_calls(self):
        return [b for b in self.content if b["type"] == "tool_use"]


def respond(blocks, messages):
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return LLMResponse(blocks, stop, {"input_tokens": sum(est_tokens(m["content"]) for m in messages), "output_tokens": sum(est_tokens(b) for b in blocks)})


text_block = lambda t: {"type": "text", "text": t}
REVENUE_SQL = ("select c.country, round(sum(i.quantity * p.unit_price), 2) as revenue from orders o join customers c using(customer_id) "
               "join order_items i using(order_id) join products p using(product_id) where o.status = 'delivered' group by c.country order by revenue desc limit {n}")
AOV_SQL = ("select c.is_premium, round(avg(t.v), 2) as avg_order_value, count(*) as orders from (select o.order_id, o.customer_id, sum(i.quantity * p.unit_price) as v "
           "from orders o join order_items i using(order_id) join products p using(product_id) where o.status = 'delivered' group by o.order_id) t "
           "join customers c using(customer_id) group by c.is_premium order by c.is_premium")
CATEGORY_SQL = ("select p.category, sum(i.quantity) as units from orders o join order_items i using(order_id) join products p using(product_id) "
                "where o.status = 'delivered' group by p.category order by units desc limit 1")
NEWEST_SQL = "select {col}, signup_date from customers order by signup_date desc, customer_id desc limit 3"
KB_WORDS = r"return|refund|cancel|shipping|warranty|points|premium membership|payment|window|policy"


def plan(question, sloppy, errors_seen):
    """Return the list of (tool, input) to request first, from the question alone."""
    q, steps = question.lower(), []
    if re.search(r"\b(delete|drop|remove)\b", q) and "order" in q:
        return [("run_sql", {"sql": "DELETE FROM orders WHERE status = 'cancelled'"})]
    if re.search(r"which tables|what tables|schema|columns", q):
        steps.append(("describe_schema", {}))
    if m := re.search(r"how many (delivered|shipped|cancelled|returned) orders", q):
        steps.append(("run_sql", {"sql": f"select count(*) as n from orders where status = '{m.group(1)}'"}))
    if m := re.search(r"top (\d+) countries", q):
        steps.append(("run_sql", {"sql": REVENUE_SQL.format(n=int(m.group(1)))}))
    if "average order value" in q and "premium" in q:
        steps.append(("run_sql", {"sql": AOV_SQL}))
    if re.search(r"category.*(most|best)|(most|best).*category", q):
        steps.append(("run_sql", {"sql": CATEGORY_SQL}))
    if "newest customers" in q:
        steps.append(("run_sql", {"sql": NEWEST_SQL.format(col="name" if sloppy and not errors_seen else "first_name")}))
    if re.search(KB_WORDS, q) and (not steps or re.search(r"policy|window|warranty|how long", q)):
        steps.append(("search_kb", {"query": question}))
    return steps


def describe_result(call, content):
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content                                   # describe_schema returns plain text
    if "columns" not in data:
        return "Policy: " + data["text"] if "text" in data else content
    cols, rows = data["columns"], data["rows"]
    if cols == ["n"]:
        return f"There are {rows[0][0]} {re.search(r'(delivered|shipped|cancelled|returned)', call['input']['sql']).group(1)} orders."
    if cols == ["country", "revenue"]:
        return "Delivered revenue by country: " + ", ".join(f"{c} {v:,.2f}" for c, v in rows) + "."
    if cols[0] == "is_premium":
        return "Average delivered order value: " + "; ".join(f"{'premium' if p else 'non-premium'} {a:,.2f} ({n} orders)" for p, a, n in rows) + "."
    if cols == ["category", "units"]:
        return f"Best-selling category: {rows[0][0]} ({rows[0][1]} units)."
    if cols == ["first_name", "signup_date"]:
        return "Newest customers: " + ", ".join(f"{n} ({d})" for n, d in rows) + "."
    return f"{len(rows)} row(s): {rows[:3]}"


class DataAgentLLM:
    def __init__(self, sloppy=False):
        self.sloppy = sloppy

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        question, uses, outcomes = messages[0]["content"], {}, []
        for m in messages[1:]:
            for b in blocks_of(m):
                if b["type"] == "tool_use":
                    uses[b["id"]] = b
                elif b["type"] == "tool_result":
                    outcomes.append((uses[b["tool_use_id"]], b.get("is_error", False), b["content"]))
        errors = [(u, c) for u, e, c in outcomes if e]
        if not outcomes:
            steps = plan(question, self.sloppy, False)
            blocks = [{"type": "tool_use", "id": f"toolu_{k + 1}", "name": n, "input": a} for k, (n, a) in enumerate(steps)]
            return respond(([text_block("Let me query that.")] + blocks) if blocks else [text_block("I cannot answer that with my tools: I can query the shop database and search the policy knowledge base.")], messages)
        if errors and all(u["name"] == "run_sql" and "no such column: name" in c for u, c in errors) and len(outcomes) == len(errors) == 1:
            steps = plan(question, self.sloppy, True)                                       # recover: re-plan using the error text (real schema is in the message)
            return respond([text_block("The column is first_name; retrying.")] + [{"type": "tool_use", "id": "toolu_2", "name": n, "input": a} for n, a in steps], messages)
        if errors and any("Rejected" in c for _, c in errors):
            return respond([text_block("I can only read data; this tool is read-only, so I cannot modify or delete orders. I can count or list them if that helps.")], messages)
        return respond([text_block(" ".join(describe_result(u, c) for u, e, c in outcomes if not e) or "I could not get an answer.")], messages)


r = run_agent(DataAgentLLM(), REGISTRY, "How many delivered orders are there?")
print(r.status, "|", r.calls, "|", r.text)

# %% [markdown]
# ## 6. Scenarios: assert the trajectory and the numbers
#
# Each scenario is data: a question, the **exact tool trajectory** expected (names in order, plus whether each call errored), and a function that computes the expected answer **from pandas over the CSV files**, an independent path
# from the SQL the agent wrote. A scenario passes only if the status is `done`, the trajectory matches, and the expected text appears in the answer.

# %%
orders = pd.read_csv(DATA / "orders.csv")
customers = pd.read_csv(DATA / "customers.csv")
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
delivered = orders[orders.status == "delivered"].merge(customers, on="customer_id").merge(items, on="order_id").merge(products, on="product_id").assign(v=lambda d: d.quantity * d.unit_price)
assert len(orders) == execute_sql("select count(*) from orders")["rows"][0][0], "CSV and database must agree"

rev_country = delivered.groupby("country").v.sum().round(2).sort_values(ascending=False).head(3)
per_order = delivered.groupby(["order_id", "is_premium"]).v.sum().reset_index()
aov = per_order.groupby("is_premium").v.agg(["mean", "count"])
cat_units = delivered.groupby("category").quantity.sum().sort_values(ascending=False)
newest = customers.sort_values(["signup_date", "customer_id"], ascending=[False, False]).head(3)
return_window = re.search(r"Electronics have a (\d+) day", KB["returns_policy"]).group(1)


@dataclass
class Scenario:
    name: str
    question: str
    trajectory: list                    # [(tool, is_error), ...] in order
    expected: list                      # substrings that must appear in the answer
    sloppy: bool = False


SCENARIOS = [
    Scenario("count", "How many delivered orders are there?", [("run_sql", False)], [f"There are {(orders.status == 'delivered').sum()} delivered orders."]),
    Scenario("revenue by country", "What are the top 3 countries by delivered revenue?", [("run_sql", False)],
             ["Delivered revenue by country: " + ", ".join(f"{c} {v:,.2f}" for c, v in rev_country.items()) + "."]),
    Scenario("average order value", "What is the average order value for premium vs other customers?", [("run_sql", False)],
             [f"non-premium {aov.loc[False, 'mean']:,.2f} ({aov.loc[False, 'count']} orders)", f"premium {aov.loc[True, 'mean']:,.2f} ({aov.loc[True, 'count']} orders)"]),
    Scenario("best category", "Which product category sold the most units?", [("run_sql", False)], [f"Best-selling category: {cat_units.index[0]} ({cat_units.iloc[0]} units)."]),
    Scenario("schema", "Which tables and columns do you have?", [("describe_schema", False)], ["customers(customer_id, first_name, country"]),
    Scenario("policy", "What is the return window for electronics?", [("search_kb", False)], [f"{return_window} day return window"]),
    Scenario("data + policy", "How many returned orders are there, and what is the return window for electronics?", [("run_sql", False), ("search_kb", False)],
             [f"There are {(orders.status == 'returned').sum()} returned orders.", f"{return_window} day return window"]),
    Scenario("recovers from a SQL error", "Who are our newest customers?", [("run_sql", True), ("run_sql", False)],
             ["Newest customers: " + ", ".join(f"{n} ({d})" for n, d in zip(newest.first_name, newest.signup_date)) + "."], sloppy=True),
    Scenario("refuses to write", "Delete all cancelled orders.", [("run_sql", True)], ["read-only"]),
    Scenario("out of scope", "Why did sales drop last quarter?", [], ["cannot answer that"]),
]
rows = []
for sc in SCENARIOS:
    res = run_agent(DataAgentLLM(sloppy=sc.sloppy), REGISTRY, sc.question)
    traj = [(n, e) for n, _, e in res.calls]
    ok = res.status == "done" and traj == sc.trajectory and all(s in res.text for s in sc.expected)
    rows.append({"scenario": sc.name, "passed": ok, "tools": [n for n, _ in traj], "errors": sum(e for _, e in traj), "answer": res.text[:70]})
    assert ok, (sc.name, res.status, traj, res.text)
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# *What just happened?* Ten scenarios, all green, covering: single and combined tool use, a SQL error recovered in a second call (the first `run_sql` errored with the real schema in the message), a destructive request turned into a
# refusal by the **tool** (not by the model's good manners), and an out-of-scope question answered honestly with no tool call. The numbers come from pandas, not from the SQL under test. Note what this does **not** show: how a
# real model phrases SQL, handles ambiguous questions ("revenue" for delivered only? net of returns?), or joins tables wrongly. Those need an evaluation set of real questions with reference answers (module 21, notebook 05) and
# are where real data agents actually fail: *plausible SQL that answers a slightly different question*. Mitigations: show the SQL to the user alongside the answer, document metric definitions in the tool description, and keep a
# golden set of questions whose answers a human verified.
#
# Look at the exact trajectory of the recovery scenario, including the message the model received:

# %%
res = run_agent(DataAgentLLM(sloppy=True), REGISTRY, "Who are our newest customers?")
for m in res.messages[1:]:
    for b in blocks_of(m):
        if b["type"] == "tool_use":
            print(f"assistant TOOL_USE   {b['name']}: {b['input'].get('sql', '')[:70]}")
        elif b["type"] == "tool_result":
            print(f"user      TOOL_RESULT{' (ERROR)' if b.get('is_error') else ''}: {b['content'][:120]}")
        elif b["type"] == "text" and m["role"] == "assistant":
            print(f"assistant: {b['text'][:100]}")

# %% [markdown]
# ## 7. The real client (`no-run`)
#
# Only the `complete()` implementation changes. Give the real model the same three tools; note that the **quality of the tool descriptions** (which tables exist, how "revenue" is defined) now does much of the work your
# templates did. Needs `ANTHROPIC_API_KEY` and network access.

# %% tags=["no-run"]
import anthropic


class AnthropicLLM:
    def __init__(self, client=None, model="claude-sonnet-5-5"):
        self.client, self.model = client or anthropic.Anthropic(), model        # reads ANTHROPIC_API_KEY from the environment

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=messages, tools=tools or [])
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]          # keep every block (thinking blocks included)
        return LLMResponse(blocks, resp.stop_reason, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens})


SYSTEM = ("You answer questions about the shop. Use describe_schema if unsure of columns; use run_sql for data (read-only, one SELECT). "
          "'Revenue' means sum of quantity * unit_price over delivered orders. Always show the SQL you ran in your answer. Use search_kb for policy questions.")
real = run_agent(AnthropicLLM(), REGISTRY, "What are the top 3 countries by delivered revenue?")      # add system=SYSTEM inside complete() calls in your own loop
print(real.text)

# %% [markdown]
# ## Common mistakes
#
# 1. **Trusting a keyword check alone** (`sql.startswith("SELECT")`). `WITH ... DELETE` and `SELECT 1; DROP ...` get through. Stack independent layers, the strongest being the **read-only connection**.
# 2. **Connecting with the application's read-write credentials.** Give the agent a dedicated read-only database user or file handle, and only the tables it needs.
# 3. **No row cap, or a cap with no flag.** The model cannot tell "the top 50" from "all rows"; always return `truncated`.
# 4. **No timeout**, so one clever cross join ties up the database for everyone.
# 5. **Returning raw exceptions** ("OperationalError: near FROM") with no schema or hint. Put the fix in the error message, as here.
# 6. **Letting the model define business metrics.** "Revenue", "active user", "churned" have one company definition; put it in the tool description or a documented view.
# 7. **Checking only the final sentence.** Assert the trajectory too: a right answer from a wrong query (or three wasted retries) is a bug waiting for a different question.
# 8. **Forgetting per-user authorisation.** Read-only is not the same as "allowed to see everyone's data".

# %%
# Demonstration of mistake 1: the naive prefix check accepts attacks that the layered validator rejects
naive = lambda sql: sql.strip().lower().startswith(("select", "with"))
for sql in ["WITH x AS (SELECT 1) DELETE FROM orders", "SELECT 1; DROP TABLE customers", "SELECT * FROM sqlite_master"]:
    try:
        validate_sql(sql)
        strict = "accepted"
    except UnsafeSQL:
        strict = "rejected"
    print(f"{sql:<44} naive prefix check: {'accepted' if naive(sql) else 'rejected'} | validator: {strict}")

# %% [markdown]
# ## Interview corner
#
# **1. How do you let an LLM run SQL safely?**
# <details><summary>Answer</summary>
# Defence in depth: a dedicated read-only connection or user limited to the needed tables (the strongest layer), a validator for a single SELECT statement, an engine-level allow-list such as SQLite's authorizer or database
# permissions, a row and size cap with a truncation flag, a statement timeout, and logging of every query. Authorise per user in code, and assume the model's SQL is untrusted input. Show the SQL to the user.
# </details>
#
# **2. A query is valid SQL, read-only, and allowed, but still harmful. How?**
# <details><summary>Answer</summary>
# It can be expensive (cross joins, missing filters), it can return far too much data, or it can expose rows the requesting user should not see. So you need timeouts, caps, and row-level authorisation enforced outside the model.
# </details>
#
# **3. The agent's answer is numerically wrong although the SQL ran without error. What are the likely causes?**
# <details><summary>Answer</summary>
# Plausible SQL for a different question: wrong metric definition (gross vs net, which statuses count), a join that duplicates rows or drops unmatched ones, a missing filter, time-zone or date-range mistakes, or truncated results treated as complete.
# Fix with documented metric definitions, schema descriptions with examples, showing SQL, a golden question set with human-verified answers, and trajectory and number assertions in tests.
# </details>
#
# **4. How do you test a data agent?**
# <details><summary>Answer</summary>
# Unit-test the tool against attack and edge cases (writes, multiple statements, hidden tables, huge results, timeouts). Then run scenario tests with a scripted or recorded model that assert the tool trajectory and the answer's numbers against an
# independent computation; for a real model, run a golden question set repeatedly and report accuracy rates, step counts and cost.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: A result cap that also bounds size
# `fetch_limited` caps rows but a few very wide rows can still be huge. Write `fetch_capped(cur, max_rows, max_chars)` returning the same dict as `fetch_limited` (`columns`, `rows`, `row_count`, `truncated`) but which **also** drops rows
# from the end until `len(json.dumps(rows)) <= max_chars`, setting `truncated` if anything was dropped for either reason. `row_count` is the number of rows returned. Keep at least one row only if it fits.
# *Hint: fetch `max_rows + 1` rows first, as in `fetch_limited`.*

# %% tags=["starter"]
def fetch_capped(cur, max_rows, max_chars):
    ...


# %%
con = sqlite3.connect(DB_URI, uri=True)
full = con.execute("select order_id, status from orders order by order_id limit 10").fetchall()
a = fetch_capped(con.execute("select order_id, status from orders order by order_id limit 10"), 3, 10_000)
assert a["row_count"] == 3 and a["truncated"] is True and a["columns"] == ["order_id", "status"] and a["rows"] == [list(r) for r in full[:3]], a
b = fetch_capped(con.execute("select order_id, status from orders order by order_id limit 10"), 50, 10_000)
assert b["row_count"] == 10 and b["truncated"] is False, "nothing dropped, so not truncated"
c = fetch_capped(con.execute("select order_id, status from orders order by order_id limit 10"), 50, 60)
assert c["truncated"] is True and c["row_count"] == len(c["rows"]) < 10 and len(json.dumps(c["rows"])) <= 60, c
assert c["rows"] == [list(r) for r in full[:c["row_count"]]], "rows must be dropped from the END"
d = fetch_capped(con.execute("select order_id, status from orders order by order_id limit 10"), 50, 5)
assert d["rows"] == [] and d["row_count"] == 0 and d["truncated"] is True, "even one row does not fit"
con.close()
print("Exercise 1 passed")

# %% tags=["solution"]
def fetch_capped(cur, max_rows, max_chars):
    rows = cur.fetchmany(max_rows + 1)
    truncated = len(rows) > max_rows
    rows = [list(r) for r in rows[:max_rows]]
    while rows and len(json.dumps(rows)) > max_chars:
        rows.pop()
        truncated = True
    return {"columns": [d[0] for d in cur.description], "rows": rows, "row_count": len(rows), "truncated": truncated}


# %% [markdown]
# ### Exercise 2: Write the SQL, verify with pandas
# Assign to `premium_share_sql` a single SELECT returning **one row, one column named `share`**: the share (between 0 and 1, rounded to 4 decimals) of **delivered** orders that were placed by premium customers
# (`customers.is_premium = 1`). It must pass `validate_sql`. The check runs it through the real tool and compares with pandas.
# *Hint: `round(avg(c.is_premium), 4)` over delivered orders joined to customers (is_premium is stored as 0/1).*

# %% tags=["starter"]
premium_share_sql = "select 0 as share"

# %%
validate_sql(premium_share_sql)
out = execute_sql(premium_share_sql)
assert out["columns"] == ["share"] and out["row_count"] == 1, out
expected_share = round(float(orders[orders.status == "delivered"].merge(customers, on="customer_id").is_premium.mean()), 4)
assert abs(out["rows"][0][0] - expected_share) < 1e-9, f"expected {expected_share}, got {out['rows'][0][0]}"
assert 0 < expected_share < 1
print(f"Exercise 2 passed (share = {expected_share})")

# %% tags=["solution"]
premium_share_sql = ("select round(avg(c.is_premium), 4) as share from orders o join customers c using(customer_id) where o.status = 'delivered'")


# %% [markdown]
# ### Exercise 3: An authorizer
# Write `make_authorizer(allowed_tables)` returning a SQLite authorizer callback `(action, arg1, arg2, dbname, source) -> int`. Allow `SQLITE_SELECT`, `SQLITE_FUNCTION`, recursive CTEs (`SQLITE_RECURSIVE`, value 33), and `SQLITE_READ`
# **only** for tables in `allowed_tables`; deny everything else (`SQLITE_DENY`). The check installs it on a read-only connection and tries legitimate and hostile statements.
# *Hint: for `SQLITE_READ`, `arg1` is the table name and `arg2` the column.*

# %% tags=["starter"]
def make_authorizer(allowed_tables):
    def cb(action, arg1, arg2, dbname, source):
        return sqlite3.SQLITE_DENY
    return cb


# %%
def try_sql(sql, allowed):
    con = sqlite3.connect(DB_URI, uri=True)
    con.set_authorizer(make_authorizer(allowed))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


assert try_sql("select count(*) from customers", {"customers"})[0][0] == len(customers), "allowed reads must work"
assert try_sql("select upper(first_name) from customers limit 1", {"customers"}), "functions must be allowed"
assert try_sql("with recursive r(n) as (select 1 union all select n + 1 from r where n < 5) select sum(n) from r", set())[0][0] == 15, "recursive CTEs over no table must work"
for sql, allowed in [("select * from orders", {"customers"}), ("select name from sqlite_master", {"customers"}), ("pragma table_info(customers)", {"customers"}),
                     ("attach database ':memory:' as x", {"customers"}), ("insert into customers values (1, 'x', 'US', '2024-01-01', 1, 0)", {"customers"}),
                     ("delete from customers", {"customers"})]:
    try:
        try_sql(sql, allowed)
    except sqlite3.DatabaseError:
        pass
    else:
        raise AssertionError(f"should have been denied: {sql}")
print("Exercise 3 passed")

# %% tags=["solution"]
def make_authorizer(allowed_tables):
    def cb(action, arg1, arg2, dbname, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, 33):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1 in allowed_tables:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    return cb


# %% [markdown]
# ### Exercise 4 (open-ended): Add a scenario for your own question
# Think of a question a colleague would really ask of this data ("which channel has the highest return rate?"). Write the SQL by hand, run it through `execute_sql`, compute the answer independently in pandas, and **decide whether the two
# agree and why the definition matters** (what counts as a "return"?). Then add a `Scenario` for it. If you can only define the metric by making an assumption, write the assumption into the `run_sql` tool description. No assert for this one.

# %% [markdown]
# ## Cleanup
# The database was opened read-only throughout. Prove it is untouched:

# %%
assert hashlib.sha256(DB_FILE.read_bytes()).hexdigest() == HASH_BEFORE, "shop.db must be unchanged"
print("shop.db hash unchanged:", HASH_BEFORE[:12])

# %% [markdown]
# ## Recap
# - A SQL tool for an agent needs **layers**: validator (single SELECT/WITH, no comments or quoted identifiers, allowed tables), read-only connection, engine authorizer, row and size cap with a `truncated` flag, and a timeout.
# - Independent layers matter: with the validator off, the read-only connection and authorizer still stopped every write and the sensitive reads; the timeout stops legal-but-huge queries.
# - Put the fix in the error message (real schema on an unknown column) so the model can recover; put metric definitions in the tool description.
# - Test the agent with scripted scenarios: assert the **tool trajectory** and the **numbers** against an independent pandas computation, including recovery, refusal and out-of-scope cases.
# - Safe SQL is not correct SQL and not authorised SQL: evaluate answers on a golden set, show the SQL, and enforce per-user access in code.
#
# **Go deeper:** the SQLite documentation on `sqlite3_set_authorizer`, URI filenames (`mode=ro`) and `sqlite3_progress_handler` (sqlite.org); the tool-use pages at docs.anthropic.com; OWASP guidance on SQL injection (owasp.org).
#
# **Next:** [Memory, state and context management](03_memory_state_and_context_management.ipynb)
