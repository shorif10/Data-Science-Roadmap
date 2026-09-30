# %% [markdown]
# # APIs and Ingestion
#
# **Why this matters.** A lot of the data you need does not sit in a file: it sits behind someone else's HTTP API (payments, CRM, ads, weather).
# Pulling it reliably is harder than `requests.get`: the API pages its results, throttles you, fails at 3 a.m., and may send the same record twice.
# A good ingestion job survives all of that and can be re-run without creating duplicates. That last property, **idempotency**, is what separates a pipeline
# you trust from one you babysit.
#
# **You will learn**
# - build a small FastAPI service and call it *in process* (no network, fully runnable)
# - paginate with offsets vs cursors, and see why offset pagination can silently duplicate rows
# - retry transient errors with exponential backoff and jitter, honouring `429 Retry-After`
# - throttle yourself to a rate limit, and make writes idempotent (upserts, idempotency keys)
# - flatten nested JSON into relational tables
#
# **Prerequisites:** [File formats](02_file_formats_csv_json_parquet.ipynb) and [Data quality](04_data_quality_and_validation.ipynb).
#
# > Data used: the synthetic shop orders (`orders.csv`, `order_items.csv`, `customers.csv`), served by a fake API we build below. Nothing touches the internet.

# %%
import random
import sqlite3
import time
from collections import deque
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.testclient import TestClient

# Starlette's TestClient is an httpx-style client; recent versions build it on the `httpx2` package. Its exception classes live in whichever
# library the client is built on, so we look that up once instead of hard-coding it (the client API is identical either way).
try:
    import httpx2
    HTTP = httpx2 if issubclass(TestClient, httpx2.Client) else httpx
except ImportError:
    HTTP = httpx
print("TestClient is built on:", HTTP.__name__)

DATA = Path("../datasets")
orders = pd.read_csv(DATA / "orders.csv")
items = pd.read_csv(DATA / "order_items.csv")
customers = pd.read_csv(DATA / "customers.csv")

# %% [markdown]
# ## 1. A fake orders API
# We turn the shop tables into the kind of **nested JSON** real APIs return: each order embeds its customer and its lines. Then we serve it from a FastAPI app with
# three realistic features: an API-key header, cursor pagination, and a switch to inject failures (so we can practise recovering from them).

# %%
country = customers.set_index("customer_id")["country"].to_dict()
lines_by_order = items.groupby("order_id")[["product_id", "quantity"]].apply(lambda g: g.to_dict("records")).to_dict()
ALL_ORDERS = [
    {"order_id": int(r.order_id), "order_date": r.order_date, "status": r.status, "channel": r.channel,
     "customer": {"customer_id": int(r.customer_id), "country": country[r.customer_id]},
     "lines": [{"product_id": int(x["product_id"]), "quantity": int(x["quantity"])} for x in lines_by_order.get(r.order_id, [])]}
    for r in orders.sort_values("order_id").itertuples(index=False)
]
print(len(ALL_ORDERS), "orders; first:", ALL_ORDERS[0])

# %%
API_KEY = "demo-key-not-a-secret"      # a test value for this local demo; real keys come from env vars or a secret manager, never source code


def make_app(orders_data):
    app = FastAPI(title="Fake Orders API")
    app.state.orders = list(orders_data)
    app.state.chaos = deque()             # statuses to return on the next requests, e.g. deque([503, 503, 429])
    app.state.refunds = {}                # idempotency-key -> response
    app.state.refund_count = 0

    def auth(x_api_key: str | None = Header(default=None)):
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="bad or missing API key")

    def chaos():
        if app.state.chaos:
            code = app.state.chaos.popleft()
            raise HTTPException(status_code=code, detail="injected failure", headers={"Retry-After": "1"} if code == 429 else None)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/orders", dependencies=[Depends(auth), Depends(chaos)])
    def list_orders(limit: int = Query(100, ge=1, le=500), cursor: int = 0):
        """Cursor pagination: 'give me orders with id greater than cursor'."""
        rows = [o for o in app.state.orders if o["order_id"] > cursor][: limit + 1]      # fetch one extra to know if more exist
        page = rows[:limit]
        return {"items": page, "next_cursor": page[-1]["order_id"] if len(rows) > limit else None}

    @app.get("/v1/orders_by_offset", dependencies=[Depends(auth)])
    def list_orders_offset(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        """Offset pagination over 'newest first'."""
        newest_first = sorted(app.state.orders, key=lambda o: -o["order_id"])
        return {"items": newest_first[offset: offset + limit]}

    @app.post("/v1/refunds", dependencies=[Depends(auth)])
    def create_refund(order_id: int, idempotency_key: str = Header()):
        if idempotency_key in app.state.refunds:                    # replay: return the ORIGINAL result, do nothing new
            return app.state.refunds[idempotency_key]
        app.state.refund_count += 1
        result = {"refund_id": app.state.refund_count, "order_id": order_id}
        app.state.refunds[idempotency_key] = result
        return result

    return app


app = make_app(ALL_ORDERS)
client = TestClient(app, headers={"X-API-Key": API_KEY})
print(client.get("/health").json(), "|", TestClient(app).get("/v1/orders").status_code, "without a key")
r = client.get("/v1/orders", params={"limit": 2})
print(r.status_code, r.json()["next_cursor"], [o["order_id"] for o in r.json()["items"]])

# %% [markdown]
# *What just happened?* `TestClient` (built on `httpx`) sends requests straight into the ASGI app in the same process. There is no server and no socket, but the request goes through routing,
# validation, dependencies and serialisation, so this is also how you would unit-test a real client against a fake. The 401 shows that auth is enforced.

# %% [markdown]
# ## 2. Pagination: offset vs cursor
# APIs never return a million rows at once. Two designs:
#
# * **Offset** (`?offset=200&limit=100`): simple, but positions shift when rows are inserted or deleted while you crawl, so you can **skip or repeat** records.
# * **Cursor / keyset** (`?cursor=<last id seen>`): "give me what comes after this key". Stable under inserts, and fast on the server (an index seek instead of counting past N rows).
#
# First the well-behaved crawl with a cursor:

# %%
def crawl_cursor(client, limit=250, between_pages=None):
    rows, cursor, pages = [], 0, 0
    while cursor is not None:
        body = client.get("/v1/orders", params={"limit": limit, "cursor": cursor}).raise_for_status().json()
        rows += body["items"]
        cursor, pages = body["next_cursor"], pages + 1
        if between_pages:
            between_pages(pages)
    return rows, pages


rows, pages = crawl_cursor(client)
print(f"{len(rows)} orders in {pages} pages; unique ids: {len({o['order_id'] for o in rows})}")

# %% [markdown]
# Now the failure mode. Someone places 5 new orders **while** we are crawling an offset-paginated, newest-first endpoint:

# %%
def crawl_offset(client, limit=250, between_pages=None):
    rows, offset, pages = [], 0, 0
    while True:
        items_ = client.get("/v1/orders_by_offset", params={"limit": limit, "offset": offset}).raise_for_status().json()["items"]
        if not items_:
            break
        rows += items_
        offset += limit
        pages += 1
        if between_pages:
            between_pages(pages)
    return rows


live = make_app(ALL_ORDERS)
live_client = TestClient(live, headers={"X-API-Key": API_KEY})
next_id = max(o["order_id"] for o in ALL_ORDERS) + 1


def insert_five(page_no):
    global next_id
    if page_no == 1:                                  # after page 1 has been read
        for _ in range(5):
            live.state.orders.append({"order_id": next_id, "order_date": "2025-06-01", "status": "shipped", "channel": "web",
                                      "customer": {"customer_id": 1, "country": "CA"}, "lines": []})
            next_id += 1


got = crawl_offset(live_client, between_pages=insert_five)
ids = [o["order_id"] for o in got]
print(f"rows fetched: {len(ids)}, unique: {len(set(ids))}, duplicates: {len(ids) - len(set(ids))}")

# %% [markdown]
# Five duplicates appeared: the new rows pushed old ones down, so page 2 repeated the last five rows of page 1. (With deletions you would *skip* rows instead, which is worse because
# nothing looks wrong.) The fix on our side is idempotent loading by key (section 6); the fix on the API side is cursors. Prefer cursor pagination when you have a choice.

# %% [markdown]
# ## 3. Retries with exponential backoff and jitter
# Networks and servers fail transiently. Retrying is right for **timeouts, `429 Too Many Requests` and 5xx**; it is wrong for most 4xx errors (a 401 or 422 will fail forever).
#
# How long to wait? A fixed delay makes 1,000 clients that failed together retry together (a "thundering herd"). **Exponential backoff** doubles the wait each attempt; **jitter**
# randomises it so clients spread out. The well-known "full jitter" form: wait a random time between 0 and `min(cap, base * 2**attempt)`.

# %%
def backoff_delay(attempt, base=0.5, cap=8.0, rng=random):
    """Full jitter: uniform(0, min(cap, base * 2**attempt))."""
    return rng.uniform(0, min(cap, base * 2 ** attempt))


rng = random.Random(42)
for attempt in range(6):
    print(f"attempt {attempt}: upper bound {min(8.0, 0.5 * 2 ** attempt):4.1f}s, sampled delay {backoff_delay(attempt, rng=rng):5.2f}s")

# %%
RETRYABLE = {429, 500, 502, 503, 504}


def get_with_retry(client, url, *, params=None, max_attempts=5, sleep=time.sleep, rng=random, waits=None):
    """GET with retries. `sleep` is injectable so tests (and this notebook) do not actually wait."""
    for attempt in range(max_attempts):
        try:
            resp = client.get(url, params=params)
        except HTTP.TransportError:
            resp = None                                              # network-level failure: retryable
        if resp is not None and resp.status_code not in RETRYABLE:
            return resp.raise_for_status()                           # success, or a non-retryable 4xx -> raises immediately
        if attempt == max_attempts - 1:
            raise RuntimeError(f"gave up after {max_attempts} attempts (last status {None if resp is None else resp.status_code})")
        delay = backoff_delay(attempt, rng=rng)
        if resp is not None and "Retry-After" in resp.headers:
            delay = max(delay, float(resp.headers["Retry-After"]))    # the server told us how long: obey it
        if waits is not None:
            waits.append(round(delay, 2))
        sleep(delay)


waits = []
app.state.chaos.extend([503, 429, 503])                              # the next 3 requests fail
resp = get_with_retry(client, "/v1/orders", params={"limit": 3}, sleep=lambda s: None, rng=random.Random(42), waits=waits)
print("succeeded with", len(resp.json()["items"]), "items after waits (s):", waits)

# %% [markdown]
# *What just happened?* Three injected failures, three waits, then success. The `429` wait was at least the server's `Retry-After: 1`. Because `sleep` is a parameter,
# the demo ran instantly; this is a **testing seam** you should copy in your own code (also for clocks and randomness).
#
# Two more properties every retry loop needs: a **cap on attempts** (otherwise a permanent outage means an infinite loop) and **non-retryable errors that fail fast**:

# %%
try:
    get_with_retry(TestClient(app), "/v1/orders", sleep=lambda s: None)          # no API key -> 401
except HTTP.HTTPStatusError as e:
    print("401 is not retried:", e.response.status_code)

app.state.chaos.extend([503] * 10)
try:
    get_with_retry(client, "/v1/orders", sleep=lambda s: None, max_attempts=3)
except RuntimeError as e:
    print("permanent outage:", e)
app.state.chaos.clear()

# %% [markdown]
# **Retries need idempotent operations.** Retrying a `GET` is harmless. Retrying a `POST /refunds` after a timeout might refund twice if the first request actually succeeded.
# The standard fix is an **idempotency key**: the client generates one unique key per logical operation and sends it on every attempt; the server remembers the result per key and replays it.

# %%
key = "refund-order-123-attempt-group-1"
first = client.post("/v1/refunds", params={"order_id": 123}, headers={"Idempotency-Key": key}).json()
retry = client.post("/v1/refunds", params={"order_id": 123}, headers={"Idempotency-Key": key}).json()          # e.g. after a timeout
other = client.post("/v1/refunds", params={"order_id": 124}, headers={"Idempotency-Key": "refund-order-124"}).json()
print(first, retry, other, "| refunds actually created:", app.state.refund_count)

# %% [markdown]
# ## 4. Respecting rate limits
# APIs publish limits ("10 requests per second"). Getting `429`s and backing off works, but it is politer (and faster overall) to **throttle yourself** below the limit.
# A simple sliding-window throttle, with an injectable clock so we can test it without sleeping:

# %%
class Throttle:
    """Allow at most `max_calls` calls per `period` seconds."""
    def __init__(self, max_calls, period, clock=time.monotonic, sleep=time.sleep):
        self.max_calls, self.period, self.clock, self.sleep = max_calls, period, clock, sleep
        self.calls = deque()

    def wait(self):
        now = self.clock()
        while self.calls and now - self.calls[0] >= self.period:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            self.sleep(self.period - (now - self.calls[0]))            # wait until the oldest call leaves the window
            return self.wait()
        self.calls.append(self.clock())


fake = {"t": 0.0}
slept = []
def fake_sleep(s):
    slept.append(round(s, 2)); fake["t"] += s

th = Throttle(max_calls=3, period=1.0, clock=lambda: fake["t"], sleep=fake_sleep)
for i in range(7):
    th.wait()
print("7 calls at 3 per second needed these waits (seconds):", slept, "| simulated total:", round(fake["t"], 2))

# %% [markdown]
# ## 5. From nested JSON to tables
# The API returns one object per order with a nested customer and a *list* of lines. Relational tables want flat rows: `orders` (one row per order) and `order_lines` (one row per line, with a foreign key).
# `pd.json_normalize` flattens nested objects; a list needs to be exploded, or handled with the `record_path` argument:

# %%
rows, _ = crawl_cursor(client)
orders_flat = pd.json_normalize(rows, sep="_").drop(columns="lines")
lines_flat = pd.json_normalize(rows, record_path="lines", meta=["order_id"])
print(orders_flat.columns.tolist())
print(lines_flat.head(3).to_string(index=False))
print("lines match the source table:", len(lines_flat) == len(items), "| orders match:", len(orders_flat) == len(orders))

# %% [markdown]
# Notice a design choice: an order with **no** lines contributes nothing to `lines_flat` (and we keep it in `orders_flat` because we built that table separately). Flattening a list column with an inner join
# style operation would silently drop such orders; always reconcile row counts, as above.

# %% [markdown]
# ## 6. Idempotent ingestion
# Now the most important property. **Idempotent** = running the load twice gives the same result as running it once. It matters because jobs *will* be re-run: after a failure, for a backfill,
# or because a scheduler fired twice. Appending blindly would double the data. The standard tool is an **upsert** keyed by the business key: insert new rows, update existing ones.

# %%
def load_orders(con, orders_df):
    con.execute("""CREATE TABLE IF NOT EXISTS raw_orders (order_id INTEGER PRIMARY KEY, order_date TEXT, status TEXT, channel TEXT,
                   customer_id INTEGER, customer_country TEXT, ingested_at TEXT)""")
    rows_ = [(int(r.order_id), r.order_date, r.status, r.channel, int(r.customer_customer_id), r.customer_country, "2025-06-01T00:00:00")
             for r in orders_df.itertuples(index=False)]
    con.executemany("""INSERT INTO raw_orders VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(order_id) DO UPDATE SET status = excluded.status, ingested_at = excluded.ingested_at""", rows_)
    con.commit()


db = sqlite3.connect(":memory:")
load_orders(db, orders_flat)
n1 = db.execute("SELECT count(*) FROM raw_orders").fetchone()[0]
load_orders(db, orders_flat)                                     # run it again: a retry, a re-run, a double-fired schedule
n2 = db.execute("SELECT count(*) FROM raw_orders").fetchone()[0]
print("after first load:", n1, "| after re-running:", n2)

# %% [markdown]
# A re-run that also brings **updated** values (an order flips from `shipped` to `delivered` upstream) updates in place:

# %%
changed = orders_flat.copy()
target = int(changed.loc[changed["status"] == "shipped", "order_id"].iloc[0])
changed.loc[changed["order_id"] == target, "status"] = "delivered"
load_orders(db, changed)
print(db.execute("SELECT order_id, status FROM raw_orders WHERE order_id = ?", (target,)).fetchone(),
      "| rows:", db.execute("SELECT count(*) FROM raw_orders").fetchone()[0])

# %% [markdown]
# *What just happened?* `ON CONFLICT ... DO UPDATE` (SQLite's upsert; `MERGE` in warehouses) makes the load safe to repeat. Contrast the broken version: `INSERT` without a key would leave 2x or 3x rows,
# and no error would tell you. The same principle applies to files: write to a path derived from the partition (`date=2025-06-01/`) and **overwrite** rather than append.

# %% [markdown]
# ## Common mistakes
#
# 1. **Offset pagination on a table that changes.** Duplicates or gaps with no error. Prefer cursors; dedupe by key on load.
# 2. **Retrying everything.** A 401/403/422 will never succeed; retrying just wastes time and can trigger lockouts. Retry only transient errors.
# 3. **No jitter, no cap.** All clients retry in lockstep and hammer a recovering server; unbounded retries turn an outage into a hung job.
# 4. **Retrying non-idempotent writes** without idempotency keys: double charges, double refunds.
# 5. **Secrets in code or notebooks.** Read keys from environment variables or a secret manager. The key above is a placeholder for a local fake.
# 6. **`response.json()` without `raise_for_status()`.** An error page then gets parsed as data. Fail loudly on non-2xx.
# 7. **No row-count reconciliation** after flattening nested JSON.

# %% [markdown]
# ## Interview corner
#
# **1. Why use cursor pagination instead of offset?**
# <details><summary>Answer</summary>
# Offsets are positions, which shift when data is inserted or deleted while paging, causing duplicates or skipped rows; large offsets are also slow because the server counts past them.
# A cursor encodes "everything after this key", which is stable and index-friendly.
# </details>
#
# **2. Explain exponential backoff with jitter.**
# <details><summary>Answer</summary>
# Wait longer after each failure (delay grows exponentially up to a cap) and randomise the delay so many clients do not retry at the same moments. Honour <code>Retry-After</code> when given, and limit total attempts.
# </details>
#
# **3. What does idempotent mean for an ingestion job and how do you achieve it?**
# <details><summary>Answer</summary>
# Re-running with the same input leaves the destination in the same state. Use upserts/MERGE on a business key, or overwrite a deterministic partition, never blind appends; add an idempotency key when calling
# non-idempotent APIs.
# </details>
#
# **4. The API returns 429. What do you do?**
# <details><summary>Answer</summary>
# Stop and wait for the time in <code>Retry-After</code> (or backoff), then retry; better, throttle your own request rate below the documented limit, use bulk or incremental endpoints, and cache what does not change.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Crawl with a different page size
# Use `crawl_cursor(client, limit=250)` semantics but with a page size of **400**. Store the rows in `rows_400`, the number of pages in `pages_400`.

# %% tags=["starter"]
rows_400, pages_400 = ..., ...

# %%
import math
assert len(rows_400) == len(orders), "you should get every order exactly once"
assert len({o["order_id"] for o in rows_400}) == len(orders), "duplicates found"
assert pages_400 == math.ceil(len(orders) / 400), f"expected {math.ceil(len(orders) / 400)} pages"
print(f"Exercise 1 passed: {pages_400} pages")

# %% tags=["solution"]
rows_400, pages_400 = crawl_cursor(client, limit=400)

# %% [markdown]
# ### Exercise 2: Backoff schedule with a cap
# Compute `delays`: a list of 8 delays for attempts `0..7` using `backoff_delay(attempt, base=1.0, cap=10.0, rng=random.Random(7))`. Then set `hit_cap` to `True` if the **upper bound** for at least one of those attempts equals the cap
# (compute the bounds yourself: `min(cap, base * 2**attempt)`).

# %% tags=["starter"]
delays = ...
hit_cap = ...

# %%
bounds = [min(10.0, 1.0 * 2 ** a) for a in range(8)]
assert len(delays) == 8, "one delay per attempt 0..7"
assert all(0 <= d <= b for d, b in zip(delays, bounds)), "each delay must lie in [0, upper bound for that attempt]"
assert hit_cap == any(b == 10.0 for b in bounds)
ref = random.Random(7)
assert delays == [ref.uniform(0, b) for b in bounds], "use one Random(7) for all eight calls, in attempt order"
print("Exercise 2 passed:", [round(d, 2) for d in delays])

# %% tags=["solution"]
r7 = random.Random(7)
delays = [backoff_delay(a, base=1.0, cap=10.0, rng=r7) for a in range(8)]
hit_cap = any(min(10.0, 1.0 * 2 ** a) == 10.0 for a in range(8))

# %% [markdown]
# ### Exercise 3: Make a load idempotent
# The function below **appends** blindly. Fix it as `load_events_idempotent(con, events)` using an upsert on `event_id`, so that loading the same batch twice leaves the row count unchanged **and** a changed `amount` for an existing `event_id` is updated.
# Assume the table `events(event_id INTEGER PRIMARY KEY, amount REAL)` already exists.

# %% tags=["starter"]
def load_events_idempotent(con, events):
    """events: list of (event_id, amount) tuples."""
    ...

# %%
c = sqlite3.connect(":memory:")
c.execute("CREATE TABLE events (event_id INTEGER PRIMARY KEY, amount REAL)")
batch = [(1, 10.0), (2, 20.0), (3, 30.0)]
load_events_idempotent(c, batch)
load_events_idempotent(c, batch)
assert c.execute("SELECT count(*) FROM events").fetchone()[0] == 3, "loading the same batch twice must not create duplicates"
load_events_idempotent(c, [(2, 25.0), (4, 40.0)])
got = dict(c.execute("SELECT event_id, amount FROM events").fetchall())
assert got == {1: 10.0, 2: 25.0, 3: 30.0, 4: 40.0}, f"unexpected table contents: {got}"
print("Exercise 3 passed")

# %% tags=["solution"]
def load_events_idempotent(con, events):
    con.executemany("INSERT INTO events VALUES (?, ?) ON CONFLICT(event_id) DO UPDATE SET amount = excluded.amount", events)
    con.commit()

# %% [markdown]
# ### Exercise 4: Units ordered per country from the raw payload
# From `rows` (the nested API payload), build `units_by_country`: a Series of **total units ordered per customer country** (sum of line quantities), sorted descending. Use `json_normalize` (or a loop), not the CSVs.

# %% tags=["starter"]
units_by_country = ...

# %%
exp = (items.merge(orders, on="order_id").merge(customers, on="customer_id").groupby("country")["quantity"].sum().sort_values(ascending=False))
assert units_by_country.sum() == exp.sum(), "totals differ; did you include every line?"
assert units_by_country.to_dict() == exp.to_dict(), "per-country totals differ"
assert list(units_by_country.index) == list(exp.index), "sort descending by units"
print("Exercise 4 passed:", units_by_country.head(3).to_dict())

# %% tags=["solution"]
ln = pd.json_normalize(rows, record_path="lines", meta=["order_id", ["customer", "country"]])
units_by_country = ln.groupby("customer.country")["quantity"].sum().sort_values(ascending=False)
units_by_country.index.name = "country"

# %% [markdown]
# ### Exercise 5 (open-ended): Design the failure story
# You must ingest 2 million records per day from a partner API with a limit of 5 requests per second and occasional 503s. In a markdown cell, sketch: page size and the number of requests,
# how long a full crawl takes at the limit, what you store to resume after a crash (cursor? high-water mark?), and how you prevent duplicates. Use `Throttle` or plain arithmetic to back your numbers.

# %% [markdown]
# ## Recap
# - Serve and test an API **in process** with FastAPI + `TestClient`; inject failures to test your client.
# - **Cursor** pagination is stable; **offset** pagination can silently duplicate or skip rows while data changes.
# - Retry only transient failures (timeouts, 429, 5xx) with capped **exponential backoff + jitter**, honour `Retry-After`, cap attempts, and make sleep/clock injectable for tests.
# - Throttle your own calls below the documented rate limit; use **idempotency keys** for non-idempotent writes.
# - **Idempotent loads** (upsert on a business key, or overwrite a deterministic partition) make re-runs safe. Reconcile row counts after flattening nested JSON.
#
# **Go deeper:** the FastAPI docs (fastapi.tiangolo.com) on testing, the `httpx` docs (python-httpx.org), and the AWS Architecture Blog article "Exponential Backoff And Jitter" for the full-jitter analysis.
#
# **Next:** [DuckDB for analytics](06_duckdb_analytics.ipynb)
