# %% [markdown]
# # Testing, Observability and Deploying Agents
#
# **Why this matters.** An agent that works in a notebook is a prototype. To run it for customers you need to know three things you cannot get from a demo: *does it still work after I change a prompt or a tool* (testing),
# *what did it actually do and what did it cost when something went wrong at 2 a.m.* (observability), and *how do I put it behind an endpoint that is authenticated, bounded and honest about failure* (deployment).
# Agents are awkward to test: the model is non-deterministic, tools have side effects, and the number of paths is open-ended. The answer is a **pyramid**: many cheap deterministic tests at the bottom, fewer recorded-trajectory tests in the middle,
# and a small number of real-model evaluations at the top, plus tracing and monitoring in production.
#
# **You will learn**
# - unit-test tools with pytest (including a mutation check that proves the tests can fail), and run pytest from a notebook safely
# - record a model's responses once and **replay** them in tests, so trajectories are checked without an API key, cost or randomness
# - emit JSON tracing spans for every run, model call and tool call; read them as a tree; compute cost and latency percentiles from them
# - serve an agent with FastAPI and test the endpoint with `TestClient`: auth that fails closed, input validation, rate limiting, honest status codes
# - use a deployment checklist
#
# **Prerequisites:** [Build an agent loop](01_build_an_agent_loop_from_scratch.ipynb) and [a data agent](02_a_data_agent_with_sql_and_knowledge_base_tools.ipynb); [agent safety and evaluation](../22_Agentic_AI/04_agent_safety_guardrails_and_evaluation.ipynb).
#
# > Data used: `datasets/shop.db` (read-only) and the knowledge base. The model is a **deterministic mock**. Latencies in the observability section come from a **simulated clock** (a model of latency, seeded and reproducible, not a measurement);
# > token counts are estimates; prices are list prices you must re-check. Real recording and deployment commands are in `no-run` cells. All temporary files live in a temp folder that the last cell deletes.

# %%
import hashlib
import io
import json
import math
import os
import random
import re
import secrets
import sqlite3
import sys
import tempfile
import textwrap
import uuid
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

DATA = Path("../datasets").resolve()
DB_URI = (DATA / "shop.db").resolve().as_uri() + "?mode=ro"
TMP = tempfile.TemporaryDirectory(prefix="agent_nb4_")
TMP_DIR = Path(TMP.name)
print("temp folder:", TMP_DIR.name, "(deleted in the last cell)")

# %% [markdown]
# ## 1. The test pyramid for agents
#
# | Level | What it checks | Uses a model? | Runs | Catches |
# |---|---|---|---|---|
# | **Tool unit tests** | each tool, including bad input, against an independent answer | no | every commit, milliseconds | logic bugs, unsafe inputs |
# | **Replay tests** | the loop + prompts + tool wiring produce the same trajectory | no (recorded) | every commit | accidental changes to prompts, schemas, tool outputs |
# | **Scenario evals** | task success, steps, cost on a golden set | yes, real, sampled repeatedly | before release, nightly | real quality regressions |
# | **Production monitoring** | errors, latency, cost, drift on real traffic | yes | continuously | what your tests did not imagine |
#
# Only the top two rows can tell you whether a *model* is good. The bottom rows make everything **around** the model trustworthy, which is where most agent bugs live. Start from the bottom.

# %% [markdown]
# ## 2. Tools live in a module; tests import the module
#
# The tool code is written to a file in the temp folder and imported like any other module, so the notebook and the pytest file exercise **the same code**. (In your own project this is just `shop_tools.py` in your repository.)
# Tools raise `ToolError` for expected problems and validate their inputs themselves: a tool is also called from places that never passed through a schema.

# %%
SHOP_TOOLS_SRC = '''
import json
import re
import sqlite3
from pathlib import Path

DB_URI = None        # set by the application / test setup
DATA_DIR = None


class ToolError(Exception):
    """An expected failure whose message is safe and useful to show the model."""


def _query(sql, params=()):
    con = sqlite3.connect(DB_URI, uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def get_order(order_id: int) -> dict:
    if not isinstance(order_id, int) or isinstance(order_id, bool) or order_id <= 0:
        raise ToolError("order_id must be a positive integer")
    rows = _query("select o.order_id, o.status, o.order_date, sum(i.quantity), round(sum(i.quantity * p.unit_price), 2) from orders o "
                  "join order_items i using(order_id) join products p using(product_id) where o.order_id = ? group by o.order_id", (order_id,))
    if not rows:
        raise ToolError(f"no order with id {order_id}")
    return dict(zip(("order_id", "status", "date", "items", "total"), rows[0]))


def count_orders(status: str) -> dict:
    if status not in ("delivered", "shipped", "cancelled", "returned"):
        raise ToolError("status must be one of delivered, shipped, cancelled, returned")
    return {"status": status, "count": _query("select count(*) from orders where status = ?", (status,))[0][0]}


_STOP = set("a an the i my me to of is it in on for do you can how what when was be and or with after have this that are does will much many".split())
_words = lambda t: {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP}


def search_kb(query_text: str) -> dict:
    kb = {p.stem: p.read_text(encoding="utf-8") for p in sorted((DATA_DIR / "knowledge_base").glob("*.md"))}
    best = max(((len(_words(query_text) & _words(line)), name, line) for name, text in kb.items() for line in text.splitlines()[1:]), key=lambda x: x[0])
    return {"document": best[1], "text": best[2]} if best[0] else {"document": None, "text": "no relevant passage found"}
'''
(TMP_DIR / "shop_tools.py").write_text(SHOP_TOOLS_SRC, encoding="utf-8")
sys.path.insert(0, str(TMP_DIR))
import shop_tools

shop_tools.DB_URI, shop_tools.DATA_DIR = DB_URI, DATA
print(shop_tools.get_order(1000), shop_tools.count_orders("returned"))

# %% [markdown]
# Now the test file. Each test compares a tool with an **independent** computation (pandas over the CSV files) or checks that bad input fails the right way. `parametrize` runs one test body per status.

# %%
TEST_SRC = '''
import pandas as pd
import pytest

import shop_tools
from shop_tools import ToolError


@pytest.fixture(scope="module")
def csv():
    d = shop_tools.DATA_DIR
    return {n: pd.read_csv(d / f"{n}.csv") for n in ("orders", "order_items", "products")}


@pytest.mark.parametrize("status", ["delivered", "shipped", "cancelled", "returned"])
def test_count_orders_matches_csv(csv, status):
    assert shop_tools.count_orders(status)["count"] == int((csv["orders"].status == status).sum())


def test_counts_add_up_to_all_orders(csv):
    total = sum(shop_tools.count_orders(s)["count"] for s in ("delivered", "shipped", "cancelled", "returned"))
    assert total == len(csv["orders"])


def test_get_order_total_matches_csv(csv):
    items = csv["order_items"].merge(csv["products"], on="product_id")
    expected = round(float((items[items.order_id == 1000].quantity * items[items.order_id == 1000].unit_price).sum()), 2)
    assert shop_tools.get_order(1000)["total"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("bad", [0, -5, True, "1000", 1.5])
def test_get_order_rejects_bad_ids(bad):
    with pytest.raises(ToolError):
        shop_tools.get_order(bad)


def test_unknown_order_is_a_clear_error():
    with pytest.raises(ToolError, match="no order with id 99999999"):
        shop_tools.get_order(99999999)


def test_count_orders_rejects_unknown_status():
    with pytest.raises(ToolError):
        shop_tools.count_orders("lost'; drop table orders; --")


def test_search_kb_finds_the_return_window():
    assert "15 day" in shop_tools.search_kb("return window electronics")["text"]
'''
(TMP_DIR / "test_shop_tools.py").write_text(TEST_SRC, encoding="utf-8")


def run_pytest():
    """Run the test file in-process, quietly; return (exit code, last summary line)."""
    import pytest
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = pytest.main(["-q", "-p", "no:cacheprovider", "--import-mode=importlib", "--rootdir", str(TMP_DIR), "--color=no", "--capture=no", str(TMP_DIR / "test_shop_tools.py")])
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    return int(code), lines[-1] if lines else ""


code, summary = run_pytest()
print("exit code", code, "|", summary)
assert code == 0, summary

# %% [markdown]
# Green tests prove little unless they **can** turn red. A quick mutation check: temporarily replace `count_orders` with an off-by-one version (via `unittest.mock`), run the same tests, and expect failures.

# %%
real_count = shop_tools.count_orders
with mock.patch.object(shop_tools, "count_orders", lambda status: {**real_count(status), "count": real_count(status)["count"] + 1}):
    bad_code, bad_summary = run_pytest()
print("with a buggy count_orders -> exit code", bad_code, "|", bad_summary)
assert bad_code != 0, "the tests failed to notice a deliberate bug"
assert run_pytest()[0] == 0, "and they are green again after the patch is removed"

# %% [markdown]
# *What just happened?* Fourteen test cases (some parametrised) passed against real data, then several **failed** when a bug was injected, then all passed again. That is the minimum evidence that a test suite is doing something. The same habit applies higher up the pyramid:
# when you add a replay test or an eval case, deliberately break the thing it guards and confirm the test notices.

# %% [markdown]
# ## 3. The agent, instrumented
#
# A compact loop (same contract as before) with the hooks this notebook needs: a **tracer** (spans for the run, each model call and each tool call), a **deadline**, and a **model label** for cost. The model here is the deterministic mock from earlier notebooks:
# it asks for needed tools in one turn and answers from the results.

# %%
def est_tokens(x):
    x = x if isinstance(x, str) else json.dumps(x)
    return max(1, math.ceil(len(x) / 4))


@dataclass
class LLMResponse:
    content: list
    stop_reason: str = "end_turn"
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    @property
    def text(self):
        return "".join(b["text"] for b in self.content if b["type"] == "text")

    @property
    def tool_calls(self):
        return [b for b in self.content if b["type"] == "tool_use"]


def respond(blocks, messages, stop=None):
    stop = stop or ("tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn")
    return LLMResponse(blocks, stop, {"input_tokens": sum(est_tokens(m["content"]) for m in messages), "output_tokens": sum(est_tokens(b) for b in blocks)})


text_block = lambda t: {"type": "text", "text": t}
blocks_of = lambda m: [text_block(m["content"])] if isinstance(m["content"], str) else m["content"]


class TransientLLMError(Exception):
    pass


class ShopAgentLLM:
    def complete(self, messages, system="", tools=None, max_tokens=1024):
        question, uses, outcomes = messages[0]["content"], {}, []
        for m in messages[1:]:
            for b in blocks_of(m):
                if b["type"] == "tool_use":
                    uses[b["id"]] = b
                elif b["type"] == "tool_result":
                    outcomes.append((uses[b["tool_use_id"]], b.get("is_error", False), b["content"]))
        if not outcomes:
            blocks = [{"type": "tool_use", "id": f"toolu_{k + 1}", "name": "get_order", "input": {"order_id": int(o)}} for k, o in enumerate(re.findall(r"\b\d{3,}\b", question))]
            if m := re.search(r"how many (\w+) orders", question, re.I):
                blocks.append({"type": "tool_use", "id": f"toolu_{len(blocks) + 1}", "name": "count_orders", "input": {"status": m.group(1).lower()}})
            elif re.search(r"return|refund|cancel|shipping|warranty|points|payment|window", question, re.I):
                blocks.append({"type": "tool_use", "id": f"toolu_{len(blocks) + 1}", "name": "search_kb", "input": {"query_text": question}})
            return respond(([text_block("Let me check.")] + blocks) if blocks else [text_block("I can help with orders and policies.")], messages)
        parts = []
        for use, err, content in outcomes:
            d = json.loads(content) if not err else None
            if err:
                parts.append(f"I could not complete {use['name']}: {content}")
            elif use["name"] == "get_order":
                parts.append(f"Order {d['order_id']} is {d['status']} ({d['items']} item(s), total {d['total']}).")
            elif use["name"] == "count_orders":
                parts.append(f"There are {d['count']} {d['status']} orders.")
            else:
                parts.append("Policy: " + d["text"])
        return respond([text_block(" ".join(parts))], messages)


class SimClock:
    """A fake clock: time only moves when we advance it, so 'latency' is reproducible."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def advance(self, s):
        self.t += s


class SimLatencyLLM:
    """Wraps a model and advances the clock by a modelled latency: 0.4 s overhead + 20 ms per output token, with seeded jitter. A model, not a measurement."""

    def __init__(self, inner, clock, rng):
        self.inner, self.clock, self.rng = inner, clock, rng

    def complete(self, *args, **kwargs):
        resp = self.inner.complete(*args, **kwargs)
        self.clock.advance((0.4 + 0.02 * resp.usage["output_tokens"]) * (0.8 + 0.4 * self.rng.random()))
        return resp


class Tracer:
    """Minimal tracing: nested spans with ids, timestamps (from the injected clock), attributes and status. Export = a list of JSON-able dicts."""

    def __init__(self, clock):
        self.clock, self.spans, self._stack = clock, [], []
        self.trace_id = uuid.uuid4().hex[:16]

    @contextmanager
    def span(self, name, **attrs):
        rec = {"trace_id": self.trace_id, "span_id": f"s{len(self.spans) + 1}", "parent_id": self._stack[-1] if self._stack else None, "name": name,
               "start": self.clock.now(), "end": None, "status": "ok", "attrs": dict(attrs)}
        self.spans.append(rec)
        self._stack.append(rec["span_id"])
        try:
            yield rec
        except Exception as exc:
            rec["status"], rec["attrs"]["error"] = "error", type(exc).__name__
            raise
        finally:
            rec["end"] = self.clock.now()
            self._stack.pop()


TOOL_FUNCS = {"get_order": shop_tools.get_order, "count_orders": shop_tools.count_orders, "search_kb": shop_tools.search_kb}
TOOL_SCHEMAS = [{"name": n, "description": n, "input_schema": {"type": "object"}} for n in TOOL_FUNCS]
TOOL_SECONDS = 0.03                               # modelled duration of one tool call


@dataclass
class RunResult:
    text: str
    status: str
    steps: int
    messages: list
    tool_calls: list
    usage: dict


class Agent:
    def __init__(self, llm, clock, tracer, model="claude-sonnet-5-5", max_steps=6, deadline_s=None):
        self.llm, self.clock, self.tracer, self.model, self.max_steps, self.deadline_s = llm, clock, tracer, model, max_steps, deadline_s

    def run(self, question):
        messages, calls, usage = [{"role": "user", "content": question}], [], {"input_tokens": 0, "output_tokens": 0}
        start = self.clock.now()
        with self.tracer.span("agent.run", model=self.model, question_chars=len(question)) as root:          # log sizes, not content
            def finish(text, status, steps):
                root["attrs"].update(status=status, steps=steps)
                return RunResult(text, status, steps, messages, calls, usage)

            for step in range(1, self.max_steps + 1):
                if self.deadline_s is not None and self.clock.now() - start > self.deadline_s:
                    return finish("", "deadline_exceeded", step - 1)
                try:
                    with self.tracer.span("llm.call", model=self.model, step=step) as sp:
                        resp = self.llm.complete(messages, system="You are a shop support assistant.", tools=TOOL_SCHEMAS)
                        sp["attrs"].update(stop_reason=resp.stop_reason, **resp.usage)
                except TransientLLMError:
                    return finish("", "llm_unavailable", step - 1)
                usage = {k: usage[k] + resp.usage[k] for k in usage}
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason != "tool_use":
                    return finish(resp.text, "done", step)
                results = []
                for c in resp.tool_calls:
                    with self.tracer.span("tool.call", tool=c["name"]) as sp:
                        try:
                            out, err = json.dumps(TOOL_FUNCS[c["name"]](**c["input"])), False
                        except shop_tools.ToolError as exc:
                            out, err = str(exc), True
                        self.clock.advance(TOOL_SECONDS)
                        sp["attrs"].update(is_error=err, result_chars=len(out))
                    calls.append((c["name"], c["input"], err))
                    results.append({"type": "tool_result", "tool_use_id": c["id"], "content": out, **({"is_error": True} if err else {})})
                messages.append({"role": "user", "content": results})
            return finish("", "max_steps", self.max_steps)


def make_agent(llm=None, seed=0, **kwargs):
    clock = SimClock()
    tracer = Tracer(clock)
    return Agent(SimLatencyLLM(llm or ShopAgentLLM(), clock, random.Random(seed)), clock, tracer, **kwargs), tracer, clock


agent, tracer, clock = make_agent()
res = agent.run("Order 1000: where is it, and how many delivered orders are there?")
print(res.status, "|", res.text)

# %% [markdown]
# ## 4. Replayable transcripts
#
# A **recording** stores, for each model call, a fingerprint of the request (messages, system prompt, tool names) and the response. A **replay** model serves the recorded responses in order **and checks each incoming request against its fingerprint**.
# Then a test can run the *whole loop* with no API key, no cost and no randomness, and it fails loudly if you changed something that affects what the model would be sent: a prompt, a tool schema, a tool's output format, the order of calls.
# Record once against the real API (a `no-run` cell below shows how), commit the JSON file, and replay it in CI. What replay does **not** test: whether the model would still answer well; that needs the real-model evals.

# %%
def request_key(messages, system, tools):
    payload = json.dumps([messages, system, [t["name"] for t in tools or []]], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class RecordingLLM:
    def __init__(self, inner):
        self.inner, self.records = inner, []

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        resp = self.inner.complete(messages, system=system, tools=tools, max_tokens=max_tokens)
        self.records.append({"request": request_key(messages, system, tools), "response": {"content": resp.content, "stop_reason": resp.stop_reason, "usage": resp.usage}})
        return resp


class ReplayMismatch(AssertionError):
    pass


class ReplayLLM:
    def __init__(self, records):
        self.records, self.i = list(records), 0

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        if self.i >= len(self.records):
            raise ReplayMismatch(f"call {self.i + 1}: more model calls than were recorded ({len(self.records)})")
        rec, key = self.records[self.i], request_key(messages, system, tools)
        if key != rec["request"]:
            raise ReplayMismatch(f"call {self.i + 1}: the request differs from the recording (expected {rec['request']}, got {key})")
        self.i += 1
        return LLMResponse(**rec["response"])


QUESTION = "Order 1000: where is it, and how many delivered orders are there?"
recorder = RecordingLLM(ShopAgentLLM())
agent, _, _ = make_agent(recorder)
original = agent.run(QUESTION)
fixture = TMP_DIR / "recording.json"
fixture.write_text(json.dumps({"question": QUESTION, "records": recorder.records}, indent=1), encoding="utf-8")
print("recorded", len(recorder.records), "model calls ->", fixture.name, f"({fixture.stat().st_size} bytes)")

loaded = json.loads(fixture.read_text(encoding="utf-8"))
agent, _, _ = make_agent(ReplayLLM(loaded["records"]))
replayed = agent.run(loaded["question"])
assert (replayed.text, replayed.status, replayed.tool_calls) == (original.text, original.status, original.tool_calls)
print("replay reproduces the run exactly:", replayed.status, "|", [n for n, _, _ in replayed.tool_calls])

# %% [markdown]
# Now the replay catches regressions. Change something that affects the model's input (here: a tool starts returning one extra field) and replay the same recording:

# %%
real_get_order = shop_tools.get_order
with mock.patch.dict(TOOL_FUNCS, {"get_order": lambda order_id: {**real_get_order(order_id), "warehouse": "W3"}}):
    agent, _, _ = make_agent(ReplayLLM(loaded["records"]))
    try:
        agent.run(loaded["question"])
    except ReplayMismatch as exc:
        print("ReplayMismatch:", exc)

# %% [markdown]
# *What just happened?* The tool change altered the `tool_result` that the loop sends back, so the second request no longer matched its recording and the test failed at the exact call that diverged. That is the point: an innocent-looking change to a tool output
# *would* change what the real model sees, and you find out in CI instead of in production. When a change is intentional, re-record the fixture and review the diff like any other code change.

# %% [markdown]
# ## 5. Observability: spans
#
# The agent above wrapped every run, model call and tool call in a **span** with an id, a parent, start and end time, attributes and a status: the same vocabulary as OpenTelemetry (trace, span, attributes), which is what you would use in production with a
# real backend (check its current status and semantic conventions for generative-AI attributes). Here the spans are plain dicts so you can see exactly what is in them. **Never put prompts, tool arguments with personal data, or results into span attributes by default**; we record
# sizes, token counts, names and error flags.

# %%
agent, tracer, clock = make_agent()
agent.run("Order 1000: where is it, and how many delivered orders are there?")
for s in tracer.spans[:3]:
    print(json.dumps(s)[:200])


def render(spans):
    by_parent = {}
    for s in spans:
        by_parent.setdefault(s["parent_id"], []).append(s)

    def walk(parent, depth):
        for s in by_parent.get(parent, []):
            a = s["attrs"]
            detail = ", ".join(f"{k}={a[k]}" for k in ("tool", "step", "stop_reason", "input_tokens", "output_tokens", "is_error", "status") if k in a)
            print(f"{'  ' * depth}{s['name']:<11} {s['end'] - s['start']:>6.3f}s  {s['status']:<5} {detail}")
            walk(s["span_id"], depth + 1)
    walk(None, 0)


print()
render(tracer.spans)

# %% [markdown]
# *What just happened?* One trace, five spans: the run, two model calls (with stop reason and token counts), and two tool calls nested under the run. Reading the tree answers the first three incident questions: *how many steps?* *which tool was slow or failing?* *what did the tokens go to?*
# Timings here are on the simulated clock, so they are exactly reproducible; real runs use the real clock.
#
# ## 6. Cost and latency accounting
#
# Cost per run = `input_tokens x input price + output_tokens x output price` summed over model calls, which the spans already carry. Keep prices in **configuration**, keyed by model id, and verify them against the provider's pricing page: they change.
# The figures below are the list prices noted when this notebook was written (USD per million tokens, input / output). The token counts are estimates (characters / 4), so the dollar figures show relative cost, not your real bill.

# %%
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00), "claude-haiku-4-5": (1.00, 5.00)}      # verify on the pricing page


def cost_of_call(model, in_tokens, out_tokens):
    p_in, p_out = PRICES[model]
    return (in_tokens * p_in + out_tokens * p_out) / 1e6


questions = ([f"What is the status of order {1000 + i}?" for i in range(0, 20)] + [f"How many {s} orders are there?" for s in ("delivered", "shipped", "cancelled", "returned")] * 2
             + ["What is the return window for electronics?", "Order 99999999 please", "Orders 1001 and 1002: status?"] * 3)
rows = []
for i, q in enumerate(questions):
    a, t, c = make_agent(seed=i)
    r = a.run(q)
    llm_spans = [s for s in t.spans if s["name"] == "llm.call"]
    rows.append({"status": r.status, "steps": r.steps, "latency_s": c.now(), "input_tokens": r.usage["input_tokens"], "output_tokens": r.usage["output_tokens"],
                 "cost_usd": sum(cost_of_call(s["attrs"]["model"], s["attrs"]["input_tokens"], s["attrs"]["output_tokens"]) for s in llm_spans),
                 "tool_errors": sum(e for *_, e in r.tool_calls)})
runs = pd.DataFrame(rows)
report = pd.Series({"runs": len(runs), "done_rate": (runs.status == "done").mean(), "steps_mean": runs.steps.mean(), "latency_p50_s": runs.latency_s.quantile(0.5),
                    "latency_p95_s": runs.latency_s.quantile(0.95), "cost_mean_usd": runs.cost_usd.mean(), "cost_per_1000_runs_usd": runs.cost_usd.mean() * 1000,
                    "tool_error_rate": (runs.tool_errors > 0).mean()})
print(report.round(4).to_string())
print("\nslowest run:", runs.latency_s.idxmax(), "->", questions[runs.latency_s.idxmax()], f"({runs.latency_s.max():.2f}s)")

# %% [markdown]
# Why **percentiles** and not the mean: a few slow runs (here, multi-tool questions) barely move the mean but define the experience of your unluckiest users, and p95 is what a timeout and an SLO must be set against. Track cost per run and its tail too: one runaway loop costs as
# much as hundreds of normal runs, which is why the step limit, deadline and budgets from earlier notebooks are operational controls, not just safety ones. Useful alerts: error and non-`done` status rate, p95 latency, cost per run above a threshold, tool error rate per tool, and
# the step-count distribution shifting (a prompt change that doubles the steps doubles the bill).
#
# ## 7. Deploying: an endpoint you can test
#
# A deployed agent is a web service with extra duties. The app below is built by a **factory** that takes a function producing an agent and tracer, so tests can inject any model (scripted, flaky, recorded) without touching the endpoint:
# - **Auth that fails closed:** the key comes from an environment variable; if it is unset, *every* request is rejected. Compared in constant time. Never hard-code or log keys.
# - **Input validation** with Pydantic: empty and oversized messages are rejected before any model call (cost control).
# - **Rate limiting** per key (a hook; Exercise 3 writes the limiter).
# - **Honest status codes:** `llm_unavailable` is a 503, a deadline overrun a 504, so monitoring and clients can distinguish "the service is down" from "the agent gave an answer". A trace id is returned in a header so a support ticket can be tied to a trace.
# - A `/healthz` endpoint for the orchestrator.

# %%
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

API_KEY = secrets.token_hex(8)                      # a throwaway test key; real keys come from a secret manager
os.environ["AGENT_API_KEY"] = API_KEY


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    status: str
    steps: int
    trace_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


STATUS_TO_HTTP = {"done": 200, "truncated": 200, "refused": 200, "max_steps": 200, "llm_unavailable": 503, "deadline_exceeded": 504}


def create_app(agent_factory, limiter=None):
    app = FastAPI(title="shop-agent")

    def require_key(x_api_key: str | None = Header(default=None)):
        expected = os.environ.get("AGENT_API_KEY")
        if not expected or not x_api_key or not secrets.compare_digest(x_api_key, expected):        # unset key -> reject everything
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        if limiter is not None and not limiter.allow(x_api_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_key)])
    def chat(req: ChatRequest, response: Response):
        agent, tracer = agent_factory()
        result = agent.run(req.message)
        response.status_code = STATUS_TO_HTTP.get(result.status, 500)
        response.headers["x-trace-id"] = tracer.trace_id
        cost = sum(cost_of_call(s["attrs"]["model"], s["attrs"]["input_tokens"], s["attrs"]["output_tokens"]) for s in tracer.spans if s["name"] == "llm.call" and "input_tokens" in s["attrs"])      # a failed model call has no usage
        return ChatResponse(answer=result.text, status=result.status, steps=result.steps, trace_id=tracer.trace_id, input_tokens=result.usage["input_tokens"],
                            output_tokens=result.usage["output_tokens"], cost_usd=round(cost, 6))
    return app


class FailingLLM:
    def complete(self, *a, **k):
        raise TransientLLMError("overloaded (simulated)")


def factory_for(llm=None, **agent_kwargs):
    def factory():
        agent, tracer, _ = make_agent(llm, **agent_kwargs)
        return agent, tracer
    return factory


client = TestClient(create_app(factory_for()))
H = {"x-api-key": API_KEY}
orders = pd.read_csv(DATA / "orders.csv")
r = client.post("/chat", json={"message": "How many delivered orders are there?"}, headers=H)
body = r.json()
expected = int((orders.status == "delivered").sum())
print(r.status_code, body["status"], "|", body["answer"], "| trace id header matches body:", r.headers["x-trace-id"] == body["trace_id"])
assert r.status_code == 200 and f"There are {expected} delivered orders." == body["answer"] and body["cost_usd"] > 0

# %% [markdown]
# Now test the endpoint's **behaviour** for every failure mode (this is ordinary API testing; nothing agent-specific except that we inject failing models):

# %%
checks = []
def check(name, response, code):
    checks.append({"case": name, "http": response.status_code, "expected": code, "ok": response.status_code == code})


check("healthz needs no key", client.get("/healthz"), 200)
check("no API key", client.post("/chat", json={"message": "hi"}), 401)
check("wrong API key", client.post("/chat", json={"message": "hi"}, headers={"x-api-key": "nope"}), 401)
check("empty message", client.post("/chat", json={"message": ""}, headers=H), 422)
check("oversized message", client.post("/chat", json={"message": "x" * 2001}, headers=H), 422)
check("missing field", client.post("/chat", json={}, headers=H), 422)
check("model API down -> 503", TestClient(create_app(factory_for(FailingLLM()))).post("/chat", json={"message": "hi"}, headers=H), 503)
check("deadline exceeded -> 504", TestClient(create_app(factory_for(deadline_s=0.5))).post("/chat", json={"message": "Orders 1001 and 1002: status?"}, headers=H), 504)
saved = os.environ.pop("AGENT_API_KEY")
check("server key not configured -> reject", client.post("/chat", json={"message": "hi"}, headers=H), 401)
os.environ["AGENT_API_KEY"] = saved
print(pd.DataFrame(checks).to_string(index=False))
assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]

# %% [markdown]
# *What just happened?* Nine endpoint behaviours, all asserted: authentication fails closed (even with the right client key, an unconfigured server rejects everything), bad input is rejected **before** any model call, and failures of the model or a slow run surface as 503 and 504
# instead of a misleading 200. The deadline case is interesting: the agent stopped *between* steps once the simulated 0.5 s budget was exceeded. A deadline checked only between steps cannot interrupt a single slow model call, so real deployments also set client
# timeouts on the model call itself (for the Anthropic SDK: the `timeout` and `max_retries` client options) and a server-level request timeout.
#
# ### Deployment checklist
#
# 1. **Secrets** from the environment or a secret manager; never in code, images, logs or notebooks. Rotate.
# 2. **Timeouts everywhere:** model call, tool call, whole request. Retries bounded with backoff and jitter (notebook 01). Think about idempotency before retrying a tool with side effects.
# 3. **Limits:** max steps, max tokens per run, deadline, per-user rate limit, per-user and global daily budget. A kill switch (feature flag) that disables the agent or an individual tool.
# 4. **Authentication and authorisation** in code, per user, including for tool arguments (a model-supplied id never decides whose data is read). Least-privilege tool set per route.
# 5. **Untrusted input handling:** everything the agent reads is untrusted (notebook 22/04); irreversible tools need confirmation; output validation on what leaves.
# 6. **Observability:** a trace id per request, spans for model and tool calls, metrics (error rate by status, p50/p95 latency, cost per run, steps, tool error rate), alerts. Logs without personal data by default; full capture only under access control and retention limits.
# 7. **Release gates:** tool unit tests, replay tests and a golden-set evaluation in CI; version-pin the model id and the prompt; canary or shadow a new prompt/model before full traffic, compare eval rates, keep rollback simple.
# 8. **Graceful failure:** honest status codes, a fallback (shorter answer, a human handoff, a cached answer) when the model is unavailable, and user-facing messages that do not promise what the agent might not do.
# 9. **Statefulness:** this endpoint is stateless (a fresh agent per request). Multi-turn conversations need a session store with expiry and deletion, and context management (notebook 03).
# 10. **Capacity:** model rate limits and concurrency limits at the provider, queueing or load shedding at your edge.

# %% [markdown]
# ## 8. Recording real fixtures and serving for real (`no-run`)
#
# Recording wraps the real client wrapper exactly as it wrapped the mock; run it once, commit the JSON. Serving uses the same app object. Needs `ANTHROPIC_API_KEY`, network access, and (for `uvicorn`) a long-running process, so nothing here executes in the notebook.

# %% tags=["no-run"]
import anthropic


class AnthropicLLM:
    def __init__(self, client=None, model="claude-sonnet-5-5"):
        self.client, self.model = client or anthropic.Anthropic(timeout=30.0, max_retries=2), model      # reads ANTHROPIC_API_KEY from the environment

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]
        return LLMResponse(blocks, resp.stop_reason, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens})


# 1) record a real trajectory once (note: real tool schemas should replace TOOL_SCHEMAS stubs so the model can see them)
rec = RecordingLLM(AnthropicLLM())
wall = SimClock()                                  # swap in a real-time clock (time.monotonic) when you trace real latency
real_agent = Agent(rec, wall, Tracer(wall))
real_agent.run(QUESTION)
Path("fixtures").mkdir(exist_ok=True)
Path("fixtures/order_and_count.json").write_text(json.dumps({"question": QUESTION, "records": rec.records}, indent=1))

# 2) serve it: save create_app / factory code in app.py, then from a shell
#    export AGENT_API_KEY=...   (from your secret manager)
#    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 5
# A minimal container image would install the dependencies, copy the app, run as a non-root user, and start uvicorn; keep the API key out of the image.

# %% [markdown]
# ## Common mistakes
#
# 1. **Testing only with the real model.** Slow, costly, flaky; you cannot run it on every commit. Put deterministic tests under it.
# 2. **Green tests that cannot fail.** Inject a deliberate bug and confirm the suite notices (as in section 2).
# 3. **Replaying without checking the request.** If the replay serves responses regardless of input, it will happily pass after you broke the prompt or a tool.
# 4. **Logging prompts and tool results by default.** You just built a personal-data store with weak access control. Log sizes, ids, names, statuses.
# 5. **Averages instead of percentiles** for latency and cost, and no alert on the tail or on runaway runs.
# 6. **Auth that fails open** (missing env var means "no auth") or keys compared with `==` and logged.
# 7. **Returning 200 for everything.** Clients and monitors cannot tell an answer from an outage.
# 8. **A deadline only between steps.** One hung model call still hangs the request; set timeouts on the call itself.

# %%
# Demonstration of mistake 3: a replay that ignores the request passes even after the tool output changed
class NaiveReplay:
    def __init__(self, records):
        self.records, self.i = list(records), 0

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        rec = self.records[self.i]
        self.i += 1
        return LLMResponse(**rec["response"])


with mock.patch.dict(TOOL_FUNCS, {"get_order": lambda order_id: {**real_get_order(order_id), "warehouse": "W3"}}):
    agent, _, _ = make_agent(NaiveReplay(loaded["records"]))
    print("naive replay status after breaking a tool:", agent.run(loaded["question"]).status, "(it did not notice)")

# %% [markdown]
# ## Interview corner
#
# **1. How do you test an LLM agent when the model is non-deterministic?**
# <details><summary>Answer</summary>
# Layer it. Unit-test tools deterministically. Test the loop, prompts and wiring with recorded model responses replayed against request fingerprints, so changes to anything the model would see fail the test. Evaluate quality with a golden set against the real model, run
# repeatedly, reporting rates with counts, plus adversarial cases. In production monitor status rates, latency, cost and tool errors, and feed incidents back as new test cases.
# </details>
#
# **2. What do you put in a trace for an agent run?**
# <details><summary>Answer</summary>
# A trace per request with spans for the run, each model call (model, stop reason, input and output tokens, latency) and each tool call (name, error flag, duration, result size), with parent-child links and a trace id returned to the caller. Keep personal content out by default.
# From spans you derive step counts, cost per run, p95 latency and tool error rates.
# </details>
#
# **3. The agent's bill tripled after a prompt change. How do you find out why?**
# <details><summary>Answer</summary>
# Compare span aggregates before and after: steps per run, input tokens per call, tool result sizes, retry and error rates, and the share of runs hitting limits. Typical causes: more steps (the model now calls an extra tool or loops), larger tool results re-sent each step, lost prompt caching
# from a changing prefix, or a fallback to a more expensive model. Prevent it with cost-per-run alerts and an eval gate that reports steps and cost.
# </details>
#
# **4. What status codes should an agent endpoint return, and why does it matter?**
# <details><summary>Answer</summary>
# 200 with a status field for any completed answer (including "could not answer"), 401/403 for auth failures, 422 for invalid input rejected before spending tokens, 429 when rate-limited, 503 when the model provider is unavailable and 504 when the request timed out. Honest codes let
# clients retry correctly and let monitoring separate outages from quality issues.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Tool report from spans
# Write `tool_report(spans)` returning a dict `{tool_name: {"calls": int, "errors": int, "seconds": float}}` built from the `tool.call` spans (`attrs["tool"]`, `attrs["is_error"]`, duration `end - start`). Ignore other span names. The check compares your report with counts taken independently
# from the agents' own `tool_calls` lists over several traced runs, including errors.
# *Hint: iterate over `spans`, filter `name == "tool.call"`.*

# %% tags=["starter"]
def tool_report(spans):
    ...


# %%
all_spans, expected_counts, expected_errors = [], {}, {}
for i, q in enumerate(["Orders 1000 and 1001: status?", "How many delivered orders are there?", "Order 99999999 please", "What is the return window for electronics?", "Order 1002: status? How many shipped orders?"]):
    a, t, c = make_agent(seed=i)
    r = a.run(q)
    all_spans += t.spans
    for name, _, err in r.tool_calls:
        expected_counts[name] = expected_counts.get(name, 0) + 1
        expected_errors[name] = expected_errors.get(name, 0) + int(err)
rep = tool_report(all_spans)
assert set(rep) == set(expected_counts), (set(rep), set(expected_counts))
for name in rep:
    assert rep[name]["calls"] == expected_counts[name] and rep[name]["errors"] == expected_errors[name], (name, rep[name])
    assert abs(rep[name]["seconds"] - TOOL_SECONDS * expected_counts[name]) < 1e-9, "each modelled tool call lasts TOOL_SECONDS"
assert rep["get_order"]["errors"] == 1, "the unknown order must be counted as an error"
assert tool_report([]) == {} and tool_report([s for s in all_spans if s["name"] != "tool.call"]) == {}
print("Exercise 1 passed", {k: v["calls"] for k, v in rep.items()})

# %% tags=["solution"]
def tool_report(spans):
    out = {}
    for s in spans:
        if s["name"] != "tool.call":
            continue
        row = out.setdefault(s["attrs"]["tool"], {"calls": 0, "errors": 0, "seconds": 0.0})
        row["calls"] += 1
        row["errors"] += int(bool(s["attrs"].get("is_error")))
        row["seconds"] += s["end"] - s["start"]
    return out


# %% [markdown]
# ### Exercise 2: Cost of a run from spans, with a budget check
# Write `run_cost(spans, prices)` summing, over `llm.call` spans, `input_tokens * price_in + output_tokens * price_out` per million tokens, where the price pair comes from `prices[span.attrs["model"]]`. Raise `KeyError` with the model id in the message for an unknown model
# (never silently price it at zero). Then write `over_budget(spans, prices, limit_usd)` returning `True` when the run costs **more than** the limit.
# *Hint: `cost_of_call` shows the arithmetic for one call.*

# %% tags=["starter"]
def run_cost(spans, prices):
    ...


def over_budget(spans, prices, limit_usd):
    ...


# %%
fake = [{"name": "agent.run", "attrs": {}}, {"name": "llm.call", "attrs": {"model": "m1", "input_tokens": 1000, "output_tokens": 100}},
        {"name": "tool.call", "attrs": {"tool": "x"}}, {"name": "llm.call", "attrs": {"model": "m1", "input_tokens": 2000, "output_tokens": 50}},
        {"name": "llm.call", "attrs": {"model": "m2", "input_tokens": 500, "output_tokens": 500}}]
prices = {"m1": (3.0, 15.0), "m2": (1.0, 5.0)}
exp = (1000 * 3 + 100 * 15 + 2000 * 3 + 50 * 15 + 500 * 1 + 500 * 5) / 1e6
assert abs(run_cost(fake, prices) - exp) < 1e-12, (run_cost(fake, prices), exp)
assert run_cost([fake[0], fake[2]], prices) == 0
try:
    run_cost(fake, {"m1": (3.0, 15.0)})
except KeyError as exc:
    assert "m2" in str(exc)
else:
    raise AssertionError("unknown model must raise KeyError")
assert over_budget(fake, prices, exp - 1e-9) is True and over_budget(fake, prices, exp) is False and over_budget(fake, prices, 1.0) is False
a, t, c = make_agent()
a.run("Orders 1001 and 1002: status?")
assert abs(run_cost(t.spans, PRICES) - sum(cost_of_call(s["attrs"]["model"], s["attrs"]["input_tokens"], s["attrs"]["output_tokens"]) for s in t.spans if s["name"] == "llm.call")) < 1e-12
print("Exercise 2 passed")

# %% tags=["solution"]
def run_cost(spans, prices):
    total = 0.0
    for s in spans:
        if s["name"] == "llm.call":
            model = s["attrs"]["model"]
            if model not in prices:
                raise KeyError(f"no price configured for model {model!r}")
            p_in, p_out = prices[model]
            total += (s["attrs"]["input_tokens"] * p_in + s["attrs"]["output_tokens"] * p_out) / 1e6
    return total


def over_budget(spans, prices, limit_usd):
    return run_cost(spans, prices) > limit_usd


# %% [markdown]
# ### Exercise 3: A rate limiter for the endpoint
# Write a class `RateLimiter(max_requests, window_s, clock)` with `allow(key) -> bool`: a **sliding window** per key. A request is allowed if fewer than `max_requests` requests from that key were *allowed* in the last `window_s` seconds (strictly: timestamps `> now - window_s` count). `clock` is a
# zero-argument function returning seconds (inject a fake one in tests). Keys are independent. The check also plugs your limiter into `create_app` and expects HTTP 429 on the request after the limit, while `/healthz` stays open.
# *Hint: keep a `collections.deque` of timestamps per key and drop old ones on each call.*

# %% tags=["starter"]
class RateLimiter:
    def __init__(self, max_requests, window_s, clock):
        ...

    def allow(self, key):
        ...


# %%
now = [0.0]
lim = RateLimiter(3, 10.0, clock=lambda: now[0])
assert [lim.allow("a") for _ in range(4)] == [True, True, True, False], "4th request inside the window must be refused"
assert lim.allow("b") is True, "keys are independent"
now[0] = 9.9
assert lim.allow("a") is False, "still inside the window (first request at t=0 has not expired)"
now[0] = 10.1
assert lim.allow("a") is True, "the t=0 requests have expired"
assert [lim.allow("c") for _ in range(3)] == [True, True, True] and lim.allow("c") is False
now[0] = 10.2
refused = RateLimiter(1, 10.0, clock=lambda: now[0])
assert refused.allow("k") and not refused.allow("k") and not refused.allow("k"), "refused requests must not extend the block"
now[0] = 20.3
assert refused.allow("k") is True
# wired into the real app
t0 = [0.0]
limited = TestClient(create_app(factory_for(), limiter=RateLimiter(2, 60.0, clock=lambda: t0[0])))
codes = [limited.post("/chat", json={"message": "How many shipped orders are there?"}, headers=H).status_code for _ in range(3)]
assert codes == [200, 200, 429], codes
assert limited.get("/healthz").status_code == 200
t0[0] = 61.0
assert limited.post("/chat", json={"message": "How many shipped orders are there?"}, headers=H).status_code == 200
print("Exercise 3 passed")

# %% tags=["solution"]
from collections import deque


class RateLimiter:
    def __init__(self, max_requests, window_s, clock):
        self.max_requests, self.window_s, self.clock, self.hits = max_requests, window_s, clock, {}

    def allow(self, key):
        now = self.clock()
        q = self.hits.setdefault(key, deque())
        while q and q[0] <= now - self.window_s:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True


# %% [markdown]
# ### Exercise 4 (open-ended): A pre-launch review
# Pick an agent you want to ship. Walk the deployment checklist above and write one sentence per item on **what you have, what is missing, and the worst thing that happens if it stays missing**. Then choose the three items you would fix before launch, and write the replay test and
# the one real-model eval case that would protect the most important behaviour. No assert for this one.

# %% [markdown]
# ## Cleanup
# Remove the temp folder, the module import and the environment variable:

# %%
sys.path.remove(str(TMP_DIR))
for name in ("shop_tools", "test_shop_tools"):
    sys.modules.pop(name, None)
os.environ.pop("AGENT_API_KEY", None)
TMP.cleanup()
print("temp folder removed:", not TMP_DIR.exists())

# %% [markdown]
# ## Recap
# - Test agents as a **pyramid**: deterministic tool tests (with a mutation check), recorded-trajectory replay tests that verify each request fingerprint, real-model evals on a golden set, and production monitoring.
# - Replay lets CI run the whole loop with no key, cost or randomness, and fails at the exact call where a prompt, schema or tool output changed; re-record deliberately.
# - Trace every run with spans (run, model call, tool call): ids, parents, durations, tokens, stop reasons, error flags; keep content out of spans by default. Derive steps, cost per run, p50/p95 latency and tool error rates.
# - Serve with a factory-built FastAPI app: auth that fails closed, validated and bounded input, rate limits, honest status codes (422, 429, 503, 504), a trace id header; test every failure mode with `TestClient`.
# - Ship with a checklist: secrets, timeouts, budgets, authorisation, untrusted-input handling, observability, release gates, graceful failure, state, capacity.
#
# **Go deeper:** the FastAPI documentation on testing (fastapi.tiangolo.com); the pytest documentation (docs.pytest.org); OpenTelemetry documentation (opentelemetry.io); Anthropic's documentation on errors, rate limits and the Python SDK (docs.anthropic.com).
#
# **Next:** [Model Context Protocol: what is MCP?](../24_MCP/01_what_is_mcp.ipynb)
