# %% [markdown]
# # Build an Agent Loop from Scratch (Production-Shaped)
#
# **Why this matters.** The ten-line agent loop in module 22 is enough to understand agents and not enough to run one for customers. A production loop has to survive a tool that throws, a model API that returns 529 at
# 3 a.m., an answer cut off by `max_tokens`, a refusal, a model that calls the same tool forever, and an engineer who wants to know *why* yesterday's run took nine steps. This notebook builds that loop piece by piece:
# message state you can trust, a tool registry with a decorator, step and error limits, retries, logging, and explicit stop reasons. Everything runs offline against a mock model; the real Claude client is a one-line swap.
#
# **You will learn**
# - generate a tool's JSON Schema from a Python function's type hints, and register tools with a decorator
# - keep conversation state valid by construction (every `tool_use` answered, even when a run is aborted)
# - handle every way a model turn can end: `end_turn`, `tool_use`, `max_tokens`, `refusal`, and transient API failures with bounded retries
# - log a run as structured events and read the trajectory afterwards
#
# **Prerequisites:** module 22 ([What is an agent?](../22_Agentic_AI/01_what_is_an_agent.ipynb), [Tool use](../22_Agentic_AI/02_tool_use_and_function_calling.ipynb), [Safety](../22_Agentic_AI/04_agent_safety_guardrails_and_evaluation.ipynb)).
#
# > Data used: `datasets/shop.db` (read-only) and the knowledge base. The "model" is a **deterministic mock** with the same `complete(messages, system, tools, max_tokens)` interface as the real client wrapper. It picks
# > tools with regular expressions; it demonstrates the loop, not intelligence. The faithful Anthropic-SDK version is in a `no-run` cell (needs network and `ANTHROPIC_API_KEY`).

# %%
import inspect
import json
import logging
import math
import random
import re
import sqlite3
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

DATA = Path("../datasets")
KB = {p.stem: p.read_text(encoding="utf-8") for p in sorted((DATA / "knowledge_base").glob("*.md"))}
DB_URI = (DATA / "shop.db").resolve().as_uri() + "?mode=ro"


def query(sql, params=()):
    con = sqlite3.connect(DB_URI, uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


print(query("select count(*) from orders")[0][0], "orders;", len(KB), "KB documents")

# %% [markdown]
# ## 1. What a production loop needs
#
# | Concern | Naive loop | Production loop |
# |---|---|---|
# | Message state | a bare list you append to | a class that refuses to become invalid and can close dangling `tool_use` blocks |
# | Tools | a dict and a dispatcher | a **registry**; schema generated from the function, validation, error results, output cap |
# | Stopping | `while True` / "until text" | `max_steps`, consecutive-error limit, explicit status for every exit |
# | Model turn outcomes | assumes `tool_use` or text | handles `end_turn`, `tool_use`, `max_tokens`, `refusal`, unknown |
# | Model API failures | crash | bounded retries with exponential backoff and jitter; a clear `llm_unavailable` status |
# | Observability | `print` | structured events (one JSON line per model call and tool call) and a result object |
#
# We build them in that order.

# %% [markdown]
# ## 2. Tools from type hints: the registry and its decorator
#
# Writing JSON Schema by hand for every tool is slow and drifts out of sync with the function. Instead derive it: parameter **names** become properties, **type hints** become JSON Schema types (`Literal[...]` becomes an
# `enum`), parameters **without a default** become `required`, and the docstring's first paragraph becomes the description the model reads. That is also the idea behind decorators like the SDK's `@beta_tool`; building one
# teaches you what they do.

# %%
SCALARS = {int: "integer", float: "number", str: "string", bool: "boolean"}


def _type_schema(tp):
    if tp in SCALARS:
        return {"type": SCALARS[tp]}
    if typing.get_origin(tp) is Literal:
        return {"type": "string", "enum": list(typing.get_args(tp))}
    raise TypeError(f"unsupported parameter type: {tp!r}")


def schema_from_function(fn, name=None):
    hints, sig = typing.get_type_hints(fn), inspect.signature(fn)
    props = {p: _type_schema(hints[p]) for p in sig.parameters}
    required = [p for p, par in sig.parameters.items() if par.default is inspect.Parameter.empty]
    description = " ".join((inspect.getdoc(fn) or "").split("\n\n")[0].split())
    return {"name": name or fn.__name__, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}


def demo(order_id: int, status: Literal["delivered", "shipped"] = "delivered", verbose: bool = False):
    """Look at an order.

    Second paragraph is for humans and is not sent to the model.
    """


print(json.dumps(schema_from_function(demo), indent=1))

# %% [markdown]
# *What just happened?* `order_id` is required (no default) and an integer; `status` is an enum with a default so it is optional. Now the registry. `@registry.tool` records the function and its schema. `execute` turns one
# `tool_use` block into one `tool_result` block and **never raises**: unknown tool, bad arguments (validated against the generated schema), an expected failure (`ToolError`, message returned verbatim so the model can act
# on it), or an unexpected exception (type and message returned; the traceback goes to the log). Results are capped with a visible truncation note.

# %%
import jsonschema


class ToolError(Exception):
    """An expected, model-actionable failure. The message is returned to the model as-is."""


class ToolRegistry:
    def __init__(self, max_result_chars=2000):
        self._tools, self.max_result_chars, self.stats = {}, max_result_chars, {}

    def tool(self, fn=None, *, name=None):
        def register(f):
            schema = schema_from_function(f, name)
            self._tools[schema["name"]] = (schema, f)
            return f
        return register(fn) if fn else register

    def schemas(self):
        return [schema for schema, _ in self._tools.values()]

    def names(self):
        return sorted(self._tools)

    def execute(self, call):
        name, args = call["name"], call["input"]
        t0 = time.perf_counter()

        def result(text, is_error=False):
            if len(text) > self.max_result_chars:
                text = text[: self.max_result_chars] + f"... [truncated {len(text) - self.max_result_chars} characters]"
            s = self.stats.setdefault(name, {"calls": 0, "errors": 0, "seconds": 0.0})
            s["calls"] += 1
            s["errors"] += int(is_error)
            s["seconds"] += time.perf_counter() - t0
            return {"type": "tool_result", "tool_use_id": call["id"], "content": text, **({"is_error": True} if is_error else {})}

        if name not in self._tools:
            return result(f"Unknown tool {name!r}. Available tools: {self.names()}", True)
        schema, fn = self._tools[name]
        problems = [f"{'/'.join(map(str, e.path)) or 'input'}: {e.message}" for e in jsonschema.Draft202012Validator(schema["input_schema"]).iter_errors(args)]
        if problems:
            return result("Invalid input. " + "; ".join(problems), True)
        try:
            out = fn(**args)
            return result(out if isinstance(out, str) else json.dumps(out))
        except ToolError as exc:
            return result(str(exc), True)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("agent").exception("tool %s crashed", name)
            return result(f"{type(exc).__name__}: {exc}", True)


registry = ToolRegistry()


@registry.tool
def get_order(order_id: int):
    """Look up ONE order by numeric id: status, date, item count and total value."""
    rows = query("select o.order_id, o.status, o.order_date, sum(i.quantity), round(sum(i.quantity * p.unit_price), 2) from orders o "
                 "join order_items i using(order_id) join products p using(product_id) where o.order_id = ? group by o.order_id", (order_id,))
    if not rows:
        raise ToolError(f"no order with id {order_id}. Order ids are positive integers.")
    return dict(zip(("order_id", "status", "date", "items", "total"), rows[0]))


@registry.tool
def count_orders(status: Literal["delivered", "shipped", "cancelled", "returned"]):
    """Count orders that have the given status."""
    return {"status": status, "count": query("select count(*) from orders where status = ?", (status,))[0][0]}


STOP = set("a an the i my me to of is it in on for do you can how what when was be and or with after have this that are does will much many".split())
words = lambda t: {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP}


@registry.tool
def search_kb(query_text: str):
    """Search the company policy knowledge base (returns, shipping, warranty, points...). Returns the best matching line and its document."""
    best = max(((len(words(query_text) & words(line)), name, line) for name, text in KB.items() for line in text.splitlines()[1:]), key=lambda x: x[0])
    return {"document": best[1], "text": best[2]} if best[0] else {"document": None, "text": "no relevant passage found"}


print(registry.names())
for c in [{"id": "1", "name": "count_orders", "input": {"status": "delivered"}}, {"id": "2", "name": "count_orders", "input": {"status": "lost"}},
          {"id": "3", "name": "get_order", "input": {"order_id": 0}}, {"id": "4", "name": "nope", "input": {}}]:
    r = registry.execute(c)
    print(f"{c['name']}({c['input']})".ljust(36), "ERR" if r.get("is_error") else "ok ", r["content"][:75])

# %% [markdown]
# ## 3. Conversation state that cannot go invalid
#
# The Messages API rejects a conversation where a `tool_use` has no matching `tool_result` in the next user message (notebook 22/02). A crashed or aborted run is the usual way to produce one. So `Conversation` owns the list:
# it validates on every model call, appends results as **one** user message, and has `close_pending()` to answer any unanswered `tool_use` with an error result when a run ends early. After that, the same conversation can
# safely be continued.

# %%
class ProtocolError(ValueError):
    pass


blocks_of = lambda m: [{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else m["content"]


class Conversation:
    def __init__(self):
        self.messages = []

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, content):
        self.messages.append({"role": "assistant", "content": content})     # the WHOLE content, unchanged

    def pending_ids(self):
        if not self.messages or self.messages[-1]["role"] != "assistant":
            return []
        return [b["id"] for b in blocks_of(self.messages[-1]) if b["type"] == "tool_use"]

    def add_tool_results(self, results):
        if {r["tool_use_id"] for r in results} != set(self.pending_ids()) or len(results) != len(self.pending_ids()):
            raise ProtocolError(f"results {[r['tool_use_id'] for r in results]} do not match pending tool_use ids {self.pending_ids()}")
        self.messages.append({"role": "user", "content": results})

    def close_pending(self, reason):
        if self.pending_ids():
            self.add_tool_results([{"type": "tool_result", "tool_use_id": i, "is_error": True, "content": f"Not executed: {reason}"} for i in self.pending_ids()])

    def validate(self):
        pending = None
        for i, m in enumerate(self.messages):
            results = [b["tool_use_id"] for b in blocks_of(m) if b["type"] == "tool_result"]
            if pending is not None and (m["role"] != "user" or sorted(results) != sorted(pending)):
                raise ProtocolError(f"message {i}: expected tool_result blocks for {sorted(pending)}")
            if pending is None and results:
                raise ProtocolError(f"message {i}: tool_result without tool_use")
            pending = [b["id"] for b in blocks_of(m) if b["type"] == "tool_use"] or None if m["role"] == "assistant" else None
        if self.messages and self.messages[-1]["role"] == "assistant" and pending:
            raise ProtocolError("conversation ends with unanswered tool_use")


conv = Conversation()
conv.add_user("q")
conv.add_assistant([{"type": "tool_use", "id": "t1", "name": "get_order", "input": {"order_id": 1}}])
try:
    conv.validate()
except ProtocolError as exc:
    print("before close_pending:", exc)
conv.close_pending("run was stopped")
conv.validate()
print("after close_pending: valid, last message:", conv.messages[-1]["content"][0]["content"])

# %% [markdown]
# ## 4. A mock model, and the ways a model call can fail
#
# `ShopAgentLLM` follows the `complete()` contract and understands three kinds of question: order status ("order 1000"), counts ("how many delivered orders"), and policies. It requests all needed tools in one turn, then
# answers from the results. To test failure handling we also need models that misbehave on purpose: a **scripted** one (returns prepared responses, including odd `stop_reason`s) and a **flaky** wrapper that raises
# transient errors before succeeding.

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
tool_use = lambda i, name, **args: {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": args}


class TransientLLMError(Exception):
    """A model-API failure worth retrying (rate limit, overload, network)."""


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
            blocks = []
            for oid in re.findall(r"\b\d{3,}\b", question):
                blocks.append(tool_use(len(blocks) + 1, "get_order", order_id=int(oid)))
            if m := re.search(r"how many (\w+) orders", question, re.I):
                blocks.append(tool_use(len(blocks) + 1, "count_orders", status=m.group(1).lower()))
            elif re.search(r"return|refund|cancel|shipping|warranty|points|premium|payment|window", question, re.I):
                blocks.append(tool_use(len(blocks) + 1, "search_kb", query_text=question))
            return respond(([text_block("Let me check.")] + blocks) if blocks else [text_block("I can help with orders and shop policies.")], messages)
        parts = []
        for use, err, content in outcomes:
            if err:
                parts.append(f"I could not complete {use['name']}: {content}")
            elif use["name"] == "get_order":
                o = json.loads(content)
                parts.append(f"Order {o['order_id']} is {o['status']} ({o['items']} item(s), total {o['total']}).")
            elif use["name"] == "count_orders":
                c = json.loads(content)
                parts.append(f"There are {c['count']} {c['status']} orders.")
            elif use["name"] == "search_kb":
                parts.append("Policy: " + json.loads(content)["text"])
        return respond([text_block(" ".join(parts))], messages)


class ScriptedLLM:
    """Returns prepared (blocks, stop_reason) items in order; records the messages it was sent."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        self.calls.append(len(messages))
        blocks, stop = self.script[len(self.calls) - 1] if len(self.calls) <= len(self.script) else self.script[-1]
        return respond(blocks, messages, stop)


class FlakyLLM:
    """Raises TransientLLMError for the first `fail_times` calls, then delegates."""

    def __init__(self, inner, fail_times):
        self.inner, self.fail_times, self.attempts = inner, fail_times, 0

    def complete(self, *args, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise TransientLLMError("529 overloaded (simulated)")
        return self.inner.complete(*args, **kwargs)


print(ShopAgentLLM().complete([{"role": "user", "content": "How many delivered orders are there, and what is the return window?"}]).content)

# %% [markdown]
# ## 5. The agent
#
# Read `run` slowly; every branch is a decision you would otherwise make at 3 a.m.
#
# - **Retries** wrap only the model call, only for `TransientLLMError`, with exponential backoff (`base * 2**attempt`, capped) and jitter so many clients do not retry in lockstep. `sleep` and the random source are injectable, so
#   tests are instant and deterministic. The real SDK already retries connection errors, 408/409/429 and 5xx twice by default (configurable with `max_retries`); keep this outer layer for *policy* (total attempts, fallbacks), not to
#   duplicate it.
# - **Stop reasons**: `end_turn` = done. `tool_use` = run tools. `max_tokens` = the answer (or a tool call) was cut off: never execute a possibly truncated tool call; report `truncated`. `refusal` = the model declined: report it.
# - **Limits**: `max_steps`, and `max_consecutive_tool_errors` (a model that keeps failing the same way should stop, not burn tokens).
# - **Aborts keep the conversation valid** by closing pending `tool_use` blocks.
# - **Events**: one JSON line per model call and per tool call through `logging`.

# %%
@dataclass
class AgentConfig:
    max_steps: int = 8
    max_consecutive_tool_errors: int = 3
    max_tokens: int = 1024
    llm_attempts: int = 3
    backoff_base: float = 0.5
    backoff_cap: float = 8.0


@dataclass
class AgentResult:
    text: str
    status: str
    messages: list
    steps: int
    tool_calls: list
    usage: dict
    retries: int = 0


log = logging.getLogger("agent")


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class Agent:
    def __init__(self, llm, registry, system="You are a shop support assistant.", config=None, sleep=time.sleep, rng=None):
        self.llm, self.registry, self.system, self.cfg = llm, registry, system, config or AgentConfig()
        self.sleep, self.rng = sleep, rng or random.Random(0)
        self.sleeps = []

    def _event(self, event, **fields):
        log.info(json.dumps({"event": event, **fields}))

    def _call_llm(self, conv):
        conv.validate()
        for attempt in range(self.cfg.llm_attempts):
            try:
                return self.llm.complete(conv.messages, system=self.system, tools=self.registry.schemas(), max_tokens=self.cfg.max_tokens), attempt
            except TransientLLMError as exc:
                if attempt == self.cfg.llm_attempts - 1:
                    raise
                delay = min(self.cfg.backoff_cap, self.cfg.backoff_base * 2**attempt) * (0.5 + self.rng.random() / 2)     # "equal jitter"
                self._event("llm_retry", attempt=attempt + 1, delay_s=round(delay, 3), error=str(exc))
                self.sleeps.append(delay)
                self.sleep(delay)

    def run(self, question, conv=None):
        conv = conv or Conversation()
        conv.add_user(question)
        usage, calls, retries, consecutive_errors = {"input_tokens": 0, "output_tokens": 0}, [], 0, 0

        def finish(text, status, step):
            conv.close_pending(f"run ended with status {status}")
            self._event("run_end", status=status, steps=step, tool_calls=len(calls), **usage)
            return AgentResult(text, status, conv.messages, step, calls, usage, retries)

        for step in range(1, self.cfg.max_steps + 1):
            try:
                resp, attempt = self._call_llm(conv)
            except TransientLLMError:
                return finish("", "llm_unavailable", step - 1)
            retries += attempt
            usage = {k: usage[k] + resp.usage[k] for k in usage}
            conv.add_assistant(resp.content)
            self._event("model_call", step=step, stop_reason=resp.stop_reason, **resp.usage)
            if resp.stop_reason == "end_turn":
                return finish(resp.text, "done", step)
            if resp.stop_reason == "max_tokens":
                return finish(resp.text, "truncated", step)
            if resp.stop_reason == "refusal":
                return finish(resp.text, "refused", step)
            if resp.stop_reason != "tool_use":
                return finish(resp.text, f"unexpected_stop_reason:{resp.stop_reason}", step)
            results = []
            for call in resp.tool_calls:
                res = self.registry.execute(call)
                calls.append((call["name"], call["input"], bool(res.get("is_error"))))
                self._event("tool_call", step=step, name=call["name"], input=call["input"], is_error=bool(res.get("is_error")), result_chars=len(res["content"]))
                results.append(res)
            conv.add_tool_results(results)
            consecutive_errors = consecutive_errors + 1 if all(r.get("is_error") for r in results) else 0
            if consecutive_errors >= self.cfg.max_consecutive_tool_errors:
                return finish("", "too_many_tool_errors", step)
        return finish("", "max_steps", self.cfg.max_steps)


handler = ListHandler()
log.handlers[:] = [handler]
log.setLevel(logging.INFO)
log.propagate = False

agent = Agent(ShopAgentLLM(), registry)
res = agent.run("Order 1000: where is it? Also, what is the return window for electronics?")
print(res.status, "|", res.steps, "steps |", [(n, i) for n, i, _ in res.tool_calls])
print(res.text)

# %% [markdown]
# The structured log of that one run (what you would ship to a log store):

# %%
for line in handler.lines:
    print(line[:150])

# %% [markdown]
# *What just happened?* Two model calls, two tool calls in one turn (parallel), a final answer, and a `run_end` event with totals. Nothing in this log needed the conversation text, yet you can already answer "how many steps, which tools,
# how many tokens". Now verify an answer against an independent path: a count question, compared with pandas over the CSV.

# %%
orders_df = pd.read_csv(DATA / "orders.csv")
for status in ("delivered", "returned"):
    r = Agent(ShopAgentLLM(), registry).run(f"How many {status} orders are there?")
    expected = int((orders_df.status == status).sum())
    assert r.status == "done" and f"There are {expected} {status} orders." == r.text, (r.text, expected)
    assert [n for n, _, _ in r.tool_calls] == ["count_orders"], r.tool_calls
    print(f"{status}: agent says {expected} (pandas agrees), trajectory = {[n for n, _, _ in r.tool_calls]}")

# %% [markdown]
# ## 6. Every way a run can end
#
# Each scenario below is a scripted or wrapped model and an expected **status**. This is what "error recovery" means in practice: not hiding failures, but giving each one a name, bounded cost, and a valid conversation afterwards.

# %%
def scenario(name, llm, question="q", **agent_kwargs):
    handler.lines.clear()
    agent = Agent(llm, registry, sleep=lambda s: None, **agent_kwargs)
    r = agent.run(question)
    check = Conversation()
    check.messages = r.messages
    check.validate()                                                      # the conversation must be valid even after an abort
    print(f"{name:<34} status={r.status:<22} steps={r.steps} retries={r.retries} tool_errors={sum(e for *_, e in r.tool_calls)} | {r.text[:48]!r}")
    return r, agent


ok_then_text = [([text_block("done")], None)]
scenario("tool raises an expected error", ShopAgentLLM(), "Status of order 99999999?")
scenario("model sends bad arguments", ScriptedLLM([([tool_use(1, "get_order", order_id="abc")], None), ([text_block("Sorry, let me fix that.")], None)]))
scenario("model calls an unknown tool", ScriptedLLM([([tool_use(1, "delete_order", order_id=1)], None), ([text_block("That tool does not exist.")], None)]))
scenario("answer cut off (max_tokens)", ScriptedLLM([([text_block("The return window for electro")], "max_tokens")]))
scenario("model refuses", ScriptedLLM([([text_block("I can't help with that.")], "refusal")]))
scenario("truncated TOOL CALL (not executed)", ScriptedLLM([([tool_use(1, "get_order", order_id=10)], "max_tokens")]))
r, _ = scenario("keeps failing the same way", ScriptedLLM([([tool_use(1, "get_order", order_id="x")], None)]))
r, _ = scenario("never finishes (max_steps)", ScriptedLLM([([tool_use(1, "count_orders", status="shipped")], None)]), config=AgentConfig(max_steps=3))

# %% [markdown]
# Transient API failures get retries with backoff (recorded, not slept, here), and a hard stop when they do not clear:

# %%
r, ag = scenario("API flaky then recovers", FlakyLLM(ShopAgentLLM(), fail_times=2), "How many shipped orders are there?")
print("   backoff delays (s):", [round(d, 3) for d in ag.sleeps], "| bounds with jitter: 0.25-0.5 then 0.5-1.0")
r, ag = scenario("API down (all attempts fail)", FlakyLLM(ShopAgentLLM(), fail_times=99), "How many shipped orders are there?")
print("   log:", [json.loads(l)["event"] for l in handler.lines])

# %% [markdown]
# *What just happened?* Eight endings, each with a distinct status and bounded work: the expected tool error became an answer; bad arguments and an unknown tool were rejected by the registry (not executed) and the model recovered;
# truncation and refusal were reported instead of pretended away (and a truncated tool call was **not** run); repeated identical failures stopped at three; a never-ending model stopped at `max_steps`; transient API
# failures retried with growing, jittered delays, and a persistent outage ended as `llm_unavailable`. After every one of them the conversation is still valid.
#
# ## 7. The real client (`no-run`)
#
# The swap is an adapter with the same `complete()` contract that also translates SDK exceptions into `TransientLLMError`. Below it, the same loop written directly against the SDK shows what the adapter hides.
# It needs `ANTHROPIC_API_KEY` and network access. Keep **every** content block (including `thinking` blocks) when you append the assistant turn.

# %% tags=["no-run"]
import anthropic


class AnthropicLLM:
    """Same `.complete()` contract as the mocks, backed by the Messages API."""

    def __init__(self, client=None, model="claude-sonnet-5-5"):
        self.client = client or anthropic.Anthropic(max_retries=2)              # SDK retries 429/5xx/connection errors itself; reads ANTHROPIC_API_KEY
        self.model = model

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        try:
            resp = self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system or anthropic.NOT_GIVEN,
                                               tools=tools or anthropic.NOT_GIVEN, messages=messages)
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientLLMError(str(exc)) from exc
            raise                                                               # 4xx (bad request, auth) are bugs, not retries
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]
        return LLMResponse(blocks, resp.stop_reason, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens})


real_agent = Agent(AnthropicLLM(), registry)                                    # <- the one-line swap
print(real_agent.run("Order 1000: where is it? Also, what is the return window for electronics?").text)


def run_with_sdk(client, question, model="claude-sonnet-5-5", max_steps=8):
    """The same loop with no adapter: what Agent.run does, in SDK terms."""
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        resp = client.messages.create(model=model, max_tokens=1024, tools=registry.schemas(), messages=messages)
        messages.append({"role": "assistant", "content": resp.content})           # SDK objects are accepted as-is
        if resp.stop_reason != "tool_use":
            return next((b.text for b in resp.content if b.type == "text"), ""), resp.stop_reason
        results = [registry.execute({"id": b.id, "name": b.name, "input": b.input}) for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "user", "content": results})
    return "", "max_steps"

# The SDK also ships a tool runner (client.beta.messages.tool_runner with @beta_tool) that runs this loop for you;
# use it when its hooks are enough, and write the loop yourself (as here) when you need your own stop reasons, limits and logging.

# %% [markdown]
# ## Common mistakes
#
# 1. **No step limit, or only a limit on tool calls.** A model that alternates between two tools forever never trips a per-tool cap. Bound steps *and* cost.
# 2. **Retrying everything.** Retry transient failures (429, 5xx, network). A 400 or a failed validation will fail identically every time; retrying is waste and hides a bug. Also never retry a *tool* with side effects blindly.
# 3. **Retry storms.** Fixed delays make every client retry at the same instant. Use exponential backoff **with jitter** and a cap, and let the SDK's own retries count.
# 4. **Executing a truncated tool call.** With `max_tokens` the JSON arguments may be incomplete. Check `stop_reason` before running anything.
# 5. **Dropping content blocks** (keeping only `.text`) when appending the assistant turn: breaks `tool_use` matching and `thinking` replay.
# 6. **Leaving the conversation invalid after an abort**, so the next user message 400s. Close pending calls.
# 7. **Logging full prompts and tool results by default**: personal data in your logs. Log sizes and ids; make full capture opt-in and access-controlled.

# %%
# Demonstration of mistake 6: an aborted run that does NOT close pending tool_use leaves an invalid conversation
bad = Conversation()
bad.add_user("q")
bad.add_assistant([tool_use(1, "get_order", order_id=1)])
bad.messages.append({"role": "user", "content": "follow-up question"})        # next turn appended to a conversation with an unanswered tool_use
try:
    bad.validate()
except ProtocolError as exc:
    print("ProtocolError:", exc)

# %% [markdown]
# ## Interview corner
#
# **1. What are the exit conditions of an agent loop, and what does each return?**
# <details><summary>Answer</summary>
# Normal completion (<code>end_turn</code>); truncation (<code>max_tokens</code>, never execute a possibly incomplete tool call); refusal; step limit; too many consecutive tool errors; unavailable model API after bounded retries;
# and any unexpected stop reason. Each gets a distinct status so callers and dashboards can tell them apart, and each leaves the conversation valid.
# </details>
#
# **2. How do you retry model API calls safely?**
# <details><summary>Answer</summary>
# Retry only transient errors (rate limits, overload, 5xx, network) with capped exponential backoff and jitter, bound the attempts and total time, let the SDK's built-in retries count rather than stacking a second unbounded layer,
# and surface a clear failure status afterwards. Do not retry 4xx client errors. Be careful with retries around non-idempotent tool side effects.
# </details>
#
# **3. Why generate tool schemas from function signatures?**
# <details><summary>Answer</summary>
# One source of truth: the schema cannot drift from the code, parameter types become validation for free, and adding a tool is one decorated function. Caveats: type hints express less than JSON Schema (ranges, patterns), so
# you sometimes add constraints or a hand-written schema; and the description written for humans must still be good enough for the model to choose correctly.
# </details>
#
# **4. What do you log for an agent run?**
# <details><summary>Answer</summary>
# Structured events per model call (step, stop reason, token usage, latency) and per tool call (name, arguments or a hash, error flag, result size, duration), plus a run summary (status, steps, total tokens, cost). Keep full
# prompts and results out of default logs for privacy and volume, but make them capturable under access control for debugging. Notebook 04 builds tracing on the same events.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Richer type hints for tool schemas
# `_type_schema` above handles only scalars and `Literal`. Write `type_schema(tp)` that also supports `list[T]` (array with `items`), `T | None` / `Optional[T]` (same schema as `T`; the caller treats the parameter as optional),
# `dict` (`{"type": "object"}`), and raises `TypeError` for anything else. `Literal` of ints should give `{"type": "integer", "enum": [...]}`, of strings `{"type": "string", "enum": [...]}`.
# *Hint: `typing.get_origin` / `typing.get_args`; `types.UnionType` is the origin of `int | None`.*

# %% tags=["starter"]
def type_schema(tp):
    ...


# %%
import types

assert type_schema(int) == {"type": "integer"} and type_schema(bool) == {"type": "boolean"} and type_schema(float) == {"type": "number"}
assert type_schema(list[str]) == {"type": "array", "items": {"type": "string"}}
assert type_schema(list[list[int]]) == {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}
assert type_schema(int | None) == {"type": "integer"} and type_schema(typing.Optional[str]) == {"type": "string"}
assert type_schema(Literal["a", "b"]) == {"type": "string", "enum": ["a", "b"]} and type_schema(Literal[1, 2]) == {"type": "integer", "enum": [1, 2]}
assert type_schema(dict) == {"type": "object"}
for bad_type in (complex, set[int], bytes):
    try:
        type_schema(bad_type)
    except TypeError:
        pass
    else:
        raise AssertionError(f"{bad_type} should raise TypeError")
print("Exercise 1 passed")

# %% tags=["solution"]
def type_schema(tp):
    if tp in SCALARS:
        return {"type": SCALARS[tp]}
    if tp is dict:
        return {"type": "object"}
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is list and len(args) == 1:
        return {"type": "array", "items": type_schema(args[0])}
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return type_schema(non_none[0])
    if origin is Literal:
        kinds = {type(a) for a in args}
        if kinds == {str}:
            return {"type": "string", "enum": list(args)}
        if kinds == {int}:
            return {"type": "integer", "enum": list(args)}
    raise TypeError(f"unsupported parameter type: {tp!r}")


# %% [markdown]
# ### Exercise 2: Detect a stuck agent
# Write `detect_loop(calls, repeat=3)` for a list of `(name, input_dict, is_error)` tuples like `AgentResult.tool_calls`: return `True` if the **same call** (same name and same input) appears `repeat` times **in a row** at the end of the
# list, or if the last `2 * repeat` calls alternate between exactly two distinct calls (A, B, A, B, A, B...). Otherwise `False`. Input dict order must not matter.
# *Hint: `json.dumps(inp, sort_keys=True)` gives a comparable key.*

# %% tags=["starter"]
def detect_loop(calls, repeat=3):
    ...


# %%
A = ("get_order", {"order_id": 1}, False)
B = ("count_orders", {"status": "shipped"}, False)
C = ("search_kb", {"query_text": "x"}, False)
assert detect_loop([A, A, A]) is True and detect_loop([C, A, A, A]) is True
assert detect_loop([A, A]) is False and detect_loop([A, A, B]) is False, "needs `repeat` identical calls in a row at the END"
assert detect_loop([A, B, A, B, A, B]) is True, "alternating between two calls is also a loop"
assert detect_loop([A, B, A, B, A, C]) is False and detect_loop([A, B, C, A, B, C]) is False
assert detect_loop([("t", {"a": 1, "b": 2}, False), ("t", {"b": 2, "a": 1}, True), ("t", {"a": 1, "b": 2}, False)]) is True, "key order and error flag must not matter"
assert detect_loop([A] * 5, repeat=6) is False and detect_loop([]) is False
assert detect_loop([("get_order", {"order_id": i}, False) for i in range(10)]) is False, "different arguments are progress, not a loop"
print("Exercise 2 passed")

# %% tags=["solution"]
def detect_loop(calls, repeat=3):
    keys = [(name, json.dumps(inp, sort_keys=True)) for name, inp, _ in calls]
    if len(keys) >= repeat and len(set(keys[-repeat:])) == 1:
        return True
    tail = keys[-2 * repeat:]
    return len(tail) == 2 * repeat and len(set(tail)) == 2 and all(tail[i] != tail[i + 1] for i in range(len(tail) - 1))


# %% [markdown]
# ### Exercise 3: Register a new tool and test the trajectory
# Register a new tool on the fresh `ToolRegistry` named `reg3` with the decorator: `revenue_by_status(status)` where `status` is one of `"delivered"`, `"shipped"`, `"cancelled"`, `"returned"` (use `Literal` so the schema gets an enum),
# returning `{"status": ..., "revenue": R}` where `R` is the sum of `quantity * unit_price` over orders with that status, rounded to 2 decimals (use `query`). Give it a docstring: it becomes the description the model reads.
# The check runs an `Agent` with a `ScriptedLLM` that calls your tool twice in one turn and compares the numbers with pandas over the CSVs.
# *Hint: join `orders`, `order_items`, `products`; add the decorator and the type hint to the starter.*

# %% tags=["starter"]
reg3 = ToolRegistry()


def revenue_by_status(status):
    ...


# %%
schemas3 = {s["name"]: s for s in reg3.schemas()}
assert "revenue_by_status" in schemas3, "decorate revenue_by_status with @reg3.tool"
sch = schemas3["revenue_by_status"]["input_schema"]
assert sch["properties"]["status"] == {"type": "string", "enum": ["delivered", "shipped", "cancelled", "returned"]} and sch["required"] == ["status"], sch
assert len(schemas3["revenue_by_status"]["description"]) > 10, "the docstring becomes the description: write one"
items_df, prods_df = pd.read_csv(DATA / "order_items.csv"), pd.read_csv(DATA / "products.csv")
rev = orders_df.merge(items_df, on="order_id").merge(prods_df, on="product_id").assign(v=lambda d: d.quantity * d.unit_price).groupby("status").v.sum().round(2)
script = ScriptedLLM([([tool_use(1, "revenue_by_status", status="delivered"), tool_use(2, "revenue_by_status", status="returned")], None), ([text_block("ok")], None)])
r3 = Agent(script, reg3).run("revenue?")
assert r3.status == "done" and [n for n, _, _ in r3.tool_calls] == ["revenue_by_status"] * 2 and not any(e for *_, e in r3.tool_calls)
results3 = [json.loads(b["content"]) for m in r3.messages if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"]]
assert len(results3) == 2 and all(abs(x["revenue"] - float(rev[x["status"]])) < 0.01 for x in results3), (results3, rev[["delivered", "returned"]].tolist())
bad_call = reg3.execute({"id": "z", "name": "revenue_by_status", "input": {"status": "lost"}})
assert bad_call.get("is_error") and "lost" in bad_call["content"], "invalid enum values must be rejected by the registry"
print("Exercise 3 passed")

# %% tags=["solution"]
reg3 = ToolRegistry()


@reg3.tool
def revenue_by_status(status: Literal["delivered", "shipped", "cancelled", "returned"]):
    """Total revenue (sum of quantity times unit price) of orders that have the given status."""
    total = query("select round(sum(i.quantity * p.unit_price), 2) from orders o join order_items i using(order_id) join products p using(product_id) "
                  "where o.status = ?", (status,))[0][0]
    return {"status": status, "revenue": total or 0.0}


# %% [markdown]
# ### Exercise 4 (open-ended): Extend the loop
# Pick one gap and close it, then write two sentences on how you would test it: (a) a **cost cap** (stop with status `budget_exceeded` when estimated dollars for the run exceed a limit); (b) a **wall-clock deadline** for the whole run;
# (c) **parallel tool execution** for the calls in one turn (notebook 22/02 shows how); (d) a **cancel** flag checked between steps. No assert for this one.

# %% [markdown]
# ## Recap
# - A production loop = valid-by-construction message state + a tool registry + explicit limits + named exits + bounded retries + structured events.
# - Generate tool schemas from type hints and docstrings; validate every input against them; return errors as `is_error` results, never raise into the loop.
# - Handle all stop reasons: only `tool_use` runs tools; `max_tokens` never executes a possibly truncated call; `refusal` is reported.
# - Retry only transient failures, with capped exponential backoff and jitter, injectable sleep for tests; count SDK retries.
# - Aborts must leave the conversation valid; logs record sizes and ids, not secrets.
# - The real client is an adapter with the same `complete()` contract; write the SDK loop by hand at least once.
#
# **Go deeper:** the Messages API, tool-use and error-handling pages at docs.anthropic.com; Anthropic's engineering article "Building effective agents"; the `anthropic` Python SDK README for retries and timeouts.
#
# **Next:** [A data agent with SQL and knowledge-base tools](02_a_data_agent_with_sql_and_knowledge_base_tools.ipynb)
