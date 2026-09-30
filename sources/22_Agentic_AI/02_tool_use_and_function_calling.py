# %% [markdown]
# # Tool Use and Function Calling: Schemas, the `tool_use` / `tool_result` Flow, Parallel Calls, Errors
#
# **Why this matters.** A model on its own can only produce text. Everything an agent *does* (look up an order, run a query, file a ticket) happens
# because your code exposes a function, the model asks for it, and **your code runs it**. Most real agent bugs live in this seam: a vague tool
# description so the model picks the wrong tool, an input the model got slightly wrong, a missing `tool_result` that makes the API reject the whole
# conversation, or an exception that kills the loop. This notebook makes that seam precise and testable.
#
# **You will learn**
# - describe a tool with a name, a description and a JSON Schema, and validate the model's input against it before running anything
# - follow the exact request / `tool_use` / `tool_result` message flow of the Claude Messages API, and enforce its rules in a checker
# - run several tool calls from one model turn in parallel and return all results in **one** user message
# - report tool errors so the model can recover, and repair a conversation that a crash left inconsistent
#
# **Prerequisites:** [What is an agent?](01_what_is_an_agent.ipynb) (the loop, messages, the `LLM.complete` interface).
#
# > Data used: `datasets/shop.db` (synthetic orders, opened **read-only**) and the knowledge-base documents. Every runnable cell uses a **deterministic mock
# > model**, so the loop runs offline and identically every time. The mock decides by regular expressions; it is a stand-in for the model, not a model.
# > The real Claude client is a one-line swap, shown in a `no-run` cell near the end (it needs network access and an API key).

# %%
import json
import math
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
import pandas as pd
from pydantic import BaseModel, Field

DATA = Path("../datasets")
KB = {p.stem: p.read_text(encoding="utf-8") for p in sorted((DATA / "knowledge_base").glob("*.md"))}
DB_URI = (DATA / "shop.db").resolve().as_uri() + "?mode=ro"     # read-only: a tool bug cannot modify the data


def query(sql, params=()):
    """One short-lived read-only connection per call (safe to use from several threads)."""
    con = sqlite3.connect(DB_URI, uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


print(query("select count(*) from orders")[0][0], "orders;", len(KB), "knowledge-base documents")

# %% [markdown]
# ## 1. A tool is a name, a description and a schema
#
# The model never sees your Python function. It sees a **tool definition**: three fields that you send with every request.
#
# | Field | Who reads it | Job |
# |---|---|---|
# | `name` | the model and your dispatcher | the identifier the model writes in `tool_use`; letters, digits, `_`, `-` |
# | `description` | the model | *when* to use the tool, what it returns, what it does **not** do (this is the most important field) |
# | `input_schema` | the model **and your validator** | a JSON Schema object describing the arguments |
#
# The schema does two jobs. It tells the model which arguments exist and what values are legal (`enum`, `minimum`, per-property `description`), and
# it lets **you** check what came back. Models follow schemas well but not perfectly, so treat `tool_use.input` as untrusted input: validate it.

# %%
GET_ORDER = {
    "name": "get_order",
    "description": "Look up ONE order by its numeric id. Returns status, date, number of items and total value as JSON. "
                   "Use for questions about a specific order; do not use for counts or policies.",
    "input_schema": {
        "type": "object",
        "properties": {"order_id": {"type": "integer", "description": "The numeric order id, e.g. 1000 (digits only, no '#')."}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
}
SEARCH_KB = {
    "name": "search_kb",
    "description": "Search the company policy knowledge base (returns, shipping, warranty, loyalty points, payment...). Returns the best matching line "
                   "and its document name. Use for policy questions, never for order data.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A short natural-language question about a policy."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}
LIST_ORDERS = {
    "name": "list_orders",
    "description": "List up to `limit` order ids that have a given status, newest first.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["delivered", "shipped", "cancelled", "returned"], "description": "Order status."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "How many ids to return (1-20)."},
        },
        "required": ["status"],
        "additionalProperties": False,
    },
}
validator = jsonschema.Draft202012Validator(GET_ORDER["input_schema"])
for candidate in [{"order_id": 1000}, {"order_id": "1000"}, {"order_id": 1000, "extra": 1}, {}]:
    errors = [e.message for e in validator.iter_errors(candidate)]
    print(f"{json.dumps(candidate):<32} -> {'valid' if not errors else errors[0]}")

# %% [markdown]
# *What just happened?* The schema rejects a string id, an unknown extra field and a missing field, with messages specific enough to send **back to the model**
# (section 5). You rarely write large schemas by hand: a Pydantic model can generate one, and the same model then validates the input. Note the extra
# `title` keys Pydantic adds; they are harmless JSON Schema but noise the model does not need.

# %%
class ListOrdersInput(BaseModel):
    status: str = Field(pattern="^(delivered|shipped|cancelled|returned)$", description="Order status.")
    limit: int = Field(default=5, ge=1, le=20, description="How many ids to return.")


print(json.dumps(ListOrdersInput.model_json_schema(), indent=1)[:420], "...")

# %% [markdown]
# ## 2. The tools themselves
#
# Plain Python functions that return **a string** (the `content` of a `tool_result`). A function signals failure by raising an exception; the dispatcher
# in section 4 turns it into an error result. Keep outputs small: they are re-sent to the model on every later step.

# %%
def get_order(order_id: int) -> str:
    rows = query(
        "select o.order_id, o.status, o.order_date, sum(i.quantity), round(sum(i.quantity * p.unit_price), 2) "
        "from orders o join order_items i using(order_id) join products p using(product_id) where o.order_id = ? group by o.order_id",
        (order_id,),
    )
    if not rows:
        raise LookupError(f"no order with id {order_id}. Order ids in this shop are positive integers.")
    oid, status, date, items, total = rows[0]
    return json.dumps({"order_id": oid, "status": status, "date": date, "items": items, "total": total})


STOP = set("a an the i my me to of is it in on for do you can how what when was be and or with after have this that are does will much many".split())
words = lambda t: {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP}


def search_kb(query: str) -> str:
    best = max(((len(words(query) & words(line)), name, line) for name, text in KB.items() for line in text.splitlines()[1:]), key=lambda x: x[0])
    return json.dumps({"document": best[1], "text": best[2]}) if best[0] else json.dumps({"document": None, "text": "no relevant passage found"})


def list_orders(status: str, limit: int = 5) -> str:
    rows = query("select order_id from orders where status = ? order by order_date desc, order_id desc limit ?", (status, limit))
    return json.dumps({"status": status, "order_ids": [r[0] for r in rows]})


REGISTRY = {"get_order": (GET_ORDER, get_order), "search_kb": (SEARCH_KB, search_kb), "list_orders": (LIST_ORDERS, list_orders)}
print(get_order(1000))
print(search_kb("How long do I have to return electronics?"))
print(list_orders("returned", 3))

# %% [markdown]
# ## 3. The wire protocol, step by step
#
# This is the exact message flow of the Claude Messages API (`POST /v1/messages`). The API is **stateless**: every request carries the whole conversation.
#
# 1. **Request.** You send `model`, `max_tokens`, `tools` (the definitions above), and `messages` (starting with the user's question).
# 2. **Response.** The model returns `content`, a list of blocks. Text blocks are commentary; a **`tool_use`** block is a request:
#    `{"type": "tool_use", "id": "toolu_...", "name": "get_order", "input": {"order_id": 1000}}`. The `id` is generated by the model side and is how you
#    match the answer to the request. When the model stops to wait for tools, `stop_reason` is `"tool_use"`.
# 3. **You run the tool.** The model cannot. Nothing happened until your code executes the function.
# 4. **Next request.** You append *two* messages to the history: the **assistant** message containing the response content **unchanged** (every block,
#    including any `thinking` blocks), then a **user** message whose content is a list of **`tool_result`** blocks:
#    `{"type": "tool_result", "tool_use_id": "toolu_...", "content": "...", "is_error": false}`.
# 5. **Repeat** until `stop_reason` is `"end_turn"` (a plain answer), or you hit your own step limit.
#
# The rules the API enforces (from Anthropic's tool-use documentation; a violation returns an HTTP 400 error and **the whole request fails**):
# - every `tool_use` id needs exactly one matching `tool_result`, in the **very next** user message;
# - when the model made several calls in one turn (**parallel tool use**), all the `tool_result` blocks go in **one** user message, not one message each;
# - inside that user message, `tool_result` blocks come **first**; any extra text goes after them;
# - a `tool_result` without a preceding `tool_use` is invalid.
#
# Because these rules are mechanical, you can check them in code. `validate_messages` below mirrors them. It is **our** approximation of the documented
# rules (the real API is the authority), but it lets every mock in this course fail the same way the real service would.

# %%
class ProtocolError(ValueError):
    """What the real API would report as HTTP 400 invalid_request_error."""


def blocks_of(message):
    c = message["content"]
    return [{"type": "text", "text": c}] if isinstance(c, str) else c


def validate_messages(messages):
    if not messages or messages[0]["role"] != "user":
        raise ProtocolError("conversation must start with a user message")
    pending = None
    for i, m in enumerate(messages):
        blocks = blocks_of(m)
        is_result = [b["type"] == "tool_result" for b in blocks]
        if pending is not None:
            if m["role"] != "user" or not any(is_result):
                raise ProtocolError(f"message {i}: tool_use id(s) {sorted(pending)} must be answered by tool_result blocks in the next user message")
            if any(is_result[k] for k in range(len(blocks)) if not all(is_result[:k])):
                raise ProtocolError(f"message {i}: tool_result blocks must come before any other content")
            ids = [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"]
            if set(ids) != pending or len(ids) != len(pending):
                raise ProtocolError(f"message {i}: tool_result ids {sorted(ids)} do not match the tool_use ids {sorted(pending)}")
            pending = None
        elif any(is_result):
            raise ProtocolError(f"message {i}: tool_result without a tool_use in the previous assistant message")
        if m["role"] == "assistant":
            ids = {b["id"] for b in blocks if b["type"] == "tool_use"}
            pending = ids or None
    if messages[-1]["role"] != "user":
        raise ProtocolError("a request must end with a user message")
    return True


uses = [{"type": "tool_use", "id": "t1", "name": "get_order", "input": {"order_id": 1000}},
        {"type": "tool_use", "id": "t2", "name": "get_order", "input": {"order_id": 1001}}]
ok_history = [{"role": "user", "content": "Compare orders 1000 and 1001"}, {"role": "assistant", "content": uses},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}, {"type": "tool_result", "tool_use_id": "t2", "content": "{}"}]}]
print("valid history:", validate_messages(ok_history))

split = ok_history[:2] + [{"role": "user", "content": [ok_history[2]["content"][0]]}, {"role": "user", "content": [ok_history[2]["content"][1]]}]
missing = ok_history[:2] + [{"role": "user", "content": [ok_history[2]["content"][0]]}]
wrong_id = ok_history[:2] + [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}, {"type": "tool_result", "tool_use_id": "zzz", "content": "{}"}]}]
text_first = ok_history[:2] + [{"role": "user", "content": [{"type": "text", "text": "thanks"}] + ok_history[2]["content"]}]
for name, bad in [("results split across messages", split), ("one result missing", missing), ("wrong id", wrong_id), ("text before results", text_first)]:
    try:
        validate_messages(bad)
    except ProtocolError as exc:
        print(f"{name:<30} -> ProtocolError: {str(exc)[:88]}")

# %% [markdown]
# *What just happened?* Four different mistakes, four rejections. In production these show up as a 400 from the API in the middle of a run, usually after a tool
# crashed or a retry duplicated a message. Notebook 23/01 builds a loop that cannot produce them by construction; exercise 3 here repairs a history that already has one.

# %% [markdown]
# ## 4. A mock model that speaks the protocol
#
# The mock below follows the same contract as the real client wrapper: `complete(messages, system, tools, max_tokens) -> LLMResponse` with `content` blocks, a
# `stop_reason` and token `usage`. It validates its input like the API would, asks for **all** the lookups a question needs **in one turn** (parallel calls),
# reads the results, and writes an answer. `sloppy=True` makes it behave like a model that sometimes formats an argument wrongly (`"#1000"` instead of `1000`),
# so we can exercise error recovery deterministically.

# %%
def est_tokens(x) -> int:
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


def make_response(blocks, messages, system=""):
    tin = est_tokens(system) + sum(est_tokens(m["content"]) for m in messages)
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return LLMResponse(blocks, stop, {"input_tokens": tin, "output_tokens": sum(est_tokens(b) for b in blocks)})


text_block = lambda t: {"type": "text", "text": t}
POLICY_WORDS = r"return|refund|cancel|shipping|warranty|points|premium|payment|privacy|support|delivery|window"


class ShopToolLLM:
    def __init__(self, sloppy=False):
        self.sloppy = sloppy

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        validate_messages(messages)                                   # the real API would return HTTP 400 here
        question = messages[0]["content"]
        uses, outcomes = {}, []
        for m in messages[1:]:
            for b in blocks_of(m):
                if b["type"] == "tool_use":
                    uses[b["id"]] = b
                elif b["type"] == "tool_result":
                    outcomes.append((uses[b["tool_use_id"]], b.get("is_error", False), b["content"]))
        ids = [int(x) for x in re.findall(r"\b\d{3,}\b", question)]
        nxt = len(uses)

        def call(name, args):
            nonlocal nxt
            nxt += 1
            return {"type": "tool_use", "id": f"toolu_{nxt}", "name": name, "input": args}

        if not outcomes:                                              # first turn: request everything needed, in parallel
            blocks = [text_block("I will look this up.")]
            blocks += [call("get_order", {"order_id": f"#{i}" if self.sloppy else i}) for i in ids]
            if re.search(POLICY_WORDS, question, re.I):
                blocks.append(call("search_kb", {"query": question}))
            if len(blocks) > 1:
                return make_response(blocks, messages, system)
            return make_response([text_block("I can help with order status and shop policies.")], messages, system)
        failed = [u["input"]["order_id"] for u, err, content in outcomes
                  if err and u["name"] == "get_order" and isinstance(u["input"]["order_id"], str) and "integer" in content]
        tried = {u["input"]["order_id"] for u, _, _ in outcomes if u["name"] == "get_order" and isinstance(u["input"]["order_id"], int)}
        retry_ids = [int(f.lstrip("#")) for f in failed if int(f.lstrip("#")) not in tried]
        if retry_ids:                                                 # recover from a validation error, once per id
            return make_response([text_block("The id must be a plain integer; retrying.")] + [call("get_order", {"order_id": i}) for i in retry_ids], messages, system)
        parts = []
        for use, err, content in outcomes:
            if use["name"] == "get_order" and isinstance(use["input"]["order_id"], str) and err:
                continue                                              # superseded by the retry
            if err:
                parts.append(f"Sorry, I could not complete {use['name']}: {content}")
            elif use["name"] == "get_order":
                o = json.loads(content)
                parts.append(f"Order {o['order_id']} is {o['status']} ({o['items']} item(s), total {o['total']}).")
            elif use["name"] == "search_kb":
                parts.append(f"Policy: {json.loads(content)['text']}")
        return make_response([text_block(" ".join(parts))], messages, system)


first = ShopToolLLM().complete([{"role": "user", "content": "Compare orders 1000 and 1001 and tell me the return window."}], tools=[GET_ORDER, SEARCH_KB])
print("stop_reason:", first.stop_reason, "| blocks:", [b["type"] for b in first.content])
print(json.dumps(first.tool_calls[0]), "\n", json.dumps(first.tool_calls[2]))

# %% [markdown]
# ## 5. The dispatcher: validate, run, never crash
#
# `run_tool` takes one `tool_use` block and returns one `tool_result` block. It is the most important 20 lines in an agent because it is where untrusted
# model output meets your code. In order:
# 1. **unknown tool name** -> error result listing the valid names (the model can correct itself);
# 2. **schema validation** -> error result with the validator message;
# 3. **run the function** inside `try/except` -> an exception becomes an error result, never a crash;
# 4. **truncate** long output with a visible note (a silent cut would let the model assume it has everything).
#
# Setting `"is_error": true` tells the model the call failed; the `content` should say *what to change*. That is how the model recovers instead of repeating
# the same mistake.

# %%
MAX_RESULT_CHARS = 2000


def run_tool(registry, call):
    name, args = call["name"], call["input"]

    def result(text, is_error=False):
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"... [truncated, {len(text) - MAX_RESULT_CHARS} more characters]"
        block = {"type": "tool_result", "tool_use_id": call["id"], "content": text}
        if is_error:
            block["is_error"] = True
        return block

    if name not in registry:
        return result(f"Unknown tool {name!r}. Available tools: {sorted(registry)}", True)
    schema, fn = registry[name]
    problems = [f"{'/'.join(map(str, e.path)) or 'input'}: {e.message}" for e in jsonschema.Draft202012Validator(schema["input_schema"]).iter_errors(args)]
    if problems:
        return result("Invalid input. " + "; ".join(problems), True)
    try:
        return result(fn(**args))
    except Exception as exc:  # noqa: BLE001  (a tool must never crash the loop)
        return result(f"{type(exc).__name__}: {exc}", True)


for c in [{"id": "a", "name": "get_order", "input": {"order_id": 1000}}, {"id": "b", "name": "get_order", "input": {"order_id": "#1000"}},
          {"id": "c", "name": "get_order", "input": {"order_id": 999999}}, {"id": "d", "name": "delete_order", "input": {}}]:
    r = run_tool(REGISTRY, c)
    print(f"{c['name']}({json.dumps(c['input'])})".ljust(36), "ERROR" if r.get("is_error") else "ok   ", r["content"][:84])

# %% [markdown]
# Next, the function that turns a whole model turn into the single user message the API wants. The simple version runs calls one after another. Exercise 2 makes it parallel.

# %%
def run_calls_sequential(registry, calls):
    return {"role": "user", "content": [run_tool(registry, c) for c in calls]}


def run_agent(llm, registry, question, run_calls=run_calls_sequential, max_steps=6):
    schemas = [schema for schema, _ in registry.values()]
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        resp = llm.complete(messages, tools=schemas)
        messages.append({"role": "assistant", "content": resp.content})     # the whole content, unchanged
        if resp.stop_reason != "tool_use":
            return resp.text, messages, "done"
        messages.append(run_calls(registry, resp.tool_calls))
    return "", messages, "max_steps"


def show(messages):
    for m in messages:
        for b in blocks_of(m):
            if b["type"] == "text":
                print(f"{m['role']:>9}: {b['text'][:100]}")
            elif b["type"] == "tool_use":
                print(f"{m['role']:>9}: TOOL_USE    {b['name']}({json.dumps(b['input'])}) id={b['id']}")
            elif b["type"] == "tool_result":
                print(f"{m['role']:>9}: TOOL_RESULT id={b['tool_use_id']}{' ERROR' if b.get('is_error') else ''} -> {b['content'][:62]}")


final, msgs, status = run_agent(ShopToolLLM(), REGISTRY, "Compare orders 1000 and 1001, and what is the return window for electronics?")
show(msgs)
print(status)

# %% [markdown]
# *What just happened?* Two model calls. The first turn contained **three** `tool_use` blocks (two order lookups and one policy search); the next user message contained
# **three** `tool_result` blocks, in one message. The model did not need to wait for one lookup before asking for the next, which is the whole point of parallel tool use:
# fewer round trips, lower latency. Independent calls can run concurrently in your code; dependent calls (you need result A to decide call B) cannot be in the same turn.
#
# Let's verify the answer against an independent path (pandas over the CSV files), and then watch the error-recovery path with the "sloppy" model.

# %%
orders_df, items_df, products_df = (pd.read_csv(DATA / f"{n}.csv") for n in ("orders", "order_items", "products"))
line_value = items_df.merge(products_df, on="product_id").assign(v=lambda d: d.quantity * d.unit_price)
expected_total = round(line_value[line_value.order_id == 1000].v.sum(), 2)
print("pandas total for order 1000:", expected_total, "| in the agent's answer:", f"total {expected_total})" in final)

final2, msgs2, status2 = run_agent(ShopToolLLM(sloppy=True), REGISTRY, "What is the status of order 1000?")
show(msgs2)
print(status2, "|", final2)

# %% [markdown]
# *What just happened?* The first call carried `"#1000"`, a string. Schema validation rejected it **before** the SQL function ran, the error text said `is not of type 'integer'`, and the
# mock retried with `1000` in the next turn. A real model recovers the same way when the error message is specific, which is why `is_error` results should say what to change,
# not just "failed".

# %% [markdown]
# ## 6. Parallel tool calls and their cost
#
# Running independent calls concurrently in *your* process matters when tools are slow (HTTP calls, big queries). Here each call waits 0.3 s; run one after another, three calls take about 0.9 s.
# Exercise 2 runs them concurrently (about 0.3 s). (The sleep stands in for network latency; real timings vary.)

# %%
def slow_echo(text: str) -> str:
    time.sleep(0.3)
    return text


SLOW = {"slow_echo": ({"name": "slow_echo", "description": "Echo after a delay.", "input_schema": {
    "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}, slow_echo)}
three_calls = [{"id": f"t{i}", "name": "slow_echo", "input": {"text": f"n{i}"}} for i in range(3)]
t0 = time.perf_counter()
run_calls_sequential(SLOW, three_calls)
print(f"sequential: {time.perf_counter() - t0:.2f}s")

# %% [markdown]
# ## 7. Choosing what the model may do: `tool_choice`
#
# The request's `tool_choice` field steers tool use. The options, as documented by Anthropic: `auto` (the model decides; the default), `any` (must call some tool),
# `tool` (must call a named tool), `none` (no tools this turn). Two practical notes. First, forced choice (`any` / `tool`) is **rejected with a 400 by the newest model
# generations** (for example Claude Fable 5.1, Opus 5.5 and Sonnet 5.5 at the time of writing, per the model documentation), so prefer `auto` plus an explicit instruction in the system prompt
# ("for order questions, call get_order first"), and use `strict: true` on the tool when you need guaranteed schema-valid arguments. Check the current docs for your model, because this changes
# between generations. Second, `tool_choice` only controls *whether* a tool is called; your validator still decides whether the arguments are acceptable.

# %% [markdown]
# ## 8. The same loop against the real API (`no-run`)
#
# Swap `ShopToolLLM` for a thin wrapper around the SDK. The wrapper must keep **all** content blocks (including `thinking` blocks if the model produced any), because the API
# requires you to send the assistant turn back unchanged. Needs `ANTHROPIC_API_KEY` in the environment and network access, so this cell is not executed here.

# %% tags=["no-run"]
import anthropic


class AnthropicLLM:
    """Same `.complete()` shape as the mock, backed by the real Messages API."""

    def __init__(self, client=None, model="claude-sonnet-5-5"):
        self.client, self.model = client or anthropic.Anthropic(), model        # reads ANTHROPIC_API_KEY from the environment

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]          # keep EVERY block, unchanged
        return LLMResponse(blocks, resp.stop_reason, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens})


final, msgs, status = run_agent(AnthropicLLM(), REGISTRY, "Compare orders 1000 and 1001, and what is the return window for electronics?")
# The SDK also has a tool runner (client.beta.messages.tool_runner with @beta_tool) that runs this loop for you;
# writing it by hand once, as above, is how you learn what it does. Always handle stop_reason == "refusal" and "max_tokens" too.

# %% [markdown]
# ## Common mistakes
#
# 1. **Sending the tool results as separate messages** (or one per call). The API rejects missing ids, and it is documented that splitting parallel results discourages the model from calling tools in parallel.
# 2. **Dropping or editing the assistant message** before appending results (for example keeping only its text). The `tool_use` blocks must be there for the ids to match, and `thinking` blocks must be passed back unchanged.
# 3. **Trusting `input` without validation.** The model can send a string for an integer, an extra key, or a value outside your enum.
# 4. **Letting a tool raise.** One exception ends the loop and leaves a dangling `tool_use` in the history. Convert to an error result.
# 5. **Vague descriptions.** Two tools described as "gets data" and "fetches information" will be confused. Say when to use each and when not to.
# 6. **Returning everything.** A 5,000-row table as a tool result is paid for on every later step. Return the answer, with a limit and a truncation note.
# 7. **Assuming the model always calls the tool you expected** (or calls one at all). Test trajectories, not only final answers (notebook 04).

# %%
# Demonstration of mistake 2: keeping only the text of the assistant turn breaks the conversation
bad_history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [text_block("I will look this up.")]},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"}]}]
try:
    validate_messages(bad_history)
except ProtocolError as exc:
    print("ProtocolError:", exc)

# %% [markdown]
# ## Interview corner
#
# **1. Walk me through what happens, message by message, when an LLM "calls a function".**
# <details><summary>Answer</summary>
# The model does not execute anything. The response contains a <code>tool_use</code> block (id, name, input) and <code>stop_reason: "tool_use"</code>. Your code runs the function,
# then sends the next request with the assistant message unchanged followed by a user message of <code>tool_result</code> blocks whose <code>tool_use_id</code> values match. The model
# continues from there; the loop ends on a plain answer or when your step limit triggers.
# </details>
#
# **2. The model issued three tool calls in one turn. How do you return the results?**
# <details><summary>Answer</summary>
# In a single user message containing three <code>tool_result</code> blocks (first in the content list), each with the matching <code>tool_use_id</code>. Run independent calls concurrently, keep
# results attached to the right ids regardless of finish order, and isolate failures so one error becomes one <code>is_error</code> result rather than aborting the turn.
# </details>
#
# **3. How do you handle a tool that fails?**
# <details><summary>Answer</summary>
# Catch the exception and return a <code>tool_result</code> with <code>is_error: true</code> and a message that says what went wrong and what to change (valid values, what to try instead). Validate input
# against the schema before running anything, cap output size, and limit retries overall with a step budget so a model that keeps failing cannot loop forever.
# </details>
#
# **4. Why validate tool input if the model has the schema?**
# <details><summary>Answer</summary>
# Schemas guide the model but do not make it infallible (and <code>strict: true</code> is an option for guaranteed-valid arguments, not a substitute for authorisation). Input is untrusted: it may also be
# influenced by text the model read (prompt injection). Validation catches type errors; allow-lists, read-only connections and permission checks stop harmful but well-formed calls.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Write a tool schema
# Define `TOP_CUSTOMERS`, a tool definition dict for `top_customers(country, limit)`: `country` is a required string of **exactly two uppercase letters** (use `pattern`), `limit` is an
# optional integer between 1 and 10. No extra properties allowed. Give the tool and each property a `description`.
# *Hint: `input_schema` is a JSON Schema object with `type`, `properties`, `required`, `additionalProperties`.*

# %% tags=["starter"]
TOP_CUSTOMERS = {
    "name": "top_customers",
    "description": ...,
    "input_schema": ...,
}

# %%
assert isinstance(TOP_CUSTOMERS["description"], str) and len(TOP_CUSTOMERS["description"]) > 15, "write a real description"
sch = TOP_CUSTOMERS["input_schema"]
jsonschema.Draft202012Validator.check_schema(sch)
assert all(sch["properties"][k].get("description") for k in ("country", "limit")), "describe each property"
v = jsonschema.Draft202012Validator(sch)
good = [{"country": "DE"}, {"country": "US", "limit": 10}, {"country": "IN", "limit": 1}]
bad = [{}, {"country": "de"}, {"country": "DEU"}, {"country": "DE", "limit": 0}, {"country": "DE", "limit": 11}, {"country": "DE", "limit": "3"}, {"country": "DE", "x": 1}, {"limit": 3}]
assert all(v.is_valid(g) for g in good), "these inputs must be valid"
assert not any(v.is_valid(b) for b in bad), f"these must be rejected: {[b for b in bad if v.is_valid(b)]}"
print("Exercise 1 passed")

# %% tags=["solution"]
TOP_CUSTOMERS = {
    "name": "top_customers",
    "description": "Return the customers of one country with the most orders, most orders first. Use for ranking questions about customers, not about a single order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "country": {"type": "string", "pattern": "^[A-Z]{2}$", "description": "Two-letter uppercase country code, e.g. DE."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "description": "How many customers to return (1-10)."},
        },
        "required": ["country"],
        "additionalProperties": False,
    },
}


# %% [markdown]
# ### Exercise 2: Run a turn's tool calls in parallel
# Write `run_calls_parallel(registry, calls)` returning **one** user message whose `content` holds one `tool_result` per call, **in the same order as `calls`**, produced by running the calls
# concurrently (use `ThreadPoolExecutor` and reuse `run_tool`). One failing call must not affect the others. The check times three 0.3 s calls and requires the whole turn to take under 0.75 s.
# *Hint: `ex.map(fn, items)` returns results in input order.*

# %% tags=["starter"]
def run_calls_parallel(registry, calls):
    ...


# %%
mixed = three_calls + [{"id": "bad", "name": "slow_echo", "input": {"text": 5}}, {"id": "nope", "name": "missing", "input": {}}]
t0 = time.perf_counter()
msg = run_calls_parallel(SLOW, mixed)
elapsed = time.perf_counter() - t0
assert msg["role"] == "user" and isinstance(msg["content"], list), "return ONE user message with a list of blocks"
assert [b["tool_use_id"] for b in msg["content"]] == ["t0", "t1", "t2", "bad", "nope"], "results must keep the order of the calls"
assert [b["content"] for b in msg["content"][:3]] == ["n0", "n1", "n2"], "each result must belong to its own call"
assert msg["content"][3].get("is_error") and msg["content"][4].get("is_error") and not msg["content"][0].get("is_error"), "failures isolated as is_error results"
assert elapsed < 0.75, f"calls must run concurrently (took {elapsed:.2f}s; sequential would be about 0.9s)"
ok_turn = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "tool_use", **{k: c[k] for k in ("id", "name", "input")}} for c in mixed]}, msg]
assert validate_messages(ok_turn), "the message must satisfy the protocol"
print(f"Exercise 2 passed ({elapsed:.2f}s for {len(mixed)} calls)")

# %% tags=["solution"]
def run_calls_parallel(registry, calls):
    with ThreadPoolExecutor(max_workers=max(1, len(calls))) as ex:
        results = list(ex.map(lambda c: run_tool(registry, c), calls))
    return {"role": "user", "content": results}


# %% [markdown]
# ### Exercise 3: Repair a broken history
# A worker crashed after the model asked for tools but before the results were appended, and the retry code wants to continue the conversation. Write `repair_history(messages)` that returns a **new**
# list in which every `tool_use` that has no matching `tool_result` gets one: an `is_error` result with content `"Tool call was interrupted; no result is available."`. Results must go in the user message
# right after the assistant message (merged into an existing results message if there is one, placed first). The input list must not be modified. Valid histories are returned unchanged.
# *Hint: walk the messages; after each assistant message collect its `tool_use` ids, compare with the ids in the next message.*

# %% tags=["starter"]
def repair_history(messages):
    ...


# %%
two = [{"type": "tool_use", "id": "t1", "name": "get_order", "input": {"order_id": 1}}, {"type": "tool_use", "id": "t2", "name": "get_order", "input": {"order_id": 2}}]
crashed_end = [{"role": "user", "content": "q"}, {"role": "assistant", "content": two}]
half_done = crashed_end + [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]}]
snapshot = json.dumps(half_done)
for broken in (crashed_end, half_done):
    before = json.dumps(broken)
    fixed = repair_history(broken)
    assert json.dumps(broken) == before, "do not modify the input"
    assert validate_messages(fixed), "repaired history must satisfy the protocol"
    results = {b["tool_use_id"]: b for b in fixed[2]["content"]}
    assert set(results) == {"t1", "t2"} and results["t2"]["is_error"] is True, "t2 needs a synthetic error result"
assert repair_history(half_done)[2]["content"][0]["content"] == "{}" or repair_history(half_done)[2]["content"][1]["content"] == "{}", "the real t1 result must be kept"
assert repair_history(ok_history) == ok_history, "a valid history is returned unchanged"
print("Exercise 3 passed")

# %% tags=["solution"]
def repair_history(messages):
    fixed = [dict(m) for m in messages]
    i = 0
    while i < len(fixed):
        m = fixed[i]
        ids = [b["id"] for b in blocks_of(m) if b["type"] == "tool_use"] if m["role"] == "assistant" else []
        if ids:
            nxt = fixed[i + 1] if i + 1 < len(fixed) else None
            have = [b for b in blocks_of(nxt) if b["type"] == "tool_result"] if nxt and nxt["role"] == "user" else []
            missing = [t for t in ids if t not in {b["tool_use_id"] for b in have}]
            if missing:
                synthetic = [{"type": "tool_result", "tool_use_id": t, "is_error": True, "content": "Tool call was interrupted; no result is available."} for t in missing]
                if have:
                    rest = [b for b in blocks_of(nxt) if b["type"] != "tool_result"]
                    fixed[i + 1] = {"role": "user", "content": have + synthetic + rest}
                else:
                    fixed.insert(i + 1, {"role": "user", "content": synthetic})
        i += 1
    return fixed


# %% [markdown]
# ### Exercise 4 (open-ended): Design the tools for your own agent
# Pick a real system you work with (a CI server, a ticket queue, an internal API). Write three tool definitions for an agent that answers questions about it: for each give the `name`, a
# description that says *when to use it and when not to*, and the schema. Then list two inputs the model could plausibly send that are well-formed but harmful or wasteful (for example an
# unbounded date range), and say where in your dispatcher you would stop them. There is no assert for this one.

# %% [markdown]
# ## Recap
# - A tool is `name` + `description` + JSON-Schema `input_schema`; the description steers the model, the schema lets you validate what comes back.
# - Flow: response with `tool_use` blocks (`stop_reason: "tool_use"`) -> you run them -> next request has the assistant content unchanged plus one user message of matching `tool_result` blocks.
# - The API rejects mismatched ids, missing results, results split across messages, and results placed after other content; a checker like `validate_messages` catches this in tests.
# - Parallel calls: one assistant turn, several `tool_use` blocks, results returned together; run independent calls concurrently and keep order and failures isolated.
# - Errors become `is_error` results with actionable text; validate input, cap output, and bound retries with a step limit.
#
# **Go deeper:** the tool-use section of the Anthropic API documentation (docs.anthropic.com), including the pages on defining tools, handling tool calls and parallel tool use; JSON Schema reference at json-schema.org.
#
# **Next:** [Agent patterns: planning, reflection and multi-agent](03_agent_patterns_planning_reflection_and_multi_agent.ipynb)
