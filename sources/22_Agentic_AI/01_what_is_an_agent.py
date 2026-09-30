# %% [markdown]
# # What Is an Agent? Workflows, the Loop, Tools, Memory
#
# **Why this matters.** "Agent" is the most overloaded word in AI engineering. Some teams call a fixed three-step prompt pipeline an agent;
# others hand a model a shell and hope. The distinction is not academic: it decides your cost, latency, reliability and how you test the
# system. This notebook gives you a precise definition, builds the smallest possible agent loop, and, just as important, teaches you to recognise
# when you should **not** build one.
#
# **You will learn**
# - tell a **workflow** (code decides the steps) from an **agent** (the model decides the steps) and choose between them
# - implement the agent loop: model call, tool call, observation, repeat, stop
# - read a transcript as a **trajectory** of messages, `tool_use` and `tool_result` blocks
# - reason about cost growth and compounding unreliability of multi-step loops with arithmetic, not vibes
#
# **Prerequisites:** [How LLMs work](../21_LLMs_and_RAG/01_how_llms_work.ipynb) and
# [Prompting and structured outputs](../21_LLMs_and_RAG/02_prompting_and_structured_outputs.ipynb) (the `LLM.complete` interface and scripted mocks).
#
# > Data used: `datasets/shop.db` (synthetic orders, read-only) and the knowledge-base documents. The "model" in every runnable cell is a
# > **deterministic mock**. It decides which tool to call by simple rules, so you can study the loop offline; it is not intelligent.
# > Real Claude calls are shown in no-run cells later in this module.

# %%
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path("../datasets")
KB = {p.stem: p.read_text(encoding="utf-8") for p in sorted((DATA / "knowledge_base").glob("*.md"))}
DB_URI = (DATA / "shop.db").resolve().as_uri() + "?mode=ro"          # read-only connection: the agent cannot modify data
db = sqlite3.connect(DB_URI, uri=True)
print(db.execute("select count(*) from orders").fetchone()[0], "orders;", len(KB), "knowledge-base documents")

# %% [markdown]
# ## 1. Workflow versus agent
#
# A useful working definition (the same split used in Anthropic's engineering guidance on building agents):
#
# - A **workflow** is a system where *your code* orchestrates the steps, with LLM calls as components: "classify the ticket, then look up the order,
#   then fill this template". The path is fixed in advance.
# - An **agent** is a system where *the model* decides, at each step, what to do next and which tools to call, looping until it decides it is done.
#   The path is chosen at run time.
#
# Autonomy is a dial, not a switch, and it has a price. Every step you hand to the model buys flexibility (it can handle a request you did not
# anticipate) and costs predictability (you cannot list the paths in advance, so you cannot test them all).
#
# | | Workflow | Agent |
# |---|---|---|
# | Who picks the next step | your code | the model |
# | Steps known in advance | yes | no |
# | Latency and cost | bounded, predictable | variable (number of loops) |
# | Testing | test each branch | test trajectories and outcomes |
# | Fails by | doing the wrong fixed thing | wandering, looping, or acting wrongly |
# | Best for | well-understood tasks | open-ended tasks where steps depend on what is found |

# %% [markdown]
# ## 2. The two tools our examples use
#
# An agent acts through **tools**: ordinary functions with a name, a description and an input schema that the model can request. Ours read real data:
# an order lookup (SQL against `shop.db`) and a knowledge-base search (word overlap). Each returns a **string** (that is what goes back into the conversation).

# %%
def get_order(order_id: int) -> str:
    """Look up an order: status, date, number of items and total value."""
    row = db.execute(
        "select o.order_id, o.status, o.order_date, sum(i.quantity), round(sum(i.quantity * p.unit_price), 2) "
        "from orders o join order_items i using(order_id) join products p using(product_id) where o.order_id = ? group by o.order_id",
        (int(order_id),),
    ).fetchone()
    if row is None:
        return json.dumps({"error": f"no order with id {order_id}"})
    return json.dumps({"order_id": row[0], "status": row[1], "date": row[2], "items": row[3], "total": row[4]})


STOP = set("a an the i my me to of is it in on for do you can how what when was be and or with after have this that are does will much many".split())
words = lambda t: {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP}


def search_kb(query: str) -> str:
    """Search the company knowledge base and return the best matching line with its document name."""
    best = max(((len(words(query) & words(line)), name, line) for name, text in KB.items() for line in text.splitlines()[1:]), key=lambda x: x[0])
    return json.dumps({"document": best[1], "text": best[2]}) if best[0] else json.dumps({"document": None, "text": "no relevant passage found"})


print(get_order(1000))
print(search_kb("Can I cancel an order?"))
print(get_order(999999))

# %% [markdown]
# Note that the **error case returns a message instead of raising**. Whatever a tool returns is what the model reads next, so a clear error text lets the
# model recover (try another id, ask the user); an uncaught exception would end your loop. We formalise that in the next notebook.

# %% [markdown]
# ## 3. The messages an agent exchanges
#
# We use the same message shapes as the Claude Messages API, so what you learn transfers directly:
#
# - the **assistant** replies with a list of content blocks: `text` and/or `tool_use` (`{"type": "tool_use", "id": ..., "name": ..., "input": {...}}`);
# - when it wants a tool, `stop_reason` is `"tool_use"` and *you* run the tool;
# - you reply with a **user** message containing a `tool_result` block that carries the same `tool_use_id`;
# - the model then continues, possibly asking for another tool, until it replies with plain text and `stop_reason` `"end_turn"`.

# %%
def est_tokens(text) -> int:
    text = text if isinstance(text, str) else json.dumps(text)
    return max(1, math.ceil(len(text) / 4))


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


class ScriptedLLM:
    """Replays prepared blocks, one list per model call. Records what it was sent."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        self.calls.append({"messages": json.loads(json.dumps(messages)), "system": system})
        if len(self.calls) > len(self.script):
            raise RuntimeError("script exhausted")
        return make_response(self.script[len(self.calls) - 1], messages, system)


def tool_use(name, tool_id="toolu_1", **inputs):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": inputs}


text_block = lambda t: {"type": "text", "text": t}

# %% [markdown]
# ## 4. The agent loop
#
# The whole idea fits in one function. Read it slowly, because every agent framework is a variation on it:
#
# 1. send the conversation and the tool list to the model;
# 2. if the model answered with plain text, we are done;
# 3. otherwise run each requested tool, append the assistant message *and* a user message with the results, and go back to 1;
# 4. **stop after `max_steps`** no matter what: a loop without a budget is an outage waiting to happen.

# %%
def run_agent(llm, tools, user_message, system="", max_steps=8):
    """Minimal agent loop. `tools` maps name -> (schema, function). Returns (final_text, messages, status)."""
    schemas = [schema for schema, _ in tools.values()]
    messages = [{"role": "user", "content": user_message}]
    for step in range(max_steps):
        resp = llm.complete(messages, system=system, tools=schemas)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return resp.text, messages, "done"
        results = []
        for call in resp.tool_calls:
            _, fn = tools[call["name"]]
            results.append({"type": "tool_result", "tool_use_id": call["id"], "content": fn(**call["input"])})
        messages.append({"role": "user", "content": results})
    return "", messages, "max_steps"


TOOLS = {
    "get_order": ({"name": "get_order", "description": "Look up an order by numeric id.",
                   "input_schema": {"type": "object", "properties": {"order_id": {"type": "integer"}}, "required": ["order_id"]}}, get_order),
    "search_kb": ({"name": "search_kb", "description": "Search the company knowledge base.",
                   "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}, search_kb),
}

script = [
    [text_block("Let me look that order up."), tool_use("get_order", "toolu_1", order_id=1000)],
    [text_block("Order 1000 is shipped. Since it has shipped, cancellation is no longer free; the returns process applies.")],
]
llm = ScriptedLLM(script)
final, messages, status = run_agent(llm, TOOLS, "What is the status of order 1000?")
print(status, "|", final)

# %%
def show(messages):
    """Print a trajectory compactly."""
    for m in messages:
        if isinstance(m["content"], str):
            print(f"{m['role']:>9}: {m['content'][:90]}")
            continue
        for b in m["content"]:
            if b["type"] == "text":
                print(f"{m['role']:>9}: {b['text'][:90]}")
            elif b["type"] == "tool_use":
                print(f"{m['role']:>9}: TOOL_USE {b['name']}({json.dumps(b['input'])})  id={b['id']}")
            elif b["type"] == "tool_result":
                print(f"{m['role']:>9}: TOOL_RESULT id={b['tool_use_id']} -> {b['content'][:70]}")


show(messages)
print("model calls:", len(llm.calls))

# %% [markdown]
# *What just happened?* Two model calls and one tool call. The second call received the whole conversation including the `tool_result`, which is how the
# model "sees" the database answer: **the model never touches your data; it only reads what you put in the messages**. That single fact is the foundation of
# tool safety (module 22, notebook 04): you control exactly what each tool can do and what it returns.
#
# ## 5. A mock model that decides for itself
#
# A fixed script only replays one story. To watch a loop *react* to different requests we need a mock that chooses tools from the request and the results so far.
# `RuleAgentLLM` is a tiny policy written with regular expressions: it works out which tools the question needs (an order id means `get_order`; a policy keyword
# means `search_kb`), calls the ones it has not called yet, and finally writes an answer from the tool results. It is **not** how an LLM decides (a real model
# reads your tool descriptions and reasons), but it exercises the same loop and gives you a deterministic system to test.

# %%
KB_KEYWORDS = r"return|refund|cancel|shipping|ship|warranty|points|premium|payment|privacy|support|delivery"


class RuleAgentLLM:
    def complete(self, messages, system="", tools=None, max_tokens=1024):
        available = {t["name"] for t in (tools or [])}
        question = messages[0]["content"]
        calls, results = {}, {}
        for m in messages[1:]:
            if isinstance(m["content"], list):
                for b in m["content"]:
                    if b["type"] == "tool_use":
                        calls[b["id"]] = b
                    elif b["type"] == "tool_result":
                        results[calls[b["tool_use_id"]]["name"]] = json.loads(b["content"])
        needs = []
        if (m := re.search(r"order\s*#?(\d+)", question, re.I)) and "get_order" in available:
            needs.append(("get_order", {"order_id": int(m.group(1))}))
        if re.search(KB_KEYWORDS, question, re.I) and "search_kb" in available and not re.search(r"how many", question, re.I):
            needs.append(("search_kb", {"query": question}))
        if (m := re.search(r"how many (\w+) orders", question, re.I)) and "count_orders" in available:
            needs.append(("count_orders", {"status": m.group(1).lower()}))
        for name, args in needs:
            if name not in results:
                blocks = [text_block(f"I will use {name}."), {"type": "tool_use", "id": f"toolu_{len(calls) + 1}", "name": name, "input": args}]
                return make_response(blocks, messages, system)
        parts = []
        if "get_order" in results:
            o = results["get_order"]
            parts.append(o["error"] if "error" in o else f"Order {o['order_id']} is {o['status']} (placed {o['date']}, {o['items']} item(s), total {o['total']}).")
        if "search_kb" in results:
            parts.append(f"Policy ({results['search_kb']['document']}): {results['search_kb']['text']}")
        if "count_orders" in results:
            parts.append(f"There are {results['count_orders']['count']} {results['count_orders']['status']} orders.")
        return make_response([text_block(" ".join(parts) or "I can help with order status and shop policies.")], messages, system)


agent_llm = RuleAgentLLM()
for q in ["What is the status of order 1000?", "Order 1000: where is it, and can I still cancel it?", "Tell me a joke", "Where is order 99999999?"]:
    final, msgs, status = run_agent(agent_llm, TOOLS, q)
    n_tools = sum(1 for m in msgs if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_use")
    print(f"Q: {q}\n   {status}, {n_tools} tool call(s): {final[:150]}")

# %% [markdown]
# The same loop handled four different requests with zero, one and two tool calls and even a not-found error. That flexibility is the point of an agent.
# The *cost* is also visible: the number of model calls now depends on the input.
#
# ### The same task as a workflow
# For a request this well-understood, a **workflow** is simpler and more predictable: code extracts the order id, code calls the tools, and (in a real system)
# one LLM call writes the reply. There is exactly one model call and the path is fixed.

# %%
def workflow_answer(question):
    calls = 0
    parts = []
    if (m := re.search(r"order\s*#?(\d+)", question, re.I)):
        o = json.loads(get_order(int(m.group(1))))
        parts.append(o.get("error") or f"Order {o['order_id']} is {o['status']}.")
    if re.search(KB_KEYWORDS, question, re.I):
        parts.append(json.loads(search_kb(question))["text"])
    calls += 1                                   # in production: ONE llm call to phrase `parts` as a friendly reply
    return " ".join(parts), calls


print(workflow_answer("Order 1000: where is it, and can I still cancel it?"))

# %% [markdown]
# Same answer content, one model call, no chance of wandering. If every request looks like "order id and/or policy question", **the workflow is the better
# engineering choice**. The agent earns its cost only when the steps genuinely depend on what earlier steps discover (for example: find the order, then
# decide which of five tools to use to investigate why it is late, then decide whether to escalate).

# %% [markdown]
# ## 6. Memory and state
#
# Where does an agent keep what it knows?
# - **Short-term memory** is the `messages` list itself: the whole conversation, resent to the model on every call.
# - **Working memory / scratchpad**: notes the agent writes down (a `notes` tool, a plan in the first message) so that facts survive when old messages are trimmed.
# - **Long-term memory**: an external store (database, vector index) the agent reads through tools. Notebook 03 of module 23 builds these.
#
# Because the *whole* history is resent each step, the input tokens billed grow with every loop iteration. Let's measure it on the trajectory we just produced.

# %%
final, msgs, status = run_agent(agent_llm, TOOLS, "Order 1000: where is it, and can I still cancel it?")
billed = []
history = []
for m in msgs:
    history.append(m)
    if m["role"] == "user":
        billed.append(sum(est_tokens(x["content"]) for x in history))
print("estimated input tokens sent on each model call:", billed)
print("tool result sizes (est. tokens):", [est_tokens(b["content"]) for m in msgs if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_result"])

# %% [markdown]
# ## 7. When NOT to build an agent
#
# Ask these in order and stop at the first "no":
# 1. **Can one well-written prompt (possibly with retrieved context) do the job?** Then use one call.
# 2. **Are the steps known in advance?** Then write a workflow (chaining, routing: next notebooks).
# 3. **Is the task valuable enough to pay for many model calls and higher latency?**
# 4. **Can mistakes be detected and undone** (tests, review, a sandbox, a confirmation step)?
#
# Only if all are "yes" do you need the model choosing its own path. Two pieces of arithmetic explain why the bar is high.
#
# **Compounding unreliability.** If each step succeeds independently with probability `p`, a chain of `n` steps succeeds with probability `p^n`
# (independence is a simplification, and real errors correlate, but the direction is right):

# %%
for p in (0.99, 0.95, 0.90):
    print(f"per-step reliability {p:.2f}: " + "  ".join(f"{n} steps -> {p ** n:.2f}" for n in (1, 5, 10, 20)))

# %% [markdown]
# **Growing context.** If each loop adds `a` tokens (model output plus tool result) to a conversation that starts at `s` tokens, then step `t` sends about `s + t*a`
# input tokens, so the total over `n` steps is `n*s + a*n(n-1)/2`: quadratic in the number of steps. Ten steps is far more than ten times one step.

# %%
def loop_input_tokens(s, a, n):
    return sum(s + t * a for t in range(n))


for n in (1, 5, 10, 20):
    print(f"{n:>2} steps: {loop_input_tokens(500, 300, n):>7,} input tokens  (n * s alone would be {n * 500:>6,})")

# %% [markdown]
# Practical consequences: cap the steps, keep tool results short (return the answer, not a raw table dump), summarise old turns, and prefer workflows whenever the path
# is known. Also decide *what happens on failure* before you ship: a hard stop with a message, a fallback to a human, a retry with a fresh context.

# %% [markdown]
# ## Common mistakes
#
# 1. **Building an agent for a fixed pipeline.** If you can draw the flowchart, write the flowchart.
# 2. **No step limit or budget.** A confused model can call the same tool forever; your bill is the stack trace.
# 3. **Forgetting that the model only knows what is in the messages.** If a fact is not in a tool result or the system prompt, the model does not have it.
# 4. **Returning huge tool results.** They are re-sent on every later step and drown the signal.
# 5. **Letting tools raise exceptions** that kill the loop, instead of returning an error message the model can act on.
# 6. **Judging an agent by its final text only.** Two runs can end with the same sentence after very different (and differently risky) trajectories.

# %%
# Demonstration of mistake 2: a confused model (here: one that always asks for the same tool) is stopped only by max_steps
class StuckLLM:
    def complete(self, messages, system="", tools=None, max_tokens=1024):
        return make_response([tool_use("get_order", f"toolu_{len(messages)}", order_id=1)], messages, system)


final, msgs, status = run_agent(StuckLLM(), TOOLS, "loop forever", max_steps=4)
print("status:", status, "| messages produced:", len(msgs))

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between a workflow and an agent, and how do you choose?**
# <details><summary>Answer</summary>
# In a workflow, code fixes the sequence of LLM and tool steps; in an agent the model chooses the next step at run time in a loop. Choose the least autonomy that works:
# a single prompt, then a workflow (chaining, routing, parallelisation), and an agent only when the path genuinely depends on intermediate results and the task is valuable
# enough to pay for variable cost, latency and harder testing.
# </details>
#
# **2. Describe the agent loop.**
# <details><summary>Answer</summary>
# Send messages and tool definitions to the model; if it returns tool calls, execute them, append the assistant message and a user message of <code>tool_result</code> blocks
# (matching <code>tool_use_id</code>), and call the model again; stop when it returns plain text or a step/budget limit is reached. State lives in the message list (plus any external memory).
# </details>
#
# **3. Why do agents get expensive, and what do you do about it?**
# <details><summary>Answer</summary>
# Every iteration resends the full history, so total input tokens grow roughly quadratically with steps, and each step adds latency. Mitigate with step limits, concise tool outputs,
# trimming or summarising history, caching a stable prefix, cheaper models for simple steps, and workflows where the path is known.
# </details>
#
# **4. How would you know an agent is working well in production?**
# <details><summary>Answer</summary>
# Evaluate outcomes and trajectories: task success on a golden set, number of steps and tool calls, wrong-tool and repeated-call rates, error and timeout rates, cost and latency per task,
# and human review of sampled traces. Final-answer checks alone miss unsafe or wasteful paths (notebook 04).
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Write the agent loop
# Write `my_agent(llm, tools, user_message, max_steps=6)` that returns `(final_text, messages, status)` exactly like `run_agent`: call `llm.complete(messages, tools=schemas)`, append the assistant
# message (`{"role": "assistant", "content": resp.content}`), stop with status `"done"` when `resp.stop_reason != "tool_use"`; otherwise execute every tool call, append **one** user message holding all the
# `tool_result` blocks (with matching `tool_use_id`), and continue. After `max_steps` model calls without finishing, return `("", messages, "max_steps")`.
# *Hint: `tools[name][1](**call["input"])` runs a tool.*

# %% tags=["starter"]
def my_agent(llm, tools, user_message, max_steps=6):
    ...


# %%
scripted = ScriptedLLM([
    [tool_use("get_order", "toolu_a", order_id=1000), tool_use("search_kb", "toolu_b", query="cancel an order")],
    [text_block("done")],
])
final, msgs, status = my_agent(scripted, TOOLS, "hello")
assert status == "done" and final == "done", f"expected status 'done', got {status!r}"
assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"], "roles must alternate user/assistant/user/assistant"
results = msgs[2]["content"]
assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"], "ONE user message with a tool_result per tool_use, ids matching"
assert json.loads(results[0]["content"])["order_id"] == 1000, "tool must really be executed"
assert len(scripted.calls) == 2 and scripted.calls[1]["messages"][-1]["role"] == "user", "the second model call must see the tool results"
stuck = StuckLLM()
_, msgs2, status2 = my_agent(stuck, TOOLS, "loop", max_steps=3)
assert status2 == "max_steps" and sum(m["role"] == "assistant" for m in msgs2) == 3, "stop after exactly max_steps model calls"
print("Exercise 1 passed")

# %% tags=["solution"]
def my_agent(llm, tools, user_message, max_steps=6):
    schemas = [schema for schema, _ in tools.values()]
    messages = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        resp = llm.complete(messages, tools=schemas)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return resp.text, messages, "done"
        results = [
            {"type": "tool_result", "tool_use_id": c["id"], "content": tools[c["name"]][1](**c["input"])}
            for c in resp.tool_calls
        ]
        messages.append({"role": "user", "content": results})
    return "", messages, "max_steps"


# %% [markdown]
# ### Exercise 2: Reliability and cost arithmetic
# (a) Write `chain_reliability(p, n)`: the probability that `n` independent steps of reliability `p` all succeed.
# (b) Write `max_steps_for(target, p)`: the **largest** number of steps `n` such that `chain_reliability(p, n) >= target` (return 0 if even one step is below target).
# (c) Store in `tokens_20_steps` the value of `loop_input_tokens(800, 250, 20)` divided by the tokens a *single* call with the same 800-token start would use, as a float
# (that is, the cost multiplier of a 20-step loop over one call).

# %% tags=["starter"]
def chain_reliability(p, n):
    ...


def max_steps_for(target, p):
    ...


tokens_20_steps = ...

# %%
assert abs(chain_reliability(0.9, 3) - 0.729) < 1e-12
assert chain_reliability(1.0, 50) == 1.0 and chain_reliability(0.5, 0) == 1.0
assert max_steps_for(0.5, 0.9) == 6, f"0.9**6={0.9**6:.3f} >= 0.5 but 0.9**7={0.9**7:.3f} < 0.5"
assert max_steps_for(0.99, 0.9) == 0, "one step at 0.9 is already below a 0.99 target"
assert max_steps_for(0.9, 0.99) == 10
exp = loop_input_tokens(800, 250, 20) / 800
assert abs(tokens_20_steps - exp) < 1e-9, f"multiplier should be {exp:.2f}"
print(f"Exercise 2 passed (a 20-step loop costs {tokens_20_steps:.1f}x the input tokens of one call)")

# %% tags=["solution"]
def chain_reliability(p, n):
    return p ** n


def max_steps_for(target, p):
    n = 0
    while chain_reliability(p, n + 1) >= target - 1e-12:
        n += 1
    return n


tokens_20_steps = loop_input_tokens(800, 250, 20) / 800


# %% [markdown]
# ### Exercise 3: Give the agent a new tool
# The mock `RuleAgentLLM` also understands the tool name `count_orders` (questions like "How many delivered orders are there?" make it call `count_orders` with `{"status": "delivered"}`),
# but only if that tool is in the tool list. Implement `count_orders(status: str) -> str` returning JSON `{"status": ..., "count": N}` using the read-only `db` connection (use a **parameterised** query),
# and register it in `TOOLS2` with a proper schema. The check runs the agent and compares its answer with a pandas count of the same table.

# %% [markdown]
# *(The mock recognises questions with the pattern `how many (\w+) orders`, asks for `count_orders`, and writes "There are N <status> orders." from the tool result.)*

# %% tags=["starter"]
def count_orders(status: str) -> str:
    ...


TOOLS2 = dict(TOOLS)
# TOOLS2["count_orders"] = (schema_dict, count_orders)

# %%
import pandas as pd

orders_df = pd.read_sql_query("select status from orders", db)
assert "count_orders" in TOOLS2, "register the tool in TOOLS2"
schema, fn = TOOLS2["count_orders"]
assert schema["name"] == "count_orders" and schema["input_schema"]["properties"]["status"]["type"] == "string", "schema needs name and a string 'status' property"
for status in ("delivered", "cancelled", "returned"):
    final, msgs, st = my_agent(RuleAgentLLM(), TOOLS2, f"How many {status} orders are there?")
    expected = int((orders_df["status"] == status).sum())
    assert st == "done" and f"There are {expected} {status} orders." == final, f"expected the count {expected} for {status}, agent said: {final}"
r = json.loads(fn("x'; drop table orders; --"))
assert r["count"] == 0, "a hostile status string must just match nothing (parameterised query)"
assert db.execute("select count(*) from orders").fetchone()[0] == len(orders_df), "the table must be untouched"
print("Exercise 3 passed")

# %% tags=["solution"]
def count_orders(status: str) -> str:
    n = db.execute("select count(*) from orders where status = ?", (status,)).fetchone()[0]
    return json.dumps({"status": status, "count": n})


TOOLS2 = dict(TOOLS)
TOOLS2["count_orders"] = (
    {"name": "count_orders", "description": "Count orders that have a given status (delivered, shipped, cancelled, returned).",
     "input_schema": {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}},
    count_orders,
)

# %% [markdown]
# ### Exercise 4 (open-ended): Workflow, agent or single call?
# Pick **three** tasks from your own job or a product you know (for example: "summarise a ticket", "triage and route incoming email", "investigate why a nightly job failed").
# For each, write in a markdown cell: your choice among *single call / workflow / agent*, which of the four questions in section 7 decided it, what the failure mode would be, and how
# you would detect it. If you chose an agent, state the step budget and what a human must confirm.

# %% [markdown]
# ## Recap
# - A workflow is code-orchestrated LLM steps; an agent lets the model choose steps in a loop. Pick the least autonomy that works.
# - The loop: model -> tool calls -> `tool_result` messages -> model, until plain text or a step limit. The model only sees what is in the messages.
# - Tools are functions with a name, description and input schema; they should return short strings and report errors as text instead of raising.
# - Multi-step loops resend history (input cost grows roughly quadratically) and compound errors (`p^n`); both push you toward fewer steps and more structure.
# - Test trajectories, not just final answers.
#
# **Go deeper:** Anthropic's engineering article "Building effective agents" and the tool-use section of the API documentation (docs.anthropic.com).
#
# **Next:** [Tool use and function calling](02_tool_use_and_function_calling.ipynb)
