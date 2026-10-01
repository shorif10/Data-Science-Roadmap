# %% [markdown]
# # Memory, State and Context Management
#
# **Why this matters.** An agent has no memory of its own: on every model call you re-send whatever you want it to "remember". A long support session, a coding agent that has read twenty files, a data agent that has run ten queries: the
# history grows with every step, each tool result is re-billed on every later call, and at some point the conversation no longer fits or quality degrades. **Context management** is the engineering of what goes into that window, in what form, and what is
# allowed to fall out. Every strategy trades cost against the risk of forgetting something that mattered, so this notebook measures both.
#
# **You will learn**
# - account for a context window: system prompt, tool definitions, messages (and which part grows), with a token estimator and an honest idea of its error
# - implement and compare five strategies on a 30-turn session: sliding window, clearing old tool results, rolling summaries, a scratchpad of notes, and retrieval memory
# - show concretely what each strategy **loses** (a fact test on a detail from turn 3), and why summaries are lossy
# - build a small retrieval memory with TF-IDF, measure recall, and see where lexical retrieval fails
#
# **Prerequisites:** [Build an agent loop](01_build_an_agent_loop_from_scratch.ipynb); [embeddings and vector search](../21_LLMs_and_RAG/03_embeddings_and_vector_search.ipynb) helps for the retrieval section.
#
# > Data used: `datasets/shop.db` (read-only) to generate a realistic 30-turn tool-using session. Token counts are an **estimate** (about four characters per token); the real tokenizer differs by text type and model, and the API
# > can count exactly (`no-run` cell). The summariser in the runnable cells is a **deterministic extractive mock**: it shows where a summary can lose facts, not how well a real model summarises.

# %%
import json
import math
import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA = Path("../datasets")
DB_URI = (DATA / "shop.db").resolve().as_uri() + "?mode=ro"
KB = {p.stem: p.read_text(encoding="utf-8") for p in sorted((DATA / "knowledge_base").glob("*.md"))}


def query(sql, params=()):
    con = sqlite3.connect(DB_URI, uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


print(query("select count(*) from orders")[0][0], "orders")

# %% [markdown]
# ## 1. What is in the window, and what grows
#
# Each request carries: the **system prompt**, the **tool definitions** (yes, these are input tokens on every call), and the **messages** (user text, assistant text, `tool_use` blocks, `tool_result` blocks). Output tokens are
# reserved out of the same window. Only the messages grow during a session, and within them **tool results dominate** for tool-heavy agents: a SQL result or a file read can be thousands of tokens, and it is re-sent on every
# later step. That is why most context strategies target tool results first.
#
# A token estimator is enough for budgeting decisions (are we at 40% or 95%?) as long as you keep a safety margin; use the provider's exact counter when precision matters.

# %%
def est_tokens(x) -> int:
    x = x if isinstance(x, str) else json.dumps(x)
    return max(1, math.ceil(len(x) / 4))


sample = " ".join(KB["returns_policy"].split())
print("characters:", len(sample), "| chars/4 estimate:", est_tokens(sample), "| words * 1.3 estimate:", math.ceil(len(sample.split()) * 1.3))
blocks_of = lambda m: [{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else m["content"]


def breakdown(system, tools, messages):
    """Estimated tokens by category."""
    parts = {"system": est_tokens(system) if system else 0, "tools": est_tokens(tools) if tools else 0, "user_text": 0, "assistant_text": 0, "tool_use": 0, "tool_result": 0}
    for m in messages:
        for b in blocks_of(m):
            if b["type"] == "text":
                parts["user_text" if m["role"] == "user" else "assistant_text"] += est_tokens(b["text"])
            elif b["type"] == "tool_use":
                parts["tool_use"] += est_tokens(b)
            elif b["type"] == "tool_result":
                parts["tool_result"] += est_tokens(b["content"])
    parts["total"] = sum(parts.values())
    return parts


def fits(parts, window, reserve_output):
    return parts["total"] + reserve_output <= window


# %% [markdown]
# Two different estimators disagree by some percent on the same paragraph, which is the honest size of the error you should budget for (real tokenizers also vary by language and by code versus prose). Now generate a realistic session.

# %% [markdown]
# ## 2. A 30-turn tool-using session from real data
#
# Each turn: the user asks about an order; the assistant calls `get_order`; the tool returns JSON; the assistant answers. Every fifth turn the assistant *also* calls `list_orders` and gets a bulky 25-row result. In turn 3 the user mentions
# a **preference that no tool ever touches** ("I prefer email contact over phone"). That single sentence is our test of what each strategy forgets. Messages follow the real API shapes (`tool_use` / `tool_result` blocks).

# %%
def order_row(oid):
    r = query("select o.order_id, o.status, o.order_date, sum(i.quantity), round(sum(i.quantity * p.unit_price), 2) from orders o join order_items i using(order_id) "
              "join products p using(product_id) where o.order_id = ? group by o.order_id", (oid,))[0]
    return dict(zip(("order_id", "status", "date", "items", "total"), r))


def tool_turn(tid, name, args, result):
    return [{"role": "assistant", "content": [{"type": "text", "text": "Checking."}, {"type": "tool_use", "id": tid, "name": name, "input": args}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": json.dumps(result)}]}]


def make_session(n_turns=30):
    messages = []
    for t in range(1, n_turns + 1):
        oid, o = 1000 + t, order_row(1000 + t)
        messages.append({"role": "user", "content": f"Please check order {oid}." + (" By the way, I prefer email contact over phone." if t == 3 else "")})
        messages += tool_turn(f"toolu_{t}a", "get_order", {"order_id": oid}, o)
        if t % 5 == 0:
            rows = query("select order_id, status, order_date from orders where order_id between ? and ? order by order_id", (oid, oid + 24))
            messages += tool_turn(f"toolu_{t}b", "list_orders", {"from": oid, "count": 25}, [dict(zip(("order_id", "status", "date"), r)) for r in rows])
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"Order {oid} is {o['status']} (total {o['total']})."}]})
    return messages


def turns_of(messages):
    """Split a message list into turns; a turn starts at a user message whose content is plain text."""
    turns = []
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], str):
            turns.append([])
        turns[-1].append(m)
    return turns


flatten = lambda turns: [m for t in turns for m in t]
session = make_session()
turns = turns_of(session)
print(len(turns), "turns,", len(session), "messages")
full = breakdown("You are a shop support assistant.", [{"name": "get_order"}] * 3, session)
print({k: v for k, v in full.items()})
print(f"tool results are {full['tool_result'] / full['total']:.0%} of the estimated tokens at the end of the session")


# %% [markdown]
# We need a validity check: whatever a strategy does to the history, every `tool_use` must keep its `tool_result` (notebook 22/02). Strategies that cut history must cut at **turn boundaries** or the API will reject the request.

# %%
def validate_messages(messages):
    pending = None
    for i, m in enumerate(messages):
        results = sorted(b["tool_use_id"] for b in blocks_of(m) if b["type"] == "tool_result")
        if pending is not None and (m["role"] != "user" or results != sorted(pending)):
            raise ValueError(f"message {i}: expected tool_result blocks for {sorted(pending)}")
        if pending is None and results:
            raise ValueError(f"message {i}: tool_result without a preceding tool_use")
        pending = [b["id"] for b in blocks_of(m) if b["type"] == "tool_use"] or None if m["role"] == "assistant" else None
    if pending:
        raise ValueError("ends with unanswered tool_use")
    return True


print("full session valid:", validate_messages(session))
try:
    validate_messages(session[2:])                                     # a careless cut in the middle of a turn
except ValueError as exc:
    print("cutting mid-turn ->", exc)

# %% [markdown]
# ## 3. Five strategies
#
# Every strategy answers the same question: given the turns so far and the new question, **what do we send?** Each class has `observe(turn)` (called after a turn finishes; lets a strategy update memory) and `build(turns, query)`
# (returns `(system, messages)`). `overhead` counts tokens spent on *extra* model calls (summarising). All of them keep the last few turns verbatim, because recent context is what the current step usually needs.
#
# 1. **Sliding window:** drop whole old turns. Cheapest, forgets everything old.
# 2. **Clear old tool results:** keep every turn but replace old `tool_result` content with a short placeholder. The assistant's own answers (which usually restate what mattered) stay. This is the idea behind the API's
#    server-side "context editing" (shown in a `no-run` cell).
# 3. **Rolling summary:** when a turn falls out of the window, fold it into a running summary (an extra model call).
# 4. **Scratchpad notes:** the agent writes durable facts into a notes store; notes are re-sent, old turns dropped.
# 5. **Retrieval memory:** store each turn in an index; retrieve the few most relevant pieces for the current question.

# %%
BASE_SYSTEM = "You are a shop support assistant."
PLACEHOLDER = "[result cleared to save context]"


class FullHistory:
    name = "full history"

    def __init__(self):
        self.overhead = 0

    def observe(self, turn):
        pass

    def build(self, turns, query):
        return BASE_SYSTEM, flatten(turns)


class SlidingWindow(FullHistory):
    name = "sliding window (3)"

    def __init__(self, keep=3):
        super().__init__()
        self.keep = keep

    def build(self, turns, query):
        return BASE_SYSTEM, flatten(turns[-self.keep:])


class ClearToolResults(SlidingWindow):
    name = "clear old tool results"

    def build(self, turns, query):
        out = []
        for i, turn in enumerate(turns):
            old = i < len(turns) - self.keep
            for m in turn:
                if old and isinstance(m["content"], list) and any(b["type"] == "tool_result" for b in m["content"]):
                    m = {"role": m["role"], "content": [{**b, "content": PLACEHOLDER} if b["type"] == "tool_result" else b for b in m["content"]]}
                out.append(m)
        return BASE_SYSTEM, out


ORDER_FACT = re.compile(r"Order (\d+) is (\w+)")


class RollingSummary(SlidingWindow):
    name = "rolling summary"

    def __init__(self, keep=3):
        super().__init__(keep)
        self.summary, self.seen = "", []

    def summarise(self, previous, dropped_turn):
        """Extractive mock of an LLM summariser: keeps only order facts. (A real model keeps more, but still drops things.)"""
        text = " ".join(b["text"] for m in dropped_turn if m["role"] == "assistant" for b in blocks_of(m) if b["type"] == "text")
        new = "".join(f"- Order {oid} is {status}\n" for oid, status in ORDER_FACT.findall(text))
        self.overhead += est_tokens(previous) + est_tokens(json.dumps(dropped_turn)) + est_tokens(previous + new)     # tokens of the extra model call (in + out)
        return previous + new

    def observe(self, turn):
        self.seen.append(turn)
        if len(self.seen) > self.keep:
            self.summary = self.summarise(self.summary, self.seen[-self.keep - 1])

    def build(self, turns, query):
        head = [{"role": "user", "content": "Summary of the conversation so far:\n" + self.summary}, {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]}] if self.summary else []
        return BASE_SYSTEM, head + flatten(turns[-self.keep:])


class ScratchpadNotes(SlidingWindow):
    name = "scratchpad notes"

    def __init__(self, keep=2):
        super().__init__(keep)
        self.notes = {}

    def observe(self, turn):
        # the agent calls a `note(key, value)` tool for facts it judges durable; this mock notes order facts only
        for m in turn:
            for b in blocks_of(m):
                if m["role"] == "assistant" and b["type"] == "text":
                    for oid, status in ORDER_FACT.findall(b["text"]):
                        self.notes[f"order_{oid}"] = status

    def build(self, turns, query):
        system = BASE_SYSTEM + "\nNOTES:\n" + "\n".join(f"{k}: {v}" for k, v in self.notes.items())
        return system, flatten(turns[-self.keep:])


class MemoryStore:
    """TF-IDF retrieval memory: add(text), search(query, k) -> top-k texts by cosine similarity."""

    def __init__(self):
        self.texts, self._vec, self._mat = [], None, None

    def add(self, text):
        self.texts.append(text)
        self._vec = None

    def search(self, query, k=3):
        if not self.texts:
            return []
        if self._vec is None:
            self._vec = TfidfVectorizer().fit(self.texts)
            self._mat = self._vec.transform(self.texts)
        scores = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.texts[i] for i in order if scores[i] > 0]


class RetrievalMemory(SlidingWindow):
    name = "retrieval memory (k=3)"

    def __init__(self, keep=2, k=3):
        super().__init__(keep)
        self.k, self.store = k, MemoryStore()

    def observe(self, turn):
        user = next(m["content"] for m in turn if m["role"] == "user" and isinstance(m["content"], str))
        answer = " ".join(b["text"] for m in turn if m["role"] == "assistant" for b in blocks_of(m) if b["type"] == "text" and b["text"] != "Checking.")
        self.store.add(f"{user} {answer}")

    def build(self, turns, query):
        hits = self.store.search(query, self.k)
        head = [{"role": "user", "content": "Relevant memory:\n" + "\n".join(f"- {h}" for h in hits)}, {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]}] if hits else []
        return BASE_SYSTEM, head + flatten(turns[-self.keep:])


s3 = order_row(1003)["status"]
ctx_text = lambda system, messages: system + "\n" + json.dumps(messages)
has_order_fact = lambda text, oid, status: re.search(rf"{oid}\D{{0,40}}{status}", text) is not None
has_preference = lambda text: "prefer email" in text
print("status of order 1003:", s3)

# %% [markdown]
# ## 4. Run every strategy over the session and measure
#
# For each turn we build the context the model would receive (strategy applied to the turns so far, plus the new user message) and record its estimated size. Then, after turn 30, we ask two **probe questions** and check whether the
# fact needed to answer is present in the context the strategy builds for that question:
# - *Probe 1:* "What was the status of order 1003?" (a fact a tool returned in turn 3).
# - *Probe 2:* "Which contact method does the customer prefer, email or phone?" (a fact only the user ever said, in turn 3).
#
# A fact absent from the context **cannot** be used by the model (except by guessing): presence is the necessary condition we can test offline. (Whether a real model then uses it correctly is a separate evaluation.)

# %%
TOOLS_STUB = [{"name": "get_order"}, {"name": "list_orders"}, {"name": "note"}]
STRATEGIES = [FullHistory, SlidingWindow, ClearToolResults, RollingSummary, ScratchpadNotes, RetrievalMemory]
series, rows = {}, []
for cls in STRATEGIES:
    s = cls()
    sizes = []
    for t, turn in enumerate(turns):
        user_text = turn[0]["content"]
        system, msgs = s.build(turns[:t], user_text)
        sizes.append(breakdown(system, TOOLS_STUB, msgs)["total"] + est_tokens(user_text))
        s.observe(turn)
    p1_sys, p1_msgs = s.build(turns, "What was the status of order 1003?")
    p2_sys, p2_msgs = s.build(turns, "Which contact method does the customer prefer, email or phone?")
    assert validate_messages(p1_msgs) and validate_messages(p2_msgs), s.name
    series[s.name] = sizes
    rows.append({"strategy": s.name, "last_turn_tokens": sizes[-1], "peak": max(sizes), "total_input_tokens": sum(sizes), "summariser_overhead": s.overhead,
                 "probe1_status": has_order_fact(ctx_text(p1_sys, p1_msgs), 1003, s3), "probe2_preference": has_preference(ctx_text(p2_sys, p2_msgs))})
result = pd.DataFrame(rows)
result["cost_incl_overhead"] = result.total_input_tokens + result.summariser_overhead
print(result.to_string(index=False))

# %% [markdown]
# Plot the size of the context at each turn:

# %%
fig, ax = plt.subplots(figsize=(8, 4))
for name, sizes in series.items():
    ax.plot(range(1, len(sizes) + 1), sizes, label=name)
ax.set_xlabel("turn")
ax.set_ylabel("estimated input tokens sent on that turn")
ax.set_title("Context size per turn for five strategies (30-turn shop session, chars/4 estimate)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* Read the table as a trade-off, not a winner list.
# - **Full history** grows steadily and sends by far the most tokens in total; it is the only one that can never "forget", and the price is the largest bill (and, eventually, a window that overflows).
# - **Sliding window** is the cheapest and forgets both probe facts: anything older than three turns is gone.
# - **Clearing old tool results** keeps the assistant's own answers, so probe 1 survives (the answer text restated the status) and probe 2 survives (user turns are kept), at about half the total tokens of full history in this run, because tool results dominated the tokens.
#   It works here because this assistant restates what it learned; an agent that *reads* a file and never restates it would lose that knowledge when the result is cleared.
# - **Rolling summary** preserved the order status but **lost the preference**: our summariser only extracts order facts. A real LLM summariser would probably keep a user preference, but it decides what matters without knowing your
#   future questions, so some detail is always lost, and each summary also costs an extra model call (`summariser_overhead`): in this run, counting those calls (`cost_incl_overhead`), the rolling summary ends up costing more than the window, notes and retrieval strategies.
# - **Scratchpad notes** preserved exactly what the agent chose to write down; nothing it did not note. Notes are cheap and precise, but only as good as the agent's judgement about what is durable.
# - **Retrieval memory** found both facts, because it stores raw text and retrieves by the current question. It depends on retrieval quality (next section) and on there being a good question to retrieve with.
#
# Numbers like these come from *this* session and estimator; your agent's tool-result sizes and turn count will move them a lot. Re-run the experiment on your own transcripts before choosing.

# %% [markdown]
# ## 5. Retrieval memory: how well does it retrieve?
#
# A retrieval memory is a search problem, so measure it like one. Build a store with one entry per turn, then ask for each order's status in natural words and check whether the gold turn is in the top k. Then try a
# paraphrase with no shared words: lexical (TF-IDF) retrieval cannot match meaning without shared vocabulary, which is what embeddings (notebook 21/03) are for.

# %%
store = MemoryStore()
for turn in turns:
    user = turn[0]["content"]
    store.add(f"{user} {turn[-1]['content'][0]['text']}")
queries = [(f"What was the status of order {1000 + t}?", t) for t in range(1, 31)]


def recall_at_k(store, queries, k):
    return float(np.mean([any(f"order {1000 + t}" in hit.lower() and f"Order {1000 + t} is" in hit for hit in store.search(q, k)) for q, t in queries]))


print("recall@1:", recall_at_k(store, queries, 1), "| recall@3:", recall_at_k(store, queries, 3))
for q in ["Which contact method does the customer prefer, email or phone?", "communication channel preference of this person"]:
    hits = store.search(q, 1)
    print(f"{q!r:<70} -> {hits[0][:60] if hits else 'NO HIT'}")

# %% [markdown]
# Order ids are distinctive tokens, so lexical retrieval is perfect on them. The first preference question shares words ("prefer", "email", "contact", "phone") with the stored sentence and is found; the paraphrase ("communication channel
# preference") shares none and finds **nothing relevant**. Real memory systems therefore combine lexical and embedding retrieval, and they must decide what to store (raw text, extracted facts, or both).
#
# **Memory risks to design for.** *Staleness*: facts change ("prefers email" becomes "prefers phone"); store timestamps and prefer the latest. *Poisoning*: anything you store and later re-inject was once input: if an attacker's text lands in memory
# it comes back on every future run, so apply the injection defences of notebook 22/04 to memory reads. *Privacy*: long-lived memory holds personal data; decide retention, let users view and delete it, and never store secrets.
# *Attribution*: keep the source and time of each memory so a wrong answer can be traced.

# %% [markdown]
# ## 6. Budget-driven context: decide from the numbers
#
# A practical policy uses the estimator as a trigger: measure the context before each call and act when it crosses thresholds, instead of a fixed number of turns.

# %%
def manage(turns_so_far, system, tools, window, reserve, soft=0.6, hard=0.85):
    """Return (messages, action). Soft limit: clear old tool results. Hard limit: also drop the oldest turns."""
    strat = ClearToolResults(keep=3)
    _, msgs = strat.build(turns_so_far, "")
    if breakdown(system, tools, flatten(turns_so_far))["total"] <= soft * (window - reserve):
        return flatten(turns_so_far), "keep everything"
    if breakdown(system, tools, msgs)["total"] <= hard * (window - reserve):
        return msgs, "cleared old tool results"
    kept = list(turns_so_far)
    while len(kept) > 1 and breakdown(system, tools, ClearToolResults(keep=3).build(kept, "")[1])["total"] > hard * (window - reserve):
        kept.pop(0)
    return ClearToolResults(keep=3).build(kept, "")[1], f"dropped {len(turns_so_far) - len(kept)} oldest turns"


for window in (20000, 6000, 3000):
    msgs, action = manage(turns, BASE_SYSTEM, TOOLS_STUB, window=window, reserve=500)
    print(f"window={window:>6}: {action:<30} -> {breakdown(BASE_SYSTEM, TOOLS_STUB, msgs)['total']:>5} est. tokens, valid={validate_messages(msgs)}")

# %% [markdown]
# ## 7. Exact counts and server-side features (`no-run`)
#
# With the real API you can count tokens exactly for a given model, and the platform offers server-side context management features in beta (clearing old tool results, compaction of long conversations) and a memory tool
# type. Names and availability change between models and releases, so treat the snippet as a pointer and check the current documentation. Needs `ANTHROPIC_API_KEY` and network access.

# %% tags=["no-run"]
import anthropic

client = anthropic.Anthropic()                                                  # reads ANTHROPIC_API_KEY from the environment
count = client.messages.count_tokens(model="claude-sonnet-5-5", system=BASE_SYSTEM, messages=flatten(turns[-2:]))
print("exact input tokens for the last turns:", count.input_tokens)

# Server-side context editing (beta): clear old tool results before the model sees them.
resp = client.beta.messages.create(
    model="claude-sonnet-5-5", max_tokens=1024, betas=["context-management-2025-06-27"],
    context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
    system=BASE_SYSTEM, messages=flatten(turns[-2:]) + [{"role": "user", "content": "What was the status of order 1029?"}],
)
# Server-side compaction (beta, long conversations): append response.content (including the compaction block) to your history, not just the text.
# Check docs.anthropic.com for the current beta flags, supported models and the memory tool before relying on any of these.

# %% [markdown]
# ## Common mistakes
#
# 1. **Cutting history mid-turn** (between a `tool_use` and its `tool_result`). The next request fails with a 400. Cut at turn boundaries and validate.
# 2. **Summarising and trusting the summary** as if it were lossless. Keep raw facts you cannot afford to lose (ids, amounts, user-stated constraints) in structured notes or retrieval, not only in prose summaries.
# 3. **Ignoring tool definitions and tool results in the budget**: they are usually the bulk of the tokens.
# 4. **Changing the system prompt every turn** (for example re-writing notes into it). It invalidates prompt caching for everything after it; append volatile state late in the messages, or accept the cost knowingly.
# 5. **Unbounded memory writes**: no retention, no deduplication, no deletion, no source tracking. Memory is a database with privacy and security requirements.
# 6. **Storing untrusted text and replaying it as trusted context** (memory poisoning).
# 7. **Choosing a strategy without measuring** on your own transcripts: the ranking depends on your tool-result sizes and on what questions come back later.

# %%
# Demonstration of mistake 2: a summary of the session cannot answer a question about a detail it did not keep
summary = RollingSummary()
for turn in turns:
    summary.observe(turn)
print("summary kept", summary.summary.count("\n"), "order facts; mentions the email preference:", "email" in summary.summary)

# %% [markdown]
# ## Interview corner
#
# **1. Your agent's context grows with every tool call. What are your options, and what does each cost?**
# <details><summary>Answer</summary>
# Clear or truncate old tool results (cheap; lose detail unless the assistant restated it), sliding window (cheapest; forgets old facts), summarisation (extra model calls and lossy), scratchpad or structured notes (cheap and precise but only what
# the agent wrote down), and retrieval memory (good recall on relevant facts, needs a good index and a good query), often combined. Also return smaller tool results in the first place. Choose by measuring cost and a fact-retention test on real transcripts.
# </details>
#
# **2. Why are tool results usually the biggest context problem?**
# <details><summary>Answer</summary>
# They are large (query results, file contents, web pages) and, because the API is stateless, every later call re-sends them. Input cost over a session grows roughly with steps squared. Returning aggregates, capping rows and characters, and clearing
# results the agent no longer needs attack the largest term.
# </details>
#
# **3. What goes wrong if you trim conversation history naively?**
# <details><summary>Answer</summary>
# Cutting between a <code>tool_use</code> and its <code>tool_result</code> gives an API error; dropping old turns makes the agent forget constraints the user stated earlier; rewriting the system prompt each turn breaks caching. Cut at turn boundaries, keep
# durable constraints in notes, and append changing state late.
# </details>
#
# **4. What are the risks of long-term agent memory?**
# <details><summary>Answer</summary>
# Stale facts, poisoned entries that replay on every future run, personal data retention and deletion obligations, wrong attribution, and unbounded growth. Store source and timestamp, sanitise and treat retrieved memory as untrusted input, give users
# visibility and deletion, and evaluate retrieval quality.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Clear old tool results
# Write `clear_old_tool_results(messages, keep_last)` returning a **new** message list in which every `tool_result` block **except the last `keep_last`** ones has its `content` replaced by `"[cleared]"` (keep `tool_use_id`, and keep `is_error` if present).
# Text and `tool_use` blocks must be untouched, the input must not be modified, and the result must still pass `validate_messages`.
# *Hint: count tool_result blocks across the whole list first, then rewrite.*

# %% tags=["starter"]
def clear_old_tool_results(messages, keep_last):
    ...


# %%
msgs = [dict(m) for m in session]
before = json.dumps(msgs)
out = clear_old_tool_results(msgs, 4)
assert json.dumps(msgs) == before, "do not modify the input list"
assert validate_messages(out) and len(out) == len(msgs)
results_in = [b for m in msgs if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_result"]
results_out = [b for m in out if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_result"]
assert len(results_in) == len(results_out) and [b["tool_use_id"] for b in results_in] == [b["tool_use_id"] for b in results_out]
assert all(b["content"] == "[cleared]" for b in results_out[:-4]) and [b["content"] for b in results_out[-4:]] == [b["content"] for b in results_in[-4:]], "only the last 4 survive"
assert [m for m in out if isinstance(m["content"], str)] == [m for m in msgs if isinstance(m["content"], str)], "user text untouched"
assert breakdown("", None, out)["tool_result"] < 0.3 * breakdown("", None, msgs)["tool_result"], "tokens must drop a lot"
err = [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "t", "input": {}}]},
       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "boom", "is_error": True}]}]
assert clear_old_tool_results(err, 0)[2]["content"][0] == {"type": "tool_result", "tool_use_id": "a", "content": "[cleared]", "is_error": True}
assert clear_old_tool_results(err, 5) == err
print("Exercise 1 passed")

# %% tags=["solution"]
def clear_old_tool_results(messages, keep_last):
    total = sum(1 for m in messages if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_result")
    cutoff, seen, out = total - keep_last, 0, []
    for m in messages:
        if isinstance(m["content"], list):
            blocks = []
            for b in m["content"]:
                if b["type"] == "tool_result":
                    seen += 1
                    if seen <= cutoff:
                        b = {**b, "content": "[cleared]"}
                blocks.append(b)
            m = {"role": m["role"], "content": blocks}
        out.append(m)
    return out


# %% [markdown]
# ### Exercise 2: Budget accounting
# Write `plan_context(system, tools, messages, window, reserve_output)` returning a dict with `total` (estimated tokens of everything, from `breakdown`), `available` (`window - reserve_output - total`), `fits` (bool) and `action`:
# `"ok"` if total is at most 60% of `window - reserve_output`, `"trim"` if above 60% but it still fits, `"overflow"` if it does not fit. The check builds contexts of known sizes and compares with values you can derive by hand.
# *Hint: reuse `breakdown`.*

# %% tags=["starter"]
def plan_context(system, tools, messages, window, reserve_output):
    ...


# %%
small = [{"role": "user", "content": "x" * 400}]                                # 400 chars -> 100 estimated tokens; the 40-char system prompt -> 10
r = plan_context("s" * 40, [], small, window=1000, reserve_output=200)
assert r["total"] == 110, r
assert r["total"] == breakdown("s" * 40, [], small)["total"], r
assert r["available"] == 1000 - 200 - r["total"] and r["fits"] is True and r["action"] == "ok", r
r2 = plan_context("s" * 40, [], small, window=200, reserve_output=50)
assert r2["total"] > 0.6 * 150 and r2["fits"] is True and r2["action"] == "trim", r2
r3 = plan_context("s" * 40, [], small, window=120, reserve_output=50)
assert r3["fits"] is False and r3["action"] == "overflow" and r3["available"] < 0, r3
big = plan_context(BASE_SYSTEM, TOOLS_STUB, session, window=2_000, reserve_output=500)
assert big["action"] == "overflow", "the full 30-turn session does not fit in 2,000 tokens"
print("Exercise 2 passed")

# %% tags=["solution"]
def plan_context(system, tools, messages, window, reserve_output):
    total = breakdown(system, tools, messages)["total"]
    budget = window - reserve_output
    action = "ok" if total <= 0.6 * budget else ("trim" if total <= budget else "overflow")
    return {"total": total, "available": budget - total, "fits": total <= budget, "action": action}


# %% [markdown]
# ### Exercise 3: A memory with recency
# `MemoryStore` returns the best lexical match; when two entries say different things about the same topic ("prefers email" at turn 3, "prefers phone" at turn 20) you want the **newer** one. Write `search_latest(store, query, k=3)`
# that returns the top-`k` TF-IDF matches (`store.search(query, k)` gives the top `k` in score order) but returns them ordered by **insertion index descending** (newest first).
# Return a list of texts. The check stores two conflicting preferences and expects the newer first.
# *Hint: map texts back to their index with `store.texts.index(text)`; texts in the check are unique.*

# %% tags=["starter"]
def search_latest(store, query, k=3):
    ...


# %%
ms = MemoryStore()
for text in ["Customer prefers email contact.", "Order 1001 is delivered.", "Shipping takes 3 days.", "Customer prefers phone contact now.", "Order 1002 is shipped."]:
    ms.add(text)
hits = search_latest(ms, "Which contact does the customer prefer?", k=2)
assert hits == ["Customer prefers phone contact now.", "Customer prefers email contact."], hits
assert search_latest(ms, "order 1002 status", k=1) == ["Order 1002 is shipped."]
assert search_latest(ms, "zebra", k=3) == [], "no lexical match -> no hits"
assert len(search_latest(ms, "customer order shipping", k=2)) == 2
print("Exercise 3 passed")

# %% tags=["solution"]
def search_latest(store, query, k=3):
    top = store.search(query, k)
    return sorted(top, key=lambda t: store.texts.index(t), reverse=True)


# %% [markdown]
# ### Exercise 4 (open-ended): Choose a strategy for your agent
# Take an agent you have built or plan to build. Estimate (or measure from a real transcript) the tokens of: system prompt, tool definitions, a typical tool result, and turns per session. Decide which combination of this notebook's strategies
# you would use, where the thresholds would sit, and **which facts must never be lost** (and where you would keep them: notes, retrieval, or the system prompt). Write the fact-retention probe you would run. No assert for this one.

# %% [markdown]
# ## Recap
# - Context = system prompt + tool definitions + messages; only messages grow, and tool results dominate. The API is stateless, so every byte you keep is re-billed on every call.
# - Estimate tokens for budgeting (about chars/4, with a safety margin) and use the provider's counter when precision matters.
# - Strategies: sliding window (cheap, forgets), clearing old tool results (cheap, keeps the assistant's own words), rolling summary (extra calls, lossy), scratchpad notes (precise, only what was written), retrieval memory (good when the query matches, needs a good index).
# - Cut at turn boundaries and validate; never split `tool_use` from `tool_result`; keep the system prompt stable for caching.
# - Test retention with probe questions, measure retrieval recall, and treat memory as data with staleness, privacy and poisoning risks.
#
# **Go deeper:** the context-management, token-counting and prompt-caching pages at docs.anthropic.com; Anthropic's engineering article "Building effective agents"; scikit-learn's `TfidfVectorizer` documentation.
#
# **Next:** [Testing, observability and deploying agents](04_testing_observability_and_deploying_agents.ipynb)
