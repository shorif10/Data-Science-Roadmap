# %% [markdown]
# # Agent Patterns: Chaining, Routing, Parallelisation, Orchestrator-Workers, Evaluator-Optimiser, Planning and Multi-Agent
#
# **Why this matters.** Most production "agents" are not free-roaming loops. They are a handful of well-understood *patterns* that combine LLM calls and code in a fixed or semi-fixed shape.
# Knowing the patterns lets you pick the simplest one that works, predict its cost and latency before you build it, and explain the trade-off to a manager who asks why the
# support bot costs what it costs. This notebook builds each pattern in a few lines and then **measures** them side by side.
#
# **You will learn**
# - build prompt chaining (with programmatic gates), routing, parallelisation (sectioning and voting), orchestrator-workers and evaluator-optimiser
# - add a plan-and-execute loop with replanning, and say where "reflection" and "multi-agent" fit
# - account for calls, tokens, cost and latency of each pattern, and read a trade-off table that is computed, not asserted
# - explain why voting helps only when errors are independent, and why more agents is not automatically better
#
# **Prerequisites:** [What is an agent?](01_what_is_an_agent.ipynb) and [Tool use](02_tool_use_and_function_calling.ipynb).
#
# > Data used: `datasets/shop.db` and the knowledge-base documents. Every "model" here is a **deterministic mock** that plays one *role* (classifier, writer, evaluator...) by simple rules, through
# > the same `complete(messages, system)` interface as the real client. The mock shows the **control flow**, the call counts and the token arithmetic exactly; it says nothing about how well a
# > real model writes or judges. The latency numbers are a **simple model** (fixed overhead plus time per output token), not measurements, and the prices are illustrative list prices you should re-check.

# %%
import json
import math
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
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


COUNTRIES = {r[0] for r in query("select distinct country from customers")}
print(sorted(COUNTRIES), "|", len(KB), "KB documents")

# %% [markdown]
# ## 0. The toolkit: a role-playing mock with accounting
#
# One `FnLLM` object stands in for the model. Each call names a **role** in the system prompt (`ROLE:classify`, `ROLE:writer`...) and a dictionary of small functions supplies the reply for that role.
# `ask()` wraps a call and records a `Call` (input tokens, output tokens, modelled latency). A `Run` is one pattern execution: its answer, its calls, and its wall-clock latency,
# which the pattern itself computes (a chain **adds** latencies; parallel branches take the **maximum**).
#
# The price constants are illustrative: Sonnet-class list prices at the time of writing ($2 per million input tokens, $10 per million output tokens). Always read the current pricing page before budgeting.

# %%
PRICE_IN, PRICE_OUT = 2.00, 10.00                 # dollars per million tokens (illustrative)
LATENCY_BASE, LATENCY_PER_TOKEN = 0.4, 0.02       # modelled seconds: fixed overhead + per output token (a model, not a measurement)


def est_tokens(x) -> int:
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


class FnLLM:
    """Mock model. `fn(prompt, system) -> str` supplies the reply; the interface matches the real client wrapper."""

    def __init__(self, fn):
        self.fn, self.n_calls = fn, 0

    def complete(self, messages, system="", tools=None, max_tokens=1024):
        self.n_calls += 1
        text = self.fn(messages[-1]["content"], system)
        usage = {"input_tokens": est_tokens(system) + sum(est_tokens(m["content"]) for m in messages), "output_tokens": est_tokens(text)}
        return LLMResponse([{"type": "text", "text": text}], "end_turn", usage)


@dataclass
class Call:
    role: str
    tin: int
    tout: int
    latency: float


@dataclass
class Run:
    pattern: str
    answer: str
    calls: list
    latency: float
    model_chooses_path: bool = False

    @property
    def cost(self):
        return sum(c.tin * PRICE_IN + c.tout * PRICE_OUT for c in self.calls) / 1e6


ROLES = {}          # role name -> function(prompt, system) -> reply text
LLM = FnLLM(lambda prompt, system: ROLES[re.match(r"ROLE:(\w+)", system).group(1)](prompt, system))


def ask(role, prompt, extra="", llm=None):
    resp = (llm or LLM).complete([{"role": "user", "content": prompt}], system=f"ROLE:{role}\n{extra}")
    u = resp.usage
    return resp.text, Call(role, u["input_tokens"], u["output_tokens"], LATENCY_BASE + LATENCY_PER_TOKEN * u["output_tokens"])


def get_order(order_id: int) -> dict:
    rows = query("select o.order_id, o.status, o.order_date, sum(i.quantity), round(sum(i.quantity * p.unit_price), 2) from orders o "
                 "join order_items i using(order_id) join products p using(product_id) where o.order_id = ? group by o.order_id", (order_id,))
    if not rows:
        return {"error": f"no order with id {order_id}"}
    return dict(zip(("order_id", "status", "date", "items", "total"), rows[0]))


STOP = set("a an the i my me to of is it in on for do you can how what when was be and or with after have this that are does will much many".split())
words = lambda t: {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP}


def search_kb(q: str) -> str:
    best = max(((len(words(q) & words(line)), line) for text in KB.values() for line in text.splitlines()[1:]), key=lambda x: x[0])
    return best[1] if best[0] else "no relevant passage found"


print(get_order(1000), "|", search_kb("return window electronics"))

# %% [markdown]
# ## 1. Prompt chaining: fixed steps with gates
#
# **Shape:** `step 1 -> gate -> step 2 -> gate -> step 3`. Code decides the order; each LLM call gets a narrow job and the previous step's output. A **gate** is ordinary code that checks an intermediate result
# and stops early (or asks for help) if it is wrong. Chaining trades latency (calls happen one after another) for accuracy and debuggability: each step is small enough to test on its own.
#
# Task: reply to a support ticket. Steps: *extract* the order id (LLM) -> gate: is there one? -> *look up* the order and the policy (tools, no LLM) -> *draft* the reply (LLM) -> gate: does the
# draft mention the order id and avoid promises we cannot keep?

# %%
ROLES["extract"] = lambda p, s: json.dumps({"order_id": int(m.group(1)) if (m := re.search(r"\b(\d{3,})\b", p)) else None})
ROLES["draft"] = lambda p, s: (lambda o, pol: f"Hello, your order {o['order_id']} is {o['status']} ({o['items']} item(s), total {o['total']}). {pol} Reply here if you need anything else.")(
    json.loads(re.search(r"ORDER: (\{.*\})", p).group(1)), re.search(r"POLICY: (.*)", p).group(1))
BANNED = ("guarantee", "100%", "definitely")


def chain_reply(ticket):
    calls = []
    raw, c = ask("extract", ticket)
    calls.append(c)
    order_id = json.loads(raw)["order_id"]
    if order_id is None:                                              # GATE 1: stop before spending more calls
        return Run("chain", "Could you tell me your order number?", calls, sum(c.latency for c in calls))
    order = get_order(order_id)
    if "error" in order:
        return Run("chain", f"I could not find order {order_id}; please check the number.", calls, sum(c.latency for c in calls))
    policy = search_kb(ticket)
    reply, c = ask("draft", f"TICKET: {ticket}\nORDER: {json.dumps(order)}\nPOLICY: {policy}")
    calls.append(c)
    if str(order_id) not in reply or any(b in reply.lower() for b in BANNED):      # GATE 2: validate the draft
        return Run("chain", "ESCALATE: draft failed validation", calls, sum(c.latency for c in calls))
    return Run("chain", reply, calls, sum(c.latency for c in calls))


for t in ["Where is order 1000? Can I still cancel it?", "Where is my parcel?", "Status of order 99999999 please"]:
    r = chain_reply(t)
    print(f"{len(r.calls)} call(s), {r.latency:.1f}s | {r.answer[:110]}")

# %% [markdown]
# *What just happened?* The second ticket had no order id, so gate 1 stopped the chain after **one** cheap call instead of drafting nonsense. Gates are the main reason chains are more reliable than one giant prompt:
# they turn "the model might be wrong" into "code checks the thing that matters". Chains multiply reliability, though. If each of `n` steps independently succeeds with probability `p`, the whole chain succeeds with `p^n`:

# %%
for p in (0.99, 0.95):
    print(f"p={p}: " + "  ".join(f"{n} steps -> {p ** n:.3f}" for n in (2, 4, 8)))

# %% [markdown]
# ## 2. Routing: classify, then specialise
#
# **Shape:** `classify -> one of several specialised handlers`. A router sends each input to a prompt, tool set or model tuned for that category (and can send simple tickets to a cheaper model,
# hard ones to a stronger one, unknown ones to a human). It wins when categories are distinct and the router is accurate. It loses when the router is wrong, because a misroute
# looks confident. So **measure routing accuracy** before trusting it.

# %%
SCORES = {"shipping": ["ship", "deliver", "arrive", "tracking", "package", "late", "parcel"],
          "returns": ["return", "refund", "exchange", "send back"],
          "billing": ["charge", "card", "payment", "invoice", "billed"],
          "warranty": ["warranty", "broken", "defect", "repair", "stopped working", "covered"]}
DOC_FOR = {"shipping": "shipping", "returns": "returns_policy", "billing": "payment", "warranty": "warranty"}


def keyword_scores(text):
    t = text.lower()
    return {cat: sum(k in t for k in kws) for cat, kws in SCORES.items()}


def pick(scores):
    best = max(scores.values())
    return next((c for c, v in scores.items() if v == best), "other") if best > 0 else "other"


ROLES["classify"] = lambda p, s: pick(keyword_scores(p))
grab = lambda pattern, text: re.search(pattern, text).group(1)        # first capture group of a regex match
ROLES["answer"] = lambda p, s: f"Per our {grab(r'DOC: (.*)', p)} policy: {grab(r'LINE: (.*)', p)}"
TICKETS = [
    ("Where is order 1000? It has not arrived.", "shipping"),
    ("I want to return the headphones from order 1003 for a refund", "returns"),
    ("You charged my card twice for order 1010", "billing"),
    ("My blender stopped working after two months, is it covered?", "warranty"),
    ("The tracking page says my package is late", "shipping"),
    ("Can I exchange a shirt I bought last week?", "returns"),
    ("I was billed the wrong amount on my invoice", "billing"),
    ("My package arrived broken, I want a refund or a repair", "warranty"),
    ("What are your opening hours?", "other"),
    ("How do I delete my account?", "other"),
]


def route_reply(ticket):
    cat, c1 = ask("classify", ticket)
    if cat == "other":                                                # unknown -> human, no more spend
        return Run("route", "Forwarded to a human agent.", [c1], c1.latency), cat
    doc = DOC_FOR[cat]
    line = KB[doc].splitlines()[1]
    ans, c2 = ask("answer", f"TICKET: {ticket}\nDOC: {doc}\nLINE: {line}")
    return Run("route", ans, [c1, c2], c1.latency + c2.latency), cat


routed = pd.DataFrame([(t, truth, route_reply(t)[1]) for t, truth in TICKETS], columns=["ticket", "truth", "routed"])
print("routing accuracy:", (routed.truth == routed.routed).mean())
print(pd.crosstab(routed.truth, routed.routed))
print(routed[routed.truth != routed.routed][["ticket", "truth", "routed"]].to_string(index=False))

# %% [markdown]
# *What just happened?* 9 of 10 routed correctly, and the one miss is exactly the kind you would meet in production: "arrived broken, I want a refund or a repair" matches words from three categories and a
# tie-break picked the wrong one. A real LLM router is usually better than keywords, but it fails in the same *shape*: ambiguous inputs that belong to two categories. Two defences: a confidence or margin
# threshold that sends close calls to a human (exercise 2), and a labelled test set that you re-run whenever the router prompt changes.

# %% [markdown]
# ## 3. Parallelisation: sectioning and voting
#
# **Shape:** run several LLM calls at the same time and combine them in code. Two variants:
# - **Sectioning:** split the work into *independent* subtasks (for example three separate guardrail checks on a draft) and run them together. Latency is the **slowest** branch, not the sum.
# - **Voting:** run the *same* task several times and take the majority. It buys reliability for a yes/no judgement at `n` times the cost.

# %%
ROLES["check_tone"] = lambda p, s: "FAIL: shouting" if ("!!!" in p or re.search(r"\b[A-Z]{5,}\b", p)) else "PASS"
ROLES["check_policy"] = lambda p, s: "FAIL: forbidden promise" if any(b in p.lower() for b in BANNED) else "PASS"
ROLES["check_pii"] = lambda p, s: "FAIL: contains personal data" if re.search(r"[\w.]+@[\w.]+|\b\d{13,16}\b", p) else "PASS"
CHECKS = ["check_tone", "check_policy", "check_pii"]


def guard_reply(draft, parallel=True):
    if parallel:
        with ThreadPoolExecutor(max_workers=len(CHECKS)) as ex:
            results = list(ex.map(lambda r: ask(r, draft), CHECKS))
    else:
        results = [ask(r, draft) for r in CHECKS]
    calls = [c for _, c in results]
    verdicts = {r: v for r, (v, _) in zip(CHECKS, results)}
    latency = max(c.latency for c in calls) if parallel else sum(c.latency for c in calls)
    return Run("sectioning", "PASS" if all(v == "PASS" for v in verdicts.values()) else str(verdicts), calls, latency)


good, bad = guard_reply("Your order 1000 has shipped."), guard_reply("We GUARANTEE delivery tomorrow!!! Mail me at bob@example.com")
print(good.answer, "|", bad.answer)
seq = guard_reply("Your order 1000 has shipped.", parallel=False)
print(f"modelled latency: parallel {good.latency:.2f}s vs sequential {seq.latency:.2f}s | calls: {len(good.calls)} vs {len(seq.calls)} | cost identical: {good.cost == seq.cost}")

# %% [markdown]
# Now voting. The mock `NoisyJudge` answers a yes/no question correctly with probability `p`, with an **independent** random error on every call (seeded, so the run is reproducible). With `n` votes,
# the majority is wrong only if more than half the votes are wrong, and for independent votes that probability is a binomial sum, so we can compare the simulation with exact arithmetic.

# %%
def majority_accuracy_formula(p, n):
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


class NoisyJudge:
    def __init__(self, p, seed=42):
        self.p, self.rng = p, np.random.default_rng(seed)

    def __call__(self, truth: int) -> int:
        return truth if self.rng.random() < self.p else 1 - truth


truths = np.random.default_rng(0).integers(0, 2, size=4000)
for n in (1, 3, 5, 9):
    judge = NoisyJudge(0.7)
    acc = np.mean([int(sum(judge(int(t)) for _ in range(n)) * 2 > n) == t for t in truths])
    print(f"n={n}: simulated accuracy {acc:.3f} | exact {majority_accuracy_formula(0.7, n):.3f} | cost x{n}")

# %% [markdown]
# *What just happened?* With a single judge at 70% accuracy, five independent votes give roughly 84% and nine roughly 90%, matching the formula, at 5x and 9x the cost. **Caveat you must remember:** this
# relies on *independent* errors. Repeated samples from the same real model on the same prompt tend to make correlated mistakes (the same ambiguity fools it every time), so real gains are smaller than this
# table. Voting helps most with genuinely different prompts, models or evidence, and you should measure it on your own labelled data rather than trusting the formula.

# %% [markdown]
# ## 4. Orchestrator-workers: the subtasks are not known in advance
#
# **Shape:** an *orchestrator* LLM decides how to split the request at run time, *workers* each do one piece, and a *synthesiser* combines the results. It differs from sectioning in one way: **the number
# and kind of subtasks depend on the input**, so you cannot hard-code the branches. Task: "Compare delivered-order revenue for DE, US and CA" produces three worker tasks; "...for FR" produces one.
# (Here revenue means the sum of `quantity * unit_price` over **delivered** orders of customers in that country.)

# %%
def revenue(country: str) -> float:
    return query("select round(sum(i.quantity * p.unit_price), 2) from orders o join customers c using(customer_id) join order_items i using(order_id) "
                 "join products p using(product_id) where o.status = 'delivered' and c.country = ?", (country,))[0][0] or 0.0


ROLES["orchestrate"] = lambda p, s: json.dumps([{"task": "delivered_revenue", "country": c} for c in dict.fromkeys(re.findall(r"\b[A-Z]{2}\b", p)) if c in COUNTRIES])
ROLES["worker"] = lambda p, s: f"{grab(r'country=(..)', p)}: {grab(r'RESULT=(.*)', p)}"
ROLES["synthesize"] = lambda p, s: (lambda rows: "Ranking: " + ", ".join(f"{c} ({v:,.2f})" for c, v in sorted(rows, key=lambda r: -r[1])))(
    [(m.group(1), float(m.group(2))) for m in re.finditer(r"(\w\w): ([\d.]+)", p)])


def orchestrated(question):
    plan_text, c0 = ask("orchestrate", question)
    subtasks = json.loads(plan_text)
    if not subtasks:
        return Run("orchestrator", "I could not identify any countries.", [c0], c0.latency, True), []

    def do(task):
        return ask("worker", f"TASK {task['task']} country={task['country']} RESULT={revenue(task['country'])}")

    with ThreadPoolExecutor(max_workers=len(subtasks)) as ex:
        worked = list(ex.map(do, subtasks))
    findings = [t for t, _ in worked]
    final, c2 = ask("synthesize", "\n".join(findings))
    calls = [c0] + [c for _, c in worked] + [c2]
    latency = c0.latency + max(c.latency for _, c in worked) + c2.latency
    return Run("orchestrator", final, calls, latency, True), subtasks


for q in ["Compare delivered-order revenue for DE, US and CA.", "What is the delivered revenue for FR?"]:
    run, subs = orchestrated(q)
    print(f"{len(subs)} subtask(s), {len(run.calls)} calls | {run.answer}")

# %% [markdown]
# Verify the numbers independently with pandas over the CSV files (not the SQL the tool used):

# %%
orders_df, items_df, prods_df, cust_df = (pd.read_csv(DATA / f"{n}.csv") for n in ("orders", "order_items", "products", "customers"))
rev = (orders_df[orders_df.status == "delivered"].merge(cust_df, on="customer_id").merge(items_df, on="order_id").merge(prods_df, on="product_id")
       .assign(v=lambda d: d.quantity * d.unit_price).groupby("country").v.sum().round(2))
run, _ = orchestrated("Compare delivered-order revenue for DE, US and CA.")
expected = "Ranking: " + ", ".join(f"{c} ({rev[c]:,.2f})" for c in sorted(["DE", "US", "CA"], key=lambda c: -rev[c]))
assert run.answer == expected, (run.answer, expected)
print("matches pandas:", expected)

# %% [markdown]
# *What just happened?* The plan itself was data (a JSON list) produced at run time, workers ran concurrently, and the final ranking matches an independent pandas calculation. This is the
# usual **multi-agent** shape in practice: one planner plus several focused workers, each with its own short context. The benefits are real (parallel speed-up, and each worker sees only what it needs,
# so contexts stay small); so are the costs: more calls, and the orchestrator can mis-split the work or a worker can mis-report, so the synthesiser only ever sees *summaries*.

# %% [markdown]
# ## 5. Evaluator-optimiser (reflection): generate, critique, revise
#
# **Shape:** a *writer* produces a draft, an *evaluator* judges it against a rubric and returns specific issues, and the writer revises, looping until the evaluator passes or a limit is hit.
# This is the concrete form of "reflection". It works when (a) you can state what "good" means as a rubric and (b) critiquing is easier than generating. The evaluator can be code (a
# validator, a test suite), which is the most reliable kind, or another LLM call, which is cheaper to write and less trustworthy. **Always bound the loop.**

# %%
ORDER = get_order(1000)
ISSUES = ["mention the order id", "remove the delivery guarantee", "state the order status"]


def writer_fn(p, s):
    draft = m.group(1) if (m := re.search(r"DRAFT: (.*)", p)) else "Your order is on its way. We guarantee delivery tomorrow!!!"
    feedback = (re.search(r"ISSUES: (.*)", p).group(1) if "ISSUES:" in p else "").split("|")
    fix = feedback[0]                                                # the mock fixes one issue per round, which makes the loop visible
    if fix == ISSUES[0]:
        draft = draft.replace("Your order", f"Your order {ORDER['order_id']}")
    elif fix == ISSUES[1]:
        draft = draft.replace("We guarantee delivery tomorrow!!!", "Delivery dates are estimates.")
    elif fix == ISSUES[2]:
        draft += f" Status: {ORDER['status']}."
    return draft


def evaluator_fn(p, s):
    text = re.search(r"DRAFT: (.*)", p).group(1)
    issues = [ISSUES[0]] if str(ORDER["order_id"]) not in text else []
    issues += [ISSUES[1]] if "guarantee" in text.lower() else []
    issues += [ISSUES[2]] if ORDER["status"] not in text else []
    return json.dumps({"pass": not issues, "issues": issues})


ROLES["writer"], ROLES["evaluator"] = writer_fn, lambda p, s: evaluator_fn(p, s)


def refine_reply(max_iters=5):
    calls, draft, feedback, history = [], "", "", []
    for i in range(max_iters):
        draft, c = ask("writer", (f"DRAFT: {draft}\n" if draft else "") + (f"ISSUES: {feedback}\n" if feedback else "") + f"ORDER: {json.dumps(ORDER)}")
        calls.append(c)
        verdict_text, c = ask("evaluator", f"DRAFT: {draft}")
        calls.append(c)
        verdict = json.loads(verdict_text)
        history.append((i + 1, draft, verdict["issues"]))
        if verdict["pass"]:
            return Run("evaluator-optimiser", draft, calls, sum(c.latency for c in calls)), history
        feedback = "|".join(verdict["issues"])
    return Run("evaluator-optimiser", draft, calls, sum(c.latency for c in calls)), history


run_eo, history = refine_reply()
for i, draft, issues in history:
    print(f"round {i}: issues={issues} | {draft}")

# %% [markdown]
# *What just happened?* Four rounds, eight calls (writer + evaluator each round), ending on a draft that passes the rubric. The cost is visible in the call count: a single-shot reply would be **one** call. Is
# the extra quality worth 8x the calls? Only if the rubric captures something you care about and single-shot fails it often enough. Also note what the loop can **not** do: if the evaluator is wrong or
# too lenient, you get a confidently "approved" bad draft. Prefer evaluators you can test (code, unit checks) and measure how often the LLM evaluator agrees with a human on a labelled sample.

# %% [markdown]
# ## 6. Planning: plan first, execute, replan on failure
#
# A chain has fixed steps; a ReAct-style agent (notebook 01) chooses one step at a time. **Plan-and-execute** sits between: the model writes the whole plan up front as data, code executes the
# steps, and the model is called again only to *replan* if a step fails. Benefits: the plan is inspectable before anything runs (you can show it to a user, or check it against an allow-list),
# and fewer model calls than step-by-step. Cost: an early wrong assumption bakes into the plan, and replanning needs an error signal.

# %%
ROLES["planner"] = lambda p, s: json.dumps(
    [{"tool": "get_order", "args": {"order_id": int(i)}} for i in re.findall(r"\b\d{3,}\b", p)]
    + ([{"tool": "search_kb", "args": {"q": p}}] if re.search(r"cancel|return|refund|warranty|shipping", p, re.I) else []))
ROLES["replan"] = lambda p, s: json.dumps([{"tool": "final", "args": {"answer": f"I could not complete the lookup ({grab(r'ERROR: (.*)', p)}). Please check the order number."}}])
TOOLBOX = {"get_order": lambda order_id: get_order(order_id), "search_kb": lambda q: search_kb(q)}


def plan_and_execute(question, max_replans=1):
    calls = []
    text, c = ask("planner", question)
    calls.append(c)
    plan, observations, replans = json.loads(text), [], 0
    while plan:
        step = plan.pop(0)
        if step["tool"] == "final":
            return step["args"]["answer"], calls, observations
        out = TOOLBOX[step["tool"]](**step["args"])
        if isinstance(out, dict) and "error" in out:
            if replans >= max_replans:
                return "Giving up: " + out["error"], calls, observations
            replans += 1
            text, c = ask("replan", f"QUESTION: {question}\nERROR: {out['error']}")
            calls.append(c)
            plan = json.loads(text)
            continue
        observations.append((step["tool"], out))
    return f"Collected {len(observations)} observation(s): " + "; ".join(f"{t}" for t, _ in observations), calls, observations


for q in ["Order 1000: is it cancellable, and what is the return window?", "What happened to order 99999999?"]:
    ans, calls, obs = plan_and_execute(q)
    print(f"{len(calls)} model call(s) | {ans}")

# %% [markdown]
# ## 7. Trade-offs, measured
#
# One table for all the patterns, built from the `Run` objects above plus a single-call baseline. Cost is for **1,000 runs** at the illustrative prices; latency is the modelled critical path;
# "model chooses path" marks where the run-time structure depended on model output (harder to test and bound).

# %%
ROLES["single"] = lambda p, s: f"Hello, your order {ORDER['order_id']} is {ORDER['status']}."
text, c = ask("single", "Write a reply about order 1000.")
single = Run("single call", text, [c], c.latency)
runs = [single, chain_reply("Where is order 1000? Can I still cancel it?"), route_reply(TICKETS[1][0])[0], guard_reply("Your order 1000 has shipped."),
        guard_reply("Your order 1000 has shipped.", parallel=False), orchestrated("Compare delivered-order revenue for DE, US and CA.")[0], run_eo]
names = ["single call", "chaining", "routing", "sectioning (parallel)", "sectioning (sequential)", "orchestrator-workers (3 tasks)", "evaluator-optimiser"]
table = pd.DataFrame({"pattern": names, "calls": [len(r.calls) for r in runs], "input_tokens": [sum(c.tin for c in r.calls) for r in runs],
                      "output_tokens": [sum(c.tout for c in r.calls) for r in runs], "cost_per_1000_runs_usd": [round(r.cost * 1000, 2) for r in runs],
                      "latency_s_modelled": [round(r.latency, 2) for r in runs], "model_chooses_path": [r.model_chooses_path for r in runs]})
print(table.to_string(index=False))

# %% [markdown]
# Read the table, not the marketing:
# - Every pattern beyond a single call multiplies **calls, tokens and cost**. Patterns that run calls in parallel (sectioning, orchestrator workers) recover *latency*, not cost.
# - Chains and evaluator loops add latency in proportion to their length; chains add reliability only if their gates check something real.
# - Orchestrator-workers and the open-ended agent loop are the two where the model decides the structure at run time: powerful, but you can no longer list every path in a test.
# - **Reliability** compounds downward with steps (`p^n`) and is pushed back up by gates, validators, voting over independent evidence, and human review.
# - A rule of thumb that survives contact with production: start with one good prompt, add a pattern only when you can **measure** that it fixes a failure, and stop adding when it stops paying for itself.
#
# **Multi-agent, honestly.** Splitting work across several LLM "agents" helps with (1) parallel breadth, (2) keeping each context small and focused, and (3) separating roles (writer versus critic). It hurts
# through extra cost, information lost in hand-offs (summaries drop detail), and new failure modes (two agents disagreeing, an orchestrator mis-splitting work). If a single loop with good tools solves the task, it is
# usually the better engineering choice.

# %% [markdown]
# ## Common mistakes
#
# 1. **Reaching for an agent (or multi-agent) when a chain with two gates would do.** Fixed paths are cheaper and testable.
# 2. **No gate between steps**, so one bad extraction flows through the whole chain and produces a confident wrong reply.
# 3. **Trusting the router without measuring it**, and no fallback for low-confidence or unknown categories.
# 4. **Voting on correlated samples** and expecting the textbook improvement; or voting on open-ended text (you need a comparable answer to take a majority of).
# 5. **Unbounded evaluator-optimiser loops**, and evaluators that approve everything (or nothing). Cap iterations, log the issues, and test the evaluator itself.
# 6. **Parallelising dependent work.** If task B needs task A's result, they are a chain; forcing them parallel changes the answer.
# 7. **Comparing patterns on cost alone or quality alone.** Decide on cost *per correct outcome* using a labelled set.

# %%
# Demonstration of mistake 5: an evaluator that never passes is stopped only by the iteration cap
ROLES["evaluator"] = lambda p, s: json.dumps({"pass": False, "issues": [ISSUES[0]]})
stuck, hist = refine_reply(max_iters=3)
print("rounds run:", len(hist), "| calls:", len(stuck.calls), "| stopped by the cap, not by success")
ROLES["evaluator"] = lambda p, s: evaluator_fn(p, s)                 # restore

# %% [markdown]
# ## Interview corner
#
# **1. Name the workflow patterns from Anthropic's "Building effective agents" and when you would use each.**
# <details><summary>Answer</summary>
# Prompt chaining (fixed sequential steps with gates, when a task decomposes cleanly), routing (classify then specialise, when categories are distinct), parallelisation (sectioning for independent subtasks,
# voting for reliability), orchestrator-workers (subtasks unknown in advance), and evaluator-optimiser (iterate against a clear rubric). Start with a single prompt and add a pattern only when you can measure the gain.
# </details>
#
# **2. Your support bot's routing step sends 8% of tickets to the wrong handler. What do you do?**
# <details><summary>Answer</summary>
# Build a labelled set from real tickets and look at the confusion matrix to see which categories collide. Fix the router prompt or definitions, add a confidence/margin threshold with a human fallback for
# ambiguous tickets, consider a fallback category, and re-run the labelled set on every change. Also check whether the handlers can recover from being sent a near-miss category.
# </details>
#
# **3. When does majority voting improve accuracy, and when does it not?**
# <details><summary>Answer</summary>
# It helps when individual judgements are better than chance and their errors are roughly independent: for p=0.7 and five independent votes the majority is right about 84% of the time. If errors are correlated
# (same model, same prompt, same ambiguity) the gain is much smaller. It multiplies cost, only works for answers you can compare, and is worth it only for high-stakes discrete decisions.
# </details>
#
# **4. Is a multi-agent system better than a single agent?**
# <details><summary>Answer</summary>
# Not by default. Multiple agents help with parallel breadth, context isolation and role separation, but add cost, hand-off information loss and coordination failures. Prefer a single agent with good tools until
# a measured limit (context size, latency, separation of duties) justifies splitting, and evaluate the whole system end to end.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Majority vote and its exact accuracy
# (a) Write `majority(labels)` returning the most common label; on a tie return the label whose **first occurrence is earliest**.
# (b) Write `vote_accuracy(p, n)` for a yes/no judge that is right with probability `p` independently on each of `n` votes (`n` odd): the probability the majority is right. The check compares it with a
# brute-force enumeration of all `2**n` outcomes, which is an independent method.
# *Hint: for (b) sum `comb(n, k) * p**k * (1-p)**(n-k)` over `k > n/2`.*

# %% tags=["starter"]
def majority(labels):
    ...


def vote_accuracy(p, n):
    ...


# %%
assert majority(["a", "b", "a"]) == "a" and majority(["x"]) == "x"
assert majority(["b", "a", "a", "b"]) == "b", "tie: earliest first occurrence wins"
assert majority(["a", "b", "c"]) == "a"
for p, n in [(0.7, 1), (0.7, 3), (0.6, 5), (0.9, 7)]:
    brute = sum(math.prod(p if bit else 1 - p for bit in bits) for bits in product([0, 1], repeat=n) if sum(bits) * 2 > n)
    assert abs(vote_accuracy(p, n) - brute) < 1e-12, f"vote_accuracy({p}, {n}) = {vote_accuracy(p, n)} but enumeration gives {brute}"
assert abs(vote_accuracy(0.7, 3) - 0.784) < 1e-9
assert vote_accuracy(0.5, 9) == 0.5 or abs(vote_accuracy(0.5, 9) - 0.5) < 1e-12, "a coin-flip judge stays a coin flip"
print("Exercise 1 passed")

# %% tags=["solution"]
def majority(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best = max(counts.values())
    return next(label for label in labels if counts[label] == best)


def vote_accuracy(p, n):
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))


# %% [markdown]
# ### Exercise 2: A router with a human fallback
# Write `route(scores, min_score=1, min_margin=1)` taking a dict of category scores (like `keyword_scores` returns) and returning the best category, or `"human"` when the best score is below `min_score`
# or the lead over the runner-up is below `min_margin`. Then compute `answered_accuracy` (accuracy on the tickets the router did **not** hand to a human, over `TICKETS`) and `coverage` (share of tickets answered by a
# category handler; tickets whose true label is `other` routed to `human` count as handled correctly, not as coverage).
# *Hint: use `keyword_scores(text)`; sort the values to get the best and second best.*

# %% tags=["starter"]
def route(scores, min_score=1, min_margin=1):
    ...


answered_accuracy = ...
coverage = ...

# %%
assert route({"a": 3, "b": 0}) == "a" and route({"a": 0, "b": 0}) == "human", "no evidence -> human"
assert route({"a": 2, "b": 2}) == "human", "tie -> human"
assert route({"a": 2, "b": 1}) == "a", "margin of 1 is enough by default"
assert route({"a": 2, "b": 1}, min_margin=2) == "human" and route({"a": 1, "b": 0}, min_score=2) == "human"
decisions = [(truth, route(keyword_scores(t))) for t, truth in TICKETS]
answered = [(truth, d) for truth, d in decisions if d != "human"]
exp_acc = sum(truth == d for truth, d in answered) / len(answered)
assert abs(answered_accuracy - exp_acc) < 1e-12, f"expected {exp_acc:.3f}"
assert abs(coverage - len(answered) / len(TICKETS)) < 1e-12
assert answered_accuracy == 1.0, "with the margin rule, every ticket the router answers should be routed correctly on this set"
print(f"Exercise 2 passed (answered accuracy {answered_accuracy:.2f}, coverage {coverage:.2f})")

# %% tags=["solution"]
def route(scores, min_score=1, min_margin=1):
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    (best, top), second = ordered[0], (ordered[1][1] if len(ordered) > 1 else 0)
    if top < min_score or top - second < min_margin:
        return "human"
    return best


decisions = [(truth, route(keyword_scores(t))) for t, truth in TICKETS]
answered = [(truth, d) for truth, d in decisions if d != "human"]
answered_accuracy = sum(truth == d for truth, d in answered) / len(answered)
coverage = len(answered) / len(TICKETS)


# %% [markdown]
# ### Exercise 3: A bounded refine loop
# Write `refine(writer, evaluator, max_iters=4)`. `writer(feedback)` returns a draft (`feedback` is `None` on the first call, afterwards the list of issues); `evaluator(draft)` returns a list of issues
# (empty means pass). Return a tuple `(draft, iterations, passed)`. Call the evaluator exactly once per draft, never call the writer more than `max_iters` times, and stop as soon as the evaluator returns no issues.

# %% tags=["starter"]
def refine(writer, evaluator, max_iters=4):
    ...


# %%
log = []
def w(feedback):
    log.append(("w", feedback))
    return f"draft{len([e for e in log if e[0] == 'w'])}"
def e(draft):
    log.append(("e", draft))
    return [] if draft == "draft3" else [f"fix-{draft}"]
assert refine(w, e) == ("draft3", 3, True), "should pass on the third draft"
assert [x for x in log if x[0] == "w"][0][1] is None and [x for x in log if x[0] == "w"][1][1] == ["fix-draft1"], "writer gets None first, then the issues"
assert sum(x[0] == "e" for x in log) == 3, "one evaluation per draft"
log.clear()
never = refine(lambda f: "same", lambda d: ["nope"], max_iters=3)
assert never == ("same", 3, False), f"must stop at the cap and report failure, got {never}"
assert refine(lambda f: "ok", lambda d: []) == ("ok", 1, True), "pass immediately when the first draft is fine"
print("Exercise 3 passed")

# %% tags=["solution"]
def refine(writer, evaluator, max_iters=4):
    feedback, draft = None, ""
    for i in range(1, max_iters + 1):
        draft = writer(feedback)
        issues = evaluator(draft)
        if not issues:
            return draft, i, True
        feedback = issues
    return draft, max_iters, False


# %% [markdown]
# ### Exercise 4 (open-ended): Choose the pattern
# Take three tasks from your own work (for example: "triage incoming alerts", "write release notes from merged PRs", "answer questions about our runbooks"). For each, pick the **simplest** pattern from this notebook that could
# work, name the gate or rubric you would use, estimate calls per task and what a failure looks like, and say what labelled data you would need to measure it. If you chose multi-agent or an open-ended agent loop,
# defend why a chain or single call was not enough. There is no assert for this one.

# %% [markdown]
# ## Recap
# - Prefer the simplest structure: single call, then chaining with gates, routing, parallelisation, orchestrator-workers, evaluator-optimiser, and only then an open-ended agent.
# - Gates (code checks between steps) and bounded loops are what make LLM pipelines reliable; chains multiply unreliability (`p^n`), gates and validators push it back up.
# - Parallel branches cut latency, not cost; voting needs independent errors; orchestrator-workers fit when subtasks are discovered at run time; evaluator-optimiser fits when "good" can be written as a rubric.
# - Plan-and-execute makes the plan inspectable and replans only on failure; "reflection" is an evaluator-optimiser loop with a critique step.
# - Multi-agent systems buy parallelism and context isolation and pay in cost, hand-off loss and coordination failures. Measure calls, tokens, cost and latency per **correct** outcome.
#
# **Go deeper:** Anthropic's engineering article "Building effective agents" (anthropic.com/engineering), and the agent and tool-use sections of docs.anthropic.com.
#
# **Next:** [Agent safety, guardrails and evaluation](04_agent_safety_guardrails_and_evaluation.ipynb)
