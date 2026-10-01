# %% [markdown]
# # Orchestration and DAGs
#
# **Why this matters.** Once you have more than one pipeline, "run this script at 2 a.m. with cron" stops working: task B needs task A's output, one step fails and you want to retry just that step, someone needs
# to re-run last March, and nobody can tell which jobs are late. An **orchestrator** (Airflow, Prefect, Dagster and friends) handles scheduling, dependencies, retries, backfills and visibility. Interviewers ask about Airflow concepts
# constantly, and the fastest way to really understand them is to build a tiny orchestrator yourself. That is what you do here, then you read a real Airflow DAG file with confidence.
#
# **You will learn**
# - what an orchestrator does, and the concepts: DAG, task, schedule, logical date, retries, trigger rules, backfill/catchup
# - build a **pure-Python DAG runner**: topological ordering, cycle detection, parallel execution, retries, failure propagation, resume
# - write a correct **Airflow DAG file** (shown, not run) and analyse it with `ast` without importing Airflow
# - compare Airflow, Prefect and Dagster honestly
#
# **Prerequisites:** [Pipeline design and idempotency](01_pipeline_design_and_idempotency.ipynb) and [Build an ETL pipeline](02_build_an_etl_pipeline_in_python.ipynb).
#
# > Data used: the synthetic shop CSVs. Airflow, Prefect and Dagster are **not installed** here, so their code appears in `no-run` cells; every idea is also implemented in runnable form.

# %%
import ast
import textwrap
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import pandas as pd

DATA = Path("../datasets")

# %% [markdown]
# ## 1. What an orchestrator gives you
#
# | Need | Without orchestrator | With orchestrator |
# |---|---|---|
# | Run daily at 02:00 | cron on one machine | schedule per pipeline, on a cluster |
# | B after A succeeds | a shell script that hopes | declared dependencies |
# | A step fails | whole script dies, rerun everything | retry that task, skip or hold downstream |
# | Rerun last March | edit dates in a script by hand | **backfill** the date range, one run per interval |
# | "Is it late?" | someone notices a stale dashboard | run history, SLAs, alerts |
# | Which logs? | grep on a server | logs per task attempt |
#
# The orchestrator **does not do the data work**: it decides *what runs when* and records what happened. Heavy work runs in your code or on external systems (warehouse, Spark, containers).
#
# **Vocabulary (Airflow terms, which the others echo)**
# * **DAG**: the set of tasks and their dependencies. **Task**: one unit of work. **Operator**: a task template (run SQL, run Python, call an API); with the TaskFlow API a Python function decorated with `@task` is a task.
# * **Schedule and logical date**: a daily DAG produces one **run per data interval**. The run for the interval `[2025-01-01, 2025-01-02)` has logical date `2025-01-01` and starts **after** that interval has ended.
#   Your code should compute *for that date*, never for "now".
#   (This is the classic data-interval behaviour. Airflow 3 changed the default for plain cron-style schedules such as `@daily` to a trigger-style timetable that runs *at* the cron time; the data-interval
#   behaviour is still available. Check the timetable docs for the version you deploy. The discipline is the same either way: derive everything from the run's date, not the clock.)
# * **Catchup / backfill**: create the runs for past intervals that never ran.
# * **Retries** with delay; **trigger rules** ("run only if all upstream succeeded" is the default; `all_done` runs a cleanup or notification whatever happened).
# * **XCom**: tiny values passed between tasks. Pass *paths or ids*, never DataFrames.
# * **Sensor**: a task that waits for something (a file, a partition) to appear.

# %% [markdown]
# ## 2. Build a DAG runner
# We use the standard-library `graphlib.TopologicalSorter`. It gives us cycle detection and an iterator of tasks that are **ready** (all their dependencies finished), which is exactly what a scheduler needs.
# Start with the ordering and the classic failure, a cycle:

# %%
deps = {"load": {"transform"}, "transform": {"extract_a", "extract_b"}, "extract_a": set(), "extract_b": set()}
print(list(TopologicalSorter(deps).static_order()))

deps_cyclic = {"a": {"c"}, "b": {"a"}, "c": {"b"}}                   # a needs c, c needs b, b needs a
try:
    TopologicalSorter(deps_cyclic).prepare()
except CycleError as e:
    print("CycleError, cycle:", e.args[1])

# %% [markdown]
# Now the runner. Read the class once; the demo afterwards is what matters. Design choices worth noticing:
# * each task is a function `fn(ctx)`; `ctx.logical_date` is the date being processed, `ctx.inputs[name]` the return value of an upstream task (our tiny XCom);
# * a task runs in a **thread pool** as soon as its dependencies finish, so independent tasks run in parallel;
# * a failed task is retried, and if it still fails, downstream tasks get the state `upstream_failed` and do **not** run (except tasks with trigger rule `all_done`);
# * `run()` can take the results of an earlier run and **resume**: tasks that already succeeded are not run again (this is what "clear failed tasks and re-run" does in Airflow).

# %%
@dataclass
class Ctx:
    logical_date: str
    inputs: dict = field(default_factory=dict)
    attempt: int = 1


@dataclass
class TaskRun:
    state: str = "pending"            # success | failed | upstream_failed | pending
    attempts: int = 0
    seconds: float = 0.0
    result: object = None
    error: str = ""
    order: int = -1                   # order in which the task finished


@dataclass
class TaskSpec:
    name: str
    fn: object
    deps: tuple = ()
    retries: int = 0
    trigger_rule: str = "all_success"      # or "all_done"


class MiniDAG:
    def __init__(self, name):
        self.name, self.tasks = name, {}

    def task(self, deps=(), retries=0, trigger_rule="all_success"):
        def deco(fn):
            self.tasks[fn.__name__] = TaskSpec(fn.__name__, fn, tuple(deps), retries, trigger_rule)
            return fn
        return deco

    def graph(self):
        return {n: set(t.deps) for n, t in self.tasks.items()}

    def validate(self):
        unknown = {d for t in self.tasks.values() for d in t.deps} - set(self.tasks)
        if unknown:
            raise ValueError(f"unknown dependencies: {sorted(unknown)}")
        TopologicalSorter(self.graph()).prepare()          # raises CycleError on a cycle

    def _execute(self, spec, ctx):
        t0, run = time.perf_counter(), TaskRun()
        for attempt in range(1, spec.retries + 2):
            run.attempts, ctx.attempt = attempt, attempt
            try:
                run.result, run.state, run.error = spec.fn(ctx), "success", ""
                break
            except Exception as exc:                       # noqa: BLE001 - an orchestrator must catch everything a task raises
                run.state, run.error = "failed", f"{type(exc).__name__}: {exc}"
        run.seconds = round(time.perf_counter() - t0, 3)
        return run

    def run(self, logical_date, max_workers=4, resume_from=None):
        self.validate()
        results = {n: TaskRun() for n in self.tasks}
        ts = TopologicalSorter(self.graph())
        ts.prepare()
        finished = 0
        with ThreadPoolExecutor(max_workers) as pool:
            futures = {}
            while ts.is_active():
                for name in ts.get_ready():
                    spec = self.tasks[name]
                    up = [results[d].state for d in spec.deps]
                    if resume_from and resume_from.get(name) and resume_from[name].state == "success":
                        results[name] = resume_from[name]               # already done in an earlier run: reuse
                        ts.done(name)
                    elif spec.trigger_rule == "all_success" and any(s != "success" for s in up):
                        results[name].state = "upstream_failed"
                        ts.done(name)
                    else:
                        ctx = Ctx(logical_date, {d: results[d].result for d in spec.deps})
                        futures[pool.submit(self._execute, spec, ctx)] = name
                if futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done:
                        name = futures.pop(fut)
                        results[name] = fut.result()
                        results[name].order = finished
                        finished += 1
                        ts.done(name)
        return results


def summary(results):
    return pd.DataFrame([{"task": n, "state": r.state, "attempts": r.attempts, "seconds": r.seconds, "finished_order": r.order}
                         for n, r in results.items()]).sort_values("finished_order").reset_index(drop=True)

# %% [markdown]
# ### A real (small) DAG with our runner
# The shop pipeline as tasks. Each task does actual work on the CSVs and passes small values (row counts, paths, numbers) to its children. `FAIL_PLAN` lets us inject failures.

# %%
FAIL_PLAN = {}          # task name -> number of attempts that should fail before it succeeds (999 = always)
SEEN = {}

def maybe_fail(name):
    SEEN[name] = SEEN.get(name, 0) + 1
    if SEEN[name] <= FAIL_PLAN.get(name, 0):
        raise ConnectionError(f"{name}: simulated transient failure #{SEEN[name]}")

dag = MiniDAG("shop_daily")

@dag.task()
def extract_orders(ctx):
    time.sleep(0.2)                                  # pretend network I/O
    maybe_fail("extract_orders")
    return len(pd.read_csv(DATA / "orders.csv"))

@dag.task()
def extract_items(ctx):
    time.sleep(0.2)
    return len(pd.read_csv(DATA / "order_items.csv"))

@dag.task(deps=["extract_orders", "extract_items"])
def clean(ctx):
    return {"orders": ctx.inputs["extract_orders"], "items": ctx.inputs["extract_items"]}

@dag.task(deps=["clean"], retries=2)
def daily_revenue(ctx):
    maybe_fail("daily_revenue")
    o, i, p = (pd.read_csv(DATA / f"{n}.csv") for n in ("orders", "order_items", "products"))
    j = i.merge(o, on="order_id").merge(p, on="product_id").query("status == 'delivered'")
    return round(float((j["quantity"] * j["unit_price"]).sum()), 2)

@dag.task(deps=["clean"])
def customer_summary(ctx):
    return int(pd.read_csv(DATA / "orders.csv")["customer_id"].nunique())

@dag.task(deps=["daily_revenue", "customer_summary"])
def quality_check(ctx):
    maybe_fail("quality_check")
    if ctx.inputs["daily_revenue"] <= 0:
        raise ValueError("revenue must be positive")
    return "ok"

@dag.task(deps=["quality_check"])
def publish(ctx):
    return f"published revenue {ctx.inputs['quality_check']} for {ctx.logical_date}"

@dag.task(deps=["publish"], trigger_rule="all_done")
def notify(ctx):
    return "notified"

dag.validate()
t0 = time.perf_counter()
res = dag.run("2025-05-30")
wall = time.perf_counter() - t0
print(f"wall time {wall:.2f}s | sum of task times {sum(r.seconds for r in res.values()):.2f}s")
summary(res)

# %% [markdown]
# *What just happened?* The two extracts each sleep 0.2 s but the wall time printed above is well below the sum of the task times (the two extracts overlapped), because the runner started both as soon as they were ready (they have no dependencies). The
# `finished_order` column shows the dependency order was respected, and `publish` received the revenue via `ctx.inputs`.

# %% [markdown]
# ### Retries, failure propagation and resume
# Now break things. First, `daily_revenue` fails twice and then works: the `retries=2` covers it.

# %%
SEEN.clear(); FAIL_PLAN.clear(); FAIL_PLAN["daily_revenue"] = 2
res = dag.run("2025-05-30")
summary(res)[["task", "state", "attempts"]].query("task == 'daily_revenue'")

# %% [markdown]
# Second, a **permanent** failure in `quality_check`. Watch what happens downstream: `publish` must **not** run (never publish unchecked data), while `notify` (trigger rule `all_done`) does run, so a human hears about it.

# %%
SEEN.clear(); FAIL_PLAN.clear(); FAIL_PLAN["quality_check"] = 999
broken = dag.run("2025-05-30")
summary(broken)[["task", "state", "attempts"]]

# %% [markdown]
# Now "fix the bug" and **resume** from the failed run: only the failed task and what depends on it should execute, everything that already succeeded is reused.

# %%
SEEN.clear(); FAIL_PLAN.clear()
resumed = dag.run("2025-05-30", resume_from=broken)
s = summary(resumed)
print("states:", dict(zip(s["task"], s["state"])))
print("tasks that actually executed this time (attempts recorded by THIS run):",
      [n for n, r in resumed.items() if r is not broken[n]])

# %% [markdown]
# Resume works because each task is idempotent, the same design requirement as in the previous notebooks. If tasks were not safe to re-run, "retry the failed task" would be a gamble.

# %% [markdown]
# ## 3. Schedules, logical dates and backfills
# A daily DAG creates one run per day. A **backfill** is nothing more than "create the runs for these intervals and execute them", each with its own logical date. Because our tasks take `ctx.logical_date`
# and do not look at the clock, running the last 7 days in any order gives the same results as running them daily.

# %%
def schedule_runs(start, now):
    """Daily logical dates whose data interval [d, d+1) has ENDED by `now` (a run happens after its interval closes)."""
    starts = pd.date_range(start, now, freq="D", inclusive="left")          # interval starts strictly before now
    return [d.strftime("%Y-%m-%d") for d in starts if d + pd.Timedelta(days=1) <= pd.Timestamp(now)]


print(schedule_runs("2025-05-24", "2025-05-31"))

# %%
backfill = {d: dag.run(d) for d in schedule_runs("2025-05-27", "2025-05-31")}
print({d: all(r.state == "success" for r in run.values()) for d, run in backfill.items()})

# %% [markdown]
# Note `now` is passed in: the last logical date is `2025-05-30` because the interval `[05-30, 05-31)` has just closed. This mirrors Airflow's rule that a run *for* a day starts *after* the day is over, and is
# the number-one source of confusion for newcomers ("why did my 05-30 run start on 05-31?"). Also note `catchup=False` in a DAG file means "do not create runs for missed intervals when the DAG is enabled";
# use an explicit backfill command instead when you want history.

# %% [markdown]
# ## 4. A real Airflow DAG file (not runnable here)
# Airflow is not installed in this environment, so we never import or run this file. It is the same shop pipeline written with Airflow's TaskFlow API, stored in a string so we can analyse it. Two things to notice: the *DAG file is Python that Airflow imports repeatedly*
# (so it must be cheap to import: no network calls or big reads at module level), and tasks return small values such as paths. In Airflow 3 the imports come from `airflow.sdk`; in Airflow 2 from `airflow.decorators`.
# The file handles both. Check the docs of the version you deploy, since the API keeps evolving.

# %% [markdown]
# We cannot import it, but we **can** read it. The next cells use the `ast` module (Python's own parser) to verify the file is valid Python, list its tasks, recover the dependency graph, and flag
# common Airflow mistakes: the same kind of check a CI job runs on DAG files before deployment. (Airflow itself ships a `dag.test()` and a `DagBag` import check for the real thing.)

# %%
AIRFLOW_DAG = textwrap.dedent('''
"""shop_daily: extract -> transform -> validate -> load, one run per day."""
from datetime import datetime, timedelta, timezone

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task


@dag(
    dag_id="shop_daily",
    schedule="@daily",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "data-team", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["shop", "etl"],
)
def shop_daily():
    @task
    def extract(**context) -> str:
        day = context["ds"]  # the logical date as YYYY-MM-DD
        return f"raw/orders/date={day}/"

    @task
    def transform(raw_path: str) -> str:
        return raw_path.replace("raw", "clean")

    @task
    def validate(clean_path: str) -> str:
        return clean_path

    @task
    def load(clean_path: str) -> None:
        pass

    raw = extract()
    clean = transform(raw)
    checked = validate(clean)
    load(checked)


shop_daily()
''')
tree = ast.parse(AIRFLOW_DAG)                       # SyntaxError here would mean the file is broken
print("valid Python, top-level statements:", len(tree.body))


def dag_structure(source):
    """Recover task names and dependencies from a TaskFlow-style DAG file without importing it."""
    tree = ast.parse(source)
    dag_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and any(getattr(getattr(d, "func", d), "id", None) == "dag" for d in n.decorator_list))
    tasks = [n.name for n in dag_fn.body if isinstance(n, ast.FunctionDef)
             and any(getattr(d, "id", None) == "task" for d in n.decorator_list)]
    produced_by, deps = {}, {t: set() for t in tasks}
    for stmt in dag_fn.body:
        call = stmt.value if isinstance(stmt, (ast.Assign, ast.Expr)) else None
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) in tasks:
            name = call.func.id
            deps[name] |= {produced_by[a.id] for a in call.args if isinstance(a, ast.Name) and a.id in produced_by}
            if isinstance(stmt, ast.Assign):
                produced_by[stmt.targets[0].id] = name
    return dag_fn.name, deps


name, deps_found = dag_structure(AIRFLOW_DAG)
print(name, deps_found)
print("execution order Airflow would use:", list(TopologicalSorter(deps_found).static_order()))

# %%
def lint_dag_source(source):
    """A few of the most common Airflow DAG-file mistakes, found statically."""
    tree, problems = ast.parse(source), []
    kw = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "dag":
            kw = {k.arg: k.value for k in n.keywords}
    if "catchup" not in kw:
        problems.append("catchup not set explicitly (default depends on config; make the intent visible)")
    if "start_date" not in kw:
        problems.append("no start_date")
    elif any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") in ("now", "utcnow", "today") for c in ast.walk(kw["start_date"])):
        problems.append("dynamic start_date (datetime.now()): the schedule then moves every time the file is parsed")
    top_level_calls = [n for stmt in tree.body if isinstance(stmt, (ast.Expr, ast.Assign)) for n in ast.walk(stmt) if isinstance(n, ast.Call)]
    for c in top_level_calls:
        f = c.func
        if isinstance(f, ast.Attribute) and getattr(f.value, "id", "") in ("requests", "pd", "pandas", "httpx"):
            problems.append(f"heavy call at import time: {getattr(f.value, 'id', '')}.{f.attr}() runs on every DAG parse")
    return problems


print("our DAG file:", lint_dag_source(AIRFLOW_DAG) or "no problems found")
bad_dag = '''
from datetime import datetime
import pandas as pd
from airflow.decorators import dag
df = pd.read_csv("s3://bucket/huge.csv")
pd.read_csv("s3://bucket/huge.csv")
@dag(dag_id="bad", schedule="@daily", start_date=datetime.now())
def bad(): pass
'''
for p in lint_dag_source(bad_dag):
    print("BAD DAG:", p)

# %% tags=["no-run"]
# With Airflow installed, you would validate and run the real file from a shell:
#   airflow dags list-import-errors                 # does every DAG file import?
#   airflow dags test shop_daily 2025-05-30          # run one logical date locally, in-process
#   airflow dags backfill shop_daily -s 2025-01-01 -e 2025-01-31    # explicit backfill in Airflow 2; Airflow 3 moved it to a separate `airflow backfill create ...` command (verify in the CLI docs of your version)

# %% [markdown]
# ## 5. Airflow, Prefect, Dagster: how they differ
# All three schedule and monitor Python workflows. The differences are about *what you model* and *how you deploy*. This is a fair one-page summary, not a ranking; check current docs before deciding.
#
# | | Airflow | Prefect | Dagster |
# |---|---|---|---|
# | Core idea | DAGs of tasks on a schedule | Python functions decorated as flows and tasks; dynamic, code-first | **assets**: you declare the *data objects* (tables, files) and their upstream assets |
# | Strength | huge ecosystem of provider integrations, widely used, mature | very little ceremony, dynamic workflows and quick local start | lineage, data-aware scheduling, testability |
# | Watch out for | heavier to operate, DAG-file parsing model, time-based mental model | smaller community than Airflow | a different mental model to learn |
#
# Which to learn first? Airflow concepts (schedule, logical date, retries, backfill, sensors) transfer to everything else, and it is the most common in job ads, so know it well. The same pipeline as Prefect and Dagster (again `no-run`, and simplified):

# %% tags=["no-run"]
# --- Prefect (pip install prefect) ---
from prefect import flow, task

@task(retries=2, retry_delay_seconds=60)
def extract(day: str) -> str: ...

@task
def transform(raw_path: str) -> str: ...

@task
def load(path: str) -> None: ...

@flow
def shop_daily(day: str):
    load(transform(extract(day)))          # dependencies are just function calls

# %% tags=["no-run"]
# --- Dagster (pip install dagster) ---
from dagster import asset

@asset
def raw_orders(): ...

@asset
def clean_orders(raw_orders):              # the argument name declares the upstream asset
    ...

@asset
def daily_revenue(clean_orders): ...

# %% [markdown]
# The runnable `MiniDAG` above corresponds to all of them: tasks, dependencies, retries, a run state per task. What real orchestrators add on top is persistence (a metadata database),
# distributed workers, a UI, alerting, calendars/cron schedules, and pools/queues to limit concurrency.

# %% [markdown]
# ## Common mistakes
#
# 1. **Using `datetime.now()` inside tasks** instead of the logical date; backfills and retries then compute the wrong day.
# 2. **Heavy work at DAG-file import time** (reading files, calling APIs). The scheduler re-parses the file constantly; keep the top level cheap.
# 3. **Passing big data through XCom** (or the runner's `ctx.inputs`). Pass paths, ids and counts; keep data in storage.
# 4. **Non-idempotent tasks** with retries enabled: a retry then double-loads. Make each task safe to run twice first.
# 5. **Off-by-one on run dates**: the run for day D executes after D ends. Design for "process D" not "process now".
# 6. **Everything in one giant task.** Tiny tasks give partial retries and clear failure locations; too many trivial tasks add scheduling overhead. Aim for steps that are meaningful units of recovery.
# 7. **Treating the orchestrator as a compute engine.** It should trigger and monitor work, not run a 10 GB pandas job on the scheduler machine.

# %% [markdown]
# ## Interview corner
#
# **1. What is a DAG and why must it be acyclic?**
# <details><summary>Answer</summary>
# A directed graph of tasks and dependencies. If it had a cycle (A needs C, C needs B, B needs A) no task could ever be the first to run, so there is no valid execution order. Orchestrators detect cycles at parse time.
# </details>
#
# **2. Explain Airflow's logical date / data interval.**
# <details><summary>Answer</summary>
# Each run represents a data interval; the logical date marks its start. The run executes after the interval ends, so the run for 2025-05-30 starts on 2025-05-31. Tasks should process data for their interval, not for "now",
# which makes runs reproducible and backfillable.
# </details>
#
# **3. A task failed in the middle of the DAG at 3 a.m. What should happen?**
# <details><summary>Answer</summary>
# Retry with backoff for transient errors; if it still fails, mark downstream tasks as upstream_failed so bad or partial data is not published, alert the owner, and run only cleanup/notify tasks (trigger rule all_done).
# After the fix, clear the failed task so it and its downstream re-run; this is safe because tasks are idempotent.
# </details>
#
# **4. How would you backfill six months of data?**
# <details><summary>Answer</summary>
# Use the orchestrator's backfill (one run per interval) with limited concurrency so you do not overwhelm the source or warehouse; ensure tasks are parameterised by logical date and idempotent; validate a small range first; monitor
# row counts per partition afterwards.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Extend the DAG
# Add a task `product_summary` to `dag2` (a copy of the structure is created below) that depends on `clean` and returns the number of **distinct products sold** (distinct `product_id` in `order_items.csv`). Also add a task `report` that depends on `product_summary` and `daily_revenue`
# and returns the string `"revenue=<daily_revenue>, products=<n>"`. Then run `dag2` and store the run results in `res2`.

# %%
def build_base_dag():
    d = MiniDAG("shop_daily_v2")
    @d.task()
    def extract_orders(ctx): return len(pd.read_csv(DATA / "orders.csv"))
    @d.task()
    def extract_items(ctx): return len(pd.read_csv(DATA / "order_items.csv"))
    @d.task(deps=["extract_orders", "extract_items"])
    def clean(ctx): return {"orders": ctx.inputs["extract_orders"], "items": ctx.inputs["extract_items"]}
    @d.task(deps=["clean"])
    def daily_revenue(ctx):
        o, i, p = (pd.read_csv(DATA / f"{n}.csv") for n in ("orders", "order_items", "products"))
        j = i.merge(o, on="order_id").merge(p, on="product_id").query("status == 'delivered'")
        return round(float((j["quantity"] * j["unit_price"]).sum()), 2)
    return d

dag2 = build_base_dag()

# %% tags=["starter"]
# add the two tasks to dag2 with @dag2.task(deps=[...]) and run it
res2 = ...

# %%
n_products = pd.read_csv(DATA / "order_items.csv")["product_id"].nunique()
rev = res2["daily_revenue"].result
assert res2["product_summary"].state == "success" and res2["product_summary"].result == n_products, f"product_summary should return {n_products}"
assert res2["report"].result == f"revenue={rev}, products={n_products}", f"report should be 'revenue={rev}, products={n_products}', got {res2['report'].result!r}"
assert res2["report"].order > res2["product_summary"].order and res2["report"].order > res2["daily_revenue"].order, "report must finish after both of its dependencies"
print("Exercise 1 passed:", res2["report"].result)

# %% tags=["solution"]
@dag2.task(deps=["clean"])
def product_summary(ctx):
    return int(pd.read_csv(DATA / "order_items.csv")["product_id"].nunique())

@dag2.task(deps=["product_summary", "daily_revenue"])
def report(ctx):
    return f"revenue={ctx.inputs['daily_revenue']}, products={ctx.inputs['product_summary']}"

res2 = dag2.run("2025-05-30")

# %% [markdown]
# ### Exercise 2: What is affected if a task fails?
# Write `downstream_of(graph, task)` returning the **set of all tasks that transitively depend on** `task` (not including itself), where `graph` maps each task to the set of tasks it depends on (like `dag.graph()`).
# This is the "blast radius" an on-call engineer wants to know.

# %% tags=["starter"]
def downstream_of(graph, task):
    ...

# %%
g = dag.graph()
assert downstream_of(g, "extract_orders") == {"clean", "daily_revenue", "customer_summary", "quality_check", "publish", "notify"}, "extract_orders feeds everything except extract_items"
assert downstream_of(g, "customer_summary") == {"quality_check", "publish", "notify"}
assert downstream_of(g, "notify") == set(), "nothing depends on notify"
assert downstream_of({"a": set(), "b": {"a"}, "c": {"b"}, "d": {"a"}}, "a") == {"b", "c", "d"}
print("Exercise 2 passed")

# %% tags=["solution"]
def downstream_of(graph, task):
    out, frontier = set(), {task}
    while frontier:
        nxt = {t for t, ds in graph.items() if ds & frontier} - out
        out |= nxt
        frontier = nxt
    return out

# %% [markdown]
# ### Exercise 3: Only re-run what is needed
# Using `broken` (the failed run from section 2, where `quality_check` failed permanently), compute `must_rerun`: the set of task names that a resume must actually execute
# (tasks whose state is **not** `success`). Then compute `reused`: the set of task names that a resume can reuse.

# %% tags=["starter"]
must_rerun = ...
reused = ...

# %%
assert must_rerun == {n for n, r in broken.items() if r.state != "success"}, "tasks that are not 'success' must be re-executed"
assert reused == {n for n, r in broken.items() if r.state == "success"}
assert "quality_check" in must_rerun and "publish" in must_rerun and "extract_orders" in reused
assert must_rerun | reused == set(broken) and not (must_rerun & reused)
print("Exercise 3 passed: re-run", sorted(must_rerun), "| reuse", sorted(reused))

# %% tags=["solution"]
must_rerun = {n for n, r in broken.items() if r.state != "success"}
reused = {n for n, r in broken.items() if r.state == "success"}

# %% [markdown]
# ### Exercise 4: How many runs does a backfill create?
# Using `schedule_runs`, compute `n_runs_2024` = the number of daily runs needed to backfill **all of calendar year 2024** if `now` is `2025-01-01`, and `n_runs_partial` = the runs created when `now` is `2025-05-31` starting from `2025-05-01`.
# Think about why the second is 30 and not 31.

# %% tags=["starter"]
n_runs_2024 = ...
n_runs_partial = ...

# %%
assert n_runs_2024 == len(pd.date_range("2024-01-01", "2024-12-31")), "2024 is a leap year: 366 daily runs"
assert n_runs_partial == 30, "the run for 05-31 has not ended yet when now = 05-31 00:00, so only 05-01 .. 05-30 exist"
print("Exercise 4 passed:", n_runs_2024, n_runs_partial)

# %% tags=["solution"]
n_runs_2024 = len(schedule_runs("2024-01-01", "2025-01-01"))
n_runs_partial = len(schedule_runs("2025-05-01", "2025-05-31"))

# %% [markdown]
# ### Exercise 5 (open-ended): Model your own workflow
# Pick a process you know (a weekly report, a model retraining job, an ETL from work) and write its `MiniDAG`: at least 5 tasks, one parallel branch, one task that should use `all_done`, and one that needs retries. Inject a failure with
# `FAIL_PLAN` and explain, in a markdown cell, what you expect to happen before you run it.

# %% [markdown]
# ## Recap
# - An orchestrator schedules and monitors: dependencies, retries, backfills, run history. The heavy data work happens elsewhere.
# - A **DAG** is tasks plus dependencies with no cycles; `graphlib.TopologicalSorter` gives the order, the ready set and cycle detection.
# - Failure semantics matter: retry transient errors, mark downstream `upstream_failed`, never publish after a failed check, run cleanup/notify with `all_done`, **resume** re-running only failed and downstream tasks.
# - **Logical date**: a run processes its data interval and starts after it ends; tasks must never use "now".
# - A DAG file must stay cheap to import; analyse it statically with `ast` in CI. Airflow, Prefect and Dagster share the concepts; Airflow's are the most transferable.
#
# **Go deeper:** the Apache Airflow documentation (airflow.apache.org/docs), especially "Core Concepts" (DAGs, Tasks, XComs, Trigger Rules) and the "TaskFlow" tutorial; the Prefect docs (docs.prefect.io) and the Dagster docs (docs.dagster.io).
#
# **Next:** [Incremental loads, backfills and CDC](04_incremental_loads_backfills_and_cdc.ipynb)
