# %% [markdown]
# # Data Quality and Validation
#
# **Why this matters.** Most production data incidents are not crashes; they are *quiet*: a column that
# silently became half empty, a new spelling of a category, a join that started duplicating rows. Nobody gets paged, the dashboard just
# shows a wrong number for two weeks. The fix is to write down what "good data" means as executable checks, run them on every load,
# and decide in advance what happens when they fail. You will build a small version of the frameworks that teams use for this
# (Great Expectations, Soda, dbt tests, pandera) so you understand what they do under the hood.
#
# **You will learn**
# - the dimensions of data quality: schema, completeness, uniqueness, validity, referential integrity, freshness
# - write reusable checks that return a structured report instead of a bare `assert`
# - validate rows with pydantic models, and separate validating from cleaning
# - run the same suite on `churn.csv`, `messy.csv` and `shop.db` and read the failures
# - quarantine bad rows instead of crashing or ignoring them
#
# **Prerequisites:** [Data modelling](03_data_modeling_star_schema_and_normalization.ipynb) (keys, grain) and [Cleaning data](../05_Pandas/).
#
# > Data used: `churn.csv`, `messy.csv` (both have planted problems), and `shop.db`. See `docs/DATASETS.md`.

# %%
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

DATA = Path("../datasets")

# %% [markdown]
# ## 1. The six questions
#
# | Dimension | The question | Example failure |
# |---|---|---|
# | **Schema** | Are the expected columns present, with the right types? | producer renamed `tenure` to `tenure_months` |
# | **Completeness** | How much is missing? | `total_charges` empty for 3% of rows |
# | **Uniqueness** | Is the key really unique? | the same customer loaded twice |
# | **Validity** | Are values in the allowed set or range? | age = -1, `month-to-month` vs `Month-to-month` |
# | **Referential integrity** | Does every foreign key have a parent? | order for a customer that does not exist |
# | **Freshness / volume** | Is the data recent and roughly the expected size? | yesterday's load never ran |
#
# A first gotcha, before any framework: how you **read** the file is itself a quality decision.

# %%
churn_default = pd.read_csv(DATA / "churn.csv")
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
print("internet_service missing with defaults:", churn_default["internet_service"].isna().sum())
print("internet_service missing when only '' is missing:", churn["internet_service"].isna().sum())
print(churn["internet_service"].value_counts().to_dict())

# %% [markdown]
# *What just happened?* The file contains the literal text `None` (meaning "no internet service"). pandas' default parser treats the string `None` as a missing
# value, so a legitimate category turned into over a thousand "missing" cells. If you had "fixed" that by imputation, you would have corrupted a real segment. Decide explicitly
# what counts as missing (`na_values`, `keep_default_na`) and check the result. From here on we read `churn` the careful way.

# %% [markdown]
# ## 2. A mini expectations framework
# Design goals: each check is a small function you can reuse across datasets, returns a **structured result** (name, how many rows failed, examples, severity), and one crash inside a
# check never hides the others. Read the code once; the usage below is the point.

# %%
@dataclass
class Result:
    name: str
    passed: bool
    n_checked: int
    n_failed: int
    severity: str = "error"                 # "error" blocks the pipeline, "warn" only reports
    examples: list = field(default_factory=list)
    detail: str = ""


def _from_mask(name, bad_mask, severity="error", detail=""):
    """Turn a boolean 'this row is bad' mask into a Result."""
    n_bad = int(bad_mask.sum())
    return Result(name, n_bad == 0, len(bad_mask), n_bad, severity, bad_mask[bad_mask].index[:3].tolist(), detail)


def expect_columns(cols, severity="error"):
    def check(df):
        missing = [c for c in cols if c not in df.columns]
        return Result(f"columns present {list(cols)[:3]}...", not missing, len(cols), len(missing), severity, missing, "missing columns" if missing else "")
    return check


def expect_dtype(col, kind, severity="error"):
    """kind: 'number', 'string' or 'datetime'."""
    tests = {"number": pd.api.types.is_numeric_dtype, "string": pd.api.types.is_string_dtype, "datetime": pd.api.types.is_datetime64_any_dtype}
    def check(df):
        ok = tests[kind](df[col])
        return Result(f"{col} is {kind}", ok, 1, 0 if ok else 1, severity, [] if ok else [str(df[col].dtype)])
    return check


def expect_not_null(col, max_frac=0.0, severity="error"):
    def check(df):
        frac = df[col].isna().mean()
        return Result(f"{col} nulls <= {max_frac:.0%}", frac <= max_frac, len(df), int(df[col].isna().sum()), severity,
                      df.index[df[col].isna()][:3].tolist(), f"null fraction {frac:.2%}")
    return check


def expect_unique(col, severity="error"):
    def check(df):
        return _from_mask(f"{col} is unique", df[col].duplicated(keep=False) & df[col].notna(), severity)
    return check


def expect_in_set(col, allowed, severity="error"):
    def check(df):
        return _from_mask(f"{col} in {sorted(allowed)}", ~df[col].isin(allowed) & df[col].notna(), severity)
    return check


def expect_between(col, lo, hi, severity="error"):
    def check(df):
        return _from_mask(f"{lo} <= {col} <= {hi}", ~df[col].between(lo, hi) & df[col].notna(), severity)
    return check


def expect_regex(col, pattern, severity="error"):
    rx = re.compile(pattern)
    def check(df):
        ok = df[col].astype("string").str.match(rx).fillna(True)
        return _from_mask(f"{col} matches /{pattern}/", ~ok, severity)
    return check


def expect_rule(name, rule: Callable[[pd.DataFrame], pd.Series], severity="error"):
    """Generic row-level business rule: `rule(df)` returns True for GOOD rows."""
    def check(df):
        return _from_mask(name, ~rule(df).fillna(True).astype(bool), severity)
    return check


def expect_row_count(lo, hi, severity="error"):
    def check(df):
        return Result(f"row count in [{lo}, {hi}]", lo <= len(df) <= hi, len(df), 0 if lo <= len(df) <= hi else 1, severity, [], f"{len(df)} rows")
    return check


def expect_fresh(col, now, max_age_days, severity="error"):
    """Newest value of `col` must be within `max_age_days` of `now` (pass `now` in: never call the clock inside a check)."""
    def check(df):
        newest = pd.to_datetime(df[col]).max()
        age = (pd.Timestamp(now) - newest).days
        return Result(f"{col} newest within {max_age_days}d", age <= max_age_days, 1, int(age > max_age_days), severity, [], f"newest {newest.date()}, age {age}d")
    return check


class Suite:
    def __init__(self, name):
        self.name, self.checks = name, []

    def add(self, *checks):
        self.checks += checks
        return self

    def run(self, df):
        results = []
        for chk in self.checks:
            try:
                results.append(chk(df))
            except Exception as exc:  # a crashing check must FAIL loudly, never pass silently
                results.append(Result(f"{getattr(chk, '__name__', 'check')} crashed", False, 0, 1, "error", [], f"{type(exc).__name__}: {exc}"))
        return Report(self.name, results)


@dataclass
class Report:
    suite: str
    results: list

    @property
    def ok(self):
        return all(r.passed or r.severity == "warn" for r in self.results)

    def to_frame(self):
        return pd.DataFrame([{"check": r.name, "status": "PASS" if r.passed else ("WARN" if r.severity == "warn" else "FAIL"),
                              "failed": r.n_failed, "of": r.n_checked, "examples": r.examples, "detail": r.detail} for r in self.results])

    def raise_if_failed(self):
        bad = [r.name for r in self.results if not r.passed and r.severity == "error"]
        if bad:
            raise ValueError(f"{self.suite}: {len(bad)} blocking check(s) failed: {bad}")


# %% [markdown]
# ## 3. Run it on `churn.csv`
# Each check is a **claim about the data**. We write the claims we believe are true, then let the data argue.

# %%
churn_suite = Suite("churn").add(
    expect_columns(["customer_id", "tenure_months", "contract", "monthly_charges", "total_charges", "churned"]),
    expect_dtype("monthly_charges", "number"),
    expect_dtype("contract", "string"),
    expect_row_count(1_000, 100_000),
    expect_unique("customer_id"),
    expect_regex("customer_id", r"^C\d{6}$"),
    expect_not_null("tenure_months"),
    expect_not_null("total_charges", max_frac=0.01),
    expect_not_null("internet_service", max_frac=0.01, severity="warn"),
    expect_in_set("contract", {"Month-to-month", "One year", "Two year"}),
    expect_in_set("internet_service", {"DSL", "Fiber", "None"}),
    expect_in_set("churned", {0, 1}),
    expect_between("tenure_months", 0, 120),
    expect_between("monthly_charges", 0, 500),
    expect_between("support_calls", 0, 50, severity="warn"),
)
report = churn_suite.run(churn)
report.to_frame().drop(columns="examples")

# %% [markdown]
# Two blocking checks fail and one warns. Look at what they tell you: the **counts** (how bad), the **examples** (where to look), and the **severity** (block or just warn).

# %%
fails = report.to_frame().query("status == 'FAIL'")
fails[["check", "failed", "of", "examples", "detail"]]

# %%
contract_result = next(r for r in report.results if r.name.startswith("contract in"))
print(churn.loc[contract_result.examples, ["customer_id", "contract"]])
print("blocking failure?", not report.ok)

# %% [markdown]
# `contract` has inconsistent casing (`month-to-month`): a *validity* failure worth fixing in the transform step, and `total_charges` is missing more than the 1% we said
# we would tolerate. `internet_service` has a small share of genuinely empty cells, which we only *warn* about. Note the threshold `max_frac=0.01` is a judgement call, not a law: **checks encode your expectations, and thresholds should come from looking at history.**
#
# Business rules are checks too. Domain knowledge: for a customer with tenure of at least 1 month, `total_charges` should be close to `tenure_months * monthly_charges`.
# Let's see how tight the data really is before writing the rule:

# %%
active = churn[churn["tenure_months"] >= 1].dropna(subset=["total_charges"])
ratio = active["total_charges"] / (active["tenure_months"] * active["monthly_charges"])
print(ratio.describe().round(3)[["min", "50%", "max"]].to_dict())

# %%
charges_rule = expect_rule("total ~ tenure * monthly (+/-6%) when tenure >= 1",
                           lambda d: (d["tenure_months"] < 1) | d["total_charges"].isna()          # nulls are the completeness check's job
                           | ((d["total_charges"] / (d["tenure_months"].clip(lower=1) * d["monthly_charges"]) - 1).abs() <= 0.06))
print(charges_rule(churn))

# %% [markdown]
# Passing is informative too: this rule can now guard against a future bug (for example, if `total_charges` is accidentally loaded in cents).
# **Test your checks**: a check that can never fail is worthless. Plant a bug in a copy and confirm it is caught (a tiny "mutation test"):

# %%
broken = churn.copy()
broken.loc[broken.index[:20], "total_charges"] *= 100          # pretend a unit bug: dollars became cents
print("clean data:  ", charges_rule(churn).n_failed, "failures")
print("planted bug: ", charges_rule(broken).n_failed, "failures")

# %% [markdown]
# ## 4. `messy.csv`: a dataset that should be rejected

# %%
messy = pd.read_csv(DATA / "messy.csv", keep_default_na=False, na_values=["", "NaN"])
messy["age"] = pd.to_numeric(messy["age"], errors="coerce")
messy_suite = Suite("messy").add(
    expect_unique("id"),
    expect_not_null("age", max_frac=0.05),
    expect_between("age", 0, 110),
    expect_rule("salary parses as a number", lambda d: pd.to_numeric(d["salary"].astype("string").str.replace(r"[$,]", "", regex=True), errors="coerce").notna()),
    expect_rule("signup is ISO date (YYYY-MM-DD)", lambda d: pd.to_datetime(d["signup"], format="%Y-%m-%d", errors="coerce").notna()),
    expect_in_set("country", {"US", "UK", "IN", "BD"}),
)
messy_report = messy_suite.run(messy)
messy_report.to_frame().drop(columns=["examples", "detail"])

# %% [markdown]
# Every check fails, as it should: that is a dataset you would send back to the producer, or clean and re-validate. Notice each failure means something different: duplicated keys need
# de-duplication rules, out-of-range ages need a decision (null them? drop the row?), and mixed date formats need a parser. Validation tells you **that** and **how much**; a cleaning step
# (see the pandas cleaning notebook) decides **what to do**. Keep the two separate so the checks stay honest.
#
# ## 5. Referential integrity and freshness on `shop.db`
# In a real warehouse keys are often not enforced, so an "orphan" (a child row without a parent) is something you must look for. In SQL, each check is a query that returns the
# **violating** rows; zero rows means pass. The same structure works in dbt tests.

# %%
con = sqlite3.connect(DATA / "shop.db")


def sql_check(con, name, violations_sql, severity="error"):
    """A SQL check: the query must return the offending rows."""
    bad = pd.read_sql(violations_sql, con)
    return Result(name, bad.empty, -1, len(bad), severity, bad.iloc[:3, 0].tolist())


shop_checks = [
    sql_check(con, "orders.customer_id exists in customers",
              "SELECT o.order_id FROM orders o LEFT JOIN customers c USING (customer_id) WHERE c.customer_id IS NULL"),
    sql_check(con, "order_items.order_id exists in orders",
              "SELECT i.item_id FROM order_items i LEFT JOIN orders o USING (order_id) WHERE o.order_id IS NULL"),
    sql_check(con, "order_items.product_id exists in products",
              "SELECT i.item_id FROM order_items i LEFT JOIN products p USING (product_id) WHERE p.product_id IS NULL"),
    sql_check(con, "orders.order_id unique", "SELECT order_id FROM orders GROUP BY 1 HAVING count(*) > 1"),
    sql_check(con, "quantity between 1 and 20", "SELECT item_id FROM order_items WHERE quantity NOT BETWEEN 1 AND 20"),
    sql_check(con, "every order has at least one item (warn)",
              "SELECT o.order_id FROM orders o LEFT JOIN order_items i USING (order_id) WHERE i.item_id IS NULL", severity="warn"),
]
pd.DataFrame([{"check": r.name, "status": "PASS" if r.passed else "FAIL", "violations": r.n_failed} for r in shop_checks])

# %% [markdown]
# Freshness is about *time*. The data's newest order defines "how fresh it is"; we pass the reference time **in** (a fixed timestamp here) so the check is deterministic and testable.
# Pipelines pass the run's scheduled time.

# %%
orders = pd.read_sql("SELECT * FROM orders", con)
print(expect_fresh("order_date", "2025-06-01", max_age_days=3)(orders))
print(expect_fresh("order_date", "2025-07-15", max_age_days=3)(orders))

# %% [markdown]
# ## 6. Row-level validation with pydantic
# DataFrame checks are set-based. When you ingest records one at a time (API payloads, events, messages) you validate each record against a **model**. Pydantic does type coercion,
# constraints and readable errors, and the same model can document your data contract and later power an API.

# %%
class ChurnRowStrict(BaseModel):
    customer_id: str = Field(pattern=r"^C\d{6}$")
    tenure_months: int = Field(ge=0, le=120)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber", "None"] | None = None
    monthly_charges: float = Field(gt=0)
    total_charges: float | None = Field(default=None, ge=0)
    support_calls: int = Field(ge=0)
    paperless_billing: bool
    autopay: bool
    senior_citizen: bool
    churned: bool


def to_records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")        # NaN -> None so 'optional' works


rows = to_records(churn)
errors = Counter()
n_bad = 0
for row in rows:
    try:
        ChurnRowStrict.model_validate(row)
    except ValidationError as e:
        n_bad += 1
        errors.update((err["loc"][0], err["type"]) for err in e.errors())
print(f"{n_bad} of {len(rows)} rows rejected; error kinds:", dict(errors))

# %% [markdown]
# The strict model rejects exactly the rows with the wrong `contract` casing. Pydantic coerced `"Yes"`/`"No"` to booleans and `0`/`1` to booleans for us: convenient, but note that this
# is **lax mode** coercion. Use `model_config = ConfigDict(strict=True)` when you do *not* want `"1"` accepted as an integer.
#
# Now the opposite policy: a *lenient* model that **cleans** known, safe variations (casing, whitespace) with a `before` validator, and still rejects everything else.

# %%
class ChurnRow(ChurnRowStrict):
    @field_validator("contract", mode="before")
    @classmethod
    def normalise_contract(cls, v):
        return {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}.get(str(v).strip().lower(), v)


n_bad_lenient = 0
for row in rows:
    try:
        ChurnRow.model_validate(row)
    except ValidationError:
        n_bad_lenient += 1
print("rejected by the lenient model:", n_bad_lenient)

# %% [markdown]
# **Which is right?** Strict validation at the boundary (reject and quarantine) makes problems visible to the producer; lenient validation makes your pipeline robust to *known* harmless
# variation. A good rule: normalise only what you can prove is harmless, log how many rows you fixed, and reject the rest.

# %% [markdown]
# ## 7. Where to run checks, and what to do on failure
#
# | Where | What to check | On failure |
# |---|---|---|
# | **On ingestion** (raw/bronze) | schema, row count, freshness, file readable | stop, alert the owner of the source |
# | **After transformation** (silver) | uniqueness, keys, validity, business rules | quarantine bad rows or stop, depending on severity |
# | **Before serving** (gold, features) | totals reconcile with silver, no fan-out, distributions sane | block the publish; keep serving the last good version |
#
# Three failure policies: **fail** (stop the pipeline; use for anything that would make downstream results wrong), **warn** (record and continue; use for soft expectations), and **quarantine**
# (split good and bad rows, continue with the good, keep the bad for review). Whatever you pick, decide *before* the incident, and make failures visible (a report saved with each run).

# %%
def quarantine(df, model):
    """Split a DataFrame into (valid rows, rejected rows with the reason)."""
    good_idx, bad = [], []
    for idx, row in zip(df.index, to_records(df)):
        try:
            model.model_validate(row)
            good_idx.append(idx)
        except ValidationError as e:
            bad.append({"index": idx, "reason": "; ".join(f"{x['loc'][0]}: {x['type']}" for x in e.errors())})
    return df.loc[good_idx], pd.DataFrame(bad)


good, rejected = quarantine(churn, ChurnRowStrict)
print(len(good), "kept,", len(rejected), "quarantined")
print(rejected["reason"].value_counts().to_dict())

# %% [markdown]
# ## Common mistakes
#
# 1. **Only checking after the dashboard looks wrong.** Put checks at the boundaries (ingest, transform, serve).
# 2. **Checks that cannot fail** (thresholds set so loose that any data passes). Mutation-test them as we did.
# 3. **Silent NaN handling.** `df[col].isin(...)` treats NaN as "not in set" and comparisons with NaN are False, so decide explicitly whether null is allowed (completeness check) and exclude it from validity checks.
# 4. **Wall-clock time inside a check** (`datetime.now()`): a re-run tomorrow behaves differently. Pass the reference time in.
# 5. **Alert fatigue.** 200 checks that fail every day teach everyone to ignore them. Use severities and fix or delete flaky checks.
# 6. **Letting parsing hide problems**, like the `None` category read as missing. Be explicit about `na_values`.

# %% [markdown]
# ## Interview corner
#
# **1. How do you make sure a pipeline's output is trustworthy?**
# <details><summary>Answer</summary>
# Layered checks: schema and volume/freshness on ingestion, uniqueness/keys/validity/business rules after transformation, reconciliation (totals match the previous layer) before serving.
# Each check has a severity and an owner, results are stored per run, and failures alert someone before consumers see bad data.
# </details>
#
# **2. What is the difference between data validation and data cleaning?**
# <details><summary>Answer</summary>
# Validation detects and measures violations of expectations; cleaning changes the data to fix them. Keeping them separate means the checks stay an honest audit and cleaning rules are explicit,
# logged and testable (for example, "casing normalised for 189 rows").
# </details>
#
# **3. A daily table normally has 100k rows and today has 40k. What do you do?**
# <details><summary>Answer</summary>
# Treat it as a failed volume check: stop downstream publishing, look for an upstream partial load or filter change, compare with the same weekday history (thresholds should account for weekly seasonality),
# and rerun once fixed. Do not quietly publish a partial day.
# </details>
#
# **4. Great Expectations, dbt tests, pandera, pydantic: how do they differ?**
# <details><summary>Answer</summary>
# All express expectations as code. pydantic validates one record against a typed model (APIs, events). pandera validates DataFrame schemas in Python. dbt tests are SQL assertions that run inside the warehouse.
# Great Expectations is a broader framework with expectation suites, data docs and stores. Pick based on where the data lives; the concepts are what we built here.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Write a business-rule check
# A customer without internet service should not pay much (in this data, the 30-unit mark is a sensible ceiling). Using `expect_rule`, build a check that **fails rows where `monthly_charges > 30` and `internet_service == "None"`**,
# run it on `churn`, and store the returned `Result` in `bill_result`. *Hint: the rule function must return True for GOOD rows, so negate the bad condition.*

# %% tags=["starter"]
bill_result = ...

# %%
expected_bad = int(((churn["monthly_charges"] > 30) & (churn["internet_service"] == "None")).sum())
assert isinstance(bill_result, Result), "call the check you built on churn: expect_rule(...)(churn)"
assert bill_result.n_failed == expected_bad, f"expected {expected_bad} violating rows, got {bill_result.n_failed}"
print(f"Exercise 1 passed: {expected_bad} rows flagged, passed={bill_result.passed}")

# %% tags=["solution"]
bill_check = expect_rule("no big bill without internet", lambda d: ~((d["monthly_charges"] > 30) & (d["internet_service"] == "None")))
bill_result = bill_check(churn)

# %% [markdown]
# ### Exercise 2: Catch an orphan
# `bad_con` below is an in-memory copy of `shop.db` into which we inserted broken rows. Write two SQL queries that **count** (a) `orphan_orders`: orders whose `customer_id` has no matching customer, and
# (b) `orphan_items`: order items whose `order_id` has no matching order. Store the two integers.
# *Hint: `LEFT JOIN ... WHERE parent.key IS NULL`, or `NOT EXISTS`.*

# %%
bad_con = sqlite3.connect(":memory:")
con.backup(bad_con)
bad_con.executemany("INSERT INTO orders VALUES (?, ?, '2025-05-30', 'delivered', 'web')", [(900001, 999001), (900002, 999002), (900003, 999003)])
bad_con.executemany("INSERT INTO order_items VALUES (?, ?, 1, 1)", [(990001, 888001), (990002, 888002)])
bad_con.commit()

# %% tags=["starter"]
orphan_orders = ...
orphan_items = ...

# %%
assert orphan_orders == 3, f"we inserted 3 orders with unknown customers, you found {orphan_orders}"
assert orphan_items == 2, f"we inserted 2 items with unknown orders, you found {orphan_items}"
print("Exercise 2 passed")

# %% tags=["solution"]
orphan_orders = bad_con.execute("SELECT count(*) FROM orders o WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = o.customer_id)").fetchone()[0]
orphan_items = bad_con.execute("SELECT count(*) FROM order_items i WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.order_id = i.order_id)").fetchone()[0]

# %% [markdown]
# ### Exercise 3: Lenient parsing with an audit trail
# Use `ChurnRow` (the lenient model) with the `quarantine` function to build `clean_good, clean_rejected`. Then compute `n_fixed`: how many rows in `clean_good` had a `contract`
# value that was **not** already in canonical form in the raw `churn` data (that is, rows the lenient model accepted only thanks to normalisation).
# *Hint: canonical values are `"Month-to-month"`, `"One year"`, `"Two year"`; compare on the raw column at the kept indices.*

# %% tags=["starter"]
clean_good, clean_rejected = ..., ...
n_fixed = ...

# %%
canon = {"Month-to-month", "One year", "Two year"}
exp_fixed = int((~churn["contract"].isin(canon)).sum())
assert len(clean_rejected) == 0, "the lenient model should accept every row of churn"
assert len(clean_good) == len(churn)
assert n_fixed == exp_fixed, f"expected {exp_fixed} normalised rows"
print(f"Exercise 3 passed: {n_fixed} rows accepted only after normalising the contract casing")

# %% tags=["solution"]
clean_good, clean_rejected = quarantine(churn, ChurnRow)
n_fixed = int((~churn.loc[clean_good.index, "contract"].isin({"Month-to-month", "One year", "Two year"})).sum())

# %% [markdown]
# ### Exercise 4 (open-ended): A suite for `customers.csv`
# Build a `Suite` for `customers.csv` (`customer_id` unique, `age` plausible, `country` in a sensible set, `signup_date` not in the future relative to a `now` you pass in...). Run it, print the report, and write in a markdown cell
# which checks you would make `error` and which `warn`, and why.

# %% [markdown]
# ## Recap
# - Data quality = schema, completeness, uniqueness, validity, referential integrity, freshness/volume. Write each expectation as a reusable check that returns a structured result.
# - Checks have **severity** (fail/warn) and a **policy** on failure (stop, warn, quarantine); decide before the incident.
# - Validation is not cleaning: measure first, then fix with explicit, logged rules. Read files explicitly (`na_values`), or the parser will hide problems (`"None"` became NaN).
# - SQL checks return offending rows; pydantic validates individual records; freshness checks take the reference time as an argument.
# - Test your checks by planting a bug; a check that cannot fail is worthless.
#
# **Go deeper:** the documentation of Great Expectations (greatexpectations.io), dbt tests (docs.getdbt.com), pandera (pandera.readthedocs.io) and pydantic (docs.pydantic.dev).
#
# **Next:** [APIs and ingestion](05_apis_and_ingestion.ipynb)
