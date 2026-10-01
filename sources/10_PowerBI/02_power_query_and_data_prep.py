# %% [markdown]
# # Power Query and Data Preparation
#
# **Why this matters.** Analysts routinely spend more time cleaning data than building visuals, and in Power BI the cleaning happens in
# **Power Query**: a recorded list of steps that re-runs on every refresh. If you understand what each step does in a language you already
# know (pandas), you can design, review and debug Power Query steps confidently, and you can *prove* your cleaned table is right.
#
# **You will learn**
# - explain Power Query's model: queries, applied steps, the M language, query folding
# - translate the common steps (trim, replace, change type, merge, append, group, pivot, unpivot) between M and pandas
# - clean a genuinely messy file (mixed date formats, currency strings, sentinel values, duplicates) and record a row-count audit for every step
# - decide what belongs in Power Query versus in DAX
#
# **Prerequisites:** [the data model notebook](01_power_bi_overview_and_data_model.ipynb) and [pandas cleaning basics](../05_Pandas/01_loading_and_inspecting_data.ipynb).
#
# > **Candour box.** Power Query is part of Power BI Desktop/Excel and was **not run** here. The **M snippets are tagged `no-run` and were not executed**: they are written from the documented
# > M library functions, but check them (and current function names and dialogs) in Microsoft's documentation before use. The **pandas mirrors run** and give you the row counts and values your Power Query steps should produce.
# > Data: `datasets/messy.csv` (a synthetic dirty employee-style file, see `docs/DATASETS.md`) and the shop tables.

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
raw = pd.read_csv(DATA / "messy.csv", dtype=str, keep_default_na=False)   # everything as text: like a fresh Power Query load
print(raw.shape)
raw.head(6)

# %% [markdown]
# ## 1. How Power Query thinks
#
# - A **query** is a named recipe that produces a table. You build it by clicking transformations; each click appends an **applied step**, written in the **M** language.
# - Steps are **functional**: each step is a named value derived from the previous step (`Trimmed = Table.TransformColumns(Promoted, ...)`). The recipe is replayed from the source on every refresh.
# - Because it is a recipe, it is **repeatable and auditable**. That is the big advantage over hand-editing a spreadsheet, and the reason to do cleaning *here* rather than in the source file.
# - **Query folding**: when the source is a database, Power Query can translate steps into one SQL query so that the *database* does the filtering and aggregating. A step that cannot be
#   folded (and every step after it) runs in Power Query's own engine on downloaded rows. Order steps so filters and column removal come first.
# - Two stages in the workflow: *transform* in Power Query (row-level cleaning, reshaping), *calculate* in DAX (aggregations that respond to slicers). Rule of thumb: **shape data in Power Query, compute answers in DAX.**
#
# ## 2. M to pandas: the translation table
#
# | Power Query step (UI wording may vary: verify) | M function | pandas |
# |---|---|---|
# | Use first row as headers | `Table.PromoteHeaders` | `read_csv(header=0)` |
# | Change type | `Table.TransformColumnTypes` | `astype`, `pd.to_numeric`, `pd.to_datetime` |
# | Trim / Clean / Capitalize each word | `Text.Trim`, `Text.Clean`, `Text.Proper` | `str.strip`, regex, `str.title` |
# | Replace values | `Table.ReplaceValue` | `replace`, `mask`, `map` |
# | Remove duplicates | `Table.Distinct` | `drop_duplicates` |
# | Filter rows | `Table.SelectRows` | boolean mask / `query` |
# | Add custom column | `Table.AddColumn` | `assign` |
# | Group by | `Table.Group` | `groupby().agg` |
# | Merge queries (join) | `Table.NestedJoin` + expand | `merge` |
# | Append queries (union) | `Table.Combine` | `pd.concat` |
# | Pivot / Unpivot columns | `Table.Pivot` / `Table.UnpivotOtherColumns` | `pivot_table` / `melt` |
# | Remove errors / Replace errors | `Table.RemoveRowsWithErrors` / `try ... otherwise` | `to_numeric(errors="coerce")` |

# %% [markdown]
# ## 3. A worked cleaning pipeline: `messy.csv`
#
# First **profile** the problems (Power Query's *Column quality / distribution / profile* toggles show the same facts: verify in View tab). Every issue below is one step to write.

# %%
profile = {
    "duplicate ids": int(raw["id"].duplicated().sum()),
    "distinct country spellings": raw["country"].nunique(),
    "age values that are sentinels or junk": int(raw["age"].isin(["-1", "999", "N/A", ""]).sum()),
    "salary values that are not plain numbers": int((~raw["salary"].str.fullmatch(r"\d+")).sum()),
    "signup values not ISO yyyy-mm-dd": int((~raw["signup"].str.fullmatch(r"\d{4}-\d{2}-\d{2}")).sum()),
    "names with stray or double spaces": int((raw["name"] != raw["name"].str.strip().str.replace(r"\s+", " ", regex=True)).sum()),
}
pd.Series(profile, name="count")

# %% [markdown]
# Now the steps. Each is a small function so it can be tested alone, then chained. Decisions are **assumptions to state explicitly** in your report; here we assume `yyyy.mm.dd` for dotted dates (it is ambiguous: `2024.02.03` could also be day-first) and that `999` and `-1` in `age` are sentinel codes, not real ages.

# %%
COUNTRY_MAP = {"us": "US", "usa": "US", "u.s.": "US", "united states": "US",
               "uk": "UK", "united kingdom": "UK", "in": "IN", "india": "IN", "bangladesh": "BD"}


def clean_name(s):
    return re.sub(r"\s+", " ", s.strip()).title()          # M: Text.Trim + collapse inner spaces + Text.Proper


def parse_age(s):
    try:
        v = int(s)
    except ValueError:
        return np.nan                                       # 'N/A', '' -> null (M: try ... otherwise null)
    return np.nan if v in (-1, 999) or not (0 < v < 120) else float(v)


def parse_salary(s):
    t = re.sub(r"[\s$,]", "", s.lower())
    if t in ("", "unknown"):
        return np.nan
    if t.endswith("k"):
        return float(t[:-1]) * 1000
    return float(t)


def parse_date(s):
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y.%m.%d", "%b %d, %Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.NaT                                           # 'not available', ''


audit = [("loaded", len(raw))]
df = raw.copy()
df["name"] = df["name"].map(clean_name)
df["country"] = df["country"].str.strip().str.lower().map(COUNTRY_MAP)
audit.append(("after text + country standardisation", len(df)))
df["age"] = df["age"].map(parse_age)
df["salary"] = df["salary"].map(parse_salary)
df["signup"] = df["signup"].map(parse_date)
df["id"] = df["id"].astype("int64")
df = df.drop_duplicates(subset="id", keep="first").sort_values("id", ignore_index=True)
audit.append(("after remove duplicates on id", len(df)))
clean = df
print(pd.DataFrame(audit, columns=["step", "rows"]).to_string(index=False))
clean.head()

# %% [markdown]
# *What just happened?* We turned six text columns into proper types, and wrote down **the row count after each step**. In Power Query the same numbers are visible by clicking through the Applied Steps (or with *Column profile > Count rows*).
# The audit is the point: if a step unexpectedly drops 300 rows, you find out now, not when a total looks wrong on a dashboard.
#
# Check what the cleaning achieved, and, just as important, **what it could not fix** (unfixable values become nulls, which you must report, not hide):

# %%
after = pd.DataFrame({
    "non-null": clean.notna().sum(),
    "null": clean.isna().sum(),
    "null %": (clean.isna().mean() * 100).round(1),
})
print("countries:", sorted(clean["country"].dropna().unique()))
print("id unique:", clean["id"].is_unique, "| age range:", clean["age"].min(), "to", clean["age"].max(), "| salary range:", clean["salary"].min(), "to", clean["salary"].max())
after

# %% [markdown]
# The same pipeline in M (not executed). Read it as the *Advanced Editor* view of the steps above. Notice how each line names its previous step:

# %% tags=["no-run"]
M_QUERY = r'''
let
    Source   = Csv.Document(File.Contents("C:\data\messy.csv"), [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),
    // keep everything as text first; convert types deliberately, column by column
    Names    = Table.TransformColumns(Promoted, {{"name",
                  each Text.Proper(Text.Combine(List.Select(Text.Split(Text.Trim(_), " "), each _ <> ""), " ")), type text}}),
    Country  = Table.TransformColumns(Names, {{"country",
                  each Text.Upper(Text.Trim(_)), type text}}),
    Mapped   = Table.ReplaceValue(Country, "USA", "US", Replacer.ReplaceValue, {"country"}),   // only 'USA' shown: repeat for UNITED STATES, U.S., INDIA, UNITED KINGDOM ..., or (better) merge a mapping table
    Age      = Table.TransformColumns(Mapped, {{"age",
                  each let n = try Number.FromText(_) otherwise null in if n = null or n = -1 or n = 999 then null else n, type nullable number}}),
    Salary   = Table.TransformColumns(Age, {{"salary",
                  each let t = Text.Replace(Text.Replace(Text.Replace(Text.Lower(_), "$", ""), ",", ""), " ", "")
                       in if t = "" or t = "unknown" then null
                          else if Text.EndsWith(t, "k") then Number.FromText(Text.Start(t, Text.Length(t) - 1)) * 1000
                          else Number.FromText(t), type nullable number}}),
    Deduped  = Table.Distinct(Salary, {"id"})
in
    Deduped
'''
# Dates with several formats need one try-per-format expression, e.g.
#   each try Date.FromText(_, [Format = "yyyy-MM-dd", Culture = "en-US"]) otherwise
#        try Date.FromText(_, [Format = "dd/MM/yyyy", Culture = "en-GB"]) otherwise null
# Verify function options in the current M reference before use.

# %% [markdown]
# **Why you keep types as text at first.** Power Query's automatic *Detect data type* looks at the first rows and can silently produce errors or wrong dates (for example reading `03/04/2025` in the wrong locale)
# further down. Set types deliberately and check the *error* count in the column-quality bar.

# %% [markdown]
# ## 4. Combining and reshaping: merge, append, group, pivot
#
# **Merge** is a join, and the join *kind* decides which rows survive. The kind names below are Power Query's; the pandas equivalent is a `merge` (verify wording in your version).

# %%
customers = pd.read_csv(DATA / "customers.csv")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])

# Left Anti: customers who have NO orders (M: Table.NestedJoin with JoinKind.LeftAnti)
m = customers.merge(orders[["customer_id"]].drop_duplicates(), on="customer_id", how="left", indicator=True)
no_orders = m[m["_merge"] == "left_only"].drop(columns="_merge")
print("customers:", len(customers), "| with orders:", orders["customer_id"].nunique(), "| left anti (no orders):", len(no_orders))
assert len(no_orders) == len(customers) - orders["customer_id"].nunique()

# %%
join_kinds = pd.DataFrame({
    "Power Query kind": ["Left Outer", "Right Outer", "Full Outer", "Inner", "Left Anti", "Right Anti"],
    "pandas": ["how='left'", "how='right'", "how='outer'", "how='inner'", "how='left' then keep _merge == 'left_only'", "how='right' then keep _merge == 'right_only'"],
})
join_kinds

# %% [markdown]
# **Append** stacks tables with the same columns (like `UNION ALL`). **Group By** aggregates. **Pivot** turns values of one column into columns; **Unpivot** turns columns into rows, and it is the fix for
# "one column per month" spreadsheets that are hard to chart. Round-trip check: a pivot followed by an unpivot must preserve totals.

# %%
orders = orders.assign(year_month=orders["order_date"].dt.strftime("%Y-%m"))
wide = orders.pivot_table(index="year_month", columns="channel", values="order_id", aggfunc="count", fill_value=0).reset_index()   # Pivot
print(wide.head(3).to_string(index=False))
long = wide.melt(id_vars="year_month", var_name="channel", value_name="orders")                                                        # Unpivot other columns
assert long["orders"].sum() == len(orders), "unpivot must preserve the total number of orders"
print("total orders before and after the round trip:", len(orders), long["orders"].sum())

# %% [markdown]
# ## 5. Query folding, as a measurable idea
#
# When folding works, the database returns only the answer; when it does not, all rows travel first. We can show the effect with DuckDB standing in for a database: same question, two ways.
# (This illustrates the *principle*. Which Power Query steps fold depends on the connector; check "View Native Query" on a step, or the documentation, for your source.)

# %%
import duckdb

con = duckdb.connect()
sql = f"""
    SELECT o.channel, COUNT(*) AS n
    FROM read_csv_auto('{(DATA / 'orders.csv').as_posix()}') o
    WHERE o.status = 'delivered' AND o.order_date >= DATE '2025-01-01'
    GROUP BY o.channel ORDER BY o.channel
"""
pushed = con.sql(sql).df()
pulled_rows = len(pd.read_csv(DATA / "orders.csv"))          # "download everything, then filter locally"
print(f"pushed-down query returns {len(pushed)} rows; download-then-filter moves {pulled_rows} rows first")
pushed

# %% [markdown]
# ## Common mistakes
#
# 1. **Cleaning by hand in the source file.** Not repeatable; the next month's file is dirty again. Put the fix in a step.
# 2. **Trusting auto-detected types** (dates, decimals with locale-dependent separators). Set them explicitly and check errors.
# 3. **Doing aggregations in Power Query that should be measures.** Pre-aggregated tables cannot respond to slicers or new questions. Keep the fact at line grain.
# 4. **Filtering late.** Put row filters and "remove other columns" first so folding keeps working and less data moves.
# 5. **Silently replacing errors with zeros.** Nulls mean "unknown"; zeros claim a value. Report the null counts (as we did).
# 6. **Not documenting assumptions** (`2024.02.03` is Feb 3 or Mar 2?). Put them in step names or query descriptions.

# %%
# Demonstration of mistake 5: replacing unknown salary with 0 changes the average
avg_null = clean["salary"].mean()
avg_zero = clean["salary"].fillna(0).mean()
print(f"mean salary ignoring unknowns: {avg_null:,.0f} | with unknowns replaced by 0: {avg_zero:,.0f}")

# %% [markdown]
# ## Interview corner
#
# **1. What is query folding and why do you care?**
# <details><summary>Answer</summary>
# Power Query translating steps into the source's native query (usually SQL) so the source does the work and returns fewer rows. It makes refresh faster and cheaper. A non-foldable step stops folding for all following steps, so put foldable steps (filters, column selection) first.
# </details>
#
# **2. Power Query or DAX: where does a given calculation belong?**
# <details><summary>Answer</summary>
# Row-level cleaning and reshaping that should be the same regardless of report filters (types, standardising labels, splitting columns, unpivoting) belongs in Power Query. Aggregations that must respond to slicers and filter context (totals, ratios, YTD) belong in DAX measures.
# </details>
#
# **3. What does Unpivot solve?**
# <details><summary>Answer</summary>
# Turning "wide" data (one column per month or per metric) into a tidy long table (one row per entity-period-value) that relationships, slicers and charts can use.
# </details>
#
# **4. How would you handle a file whose dates come in several formats?**
# <details><summary>Answer</summary>
# Load as text, try each explicit format in order with a fallback to null, count the nulls, and document the ambiguous cases (day-first vs month-first) rather than relying on auto-detection. Ideally fix the upstream export.
# </details>

# %% [markdown]
# ## Exercises
# `raw` (all text), `clean`, the `parse_*` helpers, `customers`, `orders` are defined above.

# %% [markdown]
# ### Exercise 1: Parse the salary column yourself
# Without calling `parse_salary`, write `salary_num(s)` converting text to a float: strip `$`, commas and spaces; `"75k"` means 75000; `""` and `"unknown"` become `NaN`.
# Store the mean salary of the **de-duplicated** ids (first occurrence of each id, in file order) in `mean_salary`, ignoring nulls.
# *Hint: `raw.drop_duplicates("id")["salary"].map(salary_num).mean()`.*

# %% tags=["starter"]
def salary_num(s):
    ...


mean_salary = ...

# %%
assert salary_num("$52,000") == 52000 and salary_num("$75k") == 75000 and salary_num("48,500") == 48500 and salary_num("90000") == 90000
assert np.isnan(salary_num("unknown")) and np.isnan(salary_num("")), "unknown and empty must be NaN"
_exp = raw.drop_duplicates("id")["salary"].map(parse_salary).mean()
assert abs(mean_salary - _exp) < 1e-6, f"expected {_exp:.2f}, got {mean_salary}"
print(f"Exercise 1 passed (mean salary {mean_salary:,.0f})")

# %% tags=["solution"]
def salary_num(s):
    t = s.lower().replace("$", "").replace(",", "").replace(" ", "")
    if t in ("", "unknown"):
        return float("nan")
    return float(t[:-1]) * 1000 if t.endswith("k") else float(t)


mean_salary = raw.drop_duplicates("id")["salary"].map(salary_num).mean()

# %% [markdown]
# ### Exercise 2: How many signup dates could we recover?
# After cleaning, how many of the de-duplicated rows have a **valid** `signup` date (not null), and what is the **latest** date? Store `n_valid_dates` and `latest_signup` (a `pd.Timestamp`).
# Also store `share_recovered` = `n_valid_dates` divided by the number of de-duplicated rows.

# %% tags=["starter"]
n_valid_dates = ...
latest_signup = ...
share_recovered = ...

# %%
_dd = raw.drop_duplicates("id")
_ok = _dd["signup"].str.strip().isin(["", "not available"])
assert n_valid_dates == int((~_ok).sum()), f"expected {int((~_ok).sum())}"
assert share_recovered == n_valid_dates / len(_dd)
assert isinstance(latest_signup, pd.Timestamp) and latest_signup == clean["signup"].max()
print(f"Exercise 2 passed ({n_valid_dates}/{len(_dd)} recovered, latest {latest_signup.date()})")

# %% tags=["solution"]
n_valid_dates = int(clean["signup"].notna().sum())
latest_signup = clean["signup"].max()
share_recovered = n_valid_dates / len(clean)

# %% [markdown]
# ### Exercise 3: Premium customers who never ordered (left anti join)
# Using `customers` and `orders`, find the **premium** customers (`is_premium` is True) that have **no orders**. Store how many in `n_premium_no_orders` and their sorted ids in `premium_ids`.
# Think in Power Query terms: filter the customers, then a Left Anti merge against orders on `customer_id`.

# %% tags=["starter"]
n_premium_no_orders = ...
premium_ids = ...

# %%
_prem = customers.loc[customers["is_premium"], "customer_id"]
_exp = sorted(set(_prem) - set(orders["customer_id"]))
assert n_premium_no_orders == len(_exp), f"expected {len(_exp)}, got {n_premium_no_orders}"
assert sorted(premium_ids) == _exp
print(f"Exercise 3 passed ({n_premium_no_orders} premium customers never ordered)")

# %% tags=["solution"]
_p = customers[customers["is_premium"]]
_m = _p.merge(orders[["customer_id"]].drop_duplicates(), on="customer_id", how="left", indicator=True)
premium_ids = sorted(_m.loc[_m["_merge"] == "left_only", "customer_id"].tolist())
n_premium_no_orders = len(premium_ids)

# %% [markdown]
# ### Exercise 4 (open-ended): Design your applied steps
# Take a CSV or Excel export you handle at work. In a markdown cell list the applied steps you would create **in order** (with the step names), mark which should fold if the source were a database,
# state two assumptions you must document, and name the audit numbers (row counts, null counts) you would check after each risky step.

# %% [markdown]
# ## Recap
# - Power Query = a repeatable, ordered recipe of steps (M) that reshapes data on refresh; DAX computes answers on top of it.
# - Set types deliberately, standardise labels with mappings, treat sentinels as null, and **audit row and null counts after each step**.
# - Merge kinds (left, inner, anti, ...) decide which rows survive: know them as join types.
# - Pivot/unpivot must preserve totals; check it.
# - Query folding pushes work to the database: filter and trim columns early.
#
# **Go deeper:** Microsoft Learn, "Power Query documentation" and the "Power Query M formula language reference" (learn.microsoft.com/power-query).
#
# **Next:** [DAX fundamentals](03_dax_fundamentals.ipynb)
