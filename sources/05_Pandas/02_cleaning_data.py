# %% [markdown]
# # Cleaning Data with pandas
#
# **Why this matters.** Real tables arrive with `"$52,000"` next to `"75k"` in the same column, three spellings of the United States,
# dates in four formats, and duplicate rows. A model or dashboard built on that silently gives wrong answers. Cleaning is where you
# turn "a file" into "a table you can trust", and where good analysts leave an audit trail of *what they changed and why*.
#
# **You will learn**
# - standardise inconsistent text (casing, whitespace, country names) with `.str` and `map`
# - coerce messy numbers (`"$52,000"`, `"75k"`, `"unknown"`) and mixed date formats to proper dtypes
# - find and drop duplicates, and spot the difference between a missing value and a real category
# - choose a missing-value strategy on purpose (drop, fill, flag) and know what each one costs
#
# **Prerequisites:** [Loading and inspecting data](01_loading_and_inspecting_data.ipynb).
#
# > Data used: `datasets/churn.csv` (telecom customers) and `datasets/messy.csv` (a deliberately dirty 640-row people table).
# > Both are synthetic; the problems in them are typical of real exports. See `docs/DATASETS.md`.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("../datasets")
print("pandas", pd.__version__)

# %% [markdown]
# ## 1. Cleaning principles (read once)
#
# 1. **Never edit the raw file.** Load it, clean in code, save a *new* file. Your notebook is the audit trail.
# 2. **Inspect, decide, fix, verify.** Every fix should be followed by a check that proves it worked (e.g. `value_counts()` again).
# 3. **Do not silently drop data.** Print how many rows or values each step changed.
# 4. **Keep the original column** until you trust the cleaned one (`salary_raw` next to `salary`), or clean in a copy.

# %% [markdown]
# ## 2. Text problems in `churn.csv`: inconsistent casing
# We saw in the last notebook that `contract` has the same category spelled differently.

# %%
churn = pd.read_csv(DATA / "churn.csv")
print(churn["contract"].value_counts())

# %% [markdown]
# Fix: standardise casing. `.str.lower()` gives one canonical form; we then choose the display form we want.
# In pandas 3 text columns have the dedicated `str` dtype, so `.str` methods work directly and missing text is `NaN`.

# %%
print(churn["contract"].dtype)
contract_clean = churn["contract"].str.strip().str.lower().map(
    {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}
)
print(contract_clean.value_counts(dropna=False))
print("rows changed:", int((contract_clean != churn["contract"]).sum()))

# %% [markdown]
# *What just happened?* We normalised (strip + lower) and then `map`ped to an explicit dictionary. Using a dictionary rather than `.str.title()` is deliberate:
# any value **not** in the dictionary becomes `NaN`, which is a loud signal that a new, unexpected spelling appeared. `.str.title()` would happily
# turn a typo into a new category. Always check for unmapped values:

# %%
assert contract_clean.notna().all(), "unmapped contract values found"
churn = churn.assign(contract=contract_clean)     # assign returns a new frame; the raw CSV is untouched

# %% [markdown]
# ## 3. A trap: "None" is not "missing"
#
# In the first notebook `internet_service` showed 1,141 missing values. Let us look at how they were read:

# %%
raw_text = pd.read_csv(DATA / "churn.csv", keep_default_na=False)   # do NOT auto-convert strings such as "None" or "" to NaN
print(raw_text["internet_service"].value_counts())

# %% [markdown]
# `read_csv` treats the literal text `None` (and `NA`, `null`, `N/A`, empty string, ...) as missing **by default**. Here it hides two different situations:
# customers whose service is the string `"None"` (no internet: a real category!) and customers whose cell is genuinely **empty**.
# Turning both into `NaN` would merge "no internet" customers with "we do not know". The fix is to say exactly what counts as missing:

# %%
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
churn["contract"] = contract_clean                          # re-apply the earlier fix on the freshly loaded frame
print(churn["internet_service"].value_counts(dropna=False))
print(churn["total_charges"].dtype, "|", churn["total_charges"].isna().sum(), "missing total_charges")

# %% [markdown]
# Now `"None"` is a category and only the truly empty cells are `NaN`. Does the distinction matter for the business question? Check churn by group:

# %%
print(churn.groupby("internet_service", dropna=False)["churned"].agg(["mean", "size"]).round(3))

# %% [markdown]
# The "no internet" group and the "unknown" group behave differently, so merging them would have blurred the picture. (`groupby` is the topic of a later notebook.)
# **Lesson:** ask what each missing-looking value *means* before you convert or fill it.

# %% [markdown]
# ## 4. A dirtier table: `messy.csv`

# %%
messy = pd.read_csv(DATA / "messy.csv")
print(messy.shape)
messy.sample(8, random_state=42)

# %%
messy.info()

# %% [markdown]
# `salary` is text (not numeric), `signup` is text (not a date), and `age` is float because it has missing values. Let us look at each problem column.

# %%
for col in ["signup", "country", "salary"]:
    print(messy[col].value_counts(dropna=False).to_string(), "\n")

# %% [markdown]
# ## 5. Whitespace, casing and names

# %%
print(messy["name"].value_counts().to_string())

# %% [markdown]
# The same six people appear under several spellings (`"  alice smith"`, `"dave  lee"` with a double space, `"BOB JONES"`).
# Fix in three steps: trim ends, collapse repeated spaces with a regular expression, then title-case.

# %%
name_clean = (
    messy["name"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)
print(name_clean.value_counts().to_string())

# %% [markdown]
# Caution: `.title()` turns `"frank o'neil"` into `"Frank O'Neil"`, which happens to be right here, but it would produce `"Mcdonald"` for `"mcdonald"`.
# Text normalisation is heuristic; for names that matter (customer records) use matching keys, not overwriting.

# %% [markdown]
# ## 6. Standardising categories: countries
# Same recipe as `contract`: normalise the key, then map through an explicit dictionary, then verify nothing is left unmapped.

# %%
country_key = messy["country"].str.strip().str.lower().str.replace(".", "", regex=False)   # "U.S." -> "us"
country_map = {
    "us": "United States", "usa": "United States", "united states": "United States",
    "uk": "United Kingdom", "united kingdom": "United Kingdom",
    "in": "India", "india": "India",
    "bangladesh": "Bangladesh",
}
country_clean = country_key.map(country_map)
print("unmapped:", int(country_clean.isna().sum()))
print(country_clean.value_counts())

# %% [markdown]
# ## 7. Coercing numbers: `"$52,000"`, `"75k"`, `"unknown"`
#
# `pd.to_numeric(..., errors="coerce")` turns anything unparseable into `NaN` instead of raising. That is convenient and dangerous: it also hides
# junk you did not know about. So first we handle the patterns we *do* understand, then coerce, then inspect what got coerced.

# %%
def parse_money(s: pd.Series) -> pd.Series:
    """'$52,000' -> 52000, '75k' -> 75000, 'unknown' -> NaN."""
    t = s.str.strip().str.lower().str.replace(r"[$,\s]", "", regex=True)
    has_k = t.str.endswith("k")
    number = pd.to_numeric(t.str.removesuffix("k"), errors="coerce")
    return number.where(~has_k, number * 1000)


salary_clean = parse_money(messy["salary"])
print(salary_clean.describe().round(0).to_string())

# what did coercion throw away? (present before, NaN after)
lost = messy.loc[messy["salary"].notna() & salary_clean.isna(), "salary"].unique()
print("values turned into NaN:", lost)

# %% [markdown]
# The only thing coerced away is `"unknown"`, exactly what we intended. If that list had contained something surprising, we would have found a new
# pattern to handle. **Always inspect what `errors="coerce"` swallowed.**
#
# The `age` column has *sentinel values*: numbers that mean "no data" rather than real ages. Look at the extremes:

# %%
print(messy["age"].describe().round(1).to_string())
print(messy["age"].value_counts().head(3).to_string())

# %% [markdown]
# `-1` and `999` are not ages. Replace them with `NaN` so they cannot distort averages (those sentinels inflate the mean age to well over 100).

# %%
age_clean = messy["age"].where(messy["age"].between(0, 120))    # keep plausible ages, everything else -> NaN
print("before:", messy["age"].mean().round(1), "| after:", age_clean.mean().round(1))
print("plausible range 0-120 kept", int(age_clean.notna().sum()), "of", int(messy["age"].notna().sum()), "non-missing values")

# %% [markdown]
# ## 8. Mixed date formats
#
# The `signup` column mixes four formats plus the text `"not available"`. `pd.to_datetime(format="mixed")` guesses per value, but guessing is risky
# with day/month ambiguity (`03/04/2024`). A safer approach is to **try each known format explicitly** and combine the results.

# %%
formats = ["%Y-%m-%d", "%d/%m/%Y", "%Y.%m.%d", "%b %d, %Y"]
signup_clean = pd.Series(pd.NaT, index=messy.index, dtype="datetime64[ns]")
for fmt in formats:
    parsed = pd.to_datetime(messy["signup"], format=fmt, errors="coerce")
    signup_clean = signup_clean.fillna(parsed)              # each pass only fills what earlier formats could not

still_bad = messy.loc[messy["signup"].notna() & signup_clean.isna(), "signup"].unique()
print("unparsed non-missing values:", still_bad)
print(signup_clean.value_counts(dropna=False).to_string())

# %% [markdown]
# *What just happened?* Each explicit format only succeeds on strings that match it, so `fillna` stitches the pieces together with no guessing.
# Only `"not available"` (and true blanks) remain as `NaT` ("not a time", the datetime missing marker). In this file the day-first dates such as `15/01/2024`
# are unambiguous because the day exceeds 12; if your file had `03/04/2024` you must ask the data owner which convention applies.

# %% [markdown]
# ## 9. Duplicates
# `duplicated()` marks repeat rows (the *second* and later copy by default). Check whole-row duplicates and duplicates on a key.

# %%
print("full-row duplicates:", int(messy.duplicated().sum()))
print("duplicate ids      :", int(messy.duplicated(subset="id").sum()))
messy.loc[messy.duplicated(keep=False)].sort_values("id").head(4)

# %% [markdown]
# `keep=False` shows **all** copies so you can compare them before deleting anything. Here the copies are identical. If two rows share an id but differ
# in other columns, that is a *conflict* you must resolve (latest record? most complete?), not a plain duplicate. Test for conflicts before you drop:

# %%
n_distinct_per_id = messy.groupby("id").nunique(dropna=False).max(axis=1)
print("ids whose copies disagree on some column:", int((n_distinct_per_id > 1).sum()))

# %% [markdown]
# ## 10. Putting it together: one cleaning function
# Wrap the steps in a function so the process is repeatable and testable, then verify the result.

# %%
def clean_messy(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()
    df["name"] = df["name"].str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    df["country"] = df["country"].str.strip().str.lower().str.replace(".", "", regex=False).map(country_map)
    df["salary"] = parse_money(df["salary"])
    df["age"] = df["age"].where(df["age"].between(0, 120))
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for fmt in formats:
        parsed = parsed.fillna(pd.to_datetime(df["signup"], format=fmt, errors="coerce"))
    df["signup"] = parsed
    return df


clean = clean_messy(messy)
print(clean.shape, "| rows removed:", len(messy) - len(clean))
print(clean.dtypes.to_string())
clean.head()

# %% [markdown]
# ## 11. Missing values: choose a strategy on purpose
#
# First measure how much is missing and *where*.

# %%
print((clean.isna().mean() * 100).round(1).to_string())

# %% [markdown]
# Options and what each one costs:
#
# | Strategy | When it is reasonable | Cost |
# |---|---|---|
# | **Drop rows** (`dropna`) | very few missing and it looks random | loses data; biases results if missingness is not random |
# | **Fill with a constant** (`fillna(0)`, `"Unknown"`) | a missing value has a real meaning (no purchases = 0) | wrong if 0 is not the true meaning |
# | **Fill with mean / median** | a numeric feature, modest missingness | shrinks the variance; hides uncertainty; use median for skewed data |
# | **Fill by group** (median within country) | groups differ | more code; small groups are noisy |
# | **Add an indicator flag** (`age_missing`) | missingness itself is informative | one more column |
# | **Leave as NaN** | the downstream tool handles it (many tree models) | you must know it does |
#
# Let us see the variance shrinkage claim with real numbers instead of taking it on faith:

# %%
age = clean["age"]
age_filled = age.fillna(age.median())
print(f"std before fill: {age.std():.2f} | after median fill: {age_filled.std():.2f}")
print(f"missing share: {age.isna().mean():.1%}")

# %% [markdown]
# Median filling keeps the centre but manufactures a spike of identical values, so the spread shrinks. Combine with a flag so a model can learn from the fact that the
# value was missing, and fill within groups when groups clearly differ:

# %%
clean = clean.assign(
    age_missing=clean["age"].isna(),
    age_filled=clean["age"].fillna(clean.groupby("country")["age"].transform("median")),
)
print(clean[["age", "age_missing", "age_filled"]].head(6).to_string())
print("still missing after group fill:", int(clean["age_filled"].isna().sum()))

# %% [markdown]
# **Missingness mechanisms** (worth memorising for interviews): *MCAR* (missing completely at random: safe to drop), *MAR* (depends on other observed columns: fill using those
# columns), *MNAR* (depends on the missing value itself, e.g. high earners hiding salary: no simple fix, be honest about the bias).
#
# Back to churn: `total_charges` is missing for some customers. Is there a natural value to fill it with?

# %%
miss = churn[churn["total_charges"].isna()]
print(len(miss), "rows; tenure of these rows:")
print(miss["tenure_months"].describe().round(1).to_string())

# %%
seen = churn.dropna(subset=["total_charges"])
ratio = seen["total_charges"] / (seen["tenure_months"] * seen["monthly_charges"]).where(lambda s: s > 0)
print("total_charges / (tenure * monthly) for observed rows:")
print(ratio.describe().round(3).to_string())

# %% [markdown]
# For customers with observed values, `total_charges` is close to `tenure * monthly_charges` (the ratio printed above stays roughly between 0.95 and 1.05). So filling a missing total with
# `tenure * monthly_charges` is a **defensible, explainable imputation**, better than a global mean. It is still an assumption: prices change over time, so
# record it (add a flag column) rather than pretending the value was observed. In the exercises you will do this.

# %% [markdown]
# ## Common mistakes
#
# 1. **Overwriting the raw data** or cleaning without a repeatable script.
# 2. **`errors="coerce"` without looking at what was coerced** (you may be deleting a real pattern).
# 3. **`fillna(mean)` on skewed data or before splitting train/test** (the fill value leaks information from the test set; compute it on training data only).
# 4. **Fixing the symptom, not the cause**: converting `"None"` to NaN and then filling it hides a real category.
# 5. **Using `.str.title()` or `.lower()` alone as standardisation**: without a mapping you cannot detect new variants.
# 6. **Deduplicating before checking for conflicting duplicates**: you may keep the wrong version of a record.
# 7. **Treating sentinels (`-1`, `999`, `0`, `"N/A"`) as real numbers.** Check `describe()` extremes.
# 8. **Chained assignment** such as `df[df.age > 100]["age"] = np.nan`. In pandas 3 (copy-on-write) it never modifies `df`. Use `df.loc[mask, "age"] = np.nan` or `where`.

# %%
demo = pd.DataFrame({"age": [30, 999, 45]})
demo[demo["age"] > 100]["age"] = np.nan          # chained assignment: modifies a temporary copy only
print(demo["age"].tolist())                       # unchanged in pandas 3
demo.loc[demo["age"] > 100, "age"] = np.nan       # the correct way
print(demo["age"].tolist())

# %% [markdown]
# ## Interview corner
#
# **1. How would you handle a column with 30% missing values?**
# <details><summary>Answer</summary>
# First understand why it is missing (MCAR / MAR / MNAR) and whether the missing group differs on the target. Then choose: drop the column if it carries little signal,
# impute (median, group-wise, or a model) plus a missingness flag, or use an algorithm that handles NaN natively. Fit imputers on training data only. Report the assumption.
# </details>
#
# **2. What does `pd.to_numeric(errors="coerce")` do and what is the risk?**
# <details><summary>Answer</summary>
# It converts values that cannot be parsed to NaN instead of raising. The risk is silently discarding legitimate values you did not anticipate. Inspect which non-null
# inputs became NaN.
# </details>
#
# **3. `df.duplicated()` versus `df.drop_duplicates()`?**
# <details><summary>Answer</summary>
# <code>duplicated</code> returns a boolean mask (True for repeats after the first occurrence by default) so you can inspect; <code>drop_duplicates</code> removes them.
# Use <code>subset=</code> for key-based duplicates and <code>keep=False</code> to see every copy.
# </details>
#
# **4. Why can filling with the mean be a bad idea?**
# <details><summary>Answer</summary>
# It shrinks variance and weakens correlations, is distorted by outliers, ignores group differences, and if computed on the full dataset before a train/test split
# it leaks test information. Median, group-wise fills, and indicator flags are often better.
# </details>

# %% [markdown]
# ## Exercises
# Data objects available from above: `churn` (with `contract` already cleaned and `"None"` preserved), `messy` (raw), `clean_messy`, `parse_money`.

# %% [markdown]
# ### Exercise 1: Contract mix
# On the cleaned `churn`, store in `contract_share` the share of customers on each contract type (a Series indexed by contract, values sum to 1)
# and in `top_contract` the name of the most common one. *Hint: `value_counts(normalize=True)`.*

# %% tags=["starter"]
contract_share = ...
top_contract = ...

# %%
counts = churn["contract"].value_counts()
assert set(contract_share.index) == {"Month-to-month", "One year", "Two year"}, "the cleaned contract should have exactly 3 categories"
assert abs(contract_share.sum() - 1) < 1e-9, "shares should sum to 1"
assert abs(contract_share["Two year"] - counts["Two year"] / len(churn)) < 1e-9, "check the share of Two year"
assert top_contract == counts.idxmax(), f"expected {counts.idxmax()}"
print(f"Exercise 1 passed ({top_contract}: {contract_share[top_contract]:.1%})")

# %% tags=["solution"]
contract_share = churn["contract"].value_counts(normalize=True)
top_contract = contract_share.idxmax()

# %% [markdown]
# ### Exercise 2: Average salary, honestly
# Using `parse_money`, compute `mean_salary` for `messy` **after removing duplicate rows** (`drop_duplicates()` first), ignoring the `"unknown"` values.
# Also store `n_unknown`: how many salary entries in the de-duplicated table could not be parsed to a number.

# %% tags=["starter"]
mean_salary = ...
n_unknown = ...

# %%
dedup = messy.drop_duplicates()
manual = []
for v in dedup["salary"].dropna():
    t = str(v).lower().replace("$", "").replace(",", "").strip()
    if t == "unknown":
        continue
    manual.append(float(t[:-1]) * 1000 if t.endswith("k") else float(t))
assert abs(mean_salary - np.mean(manual)) < 1e-6, f"expected {np.mean(manual):.2f}, got {mean_salary:.2f}"
assert n_unknown == len(dedup) - len(manual), "n_unknown = rows minus rows with a parsable salary"
print(f"Exercise 2 passed (mean {mean_salary:,.0f}, {n_unknown} unknown)")

# %% tags=["solution"]
dedup = messy.drop_duplicates()
sal = parse_money(dedup["salary"])
mean_salary = sal.mean()
n_unknown = int(sal.isna().sum())

# %% [markdown]
# ### Exercise 3: A tidy people table
# Call `clean_messy(messy)` and then produce `by_country`: the number of people per (standardised) country, sorted from largest to smallest.
# Also create `adults`: the rows of the cleaned table with a known `age` of at least 18.
# *Hint: `value_counts()` sorts descending by default.*

# %% tags=["starter"]
cleaned = clean_messy(messy)
by_country = ...
adults = ...

# %%
assert set(by_country.index) == {"United States", "India", "United Kingdom", "Bangladesh"}, "countries not standardised"
assert by_country.sum() == len(messy.drop_duplicates()), "each de-duplicated row belongs to exactly one country"
assert list(by_country.values) == sorted(by_country.values, reverse=True), "sort from largest to smallest"
assert (adults["age"] >= 18).all() and len(adults) == int((cleaned["age"] >= 18).sum()), "adults: age known and >= 18"
print("Exercise 3 passed")

# %% tags=["solution"]
cleaned = clean_messy(messy)
by_country = cleaned["country"].value_counts()
adults = cleaned[cleaned["age"] >= 18]

# %% [markdown]
# ### Exercise 4: Impute `total_charges` and flag it
# In `churn`, create `total_filled`: `total_charges` where known, otherwise `tenure_months * monthly_charges`. Create `was_imputed` (boolean Series):
# True where `total_charges` was missing. Do not modify `churn` itself (use `assign` or work on a copy).

# %% tags=["starter"]
total_filled = ...
was_imputed = ...

# %%
orig_missing = churn["total_charges"].isna()
assert churn["total_charges"].isna().sum() == orig_missing.sum() > 0, "leave churn untouched"
assert total_filled.notna().all(), "no missing values should remain"
assert (was_imputed == orig_missing).all(), "was_imputed marks exactly the originally missing rows"
assert np.allclose(total_filled[~orig_missing], churn.loc[~orig_missing, "total_charges"]), "observed values must not change"
assert np.allclose(total_filled[orig_missing], (churn["tenure_months"] * churn["monthly_charges"])[orig_missing]), "fill = tenure * monthly"
print(f"Exercise 4 passed ({int(was_imputed.sum())} values imputed)")

# %% tags=["solution"]
was_imputed = churn["total_charges"].isna()
total_filled = churn["total_charges"].fillna(churn["tenure_months"] * churn["monthly_charges"])

# %% [markdown]
# ### Exercise 5 (open-ended): Your cleaning log
# Pick `internet_service` in `churn`. Decide how you would treat the genuinely empty cells (drop, a new "Unknown" category, impute from other columns, ask the data owner?).
# In a markdown cell write three sentences: what you would do, what assumption it makes, and one check you would run to see whether the decision changes a result (for example, the churn rate).

# %% [markdown]
# ## Recap
# - Standardise text with `strip` / `lower` / `str.replace`, then map through an **explicit dictionary** and assert nothing is unmapped.
# - `read_csv` converts strings like `"None"` to NaN by default; control it with `keep_default_na` and `na_values`.
# - `to_numeric(errors="coerce")` and `to_datetime(format=..., errors="coerce")` are powerful; always inspect what became `NaN`/`NaT`.
# - Look for sentinels (`-1`, `999`), duplicates (check for conflicts first) and inconsistent formats.
# - Missing-value handling is a modelling decision: know what each option costs, add a flag, and never fit fill values on test data.
# - Never edit raw data; wrap the steps in a function.
#
# **Go deeper:** the pandas user guide sections "Working with missing data", "Working with text data" and "Time series / date functionality" (pandas.pydata.org/docs/user_guide).
#
# **Next:** [Transforming columns](03_transforming_columns.ipynb)
