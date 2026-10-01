# %% [markdown]
# # Feature Engineering and Pipelines
#
# **Why this matters.** Real tables arrive with missing values, inconsistent labels, text categories and columns on wildly different scales. Models need clean
# numbers. The professional way to get there is a single `Pipeline` object that takes the *raw* table in and produces predictions out: it cannot leak test information
# (each step is fitted on training data only), it can be cross-validated as one unit, and you can ship it as one file. Hand-made preprocessing scattered across a notebook
# is the main reason models cannot be reproduced or deployed.
#
# **You will learn**
# - build a `ColumnTransformer` with different treatments for numeric, nominal and ordinal columns
# - impute, one-hot encode, ordinal encode and scale inside a pipeline, and handle categories never seen in training
# - write custom steps with `FunctionTransformer` (including cleaning the dirty `contract` column *inside* the pipeline)
# - cross-validate the whole pipeline, then save it with `joblib` and reload it
#
# **Prerequisites:** [Cross-validation](07_cross_validation_and_model_evaluation.ipynb) and [leakage](02_train_test_split_and_data_leakage.ipynb).
#
# > Data used: `datasets/churn.csv` read *raw* (synthetic; dirty on purpose).

# %%
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

DATA = Path("../datasets")
# "None" in internet_service is a real category (no internet plan); pandas would read it as missing by default.
# keep_default_na=False + na_values=[""] keeps "None" as text and treats only truly blank cells as missing.
raw = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])
y = raw["churned"]
X = raw.drop(columns=["churned", "customer_id"])   # customer_id is an identifier, not a feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)
print("missing values per column:", X_train.isna().sum()[lambda s: s > 0].to_dict())
print("contract labels:", X_train["contract"].value_counts().to_dict())

# %% [markdown]
# Three problems to solve, all *inside* the model: (1) `total_charges` has blanks, (2) `internet_service` has blanks (distinct from the real "None" category),
# (3) `contract` has inconsistent casing (`Month-to-month` vs `month-to-month`).

# %% [markdown]
# ## 1. The hand-made way, and why it breaks
#
# The tempting approach is `pd.get_dummies(X_train)` and `pd.get_dummies(X_test)` separately. It fails as soon as the two sets do not contain the same categories:

# %%
train_dummies = pd.get_dummies(X_train["internet_service"].fillna("Unknown"), dtype=float)
new_batch = X_test.iloc[:200].query("internet_service == 'Fiber'")         # e.g. a day's scoring batch with only one category
batch_dummies = pd.get_dummies(new_batch["internet_service"].fillna("Unknown"), dtype=float)
print("training columns:", list(train_dummies.columns))
print("batch columns   :", list(batch_dummies.columns), "-> the model would receive the wrong number of columns")

# %% [markdown]
# A fitted `OneHotEncoder` *remembers* the training categories, so it always emits the same columns, and can be told what to do with unseen values. That memory is the whole point of the
# fit/transform split, and a `Pipeline` stores it for every step.

# %% [markdown]
# ## 2. Building blocks
#
# | Step | Learns at `fit` | Used for |
# |---|---|---|
# | `SimpleImputer(strategy="median")` | column medians | fill numeric blanks (`add_indicator=True` also adds a "was missing" flag) |
# | `SimpleImputer(strategy="constant", fill_value="Unknown")` | nothing | label blank categories explicitly |
# | `OneHotEncoder(handle_unknown="ignore")` | the category list | nominal categories (no order) |
# | `OrdinalEncoder(categories=[[...]])` | the category order you give | categories with a natural order |
# | `StandardScaler()` | mean and std | linear models, distance-based models (trees do not need it) |
# | `FunctionTransformer(func)` | nothing | any stateless custom step |
#
# `ColumnTransformer` routes different columns to different steps and concatenates the results.
#
# First, a custom cleaning step. Stateless functions fit naturally into `FunctionTransformer`; this one normalises the contract text and adds two engineered features:
# tenure flag for new customers, and support calls per year of tenure (a rate, which can tell more than the raw count).

# %%
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["contract"] = out["contract"].str.strip().str.lower()                    # fixes 'Month-to-month' vs 'month-to-month'
    out["is_new_customer"] = (out["tenure_months"] <= 6).astype(float)
    out["calls_per_year"] = out["support_calls"] / out["tenure_months"].clip(lower=1) * 12
    return out

cleaner = FunctionTransformer(clean_and_engineer)
cleaned = cleaner.fit_transform(X_train)
print("contract labels after cleaning:", cleaned["contract"].value_counts().to_dict())

# %% [markdown]
# *What just happened?* The function is applied identically at train and serving time because it is part of the pipeline, which is exactly why fixing the casing *inside* the
# pipeline beats cleaning the CSV by hand: a dirty row arriving in production will be fixed the same way. It has no parameters learned from data, so it cannot leak.

# %% [markdown]
# ## 3. The full preprocessing + model pipeline

# %%
numeric = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "calls_per_year"]
binary = ["senior_citizen", "is_new_customer"]
nominal = ["internet_service", "paperless_billing", "autopay"]
ordinal = ["contract"]
CONTRACT_ORDER = ["month-to-month", "one year", "two year"]


def make_preprocessor(contract_as="ordinal"):
    contract_step = (
        OrdinalEncoder(categories=[CONTRACT_ORDER], handle_unknown="use_encoded_value", unknown_value=-1)
        if contract_as == "ordinal" else OneHotEncoder(handle_unknown="ignore")
    )
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
        ("bin", "passthrough", binary),
        ("nom", make_pipeline(SimpleImputer(strategy="constant", fill_value="Unknown"), OneHotEncoder(handle_unknown="ignore")), nominal),
        ("ord", contract_step, ordinal),
    ])


pipe = Pipeline([
    ("clean", FunctionTransformer(clean_and_engineer)),
    ("prep", make_preprocessor()),
    ("model", LogisticRegression(max_iter=2000)),
])
pipe.fit(X_train, y_train)
print("test AUC:", round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]), 3))
print("features fed to the model:", len(pipe["prep"].get_feature_names_out()))
list(pipe["prep"].get_feature_names_out())

# %% [markdown]
# Reading the names: `num__...` scaled numbers (plus `missingindicator_total_charges`, a flag that records *that* a value was blank, which can itself be informative),
# `bin__...` untouched flags, `nom__...` one-hot columns (including `internet_service_Unknown` for the truly blank cells, separate from `None`), and `ord__contract`.
#
# **Why it is leak-safe by construction:** `pipe.fit(X_train, y_train)` fits the imputer medians, scaler statistics and encoder categories on training rows only;
# `pipe.predict(X_test)` only *applies* them. Put the pipeline in cross-validation and every fold refits all of it on that fold's training part:

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"5-fold CV AUC of the whole pipeline: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# ## 4. Encoding choices are hypotheses: test them
#
# Is the contract better as an ordered number or as unordered one-hot columns? And does a tree model care? Because the preprocessor is a function, the experiment is three lines:

# %%
rows = []
for contract_as in ["ordinal", "onehot"]:
    for name, est in [("logistic", LogisticRegression(max_iter=2000)),
                      ("hist GB", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, random_state=42))]:
        p = Pipeline([("clean", FunctionTransformer(clean_and_engineer)), ("prep", make_preprocessor(contract_as)), ("model", est)])
        sc = cross_val_score(p, X_train, y_train, cv=cv, scoring="roc_auc")
        rows.append({"contract encoding": contract_as, "model": name, "CV AUC": sc.mean(), "std": sc.std()})
pd.DataFrame(rows).round(3)

# %% [markdown]
# Compare the rows with the spread (`std`): if a difference is smaller than the fold-to-fold noise, prefer the simpler or more interpretable choice. An ordinal code assumes
# equal spacing between contract lengths, which one-hot does not; whichever the data prefers, you now measured instead of guessing.
#
# **Does the engineered feature help?** The same discipline applies to feature engineering: add, cross-validate, keep only what earns its place.

# %%
def drop_engineered(df):
    out = clean_and_engineer(df)
    return out.assign(calls_per_year=out["support_calls"].astype(float), is_new_customer=0.0)   # neutralise the two new columns

baseline_pipe = Pipeline([("clean", FunctionTransformer(drop_engineered)), ("prep", make_preprocessor()), ("model", LogisticRegression(max_iter=2000))])
base_sc = cross_val_score(baseline_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"without engineered features: {base_sc.mean():.3f} +/- {base_sc.std():.3f}   with: {scores.mean():.3f} +/- {scores.std():.3f}")

# %% [markdown]
# Read the two numbers against their spreads: here the engineered features do not move the validation score, so on this evidence they are complexity, not progress (a fine result: you learned it cheaply).

# %% [markdown]
# ## 5. Unseen categories and dirty rows at serving time
#
# A production pipeline meets inputs training never showed it. Score a hand-written row with an internet plan we have never seen and a shouting contract label:

# %%
new_customer = pd.DataFrame([{
    "tenure_months": 3, "contract": "  MONTH-TO-MONTH ", "internet_service": "Satellite", "monthly_charges": 70.0,
    "total_charges": np.nan, "support_calls": 3, "paperless_billing": "Yes", "autopay": "No", "senior_citizen": 0,
}])
print("churn probability for the odd row:", round(float(pipe.predict_proba(new_customer)[0, 1]), 3))

# %% [markdown]
# It works because the cleaner normalised the contract, the imputer filled the blank `total_charges`, and `OneHotEncoder(handle_unknown="ignore")` encoded the unseen plan as all zeros.
# (Without `handle_unknown="ignore"` the encoder raises an error, which is safer in some settings: decide deliberately and monitor how often unseen values occur.)

# %% [markdown]
# ## 6. Save, reload, reuse
#
# `joblib` serialises the *entire* fitted pipeline: imputer medians, encoder categories, scaler statistics and model weights together. We write to a temporary directory and clean up.

# %%
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "churn_pipeline.joblib"
    joblib.dump(pipe, path)
    size_kb = path.stat().st_size / 1024
    loaded = joblib.load(path)
    same = np.allclose(pipe.predict_proba(X_test), loaded.predict_proba(X_test))
print(f"saved file ~{size_kb:.0f} KB; reloaded pipeline gives identical predictions: {same}")

# %% [markdown]
# *What just happened?* The reloaded object predicts directly from the raw table, including the cleaning step, so the serving code is just `loaded.predict_proba(raw_df)`.
# Caveat: `joblib`/pickle files are tied to library versions (load with the same scikit-learn version you saved with), and **loading a pickle from an untrusted source can execute code**.
# Record versions alongside the artifact; the MLOps module covers model registries and safer packaging.

# %% [markdown]
# ## Common mistakes
#
# 1. **`get_dummies` separately on train and test**: mismatched columns (shown in section 1).
# 2. **`fit_transform` on the test set**, or fitting the imputer/scaler on all data before the split.
# 3. **Scaling for tree models** (harmless but pointless) and **forgetting to scale for linear or distance models**.
# 4. **Ordinal-encoding nominal categories** (e.g. `Fiber=2 > DSL=1`) which invents an order the model will exploit.
# 5. **Dropping rows with missing values** when the missingness itself is informative; impute and add an indicator instead.
# 6. **Leaving identifiers** (`customer_id`) in the features.

# %%
# Mistake 4 in code: an arbitrary integer code for a nominal column is a hidden ordering
codes = {"None": 0, "DSL": 1, "Fiber": 2}
print("fake order makes Fiber 'twice' DSL:", X_train["internet_service"].map(codes).dropna().astype(int).value_counts().sort_index().to_dict())

# %% [markdown]
# ## Interview corner
#
# **1. Why use a `Pipeline` instead of preprocessing the DataFrame first?**
# <details><summary>Answer</summary>
# It guarantees every transformation is fitted on training data only (no leakage, also inside cross-validation), makes train and serving use the same code path, and lets you tune and ship
# preprocessing and model as one object.
# </details>
#
# **2. One-hot vs ordinal vs target encoding?**
# <details><summary>Answer</summary>
# One-hot for nominal categories of moderate cardinality; ordinal when the order is meaningful (contract length, education level); target/mean encoding for very high-cardinality columns,
# but it uses the target, so it must be fitted inside CV folds (scikit-learn has a <code>TargetEncoder</code> that does cross-fitting) or it leaks.
# </details>
#
# **3. How do you deal with a category in production that was not in training?**
# <details><summary>Answer</summary>
# Decide at design time: <code>handle_unknown="ignore"</code> (all zeros), an explicit "other" bucket, or fail loudly. Whichever you choose, log and monitor the rate of unseen values as a drift signal.
# </details>
#
# **4. When is imputing with the median a bad idea?**
# <details><summary>Answer</summary>
# When missingness is informative (add a missing-indicator), when a large share is missing (the imputed constant dominates), or when values are missing for structural reasons
# (e.g. "no internet" implies no internet-related charges: use a category or 0, not the median).
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Unseen categories
# Fit `OneHotEncoder(handle_unknown="ignore")` on `X_train[["internet_service"]].fillna("Unknown")`. Transform a new frame with one row whose `internet_service` is `"Satellite"`.
# Store in `n_out_cols` the number of output columns (use `.toarray()` or `sparse_output=False`) and in `row_total` the sum of that encoded row.

# %% tags=["starter"]
n_out_cols = ...
row_total = ...

# %%
_cats = X_train["internet_service"].fillna("Unknown").nunique()
assert n_out_cols == _cats, f"one column per training category ({_cats})"
assert row_total == 0, "an unseen category is encoded as all zeros"
print(f"Exercise 1 passed: {n_out_cols} columns, unseen row sums to {row_total}")

# %% tags=["solution"]
_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[["internet_service"]].fillna("Unknown"))
_out = _enc.transform(pd.DataFrame({"internet_service": ["Satellite"]}))
n_out_cols = _out.shape[1]
row_total = float(_out.sum())

# %% [markdown]
# ### Exercise 2: Write a cleaner
# Define `fix_contract(df)` that returns a **copy** of `df` where `contract` is stripped of spaces and lower-cased (and does not modify the input). Then apply it to `X_train` and store
# in `n_labels` the number of distinct contract values afterwards, and in `n_changed` how many rows had their contract text changed.

# %% tags=["starter"]
def fix_contract(df):
    ...

n_labels = ...
n_changed = ...

# %%
_before = X_train["contract"].copy()
_res = fix_contract(X_train)
assert isinstance(_res, pd.DataFrame) and _res is not X_train, "return a new DataFrame"
assert (X_train["contract"] == _before).all(), "do not modify the input DataFrame"
assert n_labels == _before.str.strip().str.lower().nunique(), "number of distinct cleaned labels"
assert n_changed == int((_before != _before.str.strip().str.lower()).sum()), "rows whose text differs after cleaning"
print(f"Exercise 2 passed: {n_labels} labels after cleaning; {n_changed} rows changed")

# %% tags=["solution"]
def fix_contract(df):
    out = df.copy()
    out["contract"] = out["contract"].str.strip().str.lower()
    return out

_fixed = fix_contract(X_train)
n_labels = _fixed["contract"].nunique()
n_changed = int((_fixed["contract"] != X_train["contract"]).sum())

# %% [markdown]
# ### Exercise 3: A numeric-only pipeline
# Build a pipeline `make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))` that uses only the columns
# `["tenure_months", "monthly_charges", "total_charges", "support_calls"]`, and cross-validate it on `X_train`/`y_train` with `cv` (the `StratifiedKFold` defined above) and ROC-AUC.
# Store the mean in `num_cv_auc`, and set `full_beats_numeric = True` if the full pipeline's mean CV AUC (`scores.mean()`) is higher.

# %% tags=["starter"]
num_cv_auc = ...
full_beats_numeric = ...

# %%
_p = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
_exp = cross_val_score(_p, X_train[_cols], y_train, cv=cv, scoring="roc_auc").mean()
assert abs(num_cv_auc - _exp) < 1e-9, "select the 4 numeric columns, then cross_val_score(..., cv=cv, scoring='roc_auc')"
assert full_beats_numeric == bool(scores.mean() > _exp), "compare with the full pipeline's scores.mean()"
print(f"Exercise 3 passed: numeric-only {num_cv_auc:.3f} vs full pipeline {scores.mean():.3f}")

# %% tags=["solution"]
_numeric_only = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
num_cv_auc = float(cross_val_score(_numeric_only, X_train[["tenure_months", "monthly_charges", "total_charges", "support_calls"]], y_train, cv=cv, scoring="roc_auc").mean())
full_beats_numeric = bool(scores.mean() > num_cv_auc)

# %% [markdown]
# ### Exercise 4: Ship it
# Save `pipe` with `joblib` into a `tempfile.TemporaryDirectory()`, reload it **inside** the `with` block, and predict churn probabilities for the first 5 rows of `X_test` *after* upper-casing their `contract`
# text (simulating dirty input). Store the array in `probs`. The numbers must equal what the original `pipe` gives for the same dirty rows.

# %% tags=["starter"]
probs = ...

# %%
_dirty = X_test.head(5).copy()
_dirty["contract"] = _dirty["contract"].str.upper()
assert np.allclose(probs, pipe.predict_proba(_dirty)[:, 1]), "predict_proba(dirty_rows)[:, 1] from the reloaded pipeline"
assert len(probs) == 5, "five rows"
print("Exercise 4 passed:", np.round(probs, 3))

# %% tags=["solution"]
_dirty_rows = X_test.head(5).copy()
_dirty_rows["contract"] = _dirty_rows["contract"].str.upper()
with tempfile.TemporaryDirectory() as _tmp:
    joblib.dump(pipe, Path(_tmp) / "p.joblib")
    probs = joblib.load(Path(_tmp) / "p.joblib").predict_proba(_dirty_rows)[:, 1]

# %% [markdown]
# ### Exercise 5 (open-ended): Earn a feature
# Invent one more feature in `clean_and_engineer` (ideas: average monthly spend `total_charges / tenure`, a high-charges flag, `support_calls` interacted with month-to-month status), add it to the right list in the column
# groups, and compare 5-fold CV AUC with and without it. Write two sentences: did it help beyond the fold noise, and would you keep it given the added complexity?

# %% [markdown]
# ## Recap
# - A `Pipeline` + `ColumnTransformer` takes the raw table to predictions, fits every step on training data only, and works inside cross-validation: leakage-safe by construction.
# - Use imputers (median / constant), one-hot for nominal, ordinal with explicit order for ordered categories, scalers for linear models, `handle_unknown="ignore"` for robustness.
# - `FunctionTransformer` puts stateless cleaning and feature engineering in the pipeline, so serving applies exactly what training did.
# - Treat every encoding or feature as a hypothesis and measure it with CV against the spread.
# - `joblib` saves the whole fitted pipeline; version-pin it and never load untrusted pickles.
#
# **Go deeper:** scikit-learn user guide "Pipelines and composite estimators" and "Preprocessing data" (scikit-learn.org/stable/modules/compose.html).
#
# **Next:** [Hyperparameter tuning](09_hyperparameter_tuning.ipynb)
