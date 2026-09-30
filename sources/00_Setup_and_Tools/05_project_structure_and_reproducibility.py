# %% [markdown]
# # Project Structure and Reproducibility
#
# **Why this matters.** A folder called `final_v3_REAL.ipynb` next to `data_clean2.csv` is how most analyses start and how most
# of them become unrepeatable. Teams expect a predictable layout, configuration outside the code, fixed random seeds, logs you can read
# after a job failed at 3 a.m., and a record of what data and versions produced each result. This notebook builds all of that in a
# tiny working project.
#
# **You will learn**
# - a cookiecutter-data-science style folder layout and why each folder exists
# - keeping settings in a YAML config, anchoring paths so code runs from any working directory
# - seeding random number generators, and what seeds do **not** guarantee
# - the `logging` module instead of `print`, set up once and without duplicate handlers
# - writing a run **manifest** (config, versions, data hash, metrics) so a result can be traced and re-checked
#
# **Prerequisites:** [Environments and packages](03_environments_and_packages.ipynb), [Jupyter and VS Code workflow](04_jupyter_and_vscode_workflow.ipynb).
#
# > Data used: `datasets/churn.csv`, synthetic telecom customers (see `docs/DATASETS.md`). Everything is created in a temp folder.

# %%
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATA = Path("../datasets").resolve()
tmp = tempfile.TemporaryDirectory()
base = Path(tmp.name)

# %% [markdown]
# ## 1. A layout that scales
#
# ```
# churn-project/
# ├── README.md              what it is, how to run it (first thing a newcomer reads)
# ├── pyproject.toml         package metadata + dependency ranges
# ├── requirements.txt       pinned versions for exact reproduction
# ├── .gitignore             data/, .env, caches, .venv
# ├── configs/base.yaml      all tunable settings: no magic numbers in code
# ├── data/
# │   ├── raw/               immutable originals: never edit by hand
# │   ├── interim/           partly processed
# │   └── processed/         final, model-ready tables
# ├── notebooks/             numbered exploration: 01_eda.ipynb, 02_baseline.ipynb
# ├── src/churn/             importable code: config, data, features, models
# ├── tests/                 pytest tests for src/
# ├── reports/               figures, run manifests, logs
# └── models/                serialized models (gitignored or tracked with DVC / a registry)
# ```
#
# The principles behind it:
# 1. **Raw data is read-only.** Every processed file must be reproducible by running code on the raw data.
# 2. **Notebooks explore, `src/` computes.** Logic used twice moves from a notebook into `src/` and gets a test.
# 3. **Config is data, not code.** Changing the test size should be a one-line YAML edit, not a code change.
# 4. **Everything in one repo folder is relative to the project root**, not to whoever's laptop it is.
#
# Let's scaffold a real one in a temp folder. The scaffold function writes the files we will use in the rest of the notebook.

# %%
def scaffold(root: Path, pkg: str = "myproj"):
    for d in ["configs", "data/raw", "data/interim", "data/processed", "notebooks",
              f"src/{pkg}", "tests", "reports", "models"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    files = {
        "README.md": f"# {pkg}\n\nRun: `python -m {pkg}.train --config configs/base.yaml`\n",
        ".gitignore": "data/\nmodels/\nreports/\n.env\n.venv/\n__pycache__/\n.ipynb_checkpoints/\n",
        "requirements.txt": "# pinned by `pip freeze` in the real project\n",
        f"src/{pkg}/__init__.py": '"""Churn model package."""\n',
    }
    for rel, text in files.items():
        (root / rel).write_text(text, encoding="utf-8")


proj = base / "churn-project"
scaffold(proj)
for path in sorted(proj.rglob("*")):
    depth = len(path.relative_to(proj).parts) - 1
    print("  " * depth + path.name + ("/" if path.is_dir() else ""))

# %% [markdown]
# *What just happened?* A layout is just directories and a few conventions. Tools such as `cookiecutter` (with the "data science" template)
# generate similar trees; you do not need them to start, and you should adapt the structure to your team instead of copying it blindly.
# Small analyses do not need every folder: a one-notebook project with a `data/` folder and a README is fine. The value is *consistency*.

# %% [markdown]
# ## 2. Configuration outside the code, paths anchored to the project
#
# `config.py` below finds the project root from **its own file location** (`__file__`), so the code works no matter which folder you
# launch it from. Relative paths like `"../datasets"` break the moment you run from another directory (see notebook 01).

# %%
pkg_dir = proj / "src" / "myproj"

_ = (pkg_dir / "config.py").write_text('''\
"""Load YAML config; resolve output paths against the project root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]   # src/myproj/config.py -> project root


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("out_dir", "log_file"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg
''', encoding="utf-8")

config = {
    "seed": 42,
    "data_path": str(DATA / "churn.csv"),
    "features": ["tenure_months", "monthly_charges", "support_calls"],
    "target": "churned",
    "test_size": 0.25,
    "model": {"C": 1.0, "max_iter": 200},
    "out_dir": "reports",
    "log_file": "reports/train.log",
}
(proj / "configs" / "base.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print((proj / "configs" / "base.yaml").read_text())

# %% [markdown]
# YAML is friendlier to hand-edit than JSON (comments, no quotes), and `yaml.safe_load` only builds plain data. Never use `yaml.load` on files you
# did not write: it can construct arbitrary Python objects. Secrets (API keys, passwords) do **not** belong in this file: read them from environment variables.
#
# Prove the path anchoring: import the module from a *different* working directory and print where it thinks the root is.

# %%
env = dict(os.environ, PYTHONPATH=str(proj / "src"))
elsewhere = base  # a folder that is not the project
r = subprocess.run([sys.executable, "-c", "from myproj.config import ROOT; print(ROOT)"],
                   cwd=elsewhere, env=env, capture_output=True, text=True)
print("cwd was     :", elsewhere)
print("code says   :", r.stdout.strip())
print("same project:", Path(r.stdout.strip()) == proj.resolve())

# %% [markdown]
# ## 3. Randomness and seeds
#
# Many steps use random numbers: train/test splits, weight initialisation, bootstrap samples, shuffling. A **seed** fixes the sequence so
# the same code gives the same result. Use one explicit generator per purpose instead of hidden global state:

# %%
rng = np.random.default_rng(42)            # NumPy's modern API: a generator object you pass around
print("run 1:", rng.integers(0, 100, size=5))
print("run 2:", np.random.default_rng(42).integers(0, 100, size=5))     # same seed, same numbers
print("other:", np.random.default_rng(43).integers(0, 100, size=5))

# %% [markdown]
# Older code uses the global `np.random.seed(42)` and `np.random.rand()`. It works, but any library that also draws from the global state
# changes your numbers, so results depend on call order across your whole program. Prefer `default_rng`, and pass `random_state=42`
# to scikit-learn objects.
#
# **What a seed does not guarantee.** Python randomises string hashing per process, so the iteration order of a `set` of strings can change
# between runs even if you seeded `random`. Demonstration with two child processes:

# %%
def set_order(hashseed):
    code = "print(sorted({'a','b','c','d','e','f'}) == list({'a','b','c','d','e','f'}), *list({'a','b','c','d','e','f'}))"
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED=str(hashseed)), capture_output=True, text=True)
    return r.stdout.split()[1:]


print("PYTHONHASHSEED=1:", set_order(1))
print("PYTHONHASHSEED=2:", set_order(2))
print("PYTHONHASHSEED=1 again:", set_order(1))

# %% [markdown]
# Sets of strings iterate in a different order under different hash seeds (`PYTHONHASHSEED` pins it). Rule: **never rely on set order**; sort when order matters.
# Other sources of non-reproducibility: multi-threaded floating-point reductions, GPU kernels (deep learning frameworks have "deterministic" flags that cost speed),
# different library or BLAS versions, and data that changed under you. So a seed is necessary, not sufficient: the manifest in section 5 records the rest.

# %% [markdown]
# ## 4. Logging instead of `print`
#
# `print` goes to one place, has no severity, no timestamp, and vanishes when a scheduled job ends. The `logging` module gives levels
# (`DEBUG < INFO < WARNING < ERROR < CRITICAL`), timestamps, and multiple destinations (console + file) through **handlers**.
#
# Recipe: get a named logger per module (`logging.getLogger("myproj")`), configure handlers **once**, and never call `basicConfig` inside library code.

# %%
_ = (pkg_dir / "logs.py").write_text('''\
"""One-time logger setup (safe to call repeatedly)."""
import logging
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name, logfile=None, level=logging.INFO):
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False
    if not log.handlers:                      # idempotent: no duplicate lines when called twice
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(FORMAT))
        log.addHandler(console)
        if logfile:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(logging.Formatter(FORMAT))
            log.addHandler(fh)
    return log
''', encoding="utf-8")

# %% [markdown]
# The `if not log.handlers` guard matters in notebooks: re-running a cell that adds a handler each time gives every message printed twice, then three
# times. Let's see the bug in isolation, using the standard library only:

# %%
buggy = logging.getLogger("demo.buggy")
buggy.setLevel(logging.INFO)
buggy.propagate = False
for _ in range(3):                         # "re-running the setup cell" three times
    buggy.addHandler(logging.StreamHandler(sys.stdout))
buggy.info("this line appears once per handler")

# %% [markdown]
# Three handlers, three copies. The guarded version above is immune. Level guidance: `INFO` for milestones ("loaded 5000 rows"), `WARNING` for
# odd but survivable things (missing values dropped), `ERROR` when a step failed, `DEBUG` for details you only want when investigating.

# %% [markdown]
# ## 5. Putting it together: a reproducible training run
# The training module reads the config, seeds everything, logs milestones and writes a **run manifest**: the config, library versions, a SHA-256
# fingerprint of the input data, and the metrics. A hash detects that "the same file name" now holds different data.

# %%
_ = (pkg_dir / "train.py").write_text('''\
"""Train a small churn classifier. Usage: python -m myproj.train --config configs/base.yaml"""
import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .config import load_config
from .logs import get_logger


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger("myproj", cfg["log_file"])
    log.info("config loaded from %s", args.config)

    df = pd.read_csv(cfg["data_path"])
    cols = cfg["features"] + [cfg["target"]]
    before = len(df)
    df = df[cols].dropna()
    if len(df) < before:
        log.warning("dropped %d rows with missing values", before - len(df))

    X_tr, X_te, y_tr, y_te = train_test_split(
        df[cfg["features"]], df[cfg["target"]], test_size=cfg["test_size"], random_state=cfg["seed"], stratify=df[cfg["target"]])
    model = LogisticRegression(C=cfg["model"]["C"], max_iter=cfg["model"]["max_iter"], random_state=cfg["seed"]).fit(X_tr, y_tr)
    acc = float(accuracy_score(y_te, model.predict(X_te)))
    log.info("accuracy on %d held-out rows: %.4f", len(X_te), acc)

    manifest = {
        "config": cfg,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn"]},
        "data_sha256": sha256_of(cfg["data_path"]),
        "metrics": {"accuracy": acc, "n_train": len(X_tr), "n_test": len(X_te)},
    }
    out = Path(cfg["out_dir"]) / "run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("manifest written to %s", out.name)


if __name__ == "__main__":
    main()
''', encoding="utf-8")


def run_training(cwd, config_path):
    r = subprocess.run([sys.executable, "-m", "myproj.train", "--config", str(config_path)],
                       cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads((proj / "reports" / "run.json").read_text(encoding="utf-8")), r


m1, r1 = run_training(proj, "configs/base.yaml")
print(r1.stderr.strip())                       # logging's default stream is stderr
print("accuracy:", round(m1["metrics"]["accuracy"], 4), "| data hash:", m1["data_sha256"][:12], "...")

# %% [markdown]
# Run it a second time **from a different working directory** with an absolute config path. Same seed, same data, same versions, so the metrics should match exactly:

# %%
m2, _ = run_training(elsewhere, proj / "configs" / "base.yaml")
print("run 1 metrics:", m1["metrics"])
print("run 2 metrics:", m2["metrics"])
print("identical:", m1["metrics"] == m2["metrics"])
print("packages recorded:", m2["packages"])

# %% [markdown]
# *What just happened?* The run was reproduced from a different folder because paths were anchored to the project root and everything tunable came
# from the config. If a teammate gets a different accuracy, compare their manifest with yours: different `data_sha256` means different data, different `packages` means a
# different environment, different `seed` means a config change. That is a five-minute diagnosis instead of a week of guessing.
#
# Look at what was logged to the file (this is what you would read after a scheduled job):

# %%
print((proj / "reports" / "train.log").read_text().strip().splitlines()[0])
print("log lines written:", len((proj / "reports" / "train.log").read_text().strip().splitlines()))

# %% [markdown]
# ## Common mistakes
#
# 1. **Editing raw data in place** (fixing values in Excel, saving over the CSV). Nothing is reproducible afterwards; write a cleaning step that outputs to `data/processed/`.
# 2. **Hard-coded absolute paths and magic numbers** scattered through notebooks. Put paths and hyperparameters in one config.
# 3. **Seeding only one thing.** Setting `np.random.seed` but not `random_state` in the model or splitter still gives different results.
# 4. **`print` debugging in production jobs**, or `logging.basicConfig` called in several modules. Configure logging once at the entry point.
# 5. **Committing data, models or secrets.** Track code and config in git; keep data in storage or a versioning tool (DVC, object storage); keep secrets in the environment.
# 6. **Believing "same seed" means "same result" across machines.** Library versions, hardware and thread scheduling can change floating-point results slightly; record them and compare with a tolerance.

# %% [markdown]
# ## Interview corner
#
# **1. How do you make a machine-learning experiment reproducible?**
# <details><summary>Answer</summary>
# Fix seeds for every random component, pin the environment (lock file, Python version, ideally a container), version the code (git commit hash), record the data
# version (hash or a data-versioning tool), and store the config and metrics of each run together. Then verify by re-running from a clean checkout.
# </details>
#
# **2. Why put configuration in files rather than in code?**
# <details><summary>Answer</summary>
# It separates what to run from how to run it: you can change experiments without editing code, review changes as small diffs, keep one code path for dev/staging/production,
# and log the exact config used. Secrets are the exception: they come from the environment or a secrets manager, not from committed files.
# </details>
#
# **3. <code>logging</code> versus <code>print</code>?**
# <details><summary>Answer</summary>
# Logging has severity levels, timestamps, named loggers, and configurable destinations (console, files, remote collectors), and can be filtered without editing code.
# Print is unstructured and cannot be turned down or redirected per module.
# </details>
#
# **4. Why do relative paths break, and what is the fix?**
# <details><summary>Answer</summary>
# They are resolved against the process's current working directory, which changes with how the code is launched. Anchor to the file location
# (<code>Path(__file__).resolve()</code>), a configured project root, or an installed package, so behaviour does not depend on the launch directory.
# </details>

# %% [markdown]
# ## Exercises
# `env`, `proj`, `base`, `DATA` and the helpers above are available.

# %% [markdown]
# ### Exercise 1: Fingerprint a data file
# Write `file_sha256(path)` returning the hex SHA-256 digest of a file, reading it in **chunks** (do not read the whole file into memory). *Hint: `hashlib.sha256()`, `.update(bytes)`, `.hexdigest()`.*

# %% tags=["starter"]
def file_sha256(path, chunk_size=1 << 16):
    ...

# %%
truth = hashlib.sha256((DATA / "orders.csv").read_bytes()).hexdigest()
assert file_sha256(DATA / "orders.csv") == truth, "digest differs from hashlib on the whole file"
assert file_sha256(DATA / "orders.csv", chunk_size=7) == truth, "the digest must not depend on chunk size"
copy_path = base / "orders_copy.csv"
copy_path.write_bytes((DATA / "orders.csv").read_bytes() + b"\n")
assert file_sha256(copy_path) != truth, "a one-byte change must change the hash"
print("Exercise 1 passed:", truth[:16], "...")

# %% tags=["solution"]
def file_sha256(path, chunk_size=1 << 16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()

# %% [markdown]
# ### Exercise 2: A reproducible sample
# Write `sample_customers(n, seed)` that returns a **sorted list of `n` distinct `customer_id` values** from `../datasets/customers.csv`, chosen with `np.random.default_rng(seed)`.
# The same seed must give the same list, different seeds should give different lists, and your function must not touch NumPy's global random state.

# %% tags=["starter"]
def sample_customers(n, seed):
    ...

# %%
ids_all = pd.read_csv(DATA / "customers.csv")["customer_id"]
a, b, c = sample_customers(10, 42), sample_customers(10, 42), sample_customers(10, 7)
assert a == b, "same seed must give the same sample"
assert a != c, "different seeds should give different samples"
assert len(a) == 10 and len(set(a)) == 10 and a == sorted(a), "10 distinct ids, sorted"
assert set(a) <= set(ids_all), "ids must come from customers.csv"
state_before = np.random.get_state()[1][:5].copy()
sample_customers(5, 1)
assert (np.random.get_state()[1][:5] == state_before).all(), "do not use or reseed the global np.random state"
print("Exercise 2 passed:", a[:5], "...")

# %% tags=["solution"]
def sample_customers(n, seed):
    ids = pd.read_csv(DATA / "customers.csv")["customer_id"].to_numpy()
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(ids, size=n, replace=False))

# %% [markdown]
# ### Exercise 3: Logging that does not duplicate
# Write `setup_logging(name, logfile)` returning a logger at level `INFO` that writes to **both** the console and `logfile`, with format `"LEVEL | message"`. Calling it twice with the
# same name must **not** add duplicate handlers. *Hint: check `logger.handlers` before adding; set `propagate = False`.*

# %% tags=["starter"]
def setup_logging(name, logfile):
    ...

# %%
lf = base / "ex3.log"
lg = setup_logging("ex3.logger", lf)
lg2 = setup_logging("ex3.logger", lf)          # second call, like re-running a notebook cell
lg2.info("hello")
lg2.debug("hidden at INFO level")
lg2.warning("careful")
for h in lg2.handlers:
    h.flush()
lines = lf.read_text(encoding="utf-8").strip().splitlines()
assert lines == ["INFO | hello", "WARNING | careful"], f"unexpected log file content: {lines}"
assert len(lg2.handlers) == 2, f"expected a console + a file handler, found {len(lg2.handlers)}"
print("Exercise 3 passed:", lines)

# %% tags=["solution"]
def setup_logging(name, logfile):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        fmt = logging.Formatter("%(levelname)s | %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(logfile, encoding="utf-8")):
            handler.setFormatter(fmt)
            log.addHandler(handler)
    return log

# %% [markdown]
# ### Exercise 4: Base config plus experiment override
# Experiments usually override a few keys of `base.yaml`. Write `merge_config(base, override)` that returns a **new** dict where nested dicts are merged recursively and other values
# from `override` win. Do not mutate the inputs. *Hint: recursion on `isinstance(value, dict)`.*

# %% tags=["starter"]
def merge_config(base_cfg, override):
    ...

# %%
b0 = {"seed": 42, "model": {"C": 1.0, "max_iter": 200}, "features": ["a", "b"]}
o0 = {"model": {"C": 0.1}, "features": ["a"], "note": "try smaller C"}
snapshot_b, snapshot_o = json.dumps(b0), json.dumps(o0)
merged = merge_config(b0, o0)
assert merged == {"seed": 42, "model": {"C": 0.1, "max_iter": 200}, "features": ["a"], "note": "try smaller C"}, f"got {merged}"
assert json.dumps(b0) == snapshot_b and json.dumps(o0) == snapshot_o, "inputs must not be mutated"
merged["model"]["C"] = 99
assert b0["model"]["C"] == 1.0, "result must not share nested dicts with the inputs"
print("Exercise 4 passed")

# %% tags=["solution"]
import copy


def merge_config(base_cfg, override):
    out = copy.deepcopy(base_cfg)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

# %% [markdown]
# ### Exercise 5 (open-ended): Design your own layout
# Pick a project you want to build (a forecasting model, a dashboard, an LLM app). In a markdown cell, sketch its folder tree, list the five settings you would put in
# a config file, the seed(s) you would fix, and what you would record in a run manifest. Then run `scaffold()` with your package name in a temp folder and compare.

# %%
for h in list(logging.getLogger("ex3.logger").handlers):
    h.close()                                   # release the log file so the temp folder can be deleted on Windows
tmp.cleanup()
print("temp project removed:", not base.exists())

# %% [markdown]
# ## Recap
# - A predictable layout (raw data read-only, notebooks explore, `src/` computes, config separate) makes projects understandable and testable.
# - Anchor paths to the project root, put settings in YAML, and keep secrets in the environment.
# - Seed every random component with explicit generators; a seed is necessary but not sufficient (hash order, hardware, versions).
# - Configure `logging` once at the entry point; guard against duplicate handlers.
# - Write a manifest (config + versions + data hash + metrics) for every run so differences can be diagnosed.
#
# **Go deeper:** the Cookiecutter Data Science project (cookiecutter-data-science on GitHub), Python docs *Logging HOWTO* and *Logging Cookbook*
# (docs.python.org/3), NumPy's *Random Generator* docs (numpy.org/doc), and the scikit-learn documentation page on *Controlling randomness*.
#
# **Next module:** [Python basics](../01_Python/01_syntax_variables_types.ipynb)
