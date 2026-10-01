# %% [markdown]
# # Data and Model Versioning
#
# **Why this matters.** Six months after launch, a regulator, a customer or your own manager asks: "Why did the model score this person that way in March?" To answer, you need the exact **code, data, parameters and environment** of that training run, and you need to be able to rebuild it.
# Git versions code beautifully and data terribly: a 2 GB CSV bloats the repository, and every edit stores a full new copy. **DVC** (Data Version Control) and similar tools solve this with a simple idea you can build yourself in 40 lines.
# DVC is not installed in this environment, so this notebook does both: it teaches DVC's real files and commands (shown, not run), and it runs a **small working clone** so you see *why* each piece exists.
#
# **You will learn**
# - store files by **content hash** (content-addressed storage) and track them with small pointer files
# - describe a pipeline in **`dvc.yaml` + `params.yaml`** and derive its DAG from dependencies and outputs
# - build a mini `dvc repro`: skip stages whose inputs did not change, re-run only what did
# - restore an **old version** of a model or dataset by its hash, and diff metrics between versions
# - decide what belongs in git, what in the data store, and how a model version ties them together
#
# **Prerequisites:** [Model packaging and registry](03_model_packaging_and_registry.ipynb) and [Project structure and reproducibility](../00_Setup_and_Tools/05_project_structure_and_reproducibility.ipynb).
#
# > Data used: `datasets/churn.csv` copied into a throw-away workspace in a temp folder.

# %%
import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from graphlib import TopologicalSorter
from pathlib import Path

import pandas as pd
import yaml

DATA = Path("../datasets")
TMP = Path(tempfile.mkdtemp(prefix="versioning_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)

# %% [markdown]
# ## 1. The problem in one picture
# A model is a function of four things. Change any one and you may get a different model:
#
# | Ingredient | Versioned by | Breaks reproducibility when... |
# |---|---|---|
# | code | git commit | someone edits the training script |
# | **data** | content hash / dataset version | the source table is refreshed or re-cleaned |
# | parameters | `params.yaml` in git | a learning rate is tweaked "just once" |
# | environment | lock file / image digest | a library upgrades |
#
# Git handles rows 1, 3 and (partly) 4. Row 2 needs a different mechanism.

# %% [markdown]
# ## 2. Content-addressed storage: the core idea
# Name every file by the **hash of its bytes** (SHA-256). Two consequences: identical content is stored **once** no matter how many times or under what name it appears, and a hash **cannot silently change**: if the bytes differ, the name differs. Git itself works this way. (Our mini store uses SHA-256; real DVC has used MD5 by default, so do not expect our hashes to match its files, and check its docs for the current hash setting.)
# The data store keeps the bytes under `store/<first 2 hex>/<rest>`; git keeps only a tiny **pointer file** saying "data/churn.csv is version `ab12...`".

# %%
def file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

class MiniStore:
    """A tiny DVC-like cache: add(file) -> hash; get(hash, dest) restores the bytes."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _obj(self, h):
        return self.root / h[:2] / h[2:]

    def add(self, path):
        h = file_hash(path)
        dest = self._obj(h)
        if not dest.exists():                       # identical content is stored only once
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        return h

    def has(self, h):
        return self._obj(h).exists()

    def get(self, h, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._obj(h), dest)

    def n_objects(self):
        return sum(1 for p in self.root.rglob("*") if p.is_file())

    def size_bytes(self):
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

def write_pointer(path, h):
    """The small file you commit to git instead of the data itself."""
    ptr = Path(str(path) + ".ptr")
    ptr.write_text(json.dumps({"path": Path(path).name, "sha256": h, "size": Path(path).stat().st_size}, indent=1))
    return ptr

store = MiniStore(TMP / "store")
work = TMP / "demo"
work.mkdir()
shutil.copyfile(DATA / "churn.csv", work / "churn.csv")

h1 = store.add(work / "churn.csv")
ptr = write_pointer(work / "churn.csv", h1)
print("hash   :", h1[:16] + "...")
print("pointer:", ptr.read_text().replace("\n", " ")[:120])
print(f"data file {(work / 'churn.csv').stat().st_size / 1024:.0f} KiB vs pointer {ptr.stat().st_size} bytes")

# %% [markdown]
# Now the behaviours that matter. Add the same bytes under another name, then change one value and add again:

# %%
shutil.copyfile(work / "churn.csv", work / "churn_copy.csv")
h_copy = store.add(work / "churn_copy.csv")
print("same bytes, new name   -> same hash:", h_copy == h1, "| objects in store:", store.n_objects())

edited = pd.read_csv(work / "churn.csv", keep_default_na=False)
edited.loc[0, "monthly_charges"] += 1.0                           # a single changed cell
edited.to_csv(work / "churn.csv", index=False)
h2 = store.add(work / "churn.csv")
print("one cell changed       -> new hash:", h2 != h1, "| objects in store:", store.n_objects())

# restore the ORIGINAL version into a fresh location by its hash
store.get(h1, work / "restored_v1.csv")
print("restored v1 equals the original bytes:", file_hash(work / "restored_v1.csv") == h1)

# %% [markdown]
# *What just happened?* Adding a duplicate cost nothing; the one-cell edit created a second object (real DVC stores whole files too, so a large file with a tiny edit is stored twice; some tools such as lakeFS or table formats like Delta Lake version at finer granularity).
# Both versions are retrievable by hash. Commit the pointer to git and you have versioned the data: `git checkout <old commit>` brings back the old pointer, and "pull" fetches the matching bytes.

# %% [markdown]
# ## 3. Hashing a whole directory
# Datasets are often folders of many files. A directory's version is the hash of its **contents**, built from each file's relative path and hash, in sorted order, so that listing order and modification times do not matter. This is a **Merkle-style** tree hash (Exercise 2 asks you to write it). Here is the idea for one level:

# %%
def tree_hash_demo(root):
    entries = sorted((str(p.relative_to(root)).replace(os.sep, "/"), file_hash(p)) for p in Path(root).rglob("*") if p.is_file())
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()

print("demo folder hash:", tree_hash_demo(work)[:16], "...")

# %% [markdown]
# ## 4. Describing a pipeline: `dvc.yaml` and `params.yaml`
# Versioning files is half the story. The other half is the **recipe**: which command produces which output from which inputs and parameters. DVC records it in `dvc.yaml` (committed to git), with tunable values in `params.yaml`.
# This is real DVC syntax. The files below are **valid DVC files** that we will also execute with our mini runner:

# %%
WS = TMP / "ws"
for d in ("data", "scripts", "models"):
    (WS / d).mkdir(parents=True)
shutil.copyfile(DATA / "churn.csv", WS / "data" / "churn.csv")

DVC_YAML = textwrap.dedent("""
    stages:
      prepare:
        cmd: python scripts/prepare.py
        deps: [scripts/prepare.py, data/churn.csv]
        params: [prepare.test_size, prepare.seed]
        outs: [data/train.csv, data/test.csv]
      train:
        cmd: python scripts/train.py
        deps: [scripts/train.py, data/train.csv]
        params: [train.C, train.max_iter, prepare.seed]
        outs: [models/model.joblib]
      evaluate:
        cmd: python scripts/evaluate.py
        deps: [scripts/evaluate.py, models/model.joblib, data/test.csv]
        metrics: [metrics.json]
""").lstrip()
PARAMS = {"prepare": {"test_size": 0.25, "seed": 42}, "train": {"C": 1.0, "max_iter": 1000}}
(WS / "dvc.yaml").write_text(DVC_YAML)
(WS / "params.yaml").write_text(yaml.safe_dump(PARAMS, sort_keys=False))
print(DVC_YAML)

# %% [markdown]
# Nobody declares the order of stages: it follows from the data. `train` lists `data/train.csv` as a dependency, and `prepare` lists it as an output, so `prepare` must run first. We can derive the DAG from the file itself:

# %%
def stage_graph(dvc):
    """stage -> set of stages it depends on, derived by matching each stage's deps to other stages' outs/metrics."""
    producer = {}
    for name, st in dvc["stages"].items():
        for out in st.get("outs", []) + st.get("metrics", []):
            producer[out] = name
    return {name: {producer[d] for d in st.get("deps", []) if d in producer} for name, st in dvc["stages"].items()}

dvc = yaml.safe_load(DVC_YAML)
graph = stage_graph(dvc)
print(graph)
print("order:", list(TopologicalSorter(graph).static_order()))

# %% [markdown]
# The real DVC commands that correspond to everything above (shown, not run; DVC is not installed here):

# %% tags=["no-run"]
# git init && dvc init                      # one-time setup: creates .dvc/ (config + cache location)
# dvc add data/churn.csv                    # hashes the file, moves it to the cache, writes data/churn.csv.dvc (the pointer) and .gitignore
# git add data/churn.csv.dvc data/.gitignore && git commit -m "track raw data"
# dvc remote add -d storage s3://my-bucket/dvc-store      # where the bytes live (S3, GCS, Azure, SSH, a shared folder...)
# dvc push                                  # upload cached objects to the remote
# dvc pull                                  # on another machine / CI: fetch the bytes that the committed pointers name
#
# dvc repro                                 # run the pipeline in dvc.yaml; unchanged stages are skipped; writes dvc.lock (hashes of deps/params/outs)
# dvc params diff                           # which parameters changed since the last commit
# dvc metrics show / dvc metrics diff HEAD~1   # compare metrics.json across commits
# dvc exp run -S train.C=0.1                # run a tracked experiment with a parameter override, without committing
# git checkout v1.2 && dvc checkout         # go back in time: old code + old pointers -> old data restored from cache

# %% [markdown]
# ## 5. A mini `dvc repro`
# A stage must re-run when **anything it depends on changed**: the hash of each dependency file, the value of each declared parameter, or the command itself. Otherwise it is skipped. We record those "signatures" in a lock file after each run (DVC calls it `dvc.lock`).
# First the three stage scripts, which are ordinary Python programs reading `params.yaml`:

# %%
(WS / "scripts" / "prepare.py").write_text(textwrap.dedent('''
    import pandas as pd, yaml
    from sklearn.model_selection import train_test_split
    p = yaml.safe_load(open("params.yaml"))["prepare"]
    df = pd.read_csv("data/churn.csv", keep_default_na=False, na_values=[""])
    df["contract"] = df["contract"].str.strip().str.lower()
    train, test = train_test_split(df, test_size=p["test_size"], stratify=df["churned"], random_state=p["seed"])
    train.to_csv("data/train.csv", index=False); test.to_csv("data/test.csv", index=False)
'''))
(WS / "scripts" / "train.py").write_text(textwrap.dedent('''
    import joblib, pandas as pd, yaml
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    params = yaml.safe_load(open("params.yaml"))
    p = params["train"]
    df = pd.read_csv("data/train.csv", keep_default_na=False, na_values=[""])
    num = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
    cat = ["contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]
    df[cat] = df[cat].astype(str)
    pre = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                             ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(C=p["C"], max_iter=p["max_iter"], random_state=params["prepare"]["seed"]))])
    model.fit(df[num + cat], df["churned"])
    joblib.dump(model, "models/model.joblib")
'''))
(WS / "scripts" / "evaluate.py").write_text(textwrap.dedent('''
    import json, joblib, pandas as pd
    from sklearn.metrics import roc_auc_score
    df = pd.read_csv("data/test.csv", keep_default_na=False, na_values=[""])
    num = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
    cat = ["contract", "internet_service", "paperless_billing", "autopay", "senior_citizen"]
    df[cat] = df[cat].astype(str)
    model = joblib.load("models/model.joblib")
    auc = roc_auc_score(df["churned"], model.predict_proba(df[num + cat])[:, 1])
    json.dump({"auc": round(float(auc), 4), "n_test": int(len(df))}, open("metrics.json", "w"))
'''))
print(sorted(p.relative_to(WS).as_posix() for p in WS.rglob("*") if p.is_file()))

# %%
def get_param(params, dotted):
    cur = params
    for part in dotted.split("."):
        cur = cur[part]
    return cur

class MiniRepro:
    def __init__(self, ws, store):
        self.ws, self.store = Path(ws), store
        self.lock_path = self.ws / "mini.lock.json"

    def _lock(self):
        return json.loads(self.lock_path.read_text()) if self.lock_path.exists() else {}

    def _signature(self, st, params):
        return {"cmd": st["cmd"],
                "deps": {d: file_hash(self.ws / d) for d in st.get("deps", [])},
                "params": {k: get_param(params, k) for k in st.get("params", [])}}

    def run(self):
        dvc = yaml.safe_load((self.ws / "dvc.yaml").read_text())
        params = yaml.safe_load((self.ws / "params.yaml").read_text())
        lock, report = self._lock(), []
        for name in TopologicalSorter(stage_graph(dvc)).static_order():
            st = dvc["stages"][name]
            outs = st.get("outs", []) + st.get("metrics", [])
            sig = self._signature(st, params)
            if lock.get(name, {}).get("sig") == sig and all((self.ws / o).exists() for o in outs):
                report.append((name, "skipped"))
                continue
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.run(st["cmd"].replace("python", f'"{sys.executable}"', 1), shell=True, cwd=self.ws, capture_output=True, text=True, env=env, timeout=300)
            if proc.returncode != 0:
                raise RuntimeError(f"stage {name} failed:\n{proc.stderr[-400:]}")
            lock[name] = {"sig": sig, "outs": {o: self.store.add(self.ws / o) for o in outs}}     # remember outputs by hash, in the cache
            report.append((name, "RAN"))
        self.lock_path.write_text(json.dumps(lock, indent=1))
        return report

repro = MiniRepro(WS, MiniStore(TMP / "cache"))
t0 = time.perf_counter()
print("first run :", repro.run(), f"({time.perf_counter() - t0:.1f}s)")
print("metrics   :", json.loads((WS / "metrics.json").read_text()))

# %% [markdown]
# Everything ran once. The lock file now holds the hash of every input and output. Run again with nothing changed:

# %%
print("second run:", repro.run())

# %% [markdown]
# All three stages were **skipped**: nothing they depend on changed. That is the payoff: cheap, correct incremental rebuilds. Now change one hyper-parameter. Only `train` (which declares `train.C`) and what depends on its output (`evaluate`) should re-run; `prepare` should not:

# %%
lock_v1 = json.loads(repro.lock_path.read_text())
params_v2 = {**PARAMS, "train": {**PARAMS["train"], "C": 0.05}}
(WS / "params.yaml").write_text(yaml.safe_dump(params_v2, sort_keys=False))
print("after changing train.C:", repro.run())
print("metrics:", json.loads((WS / "metrics.json").read_text()))

# %% [markdown]
# *What just happened?* The runner compared each stage's signature with the lock: `prepare` saw the same data hash and the same `prepare.*` params (skip); `train` saw a different `train.C` (run); `evaluate` saw a new model file hash (run).
# Now edit the **data** (drop 200 rows), which should invalidate everything:

# %%
raw = pd.read_csv(WS / "data" / "churn.csv", keep_default_na=False)
raw.iloc[200:].to_csv(WS / "data" / "churn.csv", index=False)
print("after changing the data:", repro.run())
print("metrics:", json.loads((WS / "metrics.json").read_text()))

# %% [markdown]
# ## 6. Time travel: get an old version back, diff metrics
# The lock file from the first run remembers the **hash of the model** we produced back then, and the cache still holds those bytes. So we can restore that exact model, which is what `git checkout` + `dvc checkout` does:

# %%
old_model_hash = lock_v1["train"]["outs"]["models/model.joblib"]
old_metrics_hash = lock_v1["evaluate"]["outs"]["metrics.json"]
now_model_hash = json.loads(repro.lock_path.read_text())["train"]["outs"]["models/model.joblib"]
print("model hash then:", old_model_hash[:12], "| now:", now_model_hash[:12], "| differ:", old_model_hash != now_model_hash)

repro.store.get(old_metrics_hash, TMP / "restored_metrics_v1.json")
repro.store.get(old_model_hash, TMP / "restored_model_v1.joblib")
print("restored v1 metrics:", json.loads((TMP / "restored_metrics_v1.json").read_text()))
print("restored model bytes verified:", file_hash(TMP / "restored_model_v1.joblib") == old_model_hash)

# %% [markdown]
# Comparing two versions is a dictionary diff. This is what `dvc metrics diff` prints:

# %%
def metrics_diff_simple(old, new):
    return {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}

print(metrics_diff_simple(json.loads((TMP / "restored_metrics_v1.json").read_text()), json.loads((WS / "metrics.json").read_text())))

# %% [markdown]
# Notice that `n_test` changed too, because we removed rows: a metric difference between versions can come from the **data** as much as from the model. This is why the data hash belongs in every experiment record (the MLflow tags in notebook 02 and the run manifest in notebook 01).

# %% [markdown]
# ## 7. What goes where, and how a model gets its version
#
# | Artifact | Where | Why |
# |---|---|---|
# | code, `dvc.yaml`, `params.yaml`, `dvc.lock`, `*.dvc` pointers | **git** | small text, reviewable, diffable |
# | raw and processed data, model binaries | **data store / DVC remote** | big, binary, content-addressed |
# | experiment runs and metrics history | tracking server (MLflow) | queryable across runs |
# | released models with aliases | model registry | what production serves |
#
# A **model version** is then a tuple you can write down: *(git commit, dvc.lock data hashes, params, environment lock)*. If you keep all four, "rebuild March's model" is `git checkout`, `dvc pull`, `dvc repro`. In the registry, store the git commit and the data hash as tags on each model version so a serving incident can jump straight to the recipe.
# Honest trade-offs: DVC adds a tool and a remote to maintain; for small projects a dated snapshot of the dataset in object storage plus the hash in your run log is a legitimate, simpler start. Table formats with built-in time travel (Delta Lake, Iceberg) cover the "data as of date X" need inside a warehouse.

# %% [markdown]
# ## Common mistakes
#
# 1. **Committing data to git** (or Git LFS without a plan). The repository grows with every edit and clones become slow. Commit pointers, keep bytes in a store.
# 2. **Pointers without `push`.** A teammate pulls your commit and the bytes are missing because you never uploaded them. Make `dvc push` part of the same habit as `git push`, and check it in CI.
# 3. **Undeclared dependencies.** If a script reads a file or parameter that `dvc.yaml` does not list, changes to it will not trigger a re-run and you get stale results that look fresh. Declare every input.
# 4. **Parameters hard-coded in scripts** instead of `params.yaml`: they are invisible to versioning and to the stage signature.
# 5. **Mutating "the" dataset in place.** Overwriting `latest.csv` destroys the past. Write new versions, reference by hash or dated path.
# 6. **Hashing by path or modification time** instead of content. Copying a file changes its mtime without changing its meaning; content hashes are stable.
# 7. **Versioning the model but not the data** (or vice versa). You can reproduce nothing with half the tuple.
# 8. **Trusting a cache blindly.** A restored file should be verified against its hash (as we did), which is cheap and catches corruption.

# %% [markdown]
# ## Interview corner
#
# **1. Why not just put data in git?**
# <details><summary>Answer</summary>
# Git stores full snapshots of each changed binary/large file, so history balloons, clones and diffs get slow, and hosting limits are hit. Data tools store bytes by content hash in external storage and keep small pointer files in git, so git history still pins exactly which data each commit used.
# </details>
#
# **2. How does DVC know a pipeline stage is up to date?**
# <details><summary>Answer</summary>
# It hashes each stage's command, dependency files and the declared parameter values and compares them with the record in <code>dvc.lock</code>. If they match and outputs exist, the stage is skipped; otherwise it re-runs, and downstream stages whose inputs changed re-run too.
# </details>
#
# **3. You must reproduce a model trained last year. What do you need?**
# <details><summary>Answer</summary>
# The git commit of the code and config, the data version (hashes or dataset snapshot id), the parameters and seeds, and the environment (lock file or image digest). Ideally all are recorded automatically with the experiment and the registered model version so nothing relies on memory.
# </details>
#
# **4. Data versioning vs model registry: how do they relate?**
# <details><summary>Answer</summary>
# Data versioning tracks inputs and pipeline intermediates, the registry tracks released model artifacts and their lifecycle (aliases, stages, approvals). Link them by storing the data hash and code commit as metadata on every registered model version, so lineage runs both ways.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Which stage runs first?
# Write `execution_order(dvc_yaml_text)` that parses a `dvc.yaml` string and returns the list of stage names in a valid execution order, using `stage_graph` and `TopologicalSorter`. It must raise `graphlib.CycleError` if the stages depend on each other in a loop.

# %% tags=["starter"]
def execution_order(dvc_yaml_text):
    ...

# %%
from graphlib import CycleError

order = execution_order(DVC_YAML)
assert order.index("prepare") < order.index("train") < order.index("evaluate"), order
five = """
stages:
  d: {cmd: x, deps: [b, c], outs: [d]}
  b: {cmd: x, deps: [a], outs: [b]}
  c: {cmd: x, deps: [a], outs: [c]}
  a: {cmd: x, deps: [raw], outs: [a]}
"""
o = execution_order(five)
assert o[0] == "a" and o[-1] == "d" and set(o[1:3]) == {"b", "c"}, o
loop = """
stages:
  x: {cmd: c, deps: [y_out], outs: [x_out]}
  y: {cmd: c, deps: [x_out], outs: [y_out]}
"""
try:
    execution_order(loop)
    raise AssertionError("a dependency loop must raise CycleError")
except CycleError:
    pass
print("Exercise 1 passed:", order)

# %% tags=["solution"]
def execution_order(dvc_yaml_text):
    g = stage_graph(yaml.safe_load(dvc_yaml_text))
    return list(TopologicalSorter(g).static_order())

# %% [markdown]
# ### Exercise 2: A directory hash you can trust
# Write `tree_hash(root)` returning a hex SHA-256 string for a **directory**: it must (a) change when any file's **content** changes, (b) change when a file is **renamed or moved**, (c) **not** change when files are merely touched (new modification time) or listed in a different order, and (d) be identical for two different folders holding identical files at the same relative paths.
# *Hint: sort `(relative_posix_path, file_hash)` pairs and hash their JSON, like the demo, but write it yourself from the properties.*

# %% tags=["starter"]
def tree_hash(root):
    ...

# %%
def make_tree(base, files):
    base.mkdir(parents=True)
    for rel, content in files.items():
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text(content)
    return base

t1 = make_tree(TMP / "t1", {"a.txt": "alpha", "sub/b.txt": "beta"})
t2 = make_tree(TMP / "t2", {"sub/b.txt": "beta", "a.txt": "alpha"})          # same content, created in another order
assert tree_hash(t1) == tree_hash(t2), "identical contents at identical relative paths must give the same hash"
h0 = tree_hash(t1)
os.utime(t1 / "a.txt", (1, 1))                                               # touch: only the mtime changes
assert tree_hash(t1) == h0, "modification time must not matter"
(t1 / "a.txt").write_text("alpha!")
assert tree_hash(t1) != h0, "content change must change the hash"
(t1 / "a.txt").write_text("alpha")
assert tree_hash(t1) == h0
(t1 / "a.txt").rename(t1 / "renamed.txt")
assert tree_hash(t1) != h0, "renaming a file must change the hash"
print("Exercise 2 passed:", h0[:12])

# %% tags=["solution"]
def tree_hash(root):
    root = Path(root)
    entries = sorted((p.relative_to(root).as_posix(), file_hash(p)) for p in root.rglob("*") if p.is_file())
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()

# %% [markdown]
# ### Exercise 3: `metrics diff` with deltas
# Write `metrics_diff(old, new)` like `dvc metrics diff`: for every metric key present in **both** dicts whose numeric value changed, return `{key: {"old": o, "new": n, "delta": n - o}}` (delta rounded to 6 decimals). Keys present in only one dict go to a separate list under the special key `"_only_in_one"` (sorted), **only if there are any**. Unchanged metrics are omitted.

# %% tags=["starter"]
def metrics_diff(old, new):
    ...

# %%
d = metrics_diff({"auc": 0.80, "n_test": 1250, "f1": 0.5}, {"auc": 0.82, "n_test": 1250, "brier": 0.2})
assert d["auc"] == {"old": 0.8, "new": 0.82, "delta": 0.02}, d
assert "n_test" not in d, "unchanged metrics are omitted"
assert d["_only_in_one"] == ["brier", "f1"], d
assert metrics_diff({"a": 1}, {"a": 1}) == {}, "no changes -> empty dict (no _only_in_one key)"
real = metrics_diff(json.loads((TMP / "restored_metrics_v1.json").read_text()), json.loads((WS / "metrics.json").read_text()))
assert "n_test" in real and real["n_test"]["delta"] == real["n_test"]["new"] - real["n_test"]["old"]
print("Exercise 3 passed:", real)

# %% tags=["solution"]
def metrics_diff(old, new):
    out = {}
    for k in sorted(set(old) & set(new)):
        if old[k] != new[k]:
            out[k] = {"old": old[k], "new": new[k], "delta": round(new[k] - old[k], 6)}
    only = sorted(set(old) ^ set(new))
    if only:
        out["_only_in_one"] = only
    return out

# %% [markdown]
# ### Exercise 4: Predict, then verify
# Before running, decide which stages will re-run if you change **only `prepare.seed`** (from 42 to 7). Store your prediction in `predicted` (a set of stage names), then change the parameter, run `repro.run()` and store the actual set of stages that ran in `actual`.
# *Read `dvc.yaml` carefully: which stages list `prepare.seed` under `params`, and what depends on their outputs? (Run the solution or your answer only once: a second `repro.run()` finds nothing to do, which is the point of the tool.)*

# %% tags=["starter"]
predicted = ...
actual = ...

# %%
assert predicted == actual == {"prepare", "train", "evaluate"}, f"predicted {predicted}, actual {actual}"
print("Exercise 4 passed: a parameter consumed early cascades through every downstream stage")

# %% tags=["solution"]
predicted = {"prepare", "train", "evaluate"}
(WS / "params.yaml").write_text(yaml.safe_dump({**params_v2, "prepare": {**params_v2["prepare"], "seed": 7}}, sort_keys=False))
actual = {name for name, status in repro.run() if status == "RAN"}

# %% [markdown]
# ### Exercise 5 (open-ended): Draw your project's lineage
# For a model you own, write its `dvc.yaml` (even if you never run it): at least three stages, every input file and parameter declared. Then list one thing in your current workflow that is **not** versioned and would make last month's model impossible to rebuild.

# %% [markdown]
# ## Recap
# - **Content-addressed storage** names bytes by their hash: duplicates cost nothing, a changed byte is a new version, and old versions stay retrievable. Git holds small **pointer files**; a data store holds the bytes.
# - A directory's version is a **tree hash** of relative paths plus file hashes, independent of mtime and listing order.
# - `dvc.yaml` declares stages (cmd, deps, params, outs, metrics); the **DAG follows from matching deps to outs**; `params.yaml` holds tunables; the lock file records the hashes that make a stage reproducible.
# - A stage is **skipped** if its command, dependency hashes and params are unchanged; a change cascades to everything downstream.
# - A model version is *(code commit, data hashes, params, environment)*; record data hash and commit on each registered model. Metric differences between versions can come from the data, not just the model.
#
# **Go deeper:** the DVC documentation (dvc.org/doc): "Get Started", "Data Pipelines" and the `dvc.yaml` reference; the lakeFS and Delta Lake documentation for data version control inside lakes and warehouses; Pro Git (git-scm.com/book) for how git stores objects by hash.
#
# **Next:** [Docker for ML services](06_docker_for_ml_services.ipynb)
