# %% [markdown]
# # Python Environments and Packages
#
# **Why this matters.** "It works on my machine" is the most expensive sentence in data science. A model trained with
# one version of scikit-learn can fail to load, or quietly give different numbers, under another. Isolated environments
# and a pinned dependency list are what make your work runnable by a teammate, a CI server, or you in six months.
#
# **You will learn**
# - what a virtual environment physically is, and how to tell which Python and which packages you are really running
# - install, list, freeze and pin packages with `pip`; the difference between loose and pinned requirements
# - inspect installed packages at run time with `importlib.metadata`, and detect version drift in code
# - build a real throw-away virtual environment, install a local package into it, and see it is isolated from this one
#
# **Prerequisites:** [Terminal and files](01_terminal_and_files.ipynb).
#
# > The notebook runs offline: the one package we install is a tiny one we create ourselves.

# %%
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

print("Python", sys.version.split()[0], "on", platform.system())

# %% [markdown]
# ## 1. Which Python am I running?
#
# Your machine can have many Pythons: the OS one, one from python.org, one from Anaconda, one per project.
# A **virtual environment** is just a folder holding (a link to) a Python interpreter, plus its own `site-packages`
# folder where `pip install` puts libraries. "Activating" it only edits `PATH` so the shell finds that interpreter first.
#
# Two attributes tell you where you are:

# %%
print("sys.executable :", sys.executable)
print("sys.prefix     :", sys.prefix)
print("sys.base_prefix:", sys.base_prefix)
in_venv = sys.prefix != sys.base_prefix
print("running inside a virtual environment:", in_venv)

# %% [markdown]
# `sys.prefix` is where the *current* environment lives; `sys.base_prefix` is the Python it was created from. In a venv they
# differ. This is the first thing to print when something "is installed but cannot be imported": the package went into a
# different Python than the one running your notebook.
#
# Where do imports come from? `sys.path` is searched in order:

# %%
for p in sys.path[:6]:
    print(p)

# %% [markdown]
# ### The kernel trap
# A Jupyter notebook runs in a **kernel**, which is a Python process tied to one interpreter. Running `pip install x` in a
# terminal whose venv is not the kernel's will not make `import x` work in the notebook. Inside a notebook, install with
# the interpreter you are actually running:
#
# ```python
# %pip install some-package          # notebook magic: uses the kernel's own pip (preferred)
# !{sys.executable} -m pip install some-package
# ```
#
# and always prefer `python -m pip ...` to bare `pip ...`, since it guarantees the pip belongs to that `python`.

# %% [markdown]
# ## 2. What is installed, and which version?
# `importlib.metadata` reads the metadata pip left behind. Package (distribution) names and import names can differ:
# you `pip install scikit-learn` but `import sklearn`.

# %%
for name in ["numpy", "pandas", "scikit-learn", "matplotlib"]:
    print(f"{name:14s} {md.version(name)}")

dist_by_module = md.packages_distributions()
print("import sklearn is provided by:", dist_by_module["sklearn"])
print("import yaml is provided by   :", dist_by_module["yaml"])

# %%
import numpy as np
import pandas as pd

print(np.__version__, pd.__version__)
assert np.__version__ == md.version("numpy")   # two sources, same answer

n = len(list(md.distributions()))
print(f"{n} distributions installed in this environment")

# %% [markdown]
# *What just happened?* The library reports its own version (`np.__version__`) and pip's metadata records the same thing. When they
# disagree you have a corrupted or mixed install, a reason to recreate the environment.

# %% [markdown]
# ### Using pip from Python (and the CLI equivalents)
# | Task | Command |
# |---|---|
# | install | `python -m pip install pandas` / `pandas==3.0.0` / `"pandas>=2,<4"` |
# | list | `python -m pip list` |
# | details | `python -m pip show numpy` |
# | snapshot exact versions | `python -m pip freeze > requirements.txt` |
# | install from a snapshot | `python -m pip install -r requirements.txt` |
# | look for conflicts | `python -m pip check` |

# %%
def pip(*args, python=sys.executable):
    r = subprocess.run([python, "-m", "pip", *args, "--disable-pip-version-check"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r


r = pip("show", "numpy")
print("\n".join(r.stdout.splitlines()[:3] + [l for l in r.stdout.splitlines() if l.startswith("Requires")]))
r = pip("check")
print("pip check:", r.stdout.strip() or r.stderr.strip())

# %% [markdown]
# `pip check` verifies that installed packages' declared requirements are satisfied. Run it after installing things; a
# "has requirement X but you have Y" message is an early warning.

# %% [markdown]
# ## 3. Loose versus pinned requirements
#
# ```text
# # loose (intent)               # pinned (reproducible)
# pandas                         pandas==3.0.0
# scikit-learn>=1.5              scikit-learn==1.9.0
# ```
#
# - **Loose** says what you need; the resolver picks whatever is newest today. Good for libraries and for a human-edited
#   "top-level" list.
# - **Pinned** (`==`) freezes every package, including the dependencies of your dependencies. Good for applications, analyses
#   and deployments, where the goal is *the same result next month*.
#
# A common pattern: keep a short hand-written `requirements.in` of top-level packages and a generated, fully pinned
# `requirements.txt` (tools such as `pip-tools`, `uv` or `poetry` automate this). `pip freeze` is the simplest form.
# Pinning does not make you immune: the Python version, the OS, and compiled libraries (BLAS, CUDA) still matter, which is why
# containers exist (see the Docker notebooks later).

# %% [markdown]
# Let's write code that does what `pip freeze` would tell us, and then checks a requirements file against reality.

# %%
def freeze(packages):
    """'name==version' lines for the given distributions."""
    return [f"{p}=={md.version(p)}" for p in packages]


print("\n".join(freeze(["numpy", "pandas", "scipy"])))

# %%
def parse_requirements(text):
    """Parse 'name==version' lines into {name: version}. Ignores blank lines, comments and non-pinned lines."""
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


sample = """
# core
numpy==2.0.0   # deliberately old
pandas>=2      # loose, ignored by this parser
scipy==1.0.0
"""
print(parse_requirements(sample))

# %% [markdown]
# Now a drift check: which pinned packages are missing or at a different version than installed? This is the core of what a
# CI job or a startup check can do to fail fast instead of producing subtly different results.

# %%
def find_drift(pins):
    """Return {name: (pinned, installed_or_None)} for every mismatch."""
    drift = {}
    for name, wanted in pins.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        if have != wanted:
            drift[name] = (wanted, have)
    return drift


demo_pins = {"numpy": md.version("numpy"), "pandas": "0.0.1", "not-a-real-package": "1.0"}
print(find_drift(demo_pins))

# %% [markdown]
# ## 4. Build a real virtual environment
# `python -m venv` is in the standard library. We create one in a temp folder (takes a few seconds because it bootstraps pip),
# look inside, and install a package we create locally, so there is no network involved.

# %%
tmp = tempfile.TemporaryDirectory()
base = Path(tmp.name)
env_dir = base / "demo_env"
venv.create(env_dir, with_pip=True)

bin_dir = env_dir / ("Scripts" if platform.system() == "Windows" else "bin")
venv_python = bin_dir / ("python.exe" if platform.system() == "Windows" else "python")
print("top level:", sorted(p.name for p in env_dir.iterdir()))
print("interpreter exists:", venv_python.exists())
print((env_dir / "pyvenv.cfg").read_text())

# %% [markdown]
# `pyvenv.cfg` is the whole trick: it records which base Python the environment came from. Python reads that file at startup
# and adjusts `sys.prefix` and `sys.path`. Now compare the two environments:

# %%
print("this kernel :", len(list(md.distributions())), "packages")
r = pip("list", python=str(venv_python))
print("fresh venv  :")
print(r.stdout)

# %% [markdown]
# The fresh venv contains only pip (and setuptools on some Python versions). None of the libraries installed for this notebook are
# visible there: that is *isolation*, and it is why you make one environment per project.
#
# Now create a tiny package, install it into the new environment, and prove it is importable there but not here.

# %%
pkg = base / "mini_stats_src"
(pkg / "mini_stats").mkdir(parents=True)
(pkg / "pyproject.toml").write_text(
    '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n\n'
    '[project]\nname = "mini-stats"\nversion = "0.1.0"\n\n[tool.setuptools]\npackages = ["mini_stats"]\n', encoding="utf-8")
(pkg / "mini_stats" / "__init__.py").write_text("def mean(x):\n    return sum(x) / len(x)\n", encoding="utf-8")

r = pip("install", "--no-index", "--no-build-isolation", str(pkg), python=str(venv_python))
print("install exit code:", r.returncode, "|", r.stdout.strip().splitlines()[-1])

# %%
inside = subprocess.run([str(venv_python), "-c", "import mini_stats, sys; print(mini_stats.mean([1, 2, 3]), '| prefix:', sys.prefix)"],
                        capture_output=True, text=True)
print("in the new venv:", inside.stdout.strip())
try:
    import mini_stats  # noqa: F401
except ModuleNotFoundError as e:
    print("in this kernel :", e)

# %% [markdown]
# *What just happened?* `--no-index` told pip not to contact PyPI, and `--no-build-isolation` reused the setuptools already in the
# venv instead of downloading a build environment. Normally you drop both flags. The package now lives in the venv's own
# `site-packages`, invisible to any other Python.
#
# ### Activation, in both shells
# | | bash / zsh | PowerShell | cmd |
# |---|---|---|---|
# | create | `python -m venv .venv` | `python -m venv .venv` | `python -m venv .venv` |
# | activate | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` | `.venv\Scripts\activate.bat` |
# | leave | `deactivate` | `deactivate` | `deactivate` |
#
# You do not need to activate at all: calling `.venv/bin/python` (or `.venv\Scripts\python.exe`) directly uses that
# environment, which is exactly what we did through `subprocess`. In PowerShell, scripts may be blocked by the execution policy
# (`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` is the usual fix, but that is a security setting: understand it before
# changing it).
#
# To make a venv selectable as a Jupyter kernel: `python -m ipykernel install --user --name my-project`.
# Other tools solve the same problem differently: **conda** manages Python itself and non-Python libraries (useful for
# some GPU/scientific stacks), **uv** and **poetry** add fast resolving/lock files. The mental model above applies to all.

# %% [markdown]
# ## Common mistakes
#
# 1. **Installing into the wrong Python.** `pip` on PATH is not necessarily the pip of your kernel. Use `python -m pip`, and check
#    `sys.executable`.
# 2. **Committing the virtual environment** to git. It is large, OS-specific and reproducible from `requirements.txt`. Put `.venv/` in `.gitignore`.
# 3. **Unpinned `pip install` in a Dockerfile or CI** and being surprised when tomorrow's build breaks. Pin, and update deliberately.
# 4. **One global environment for everything.** Project A needs pandas 1.x, project B needs 3.x; a shared environment cannot satisfy both.
# 5. **Moving a venv folder.** Virtual environments contain absolute paths; copying one to another location or machine breaks it. Recreate from the requirements file.

# %%
# Mistake 5, visible: the venv records an absolute path to the base Python, and its launcher scripts embed absolute paths too
cfg = dict(line.split(" = ", 1) for line in (env_dir / "pyvenv.cfg").read_text().splitlines() if " = " in line)
print("home (absolute path to the base Python):", cfg["home"])
print("absolute path in cfg?", Path(cfg["home"]).is_absolute())

# %% [markdown]
# ## Interview corner
#
# **1. What is a virtual environment and why use one?**
# <details><summary>Answer</summary>
# An isolated directory with its own interpreter link and <code>site-packages</code>, so each project has its own dependency versions. It prevents version
# conflicts between projects and keeps the system Python clean. It is cheap to create and delete.
# </details>
#
# **2. How do you make a data-science project reproducible across machines?**
# <details><summary>Answer</summary>
# Pin dependency versions (a lock or frozen requirements file), record the Python version, fix random seeds, keep data versioned or its source documented, and
# ideally run in a container or CI to confirm a clean machine can rebuild and run it. Pinning alone is not enough when compiled libraries or the OS differ.
# </details>
#
# **3. <code>pip install</code> succeeded but <code>import</code> fails in Jupyter. What do you check?**
# <details><summary>Answer</summary>
# Whether the notebook kernel's interpreter (<code>sys.executable</code>) is the same one pip installed into. Fix with <code>%pip install</code> or
# <code>python -m pip</code> using that interpreter, then restart the kernel.
# </details>
#
# **4. <code>requirements.txt</code> versus <code>pyproject.toml</code>?**
# <details><summary>Answer</summary>
# <code>requirements.txt</code> is a flat list of installable pins, typically for applications and deployment. <code>pyproject.toml</code> is the standard
# project metadata file (name, version, dependency ranges, build system, tool settings) and is what makes a folder an installable package.
# </details>

# %% [markdown]
# ## Exercises
# The helpers `pip`, `freeze`, `parse_requirements`, `find_drift` and the temp venv from above are available.

# %% [markdown]
# ### Exercise 1: Am I in a virtual environment?
# Store `in_venv_ex` (bool) using `sys.prefix` and `sys.base_prefix`, and `kernel_python_name` = the file name of the running interpreter
# (for example `python.exe` on Windows). *Hint: `Path(sys.executable).name`.*

# %% tags=["starter"]
in_venv_ex = ...
kernel_python_name = ...

# %%
assert in_venv_ex == (sys.prefix != sys.base_prefix), "compare sys.prefix with sys.base_prefix"
assert kernel_python_name == os.path.basename(sys.executable)
print("Exercise 1 passed:", in_venv_ex, kernel_python_name)

# %% tags=["solution"]
in_venv_ex = sys.prefix != sys.base_prefix
kernel_python_name = Path(sys.executable).name

# %% [markdown]
# ### Exercise 2: A version table
# Build `versions`: a dict mapping each of `"numpy"`, `"pandas"`, `"scipy"`, `"scikit-learn"`, `"matplotlib"` to its installed version string, using `importlib.metadata`.

# %% tags=["starter"]
versions = ...

# %%
import scipy, sklearn, matplotlib  # independent source of truth: the modules' own __version__
truth = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
         "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
assert versions == truth, f"got {versions}, expected {truth}"
print("Exercise 2 passed:", versions)

# %% tags=["solution"]
versions = {name: md.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]}

# %% [markdown]
# ### Exercise 3: Extend the parser
# Real requirements files also contain **extras and comparisons**. Write `parse_pin(line)` that returns `(name, version)` for a line like
# `"scikit-learn==1.9.0  # model code"` or `"uvicorn[standard]==0.30.0"` (strip the `[extras]` from the name, lower-case it) and returns `None`
# for blank lines, comment-only lines, and lines that are not exact `==` pins (like `pandas>=2`).

# %% tags=["starter"]
def parse_pin(line):
    ...

# %%
assert parse_pin("scikit-learn==1.9.0  # model code") == ("scikit-learn", "1.9.0")
assert parse_pin("uvicorn[standard]==0.30.0") == ("uvicorn", "0.30.0")
assert parse_pin("  NumPy == 2.1.0 ") == ("numpy", "2.1.0")
assert parse_pin("") is None and parse_pin("# only a comment") is None
assert parse_pin("pandas>=2") is None
print("Exercise 3 passed")

# %% tags=["solution"]
def parse_pin(line):
    line = line.split("#", 1)[0].strip()
    if "==" not in line:
        return None
    name, version = (part.strip() for part in line.split("==", 1))
    name = name.split("[", 1)[0].strip().lower()
    return (name, version) if name and version else None

# %% [markdown]
# ### Exercise 4: Fail fast on drift
# Write `assert_environment(pins)`: it should raise `RuntimeError` whose message lists the drifted package names when `find_drift(pins)` is
# non-empty, and return `True` otherwise. (This is the kind of guard you put at the top of a training script.)

# %% tags=["starter"]
def assert_environment(pins):
    ...

# %%
ok = {"numpy": md.version("numpy"), "pandas": md.version("pandas")}
assert assert_environment(ok) is True
try:
    assert_environment({"numpy": md.version("numpy"), "pandas": "0.0.1", "ghost-pkg": "1"})
    raise AssertionError("should have raised RuntimeError")
except RuntimeError as e:
    assert "pandas" in str(e) and "ghost-pkg" in str(e) and "numpy" not in str(e), f"message should list only drifted packages: {e}"
print("Exercise 4 passed")

# %% tags=["solution"]
def assert_environment(pins):
    drift = find_drift(pins)
    if drift:
        raise RuntimeError("environment drift: " + ", ".join(f"{n} (pinned {w}, installed {h})" for n, (w, h) in drift.items()))
    return True

# %% [markdown]
# ### Exercise 5 (open-ended): Write the requirements for a project
# Pick a small project you would like to do (for example the churn model from the ML module). In a markdown cell, list the top-level packages it
# needs as a loose `requirements.in`, then use `freeze([...])` to generate the pinned lines for the ones installed here. Note one package whose
# import name differs from its distribution name.

# %%
tmp.cleanup()
print("temp venv removed:", not base.exists())

# %% [markdown]
# ## Recap
# - A venv is a folder with an interpreter link and its own `site-packages`; `sys.prefix != sys.base_prefix` tells you if you are in one.
# - Use `python -m pip`, and make sure it is the interpreter your notebook kernel runs (`sys.executable`).
# - Loose requirements express intent; pinned ones reproduce a result. `pip freeze`, `pip check` and `importlib.metadata` help you verify.
# - One environment per project; never commit the environment, commit the requirements.
# - Drift checks in code turn silent version differences into loud errors.
#
# **Go deeper:** the Python Packaging User Guide (packaging.python.org), the `venv` and `importlib.metadata` docs (docs.python.org/3),
# and the pip documentation (pip.pypa.io).
#
# **Next:** [Jupyter and VS Code workflow](04_jupyter_and_vscode_workflow.ipynb)
