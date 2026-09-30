# %% [markdown]
# # Jupyter and VS Code Workflow
#
# **Why this matters.** Notebooks are the fastest way to explore data, and also the fastest way to produce results nobody
# (including you) can reproduce. The difference is not the tool, it is a few habits: run top to bottom, keep the notebook
# small, and never commit stale outputs. Reviewers and hiring managers can tell within a minute whether a notebook was
# written with those habits.
#
# **You will learn**
# - the notebook model: a kernel holding state, cells you can run in any order
# - magics (`%timeit`, `%pwd`, `%env`, `!cmd`) and the keyboard shortcuts that make you fast
# - how **hidden state** breaks notebooks, demonstrated by running real kernels, and the "Restart and Run All" discipline
# - how to keep notebooks clean in git (strip outputs, review in pair with plain-text formats)
# - the VS Code equivalents (`# %%` cells, Interactive Window, debugger, interpreter selection)
#
# **Prerequisites:** [Environments and packages](03_environments_and_packages.ipynb) and [Git and GitHub](02_git_and_github.ipynb).

# %%
import json
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# %% [markdown]
# ## 1. The model: a document plus a kernel
#
# A notebook (`.ipynb`) is a **JSON document** of cells. It does not run anything itself. A separate **kernel** process (one Python
# interpreter) executes the cells you send it, one at a time, **in whatever order you click**, and keeps every variable alive between
# them. The order of execution is stored as the small number in `[ ]` beside each cell.
#
# Let's look at the document side: a notebook is just JSON you can read with a few lines of code.

# %%
nb = new_notebook(cells=[new_markdown_cell("# Title"), new_code_cell("x = 1 + 1\nprint(x)")])
print("keys:", list(nb.keys()))
print("cell types:", [c.cell_type for c in nb.cells])
print(json.dumps(nb.cells[1], indent=1)[:260])

# %% [markdown]
# *What just happened?* A code cell is a dict with `source`, `outputs` and `execution_count`. Outputs and counters live **inside the file**,
# which is why notebooks diff badly in git (see section 5).

# %% [markdown]
# ## 2. Magics, shell escapes, and shortcuts
#
# **Magics** are IPython commands starting with `%` (one line) or `%%` (whole cell). `!cmd` runs a shell command.
# They only work in IPython/Jupyter, not in plain Python files, so keep them out of code you plan to import elsewhere.

# %%
# %pwd  -> the kernel's working directory (same idea as os.getcwd)
# %who  -> names defined in this session
# %timeit -> statistical timing of one statement
%pwd

# %%
%timeit -n 200 -r 3 sum(i * i for i in range(1_000))

# %%
%%time
total = sum(i * i for i in range(200_000))

# %% [markdown]
# `%timeit` runs the statement many times and reports mean ± standard deviation (use it for micro-benchmarks); `%%time` times a whole
# cell once (good for "how long does this load take?"). Timings vary from run to run: never hard-code them into a conclusion.
#
# More magics worth knowing:
#
# | Magic | Use |
# |---|---|
# | `%run script.py` | run a script inside the session |
# | `%load_ext autoreload` then `%autoreload 2` | re-import edited modules automatically |
# | `%matplotlib inline` | show plots inside the notebook (usually the default) |
# | `%env NAME=value` | set/show environment variables for this kernel |
# | `%who`, `%whos` | list variables (whos adds types and sizes) |
# | `%%writefile path` | write the cell body to a file (handy for creating a module from a notebook) |
# | `?obj`, `??obj` | show the docstring / source of `obj` |
# | `!ls`, `!dir` | shell command (OS-specific!) |
#
# The pip magic from the previous notebook is `%pip install ...`.

# %%
%whos int float

# %% [markdown]
# ### Shortcuts that pay off (JupyterLab; they can differ slightly between versions and VS Code)
#
# | Keys | Does |
# |---|---|
# | `Shift+Enter` / `Ctrl+Enter` | run cell and move on / run and stay |
# | `Esc` then `A` / `B` | insert cell above / below (command mode) |
# | `Esc` then `D D` | delete cell |
# | `Esc` then `M` / `Y` | convert cell to markdown / code |
# | `Shift+Tab` (inside a call) | show the docstring: fastest way to learn a function's arguments |
# | `Tab` | autocomplete |
# | `Ctrl+/` | comment/uncomment lines |
# | `0 0` (command mode) | restart the kernel |
# | `Ctrl+Shift+C` | command palette: search every action by name |

# %% [markdown]
# ## 3. Hidden state: why notebooks lie
#
# The kernel remembers everything you ever ran, including code you have since edited or deleted. The saved file only contains what is
# *currently* in the cells. When those two disagree, the notebook works for you and fails for everyone else.
#
# To show that honestly we need real kernels. This helper runs a list of code cells **in order, in one fresh kernel** and reports what
# each printed (or which exception it raised). It is just `nbclient`, the same library our checker uses.

# %%
def run_kernel(sources):
    """Execute code cells sequentially in a fresh kernel. Returns [(stdout, error_name_or_None), ...]."""
    nb = new_notebook(cells=[new_code_cell(s) for s in sources])
    NotebookClient(nb, timeout=60, kernel_name="python3", allow_errors=True).execute()
    results = []
    for c in nb.cells:
        out = "".join(o.get("text", "") for o in c.outputs if o.output_type == "stream" and o.name == "stdout")
        err = next((o.ename for o in c.outputs if o.output_type == "error"), None)
        results.append((out.strip(), err))
    return results


print(run_kernel(["a = 20", "b = 22", "print(a + b)"]))

# %% [markdown]
# ### Bug 1: the deleted cell
# While exploring you ran `total = price * 3` in a cell, then deleted that cell because it "looked messy". The variable stays alive in the kernel.
# The **history** (what you did) works; the **saved notebook** (what is left) does not.

# %%
history = ["price = 100", "total = price * 3", "print(total)"]            # what your kernel executed
saved_notebook = ["price = 100", "print(total)"]                          # what the file contains after deleting cell 2

print("your session      :", run_kernel(history)[-1])
print("Restart & Run All :", run_kernel(saved_notebook)[-1])

# %% [markdown]
# ### Bug 2: out-of-order execution
# Re-running a cell that changes its own input (`x = x + 1`) gives a different answer each time. The saved notebook *looks* like it prints 2:

# %%
cells = ["x = 1", "x = x + 1", "print(x)"]
clicked = ["x = 1", "x = x + 1", "x = x + 1", "print(x)"]   # you ran the middle cell twice while experimenting
print("what the file says :", run_kernel(cells)[-1][0])
print("what you saw       :", run_kernel(clicked)[-1][0])

# %% [markdown]
# ### Bug 3: stale imports
# You edit a helper module, but `import mymodule` was already executed, so the kernel keeps using the old code.
# Cells here (a) import a module we write into a temp folder, (b) edit the file, (c) call the function again.

# %%
work = tempfile.TemporaryDirectory()
mod_dir = Path(work.name)
setup = f"""
import sys, pathlib
sys.path.insert(0, {str(mod_dir)!r})
pathlib.Path({str(mod_dir / 'helpers.py')!r}).write_text('def rate():\\n    return 0.10\\n')
import helpers
print('first  :', helpers.rate())
"""
edit = f"pathlib.Path({str(mod_dir / 'helpers.py')!r}).write_text('def rate():\\n    return 0.25  # edited\\n')"
without_reload = run_kernel([setup, edit, "import helpers\nprint('after edit, plain import :', helpers.rate())"])
with_reload = run_kernel([setup, edit, "import importlib, helpers\nimportlib.reload(helpers)\nprint('after edit, reload      :', helpers.rate())"])
print(without_reload[0][0]); print(without_reload[2][0]); print(with_reload[2][0])

# %% [markdown]
# `import` a second time is a no-op because modules are cached in `sys.modules`. `importlib.reload` (or `%autoreload 2`) refreshes it; a
# **kernel restart** is the reliable fix. `%autoreload` has limits (it does not update already-created objects or `from x import y` copies in
# every case), so treat it as a convenience, not a guarantee.

# %% [markdown]
# ### The discipline
# 1. Before you share, commit or believe a result: **Kernel -> Restart Kernel and Run All Cells**. If it errors, the notebook is broken; fix it.
# 2. Define imports and constants at the top; put functions in cells you can re-run safely (no cell should depend on being run twice).
# 3. Do not delete cells that define state without checking what depends on them; use "Run All Above" to test.
# 4. When a notebook grows past a few hundred lines, move stable code into a `.py` module (next notebook) and import it.
# 5. Treat a notebook as a **report + exploration log**, not as your only copy of the logic.

# %% [markdown]
# ## 4. VS Code equivalents
#
# You can work in classic `.ipynb` files in VS Code (Jupyter extension) or in plain **`.py` files with `# %%` cell markers**. This roadmap is
# itself authored that way: every notebook here is generated from a percent-format source file.
#
# | Need | JupyterLab | VS Code |
# |---|---|---|
# | choose the environment | Kernel -> Change Kernel | "Select Kernel" (top right) / *Python: Select Interpreter* |
# | run a cell | `Shift+Enter` | `Shift+Enter` (in `.py`, cells start with `# %%`) |
# | see variables | `%whos` | Variables view / Interactive Window |
# | debug | limited | full debugger with breakpoints (works in `.py` and `.ipynb`) |
# | refactor / lint | basic | rename symbol, linting, type checking |
#
# Why `.py` with `# %%` is popular: git diffs are readable line by line, there are no outputs to strip, the same file can be run as a script,
# and code review works. Tools such as `jupytext` keep an `.ipynb` and a `.py` in sync if a team needs both.

# %% [markdown]
# A percent-format file looks like this (shown as text; it needs the VS Code GUI to run interactively):
#
# ```python
# # %% [markdown]
# # # Analysis
# # %%
# import pandas as pd
# df = pd.read_csv("../datasets/churn.csv")
# df.head()
# ```

# %% [markdown]
# ## 5. Keeping notebooks clean in git
#
# A saved notebook stores every output (tables, images as base64, execution counters). Effects:
# - repos bloat, and every re-run changes the file even if the code did not
# - merge conflicts appear in JSON blobs nobody can read
# - outputs may leak data or secrets that were printed
#
# The standard remedy: **strip outputs before committing**. Let's see the size effect with a notebook that produces a plot-sized output,
# then write the stripper.

# %%
run_nb = new_notebook(cells=[
    new_code_cell("import random\nrandom.seed(0)\nrows = [random.random() for _ in range(2000)]\nprint(len(rows))\nrows[:400]"),
    new_code_cell("print(sum(rows) / len(rows) > 0)"),
])
NotebookClient(run_nb, timeout=60, kernel_name="python3").execute()
executed_json = nbformat.writes(run_nb)
print("execution counts:", [c.execution_count for c in run_nb.cells])
print("size with outputs:", len(executed_json), "characters")

# %%
def strip_outputs_demo(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
    return nb


import copy
stripped = strip_outputs_demo(copy.deepcopy(run_nb))
print("size stripped    :", len(nbformat.writes(stripped)), "characters")

# %% [markdown]
# In practice you do not do this by hand: install a **git filter** such as `nbstripout` (`pip install nbstripout`, then `nbstripout --install` inside
# the repo) so stripping happens automatically on every `git add`, or a `pre-commit` hook. Some teams *want* outputs committed (published
# reports, tutorials): then commit them deliberately and make sure the notebook was run top to bottom first.

# %% [markdown]
# ## Common mistakes
#
# 1. **Sharing a notebook you never restarted and re-ran.** It may depend on cells you deleted or on a different execution order.
# 2. **Magic commands in reusable code.** `%timeit` in a `.py` module is a syntax error; use the `timeit` module there.
# 3. **Absolute paths** (`C:\Users\me\data.csv`) that only exist on your machine; use paths relative to the project (see the next notebook).
# 4. **Giant outputs** (printing a 1M-row DataFrame). Use `.head()`, `.sample()`, summaries; it also keeps the file small.
# 5. **Secrets in notebooks.** An API key that appears in a cell or output is in the file and in git history.

# %% [markdown]
# ## Interview corner
#
# **1. Why do notebooks cause reproducibility problems and how do you mitigate them?**
# <details><summary>Answer</summary>
# Kernel state persists across edits and arbitrary execution order, so results can depend on cells that no longer exist. Mitigate with
# Restart-and-Run-All before sharing, moving logic into tested modules, pinning the environment, and stripping outputs / reviewing diffs in text form.
# </details>
#
# **2. Notebook or script: how do you choose?**
# <details><summary>Answer</summary>
# Notebooks for exploration, visualisation and communicating results; scripts/modules for anything repeated, scheduled, tested or deployed. A common path is
# to explore in a notebook, then extract the stable functions into a package that the notebook imports.
# </details>
#
# **3. What is a Jupyter kernel?**
# <details><summary>Answer</summary>
# A separate process that executes code and holds variables; the notebook UI sends cells to it over a messaging protocol. Different kernels can run different
# Python environments or even other languages.
# </details>
#
# **4. How would you code-review a notebook?**
# <details><summary>Answer</summary>
# Ask for outputs stripped or a paired plain-text version so the diff is readable, check it runs top to bottom in a clean environment, and focus the review on
# logic in extracted modules and tests rather than on JSON.
# </details>

# %% [markdown]
# ## Exercises
# The helpers `run_kernel` and the imports above are available. `run_kernel` starts a real kernel, so each call takes a few seconds.

# %% [markdown]
# ### Exercise 1: Fix a notebook that only works in your session
# The list `broken` is a saved notebook. Run in order, it fails. Rewrite it as `fixed` (a list of cell sources) so that **Restart and Run All** prints exactly `42`
# and no cell raises an error. *Hint: look at what each cell needs from the others, and in what order they must appear.*

# %%
broken = [
    "answer = base * 2",
    "print(answer)",
    "base = 21",
]
print(run_kernel(broken))

# %% tags=["starter"]
fixed = ...

# %%
res = run_kernel(fixed)
assert all(err is None for _, err in res), f"a cell still fails: {res}"
assert res[-1][0] == "42" or any(out == "42" for out, _ in res), f"expected the notebook to print 42, got {res}"
print("Exercise 1 passed")

# %% tags=["solution"]
fixed = [
    "base = 21",
    "answer = base * 2",
    "print(answer)",
]

# %% [markdown]
# ### Exercise 2: A real output stripper
# Write `strip_outputs(nb)` for an `nbformat` notebook object: for every **code** cell set `outputs` to `[]` and `execution_count` to `None`. Do not change any cell's `source`,
# and leave markdown cells alone. Return the notebook.

# %% tags=["starter"]
def strip_outputs(nb):
    ...

# %%
nb2 = copy.deepcopy(run_nb)
nb2.cells.insert(0, new_markdown_cell("# notes"))
sources_before = [c.source for c in nb2.cells]
out_nb = strip_outputs(nb2)
assert [c.source for c in out_nb.cells] == sources_before, "sources must stay unchanged"
assert all(c.outputs == [] and c.execution_count is None for c in out_nb.cells if c.cell_type == "code"), "outputs/counts must be cleared"
assert out_nb.cells[0].cell_type == "markdown"
assert len(nbformat.writes(out_nb)) < len(executed_json)
print("Exercise 2 passed")

# %% tags=["solution"]
def strip_outputs(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
    return nb

# %% [markdown]
# ### Exercise 3: A pre-commit style check
# Write `is_clean(path)` that returns `True` if the `.ipynb` file at `path` has **no outputs and no execution counts** in any code cell, else `False`. Read the file with `json`
# (not nbformat) to practice the raw structure. *Hint: `json.load(...)["cells"]`, keys `"cell_type"`, `"outputs"`, `"execution_count"`.*

# %% tags=["starter"]
def is_clean(path):
    ...

# %%
p_dirty = Path(work.name) / "dirty.ipynb"
p_clean = Path(work.name) / "clean.ipynb"
nbformat.write(run_nb, p_dirty)
nbformat.write(strip_outputs(copy.deepcopy(run_nb)), p_clean)
assert is_clean(p_dirty) is False, "the executed notebook has outputs"
assert is_clean(p_clean) is True, "the stripped notebook should count as clean"
print("Exercise 3 passed")

# %% tags=["solution"]
def is_clean(path):
    with open(path, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    return all(not c.get("outputs") and c.get("execution_count") is None for c in cells if c["cell_type"] == "code")

# %% [markdown]
# ### Exercise 4: Predict, then verify
# The cell list `mystery` is run in a fresh kernel. Without running it, decide what the **last cell** prints, and store it as a string in `prediction`. The check runs the kernel and compares.
# *Hint: the name `data` is rebound; think about which `data` the function sees when it is called, not when it was defined.*

# %%
mystery = [
    "data = [1, 2, 3]",
    "def total():\n    return sum(data)",
    "data = [10, 20]",
    "print(total())",
]

# %% tags=["starter"]
prediction = ...

# %%
actual = run_kernel(mystery)[-1][0]
assert prediction == actual, f"the kernel printed {actual!r}, you predicted {prediction!r}"
print("Exercise 4 passed: functions look names up when called, not when defined")

# %% tags=["solution"]
prediction = "30"

# %% [markdown]
# ### Exercise 5 (open-ended): Your notebook checklist
# Write a 6-line "before I share a notebook" checklist in a markdown cell. Then take one of *your own* older notebooks (or any file from this roadmap), run Restart and Run All, and
# note what broke or surprised you.

# %%
work.cleanup()
print("temp files removed:", not Path(work.name).exists())

# %% [markdown]
# ## Recap
# - A notebook is a JSON document plus a kernel that keeps state; the file only records what is in the cells now.
# - Hidden state (deleted cells, out-of-order runs, stale imports) makes notebooks work for you and fail for others. **Restart and Run All** is the test.
# - Magics and shortcuts speed you up but tie code to Jupyter; keep reusable logic in `.py` modules.
# - Strip outputs (nbstripout / pre-commit) or pair notebooks with plain-text sources so git stays useful.
# - VS Code supports both `.ipynb` and `# %%` percent scripts, with a full debugger.
#
# **Go deeper:** the Jupyter and IPython documentation (jupyter.org, ipython.readthedocs.io), and the VS Code docs on Jupyter Notebooks (code.visualstudio.com/docs).
#
# **Next:** [Project structure and reproducibility](05_project_structure_and_reproducibility.ipynb)
