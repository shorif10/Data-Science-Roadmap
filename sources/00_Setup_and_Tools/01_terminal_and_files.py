# %% [markdown]
# # The Terminal and the File System
#
# **Why this matters.** Every tool in this roadmap (git, pip, Jupyter, Docker, cloud CLIs, CI pipelines) is driven
# from a terminal, and most "it does not work" moments are really "I am in the wrong folder" or "that path has a
# space in it". A data scientist who is comfortable in a shell can inspect a 5 GB log file in seconds, chain small
# tools together, and reproduce an environment on a server that has no GUI.
#
# **You will learn**
# - what a shell, a working directory and a path are, and how absolute and relative paths differ
# - the everyday commands in **bash** and **PowerShell** side by side (`ls`, `cd`, `cat`, `grep`, ...)
# - how pipes, redirection, exit codes and environment variables work, by running real child processes from Python
# - the classic traps: spaces in paths, backslashes, line endings, destructive commands
#
# **Prerequisites:** none. If you can open a terminal you can start.
#
# > Everything below runs from Python via `subprocess`, in a throw-away temp folder, so it works the same on
# > Windows, macOS and Linux. Where the two shells differ, a table shows both. Data used: `datasets/orders.csv`
# > (synthetic; see `docs/DATASETS.md`).

# %%
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path("../datasets")
IS_WINDOWS = platform.system() == "Windows"
print("OS:", platform.system(), "| Python:", sys.version.split()[0])

# %% [markdown]
# ## 1. Terminal, shell, working directory
#
# The **terminal** is the window. The **shell** (bash, zsh, PowerShell, cmd) is the program inside it that reads a
# line, starts other programs and prints their output. Every running program (a *process*) has a **current working
# directory** (cwd): the folder that relative paths are resolved against. Your notebook is a process too:

# %%
print("cwd:", Path.cwd())
print("notebook data folder resolves to:", DATA.resolve())

# %% [markdown]
# *What just happened?* `../datasets` is a **relative path**: "go up one folder, then into `datasets`". It only means
# something relative to the cwd. That is why this roadmap tells you to open notebooks from their own module folder:
# move the cwd and the same string points somewhere else (or nowhere).

# %% [markdown]
# ## 2. Paths: absolute, relative, and the special names
#
# | Name | Meaning |
# |---|---|
# | `/` (Linux/macOS root) or `C:\` (Windows drive) | start of an **absolute** path |
# | `.` | the current directory |
# | `..` | the parent directory |
# | `~` | your home folder (shell expands it; in Python use `Path.home()`) |
#
# Windows separates with `\`, Linux/macOS with `/`. Python on Windows accepts `/` too, and `pathlib` hides the
# difference, so **write paths with `pathlib` and forward slashes** in code.

# %%
p = Path("../datasets/orders.csv")
print("absolute?      ", p.is_absolute())
print("name / stem    ", p.name, "/", p.stem)
print("suffix         ", p.suffix)
print("parent         ", p.parent)
print("resolved       ", p.resolve())
print("joined with /  ", Path("a") / "b" / "c.txt")
print("home is        ", Path.home())

# %% [markdown]
# The `/` operator on `Path` objects joins path pieces with the right separator for the OS. Never build paths by
# gluing strings with `+ "/" +`.

# %% [markdown]
# ## 3. Cheat sheet: the same task in bash and PowerShell
#
# | Task | bash / zsh | PowerShell |
# |---|---|---|
# | where am I | `pwd` | `pwd` or `Get-Location` |
# | list files | `ls -la` | `ls` or `Get-ChildItem` |
# | change folder | `cd data` | `cd data` (alias of `Set-Location`) |
# | make folder | `mkdir -p a/b` | `mkdir a/b` (creates parents) |
# | copy / move | `cp a b` / `mv a b` | `cp a b` / `mv a b` (`Copy-Item`, `Move-Item`) |
# | delete file / folder | `rm f` / `rm -r dir` | `rm f` / `rm -r dir` (`Remove-Item`) |
# | print a file | `cat f` | `cat f` (`Get-Content`) |
# | first N lines | `head -n 3 f` | `Get-Content f -TotalCount 3` |
# | count lines | `wc -l f` | `(Get-Content f).Count` |
# | search text | `grep "web" f` | `Select-String "web" f` |
# | find a program | `which python` | `Get-Command python` |
# | env variable | `echo $HOME`, `export X=1` | `$env:HOME`, `$env:X = "1"` |
# | pipe | `a \| b` | `a \| b` (passes **objects**, not text) |
#
# The big conceptual difference: bash pipes **text** between programs; PowerShell pipes **.NET objects**, so
# `Get-ChildItem | Where-Object Length -gt 1000` filters on a real `Length` property instead of parsing columns.
# `cmd.exe` (the old Windows shell) is a third dialect (`dir`, `type`, `findstr`); you will mostly meet it inside
# `subprocess` calls and CI logs.

# %% [markdown]
# ## 4. Running commands from Python
#
# `subprocess.run` starts a child process. Two ways to call it:
# - **list form** `["git", "--version"]`: no shell involved, arguments are passed exactly as given. Safe and portable.
# - **string + `shell=True`**: a shell parses the line, so pipes, `*` globbing and redirection work, but quoting rules
#   now depend on the shell.

# %%
def sh(cmd, cwd=None, **kw):
    """Run a shell command line, print its output, return the CompletedProcess."""
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", **kw)
    out = (r.stdout + r.stderr).rstrip()
    print(f"$ {cmd}\n{out}\n[exit code {r.returncode}]")
    return r


r = subprocess.run(["git", "--version"], capture_output=True, text=True)
print(r.stdout.strip(), "| exit code:", r.returncode)
_ = sh("echo hello from a shell")

# %% [markdown]
# *What just happened?* `capture_output=True` collected the child's **stdout** and **stderr** instead of printing them;
# `text=True` decoded bytes to `str`; the **exit code** (`returncode`) tells you whether it worked: `0` means success,
# anything else means failure. Shells and CI systems use exactly this convention.

# %% [markdown]
# ## 5. A sandbox to play in
# We build a small project folder in a temp directory (deleted at the end of the notebook), so you can experiment
# without touching real files.

# %%
tmp = tempfile.TemporaryDirectory()
box = Path(tmp.name)
(box / "project" / "data" / "raw").mkdir(parents=True)
(box / "project" / "notebooks").mkdir()
(box / "project" / "README.md").write_text("# Demo project\n", encoding="utf-8")
shutil.copy(DATA / "orders.csv", box / "project" / "data" / "raw" / "orders.csv")

for path in sorted(box.rglob("*")):
    print("  " * (len(path.relative_to(box).parts) - 1) + path.name + ("/" if path.is_dir() else ""))

# %% [markdown]
# Now list it with the shell of your OS. `dir` is a cmd builtin; `ls` exists on Linux and macOS:

# %%
_ = sh("dir /b" if IS_WINDOWS else "ls", cwd=box / "project")

# %% [markdown]
# ### Reading a big file without opening it
# With `head`, `wc`, `grep` you never load the whole file. The PowerShell versions are shown for Windows; bash for others.
# (On Windows, `Get-Content` streams a file lazily, which matters for large files.)

# %%
orders = box / "project" / "data" / "raw" / "orders.csv"
have_ps = shutil.which("powershell") or shutil.which("pwsh")


def demo(bash_cmd, ps_cmd, cwd):
    """Run the PowerShell version if it exists, else the bash version."""
    if have_ps:
        exe = have_ps
        r = subprocess.run([exe, "-NoProfile", "-Command", ps_cmd], cwd=cwd, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        print(f"PS> {ps_cmd}\n{(r.stdout + r.stderr).rstrip()}\n")
        return r
    return sh(bash_cmd, cwd=cwd)


_ = demo("head -n 3 data/raw/orders.csv", "Get-Content data/raw/orders.csv -TotalCount 3", box / "project")
_ = demo("wc -l data/raw/orders.csv", "(Get-Content data/raw/orders.csv).Count", box / "project")
_ = demo("grep -c ',returned,' data/raw/orders.csv", "(Select-String ',returned,' data/raw/orders.csv).Count", box / "project")

# %% [markdown]
# The line count includes the header, so rows = lines - 1. We can verify the shell answer in Python:

# %%
n_lines = sum(1 for _ in orders.open(encoding="utf-8"))
n_returned = sum(",returned," in line for line in orders.open(encoding="utf-8"))
print("python says:", n_lines, "lines,", n_returned, "returned orders")

# %% [markdown]
# ## 6. stdin, stdout, stderr, pipes and redirection
#
# Every process has three text streams: **stdin** (input), **stdout** (normal output), **stderr** (errors/log noise).
# - `a | b` connects a's stdout to b's stdin.
# - `a > file` writes stdout to a file (overwrite), `>>` appends, `2> file` redirects stderr.
#
# Rather than depend on `grep`/`sort` being installed, we write two tiny Python programs and pipe them together,
# which is exactly the mechanism behind `cat file | grep x | sort`.

# %%
(box / "producer.py").write_text(
    "import sys\n"
    "for line in open(sys.argv[1], encoding='utf-8'):\n"
    "    print(line.rstrip())\n", encoding="utf-8")
(box / "count_status.py").write_text(
    "import sys, collections\n"
    "rows = [l.rstrip().split(',') for l in sys.stdin][1:]\n"
    "c = collections.Counter(r[3] for r in rows)\n"
    "for status, n in c.most_common():\n"
    "    print(status, n)\n", encoding="utf-8")

py = f'"{sys.executable}"'  # quote: this path contains spaces on this machine
_ = sh(f'{py} producer.py "{orders}" | {py} count_status.py', cwd=box)

# %% [markdown]
# The same pipeline **without a shell**: connect the processes yourself. This is what the shell does for you.

# %%
p1 = subprocess.Popen([sys.executable, "producer.py", str(orders)], cwd=box, stdout=subprocess.PIPE)
p2 = subprocess.run([sys.executable, "count_status.py"], cwd=box, stdin=p1.stdout, capture_output=True, text=True)
p1.stdout.close()
p1.wait()
print(p2.stdout.strip())

# %% [markdown]
# Redirection and exit codes: stdout and stderr are separate streams, and a failing command signals it through a non-zero
# exit code even when it prints nothing to stdout.

# %%
(box / "noisy.py").write_text(
    "import sys\n"
    "print('result: 42')\n"
    "print('warning: something odd', file=sys.stderr)\n"
    "sys.exit(3)\n", encoding="utf-8")
r = subprocess.run([sys.executable, "noisy.py"], cwd=box, capture_output=True, text=True)
print("stdout    :", r.stdout.strip())
print("stderr    :", r.stderr.strip())
print("returncode:", r.returncode)

_ = sh(f'{py} noisy.py > out.txt 2> err.txt', cwd=box)
print("out.txt ->", (box / "out.txt").read_text().strip(), "| err.txt ->", (box / "err.txt").read_text().strip())

# %% [markdown]
# Why you care: a pipeline step that prints an error but exits `0` looks successful to CI and to `&&` chains.
# Conversely `cmd1 && cmd2` runs `cmd2` **only if** `cmd1` exited 0. Check `check=True` in `subprocess.run` to make
# Python raise `CalledProcessError` on a non-zero exit instead of ignoring it.

# %%
try:
    subprocess.run([sys.executable, "noisy.py"], cwd=box, check=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print("raised CalledProcessError, exit code", e.returncode)

# %% [markdown]
# ## 7. Environment variables and PATH
#
# **Environment variables** are key/value strings each process inherits from its parent. `PATH` is the most important:
# an ordered list of folders the shell searches when you type a command name. That is why "`python` runs the wrong
# Python" is really a PATH-order problem, which the next notebooks come back to.

# %%
entries = os.environ["PATH"].split(os.pathsep)
print(f"PATH has {len(entries)} folders, separated by {os.pathsep!r}; first two:")
for e in entries[:2]:
    print("  ", e)
print("git resolves to:", shutil.which("git"))
print("this notebook's python:", sys.executable)

# %% [markdown]
# A child process gets a **copy** of the environment. Changing it in the child never affects the parent, and you can pass
# a modified copy explicitly:

# %%
env = dict(os.environ, EXPERIMENT_NAME="churn_v1")
r = subprocess.run([sys.executable, "-c", "import os; print(os.environ['EXPERIMENT_NAME'])"],
                   env=env, capture_output=True, text=True)
print("child saw:", r.stdout.strip(), "| parent has it?", "EXPERIMENT_NAME" in os.environ)

# %% [markdown]
# Environment variables are the standard place for secrets and per-machine settings (API keys, database URLs), because
# they stay out of your code and out of git.

# %% [markdown]
# ## Common mistakes
#
# 1. **Unquoted paths with spaces.** `cd My Documents` passes two arguments. Quote: `cd "My Documents"`. (This repo lives in
#    a folder with a space in its name, so the traps are real.)
# 2. **Backslashes in Python strings.** `"C:\new\table.csv"` contains a newline and a tab. Use `Path("C:/new/table.csv")`,
#    a raw string `r"C:\new\table.csv"`, or `pathlib`.
# 3. **Deleting with the wrong cwd.** `rm -rf *` (or `Remove-Item * -Recurse`) removes what is *here*. Run `pwd` first; prefer
#    `ls` with the same glob before you delete; there is no recycle bin on the command line.
# 4. **Assuming case-insensitivity.** `Data.csv` and `data.csv` are the same file on default Windows/macOS but different on Linux.
#    Code that works on your laptop then fails on a Linux server.
# 5. **Line endings.** Windows text files often end lines with `\r\n`, Linux with `\n`. Reading in text mode hides it; reading
#    bytes does not.

# %%
print(len("C:\new\table.csv"), "chars: \\n and \\t became one real character each")  # mistake 2
print(len(r"C:\new\table.csv"), "chars: the raw string keeps the backslashes")

crlf = box / "crlf.txt"
crlf.write_bytes(b"a,b\r\n1,2\r\n")
print("as bytes:", crlf.read_bytes())
print("as text :", repr(crlf.read_text()))  # Python's universal newlines translated \r\n

# %%
# Mistake 1 demonstrated with a real command: mkdir with an unquoted name that contains a space makes TWO folders
_ = sh("mkdir my folder", cwd=box)
print(sorted(p.name for p in box.iterdir() if p.is_dir()))

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between an absolute and a relative path, and why do relative paths break?**
# <details><summary>Answer</summary>
# An absolute path names a location from the root of the file system; a relative path is resolved against the process's current
# working directory. Relative paths break when the code is launched from a different directory (a scheduler, another IDE, a container).
# Robust code anchors paths to the script location (<code>Path(__file__).parent</code>) or to a configured project root.
# </details>
#
# **2. What does <code>cmd1 | cmd2</code> do, and how is PowerShell's pipe different?**
# <details><summary>Answer</summary>
# It connects cmd1's stdout to cmd2's stdin so both run concurrently and data streams between them without an intermediate file.
# In bash the streams are plain text; PowerShell passes structured .NET objects, so you filter by property instead of parsing text.
# </details>
#
# **3. A script prints an error but your CI job is green. Why?**
# <details><summary>Answer</summary>
# CI decides pass/fail from the process exit code. If the program catches the error and exits normally (code 0), or the failing
# command sits in a pipeline where only the last command's status counts, the job looks successful. Exit non-zero on failure,
# and use <code>set -e</code> / <code>set -o pipefail</code> in bash or <code>check=True</code> in Python.
# </details>
#
# **4. Where do secrets like API keys belong?**
# <details><summary>Answer</summary>
# In environment variables or a secrets manager, injected at run time, never committed to git. A local <code>.env</code> file is
# convenient but must be listed in <code>.gitignore</code>.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.
# The sandbox variables `box`, `orders`, `py` and the `sh` helper from above are available.

# %% [markdown]
# ### Exercise 1: Which CSV files ship with the datasets?
# Store in `csv_names` the **sorted list of file names** (just the name, e.g. `"orders.csv"`) of every `.csv` file
# anywhere under `DATA` (including subfolders). *Hint: `Path.rglob("*.csv")`.*

# %% tags=["starter"]
csv_names = ...

# %%
expected = sorted(f for _, _, files in os.walk(DATA) for f in files if f.endswith(".csv"))
assert csv_names == expected, f"expected {len(expected)} names like {expected[:3]}, got {csv_names!r}"
assert "orders.csv" in csv_names
print("Exercise 1 passed:", len(csv_names), "csv files")

# %% tags=["solution"]
csv_names = sorted(p.name for p in DATA.rglob("*.csv"))

# %% [markdown]
# ### Exercise 2: A pipeline of your own
# The file `box/count_channel.py` below reads CSV lines on **stdin** and prints `channel count` lines. Run it as a child
# process, feeding it `orders.csv` through `stdin`, and store the parsed result in `channel_counts` as a `dict[str, int]`.
# *Hint: `subprocess.run([...], stdin=open(orders), capture_output=True, text=True)`; parse each output line with `split()`.*

# %%
_ = (box / "count_channel.py").write_text(
    "import sys, collections\n"
    "rows = [l.rstrip().split(',') for l in sys.stdin][1:]\n"
    "for k, v in sorted(collections.Counter(r[4] for r in rows).items()):\n"
    "    print(k, v)\n", encoding="utf-8")

# %% tags=["starter"]
channel_counts = ...

# %%
import csv as _csv
with orders.open(encoding="utf-8") as f:
    truth = {}
    for row in _csv.DictReader(f):
        truth[row["channel"]] = truth.get(row["channel"], 0) + 1
assert channel_counts == truth, f"expected {truth}, got {channel_counts}"
print("Exercise 2 passed:", channel_counts)

# %% tags=["solution"]
with orders.open(encoding="utf-8") as f:
    r = subprocess.run([sys.executable, str(box / "count_channel.py")], stdin=f, capture_output=True, text=True, check=True)
channel_counts = {k: int(v) for k, v in (line.split() for line in r.stdout.splitlines())}

# %% [markdown]
# ### Exercise 3: Pass configuration through the environment
# Run a child Python process that prints `os.environ["ROLE"].upper()`. Give it the variable `ROLE=analyst` **without**
# setting it in this notebook's own environment. Store the child's stripped stdout in `child_out`.

# %% tags=["starter"]
child_out = ...

# %%
assert child_out == "ANALYST", f"expected 'ANALYST', got {child_out!r}"
assert "ROLE" not in os.environ, "the variable must be set only for the child process (pass env=...)"
print("Exercise 3 passed")

# %% tags=["solution"]
r = subprocess.run([sys.executable, "-c", "import os; print(os.environ['ROLE'].upper())"],
                   env=dict(os.environ, ROLE="analyst"), capture_output=True, text=True, check=True)
child_out = r.stdout.strip()

# %% [markdown]
# ### Exercise 4: Find the biggest file
# Which file under `DATA` (any type, any depth, excluding folders) is largest? Store its **name** in `biggest_name` and its size in
# **KiB** (bytes / 1024, rounded to 1 decimal) in `biggest_kib`. *Hint: `p.stat().st_size`, `max(..., key=...)`.*

# %% tags=["starter"]
biggest_name = ...
biggest_kib = ...

# %%
sizes = {}
for root, _, files in os.walk(DATA):
    for f in files:
        sizes[f] = os.path.getsize(os.path.join(root, f))
name = max(sizes, key=sizes.get)
assert biggest_name == name, f"the largest file is not {biggest_name!r}"
assert biggest_kib == round(sizes[name] / 1024, 1), "size should be bytes/1024 rounded to 1 decimal"
print(f"Exercise 4 passed: {biggest_name} is {biggest_kib} KiB")

# %% tags=["solution"]
files = [p for p in DATA.rglob("*") if p.is_file()]
biggest = max(files, key=lambda p: p.stat().st_size)
biggest_name = biggest.name
biggest_kib = round(biggest.stat().st_size / 1024, 1)

# %% [markdown]
# ### Exercise 5 (open-ended): Your own cheat sheet
# In a markdown cell, write the five terminal commands you expect to use most often in your work, with the bash **and** PowerShell
# form of each. Then run one of them against the sandbox folder `box` and note what surprised you.

# %%
tmp.cleanup()  # remove the sandbox
print("sandbox removed:", not box.exists())

# %% [markdown]
# ## Recap
# - A process has a working directory; relative paths are resolved against it. Use `pathlib` and forward slashes in code.
# - bash and PowerShell cover the same basics with different names; bash pipes text, PowerShell pipes objects.
# - stdin/stdout/stderr, `|`, `>`, and **exit codes** are how programs compose; non-zero means failure.
# - Environment variables (`PATH`, secrets, config) are inherited by child processes as copies.
# - Quote paths with spaces, prefer list-form `subprocess`, and check your cwd before destructive commands.
#
# **Go deeper:** the Python docs for `subprocess` and `pathlib` (docs.python.org/3); the PowerShell documentation on
# learn.microsoft.com; the GNU Bash reference manual (gnu.org/software/bash/manual).
#
# **Next:** [Git and GitHub](02_git_and_github.ipynb)
