# %% [markdown]
# # Git and GitHub
#
# **Why this matters.** Every team you will join keeps its code, notebooks and configs in git, and "send me the
# latest version" is not a workflow. Git gives you a safety net (undo anything that was committed), a way to
# work on an idea without breaking the working version (branches), and a shared history other people can review
# (pull requests). In interviews, "walk me through how you'd contribute a change" is a standard question.
#
# **You will learn**
# - the mental model: working tree, staging area, commits, branches
# - init, add, commit, log, diff, restore and revert, by running them in a real throw-away repository
# - branch and merge, and how to **resolve a merge conflict** by reading the conflict markers
# - `.gitignore` (and why it does not un-track files) and why deleting a secret is not enough
# - the GitHub workflow (clone, push, fork, pull request), rehearsed with a local "remote"
#
# **Prerequisites:** [Terminal and files](01_terminal_and_files.ipynb).
#
# > Everything here runs `git` through `subprocess` in a temp folder, so your own repos are never touched.
# > The parts that need a GitHub account are shown as `no-run` cells and rehearsed locally instead.

# %%
import shutil
import subprocess
import tempfile
from pathlib import Path

assert shutil.which("git"), "git is not installed or not on PATH"
tmp = tempfile.TemporaryDirectory()
base = Path(tmp.name)


def write(path, text):
    """Write text with plain \n line endings on every OS (keeps git output stable)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def git(*args, cwd, check=True, show=True):
    """Run `git <args>` in cwd. show=True prints the output; show=False returns it instead."""
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (r.stdout + r.stderr).strip()
    if show:
        print("$ git " + " ".join(args) + (f"\n{out}" if out else ""))
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args} failed ({r.returncode}): {out}")
    if show:
        return None  # output already printed; keeps notebook cells free of duplicate echo
    return r.stdout.strip() if check else r


def new_repo(name):
    """Create an empty repo with a local identity so commits work anywhere."""
    repo = base / name
    repo.mkdir()
    git("init", "-b", "main", cwd=repo, show=False)
    for k, v in [("user.name", "Learner"), ("user.email", "learner@example.com"),
                 ("commit.gpgsign", "false"), ("core.autocrlf", "false")]:
        git("config", k, v, cwd=repo, show=False)
    return repo


print(git("--version", cwd=base, show=False))

# %% [markdown]
# ## 1. The mental model: three areas
#
# ```
#  working tree  --git add-->  staging area (index)  --git commit-->  repository (history)
#  (your files)                 (what the next commit will contain)    (immutable snapshots)
# ```
#
# A **commit** is a snapshot of the whole project plus a message, an author, and a pointer to its parent commit(s).
# A **branch** is just a movable label pointing at a commit. `HEAD` says which branch (or commit) you are on.
# The staging area exists so you can commit *some* of your changes as one logical unit, not everything at once.

# %% [markdown]
# ## 2. First repository: init, add, commit

# %%
repo = new_repo("analysis")
write(repo / "README.md", "# Churn analysis\n")
git("status", "--short", cwd=repo)          # ?? means untracked

# %%
git("add", "README.md", cwd=repo)
git("status", "--short", cwd=repo)          # A = staged (added), waiting for a commit

# %%
git("commit", "-m", "Add README", cwd=repo)
git("log", "--oneline", cwd=repo);

# %% [markdown]
# *What just happened?* `add` copied the file into the staging area, `commit` turned the staged snapshot into a permanent
# commit with a short hash (the 7 hex characters). The hash is derived from the content and history, so two people cannot
# accidentally create the same commit id for different content.
#
# Let's make a couple more commits and look at what changed between them.

# %%
write(repo / "clean.py", "def clean(df):\n    return df.dropna()\n")
git("add", "clean.py", cwd=repo, show=False)
git("commit", "-m", "Add cleaning function", cwd=repo, show=False)

write(repo / "clean.py", "def clean(df):\n    return df.dropna().drop_duplicates()\n")
git("diff", cwd=repo)                        # unstaged changes: working tree vs index

# %%
git("commit", "-am", "Also drop duplicate rows", cwd=repo, show=False)   # -a stages tracked files, -m the message
git("log", "--oneline", cwd=repo);

# %% [markdown]
# `git diff` shows what you changed but have not staged (`--staged` shows what is staged). Habit: run `git status` and
# `git diff` **before** every commit, so you know exactly what you are saving.
#
# ### Undoing things
# | Situation | Command | Note |
# |---|---|---|
# | edited a file, want it back as committed | `git restore file` | throws away the edit, no undo |
# | staged a file by mistake | `git restore --staged file` | keeps the edit |
# | a **pushed** commit was wrong | `git revert <hash>` | adds a new commit that undoes it; safe for shared history |
# | a **local, unpushed** commit was wrong | `git reset --hard <hash>` | rewrites history; destructive |

# %%
write(repo / "clean.py", "OOPS, broke it\n")
git("restore", "clean.py", cwd=repo)
print((repo / "clean.py").read_text())

# %%
bad = git("rev-parse", "--short", "HEAD", cwd=repo, show=False)
git("revert", "--no-edit", "HEAD", cwd=repo, show=False)
git("log", "--oneline", cwd=repo);

# %% [markdown]
# The revert did not delete the "drop duplicates" commit: history stays honest and a new commit reverses its effect.

# %% [markdown]
# ## 3. Branches and merging
#
# A branch lets you try something without disturbing `main`. Create one with `git switch -c <name>`
# (older material uses `git checkout -b`, same effect).

# %%
git("switch", "-c", "add-eval", cwd=repo, show=False)
write(repo / "evaluate.py", "def accuracy(y, p):\n    return sum(a == b for a, b in zip(y, p)) / len(y)\n")
git("add", ".", cwd=repo, show=False)
git("commit", "-m", "Add evaluation helper", cwd=repo, show=False)

git("switch", "main", cwd=repo, show=False)
print("evaluate.py on main?", (repo / "evaluate.py").exists())   # the file is not there: it lives on the branch

# %%
git("merge", "--no-ff", "-m", "Merge add-eval", "add-eval", cwd=repo, show=False)
print("evaluate.py on main?", (repo / "evaluate.py").exists())
git("log", "--oneline", "--graph", cwd=repo);

# %% [markdown]
# *What just happened?* Switching branches rewrites your working tree to match that branch. The merge created a **merge
# commit** with two parents (`--no-ff` forces one, so the branch stays visible in the graph; without it git would
# "fast-forward" the label because `main` had not moved).

# %% [markdown]
# ## 4. Merge conflicts
#
# If both branches changed the *same lines*, git cannot choose and asks you. A conflict is not an error in git; it is a
# question. Let's manufacture one: two people set the model in a config file differently.

# %%
def make_conflict(name):
    r = new_repo(name)
    write(r / "settings.txt", "project = churn\nmodel = baseline\n")
    git("add", ".", cwd=r, show=False)
    git("commit", "-m", "Initial settings", cwd=r, show=False)
    git("switch", "-c", "experiment", cwd=r, show=False)
    write(r / "settings.txt", "project = churn\nmodel = random_forest\n")
    git("commit", "-am", "Try random forest", cwd=r, show=False)
    git("switch", "main", cwd=r, show=False)
    write(r / "settings.txt", "project = churn\nmodel = logistic\n")
    git("commit", "-am", "Use logistic regression", cwd=r, show=False)
    return r


crepo = make_conflict("conflict_demo")
res = git("merge", "experiment", cwd=crepo, check=False, show=False)
print("merge exit code:", res.returncode)
print(res.stdout.strip())
print("----- settings.txt now -----")
print((crepo / "settings.txt").read_text())

# %% [markdown]
# Read the markers:
# - `<<<<<<< HEAD` ... `=======` is **your side** (the branch you are on, `main`).
# - `=======` ... `>>>>>>> experiment` is **their side** (the branch being merged).
#
# Resolving = edit the file to the text you want (delete all three marker lines), `git add` it, then `git commit`.
# `git status` tells you which files are still "unmerged".

# %%
git("status", "--short", cwd=crepo)
write(crepo / "settings.txt", "project = churn\nmodel = random_forest\n")   # our decision
git("add", "settings.txt", cwd=crepo, show=False)
git("commit", "--no-edit", cwd=crepo, show=False)
git("log", "--oneline", "--graph", cwd=crepo);

# %% [markdown]
# Tip: conflicts in **notebooks** are painful because a `.ipynb` is JSON with outputs and execution counters that change on every
# run. That is why the next-but-one notebook teaches you to strip outputs before committing.

# %% [markdown]
# ## 5. `.gitignore`: what never belongs in a repo
#
# Data files, secrets (`.env`), virtual environments, caches and notebook checkpoints do not belong in git.
# List patterns in a `.gitignore` file at the repo root.

# %%
def make_project(name):
    r = new_repo(name)
    write(r / "README.md", "# demo\n")
    write(r / "src/app.py", "print('hi')\n")
    write(r / "data/big.csv", "a,b\n1,2\n")
    write(r / ".env", "API_KEY=super-secret\n")
    write(r / "src/__pycache__/app.cpython-311.pyc", "binary junk\n")
    write(r / "notebooks/.ipynb_checkpoints/x-checkpoint.ipynb", "{}\n")
    return r


def untracked(r):
    out = git("status", "--porcelain", "-uall", cwd=r, show=False)
    return sorted(line[3:] for line in out.splitlines())


proj = make_project("ignore_demo")
print("git would add:", untracked(proj))

# %%
write(proj / ".gitignore", ".env\ndata/\n__pycache__/\n.ipynb_checkpoints/\n")
print("git would add:", untracked(proj))
git("check-ignore", "-v", ".env", cwd=proj)

# %% [markdown]
# `check-ignore -v` tells you **which rule** ignores a path, the fastest way to debug "why isn't git seeing my file?".
#
# **Trap:** `.gitignore` only affects files that are *not yet tracked*. If you committed `.env` first and add it to `.gitignore`
# afterwards, git keeps tracking it. Use `git rm --cached .env` to stop tracking it (keeps the file on disk):

# %%
leaky = make_project("leaky")
git("add", "-A", cwd=leaky, show=False)
git("commit", "-m", "Oops, committed everything", cwd=leaky, show=False)
write(leaky / ".gitignore", ".env\n")
print(".env still tracked after adding to .gitignore:", ".env" in git("ls-files", cwd=leaky, show=False).split())
git("rm", "--cached", ".env", cwd=leaky, show=False)
print(".env tracked now:", ".env" in git("ls-files", cwd=leaky, show=False).split())

# %% [markdown]
# **Worse:** the secret is still in the *history*. Anyone who clones the repo can find it:

# %%
found = git("log", "--all", "--oneline", "-S", "super-secret", cwd=leaky, show=False)
print("commits that ever contained the secret:", found or "none")

# %% [markdown]
# If a secret is committed, **rotate it** (issue a new key and revoke the old one) first; rewriting history (`git filter-repo`,
# BFG) is a cleanup step, not a fix, and it does not help if the repo was already pushed or forked.

# %% [markdown]
# ## 6. Remotes and the GitHub workflow
#
# GitHub is a hosted **remote**: another copy of the repository plus a web UI for review. The commands are the same as for any
# remote, so we rehearse with a local *bare* repository (a repo with no working tree, which is what a server holds):

# %%
server = base / "server.git"
git("init", "--bare", "-b", "main", str(server), cwd=base, show=False)

alice = base / "alice"
git("clone", str(server), str(alice), cwd=base, show=False)
for k, v in [("user.name", "Alice"), ("user.email", "alice@example.com"), ("commit.gpgsign", "false"), ("core.autocrlf", "false")]:
    git("config", k, v, cwd=alice, show=False)
write(alice / "README.md", "# team project\n")
git("add", ".", cwd=alice, show=False)
git("commit", "-m", "Initial commit", cwd=alice, show=False)
git("push", "-u", "origin", "main", cwd=alice, show=False)

bob = base / "bob"
git("clone", str(server), str(bob), cwd=base, show=False)
for k, v in [("user.name", "Bob"), ("user.email", "bob@example.com"), ("commit.gpgsign", "false"), ("core.autocrlf", "false")]:
    git("config", k, v, cwd=bob, show=False)
print("Bob cloned:", (bob / "README.md").read_text().strip())

# %% [markdown]
# Now Alice pushes a new commit. Bob's copy does not know yet. **`fetch`** downloads new commits without touching your files;
# **`pull`** is `fetch` + merge (or rebase) in one step.

# %%
write(alice / "notes.md", "first finding: month-to-month churns more\n")
git("add", ".", cwd=alice, show=False)
git("commit", "-m", "Add notes", cwd=alice, show=False)
git("push", cwd=alice, show=False)

git("fetch", cwd=bob, show=False)
print("Bob is behind by:", git("rev-list", "--count", "HEAD..origin/main", cwd=bob, show=False), "commit(s)")
git("pull", "--ff-only", cwd=bob, show=False)
print("Bob now has notes.md:", (bob / "notes.md").exists())

# %% [markdown]
# ### The pull-request workflow on real GitHub
# The cells below need a GitHub account and network, so they are display-only (`no-run`).
# This is the flow you will use in almost every team:
#
# 1. **Fork** (open source) or get write access (your company repo), then `git clone`.
# 2. `git switch -c fix-missing-values`: **one branch per change**, never work directly on `main`.
# 3. Commit small, focused changes with messages that say *why* ("Impute tenure with median, 3% missing").
# 4. `git push -u origin fix-missing-values`, then open a **pull request** (PR) on GitHub.
# 5. A reviewer comments; you push more commits to the same branch; CI runs tests.
# 6. Merge (merge commit, squash or rebase, per team policy), delete the branch, `git switch main && git pull`.

# %% tags=["no-run"]
# Needs a GitHub account + network, so it is not executed here.
# git clone https://github.com/<you>/<repo>.git
# cd <repo>
# git switch -c fix-missing-values
# ... edit, then:
# git add -p                # review and stage hunk by hunk
# git commit -m "Impute tenure with median (3% missing)"
# git push -u origin fix-missing-values
# gh pr create --fill       # GitHub CLI; or click "Compare & pull request" in the web UI
#
# # keeping a fork in sync with the original ("upstream")
# git remote add upstream https://github.com/<original-owner>/<repo>.git
# git fetch upstream && git switch main && git merge upstream/main

# %% [markdown]
# ## Common mistakes
#
# 1. **`git add .` without looking** and committing data, `.env` or a 200 MB model. Run `git status` first; keep a `.gitignore` from day one.
# 2. **One giant commit** ("stuff") at the end of the week. Small commits are reviewable, revertable and bisectable.
# 3. **Committing on `main` directly** in a team repo, or force-pushing (`git push --force`) a branch others use. Force-push rewrites
#    shared history; if you must, use `--force-with-lease` on your own branch only.
# 4. **Detached HEAD.** `git switch --detach <commit-hash>` (or `git checkout <commit-hash>`) puts you on no branch; commits made there are easy to lose. Make a branch
#    (`git switch -c rescue`) before you continue.
# 5. **Assuming `git rm` or a new commit erases a secret.** It stays in history (shown above).

# %% [markdown]
# ## Interview corner
#
# **1. What is the staging area for?**
# <details><summary>Answer</summary>
# It lets you assemble the next commit deliberately: stage only the files (or hunks with <code>git add -p</code>) that belong to one logical
# change, even if your working tree has other edits.
# </details>
#
# **2. `git merge` versus `git rebase`?**
# <details><summary>Answer</summary>
# Merge joins two histories with a merge commit and preserves what actually happened. Rebase replays your commits on top of another base,
# giving a linear history but rewriting commit ids; never rebase commits that others have already pulled.
# </details>
#
# **3. `git fetch` versus `git pull`?**
# <details><summary>Answer</summary>
# <code>fetch</code> only updates your remote-tracking branches (<code>origin/main</code>); <code>pull</code> is fetch followed by merging (or rebasing)
# into your current branch.
# </details>
#
# **4. You committed an API key and pushed. What do you do?**
# <details><summary>Answer</summary>
# Revoke and rotate the key immediately, because it must be treated as compromised. Then remove it from the code and, if policy requires,
# scrub history. Add the file to <code>.gitignore</code> and load secrets from environment variables.
# </details>

# %% [markdown]
# ## Exercises
#
# The helpers `git`, `write`, `new_repo`, `untracked`, `make_project`, `make_conflict` from above are available.

# %% [markdown]
# ### Exercise 1: Branch, commit, merge
# In the fresh repo `ex1` (one commit on `main`), do this with the `git(...)` helper: create branch `add-notes`, add a file `notes.md`
# containing any text, commit it with the message `Add notes`, switch back to `main`, merge with `--no-ff` and a merge message of your choice,
# then delete the branch (`git branch -d add-notes`). *Hint: the sections above show every command.*

# %%
ex1 = new_repo("ex1")
write(ex1 / "README.md", "# ex1\n")
git("add", ".", cwd=ex1, show=False)
git("commit", "-m", "Initial commit", cwd=ex1, show=False)

# %% tags=["starter"]
# git("switch", "-c", "add-notes", cwd=ex1)
# ...

# %%
branches = git("branch", "--format=%(refname:short)", cwd=ex1, show=False).split()
assert branches == ["main"], f"only 'main' should remain, found {branches}"
assert (ex1 / "notes.md").exists(), "notes.md should be on main after the merge"
subjects = git("log", "--format=%s", cwd=ex1, show=False).splitlines()
assert "Add notes" in subjects, "commit message 'Add notes' not found in history"
parents = git("rev-list", "--parents", "-n", "1", "HEAD", cwd=ex1, show=False).split()
assert len(parents) == 3, "HEAD should be a merge commit with two parents (use --no-ff)"
print("Exercise 1 passed:", subjects)

# %% tags=["solution"]
git("switch", "-c", "add-notes", cwd=ex1, show=False)
write(ex1 / "notes.md", "first notes\n")
git("add", "notes.md", cwd=ex1, show=False)
git("commit", "-m", "Add notes", cwd=ex1, show=False)
git("switch", "main", cwd=ex1, show=False)
git("merge", "--no-ff", "-m", "Merge add-notes", "add-notes", cwd=ex1, show=False)
git("branch", "-d", "add-notes", cwd=ex1, show=False)

# %% [markdown]
# ### Exercise 2: A `.gitignore` that fits a data project
# `make_project` created a folder with `README.md`, `src/app.py`, `data/big.csv`, `.env`, a `__pycache__` file and a notebook checkpoint.
# Write the text of a `.gitignore` in `gitignore_text` (a string) so that git would add **only** `.gitignore`, `README.md` and `src/app.py`.

# %% tags=["starter"]
gitignore_text = ...

# %%
p2 = make_project("ex2")
write(p2 / ".gitignore", gitignore_text)
got = untracked(p2)
assert got == [".gitignore", "README.md", "src/app.py"], f"git would still add: {got}"
print("Exercise 2 passed: git sees", got)

# %% tags=["solution"]
gitignore_text = ".env\ndata/\n__pycache__/\n.ipynb_checkpoints/\n"

# %% [markdown]
# ### Exercise 3: Resolve a conflict
# `ex3` below is mid-merge with a conflict in `settings.txt` (`main` says `logistic`, `experiment` says `random_forest`).
# Your team decided: **keep random_forest**. Store the final file content (no markers) in `resolved`. The check finishes the merge for you.

# %%
ex3 = make_conflict("ex3")
res = git("merge", "experiment", cwd=ex3, check=False, show=False)
assert res.returncode != 0
print((ex3 / "settings.txt").read_text())

# %% tags=["starter"]
resolved = ...

# %%
write(ex3 / "settings.txt", resolved)
assert "<<<<<<<" not in resolved and ">>>>>>>" not in resolved and "=======" not in resolved, "conflict markers are still in your text"
git("add", "settings.txt", cwd=ex3, show=False)
git("commit", "--no-edit", cwd=ex3, show=False)
parents = git("rev-list", "--parents", "-n", "1", "HEAD", cwd=ex3, show=False).split()
assert len(parents) == 3, "the merge should be completed (merge commit with two parents)"
content = (ex3 / "settings.txt").read_text()
assert "model = random_forest" in content and "logistic" not in content and "project = churn" in content, "keep project line and the random_forest choice"
print("Exercise 3 passed")

# %% tags=["solution"]
resolved = "project = churn\nmodel = random_forest\n"

# %% [markdown]
# ### Exercise 4: Read the history
# From repo `repo` (built in section 2), store in `subjects` the list of commit messages **oldest first**.
# *Hint: `git log --format=%s` prints newest first, and `--reverse` flips it.*

# %% tags=["starter"]
subjects = ...

# %%
newest_first = git("log", "--format=%s", cwd=repo, show=False).splitlines()
assert subjects == newest_first[::-1], "expected oldest-first order"
assert subjects[0] == "Add README"
print("Exercise 4 passed:", subjects)

# %% tags=["solution"]
subjects = git("log", "--reverse", "--format=%s", cwd=repo, show=False).splitlines()

# %% [markdown]
# ### Exercise 5 (open-ended): Your commit-message convention
# Look at the `git log --oneline` output above. Rewrite two of the messages so a teammate would know *why* the change was made.
# Then write in a markdown cell your team's rule for when to branch, and when a change deserves its own pull request.

# %%
tmp.cleanup()
print("temp repos removed:", not base.exists())

# %% [markdown]
# ## Recap
# - working tree -> `add` -> staging area -> `commit` -> history; a branch is a label on a commit.
# - `status` and `diff` before every commit; `restore` for uncommitted edits, `revert` for shared history.
# - Conflicts are markers in a file: pick the final text, `add`, `commit`.
# - `.gitignore` affects only untracked files; secrets in history must be rotated, not just deleted.
# - GitHub is a remote plus review: branch, push, pull request, merge.
#
# **Go deeper:** the free *Pro Git* book (git-scm.com/book) and the GitHub documentation (docs.github.com).
#
# **Next:** [Environments and packages](03_environments_and_packages.ipynb)
