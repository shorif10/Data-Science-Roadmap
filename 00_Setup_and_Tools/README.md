# 00 Setup and Tools

**Goal:** be fluent in the daily tooling of a data scientist: terminal, git, Python environments, Jupyter/VS Code, and a reproducible project layout.

**Prerequisites:** none (a working Python 3.11+ and git).

## Notebooks (about 4 hours)

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 1 | [Terminal and files](01_terminal_and_files.ipynb) | paths, bash vs PowerShell, pipes, exit codes, env vars via `subprocess` | 0.75 |
| 2 | [Git and GitHub](02_git_and_github.ipynb) | commit, branch, merge, resolve a conflict, `.gitignore`, local "remote" push/pull, PR workflow | 1 |
| 3 | [Environments and packages](03_environments_and_packages.ipynb) | venv, pip, pinning, drift checks, build and inspect a real venv | 0.75 |
| 4 | [Jupyter and VS Code workflow](04_jupyter_and_vscode_workflow.ipynb) | magics, shortcuts, hidden-state bugs on real kernels, clean notebooks for git | 0.75 |
| 5 | [Project structure and reproducibility](05_project_structure_and_reproducibility.ipynb) | layout, YAML config, seeds, logging, run manifest | 1 |

## You are done when you can...
- [ ] explain absolute vs relative paths and fix a "file not found" caused by the working directory
- [ ] combine commands with pipes and read an exit code in bash and PowerShell
- [ ] create a branch, commit, merge it, and resolve a merge conflict by hand
- [ ] write a `.gitignore` and explain why a committed secret must be rotated, not just deleted
- [ ] create a venv, tell which interpreter a notebook kernel uses, and freeze/pin requirements
- [ ] demonstrate and fix a hidden-state notebook bug with Restart and Run All
- [ ] lay out a project with config, seeds and logging so a teammate gets identical results

## Try next
- The Pro Git book (git-scm.com/book) and GitHub Skills courses (skills.github.com)
- The Python Packaging User Guide (packaging.python.org)
- The Cookiecutter Data Science template (github.com/drivendataorg/cookiecutter-data-science)
- Practise with a real repo: fork a small open-source project and open a documentation pull request
