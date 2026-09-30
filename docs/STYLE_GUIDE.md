# Notebook Style Guide

Every notebook in this roadmap is written as a plain-text source in `sources/<module>/<name>.py`
(percent format) and built into `<module>/<name>.ipynb` by `tools/build_notebooks.py`.
`tools/check_notebooks.py` then **executes every notebook end to end**. A notebook that does not
run is a bug. Read `sources/05_Pandas/01_loading_and_inspecting_data.py` first: it is the reference example.

## Workflow

```bash
.venv/Scripts/python tools/build_notebooks.py 05_Pandas      # sources -> .ipynb (also syntax-checks)
.venv/Scripts/python tools/check_notebooks.py 05_Pandas      # execute them for real
```

## Source format

```
# %% [markdown]
# # Title                      <- markdown: every line prefixed "# "

# %%
code here                    <- code cell

# %% tags=["starter"]         <- exercise starter (learner writes here)
# %% tags=["solution"]        <- hidden solution; k-th solution pairs with k-th starter
# %% tags=["no-run"]          <- displayed, but skipped by the checker
```

`no-run` is only for things that truly cannot run offline: paid/cloud accounts (Snowflake, Power BI,
Tableau), API keys, model downloads, GUI apps. Everything else must execute.

## Anatomy of a notebook (in this order)

1. **H1 title**, then a 2-3 sentence *Why this matters* grounded in a real job task, not "X is a powerful library".
2. **You will learn** (3-5 bullets, concrete verbs) and **Prerequisites** (link previous notebooks).
3. **Concept blocks**: intuition first (analogy or tiny example), then code, then a one-line
   *What just happened?* that names the idea. Small steps, one idea per code cell.
4. **Common mistakes**: the real bugs learners hit, shown with code where possible.
5. **Interview corner**: 3-4 questions actually asked in interviews, each with a short answer inside
   `<details><summary>Answer</summary> ... </details>`.
6. **Exercises**: 3-5, escalating. See below.
7. **Recap** bullets, **Go deeper** (official docs / well-known references only; never invent URLs),
   and **Next** link to the following notebook.

Target: 20-40 cells, about 30 minutes of reading and typing, and it must execute in under ~2 minutes.

## Exercises (the part that makes learning stick)

Each exercise is four cells, in this order:

1. Markdown: the task, phrased as a business question against a provided dataset
   ("Which country has the highest average order value?"), plus a hint.
2. `starter` cell: variable to assign, with `...` or a comment. Must be valid Python.
3. Check cell (untagged): `assert` statements with helpful messages. Compute the expected values
   from the data or an independent method inside the check; never hard-code numbers you did not compute.
4. `solution` cell: the reference answer (collapsed in JupyterLab).

Learner loop: write in the starter, run the check, and open the solution only when stuck. The checker
substitutes each solution for its starter, so every exercise is proven solvable.
The last exercise in a notebook should be open-ended ("explore and report one finding") with no assert.

## Data

* Use `Path("../datasets/...")`: notebooks run with their module folder as the working directory.
* Datasets are synthetic and reproducible (`datasets/make_datasets.py`, described in `docs/DATASETS.md`).
  Say so once, and use real-world framing. End each module with pointers to real public datasets
  (UCI, Kaggle, `sklearn.datasets` bundled ones) for the learner to try next.
* Never download anything at run time. `sklearn.datasets` bundled sets (iris, wine, digits, breast_cancer,
  diabetes) are fine; `fetch_*` functions are not.

## Code rules

* `np.random.default_rng(42)` / `random_state=42` everywhere. Same output every run.
* Current library behaviour: NumPy 2.x, pandas 3.x (copy-on-write, `str` dtype, no chained assignment),
  scikit-learn 1.9, PyTorch 2.x, Keras 3, TensorFlow 2.21. Do not write deprecated idioms.
* Plots: always title, axis labels, and units; call `plt.show()`; no seaborn styles that need downloads.
* Keep prints short. Show results, not walls of text.
* No `input()`, no `time.sleep()` > 1s, no writes outside the notebook folder or a temp dir (clean up after).

## Content honesty

* Every claim you can test, test in a cell. Do not state library behaviour from memory if unsure; run it.
* Do not invent statistics, benchmarks, papers, or URLs. Say "roughly" and show the measurement instead.
* When a technique has a trade-off or a failure mode, say so; do not oversell.
* Prefer plain words. Define a term the first time it appears.
