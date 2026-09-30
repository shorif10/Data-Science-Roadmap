# Data Science Roadmap

A self-study curriculum as **executable notebooks**: read the concept, run the code, solve exercises that check
your answers automatically. Roughly 25 modules from Python to LLM agents, ending in a full capstone.

- **What to learn and in which order:** [ROADMAP.md](ROADMAP.md)
- **Every notebook with a progress checklist:** [docs/INDEX.md](docs/INDEX.md)
- **The data you will use:** [docs/DATASETS.md](docs/DATASETS.md)

## Quick start (Windows)

```powershell
cd "D:\Learning Journey\notebook\Data-Science-Roadmap"
.venv\Scripts\activate                    # environment is already created
jupyter lab                               # then open 00_Setup_and_Tools or 01_Python
```

Starting from scratch on another machine:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt            # core: Phases 0-2
pip install -r requirements-full.txt       # everything: DL, MLOps, LLM/agents (large download)
python datasets\make_datasets.py           # regenerate the datasets (offline, seeded)
```

`requirements-lock.txt` records the exact versions everything here was verified with.

## How the notebooks work

Each notebook follows one pattern: **intuition → runnable code → common mistakes → interview questions → exercises → recap**.

Exercises have four parts: the task, a **starter cell** where you write, a **check cell** that tells you if you are right,
and a **collapsed solution**. Try first; open the solution only when stuck (in JupyterLab, click the cell's collapsed
input to expand it).

Cells marked *no-run* need something this repo cannot ship (a cloud account, an API key, downloading model weights, a GUI).
The README of that module tells you exactly what to set up, and where a runnable local equivalent exists it is provided.

## Repository layout

```
NN_Module/          notebooks (.ipynb) + README for each module, in learning order
sources/            the plain-text source of every notebook (edit these, then rebuild)
datasets/           generated data + make_datasets.py
tools/              build_notebooks.py, check_notebooks.py
src/ tests/ docker/ airflow/ dvc/ configs/     the capstone project code (25_End_to_End_Project)
docs/               STYLE_GUIDE, DATASETS, INDEX
legacy/             the original generator script (kept for reference; it did not run)
```

## Maintaining and extending

```powershell
python tools\build_notebooks.py 05_Pandas        # sources -> notebooks (and syntax check)
python tools\check_notebooks.py 05_Pandas        # execute them end to end, exercises solved
python tools\check_notebooks.py                  # everything
```

To add a notebook, write `sources/<module>/<NN_name>.py` following [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md), then build
and check it. `check_notebooks.py` substitutes every solution into its starter cell and runs the whole thing, so an exercise
that cannot be solved, or a notebook that no longer runs after a library upgrade, shows up immediately.

## Honest caveats

- The datasets are **synthetic**. Patterns in them are real (planted) and results are reproducible, but real-world data is
  messier. Every module points to real public datasets to practise on next.
- Snowflake, Power BI and Tableau are cloud/GUI tools. The notebooks teach the concepts and prepare data, and they compute the
  numbers you should see in the tool. The vendor-specific steps could not be executed here and are marked *verify in the current docs*.
- LLM/agent notebooks run offline against a deterministic mock model so you can execute the whole loop; swapping in the real API
  is a one-line change shown in each notebook.
