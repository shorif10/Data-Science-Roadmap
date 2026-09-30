"""
Execute notebooks end-to-end and report failures.

    python tools/check_notebooks.py                 # every notebook
    python tools/check_notebooks.py 04_NumPy        # one module
    python tools/check_notebooks.py 04_NumPy/01_arrays.ipynb
    python tools/check_notebooks.py -j 4 --timeout 300

How it works
  * Each notebook is copied in memory; every "starter" cell is replaced with its paired
    "solution" cell (so exercises are proven solvable and their asserts pass).
  * Cells tagged "no-run" (network, API keys, cloud accounts, GUI) are skipped.
  * The notebook executes with its own folder as working directory, exactly like a learner
    opening it in JupyterLab, so ../datasets/... paths are tested for real.
  * Nothing is written back: your notebooks stay output-free.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


def prepare(nb):
    nb = copy.deepcopy(nb)
    starters = [i for i, c in enumerate(nb.cells) if "starter" in c.metadata.get("tags", [])]
    solutions = [i for i, c in enumerate(nb.cells) if "solution" in c.metadata.get("tags", [])]
    if len(starters) != len(solutions):
        raise ValueError(f"{len(starters)} starters vs {len(solutions)} solutions")
    for s, sol in zip(starters, solutions):
        nb.cells[s].source = nb.cells[sol].source
    drop = set(solutions)
    skipped = 0
    keep = []
    for i, c in enumerate(nb.cells):
        if i in drop:
            continue
        if c.cell_type == "code" and "no-run" in c.metadata.get("tags", []):
            skipped += 1
            continue
        keep.append(c)
    nb.cells = keep
    return nb, skipped, len(starters)


def run_one(path: str, timeout: int):
    p = Path(path)
    t0 = time.time()
    try:
        nb = nbformat.read(p, as_version=4)
        nb, skipped, n_ex = prepare(nb)
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0", TF_CPP_MIN_LOG_LEVEL="2",
                   TF_ENABLE_ONEDNN_OPTS="0", KERAS_BACKEND=os.environ.get("KERAS_BACKEND", "tensorflow"),
                   MLFLOW_DISABLE_AGENT_HINT="1", PYTHONIOENCODING="utf-8")
        os.environ.update(env)
        client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(p.parent)}}, allow_errors=False)
        client.execute()
        return path, True, f"{time.time() - t0:5.1f}s  exercises={n_ex} skipped={skipped}", ""
    except CellExecutionError as exc:
        msg = str(exc)
        return path, False, f"{time.time() - t0:5.1f}s", msg[-1800:]
    except Exception:  # noqa: BLE001
        return path, False, f"{time.time() - t0:5.1f}s", traceback.format_exc()[-1800:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--timeout", type=int, default=300, help="seconds per cell")
    args = ap.parse_args()

    files = []
    if args.targets:
        for t in args.targets:
            p = ROOT / t
            if p.is_dir():
                files += sorted(p.glob("*.ipynb"))
            elif p.is_file():
                files.append(p)
            else:
                sys.exit(f"not found: {t}")
    else:
        files = sorted(ROOT.glob("[0-9][0-9]_*/*.ipynb"))
    files = [f for f in files if ".ipynb_checkpoints" not in str(f)]
    if not files:
        sys.exit("no notebooks found - run tools/build_notebooks.py first")

    failed = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_one, str(f), args.timeout) for f in files]
        for fu in as_completed(futs):
            path, ok, info, err = fu.result()
            rel = Path(path).relative_to(ROOT)
            print(f"{'PASS' if ok else 'FAIL'}  {rel}  {info}", flush=True)
            if not ok:
                failed.append((rel, err))
    print(f"\n{len(files) - len(failed)}/{len(files)} notebooks passed")
    for rel, err in sorted(failed):
        print(f"\n=== {rel} ===\n{err}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
