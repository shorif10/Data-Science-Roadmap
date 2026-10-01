"""
Audit the notebooks beyond "does it run".

    python tools/audit_notebooks.py structure        # fast: sections, links, tags
    python tools/audit_notebooks.py discriminate     # slow: run notebooks UNSOLVED; every exercise check must FAIL

`discriminate` proves each exercise check actually tests something: with the starter left
untouched, the check cell that follows must raise. A check that passes on an unsolved
starter is worthless to the learner.
"""
from __future__ import annotations

import copy
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(([^)#\s]+)(#[^)]*)?\)")


def notebooks():
    return sorted(p for p in ROOT.glob("[0-9][0-9]_*/*.ipynb") if ".ipynb_checkpoints" not in str(p))


CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def strip_code(text):
    return CODE.sub("", text)


def md_of(nb):
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")


def structure():
    problems = []
    names = {p.parent.name: [] for p in notebooks()}
    for p in notebooks():
        nb = nbformat.read(p, as_version=4)
        rel = p.relative_to(ROOT).as_posix()
        text = md_of(nb)
        names[p.parent.name].append(p.name)
        if not nb.cells[0].source.lstrip().startswith("# "):
            problems.append((rel, "first cell is not an H1"))
        for needed in ("Exercise", "Recap"):
            if needed not in text:
                problems.append((rel, f"missing '{needed}' section"))
        if "Interview" not in text:
            problems.append((rel, "missing Interview section"))
        if "Next" not in text:
            problems.append((rel, "missing Next link"))
        n_start = sum("starter" in c.metadata.get("tags", []) for c in nb.cells)
        n_sol = sum("solution" in c.metadata.get("tags", []) for c in nb.cells)
        if n_start < 2:
            problems.append((rel, f"only {n_start} checked exercises"))
        if n_start != n_sol:
            problems.append((rel, "starter/solution mismatch"))
        for c in nb.cells:
            if "solution" in c.metadata.get("tags", []) and not c.metadata.get("jupyter", {}).get("source_hidden"):
                problems.append((rel, "solution cell not collapsed"))
                break
        for m in LINK.finditer(strip_code(text)):
            tgt = m.group(1)
            if re.match(r"^[a-z]+:", tgt) or tgt.startswith("mailto"):
                continue
            if not (p.parent / tgt).resolve().exists():
                problems.append((rel, f"broken link: {tgt}"))
    for md in list(ROOT.glob("*.md")) + list(ROOT.glob("[0-9][0-9]_*/README.md")) + list((ROOT / "docs").glob("*.md")):
        for m in LINK.finditer(strip_code(md.read_text(encoding="utf-8", errors="ignore"))):
            tgt = m.group(1)
            if re.match(r"^[a-z]+:", tgt):
                continue
            if not (md.parent / tgt).resolve().exists():
                problems.append((md.relative_to(ROOT).as_posix(), f"broken link: {tgt}"))
    for mod, files in names.items():
        if not (ROOT / mod / "README.md").exists():
            problems.append((mod, "no README.md"))
    return problems


def run_unsolved(path: str):
    from nbclient import NotebookClient

    p = Path(path)
    nb = nbformat.read(p, as_version=4)
    nb = copy.deepcopy(nb)
    keep, checks = [], []
    for c in nb.cells:
        tags = c.metadata.get("tags", [])
        if "solution" in tags:
            continue
        if c.cell_type == "code" and "no-run" in tags:
            continue
        keep.append(c)
    nb.cells = keep
    # a check cell = first untagged code cell after each starter
    for i, c in enumerate(nb.cells):
        if "starter" in c.metadata.get("tags", []):
            for j in range(i + 1, len(nb.cells)):
                cj = nb.cells[j]
                if cj.cell_type == "code" and not cj.metadata.get("tags"):
                    checks.append(j)
                    break
                if "starter" in cj.metadata.get("tags", []):
                    break
    os.environ.update(MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="2", TF_ENABLE_ONEDNN_OPTS="0",
                      MLFLOW_DISABLE_AGENT_HINT="1", PYTHONIOENCODING="utf-8")
    client = NotebookClient(nb, timeout=400, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": str(p.parent)}})
    client.execute()
    passed_unsolved = []
    for j in checks:
        errs = [o for o in nb.cells[j].get("outputs", []) if o.get("output_type") == "error"]
        if not errs:
            passed_unsolved.append(j)
    return path, len(checks), [nb.cells[j].source.splitlines()[0][:80] if nb.cells[j].source else "" for j in passed_unsolved]


def discriminate(jobs=4):
    files = notebooks()
    bad, total = [], 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_unsolved, str(f)): f for f in files}
        for fu in as_completed(futs):
            try:
                path, n, leaked = fu.result()
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR {futs[fu].name}: {exc}", flush=True)
                continue
            total += n
            rel = Path(path).relative_to(ROOT).as_posix()
            if leaked:
                bad.append((rel, leaked))
                print(f"WEAK  {rel}: {len(leaked)}/{n} checks pass on the unsolved starter", flush=True)
            else:
                print(f"ok    {rel} ({n} checks)", flush=True)
    print(f"\n{total} exercise checks examined; {sum(len(l) for _, l in bad)} do not fail when unsolved")
    return bad


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "structure"
    if mode == "structure":
        probs = structure()
        for rel, msg in probs:
            print(f"{rel}: {msg}")
        print(f"\n{len(probs)} structural problem(s) across {len(notebooks())} notebooks")
        sys.exit(1 if probs else 0)
    else:
        sys.exit(1 if discriminate() else 0)
