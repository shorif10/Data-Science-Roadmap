"""
Build .ipynb notebooks from plain-text sources.

    python tools/build_notebooks.py                 # build everything
    python tools/build_notebooks.py 04_NumPy        # build one module
    python tools/build_notebooks.py 04_NumPy/01_arrays.py

Sources live in sources/<module>/<name>.py and use the "percent" format:

    # %% [markdown]
    # # Title            <- markdown cell: every line starts with "# "
    # Some *text*.

    # %%
    x = 1 + 1           <- code cell

    # %% tags=["starter"]      <- exercise starter (learner edits this)
    # %% tags=["solution"]     <- hidden solution (collapsed in JupyterLab)
    # %% tags=["no-run"]       <- shown to the learner, skipped by the checker
                                 (needs network / API key / cloud account / GUI)

Output goes to <module>/<name>.ipynb, next to the data-relative paths the notebooks expect.
Also regenerates docs/INDEX.md (the table of contents / progress tracker).
"""
from __future__ import annotations

import ast
import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources"

CELL_RE = re.compile(r"^# %%(?P<rest>.*)$")


def parse_source(text: str, name: str = "<source>"):
    """Return a list of (cell_type, source_text, tags)."""
    cells = []
    cur = None  # [type, lines, tags]
    for lineno, line in enumerate(text.splitlines(), 1):
        m = CELL_RE.match(line)
        if m:
            if cur:
                cells.append(cur)
            rest = m.group("rest").strip()
            ctype = "markdown" if rest.startswith("[markdown]") else "code"
            tags = []
            tm = re.search(r"tags=(\[[^\]]*\])", rest)
            if tm:
                try:
                    tags = ast.literal_eval(tm.group(1))
                except Exception as exc:  # noqa: BLE001
                    raise SystemExit(f"{name}:{lineno}: bad tags {tm.group(1)!r}: {exc}")
            cur = [ctype, [], tags]
        elif cur is None:
            if line.strip():
                raise SystemExit(f"{name}:{lineno}: content before the first '# %%' marker")
        else:
            cur[1].append(line)
    if cur:
        cells.append(cur)

    out = []
    for ctype, lines, tags in cells:
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if ctype == "markdown":
            lines = [re.sub(r"^# ?", "", ln, count=1) for ln in lines]
        if lines:
            out.append((ctype, "\n".join(lines), tags))
    return out


def _lines(src: str):
    parts = src.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build_notebook(cells) -> dict:
    nb_cells = []
    for ctype, src, tags in cells:
        cell = {"id": uuid.uuid4().hex[:8], "cell_type": ctype, "metadata": {}, "source": _lines(src)}
        if tags:
            cell["metadata"]["tags"] = list(tags)
        if "solution" in tags:
            cell["metadata"]["jupyter"] = {"source_hidden": True}
        if ctype == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": nb_cells,
    }


def validate_source(cells, name: str):
    problems = []
    if not cells or cells[0][0] != "markdown" or not cells[0][1].lstrip().startswith("# "):
        problems.append("first cell must be a markdown cell starting with an H1 title")
    starters = sum("starter" in t for _, _, t in cells)
    solutions = sum("solution" in t for _, _, t in cells)
    if starters != solutions:
        problems.append(f"{starters} starter cells but {solutions} solution cells (must pair up)")
    for ctype, src, tags in cells:
        if ctype == "code" and "no-run" not in tags:
            try:
                ast.parse(re.sub(r"^\s*[!%].*$", "pass", src, flags=re.M))
            except SyntaxError as exc:
                problems.append(f"syntax error in code cell: {exc.msg} (line {exc.lineno}): {src.splitlines()[max((exc.lineno or 1) - 1, 0)][:70]!r}")
    return problems


def title_of(cells) -> str:
    return re.sub(r"^#\s+", "", cells[0][1].splitlines()[0]).strip()


def build_one(src_path: Path) -> tuple[Path, list[str], str]:
    rel = src_path.relative_to(SOURCES)
    cells = parse_source(src_path.read_text(encoding="utf-8"), str(rel))
    problems = validate_source(cells, str(rel))
    out = ROOT / rel.with_suffix(".ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out, problems, title_of(cells) if cells else rel.stem


def write_index():
    lines = [
        "# Roadmap Index & Progress Tracker",
        "",
        "Tick a box when you have **read it, run it, and solved the exercises** without peeking at the solutions.",
        "",
    ]
    for mod in sorted(p for p in SOURCES.iterdir() if p.is_dir()):
        srcs = sorted(mod.glob("*.py"))
        if not srcs:
            continue
        lines += [f"## {mod.name.replace('_', ' ')}", ""]
        for s in srcs:
            cells = parse_source(s.read_text(encoding="utf-8"), s.name)
            if not cells:
                continue
            lines.append(f"- [ ] [{title_of(cells)}](../{mod.name}/{s.stem}.ipynb)")
        lines.append("")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv):
    targets = []
    if argv:
        for a in argv:
            p = SOURCES / a
            if p.is_dir():
                targets += sorted(p.glob("*.py"))
            elif p.is_file():
                targets.append(p)
            else:
                raise SystemExit(f"not found under sources/: {a}")
    else:
        targets = sorted(SOURCES.glob("*/*.py"))
    bad = 0
    for t in targets:
        out, problems, _ = build_one(t)
        status = "OK " if not problems else "BAD"
        print(f"{status} {out.relative_to(ROOT)}")
        for pr in problems:
            bad += 1
            print(f"     - {pr}")
    write_index()
    print(f"\nBuilt {len(targets)} notebooks, {bad} problem(s).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
