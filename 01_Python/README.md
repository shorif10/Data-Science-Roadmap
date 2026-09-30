# 01 Python

**Goal:** write correct, idiomatic Python for data work, and understand the language rules behind the classic bugs (aliasing, mutable defaults, late-binding closures, exhausted generators).

**Prerequisites:** [00 Setup and Tools](../00_Setup_and_Tools/README.md); experience with any programming language.

## Notebooks (about 7 hours)

| # | Notebook | Focus | Hours |
|---|---|---|---|
| 1 | [Syntax, variables, types](01_syntax_variables_types.ipynb) | names vs objects, `is` vs `==`, mutability, truthiness | 0.5 |
| 2 | [Numbers, strings, f-strings](02_numbers_strings_fstrings.ipynb) | floats, `Decimal`, rounding, text, encodings, formatting | 0.5 |
| 3 | [Lists, tuples, sets, dicts](03_lists_tuples_sets_dicts.ipynb) | containers, speed, copies, set algebra | 0.5 |
| 4 | [Control flow](04_control_flow.ipynb) | loops, `for ... else`, `match`, walrus | 0.5 |
| 5 | [Functions](05_functions.ipynb) | parameters, scope, closures, mutable-default trap | 0.5 |
| 6 | [Comprehensions and generators](06_comprehensions_and_generators.ipynb) | lazy pipelines, exhaustion | 0.5 |
| 7 | [Files, JSON, CSV](07_files_json_csv.ipynb) | `with`, `csv`, `json`, JSON Lines, atomic writes | 0.5 |
| 8 | [Errors and debugging](08_errors_and_debugging.ipynb) | exceptions, tracebacks, pdb, logging | 0.5 |
| 9 | [Standard library essentials](09_stdlib_essentials.ipynb) | collections, itertools, datetime, re, functools | 0.75 |
| 10 | [OOP, classes, dataclasses](10_oop_classes_dataclasses.ipynb) | classes, dunder methods, dataclasses | 0.5 |
| 11 | [Iterators, decorators, context managers](11_iterators_decorators_context_managers.ipynb) | protocols and recipes | 0.5 |
| 12 | [Testing, typing, pytest](12_testing_typing_pytest.ipynb) | hints, Pydantic, pytest on a temp package | 0.75 |
| 13 | [Python for data practice](13_python_for_data_practice.ipynb) | stdlib-only revenue analysis verified with SQL | 1 |

## You are done when you can...
- [ ] predict what happens when two names refer to one list, and choose `copy` vs `deepcopy`
- [ ] explain floating-point error and choose between `float`, `Decimal` and integer cents
- [ ] pick the right container and justify it by lookup cost
- [ ] write functions without mutable defaults and explain the late-binding closure trap
- [ ] stream a large CSV with a generator pipeline instead of loading it
- [ ] read a traceback bottom-up and raise/chain custom exceptions
- [ ] write a decorator, a context manager and a dataclass from memory
- [ ] write a pytest file with parametrized cases that catches a planted bug

## Try next
- Python tutorial and standard library reference (docs.python.org/3)
- Exercism's Python track (exercism.org) and Advent of Code (adventofcode.com) for practice problems
- UCI Machine Learning Repository (archive.ics.uci.edu) and Kaggle Datasets (kaggle.com/datasets): redo notebook 13 on a real CSV with the stdlib only
- The scikit-learn bundled datasets (`sklearn.datasets.load_iris`, `load_wine`) as tiny real tables to explore with plain Python
