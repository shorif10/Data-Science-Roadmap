# Module 05: Pandas

**Goal:** load, clean, reshape, combine and summarise tabular data with pandas 3, and finish with an exploratory analysis that you can defend with numbers.

## Prerequisites
- Python basics (`01_Python`), especially lists, dicts, functions and comprehensions
- NumPy arrays and vectorisation (`04_NumPy`)

## Notebooks (about 9 to 10 hours in total)
| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | [loading_and_inspecting_data](01_loading_and_inspecting_data.ipynb) | `read_csv`, `shape`/`dtypes`/`info`, missing values, `[]`, `.loc`, `.iloc`; write a data-quality report | 1 |
| 02 | [cleaning_data](02_cleaning_data.ipynb) | Missing values and the parser trap, duplicates, types, text and date cleaning, imputation with a flag, a cleaning log | 1 |
| 03 | [transforming_columns](03_transforming_columns.ipynb) | New features: bands with `cut`/`qcut`, mapping, string and date parts, a risk score | 1 |
| 04 | [groupby_and_aggregation](04_groupby_and_aggregation.ipynb) | `groupby`, named aggregation, `transform`, minimum group sizes, verifying against an independent method | 1.5 |
| 05 | [merge_join_concat](05_merge_join_concat.ipynb) | Merge types, key validation, the row-explosion trap, anti-joins, safe appends | 1.5 |
| 06 | [reshaping_and_window_functions](06_reshaping_and_window_functions.ipynb) | `pivot_table`, `melt`, rolling windows, shifts, year-over-year | 1 |
| 07 | [time_series_in_pandas](07_time_series_in_pandas.ipynb) | `DatetimeIndex`, resampling, limited gap filling, weekday effects, thresholds | 1 |
| 08 | [pandas_performance_and_best_practices](08_pandas_performance_and_best_practices.ipynb) | dtypes and memory, vectorising, method chains, copy-on-write, file formats | 1 |
| 09 | [eda_project_ecommerce](09_eda_project_ecommerce.ipynb) | End-to-end EDA on the shop: monthly table, cohorts, concentration, five defended findings | 1.5 |

## You are done when you can...
1. Load a CSV, and say how many rows, columns, missing values and suspicious values it has, including cases where the parser itself creates the missing values.
2. Clean a messy table (duplicates, mixed formats, junk values) and keep a log of every change and how many rows it affected.
3. Create features with `cut`/`qcut`, `map`, string and date accessors without writing row-by-row loops.
4. Compute group summaries with `groupby`, add a per-group comparison column with `transform`, and exclude groups that are too small to trust.
5. Merge tables with `validate=`, predict the row count before running, and detect a fan-out or orphan keys.
6. Reshape between long and wide with `pivot_table` and `melt`, and compute rolling averages and year-over-year changes.
7. Resample a time series, fill short gaps without inventing long ones, and quantify a weekday effect.
8. Reduce a DataFrame's memory with dtypes, replace a slow `apply` with a vectorised version, and explain what copy-on-write does to your code.

## Try next (real public resources)
- pandas documentation, "10 minutes to pandas" and the User Guide (pandas.pydata.org)
- Wes McKinney, *Python for Data Analysis* (the author publishes the 3rd edition free online at wesmckinney.com/book)
- Kaggle Learn: "Pandas" micro-course (kaggle.com/learn)
- Kaggle Datasets (kaggle.com/datasets) and the UCI Machine Learning Repository (archive.ics.uci.edu): rerun notebook 09's approach on a real table
