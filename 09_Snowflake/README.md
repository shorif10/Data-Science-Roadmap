# Module 09: Snowflake

**Goal:** understand how Snowflake works well enough to use it on a data team, predict its cost, and write pipeline code that can be tested without an account.

> **Honest status.** Snowflake is a paid cloud service and it was **not available** when these notebooks were written. Nothing in the module was run against a real Snowflake account.
> Every cell that needs Snowflake (SQL, connector, Snowpark) is tagged `no-run`, was written from the documented behaviour, and should be treated as *verify in the current docs*.
> UIs, editions, defaults, credit rates and prices change; prices in the notebooks are labelled placeholders you edit. What **is** verified: toy models of the concepts, portable SQL on DuckDB,
> a DuckDB-backed fake connection, and Parquet/CSV preparation and validation with `pyarrow`. All of that executes offline.

## Prerequisites
- SQL basics (joins, aggregation, window functions): module `06_SQL`
- pandas basics: module `05_Pandas`
- Optional: a Snowflake trial or company account to try the `no-run` cells. Check the vendor's current trial terms yourself.

## Notebooks (about 4 to 5 hours in total)
| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | [snowflake_architecture_and_concepts](01_snowflake_architecture_and_concepts.ipynb) | Storage/compute separation, micro-partitions, Time Travel and clone via runnable toy models; when not to use it | 1 |
| 02 | [snowflake_sql_and_dialect_differences](02_snowflake_sql_and_dialect_differences.ipynb) | `QUALIFY`, JSON with `FLATTEN`, `MERGE`, dialect differences; portable parts on DuckDB | 1 |
| 03 | [python_connector_and_snowpark](03_python_connector_and_snowpark.ipynb) | Credentials from the environment, connector and Snowpark patterns, a fake connection for testing | 1 |
| 04 | [loading_data_stages_copy_into_and_formats](04_loading_data_stages_copy_into_and_formats.ipynb) | Stages, `COPY INTO`, Parquet vs CSV; produces upload-ready files in `exports/` and control totals | 1 |
| 05 | [cost_performance_and_security_basics](05_cost_performance_and_security_basics.ipynb) | Editable cost calculator, auto-suspend, sizing, clustering, RBAC, resource monitors | 1 |

Running notebook 04 writes `exports/order_lines.{csv,parquet,ndjson}` in this folder. Upload them to your own Snowflake stage and compare your totals with the notebook's control totals.

## You are done when you can...
1. Explain storage/compute separation, micro-partitions and pruning, and say why physical order matters for filters.
2. Explain Time Travel and zero-copy clone as consequences of immutable partitions, and name their limits.
3. Write `QUALIFY`, `MERGE` and a `FLATTEN` query, and validate each against a pandas ground truth.
4. Load credentials from environment variables and structure pipeline code so it can be tested with a fake connection.
5. Choose a file format and file size for a bulk load, and reconcile row counts and sums after `COPY INTO`.
6. Estimate warehouse cost from size, run time and auto-suspend, and explain when a bigger warehouse is not cheaper.
7. Design roles with least privilege and describe what a resource monitor does and does not do.

## Try next (real public resources)
- Snowflake documentation and its official quickstart guides: docs.snowflake.com
- DuckDB documentation for local SQL practice with the same style of queries: duckdb.org/docs
- UCI Machine Learning Repository (archive.ics.uci.edu): real tables to practise loading and validating
- Kaggle Datasets (kaggle.com/datasets): pick a CSV, convert it to Parquet with `pyarrow`, and write its control totals
- Apache Arrow / Parquet documentation: arrow.apache.org/docs/python
