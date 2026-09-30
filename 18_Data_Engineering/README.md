# 18 Data Engineering

**Goal.** Understand how data gets from source systems to an analyst's table, and build the pieces yourself: choose file formats, model data for analytics, check its quality, ingest from an API without losing or duplicating rows, and query it fast with DuckDB. Every notebook runs offline on the synthetic shop and churn datasets.

**Prerequisites.** Python, pandas and basic SQL (modules 01, 05, 06). Comfort with the terminal helps.

## Notebooks (about 6.5 hours in total)

| # | Notebook | What you do | Time |
|---|---|---|---|
| 1 | [01_data_engineering_landscape](01_data_engineering_landscape.ipynb) | ETL vs ELT, batch vs streaming, OLTP vs OLAP, row vs column storage measured | 60 min |
| 2 | [02_file_formats_csv_json_parquet](02_file_formats_csv_json_parquet.ipynb) | Compare sizes and read times, types that survive Parquet, partitioning, schema evolution | 60 min |
| 3 | [03_data_modeling_star_schema_and_normalization](03_data_modeling_star_schema_and_normalization.ipynb) | Find functional dependencies, normalise to 3NF, build a star schema, slowly changing dimensions | 75 min |
| 4 | [04_data_quality_and_validation](04_data_quality_and_validation.ipynb) | A mini expectations framework, row-level pydantic validation, referential integrity, freshness | 70 min |
| 5 | [05_apis_and_ingestion](05_apis_and_ingestion.ipynb) | Pagination, retries with backoff and jitter, rate limits, idempotent ingestion against a fake API | 70 min |
| 6 | [06_duckdb_analytics](06_duckdb_analytics.ipynb) | SQL over CSV/Parquet/DataFrames, joins, window functions, DuckDB vs pandas measured | 60 min |

Run them in order from JupyterLab; each notebook uses its own folder as working directory. Continue with [19 Data Pipelines](../19_Data_Pipelines/README.md).

## You are done when you can...

- [ ] explain ETL vs ELT and batch vs streaming, and pick one for a concrete requirement
- [ ] choose between CSV, JSON and Parquet and justify it with size, speed, typing and schema evolution
- [ ] spot update, insert and delete anomalies in a flat table and normalise it to 3NF
- [ ] design a star schema (one fact table, conformed dimensions) and answer a business question with it
- [ ] explain Type 1 vs Type 2 slowly changing dimensions and say when each is needed
- [ ] write data-quality checks (completeness, uniqueness, validity, referential integrity, freshness) and decide what to do when one fails
- [ ] ingest a paginated API with retries, backoff and rate limiting without duplicating rows on re-run
- [ ] query files in place with DuckDB and say when it beats pandas (and when it does not)

## Try next (real resources)

- *Designing Data-Intensive Applications*, Martin Kleppmann (O'Reilly)
- *Fundamentals of Data Engineering*, Joe Reis and Matt Housley (O'Reilly)
- *The Data Warehouse Toolkit*, Ralph Kimball and Margy Ross (Wiley), the reference for dimensional modelling
- DuckDB documentation (duckdb.org/docs) and Apache Parquet documentation (parquet.apache.org)
- Public datasets to ingest and model yourself: the NYC Taxi trip records (published by the NYC Taxi and Limousine Commission), and any dataset on Kaggle or the UCI repository
