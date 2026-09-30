# 19 Data Pipelines

**Goal.** Build pipelines that survive real life: re-runs, partial failures, late and duplicated data, backfills, schema changes, and silent data problems. You design for idempotency, write an ETL in plain Python, model orchestration with a DAG runner, load incrementally, and test and monitor the result. Airflow, dbt and other tools that are not installed here appear as correct example code next to a runnable local analogue that teaches the same idea.

**Prerequisites.** [18 Data Engineering](../18_Data_Engineering/README.md) (especially data quality and ingestion), Python, SQL, and basic `pytest` (module 01, notebook 12).

## Notebooks (about 6 hours in total)

| # | Notebook | What you do | Time |
|---|---|---|---|
| 1 | [01_pipeline_design_and_idempotency](01_pipeline_design_and_idempotency.ipynb) | Pipelines as DAGs, idempotent and atomic writes, late data with a lookback window, schema drift | 70 min |
| 2 | [02_build_an_etl_pipeline_in_python](02_build_an_etl_pipeline_in_python.ipynb) | Extract, transform, load, with config, logging, validation and a run report | 75 min |
| 3 | [03_orchestration_and_dags](03_orchestration_and_dags.ipynb) | A topological DAG runner with retries and resume; an Airflow DAG file shown and checked with `ast` | 80 min |
| 4 | [04_incremental_loads_backfills_and_cdc](04_incremental_loads_backfills_and_cdc.ipynb) | Watermarks and their holes, guarded upserts, trigger-based CDC, partition backfills, effectively-once delivery (SQLite) | 75 min |
| 5 | [05_testing_monitoring_and_alerting_pipelines](05_testing_monitoring_and_alerting_pipelines.ipynb) | `pytest` on a generated package, data contracts, robust volume/freshness/null monitors, alert cooldown, SLOs | 80 min |

Run them in order from JupyterLab. The next module, [20 MLOps](../20_MLOps/README.md), builds on all of this.

## You are done when you can...

- [ ] take a pipeline step and make it idempotent (partition overwrite or keyed upsert) and atomic (temp file plus rename, or one transaction)
- [ ] explain why a failed run must be safe to restart, and demonstrate it by running a step twice
- [ ] write an ETL with separated extract/transform/load functions, validation before load, and a run report
- [ ] describe a DAG, detect a cycle, compute a valid order, and explain retries, `upstream_failed` and the logical date
- [ ] read an Airflow DAG file and say what runs when, and why the top level of a DAG file must stay cheap
- [ ] implement a watermark load, explain how it can lose rows, and fix it with a lookback and an idempotent merge that refuses stale versions
- [ ] explain what a watermark cannot see (deletes) and how CDC or soft deletes cover it
- [ ] backfill a date range partition by partition, repeatably
- [ ] explain at-least-once, at-most-once and effectively-once, and build the last from the first
- [ ] unit-test transformations, write a data contract and diff two versions, and set a volume monitor from a rolling baseline with a known false-alarm rate
- [ ] design alerts with severity, de-duplication and cooldown, and compute an SLO and error budget from run history

## Try next (real resources)

- Apache Airflow documentation (airflow.apache.org/docs): Core Concepts, and the TaskFlow tutorial
- dbt documentation (docs.getdbt.com): models, tests and incremental models, the SQL-first way to do transformation
- Great Expectations documentation (docs.greatexpectations.io): a production data-validation framework
- *Site Reliability Engineering* (Google, free at sre.google/sre-book): the chapters on service level objectives
- *Designing Data-Intensive Applications*, Martin Kleppmann: the batch and stream processing chapters
