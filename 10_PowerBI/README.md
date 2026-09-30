# Module 10: Power BI

**Goal:** design a correct data model, write core DAX measures, build a report to a spec, and **prove its numbers are right** against independent ground truth.

> **Honest status.** Power BI Desktop and the Service are GUI/cloud products and were **not run** while writing this module. Nothing here was executed inside Power BI.
> What *is* verified and runnable offline: the data exports, pandas/DuckDB computations of every number your report should show, a small filter-context engine that mimics how
> DAX measures evaluate, and diagnostics for common mistakes. All DAX and Power Query M code is tagged `no-run`, was written from documented semantics, and should be checked in the
> current Microsoft documentation. Interface names, licences and limits change, so they are described conceptually and marked *verify*. The **local equivalents are the validation tool**: build in Power BI, then compare your visuals with the notebooks' numbers.

## Prerequisites
- pandas basics (`05_Pandas`) and SQL basics (`06_SQL`)
- Power BI Desktop (free, Windows) to follow along in the real tool. Check Microsoft's current system requirements and terms yourself.

## Notebooks (about 5 to 6 hours in total)
| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | [power_bi_overview_and_data_model](01_power_bi_overview_and_data_model.ipynb) | Star schema for the shop; export tables to `exports/`; relationships, cardinality, filter direction, orphan and duplicate-key bugs | 1 |
| 02 | [power_query_and_data_prep](02_power_query_and_data_prep.ipynb) | Power Query steps mirrored in pandas; clean the messy file with a row-count audit; merges, pivots, query folding idea | 1 |
| 03 | [dax_fundamentals](03_dax_fundamentals.ipynb) | Measures vs columns, row vs filter context, `CALCULATE`, `DIVIDE`, `SUMX`, `ALL`/`ALLEXCEPT`, `VAR`; expected KPI values | 1.5 |
| 04 | [time_intelligence_and_visual_design](04_time_intelligence_and_visual_design.ipynb) | Date table contract, YTD/MoM/YoY/rolling, traps, chart choice, design rules | 1.5 |
| 05 | [dashboard_project_and_publishing](05_dashboard_project_and_publishing.ipynb) | "Sales Performance" spec, 42 acceptance numbers, mistake signatures, publishing, refresh, RLS | 1.5 |

Files written to `exports/` (load these into Power BI): `fact_sales.csv`, `dim_customer.csv`, `dim_product.csv`, `dim_date.csv`; acceptance targets: `acceptance_targets_kpis.csv`, `acceptance_targets_visuals.csv`.

## You are done when you can...
1. Draw a star schema for a new dataset, state its grain, and choose relationship cardinality and direction.
2. Detect orphan keys and duplicate dimension keys before loading, and explain the (Blank) category.
3. Write Power Query-style cleaning steps and audit row and null counts after each.
4. Explain row context vs filter context and predict what `CALCULATE` does to an existing filter.
5. Write Net Revenue, Margin %, AOV, Premium Share and revenue-by-month measures and match the expected values in notebook 03.
6. Build a correct date table and YTD/MoM/YoY measures, and avoid the future-month and incomplete-year traps.
7. Choose a chart by question and field cardinality, and apply a pre-publish checklist.
8. Reconcile a report to ground truth for several slicer states and test row-level security as a partition of the total.

## Try next (real public resources)
- Microsoft Learn: Power BI documentation and learning paths (learn.microsoft.com/power-bi)
- DAX reference on Microsoft Learn (learn.microsoft.com/dax)
- Kaggle Datasets (kaggle.com/datasets): pick a sales or retail CSV and rebuild this project on it
- UCI Machine Learning Repository (archive.ics.uci.edu): a real table to model and dashboard
