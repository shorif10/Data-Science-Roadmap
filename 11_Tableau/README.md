# Module 11: Tableau

**Goal:** understand Tableau's data model and calculation order well enough to predict what a view will show, prepare data that behaves, and **prove a dashboard's numbers are right** against independent ground truth.

> **Honest status.** Tableau is a licensed GUI product and was **not run** while writing this module. Nothing here was executed inside Tableau.
> What *is* verified and runnable offline: the data exports, pandas/SQLite computations of every number your views should show, a pandas equivalent for each LOD expression, simulations of filter actions and extract refresh,
> and diagnostics for common mistakes. Tableau calculation syntax and build steps are tagged `no-run`, were written from general knowledge of the product, and **must be verified in the current Tableau documentation**
> (help.tableau.com). Interface names, editions, licences and limits change, so they are described conceptually and marked *verify*; no pricing is claimed. The **local equivalents are the validation tool**:
> build in Tableau, then compare your views with the notebooks' numbers.

## Prerequisites
- pandas basics (`05_Pandas`) and SQL basics (`06_SQL`); `10_PowerBI` helps but is not required
- To follow along in the real tool: Tableau Desktop or Tableau Public. Check current terms yourself, and remember that Tableau Public publishes to the public web: never use confidential data in it.

## Notebooks (about 4 to 5 hours in total)
| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | [tableau_concepts_and_data_prep](01_tableau_concepts_and_data_prep.ipynb) | Dimension/measure versus discrete/continuous, relationships versus joins (fan-out), live versus extract, tidy exports to `exports/` | 1 |
| 02 | [calculated_fields_and_lod_expressions](02_calculated_fields_and_lod_expressions.ipynb) | Row, aggregate and table calculations; `FIXED`/`INCLUDE`/`EXCLUDE` with pandas twins; order of operations; the grain trap | 1.5 |
| 03 | [charts_dashboards_and_interactivity](03_charts_dashboards_and_interactivity.ipynb) | Numbers behind each chart (Pareto, YoY, scatter, percent of row, bins, Top N + Other); filter actions simulated; design checklist | 1 |
| 04 | [storytelling_project](04_storytelling_project.ipynb) | "Customer & Revenue" dashboard spec, acceptance targets, mistake signatures, story outline, review rubric | 1.5 |

Files written to `exports/` (load these into Tableau): `order_lines_flat.csv`, `customers.csv`, `products.csv`, `orders.csv`, `order_items.csv`, `customers_dashboard_safe.csv` (no names); acceptance targets: `acceptance_targets_customer_revenue_kpis.csv`, `acceptance_targets_customer_revenue_visuals.csv`.

## You are done when you can...
1. Classify fields as dimension or measure, discrete or continuous, and explain why an id column needs re-classifying.
2. Show with numbers how a one-to-many join inflates a count and how a distinct count fixes it.
3. Decide live versus extract and describe what an incremental refresh would add.
4. Write `FIXED`, `INCLUDE` and `EXCLUDE` calculations and reproduce each with `groupby`/`transform` in pandas.
5. Explain why a dimension filter does not change a `FIXED` value, and what a context filter changes.
6. Compute the numbers behind a Pareto, a year-over-year line and a Top N + Other view, and check that the pieces add up.
7. Explain why distinct counts are not additive across categories, and reconcile a dashboard in several filter states.
8. Recognise a wrong KPI from its size (lines instead of orders, cancelled orders included, average of line values) and score a dashboard with a rubric.

## Try next (real public resources)
- Tableau Help (help.tableau.com): calculations, LOD expressions, order of operations, dashboards
- Tableau Public (public.tableau.com): browse published workbooks to see how others structure dashboards (check the current terms)
- Cole Nussbaumer Knaflic, *Storytelling with Data*
- Kaggle Datasets (kaggle.com/datasets) and the UCI Machine Learning Repository (archive.ics.uci.edu): rebuild notebook 04's dashboard on a real table
