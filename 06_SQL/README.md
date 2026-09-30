# Module 06: SQL

**Goal:** query, aggregate, join and reshape relational data with confidence, reason about performance, and call SQL from Python, using the synthetic shop database (SQLite) and DuckDB.

## Prerequisites
- Python basics (`01_Python`)
- Helpful: pandas basics (`05_Pandas`); many exercises are verified against a pandas computation

## Notebooks (about 9 to 10 hours in total)
| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | [select_where_orderby_limit](01_select_where_orderby_limit.ipynb) | Filter, sort, limit, `CASE`; dates and NULL behaviour | 1 |
| 02 | [aggregation_group_by_having](02_aggregation_group_by_having.ipynb) | `GROUP BY`, `HAVING` versus `WHERE`, status and channel breakdowns, seasonality | 1 |
| 03 | [joins](03_joins.ipynb) | Inner/left joins, anti-joins, self-join, the fan-out trap and how to beat it | 1.5 |
| 04 | [subqueries_and_ctes](04_subqueries_and_ctes.ipynb) | Scalar and correlated subqueries, `EXISTS`, CTEs, refactoring nested queries | 1 |
| 05 | [window_functions](05_window_functions.ipynb) | `ROW_NUMBER`, `LAG`, running totals, moving averages, concentration | 1.5 |
| 06 | [ddl_dml_and_constraints](06_ddl_dml_and_constraints.ipynb) | `CREATE TABLE`, constraints, `INSERT ... SELECT`, `UPDATE`, transactions | 1 |
| 07 | [indexes_and_query_plans](07_indexes_and_query_plans.ipynb) | `EXPLAIN QUERY PLAN`, single/composite/covering indexes, sargable predicates | 1 |
| 08 | [sql_analytics_patterns](08_sql_analytics_patterns.ipynb) | Top-N per group, retention, year-over-year, streaks, RFM | 1.5 |
| 09 | [sql_from_python](09_sql_from_python.ipynb) | Parameterised queries (no injection), safe dynamic SQL, DuckDB on CSVs, `QUALIFY` | 1 |

## You are done when you can...
1. Write `SELECT` queries with filters, `CASE` bands and correct `NULL` handling, and explain `WHERE` versus `HAVING`.
2. Aggregate by one or more keys and verify the result against an independent pandas computation.
3. Choose between inner, left and anti joins, and demonstrate a fan-out (inflated sums) and fix it by aggregating before joining.
4. Use a CTE or window function to get top-N per group, month-over-month growth, and a cumulative total with a moving average.
5. Find customers with no orders or with a specific order pattern using `NOT EXISTS`/`EXISTS`.
6. Create a table with constraints, backfill it with `INSERT ... SELECT`, and make a multi-statement change atomic with a transaction.
7. Read a query plan, add the right index (composite or covering) and rewrite a predicate so it can use it.
8. Run parameterised queries from Python without string-formatting user input, and query CSV files directly with DuckDB.

## Try next (real public resources)
- SQLite documentation, SQL language reference (sqlite.org/lang.html)
- DuckDB documentation (duckdb.org/docs)
- Markus Winand, *Use The Index, Luke* (use-the-index-luke.com): free online book on SQL indexing
- Kaggle Learn: "Intro to SQL" and "Advanced SQL" (kaggle.com/learn)
- SQL Murder Mystery (mystery.knightlab.com): a puzzle to practise joins and filters
