# Module 08: Seaborn

**Goal:** explore distributions, relationships and group differences quickly with seaborn (with honest uncertainty), scan many variables at once with matrix and grid plots, and turn one messy finding into one clear chart.

## Prerequisites
- Matplotlib basics (`07_Matplotlib`) and pandas (`05_Pandas`)

## Notebooks (about 4 to 5 hours in total)
| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | [distributions](01_distributions.ipynb) | Histograms, KDE, ECDF, box/violin; shares versus counts; small multiples | 1 |
| 02 | [relationships_and_categorical_plots](02_relationships_and_categorical_plots.ipynb) | Scatter with hue, regression lines, bar plots with confidence intervals, faceted rates | 1.5 |
| 03 | [matrix_and_grid_plots](03_matrix_and_grid_plots.ipynb) | Masked correlation heatmap, pivot heatmaps, `pairplot`, `PairGrid`, `FacetGrid` | 1 |
| 04 | [visual_storytelling](04_visual_storytelling.ipynb) | Before/after redraws: finding first, grey-plus-accent colour, sorted and labelled bars, intervals and `n`; choose the chart by the question | 1 |

## You are done when you can...
1. Choose between histogram, KDE, ECDF and box/violin for a distribution question, and compare groups on shares rather than raw counts.
2. Draw a scatter with a colour grouping and a regression line, and say what the fit does and does not show.
3. Explain what the error bars on a seaborn bar plot are (a bootstrapped confidence interval by default) and use them to judge whether groups differ.
4. Draw a correlation heatmap with the upper triangle masked, a diverging palette centred at 0 and `vmin=-1, vmax=1`, and name the strongest pair from the matrix in code.
5. Build a pivot-table heatmap of rates and show the group sizes behind it.
6. Produce a sampled `pairplot` and a `FacetGrid` with a shared reference line.
7. Redraw a cluttered chart so its title states the finding, one accent colour marks the evidence, categories are sorted and bars are labelled.
8. Pick a chart from the question (compare, trend, distribution, relationship, composition) and audit a chart for missing titles, labels and overloaded legends.

## Try next (real public resources)
- seaborn documentation: the User guide and tutorial (seaborn.pydata.org)
- Claus Wilke, *Fundamentals of Data Visualization* (free online: clauswilke.com/dataviz)
- Kaggle Learn: "Data Visualization" micro-course (kaggle.com/learn)
- UCI Machine Learning Repository (archive.ics.uci.edu): pick a table and write a finding-first chart
