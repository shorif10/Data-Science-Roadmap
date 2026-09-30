# Module 07: Matplotlib

**Goal:** build clear, correctly labelled charts with the object-oriented API, choose the right chart for the question, assemble multi-panel figures, and export publication-quality files.

## Prerequisites
- Python basics (`01_Python`), NumPy (`04_NumPy`) and pandas (`05_Pandas`) for the data handling

## Notebooks (about 5 to 6 hours in total)
| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | [anatomy_figure_axes](01_anatomy_figure_axes.ipynb) | Figure versus Axes, `plt.subplots`, the OO API, a reusable plotting function, mark the average | 1 |
| 02 | [core_chart_types_and_when_to_use_them](02_core_chart_types_and_when_to_use_them.ipynb) | Bar, line, histogram, scatter, box: pick the chart by question; fix a bad chart | 1 |
| 03 | [subplots_layout_annotations_styling](03_subplots_layout_annotations_styling.ipynb) | Shared scales, annotations, shaded spans, marginal histogram, a house style | 1.5 |
| 04 | [time_series_and_multi_panel_dashboards](04_time_series_and_multi_panel_dashboards.ipynb) | Trend over noise, anomalies on top, panels sharing one time axis, seasonality | 1.5 |
| 05 | [saving_and_publication_quality](05_saving_and_publication_quality.ipynb) | DPI and size, vector exports, colour-blind-safe lines, a reliable export helper | 1 |

## You are done when you can...
1. Explain the difference between a `Figure` and an `Axes`, and build a chart entirely with `fig, ax = plt.subplots()`.
2. Wrap a chart in a function that takes an `Axes`, so it can be reused in any layout.
3. Choose between bar, line, histogram, scatter and box for a stated question, and justify the choice.
4. Build a grid of panels with a shared scale and annotate a specific point on a plot.
5. Shade a date range (for example a holiday season) and overlay a trend on a noisy series.
6. Stack panels that share one time axis and mark detected anomalies on the top panel.
7. Export a figure at the right size and DPI as PNG and as a vector format, and check the file afterwards.
8. Make lines distinguishable without relying on colour alone.

## Try next (real public resources)
- Matplotlib documentation: Tutorials and the example gallery (matplotlib.org), including the "Anatomy of a figure" example
- Claus Wilke, *Fundamentals of Data Visualization* (free online: clauswilke.com/dataviz)
- Kaggle Learn: "Data Visualization" micro-course (kaggle.com/learn)
- Kaggle Datasets (kaggle.com/datasets): redraw a chart from a public report using your own data
