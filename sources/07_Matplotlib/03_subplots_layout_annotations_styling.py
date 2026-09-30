# %% [markdown]
# # Subplots, Layout, Annotations and Styling
#
# **Why this matters.** A single chart rarely tells the whole story: you compare segments side by side, add a note at the peak, shade a promotion period, and make everything match your company's look.
# These are the touches that separate a chart people *understand in five seconds* from a chart people politely ignore. They are also where beginners lose hours fighting overlapping labels.
#
# **You will learn**
# - build grids of panels with `plt.subplots`, share axes, and switch off unused panels
# - make **small multiples** (the same chart repeated per group) with identical scales so panels are comparable
# - build irregular layouts with `subplot_mosaic` (for example a scatter with a marginal histogram)
# - annotate data: text, arrows, shaded periods, reference lines
# - restyle consistently: style sheets, `rcParams`, colours, spines, legend placement
#
# **Prerequisites:** [Core chart types](02_core_chart_types_and_when_to_use_them.ipynb).
#
# > Data used: `datasets/orders.csv` and `datasets/housing.csv` (synthetic, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.max_open_warning"] = 0
DATA = Path("../datasets")

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
monthly = orders.groupby(orders.order_date.dt.to_period("M").dt.to_timestamp()).size()

housing = pd.read_csv(DATA / "housing.csv")
homes = housing[(housing.price / housing.area_sqft).between(20, 500)].copy()   # drop the implausible price-per-sqft listings (see previous notebook)
print(len(monthly), "months of orders;", len(homes), "plausible listings")

# %% [markdown]
# ## 1. Grids of panels
# `plt.subplots(nrows, ncols)` returns the figure and a NumPy array of Axes. `sharex`/`sharey` link the axes so panels use the **same scale** (essential when the point is comparison), and hide redundant tick labels.

# %%
fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True, layout="constrained")
print("axs is a", type(axs).__name__, "with shape", axs.shape)
for ax, (channel, grp) in zip(axs.flat, orders.groupby("channel")):        # only 3 channels: the 4th panel stays empty
    m = grp.groupby(grp.order_date.dt.to_period("M").dt.to_timestamp()).size()
    ax.plot(m.index, m.values)
    ax.set_title(channel)
axs[1, 1].set_visible(False)                                              # hide the unused panel
fig.supxlabel("Month")
fig.supylabel("Orders (count)")
fig.suptitle("Orders per month by channel (same y-scale in every panel)")
plt.show()

# %% [markdown]
# *What just happened?* `axs.flat` iterates the 2-D array as a flat sequence, `zip` pairs each panel with one channel. Sharing y makes the height of a peak comparable across panels; **without** it, each panel would auto-scale and a tiny channel would look as busy as a big one.
# `fig.supxlabel/supylabel/suptitle` label the whole figure once instead of repeating labels on every panel.
#
# **Layout managers.** Without help, labels collide. `layout="constrained"` (recommended) re-arranges margins automatically; `fig.tight_layout()` is the older alternative. Compare:

# %%
fig, axs = plt.subplots(1, 2, figsize=(7, 2.6))                    # no layout manager
for ax in axs:
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set(title="A rather long title that wants space", xlabel="An x label that is long", ylabel="A long y label (units)")
plt.show()

# %%
fig, axs = plt.subplots(1, 2, figsize=(7, 2.6), layout="constrained")    # same content, constrained layout
for ax in axs:
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set(title="A rather long title that wants space", xlabel="An x label that is long", ylabel="A long y label (units)")
plt.show()

# %% [markdown]
# ## 2. Small multiples
# **Small multiples** repeat one simple chart for each group, on **identical axes**, so the eye compares shapes instead of decoding a legend with six colours. It is the single most effective upgrade for a cluttered multi-series plot.

# %%
hoods = sorted(homes.neighborhood.unique())
fig, axs = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True, sharey=True, layout="constrained")
for ax, hood in zip(axs.flat, hoods):
    d = homes[homes.neighborhood == hood]
    ax.scatter(d.area_sqft, d.price / 1000, s=6, alpha=0.3)
    ax.set_title(f"{hood} (n={len(d)})")
fig.supxlabel("Living area (sq ft)")
fig.supylabel("Price (USD thousands)")
fig.suptitle("Price vs area: the slope differs by neighbourhood")
plt.show()

# %% [markdown]
# Because all six panels share both axes, a steeper cloud really is a steeper relationship and a higher cloud really is more expensive; the title carries the group size so nobody over-reads a thin panel.

# %% [markdown]
# ## 3. Irregular layouts with `subplot_mosaic`
# Describe the layout as a picture: each label is a panel, repeating a label makes it span cells. Great for a main chart plus marginal distributions.

# %%
fig = plt.figure(figsize=(8, 4.5), layout="constrained")
axd = fig.subplot_mosaic([["main", "main", "side"],
                          ["main", "main", "side"]], gridspec_kw={"width_ratios": [3, 3, 1]}, sharey=True)
axd["main"].scatter(homes.area_sqft, homes.price / 1000, s=6, alpha=0.3)
axd["main"].set(title="Price vs area", xlabel="Living area (sq ft)", ylabel="Price (USD thousands)")
axd["side"].hist(homes.price / 1000, bins=40, orientation="horizontal")
axd["side"].set(title="Price distribution", xlabel="Listings (count)")
axd["side"].tick_params(labelleft=False)
plt.show()
print("panels:", list(axd))

# %% [markdown]
# `sharey=True` makes the marginal histogram line up with the scatter's y-axis, so a horizontal bar sits exactly beside the prices it counts. Panels are accessed by **name** (`axd["main"]`), which reads better than `axs[0, 1]`.

# %% [markdown]
# ## 4. Annotations: tell the reader what to notice
# A chart shows data; an **annotation** shows the point. Three tools:
# - `ax.annotate(text, xy=point, xytext=label_position, arrowprops=...)`: a label with an arrow to a data point
# - `ax.axvspan(x0, x1)` / `ax.axhspan(y0, y1)`: shade a period or a band
# - `ax.axhline(y)` / `ax.axvline(x)`: reference lines (a target, an average)

# %%
peak = monthly.idxmax()
fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
ax.plot(monthly.index, monthly.values, marker="o", markersize=3)
for year in sorted(monthly.index.year.unique()):
    start, end = pd.Timestamp(f"{year}-11-01"), pd.Timestamp(f"{year}-12-31")
    if start <= monthly.index.max():
        ax.axvspan(start, end, color="orange", alpha=0.2, label="Nov-Dec season" if year == monthly.index.year.min() else None)
ax.annotate(f"Peak: {peak:%b %Y}\n{monthly.max()} orders", xy=(peak, monthly.max()),
            xytext=(peak - pd.Timedelta(days=200), monthly.max() * 0.93),
            arrowprops={"arrowstyle": "->"}, ha="center")
ax.axhline(monthly.mean(), linestyle="--", color="gray", linewidth=1)
ax.text(monthly.index[0], monthly.mean(), " monthly average", va="bottom", color="gray", fontsize=8)
ax.set(title="Orders peak every Nov-Dec", xlabel="Month", ylabel="Orders (count)")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# Notice the title states the **finding** ("Orders peak every Nov-Dec"), not just the topic ("Orders per month"): a headline that tells the reader what to see. The shaded bands share **one legend entry** (we only labelled the first).
# Every annotation is an Artist you can inspect: `ax.texts` holds the note and the average label, `ax.patches` holds the shaded rectangles.

# %%
print("texts  :", [t.get_text().replace("\n", " / ") for t in ax.texts])
print("patches:", len(ax.patches), "shaded periods")
print("annotation has an arrow:", ax.texts[0].arrow_patch is not None)

# %% [markdown]
# Restraint matters: if you annotate everything, nothing stands out. One or two annotations per chart, placed away from the data they describe.

# %% [markdown]
# ## 5. Styling
# Three levels, from broad to narrow:
# 1. **Style sheets**: a bundle of defaults, applied globally with `plt.style.use(...)` or temporarily with `with plt.style.context(...)`. They ship with Matplotlib (no download).
# 2. **`rcParams`**: individual defaults (font size, line width, colour cycle).
# 3. **Per-artist**: `ax.plot(color=..., linewidth=...)`, `ax.spines[...]`.

# %%
print(sorted(s for s in plt.style.available if not s.startswith("_"))[:8], "...")

def demo(ax):
    for ch, grp in orders.groupby("channel"):
        m = grp.groupby(grp.order_date.dt.to_period("M").dt.to_timestamp()).size()
        ax.plot(m.index, m.values, label=ch)
    ax.set(title="Orders per month by channel", xlabel="Month", ylabel="Orders (count)")
    ax.legend()


fig, axs = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
for ax, style in zip(axs, ["default", "ggplot", "bmh"]):
    with plt.style.context(style):
        pass
    demo(ax)
plt.show()

# %% [markdown]
# The loop above did **not** restyle anything: a style must be active *while the figure is created*, not afterwards. This is a very common surprise. The correct pattern wraps the creation:

# %%
def make(style):
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
        demo(ax)
    return fig, ax


styled = [make(s) for s in ("default", "ggplot", "bmh")]
plt.show()
print("gridlines visible per style:", [bool(ax.xaxis.get_gridlines()[0].get_visible()) for _, ax in styled])

# %% [markdown]
# A style is a starting point; house style usually comes from a small function you apply to every axes, which is more explicit and testable than global state:

# %%
def tidy(ax):
    """Minimal, readable defaults: no top/right frame, light horizontal grid, tick labels a comfortable size."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=9)
    ax.set_axisbelow(True)
    return ax


fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
for ch, grp in orders.groupby("channel"):
    m = grp.groupby(grp.order_date.dt.to_period("M").dt.to_timestamp()).size()
    ax.plot(m.index, m.values, label=ch)
tidy(ax)
ax.set(title="Orders per month by channel", xlabel="Month", ylabel="Orders (count)")
ax.legend(title="Channel", frameon=False, loc="upper left")
plt.show()

# %% [markdown]
# Removing the top/right spines and using a faint grid reduces "chart junk" (ink that carries no data). **Legend placement**: `loc="upper left"` works when that corner is empty; otherwise put it outside with
# `ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")`, or better, label lines **directly** (`ax.text` at the line end) and drop the legend, a technique we use in the Seaborn storytelling notebook.
#
# Colours: the default cycle (`tab:blue`, `tab:orange`, ...) is fine for exploration. For final charts use **few colours with a purpose** and check colour-blind safety (notebook 05).

# %% [markdown]
# ## Common mistakes
#
# 1. **Applying a style after drawing** (nothing changes). Use `with plt.style.context(...)` around figure creation, or `plt.style.use` first.
# 2. **Panels with independent axes** when the goal is comparison. Use `sharex`/`sharey` (or state clearly that scales differ).
# 3. **Iterating `axs` when there is only one row/column** (1-D array) or forgetting `.flat` for 2-D arrays.
# 4. **Leaving unused panels empty and visible.** `ax.set_visible(False)` or `ax.axis("off")`.
# 5. **Overlapping labels** because no layout manager was used. `layout="constrained"`.
# 6. **Over-annotating**, or arrows that cover the data.
# 7. **Legend hiding data**; pick a free corner, move it outside, or label directly.

# %%
# Mistake 3: a 1-D vs 2-D axes array
_, one_row = plt.subplots(1, 3)
_, grid = plt.subplots(2, 3)
print("1x3 ->", one_row.shape, "  2x3 ->", grid.shape, "  grid.flat gives", len(list(grid.flat)), "axes")
plt.close("all")

# %% [markdown]
# ## Interview corner
#
# **1. How do you make several panels comparable?**
# <details><summary>Answer</summary>
# Give them identical axes: <code>sharex=True, sharey=True</code> (or set the same limits) and the same units and bin widths. Small multiples work precisely because the scale is constant across panels.
# </details>
#
# **2. <code>tight_layout</code> versus <code>constrained</code> layout?**
# <details><summary>Answer</summary>
# Both avoid overlapping text. <code>tight_layout</code> is an older one-shot adjustment; <code>layout="constrained"</code> is a constraint solver that also handles colourbars, legends and suptitles better and re-runs when the figure changes.
# </details>
#
# **3. How do you highlight a period (for example a promotion) on a time-series plot?**
# <details><summary>Answer</summary>
# <code>ax.axvspan(start, end, alpha=0.2)</code> for a shaded band, plus a label or legend entry, and an <code>annotate</code> for the key point if needed.
# </details>
#
# **4. Why write a title like "Orders peak every Nov-Dec" instead of "Orders per month"?**
# <details><summary>Answer</summary>
# An action title states the finding, so the reader knows what to look for; a descriptive title makes them work it out. Use the descriptive form only when you truly want open exploration.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Six panels, one scale
# Build a 2 x 3 grid `fig1, axs1` of scatter plots (`area_sqft` on x, `price / 1000` on y) **one panel per neighbourhood** in alphabetical order, with **shared x and y axes**, marker size `s <= 15`, each panel's title equal to the neighbourhood name, and
# axis labels (with units) on the figure via `supxlabel`/`supylabel` or on the outer panels. *Hint: reuse the small-multiples cell.*

# %% tags=["starter"]
fig1, axs1 = plt.subplots(2, 3)
# your code here

# %%
assert axs1.shape == (2, 3), "2 x 3 grid"
hoods = sorted(homes.neighborhood.unique())
assert [ax.get_title().split(" ")[0] for ax in axs1.flat] == hoods, "one panel per neighbourhood, alphabetical, title starting with its name"
for ax, hood in zip(axs1.flat, hoods):
    d = homes[homes.neighborhood == hood]
    assert len(ax.collections) == 1 and ax.collections[0].get_offsets().shape[0] == len(d), f"panel {hood}: plot exactly that neighbourhood's listings"
    assert np.max(ax.collections[0].get_sizes()) <= 15, "marker size s <= 15"
a0 = axs1.flat[0]
for ax in axs1.flat[1:]:
    assert a0.get_shared_x_axes().joined(a0, ax), "panels must share the x axis (sharex=True)"
    assert a0.get_shared_y_axes().joined(a0, ax), "panels must share the y axis (sharey=True)"
texts = " ".join([t.get_text() for t in fig1.texts] + [ax.get_xlabel() + " " + ax.get_ylabel() for ax in axs1.flat]).lower()
assert "sq ft" in texts and "usd" in texts, "label the axes with units: sq ft and USD"
print("Exercise 1 passed")

# %% tags=["solution"]
fig1, axs1 = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True, sharey=True, layout="constrained")
for ax, hood in zip(axs1.flat, sorted(homes.neighborhood.unique())):
    d = homes[homes.neighborhood == hood]
    ax.scatter(d.area_sqft, d.price / 1000, s=6, alpha=0.3)
    ax.set_title(hood)
fig1.supxlabel("Living area (sq ft)")
fig1.supylabel("Price (USD thousands)")

# %% [markdown]
# ### Exercise 2: Point at the peak
# Plot `monthly` (`fig2, ax2`) and add **one annotation with an arrow** whose text includes the peak month formatted like `Dec 2024` and the number of orders at the peak. Title and axis labels required.
# *Hint: `ax.annotate(text, xy=(peak_month, peak_value), xytext=..., arrowprops={"arrowstyle": "->"})`; `peak = monthly.idxmax()`, format with `f"{peak:%b %Y}"`.*

# %% tags=["starter"]
fig2, ax2 = plt.subplots()
# your code here

# %%
peak = monthly.idxmax()
ann = [t for t in ax2.texts if getattr(t, "arrow_patch", None) is not None]
assert len(ann) >= 1, "add an annotation with arrowprops so it has an arrow"
txt = ann[0].get_text()
assert f"{peak:%b %Y}" in txt, f"annotation text should contain the peak month '{peak:%b %Y}'"
assert str(int(monthly.max())) in txt, f"annotation text should contain the peak count {int(monthly.max())}"
assert np.allclose(ann[0].xy[1], monthly.max()), "the arrow should point at the peak value"
assert ax2.get_title() and ax2.get_xlabel() and ax2.get_ylabel()
assert len(ax2.lines) >= 1 and np.array_equal(ax2.lines[0].get_ydata(), monthly.values)
print("Exercise 2 passed")

# %% tags=["solution"]
peak = monthly.idxmax()
fig2, ax2 = plt.subplots(figsize=(9, 3.6), layout="constrained")
ax2.plot(monthly.index, monthly.values)
ax2.annotate(f"Peak: {peak:%b %Y} ({monthly.max()} orders)", xy=(peak, monthly.max()),
             xytext=(peak - pd.Timedelta(days=250), monthly.max() * 0.9), arrowprops={"arrowstyle": "->"})
ax2.set(title="Orders per month", xlabel="Month", ylabel="Orders (count)")

# %% [markdown]
# ### Exercise 3: Shade the holiday season
# On a new plot of `monthly` (`fig3, ax3`), shade **1 Nov to 31 Dec of every year that appears in the data** with `axvspan` (one band per year), colour of your choice with `alpha <= 0.4`. Add title and axis labels.
# *Hint: loop over `monthly.index.year.unique()`. The data starts in June 2023 and ends in May 2025.*

# %% tags=["starter"]
fig3, ax3 = plt.subplots()
# your code here

# %%
years = [y for y in sorted(monthly.index.year.unique()) if pd.Timestamp(f"{y}-11-01") <= monthly.index.max()]
assert len(years) == 2, "sanity: two holiday seasons are in the data"
spans = [p for p in ax3.patches]
assert len(spans) == len(years), f"expected {len(years)} shaded bands (one per holiday season), found {len(spans)}"
starts = sorted(mdates.num2date(p.get_x()).date() for p in spans)
assert starts == [pd.Timestamp(f"{y}-11-01").date() for y in years], f"bands should start on 1 Nov; got {starts}"
for p in spans:
    width_days = p.get_width()
    assert 59 <= width_days <= 61, f"a Nov-Dec band should be about 60 days wide, got {width_days:.0f}"
    assert p.get_alpha() is not None and p.get_alpha() <= 0.4, "alpha <= 0.4 so the data stays visible"
assert len(ax3.lines) >= 1 and ax3.get_title() and ax3.get_xlabel() and ax3.get_ylabel()
print("Exercise 3 passed")

# %% tags=["solution"]
fig3, ax3 = plt.subplots(figsize=(9, 3.6), layout="constrained")
ax3.plot(monthly.index, monthly.values)
for y in monthly.index.year.unique():
    if pd.Timestamp(f"{y}-11-01") <= monthly.index.max():
        ax3.axvspan(pd.Timestamp(f"{y}-11-01"), pd.Timestamp(f"{y}-12-31"), color="orange", alpha=0.25)
ax3.set(title="Orders per month, holiday season shaded", xlabel="Month", ylabel="Orders (count)")

# %% [markdown]
# ### Exercise 4: A scatter with a marginal histogram
# Recreate the mosaic layout: `fig4` with panels named `"main"` (scatter of `area_sqft` vs `price / 1000`) and `"side"` (a **horizontal** histogram of `price / 1000` with 40 bins) that **shares the y axis** with `main`.
# Keep the axes dict in `axd4`. Titles and labels with units on both.

# %% tags=["starter"]
fig4 = plt.figure(layout="constrained")
axd4 = {}
# your code here

# %%
assert set(axd4) == {"main", "side"}, "the axes dict needs the keys 'main' and 'side'"
main, side = axd4["main"], axd4["side"]
assert len(main.collections) == 1 and main.collections[0].get_offsets().shape[0] == len(homes)
assert len(side.patches) == 40, "40 histogram bins"
assert abs(sum(p.get_width() for p in side.patches) - len(homes)) < 1e-6, "a horizontal histogram's counts are the bar WIDTHS; every listing must be counted"
assert all(p.get_width() >= 0 and p.get_height() > 0 for p in side.patches), "bars should extend horizontally (orientation='horizontal')"
assert main.get_shared_y_axes().joined(main, side), "share the y axis (sharey=True in subplot_mosaic)"
for ax in (main, side):
    assert ax.get_title() and ax.get_xlabel(), "title and x label on both panels"
assert "usd" in main.get_ylabel().lower(), "y label with unit (USD) on the main panel"
print("Exercise 4 passed")

# %% tags=["solution"]
fig4 = plt.figure(figsize=(8, 4.5), layout="constrained")
axd4 = fig4.subplot_mosaic([["main", "main", "side"], ["main", "main", "side"]], gridspec_kw={"width_ratios": [3, 3, 1]}, sharey=True)
axd4["main"].scatter(homes.area_sqft, homes.price / 1000, s=6, alpha=0.3)
axd4["main"].set(title="Price vs area", xlabel="Living area (sq ft)", ylabel="Price (USD thousands)")
axd4["side"].hist(homes.price / 1000, bins=40, orientation="horizontal")
axd4["side"].set(title="Price distribution", xlabel="Listings (count)")

# %% [markdown]
# ### Exercise 5 (open-ended): Your house style
# Write `house_style(ax)` that applies four choices you would want on every chart at work (spines, grid, font sizes, colours, legend). Apply it to the channel chart from section 5 and to a chart of your own.
# In a markdown cell, justify each choice in one sentence (what does it remove or add for the reader?).

# %% [markdown]
# ## Recap
# - `plt.subplots(r, c, sharex=, sharey=, layout="constrained")` gives an array of Axes; use `.flat`, hide spare panels, label once with `supxlabel`/`supylabel`.
# - **Small multiples** with shared axes let readers compare groups without a rainbow legend; put the group size in each title.
# - `subplot_mosaic` builds irregular layouts and names the panels.
# - Annotate the finding: `annotate` + arrow, `axvspan` bands, reference lines; use few, well-placed notes and finding-style titles.
# - Style sheets must be active **when the figure is made**; a small `tidy(ax)` function is often clearer than global state.
#
# **Go deeper:** Matplotlib docs "Arranging multiple Axes in a Figure", "Annotations" and "Style sheets reference" (matplotlib.org/stable).
#
# **Next:** [Time series and multi-panel dashboards](04_time_series_and_multi_panel_dashboards.ipynb)
