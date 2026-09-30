# %% [markdown]
# # Anatomy of a Matplotlib Figure: Figure, Axes and the Object-Oriented API
#
# **Why this matters.** Every chart you send to a stakeholder is a small argument, and a chart you cannot control is a chart you cannot fix: the legend overlaps, the labels are cut off, the second plot
# lands on the wrong panel. Matplotlib sits underneath pandas' `.plot()`, seaborn and many other libraries, so understanding its object model is what lets you fix *any* of them.
# The single most useful idea is to stop using the implicit `plt.plot(...)` state machine and hold on to `fig` and `ax` explicitly.
#
# **You will learn**
# - name the parts of a figure: `Figure`, `Axes`, `Axis`, `Artist` (lines, text, patches)
# - create figures with `fig, ax = plt.subplots()` and draw with `ax.plot(...)`, `ax.set_title(...)`
# - explain why the object-oriented style beats `plt.` state-machine calls once a chart has more than one panel
# - control limits, ticks, labels, legends and number formats
# - **inspect** a figure programmatically (how many lines? what is the title?), which is how this notebook checks your exercises
#
# **Prerequisites:** [NumPy arrays](../04_NumPy/) and [pandas loading](../05_Pandas/01_loading_and_inspecting_data.ipynb).
#
# > Data used: `datasets/orders.csv` (synthetic online-shop orders, see `docs/DATASETS.md`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

plt.rcParams["figure.max_open_warning"] = 0     # we make many small figures on purpose

DATA = Path("../datasets")
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
monthly = orders.groupby(orders.order_date.dt.to_period("M")).size().rename("n_orders")
monthly.index = monthly.index.to_timestamp()        # PeriodIndex -> first day of each month (plays well with matplotlib)
monthly.head(3)

# %% [markdown]
# ## 1. The two ways to plot (and why only one scales)
#
# **State-machine style** (the `pyplot` interface): `plt.plot(...)`, `plt.title(...)`. Matplotlib quietly keeps a "current figure" and "current axes" and all `plt.*` calls act on them. Fine in a 5-line demo.
#
# **Object-oriented style**: you ask for a figure and its axes, keep the variables, and call methods on them. Everything is explicit.

# %%
# State-machine style
plt.plot(monthly.index, monthly.values)
plt.title("Orders per month (state machine)")
plt.xlabel("Month")
plt.ylabel("Orders (count)")
plt.show()

# %%
# Object-oriented style: same picture, but now we own `fig` and `ax`
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(monthly.index, monthly.values)
ax.set_title("Orders per month")
ax.set_xlabel("Month")
ax.set_ylabel("Orders (count)")
plt.show()

# %% [markdown]
# *What just happened?* Both drew the same line. The difference shows up when you have **several panels** or helper functions. Watch what the state machine does when we write a helper that "just plots":

# %%
def add_trend_implicit(values):
    plt.plot(values)              # draws on whatever axes is "current"... which one is that?


fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 3))
add_trend_implicit([1, 2, 3])     # we *meant* the left panel
print(f"left axes has {len(ax_left.lines)} lines, right axes has {len(ax_right.lines)} lines")
plt.show()

# %% [markdown]
# The line landed in the **right** panel, because the last axes created is the "current" one. With a helper that takes an `ax` argument there is no ambiguity:

# %%
def add_trend(ax, values):
    ax.plot(values)
    return ax


fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 3))
add_trend(ax_left, [1, 2, 3])
print(f"left axes has {len(ax_left.lines)} lines, right axes has {len(ax_right.lines)} lines")
plt.show()

# %% [markdown]
# **Rule:** write `fig, ax = plt.subplots()` and use `ax.` methods. Keep `plt.` for creating figures (`plt.subplots`, `plt.figure`), `plt.show()` and `plt.savefig`. Functions that accept an `ax` are reusable, testable, and composable.
# Note that the method names differ slightly: `plt.title` is `ax.set_title`, `plt.xlabel` is `ax.set_xlabel`, `plt.xlim` is `ax.set_xlim`. Or set many at once: `ax.set(title=..., xlabel=..., ylabel=...)`.

# %% [markdown]
# ## 2. The anatomy
#
# | Name | What it is | How you reach it |
# |---|---|---|
# | **Figure** | the whole canvas (the window or image file); holds one or more Axes | `fig` from `plt.subplots()` |
# | **Axes** | *one plot*: a pair of axes, the data area, title, labels (confusingly not the plural of "axis") | `ax` |
# | **Axis** | the number line itself (`ax.xaxis`, `ax.yaxis`): ticks, tick labels, scale | `ax.xaxis` |
# | **Artist** | anything drawn: lines (`Line2D`), bars (`Rectangle` patches), text, legends | `ax.lines`, `ax.patches`, `ax.texts` |
#
# Because everything is an object, you can **ask** a figure what it contains. That is also how automated tests (and this notebook's exercise checks) verify a chart without looking at pixels.

# %%
fig, ax = plt.subplots(figsize=(8, 3.5))
(line,) = ax.plot(monthly.index, monthly.values, marker="o", label="all channels")   # plot returns a list of Line2D artists
ax.set(title="Orders per month", xlabel="Month", ylabel="Orders (count)")
ax.legend()
plt.show()

print("axes in figure       :", len(fig.axes))
print("lines in axes        :", len(ax.lines))
print("title / labels       :", ax.get_title(), "|", ax.get_xlabel(), "|", ax.get_ylabel())
print("first 3 y-values     :", ax.lines[0].get_ydata()[:3])
print("legend entries       :", [t.get_text() for t in ax.get_legend().get_texts()])
print("y-axis limits        :", tuple(round(float(v), 1) for v in ax.get_ylim()))

# %% [markdown]
# `ax.plot` returned the `Line2D` object; we can change it afterwards (`line.set_linewidth(3)`, `line.set_color("tab:red")`). A figure is a **tree** of objects: Figure -> Axes -> (Axis, lines, patches, texts). Matplotlib "draws" by walking that tree.

# %%
line.set_linewidth(3)
line.set_color("tab:red")
print("linewidth now:", line.get_linewidth(), "| color:", line.get_color())

# %% [markdown]
# ## 3. Limits, ticks and number formats
#
# Axes limits default to "a bit more than the data". Always check whether that default tells the truth: for a **bar chart**, the y-axis must start at 0 (bar length encodes the value); for a **line chart** a non-zero baseline is fine because we read the *shape*.
# Format numbers for humans: thousands separators, units in the label (`Revenue (USD thousands)`), sensible date ticks.

# %%
monthly_rev = pd.Series(monthly.values * 480.0, index=monthly.index)     # a made-up revenue figure: orders x 480, just for formatting practice

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(monthly_rev.index, monthly_rev.values)
ax.set_ylim(0, None)                                                     # anchor at zero, let matplotlib choose the top
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
ax.set_title("Illustrative monthly revenue (orders x 480)")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue (USD thousands)")
ax.grid(alpha=0.3)
fig.autofmt_xdate()                                                      # tilt the date labels so they do not collide
plt.show()

# %% [markdown]
# `FuncFormatter` takes `(value, position)` and returns the text for a tick. `ax.grid(alpha=0.3)` adds a light grid (strong grids add ink without adding information). `fig.autofmt_xdate()` rotates date labels.
# (The revenue is fabricated by multiplying counts, purely to practise formatting: never present invented numbers as data.)

# %% [markdown]
# ## 4. Several series, a legend, and three coordinate systems
#
# Plot one line per channel and let the **legend** identify them. Give each `ax.plot` a `label=`; `ax.legend()` collects them. Then annotate in **axes coordinates** (0 to 1 across the plot, independent of the data) so a note stays in a corner even if the data change.

# %%
by_channel = orders.groupby([orders.order_date.dt.to_period("M").dt.to_timestamp(), "channel"]).size().unstack("channel")
fig, ax = plt.subplots(figsize=(8, 3.8))
for ch in by_channel.columns:
    ax.plot(by_channel.index, by_channel[ch], label=ch)
ax.set(title="Orders per month by channel", xlabel="Month", ylabel="Orders (count)")
ax.legend(title="Channel", loc="upper left")
ax.text(0.99, 0.02, "source: synthetic shop data", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray")
fig.autofmt_xdate()
plt.show()

# %% [markdown]
# Three coordinate systems are in play: **data** coordinates (month, count: the default for `ax.plot`/`ax.text`), **axes** coordinates (`transform=ax.transAxes`, 0-1 inside the plot area), and **figure** coordinates (`fig.transFigure`, 0-1 across the whole canvas).
# Use axes coordinates for "a note in the corner", data coordinates for "label this particular point".

# %% [markdown]
# ## 5. Figure size, resolution and layout
#
# `figsize=(width, height)` is in **inches**; `dpi` is dots per inch, so pixels = inches x dpi. Screen display is about 100 dpi; print wants 300. Layout managers prevent overlapping labels:
# `layout="constrained"` (modern, preferred) or `fig.tight_layout()`.

# %%
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
axs[0].hist(orders.order_date.dt.dayofweek, bins=np.arange(8) - 0.5, rwidth=0.85)
axs[0].set(title="Orders by weekday", xlabel="Day of week (0 = Monday)", ylabel="Orders (count)")
axs[1].bar(orders.channel.value_counts().index, orders.channel.value_counts().values)
axs[1].set(title="Orders by channel", xlabel="Channel", ylabel="Orders (count)")
print("figure size in pixels at dpi", fig.dpi, ":", (fig.get_size_inches() * fig.dpi).astype(int))
plt.show()

# %% [markdown]
# With one row and two columns `axs` is a 1-D array of two Axes; with `plt.subplots(2, 2)` it is a 2-D array (`axs[0, 1]`); with a single plot `plt.subplots()` returns a **single** Axes, not an array. That difference is a top source of `TypeError: 'Axes' object is not subscriptable`.
# Pass `squeeze=False` if you want an array every time.

# %% [markdown]
# ## Common mistakes
#
# 1. **Mixing `plt.` and `ax.` calls** without knowing which axes is current: the line or title lands in the wrong panel (shown above).
# 2. **Expecting an array of Axes** from `plt.subplots()` with one plot (it returns one `Axes`).
# 3. **Forgetting labels and units**: "Revenue" is ambiguous, "Revenue (USD thousands)" is not. Every chart needs a title, axis labels and units.
# 4. **Bars with a truncated y-axis** exaggerate differences (see the next notebook).
# 5. **Creating figures in a loop without closing them** (`plt.close(fig)`), which leaks memory in scripts. Jupyter's inline backend closes them after display.
# 6. **Forgetting `plt.show()`** in scripts (in notebooks the figure appears anyway, but make the habit).
# 7. **Overlapping tick labels**: `fig.autofmt_xdate()`, `ax.tick_params(axis="x", rotation=45)`, or `layout="constrained"`.

# %%
# Mistake 2 in action
fig, ax = plt.subplots()
try:
    ax[0].plot([1, 2, 3])
except TypeError as e:
    print("TypeError:", e)
plt.close(fig)

# %% [markdown]
# ## Interview corner
#
# **1. What is the difference between a Figure and an Axes?**
# <details><summary>Answer</summary>
# The Figure is the whole canvas; an Axes is one plot inside it (data area, axis lines, ticks, title, labels). A figure can hold many Axes. The word "Axis" refers to the individual x or y number line belonging to an Axes.
# </details>
#
# **2. Why prefer <code>fig, ax = plt.subplots()</code> over <code>plt.plot()</code>?**
# <details><summary>Answer</summary>
# Explicit objects avoid the hidden "current axes" state, so helper functions and multi-panel figures behave predictably, code is reusable (pass <code>ax</code> in), and you can test a figure by inspecting its objects.
# </details>
#
# **3. How do you place a text note in the corner of a plot that stays there whatever the data range is?**
# <details><summary>Answer</summary>
# Use axes coordinates: <code>ax.text(0.98, 0.02, "note", transform=ax.transAxes, ha="right")</code>. Data coordinates would move the note when limits change.
# </details>
#
# **4. How do you make a figure 6 inches wide and print-sharp?**
# <details><summary>Answer</summary>
# <code>figsize=(6, h)</code> sets inches; save with <code>fig.savefig("f.png", dpi=300)</code> (pixels = inches x dpi) or use a vector format like SVG/PDF that has no fixed resolution.
# </details>

# %% [markdown]
# ## Exercises
# Each exercise asks you to build a figure. The check inspects the **figure objects** (lines, labels, data), not the pixels. Keep `fig` and `ax` as variables.

# %% [markdown]
# ### Exercise 1: Monthly orders, done properly
# Create `fig, ax` with a single line showing the number of orders per month (`monthly`, index = month start). Give it a title, an x label and a y label that mentions the unit (orders or count). Do not call `plt.show()` before the check.
# *Hint: `ax.plot(monthly.index, monthly.values)` and `ax.set(...)`.*

# %% tags=["starter"]
fig, ax = plt.subplots()
# your code here

# %%
assert len(fig.axes) == 1, "exactly one Axes expected"
assert len(ax.lines) == 1, f"expected 1 line, found {len(ax.lines)}"
assert np.array_equal(ax.lines[0].get_ydata(), monthly.values), "the line's y-data must be the monthly order counts"
assert ax.get_title().strip(), "give the chart a title"
assert ax.get_xlabel().strip(), "label the x axis"
assert any(w in ax.get_ylabel().lower() for w in ("order", "count")), "y label should state the unit, e.g. 'Orders (count)'"
print("Exercise 1 passed")

# %% tags=["solution"]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(monthly.index, monthly.values)
ax.set(title="Orders per month", xlabel="Month", ylabel="Orders (count)")

# %% [markdown]
# ### Exercise 2: One line per channel with a legend
# Using `by_channel` (one column per channel), draw **one line per channel** on a single Axes, each with a `label` equal to the channel name, add a legend, a title, and axis labels with units.
# Store the figure in `fig2` and the axes in `ax2`.

# %% tags=["starter"]
fig2, ax2 = plt.subplots()
# your code here

# %%
assert len(ax2.lines) == by_channel.shape[1], f"expected one line per channel ({by_channel.shape[1]})"
legend = ax2.get_legend()
assert legend is not None, "add ax2.legend()"
assert sorted(t.get_text() for t in legend.get_texts()) == sorted(by_channel.columns), "legend labels must be the channel names"
for ch in by_channel.columns:
    match = [ln for ln in ax2.lines if ln.get_label() == ch]
    assert len(match) == 1 and np.array_equal(match[0].get_ydata(), by_channel[ch].values), f"line for {ch} has wrong data"
assert ax2.get_title() and ax2.get_xlabel() and "order" in ax2.get_ylabel().lower(), "title, x label and a y label mentioning orders are required"
print("Exercise 2 passed")

# %% tags=["solution"]
fig2, ax2 = plt.subplots(figsize=(8, 3.8))
for ch in by_channel.columns:
    ax2.plot(by_channel.index, by_channel[ch], label=ch)
ax2.set(title="Orders per month by channel", xlabel="Month", ylabel="Orders (count)")
ax2.legend(title="Channel")

# %% [markdown]
# ### Exercise 3: A reusable plotting function
# Write `draw_weekday_counts(ax, dates)` that draws a bar chart of the number of orders per weekday (Monday to Sunday, 7 bars) **into the axes you pass in** (no `plt.` calls), sets a title and labels, and returns the `ax`.
# The check calls it on two different Axes of one figure. *Hint: `dates.dt.dayofweek.value_counts().sort_index()`, then `ax.bar(...)`.*

# %% tags=["starter"]
def draw_weekday_counts(ax, dates):
    ...

# %%
fig3, (a1, a2) = plt.subplots(1, 2)
r1 = draw_weekday_counts(a1, orders.order_date)
r2 = draw_weekday_counts(a2, orders.order_date[orders.order_date.dt.year == 2024])
exp_all = orders.order_date.dt.dayofweek.value_counts().sort_index()
exp_24 = orders.order_date[orders.order_date.dt.year == 2024].dt.dayofweek.value_counts().sort_index()
assert r1 is a1 and r2 is a2, "return the axes you were given"
assert len(a1.patches) == 7 and len(a2.patches) == 7, "7 bars per Axes (Monday to Sunday); did you draw into the passed ax?"
assert [int(p.get_height()) for p in a1.patches] == exp_all.tolist(), "wrong bar heights on the first axes"
assert [int(p.get_height()) for p in a2.patches] == exp_24.tolist(), "wrong bar heights on the second axes"
assert a1.get_title() and a1.get_ylabel(), "set a title and a y label inside the function"
print("Exercise 3 passed")

# %% tags=["solution"]
def draw_weekday_counts(ax, dates):
    counts = dates.dt.dayofweek.value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set(title="Orders by weekday", xlabel="Day of week", ylabel="Orders (count)")
    return ax

# %% [markdown]
# ### Exercise 4: Mark the average
# Draw the monthly order counts again (`fig4`, `ax4`) and add a **horizontal dashed line at the mean monthly order count** with a label `"mean"` and a legend. Also set the y-axis to start at 0.
# *Hint: `ax.axhline(y, linestyle="--", label=...)`. Note `axhline` adds a Line2D to `ax.lines`.*

# %% tags=["starter"]
fig4, ax4 = plt.subplots()
# your code here

# %%
mean_val = monthly.mean()
assert len(ax4.lines) == 2, "expected the data line plus the mean line"
mean_lines = [ln for ln in ax4.lines if np.allclose(np.unique(ln.get_ydata()), mean_val)]
assert len(mean_lines) == 1, f"no horizontal line at the mean ({mean_val:.1f}) found"
assert mean_lines[0].get_linestyle() == "--", "make the mean line dashed"
assert mean_lines[0].get_label() == "mean" and ax4.get_legend() is not None, "label it 'mean' and show the legend"
assert ax4.get_ylim()[0] == 0, "start the y-axis at 0"
assert ax4.get_title() and ax4.get_xlabel() and ax4.get_ylabel()
print(f"Exercise 4 passed (mean = {mean_val:.1f} orders per month)")

# %% tags=["solution"]
fig4, ax4 = plt.subplots(figsize=(8, 3.5))
ax4.plot(monthly.index, monthly.values, label="orders")
ax4.axhline(monthly.mean(), linestyle="--", color="gray", label="mean")
ax4.set_ylim(0, None)
ax4.set(title="Orders per month vs the average", xlabel="Month", ylabel="Orders (count)")
ax4.legend()

# %% [markdown]
# ### Exercise 5 (open-ended): Read a figure like a tree
# Take the chart from exercise 2 and write code that prints, for your figure: the number of Axes, every Line2D label, the y-range actually shown, and the font size of the title.
# Then change **three** properties through the objects (not by re-plotting): a line colour, the title font size, and the x-axis tick label rotation. In a markdown cell, say which properties were hardest to find and how you located them (`help(line)`, `dir(ax)`, `line.properties()`).

# %% [markdown]
# ## Recap
# - A **Figure** holds **Axes**; an Axes owns its **Axis** objects and all **Artists** (lines, bars, text).
# - Write `fig, ax = plt.subplots()` and call `ax.` methods; helper functions should take an `ax` argument. Avoid the hidden current-axes state of bare `plt.` calls.
# - Always give a title, axis labels and **units**; anchor bar charts at zero; format numbers for humans.
# - Axes coordinates (`transAxes`) for notes in a corner, data coordinates for labelling points.
# - Figures are inspectable objects: test them (`ax.lines`, `ax.patches`, `ax.get_title()`).
#
# **Go deeper:** the Matplotlib documentation, "Quick start guide" and "Anatomy of a figure" (matplotlib.org/stable).
#
# **Next:** [Core chart types and when to use them](02_core_chart_types_and_when_to_use_them.ipynb)
