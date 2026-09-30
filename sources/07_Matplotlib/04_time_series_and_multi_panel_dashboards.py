# %% [markdown]
# # Time Series and Multi-Panel Dashboards
#
# **Why this matters.** Most business data is a time series: orders per day, sensor readings per hour, sign-ups per week. The daily line is noisy, the gaps are meaningful, the anomalies are the reason someone asked you
# for the chart. Turning a raw timestamped table into a readable multi-panel view (trend, seasonality, outliers) is a core analyst skill.
#
# **You will learn**
# - put a datetime index to work: `resample`, rolling means, and date-aware axis formatting
# - show **missing data** honestly (gaps, not interpolated lies) and highlight anomalies
# - stack panels that **share a time axis**
# - build a small dashboard (several panels, one message) and know when a twin axis is a bad idea
#
# **Prerequisites:** [Subplots, layout, annotations](03_subplots_layout_annotations_styling.ipynb) and pandas datetimes.
#
# > Data used: `datasets/orders.csv` (daily shop orders) and `datasets/sensors.csv` (hourly temperature and power, with sensor dropouts and labelled anomalies). Both synthetic; see `docs/DATASETS.md`.

# %%
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.max_open_warning"] = 0
DATA = Path("../datasets")

orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
daily = orders.groupby("order_date").size().asfreq("D", fill_value=0).rename("orders")     # one row per calendar day, zeros where nothing happened
sensors = pd.read_csv(DATA / "sensors.csv", parse_dates=["timestamp"]).set_index("timestamp")
print(f"daily orders: {len(daily)} days | sensors: {len(sensors)} hourly rows, {sensors.temperature_c.isna().sum()} missing temperatures, {sensors.is_anomaly.sum()} labelled anomalies")

# %% [markdown]
# ## 1. Raw vs smoothed
# Daily counts jitter; the *trend* lives in a rolling average. Plot both: the raw series thin and pale (it is the evidence), the smoothed one thick (it is the message).

# %%
smooth = daily.rolling(28, center=True, min_periods=14).mean()
fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
ax.plot(daily.index, daily.values, linewidth=0.6, alpha=0.5, label="daily")
ax.plot(smooth.index, smooth.values, linewidth=2.2, label="28-day rolling mean")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set(title="Daily orders with 28-day trend", xlabel="Date", ylabel="Orders per day (count)")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# *What just happened?* `rolling(28, center=True)` averages 4 weeks around each day; `center=True` avoids shifting the trend late, `min_periods` lets the edges be computed from fewer days (note: less reliable).
# 28 days is a multiple of 7, so every weekday is represented equally. `ConciseDateFormatter` picks readable date labels automatically. **Aggregating to a coarser period** is the other smoothing tool:

# %%
weekly = daily.resample("W").sum()
monthly = daily.resample("MS").sum()
print("days:", len(daily), "| weeks:", len(weekly), "| months:", len(monthly))
print("the first and last period may be partial:", weekly.index[0].date(), "...", weekly.index[-1].date())

# %% [markdown]
# Partial first/last periods (for instance a half-filled final week) look like a crash at the end of a line: drop or flag them before interpreting.

# %% [markdown]
# ## 2. Missing data: show the gap
# The sensor file has dropouts: hours with no temperature. If you fill them (or let the line interpolate silently), the chart claims knowledge you do not have. Matplotlib breaks a line at `NaN`, which is exactly the honest behaviour.

# %%
first_week = sensors.loc["2024-01-01":"2024-06-30"]
gaps = first_week.temperature_c.isna()
print("missing temperature hours:", int(gaps.sum()), "| longest run of consecutive missing hours:", int(gaps.groupby((~gaps).cumsum()).sum().max()))

fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
ax.plot(first_week.index, first_week.temperature_c, linewidth=0.6)
ax.plot(first_week.index, first_week.temperature_c.interpolate(), linewidth=0.6, color="red", alpha=0.25, label="interpolated (hides dropouts)")
ax.set(title="Hourly temperature: dropouts appear as gaps", xlabel="Date", ylabel="Temperature (deg C)")
ax.legend()
plt.show()

# %% [markdown]
# At this zoom the gaps are tiny; zoom into the dropout hours to see the difference. Decide *explicitly* how to treat gaps (leave, interpolate short ones, flag long ones) and say so in the chart's note.

# %%
bad = sensors.index[sensors.temperature_c.isna()][0]
win = sensors.loc[bad - pd.Timedelta(hours=12): bad + pd.Timedelta(hours=12)]
fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
ax.plot(win.index, win.temperature_c, marker="o", markersize=3, label="as recorded (NaN breaks the line)")
ax.plot(win.index, win.temperature_c.interpolate(), linestyle=":", label="linear interpolation")
ax.set(title="Zoom on a sensor dropout", xlabel="Time", ylabel="Temperature (deg C)")
ax.legend()
fig.autofmt_xdate()
plt.show()

# %% [markdown]
# ## 3. Highlight anomalies
# Overlay the labelled anomalies as a second artist (a scatter) on top of the line, so they are visible without hiding the normal data.

# %%
power = sensors.power_kw
anom = sensors[sensors.is_anomaly == 1]
fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
ax.plot(power.index, power.values, linewidth=0.5, label="power")
ax.scatter(anom.index, anom.power_kw, color="crimson", s=30, zorder=3, label=f"anomaly (n={len(anom)})")
ax.set(title="Hourly power draw with labelled anomalies", xlabel="Date", ylabel="Power (kW)")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# `zorder` controls drawing order (higher on top). Notice how the anomaly points relate to the **normal range**: a point is only "anomalous" relative to the pattern. The sensors show a daily cycle; look at it by hour of day:

# %%
by_hour = sensors.groupby(sensors.index.hour).power_kw.agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(7, 3.2), layout="constrained")
ax.plot(by_hour.index, by_hour["mean"], marker="o")
ax.fill_between(by_hour.index, by_hour["mean"] - by_hour["std"], by_hour["mean"] + by_hour["std"], alpha=0.25, label="mean +/- 1 std")
ax.set_xticks(range(0, 24, 3))
ax.set(title="Average power by hour of day", xlabel="Hour of day", ylabel="Power (kW)")
ax.legend()
plt.show()

# %% [markdown]
# A band (`fill_between`) shows spread around the mean. A daily cycle means a fixed threshold (say "above 300 kW") is the wrong anomaly rule: a value normal at 14:00 may be extreme at 03:00. Compare to the expected value **for that hour**.

# %% [markdown]
# ## 4. Panels that share a time axis
# Stacked panels with `sharex=True` let you read across: "when power spikes, what does temperature do?" Each panel keeps its own y-scale and units, so you avoid forcing two variables onto one axis.

# %%
d = sensors.resample("D").agg({"temperature_c": "mean", "power_kw": "mean", "is_anomaly": "sum"})
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True, layout="constrained", height_ratios=[3, 3, 1.2])
a1.plot(d.index, d.power_kw)
a1.set_ylabel("Mean power (kW)")
a2.plot(d.index, d.temperature_c, color="tab:orange")
a2.set_ylabel("Mean temp (deg C)")
a3.bar(d.index, d.is_anomaly, width=1.0, color="crimson")
a3.set(ylabel="Anomalies (count)", xlabel="Date")
a1.set_title("Sensor dashboard: daily mean power, temperature, anomaly counts")
plt.show()
print("shared x:", a1.get_shared_x_axes().joined(a1, a3))

# %% [markdown]
# **Twin axes** (`ax.twinx()`) put two y-scales on one panel. Tempting, but the apparent correlation depends on how you scale each axis, and readers cannot tell which line belongs to which scale without colour coding.
# Prefer stacked panels. If you must use twin axes, colour each y-axis label to match its line and say why.

# %% [markdown]
# ## 5. A compact dashboard
# Combine what we have into one figure with a single message: *orders are seasonal, sensors are mostly steady apart from a few anomalies.* Use a mosaic, identical fonts, units everywhere.

# %%
fig = plt.figure(figsize=(11, 6), layout="constrained")
axd = fig.subplot_mosaic([["trend", "trend"], ["hour", "chan"]])
axd["trend"].plot(weekly.index[1:-1], weekly.values[1:-1])
axd["trend"].set(title="Weekly orders (partial first/last weeks dropped)", xlabel="Week", ylabel="Orders per week (count)")
axd["hour"].plot(by_hour.index, by_hour["mean"], marker="o")
axd["hour"].set(title="Power by hour of day", xlabel="Hour", ylabel="Power (kW)")
share = orders.channel.value_counts(normalize=True).mul(100).sort_values()
axd["chan"].barh(share.index, share.values)
axd["chan"].set(title="Order share by channel", xlabel="Share of orders (%)", ylabel="Channel")
fig.suptitle("Shop and sensor overview")
plt.show()

# %% [markdown]
# ## Common mistakes
#
# 1. **Unsorted dates**: a line drawn through unsorted timestamps zig-zags. `sort_index()` first.
# 2. **Filling gaps silently** (interpolation, `fillna(0)`): zeros in a sensor series are *readings*, not absence.
# 3. **Partial first/last period** drawn as a crash (incomplete week or month).
# 4. **Rolling window not matching the cycle** (use 7 or 28 days for weekly seasonality).
# 5. **Strings instead of datetimes**: the axis becomes categorical with hundreds of unreadable ticks. Parse dates (`parse_dates=`).
# 6. **Twin axes with free scales** suggesting false correlation.
# 7. **Fixed thresholds on cyclical data** for anomalies.

# %%
# Mistake 1: shuffled timestamps make the line jump back and forth
rng = np.random.default_rng(42)
shuf = daily.iloc[:60].sample(frac=1, random_state=42)
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3), layout="constrained", sharey=True)
a.plot(shuf.index, shuf.values)
a.set(title="Unsorted: zig-zag", xlabel="Date", ylabel="Orders per day (count)")
b.plot(shuf.sort_index().index, shuf.sort_index().values)
b.set(title="Sorted", xlabel="Date")
fig.autofmt_xdate()
plt.show()

# %% [markdown]
# ## Interview corner
#
# **1. How do you smooth a noisy daily series without distorting it?**
# <details><summary>Answer</summary>
# Use a centered rolling mean with a window matching the seasonality (7 or 28 days for weekly patterns) or resample to a coarser period, keep the raw series faintly visible, and be aware that edges use fewer points.
# </details>
#
# **2. What should a chart do about missing values?**
# <details><summary>Answer</summary>
# Show them as gaps (NaN breaks the line) or mark them, and document any imputation. Silent interpolation or zero-filling makes the chart claim data you do not have.
# </details>
#
# **3. Why can a fixed threshold be a poor anomaly rule for a sensor?**
# <details><summary>Answer</summary>
# Seasonality and daily cycles shift the normal range: a value that is normal at peak hours is extreme at night. Compare against the expected value for the same hour/day, for example a rolling or per-hour baseline with a spread measure.
# </details>
#
# **4. Why avoid twin y-axes?**
# <details><summary>Answer</summary>
# Each axis scale is arbitrary, so the two lines can be made to cross or correlate visually at will, and readers struggle to tell which scale applies. Stacked panels with a shared x-axis are clearer.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Trend over noise
# Plot `daily` (raw, thin) and its **centered 28-day rolling mean** (`min_periods=14`) on one Axes `fig1, ax1`. The raw line first, the smoothed line second and thicker. Give labels for a legend, a title and axis labels with units.

# %% tags=["starter"]
fig1, ax1 = plt.subplots()
# your code here

# %%
exp = daily.rolling(28, center=True, min_periods=14).mean()
assert len(ax1.lines) == 2, "two lines: raw then smoothed"
raw_l, sm_l = ax1.lines
assert np.array_equal(raw_l.get_ydata(), daily.values), "first line = raw daily counts"
assert np.allclose(sm_l.get_ydata(), exp.values, equal_nan=True), "second line = centered 28-day rolling mean with min_periods=14"
assert sm_l.get_linewidth() > raw_l.get_linewidth(), "make the smoothed line thicker than the raw one"
assert ax1.get_legend() is not None and ax1.get_title() and ax1.get_xlabel() and "order" in ax1.get_ylabel().lower()
print("Exercise 1 passed")

# %% tags=["solution"]
smooth28 = daily.rolling(28, center=True, min_periods=14).mean()
fig1, ax1 = plt.subplots(figsize=(10, 3.6), layout="constrained")
ax1.plot(daily.index, daily.values, linewidth=0.6, alpha=0.5, label="daily")
ax1.plot(smooth28.index, smooth28.values, linewidth=2.2, label="28-day rolling mean")
ax1.set(title="Daily orders with 28-day trend", xlabel="Date", ylabel="Orders per day (count)")
ax1.legend()

# %% [markdown]
# ### Exercise 2: Anomalies on top
# Plot `sensors.temperature_c` as a line and overlay the **labelled anomalies** (`is_anomaly == 1`) as a red scatter with a legend label. Store `fig2, ax2`. The line must keep its `NaN` gaps (do not fill them).

# %% tags=["starter"]
fig2, ax2 = plt.subplots()
# your code here

# %%
an = sensors[sensors.is_anomaly == 1]
assert len(ax2.lines) == 1 and len(ax2.collections) == 1, "one line plus one scatter collection"
y = ax2.lines[0].get_ydata()
assert np.isnan(y).sum() == sensors.temperature_c.isna().sum(), "keep the NaN gaps: do not fill or drop missing temperatures"
off = ax2.collections[0].get_offsets()
assert off.shape[0] == len(an), f"expected {len(an)} anomaly points"
assert np.allclose(off[:, 1], an.temperature_c.values, equal_nan=True), "scatter y should be the temperature at the anomaly hours"
assert ax2.get_legend() is not None and ax2.get_title() and ax2.get_xlabel() and "c" in ax2.get_ylabel().lower()
print(f"Exercise 2 passed ({len(an)} anomalies)")

# %% tags=["solution"]
an = sensors[sensors.is_anomaly == 1]
fig2, ax2 = plt.subplots(figsize=(10, 3.6), layout="constrained")
ax2.plot(sensors.index, sensors.temperature_c, linewidth=0.5, label="temperature")
ax2.scatter(an.index, an.temperature_c, color="red", s=25, zorder=3, label="anomaly")
ax2.set(title="Hourly temperature with labelled anomalies", xlabel="Date", ylabel="Temperature (deg C)")
ax2.legend()

# %% [markdown]
# ### Exercise 3: Two panels, one clock
# Make `fig3` with **two stacked panels sharing the x-axis**: daily **mean** `power_kw` on top, daily **maximum** `power_kw` below (use `sensors.resample("D")`). Each panel needs its own y label with units (kW); put the x label and a figure title on the figure.
# Store the axes as `top, bottom`.

# %% tags=["starter"]
fig3, (top, bottom) = plt.subplots(2, 1)
# your code here

# %%
dmean = sensors.power_kw.resample("D").mean()
dmax = sensors.power_kw.resample("D").max()
assert len(top.lines) >= 1 and len(bottom.lines) >= 1
assert np.allclose(top.lines[0].get_ydata(), dmean.values), "top panel: daily mean power"
assert np.allclose(bottom.lines[0].get_ydata(), dmax.values), "bottom panel: daily maximum power"
assert top.get_shared_x_axes().joined(top, bottom), "share the x axis (sharex=True)"
assert "kw" in top.get_ylabel().lower() and "kw" in bottom.get_ylabel().lower() and top.get_ylabel() != bottom.get_ylabel(), "distinct y labels with unit kW"
assert bottom.get_xlabel() or fig3.get_supxlabel(), "x label needed"
assert fig3.get_suptitle() or top.get_title(), "figure title needed"
print("Exercise 3 passed")

# %% tags=["solution"]
dmean = sensors.power_kw.resample("D").mean()
dmax = sensors.power_kw.resample("D").max()
fig3, (top, bottom) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
top.plot(dmean.index, dmean.values)
top.set_ylabel("Mean power (kW)")
bottom.plot(dmax.index, dmax.values, color="tab:red")
bottom.set(ylabel="Max power (kW)", xlabel="Date")
fig3.suptitle("Daily mean and maximum power draw")

# %% [markdown]
# ### Exercise 4: Order seasonality in numbers and a picture
# Using only orders dated in **calendar year 2024**, compute `monthly_orders` (a Series of order counts indexed by month number 1-12) and draw it as a bar chart `fig4, ax4` with 12 bars and month names on the x ticks.
# Also set `peak_month` to the number (1-12) of the busiest month. *Hint: `orders.loc[orders.order_date.dt.year == 2024]`.*

# %% tags=["starter"]
fig4, ax4 = plt.subplots()
monthly_orders = ...
peak_month = ...

# %%
o24 = orders[orders.order_date.dt.year == 2024]
exp = o24.groupby(o24.order_date.dt.month).size()
assert len(exp) == 12
assert len(ax4.patches) == 12 and [int(p.get_height()) for p in ax4.patches] == exp.tolist(), "12 bars with the 2024 monthly order counts"
assert [t.get_text() for t in ax4.get_xticklabels()][0].lower().startswith("jan"), "month names on the x ticks (Jan ... Dec)"
assert list(monthly_orders.values) == exp.tolist() and list(monthly_orders.index) == list(range(1, 13)), "monthly_orders: counts indexed by month 1-12"
assert peak_month == int(exp.idxmax())
assert ax4.get_title() and ax4.get_xlabel() and "order" in ax4.get_ylabel().lower()
print(f"Exercise 4 passed (peak month = {peak_month})")

# %% tags=["solution"]
o24 = orders[orders.order_date.dt.year == 2024]
monthly_orders = o24.groupby(o24.order_date.dt.month).size()
peak_month = int(monthly_orders.idxmax())
fig4, ax4 = plt.subplots(figsize=(8, 3.4), layout="constrained")
ax4.bar(monthly_orders.index, monthly_orders.values)
ax4.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax4.set(title="Orders by month, 2024", xlabel="Month", ylabel="Orders (count)")

# %% [markdown]
# ### Exercise 5 (open-ended): Build your own dashboard
# Combine three panels of your choice from `orders` and `sensors` into one figure with a headline title that states a finding. Remove one element you think adds noise (grid, legend, spines) and say why in markdown. Check that every panel has units.

# %% [markdown]
# ## Recap
# - Put timestamps in a datetime index; `resample` for coarser periods, centered `rolling` for trends (window = multiple of the cycle).
# - Show raw data faintly under the smoothed line; watch partial first/last periods.
# - Let `NaN` create gaps; never fill silently. Overlay anomalies with a second artist and `zorder`.
# - Stack panels with `sharex=True` rather than using twin axes; build dashboards with `subplot_mosaic` and finding-style titles.
#
# **Go deeper:** Matplotlib docs, "Date tick labels" and "Shared axis" examples; the pandas user guide "Time series / date functionality" (pandas.pydata.org/docs/user_guide).
#
# **Next:** [Saving and publication quality](05_saving_and_publication_quality.ipynb)
