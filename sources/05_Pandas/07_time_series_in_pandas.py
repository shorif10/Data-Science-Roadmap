# %% [markdown]
# # Time Series in pandas
#
# **Why this matters.** Sensor readings, sales, web traffic, prices: most business data is indexed by time. The questions are always the same:
# "what is the trend", "what is the daily/weekly pattern", "where are the gaps", "what is unusual right now". pandas has a dedicated toolkit
# (`DatetimeIndex`, `resample`, time-based `rolling`, time zones) and using it instead of string manipulation prevents whole classes of bugs.
#
# **You will learn**
# - set a `DatetimeIndex` and select by date strings
# - find and fill gaps and dropouts, and *measure* which filling method works best
# - `resample` to daily / weekly / monthly and use time-based rolling windows
# - extract seasonality by hour and weekday, and handle time zones and daylight-saving changes
# - flag anomalies against a seasonal baseline
#
# **Prerequisites:** [Reshaping and window functions](06_reshaping_and_window_functions.ipynb).
#
# > Data used: `sensors.csv` (hourly temperature and power of a facility, with sensor dropouts and labelled anomalies) and `orders.csv`.
# > Both synthetic; see `docs/DATASETS.md`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("../datasets")
sensors = pd.read_csv(DATA / "sensors.csv", parse_dates=["timestamp"]).set_index("timestamp")
print(sensors.index.dtype, "|", sensors.shape)
sensors.head(3)

# %% [markdown]
# ## 1. A `DatetimeIndex` and partial-string selection
# Parsing the timestamp column and making it the index unlocks the time-aware tools. First check the index is what you think it is: sorted, unique, regular.

# %%
idx = sensors.index
print("sorted:", idx.is_monotonic_increasing, "| unique:", idx.is_unique, "| inferred frequency:", pd.infer_freq(idx))
print("from", idx.min(), "to", idx.max(), "|", len(idx), "rows =", (idx.max() - idx.min()) / pd.Timedelta("1h") + 1, "hours expected")

# %% [markdown]
# Because the index is datetime-aware you can slice with *partial strings*: a month, a day, or a range (with `.loc`; both ends included).

# %%
print("March rows:", len(sensors.loc["2024-03"]))
print("one day  :", len(sensors.loc["2024-03-05"]), "rows")
print(sensors.loc["2024-03-05 08:00":"2024-03-05 10:00", ["temperature_c", "power_kw"]].to_string())

# %% [markdown]
# *What just happened?* `"2024-03"` means "the whole of March". Slicing by label on a sorted DatetimeIndex is fast (binary search) and needs no `>=` / `<=` masks.

# %% [markdown]
# ## 2. Gaps and dropouts
# Two different problems hide under "missing data in a time series": **missing rows** (timestamps absent) and **missing values** (row present, value `NaN`).
# Check both. `asfreq("h")` (or `reindex`) inserts any absent timestamps as `NaN`:

# %%
full = sensors.asfreq("h")
print("rows absent from the index:", len(full) - len(sensors))
print("missing values per column:")
print(sensors.isna().sum().to_string())

# %% [markdown]
# This file has every hour present, but `temperature_c` has dropouts (`NaN` values). How long are they? Consecutive-run lengths decide the filling strategy:
# a one-hour blip can be interpolated, a two-day outage cannot be trusted to any method.

# %%
na = sensors["temperature_c"].isna()
run_id = (na != na.shift()).cumsum()                 # new id whenever the True/False status changes
run_lengths = na.groupby(run_id).sum()               # number of missing hours in each run (0 for non-missing runs)
run_lengths = run_lengths[run_lengths > 0]
print("dropout runs:", len(run_lengths), "| longest:", int(run_lengths.max()), "hours | total missing hours:", int(na.sum()))

# %% [markdown]
# Which filling method should we use for these short gaps? Instead of guessing, **test them**: hide 200 *known* readings, fill with each method, and measure the error against the truth.
# (Method: mask, fill, compare. The same trick works for any imputation question.)

# %%
temp = sensors["temperature_c"]
rng = np.random.default_rng(42)
known_pos = np.flatnonzero(temp.notna().to_numpy())
test_pos = rng.choice(known_pos, size=200, replace=False)
masked = temp.copy()
masked.iloc[test_pos] = np.nan
truth = temp.iloc[test_pos]

candidates = {
    "forward fill": masked.ffill(),
    "linear interpolation": masked.interpolate(method="time"),
    "overall mean": masked.fillna(masked.mean()),
    "hour-of-day mean": masked.fillna(masked.groupby(masked.index.hour).transform("mean")),
}
errors = pd.Series({name: (filled.iloc[test_pos] - truth).abs().mean() for name, filled in candidates.items()}).round(2)
print("mean absolute error (degrees C) on hidden readings:")
print(errors.sort_values().to_string())

# %% [markdown]
# Interpolating between neighbours wins clearly here because temperature changes smoothly hour to hour; a constant (mean) ignores that structure.
# That would **not** be true for a spiky signal. Two rules: (1) limit how far you fill (`limit=3` fills at most 3 consecutive hours), and (2) never fill *targets* or anomaly labels.

# %%
temp_filled = temp.interpolate(method="time", limit=3)
print("missing before:", int(temp.isna().sum()), "| after limited interpolation:", int(temp_filled.isna().sum()))

# %% [markdown]
# ## 3. `resample`: change the frequency
# `resample` is `groupby` for time: it buckets the index by a frequency (`"D"` day, `"W"` week, `"MS"` month start, `"h"` hour) and aggregates each bucket.
# Downsampling (hourly to daily) needs an aggregation; choose the one that matches the physics: mean temperature, *sum* of energy, *max* of a peak.

# %%
daily = sensors.resample("D").agg(
    temp_mean=("temperature_c", "mean"),
    temp_max=("temperature_c", "max"),
    power_mean=("power_kw", "mean"),
    power_max=("power_kw", "max"),
    readings=("temperature_c", "count"),       # non-missing readings: a data-quality column
)
print(daily.round(1).head(3).to_string())
print("days with fewer than 24 valid temperature readings:", int((daily["readings"] < 24).sum()), "of", len(daily))

# %% [markdown]
# Keeping a `readings` count beside each aggregate tells the reader how much data stands behind the number: a daily mean over 20 readings is less solid than over 24.
# Energy note: hourly average power in kW over one hour is kWh, so daily energy is the **sum**, not the mean.

# %%
monthly = sensors["power_kw"].resample("MS").sum() / 1000
print("energy per month (MWh):")
print(monthly.round(0).to_string())

# %% [markdown]
# The last month (June) is shorter because the data ends on 28 June. Comparing a partial month with full months is a classic reporting mistake, so look at the **mean** or flag the partial period.

# %% [markdown]
# ## 4. Rolling windows on time
# With a DatetimeIndex, `rolling("24h")` is a *time* window. `rolling(24)` would count rows and go wrong if rows were missing. `min_periods` controls how many valid values you require.

# %%
trend = sensors["temperature_c"].rolling("24h", min_periods=12).mean()
ewm = sensors["temperature_c"].ewm(span=24).mean()
fig, ax = plt.subplots(figsize=(9, 3.2))
sensors.loc["2024-04-01":"2024-04-07", "temperature_c"].plot(ax=ax, label="hourly reading", alpha=0.5)
trend.loc["2024-04-01":"2024-04-07"].plot(ax=ax, label="24h rolling mean", lw=2)
ewm.loc["2024-04-01":"2024-04-07"].plot(ax=ax, label="exponentially weighted (span 24)", lw=1.5, ls="--")
ax.set_title("Temperature: hourly readings and smoothers (one week)")
ax.set_xlabel("time")
ax.set_ylabel("temperature (degrees C)")
ax.legend()
plt.show()

# %% [markdown]
# Both smoothers lag behind the raw signal because they use only past values. `ewm` weights recent points more heavily, so it reacts faster.

# %% [markdown]
# ## 5. Seasonality: hour of day and day of week
# Extract the time component from the index and `groupby` on it. This reveals the repeating patterns that dominate most operational data.

# %%
by_hour = sensors.groupby(sensors.index.hour)[["temperature_c", "power_kw"]].mean()
print("power peak hour:", int(by_hour["power_kw"].idxmax()), "| low hour:", int(by_hour["power_kw"].idxmin()),
      "| temperature peak hour:", int(by_hour["temperature_c"].idxmax()))

weekday = sensors.groupby(sensors.index.day_name())["power_kw"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
print(weekday.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
by_hour["power_kw"].plot(ax=axes[0], marker="o")
axes[0].set_title("Average power by hour of day")
axes[0].set_xlabel("hour of day")
axes[0].set_ylabel("power (kW)")
weekday.plot.bar(ax=axes[1])
axes[1].set_title("Average power by weekday")
axes[1].set_xlabel("weekday")
axes[1].set_ylabel("power (kW)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Power follows a clear daily cycle and is lower on the weekend. Temperature peaks in the afternoon and also drifts upward over the months, which is a *trend* layered on top of the
# *seasonal* daily cycle. To see both at once, pivot hour against weekday:

# %%
hw = sensors.pivot_table(index=sensors.index.hour, columns=sensors.index.dayofweek, values="power_kw", aggfunc="mean")
hw.index.name, hw.columns.name = "hour", "weekday (0=Mon)"
hw.round(0).astype(int).iloc[::4]

# %% [markdown]
# ## 6. Time zones and daylight saving (the basics)
# A **naive** timestamp has no zone; an **aware** one does. Store and compute in **UTC**, convert to local time only for display.
# The classic trap is a local day that is not 24 hours long when clocks change:

# %%
utc = pd.date_range("2024-03-09", periods=72, freq="h", tz="UTC")
local = utc.tz_convert("America/New_York")
print("UTC start:", utc[0], "| New York start:", local[0])

ny_days = pd.Series(1, index=pd.date_range("2024-03-09", periods=72, freq="h", tz="America/New_York")).resample("D").size()
print("hours per local day around the March clock change:", ny_days.tolist()[:3])

# %% [markdown]
# On 10 March 2024 New York loses an hour (23 hours), so "daily totals" built on local days are not comparable with the days either side of it.
# Localising a naive series is done with `tz_localize` (which can raise on ambiguous or non-existent local times, so prefer UTC as the storage format):

# %%
naive = sensors.index[:3]
print(naive.tz_localize("UTC").tz_convert("Asia/Dhaka")[0], "(same instant, different wall clock)")

# %% [markdown]
# ## 7. Flagging anomalies against a seasonal baseline
# A reading is "unusual" relative to *what is normal for that hour and day type*, not relative to the overall mean. Build a baseline (median for each weekday/weekend and hour),
# take the residual, and scale by a robust spread (MAD, median absolute deviation). Then compare to the file's own `is_anomaly` labels, **including what we miss and what we wrongly flag**.

# %%
p = sensors["power_kw"]
baseline = p.groupby([p.index.dayofweek >= 5, p.index.hour]).transform("median")
resid = p - baseline
robust_sd = 1.4826 * (resid - resid.median()).abs().median()        # MAD scaled to behave like a standard deviation
z = resid / robust_sd

labels = sensors["is_anomaly"] == 1
for threshold in [3, 4, 5]:
    flagged = z.abs() > threshold
    tp = int((flagged & labels).sum())
    print(f"|z| > {threshold}: flagged {int(flagged.sum()):3d} | true anomalies caught {tp}/{int(labels.sum())} | precision {tp / flagged.sum():.0%}")

# %% [markdown]
# A lower threshold catches more but adds false alarms (lower precision); a higher one is cleaner but may miss subtle events. On real data there are no labels and the choice is a business
# trade-off (cost of a missed fault versus cost of an unnecessary call-out). These anomalies are synthetic and deliberately large, so expect real data to be harder.

# %% [markdown]
# ## 8. Gaps of a different kind: daily order counts
# Sometimes a missing day is not a dropout but a real zero. Count orders per day and compare with a full calendar:

# %%
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
per_day = orders.groupby("order_date").size()
calendar = pd.date_range(orders["order_date"].min(), orders["order_date"].max(), freq="D")
print("days in calendar:", len(calendar), "| days with at least one order:", len(per_day))
daily_orders = per_day.reindex(calendar, fill_value=0)          # a day with no orders IS zero orders
print("days with zero orders:", int((daily_orders == 0).sum()))

# %% [markdown]
# Here `fill_value=0` is correct because "no rows" means "no orders". For a temperature sensor the same missing row would mean "unknown", and filling with 0 would be absurd.
# **The meaning of the gap decides the fill.** Finally, orders have the same kind of calendar structure; check the month pattern with `resample`:

# %%
orders_monthly = daily_orders.resample("MS").sum()
print(orders_monthly.loc[["2023-10-01", "2023-11-01", "2023-12-01", "2024-01-01"]].to_string())

# %% [markdown]
# ## Common mistakes
#
# 1. **Leaving dates as strings**: sorting and slicing work on text, not time. Parse with `parse_dates` or `to_datetime`.
# 2. **Not sorting the index** before `rolling`, `shift` or slicing.
# 3. **`rolling(24)` on data with missing rows** (counts rows, not hours). Use `rolling("24h")`.
# 4. **Mean when you need sum** (energy, revenue) or sum when you need mean (temperature) in `resample`.
# 5. **Comparing partial periods** (June with 28 days) to full ones.
# 6. **Filling gaps without looking at how long they are**, or filling a real zero with interpolation (and vice versa).
# 7. **Mixing naive and aware timestamps**: pandas refuses to compare them. Store UTC, convert to local for display.
# 8. **Random train/test splits on time series** leak the future into the past. Split by time.

# %%
naive_ts = pd.Timestamp("2024-03-10 12:00")
aware_ts = pd.Timestamp("2024-03-10 12:00", tz="UTC")
try:
    naive_ts < aware_ts
except TypeError as e:
    print("TypeError:", str(e)[:80])

# %% [markdown]
# ## Interview corner
#
# **1. How do you aggregate hourly data to daily, and what should you think about?**
# <details><summary>Answer</summary>
# <code>df.resample("D").agg(...)</code> on a DatetimeIndex. Choose the aggregation that matches the quantity (mean for levels like temperature, sum for flows like energy or revenue, max for peaks),
# include a count so readers know how many observations back each value, and watch partial periods.
# </details>
#
# **2. How would you handle missing values in a sensor time series?**
# <details><summary>Answer</summary>
# Distinguish missing rows from missing values, measure the gap lengths, interpolate only short gaps (<code>limit</code>), keep an indicator of what was filled, never fill the target, and validate the method by
# hiding known values and measuring the error.
# </details>
#
# **3. Why store timestamps in UTC?**
# <details><summary>Answer</summary>
# UTC is unambiguous and has no daylight-saving jumps, so arithmetic and joins are correct; local times can repeat or not exist on clock-change days and differ between users. Convert to local only for presentation.
# </details>
#
# **4. `rolling(7)` versus `rolling("7D")`, and why does it matter for time series with gaps?**
# <details><summary>Answer</summary>
# The integer window is the last 7 <em>rows</em>; the offset window is the last 7 <em>days</em> of time. With missing days, only the latter measures a true week.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Hottest day
# From `sensors`, compute `daily_max`: the maximum `temperature_c` per calendar day (a Series with a daily DatetimeIndex), and `hottest_day`: the date (a `Timestamp`) of the highest value.

# %% tags=["starter"]
daily_max = ...
hottest_day = ...

# %%
manual = sensors["temperature_c"].groupby(sensors.index.normalize()).max()
assert len(daily_max) == len(manual) and np.allclose(daily_max.to_numpy(), manual.to_numpy(), equal_nan=True), "daily maximum of temperature_c"
assert pd.Timestamp(hottest_day) == manual.idxmax(), f"expected {manual.idxmax().date()}"
print(f"Exercise 1 passed (hottest day {pd.Timestamp(hottest_day).date()}, {daily_max.max():.1f} C)")

# %% tags=["solution"]
daily_max = sensors["temperature_c"].resample("D").max()
hottest_day = daily_max.idxmax()

# %% [markdown]
# ### Exercise 2: Fill short dropouts only
# Create `temp_ok`: `temperature_c` with gaps of **at most 2 consecutive hours** filled by time interpolation (longer gaps, if any, stay `NaN`; use `limit=2`).
# Store in `n_still_missing` how many values are still `NaN`.

# %% tags=["starter"]
temp_ok = ...
n_still_missing = ...

# %%
t = sensors["temperature_c"]
assert temp_ok.notna().sum() >= t.notna().sum(), "you must not remove data"
assert np.allclose(temp_ok[t.notna()], t[t.notna()]), "observed values must not change"
assert n_still_missing == int(temp_ok.isna().sum()), "n_still_missing counts the remaining NaN"
na = t.isna()
runs = na.groupby((na != na.shift()).cumsum()).transform("sum")
assert not temp_ok[na & (runs <= 2)].isna().any(), "every gap of up to 2 hours must be filled"
print(f"Exercise 2 passed ({int(na.sum())} missing -> {n_still_missing} still missing)")

# %% tags=["solution"]
temp_ok = sensors["temperature_c"].interpolate(method="time", limit=2)
n_still_missing = int(temp_ok.isna().sum())

# %% [markdown]
# ### Exercise 3: How much lower is the weekend?
# Compute `weekend_drop_pct`: how much lower (in percent of the weekday mean) the average `power_kw` is on Saturday and Sunday compared with Monday to Friday.
# A positive number means the weekend is lower. *Hint: `sensors.index.dayofweek >= 5` flags weekends.*

# %% tags=["starter"]
weekend_drop_pct = ...

# %%
is_we = np.array([d >= 5 for d in sensors.index.dayofweek])
wk, we = sensors["power_kw"].to_numpy()[~is_we].mean(), sensors["power_kw"].to_numpy()[is_we].mean()
assert abs(weekend_drop_pct - (wk - we) / wk * 100) < 1e-6, f"expected {(wk - we) / wk * 100:.3f}"
print(f"Exercise 3 passed (weekend power is {weekend_drop_pct:.1f}% lower)")

# %% tags=["solution"]
is_weekend = sensors.index.dayofweek >= 5
weekday_mean = sensors.loc[~is_weekend, "power_kw"].mean()
weekend_mean = sensors.loc[is_weekend, "power_kw"].mean()
weekend_drop_pct = (weekday_mean - weekend_mean) / weekday_mean * 100

# %% [markdown]
# ### Exercise 4: Weekly orders and the busiest week
# Using the `orders` table, compute `weekly_orders`: the number of orders per week (weeks ending on Sunday, pandas' default `"W"`), as a Series indexed by week end date.
# Store `busiest_week_end` (the index label of the week with the most orders) and `busiest_count`.

# %% tags=["starter"]
weekly_orders = ...
busiest_week_end = ...
busiest_count = ...

# %%
manual = orders.set_index("order_date")["order_id"].resample("W").count()
assert len(weekly_orders) == len(manual) and (weekly_orders.to_numpy() == manual.to_numpy()).all(), "orders per week (W = weeks ending Sunday)"
assert weekly_orders.sum() == len(orders), "no orders lost"
assert pd.Timestamp(busiest_week_end) == manual.idxmax() and busiest_count == manual.max(), "busiest week"
print(f"Exercise 4 passed (week ending {pd.Timestamp(busiest_week_end).date()}: {busiest_count} orders)")

# %% tags=["solution"]
weekly_orders = orders.set_index("order_date")["order_id"].resample("W").count()
busiest_week_end = weekly_orders.idxmax()
busiest_count = int(weekly_orders.max())

# %% [markdown]
# ### Exercise 5 (open-ended): Choose a threshold
# Using the anomaly detector of section 7, pick a threshold and justify it. Suppose a missed fault costs 50 times more than a false alarm.
# Plot or tabulate precision and recall for thresholds 2 to 6 and write three sentences in a markdown cell: which threshold you chose, what it costs you, and why
# real data (with no labels) would make this decision harder.

# %% [markdown]
# ## Recap
# - Parse timestamps, set a sorted `DatetimeIndex`, and slice with date strings.
# - Distinguish missing rows from missing values; measure gap lengths; choose a fill method by **testing it on hidden known values** and cap how far you fill.
# - `resample` aggregates by time bucket (match the aggregation to the quantity, keep a count); `rolling("24h")` is a true time window.
# - Seasonality: `groupby` on hour, weekday or month components; beware partial periods.
# - Store UTC, convert for display, and remember that local days can be 23 or 25 hours.
# - Anomaly detection is relative to a seasonal baseline; the threshold is a cost trade-off.
#
# **Go deeper:** pandas user guide "Time series / date functionality" and "Resampling" (pandas.pydata.org/docs/user_guide/timeseries.html); statsmodels time series documentation for decomposition and ARIMA.
#
# **Next:** [Pandas performance and best practices](08_pandas_performance_and_best_practices.ipynb)
