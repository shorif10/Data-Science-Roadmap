# %% [markdown]
# # Time-Series Forecasting Baselines
#
# **Why this matters.** "Forecast next week's demand" or "next day's power load" is among the most common data-science requests, and it is where ordinary machine-learning habits fail:
# random train/test splits leak the future, a fancy model often loses to "same as last week", and the popular percentage-error metric (MAPE) breaks the moment a value is zero.
# The professional workflow is: **build the dumbest forecasts first, evaluate by walking forward in time, and only then ask whether a model beats them.**
#
# **You will learn**
# - state a forecasting problem precisely: what is known at the forecast origin, and how far ahead (the horizon)
# - build naive, seasonal-naive, moving-average and seasonal-average baselines and score them with MAE, RMSE and sMAPE
# - turn a series into a supervised table with lag features, fit gradient boosting, and check it against the baselines
# - validate with `TimeSeriesSplit` and see what shuffled cross-validation does wrong
# - recognise when MAPE is unusable and what to report instead
#
# **Prerequisites:** [Cross-validation](07_cross_validation_and_model_evaluation.ipynb) (time-ordered splitting) and [Ensembles](06_ensembles_random_forest_and_gradient_boosting.ipynb).
#
# > Data used: `datasets/sensors.csv` (hourly power draw of a machine, strong daily and weekly pattern) and daily order counts from `datasets/orders.csv` (growing, noisy, small counts). Both are synthetic.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold, TimeSeriesSplit

DATA = Path("../datasets")
sensors = pd.read_csv(DATA / "sensors.csv", parse_dates=["timestamp"]).set_index("timestamp")
power = sensors["power_kw"]
print(power.index.min(), "->", power.index.max(), "| hourly, rows:", len(power), "| missing power values:", int(power.isna().sum()))

# %% [markdown]
# ## 1. Define the forecast before you touch a model
#
# **Problem.** Every day at midnight we must forecast the machine's power draw for each of the next 24 hours. So the *forecast origin* is midnight, the *horizon* is 1 to 24 hours ahead, and the only data we may use for hour `t` is what had been
# recorded **before midnight of that day**. Write that sentence down before anything else: it decides which features are legal.
#
# Look at the structure first:

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
power.iloc[: 24 * 14].plot(ax=axes[0])
axes[0].set_title("First two weeks of power draw")
axes[0].set_xlabel("time")
axes[0].set_ylabel("power (kW)")
power.groupby(power.index.dayofweek).mean().plot(kind="bar", ax=axes[1])
axes[1].set_title("Mean power by weekday (0 = Monday)")
axes[1].set_xlabel("weekday")
axes[1].set_ylabel("mean power (kW)")
plt.tight_layout()
plt.show()

# %% [markdown]
# A daily cycle (peak around midday), a lower level at weekends, and no obvious long-term trend. Those three facts predict which baselines will work. We hold out the **last 28 days** as the test period and evaluate
# as if we had stood at midnight each of those days.

# %%
y = power.to_numpy()
n = len(y)
TEST_H = 24 * 28
test_start = n - TEST_H
t = np.arange(test_start, n)  # integer positions of the hours we forecast
origin = (t // 24) * 24  # midnight of each target's day: the last moment we may look back from
print("test period:", power.index[test_start], "->", power.index[-1], "| every target's origin is before its own day begins:", bool((origin <= t).all()))

# %% [markdown]
# ## 2. Metrics by hand
#
# - **MAE** = mean of `|actual - forecast|`: in kW, easy to explain.
# - **RMSE** = square root of the mean squared error: punishes big misses more.
# - **sMAPE** = mean of `2|a - f| / (|a| + |f|)`: a relative error that stays bounded. (Other definitions exist; say which you use.)
# - **MAPE** = mean of `|a - f| / |a|`: fine when actuals are large and far from zero; dangerous otherwise (section 7).

# %%
def mae(a, f):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(f))))


def rmse(a, f):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(f)) ** 2)))


def smape(a, f):
    a, f = np.asarray(a, dtype=float), np.asarray(f, dtype=float)
    denom = np.abs(a) + np.abs(f)
    return float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(a - f) / np.where(denom == 0, 1, denom))))


def mape(a, f):
    a, f = np.asarray(a, dtype=float), np.asarray(f, dtype=float)
    return float(np.mean(np.abs(a - f) / np.abs(a)))


def report(actual, forecasts):
    rows = {name: {"MAE": mae(actual, f), "RMSE": rmse(actual, f), "sMAPE %": 100 * smape(actual, f), "MAPE %": 100 * mape(actual, f)} for name, f in forecasts.items()}
    return pd.DataFrame(rows).T.sort_values("MAE")


# %% [markdown]
# ## 3. Baselines: the dumb forecasts every model must beat
#
# | baseline | forecast for hour `t` |
# |---|---|
# | naive | the last value seen before the origin (repeated for all 24 hours) |
# | mean of last 24 h | the average of the day before the origin (a flat line) |
# | seasonal naive, 24 h | the same hour yesterday (`y[t-24]`) |
# | seasonal naive, 168 h | the same hour, same weekday, last week (`y[t-168]`) |
# | seasonal average | the mean of the same hour over the last 7 days |
#
# Every index used is at or before the origin, so none of them peeks.

# %%
forecasts = {
    "naive (last value)": y[origin - 1],
    "mean of last 24h": np.array([y[o - 24 : o].mean() for o in origin]),
    "seasonal naive 24h": y[t - 24],
    "seasonal naive 168h": y[t - 168],
    "seasonal avg (7 days)": np.mean([y[t - 24 * k] for k in range(1, 8)], axis=0),
}
baseline_table = report(y[t], forecasts)
baseline_table.round(2)

# %% [markdown]
# *What just happened?* Ranking is driven by structure. "Naive" and the flat 24-hour mean ignore the daily cycle and lose badly. "Same hour yesterday" captures the cycle, but it is wrong every Monday and Saturday because of the weekend effect; "same hour last week"
# fixes that and is hard to beat with no model at all. **Pick the baseline whose period matches the seasonality of the data.** Averaging the same hour over the last 7 days smooths noise but blends weekdays with weekend days, which adds bias, so it still loses to last week's same weekday. Whatever model we build next has to beat the best row of this table, or it is not worth deploying.

# %%
day = pd.Timestamp(power.index[test_start]) + pd.Timedelta(days=9)  # one test day, to see the forecasts
k = int(np.flatnonzero(power.index == day)[0]) - test_start
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(range(24), y[test_start + k : test_start + k + 24], "k-o", label="actual")
ax.plot(range(24), forecasts["seasonal naive 168h"][k : k + 24], "--", label="seasonal naive 168h")
ax.plot(range(24), forecasts["seasonal naive 24h"][k : k + 24], "--", label="seasonal naive 24h")
ax.plot(range(24), forecasts["mean of last 24h"][k : k + 24], ":", label="mean of last 24h")
ax.set_title(f"One test day ({day.date()}): actual vs baselines")
ax.set_xlabel("hour of day")
ax.set_ylabel("power (kW)")
ax.legend()
plt.show()

# %% [markdown]
# ## 4. Lag features and gradient boosting
#
# To use a general model, turn the series into an ordinary table: one row per target hour `t`, columns built from **past values that would be known at the origin**. Because our horizon reaches 24 hours ahead, every lag must be **at least 24 hours**
# (a 1-hour lag would be unavailable for the 10 pm forecast made at midnight). We add the calendar (hour, weekday), a few lags and two rolling summaries.

# %%
def make_features(series):
    d = pd.DataFrame({"y": series})
    for lag in (24, 48, 168):
        d[f"lag_{lag}"] = series.shift(lag)
    d["same_hour_mean_7d"] = pd.concat([series.shift(24 * j) for j in range(1, 8)], axis=1).mean(axis=1)
    d["mean_prev_day"] = series.shift(24).rolling(24).mean()  # the 24 h window that ENDS at t-24
    d["hour"] = series.index.hour
    d["weekday"] = series.index.dayofweek
    return d.dropna()


feat = make_features(power)
X_all, y_all = feat.drop(columns="y"), feat["y"]
is_train = feat.index < power.index[test_start]
gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=42).fit(X_all[is_train], y_all[is_train])
gb_pred = gb.predict(X_all[~is_train])
print("rows: train", int(is_train.sum()), "| test", int((~is_train).sum()), "| features:", list(X_all.columns))

models = dict(forecasts)
models["gradient boosting (lags)"] = gb_pred
report(y[t], models).round(2)

# %% [markdown]
# *What just happened?* Two things to notice. One, the boosted model beats even the best baseline here because it blends several lags and the calendar and averages away noise. Two, **the margin over "same hour last week" is modest** compared with the big
# gap between naive and seasonal-naive. Most of the value came from recognising the seasonality, not from the model. That is typical, and it is why the baseline table goes in every forecasting report. (The test hours are the last 28 days; nothing in the training rows comes after
# `test_start`, and the features of each row only look at least 24 hours back.)

# %% [markdown]
# ## 5. Validate by walking forward: `TimeSeriesSplit`
#
# One 28-day test window is one sample of "how good are we?". Walk-forward (expanding window) validation repeats the exercise: train on an initial stretch, test on the next block, then extend the training window and test on the following block.
# `gap` leaves a buffer between train and test, useful when features use rolling windows that overlap the boundary.

# %%
def walk_forward(model_factory, X, y_, n_splits=4, test_size=24 * 14, gap=24):
    scores = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap).split(X):
        m = model_factory().fit(X.iloc[tr], y_.iloc[tr])
        scores.append(mae(y_.iloc[te], m.predict(X.iloc[te])))
    return np.array(scores)


make_gb = lambda: HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3, random_state=42)
wf = walk_forward(make_gb, X_all, y_all)
print("walk-forward MAE per fold (kW):", wf.round(2), "| mean", wf.mean().round(2), "+-", wf.std().round(2))
same_week = [mae(y_all.iloc[te], X_all["lag_168"].iloc[te]) for _, te in TimeSeriesSplit(n_splits=4, test_size=24 * 14, gap=24).split(X_all)]
print("seasonal naive 168h on the same folds  :", np.round(same_week, 2), "| mean", np.mean(same_week).round(2))

# %% [markdown]
# Look at the per-fold numbers instead of trusting one mean: the model beats seasonal naive on every fold here (compare the two lists), which makes the claim sturdier than a win on average only. The spread between folds (about a third of a kW) is the honest size of the uncertainty.

# %% [markdown]
# ## 6. A second series: daily orders, where baselines are hard to beat
#
# Now count orders per day from `orders.csv`. This series is small-count, noisy, **growing** and has no weekday pattern. Repeat the workflow with a 7-day horizon: at the start of each week, forecast the 7 days ahead.

# %%
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
daily = orders.groupby("order_date").size().asfreq("D", fill_value=0)
print("days:", len(daily), "| mean orders/day:", round(daily.mean(), 1), "| days with zero orders:", int((daily == 0).sum()))
print("mean orders/day, by weekday (0=Mon):", daily.groupby(daily.index.dayofweek).mean().round(1).tolist())
print("mean orders/day: first 180 days", round(daily.iloc[:180].mean(), 1), "| last 180 days", round(daily.iloc[-180:].mean(), 1))

# %%
yd = daily.to_numpy()
nd, TD = len(yd), 180
td_start = nd - TD
td = np.arange(td_start, nd)
od = td_start + (td - td_start) // 7 * 7  # forecast origin = start of each week
daily_fc = {
    "naive (last day)": yd[od - 1],
    "mean of last 7 days": np.array([yd[o - 7 : o].mean() for o in od]),
    "mean of last 28 days": np.array([yd[o - 28 : o].mean() for o in od]),
    "seasonal naive 7d": yd[td - 7],
    "same day last year (364d)": yd[td - 364],
}
report(yd[td], daily_fc).round(2)

# %% [markdown]
# *What just happened?* Here recency beats seasonality: with no weekday pattern, "same day last week" is just a noisier copy of the recent level, and "same day last year" is badly wrong because the level grew a lot (compare the first-180-day and last-180-day means printed above). (Seasonal patterns that do exist, such as the
# November/December peak, need more than two years of history to exploit.) Now fit the same lag-feature gradient boosting and compare:

# %%
fd = pd.DataFrame({"y": daily})
for lag in (7, 14, 21, 28):
    fd[f"lag_{lag}"] = daily.shift(lag)
fd["mean_7"] = daily.shift(7).rolling(7).mean()  # ends 7 days before the target, so it is known at the origin
fd["mean_28"] = daily.shift(7).rolling(28).mean()
fd["weekday"], fd["month"] = daily.index.dayofweek, daily.index.month
fd = fd.dropna()
Xo, yo = fd.drop(columns="y"), fd["y"]
tr_o = fd.index < daily.index[td_start]
gb_o = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=42).fit(Xo[tr_o], yo[tr_o])
daily_models = dict(daily_fc)
daily_models["gradient boosting (lags)"] = gb_o.predict(Xo[~tr_o])
report(yd[td], daily_models).round(2)

# %% [markdown]
# The boosted model does not beat the simple "last value" baseline on MAE (it looks better on the two percentage metrics, a hint about how differently the metrics treat small counts: section 7), and that is an honest result: a tree model cannot extrapolate a growing level (notebook 05), and there is little structure beyond noise to learn. Do not hide such a result. "We tried a model and the baseline is as good" is a
# useful finding that saves maintenance cost. Options to try next: predict the ratio to a recent average so the trend is removed, or add promotions and calendar events as features.

# %% [markdown]
# ## 7. When MAPE breaks
#
# Orders can be zero. MAPE divides by the actual value, so a zero makes it infinite, and small counts make it explode:

# %%
a, f = yd[28:], yd[27:-1]  # every day from index 28 onward; forecast = yesterday's count
print("zero-order days in the evaluation window:", int((a == 0).sum()))
print("sklearn MAPE with zeros present        :", f"{mean_absolute_percentage_error(a, f):.3g}", "(it clips the denominator to a tiny number, so the result is meaningless)")
nz = a > 0
print(f"MAPE dropping zero days                : {100 * mape(a[nz], f[nz]):.0f}%  (silently ignores the days that mattered most)")
print(f"sMAPE (bounded, handles zero actual)   : {100 * smape(a, f):.0f}%")
print(f"MAE (orders/day)                       : {mae(a, f):.2f}")
print(f"WAPE = sum |error| / sum actual        : {100 * np.abs(a - f).sum() / a.sum():.0f}%")

# %% [markdown]
# Two more reasons MAPE misleads. It treats a 100% miss on a 1-order day exactly like a 100% miss on a 40-order day, so a handful of quiet days (where missing by one order is a huge percentage) dominate. And it **rewards under-forecasting**: the best constant forecast under MAPE is lower than the best under MAE.
# Check it:

# %%
window = yd[-180:]
consts = np.arange(1, 30, 0.25)
best_mae_c = consts[np.argmin([mae(window, c) for c in consts])]
best_mape_c = consts[np.argmin([mape(window, c) for c in consts])]
print(f"constant forecast minimising MAE: {best_mae_c:.2f} orders/day; minimising MAPE: {best_mape_c:.2f} (median of the window: {np.median(window):.1f}, mean: {window.mean():.1f})")

# %% [markdown]
# *What just happened?* MAE is minimised near the median; MAPE is minimised at a *lower* value, so optimising it makes a model deliberately forecast low. For volumes that can reach zero, report **MAE** (in units), **WAPE** (total error as a share of total volume)
# or **sMAPE**, and say which one. For stakeholders who insist on a percentage, WAPE is usually the defensible one.

# %% [markdown]
# ## 8. Shuffled cross-validation lies
#
# Rows in a time series are neighbours in time. With ordinary shuffled K-fold, each test day is surrounded by training days from the same weeks (and with a time-index feature the model can memorise the level of each period), so you are
# evaluating interpolation, not forecasting. Compare with walk-forward on the daily orders (`Xo`, `yo`), adding a simple day-counter feature that gives the model a timeline:

# %%
Xo_t = Xo.assign(day_index=np.arange(len(Xo)))
make_gb_o = lambda: HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=3, random_state=42)
wf_o = walk_forward(make_gb_o, Xo_t, yo, n_splits=4, test_size=60, gap=7)
kf_o = []
for tr, te in KFold(5, shuffle=True, random_state=42).split(Xo_t):
    kf_o.append(mae(yo.iloc[te], make_gb_o().fit(Xo_t.iloc[tr], yo.iloc[tr]).predict(Xo_t.iloc[te])))
print(f"walk-forward MAE per fold: {np.round(wf_o, 2)} -> mean {wf_o.mean():.2f}")
print(f"shuffled 5-fold MAE       : {np.round(kf_o, 2)} -> mean {np.mean(kf_o):.2f}")
kf_h = [mae(y_all.iloc[te], make_gb().fit(X_all.iloc[tr], y_all.iloc[tr]).predict(X_all.iloc[te])) for tr, te in KFold(5, shuffle=True, random_state=42).split(X_all)]
print(f"hourly power, for contrast: walk-forward mean {wf.mean():.2f} vs shuffled 5-fold mean {np.mean(kf_h):.2f}")

# %% [markdown]
# Shuffled CV reports a clearly better score than the walk-forward one. Only the second number answers the question the business is asking ("how well will it do *next*?"). On the hourly power data (last printed line) the two schemes come out much closer, because the series has no trend and all its features are
# already lags; a method that happens to be harmless in one dataset is still the wrong default.

# %% [markdown]
# ## Common mistakes
#
# 1. **Random train/test split or shuffled K-fold** on time-ordered data (section 8).
# 2. **Features that are unavailable at forecast time**: a lag shorter than the horizon, a rolling window that includes the target row, or the *actual* future temperature instead of a forecast of it. Test availability (below), do not just trust the score.
# 3. **No baseline**, or a weak one (naive when the data has a weekly cycle).
# 4. **MAPE on series with zeros or small values**; or comparing models on different metrics than the business cares about.
# 5. **Scaling or imputing with statistics of the whole series** before splitting (same leak as notebook 02).
# 6. **Expecting trees to extrapolate a trend**, or ignoring that the level changed.

# %%
# Mistake 2: how do you KNOW a feature is legal? Test it: corrupt the series from position k onward; a feature row for a target at position t
# may only change if the target is at least 24 hours past k (its newest input is at t-24). If earlier rows change, the feature peeks at the future.
def availability_test(build, series, k=3000, horizon=24):
    bad = series.copy()
    bad.iloc[k:] = bad.iloc[k:] + 1000.0
    a, b = build(series), build(bad)
    rows = a.index < series.index[k + horizon]
    return bool(np.allclose(a.loc[rows].drop(columns="y"), b.loc[rows].drop(columns="y")))


def build_leaky(series):
    d = make_features(series)
    d["roll3_with_now"] = series.rolling(3).mean().reindex(d.index)  # window = t-2, t-1, t  <- includes y_t itself
    return d


print("make_features is legal for a 24 h horizon:", availability_test(make_features, power))
print("the rolling mean that includes 'now' is legal:", availability_test(build_leaky, power), "  <- it peeks at the target")

# %% [markdown]
# *What just happened?* The test changes the future and checks that the past does not notice; it is a cheap unit test worth keeping next to any forecasting feature code. (A leaky feature does not always raise the score visibly, especially on a noisy series, which is why you test availability
# rather than trusting the validation number.)

# %% [markdown]
# ## Interview corner
#
# **1. How do you evaluate a forecasting model?**
# <details><summary>Answer</summary>
# Walk forward in time (expanding or rolling window), evaluate at the same horizon used in production, compare against naive and seasonal-naive baselines on the same origins, and report an error metric in business units (MAE/RMSE) plus a
# relative one that is safe for the data (sMAPE or WAPE). Shuffled cross-validation is invalid because it lets the model see the future.
# </details>
#
# **2. Your daily demand has a weekly cycle. What is the first baseline you build?**
# <details><summary>Answer</summary>
# Seasonal naive with period 7: forecast each day with the same weekday last week (optionally the average of the last few weeks). It is easy to explain, captures the main structure, and any model must beat it.
# </details>
#
# **3. What is wrong with MAPE?**
# <details><summary>Answer</summary>
# It is undefined or infinite when actuals are zero, explodes for small actuals, is asymmetric (it favours forecasts that are too low), and overweights low-volume periods. Alternatives: MAE, RMSE, WAPE, sMAPE (with its own quirks), or MASE (error scaled
# by the naive baseline's error).
# </details>
#
# **4. How do you choose lag features for a multi-step forecast?**
# <details><summary>Answer</summary>
# Use only lags at least as long as the horizon (for a 24-hour horizon, lag 24 and beyond), or use a recursive approach that feeds its own predictions back (which accumulates error), or a separate model per horizon (the direct approach). Include calendar features and
# rolling summaries that end before the forecast origin.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: The weekly baseline
# Using `y`, `t` and the `mae` function above, compute the MAE over the test period of the forecast "same hour one week earlier" (`y[t - 168]`) and store it in `mae_weekly`. Then store in `best_baseline` the **name** (a key of `forecasts`) of the baseline
# with the lowest MAE on the test period, found by computing it yourself from `forecasts` rather than copying the table.

# %% tags=["starter"]
mae_weekly = ...
best_baseline = ...

# %%
_scores = {k: float(np.mean(np.abs(y[t] - v))) for k, v in forecasts.items()}
assert abs(mae_weekly - float(np.mean(np.abs(y[t] - y[t - 168])))) < 1e-9, "mean of |y[t] - y[t-168]| over the test positions t"
assert best_baseline == min(_scores, key=_scores.get), f"name of the baseline with the lowest MAE; expected a key of forecasts such as {min(_scores, key=_scores.get)!r}"
print(f"Exercise 1 passed: weekly seasonal naive MAE {mae_weekly:.2f} kW; best baseline: {best_baseline}")

# %% tags=["solution"]
mae_weekly = float(np.mean(np.abs(y[t] - y[t - 168])))
best_baseline = min(forecasts, key=lambda name: np.mean(np.abs(y[t] - forecasts[name])))

# %% [markdown]
# ### Exercise 2: Write sMAPE and use it
# Define your own `my_smape(actual, forecast)` returning the **mean of `2*|a-f| / (|a|+|f|)`** as a fraction (0.1 means 10%), treating a point where both are zero as zero error (not `nan`). Store its value for the gradient-boosting forecast `gb_pred` on the
# hourly test targets `y_all[~is_train]` in `smape_gb`.
# *Hint: `np.where` with a safe denominator.*

# %% tags=["starter"]
def my_smape(actual, forecast):
    ...


smape_gb = ...

# %%
_a, _f = np.array([0.0, 10.0, 5.0, 0.0]), np.array([0.0, 5.0, 5.0, 3.0])
_exp = np.mean([0.0, 2 * 5 / 15, 0.0, 2 * 3 / 3])
assert abs(my_smape(_a, _f) - _exp) < 1e-9, "both zero -> 0 error; |a-f|=3 with a=0,f=3 -> 2*3/3 = 2"
assert abs(my_smape(np.array([100.0]), np.array([100.0]))) < 1e-12, "a perfect forecast has zero error"
assert abs(smape_gb - smape(y_all[~is_train], gb_pred)) < 1e-9, "use y_all[~is_train] as actuals and gb_pred as the forecast"
print(f"Exercise 2 passed: sMAPE of the boosted model is {100 * smape_gb:.2f}%")

# %% tags=["solution"]
def my_smape(actual, forecast):
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = np.abs(a) + np.abs(f)
    return float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(a - f) / np.where(denom == 0, 1.0, denom))))


smape_gb = my_smape(y_all[~is_train], gb_pred)

# %% [markdown]
# ### Exercise 3: Count the damage MAPE does
# For the daily orders, forecast each day with **the previous day's count** and evaluate over every day from index 28 onward (as in section 7: `a = yd[28:]`, forecast `yd[27:-1]`). Store in `n_zero_actual` the number of zero-order days in that window,
# in `mape_nonzero` the MAPE as a **fraction** after dropping the zero-actual days, and in `wape` the weighted absolute percentage error `sum|a-f| / sum(a)` as a fraction.

# %% tags=["starter"]
n_zero_actual = ...
mape_nonzero = ...
wape = ...

# %%
_a, _f = yd[28:], yd[27:-1]
_nz = _a > 0
assert n_zero_actual == int((_a == 0).sum()), "count days where the actual is zero"
assert n_zero_actual > 0, "there must be zero days in this window, that is the point of the exercise"
assert abs(mape_nonzero - np.mean(np.abs(_a[_nz] - _f[_nz]) / _a[_nz])) < 1e-9, "mean(|a-f|/a) over days with a > 0, as a fraction"
assert abs(wape - np.abs(_a - _f).sum() / _a.sum()) < 1e-9, "sum of absolute errors divided by the sum of actuals"
print(f"Exercise 3 passed: {n_zero_actual} zero days; MAPE(nonzero) {mape_nonzero:.0%}; WAPE {wape:.0%}")

# %% tags=["solution"]
a_ex, f_ex = yd[28:], yd[27:-1]
n_zero_actual = int((a_ex == 0).sum())
keep = a_ex > 0
mape_nonzero = float(np.mean(np.abs(a_ex[keep] - f_ex[keep]) / a_ex[keep]))
wape = float(np.abs(a_ex - f_ex).sum() / a_ex.sum())

# %% [markdown]
# ### Exercise 4: Quantify how optimistic shuffled CV is
# For the daily-orders feature table `Xo_t, yo` (with the `day_index` column), compute the **mean MAE** of the same model `make_gb_o()` under (a) walk-forward (`TimeSeriesSplit(n_splits=4, test_size=60, gap=7)`) into `ts_mae` and (b) shuffled `KFold(5, shuffle=True, random_state=42)` into `kfold_mae`.
# Store `optimism` = how much lower (better) the shuffled number is, as a fraction of the walk-forward one: `(ts_mae - kfold_mae) / ts_mae`.

# %% tags=["starter"]
ts_mae = ...
kfold_mae = ...
optimism = ...

# %%
_ts = np.mean([mae(yo.iloc[te], make_gb_o().fit(Xo_t.iloc[tr], yo.iloc[tr]).predict(Xo_t.iloc[te])) for tr, te in TimeSeriesSplit(n_splits=4, test_size=60, gap=7).split(Xo_t)])
_kf = np.mean([mae(yo.iloc[te], make_gb_o().fit(Xo_t.iloc[tr], yo.iloc[tr]).predict(Xo_t.iloc[te])) for tr, te in KFold(5, shuffle=True, random_state=42).split(Xo_t)])
assert abs(ts_mae - _ts) < 1e-9, "mean of the four walk-forward fold MAEs"
assert abs(kfold_mae - _kf) < 1e-9, "mean of the five shuffled-fold MAEs, random_state=42"
assert abs(optimism - (_ts - _kf) / _ts) < 1e-9, "(ts_mae - kfold_mae) / ts_mae"
assert optimism > 0.05, "shuffled CV should look clearly better than walk-forward here"
print(f"Exercise 4 passed: walk-forward {ts_mae:.2f} vs shuffled {kfold_mae:.2f}: shuffled CV flatters the model by {optimism:.0%}")

# %% tags=["solution"]
ts_folds = [mae(yo.iloc[te], make_gb_o().fit(Xo_t.iloc[tr], yo.iloc[tr]).predict(Xo_t.iloc[te])) for tr, te in TimeSeriesSplit(n_splits=4, test_size=60, gap=7).split(Xo_t)]
kf_folds = [mae(yo.iloc[te], make_gb_o().fit(Xo_t.iloc[tr], yo.iloc[tr]).predict(Xo_t.iloc[te])) for tr, te in KFold(5, shuffle=True, random_state=42).split(Xo_t)]
ts_mae, kfold_mae = float(np.mean(ts_folds)), float(np.mean(kf_folds))
optimism = (ts_mae - kfold_mae) / ts_mae

# %% [markdown]
# ### Exercise 5 (open-ended): Would you deploy the boosted hourly model?
# Write a short recommendation for the operations team. Compare the boosted model with "same hour last week" using the walk-forward fold table from section 5: how much better is it in kW and in percent, is it better on every fold, and what is the maintenance cost of the extra model
# (features, retraining, monitoring) against that gain? Name one scenario (for example a holiday or a sensor fault) where you expect each approach to fail.

# %% [markdown]
# ## Recap
# - Define origin, horizon and what is known at the origin first; that dictates which features are legal.
# - Always start with naive and seasonal-naive baselines; the one matching the data's period is often hard to beat.
# - Lag features must be at least as old as the horizon; rolling windows must end before the origin.
# - Validate by walking forward (`TimeSeriesSplit`), never by shuffling; look at per-fold scores.
# - MAPE fails on zeros and small values and rewards under-forecasting: report MAE plus WAPE or sMAPE, and state which.
# - A model that ties the baseline is a legitimate result: report it.
#
# **Go deeper:** scikit-learn user guide on `TimeSeriesSplit` and on time-related feature engineering (the example gallery has a lag-features forecasting example); "Forecasting: Principles and Practice" by Hyndman and Athanasopoulos (free online) for baselines and accuracy metrics; `statsmodels` documentation for ETS and ARIMA, the classical next step.
#
# **Next:** [Anomaly detection](15_anomaly_detection.ipynb)
