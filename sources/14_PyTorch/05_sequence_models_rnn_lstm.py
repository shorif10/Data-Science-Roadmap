# %% [markdown]
# # Sequence Models: RNN and LSTM on Sensor Data
#
# **Why this matters.** A facility manager wants to forecast next hour's power draw to schedule cooling and catch equipment that behaves unexpectedly. Time-ordered data breaks two habits from ordinary machine learning:
# you **cannot shuffle** it (the future must never inform the past), and the model must **carry memory** of what just happened. Recurrent networks such as the LSTM do the second part. This notebook does both
# carefully: leak-free windowing, honest baselines (a model that cannot beat "repeat the last value" is not useful), and a small LSTM.
#
# **You will learn**
# - turn a series into supervised windows `(past 24 hours -> next hour)` with no leakage, and split chronologically
# - see what leakage looks like in the numbers (random window split, off-by-one windows)
# - read an `nn.LSTM`: input `(batch, time, features)`, outputs, hidden and cell state, and its parameter count
# - compare an LSTM with naive, seasonal-naive and linear baselines, and measure the error on anomalies separately
#
# **Prerequisites:** [02 `nn.Module` and the training loop](02_nn_module_and_training_loop.ipynb) and [03 `Dataset`/`DataLoader` practice](03_classification_with_dataset_and_dataloader.ipynb).
#
# > Data used: `datasets/sensors.csv`, 4,320 hourly readings (January to June) of `temperature_c` and `power_kw` from a synthetic facility. About 60 temperature readings are missing (sensor dropouts) and about 0.6% of rows are flagged `is_anomaly`
# > (spikes). See `docs/DATASETS.md`.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.linear_model import Ridge

torch.manual_seed(42)
torch.set_num_threads(2)

df = pd.read_csv(Path("../datasets/sensors.csv"), parse_dates=["timestamp"])
print(df.shape, "| missing temperature readings:", int(df["temperature_c"].isna().sum()), "| anomalies:", int(df["is_anomaly"].sum()))
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(df["timestamp"][:24 * 7], df["power_kw"][:24 * 7])
ax.set_xlabel("time (first 7 days)")
ax.set_ylabel("power (kW)")
ax.set_title("Hourly power draw")
plt.show()

# %% [markdown]
# ## 1. Missing values without peeking at the future
#
# Sensor dropouts leave gaps in `temperature_c`. The common `interpolate()` draws a line between the readings *before and after* a gap, so the filled value uses **future** information. For a forecaster that is a small leak.
# Forward-fill (carry the last known reading) uses only the past, which is also what a live system could do.

# %%
df["temperature_c"] = df["temperature_c"].ffill().bfill()         # bfill only matters if the very first reading is missing
print("remaining missing:", int(df["temperature_c"].isna().sum()))

# %% [markdown]
# ## 2. Chronological split, train-only scaling
#
# Split by **time**: first 70% to train, next 15% to validate, last 15% to test. Scaling statistics come from the training period only. The model is asked: given the previous 24 hours of power and temperature,
# predict the next hour's power.

# %%
n = len(df)
i_train, i_val = int(0.70 * n), int(0.85 * n)
print("train:", df["timestamp"].iloc[0].date(), "->", df["timestamp"].iloc[i_train - 1].date())
print("val  :", df["timestamp"].iloc[i_train].date(), "->", df["timestamp"].iloc[i_val - 1].date())
print("test :", df["timestamp"].iloc[i_val].date(), "->", df["timestamp"].iloc[-1].date())

features = df[["power_kw", "temperature_c"]].to_numpy()
mu, sd = features[:i_train].mean(axis=0), features[:i_train].std(axis=0)           # TRAIN statistics only
scaled = ((features - mu) / sd).astype(np.float32)
power = df["power_kw"].to_numpy()
WINDOW = 24

# %% [markdown]
# ## 3. Windowing, carefully
#
# For a target at time `t`, the input is the rows `t-24 ... t-1` (strictly earlier), and the target is row `t`. A window belongs to a split according to its **target** time. Windows in the validation split may use
# inputs from the end of training (that is just the past, fine); what must never happen is a training target in the validation period.

# %%
def make_windows(arr, lo, hi, window=WINDOW):
    """Inputs arr[t-window:t] and target arr[t, 0] for every target time t in [lo, hi)."""
    idx = np.arange(max(lo, window), hi)
    X = np.stack([arr[t - window:t] for t in idx])
    y = arr[idx, 0]
    return torch.tensor(X), torch.tensor(y), idx


X_tr, y_tr, idx_tr = make_windows(scaled, 0, i_train)
X_va, y_va, idx_va = make_windows(scaled, i_train, i_val)
X_te, y_te, idx_te = make_windows(scaled, i_val, n)
print("X shapes (samples, time steps, features):", tuple(X_tr.shape), tuple(X_va.shape), tuple(X_te.shape))

# the no-leak test, written as an assertion: the last input step is exactly one hour before the target
t = idx_te[5]
assert np.allclose(X_te[5, -1].numpy(), scaled[t - 1]) and np.isclose(y_te[5].item(), scaled[t, 0])
assert X_te[5, :, 0].numpy().tolist() != [y_te[5].item()] * WINDOW
print("window 5 of the test set ends at t-1 and predicts t: OK")

# %% [markdown]
# ### What leakage looks like
#
# Two classic mistakes, measured with a cheap linear model (`Ridge` on the flattened 24x2 window) so the effect is not muddied by neural-network training noise. Error metric: mean absolute error (MAE) in kW.

# %%
def ridge_mae(Xa, ya, Xb, yb):
    r = Ridge(alpha=1.0).fit(Xa.reshape(len(Xa), -1), ya)
    return float(np.abs(r.predict(Xb.reshape(len(Xb), -1)) - yb).mean()) * sd[0]        # back to kW (error scale = std of power)


Xt_np, yt_np, Xv_np, yv_np = X_tr.numpy(), y_tr.numpy(), X_te.numpy(), y_te.numpy()
honest = ridge_mae(Xt_np, yt_np, Xv_np, yv_np)

# Mistake A: shuffle all windows randomly into train/test. Neighbouring windows overlap almost completely.
X_all = np.concatenate([Xt_np, X_va.numpy(), Xv_np])
y_all = np.concatenate([yt_np, y_va.numpy(), yv_np])
perm = np.random.default_rng(0).permutation(len(X_all))
cut = int(0.85 * len(X_all))
random_split = ridge_mae(X_all[perm[:cut]], y_all[perm[:cut]], X_all[perm[cut:]], y_all[perm[cut:]])

# Mistake B: off-by-one window that includes the target hour itself (arr[t-window+1 : t+1])
def make_leaky(arr, lo, hi, window=WINDOW):
    idx = np.arange(max(lo, window), hi)
    return np.stack([arr[t - window + 1:t + 1] for t in idx]), arr[idx, 0]


XL_tr, yL_tr = make_leaky(scaled, 0, i_train)
XL_te, yL_te = make_leaky(scaled, i_val, n)
off_by_one = ridge_mae(XL_tr, yL_tr, XL_te, yL_te)
print(f"honest chronological test MAE : {honest:6.2f} kW")
print(f"random window split MAE       : {random_split:6.2f} kW   <- neighbours overlap, so this is not a forecasting test")
print(f"off-by-one window MAE         : {off_by_one:6.2f} kW   <- 'too good': the answer is in the input")

# %% [markdown]
# *What just happened?* The off-by-one version is absurdly accurate because the target hour is part of its input; the random split is close to the honest number here (a linear model on a fairly stationary series is hardly hurt by overlap), but it is still the wrong protocol:
# a test window is nearly identical to training windows that surround it in time, and with a flexible model or a trending series the score would be flattered. **A forecast score that is much better than a naive baseline is a reason to hunt for a leak before celebrating.**
#
# ## 4. Baselines first
#
# Three cheap forecasts on the test period, all using only the past:
# - **Last value (persistence)**: next hour = this hour.
# - **Seasonal naive (24 h)**: next hour = the same hour yesterday.
# - **Ridge on the 24-hour window**: a linear model on the same inputs the LSTM will see.

# %%
def mae_kw(pred, idx):
    return float(np.abs(pred - power[idx]).mean())


baselines = {
    "train mean": mae_kw(np.full(len(idx_te), power[:i_train].mean()), idx_te),
    "last value": mae_kw(power[idx_te - 1], idx_te),
    "same hour yesterday": mae_kw(power[idx_te - 24], idx_te),
    "ridge on 24 h window": honest,
}
for name, v in baselines.items():
    print(f"{name:22s} test MAE {v:6.2f} kW")

# %% [markdown]
# "Last value" is the bar to clear. The series changes smoothly hour to hour but has a daily rhythm plus noise, so persistence is decent. Here yesterday's value is worse than persistence.
#
# ## 5. Reading an `nn.LSTM`
#
# An **RNN** reads the sequence one step at a time and keeps a hidden vector `h` that summarises what it has seen. A plain RNN's memory fades over many steps because gradients shrink when multiplied through time
# (vanishing gradients again). The **LSTM** adds a *cell state* `c`, a conveyor belt that information can ride with little change, plus three learned gates (forget, input, output) that decide what to erase, write and expose.
#
# Shapes with `batch_first=True`: input `(batch, time, features)`; `output` is `(batch, time, hidden)` (the hidden state after every step); `h_n` and `c_n` are `(layers, batch, hidden)` (the final states).

# %%
lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True)
out, (h_n, c_n) = lstm(X_tr[:5])
print("input  :", tuple(X_tr[:5].shape))
print("output :", tuple(out.shape), "| h_n:", tuple(h_n.shape), "| c_n:", tuple(c_n.shape))
print("output[:, -1] equals h_n[0]:", torch.allclose(out[:, -1], h_n[0]))
n_params = sum(p.numel() for p in lstm.parameters())
by_hand = 4 * (2 * 32 + 32 * 32 + 2 * 32)          # 4 gates x (input weights + recurrent weights + two bias vectors)
print("LSTM parameters:", n_params, "| by hand: 4 * (in*h + h*h + 2h) =", by_hand)

# %% [markdown]
# For forecasting we take the **last time step's** output (`out[:, -1]`, which has read the whole window) and map it to one number with a `Linear` layer.

# %%
class Forecaster(nn.Module):
    def __init__(self, n_features=2, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                              # x: (batch, time, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(1)        # (batch,)


print("parameters in the forecaster:", sum(p.numel() for p in Forecaster().parameters()))

# %% [markdown]
# ## 6. Training and evaluating
#
# MSE loss on the scaled target, Adam, mini-batches, and early stopping on the validation period (best-validation weights restored). Predictions are converted back to kW before computing the MAE.

# %%
import copy


def train_forecaster(seed, epochs=40, hidden=32, lr=1e-2):
    torch.manual_seed(seed)
    model = Forecaster(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, curve = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), generator=g)
        for i in range(0, len(X_tr), 128):
            ix = perm[i:i + 128]
            opt.zero_grad()
            loss_fn(model(X_tr[ix]), y_tr[ix]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_va), y_va).item()
        curve.append(val)
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, curve


@torch.no_grad()
def forecast_kw(model, X):
    return model(X).numpy() * sd[0] + mu[0]


models, curves, lstm_mae = [], [], []
for seed in range(3):
    m, c = train_forecaster(seed)
    models.append(m)
    curves.append(c)
    lstm_mae.append(mae_kw(forecast_kw(m, X_te), idx_te))
print("LSTM test MAE per seed (kW):", np.round(lstm_mae, 2))
print(f"LSTM mean {np.mean(lstm_mae):.2f} | last value {baselines['last value']:.2f} | ridge {baselines['ridge on 24 h window']:.2f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
for c in curves:
    ax.plot(c)
ax.set_xlabel("epoch")
ax.set_ylabel("validation MSE (scaled units)")
ax.set_title("LSTM validation loss, 3 seeds")
plt.show()

# %% [markdown]
# *What just happened?* The LSTM clearly beats persistence and the seasonal baseline in MAE. Compare it with the **ridge** row as well: a linear model on the same window gets very close (the LSTM is only a few percent better), which says most of the
# predictable structure is linear and the LSTM's extra machinery buys a modest gain. Always look at that row before claiming the recurrent model is "needed". The three seeds agree closely, so the gain is not luck.
#
# Look at the actual forecasts over three days of the test period:

# %%
best_model = models[int(np.argmin(lstm_mae))]
pred = forecast_kw(best_model, X_te)
sl = slice(0, 72)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(df["timestamp"].iloc[idx_te[sl]], power[idx_te[sl]], label="actual")
ax.plot(df["timestamp"].iloc[idx_te[sl]], pred[sl], label="LSTM one-step forecast")
ax.plot(df["timestamp"].iloc[idx_te[sl]], power[idx_te[sl] - 1], linestyle=":", label="last value")
ax.set_xlabel("time (first 72 h of the test period)")
ax.set_ylabel("power (kW)")
ax.set_title("One-step-ahead forecasts on unseen data")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# Note that a *one-step-ahead* forecast always sees the true previous hour, so even persistence looks deceptively good on a plot. A forecast several hours ahead is much harder and is scored differently (multi-step
# forecasting needs either direct multi-output training or feeding predictions back in). Do not quote one-step accuracy as if it were "we can predict tomorrow".
#
# ## 7. Where does the error come from?
#
# The data has planted spikes (`is_anomaly`). No forecaster can predict a random spike, so the error on those hours is huge per hour, even if there are few of them, and is best reported separately:

# %%
is_anom = df["is_anomaly"].to_numpy()[idx_te].astype(bool)
abs_err = np.abs(pred - power[idx_te])
print(f"test hours: {len(idx_te)}, anomalous: {int(is_anom.sum())}")
print(f"LSTM MAE on normal hours   : {abs_err[~is_anom].mean():6.2f} kW")
if is_anom.any():
    print(f"LSTM MAE on anomalous hours: {abs_err[is_anom].mean():6.2f} kW")
print(f"share of total absolute error from anomalies: {abs_err[is_anom].sum() / abs_err.sum():.1%}")

# %% [markdown]
# Here is a practical use: a forecaster that is good on normal hours becomes an **anomaly detector** when you flag hours whose forecast error is unusually large. That idea (forecast residuals as an alert signal) is how
# many monitoring systems work.

# %% [markdown]
# ## Common mistakes
#
# 1. **Random train/test split on a time series.** Neighbouring windows overlap, so the test looks like the training data. Split chronologically.
# 2. **Off-by-one windows** that include the target (or scaling with statistics from the whole series). Write an assertion that the last input step is strictly before the target.
# 3. **Filling gaps with interpolation** (uses the future). Forward-fill, or model the missingness.
# 4. **No baseline.** "Predict the last value" beats many fancy models on smooth series. Report it first.
# 5. **Wrong input shape.** `nn.LSTM` expects `(time, batch, features)` by default; pass `batch_first=True` if your data is `(batch, time, features)`, otherwise it silently treats batch as time.
# 6. **Using `out[:, 0]` or the wrong index** for the summary of the sequence. The state after the whole window is `out[:, -1]` (equivalently `h_n[-1]`).

# %%
# Mistake 5 in numbers: same tensor, batch_first forgotten -> runs without error but reads the wrong axis
wrong = nn.LSTM(2, 8)                                   # batch_first=False
o, _ = wrong(X_tr[:5])
print("batch_first forgotten: output shape", tuple(o.shape), "(it treated the 5 samples as 5 TIME STEPS of a 24-item batch)")

# %% [markdown]
# ## Interview corner
#
# **1. Why can't you shuffle a time series before splitting it?**
# <details><summary>Answer</summary>
# Shuffling puts future observations into the training set and makes test points neighbours of training points, so the model is scored on interpolation rather than forecasting. Split by time,
# and use rolling-origin (walk-forward) evaluation when you need several estimates.
# </details>
#
# **2. What problem does an LSTM solve compared with a plain RNN?**
# <details><summary>Answer</summary>
# Plain RNNs lose long-range information because gradients vanish or explode when backpropagated through many steps. The LSTM's cell state and gates give gradients a more direct path and let the network learn what to remember or forget.
# </details>
#
# **3. The LSTM barely beats the linear baseline. What do you conclude?**
# <details><summary>Answer</summary>
# Much of the signal is linear and short-range, so the simpler model is preferable (cheaper, more interpretable, easier to maintain). Consider better features (hour-of-day, day-of-week, lags, weather) before a bigger network.
# </details>
#
# **4. How do you forecast more than one step ahead?**
# <details><summary>Answer</summary>
# Recursive: feed each prediction back as input (errors compound). Direct: train one output per horizon (or a multi-output head). Sequence-to-sequence models also exist. Evaluate each horizon separately against a seasonal or persistence baseline.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell at the bottom of each exercise.

# %% [markdown]
# ### Exercise 1: Write a leak-free window function for one series
# Write `windows_1d(series, window)` returning `X` of shape `(len(series) - window, window)` and `y` of shape `(len(series) - window,)`, where `X[i]` is `series[i : i + window]` and `y[i]` is the value right after it, `series[i + window]`.
# *Hint: a list comprehension or `np.lib.stride_tricks.sliding_window_view`, but drop the last window (it has no target).*

# %% tags=["starter"]
def windows_1d(series, window):
    ...

# %%
s = np.arange(10, dtype=float) ** 2
Xw, yw = windows_1d(s, 3)
assert Xw.shape == (7, 3) and yw.shape == (7,), f"expected shapes (7, 3) and (7,), got {Xw.shape} and {yw.shape}"
assert np.allclose(Xw[0], [0, 1, 4]) and yw[0] == 9, "first window should be s[0:3] with target s[3]"
for i in range(len(yw)):
    assert np.allclose(Xw[i], s[i:i + 3]) and yw[i] == s[i + 3], f"window {i} is misaligned"
assert not any(yw[i] in Xw[i] for i in range(len(yw))), "a target must never appear inside its own input window"
print("Exercise 1 passed")

# %% tags=["solution"]
def windows_1d(series, window):
    series = np.asarray(series)
    X = np.stack([series[i:i + window] for i in range(len(series) - window)])
    y = series[window:]
    return X, y

# %% [markdown]
# ### Exercise 2: Choose the seasonal lag on validation, report on test
# For the lags `[1, 12, 24, 48]` compute the validation MAE (kW) of the forecast "the value `lag` hours before the target", over the validation targets `idx_va`. Store the best lag in `best_lag` and its **test** MAE (over `idx_te`) in `test_mae_best`.
# *Hint: `power[idx - lag]`; use the function `mae_kw`.*

# %% tags=["starter"]
best_lag = ...
test_mae_best = ...

# %%
val_scores = {lag: float(np.abs(power[idx_va - lag] - power[idx_va]).mean()) for lag in [1, 12, 24, 48]}
expected_lag = min(val_scores, key=val_scores.get)
assert best_lag == expected_lag, f"expected lag {expected_lag} (validation MAEs: { {k: round(v, 2) for k, v in val_scores.items()} }), got {best_lag}"
expected_test = float(np.abs(power[idx_te - expected_lag] - power[idx_te]).mean())
assert abs(test_mae_best - expected_test) < 1e-9, f"expected test MAE {expected_test:.3f}, got {test_mae_best:.3f}"
print(f"Exercise 2 passed (best lag {best_lag}, test MAE {test_mae_best:.2f} kW)")

# %% tags=["solution"]
val_mae = {lag: mae_kw(power[idx_va - lag], idx_va) for lag in [1, 12, 24, 48]}
best_lag = min(val_mae, key=val_mae.get)
test_mae_best = mae_kw(power[idx_te - best_lag], idx_te)

# %% [markdown]
# ### Exercise 3: Count the parameters of a stacked LSTM
# A model has `nn.LSTM(input_size=2, hidden_size=16, num_layers=2, batch_first=True)` followed by `nn.Linear(16, 1)`. Compute the total parameter count with the formula only (layer 1 has input size 2; layer 2 has input size 16),
# store it in `total_params`, and check against PyTorch. *Hint: each layer has `4 * (in*h + h*h + 2*h)` parameters.*

# %% tags=["starter"]
total_params = ...

# %%
stack = nn.Sequential(nn.LSTM(2, 16, num_layers=2, batch_first=True))
head = nn.Linear(16, 1)
expected = sum(p.numel() for p in stack.parameters()) + sum(p.numel() for p in head.parameters())
assert total_params == expected, f"expected {expected}, got {total_params}"
print("Exercise 3 passed:", total_params)

# %% tags=["solution"]
layer1 = 4 * (2 * 16 + 16 * 16 + 2 * 16)
layer2 = 4 * (16 * 16 + 16 * 16 + 2 * 16)
total_params = layer1 + layer2 + (16 * 1 + 1)

# %% [markdown]
# ### Exercise 4: How much do anomalies hurt the naive forecast?
# For the "last value" forecast on the test period, compute `mae_all` (all test hours) and `mae_normal` (only hours where `is_anomaly == 0`), in kW. Is the gap big? (Think about what a spike does to a forecast made from the previous hour, and to the hour after it.)

# %% tags=["starter"]
mae_all = ...
mae_normal = ...

# %%
flag = df["is_anomaly"].to_numpy()[idx_te] == 0
err = np.abs(power[idx_te - 1] - power[idx_te])
assert abs(mae_all - err.mean()) < 1e-9, "mae_all: mean absolute error of 'last value' over every test hour"
assert abs(mae_normal - err[flag].mean()) < 1e-9, "mae_normal: same, restricted to rows with is_anomaly == 0"
assert mae_normal <= mae_all, "removing the anomalies should not increase the error"
print(f"Exercise 4 passed (all {mae_all:.2f} kW, normal only {mae_normal:.2f} kW)")

# %% tags=["solution"]
flag = df["is_anomaly"].to_numpy()[idx_te] == 0
err = np.abs(power[idx_te - 1] - power[idx_te])
mae_all = float(err.mean())
mae_normal = float(err[flag].mean())

# %% [markdown]
# ### Exercise 5 (open-ended): Give the model better inputs
# Add two features to the window: `hour_of_day` as `sin` and `cos` of `2*pi*hour/24` (derive the hour from `df["timestamp"]`). Retrain `Forecaster(n_features=4)` for a few seeds and compare test MAE with the
# 2-feature model and with the ridge baseline. Does the LSTM need the extra features, given it sees 24 hours of history? Report what you measured. No assert: explore and report.

# %% [markdown]
# ## Recap
# - Time series: split chronologically, scale with training statistics, forward-fill gaps, and build windows whose inputs are strictly before the target.
# - Leaks announce themselves as scores that are too good: test with an assertion on window alignment and compare against a naive baseline.
# - Baselines first: last value, seasonal naive, and a linear model on the same window. The LSTM has to beat all three to justify itself.
# - `nn.LSTM` takes `(batch, time, features)` with `batch_first=True`; use `out[:, -1]` (or `h_n[-1]`) as the sequence summary; parameters per layer are `4 * (in*h + h*h + 2h)`.
# - One-step-ahead accuracy is not multi-step forecasting skill. Report error on anomalies separately; forecast residuals can drive alerts.
#
# **Go deeper:** the PyTorch docs for `torch.nn.LSTM` (pytorch.org/docs); Christopher Olah's blog post "Understanding LSTM Networks" (colah.github.io); "Forecasting: Principles and Practice" by Hyndman and Athanasopoulos (free online, otexts.com/fpp3).
#
# **Next:** [Saving, loading, reproducibility and devices](06_saving_loading_reproducibility_and_devices.ipynb)
