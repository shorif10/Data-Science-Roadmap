# %% [markdown]
# # `nn.Module` and the Training Loop
#
# **Why this matters.** A real-estate team wants a model that estimates a house price from its size, age, location and quality.
# In PyTorch that job is always the same skeleton: prepare the data without leaking information, define a model as an
# `nn.Module`, run the five-step training loop, watch the loss curves, and then **judge the result in dollars against a simple
# baseline**. The skeleton is the same for a three-layer network and for a giant transformer, so learning it properly once pays off every time.
#
# **You will learn**
# - prepare tabular data for a network: clean bad labels, impute, one-hot encode, scale using **training data only**
# - write a model as an `nn.Module` and explain what `parameters()` and `forward()` do
# - run the 5-step loop (forward, loss, `zero_grad`, `backward`, `step`) with mini-batches and read train/validation curves
# - evaluate in original units (dollars) and compare honestly with `LinearRegression` and a mean-only baseline
#
# **Prerequisites:** [Tensors and autograd](01_tensors_and_autograd.ipynb) and
# [Backpropagation from scratch](../13_Deep_Learning/02_backpropagation_from_scratch.ipynb).
#
# > Data used: `datasets/housing.csv`, a synthetic table of 3,000 houses (`price` is the target). It has missing `age_years` and
# > `quality_score` values and about 20 absurd prices, both planted on purpose. See `docs/DATASETS.md`.

# %%
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

torch.manual_seed(42)
torch.set_num_threads(2)
rng = np.random.default_rng(42)

df = pd.read_csv(Path("../datasets/housing.csv"))
print(df.shape)
df.head(3)

# %% [markdown]
# ## 1. Clean the labels first
#
# A network will happily try to fit a price of -25,700 or of 3 million for a 1,600 sqft house. Squared-error loss makes it worse: one absurd row
# can dominate the gradient. Let us look at **price per square foot**, which should be fairly stable, to find impossible rows.

# %%
pps = df["price"] / df["area_sqft"]
print(pps.describe(percentiles=[0.01, 0.5, 0.99]).round(1).to_string())
print("non-positive prices:", int((df["price"] <= 0).sum()))

# %% [markdown]
# The bulk sits in a band, with a few rows far outside it on both sides. We will drop rows whose price is not positive or whose price per sqft is outside `[20, 400]`.
# That threshold is a judgement call made by looking at the distribution, not a law; in a real project you would confirm such rows with the data owner.

# %%
valid = (df["price"] > 0) & pps.between(20, 400)
print(f"removing {(~valid).sum()} of {len(df)} rows")
data = df[valid].reset_index(drop=True)
print("price range after cleaning:", data["price"].min(), "-", data["price"].max())

# %% [markdown]
# *What just happened?* We filtered on the **target** and on an input together. That is acceptable for *training* data. Be aware that it also
# cleans our evaluation data, so the numbers below describe performance on plausible records, and a deployed model would still meet bad inputs.
#
# ## 2. Split first, then fit the preprocessing on the training part only
#
# Anything that learns from data (medians for imputation, means and standard deviations for scaling, the price mean) must be learned from the **training**
# rows and merely *applied* to validation and test rows. Otherwise information about the held-out rows leaks in.
# We use three parts: train (fit weights), validation (choose when to stop), test (one final, honest score).

# %%
train_val, test_df = train_test_split(data, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val, test_size=0.25, random_state=42)
print("train / val / test:", len(train_df), len(val_df), len(test_df))

num_cols = ["area_sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "distance_to_center_km", "quality_score"]
preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["neighborhood"]),
])
X_train = preprocess.fit_transform(train_df).astype(np.float32)      # fit on train ONLY
X_val = preprocess.transform(val_df).astype(np.float32)              # only transform
X_test = preprocess.transform(test_df).astype(np.float32)
print("feature matrix:", X_train.shape, X_train.dtype, "| NaNs left:", int(np.isnan(X_train).sum()))

# %% [markdown]
# The target needs scaling too. A network starting near zero output on prices around a quarter million would spend its first epochs just learning the scale.
# We standardise the price with the **training** mean and standard deviation, and convert back to dollars whenever we report a result.

# %%
price_mean, price_std = train_df["price"].mean(), train_df["price"].std()


def scale_price(s):
    return torch.tensor(((s - price_mean) / price_std).to_numpy(), dtype=torch.float32)


y_train, y_val, y_test = scale_price(train_df["price"]), scale_price(val_df["price"]), scale_price(test_df["price"])
X_train_t, X_val_t, X_test_t = map(torch.from_numpy, (X_train, X_val, X_test))
print(f"price mean {price_mean:,.0f}  std {price_std:,.0f}   -> y_train mean {y_train.mean():.3f}, std {y_train.std():.3f}")

# %% [markdown]
# ## 3. A model is an `nn.Module`
#
# Subclass `nn.Module`, create layers in `__init__` (PyTorch registers them, so `model.parameters()` finds every weight) and describe the computation in
# `forward`. You call the model like a function, `model(x)`, never `model.forward(x)` directly (the call runs hooks around `forward`).

# %%
class PriceNet(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.body(x).squeeze(-1)          # (n, 1) -> (n,) so it matches the target shape


model = PriceNet(X_train.shape[1])
print(model)
n_params = sum(p.numel() for p in model.parameters())
print("trainable parameters:", n_params, "| training rows:", len(X_train))
print("output shape for 5 rows:", model(X_train_t[:5]).shape)

# %% [markdown]
# *What just happened?* The model has about as many parameters as we have training rows (compare the two numbers above), yet this is a small network. Keep that in mind: it can memorise the training set, and the validation curve will tell us if it does.
#
# ## 4. The five-step training loop
#
# Every PyTorch training loop is these five lines, in this order, for every mini-batch:
#
# 1. **forward**: `pred = model(xb)`
# 2. **loss**: `loss = loss_fn(pred, yb)`
# 3. **clear** old gradients: `optimizer.zero_grad()`
# 4. **backward**: `loss.backward()`
# 5. **step**: `optimizer.step()`
#
# Around it: shuffle the rows each epoch, call `model.train()` before training and `model.eval()` + `torch.no_grad()` when measuring.
# We write it once as a function and reuse it. It also keeps a copy of the weights from the epoch with the lowest validation loss
# (simple **early stopping by checkpoint**) and restores them at the end.

# %%
def fit(model, X_tr, y_tr, X_va, y_va, epochs=40, lr=3e-3, batch_size=64, weight_decay=0.0, seed=42):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    history = {"train": [], "val": []}
    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X_tr), generator=gen)
        for i in range(0, len(X_tr), batch_size):
            idx = order[i:i + batch_size]
            pred = model(X_tr[idx])                   # 1. forward
            loss = loss_fn(pred, y_tr[idx])           # 2. loss
            optimizer.zero_grad()                     # 3. clear gradients
            loss.backward()                           # 4. backward
            optimizer.step()                          # 5. update
        model.eval()
        with torch.no_grad():
            tr_loss = loss_fn(model(X_tr), y_tr).item()
            va_loss = loss_fn(model(X_va), y_va).item()
        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        if va_loss < best_val:
            best_val, best_epoch, best_state = va_loss, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return history


torch.manual_seed(42)
net = PriceNet(X_train.shape[1], hidden=32)
hist = fit(net, X_train_t, y_train, X_val_t, y_val)
print(f"best epoch: {hist['best_epoch'] + 1} of {len(hist['val'])}")
print(f"train MSE at the end: {hist['train'][-1]:.4f}   validation MSE at the end: {hist['val'][-1]:.4f}   best validation MSE: {min(hist['val']):.4f}")

# %%
plt.figure(figsize=(7, 3.8))
plt.plot(range(1, len(hist["train"]) + 1), hist["train"], label="train")
plt.plot(range(1, len(hist["val"]) + 1), hist["val"], label="validation")
plt.axvline(hist["best_epoch"] + 1, color="gray", linestyle="--", label="best validation epoch")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("MSE on standardised price (log scale)")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# %% [markdown]
# Reading the curves: both losses fall quickly at first. Whether the two curves separate later, and when validation loss stops improving while training loss keeps falling (the signature
# of **overfitting**), you can see in your own plot. The dashed line marks the epoch whose weights we kept, which is why `fit` restores them rather than using the last epoch.
#
# ## 5. Evaluate in dollars, and compare honestly
#
# A standardised MSE of 0.05 means nothing to a stakeholder. Convert predictions back to dollars with the **training** statistics and report MAE (typical error)
# and RMSE (punishes big misses). Then compare with two baselines that any deep model must beat to justify its existence:
# always predicting the training mean, and scikit-learn's `LinearRegression` on the *same* preprocessed features.

# %%
def predict_dollars(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy() * price_std + price_mean


def report(name, y_true, y_pred):
    return {"model": name, "MAE ($)": mean_absolute_error(y_true, y_pred), "RMSE ($)": root_mean_squared_error(y_true, y_pred), "R2": r2_score(y_true, y_pred)}


true_test = test_df["price"].to_numpy()
linreg = LinearRegression().fit(X_train, train_df["price"])
rows = [
    report("predict the training mean", true_test, np.full(len(true_test), train_df["price"].mean())),
    report("LinearRegression (sklearn)", true_test, linreg.predict(X_test)),
    report("PriceNet (hidden=32)", true_test, predict_dollars(net, X_test_t)),
]
results = pd.DataFrame(rows).set_index("model")
results.round(3)

# %% [markdown]
# Read this table before believing the network: the mean baseline tells you how much signal exists at all, and `LinearRegression` tells you how much of it a
# straight-line model already captures. Whether the network is meaningfully better than the linear model is a **measurement**, not an assumption;
# look at the numbers above. When the gap is small, the simpler model wins on speed, stability and explainability.
#
# One more sanity check: a single `nn.Linear` layer trained by the very same loop is just linear regression with gradient descent. If our loop is correct it should land
# close to `LinearRegression`.

# %%
torch.manual_seed(42)
linear_net = nn.Sequential(nn.Linear(X_train.shape[1], 1), nn.Flatten(0))
fit(linear_net, X_train_t, y_train, X_val_t, y_val, epochs=60, lr=1e-2)
lin_torch = predict_dollars(linear_net, X_test_t)
print(f"test MAE  torch linear: {mean_absolute_error(true_test, lin_torch):,.0f}   sklearn linear: {mean_absolute_error(true_test, linreg.predict(X_test)):,.0f}")
print(f"max gap between their predictions: {np.abs(lin_torch - linreg.predict(X_test)).max():,.0f} dollars")

# %% [markdown]
# *What just happened?* A one-layer "network" trained by gradient descent gets near the closed-form answer (similar MAE; the maximum gap above shows it has not fully converged, since training stops at the best validation epoch). This is a good **debugging ladder**: if a complex model
# disappoints, check that the loop can at least recover the simple model.
#
# Finally, where does the network make its errors?

# %%
pred_test = predict_dollars(net, X_test_t)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(true_test / 1000, pred_test / 1000, s=8, alpha=0.5)
lims = [0, max(true_test.max(), pred_test.max()) / 1000]
ax[0].plot(lims, lims, color="C3", label="perfect prediction")
ax[0].set_xlabel("actual price (thousand $)")
ax[0].set_ylabel("predicted price (thousand $)")
ax[0].set_title("PriceNet on the test set")
ax[0].legend()
ax[1].hist((pred_test - true_test) / 1000, bins=40)
ax[1].set_xlabel("prediction error (thousand $)")
ax[1].set_ylabel("number of houses")
ax[1].set_title("Error distribution (predicted - actual)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Common mistakes
#
# 1. **Fitting the scaler / imputer / target mean on all data** before splitting. Validation and test scores become optimistic. Fit on train, transform the rest.
# 2. **Forgetting `zero_grad()`** (accumulation, see the previous notebook) or putting it after `backward()` (which wipes the gradient you just computed).
# 3. **Forgetting `model.eval()` and `torch.no_grad()`** when measuring. Dropout and batch-norm layers behave differently in training mode (demo below), and without `no_grad` you build a graph for nothing.
# 4. **Reporting loss in scaled units**, or forgetting to invert the target scaling. Always finish with an error in the original units.
# 5. **Judging the last epoch instead of the best validation epoch**, or choosing it with the *test* set. The test set is for one final number.
# 6. **Skipping the baseline.** A network that "works" but does not beat linear regression is not a win.

# %%
torch.manual_seed(0)
drop_net = nn.Sequential(nn.Linear(X_train.shape[1], 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 1))
xs = X_train_t[:3]
drop_net.train()
print("train mode, same input twice:", drop_net(xs).detach().flatten()[:2].tolist(), drop_net(xs).detach().flatten()[:2].tolist())
drop_net.eval()
print("eval mode,  same input twice:", drop_net(xs).detach().flatten()[:2].tolist(), drop_net(xs).detach().flatten()[:2].tolist())

# %% [markdown]
# ## Interview corner
#
# **1. Walk me through one iteration of a PyTorch training loop.**
# <details><summary>Answer</summary>
# Forward pass to get predictions, compute the loss, <code>optimizer.zero_grad()</code> to clear old gradients, <code>loss.backward()</code> to fill <code>.grad</code> for every parameter,
# and <code>optimizer.step()</code> to update the parameters. Repeat per mini-batch; one pass over the data is an epoch. Evaluate on validation data with <code>model.eval()</code> and <code>torch.no_grad()</code>.
# </details>
#
# **2. Why do we scale inputs (and sometimes the target), and why fit the scaler on the training set only?**
# <details><summary>Answer</summary>
# Features on wildly different scales make the loss surface badly conditioned, so gradient descent is slow or unstable; a target of hundreds of thousands forces the network to learn the scale first.
# Fitting the scaler on all data lets statistics of the validation/test rows influence training (data leakage) and inflates the estimate of generalisation.
# </details>
#
# **3. What does `model.eval()` do, and is it the same as `torch.no_grad()`?**
# <details><summary>Answer</summary>
# No. <code>eval()</code> switches layers such as dropout and batch-norm to inference behaviour. <code>no_grad()</code> stops gradient tracking to save memory and time. For evaluation you normally use both.
# </details>
#
# **4. Your neural network has the same test error as linear regression. What do you do?**
# <details><summary>Answer</summary>
# Ship the linear model if it is good enough: it is cheaper, faster and easier to explain. Before giving up on the network check the data pipeline and the loop (can it reproduce linear regression?),
# tune learning rate, width, regularisation and early stopping, and ask whether the dataset has enough rows and non-linear structure for a network to add value.
# Also try a gradient-boosted tree model, which is often the stronger non-linear baseline on tabular data.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell.

# %% [markdown]
# ### Exercise 1: Scale the validation set without leaking
# Standardise `area_sqft` for the **validation** rows using the mean and standard deviation of the **training** rows (`std` with `ddof=0`, as `StandardScaler` does).
# Store the result as a NumPy array in `val_area_scaled`, and store the training mean you used in `area_mean`.
# *Hint: the statistics come from `train_df`, the values being transformed come from `val_df`.*

# %% tags=["starter"]
area_mean = ...
val_area_scaled = ...

# %%
scaler_ref = StandardScaler().fit(train_df[["area_sqft"]])
assert abs(area_mean - scaler_ref.mean_[0]) < 1e-6, "area_mean should be the mean of area_sqft over the TRAINING rows"
expected = scaler_ref.transform(val_df[["area_sqft"]]).ravel()
assert np.allclose(val_area_scaled, expected, atol=1e-6), "use the training mean and the training std (ddof=0)"
leaky = ((val_df["area_sqft"] - val_df["area_sqft"].mean()) / val_df["area_sqft"].std(ddof=0)).to_numpy()
assert not np.allclose(val_area_scaled, leaky), "that is the validation set scaled with its own statistics (leakage)"
print("Exercise 1 passed")

# %% tags=["solution"]
area_mean = train_df["area_sqft"].mean()
area_std = train_df["area_sqft"].std(ddof=0)
val_area_scaled = ((val_df["area_sqft"] - area_mean) / area_std).to_numpy()

# %% [markdown]
# ### Exercise 2: One correct training step
# Write `train_step(model, optimizer, loss_fn, xb, yb)` that performs the five steps once and returns the loss as a Python float.
# *Hint: forward, loss, `zero_grad`, `backward`, `step`; return `loss.item()`.*

# %% tags=["starter"]
def train_step(model, optimizer, loss_fn, xb, yb):
    ...

# %%
def _make():
    torch.manual_seed(5)
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 1), nn.Flatten(0))


g_chk = torch.Generator().manual_seed(1)
xb_chk, yb_chk = torch.randn(32, 4, generator=g_chk), torch.randn(32, generator=g_chk)
m_user, m_ref = _make(), _make()
opt_user = torch.optim.SGD(m_user.parameters(), lr=0.1)
opt_ref = torch.optim.SGD(m_ref.parameters(), lr=0.1)
losses_user = []
for _ in range(5):
    out = train_step(m_user, opt_user, nn.MSELoss(), xb_chk, yb_chk)
    assert isinstance(out, float), "return loss.item(), a Python float"
    losses_user.append(out)
    opt_ref.zero_grad()
    ref_loss = nn.functional.mse_loss(m_ref(xb_chk), yb_chk)
    ref_loss.backward()
    opt_ref.step()
for p_u, p_r in zip(m_user.parameters(), m_ref.parameters()):
    assert torch.allclose(p_u, p_r, atol=1e-6), "parameters differ from a reference loop: check the order zero_grad -> backward -> step"
assert losses_user[-1] < losses_user[0], "loss should go down on a repeated batch"
print(f"Exercise 2 passed (loss {losses_user[0]:.3f} -> {losses_user[-1]:.3f})")

# %% tags=["solution"]
def train_step(model, optimizer, loss_fn, xb, yb):
    pred = model(xb)
    loss = loss_fn(pred, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# %% [markdown]
# ### Exercise 3: Error in dollars
# Write `test_metrics(model)` that returns a tuple `(mae, rmse)` of the model's errors on the **test set in dollars** (use `X_test_t`, `test_df["price"]`,
# `price_mean`, `price_std`). Put the model in eval mode and do not track gradients. Then compute `mae_net, rmse_net = test_metrics(net)` and
# store in `beats_mean` whether the network's MAE is lower than the MAE of always predicting the training mean price.

# %% tags=["starter"]
def test_metrics(model):
    ...


mae_net, rmse_net = ...
beats_mean = ...

# %%
net.eval()
with torch.no_grad():
    manual = net(X_test_t).numpy() * price_std + price_mean
assert abs(mae_net - np.mean(np.abs(manual - test_df["price"].to_numpy()))) < 1e-3, "MAE must be in dollars: undo the target scaling first"
assert abs(rmse_net - np.sqrt(np.mean((manual - test_df["price"].to_numpy()) ** 2))) < 1e-3, "RMSE must be in dollars"
assert rmse_net >= mae_net, "RMSE is never smaller than MAE"
mean_mae = np.mean(np.abs(test_df["price"].to_numpy() - train_df["price"].mean()))
assert beats_mean == bool(mae_net < mean_mae), "compare with the MAE of predicting the training mean"
print(f"Exercise 3 passed (net MAE {mae_net:,.0f} vs mean-baseline MAE {mean_mae:,.0f})")

# %% tags=["solution"]
def test_metrics(model):
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).numpy() * price_std + price_mean
    truth = test_df["price"].to_numpy()
    return float(np.mean(np.abs(pred - truth))), float(np.sqrt(np.mean((pred - truth) ** 2)))


mae_net, rmse_net = test_metrics(net)
beats_mean = bool(mae_net < np.mean(np.abs(test_df["price"].to_numpy() - train_df["price"].mean())))

# %% [markdown]
# ### Exercise 4 (open-ended): Can you beat linear regression?
# Train `PriceNet` with two or three other settings (hidden width 4 / 128, `weight_decay=1e-3`, fewer or more epochs, a different learning rate) using `fit` and compare the test MAE in dollars with the
# `LinearRegression` row in the results table. Choose settings using **validation** loss, not the test set. Write a short paragraph: what did you try, did any setting clearly beat
# the linear baseline, and given your findings would you deploy the network or the linear model? No assert: explore and report.

# %% [markdown]
# ## Recap
# - Clean absurd labels, then **split before** fitting imputers, scalers and target statistics; fit them on train only.
# - An `nn.Module` declares layers in `__init__` and the computation in `forward`; `parameters()` feeds the optimizer.
# - The loop: forward, loss, `zero_grad`, `backward`, `step`; use `model.eval()` + `torch.no_grad()` to measure, and keep the best-validation weights.
# - Report errors in original units and always against a baseline (mean, linear regression). A deep model has to earn its complexity.
# - A one-layer network trained with your loop should approximate linear regression: a cheap correctness check.
#
# **Go deeper:** the PyTorch docs for `torch.nn` (Module, Sequential, Linear, MSELoss), `torch.optim`, and the tutorial "Learn the Basics" (pytorch.org/tutorials);
# the scikit-learn user guide section on preprocessing data (scikit-learn.org/stable/modules/preprocessing.html).
#
# **Next:** [Classification with Dataset and DataLoader](03_classification_with_dataset_and_dataloader.ipynb)
