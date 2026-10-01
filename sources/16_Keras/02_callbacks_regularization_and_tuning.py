# %% [markdown]
# # Callbacks, Regularization and Tuning
#
# **Why this matters.** A network with enough capacity will memorise its training data. The training loss keeps falling while the
# model quietly gets *worse* on new customers, and you only find out when it is live. Professionals deal with this with three habits:
# **stop training at the right moment** (callbacks), **make memorisation harder** (dropout, L2), and **try a few configurations
# systematically** instead of guessing. This notebook shows each on a dataset small enough to overfit on purpose.
#
# **You will learn**
# - recognise overfitting from training vs validation curves
# - use `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` and write your own callback
# - apply dropout and L2 weight decay, and measure whether they actually help
# - run a small, honest hyper-parameter sweep (and know its limits)
#
# **Prerequisites:** [Sequential and Functional API](01_sequential_and_functional_api.ipynb).
#
# > Data used: `datasets/churn.csv` (synthetic telecom customers, see `docs/DATASETS.md`). We train on only **600 rows** so that
# > overfitting is easy to see; 1,000 rows are used for validation and 1,000 for the final test.

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tempfile
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

c = pd.read_csv(Path("../datasets/churn.csv"))
c["contract"] = c["contract"].str.lower()
c["internet_service"] = c["internet_service"].fillna("None")
c["total_charges"] = c["total_charges"].fillna(c["monthly_charges"] * c["tenure_months"])
for col in ["paperless_billing", "autopay"]:
    c[col] = (c[col] == "Yes").astype("int64")
X = pd.concat(
    [c[["tenure_months", "monthly_charges", "total_charges", "support_calls", "paperless_billing", "autopay", "senior_citizen"]],
     pd.get_dummies(c[["contract", "internet_service"]], dtype="float32")], axis=1).astype("float32").to_numpy()
y = c["churned"].to_numpy("float32")

perm = np.random.default_rng(42).permutation(len(X))
tr, va, te = perm[:600], perm[3000:4000], perm[4000:]  # small train set on purpose
print("train/val/test:", len(tr), len(va), len(te), "| positive rate in train:", y[tr].mean().round(3))

# %%
def build(units=128, dropout=0.0, l2=0.0, lr=1e-3):
    """Two hidden layers; optional dropout and L2 on the hidden weights."""
    keras.utils.set_random_seed(42)
    norm = keras.layers.Normalization()
    norm.adapt(X[tr])  # training rows only
    layers = [keras.layers.Input((X.shape[1],)), norm]
    for _ in range(2):
        layers.append(keras.layers.Dense(units, activation="relu", kernel_regularizer=keras.regularizers.l2(l2) if l2 else None))
        if dropout:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    return model


def fit(model, epochs=40, callbacks=None):
    return model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=epochs, batch_size=64, callbacks=callbacks, verbose=0)


# %% [markdown]
# ## 1. Seeing overfitting
#
# **Intuition.** Fitting noise is like a student memorising last year's exam instead of learning the subject: perfect on the
# questions seen before, poor on new ones. The signature is a training loss that keeps dropping while the **validation** loss
# turns around and climbs. The model below has 128 units per layer for only 600 rows, so it has plenty of room to memorise.

# %%
big = build()
hist = fit(big, epochs=40)
val = np.array(hist.history["val_loss"])
best_epoch = int(val.argmin()) + 1
print(f"validation loss is lowest at epoch {best_epoch} ({val.min():.3f}); at epoch 40 it is {val[-1]:.3f}")
print(f"training loss at epoch 40: {hist.history['loss'][-1]:.3f}")

# %%
def plot_curves(history, ax, title):
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="validation loss")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross-entropy (lower is better)")
    ax.legend()


fig, ax = plt.subplots(figsize=(6, 3.5))
plot_curves(hist, ax, "No regularisation: validation loss turns up")
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* The gap between the two curves is the model memorising the 600 training rows. The best model existed
# early on, around the epoch where validation loss bottomed out; training longer only made it worse. So we want to *notice* the turn and stop there.

# %% [markdown]
# ## 2. Callbacks
#
# A **callback** is an object Keras calls at fixed moments (start/end of training, of each epoch, of each batch). The three to know:
#
# | Callback | What it does | Key arguments |
# |---|---|---|
# | `EarlyStopping` | stop when a monitored metric has not improved | `monitor`, `patience`, `restore_best_weights` |
# | `ModelCheckpoint` | save the model (best or every epoch) | `filepath` (`.keras`), `save_best_only` |
# | `ReduceLROnPlateau` | multiply the learning rate by `factor` when progress stalls | `factor`, `patience`, `min_lr` |
#
# `patience=5` means "tolerate 5 epochs without improvement". `restore_best_weights=True` rolls the model back to the best epoch; without
# it you keep the *last* (already overfit) weights.

# %%
ckpt_dir = tempfile.TemporaryDirectory()
ckpt_path = Path(ckpt_dir.name) / "best.keras"

model = build()
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
]
hist_cb = fit(model, epochs=100, callbacks=callbacks)  # up to 100 epochs, but early stopping decides
n_run = len(hist_cb.epoch)
print(f"ran {n_run} of 100 epochs; best val loss {min(hist_cb.history['val_loss']):.3f}")
print("learning rate went:", [round(float(v), 6) for v in sorted(set(np.round(hist_cb.history["learning_rate"], 6)), reverse=True)])
print("checkpoint file exists:", ckpt_path.exists())

# %% [markdown]
# Training stopped by itself shortly after the best epoch, and `restore_best_weights` left us with the model from the best epoch
# rather than the last. The checkpoint file holds the best model seen *during* training: useful when a job may crash or you
# want to resume. Reload it like any `.keras` file:

# %%
best_from_disk = keras.models.load_model(ckpt_path)
for name, m in [("in memory (restored best)", model), ("checkpoint from disk", best_from_disk)]:
    print(f"{name:26s} val AUC {roc_auc_score(y[va], m.predict(X[va], verbose=0)[:, 0]):.3f}")

# %% [markdown]
# ### Writing your own callback
# Subclass `keras.callbacks.Callback` and override the hooks you need (`on_epoch_end(self, epoch, logs)` receives the metrics of that epoch).
# This one records the train/validation gap so you can watch overfitting grow numerically:

# %%
class GapLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.gaps = []

    def on_epoch_end(self, epoch, logs=None):
        self.gaps.append(logs["val_loss"] - logs["loss"])  # positive and growing = overfitting


gap = GapLogger()
fit(build(), epochs=30, callbacks=[gap])
print("gap at epochs 1, 10, 20, 30:", [round(gap.gaps[i], 3) for i in (0, 9, 19, 29)])

# %% [markdown]
# ## 3. Regularization: make memorising harder
#
# - **Dropout(p)**: during training, randomly zero a fraction `p` of a layer's outputs on each step. The network cannot rely on any one unit, so it
#   learns redundant, robust features (like a team that cannot depend on one star player). It is switched **off** at inference automatically.
# - **L2 weight decay** (`kernel_regularizer=l2(λ)`): adds `λ * sum(w²)` to the loss, discouraging large weights and therefore extreme, wiggly functions.
#
# Compare four variants on the same split, all with early stopping (so the comparison is at each model's best epoch, which is fair):

# %%
def run(name, **kw):
    m = build(**kw)
    h = fit(m, epochs=100, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)])
    p_val = m.predict(X[va], verbose=0)[:, 0]
    # Keras' `val_loss` INCLUDES the L2 penalty, so it is not comparable across L2/non-L2 models: recompute plain cross-entropy
    return {"model": name, "epochs": len(h.epoch), "val log loss": round(log_loss(y[va], p_val), 3), "val AUC": round(roc_auc_score(y[va], p_val), 3)}, h


rows, hists = [], {}
for name, kw in {"baseline": {}, "L2 1e-3": {"l2": 1e-3}, "dropout 0.4": {"dropout": 0.4}, "dropout 0.4 + L2 1e-3": {"dropout": 0.4, "l2": 1e-3}}.items():
    row, hists[name] = run(name, **kw)
    rows.append(row)
pd.DataFrame(rows).set_index("model")

# %% [markdown]
# Read this table carefully. The "val log loss" column is plain cross-entropy computed from the restored weights' predictions: Keras' own `val_loss`
# (which the early-stopping callback monitors) also adds the L2 penalty for models that have one, so it would look misleadingly worse for the L2 rows.
# The honest outcome is usually modest: with early stopping already in place, dropout is hard to tell from the baseline here, and the L2 rows come out slightly
# worse on both log loss and AUC (the penalty also shifts where early stopping fires, because the monitored `val_loss` includes it). Differences of a few thousandths are
# within noise and another seed can reorder them, so do not read more into this table than "no clear gain". What regularisation buys
# you most reliably is **slower, steadier training and a less fragile stopping point**, plus better results when the gap would otherwise be large.
# Treat it as insurance you verify, not a magic ingredient.

# %% [markdown]
# ## 4. A small hyper-parameter sweep
#
# **Hyper-parameters** are the settings you choose rather than learn: units, dropout, learning rate. A **grid search** simply tries every combination.
# We pick using the **validation** set and look at the **test** set exactly once at the end; choosing on test data would make the reported score optimistic.
# (For bigger searches there is the separate `keras-tuner` package, not installed here, and scikit-learn's search tools.)

# %%
import itertools
import time

grid = list(itertools.product([8, 32, 128], [0.0, 0.3], [1e-2, 1e-3]))  # units x dropout x learning rate
results = []
t0 = time.perf_counter()
for units, dropout, lr in grid:
    m = build(units=units, dropout=dropout, lr=lr)
    h = fit(m, epochs=40, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)])
    results.append({"units": units, "dropout": dropout, "lr": lr, "epochs": len(h.epoch),
                    "val AUC": roc_auc_score(y[va], m.predict(X[va], verbose=0)[:, 0]), "model": m})
print(f"{len(grid)} configurations in {time.perf_counter() - t0:.0f} s")
table = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results]).sort_values("val AUC", ascending=False)
table.head(6).round(3)

# %%
best = max(results, key=lambda r: r["val AUC"])
test_auc = roc_auc_score(y[te], best["model"].predict(X[te], verbose=0)[:, 0])
print(f"chosen on validation: units={best['units']}, dropout={best['dropout']}, lr={best['lr']}")
print(f"validation AUC {best['val AUC']:.3f} -> test AUC {test_auc:.3f} (the test number is the honest one)")
print(f"spread of validation AUC across the grid: {table['val AUC'].min():.3f} to {table['val AUC'].max():.3f}")

# %% [markdown]
# *What just happened?* Twelve trainings, one winner picked on validation data, one final test score. Notice the **spread** across the grid:
# if the best and the tenth-best differ by less than the seed-to-seed noise (try another seed in `build`), the "winner" is partly luck and you
# should prefer the simpler, cheaper configuration. For anything important, repeat each config with several seeds or use cross-validation.
# Grid search also scales badly (every new hyper-parameter multiplies the runs); random search usually finds good settings faster.

# %%
ckpt_dir.cleanup()  # remove the temporary checkpoint directory

# %% [markdown]
# ## Common mistakes
#
# 1. **Early stopping on the training loss.** It keeps improving, so it never stops. Monitor a validation metric (`val_loss`, `val_auc`).
# 2. **Forgetting `restore_best_weights=True`**: you keep the last, slightly overfit weights, not the best.
# 3. **Choosing hyper-parameters on the test set**, then reporting that test score. Select on validation, test once.
# 4. **Validation data that also leaks into training** (for example normalising or oversampling before the split).
# 5. **`validation_split` with ordered data**: Keras takes the *last* fraction of the rows without shuffling, so sorted data gives a biased validation set. Shuffle first or pass `validation_data`.
# 6. **Dropout after the output layer or at inference** (Keras turns it off for `predict`/`evaluate`; in a custom loop pass `training=False`).
# 7. **Over-reading tiny differences** between configs from a single run and a single seed.
# 8. **`monitor` name typos** (`"val_acc"` vs `"val_accuracy"`, `"val_auc"` depends on the metric's `name`): Keras warns and the callback silently does nothing.

# %% [markdown]
# ## Interview corner
#
# **1. How do you detect and handle overfitting in a neural network?**
# <details><summary>Answer</summary>
# Detect: training metric keeps improving while validation stalls or worsens. Handle: more data or augmentation, a smaller model, early stopping,
# dropout, L2/weight decay, and simpler features. Always verify on untouched test data.
# </details>
#
# **2. What does dropout do and why is it disabled at inference time?**
# <details><summary>Answer</summary>
# During training it randomly zeroes activations (and rescales the rest) so units cannot co-adapt, acting like an ensemble of thinned networks. At
# inference you want deterministic predictions using all units, and Keras has already scaled activations during training so the expected magnitude matches.
# </details>
#
# **3. L1 vs L2 regularisation?**
# <details><summary>Answer</summary>
# L2 penalises squared weights, shrinking all weights smoothly (weight decay). L1 penalises absolute values and drives many weights exactly to zero
# (sparsity). In deep learning L2 (or decoupled weight decay in AdamW) is the default; L1 is used when you want feature selection.
# </details>
#
# **4. What are training, validation and test sets for when tuning?**
# <details><summary>Answer</summary>
# Train fits the weights. Validation guides decisions (epochs, hyper-parameters, model choice) so it gets "used up". Test is touched once at the end to estimate real-world
# performance. Tuning on the test set leaks information and inflates the final number.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Stop at the right time
# Train `build()` with an `EarlyStopping` callback (monitor `val_loss`, `patience=3`, `restore_best_weights=True`, up to 100 epochs, using the `fit` helper).
# Store the trained model in `es_model`, the history in `es_hist`. The check verifies that training stopped early and that the restored weights are the best ones.

# %% tags=["starter"]
es_model = ...
es_hist = ...

# %%
epochs_run = len(es_hist.epoch)
assert epochs_run < 100, "early stopping should end training before 100 epochs"
val_loss_now = es_model.evaluate(X[va], y[va], verbose=0)[0]
assert abs(val_loss_now - min(es_hist.history["val_loss"])) < 2e-3, (
    f"restore_best_weights should leave the model at its best validation loss ({min(es_hist.history['val_loss']):.4f}), got {val_loss_now:.4f}")
print(f"Exercise 1 passed (stopped after {epochs_run} epochs)")

# %% tags=["solution"]
es_model = build()
es_hist = fit(es_model, epochs=100, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])

# %% [markdown]
# ### Exercise 2: A callback that remembers the best epoch
# Write a `Callback` subclass `BestEpoch` that stores, after training, `best_epoch` (1-based number of the epoch with the lowest `val_loss`) and `best_val`
# (that loss). Train `build()` for 15 epochs with it as the only callback and store the callback instance in `tracker`.
# *Hint: keep the lowest value seen so far in `on_epoch_end`; `logs["val_loss"]`.*

# %% tags=["starter"]
class BestEpoch(keras.callbacks.Callback):
    ...


tracker = ...

# %%
check_hist = fit(build(), epochs=15)  # same seed, same data: identical history
vals = check_hist.history["val_loss"]
assert tracker.best_epoch == int(np.argmin(vals)) + 1, f"best epoch should be {int(np.argmin(vals)) + 1}, got {tracker.best_epoch}"
assert abs(tracker.best_val - min(vals)) < 1e-4, "best_val should be the minimum validation loss"
print(f"Exercise 2 passed (best epoch {tracker.best_epoch}, val loss {tracker.best_val:.3f})")

# %% tags=["solution"]
class BestEpoch(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.best_val, self.best_epoch = float("inf"), 0

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.best_val:
            self.best_val, self.best_epoch = logs["val_loss"], epoch + 1


tracker = BestEpoch()
fit(build(), epochs=15, callbacks=[tracker])

# %% [markdown]
# ### Exercise 3: Verify what L2 adds to the loss
# Build `reg_model = build(units=16, l2=0.01)` (no training needed). Keras keeps the penalty terms in `reg_model.losses`. Compute by hand the total L2 penalty
# `0.01 * sum(w**2)` over the **kernels** (not biases) of the two hidden Dense layers, store it in `penalty_by_hand`, and store Keras's own number in
# `penalty_keras` (the sum of `reg_model.losses`, as a float). They must agree. *Hint: `layer.get_weights()[0]` is the kernel; Dense layers are the ones with a `kernel_regularizer`.*

# %% tags=["starter"]
reg_model = build(units=16, l2=0.01)
penalty_by_hand = ...
penalty_keras = ...

# %%
assert len(reg_model.losses) == 2, "two hidden Dense layers carry an L2 regulariser"
assert abs(penalty_by_hand - penalty_keras) < 1e-4, f"by hand {penalty_by_hand:.5f} vs Keras {penalty_keras:.5f}"
assert penalty_keras > 0
print(f"Exercise 3 passed (penalty = {penalty_keras:.4f})")

# %% tags=["solution"]
reg_model = build(units=16, l2=0.01)
kernels = [l.get_weights()[0] for l in reg_model.layers if isinstance(l, keras.layers.Dense) and l.kernel_regularizer is not None]
penalty_by_hand = float(sum(0.01 * np.sum(k**2) for k in kernels))
penalty_keras = float(sum(float(v) for v in reg_model.losses))

# %% [markdown]
# ### Exercise 4: Sweep the dropout rate
# For dropout in `[0.0, 0.2, 0.4, 0.6]` (with `units=64`, `lr=1e-3`), train with `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`
# and record the **validation AUC**. Store the results in a dict `sweep` mapping dropout -> val AUC, and the best rate in `best_dropout`.
# *Hint: reuse `build`, `fit`, `roc_auc_score`.*

# %% tags=["starter"]
sweep = ...
best_dropout = ...

# %%
assert sorted(sweep) == [0.0, 0.2, 0.4, 0.6], "sweep should have one entry per dropout rate"
assert all(0.5 < v < 1.0 for v in sweep.values()), "validation AUCs should be plausible (above chance)"
assert best_dropout == max(sweep, key=sweep.get), "best_dropout must be the key with the highest validation AUC"
print("Exercise 4 passed:", {k: round(v, 3) for k, v in sweep.items()})

# %% tags=["solution"]
sweep = {}
for d in [0.0, 0.2, 0.4, 0.6]:
    m = build(units=64, dropout=d, lr=1e-3)
    fit(m, epochs=100, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)])
    sweep[d] = float(roc_auc_score(y[va], m.predict(X[va], verbose=0)[:, 0]))
best_dropout = max(sweep, key=sweep.get)

# %% [markdown]
# ### Exercise 5 (open-ended): Is the winner real?
# Re-run the sweep of Exercise 4 with a different seed inside `build` (change `set_random_seed(42)` to 7) and compare. Did the ranking change? Write two sentences:
# what does that tell you about trusting a single sweep, and what would you do to make the choice more reliable (more seeds, cross-validation, a bigger validation set)?

# %% [markdown]
# ## Recap
# - Overfitting = training loss falls while validation loss rises. Look at both curves, always.
# - `EarlyStopping(restore_best_weights=True)` + `ModelCheckpoint(save_best_only=True)` + `ReduceLROnPlateau` are the standard trio; custom callbacks hook into any training event.
# - Dropout and L2 make memorisation harder; verify with a table, the gain is often modest once early stopping is in place.
# - Tune on validation, report test once, and treat small differences between configs as noise unless repeated across seeds.
#
# **Go deeper:** the Keras guides "Training & evaluation with the built-in methods" and "Writing your own callbacks", and the API docs for
# `keras.callbacks` and `keras.regularizers` at keras.io.
#
# **Next:** [A CNN on digits](03_cnn_on_digits_with_keras.ipynb)
