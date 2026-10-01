# %% [markdown]
# # Classification with `Dataset` and `DataLoader`
#
# **Why this matters.** A telecom retention team wants to know which customers are likely to leave, so they can call the risky ones first. In PyTorch, two small abstractions turn a table into training batches:
# a **`Dataset`** (how to fetch example number `i`) and a **`DataLoader`** (how to group, shuffle and serve batches). Every PyTorch project, from this churn model to an image pipeline with a million files,
# is built on that pair. We will also ask the question a senior reviewer asks: *does the neural network actually beat a scikit-learn baseline on this tabular data?*
#
# **You will learn**
# - write a custom `Dataset` and wrap it in a `DataLoader` (batching, shuffling, reproducible order)
# - build a binary classifier with `BCEWithLogitsLoss` and explain why the model outputs logits, not probabilities
# - prevent leakage in the preprocessing (split first, fit encoders and scalers on the training part) and use early stopping
# - compare honestly with logistic regression and gradient boosting over several seeds, using ROC AUC as well as accuracy
#
# **Prerequisites:** [02 `nn.Module` and the training loop](02_nn_module_and_training_loop.ipynb) and the ML module's classification and metrics notebooks.
#
# > Data used: `datasets/churn.csv`, a synthetic table of 5,000 customers (`churned` = 1 if they left, about 37% positive). The text `None` in `internet_service` is a real category (no internet plan), so we read the file
# > with `keep_default_na=False`; see `docs/DATASETS.md`.

# %%
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

torch.manual_seed(42)
torch.set_num_threads(2)

df = pd.read_csv(Path("../datasets/churn.csv"), keep_default_na=False, na_values=[""])
df["contract"] = df["contract"].str.strip().str.lower()        # 'Month-to-month' vs 'month-to-month' are the same category
print(df.shape, "| churn rate:", round(df["churned"].mean(), 3))
print(df["contract"].value_counts().to_dict())
print("genuinely missing:", df.isna().sum()[lambda s: s > 0].to_dict())

# %% [markdown]
# ## 1. Split first, then preprocess using the training rows only
#
# Same discipline as the last notebook. Three stratified parts (stratified keeps the 37% churn rate in each): train 60%, validation 20% (early stopping and threshold choice), test 20% (one final score).
# The `ColumnTransformer` imputes missing values (median for numbers, a `"missing"` label for the text column), scales numbers and one-hot encodes categories. It is **fit on train only**.

# %%
train_val, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["churned"])
train_df, val_df = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val["churned"])

num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
cat_cols = ["contract", "internet_service", "paperless_billing", "autopay"]
preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat_cols),
])
X_train = preprocess.fit_transform(train_df).astype(np.float32)
X_val = preprocess.transform(val_df).astype(np.float32)
X_test = preprocess.transform(test_df).astype(np.float32)
y_train, y_val, y_test = (d["churned"].to_numpy().astype(np.float32) for d in (train_df, val_df, test_df))
print("features:", X_train.shape[1], "| train/val/test:", len(X_train), len(X_val), len(X_test))
print("churn rate in each part:", [round(float(y.mean()), 3) for y in (y_train, y_val, y_test)])

# %% [markdown]
# ## 2. A custom `Dataset`
#
# A `Dataset` is any object with two methods: `__len__` (how many examples) and `__getitem__(i)` (example `i`, as tensors). That is the whole contract. For data that fits in memory it just indexes tensors;
# for huge data `__getitem__` would read a file from disk or decode an image on the fly, and the `DataLoader` would not care.

# %%
class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)                # float32 features
        self.y = torch.from_numpy(y)                # float32 labels (0. or 1.), as BCEWithLogitsLoss expects

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_ds, val_ds, test_ds = ChurnDataset(X_train, y_train), ChurnDataset(X_val, y_val), ChurnDataset(X_test, y_test)
x0, y0 = train_ds[0]
print("len:", len(train_ds), "| one item:", tuple(x0.shape), x0.dtype, "| label", float(y0))

# %% [markdown]
# ## 3. The `DataLoader`: batches, shuffling, reproducibility
#
# `DataLoader(dataset, batch_size, shuffle=True)` calls `__getitem__` for a batch of indices, stacks the results into tensors and hands them to your loop. Shuffle the **training** loader every epoch (so
# batches differ) and never the evaluation loaders. Pass a seeded `torch.Generator` to make the shuffle order reproducible.

# %%
def make_loader(ds, shuffle, seed=0, batch_size=128):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


loader = make_loader(train_ds, shuffle=True)
xb, yb = next(iter(loader))
print("one batch:", tuple(xb.shape), tuple(yb.shape), "| batches per epoch:", len(loader), "| last batch size:", len(train_ds) % 128)
first_a = [b[1][:3].tolist() for b, _ in zip(make_loader(train_ds, True, seed=1), range(1))]
first_b = [b[1][:3].tolist() for b, _ in zip(make_loader(train_ds, True, seed=1), range(1))]
print("same seed gives the same first batch:", first_a == first_b)

# %% [markdown]
# *What just happened?* 3,000 training rows became 24 batches (the last one smaller). With `num_workers=0` (the default) batches are built in the main process, which is fine for in-memory tables. Use `num_workers > 0`
# only when `__getitem__` does real work (disk, decoding); on Windows and macOS it needs the `if __name__ == "__main__":` guard in scripts, and it is wasteful for small data.
#
# ## 4. Model, loss and why logits
#
# For a yes/no target the network ends in **one** number per example, the *logit* (any real number; positive means "more likely churn"). `BCEWithLogitsLoss` applies the sigmoid and the log loss together in one
# numerically stable step, so the model itself should **not** end with a sigmoid. You only apply `torch.sigmoid` when you want a probability to show or threshold.

# %%
class ChurnNet(nn.Module):
    def __init__(self, n_features, hidden=32, p_drop=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.body(x).squeeze(1)              # (N, 1) -> (N,) logits, matching the (N,) labels


net = ChurnNet(X_train.shape[1])
print(net)
with torch.no_grad():
    logits = net(torch.from_numpy(X_train[:4]))
    print("logits:", logits.numpy().round(3), "-> probabilities:", torch.sigmoid(logits).numpy().round(3))
print(f"untrained loss {nn.BCEWithLogitsLoss()(net(train_ds.X), train_ds.y).item():.3f}  (coin-flip reference ln(2) = {np.log(2):.3f})")

# %% [markdown]
# ## 5. Training with early stopping
#
# The loop is the one from the previous notebook, plus: evaluate on the validation loader each epoch, remember the weights with the lowest validation loss, stop when it has not improved for `patience` epochs.
# We write two helpers, `predict_proba` (evaluation mode, no gradients, returns probabilities as NumPy) and `fit`.

# %%
loss_fn = nn.BCEWithLogitsLoss()


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(X))).numpy()


def fit(seed=0, hidden=32, p_drop=0.0, weight_decay=0.0, pos_weight=None, max_epochs=80, patience=8, lr=1e-3):
    torch.manual_seed(seed)
    model = ChurnNet(X_train.shape[1], hidden, p_drop)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else torch.tensor(pos_weight))
    loader = make_loader(train_ds, shuffle=True, seed=seed)
    best, best_state, waited, curves = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(yb)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_ds.X), val_ds.y).item()
        curves.append((running / len(train_ds), val_loss))
        if val_loss < best:
            best, best_state, waited, best_epoch = val_loss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return model, np.array(curves), best_epoch + 1


model, curves, best_epoch = fit(seed=0)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(curves[:, 0], label="train")
ax.plot(curves[:, 1], label="validation")
ax.axvline(best_epoch - 1, color="gray", linestyle=":", label="best epoch")
ax.set_xlabel("epoch")
ax.set_ylabel("log loss")
ax.set_title("Churn MLP: train vs validation loss")
ax.legend()
plt.show()
print("best epoch:", best_epoch, "| stopped after", len(curves), "epochs")

# %% [markdown]
# *What just happened?* Validation loss bottoms out after only a handful of epochs, while the training loss keeps falling: this data has a modest signal, and the network starts memorising noise early. Early stopping
# restored the weights from the best epoch.
#
# ## 6. Honest comparison: is the network worth it?
#
# A neural network is one candidate among several. We compare it with **logistic regression** (simple, hard to beat on small tabular data) and **gradient boosting** (`HistGradientBoostingClassifier`, the usual
# champion on tabular data), all on the same split and the same preprocessed features, using the test set once.
# Metrics: **ROC AUC** (how well scores rank churners above non-churners, independent of threshold) and accuracy at 0.5, next to the **majority-class accuracy** (always predict "stays"), which is the bar any accuracy must clear.

# %%
net_scores = []
for seed in range(5):
    m, _, _ = fit(seed=seed)
    p = predict_proba(m, X_test)
    net_scores.append((roc_auc_score(y_test, p), accuracy_score(y_test, p > 0.5)))
net_scores = np.array(net_scores)

logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
gbm_default = HistGradientBoostingClassifier(random_state=42).fit(X_train, y_train)
gbm_small = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=42).fit(X_train, y_train)

rows = [("majority class", 0.5, 1 - y_test.mean())]
for name, clf in [("logistic regression", logreg), ("gradient boosting (defaults)", gbm_default), ("gradient boosting (depth 3, slower)", gbm_small)]:
    p = clf.predict_proba(X_test)[:, 1]
    rows.append((name, roc_auc_score(y_test, p), accuracy_score(y_test, p > 0.5)))
rows.append((f"PyTorch MLP (mean of 5 seeds)", net_scores[:, 0].mean(), net_scores[:, 1].mean()))
comparison = pd.DataFrame(rows, columns=["model", "test ROC AUC", "test accuracy"]).set_index("model")
print(comparison.round(3).to_string())
print(f"MLP seed-to-seed std: AUC {net_scores[:, 0].std():.4f}, accuracy {net_scores[:, 1].std():.4f}")

# %% [markdown]
# Read this table without wishful thinking. On 3,000 training rows of mostly-linear signal, **logistic regression is about as good as the neural network (here marginally better)**, and the network's seed-to-seed spread tells you the
# differences between the top models are small. Untuned gradient boosting can even trail, because it overfits easily on small data (see the depth-3 variant). This is typical: for tabular data, deep learning rarely
# beats well-tuned gradient boosting or even a linear model unless the dataset is large or has special structure (text, images, sequences, high-cardinality categories). In a real project you would pick the simplest model
# that meets the metric, not the most fashionable one. The PyTorch version earns its keep when you need a custom loss, multiple inputs (tabular plus text or images), or an end-to-end trainable pipeline.
#
# ## 7. Imbalance, thresholds and what the business cares about
#
# At the default threshold of 0.5 the model misses many churners (recall is low), because only 37% of rows are positive. Two levers:
#
# - **`pos_weight`** in `BCEWithLogitsLoss` makes a missed positive cost more in training.
# - **Move the threshold** on the predicted probability, chosen on the **validation** set, never on the test set.

# %%
n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
pw = n_neg / n_pos
print(f"negatives {n_neg}, positives {n_pos} -> pos_weight = {pw:.2f}")

m_plain, _, _ = fit(seed=0)
m_weighted, _, _ = fit(seed=0, pos_weight=pw)
for name, m in [("plain loss", m_plain), ("pos_weight loss", m_weighted)]:
    p = predict_proba(m, X_test)
    print(f"{name:16s} AUC {roc_auc_score(y_test, p):.3f} | at 0.5: recall {recall_score(y_test, p > 0.5):.2f}, accuracy {accuracy_score(y_test, p > 0.5):.3f}, F1 {f1_score(y_test, p > 0.5):.3f}")

# %% [markdown]
# Weighting trades accuracy for recall, which is often what a retention team wants (calling an unnecessary customer is cheaper than losing one). The ranking quality (AUC) barely changes, as expected: the
# weight shifts the operating point rather than making the model smarter.

# %% [markdown]
# ## Common mistakes
#
# 1. **Fitting the scaler/encoder before splitting**, then reporting test results that quietly used test statistics.
# 2. **A sigmoid at the end of the model plus `BCEWithLogitsLoss`** (a double sigmoid) or **logits fed to `BCELoss`**. Pick one pairing: raw logits with `BCEWithLogitsLoss`, or sigmoid outputs with `BCELoss`; the first is more stable.
# 3. **Integer labels with `BCEWithLogitsLoss`**: targets must be floats, same shape as the logits.
# 4. **Shuffling the validation or test loader** is harmless for metrics but shuffling the training loader is essential; forgetting `shuffle=True` makes batches follow the file order.
# 5. **Comparing one lucky seed of the network with one deterministic baseline.** Report a mean and a spread over several seeds, as we did.
# 6. **Reporting accuracy only on imbalanced data.** Always show the majority-class accuracy and a threshold-free metric like AUC.

# %%
try:
    nn.BCEWithLogitsLoss()(torch.zeros(4), torch.tensor([0, 1, 1, 0]))        # integer targets
except RuntimeError as e:
    print("RuntimeError:", str(e)[:100])

# %% [markdown]
# ## Interview corner
#
# **1. What do `Dataset` and `DataLoader` each do?**
# <details><summary>Answer</summary>
# A <code>Dataset</code> defines how many examples there are and how to fetch example <code>i</code>. A <code>DataLoader</code> wraps a dataset to produce shuffled mini-batches, optionally in parallel worker processes, and collates
# the examples into stacked tensors.
# </details>
#
# **2. Why does the model output logits when using `BCEWithLogitsLoss`?**
# <details><summary>Answer</summary>
# The loss fuses sigmoid and log loss using the log-sum-exp trick, which avoids overflow and log(0). Applying a separate sigmoid first, then a plain log loss, can give infinities for very confident predictions.
# You apply <code>torch.sigmoid</code> only at prediction time.
# </details>
#
# **3. Would you use a neural network for this churn problem in production?**
# <details><summary>Answer</summary>
# Probably start with logistic regression and gradient boosting: they train in seconds, are easier to explain and often match or beat a small MLP on tabular data, as the measured comparison shows. Choose a network only if
# there is a concrete reason (multimodal inputs, very large data, a custom objective).
# </details>
#
# **4. Class imbalance: what can you do?**
# <details><summary>Answer</summary>
# Use a threshold-free metric (ROC AUC, PR AUC) for model selection, weight the loss (<code>pos_weight</code>) or resample, and choose the decision threshold on validation data to match the business cost of false negatives vs false positives.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell at the bottom of each exercise.

# %% [markdown]
# ### Exercise 1: Write a `Dataset` that returns a dictionary
# Real pipelines often return named fields. Write `DictChurnDataset(X, y)` whose `__getitem__(i)` returns `{"features": ..., "label": ...}` (tensors), and whose `__len__` works. Default collation turns a list
# of such dictionaries into one dictionary of stacked tensors. *Hint: copy `ChurnDataset` and change the return value.*

# %% tags=["starter"]
class DictChurnDataset(Dataset):
    ...

# %%
ds = DictChurnDataset(X_train, y_train)
assert len(ds) == len(X_train), "__len__ should be the number of rows"
item = ds[5]
assert set(item.keys()) == {"features", "label"}, "keys must be 'features' and 'label'"
assert torch.allclose(item["features"], torch.from_numpy(X_train[5])) and float(item["label"]) == float(y_train[5]), "item 5 does not match row 5"
batch = next(iter(DataLoader(ds, batch_size=16)))
assert batch["features"].shape == (16, X_train.shape[1]) and batch["label"].shape == (16,), "DataLoader should stack a dict of tensors"
print("Exercise 1 passed")

# %% tags=["solution"]
class DictChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"features": self.X[i], "label": self.y[i]}

# %% [markdown]
# ### Exercise 2: A class-weight from data
# Store in `pw_val` the `pos_weight` (negatives divided by positives) computed from the **validation** labels, and in `loss_w` the value of `BCEWithLogitsLoss(pos_weight=...)` for a zero-logit
# prediction on the first 50 validation rows. *Hint: a logit of 0 means probability 0.5; build the loss with `torch.tensor(pw_val)`.*

# %% tags=["starter"]
pw_val = ...
loss_w = ...

# %%
neg, pos = (y_val == 0).sum(), (y_val == 1).sum()
assert abs(pw_val - neg / pos) < 1e-9, "pos_weight is (number of negatives) / (number of positives)"
targets = torch.from_numpy(y_val[:50])
manual = -(float(pw_val) * targets * np.log(0.5) + (1 - targets) * np.log(0.5)).mean()     # weighted log loss at p = 0.5
assert abs(float(loss_w) - float(manual)) < 1e-5, f"expected {float(manual):.5f}, got {float(loss_w):.5f}"
print("Exercise 2 passed")

# %% tags=["solution"]
pw_val = float((y_val == 0).sum() / (y_val == 1).sum())
loss_w = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw_val))(torch.zeros(50), torch.from_numpy(y_val[:50]))

# %% [markdown]
# ### Exercise 3: Choose the decision threshold on the validation set
# Write `best_f1_threshold(y_true, proba)` that tries thresholds `0.05, 0.10, ..., 0.95`, and returns the one with the highest F1 (`sklearn.metrics.f1_score`). Then use it with the validation
# probabilities of `m_plain` to set `thr`, and compute the test F1 at that threshold as `test_f1`. Is it better than F1 at 0.5?

# %% tags=["starter"]
def best_f1_threshold(y_true, proba):
    ...


thr = ...
test_f1 = ...

# %%
grid = np.arange(0.05, 0.951, 0.05)
p_val_chk = predict_proba(m_plain, X_val)
best_chk = max(f1_score(y_val, p_val_chk > t) for t in grid)
assert f1_score(y_val, p_val_chk > thr) >= best_chk - 1e-9, "thr should reach the best validation F1 on the grid"
p_test_chk = predict_proba(m_plain, X_test)
assert abs(test_f1 - f1_score(y_test, p_test_chk > thr)) < 1e-9, "test_f1 should be the F1 on the TEST set at your validation-chosen threshold"
print(f"Exercise 3 passed (threshold {thr:.2f}: test F1 {test_f1:.3f} vs {f1_score(y_test, p_test_chk > 0.5):.3f} at 0.5)")

# %% tags=["solution"]
def best_f1_threshold(y_true, proba):
    grid = np.arange(0.05, 0.951, 0.05)
    scores = [f1_score(y_true, proba > t) for t in grid]
    return float(grid[int(np.argmax(scores))])


thr = best_f1_threshold(y_val, predict_proba(m_plain, X_val))
test_f1 = f1_score(y_test, predict_proba(m_plain, X_test) > thr)

# %% [markdown]
# ### Exercise 4: Does a bigger network help? Measure it
# Train `fit(seed=s, hidden=128)` for seeds 0-2 and compute the mean test ROC AUC as `auc_big`; also compute the mean test AUC of the default `hidden=32` network over the same seeds as `auc_small`.
# Store them, then decide whether the difference is bigger than the seed-to-seed spread. The check only validates the numbers.

# %% tags=["starter"]
auc_big = ...
auc_small = ...

# %%
def mean_auc(hidden):
    return float(np.mean([roc_auc_score(y_test, predict_proba(fit(seed=s, hidden=hidden)[0], X_test)) for s in range(3)]))


assert abs(auc_big - mean_auc(128)) < 1e-6, "auc_big should be the mean test AUC of hidden=128 over seeds 0-2"
assert abs(auc_small - mean_auc(32)) < 1e-6, "auc_small should be the mean test AUC of hidden=32 over seeds 0-2"
print(f"Exercise 4 passed (hidden=128: {auc_big:.4f}, hidden=32: {auc_small:.4f})")

# %% tags=["solution"]
def _mean_auc(hidden):
    return float(np.mean([roc_auc_score(y_test, predict_proba(fit(seed=s, hidden=hidden)[0], X_test)) for s in range(3)]))


auc_big = _mean_auc(128)
auc_small = _mean_auc(32)

# %% [markdown]
# ### Exercise 5 (open-ended): Beat logistic regression, or explain why you cannot
# Try one idea (an interaction feature such as `tenure_months * monthly_charges`, a smaller network with `p_drop=0.3`, weight decay, or a different learning rate) and compare test AUC with logistic regression over 5 seeds.
# Report the mean and spread. If you cannot beat the baseline, say what that tells you about this dataset. No assert: explore and report.

# %% [markdown]
# ## Recap
# - `Dataset` = `__len__` + `__getitem__`; `DataLoader` = batching, shuffling (train only), optional workers. Seed a `Generator` for reproducible order.
# - Binary classification: one logit out, `BCEWithLogitsLoss` (float targets), sigmoid only when you need probabilities.
# - Fit preprocessing on the training split only; use early stopping on validation loss; touch the test set once.
# - Compare with logistic regression and gradient boosting, over several seeds, using ROC AUC plus accuracy and the majority baseline. On small tabular data, simple models often match or beat a neural net.
# - Imbalance: `pos_weight` or threshold tuning (on validation) moves the operating point; it does not improve ranking quality.
#
# **Go deeper:** the PyTorch tutorial "Datasets and DataLoaders" and the docs for `torch.nn.BCEWithLogitsLoss` (pytorch.org); the scikit-learn user guide on `HistGradientBoostingClassifier` and on metrics (scikit-learn.org).
#
# **Next:** [CNN on digits](04_cnn_on_digits.ipynb)
