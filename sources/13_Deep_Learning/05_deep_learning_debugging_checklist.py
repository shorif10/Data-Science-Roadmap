# %% [markdown]
# # The Deep Learning Debugging Checklist
#
# **Why this matters.** Ordinary software fails loudly: an exception, a red test. A neural network with a bug usually *trains anyway* and gives you a mediocre number, and you
# cannot tell "the idea does not work" from "I forgot one line". Practitioners who ship models follow a fixed checklist, cheapest checks first, so that they never spend three days tuning
# a model that had a silent bug. This notebook **creates each classic failure on purpose**, shows what it looks like in the numbers, and shows the fix.
#
# **You will learn**
# - run the overfit-one-batch test (and know what it can and cannot prove)
# - use a playbook for "the loss is not going down": learning rate, input scale, `zero_grad`, double softmax
# - recognise shape/broadcasting bugs that raise no error, and loss explosions (`inf`, `nan`) and how to fix or contain them
# - spot data leakage: duplicates across splits, and labels that no longer match inputs
#
# **Prerequisites:** [03 Training dynamics](03_training_dynamics_optimizers_and_regularization.ipynb) and [02 Backpropagation from scratch](02_backpropagation_from_scratch.ipynb).
#
# > Data used: scikit-learn's bundled `digits` (1,797 real 8x8 handwritten digits). A healthy small MLP reaches roughly 93-97% validation accuracy here, which gives us a clear reference for "working".

# %%
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
torch.set_num_threads(2)

X_raw, y_all = load_digits(return_X_y=True)
X_all = X_raw / 16.0
X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
X_tr, X_va = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_va, dtype=torch.float32)
y_tr, y_va = torch.tensor(y_tr), torch.tensor(y_va)
loss_fn = nn.CrossEntropyLoss()


def make_model():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))


def train(model, Xtr, ytr, lr=0.1, epochs=20, batch_size=64, zero_grad=True, softmax_first=False, clip=None, seed=0):
    """A deliberately configurable loop: each flag switches one classic bug on. Returns (final batch loss, validation accuracy)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            if zero_grad:
                opt.zero_grad()
            out = model(Xtr[idx])
            if softmax_first:
                out = out.softmax(dim=1)                 # BUG when followed by CrossEntropyLoss
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    with torch.no_grad():
        acc = (model(X_va).argmax(1) == y_va).float().mean().item()
    return loss.item(), acc


torch.manual_seed(0)
healthy_loss, healthy_acc = train(make_model(), X_tr, y_tr)
print(f"healthy reference: final batch loss {healthy_loss:.3f}, validation accuracy {healthy_acc:.1%}")

# %% [markdown]
# ## The checklist, in the order to run it
#
# | # | Check | Cost | Catches |
# |---|---|---|---|
# | 1 | Look at the data, shapes and dtypes; confirm labels match inputs | minutes | wrong labels, wrong scale, wrong shapes |
# | 2 | Compute the loss of a *random* model and compare with theory (`ln(K)` for K classes) | seconds | wrong loss, wrong output layer |
# | 3 | **Overfit one small batch** to ~zero loss | seconds | broken gradient flow, bad LR, bad loss wiring |
# | 4 | Check the learning rate and input scaling (sweep on a log grid) | minutes | slow learning, explosions |
# | 5 | Train on the full training set; compare with a simple baseline | minutes | "is the network even worth it?" |
# | 6 | Watch train vs validation curves; then regularize, not before | minutes | over/underfitting |
# | 7 | Check for leakage and split hygiene | minutes | fake-good results |
#
# ## 1 and 2. Sanity checks before any training
#
# With 10 balanced classes a network that knows nothing should predict 10% per class, so its cross-entropy should be `ln(10) = 2.303`. If the first loss is wildly different, something is wired wrong (an output
# activation, a missing scaling, a wrong number of classes).

# %%
model = make_model()
with torch.no_grad():
    first_loss = loss_fn(model(X_tr), y_tr).item()
print(f"initial loss {first_loss:.3f}   vs   ln(10) = {np.log(10):.3f}")
print("input range:", float(X_tr.min()), "to", float(X_tr.max()), "| dtypes:", X_tr.dtype, y_tr.dtype, "| label range:", int(y_tr.min()), "-", int(y_tr.max()))

# %% [markdown]
# ## 3. Overfit one batch
#
# Take **one** batch of 32 examples and train on it only. A correct network with enough capacity must drive the loss to nearly zero (it simply memorises 32 images). If it cannot, the problem is in the code, not in the data or the
# architecture. It is the single most valuable debugging test in deep learning because it takes seconds.

# %%
def overfit_one_batch(model, xb, yb, lr=1e-2, steps=300):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


xb, yb = X_tr[:32], y_tr[:32]
torch.manual_seed(0)
good = overfit_one_batch(make_model(), xb, yb)
print(f"correct setup:            loss {good[0]:.3f} -> {good[-1]:.4f}")

# Bug 1: the optimizer was given the wrong parameter list (easy to do with a second model around)
torch.manual_seed(0)
m = make_model()
other = make_model()
opt_bad = torch.optim.Adam(other.parameters(), lr=1e-2)        # should be m.parameters()
stuck = []
for _ in range(300):
    opt_bad.zero_grad()
    loss = loss_fn(m(xb), yb)
    loss.backward()
    opt_bad.step()
    stuck.append(loss.item())
print(f"optimizer on wrong params: loss {stuck[0]:.3f} -> {stuck[-1]:.4f}   (never moves)")

# Bug 2: learning rate of 1e-6
torch.manual_seed(0)
slow = overfit_one_batch(make_model(), xb, yb, lr=1e-6)
print(f"learning rate 1e-6:        loss {slow[0]:.3f} -> {slow[-1]:.4f}")

# %% [markdown]
# *What just happened?* A healthy setup goes from about `2.3` to nearly 0; both bugs leave the loss flat, and the test told us so within seconds and before touching any real data.
# **Important limit:** passing this test proves the plumbing works, *not* that the data is right. A network can memorise 32 examples even if their labels are garbage. Watch:

# %%
torch.manual_seed(0)
yb_wrong = yb[torch.randperm(len(yb))]                         # labels no longer match the images
wrong = overfit_one_batch(make_model(), xb, yb_wrong)
print(f"one batch with shuffled labels: loss -> {wrong[-1]:.4f}   (it 'passes' the overfit test!)")

torch.manual_seed(0)
y_shuffled_all = y_tr[torch.randperm(len(y_tr))]               # the same mistake over the whole training set
loss_bad, acc_bad = train(make_model(), X_tr, y_shuffled_all)
print(f"full training with shuffled labels: validation accuracy {acc_bad:.1%} (chance is 10%)")

# %% [markdown]
# Misaligned labels (for example shuffling `X` but not `y`, a real and common bug in hand-written data pipelines) slip past the overfit test and show up only as a validation score near chance.
# That is why step 1 (look at a few inputs with their labels) comes first, and why step 5 (a baseline) matters.

# %%
fig, axes = plt.subplots(1, 6, figsize=(9, 2))
for ax, i in zip(axes, range(6)):
    ax.imshow(X_tr[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"label {int(y_tr[i])}")
    ax.axis("off")
plt.suptitle("Step 1: do the labels match the pictures?")
plt.show()

# %% [markdown]
# ## 4. The loss is not going down: a playbook
#
# Ask, in this order: (1) is the **learning rate** sensible? (2) are the **inputs scaled**? (3) is `zero_grad()` called? (4) does the **loss match the output layer**? Here is each failure, run with the same
# model and data as the healthy reference, so you can compare like with like.

# %%
torch.manual_seed(0)
Xraw_tr = torch.tensor(train_test_split(X_raw, test_size=0.3, random_state=42, stratify=y_all)[0] * 10, dtype=torch.float32)   # pixels 0..160: badly scaled
experiments = {
    "healthy": lambda: train(make_model(), X_tr, y_tr),
    "learning rate too small (1e-5)": lambda: train(make_model(), X_tr, y_tr, lr=1e-5),
    "inputs not scaled (range 0-160)": lambda: train(make_model(), Xraw_tr, y_tr),
    "no zero_grad (gradients pile up)": lambda: train(make_model(), X_tr, y_tr, zero_grad=False),
    "softmax before CrossEntropyLoss": lambda: train(make_model(), X_tr, y_tr, softmax_first=True),
}
print(f"{'experiment':36s} {'final loss':>10s} {'val accuracy':>13s}")
for name, run in experiments.items():
    torch.manual_seed(0)
    loss_end, acc = run()
    print(f"{name:36s} {loss_end:10.3f} {acc:13.1%}")

# %% [markdown]
# Read each row as a signature:
#
# - **Too-small learning rate**: loss barely leaves `ln(10) = 2.30`. Fix: raise the rate; sweep on a log grid.
# - **Unscaled inputs**: huge activations saturate the network, and the gradient steps are too large for the chosen rate. Fix: scale inputs (divide by the max, or standardise with *training* statistics).
# - **No `zero_grad`**: each step uses the *sum of all previous gradients*, so the effective step keeps growing and the loss is noisy or high. Fix: call `optimizer.zero_grad()` every step (PyTorch accumulates by design).
# - **Softmax before `CrossEntropyLoss`**: `CrossEntropyLoss` applies log-softmax itself. Feeding it probabilities squashes the gradient signal, and training gets slow and worse, with no error message. Fix: pass raw logits.
#
# Notice that every one of these ran without an exception.

# %% [markdown]
# ## 5. Shape errors: the loud ones and the silent ones
#
# Loud shape errors (a `RuntimeError` naming the shapes) are the lucky case. Read them from the bottom: they say which operation failed and with which shapes.

# %%
try:
    nn.Linear(64, 10)(torch.zeros(5, 63))                 # wrong number of input features
except RuntimeError as e:
    print("loud:", str(e)[:100])

try:
    loss_fn(torch.randn(4, 10), torch.tensor([1.0, 2.0, 3.0, 4.0]))   # class labels must be integer indices
except (RuntimeError, TypeError) as e:
    print("loud:", str(e)[:100])

# %% [markdown]
# The silent one is **broadcasting**. A regression head usually outputs `(N, 1)` while the target is `(N,)`. `MSELoss` of those two does not raise: it broadcasts to an `(N, N)` grid of differences and averages
# everything. PyTorch only emits a *warning*, which is easy to miss in a notebook. Measure the damage:

# %%
torch.manual_seed(0)
pred = torch.randn(8, 1)
target = torch.randn(8)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    wrong_loss = nn.functional.mse_loss(pred, target).item()
right_loss = nn.functional.mse_loss(pred.squeeze(1), target).item()
print(f"MSE with shapes (8,1) vs (8,): {wrong_loss:.3f}   correct MSE: {right_loss:.3f}")
print("warning text:", str(caught[0].message)[:90] if caught else "none")

# %% [markdown]
# The "wrong" number is perfectly valid-looking, yet it is the mean over 64 pairs instead of 8. Training with it tends to regress towards predicting the mean of the targets. **Habit:** print the shape
# of `pred` and `target` right before the loss once, and treat warnings as errors while debugging (`warnings.simplefilter("error")`).
#
# ## 6. Exploding loss: `inf` and `nan`
#
# If the learning rate is too large for the scale of the problem, each step overshoots, the loss grows, the next gradient is even larger, and within a few steps the numbers overflow to `inf` and then `nan`.
# We show it on the simplest possible model, a line `y = 3x + 5` with inputs up to 100, written by hand so nothing is hidden.

# %%
rng = np.random.default_rng(0)
xs = torch.tensor(rng.uniform(0, 100, (200, 1)), dtype=torch.float32)
ys = 3 * xs + 5


def fit_line(x, y, lr, steps=30):
    w, b = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)
    losses = []
    for _ in range(steps):
        loss = ((x * w + b - y) ** 2).mean()
        w.grad = b.grad = None
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        losses.append(loss.item())
    return losses


exploded = fit_line(xs, ys, lr=0.1)
print("lr=0.1, raw inputs    :", [f"{v:.1e}" for v in exploded[:5]], "...", exploded[-1])
print("first non-finite step :", next((i for i, v in enumerate(exploded) if not np.isfinite(v)), None))

fix_small_lr = fit_line(xs, ys, lr=1e-5)
fix_scaled = fit_line((xs - xs.mean()) / xs.std(), ys, lr=0.3)   # standardised inputs
print(f"fix A (lr=1e-5, raw inputs): final loss {fix_small_lr[-1]:.1f}  (stable but slow)")
print(f"fix B (standardised, lr=0.3): final loss {fix_scaled[-1]:.1f}  (stable and fast: scaling is the real cure)")

# %% [markdown]
# Two tools for the general case. **Gradient clipping** caps the size of each update (`clip_grad_norm_`), which contains occasional spikes, for example in recurrent networks. It is a safety belt, not a fix for a wildly
# wrong learning rate. Compare on digits with an absurd rate:

# %%
for name, kw in [("lr=50, no clipping", dict(lr=50.0)), ("lr=50, clip norm 1.0", dict(lr=50.0, clip=1.0)), ("lr=0.1 (sensible)", dict(lr=0.1))]:
    torch.manual_seed(0)
    l, a = train(make_model(), X_tr, y_tr, **kw)
    print(f"{name:22s} final loss {l:8.3f}   val accuracy {a:.1%}")

# %% [markdown]
# Clipping does not rescue a learning rate that is 500x too large (both runs stay at chance); it only bounds the size of each update, which helps against occasional spikes, not against a wrong rate. The second tool finds **where** a `nan` first appears. `torch.autograd.detect_anomaly` makes the backward
# pass raise an error naming the operation that produced it (slow, so only while debugging):

# %%
x = torch.tensor([0.0, 4.0], requires_grad=True)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.autograd.detect_anomaly():
            (torch.sqrt(x) * 0).sum().backward()      # sqrt has an infinite slope at 0; times 0 gives nan
except RuntimeError as e:
    print("anomaly detection says:", str(e)[:80])

# %% [markdown]
# ## 7. Leakage: the bug that makes results look *too good*
#
# If the same example sits in both the training and validation sets, validation accuracy measures memory, not learning. Duplicates creep in through joins, repeated rows in logs and random splits of
# augmented data. Let us train a deliberately memory-heavy model (wide network, only 150 training images) and compare its accuracy on duplicated validation images vs untouched ones.

# %%
torch.manual_seed(0)
n_dup = 200
dup_X, dup_y = X_va[:n_dup], y_va[:n_dup]                      # these 200 validation images will ALSO be in training
clean_X, clean_y = X_va[n_dup:], y_va[n_dup:]
Xtrain_leaky = torch.cat([X_tr[:150], dup_X])
ytrain_leaky = torch.cat([y_tr[:150], dup_y])

model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for _ in range(200):
    opt.zero_grad()
    loss_fn(model(Xtrain_leaky), ytrain_leaky).backward()
    opt.step()
with torch.no_grad():
    acc_dup = (model(dup_X).argmax(1) == dup_y).float().mean().item()
    acc_clean = (model(clean_X).argmax(1) == clean_y).float().mean().item()
print(f"accuracy on images that leaked into training: {acc_dup:.1%}")
print(f"accuracy on genuinely unseen images         : {acc_clean:.1%}")

# %% [markdown]
# The leaked images are scored near perfectly because the model has seen their labels. A cheap guard is to hash or compare rows across splits before training:

# %%
def count_overlap(A, B):
    """Number of rows of B that also appear in A (exact match)."""
    seen = {row.tobytes() for row in A.numpy()}
    return sum(row.tobytes() in seen for row in B.numpy())


print("overlap between clean train and validation:", count_overlap(X_tr, X_va))
print("overlap between leaky train and validation:", count_overlap(Xtrain_leaky, X_va))

# %% [markdown]
# A related leak: fitting a scaler (mean and standard deviation) on *all* data before splitting. The effect is usually tiny, but it is still information flowing from the validation rows into training,
# and in time-series or small-data problems it can matter. The rule is the same as in the ML module: **split first, fit every statistic on the training part only.**

# %% [markdown]
# ## Common mistakes
#
# 1. **Tuning before sanity-checking.** Weeks lost on hyperparameters of a pipeline that could not pass the overfit-one-batch test.
# 2. **Debugging with the full dataset.** Shrink it: one batch, then 1,000 rows. Fast feedback beats clever guesses.
# 3. **Changing several things at once.** Change one thing, rerun, record the number. Keep a short log.
# 4. **Not seeding.** Without seeds you cannot tell a real improvement from noise. Compare across several seeds before believing a difference.
# 5. **Reading only the loss.** Watch the validation metric, the gradient norm, and a few predictions. A falling loss with chance-level accuracy means a labelling bug.

# %% [markdown]
# ## Interview corner
#
# **1. Your model's loss is not decreasing. What do you check first?**
# <details><summary>Answer</summary>
# Start with cheap, deterministic checks: verify data and labels by eye, compare the initial loss with ln(K), then overfit a single batch. If that fails, inspect the learning rate, input scaling,
# that <code>zero_grad()</code> and <code>step()</code> are called on the right parameters, and that the loss matches the output layer (logits into CrossEntropyLoss). Only then touch architecture or regularization.
# </details>
#
# **2. Why is overfitting a small batch a useful test?**
# <details><summary>Answer</summary>
# Any correct network with enough capacity can memorise a few examples, so a failure points at code (gradients, optimizer, loss, learning rate) rather than at data or model size. It does not validate data quality or labels.
# </details>
#
# **3. The loss went to nan. How do you find the cause?**
# <details><summary>Answer</summary>
# Lower the learning rate and check input scaling; look for log(0), division by zero or sqrt(0) in custom code; add epsilon terms; enable <code>torch.autograd.detect_anomaly()</code> to locate the operation;
# check the gradient norm each step; and use gradient clipping as a safety belt.
# </details>
#
# **4. Training accuracy is 99% and validation accuracy is 99%, but the model fails in production. Why might that be?**
# <details><summary>Answer</summary>
# Leakage (duplicates or shared groups across splits, preprocessing fit on all data), or a validation set that does not resemble production data (different time period, different users). Check the split strategy first.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell at the bottom of each exercise.

# %% [markdown]
# ### Exercise 1: Build the overfit-one-batch verdict
# Write `passes_overfit_test(model, xb, yb, lr, steps=300, threshold=0.05)` that trains `model` on the single batch with Adam (use `overfit_one_batch` from section 3) and returns `True` only if the **final loss is below the threshold**.
# The check calls it on a healthy setup and on one with a learning rate of `1e-7`.

# %% tags=["starter"]
def passes_overfit_test(model, xb, yb, lr, steps=300, threshold=0.05):
    ...

# %%
torch.manual_seed(0)
assert passes_overfit_test(make_model(), X_tr[:32], y_tr[:32], lr=1e-2) is True, "a healthy setup should pass"
torch.manual_seed(0)
assert passes_overfit_test(make_model(), X_tr[:32], y_tr[:32], lr=1e-7) is False, "a learning rate of 1e-7 should fail the test"
print("Exercise 1 passed")

# %% tags=["solution"]
def passes_overfit_test(model, xb, yb, lr, steps=300, threshold=0.05):
    losses = overfit_one_batch(model, xb, yb, lr=lr, steps=steps)
    return bool(losses[-1] < threshold)

# %% [markdown]
# ### Exercise 2: Fix the training loop
# The function below has **two** bugs: it never clears old gradients, and it puts a softmax in front of `CrossEntropyLoss`. Copy it into the starter cell as `fixed_train` and repair it.
# The check compares your loop against the healthy reference using the same seed, so the final validation accuracy has to match.

# %% tags=["starter"]
def fixed_train(model, Xtr, ytr, lr=0.1, epochs=20, batch_size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            out = model(Xtr[idx]).softmax(dim=1)
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (model(X_va).argmax(1) == y_va).float().mean().item()

# %%
torch.manual_seed(0)
expected = train(make_model(), X_tr, y_tr)[1]
torch.manual_seed(0)
got = fixed_train(make_model(), X_tr, y_tr)
assert abs(got - expected) < 1e-9, f"expected validation accuracy {expected:.4f} (healthy loop), got {got:.4f}: still a bug left"
print(f"Exercise 2 passed (validation accuracy {got:.1%})")

# %% tags=["solution"]
def fixed_train(model, Xtr, ytr, lr=0.1, epochs=20, batch_size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (model(X_va).argmax(1) == y_va).float().mean().item()

# %% [markdown]
# ### Exercise 3: A safe regression loss
# Write `safe_mse(pred, target)` that returns the mean squared error for `pred` of shape `(N, 1)` or `(N,)` and `target` of shape `(N,)` **without any broadcasting accident**. It must raise a `ValueError`
# if, after your reshaping, the number of elements differs.

# %% tags=["starter"]
def safe_mse(pred, target):
    ...

# %%
p, t = torch.randn(10, 1), torch.randn(10)
expected = ((p[:, 0] - t) ** 2).mean()
assert torch.isclose(safe_mse(p, t), expected), "MSE should average over N pairs, not N*N"
assert torch.isclose(safe_mse(p[:, 0], t), expected), "it should also accept (N,) predictions"
try:
    safe_mse(torch.randn(10, 1), torch.randn(9))
    raise AssertionError("a size mismatch (10 vs 9) must raise ValueError")
except ValueError:
    pass
print("Exercise 3 passed")

# %% tags=["solution"]
def safe_mse(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f"size mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return ((pred - target) ** 2).mean()

# %% [markdown]
# ### Exercise 4: A gradient-norm monitor
# Write `global_grad_norm(model)` returning the L2 norm of **all** parameter gradients taken together as one vector (a Python float). Watching this number every few steps reveals vanishing and exploding gradients early.
# The check compares it with the value `torch.nn.utils.clip_grad_norm_` reports.

# %% tags=["starter"]
def global_grad_norm(model):
    ...

# %%
torch.manual_seed(0)
m = make_model()
loss_fn(m(X_tr[:64]), y_tr[:64]).backward()
reference = float(nn.utils.clip_grad_norm_(m.parameters(), max_norm=1e9))       # huge max_norm: only reports, never clips
mine = global_grad_norm(m)
assert isinstance(mine, float), "return a Python float (use .item())"
assert abs(mine - reference) < 1e-5 * max(1.0, reference), f"expected {reference:.6f}, got {mine:.6f}"
print(f"Exercise 4 passed (gradient norm {mine:.4f})")

# %% tags=["solution"]
def global_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float((p.grad ** 2).sum())
    return total ** 0.5

# %% [markdown]
# ### Exercise 5 (open-ended): Break it yourself
# Invent a bug that is *not* in this notebook (for example: dropout left on during evaluation, a wrong `view` that scrambles pixels, a learning rate schedule that goes the wrong way, normalising labels as if they were features)
# and reproduce it on the digits MLP. Record the validation accuracy and how it differs from the healthy `93-97%`. Then write the one-line symptom-to-cause rule you would give a teammate. No assert: explore and report.

# %% [markdown]
# ## Recap
# - Run the checklist cheapest-first: look at data and labels, compare the first loss with `ln(K)`, overfit one batch, sweep the learning rate, then train fully against a baseline.
# - Overfit-one-batch proves the plumbing works; it cannot catch mislabeled data or leakage.
# - Silent killers: no `zero_grad`, softmax before `CrossEntropyLoss`, unscaled inputs, `(N,1)` vs `(N,)` broadcasting, and learning rates that are far too high or low.
# - For `nan`: lower the rate, scale the inputs, guard `log`/`sqrt`/division, use `detect_anomaly` to locate it; clipping is a safety belt, not a cure.
# - Split first, then fit every statistic on the training split; check for duplicates across splits.
#
# **Go deeper:** Andrej Karpathy's blog post "A Recipe for Training Neural Networks"; the PyTorch docs for `torch.autograd.detect_anomaly` and `torch.nn.utils.clip_grad_norm_` (pytorch.org/docs);
# Google's "Deep Learning Tuning Playbook" (published on GitHub by Google Research).
#
# **Next:** the [PyTorch module](../14_PyTorch/README.md), where you rebuild all of this with `nn.Module`, `Dataset` and `DataLoader`.
