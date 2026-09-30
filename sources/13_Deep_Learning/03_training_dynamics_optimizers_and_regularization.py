# %% [markdown]
# # Training Dynamics: Initialisation, Optimizers and Regularization
#
# **Why this matters.** A network that is correct on paper can still refuse to learn (loss stuck at `2.30`), learn slowly, or learn the
# training set perfectly and fail on new data. Those three symptoms trace back to three choices you make before `fit()`:
# **how the weights start**, **which optimizer moves them**, and **what stops the model from memorising**. Knowing these
# knobs is what separates "I ran a tutorial" from "I can rescue a model that does not train".
#
# **You will learn**
# - measure how activations shrink or blow up with depth, and pick an initialisation (Xavier, He) that keeps them stable
# - see vanishing gradients directly, by printing the gradient size in every layer
# - implement SGD, momentum and Adam updates by hand, then compare the `torch.optim` versions on digits
# - measure what dropout, weight decay and early stopping buy you (and where the gain is small)
#
# **Prerequisites:** [01 Neurons, layers and the forward pass](01_neurons_layers_and_the_forward_pass.ipynb) and
# [02 Backpropagation from scratch](02_backpropagation_from_scratch.ipynb). This notebook uses PyTorch only as a fast
# calculator for gradients; the ideas are framework-independent.
#
# > Data used: scikit-learn's bundled `digits` (1,797 real 8x8 handwritten digit images, pixels scaled to 0-1).

# %%
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
torch.set_num_threads(2)
np.set_printoptions(precision=4, suppress=True)

X, y = load_digits(return_X_y=True)
X = X / 16.0                                   # pixels are 0..16
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_tr, X_va = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_va, dtype=torch.float32)
y_tr, y_va = torch.tensor(y_tr), torch.tensor(y_va)
print("train:", tuple(X_tr.shape), " validation:", tuple(X_va.shape))

# %% [markdown]
# ## 1. Initialisation: the signal has to survive the trip through the layers
#
# Each layer multiplies its input by a random matrix. If that matrix scales the signal down even a little, the *same* shrinkage repeats at every layer and
# the output decays exponentially with depth (`0.5^30` is about `1e-9`). Scale up a little and it explodes instead. The fix is to choose the weight variance so
# that one layer **preserves** the spread of its input:
#
# | Scheme | Weight std | Designed for |
# |---|---|---|
# | Xavier / Glorot | `sqrt(1 / fan_in)` (for square layers) | tanh, sigmoid (roughly linear near 0) |
# | He / Kaiming | `sqrt(2 / fan_in)` | ReLU (which zeroes half the signal, so needs 2x variance) |
#
# Let us check that empirically: push random data through 30 layers of width 256 and record the standard deviation of the activations after each layer.

# %%
def activation_stds(act, weight_std, depth=30, width=256, n=512, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, width))                # input with std 1
    stds = []
    for _ in range(depth):
        W = weight_std(width) * rng.standard_normal((width, width))
        h = act(h @ W)
        stds.append(h.std())
    return np.array(stds), h


relu = lambda z: np.maximum(z, 0.0)
cases = {
    "tanh, std=0.01":   (np.tanh, lambda w: 0.01),
    "tanh, std=1.0":    (np.tanh, lambda w: 1.0),
    "tanh, Xavier":     (np.tanh, lambda w: np.sqrt(1.0 / w)),
    "ReLU, Xavier":     (relu,    lambda w: np.sqrt(1.0 / w)),
    "ReLU, He":         (relu,    lambda w: np.sqrt(2.0 / w)),
}
results = {}
fig, ax = plt.subplots(figsize=(7, 4))
for name, (act, wstd) in cases.items():
    stds, h_last = activation_stds(act, wstd)
    results[name] = (stds, h_last)
    ax.plot(range(1, len(stds) + 1), stds, label=name)
ax.set_yscale("log")
ax.set_xlabel("layer number")
ax.set_ylabel("std of activations (log scale)")
ax.set_title("Activation spread through 30 layers")
ax.legend(fontsize=8)
plt.show()
for name, (stds, h_last) in results.items():
    print(f"{name:16s} std after layer 30 = {stds[-1]:.2e}")

# %% [markdown]
# *What just happened?* Tiny weights make the signal vanish (around `1e-24`: the network outputs a constant whatever you feed it). Xavier keeps tanh usable but the std still drifts down;
# ReLU with Xavier shrinks the signal by a factor of roughly `sqrt(2)` per layer, which He initialisation corrects.
# Notice that `std=1.0` with tanh *looks* stable, yet it is a trap. Check how many units are saturated (output almost exactly +/-1, where the slope of tanh is about zero):

# %%
for name in ["tanh, std=1.0", "tanh, Xavier"]:
    h_last = results[name][1]
    print(f"{name:14s}: {np.mean(np.abs(h_last) > 0.99):.0%} of last-layer units saturated")

# %% [markdown]
# A saturated unit passes almost no gradient back, so a well-sized *forward* signal is not enough. That leads to the second symptom.
#
# ## 2. Vanishing gradients, measured
#
# Backpropagation multiplies one local derivative per layer. The sigmoid's slope is at most `0.25`, so ten sigmoid layers can multiply the gradient by
# `0.25^10 = 1e-6` before it reaches the first layer. We will build a 10-hidden-layer network with three activations (all using a sensible initialisation
# for that activation), run one backward pass, and print the gradient norm of each layer's weights.

# %%
def deep_mlp(act_cls, init_fn, depth=10, width=64):
    layers, d = [], 64
    for _ in range(depth):
        layers += [nn.Linear(d, width), act_cls()]
        d = width
    layers.append(nn.Linear(d, 10))
    model = nn.Sequential(*layers)
    for m in model:
        if isinstance(m, nn.Linear):
            init_fn(m.weight)
            nn.init.zeros_(m.bias)
    return model


xavier = nn.init.xavier_normal_
he = lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu")
configs = {"sigmoid": (nn.Sigmoid, xavier), "tanh": (nn.Tanh, xavier), "ReLU": (nn.ReLU, he)}

loss_fn = nn.CrossEntropyLoss()
grad_norms = {}
for name, (act_cls, init_fn) in configs.items():
    torch.manual_seed(0)
    model = deep_mlp(act_cls, init_fn)
    loss_fn(model(X_tr), y_tr).backward()
    grad_norms[name] = [m.weight.grad.norm().item() for m in model if isinstance(m, nn.Linear)]

fig, ax = plt.subplots(figsize=(7, 4))
for name, g in grad_norms.items():
    ax.plot(range(1, len(g) + 1), g, marker="o", label=name)
ax.set_yscale("log")
ax.set_xlabel("layer (1 = closest to the input)")
ax.set_ylabel("gradient norm of weights (log scale)")
ax.set_title("Gradient size per layer, 10 hidden layers")
ax.legend()
plt.show()
for name, g in grad_norms.items():
    print(f"{name:8s} first-layer / last-layer gradient = {g[0] / g[-1]:.1e}")

# %% [markdown]
# The sigmoid network's first layers get gradients many orders of magnitude smaller than its last layer, so they barely learn. Does that really stop training? Train each network
# with plain SGD for 300 full-batch steps and see:

# %%
def train_full_batch(model, steps=300, lr=0.1):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(X_va).argmax(1) == y_va).float().mean().item()
    return loss.item(), acc


print(f"chance-level loss for 10 classes = ln(10) = {np.log(10):.3f}")
for name, (act_cls, init_fn) in configs.items():
    torch.manual_seed(0)
    loss_end, acc = train_full_batch(deep_mlp(act_cls, init_fn))
    print(f"{name:8s} final train loss {loss_end:.3f}   validation accuracy {acc:.1%}")

# %% [markdown]
# *What just happened?* The deep sigmoid network sits at the chance-level loss `ln(10)` and predicts one class: its early layers never receive a usable gradient signal.
# Tanh and ReLU (with a matching initialisation) train the same depth fine. This is why ReLU-family activations and careful initialisation became standard,
# and why later architectures add skip connections and normalisation layers.

# %% [markdown]
# ## 3. Optimizers: same gradient, different step
#
# Plain **SGD** moves against the gradient: `p = p - lr * g`. Two well-known improvements:
#
# - **Momentum** keeps a running velocity `v = beta*v + g` and steps along `v`. Consistent directions add up; zig-zags cancel.
# - **Adam** keeps running averages of the gradient `m` and of its square `v`, corrects their start-up bias, and divides: `p = p - lr * m_hat / (sqrt(v_hat) + eps)`.
#   Each parameter gets its own effective step size.
#
# First by hand on a tiny valley, `loss = 0.5 * (1 * x^2 + 30 * y^2)`: steep in `y`, shallow in `x` (the classic case where plain gradient descent is slow along the shallow direction).

# %%
curv = np.array([1.0, 30.0])


def valley_loss(p):
    return 0.5 * float((curv * p * p).sum())


def sgd_step(p, g, state, t, lr=0.03):
    return p - lr * g


def momentum_step(p, g, state, t, lr=0.03, beta=0.9):
    state["v"] = beta * state.get("v", 0.0) + g
    return p - lr * state["v"]


def adam_step(p, g, state, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
    state["m"] = b1 * state.get("m", 0.0) + (1 - b1) * g
    state["v"] = b2 * state.get("v", 0.0) + (1 - b2) * g * g
    m_hat = state["m"] / (1 - b1 ** t)
    v_hat = state["v"] / (1 - b2 ** t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps)


def descend(step_fn, n_steps=100):
    p, state, losses = np.array([1.0, 1.0]), {}, []
    for t in range(1, n_steps + 1):
        g = curv * p                              # gradient of the valley
        p = step_fn(p, g, state, t)
        losses.append(valley_loss(p))
    return losses


valley = {"SGD": descend(sgd_step), "Momentum": descend(momentum_step), "Adam": descend(adam_step)}
fig, ax = plt.subplots(figsize=(7, 4))
for name, l in valley.items():
    ax.plot(l, label=name)
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("loss (log scale)")
ax.set_title("Three optimizers on a narrow valley")
ax.legend()
plt.show()
for name, l in valley.items():
    first = next((i + 1 for i, v in enumerate(l) if v < 1e-3), None)
    print(f"{name:9s} steps to reach loss < 1e-3: {first}")

# %% [markdown]
# Read the table above rather than trusting a slogan: the number of steps each method needs depends on the learning rate we hand-picked for it. This is one toy problem, not a law.
# Now the same three on a real (small) network, using `torch.optim`:

# %%
def make_mlp(hidden=64, p_drop=0.0):
    return nn.Sequential(
        nn.Linear(64, hidden), nn.ReLU(), nn.Dropout(p_drop),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
        nn.Linear(hidden, 10),
    )


def fit(model, opt, Xtr, ytr, epochs=30, batch_size=64, seed=0):
    """Mini-batch training. Returns per-epoch (train loss, val loss, val accuracy) on the full sets."""
    g = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            history.append((loss_fn(model(Xtr), ytr).item(),
                            loss_fn(model(X_va), y_va).item(),
                            (model(X_va).argmax(1) == y_va).float().mean().item()))
    return np.array(history)


optimizers = {
    "SGD lr=0.05":            lambda p: torch.optim.SGD(p, lr=0.05),
    "SGD+momentum lr=0.05":   lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "Adam lr=1e-3":           lambda p: torch.optim.Adam(p, lr=1e-3),
}
opt_hist = {}
for name, make_opt in optimizers.items():
    torch.manual_seed(0)
    m = make_mlp()
    opt_hist[name] = fit(m, make_opt(m.parameters()), X_tr, y_tr)

fig, ax = plt.subplots(figsize=(7, 4))
for name, h in opt_hist.items():
    ax.plot(h[:, 0], label=name)
ax.set_xlabel("epoch")
ax.set_ylabel("training loss")
ax.set_title("Same network, three optimizers (digits)")
ax.legend()
plt.show()
for name, h in opt_hist.items():
    print(f"{name:22s} train loss after 30 epochs {h[-1, 0]:.3f}   validation accuracy {h[-1, 2]:.1%}")

# %% [markdown]
# *What just happened?* With these (untuned) learning rates, plain SGD is slowest in training loss. Do not read "Adam always wins" from this:
# given time and a good learning rate, well-tuned SGD with momentum matches or beats Adam on many problems, and Adam's default `1e-3` is simply a forgiving starting point.
# The learning rate matters more than the optimizer choice. Sweep it once to see why:

# %%
lrs = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
for lr in lrs:
    torch.manual_seed(0)
    m = make_mlp()
    h = fit(m, torch.optim.Adam(m.parameters(), lr=lr), X_tr, y_tr, epochs=10)
    print(f"Adam lr={lr:<7g} validation loss after 10 epochs: {h[-1, 1]:.3f}")

# %% [markdown]
# Too small learns slowly, too large bounces or diverges, and there is a good region in between. A "learning-rate finder" is exactly this sweep, done on a log grid, before a long run.
#
# ## 4. Regularization: fighting memorisation
#
# Large networks can memorise a small training set. We simulate that regime: only **150 training images**, a wide network (256 units), and 5 random seeds for every setting so that we do
# not mistake noise for an effect. Three remedies:
#
# - **Nothing** (baseline).
# - **Dropout** (`p=0.5`): randomly zero half the hidden units on each training step; at evaluation time use all of them (`model.eval()`!).
# - **Weight decay** (AdamW, `0.5`): shrink weights a little every step, which discourages large, brittle weights.
#
# The fourth remedy, **early stopping** (keep the weights from the epoch with the best validation loss), comes after.

# %%
rng = np.random.default_rng(0)
small = torch.tensor(rng.choice(len(X_tr), size=150, replace=False))
Xs, ys = X_tr[small], y_tr[small]

settings = {
    "no regularization": dict(p_drop=0.0, opt=lambda p: torch.optim.Adam(p, lr=3e-3)),
    "dropout 0.5":       dict(p_drop=0.5, opt=lambda p: torch.optim.Adam(p, lr=3e-3)),
    "weight decay 0.5":  dict(p_drop=0.0, opt=lambda p: torch.optim.AdamW(p, lr=3e-3, weight_decay=0.5)),
}
reg_hist = {name: [] for name in settings}
for name, cfg in settings.items():
    for seed in range(5):
        torch.manual_seed(seed)
        m = make_mlp(hidden=256, p_drop=cfg["p_drop"])
        reg_hist[name].append(fit(m, cfg["opt"](m.parameters()), Xs, ys, epochs=150, batch_size=32, seed=seed))
    reg_hist[name] = np.array(reg_hist[name])          # (seeds, epochs, 3)

print(f"{'setting':18s} {'val loss @ end':>15s} {'best val loss':>14s} {'val acc @ end':>14s}")
for name, h in reg_hist.items():
    print(f"{name:18s} {h[:, -1, 1].mean():15.3f} {h[:, :, 1].min(axis=1).mean():14.3f} {h[:, -1, 2].mean():14.1%}")

# %% [markdown]
# The "val loss @ end" column exposes overfitting: a model that memorised the 150 images is *confidently wrong* on new ones, which inflates cross-entropy even when accuracy holds up.
# Plot the mean validation loss per epoch:

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for name, h in reg_hist.items():
    ax.plot(h[:, :, 1].mean(axis=0), label=name)
ax.set_xlabel("epoch")
ax.set_ylabel("validation loss (mean of 5 seeds)")
ax.set_title("Validation loss with 150 training images")
ax.legend()
plt.show()
base = reg_hist["no regularization"]
print("epoch of lowest mean val loss (no regularization):", int(base[:, :, 1].mean(axis=0).argmin()) + 1, "of 150")

# %% [markdown]
# Without regularization the validation loss reaches a minimum early and then **climbs** while the training loss keeps falling: the textbook overfitting picture.
# Compare the end-of-training numbers of the regularized runs with the baseline in the table. Early stopping is the cheapest fix of all, because it needs no change to the model.
# The rule: stop when validation loss has not improved for `patience` epochs, and restore the best weights.

# %%
def fit_early_stopping(model, opt, Xtr, ytr, max_epochs=150, patience=15, batch_size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item()
        if val_loss < best_loss:
            best_loss, best_state, best_epoch, waited = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    return best_epoch + 1, epoch + 1


es_loss, es_acc, es_epochs = [], [], []
for seed in range(5):
    torch.manual_seed(seed)
    m = make_mlp(hidden=256)
    best_ep, stopped = fit_early_stopping(m, torch.optim.Adam(m.parameters(), lr=3e-3), Xs, ys, seed=seed)
    with torch.no_grad():
        es_loss.append(loss_fn(m(X_va), y_va).item())
        es_acc.append((m(X_va).argmax(1) == y_va).float().mean().item())
    es_epochs.append(stopped)
print(f"early stopping: val loss {np.mean(es_loss):.3f}, val accuracy {np.mean(es_acc):.1%}, stopped after ~{np.mean(es_epochs):.0f} epochs (max 150)")

# %% [markdown]
# *What just happened?* Early stopping lands near the best validation loss of the unregularized run (the "best val loss" column) while training for a fraction of the epochs.
# One honest caveat: we used the *validation* set both to stop and to report. For a final number you need a separate test set, otherwise the score is optimistically biased (same lesson as in the ML module).
# Also notice the accuracy differences between settings are small compared with the loss differences; with 540 validation images, a gap of one or two points is within noise.

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting `model.eval()` when measuring** a model with dropout: predictions stay random and noisy. Switch back with `model.train()` before training again.
# 2. **Deep sigmoid/tanh stacks with default or tiny init**, then blaming the optimizer. Check the per-layer gradient norms first.
# 3. **Changing the optimizer and the learning rate at once.** If Adam at `1e-3` beats SGD at `1e-3`, you learned nothing: SGD wants a bigger rate. Compare each at its best rate.
# 4. **Using weight decay inside plain `Adam`.** In `Adam`, `weight_decay` adds an L2 term to the gradient, which Adam then rescales; `AdamW` decouples it. Prefer `AdamW` when you want real weight decay.
# 5. **Early stopping on the test set.** Stopping on the data you report is selection on the test set.

# %%
# Mistake 1 in numbers: the same trained dropout model, evaluated in train mode vs eval mode
torch.manual_seed(0)
m = make_mlp(hidden=256, p_drop=0.5)
fit(m, torch.optim.Adam(m.parameters(), lr=3e-3), Xs, ys, epochs=30, batch_size=32)
m.train()
with torch.no_grad():
    a, b = m(X_va[:5]), m(X_va[:5])
print("train mode: two passes on the same input agree?", torch.allclose(a, b))
m.eval()
with torch.no_grad():
    a, b = m(X_va[:5]), m(X_va[:5])
print("eval mode : two passes on the same input agree?", torch.allclose(a, b))

# %% [markdown]
# ## Interview corner
#
# **1. Why can't you initialise all weights to zero (or all to the same constant)?**
# <details><summary>Answer</summary>
# Every unit in a layer would compute the same output and receive the same gradient, so they would stay identical forever: the layer behaves like one unit.
# Random initialisation breaks that symmetry. (Biases can start at zero.)
# </details>
#
# **2. What is the vanishing gradient problem and how do you mitigate it?**
# <details><summary>Answer</summary>
# Backprop multiplies many local derivatives; if each is below 1 (sigmoid's slope is at most 0.25) the gradient reaching early layers shrinks exponentially, so they stop learning.
# Mitigations: ReLU-family activations, He/Xavier initialisation, normalisation layers, skip (residual) connections, and gated units in recurrent nets.
# </details>
#
# **3. How does Adam differ from SGD with momentum?**
# <details><summary>Answer</summary>
# Momentum smooths the gradient direction with one learning rate for all parameters. Adam also tracks the running average of squared gradients and divides by its root, so each parameter
# gets its own adaptive step size, plus bias correction for the first steps. Adam is forgiving to tune; SGD+momentum often generalises as well or better when carefully tuned.
# </details>
#
# **4. Dropout at train vs test time?**
# <details><summary>Answer</summary>
# During training each unit is zeroed with probability p and the survivors are scaled by 1/(1-p) (PyTorch's "inverted dropout"), so at test time you simply use the full network
# with no scaling. In PyTorch that means <code>model.eval()</code> disables dropout.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell at the bottom of each exercise.

# %% [markdown]
# ### Exercise 1: Choose a weight scale that keeps ReLU activations stable
# Write `my_weight_std(width)` returning the weight standard deviation that should keep the activation std roughly constant through deep ReLU layers.
# The helper `activation_stds` (above) will test it over 30 layers of width 256. *Hint: He initialisation.*

# %% tags=["starter"]
def my_weight_std(width):
    ...

# %%
stds, _ = activation_stds(relu, my_weight_std, depth=30)
xavier_stds, _ = activation_stds(relu, lambda w: np.sqrt(1.0 / w), depth=30)
ratio = stds[-1] / stds[0]
assert 0.3 < ratio < 3.0, f"activation std changed by a factor of {ratio:.3g} over 30 layers; it should stay within about 3x"
assert stds[-1] > 100 * xavier_stds[-1], "your scale should be far better than Xavier for ReLU"
print(f"Exercise 1 passed (std went {stds[0]:.2f} -> {stds[-1]:.2f}; Xavier ended at {xavier_stds[-1]:.1e})")

# %% tags=["solution"]
def my_weight_std(width):
    return np.sqrt(2.0 / width)

# %% [markdown]
# ### Exercise 2: Adam by hand, checked against PyTorch
# Complete `my_adam(grads, lr=0.01)`: starting from the parameter vector `[1.0, -2.0, 0.5]`, apply Adam (`beta1=0.9`, `beta2=0.999`, `eps=1e-8`) for each gradient in `grads` (one step per gradient,
# with the bias correction) and return the final parameters as a NumPy array. The check compares it with `torch.optim.Adam`. *Hint: reuse the `adam_step` function from section 3.*

# %%
grads_seq = [np.array([0.5, -1.0, 2.0]), np.array([0.4, -0.8, 1.0]), np.array([-0.1, 0.3, 0.2]), np.array([0.2, 0.2, -0.5])]

# %% tags=["starter"]
def my_adam(grads, lr=0.01):
    p = np.array([1.0, -2.0, 0.5])
    ...
    return p

# %%
result = my_adam(grads_seq)
param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64))
ref_opt = torch.optim.Adam([param], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
for g in grads_seq:
    ref_opt.zero_grad()
    param.grad = torch.tensor(g, dtype=torch.float64)
    ref_opt.step()
assert np.allclose(result, param.detach().numpy(), atol=1e-8), f"expected {param.detach().numpy()}, got {result}"
print("Exercise 2 passed:", result)

# %% tags=["solution"]
def my_adam(grads, lr=0.01):
    p = np.array([1.0, -2.0, 0.5])
    state = {}
    for t, g in enumerate(grads, start=1):
        p = adam_step(p, g, state, t, lr=lr)
    return p

# %% [markdown]
# ### Exercise 3: Early stopping as a pure function
# Write `stop_epoch(val_losses, patience)`. Given the list of validation losses recorded after each epoch (index 0 = epoch 1), return a tuple `(best_epoch, stopped_epoch)` using the rule from the notebook:
# track the best loss; stop at the first epoch where `patience` consecutive epochs have passed without a *new* best. If that never happens, `stopped_epoch` is the last epoch. Epochs are 1-based.

# %% tags=["starter"]
def stop_epoch(val_losses, patience):
    ...

# %%
rng_ex = np.random.default_rng(3)
curve = list(0.3 + (np.arange(60) - 20.0) ** 2 / 800 + 0.02 * rng_ex.standard_normal(60))   # U-shaped + noise
best_e, stop_e = stop_epoch(curve, patience=8)
assert best_e == int(np.argmin(curve[:stop_e])) + 1, "best_epoch must be the argmin of the losses seen up to the stopping epoch"
assert stop_e - best_e == 8, "training should stop exactly `patience` epochs after the last improvement"
assert stop_e < len(curve), "with a U-shaped curve the rule should fire before the last epoch"
assert stop_epoch([1.0, 0.9, 0.8, 0.7], patience=3) == (4, 4), "a steadily improving run should never stop early"
print(f"Exercise 3 passed (best epoch {best_e}, stopped at {stop_e} of {len(curve)})")

# %% tags=["solution"]
def stop_epoch(val_losses, patience):
    best, best_epoch, waited = float("inf"), 0, 0
    for epoch, loss in enumerate(val_losses, start=1):
        if loss < best:
            best, best_epoch, waited = loss, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                return best_epoch, epoch
    return best_epoch, len(val_losses)

# %% [markdown]
# ### Exercise 4: Does dropout help a model that cannot overfit?
# Using the **full** training set (`X_tr`, `y_tr`, about 1,250 images) train `make_mlp(hidden=64, p_drop=0.0)` and `make_mlp(hidden=64, p_drop=0.5)` with Adam `lr=3e-3` for 30 epochs
# (use `fit`, and `torch.manual_seed(0)` before building each). Store the final validation accuracies in `acc_plain` and `acc_dropout`. The check only asserts that both are above 90% (both models work);
# the interesting part is what you see: write one sentence on whether heavy dropout helped a small network with plenty of data.

# %% tags=["starter"]
acc_plain = ...
acc_dropout = ...

# %%
assert 0.9 < acc_plain <= 1.0 and 0.9 < acc_dropout <= 1.0, f"both models should exceed 90% validation accuracy, got {acc_plain:.3f} and {acc_dropout:.3f}"
print(f"Exercise 4 passed (plain {acc_plain:.1%}, dropout {acc_dropout:.1%})")

# %% tags=["solution"]
accs = []
for p in (0.0, 0.5):
    torch.manual_seed(0)
    m = make_mlp(hidden=64, p_drop=p)
    accs.append(fit(m, torch.optim.Adam(m.parameters(), lr=3e-3), X_tr, y_tr, epochs=30)[-1, 2])
acc_plain, acc_dropout = accs

# %% [markdown]
# ### Exercise 5 (open-ended): Find a better recipe for the 150-image problem
# Try one change of your own on the 150-image setup (different dropout rate, weight decay value, hidden size, learning rate, or a combination) and compare to the table in section 4
# using at least 3 seeds. Report what you tried and whether the difference is larger than the seed-to-seed spread. No assert: explore and report.

# %% [markdown]
# ## Recap
# - Initialisation must keep activation (and gradient) spread stable with depth: Xavier for tanh/sigmoid, He for ReLU. All-zero weights never break symmetry.
# - Deep sigmoid stacks lose gradient exponentially; measure per-layer gradient norms to diagnose it.
# - Momentum smooths the step; Adam adapts it per parameter. The learning rate matters more than the optimizer; sweep it on a log grid.
# - Dropout, weight decay and early stopping limit memorisation. Judge them on validation **loss** over several seeds, and expect small accuracy gaps.
# - `model.eval()` for measuring; keep validation (for decisions) and test (for the final number) separate.
#
# **Go deeper:** the PyTorch docs for `torch.optim` and `torch.nn.init` (pytorch.org/docs); Stanford CS231n notes on neural networks (cs231n.github.io);
# the Adam paper (Kingma and Ba, "Adam: A Method for Stochastic Optimization") and the AdamW paper (Loshchilov and Hutter, "Decoupled Weight Decay Regularization").
#
# **Next:** [Convolutions and CNN intuition](04_convolutions_and_cnn_intuition.ipynb)
