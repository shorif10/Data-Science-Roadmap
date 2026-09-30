# %% [markdown]
# # Saving, Loading, Reproducibility and Devices
#
# **Why this matters.** Training takes minutes or days; the model must then *outlive the notebook*: load it in a service, resume a run that crashed at epoch 37, hand it to a teammate and get the same predictions.
# Three practical skills make that possible: saving and loading the right things (the `state_dict`, and for resuming, the optimizer too), controlling randomness so a result can be reproduced, and writing code that runs
# on a laptop CPU and a GPU without edits. These are the things that go wrong the first time someone else tries to use your model.
#
# **You will learn**
# - read a `state_dict`, and know which entries are parameters and which are buffers (batch-norm statistics)
# - save and restore a model with `torch.save` / `torch.load`, and prove the round trip with identical predictions
# - write a checkpoint that lets you resume training and reproduces the uninterrupted run exactly
# - seed Python, NumPy and PyTorch, understand what deterministic mode does, and measure seed-to-seed variation
# - write device-agnostic code (`device`, `.to(device)`, `map_location`)
#
# **Prerequisites:** [02 `nn.Module` and the training loop](02_nn_module_and_training_loop.ipynb) and [03 `Dataset`/`DataLoader`](03_classification_with_dataset_and_dataloader.ipynb).
#
# > Data used: scikit-learn's bundled `digits` (1,797 real 8x8 handwritten digits). All files are written to a temporary directory that is deleted at the end of each cell that needs one.

# %%
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(2)
print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available())

X, y = load_digits(return_X_y=True)
X_tr, X_va, y_tr, y_va = train_test_split(X / 16.0, y, test_size=0.3, random_state=42, stratify=y)
X_tr, X_va = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_va, dtype=torch.float32)
y_tr, y_va = torch.tensor(y_tr), torch.tensor(y_va)
loss_fn = nn.CrossEntropyLoss()


def make_model():
    return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 10))


def train_epochs(model, opt, start, end, batch_size=64, data_seed=0):
    """Train epochs start..end-1. The shuffle order of epoch e depends only on (data_seed, e), so a resumed run sees the same batches."""
    for epoch in range(start, end):
        model.train()
        g = torch.Generator().manual_seed(data_seed * 1000 + epoch)
        perm = torch.randperm(len(X_tr), generator=g)
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X_tr[idx]), y_tr[idx]).backward()
            opt.step()


@torch.no_grad()
def accuracy(model, X_, y_):
    model.eval()
    return (model(X_).argmax(1) == y_).float().mean().item()


# %% [markdown]
# ## 1. What is in a `state_dict`?
#
# A model's learned state is a plain ordered dictionary mapping names to tensors. It contains the **parameters** (what the optimizer updates) and **buffers** (tensors that belong to the model but are not trained by
# gradient descent, such as the running mean and variance of batch normalisation). Our model has a `BatchNorm1d`, so we can see both.

# %%
model = make_model()
for name, tensor in model.state_dict().items():
    kind = "buffer   " if name in dict(model.named_buffers()) else "parameter"
    print(f"{name:22s} {kind} {tuple(tensor.shape)}")

# %% [markdown]
# `running_mean`, `running_var` and `num_batches_tracked` are buffers: they are saved in the `state_dict` (you need them at inference time) but `model.parameters()` does not list them and the optimizer never sees them.
# If you saved only `parameters()`, a batch-norm model would silently behave differently after loading.
#
# ## 2. Save and load the `state_dict`
#
# The recommended pattern: save only the `state_dict`; to load, **build the model from code first**, then load the weights into it. (`torch.save(model)` pickles the whole object, which ties the file to your exact
# class and file layout, and pickle can execute code when loading, so recent PyTorch versions refuse it by default, as we will see.)

# %%
torch.manual_seed(0)
model = make_model()
train_epochs(model, torch.optim.Adam(model.parameters(), lr=1e-2), 0, 5)
print(f"trained model validation accuracy: {accuracy(model, X_va, y_va):.1%}")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "digits_mlp.pt"
    torch.save(model.state_dict(), path)
    print("file size:", path.stat().st_size, "bytes")

    restored = make_model()                                   # same architecture, fresh random weights
    print("before loading, accuracy:", f"{accuracy(restored, X_va, y_va):.1%}")
    restored.load_state_dict(torch.load(path))                # weights_only=True is the default in current PyTorch
    print("after loading,  accuracy:", f"{accuracy(restored, X_va, y_va):.1%}")

model.eval()
restored.eval()
with torch.no_grad():
    same = torch.equal(model(X_va), restored(X_va))
print("predictions bit-for-bit identical:", same)

# %% [markdown]
# *What just happened?* A fresh model (random weights, near-chance accuracy) became an exact copy of the trained one after `load_state_dict`. The equality check of the raw outputs is the test to run after any save/load code
# you write: if it is `False`, something (usually `eval()` mode, or a missed buffer) differs.
#
# `load_state_dict` is strict by default: a missing or unexpected key raises an error, which is what you want. See the failure mode, and why whole-model pickles are discouraged:

# %%
wrong_arch = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.BatchNorm1d(16), nn.Linear(16, 10))
try:
    wrong_arch.load_state_dict(model.state_dict())
except RuntimeError as e:
    print("architecture mismatch ->", str(e).splitlines()[0][:70], "...")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "whole_model.pt"
    torch.save(model, path)                                   # whole-object pickle
    try:
        torch.load(path)
    except Exception as e:
        print("loading a whole pickled model with the safe default ->", type(e).__name__)

# %% [markdown]
# ## 3. Checkpoints: resume training exactly
#
# To *resume* training you need more than the weights: the **optimizer state** (Adam keeps running averages per parameter) and the **epoch number**. A checkpoint is a dictionary holding all of them.
# Test of a good checkpoint: a run interrupted at epoch 5 and resumed must end **exactly** where an uninterrupted 10-epoch run ends. Because `train_epochs` derives each epoch's shuffle from `(seed, epoch)`, the two runs see identical batches.

# %%
def fresh(seed=0):
    torch.manual_seed(seed)
    m = make_model()
    return m, torch.optim.Adam(m.parameters(), lr=1e-2)


# uninterrupted: 10 epochs
m_full, o_full = fresh()
train_epochs(m_full, o_full, 0, 10)

# interrupted: 5 epochs, checkpoint, "crash", rebuild everything from scratch, load, 5 more
m_a, o_a = fresh()
train_epochs(m_a, o_a, 0, 5)
with tempfile.TemporaryDirectory() as tmp:
    ckpt_path = Path(tmp) / "ckpt.pt"
    torch.save({"model": m_a.state_dict(), "optimizer": o_a.state_dict(), "epoch": 5}, ckpt_path)
    m_b, o_b = fresh(seed=123)                                # different init on purpose: loading must overwrite it
    ckpt = torch.load(ckpt_path)
    m_b.load_state_dict(ckpt["model"])
    o_b.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"]
train_epochs(m_b, o_b, start_epoch, 10)

identical = all(torch.equal(p, q) for p, q in zip(m_full.state_dict().values(), m_b.state_dict().values()))
print("resumed run ends with exactly the same weights as the uninterrupted run:", identical)

# Without the optimizer state, the resumed run drifts away
m_c, o_c = fresh(seed=123)
m_c.load_state_dict({k: v.clone() for k, v in m_a.state_dict().items()})
train_epochs(m_c, o_c, 5, 10)                                # new Adam with empty state
diff = max((p - q).abs().max().item() for p, q in zip(m_full.parameters(), m_c.parameters()))
print(f"forgot the optimizer state: largest weight difference vs the uninterrupted run = {diff:.4f}")

# %% [markdown]
# Keep the checkpoint dictionary small and explicit. Also record what you will need to *rebuild* the model (hyper-parameters such as the hidden size, the feature list, the scaler) and a version or date:
# a `.pt` file of bare weights is useless a year later if nobody remembers the architecture.
#
# ## 4. Reproducibility: seeds and deterministic mode
#
# Randomness enters a PyTorch run in several places: weight initialisation, dropout masks, shuffling in the `DataLoader`, NumPy or Python's `random` in your own code, and (on GPU) non-deterministic kernels.
# One helper to seed the three generators you control:

# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                    # seeds the CPU generator (and CUDA generators too)


def run_once(seed):
    seed_everything(seed)
    m = make_model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    train_epochs(m, opt, 0, 8, data_seed=seed)
    return m


a, b = run_once(7), run_once(7)
c = run_once(8)
same_seed_equal = all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
diff_seed_equal = all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), c.state_dict().values()))
print("same seed twice -> identical weights:", same_seed_equal)
print("different seed  -> identical weights:", diff_seed_equal)

# %% [markdown]
# Seeding gives you a *repeatable* run, but one seed is one sample from a distribution. To know whether a change really helps, look at the **spread across seeds**:

# %%
accs = np.array([accuracy(run_once(s), X_va, y_va) for s in range(8)])
print("validation accuracy over 8 seeds:", np.round(accs, 3))
print(f"mean {accs.mean():.3f}, std {accs.std():.4f}, range {accs.min():.3f} - {accs.max():.3f}")

# %% [markdown]
# With 540 validation images one image is 0.19 percentage points, and in the run above seeds alone moved accuracy by about a point (standard deviation) and by several points from best to worst. A claimed improvement smaller than this spread is noise. (That is why benchmark
# tables report means over seeds.)
#
# **Deterministic mode.** `torch.use_deterministic_algorithms(True)` makes PyTorch pick deterministic implementations of operations, or raise an error if an operation has none. On CPU most operations are already
# deterministic; on GPU some (for example certain scatter/index operations and some convolution algorithms) are not by default, and CUDA may additionally require the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
# The trade-off is speed. Turn it on when debugging or when exact repeatability is a requirement, not by default. Results are also only guaranteed identical on the **same hardware and library versions**.

# %%
torch.use_deterministic_algorithms(True)
print("deterministic algorithms enabled:", torch.are_deterministic_algorithms_enabled())
a, b = run_once(7), run_once(7)
print("still identical:", all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values())))
torch.use_deterministic_algorithms(False)             # restore the default

# %% [markdown]
# For a `DataLoader` with shuffling, pass `generator=torch.Generator().manual_seed(seed)`; if you use `num_workers > 0` with random augmentation, also set a `worker_init_fn`, because each worker process has its own random state.
#
# ## 5. Devices: write the code once
#
# Tensors and models live on a **device** (`cpu`, `cuda:0`, `mps` on Apple silicon). An operation between tensors on different devices is an error. The portable pattern: pick the device once, move the model once, move
# each batch as you use it, and load checkpoints with `map_location`.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

model = make_model().to(device)                      # moves parameters AND buffers
xb = X_va[:8].to(device)
with torch.no_grad():
    model.eval()
    logits = model(xb)
print("model parameter device:", next(model.parameters()).device, "| output device:", logits.device)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "m.pt"
    torch.save(model.state_dict(), path)
    # map_location lets a checkpoint saved on a GPU machine load on a CPU-only one (and vice versa)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
print("loaded with map_location =", device)

# %% [markdown]
# A device mismatch raises a `RuntimeError`. We have no GPU here, so to show the *kind* of error we use PyTorch's `meta` device (tensors with shape but no data) as a stand-in. With a real GPU the message says
# "expected all tensors to be on the same device" (the exact wording varies by PyTorch version):

# %%
try:
    _ = torch.zeros(2, device="meta") + torch.zeros(2)
except RuntimeError as e:
    print("RuntimeError:", str(e)[:80])

# %% [markdown]
# Things that commonly trip people up with devices:
#
# - `model.to(device)` modifies the model **in place** and returns it; `tensor.to(device)` returns a **new** tensor, so you must assign it (`x = x.to(device)`).
# - Create the optimizer **after** moving the model to the device (or at least make sure it references the final parameters).
# - Convert to NumPy only from the CPU: `t.detach().cpu().numpy()`.
# - Sorting out speed needs care: a GPU only helps when the model and batch are big enough to outweigh the copy overhead; tiny models like ours can be *faster* on CPU.
#
# The code above never mentions CUDA explicitly except in the one `device` line, so the same notebook runs unchanged on both. (We cannot run the GPU branch in this environment, so treat the GPU behaviour described here as documented PyTorch behaviour, not something this notebook measured.)
#
# ## 6. Inference mode
#
# For serving, call `model.eval()` (turns off dropout, makes batch-norm use its stored statistics) and wrap predictions in `torch.no_grad()` (or `torch.inference_mode()`, a slightly faster variant) so no computation graph is built.

# %%
model.eval()
with torch.inference_mode():
    pred = model(X_va)
print("requires_grad inside inference_mode:", pred.requires_grad)
pred_grad = model(X_va)
print("requires_grad outside (graph is built, wasting memory):", pred_grad.requires_grad)

# %% [markdown]
# ## Common mistakes
#
# 1. **Saving `model.parameters()` or only weights when the model has buffers** (batch-norm) and then seeing different predictions after loading. Use `state_dict()`.
# 2. **Forgetting `model.eval()` after loading.** Dropout stays active and batch-norm uses batch statistics: predictions are random and differ run to run.
# 3. **Saving the whole model with `torch.save(model)`.** It breaks when you rename a module or move a file, and unpickling arbitrary files can run code. Save the `state_dict`, and only `torch.load` files you trust.
# 4. **Resuming without the optimizer state** (or the epoch and learning-rate scheduler state). The curve jumps; results differ from an uninterrupted run, as measured above.
# 5. **Comparing two models using one seed each.** Report means and spread across seeds.
# 6. **Creating tensors on the wrong device** (`torch.zeros(...)` defaults to CPU). Use `device=x.device` or `.to(device)` for anything created inside `forward`.

# %% [markdown]
# ## Interview corner
#
# **1. What is a `state_dict` and what is the recommended way to save a model?**
# <details><summary>Answer</summary>
# An ordered dictionary of parameter and buffer tensors. Save <code>model.state_dict()</code> with <code>torch.save</code>; to restore, instantiate the architecture in code and call <code>load_state_dict</code>.
# This decouples the weights from the class definition and file layout.
# </details>
#
# **2. How do you resume an interrupted training run properly?**
# <details><summary>Answer</summary>
# Save a checkpoint with the model state, optimizer state, epoch (and scheduler and RNG or data-order state if you need exact repeatability). On restart rebuild model and optimizer, load both states, and continue from the stored epoch.
# </details>
#
# **3. How do you make a PyTorch experiment reproducible, and what are the limits?**
# <details><summary>Answer</summary>
# Seed Python, NumPy and PyTorch; seed DataLoader generators and worker processes; optionally enable <code>torch.use_deterministic_algorithms(True)</code>. Limits: results are guaranteed only on the same hardware, library versions and
# settings, deterministic mode costs speed, and some ops have no deterministic implementation. Even with seeds you should still report variation across seeds.
# </details>
#
# **4. What does `map_location` do?**
# <details><summary>Answer</summary>
# It remaps the device of the tensors inside a saved file while loading, e.g. <code>torch.load(path, map_location="cpu")</code> to open a GPU-trained checkpoint on a machine without a GPU.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell at the bottom of each exercise.

# %% [markdown]
# ### Exercise 1: Count parameters vs buffers
# For `bn_model = nn.Sequential(nn.Linear(10, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 3), nn.BatchNorm1d(3))`, store the number of entries in its `state_dict` in `n_entries`, the number of **trainable parameter
# values** (elements) in `n_trainable`, and the number of **buffer entries** in `n_buffer_entries`. *Hint: `model.state_dict()`, `model.parameters()`, `model.named_buffers()`.*

# %% tags=["starter"]
bn_model = nn.Sequential(nn.Linear(10, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 3), nn.BatchNorm1d(3))
n_entries = ...
n_trainable = ...
n_buffer_entries = ...

# %%
assert n_entries == len(bn_model.state_dict()), "n_entries = number of keys in the state_dict"
assert n_trainable == (10 * 6 + 6) + (6 + 6) + (6 * 3 + 3) + (3 + 3), "n_trainable = elements in weights and biases (Linear) and scale/shift (BatchNorm)"
assert n_buffer_entries == 6, "each BatchNorm has running_mean, running_var and num_batches_tracked -> 2 layers x 3 = 6 buffer entries"
assert n_entries == len(list(bn_model.parameters())) + n_buffer_entries, "state_dict = parameters + buffers"
print("Exercise 1 passed")

# %% tags=["solution"]
bn_model = nn.Sequential(nn.Linear(10, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 3), nn.BatchNorm1d(3))
n_entries = len(bn_model.state_dict())
n_trainable = sum(p.numel() for p in bn_model.parameters())
n_buffer_entries = len(list(bn_model.named_buffers()))

# %% [markdown]
# ### Exercise 2: A safe save/load round trip
# Write `roundtrip(model)` that saves the `state_dict` of `model` into a temporary directory, builds a **new** `make_model()`, loads the file into it, sets it to eval mode and returns it.
# The check trains a model, round-trips it and asserts identical predictions. *Hint: `tempfile.TemporaryDirectory()`; remember `eval()`.*

# %% tags=["starter"]
def roundtrip(model):
    ...

# %%
torch.manual_seed(1)
trained = make_model()
train_epochs(trained, torch.optim.Adam(trained.parameters(), lr=1e-2), 0, 3)
clone = roundtrip(trained)
assert clone is not trained, "return a NEW model object"
assert not clone.training, "the returned model must be in eval mode"
trained.eval()
with torch.no_grad():
    assert torch.equal(trained(X_va), clone(X_va)), "predictions differ after the round trip (buffers? eval mode?)"
print("Exercise 2 passed")

# %% tags=["solution"]
def roundtrip(model):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.pt"
        torch.save(model.state_dict(), path)
        new = make_model()
        new.load_state_dict(torch.load(path))
    new.eval()
    return new

# %% [markdown]
# ### Exercise 3: Write the checkpoint functions
# Implement `save_checkpoint(path, model, opt, epoch)` and `load_checkpoint(path, model, opt)` (returns the stored epoch). The check repeats the section 3 experiment with your functions: train 4 epochs, checkpoint,
# rebuild, load, train to epoch 8, and compare with an uninterrupted 8-epoch run.

# %% tags=["starter"]
def save_checkpoint(path, model, opt, epoch):
    ...


def load_checkpoint(path, model, opt):
    ...

# %%
m_ref, o_ref = fresh()
train_epochs(m_ref, o_ref, 0, 8)
m1, o1 = fresh()
train_epochs(m1, o1, 0, 4)
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "c.pt"
    save_checkpoint(p, m1, o1, 4)
    m2, o2 = fresh(seed=99)
    ep = load_checkpoint(p, m2, o2)
assert ep == 4, f"load_checkpoint should return the stored epoch (4), got {ep}"
train_epochs(m2, o2, ep, 8)
assert all(torch.equal(a_, b_) for a_, b_ in zip(m_ref.state_dict().values(), m2.state_dict().values())), "resumed run does not match the uninterrupted run: did you save the optimizer state?"
print("Exercise 3 passed")

# %% tags=["solution"]
def save_checkpoint(path, model, opt, epoch):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch}, path)


def load_checkpoint(path, model, opt):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]

# %% [markdown]
# ### Exercise 4: How much does the seed matter here?
# Using `run_once(seed)` and `accuracy` (above), compute the validation accuracy for seeds 0-5, and store the **standard deviation** in `seed_std` and the **best-minus-worst gap** in `seed_gap` (as fractions, e.g. 0.012).
# Then write one sentence: if you tuned a setting and saw a 0.3 percentage-point gain on one seed, would you trust it?

# %% tags=["starter"]
seed_std = ...
seed_gap = ...

# %%
ref = np.array([accuracy(run_once(s), X_va, y_va) for s in range(6)])
assert abs(seed_std - ref.std()) < 1e-9, f"seed_std should be the (population) std of the 6 accuracies, expected {ref.std():.5f}"
assert abs(seed_gap - (ref.max() - ref.min())) < 1e-9, f"seed_gap should be max - min, expected {ref.max() - ref.min():.5f}"
print(f"Exercise 4 passed (std {seed_std:.4f}, gap {seed_gap:.4f})")

# %% tags=["solution"]
accs6 = np.array([accuracy(run_once(s), X_va, y_va) for s in range(6)])
seed_std = float(accs6.std())
seed_gap = float(accs6.max() - accs6.min())

# %% [markdown]
# ### Exercise 5 (open-ended): A portable inference script
# Sketch (in a markdown cell, or as code) a small `predict.py`-style function that (1) rebuilds the model, (2) loads a checkpoint with `map_location`, (3) applies the **same scaling** used in training, and (4) returns class
# probabilities in eval mode with no gradients. List the three things you must save *besides* the weights for someone else to run it. No assert: explore and report.

# %% [markdown]
# ## Recap
# - A `state_dict` holds parameters and buffers; save it with `torch.save`, rebuild the architecture in code, then `load_state_dict`. Verify with identical outputs, in `eval()` mode.
# - To resume, checkpoint model + optimizer (+ epoch, scheduler); a correct checkpoint reproduces the uninterrupted run exactly.
# - Seed Python, NumPy and PyTorch (and DataLoader generators); `torch.use_deterministic_algorithms(True)` trades speed for repeatability. Always look at the spread across seeds.
# - Device-agnostic code: one `device` line, `.to(device)` for model and batches, `map_location` when loading, `.cpu()` before NumPy.
# - Save the preprocessing and architecture settings with the weights; a bare weights file is not a reproducible model.
#
# **Go deeper:** the PyTorch docs "Saving and Loading Models" and "Reproducibility" notes (pytorch.org/docs and pytorch.org/tutorials), and the docs for `torch.use_deterministic_algorithms`.
#
# **Next:** [Transfer learning and pretrained models](07_transfer_learning_and_pretrained_models.ipynb)
