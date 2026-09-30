# %% [markdown]
# # Transfer Learning and Pretrained Models
#
# **Why this matters.** Most teams do not have a million labelled images. They have a few hundred. Transfer learning is the standard answer: start from a network already trained on a large dataset, keep the layers that learned
# general features (edges, textures, word pieces), and train only a small new part (or gently adjust everything) on your data. It is how a two-person team ships an image or text classifier in days. It is also easy to do wrongly,
# and it does not always help; this notebook teaches the mechanics and then **measures** when the benefit shows up.
#
# **You will learn**
# - the three recipes: feature extraction (frozen backbone), full fine-tuning, and discriminative learning rates
# - freeze layers correctly (`requires_grad`), verify that frozen weights really do not move, and build optimizer parameter groups
# - run a measured experiment: a small CNN pretrained on digits, reused for a related task with few labels
# - write the standard `torchvision` pretrained-model code (shown, not executed, because it downloads weights) and know its pitfalls
#
# **Prerequisites:** [04 CNN on digits](04_cnn_on_digits.ipynb) and [06 Saving, loading and reproducibility](06_saving_loading_reproducibility_and_devices.ipynb).
#
# > Data used: scikit-learn's bundled `digits` (1,797 real 8x8 images). `torchvision` is not installed in this course environment and the pretrained weights need a download, so the real-world recipe is shown as code you run on your own machine.

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

digits = load_digits()
X = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)       # (1797, 1, 8, 8)
y = torch.tensor(digits.target)
print("images:", tuple(X.shape), "| classes:", sorted(set(y.tolist())))

# %% [markdown]
# ## 1. The idea in one picture
#
# A CNN has two parts:
#
# - a **backbone** (`features`): convolutional layers that turn an image into a vector of learned features. Early layers detect generic patterns (edges, blobs), later ones more task-specific shapes.
# - a **head**: a small final layer that maps those features to *your* classes.
#
# The generic early features transfer across tasks; the head is task-specific and must be replaced. Recipes, cheapest first:
#
# | Recipe | What trains | Use when |
# |---|---|---|
# | **Feature extraction** | only the new head (backbone frozen) | very little data, or a very similar domain |
# | **Fine-tuning** | everything, backbone at a *small* learning rate | moderate data, somewhat different domain |
# | **Gradual unfreezing** | head first, then deeper layers one block at a time | you need to protect the pretrained weights from a noisy start |
#
# The danger of fine-tuning with a big learning rate is **catastrophic forgetting**: the first large gradients from a randomly-initialised head wreck the carefully learned backbone.

# %% [markdown]
# ## 2. A small CNN with a named backbone and head
#
# The naming matters: `model.features` and `model.head` let us treat the two parts separately, exactly like `model.fc` in a torchvision ResNet.

# %%
class DigitNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 8x8 -> 4x4
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 4x4 -> 2x2
            nn.Flatten(),                                                     # 16*2*2 = 64 features
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.features(x))


def count(params):
    return sum(p.numel() for p in params)


net = DigitNet(10)
print("backbone parameters:", count(net.features.parameters()), "| head parameters:", count(net.head.parameters()))

# %% [markdown]
# ## 3. Pretrain, using one half of the data
#
# We split the 1,797 images into two **disjoint** halves: A (for pretraining, all 10 classes, plays the role of "a big source dataset") and B (for the new task; we will use only a few labels from it). The downstream task is a
# *related but different* one: **is the digit even or odd** (2 classes). Keeping A and B separate guarantees that no image used for pretraining leaks into the downstream evaluation.

# %%
def fit(model, params, Xtr, ytr, epochs=30, lr=1e-2, batch_size=32, seed=0):
    """Train with Adam. `params` may be a list of parameters or of parameter groups (each with its own lr)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, Xv, yv):
    model.eval()
    return (model(Xv).argmax(1) == yv).float().mean().item()


idx_A, idx_B = train_test_split(np.arange(len(X)), test_size=0.5, random_state=0, stratify=y.numpy())
print("half A (pretraining):", len(idx_A), "| half B (downstream task):", len(idx_B), "| overlap:", len(set(idx_A) & set(idx_B)))

torch.manual_seed(0)
pretrained = DigitNet(10)
fit(pretrained, pretrained.parameters(), X[idx_A], y[idx_A], epochs=20)
print(f"pretrained 10-class model, accuracy on the unseen half B: {accuracy(pretrained, X[idx_B], y[idx_B]):.1%}")

# %% [markdown]
# The downstream labels: parity (`digit % 2`). The pretrained network never saw that label, only digit identities; if its features capture stroke shapes, they should be useful.
#
# ## 4. Freezing layers, and proving it worked
#
# Freezing means setting `requires_grad = False` on the parameters, so autograd computes no gradient for them and the optimizer leaves them alone. Then pass the optimizer **only the parameters that still train**.

# %%
def freeze(module):
    for p in module.parameters():
        p.requires_grad = False
    return module


def transfer_model(n_classes=2, frozen=True):
    m = DigitNet(n_classes)
    m.features.load_state_dict(pretrained.features.state_dict())        # copy the backbone, NOT the old head
    if frozen:
        freeze(m.features)
    return m


m = transfer_model()
trainable = [p for p in m.parameters() if p.requires_grad]
print("trainable parameters:", count(trainable), "of", count(m.parameters()))

# Proof: train and check that the backbone did not change while the head did
before_backbone = copy.deepcopy(m.features.state_dict())
before_head = copy.deepcopy(m.head.state_dict())
Xb_small, _, yb_small, _ = train_test_split(X[idx_B], (y[idx_B] % 2), train_size=40, random_state=0, stratify=(y[idx_B] % 2))
fit(m, trainable, Xb_small, yb_small, epochs=10)
backbone_same = all(torch.equal(before_backbone[k], v) for k, v in m.features.state_dict().items())
head_same = all(torch.equal(before_head[k], v) for k, v in m.head.state_dict().items())
print("backbone unchanged after training:", backbone_same, "| head unchanged:", head_same)

# %% [markdown]
# *What just happened?* The backbone tensors are bit-for-bit identical after training, while the head moved. Always add a check like this when you freeze something: forgetting to pass the right parameters to the
# optimizer, or freezing after creating it, are silent bugs.
#
# **Batch-norm gotcha.** `requires_grad=False` does not stop a `BatchNorm` layer from updating its running mean and variance while the model is in `train()` mode, because those are buffers, not trainable parameters.
# A "frozen" backbone with batch-norm keeps drifting unless you also put those layers in eval mode. Small demonstration:

# %%
bn = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
freeze(bn)
stats_before = bn[1].running_mean.clone()
bn.train()
bn(torch.randn(32, 4) + 5.0)                               # data with a different mean than the BN statistics
print("running_mean changed although frozen:", not torch.equal(stats_before, bn[1].running_mean))
bn.eval()
stats_mid = bn[1].running_mean.clone()
bn(torch.randn(32, 4) + 5.0)
print("running_mean changed in eval mode    :", not torch.equal(stats_mid, bn[1].running_mean))

# %% [markdown]
# Our digit network has no batch-norm, so it is unaffected; for a ResNet backbone, the usual fix is to call `backbone.eval()` after `model.train()` each epoch (or freeze the BN layers explicitly).
#
# ## 5. A measured experiment: scratch vs frozen vs fine-tuned
#
# We compare three models on the parity task with `n` labelled images from half B (evaluating on the rest of B), for several `n`, and 5 random subsets for each:
#
# - **scratch**: a new `DigitNet(2)`, all weights random.
# - **frozen**: pretrained backbone frozen, only the head trained.
# - **fine-tune**: pretrained backbone at learning rate `1e-3`, new head at `1e-2` (two **parameter groups**).

# %%
def run_parity(n, seed):
    parity = y[idx_B] % 2
    Xa, Xe, ya, ye = train_test_split(X[idx_B], parity, train_size=n, random_state=seed, stratify=parity)
    out = {}
    torch.manual_seed(seed)
    scratch = DigitNet(2)
    fit(scratch, scratch.parameters(), Xa, ya, epochs=60, lr=3e-3, seed=seed)
    out["scratch"] = accuracy(scratch, Xe, ye)

    fz = transfer_model(frozen=True)
    fit(fz, [p for p in fz.parameters() if p.requires_grad], Xa, ya, epochs=60, lr=1e-2, seed=seed)
    out["frozen"] = accuracy(fz, Xe, ye)

    ft = transfer_model(frozen=False)
    groups = [{"params": ft.features.parameters(), "lr": 1e-3}, {"params": ft.head.parameters(), "lr": 1e-2}]
    fit(ft, groups, Xa, ya, epochs=60, seed=seed)
    out["fine-tune"] = accuracy(ft, Xe, ye)
    return out


sizes = [10, 40, 100]
results = {n: [run_parity(n, s) for s in range(5)] for n in sizes}
print(f"{'labelled images':>16s} {'scratch':>9s} {'frozen':>9s} {'fine-tune':>10s}     (mean test accuracy over 5 subsets)")
for n in sizes:
    row = {k: np.mean([r[k] for r in results[n]]) for k in ("scratch", "frozen", "fine-tune")}
    print(f"{n:16d} {row['scratch']:9.3f} {row['frozen']:9.3f} {row['fine-tune']:10.3f}")

fig, ax = plt.subplots(figsize=(6, 3.5))
for k in ("scratch", "frozen", "fine-tune"):
    ax.plot(sizes, [np.mean([r[k] for r in results[n]]) for n in sizes], marker="o", label=k)
ax.set_xlabel("labelled training images for the new task")
ax.set_ylabel("test accuracy (mean of 5 subsets)")
ax.set_title("Even/odd digits: does pretraining help?")
ax.legend()
plt.show()

# %% [markdown]
# Read this carefully and honestly. The gap between pretrained and scratch models is **small on this toy problem**: at 10 and 40 labels it is within the noise across subsets (see the spread below), and only at 100 labels
# does pretraining help consistently, by a few points. Plausible reasons: our pretraining set is tiny, and the network is so small that learning from scratch is cheap. Real transfer learning shines when the source dataset is
# **orders of magnitude** larger than your data (ImageNet versus your few hundred photos) and the backbone is big. Check the spread across subsets before believing any gap:

# %%
for n in sizes:
    gaps = np.array([r["fine-tune"] - r["scratch"] for r in results[n]])
    print(f"n={n:3d}: fine-tune minus scratch = {gaps.mean():+.3f} (std over subsets {gaps.std():.3f})")

# %% [markdown]
# ## 6. When transfer does *not* help: a different target task
#
# Parity is a task whose inputs (and underlying digit shapes) the backbone has seen. A less related target is harder. As a contrast, take a **held-out-classes** setup: pretrain on digits 0-4 only,
# then learn digits 5-9 (five classes the backbone never saw) from 40 labelled images.

# %%
src_mask, tgt_mask = (y < 5), (y >= 5)
torch.manual_seed(0)
base_04 = DigitNet(5)
fit(base_04, base_04.parameters(), X[src_mask], y[src_mask], epochs=20)

scores = {"scratch": [], "frozen": [], "fine-tune": []}
Xt_, yt_ = X[tgt_mask], y[tgt_mask] - 5
for seed in range(5):
    Xa, Xe, ya, ye = train_test_split(Xt_, yt_, train_size=40, random_state=seed, stratify=yt_)
    torch.manual_seed(seed)
    m_s = DigitNet(5)
    fit(m_s, m_s.parameters(), Xa, ya, epochs=60, lr=3e-3, seed=seed)
    scores["scratch"].append(accuracy(m_s, Xe, ye))
    m_f = DigitNet(5)
    m_f.features.load_state_dict(base_04.features.state_dict())
    freeze(m_f.features)
    fit(m_f, list(m_f.head.parameters()), Xa, ya, epochs=60, lr=1e-2, seed=seed)
    scores["frozen"].append(accuracy(m_f, Xe, ye))
    m_t = DigitNet(5)
    m_t.features.load_state_dict(base_04.features.state_dict())
    fit(m_t, [{"params": m_t.features.parameters(), "lr": 1e-3}, {"params": m_t.head.parameters(), "lr": 1e-2}], Xa, ya, epochs=60, seed=seed)
    scores["fine-tune"].append(accuracy(m_t, Xe, ye))
for k, v in scores.items():
    print(f"{k:10s} accuracy on digits 5-9 with 40 labels: {np.mean(v):.3f} +/- {np.std(v):.3f}")

# %% [markdown]
# *What just happened?* Features learned from digits 0-4 are **no better** than training from scratch on digits 5-9 here (the frozen version is a bit lower, and the differences are within about one standard deviation). A plausible reason is that the tiny backbone
# learned features specific to those five shapes while the from-scratch network is perfectly capable on this small task. The lesson is not "transfer learning is bad" but **"measure it"**: the benefit depends on the relation between tasks, the size of the source data and the
# size of your data. A from-scratch baseline is part of every transfer-learning experiment.

# %% [markdown]
# ## 7. The real thing: pretrained torchvision models (not executed here)
#
# The cells below are tagged `no-run`: `torchvision` is not installed in this environment and the pretrained weights are **downloaded from the internet on first use**. They show the standard, current API
# (the `weights=` argument and the matching preprocessing transform); run them on your own machine after `pip install torchvision`.

# %% tags=["no-run"]
from torchvision import models

weights = models.ResNet18_Weights.DEFAULT           # pretrained on ImageNet-1k; downloads on first use
backbone = models.resnet18(weights=weights)
preprocess = weights.transforms()                   # the resize / crop / normalisation the weights were trained with: USE IT on your images
print(preprocess)

# %% [markdown]
# Step 1, **feature extraction**: freeze everything, replace the final layer (`fc`) with one for your number of classes. The new layer has `requires_grad=True` by default.

# %% tags=["no-run"]
n_classes = 5                                        # your task
for p in backbone.parameters():
    p.requires_grad = False
backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

optimizer = torch.optim.Adam(backbone.fc.parameters(), lr=1e-3)   # only the new head
# train loop: model.train() would update BatchNorm statistics of the frozen layers, so keep the backbone in eval mode:
#   backbone.train(); [m.eval() for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]

# %% [markdown]
# Step 2, **fine-tuning** with discriminative learning rates, after the head has converged: unfreeze the last block (or everything) and give the backbone a much smaller rate than the head.

# %% tags=["no-run"]
for p in backbone.layer4.parameters():               # unfreeze the last residual stage
    p.requires_grad = True

optimizer = torch.optim.Adam([
    {"params": backbone.layer4.parameters(), "lr": 1e-5},
    {"params": backbone.fc.parameters(), "lr": 1e-3},
])

# %% [markdown]
# **Pitfalls with real pretrained models**
#
# - **Use the matching preprocessing** (`weights.transforms()`): the wrong image size or normalisation silently degrades accuracy.
# - **Grayscale or other channel counts**: pretrained vision backbones expect 3 channels; repeat the channel or adapt the first convolution.
# - **Domain gap**: ImageNet photos transfer well to other photos, less well to X-rays, satellite tiles or spectrograms. Fine-tuning more layers then helps more.
# - **Licences and provenance**: check the licence of pretrained weights before shipping a product, and only load checkpoints from sources you trust (unpickling can execute code).
# - **Same recipe for text**: a pretrained language model with a new classification head is transfer learning too; the NLP and LLM modules revisit it.
#
# ## Common mistakes
#
# 1. **Replacing the head but leaving old parameters in the optimizer** (or creating the optimizer before freezing). Build the optimizer from the parameters that should train, and verify frozen weights did not move, as we did.
# 2. **A single large learning rate for a fine-tuned backbone**, which destroys pretrained features. Use 10-100x smaller for the backbone.
# 3. **Frozen BatchNorm still updating** its statistics in `train()` mode.
# 4. **Wrong preprocessing** (no normalisation, different size) relative to pretraining.
# 5. **No from-scratch baseline**, so you cannot tell whether transfer helped.
# 6. **Leakage between pretraining data and your test set** (for example, a test image that is in the source dataset); keep them disjoint, as in section 3.

# %%
# A cheap guard against mistake 1: count the parameters the optimizer will actually update.
m = transfer_model(frozen=True)
opt_ok = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
n_opt = sum(p.numel() for g in opt_ok.param_groups for p in g["params"])
print("parameters the optimizer will update:", n_opt, "(should equal the head size:", count(m.head.parameters()), ")")

# %% [markdown]
# ## Interview corner
#
# **1. What is transfer learning and when does it help most?**
# <details><summary>Answer</summary>
# Reusing a model trained on a large source task as the starting point for a related target task. It helps most when labelled target data is scarce, the source dataset is large and the domains are related. It can give little or no
# gain when the tasks are unrelated or when the target dataset is large enough to train from scratch.
# </details>
#
# **2. Feature extraction vs fine-tuning?**
# <details><summary>Answer</summary>
# Feature extraction freezes the backbone and trains only a new head (fast, low risk of overfitting, good for very small data). Fine-tuning also updates backbone weights, usually with a smaller learning rate, which adapts features
# to the new domain and helps with more data or a larger domain gap.
# </details>
#
# **3. Why use a smaller learning rate for pretrained layers?**
# <details><summary>Answer</summary>
# The new head starts random and produces large, misleading gradients; a big step would wreck the pretrained features (catastrophic forgetting). Small backbone rates keep the weights near their useful values while the head catches up.
# </details>
#
# **4. What goes wrong with BatchNorm during transfer learning?**
# <details><summary>Answer</summary>
# Running statistics update in train mode even when parameters are frozen, and with small batches they become noisy. Put frozen BN layers in eval mode, or fine-tune with batches large enough for stable statistics.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell at the bottom of each exercise.

# %% [markdown]
# ### Exercise 1: Count what trains
# For `DigitNet(7)` with the **backbone frozen**, store the number of trainable parameters in `n_trainable` and the number of frozen ones in `n_frozen`, computed from the layer sizes (conv 1->8 and 8->16 with 3x3 kernels; head `Linear(64, 7)`).
# The check builds the model and counts `requires_grad`. *Hint: parameters = `out*in*k*k + out` for conv, `in*out + out` for linear.*

# %% tags=["starter"]
n_trainable = ...
n_frozen = ...

# %%
chk = DigitNet(7)
freeze(chk.features)
assert n_trainable == count(p for p in chk.parameters() if p.requires_grad), f"trainable should be the head only: {count(p for p in chk.parameters() if p.requires_grad)}"
assert n_frozen == count(p for p in chk.parameters() if not p.requires_grad), f"frozen should be the backbone: {count(p for p in chk.parameters() if not p.requires_grad)}"
print(f"Exercise 1 passed (trainable {n_trainable}, frozen {n_frozen})")

# %% tags=["solution"]
n_trainable = 64 * 7 + 7
n_frozen = (8 * 1 * 3 * 3 + 8) + (16 * 8 * 3 * 3 + 16)

# %% [markdown]
# ### Exercise 2: Unfreeze the last conv layer
# Write `unfreeze_last_conv(model)` that keeps the first conv layer frozen but makes the **second** conv layer (`model.features[3]`) trainable again, and returns the model. Start from a fully frozen backbone.

# %% tags=["starter"]
def unfreeze_last_conv(model):
    ...

# %%
mm = DigitNet(2)
freeze(mm.features)
out = unfreeze_last_conv(mm)
assert out is mm, "return the same model object"
assert all(not p.requires_grad for p in mm.features[0].parameters()), "the first conv must stay frozen"
assert all(p.requires_grad for p in mm.features[3].parameters()), "the second conv (features[3]) must be trainable"
assert all(p.requires_grad for p in mm.head.parameters()), "the head should be trainable"
assert count(p for p in mm.parameters() if p.requires_grad) == (16 * 8 * 9 + 16) + (64 * 2 + 2)
print("Exercise 2 passed")

# %% tags=["solution"]
def unfreeze_last_conv(model):
    for p in model.features[3].parameters():
        p.requires_grad = True
    return model

# %% [markdown]
# ### Exercise 3: Discriminative learning rates
# Write `make_optimizer(model, lr_backbone, lr_head)` returning an `Adam` optimizer with **two parameter groups**: the backbone (`model.features`) at `lr_backbone` and the head at `lr_head`. The check verifies the groups, the rates and that every
# parameter appears exactly once.

# %% tags=["starter"]
def make_optimizer(model, lr_backbone, lr_head):
    ...

# %%
mm = DigitNet(3)
opt = make_optimizer(mm, 1e-4, 1e-2)
assert isinstance(opt, torch.optim.Adam), "use torch.optim.Adam"
assert len(opt.param_groups) == 2, "expected two parameter groups"
assert [g["lr"] for g in opt.param_groups] == [1e-4, 1e-2], "group 0 = backbone lr, group 1 = head lr"
ids = [id(p) for g in opt.param_groups for p in g["params"]]
assert len(ids) == len(set(ids)) == len(list(mm.parameters())), "every parameter must be in exactly one group"
assert {id(p) for p in opt.param_groups[1]["params"]} == {id(p) for p in mm.head.parameters()}, "group 1 must be the head"
print("Exercise 3 passed")

# %% tags=["solution"]
def make_optimizer(model, lr_backbone, lr_head):
    return torch.optim.Adam([
        {"params": model.features.parameters(), "lr": lr_backbone},
        {"params": model.head.parameters(), "lr": lr_head},
    ])

# %% [markdown]
# ### Exercise 4: Does fine-tuning beat scratch at 40 labels?
# Using `run_parity(40, seed)` for seeds 0, 1 and 2, compute `gain_mean` (the mean of `fine-tune` minus `scratch` accuracy across the three seeds) and `gain_positive` (how many of the three seeds had a positive gain).
# Decide in words whether three seeds are enough to conclude anything.

# %% tags=["starter"]
gain_mean = ...
gain_positive = ...

# %%
ref_runs = [run_parity(40, s) for s in range(3)]
ref_gains = [r["fine-tune"] - r["scratch"] for r in ref_runs]
assert abs(gain_mean - float(np.mean(ref_gains))) < 1e-9, f"expected mean gain {np.mean(ref_gains):+.4f}, got {gain_mean:+.4f}"
assert gain_positive == sum(g > 0 for g in ref_gains), f"expected {sum(g > 0 for g in ref_gains)} positive gains, got {gain_positive}"
print(f"Exercise 4 passed (mean gain {gain_mean:+.3f}, positive in {gain_positive}/3 seeds)")

# %% tags=["solution"]
runs3 = [run_parity(40, s) for s in range(3)]
gains3 = [r["fine-tune"] - r["scratch"] for r in runs3]
gain_mean = float(np.mean(gains3))
gain_positive = int(sum(g > 0 for g in gains3))

# %% [markdown]
# ### Exercise 5 (open-ended): Find the crossover
# Re-run the sizes experiment with `sizes = [5, 20, 200]` (or other sizes) and look at where pretraining stops helping: at 200 labels, is scratch already as good? Also try unfreezing only the last conv layer
# (Exercise 2) and compare it with frozen and fully fine-tuned. Write two sentences on when you would bother with transfer learning for a real project. No assert: explore and report.

# %% [markdown]
# ## Recap
# - Backbone = general features, head = task-specific layer. Replace the head; freeze or gently fine-tune the backbone.
# - Freeze with `requires_grad=False`, give the optimizer only trainable parameters, and **verify** frozen tensors did not change. Frozen BatchNorm layers still update statistics in `train()` mode.
# - Fine-tuning uses a smaller learning rate for pretrained layers (parameter groups) to avoid catastrophic forgetting.
# - Transfer is not magic: on this toy it helped modestly at 100 labels (not distinguishable from noise at 10-40) and not at all for unseen classes. Always compare with a from-scratch baseline over several subsets.
# - With torchvision: `weights=...DEFAULT`, use `weights.transforms()`, replace `fc`, mind BatchNorm, licences and domain gap (code shown, run it locally).
#
# **Go deeper:** the PyTorch tutorial "Transfer Learning for Computer Vision Tutorial" and the torchvision documentation on models and pretrained weights (pytorch.org/tutorials, pytorch.org/vision);
# Stanford CS231n notes "Transfer Learning" (cs231n.github.io/transfer-learning).
#
# **Back to the module:** [PyTorch README](README.md). **Next module:** [15 TensorFlow](../15_TensorFlow/README.md)
