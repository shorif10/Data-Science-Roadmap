# %% [markdown]
# # A Convolutional Network on Digits in PyTorch
#
# **Why this matters.** Images, spectrograms, sensor grids and scanned documents all share one property: nearby values
# belong together. A plain fully-connected network ignores that and has to re-learn "a stroke here" for every position.
# A convolutional network (CNN) bakes the locality in, so it needs fewer parameters and is the default starting point
# for image work. In a job you will mostly *fine-tune* existing CNNs, but you can only debug them if you can read the
# shapes and parameter counts of a small one you built yourself.
#
# **You will learn**
# - build a CNN with `nn.Conv2d`, `nn.MaxPool2d` and `nn.Linear`, and predict every tensor shape by hand
# - count parameters and compare a CNN with an MLP honestly (same data, same training recipe)
# - use a proper train / validation / test split, and report a confusion matrix
# - inspect misclassified images, and test a claimed benefit of convolutions (shift robustness) instead of assuming it
#
# **Prerequisites:** [Convolutions and CNN intuition](../13_Deep_Learning/04_convolutions_and_cnn_intuition.ipynb),
# [Classification with Dataset and DataLoader](03_classification_with_dataset_and_dataloader.ipynb).
#
# > Data used: scikit-learn's bundled `digits` (1,797 real 8x8 grayscale handwritten digits). Tiny images keep everything
# > fast on a CPU; the ideas are identical for 224x224 photos.

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
torch.set_num_threads(2)
np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. Data: images keep their 2-D shape
#
# A PyTorch 2-D convolution expects **`(batch, channels, height, width)`** (called NCHW). Greyscale means 1 channel.
# We split 60 / 20 / 20 into train / validation / test. The validation set is for choices (epochs, architecture);
# the test set is opened **once**, at the end.

# %%
digits = load_digits()
X_img = (digits.images / 16.0).astype(np.float32)   # (1797, 8, 8), pixel values 0..16 scaled to 0..1
y_all = digits.target

X_trval, X_te, y_trval, y_te = train_test_split(X_img, y_all, test_size=0.2, stratify=y_all, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=42)


def to_nchw(a):
    return torch.from_numpy(a).unsqueeze(1)          # add the channel axis -> (N, 1, 8, 8)


Xtr_t, Xva_t, Xte_t = to_nchw(X_tr), to_nchw(X_va), to_nchw(X_te)
ytr_t, yva_t, yte_t = (torch.from_numpy(v).long() for v in (y_tr, y_va, y_te))
print("train", tuple(Xtr_t.shape), " val", tuple(Xva_t.shape), " test", tuple(Xte_t.shape))

# %%
fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, img, lab in zip(axes.ravel(), X_tr[:16], y_tr[:16]):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(f"label {lab}", fontsize=9)
    ax.axis("off")
fig.suptitle("Training images (8x8 pixels, darker = more ink)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. What a convolution layer does to shapes
#
# `nn.Conv2d(in_channels, out_channels, kernel_size)` slides `out_channels` small filters over the image. Each filter is a
# `in_channels x k x k` block of weights shared across *every position*, which is the whole trick: a stroke detector
# learned once works everywhere.
#
# Output side length: `floor((n + 2*padding - kernel) / stride) + 1`. Parameters: `out * in * k * k + out` (the `+ out` are biases).

# %%
conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
batch = Xtr_t[:5]
out = conv(batch)
print("input :", tuple(batch.shape))
print("output:", tuple(out.shape), " (padding=1 with a 3x3 kernel keeps 8x8)")
print("weight:", tuple(conv.weight.shape), " bias:", tuple(conv.bias.shape))
print("params:", sum(p.numel() for p in conv.parameters()), "= 8*1*3*3 + 8")

# %% [markdown]
# *What just happened?* 8 filters turned one 8x8 image into 8 "feature maps" of size 8x8. `MaxPool2d(2)` then keeps the
# largest value in each 2x2 block, halving height and width. That gives some tolerance to small shifts and cuts computation.

# %%
pool = nn.MaxPool2d(2)
print("after pool:", tuple(pool(torch.relu(out)).shape))

# %% [markdown]
# ## 3. A small CNN
#
# `Conv -> ReLU -> Pool` twice, then flatten and a linear layer produces 10 **logits** (raw scores; `nn.CrossEntropyLoss` applies the softmax itself).
# Shapes for one image: `1x8x8 -> 8x8x8 -> 8x4x4 -> 16x4x4 -> 16x2x2 -> 64 -> 10`.

# %%
class SmallCNN(nn.Module):
    def __init__(self, c1=8, c2=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(c2 * 2 * 2, 10)

    def forward(self, x):
        x = self.features(x)             # (N, c2, 2, 2)
        return self.head(x.flatten(1))   # flatten everything except the batch axis


class MLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


cnn, mlp = SmallCNN(), MLP()
print("CNN logits shape:", tuple(cnn(Xtr_t[:5]).shape))
print(f"parameters: CNN {n_params(cnn)}, MLP(32 hidden) {n_params(mlp)}")

# %% [markdown]
# Both are tiny. On these 8x8 images the parameter counts are comparable, so do not expect a dramatic saving here. The
# saving shows up on large images: an MLP's first layer grows with `height x width x hidden`, while a conv layer's size
# does not depend on the image size at all.

# %%
big_mlp_first_layer = 224 * 224 * 3 * 100 + 100          # dense layer on a 224x224 RGB image with 100 hidden units
big_conv_layer = 100 * 3 * 3 * 3 + 100                   # 100 3x3 filters on the same image
print(f"dense first layer: {big_mlp_first_layer:,} params   conv layer: {big_conv_layer:,} params")

# %% [markdown]
# ## 4. Training and evaluating, done carefully
#
# One function trains either model with the same recipe (Adam, lr 0.01, batch 32, 30 epochs) and records validation accuracy each epoch.
# Two habits matter: `model.train()` / `model.eval()` (irrelevant here because we have no dropout or batch-norm, but essential
# once you add them) and `torch.no_grad()` for evaluation.

# %%
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def fit(model, epochs=30, lr=0.01, batch_size=32, seed=42):
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        hist["train_loss"].append(evaluate(model, Xtr_t, ytr_t)[0])
        vl, va = evaluate(model, Xva_t, yva_t)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
    return hist


torch.manual_seed(42)
cnn, mlp = SmallCNN(), MLP()
h_cnn, h_mlp = fit(cnn), fit(mlp)
print(f"final validation accuracy: CNN {h_cnn['val_acc'][-1]:.3f}   MLP {h_mlp['val_acc'][-1]:.3f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for name, h in [("CNN", h_cnn), ("MLP", h_mlp)]:
    ax[0].plot(h["val_loss"], label=f"{name} validation")
    ax[0].plot(h["train_loss"], "--", label=f"{name} train")
    ax[1].plot(h["val_acc"], label=name)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("cross-entropy loss")
ax[0].set_title("Loss per epoch")
ax[0].legend(fontsize=8)
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("validation accuracy")
ax[1].set_title("Validation accuracy per epoch")
ax[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* Both models learn digits well. With only ~360 validation images, one image is about 0.28 percentage points, so a
# gap of a point or two between the models is within what a different seed could change. To claim "CNN beats MLP" you would
# repeat with several seeds; we do that next and report what was measured rather than declare a winner.

# %%
seeds = [0, 1, 2]
rows = []
for s in seeds:
    torch.manual_seed(s)
    rows.append((fit(SmallCNN(), seed=s)["val_acc"][-1], fit(MLP(), seed=s)["val_acc"][-1]))
rows = np.array(rows)
print("val accuracy over seeds 0,1,2  ->  CNN", rows[:, 0].round(3), " MLP", rows[:, 1].round(3))

# %% [markdown]
# ## 5. Test set, confusion matrix, and the mistakes
#
# Now, **once**, the test set. A confusion matrix shows which digits get mistaken for which: rows are the true digit, columns the prediction.

# %%
test_loss, test_acc = evaluate(cnn, Xte_t, yte_t)
print(f"CNN test accuracy: {test_acc:.3f}   (MLP: {evaluate(mlp, Xte_t, yte_t)[1]:.3f})")
with torch.no_grad():
    pred_te = cnn.eval()(Xte_t).argmax(dim=1).numpy()
cm = confusion_matrix(y_te, pred_te)

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel("predicted digit")
ax.set_ylabel("true digit")
ax.set_title("CNN confusion matrix on the test set")
for i in range(10):
    for j in range(10):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, label="count")
plt.tight_layout()
plt.show()

# %%
wrong = np.flatnonzero(pred_te != y_te)
print(f"{len(wrong)} of {len(y_te)} test images misclassified")
if len(wrong):
    n_show = min(8, len(wrong))
    fig, axes = plt.subplots(1, n_show, figsize=(1.6 * n_show, 2.2), squeeze=False)
    for ax, k in zip(axes[0], wrong[:n_show]):
        ax.imshow(X_te[k], cmap="gray_r")
        ax.set_title(f"true {y_te[k]}\npred {pred_te[k]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Misclassified test images")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Reading the mistakes is an underrated habit: if the errors look like digits *you* would hesitate on at 8x8 resolution,
# the model is near the limit of the data. If they look easy, suspect a bug or a data problem.

# %% [markdown]
# ## 6. Testing a claim: are CNNs robust to shifts?
#
# A textbook claim says weight sharing plus pooling makes CNNs tolerant to small translations and MLPs not. We can test it.
# Shift every test image one pixel to the right with `torch.roll` (the image is mostly blank at the edges, so little ink wraps around)
# and re-measure. Neither model saw shifted images during training.

# %%
shifted = torch.roll(Xte_t, shifts=1, dims=3)     # dims=3 is the width axis in NCHW
acc = {name: (evaluate(m, Xte_t, yte_t)[1], evaluate(m, shifted, yte_t)[1]) for name, m in [("CNN", cnn), ("MLP", mlp)]}
for name, (a0, a1) in acc.items():
    print(f"{name}: original {a0:.3f} -> shifted by 1 pixel {a1:.3f}   (drop {a0 - a1:+.3f})")

# %% [markdown]
# Whatever the numbers say is the answer for *this* pair of models. In general convolution gives translation *equivariance*
# (shift the input, the feature maps shift) and pooling adds a little *invariance*, but robustness to large shifts still
# comes mostly from training data augmentation, not from the architecture alone.

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting the channel axis.** `Conv2d` wants `(N, C, H, W)`; passing `(N, H, W)` is misread or raises a shape error. Use `unsqueeze(1)`.
# 2. **Wrong `in_features` on the first `Linear` after the conv stack.** It must equal `channels * H * W` after the last pool. Compute it with a dummy forward pass when unsure.
# 3. **Putting `softmax` before `CrossEntropyLoss`.** The loss applies log-softmax itself; doing it twice squashes gradients and hurts training.
# 4. **Evaluating in `train()` mode or with gradients on.** Dropout and batch-norm behave differently, and you waste memory. Use `model.eval()` and `torch.no_grad()`.
# 5. **Choosing by test accuracy.** If you tune on the test set it stops being an honest estimate. Tune on validation, look at test once.

# %%
# Demonstration of mistake 1 and the dummy-forward trick from mistake 2
try:
    cnn(Xte_t[:4].squeeze(1))            # (4, 8, 8): no channel axis
except (RuntimeError, ValueError) as e:
    print(type(e).__name__, ":", str(e)[:100])

flat_features = cnn.features(torch.zeros(1, 1, 8, 8)).numel()
print("flattened feature size found by a dummy pass:", flat_features, "== head.in_features:", cnn.head.in_features)

# %% [markdown]
# ## Interview corner
#
# **1. Why do CNNs use fewer parameters than fully-connected networks on images?**
# <details><summary>Answer</summary>
# Two ideas: <em>local connectivity</em> (each output looks at a small neighbourhood, not the whole image) and <em>weight sharing</em> (the same filter is reused at every position).
# A conv layer's parameter count depends on kernel size and channels, not on image size.
# </details>
#
# **2. What does a 3x3 convolution with padding 1 and stride 1 do to the spatial size? What about padding 0?**
# <details><summary>Answer</summary>
# Padding 1 keeps it the same (<code>(n + 2 - 3)/1 + 1 = n</code>); padding 0 shrinks each side by 2. A 2x2 max-pool with stride 2 halves height and width.
# </details>
#
# **3. What is the difference between translation equivariance and invariance?**
# <details><summary>Answer</summary>
# Equivariance: shifting the input shifts the feature map the same way (convolution). Invariance: the output does not change when the input shifts (approximately achieved by pooling and global pooling near the end).
# </details>
#
# **4. Your CNN gets 99% train but much lower validation accuracy. What do you try first?**
# <details><summary>Answer</summary>
# That is overfitting. Options in rough order of cost: more data or augmentation, weight decay, dropout, a smaller model, early stopping on validation loss.
# Also confirm the split has no leakage (for example near-duplicate images on both sides).
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Predict the conv output size
# Write `conv_out(n, kernel, stride=1, padding=0)` returning the output side length of a square conv (or pooling) layer.
# Then use it to store in `final_side` the side length after this stack applied to an 8x8 image:
# `Conv2d(k=3, padding=0)` then `MaxPool2d(2)` (stride 2, no padding).
# *Hint: `(n + 2*padding - kernel) // stride + 1`.*

# %% tags=["starter"]
def conv_out(n, kernel, stride=1, padding=0):
    ...


final_side = ...

# %%
for n, k, s, p in [(8, 3, 1, 1), (8, 3, 1, 0), (28, 5, 2, 2), (32, 2, 2, 0)]:
    real = nn.Conv2d(1, 1, k, stride=s, padding=p)(torch.zeros(1, 1, n, n)).shape[-1]
    assert conv_out(n, k, s, p) == real, f"conv_out({n}, {k}, {s}, {p}) gave {conv_out(n, k, s, p)}, PyTorch says {real}"
stack = nn.Sequential(nn.Conv2d(1, 1, 3, padding=0), nn.MaxPool2d(2))
assert final_side == stack(torch.zeros(1, 1, 8, 8)).shape[-1], "run the stack on a dummy tensor to see the true size"
print("Exercise 1 passed (final side:", final_side, ")")

# %% tags=["solution"]
def conv_out(n, kernel, stride=1, padding=0):
    return (n + 2 * padding - kernel) // stride + 1


final_side = conv_out(conv_out(8, 3), 2, stride=2)

# %% [markdown]
# ### Exercise 2: Count parameters by formula
# Write `conv_params(c_in, c_out, k)` returning the number of parameters (weights + biases) of one `Conv2d`.
# Store in `cnn_total` the total for `SmallCNN()` computed **by formula** (two conv layers plus the `Linear(64, 10)`), not with `n_params`.

# %% tags=["starter"]
def conv_params(c_in, c_out, k):
    ...


cnn_total = ...

# %%
assert conv_params(1, 8, 3) == sum(p.numel() for p in nn.Conv2d(1, 8, 3).parameters())
assert conv_params(3, 16, 5) == sum(p.numel() for p in nn.Conv2d(3, 16, 5).parameters())
assert cnn_total == n_params(SmallCNN()), f"formula gave {cnn_total}, PyTorch counts {n_params(SmallCNN())}"
print("Exercise 2 passed (SmallCNN has", cnn_total, "parameters)")

# %% tags=["solution"]
def conv_params(c_in, c_out, k):
    return c_out * c_in * k * k + c_out


cnn_total = conv_params(1, 8, 3) + conv_params(8, 16, 3) + (64 * 10 + 10)

# %% [markdown]
# ### Exercise 3: Accuracy per digit
# Overall accuracy hides weak classes. Using the trained `cnn` and the test set (`Xte_t`, `yte_t`), create `per_class_acc`:
# a NumPy array of length 10 with the accuracy for each true digit. Store the digit with the lowest accuracy in `worst_digit`
# (ties: the smallest digit). *Hint: `confusion_matrix` diagonal divided by row sums, or a boolean mask per class.*

# %% tags=["starter"]
per_class_acc = ...
worst_digit = ...

# %%
with torch.no_grad():
    _pred = cnn.eval()(Xte_t).argmax(dim=1)
ref = np.array([(_pred[yte_t == d] == d).float().mean().item() for d in range(10)])
assert np.shape(per_class_acc) == (10,), "one accuracy per digit 0..9"
assert np.allclose(per_class_acc, ref), f"expected {ref.round(3)}, got {np.round(per_class_acc, 3)}"
assert worst_digit == int(np.argmin(ref))
print(f"Exercise 3 passed (weakest digit: {worst_digit}, accuracy {per_class_acc[worst_digit]:.3f})")

# %% tags=["solution"]
cm_ex = confusion_matrix(y_te, pred_te, labels=range(10))
per_class_acc = np.diag(cm_ex) / cm_ex.sum(axis=1)
worst_digit = int(np.argmin(per_class_acc))

# %% [markdown]
# ### Exercise 4: Shift the other way
# Section 6 shifted images one pixel to the right. Measure the test accuracy of `cnn` after shifting **two pixels down**
# (the height axis, `dims=2`) and store it in `cnn_down2`. Store the matching number for `mlp` in `mlp_down2`.

# %% tags=["starter"]
cnn_down2 = ...
mlp_down2 = ...

# %%
_sh = torch.roll(Xte_t, shifts=2, dims=2)
with torch.no_grad():
    exp_c = (cnn.eval()(_sh).argmax(1) == yte_t).float().mean().item()
    exp_m = (mlp.eval()(_sh).argmax(1) == yte_t).float().mean().item()
assert abs(cnn_down2 - exp_c) < 1e-6, f"CNN: expected {exp_c:.4f}, got {cnn_down2}"
assert abs(mlp_down2 - exp_m) < 1e-6, f"MLP: expected {exp_m:.4f}, got {mlp_down2}"
print(f"Exercise 4 passed (CNN {cnn_down2:.3f}, MLP {mlp_down2:.3f})")

# %% tags=["solution"]
shifted2 = torch.roll(Xte_t, shifts=2, dims=2)
cnn_down2 = evaluate(cnn, shifted2, yte_t)[1]
mlp_down2 = evaluate(mlp, shifted2, yte_t)[1]

# %% [markdown]
# ### Exercise 5 (open-ended): Make the CNN wider or narrower
# Train `SmallCNN(c1=4, c2=8)` and `SmallCNN(c1=16, c2=32)` with `fit` and compare validation accuracy and parameter counts with the default.
# Write two sentences: does more capacity help on this dataset, and how sure are you given the validation-set size? No assert: explore and report.

# %% [markdown]
# ## Recap
# - Conv layers take `(N, C, H, W)`; output side is `(n + 2p - k) // s + 1`; parameters are `out * in * k * k + out`, independent of image size.
# - `Conv -> ReLU -> Pool` blocks, then `flatten` and `Linear` to logits; `CrossEntropyLoss` wants raw logits.
# - Compare models with the same recipe and several seeds; small validation sets make small gaps meaningless.
# - Use validation for decisions and the test set once; read the confusion matrix and the misclassified images.
# - Translation robustness is something you measure, and augmentation matters as much as architecture.
#
# **Go deeper:** the PyTorch documentation for `torch.nn.Conv2d` and `torch.nn.MaxPool2d` (pytorch.org/docs), and the
# official PyTorch tutorials (pytorch.org/tutorials).
#
# **Next:** [Sequence models: RNN and LSTM](05_sequence_models_rnn_lstm.ipynb)
