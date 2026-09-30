# %% [markdown]
# # Convolutions and CNN Intuition
#
# **Why this matters.** A dense layer treats an image as a bag of unrelated numbers: a cat in the top-left corner and the same cat in the bottom-right look like completely different inputs,
# and every pixel needs its own weight to every unit. Convolutional layers bake in two facts about images (nearby pixels belong together, and a pattern means the same thing wherever it appears),
# which is why they need far fewer parameters and generalise better. If you can compute a convolution by hand and count the parameters of a conv stack, you can read any CNN architecture
# in a paper or code review.
#
# **You will learn**
# - compute a 2-D convolution (cross-correlation, as deep-learning libraries define it) with loops and check it against SciPy and PyTorch
# - see what hand-made filters (vertical edge, horizontal edge, blur) do to a real digit image
# - predict output sizes from kernel, stride and padding, and compute pooling by hand
# - count parameters: conv vs dense, and why the conv count does not depend on the image size
# - check translation equivariance and see whether edge features help a simple classifier
#
# **Prerequisites:** [01 Neurons, layers and the forward pass](01_neurons_layers_and_the_forward_pass.ipynb) (parameter counting) and NumPy array slicing.
#
# > Data used: scikit-learn's bundled `digits` (1,797 real 8x8 grayscale handwritten digits, pixel values 0-16).

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from scipy.signal import correlate2d
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
torch.set_num_threads(2)
np.set_printoptions(precision=2, suppress=True, linewidth=120)

digits = load_digits()
images = digits.images / 16.0                   # (1797, 8, 8), scaled to 0-1
labels = digits.target
print("images:", images.shape, " one pixel row of a '3':")
print(images[labels == 3][0][3])

# %% [markdown]
# ## 1. A convolution by hand
#
# Take a small grid of weights, the **kernel** (or filter), say 3x3. Slide it over the image. At each position multiply the kernel with the 3x3 patch underneath, element by element, and add everything up:
# that sum is one output pixel. The output image is called a **feature map**.
#
# A note on naming: what deep-learning libraries call "convolution" is technically *cross-correlation* (the kernel is not flipped). Since the kernel is learned, the flip does not matter, and we follow the library convention.

# %%
def conv2d_valid(image, kernel):
    """Slide `kernel` over `image` with no padding and stride 1."""
    H, W = image.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = image[i:i + kh, j:j + kw]
            out[i, j] = np.sum(patch * kernel)
    return out


img = images[labels == 3][0]
vertical_edge = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=float)
fmap = conv2d_valid(img, vertical_edge)
print("input", img.shape, "-> feature map", fmap.shape)

# Three independent implementations must agree
scipy_out = correlate2d(img, vertical_edge, mode="valid")
torch_out = F.conv2d(torch.tensor(img)[None, None], torch.tensor(vertical_edge)[None, None])[0, 0].numpy()
print("matches SciPy:", np.allclose(fmap, scipy_out), "| matches PyTorch:", np.allclose(fmap, torch_out))

# %% [markdown]
# *What just happened?* Our loops, SciPy and PyTorch give the same numbers, so `F.conv2d` is no more than this sliding dot product, run fast. Note the shapes: PyTorch wants
# `(batch, channels, height, width)`, which is why we added two leading dimensions with `[None, None]`.
#
# ## 2. What a filter detects
#
# The `vertical_edge` kernel subtracts the left column from the right column. On a flat region the sum is zero. Where pixel values jump from dark to bright moving rightwards, the response is large and positive;
# bright to dark gives a large negative value. Transpose it for horizontal edges; average a patch for a blur.

# %%
kernels = {
    "original": None,
    "vertical edge": vertical_edge,
    "horizontal edge": vertical_edge.T,
    "blur (mean)": np.ones((3, 3)) / 9.0,
}
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
for ax, (name, k) in zip(axes, kernels.items()):
    shown = img if k is None else conv2d_valid(img, k)
    im = ax.imshow(shown, cmap="RdBu_r" if k is not None and k.sum() == 0 else "gray_r")
    ax.set_title(name)
    ax.set_xlabel("column (pixel)")
    ax.set_ylabel("row (pixel)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Red and blue mark opposite edge directions; grey areas are flat. A CNN does not use hand-made kernels: it **learns** the nine numbers of every kernel by backpropagation, and the first layer of a trained
# CNN typically ends up with edge-and-blob detectors much like these. Deeper layers combine them into strokes, loops and parts.
#
# ## 3. Padding, stride and output size
#
# Two knobs control the output size. **Padding** `p` adds a border of zeros (so the kernel can be centred on edge pixels); **stride** `s` moves the window `s` pixels at a time.
# For input side `n` and kernel side `k`:
#
# `output side = floor((n + 2p - k) / s) + 1`
#
# "Valid" convolution is `p=0`. With `k=3, p=1, s=1` the size is preserved ("same" padding for odd kernels).

# %%
def conv_out_size(n, k, p=0, s=1):
    return (n + 2 * p - k) // s + 1


for n, k, p, s in [(8, 3, 0, 1), (8, 3, 1, 1), (8, 3, 1, 2), (8, 5, 2, 1), (28, 5, 0, 1), (28, 3, 1, 2)]:
    x = torch.zeros(1, 1, n, n)
    actual = nn.Conv2d(1, 1, kernel_size=k, padding=p, stride=s)(x).shape[-1]
    print(f"n={n:2d} k={k} p={p} s={s} -> formula {conv_out_size(n, k, p, s):2d}, PyTorch {actual:2d}")

# %% [markdown]
# ## 4. Pooling: shrink and tolerate small shifts
#
# **Max pooling** with a 2x2 window and stride 2 keeps only the largest value in each 2x2 block, halving height and width. It has no parameters. It gives a little tolerance to exact position:
# if a stroke moves by one pixel *within* a block, the pooled output does not change.

# %%
def max_pool2d_2x2(fm):
    H, W = fm.shape
    out = np.zeros((H // 2, W // 2))
    for i in range(H // 2):
        for j in range(W // 2):
            out[i, j] = fm[2 * i:2 * i + 2, 2 * j:2 * j + 2].max()
    return out


fm = np.abs(conv2d_valid(img, vertical_edge))                  # 6x6
pooled = max_pool2d_2x2(fm)
torch_pooled = F.max_pool2d(torch.tensor(fm)[None, None], 2)[0, 0].numpy()
print("feature map", fm.shape, "-> pooled", pooled.shape, "| matches F.max_pool2d:", np.allclose(pooled, torch_pooled))

# %% [markdown]
# ## 5. Parameters: conv vs dense
#
# A **dense** layer from an input with `N` values to `M` units has `N*M + M` parameters. A **conv** layer with `C_in` input channels, `C_out` filters and a `k x k` kernel has
# `C_out * C_in * k * k + C_out` parameters, *whatever the image size*, because the same small kernel is reused at every position ("weight sharing").
# Channels matter: each filter spans **all** input channels, so a `3x3` filter on an RGB image has `3*3*3 = 27` weights plus one bias.

# %%
def count_params(module):
    return sum(p.numel() for p in module.parameters())


conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
print("conv 1->8 channels, 3x3 :", count_params(conv), "= 8*1*3*3 + 8 =", 8 * 1 * 3 * 3 + 8)
print("output for one 8x8 image :", tuple(conv(torch.zeros(1, 1, 8, 8)).shape))

# A dense layer producing the SAME output volume (8 x 8 x 8 = 512 values) from the 64 pixels
dense = nn.Linear(64, 8 * 8 * 8)
print("dense giving same output :", count_params(dense))

# %% [markdown]
# On tiny 8x8 images the gap is modest. It grows quickly with image size. Here is a simple calculation (no big layers actually allocated) for a 224x224 RGB image and 64 output feature maps of the same size:

# %%
side, c_in, c_out, k = 224, 3, 64, 3
conv_params = c_out * c_in * k * k + c_out
dense_params = (side * side * c_in) * (side * side * c_out) + side * side * c_out
print(f"conv 3x3, 3 -> 64 channels : {conv_params:,} parameters")
print(f"dense, same input/output   : {dense_params:,} parameters ({dense_params / conv_params:,.0f}x more)")

# %% [markdown]
# *What just happened?* The conv layer needs 1,792 numbers and does not care whether the image is 8 pixels or 2,000 pixels wide; the dense equivalent is impractically large.
# Weight sharing is both a saving and a **prior belief**: "a useful pattern detector is useful everywhere". That belief is usually true for images and false for, say, a table of unrelated features.
#
# A typical small CNN stacks `Conv -> ReLU -> Pool` blocks then flattens into a dense classifier. Count one by hand, then verify:

# %%
net = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 8x8 -> 8x8 -> 4x4, 8 channels
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 4x4 -> 4x4 -> 2x2, 16 channels
    nn.Flatten(),                                                     # 16*2*2 = 64
    nn.Linear(64, 10),
)
x = torch.zeros(1, 1, 8, 8)
for layer in net:
    x = layer(x)
    print(f"{layer.__class__.__name__:10s} -> {tuple(x.shape)}")
by_hand = (8 * 1 * 9 + 8) + (16 * 8 * 9 + 16) + (64 * 10 + 10)
print("parameters by hand:", by_hand, "| counted:", count_params(net))

# %% [markdown]
# About 1,250 parameters in the two conv layers plus 650 in the classifier (1,898 in total). A dense `64 -> 128 -> 10` network has 9,610, roughly five times more than this whole CNN.
#
# ## 6. Translation equivariance: shift the input, the feature map shifts
#
# If you slide the digit one pixel to the right, a convolution's feature map slides one pixel to the right too (in the interior, away from the padded border). Pooling then
# turns "moves with the input" into "changes less". Let us verify the first claim numerically with a 12x12 canvas so the digit has room.

# %%
canvas = np.zeros((12, 12))
canvas[2:10, 2:10] = img
shifted = np.roll(canvas, shift=1, axis=1)                      # move one column right (no wrap-around: canvas border is empty)
k = vertical_edge
a = correlate2d(canvas, k, mode="valid")
b = correlate2d(shifted, k, mode="valid")
print("feature map of shifted image equals the shifted feature map:", np.allclose(b[:, 1:], a[:, :-1]))

# %% [markdown]
# A dense network has no such property: shifting the digit changes *every* input pixel position and so every weight that sees it.
#
# ## 7. Do edge features actually help? A measured answer
#
# Let us give a plain **logistic regression** two kinds of input: the 64 raw pixels, or pooled vertical+horizontal edge maps computed by the hand-made kernels above (this is, roughly, what a one-layer CNN
# with two fixed filters would hand to its classifier). We tune nothing and report both on the same held-out split.

# %%
def edge_features(batch):
    feats = []
    for im in batch:
        maps = [np.abs(conv2d_valid(im, k)) for k in (vertical_edge, vertical_edge.T)]
        feats.append(np.concatenate([max_pool2d_2x2(m).ravel() for m in maps]))
    return np.array(feats)


idx_tr, idx_te = train_test_split(np.arange(len(images)), test_size=0.3, random_state=42, stratify=labels)
raw = images.reshape(len(images), -1)
edge = edge_features(images)
print("raw features:", raw.shape[1], "| pooled edge features:", edge.shape[1])

for name, feats in [("raw pixels", raw), ("edge maps", edge), ("pixels + edges", np.hstack([raw, edge]))]:
    clf = LogisticRegression(max_iter=2000).fit(feats[idx_tr], labels[idx_tr])
    print(f"{name:15s} test accuracy: {clf.score(feats[idx_te], labels[idx_te]):.1%}")

# %% [markdown]
# Be careful how you read this: digits are tiny and already quite separable, so raw pixels are a strong baseline. In our run the two fixed edge maps alone did clearly worse (they throw away information),
# and adding them to the pixels helped by only about a point, which on 540 test images is within noise.
# The point of a CNN is not that edges beat pixels on 8x8 digits; it is that learned filters scale to large, messy images where raw pixels and dense layers do not. You will train a real CNN in the PyTorch module.

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting the channel dimension.** `Conv2d` wants `(N, C, H, W)`. A `(H, W)` array or an `(N, H, W)` batch raises an error (or, worse, is read as the wrong thing). Use `.unsqueeze(1)` for grayscale batches.
# 2. **Flattening to the wrong size.** The first `Linear` after the conv stack needs `C * H * W` of the *last* feature map. Compute it, or push a dummy tensor through and read the shape (as we did).
# 3. **Counting bias once per layer, not once per filter.** A conv layer has one bias per output channel.
# 4. **Believing convolution makes a CNN rotation- or scale-invariant.** It gives translation equivariance only; rotation and scale robustness come from data augmentation.
# 5. **Expecting the "same" size without padding.** With `k=3` and `p=0` you lose two pixels per side per layer; deep stacks without padding can shrink to nothing.

# %%
try:
    nn.Conv2d(1, 4, 3)(torch.zeros(5, 8, 8))              # missing the channel dimension
except RuntimeError as e:
    print("RuntimeError:", str(e)[:110], "...")

# %% [markdown]
# ## Interview corner
#
# **1. Why does a CNN have fewer parameters than a dense network for the same image?**
# <details><summary>Answer</summary>
# Local connectivity (each output looks at a small patch) and weight sharing (the same kernel is reused at every position). The parameter count of a conv layer is
# <code>C_out * C_in * k * k + C_out</code> regardless of image size.
# </details>
#
# **2. What is the output size of a conv layer?**
# <details><summary>Answer</summary>
# <code>floor((n + 2p - k) / s) + 1</code> per spatial dimension, where n is the input size, k the kernel size, p the padding and s the stride. The number of output channels equals the number of filters.
# </details>
#
# **3. What does pooling do, and is it necessary?**
# <details><summary>Answer</summary>
# It downsamples feature maps (less compute, larger receptive field) and adds small-shift tolerance. It has no parameters. It is not strictly necessary: many modern networks downsample with strided convolutions instead.
# </details>
#
# **4. Is a convolutional layer translation invariant?**
# <details><summary>Answer</summary>
# Equivariant, not invariant: shifting the input shifts the feature map. Invariance (the final prediction not changing) emerges from pooling and the classifier on top, and only approximately.
# </details>

# %% [markdown]
# ## Exercises
#
# Write your answer in the starter cell, then run the check cell below it. Stuck? Expand the hidden solution cell at the bottom of each exercise.

# %% [markdown]
# ### Exercise 1: Output size, including stride and padding
# Write `feature_map_side(n, k, p, s)` returning the side length after a conv layer, then use it to store in `side_after_two` the side of a 32-pixel image after
# `Conv(k=5, p=0, s=1)` followed by `Conv(k=3, p=1, s=2)`. *Hint: the formula in section 3.*

# %% tags=["starter"]
def feature_map_side(n, k, p, s):
    ...


side_after_two = ...

# %%
for n, k, p, s in [(32, 5, 0, 1), (17, 3, 1, 2), (10, 4, 0, 3), (8, 1, 0, 1)]:
    actual = nn.Conv2d(1, 1, k, padding=p, stride=s)(torch.zeros(1, 1, n, n)).shape[-1]
    assert feature_map_side(n, k, p, s) == actual, f"wrong for n={n}, k={k}, p={p}, s={s}: PyTorch gives {actual}"
chain = nn.Sequential(nn.Conv2d(1, 1, 5), nn.Conv2d(1, 1, 3, padding=1, stride=2))(torch.zeros(1, 1, 32, 32)).shape[-1]
assert side_after_two == chain, f"expected {chain}, got {side_after_two}"
print("Exercise 1 passed, final side:", side_after_two)

# %% tags=["solution"]
def feature_map_side(n, k, p, s):
    return (n + 2 * p - k) // s + 1


side_after_two = feature_map_side(feature_map_side(32, 5, 0, 1), 3, 1, 2)

# %% [markdown]
# ### Exercise 2: Average pooling by hand
# Write `avg_pool2d_2x2(fm)` for an array with even height and width: replace every non-overlapping 2x2 block by its **mean**. (Section 4 did max pooling with loops; try to do this one
# with a single `reshape` and `.mean`.) *Hint: `fm.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))`.*

# %% tags=["starter"]
def avg_pool2d_2x2(fm):
    ...

# %%
rng = np.random.default_rng(0)
test_fm = rng.standard_normal((6, 8))
expected = F.avg_pool2d(torch.tensor(test_fm)[None, None], 2)[0, 0].numpy()
assert avg_pool2d_2x2(test_fm).shape == (3, 4), "output should be half the height and width"
assert np.allclose(avg_pool2d_2x2(test_fm), expected), "values differ from torch.nn.functional.avg_pool2d"
print("Exercise 2 passed")

# %% tags=["solution"]
def avg_pool2d_2x2(fm):
    H, W = fm.shape
    return fm.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))

# %% [markdown]
# ### Exercise 3: Count the parameters of a CNN without running it
# A network is: `Conv2d(3, 16, 5)`, `ReLU`, `MaxPool2d(2)`, `Conv2d(16, 32, 3)`, `ReLU`, `AdaptiveAvgPool2d(1)`, `Flatten`, `Linear(32, 5)`. (Adaptive average pooling reduces every
# channel to one number, so the image size does not matter.) Store the total in `total_params` using the formulas only, then the check builds the model and counts.

# %% tags=["starter"]
total_params = ...

# %%
model = nn.Sequential(
    nn.Conv2d(3, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(32, 5),
)
assert total_params == count_params(model), f"counted {count_params(model)}, you computed {total_params}"
print("Exercise 3 passed:", total_params)

# %% tags=["solution"]
total_params = (16 * 3 * 5 * 5 + 16) + (32 * 16 * 3 * 3 + 32) + (32 * 5 + 5)

# %% [markdown]
# ### Exercise 4: Which digit has the most vertical edge energy?
# Using `vertical_edge` and `conv2d_valid` on every image, compute each image's mean absolute response (a number per image), then average those by digit class. Store the class (0-9)
# with the largest average in `most_vertical_digit` and the per-class array (length 10) in `class_energy`. Before you run it, guess which digit it will be, then look at the ranking.

# %% tags=["starter"]
class_energy = ...
most_vertical_digit = ...

# %%
ref = np.array([np.abs(correlate2d(im, vertical_edge, mode="valid")).mean() for im in images])
ref_class = np.array([ref[labels == c].mean() for c in range(10)])
assert np.shape(class_energy) == (10,), "class_energy must have one value per digit class"
assert np.allclose(class_energy, ref_class), "class_energy does not match an independent SciPy computation"
assert most_vertical_digit == int(ref_class.argmax()), f"expected digit {int(ref_class.argmax())}"
print(f"Exercise 4 passed: digit {most_vertical_digit} (ranking: {np.argsort(-ref_class).tolist()})")

# %% tags=["solution"]
per_image = np.array([np.abs(conv2d_valid(im, vertical_edge)).mean() for im in images])
class_energy = np.array([per_image[labels == c].mean() for c in range(10)])
most_vertical_digit = int(class_energy.argmax())

# %% [markdown]
# ### Exercise 5 (open-ended): Design a filter
# Invent a 3x3 kernel (for example a diagonal-edge detector, or a corner detector), apply it to a few digits with `conv2d_valid`, and plot the feature maps. Which digit classes respond most to it?
# Write two sentences on what kind of stroke your filter finds. No assert: explore and report.

# %% [markdown]
# ## Recap
# - A convolution slides a small kernel over the image and takes a dot product at each position; libraries actually compute cross-correlation.
# - Output side is `floor((n + 2p - k) / s) + 1`; output channels equal the number of filters; `Conv2d` wants `(N, C, H, W)`.
# - Conv parameters are `C_out * C_in * k * k + C_out`, independent of image size; that is weight sharing, which is also a prior about images.
# - Convolution is translation-equivariant; pooling adds tolerance to small shifts. Rotation and scale robustness come from augmentation.
# - On tiny, easy data, simple baselines (raw pixels + logistic regression) are strong. CNNs earn their place on bigger, messier images.
#
# **Go deeper:** Stanford CS231n notes on convolutional networks (cs231n.github.io/convolutional-networks); the PyTorch docs for `torch.nn.Conv2d` and `torch.nn.MaxPool2d` (pytorch.org/docs);
# "A guide to convolution arithmetic for deep learning" (Dumoulin and Visin) for padding and stride diagrams.
#
# **Next:** [Deep learning debugging checklist](05_deep_learning_debugging_checklist.ipynb)
