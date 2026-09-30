# %% [markdown]
# # NumPy Mini Project: Images as Arrays and k-means from Scratch
#
# **Why this matters.** An image is just a 2-D (grayscale) or 3-D (colour) array, and clustering is just distance computations plus
# averaging. Building both with only NumPy is the best test that you have absorbed shapes, axes, broadcasting and masking. It is also a
# realistic interview task: "implement k-means without scikit-learn".
#
# **You will learn**
# - treat `sklearn.datasets.load_digits` images as arrays: normalise, flip, average
# - implement a 2-D convolution by hand and validate it against SciPy
# - implement k-means using only NumPy and check it against scikit-learn
#
# **Prerequisites:** all previous NumPy notebooks; the last one is [NumPy pitfalls](05_numpy_pitfalls_and_performance.ipynb).
#
# > Data used: scikit-learn's bundled `digits` (real 8x8 handwritten-digit scans, ships with the library, no download).

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.datasets import load_digits

digits = load_digits()
images, labels = digits.images, digits.target
print("images:", images.shape, images.dtype, "| labels:", labels.shape)
print("pixel range:", images.min(), "to", images.max())

# %% [markdown]
# `images` is `(1797, 8, 8)`: axis 0 indexes the image, axes 1 and 2 are rows and columns of pixels. Values run from 0 (background) to 16 (ink).

# %%
def show_grid(imgs, titles, cols=5, cmap="gray_r"):
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows))
    for ax, im, t in zip(np.ravel(axes), imgs, titles):
        ax.imshow(im, cmap=cmap)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    for ax in np.ravel(axes)[len(imgs):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

show_grid(images[:5], [f"label {l}" for l in labels[:5]])

# %% [markdown]
# ## Part 1. Normalise, flip, average
#
# **Normalising** rescales pixel values to a common range, so downstream algorithms are not dominated by scale.
# We scale to `[0, 1]` using the dataset-wide maximum (16), then also compute a per-image standardisation (zero mean, unit std).

# %%
X01 = images / 16.0
print("range after scaling:", X01.min(), X01.max())

flat = images.reshape(len(images), -1)                 # (1797, 64): one row per image, the format ML libraries want
mu = flat.mean(axis=1, keepdims=True)
sd = flat.std(axis=1, keepdims=True)
print("blank images (std == 0):", int((sd == 0).sum()))
standardised = (flat - mu) / np.where(sd == 0, 1, sd)  # guard against divide-by-zero on blank images
print(standardised.mean(axis=1)[:3].round(6), standardised.std(axis=1)[:3].round(6))

# %% [markdown]
# The `np.where(sd == 0, 1, sd)` guard matters: an all-blank image has zero variance and would produce `NaN`. We check first instead of hoping.
#
# **Flipping** is just slicing with a negative step (a view, no copy). `[:, ::-1]` flips left-right, `[::-1, :]` flips up-down.

# %%
one = images[3]
show_grid([one, one[:, ::-1], one[::-1, :], one.T], ["original", "flip LR", "flip UD", "transpose"], cols=4)

# %% [markdown]
# **Average image per digit:** boolean mask on axis 0, then mean over axis 0.

# %%
means = np.stack([images[labels == d].mean(axis=0) for d in range(10)])   # (10, 8, 8)
print(means.shape)
show_grid(means, [f"mean of {d}" for d in range(10)])

# %% [markdown]
# *What just happened?* `images[labels == d]` selects all images of digit `d` (a copy, by the mask rule); `.mean(axis=0)` collapses the image axis.
# The blurry averages already look like the digits, which is why even a very simple nearest-mean classifier works decently on this data.
# Let us test that claim honestly instead of asserting it:

# %%
dists = ((flat[:, None, :] - means.reshape(10, 64)[None, :, :]) ** 2).sum(axis=2)   # (1797, 10) via broadcasting
pred = dists.argmin(axis=1)
print(f"nearest-mean accuracy (on the same data it was fitted on): {(pred == labels).mean():.1%}")

# %% [markdown]
# Caveat: this is **training accuracy**; it is optimistic. The machine-learning module does this properly with a train/test split.

# %% [markdown]
# ## Part 2. Convolution by hand
#
# A 2-D convolution (strictly, *cross-correlation*, which is what deep-learning libraries call convolution) slides a small kernel over the image
# and, at each position, takes the sum of element-wise products. Edge-detecting kernels respond where brightness changes.
#
# We write it two ways: an obvious double loop, and a vectorised version using `sliding_window_view` (each window is a `(kh, kw)` patch).

# %%
def conv2d_loop(img, kernel):
    kh, kw = kernel.shape
    out_h, out_w = img.shape[0] - kh + 1, img.shape[1] - kw + 1      # "valid" convolution: no padding
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * kernel)
    return out


def conv2d_vec(img, kernel):
    windows = sliding_window_view(img, kernel.shape)                 # (out_h, out_w, kh, kw), a view
    return np.einsum("ijkl,kl->ij", windows, kernel)


sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)                        # responds to vertical edges

img = X01[3]
a, b = conv2d_loop(img, sobel_x), conv2d_vec(img, sobel_x)
print("output shape:", a.shape, "(8 - 3 + 1 = 6 on each side)")
print("loop == vectorised:", np.allclose(a, b))

# %% [markdown]
# Never trust your own implementation without an independent check. SciPy has one (`correlate2d`, `mode="valid"`):

# %%
from scipy.signal import correlate2d

print("matches scipy:", np.allclose(conv2d_vec(img, sobel_x), correlate2d(img, sobel_x, mode="valid")))

# %%
sobel_y = sobel_x.T
edges = np.hypot(conv2d_vec(img, sobel_x), conv2d_vec(img, sobel_y))    # edge strength = gradient magnitude
show_grid([img, conv2d_vec(img, sobel_x), conv2d_vec(img, sobel_y), edges], ["digit", "sobel x", "sobel y", "edge strength"], cols=4, cmap="viridis")

# %% [markdown]
# Timing the two versions on all images (best of a few runs, same output asserted) shows why vectorising matters even at 8x8:

# %%
import time

def time_it(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out

t_loop, r1 = time_it(lambda: np.stack([conv2d_loop(im, sobel_x) for im in X01[:300]]))
t_vec, r2 = time_it(lambda: np.stack([conv2d_vec(im, sobel_x) for im in X01[:300]]))
print(f"300 images: loop {t_loop * 1000:.0f} ms | vectorised {t_vec * 1000:.0f} ms | same result: {np.allclose(r1, r2)}")

# %% [markdown]
# ## Part 3. k-means from scratch (NumPy only)
#
# **Idea.** k-means groups points into `k` clusters by repeating two steps until nothing changes:
# 1. **Assign** each point to the nearest centroid.
# 2. **Update** each centroid to the mean of the points assigned to it.
#
# It minimises *inertia* (sum of squared distances to the assigned centroid), but only finds a **local** optimum that depends on the starting
# centroids, so we run several restarts and keep the best. First, some 2-D data with three known groups:

# %%
rng = np.random.default_rng(42)
true_centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 6.0]])
points = np.vstack([c + rng.normal(0, 0.9, size=(150, 2)) for c in true_centers])   # (450, 2)
true_group = np.repeat([0, 1, 2], 150)
print(points.shape)

# %%
def assign(points, centroids):
    """Index of the nearest centroid for every point (broadcast (n,1,d) - (1,k,d))."""
    d2 = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def kmeans(points, k, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), size=k, replace=False)].copy()   # start at k random data points
    for _ in range(n_iter):
        labels = assign(points, centroids)
        new = centroids.copy()
        for j in range(k):
            members = points[labels == j]
            if len(members):                       # an empty cluster keeps its old centroid
                new[j] = members.mean(axis=0)
        if np.allclose(new, centroids):
            break
        centroids = new
    labels = assign(points, centroids)
    inertia = ((points - centroids[labels]) ** 2).sum()
    return centroids, labels, inertia


def kmeans_best(points, k, restarts=10):
    runs = [kmeans(points, k, seed=s) for s in range(restarts)]
    return min(runs, key=lambda r: r[2])           # lowest inertia wins


centroids, cluster, inertia = kmeans_best(points, 3)
print("centroids:\n", centroids.round(2))
print("inertia:", round(float(inertia), 1))

# %% [markdown]
# Compare against scikit-learn on the same data. Cluster numbering is arbitrary, so we compare the **partitions** with the adjusted Rand index
# (1.0 means identical groupings) and the inertia:

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sk = KMeans(n_clusters=3, n_init=10, random_state=42).fit(points)
print("adjusted Rand vs sklearn :", round(adjusted_rand_score(sk.labels_, cluster), 3))
print("adjusted Rand vs truth   :", round(adjusted_rand_score(true_group, cluster), 3))
print("inertia ours / sklearn   :", round(float(inertia), 2), "/", round(float(sk.inertia_), 2))

# %%
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(points[:, 0], points[:, 1], c=cluster, s=10, cmap="viridis")
ax.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=120, label="centroids")
ax.set_title("k-means from scratch (k=3)")
ax.set_xlabel("feature 1 (arbitrary units)")
ax.set_ylabel("feature 2 (arbitrary units)")
ax.legend()
plt.show()

# %% [markdown]
# *What just happened?* Each iteration used broadcasting for the assign step (`(n, 1, d) - (1, k, d)`) and a boolean mask plus `mean(axis=0)` for the
# update step. Failure modes worth knowing: a bad initialisation can merge two true clusters (hence restarts); `k` must be chosen by you;
# k-means assumes roughly round, similar-sized clusters and is sensitive to feature scale (standardise first, as in the broadcasting notebook).

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting to normalise** before computing distances, so one large-scale feature dominates.
# 2. **A single k-means run** and trusting it: initialisation matters, so use restarts and compare inertia.
# 3. **Empty clusters** produce `NaN` means. Handle them (keep the old centroid or reinitialise).
# 4. **Comparing cluster ids directly** with true labels. Ids are arbitrary; use an alignment-free metric such as the adjusted Rand index.
# 5. **Convolution vs cross-correlation.** True convolution flips the kernel; deep-learning "convolution" does not. Kernels like Sobel change sign if you mix them up.
# 6. **Reporting training accuracy** as if it were performance on new data.

# %% [markdown]
# ## Interview corner
#
# **1. Walk me through k-means. What is its objective and is it guaranteed to find the best clustering?**
# <details><summary>Answer</summary>
# It alternates assigning points to the nearest centroid and recomputing centroids as means; this never increases the sum of squared distances, so it converges,
# but only to a local minimum that depends on initialisation. Use several restarts or k-means++ initialisation.
# </details>
#
# **2. How do you choose k?**
# <details><summary>Answer</summary>
# There is no single right method: the elbow plot of inertia against k, the silhouette score, stability across runs, and (most importantly) whether the
# clusters are useful for the business question. Inertia always decreases as k grows, so it cannot be minimised blindly.
# </details>
#
# **3. What is the time complexity of one k-means iteration?**
# <details><summary>Answer</summary>
# Roughly O(n * k * d) for n points, k clusters and d dimensions, dominated by the distance computations in the assign step.
# </details>
#
# **4. How would you flip an image, and does it copy the data?**
# <details><summary>Answer</summary>
# <code>img[:, ::-1]</code> for left-right; it is a view (negative-stride slice), so no copy is made until you write to it or force one with <code>.copy()</code>.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Ink coverage
# Compute `ink_fraction`: for each image, the fraction of its 64 pixels that are **greater than 8** (shape `(1797,)`), and `most_ink_digit`:
# the digit (0-9) whose images have the highest **average** `ink_fraction`.

# %% tags=["starter"]
ink_fraction = ...
most_ink_digit = ...

# %%
manual = np.array([np.mean([p > 8 for p in im.ravel()]) for im in images[:50]])
assert ink_fraction.shape == (1797,) and np.allclose(ink_fraction[:50], manual), "fraction of pixels > 8 per image"
avg = {d: ink_fraction[labels == d].mean() for d in range(10)}
assert most_ink_digit == max(avg, key=avg.get), f"expected digit {max(avg, key=avg.get)}"
print(f"Exercise 1 passed (digit {most_ink_digit} has the most ink)")

# %% tags=["solution"]
ink_fraction = (images > 8).mean(axis=(1, 2))
most_ink_digit = int(np.array([ink_fraction[labels == d].mean() for d in range(10)]).argmax())

# %% [markdown]
# ### Exercise 2: Symmetry score
# A digit like 0 or 8 is more left-right symmetric than a 7. Compute `asym`: for every image, the mean absolute difference between the image and its
# left-right mirror (shape `(1797,)`, no loops), and `most_symmetric_digit`: the digit with the **lowest** average `asym`.

# %% tags=["starter"]
asym = ...
most_symmetric_digit = ...

# %%
manual = np.array([np.abs(im - im[:, ::-1]).mean() for im in images[:20]])
assert asym.shape == (1797,) and np.allclose(asym[:20], manual), "mean(|img - mirrored img|) per image"
avg = [asym[labels == d].mean() for d in range(10)]
assert most_symmetric_digit == int(np.argmin(avg)), f"expected {int(np.argmin(avg))}"
print(f"Exercise 2 passed (most symmetric digit: {most_symmetric_digit})")

# %% tags=["solution"]
asym = np.abs(images - images[:, :, ::-1]).mean(axis=(1, 2))
most_symmetric_digit = int(np.array([asym[labels == d].mean() for d in range(10)]).argmin())

# %% [markdown]
# ### Exercise 3: A blur kernel
# Build `blur`: a 3x3 kernel where every entry is `1/9`. Apply it with `conv2d_vec` to image `X01[0]` and store the result in `blurred`.
# Then store in `energy_ratio` the ratio `blurred.var() / X01[0].var()` (blurring should lower the variance, so expect less than 1).

# %% tags=["starter"]
blur = ...
blurred = ...
energy_ratio = ...

# %%
assert blur.shape == (3, 3) and np.isclose(blur.sum(), 1.0), "a 3x3 averaging kernel sums to 1"
assert blurred.shape == (6, 6), f"valid convolution of 8x8 with 3x3 gives 6x6, got {blurred.shape}"
assert np.allclose(blurred[0, 0], X01[0][:3, :3].mean()), "each output pixel is the mean of a 3x3 patch"
assert np.isclose(energy_ratio, blurred.var() / X01[0].var()) and energy_ratio < 1, "blurring reduces variance"
print(f"Exercise 3 passed (variance ratio {energy_ratio:.2f})")

# %% tags=["solution"]
blur = np.full((3, 3), 1 / 9)
blurred = conv2d_vec(X01[0], blur)
energy_ratio = blurred.var() / X01[0].var()

# %% [markdown]
# ### Exercise 4: Update step of k-means
# Write `update_centroids(points, labels, k)` returning a `(k, d)` array of cluster means (assume no cluster is empty). Use it on the
# labels from `assign(points, centroids)` for the starting centroids `start`, and store the result in `new_centroids`.

# %%
start = points[[0, 150, 300]].copy()

# %% tags=["starter"]
def update_centroids(points, labels, k):
    ...

lab = assign(points, start)
new_centroids = update_centroids(points, lab, 3)

# %%
lab = assign(points, start)
expected = np.array([points[lab == j].mean(axis=0) for j in range(3)])
assert new_centroids.shape == (3, 2), f"expected (3, 2), got {new_centroids.shape}"
assert np.allclose(new_centroids, expected), "each row is the mean of the points assigned to that cluster"
inertia_before = ((points - start[lab]) ** 2).sum()
inertia_after = ((points - new_centroids[lab]) ** 2).sum()
assert inertia_after <= inertia_before + 1e-9, "the update step can only lower (or keep) the inertia"
print(f"Exercise 4 passed (inertia {inertia_before:.0f} -> {inertia_after:.0f})")

# %% tags=["solution"]
def update_centroids(points, labels, k):
    return np.stack([points[labels == j].mean(axis=0) for j in range(k)])

lab = assign(points, start)
new_centroids = update_centroids(points, lab, 3)

# %% [markdown]
# ### Exercise 5 (open-ended): Cluster the digits
# Run `kmeans_best(flat.astype(float), 10)` on the 64-dimensional digit vectors (no labels used). For each cluster, find which true digit is
# the most common inside it and what share of the cluster that digit makes up (its *purity*). Report in a markdown cell which digits k-means
# mixes together and one idea to improve it (scaling, PCA first, more restarts).

# %% [markdown]
# ## Recap
# - Images are arrays: `(N, H, W)`; normalise with broadcasting, flip with negative-step slices, average with masks and `mean(axis=0)`.
# - A convolution slides a kernel and sums products; `sliding_window_view` + `einsum` is the vectorised form. Validate against SciPy.
# - k-means = assign (broadcast distances + `argmin`) then update (mask + `mean`); use restarts and compare against a trusted implementation.
# - Cluster ids are arbitrary; compare partitions with the adjusted Rand index.
#
# **You have finished the NumPy module.** Head to the [module README](README.md) for a checklist and next resources.
#
# **Go deeper:** scikit-learn user guide "Clustering" (k-means, k-means++ and evaluation metrics), scikit-learn.org/stable.
#
# **Next:** [Loading and inspecting data with pandas](../05_Pandas/01_loading_and_inspecting_data.ipynb)
