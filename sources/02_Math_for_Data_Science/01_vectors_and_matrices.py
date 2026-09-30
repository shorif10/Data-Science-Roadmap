# %% [markdown]
# # Vectors and Matrices: The Language of Data
#
# **Why this matters.** A row in your customer table, a word embedding, the weights of a linear model and the
# pixels of an image are all *vectors*. "Find customers similar to this one", "score every house with a linear
# model" and "run one layer of a neural network" are all a dot product or a matrix multiplication. If you
# can read `X @ w`, most of machine learning stops looking like magic.
#
# **You will learn**
# - treat a vector as an arrow *and* as a list of numbers, and use the dot product as a similarity score
# - compute L1, L2 and cosine measures, and know which one to use when
# - see a matrix as a machine that moves points (rotate, stretch, shear) and compose two machines
# - score a whole dataset with one `X @ w` and debug shape errors
#
# **Prerequisites:** [NumPy arrays](../04_NumPy/). No calculus needed yet.
#
# > Data used: `datasets/housing.csv` (synthetic house prices, see `docs/DATASETS.md`) plus tiny hand-made examples.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)
DATA = Path("../datasets")

# %% [markdown]
# ## 1. A vector is a list of numbers you can also draw
#
# The vector `[3, 4]` is a point at x=3, y=4, or equivalently an arrow from the origin to that point.
# A customer described by `[tenure, monthly_charge, support_calls]` is the same idea in 3 dimensions;
# we just cannot draw it. All the rules below work in any number of dimensions.

# %%
a = np.array([3.0, 1.0])
b = np.array([1.0, 2.0])

print("a + b  =", a + b)      # add component by component: walk along a, then along b
print("2 * a  =", 2 * a)      # scaling stretches the arrow, same direction
print("a - b  =", a - b)      # the arrow that goes from b to a

# %% [markdown]
# *What just happened?* Vector addition and scaling are just element-wise NumPy arithmetic. Every linear model
# is built from only these two operations: scale some vectors and add them up (a *linear combination*).

# %% [markdown]
# ## 2. The dot product: one number that says "how aligned"
#
# Multiply matching components and add: `a . b = a1*b1 + a2*b2 + ...`. The geometric meaning is more useful:
#
# `a . b = |a| * |b| * cos(theta)`, where `theta` is the angle between the arrows.
#
# - pointing the same way -> large positive; at right angles (**orthogonal**) -> exactly 0; opposite -> negative.
# - So the dot product is an *agreement score*. Recommenders, search and attention layers in transformers all use it.

# %%
manual = a[0] * b[0] + a[1] * b[1]
print("by hand :", manual)
print("np.dot  :", np.dot(a, b), "  a @ b:", a @ b)

# orthogonal vectors have dot product 0
print("[1,0] . [0,5] =", np.array([1, 0]) @ np.array([0, 5]))

# %% [markdown]
# Let us see it. We plot two arrows and print the angle recovered from the dot product formula.

# %%
theta = np.degrees(np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))

fig, ax = plt.subplots(figsize=(4.5, 4.5))
for v, name, c in [(a, "a", "tab:blue"), (b, "b", "tab:orange")]:
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    ax.text(v[0] + 0.05, v[1] + 0.05, name, color=c, fontsize=12)
ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.5, 3.5); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title(f"Angle between a and b = {theta:.1f} degrees")
ax.set_xlabel("x component"); ax.set_ylabel("y component")
plt.show()

# %% [markdown]
# ## 3. Length (norm) and three ways to measure "distance"
#
# | Name | Formula | Intuition | Typical use |
# |---|---|---|---|
# | L2 (Euclidean) | `sqrt(sum(x_i^2))` | straight-line length | k-NN, k-means, ridge penalty |
# | L1 (Manhattan) | `sum(abs(x_i))` | walking along city blocks | lasso penalty, MAE loss, robust to outliers |
# | Cosine similarity | `a.b / (|a||b|)` | ignores length, compares *direction* | text/embedding similarity |

# %%
v = np.array([3.0, -4.0])
print("L2 :", np.linalg.norm(v))            # 5.0 : the 3-4-5 triangle
print("L1 :", np.linalg.norm(v, ord=1))     # 7.0
print("Linf (largest abs component):", np.linalg.norm(v, ord=np.inf))

# %% [markdown]
# Cosine similarity is the dot product after shrinking both vectors to length 1. Why does that matter?
# A short document and a long one about the same topic have very different *lengths* of word-count vector but the
# same *direction*.

# %%
def cosine(u, w):
    return (u @ w) / (np.linalg.norm(u) * np.linalg.norm(w))

# word counts for the vocabulary [refund, shipping, delay, price]
short_doc = np.array([2, 1, 1, 0])
long_doc = np.array([20, 10, 10, 0])       # same topic, 10x longer
other_doc = np.array([0, 0, 1, 9])         # about price

print("Euclidean distance short vs long :", np.linalg.norm(short_doc - long_doc).round(2))
print("Euclidean distance short vs other:", np.linalg.norm(short_doc - other_doc).round(2))
print("Cosine short vs long :", cosine(short_doc, long_doc).round(3))
print("Cosine short vs other:", cosine(short_doc, other_doc).round(3))

# %% [markdown]
# *What just happened?* By Euclidean distance the short document looks *closer* to the unrelated one than to its
# own 10x-longer twin, because length dominates. Cosine says the twin is a perfect match (1.0). Pick the measure
# that matches what "similar" should mean for your problem.

# %% [markdown]
# ## 4. A matrix is a machine that moves points
#
# A matrix `M` with shape (2, 2) takes a 2-D vector `x` and returns a new 2-D vector `M @ x`.
# The columns of `M` tell you where the unit arrows `[1,0]` and `[0,1]` land; everything else follows because
# transformations by matrices are *linear* (straight lines stay straight, the origin stays put).
#
# We push a square and a unit circle through three different matrices.

# %%
angle = np.radians(30)
rotate = np.array([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle),  np.cos(angle)]])
stretch = np.array([[2.0, 0.0],
                    [0.0, 0.5]])
shear = np.array([[1.0, 0.8],
                  [0.0, 1.0]])

t = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(t), np.sin(t)])              # shape (2, 200): one point per column
square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, (name, M) in zip(axes, [("identity", np.eye(2)), ("rotate 30 deg", rotate),
                                ("stretch x2, y0.5", stretch), ("shear", shear)]):
    for shape, c in [(circle, "tab:blue"), (square, "tab:red")]:
        moved = M @ shape
        ax.plot(moved[0], moved[1], color=c)
    ax.set_title(name); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout(); plt.show()

# %% [markdown]
# *What just happened?* The rotation kept shapes intact, the stretch turned the circle into an ellipse, the shear
# slanted the square. In a neural network, each layer is exactly this: a matrix moves the data, then a
# non-linear function bends it. Rotation matrices are special: they preserve length and angles. Check that:

# %%
p = np.array([3.0, 4.0])
print("length before:", np.linalg.norm(p), " after rotation:", np.linalg.norm(rotate @ p).round(6))
print("R.T @ R is the identity:", np.allclose(rotate.T @ rotate, np.eye(2)))

# %% [markdown]
# ## 5. Multiplying matrices = doing one transformation after another
#
# `(B @ A) @ x` means "apply A first, then B". Order matters: `A @ B` is generally *not* `B @ A`.

# %%
rot_then_stretch = stretch @ rotate
stretch_then_rot = rotate @ stretch
print("rotate first then stretch:\n", rot_then_stretch)
print("stretch first then rotate:\n", stretch_then_rot)
print("same result?", np.allclose(rot_then_stretch, stretch_then_rot))

# %% [markdown]
# ## 6. Real data: scoring every house at once with `X @ w`
#
# A data table with `n` rows and `d` features is a matrix `X` of shape `(n, d)`. A linear model has a weight
# vector `w` of shape `(d,)`. Then `X @ w` gives all `n` predictions in one shot: each row is dotted with `w`.
# We use made-up weights (not fitted; fitting comes in the gradient descent notebook).

# %%
housing = pd.read_csv(DATA / "housing.csv")
h = housing[(housing["price"] > 0) & (housing["price"] < 2_000_000)].dropna(subset=["bedrooms", "bathrooms", "area_sqft"])
X = h[["area_sqft", "bedrooms", "bathrooms"]].to_numpy()
w = np.array([150.0, 5_000.0, 10_000.0])   # dollars per sqft, per bedroom, per bathroom (illustrative)
bias = 20_000.0

pred = X @ w + bias
print("X shape:", X.shape, " w shape:", w.shape, " predictions shape:", pred.shape)
print("first 3 predictions:", pred[:3].round(0))
print("first 3 actual     :", h["price"].to_numpy()[:3])

# %% [markdown]
# *What just happened?* `X @ w` replaced a Python loop over every row. Row 0 is `150*area + 5000*beds + 10000*baths`.
# Let us prove that the vectorised form equals the loop, and see how much faster it is.

# %%
import time

t0 = time.perf_counter()
loop_pred = np.array([sum(X[i, j] * w[j] for j in range(3)) + bias for i in range(len(X))])
t_loop = time.perf_counter() - t0
t0 = time.perf_counter()
vec_pred = X @ w + bias
t_vec = time.perf_counter() - t0
print("identical:", np.allclose(loop_pred, vec_pred), f"| loop {t_loop*1000:.1f} ms vs matmul {t_vec*1000:.3f} ms")

# %% [markdown]
# ## Common mistakes
#
# 1. **Shape mismatch.** `(n, d) @ (d,)` works; `(n, d) @ (n,)` does not. The inner dimensions must match.
# 2. **`*` is not `@`.** `*` multiplies element by element (and broadcasts); `@` is the matrix/dot product.
# 3. **Using Euclidean distance on unscaled features.** Salary in dollars swamps age in years. Standardise first.
# 4. **Assuming `A @ B == B @ A`.** It is only true in special cases.

# %%
try:
    X @ np.ones(5)
except ValueError as e:
    print("ValueError:", str(e)[:110])

u = np.array([1, 2, 3]); z = np.array([4, 5, 6])
print("u * z =", u * z, "  but u @ z =", u @ z)

# %% [markdown]
# ## Interview corner
#
# **1. Why use cosine similarity instead of Euclidean distance for text embeddings?**
# <details><summary>Answer</summary>
# Cosine compares direction and ignores magnitude, and magnitude often reflects length or frequency rather than meaning.
# For vectors normalised to length 1 the two are monotonically related, so the choice matters only when lengths differ.
# </details>
#
# **2. What does it mean geometrically that two vectors have a zero dot product?**
# <details><summary>Answer</summary>
# They are orthogonal (perpendicular). In statistics, uncorrelated centred variables have orthogonal vectors; in PCA the components are orthogonal.
# </details>
#
# **3. Explain `X @ w` for linear regression. What are the shapes?**
# <details><summary>Answer</summary>
# X is (n_samples, n_features), w is (n_features,), so X @ w is (n_samples,): one prediction per row, each a weighted sum of that row's features (plus a bias).
# </details>
#
# **4. Is matrix multiplication commutative? Associative?**
# <details><summary>Answer</summary>
# Not commutative in general, but associative: (AB)C = A(BC). Associativity lets frameworks choose the cheapest multiplication order.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Which support ticket is the closest match?
# A new ticket has word counts `query` over the vocabulary `[refund, shipping, delay, price, warranty]`. Three archived tickets are
# rows of `archive`. Store in `best_row` the **row index** of the archive ticket with the highest **cosine similarity** to `query`.
# *Hint: normalise each row with `np.linalg.norm(archive, axis=1)`.*

# %%
query = np.array([0, 3, 4, 1, 0])
archive = np.array([
    [5, 0, 0, 2, 3],     # refund/warranty topic
    [0, 30, 41, 12, 0],  # shipping/delay topic, long
    [1, 1, 1, 8, 0],     # price topic
])

# %% tags=["starter"]
best_row = ...

# %%
sims = (archive @ query) / (np.linalg.norm(archive, axis=1) * np.linalg.norm(query))
assert best_row == int(np.argmax(sims)), f"cosine similarities are {sims.round(3)}; pick the largest"
print("Exercise 1 passed, similarities:", sims.round(3))

# %% tags=["solution"]
best_row = int(np.argmax((archive @ query) / (np.linalg.norm(archive, axis=1) * np.linalg.norm(query))))

# %% [markdown]
# ### Exercise 2: Build a rotation matrix
# Write a function `rotation(deg)` that returns the 2x2 matrix rotating vectors counter-clockwise by `deg` degrees.
# Then store in `moved` the point `[1, 0]` rotated by 90 degrees. *Hint: the section 4 formula; `np.radians` converts.*

# %% tags=["starter"]
def rotation(deg):
    ...

moved = ...

# %%
R = rotation(37)
assert R.shape == (2, 2), "rotation should return a 2x2 array"
assert np.allclose(R.T @ R, np.eye(2)), "a rotation matrix must satisfy R.T @ R = I"
assert np.allclose(rotation(90) @ np.array([1, 0]), [0, 1]), "rotating [1,0] by 90 deg should give [0,1]"
assert np.allclose(moved, [0, 1]), f"moved should be about [0, 1], got {moved}"
assert np.allclose(rotation(20) @ rotation(25), rotation(45)), "two rotations compose by adding angles"
print("Exercise 2 passed")

# %% tags=["solution"]
def rotation(deg):
    r = np.radians(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])

moved = rotation(90) @ np.array([1, 0])

# %% [markdown]
# ### Exercise 3: Score houses with a weight vector
# Using `h`, `X` and the same `w`, `bias` from section 6, compute the **mean absolute error** between `X @ w + bias` and the true
# `h["price"]` and store it in `mae`. Then store in `best_area_coef` whichever of `[100.0, 150.0, 200.0]` gives the lowest MAE when used
# as the first weight (keep the other weights and bias unchanged).

# %% tags=["starter"]
mae = ...
best_area_coef = ...

# %%
y = h["price"].to_numpy()
exp_mae = np.mean(np.abs(X @ w + bias - y))
assert abs(mae - exp_mae) < 1e-6, f"MAE should be mean(|pred - actual|); expected about {exp_mae:.1f}"
maes = {c: np.mean(np.abs(X @ np.array([c, 5000.0, 10000.0]) + bias - y)) for c in [100.0, 150.0, 200.0]}
assert best_area_coef == min(maes, key=maes.get), f"MAE by coefficient: { {k: round(v) for k, v in maes.items()} }"
print("Exercise 3 passed:", {k: round(v) for k, v in maes.items()})

# %% tags=["solution"]
mae = float(np.mean(np.abs(X @ w + bias - h["price"].to_numpy())))
cands = [100.0, 150.0, 200.0]
scores = [np.mean(np.abs(X @ np.array([c, 5000.0, 10000.0]) + bias - h["price"].to_numpy())) for c in cands]
best_area_coef = cands[int(np.argmin(scores))]

# %% [markdown]
# ### Exercise 4 (open-ended): Which kind of "similar"?
# Take any two rows of `h` and compute both their Euclidean distance and cosine similarity on `[area_sqft, bedrooms, bathrooms]`.
# Then standardise the three columns (subtract the mean, divide by the std) and recompute both. In a markdown cell, explain which
# number changed the most and why the raw Euclidean distance was almost entirely about one column.

# %% [markdown]
# ## Recap
# - A vector is an arrow and a list of numbers; linear models only ever scale and add vectors.
# - The dot product measures agreement: `a.b = |a||b|cos(theta)`; zero means orthogonal.
# - L2, L1 and cosine answer different questions; scale features before using distances.
# - A matrix moves points; multiplying matrices composes moves and order matters.
# - `X @ w` scores a whole dataset in one vectorised step; always check shapes `(n, d) @ (d,)`.
#
# **Go deeper:** the NumPy documentation section "Linear algebra (numpy.linalg)" (numpy.org/doc); the free "Linear Algebra"
# lectures and "Essence of linear algebra" video series by 3Blue1Brown are widely recommended visual companions.
#
# **Next:** [Matrix decompositions: eigenvectors and SVD](02_matrix_decompositions_eigen_svd.ipynb)
