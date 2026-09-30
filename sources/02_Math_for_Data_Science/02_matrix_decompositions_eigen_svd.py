# %% [markdown]
# # Matrix Decompositions: Eigenvectors, SVD and a PCA Preview
#
# **Why this matters.** Your dataset has 64 (or 6,000) columns, but most of the information lives in a handful of
# directions. Eigenvectors and the SVD find those directions. They power PCA, image and embedding compression,
# recommender systems (matrix factorisation) and the analysis of why some models are numerically unstable.
# In an interview, "what does PCA actually compute?" has a one-line answer once this notebook clicks.
#
# **You will learn**
# - see an eigenvector as "a direction the matrix only stretches, never turns"
# - compute eigen-decompositions with `np.linalg.eigh` and verify `A v = lambda v`
# - use the SVD to build the best low-rank approximation and compress an image
# - connect the two to PCA: principal components are eigenvectors of the covariance matrix
#
# **Prerequisites:** [Vectors and matrices](01_vectors_and_matrices.ipynb).
#
# > Data used: scikit-learn's bundled `load_digits` (1,797 real 8x8 handwritten digit images, no download).

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Eigenvectors: directions that survive a transformation
#
# Most vectors get rotated *and* stretched when you multiply by a matrix. An **eigenvector** `v` of `A` is special:
# `A v = lambda v`. It only gets scaled, by the **eigenvalue** `lambda`. Picture a rubber sheet being stretched:
# the stretch axes stay pointing the same way.

# %%
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])          # symmetric matrix: stretches along the diagonals

eigvals, eigvecs = np.linalg.eigh(A)   # eigh is for symmetric matrices; columns of eigvecs are the eigenvectors
print("eigenvalues :", eigvals)
print("eigenvectors (as columns):\n", eigvecs)

for lam, v in zip(eigvals, eigvecs.T):
    print(f"lambda={lam:.1f}:  A@v = {A @ v},  lambda*v = {lam * v}")

# %% [markdown]
# *What just happened?* `A @ v` equals `lambda * v` for both eigenvectors, so multiplying by `A` scaled one diagonal
# direction by 1 and the other by 3. Now see it: push the unit circle through `A`.

# %%
t = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(t), np.sin(t)])
ellipse = A @ circle

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(*circle, color="tab:blue", label="unit circle")
ax.plot(*ellipse, color="tab:red", label="after A")
for lam, v in zip(eigvals, eigvecs.T):
    ax.plot([-lam * v[0], lam * v[0]], [-lam * v[1], lam * v[1]], "k--", lw=1)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper left")
ax.set_title("Eigenvectors are the axes of the stretched ellipse")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.show()

# %% [markdown]
# The dashed lines are the eigenvector directions and they coincide with the long and short axes of the ellipse; the
# semi-axis lengths are the eigenvalues (3 and 1). Two handy checks that hold for any square matrix:
# the **trace** (sum of diagonal) equals the sum of eigenvalues, and the **determinant** equals their product.

# %%
print("trace:", np.trace(A), " sum of eigenvalues:", eigvals.sum())
print("det  :", np.linalg.det(A).round(6), " product of eigenvalues:", eigvals.prod())

# %% [markdown]
# ## 2. Eigenvectors of a covariance matrix = the shape of your data cloud
#
# The covariance matrix of a dataset describes how features vary together. Its eigenvectors point along the
# directions of greatest and least spread, and the eigenvalues are the variances along them. This is PCA.

# %%
n = 500
z = rng.normal(size=(n, 2))
cloud = z @ np.array([[3.0, 0.0], [0.0, 0.6]])                      # wide and thin
rot = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
cloud = cloud @ rot.T                                               # tilt it

centered = cloud - cloud.mean(axis=0)
cov = centered.T @ centered / (n - 1)
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]                                      # eigh sorts ascending; we want biggest first
vals, vecs = vals[order], vecs[:, order]
print("variances along the principal axes:", vals)
print("first principal axis:", vecs[:, 0])

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(*centered.T, s=8, alpha=0.4)
for lam, v in zip(vals, vecs.T):
    ax.plot([0, 2 * np.sqrt(lam) * v[0]], [0, 2 * np.sqrt(lam) * v[1]], lw=3)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title("Eigenvectors of the covariance matrix = principal axes")
ax.set_xlabel("feature 1 (centred)"); ax.set_ylabel("feature 2 (centred)")
plt.show()

# %% [markdown]
# ## 3. The SVD: every matrix, rectangular or not, splits into rotate - stretch - rotate
#
# Eigen-decomposition needs a square matrix. The **singular value decomposition** works for any `m x n` matrix `M`:
#
# `M = U @ diag(s) @ Vt`
#
# - `Vt` rotates the input space, `s` (singular values, sorted largest first) stretches, `U` rotates into the output space.
# - Keep only the largest `k` singular values and you get the **best possible rank-k approximation** of `M`
#   (a theorem called Eckart-Young; best in the least-squares sense).

# %%
M = rng.normal(size=(6, 4))
U, s, Vt = np.linalg.svd(M, full_matrices=False)
print("shapes:", U.shape, s.shape, Vt.shape)
print("singular values:", s)
print("reconstruction exact?", np.allclose(U @ np.diag(s) @ Vt, M))

# %% [markdown]
# ## 4. Compressing images with a truncated SVD
#
# A grayscale image is a matrix of pixel intensities. If rows are highly correlated, few singular values carry most
# of the picture. We use one digit from `load_digits` (8x8, values 0-16). It is small enough to see every pixel.

# %%
digits = load_digits()
img = digits.images[0]                     # shape (8, 8)
U, s, Vt = np.linalg.svd(img, full_matrices=False)

def rank_k(U, s, Vt, k):
    return (U[:, :k] * s[:k]) @ Vt[:k]

fig, axes = plt.subplots(1, 5, figsize=(12, 2.8))
for ax, k in zip(axes, [1, 2, 3, 5, 8]):
    ax.imshow(rank_k(U, s, Vt, k), cmap="gray_r")
    err = np.linalg.norm(img - rank_k(U, s, Vt, k)) / np.linalg.norm(img)
    ax.set_title(f"rank {k}\nrel. error {err:.2f}"); ax.axis("off")
plt.suptitle("Rank-k SVD reconstructions of one 8x8 digit")
plt.tight_layout(); plt.show()

# %% [markdown]
# *What just happened?* With `k=8` we recover the original exactly (an 8x8 matrix has at most 8 singular values). Fewer
# values give a blurrier but recognisable digit. Storage for rank `k` is `k*(8 + 8 + 1)` numbers instead of `64`, so for a
# tiny 8x8 image compression only pays off at very low rank. For a 1000x1000 photo the same idea saves a lot; the
# principle is what matters here.
#
# A more useful question: across **all 1,797 digits**, how many directions matter?

# %%
X = digits.data                             # (1797, 64): each row is a flattened image
Xc = X - X.mean(axis=0)                     # centre each pixel column
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
energy = s**2 / np.sum(s**2)                # fraction of total variance per direction
cum = np.cumsum(energy)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
ax.axhline(0.9, color="gray", ls="--")
ax.set_xlabel("number of components kept"); ax.set_ylabel("cumulative fraction of variance")
ax.set_title("How many directions carry the digits data?")
ax.grid(alpha=0.3)
plt.show()
k90 = int(np.searchsorted(cum, 0.9) + 1)
print(f"{k90} of {X.shape[1]} directions capture 90% of the variance")

# %% [markdown]
# ## 5. PCA preview: the SVD of centred data *is* PCA
#
# PCA = centre the data, take the SVD, keep the top `k` rows of `Vt` (the **principal components**) and project.
# Variance along component `i` is `s_i^2 / (n-1)`. `sklearn.decomposition.PCA` does exactly this. Verify:

# %%
pca = PCA(n_components=k90, random_state=42).fit(X)
print("sklearn explained variance ratio (first 3):", pca.explained_variance_ratio_[:3])
print("from our SVD                    (first 3):", energy[:3])
print("same components up to sign?",
      np.allclose(np.abs(pca.components_[0]), np.abs(Vt[0])))

X2 = pca.transform(X)[:, :2]
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(X2[:, 0], X2[:, 1], c=digits.target, cmap="tab10", s=8)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("64 pixels squeezed to 2 numbers, coloured by digit")
plt.colorbar(sc, label="digit")
plt.show()

# %% [markdown]
# Look at how much the classes already separate using only 2 of 64 directions (overlap remains: 2 components are far from the whole story). The full treatment of PCA (when to use it, scaling,
# pitfalls) is in the machine learning module; here the point is that it is just linear algebra you can now read.

# %% [markdown]
# ## Common mistakes
#
# 1. **Using `np.linalg.eig` on a symmetric matrix and expecting sorted, real output.** Use `eigh` for symmetric/covariance
#    matrices: it is faster, returns real values, and sorts ascending (so reverse for "largest first").
# 2. **Forgetting to centre the data before SVD/PCA.** Without centring, the first component mostly points at the mean.
# 3. **Expecting eigenvector signs to be stable.** `v` and `-v` are equally valid; compare with `np.abs` or up to sign.
# 4. **Not scaling features.** PCA follows variance, so a column in dollars beats one in years. Standardise first.

# %%
Xtiny = np.array([[100.0, 1.0], [102.0, 3.0], [98.0, 2.0], [101.0, 4.0]])
_, _, Vt_raw = np.linalg.svd(Xtiny, full_matrices=False)
_, _, Vt_cen = np.linalg.svd(Xtiny - Xtiny.mean(axis=0), full_matrices=False)
print("first direction, uncentred:", np.abs(Vt_raw[0]), "(just points at the mean)")
print("first direction, centred  :", np.abs(Vt_cen[0]))

# %% [markdown]
# ## Interview corner
#
# **1. What is an eigenvector, in plain words?**
# <details><summary>Answer</summary>
# A direction that a linear transformation only stretches or flips, never turns. The stretch factor is the eigenvalue.
# </details>
#
# **2. How are PCA, SVD and the covariance matrix related?**
# <details><summary>Answer</summary>
# PCA components are the eigenvectors of the covariance matrix, equivalently the right singular vectors (rows of Vt) of the centred data matrix.
# Eigenvalues equal squared singular values divided by n-1. SVD is the numerically preferred route.
# </details>
#
# **3. Why does a truncated SVD give a good compression?**
# <details><summary>Answer</summary>
# Eckart-Young: keeping the k largest singular values yields the closest rank-k matrix in Frobenius (and spectral) norm. If the singular values decay
# quickly, little is lost. It is lossy, and it is only a good idea when the data really is close to low rank.
# </details>
#
# **4. When would PCA be a bad idea?**
# <details><summary>Answer</summary>
# When structure is non-linear (a spiral), when low-variance directions carry the signal for your target, when features are unscaled, or when you need interpretable original features.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Eigenvalues and the trace/determinant check
# For `B` below (symmetric), store its eigenvalues in ascending order in `ev` (use `np.linalg.eigvalsh`) and store `True` in
# `trace_ok` if the sum of eigenvalues equals `np.trace(B)` (within `1e-9`).

# %%
B = np.array([[4.0, 1.0, 0.0],
              [1.0, 3.0, 1.0],
              [0.0, 1.0, 2.0]])

# %% tags=["starter"]
ev = ...
trace_ok = ...

# %%
w, V = np.linalg.eigh(B)
assert np.allclose(np.sort(ev), w), "ev should be the eigenvalues of B, ascending"
assert list(ev) == sorted(ev), "sort ascending"
assert trace_ok is True or trace_ok == True, "trace_ok should be True (the sum of eigenvalues equals the trace)"
assert np.allclose(B @ V[:, 0], w[0] * V[:, 0]), "sanity: B v = lambda v"
print("Exercise 1 passed:", ev.round(3))

# %% tags=["solution"]
ev = np.linalg.eigvalsh(B)
trace_ok = bool(abs(ev.sum() - np.trace(B)) < 1e-9)

# %% [markdown]
# ### Exercise 2: How many components for 95% of the digits variance?
# Using `s` from section 4 (the singular values of the *centred* digits matrix), compute the smallest number of components `k95`
# whose cumulative variance fraction is at least 0.95. Also store `err_k95`: the relative Frobenius reconstruction error
# `norm(Xc - Xc_k) / norm(Xc)` when keeping exactly `k95` components.
# *Hint: `np.cumsum(s**2) / np.sum(s**2)`, then `np.searchsorted`. Rank-k reconstruction is `(U[:, :k] * s[:k]) @ Vt[:k]`.*

# %% tags=["starter"]
k95 = ...
err_k95 = ...

# %%
p = PCA(n_components=0.95, svd_solver="full").fit(X)
assert k95 == p.n_components_, f"an independent sklearn PCA(0.95) keeps {p.n_components_} components; you got {k95}"
Xk = (U[:, :k95] * s[:k95]) @ Vt[:k95]
exp = np.linalg.norm(Xc - Xk) / np.linalg.norm(Xc)
assert abs(err_k95 - exp) < 1e-9, f"expected relative error about {exp:.4f}"
print(f"Exercise 2 passed: {k95} components, relative error {err_k95:.3f}")

# %% tags=["solution"]
cum_ = np.cumsum(s**2) / np.sum(s**2)
k95 = int(np.searchsorted(cum_, 0.95) + 1)
Xk_ = (U[:, :k95] * s[:k95]) @ Vt[:k95]
err_k95 = float(np.linalg.norm(Xc - Xk_) / np.linalg.norm(Xc))

# %% [markdown]
# ### Exercise 3: Reconstruction error is the discarded singular values
# The Eckart-Young theorem says the squared Frobenius error of the best rank-k approximation equals the sum of the squared discarded
# singular values. For the single digit `img` (section 4) with rank `k = 3`, store the **error computed directly**
# (`np.linalg.norm(img - approx)**2`) in `direct` and the **error from the singular values** in `from_s`. They should match.

# %% tags=["starter"]
Ui, si, Vti = np.linalg.svd(img, full_matrices=False)
direct = ...
from_s = ...

# %%
approx = (Ui[:, :3] * si[:3]) @ Vti[:3]
assert abs(direct - np.linalg.norm(img - approx) ** 2) < 1e-9, "direct = squared Frobenius norm of (img - rank-3 approximation)"
assert abs(from_s - np.sum(si[3:] ** 2)) < 1e-9, "from_s = sum of squares of the singular values beyond the first 3"
assert abs(direct - from_s) < 1e-8, "the two should agree"
print(f"Exercise 3 passed: {direct:.4f} == {from_s:.4f}")

# %% tags=["solution"]
Ui, si, Vti = np.linalg.svd(img, full_matrices=False)
approx_ = (Ui[:, :3] * si[:3]) @ Vti[:3]
direct = float(np.linalg.norm(img - approx_) ** 2)
from_s = float(np.sum(si[3:] ** 2))

# %% [markdown]
# ### Exercise 4 (open-ended): What does a component look like?
# Reshape `Vt[0]` and `Vt[1]` (from section 4) to 8x8 and show them with `plt.imshow`. In a markdown cell, describe in one or two sentences which
# pixel regions each component contrasts, and why those regions would help separate digits.

# %% [markdown]
# ## Recap
# - An eigenvector is a direction that a matrix only scales; the eigenvalue is the scale factor.
# - For a covariance matrix, eigenvectors are the principal axes of the data and eigenvalues the variances along them.
# - The SVD `M = U diag(s) Vt` exists for any matrix; truncating it gives the best low-rank approximation, with error equal to the discarded `s^2`.
# - PCA is the SVD of centred data. Centre, scale, then read off `explained_variance_ratio_`.
#
# **Go deeper:** NumPy docs for `numpy.linalg.svd` and `numpy.linalg.eigh`; the scikit-learn user guide chapter "Decomposing signals in components (PCA)".
#
# **Next:** [Calculus: derivatives and gradients](03_calculus_derivatives_and_gradients.ipynb)
