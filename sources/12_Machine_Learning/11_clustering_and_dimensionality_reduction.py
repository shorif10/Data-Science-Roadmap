# %% [markdown]
# # Clustering and Dimensionality Reduction
#
# **Why this matters.** Marketing asks "what kinds of customers do we have?" and there is no target column to
# predict. Unsupervised learning finds structure without labels: **clustering** groups similar rows, and
# **dimensionality reduction** compresses many columns into a few (for plotting, denoising, or feeding a model).
# The hard part is not running `KMeans`, it is answering "how do I know these clusters are any good?", since
# there is no accuracy to compute. This notebook spends as much time on that question as on the algorithms.
#
# **You will learn**
# - implement k-means from scratch in ten lines and know exactly what `sklearn` does
# - choose `k` with the elbow and silhouette, and see when both fail
# - use DBSCAN and hierarchical clustering where k-means breaks
# - use PCA and read explained variance, and know why scaling comes first
# - build an RFM customer segmentation from raw shop tables and judge whether it is useful
#
# **Prerequisites:** [Feature engineering and pipelines](08_feature_engineering_and_pipelines.ipynb) (scaling), and the
# pandas `groupby`/`merge` notebooks.
#
# > Data used: `datasets/orders.csv`, `order_items.csv`, `products.csv`, `customers.csv` (synthetic shop; see
# > `docs/DATASETS.md`), `datasets/churn.csv`, and the bundled `sklearn` digits images (real 8x8 handwritten digits).

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = Path("../datasets")

# %% [markdown]
# ## 1. k-means: two steps, repeated
#
# Pick `k`. Then alternate: (1) **assign** every point to its nearest centre; (2) **move** each centre to the mean of its
# points. Stop when nothing changes. That is all of it. Here it is in NumPy, on blobs with four known groups:

# %%
X_blobs, y_blobs = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)


def lloyd(X, centers, max_iter=100):
    for _ in range(max_iter):
        dist2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n_points, k)
        labels = dist2.argmin(axis=1)  # step 1: assign
        new = np.array([X[labels == j].mean(axis=0) for j in range(len(centers))])  # step 2: move
        if np.allclose(new, centers):
            break
        centers = new
    return labels, centers


init = X_blobs[[0, 1, 2, 3]]  # a deliberately naive start: the first four rows
labels_mine, centers_mine = lloyd(X_blobs, init.copy())
km_check = KMeans(n_clusters=4, init=init, n_init=1, algorithm="lloyd").fit(X_blobs)
print("same centres as sklearn from the same start:", np.allclose(np.sort(centers_mine, axis=0), np.sort(km_check.cluster_centers_, axis=0), atol=1e-3))
inertia = ((X_blobs - centers_mine[labels_mine]) ** 2).sum()
print(f"inertia (sum of squared distances to own centre): {inertia:.1f}")

# %% [markdown]
# *What just happened?* k-means minimises **inertia**, the total squared distance from points to their centre. Two
# consequences you must remember: it prefers **round, similarly-sized blobs** (it uses Euclidean distance to a mean), and the result
# depends on the **starting centres** (it finds a local optimum). `sklearn` protects you with smart initialisation
# (`k-means++`) and `n_init` restarts, keeping the best run.

# %% [markdown]
# ## 2. Choosing k: elbow and silhouette
#
# Inertia always falls as `k` grows (with `k = n` it is zero), so we look for the **elbow** where extra clusters stop paying off.
# The **silhouette** score is more direct: for each point compare *a* = mean distance to its own cluster with *b* = mean distance
# to the nearest other cluster; `(b - a) / max(a, b)` is near 1 for well separated clusters, about 0 when clusters overlap
# and negative for points that look misassigned. The average over points scores the whole clustering.

# %%
rows = []
for k in range(1, 9):
    km = KMeans(k, n_init=10, random_state=42).fit(X_blobs)
    rows.append((k, km.inertia_, silhouette_score(X_blobs, km.labels_) if k > 1 else np.nan))
sel = pd.DataFrame(rows, columns=["k", "inertia", "silhouette"]).set_index("k")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(sel.index, sel["inertia"], "o-")
axes[0].set_xlabel("k (number of clusters)")
axes[0].set_ylabel("inertia (squared distance units)")
axes[0].set_title("Elbow")
axes[1].plot(sel.index, sel["silhouette"], "o-", color="tab:green")
axes[1].set_xlabel("k (number of clusters)")
axes[1].set_ylabel("mean silhouette")
axes[1].set_title("Silhouette")
plt.tight_layout()
plt.show()
print("best k by silhouette:", int(sel["silhouette"].idxmax()), "(true number of blobs is 4)")

# %% [markdown]
# On clean blobs both tools agree. Real data is rarely this kind: **we will see below that on real customers there is no clear
# elbow and the silhouette just declines**. That is not a bug in the tool, it is information about the data.

# %% [markdown]
# ## 3. Where k-means breaks, and DBSCAN
#
# Two interleaved half-moons are obviously two groups to a human. k-means cuts them with a straight line, because a mean-based
# round cluster is all it can express. **DBSCAN** defines clusters as dense regions: a point is *core* if at least
# `min_samples` points lie within radius `eps`; clusters are chains of core points; anything not reachable is **noise (label -1)**.
# No `k` needed, arbitrary shapes allowed, outliers detected for free.

# %%
X_moons, y_moons = make_moons(n_samples=500, noise=0.07, random_state=42)
lab_km = KMeans(2, n_init=10, random_state=42).fit_predict(X_moons)
lab_db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X_moons)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, lab, title in zip(axes, [lab_km, lab_db], ["k-means (k=2)", "DBSCAN (eps=0.2)"]):
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=lab, cmap="coolwarm", s=10)
    ax.set_title(title)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
plt.tight_layout()
plt.show()
print("agreement with the true moons (adjusted Rand index, 1 = perfect, ~0 = random):")
print(f"  k-means {adjusted_rand_score(y_moons, lab_km):.2f}   DBSCAN {adjusted_rand_score(y_moons, lab_db):.2f}")

# %% [markdown]
# *What just happened?* The **adjusted Rand index (ARI)** compares two labelings ignoring cluster numbering. We may use it here only because we
# *know* the true moons; on real unlabeled data you cannot. DBSCAN's weak spot is its parameters: `eps` is a distance, so it is
# scale-dependent, and one `eps` cannot fit clusters of very different density. Watch how sensitive it is:

# %%
rows = []
for eps in [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X_moons)
    rows.append((eps, len(set(lab) - {-1}), (lab == -1).mean(), adjusted_rand_score(y_moons, lab)))
pd.DataFrame(rows, columns=["eps", "n_clusters", "noise_share", "ARI"]).set_index("eps").round(3)

# %% [markdown]
# Too small an `eps` shatters the data into many clusters and calls most points noise; too large merges everything.
# A common heuristic is to plot each point's distance to its k-th nearest neighbour and pick `eps` near the "knee".

# %% [markdown]
# ## 4. Hierarchical clustering: a family tree of points
#
# Start with every point alone and repeatedly **merge the two closest clusters**. With **Ward linkage**, "closest" means "merging them
# increases within-cluster variance the least". The result is a tree (a **dendrogram**); cutting it at a height gives clusters, so you
# can look at the structure before committing to `k`. It needs pairwise distances, so it is for thousands of points, not millions.

# %%
sample = X_blobs[np.random.default_rng(42).choice(len(X_blobs), 40, replace=False)]
Z_link = linkage(sample, method="ward")
fig, ax = plt.subplots(figsize=(9, 3.5))
dendrogram(Z_link, ax=ax, no_labels=True, color_threshold=0.4 * Z_link[:, 2].max())
ax.set_xlabel("40 sampled points")
ax.set_ylabel("Ward merge distance")
ax.set_title("Dendrogram: tall vertical gaps suggest natural numbers of clusters")
plt.show()

# %% [markdown]
# Long vertical lines mean "these clusters were far apart when merged": cut just below the longest gap. `AgglomerativeClustering`
# gives flat labels from the same idea, which we will use on digits shortly.

# %% [markdown]
# ## 5. PCA: fewer columns, most of the information
#
# **Principal Component Analysis** finds new axes (rotations of your features) ordered by how much variance they capture. Keep
# the first few and you have a compressed version of the data. Uses: visualise in 2-D, drop redundancy, denoise, speed up models.
# Digits are the classic playground: 1,797 images of 8x8 pixels = **64 features** each.

# %%
digits = load_digits()
X_dig, y_dig = digits.data, digits.target
pca_full = PCA().fit(X_dig)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n90 = int(np.searchsorted(cum, 0.90) + 1)
n95 = int(np.searchsorted(cum, 0.95) + 1)
print(f"64 pixels -> {n90} components keep 90% of the variance, {n95} keep 95%")

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(range(1, 65), cum, "o-", markersize=3)
ax.axhline(0.95, color="grey", linestyle=":")
ax.set_xlabel("number of principal components")
ax.set_ylabel("cumulative explained variance ratio")
ax.set_title("Digits: how many components do we need?")
plt.show()

# %% [markdown]
# *What just happened?* `explained_variance_ratio_` says what share of total variance each component carries. The cumulative curve
# tells you the price of compression. Now use two components as a map of the data (colours are the true digits, used **only** to colour
# the plot; PCA never saw them):

# %%
X2 = PCA(n_components=2).fit_transform(X_dig)
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(X2[:, 0], X2[:, 1], c=y_dig, cmap="tab10", s=8)
ax.set_xlabel("principal component 1")
ax.set_ylabel("principal component 2")
ax.set_title("Digits in two dimensions (colour = true digit)")
fig.colorbar(sc, ax=ax, label="digit")
plt.show()

# %% [markdown]
# Some digits separate, others overlap heavily: two components cannot hold 64 dimensions of information. **Reconstructing**
# images from `k` components shows what is lost:

# %%
fig, axes = plt.subplots(2, 4, figsize=(8, 4.4))
for row, idx in enumerate([0, 5]):
    axes[row, 0].imshow(X_dig[idx].reshape(8, 8), cmap="gray_r")
    axes[row, 0].set_title("original")
    for col, k in enumerate([2, 10, 30], start=1):
        p = PCA(n_components=k).fit(X_dig)
        rec = p.inverse_transform(p.transform(X_dig[[idx]]))[0]
        axes[row, col].imshow(rec.reshape(8, 8), cmap="gray_r")
        axes[row, col].set_title(f"{k} components")
for ax in axes.ravel():
    ax.axis("off")
plt.suptitle("Reconstruction from k principal components")
plt.show()

# %% [markdown]
# ### PCA needs scaling (usually)
# PCA chases variance, and variance depends on units. On the churn table, unscaled PCA is basically a picture of the biggest-unit column:

# %%
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])  # "None" is a real category, not missing
num = churn[["tenure_months", "monthly_charges", "total_charges", "support_calls"]].dropna()
raw_ratio = PCA().fit(num).explained_variance_ratio_
scaled_ratio = PCA().fit(StandardScaler().fit_transform(num)).explained_variance_ratio_
print("first component's variance share  unscaled:", round(raw_ratio[0], 3), "  scaled:", round(scaled_ratio[0], 3))

# %% [markdown]
# Unscaled, the first component is "total_charges" in disguise (dollars in the thousands versus calls in single digits). Scaled, the
# variance is shared out by genuine correlation (tenure and total charges move together).
#
# ### Does compressing help clustering?
# Compare k-means on all 64 pixels with k-means on the first 2 and the first 20 components, using the true digits **only as a referee**
# (we can do that because this is a teaching data set).

# %%
rows = []
for name, feats in [("all 64 pixels", X_dig), ("PCA 2", PCA(2).fit_transform(X_dig)), ("PCA 20", PCA(20).fit_transform(X_dig))]:
    lab = KMeans(10, n_init=10, random_state=42).fit_predict(feats)
    rows.append((name, adjusted_rand_score(y_dig, lab)))
rows.append(("Ward hierarchical, 64 pixels", adjusted_rand_score(y_dig, AgglomerativeClustering(10).fit_predict(X_dig))))
pd.DataFrame(rows, columns=["method", "ARI vs true digits"]).set_index("method").round(3)

# %% [markdown]
# Two components throw away too much; 20 components are about as good as all 64 while being much smaller, and a different algorithm
# (Ward) can beat k-means on the same features. There is no universally best method. (For nonlinear 2-D maps sklearn also has
# `sklearn.manifold.TSNE`; it is good for pictures but its distances and cluster sizes are not meaningful, so we stay with PCA here.)

# %% [markdown]
# ## 6. Customer segmentation with RFM
#
# The most common business clustering is **RFM**: per customer, **R**ecency (days since last order), **F**requency (number of orders),
# **M**onetary (total spend). We build it from the raw shop tables. Choices to make explicitly: only completed orders count
# (`delivered`, `shipped`), and "today" is the day after the last order in the data.

# %%
orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
items = pd.read_csv(DATA / "order_items.csv")
products = pd.read_csv(DATA / "products.csv")
customers = pd.read_csv(DATA / "customers.csv")

items = items.merge(products[["product_id", "unit_price"]], on="product_id")
items["line_revenue"] = items["quantity"] * items["unit_price"]
order_rev = items.groupby("order_id")["line_revenue"].sum().rename("revenue")

completed = orders[orders["status"].isin(["delivered", "shipped"])].join(order_rev, on="order_id")
today = orders["order_date"].max() + pd.Timedelta(days=1)
rfm = completed.groupby("customer_id").agg(
    recency=("order_date", lambda s: (today - s.max()).days),
    frequency=("order_id", "count"),
    monetary=("revenue", "sum"),
)
print(f"{len(rfm)} of {len(customers)} customers have at least one completed order (the rest have no RFM row)")
rfm.describe().round(1)

# %%
print("skewness:", rfm.skew().round(2).to_dict())
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
for ax, col in zip(axes, rfm.columns):
    ax.hist(rfm[col], bins=30)
    ax.set_xlabel(col + (" (days)" if col == "recency" else " (orders)" if col == "frequency" else " (currency units)"))
    ax.set_ylabel("customers")
plt.suptitle("RFM features are right-skewed")
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* Spend and order count are right-skewed: a few big customers stretch the scale. k-means uses squared
# distances, so those outliers would pull centres toward themselves. The usual fix is `log1p` (log of 1+x) and then standardising.

# %%
Z_rfm = StandardScaler().fit_transform(np.log1p(rfm))

# %% [markdown]
# ### Choosing k on real data, and a noise baseline
# A silhouette of 0.3 sounds meh, but compared to what? Calibrate by scoring k-means on **structureless random data of the same shape**:

# %%
rng = np.random.default_rng(42)
noise = rng.normal(size=Z_rfm.shape)
rows = []
for k in range(2, 9):
    km = KMeans(k, n_init=10, random_state=42).fit(Z_rfm)
    kn = KMeans(k, n_init=10, random_state=42).fit(noise)
    rows.append((k, km.inertia_, silhouette_score(Z_rfm, km.labels_), silhouette_score(noise, kn.labels_)))
rfm_sel = pd.DataFrame(rows, columns=["k", "inertia", "silhouette_rfm", "silhouette_random_noise"]).set_index("k")
rfm_sel.round(3)

# %% [markdown]
# Read this carefully. There is **no sharp elbow**, and the silhouette broadly declines as `k` grows (two groups are the best-separated cut).
# Random noise still scores a silhouette clearly above zero, so **a "decent" silhouette is not proof of real clusters**; the meaningful
# number is the *gap* over noise. The picture is one of a continuum that we choose to cut into useful pieces, not of natural islands. So `k` is a
# business decision: we take `k=4` because four segments is a number a marketing team can act on, and we then test whether the segments are useful.

# %%
km4 = KMeans(4, n_init=10, random_state=42).fit(Z_rfm)
rfm["segment"] = km4.labels_
profile = rfm.groupby("segment").agg(
    customers=("recency", "size"),
    median_recency_days=("recency", "median"),
    median_orders=("frequency", "median"),
    median_spend=("monetary", "median"),
).round(1)
profile

# %% [markdown]
# ## 7. How to judge unsupervised results (there is no accuracy)
#
# 1. **Interpretability**: can you describe each cluster in a sentence from its profile above ("many orders, high spend, ordered recently")? If not, the clustering is not useful.
# 2. **Stability**: re-run with other seeds or on resampled data. If assignments change wildly, the structure is not reliable.
# 3. **External validation**: do the segments differ on variables you did **not** cluster on (age, country, return rate, later churn)? That is the closest thing to a test set.
# 4. **Separation vs noise**: silhouette *relative to a random baseline*, not in isolation.
# 5. **Actionability**: would the business treat the segments differently? If the answer is no, you do not need the segmentation.
#
# Let us apply 2 and 3.

# %%
labels_by_seed = [KMeans(4, n_init=1, random_state=s).fit_predict(Z_rfm) for s in range(6)]
ari = [adjusted_rand_score(labels_by_seed[0], other) for other in labels_by_seed[1:]]
print("stability: ARI of seed 0 vs seeds 1-5 =", np.round(ari, 2))

# a bootstrap version: fit on a random 80% of customers, label the other 20%, compare with the full-data labels
idx = rng.permutation(len(Z_rfm))
train_i, hold_i = idx[: int(0.8 * len(idx))], idx[int(0.8 * len(idx)):]
km_part = KMeans(4, n_init=10, random_state=42).fit(Z_rfm[train_i])
print("held-out 20% agreement with full-data segmentation (ARI):",
      round(adjusted_rand_score(km4.labels_[hold_i], km_part.predict(Z_rfm[hold_i])), 2))

# %%
ret = orders.assign(is_returned=(orders["status"] == "returned")).groupby("customer_id")["is_returned"].mean().rename("return_rate")
validation = rfm.join(customers.set_index("customer_id")[["age", "is_premium"]]).join(ret)
validation.groupby("segment")[["age", "is_premium", "return_rate"]].mean().round(3)

# %% [markdown]
# *What just happened?* The segments are stable across seeds and hold up on a held-out slice (that is what makes them trustworthy), and we checked them
# against columns that never entered the clustering. Be careful with interpretation: `is_premium` is linked to order frequency by how this
# synthetic shop was generated, so it validates the *pipeline* more than it discovers something new. Age and return rate are
# the more independent checks, and in the table above they barely differ between segments (age is almost flat, return rate moves only
# a little). The honest conclusion: these segments describe *purchase behaviour*, not demographics. A segmentation that differs only on the
# variables you clustered on is a description, not a discovery, which is fine as long as you do not oversell it.

# %% [markdown]
# ## Common mistakes
#
# 1. **Not scaling** before k-means/DBSCAN/PCA: the biggest-unit column decides everything.
# 2. **Clustering skewed features raw** (money, counts): use `log1p` first.
# 3. **Treating a positive silhouette as proof** that clusters exist; compare with random data of the same shape.
# 4. **Choosing `k` from the elbow alone** when there is no elbow.
# 5. **Fitting PCA on train+test** (or before a split) when the components feed a supervised model: fit inside the `Pipeline` on training data only.
# 6. **Interpreting t-SNE/PCA pictures as distances**: PCA axes are rotations, and t-SNE distances between islands are not meaningful.

# %%
# Mistake 3 in one line: silhouette of k-means on pure noise, k=3
print("silhouette on structureless data:", round(silhouette_score(noise, KMeans(3, n_init=10, random_state=42).fit_predict(noise)), 3))

# %% [markdown]
# ## Interview corner
#
# **1. How do you choose the number of clusters?**
# <details><summary>Answer</summary>
# Combine the elbow (inertia), the silhouette (compared with a random baseline), stability across resamples, and above all usefulness for
# the business question. When the data is a continuum there is no true k, so say so and justify a practical choice.
# </details>
#
# **2. When does k-means fail and what do you use instead?**
# <details><summary>Answer</summary>
# Non-spherical or unequal-density clusters, outliers dragging centres, very different cluster sizes. DBSCAN (arbitrary shapes, noise labelling),
# hierarchical clustering (a tree to inspect), or Gaussian mixtures (elliptical clusters, soft assignments) are common alternatives.
# </details>
#
# **3. What is PCA and what does "explained variance" mean?**
# <details><summary>Answer</summary>
# PCA rotates the data onto orthogonal axes ordered by variance. The explained variance ratio of a component is its share of total variance,
# so the cumulative sum tells you how much information a k-dimensional summary keeps. Scale features first, since PCA maximises variance and variance depends on units.
# </details>
#
# **4. How do you validate a clustering without labels?**
# <details><summary>Answer</summary>
# Internal measures (silhouette against a noise baseline), stability under resampling and different seeds, profiling clusters for
# interpretability, and external validation on variables not used for clustering (or on later outcomes). Ultimately, whether the business acts differently.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Silhouette-best k
# For `X_blobs`, fit `KMeans(k, n_init=10, random_state=42)` for `k` from 2 to 8 and store the `k` with the highest silhouette in
# `best_k`, plus its silhouette in `best_sil`.
# *Hint: loop, collect scores in a dict, `max(d, key=d.get)`.*

# %% tags=["starter"]
best_k = ...
best_sil = ...

# %%
_s = {k: silhouette_score(X_blobs, KMeans(k, n_init=10, random_state=42).fit_predict(X_blobs)) for k in range(2, 9)}
assert best_k == max(_s, key=_s.get), f"expected k={max(_s, key=_s.get)}, got {best_k}"
assert abs(best_sil - _s[best_k]) < 1e-9, "best_sil should be the silhouette of best_k"
print(f"Exercise 1 passed (k={best_k}, silhouette={best_sil:.3f})")

# %% tags=["solution"]
scores = {}
for k in range(2, 9):
    scores[k] = silhouette_score(X_blobs, KMeans(k, n_init=10, random_state=42).fit_predict(X_blobs))
best_k = max(scores, key=scores.get)
best_sil = scores[best_k]

# %% [markdown]
# ### Exercise 2: How many components?
# On the scaled churn numeric table `Z_churn` (defined below), how many principal components are needed to keep **at least 90%** of the total
# variance? Store the count in `n_components_90`.
# *Hint: `np.cumsum(pca.explained_variance_ratio_)` and `np.searchsorted`, or a loop.*

# %%
Z_churn = StandardScaler().fit_transform(num)

# %% tags=["starter"]
n_components_90 = ...

# %%
_cum = np.cumsum(PCA().fit(Z_churn).explained_variance_ratio_)
_exp = int(np.argmax(_cum >= 0.90) + 1)
assert n_components_90 == _exp, f"expected {_exp} components (cumulative variance: {np.round(_cum, 3)}), got {n_components_90}"
print(f"Exercise 2 passed ({n_components_90} of {Z_churn.shape[1]} components)")

# %% tags=["solution"]
cum_var = np.cumsum(PCA().fit(Z_churn).explained_variance_ratio_)
n_components_90 = int(np.argmax(cum_var >= 0.90) + 1)

# %% [markdown]
# ### Exercise 3: Rescue the moons with DBSCAN
# Choose an `eps` from `[0.05, 0.1, 0.2, 0.4, 0.8, 1.5]` (with `min_samples=5`) so that DBSCAN on `X_moons` recovers the true moons:
# store it in `good_eps`, and store the number of clusters it finds (not counting noise) in `n_found`. The check requires an ARI of at least 0.9.
# *Hint: reread the eps sweep table above.*

# %% tags=["starter"]
good_eps = ...
n_found = ...

# %%
_lab = DBSCAN(eps=good_eps, min_samples=5).fit_predict(X_moons)
assert adjusted_rand_score(y_moons, _lab) >= 0.9, f"eps={good_eps} gives ARI {adjusted_rand_score(y_moons, _lab):.2f}; need >= 0.9"
assert n_found == len(set(_lab) - {-1}), "n_found = number of distinct labels excluding -1 (noise)"
print(f"Exercise 3 passed (eps={good_eps}, clusters={n_found})")

# %% tags=["solution"]
good_eps = 0.2
n_found = len(set(DBSCAN(eps=good_eps, min_samples=5).fit_predict(X_moons)) - {-1})

# %% [markdown]
# ### Exercise 4: Does skew hurt k-means?
# Repeat the RFM segmentation with `k=4` **without** the log transform (just `StandardScaler` on `rfm[["recency", "frequency", "monetary"]]`,
# `KMeans(4, n_init=10, random_state=42)`). Store the share of customers in the **largest** segment in `biggest_share_raw` and the same share for the
# log version (`Z_rfm`) in `biggest_share_log`. Then set `raw_more_lopsided` to `True` if the raw version's largest segment is bigger.
# *Hint: `np.bincount(labels).max() / len(labels)`.*

# %% tags=["starter"]
biggest_share_raw = ...
biggest_share_log = ...
raw_more_lopsided = ...

# %%
_raw_z = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
_lr = KMeans(4, n_init=10, random_state=42).fit_predict(_raw_z)
_ll = KMeans(4, n_init=10, random_state=42).fit_predict(Z_rfm)
_a, _b = np.bincount(_lr).max() / len(_lr), np.bincount(_ll).max() / len(_ll)
assert abs(biggest_share_raw - _a) < 1e-9, f"raw version: expected {_a:.3f}"
assert abs(biggest_share_log - _b) < 1e-9, f"log version: expected {_b:.3f}"
assert raw_more_lopsided == (_a > _b), "compare the two shares"
print(f"Exercise 4 passed (raw {biggest_share_raw:.1%} vs log {biggest_share_log:.1%}; raw more lopsided: {raw_more_lopsided})")

# %% tags=["solution"]
raw_z = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
lab_raw = KMeans(4, n_init=10, random_state=42).fit_predict(raw_z)
lab_log = KMeans(4, n_init=10, random_state=42).fit_predict(Z_rfm)
biggest_share_raw = np.bincount(lab_raw).max() / len(lab_raw)
biggest_share_log = np.bincount(lab_log).max() / len(lab_log)
raw_more_lopsided = bool(biggest_share_raw > biggest_share_log)

# %% [markdown]
# ### Exercise 5 (open-ended): Name the segments and pick an action
# Look at the `profile` table and the validation table. Give each of the four segments a business name (for example "Champions",
# "At risk") and write one concrete action for each (a discount, a win-back email, do nothing). Then say what you would check
# next before trusting the segmentation in production (stability over time, a holdout campaign, additional features).

# %% [markdown]
# ## Recap
# - k-means alternates "assign to nearest centre" and "move centres to means"; it minimises inertia and likes round, similar-sized blobs.
# - Elbow and silhouette guide `k`, but on real, continuous data they may show no clear answer; compare the silhouette with a random-data baseline.
# - DBSCAN finds arbitrary shapes and noise but needs a sensible `eps`; hierarchical clustering gives a tree you can cut anywhere.
# - PCA compresses by variance: scale first, read explained variance, and remember it is a rotation, not a magic map.
# - With no labels, judge clusters by interpretability, stability, external validation and actionability.
#
# **Go deeper:** the scikit-learn user guide chapters "Clustering" and "Decomposing signals in components (matrix factorization
# problems)", and the "Selecting the number of clusters with silhouette analysis" example (scikit-learn.org/stable/).
#
# **Next:** [Imbalanced data and decision thresholds](12_imbalanced_data_and_decision_thresholds.ipynb)
