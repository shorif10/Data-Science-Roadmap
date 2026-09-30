# %% [markdown]
# # k-NN, Support Vector Machines and Naive Bayes
#
# **Why this matters.** Gradient boosting wins most tabular benchmarks, so why learn three "older" models?
# Because each one teaches a different way of thinking (distance, margins, probabilities from counts), each has a
# niche where it is still the pragmatic choice (Naive Bayes as a 10-line text baseline, SVMs on small
# high-dimensional data, k-NN for "find similar items"), and interviewers love asking *why* they break.
# We will not just describe their strengths and weaknesses; we will **measure** them.
#
# **You will learn**
# - explain why distance-based models need scaling, and watch irrelevant features wreck k-NN
# - read an SVM through margins, support vectors, `C` and the kernel trick, and see kernels rescue non-linear data
# - build Naive Bayes for text from counts, and see why its probabilities cannot be trusted at face value
# - back every claim in a strengths/weaknesses table with a mini-experiment
#
# **Prerequisites:** [Logistic regression and metrics](04_logistic_regression_and_classification_metrics.ipynb) and
# [Cross-validation](07_cross_validation_and_model_evaluation.ipynb).
#
# > Data used: `datasets/churn.csv` and `datasets/reviews.csv` (synthetic; see `docs/DATASETS.md`) plus small
# > generated 2-D data sets for pictures.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA = Path("../datasets")
churn = pd.read_csv(DATA / "churn.csv", keep_default_na=False, na_values=[""])  # "None" is a real category, not missing
num_cols = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
X = churn[num_cols]
y = churn["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

evidence = {}  # we will collect one measured number per claim and print them together at the end
print(X_train.shape, f"churn rate {y_train.mean():.1%}")

# %% [markdown]
# ## 1. k-NN: "you are like your neighbours"
#
# To classify a new customer, find the `k` most similar training customers and let them vote. There is **no
# training step** (the model just stores the data), and "similar" means **small distance**. That single design
# choice is both the charm and the weakness: everything depends on how distance is measured.
#
# Look at two customers and see which feature dominates the ordinary (Euclidean) distance:

# %%
a, b = X_train.iloc[0], X_train.iloc[1]
a, b = a.fillna(X_train.median()), b.fillna(X_train.median())
sq = (a - b) ** 2
contrib = pd.DataFrame({"customer_A": a, "customer_B": b, "share_of_squared_distance": sq / sq.sum()}).round(3)
contrib

# %% [markdown]
# *What just happened?* Distance is a sum of squared differences per feature. `total_charges` is measured in
# thousands of dollars while `senior_citizen` is 0 or 1, so the big-unit feature swamps everything else. The model
# does not "know" `total_charges` is more important; it just has bigger numbers. **Rule: any distance-based model
# needs features on comparable scales** (standardise them).

# %%
ks = [1, 5, 15, 51, 101]
rows = []
for k in ks:
    raw = make_pipeline(SimpleImputer(strategy="median"), KNeighborsClassifier(k))
    scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(k))
    rows.append((k, cross_val_score(raw, X_train, y_train, cv=5, scoring="roc_auc").mean(),
                 cross_val_score(scaled, X_train, y_train, cv=5, scoring="roc_auc").mean()))
knn_table = pd.DataFrame(rows, columns=["k", "auc_unscaled", "auc_scaled"]).set_index("k")
evidence["k-NN: scaling helps (AUC, k=51)"] = (knn_table.loc[51, "auc_unscaled"], knn_table.loc[51, "auc_scaled"])
knn_table.round(4)

# %%
fig, ax = plt.subplots(figsize=(6, 4))
knn_table.plot(ax=ax, marker="o")
ax.set_xscale("log")
ax.set_xlabel("k (number of neighbours, log scale)")
ax.set_ylabel("5-fold CV ROC AUC")
ax.set_title("k-NN on churn: scaling and the choice of k")
plt.show()

# %% [markdown]
# *What just happened?* Two lessons in one table. (1) Scaling changes the score, sometimes a lot, sometimes little
# (which features carry the signal decides that; here look at how the columns compare). (2) **`k` is the bias-variance
# dial**: `k=1` memorises noise (jagged, high variance), large `k` smooths toward the majority class (high bias).
# Pick `k` by cross-validation, and prefer odd `k` for two-class votes to avoid ties.

# %% [markdown]
# ### The curse of dimensionality, measured
# k-NN treats every feature as equally relevant. Add columns of pure noise and neighbours stop being genuinely
# similar, because random coordinates dominate the distance.

# %%
rng = np.random.default_rng(42)
rows = []
for n_noise in [0, 5, 20, 50]:
    Xn = np.hstack([X_train.to_numpy(), rng.normal(size=(len(X_train), n_noise))])
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(15))
    rows.append((n_noise, cross_val_score(model, Xn, y_train, cv=5, scoring="roc_auc").mean()))
noise_table = pd.DataFrame(rows, columns=["noise_features_added", "cv_auc"]).set_index("noise_features_added")
evidence["k-NN: irrelevant features hurt (AUC, 0 vs 50 noise cols)"] = (noise_table.iloc[0, 0], noise_table.iloc[-1, 0])
noise_table.round(4)

# %% [markdown]
# *What just happened?* The five real features never changed, yet the score fell as we added noise columns. A
# regularised linear model or a tree can down-weight or ignore useless columns; plain k-NN cannot. Feature selection
# matters far more for k-NN than for most models.
#
# The other practical catch is **prediction cost**: all the work is deferred to query time, when each new point is
# compared with every stored point. We can time it by repeating the training set to make it bigger (brute-force search, 500 queries):

# %%
base = SimpleImputer(strategy="median").fit_transform(X_train)
queries = base[:500]
timings = {}
for reps in [1, 10, 100]:
    big = np.tile(base, (reps, 1))
    labels = np.tile(y_train.to_numpy(), reps)
    knn = KNeighborsClassifier(15, algorithm="brute").fit(big, labels)
    t0 = time.perf_counter()
    knn.predict(queries)
    timings[len(big)] = time.perf_counter() - t0
predict_time = pd.Series(timings, name="seconds to predict 500 points").rename_axis("stored training rows")
evidence["k-NN: predict time grows with stored rows (s, smallest vs largest)"] = (predict_time.iloc[0], predict_time.iloc[-1])
predict_time.round(4)

# %% [markdown]
# Prediction time grows with the number of stored rows. (Tree-based indexes and approximate-neighbour libraries reduce
# this; that is how "find similar items" systems work at scale.)

# %% [markdown]
# ## 2. Support Vector Machines: the widest street
#
# Among all lines that separate two classes, an SVM picks the one with the **widest margin** (the empty street
# between the classes). Only the points touching the street, the **support vectors**, matter; delete any other
# point and the boundary does not move. Let us draw it.

# %%
Xb, yb = make_blobs(n_samples=80, centers=[(-1.5, -1.0), (1.5, 1.0)], cluster_std=1.1, random_state=42)
svm_lin = SVC(kernel="linear", C=1.0).fit(Xb, yb)

fig, ax = plt.subplots(figsize=(6, 4.5))
DecisionBoundaryDisplay.from_estimator(svm_lin, Xb, ax=ax, response_method="decision_function",
                                       plot_method="contour", levels=[-1, 0, 1], colors="k",
                                       linestyles=["--", "-", "--"])
ax.scatter(Xb[:, 0], Xb[:, 1], c=yb, cmap="coolwarm", s=25)
ax.scatter(*svm_lin.support_vectors_.T, s=110, facecolors="none", edgecolors="green", label="support vectors")
ax.set_xlabel("feature 1 (arbitrary units)")
ax.set_ylabel("feature 2 (arbitrary units)")
ax.set_title("Linear SVM: solid line = boundary, dashed = margin edges")
ax.legend()
plt.show()
print("support vectors:", svm_lin.n_support_.sum(), "of", len(Xb), "points")

# %% [markdown]
# ### `C`: how much are margin violations punished?
# Real classes overlap, so a "hard" margin is impossible. `C` sets the price of a point on the wrong side of the
# street: **small `C` = tolerate violations, wide street, simpler model; large `C` = punish violations, narrow street,
# closer fit to the training points.** We can measure the street width (`2 / ||w||`).

# %%
rows = []
for C in [0.01, 0.1, 1, 100]:
    m = SVC(kernel="linear", C=C).fit(Xb, yb)
    rows.append((C, 2 / np.linalg.norm(m.coef_), m.n_support_.sum(), m.score(Xb, yb)))
c_table = pd.DataFrame(rows, columns=["C", "margin_width", "n_support_vectors", "train_accuracy"]).set_index("C")
c_table.round(3)

# %% [markdown]
# *What just happened?* As `C` grows, the street narrows and fewer points are allowed inside it. `C` plays
# the role of "inverse regularisation" (like `C` in logistic regression): tune it by cross-validation.

# %% [markdown]
# ### The kernel trick: curved boundaries without new columns
# A linear boundary cannot separate a ring around a blob. A **kernel** lets the SVM behave as if it had mapped
# the data into a much richer space where a straight cut works, without ever building that space. The popular
# **RBF kernel** measures similarity as `exp(-gamma * distance^2)`: `gamma` sets how far a point's influence reaches.

# %%
Xc, yc = make_circles(n_samples=600, noise=0.1, factor=0.4, random_state=42)
Xm, ym = make_moons(n_samples=600, noise=0.25, random_state=42)
rows = []
for name, (Xd, yd) in {"circles": (Xc, yc), "moons": (Xm, ym)}.items():
    for kernel in ["linear", "poly", "rbf"]:
        rows.append((name, kernel, cross_val_score(SVC(kernel=kernel), Xd, yd, cv=5).mean()))
kern_table = pd.DataFrame(rows, columns=["data", "kernel", "cv_accuracy"]).pivot(index="kernel", columns="data", values="cv_accuracy")
evidence["SVM: RBF beats linear on circles (accuracy)"] = (kern_table.loc["linear", "circles"], kern_table.loc["rbf", "circles"])
kern_table.round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (title, model) in zip(axes, {"linear": SVC(kernel="linear"),
                                     "rbf, gamma=1 (default-ish)": SVC(kernel="rbf", gamma=1),
                                     "rbf, gamma=100 (too local)": SVC(kernel="rbf", gamma=100)}.items()):
    model.fit(Xc, yc)
    DecisionBoundaryDisplay.from_estimator(model, Xc, ax=ax, response_method="predict", alpha=0.25, cmap="coolwarm")
    ax.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap="coolwarm", s=12, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
plt.tight_layout()
plt.show()

# %% [markdown]
# *What just happened?* On concentric circles the linear kernel is no better than a coin flip, while RBF is
# nearly perfect. But `gamma` is a variance dial like `1/k` in k-NN: a huge `gamma` draws a tiny island around each
# training point (overfit). Check it with a train-vs-CV comparison:

# %%
rows = []
for gamma in [0.1, 1, 10, 100, 1000]:
    m = SVC(kernel="rbf", gamma=gamma)
    train_acc = m.fit(Xm, ym).score(Xm, ym)
    rows.append((gamma, train_acc, cross_val_score(SVC(kernel="rbf", gamma=gamma), Xm, ym, cv=5).mean()))
gamma_table = pd.DataFrame(rows, columns=["gamma", "train_accuracy", "cv_accuracy"]).set_index("gamma")
gamma_table["gap"] = gamma_table["train_accuracy"] - gamma_table["cv_accuracy"]
gamma_table.round(3)

# %% [markdown]
# Training accuracy keeps rising while CV accuracy peaks and then falls: the gap is the overfitting. In practice you
# tune `C` and `gamma` together (log-spaced grid or random search, see the previous notebook) **on scaled features**.
#
# ### SVMs on churn: scaling and cost
# SVMs use distances/dot products too, so scaling is the rule (on this particular table the gap below turns out to be small; do not count on that). Their other practical limit is training time, which
# grows faster than linearly with the number of rows (kernel SVMs need pairwise similarities).

# %%
svm_raw = make_pipeline(SimpleImputer(strategy="median"), SVC())
svm_scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC())
auc_raw = cross_val_score(svm_raw, X_train, y_train, cv=3, scoring="roc_auc").mean()
auc_scaled = cross_val_score(svm_scaled, X_train, y_train, cv=3, scoring="roc_auc").mean()
print(f"RBF-SVM CV AUC   unscaled {auc_raw:.4f}   scaled {auc_scaled:.4f}")

fit_times = {}
for n in [500, 1000, 2000, 4000]:
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC())
    t0 = time.perf_counter()
    m.fit(X_train.iloc[:n], y_train.iloc[:n])
    fit_times[n] = time.perf_counter() - t0
svm_time = pd.Series(fit_times, name="fit seconds").rename_axis("training rows")
evidence["SVM: fit time grows faster than rows (8x rows -> x times slower)"] = (8, svm_time.iloc[-1] / svm_time.iloc[0])
svm_time.round(3)

# %% [markdown]
# Doubling the rows more than doubles the time. Beyond roughly tens of thousands of rows, kernel SVMs stop being a
# practical choice (a linear SVM, `LinearSVC` or `SGDClassifier`, scales much better). Also note SVMs do not output
# probabilities natively: `probability=True` runs an internal cross-validation on top, which is extra cost.
# For ranking, use `decision_function` (the signed distance to the boundary).

# %% [markdown]
# ## 3. Naive Bayes: probabilities from counts
#
# Bayes' rule says `P(class | words) is proportional to P(class) x P(words | class)`. Computing `P(words | class)`
# exactly is hopeless, so Naive Bayes makes a bold, false assumption: **words are independent given the class**.
# Then it is just a product of per-word probabilities, each estimated by counting.
#
# A four-document toy corpus so you can do it by hand:

# %%
docs = ["free prize free money", "win free prize now", "meeting notes attached", "lunch meeting tomorrow"]
labels = ["spam", "spam", "ham", "ham"]
vec = CountVectorizer()
counts = vec.fit_transform(docs)
toy = MultinomialNB(alpha=1.0).fit(counts, labels)

words = vec.get_feature_names_out()
count_table = pd.DataFrame(counts.toarray(), columns=words, index=labels).groupby(level=0).sum()
count_table

# %%
# P(word | class) with add-alpha ("Laplace") smoothing: (count + alpha) / (total words in class + alpha * vocab size)
alpha = 1.0
manual = (count_table + alpha).div(count_table.sum(axis=1) + alpha * len(words), axis=0)
sk = pd.DataFrame(np.exp(toy.feature_log_prob_), index=toy.classes_, columns=words)
print("manual == sklearn:", np.allclose(manual.loc[sk.index], sk))
manual.round(3)

# %% [markdown]
# *What just happened?* Each row is a class's word distribution. **Smoothing** (`alpha`) adds a pseudo-count to
# every word so a word never seen in a class does not get probability zero and veto the whole product. Now score a
# new message:

# %%
new = vec.transform(["free lunch now"])
print({str(c): round(float(q), 3) for c, q in zip(toy.classes_, toy.predict_proba(new)[0])})

# %% [markdown]
# ### On the reviews data
# Now a realistic-size run: predict `sentiment` from the review text, with a bag-of-words + Multinomial NB pipeline.
# **Caveat:** these reviews are synthetic and assembled from a limited set of phrases, so the task is far
# easier than real reviews. Treat the near-perfect score as a property of the toy data, not of Naive Bayes.

# %%
reviews = pd.read_csv(DATA / "reviews.csv")
rt, rv, st, sv = train_test_split(reviews["review_text"], reviews["sentiment"], test_size=0.25,
                                  stratify=reviews["sentiment"], random_state=42)
nb_text = make_pipeline(CountVectorizer(), MultinomialNB()).fit(rt, st)
print(f"sentiment accuracy: {nb_text.score(rv, sv):.3f}   majority-class baseline: {sv.value_counts(normalize=True).max():.3f}")

vec_, nb_ = nb_text[0], nb_text[1]
vocab = np.array(vec_.get_feature_names_out())
for i, cls in enumerate(nb_.classes_):
    lift = nb_.feature_log_prob_[i] - np.delete(nb_.feature_log_prob_, i, axis=0).mean(axis=0)
    print(f"{cls:>9}: most class-specific words -> {list(vocab[np.argsort(lift)[-6:]][::-1])}")

# %% [markdown]
# Naive Bayes is also **inspectable**: the per-class word tables tell you *why* it predicts what it does.
# A useful habit is to run the same pipeline on a target that should *not* be predictable from the text, as a sanity check
# for leakage or for a pipeline bug. Here `category` (Home, Sports, ...) is not encoded in the review wording:

# %%
for target in ["stars", "category"]:
    a_tr, a_te, b_tr, b_te = train_test_split(reviews["review_text"], reviews[target], test_size=0.25,
                                              stratify=reviews[target], random_state=42)
    acc = make_pipeline(CountVectorizer(), MultinomialNB()).fit(a_tr, b_tr).score(a_te, b_te)
    base = DummyClassifier(strategy="most_frequent").fit(a_tr, b_tr).score(a_te, b_te)
    print(f"{target:>9}: NB accuracy {acc:.3f}   most-frequent baseline {base:.3f}")

# %% [markdown]
# `stars` (5 classes) is only partly recoverable, and `category` sits at the baseline: **no signal means no lift**, and
# the baseline is what tells you so. A model that could not beat `DummyClassifier` has learned nothing.

# %% [markdown]
# ### The price of "naive": overconfident probabilities
# Independence is false whenever features are correlated. The *ranking* of predictions often survives, but the
# *probabilities* do not, because correlated evidence gets counted several times. Extreme demonstration: give
# `GaussianNB` copies of the same two churn features, and nothing new is learned.

# %%
X_two = X_train[["tenure_months", "support_calls"]]
X_two_te = X_test[["tenure_months", "support_calls"]]
rows = []
for copies in [1, 3, 10]:
    A = np.hstack([X_two.to_numpy()] * copies)
    A_te = np.hstack([X_two_te.to_numpy()] * copies)
    p = GaussianNB().fit(A, y_train).predict_proba(A_te)[:, 1]
    rows.append((copies, roc_auc_score(y_test, p), log_loss(y_test, p), ((p < 0.05) | (p > 0.95)).mean()))
dup = pd.DataFrame(rows, columns=["copies_of_same_features", "auc", "log_loss", "share_of_extreme_probs"]).set_index("copies_of_same_features")
evidence["Naive Bayes: duplicated features keep AUC but inflate log loss (1 vs 10 copies)"] = (dup.iloc[0]["log_loss"], dup.iloc[-1]["log_loss"])
dup.round(4)

# %% [markdown]
# *What just happened?* The ranking quality (AUC) is essentially unchanged, but log loss, which punishes confident wrong
# answers, gets much worse and more predictions become extreme. If you need trustworthy probabilities from
# Naive Bayes, recalibrate them (see the imbalanced-data notebook) or use a different model.

# %% [markdown]
# ## 4. Strengths and weaknesses: with receipts
# Every row below is backed by a number computed earlier in this notebook:

# %%
receipts = pd.DataFrame(
    [(k, f"{a:.4f}", f"{b:.4f}") for k, (a, b) in evidence.items()],
    columns=["claim", "first value", "second value"],
)
pd.set_option("display.max_colwidth", 90)
receipts

# %% [markdown]
# | Model | Strong when | Weak when | Must-do |
# |---|---|---|---|
# | **k-NN** | little training data, local structure, need "similar items"; no training step | many features or irrelevant ones; large stored data (slow predictions) | scale features; choose `k` by CV; select features |
# | **SVM (RBF)** | small/medium data, non-linear boundary, many features | more than tens of thousands of rows; you need probabilities; hard-to-tune `C`/`gamma` | scale features; tune `C` and `gamma` together |
# | **Naive Bayes** | text with bag-of-words; tiny data; instant baseline; interpretable word tables | correlated features; you need calibrated probabilities | smoothing (`alpha`); recalibrate if you use the probabilities |
#
# None of these is a default choice for modern tabular data (boosting and regularised linear models usually win), but
# they are fast to try, and a big gap between them and a boosted model is a diagnostic worth having.

# %% [markdown]
# ## Common mistakes
#
# 1. **Forgetting to scale** for k-NN / SVM (or scaling the whole data set before the split, which leaks; put the scaler in a `Pipeline`).
# 2. **Default `gamma`/`C` and never tuning**: a kernel SVM's quality depends on them a lot (see the gamma table).
# 3. **Using `k=1`** and being impressed by training accuracy (it is 100% by construction, since each point is its own neighbour).
# 4. **Reading `predict_proba` of Naive Bayes / SVM as real probabilities.**
# 5. **Judging text models by accuracy alone** without a most-frequent baseline (the `category` experiment).
# 6. **Fitting a kernel SVM on 500k rows** and wondering why the job never finishes.

# %%
# Mistake 3 in one line: k=1 memorises the training set
k1 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(1)).fit(X_train, y_train)
print("k=1 train accuracy:", round(k1.score(X_train, y_train), 3), "  test accuracy:", round(k1.score(X_test, y_test), 3))

# %% [markdown]
# ## Interview corner
#
# **1. Why must you scale features for k-NN and SVM but not for a decision tree?**
# <details><summary>Answer</summary>
# k-NN and SVMs are driven by distances or dot products, so a feature with big units dominates. A tree splits on one feature
# at a time using a threshold, so the units of a feature do not matter (only the order of its values).
# </details>
#
# **2. What are support vectors and why do they matter?**
# <details><summary>Answer</summary>
# The training points on or inside the margin. They alone define the boundary; the other points can be removed without changing
# the model. It is why an SVM's prediction cost depends on the number of support vectors and why it is fairly robust to
# far-away points.
# </details>
#
# **3. What does the kernel trick buy you? What does <code>gamma</code> control in an RBF kernel?**
# <details><summary>Answer</summary>
# It computes similarities as if points were mapped into a high-dimensional space, giving non-linear boundaries without building
# the new features. <code>gamma</code> is the inverse of the influence radius of a point: large gamma means very local, wiggly boundaries
# (overfitting), small gamma means smoother, nearly linear ones.
# </details>
#
# **4. Naive Bayes is "wrong" (features are not independent) yet works on text. Why, and what breaks?**
# <details><summary>Answer</summary>
# Classification only needs the correct class to have the highest score, and that ordering is often robust even when the
# probabilities are distorted. What breaks is calibration: correlated evidence is double counted, so the probabilities are
# pushed toward 0 and 1.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Pick `k` by cross-validation
# Using a pipeline of median imputation, `StandardScaler` and `KNeighborsClassifier`, evaluate `k` in `[3, 9, 27, 81]` with
# 5-fold CV and ROC AUC on `X_train, y_train`. Store the k with the best mean score in `best_k` and that score in `best_score`.
# *Hint: a dict comprehension over `cross_val_score(...).mean()` plus `max(d, key=d.get)`.*

# %% tags=["starter"]
best_k = ...
best_score = ...

# %%
def _score(k):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(k))
    return cross_val_score(m, X_train, y_train, cv=5, scoring="roc_auc").mean()

_exp = {k: _score(k) for k in [3, 9, 27, 81]}
assert best_k == max(_exp, key=_exp.get), f"expected k={max(_exp, key=_exp.get)} from {_exp}"
assert abs(best_score - _exp[best_k]) < 1e-9, "best_score should be the mean CV AUC of best_k"
print(f"Exercise 1 passed (k={best_k}, AUC={best_score:.4f})")

# %% tags=["solution"]
scores = {}
for k in [3, 9, 27, 81]:
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(k))
    scores[k] = cross_val_score(m, X_train, y_train, cv=5, scoring="roc_auc").mean()
best_k = max(scores, key=scores.get)
best_score = scores[best_k]

# %% [markdown]
# ### Exercise 2: Who dominates the distance?
# For the first two rows of `X_train` (missing values filled with the training median, as in section 1), compute the
# **unscaled** Euclidean distance and, after standardising all five features with a `StandardScaler` fitted on `X_train`
# (median-imputed), the **scaled** distance. Store them in `dist_raw` and `dist_scaled`, and store the name of the feature with
# the largest squared difference in the unscaled space in `dominant_feature`.
# *Hint: `np.linalg.norm(a - b)`; the scaler needs an imputed matrix.*

# %% tags=["starter"]
dist_raw = ...
dist_scaled = ...
dominant_feature = ...

# %%
_imp = SimpleImputer(strategy="median").fit(X_train)
_M = _imp.transform(X_train)
_two = _M[:2]
_raw = np.linalg.norm(_two[0] - _two[1])
_sc = StandardScaler().fit(_M)
_scaled = np.linalg.norm(_sc.transform(_two)[0] - _sc.transform(_two)[1])
_dom = num_cols[int(np.argmax((_two[0] - _two[1]) ** 2))]
assert abs(dist_raw - _raw) < 1e-6, f"unscaled distance should be {_raw:.3f}"
assert abs(dist_scaled - _scaled) < 1e-6, f"scaled distance should be {_scaled:.3f}"
assert dominant_feature == _dom, f"the dominant feature is {_dom}"
print(f"Exercise 2 passed (raw {dist_raw:.1f}, scaled {dist_scaled:.2f}, dominant: {dominant_feature})")

# %% tags=["solution"]
M = SimpleImputer(strategy="median").fit_transform(X_train)
two = M[:2]
dist_raw = float(np.linalg.norm(two[0] - two[1]))
sc = StandardScaler().fit(M)
dist_scaled = float(np.linalg.norm(sc.transform(two)[0] - sc.transform(two)[1]))
dominant_feature = num_cols[int(np.argmax((two[0] - two[1]) ** 2))]

# %% [markdown]
# ### Exercise 3: Tune `gamma` for an RBF-SVM
# On the moons data (`Xm`, `ym`) evaluate `SVC(kernel="rbf", gamma=g)` for `g` in `[0.01, 0.1, 1, 10, 100]` with 5-fold CV
# accuracy. Store the best gamma in `best_gamma`, and store in `overfit_gap` the (train accuracy minus CV accuracy) of `gamma=100`.
# The check asserts your overfit gap for `gamma=100` is larger than the gap for your best gamma.
# *Hint: train accuracy is `model.fit(Xm, ym).score(Xm, ym)`.*

# %% tags=["starter"]
best_gamma = ...
overfit_gap = ...

# %%
def _cv(g):
    return cross_val_score(SVC(kernel="rbf", gamma=g), Xm, ym, cv=5).mean()

def _gap(g):
    return SVC(kernel="rbf", gamma=g).fit(Xm, ym).score(Xm, ym) - _cv(g)

_g = {g: _cv(g) for g in [0.01, 0.1, 1, 10, 100]}
assert best_gamma == max(_g, key=_g.get), f"expected gamma={max(_g, key=_g.get)} from {_g}"
assert abs(overfit_gap - _gap(100)) < 1e-9, "overfit_gap = train accuracy - CV accuracy at gamma=100"
assert overfit_gap > _gap(best_gamma), "the too-local gamma should overfit more than the best gamma"
print(f"Exercise 3 passed (best gamma {best_gamma}, gap at 100 = {overfit_gap:.3f})")

# %% tags=["solution"]
cv_by_gamma = {g: cross_val_score(SVC(kernel="rbf", gamma=g), Xm, ym, cv=5).mean() for g in [0.01, 0.1, 1, 10, 100]}
best_gamma = max(cv_by_gamma, key=cv_by_gamma.get)
overfit_gap = SVC(kernel="rbf", gamma=100).fit(Xm, ym).score(Xm, ym) - cv_by_gamma[100]

# %% [markdown]
# ### Exercise 4: Naive Bayes by hand
# A tiny corpus: in the **spam** class the word counts are `free: 6, prize: 3, meeting: 0` (9 words total in these three
# vocabulary words). With `alpha=1` smoothing and a vocabulary of just these 3 words, compute `p_meeting_spam`, the smoothed
# `P(meeting | spam)`. Then confirm it matches what `MultinomialNB` learns from that count vector.
# *Hint: (count + alpha) / (total + alpha * vocabulary size).*

# %% tags=["starter"]
p_meeting_spam = ...

# %%
_C = np.array([[6, 3, 0], [0, 1, 5]])  # rows: spam, ham; columns: free, prize, meeting
_nb = MultinomialNB(alpha=1.0).fit(_C, ["spam", "ham"])
_sk = float(np.exp(_nb.feature_log_prob_[list(_nb.classes_).index("spam")][2]))
assert abs(p_meeting_spam - _sk) < 1e-9, f"sklearn learned {_sk:.4f} for P(meeting | spam); you got {p_meeting_spam}"
assert p_meeting_spam > 0, "smoothing must keep the probability above zero"
print(f"Exercise 4 passed (P(meeting | spam) = {p_meeting_spam:.4f})")

# %% tags=["solution"]
p_meeting_spam = (0 + 1.0) / (9 + 1.0 * 3)

# %% [markdown]
# ### Exercise 5 (open-ended): Which model would you ship?
# Take the churn numeric features and compare, with 5-fold CV ROC AUC, a scaled k-NN, a scaled RBF-SVM and `GaussianNB`.
# Add one more you know from earlier notebooks (logistic regression). In a markdown cell write which you would ship and
# why: consider the score gap, training and prediction time, whether you need probabilities, and how you would explain it
# to a colleague.

# %% [markdown]
# ## Recap
# - k-NN votes among the `k` closest points: scale features, tune `k` by CV, beware irrelevant features and slow prediction.
# - An SVM finds the widest margin; `C` trades margin width for training errors, kernels bend the boundary, `gamma` sets locality.
#   Kernel SVMs get slow beyond tens of thousands of rows and need scaling.
# - Naive Bayes multiplies per-feature probabilities estimated by counting (with smoothing): a great fast text baseline,
#   but its probabilities are overconfident when features are correlated.
# - Always compare against a trivial baseline (`DummyClassifier`), and back tool-choice claims with an experiment.
#
# **Go deeper:** scikit-learn user guide chapters "Nearest Neighbors", "Support Vector Machines" and "Naive Bayes"
# (scikit-learn.org/stable/).
#
# **Next:** [Clustering and dimensionality reduction](11_clustering_and_dimensionality_reduction.ipynb)
