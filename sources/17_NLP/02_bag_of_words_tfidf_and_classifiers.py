# %% [markdown]
# # Bag of Words, TF-IDF and Classic Text Classifiers
#
# **Why this matters.** Before anyone trains a transformer, a good practitioner builds a **bag-of-words + linear model** baseline. It trains in milliseconds, you can read its weights to see what it learned,
# and on many real tasks (ticket routing, spam, topic tagging) it is within a few points of far heavier models. If you cannot beat this baseline convincingly, you do not yet have a reason to ship
# something bigger. This notebook builds the pieces by hand so nothing is magic, then uses scikit-learn the way a team would.
#
# **You will learn**
# - represent text as counts (bag of words) and as TF-IDF, and reproduce `TfidfVectorizer` by hand
# - train and compare logistic regression, linear SVM and naive Bayes with a leak-free `Pipeline`
# - split text data by *distinct text* and read results against a computable ceiling
# - do error analysis: inspect the largest weights, the confusion matrix, and decide which errors are fixable
#
# **Prerequisites:** [Text preprocessing and tokenization](01_text_preprocessing_and_tokenization.ipynb); from earlier modules, train/test splits and classification metrics
# ([Machine Learning](../12_Machine_Learning/)).
#
# > Data used: `datasets/reviews.csv` (synthetic, `docs/DATASETS.md`). **An honest warning:** the 4,000 reviews are generated from a few templates; only a few hundred texts are distinct and the sentiment is
# > determined by the opening phrase. Accuracy on sentiment will be optimistic (close to perfect). We therefore also predict the **star rating**, where the generator adds genuine randomness, so there is something left to analyse.

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

reviews = pd.read_csv(Path("../datasets/reviews.csv"))
print(f"{len(reviews)} reviews, {reviews['review_text'].nunique()} distinct texts")
reviews.head(3)

# %% [markdown]
# ## 1. Bag of words by hand
# **Intuition.** Forget grammar. Count how many times each vocabulary word occurs in each document, and you get one row of numbers per document. The order of words is lost (hence "bag").
# Two tiny documents make the whole idea visible:

# %%
docs = ["great product great price", "terrible product"]
vocab = sorted({w for d in docs for w in d.split()})
bow = np.array([[d.split().count(w) for w in vocab] for d in docs])
print(vocab)
print(bow)
cv = CountVectorizer()
print("CountVectorizer agrees:", (cv.fit_transform(docs).toarray() == bow).all(), cv.get_feature_names_out().tolist() == vocab)

# %% [markdown]
# ## 2. TF-IDF by hand
# Raw counts over-reward common words: "the" is everywhere and tells you nothing. **TF-IDF** rescales each count:
#
# - **tf(t, d)**: how often term `t` occurs in document `d`.
# - **idf(t)** = `ln((1 + n) / (1 + df(t))) + 1` (scikit-learn's smoothed form): `n` is the number of documents, `df(t)` the number containing `t`. Rare terms get a larger idf.
# - weight = `tf * idf`, then each document vector is scaled to unit length (L2 norm), so long and short documents are comparable.
#
# Reproduce it on a slightly bigger toy corpus and compare with scikit-learn to machine precision:

# %%
toy = ["great product great price", "terrible product", "great price but slow shipping", "slow shipping terrible support"]
cvt = CountVectorizer()
tf = cvt.fit_transform(toy).toarray().astype(float)
n_docs = len(toy)
df_t = (tf > 0).sum(axis=0)
idf = np.log((1 + n_docs) / (1 + df_t)) + 1
manual = tf * idf
manual /= np.linalg.norm(manual, axis=1, keepdims=True)

tfidf = TfidfVectorizer()
sk = tfidf.fit_transform(toy).toarray()
print("max difference vs scikit-learn:", np.abs(manual - sk).max())
pd.DataFrame({"df": df_t, "idf": idf.round(3)}, index=cvt.get_feature_names_out()).T

# %% [markdown]
# *What just happened?* `product` occurs in 2 of 4 documents and gets a low idf; `support` occurs in one and gets the highest. The difference from scikit-learn is at floating-point rounding level,
# so you now know exactly what `TfidfVectorizer` computes. (Other libraries use slightly different formulas; the idea is the same.)

# %% [markdown]
# ## 3. A split that does not leak
# Identical texts on both sides of a random split make the test set a memory test (see the Keras text notebook for the measurement). Split by *distinct text* so every copy of a text lands in the same side.

# %%
groups = pd.factorize(reviews["review_text"])[0]
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr, te = next(gss.split(reviews, groups=groups))
X_tr, X_te = reviews["review_text"].iloc[tr], reviews["review_text"].iloc[te]
print(f"train {len(tr)} rows / {len(set(groups[tr]))} texts | test {len(te)} rows / {len(set(groups[te]))} texts | shared texts: {len(set(groups[tr]) & set(groups[te]))}")

# %% [markdown]
# ## 4. Three classic classifiers
# All three consume sparse TF-IDF (or count) vectors:
# - **Logistic regression**: a weighted sum of word features passed through a softmax; the weights are readable.
# - **Linear SVM (`LinearSVC`)**: finds the separating hyperplane with the widest margin; often a touch stronger on text and very fast.
# - **Multinomial naive Bayes**: assumes words are independent given the class and multiplies class-conditional word probabilities; needs almost no training and works with counts.
#
# A `Pipeline` bundles the vectoriser and the model so `fit` learns the vocabulary from the training fold only, which is what makes cross-validation leak-free.

# %%
def make_models():
    return {
        "majority baseline": make_pipeline(CountVectorizer(), DummyClassifier(strategy="most_frequent")),
        "TF-IDF + logistic regression": make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
        "TF-IDF + linear SVM": make_pipeline(TfidfVectorizer(), LinearSVC()),
        "counts + naive Bayes": make_pipeline(CountVectorizer(), MultinomialNB()),
    }


def evaluate(target, models=None):
    y = reviews[target]
    rows = {}
    for name, model in (models or make_models()).items():
        model.fit(X_tr, y.iloc[tr])
        pred = model.predict(X_te)
        rows[name] = {"accuracy": accuracy_score(y.iloc[te], pred), "macro-F1": f1_score(y.iloc[te], pred, average="macro")}
    return pd.DataFrame(rows).T.round(3)


print("Target: sentiment (negative / neutral / positive)")
evaluate("sentiment")

# %% [markdown]
# Everything real is at or near 100%, and the majority baseline is about half. **When every model is perfect, the comparison says nothing about which is better**: the task is too easy. That is the
# honest consequence of templated data, and a real warning sign in practice (check for leakage and trivial cues). To make the models work for their living, switch to the five-class **star rating**.

# %% [markdown]
# ## 5. A harder target with a computable ceiling
# In this dataset 4 vs 5 stars (for positive reviews) and 1 vs 2 stars (for negative ones) are drawn at random, independent of the text. So no model can reliably beat "guess the most common star for that sentiment".
# We can **measure that ceiling from the data** instead of guessing it:

# %%
ceiling_counts = reviews.groupby("sentiment")["stars"].agg(lambda s: s.value_counts().iloc[0])
ceiling = float(ceiling_counts.sum() / len(reviews))
print(reviews.groupby("sentiment")["stars"].value_counts(normalize=True).round(2).to_string())
print(f"\nceiling accuracy for stars if the sentiment is known perfectly: {ceiling:.3f}")

# %% [markdown]
# If the text only tells us the sentiment, the best accuracy on the star rating is the share of the most common star in each sentiment group. Now let the models try:

# %%
stars_table = evaluate("stars")
stars_table["ceiling (accuracy)"] = round(ceiling, 3)
stars_table

# %% [markdown]
# *What just happened?* Logistic regression and the SVM land on the ceiling (an SVM number slightly *above* it is luck on a 898-row test set, not skill: there is nothing left to learn).
# Naive Bayes has lower accuracy but the **highest macro-F1**. Both facts are real: it predicts the rarer 1-star and 4-star classes more often, which costs accuracy but helps the per-class average. Metrics
# encode a preference, so pick the one that matches the business question (overall hit rate, or fairness to rare classes) *before* comparing models. A small gap below the ceiling is model error;
# a gap above it is noise. Knowing the irreducible-error ceiling is one of the best habits in applied ML: it tells you when to stop tuning.

# %% [markdown]
# ## 6. Small data: where model choice starts to matter
# Models differ most when data is scarce. Train each on only `k` randomly drawn distinct training texts and score on the same test set (sentiment target), averaging over a few random draws:

# %%
rng = np.random.default_rng(42)
train_texts = reviews.iloc[tr].drop_duplicates("review_text")
sizes = [5, 10, 20, 50, 200]
curve = {name: [] for name in list(make_models())[1:]}
for k in sizes:
    accs = {name: [] for name in curve}
    for _ in range(5):
        sub = train_texts.iloc[rng.choice(len(train_texts), size=k, replace=False)]
        for name, model in make_models().items():
            if name in curve:
                model.fit(sub["review_text"], sub["sentiment"])
                accs[name].append(accuracy_score(reviews["sentiment"].iloc[te], model.predict(X_te)))
    for name in curve:
        curve[name].append(np.mean(accs[name]))

fig, ax = plt.subplots(figsize=(6.5, 3.8))
for name, vals in curve.items():
    ax.plot(sizes, vals, marker="o", label=name)
ax.set_xscale("log")
ax.set_xlabel("distinct training texts (log scale)")
ax.set_ylabel("test accuracy (mean of 5 draws)")
ax.set_title("Sentiment accuracy versus training-set size")
ax.legend()
plt.tight_layout()
plt.show()
pd.DataFrame(curve, index=sizes).round(3)

# %% [markdown]
# With few examples the methods differ and accuracy climbs quickly; with a couple of hundred distinct texts they all converge. Real tasks usually sit somewhere on this curve where the choice still matters,
# and the curve itself answers a business question: *would collecting more labelled data help?* If the line is already flat, the answer is no.

# %% [markdown]
# ## 7. Reading the model: what did it learn?
# A linear model's weights are a direct explanation. For each class, list the words with the largest weight:

# %%
pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)).fit(X_tr, reviews["sentiment"].iloc[tr])
vec, clf = pipe.named_steps["tfidfvectorizer"], pipe.named_steps["logisticregression"]
words = np.array(vec.get_feature_names_out())
for k, cls in enumerate(clf.classes_):
    print(f"{cls:9s}", list(words[np.argsort(-clf.coef_[k])[:6]]))

# %% [markdown]
# These are the opening phrases of the generator's templates ("terrible", "okay", "perfectly"), exactly what we expected. On real data this view is how you catch a model that learned
# the wrong thing (a shipping-company name predicting "negative", or an ID token predicting the label).

# %% [markdown]
# ## 8. Error analysis on the star rating
# Where do the mistakes of the star model go?

# %%
star_pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)).fit(X_tr, reviews["stars"].iloc[tr])
pred = star_pipe.predict(X_te)
y_star = reviews["stars"].iloc[te].to_numpy()
labels = [1, 2, 3, 4, 5]
cm = pd.DataFrame(confusion_matrix(y_star, pred, labels=labels), index=[f"true {s}" for s in labels], columns=[f"pred {s}" for s in labels])
print(cm)
off = confusion_matrix(y_star, pred, labels=labels)
adjacent = sum(off[i, j] for i in range(5) for j in range(5) if i != j and abs(i - j) == 1)
print(f"\nmistakes: {off.sum() - np.trace(off)}; of which between adjacent star levels: {adjacent}")

# %% [markdown]
# Virtually every mistake is 1 versus 2 or 4 versus 5: the very pairs whose labels are random in the generator. That is what "irreducible" looks like in a confusion matrix. On real reviews the same pattern
# often appears (people disagree about 4 versus 5 stars), which is an argument for reporting accuracy *within one star* or treating ratings as an ordered target instead of five unrelated classes.
#
# **Practical caveat.** Real review text is not templated. There, bag-of-words models struggle with negation ("not bad"), sarcasm, and words they never saw. The next notebooks address those limits (word embeddings, sequence models, transformers).

# %% [markdown]
# ## Common mistakes
#
# 1. **Fitting TF-IDF on the full dataset before splitting.** The idf values and vocabulary leak test information. Use a `Pipeline`, and cross-validate the pipeline, not the matrix.
# 2. **Random splits with duplicated text** inflate scores; group by distinct text (or near-duplicate cluster).
# 3. **Accuracy on imbalanced classes** without the majority-class baseline next to it.
# 4. **Reporting one number for models that differ by less than the noise.** With a few hundred distinct test texts, one text is a large fraction of a percentage point.
# 5. **Tuning against the test set.** Compare models with cross-validation on the training data; look at the test set once.
# 6. **Ignoring the ceiling.** If the labels contain irreducible noise, chasing the last points is wasted effort.
# 7. **Treating `MultinomialNB` like the others**: it expects non-negative counts or tf-idf; it cannot take centred or standardised features.

# %%
# Demonstration of mistake 7: naive Bayes refuses negative feature values
from sklearn.preprocessing import StandardScaler

Xc = StandardScaler(with_mean=True).fit_transform(CountVectorizer().fit_transform(X_tr).toarray())
try:
    MultinomialNB().fit(Xc, reviews["sentiment"].iloc[tr])
except ValueError as e:
    print("ValueError:", str(e)[:70])

# %% [markdown]
# ## Interview corner
#
# **1. What does TF-IDF do and why is it better than raw counts?**
# <details><summary>Answer</summary>
# It multiplies a term's frequency in a document by the log of how rare the term is across documents, so ubiquitous words are down-weighted and distinctive words stand out. L2 normalisation then makes documents of different length comparable.
# It is a cheap, strong representation for linear models and search. It does not capture word order or meaning.
# </details>
#
# **2. Why is naive Bayes called "naive" and why does it still work on text?**
# <details><summary>Answer</summary>
# It assumes features (words) are conditionally independent given the class, which is false. It often works because classification only needs the class probabilities to be ranked correctly, not calibrated,
# and it has few parameters, so it does well with little data. Its probabilities, however, are usually badly calibrated.
# </details>
#
# **3. Your classifier gets 99% accuracy. What do you do next?**
# <details><summary>Answer</summary>
# Compare to the majority-class baseline, check for leakage and duplicates across splits, look at the largest model weights for suspicious features, and evaluate on data from the real deployment source.
# </details>
#
# **4. How would you deal with a word that is not in the training vocabulary?**
# <details><summary>Answer</summary>
# A bag-of-words vectoriser ignores it (its column does not exist). Options: character n-grams (robust to typos), sub-word tokenisation, hashing vectoriser (no fixed vocabulary), or pretrained embeddings that have seen many more words.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: The IDF of one word
# Fit a default `TfidfVectorizer()` on the **training** texts `X_tr`. Store in `idf_support` the idf value of the word `"support"`. Then compute the same number by hand from the formula
# `ln((1 + n) / (1 + df)) + 1` (count documents containing the word with a regular expression, `\bsupport\b`, case-insensitive) and store it in `idf_by_hand`. They must match.
# *Hint: `vec.idf_[vec.vocabulary_["support"]]`.*

# %% tags=["starter"]
idf_support = ...
idf_by_hand = ...

# %%
n = len(X_tr)
df_count = int(X_tr.str.contains(r"\bsupport\b", case=False).sum())
assert df_count > 0, "the word should occur in the training texts"
expected = np.log((1 + n) / (1 + df_count)) + 1
assert abs(idf_by_hand - expected) < 1e-9, f"by hand: expected {expected:.6f}"
assert abs(idf_support - expected) < 1e-9, f"TfidfVectorizer gives {expected:.6f}; yours is {idf_support:.6f}"
print(f"Exercise 1 passed (idf of 'support' = {idf_support:.4f})")

# %% tags=["solution"]
v = TfidfVectorizer().fit(X_tr)
idf_support = float(v.idf_[v.vocabulary_["support"]])
n = len(X_tr)
df_count = int(X_tr.str.contains(r"\bsupport\b", case=False).sum())
idf_by_hand = float(np.log((1 + n) / (1 + df_count)) + 1)

# %% [markdown]
# ### Exercise 2: Does the category column carry sentiment?
# A product manager asks whether some product `category` gets systematically worse reviews. Store in `neg_rate` a pandas Series: the fraction of reviews with `sentiment == "negative"` per `category`,
# sorted from highest to lowest. Store in `chi_p` the p-value of a chi-square test of independence between `category` and `sentiment` (`scipy.stats.chi2_contingency` on the contingency table).
# Then decide: store `True` in `category_matters` if `chi_p < 0.05`. (The generator draws category independently of sentiment, so what do you expect?)

# %% tags=["starter"]
neg_rate = ...
chi_p = ...
category_matters = ...

# %%
from scipy.stats import chi2_contingency

exp_rate = (reviews["sentiment"] == "negative").groupby(reviews["category"]).mean().sort_values(ascending=False)
assert list(neg_rate.index) == list(exp_rate.index), "sort from highest to lowest negative rate"
assert np.allclose(neg_rate.to_numpy(), exp_rate.to_numpy()), "neg_rate is the negative fraction per category"
exp_p = chi2_contingency(pd.crosstab(reviews["category"], reviews["sentiment"]))[1]
assert abs(chi_p - exp_p) < 1e-9, f"expected p = {exp_p:.4f}"
assert category_matters == (exp_p < 0.05), "compare the p-value with 0.05"
print(f"Exercise 2 passed (p = {chi_p:.3f}, category_matters = {category_matters})")

# %% tags=["solution"]
from scipy.stats import chi2_contingency

neg_rate = (reviews["sentiment"] == "negative").groupby(reviews["category"]).mean().sort_values(ascending=False)
chi_p = float(chi2_contingency(pd.crosstab(reviews["category"], reviews["sentiment"]))[1])
category_matters = bool(chi_p < 0.05)

# %% [markdown]
# ### Exercise 3: Cross-validated macro-F1 without leakage
# Evaluate `make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))` for the **stars** target with 4-fold `GroupKFold` (groups = distinct text, `groups` array from section 3),
# scoring macro-F1 on each held-out fold. Store the list of four fold scores in `fold_f1` and their mean in `mean_f1`.
# *Hint: loop over `GroupKFold(n_splits=4).split(reviews, groups=groups)`, fit on the training indices, predict the held-out ones.*

# %% tags=["starter"]
fold_f1 = ...
mean_f1 = ...

# %%
assert len(fold_f1) == 4, "four folds"
assert abs(mean_f1 - float(np.mean(fold_f1))) < 1e-12, "mean_f1 is the mean of the fold scores"
chk = []
for a, b in GroupKFold(n_splits=4).split(reviews, groups=groups):
    assert len(set(groups[a]) & set(groups[b])) == 0
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000)).fit(reviews["review_text"].iloc[a], reviews["stars"].iloc[a])
    chk.append(f1_score(reviews["stars"].iloc[b], m.predict(reviews["review_text"].iloc[b]), average="macro"))
assert np.allclose(fold_f1, chk), f"expected fold scores {np.round(chk, 4)}"
print(f"Exercise 3 passed (macro-F1 per fold {np.round(fold_f1, 3)}, mean {mean_f1:.3f})")

# %% tags=["solution"]
fold_f1 = []
for a, b in GroupKFold(n_splits=4).split(reviews, groups=groups):
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000)).fit(reviews["review_text"].iloc[a], reviews["stars"].iloc[a])
    fold_f1.append(float(f1_score(reviews["stars"].iloc[b], m.predict(reviews["review_text"].iloc[b]), average="macro")))
mean_f1 = float(np.mean(fold_f1))

# %% [markdown]
# ### Exercise 4: The words that signal "negative"
# Using the sentiment pipeline `pipe` from section 7 (TF-IDF + logistic regression, trained on `X_tr`), store in `top_negative` the **set** of the three words with the largest weight for the class `"negative"`,
# and in `top_positive` the set of the three words with the largest weight for `"positive"`. The check recomputes them from a freshly trained model.

# %% tags=["starter"]
top_negative = ...
top_positive = ...

# %%
fresh = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)).fit(X_tr, reviews["sentiment"].iloc[tr])
fw = np.array(fresh.named_steps["tfidfvectorizer"].get_feature_names_out())
fc = fresh.named_steps["logisticregression"]
cls = list(fc.classes_)
assert set(top_negative) == set(fw[np.argsort(-fc.coef_[cls.index("negative")])[:3]]), "top 3 by coefficient for the negative class"
assert set(top_positive) == set(fw[np.argsort(-fc.coef_[cls.index("positive")])[:3]]), "top 3 by coefficient for the positive class"
assert not set(top_negative) & set(top_positive), "the two sets should not overlap"
print("Exercise 4 passed:", sorted(top_negative), sorted(top_positive))

# %% tags=["solution"]
cls = list(clf.classes_)
top_negative = set(words[np.argsort(-clf.coef_[cls.index("negative")])[:3]])
top_positive = set(words[np.argsort(-clf.coef_[cls.index("positive")])[:3]])

# %% [markdown]
# ### Exercise 5 (open-ended): Would you ship this?
# Imagine this bag-of-words model is proposed for a real customer-review dashboard. In a markdown cell, list three checks you would run before trusting it (for example: performance on a sample of real, unseen reviews,
# behaviour on negation such as "not bad", on other languages or spelling errors, agreement between two human labellers) and what result would make you say no.

# %% [markdown]
# ## Recap
# - Bag of words counts terms and ignores order; TF-IDF down-weights words that appear in many documents (idf = ln((1+n)/(1+df)) + 1 in scikit-learn) and normalises each row.
# - Wrap the vectoriser and classifier in a `Pipeline` so cross-validation cannot leak vocabulary or idf from the held-out data.
# - Logistic regression, linear SVM and naive Bayes are cheap and close to each other on easy tasks; compare them to a majority baseline and on learning curves, not on one saturated score.
# - A ceiling computed from the data (here, the irreducible randomness of star ratings) tells you when further tuning is pointless.
# - Linear weights are an explanation: read them for suspicious features. This dataset is templated, so real-world scores will be lower.
#
# **Go deeper:** the scikit-learn user guide chapters "Working with text data" and "Feature extraction" (text), and the Naive Bayes section; the book *Speech and Language Processing* by Jurafsky and Martin (free draft online), chapters on naive Bayes and logistic regression.
#
# **Next:** [Word embeddings from scratch](03_word_embeddings_from_scratch.ipynb)
