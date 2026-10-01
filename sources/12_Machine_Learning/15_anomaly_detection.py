# %% [markdown]
# # Anomaly Detection
#
# **Why this matters.** A machine draws the wrong amount of power, a payment pattern looks odd, a sensor spikes: someone wants an alert, but not 500 a day. Anomaly detection is classification's awkward cousin. Anomalies are
# rare, usually **unlabelled**, and what counts as "unusual" depends on context (200 kW at 3 am is fine; the same reading at noon is a fault). The skill is less about the algorithm and more about defining "normal",
# scoring against it, and choosing how many alerts a human can actually handle.
#
# **You will learn**
# - flag outliers with a z-score and the IQR rule, and see exactly when they miss contextual anomalies
# - build a causal "expected value" from the recent past and score the residual
# - use `IsolationForest` and `LocalOutlierFactor`, and understand why the features you feed them matter more than the model
# - evaluate a detector **without** treating the labels as features: precision/recall at k and PR-AUC, under an alert budget
#
# **Prerequisites:** [Imbalanced data and thresholds](12_imbalanced_data_and_decision_thresholds.ipynb) (PR-AUC) and [Time-series baselines](14_time_series_forecasting_baselines.ipynb) (seasonal expectations).
#
# > Data used: `datasets/sensors.csv` (synthetic hourly machine data). The `is_anomaly` column exists **only to grade our detectors**; in real life you would not have it.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

DATA = Path("../datasets")
sensors = pd.read_csv(DATA / "sensors.csv", parse_dates=["timestamp"]).set_index("timestamp")
power, temp = sensors["power_kw"], sensors["temperature_c"]
print(f"rows {len(sensors)}, labelled anomalies {int(sensors['is_anomaly'].sum())} ({sensors['is_anomaly'].mean():.2%}), missing temperature readings {int(temp.isna().sum())}")

# %% [markdown]
# Missing temperature readings are **sensor dropouts**, which is a data-quality problem rather than an anomaly. Detect them with `isna()` and handle them separately; do not let them leak into the anomaly score as zeros. Now look at the data, with the labelled anomalies marked
# (for our eyes only):

# %%
fig, ax = plt.subplots(figsize=(11, 3.6))
span = slice("2024-02-20", "2024-03-01")
ax.plot(power[span].index, power[span], lw=1)
hit = sensors.loc[span][sensors.loc[span, "is_anomaly"] == 1]
ax.scatter(hit.index, hit["power_kw"], color="tab:red", zorder=3, label="labelled anomaly")
ax.set_title("Hourly power with labelled anomalies, 20 Feb to 1 Mar")
ax.set_xlabel("time")
ax.set_ylabel("power (kW)")
ax.legend()
plt.show()

# %% [markdown]
# ## 1. The global z-score and the IQR rule
#
# The simplest detector: standardise and flag `|z| > 3`. The **IQR rule** (Tukey fences) flags values below `Q1 - 1.5 IQR` or above `Q3 + 1.5 IQR`; it is more robust because quartiles do not move when a few extremes are present. Both
# look at the values **without any notion of time or context**.

# %%
def prec_rec_at_k(score, labels, k):
    """Precision and recall if we investigate only the k highest-scoring points."""
    top = np.argsort(-np.asarray(score))[:k]
    hits = np.asarray(labels)[top].sum()
    return float(hits / k), float(hits / np.asarray(labels).sum())


labels = sensors["is_anomaly"].to_numpy()
z_score = np.abs((power - power.mean()) / power.std()).to_numpy()
q1, q3 = power.quantile([0.25, 0.75])
iqr = q3 - q1
iqr_flag = ((power < q1 - 1.5 * iqr) | (power > q3 + 1.5 * iqr)).to_numpy()
print(f"|z| > 3 flags {int((z_score > 3).sum())} points, IQR fences flag {int(iqr_flag.sum())} points")
print("precision/recall of the |z|>3 flags:", round(float(labels[z_score > 3].mean()), 2), "/", round(float(labels[z_score > 3].sum() / labels.sum()), 2))
print(f"PR-AUC of the raw |z| as a score: {average_precision_score(labels, z_score):.2f}")

# %% [markdown]
# *What just happened?* On this file the labelled anomalies are **large** (far outside the normal range), so even a global rule ranks all of them on top (PR-AUC 1.00), although the fixed `|z| > 3` cut-off still misses a few of them: ranking and thresholding are separate questions. That is the easy case, and it is misleading. Real faults are often subtle: a perfectly normal *value* at the wrong *time*.
# To test whether a detector understands context, we **inject** realistic contextual anomalies into a copy of the series: 8 midday readings set to a normal night-time level (182 kW instead of about 240) and 8 evening/night readings set to a midday level (232 kW instead of about 185). Each value is inside
# the series' normal overall range, so a global rule cannot see them.

# %%
rng = np.random.default_rng(7)
pos = np.arange(len(power))
hours = power.index.hour.to_numpy()
ok_pos = pos[(pos >= 168) & (labels == 0)]  # keep clear of the first week (no history yet) and of the original anomalies
day_pos = rng.choice(ok_pos[(hours[ok_pos] >= 9) & (hours[ok_pos] <= 13)], 8, replace=False)
night_pos = rng.choice(ok_pos[(hours[ok_pos] >= 21) | (hours[ok_pos] <= 2)], 8, replace=False)

power_inj = power.copy()
power_inj.iloc[day_pos] = 182.0
power_inj.iloc[night_pos] = 232.0
labels_inj = labels.copy()
labels_inj[np.r_[day_pos, night_pos]] = 1
print("injected:", len(day_pos) + len(night_pos), "| total anomalies to find now:", int(labels_inj.sum()))
print("injected values are within the normal range of the data:", bool(power[labels == 0].min() <= 182 <= power[labels == 0].max() and power[labels == 0].min() <= 232 <= power[labels == 0].max()))

# %% [markdown]
# From now on the detectors see `power_inj` and are graded against `labels_inj`. We evaluate on rows from the 8th day onward (`pos >= 168`), because a causal detector needs a week of history; a brand-new machine has a **cold-start** period where you need a different rule.

# %%
eval_mask = pos >= 168
print("anomalies in the evaluated rows:", int(labels_inj[eval_mask].sum()), "of", int(labels_inj.sum()))


def grade(name, score, k=50):
    s, l = np.asarray(score)[eval_mask], labels_inj[eval_mask]
    p, r = prec_rec_at_k(s, l, k)
    return {"detector": name, "PR-AUC": average_precision_score(l, s), f"precision@{k}": p, f"recall@{k}": r}


z_inj = np.abs((power_inj - power_inj.mean()) / power_inj.std()).to_numpy()
scores = {"global |z|": z_inj}
pd.DataFrame([grade(n, s) for n, s in scores.items()]).set_index("detector").round(2)

# %% [markdown]
# Precision@k asks: "if the on-call engineer looks at the top k alerts, what fraction are real?" and recall@k: "what fraction of all real anomalies are inside those k?". With only 50 alerts in 4,152 evaluated hours, we are working under an **alert budget**, which is how detectors are used in practice.
# The global z-score catches the large original anomalies and is blind to the injected ones.

# %% [markdown]
# ## 2. Context: a rolling expectation and its residual
#
# "Normal" for 12:00 on a Tuesday is what 12:00 looked like on recent days, not the global mean. Build a **causal expected value**: the median of the same hour over the previous 7 days (the median ignores earlier anomalies; a mean would be dragged by them).
# The **residual** is actual minus expected; scale it by a robust spread (the MAD, median absolute deviation, times 1.4826 to match a standard deviation for normal data). High |robust z| means "unusual for this hour".

# %%
def seasonal_expectation(series, days=7):
    prev = np.stack([series.shift(24 * k).to_numpy() for k in range(1, days + 1)], axis=1)
    exp = np.full(len(series), np.nan)
    full = ~np.isnan(prev).any(axis=1)
    exp[full] = np.median(prev[full], axis=1)
    return exp


exp_p = seasonal_expectation(power_inj)
resid_p = power_inj.to_numpy() - exp_p
valid = ~np.isnan(resid_p)
mad = np.median(np.abs(resid_p[valid] - np.median(resid_p[valid]))) * 1.4826
robust_z = np.where(valid, np.abs(resid_p) / mad, 0.0)
scores["seasonal residual"] = robust_z
print(f"robust scale of the residual (MAD x 1.4826): {mad:.1f} kW")
print(f"alerts at robust z > 4: {int((robust_z > 4).sum())}, of which true: {int(labels_inj[robust_z > 4].sum())}")
pd.DataFrame([grade(n, s) for n, s in scores.items()]).set_index("detector").round(2)

# %% [markdown]
# *What just happened?* Removing the daily pattern lets both kinds of anomaly stand out: the residual is large for a 182 kW reading at noon and for a 232 kW reading at 11 pm. This one idea, **model normal, then score the deviation**, is the core of most practical detectors. It
# uses only past data at each hour (`shift(24*k)`), so it could run in production; a centred rolling window would peek ahead and inflate the score. (That also motivates the thresholds: alerts at robust z > 4 are a judgement about how much spread is normal, not a law.)

# %% [markdown]
# ## 3. Isolation Forest and Local Outlier Factor
#
# **Isolation Forest** builds random trees that split at random thresholds. A point that is unusual gets isolated with few splits, so the average path length is short; `score_samples` turns that into an anomaly score (we negate it so larger = more anomalous). It needs no labels and
# scales well. **Local Outlier Factor (LOF)** compares each point's local density with the density around its `k` nearest neighbours: a point in a much sparser spot than its neighbours gets a high score. Both work on a feature table, so *you* decide what "unusual" is measured on.
# First give them the raw values plus the hour of day (as a sine and cosine so 23:00 and 00:00 are neighbours):

# %%
temp_filled = temp.fillna(temp.median())
raw_feats = np.c_[power_inj.to_numpy(), temp_filled.to_numpy(), np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)]
Zraw = StandardScaler().fit_transform(raw_feats)
scores["IsolationForest, raw+hour"] = -IsolationForest(n_estimators=200, random_state=42).fit(Zraw).score_samples(Zraw)
scores["LOF, raw+hour"] = -LocalOutlierFactor(n_neighbors=35).fit(Zraw).negative_outlier_factor_
pd.DataFrame([grade(n, s) for n, s in scores.items()]).set_index("detector").round(2)

# %% [markdown]
# Now give the same models the **residual** features (how far power and temperature are from their own hour-of-day expectation) instead of raw values:

# %%
# temperature has dropouts (NaN), so use a NaN-ignoring median of the same hour over the previous 7 days
prev_t = np.stack([temp.shift(24 * k).to_numpy() for k in range(1, 8)], axis=1)
with np.errstate(all="ignore"):
    exp_t = np.nanmedian(prev_t, axis=1)
resid_t = temp.to_numpy() - exp_t
res_feats = np.c_[np.nan_to_num(resid_p), np.nan_to_num(resid_t)]  # 0 = "no deviation" where history or the reading is missing
Zres = StandardScaler().fit_transform(res_feats)
scores["IsolationForest, residuals"] = -IsolationForest(n_estimators=200, random_state=42).fit(Zres).score_samples(Zres)
scores["LOF, residuals"] = -LocalOutlierFactor(n_neighbors=35).fit(Zres).negative_outlier_factor_
table = pd.DataFrame([grade(n, s) for n, s in scores.items()]).set_index("detector").round(2)
table

# %% [markdown]
# *What just happened?* Compare rows, not just algorithms. Isolation Forest on raw values plus the hour scores exactly like the global z-score (PR-AUC about 0.61): its random axis-parallel splits isolate the big original anomalies and never see "a normal power value for the wrong hour". Hand it **residual** features and it
# improves clearly (about 0.85). LOF is different: it is *local* by design and we gave it the hour of day, so it finds the context by itself and is near-perfect in both versions. The plain seasonal residual (no machine learning at all) is just as good. The lesson is that **context must be present somewhere**,
# either in the features you construct or in a method that compares each point only with its neighbours. Do not read a winner off these numbers: there are only 40 anomalies to find, the anomalies here are large relative to the noise, and real data is rarely this kind.
# Practical notes: LOF's `fit_predict` only scores the data it was fitted on (for new points use `novelty=True`) and is sensitive to `n_neighbors`; Isolation Forest is randomised (fix `random_state` and check that your top alerts are stable across seeds).
# The "contamination" parameter both expose is just a guess of the anomaly rate used to set a cut-off; it does not change the ranking, so rank first and choose the cut-off separately (next section).

# %% [markdown]
# ## 4. Choosing a cut-off: the alert budget
#
# Because the score is a ranking, the deployment decision is how far down the list to go. Two honest framings: pick **k** from what a human can review per week, or pick a **threshold** from a known false-alarm tolerance. Sweep k for the residual detector:

# %%
ks = np.array([10, 20, 30, 40, 50, 75, 100, 150])
sweep = pd.DataFrame(
    [(k, *prec_rec_at_k(scores["seasonal residual"][eval_mask], labels_inj[eval_mask], k)) for k in ks],
    columns=["k alerts", "precision", "recall"],
).set_index("k alerts")
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(sweep.index, sweep["precision"], "o-", label="precision@k")
ax.plot(sweep.index, sweep["recall"], "o-", label="recall@k")
ax.set_title("Seasonal-residual detector: precision and recall vs alert budget")
ax.set_xlabel("k = number of alerts reviewed")
ax.set_ylabel("fraction")
ax.legend()
plt.show()
sweep.round(2)

# %% [markdown]
# Precision falls and recall rises as the budget grows. Where you stop depends on costs: a missed overheating fault may be far more expensive than an engineer's ten wasted minutes (notebook 12's cost table applies here too). Report the operating point you choose, along with what it implies
# ("reviewing 40 alerts a month catches N of M known faults"). Monitor it over time: sensors drift, and "normal" moves.

# %% [markdown]
# ## Common mistakes
#
# 1. **Using `is_anomaly` as a feature**, or tuning thresholds on the same labels you report. The labels are for grading only (and usually do not exist).
# 2. **Global thresholds on seasonal data** (section 1): a normal value at the wrong time is invisible.
# 3. **Centred rolling windows or full-series statistics** that peek into the future; a production detector only has the past.
# 4. **Treating missing data as anomalies** or filling them with zeros; detect dropouts separately.
# 5. **Reporting accuracy** (about 99% by flagging nothing). Use PR-AUC and precision/recall at k.
# 6. **Trusting `contamination`** as if it were known, and never checking alert volume and review cost.

# %%
# Mistake 5: 'accuracy' of a detector that flags nothing
print(f"accuracy of flagging nothing: {1 - labels_inj.mean():.1%}  |  recall: 0%  |  it would never raise a single alert")
# Mistake 1 (demo): a model that is handed the label column 'detects' perfectly and is useless
print("PR-AUC of the label itself used as a score:", round(average_precision_score(labels_inj, labels_inj), 2), "<- meaningless")

# %% [markdown]
# ## Interview corner
#
# **1. How do you detect anomalies when you have no labels?**
# <details><summary>Answer</summary>
# Define normal and score deviation from it: statistical rules (z-score, IQR), a seasonal or model-based expectation with a residual, or unsupervised models such as Isolation Forest, LOF or an autoencoder. Then validate by sampling the top alerts with domain experts,
# and choose the cut-off from the review budget or the cost of misses versus false alarms.
# </details>
#
# **2. How does Isolation Forest work?**
# <details><summary>Answer</summary>
# It builds many random trees that split on random features at random thresholds. Anomalies are few and different, so they are isolated in fewer splits (shorter average path length); the anomaly score is derived from that path length. No distance computations,
# linear-ish scaling, but sensitive to which features you provide.
# </details>
#
# **3. Why can a z-score fail on time-series data?**
# <details><summary>Answer</summary>
# Trends and seasonality make the mean and variance depend on time, so a value that is normal in context can look extreme globally and vice versa. Remove the expected pattern first (seasonal baseline, rolling median, model residual), then score the residual with a robust scale.
# </details>
#
# **4. How do you evaluate a detector when anomalies are rare?**
# <details><summary>Answer</summary>
# Not with accuracy. Use PR-AUC and precision/recall at a realistic alert budget k, on data the detector was not tuned on, with time-ordered splits. Consider labelling noise (labels often come from incidents that were noticed) and report confidence, since there may be only a few dozen positives.
# </details>

# %% [markdown]
# ## Exercises

# %% [markdown]
# ### Exercise 1: Tukey fences on the original data
# Using the **original** `power` series (not the injected one), compute the IQR fences with the usual multiplier `1.5`, and store the number of points outside the fences in `n_iqr_flagged`. Store in `iqr_precision` the fraction of those flagged points that are labelled anomalies in `labels`.
# Then store in `iqr_recall` the fraction of all labelled anomalies that the fences flag.

# %% tags=["starter"]
n_iqr_flagged = ...
iqr_precision = ...
iqr_recall = ...

# %%
_q1, _q3 = power.quantile([0.25, 0.75])
_w = _q3 - _q1
_flag = ((power < _q1 - 1.5 * _w) | (power > _q3 + 1.5 * _w)).to_numpy()
assert n_iqr_flagged == int(_flag.sum()), "count values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR"
assert abs(iqr_precision - labels[_flag].mean()) < 1e-9, "true anomalies among flagged / flagged"
assert abs(iqr_recall - labels[_flag].sum() / labels.sum()) < 1e-9, "true anomalies flagged / all anomalies"
print(f"Exercise 1 passed: {n_iqr_flagged} flagged, precision {iqr_precision:.2f}, recall {iqr_recall:.2f}")

# %% tags=["solution"]
lo_q, hi_q = power.quantile([0.25, 0.75])
width = hi_q - lo_q
flag = ((power < lo_q - 1.5 * width) | (power > hi_q + 1.5 * width)).to_numpy()
n_iqr_flagged = int(flag.sum())
iqr_precision = float(labels[flag].mean())
iqr_recall = float(labels[flag].sum() / labels.sum())

# %% [markdown]
# ### Exercise 2: A detector that remembers a different window
# Section 2 used the median of the same hour over the previous **7** days. Build the same detector with the previous **3** days (`seasonal_expectation(power_inj, days=3)`), scale the residual by its own MAD x 1.4826 (computed on the valid rows), and store the resulting score array (absolute robust z, `0.0` where the expectation is unavailable) in `score_3d`.
# Store in `pr_auc_3d` its PR-AUC on the evaluated rows (`eval_mask`, `labels_inj`) and set `three_beats_seven` to whether it beats the 7-day version's PR-AUC.

# %% tags=["starter"]
score_3d = ...
pr_auc_3d = ...
three_beats_seven = ...

# %%
_e = seasonal_expectation(power_inj, days=3)
_r = power_inj.to_numpy() - _e
_v = ~np.isnan(_r)
_m = np.median(np.abs(_r[_v] - np.median(_r[_v]))) * 1.4826
_s = np.where(_v, np.abs(_r) / _m, 0.0)
assert np.allclose(score_3d, _s), "abs(residual) / (MAD*1.4826), using days=3 and 0.0 where the expectation is NaN"
_ap3 = average_precision_score(labels_inj[eval_mask], _s[eval_mask])
_ap7 = average_precision_score(labels_inj[eval_mask], robust_z[eval_mask])
assert abs(pr_auc_3d - _ap3) < 1e-9, "average_precision_score on the rows where eval_mask is True"
assert bool(three_beats_seven) == bool(_ap3 > _ap7), "compare with the 7-day PR-AUC"
print(f"Exercise 2 passed: 3-day PR-AUC {pr_auc_3d:.3f} vs 7-day {_ap7:.3f}")

# %% tags=["solution"]
exp3 = seasonal_expectation(power_inj, days=3)
res3 = power_inj.to_numpy() - exp3
ok3 = ~np.isnan(res3)
scale3 = np.median(np.abs(res3[ok3] - np.median(res3[ok3]))) * 1.4826
score_3d = np.where(ok3, np.abs(res3) / scale3, 0.0)
pr_auc_3d = float(average_precision_score(labels_inj[eval_mask], score_3d[eval_mask]))
three_beats_seven = bool(pr_auc_3d > average_precision_score(labels_inj[eval_mask], robust_z[eval_mask]))

# %% [markdown]
# ### Exercise 3: Precision at k for Isolation Forest
# Write `my_precision_at_k(score, labels, k)` from scratch (no sklearn metrics) that returns the fraction of true anomalies among the `k` highest scores. Use it on the evaluated rows of the `"IsolationForest, residuals"` scores
# to fill `p_at_25` (k = 25) and `p_at_100` (k = 100). The residual-based Isolation Forest should be noticeably precise at 25.
# *Hint: `np.argsort(-score)[:k]`.*

# %% tags=["starter"]
def my_precision_at_k(score, lab, k):
    ...


p_at_25 = ...
p_at_100 = ...

# %%
assert abs(my_precision_at_k(np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 0, 0, 1]), 2) - 0.5) < 1e-12, "top 2 scores are items 0 and 2; only item 0 is a true anomaly"
_iso = scores["IsolationForest, residuals"][eval_mask]
_l = labels_inj[eval_mask]
assert abs(p_at_25 - _l[np.argsort(-_iso)[:25]].mean()) < 1e-9, "evaluate on the rows where eval_mask is True"
assert abs(p_at_100 - _l[np.argsort(-_iso)[:100]].mean()) < 1e-9, "same, with k = 100"
assert p_at_25 > p_at_100, "precision should fall as the alert budget grows"
print(f"Exercise 3 passed: precision@25 = {p_at_25:.2f}, precision@100 = {p_at_100:.2f}")

# %% tags=["solution"]
def my_precision_at_k(score, lab, k):
    top = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(lab)[top].mean())


iso_scores = scores["IsolationForest, residuals"][eval_mask]
p_at_25 = my_precision_at_k(iso_scores, labels_inj[eval_mask], 25)
p_at_100 = my_precision_at_k(iso_scores, labels_inj[eval_mask], 100)

# %% [markdown]
# ### Exercise 4: How big an alert budget do we need?
# For the 7-day seasonal-residual score `robust_z` (evaluated rows only), find the **smallest k** such that the top-k alerts contain at least **80%** of the anomalies present in the evaluated rows. Store it in `k_needed`, and the precision at that k in `precision_at_k_needed`.
# *Hint: sort by score, take `np.cumsum` of the sorted labels.*

# %% tags=["starter"]
k_needed = ...
precision_at_k_needed = ...

# %%
_order = np.argsort(-robust_z[eval_mask])
_cum = np.cumsum(labels_inj[eval_mask][_order])
_target = 0.8 * labels_inj[eval_mask].sum()
_k = int(np.argmax(_cum >= _target) + 1)
assert k_needed == _k, f"the first k where cumulative true anomalies reach 80% of {int(labels_inj[eval_mask].sum())}; expected {_k}"
assert abs(precision_at_k_needed - _cum[_k - 1] / _k) < 1e-9, "true anomalies in the top k divided by k"
print(f"Exercise 4 passed: {k_needed} alerts catch 80% of anomalies at precision {precision_at_k_needed:.2f}")

# %% tags=["solution"]
order = np.argsort(-robust_z[eval_mask])
cum_hits = np.cumsum(labels_inj[eval_mask][order])
k_needed = int(np.argmax(cum_hits >= 0.8 * labels_inj[eval_mask].sum()) + 1)
precision_at_k_needed = float(cum_hits[k_needed - 1] / k_needed)

# %% [markdown]
# ### Exercise 5 (open-ended): Design the production alert
# You must ship the seasonal-residual detector to the operations team. Decide and justify: the review budget per week, how you would compute the robust scale in production (rolling window, refit schedule), how you would treat sensor dropouts, what you would do in a machine's first week (cold start),
# and how you would find out whether the detector has gone stale (drift in normal behaviour). Which one of these would you prototype first, and what would success look like?

# %% [markdown]
# ## Recap
# - Anomaly detection is scoring deviation from "normal"; most of the work is defining normal in context (seasonal expectation, rolling window).
# - Global z-score and IQR rules are fine for large context-free outliers and blind to a normal value at the wrong time.
# - Isolation Forest and LOF are only as good as their features: residual features beat raw values here.
# - Evaluate with PR-AUC and precision/recall at k under an alert budget; `is_anomaly` is for grading, never for features.
# - Keep production constraints in mind: causal windows only, dropouts handled separately, cold start, drift monitoring.
#
# **Go deeper:** scikit-learn user guide "Novelty and Outlier Detection" (IsolationForest, LocalOutlierFactor, `novelty=True`); the original papers are "Isolation Forest" (Liu, Ting and Zhou, 2008) and "LOF: Identifying Density-Based Local Outliers" (Breunig et al., 2000).
#
# **Next:** [Capstone: end-to-end churn model](16_ml_capstone_end_to_end_churn.ipynb)
