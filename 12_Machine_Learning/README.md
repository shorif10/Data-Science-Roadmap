# 12 Machine Learning

**Goal:** go from "I can load a DataFrame" to "I can frame, build, evaluate and defend a tabular ML model", with honest baselines, leakage-safe evaluation and reproducible pipelines.

**Prerequisites:** Python, NumPy, pandas, basic statistics and probability (modules 01, 03, 04, 05), a little linear algebra (module 02).

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 01 | `01_ml_problem_framing_and_workflow` | Frame a target, tie the metric to a cost, build baselines, run the workflow once | 1 |
| 02 | `02_train_test_split_and_data_leakage` | Splits, stratification, and four leaks shown and measured | 1 |
| 03 | `03_linear_regression_and_regularization` | Least squares by hand, residuals, Ridge/Lasso/ElasticNet, polynomial overfitting | 1.5 |
| 04 | `04_logistic_regression_and_classification_metrics` | Sigmoid, thresholds, confusion matrix, ROC vs PR AUC, odds ratios | 1.5 |
| 05 | `05_decision_trees_and_overfitting` | Gini/entropy splits by hand, depth curves, pruning, instability | 1 |
| 06 | `06_ensembles_random_forest_and_gradient_boosting` | Bagging and boosting from scratch, RF/HistGB, compare against baselines | 1.5 |
| 07 | `07_cross_validation_and_model_evaluation` | k-fold, stratified/repeated/group/time CV, learning curves, paired comparison, nested CV | 1.5 |
| 08 | `08_feature_engineering_and_pipelines` | ColumnTransformer + Pipeline, custom transformers, joblib save/load | 1.5 |
| 09 | `09_hyperparameter_tuning` | Grid, random and successive-halving search over Pipelines; count fits; watch over-tuning inflate scores; nested CV | 1.5 |
| 10 | `10_knn_svm_and_naive_bayes` | k-NN, SVMs and Naive Bayes: when distance, margins or independence assumptions work or break | 1.5 |
| 11 | `11_clustering_and_dimensionality_reduction` | k-means, elbow/silhouette limits, DBSCAN, hierarchical, PCA on digits (scaling, reconstruction), RFM segmentation | 1.5 |
| 12 | `12_imbalanced_data_and_decision_thresholds` | Rare events: PR-AUC vs prevalence, class weights, resampling only inside folds, cost-based threshold, calibration | 1.5 |
| 13 | `13_model_interpretation_and_explainability` | Coefficients, permutation importance, PDP/ICE by hand, correlated-feature pitfalls, local explanations | 1.5 |
| 14 | `14_time_series_forecasting_baselines` | Naive and seasonal baselines, lag features + boosting, walk-forward CV, MAPE failure modes | 1.5 |
| 15 | `15_anomaly_detection` | z-score/IQR, seasonal residuals, Isolation Forest, LOF, precision/recall at k under an alert budget | 1 |
| 16 | `16_ml_capstone_end_to_end_churn` | Raw CSV to a defensible model: sealed test set, paired model comparison, cost-based threshold, model card | 3 |

**Total: about 23 hours** of reading, typing and exercises.

Datasets are synthetic (`../datasets`, see `../docs/DATASETS.md`); `scikit-learn` bundled sets are used where they teach better.

## You are done when you can...

- [ ] write down the target, horizon, metric and cost of errors for a business request, and build the dummy baseline it must beat
- [ ] split data correctly (stratified, grouped or chronological) and spot four kinds of leakage, including by running a single-feature check
- [ ] fit a linear or logistic model, read coefficients correctly, and choose a decision threshold from a cost ratio
- [ ] choose between ROC-AUC and PR-AUC, and explain why accuracy misleads on imbalanced data
- [ ] explain bagging vs boosting and control tree overfitting with depth, leaf size and learning rate
- [ ] cross-validate with the right splitter, report mean plus spread, and compare two models on identical folds
- [ ] build one `Pipeline` from raw table to prediction (imputing, encoding, scaling, custom cleaning), cross-validate it, and save/reload it
- [ ] run a search over a `Pipeline`, budget the number of fits, and explain why the best search score is optimistic (and how nested CV fixes it)
- [ ] choose and scale a k-NN, SVM or Naive Bayes model, and cluster or compress data while saying how you judge clusters with no labels
- [ ] handle a rare positive class: report PR-AUC against the prevalence floor, resample only inside training folds, pick a threshold from costs, and check calibration
- [ ] explain a model with permutation importance and PDP/ICE, name what correlated features do to each, and refuse to call importance causal
- [ ] forecast with honest baselines and walk-forward validation, and explain why MAPE fails on small or zero values
- [ ] score anomalies against a contextual "normal" and evaluate them with precision/recall at a realistic alert budget
- [ ] take one raw dataset from split to sealed-test evaluation with an uncertainty interval, a cost-based threshold and a filled-in model card
- [ ] say honestly when a fancy model does not beat a simple one

## Try next (real public resources)

- UCI Machine Learning Repository (archive.ics.uci.edu): for example the Adult income and Bank Marketing datasets
- Kaggle (kaggle.com): the "Titanic" and "House Prices" learning competitions
- scikit-learn bundled datasets: `load_breast_cancer`, `load_diabetes`, `load_digits`, `load_wine`
- scikit-learn user guide and example gallery (scikit-learn.org)
- OpenML (openml.org): thousands of tabular datasets with baseline results
