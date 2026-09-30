# 03 - Statistics and Probability

**Goal:** reason correctly about uncertainty: summarise data, simulate probabilities, quantify the noise in an estimate, run and interpret tests and A/B experiments, fit interpretable regressions, and spot the classic fallacies.

**Prerequisites:** `02_Math_for_Data_Science` (especially notebook 6), plus NumPy and pandas basics.

## Notebook order

| # | Notebook | What you do | Hours |
|---|---|---|---|
| 1 | [01_descriptive_statistics](01_descriptive_statistics.ipynb) | centre, spread, skew, robust statistics, outlier rules | 1 |
| 2 | [02_probability_and_simulation](02_probability_and_simulation.ipynb) | probability rules, Bayes, birthday and Monty Hall by Monte Carlo | 1 |
| 3 | [03_distributions](03_distributions.ipynb) | normal, binomial, Poisson, exponential, lognormal; fitting and QQ plots | 1 |
| 4 | [04_sampling_clt_and_standard_error](04_sampling_clt_and_standard_error.ipynb) | sampling distributions, CLT, standard error, bias vs noise | 1 |
| 5 | [05_confidence_intervals_and_bootstrap](05_confidence_intervals_and_bootstrap.ipynb) | Wilson and t intervals, bootstrap, coverage | 1 |
| 6 | [06_hypothesis_testing](06_hypothesis_testing.ipynb) | p-values, permutation, z/chi-square/Welch/Mann-Whitney, type I/II errors | 1.5 |
| 7 | [07_ab_testing_end_to_end](07_ab_testing_end_to_end.ipynb) | sample size, SRM, analysis, peeking, multiple comparisons | 1.5 |
| 8 | [08_correlation_and_regression_with_statsmodels](08_correlation_and_regression_with_statsmodels.ipynb) | correlation, OLS, diagnostics, logistic regression and odds ratios | 1.5 |
| 9 | [09_bayesian_thinking_intro](09_bayesian_thinking_intro.ipynb) | beta-binomial, credible intervals, prior sensitivity | 1 |
| 10 | [10_statistical_fallacies](10_statistical_fallacies.ipynb) | Simpson, survivorship, p-hacking, base rates, regression to the mean | 1 |

Total: about 11 hours.

## You are done when you can...

- choose mean vs median and IQR vs standard deviation, and detect outliers without being fooled by masking
- verify a probability or Bayes calculation by simulation
- check a distribution fit with a QQ plot and a variance-to-mean check
- explain standard error and compute a confidence interval analytically and by bootstrap
- state the null, assumptions and failure mode of a t-test, chi-square test and Mann-Whitney test
- size an A/B test, run an SRM check, and explain why peeking and many metrics inflate false positives
- read an OLS table and a logistic-regression odds ratio, and run residual diagnostics
- explain a credible interval versus a confidence interval and name five statistical fallacies

## Try next (real public resources)

- scikit-learn bundled datasets (`load_diabetes`, `load_breast_cancer`): run OLS/logistic regression with statsmodels
- UCI Machine Learning Repository (archive.ics.uci.edu): hypothesis tests on real tabular data
- Kaggle public datasets (kaggle.com/datasets): practise bootstrap intervals and A/B style comparisons
- SciPy `scipy.stats` and statsmodels official documentation
- Free books: "OpenIntro Statistics", "Think Stats" and "Think Bayes"

**Next module:** machine learning (`12_Machine_Learning`).
