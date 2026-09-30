import numpy as np
import pandas as pd

from churn.data import FEATURE_COLUMNS, TARGET, clean
from churn.features import ENGINEERED_COLUMNS, build_pipeline, build_preprocessor, engineer, make_estimator


def _train_test(raw_df):
    df = clean(raw_df).sort_values("total_charges", na_position="last").reset_index(drop=True)
    cut = int(len(df) * 0.5)  # train = the cheaper half, so train stats differ clearly from full-data stats
    return df.iloc[:cut], df.iloc[cut:], df


def test_preprocessing_statistics_come_from_train_only(raw_df):
    train, _, full = _train_test(raw_df)
    pipe = build_pipeline(make_estimator("logistic"), engineered=True)
    pipe.fit(train[FEATURE_COLUMNS], train[TARGET])
    prep = pipe.named_steps["prep"]

    num = prep.named_transformers_["num"]
    names = list(prep.transformers[0][2])
    i = names.index("total_charges")
    imputed_median = num.named_steps["impute"].statistics_[i]
    scaler_mean_tenure = num.named_steps["scale"].mean_[names.index("tenure_months")]

    assert np.isclose(imputed_median, train["total_charges"].median())
    assert not np.isclose(imputed_median, full["total_charges"].median()), "test setup should make these differ"
    # StandardScaler sees the imputed train column; tenure has no missing values, so mean is exact
    assert np.isclose(scaler_mean_tenure, train["tenure_months"].mean())
    assert not np.isclose(scaler_mean_tenure, full["tenure_months"].mean())


def test_transform_and_predict_do_not_change_fitted_state(raw_df):
    train, test, _ = _train_test(raw_df)
    pipe = build_pipeline(make_estimator("logistic")).fit(train[FEATURE_COLUMNS], train[TARGET])
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    before = scaler.mean_.copy(), scaler.scale_.copy()
    pipe.predict_proba(test[FEATURE_COLUMNS])
    assert np.array_equal(before[0], scaler.mean_) and np.array_equal(before[1], scaler.scale_)


def test_unseen_category_is_ignored_not_fatal(raw_df):
    train, test, _ = _train_test(raw_df)
    pipe = build_pipeline(make_estimator("logistic")).fit(train[FEATURE_COLUMNS], train[TARGET])
    odd = test[FEATURE_COLUMNS].head(3).copy()
    odd["internet_service"] = "Satellite"  # never seen in training, bypasses validation on purpose
    assert pipe.predict_proba(odd).shape == (3, 2)


def test_engineered_features_are_stateless_rowwise(raw_df):
    df = clean(raw_df)[FEATURE_COLUMNS]
    whole = engineer(df).head(20).reset_index(drop=True)
    part = engineer(df.head(20)).reset_index(drop=True)
    pd.testing.assert_frame_equal(whole, part)  # a row's features never depend on other rows


def test_feature_names_include_engineered_and_missing_indicator(raw_df):
    df = clean(raw_df)
    pipe = build_pipeline(make_estimator("logistic"), engineered=True).fit(df[FEATURE_COLUMNS], df[TARGET])
    names = list(pipe[:-1].get_feature_names_out())
    assert set(ENGINEERED_COLUMNS) <= set(names)
    assert any(n.startswith("missingindicator_") for n in names)
    assert "internet_service_Unknown" in names


def test_pipeline_without_engineering(raw_df):
    df = clean(raw_df)
    pipe = build_pipeline(make_estimator("hgb", {"max_iter": 10}), engineered=False).fit(df[FEATURE_COLUMNS], df[TARGET])
    assert not set(ENGINEERED_COLUMNS) & set(pipe[:-1].get_feature_names_out())
    assert build_preprocessor(False) is not None


def test_custom_engineer_fn_and_extra_numeric(raw_df):
    df = clean(raw_df)

    def my_engineer(X):
        X = X.copy()
        X["calls_per_month"] = X["support_calls"] / X["tenure_months"].clip(lower=1)
        return X

    pipe = build_pipeline(make_estimator("logistic"), engineer_fn=my_engineer, extra_numeric=["calls_per_month"])
    pipe.fit(df[FEATURE_COLUMNS], df[TARGET])
    assert "calls_per_month" in pipe.named_steps["prep"].get_feature_names_out()
    assert pipe.predict_proba(df[FEATURE_COLUMNS].head(5)).shape == (5, 2)


def test_training_runs_with_engineered_features_switched_on(cfg, tmp_path):
    from churn.train import run_training

    cfg["features"]["engineer"] = True
    m = run_training(cfg, output_dir=tmp_path / "eng")
    assert m["test"]["roc_auc"] > 0.5
