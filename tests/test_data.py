import pandas as pd
import pytest

from churn import data
from churn.data import DataValidationError, clean, clean_with_report, load_dataset, load_raw, validate
from conftest import REAL_CSV


def test_load_raw_keeps_literal_none_level(churn_csv):
    df = load_raw(churn_csv)
    assert (df["internet_service"] == "None").sum() > 0, "'None' is a real level, not missing"
    assert df["internet_service"].isna().sum() > 0, "empty fields are still missing"
    # the trap this avoids: pandas' default parsing turns "None" into NaN
    naive = pd.read_csv(churn_csv)
    assert naive["internet_service"].isna().sum() > df["internet_service"].isna().sum()


def test_clean_fixes_contract_casing(raw_df):
    assert raw_df["contract"].nunique() > 3, "fixture must contain casing noise"
    fixed, report = clean_with_report(raw_df)
    assert set(fixed["contract"]) == set(data.CONTRACT_LEVELS)
    assert report["casing_fixed"]["contract"] > 0


def test_clean_coerces_bad_numbers_and_reports(raw_df):
    dirty = raw_df.copy()
    dirty["monthly_charges"] = dirty["monthly_charges"].astype("object")
    dirty.loc[0, "monthly_charges"] = "N/A"
    cleaned, report = clean_with_report(dirty)
    assert pd.isna(cleaned.loc[0, "monthly_charges"])
    assert report["coerced_to_nan"] == {"monthly_charges": 1}
    with pytest.raises(DataValidationError, match="monthly_charges.*missing"):
        validate(cleaned)


def test_clean_does_not_invent_values(raw_df):
    cleaned = clean(raw_df)
    assert cleaned["total_charges"].isna().sum() == raw_df["total_charges"].isna().sum()
    assert cleaned["internet_service"].isna().sum() == raw_df["internet_service"].isna().sum()


def test_clean_drops_duplicates_only_when_asked(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    assert len(clean(doubled)) == len(doubled)
    assert len(clean(doubled, drop_duplicates=True)) == len(raw_df)


def test_valid_frame_passes(raw_df):
    validate(clean(raw_df))


def test_validation_reports_every_problem_at_once(raw_df):
    bad = clean(raw_df).copy()
    bad.loc[0, "tenure_months"] = -3
    bad.loc[1, "contract"] = "Weekly"
    bad.loc[2, "churned"] = 2
    bad.loc[3, "customer_id"] = bad.loc[4, "customer_id"]
    with pytest.raises(DataValidationError) as exc:
        validate(bad)
    text = str(exc.value)
    assert len(exc.value.problems) == 4, text
    for needle in ("tenure_months", "contract", "churned", "customer_id"):
        assert needle in text


def test_validation_missing_column_and_empty(raw_df):
    with pytest.raises(DataValidationError, match="missing required column"):
        validate(clean(raw_df).drop(columns=["autopay"]))
    with pytest.raises(DataValidationError, match="no rows"):
        validate(clean(raw_df).iloc[0:0])


def test_target_not_required_for_inference(raw_df):
    validate(clean(raw_df).drop(columns=["churned"]), require_target=False)


def test_prepare_for_inference_normalises_and_orders_columns(good_record):
    good_record["contract"] = "month-to-month"
    X = data.prepare_for_inference(good_record)
    assert list(X.columns) == data.FEATURE_COLUMNS
    assert X.loc[0, "contract"] == "Month-to-month"


def test_prepare_for_inference_rejects_bad_input(good_record):
    good_record["autopay"] = "maybe"
    with pytest.raises(DataValidationError, match="autopay"):
        data.prepare_for_inference(good_record)
    with pytest.raises(DataValidationError, match="unsupported"):
        data.prepare_for_inference("not a record")


@pytest.mark.skipif(not REAL_CSV.exists(), reason="datasets/churn.csv not generated")
def test_real_dataset_loads_and_is_valid():
    df = load_dataset(REAL_CSV)
    n_lines = sum(1 for _ in open(REAL_CSV, encoding="utf-8")) - 1
    assert len(df) == n_lines
    assert set(df["contract"]) == set(data.CONTRACT_LEVELS)
    assert (df["internet_service"] == "None").sum() > df["internet_service"].isna().sum()
