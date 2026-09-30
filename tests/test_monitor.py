import numpy as np
import pytest

from churn import monitor
from churn.config import load_config
from churn.data import clean
from conftest import ROOT, make_raw_churn

TH = load_config(ROOT / "configs" / "config.yaml")["drift"]


@pytest.fixture(scope="module")
def big():
    return clean(make_raw_churn(6000, seed=7))


def test_identical_data_has_zero_drift(big):
    rep = monitor.drift_report(big, big.copy(), TH)
    assert rep.attrs["overall"] == "ok"
    assert (rep["psi"] < 1e-9).all()
    assert (rep.loc[rep["type"] == "numeric", "ks_stat"] == 0).all()


def test_two_random_halves_of_the_same_population_do_not_alert(big):
    shuffled = big.sample(frac=1, random_state=0)
    a, b = shuffled.iloc[:3000], shuffled.iloc[3000:]
    rep = monitor.drift_report(a, b, TH)
    assert rep.attrs["overall"] == "ok", rep[["feature", "psi", "ks_stat", "status"]]
    assert rep["psi"].max() < TH["psi_warn"]


def test_planted_shift_is_detected_only_where_planted(big):
    cur = big.copy()
    cur["tenure_months"] = (cur["tenure_months"] * 0.5).round()            # customers are newer
    cur["monthly_charges"] = cur["monthly_charges"] * 1.4                   # price rise
    flip = np.random.default_rng(0).random(len(cur)) < 0.6
    cur.loc[flip, "contract"] = "Month-to-month"                            # contract mix moves
    full = monitor.drift_report(big, cur, TH)
    assert full.attrs["overall"] == "alert"
    rep = full.set_index("feature")
    for col in ("tenure_months", "monthly_charges", "contract"):
        assert rep.loc[col, "status"] == "alert", (col, rep.loc[col, "psi"])
    for col in ("autopay", "paperless_billing", "senior_citizen"):
        assert rep.loc[col, "status"] == "ok", col


def test_report_overall_and_sorting(big):
    cur = big.copy()
    cur["support_calls"] = cur["support_calls"] + 3
    rep = monitor.drift_report(big, cur, TH)
    assert rep.attrs["overall"] == "alert"
    assert rep.iloc[0]["status"] == "alert" and rep.iloc[0]["feature"] == "support_calls"


def test_psi_known_values():
    x = np.arange(1000)
    assert monitor.psi_numeric(x, x) == pytest.approx(0, abs=1e-9)
    assert monitor.psi_numeric(x, x + 500) > 0.25
    # categorical: [0.5, 0.5] vs [0.9, 0.1]  ->  (0.4)ln(1.8) + (-0.4)ln(0.2)
    ref = ["a"] * 50 + ["b"] * 50
    cur = ["a"] * 90 + ["b"] * 10
    assert monitor.psi_categorical(ref, cur) == pytest.approx(0.4 * np.log(1.8) + 0.4 * np.log(5), rel=1e-6)


def test_psi_handles_new_category_and_constant_feature():
    assert monitor.psi_categorical(["a"] * 100, ["a"] * 50 + ["z"] * 50) > 0.25
    assert monitor.psi_numeric([1.0] * 100, [1.0] * 100) == pytest.approx(0, abs=1e-9)


def test_ks_statistic_detects_shift():
    rng = np.random.default_rng(0)
    same = monitor.ks_numeric(rng.normal(size=2000), rng.normal(size=2000))
    shifted = monitor.ks_numeric(rng.normal(size=2000), rng.normal(0.5, 1, size=2000))
    assert same[0] < 0.1 < shifted[0]


def test_too_few_rows_refuses_to_judge(big):
    with pytest.raises(ValueError, match="at least"):
        monitor.drift_report(big, big.head(10), TH)


def test_missing_rate_is_reported(big):
    cur = big.copy()
    cur.loc[cur.index[:600], "total_charges"] = np.nan
    row = monitor.drift_report(big, cur, TH).set_index("feature").loc["total_charges"]
    assert row["missing_cur"] > row["missing_ref"] + 0.05


def test_cli_prints_report_and_exit_code(tmp_path, big, capsys):
    big.to_csv(tmp_path / "ref.csv", index=False)
    shifted = big.assign(monthly_charges=big["monthly_charges"] * 2)
    shifted.to_csv(tmp_path / "cur.csv", index=False)
    base = ["--reference", str(tmp_path / "ref.csv"), "--config", str(ROOT / "configs" / "config.yaml")]
    assert monitor.main([*base, "--current", str(tmp_path / "ref.csv"), "--fail-on-alert"]) == 0
    assert monitor.main([*base, "--current", str(tmp_path / "cur.csv"), "--fail-on-alert"]) == 3
    assert "monthly_charges" in capsys.readouterr().out
