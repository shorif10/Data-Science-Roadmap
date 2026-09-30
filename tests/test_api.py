import pytest
from fastapi.testclient import TestClient

from churn.api import create_app
from churn.train import MODEL_FILE


@pytest.fixture(scope="module")
def client(trained):
    with TestClient(create_app(str(trained["dir"] / MODEL_FILE))) as c:
        yield c


def test_health(client, trained):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_version"] == trained["metrics"]["model_version"]
    assert body["threshold"] == pytest.approx(trained["metrics"]["threshold"]["value"])


def test_predict_returns_versioned_probability(client, good_record, trained):
    r = client.post("/predict", json=good_record)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["customer_id"] == "X1"
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_flag"] == (body["churn_probability"] >= body["threshold"])
    assert body["model_version"] == trained["metrics"]["model_version"]


def test_risky_customer_scores_higher_than_loyal_one(client, good_record):
    loyal = {**good_record, "tenure_months": 60, "contract": "Two year", "support_calls": 0, "autopay": "Yes",
             "total_charges": 4830.0}
    p_risky = client.post("/predict", json=good_record).json()["churn_probability"]
    p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
    assert p_risky > p_loyal


def test_contract_casing_is_normalised_and_nulls_are_allowed(client, good_record):
    a = client.post("/predict", json=good_record).json()["churn_probability"]
    b = client.post("/predict", json={**good_record, "contract": "MONTH-TO-MONTH"}).json()["churn_probability"]
    assert a == pytest.approx(b)
    r = client.post("/predict", json={**good_record, "total_charges": None, "internet_service": None})
    assert r.status_code == 200


@pytest.mark.parametrize("patch", [
    {"tenure_months": -1}, {"support_calls": "many"}, {"senior_citizen": 2}, {"autopay": "maybe"},
    {"monthly_charges": -5.0}, {"surprise_field": 1},
])
def test_invalid_payloads_are_rejected_with_422(client, good_record, patch):
    assert client.post("/predict", json={**good_record, **patch}).status_code == 422


def test_missing_required_field_is_422(client, good_record):
    del good_record["contract"]
    assert client.post("/predict", json=good_record).status_code == 422


def test_unknown_contract_level_is_422_with_readable_detail(client, good_record):
    r = client.post("/predict", json={**good_record, "contract": "Weekly"})
    assert r.status_code == 422
    assert any("contract" in p for p in r.json()["detail"])


def test_batch_matches_single_calls_in_order(client, good_record):
    customers = [{**good_record, "customer_id": f"B{i}", "tenure_months": t} for i, t in enumerate((1, 12, 40))]
    r = client.post("/predict/batch", json={"customers": customers})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 3 and [p["customer_id"] for p in body["predictions"]] == ["B0", "B1", "B2"]
    singles = [client.post("/predict", json=c).json()["churn_probability"] for c in customers]
    assert [p["churn_probability"] for p in body["predictions"]] == pytest.approx(singles)


def test_identical_customers_in_a_batch_are_not_collapsed(client, good_record):
    r = client.post("/predict/batch", json={"customers": [good_record, good_record]})
    assert r.json()["n"] == 2


def test_empty_and_oversized_batches(trained, good_record):
    with TestClient(create_app(str(trained["dir"] / MODEL_FILE), max_batch_size=2)) as c:
        assert c.post("/predict/batch", json={"customers": []}).status_code == 422
        assert c.post("/predict/batch", json={"customers": [good_record] * 3}).status_code == 413
        assert c.post("/predict/batch", json={"customers": [good_record] * 2}).status_code == 200


def test_service_fails_fast_when_model_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="model file not found"), TestClient(create_app(str(tmp_path / "nope.joblib"))):
        pass


def test_model_is_loaded_once_not_per_request(trained, good_record, monkeypatch):
    from churn import predict

    calls = {"n": 0}
    real = predict.ChurnModel.load.__func__

    def counting(cls, path):
        calls["n"] += 1
        return real(cls, path)

    monkeypatch.setattr(predict.ChurnModel, "load", classmethod(counting))
    with TestClient(create_app(str(trained["dir"] / MODEL_FILE))) as c:
        for _ in range(3):
            c.post("/predict", json=good_record)
    assert calls["n"] == 1


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/predict", "/predict/batch"} <= set(paths)
