"""Exercise the Flask REST API in-process (no network, no TensorFlow)."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def client():
    import rest_api

    rest_api.app.config["TESTING"] = True
    return rest_api.app.test_client()


def test_get_params(client):
    r = client.get("/get_params")
    assert r.status_code == 200
    assert set(r.get_json()) == {"CR", "CF", "TDI", "Age"}


def test_reset_with_patient_name(client):
    r = client.get("/reset", query_string={"patient_name": "adult#002"})
    assert r.status_code == 200
    assert 39 <= r.get_json()["glucose"] <= 600
    ages = client.get("/get_age").get_json()["age"]
    assert len(ages) == 20 and len(set(ages)) == 1


def test_step_and_history(client):
    client.get("/reset", query_string={"patient_name": "adolescent#001"})
    r = client.post("/step", json={"action": 0.01, "meal": 0})
    assert r.status_code == 200
    body = r.get_json()
    assert {"time", "glucose", "action", "meal"} <= set(body)
    hist = client.post("/get_cgm_hist", json={"horizon": 2}).get_json()["cgm_hist"]
    assert len(hist) == 2
    times = client.post("/get_time_hist", json={"horizon": 2}).get_json()["time_hist"]
    assert all(isinstance(t, int) for t in times)


def test_model_download(client):
    r = client.get("/get_model/GlucoseModel2")
    assert r.status_code == 200
    assert len(r.data) > 1000
    r.close()
