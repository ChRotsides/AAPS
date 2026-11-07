"""Sanity checks on the data files that are committed to the repo."""
import glob
import json
import os
import pickle

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENT_FILES = sorted(glob.glob(os.path.join(ROOT, "PatientData", "*_Dexcom_Insulet_patientData.json")))


def test_patient_files_present():
    assert len(PATIENT_FILES) == 9  # 3 adolescents, 3 adults, 3 children


@pytest.mark.parametrize("path", PATIENT_FILES, ids=os.path.basename)
def test_generated_patient_data_shape_and_ranges(path):
    with open(path) as f:
        d = json.load(f)
    assert set(d) == {"time", "glucose", "action", "meal"}
    n = len(d["glucose"])
    assert n == 14 * 24 * 20  # 14 days at one Dexcom sample every 3 minutes
    assert all(len(d[k]) == n for k in d)
    glucose = np.array(d["glucose"])
    assert np.isfinite(glucose).all()
    assert 39 <= glucose.min() and glucose.max() <= 600
    assert np.array(d["action"]).min() >= 0
    assert np.array(d["meal"]).min() >= 0


def test_android_upload_format():
    with open(os.path.join(ROOT, "PatientData", "patient_data.json")) as f:
        d = json.load(f)
    assert {"data", "patient_name", "age"} <= set(d)
    assert len(d["data"]) > 20
    assert {"glucose", "insulin", "meal", "time"} <= set(d["data"][0])


def test_scaler_pickle_matches_four_input_features():
    sklearn = pytest.importorskip("sklearn")
    assert sklearn is not None
    with open(os.path.join(ROOT, "scalar20-20-age-3layers-256-time"), "rb") as f:
        scaler = pickle.load(f)
    # feature order used throughout the code: time, glucose, meal, insulin
    assert scaler.n_features_in_ == 4
    assert scaler.data_min_[1] == 39  # CGM lower bound
    assert scaler.data_max_[1] > scaler.data_min_[1]
