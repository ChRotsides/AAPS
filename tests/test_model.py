"""Model tests. Need TensorFlow; they are skipped automatically when it is not installed."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytestmark = pytest.mark.tf

from machineLearn4 import LSTMModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(*parts):
    return os.path.join(ROOT, *parts)


@pytest.fixture(scope="module")
def glucose_model():
    """Glucose predictor with the committed pre-trained weights and scaler."""
    m = LSTMModel(256, 20, 4, 20)
    m.load_weights(path("OrginalModels", "weights20-20-age-3layers-256-time.h5"))
    m.load_scalar(path("scalar20-20-age-3layers-256-time"))
    return m


@pytest.fixture(scope="module")
def windows(glucose_model):
    with open(path("PatientData", "adult#002_Dexcom_Insulet_patientData.json")) as f:
        d = json.load(f)
    raw = np.array([d["time"], d["glucose"], d["meal"], d["action"]]).T
    norm = glucose_model.normalize_data(raw).reshape(1, -1, 4)
    x, y, ages = glucose_model.create_look_back(norm, 0.5)
    return x, y, ages


def test_look_back_windows_have_expected_shapes(windows):
    x, y, ages = windows
    n = 6720 - 20 - 20  # samples - look-back - prediction horizon
    assert x.shape == (n, 20, 4)
    assert y.shape == (n, 20)
    assert ages.shape == (n,)


def test_untrained_tiny_model_forward_pass():
    m = LSTMModel(8, 20, 4, 20)
    out = m.predict([np.random.rand(3, 20, 4), np.random.rand(3, 1)])
    assert out.shape == (3, 20)
    assert np.isfinite(out).all()


def test_pretrained_glucose_model_predicts_20_future_values(glucose_model, windows):
    x, y, ages = windows
    pred = glucose_model.predict([x[:4], ages[:4]])
    assert pred.shape == (4, 20)
    assert np.isfinite(pred).all()
    assert -0.5 < pred.min() and pred.max() < 1.5  # normalised glucose range


def test_insulin_model_outputs_one_q_value_per_dose_level(glucose_model, windows):
    x, y, ages = windows
    insulin = glucose_model.create_insulin_full_model()
    insulin.load_weights(path("OrginalModels", "model_20_action_weights_dqn_keras.h5"))
    pred = glucose_model.predict([x[:2], ages[:2]])
    q = insulin.predict([x[:2, :, 1:], pred.reshape(-1, 20, 1)], verbose=0)
    assert q.shape == (2, 20)  # 20 discrete insulin doses
    assert np.isfinite(q).all()


@pytest.mark.parametrize(
    "name, in_shapes, out_shape",
    [
        ("GlucoseModel1", [[1, 20, 4]], [1, 128]),
        ("GlucoseModel2", [[1, 129]], [1, 20]),
        ("InsulinModel1", [[1, 20, 3]], [1, 20, 128]),
        ("InsulinModel2", [[1, 20, 129]], [1, 20]),
    ],
)
def test_android_tflite_models_run(name, in_shapes, out_shape):
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=path("AndroidModels", name + ".tflite"))
    interp.allocate_tensors()
    inputs = interp.get_input_details()
    assert [list(i["shape"]) for i in inputs] == in_shapes
    for i in inputs:
        interp.set_tensor(i["index"], np.random.rand(*i["shape"]).astype(np.float32))
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]["index"])
    assert list(out.shape) == out_shape
    assert np.isfinite(out).all()
