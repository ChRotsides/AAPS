"""Smoke tests for the simulator wrapper (GlucoseEnv.py). No TensorFlow needed."""
import math

import pytest
from simglucose.simulation.scenario import Action as ScenarioAction

from GlucoseEnv import GlucoseEnvironment, risk_index


@pytest.fixture()
def env():
    e = GlucoseEnvironment()
    e.reset(patient_name="adolescent#001")
    # Remove the random meals so that the tests do not depend on the clock-seeded scenario.
    e.patient.scenario.get_action = lambda t: ScenarioAction(meal=0)
    return e


def test_reset_returns_plausible_cgm_and_patient_params():
    e = GlucoseEnvironment()
    cgm = e.reset(patient_name="adolescent#001")
    assert 39 <= cgm <= 600  # Dexcom model clips to its measurable range
    assert e.age == 18
    assert e.CR > 0 and e.CF > 0 and e.TDI > 0 and e.BW > 0
    assert len(e.patient.CGM_hist) == 1


def test_reset_defaults_to_adolescent_001_for_empty_name():
    e = GlucoseEnvironment()
    e.reset(patient_name="")
    assert e.patient_name == "adolescent#001"


@pytest.mark.parametrize("name", ["adult#002", "child#002"])
def test_other_patient_groups_load(name):
    e = GlucoseEnvironment()
    assert math.isfinite(e.reset(patient_name=name))
    assert name.split("#")[0] in e.patient_name


def test_step_appends_history_and_returns_cgm(env):
    p = env.patient
    n_cgm, n_ins, n_cho = len(p.CGM_hist), len(p.insulin_hist), len(p.CHO_hist)
    cgm = env.step(0.0)
    assert math.isfinite(cgm)
    assert (len(p.CGM_hist), len(p.insulin_hist), len(p.CHO_hist)) == (n_cgm + 1, n_ins + 1, n_cho + 1)


def test_extra_carbs_raise_glucose(env):
    base = GlucoseEnvironment()
    base.reset(patient_name="adolescent#001")
    base.patient.scenario.get_action = lambda t: ScenarioAction(meal=0)
    for i in range(30):  # 30 samples x 3 min = 90 min
        env.step(0.0, extra_carbs=100.0 if i == 0 else 0.0)
        base.step(0.0)
    assert env.patient.CGM_hist[-1] > base.patient.CGM_hist[-1] + 20


def test_insulin_lowers_glucose(env):
    base = GlucoseEnvironment()
    base.reset(patient_name="adolescent#001")
    base.patient.scenario.get_action = lambda t: ScenarioAction(meal=0)
    for _ in range(40):
        env.step(0.5)  # large basal dose
        base.step(0.0)
    assert env.patient.CGM_hist[-1] < base.patient.CGM_hist[-1] - 10


def test_risk_index_is_zero_near_reference_and_splits_low_and_high():
    lbgi, hbgi, ri = risk_index([112.5], 1)
    assert ri < 0.1
    lbgi, hbgi, ri = risk_index([50.0], 1)
    assert lbgi > 0 and hbgi == 0 and ri == lbgi
    lbgi, hbgi, ri = risk_index([300.0], 1)
    assert hbgi > 0 and lbgi == 0 and ri == hbgi
