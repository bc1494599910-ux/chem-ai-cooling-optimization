import copy

import numpy as np
import pytest

from cooling_project.config import load_config
from cooling_project.data import generate
from cooling_project.design import choose_candidate, pareto_front
from cooling_project.evaluate import prediction_metrics
from cooling_project.io import new_output
from cooling_project.surrogate import fit_models


def test_false_accept_denominator_and_empty_acceptance():
    m = prediction_metrics(np.array([54,56,57,54]), np.array([53,54,56,56]), 55)
    assert m["false_accept_n"] == 1 and m["predicted_feasible_n"] == 2
    assert m["false_accept_rate"] == .5
    assert prediction_metrics(np.array([54]), np.array([56]), 55)["false_accept_rate"] is None


def test_hybrid_rejects_optimistic_first_choice_and_retains_failure():
    x = np.array([[20.,3.,33.],[25.,4.,33.],[30.,5.,33.]])
    pred, scores = np.array([54.,54.,54.]), np.array([1.,2.,3.])
    verifier = lambda z: np.where(z[:,0] == 20, 56., 54.)
    raw, raw_attempts = choose_candidate(x, pred, scores, 55, verifier)
    assert raw["feasible"] is False and len(raw_attempts) == 1
    guarded, attempts = choose_candidate(x, pred, scores, 55, verifier, retry=True)
    assert guarded["area_m2"] == 25 and guarded["feasible"]
    assert len(attempts) == 2 and attempts[0]["feasible"] is False
    none, attempts = choose_candidate(x, pred, scores, 55, verifier, margin=2)
    assert none is None and attempts == []


def test_pareto_drops_dominated_and_duplicate_flow_points():
    x = np.array([[20,4,33],[20,5,33],[21,4,33],[22,3,33],[19,8,33]])
    front = pareto_front(x, np.array([54,54,54,54,56]), 55)
    assert list(front) == [0,3]


def test_splits_are_reproducible_unique_and_boundary_restricted():
    cfg,p = load_config()
    e=copy.deepcopy(cfg["experiment"])
    for name in ("n_train","n_validation","n_test","n_boundary_test","n_extrapolation_test"):
        e[name]=20
    first, second=generate(p,e),generate(p,e)
    assert all(np.array_equal(first[k][0],second[k][0]) for k in first)
    assert np.unique(np.concatenate([x for x,y in first.values()]),axis=0).shape[0] == 100
    assert np.max(abs(first["boundary_test"][1]-p.target_C)) <= 1
    assert first["extrapolation_test"][0][:,2].min() >= 36


def test_scaling_fits_training_only():
    cfg,p=load_config()
    e=copy.deepcopy(cfg["experiment"])
    for name in ("n_train","n_validation","n_test","n_boundary_test","n_extrapolation_test"):
        e[name]=20
    e["surrogate"]["seeds"]=[7]
    e["surrogate"]["max_iter"]=2
    data=generate(p,e)
    # A very different validation distribution must not alter fitted scaler means.
    data["validation"]=(data["validation"][0]+1000, data["validation"][1]+1000)
    models,selected,records=fit_models(data,e)
    scaler=models[selected].regressor_.steps[0][1]
    np.testing.assert_allclose(scaler.mean_,data["train"][0].mean(axis=0))
    assert models[selected].transformer_.mean_[0] == pytest.approx(data["train"][1].mean())


def test_existing_results_are_not_overwritten(tmp_path):
    (tmp_path/"keep.txt").write_text("existing result")
    with pytest.raises(ValueError, match="not empty"):
        new_output(tmp_path)
    assert (tmp_path/"keep.txt").read_text() == "existing result"
