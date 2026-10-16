"""Metrics retain numerators and denominators; timers use matched workloads."""

from time import perf_counter

import numpy as np

from .data import uniform_inputs
from .design import choose_candidate, make_grid, objective
from .physics import temperatures


def classification_metrics(reference, decision_temperature, target):
    """Decision scores may include a margin; they are not raw predictions."""
    predicted_ok, actual_ok = decision_temperature <= target, reference <= target
    accepted, actual_feasible = int(predicted_ok.sum()), int(actual_ok.sum())
    false_accepts = int((predicted_ok & ~actual_ok).sum())
    false_rejects = int((~predicted_ok & actual_ok).sum())
    return {"predicted_feasible_n": accepted, "false_accept_n": false_accepts,
            "false_accept_rate": false_accepts / accepted if accepted else None,
            "reference_feasible_n": actual_feasible, "false_reject_n": false_rejects,
            "false_reject_rate_among_reference_feasible": false_rejects / actual_feasible if actual_feasible else None}


def prediction_metrics(reference, predicted, target):
    reference, predicted = np.asarray(reference), np.asarray(predicted)
    if reference.shape != predicted.shape or reference.size == 0:
        raise ValueError("Metrics require nonempty arrays of the same shape")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(reference)):
        raise ValueError("Nonfinite predictions must not be silently excluded")
    error = reference - predicted  # Positive means dangerously optimistic cooling.
    return {
        "n": int(reference.size), "mae_C": float(np.mean(abs(error))),
        "rmse_C": float(np.sqrt(np.mean(error**2))),
        "absolute_error_p95_C": float(np.quantile(abs(error), .95)),
        "max_absolute_error_C": float(np.max(abs(error))),
        "max_optimistic_error_C": float(max(0, np.max(error))),
        **classification_metrics(reference, predicted, target),
    }


def timed(function, settings, number=1):
    for _ in range(settings["warmups"]):
        function()
    observations = []
    for _ in range(settings["repeats"]):
        started = perf_counter()
        for _ in range(number):
            function()
        observations.append((perf_counter() - started) / number)
    return {"median_seconds": float(np.median(observations)),
            "p10_seconds": float(np.quantile(observations, .1)),
            "p90_seconds": float(np.quantile(observations, .9)),
            "observations_seconds": observations}


def benchmark(physics, experiment, models, selected_name, margin):
    settings = experiment["timing"]
    x = uniform_inputs(experiment, settings["batch_size"], 1000)
    functions = {"physics": lambda z: temperatures(physics, z),
                 "linear": models["linear"].predict,
                 selected_name: models[selected_name].predict}
    result = {"settings": settings, "inference": {}, "full_search": {}}
    for name, function in functions.items():
        result["inference"][name] = {
            "single": timed(lambda: function(x[:1]), settings, settings["single_calls_per_repeat"]),
            "batch": timed(lambda: function(x), settings),
        }

    def direct_search():
        grid = make_grid(experiment, physics.baseline_cold_inlet_C)
        score = objective(physics, grid, .5)
        actual = temperatures(physics, grid)
        eligible = np.flatnonzero(actual <= physics.target_C)
        return None if not eligible.size else eligible[np.argmin(score[eligible])]

    def proxy_search(delta, retry):
        grid = make_grid(experiment, physics.baseline_cold_inlet_C)
        score = objective(physics, grid, .5)
        pred = models[selected_name].predict(grid)
        return choose_candidate(grid, pred, score, physics.target_C,
                                lambda z: temperatures(physics, z), delta, retry)

    result["full_search"]["direct"] = timed(direct_search, settings)
    result["full_search"]["surrogate_raw_audited"] = timed(lambda: proxy_search(0, False), settings)
    result["full_search"]["surrogate_margin_verify"] = timed(lambda: proxy_search(margin, True), settings)
    result["full_search_case"] = {"cold_inlet_C": physics.baseline_cold_inlet_C, "area_weight": .5,
                                  "grid_points": experiment["grid_points_per_axis"]**2,
                                  "includes": "grid construction, objective, prediction, selection and candidate audit",
                                  "excludes": "data generation, training and margin calibration; separately reported"}
    return result
