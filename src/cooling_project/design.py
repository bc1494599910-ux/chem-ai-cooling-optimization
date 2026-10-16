"""Transparent finite-grid search; objective is dimensionless, not money."""

from time import perf_counter

import numpy as np

from .physics import temperatures


def make_grid(experiment, cold_inlet_C):
    n = experiment["grid_points_per_axis"]
    bounds = experiment["bounds"]
    a, m = np.meshgrid(np.linspace(*bounds["area_m2"], n),
                       np.linspace(*bounds["cold_flow_kg_s"], n), indexing="ij")
    return np.column_stack((a.ravel(), m.ravel(), np.full(a.size, cold_inlet_C)))


def objective(physics, inputs, area_weight):
    return (area_weight * inputs[:, 0] / physics.baseline_area_m2
            + (1 - area_weight) * inputs[:, 1] / physics.baseline_cold_flow_kg_s)


def choose_candidate(inputs, predicted, scores, target, verifier, margin=0.0, retry=False):
    """Audit every proposed point; hybrid can continue after rejected points.

    verifier receives one-row input arrays. Empty predicted-feasible sets return
    an explicit missing result rather than a zero-temperature/zero-cost design.
    """
    indices = np.flatnonzero(predicted + margin <= target)
    indices = indices[np.argsort(scores[indices], kind="stable")]
    attempts = []
    for index in indices:
        actual = float(verifier(inputs[index:index+1])[0])
        row = {
            "grid_index": int(index), "area_m2": float(inputs[index, 0]),
            "cold_flow_kg_s": float(inputs[index, 1]), "cold_inlet_C": float(inputs[index, 2]),
            "predicted_C": float(predicted[index]), "reference_C": actual,
            "constraint_residual_C": actual - target, "feasible": bool(actual <= target),
            "objective": float(scores[index]), "margin_C": float(margin),
        }
        attempts.append(row)
        if row["feasible"] or not retry:
            return row, attempts
    return None, attempts


def compare_routes(physics, experiment, predictor, margin, inlet_temperatures):
    records, attempt_records = [], []
    for tin in inlet_temperatures:
        x = make_grid(experiment, tin)
        started = perf_counter()
        actual = temperatures(physics, x)
        physical_eval_seconds = perf_counter() - started
        started = perf_counter()
        predicted = np.asarray(predictor(x), dtype=float)
        surrogate_eval_seconds = perf_counter() - started
        for weight in experiment["area_weights"]:
            scores = objective(physics, x, weight)
            feasible_indices = np.flatnonzero(actual <= physics.target_C)
            started = perf_counter()
            best = None if not feasible_indices.size else int(feasible_indices[np.argmin(scores[feasible_indices])])
            direct_search_seconds = perf_counter() - started
            best_score = None if best is None else float(scores[best])
            if best is not None:
                records.append({
                    "strategy": "direct", "cold_inlet_C": tin, "area_weight": weight,
                    "area_m2": float(x[best, 0]), "cold_flow_kg_s": float(x[best, 1]),
                    "predicted_C": float(actual[best]), "reference_C": float(actual[best]),
                    "constraint_residual_C": float(actual[best] - physics.target_C),
                    "feasible": True, "status": "accepted", "objective": best_score,
                    "gap_vs_best_feasible_grid_pct": 0.0, "margin_C": 0.0,
                    "reference_evaluations": len(x), "rejected_candidates": 0,
                    "online_seconds": physical_eval_seconds + direct_search_seconds,
                })
            else:
                records.append({"strategy": "direct", "cold_inlet_C": tin, "area_weight": weight,
                                "feasible": False, "status": "no_feasible_grid_point",
                                "reference_evaluations": len(x), "online_seconds": physical_eval_seconds + direct_search_seconds})
            for strategy, delta, retry in (("surrogate_raw", 0.0, False), ("surrogate_margin_verify", margin, True)):
                started = perf_counter()
                chosen, attempts = choose_candidate(x, predicted, scores, physics.target_C,
                                                    lambda z: temperatures(physics, z), delta, retry)
                seconds = surrogate_eval_seconds + perf_counter() - started
                for rank, attempt in enumerate(attempts, 1):
                    attempt_records.append(dict(attempt, strategy=strategy, area_weight=weight, rank=rank))
                row = {} if chosen is None else dict(chosen)
                row.update(strategy=strategy, cold_inlet_C=tin, area_weight=weight,
                           status=("no_verified_candidate" if chosen is None else "accepted" if chosen["feasible"] else "rejected"),
                           feasible=bool(chosen and chosen["feasible"]),
                           reference_evaluations=len(attempts), rejected_candidates=sum(not a["feasible"] for a in attempts),
                           online_seconds=seconds)
                row["gap_vs_best_feasible_grid_pct"] = (
                    100 * (chosen["objective"] / best_score - 1)
                    if chosen and chosen["feasible"] and best_score is not None else None)
                records.append(row)
    return records, attempt_records


def pareto_front(inputs, actual, target):
    indices = np.flatnonzero(actual <= target)
    indices = indices[np.lexsort((inputs[indices, 1], inputs[indices, 0]))]
    front, smallest_flow = [], float("inf")
    for index in indices:
        if inputs[index, 1] < smallest_flow:
            front.append(int(index))
            smallest_flow = inputs[index, 1]
    return np.asarray(front, dtype=int)
