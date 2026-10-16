"""Independent synthetic splits with explicit boundary and extrapolation tests."""

import numpy as np

from .physics import temperatures

FEATURES = ("area_m2", "cold_flow_kg_s", "cold_inlet_C")


def uniform_inputs(experiment, n, stream, cold_bounds=None):
    rng = np.random.default_rng(np.random.SeedSequence([experiment["data_seed"], stream]))
    bounds = [experiment["bounds"][name] for name in FEATURES]
    if cold_bounds is not None:
        bounds[2] = cold_bounds
    return np.column_stack([rng.uniform(low, high, n) for low, high in bounds])


def generate(physics, experiment):
    datasets = {}
    for stream, (split, n) in enumerate((("train", experiment["n_train"]),
                                        ("validation", experiment["n_validation"]),
                                        ("test", experiment["n_test"])), 1):
        x = uniform_inputs(experiment, n, stream)
        datasets[split] = (x, temperatures(physics, x))
    needed = experiment["n_boundary_test"]
    chunks, collected = [], 0
    for stream in range(100, 200):
        x = uniform_inputs(experiment, max(10000, needed * 10), stream)
        y = temperatures(physics, x)
        mask = abs(y - physics.target_C) <= experiment["boundary_half_width_C"]
        chunks.append(x[mask])
        collected += int(mask.sum())
        if collected >= needed:
            break
    if collected < needed:
        raise ValueError("Insufficient boundary samples; change experimental bounds explicitly")
    x = np.concatenate(chunks)[:needed]
    datasets["boundary_test"] = (x, temperatures(physics, x))
    x = uniform_inputs(experiment, experiment["n_extrapolation_test"], 4,
                       experiment["extrapolation_cold_inlet_C"])
    datasets["extrapolation_test"] = (x, temperatures(physics, x))
    # Exact duplicate detection guards accidental data-reuse changes.
    all_x = np.concatenate([x for x, _ in datasets.values()])
    if len(np.unique(all_x, axis=0)) != len(all_x):
        raise ValueError("Duplicate input rows across synthetic splits")
    return datasets
