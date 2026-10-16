"""Small fixed experiment: linear baseline and one MLP configuration."""

import warnings
from time import perf_counter

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .design import make_grid, objective
from .physics import temperatures


def fit_models(datasets, experiment):
    train_x, train_y = datasets["train"]
    val_x, val_y = datasets["validation"]
    settings = experiment["surrogate"]
    candidates = [("linear", LinearRegression())]
    for seed in settings["seeds"]:
        candidates.append((f"mlp_seed_{seed}", MLPRegressor(
            hidden_layer_sizes=tuple(settings["hidden_layer_sizes"]), activation=settings["activation"],
            solver=settings["solver"], alpha=settings["alpha"], max_iter=settings["max_iter"],
            max_fun=settings["max_fun"], tol=settings["tol"], random_state=seed)))
    models, records = {}, []
    for name, regressor in candidates:
        model = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), regressor), transformer=StandardScaler())
        started = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train_y)
        elapsed = perf_counter() - started
        validation_mae = float(np.mean(abs(model.predict(val_x) - val_y)))
        models[name] = model
        fitted = model.regressor_.steps[-1][1]
        records.append({"model": name, "training_seconds": elapsed, "validation_mae_C": validation_mae,
                        "iterations": getattr(fitted, "n_iter_", None),
                        "warnings": [str(w.message) for w in caught]})
    mlp_records = [row for row in records if row["model"].startswith("mlp_")]
    selected = min(mlp_records, key=lambda row: row["validation_mae_C"])["model"]
    return models, selected, records


def freeze_margin(model, physics, experiment, validation):
    x, y = validation
    point_errors = y - model.predict(x)
    chosen_errors = []
    for tin in experiment["design_validation_temperatures_C"]:
        grid = make_grid(experiment, tin)
        predicted = model.predict(grid)
        eligible = np.flatnonzero(predicted <= physics.target_C)
        if not eligible.size:
            continue
        for weight in experiment["area_weights"]:
            scores = objective(physics, grid, weight)
            index = eligible[np.argmin(scores[eligible])]
            chosen_errors.append(float(temperatures(physics, grid[index:index+1])[0] - predicted[index]))
    quantile = float(np.quantile(point_errors, experiment["margin_quantile"]))
    selected_max = max(chosen_errors, default=0.0)
    return {"margin_C": max(0.0, quantile, selected_max),
            "validation_one_sided_quantile_C": quantile,
            "validation_design_max_optimistic_error_C": selected_max,
            "validation_design_candidates": len(chosen_errors),
            "coverage_guarantee": False}
