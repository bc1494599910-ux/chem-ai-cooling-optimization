"""One command runs the predeclared experiment and saves inspectable evidence."""

from datetime import datetime, timezone
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .config import load_config
from .data import FEATURES, generate
from .design import compare_routes
from .evaluate import benchmark, classification_metrics, prediction_metrics
from .io import dump_csv, dump_json, manifest, new_output
from .physics import forward
from .plots import design_plot, prediction_plot, timing_plot
from .report_case import recalculate
from .surrogate import fit_models, freeze_margin


def run(config_path, output_path):
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    config, physics = load_config(config_path)
    e = config["experiment"]
    output = new_output(output_path)
    # Save the protocol before sampling, fitting or examining any test result.
    dump_json(output / "frozen_config.json", config)
    dump_json(output / "report_recalculation.json", recalculate())
    print("Protocol saved; generating independent synthetic splits...", flush=True)
    with threadpool_limits(limits=e["blas_threads"]):
        stage_started = perf_counter()
        datasets = generate(physics, e)
        data_seconds = perf_counter() - stage_started
        dataset_rows = []
        for split, (x, y) in datasets.items():
            for index, (features, target) in enumerate(zip(x, y)):
                dataset_rows.append(dict(zip(FEATURES, map(float, features)), split=split,
                                         row_in_split=index, reference_hot_out_C=float(target), model_version=config["model_version"]))
        dump_csv(output / "dataset.csv", dataset_rows)
        print("Fitting linear baseline and three seeds of one MLP configuration...", flush=True)
        models, selected, fit_records = fit_models(datasets, e)
        dump_json(output / "training.json", {"selected_model": selected, "fits": fit_records})
        print(f"Selected {selected} using validation MAE; freezing margins...", flush=True)
        calibration_started = perf_counter()
        margins = {name: freeze_margin(models[name], physics, e, datasets["validation"])
                   for name in ("linear", selected)}
        calibration_seconds = perf_counter() - calibration_started
        dump_json(output / "frozen_selection.json", {"selected_model": selected, "margins": margins,
                  "selection_basis": "validation only; final tests not used"})
        # Final tests are evaluated only after both model and margins are frozen.
        metrics = {}
        prediction_rows = []
        for name, model in models.items():
            metrics[name] = {}
            for split in ("test", "boundary_test", "extrapolation_test"):
                x, y = datasets[split]
                pred = model.predict(x)
                metrics[name][split] = prediction_metrics(y, pred, physics.target_C)
                if name in margins:
                    metrics[name][split]["guarded_classification"] = classification_metrics(
                        y, pred + margins[name]["margin_C"], physics.target_C)
                for index, (truth, estimated) in enumerate(zip(y, pred)):
                    prediction_rows.append({"model": name, "split": split, "row_in_split": index,
                                            "reference_C": float(truth), "predicted_C": float(estimated)})
        dump_csv(output / "predictions.csv", prediction_rows)
        design_rows, attempts = [], []
        for name in ("linear", selected):
            rows, tried = compare_routes(physics, e, models[name].predict, margins[name]["margin_C"],
                                        e["design_test_temperatures_C"])
            design_rows.extend(dict(row, model=name) for row in rows)
            attempts.extend(dict(row, model=name) for row in tried)
        dump_csv(output / "designs.csv", design_rows)
        dump_csv(output / "candidate_audits.csv", attempts)
        print("Final design cases audited; benchmarking matched workloads...", flush=True)
        timings = benchmark(physics, e, models, selected, margins[selected]["margin_C"])
        dump_json(output / "timings.json", timings)
        baseline = forward(physics, physics.baseline_area_m2, physics.baseline_cold_flow_kg_s,
                           physics.baseline_cold_inlet_C)
        grid, actual, front = design_plot(physics, e, output)
        dump_csv(output / "pareto.csv", [dict(zip(FEATURES, map(float, grid[i])), reference_hot_out_C=float(actual[i])) for i in front])
        prediction_plot(physics, datasets, models[selected], selected, output)
        timing_plot(timings, selected, output)
        design_summary = {}
        for name in ("linear", selected):
            design_summary[name] = {}
            for strategy in ("direct", "surrogate_raw", "surrogate_margin_verify"):
                rows = [r for r in design_rows if r["model"] == name and r["strategy"] == strategy]
                valid_gaps = [r["gap_vs_best_feasible_grid_pct"] for r in rows if r["feasible"]]
                design_summary[name][strategy] = {
                    "cases": len(rows), "reference_feasible_cases": sum(r["feasible"] for r in rows),
                    "missing_candidates": sum(r["status"].startswith("no_") for r in rows),
                    "rejected_candidates": sum(r.get("rejected_candidates", 0) for r in rows),
                    "mean_gap_on_feasible_cases_pct": float(np.mean(valid_gaps)) if valid_gaps else None,
                    "max_gap_on_feasible_cases_pct": float(max(valid_gaps)) if valid_gaps else None,
                }
        summary = {
            "status": "completed", "model_version": config["model_version"], "selected_model": selected,
            "reference_is_plant_truth": False, "data_counts": {name: len(x) for name, (x, y) in datasets.items()},
            "baseline": {"area_m2": physics.baseline_area_m2, "cold_flow_kg_s": physics.baseline_cold_flow_kg_s,
                         "cold_inlet_C": physics.baseline_cold_inlet_C, "hot_out_C": float(baseline.hot_out_C),
                         "cold_out_C": float(baseline.cold_out_C), "heat_kW": float(baseline.heat_W/1000)},
            "nominal_required_heat_kW": physics.hot_mass_flow_kg_s * physics.hot_cp_J_kg_K * (physics.hot_inlet_C - physics.target_C) / 1000,
            "metrics": metrics, "margins": margins, "design_summary": design_summary,
            "training": fit_records, "data_generation_seconds_all_splits": data_seconds,
            "training_seconds_all_candidates": sum(r["training_seconds"] for r in fit_records),
            "margin_calibration_seconds": calibration_seconds,
            "test_seed_mae_range_C": [min(metrics[n]["test"]["mae_C"] for n in models if n.startswith("mlp")),
                                      max(metrics[n]["test"]["mae_C"] for n in models if n.startswith("mlp"))],
            "timing_interpretation": "Inference and full-search timings exclude offline preparation. All preparation costs are reported separately. No speedup is assumed.",
            "scope": "finite grid, equivalent active area, constant U/Cp, cooling capability only; no hydraulic or physical equipment validation",
        }
        dump_json(output / "summary.json", summary)
    dump_json(output / "manifest.json", manifest(config_path, output, started_at, perf_counter() - started))
    print(f"Completed: {output / 'summary.json'}", flush=True)
    return summary
