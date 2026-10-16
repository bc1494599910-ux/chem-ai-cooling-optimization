"""Audit stored evidence and optionally compare two independently produced runs.

Usage: python scripts/verify_results.py results/baseline [runs/reproduction]
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cooling_project.config import load_config
from cooling_project.evaluate import classification_metrics, prediction_metrics
from cooling_project.physics import temperatures


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for rel, expected in manifest["output_sha256"].items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected, f"Artifact hash changed: {rel}"
    for rel, expected in manifest["source_sha256"].items():
        assert hashlib.sha256(Path(rel).read_bytes()).hexdigest() == expected, f"Executable source changed: {rel}"
    cfg, physics = load_config(root / "frozen_config.json")
    summary = json.loads((root / "summary.json").read_text())
    rows = read_csv(root / "dataset.csv")
    keys = [(r["split"], int(r["row_in_split"])) for r in rows]
    assert len(set(keys)) == len(keys)
    x = np.array([[float(r[k]) for k in ("area_m2", "cold_flow_kg_s", "cold_inlet_C")] for r in rows])
    y = np.array([float(r["reference_hot_out_C"]) for r in rows])
    assert len(np.unique(x, axis=0)) == len(x)
    np.testing.assert_allclose(temperatures(physics, x), y, rtol=0, atol=1e-10)
    truth = dict(zip(keys, y))
    predictions = read_csv(root / "predictions.csv")
    for model in summary["metrics"]:
        for split in ("test", "boundary_test", "extrapolation_test"):
            chosen = [r for r in predictions if r["model"] == model and r["split"] == split]
            assert len(chosen) == summary["data_counts"][split]
            actual = np.array([truth[(split, int(r["row_in_split"]))] for r in chosen])
            predicted = np.array([float(r["predicted_C"]) for r in chosen])
            np.testing.assert_allclose(actual, [float(r["reference_C"]) for r in chosen], rtol=0, atol=1e-10)
            metric = prediction_metrics(actual, predicted, physics.target_C)
            recorded = summary["metrics"][model][split]
            for key, value in metric.items():
                if value is None:
                    assert recorded[key] is None
                else:
                    assert np.isclose(value, recorded[key], rtol=0, atol=1e-10), (model, split, key)
            if model in summary["margins"]:
                guarded = classification_metrics(actual, predicted + summary["margins"][model]["margin_C"], physics.target_C)
                assert guarded == recorded["guarded_classification"]
    audits = read_csv(root / "candidate_audits.csv")
    for row in audits:
        xx = np.array([[float(row[k]) for k in ("area_m2", "cold_flow_kg_s", "cold_inlet_C")]])
        actual = float(temperatures(physics, xx)[0])
        assert abs(actual-float(row["reference_C"])) < 1e-10
        assert (row["feasible"] == "True") == (actual <= physics.target_C)
    mlps = [r for r in summary["training"] if r["model"].startswith("mlp_")]
    assert min(mlps, key=lambda r: r["validation_mae_C"])["model"] == summary["selected_model"]
    print(f"Verified {root}: {len(rows)} unique data rows, {len(predictions)} predictions, {len(audits)} candidate audits, artifact/source hashes.")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("comparison", type=Path, nargs="?")
    args = parser.parse_args()
    original = verify(args.run)
    if args.comparison:
        repeated = verify(args.comparison)
        for key in ("selected_model", "data_counts", "metrics", "margins", "baseline", "design_summary"):
            assert original[key] == repeated[key], f"Reproduction differs: {key}"
        for name in ("dataset.csv", "predictions.csv", "pareto.csv", "candidate_audits.csv"):
            assert (args.run/name).read_bytes() == (args.comparison/name).read_bytes(), f"Reproduction differs: {name}"
        print("Reproduction matched: numerical metrics, selected model, designs, data and predictions. Timing values are intentionally not compared.")


if __name__ == "__main__":
    main()
