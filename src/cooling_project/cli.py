"""Run from the repository root so configuration and source paths are explicit."""

import argparse
import json

from .config import load_config
from .io import dump_csv, dump_json, json_default, new_output
from .physics import forward
from .report_case import recalculate


def main(argv=None):
    parser = argparse.ArgumentParser(description="E-101 cooling model and surrogate experiment")
    parser.add_argument("--config", default="configs/project.json")
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="Recalculate Appendix B as printed")
    report.add_argument("--report-config", default="configs/report_appendix_b.json")
    pred = sub.add_parser("predict", help="Predict one thermal operating point")
    pred.add_argument("--area", type=float, help="Active outside area [m2]")
    pred.add_argument("--water-flow", type=float, help="Water mass flow [kg/s]")
    pred.add_argument("--water-inlet", type=float, help="Water inlet [C]")
    search = sub.add_parser("search", help="Create a physical design-space plot and Pareto CSV")
    search.add_argument("--output", default="runs/search")
    experiment = sub.add_parser("experiment", help="Run the fixed surrogate comparison and save results")
    experiment.add_argument("--output", default="runs/experiment")
    args = parser.parse_args(argv)
    try:
        if args.command == "report":
            result = recalculate(args.report_config)
        else:
            config, physics = load_config(args.config)
            if args.command == "predict":
                area = physics.baseline_area_m2 if args.area is None else args.area
                flow = physics.baseline_cold_flow_kg_s if args.water_flow is None else args.water_flow
                inlet = physics.baseline_cold_inlet_C if args.water_inlet is None else args.water_inlet
                thermal = forward(physics, area, flow, inlet)
                result = {"model_version": config["model_version"], "config": args.config,
                          "area_m2": area, "cold_flow_kg_s": flow, "cold_inlet_C": inlet,
                          "hot_mass_flow_kg_s": physics.hot_mass_flow_kg_s,
                          "heat_kW": float(thermal.heat_W/1000), "hot_out_C": float(thermal.hot_out_C),
                          "cold_out_C": float(thermal.cold_out_C), "energy_residual_W": float(thermal.energy_residual_W),
                          "target_C": physics.target_C, "cooling_capability_met": bool(thermal.feasible),
                          "area_basis": physics.area_basis, "U_basis": physics.U_basis}
            elif args.command == "search":
                from .plots import design_plot
                from .data import FEATURES
                output = new_output(args.output)
                grid, actual, front = design_plot(physics, config["experiment"], output)
                dump_csv(output / "pareto.csv", [dict(zip(FEATURES, map(float, grid[i])), reference_hot_out_C=float(actual[i])) for i in front])
                dump_json(output / "config.json", config)
                result = {"grid_points": len(grid), "feasible_points": int((actual <= physics.target_C).sum()),
                          "pareto_points": len(front), "output": str(output)}
            else:
                from .experiment import run
                summary = run(args.config, args.output)
                result = {"status": summary["status"], "selected_model": summary["selected_model"],
                          "output": args.output, "test_mae_C": summary["metrics"][summary["selected_model"]]["test"]["mae_C"]}
        print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
