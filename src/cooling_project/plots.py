"""Scientific figures generated from saved run inputs and measured results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .design import make_grid, pareto_front
from .physics import temperatures

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 180})


def design_plot(physics, experiment, output):
    grid = make_grid(experiment, physics.baseline_cold_inlet_C)
    actual = temperatures(physics, grid)
    n = experiment["grid_points_per_axis"]
    a, m, t = grid[:, 0].reshape(n, n), grid[:, 1].reshape(n, n), actual.reshape(n, n)
    front = pareto_front(grid, actual, physics.target_C)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), layout="constrained")
    mesh = axes[0].pcolormesh(a, m, t, shading="auto", cmap="RdYlBu_r")
    contour = axes[0].contour(a, m, t, levels=[physics.target_C], colors="black", linewidths=1.4)
    axes[0].clabel(contour, fmt=lambda value: f"{value:g} C", inline=True)
    axes[0].plot(physics.baseline_area_m2, physics.baseline_cold_flow_kg_s, "*", color="white",
                 markeredgecolor="black", markersize=12, label="Report-based baseline")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.colorbar(mesh, ax=axes[0], label="Predicted hot outlet (C)")
    axes[0].set_title("Cooling capacity across candidate designs")
    if front.size:
        axes[1].plot(grid[front, 0], grid[front, 1], color="#087e8b", marker=".", markersize=4)
    axes[1].set_title("Non-dominated feasible grid points")
    axes[1].grid(alpha=.2)
    for ax in axes:
        ax.set_xlabel("Active outside heat-transfer area (m²)")
        ax.set_ylabel("Cooling-water mass flow (kg/s)")
    fig.suptitle(f"E-101 thermal model | water inlet {physics.baseline_cold_inlet_C:g} C | constant U and Cp")
    fig.savefig(output / "design_space.png")
    plt.close(fig)
    return grid, actual, front


def prediction_plot(physics, datasets, model, selected_name, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
    x, y = datasets["test"]
    pred = model.predict(x)
    axes[0].scatter(y, pred, s=9, alpha=.4, color="#087e8b", edgecolors="none")
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    axes[0].plot([lo, hi], [lo, hi], color="#333333", linewidth=1)
    axes[0].set(xlabel="Reference hot outlet (C)", ylabel="MLP prediction (C)", title="Independent in-range test")
    x, y = datasets["boundary_test"]
    pred = model.predict(x)
    wrong = (pred <= physics.target_C) & (y > physics.target_C)
    axes[1].scatter(y[~wrong], (y-pred)[~wrong], s=11, alpha=.4, color="#087e8b", edgecolors="none", label="Other points")
    axes[1].scatter(y[wrong], (y-pred)[wrong], s=24, color="#bd3c36", marker="x", label=f"False acceptance: {wrong.sum()}")
    axes[1].axhline(0, color="#444444", linewidth=.8)
    axes[1].axvline(physics.target_C, color="#444444", linestyle="--", linewidth=.8)
    axes[1].set(xlabel="Reference hot outlet (C)", ylabel="Reference - prediction (C)",
                title="Boundary test: positive error is optimistic")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{selected_name} | synthetic model data, not plant measurements")
    fig.savefig(output / "surrogate_errors.png")
    plt.close(fig)


def timing_plot(timings, selected_name, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
    values = [timings["inference"][name]["batch"]["median_seconds"]*1000
              for name in ("physics", "linear", selected_name)]
    search_values = [timings["full_search"][name]["median_seconds"]*1000
                     for name in ("direct", "surrogate_raw_audited", "surrogate_margin_verify")]
    for ax, labels, data in ((axes[0], ["Physics", "Linear", "MLP"], values),
                              (axes[1], ["Direct", "MLP + audit", "MLP margin\n+ verification"], search_values)):
        bars = ax.bar(labels, data, color=["#334e68", "#6c8ead", "#087e8b"])
        ax.set_yscale("log")
        ax.set_ylabel("Median elapsed time (ms, log scale)")
        for bar, value in zip(bars, data):
            ax.annotate(f"{value:.3g} ms", (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        ax.set_ylim(min(data)*.5, max(data)*3)
    axes[0].set_title(f"Matched batch: {timings['settings']['batch_size']:,} predictions")
    axes[1].set_title("Complete grid search, including candidate checks")
    fig.suptitle("Same process, one BLAS thread | lower is faster | training reported separately")
    fig.savefig(output / "timings.png")
    plt.close(fig)
