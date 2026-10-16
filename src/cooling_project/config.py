"""Small, explicit JSON configuration interface; units live in field names."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Physics:
    hot_mass_flow_kg_h: float = 6000.0
    hot_inlet_C: float = 110.0
    hot_cp_J_kg_K: float = 2862.9
    cold_cp_J_kg_K: float = 4180.0
    overall_U_W_m2_K: float = 282.6992163028217
    target_C: float = 55.0
    baseline_area_m2: float = 27.547012457342014
    baseline_cold_flow_kg_s: float = 8.971
    baseline_cold_inlet_C: float = 33.0
    area_basis: str = "active outside area"
    U_basis: str = "report B.7 as printed"
    flow_arrangement: str = "one shell pass, even tube passes"

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for name in ("hot_mass_flow_kg_h", "hot_cp_J_kg_K", "cold_cp_J_kg_K", "baseline_cold_flow_kg_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.overall_U_W_m2_K < 0 or self.baseline_area_m2 < 0:
            raise ValueError("U and area must be non-negative")

    @property
    def hot_mass_flow_kg_s(self):
        return self.hot_mass_flow_kg_h / 3600.0


def load_config(path="configs/project.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Only configuration schema_version=1 is supported")
    physics = Physics(**config["physics"])
    e = config["experiment"]
    for name, bounds in e["bounds"].items():
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid experimental bounds: {name}")
    if e["bounds"]["area_m2"][0] < 0 or e["bounds"]["cold_flow_kg_s"][0] <= 0:
        raise ValueError("Experimental area must be non-negative and flow positive")
    if e["bounds"]["cold_inlet_C"][1] > physics.hot_inlet_C:
        raise ValueError("Only cooling configurations are supported")
    if e["grid_points_per_axis"] < 2 or not 0 <= e["margin_quantile"] <= 1:
        raise ValueError("Invalid grid size or margin quantile")
    if any(not 0 < w < 1 for w in e["area_weights"]):
        raise ValueError("Area weights must lie strictly between 0 and 1")
    for name in ("n_train", "n_validation", "n_test", "n_boundary_test", "n_extrapolation_test"):
        if e[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if set(e["design_validation_temperatures_C"]) & set(e["design_test_temperatures_C"]):
        raise ValueError("Validation and final design scenarios must be separate")
    return config, physics
