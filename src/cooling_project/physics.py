"""Constant-property, constant-U one-shell/even-tube-pass thermal model.

Reference: ht.hx.effectiveness_from_NTU, subtype='S&T', one shell.
The TEMA E effectiveness relation is a thermal idealization, not a mechanical
rating, pressure-drop model, or validation against an operating plant.
"""

from dataclasses import dataclass

import numpy as np

from .config import Physics


@dataclass(frozen=True)
class ThermalResult:
    heat_W: np.ndarray
    hot_out_C: np.ndarray
    cold_out_C: np.ndarray
    effectiveness: np.ndarray
    energy_residual_W: np.ndarray
    feasible: np.ndarray


def one_shell_effectiveness(ntu, capacity_ratio):
    """Stable form using tanh, including zero NTU and equal capacity rates."""
    ntu, cr = np.broadcast_arrays(np.asarray(ntu, dtype=float), np.asarray(capacity_ratio, dtype=float))
    if not np.all(np.isfinite(ntu)) or not np.all(np.isfinite(cr)):
        raise ValueError("NTU and capacity ratio must be finite")
    if np.any(ntu < 0) or np.any((cr < 0) | (cr > 1)):
        raise ValueError("NTU >= 0 and 0 <= capacity ratio <= 1 are required")
    s = np.sqrt(1.0 + cr * cr)
    t = np.tanh(0.5 * ntu * s)
    return 2.0 * t / ((1.0 + cr) * t + s)


def forward(physics: Physics, area_m2, cold_flow_kg_s, cold_inlet_C):
    area, flow, tin = np.broadcast_arrays(
        np.asarray(area_m2, dtype=float),
        np.asarray(cold_flow_kg_s, dtype=float),
        np.asarray(cold_inlet_C, dtype=float),
    )
    if not all(np.all(np.isfinite(v)) for v in (area, flow, tin)):
        raise ValueError("Inputs must be finite")
    if np.any(area < 0) or np.any(flow <= 0):
        raise ValueError("Area must be non-negative and cooling-water flow positive")
    if np.any(tin > physics.hot_inlet_C):
        raise ValueError("Cold inlet must not exceed hot inlet in this cooling model")
    # The kg/h -> kg/s conversion belongs here, via the explicit property.
    ch = physics.hot_mass_flow_kg_s * physics.hot_cp_J_kg_K
    cc = flow * physics.cold_cp_J_kg_K
    cmin, cmax = np.minimum(ch, cc), np.maximum(ch, cc)
    ntu = physics.overall_U_W_m2_K * area / cmin
    eps = one_shell_effectiveness(ntu, cmin / cmax)
    q = eps * cmin * (physics.hot_inlet_C - tin)
    hot_out = physics.hot_inlet_C - q / ch
    cold_out = tin + q / cc
    residual = ch * (physics.hot_inlet_C - hot_out) - cc * (cold_out - tin)
    return ThermalResult(q, hot_out, cold_out, eps, residual, hot_out <= physics.target_C)


def temperatures(physics: Physics, inputs):
    """Batch adapter. Column order is A [m2], m_c [kg/s], T_c,in [C]."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError("Expected an (n, 3) array: area, cold flow, cold inlet")
    return forward(physics, inputs[:, 0], inputs[:, 1], inputs[:, 2]).hot_out_C
