"""Recalculate the printed Appendix B basis without silently repairing it."""

import json
import math
from pathlib import Path


def recalculate(path="configs/report_appendix_b.json"):
    p = json.loads(Path(path).read_text(encoding="utf-8"))
    mh, mc = p["hot_mass_flow_kg_s"], p["cold_mass_flow_kg_s"]
    cph, cpc = p["hot_cp_J_kg_K"], p["cold_cp_J_kg_K"]
    thi, tho = p["hot_inlet_C"], p["hot_design_outlet_C"]
    tci, tco = p["cold_inlet_C"], p["cold_design_outlet_C"]
    do, di, pitch = p["tube_outer_diameter_m"], p["tube_inner_diameter_m"], p["tube_pitch_m"]
    nt, passes, length = p["tube_count"], p["tube_passes"], p["tube_length_m"]
    ds, b = p["shell_diameter_m"], p["baffle_spacing_m"]
    mus, mut = p["hot_viscosity_Pa_s"], p["cold_viscosity_Pa_s"]
    ks, kt = p["hot_conductivity_W_m_K"], p["cold_conductivity_W_m_K"]
    area = nt * math.pi * do * length
    effective_area = nt * math.pi * do * (length - 2 * p["dead_length_per_end_m"])
    de = 4 * (pitch**2 - math.pi * do**2 / 4) / (math.pi * do)
    at = nt / passes * math.pi * di**2 / 4
    ass = ds * (pitch - do) * b / pitch
    ret, res = mc * di / (at * mut), mh * de / (ass * mus)
    hi = 0.023 * ret**0.8 * (cpc * mut / kt)**0.4 * kt / di
    ho = 0.36 * res**0.55 * (cph * mus / ks)**(1 / 3) * ks / de
    resistance = (1 / ho + do * math.log(do / di) / (2 * p["wall_conductivity_W_m_K"])
                  + do / di / hi + p["shell_fouling_m2_K_W"] + p["tube_fouling_m2_K_W"])
    u = 1 / resistance
    dt1, dt2 = thi - tco, tho - tci
    lm = (dt1 - dt2) / math.log(dt1 / dt2)
    r, pp = (thi - tho) / (tco - tci), (tco - tci) / (thi - tci)
    s = math.sqrt(r*r + 1)
    f = s / (r - 1) * math.log((1 - pp) / (1 - pp*r)) / math.log((2/pp - 1 - r + s) / (2/pp - 1 - r - s))
    qh, qc = mh * cph * (thi - tho), mc * cpc * (tco - tci)
    f1 = 0.0076 + 0.000166 * (ds / 0.0254) * res**(-0.125)
    f2 = 0.0016 + 0.000058 * (ds / 0.0254) * res**(-0.157)
    return {
        "config_name": p["config_name"],
        "hot_mass_flow_kg_s": mh,
        "hot_required_W": qh,
        "cold_design_W": qc,
        "design_energy_mismatch_W": qh - qc,
        "report_average_required_W": (qh + qc) / 2,
        "gross_area_m2": area,
        "effective_area_m2": effective_area,
        "LMTD_K": lm,
        "F_at_design_endpoints": f,
        "tube_Re": ret,
        "shell_Re": res,
        "tube_h_W_m2_K": hi,
        "shell_h_W_m2_K": ho,
        "overall_U_W_m2_K": u,
        "capacity_at_design_endpoints_gross_W": u * area * f * lm,
        "capacity_at_design_endpoints_effective_W": u * effective_area * f * lm,
        "printed_shell_friction_audit": {
            "f1_recalculated": f1, "f1_reported": 0.00520571,
            "f2_recalculated": f2, "f2_reported": 0.00107650,
            "fs_recalculated": 144*f1 - 1.25*(1-b/ds)*(f1-f2),
            "fs_reported": 0.479125,
            "status": "unresolved; excluded from design feasibility",
        },
        "limitations": p["limitations"],
    }
