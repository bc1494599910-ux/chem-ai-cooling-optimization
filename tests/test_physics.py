from dataclasses import replace
import math

import numpy as np
import pytest
from ht.hx import effectiveness_NTU_method, effectiveness_from_NTU
from scipy.optimize import brentq

from cooling_project.config import Physics
from cooling_project.physics import forward, one_shell_effectiveness
from cooling_project.report_case import recalculate


@pytest.mark.parametrize("ntu", [0.0, 1e-9, .1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("cr", [0.0, .2, .999999999, 1.0])
def test_independent_ht_effectiveness(ntu, cr):
    expected = 0 if ntu == 0 else effectiveness_from_NTU(ntu, cr, subtype="S&T")
    assert float(one_shell_effectiveness(ntu, cr)) == pytest.approx(expected, abs=1e-8)


def test_reference_differential_over_experimental_domain():
    p = Physics()
    rng = np.random.default_rng(13)
    for area, flow, water in zip(rng.uniform(10, 40, 30), rng.uniform(2, 12, 30), rng.uniform(28, 35, 30)):
        result = forward(p, area, flow, water)
        independent = effectiveness_NTU_method(mh=6000/3600, mc=flow, Cph=p.hot_cp_J_kg_K,
                                               Cpc=p.cold_cp_J_kg_K, subtype="S&T", Thi=110, Tci=water,
                                               UA=p.overall_U_W_m2_K*area)
        assert float(result.hot_out_C) == pytest.approx(independent["Tho"], abs=1e-10)
        assert float(result.cold_out_C) == pytest.approx(independent["Tco"], abs=1e-10)


def test_report_basis_and_independent_lmtd_root():
    r = recalculate()
    assert r["hot_required_W"] / 1000 == pytest.approx(262.485, abs=.0005)
    assert r["report_average_required_W"] / 1000 == pytest.approx(262.488, abs=.0005)
    assert r["overall_U_W_m2_K"] == pytest.approx(282.699, abs=.0005)
    assert r["capacity_at_design_endpoints_gross_W"] / 1000 == pytest.approx(315.112, abs=.0005)
    assert r["effective_area_m2"] == pytest.approx(27.5470, abs=.00005)
    p = replace(Physics(), hot_mass_flow_kg_h=1.667*3600)
    ch, cc = 1.667*p.hot_cp_J_kg_K, 8.971*p.cold_cp_J_kg_K

    def residual(q):
        tho, tco = 110-q/ch, 33+q/cc
        d1, d2 = 110-tco, tho-33
        lm = (d1-d2)/math.log(d1/d2)
        rr, pp = (110-tho)/(tco-33), (tco-33)/77
        s = math.sqrt(rr*rr+1)
        f = s/(rr-1)*math.log((1-pp)/(1-pp*rr))/math.log((2/pp-1-rr+s)/(2/pp-1-rr-s))
        return q-r["overall_U_W_m2_K"]*r["gross_area_m2"]*f*lm

    root = brentq(residual, 250000, 300000)
    result = forward(p, r["gross_area_m2"], 8.971, 33)
    assert float(result.heat_W) == pytest.approx(root, abs=1e-6)
    assert float(result.hot_out_C) == pytest.approx(50.77989360077447, abs=1e-9)
    # Project's exact nominal flow must not be silently rounded to the report's flow.
    assert Physics().hot_mass_flow_kg_s == 6000/3600
    assert Physics().hot_mass_flow_kg_s != 1.667


def test_energy_conservation_and_physical_bounds():
    p = Physics()
    area = np.linspace(0, 1000, 500)
    result = forward(p, area, 3.5, 33)
    assert np.max(abs(result.energy_residual_W)) < 1e-8
    assert np.all((result.effectiveness >= 0) & (result.effectiveness <= 1))
    assert np.all((result.hot_out_C >= 33) & (result.hot_out_C <= 110))
    assert np.all((result.cold_out_C >= 33) & (result.cold_out_C <= 110))
    # At the asymptote, rounding can shift a temperature by a few ULPs.
    assert np.all(np.diff(result.hot_out_C) <= 16 * np.finfo(float).eps * p.hot_inlet_C)
    assert result.heat_W[0] == 0
    assert float(forward(replace(p, overall_U_W_m2_K=0), 25, 8, 33).heat_W) == 0
    assert float(forward(p, 25, 8, 110).heat_W) == 0


@pytest.mark.parametrize("area,flow,tin", [(-1,8,33),(20,0,33),(20,-1,33),(np.nan,8,33),(20,8,np.inf),(20,8,111)])
def test_bad_inputs_rejected(area, flow, tin):
    with pytest.raises(ValueError):
        forward(Physics(), area, flow, tin)


def test_known_report_hydraulic_discrepancy_stays_visible():
    audit = recalculate()["printed_shell_friction_audit"]
    assert audit["f1_recalculated"] > .0076 > audit["f1_reported"]
    assert audit["status"].startswith("unresolved")
