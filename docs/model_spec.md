# E-101 thermal model v1

## Question and scope

Within a declared thermal idealization, which combinations of active outside
heat-transfer area A and cooling-water mass flow m_c achieve T_h,out <= 55 C?
Compare direct finite-grid search with a learned surrogate and verified screening.
This is a cooling-capability criterion, not the original process's complete
constant-55-C control requirement.

One TEMA E shell pass and an even number of tube passes represent the report's
one-shell/four-pass exchanger. The model is steady, single phase, constant Cp,
constant effective U, without ambient heat loss. Water inlet temperature is an
external scenario, never a variable the design optimizer can select.

## Inputs and outputs

`configs/project.json` is the versioned source of project inputs and experimental
choices. Columns are A [m2], m_c [kg/s], T_c,in [C]. The hot flow is exactly
6000/3600 kg/s at 110 C. Hot/cold Cp are 2862.9/4180 J/(kg K).
U = 282.6992163028217 W/(m2 K) comes from recalculating Appendix B.7 as printed.
Baseline active area is 27.547012457342014 m2, with 8.971 kg/s water at 33 C.

Outputs: exchanged heat [W], both outlet temperatures [C], effectiveness,
energy-balance residual [W], and the declared thermal feasibility flag.
Negative area, nonpositive flow, nonfinite values and reversed inlet temperature
ordering are rejected. Zero area/U and equal inlet temperatures are supported limits.

## Equations

Let C_h = m_h Cp_h, C_c = m_c Cp_c, C_min = min(C_h,C_c),
C_r = C_min/max(C_h,C_c), and NTU = UA/C_min.
For one shell pass and even tube passes, define s = sqrt(1+C_r^2) and
t = tanh(NTU s/2). Then:

```
effectiveness = 2 t / ((1 + C_r) t + s)
Q             = effectiveness C_min (T_h,in - T_c,in)
T_h,out       = T_h,in - Q/C_h
T_c,out       = T_c,in + Q/C_c
```

The tanh form avoids exponential overflow and the zero-NTU division in a coth
form. C_r=1 does not require a special division. The independent `ht` implementation
and a separate F-LMTD root calculation are used in tests.

Sources: [ht shell-and-tube effectiveness relation](https://ht.readthedocs.io/en/latest/ht.hx.html#ht.hx.effectiveness_from_NTU),
PDC Appendix B pp.59-64. The independent library is a numerical cross-check,
not experimental evidence for the actual equipment.

## Report reproduction versus prediction

`report_appendix_b.json` uses the report's rounded 1.667 kg/s hot flow, the stated
55/40 C design outlets, original heat-transfer correlations and fouling convention.
It recomputes heat balances, geometry, h_i/h_o, U, F and LMTD. Its
`capacity_at_design_endpoints_*` fields are capacity checks, not the forward-model Q.
The report's two heat balances differ by about 6.47 W; that residual is retained.
The report averages the two duty values. The forward model instead enforces one Q.

The report's gross area is 28.0373 m2. The effective area deducts an explicitly
provisional 32 mm at each tube end. Varying A in v1 means varying equivalent active
area; it does not select a buildable tube count, diameter, length or shell geometry.

## Limitations retained from the source

- The external property references were not independently revalidated. These are
  report constant-property inputs, not a fitted ternary mixture property model.
- Appendix B.7 adds the tube fouling term directly; its area basis is unresolved.
  Reproduction preserves that convention rather than silently correcting it.
- Appendix B.8.3 printed shell friction equations do not reproduce their reported
  values. Pressure drop is not included in v1 feasibility or operating-cost claims.
- Changing water flow while retaining U is a simplifying conditional model.
  Flow/geometry-dependent coefficients require a separately validated model version.
- No calibrated area-cost, pump-power or makeup-water model is available here.
  The optimization score is dimensionless, not currency, electricity or fresh-water use.

## Sources and privacy

R1: [PDC final report](references/PDC_Final_report.pdf), CHEE2050 group course report, 104 PDF pages.
SHA256: `a85715d57269ea4b4cd7f9722107100a277f33f1e1119bfa53ac83cb71ae67fd`.
Page numbers include the cover. The PDF is not required to execute the extracted
baseline. Its unchanged original is included at the user's explicit request on
2026-10-16; see the [source note](references/README.md).

R2: user-supplied `Project3_PDC_Report_Supplement_ZH.md` (2026-10-16).
SHA256: `be733944fbdf58943bd2c207c140299344da4b39cfe55ed0f24e57c9f1a61712`.
Its technical suggestions are secondary context; adopted numerical inputs were
checked against R1 and arithmetic was rerun. See [sources.csv](sources.csv).
