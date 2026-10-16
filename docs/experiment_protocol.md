# Frozen starter experiment

The user approved simulated data and route A (thermal design comparison) on
2026-10-16. The bounded protocol below was selected by the implementation assistant
and saved in configuration before running the first experiment. No claim is made
that the user independently designed these hyperparameters.

- Inputs: equivalent active outside area 10-40 m2, water mass flow 2-12 kg/s,
  water inlet 28-35 C; independent uniform draws. Hot conditions and U/Cp fixed.
- Independently seeded train/validation/test sets: 4000/1000/1000 rows.
- Independent boundary test: 1000 reference-labelled samples within 1 C of the
  55 C threshold, by predeclared rejection sampling. It is not population-weighted.
- Extrapolation diagnostic: 1000 points with water inlet 36-38 C, the other
  dimensions within training bounds. No reliability claim beyond this test.
- Linear regression baseline; one tanh MLP (32,32), L-BFGS, 1500 iteration limit,
  alpha=1e-4, three initialization seeds. All fit warnings and iteration counts saved.
- Fit input and target standardizers on training only. Select the MLP seed by
  validation MAE. Report all three seeds' final scores as seed sensitivity, not as
  independent datasets or confidence intervals. No final-test model selection.
- Margin: max of zero, the validation 99th percentile of reference-minus-prediction,
  and the maximum optimistic error at the raw surrogate's validation-design choices.
  Those choices use water inlets 29/33/35 C and weights 0.25/0.5/0.75. This is a
  heuristic validation-calibrated guard, not a coverage or plant-safety guarantee.
- Final design cases: water inlets 30/34 C crossed with the same three weights,
  giving six distinct scenario/objective pairs. They were not used for calibration.
  Training may include nearby water temperatures; this is decision-level holdout,
  not a claim of unseen temperature intervals.
- Every design uses the same 101x101 candidate grid. Objective:
  `J = w A/A_reference + (1-w) m_c/m_reference`. Reference quantities are the
  report-based effective area and water flow. J is dimensionless, not cost.
- Compare direct search, raw surrogate choice followed by an audit, and a
  margin-screened ranked search that verifies candidates until one passes.
  Retain every rejected proposal in `candidate_audits.csv`. Empty results stay empty.
- Report objective gaps only for reference-feasible designs, relative to the best
  feasible point on the identical finite grid. This is not continuous global optimality.
- Prediction metrics include MAE, RMSE, absolute-error p95, max optimistic error,
  and false acceptance count / predicted-feasible count. A zero denominator is null.
- Timing: one BLAS thread, three warmups, eleven repeats; both single calls and
  matched 10000-row batches. Full search includes grid construction, scoring,
  model evaluation, selection and verification. Data generation, all model fitting
  and margin calibration are reported separately. A direct algebraic baseline is
  deliberately retained in its vectorized form.

`frozen_config.json` is written before sampling. `frozen_selection.json` is written
before final test evaluation. Configuration/source hashes, package versions, seeds,
raw predictions, raw candidate audits and individual timing observations are saved.
The predictions CSV joins to dataset CSV by the explicit `(split, row_in_split)`
key. Guarded classification metrics use prediction plus margin as a decision score;
they do not relabel that score as a new temperature prediction. No serialized
executable model object is required; rerun to refit.

Data-splitting and preprocessing follow the
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
