# Working scope

The user approved implementation of a small, reproducible starter project on
2026-10-16: use a physical E-101 model to generate synthetic data, compare
area/water-flow designs meeting hot outlet <= 55 C, and compare a small surrogate.
Routine implementation, bounded configured experiments, tests and fixes are in scope.

- Preserve unrelated changes and the original local `Project3_AI_Workflow.md`
  (ignored as planning context, not deleted or rewritten).
- Keep report reproduction (1.667 kg/s, gross/effective area separately) distinct
  from project nominal conditions (exactly 6000/3600 kg/s, active outside area).
- Use the one-shell/even-tube-pass thermal model. U and Cp are constant in v1;
  changing water flow does not update U. Record this limitation explicitly.
- Never label synthetic data as plant measurements or a thermal screen as a full
  equipment design, hydraulic approval, constant-temperature controller or cost saving.
- Keep the printed Appendix B fouling convention and unresolved hydraulic issues visible.
- Preserve the direct, vectorized physical baseline. Include offline training costs,
  failed candidates and test denominators. Never tune on the final tests.
- Put exploratory runs in `runs/`; do not overwrite existing results. Save the
  experiment configuration before sampling/training and maintain result provenance.
- The user explicitly requested inclusion of the original course PDF on 2026-10-16.
  Keep `docs/references/PDC_Final_report.pdf` unchanged and its SHA-256 recorded.
  Other unrelated private material stays outside tracked files. The source course
  design was group work; document the new user's and AI's actual work.
- Update `PROJECT_STATE.md` with actual commands, results and remaining work.
- Commit and push to the existing repository were explicitly requested on 2026-10-16.
- Preserve `.gitattributes`: executable sources use LF; saved results retain their
  original bytes so the published manifests remain verifiable after Git checkout.

Run from the repository root: `uv sync --locked --extra dev`, then
`uv run --extra dev python -m pytest -q`.
