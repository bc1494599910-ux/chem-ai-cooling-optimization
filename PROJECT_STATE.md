# Project state

Updated: 2026-10-16, Asia/Shanghai.

## Approved goal

Small GitHub-ready starter project: physics-generated data, route A comparing
area/water-flow combinations meeting hot outlet <=55 C. User instructed implementation.

## Current work

- Implemented report reproduction, constant-U one-shell thermal model, grid search,
  independent synthetic splits, linear/MLP comparison, candidate auditing and CLI.
- Working/delivery branch: `codex/e101-starter`; original planning workflow
  preserved unchanged locally and excluded from publication with `.gitignore`.
- Dependency environment created with `uv sync --locked --extra dev` (Python 3.12.14).
- `python -m pytest -q`: 40 passed after correcting a numerical test tolerance.
  The initial strict monotonicity assertion detected 1.42e-14 C rounding at an
  asymptote; a documented, temperature-scaled floating-point tolerance is now used.
- `python -m cooling_project report` and `predict`: completed successfully.
- `python -m cooling_project experiment --output results/baseline`: completed,
  exit 0, approximately 9.93 seconds after installation. Selected MLP seed 27 by
  validation MAE; final in-range MAE 0.014867631 C. No training convergence warnings.
- Three generated figures visually inspected: design/Pareto, errors, timing.
- Created a separate `.venv-verify` from the same lockfile; its test suite also
  passed 40 tests. Full rerun to `runs/reproduction` completed with exit 0.
- `python scripts/verify_results.py results/baseline runs/reproduction`: exit 0;
  8000 unique input rows, 12000 predictions, 76 candidate audits verified in both
  runs, with identical numerical metrics, selected model, designs and data.
- `python -m cooling_project search --output runs/search`: exit 0;
  10201 grid points, 4846 thermally feasible points, 33 finite-grid Pareto points
  at 33 C cooling-water inlet.
- README `uv run --locked --extra dev` prediction commands also completed with
  exit 0: area 20/30 m2 at 8.971 kg/s and 33 C yielded hot outlets
  58.85680995/49.33507834 C respectively. All 21 README local links resolve;
  `git diff --check` passed. The final timing figure was visually rechecked.
- Output-only clarification after the initial run: added explicit dataset row IDs
  and separated guarded classification from raw regression metrics. The original
  run was preserved under `runs/initial-result-before-output-clarification`.
  Configuration, model choices, samples, training and success criteria were not tuned.

## Results and entry points

- README.md: runnable English overview with measured results and plots.
- docs/methods_zh.md: Chinese explanation and a small user exercise.
- docs/results_zh.md: interpretation, limitations and observed decision errors.
- results/baseline/: full evidence; results/verification.json: reproduction record.
- The raw MLP route was reference-feasible in 4/6 design cases; margin plus
  verification achieved 6/6 in this small test. Direct physics was faster.

## Boundaries

No real-plant validation, variable-U model, hydraulic feasibility, monetary cost
or dynamic control has been performed. User learning exercises have not yet been
performed. The starter implementation and its local reproduction are complete.

## GitHub delivery preparation

The user explicitly requested original-PDF inclusion, repository preparation,
commit and push on 2026-10-16. The origin repository was checked as private, with
push permission available. Delivery targets the existing `codex/e101-starter`
branch. The default `main` branch was still at the initial commit during preparation.

- Original PDF copied unchanged to `docs/references/PDC_Final_report.pdf`;
  SHA-256 `a85715d57269ea4b4cd7f9722107100a277f33f1e1119bfa53ac83cb71ae67fd`
  matches the supplied source. Source links and attribution updated.
- `.gitattributes` keeps executable-source LF endings stable and preserves the
  exact bytes of saved results, including CSV CRLF endings used by existing hashes.
- GitHub Actions workflow added for tests, evidence verification and PDF integrity.
- Historical run manifests retain their original pre-commit metadata and hashes;
  they are not rewritten to pretend the experiment ran after the delivery commit.
- Check the branch's Git history and GitHub Actions for the current delivery status.
