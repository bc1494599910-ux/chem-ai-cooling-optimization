# Decisions, 2026-10-16

| Decision | Basis / owner |
|---|---|
| Deliver a small runnable GitHub-ready project for a prospective supervisor | User stated goal; finish promptly without a hard deadline |
| Generate learning data with a physical model | User explicitly accepted |
| Route A: compare area and water-flow designs meeting outlet <=55 C | User selected A and instructed implementation |
| Use E-101 report parameters | User supplied the original PDF and Chinese supplement for reference |
| Retain one shell / four tube passes in the thermal idealization | Assistant source audit: PDF pp.29/60; independent relation checked |
| Use exact nominal mass flow and effective area for project baseline | Assistant implementation convention; report-reproduction configuration remains separate |
| Constant U and Cp in v1 | Scope choice proposed in the discussion; limitation explicit in model specification |
| One MLP configuration, three seeds, linear baseline | Assistant bounded experimental choice; saved before running tests |
| Dimensionless area/flow trade-off; no currency cost | Available material does not establish a reliable variable-cost function |
| Keep pressure drop and dynamic control outside v1 | Thermal screening does not depend on unresolved report hydraulic equations |
| Preserve original user workflow file locally | Existing planning material; kept unchanged and ignored rather than published |
| Include original PDF and commit/push project | User explicitly requested this on 2026-10-16; copied unchanged, hash checked |
| Delivery branch | Existing `codex/e101-starter`; origin checked, repository private and push permission available |

The report's group design, source arithmetic verification, software validation,
surrogate results and validation against a real plant are different evidence levels.
Only the levels actually performed are reported as completed.
