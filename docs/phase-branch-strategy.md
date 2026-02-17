# Phase Branch Strategy

**Purpose:** Keep git history clean and reviewable by implementing Phase-1 through Phase-8 as **separate branches**, each merged to `main` before starting the next phase.

## Order of Phases

| Order | Phase | Scope | Branch name (suggested) |
|-------|--------|--------|--------------------------|
| 0 | Pre-Phase-1 | Schema v0.2/0.3, ModelRegistry, CI/CD, sinks | (already on main) |
| 1 | **Phase-1** | Instrumentation + schema + health flags + dashboard | `phase-1` (or split: phase-1a, phase-1b, phase-1c, phase-1d) |
| 2 | **Phase-2** | Risk scores + layer blame | `phase-2` |
| 3 | **Phase-3** | Prompt fingerprints (vector + prompt_hash) | `phase-3` |
| 4 | **Phase-4** | Early warning (failure_risk, warning_signals) | `phase-4` |
| 5 | **Phase-5** | Health-aware decoding (`should_intervene()`) | `phase-5` |
| 6 | **Phase-6** | Cross-model comparison (dashboard Compare, `corevital compare`) | `phase-6` |
| 7 | **Phase-7** | Human-readable narratives | `phase-7` |
| 8 | **Phase-8** | Dashboard filters, export, packaging extras | `phase-8` |

**Note:** Phase-1 is often split into 1a (schema + metrics), 1b (prompt telemetry), 1c (health flags), 1d (dashboard + sinks). You can either use one `phase-1` branch (all 1a–1d) or four branches `phase-1a` … `phase-1d`, each merged before the next.

## Workflow

1. **Start from `main`.**  
   `git checkout main && git pull origin main`

2. **Create phase branch.**  
   `git checkout -b phase-N` (e.g. `phase-2`).

3. **Implement only that phase’s scope.**  
   Commit in logical chunks (e.g. risk module, report_builder wiring, tests).

4. **Open PR into `main`.**  
   Get review, fix CI, then merge.

5. **Repeat for next phase.**  
   `git checkout main && git pull && git checkout -b phase-N+1`.

## Benefits

- **Clear history:** `git log main` shows a linear story: merge phase-1, merge phase-2, …
- **Easier review:** Each PR is scoped to one phase.
- **Easier rollback:** Revert a single merge if needed.
- **Documentation alignment:** Roadmap and execution plans map to branch names.

## Current State

Today the codebase has **all phases through Phase-8 implemented** on top of Phase-1 (e.g. on a single long-lived branch or unmerged work). To **retrofit** this strategy for future work:

- Treat `main` as the current “everything merged” state.
- For **new** features, create `phase-9`, `phase-10`, etc., or use feature branches that merge to `main` in order.

To **recreate** history as phase branches (optional, only if you want to rewrite history before a big push):

- Use interactive rebase or recreate branches from tagged points; this is invasive and only worth it if you need a clean history for a specific audience. Prefer the “going forward” approach above.

## References

- Phase-1 execution: `docs/phase-1-execution-plan.md`
- Phase-2–8 plan: `docs/phase-2-through-8-execution-plan.md`
- Roadmap: `README.md` (Roadmap section)
