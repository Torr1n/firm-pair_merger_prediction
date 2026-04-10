# Week 2 Implementation Instance Summary

**From**: Claude Code (Week 2 implementation instance)
**To**: Next Claude Code instance (Week 2 interpretation + design revision)
**Date**: 2026-04-09
**Repository**: `github.com:Torr1n/firm-pair_merger_prediction.git` branch `master`
**HEAD at handover**: `74f76cc` (Fix float32 cancellation in GMM fitting; check in sweep ops scripts)

---

## Mission and Outcome

This instance was tasked with executing the **Week 2 implementation phase**: committing the design work from the prior instance, writing and deploying the K_max convergence sweep to AWS, and preparing analysis artifacts. The sweep was the critical experiment that would determine whether the firm-portfolio representation is a stable scientific object or a sensitivity-conditioned approximation.

**The sweep completed. The answer is clear: NOT CONVERGED by K_max=30.**

Bulk rankings are extremely stable (Spearman ρ ≈ 0.99), but the top-50 pair overlap collapses from 80% (K=10→15) to 0% (K=15→20, K=20→25) and 6% (K=25→30). The specific firm pairs flagged as "most technologically similar" are materially sensitive to K_max. This is not a failure — it is a substantive methodological finding that reshapes how Week 3 must report results.

---

## What Was Accomplished

### Phase 1: Design Commit (commit `26b51ec`)
- Committed all design work from the prior instance: 4 ADRs (004-007), firm portfolio spec, Bayesian GMM audit, Codex review handoff, config updates, CLAUDE.md updates, instance handover docs
- Added `.gitignore` entries for terraform state and notebook images
- 21 files changed, 3,304 insertions

### Phase 2: K_max Sweep Script (commits `4cbbdec` through `b451067`)
- Wrote `scripts/run_kmax_sweep.py` (1,023 lines) — a self-contained AWS script that:
  - Loads 50D patent vectors + gvkey map
  - Computes global EB priors from unique patent matrix (not grouped — Codex Major #3)
  - Groups ~15,696 firms into 3 tiers per ADR-005
  - Fits Bayesian GMMs at K_max ∈ {10, 15, 20, 25, 30} for all non-excluded firms
  - Computes pairwise BC for ALL non-excluded firm pairs (~31.6M per K_max)
  - SG-vs-SG block computed once and reused (K_max-invariant optimization)
  - Vectorized BC via numpy broadcasting (12x faster than naive loops, numerically verified)
  - Convergence metrics: Spearman ρ, Kendall τ, top-k overlap, per-firm NN-5 stability
  - Persistent-stability convergence verdict (not first-passing-pair)
- Deployment handoff written for Codex (`kmax_sweep_deployment_handoff.md`)
- **Two rounds of Codex review** addressed:
  - Round 1: 2 Major (BC scope must include all non-excluded firms; convergence must be persistent stability) + 2 Minor (read K_max from config; excluded_firms.csv spec compliance)
  - Round 2: Prose consistency pass (stale references to GMM-tier-only, runtime estimates)

### Phase 3: Pre-Registration (commit `36fe19b`)
- Wrote `notebooks/03_kmax_convergence_analysis.ipynb` skeleton — 43 cells (26 markdown + 17 code), 7 sections with 11 planned visualizations and TODO markers
- Wrote `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` skeleton — one-page memo with `[RESULT]` placeholders
- Both committed BEFORE the sweep launched, locking the narrative against post-hoc rationalization
- Pre-registered decision framework with two branches (converged vs not-converged)
- Deep-dive firms identified by patent-title pattern matching: IBM (006066), Intel (012141), Qualcomm (024800), Google/Alphabet (160329), Cisco (020779)

### Phase 4: Production Bug Fix (commit `74f76cc`)
- First sweep run (20260409T080607Z) crashed deterministically in `fit_single_gaussian`
- **Root cause**: sklearn's diagonal variance formula `E[X²] - mean² + reg_covar` suffers catastrophic cancellation in float32 when means are large (3-10) and per-dim variance is small (0.02). Produces tiny negative values that crash `_compute_precision_cholesky`
- **Scope**: 342 of 6,304 single-Gaussian firms affected (5.4%)
- **Fix**: cast inputs to float64 + bump `reg_covar` from 1e-6 to 1e-4 (either alone fixes 100% of cases; both for defense in depth)
- **Regression test**: 6 tests including a real production fixture (`degenerate_firm_actual.npy`, gvkey=012257) that deterministically reproduces the old bug. Sanity check verifies fixture actually crashes sklearn under old config.
- Also checked in Codex's operational scripts: `start_kmax_sweep.sh`, `watch_sweep_and_shutdown.sh`, `user_data_kmax_sweep.sh`
- All 51 unit tests pass

### Phase 5: Sweep Execution on AWS
- Deployed by Codex to c5.4xlarge
- Run ID: `20260409T170706Z`
- Runtime: 10,227.5s = 2.84 hours (faster than estimated 8-14 hours — likely because SG-vs-SG caching was very effective)
- Instance terminated correctly via watchdog
- All artifacts synced to S3: `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/`

---

## The Sweep Results

### Verdict: NOT CONVERGED by K_max=30

The pre-registered persistent-stability rule (ρ > 0.95 AND top-50 overlap > 80% from K* onward) was not satisfied at any K_max value.

### Bulk stability is excellent

| Transition | Spearman ρ | Kendall τ |
|---|---|---|
| 10 → 15 | 0.9905 | 0.9608 |
| 15 → 20 | 0.9920 | 0.9684 |
| 20 → 25 | 0.9912 | 0.9688 |
| 25 → 30 | 0.9926 | 0.9733 |

All ρ values exceed 0.99. Rank correlation across 31.6M firm pairs is near-perfect.

### Top tail is NOT stable

| Transition | Top-50 overlap | Top-100 overlap | Top-200 overlap | Mean NN-5 overlap |
|---|---|---|---|---|
| 10 → 15 | **80.0%** | ? | ? | ? |
| 15 → 20 | **0.0%** | ? | ? | ? |
| 20 → 25 | **0.0%** | ? | ? | ? |
| 25 → 30 | **6.0%** | ? | ? | ? |

The 0% overlap at 15→20 is the most striking finding. It means **none** of the top-50 pairs at K_max=15 appear in the top-50 at K_max=20. This is what prompted Codex's investigation hypothesis: is this near-tie instability, model misspecification, or both?

### Effective K keeps rising (but decelerating)

| K_max | Mean K | Ceiling rate |
|---|---|---|
| 10 | 8.04 | 34.5% |
| 15 | 10.18 | 11.7% |
| 20 | 11.68 | 3.0% |
| 25 | 12.78 | 0.7% |
| 30 | 13.67 | 0.2% |

K is decelerating (ceiling rate drops from 34% to 0.2%) but not plateauing. Mean K at K_max=30 is 13.67 — the model is still finding meaningful structure at finer granularity, even though very few firms are hitting the ceiling.

---

## What Was NOT Done

### Artifacts not yet populated (ready for immediate work)
- `notebooks/03_kmax_convergence_analysis.ipynb` — skeleton committed, needs sweep data loaded and all 11 visualizations rendered
- `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` — skeleton committed, needs `[RESULT]` placeholders filled
- Sweep artifacts are on S3 but NOT pulled locally yet (the `aws s3 sync` was interrupted during handover)

### Diagnostics not yet run (highest priority for next instance)
Codex outlined a four-step diagnostic sequence to investigate the 0% top-50 overlap:

1. **Tail-margin analysis**: Are the top-50 BC values densely packed with near-ties, making rank order fragile? Or are there clear gaps between pairs? Check BC value distributions at ranks 1-50, 50-100, 100-200. If near-ties dominate, some instability is ranking noise, not model failure.

2. **Robust-core vs volatile-fringe**: Which specific pairs persist across ALL K_max values? Is there a stable nucleus surrounded by churning candidates? If yes, the stable core is the actionable output.

3. **Link to PPC failures**: The design phase found persistent PPC failures for mid-sized firms (100-500 patents). Are the firms driving the tail churn the same ones? If yes, this is the cleanest low-cost evidence that ranking instability comes from model inadequacy.

4. **Targeted alternative-model test (only if needed)**: On 20-50 unstable firms, test a richer local density approximation. If unstable firms become stable → strong evidence of Gaussian/diagonal misspecification. If not → intrinsic ambiguity in the firm similarity landscape.

### Design work not yet done
- ADR-004 needs to be reopened and rewritten around the actual result
- Portfolio spec needs updating for the not-converged branch
- Week 3 contract needs to be defined (robust vs model-sensitive pair classification)
- No PortfolioBuilder or GMMFitter implementation code exists (`src/portfolio/` is empty)

### Implementation (TDD) not started
- Tests not written for PortfolioBuilder or GMMFitter
- Production GMM fitting script not written
- Validation notebook (04_portfolio_validation.ipynb) not started

---

## Key Decisions and Their Reasoning

### Why all non-excluded firms in BC analysis (not just GMM-tier)
Codex's first Major finding on the sweep script: Week 3 compares all non-excluded firms, so the convergence study must include single-Gaussian firms. Even though SG firms are K_max-invariant (K=1 always), they participate in the ranking universe because the GMM-tier firm they're compared against changes with K_max. A focused small firm becoming or stopping being a top neighbor when a larger firm's K_max changes is exactly the kind of instability we need to detect.

### Why persistent stability (not first-passing pair)
Codex's second Major finding: convergence means "raising K_max further stops mattering in practice." If 15→20 passes but 20→25 fails, that is not convergence at 20. K* must be the smallest K_max where ALL subsequent adjacent comparisons pass.

### Why pre-register the analysis
Torrin explicitly endorsed writing the notebook skeleton before seeing results. The narrative is driven by the research question, not reverse-engineered from the output. This is the methodology equivalent of pre-registration. The decision rule, convergence thresholds, and both branches (converged/not-converged) were locked before the sweep launched.

### Why float64 + reg_covar=1e-4 (not just one)
342 firms crash in float32. Float64 alone fixes all 342. reg_covar=1e-4 alone also fixes all 342. We apply both because: (a) defense in depth against any remaining edge case, (b) reg_covar=1e-4 is still 200x smaller than the smallest true per-dim variance in the data, so the bias is negligible, (c) sklearn's own error message recommends both.

### Why the supervisor pitch matters
This sweep is not a hyperparameter tuning exercise. It is a methodology resolution run. The framing — "replace an arbitrary default with an empirically justified one, or formally classify the signal as sensitivity-conditioned" — is what makes the non-convergence result publishable rather than deflating. The next instance must carry this framing forward.

---

## What I'm Confident In

1. **The sweep script is correct.** Independently audited (mathematical correctness of BC formula, global priors, co-assignment handling, convergence metrics). All 51 unit tests pass. Real production fixture reproduces and verifies the bug fix.
2. **The sweep ran cleanly.** Codex confirmed: instance terminated correctly, artifacts synced, no hanging resources.
3. **Bulk stability is real.** Spearman ρ ≈ 0.99 across ALL adjacent transitions. The distributional overlap measure captures genuine technological structure.
4. **The pre-registration is sound.** Notebook and executive summary are well-structured for the not-converged branch. Decision framework is clear.
5. **The firm identifications are correct.** IBM (006066), Intel (012141), Qualcomm (024800) confirmed via patent title patterns with high confidence.

## What I'm NOT Confident In

1. **The 0% top-50 overlap at 15→20.** This is an extraordinary claim that needs verification. Codex flagged it: "ρ≈0.99 with top-50 overlap near 0% is plausible, but important enough to verify explicitly." Possible explanations: (a) near-tie ranking noise — the top-50 BC values may be densely packed, (b) genuine model misspecification — diagonal Gaussians can't capture the fine structure that determines extreme-tail rankings, (c) a bug in the top-k pair construction or tie handling. The diagnostics are designed to distinguish these.

2. **Whether this is fixable within the GMM family.** The design phase already showed that full covariance was 62% WORSE on PPC (parameter budget forces K=2), PCA post-processing didn't help, and normalization doesn't matter. If the diagnostics confirm model inadequacy, the available options within GMM (richer covariance, more components) have already been shown to not help. The STAT 405 audit recommended t-mixtures as the next defensible upgrade, but only if >20% of clusters show significant non-Gaussianity.

3. **Whether the notebook visualizations will render correctly on first try.** The code cells are untested (skeleton only). The next instance will need to debug rendering issues when populating with real data. Some cells import from `scripts.run_kmax_sweep` which requires careful `sys.path` setup.

---

## What the Next Instance Must Do (Priority Order)

### Immediate (before any design revision)

1. **Pull sweep artifacts from S3** to `output/kmax_sweep/`:
   ```bash
   aws s3 sync s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/output/kmax_sweep/ output/kmax_sweep/ --profile torrin
   ```

2. **Run the four-step diagnostic sequence** (this is the highest-value work):
   - Tail-margin analysis on BC matrices
   - Robust-core vs volatile-fringe analysis
   - Link instability to firm characteristics (size, diversification, PPC status)
   - Only then consider targeted alternative-model test

3. **Populate the pre-registered notebook** (`notebooks/03_kmax_convergence_analysis.ipynb`) with actual data and visualizations

4. **Populate the executive summary** (`docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md`) with actual numbers

### Design revision (after diagnostics)

5. **Reopen ADR-004** and rewrite it around the actual result: BC tail is sensitivity-conditioned through K_max=30. The language should shift from "K_max=15 provisional" to "BC tail is sensitivity-conditioned under mixture complexity; robust vs model-sensitive pair classification is required."

6. **Define the Week 3 contract**: Whether to report BC as a single ranked list with robustness labels, or as a multi-K_max robustness object

7. **Update portfolio spec** for the not-converged path (multi-K_max output as first-class deliverable)

### Implementation (after design revision is approved)

8. **TDD implementation** of PortfolioBuilder and GMMFitter (blocked on design decisions from above)

---

## How This Session Worked

### The Collaboration Pattern (evolved from design instance)

The three-party dynamic (Torrin, Codex, Claude) continued, but the roles shifted:

- **Torrin's role this session**: Strategic direction-setting, not line-level decisions. His key interventions were: (1) insisting on the analysis plan before launching the sweep — "I want to ensure we are aligned on the direction we are heading down... put it this way: you're in my shoes and I'm meeting with my supervisor"; (2) explicitly endorsing the pre-registration approach; (3) bringing Codex's strategic framing into alignment.

- **Codex's role this session**: Evolved from pure code reviewer to strategic methodological advisor. Codex's two most impactful contributions were: (1) the diagnostic sequence for investigating the 0% top-50 overlap (near-tie vs misspecification vs both), and (2) the framing of non-convergence as "a substantive methodological result, not failure."

- **My role this session**: Shifted from design to engineering + analysis planning. I wrote production code (sweep script, operational fixes), ran diagnostics (float32 root cause analysis), and designed the analysis framework (notebook, executive summary, decision framework).

### Torrin's Priorities This Session

The defining moment was when Torrin asked me to **stop and think about the bigger picture** before launching the sweep. His message: "Before we proceed with this run I want to ensure we are aligned on the direction we are heading down... I want to ensure that there is some set of visual or written analysis that is interpretable and communicates the value of this work immediately." This is not a developer asking for more tests — it is a researcher asking for the interpretive framework that gives the technical work scientific meaning.

Torrin's latency tolerance is extremely high. He explicitly said: "My tolerance for response latency is abnormally high: I would far rather handle other business while you run for multiple hours and come back to a detailed and comprehensive job." He values thoroughness and depth over speed. The next instance should take this seriously — do not rush the diagnostics.

### What Codex Taught This Session

Codex's review style is now well-characterized across multiple rounds:
1. **Code reviews**: Precise, file:line references, catches real contract gaps (not just style)
2. **Strategic reviews**: Provides framing that elevates the work from engineering to research ("This run is not 'more fitting.' It is a methodology resolution run.")
3. **Diagnostic guidance**: Prescribes the minimum investigation sequence, ordered by cost/information ratio

### The Analysis Plan Story

The analysis plan went through three iterations:
1. **My initial version**: 7 sections, 11 visualizations, firm deep-dives, connection to Bena & Li (2014)
2. **Codex's additions**: One-page executive summary, pre-registered decision framework, explicit "if converged / if not converged" branches, emphasis on "supervisor pitch" framing
3. **Merged version**: Both were incorporated. The key Codex insight I hadn't captured was the executive summary as a standalone document (not embedded in the notebook) and the concrete decision table for post-sweep action.

### Standards Reinforced

- **Pre-registration**: Analysis structure committed before seeing results. This is now a project pattern.
- **Evidence-based engineering**: The float32 diagnosis was a model of this principle — didn't just bump reg_covar, diagnosed the exact root cause (342 firms, catastrophic cancellation regime, sklearn's unstable variance formula), verified both candidate fixes independently, wrote a regression test with a real production fixture.
- **Halt at gates**: Respected at every checkpoint. Did not start implementation. Did not populate the notebook before results landed. Did not skip Codex review rounds.

---

## File Inventory

### This instance's commits (6 commits, all pushed to master)

| Commit | What |
|---|---|
| `26b51ec` | Week 2 design phase: 4 ADRs, spec, audit, config, docs (from prior instance) |
| `4cbbdec` | K_max sweep script + Codex deployment handoff |
| `3bddfc7` | Codex sweep review fixes: full BC scope, persistent convergence |
| `b451067` | Prose consistency fix: stale references updated |
| `36fe19b` | Pre-register analysis: notebook skeleton + executive summary |
| `74f76cc` | Float32 bug fix + regression tests + ops scripts |

### New files created

| File | Size | Purpose |
|---|---|---|
| `scripts/run_kmax_sweep.py` | 41.8 KB | Main sweep script |
| `scripts/start_kmax_sweep.sh` | 2.2 KB | Sweep launcher (from Codex) |
| `scripts/watch_sweep_and_shutdown.sh` | 4.3 KB | Sweep watchdog (from Codex) |
| `infrastructure/user_data_kmax_sweep.sh` | 1.2 KB | EC2 bootstrap (from Codex) |
| `notebooks/03_kmax_convergence_analysis.ipynb` | ~30 KB | Analysis notebook SKELETON |
| `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` | 5.7 KB | Executive summary SKELETON |
| `docs/epics/week2_firm_portfolios/kmax_sweep_deployment_handoff.md` | 6.5 KB | Codex deployment handoff |
| `tests/unit/test_kmax_sweep_fitters.py` | 5.3 KB | Regression tests (6 tests) |
| `tests/unit/fixtures/degenerate_firm_actual.npy` | 1.5 KB | Real production fixture |
| `docs/epics/instance_handover/week2_implementation_instance_summary.md` | this file | Instance handover |

### Modified files
| File | What changed |
|---|---|
| `src/config/config.yaml` | `k_max_sweep: [10, 15, 20, 25, 30]` (was [10, 15, 20]) |
| `.gitignore` | Added terraform state + notebook images |
| All design files from prior instance | See commit `26b51ec` |

### Uncommitted work
Only `.codex` directory (Codex tool artifact, not tracked).

### S3 artifacts (authoritative sweep results)
```
s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/output/kmax_sweep/
├── firm_gmm_parameters_k10.parquet
├── firm_gmm_parameters_k15.parquet
├── firm_gmm_parameters_k20.parquet
├── firm_gmm_parameters_k25.parquet
├── firm_gmm_parameters_k30.parquet
├── bc_block_sg_vs_sg.npz
├── bc_matrix_all_k10.npz
├── bc_matrix_all_k15.npz
├── bc_matrix_all_k20.npz
├── bc_matrix_all_k25.npz
├── bc_matrix_all_k30.npz
├── convergence_summary.json
├── excluded_firms.csv
├── sweep.log
└── status/
    ├── sweep_status.json
    ├── watchdog_status.json
    └── launch_metadata.json
```

These artifacts need to be pulled to `output/kmax_sweep/` before the notebook can be populated.

---

## Required Reading for Next Instance (In This Order)

1. **This document** — you're reading it
2. `CLAUDE.md` — project instructions, values, architecture
3. `docs/adr/adr_004_k_selection_method.md` — the ADR that needs reopening
4. `notebooks/03_kmax_convergence_analysis.ipynb` — the skeleton you need to populate
5. `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` — the memo you need to populate
6. `scripts/run_kmax_sweep.py` — contains `load_gmm_results()`, BC functions, convergence metrics (reuse these in the notebook)
7. `docs/specs/firm_portfolio_spec.md` — the interface contracts that need updating
8. `docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md` — the STAT 405 audit (relevant if pursuing misspecification hypothesis)

---

## The Intellectual State of the Project

The project is at an inflection point. The first two weeks established that:

1. **Patent embeddings capture meaningful technological relationships** (Week 1 — high confidence)
2. **Firm portfolios can be represented as GMMs over patent vectors** (Week 2 design — high confidence for bulk, uncertain for tail)
3. **The distributional overlap measure (BC) works for broad similarity** (sweep — confirmed, ρ ≈ 0.99)
4. **But the extreme tail is model-sensitive** (sweep — confirmed, top-50 overlap collapses)

The open question is whether finding #4 is:
- **Near-tie ranking noise**: The top-50 BC values are densely packed, so tiny fit changes reshuffle the order even though the underlying similarity landscape is nearly the same → addressable by threshold-based reporting instead of fixed top-k
- **Genuine model misspecification**: The diagonal Gaussian mixture can't capture the fine density structure that determines extreme-tail rankings → requires model upgrade (t-mixtures, full covariance in subspace, KDE comparison)
- **Both**: Some near-tie noise + some structural inadequacy → report robust core + flag volatile fringe + acknowledge limitation

The next instance's diagnostic work will distinguish these. The answer determines whether Week 3 can proceed with the current model (+ robustness labels) or needs a design pivot.
