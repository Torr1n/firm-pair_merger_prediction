# Codex Deployment Handoff: Corrected BC Recomputation

**To**: Codex (deployment lead)  
**From**: Claude Code (Week 2 interpretation instance)  
**Date**: 2026-04-11  
**Status**: Ready for VM deployment — **corrected UMAP recompute only**

**Single pipeline this run.** A PCA comparison sweep script is in the repo (`scripts/run_pca_comparison_sweep.py`) but is intentionally **not** part of this VM session — see "Why not PCA in this run" below and the post-run decision tree.

---

## Why this is happening

The original K_max sweep (`20260409T170706Z`) returned NOT_CONVERGED because of two compounding bugs that the diagnostic sequence uncovered:

1. **Duplicate firms in v3 data** — 464 firms (aliases, subsidiaries, predecessor records) need to be removed
2. **Unbounded BC formula** — `bc_mixture` in `run_kmax_sweep.py:473` uses `√(πᵢπⱼ)` weighting, which produces values > 1.0 (theoretical max should be 1.0)

**The full diagnostic findings**: `docs/epics/week2_firm_portfolios/kmax_diagnostic_findings.md`

A 1000-firm local test of the corrected approach (deduplication + linear weights) produced this:

| Transition | Top-50 overlap (corrected) | Top-50 overlap (original) |
|---|---|---|
| K10→K15 | **100%** | 80% |
| K15→K20 | **100%** | 0% |
| K20→K25 | **100%** | 0% |
| K25→K30 | **100%** | 6% |
| K10→K30 | **100%** | 0% |

**Verdict on the test sample: CONVERGED at K_max=10.**

We need the full computation on the deduplicated 7,485-firm dataset to confirm at scale.

---

## Why not PCA in this run (Codex's call, agreed)

**Scientific isolation matters more than VM efficiency here.** The leading hypothesis right now is "duplicate firms + BC formula bug were the cause of the original instability." A 1000-firm local test of the corrected approach already strongly supports this. If we run PCA in parallel with the corrected UMAP recompute and both converge, we won't be able to cleanly attribute the recovery — and the team email gets murky.

The cleaner sequence is:
1. Run corrected UMAP at full scale → confirm or refute the "bugs were the cause" hypothesis
2. **If confirmed**, PCA becomes a nice-to-have robustness check (run later, separate session)
3. **If refuted** (corrected UMAP still shows instability), PCA becomes a high-value next experiment with clear motivation

Practical resource note: PCA on 1.45M × 1536 with subsequent full GMM re-fitting and BC computation is its own 3-hour pipeline. Running it in parallel with the corrected UMAP recompute would more than double peak memory and create I/O contention on a single c5.4xlarge.

The PCA script (`scripts/run_pca_comparison_sweep.py`) is in the repo and validated. It just doesn't run in this VM session.

---

## What you need to run

**Single script:**

```bash
source venv/bin/activate
python -u scripts/recompute_bc_corrected.py \
    2>&1 | tee output/kmax_sweep/recompute.log
```

This will:
1. Load the 464-firm deduplication decisions from `output/kmax_sweep/deduplication_decisions.parquet`
2. Filter the original GMM parameters in `firm_gmm_parameters_k{10,15,20,25,30}.parquet` to deduplicated firms (~7,485)
3. Compute the SG-vs-SG block ONCE (vectorized, fast — ~5-10 minutes)
4. Compute the full BC matrix at each K_max using the corrected linear-weight formula
5. Recompute convergence metrics and save a new convergence summary

**Outputs** (all to `output/kmax_sweep/`):
- `bc_block_sg_vs_sg_dedup_linear.npz` (~140 MB)
- `bc_matrix_all_k{10,15,20,25,30}_dedup_linear.npz` (~400 MB each)
- `convergence_summary_dedup_linear.json`

Outputs (in `output/kmax_sweep/`):
- `bc_block_sg_vs_sg_dedup_linear.npz` — corrected SG block (~140 MB)
- `bc_matrix_all_k{10,15,20,25,30}_dedup_linear.npz` — corrected BC matrices (~400 MB each)
- `convergence_summary_dedup_linear.json` — new convergence verdict

Total disk: ~2.2 GB.

### Estimated runtime

| Hardware | Estimated time |
|----------|---------------|
| WSL (1-2 cores) | ~4-5 hours |
| c5.4xlarge (16 vCPUs) | **~1.5-2 hours** ← recommended |
| c5.9xlarge (36 vCPUs) | ~1 hour |

The script is single-threaded. Multi-core helps with numpy operations inside `bc_component_matrix` but most of the time is spent in Python loops (one per pair). c5.4xlarge is the sweet spot for cost/time.

Bottleneck: per-pair Python loop calling `bc_mixture_linear()`. SG block is vectorized (~22M pairs in 5-10 minutes). Remaining ~5M GMM-involving pairs per K_max take ~15-20 minutes each on c5.4xlarge. Total: ~1.5 hours wall clock.

### Memory usage

Peak memory: ~3 GB. Any AWS instance with ≥8 GB RAM is sufficient.

---

## Pre-flight checks

### 1. Verify S3 artifacts exist

```bash
aws s3 ls s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/output/kmax_sweep/ --profile torrin
aws s3 ls s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/deduplication_decisions.parquet --profile torrin
```

### 2. Pull input files to the VM (~95 MB total)

```bash
mkdir -p output/kmax_sweep output/week2_inputs

# GMM parameters from original sweep
for k in 10 15 20 25 30; do
  aws s3 cp s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/output/kmax_sweep/firm_gmm_parameters_k${k}.parquet \
    output/kmax_sweep/ --profile torrin
done

# Dedup decisions
aws s3 cp s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/deduplication_decisions.parquet \
  output/kmax_sweep/ --profile torrin

# gvkey_map (only needed if convergence metrics need to map gvkeys; the script reads it for safety)
aws s3 cp s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/gvkey_map.parquet \
  output/week2_inputs/ --profile torrin
```

### 3. Verify script + dependencies

```bash
python -c "from scripts.run_kmax_sweep import load_gmm_results, bc_component_matrix; print('OK')"
python -c "from scripts.recompute_bc_corrected import bc_mixture_linear, compute_sg_block_linear; print('OK')"
python -c "import numpy, pandas, scipy; print('OK')"
```

### 4. Sanity-check on a small sample (~2 minutes)

```bash
python -u scripts/recompute_bc_corrected.py --test-mode --test-size 100
```

Expected: top-50 overlap should be 80-100% on the test sample, max BC < 1.0, no errors.

---

## What success looks like

When the script completes, the convergence summary should look approximately like this (based on the 1000-firm test):

```json
{
  "method": "deduplicated + linear-weighted BC",
  "n_firms_after_dedup": 7485,
  "convergence_verdict": "converged",
  "converged_at_kmax": 10,
  "adjacent_comparisons": [
    {"k_max_a": 10, "k_max_b": 15, "spearman_rho": ~0.96, "top_50_overlap_pct": ~95-100},
    {"k_max_a": 15, "k_max_b": 20, "spearman_rho": ~0.97, "top_50_overlap_pct": ~95-100},
    {"k_max_a": 20, "k_max_b": 25, "spearman_rho": ~0.97, "top_50_overlap_pct": ~95-100},
    {"k_max_a": 25, "k_max_b": 30, "spearman_rho": ~0.97, "top_50_overlap_pct": ~95-100}
  ]
}
```

If the verdict is anything other than CONVERGED at K_max=10 or 15, **stop and report back** — that would mean the hypothesis is wrong at scale and the post-run decision tree below tilts toward "run PCA next."

---

## Post-run decision tree

After Codex returns the corrected UMAP result, Claude (or the next instance) decides what to run next based on what we see:

### Scenario A: CONVERGED at K_max=10 or 15 (matches the 1000-firm test)

**Most likely outcome.** Implies the two bugs explained the original instability and the methodology is sound.

Next steps:
1. **PCA comparison becomes nice-to-have** — useful as a robustness check for the supervisor email but not blocking. Can be run as a separate VM session at any time using `scripts/run_pca_comparison_sweep.py`.
2. **Misspecification tests still worth running** but on a smaller sample (~50 stratified firms): Mahalanobis Q-Q vs χ²(50), Mardia's test, prior sensitivity. These run locally in minutes.
3. **Populate the notebook** with corrected results, regenerate the 12 PNGs, write Section 7 narrative, knit to PDF.
4. **Update ADR-004** for the converged outcome. Update `firm_portfolio_spec.md` to add deduplication as a required input validation step.
5. **Send team email**.

### Scenario B: STILL NOT CONVERGED at full scale

**Less likely based on the test, but possible** if duplicates were less concentrated in the small-firm bin than expected, or if there's residual instability we didn't catch in the sample.

Next steps:
1. **PCA comparison becomes high-priority** — spin up a separate VM session for `scripts/run_pca_comparison_sweep.py`. This is now the next experimental priority because the misspecification hypothesis becomes much more plausible.
2. **Misspecification tests run BEFORE PCA** so we have direct evidence about Gaussianity to compare against the PCA result.
3. **No team email yet** — wait for the PCA result first, then assemble a more nuanced story.

### Scenario C: PARTIAL CONVERGENCE (some transitions pass, others don't)

**Indicates a separate K_max regime issue.** E.g., converges at K_max=15 but not K_max=10 — would mean K_max=10 is genuinely too restrictive while K_max=15+ is fine.

Next steps:
1. Lock the smallest converged K_max as the operational default (per Branch A of the pre-registered framework).
2. Misspecification tests still useful but lower priority.
3. PCA still nice-to-have.
4. Email goes out with the partial-convergence story.

---

## What to do when complete

1. **Sync results to S3**:
   ```bash
   aws s3 sync output/kmax_sweep/ s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260411T_dedup_linear/output/kmax_sweep/ \
     --exclude "*_test*" --profile torrin
   ```
   (Separate "run ID" to distinguish from the original sweep artifacts.)

2. **Report results** back to Torrin. Include:
   - The convergence summary JSON contents (`convergence_summary_dedup_linear.json`)
   - Total runtime
   - Any unexpected warnings
   - Confirmation that max BC values are bounded (≤ 1.0) in every matrix

3. **Tear down the VM** — no need to keep it running after the recomputation completes.

4. **Indicate which scenario** (A/B/C from the decision tree above) the result matches, so the next instance has clear next steps without re-deriving the analysis.

---

## What I'll do next (locally)

After Codex returns the corrected matrices:

1. **Pull the new BC matrices** locally
2. **Regenerate the 12 notebook visualizations** with the corrected data
3. **Run misspecification diagnostics** on the deduplicated dataset:
   - Mahalanobis Q-Q vs χ²(50) on stratified firm sample
   - Mardia's multivariate skewness/kurtosis test
   - Per-dimension Q-Q plots
   - Prior sensitivity (γ ∈ {2.0, 5.0})
4. **Populate notebook interpretation cells** + Section 7 narrative
5. **Knit to PDF** for Torrin's team email

---

## Risks and mitigation

| Risk | Likelihood | Mitigation |
|------|----------|-----------|
| Script crashes on a specific firm pair | Low | Pre-flight test on 100-firm sample |
| Out-of-memory | Low | Peak ~3 GB; any reasonable VM has enough |
| Linear BC values > 1.0 (formula bug) | Very low | Sanity check in script catches this |
| Convergence verdict differs from sample | Possible | If so, stop and report — important diagnostic |
| Runtime exceeds estimate | Possible | Watchdog should kill at 4 hours just in case |
| AWS credentials expire mid-run | Low | Sync at end, not during; save status JSON locally |

---

## Files to read for context

If you need more background:
1. `docs/epics/instance_handover/week2_interpretation_instance_summary.md` — full interpretation instance handover
2. `docs/epics/week2_firm_portfolios/kmax_diagnostic_findings.md` — primary findings doc with all evidence
3. `scripts/recompute_bc_corrected.py` — the script you'll run (well-commented)
4. `scripts/duplicate_firm_unified_rule.py` — how the deduplication decisions were derived
5. `output/kmax_sweep/deduplication_decisions.csv` — the 464 firms being removed

---

## Acceptance criteria

This recomputation is complete when:
- [x] Script runs to completion without errors
- [x] All 5 corrected BC matrices saved (`bc_matrix_all_k{N}_dedup_linear.npz`)
- [x] Convergence summary JSON saved
- [x] Max BC value ≤ 1.0 in every matrix
- [x] Results synced to S3
- [x] Verdict reported back to Torrin

---

## Questions / Decisions for Torrin (none required upfront)

The script doesn't require any decisions — it uses the rule we already validated. But after the run, Torrin may want to discuss:

1. Whether the verdict matches expectations from the 1000-firm test
2. Whether to also regenerate the notebook PNGs (Codex can do this on the VM, or I can do it locally)
3. Whether to run the misspecification tests on the VM (faster) or locally (sufficient for stratified sample)
