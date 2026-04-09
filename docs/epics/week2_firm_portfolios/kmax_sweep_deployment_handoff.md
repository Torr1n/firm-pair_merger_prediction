# K_max Convergence Sweep: Deployment Handoff

**From**: Claude Code (development instance)
**To**: Codex (deployment lead)
**Date**: 2026-04-08
**Context**: The Week 2 design phase is complete (ADRs 004-007, spec, Codex review fixes applied). The K_max convergence sweep is the critical experiment blocking implementation. This document hands off the deployment-ready script for AWS execution.

---

## What This Does

Fits Bayesian GMMs for all ~7,949 non-excluded firms (single-Gaussian + GMM-tier) at K_max values {10, 15, 20, 25, 30}, computes pairwise Bhattacharyya Coefficients between all ~31.6M firm pairs at each K_max, and measures whether BC rankings converge as K_max increases.

**BC scope**: All non-excluded firms are included (not just GMM-tier). Single-Gaussian firms are K_max-invariant (K=1 always), but they participate in the ranking universe because the GMM-tier firm they're compared against changes with K_max. The SG-vs-SG block is computed once and reused across all K_max values for efficiency.

**Why it matters**: The design phase found BC rankings moderately stable (Spearman rho ~0.80) but the top tail -- where M&A candidate pairs live -- is materially unstable (top-50 overlap 22-48%). If rankings converge by K_max=25-30, we have our production K_max. If they don't, we escalate per the trigger framework in ADR-004.

**Convergence rule**: Persistent stability — K* is the smallest K_max such that ALL subsequent adjacent comparisons from K* onward pass both Spearman rho > 0.95 AND top-50 overlap > 80%. A single good transition is not enough if later transitions destabilize.

---

## Script Location

```
scripts/run_kmax_sweep.py
```

Committed at: `26b51ec` (Week 2 design commit) + subsequent sweep script commit.

---

## Instance Recommendation

| Spec | Value | Rationale |
|------|-------|-----------|
| **Instance type** | `c5.4xlarge` | 16 vCPUs, 32 GB RAM. CPU-optimized — no GPU needed for sklearn GMM fitting. |
| **Alternative** | `c6i.4xlarge` | Same specs, newer generation. Either works. |
| **Root volume** | 50 GB gp3 | Script + data + outputs ~3 GB; generous headroom for logs and checkpoints. |
| **Cost** | ~$0.68/hr (c5.4xlarge) | |
| **Estimated runtime** | 8-14 hours | 5 K_max values x ~7,949 GMM fits each + 5 BC matrices (~31.6M pairs each, with SG block caching) |
| **Estimated cost** | ~$5-10 | |

The bottleneck is now BC computation on ~7,949 firms (~31.6M pairs per K_max). The SG-vs-SG block (~19.9M pairs) is computed once and reused, reducing per-K_max work to ~11.7M new pairs. GMM fitting is secondary. BC computation is vectorized with numpy broadcasting.

---

## Data Requirements

The script needs two input files. Two options:

### Option A: Copy from Week 1 S3 outputs (recommended)
```bash
mkdir -p output/week2_inputs
aws s3 cp s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/patent_vectors_50d.parquet \
    output/week2_inputs/patent_vectors_50d.parquet --profile torrin
aws s3 cp s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/gvkey_map.parquet \
    output/week2_inputs/gvkey_map.parquet --profile torrin
```

### Option B: Use config paths (if data is at output/embeddings/)
The script also supports reading from the paths in `config.yaml` (default). Use `--local` flag to read from `output/week2_inputs/` instead.

---

## Execution

```bash
cd /home/ubuntu/firm-pair_merger_prediction
source venv/bin/activate

# Install dependencies (should already be present from bootstrap)
pip install -r requirements.txt

# Run with local data paths
python scripts/run_kmax_sweep.py --local 2>&1 | tee output/kmax_sweep/sweep.log
```

### Checkpoint/Resume

If interrupted, re-running the same command will:
1. Detect existing `firm_gmm_parameters_k{N}.parquet` files and skip completed K_max fitting stages
2. Detect existing `bc_matrix_k{N}.npz` files and skip completed BC computation stages
3. Always recompute convergence metrics (cheap, <1 second)

---

## Output Structure

```
output/kmax_sweep/
    firm_gmm_parameters_k10.parquet    # GMM results for all non-excluded firms at K_max=10
    firm_gmm_parameters_k15.parquet    # ... K_max=15
    firm_gmm_parameters_k20.parquet    # ... K_max=20
    firm_gmm_parameters_k25.parquet    # ... K_max=25
    firm_gmm_parameters_k30.parquet    # ... K_max=30
    bc_block_sg_vs_sg.npz              # SG-vs-SG BC block (K_max-invariant, computed once)
    bc_matrix_all_k10.npz              # Pairwise BC matrix (ALL non-excluded firms)
    bc_matrix_all_k15.npz
    bc_matrix_all_k20.npz
    bc_matrix_all_k25.npz
    bc_matrix_all_k30.npz
    convergence_summary.json           # Full convergence analysis
    excluded_firms.csv                 # Firms below threshold (gvkey, n_patents, reason)
    status/
        sweep_status.json              # Machine-readable status for monitoring
    sweep.log                          # Full stdout/stderr log
```

### S3 Sync (after completion)
```bash
aws s3 sync output/kmax_sweep/ \
    s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/ \
    --profile torrin
```

---

## Monitoring

### Status file
```bash
cat output/kmax_sweep/status/sweep_status.json
```

Status values:
- `running` — sweep in progress (includes `completed_k_max` list showing progress)
- `success` — converged (includes `converged_at` K_max value)
- `completed_no_convergence` — all K_max values fitted but convergence criteria not met

### Log monitoring
```bash
tail -f output/kmax_sweep/sweep.log
```

The script reports progress every 500 firms (during fitting) and every 100 rows (during BC computation). Expected log pattern:

```
[Stage 4.1] Fitting GMMs at K_max=10...
    [500/7949] 120s elapsed, 4.2 firms/s, ETA 1786s
    ...
[Stage 5] Computing pairwise Bhattacharyya Coefficients...
  [K_max=10] Computing BC matrix (GMM-tier firms only)...
    BC [100/1645 rows] 134,450/1,352,490 pairs (9.9%) 45s elapsed, ETA 410s
    ...
[Stage 6] Computing convergence metrics...
  Comparing K_max=10 vs K_max=15...
    Spearman rho = 0.XXXX
    Top-50 overlap = XX.X%
```

---

## Success Criteria

The script prints a clear verdict at the end:

```
VERDICT: CONVERGED at K_max=N
```
or
```
VERDICT: NOT CONVERGED by K_max=30
```

**Convergence decision rule** (persistent stability, from Codex review):
- Spearman rho > 0.95 AND top-50 overlap > 80% between adjacent K_max values
- Must hold for ALL subsequent adjacent pairs from the convergence point onward
- Example: if 15->20 passes but 20->25 fails, NOT converged at 20

### What to send back

1. `convergence_summary.json` — the full analysis (machine-readable)
2. The verdict line from the log
3. The effective K progression table (printed at end of log)
4. If NOT converged: the full adjacent-comparison table for the team to review

---

## Watcher Script

The existing `scripts/watch_pipeline_and_shutdown.sh` can be adapted for this sweep. Key changes:
- `LOG_PATH` → `output/kmax_sweep/sweep.log`
- `STATUS_DIR` → `output/kmax_sweep/status`
- Success detection: `grep -q "Sweep complete"` in the log
- S3 sync target: `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/`

Alternatively, since estimated runtime is 3-6 hours (well within the 8-hour default timeout), the watcher can be used as-is with adjusted paths.

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| OOM on BC matrix | Low | BC matrices are ~505 MB each (7949^2 * 8 bytes). Peak memory ~4 GB with SG block + 1 active matrix. 32 GB available. |
| sklearn convergence warnings | Expected | Suppressed with `warnings.catch_warnings()`. Non-convergence is tracked in `converged` field. |
| K_max > n_patents crash | Guarded | `actual_kmax = min(k_max, len(X) - 1)` prevents sklearn ValueError. |
| Interrupted mid-run | Guarded | Per-K_max checkpointing. Re-run resumes from last completed stage. |
| Very slow BC computation | Mitigated | Vectorized with numpy broadcasting (~5 min per K_max, not ~90 min). |

---

## Dependencies

Standard from `requirements.txt`:
- numpy, pandas, pyarrow (data handling)
- scikit-learn (GMM fitting)
- scipy (Spearman, Kendall, distance metrics)

No torch, no GPU libraries needed. This is a pure CPU workload.

---

## Relation to Design Review

This sweep addresses the critical open question from the Week 2 design phase. The three Codex Major findings from the design review have been fixed:

1. **Multi-K_max output contract** — The sweep produces `firm_gmm_parameters_k{N}.parquet` for each K_max value, exactly as specified in the updated config and spec.
2. **ELBO not BIC** — The GMM results use `lower_bound` (ELBO) throughout, not BIC.
3. **Global priors from unique patent matrix** — The script computes `global_mean` and `global_var` from the raw patent vectors array (line 675), not from grouped firm vectors.

After sweep results are in:
1. Development instance analyzes convergence
2. ADR-004 and config are updated with the final K_max
3. Codex re-reviews the design (incorporating both the original Major fixes and the sweep results)
4. Implementation proceeds (TDD)
