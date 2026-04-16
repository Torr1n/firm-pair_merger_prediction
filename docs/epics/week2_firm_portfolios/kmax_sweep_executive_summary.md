# K_max Convergence Sweep — Executive Summary

**Status**: CONVERGED — Branch A applies  
**Run ID**: `20260412T043407Z-dedup-linear` (corrected recomputation)  
**Date**: 2026-04-12  
**Author**: Claude Code (Week 2 implementation instance) for Torrin Pataki

---

## TL;DR

BC rankings converge at K_max=10 with persistent stability across all higher settings tested. After correcting two data/metric issues (464 duplicate firms removed, BC formula bounded), the firm-similarity measure is a stable scientific object — the top-50 candidate pairs are 96-100% consistent across K_max=10 through K_max=30.

---

## What we ran and why

We fitted Bayesian Gaussian Mixture Models for 7,485 deduplicated firms at five K_max values: {10, 15, 20, 25, 30}. For each K_max we computed pairwise Bhattacharyya Coefficients (BC) — a measure of distributional overlap — between all ~28.0 million firm pairs. We then measured how BC rankings change between adjacent K_max settings.

**The framing**: This is not a hyperparameter tuning exercise. It is a methodology resolution run that determines whether the Step 2 firm-portfolio representation is a stable scientific object or still a moving target. The Week 2 design phase had already shown that BC rankings depend meaningfully on the allowed number of mixture components (top-50 pair overlap as low as 22% in pilot results). This sweep was designed to either replace an arbitrary default with an empirically justified one, or — if convergence does not emerge — to formally classify the firm-similarity signal as sensitivity-conditioned.

**Data corrections applied before this run:**

1. **Deduplication**: 464 firms removed (264 aliases, 141 subsidiaries, 59 predecessor records) using a containment ≥ 0.95 rule. These firms would have generated false-positive M&A predictions for already-completed acquisitions.
2. **BC formula fix**: Changed from √(πᵢπⱼ) weighting (unbounded, up to 5.39) to linear πᵢπⱼ weighting (properly bounded in [0, 1]).

**Pre-registered convergence rule** (locked before sweep launch):
> K\* = the smallest K_max such that ALL subsequent adjacent comparisons from K\* onward pass both:
> - Spearman ρ > 0.95
> - Top-50 pair overlap > 80%

---

## The four questions

### 1. Did rankings converge?

**Answer**: YES at K_max=10

From K_max=10 onward, every adjacent transition passes both convergence thresholds. Raising K_max further produces no material change in which firm pairs are flagged as most technologically similar.

### 2. If yes, at what K_max?

**Answer**: K_max=10

K\*=10. This is the smallest K_max tested, and all subsequent adjacent comparisons pass both thresholds. The persistent stability rule is satisfied at the smallest value in our sweep.

### 3. If not, how unstable is the tail?

**Answer**: N/A — convergence confirmed. For the record:

Of the firm pairs that appear in the top-200 across the five K_max settings:
- Top-200 overlap between any adjacent K_max pair: 98.5-99.5%
- Top-50 overlap: 96-100%
- Top-100 overlap: 98-99%

Per-firm nearest-neighbor stability:
- Mean NN-5 overlap: 88-91% (across adjacent transitions)
- Median: 100% (the typical firm's top-5 peers do not change)
- P10: 60-80% (even the worst 10% retain most neighbors)

### 4. What does that imply for Week 2 and Week 3?

**Answer**: Branch A — Convergence emerges.

---

## Decision: Branch A — Convergence Confirmed

**Production K_max**: K\*=10 (smallest converged value)

**Note on K_max=10 vs K_max=15**: All five deep-dive mega-firms (IBM, Intel, Qualcomm, Google/Alphabet, Cisco) hit the effective K ceiling at K_max=10. This means K_max=10 is a binding constraint for the largest, most diversified firms. K_max=15 gives these firms room to express finer-grained technological structure without affecting convergence (K15→K20 also passes all thresholds). **This is a judgment call for the team — K_max=10 is sufficient for stable rankings; K_max=15 may be preferable for richer firm characterization.**

**Week 2 implementation**:
- Lock K_max=10 (or 15, pending team decision) as the production default in `config.yaml`
- Implement PortfolioBuilder and GMMFitter with this default
- Generate one primary `firm_gmm_parameters_k{N}.parquet` artifact
- Keep the neighbor K_max as a robustness check artifact

**Week 3 reporting**:
- Single primary BC specification
- Robustness check using the neighbor artifact (report top-pair overlap, expect >96%)
- Methodology section: "We adopted K_max=10 as the production specification based on a pre-registered persistent-stability convergence study (Spearman ρ > 0.95 and top-50 pair overlap > 80% from K_max=10 onward across all higher values tested)."

---

## Key numbers

| Metric | Value |
|---|---|
| Sweep run ID | 20260412T043407Z-dedup-linear |
| Total firms in BC analysis | 7,485 (after deduplication) |
| Firms removed by deduplication | 464 (5.8%) |
| GMM-tier firms (50+ patents) | ~1,506 |
| Single-Gaussian firms (5-49 patents) | ~5,979 |
| Excluded firms (<5 patents) | 7,865 |
| Pairs evaluated per K_max | 28,008,870 |
| K_max values tested | 10, 15, 20, 25, 30 |
| Max BC value (corrected) | 0.997 (non-self) |
| BC values above 1.0 | 0 |
| Total runtime (corrected recomputation) | ~2 hours on c5.4xlarge |

### Adjacent comparison metrics

| K_max transition | Spearman ρ | Top-50 overlap | Top-100 overlap | Top-200 overlap | Mean NN-5 overlap | Passes? |
|---|---|---|---|---|---|---|
| 10 → 15 | 0.9912 | 98% | 99% | 99.5% | 88.2% | YES |
| 15 → 20 | 0.9925 | 100% | 99% | 99.5% | 90.7% | YES |
| 20 → 25 | 0.9917 | 98% | 98% | 99.0% | 91.2% | YES |
| 25 → 30 | 0.9930 | 96% | 98% | 98.5% | 91.2% | YES |

### Non-adjacent comparison

| K_max transition | Spearman ρ | Top-50 overlap | Top-100 overlap | Top-200 overlap | Mean NN-5 overlap |
|---|---|---|---|---|---|
| 10 → 30 | 0.9833 | 96% | 98% | 99.5% | 84.2% |

### Effective K progression (GMM-tier firms)

| K_max | Mean K | Median K | P90 K | Ceiling rate |
|---|---|---|---|---|
| 10 | ~8.0 | ~9 | ~10 | ~35% |
| 15 | ~10.2 | ~11 | ~15 | ~12% |
| 20 | ~11.7 | ~12 | ~18 | ~3% |
| 25 | ~12.8 | ~13 | ~20 | ~1% |
| 30 | ~13.7 | ~14 | ~22 | ~0.2% |

*Note: Effective K statistics are from the original (pre-dedup) GMM parameters. Deduplication removed 139 GMM-tier firms but does not change the fitted parameters for remaining firms.*

---

## Supporting analysis

The full analysis with visualizations is in:
- `notebooks/03_kmax_convergence_analysis.ipynb`

Raw artifacts on S3:
- Corrected: `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/`
- Original (for audit): `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/`

This includes:
- Five `bc_matrix_all_k{N}_dedup_linear.npz` files (corrected pairwise BC matrices)
- One `bc_block_sg_vs_sg_dedup_linear.npz` (K_max-invariant single-Gaussian block)
- `convergence_summary_dedup_linear.json` (machine-readable metrics)
- Five `firm_gmm_parameters_k{N}.parquet` files (GMM results per K_max — shared with original run)
- `deduplication_decisions.csv` (464 firms removed with reasons)

---

## What this enables

With a defensible K_max default and a corrected, properly-bounded similarity metric, Week 2 implementation can proceed to TDD without further methodology debate. The PortfolioBuilder and GMMFitter modules implement the spec with K_max=10 (or 15). Week 3 BC computation uses a single primary specification with the neighbor as a sanity check. Codex re-review can focus on implementation quality, not design.

The deduplication step uncovered by this sweep becomes a required preprocessing step in the PortfolioBuilder — ensuring that the M&A prediction task operates on distinct, currently-independent legal entities rather than duplicate or subsidiary records.

**The next gate is implementation, not more design work.** This sweep was designed to settle the methodology question definitively — and it has.

---

## Provenance

- **Pre-registration**: This document and `notebooks/03_kmax_convergence_analysis.ipynb` were committed before the sweep launched, locking the decision rule and analysis structure
- **Original sweep**: `scripts/run_kmax_sweep.py` (committed before launch, contains the √-weight bug at line 473)
- **Corrected recomputation**: `scripts/recompute_bc_corrected.py` (linear weights, deduplication applied)
- **Decision rule**: `ADR-004` (K selection method) — persistent stability rule
- **Data**: `output/week2_inputs/patent_vectors_50d.parquet` from Week 1 production run `20260408T005013Z`
- **Deduplication**: `scripts/duplicate_firm_unified_rule.py` — containment ≥ 0.95 rule producing `deduplication_decisions.csv`
- **Diagnostic findings**: `docs/epics/week2_firm_portfolios/kmax_diagnostic_findings.md` — full evidence trail
