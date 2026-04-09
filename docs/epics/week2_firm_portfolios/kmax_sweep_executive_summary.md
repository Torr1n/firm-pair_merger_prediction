# K_max Convergence Sweep — Executive Summary

**Status**: `[SKELETON — populate after sweep results return]`
**Run ID**: `[populated from convergence_summary.json]`
**Date**: `[YYYY-MM-DD]`
**Author**: Claude Code (Week 2 implementation instance) for Torrin Pataki

---

## TL;DR

`[ONE-SENTENCE VERDICT, e.g.: "BC rankings converge at K_max=20 with persistent stability across all higher settings tested." OR: "BC rankings do not converge within K_max ≤ 30; we adopt operational K_max=15 with mandatory robustness reporting in Week 3."]`

---

## What we ran and why

We fitted Bayesian Gaussian Mixture Models for ~7,949 non-excluded firms at five K_max values: {10, 15, 20, 25, 30}. For each K_max we computed pairwise Bhattacharyya Coefficients (BC) — a measure of distributional overlap — between all ~31.6 million firm pairs. We then measured how BC rankings change between adjacent K_max settings.

**The framing**: This is not a hyperparameter tuning exercise. It is a methodology resolution run that determines whether the Step 2 firm-portfolio representation is a stable scientific object or still a moving target. The Week 2 design phase had already shown that BC rankings depend meaningfully on the allowed number of mixture components (top-50 pair overlap as low as 22% in pilot results). This sweep was designed to either replace an arbitrary default with an empirically justified one, or — if convergence does not emerge — to formally classify the firm-similarity signal as sensitivity-conditioned.

**Pre-registered convergence rule** (locked before sweep launch):
> K\* = the smallest K_max such that ALL subsequent adjacent comparisons from K\* onward pass both:
> - Spearman ρ > 0.95
> - Top-50 pair overlap > 80%

---

## The four questions

### 1. Did rankings converge?

**Answer**: `[YES at K_max=N | NO]`

`[If yes]`: From K_max=`[N]` onward, every adjacent transition passes both convergence thresholds. Raising K_max further produces no material change in which firm pairs are flagged as most technologically similar.

`[If no]`: Within the range tested, BC rankings remain materially sensitive to K_max. The persistent-stability rule was not satisfied at any K_max value. The strongest adjacent transition was K_max=`[X]`→K_max=`[Y]` with Spearman ρ=`[ρ]` and top-50 overlap=`[N]%`. Both fall short of the pre-registered thresholds.

### 2. If yes, at what K_max?

**Answer**: `[K_max=N | N/A — not converged]`

`[If converged]`: K\*=`[N]`. Below this, at least one downstream adjacent comparison fails the threshold. From this point onward, all comparisons pass.

### 3. If not, how unstable is the tail?

**Answer**: `[summary statistics of tail instability]`

Of the firm pairs that ever appear in the top-200 across the five K_max settings:
- **Robust pairs** (in top-200 at all K_max): `[N]` (`[%]` of total)
- **Model-sensitive pairs** (in top-200 at some but not all): `[N]` (`[%]`)

The most stable region of the candidate set is `[describe — e.g., "top-200 from K_max=20 onward shows >80% pairwise overlap, despite the strict persistent-stability rule not being met"]`.

Per-firm nearest-neighbor stability:
- Median firm has `[N]/5` of its top-5 nearest neighbors stable across adjacent K_max settings
- The worst 10% of firms (P10) have only `[N]/5` stable

### 4. What does that imply for Week 2 and Week 3?

**Answer**: `[ONE OF THE TWO PRE-REGISTERED BRANCHES]`

---

## Decision: Pre-Registered Branches

### Branch A — Convergence emerges

**Production K_max**: K\*=`[N]` (smallest converged value)

**Week 2 implementation**:
- Lock K_max=`[N]` as the production default in `config.yaml`
- Implement PortfolioBuilder and GMMFitter with this default
- Generate one primary `firm_gmm_parameters_k[N].parquet` artifact
- Keep one neighbor (K_max=`[N+5]`) as a robustness check artifact

**Week 3 reporting**:
- Single primary BC specification
- Robustness check using the neighbor artifact (report top-pair overlap, expect >80%)
- Methodology section: "We adopted K_max=`[N]` as the production specification based on a pre-registered persistent-stability convergence study (Spearman ρ > 0.95 and top-50 pair overlap > 80% from K_max=`[N]` onward across all higher values tested)."

### Branch B — Convergence does not emerge

**Operational K_max**: `[default — likely 15 from current config, or the largest converged value if any]`

**Week 2 implementation**:
- Generate `firm_gmm_parameters_k{N}.parquet` for all five K_max values as first-class outputs
- The PortfolioBuilder/GMMFitter modules treat K_max as a configurable parameter, not a final choice
- Document explicitly that the default is operational, not final

**Week 3 reporting** (mandatory):
- All top-pair conclusions must include K_max robustness classification
- "Robust" pairs: in top-200 at all five K_max settings → strongest M&A candidates
- "Model-sensitive" pairs: in top-200 at some K_max settings → reported with K_max profile
- Methodology section explicitly acknowledges sensitivity as a condition: "The choice of maximum mixture components K_max is an influential specification parameter for top-pair identification. We report all candidate pairs along with their K_max robustness classification..."

**This is still publishable and intellectually valuable.** Sensitivity to K_max is itself a substantive finding about the structure of patent portfolios in the technology sector. It tells us that "how granularly we define a technology area" is not a free parameter — it affects which firms appear similar at the boundary of the candidate set.

---

## Key numbers

| Metric | Value |
|---|---|
| Sweep run ID | `[ID]` |
| Total firms in BC analysis | `[N]` |
| GMM-tier firms (50+ patents) | `[N]` |
| Single-Gaussian firms (5-49 patents) | `[N]` |
| Excluded firms (<5 patents) | `[N]` |
| Pairs evaluated per K_max | `[N]` |
| K_max values tested | 10, 15, 20, 25, 30 |
| Total runtime | `[N]` hours |
| AWS instance | `[type, e.g., c5.4xlarge]` |
| Total cost | `~$[N]` |

### Adjacent comparison metrics

| K_max transition | Spearman ρ | Top-50 overlap | Top-100 overlap | Mean NN-5 overlap | Passes threshold? |
|---|---|---|---|---|---|
| 10 → 15 | `[ρ]` | `[%]` | `[%]` | `[%]` | `[Y/N]` |
| 15 → 20 | `[ρ]` | `[%]` | `[%]` | `[%]` | `[Y/N]` |
| 20 → 25 | `[ρ]` | `[%]` | `[%]` | `[%]` | `[Y/N]` |
| 25 → 30 | `[ρ]` | `[%]` | `[%]` | `[%]` | `[Y/N]` |

### Effective K progression (GMM-tier firms)

| K_max | Mean K | Median K | P90 K | Ceiling rate |
|---|---|---|---|---|
| 10 | `[N]` | `[N]` | `[N]` | `[%]` |
| 15 | `[N]` | `[N]` | `[N]` | `[%]` |
| 20 | `[N]` | `[N]` | `[N]` | `[%]` |
| 25 | `[N]` | `[N]` | `[N]` | `[%]` |
| 30 | `[N]` | `[N]` | `[N]` | `[%]` |

---

## Supporting analysis

The full analysis with visualizations is in:
- `notebooks/03_kmax_convergence_analysis.ipynb`

Raw artifacts on S3:
- `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/`

This includes:
- Five `firm_gmm_parameters_k{N}.parquet` files (GMM results per K_max)
- Five `bc_matrix_all_k{N}.npz` files (pairwise BC matrices)
- One `bc_block_sg_vs_sg.npz` (K_max-invariant single-Gaussian block, computed once)
- `convergence_summary.json` (machine-readable metrics)
- `excluded_firms.csv` (with gvkey, n_patents, reason)

---

## What this enables

`[POPULATE: one paragraph explaining how the verdict unlocks the next phase. E.g.,]`

`[If converged]`: With a defensible K_max default, Week 2 implementation can proceed to TDD without further methodology debate. The PortfolioBuilder and GMMFitter modules implement the spec with K_max=`[N]`. Week 3 BC computation uses a single primary specification with the neighbor as a sanity check. Codex re-review can focus on implementation quality, not design.

`[If not converged]`: The Week 2 implementation proceeds with multi-K_max output as a first-class deliverable. Week 3 BC analysis is paired with mandatory robustness classification. The "robust pair" set becomes a high-confidence input to the M&A prediction model in Week 4. Sensitivity analysis becomes part of the published methodology, not an appendix.

In either case, **the next gate is implementation, not more design work.** This sweep was designed to settle the methodology question definitively.

---

## Provenance

- **Pre-registration**: This document and `notebooks/03_kmax_convergence_analysis.ipynb` were committed before the sweep launched, locking the decision rule and analysis structure
- **Sweep script**: `scripts/run_kmax_sweep.py` (committed before launch)
- **Decision rule**: `ADR-004` (K selection method) — persistent stability rule
- **Data**: `output/week2_inputs/patent_vectors_50d.parquet` from Week 1 production run `20260408T005013Z`
