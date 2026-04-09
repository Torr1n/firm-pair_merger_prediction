# Week 2 Design Review Handoff

**From**: Claude Code (Week 2 development instance)
**To**: Codex (impartial reviewer)
**Date**: 2026-04-08
**Gate**: Phase 2 design review — implementation HALTS until approved
**Repository**: `git@github.com:Torr1n/firm-pair_merger_prediction.git` (branch: `main`)

---

## What You're Reviewing

Four ADRs, one interface spec, and supporting EDA evidence for the Week 2 firm portfolio construction pipeline. This is the design gate before implementation begins.

**Files to review:**

| File | Status | What It Decides |
|------|--------|-----------------|
| `docs/adr/adr_004_k_selection_method.md` | Accepted with sensitivity requirement | Bayesian GMM, γ=1.0, K_max=15, global EB priors |
| `docs/adr/adr_005_minimum_patent_threshold.md` | Accepted | Three tiers: <5 exclude, 5-49 single Gaussian, 50+ GMM |
| `docs/adr/adr_006_covariance_type.md` | Accepted | Diagonal covariance |
| `docs/adr/adr_007_prenormalization_strategy.md` | Accepted (raw) | No pre-normalization |
| `docs/specs/firm_portfolio_spec.md` | Proposed | PortfolioBuilder + GMMFitter interfaces |
| `src/config/config.yaml` | Updated | `portfolio` section with all hyperparameters |

**Supporting evidence:**

| File | Contents |
|------|----------|
| `docs/epics/week2_firm_portfolios/bayesian_gmm_rfi.md` | RFI sent to STAT 405 instance |
| `docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md` | Full Bayesian methodology audit response |

---

## Executive Summary

Week 2 fits a Gaussian Mixture Model per firm on 50D patent vectors (Week 1 output). The design was developed through three phases of increasing rigor:

1. **Initial ADR drafting** — standard literature-informed decisions
2. **Bayesian methodology audit** (STAT 405) — identified that our initial γ=0.1 was too sparse, sklearn defaults double-count data, and we needed a proper Bayesian workflow
3. **EDA diagnostics on production data** (1,447,673 patents, 15,696 firms) — validated some assumptions, falsified others, and revealed a K_max sensitivity issue

The result is a design that is **empirically grounded but methodologically honest about its limitations**. The most important finding: BC rankings are moderately stable across K_max settings (Spearman ρ ≈ 0.80), but the top tail — where M&A candidate pairs live — is materially sensitive to K_max. This is documented as a methodological condition, not swept under the rug.

---

## ADR-004: K Selection Method

**Decision**: Bayesian GMM with Dirichlet Process prior (sklearn `BayesianGaussianMixture`). Variational inference. K_max=15 operational default.

**Key design choices and their justification:**

| Choice | Value | Why |
|--------|-------|-----|
| Inference method | Variational Bayes (sklearn) | Point estimates sufficient for BC computation; VI sidesteps label switching; 10-30x faster than MCMC. Stan/NUTS explicitly unsuitable for mixtures (can't sample discrete assignments). |
| γ (concentration) | 1.0 | E[K] ≈ γ·log(n) gives natural logarithmic scaling with firm size. γ=0.1 (initial choice) gives E[K]≈0.6 at n=50 — actively suppresses multi-component solutions. EDA confirmed γ has negligible effect on mid/large firms; K is data-driven. |
| Priors | Global empirical Bayes | `mean_prior` and `covariance_prior` computed from pooled dataset (all 1.45M patents), not per-firm. Avoids double-counting. Approximates hierarchical model. |
| ν₀ | 52 (= d+2) | Minimum for finite posterior mean. Ensures regularization anchor for singleton clusters. |
| K_max | 15 (with sensitivity requirement) | See below. |

**The K_max sensitivity issue:**

EDA diagnostics revealed that K does not plateau — it climbs from 9.1 → 12.8 → 16.0 for large firms as K_max increases from 10 → 15 → 20, with substantive (non-fragmenting) weights at each level. BC ranking stability across K_max settings:

| Metric | K=10 vs K=15 | K=10 vs K=20 | K=15 vs K=20 |
|--------|:------------:|:------------:|:------------:|
| Spearman ρ | 0.807 | 0.774 | 0.792 |
| Top-50 overlap | 48% | 22% | 36% |
| Top-5 NN overlap | 62% | 53% | 64% |

**K_max=15 is the operational default because:**
- K=10 is clearly binding (35% ceiling rate)
- K=20 is not obviously converged (9/50 large firms still hit ceiling)
- K=15 is the least arbitrary midpoint

**Sensitivity requirement baked into Week 3 contract:**
- Production must fit at K_max ∈ {10, 15, 20}
- BC-based findings must classify firm pairs as **robust** (stable across K_max) or **model-sensitive**

**What I want Codex to scrutinize:**
- Is the "accepted with sensitivity requirement" framing appropriate, or should K_max selection be deferred entirely?
- Is fitting at three K_max values computationally tractable for 15K firms?
- Does the Week 3 sensitivity contract adequately address the ranking instability?

---

## ADR-005: Minimum Patent Threshold

**Decision**: Three tiers — <5 exclude, 5-49 single Gaussian (K=1), 50+ full GMM.

**EDA validation on production data:**

| Tier | Firms | % of Firms | Patents | % of Patents |
|------|:-----:|:----------:|:-------:|:------------:|
| Exclude (<5) | 7,747 | 49.4% | 14,938 | ~1.0% |
| Single Gaussian (5-49) | 6,304 | 40.2% | 91,498 | ~6.3% |
| Full GMM (50+) | 1,645 | 10.5% | 1,343,476 | 92.8% |

Excluding half the firms loses only ~1% of patents. The 1,645 GMM-tier firms hold 93% of all patents. Synthetic experiments confirmed that Bayesian GMM cannot reliably prune in 50D until n ≈ 50.

**What I want Codex to scrutinize:**
- Is n=50 as the GMM boundary justified, or should the Bayesian audit's insight (diagonal = 50 independent 1D Normal-Gamma updates, data-informed even at n=5) lower it?
- Should excluded firms get a flag in the output rather than being silently dropped?

---

## ADR-006: Covariance Type

**Decision**: Diagonal covariance for all components.

**Important context**: The original rationale ("UMAP dimensions are weakly correlated") was **empirically falsified** — EDA found mean |r|=0.33, top pairs |r|=0.95. The ADR was revised with a different justification:

**Diagonal wins because of parameter budget, not independence.** A direct diagnostic comparing diagonal, full, and tied covariance on 20 firms (n=200-1000) showed:

| Covariance | Mean KS (PPC) | vs Diagonal |
|:----------:|:-------------:|:-----------:|
| Diagonal | 0.118 | — |
| Tied | 0.142 | 20% worse |
| Full | 0.191 | 62% worse |

Full covariance has 1,275 params/component in 50D, forcing K=2 for most firms. Diagonal allows K=8-10 with far better density coverage. The trade-off decisively favors more diagonal components over fewer full-covariance components.

Additional theoretical justification: diagonal avoids the Inverse-Wishart prior's three pathologies in high dimensions (variance-correlation coupling, uniform shrinkage, concentration effect).

**What I want Codex to scrutinize:**
- The covariance diagnostic capped full covariance at K=2-5 via `safe_k = min(5, n//300)`. Is this too conservative? Would a less restricted comparison change the conclusion?
- Is the "best practical choice under constraints" framing defensible for a research paper, or does it need a stronger theoretical grounding?

---

## ADR-007: Pre-Normalization Strategy

**Decision**: Raw (no normalization).

**EDA evidence**: All three normalizations (raw, L2, z-score) produce nearly identical results on 50 mid-sized firms:

| Normalization | Silhouette | Effective K | Cross-norm K agreement |
|:-------------:|:----------:|:-----------:|:---------------------:|
| Raw | 0.586 | 8.4 | — |
| L2 | 0.585 | 8.2 | 90% within ±1 |
| Z-score | 0.587 | 8.4 | 96% within ±1 |

Despite 16x scale variation across dimensions (std range [0.14, 2.32]), diagonal covariance handles per-dimension scale by construction. Raw is the simplest option with no measurable downside. This satisfies the Codex-committed normalization sensitivity check from Week 1.

**What I want Codex to scrutinize:**
- Is the evidence sufficient to close this ADR, or should we re-check in the PCA-reduced space if PCA is ever adopted?

---

## Firm Portfolio Spec

**Modules**: `PortfolioBuilder` (load, group, classify, normalize) and `GMMFitter` (set_global_priors, fit_firm, fit_all, serialize, load).

**Key design elements:**
- `GMMFitter.set_global_priors()` computes global mean/var from pooled data before any fitting (global empirical Bayes)
- `GMMResult` dataclass with means, covariances, weights, convergence, BIC, tier
- Checkpoint-resume for batch fitting (save every N firms)
- Pruned components (weight < 0.01) removed from serialized output, weights renormalized
- Bayesian workflow steps (prior predictive simulation, PPC, sensitivity audit) are documented as required, not optional

**Week 3 contract**: BC computation consumes `firm_gmm_parameters.parquet`. K_max sensitivity reporting is contractually required.

**What I want Codex to scrutinize:**
- Interface completeness — are there missing methods or edge cases?
- Checkpoint format — is the parquet schema sufficient for Week 3?
- Is the `GMMResult` dataclass the right abstraction, or should fitted sklearn models be preserved?

---

## EDA Findings That Inform the Design

These are not separate deliverables — they are the empirical foundation for the ADR decisions above. Summary of findings that surprised us or changed decisions:

| Finding | Impact |
|---------|--------|
| Inter-dimension correlations strong (mean \|r\|=0.33, max 0.95) | Falsified "weakly correlated" assumption in ADR-006; revised justification to "best practical choice" |
| Intrinsic dimensionality ~12 (PCA: 99% variance in 12 dims) | Investigated PCA post-processing; found it does NOT improve GMM fits; kept raw 50D |
| K does not plateau (climbs to 16 at K_max=20) | K_max=10 is binding; raised to 15 with sensitivity requirement |
| PPC failures for mid-sized firms persist across all configurations | Model misspecification, not a tuning problem; documented as limitation |
| Full covariance worse than diagonal (parameter budget) | Resolved ADR-006 in favor of diagonal on empirical, not theoretical, grounds |
| Normalization doesn't matter | Resolved ADR-007 as raw |
| Prior predictive simulation passed | Calibrated priors (global EB) produce plausible synthetic portfolios |
| γ has negligible effect on mid/large firms | K selection is data-driven for the firms that matter (93% of patents) |

---

## Verification Steps for Codex

1. Read all four ADRs and verify internal consistency
2. Read the spec and verify it implements the ADR decisions
3. Verify `config.yaml` matches ADR parameters (K_max=15, γ=1.0, ν₀=52, n_init=5, norm=raw)
4. Assess whether the K_max sensitivity condition is adequately specified for Week 3
5. Assess whether the Bayesian workflow requirements (prior predictive sim, PPC, sensitivity audit) are feasible within the sprint
6. Flag any spec gaps that would block implementation

---

## Constraints

- **Do not proceed to implementation until this review is complete.**
- The production 50D vectors and gvkey_map are available locally at `output/week2_inputs/`.
- All EDA was conducted on real production data (1,447,673 patents, 15,696 firms).
- The Bayesian methodology audit (STAT 405) is incorporated into all ADRs.
- The team has been briefed on the K_max sensitivity finding.
