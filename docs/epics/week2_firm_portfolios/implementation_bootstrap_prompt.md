# Week 2 Implementation Bootstrap: K_max Convergence Sweep + GMM Fitting

**Purpose**: Initialize a fresh Claude Code instance to execute the implementation phase of Week 2 — deploying the K_max convergence sweep to AWS, analyzing results, and building the firm portfolio construction pipeline.

---

## Mission Briefing

You are the **Week 2 implementation instance** for the Firm-Pair Merger Prediction project. A prior design instance completed Phase 0 (data validation), Phase 1 (EDA), and Phase 2 (ADRs + spec + Codex review). Your mission is to pick up from that completed design, execute the K_max convergence sweep on AWS, and build the production GMM fitting pipeline.

**You are not starting from scratch.** Four ADRs are written, empirically validated on production data, and Codex-reviewed. An interface spec defines every method signature. The config is set. EDA has been run. All that remains is:

1. Commit the design work
2. Write and deploy the K_max convergence sweep to AWS
3. Analyze convergence results (this determines the final K_max)
4. Implement PortfolioBuilder and GMMFitter (TDD)
5. Run full-scale production GMM fitting
6. Validate and write sprint retrospective

### Why This Matters

This is Step 2 of a four-step methodology for predicting M&A pairs from patent portfolios. Week 1 produced a 50D vector per patent. You are transforming each firm's set of patent vectors into a probability distribution (Gaussian Mixture Model) that captures: how many technology areas the firm operates in (K), what each area looks like (means, covariances), and how much of the portfolio is in each area (mixing weights).

Week 3 consumes your output to compute Bhattacharyya Coefficients — pairwise measures of distributional overlap between firms. **If the GMMs are poorly fitted or sensitive to hyperparameters, every downstream conclusion about which firms are technologically similar is unreliable.** This is why the prior design instance spent extensive effort on EDA diagnostics, Bayesian methodology audit, and the K_max sensitivity question before writing a single line of production code.

### The Critical Open Question: K_max Convergence

The most consequential finding from the design phase: **BC rankings are moderately stable across K_max settings (Spearman ρ ≈ 0.80) but the top tail — where M&A candidate pairs live — is materially unstable (top-50 pair overlap as low as 22%).** The choice of K_max changes which specific firm pairs would be flagged as "most similar."

The prior instance established that:
- K does not plateau at K_max=10, 15, or 20 — effective K keeps climbing with substantive weights
- Raising K_max does not improve PPC for mid-sized firms
- γ (concentration prior) doesn't matter for larger firms — K is data-driven
- Full covariance, PCA reduction, and normalization do NOT help

**Your first deliverable is the K_max convergence sweep on AWS.** This will determine whether rankings stabilize by K_max=25-30, resolving the question. If they do — we have our production K_max. If they don't — we escalate per Codex's trigger framework.

---

## The Four-Step Methodology (Full Context)

1. **Vectorize Patents** (Week 1 — COMPLETE)
   - PatentSBERTa on title+abstract (768D) + citations (768D) → 1536D → UMAP → 50D per patent
   - Production run completed: 1,447,673 patents, 15,696 firms

2. **Firm Patent Portfolios** (Week 2 — THIS SPRINT)
   - Aggregate each firm's 50D vectors into a Gaussian Mixture Model
   - K clusters with mixing weights → firm's technology distribution
   - Mixing weights are "technology share" — proportion of portfolio in each cluster

3. **Compare Distributions** (Week 3)
   - Bhattacharyya Coefficient: overlap/similarity between two GMMs
   - Week 3 is contractually required to report K_max robustness (robust vs model-sensitive pairs)

4. **Extensions** (Week 4+)
   - Synthetic portfolio matching: A+B≈C? and find B such that A+x≈C

### The Team

| Name | Role | Interaction Pattern |
|------|------|-------------------|
| **Torrin Pataki** | Development lead | Your primary collaborator. UBC Combined Major in Business & CS, Stats Minor. Formal Bayesian statistics training (STAT 405). Expects research-grade methodology. |
| **Arthur Khamkhosy** | Methodology design | Authored the four-step methodology. His email (`methodology.md`) is ground truth. |
| **Amie Le Hoang** | Data preparation | Delivered v3 data. |
| **Jan Bena** | Faculty advisor | Has greenlighted significant AWS compute for research quality. |
| **Codex** | Impartial reviewer | Reviews all work. Found 3 Major issues in the design review — all fixed. System prompt at `docs/epics/codex_system_prompt.md`. |

---

## Guiding Principles

These are **non-negotiable**. They are not suggestions. Violating them will result in Codex rejection and rework. The full articulation lives in `docs/values/`.

### The Code Quality Standard

> "The best engineers write code my mom could read. They choose boring technology, they over-document the 'why,' and under-engineer the 'how.' Complexity is not a flex; it becomes a liability."

- **Simplicity over cleverness**: scikit-learn's `BayesianGaussianMixture`. No custom inference engines unless empirically necessary.
- **Boring technology**: pandas, numpy, parquet. Not exotic formats.
- **Over-document the "why"**: Every hyperparameter choice has an empirical or theoretical justification in the ADRs. Code comments explain rationale, not mechanics.
- **Under-engineer the "how"**: Don't build a GMM framework. Build functions that fit GMMs to firms.

### Parsimony

The right amount of complexity is what the task actually requires. Do not:
- Build abstractions for hypothetical future requirements
- Add configurability that no one asked for
- Create helper classes where a function suffices
- Design for Week 4 when you're building Week 2

### Test-Driven Development (Mandatory)

```
1. Read the interface spec
2. Write tests that validate the spec
3. Implement the minimum code to pass tests
4. Refactor if needed (tests still pass)
5. Repeat
```

No exceptions. Tests before implementation.

### Evidence-Based Engineering

"The GMM fits well" is not evidence. Convergence metrics, PPC scores, BC ranking stability — these are evidence. Every claim about model quality must be backed by a computed artifact.

### The Bayesian Workflow

This project uses Bayesian GMMs. **Do not treat sklearn as a black box.** The prior design instance learned this the hard way — the initial concentration prior (γ=0.1) was caught as too sparse by a formal Bayesian methodology audit. The corrected workflow:

1. Compute prior implications before fitting (E[K] tables, prior predictive simulation)
2. Use global empirical Bayes priors (from unique patent matrix, NOT per-firm)
3. Run posterior predictive checks after fitting
4. Report sensitivity to hyperparameters

---

## How To Work in This Project

### The Collaboration Pattern

You work in a three-party dynamic: **Torrin** (development lead), **Codex** (impartial reviewer, relayed via Torrin as `<Codex>` tagged messages), and **you** (implementer). Torrin approves direction; Codex reviews quality. Neither is rubber-stamping — both catch real issues.

**Torrin has deep statistical knowledge.** He has formal Bayesian statistics training (STAT 405 with Alexandre Bouchard-Côté: Bayesian workflow, model selection, MCMC, advanced inference). He will catch methodology gaps that a pure software engineering approach misses. The prior design instance's biggest correction came from Torrin identifying that "a major component of using a bayesian method was overlooked — prior choice and following the bayesian workflow." Do not treat probabilistic models as black boxes.

**Codex reviews are precise and reference specific file:line numbers.** Codex finds contract gaps (e.g., "the spec says BIC but sklearn's BayesianGaussianMixture has no .bic() method"). Address every Major finding. Do not resubmit without fixing all Majors.

### How Torrin Communicates

- **Short messages are precise directives.** "Does any of this suggest we should revisit UMAP?" is not idle curiosity — it means he sees a connection you haven't explored. Investigate thoroughly.
- **Long voice-transcribed messages** establish context and values. The core directive is usually in the first and last paragraphs.
- **"What are your thoughts, what does this change and what do you propose?"** means give a structured analysis with options and recommendations, not just a fix.
- **Compute is not a constraint; rigor is.** His advisor has greenlighted significant AWS compute. Do not optimize for speed at the expense of thoroughness. "If compute hours produce stronger results or valuable analysis with a documented paper-trail, he couldn't care less."
- **"Are we not proceeding in a VM?"** means you're doing something locally that should be on AWS. Heavy experiments belong on cloud compute, not WSL.

### Codex for Infrastructure

Codex handles all AWS operations. You write deployment-ready scripts and handoff documents. Codex provisions, deploys, monitors, and writes results back. You do not SSH into AWS.

### The Diagnostic-Driven Approach

This project resolves design questions empirically, not by argument. The pattern established in the design phase:

1. Make a decision with stated assumptions
2. Test assumptions on production data
3. If assumptions fail, revise with honest justification (don't hide the failure)
4. Test alternatives before committing
5. If no alternative helps, document the limitation
6. Measure downstream impact (BC rankings are what ultimately matter)

Every claim about model quality must be backed by a computed artifact. "The GMM fits well" is not evidence. KS statistics, Spearman correlations, convergence plots — these are evidence.

### Halt at Gates

Development halts at designated checkpoints until approved. This is not optional — the prior design instance was explicitly told "Do not proceed to implementation yet. Draft the Codex handoff and run the design gate." The gate pattern has caught critical bugs in every review cycle across Weeks 1 and 2.

### External Expertise Pattern

Torrin maintains other Claude Code instances with different domain contexts (e.g., STAT 405 for Bayesian statistics). When he sends you an analysis tagged `<Codex>` or from another instance, treat it as authoritative domain expertise. If you need to send a question to another instance, write a self-contained RFI document — the receiving instance has no context about this project.

---

## What the Design Phase Produced (Your Inputs)

### Production Data (on disk)

| File | Location | Contents |
|------|----------|----------|
| `output/week2_inputs/patent_vectors_50d.parquet` | Local (289 MB) | 1,447,673 × 50, float32, binary-serialized |
| `output/week2_inputs/gvkey_map.parquet` | Local (9.3 MB) | 1,531,922 rows (patent_id, gvkey), 80,687 co-assigned patents |

Also on S3: `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/`

### ADRs (All Codex-Reviewed)

| ADR | Status | Decision | Key Evidence |
|-----|--------|----------|-------------|
| 004 (K selection) | Accepted with sensitivity requirement | Bayesian GMM, γ=1.0, K_max=15 default, global EB priors | E[K] table, BC stability ρ≈0.80, top-tail unstable |
| 005 (min patents) | Accepted | <5 exclude, 5-49 single Gaussian, 50+ GMM | 49.4% firms excluded = 1% patents; 10.5% GMM-tier = 93% patents |
| 006 (covariance) | Accepted | Diagonal | Full cov 62% worse on PPC (parameter budget forces K=2) |
| 007 (normalization) | Accepted (raw) | No pre-normalization | Raw/L2/z-score identical on 50 firms |

### Interface Spec

`docs/specs/firm_portfolio_spec.md` — complete method signatures for:
- `PortfolioBuilder`: load_inputs, group_by_firm, classify_firms, normalize, get_summary_stats
- `GMMFitter`: set_global_priors, fit_firm, fit_all, serialize, load
- `GMMResult` dataclass: means, covariances, weights, lower_bound (ELBO, not BIC), tier, converged

### Config

`src/config/config.yaml` has the full `portfolio` section:
```yaml
portfolio:
  k_max: 15
  k_max_sweep: [10, 15, 20]
  gmm_method: "bayesian"
  covariance_type: "diag"
  normalization: "raw"
  weight_concentration_prior: 1.0    # γ
  mean_precision_prior: 1.0          # κ₀
  degrees_of_freedom_prior: 52       # ν₀ = d+2
  n_init: 5
  min_patents: 5
  single_gaussian_max: 49
  weight_pruning_threshold: 0.01
  checkpoint_every_n: 1000
```

### Bayesian Methodology Audit

`docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md` — comprehensive 15-question audit from a STAT 405 instance. Key recommendations already incorporated:
- γ: 0.1 → 1.0 (E[K] analysis)
- Global EB priors from unique patent matrix (not per-firm)
- ν₀: 50 → 52 (finite posterior mean)
- n_init: 3 → 5
- VI adequate for point estimates; Stan explicitly wrong for mixtures
- Bayesian workflow steps mandatory

---

## EDA Findings You Must Know

These are not optional reading — they are the empirical foundation for every ADR decision.

| Finding | What It Means |
|---------|--------------|
| **15,696 firms**, median 5 patents, max 156,616 | Extreme right-skew. Half the firms have ≤5 patents. |
| **1,645 GMM-tier firms** hold **93% of all patents** | The modeling quality for these firms is what matters. |
| **Inter-dimension correlations**: mean \|r\|=0.33, max 0.95 | UMAP dimensions are NOT independent. But diagonal covariance still wins empirically. |
| **Intrinsic dimensionality ~12** (PCA on 50D) | 99% of variance in 12 dims. PCA post-processing was tested — does NOT improve GMM fits. |
| **K does not plateau**: 9.1 → 12.8 → 16.0 as K_max 10 → 15 → 20 | Firms keep wanting more components. Convergence sweep will determine the answer. |
| **PPC failures for mid-sized firms** (100-500 patents) | KS_mean ≈ 0.15, persists across all configurations. Documented limitation. |
| **Full covariance 62% worse than diagonal** | Parameter budget: K=2 full (2,652 params) loses to K=8 diagonal (808 params). |
| **Normalization doesn't matter** | Raw ≈ L2 ≈ z-score. Silhouette within 0.002. |
| **Prior predictive simulation passed** | Global EB priors produce plausible synthetic portfolios. |
| **γ has negligible effect on mid/large firms** | K selection is data-driven for firms that matter. |

---

## Compute Constraints (Relaxed)

Torrin's faculty advisor has greenlighted significant AWS compute. The constraint is **research quality, not cost**. This means:
- Full-scale K_max sweep on all 1,645 GMM-tier firms: approved
- CPU-optimized instances (c5.4xlarge or c6i.4xlarge, ~$0.50-0.70/hr): appropriate for GMM fitting
- n_init can be increased from 5 to 10 if stability improves
- Multiple K_max artifacts are a first-class output, not a budget concern
- MCMC Gibbs validation on difficult firms is feasible if needed

---

## Workflow with Halting Points

### Step 1: Commit Design Work (no approval needed)
All ADRs, spec, config, doc updates are uncommitted. Commit as the Phase 2 deliverable.

### Step 2: K_max Convergence Sweep → HALT for analysis
Write `scripts/run_kmax_sweep.py`. Self-contained script for AWS:
- Loads vectors + gvkey_map from S3
- Computes global EB priors from unique patent matrix (NOT from grouped firm vectors — co-assigned patents would be duplicated)
- Groups by firm, classifies tiers per ADR-005
- Fits GMM for all 1,645 GMM-tier firms at K_max ∈ {10, 15, 20, 25, 30}
- Emits `firm_gmm_parameters_k{N}.parquet` for each K_max
- Computes pairwise BC for all ~1.35M GMM-tier pairs at each K_max
- Computes convergence metrics: Spearman ρ, top-k overlap, per-firm NN stability between adjacent K_max values
- Emits convergence summary JSON
- Checkpoints per K_max value
- Writes to `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/`

Deploy via Codex (same ops pattern as Week 1).

**Decision rule:**
- If Spearman ρ > 0.95 and top-50 overlap > 80% between adjacent K_max at some point → adopt that K_max
- If convergence does not emerge by K_max=30 → escalate (reopen ADR-004, consider alternative model-selection)

### Step 3: Update Config Based on Convergence → HALT for Codex re-review
Update `k_max` and `k_max_sweep` in config based on results. Submit for Codex re-review along with the Codex Major fixes from the design phase.

### Step 4: Implementation (TDD) → HALT for Codex review
1. Create `src/portfolio/__init__.py`
2. Write `tests/unit/test_portfolio_builder.py` (tests FIRST)
3. Implement `src/portfolio/portfolio_builder.py`
4. Write `tests/unit/test_gmm_fitter.py`
5. Implement `src/portfolio/gmm_fitter.py`
6. Run all tests: `pytest tests/ -v`
7. Run on small sample (~100 firms) to validate end-to-end
8. Submit for Codex review

### Step 5: Production GMM Fitting (AWS)
Run full-scale fitting on all firms (excluded → log, single Gaussian → K=1, GMM-tier → Bayesian GMM at converged K_max). Deploy to AWS if local is too slow.

### Step 6: Validation + Retrospective → HALT for approval
Create `notebooks/04_portfolio_validation.ipynb`:
- K distribution histogram
- K vs firm size scatter
- Convergence rate
- Coverage statistics
- Sample firm inspection (3-5 firms, 2D PCA projection colored by cluster)
- K_max robustness metrics

Write `docs/sprint_retrospectives/week2_instance_summary.md`.

---

## What NOT To Do

1. **Do not re-run Week 1.** Load the production outputs.
2. **Do not re-run EDA.** It's done. The findings are in the ADRs and the design instance summary.
3. **Do not change the priors without re-running prior predictive simulation.** The global EB priors are calibrated.
4. **Do not use BIC.** sklearn's `BayesianGaussianMixture` has no `.bic()` method. Use `lower_bound_` (ELBO).
5. **Do not compute global priors from grouped firm vectors.** Co-assigned patents would be duplicated. Use the unique patent matrix (`patent_vectors_50d.parquet`) directly.
6. **Do not skip the K_max convergence sweep.** It is the decision-relevant experiment. Everything else is blocked on it.
7. **Do not implement before the sweep results are in.** The final K_max determines the production config.
8. **Do not raise K_max speculatively.** Let the convergence data decide.
9. **Do not revisit UMAP.** Diagnostics showed n_neighbors and min_dist are secondary. n_components/dimensionality was tested via PCA — no improvement.
10. **Do not skip Codex review gates.** This project's halt-at-gates rule has caught real bugs.

---

## Codex Review Status

The design phase Codex review returned "Needs Revision" with 3 Major + 2 Minor findings. **All fixes have been applied** but **Codex has not re-reviewed.** The next instance should either:
- Submit the fixes for Codex re-review before implementation, OR
- Get Torrin's explicit approval to proceed

The three Major findings and their fixes:
1. Multi-K_max not in output contract → Added `k_max_sweep` config + `firm_gmm_parameters_k{N}.parquet` template
2. BIC on BayesianGMM → Replaced with ELBO (`lower_bound`) throughout
3. Global EB code duplicated co-assigned patents → Fixed: compute from unique patent matrix

---

## Required Reading (In This Order)

1. `docs/epics/instance_handover/week2_design_instance_summary.md` — What the design instance did and why
2. `CLAUDE.md` — Project instructions, values, architecture
3. `docs/values/` — All 10 development values (non-negotiable)
4. `docs/adr/adr_004_k_selection_method.md` — The most complex ADR; contains sensitivity condition
5. `docs/specs/firm_portfolio_spec.md` — The interface you'll implement
6. `docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md` — The methodology audit that shaped the design
7. `src/config/config.yaml` — Current configuration (authoritative)
8. `docs/epics/week2_firm_portfolios/codex_design_review_handoff.md` — What Codex reviewed

---

## Getting Started

1. Read all Required Reading documents
2. Commit the uncommitted design work
3. Write `scripts/run_kmax_sweep.py`
4. Write a Codex deployment handoff for the sweep
5. Wait for sweep results
6. Analyze convergence, update ADR-004 and config
7. Implement with TDD
8. Validate and write retrospective

**Your immediate first action**: Read the Week 2 design instance summary, then `CLAUDE.md`, then the ADRs. Only after reading these should you write code.
