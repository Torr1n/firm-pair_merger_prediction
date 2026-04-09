# Instance Summary: Week 2 Design Session

**From**: Claude Code (Week 2 design instance)
**To**: Incoming development instance
**Date**: 2026-04-08
**Repository**: `git@github.com:Torr1n/firm-pair_merger_prediction.git` (branch: `main`)
**HEAD at handover**: Uncommitted — all Week 2 design work is staged but not committed. Previous HEAD is `736b968`.

---

## Purpose of This Document

You are picking up the firm-pair merger prediction project at the transition from Week 2 design (complete) to Week 2 implementation (not started). This session completed Phase 2 (ADRs + spec + EDA) and went through one Codex review cycle. The design is approved with fixes applied. The immediate next step is writing and deploying the K_max convergence sweep script to AWS, followed by Phase 3 implementation.

---

## What This Session Accomplished

### Phase 0: Production Output Validation (COMPLETE)
- Downloaded `patent_vectors_50d.parquet` (289 MB) and `gvkey_map.parquet` (9.3 MB) from S3
- Located at `output/week2_inputs/`
- All counts match Codex Week 1 handoff: 1,447,673 patents, 15,696 firms, 80,687 co-assigned patents
- Join integrity: perfect (zero orphans)

### Phase 1: EDA (COMPLETE)
Extensive diagnostics on production data. Key findings that shaped the design:

| Finding | Impact on Design |
|---------|-----------------|
| **Firm-size tiers validated**: 49.4% excluded (<5 patents) = only 1% of patents lost; 10.5% GMM-tier (50+) = 93% of patents | ADR-005 accepted |
| **Inter-dimension correlations strong** (mean \|r\|=0.33, max 0.95) | Falsified ADR-006's "weakly correlated" assumption; revised justification to "best practical choice" |
| **Intrinsic dimensionality ~12** (PCA: 99% variance in 12 dims out of 50) | Investigated PCA post-processing; found it does NOT improve GMM fits; kept raw 50D |
| **Normalization doesn't matter** (raw/L2/z-score produce identical results) | ADR-007 finalized as raw |
| **K does not plateau** (climbs from 9→13→16 as K_max increases 10→15→20) | K_max=10 is binding; raised to 15 operational default |
| **PPC failures for mid-sized firms** persist across all configurations | Model misspecification (diagonal Gaussian), not a tuning problem |
| **Full covariance is 62% WORSE** than diagonal on PPC | Parameter budget forces K=2 for full cov; diagonal with K=8-10 wins |
| **BC rankings moderately stable** (Spearman ρ≈0.80) but **top-tail unstable** (top-50 overlap 22-48%) | K_max sensitivity elevated to methodological condition |
| **γ (concentration prior) has negligible effect** on mid/large firms | K selection is data-driven |
| **Prior predictive simulation passed** (global EB priors well-calibrated) | Priors validated |

### Phase 2: ADRs + Spec (COMPLETE, CODEX-REVIEWED)

| ADR | Status | Decision |
|-----|--------|----------|
| **004** (K selection) | Accepted with sensitivity requirement | Bayesian GMM, γ=1.0, K_max=15 default, global EB priors, ELBO (not BIC), multi-K_max output |
| **005** (min patents) | Accepted | <5 exclude, 5-49 single Gaussian, 50+ GMM |
| **006** (covariance) | Accepted | Diagonal — best practical choice under parameter budget constraints |
| **007** (normalization) | Accepted (raw) | No pre-normalization |

**Spec**: `docs/specs/firm_portfolio_spec.md` — PortfolioBuilder + GMMFitter with `set_global_priors()`, `GMMResult` dataclass (ELBO not BIC), multi-K_max output contract, Bayesian workflow requirements.

### Codex Review Cycle (COMPLETE)
First review returned "Needs Revision" with 3 Major + 2 Minor findings. All fixed:

| Finding | Fix |
|---------|-----|
| [Major] Multi-K_max not in output contract | Config: `k_max_sweep: [10,15,20]`. Output: `firm_gmm_parameters_k{N}.parquet` per K_max |
| [Major] BIC on BayesianGMM (no `.bic()` method) | Replaced with `lower_bound` (ELBO) throughout |
| [Major] Global EB code concatenated firm vectors (duplicates co-assigned patents) | Fixed: compute from unique patent matrix before grouping |
| [Minor] ADR-005 stale K_max=10 | Updated to "per ADR-004 (default 15)" |
| [Minor] ADR-006 stale revisit trigger | Updated: trigger has fired, resolved empirically |

**Codex has NOT re-reviewed after fixes.** The next instance should submit for re-review or proceed if Torrin approves.

### Bayesian Methodology Audit (COMPLETE)
Sent RFI to Torrin's STAT 405 Claude Code instance. Received comprehensive audit (`docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md`). Key changes driven by audit:

- γ: 0.1 → 1.0 (E[K] table showed γ=0.1 suppresses multi-component solutions)
- Priors: per-firm empirical Bayes → global empirical Bayes (avoids double-counting)
- ν₀: 50 → 52 (d+2 for finite posterior mean)
- n_init: 3 → 5
- VI validated as adequate for point estimates; Stan/NUTS explicitly wrong for mixtures
- Bayesian workflow (prior predictive sim, PPC, sensitivity audit) made mandatory

---

## What Was NOT Done

### Not Started
- **Phase 3 implementation** — no code written for PortfolioBuilder or GMMFitter
- **K_max convergence sweep on AWS** — script not written, instance not provisioned
- **Phase 4 validation notebook** — blocked on implementation
- **Sprint retrospective** — blocked on completion
- **`__init__.py` files for `src/portfolio/`** — directory has only `.gitkeep`
- **Tests** — no test files written (TDD: tests come first in Phase 3)
- **Uncommitted work** — everything from this session is uncommitted

### Partially Complete
- **K_max convergence sweep design** — the plan is agreed (1,645 GMM-tier firms, K_max ∈ {10,15,20,25,30}, full BC ranking analysis), but the script is not written
- **Codex re-review** — fixes applied but not re-submitted

### Explicitly Deferred
- UMAP hyperparameter revisit (diagnostics showed it's not the primary issue)
- PCA post-processing (tested, does not improve GMM fits)
- MCMC Gibbs validation (high value but requires custom sampler implementation)
- t-mixture upgrade (only if >20% of clusters show non-Gaussianity in Week 3)

---

## Key Decisions and Their Reasoning

### Why Bayesian GMM over BIC Sweep?
One fit per firm vs K_max fits. E[K] scales logarithmically with firm size (γ·log(n)), which is economically meaningful. BIC can behave poorly in 50D. The Dirichlet process prior with global empirical Bayes hyperparameters provides principled regularization.

### Why K_max=15 and Not 10 or 20?
K=10 is clearly binding (35% ceiling rate). K=20 is not converged (firms keep wanting more). K=15 is the operational default, but K_max sensitivity is a **reported methodological condition**: Week 2 must produce fits at {10, 15, 20} minimum, and Week 3 must report robustness. The K_max convergence sweep (planned for AWS) will determine whether rankings stabilize by K=25-30.

### Why Diagonal Covariance Despite Strong Correlations?
The original "weakly correlated" justification was empirically falsified (mean |r|=0.33). But a direct diagnostic showed full covariance is 62% WORSE on PPC because it burns the parameter budget on per-component covariance structure (1,275 params/component) instead of more components. K=8-10 diagonal components outperform K=2 full-covariance components. This is a parameter budget argument, not an independence argument.

### Why Raw Normalization?
All three options (raw, L2, z-score) produce identical results on 50 mid-sized firms — silhouette within 0.002, K within 0.2, 96% K agreement. Despite 16x dimension scale variation, diagonal covariance adapts per-dimension by construction.

### Why Global Empirical Bayes Instead of sklearn Defaults?
sklearn defaults set `mean_prior=mean(X_firm)` and `covariance_prior=empirical_cov(X_firm)` per firm — using each firm's data twice. Global EB computes priors from the pooled unique patent matrix (1,447,673 × 50) once, then applies as fixed priors to each firm. This approximates a hierarchical model: small firms shrink toward global structure, large firms are nearly unaffected.

### Why ELBO Instead of BIC?
sklearn's `BayesianGaussianMixture` does not provide `.bic()`. ELBO (variational lower bound on log-evidence) serves the same purpose as a model evidence proxy. This was caught as a Major finding in Codex review.

---

## The K_max Convergence Problem

This is the most important open question in the design. Summary:

**The problem**: BC rankings are moderately stable across K_max (Spearman ρ≈0.80) but the top tail — where M&A candidate pairs live — is materially unstable (top-50 overlap as low as 22%). The choice of K_max changes which specific firm pairs would be flagged as "most similar."

**What we know**:
- K does not plateau at 10, 15, or 20 — effective K keeps climbing with substantive weights
- Raising K_max does not improve PPC for mid-sized firms
- Full/tied covariance does not help
- PCA dimensionality reduction does not help
- γ doesn't matter for larger firms — K is data-driven

**The plan**: Run a full K_max convergence sweep on AWS. All 1,645 GMM-tier firms. K_max ∈ {10, 15, 20, 25, 30}. Compute BC rankings between all ~1.35M GMM-tier pairs at each K_max. Measure convergence via Spearman ρ, top-k overlap, and per-firm NN stability between adjacent K_max values.

**Decision rule** (from Codex):
- If rankings stabilize by K_max=25-30 → adopt the converged K_max
- If rankings don't stabilize → K_max is an unresolved model-selection problem; reopen ADR-004

**Codex's revisit triggers** (for the overall model family):
1. K_max fails to converge by K=30
2. Top-tail remains unstable even if bulk correlations improve
3. Effective K keeps climbing with substantive weights (already happening)
4. Converged K_max is so large that interpretability breaks down
5. Known merger pairs are sensitive to K_max (testable in Week 3)
6. Runtime becomes disproportionate
7. PPC remains poor at converged K_max

**What would NOT trigger revisiting**: strong correlations alone, imperfect PPC alone, high K alone, moderate sensitivity at low K_max.

---

## Compute Constraints (Updated)

Torrin's faculty advisor (Jan Bena) has greenlighted significant AWS compute. The constraint is no longer cost but research quality. This means:
- Full-scale K_max sweep on all GMM-tier firms is approved
- n_init can be increased (currently 5, could go to 10)
- MCMC Gibbs validation on difficult firms is feasible if needed
- Multiple K_max artifacts for production are not a budget concern

For the sweep, a CPU-optimized instance (c5.4xlarge or c6i.4xlarge, ~$0.50-0.70/hr) is appropriate. No GPU needed for GMM fitting.

---

## File Inventory

### New Files Created This Session
| File | Purpose |
|------|---------|
| `docs/adr/adr_004_k_selection_method.md` | K selection ADR |
| `docs/adr/adr_005_minimum_patent_threshold.md` | Minimum patent threshold ADR |
| `docs/adr/adr_006_covariance_type.md` | Covariance type ADR |
| `docs/adr/adr_007_prenormalization_strategy.md` | Normalization ADR |
| `docs/specs/firm_portfolio_spec.md` | PortfolioBuilder + GMMFitter interface spec |
| `docs/epics/week2_firm_portfolios/bayesian_gmm_rfi.md` | RFI to STAT 405 instance |
| `docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md` | Audit response (from STAT 405) |
| `docs/epics/week2_firm_portfolios/codex_design_review_handoff.md` | Codex review handoff |
| `docs/epics/week2_firm_portfolios/week1_full_run_handoff.md` | Codex production run handoff |
| `docs/epics/instance_handover/week2_design_instance_summary.md` | This document |
| `output/week2_inputs/patent_vectors_50d.parquet` | Production 50D vectors (289 MB) |
| `output/week2_inputs/gvkey_map.parquet` | Production firm mapping (9.3 MB) |

### Modified Files
| File | Change |
|------|--------|
| `src/config/config.yaml` | Added `portfolio` + `output_portfolios` sections |
| `CLAUDE.md` | Updated data table to v3, added portfolio config docs |
| `docs/adr/adr_001_patent_embedding_model.md` | Status: Proposed → Accepted |
| `docs/adr/adr_002_citation_aggregation.md` | Status: Proposed → Accepted |
| `docs/specs/patent_vectorizer_spec.md` | Reviewers: pending → approved |

### Nothing is committed. All changes are working-tree modifications.

---

## Immediate Next Actions (Priority Order)

### 1. Commit the design work
All ADRs, spec, config, and doc updates should be committed as the Phase 2 design deliverable.

### 2. Write the K_max convergence sweep script
`scripts/run_kmax_sweep.py` — self-contained script for AWS:
- Loads vectors + gvkey_map from S3
- Computes global EB priors from unique patent matrix
- Groups by firm, classifies tiers (per ADR-005)
- Fits GMM for all 1,645 GMM-tier firms at K_max ∈ {10, 15, 20, 25, 30}
- Emits `firm_gmm_parameters_k{N}.parquet` for each K_max
- Computes pairwise BC for all ~1.35M GMM-tier pairs at each K_max
- Computes convergence metrics (Spearman ρ, top-k overlap, NN stability)
- Emits convergence summary JSON
- Checkpoints per K_max value
- Writes to `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/`

### 3. Deploy to AWS via Codex
Write a Codex deployment handoff (same pattern as Week 1). CPU-optimized instance (c5.4xlarge or similar).

### 4. Analyze convergence results
When sweep completes:
- If K_max converges → update ADR-004 with final K_max, update `k_max_sweep` in config
- If K_max does not converge → escalate per Codex's trigger framework

### 5. Submit for Codex re-review (or proceed if Torrin approves)
The Codex review findings have been fixed but not re-reviewed. Either submit fixes for re-review or get Torrin's approval to proceed.

### 6. Phase 3: Implementation (TDD)
Only after sweep results and design approval:
1. Create `src/portfolio/__init__.py`
2. Write `tests/unit/test_portfolio_builder.py` (tests first)
3. Implement `src/portfolio/portfolio_builder.py`
4. Write `tests/unit/test_gmm_fitter.py`
5. Implement `src/portfolio/gmm_fitter.py`
6. Write the full-scale portfolio fitting script
7. Run on AWS at the converged K_max values

### 7. Phase 4: Validation + Retrospective
Validation notebook, sprint retrospective, Codex final review.

---

## How to Bootstrap

Read these documents in order:
1. **This document** — full session context
2. `docs/epics/instance_handover/instance_summary.md` — Week 1 handover (project origin, team, process)
3. `CLAUDE.md` — project instructions and values
4. `docs/epics/week2_firm_portfolios/bootstrap_prompt.md` — Week 2 mission briefing
5. `docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md` — the Bayesian methodology audit that shaped the design
6. `docs/specs/firm_portfolio_spec.md` — the interface spec you'll implement
7. `src/config/config.yaml` — current configuration (authoritative)

Then start with action #1 (commit) and #2 (sweep script).

---

## How This Session Worked: Patterns and Standards

### The Three-Way Collaboration Pattern

This session operated with a three-party dynamic: **Torrin** (development lead, approval authority), **Codex** (impartial reviewer, relayed through Torrin), and **Claude Code** (implementer). The communication flow:

1. Claude produces work (ADRs, EDA, diagnostics)
2. Torrin reviews, pushes back where needed, adds domain expertise
3. When design is ready, Torrin relays to Codex with `<Codex>` tagged messages
4. Codex reviews with structured findings (Major/Minor/Note)
5. Claude addresses findings, Torrin confirms
6. Cycle until approved

**Torrin does NOT just rubber-stamp.** His most impactful intervention this session was catching that the Bayesian workflow was overlooked: "I feel like a major component of using a bayesian method was overlooked - prior choice and following the bayesian workflow." He provided his STAT 405 course context and a detailed AI Mode conversation about prior elicitation, then directed creation of the RFI for his STAT 405 instance. This fundamentally reshaped ADR-004. The lesson: **Torrin has deep statistical knowledge and will catch methodology gaps that a pure software engineering approach misses.**

### Codex's Review Style

Codex reviews are structured, precise, and reference specific file:line numbers. Codex does NOT just approve — it finds real contract gaps. This session's review caught three Major issues (multi-K_max output contract missing, BIC on BayesianGMM impossible, global EB code duplicating co-assigned patents). Codex also provides strategic framing beyond line-level issues: the K_max sensitivity decision framework ("If rankings are highly stable... keep K_max=10. If they shift materially... raise K_max. If unstable between 15 and 20... pause.") came from Codex.

**Always take Codex reviews seriously.** Address every finding explicitly. Do not resubmit without fixing all Majors.

### How Torrin Gives Direction

Torrin's messages this session ranged from terse ("Interesting - before we continue, what about the other relevant hyperparameters n_neighbors and min_dist?") to extensive voice-transcribed context (the Bayesian workflow intervention with course syllabus and AI Mode conversation). Key patterns:

- **Short messages are precise directives.** When Torrin asks a brief question, it's because he sees something you missed. Don't treat it as optional.
- **Long messages establish context and values.** The Bayesian workflow message wasn't just "fix the priors" — it was establishing that this project operates at a research-grade methodology standard, grounded in his formal training.
- **"What are your thoughts, what does this change and what do you propose?"** — This is Torrin inviting you to think architecturally. Give a structured analysis with options and recommendations, not just a fix.
- **Compute is not a constraint; rigor is.** Torrin explicitly said his advisor "couldn't care less" about compute cost if it produces stronger results. Don't optimize for speed when you should optimize for quality.
- **"Are we not proceeding in a VM?"** — Torrin will redirect if you're doing something that should be done differently. In this case, running a heavy experiment locally instead of on AWS.

### The Diagnostic-Driven Workflow That Emerged

This session established a pattern that should continue:

1. **Make a design decision** (e.g., diagonal covariance, K_max=10)
2. **Test the assumption empirically** (e.g., run PPC, compute correlations)
3. **If the assumption fails, revise the decision with honest justification** (e.g., "weakly correlated" → "best practical choice under parameter budget")
4. **Test alternatives before committing** (e.g., PCA post-processing, full covariance)
5. **If no alternative helps, document the limitation** (e.g., "PPC failures for mid-sized firms are model misspecification, not a tuning problem")
6. **Measure downstream impact** (e.g., BC ranking stability across K_max)

This is more rigorous than the typical "decide → implement → validate" pattern. It emerged because early assumptions were wrong (weak correlations, K_max=10 sufficient) and the diagnostic approach caught those errors before they were baked into code.

### When Torrin Involves External Expertise

Torrin maintains multiple Claude Code instances with different domain contexts. When he identifies a gap outside the current instance's expertise, he:

1. Asks the current instance to draft a precise RFI document
2. Sends it to the domain-specific instance
3. Brings back the analysis as input

This happened with the STAT 405 instance for the Bayesian methodology audit. The RFI must be **self-contained** — the receiving instance has no context about this project. Include: problem statement, data characteristics, current plan, specific questions, constraints.

### The Codex Relay for Infrastructure

Codex handles all AWS operations. The pattern:
1. Claude writes a handoff document (what to deploy, why, verification steps)
2. Torrin relays to Codex
3. Codex provisions infrastructure, deploys, monitors
4. Codex writes a results handoff back (see `week1_full_run_handoff.md`)

Claude does NOT SSH into AWS. Claude writes scripts that are deployment-ready. Codex executes them.

### Standards That Were Reinforced

- **Halt at gates.** Torrin explicitly said: "Do not proceed to implementation yet. Draft the Codex handoff and run the design gate." The project's halt-at-gates rule is not optional.
- **Commit what works.** The design work from this session is uncommitted — that should be the first action of the next instance.
- **When in doubt, test.** Every design question this session was resolved empirically, not by argument. The PCA experiment, the covariance diagnostic, the normalization comparison — all were run on production data, not decided by reasoning alone.
- **Present findings, then recommendations.** Show the data first, then say what you think it means. Torrin and Codex will form their own conclusions.

---

## What I'm Confident In

- **ADR-005 tiers**: Strongly validated by production data. The numbers are definitive.
- **ADR-007 raw normalization**: Three normalizations tested, identical results. Clear decision.
- **ADR-006 diagonal covariance**: Direct empirical comparison against full and tied. Parameter budget argument is solid.
- **Global EB priors**: Calibrated, prior predictive simulation passed, audit-recommended.
- **VI is adequate**: For point estimates feeding BC computation, not for posterior uncertainty.

## What I'm Not Confident In

- **K_max convergence**: We don't yet know if rankings stabilize. The sweep will answer this definitively. If they don't converge by K=30, the model family itself may be inadequate for tail ranking.
- **Mid-sized firm PPC failures**: These persist across all configurations tested. The diagonal Gaussian mixture genuinely struggles with firms in the 100-500 patent range. This may or may not affect BC rankings for those firms.
- **Whether the UMAP representation is optimal**: We showed n_neighbors and min_dist are secondary, but the intrinsic dimensionality (~12 in 50D space) is a real structural issue. We deferred this because PCA post-processing didn't help, but a UMAP re-run at n_components=15 might produce a fundamentally different embedding. This is a research question, not an engineering question.

## What I'd Do Differently

- **I should have tested sklearn's BIC limitation earlier.** The `BayesianGaussianMixture` missing `.bic()` was caught by Codex in review. I hit this exact error during EDA but only fixed it locally in the script, not in the spec.
- **The initial γ=0.1 was a poor default.** I should have computed the E[K] table before drafting ADR-004 rather than picking a value that "encourages sparsity." The Bayesian audit caught this.
- **The "weakly correlated" assumption in ADR-006 should have been tested before the ADR was written.** I assumed UMAP output would be independent and wrote the ADR around that assumption. The EDA falsified it. The decision survived (diagonal is still best) but the reasoning had to be rebuilt.
