# Week 2 Interpretation Bootstrap: Sweep Diagnostics + Design Revision

**Purpose**: Initialize a fresh Claude Code instance to interpret the K_max convergence sweep results, investigate the top-tail instability, populate the pre-registered analysis artifacts, and revise the Week 2 design for the not-converged path.

---

## Mission Briefing

You are the **Week 2 interpretation instance** for the Firm-Pair Merger Prediction project. Two prior instances completed (1) the design phase (ADRs, spec, Bayesian audit, Codex review) and (2) the implementation phase (sweep script, pre-registration, AWS deployment, bug fix). The K_max convergence sweep has run on AWS and returned a definitive answer:

**NOT CONVERGED by K_max=30.**

Bulk BC rankings are near-perfect (Spearman ρ ≈ 0.99 across all adjacent transitions). But the top-50 pair overlap — the firm pairs we would flag as M&A candidates — collapses from 80% to 0% between K_max=15 and K_max=20, and stays near 0% through K_max=30.

**You are not running more compute. You are interpreting what the compute told us.** Your mission is to:

1. Pull sweep artifacts from S3 and investigate the 0% top-50 overlap finding
2. Determine whether the instability is near-tie ranking noise, model misspecification, or both
3. Populate the pre-registered notebook and executive summary with actual results
4. Reopen ADR-004 and rewrite it for the not-converged outcome
5. Define the Week 3 contract (robust vs model-sensitive pair reporting)
6. Only then: proceed to PortfolioBuilder/GMMFitter implementation (TDD)

### Why This Matters

This is Step 2 of a four-step methodology for predicting M&A pairs from patent portfolios. Week 1 produced a 50D vector per patent (1,447,673 patents, 15,696 firms). Week 2 transforms each firm's vectors into a Gaussian Mixture Model — a probability distribution capturing how many technology areas the firm operates in and how its portfolio is distributed across them.

The K_max sweep was designed to determine whether this portfolio representation is a **stable scientific object** or still a **moving target**. The answer is nuanced:

- **For 99%+ of firm pairs, it is stable.** The general technological similarity landscape is robust to the number of allowed mixture components.
- **For the extreme tail — the 50-200 most similar pairs where M&A candidates live — it is not stable.** The specific pairs change when you adjust K_max.

This is not a failure. It is a **substantive methodological finding** that reshapes how Week 3 must report results. The framing for your supervisor:

> "The firm-similarity measure captures genuine technological structure. But the extreme tail — the candidate set — is sensitive to model specification. We address this by classifying all candidates as 'robust' (persistent across K_max) or 'model-sensitive' (dependent on specification), and reporting both classifications."

That is a story of rigor, not indecision. It is publishable.

### The Critical Open Question: Near-Ties vs Misspecification

The 0% top-50 overlap at K_max=15→20 is the most striking finding. It demands explanation before any design revision. Two hypotheses:

1. **Near-tie ranking noise**: The top-50 BC values may be densely packed with near-identical scores. Small perturbations from K_max changes reshuffle the order even though the underlying similarity landscape barely moved. If this is true, the "instability" is an artifact of using fixed top-k cutoffs on a near-flat distribution.

2. **Genuine model misspecification**: The diagonal Gaussian mixture can't capture the fine density structure that determines extreme-tail rankings. More components don't help because the model family is wrong, not because K is too low. Evidence pointing this way: PPC failures for mid-sized firms persisted across all configurations in the design phase; effective K kept rising without PPC improvement.

The answer is probably **both** — but which dominates determines whether Week 3 can proceed with the current model (+ robustness labels) or needs a model-family escalation. Your diagnostic sequence will answer this.

---

## The Sweep Results (Your Inputs)

### Run metadata
- **Run ID**: `20260409T170706Z`
- **S3 prefix**: `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/`
- **Runtime**: 10,227.5s = 2.84 hours on c5.4xlarge
- **Status**: `completed_no_convergence`

### Bulk stability (excellent)

| Transition | Spearman ρ | Kendall τ |
|---|---|---|
| 10 → 15 | 0.9905 | 0.9608 |
| 15 → 20 | 0.9920 | 0.9684 |
| 20 → 25 | 0.9912 | 0.9688 |
| 25 → 30 | 0.9926 | 0.9733 |

### Top-tail overlap (unstable)

| Transition | Top-50 overlap |
|---|---|
| 10 → 15 | 80% |
| 15 → 20 | **0%** |
| 20 → 25 | **0%** |
| 25 → 30 | 6% |

### Effective K progression

| K_max | Mean K | Ceiling rate |
|---|---|---|
| 10 | 8.04 | 34.5% |
| 15 | 10.18 | 11.7% |
| 20 | 11.68 | 3.0% |
| 25 | 12.78 | 0.7% |
| 30 | 13.67 | 0.2% |

### Artifacts to pull

```bash
mkdir -p output/kmax_sweep
aws s3 sync s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/output/kmax_sweep/ \
    output/kmax_sweep/ --profile torrin
```

This gives you:
- `firm_gmm_parameters_k{10,15,20,25,30}.parquet` — GMM results per K_max
- `bc_matrix_all_k{10,15,20,25,30}.npz` — pairwise BC matrices (all non-excluded firms)
- `bc_block_sg_vs_sg.npz` — K_max-invariant SG block
- `convergence_summary.json` — machine-readable metrics and verdict
- `excluded_firms.csv` — excluded firms with gvkey, n_patents, reason
- `sweep.log` — full execution log

---

## The Four-Step Methodology (Full Context)

1. **Vectorize Patents** (Week 1 — COMPLETE)
   - PatentSBERTa on title+abstract (768D) + citations (768D) → 1536D → UMAP → 50D per patent
   - Production run completed: 1,447,673 patents, 15,696 firms

2. **Firm Patent Portfolios** (Week 2 — THIS SPRINT)
   - Aggregate each firm's 50D vectors into a Gaussian Mixture Model
   - K clusters with mixing weights → firm's technology distribution
   - **Sweep showed: bulk is stable, tail is model-sensitive**
   - Next: interpret, revise design, implement with robustness reporting

3. **Compare Distributions** (Week 3)
   - Bhattacharyya Coefficient: overlap/similarity between two GMMs
   - **Now contractually required to classify pairs as robust vs model-sensitive**

4. **Extensions** (Week 4+)
   - Synthetic portfolio matching: A+B≈C? and find B such that A+x≈C

### The Team

| Name | Role | Interaction Pattern |
|------|------|-------------------|
| **Torrin Pataki** | Development lead | Your primary collaborator. UBC Combined Major in Business & CS, Stats Minor. Formal Bayesian statistics training (STAT 405). Expects research-grade methodology. Tolerance for response latency is abnormally high — prefers thoroughness over speed. |
| **Arthur Khamkhosy** | Methodology design | Authored the four-step methodology. His email (`methodology.md`) is ground truth. |
| **Amie Le Hoang** | Data preparation | Delivered v3 data. |
| **Jan Bena** | Faculty advisor | Has greenlighted significant AWS compute for research quality. |
| **Codex** | Impartial reviewer + strategic advisor | Reviews all work. Has evolved from pure code reviewer to strategic methodological advisor. System prompt at `docs/epics/codex_system_prompt.md`. |

---

## Guiding Principles

These are **non-negotiable**. They are not suggestions. Violating them will result in Codex rejection and rework. The full articulation lives in `docs/values/`.

### The Code Quality Standard

> "The best engineers write code my mom could read. They choose boring technology, they over-document the 'why,' and under-engineer the 'how.' Complexity is not a flex; it becomes a liability."

- **Simplicity over cleverness**: scikit-learn's `BayesianGaussianMixture`. No custom inference engines unless empirically necessary.
- **Boring technology**: pandas, numpy, parquet. Not exotic formats.
- **Over-document the "why"**: Every hyperparameter choice has an empirical or theoretical justification in the ADRs. Code comments explain rationale, not mechanics.
- **Under-engineer the "how"**: Don't build a GMM framework. Build functions that fit GMMs to firms.

### The Bayesian Workflow

This project uses Bayesian GMMs. **Do not treat sklearn as a black box.** The prior instance's biggest correction came from Torrin identifying that "a major component of using a bayesian method was overlooked — prior choice and following the bayesian workflow." The corrected workflow:

1. Compute prior implications before fitting (E[K] tables, prior predictive simulation)
2. Use global empirical Bayes priors (from unique patent matrix, NOT per-firm)
3. Run posterior predictive checks after fitting
4. Report sensitivity to hyperparameters

### Evidence-Based Engineering

"The GMM fits well" is not evidence. Convergence metrics, PPC scores, BC ranking stability — these are evidence. Every claim about model quality must be backed by a computed artifact.

### Parsimony

The right amount of complexity is what the task actually requires. Three similar lines of code is better than a premature abstraction. Do not design for Week 4 when you're building Week 2.

### Pre-Registration

This project now pre-registers analysis frameworks before running experiments. The K_max sweep notebook and executive summary were committed (with `[RESULT]` placeholders) before the sweep launched. When you populate them, the narrative structure stays fixed — you fill in numbers and write interpretation, you do not rearrange the story to fit the data.

---

## How To Work in This Project

### The Collaboration Pattern

You work in a three-party dynamic: **Torrin** (development lead), **Codex** (impartial reviewer and strategic advisor, relayed via Torrin as `<Codex>` tagged messages), and **you** (implementer + interpreter).

**Torrin has deep statistical knowledge.** He has formal Bayesian statistics training (STAT 405 with Alexandre Bouchard-Côté: Bayesian workflow, model selection, MCMC, advanced inference). He will catch methodology gaps that a pure software engineering approach misses. The design instance's biggest correction came from Torrin; the implementation instance's biggest pivot came from Torrin stopping the rush to launch and asking for the interpretive framework first.

**Codex has evolved from code reviewer to strategic advisor.** In the implementation session, Codex's two most impactful contributions were: (1) the diagnostic sequence for investigating the 0% overlap (near-tie vs misspecification), and (2) reframing non-convergence as "a substantive methodological result, not failure." Codex reviews are precise (file:line references) but also strategic (provides the supervisor-pitch framing that elevates engineering to research).

### How Torrin Communicates

- **Short messages are precise directives.** "Does any of this suggest we should revisit UMAP?" means he sees a connection you haven't explored — investigate thoroughly.
- **Long voice-transcribed messages** establish context and values. The core directive is usually in the first and last paragraphs.
- **"What are your thoughts, what does this change and what do you propose?"** means give a structured analysis with options and recommendations, not just a fix.
- **Compute is not a constraint; rigor is.** His advisor has greenlighted significant AWS compute. Do not optimize for speed at the expense of thoroughness.
- **Latency tolerance is high.** Torrin explicitly said he would "far rather handle other business while you run for multiple hours and come back to a detailed and comprehensive job." Do not rush the diagnostics. Take the time to be thorough.
- **He thinks in terms of supervisor meetings.** When he asked to pause before launching the sweep, it was because he was imagining presenting the results to Jan Bena and wanted the interpretive framework ready. Frame your work the same way: what would a finance professor want to see?

### Codex for Infrastructure

Codex handles all AWS operations. You write deployment-ready scripts and handoff documents. Codex provisions, deploys, monitors, and writes results back. You do not SSH into AWS.

### The Diagnostic-Driven Approach

This project resolves questions empirically, not by argument. The pattern established across both prior sessions:

1. Make a decision with stated assumptions
2. Test assumptions on production data
3. If assumptions fail, revise with honest justification (don't hide the failure)
4. Test alternatives before committing
5. If no alternative helps, document the limitation
6. Measure downstream impact (BC rankings are what ultimately matter)

### Halt at Gates

Development halts at designated checkpoints until approved. This is not optional — every review cycle across Weeks 1 and 2 has caught real bugs. The implementation instance was explicitly told by Torrin (via Codex) "Do not resume implementation yet. Treat this as interpretation/design review work first."

### External Expertise Pattern

Torrin maintains other Claude Code instances with different domain contexts (e.g., STAT 405 for Bayesian statistics). When analysis tagged `<Codex>` arrives, treat it as authoritative. If you need to send a question to another instance, write a self-contained RFI document.

---

## The Diagnostic Sequence (Your First Priority)

Codex prescribed this exact four-step sequence. Execute it before any design revision.

### Step 1: Tail-Margin Analysis
**Question**: Are the top-50 BC values densely packed with near-ties?

Load `bc_matrix_all_k{N}.npz` for each K_max. Extract the upper-triangle BC values. Sort descending. Examine:
- BC values at ranks 1, 10, 25, 50, 100, 200
- Gaps between adjacent ranks in the top-50 (are they 0.001 or 0.1?)
- Distribution of BC values in the top-200 vs the broader tail
- Compare: if you use a BC threshold (e.g., "all pairs with BC > X") instead of fixed top-k, how does the overlap change?

**Interpretation**: If the top-50 are separated by gaps of 1e-4 while the underlying BC values shift by 1e-3 with K_max, most of the 0% overlap is ranking noise. If the gaps are 0.01+ and pairs still change, it's genuine structural sensitivity.

### Step 2: Robust-Core vs Volatile-Fringe
**Question**: Is there a stable nucleus of firm pairs that persist across all K_max values?

For each K_max, compute the top-200 pair set. Find:
- **Robust pairs**: appear in the top-200 at all 5 K_max values
- **Model-sensitive pairs**: appear at some but not all
- How many robust pairs exist? If it's 20 out of 200, the actionable output is small but trustworthy. If it's 150, the methodology is solid despite the top-50 metric.

**Interpretation**: A stable core with a volatile fringe is the best-case outcome for the not-converged path. The robust pairs are the Week 3 high-confidence candidates; the volatile pairs are reported with their K_max profile.

### Step 3: Link Instability to Firm Characteristics
**Question**: Are the volatile firms systematically different?

Cross-reference the firms driving tail churn with:
- Firm size (patent count)
- Effective K at different K_max values (are these the diversified conglomerates?)
- Earlier PPC failure status (if the same firms failed PPC and drive tail churn, that's strong evidence of model inadequacy)
- Whether instability is concentrated in a small number of firms or spread broadly

**Interpretation**: If 10 mega-firms drive all the churn → the model is fine for focused firms, inadequate for conglomerates. If churn is spread broadly → deeper structural issue.

### Step 4: Targeted Alternative Test (only if needed)
If Steps 1-3 leave the near-tie vs misspecification question unresolved, propose a small experiment on 20-50 unstable firms with a richer local density approximation. The goal is not to replace the production model — it is to determine whether unstable firms become stable under a richer model. **Do not run this without presenting the proposal to Torrin first.**

---

## Workflow with Halting Points

### Step 1: Pull artifacts and run diagnostics → HALT for review
Pull all S3 artifacts. Execute the four-step diagnostic sequence. Present findings to Torrin with clear interpretation: near-tie, misspecification, or both?

### Step 2: Populate pre-registered artifacts → HALT for review
Fill in `notebooks/03_kmax_convergence_analysis.ipynb` and `kmax_sweep_executive_summary.md` with actual data. All 11 visualizations rendered. Section 7 narrative written based on what the data actually says.

### Step 3: Reopen ADR-004 + define Week 3 contract → HALT for Codex review
Rewrite ADR-004 around the actual result. Update portfolio spec for multi-K_max output as first-class deliverable. Define the Week 3 reporting contract (robust vs model-sensitive pairs).

### Step 4: Implementation (TDD) → HALT for Codex review
Implement PortfolioBuilder and GMMFitter per the (revised) spec. Tests first, code second. Submit for Codex review before production run.

### Step 5: Production fitting + validation → HALT for approval
Run full-scale portfolio fitting. Create validation notebook. Write sprint retrospective.

---

## What NOT To Do

1. **Do not re-run the sweep.** The results are definitive. 0% overlap at 15→20 is not an artifact of insufficient K_max range — it persists through 30.
2. **Do not jump to model-family escalation before running the diagnostics.** The near-tie hypothesis must be tested first. It's cheaper and may explain most of the instability.
3. **Do not treat non-convergence as failure.** It is a substantive methodological finding. Frame it as rigor.
4. **Do not re-run EDA or Week 1.** Those are done.
5. **Do not change the priors without re-running prior predictive simulation.** The global EB priors are calibrated.
6. **Do not use BIC.** sklearn's `BayesianGaussianMixture` has no `.bic()` method. Use `lower_bound_` (ELBO).
7. **Do not compute global priors from grouped firm vectors.** Co-assigned patents would be duplicated.
8. **Do not implement before the design revision is approved.** The not-converged result changes the design contracts.
9. **Do not skip Codex review gates.** Every review cycle has caught real bugs.
10. **Do not rearrange the pre-registered notebook to fit the data.** Fill in the placeholders. Write interpretation. But the section structure and visualization plan are locked.
11. **Do not rush.** Torrin values thoroughness over speed. Take the time to understand what the data is telling us.

---

## Pre-Registered Decision Framework (Locked Before Sweep)

The sweep triggered **Branch B: Convergence does not emerge**.

**What Branch B requires:**
- Proceed with Week 2 implementation, but explicitly reframe:
  - Default K_max is operational, not final
  - All top-pair conclusions must include K_max robustness classification ("robust" vs "model-sensitive")
  - Week 3 methodology section acknowledges sensitivity as a condition
- This is still publishable and intellectually valuable — sensitivity to K_max is itself a finding

**What this means for implementation:**
- Multi-K_max output is a first-class deliverable (not just the default K_max)
- The PortfolioBuilder/GMMFitter modules treat K_max as a configurable parameter
- Production fitting generates artifacts at all sweep K_max values
- Week 3 BC computation runs on multiple K_max artifacts and classifies pairs

---

## Key Artifacts and Their State

| File | State | Action needed |
|------|-------|---------------|
| `output/kmax_sweep/` | On S3, NOT local | Pull from S3 |
| `notebooks/03_kmax_convergence_analysis.ipynb` | Skeleton with TODOs | Populate with real data |
| `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` | Skeleton with `[RESULT]` | Populate with actual numbers |
| `docs/adr/adr_004_k_selection_method.md` | Says "K_max=15 provisional" | Rewrite for not-converged result |
| `docs/specs/firm_portfolio_spec.md` | Single-K_max contract | Update for multi-K_max |
| `src/portfolio/` | Empty (.gitkeep only) | TDD implementation after design revision |
| `scripts/run_kmax_sweep.py` | Production-ready, tested | Reuse `load_gmm_results()`, BC functions in notebook |
| `tests/unit/test_kmax_sweep_fitters.py` | 6 tests, all passing | Reference for testing patterns |

---

## Deep-Dive Firms (Pre-Identified)

These were identified by patent-title pattern matching during the implementation session:

| gvkey | Firm | Patents | Expected behavior |
|---|---|---|---|
| 006066 | **IBM** | ~157K | Mega-conglomerate. K keeps growing — diverse across semiconductors, software, AI, quantum, cloud. |
| 012141 | **Intel** | ~47K | Focused semiconductor giant. K should plateau. |
| 024800 | **Qualcomm** | ~36K | Ultra-focused telecom IP (CDMA). K stabilizes early. |
| 160329 | **Google/Alphabet** | ~30K | Software/internet, mid-diversified. |
| 020779 | **Cisco Systems** | ~18K | Networking equipment specialist. K should stabilize. |

---

## Required Reading (In This Order)

1. `docs/epics/instance_handover/week2_implementation_instance_summary.md` — What the implementation instance did and why (including sweep results, what's confident/uncertain, and the diagnostic sequence)
2. `CLAUDE.md` — Project instructions, values, architecture
3. `docs/adr/adr_004_k_selection_method.md` — The ADR you need to rewrite
4. `notebooks/03_kmax_convergence_analysis.ipynb` — The skeleton you need to populate
5. `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` — The memo you need to populate
6. `scripts/run_kmax_sweep.py` — Contains `load_gmm_results()`, BC functions, convergence metrics (reuse in notebook)
7. `docs/specs/firm_portfolio_spec.md` — The interface contracts that need updating
8. `docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md` — The STAT 405 audit (relevant if pursuing misspecification hypothesis)

---

## Getting Started

1. Read all Required Reading documents (implementation instance summary is most critical)
2. Pull sweep artifacts from S3
3. Run the four-step diagnostic sequence
4. Present diagnostic findings to Torrin → HALT
5. Populate notebook and executive summary
6. Rewrite ADR-004, update spec → HALT for Codex review
7. TDD implementation of PortfolioBuilder/GMMFitter
8. Production fitting and validation

**Your immediate first action**: Read the implementation instance summary, then `CLAUDE.md`, then ADR-004. Only after understanding the full context should you pull data and begin diagnostics.
