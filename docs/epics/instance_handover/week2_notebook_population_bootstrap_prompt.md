# Week 2 Notebook Population + Design Update: Bootstrap Prompt

**Purpose**: Initialize a fresh Claude Code instance to populate the K_max convergence analysis with corrected results, update all design artifacts for the converged outcome, prepare the team-facing deliverables, and then proceed to the TDD implementation of PortfolioBuilder/GMMFitter.

---

## Mission Briefing

You are the **Week 2 notebook-population and implementation instance** for the Firm-Pair Merger Prediction project. Three prior instances completed (1) the design phase (ADRs, spec, Bayesian audit), (2) the implementation phase (sweep script, pre-registration, AWS deployment, float32 bug fix), and (3) the interpretation phase (diagnostic sequence, bug discovery, deduplication, corrected BC recomputation). The interpretation instance left everything ready for you.

**The K_max convergence question is RESOLVED.** The original sweep returned NOT_CONVERGED, but the interpretation instance discovered this was caused by two compounding bugs — duplicate firms in v3 data and an unbounded BC formula. After fixing both, the corrected full-scale result is:

**CONVERGED at K_max=10.** Top-50 overlap 96-100% across all transitions. Spearman ρ ≈ 0.99. Max BC = 0.997 (properly bounded).

This triggers **Branch A of the pre-registered decision framework**: lock K_max=10 as the production default. A single primary specification. A neighbor (K_max=15) as a robustness check.

**You are not diagnosing anything. You are populating artifacts with a definitive result, updating the design for the converged outcome, and then building the production system.**

---

## The Story Arc (Why This Matters — Read Carefully)

This project asks: *Can firms' patent portfolios predict M&A pairs in the technology sector?* Building on Bena & Li (2014), we replace a static similarity score with probabilistic overlap of entire patent portfolio distributions.

**Step 1 (Week 1 — COMPLETE)**: Vectorize 1.45M patents → 50D UMAP vectors per patent.

**Step 2 (Week 2 — YOUR SPRINT)**: Aggregate each firm's vectors into a Gaussian Mixture Model → a probability distribution capturing how many technology areas the firm operates in and how its portfolio is distributed across them.

**Step 3 (Week 3 — NEXT)**: Compute Bhattacharyya Coefficient between every firm pair → rank pairs by technological similarity → identify M&A candidates.

The K_max convergence sweep was designed to determine whether the Step 2 portfolio representation is a **stable scientific object** — whether the downstream firm-similarity rankings depend on the arbitrary choice of how many mixture components we allow. The answer, after fixing two bugs, is **yes, it is stable.** The representation converges at the smallest K_max we tested (K_max=10), and rankings are robust through K_max=30.

**The supervisor pitch**: "We ran a pre-registered convergence study across five K_max values. During diagnostics, we discovered and fixed two data/metric issues — duplicate firms in the patent database and an unbounded similarity formula. After correction, the firm-similarity rankings converge immediately and remain stable through K_max=30. The portfolio representation is a stable scientific object."

That is a story of rigor, diagnostic discipline, and honest methodology. It is exactly what Jan Bena (the faculty advisor) values.

---

## What the Interpretation Instance Discovered (The Bugs and Their Resolution)

### Bug 1: Duplicate Firms in v3 Data

The Compustat patent database contains firms that appear under multiple names with overlapping or identical patent sets:

**Aliases** (341 pairs at Jaccard = 1.0): Same entity under anagram-style PRIV_ names. Example: PRIV_PARADETECHNOLOGIES / PRIV_PRIDETECHNOLOGIES — 136 patents each, 100% shared.

**Subsidiaries** (183 pairs, containment ≥ 0.95): Already-acquired subsidiaries appearing as separate firms. Example: Alphabet/Waymo (1,190 of Waymo's 1,190 patents are in Alphabet's 29,813). Without dedup, the model "predicts" Alphabet should acquire Waymo — an event that happened in 2016.

**Predecessors** (59 pairs): Pre-/post-IPO records. Example: 034873 (Lyft public, 274 patents) fully nested in PRIV_LYFT (pre-IPO, 296 patents).

**Unified dedup rule**: Drop firm B if there exists firm A with |A| ≥ |B| and containment(B→A) ≥ 0.95.

**Result**: 464 firms removed (5.8% of 7,949), leaving 7,485 distinct legal entities. Breakdown: 264 aliases, 141 subsidiaries, 59 predecessors.

**Decision log**: `output/kmax_sweep/deduplication_decisions.csv` (also on S3).

### Bug 2: Unbounded BC Formula

`bc_mixture` in `scripts/run_kmax_sweep.py:473` used `√(πᵢ·πⱼ)` weighting for mixture-level BC aggregation. This is mathematically an upper bound on the true Bhattacharyya Coefficient — it equals the true BC when K=1, but inflates with K. At K_max=30, max "BC" was 5.39 (theoretical max is 1.0).

**Fix**: Linear weights `πᵢπⱼ`. Bounded in [0,1], aligns with Arthur's methodology.md ("aggregate using GMM weights"), eliminates K-dependent inflation.

**Important**: The fix is in `scripts/recompute_bc_corrected.py`, NOT in `run_kmax_sweep.py` (which still has the old formula). The corrected BC matrices on S3 were computed with the fixed formula.

### The Phase Transition Explained

At K_max=10-15: the top-50 was dominated by duplicate firms (all with BC ≈ 1.0, all tied). At K_max=20+: the √-weighted formula inflated values for multi-component firms past 1.0, displacing the duplicates entirely. The 0% overlap at K15→K20 was the crossover between these two failure modes. Neither represented genuine technological similarity. After fixing both, the top-50 is populated by real, non-duplicate firm pairs with properly-bounded BC values.

---

## The Corrected Result (Your Starting Point)

Run `20260412T043407Z-dedup-linear` on c5.4xlarge, Codex-deployed:

```
convergence_verdict: converged
converged_at_kmax: 10

K10→K15: ρ=0.9912, top-50=98%
K15→K20: ρ=0.9925, top-50=100%
K20→K25: ρ=0.9917, top-50=98%
K25→K30: ρ=0.9930, top-50=96%
K10→K30: ρ=0.9833, top-50=96%

Max BC: 0.997197
Pairs above 1.0: 0
```

**Artifacts on S3** (pull these first):
```bash
mkdir -p output/kmax_sweep/corrected
aws s3 sync s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/ \
    output/kmax_sweep/corrected/ --profile torrin
```

This gives you:
- `convergence_summary_dedup_linear.json` — the authoritative convergence metrics
- `bc_matrix_all_k{10,15,20,25,30}_dedup_linear.npz` — 5 corrected BC matrices
- `bc_block_sg_vs_sg_dedup_linear.npz` — K_max-invariant SG block
- `recompute.log` — full execution log

**Also still available locally** (from the original buggy sweep):
- `output/kmax_sweep/firm_gmm_parameters_k{10,15,20,25,30}.parquet` — GMM parameters (these are CORRECT — the bug was in BC aggregation, not GMM fitting)
- `output/kmax_sweep/deduplication_decisions.csv` — which firms to drop
- `output/kmax_sweep/diagnostic_results.json` — original diagnostic findings
- `output/week2_inputs/patent_vectors_50d.parquet` — Week 1 UMAP vectors
- `output/week2_inputs/gvkey_map.parquet` — patent-to-firm mapping

---

## Your Deliverables (In Priority Order)

### 1. Populate the Notebook: `notebooks/03_kmax_convergence_analysis.ipynb`

The notebook has a pre-registered 7-section structure with 43 cells (26 markdown, 17 code). Code cells were executed on the BUGGY data and produced 12 PNGs. You need to:

**a) Re-execute all code cells** against the corrected BC matrices. The code should work with minimal path changes — update file references from `bc_matrix_all_k{N}.npz` to the corrected `bc_matrix_all_k{N}_dedup_linear.npz` files. GMM parameter files remain the same but must be filtered to exclude the 464 deduplicated firms.

**b) Write 4 interpretation cells** (cells 14, 20, 28, 36). Each has a `[INTERPRETATION — populate after results]` placeholder. These should:
- Describe what the corrected visualizations show
- Connect the findings to the broader methodology
- Note what changed between the buggy and corrected results where relevant

**c) Write Section 7 narrative** ("Implications for M&A Prediction"). Three findings to communicate:
1. The firm-similarity landscape is fundamentally stable (ρ ≈ 0.99, K_max=10 is sufficient)
2. The sweep exposed and resolved a data quality issue (duplicate firms) that would have corrupted Week 3 predictions
3. The diagnostic process itself (pre-registration → sweep → diagnostics → correction → re-verification) demonstrates methodological rigor

**d) Consider adding** (NOT pre-registered, but valuable):
- An additional subsection acknowledging the two bugs and showing the corrected results alongside the original (a before/after comparison)
- An unclipped BC scatter plot that shows BC values > 1.0 from the original buggy sweep (makes the formula bug visually obvious)
- A note about the 12 existing PNGs being from the buggy sweep (for the archival record)

**CRITICAL**: The pre-registered section structure must NOT be rearranged. You fill in placeholders and write interpretation, you do NOT reorganize the story to fit the data. This is the pre-registration discipline.

### 2. Populate the Executive Summary: `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md`

Fill in all `[RESULT]` placeholders with actual numbers from the corrected convergence summary. Apply **Branch A** of the pre-registered decision framework:
- Lock K_max=10 as the production default in `config.yaml`
- Implement PortfolioBuilder/GMMFitter with this default
- Week 3: single primary BC specification
- Robustness check at K_max=15

### 3. Update ADR-004: `docs/adr/adr_004_k_selection_method.md`

The current ADR says "K_max = 15 (provisional)" and has extensive language about sensitivity and the not-converged path. Update for the converged outcome:
- K_max=10 is the production default (converged, persistent stability satisfied)
- Document the two bugs that caused the original non-convergence
- Keep the sensitivity analysis framework (it's sound methodology) but note that the corrected result resolves the sensitivity concern
- Note: whether to set K_max=10 or K_max=15 is a judgment call. The convergence rule says K*=10 (smallest where all subsequent pass). But ALL five mega-firms hit the ceiling at K_max=10 (Viz 5A), suggesting K_max=10 IS binding for the largest firms. K_max=15 might be more appropriate as a production default to give large firms room. Present this as a decision point for Torrin.

### 4. Update firm_portfolio_spec: `docs/specs/firm_portfolio_spec.md`

Add a deduplication step to the PortfolioBuilder's input validation:
- After `load_inputs()` and before `group_by_firm()`, apply the unified containment ≥ 0.95 rule
- Reference `output/kmax_sweep/deduplication_decisions.csv` as the source of truth
- Update the output file inventory for Branch A (single primary K_max, one neighbor artifact)

### 5. Knit to PDF and Prepare Team Email

The notebook should render cleanly in JupyterLab and convert to PDF. Torrin will use this as the primary deliverable for the team email to Arthur, Amie, Duncan, and Jan.

### 6. TDD Implementation of PortfolioBuilder/GMMFitter

After the above deliverables are approved, proceed to implementing:
- `src/portfolio/portfolio_builder.py` — per the spec (with the new dedup step)
- `src/portfolio/gmm_fitter.py` — per the spec
- Tests first, code second (TDD)
- **HALT for Codex review** before production fitting

### 7. Optional Follow-ups (Lower Priority)

- **Misspecification diagnostics**: Mahalanobis Q-Q vs χ²(50) on a stratified sample of ~50 firms, Mardia's multivariate skewness/kurtosis test. The convergence result weakens the misspecification urgency, but these are cheap to run and would strengthen the methodology section.
- **PCA comparison**: `scripts/run_pca_comparison_sweep.py` is written and validated. Requires the 8.6 GB 1536D vectors from S3. Only needed if misspecification tests show non-Gaussianity.

---

## Guiding Values and Standards (Non-Negotiable)

These were established across all prior instances and are enforced by Codex review. Violating them results in rejection and rework.

### The Code Quality Standard

> "The best engineers write code my mom could read. They choose boring technology, they over-document the 'why,' and under-engineer the 'how.' Complexity is not a flex; it becomes a liability."

- **Simplicity over cleverness**: scikit-learn's `BayesianGaussianMixture`. No custom inference engines.
- **Boring technology**: pandas, numpy, parquet. Not exotic formats.
- **Over-document the "why"**: Every hyperparameter choice has an empirical or theoretical justification in the ADRs.
- **Under-engineer the "how"**: Don't build a GMM framework. Build functions that fit GMMs to firms.

### Pre-Registration Discipline

The notebook structure and convergence thresholds were committed BEFORE the sweep launched. When you populate the notebook:
- The section structure stays fixed
- The visualization plan stays fixed
- You fill in numbers and write interpretation
- You do NOT rearrange the story to fit the data
- You CAN add supplementary subsections acknowledging the bug discovery — this is honest science, not post-hoc rationalization

### Evidence-Based Engineering

"The GMM fits well" is not evidence. Convergence metrics, PPC scores, BC ranking stability — these are evidence. Every claim about model quality must be backed by a computed artifact. The diagnostic scripts + output JSON files are the evidence base.

### Experiment Isolation

Per Codex's recommendation (saved as a feedback memory): when planning experiments, default to sequencing over parallelizing. Fix known bugs first, see the result, then test secondary hypotheses. Marginal compute cost is not the right frame for experimental design — scientific clarity is.

### Parsimony

The right amount of complexity is what the task actually requires. Three similar lines is better than a premature abstraction. Do not design for Week 4 when you're building Week 2.

---

## How Torrin Communicates

- **Short messages are precise directives.** "Does any of this suggest we should revisit UMAP?" means he sees a connection — investigate thoroughly.
- **Long voice-transcribed messages** establish context and values. Core directive in first and last paragraphs.
- **"What are your thoughts?"** means give a structured analysis with options and recommendations.
- **Compute is not a constraint; rigor is.** His advisor has greenlighted significant AWS compute.
- **Latency tolerance is abnormally high.** He explicitly prefers waiting hours for thorough work. Never rush.
- **He thinks in terms of supervisor meetings.** Frame work as what Jan Bena would want to see.
- **He values the "why" behind the "what."** The interpretation instance's biggest contribution wasn't the code — it was the narrative that connected "BC > 1.0 is impossible" → "the formula is unbounded" → "the top pairs are all duplicates" → "subsidiaries would create false-positive M&A predictions."
- **He will relay Codex messages as `<Codex>` tagged blocks.** Treat these as authoritative review feedback.

---

## The Three-Party Dynamic

| Role | Who | Interaction |
|------|-----|-------------|
| **Development Lead** | Torrin Pataki | UBC student, formal Bayesian stats training (STAT 405). Will catch methodology gaps. Initiates pivots that produce the project's best work. |
| **Implementer + Interpreter** | You (Claude Code) | Write code, analyze data, present findings. HALT at designated gates. |
| **Impartial Reviewer + Strategic Advisor** | Codex (relayed via Torrin as `<Codex>` messages) | Reviews for spec conformance, test coverage, value adherence, correctness. Also provides strategic framing (e.g., the experiment-isolation principle). Can reject work. |

**Codex review gates** (development HALTS until approved):
- After ADR updates (before implementation begins)
- After spec updates (before implementation begins)
- After each sprint's implementation is complete
- Before any cloud deployment

---

## Workflow Patterns That Work

1. **Ask "what ARE these?" before accepting statistics.** The interpretation instance's breakthrough came from looking at actual firm names and patent counts, not from abstract metrics. When you see anomalies in the corrected data, investigate the specific firms involved.
2. **Test locally before handing to Codex.** The 1000-firm validation test saved a potentially wasted VM session. Always validate on a sample first.
3. **Torrin's domain questions catch structural issues.** His "would we predict Alphabet should acquire Waymo?" question expanded the dedup scope from aliases to subsidiaries. When he asks a question, it usually reveals something the data analysis missed.
4. **Memory files matter.** Check `/home/torrin/.claude/projects/-mnt-c-Users-TPata-firm-pair-merger-prediction/memory/MEMORY.md` for saved context about the user, project state, and feedback.
5. **Documentation before code.** The interpretation instance wrote the diagnostic findings doc, instance handover, and Codex handoff BEFORE committing. This ensures nothing is lost even if the next instance has no context from this conversation.

---

## What NOT To Do

1. **Do not re-run the sweep or the corrected BC recomputation.** These are done. Use the artifacts from S3.
2. **Do not modify `run_kmax_sweep.py` to fix the BC formula in place.** The fix lives in `recompute_bc_corrected.py`. The original script is preserved for auditing (the commit history shows what changed and why).
3. **Do not change the pre-registered notebook structure.** Add subsections for the bug discovery if needed, but do not rearrange existing sections.
4. **Do not re-run Gaussianity diagnostics before populating the notebook.** The notebook goes out first (for the team email). Misspecification tests are a follow-up.
5. **Do not skip Codex review gates.** Every review cycle across all prior instances has caught real bugs.
6. **Do not treat K_max=10 as obviously correct.** It satisfies the persistent stability rule, but all five mega-firms are ceiling-bound at K_max=10. Present K_max=10 vs K_max=15 as a decision point for Torrin.
7. **Do not use the original (buggy) BC matrices or convergence summary.** Always use the `_dedup_linear` versions from `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/`.
8. **Do not rush.** Torrin values thoroughness over speed. The team email will be more impactful if it's comprehensive and well-framed than if it's sent an hour earlier.

---

## Required Reading (In This Order)

1. `CLAUDE.md` — Project-level instructions, architecture, design principles
2. `docs/epics/instance_handover/week2_interpretation_instance_summary.md` — **START HERE**: full narrative of what the interpretation instance did and why
3. `docs/epics/week2_firm_portfolios/kmax_diagnostic_findings.md` — Primary findings document with all evidence tables
4. `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` — The skeleton you need to populate
5. `notebooks/03_kmax_convergence_analysis.ipynb` — The notebook you need to populate
6. `docs/adr/adr_004_k_selection_method.md` — The ADR you need to update
7. `docs/specs/firm_portfolio_spec.md` — The spec you need to update
8. `docs/epics/week2_firm_portfolios/codex_bc_recomputation_handoff.md` — Contains the post-run decision tree (Scenario A applies)

---

## Getting Started

1. Read all Required Reading documents (instance summary is the most critical — it has the full narrative)
2. Pull corrected artifacts from S3 (the exact command is above)
3. Verify the corrected convergence summary matches Codex's report
4. Re-execute notebook code cells against corrected data
5. Write interpretation cells + Section 7 narrative
6. Populate executive summary
7. Update ADR-004 + spec → HALT for Codex review
8. Knit to PDF → team email
9. TDD implementation of PortfolioBuilder/GMMFitter → HALT for Codex review

**Your immediate first action**: Read the instance summary, then CLAUDE.md, then pull the corrected artifacts. Only after understanding the full context should you begin writing.
