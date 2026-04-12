# Week 2 Interpretation Instance Summary

**Instance role**: Interpret K_max sweep results, diagnose top-tail instability, fix root causes  
**Started**: 2026-04-09  
**Completed**: 2026-04-12  
**Final commit**: `170efc3` (Diagnose and fix K_max sweep top-tail instability)  
**Repository**: `github.com:Torr1n/firm-pair_merger_prediction.git` branch `master`

---

## Mission and Outcome

This instance was asked to interpret the K_max convergence sweep results (run `20260409T170706Z`), which had returned **NOT_CONVERGED** — Spearman ρ ≈ 0.99 but top-50 pair overlap collapsed from 80% to 0% at K_max=15→20.

**The original non-convergence was entirely an artifact of two bugs.** After fixing them, the corrected full-scale result (run `20260412T043407Z`, 7,485 deduplicated firms) is:

| Transition | Spearman ρ | Top-50 overlap |
|---|---|---|
| K10→K15 | 0.9912 | **98%** |
| K15→K20 | 0.9925 | **100%** |
| K20→K25 | 0.9917 | **98%** |
| K25→K30 | 0.9930 | **96%** |
| K10→K30 | 0.9833 | **96%** |

**VERDICT: CONVERGED at K_max=10.** Max BC = 0.997, zero pairs above 1.0. The persistent stability rule (ρ > 0.95 AND top-50 > 80% from K* onward) is satisfied at the smallest K_max tested.

This means **Branch A of the pre-registered decision framework** applies: lock K_max=10 as the production default, implement with a single primary specification, keep a neighbor (K_max=15) as a robustness check.

---

## What I Actually Did (Narrative Arc)

The session unfolded as a detective story. I was asked to distinguish "near-tie noise" from "model misspecification." I found neither — instead I found two compounding bugs that fully explained the instability. The key insight came not from running the prescribed diagnostics, but from noticing that BC values exceeded 1.0 (theoretically impossible), which led to the formula bug, which led to asking "what ARE these top-50 pairs?", which led to the duplicate firm discovery, which led to Torrin's question about subsidiaries, which led to the unified deduplication rule.

### Phase 1: Prescribed Diagnostics (Correct Process, Misleading Initial Results)

Ran the four-step diagnostic sequence prescribed by Codex in the interpretation bootstrap prompt:

1. **Tail-margin analysis**: shift/gap ratio = 13.2 (shifts 13x larger than inter-rank gaps). Looked like near-ties.
2. **Robust-core analysis**: 0 robust pairs at ANY top-k level. 79% of top-50 pairs appear at only 1 of 5 K_max values.
3. **Firm characteristics**: 464 unique firms in any top-200, all volatile. Hub firms with K=16-24 from 50-68 patents dominate at high K_max.
4. **BC distribution**: Max BC grows from 1.0007 to 5.39 across K_max.

The diagnostics were valuable not for the answers they gave to the original question, but for the anomalies they surfaced — particularly BC values exceeding 1.0.

### Phase 2: BC Formula Bug Discovery (Critical Finding #1)

The max BC of 5.39 is impossible for a Bhattacharyya Coefficient (which is bounded in [0,1]). Investigation revealed:

`bc_mixture` in `run_kmax_sweep.py:473` uses `√(πᵢπⱼ)` weighting. This is mathematically an **upper bound** on the true BC (proved via triangle inequality on the √(p·q) integrand). The bound is tight at K=1 but grows as K increases. For K equal-weight components: max BC = K.

Arthur's methodology.md says "aggregate using GMM weights" — no √ specified. The √ was an implementation artifact from conflating the √ in the BC *definition* (which applies to probability densities) with the mixture weight *aggregation*.

**Fix**: linear weights `πᵢπⱼ`. Bounded in [0,1], aligns with methodology.

**Tested alternatives**: Cosine normalization (BC_raw / √(self_A × self_B)) was tried and **FAILED** — it compressed all top-200 to BC=1.0000 exactly, making rankings worse. This failure led directly to Phase 3.

### Phase 3: Duplicate Firm Discovery (Critical Finding #2)

When cosine normalization failed, I asked: "Why do 300+ pairs have BC = 1.0 at every K_max?" Running `scripts/identify_top_pairs.py` revealed the answer: **ALL 50 top pairs at K_max=10 are pairs of firms sharing 100% of their patents.** They are the same company under anagram-style PRIV_ name variations:

- PRIV_PARADETECHNOLOGIES / PRIV_PRIDETECHNOLOGIES (136 patents, 100% shared)
- PRIV_VIATECHNOLOGIES / PRIV_VTECHNOLOGIES (2,016 patents, 100% shared)

At K_max=20+: ZERO shared patents in the top-50. The 0% overlap at K15→K20 is a **phase transition** between duplicate-dominated ranking (K_max≤15) and inflation-dominated ranking (K_max≥20).

### Phase 4: Subsidiary Discussion (Torrin's Critical Question)

I initially proposed deduplicating only the strict aliases (Jaccard ≥ 0.99). Torrin asked: "Would we ever 'predict' a corporation should acquire a subsidiary it already owns?"

This question expanded the deduplication scope fundamentally. Patent assignment in Compustat reflects **current ownership** — Alphabet's portfolio already includes Waymo's 1,190 patents. Including both as separate firms creates double-counting AND false-positive M&A predictions for already-completed deals.

A project-wide duplicate scan (`scripts/duplicate_firm_scan.py`) characterized the full Jaccard distribution:
- 991 overlapping pairs (0.003% of all possible)
- 341 strict aliases at Jaccard = 1.0
- 183 nested subsidiary relationships (containment ≥ 0.95, Jaccard < 0.80)
- Clean natural gap in the histogram at Jaccard ≈ 0.95-0.99

**Unified rule**: Drop firm B if there exists firm A with |A| ≥ |B| and containment(B→A) ≥ 0.95. This catches aliases, subsidiaries, AND predecessor records.

**Result**: 464 firms removed (264 aliases, 141 subsidiaries, 59 predecessors). 7,485 firms remain.

### Phase 5: Misspecification Discussion (Important but Unconfirmed)

Independent of the two bugs, effective K does not saturate by K_max=30 and small firms show extreme K values. Torrin and I discussed whether this indicates the Gaussian assumption is inadequate for UMAP-reduced data. My assessment: suggestive but not conclusive. Direct Gaussianity diagnostics (Mahalanobis Q-Q, Mardia's test) would settle it, but the convergence result materially weakens the misspecification urgency.

### Phase 6: Corrected Recomputation (Codex VM)

Wrote `scripts/recompute_bc_corrected.py` — loads saved GMM parameters, filters to deduplicated firms, recomputes BC with linear weights. Validated locally on a 1000-firm sample (top-50 = 100% at every transition).

**Codex ran the full-scale recomputation on c5.4xlarge (run 20260412T043407Z).** Result: CONVERGED at K_max=10 with top-50 overlap 96-100% across all transitions. Max BC = 0.997. Zero pairs above 1.0.

### Phase 7: PCA Comparison Script (Written, Deferred)

Wrote `scripts/run_pca_comparison_sweep.py` for testing whether UMAP introduces Gaussian-incompatible distortion. Per Codex's recommendation, this was **NOT** bundled into the corrected BC recomputation — the experiment-isolation principle says "fix the known bugs first, see the result, then test representation choice as a separate follow-up." The PCA script is validated and in the repo, ready to run if/when needed.

### Phase 8: Notebook Analysis (Completed for Original Data, Awaiting Corrected Data)

The pre-registered notebook was executed on the ORIGINAL (buggy) sweep data, generating 12 visualization PNGs. I did a deep independent analysis of all 12 visualizations and wrote detailed interpretive notes. The key observations:

- **Viz 2A (K progression)**: Mean effective K decelerates but doesn't plateau. Ceiling rate drops from 35% to 0.2%.
- **Viz 2C (K vs firm size)**: Spearman ρ = 0.336 (weak). Small firms with extreme K are outliers — some may be duplicates.
- **Viz 3B (BC scatter K15 vs K20)**: Axes clipped at [0,1], hiding the inflation (BC values up to 2.59 at K_max=20). Needs an unclipped version.
- **Viz 4A (Top-100 overlap heatmap)**: Block-diagonal structure shows three regimes (K10-K15, K20, K25-K30).
- **Viz 4B (Rank trajectories)**: Waterfall pattern — all top-200 pairs drop below rank 200 by K_max=20.
- **Viz 5A (Named firms)**: ALL five mega-firms hit ceiling at K_max=10 and K_max=15. Confirms K_max=10 is too restrictive for mega-firms.
- **Viz 5B (Weight evolution)**: Orderly weight subdivision at higher K_max for mega-firms.
- **Viz 6A (Convergence dashboard)**: Split personality — three metrics pass, one (top-50) catastrophically fails.

**These PNGs are from the BUGGY sweep data.** They need to be regenerated from the corrected BC matrices before the team email goes out. The notebook cell code should work as-is on the corrected data files — only the file paths need to change to point to the `_dedup_linear` versions.

---

## What Was NOT Completed

1. **Notebook NOT regenerated** with corrected BC matrices (12 PNGs need regeneration, 4 interpretation cells need writing, Section 7 narrative needs writing)
2. **Executive summary NOT populated** (`docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` still has `[RESULT]` placeholders)
3. **ADR-004 NOT updated** for the CONVERGED outcome (needs to switch from "provisional K_max=15" to "converged at K_max=10, production default = 10")
4. **firm_portfolio_spec NOT updated** (needs deduplication as a required input validation step; needs to reflect CONVERGED outcome under Branch A)
5. **Misspecification diagnostics NOT run** (Mahalanobis Q-Q, Mardia, prior sensitivity) — less urgent now but still worth doing as follow-up
6. **PCA comparison NOT run** — deferred per Codex recommendation
7. **Team email NOT sent** — waiting on notebook population
8. **TDD implementation of PortfolioBuilder/GMMFitter NOT started** — blocked on design revision
9. **Corrected artifacts NOT pulled locally** — the corrected BC matrices are on S3 at `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/` but have not been downloaded to the local working directory

---

## What I'm Confident In

1. **The two bugs fully explain the original instability.** The corrected result converges cleanly at full scale (7,485 firms, 28M+ pairs per K_max). The 1000-firm test was directionally correct and the full-scale result was even slightly better.
2. **The unified deduplication rule is correct.** All test cases (Alphabet/Waymo, Qualcomm/Snaptrack, IBM, J&J subs, Lyft, GeneralData) produce the right answer. The 0.95 containment threshold has a natural gap in the data supporting it.
3. **The linear-weight BC formula is correct.** It's bounded in [0,1], aligns with Arthur's methodology, and produces convergent results. Max BC = 0.997 on the full dataset.
4. **The bulk firm-similarity landscape is stable.** Spearman ρ ≈ 0.99 across all transitions, median NN-5 = 100%. This finding was consistent across the original buggy analysis AND the corrected analysis.
5. **The pre-registered analysis structure is sound.** The 7-section notebook, the decision framework with Branches A and B, the convergence thresholds — all designed before seeing data, all work as intended with the corrected results.

## What I'm NOT Confident In

1. **Whether the misspecification signal is genuine or a duplicate-artifact residual.** Effective K not saturating is concerning, but the corrected convergence result materially weakens this concern. The 9.3% reduction in the small-firm bin from deduplication may have cleaned up the worst offenders. Need Gaussianity diagnostics to settle.
2. **Whether the 0.95 containment threshold catches all true duplicates.** 34 pairs in [0.85, 0.95) containment range include some that look like real subsidiaries (Hughes Network Systems, Switchcraft, McAfee). Manual review of these is valuable but wasn't done.
3. **Whether the notebook PNGs will look clean after regeneration.** The corrected data should produce much more interpretable visualizations (no BC > 1, no duplicate-dominated top-k), but I haven't actually seen them yet. Edge cases might emerge.
4. **Whether K_max=10 is genuinely the right production default or just the smallest value we tested.** The persistent stability rule says K*=10 because all transitions from K_max=10 onward pass. But we didn't test K_max=5 or K_max=8. The mega-firm deep dives (Viz 5A) show ALL five firms hit the ceiling at K_max=10, which suggests K_max=10 is binding for large firms and a higher default (15 or 20) might be more appropriate. This needs discussion.
5. **Whether the linear-weight BC is economically the "right" metric.** It measures "expected overlap between random technology areas from each firm" — which naturally favors focused firms (two single-Gaussian firms in the same niche always dominate). For M&A prediction where acquirers are often large diversified firms, this may not rank the right pairs highest. This is a Week 3 question.

---

## Decisions Made (and Why)

| Decision | Rationale | Who decided |
|----------|-----------|-------------|
| Linear weights `πᵢπⱼ` for BC formula | Bounded [0,1], aligns with methodology.md, eliminates K-dependent inflation | Claude, validated by test |
| Containment ≥ 0.95 for dedup | Natural gap in Jaccard histogram; catches aliases + subs + predecessors | Torrin + Claude after subsidiary discussion |
| Drop subsidiary firms (not just aliases) | Torrin's question: "Would we predict a corp should acquire what it already owns?" Answer: yes, which is structurally bad | Torrin initiated, Claude formalized |
| Sequence UMAP fix before PCA comparison | Codex: "first fix the known bugs, then see the corrected result, then test representation choice as secondary." Experiment isolation > VM efficiency | Codex recommended, Torrin + Claude agreed |
| PCA script written but not run | Available as follow-up if corrected UMAP shows residual issues | Codex recommendation |
| K_max=10 as converged production default | Pre-registered persistent stability rule. All transitions from K_max=10 pass both thresholds | Pre-registered rule, data confirmed |

---

## Workflow Patterns That Worked Well

1. **Diagnostic-first approach**: Running the four-step diagnostic before any fixes was essential. The anomalies (BC > 1, hub firms, near-ties) built the evidence for the actual root causes.
2. **Asking "what ARE these pairs?"**: The identify_top_pairs.py analysis was the breakthrough. Abstract statistics (shift/gap ratios, overlap percentages) didn't reveal the duplicate-firm problem. Looking at the actual firm names and shared patents did.
3. **Torrin's subsidiary question**: This wasn't in the diagnostic plan. It came from Torrin thinking about the downstream M&A use case. The user's domain knowledge caught a structural issue that pure data analysis would have missed.
4. **Test-before-deploy**: The 1000-firm local test of the corrected approach gave confidence to hand off to Codex. The full-scale result confirmed the test's prediction.
5. **Codex's experiment-isolation principle**: Separating the UMAP fix from PCA comparison kept the causal narrative clean. "The bugs caused the instability" is a much stronger story than "the bugs plus/or UMAP caused it."
6. **Pre-registration**: The notebook structure and decision framework were designed before seeing any results. When the corrected results landed in Scenario A, the path forward was unambiguous.

## Patterns That Didn't Work Well

1. **I initially under-scoped the dedup**: I proposed Jaccard ≥ 0.99 (catching only strict aliases), which would have missed 183 subsidiary pairs. Torrin's question was the correction. Lesson: always think about the downstream use case, not just the data cleaning.
2. **I over-proposed PCA bundling**: I proposed running PCA in parallel with the UMAP fix on the same VM. Codex correctly identified this as experiment-conflation. Lesson: marginal compute cost is not the right frame for experimental design.
3. **Cosine normalization was a dead end**: I spent time implementing and testing it when the core issue was duplicate firms, not metric scaling. Lesson: when all top-200 values are at the theoretical maximum, the problem is in the DATA, not the metric.

---

## Memory Files Saved

| File | Type | Content |
|------|------|---------|
| `user_torrin.md` | user | Role, background, communication style, preferences |
| `project_kmax_sweep_result.md` | project | Original NOT_CONVERGED result (now superseded by corrected result) |
| `project_duplicate_firms_bc_formula.md` | project | The two bug discoveries |
| `project_dedup_rule.md` | project | Unified deduplication rule details |
| `project_corrected_bc_converges.md` | project | Corrected result CONVERGED at K_max=10 |
| `feedback_thoroughness.md` | feedback | Preference for depth over speed |
| `feedback_isolate_experiments.md` | feedback | Sequence over parallelize, experiment isolation |
| `reference_s3_artifacts.md` | reference | S3 paths for all data |

---

## Critical Files for Downstream Instance

### Must-read (in order)
1. `docs/epics/week2_firm_portfolios/kmax_diagnostic_findings.md` — **PRIMARY FINDINGS DOC**, fully updated with all discoveries
2. `docs/epics/week2_firm_portfolios/codex_bc_recomputation_handoff.md` — contains the post-run decision tree (Scenario A applies)
3. `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` — skeleton with `[RESULT]` placeholders to populate

### S3 locations
- **Corrected artifacts (use these)**: `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/`
  - `convergence_summary_dedup_linear.json`
  - `bc_matrix_all_k{10,15,20,25,30}_dedup_linear.npz`
  - `bc_block_sg_vs_sg_dedup_linear.npz`
  - `recompute.log`
- **Original (buggy) artifacts (for reference only)**: `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/`
- **Dedup decisions**: `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/deduplication_decisions.parquet`
- **1536D pre-UMAP vectors (for PCA follow-up)**: `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/concatenated_1536d.parquet`

### Scripts
- `scripts/recompute_bc_corrected.py` — the script Codex ran successfully
- `scripts/run_pca_comparison_sweep.py` — written, validated on sample, deferred
- `scripts/run_kmax_sweep.py` — original sweep script (still has the sqrt-weight bug at line 473; the fix is in recompute_bc_corrected.py, not patched in place)

### Notebook
- `notebooks/03_kmax_convergence_analysis.ipynb` — code cells executed on BUGGY data; needs re-execution on corrected data
- `notebooks/03_viz*.png` — 12 PNGs from BUGGY data; will need regeneration
- 4 interpretation cells (14, 20, 28, 36) need writing
- Section 7 narrative needs writing

---

## The Downstream Instance's Mission

**Scenario A confirmed.** The next instance should:

1. **Pull corrected artifacts from S3** (the 5 corrected BC matrices, convergence summary, recompute log)
2. **Regenerate all 12 notebook visualizations** from the corrected BC matrices
3. **Write the 4 interpretation cells** in the notebook (cells 14, 20, 28, 36)
4. **Write Section 7 narrative** (Implications for M&A Prediction)
5. **Populate the executive summary** with actual numbers from the corrected convergence summary
6. **Update ADR-004** for the converged outcome (K_max=10 production default)
7. **Update firm_portfolio_spec** to add deduplication as a required input validation step and reflect Branch A
8. **Knit notebook to PDF** for team email
9. **Commit** all updates
10. **Optional follow-ups**: misspecification diagnostics (Mahalanobis Q-Q, Mardia), PCA comparison sweep
11. **Then proceed to TDD implementation** of PortfolioBuilder/GMMFitter per the (updated) spec
