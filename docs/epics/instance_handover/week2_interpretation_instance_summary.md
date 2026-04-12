# Week 2 Interpretation Instance Summary

**Instance role**: Interpret K_max sweep results, diagnose top-tail instability, propose fixes  
**Started**: 2026-04-09  
**Last update**: 2026-04-11  
**Status**: HALT after deduplication scan complete; awaiting Codex BC recomputation + misspecification tests

---

## Mission as inherited

The K_max convergence sweep (run 20260409T170706Z) returned **NOT_CONVERGED**:
- Bulk Spearman ρ ≈ 0.99 across all transitions (passes threshold)
- Top-50 overlap collapsed from 80% (K10→K15) to 0% (K15→K20) and stayed near 0% through K30 (fails threshold)

The previous instance had pre-registered an analysis notebook + executive summary with `[RESULT]` placeholders, run the sweep on AWS, and saved artifacts to S3 at `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260409T170706Z/`.

I was asked to:
1. Pull the sweep artifacts
2. Run a four-step diagnostic sequence (tail-margin, robust-core, firm characteristics, optional alt-model)
3. Determine whether the instability was near-tie noise, model misspecification, or both
4. Populate the pre-registered analysis artifacts
5. Reopen ADR-004 for the not-converged case
6. Then proceed to TDD implementation of PortfolioBuilder/GMMFitter

---

## What I actually found (in narrative order)

### Step 1 — Diagnostic sequence revealed something different than expected

The four-step diagnostic (results in `output/kmax_sweep/diagnostic_results.json`) showed:

- **Diagnostic 1 (Tail margins)**: Cross-K_max BC shifts are 13.2x larger than gaps between adjacent top-50 ranks. Looks like near-ties → BUT see Step 2.
- **Diagnostic 2 (Robust core)**: 0 robust pairs at any top-k level. ALL top-200 pairs are populated by different sets at each K_max.
- **Diagnostic 3 (Firm characteristics)**: 464 unique firms appear in any top-200; volatile pairs are 252 single-Gaussian + 212 GMM. Hub firms (small firms with K=16-24) dominate at high K_max.
- **Diagnostic 4 (BC distribution)**: Mean BC grows from 0.000364 to 0.001858. Max BC grows from 1.0007 to 5.39 — **theoretically impossible since BC ∈ [0,1]**.

### Step 2 — The BC formula is unbounded

`bc_mixture` in `scripts/run_kmax_sweep.py:473` uses `√(πᵢ·πⱼ)` weighting for mixture-level BC:

```python
weight_grid = np.sqrt(gmm_a["weights"][:, None] * gmm_b["weights"][None, :])
return float(np.sum(weight_grid * bc_grid))
```

This is mathematically an **upper bound** on the true BC (proof via triangle inequality on the integrand `√(p·q)`). The bound is tight at K=1 and grows progressively looser as K increases. For K equal-weight components: max value = K.

Arthur's methodology.md says "aggregate using GMM weights" — it does NOT specify √ weights. The √ was an implementation artifact from conflating the √ in the BC definition (which applies to densities) with the mixture aggregation.

**Fix**: change to linear weights `πᵢπⱼ`. This is bounded in [0,1] and aligns with the methodology.

### Step 3 — Cosine normalization was tested and FAILED

I tried normalizing by self-similarity (`scripts/verify_bc_normalization.py`):
- BC_norm(A,B) = BC_raw(A,B) / √(BC_raw(A,A) × BC_raw(B,B))

Result: WORSE than raw. The normalized top-200 all collapse to BC = 1.0000 exactly (span = 0.0), making top-k selection completely random. Convergence metrics dropped:

| Transition | Raw top-50 | Normalized top-50 |
|---|---|---|
| 10→15 | 80% | 24% |
| 15→20 | 0% | 6% |
| 20→25 | 0% | 18% |
| 25→30 | 6% | 8% |

This led me to discover that the underlying issue at low K_max wasn't metric-related at all — see Step 4.

### Step 4 — Top-50 pairs at K_max=10-15 are duplicate firms

Running `scripts/identify_top_pairs.py` revealed that ALL 50 top pairs at K_max=10 share **100% of their patents** (8,098 total shared). They're anagram-style PRIV_ name variants:
- PRIV_PARADETECHNOLOGIES / PRIV_PRIDETECHNOLOGIES (136 patents each)
- PRIV_VIATECHNOLOGIES / PRIV_VTECHNOLOGIES (2,016 each)
- PRIV_AASKITECHNOLOGY / PRIV_OSKITECHNOLOGY (579 each)
- ... and so on

These are the same legal entity appearing under multiple names in v3 data. They have identical GMM parameters and therefore BC ≈ 1.0 with massive ties.

At K_max=20+, the top-50 transitions to a completely different population: small firms (50-68 patents) with very high K (16-24) whose components inflate the √-weighted BC. ZERO shared patents at K_max=20+. The 0% top-50 overlap at K15→K20 is a **phase transition** between these two failure regimes.

### Step 5 — Subsidiary discussion (Torrin's question)

After I proposed deduplicating only the strict aliases, Torrin asked whether subsidiary relationships in the data would create false-positive M&A predictions ("predicting" Alphabet should acquire Waymo when Alphabet already owns Waymo).

Yes — and this is structurally bad, not just noisy. There are 183+ parent/subsidiary pairs in the data:
- Alphabet/Waymo, Alphabet/Verily, Alphabet/PRIV_GOOGLE
- Qualcomm/Snaptrack/Atheros/Pixtronix/etc. (8 subs)
- J&J/Ethicon/Janssen/Johnson&JohnsonConsumer/BioSenseWebster (6 subs)
- IBM/PRIV_INTERNATIONALBUSINESS
- 007585/Motorola Mobility

Each generates a high-confidence M&A "prediction" for an event that already occurred. This would crater precision@K when validating against 2021 M&A labels. Conceptually: patent assignment in Compustat reflects **current ownership**, so the parent's portfolio already includes the subsidiary's patents. Including both creates double-counting AND false positives.

### Step 6 — Project-wide duplicate scan

Ran `scripts/duplicate_firm_scan.py` to characterize the full Jaccard distribution across all firm pairs that share at least one patent.

**Findings:**
- 991 firm pairs share at least one patent (only 0.003% of 31.6M possible)
- 341 pairs at Jaccard = 1.0 (strict aliases — 196 cliques, 450 firms)
- Clean break in the histogram: 9 pairs in [0.95, 0.99), ZERO in [0.99, 1.0), spike at 1.0
- 183 nested pairs (containment ≥ 0.95, Jaccard < 0.80) — subsidiaries
- 124 moderate-overlap pairs (0.50 ≤ J < 0.95) — corporate reorganizations / spinoffs

### Step 7 — Unified deduplication rule

Ran `scripts/duplicate_firm_unified_rule.py` with:
**Rule**: Drop firm B if there exists firm A with `|A| ≥ |B|` and `containment(B → A) ≥ 0.95`.

**Result**:
- 568 pairs trigger the rule
- 464 firms removed
- 7,485 firms remain (down from 7,949 — 5.8% reduction)
- Breakdown: 264 aliases, 141 subsidiaries, 59 predecessors

All test cases (Alphabet/Waymo, Qualcomm/Snaptrack, IBM, J&J subs) produced correct results. Notable edge case: Lyft and GeneralData pairs have the PRIV_ record with MORE patents than the public record, so the rule keeps PRIV_ — this is correct (more complete record).

### Step 8 — Misspecification (suggestive but unconfirmed)

Independent of the data/formula bugs, the visualizations show a separate concerning pattern:

1. **Effective K does not saturate** by K_max=30 (mean 8.0 → 13.7, decelerating but not plateauing)
2. **Small firms with extreme K** (Viz 2C): firms with 50-100 patents getting K=15-25, vastly exceeding Bayesian prior expectations of E[K] ≈ log(n) ≈ 4-5
3. **Bimodal distribution emerging** at higher K_max (Viz 2B)
4. **All 5 mega-firms hit the K_max ceiling** at K_max=10 AND K_max=15 (Viz 5A)

These patterns are consistent with the Gaussian assumption being inadequate for UMAP-reduced data — UMAP uses a Student-t kernel and produces non-convex clusters. The Bayesian audit (`docs/epics/week2_firm_portfolios/bayesian_gmm_audit.md`) flagged this risk explicitly and recommended Gaussianity diagnostics that we never ran.

**This is unconfirmed**: it could be partly explained by duplicates (which inflate the small-firm K bin). After deduplication, if the K explosion persists, that's stronger evidence of misspecification.

---

## What's been decided

1. **Apply the unified deduplication rule** (containment ≥ 0.95) — 464 firms dropped, 7,485 remain
2. **Fix the BC formula** to use linear weights `πᵢπⱼ` instead of `√(πᵢπⱼ)`
3. **Recompute BC matrices** from saved GMM parameters (no re-fitting needed)
4. **Run misspecification diagnostics** on the deduplicated, corrected dataset to settle the Gaussian-assumption question definitively
5. **Then populate the notebook** and knit to PDF for the team email

---

## What's NOT yet decided / open questions

1. **Borderline cases** (16 pairs at containment 0.95-0.97, 34 pairs at 0.85-0.95): the current 0.95 threshold leaves ~34 likely-but-not-definite subsidiary cases unmerged. Manual review is the ideal but optional.
2. **BC formula choice**: linear weights is the cleanest, but Monte Carlo true BC is also an option (more expensive but theoretically exact). Decision is to start with linear weights.
3. **If misspecification tests confirm non-Gaussianity**, what's the response? Options:
   - (a) Switch to t-mixtures (heavier tails, easy swap)
   - (b) Re-tune UMAP parameters (n_neighbors, min_dist, metric)
   - (c) Use PCA instead of UMAP for the dimensionality reduction
   - (d) Accept misspecification and proceed with the caveat in Week 3 reporting
4. **The 9 near-aliases at 0.95-0.99 Jaccard**: should we manually merge these too? Probably yes, but they're a small population.

---

## Critical files and their roles

### Documentation
- `docs/epics/week2_firm_portfolios/kmax_diagnostic_findings.md` — **PRIMARY FINDINGS DOCUMENT**, fully updated
- `docs/epics/instance_handover/week2_implementation_instance_summary.md` — previous instance's handover
- `docs/epics/instance_handover/week2_interpretation_instance_summary.md` — this document

### Scripts (all under `scripts/`)
- `run_kmax_sweep.py` — original sweep script. **Bug at line 473** in `bc_mixture` (sqrt weights)
- `diagnostic_kmax_stability.py` — four-step diagnostic
- `verify_bc_normalization.py` — cosine normalization test (rejected)
- `verify_linear_weights.py` — linear weight comparison
- `identify_top_pairs.py` — top-pair identity analysis (revealed duplicates)
- `duplicate_firm_scan.py` — project-wide Jaccard scan
- `duplicate_firm_unified_rule.py` — containment-based dedup rule

### Sweep outputs (under `output/kmax_sweep/`)
- `bc_matrix_all_k{10,15,20,25,30}.npz` — original BC matrices (BUGGY formula, will need recomputation)
- `firm_gmm_parameters_k{10,15,20,25,30}.parquet` — GMM parameters (still valid)
- `bc_block_sg_vs_sg.npz` — K_max-invariant SG block (will also need recomputation with linear weights)
- `convergence_summary.json` — original convergence metrics
- `diagnostic_results.json` — diagnostic findings
- `duplicate_pairs.parquet` — 991 overlapping firm pairs
- `duplicate_scan.json` — duplicate scan summary
- `deduplication_decisions.{csv,parquet}` — 464 dropped firms with reasons
- `deduplication_summary.json` — deduplication summary

### Notebook
- `notebooks/03_kmax_convergence_analysis.ipynb` — pre-registered structure, mostly populated
- `notebooks/03_viz*.png` — 12 visualizations from the BUGGY sweep results, **will need regeneration after BC recomputation**
- 4 interpretation cells need writing (cells 14, 20, 28, 36) plus Section 7 narrative

### Pre-registered artifacts (will need updating)
- `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` — skeleton with `[RESULT]` placeholders

### ADRs (will need updating)
- `docs/adr/adr_004_k_selection_method.md` — needs reopening for the not-converged + misspecification context
- `docs/specs/firm_portfolio_spec.md` — needs deduplication step added as input validation

---

## Codex deployment scripts (still valid)
- `scripts/start_kmax_sweep.sh` — launcher
- `scripts/watch_sweep_and_shutdown.sh` — watchdog
- `infrastructure/user_data_kmax_sweep.sh` — EC2 bootstrap

---

## What I'd tell future Claude (or current Codex) starting from here

1. **Read the diagnostic findings doc first** (`docs/epics/week2_firm_portfolios/kmax_diagnostic_findings.md`). It's the single source of truth.
2. **The 12 PNGs in `notebooks/` are from the BUGGY sweep** — they tell the story of the discovery, but should NOT be presented as final results. Regenerate after the corrected BC matrices are computed.
3. **Don't re-fit GMMs**. The fitted parameters in `firm_gmm_parameters_k*.parquet` are unaffected by the BC formula bug or duplicate firms. Only the BC matrices need recomputation.
4. **The deduplication is a 5.8% removal**. This is a substantial reduction in the small-firm bin (9.3% there) but NOT the dominant population. The misspecification test will reveal whether the K explosion is purely a duplicate artifact or has a residual non-Gaussian component.
5. **The user (Torrin) has high latency tolerance and prefers depth.** Don't rush. Run thorough analyses, present findings with evidence, propose options, let him decide.
6. **The user thinks in terms of supervisor meetings.** Frame work as what Jan Bena would want to see. Pre-registration discipline matters. Honest scientific framing of negative findings is preferred over forced positive narratives.
7. **Findings should be evidence-backed.** Every claim should have a script + output file + table. The diagnostic findings doc is the template.

---

## Memory entries I've saved

- `user_torrin.md` — user role, communication style, preferences
- `project_kmax_sweep_result.md` — original NOT_CONVERGED result
- `project_duplicate_firms_bc_formula.md` — the two main findings
- `feedback_thoroughness.md` — preference for depth over speed
- `reference_s3_artifacts.md` — S3 paths

---

## Status at handover (2026-04-11)

- ✅ Diagnostic sequence complete
- ✅ Root causes identified (duplicates + BC formula + suggestive misspecification)
- ✅ Duplicate firm scan complete
- ✅ Unified deduplication rule defined and validated
- ⏳ **NEXT**: Apply deduplication, recompute BC matrices with linear weights (Codex VM)
- ⏳ **NEXT**: Run misspecification diagnostics on cleaned dataset
- ⏳ **NEXT**: Populate notebook + knit to PDF + email team
- ⏳ **NEXT**: Update ADR-004, firm_portfolio_spec.md
- ⏳ **NEXT**: TDD implementation of PortfolioBuilder/GMMFitter (after design revision)
