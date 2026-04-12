# K_max Sweep Diagnostic Findings

**Status**: HALT — awaiting BC recomputation + misspecification tests  
**Date**: 2026-04-09 (initial), updated 2026-04-11  
**Run**: 20260409T170706Z (c5.4xlarge, 2.84 hours, all 7,949 non-excluded firms)  
**Author**: Claude Code (Week 2 interpretation instance)

---

## Executive Summary

The diagnostic sequence prescribed by Codex was designed to distinguish near-tie ranking noise from genuine model misspecification. **It found neither — at least not as the dominant cause.** Instead, it uncovered compounding issues — one in the data, one in the metric, plus suggestive (but unconfirmed) evidence of model misspecification — that together explain the observed top-50 instability. All must be addressed before Week 3.

**Finding 1 (Data — Aliases):** The top-50 firm pairs at K_max=10-15 are **entirely composed of duplicate firms** — pairs that share 100% of their patents under anagram-style PRIV_ names. These are data quality artifacts, not M&A candidates. Total: 341 alias pairs, 196 cliques, 264 firms involved.

**Finding 2 (Data — Subsidiaries):** The data also contains 141 parent/subsidiary pairs (containment ≥ 0.95) and 59 predecessor records (similar size with near-perfect nesting). These represent already-completed M&A transactions encoded as duplicate firm records. **Critically: leaving these in the dataset would generate false-positive M&A predictions** ("Alphabet should acquire Waymo" when Alphabet acquired Waymo in 2016).

**Finding 3 (Metric):** The mixture-level BC formula uses `√(πᵢ·πⱼ)` weighting, which is an **upper bound on the true Bhattacharyya Coefficient**, not the BC itself. It produces values up to 5.39 (theoretical max is 1.0). At K_max≥20, inflated values from small firms with many components completely displace the duplicate-firm pairs.

**Finding 4 (Compounding):** The 0% top-50 overlap at K_max=15→20 is a phase transition between two different failure modes: duplicate-dominated ranking (K_max≤15) versus inflation-dominated ranking (K_max≥20). Neither represents genuine technological similarity.

**Finding 5 (Misspecification — UNCONFIRMED):** Even after accounting for findings 1-4, **effective K does not saturate by K_max=30** (mean grows from 8.0 to 13.7), and small firms (50-200 patents) show K values that vastly exceed Bayesian prior expectations. This is consistent with — but not yet proof of — the Gaussian assumption being inadequate for UMAP-reduced patent vectors. Direct Gaussianity diagnostics (Mahalanobis Q-Q vs χ²(50), Mardia's test, prior sensitivity) are required to confirm or rule out.

**The underlying firm-similarity landscape IS stable** (Spearman ρ ≈ 0.99, median NN-5 overlap = 100%, 59% of firms preserve all 5 nearest neighbors across K_max=15→20). The instability is entirely in the extreme tail.

---

## Finding 1: Duplicate Firms in the Data

### Evidence

At K_max=10, ALL 50 top-ranked pairs are pairs of firms that share 100% of their patents:

| Rank | Firm A | Firm B | Shared Patents | BC |
|------|--------|--------|---------------|-----|
| 1 | PRIV_PARADETECHNOLOGIES | PRIV_PRIDETECHNOLOGIES | 136/136 (100%) | 1.000700 |
| 2 | PRIV_ZINTECHNOLOGIES | PRIV_ZONETECHNOLOGIES | 57/57 (100%) | 1.000173 |
| 6 | PRIV_AASKITECHNOLOGY | PRIV_OSKITECHNOLOGY | 579/579 (100%) | 1.000006 |
| 10 | PRIV_VIATECHNOLOGIES | PRIV_VTECHNOLOGIES | 2016/2016 (100%) | 1.000000 |

These are the **same company appearing under anagrammatic or near-anagrammatic name variations** (PARADE/PRIDE, ZIN/ZONE, ASKI/OSKI, VIA/V). They have identical patent portfolios because they ARE the same entity.

**Shared patent summary across K_max values:**
- **K_max=10: 50/50** top pairs share 100% of patents (8,098 total shared patents)
- **K_max=15: 50/50** top pairs share 100% of patents (same firms, slightly different ordering)
- **K_max=20: 0/50** top pairs share any patents (entirely different population)
- **K_max=30: 0/50** top pairs share any patents

### Interpretation

The 337 pairs at BC within 1e-4 of 1.0 at K_max=10 are not "near-ties" in the statistical sense — they are **exact duplicates**. Two firms with identical patent portfolios produce identical GMM parameters and therefore BC = 1.0 (up to numerical precision). Top-k ranking among these pairs is arbitrary because they are all equally "similar" — they're the same firm.

### Impact

This means the 80% top-50 overlap at K_max=10→15 (the only transition that *appeared* to pass the threshold) was measuring **stability of duplicate-pair identification**, not stability of technological similarity rankings. The entire top-k convergence analysis is corrupted by this data issue.

### Required Action

Deduplicate firms before any downstream BC analysis. Proposed rule: if two firms share >80% of their patents, merge them into a single entity (keeping the gvkey with more unique patents). This is a **PortfolioBuilder preprocessing step** — add it to the spec before implementation.

---

## Finding 2: Metric Inflation (√-Weighted BC Formula)

### The Formula

The current `bc_mixture` function (line 473 of `run_kmax_sweep.py`) computes:

```
BC(A, B) = Σᵢ Σⱼ √(πᵢᴬ · πⱼᴮ) · BC(Nᵢᴬ, Nⱼᴮ)
```

The component-level `BC(Nᵢ, Nⱼ) ∈ [0, 1]` is correct. The issue is the `√(πᵢπⱼ)` weighting.

### Why It Exceeds 1.0

The sum `Σᵢⱼ √(πᵢπⱼ) = (Σᵢ √πᵢ)(Σⱼ √πⱼ)`. For K equal-weight components: `(K · √(1/K))² = K`. So when all component pairs overlap perfectly, the formula yields BC = K, not BC = 1.

| K_max | Max BC | Rank-50 BC | Mean self-BC | Firms with self-BC > 1 |
|-------|--------|-----------|-------------|----------------------|
| 10 | 1.0007 | 1.0000 | 1.0007 | 90 |
| 15 | 1.0328 | 1.0000 | 1.0029 | 213 |
| 20 | 2.5911 | 1.5421 | 1.0080 | 337 |
| 25 | 3.6923 | 2.6992 | 1.0171 | 422 |
| 30 | 5.3907 | 3.6177 | 1.0283 | 485 |

### Mathematical Status

This formula is an **upper bound** on the true Bhattacharyya Coefficient. Proof:

```
BC_true(A,B) = ∫ √(p(x)·q(x)) dx
             = ∫ √(Σᵢ πᵢ Nᵢ(x)) · √(Σⱼ ρⱼ Mⱼ(x)) dx
             ≤ ∫ [Σᵢ √πᵢ √Nᵢ(x)] · [Σⱼ √ρⱼ √Mⱼ(x)] dx   [triangle inequality: √(Σaᵢ) ≤ Σ√aᵢ]
             = Σᵢⱼ √(πᵢρⱼ) · ∫ √(Nᵢ(x) Mⱼ(x)) dx
             = Σᵢⱼ √(πᵢρⱼ) · BC(Nᵢ, Mⱼ)               [← THIS IS THE FORMULA IN OUR CODE]
```

The bound is tight when K=1 (exact). It grows progressively looser as K increases and components overlap. This creates a **systematic K-dependent bias**: firms with more components accumulate higher "BC" values.

### Who Dominates at High K_max

At K_max=20, the top-30 is dominated by **two hub firms**:
- **128663** (50 patents, K=16): appears in 20/30 top pairs, BC up to 2.59
- **PRIV_NEOTRACT** (68 patents, K=16): appears in 15/30 top pairs

At K_max=30:
- **029086** (66 patents, K=24): appears in 22/30 top pairs, BC up to 5.39
- **PRIV_RICON** (51 patents, K=23): appears in 12/30 top pairs

**None of these top pairs share any patents.** These are small firms with disproportionately many components (K=16-24 for n=50-68), whose many overlapping component pairs inflate the metric. A firm with K=24 components sharing 50% overlap with a partner creates a "BC" of ~24 × 0.5 × weight_factor — far exceeding the theoretical BC maximum of 1.0.

### Alignment with Original Methodology

Arthur's methodology (methodology.md) states: "calculate BC comparing each GM cluster across firms, **then aggregate using GMM weights**." It does NOT specify √ weights. The √ was an implementation artifact from conflating the √ in the BC *definition* (which applies to probability densities) with the mixture *weight aggregation*.

### Correction Options

| Option | Formula | Bounded? | Self-similarity | K_max stable? |
|--------|---------|----------|----------------|---------------|
| **Current (√ weights)** | `Σ √(πᵢπⱼ) BC(Nᵢ,Nⱼ)` | No (up to K) | BC(A,A)≈1 | No |
| **Linear weights** | `Σ πᵢπⱼ BC(Nᵢ,Nⱼ)` | Yes [0,1] | BC(A,A)=Σπᵢ²<1 | Yes but ties persist |
| **Cosine normalization** | `√_BC / √(self_A·self_B)` | Yes [0,1] | 1.0 | No (tested — creates massive ties) |
| **Monte Carlo true BC** | `∫√(p·q)dx` sampled | Yes [0,1] | 1.0 | Unknown (expensive) |

**Cosine normalization was tested and FAILED** — it compressed all top-200 to BC=1.0000 at every K_max (identical to the duplicate-firm problem), making top-k selection completely random. Normalized convergence metrics were **worse** than raw:

| Transition | Raw top-50 | Normalized top-50 |
|---|---|---|
| 10→15 | 80% | **24%** |
| 15→20 | 0% | 6% |
| 20→25 | 0% | 18% |
| 25→30 | 6% | 8% |

---

## Finding 3: The Phase Transition

The 0% overlap at K_max=15→20 marks a **phase transition between two regimes**:

**Regime 1 (K_max ≤ 15): Duplicate-dominated.** The top-50 is composed of firms sharing 100% patents. Effective K is moderate (mean 8-10), ceiling rate is high (11-35%), and the √-weighted formula stays near 1.0. Rankings are "stable" (80% overlap at K10→K15) because the same duplicates dominate at both settings.

**Regime 2 (K_max ≥ 20): Inflation-dominated.** Effective K grows (mean 12-14), ceiling rate drops to 0.2-3%, and the √-weighted formula produces values up to 2.6-5.4. Hub firms with many components dominate the top-50. The duplicates are pushed out entirely.

The transition at K_max=15→20 coincides with the ceiling rate dropping from 11.7% to 3.0% — the point where most GMM-tier firms first express their full component complexity, unleashing the inflation effect.

---

## What the Sweep Actually Proved

Despite the issues above, the sweep produced valuable results:

1. **The bulk firm-similarity landscape IS stable.** Spearman ρ ≈ 0.99 across all K_max transitions. The relative similarity ordering of 99%+ of firm pairs is robust.

2. **Per-firm nearest-neighbor stability is excellent.** Median NN-5 overlap = 100% across all transitions. The typical firm's most similar partners do not change with K_max.

3. **Effective K decelerates appropriately.** Mean K grows from 8.04 to 13.67, ceiling rate drops from 34.5% to 0.2%. The Bayesian pruning is working as designed for the population.

4. **The data has a deduplication problem** that would have corrupted Week 3 M&A prediction. The sweep caught it.

---

## Recommended Actions

### Immediate (Before Any Design Revision)

1. **Deduplicate firms with >80% shared patents.** This is a data preprocessing step that should have been caught in EDA. Add it to the PortfolioBuilder spec as a mandatory input validation step.

2. **Fix the BC formula.** Change `bc_mixture` to use linear weights (`πᵢπⱼ` instead of `√(πᵢπⱼ)`). This:
   - Keeps BC bounded in [0, 1]
   - Aligns with Arthur's methodology ("aggregate using GMM weights")
   - Eliminates K-dependent inflation
   - Has clean interpretation: "expected overlap between random technology areas from each firm"

3. **Recompute convergence from saved GMM parameters.** The parquet files contain all GMM parameters. We can recompute BC matrices with the corrected formula without re-running GMM fitting. This is a ~2-3 hour computation.

### After Recomputation

4. **Re-evaluate convergence** with deduplicated firms and corrected metric. Convergence MAY emerge — we don't know until we recompute.

5. **Populate the notebook and executive summary** with corrected results. The pre-registered structure stays; the data changes.

6. **Update ADR-004** based on the corrected convergence analysis, not the corrupted one.

### Questions for Torrin

- **Deduplication threshold**: 80% shared patents? 100%? Or a Jaccard threshold on patent sets?
- **BC formula choice**: Linear weights (πᵢπⱼ) is the cleanest fix. Any preference for MC true BC instead? (Much more expensive but theoretically exact.)
- **Should we investigate the high-K firms** (K=16-24 from 50-68 patents)? Their effective K seems high — possible overfitting despite Bayesian pruning.

---

## Diagnostic Evidence Summary

### Diagnostic 1: Tail-Margin Analysis

| K_max | BC at rank 1 | BC at rank 50 | Top-50 span | Pairs within 1e-4 of cutoff |
|-------|-------------|---------------|-------------|----------------------------|
| 10 | 1.0007 | 1.0000 | 0.000699 | 337 |
| 15 | 1.0328 | 1.0000 | 0.032754 | 322 |
| 20 | 2.5911 | 1.5421 | 1.048956 | 1 |
| 25 | 3.6923 | 2.6992 | 0.993129 | 1 |
| 30 | 5.3907 | 3.6177 | 1.773056 | 1 |

Cross-K_max shift/gap ratio = **13.2** (K=10→15). BC shifts between K_max values are 13x larger than gaps between adjacent ranks in the top-50, confirming the near-tie mechanism. But the ties are between duplicate firms.

### Diagnostic 2: Robust-Core Analysis

| Top-k | Robust (all 5 K_max) | Total unique | 1-of-5 only |
|-------|---------------------|-------------|-------------|
| 50 | **0** | 207 | 164 (79.2%) |
| 100 | **0** | 399 | 301 (75.4%) |
| 200 | **0** | 841 | 686 (81.6%) |
| 500 | **0** | 1,777 | 1,278 (71.9%) |

**Zero** robust pairs at any top-k level. 79% of top-50 pairs appear at only 1 of 5 K_max values.

### Diagnostic 3: Firm Characteristics

- 464 unique firms appear in any top-200 across all K_max values
- **0 robust firms** (appear at all 5 K_max)
- 252 of 464 volatile firms are single-Gaussian (displaced innocent bystanders)
- 212 are GMM-tier (the inflated firms and their partners)
- Top-200 Jaccard overlap matrix shows near-zero cross-regime overlap:
  - K10↔K15: 54.4% (same-regime — duplicate dominated)
  - K10↔K20: 0.8% (cross-regime)
  - K10↔K30: 0.0% (complete turnover)

### Diagnostic 4: BC Distribution

| K_max | Mean BC | P99.99 | Max | Pairs > 0.5 |
|-------|---------|--------|-----|-------------|
| 10 | 0.000364 | 0.259 | 1.001 | 582 |
| 20 | 0.000745 | 0.490 | 2.591 | 3,016 |
| 30 | 0.001858 | 1.419 | 5.391 | 26,656 |

The entire distribution shifts right with K_max. The number of pairs with BC > 0.5 grows from 582 to 26,656 — a 46x increase driven by metric inflation.

---

## Finding 5 (Added 2026-04-11): The Subsidiary Problem

### The question that surfaced this finding

While discussing the deduplication strategy, Torrin asked: "If our goal is to identify mimicking portfolios, would these subsidiary relationships affect this goal? Would we ever 'predict' that a corporation should acquire a subsidiary it already owns?"

The answer is yes — and this is structurally bad, not just noisy.

### The pathological case

If we leave subsidiary records in the dataset, the BC analysis would correctly identify pairs like (Alphabet, Waymo) as high-similarity. The M&A prediction model would then surface Waymo as an acquisition target for Alphabet. **But Alphabet acquired Waymo in 2016.** No transaction will occur. The "prediction" is a false positive — not because the similarity measurement is wrong, but because we're predicting an event that already happened.

This applies to **183+ pairs** in the data:
- (Alphabet, Waymo) — autonomous driving, acquired 2016
- (Alphabet, Verily) — life sciences, founded 2015
- (Qualcomm, Snaptrack) — GPS, acquired 2000
- (Qualcomm, Qualcomm Atheros) — networking, acquired 2011
- (J&J, Ethicon) — surgical devices, long-time subsidiary
- (J&J, Janssen Pharmaceuticals) — pharma division
- (007585, Motorola Mobility) — Motorola's 2,356 patents are 99% nested in 007585's 23K
- (IBM, IBM-International predecessor record)

Each generates a high-confidence M&A "prediction" for an event that already occurred.

### Why this is structurally bad

1. **Precision@K cratering**: When we evaluate by ranking the top-K predicted M&A pairs against actual 2021 M&A labels, parent/subsidiary pairs CANNOT match any 2021 transaction. They guarantee false positives.
2. **Validation contamination**: The 2021 "Springboard Effect" labels won't contain any of these because the deals already closed.
3. **Rank pollution**: Even pairs ranked 50-200 (not top) crowd out genuine candidates.
4. **Conceptual confusion**: Patent assignment in Compustat reflects current ownership. Alphabet's patent portfolio already includes Waymo's patents. Adding Waymo as a separate firm record means we're double-counting that same technology.

### The deeper conceptual fix

For the M&A prediction task to make sense, **each row in our firm dataset should represent one distinct, currently-independent legal entity.** Subsidiaries don't satisfy this — they're already merged. The dataset gives us legal entities under different levels of corporate hierarchy, but a parent firm and its wholly-owned subsidiary are not two distinct technology bundles — they're one bundle plus a subset view.

---

## Duplicate Firm Scan Results (2026-04-11)

A project-wide scan was run to characterize the full distribution of firm pair overlaps. Script: `scripts/duplicate_firm_scan.py`. Output: `output/kmax_sweep/duplicate_pairs.parquet`, `output/kmax_sweep/duplicate_scan.json`.

### Distribution

- **991 firm pairs** share at least one patent (out of 31.6M possible non-excluded pairs — 0.003%)
- **341 pairs at Jaccard = 1.000 exactly** (strict aliases — anagram cliques)
- **9 pairs in [0.95, 0.99)** (near-aliases — likely same entity, different records)
- **0 pairs in [0.99, 1.0)** — clean break in the distribution
- **183 nested pairs** (containment_max ≥ 0.95, Jaccard < 0.80) — subsidiary relationships
- **124 moderate-overlap pairs** (Jaccard 0.50-0.95) — corporate reorganizations / spinoffs

### The histogram shows a natural gap

```
[0.000, 0.001):     167  
[0.001, 0.010):      98  
[0.010, 0.050):      72  
[0.050, 0.100):      47  
[0.100, 0.200):      41  
[0.200, 0.300):      27  
[0.300, 0.500):      65  
[0.500, 0.700):      62  
[0.700, 0.800):      26  
[0.800, 0.900):      21  
[0.900, 0.950):      15  
[0.950, 0.990):       9  ← only 9 pairs in this band
[0.990, 0.999):       0  ← ZERO
[0.999, 1.000]:     341  ← spike (anagram aliases)
```

This natural gap supports a containment-based deduplication rule.

### Strict alias cliques

196 cliques involving 450 unique firms. Mostly pairs (159), but some larger:
- 1 clique of 7 firms (PRIV_EASETECHNOLOGIES, IOSTECHNOLOGIES, ISOTECHNOLOGIES, OSATECHNOLOGIES, OSOTECHNOLOGIES, OSTECHNOLOGIES, SITECHNOLOGIES — 88 patents each)
- 2 cliques of 5 firms
- 13 cliques of 4 firms
- 21 cliques of 3 firms
- 159 cliques of 2 firms

The pattern (anagram-style PRIV_ names with identical patent sets) is unmistakable as data quality artifacts.

---

## The Unified Deduplication Rule

### Rule

**Drop firm B if there exists another firm A in the dataset such that:**
- `|A| ≥ |B|` (A has at least as many patents as B)
- `containment(B → A) ≥ 0.95` (at least 95% of B's patents are also in A)

**Tiebreaker** (when sizes are equal): drop the PRIV_-prefixed firm if mixed; otherwise alphabetically larger. **Important caveat**: in some cases (e.g., Lyft, GeneralData), the PRIV_ record is actually MORE complete (has more patents). The rule "keep the larger" handles this correctly — it keeps the more complete record regardless of prefix.

### Application

Script: `scripts/duplicate_firm_unified_rule.py`. Output: `output/kmax_sweep/deduplication_decisions.csv`, `output/kmax_sweep/deduplication_summary.json`.

| Metric | Count |
|--------|-------|
| Pairs with `containment_max ≥ 0.95` | **568** |
| Firms marked for removal | **464** |
| Remaining non-excluded firms | **7,485** (down from 7,949) |
| Reduction | **5.8%** |

**Breakdown by removal category:**

| Category | Count | Description |
|----------|-------|-------------|
| Alias | 264 | Bidirectional ≥0.95 containment (anagram cliques + same-size near-duplicates) |
| Subsidiary | 141 | Smaller firm fully nested in much larger parent (size ratio ≥1.5) |
| Predecessor | 59 | Similar-size firms with near-perfect nesting (typically pre/post IPO records) |

### Test cases (all produced correct answers)

| Pair | Result |
|------|--------|
| (Alphabet 160329, PRIV_WAYMO) | Drop Waymo, keep Alphabet ✓ |
| (Alphabet 160329, PRIV_VERILYLIFESCIENCES) | Drop Verily, keep Alphabet ✓ |
| (Alphabet 160329, PRIV_GOOGLE) | Drop PRIV_GOOGLE, keep Alphabet ✓ |
| (Qualcomm 024800, PRIV_SNAPTRACK) | Drop Snaptrack, keep Qualcomm ✓ |
| (Qualcomm 024800, PRIV_QUALCOMMATHEROS) | Drop Atheros, keep Qualcomm ✓ |
| (IBM 006066, PRIV_INTERNATIONALBUSINESS) | Drop PRIV_, keep IBM ✓ |
| (007585, PRIV_MOTOROLAMOBILITY) | Drop Motorola Mobility ✓ |
| (034873, PRIV_LYFT) | Drop **034873**, keep PRIV_LYFT (PRIV_ has more patents) |
| (003760, PRIV_GENERALDATA) | Drop **003760**, keep PRIV_GENERALDATA (PRIV_ has more patents) |

### Notable subsidiary discoveries

The unified rule catches **227 pairs** that the strict Jaccard ≥ 0.99 rule would miss. Major findings:

- **Qualcomm (024800)**: 8 subsidiary records caught (Snaptrack, QualcommTech, Pixtronix, QualcommAtheros, DigitalFountain, SummitMicroelectronics, IridigmDisplay, RapidBridge)
- **Johnson & Johnson (006266)**: 6 subsidiary records caught (Ethicon, EthiconEndoSurgery, JanssenPharmaceuticals, JohnsonJohnsonConsumer, BioSenseWebster, JanssenBiotech)
- **Alphabet (160329)**: 3 subsidiary records (Google predecessor, Waymo, Verily)

### Impact on the K_max sweep dataset

- **GMM-tier (≥50 patents) firms dropped: 139**
- Single-Gaussian (5-49) firms dropped: 325
- **Effect on the small GMM-tier bin (50-200 patents)**: dropped 103 of 1,102 firms (9.3% reduction). This is the bin where the K explosion lives in Viz 2C.

This sets up a controlled experiment: if the K explosion was purely a duplicate artifact, the post-deduplication Viz 2C should look much cleaner. If genuine misspecification persists in the remaining ~999 small firms, we'll see it directly.

### Borderline cases for follow-up

**16 pairs in [0.95, 0.97)** — closest calls, all look like genuine subsidiaries:
- Qualcomm/Snaptrack (0.966)
- J&J/Janssen Biotech (0.961), Janssen Pharmaceuticals (0.962)
- 001013/DCI Telecommunications (0.958, equal sizes — corporate restructuring)

**34 pairs in [0.85, 0.95)** — NOT caught, would require manual review for any misses:
- 179202 / Hughes Network Systems (0.950 — just below threshold)
- 008972 / Switchcraft (0.935)
- 025783 / McAfee (0.921)
- 003520 / CordisDow (0.933 — likely a J&J entity)

These ~34 pairs are left in the dataset. If the misspecification investigation shows residual issues in this size range, we may want to inspect them manually.

---

## Recommended Next Actions (in order)

1. **Apply the unified deduplication rule** to produce `firms_dedup.parquet` (7,485 firms) — local, fast
2. **Recompute BC matrices** from saved GMM parameters with corrected linear-weight formula, restricted to deduplicated set — Codex VM (~1-2 hours)
3. **Recompute convergence metrics** to see if top-50 instability disappears — local, fast
4. **Run misspecification diagnostics** on deduplicated, corrected dataset:
   - Mahalanobis vs χ²(50) Q-Q on stratified sample (~50 firms)
   - Mardia's multivariate skewness/kurtosis test
   - Per-dimension Q-Q plots for visual inspection (~10 firms)
   - Prior sensitivity test: refit ~30 firms with γ ∈ {2.0, 5.0}
   - Optional: PCA-50 vs UMAP-50 sensitivity test on ~20 firms
5. **Populate notebook** with corrected results + new findings
6. **Knit to PDF** and email team

---

## Provenance

### Scripts
- `scripts/diagnostic_kmax_stability.py` — original four-step diagnostic
- `scripts/verify_bc_normalization.py` — cosine normalization test (rejected)
- `scripts/verify_linear_weights.py` — linear-weight comparison test
- `scripts/identify_top_pairs.py` — top-pair identity analysis (revealed duplicates)
- `scripts/duplicate_firm_scan.py` — project-wide Jaccard scan
- `scripts/duplicate_firm_unified_rule.py` — containment-based deduplication

### Outputs
- `output/kmax_sweep/diagnostic_results.json` — original diagnostic findings
- `output/kmax_sweep/duplicate_pairs.parquet` — all 991 overlapping pairs with Jaccard/containment
- `output/kmax_sweep/duplicate_scan.json` — distribution summary
- `output/kmax_sweep/deduplication_decisions.csv` — 464 dropped firms with reasons
- `output/kmax_sweep/deduplication_summary.json` — summary statistics
- `output/kmax_sweep/bc_matrix_all_k{10,15,20,25,30}.npz` — original BC matrices (with √-weighted formula)
- `output/kmax_sweep/firm_gmm_parameters_k{10,15,20,25,30}.parquet` — GMM parameters (still valid, formula doesn't affect fits)

### Code under review
- `scripts/run_kmax_sweep.py:473` (`bc_mixture` function) — needs linear-weight fix (`πᵢπⱼ` instead of `√(πᵢπⱼ)`)

### Notebook visualizations
- `notebooks/03_kmax_convergence_analysis.ipynb` — pre-registered structure, mostly populated, 4 interpretation cells + Section 7 narrative remaining
- `notebooks/03_viz*.png` — 12 visualizations generated from the original (uncorrected) sweep results. Will need regeneration after BC recomputation.
