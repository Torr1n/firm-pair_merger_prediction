# Week 2 Bootstrap: Firm Patent Portfolio Construction

**Purpose**: Initialize a fresh Claude Code instance to execute the second week of the Firm-Pair Merger Prediction project — building firm-level patent portfolio representations using Gaussian Mixture Models (Step 2 of 4).

---

## Mission Briefing

You are the Week 2 development instance for the **Firm-Pair Merger Prediction** project. Week 1 is complete: a patent vectorization pipeline produced 50-dimensional patent vectors for ~1.45M patents across 15,814 firms. Your mission is to build the second stage of the pipeline: aggregating each firm's patent vectors into a probabilistic portfolio representation using Gaussian Mixture Models (GMMs).

### What You're Building

A firm portfolio construction pipeline that:

1. Loads Week 1 outputs (50D patent vectors + gvkey mapping)
2. Groups patents by firm (gvkey), handling co-assigned patents
3. Runs EDA on firm-size distribution and normalization sensitivity
4. Fits a GMM per firm on its 50D patent vectors
5. Serializes each firm's GMM parameters (means, covariances, weights) for downstream use
6. Saves checkpoints at every stage

### Why This Matters

This step is the bridge between individual patents and firm-level comparison. Week 1 produced a 50D vector per patent. Week 3 needs a probability distribution per firm to compute Bhattacharyya Coefficients and complementarity metrics. The GMM transforms a bag of discrete patent vectors into a continuous distribution that captures: how many technology areas the firm operates in (K clusters), what each area looks like (means, covariances), and how much of the portfolio is in each area (mixing weights).

If the GMMs are poorly fitted — wrong K, unstable covariances, sensitivity to normalization — every downstream comparison will be unreliable. This is why EDA and normalization sensitivity come first, why K selection requires an ADR, and why we validate GMM quality before declaring the sprint complete.

### The Four-Step Methodology (Full Context)

You are implementing **Step 2 only**. Understanding the full pipeline informs your design:

1. **Vectorize Patents** (Week 1 — COMPLETE)
   - PatentSBERTa on title+abstract (768D) + citations (768D) -> 1536D -> UMAP -> 50D per patent

2. **Firm Patent Portfolios** (Week 2 — THIS SPRINT)
   - Aggregate each firm's 50D vectors into a Gaussian Mixture Model
   - K clusters with mixing weights -> firm's technology distribution
   - Mixing weights are "technology share" — the proportion of the portfolio in each cluster

3. **Compare Distributions** (Week 3+)
   - Bhattacharyya Coefficient: overlap/similarity between two GMMs
   - Directional Technology Complementarity: asymmetric gap-filling measure
   - These require GMM parameters (means, covariances, weights) as input

4. **Extensions** (Week 4+)
   - Synthetic portfolio matching: forward problem (A+B=C?) and inverse problem (find B)

### The Team

| Name | Role |
|------|------|
| **Torrin Pataki** | Development lead. Your primary collaborator and approval authority. |
| **Arthur Khamkhosy** | Methodology design. Authored the four-step methodology. |
| **Amie Le Hoang** | Data preparation. Prepared the three parquet datasets. |
| **Ananya Ravichandran** | Researcher |
| **Duncan Harrop** | Researcher |
| **Jan Bena** | Faculty advisor (UBC Sauder). Co-author of Bena & Li (2014). |
| **Codex** | Impartial reviewer, pair-programmer, and deployment lead. |

---

## What Week 1 Produced (Your Inputs)

Week 1 delivered a complete patent vectorization pipeline. Your inputs are:

| File | Description | Shape |
|------|-------------|-------|
| `output/embeddings/patent_vectors_50d.parquet` | 50D patent vectors (binary-serialized numpy arrays) | ~1.45M rows x 50D |
| `output/embeddings/gvkey_map.parquet` | patent_id -> gvkey mapping (includes co-assignments) | ~1.53M rows (patent_id, gvkey) |

**Key characteristics of the input data:**

- **15,814 unique firms** (gvkeys) after v3 data + post_deal_flag filtering
- **~1.45M unique patents** with 50D vectors; ~85K co-assigned patents appear in multiple firms' portfolios (correct behavior — a jointly-owned patent should contribute to both firms)
- **post_deal_flag already filtered upstream** — Week 2 receives only pre-deal patents. Do not re-filter.
- **Embeddings are NOT L2-normalized**: Title+abstract L2 norms have mean=6.79, std=0.14. UMAP output dimensions will have their own scale characteristics that the EDA must quantify.
- **Citation norms are bimodal** (mean pooling effect): The 50D UMAP vectors preserve this structure. Normalization sensitivity is an open question for GMM fitting.
- **2D UMAP shows meaningful structure**: Visible firm-level clustering tendencies. Patents from focused firms cluster tighter than patents from diversified mega-firms (e.g., IBM spans the space). This is encouraging for GMM fitting.

### Loading Pattern

```python
from src.utils.checkpointing import CheckpointManager

cm = CheckpointManager("output/embeddings")
patent_ids, vectors_50d, metadata = cm.load_embeddings("output/embeddings/patent_vectors_50d.parquet")
# vectors_50d: np.ndarray shape (N, 50), dtype float32

import pandas as pd
gvkey_map = pd.read_parquet("output/embeddings/gvkey_map.parquet")
# Columns: patent_id, gvkey
```

---

## Required Reading (In This Order)

### 1. Project Foundation

| Document | Location | Why Read It |
|----------|----------|-------------|
| **Project Instructions** | `CLAUDE.md` | Development values, architecture, module structure, environment |
| **Development Values** | `docs/values/` (all 10 files) | Non-negotiable operational principles |

### 2. Week 1 Artifacts

| Document | Location | Why Read It |
|----------|----------|-------------|
| **Week 1 Retrospective** | `docs/sprint_retrospectives/week1_instance_summary.md` | What was built, key findings, lessons learned |
| **ADR-001** | `docs/adr/adr_001_patent_embedding_model.md` | L2 norm finding — embeddings are NOT unit-normalized |
| **ADR-002** | `docs/adr/adr_002_citation_aggregation.md` | Bimodal citation norms from mean pooling |
| **Patent Vectorizer Spec** | `docs/specs/patent_vectorizer_spec.md` | Week 1 interface contracts, storage format |
| **Pipeline Script** | `scripts/run_full_pipeline.py` | How Week 1 outputs are produced and saved |

### 3. Methodology

| Document | Location | Why Read It |
|----------|----------|-------------|
| **Methodology Email** | `methodology.md` | Arthur's four-step pipeline — ground truth for Step 2 |
| **Presentation** | `docs/references/presentation_methodology.pdf` | Pages 7-8: GMM methodology, K selection (BIC vs Bayesian), pros/cons table |

### 4. Configuration

| Document | Location | Why Read It |
|----------|----------|-------------|
| **Pipeline Config** | `src/config/config.yaml` | Current config structure — you will add a `portfolio` section |

---

## Guiding Principles

These are **non-negotiable**. Violating them will result in Codex reviewer rejection. Read the full articulation in `docs/values/`.

### 1. The Code Quality Standard

> "The best engineers write code my mom could read. They choose boring technology, they over-document the 'why,' and under-engineer the 'how.' Complexity is not a flex; it becomes a liability."

- **Simplicity over cleverness**: If there are two ways to do something, choose the simpler one
- **Boring technology**: scikit-learn's `GaussianMixture` and `BayesianGaussianMixture` — nothing exotic
- **Over-document the "why"**: Every K selection decision, covariance type choice, and threshold needs rationale
- **Under-engineer the "how"**: Don't build a GMM framework. Build functions that fit GMMs to firms.

### 2. Test-Driven Development (Mandatory)

```
1. Read the interface spec
2. Write tests that validate the spec
3. Implement the minimum code to pass tests
4. Refactor if needed (tests still pass)
5. Repeat
```

**No exceptions.** Tests are written BEFORE implementation.

### 3. Spec-Driven Development

Write the spec, get it approved, then build to spec. Deviations require explicit user approval.

### 4. Checkpoint Everything

Save GMM parameters after fitting. If the process crashes at firm 10,000 of 15,814, resume from firm 10,000 — not from scratch.

### 5. Evidence-Based Engineering

"The GMM fits well" is not evidence. BIC scores, silhouette scores, a firm-size vs K scatter plot — these are evidence.

---

## Codex Review Protocol

**Codex Review Checkpoints** — development HALTS until Codex approves:

| Checkpoint | When | What Codex Reviews |
|------------|------|-------------------|
| **Design Review** | After ADRs 004-007 and spec are written | K selection method, covariance type, thresholds, interface design |
| **Implementation Review** | After all code and tests are written | Spec conformance, test coverage, value adherence, correctness |
| **Validation Review** | After GMMs are fitted and validation notebook is produced | Evidence quality, statistical soundness, coverage |

---

## Environment Setup

The project venv from Week 1 should already exist. Verify it has the required dependencies:

```bash
cd /mnt/c/Users/TPata/firm-pair_merger_prediction
source venv/bin/activate

# Verify Week 2 dependencies (all should be in requirements.txt already)
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__} OK')"
python -c "import pandas; print('pandas OK')"
python -c "import numpy; print('numpy OK')"
python -c "import matplotlib; print('matplotlib OK')"
```

**No GPU needed for Week 2.** GMM fitting is CPU-only and the full 15K-firm run should complete in 30 minutes to 2 hours. Main memory concern: loading all 50D vectors (~275 MB for 1.45M x 50 x 4 bytes) is trivial.

---

## Week 2 Deliverables

| # | File | Description | Phase |
|---|------|-------------|-------|
| **EDA** | | | |
| 1 | `notebooks/03_portfolio_eda.ipynb` | Firm-size distribution, UMAP dimension scale analysis, normalization sensitivity | Phase 1 |
| **Architecture** | | | |
| 2 | `docs/adr/adr_004_k_selection_method.md` | ADR: Bayesian GMM vs BIC sweep for determining K | Phase 2 |
| 3 | `docs/adr/adr_005_minimum_patent_threshold.md` | ADR: Small-firm handling and minimum patent count per tier | Phase 2 |
| 4 | `docs/adr/adr_006_covariance_type.md` | ADR: Diagonal vs full covariance (and tiered approach) | Phase 2 |
| 5 | `docs/adr/adr_007_prenormalization_strategy.md` | ADR: Raw vs L2-normalized vs z-score standardized input vectors | Phase 2 |
| 6 | `docs/specs/firm_portfolio_spec.md` | Interface spec: PortfolioBuilder and GMMFitter | Phase 2 |
| **Implementation** | | | |
| 7 | `src/portfolio/portfolio_builder.py` | Group patents by firm, apply thresholds, dispatch to GMMFitter | Phase 3 |
| 8 | `src/portfolio/gmm_fitter.py` | Fit GMM per firm, extract parameters, handle edge cases | Phase 3 |
| **Tests** | | | |
| 9 | `tests/unit/test_portfolio_builder.py` | Firm grouping, co-assignment handling, threshold logic | Phase 3 |
| 10 | `tests/unit/test_gmm_fitter.py` | GMM fitting, parameter extraction, edge cases (small firms, single-cluster) | Phase 3 |
| **Checkpoints** | | | |
| 11 | `output/portfolios/firm_gmm_parameters.parquet` | Serialized GMM parameters for all firms (FINAL OUTPUT) | Phase 3 |
| **Validation** | | | |
| 12 | `notebooks/04_portfolio_validation.ipynb` | GMM quality metrics, K distribution, coverage stats | Phase 4 |
| **Documentation** | | | |
| 13 | `docs/sprint_retrospectives/week2_instance_summary.md` | Sprint retrospective | Phase 4 |

---

## Technical Design Decisions

### 1. GMM Fitting: One Model Per Firm

Each firm gets its own GMM fitted on its set of 50D patent vectors. The GMM output per firm:

- **K means**: K vectors of 50D each — the cluster centers (technology area centroids)
- **K covariances**: K matrices (50x50 for full, 50-length vectors for diagonal) — the spread of each cluster
- **K mixing weights**: K scalars summing to 1.0 — "technology share," the proportion of the firm's portfolio in each cluster

The mixing weights have direct economic interpretation: a firm with weights [0.6, 0.3, 0.1] has 60% of its patents in one technology area, 30% in a second, and 10% in a third. This is what makes GMMs suitable for our methodology — they naturally decompose a portfolio into weighted technology areas.

### 2. K Selection (ADR-004 Required)

Two approaches to evaluate in the ADR:

**Option A — BIC Sweep**: Fit K=1 through K_max per firm, select K minimizing BIC.
- Pros: Well-understood, deterministic, interpretable
- Cons: 15,814 firms x K_max fits = computationally expensive; requires choosing K_max

**Option B — Bayesian GMM**: Set K_max and a Dirichlet concentration prior. The model auto-prunes unused components (sets their weight to near-zero).
- Pros: One fit per firm; naturally handles firm-size skew (small firms auto-select fewer components); no BIC loop
- Cons: Hyperparameter sensitivity (concentration prior); less interpretable pruning behavior

**Recommendation**: Lean toward Bayesian GMM for computational efficiency and natural handling of the extreme firm-size skew (50.8% of firms have <10 patents). But the ADR must evaluate both approaches with evidence from a subsample (e.g., 100 firms spanning the size distribution).

The presentation (page 7) explicitly lists both options. The "Other?" in red on page 8 invites exploration but is not required for Week 2.

### 3. Minimum Patent Threshold (ADR-005 Required)

Fitting a GMM in 50D requires sufficient data. The parameter counts:

| Covariance Type | Params Per Component | Min Patents (rough rule: 10x params) |
|-----------------|---------------------|--------------------------------------|
| Full (50x50) | 1,325 + 50 mean + 1 weight = 1,376 | ~13,760 |
| Diagonal (50) | 50 + 50 mean + 1 weight = 101 | ~1,010 |

Most firms do NOT have enough patents for full covariance. Proposed tiering:

| Tier | Patent Count | Treatment |
|------|-------------|-----------|
| Exclude | <5 patents | Too few for any meaningful distribution. Exclude from downstream analysis. |
| Single Gaussian | 5-19 patents | K=1 (single component). Mean + diagonal covariance. No mixture. |
| GMM | 20+ patents | Full GMM fitting with K selection. |

**The EDA must quantify**: How many firms fall in each tier? What fraction of total patents do excluded firms represent? These numbers determine whether the thresholds are acceptable.

### 4. Covariance Type (ADR-006 Required)

**Start with diagonal covariance.** Rationale:
- UMAP output dimensions have weak inter-dimension correlation (UMAP optimizes for local structure, not global linear relationships)
- Diagonal covariance requires 50 parameters per component (vs 1,325 for full)
- More firms have enough patents to support diagonal fitting
- The Bhattacharyya Coefficient has a closed-form solution for both diagonal and full Gaussian components

**Consider full covariance for large firms** (>200 patents per component) where the data supports it. But this is a refinement, not a baseline requirement.

The EDA should check: after UMAP reduction, what are the pairwise correlations between the 50 dimensions? If they are negligible, diagonal is well-justified.

### 5. Pre-Normalization (ADR-007 Required)

This is the EDA's most important analysis. Week 1 found that embeddings are NOT L2-normalized (mean L2 norm ~6.8 in 768D space). After UMAP reduction to 50D, the scale characteristics may be different. The normalization choice affects GMM fitting.

**Three options to evaluate**:

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| Raw | Use 50D vectors as-is from UMAP | Preserves UMAP's learned structure | Scale imbalance across dimensions could bias GMM |
| L2-normalize | Normalize each 50D vector to unit length | Scale-invariant; consistent with cosine distance | Projects onto hypersphere; destroys magnitude information |
| Z-score standardize | Per-dimension mean=0, std=1 | Each dimension contributes equally | May distort UMAP's learned relative scales |

**Sensitivity check required**: Fit GMMs on a subsample (e.g., 50 mid-sized firms) under all three normalizations. Compare BIC scores, silhouette scores, and visual inspection of cluster assignments. The ADR documents the winner.

### 6. GMM Parameter Serialization

Consistent with Week 1's binary parquet checkpoint pattern:

**Schema for `output/portfolios/firm_gmm_parameters.parquet`**:

| Column | Type | Description |
|--------|------|-------------|
| gvkey | string | Firm identifier |
| n_components | int | Number of GMM components (K) |
| n_patents | int | Number of patents used to fit |
| covariance_type | string | "diagonal" or "full" |
| means | binary | Serialized numpy array, shape (K, 50) |
| covariances | binary | Serialized numpy array, shape (K, 50) for diagonal or (K, 50, 50) for full |
| weights | binary | Serialized numpy array, shape (K,) |
| converged | bool | Whether GMM fitting converged |
| bic | float | BIC score (if applicable) |

Week 3 loads this file to compute pairwise Bhattacharyya Coefficients.

### 7. Configuration Additions

Add a `portfolio` section to `src/config/config.yaml`:

```yaml
# --- Portfolio Construction ---
portfolio:
  min_patents: 5                # Firms with fewer patents are excluded
  single_gaussian_max: 19       # Firms with 5-19 patents get K=1 (single Gaussian)
  gmm_method: "bayesian"        # Options: bayesian, bic_sweep
  k_max: 10                     # Maximum components for GMM
  covariance_type: "diag"       # Options: diag, full
  normalization: "raw"          # Options: raw, l2, zscore (pending ADR-007)
  random_state: 42
  max_iter: 200
  n_init: 3                     # Number of initializations (best selected)
  checkpoint_every_n: 1000      # Save progress every N firms
```

---

## Data Characteristics You Must Know

- **Extreme firm-size skew**: ~50% of firms have <10 patents. Top firms have >100K patents. This skew is the central challenge — K selection, covariance type, and thresholds all depend on firm size.
- **Co-assigned patents**: ~85K patents appear in multiple firms' portfolios via the gvkey_map. Each firm's GMM should include all patents mapped to its gvkey (a co-assigned patent counts fully in both firms). Do NOT split weights.
- **post_deal_flag already filtered**: The gvkey_map and patent_vectors_50d only contain pre-deal patents. Do not re-filter.
- **Biotech caveats**: Citation coverage is lower for biotech firms (~84.9% unique-level vs higher for pure tech). Biotech M&A is often FDA-driven rather than technology-overlap-driven. Non-patent literature (NPL) — critical in biotech — is invisible to our citation embeddings. These limitations should be noted but do not require special handling in Week 2.

---

## Workflow with Halting Points

### Phase 0: Load and Verify Week 1 Outputs (no approval needed)
- Activate venv, verify dependencies
- Load `patent_vectors_50d.parquet` and `gvkey_map.parquet`
- Verify shapes: ~1.45M patent vectors at 50D, ~1.53M gvkey_map rows, 15,814 unique firms
- Verify join: all patent_ids in gvkey_map exist in patent_vectors_50d

### Phase 1: EDA → HALT for approval
- Create `notebooks/03_portfolio_eda.ipynb`
- **Required analyses:**
  1. **Firm-size distribution**: Histogram of patent counts per firm. Quantify tiers (<5, 5-19, 20-99, 100-999, 1000+). What fraction of firms and patents in each?
  2. **UMAP dimension scale analysis**: Per-dimension mean, std, min, max of the 50D vectors. Are dimensions on comparable scales? What is the inter-dimension correlation structure?
  3. **Normalization sensitivity check**: Fit GMMs on ~50 mid-sized firms (100-500 patents) under raw, L2-normalized, and z-score standardized vectors. Compare BIC, silhouette, and cluster stability across normalizations. This is the most important EDA analysis.
  4. **K sensitivity on subsample**: For 20 firms of varying sizes, fit with K=1..8. Plot BIC vs K. Does BIC have a clear elbow or is it monotonically decreasing?
  5. **Co-assignment statistics**: How many firms have co-assigned patents? Distribution of co-assignment counts.
- Present EDA summary to Torrin with key findings
- **Do NOT proceed to Phase 2 until EDA findings are reviewed**

### Phase 2: ADRs and Spec → HALT for Codex review
- Write ADRs 004-007 (K selection, minimum threshold, covariance type, normalization)
- Write `docs/specs/firm_portfolio_spec.md` with full interface contracts for PortfolioBuilder and GMMFitter
- **HALT**: Submit for Codex review. Do NOT proceed until approved.

### Phase 3: Implementation (TDD) → HALT for Codex review
- **For each module, follow the TDD cycle:**
  1. Write tests first (`tests/unit/test_<module>.py`)
  2. Implement minimum code to pass tests (`src/portfolio/<file>.py`)
  3. Run tests, fix issues
  4. Move to next module

- **Implementation order:**
  1. `src/portfolio/portfolio_builder.py` — firm grouping, threshold application, co-assignment handling
  2. `src/portfolio/gmm_fitter.py` — GMM fitting, parameter extraction, checkpoint-driven batch processing

- Run on a small sample first (~100 firms) to validate end-to-end before the full 15K-firm run
- Generate `output/portfolios/firm_gmm_parameters.parquet`
- **HALT**: Submit implementation for Codex review.

### Phase 4: Validation and Retrospective → HALT for approval
- Create `notebooks/04_portfolio_validation.ipynb`
- **Required visualizations and metrics:**
  1. **K distribution histogram**: How many firms got K=1, K=2, etc.? Is the distribution reasonable?
  2. **K vs firm size scatter**: Does K scale with patent count? Expected: positive correlation, but sub-linear.
  3. **Convergence rate**: What fraction of GMM fits converged? Any pathological firms?
  4. **Coverage statistics**: How many firms were excluded (<5 patents)? How many got single Gaussian? How many got full GMM? What fraction of total patents does each tier represent?
  5. **Sample firm inspection**: Pick 3-5 firms of different sizes. Show their K, weights, and a 2D PCA projection of their patent vectors colored by GMM cluster assignment. Do the clusters make visual sense?
  6. **BIC score distribution**: Histogram of BIC scores across firms. Any outliers?
- Write sprint retrospective: `docs/sprint_retrospectives/week2_instance_summary.md`
- **HALT**: Present validation evidence to Torrin.

---

## What NOT To Do

1. **Do not re-run Week 1** (patent vectorization is complete). Load the outputs.

2. **Do not implement Week 3** (Bhattacharyya Coefficient is next week). Resist the urge to compare firms.

3. **Do not optimize GMM fitting before having a working baseline.** Get K=1..K_max working with diagonal covariance first. Then consider Bayesian pruning, full covariance for large firms, etc.

4. **Do not use full covariance universally.** Most firms do not have enough patents. Full covariance on a 20-patent firm in 50D will be numerically singular. Start with diagonal.

5. **Do not skip the normalization sensitivity check.** This is a Codex-committed requirement. The choice between raw, L2, and z-score normalization must be backed by empirical evidence from the EDA.

6. **Do not assume all firms should get a GMM.** Firms with <5 patents cannot support even a single Gaussian meaningfully. The threshold must be justified by the ADR.

7. **Do not write a "framework."** Write concrete functions: `fit_firm_gmm(vectors, config) -> GMMResult`. If Week 3 needs different abstractions, that is Week 3's problem.

8. **Do not skip the Codex review checkpoints.** Development halts at each gate.

9. **Do not split co-assigned patent weights.** A co-assigned patent counts fully in each firm's portfolio. The gvkey_map already encodes this correctly.

10. **Do not commit large output files to git.** The `.gitignore` should already exclude `output/`.

---

## Open Questions for Team Discussion

These questions do not block Week 2 implementation but should be raised with the team:

1. **BIC sweep vs Bayesian GMM**: Does Arthur/Jan have a preference? The presentation lists both. The ADR should evaluate both but a team preference would inform the default.

2. **Co-assigned patent weighting**: Currently, a co-assigned patent counts fully in both firms' GMMs. Should it be down-weighted (e.g., 0.5 weight if shared by 2 firms)? Current recommendation: equal weight (simpler, defensible).

3. **Biotech separation**: Given the citation coverage gap and FDA-driven M&A dynamics in biotech, should biotech firms be flagged in the output or analyzed separately? Current recommendation: include them with a flag, let Week 3/4 decide.

4. **Rare technology handling**: If a firm's GMM has a component with very low weight (e.g., <1%), should it be preserved or pruned? Bayesian GMM auto-prunes; BIC sweep does not. This is relevant for the "rare technology acquisition" use case.

5. **Temporal considerations**: The 50D vectors include patents from 1976-2020 (pre-deal). Should older patents be down-weighted? Current recommendation: no (not in scope for Week 2; could be a Week 4 extension).

---

## Definition of Done

Week 2 is **complete** when ALL of the following are true:

- [ ] Week 1 outputs loaded and verified (shapes, counts, join integrity)
- [ ] EDA notebook documents firm-size distribution, dimension scale analysis, and normalization sensitivity
- [ ] ADR-004 (K selection) is written, reviewed, and approved
- [ ] ADR-005 (minimum patent threshold) is written, reviewed, and approved
- [ ] ADR-006 (covariance type) is written, reviewed, and approved
- [ ] ADR-007 (pre-normalization) is written, reviewed, and approved
- [ ] Firm portfolio spec is written, reviewed, and approved
- [ ] All unit tests pass (`pytest tests/ -v` green)
- [ ] Every firm with sufficient patents has a fitted GMM (or is explicitly excluded with a documented reason)
- [ ] `output/portfolios/firm_gmm_parameters.parquet` exists and is loadable
- [ ] Validation notebook shows: K distribution, K vs firm size, convergence rate, coverage stats, sample firm inspections
- [ ] Config updated with `portfolio` section
- [ ] Sprint retrospective is written with: What Was Built, Decisions Made, Confidence Levels, What Was NOT Done, Next Phase

---

## Getting Started

1. Read all Required Reading documents (in the order listed above)
2. Set up environment (Phase 0) — load and verify Week 1 outputs
3. Run EDA (Phase 1) — present findings, especially normalization sensitivity
4. Write ADRs 004-007 and spec (Phase 2) — submit for Codex review
5. Implement with TDD (Phase 3) — submit for Codex review
6. Validate and visualize (Phase 4) — present evidence
7. Write sprint retrospective

**Your immediate first action**: Read `CLAUDE.md`, then read all 10 files in `docs/values/`, then read `methodology.md` and the Week 1 retrospective. Only after reading these should you proceed to loading Week 1 outputs.
