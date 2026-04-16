# Firm-Pair Merger Prediction in the Technology Sector

Can firms' patent portfolios be used as a predictor of M&A pairs in the Technology Sector?

## Quickstart for Teammates (Week 2 Handover)

You have been handed a validated patent-portfolio distance matrix over **7,485 firms** in the technology and biotech sectors, ready for economic hypothesis testing. You should have received an artifact bundle from Torrin (email attachment or file-share link, ~860 MB).

### 1. Clone the repo and install dependencies

```bash
git clone <repo-url>
cd firm-pair_merger_prediction
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place the artifact bundle files

After extracting Torrin's bundle (8 files plus `SHA256SUMS.txt`), place them as follows, from the repo root:

**Into `output/kmax_sweep/corrected/output/kmax_sweep/`** (the nested path is intentional — it's the S3-sync layout):

- `firm_gmm_parameters_k15.parquet`
- `bc_matrix_all_k15_dedup_linear.npz`
- `firm_gmm_parameters_k10.parquet`
- `bc_matrix_all_k10_dedup_linear.npz`

**Into `output/kmax_sweep/`:**

- `deduplication_decisions.csv`
- `excluded_firms.csv`
- `coassignment_audit.parquet`

**Into `notebooks/`** (if not already in your clone):

- `04_pipeline_output_overview.ipynb`

### 3. Verify transfer integrity

```bash
cd /path/to/extracted/bundle
sha256sum -c SHA256SUMS.txt
```

### 4. Run Notebook 04 (main walkthrough)

```bash
jupyter notebook notebooks/04_pipeline_output_overview.ipynb
```

Plan on 4-6 hours to work through it. The notebook covers: loading the artifacts, a worked two-firm BC example (with reproducibility assertion), finding top-k partners for any firm, distributional sanity plots, the co-assignment caveat (critical for regression design), and the caveats/roadmap table.

### Key Artifacts

| File | Description |
|---|---|
| `firm_gmm_parameters_k15.parquet` | **Primary** — per-firm Bayesian GMM at K_max=15 (production lock) |
| `bc_matrix_all_k15_dedup_linear.npz` | **Primary** — pairwise Bhattacharyya Coefficient matrix at K=15 |
| `firm_gmm_parameters_k10.parquet` | Convergence-floor reference (K_max=10) for robustness checks |
| `bc_matrix_all_k10_dedup_linear.npz` | Reference BC matrix at K=10 |
| `deduplication_decisions.csv` | Audit trail — 464 aliases/subsidiaries/predecessors removed |
| `excluded_firms.csv` | Firms excluded for <5 patents |
| `coassignment_audit.parquet` | Top-100 BC pair shared-patents audit (see Section 6 of Notebook 04) |

### Methodology at a Glance

- **`notebooks/03_kmax_convergence_analysis.ipynb`** — K_max convergence story (Spearman ρ=0.991-0.993, top-50 overlap 96-100%)
- **`docs/adr/adr_004_k_selection_method.md`** — K_max=15 production decision (locked 2026-04-14)
- **`docs/adr/adr_005_..._prior_global_empirical.md`**, **`adr_006_diagonal_covariance.md`**, **`adr_007_normalization.md`** — other methodology choices
- **`docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md`** — decision narrative
- **`docs/epics/week2_firm_portfolios/coassignment_audit_summary.md`** — aggregate co-assignment stats

### BC Formula Correctness Note

The BC formula uses **linear** mixing weights πᵢπⱼ (bounded in [0, 1]). An earlier iteration used `√(πᵢπⱼ)` which is mathematically an upper bound and can exceed 1.0 (observed up to 5.39), causing the original K_max=15→20 top-tail ranking instability we caught on 2026-04-12. The shipped matrix uses the corrected formula; Notebook 04 Section 3 asserts matrix-vs-formula agreement to float64 tolerance as a reproducibility anchor.

### Open Items (staged delivery, 1-4 weeks from 2026-04-15)

See Section 7 of Notebook 04 for the full caveats table. Headline items still in flight:

- **Gaussian adequacy audit** (Week 1-2) — checking whether the Bayesian GMM's Gaussian component assumption holds empirically on UMAP-reduced patent vectors
- **Pruning-threshold audit** (Week 1-2) — sensitivity of effective K to the DP weight-threshold choice
- **BC spec (Codex review in progress)** (Week 1) — 5 revisions pending before the spec is approved for implementation
- **BC and PortfolioBuilder module TDD** (Week 2-3) — extract production logic from `scripts/*.py` into tested modules under `src/comparison/` and `src/portfolio/`
- **Directional complementarity v2 dataset** (Week 2-4) — ADR-008 → spec → implementation for the asymmetric "gap-filling" metric

### Asking questions

Contact Torrin for any issues. For methodology questions, reference the ADRs. For open-item timelines or early-delivery requests, ping Torrin directly — order-of-operations may shift if something is blocking your regressions.

## Research Motivation

M&A is a driver of growth that allows larger firms with low R&D to acquire new technology. Building on Bena & Li (2014), this project redefines the "technological overlap" variable with one that has more economic interpretation. Instead of a static similarity score, we measure overlap in the probability distributions of entire patent portfolios — moving from "how similar are two patent portfolios?" to "in what way are patent portfolios interacting in the technology space?"

**Training window**: 2020 patent data. Crisis-period data surfaces latent value by suppressing market noise, yielding models that identify genuine fundamentals rather than momentum. Validated against the 2021 M&A "Springboard Effect" (352% YoY target deal value rebound, 257% acquiror deal value rebound).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Step 1: Vectorize Patents                 │
│  PatentSBERTa(title+abstract) ──┐                           │
│                                 ├── concat(1536D) ── UMAP ──│──► 50D vectors
│  PatentSBERTa(citations) ───────┘                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 2: Firm Patent Portfolios                  │
│  Aggregate 50D vectors per firm ── GMM(K clusters) ──────── │──► probability distributions
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Step 3: Compare Distributions                     │
│  Bhattacharyya Coefficient (overlap) ─────────────────────── │──► similarity metric
│  Directional Technology Complementarity (gap-filling) ────── │──► asymmetric metric
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Step 4: Extensions                        │
│  Forward: A+B ≈ C?  │  Inverse: find B s.t. A+x ≈ C       │
│  Synthetic portfolio matching for target identification      │
└─────────────────────────────────────────────────────────────┘
```

## Data

Three parquet files (~1.3GB total, gitignored):

| File | Description | Scale |
|------|-------------|-------|
| `patent_metadata.parquet` | Tech sector patents from Compustat sample | ~2.7M patents |
| `cited_abstracts.parquet` | Abstracts of cited patents | ~3.7M abstracts |
| `citation_network.parquet` | Citation edges | ~46M edges |

## Project Structure

```
firm-pair_merger_prediction/
├── CLAUDE.md                    # AI agent instructions
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore
├── data/                        # Parquet data files (gitignored)
├── docs/
│   ├── adr/                     # Architecture Decision Records
│   ├── specs/                   # Interface specifications
│   ├── epics/                   # Epic packages with sprint breakdowns
│   ├── diagrams/                # PlantUML diagrams
│   ├── sprint_retrospectives/   # Per-sprint instance summaries
│   ├── references/              # Academic papers, presentation
│   └── values/                  # Development values and principles
├── src/
│   ├── config/                  # YAML configuration
│   ├── data_loading/            # Parquet loading and validation
│   ├── embeddings/              # PatentSBERTa encoding
│   ├── dimensionality_reduction/# UMAP reduction
│   ├── portfolio/               # GMM fitting (Step 2)
│   ├── comparison/              # BC, complementarity (Step 3)
│   └── utils/                   # Logging, checkpointing
├── tests/
│   ├── unit/
│   └── integration/
├── notebooks/                   # EDA and visualization
└── output/                      # Pipeline outputs (gitignored)
```

## Team

- **Ananya Ravichandran** — Researcher
- **Arthur Khamkhosy** — Methodology design
- **Torrin Pataki** — Development lead
- **Amie Le Hoang** — Data preparation
- **Duncan Harrop** — Researcher
- **Jan Bena** — Faculty advisor (UBC Sauder)

## References

- Bena, J. and Li, K. (2014). Corporate Innovations and Mergers and Acquisitions. *The Journal of Finance*, 69: 1923-1960.
- Bhattacharyya, A. (1946). On a measure of divergence between two multinomial populations. *Sankhya*, 7(4), 401-406.
- Bloom, N., Schankerman, M. and Van Reenen, J. (2013). Identifying Technology Spillovers and Product Market Rivalry. *Econometrica*, 81: 1347-1393.
- Blei, D. M., & Jordan, M. I. (2006). Variational inference for Dirichlet process mixtures. *Bayesian Analysis*, 1(1), 121-143.
- Chen, L. (2017). Do patent citations indicate knowledge linkage? *Journal of Informetrics*, 11(1), 63-79.
- Choi, S., Lee, H., Park, E. L., & Choi, S. (2019). Deep patent landscaping model using transformer and graph embedding. arXiv:1903.05823.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *JRSS Series B*, 39(1), 1-22.
- Kailath, T. (1967). The Divergence and Bhattacharyya Distance Measures in Signal Selection. *IEEE Transactions on Communications*, 15, 52-60.
