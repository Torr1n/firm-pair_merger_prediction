# Firm-Pair Merger Prediction in the Technology Sector

Can firms' patent portfolios be used as a predictor of M&A pairs in the Technology Sector?

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
