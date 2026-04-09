# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Firm-Pair Merger Prediction in the Technology Sector**

Research question: *Can firms' patent portfolios be used as a predictor of M&A pairs in the Technology Sector?*

Building on Bena & Li (2014), this project redefines "technological overlap" from a static similarity score to a probabilistic measure of overlap in entire patent portfolio distributions. We move from "how similar are two patent portfolios?" to "in what way are patent portfolios interacting in the technology space?"

**Team**: Ananya Ravichandran, Arthur Khamkhosy, Torrin Pataki (development lead), Amie Le Hoang, Duncan Harrop
**Advisor**: Jan Bena (UBC Sauder)
**Training window**: 2020 patent data (crisis data surfaces latent value; validated against 2021 M&A "Springboard Effect")

## Architecture

The pipeline follows a four-step methodology:

1. **Vectorize Patents**: PatentSBERTa embeddings of title+abstract (768D) and citations (768D), concatenated (1536D), reduced via UMAP to 50D
2. **Firm Patent Portfolios**: Aggregate 50D patent vectors per firm into Gaussian Mixture Models (GMMs) with K clusters
3. **Compare Distributions**: Bhattacharyya Coefficient (overlap/similarity) + Directional Technology Complementarity (asymmetric gap-filling)
4. **Extensions**: Synthetic portfolio matching — forward problem (A+B≈C?), inverse problem (find B such that A+x≈C)

Key architectural patterns:
- **Checkpoint-driven**: Save intermediate results at every pipeline stage as parquet
- **Config-driven**: All hyperparameters exposed in `src/config/config.yaml`
- **Spec-driven development**: Interface specs define contracts before implementation begins
- **ADR-documented**: Every non-obvious architectural decision gets an ADR

## Development Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
pytest tests/ -v

# Run specific test module
pytest tests/unit/test_patent_encoder.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Key Configuration

Configuration lives in `src/config/config.yaml`. Important settings:
- `embedding.model_name`: PatentSBERTa model (default: AI-Growth-Lab/PatentSBERTa)
- `embedding.batch_size`: Encoding batch size (256 GPU, 64 CPU)
- `umap.n_components`: Target dimensionality (default: 50)
- `citation_aggregation.method`: How to aggregate citation embeddings (default: mean_pooling)
- `portfolio.gmm_method`: K selection method — "bayesian" (Dirichlet process) or "bic_sweep"
- `portfolio.k_max`: Maximum GMM components per firm (default: 10)
- `portfolio.min_patents`: Minimum patents to include firm (default: 5)
- `portfolio.covariance_type`: "diag" (diagonal covariance, per ADR-006)

## Module Structure (src/)

- `config/` — YAML configuration management
- `data_loading/` — Parquet loading, validation, and data cleaning
- `embeddings/` — PatentSBERTa encoding, citation aggregation
- `dimensionality_reduction/` — UMAP reduction pipeline
- `portfolio/` — GMM fitting, firm portfolio representation (Step 2)
- `comparison/` — Bhattacharyya Coefficient, complementarity metrics (Step 3)
- `utils/` — Logging, checkpointing, batching helpers

## Design Principles

This project inherits its development culture from the Financial Topic Modeling project. The full articulation of each principle lives in `docs/values/`. The non-negotiable standards:

> "The best engineers write code my mom could read. They choose boring technology, they over-document the 'why,' and under-engineer the 'how.' Complexity is not a flex; it becomes a liability."

1. **Simplicity over complexity** — boring technology, readable code, under-engineer the "how"
2. **Over-document the "why"** — ADRs, comments on rationale, not mechanics
3. **Test-driven development** — tests before implementation, no exceptions
4. **Spec-driven development** — specs as contracts, ADRs before code
5. **Incremental validation** — small steps, validate at each stage
6. **Checkpoint everything** — save intermediate results, fault tolerance through resumability
7. **Evidence-based engineering** — claims backed by artifacts, not assumptions
8. **Operational discipline** — approval gates, halting points, tuning matrices
9. **Institutional memory** — retrospectives, handovers, memory files preserve lessons
10. **Parsimony** — minimum complexity for the current task

## Team Roles

### Development Lead (Claude Code / Torrin)
Primary implementation agent. Follows spec-driven, test-driven workflow. Halts at designated approval gates.

### Codex: Impartial Reviewer, Pair-Programmer, and Deployment Lead
- **Impartial Code Review**: Reviews all sprint outputs for spec conformance, test coverage, value adherence, and correctness. Can reject work that violates principles.
- **Pair Programming**: Available for architectural consultation. Provides second opinions on design decisions.
- **Deployment Lead**: Owns cloud deployment execution, E2E test running, and production validation.
- **CI/CD Auditing**: Controls deployment pipeline, ensures pre-flight checks and rollback procedures.

**Codex Review Checkpoints** (development HALTS until Codex approves):
- After ADRs are written (before implementation begins)
- After specs are written (before implementation begins)
- After each sprint's implementation is complete (before next sprint)
- Before any cloud deployment
- When making decisions not covered by existing ADRs

## Data Files (v3)

Four parquet files in `data/` (gitignored — contact Amie Le Hoang for access). v3 data (2026-04-07): tech + biotech, clean scope, post_deal_flag, pre-deduplicated encoding file.

| File | Description | Rows | Key Columns |
|------|-------------|------|-------------|
| `firm_patents_text_metadata_techbio_v3.parquet` | Full patent metadata with co-assignments | 1,604,583 | gvkey, patent_id, post_deal_flag, title, abstract |
| `firm_patents_dedup_techbio_v3.parquet` | Deduplicated for encoding (unique patent_ids) | 1,519,401 | patent_id, title, abstract |
| `cited_abstracts_techbio_v3.parquet` | Abstracts of every patent cited by core sample | 2,623,183 | patent_id (cited ID), abstract |
| `citation_network_techbio_v3.parquet` | Citation edges linking the two text datasets | 35,424,315 | patent_id (firm's patent), citation_id |

**Two-file pattern**: Use dedup file for encoding (unique patents), full file for portfolio construction (co-assignments preserved). Filter `post_deal_flag == 0` for clean pre-acquisition features. 15,814 unique firms.

## Environment Constraint

The WSL local environment has NO heavy ML dependencies (no torch, no pandas, no scikit-learn). Always use the project-specific virtual environment:

```bash
cd /path/to/firm-pair_merger_prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Do NOT pip install into the system Python or any other project's venv.
