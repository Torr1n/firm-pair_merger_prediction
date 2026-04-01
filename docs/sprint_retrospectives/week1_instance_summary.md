# Week 1 Sprint Retrospective: Patent Vectorization Pipeline

**Sprint**: Week 1 — Patent Vectorization (Step 1 of 4)  
**Date**: 2026-04-01  
**Team**: Torrin Pataki (dev lead), Claude Code (implementation)

---

## What Was Built

A complete patent vectorization pipeline with 5 modules, 38 passing unit tests, and checkpoint-driven architecture:

| Module | Purpose | Tests |
|--------|---------|-------|
| `CheckpointManager` | Binary parquet save/load with metadata | 7 |
| `PatentLoader` | Column-selective parquet loading with validation | 9 |
| `PatentEncoder` | PatentSBERTa encoding with batching and checkpointing | 9 |
| `CitationAggregator` | Mean-pooled citation embeddings, zero-citation handling | 8 |
| `UMAPReducer` | 1536D → 50D dimensionality reduction | 5 |

Plus: EDA notebook (8 data quality questions answered), 2 ADRs, 1 interface spec, validation pipeline script, validation notebook.

## Decisions Made

| Decision | Rationale | Confidence |
|----------|-----------|------------|
| PatentSBERTa (not general SBERT) | Patent-domain fine-tuned; matches our input format | High |
| Default 512-token truncation | Only 0.03% of patents exceed limit | High — empirically validated |
| Mean pooling for citations | Simplest, standard in literature, deterministic | High |
| Zero vector for zero-citation patents | Semantically correct (absence of signal); 7.6% affected | High |
| Embed unique cited abstracts once | 9x speedup over naive per-edge embedding | High |
| Binary parquet storage | Not 768 float columns; compact, fast to load | High |

## Key Data Findings

- **1,211,889 patents** across 1,892 firms (1976-2023, cumulative)
- **Data is clean**: 0 duplicate patent_ids, 0.02% null abstracts, 0 patents with both title and abstract missing
- **Token distribution**: median 152, only 0.03% exceed 512 — truncation is a non-issue
- **Citation coverage**: 97% edge-level — excellent for mean pooling
- **Firm size skew**: 50.8% of firms have <10 patents; top firm has 156K — GMM fitting (Week 2) will need minimum patent threshold
- **Dataset smaller than documented**: ~1.2M patents (not 2.7M), 1.9M cited (not 3.7M), 16.7M edges (not 46M)

## What Was NOT Done

- **Full-scale pipeline execution**: Unit tests validate correctness on small samples. Full 1.2M-patent run requires GPU/more compute (desktop or AWS).
- **PatentSBERTa vs general-model comparison**: Bootstrap prompt suggested this; deferred as the domain-specific model is well-justified by literature.
- **UMAP hyperparameter tuning**: Baseline defaults only. Tuning deferred to after full-scale run produces real embeddings.
- **Codex review**: ADRs and spec written but not yet reviewed by Codex. Torrin approved proceeding to implementation before formal audit.

## Confidence Levels

| Component | Confidence | Notes |
|-----------|-----------|-------|
| Data quality | **High** | Thoroughly validated in EDA |
| Module correctness | **High** | 38 tests, all passing |
| Full-scale feasibility | **Medium** | Untested at 1.2M scale; memory and I/O are open questions |
| UMAP quality at scale | **Medium** | 1.2M × 1536 is ~7GB input; may need subsampling fallback |

## Next Phase (Week 2: Firm Patent Portfolios)

- Aggregate 50D vectors per firm → fit GMMs
- Key open question: method for determining optimal K (number of clusters per firm)
- Firm size skew (50.8% have <10 patents) will require minimum patent threshold decision
- Need full-scale 50D vectors as input — requires completing full pipeline run first

## Operational Lessons

- **WSL-NTFS I/O is a bottleneck**: Model loading, pip installs, and any heavy file I/O is 10-100x slower than native Linux. Full-scale runs must happen on native Linux (desktop or AWS).
- **Checkpoint-driven design pays off**: Every stage produces a loadable parquet file, enabling resume-from-checkpoint and stage-by-stage validation.
