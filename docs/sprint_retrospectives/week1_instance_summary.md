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

Plus: EDA notebook (8 data quality questions answered), validation notebook with 3 visualizations (all checks passed), 2 ADRs, 1 interface spec, validation pipeline script.

**1K-sample pipeline validated end-to-end** on CPU desktop in 13.4 minutes, producing all 4 checkpoint parquet files.

## Decisions Made

| Decision | Rationale | Confidence |
|----------|-----------|------------|
| PatentSBERTa (not general SBERT) | Patent-domain fine-tuned; matches our input format | High |
| Default 512-token truncation | Only 0.03% of patents exceed limit (~360 out of 1.2M) | High — empirically validated |
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

## Validation Findings (1K Sample Pipeline)

- **L2 norms are NOT unit-normalized**: Title+abstract norms: mean=6.79, std=0.14. This is consistent and healthy (tight distribution, no degenerate vectors). Not a problem — UMAP uses cosine metric (scale-invariant), and downstream GMMs can normalize if needed.
- **Citation norms are bimodal**: Upper peak (~6.75-6.9) = patents with 1-3 citations (mean retains near-individual norm). Lower peak (~5.0-5.8) = patents with 4+ citations (mean pooling compresses norm as diverse vectors partially cancel). Pearson r(citation_count, norm) = -0.317. This is an expected mathematical property of mean pooling, not a data quality issue. The norm effectively encodes citation coherence.
- **2D UMAP shows meaningful structure**: Not random noise, not a single blob. Visible firm-level clustering tendencies (e.g., focused-portfolio firms cluster tighter than mega-firms like IBM which span the space). Exactly the behavior expected for downstream GMM portfolio fitting.
- **Pipeline coverage**: 100% at title+abstract, 92.5% nonzero at citation (matching 7.6% zero-citation rate from EDA), 100% at concatenation and UMAP. No patents lost.

## What Was NOT Done

- **Full-scale pipeline execution**: 1K-sample pipeline validated end-to-end (13.4 min on CPU). Full 1.2M-patent run requires more compute (desktop with proper GPU or AWS).
- **PatentSBERTa vs general-model comparison**: Bootstrap prompt suggested this; deferred as the domain-specific model is well-justified by literature.
- **UMAP hyperparameter tuning**: Baseline defaults only. Tuning deferred to after full-scale run produces real embeddings.
- **Codex review**: ADRs and spec written but not yet reviewed by Codex. Torrin approved proceeding to implementation before formal audit.

## Confidence Levels

| Component | Confidence | Notes |
|-----------|-----------|-------|
| Data quality | **High** | Thoroughly validated in EDA |
| Module correctness | **High** | 38 tests, all passing; 1K sample pipeline validated end-to-end |
| Embedding quality | **High** | L2 norms consistent, UMAP shows meaningful structure, coverage matches EDA |
| Full-scale feasibility | **Medium** | Untested at 1.2M scale; memory and I/O are open questions |
| UMAP quality at scale | **Medium** | 1.2M × 1536 is ~7GB input; may need subsampling fallback |

## Next Phase (Week 2: Firm Patent Portfolios)

- Aggregate 50D vectors per firm → fit GMMs
- Key open question: method for determining optimal K (number of clusters per firm)
- Firm size skew (50.8% have <10 patents) will require minimum patent threshold decision
- Need full-scale 50D vectors as input — requires completing full pipeline run first

## Operational Lessons

- **WSL-NTFS I/O is a bottleneck**: Model loading, pip installs, and any heavy file I/O is 10-100x slower than native Linux. Full-scale runs must happen on native Linux (desktop or AWS).
- **GPU compatibility requires probing**: The 1080 Ti (compute capability 6.1) is detected by PyTorch but not supported by the installed CUDA build (requires CC 7.5+). PatentEncoder now probes CUDA with a test allocation before using it, falling back to CPU gracefully. No manual intervention needed.
- **Checkpoint-driven design pays off**: Every stage produces a loadable parquet file, enabling resume-from-checkpoint and stage-by-stage validation.
