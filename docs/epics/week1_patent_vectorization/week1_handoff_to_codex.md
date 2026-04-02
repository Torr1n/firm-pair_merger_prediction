# Week 1 Handoff: Development Instance → Codex

**From**: Claude Code (development instance)  
**To**: Codex (impartial reviewer)  
**Date**: 2026-04-01  
**Sprint**: Week 1 — Patent Vectorization Pipeline (Step 1 of 4)

---

## What I'm Asking You to Review

This is the first design + implementation review for the project. I'm submitting the complete Week 1 output: 2 ADRs, 1 interface spec, 5 implemented modules, 38 unit tests, 2 notebooks, and a validation pipeline. Everything is on `master` at commit `2ee3370`.

I'll walk you through what was built, what decisions were made and why, where I'm confident, and where I'd appreciate your scrutiny.

---

## What Was Built

### The Pipeline

The patent vectorization pipeline transforms 1.2M patents into 50-dimensional vectors through four stages:

```
patent_metadata.parquet → PatentEncoder (title+abstract) → 768D
cited_abstracts.parquet → PatentEncoder → CitationAggregator (mean pool) → 768D
                                                    ↓
                                              Concatenate → 1536D
                                                    ↓
                                              UMAPReducer → 50D
```

Each stage checkpoints to parquet. The pipeline was validated end-to-end on a 1K-patent sample (13.4 min on CPU).

### Artifact Inventory

| # | Artifact | Path | Purpose |
|---|----------|------|---------|
| 1 | EDA notebook | `notebooks/01_data_eda.ipynb` | 8 data quality questions, executed with outputs |
| 2 | ADR-001 | `docs/adr/adr_001_patent_embedding_model.md` | PatentSBERTa selection, token handling, text concatenation |
| 3 | ADR-002 | `docs/adr/adr_002_citation_aggregation.md` | Mean pooling, zero-citation handling, embed-once optimization |
| 4 | Interface spec | `docs/specs/patent_vectorizer_spec.md` | Contracts for all 5 modules |
| 5 | CheckpointManager | `src/utils/checkpointing.py` | Binary parquet save/load with metadata |
| 6 | PatentLoader | `src/data_loading/patent_loader.py` | Column-selective loading with validation |
| 7 | PatentEncoder | `src/embeddings/patent_encoder.py` | PatentSBERTa encoding with batching, checkpointing, GPU fallback |
| 8 | CitationAggregator | `src/embeddings/citation_aggregator.py` | Mean-pooled citation embeddings |
| 9 | UMAPReducer | `src/dimensionality_reduction/umap_reducer.py` | 1536D → 50D reduction |
| 10 | Config loader | `src/config/__init__.py` | YAML config loading utility |
| 11-15 | Unit tests | `tests/unit/test_*.py` | 38 tests across 5 test files |
| 16 | Pipeline script | `scripts/run_sample_pipeline.py` | End-to-end 1K sample validation |
| 17 | Validation notebook | `notebooks/02_embedding_validation.ipynb` | 3 visualizations with findings |
| 18 | Retrospective | `docs/sprint_retrospectives/week1_instance_summary.md` | Full sprint summary |

---

## Key Decisions and Their Evidence

### ADR-001: PatentSBERTa with Default 512-Token Truncation

**Decision**: Use `AI-Growth-Lab/PatentSBERTa` (768D), concatenate title+abstract as a single string, rely on default 512-token truncation.

**Evidence**: EDA tokenized a 50K sample. Median token count is 152, P99 is 322, only 0.03% exceed 512 tokens (~360 patents in the full 1.2M dataset). The cost of a more complex strategy (chunking, hierarchical encoding) is not justified.

**What I'd flag**: The bootstrap prompt suggested empirically comparing PatentSBERTa to a general-purpose model (all-mpnet-base-v2) via intra-firm clustering. I did not do this comparison. The rationale is that PatentSBERTa was trained specifically on patent title+abstract pairs, which is our exact input format, and domain-specific models consistently outperform general-purpose models in specialized registers. Torrin agreed this was sufficient. Your call on whether to require the comparison.

### ADR-002: Mean Pooling with Zero Vectors

**Decision**: Mean-pool citation embeddings per patent. Zero-citation patents (7.6%) get a 768D zero vector. Embed the 1.9M unique cited abstracts once, then look up per patent (~9x speedup).

**Evidence**: 97.08% edge-level citation coverage. Mean pooling is the standard aggregation in the embedding literature. The zero vector correctly encodes "no citation signal" — the title+abstract component still carries the content signal in the concatenated 1536D vector.

**Validation finding worth your attention**: Citation embedding L2 norms are bimodal. The upper peak (~6.75-6.9) corresponds to patents with 1-3 citations (the mean of few vectors retains near-individual norm). The lower peak (~5.0-5.8) corresponds to patents with 4+ citations (averaging many diverse vectors compresses the norm). Pearson r(citation_count, norm) = -0.317. This is a mathematical property of mean pooling, not a data quality issue. The norm effectively encodes citation coherence. We documented this but did not act on it — it may or may not matter for Week 2's GMM fitting. I'd value your opinion on whether pre-normalization should be considered.

### Non-Normalized Embeddings

**Finding**: PatentSBERTa does NOT L2-normalize its output. Title+abstract norms: mean=6.79, std=0.14. Citation norms (nonzero): mean=5.46, std=0.59.

**Why this is fine for now**: UMAP uses cosine metric, which is scale-invariant. The 50D output vectors are what Week 2 consumes for GMM fitting. However, if GMMs are sensitive to the scale difference between the title+abstract component (norm ~6.8) and the citation component (norm ~5.5) within the concatenated 1536D vector, normalization before concatenation may be needed. This is a Week 2 decision, but I'm flagging it now.

---

## Data Findings

The dataset differs significantly from the documentation (`data_prep.md`):

| Metric | Documented | Actual | Delta |
|--------|-----------|--------|-------|
| Patents | ~2.7M | 1,211,889 | -55% |
| Cited abstracts | ~3.7M | 1,872,555 | -49% |
| Citation edges | ~46M | 16,698,056 | -64% |

This is good news for compute costs but should be confirmed with Amie — is the documentation stale, or was the dataset filtered after documentation was written?

Additional findings:
- **Year range**: 1976-2023 (cumulative, not just 2020). The "2020 training window" refers to which firms are in the Compustat sample, not which patents.
- **Zero duplicates**: patent_id is a clean unique key.
- **241 null abstracts** (0.02%): All have valid titles. Handled by encoding title-only.
- **Firm size skew**: 1,892 firms, median 9 patents/firm, max 156,616 (likely IBM). 50.8% of firms have <10 patents. This will be a significant design consideration for Week 2's GMM fitting — you can't fit a meaningful mixture model on 3 patents.

---

## Test Coverage

38 tests across 5 modules:

| Module | Tests | What's Covered | What's NOT Covered |
|--------|-------|---------------|-------------------|
| CheckpointManager (7) | Round-trip save/load, metadata, single patent, length mismatch, existence checks | Comprehensive for the scope | Large file handling, concurrent writes |
| PatentLoader (9) | Default columns, selected columns, duplicate detection, null detection, missing files, row counts | All validation paths | Performance on full dataset |
| PatentEncoder (9) | Shape, dtype, determinism, empty input, batch override, null abstracts, checkpointing | Real model (not mocked) | Resume-from-checkpoint mid-encoding |
| CitationAggregator (8) | Mean pooling correctness, zero-citation, missing citations, all-missing, column validation, coverage stats | All edge cases hand-verified | Performance at scale |
| UMAPReducer (5) | Shape, dtype, reproducibility, too-few-samples | Core contract | Memory limits, quality metrics |

**Tests use the real PatentSBERTa model** (not mocked) for the encoder — this makes them slow (~9 min) but validates actual model behavior.

---

## Where I'd Appreciate Your Scrutiny

1. **The spec-implementation gap on PatentLoader default columns.** The original spec said default columns were `['patent_id', 'gvkey', 'title', 'abstract']`, but I implemented it as "load all available columns." I updated the spec to match the implementation. The rationale is that the `year` and `grant_date` columns we discovered in EDA are useful and shouldn't be excluded by a narrow default. But this does weaken the "memory safety" argument for column-selective reads. Is this the right call?

2. **The bimodal citation norm question.** Mean pooling produces norms that encode citation count information. Is this a feature (useful signal for GMMs) or a bug (confounding factor that should be normalized away)? I genuinely don't know the right answer here and would value a second opinion before Week 2.

3. **The `every_n` checkpoint logic in PatentEncoder.** The `checkpoint_every_n` parameter is stored but the current implementation only saves a final checkpoint, not intermediate ones during encoding. The `encode_texts` method processes everything in one `model.encode()` call. For full-scale runs (1.2M patents), we'd want intermediate checkpoints. This is a known gap — the resume-from-checkpoint logic works, but it requires a crash between encoding batches to have something to resume from, and right now there's only one batch. This needs to be addressed before the full-scale AWS run.

4. **CitationAggregator performance at scale.** The `aggregate()` method uses a Python loop over patent_ids with dict lookups. This is fine for 1K patents but will be slow for 1.2M. The inner loop does `np.mean(vectors, axis=0)` per patent, which is efficient, but the outer loop is O(n_patents * avg_citations). At scale, this may need vectorization or batching. I didn't optimize prematurely per our values, but flagging it for your awareness.

5. **The `__index_level_0__` silent exclusion.** PatentLoader silently excludes this pandas artifact column. If a user explicitly requests it, they get no error and no column — it's just missing. This is arguably the right behavior (the column is meaningless), but the silence could be surprising.

---

## What Was NOT Done

- **Full-scale execution** (1.2M patents): Validated on 1K sample only. Full run is planned for desktop/AWS.
- **PatentSBERTa vs general-model comparison**: Deferred; justified by domain-specific training.
- **UMAP hyperparameter tuning**: Baseline defaults only. Tuning requires full-scale embeddings.
- **Parametric UMAP fallback**: Not implemented. May be needed if 1.2M x 1536 exceeds memory.
- **Integration tests**: Only unit tests exist. No end-to-end test that loads real data through all modules.

---

## Recommendation

I believe this work is ready for approval with the understanding that items 3 and 4 in "Where I'd Appreciate Your Scrutiny" are known gaps that must be addressed before full-scale deployment. The architecture is sound, the tests validate correctness, the data is well-understood, and the validation evidence supports the design decisions.

Looking forward to your review.
