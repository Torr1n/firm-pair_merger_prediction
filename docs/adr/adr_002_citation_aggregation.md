# ADR-002: Citation Embedding Aggregation Strategy

**Status**: Proposed  
**Date**: 2026-04-01  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (pending)

## Context

Each patent in our dataset cites zero or more other patents. The citation network captures "knowledge base linkage" (Chen 2017) — what prior art a patent builds upon. Our methodology (Choi et al. 2019) requires a single 768D "citation embedding" per patent that captures this linkage signal.

We need to decide:
1. How to aggregate multiple citation embeddings into one vector per patent
2. How to handle patents with zero citations
3. How to efficiently compute citation embeddings at scale

### Data Characteristics (from EDA)

| Metric | Value |
|--------|-------|
| Total patents | 1,211,889 |
| Zero-citation patents | 92,139 (7.60%) |
| Median citations per patent | 6 |
| Mean citations per patent | 13.8 |
| P95 citations per patent | 45 |
| Max citations per patent | 4,468 |
| Unique cited abstracts | 1,872,555 |
| Citation edge coverage (abstracts available) | 97.08% |
| Total citation edges | 16,698,056 |

## Decision

### Aggregation Method: Mean Pooling

For each patent, embed all of its cited patent abstracts using PatentSBERTa, then compute the element-wise mean of the resulting 768D vectors.

```
citation_embedding(patent_i) = (1/N) * Σ_{j=1}^{N} PatentSBERTa(cited_abstract_j)
```

where N is the number of citations with available abstracts.

**Why mean pooling?**

- **Simplest defensible option.** Mean pooling is the standard aggregation in the NLP embedding literature for combining multiple vectors into a single representation.
- **Preserves average direction.** The mean vector captures the "center of mass" of a patent's knowledge base — the average technological direction it draws from. This is exactly what matters for downstream GMM clustering (we're modeling portfolio *distributions*, not individual citations).
- **Deterministic.** No hyperparameters, no training, no randomness.
- **Choi et al. (2019) precedent.** The methodology paper we build on used aggregated citation embeddings.

### Zero-Citation Handling: 768D Zero Vector

Patents with no citations receive a 768D zero vector for their citation component.

**Why zero vector (not mean of all citations, random vector, or exclusion)?**

- **Semantic correctness.** A patent with no citations has *no knowledge base linkage signal*. The zero vector correctly encodes "absence of signal" rather than fabricating one.
- **Safe for concatenation.** When concatenated with the 768D title+abstract embedding to form the 1536D Patent Vector, the zero citation component means the patent's representation depends entirely on its own content — which is the correct behavior.
- **No information loss.** The patent is still in the dataset. Its title+abstract embedding carries the content signal. Only the citation channel is silent.
- **7.6% affected.** This is a significant but manageable fraction. Excluding them would lose ~92K patents and their firms' portfolio information. Imputing would introduce noise.

**Configuration**: `config.yaml` already specifies `citation_aggregation.zero_citation_strategy: "zero_vector"`.

### Computation Strategy: Embed Unique Cited Abstracts Once, Then Look Up

The naive approach (embed each citation edge independently) would require ~16.7M embedding operations. Instead:

1. **Embed all 1,872,555 unique cited abstracts once** → lookup table of `{cited_patent_id: 768D vector}`
2. **For each patent**, use the citation network to find its cited patent IDs
3. **Look up** pre-computed citation embeddings from the table
4. **Mean-pool** the looked-up vectors

This reduces embedding operations from 16.7M to 1.9M — a **~9x speedup**.

Citations whose `citation_id` has no matching abstract in `cited_abstracts.parquet` (9.86% of unique IDs, but only 2.92% of edges) are simply skipped. The mean is computed over available citations only.

**Edge case**: If a patent has citations but *none* of them have available abstracts, it is treated as zero-citation (receives zero vector). This is expected to be extremely rare given 97% edge coverage.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| **Weighted mean (by recency)** | Adds complexity; requires citation year data that we haven't validated; marginal benefit for portfolio-level analysis |
| **Concatenate all citation text, embed once** | 512-token BERT limit means most citation text is truncated. A patent with 45 citations (P95) would lose >99% of citation content |
| **Max pooling** | Takes element-wise max; uncommon in literature; harder to interpret; loses the "average direction" semantic |
| **Attention-based aggregation** | Requires training an attention layer; violates parsimony; unnecessary for distributional comparison |
| **Exclude zero-citation patents** | Loses 92K patents (7.6%); biases dataset toward older, more-cited patents |
| **Impute with firm mean** | Circular — we don't have firm embeddings yet; introduces noise |

## Consequences

- Every patent gets exactly one 768D citation embedding (either mean-pooled or zero vector)
- 92,139 patents (7.6%) will have zero vectors for the citation component
- Citation embeddings use the same PatentSBERTa model as title+abstract embeddings (consistency)
- The 1,872,555 unique cited abstracts are embedded once and cached as a checkpoint
- 2.92% of citation edges are skipped due to missing abstracts (97.08% coverage is sufficient)
- No hyperparameters to tune in the aggregation step

## Validation

- Unit tests will verify: mean pooling correctness, zero-citation handling, shape (768D), edge coverage statistics
- Phase 4 validation will compare L2 norm distributions of citation vs. title+abstract embeddings
- Coverage bar chart will show patent counts at each pipeline stage
