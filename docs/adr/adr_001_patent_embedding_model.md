# ADR-001: Patent Embedding Model Selection and Text Handling

**Status**: Accepted  
**Date**: 2026-04-01  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (approved 2026-04-02)

## Context

Step 1 of our pipeline requires encoding patent text (title + abstract) into dense vector representations. The choice of embedding model and text handling strategy directly determines the quality of all downstream analysis — GMM portfolio fitting, Bhattacharyya coefficients, and complementarity metrics are only as good as the input vectors.

We need to decide:
1. Which embedding model to use
2. How to handle the model's token limit
3. How to combine title and abstract text

## Decision

### Model: AI-Growth-Lab/PatentSBERTa

We use `AI-Growth-Lab/PatentSBERTa`, a BERT-base model fine-tuned on patent text (PatentMatch dataset). It produces 768-dimensional embeddings.

**Why not a general-purpose model (e.g., all-mpnet-base-v2)?**

Patent language is a specialized register — legal terminology, technical jargon, and a formal structure unlike natural language. A model fine-tuned on patent text will produce representations where semantic similarity in patent space is better preserved. PatentSBERTa was trained specifically on patent title+abstract pairs, which is exactly our input format.

**Why not a larger model (e.g., patent-specific LLM embeddings)?**

Parsimony. BERT-base (110M parameters) is well-understood, fast to run, and produces embeddings that have been validated in patent similarity tasks. A larger model would increase compute cost without a demonstrated benefit for our use case (portfolio-level distributional comparison, not fine-grained patent matching).

### Token Limit: 512 tokens — no special handling needed

BERT has a hard limit of 512 tokens. Our EDA (Phase 1) measured the token count distribution on a 50K patent sample:

| Statistic | Value |
|-----------|-------|
| Mean | 154 tokens |
| Median | 152 tokens |
| P95 | 247 tokens |
| P99 | 322 tokens |
| Fraction > 512 | **0.03%** (15 out of 50,000 sample; ~360 estimated in full 1.2M dataset) |

**Decision**: Use default truncation (PatentSBERTa/sentence-transformers truncates at 512 tokens automatically). The ~360 patents (0.03%) that exceed this limit lose some trailing abstract text, which is acceptable given:
- The title (which carries the core invention description) is always preserved
- The first ~500 tokens of the abstract contain the most information-dense content (background, summary of invention)
- 99.97% of patents are fully captured
- The cost of a more complex strategy (chunking, separate embeddings, hierarchical encoding) is not justified for 0.03% of patents

### Text Input: Title + Abstract concatenated as single string

Format: `"{title} {abstract}"` — a single space-separated string passed to `model.encode()`.

**Why concatenate rather than embed separately?**

- PatentSBERTa was trained on title+abstract pairs as single inputs
- Concatenation produces one 768D vector per patent (simpler downstream pipeline)
- The model's attention mechanism can capture cross-references between title and abstract
- Separate embeddings would require a secondary aggregation step with no demonstrated benefit

**Null abstract handling**: 241 patents (0.02%) have null abstracts but valid titles. For these, the input is just the title string. This is acceptable — the title alone carries meaningful semantic signal.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| all-mpnet-base-v2 (general SBERT) | Not fine-tuned on patent text; worse patent semantic similarity |
| SciBERT | Trained on scientific papers, not patents specifically |
| Chunking for long patents | 0.03% affected — complexity not justified |
| Separate title and abstract embeddings | Requires secondary aggregation; model was trained on concatenated input |
| Larger context window models (Longformer) | Unnecessary — 99.97% fit in 512 tokens |

## Consequences

- Every patent in `patent_metadata.parquet` produces exactly one 768D vector
- No patents are excluded at this stage (all 1,211,889 are embeddable)
- The ~360 patents (0.03%) exceeding 512 tokens lose trailing abstract text (acceptable)
- PatentSBERTa does NOT L2-normalize output vectors (see Validation below)
- Downstream citation embeddings (ADR-002) use the same model for consistency
- Embedding is deterministic given the same model weights and input text

## Validation

- EDA notebook `notebooks/01_data_eda.ipynb` provides the token distribution evidence
- Unit tests verify output shape (768D), determinism, and null handling (38 tests, all passing)
- **L2 norm finding**: PatentSBERTa does NOT L2-normalize embeddings. Title+abstract L2 norms: mean=6.79, std=0.14, range 6.0-7.1. This is consistent and healthy (tight distribution, no degenerate vectors). Not an issue for downstream analysis — UMAP uses cosine metric, which is scale-invariant. However, any downstream consumer that assumes unit-normalized embeddings must normalize explicitly.
- 2D UMAP projection shows meaningful semantic structure with visible firm-level clustering tendencies
