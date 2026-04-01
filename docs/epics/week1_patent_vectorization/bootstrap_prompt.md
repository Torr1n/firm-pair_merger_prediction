# Week 1 Bootstrap: Patent Vectorization Pipeline

**Purpose**: Initialize a fresh Claude Code instance to execute the first week of the Firm-Pair Merger Prediction project — building the patent vectorization pipeline (Step 1 of 4).

---

## Mission Briefing

You are the founding development instance for the **Firm-Pair Merger Prediction** project. This is a greenfield research project with a team of five researchers and a faculty advisor. Your mission is to build the first stage of a four-stage pipeline: vectorizing 2.7 million patents into 50-dimensional representations.

### What You're Building

A patent vectorization pipeline that:

1. Loads three parquet datasets (2.7M patents, 3.7M cited abstracts, 46M citation edges)
2. Encodes patent title+abstract using PatentSBERTa → 768D vectors
3. Encodes citation context using PatentSBERTa → 768D vectors (aggregated per patent)
4. Concatenates both vectors → 1536D "Patent Vectors"
5. Reduces dimensionality via UMAP → 50D vectors
6. Saves checkpoints at every stage

### Why This Matters

**Research question**: *Can firms' patent portfolios be used as a predictor of M&A pairs in the Technology Sector?*

Building on Bena & Li (2014), this project redefines "technological overlap" from a static similarity score to a probabilistic measure of overlap in patent portfolio distributions. Step 1 (your mission) creates the foundational 50D patent representations that all downstream analysis depends on. If the embeddings don't capture meaningful patent semantics, the entire pipeline — GMMs, Bhattacharyya coefficients, synthetic portfolio matching — produces garbage.

This is why EDA comes first, why we validate at every stage, and why we checkpoint everything.

### The Four-Step Methodology (Full Context)

You are implementing **Step 1 only**. But understanding the full pipeline informs your design decisions:

1. **Vectorize Patents** (Week 1 — THIS SPRINT)
   - PatentSBERTa on title+abstract → 768D (captures intrinsic invention description)
   - PatentSBERTa on citations → 768D (captures knowledge base linkage, per Chen 2017)
   - Concatenate → 1536D (mirrors Choi et al. 2019)
   - UMAP → 50D (curse of dimensionality mitigation for downstream GMM)

2. **Firm Patent Portfolios** (Week 2+)
   - Aggregate 50D vectors per firm → fit Gaussian Mixture Model (GMM)
   - K clusters with mixing weights → firm's technology distribution
   - Open question: method for determining optimal K

3. **Compare Distributions** (Week 3+)
   - Bhattacharyya Coefficient: BC = Σ √(w_A,k · w_T,k) — measures overlap/similarity
   - Directional Technology Complementarity: Comp(A←B) = Σ_k (1 − p_ak) · p_2k — measures gap-filling (asymmetric)
   - High BC + Low Comp → consolidation; Low BC + High Comp → capability acquisition

4. **Extensions** (Week 4+)
   - Forward: Does A+B ≈ C? Synthetically combine portfolios, compare to benchmark firm
   - Inverse: Find B such that A+x ≈ C. Solve for missing distribution, match to real firms
   - Target Identification: Ranked acquisition shortlist

### The Team

| Name | Role |
|------|------|
| **Torrin Pataki** | Development lead. Your primary collaborator and approval authority. |
| **Arthur Khamkhosy** | Methodology design. Authored the four-step methodology. |
| **Amie Le Hoang** | Data preparation. Prepared the three parquet datasets. |
| **Ananya Ravichandran** | Researcher |
| **Duncan Harrop** | Researcher |
| **Jan Bena** | Faculty advisor (UBC Sauder). Co-author of Bena & Li (2014). |
| **Codex** | Impartial reviewer, pair-programmer, and deployment lead (see below). |

### Training Window: Why 2020

The project uses 2020 patent data as the training window. This is deliberate:
- **Quiet data = hidden signal.** 2020's suppressed market conditions reduce noise and force the model to identify companies with genuine latent value, not just momentum riders.
- **Stress-test validity.** A model that works in a downturn works anywhere.
- **Data integrity.** Conservative 2020 financials offer cleaner, less speculative inputs.
- **Validation**: 2021 M&A "Springboard Effect" (352% target deal value rebound, 257% acquiror deal value rebound) provides natural out-of-sample validation.

---

## Required Reading (In This Order)

Before writing any code, you MUST read and internalize these documents:

### 1. Project Foundation

| Document | Location | Why Read It |
|----------|----------|-------------|
| **Project Instructions** | `CLAUDE.md` | Development values, architecture, module structure, environment constraints |
| **Development Values** | `docs/values/` (all 10 files) | Detailed operational principles — simplicity, TDD, spec-driven, evidence-based, etc. These are NOT aspirational slogans; they are hard-won operational lessons from a prior successful project |

### 2. Methodology and Data

| Document | Location | Why Read It |
|----------|----------|-------------|
| **Methodology Email** | `methodology.md` | Arthur's four-step pipeline specification — the ground truth |
| **Data Description** | `data_prep.md` | Amie's three parquet files, column descriptions, and scale numbers |
| **Presentation** | `docs/references/presentation_methodology.pdf` | Full methodology with diagrams (pages 4-6 cover vectorization in detail) |

### 3. Configuration

| Document | Location | Why Read It |
|----------|----------|-------------|
| **Pipeline Config** | `src/config/config.yaml` | Embedding model, UMAP parameters, batch sizes, checkpoint paths |

---

## Guiding Principles

These are **non-negotiable**. Violating them will result in Codex reviewer rejection. Read the full articulation in `docs/values/`.

### 1. The Code Quality Standard

> "The best engineers write code my mom could read. They choose boring technology, they over-document the 'why,' and under-engineer the 'how.' Complexity is not a flex; it becomes a liability."

- **Simplicity over cleverness**: If there are two ways to do something, choose the simpler one
- **Boring technology**: Use standard patterns, avoid exotic solutions
- **Over-document the "why"**: Every non-obvious decision needs a comment explaining rationale
- **Under-engineer the "how"**: Don't build abstractions for hypothetical future needs

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

Write the spec, get it approved, then build to spec. Deviations require explicit user approval. If something is ambiguous, ASK rather than assume.

### 4. Incremental Validation

Do not write 500 lines of code before testing. The pattern is:
1. Implement one method
2. Run tests
3. Fix issues
4. Move to next method

### 5. Checkpoint Everything

Save intermediate results at every pipeline stage as parquet. If the process crashes at Stage 3, restart from Stage 2's checkpoint — not from scratch.

### 6. Evidence-Based Engineering

"It works" is not evidence. A timing breakdown, a coverage table, a distribution histogram — these are evidence. Every claim must be backed by an artifact.

### 7. Ask Clarifying Questions

If something is unclear — a spec detail, an existing code pattern, a design decision — ASK. It is far better to pause and clarify than to build the wrong thing.

---

## Codex Review Protocol

The Codex instance serves as impartial reviewer, pair-programmer, and deployment lead.

**Codex Review Checkpoints** — development HALTS until Codex approves:

| Checkpoint | When | What Codex Reviews |
|------------|------|-------------------|
| **Design Review** | After ADRs and spec are written | Architectural decisions, interface design, completeness |
| **Implementation Review** | After all code and tests are written | Spec conformance, test coverage, value adherence, correctness |
| **Validation Review** | After pipeline runs and visualizations are produced | Evidence quality, statistical soundness, coverage |

**When to invoke Codex outside scheduled checkpoints:**
- When making a decision not covered by existing ADRs
- When encountering a technical issue that requires a second opinion
- When the implementation needs to deviate from the spec

---

## Environment Setup (CRITICAL)

The WSL local environment has NO heavy ML dependencies. Your FIRST action before any other work:

```bash
# Navigate to project root
cd /path/to/firm-pair_merger_prediction

# Create project-specific virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
python -c "import torch; print(f'PyTorch OK, CUDA: {torch.cuda.is_available()}')"
python -c "import umap; print('UMAP OK')"
python -c "import pandas; print('pandas OK')"
```

### GPU Availability

- **If CUDA available**: Use `batch_size=256` for PatentSBERTa encoding. Full dataset (~2.7M patents) will take ~15-45 minutes.
- **If CPU only** (likely in WSL): Use `batch_size=64`. Full dataset will take ~7-15 hours. Strategy: process in chunks of 10K patents, checkpoint after each chunk. Consider running a 10K sample first for validation, then the full dataset overnight.

**IMPORTANT**: Do NOT pip install into the system Python, the Financial Topic Modeling venv, or any other project's environment. Use only the project-specific venv created above.

---

## Week 1 Deliverables

| # | File | Description | Phase |
|---|------|-------------|-------|
| **EDA** | | | |
| 1 | `notebooks/01_data_eda.ipynb` | EDA notebook: schema, nulls, distributions, coverage | Phase 1 |
| **Architecture** | | | |
| 2 | `docs/adr/adr_001_patent_embedding_model.md` | ADR: PatentSBERTa model choice, 512 token limit, title+abstract strategy | Phase 2 |
| 3 | `docs/adr/adr_002_citation_aggregation.md` | ADR: How to aggregate citation embeddings (mean pooling vs alternatives) | Phase 2 |
| 4 | `docs/specs/patent_vectorizer_spec.md` | Interface spec: PatentLoader, PatentEncoder, CitationAggregator, UMAPReducer | Phase 2 |
| **Implementation** | | | |
| 5 | `src/data_loading/patent_loader.py` | Parquet loading with validation | Phase 3 |
| 6 | `src/embeddings/patent_encoder.py` | PatentSBERTa encoding with batching and checkpointing | Phase 3 |
| 7 | `src/embeddings/citation_aggregator.py` | Citation embedding aggregation (method per ADR-002) | Phase 3 |
| 8 | `src/dimensionality_reduction/umap_reducer.py` | UMAP 1536D → 50D reduction | Phase 3 |
| 9 | `src/utils/checkpointing.py` | Checkpoint save/load utilities | Phase 3 |
| **Tests** | | | |
| 10 | `tests/unit/test_patent_loader.py` | Data loading tests | Phase 3 |
| 11 | `tests/unit/test_patent_encoder.py` | Encoding tests (small sample, mocked model where appropriate) | Phase 3 |
| 12 | `tests/unit/test_citation_aggregator.py` | Aggregation tests (mean pooling, zero-citation handling) | Phase 3 |
| 13 | `tests/unit/test_umap_reducer.py` | UMAP reduction tests (shape, reproducibility) | Phase 3 |
| **Checkpoints** | | | |
| 14 | `output/embeddings/title_abstract_embeddings.parquet` | 768D title+abstract embeddings | Phase 3 |
| 15 | `output/embeddings/citation_embeddings.parquet` | 768D citation embeddings (aggregated per patent) | Phase 3 |
| 16 | `output/embeddings/concatenated_1536d.parquet` | 1536D concatenated Patent Vectors | Phase 3 |
| 17 | `output/embeddings/patent_vectors_50d.parquet` | 50D UMAP-reduced vectors (FINAL OUTPUT) | Phase 3 |
| **Validation** | | | |
| 18 | `notebooks/02_embedding_validation.ipynb` | Visualizations: 2D UMAP, L2 norms, coverage stats | Phase 4 |
| **Documentation** | | | |
| 19 | `docs/sprint_retrospectives/week1_instance_summary.md` | Sprint retrospective | Phase 4 |

---

## Technical Design Decisions

### 1. PatentSBERTa Model Selection

**Model**: `AI-Growth-Lab/PatentSBERTa` from HuggingFace

This is a BERT-base model fine-tuned on patent text (title+abstract pairs from the PatentMatch dataset). It produces 768-dimensional embeddings.

**Loading pattern**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
```

**512 token limit**: BERT has a hard limit of 512 tokens. The presentation (slide 5) explicitly calls this out and uses the strategy of embedding title+abstract together as a single text to stay within this limit. The EDA must quantify what fraction of patents exceed 512 tokens when title and abstract are concatenated — if significant (>10%), this needs to be addressed (truncation, separate embedding, or chunking) and documented in ADR-001.

**Why PatentSBERTa over general-purpose models**: Patent language is highly specialized — legal terminology, technical jargon, and a formal structure unlike natural language. A model fine-tuned on patent text (PatentSBERTa) will produce better representations than a general-purpose model (all-mpnet-base-v2). This should be validated empirically in the EDA: compare a small sample of PatentSBERTa embeddings to all-mpnet-base-v2 embeddings and verify that PatentSBERTa produces tighter intra-firm clustering.

### 2. Citation Embedding Aggregation (Needs ADR-002)

This is the key open design decision. Arthur's methodology email says "run PatentSBERTa on citations (authors + titles?)" with a question mark. The actual data file (`cited_abstracts.parquet`) contains abstracts, not just titles.

**Data relationships**:
- Each patent in `patent_metadata.parquet` has zero or more citations in `citation_network.parquet`
- Each citation links to an abstract in `cited_abstracts.parquet`
- For each patent, we need ONE 768D "citation embedding"

**The question**: How to produce a single 768D vector from N citation abstracts?

**Recommended approach: Mean pooling**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Mean pooling** (recommended) | Embed each citation abstract, average all vectors | Simple, deterministic, standard in literature | Loses distinguishing detail |
| Weighted mean (by recency) | Weight by citation year | Captures temporal signal | Requires year data (may not have) |
| Concatenate then embed | Concatenate all citation text, embed once | Single model call | Truncation at 512 tokens — most information lost |
| Max pooling | Element-wise max across citation vectors | Preserves strongest signals | Uncommon, harder to interpret |

**Rationale for mean pooling**: It is the simplest and most defensible option. Choi et al. (2019) — cited in the presentation — used aggregated citation embeddings. Mean pooling is the standard aggregation in the literature. It preserves the average direction of knowledge base linkage, which is what matters for downstream GMM clustering. This aligns with the "boring technology" and parsimony principles.

**Zero-citation patents**: Patents with no citations get a 768D zero vector for the citation component. This is deliberate — a patent with no citations has no knowledge base linkage signal, and the zero vector correctly represents "no citation signal." The title+abstract embedding still carries the content signal. This must be documented explicitly in ADR-002.

**Citation embedding batching**: Each patent's citations must be embedded individually, then aggregated. With 46M citation edges and 3.7M unique cited abstracts:
- Embed all 3.7M unique cited abstracts ONCE (not per-citing-patent)
- Use the citation network to look up pre-computed citation embeddings per patent
- Mean-pool the looked-up embeddings

This is much more efficient than re-embedding duplicates.

### 3. Batching Strategy for 2.7M Patents

PatentSBERTa encoding is the computational bottleneck:
- GPU (T4/A10G): ~1000-3000 patents/sec with batch_size=256 → ~15-45 minutes
- CPU: ~50-100 patents/sec with batch_size=64 → ~7.5-15 hours

**Strategy**:
- Use `model.encode(texts, batch_size=256, show_progress_bar=True)` from sentence-transformers
- Process in chunks of 100K patents (GPU) or 10K patents (CPU)
- Checkpoint after each chunk
- Resume from last checkpoint on interruption (check existing checkpoint file for last processed index)

### 4. UMAP Hyperparameters

Target: 1536D → 50D (per methodology specification)

**Initial parameters**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_components | 50 | Per methodology specification |
| n_neighbors | 15 | Default, preserves local structure |
| min_dist | 0.1 | Default, allows moderate clustering |
| metric | cosine | Standard for embedding spaces |
| random_state | 42 | Reproducibility |

**Tuning strategy** (following the one-variable-at-a-time principle):
1. First run with defaults above — this is the baseline
2. Tune n_neighbors: [5, 15, 30, 50] — measure trustworthiness score
3. Tune min_dist: [0.0, 0.1, 0.25, 0.5] — measure separation in downstream GMM (Week 2)
4. Each tuning run produces a 2D UMAP visualization (n_components=2) for sanity checking

**Scale concern**: UMAP on 2.7M × 1536 is memory-intensive (~16GB for the input matrix alone). UMAP uses approximate nearest neighbors (pynndescent) by default, which helps. If memory is a constraint:
- Fallback 1: Subsample 500K patents for UMAP fit, then transform the remaining 2.2M using the fitted model
- Fallback 2: Use parametric UMAP (neural network approximation)
- Start with the full dataset; fall back to subsampling only if needed

### 5. Data Quality Checks (EDA — Phase 1)

The EDA notebook (`notebooks/01_data_eda.ipynb`) MUST answer these questions before any pipeline code is written:

1. **Schema verification**: Do columns match Amie's description?
   - `patent_metadata.parquet`: gvkey, patent_id, title, abstract
   - `cited_abstracts.parquet`: patent_id (cited ID), abstract
   - `citation_network.parquet`: patent_id (firm's patent), citation_id

2. **Null rates**: What percentage of patents have null/empty abstracts? Null titles?

3. **Text length distribution**: Histogram of title+abstract token counts (use PatentSBERTa tokenizer). What fraction exceeds 512 tokens?

4. **Citation count distribution**: Histogram of citations per patent. What fraction have zero citations? What is the median and max?

5. **Firm distribution**: How many unique gvkeys? Patent count per firm (min, median, max, p95)?

6. **Citation coverage**: What fraction of citation_ids in `citation_network.parquet` have matching abstracts in `cited_abstracts.parquet`? Missing citations will reduce citation embedding quality.

7. **Duplicate detection**: Any duplicate patent_ids in `patent_metadata.parquet`?

8. **Sample inspection**: Print 5-10 example patents with their titles and abstracts. Do they look like real patents? Any obvious data quality issues?

### 6. Embedding Storage Format

**Do NOT store embeddings as individual float columns** (768 columns is unwieldy). Instead:

**Recommended format**: Parquet with `patent_id` column + a single binary column containing the serialized numpy array.

```python
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Save
table = pa.table({
    'patent_id': patent_ids,
    'embedding': [embedding.tobytes() for embedding in embeddings]
})
pq.write_table(table, 'output/embeddings/title_abstract_embeddings.parquet')

# Load
table = pq.read_table('output/embeddings/title_abstract_embeddings.parquet')
embeddings = np.array([
    np.frombuffer(b, dtype=np.float32) for b in table['embedding'].to_pylist()
])
```

Include a metadata column or file-level metadata recording: embedding dimension, model name, timestamp, row count.

### 7. Visualizations Required (Phase 4)

1. **2D UMAP projection**: Fit a separate 2D UMAP on the 1536D concatenated vectors. Color by gvkey (firm) for a subsample (~10K patents from ~50 firms). Sanity check: do patents from the same firm loosely cluster? They should cluster somewhat but not form perfect blobs (firms span multiple technology areas).

2. **Embedding L2 norm histograms**: Distribution of L2 norms for title+abstract embeddings and citation embeddings. Should be approximately unit-normalized (sentence-transformers typically normalizes).

3. **Coverage statistics bar chart**: At each pipeline stage, what fraction of patents have valid outputs? (e.g., 100% for title+abstract embedding, 85% for citation embedding if 15% have no citations, 100% for concatenation, 100% for UMAP)

4. **Token length histogram**: Title+abstract token count distribution with a vertical line at 512 showing the truncation boundary.

5. **Citation count distribution**: Histogram showing how many citations each patent has. Log scale if heavy-tailed.

---

## Workflow with Halting Points

### Phase 0: Environment Setup (no approval needed)
- Create venv, install dependencies, verify installation
- Confirm data files are accessible and readable

### Phase 1: EDA and Data Understanding → HALT for approval
- Create and run `notebooks/01_data_eda.ipynb`
- Answer all 8 data quality questions listed above
- Present EDA summary to Torrin with key findings and any concerns
- **Do NOT proceed to Phase 2 until EDA findings are reviewed**

### Phase 2: ADRs and Spec → HALT for Codex review
- Write `docs/adr/adr_001_patent_embedding_model.md`
  - PatentSBERTa choice, 512 token handling, title+abstract concatenation strategy
- Write `docs/adr/adr_002_citation_aggregation.md`
  - Mean pooling decision, zero-citation handling, embedding-once-then-lookup optimization
- Write `docs/specs/patent_vectorizer_spec.md`
  - Full interface spec for: PatentLoader, PatentEncoder, CitationAggregator, UMAPReducer
  - Method signatures, type hints, docstrings, error handling, testing strategy
- **HALT**: Submit ADRs and spec for Codex review. Do NOT proceed until approved.

### Phase 3: Implementation (TDD) → HALT for Codex review
- **For each module, follow the TDD cycle:**
  1. Write tests first (`tests/unit/test_<module>.py`)
  2. Implement minimum code to pass tests (`src/<module>/<file>.py`)
  3. Run tests, fix issues
  4. Move to next module

- **Implementation order** (each builds on the previous):
  1. `src/utils/checkpointing.py` — checkpoint save/load utilities
  2. `src/data_loading/patent_loader.py` — parquet loading with validation
  3. `src/embeddings/patent_encoder.py` — PatentSBERTa encoding with batching
  4. `src/embeddings/citation_aggregator.py` — citation embedding aggregation
  5. `src/dimensionality_reduction/umap_reducer.py` — UMAP reduction

- **Run on small sample first** (1K patents) to validate end-to-end before full dataset
- Generate all four checkpoint files on full dataset
- **HALT**: Submit implementation for Codex review. Do NOT proceed until approved.

### Phase 4: Validation and Visualization → HALT for approval
- Create `notebooks/02_embedding_validation.ipynb`
- Generate all 5 required visualizations
- Run full pipeline on complete dataset (or document why a subsample was used)
- Write sprint retrospective: `docs/sprint_retrospectives/week1_instance_summary.md`
- **HALT**: Present validation evidence to Torrin

---

## What NOT To Do

1. **Do not implement Steps 2-4** (GMM, Bhattacharyya, extensions). Week 1 is vectorization only. Resist the urge to "get ahead."

2. **Do not skip EDA.** Understanding the data is the foundation. Phase 1 must complete before Phase 2 begins.

3. **Do not optimize prematurely.** Get a working pipeline first, then optimize. If CPU encoding takes 15 hours, that's fine for Week 1. Optimization is a future sprint.

4. **Do not use the WSL system venv or the Financial Topic Modeling venv.** Create a project-specific venv.

5. **Do not commit parquet data files to git.** They total 1.3GB. The `.gitignore` already excludes them.

6. **Do not tune UMAP without baseline measurements first.** Run defaults, measure, then tune one variable at a time.

7. **Do not assume all patents have citations.** The EDA will quantify this. Handle zero-citation patents explicitly.

8. **Do not store embeddings as 768 individual float columns.** Use the binary serialization format described above.

9. **Do not write a "framework."** Write concrete functions that solve the concrete problem. If Step 2 needs different abstractions, that's Step 2's problem.

10. **Do not skip the Codex review checkpoints.** Development halts at each gate. This is not overhead — it's the mechanism that catches mistakes early.

---

## Definition of Done

Week 1 is **complete** when ALL of the following are true:

- [ ] EDA notebook documents all 8 data quality findings
- [ ] ADR-001 (PatentSBERTa) is written, reviewed, and approved
- [ ] ADR-002 (citation aggregation) is written, reviewed, and approved
- [ ] Patent vectorizer spec is written, reviewed, and approved
- [ ] All unit tests pass (`pytest tests/ -v` green)
- [ ] Every patent in `patent_metadata.parquet` has a 50D vector (or is explicitly excluded with a documented reason)
- [ ] All four checkpoint parquet files exist and are loadable
- [ ] 2D UMAP visualization shows reasonable structure (not random noise, not a single blob)
- [ ] Embedding L2 norm histograms show expected distribution
- [ ] Coverage statistics document patent counts at each stage
- [ ] Sprint retrospective is written with: What Was Built, Decisions Made, Confidence Levels, What Was NOT Done, Next Phase

---

## Getting Started

1. Read all Required Reading documents (in the order listed above)
2. Set up environment (Phase 0)
3. Load the data and run EDA (Phase 1) — present findings
4. Write ADRs and spec (Phase 2) — submit for Codex review
5. Implement with TDD (Phase 3) — submit for Codex review
6. Validate and visualize (Phase 4) — present evidence
7. Write sprint retrospective

**Your immediate first action**: Read `CLAUDE.md`, then read all 10 files in `docs/values/`, then read `methodology.md` and `data_prep.md`. Only after reading these should you proceed to environment setup.
