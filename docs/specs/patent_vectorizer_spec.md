# Patent Vectorizer Pipeline — Interface Specification

**Status**: Accepted (Codex-approved)  
**Date**: 2026-04-07  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (pending)  
**ADR Dependencies**: ADR-001 (embedding model), ADR-002 (citation aggregation)

---

## Overview

This spec defines the interfaces for five modules that compose the Week 1 patent vectorization pipeline. Each module is independently testable, checkpointed, and follows the contracts below.

**Pipeline flow**:
```
patent_metadata_dedup.parquet ──→ PatentLoader(source="dedup") ──→ PatentEncoder ──→ title_abstract_embeddings.parquet
                                                                                │
patent_metadata.parquet ──→ PatentLoader(source="full") ──→ gvkey_map.parquet   │
                                                                                │
cited_abstracts.parquet ──→ PatentLoader ──→ PatentEncoder ──→ ┐               │
citation_network.parquet ─→ PatentLoader ──────────────────→ CitationAggregator ──→ citation_embeddings.parquet
                                                                                │
                                                                 ┌──────────────┘
                                                            Concatenate (1536D)
                                                                 │
                                                            UMAPReducer ──→ patent_vectors_50d.parquet
```

---

## Module 1: CheckpointManager

**File**: `src/utils/checkpointing.py`

Handles saving and loading intermediate parquet checkpoints with metadata.

### Interface

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: Directory for checkpoint files. Created if it doesn't exist.
        """

    def save_embeddings(
        self,
        patent_ids: list[str],
        embeddings: np.ndarray,
        output_path: str,
        metadata: dict | None = None,
    ) -> Path:
        """Save patent embeddings as parquet with binary-serialized vectors.

        Storage format (per ADR-001):
            - Column 'patent_id': string
            - Column 'embedding': binary (numpy array .tobytes())

        File-level metadata stored in parquet schema metadata:
            - 'embedding_dim': str(int)
            - 'model_name': str
            - 'row_count': str(int)
            - 'created_at': ISO 8601 timestamp
            - Any additional keys from `metadata` dict

        Args:
            patent_ids: List of patent ID strings. Length must equal embeddings.shape[0].
            embeddings: numpy array of shape (n_patents, embedding_dim), dtype float32.
            output_path: Full path for the output parquet file.
            metadata: Optional dict of additional metadata to store.

        Returns:
            Path to the saved parquet file.

        Raises:
            ValueError: If len(patent_ids) != embeddings.shape[0].
        """

    def load_embeddings(self, path: str) -> tuple[list[str], np.ndarray, dict]:
        """Load patent embeddings from a checkpoint parquet file.

        Returns:
            Tuple of (patent_ids, embeddings, metadata).
            - patent_ids: list[str]
            - embeddings: np.ndarray of shape (n, embedding_dim), dtype float32
            - metadata: dict from parquet schema metadata

        Raises:
            FileNotFoundError: If path does not exist.
        """

    def checkpoint_exists(self, path: str) -> bool:
        """Check if a checkpoint file exists and is a valid parquet file."""
```

### Constraints

- Embedding arrays are stored as binary blobs, NOT as individual float columns (per bootstrap prompt section 6)
- All metadata values are stored as strings in parquet schema metadata
- `load_embeddings` must reconstruct the exact same numpy array that was saved (float32, same shape)

---

## Module 2: PatentLoader

**File**: `src/data_loading/patent_loader.py`

Loads and validates the parquet data files (full metadata, dedup metadata, cited abstracts, citation network).

### Interface

```python
class PatentLoader:
    def __init__(self, config: dict):
        """
        Args:
            config: Parsed YAML config dict. Expected keys under 'data':
                - patent_metadata: path to full patent metadata parquet (with co-assignments)
                - patent_metadata_dedup: path to pre-deduplicated patent metadata parquet
                  (unique patent_ids, used for encoding)
                - cited_abstracts: path to cited_abstracts.parquet
                - citation_network: path to citation_network.parquet
        """

    def load_patent_metadata(
        self, columns: list[str] | None = None, source: str = "full"
    ) -> pd.DataFrame:
        """Load patent metadata with optional column selection.

        Default columns if None: all available columns (excluding __index_level_0__)

        Args:
            columns: Columns to load. None loads all available.
            source: 'full' for complete metadata (includes co-assignments, requires
                    ["patent_id", "title", "abstract"]). 'dedup' for pre-deduplicated
                    file with unique patent_ids for encoding (requires ["patent_id"]).

        Note: The dedup file has unique patent_ids (one row per patent). The full
        file may contain co-assignments (same patent_id linked to multiple gvkeys).

        Validation:
            - File exists and is readable
            - Required columns are present:
              - source="full": ["patent_id", "title", "abstract"]
              - source="dedup": ["patent_id"]
            - patent_id has no nulls
            - Warns (UserWarning) if patent_id has duplicates. v3 data's full file
              contains co-assignments (same patent linked to multiple gvkeys). Callers
              must deduplicate before encoding if unique patent vectors are required.

        Returns:
            DataFrame with requested columns (including any duplicates).

        Raises:
            FileNotFoundError: If parquet file doesn't exist.
            ValueError: If required columns are missing or validation fails.
        """

    def load_cited_abstracts(
        self, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Load cited abstracts with optional column selection.

        Default columns if None: all available columns (excluding __index_level_0__)

        Validation:
            - File exists and is readable
            - Required columns present: patent_id, abstract

        Returns:
            DataFrame with requested columns.

        Raises:
            FileNotFoundError: If parquet file doesn't exist.
            ValueError: If required columns are missing.
        """

    def load_citation_network(
        self, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Load citation network edges.

        Default columns if None: ['patent_id', 'citation_id']

        Validation:
            - File exists and is readable
            - Required columns present: patent_id, citation_id
            - No null values in either column

        Returns:
            DataFrame with requested columns.

        Raises:
            FileNotFoundError: If parquet file doesn't exist.
            ValueError: If required columns are missing or validation fails.
        """

    def get_row_counts(self) -> dict[str, int]:
        """Return row counts for all data files without loading data.

        Returns:
            Dict with keys 'patent_metadata', 'patent_metadata_dedup',
            'cited_abstracts', 'citation_network'.
        """
```

### Constraints

- When `columns=None`, loads all available columns (excluding artifacts). Callers should pass explicit column lists for memory-sensitive paths (e.g., full-scale encoding).
- `__index_level_0__` column is always excluded from loads (pandas artifact)
- All validation happens at load time, not lazily

---

## Module 3: PatentEncoder

**File**: `src/embeddings/patent_encoder.py`

Encodes patent text into 768D embeddings using PatentSBERTa.

### Interface

```python
class PatentEncoder:
    def __init__(self, config: dict):
        """
        Args:
            config: Parsed YAML config dict. Expected keys under 'embedding':
                - model_name: HuggingFace model name (default: 'AI-Growth-Lab/PatentSBERTa')
                - batch_size: Encoding batch size (default: 256 for GPU, 64 for CPU)
                - output_dim: Expected output dimension (768)

        Loads the SentenceTransformer model on initialization. Device selection:
        probes CUDA with a test allocation; falls back to CPU if CUDA is
        non-functional (e.g., GPU compute capability not supported by PyTorch build).
        """

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of text strings into embeddings.

        Args:
            texts: List of text strings to encode. Empty strings are allowed.
            batch_size: Override config batch_size if provided.
            show_progress: Show progress bar during encoding.

        Returns:
            np.ndarray of shape (len(texts), 768), dtype float32.

        Raises:
            ValueError: If texts is empty.
        """

    def encode_patents(
        self,
        patent_ids: list[str],
        titles: list[str],
        abstracts: list[str],
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every_n: int | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Encode patent title+abstract pairs with optional checkpointing.

        Text preparation: For each patent, concatenates title and abstract
        as '{title} {abstract}'. If abstract is None/empty, uses title only.

        Checkpointing: If checkpoint_manager and checkpoint_path are provided:
            - Checks for existing checkpoint and resumes from last saved index
            - Saves intermediate results every checkpoint_every_n patents

        Args:
            patent_ids: List of patent ID strings.
            titles: List of title strings (same length as patent_ids).
            abstracts: List of abstract strings (same length, may contain None).
            checkpoint_manager: Optional CheckpointManager for saving progress.
            checkpoint_path: Path for the checkpoint file.
            checkpoint_every_n: Save checkpoint every N patents. Defaults to
                config['embedding']['checkpoint_every_n'] or 100000.

        Returns:
            Tuple of (patent_ids, embeddings) where embeddings is np.ndarray
            of shape (len(patent_ids), 768), dtype float32.

        Raises:
            ValueError: If input lists have different lengths.
        """
```

### Constraints

- Model is loaded once at init, reused for all encode calls
- `encode_texts` is the low-level method; `encode_patents` handles text preparation and checkpointing
- Null/empty abstracts are handled by using title-only text (per ADR-001)
- Output dtype is always float32
- The model's default truncation at 512 tokens is used (per ADR-001 — no custom truncation logic)

---

## Module 4: CitationAggregator

**File**: `src/embeddings/citation_aggregator.py`

Aggregates citation embeddings per patent using mean pooling (per ADR-002).

### Interface

```python
class CitationAggregator:
    def __init__(self, config: dict):
        """
        Args:
            config: Parsed YAML config dict. Expected keys under 'citation_aggregation':
                - method: 'mean_pooling' (only supported method)
                - zero_citation_strategy: 'zero_vector'
        """

    def build_citation_lookup(
        self,
        cited_patent_ids: list[str],
        cited_embeddings: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build a lookup table mapping cited patent IDs to their embeddings.

        Args:
            cited_patent_ids: List of cited patent ID strings.
            cited_embeddings: np.ndarray of shape (n_cited, embedding_dim).

        Returns:
            Dict mapping patent_id -> 768D numpy array.

        Raises:
            ValueError: If lengths don't match.
        """

    def aggregate(
        self,
        patent_ids: list[str],
        citation_network: pd.DataFrame,
        citation_lookup: dict[str, np.ndarray],
        embedding_dim: int = 768,
    ) -> tuple[list[str], np.ndarray]:
        """Compute mean-pooled citation embeddings for each patent.

        For each patent_id:
            1. Find all citation_ids from the citation_network
            2. Look up each citation_id in citation_lookup
            3. Mean-pool all found embeddings
            4. If no citations found (or none have embeddings), use zero vector

        Args:
            patent_ids: List of patent IDs to compute citation embeddings for.
            citation_network: DataFrame with columns ['patent_id', 'citation_id'].
            citation_lookup: Dict from build_citation_lookup().
            embedding_dim: Embedding dimension (default 768).

        Returns:
            Tuple of (patent_ids, citation_embeddings) where citation_embeddings
            is np.ndarray of shape (len(patent_ids), embedding_dim), dtype float32.

        Raises:
            ValueError: If citation_network missing required columns.
        """

    def get_coverage_stats(
        self,
        patent_ids: list[str],
        citation_network: pd.DataFrame,
        citation_lookup: dict[str, np.ndarray],
    ) -> dict:
        """Compute citation coverage statistics.

        Returns:
            Dict with keys:
                - 'total_patents': int
                - 'zero_citation_patents': int (no citations in network)
                - 'zero_citation_pct': float
                - 'mean_citations_per_patent': float
                - 'median_citations_per_patent': float
                - 'total_edges': int
                - 'edges_with_embeddings': int
                - 'edge_coverage_pct': float
        """
```

### Constraints

- Only `mean_pooling` aggregation is implemented (per ADR-002)
- Zero-citation patents get a zero vector of the correct dimension
- The citation_lookup is built once from pre-computed cited abstract embeddings (embed-once-then-lookup per ADR-002)
- Citation IDs not found in the lookup are silently skipped (their absence reduces the denominator for mean pooling)
- If a patent has citations but none are in the lookup, it is treated as zero-citation

---

## Module 5: UMAPReducer

**File**: `src/dimensionality_reduction/umap_reducer.py`

Reduces 1536D concatenated vectors to 50D using UMAP.

### Interface

```python
class UMAPReducer:
    def __init__(self, config: dict):
        """
        Args:
            config: Parsed YAML config dict. Expected keys under 'umap':
                - n_components: Target dimensionality (default: 50)
                - n_neighbors: UMAP n_neighbors (default: 15)
                - min_dist: UMAP min_dist (default: 0.1)
                - metric: Distance metric (default: 'cosine')
                - random_state: Random seed (default: 42)
        """

    def fit_transform(
        self, vectors: np.ndarray
    ) -> np.ndarray:
        """Fit UMAP on the input vectors and return reduced vectors.

        Args:
            vectors: np.ndarray of shape (n_samples, input_dim), dtype float32.

        Returns:
            np.ndarray of shape (n_samples, n_components), dtype float32.

        Raises:
            ValueError: If vectors has fewer than n_neighbors samples.
        """

    def get_params(self) -> dict:
        """Return the UMAP parameters used for reproducibility logging."""
```

### Constraints

- UMAP parameters come from config (no hardcoded defaults)
- `random_state` must be set for reproducibility
- Output dtype is float32
- The fitted UMAP model is stored on the instance for potential later use (e.g., transforming held-out data)

---

## Concatenation Step

Concatenation is a simple operation that does not warrant its own module. It is handled inline in the pipeline orchestration:

```python
# After encoding title+abstract (768D) and citation aggregation (768D):
concatenated = np.concatenate([title_abstract_embeddings, citation_embeddings], axis=1)
# Result: (n_patents, 1536) array
```

**Validation**: Assert that both input arrays have the same number of rows and that patent_id ordering matches.

---

## Configuration Reference

All modules read from the same parsed YAML config (`src/config/config.yaml`):

```yaml
embedding:
  model_name: "AI-Growth-Lab/PatentSBERTa"
  output_dim: 768
  max_token_length: 512
  batch_size: 256          # GPU; use 64 for CPU
  checkpoint_every_n: 100000

citation_aggregation:
  method: "mean_pooling"
  zero_citation_strategy: "zero_vector"

umap:
  n_components: 50
  n_neighbors: 15
  min_dist: 0.1
  metric: "cosine"
  random_state: 42

data:
  patent_metadata: "data/firm_patents_text_metadata_techbio_v3.parquet"
  patent_metadata_dedup: "data/firm_patents_dedup_techbio_v3.parquet"
  cited_abstracts: "data/cited_abstracts_techbio_v3.parquet"
  citation_network: "data/citation_network_techbio_v3.parquet"

output:
  checkpoint_dir: "output/embeddings"
  title_abstract_embeddings: "output/embeddings/title_abstract_embeddings.parquet"
  citation_embeddings: "output/embeddings/citation_embeddings.parquet"
  concatenated_vectors: "output/embeddings/concatenated_1536d.parquet"
  patent_vectors_50d: "output/embeddings/patent_vectors_50d.parquet"
  gvkey_map: "output/embeddings/gvkey_map.parquet"
```

---

## Testing Strategy

Each module has a corresponding test file in `tests/unit/`. Tests are written BEFORE implementation (TDD).

### test_checkpointing.py
- Save and load round-trip preserves patent_ids, embeddings (exact float32 equality), and metadata
- Handles edge case: single patent
- `checkpoint_exists` returns False for non-existent path
- Validates patent_id/embedding length mismatch raises ValueError

### test_patent_loader.py
- Loads each file and verifies expected columns are present
- Validates patent_id uniqueness and non-null constraints
- Column-selective loading works (only requested columns returned)
- FileNotFoundError for missing files
- ValueError for missing required columns
- Uses small fixture parquet files (not the full dataset)

### test_patent_encoder.py
- `encode_texts` returns correct shape (n, 768) and dtype (float32)
- `encode_patents` handles null abstracts (uses title-only)
- Deterministic: same input produces same output
- Checkpointing: encode with checkpoint, verify checkpoint file exists, resume from checkpoint produces same result
- Uses small sample (10-50 patents) with the real model (not mocked)

### test_citation_aggregator.py
- `build_citation_lookup` creates correct mapping
- `aggregate` with known inputs produces expected mean-pooled output
- Zero-citation patent gets zero vector
- Patent with citations but none in lookup gets zero vector
- Coverage stats are accurate
- Uses synthetic data (hand-crafted patent_ids, embeddings, citation network)

### test_umap_reducer.py
- Output shape is (n_samples, n_components)
- Output dtype is float32
- Reproducible: same input + same random_state → same output
- ValueError if input has fewer samples than n_neighbors
- Uses synthetic random data (not real embeddings)

---

## Checkpoint File Inventory

| Stage | File | Shape | Contents |
|-------|------|-------|----------|
| 0 | `gvkey_map.parquet` | (~1,447,673 × 2) | patent_id to gvkey mapping from full metadata |
| 1a | `title_abstract_embeddings.parquet` | (~1,447,673 × 768) | PatentSBERTa on title+abstract |
| 1b | `citation_embeddings.parquet` | (~1,447,673 × 768) | Mean-pooled citation embeddings (~8% zero vectors) |
| 2 | `concatenated_1536d.parquet` | (~1,447,673 × 1536) | Concatenation of 1a and 1b |
| 3 | `patent_vectors_50d.parquet` | (~1,447,673 × 50) | UMAP reduction of stage 2 |

Row count note: ~1,519,401 dedup patents minus ~71K with post_deal_flag=1 yields ~1,447,673 patents for encoding.

All embedding files include `patent_id` column and binary-serialized embedding column.
