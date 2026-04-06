"""Run the patent vectorization pipeline on the full v2 dataset.

Usage (on AWS g5.8xlarge or equivalent):
    source venv/bin/activate
    python scripts/run_full_pipeline.py 2>&1 | tee output/pipeline.log

After completion:
    aws s3 sync output/ s3://ubc-torrin/firm-pair-merger/output/

Checkpoint behavior:
    - Title+abstract encoding checkpoints every 100K patents (~5 min on GPU)
    - Citation encoding checkpoints every 100K abstracts
    - If interrupted, re-run the same command to resume from last checkpoint
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loading.patent_loader import PatentLoader
from src.embeddings.patent_encoder import PatentEncoder
from src.embeddings.citation_aggregator import CitationAggregator
from src.dimensionality_reduction.umap_reducer import UMAPReducer
from src.utils.checkpointing import CheckpointManager

OUTPUT_DIR = "output/embeddings"


def main():
    t0 = time.time()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    config = load_config()

    print("=" * 70)
    print("Patent Vectorization Pipeline — Full Scale (v2 dataset)")
    print("=" * 70)

    # --- Stage 0: Load and prepare data ---
    print("\n[Stage 0] Loading data...")
    loader = PatentLoader(config)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        pm = loader.load_patent_metadata(columns=["patent_id", "gvkey", "title", "abstract"])

    cn = loader.load_citation_network()
    print(f"  Patents: {len(pm):,}")
    print(f"  Citation edges: {len(cn):,}")

    # Deduplicate patents by patent_id (keep first occurrence).
    # Multi-firm patents are preserved in the gvkey mapping but encoded once.
    n_before = len(pm)
    pm_deduped = pm.drop_duplicates(subset="patent_id", keep="first").reset_index(drop=True)
    n_after = len(pm_deduped)
    print(f"  Deduplicated: {n_before:,} -> {n_after:,} unique patents "
          f"({n_before - n_after:,} multi-firm duplicates)")

    # Drop patents with no title AND no abstract (cannot embed)
    has_text = pm_deduped["title"].notna() | pm_deduped["abstract"].notna()
    n_dropped = (~has_text).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped:,} patents with no title and no abstract")
        pm_deduped = pm_deduped[has_text].reset_index(drop=True)

    print(f"  Final patent count for encoding: {len(pm_deduped):,}")

    # Save gvkey mapping ONLY for patents that will have vectors
    encodable_ids = set(pm_deduped["patent_id"])
    gvkey_map = pm[pm["patent_id"].isin(encodable_ids)][["patent_id", "gvkey"]]
    gvkey_map.to_parquet(f"{OUTPUT_DIR}/gvkey_map.parquet", index=False)
    print(f"  Gvkey map: {len(gvkey_map):,} rows "
          f"({gvkey_map['gvkey'].nunique():,} firms)")
    del pm  # free original

    cm = CheckpointManager(OUTPUT_DIR)

    # --- Stage 1a: Encode title+abstract ---
    print("\n[Stage 1a] Encoding title+abstract with PatentSBERTa...")
    encoder = PatentEncoder(config)

    patent_ids = pm_deduped["patent_id"].tolist()
    titles = pm_deduped["title"].fillna("").tolist()
    abstracts = pm_deduped["abstract"].fillna("").tolist()
    del pm_deduped

    ta_path = f"{OUTPUT_DIR}/title_abstract_embeddings.parquet"
    _, ta_embeddings = encoder.encode_patents(
        patent_ids, titles, abstracts,
        checkpoint_manager=cm,
        checkpoint_path=ta_path,
    )
    t_ta = time.time() - t0
    print(f"  Title+abstract embeddings: {ta_embeddings.shape} ({t_ta/60:.1f} min elapsed)")

    # --- Stage 1b: Encode cited abstracts ---
    print("\n[Stage 1b] Encoding cited abstracts...")
    ca = loader.load_cited_abstracts()
    ca = ca.dropna(subset=["abstract"])
    ca = ca[ca["abstract"].str.strip() != ""]
    print(f"  Cited abstracts to encode: {len(ca):,}")

    cited_path = f"{OUTPUT_DIR}/cited_abstract_embeddings.parquet"
    cited_patent_ids = ca["patent_id"].tolist()
    cited_texts = ca["abstract"].tolist()
    del ca

    # Use encode_texts for cited abstracts (single text, not title+abstract pairs).
    # Supports prefix-resume: if a partial checkpoint exists, resume from where it stopped.
    every_n = config["embedding"].get("checkpoint_every_n", 100_000)
    start_idx = 0
    chunks = []

    if cm.checkpoint_exists(cited_path):
        loaded_ids, loaded_emb, _ = cm.load_embeddings(cited_path)
        expected_prefix = cited_patent_ids[:len(loaded_ids)]
        if loaded_ids == expected_prefix:
            start_idx = len(loaded_ids)
            chunks.append(loaded_emb)
            print(f"  Resuming from checkpoint: {start_idx:,}/{len(cited_texts):,} already encoded")
        else:
            raise ValueError(
                f"Cited abstracts checkpoint ID mismatch. "
                f"Delete {cited_path} to re-encode."
            )

    remaining = cited_texts[start_idx:]
    for chunk_start in range(0, len(remaining), every_n):
        chunk_end = min(chunk_start + every_n, len(remaining))
        chunk = remaining[chunk_start:chunk_end]
        chunk_emb = encoder.encode_texts(chunk, show_progress=True)
        chunks.append(chunk_emb)

        # Save intermediate checkpoint
        progress = start_idx + chunk_end
        all_so_far = np.concatenate(chunks, axis=0)
        cm.save_embeddings(
            cited_patent_ids[:progress], all_so_far, cited_path,
            metadata={"model_name": config["embedding"]["model_name"]},
        )
        print(f"  Cited checkpoint: {progress:,}/{len(cited_texts):,}")

    cited_embeddings = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 768), dtype=np.float32)

    t_cit_enc = time.time() - t0
    print(f"  Cited embeddings: {cited_embeddings.shape} ({t_cit_enc/60:.1f} min elapsed)")

    # --- Stage 1c: Aggregate citations per patent ---
    print("\n[Stage 1c] Aggregating citation embeddings (mean pooling)...")
    aggregator = CitationAggregator(config)
    citation_lookup = aggregator.build_citation_lookup(cited_patent_ids, cited_embeddings)
    del cited_embeddings  # free ~11 GB

    # Compute coverage stats BEFORE freeing the lookup
    stats = aggregator.get_coverage_stats(patent_ids, cn, citation_lookup)
    print(f"  Zero-citation patents: {stats['zero_citation_patents']:,} "
          f"({stats['zero_citation_pct']:.1%})")
    print(f"  Edge coverage: {stats['edges_with_embeddings']:,}/{stats['total_edges']:,} "
          f"({stats['edge_coverage_pct']:.1%})")

    _, cit_embeddings = aggregator.aggregate(patent_ids, cn, citation_lookup)
    del citation_lookup, cn

    agg_path = f"{OUTPUT_DIR}/citation_embeddings.parquet"
    cm.save_embeddings(patent_ids, cit_embeddings, agg_path,
                       metadata={"model_name": config["embedding"]["model_name"],
                                 "aggregation": "mean_pooling"})
    t_agg = time.time() - t0
    print(f"  Citation embeddings: {cit_embeddings.shape} ({t_agg/60:.1f} min elapsed)")

    # --- Stage 2: Concatenate ---
    print("\n[Stage 2] Concatenating to 1536D...")
    concatenated = np.concatenate([ta_embeddings, cit_embeddings], axis=1)
    del ta_embeddings, cit_embeddings

    concat_path = f"{OUTPUT_DIR}/concatenated_1536d.parquet"
    cm.save_embeddings(patent_ids, concatenated, concat_path,
                       metadata={"dimensions": "1536"})
    print(f"  Concatenated: {concatenated.shape}")

    # --- Stage 3: UMAP ---
    print("\n[Stage 3] UMAP reduction 1536D -> 50D...")
    print(f"  Input matrix: {concatenated.nbytes / 1e9:.1f} GB")
    reducer = UMAPReducer(config)
    vectors_50d = reducer.fit_transform(concatenated)
    del concatenated

    umap_path = f"{OUTPUT_DIR}/patent_vectors_50d.parquet"
    cm.save_embeddings(patent_ids, vectors_50d, umap_path,
                       metadata={"umap_params": str(reducer.get_params())})
    print(f"  UMAP output: {vectors_50d.shape}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min, {elapsed/3600:.1f} hr)")
    print(f"  Patents encoded: {len(patent_ids):,}")
    print(f"  Checkpoint files in {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
