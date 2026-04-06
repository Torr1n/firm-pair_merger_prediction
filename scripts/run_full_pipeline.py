"""Run the patent vectorization pipeline on the full v2 dataset.

Usage (on AWS g5.8xlarge or equivalent):
    source venv/bin/activate
    python scripts/run_full_pipeline.py 2>&1 | tee output/pipeline.log

After completion:
    aws s3 sync output/ s3://ubc-torren/firm-pair-merger/output/

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
    gvkey_map_full = pm[["patent_id", "gvkey"]].copy()
    pm = pm.drop_duplicates(subset="patent_id", keep="first").reset_index(drop=True)
    n_after = len(pm)
    print(f"  Deduplicated: {n_before:,} -> {n_after:,} unique patents "
          f"({n_before - n_after:,} multi-firm duplicates)")

    # Drop patents with no title AND no abstract (cannot embed)
    has_text = pm["title"].notna() | pm["abstract"].notna()
    n_dropped = (~has_text).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped:,} patents with no title and no abstract")
        pm = pm[has_text].reset_index(drop=True)

    print(f"  Final patent count for encoding: {len(pm):,}")

    cm = CheckpointManager(OUTPUT_DIR)

    # --- Stage 1a: Encode title+abstract ---
    print("\n[Stage 1a] Encoding title+abstract with PatentSBERTa...")
    encoder = PatentEncoder(config)

    patent_ids = pm["patent_id"].tolist()
    titles = pm["title"].fillna("").tolist()
    abstracts = pm["abstract"].fillna("").tolist()

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

    _, cited_embeddings = encoder.encode_patents(
        cited_patent_ids, cited_texts,
        # For cited abstracts, title IS the abstract (single text field)
        cited_texts,
        checkpoint_manager=cm,
        checkpoint_path=cited_path,
    )
    t_cit_enc = time.time() - t0
    print(f"  Cited embeddings: {cited_embeddings.shape} ({t_cit_enc/60:.1f} min elapsed)")

    # --- Stage 1c: Aggregate citations per patent ---
    print("\n[Stage 1c] Aggregating citation embeddings (mean pooling)...")
    aggregator = CitationAggregator(config)
    citation_lookup = aggregator.build_citation_lookup(cited_patent_ids, cited_embeddings)
    del cited_embeddings  # free ~11 GB

    _, cit_embeddings = aggregator.aggregate(patent_ids, cn, citation_lookup)
    del citation_lookup, cn

    agg_path = f"{OUTPUT_DIR}/citation_embeddings.parquet"
    cm.save_embeddings(patent_ids, cit_embeddings, agg_path,
                       metadata={"model_name": config["embedding"]["model_name"],
                                 "aggregation": "mean_pooling"})

    stats = aggregator.get_coverage_stats(patent_ids,
        loader.load_citation_network(),
        aggregator.build_citation_lookup(cited_patent_ids,
            cm.load_embeddings(cited_path)[1]))
    t_agg = time.time() - t0
    print(f"  Citation embeddings: {cit_embeddings.shape} ({t_agg/60:.1f} min elapsed)")
    print(f"  Zero-citation patents: {stats['zero_citation_patents']:,} "
          f"({stats['zero_citation_pct']:.1%})")

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

    # --- Save gvkey mapping (all rows, including multi-firm) ---
    gvkey_map_full.to_parquet(f"{OUTPUT_DIR}/gvkey_map.parquet", index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min, {elapsed/3600:.1f} hr)")
    print(f"  Patents encoded: {len(patent_ids):,}")
    print(f"  Checkpoint files in {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
