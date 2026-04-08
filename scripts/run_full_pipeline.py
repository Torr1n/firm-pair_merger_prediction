"""Run the patent vectorization pipeline on the full v3 dataset.

Usage (on AWS g5.8xlarge or equivalent):
    source venv/bin/activate
    python scripts/run_full_pipeline.py 2>&1 | tee output/pipeline.log

After completion:
    aws s3 sync output/ s3://ubc-torrin/firm-pair-merger/output/ --profile torrin

Checkpoint behavior:
    - Title+abstract encoding checkpoints every 100K patents (~5 min on GPU)
    - Citation encoding checkpoints every 100K abstracts
    - If interrupted, re-run the same command to resume from last checkpoint

Data files (v3):
    - patent_metadata_dedup: Pre-deduplicated by Amie for encoding (1.52M unique patents)
    - patent_metadata: Full file with co-assignments for Week 2 portfolio construction
    - post_deal_flag: Filter to ==0 for clean pre-acquisition features
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
    print("Patent Vectorization Pipeline — Full Scale (v3 dataset)")
    print("=" * 70)

    # --- Stage 0: Load data ---
    print("\n[Stage 0] Loading data...")
    loader = PatentLoader(config)

    # Use the pre-deduplicated file for encoding (Amie handled dedup in v3)
    print("  Loading deduplicated patents for encoding...")
    pm_dedup = loader.load_patent_metadata(
        columns=["patent_id", "gvkey", "title", "abstract", "post_deal_flag"],
        source="dedup",
    )
    print(f"  Dedup patents: {len(pm_dedup):,} (unique patent_ids for encoding)")

    # Filter to pre-deal patents only (post_deal_flag == 0)
    pre_deal_mask = pm_dedup["post_deal_flag"] == 0
    n_post_deal = (~pre_deal_mask).sum()
    pm_dedup = pm_dedup[pre_deal_mask].reset_index(drop=True)
    print(f"  Filtered post-deal patents: {n_post_deal:,} removed")
    print(f"  Pre-deal patents for encoding: {len(pm_dedup):,}")

    # Drop patents with no title AND no abstract
    has_text = pm_dedup["title"].notna() | pm_dedup["abstract"].notna()
    n_dropped = (~has_text).sum()
    if n_dropped > 0:
        print(f"  Dropping {n_dropped:,} patents with no title and no abstract")
        pm_dedup = pm_dedup[has_text].reset_index(drop=True)

    print(f"  Final patent count for encoding: {len(pm_dedup):,}")

    # Save gvkey mapping from FULL metadata (includes co-assignments) for Week 2
    print("  Loading full metadata for gvkey mapping...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm_full = loader.load_patent_metadata(
            columns=["patent_id", "gvkey", "post_deal_flag"],
        )
    # Only keep mappings for patents we're encoding AND pre-deal
    encodable_ids = set(pm_dedup["patent_id"])
    gvkey_map = pm_full[
        (pm_full["patent_id"].isin(encodable_ids)) & (pm_full["post_deal_flag"] == 0)
    ][["patent_id", "gvkey"]]
    gvkey_map.to_parquet(f"{OUTPUT_DIR}/gvkey_map.parquet", index=False)
    print(f"  Gvkey map: {len(gvkey_map):,} rows "
          f"({gvkey_map['gvkey'].nunique():,} firms, "
          f"includes co-assignments)")
    del pm_full

    cn = loader.load_citation_network()
    print(f"  Citation edges: {len(cn):,}")

    cm = CheckpointManager(OUTPUT_DIR)

    # --- Stage 1a: Encode title+abstract ---
    print("\n[Stage 1a] Encoding title+abstract with PatentSBERTa...")
    encoder = PatentEncoder(config)

    patent_ids = pm_dedup["patent_id"].tolist()
    titles = pm_dedup["title"].fillna("").tolist()
    abstracts = pm_dedup["abstract"].fillna("").tolist()
    del pm_dedup

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

    _, cited_embeddings = encoder.encode_texts_checkpointed(
        cited_patent_ids, cited_texts,
        checkpoint_manager=cm,
        checkpoint_path=cited_path,
    )

    t_cit_enc = time.time() - t0
    print(f"  Cited embeddings: {cited_embeddings.shape} ({t_cit_enc/60:.1f} min elapsed)")

    # --- Stage 1c: Aggregate citations per patent ---
    print("\n[Stage 1c] Aggregating citation embeddings (mean pooling)...")
    aggregator = CitationAggregator(config)
    citation_lookup = aggregator.build_citation_lookup(cited_patent_ids, cited_embeddings)
    del cited_embeddings

    # Coverage stats before freeing lookup
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
