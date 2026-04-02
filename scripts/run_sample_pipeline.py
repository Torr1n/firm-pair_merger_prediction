"""Run the patent vectorization pipeline on a 1K sample for local validation.

Usage:
    source venv/bin/activate
    python scripts/run_sample_pipeline.py

Outputs checkpoint files to output/embeddings/sample_* for validation notebook.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data_loading.patent_loader import PatentLoader
from src.embeddings.patent_encoder import PatentEncoder
from src.embeddings.citation_aggregator import CitationAggregator
from src.dimensionality_reduction.umap_reducer import UMAPReducer
from src.utils.checkpointing import CheckpointManager

SAMPLE_SIZE = 1000
OUTPUT_DIR = "output/embeddings"


def main():
    t0 = time.time()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    config = load_config()

    # Override batch size for CPU
    config["embedding"]["batch_size"] = 32

    print("=" * 60)
    print("Patent Vectorization Pipeline — 1K Sample Validation")
    print("=" * 60)

    # --- Stage 0: Load data ---
    print("\n[Stage 0] Loading data...")
    loader = PatentLoader(config)
    pm = loader.load_patent_metadata(columns=["patent_id", "gvkey", "title", "abstract"])
    cn = loader.load_citation_network()
    ca = loader.load_cited_abstracts()
    print(f"  Loaded {len(pm):,} patents, {len(cn):,} edges, {len(ca):,} cited abstracts")

    # --- Stage 0b: Sample 1K patents ---
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(pm), size=SAMPLE_SIZE, replace=False)
    pm_sample = pm.iloc[sample_idx].reset_index(drop=True)
    sample_ids = set(pm_sample["patent_id"])
    print(f"  Sampled {len(pm_sample):,} patents")

    # Filter citation network to sample patents
    cn_sample = cn[cn["patent_id"].isin(sample_ids)].reset_index(drop=True)
    print(f"  Sample has {len(cn_sample):,} citation edges")

    # Free full dataframes
    del pm, cn

    cm = CheckpointManager(OUTPUT_DIR)

    # --- Stage 1a: Encode title+abstract ---
    print("\n[Stage 1a] Encoding title+abstract with PatentSBERTa...")
    encoder = PatentEncoder(config)

    patent_ids = pm_sample["patent_id"].tolist()
    titles = pm_sample["title"].tolist()
    abstracts = pm_sample["abstract"].tolist()

    ta_path = f"{OUTPUT_DIR}/sample_title_abstract_embeddings.parquet"
    _, ta_embeddings = encoder.encode_patents(
        patent_ids, titles, abstracts,
        checkpoint_manager=cm,
        checkpoint_path=ta_path,
    )
    print(f"  Title+abstract embeddings: {ta_embeddings.shape}")

    # --- Stage 1b: Encode cited abstracts and aggregate ---
    print("\n[Stage 1b] Encoding cited abstracts...")
    # Only encode cited abstracts that are actually referenced by sample
    cited_ids_needed = set(cn_sample["citation_id"].unique())
    ca_relevant = ca[ca["patent_id"].isin(cited_ids_needed)].reset_index(drop=True)
    del ca

    # Handle null abstracts in cited patents
    ca_relevant = ca_relevant.dropna(subset=["abstract"])
    ca_relevant = ca_relevant[ca_relevant["abstract"].str.strip() != ""]
    print(f"  Relevant cited abstracts to encode: {len(ca_relevant):,}")

    if len(ca_relevant) > 0:
        cited_patent_ids = ca_relevant["patent_id"].tolist()
        cited_texts = ca_relevant["abstract"].tolist()
        cited_embeddings = encoder.encode_texts(cited_texts, show_progress=True)
    else:
        cited_patent_ids = []
        cited_embeddings = np.empty((0, 768), dtype=np.float32)

    print("\n[Stage 1b] Aggregating citation embeddings (mean pooling)...")
    aggregator = CitationAggregator(config)
    citation_lookup = aggregator.build_citation_lookup(cited_patent_ids, cited_embeddings)
    _, cit_embeddings = aggregator.aggregate(patent_ids, cn_sample, citation_lookup)

    cit_path = f"{OUTPUT_DIR}/sample_citation_embeddings.parquet"
    cm.save_embeddings(patent_ids, cit_embeddings, cit_path,
                       metadata={"model_name": config["embedding"]["model_name"],
                                 "aggregation": "mean_pooling"})
    print(f"  Citation embeddings: {cit_embeddings.shape}")

    stats = aggregator.get_coverage_stats(patent_ids, cn_sample, citation_lookup)
    print(f"  Zero-citation patents: {stats['zero_citation_patents']} "
          f"({stats['zero_citation_pct']:.1%})")
    print(f"  Edge coverage: {stats['edges_with_embeddings']}/{stats['total_edges']} "
          f"({stats['edge_coverage_pct']:.1%})")

    # --- Stage 2: Concatenate ---
    print("\n[Stage 2] Concatenating to 1536D...")
    concatenated = np.concatenate([ta_embeddings, cit_embeddings], axis=1)
    concat_path = f"{OUTPUT_DIR}/sample_concatenated_1536d.parquet"
    cm.save_embeddings(patent_ids, concatenated, concat_path,
                       metadata={"dimensions": "1536"})
    print(f"  Concatenated: {concatenated.shape}")

    # --- Stage 3: UMAP ---
    print("\n[Stage 3] UMAP reduction 1536D -> 50D...")
    reducer = UMAPReducer(config)
    vectors_50d = reducer.fit_transform(concatenated)

    umap_path = f"{OUTPUT_DIR}/sample_patent_vectors_50d.parquet"
    cm.save_embeddings(patent_ids, vectors_50d, umap_path,
                       metadata={"umap_params": str(reducer.get_params())})
    print(f"  UMAP output: {vectors_50d.shape}")

    # --- Save gvkey mapping for validation notebook ---
    gvkey_map = pm_sample[["patent_id", "gvkey"]].to_parquet(
        f"{OUTPUT_DIR}/sample_gvkey_map.parquet", index=False
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Checkpoint files in {OUTPUT_DIR}/sample_*")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
