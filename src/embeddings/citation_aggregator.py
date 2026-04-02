"""Citation embedding aggregation via mean pooling (per ADR-002)."""

import numpy as np
import pandas as pd


class CitationAggregator:
    def __init__(self, config: dict):
        cfg = config["citation_aggregation"]
        self._method = cfg["method"]
        self._zero_strategy = cfg["zero_citation_strategy"]

        if self._method != "mean_pooling":
            raise ValueError(f"Unsupported aggregation method: {self._method}")
        if self._zero_strategy != "zero_vector":
            raise ValueError(f"Unsupported zero-citation strategy: {self._zero_strategy}")

    def build_citation_lookup(
        self,
        cited_patent_ids: list[str],
        cited_embeddings: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if len(cited_patent_ids) != cited_embeddings.shape[0]:
            raise ValueError(
                f"Length mismatch: {len(cited_patent_ids)} ids vs "
                f"{cited_embeddings.shape[0]} embeddings"
            )
        return {pid: emb for pid, emb in zip(cited_patent_ids, cited_embeddings)}

    def aggregate(
        self,
        patent_ids: list[str],
        citation_network: pd.DataFrame,
        citation_lookup: dict[str, np.ndarray],
        embedding_dim: int = 768,
    ) -> tuple[list[str], np.ndarray]:
        required = {"patent_id", "citation_id"}
        if not required.issubset(citation_network.columns):
            raise ValueError(
                f"citation_network missing required columns: "
                f"{required - set(citation_network.columns)}"
            )

        # Group citation_ids by patent_id for fast lookup
        grouped = citation_network.groupby("patent_id")["citation_id"].apply(list)

        result = np.zeros((len(patent_ids), embedding_dim), dtype=np.float32)

        for i, pid in enumerate(patent_ids):
            if pid not in grouped.index:
                continue  # zero vector (no citations)

            cit_ids = grouped[pid]
            vectors = []
            for cid in cit_ids:
                if cid in citation_lookup:
                    vectors.append(citation_lookup[cid])

            if vectors:
                result[i] = np.mean(vectors, axis=0).astype(np.float32)
            # else: stays zero vector

        return patent_ids, result

    def get_coverage_stats(
        self,
        patent_ids: list[str],
        citation_network: pd.DataFrame,
        citation_lookup: dict[str, np.ndarray],
    ) -> dict:
        pid_set = set(patent_ids)
        relevant_network = citation_network[citation_network["patent_id"].isin(pid_set)]

        grouped = relevant_network.groupby("patent_id")["citation_id"].apply(list)
        patents_with_citations = set(grouped.index) & pid_set

        zero_citation = len(patent_ids) - len(patents_with_citations)

        total_edges = len(relevant_network)
        edges_with_emb = relevant_network["citation_id"].isin(citation_lookup).sum()

        citation_counts = [len(grouped.get(pid, [])) for pid in patent_ids]

        return {
            "total_patents": len(patent_ids),
            "zero_citation_patents": zero_citation,
            "zero_citation_pct": zero_citation / len(patent_ids) if patent_ids else 0,
            "mean_citations_per_patent": float(np.mean(citation_counts)),
            "median_citations_per_patent": float(np.median(citation_counts)),
            "total_edges": total_edges,
            "edges_with_embeddings": int(edges_with_emb),
            "edge_coverage_pct": int(edges_with_emb) / total_edges if total_edges > 0 else 0,
        }
