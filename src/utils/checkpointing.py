"""Checkpoint save/load for embedding parquet files."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_embeddings(
        self,
        patent_ids: list[str],
        embeddings: np.ndarray,
        output_path: str,
        metadata: dict | None = None,
    ) -> Path:
        if len(patent_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Length mismatch: {len(patent_ids)} patent_ids vs "
                f"{embeddings.shape[0]} embeddings"
            )

        table = pa.table({
            "patent_id": patent_ids,
            "embedding": [row.astype(np.float32).tobytes() for row in embeddings],
        })

        file_metadata = {
            "embedding_dim": str(embeddings.shape[1]),
            "row_count": str(len(patent_ids)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            file_metadata.update({k: str(v) for k, v in metadata.items()})

        # Merge into schema metadata
        existing = table.schema.metadata or {}
        merged = {k.encode(): v.encode() for k, v in file_metadata.items()}
        merged.update(existing)
        table = table.replace_schema_metadata(merged)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(path))
        return path

    def load_embeddings(self, path: str) -> tuple[list[str], np.ndarray, dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        table = pq.read_table(str(path))
        patent_ids = table["patent_id"].to_pylist()
        embedding_dim = int(table.schema.metadata.get(b"embedding_dim", b"0"))

        embeddings = np.array(
            [np.frombuffer(b, dtype=np.float32) for b in table["embedding"].to_pylist()]
        )

        meta = {
            k.decode(): v.decode()
            for k, v in (table.schema.metadata or {}).items()
        }

        return patent_ids, embeddings, meta

    def checkpoint_exists(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            pq.read_metadata(str(p))
            return True
        except Exception:
            return False
