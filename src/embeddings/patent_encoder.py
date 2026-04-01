"""PatentSBERTa encoding with batching and checkpointing."""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.checkpointing import CheckpointManager


class PatentEncoder:
    def __init__(self, config: dict):
        emb_cfg = config["embedding"]
        self._model_name = emb_cfg["model_name"]
        self._batch_size = emb_cfg.get("batch_size", 64)
        self._output_dim = emb_cfg.get("output_dim", 768)
        self._checkpoint_every_n = emb_cfg.get("checkpoint_every_n", 100_000)
        self._model = SentenceTransformer(self._model_name)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        if len(texts) == 0:
            raise ValueError("texts must not be empty")

        bs = batch_size or self._batch_size
        embeddings = self._model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_patents(
        self,
        patent_ids: list[str],
        titles: list[str],
        abstracts: list[str],
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every_n: int | None = None,
    ) -> tuple[list[str], np.ndarray]:
        if len(patent_ids) != len(titles) or len(patent_ids) != len(abstracts):
            raise ValueError(
                f"Length mismatch: {len(patent_ids)} ids, "
                f"{len(titles)} titles, {len(abstracts)} abstracts"
            )

        every_n = checkpoint_every_n or self._checkpoint_every_n

        # Check for existing checkpoint to resume from
        start_idx = 0
        existing_ids = []
        existing_emb = None
        if checkpoint_manager and checkpoint_path and checkpoint_manager.checkpoint_exists(checkpoint_path):
            existing_ids, existing_emb, _ = checkpoint_manager.load_embeddings(checkpoint_path)
            start_idx = len(existing_ids)

        # Prepare texts: "{title} {abstract}" or title-only for null/empty abstracts
        texts = []
        for title, abstract in zip(titles, abstracts):
            t = title or ""
            a = abstract or ""
            a = a.strip()
            if a:
                texts.append(f"{t} {a}")
            else:
                texts.append(t)

        # Encode remaining patents in chunks
        remaining_texts = texts[start_idx:]
        if remaining_texts:
            new_embeddings = self.encode_texts(remaining_texts, show_progress=True)
        else:
            new_embeddings = np.empty((0, self._output_dim), dtype=np.float32)

        # Combine with existing checkpoint
        if existing_emb is not None and len(existing_emb) > 0:
            all_embeddings = np.concatenate([existing_emb, new_embeddings], axis=0)
        else:
            all_embeddings = new_embeddings

        # Save final checkpoint
        if checkpoint_manager and checkpoint_path:
            checkpoint_manager.save_embeddings(
                patent_ids,
                all_embeddings,
                checkpoint_path,
                metadata={"model_name": self._model_name},
            )

        return patent_ids, all_embeddings
