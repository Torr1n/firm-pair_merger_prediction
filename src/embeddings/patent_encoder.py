"""PatentSBERTa encoding with batching and checkpointing."""

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


class PatentEncoder:
    def __init__(self, config: dict):
        emb_cfg = config["embedding"]
        self._model_name = emb_cfg["model_name"]
        self._batch_size = emb_cfg.get("batch_size", 64)
        self._output_dim = emb_cfg.get("output_dim", 768)
        self._checkpoint_every_n = emb_cfg.get("checkpoint_every_n", 100_000)

        # Use GPU only if CUDA is actually functional, otherwise CPU
        device = "cpu"
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                device = "cuda"
            except Exception:
                pass

        self._model = SentenceTransformer(self._model_name, device=device)

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

        # Resume from checkpoint if available
        start_idx = 0
        chunks = []
        if checkpoint_manager and checkpoint_path and checkpoint_manager.checkpoint_exists(checkpoint_path):
            existing_ids, existing_emb, _ = checkpoint_manager.load_embeddings(checkpoint_path)
            # Verify ID prefix matches to prevent silent misalignment
            expected_prefix = patent_ids[:len(existing_ids)]
            if existing_ids != expected_prefix:
                raise ValueError(
                    f"Checkpoint ID mismatch: checkpoint contains {len(existing_ids)} "
                    f"patents starting with {existing_ids[:3]}, but current workload "
                    f"starts with {expected_prefix[:3]}. Cannot safely resume — delete "
                    f"the checkpoint file to restart from scratch."
                )
            start_idx = len(existing_ids)
            chunks.append(existing_emb)
            logger.info("Resuming from checkpoint: %s patents already encoded", f"{start_idx:,}")

        # Encode remaining patents in chunks of every_n
        remaining_texts = texts[start_idx:]
        for chunk_start in range(0, len(remaining_texts), every_n):
            chunk_end = min(chunk_start + every_n, len(remaining_texts))
            chunk_texts = remaining_texts[chunk_start:chunk_end]

            chunk_emb = self.encode_texts(chunk_texts, show_progress=True)
            chunks.append(chunk_emb)

            # Save intermediate checkpoint
            if checkpoint_manager and checkpoint_path:
                progress_idx = start_idx + chunk_end
                all_so_far = np.concatenate(chunks, axis=0)
                checkpoint_manager.save_embeddings(
                    patent_ids[:progress_idx],
                    all_so_far,
                    checkpoint_path,
                    metadata={"model_name": self._model_name},
                )
                logger.info("Checkpoint saved: %s/%s patents", f"{progress_idx:,}", f"{len(patent_ids):,}")

        if chunks:
            all_embeddings = np.concatenate(chunks, axis=0)
        else:
            all_embeddings = np.empty((0, self._output_dim), dtype=np.float32)

        return patent_ids, all_embeddings

    def encode_texts_checkpointed(
        self,
        ids: list[str],
        texts: list[str],
        checkpoint_manager: CheckpointManager,
        checkpoint_path: str,
        checkpoint_every_n: int | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Encode plain texts with checkpoint-resume support.

        Same prefix-resume pattern as encode_patents: verify ID prefix match,
        encode remaining in chunks, save intermediate checkpoints.
        """
        if len(ids) != len(texts):
            raise ValueError(
                f"Length mismatch: {len(ids)} ids, {len(texts)} texts"
            )

        every_n = checkpoint_every_n or self._checkpoint_every_n

        # Resume from checkpoint if available
        start_idx = 0
        chunks = []
        if checkpoint_manager.checkpoint_exists(checkpoint_path):
            existing_ids, existing_emb, _ = checkpoint_manager.load_embeddings(checkpoint_path)
            expected_prefix = ids[:len(existing_ids)]
            if existing_ids != expected_prefix:
                raise ValueError(
                    f"Checkpoint ID mismatch: checkpoint contains {len(existing_ids)} "
                    f"texts starting with {existing_ids[:3]}, but current workload "
                    f"starts with {expected_prefix[:3]}. Cannot safely resume — delete "
                    f"the checkpoint file to restart from scratch."
                )
            start_idx = len(existing_ids)
            chunks.append(existing_emb)
            logger.info("Resuming from checkpoint: %s texts already encoded", f"{start_idx:,}")

        # Encode remaining texts in chunks of every_n
        remaining = texts[start_idx:]
        for chunk_start in range(0, len(remaining), every_n):
            chunk_end = min(chunk_start + every_n, len(remaining))
            chunk_texts = remaining[chunk_start:chunk_end]

            chunk_emb = self.encode_texts(chunk_texts, show_progress=True)
            chunks.append(chunk_emb)

            # Save intermediate checkpoint
            progress_idx = start_idx + chunk_end
            all_so_far = np.concatenate(chunks, axis=0)
            checkpoint_manager.save_embeddings(
                ids[:progress_idx],
                all_so_far,
                checkpoint_path,
                metadata={"model_name": self._model_name},
            )
            logger.info("Checkpoint saved: %s/%s texts", f"{progress_idx:,}", f"{len(ids):,}")

        if chunks:
            all_embeddings = np.concatenate(chunks, axis=0)
        else:
            all_embeddings = np.empty((0, self._output_dim), dtype=np.float32)

        return ids, all_embeddings
