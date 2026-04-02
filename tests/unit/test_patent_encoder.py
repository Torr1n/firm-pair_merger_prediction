"""Tests for PatentEncoder — PatentSBERTa encoding with batching and checkpointing.

These tests load the real PatentSBERTa model (not mocked) and require network
access on first run to download model weights. Subsequent runs use the HF cache.
If the model is unavailable, all tests in this module are skipped.
"""

import numpy as np
import pytest

from src.utils.checkpointing import CheckpointManager

try:
    from src.embeddings.patent_encoder import PatentEncoder

    _CONFIG = {
        "embedding": {
            "model_name": "AI-Growth-Lab/PatentSBERTa",
            "output_dim": 768,
            "batch_size": 32,
            "checkpoint_every_n": 5,
        }
    }
    _ENCODER = PatentEncoder(_CONFIG)
    _MODEL_AVAILABLE = True
except Exception:
    _MODEL_AVAILABLE = False
    _CONFIG = None
    _ENCODER = None

pytestmark = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="PatentSBERTa model not available (offline or not cached)",
)


@pytest.fixture
def config():
    return _CONFIG


@pytest.fixture
def encoder():
    return _ENCODER


class TestEncodeTexts:
    def test_output_shape(self, encoder):
        texts = ["A method for data processing", "Semiconductor device"]
        result = encoder.encode_texts(texts, show_progress=False)
        assert result.shape == (2, 768)

    def test_output_dtype(self, encoder):
        result = encoder.encode_texts(["test patent"], show_progress=False)
        assert result.dtype == np.float32

    def test_deterministic(self, encoder):
        texts = ["Wireless communication method"]
        r1 = encoder.encode_texts(texts, show_progress=False)
        r2 = encoder.encode_texts(texts, show_progress=False)
        np.testing.assert_array_equal(r1, r2)

    def test_empty_list_raises(self, encoder):
        with pytest.raises(ValueError):
            encoder.encode_texts([], show_progress=False)

    def test_batch_size_override(self, encoder):
        texts = ["text"] * 10
        result = encoder.encode_texts(texts, batch_size=3, show_progress=False)
        assert result.shape == (10, 768)


class TestEncodePatents:
    def test_basic_encoding(self, encoder):
        ids = ["p1", "p2", "p3"]
        titles = ["Method A", "Device B", "System C"]
        abstracts = ["Abstract about A", "Abstract about B", "Abstract about C"]

        result_ids, result_emb = encoder.encode_patents(ids, titles, abstracts)
        assert result_ids == ids
        assert result_emb.shape == (3, 768)
        assert result_emb.dtype == np.float32

    def test_null_abstract_uses_title(self, encoder):
        ids = ["p1", "p2"]
        titles = ["Method A", "Device B"]
        abstracts = [None, ""]

        result_ids, result_emb = encoder.encode_patents(ids, titles, abstracts)
        assert result_emb.shape == (2, 768)
        # Should not be zero vectors (title is still encoded)
        assert np.linalg.norm(result_emb[0]) > 0
        assert np.linalg.norm(result_emb[1]) > 0

    def test_length_mismatch_raises(self, encoder):
        with pytest.raises(ValueError):
            encoder.encode_patents(["p1"], ["t1", "t2"], ["a1"])


class TestCheckpointing:
    def test_checkpoint_saved(self, encoder, tmp_path):
        cm = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "test_checkpoint.parquet")

        ids = [f"p{i}" for i in range(10)]
        titles = [f"Title {i}" for i in range(10)]
        abstracts = [f"Abstract {i}" for i in range(10)]

        result_ids, result_emb = encoder.encode_patents(
            ids, titles, abstracts,
            checkpoint_manager=cm,
            checkpoint_path=path,
            checkpoint_every_n=5,
        )

        assert result_ids == ids
        assert result_emb.shape == (10, 768)
        assert cm.checkpoint_exists(path)

    def test_resume_produces_same_result(self, encoder, tmp_path):
        """Encode, then re-call with same IDs — should resume from checkpoint."""
        cm = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "resume_test.parquet")

        ids = [f"p{i}" for i in range(8)]
        titles = [f"Title {i}" for i in range(8)]
        abstracts = [f"Abstract {i}" for i in range(8)]

        # First run
        _, emb1 = encoder.encode_patents(
            ids, titles, abstracts,
            checkpoint_manager=cm, checkpoint_path=path, checkpoint_every_n=4,
        )

        # Second run — should resume from checkpoint, produce identical output
        _, emb2 = encoder.encode_patents(
            ids, titles, abstracts,
            checkpoint_manager=cm, checkpoint_path=path, checkpoint_every_n=4,
        )

        np.testing.assert_array_equal(emb1, emb2)

    def test_resume_with_mismatched_ids_raises(self, encoder, tmp_path):
        """If checkpoint IDs don't match the current workload prefix, raise."""
        cm = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "mismatch_test.parquet")

        # First run with IDs [a, b, c]
        encoder.encode_patents(
            ["a", "b", "c"], ["T1", "T2", "T3"], ["A1", "A2", "A3"],
            checkpoint_manager=cm, checkpoint_path=path, checkpoint_every_n=10,
        )

        # Second run with different ID order — should raise
        with pytest.raises(ValueError, match="mismatch"):
            encoder.encode_patents(
                ["x", "y", "z", "w"], ["T1", "T2", "T3", "T4"], ["A1", "A2", "A3", "A4"],
                checkpoint_manager=cm, checkpoint_path=path,
            )

    def test_intermediate_checkpoints_saved(self, encoder, tmp_path, monkeypatch):
        """With checkpoint_every_n=4 and 10 patents, save_embeddings is called
        3 times (chunks of 4, 4, 2) — not once at the end."""
        cm = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "intermediate_test.parquet")

        save_calls = []
        original_save = cm.save_embeddings

        def tracking_save(*args, **kwargs):
            save_calls.append(len(args[0]))  # track patent count per save
            return original_save(*args, **kwargs)

        monkeypatch.setattr(cm, "save_embeddings", tracking_save)

        ids = [f"p{i}" for i in range(10)]
        titles = [f"Title {i}" for i in range(10)]
        abstracts = [f"Abstract {i}" for i in range(10)]

        encoder.encode_patents(
            ids, titles, abstracts,
            checkpoint_manager=cm, checkpoint_path=path, checkpoint_every_n=4,
        )

        # Should have saved 3 times: after 4, 8, and 10 patents
        assert len(save_calls) == 3
        assert save_calls == [4, 8, 10]

        # Final checkpoint has all 10
        loaded_ids, loaded_emb, _ = cm.load_embeddings(path)
        assert len(loaded_ids) == 10
        assert loaded_emb.shape == (10, 768)
