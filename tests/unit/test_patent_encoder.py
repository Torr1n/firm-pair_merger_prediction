"""Tests for PatentEncoder — PatentSBERTa encoding with batching and checkpointing."""

import numpy as np
import pytest

from src.embeddings.patent_encoder import PatentEncoder
from src.utils.checkpointing import CheckpointManager


@pytest.fixture(scope="module")
def config():
    return {
        "embedding": {
            "model_name": "AI-Growth-Lab/PatentSBERTa",
            "output_dim": 768,
            "batch_size": 32,
            "checkpoint_every_n": 5,
        }
    }


@pytest.fixture(scope="module")
def encoder(config):
    """Load model once for all tests in this module (expensive)."""
    return PatentEncoder(config)


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

    def test_checkpointing(self, encoder, tmp_path):
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
        # Final checkpoint should exist
        assert cm.checkpoint_exists(path)
