"""Tests for CheckpointManager — save/load embedding parquet files."""

import numpy as np
import pytest
from pathlib import Path

from src.utils.checkpointing import CheckpointManager


@pytest.fixture
def checkpoint_dir(tmp_path):
    return str(tmp_path / "checkpoints")


@pytest.fixture
def manager(checkpoint_dir):
    return CheckpointManager(checkpoint_dir)


@pytest.fixture
def sample_data():
    patent_ids = ["pat_001", "pat_002", "pat_003"]
    embeddings = np.random.RandomState(42).randn(3, 768).astype(np.float32)
    return patent_ids, embeddings


class TestSaveAndLoad:
    def test_round_trip_preserves_data(self, manager, sample_data, tmp_path):
        patent_ids, embeddings = sample_data
        path = str(tmp_path / "test.parquet")

        manager.save_embeddings(patent_ids, embeddings, path)
        loaded_ids, loaded_emb, meta = manager.load_embeddings(path)

        assert loaded_ids == patent_ids
        np.testing.assert_array_equal(loaded_emb, embeddings)
        assert loaded_emb.dtype == np.float32

    def test_metadata_preserved(self, manager, sample_data, tmp_path):
        patent_ids, embeddings = sample_data
        path = str(tmp_path / "test.parquet")
        extra_meta = {"model_name": "PatentSBERTa", "stage": "title_abstract"}

        manager.save_embeddings(patent_ids, embeddings, path, metadata=extra_meta)
        _, _, meta = manager.load_embeddings(path)

        assert meta["embedding_dim"] == "768"
        assert meta["row_count"] == "3"
        assert meta["model_name"] == "PatentSBERTa"
        assert meta["stage"] == "title_abstract"
        assert "created_at" in meta

    def test_single_patent(self, manager, tmp_path):
        patent_ids = ["single"]
        embeddings = np.ones((1, 50), dtype=np.float32)
        path = str(tmp_path / "single.parquet")

        manager.save_embeddings(patent_ids, embeddings, path)
        loaded_ids, loaded_emb, _ = manager.load_embeddings(path)

        assert loaded_ids == ["single"]
        assert loaded_emb.shape == (1, 50)

    def test_length_mismatch_raises(self, manager, tmp_path):
        path = str(tmp_path / "bad.parquet")
        with pytest.raises(ValueError):
            manager.save_embeddings(
                ["a", "b"], np.zeros((3, 10), dtype=np.float32), path
            )


class TestCheckpointExists:
    def test_exists_for_valid_file(self, manager, sample_data, tmp_path):
        patent_ids, embeddings = sample_data
        path = str(tmp_path / "exists.parquet")
        manager.save_embeddings(patent_ids, embeddings, path)

        assert manager.checkpoint_exists(path) is True

    def test_not_exists_for_missing_file(self, manager):
        assert manager.checkpoint_exists("/nonexistent/path.parquet") is False

    def test_not_exists_for_non_parquet(self, tmp_path):
        bad_file = tmp_path / "not_parquet.txt"
        bad_file.write_text("hello")
        manager = CheckpointManager(str(tmp_path))
        assert manager.checkpoint_exists(str(bad_file)) is False
