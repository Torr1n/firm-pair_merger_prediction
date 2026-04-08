"""Tests for PatentLoader — parquet loading with validation."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data_loading.patent_loader import PatentLoader


@pytest.fixture
def data_dir(tmp_path):
    """Create minimal fixture parquet files (full + dedup)."""
    # patent_metadata (full — includes co-assignments: p1 linked to two gvkeys)
    pm = pa.table({
        "patent_id": ["p1", "p1", "p2", "p3"],
        "gvkey": ["001", "002", "001", "002"],
        "title": ["Title A", "Title A", "Title B", "Title C"],
        "abstract": ["Abs A", "Abs A", "Abs B", "Abs C"],
        "year": [2020, 2020, 2020, 2021],
    })
    pq.write_table(pm, str(tmp_path / "patent_metadata.parquet"))

    # patent_metadata_dedup (unique patent_ids, for encoding)
    pm_dedup = pa.table({
        "patent_id": ["p1", "p2", "p3"],
        "title": ["Title A", "Title B", "Title C"],
        "abstract": ["Abs A", "Abs B", "Abs C"],
        "post_deal_flag": [0, 0, 1],
    })
    pq.write_table(pm_dedup, str(tmp_path / "patent_metadata_dedup.parquet"))

    # cited_abstracts
    ca = pa.table({
        "patent_id": ["c1", "c2", "c3", "c4"],
        "abstract": ["Cited abs 1", "Cited abs 2", None, "Cited abs 4"],
    })
    pq.write_table(ca, str(tmp_path / "cited_abstracts.parquet"))

    # citation_network
    cn = pa.table({
        "patent_id": ["p1", "p1", "p2", "p3"],
        "citation_id": ["c1", "c2", "c3", "c4"],
    })
    pq.write_table(cn, str(tmp_path / "citation_network.parquet"))

    return tmp_path


@pytest.fixture
def config(data_dir):
    return {
        "data": {
            "patent_metadata": str(data_dir / "patent_metadata.parquet"),
            "patent_metadata_dedup": str(data_dir / "patent_metadata_dedup.parquet"),
            "cited_abstracts": str(data_dir / "cited_abstracts.parquet"),
            "citation_network": str(data_dir / "citation_network.parquet"),
        }
    }


@pytest.fixture
def loader(config):
    return PatentLoader(config)


class TestLoadPatentMetadata:
    def test_loads_default_columns(self, loader):
        # Full file has co-assignments (4 rows, p1 appears twice)
        with pytest.warns(UserWarning, match="duplicate"):
            df = loader.load_patent_metadata()
        assert {"patent_id", "gvkey", "title", "abstract", "year"} <= set(df.columns)
        assert len(df) == 4

    def test_loads_selected_columns(self, loader):
        with pytest.warns(UserWarning, match="duplicate"):
            df = loader.load_patent_metadata(columns=["patent_id", "title"])
        assert set(df.columns) == {"patent_id", "title"}

    def test_warns_on_duplicate_patent_ids(self, tmp_path):
        pm = pa.table({
            "patent_id": ["p1", "p1", "p2"],
            "title": ["A", "B", "C"],
            "abstract": ["a", "b", "c"],
        })
        pq.write_table(pm, str(tmp_path / "patent_metadata.parquet"))
        config = {"data": {
            "patent_metadata": str(tmp_path / "patent_metadata.parquet"),
            "patent_metadata_dedup": str(tmp_path / "patent_metadata.parquet"),
            "cited_abstracts": "dummy",
            "citation_network": "dummy",
        }}
        loader = PatentLoader(config)
        with pytest.warns(UserWarning, match="duplicate"):
            df = loader.load_patent_metadata()
        assert len(df) == 3  # all rows returned, caller decides dedup

    def test_missing_file_raises(self, tmp_path):
        config = {"data": {
            "patent_metadata": str(tmp_path / "missing.parquet"),
            "patent_metadata_dedup": str(tmp_path / "missing_dedup.parquet"),
            "cited_abstracts": "dummy",
            "citation_network": "dummy",
        }}
        loader = PatentLoader(config)
        with pytest.raises(FileNotFoundError):
            loader.load_patent_metadata()


class TestLoadCitedAbstracts:
    def test_loads_default_columns(self, loader):
        df = loader.load_cited_abstracts()
        assert set(df.columns) == {"patent_id", "abstract"}
        assert len(df) == 4

    def test_loads_selected_columns(self, loader):
        df = loader.load_cited_abstracts(columns=["patent_id"])
        assert set(df.columns) == {"patent_id"}


class TestLoadCitationNetwork:
    def test_loads_default_columns(self, loader):
        df = loader.load_citation_network()
        assert set(df.columns) == {"patent_id", "citation_id"}
        assert len(df) == 4

    def test_validates_no_nulls(self, tmp_path):
        cn = pa.table({
            "patent_id": ["p1", None, "p3"],
            "citation_id": ["c1", "c2", "c3"],
        })
        pq.write_table(cn, str(tmp_path / "citation_network.parquet"))
        config = {"data": {
            "patent_metadata": "dummy",
            "patent_metadata_dedup": "dummy",
            "cited_abstracts": "dummy",
            "citation_network": str(tmp_path / "citation_network.parquet"),
        }}
        loader = PatentLoader(config)
        with pytest.raises(ValueError, match="null"):
            loader.load_citation_network()


class TestGetRowCounts:
    def test_returns_correct_counts(self, loader):
        counts = loader.get_row_counts()
        assert counts["patent_metadata"] == 4  # full file has co-assignments
        assert counts["patent_metadata_dedup"] == 3
        assert counts["cited_abstracts"] == 4
        assert counts["citation_network"] == 4


class TestV3DataPatterns:
    """Tests for v3 data patterns: dedup source, co-assignments, post_deal_flag."""

    def test_loads_dedup_source(self, loader):
        """Loading source='dedup' reads the dedup file with unique patent_ids."""
        df = loader.load_patent_metadata(source="dedup")
        # Dedup file has 3 unique patents (no co-assignments)
        assert len(df) == 3
        assert df["patent_id"].is_unique

    def test_loads_full_source_with_coassignments(self, loader):
        """Loading source='full' returns all rows including co-assignments and warns."""
        with pytest.warns(UserWarning, match="duplicate"):
            df = loader.load_patent_metadata(source="full")
        # Full file has 4 rows (p1 appears twice for two gvkeys)
        assert len(df) == 4
        assert not df["patent_id"].is_unique

    def test_post_deal_flag_column_present(self, loader):
        """The dedup file includes post_deal_flag when loaded."""
        df = loader.load_patent_metadata(
            columns=["patent_id", "post_deal_flag"], source="dedup"
        )
        assert "post_deal_flag" in df.columns
        assert set(df["post_deal_flag"].unique()) <= {0, 1}

    def test_get_row_counts_includes_dedup(self, loader):
        """get_row_counts returns 4 keys including patent_metadata_dedup."""
        counts = loader.get_row_counts()
        assert len(counts) == 4
        assert "patent_metadata_dedup" in counts
        assert counts["patent_metadata_dedup"] == 3
