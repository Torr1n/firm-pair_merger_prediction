"""Tests for CitationAggregator — mean pooling with zero-citation handling."""

import numpy as np
import pandas as pd
import pytest

from src.embeddings.citation_aggregator import CitationAggregator


@pytest.fixture
def config():
    return {
        "citation_aggregation": {
            "method": "mean_pooling",
            "zero_citation_strategy": "zero_vector",
        }
    }


@pytest.fixture
def aggregator(config):
    return CitationAggregator(config)


@pytest.fixture
def sample_lookup():
    """4 cited patents with known embeddings."""
    rng = np.random.RandomState(42)
    ids = ["c1", "c2", "c3", "c4"]
    embs = rng.randn(4, 768).astype(np.float32)
    return ids, embs


@pytest.fixture
def sample_network():
    """Patent p1 cites c1,c2; p2 cites c3; p3 has no citations."""
    return pd.DataFrame({
        "patent_id": ["p1", "p1", "p2"],
        "citation_id": ["c1", "c2", "c3"],
    })


class TestBuildCitationLookup:
    def test_creates_correct_mapping(self, aggregator, sample_lookup):
        ids, embs = sample_lookup
        lookup = aggregator.build_citation_lookup(ids, embs)
        assert len(lookup) == 4
        np.testing.assert_array_equal(lookup["c1"], embs[0])

    def test_length_mismatch_raises(self, aggregator):
        with pytest.raises(ValueError):
            aggregator.build_citation_lookup(
                ["a", "b"], np.zeros((3, 10), dtype=np.float32)
            )


class TestAggregate:
    def test_mean_pooling_correctness(self, aggregator, sample_lookup, sample_network):
        ids, embs = sample_lookup
        lookup = aggregator.build_citation_lookup(ids, embs)

        patent_ids = ["p1", "p2", "p3"]
        result_ids, result_emb = aggregator.aggregate(
            patent_ids, sample_network, lookup
        )

        assert result_ids == patent_ids
        assert result_emb.shape == (3, 768)
        assert result_emb.dtype == np.float32

        # p1 = mean(c1, c2)
        expected_p1 = (embs[0] + embs[1]) / 2
        np.testing.assert_allclose(result_emb[0], expected_p1, rtol=1e-5)

        # p2 = c3 (single citation)
        np.testing.assert_allclose(result_emb[1], embs[2], rtol=1e-5)

        # p3 = zero vector (no citations)
        np.testing.assert_array_equal(result_emb[2], np.zeros(768, dtype=np.float32))

    def test_zero_citation_patent(self, aggregator):
        lookup = {"c1": np.ones(768, dtype=np.float32)}
        network = pd.DataFrame({"patent_id": ["other"], "citation_id": ["c1"]})

        _, result = aggregator.aggregate(["lonely"], network, lookup)
        np.testing.assert_array_equal(result[0], np.zeros(768, dtype=np.float32))

    def test_citation_not_in_lookup_skipped(self, aggregator):
        """Patent cites c1 and c_missing. Only c1 is in lookup. Mean = c1."""
        lookup = {"c1": np.ones(768, dtype=np.float32) * 2.0}
        network = pd.DataFrame({
            "patent_id": ["p1", "p1"],
            "citation_id": ["c1", "c_missing"],
        })

        _, result = aggregator.aggregate(["p1"], network, lookup)
        expected = np.ones(768, dtype=np.float32) * 2.0
        np.testing.assert_allclose(result[0], expected, rtol=1e-5)

    def test_all_citations_missing_from_lookup(self, aggregator):
        """Patent has citations but none are in lookup — treated as zero-citation."""
        lookup = {}
        network = pd.DataFrame({
            "patent_id": ["p1"],
            "citation_id": ["c_missing"],
        })

        _, result = aggregator.aggregate(["p1"], network, lookup)
        np.testing.assert_array_equal(result[0], np.zeros(768, dtype=np.float32))

    def test_missing_columns_raises(self, aggregator):
        with pytest.raises(ValueError, match="columns"):
            aggregator.aggregate(
                ["p1"],
                pd.DataFrame({"bad_col": [1]}),
                {},
            )


class TestCoverageStats:
    def test_coverage_stats(self, aggregator, sample_lookup, sample_network):
        ids, embs = sample_lookup
        lookup = aggregator.build_citation_lookup(ids, embs)
        patent_ids = ["p1", "p2", "p3"]

        stats = aggregator.get_coverage_stats(patent_ids, sample_network, lookup)
        assert stats["total_patents"] == 3
        assert stats["zero_citation_patents"] == 1
        assert stats["total_edges"] == 3
        assert stats["edges_with_embeddings"] == 3
