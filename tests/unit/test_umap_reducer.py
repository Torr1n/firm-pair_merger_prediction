"""Tests for UMAPReducer — dimensionality reduction from 1536D to 50D."""

import numpy as np
import pytest

from src.dimensionality_reduction.umap_reducer import UMAPReducer


@pytest.fixture
def config():
    return {
        "umap": {
            "n_components": 10,  # Small for fast tests
            "n_neighbors": 5,
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": 42,
        }
    }


@pytest.fixture
def reducer(config):
    return UMAPReducer(config)


@pytest.fixture
def sample_vectors():
    rng = np.random.RandomState(99)
    return rng.randn(50, 100).astype(np.float32)


class TestFitTransform:
    def test_output_shape(self, reducer, sample_vectors):
        result = reducer.fit_transform(sample_vectors)
        assert result.shape == (50, 10)

    def test_output_dtype(self, reducer, sample_vectors):
        result = reducer.fit_transform(sample_vectors)
        assert result.dtype == np.float32

    def test_reproducible(self, config, sample_vectors):
        r1 = UMAPReducer(config)
        r2 = UMAPReducer(config)
        result1 = r1.fit_transform(sample_vectors)
        result2 = r2.fit_transform(sample_vectors)
        np.testing.assert_array_equal(result1, result2)

    def test_too_few_samples_raises(self, config):
        config["umap"]["n_neighbors"] = 10
        reducer = UMAPReducer(config)
        small = np.random.randn(5, 20).astype(np.float32)
        with pytest.raises(ValueError):
            reducer.fit_transform(small)


class TestGetParams:
    def test_returns_config(self, reducer):
        params = reducer.get_params()
        assert params["n_components"] == 10
        assert params["n_neighbors"] == 5
        assert params["metric"] == "cosine"
        assert params["random_state"] == 42
