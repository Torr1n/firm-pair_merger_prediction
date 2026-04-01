"""UMAP dimensionality reduction for patent vectors."""

import numpy as np
import umap


class UMAPReducer:
    def __init__(self, config: dict):
        cfg = config["umap"]
        self._params = {
            "n_components": cfg["n_components"],
            "n_neighbors": cfg["n_neighbors"],
            "min_dist": cfg["min_dist"],
            "metric": cfg["metric"],
            "random_state": cfg["random_state"],
        }
        self._reducer = umap.UMAP(**self._params)

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.shape[0] < self._params["n_neighbors"]:
            raise ValueError(
                f"Need at least {self._params['n_neighbors']} samples "
                f"(n_neighbors), got {vectors.shape[0]}"
            )

        result = self._reducer.fit_transform(vectors)
        return result.astype(np.float32)

    def get_params(self) -> dict:
        return dict(self._params)
