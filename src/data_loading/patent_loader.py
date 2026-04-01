"""Parquet loading with validation for patent data files."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


class PatentLoader:
    # Columns to always exclude (pandas index artifacts)
    _EXCLUDE_COLS = {"__index_level_0__"}

    def __init__(self, config: dict):
        data_cfg = config["data"]
        self._paths = {
            "patent_metadata": data_cfg["patent_metadata"],
            "cited_abstracts": data_cfg["cited_abstracts"],
            "citation_network": data_cfg["citation_network"],
        }

    def _read(self, key: str, columns: list[str] | None, required: list[str]) -> pd.DataFrame:
        path = Path(self._paths[key])
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        schema = pq.read_schema(str(path))
        available = {f.name for f in schema} - self._EXCLUDE_COLS

        # Validate required columns exist
        missing = set(required) - available
        if missing:
            raise ValueError(f"{key} missing required columns: {missing}")

        # Determine columns to load
        if columns is None:
            read_cols = sorted(available)
        else:
            read_cols = [c for c in columns if c not in self._EXCLUDE_COLS]

        return pq.read_table(str(path), columns=read_cols).to_pandas()

    def load_patent_metadata(self, columns: list[str] | None = None) -> pd.DataFrame:
        # Always load patent_id for validation even if not requested
        required = ["patent_id", "title", "abstract"]
        if columns is not None:
            validation_cols = list(set(columns) | {"patent_id"})
        else:
            # Load all available columns (filtered by _read)
            validation_cols = None
        df = self._read("patent_metadata", validation_cols, required=required)

        if df["patent_id"].isna().any():
            raise ValueError("patent_metadata contains null patent_ids")
        if df["patent_id"].duplicated().any():
            raise ValueError("patent_metadata contains duplicate patent_ids")

        # Return only the requested columns
        if columns is not None:
            df = df[[c for c in columns if c in df.columns]]
        return df

    def load_cited_abstracts(self, columns: list[str] | None = None) -> pd.DataFrame:
        return self._read("cited_abstracts", columns, required=["patent_id", "abstract"])

    def load_citation_network(self, columns: list[str] | None = None) -> pd.DataFrame:
        default_cols = columns or ["patent_id", "citation_id"]
        validation_cols = list(set(default_cols) | {"patent_id", "citation_id"})
        df = self._read("citation_network", validation_cols, required=["patent_id", "citation_id"])

        if df["patent_id"].isna().any() or df["citation_id"].isna().any():
            raise ValueError("citation_network contains null values in patent_id or citation_id")

        if columns is not None:
            df = df[[c for c in columns if c in df.columns]]
        else:
            df = df[default_cols]
        return df

    def get_row_counts(self) -> dict[str, int]:
        counts = {}
        for key, path_str in self._paths.items():
            meta = pq.read_metadata(path_str)
            counts[key] = meta.num_rows
        return counts
