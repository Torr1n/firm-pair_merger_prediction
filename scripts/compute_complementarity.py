"""
Compute directional technology complementarity matrix from GMM firm weights.

Comp(A->B) = sum_k (1 - p_{A,k}) * p_{B,k}

Measures how well B's technological strengths fill A's gaps.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "output" / "comparison"

DEEP_DIVE_FIRMS = {
    "006066": "IBM",
    "012141": "Intel",
    "024800": "Qualcomm",
    "160329": "Google/Alphabet",
    "020779": "Cisco Systems",
}


def load_gmm_results(path: Path) -> list[dict]:
    df = pd.read_parquet(path)
    records = df.to_dict("records")
    for r in records:
        if isinstance(r["weights"], (bytes, bytearray)):
            r["weights"] = np.frombuffer(r["weights"], dtype=np.float32).tolist()
        elif isinstance(r["weights"], str):
            import json
            r["weights"] = json.loads(r["weights"])
        r["gvkey"] = str(r["gvkey"])
    return records


def build_weight_matrix(gmm_records: list[dict]) -> tuple[np.ndarray, list[str]]:
    gvkeys = [r["gvkey"] for r in gmm_records]
    K = max(len(r["weights"]) for r in gmm_records)
    P = np.zeros((len(gvkeys), K), dtype=np.float32)
    for i, r in enumerate(gmm_records):
        w = r["weights"]
        P[i, : len(w)] = w
    return P, gvkeys


def compute_matrices(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    one_minus_P = 1.0 - P
    Comp = one_minus_P @ P.T  # Comp[i,j] = Comp(i->j)
    BC = P @ P.T               # symmetric similarity
    return Comp, BC


def validate(Comp: np.ndarray, BC: np.ndarray):
    assert np.allclose(np.diag(Comp), 0, atol=1e-5), "Self-complementarity should be 0"
    assert Comp.min() >= -1e-5 and Comp.max() <= 1 + 1e-5, "Comp values out of [0,1]"
    asymmetric = not np.allclose(Comp, Comp.T, atol=1e-4)
    print(f"Directional (Comp != Comp.T): {asymmetric}")
    print(f"Comp range: [{Comp.min():.4f}, {Comp.max():.4f}]")
    print(f"BC  range:  [{BC.min():.4f},  {BC.max():.4f}]")


def main():
    dedup_path = DATA_DIR / "deduplication_decisions.csv"
    gmm_path = DATA_DIR / "firm_gmm_parameters_k10.parquet"

    print("Loading deduplication list...")
    dedup_df = pd.read_csv(dedup_path)
    excluded = set(dedup_df["dropped"].astype(str).values)
    print(f"  Excluding {len(excluded)} firms")

    print("Loading GMM results...")
    raw = load_gmm_results(gmm_path)
    gmm = [r for r in raw if r["gvkey"] not in excluded]
    print(f"  {len(gmm)} firms after deduplication (from {len(raw)})")

    print("Building weight matrix...")
    P, gvkeys = build_weight_matrix(gmm)
    print(f"  P shape: {P.shape}")

    print("Computing complementarity and BC matrices...")
    Comp, BC = compute_matrices(P)

    print("Validating...")
    validate(Comp, BC)

    gvkey_to_idx = {gvkey: i for i, gvkey in enumerate(gvkeys)}

    print("\nSpot checks:")
    pairs = [("006066", "012141"), ("012141", "006066")]
    for a, b in pairs:
        if a in gvkey_to_idx and b in gvkey_to_idx:
            i, j = gvkey_to_idx[a], gvkey_to_idx[b]
            print(f"  Comp({DEEP_DIVE_FIRMS.get(a, a)}->{DEEP_DIVE_FIRMS.get(b, b)}): {Comp[i,j]:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "complementarity_matrix_k10.npz"
    np.savez_compressed(out, gvkeys=np.array(gvkeys), comp_matrix=Comp, bc_matrix=BC)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
