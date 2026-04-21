"""
Compute technology complementarity features between all firm pairs.

Two features are computed and saved together:

  1. dissimilarity  = 1 - BC(A,B)  [symmetric]
     BC(A,B) = sum_k p_{A,k} * p_{B,k}
     Measures how different two firms' portfolios are. Symmetric by construction
     (mathematically identical to 1 - dot(p_A, p_B) when weights sum to 1).
     Useful alongside BC as a dissimilarity feature in the downstream model.

  2. directional_comp = Comp(A->B)  [asymmetric]
     Comp(A->B) = sum_k max(0, p_market_k - p_{A,k}) * p_{B,k}
     where p_market_k = fraction of ALL patents globally in zone k.
     Measures how well B's strengths fill A's *market-relative* gaps —
     zones where A is underweight compared to the overall patent landscape.
     Genuinely asymmetric: Comp(A->B) != Comp(B->A) because A's and B's
     market-relative gaps differ. Bounded in [0, 1].

Pipeline:
  - Fit K-means (K=K_GLOBAL) on all 1.4M patent embeddings -> shared technology zones
  - Each firm's portfolio = fraction of its patents in each zone (sums to 1)
  - Compute both features via vectorised matrix operations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

EMBEDDINGS_DIR = Path.home() / "Desktop" / "output" / "week2_inputs"
KMAX_DIR       = Path.home() / "Desktop" / "output" / "kmax_sweep"
RESULTS_DIR    = Path(__file__).parent.parent / "output" / "comparison"

K_GLOBAL   = 50   # global technology zones
MIN_PATENTS = 5   # minimum patents to include a firm (matches existing threshold)

DEEP_DIVE_FIRMS = {
    "006066": "IBM",
    "012141": "Intel",
    "024800": "Qualcomm",
    "160329": "Google/Alphabet",
    "020779": "Cisco Systems",
}


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load patent vectors. Returns (patent_ids, embeddings) arrays."""
    print("  Reading parquet...")
    df = pd.read_parquet(path, columns=["patent_id", "embedding"])
    print("  Decoding embeddings...")
    X = np.vstack(df["embedding"].apply(lambda b: np.frombuffer(b, dtype=np.float32)).values)
    return df["patent_id"].values, X


def fit_global_kmeans(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Fit MiniBatchKMeans on all patent embeddings. Returns cluster labels (n_patents,)."""
    print(f"  Fitting MiniBatchKMeans K={k} on {len(X):,} patents...")
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3, verbose=0)
    labels = km.fit_predict(X)
    print(f"  Done. Cluster sizes: min={np.bincount(labels).min():,}  max={np.bincount(labels).max():,}")
    return labels


def build_firm_weights(
    patent_ids: np.ndarray,
    labels: np.ndarray,
    gvkey_map: pd.DataFrame,
    excluded: set,
    k: int,
    min_patents: int,
) -> tuple[np.ndarray, list[str]]:
    """
    For each firm, compute the fraction of its patents in each global cluster.
    Returns P (n_firms, k) and sorted gvkey list.
    """
    df = pd.DataFrame({"patent_id": patent_ids, "label": labels})
    df = df.merge(gvkey_map, on="patent_id", how="inner")
    df["gvkey"] = df["gvkey"].astype(str)

    # Filter excluded firms
    df = df[~df["gvkey"].isin(excluded)]

    # Count patents per firm per cluster
    counts = df.groupby(["gvkey", "label"]).size().unstack(fill_value=0)

    # Apply minimum patent threshold
    counts = counts[counts.sum(axis=1) >= min_patents]

    # Ensure all K columns present (some clusters may be absent for small datasets)
    for col in range(k):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[list(range(k))]

    # Normalize to probabilities
    P = counts.values.astype(np.float64)
    P = P / P.sum(axis=1, keepdims=True)

    gvkeys = counts.index.tolist()
    return P, gvkeys


def compute_matrices(
    P: np.ndarray, p_market: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BC[i,j]        = sum_k p_{i,k} * p_{j,k}                         (symmetric)
    dissimilarity  = 1 - BC                                            (symmetric)
    comp_dir[i,j]  = sum_k max(0, p_market_k - p_{i,k}) * p_{j,k}   (asymmetric)

    comp_dir uses market-relative gaps: zones where firm i is underweight relative
    to the global patent distribution. Comp(A->B) != Comp(B->A) because different
    firms have different market-relative gap profiles.
    """
    BC   = P @ P.T
    diss = 1.0 - BC

    gaps = np.maximum(0.0, p_market[None, :] - P)  # (n, K) — A's market-relative gaps
    comp_dir = gaps @ P.T                           # (n, n) — Comp(A->B)

    return BC, diss, comp_dir


def validate(BC: np.ndarray, diss: np.ndarray, comp_dir: np.ndarray):
    for name, M in [("BC", BC), ("diss", diss), ("comp_dir", comp_dir)]:
        assert M.min() >= -1e-6 and M.max() <= 1 + 1e-6, \
            f"{name} out of [0,1]: [{M.min():.4f}, {M.max():.4f}]"

    max_asym = float(np.abs(comp_dir - comp_dir.T).max())
    sym_check = np.allclose(BC, BC.T, atol=1e-8) and np.allclose(diss, diss.T, atol=1e-8)
    print(f"BC + diss symmetric: {sym_check}")
    print(f"comp_dir max |M[i,j]-M[j,i]|: {max_asym:.6f}  (0 = symmetric, >0 = directional)")

    for name, M in [("BC", BC), ("diss", diss), ("comp_dir", comp_dir)]:
        np.fill_diagonal(M.view(), np.nan)
        flat = M[~np.isnan(M)]
        np.fill_diagonal(M.view(), 0)
        print(f"{name:10s}  range [{flat.min():.3f}, {flat.max():.3f}]  "
              f"p50={np.percentile(flat,50):.3f}  p99={np.percentile(flat,99):.3f}  "
              f"at_zero={(flat < 1e-4).mean():.3f}  at_one={(flat > 0.9999).mean():.3f}")


def main():
    print("Loading deduplication list...")
    excluded = set(pd.read_csv(KMAX_DIR / "deduplication_decisions.csv")["dropped"].astype(str))
    print(f"  Excluding {len(excluded)} firms")

    print("Loading patent embeddings...")
    patent_ids, X = load_embeddings(EMBEDDINGS_DIR / "patent_vectors_50d.parquet")
    print(f"  {len(X):,} patents, shape {X.shape}")

    print("Loading gvkey map...")
    gvkey_map = pd.read_parquet(EMBEDDINGS_DIR / "gvkey_map.parquet", columns=["patent_id", "gvkey"])
    print(f"  {len(gvkey_map):,} patent-firm assignments")

    print(f"Fitting global K-means (K={K_GLOBAL})...")
    labels = fit_global_kmeans(X, K_GLOBAL)

    # Global patent distribution — fraction of all patents in each zone
    p_market = np.bincount(labels, minlength=K_GLOBAL).astype(np.float64)
    p_market /= p_market.sum()
    print(f"  Market distribution: min={p_market.min():.3f}  max={p_market.max():.3f}")

    print("Building per-firm weight vectors...")
    P, gvkeys = build_firm_weights(patent_ids, labels, gvkey_map, excluded, K_GLOBAL, MIN_PATENTS)
    print(f"  {len(gvkeys)} firms, P shape: {P.shape}")

    print("Computing BC, dissimilarity, and directional complementarity matrices...")
    BC, diss, comp_dir = compute_matrices(P, p_market)

    print("Validating...")
    validate(BC, diss, comp_dir)

    gvkey_to_idx = {gv: i for i, gv in enumerate(gvkeys)}

    print("\nSpot checks (comp_dir(A->B) should differ from comp_dir(B->A)):")
    pairs = [("006066", "012141"), ("012141", "006066"),
             ("006066", "024800"), ("024800", "006066")]
    for a, b in pairs:
        if a in gvkey_to_idx and b in gvkey_to_idx:
            i, j = gvkey_to_idx[a], gvkey_to_idx[b]
            print(f"  {DEEP_DIVE_FIRMS.get(a,a):20s} -> {DEEP_DIVE_FIRMS.get(b,b):20s} "
                  f"comp_dir={comp_dir[i,j]:.4f}  comp_dir_rev={comp_dir[j,i]:.4f}  BC={BC[i,j]:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "complementarity_matrix_global_k50.npz"
    np.savez_compressed(
        out,
        gvkeys=np.array(gvkeys),
        bc_matrix=BC,
        dissimilarity_matrix=diss,
        comp_dir_matrix=comp_dir,
        p_market=p_market,
        k_global=np.array(K_GLOBAL),
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
