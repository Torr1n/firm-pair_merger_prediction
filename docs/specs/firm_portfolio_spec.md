# Firm Portfolio Construction — Interface Specification

**Status**: Proposed  
**Date**: 2026-04-08  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (pending)  
**ADR Dependencies**: ADR-004 (K selection), ADR-005 (minimum patents), ADR-006 (covariance type), ADR-007 (normalization)

---

## Overview

This spec defines the interfaces for two modules that compose the Week 2 firm portfolio construction pipeline. Each module is independently testable, checkpointed, and follows the contracts below.

**Pipeline flow**:
```
patent_vectors_50d.parquet ──→ PortfolioBuilder.load_inputs() ──→ (vectors, gvkey_map)
                                        │
gvkey_map.parquet ──────────────────────┘
                                        │
                            PortfolioBuilder.group_by_firm() ──→ {gvkey: vectors_50d}
                                        │
                            PortfolioBuilder.classify_firms() ──→ tier assignments
                                        │
                            PortfolioBuilder.normalize() ──→ normalized vectors (if configured)
                                        │
                            GMMFitter.fit_all() ──→ per-firm GMM parameters
                                        │
                            GMMFitter.serialize() ──→ firm_gmm_parameters.parquet
```

**Input**: Week 1 outputs (`patent_vectors_50d.parquet`, `gvkey_map.parquet`)  
**Output**: `output/portfolios/firm_gmm_parameters.parquet` — one row per firm with serialized GMM parameters

---

## Module 1: PortfolioBuilder

**File**: `src/portfolio/portfolio_builder.py`

Handles loading Week 1 outputs, grouping patents by firm, applying tier thresholds, and optional normalization.

### Interface

```python
class PortfolioBuilder:
    def __init__(self, config: dict):
        """
        Args:
            config: Parsed YAML config dict. Expected keys under 'portfolio':
                - min_patents: int — firms below this are excluded (default: 5)
                - single_gaussian_max: int — upper bound for single-Gaussian tier (default: 49)
                - normalization: str — "raw", "l2", or "zscore" (default: "raw")
        """

    def load_inputs(
        self,
        vectors_path: str,
        gvkey_map_path: str,
    ) -> tuple[list[str], np.ndarray, pd.DataFrame]:
        """Load and validate Week 1 outputs.

        Validation:
            - patent_vectors_50d.parquet is loadable and has expected shape (N, 50)
            - gvkey_map.parquet has columns ['patent_id', 'gvkey']
            - All patent_ids in gvkey_map exist in patent_vectors_50d
            - Logs counts: unique patents, unique firms, co-assigned patents

        Args:
            vectors_path: Path to patent_vectors_50d.parquet
            gvkey_map_path: Path to gvkey_map.parquet

        Returns:
            Tuple of (patent_ids, vectors_50d, gvkey_map_df)
            - patent_ids: list[str] of patent IDs (from vectors file)
            - vectors_50d: np.ndarray shape (n_patents, 50), dtype float32
            - gvkey_map_df: pd.DataFrame with columns ['patent_id', 'gvkey']

        Raises:
            FileNotFoundError: If either file doesn't exist.
            ValueError: If shapes don't match or required columns are missing.
            ValueError: If gvkey_map contains patent_ids not in vectors file.
        """

    def group_by_firm(
        self,
        patent_ids: list[str],
        vectors: np.ndarray,
        gvkey_map: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Group patent vectors by firm (gvkey).

        Each firm's array contains the 50D vectors for all patents mapped to
        its gvkey. Co-assigned patents (same patent_id, multiple gvkeys) appear
        in full in each firm's array — no weight splitting (per bootstrap prompt).

        Args:
            patent_ids: Patent ID list aligned with vectors.
            vectors: np.ndarray shape (n_patents, 50).
            gvkey_map: DataFrame with columns ['patent_id', 'gvkey'].

        Returns:
            Dict mapping gvkey (str) -> np.ndarray shape (n_firm_patents, 50).
        """

    def classify_firms(
        self,
        firm_vectors: dict[str, np.ndarray],
    ) -> tuple[dict[str, str], list[str]]:
        """Classify firms into tiers based on patent count (per ADR-005).

        Tier assignments:
            - "exclude": n < min_patents (default: n < 5)
            - "single_gaussian": min_patents <= n <= single_gaussian_max (default: 5-49)
            - "gmm": n > single_gaussian_max (default: n >= 50)

        Args:
            firm_vectors: Dict from group_by_firm().

        Returns:
            Tuple of (tier_assignments, excluded_gvkeys)
            - tier_assignments: Dict mapping gvkey -> tier string
              ("single_gaussian" or "gmm"). Excluded firms are NOT in this dict.
            - excluded_gvkeys: List of gvkeys that were excluded (for logging).
        """

    def normalize(
        self,
        firm_vectors: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Apply pre-normalization to all firms' vectors (per ADR-007).

        If config normalization is "raw", returns input unchanged.
        If "l2", L2-normalizes each vector to unit length.
        If "zscore", z-score standardizes per dimension using global statistics
        (computed across ALL patents, not per firm).

        For z-score: global mean and std are computed from the union of all
        firm vectors on first call and cached. This ensures consistent scaling
        across firms.

        Args:
            firm_vectors: Dict mapping gvkey -> np.ndarray shape (n, 50).

        Returns:
            Dict mapping gvkey -> normalized np.ndarray shape (n, 50).
            If normalization is "raw", the arrays are the same objects (no copy).
        """

    def get_summary_stats(
        self,
        firm_vectors: dict[str, np.ndarray],
        tier_assignments: dict[str, str],
        excluded_gvkeys: list[str],
    ) -> dict:
        """Compute summary statistics for logging and validation.

        Returns:
            Dict with keys:
                - 'total_firms': int
                - 'excluded_firms': int
                - 'excluded_patents': int
                - 'single_gaussian_firms': int
                - 'single_gaussian_patents': int
                - 'gmm_firms': int
                - 'gmm_patents': int
                - 'co_assigned_patents': int (patents appearing in 2+ firms)
                - 'firm_size_percentiles': dict with p10, p25, p50, p75, p90, p99
        """
```

### Constraints

- Co-assigned patents count fully in each firm's portfolio (no weight splitting)
- Normalization statistics (for z-score) are computed globally, not per-firm
- `classify_firms` thresholds come from config, not hardcoded
- All patent_ids in gvkey_map must have corresponding vectors; orphaned mappings raise ValueError

---

## Module 2: GMMFitter

**File**: `src/portfolio/gmm_fitter.py`

Fits a Gaussian Mixture Model per firm and serializes the results.

### Interface

```python
@dataclass
class GMMResult:
    """Result of fitting a GMM to a single firm's patent vectors."""
    gvkey: str
    n_patents: int
    n_components: int          # Effective K (after pruning for Bayesian GMM)
    tier: str                  # "single_gaussian" or "gmm"
    covariance_type: str       # "diagonal" (per ADR-006)
    means: np.ndarray          # Shape (K, 50)
    covariances: np.ndarray    # Shape (K, 50) for diagonal
    weights: np.ndarray        # Shape (K,), sums to 1.0
    converged: bool
    lower_bound: float         # ELBO (variational lower bound on log-evidence)
    n_iter: int                # Number of EM iterations to converge


class GMMFitter:
    def __init__(self, config: dict):
        """
        Args:
            config: Parsed YAML config dict. Expected keys under 'portfolio':
                - gmm_method: "bayesian" or "bic_sweep" (default: "bayesian")
                - k_max: int — maximum components (default: 10)
                - covariance_type: "diag" (default, per ADR-006)
                - max_iter: int — EM max iterations (default: 200)
                - n_init: int — number of initializations (default: 5)
                - random_state: int (default: 42)
                - weight_pruning_threshold: float — components below this
                  weight are pruned (default: 0.01)
                - weight_concentration_prior: float — Dirichlet process
                  concentration γ (default: 1.0, per ADR-004 post-audit)
                - mean_precision_prior: float — κ₀ shrinkage toward global
                  mean (default: 1.0)
                - degrees_of_freedom_prior: int — ν₀ for Wishart/Gamma
                  prior (default: 52 = d+2)
                - checkpoint_every_n: int — save progress every N firms
                  (default: 1000)
        """

    def set_global_priors(
        self,
        all_vectors: np.ndarray,
    ) -> None:
        """Compute global empirical Bayes hyperparameters from pooled data.

        Must be called ONCE before fit_firm() or fit_all(). Computes
        global_mean and global_var from the pooled dataset of all patents
        (across all firms). These are used as fixed priors for every
        firm's independent fit, avoiding per-firm double-counting.

        This is a tractable approximation to a full hierarchical model
        (Type II maximum likelihood / global empirical Bayes). Small
        firms' estimates are shrunk toward the global structure; large
        firms are nearly unaffected.

        Args:
            all_vectors: np.ndarray shape (n_total_patents, 50) — the
                pooled vectors across ALL firms (before grouping).

        Sets:
            self.global_mean: np.ndarray shape (50,)
            self.global_var: np.ndarray shape (50,)
        """

    def fit_firm(
        self,
        gvkey: str,
        vectors: np.ndarray,
        tier: str,
    ) -> GMMResult:
        """Fit a GMM to a single firm's patent vectors.

        Behavior depends on tier:
            - "single_gaussian": Fits GaussianMixture with n_components=1
            - "gmm": Fits BayesianGaussianMixture with n_components=k_max,
              then prunes components with weight < weight_pruning_threshold

        For "gmm" tier, effective K is determined by counting components
        with weight above the pruning threshold. Pruned components are
        removed from the returned means, covariances, and weights. Remaining
        weights are renormalized to sum to 1.0.

        Args:
            gvkey: Firm identifier.
            vectors: np.ndarray shape (n_patents, 50), dtype float32.
            tier: "single_gaussian" or "gmm" (from PortfolioBuilder.classify_firms).

        Returns:
            GMMResult with fitted parameters.

        Raises:
            ValueError: If vectors has fewer rows than min_patents (shouldn't
                happen if classify_firms was applied correctly).
            ValueError: If tier is not "single_gaussian" or "gmm".
        """

    def fit_all(
        self,
        firm_vectors: dict[str, np.ndarray],
        tier_assignments: dict[str, str],
        checkpoint_path: str | None = None,
    ) -> list[GMMResult]:
        """Fit GMMs for all firms with checkpoint-resume support.

        Processes firms in sorted gvkey order (deterministic). Saves a
        checkpoint every checkpoint_every_n firms.

        Checkpoint format: partial firm_gmm_parameters.parquet containing
        results for all firms processed so far. On resume, loads the checkpoint
        and skips firms already present.

        Args:
            firm_vectors: Dict from PortfolioBuilder.group_by_firm()
                (should be filtered to exclude sub-threshold firms).
            tier_assignments: Dict from PortfolioBuilder.classify_firms().
            checkpoint_path: Path for checkpoint file. If None, no
                checkpointing (all results held in memory).

        Returns:
            List of GMMResult, one per firm (sorted by gvkey).
        """

    def serialize(
        self,
        results: list[GMMResult],
        output_path: str,
    ) -> Path:
        """Serialize GMM results to parquet.

        Output schema:
            - gvkey: string
            - n_patents: int32
            - n_components: int32
            - tier: string ("single_gaussian" or "gmm")
            - covariance_type: string ("diagonal")
            - means: binary (numpy .tobytes(), shape (K, 50), float64)
            - covariances: binary (numpy .tobytes(), shape (K, 50), float64)
            - weights: binary (numpy .tobytes(), shape (K,), float64)
            - converged: bool
            - lower_bound: float64 (ELBO from variational inference)
            - n_iter: int32

        File-level metadata in parquet schema:
            - 'gmm_method': str (e.g., "bayesian")
            - 'k_max': str(int)
            - 'covariance_type': str
            - 'weight_pruning_threshold': str(float)
            - 'weight_concentration_prior': str(float)
            - 'normalization': str
            - 'total_firms': str(int)
            - 'created_at': ISO 8601 timestamp

        Args:
            results: List of GMMResult from fit_all().
            output_path: Path for the output parquet file.

        Returns:
            Path to the saved parquet file.
        """

    @staticmethod
    def load(path: str) -> list[GMMResult]:
        """Load serialized GMM results from parquet.

        Deserializes binary columns back into numpy arrays. Returns a list
        of GMMResult in the same order as the file.

        Args:
            path: Path to firm_gmm_parameters.parquet.

        Returns:
            List of GMMResult, one per firm.

        Raises:
            FileNotFoundError: If path doesn't exist.
        """
```

### Constraints

- `set_global_priors()` MUST be called before any `fit_firm()` or `fit_all()` calls. It computes global_mean and global_var from the pooled dataset. These are passed to sklearn as `mean_prior` and `covariance_prior`, replacing sklearn's per-firm empirical Bayes defaults (which double-count data).
- Firms are processed in sorted gvkey order for deterministic output
- The `fit_firm` method is pure: given the same inputs, global priors, and random_state, it produces the same output
- Pruned components are removed from serialized output (only effective K components stored)
- Remaining weights are renormalized to sum to 1.0 after pruning
- ELBO (variational lower bound) is always stored. For single-Gaussian fits (using `GaussianMixture`, not Bayesian), `lower_bound` stores the log-likelihood instead. Both serve as model evidence proxies for validation.
- `reg_covar=1e-6` (scikit-learn default) is used for numerical stability
- For "gmm" tier with Bayesian method: if all components have weight below threshold (degenerate fit), fall back to single Gaussian (K=1) and log a warning
- Checkpoint parquet has the same schema as the final output — a partial checkpoint IS a valid output file for the firms it contains

### Bayesian Workflow Requirements (from STAT 405 audit)

The following workflow steps are **required**, not optional:

1. **Prior predictive simulation** (Phase 0, before any real data fitting): Draw 500+ parameter sets from the prior, simulate synthetic portfolios, verify plausibility (effective K distribution, per-dimension ranges, no degenerate draws).

2. **Pilot fit** (Phase 1, 200-300 stratified firms): Fit with calibrated priors, run full PPC battery, sensitivity sweep. Iterate prior revision if needed.

3. **Automated PPC scores** (Phase 2, all firms): Per-dimension KS statistics, pairwise distance Wasserstein, assignment entropy. Flag worst 2-5% for manual review.

4. **Sensitivity audit** (Phase 3, 200-500 stratified firms): Sweep γ, κ₀, ν₀. Stability criterion: effective K changes < 0.5 across one log-decade of γ for >90% of subsample.

5. **Optional MCMC validation**: Conjugate Gibbs sampling (Normal-Gamma for diagonal case) on ~500 highest-difficulty firms. If VI and Gibbs agree (Spearman r > 0.90 on parameter estimates), VI is validated.

---

## Configuration Additions

Added to `src/config/config.yaml` (post-Bayesian audit revision):

```yaml
# --- Portfolio Construction (Week 2) ---
portfolio:
  min_patents: 5                    # Firms below this are excluded (ADR-005)
  single_gaussian_max: 49           # Firms with min_patents..this get K=1 (ADR-005)
  gmm_method: "bayesian"            # "bayesian" or "bic_sweep" (ADR-004)
  k_max: 15                         # Operational default (ADR-004)
  k_max_sweep: [10, 15, 20]         # Produce artifacts at each value for sensitivity
  covariance_type: "diag"           # "diag" per ADR-006
  normalization: "raw"              # "raw", "l2", or "zscore" (ADR-007, pending EDA)
  weight_pruning_threshold: 0.01    # Components below this weight are pruned
  weight_concentration_prior: 1.0   # DP γ — E[K]≈γ·log(n) (ADR-004, post-audit)
  mean_precision_prior: 1.0         # κ₀: weakly informative mean shrinkage
  degrees_of_freedom_prior: 52      # ν₀ = d+2 for finite posterior mean
  random_state: 42
  max_iter: 200
  n_init: 5                         # Best of N initializations (post-audit)
  checkpoint_every_n: 1000          # Save progress every N firms

output_portfolios:
  checkpoint_dir: "output/portfolios"
  firm_gmm_parameters: "output/portfolios/firm_gmm_parameters.parquet"
  excluded_firms_log: "output/portfolios/excluded_firms.csv"
```

**Global empirical Bayes priors** (computed at runtime, not in config):
- `mean_prior = np.mean(X_pool, axis=0)` — global centroid across all firms
- `covariance_prior = np.var(X_pool, axis=0)` — global per-dimension variance
- These are computed once by `GMMFitter.set_global_priors()` from the pooled dataset

---

## Output File Inventory

The pipeline produces one GMM artifact per K_max value in the sensitivity sweep (config `portfolio.k_max_sweep`). The operational default (`portfolio.k_max`) determines which artifact is the primary output; the others are for Week 3 robustness reporting.

| File | Description | Schema |
|------|-------------|--------|
| `firm_gmm_parameters_k{N}.parquet` | Per-firm GMM parameters at K_max=N | See GMMFitter.serialize() above |
| `excluded_firms.csv` | Firms excluded due to insufficient patents | `gvkey, n_patents, reason` |

Example with default config (`k_max_sweep: [10, 15, 20]`):
- `output/portfolios/firm_gmm_parameters_k10.parquet`
- `output/portfolios/firm_gmm_parameters_k15.parquet` (primary)
- `output/portfolios/firm_gmm_parameters_k20.parquet`

Week 3 loads all sweep artifacts to classify firm pairs as **robust** or **model-sensitive**.

---

## Testing Strategy

Each module has a corresponding test file. Tests are written BEFORE implementation (TDD).

### test_portfolio_builder.py

- **load_inputs**: Loads synthetic Week 1 outputs; validates shapes and join integrity; raises ValueError for orphaned patent_ids in gvkey_map
- **group_by_firm**: Correct grouping; co-assigned patents appear in both firms' arrays; vectors match original patent vectors
- **classify_firms**: Correct tier assignment at each boundary (n=4 excluded, n=5 single_gaussian, n=49 single_gaussian, n=50 gmm); configurable thresholds
- **normalize**: Raw returns input unchanged; L2 produces unit-length vectors; z-score produces mean=0 std=1 per dimension (globally); z-score uses global stats not per-firm
- **get_summary_stats**: Correct counts, percentiles, co-assignment detection
- Uses synthetic data (hand-crafted patent_ids, vectors, gvkey_map)

### test_gmm_fitter.py

- **fit_firm (single_gaussian)**: Returns K=1, correct shapes, weights=[1.0], converged=True
- **fit_firm (gmm)**: Returns 1 ≤ K ≤ k_max; weights sum to 1.0; means shape (K, 50); covariances shape (K, 50); converged flag accurate
- **fit_firm (gmm, Bayesian pruning)**: With well-separated synthetic clusters, effective K matches true K
- **fit_firm (degenerate)**: All components pruned → falls back to K=1 with warning
- **fit_all**: Processes all firms; results sorted by gvkey; checkpoint file created
- **fit_all (resume)**: Partial checkpoint → resumes from last saved firm; final output matches non-checkpointed run
- **serialize/load round-trip**: All GMMResult fields preserved exactly; binary arrays deserialize to same dtype and shape
- **BIC computation**: BIC is finite and stored correctly for both tiers
- Uses synthetic 50D data with known cluster structure (2-3 Gaussians with known parameters)

---

## Week 3 Interface Contract

Week 3 consumes `firm_gmm_parameters.parquet` to compute pairwise Bhattacharyya Coefficients. The contract:

- Each row is one firm. The `n_components` field tells Week 3 how many components to expect in the deserialized arrays.
- `means` deserializes to shape `(n_components, 50)`, `covariances` to shape `(n_components, 50)` (diagonal), `weights` to shape `(n_components,)`.
- Week 3 computes BC between every pair of firms' GMMs using the component-wise BC formula aggregated by mixing weights.
- The `covariance_type` field allows Week 3 to select the correct BC closed-form (diagonal vs full, if full is added later).
- Firms not in this file were excluded and should not participate in pairwise comparison.

### K_max Sensitivity Requirement (ADR-004 Condition)

Week 2 must produce GMM fits at K_max ∈ {10, 15, 20} — not just the default K_max=15. Week 3 is contractually required to:

1. Compute BC rankings under all three K_max settings
2. Report firm pairs as **robust** (consistent ranking across K_max) or **model-sensitive** (ranking depends on K_max)
3. Any top-pair conclusions (e.g., "firms X and Y are most technologically similar") must be accompanied by K_max robustness classification

This is driven by EDA findings: BC Spearman ρ ≈ 0.77-0.81 across K_max settings, with top-50 pair overlap as low as 22%. Bulk rankings are directionally stable but the tail — where M&A candidate pairs live — is materially sensitive to K_max.
