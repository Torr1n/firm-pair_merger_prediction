"""Builds notebooks/04_pipeline_output_overview.ipynb via nbformat.

Run:
    source venv/bin/activate
    python3 scripts/build_notebook_04.py

The notebook is the teammate walkthrough for the Week 2 handover. It inlines all
BC helpers (no src/comparison/ dependency) and reads artifacts from the nested
corrected path + flat kmax_sweep path. Matches Notebook 03's narrative voice.

Re-running overwrites notebooks/04_pipeline_output_overview.ipynb. Execute the
notebook with `jupyter nbconvert --to notebook --execute --inplace` to populate
cell outputs before committing.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path("notebooks/04_pipeline_output_overview.ipynb")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    # ----- Title + purpose -----
    cells.append(md(
        "# Notebook 04: Pipeline Output Overview (Week 2 Handover)\n"
        "\n"
        "**Purpose**: This notebook is the teammate-facing walkthrough of the Week 2 "
        "deliverable. It shows you how to load the per-firm Gaussian Mixture Model "
        "parameters and the pairwise Bhattacharyya Coefficient (BC) matrix, how to "
        "compute BC between two firms by hand so you can trust the matrix values, "
        "how to find the top-k most-similar firms for any target firm, and what "
        "caveats are still in progress.\n"
        "\n"
        "The production dataset covers **7,485 deduplicated firms** in the technology "
        "and biotech sectors, fitted at **K_max=15** (the production lock; see ADR-004). "
        "The BC matrix uses the **linear-weighted formula** (bounded in [0, 1]; see the "
        "correctness note in Section 3).\n"
        "\n"
        "Plan on 4-6 hours to work through this notebook — Sections 3-4 are where the "
        "practical feel for the BC matrix comes from, and Section 7's caveats table is "
        "what you need before framing any regression.\n"
    ))

    # ----- Section 1: Setup -----
    cells.append(md(
        "## Section 1: Setup — placing the handoff bundle\n"
        "\n"
        "You should have received an artifact bundle from Torrin (email attachment or "
        "shared link). It contains 8 files plus `SHA256SUMS.txt`. Extract it and place "
        "the files as follows, from the repository root:\n"
        "\n"
        "**Into `output/kmax_sweep/corrected/output/kmax_sweep/`** (the nested path is "
        "intentional — it's the S3-sync layout; we document rather than flatten it):\n"
        "\n"
        "- `firm_gmm_parameters_k15.parquet` — primary per-firm GMM parameters (K_max=15)\n"
        "- `bc_matrix_all_k15_dedup_linear.npz` — primary pairwise BC matrix at K=15\n"
        "- `firm_gmm_parameters_k10.parquet` — convergence-floor reference (K_max=10)\n"
        "- `bc_matrix_all_k10_dedup_linear.npz` — reference BC matrix at K=10\n"
        "\n"
        "**Into `output/kmax_sweep/`:**\n"
        "\n"
        "- `deduplication_decisions.csv` — audit trail for 464 firms removed\n"
        "- `excluded_firms.csv` — firms removed for <5 patents\n"
        "- `coassignment_audit.parquet` — top-100 BC pair shared-patent audit\n"
        "\n"
        "**Verify transfer integrity** before running the notebook:\n"
        "\n"
        "```bash\n"
        "cd /path/to/extracted/bundle\n"
        "sha256sum -c SHA256SUMS.txt\n"
        "```\n"
        "\n"
        "Then launch Jupyter from the repo root and run all cells.\n"
    ))

    cells.append(code(
        "# Imports and global plot style (matches Notebook 03's conventions)\n"
        "import os\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams[\"figure.figsize\"] = (10, 5)\n"
        "\n"
        "# Auto-anchor to repo root so relative paths work whether you launch Jupyter\n"
        "# from the repo root (recommended) or from notebooks/\n"
        "if not Path(\"output/kmax_sweep\").exists() and Path(\"../output/kmax_sweep\").exists():\n"
        "    os.chdir(\"..\")\n"
        "    print(f\"Changed working directory to repo root: {os.getcwd()}\")\n"
        "\n"
        "# Artifact paths — the nested `corrected/output/kmax_sweep/` layout is a sync artifact\n"
        "# from the 2026-04-12 S3 re-deploy; we document rather than flatten it.\n"
        "CORRECTED = Path(\"output/kmax_sweep/corrected/output/kmax_sweep\")\n"
        "SWEEP = Path(\"output/kmax_sweep\")\n"
        "\n"
        "GMM_K15 = CORRECTED / \"firm_gmm_parameters_k15.parquet\"\n"
        "BC_K15 = CORRECTED / \"bc_matrix_all_k15_dedup_linear.npz\"\n"
        "GMM_K10 = CORRECTED / \"firm_gmm_parameters_k10.parquet\"\n"
        "BC_K10 = CORRECTED / \"bc_matrix_all_k10_dedup_linear.npz\"\n"
        "DEDUP_CSV = SWEEP / \"deduplication_decisions.csv\"\n"
        "EXCLUDED_CSV = SWEEP / \"excluded_firms.csv\"\n"
        "COASSIGN_PARQUET = SWEEP / \"coassignment_audit.parquet\"\n"
        "\n"
        "# Fail early with a helpful message if teammates haven't placed the bundle files yet\n"
        "for p in [GMM_K15, BC_K15, GMM_K10, BC_K10, DEDUP_CSV, EXCLUDED_CSV, COASSIGN_PARQUET]:\n"
        "    assert p.exists(), f\"Missing {p}. Did you place the bundle files per Section 1?\"\n"
        "\n"
        "print(\"All bundle files present. Ready to load.\")\n"
    ))

    cells.append(md(
        "### Firm name mapping\n"
        "\n"
        "We hardcode a small mapping from `gvkey` to firm name for the 5 deep-dive firms "
        "used in Notebook 03's convergence analysis. These are the only names we verify "
        "against the deduplicated gvkey set; teammates can extend the dict with other "
        "Compustat-verified gvkeys of economic interest (the top-15 firms by patent count "
        "are printed in Section 2 for reference).\n"
    ))

    cells.append(code(
        "# Verified against Notebook 03's DEEP_DIVE_FIRMS (all 5 present in the K=15\n"
        "# deduplicated BC matrix). Extend with Compustat-verified gvkeys as needed;\n"
        "# do not guess — unknown firms should remain as their gvkey.\n"
        "FIRM_NAMES = {\n"
        "    \"006066\": \"IBM\",\n"
        "    \"012141\": \"Intel\",\n"
        "    \"024800\": \"Qualcomm\",\n"
        "    \"160329\": \"Google / Alphabet\",\n"
        "    \"020779\": \"Cisco Systems\",\n"
        "}\n"
    ))

    # ----- Section 2: Loading -----
    cells.append(md(
        "## Section 2: Loading the data\n"
        "\n"
        "Two primary artifacts back every downstream analysis:\n"
        "\n"
        "- **`firm_gmm_parameters_k15.parquet`** — one row per firm. Columns include "
        "`gvkey`, `n_patents`, `n_components` (the effective K after Dirichlet-process "
        "pruning), `tier` (`single_gaussian` or `gmm`), plus three binary blobs for "
        "means, covariances, and mixture weights. The deserializer below unpacks them.\n"
        "- **`bc_matrix_all_k15_dedup_linear.npz`** — a dense 7485×7485 symmetric "
        "float64 matrix of pairwise BC values with diagonal = 1.0 (self-similarity).\n"
        "\n"
        "Note: the GMM parquet contains **all 7,949 fitted firms** (pre-dedup), while "
        "the BC matrix covers only the **7,485 deduplicated firms**. We filter the GMM "
        "results to the BC matrix's `gvkeys` so indexing stays consistent.\n"
    ))

    cells.append(code(
        "def load_bc_matrix(path: Path) -> tuple[list[str], np.ndarray]:\n"
        "    \"\"\"Load a corrected BC matrix .npz archive.\n"
        "\n"
        "    Returns (gvkeys, bc_matrix). The matrix is symmetric float64 with diagonal 1.0.\n"
        "    \"\"\"\n"
        "    data = np.load(path, allow_pickle=True)\n"
        "    gvkeys = [str(g) for g in data[\"gvkeys\"]]\n"
        "    bc_matrix = data[\"bc_matrix\"]\n"
        "    assert bc_matrix.shape == (len(gvkeys), len(gvkeys)), \"gvkey/matrix shape mismatch\"\n"
        "    return gvkeys, bc_matrix\n"
        "\n"
        "\n"
        "def load_gmm_results(path: Path) -> dict[str, dict]:\n"
        "    \"\"\"Load per-firm GMM parameters from parquet; return dict keyed by gvkey.\"\"\"\n"
        "    df = pd.read_parquet(path)\n"
        "    d = 50  # UMAP output dimensionality\n"
        "    lookup = {}\n"
        "    for _, row in df.iterrows():\n"
        "        k = int(row[\"n_components\"])\n"
        "        lookup[str(row[\"gvkey\"])] = {\n"
        "            \"gvkey\": str(row[\"gvkey\"]),\n"
        "            \"n_patents\": int(row[\"n_patents\"]),\n"
        "            \"n_components\": k,\n"
        "            \"tier\": row[\"tier\"],\n"
        "            \"means\": np.frombuffer(row[\"means\"], dtype=np.float64).reshape(k, d),\n"
        "            \"covariances\": np.frombuffer(row[\"covariances\"], dtype=np.float64).reshape(k, d),\n"
        "            \"weights\": np.frombuffer(row[\"weights\"], dtype=np.float64).reshape(k),\n"
        "        }\n"
        "    return lookup\n"
        "\n"
        "\n"
        "gvkeys, bc_matrix = load_bc_matrix(BC_K15)\n"
        "gvkey_to_idx = {gv: i for i, gv in enumerate(gvkeys)}\n"
        "gmm_lookup = load_gmm_results(GMM_K15)\n"
        "\n"
        "print(f\"BC matrix: {bc_matrix.shape}, diagonal mean = {np.diag(bc_matrix).mean():.6f}\")\n"
        "print(f\"GMM parameters: {len(gmm_lookup):,} firms pre-dedup\")\n"
        "print(f\"Deduplicated set (BC matrix rows): {len(gvkeys):,} firms\")\n"
    ))

    cells.append(code(
        "# Sanity checks — if any fail, the artifacts are corrupted or mismatched.\n"
        "assert bc_matrix.shape[0] == bc_matrix.shape[1] == len(gvkeys), \"BC matrix is not square\"\n"
        "assert np.allclose(bc_matrix, bc_matrix.T, atol=1e-12), \"BC matrix is not symmetric\"\n"
        "assert np.allclose(np.diag(bc_matrix), 1.0, atol=1e-9), \"BC diagonal is not all 1.0\"\n"
        "assert (bc_matrix >= 0).all() and (bc_matrix <= 1 + 1e-9).all(), \"BC values outside [0, 1]\"\n"
        "\n"
        "# Every BC-matrix firm should have GMM parameters\n"
        "missing = [gv for gv in gvkeys if gv not in gmm_lookup]\n"
        "assert not missing, f\"{len(missing)} firms in BC matrix but missing GMM params\"\n"
        "print(\"All sanity checks pass.\")\n"
    ))

    cells.append(md(
        "### Summary statistics — effective K distribution and tier breakdown\n"
    ))

    cells.append(code(
        "# Summarize effective K for deduplicated firms only (BC matrix members)\n"
        "dedup_gmm = pd.DataFrame(\n"
        "    [{\"gvkey\": gv, \"n_patents\": gmm_lookup[gv][\"n_patents\"],\n"
        "      \"n_components\": gmm_lookup[gv][\"n_components\"], \"tier\": gmm_lookup[gv][\"tier\"]}\n"
        "     for gv in gvkeys]\n"
        ")\n"
        "\n"
        "print(\"Tier distribution (K=15 deduplicated):\")\n"
        "print(dedup_gmm[\"tier\"].value_counts().to_string())\n"
        "\n"
        "print(\"\\nEffective K distribution:\")\n"
        "print(dedup_gmm[\"n_components\"].value_counts().sort_index().to_string())\n"
        "\n"
        "print(\"\\nTop 15 firms by patent count (useful for identifying Compustat gvkeys):\")\n"
        "top_firms = dedup_gmm.nlargest(15, \"n_patents\")[[\"gvkey\", \"n_patents\", \"n_components\", \"tier\"]]\n"
        "top_firms[\"name\"] = top_firms[\"gvkey\"].map(FIRM_NAMES).fillna(\"(unknown — verify via Compustat)\")\n"
        "print(top_firms.to_string(index=False))\n"
    ))

    # ----- Section 3: Worked example -----
    cells.append(md(
        "## Section 3: Worked example — IBM vs Intel\n"
        "\n"
        "This section computes the BC between IBM (gvkey 006066) and Intel (gvkey "
        "012141) two ways — by matrix lookup and by re-deriving the formula from "
        "their GMM parameters — and asserts they match to float64 tolerance. If this "
        "assertion fails, either the matrix is the old (buggy) √-weighted variant or "
        "the formula has been modified; see the correctness note below.\n"
        "\n"
        "**Correctness note**. The BC formula uses **linear** mixing weights πᵢπⱼ, "
        "not the √-weighted variant `√(πᵢ·πⱼ)`. The √-weighted variant is "
        "mathematically an upper bound that exceeds 1.0 for multi-component mixtures "
        "(observed up to 5.39 in an earlier iteration of the sweep), and it was the "
        "root cause of the original K_max=15→20 ranking instability we caught during "
        "the 2026-04-12 correction. The corrected linear formula is bounded in [0, 1] "
        "by Cauchy-Schwarz, and it is the production formula throughout this notebook "
        "and the shipped BC matrix.\n"
    ))

    cells.append(code(
        "def bc_component_matrix(mu_a: np.ndarray, var_a: np.ndarray,\n"
        "                       mu_b: np.ndarray, var_b: np.ndarray) -> np.ndarray:\n"
        "    \"\"\"BC between all component pairs of two GMMs (diagonal covariance, vectorized).\n"
        "\n"
        "    Closed form under diagonal covariance (ADR-006):\n"
        "        D_B = (1/8) Σ_d (μᵢ_d - μⱼ_d)² / σ̄²_d\n"
        "            + (1/2) Σ_d ln(σ̄²_d / √(σ²ᵢ_d · σ²ⱼ_d))\n"
        "        BC  = exp(-D_B)\n"
        "    where σ̄²_d = (σ²ᵢ_d + σ²ⱼ_d) / 2.\n"
        "\n"
        "    Returns (K_A, K_B) float64 matrix of component-pair BC values in [0, 1].\n"
        "    \"\"\"\n"
        "    sigma_avg = (var_a[:, None, :] + var_b[None, :, :]) / 2.0   # (K_A, K_B, D)\n"
        "    diff = mu_a[:, None, :] - mu_b[None, :, :]                  # (K_A, K_B, D)\n"
        "    mahal = 0.125 * np.sum(diff**2 / sigma_avg, axis=2)         # (K_A, K_B)\n"
        "    log_det_avg = np.sum(np.log(sigma_avg), axis=2)             # (K_A, K_B)\n"
        "    log_det_a = np.sum(np.log(var_a), axis=1)                   # (K_A,)\n"
        "    log_det_b = np.sum(np.log(var_b), axis=1)                   # (K_B,)\n"
        "    det_term = 0.5 * (log_det_avg - 0.5 * (log_det_a[:, None] + log_det_b[None, :]))\n"
        "    return np.exp(-(mahal + det_term))\n"
        "\n"
        "\n"
        "def bc_mixture_linear(gmm_a: dict, gmm_b: dict) -> float:\n"
        "    \"\"\"Mixture-level BC with linear πᵢπⱼ weights (bounded in [0, 1]).\n"
        "\n"
        "        BC(A, B) = Σᵢ Σⱼ πᵢᴬ · πⱼᴮ · BC(Nᵢᴬ, Nⱼᴮ)\n"
        "\n"
        "    Do NOT use √(πᵢπⱼ) — that is an upper bound that can exceed 1 for multi-\n"
        "    component mixtures and caused the original K_max top-tail instability bug.\n"
        "    \"\"\"\n"
        "    bc_grid = bc_component_matrix(\n"
        "        gmm_a[\"means\"], gmm_a[\"covariances\"],\n"
        "        gmm_b[\"means\"], gmm_b[\"covariances\"],\n"
        "    )\n"
        "    weight_grid = gmm_a[\"weights\"][:, None] * gmm_b[\"weights\"][None, :]\n"
        "    return float(np.sum(weight_grid * bc_grid))\n"
    ))

    cells.append(code(
        "# Compare IBM and Intel two ways: matrix lookup vs formula recomputation\n"
        "gv_ibm, gv_intel = \"006066\", \"012141\"\n"
        "i_ibm, i_intel = gvkey_to_idx[gv_ibm], gvkey_to_idx[gv_intel]\n"
        "\n"
        "bc_from_matrix = float(bc_matrix[i_ibm, i_intel])\n"
        "bc_from_formula = bc_mixture_linear(gmm_lookup[gv_ibm], gmm_lookup[gv_intel])\n"
        "\n"
        "print(f\"BC(IBM, Intel) via matrix lookup   = {bc_from_matrix:.12f}\")\n"
        "print(f\"BC(IBM, Intel) via formula         = {bc_from_formula:.12f}\")\n"
        "print(f\"Absolute difference                = {abs(bc_from_matrix - bc_from_formula):.2e}\")\n"
        "\n"
        "# This assertion is the notebook's reproducibility anchor. If it ever fails,\n"
        "# the matrix and the formula have drifted out of agreement — do not trust\n"
        "# downstream analysis until the discrepancy is resolved.\n"
        "assert abs(bc_from_matrix - bc_from_formula) < 1e-9, \\\n"
        "    \"Matrix lookup disagrees with formula recomputation — investigate!\"\n"
        "print(\"\\nReproducibility anchor: matrix == formula (within float64 tolerance).\")\n"
    ))

    # ----- Section 4: Top-k partners -----
    cells.append(md(
        "## Section 4: Top-k partners for a given firm\n"
        "\n"
        "The most common downstream operation is \"for firm X, who are its top-k most "
        "technologically similar firms?\". The helper below returns a ranked DataFrame "
        "with partner gvkey, known name (or the gvkey itself if unknown), BC value, "
        "effective K, and patent count.\n"
        "\n"
        "Heads-up on naming: private firms in Compustat have synthetic gvkeys prefixed "
        "`PRIV_` (e.g., `PRIV_ENDOLOGIX`). They are legitimate firms, not dedup leaks. "
        "The dedup rule (containment ≥ 0.95) was applied, and 464 aliases/subsidiaries/"
        "predecessors were already removed; see `deduplication_decisions.csv` for the "
        "audit trail.\n"
    ))

    cells.append(code(
        "def top_k_partners(query_gvkey: str, k: int = 20) -> pd.DataFrame:\n"
        "    \"\"\"Return the top-k firms most similar to query_gvkey by BC value.\"\"\"\n"
        "    if query_gvkey not in gvkey_to_idx:\n"
        "        raise KeyError(\n"
        "            f\"{query_gvkey} is not in the 7,485-firm deduplicated set. \"\n"
        "            \"Check deduplication_decisions.csv or excluded_firms.csv.\"\n"
        "        )\n"
        "    idx = gvkey_to_idx[query_gvkey]\n"
        "    row = bc_matrix[idx].copy()\n"
        "    row[idx] = -np.inf  # exclude self\n"
        "    top_idx = np.argpartition(-row, k)[:k]\n"
        "    top_idx = top_idx[np.argsort(-row[top_idx])]\n"
        "    records = []\n"
        "    for rank, j in enumerate(top_idx, start=1):\n"
        "        partner = gvkeys[j]\n"
        "        gmm = gmm_lookup.get(partner, {})\n"
        "        records.append({\n"
        "            \"rank\": rank,\n"
        "            \"gvkey\": partner,\n"
        "            \"name\": FIRM_NAMES.get(partner, partner),\n"
        "            \"bc\": float(row[j]),\n"
        "            \"effective_k\": gmm.get(\"n_components\"),\n"
        "            \"n_patents\": gmm.get(\"n_patents\"),\n"
        "        })\n"
        "    return pd.DataFrame(records)\n"
        "\n"
        "\n"
        "print(\"=== Top-20 partners for IBM (gvkey 006066) ===\")\n"
        "print(top_k_partners(\"006066\", k=20).to_string(index=False))\n"
    ))

    cells.append(code(
        "print(\"=== Top-20 partners for Google / Alphabet (gvkey 160329) ===\")\n"
        "print(top_k_partners(\"160329\", k=20).to_string(index=False))\n"
    ))

    # ----- Section 5: Distribution + sanity plots -----
    cells.append(md(
        "## Section 5: Distribution and sanity plots\n"
        "\n"
        "Three quick visualizations ground our intuition for what the BC matrix looks "
        "like globally: the distribution of BC values across all off-diagonal pairs, "
        "the relationship between BC and firm-size product, and the effective-K "
        "distribution by tier.\n"
    ))

    cells.append(code(
        "# Plot 1: BC value distribution (off-diagonal upper triangle, log y-axis)\n"
        "iu = np.triu_indices(len(gvkeys), k=1)\n"
        "off_diag_bc = bc_matrix[iu]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "ax.hist(off_diag_bc, bins=100, color=\"steelblue\", edgecolor=\"white\")\n"
        "ax.set_yscale(\"log\")\n"
        "ax.set_xlabel(\"Bhattacharyya Coefficient (linear-weighted)\")\n"
        "ax.set_ylabel(\"Number of firm pairs (log scale)\")\n"
        "ax.set_title(\n"
        "    f\"Distribution of BC values across {len(off_diag_bc):,} off-diagonal pairs\\n\"\n"
        "    f\"median = {np.median(off_diag_bc):.4f}, 99th percentile = {np.quantile(off_diag_bc, 0.99):.4f}\",\n"
        "    fontweight=\"bold\",\n"
        ")\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(code(
        "# Plot 2: BC vs firm-size-product (5,000-pair random sample for readability)\n"
        "rng = np.random.default_rng(42)\n"
        "sample_size = 5000\n"
        "sample_i = rng.integers(0, len(gvkeys), size=sample_size)\n"
        "sample_j = rng.integers(0, len(gvkeys), size=sample_size)\n"
        "mask = sample_i != sample_j\n"
        "sample_i, sample_j = sample_i[mask], sample_j[mask]\n"
        "n_patents_arr = np.array([gmm_lookup[gv][\"n_patents\"] for gv in gvkeys])\n"
        "size_product = n_patents_arr[sample_i] * n_patents_arr[sample_j]\n"
        "bc_sample = bc_matrix[sample_i, sample_j]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "ax.scatter(size_product, bc_sample, s=4, alpha=0.3, color=\"darkblue\")\n"
        "ax.set_xscale(\"log\")\n"
        "ax.set_xlabel(\"n_patents_a × n_patents_b (log scale)\")\n"
        "ax.set_ylabel(\"BC value\")\n"
        "ax.set_title(\n"
        "    f\"BC vs firm-size product ({len(bc_sample):,}-pair random sample, seed=42)\",\n"
        "    fontweight=\"bold\",\n"
        ")\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    cells.append(code(
        "# Plot 3: Effective K distribution by tier\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "for tier, color in [(\"single_gaussian\", \"lightgray\"), (\"gmm\", \"steelblue\")]:\n"
        "    subset = dedup_gmm[dedup_gmm[\"tier\"] == tier][\"n_components\"]\n"
        "    ax.hist(subset, bins=np.arange(0.5, 17.5, 1.0), label=f\"{tier} (n={len(subset):,})\",\n"
        "            color=color, edgecolor=\"white\", alpha=0.85)\n"
        "ax.set_xlabel(\"Effective K (post-DP-pruning mixture components)\")\n"
        "ax.set_ylabel(\"Number of firms\")\n"
        "ax.set_title(\"Effective K distribution by tier (K_max=15)\", fontweight=\"bold\")\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))

    # ----- Section 6: Co-assignment caveat -----
    cells.append(md(
        "## Section 6: Co-assignment caveat (important for regression design)\n"
        "\n"
        "A firm pair can have high BC for two reasons: (a) their patent portfolios "
        "genuinely occupy similar technological regions (the signal we want), or "
        "(b) they share a lot of patents already — joint ventures, subsidiaries missed "
        "by our containment-based dedup, or long-running collaborations. We ran a "
        "co-assignment audit on the top-100 BC pairs to quantify how much of (b) is in "
        "there. The parquet `output/kmax_sweep/coassignment_audit.parquet` has the "
        "full per-pair breakdown; aggregate stats follow.\n"
    ))

    cells.append(code(
        "audit = pd.read_parquet(COASSIGN_PARQUET)\n"
        "assert len(audit) == 100, f\"Expected 100 rows, got {len(audit)}\"\n"
        "\n"
        "median_shared = int(audit[\"n_shared\"].median())\n"
        "mean_jaccard = float(audit[\"jaccard\"].mean())\n"
        "n_ge10 = int((audit[\"overlap_fraction\"] > 0.10).sum())\n"
        "n_ge25 = int((audit[\"overlap_fraction\"] > 0.25).sum())\n"
        "n_zero = int((audit[\"n_shared\"] == 0).sum())\n"
        "\n"
        "print(\"Top-100 BC pair co-assignment audit (K_max=15, linear-weighted):\")\n"
        "print(f\"  Median shared patent count: {median_shared}\")\n"
        "print(f\"  Mean Jaccard similarity:    {mean_jaccard:.4f}\")\n"
        "print(f\"  Pairs with >10% overlap:    {n_ge10} / 100\")\n"
        "print(f\"  Pairs with >25% overlap:    {n_ge25} / 100\")\n"
        "print(f\"  Pairs with 0 shared patents: {n_zero} / 100\")\n"
        "\n"
        "print(\"\\nTop 5 pairs by overlap_fraction (candidate dedup misses to eyeball):\")\n"
        "print(audit.nlargest(5, \"overlap_fraction\")[\n"
        "    [\"rank\", \"gvkey_a\", \"gvkey_b\", \"bc\", \"n_shared\", \"overlap_fraction\"]\n"
        "].to_string(index=False))\n"
    ))

    cells.append(md(
        "### Interpretation\n"
        "\n"
        "**The dedup rule worked well.** Of the top-100 BC pairs, **98 share zero "
        "patents** and only **2 exceed 10% overlap**. Mean Jaccard is 0.014 — "
        "essentially zero. BC is substantially independent of co-assignment structure "
        "for the top-tier pairs; the signal is genuinely distributional rather than "
        "structural.\n"
        "\n"
        "**Two outliers are worth flagging by name.** Rank 37 (`060888` + "
        "`PRIV_OBLONGINDUSTRIES`, 94% overlap) and rank 20 (`063083` + "
        "`PRIV_ENDOLOGIX`, 75% overlap) appear to be parent-subsidiary relationships "
        "that fell just below the 0.95 containment threshold. If your regressions rank "
        "top-20 or top-50 partners per firm, you will encounter these; treat them as "
        "known dedup misses rather than genuine tech-neighbour findings.\n"
        "\n"
        "**Recommendation for regressions.** Include `n_shared` (or `jaccard`) from "
        "the audit parquet as a control covariate when running BC-based predictors. "
        "The effect size should be small given the audit numbers, but it is defensive "
        "against the 2 outliers and against the long tail of <top-100 pairs we did "
        "not audit.\n"
    ))

    # ----- Section 7: Caveats + roadmap table -----
    cells.append(md(
        "## Section 7: Known caveats and open items\n"
        "\n"
        "The dataset is ready for economic analysis. Several methodology and "
        "engineering items remain open and are shipped as promises, not finished work. "
        "The caveats table is honest about what has been checked and what is still in "
        "progress; the team email and README mirror it.\n"
        "\n"
        "| Item | Status | Expected delivery |\n"
        "|---|---|---|\n"
        "| K_max=15 production lock | ✓ Done (ADR-004, 2026-04-14) | — |\n"
        "| Deduplication (containment ≥ 0.95) | ✓ Done (464 firms removed; see `deduplication_decisions.csv`) | — |\n"
        "| Linear-weighted BC formula | ✓ Done (corrected from √-weighted variant, 2026-04-12) | — |\n"
        "| Convergence sweep (K_max ∈ {10,15,20,25,30}) | ✓ Done (Spearman ρ=0.991-0.993, top-50 overlap 96-100%; see Notebook 03) | — |\n"
        "| Co-assignment audit (top-100) | ✓ Done (this notebook, Section 6) | — |\n"
        "| BC spec (Codex review) | Reviewed; revisions in progress; **not yet approved for implementation** | Week 1 |\n"
        "| BC module TDD (`src/comparison/bhattacharyya.py`) | Not started — production logic lives in `scripts/recompute_bc_corrected.py` | Week 2 |\n"
        "| PortfolioBuilder / GMMFitter TDD (`src/portfolio/`) | Not started — production logic in `scripts/run_kmax_sweep.py` | Week 2-3 |\n"
        "| Pruning-threshold audit | Not started | Week 1-2 |\n"
        "| Gaussian adequacy audit | Not started | Week 1-2 |\n"
        "| Directional complementarity (ADR-008 → v2 dataset) | Not started | Week 2-4 |\n"
        "| Hyperparameter sensitivity (γ, κ₀, ν₀) | Contingent on Gaussian adequacy passing | Week 3-4 |\n"
        "\n"
        "**Weeks are measured from 2026-04-15.** Each open item has its own acceptance "
        "criteria which will be posted as we go; ask in your regression work if any of "
        "these are blocking you and we can re-prioritize.\n"
    ))

    # ----- Section 8: Where to find things -----
    cells.append(md(
        "## Section 8: Where to find things\n"
        "\n"
        "**In this repository:**\n"
        "\n"
        "- `src/config/config.yaml` — all pipeline hyperparameters (K_max, prior, "
        "covariance type, etc.)\n"
        "- `docs/adr/` — architecture decision records, especially:\n"
        "  - `adr_004_k_selection_method.md` — K_max=15 production lock rationale\n"
        "  - `adr_005_bayesian_prior_global_empirical.md` — prior choice\n"
        "  - `adr_006_diagonal_covariance.md` — covariance structure\n"
        "  - `adr_007_normalization.md` — normalization decision\n"
        "- `docs/specs/firm_portfolio_spec.md` — GMM fitting contract\n"
        "- `docs/specs/comparison_spec.md` — BC module spec (reviewed; revisions in progress)\n"
        "- `docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md` — decision narrative\n"
        "- `docs/epics/week2_firm_portfolios/coassignment_audit_summary.md` — Section 6 summary\n"
        "- `notebooks/03_kmax_convergence_analysis.ipynb` — full convergence story (run before this one if you want the methodology defense)\n"
        "- `scripts/recompute_bc_corrected.py` — production source for the linear-weighted BC formula\n"
        "- `scripts/coassignment_audit.py` — Section 6 audit source (re-runnable)\n"
        "\n"
        "**Outside the repo:**\n"
        "\n"
        "- Raw PatentSBERTa embeddings live on S3 at "
        "`s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/` — "
        "contact Torrin for access.\n"
        "- Corrected sweep outputs (what's in your bundle) live at "
        "`s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/`.\n"
        "\n"
        "**When you hit a wall**: contact Torrin directly. For methodology questions "
        "not covered by the ADRs or specs, flag them and they'll become new ADRs.\n"
    ))

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }
    return nb


def main() -> None:
    nb = build()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {NB_PATH} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
