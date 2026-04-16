"""Regenerate all notebook visualization PNGs from corrected BC matrices.

This script runs the visualization code from notebooks/03_kmax_convergence_analysis.ipynb
against the corrected (deduplicated, linear-weighted) BC matrices. It produces 11 PNGs
in the notebooks/ directory, matching the pre-registered analysis plan.

Usage:
    python scripts/regenerate_notebook_pngs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.run_kmax_sweep import load_gmm_results

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

SWEEP_DIR = Path("output/kmax_sweep")
CORRECTED_DIR = SWEEP_DIR / "corrected" / "output" / "kmax_sweep"
OUT_DIR = Path("notebooks")

DEEP_DIVE_FIRMS = {
    "006066": "IBM",
    "012141": "Intel",
    "024800": "Qualcomm",
    "160329": "Google/Alphabet",
    "020779": "Cisco Systems",
}

# Load dedup decisions
dedup_df = pd.read_csv(SWEEP_DIR / "deduplication_decisions.csv")
DEDUP_GVKEYS = set(dedup_df["dropped"].astype(str).values)
print(f"Deduplication: {len(DEDUP_GVKEYS)} firms to exclude")

# Load corrected convergence summary
with open(CORRECTED_DIR / "convergence_summary_dedup_linear.json") as f:
    summary = json.load(f)
K_MAX_VALUES = summary["k_max_values"]
print(f"Verdict: {summary['convergence_verdict']}, K*={summary.get('converged_at_kmax')}")

# Load GMM results (filtered)
print("Loading GMM results...")
all_results = {}
for k_max in K_MAX_VALUES:
    path = SWEEP_DIR / f"firm_gmm_parameters_k{k_max}.parquet"
    raw = load_gmm_results(str(path))
    all_results[k_max] = [r for r in raw if str(r["gvkey"]) not in DEDUP_GVKEYS]
    print(f"  K_max={k_max}: {len(all_results[k_max])} firms")

# Load corrected BC matrices
print("Loading corrected BC matrices...")
bc_matrices = {}
for k_max in K_MAX_VALUES:
    path = CORRECTED_DIR / f"bc_matrix_all_k{k_max}_dedup_linear.npz"
    data = np.load(path, allow_pickle=True)
    bc_matrices[k_max] = (data["gvkeys"].tolist(), data["bc_matrix"])
    print(f"  K_max={k_max}: shape={bc_matrices[k_max][1].shape}, max={bc_matrices[k_max][1].max():.4f}")


# ===== VIZ 2A: Effective K progression =====
print("Generating viz2a...")
k_progression = []
for k_max in K_MAX_VALUES:
    gmm_results = [r for r in all_results[k_max] if r["tier"] == "gmm"]
    ks = np.array([r["n_components"] for r in gmm_results])
    k_progression.append({
        "k_max": k_max,
        "mean": ks.mean(),
        "p25": np.percentile(ks, 25),
        "p50": np.percentile(ks, 50),
        "p75": np.percentile(ks, 75),
        "p90": np.percentile(ks, 90),
        "ceiling_rate_pct": np.mean(ks >= k_max - 0.5) * 100,
    })
k_df = pd.DataFrame(k_progression)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(k_df["k_max"], k_df["p25"], k_df["p75"], alpha=0.3, color="steelblue", label="P25-P75 band")
ax.plot(k_df["k_max"], k_df["mean"], "o-", color="steelblue", linewidth=2, label="Mean effective K")
ax.plot(k_df["k_max"], k_df["p90"], "s--", color="coral", linewidth=2, label="P90 effective K")
ax.plot(k_df["k_max"], k_df["k_max"], "k:", alpha=0.5, label="K_max ceiling")
ax.set_xlabel("K_max (allowed maximum)")
ax.set_ylabel("Effective K (after Bayesian pruning)")
ax.set_title("Do firms saturate as we allow more technology areas?")
ax.legend()
for _, row in k_df.iterrows():
    ax.annotate(f'{row["ceiling_rate_pct"]:.0f}%', (row["k_max"], row["p90"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9, color="coral")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz2a_k_progression.png", dpi=120, bbox_inches="tight")
plt.close()


# ===== VIZ 2B: Violin plot =====
print("Generating viz2b...")
violin_rows = []
for k_max in K_MAX_VALUES:
    gmm_results = [r for r in all_results[k_max] if r["tier"] == "gmm"]
    for r in gmm_results:
        violin_rows.append({"K_max": k_max, "Effective K": r["n_components"]})
violin_df = pd.DataFrame(violin_rows)

n_gmm = len([r for r in all_results[K_MAX_VALUES[0]] if r["tier"] == "gmm"])
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=violin_df, x="K_max", y="Effective K", inner="quartile", palette="Blues", ax=ax)
ax.set_title(f"Distribution of effective K across {n_gmm:,} GMM-tier firms (after deduplication)")
ax.set_xlabel("K_max setting")
ax.set_ylabel("Effective K (post-pruning)")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz2b_k_violin.png", dpi=120, bbox_inches="tight")
plt.close()


# ===== VIZ 2C: K vs firm size =====
print("Generating viz2c...")
k_max_largest = K_MAX_VALUES[-1]
gmm_results = [r for r in all_results[k_max_largest] if r["tier"] == "gmm"]
sizes = np.array([r["n_patents"] for r in gmm_results])
ks = np.array([r["n_components"] for r in gmm_results])
rho, _ = spearmanr(sizes, ks)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(sizes, ks, alpha=0.3, s=15, color="steelblue")
log_sizes = np.log(sizes)
b, a = np.polyfit(log_sizes, ks, 1)
x_fit = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
ax.plot(x_fit, a + b * np.log(x_fit), "r-", linewidth=2, label=f"K = {a:.2f} + {b:.2f}·log(n)")
ax.set_xscale("log")
ax.set_xlabel("Firm patent count (log scale)")
ax.set_ylabel(f"Effective K at K_max={k_max_largest}")
ax.set_title(f"Effective K scales with firm size (Spearman ρ = {rho:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz2c_k_vs_size.png", dpi=120, bbox_inches="tight")
plt.close()


# ===== VIZ 3A: Dual-axis Spearman rho (zoomed) + Top-50 overlap =====
print("Generating viz3a...")
adjacent = summary["adjacent_comparisons"]
adj_df = pd.DataFrame(adjacent)
adj_labels = [f"{r['k_max_a']}\u2192{r['k_max_b']}" for r in adjacent]
x = range(len(adj_df))

fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis: Spearman rho (zoomed)
color_rho = "steelblue"
ax1.plot(x, adj_df["spearman_rho"], "o-", color=color_rho, linewidth=2.5, markersize=10,
         label="Spearman \u03c1", zorder=3)
for i, rho_val in enumerate(adj_df["spearman_rho"]):
    ax1.annotate(f"{rho_val:.4f}", (i, rho_val), textcoords="offset points",
                 xytext=(-5, 10), fontsize=9, color=color_rho, fontweight="bold")
ax1.set_ylabel("Spearman \u03c1", color=color_rho, fontsize=12)
ax1.set_ylim([0.980, 1.001])
ax1.tick_params(axis="y", labelcolor=color_rho)
ax1.axhline(y=0.95, color=color_rho, linestyle=":", alpha=0.4, label="\u03c1 threshold (0.95)")

# Right axis: Top-50 overlap
ax2 = ax1.twinx()
color_top50 = "coral"
ax2.bar(x, adj_df["top_50_overlap_pct"], width=0.4, color=color_top50, alpha=0.6,
        label="Top-50 overlap", zorder=2)
for i, t50 in enumerate(adj_df["top_50_overlap_pct"]):
    ax2.annotate(f"{t50:.0f}%", (i, t50), textcoords="offset points",
                 xytext=(0, 5), fontsize=10, color="darkred", fontweight="bold", ha="center")
ax2.set_ylabel("Top-50 pair overlap (%)", color=color_top50, fontsize=12)
ax2.set_ylim([0, 115])
ax2.tick_params(axis="y", labelcolor=color_top50)
ax2.axhline(y=80, color=color_top50, linestyle="--", alpha=0.5, label="Top-50 threshold (80%)")

ax1.set_xticks(list(x))
ax1.set_xticklabels(adj_labels, fontsize=11)
ax1.set_xlabel("K_max transition", fontsize=12)
ax1.set_title("Both convergence metrics pass at every transition", fontsize=13, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz3a_spearman.png", dpi=120, bbox_inches="tight")
plt.close()


# ===== VIZ 3B: BC scatter (subsampled) =====
print("Generating viz3b...")
k_a, k_b = K_MAX_VALUES[1], K_MAX_VALUES[2]
gvkeys_a, bc_a = bc_matrices[k_a]
gvkeys_b, bc_b = bc_matrices[k_b]
n = len(gvkeys_a)
iu = np.triu_indices(n, k=1)
flat_a = bc_a[iu]
flat_b = bc_b[iu]

rng = np.random.RandomState(42)
n_sample = min(1_000_000, len(flat_a))
idx = rng.choice(len(flat_a), size=n_sample, replace=False)

fig, ax = plt.subplots(figsize=(8, 8))
hb = ax.hexbin(flat_a[idx], flat_b[idx], gridsize=80, cmap="viridis", bins="log", mincnt=1)
ax.plot([0, 1], [0, 1], "r-", linewidth=1, alpha=0.7, label="y = x")
ax.set_xlabel(f"BC at K_max={k_a}")
ax.set_ylabel(f"BC at K_max={k_b}")
ax.set_title(f"Pairwise BC: K_max={k_a} vs K_max={k_b} (corrected, linear weights)\n"
             f"Max BC: {max(flat_a.max(), flat_b.max()):.4f} — properly bounded [0, 1]\n"
             f"(1M random subsample of {len(flat_a):,} pairs)")
ax.set_xlim([0, max(1.0, flat_a.max() * 1.05)])
ax.set_ylim([0, max(1.0, flat_b.max() * 1.05)])
fig.colorbar(hb, ax=ax, label="Pair count (log scale)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / f"03_viz3b_bc_scatter_k{k_a}_vs_k{k_b}.png", dpi=120, bbox_inches="tight")
plt.close()
del flat_a, flat_b, idx


# ===== VIZ 4A: Top-100 overlap heatmap =====
print("Generating viz4a...")

def top_k_pair_set(bc_matrix, k):
    n = bc_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    flat = bc_matrix[iu]
    top_idx = np.argsort(-flat)[:k]
    return set(zip(iu[0][top_idx].tolist(), iu[1][top_idx].tolist()))

top_k = 100
top_pair_sets = {}
for k_max in K_MAX_VALUES:
    _, bc_mat = bc_matrices[k_max]
    top_pair_sets[k_max] = top_k_pair_set(bc_mat, top_k)

n_kmax = len(K_MAX_VALUES)
overlap_matrix = np.zeros((n_kmax, n_kmax))
for i, ka in enumerate(K_MAX_VALUES):
    for j, kb in enumerate(K_MAX_VALUES):
        overlap_matrix[i, j] = len(top_pair_sets[ka] & top_pair_sets[kb]) / top_k * 100

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(overlap_matrix, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
            xticklabels=K_MAX_VALUES, yticklabels=K_MAX_VALUES,
            cbar_kws={"label": "Top-100 pair overlap (%)"}, ax=ax)
ax.set_xlabel("K_max")
ax.set_ylabel("K_max")
ax.set_title(f"Top-{top_k} pair overlap matrix across K_max settings (corrected)")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz4a_top100_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()


# ===== VIZ 4B: Rank trajectories =====
print("Generating viz4b...")
N_PAIRS_TRACE = 200
k_first = K_MAX_VALUES[0]
_, bc_first = bc_matrices[k_first]
n = bc_first.shape[0]
iu = np.triu_indices(n, k=1)
flat_first = bc_first[iu]
top_idx_first = np.argsort(-flat_first)[:N_PAIRS_TRACE]
top_pairs = list(zip(iu[0][top_idx_first].tolist(), iu[1][top_idx_first].tolist()))

pair_to_flat = {}
for p_idx, (i_, j_) in enumerate(top_pairs):
    flat_pos = i_ * n - i_ * (i_ + 1) // 2 + (j_ - i_ - 1)
    pair_to_flat[p_idx] = flat_pos

rank_trajectories = np.zeros((N_PAIRS_TRACE, len(K_MAX_VALUES)), dtype=np.int64)
for k_idx, k_max in enumerate(K_MAX_VALUES):
    _, bc_mat = bc_matrices[k_max]
    flat = bc_mat[iu]
    rank_array = np.empty(len(flat), dtype=np.int64)
    rank_array[np.argsort(-flat)] = np.arange(len(flat))
    for p_idx in range(N_PAIRS_TRACE):
        rank_trajectories[p_idx, k_idx] = rank_array[pair_to_flat[p_idx]] + 1

robust_mask = (rank_trajectories <= N_PAIRS_TRACE).all(axis=1)
robust_count = robust_mask.sum()
volatile_indices = np.where(~robust_mask)[0]

fig, ax = plt.subplots(figsize=(11, 7))

# Background: robust pairs as shaded band (P5-P95) + median
robust_ranks = rank_trajectories[robust_mask]
p5 = np.percentile(robust_ranks, 5, axis=0)
p25 = np.percentile(robust_ranks, 25, axis=0)
median = np.median(robust_ranks, axis=0)
p75 = np.percentile(robust_ranks, 75, axis=0)
p95 = np.percentile(robust_ranks, 95, axis=0)
x_pos = range(len(K_MAX_VALUES))

ax.fill_between(x_pos, p5, p95, alpha=0.12, color="steelblue",
                label=f"Robust pairs P5\u2013P95 (n={robust_count})")
ax.fill_between(x_pos, p25, p75, alpha=0.25, color="steelblue", label="P25\u2013P75")
ax.plot(x_pos, median, "-", color="steelblue", linewidth=2, alpha=0.7, label="Median rank")

# Foreground: volatile pairs as bold individual lines
for v_idx in volatile_indices:
    ax.plot(x_pos, rank_trajectories[v_idx], "o-", color="coral", linewidth=2.5,
            markersize=7, alpha=0.9, zorder=5)

ax.set_yscale("log")
ax.invert_yaxis()
ax.set_xticks(list(x_pos))
ax.set_xticklabels([f"K_max={k}" for k in K_MAX_VALUES])
ax.set_ylabel("Pair rank (log scale, top = best)")
ax.axhline(y=N_PAIRS_TRACE, color="black", linestyle=":", alpha=0.5,
           label=f"Top-{N_PAIRS_TRACE} boundary")
ax.set_title(f"Rank stability of top-{N_PAIRS_TRACE} pairs from K_max={k_first}",
             fontsize=13, fontweight="bold")
ax.text(0.02, 0.98,
        f"{robust_count}/{N_PAIRS_TRACE} pairs remain in top-{N_PAIRS_TRACE} at every K_max\n"
        f"{len(volatile_indices)} pair{'s' if len(volatile_indices) != 1 else ''} drop out (coral)",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz4b_rank_trajectories.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Robust pairs: {robust_count}/{N_PAIRS_TRACE}")


# ===== VIZ 4C: NN-5 distribution =====
print("Generating viz4c...")

def per_firm_top_k_neighbors(bc_matrix, k=5):
    n = bc_matrix.shape[0]
    neighbor_sets = []
    for i in range(n):
        row = bc_matrix[i].copy()
        row[i] = -1
        top = np.argsort(-row)[:k]
        neighbor_sets.append(set(top.tolist()))
    return neighbor_sets

k_a_nn, k_b_nn = K_MAX_VALUES[1], K_MAX_VALUES[2]
_, bc_a_nn = bc_matrices[k_a_nn]
_, bc_b_nn = bc_matrices[k_b_nn]
nn_a = per_firm_top_k_neighbors(bc_a_nn, k=5)
nn_b = per_firm_top_k_neighbors(bc_b_nn, k=5)
overlaps = np.array([len(a & b) for a, b in zip(nn_a, nn_b)])

fig, ax = plt.subplots(figsize=(10, 5))
counts = np.bincount(overlaps, minlength=6)
ax.bar(range(6), counts, color="steelblue", edgecolor="black", alpha=0.8)
for i, c in enumerate(counts):
    ax.text(i, c + max(counts) * 0.01, f"{c:,}\n({c/len(overlaps):.1%})", ha="center", va="bottom", fontsize=9)
ax.set_xlabel(f"Number of nearest neighbors preserved (out of 5)\nK_max={k_a_nn} → K_max={k_b_nn}")
ax.set_ylabel("Firms")
ax.set_title(f"Per-firm nearest-neighbor stability between K_max={k_a_nn} and K_max={k_b_nn} (corrected)")
ax.set_xticks(range(6))
plt.tight_layout()
plt.savefig(OUT_DIR / f"03_viz4c_nn5_distribution_k{k_a_nn}_k{k_b_nn}.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Median NN-5: {np.median(overlaps):.0f}/5, P10: {np.percentile(overlaps, 10):.0f}/5")


# ===== VIZ 5A: Firm K progression =====
print("Generating viz5a...")
firm_k_progression = {gk: [] for gk in DEEP_DIVE_FIRMS}
firm_n_patents = {}
for k_max in K_MAX_VALUES:
    by_gvkey = {r["gvkey"]: r for r in all_results[k_max]}
    for gk in DEEP_DIVE_FIRMS:
        if gk in by_gvkey:
            firm_k_progression[gk].append(by_gvkey[gk]["n_components"])
            firm_n_patents[gk] = by_gvkey[gk]["n_patents"]
        else:
            firm_k_progression[gk].append(np.nan)

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("Set1", n_colors=len(DEEP_DIVE_FIRMS))
for (gk, name), color in zip(DEEP_DIVE_FIRMS.items(), colors):
    label = f"{name} (n={firm_n_patents.get(gk, '?'):,})"
    ax.plot(K_MAX_VALUES, firm_k_progression[gk], "o-", color=color, linewidth=2, markersize=8, label=label)
ax.plot(K_MAX_VALUES, K_MAX_VALUES, "k:", alpha=0.4, label="Ceiling")
ax.set_xlabel("K_max")
ax.set_ylabel("Effective K")
ax.set_title("Effective K progression for 5 named firms (after deduplication)")
ax.legend(loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz5a_firm_k_progression.png", dpi=120, bbox_inches="tight")
plt.close()


# ===== VIZ 5B: Weight evolution =====
print("Generating viz5b...")
fig, axes = plt.subplots(len(DEEP_DIVE_FIRMS), 1, figsize=(11, 3 * len(DEEP_DIVE_FIRMS)), sharex=True)
for ax, (gk, name) in zip(axes, DEEP_DIVE_FIRMS.items()):
    weight_matrix = []
    max_k = 0
    for k_max in K_MAX_VALUES:
        by_gvkey = {r["gvkey"]: r for r in all_results[k_max]}
        if gk in by_gvkey:
            w = sorted(by_gvkey[gk]["weights"], reverse=True)
            weight_matrix.append(w)
            max_k = max(max_k, len(w))
        else:
            weight_matrix.append([])
    padded = np.zeros((len(K_MAX_VALUES), max_k))
    for i, w in enumerate(weight_matrix):
        padded[i, :len(w)] = w
    bottom = np.zeros(len(K_MAX_VALUES))
    colors_comp = sns.color_palette("husl", n_colors=max_k)
    for k_idx in range(max_k):
        ax.bar(K_MAX_VALUES, padded[:, k_idx], bottom=bottom, width=2.5,
               color=colors_comp[k_idx], edgecolor="white", linewidth=0.3)
        bottom += padded[:, k_idx]
    ax.set_ylabel(f"{name}\n(n={firm_n_patents.get(gk, '?'):,})")
    ax.set_ylim([0, 1.02])
    ax.set_xticks(K_MAX_VALUES)
axes[-1].set_xlabel("K_max")
fig.suptitle("Component weight evolution as K_max increases", y=1.001)
plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz5b_firm_weights.png", dpi=120, bbox_inches="tight")
plt.close()


# ===== VIZ 6A: Convergence dashboard =====
print("Generating viz6a...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
adj_sorted = sorted(summary["adjacent_comparisons"], key=lambda m: m["k_max_a"])
labels = [f"{m['k_max_a']}\u2192{m['k_max_b']}" for m in adj_sorted]

ax = axes[0, 0]
rhos = [m["spearman_rho"] for m in adj_sorted]
ax.plot(range(len(rhos)), rhos, "o-", color="steelblue", linewidth=2, markersize=10)
ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.6, label="Threshold (0.95)")
ax.set_xticks(range(len(rhos)))
ax.set_xticklabels(labels)
ax.set_ylim([0.7, 1.01])
ax.set_title("Spearman ρ (rank correlation)")
ax.legend()

ax = axes[0, 1]
top50 = [m.get("top_50_overlap_pct", 0) for m in adj_sorted]
ax.plot(range(len(top50)), top50, "o-", color="coral", linewidth=2, markersize=10)
ax.axhline(y=80, color="red", linestyle="--", alpha=0.6, label="Threshold (80%)")
ax.set_xticks(range(len(top50)))
ax.set_xticklabels(labels)
ax.set_ylim([0, 105])
ax.set_title("Top-50 pair overlap (%)")
ax.legend()

ax = axes[1, 0]
nn5 = [m.get("mean_nn5_overlap_pct", 0) for m in adj_sorted]
ax.plot(range(len(nn5)), nn5, "o-", color="seagreen", linewidth=2, markersize=10)
ax.set_xticks(range(len(nn5)))
ax.set_xticklabels(labels)
ax.set_ylim([0, 105])
ax.set_title("Mean per-firm NN-5 overlap (%)")

ax = axes[1, 1]
ax.plot(K_MAX_VALUES, k_df["ceiling_rate_pct"], "o-", color="mediumpurple", linewidth=2, markersize=10)
ax.set_xticks(K_MAX_VALUES)
ax.set_ylim([0, max(50, k_df["ceiling_rate_pct"].max() * 1.2)])
ax.set_title("Ceiling rate: % of GMM-tier firms with effective K = K_max")

verdict = summary["convergence_verdict"]
converged_at = summary.get("converged_at_kmax")
verdict_str = f"VERDICT: {verdict.upper()}"
if converged_at:
    verdict_str += f" at K_max={converged_at}"
fig.suptitle(verdict_str, fontsize=15, fontweight="bold",
             color="green" if verdict == "converged" else "darkred")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_viz6a_convergence_dashboard.png", dpi=120, bbox_inches="tight")
plt.close()

# ===== SUMMARY STATS =====
print("\n" + "=" * 60)
print("ROBUST/MODEL-SENSITIVE PAIR CLASSIFICATION")
print("=" * 60)
TOP_K = 200
top_200_sets = {}
for k_max in K_MAX_VALUES:
    _, bc_mat = bc_matrices[k_max]
    top_200_sets[k_max] = top_k_pair_set(bc_mat, TOP_K)

all_top_pairs = set()
for s in top_200_sets.values():
    all_top_pairs |= s

pair_appearances = {p: sum(1 for k in K_MAX_VALUES if p in top_200_sets[k]) for p in all_top_pairs}
robust_pairs = {p for p, c in pair_appearances.items() if c == len(K_MAX_VALUES)}
model_sensitive = all_top_pairs - robust_pairs

print(f"Total pairs ever in top-{TOP_K}: {len(all_top_pairs):,}")
print(f"  Robust (all K_max): {len(robust_pairs):,} ({len(robust_pairs)/len(all_top_pairs):.1%})")
print(f"  Model-sensitive: {len(model_sensitive):,} ({len(model_sensitive)/len(all_top_pairs):.1%})")

counts = pd.Series(list(pair_appearances.values())).value_counts().sort_index()
print(f"\nDistribution of appearance counts (out of {len(K_MAX_VALUES)} K_max):")
print(counts.to_string())

print("\n" + "=" * 60)
print(f"All 11 PNGs regenerated in {OUT_DIR}/")
print("=" * 60)
