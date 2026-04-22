"""Builds notebooks/05_synthetic_merger_case_study.ipynb via nbformat.

Run:
    source venv/bin/activate
    python3 scripts/build_notebook_05.py
    jupyter nbconvert --to notebook --execute notebooks/05_synthetic_merger_case_study.ipynb --inplace

Walkthrough for Arthur's synthetic-merger case study (prototype of Step 4).
Reads the output parquets produced by scripts/case_study_synthetic_mergers.py.

This build reports both the raw ΔBC (vs production acquirer GMM) and the
clean ΔBC (vs a fresh same-seed refit of the acquirer alone). The empirical
finding from this run is that refit noise on BC-to-external-firms is
negligible — raw and clean ΔBC agree to 4+ decimal places — so the original
raw ΔBC values are clean merger-effect estimates at this level. A different
confound (VI-sensitivity to small data perturbations) remains and is
discussed in the notebook.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path("notebooks/05_synthetic_merger_case_study.ipynb")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    # ----- Title + purpose -----
    cells.append(md(
        "# Notebook 05: Synthetic Merger Case Study (Prototype of Step 4)\n"
        "\n"
        "**Purpose**: walk through the synthetic-merger case study Arthur requested "
        "on 2026-04-19. For each of four (acquirer, target) pairs, we concatenate "
        "the firms' patent portfolios, fit a new Bayesian GMM over the combined "
        "patents, and compute BC between that synthetic firm and the top-20 firms "
        "with highest BC to the acquirer. ΔBC = BC(synth, f) − BC(baseline, f) then "
        "proxies how much the merger pulls the acquirer toward each firm f in "
        "technology space.\n"
        "\n"
        "**This run includes a refit-baseline experiment.** For every pair we "
        "also refit the acquirer *alone* from scratch with the same priors, seed, "
        "n_init, and hyperparameters as the synthetic. We report two ΔBC "
        "measurements per comparator:\n"
        "\n"
        "```\n"
        "ΔBC_raw(f)   = BC(synth, f) − BC(acquirer_production, f)\n"
        "ΔBC_clean(f) = BC(synth, f) − BC(acquirer_refit,      f)\n"
        "```\n"
        "\n"
        "**Headline finding from the baseline experiment**: `ΔBC_raw ≈ ΔBC_clean` "
        "to 4+ decimal places across all four pairs. Empirically, `BC(acq_refit, "
        "f) ≈ BC(acq_production, f)` for every external firm f — the two GMMs "
        "disagree on *self-BC* (BC(acq_refit, acq_prod) ≈ 0.08–0.11, not 1.0) "
        "but they produce indistinguishable BC values against any third firm. "
        "That means VI refit noise does **not** contaminate ΔBC measurements to "
        "external firms, and the raw ΔBC values are already clean merger-effect "
        "estimates at the refit-noise level. Section 4 unpacks what this does "
        "and doesn't rule out.\n"
        "\n"
        "**Status**: prototype — NOT production methodology for Step 4. See "
        "`docs/epics/week2_firm_portfolios/synthetic_merger_case_study.md` for "
        "promotion-path requirements (ADR-008 + spec + significance framework).\n"
    ))

    # ----- Section 1: Setup -----
    cells.append(md(
        "## Section 1: Setup\n"
        "\n"
        "Loads the outputs of `scripts/case_study_synthetic_mergers.py` from "
        "`output/case_studies/synthetic_mergers/`. Run the script first if the "
        "artifacts are missing:\n"
        "\n"
        "```bash\n"
        "source venv/bin/activate\n"
        "python scripts/case_study_synthetic_mergers.py\n"
        "```\n"
        "\n"
        "Expected artifacts: `synthetic_merger_pairs_k15.parquet` (4 rows), "
        "`synthetic_merger_comparators_k15.parquet` (88 rows = 4 pairs × 22 rows: "
        "20 comparators + 2 self-sanity), `synthetic_firm_gmm_parameters_k15.parquet` "
        "(8 rows = 4 synth + 4 acquirer-refit baselines, schema-compatible with "
        "the production parquet), and `run_metadata.json`.\n"
    ))

    cells.append(code(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "# Auto-chdir if running from notebooks/ subdir (nbconvert cwd)\n"
        "if not Path('output/case_studies').exists() and Path('../output/case_studies').exists():\n"
        "    os.chdir('..')\n"
        "\n"
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "CASE_DIR = Path('output/case_studies/synthetic_mergers')\n"
        "PAIRS = pd.read_parquet(CASE_DIR / 'synthetic_merger_pairs_k15.parquet')\n"
        "COMP  = pd.read_parquet(CASE_DIR / 'synthetic_merger_comparators_k15.parquet')\n"
        "META  = json.loads((CASE_DIR / 'run_metadata.json').read_text())\n"
        "\n"
        "assert len(PAIRS) == 4, f'Expected 4 pairs, got {len(PAIRS)}'\n"
        "assert len(COMP) == 88, f'Expected 88 comparator rows, got {len(COMP)}'\n"
        "assert {'delta_bc', 'delta_bc_clean', 'refit_noise'}.issubset(COMP.columns), 'Run script with acquirer-refit baseline'\n"
        "\n"
        "print(f'Loaded {len(PAIRS)} pairs and {len(COMP)} comparator rows')\n"
        "print(f'Run git SHA: {META[\"git_sha\"]}, K_max={META[\"k_max\"]}, '\n"
        "      f'total elapsed: {META[\"total_elapsed_s\"]:.1f}s')\n"
    ))

    # ----- Section 2: Methodology -----
    cells.append(md(
        "## Section 2: Methodology\n"
        "\n"
        "**Arthur's request** (email, 2026-04-19): for each (acquirer, target) pair, "
        "merge the patent portfolios, fit a new GMM on the combined portfolio, and "
        "compute pairwise BC against the top-20 firms by BC(acquirer, *). Large "
        "positive ΔBC for firm f → merger repositions acquirer toward f's "
        "technology space.\n"
        "\n"
        "**What this run does**:\n"
        "\n"
        "1. **Synthetic portfolio = vector concatenation.** Acquirer and target "
        "patent vectors are concatenated (zero patent-id overlap verified upstream).\n"
        "2. **Two GMMs are re-fit** per pair, both with identical priors, "
        "hyperparameters, random_state=42, and n_init=5:\n"
        "   - `synth`: fit on the combined (acquirer + target) portfolio.\n"
        "   - `acq_refit`: fit on the acquirer alone — baseline for testing "
        "     whether VI refit stochasticity contaminates ΔBC to external firms.\n"
        "3. **Priors and hyperparameters match production** (ADR-004/005/006): "
        "diagonal covariance, K_max=15, empirical-Bayes priors computed once from "
        "the full unique-patent matrix, `reg_covar=1e-4`, weight pruning at 0.01.\n"
        "4. **BC formula is linear-weighted** (`bc_mixture_linear`, bounded [0, 1]).\n"
        "5. **Comparator set**: top-20 firms by BC(acquirer, *) in the production "
        "BC matrix, excluding acquirer and target. `BC(synth, acquirer)` and "
        "`BC(synth, target)` are kept as self-sanity rows for diagnostics.\n"
        "\n"
        "**What each comparator row reports**:\n"
        "\n"
        "| Column | Meaning |\n"
        "|---|---|\n"
        "| `bc_acquirer_to_comparator` | BC(acquirer, f) — from the production matrix |\n"
        "| `bc_target_to_comparator` | BC(target, f) — from the production matrix |\n"
        "| `bc_synthetic_to_comparator` | BC(synth, f) — fresh this run |\n"
        "| `bc_acquirer_refit_to_comparator` | BC(acq_refit, f) — fresh this run, same VI procedure as synth |\n"
        "| `delta_bc` | BC(synth, f) − BC(acq, f) — raw measure |\n"
        "| `delta_bc_clean` | BC(synth, f) − BC(acq_refit, f) — baseline-subtracted measure |\n"
        "| `refit_noise` | BC(acq_refit, f) − BC(acq, f) — how much refit alone moved BC(*, f) |\n"
    ))

    # ----- Section 3: Pair-level overview -----
    cells.append(md(
        "## Section 3: Pair-level overview\n"
        "\n"
        "Each row combines pre-merger facts, GMM-fit diagnostics, and both the "
        "raw and clean argmax summaries. Note that `max_abs_refit_noise` is the "
        "per-pair empirical floor of the refit-noise contribution to ΔBC.\n"
    ))

    cells.append(code(
        "overview_cols = [\n"
        "    'pair_id', 'acquirer', 'target',\n"
        "    'n_patents_acquirer', 'n_patents_target', 'target_patent_share',\n"
        "    'pre_merger_bc',\n"
        "    'synthetic_k_effective', 'acquirer_refit_k_effective',\n"
        "    'bc_acquirer_refit_vs_production',\n"
        "    'max_delta_bc', 'argmax_delta_bc_comparator',\n"
        "    'max_delta_bc_clean', 'argmax_delta_bc_clean_comparator',\n"
        "    'max_abs_refit_noise',\n"
        "]\n"
        "display = PAIRS[overview_cols].copy()\n"
        "display['target_patent_share'] = display['target_patent_share'].apply(lambda x: f'{x*100:.2f}%')\n"
        "for col in ['pre_merger_bc','bc_acquirer_refit_vs_production']:\n"
        "    display[col] = display[col].apply(lambda x: f'{x:.4f}')\n"
        "display['max_abs_refit_noise'] = display['max_abs_refit_noise'].apply(lambda x: f'{x:.2e}')\n"
        "for col in ['max_delta_bc','max_delta_bc_clean']:\n"
        "    display[col] = display[col].apply(lambda x: f'{x:+.4f}')\n"
        "display\n"
    ))

    cells.append(md(
        "**How to read this table**:\n"
        "\n"
        "- `bc_acquirer_refit_vs_production`: BC between the fresh-refit acquirer "
        "GMM and the production acquirer GMM. Values in the 0.03–0.11 range show "
        "that VI converges to substantially different component decompositions "
        "run-to-run — the self-BC between two fits of the same data is far from 1. "
        "But this is about component alignment, not density.\n"
        "- `max_delta_bc` (raw): largest |ΔBC| against production acquirer GMM.\n"
        "- `max_delta_bc_clean`: largest |ΔBC| against the refit-acquirer baseline.\n"
        "- `max_abs_refit_noise`: largest |BC(acq_refit, f) − BC(acq, f)| across "
        "the top-20. This is the empirical upper bound on the refit-noise "
        "contribution to ΔBC *at external-firm BCs*. For every pair in this run, "
        "it's ≤ 4×10⁻⁵ — five orders of magnitude below `max_delta_bc`. That's "
        "the punchline of the baseline experiment.\n"
    ))

    # ----- Section 4: Baseline verdict + residual confound -----
    cells.append(md(
        "## Section 4: What the baseline experiment actually ruled out\n"
        "\n"
        "Running the acquirer-refit baseline was a falsification test for one "
        "specific concern: *does VI refit stochasticity contaminate ΔBC "
        "measurements to external firms?* The experiment came back with a "
        "clear answer.\n"
    ))

    cells.append(code(
        "# Empirical distribution of refit_noise across top-20 comparators, per pair\n"
        "noise_df = COMP[COMP['role']=='comparator'].copy()\n"
        "noise_summary = noise_df.groupby('pair_id')['refit_noise'].agg(['mean','std','min','max'])\n"
        "noise_summary['max_abs'] = noise_df.groupby('pair_id')['refit_noise'].apply(lambda s: s.abs().max())\n"
        "noise_summary.round(6)\n"
    ))

    cells.append(code(
        "# Correlation and absolute gap between raw and clean ΔBC, per pair\n"
        "rows = []\n"
        "for pid in [1,2,3,4]:\n"
        "    sub = COMP[(COMP['pair_id']==pid) & (COMP['role']=='comparator')]\n"
        "    rho = sub[['delta_bc','delta_bc_clean']].corr().iloc[0,1]\n"
        "    max_gap = (sub['delta_bc'] - sub['delta_bc_clean']).abs().max()\n"
        "    rows.append({'pair_id': pid, 'corr(raw, clean)': round(rho, 6), 'max |raw − clean|': f'{max_gap:.2e}'})\n"
        "pd.DataFrame(rows)\n"
    ))

    cells.append(md(
        "**What these two tables say**:\n"
        "\n"
        "- `refit_noise` across the top-20 comparators has `max_abs` in the "
        "~10⁻⁵ to 10⁻⁶ range per pair. That's effectively zero in ΔBC units.\n"
        "- Raw and clean ΔBC correlate at 1.000 with `|raw − clean|` bounded by "
        "~4×10⁻⁵. They agree to every decimal place we report.\n"
        "- *Conclusion*: BC to an external firm f is the same function of the "
        "underlying acquirer distribution regardless of which particular VI fit "
        "represents it. The `BC(acq_refit, acq_prod)` = 0.03–0.11 self-BC "
        "observed in Section 3 reflects component-permutation non-identifiability "
        "(VI picks different component centers run-to-run) but does not propagate "
        "to BC against any third firm.\n"
        "\n"
        "**So the original raw ΔBC values are clean merger-effect estimates at "
        "the refit-noise level.** The baseline subtraction doesn't remove noise "
        "because there wasn't noise to remove along this dimension — it "
        "confirms that conclusion rather than correcting for it.\n"
        "\n"
        "### What the baseline does NOT rule out\n"
        "\n"
        "There is a separate confound this experiment does not address: "
        "**VI-sensitivity to small data perturbations**. The synthetic and "
        "acquirer-refit GMMs share `random_state=42` but see *different data* "
        "(combined vs acquirer-alone). Even a tiny data change can push VI to a "
        "different local optimum, producing non-trivial ΔBC against external "
        "firms that reflects the perturbation, not a merger effect per se.\n"
        "\n"
        "Pair 4 is the clearest illustration: the target is 9 patents out of "
        "39,728 (0.023% of the merged portfolio), yet `max |ΔBC|` = 0.019. The "
        "target is too small for this magnitude to be explained by its own "
        "content; it's explained by 'adding these 9 patents tipped VI into a "
        "slightly different optimum with different external-firm BCs'. Adding "
        "any 9 patents to this acquirer would plausibly produce a similar-"
        "magnitude shift.\n"
        "\n"
        "**The clean falsification of this confound** (the natural next "
        "experiment, see Section 10): random-patent-injection. For each pair, "
        "sample `n_target` random patents from the acquirer's own portfolio, "
        "add them to the acquirer, refit, and compute ΔBC_null. Compare to the "
        "actual ΔBC_merger. If |ΔBC_null| is comparable in magnitude, the merger "
        "signal isn't distinguishable from the VI perturbation floor for that "
        "pair.\n"
    ))

    # ----- Section 5: per-pair tables -----
    cells.append(md(
        "## Section 5: Per-pair results — top-ΔBC comparators\n"
        "\n"
        "For each pair, 22 rows (20 comparators + 2 self-sanity rows) sorted by "
        "`delta_bc` descending. `delta_bc` and `delta_bc_clean` are effectively "
        "identical (see Section 4); both columns are shown for transparency.\n"
    ))

    for pid in [1, 2, 3, 4]:
        cells.append(md(f"### Pair {pid}"))
        cells.append(code(
            f"pair_row = PAIRS[PAIRS['pair_id']=={pid}].iloc[0]\n"
            f"print(f'Pair {pid}: acq={{pair_row.acquirer}}  tgt={{pair_row.target}}  '\n"
            f"      f'n_acq={{pair_row.n_patents_acquirer}}  n_tgt={{pair_row.n_patents_target}}  '\n"
            f"      f'target_share={{pair_row.target_patent_share*100:.2f}}%  '\n"
            f"      f'pre_merger_BC={{pair_row.pre_merger_bc:.4f}}  '\n"
            f"      f'synth_k_eff={{pair_row.synthetic_k_effective}}  '\n"
            f"      f'refit_k_eff={{pair_row.acquirer_refit_k_effective}}  '\n"
            f"      f'BC(refit, prod_acq)={{pair_row.bc_acquirer_refit_vs_production:.4f}}')\n"
            f"\n"
            f"tbl = COMP[COMP['pair_id']=={pid}].copy()\n"
            f"tbl = tbl.sort_values('delta_bc', ascending=False)\n"
            f"view = tbl[['rank_in_acquirer_top20','role','comparator',\n"
            f"            'bc_acquirer_to_comparator','bc_acquirer_refit_to_comparator',\n"
            f"            'bc_synthetic_to_comparator','bc_target_to_comparator',\n"
            f"            'delta_bc','delta_bc_clean','refit_noise']].copy()\n"
            f"for col in ['bc_acquirer_to_comparator','bc_acquirer_refit_to_comparator',\n"
            f"            'bc_synthetic_to_comparator','bc_target_to_comparator']:\n"
            f"    view[col] = view[col].apply(lambda x: f'{{x:.4f}}')\n"
            f"for col in ['delta_bc','delta_bc_clean']:\n"
            f"    view[col] = view[col].apply(lambda x: f'{{x:+.4f}}')\n"
            f"view['refit_noise'] = view['refit_noise'].apply(lambda x: f'{{x:+.2e}}')\n"
            f"view\n"
        ))

    # ----- Section 6: visualization -----
    cells.append(md(
        "## Section 6: Visualization — ΔBC by rank, per pair\n"
        "\n"
        "Four panels, one per pair. x-axis is the comparator's rank in the "
        "acquirer's top-20 (rank 1 = highest BC to acquirer pre-merger). y-axis "
        "is ΔBC. Green = positive (merger pulls acquirer toward f); red = "
        "negative. We plot only `delta_bc` because `delta_bc_clean` is "
        "indistinguishable (see Section 4).\n"
        "\n"
        "The gray band on each panel is `±max_abs_refit_noise` — a floor so low "
        "it's barely visible. The question left open by this run is a different "
        "floor: the VI-sensitivity-to-data-perturbation floor, which is not "
        "measured here. Pair 4's magnitude (0.019 on 9 target patents) is the "
        "clue for that floor — plausibly ~0.02 for this acquirer size.\n"
    ))

    cells.append(code(
        "fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)\n"
        "for ax, pid in zip(axes.flat, [1, 2, 3, 4]):\n"
        "    pair_row = PAIRS[PAIRS['pair_id']==pid].iloc[0]\n"
        "    tbl = COMP[(COMP['pair_id']==pid) & (COMP['role']=='comparator')].copy()\n"
        "    tbl = tbl.sort_values('rank_in_acquirer_top20')\n"
        "    colors = ['C2' if x > 0 else 'C3' for x in tbl['delta_bc']]\n"
        "    ax.bar(tbl['rank_in_acquirer_top20'], tbl['delta_bc'], color=colors, alpha=0.85)\n"
        "    noise_floor = float(pair_row.max_abs_refit_noise)\n"
        "    ax.axhspan(-noise_floor, noise_floor, alpha=0.3, color='gray',\n"
        "               label=f'±refit-noise floor ({noise_floor:.1e})')\n"
        "    ax.axhline(0, color='black', linewidth=0.5)\n"
        "    ax.set_xlabel('Rank in acquirer top-20')\n"
        "    ax.set_ylabel('ΔBC = BC(synth,f) − BC(acq,f)')\n"
        "    ax.set_title(f'Pair {pid}: {pair_row.acquirer}+{pair_row.target}  '\n"
        "                 f'(target share {pair_row.target_patent_share*100:.2f}%)')\n"
        "    ax.legend(loc='best', fontsize=8)\n"
        "fig.suptitle('ΔBC by comparator rank — refit-noise floor shown; VI-perturbation floor not measured in this run', fontsize=12)\n"
        "fig.tight_layout()\n"
        "plt.savefig('notebooks/05_viz_delta_bc.png', dpi=100, bbox_inches='tight')\n"
        "plt.show()\n"
    ))

    # ----- Section 7: Interpretation -----
    cells.append(md(
        "## Section 7: Interpretation\n"
        "\n"
        "We now have ΔBC values we can treat as clean of refit noise but that "
        "still mix merger-effect with VI-sensitivity-to-data-perturbation. The "
        "honest pair-by-pair reading:\n"
        "\n"
        "**Pair 1 (001161 + 022325)** — target share 20.5%, pre-merger BC "
        "0.0014. Argmax ΔBC = +0.027 at `PRIV_AILTECHNOLOGIES`, rank 6 in "
        "acquirer's top-20. Look at the three BC values: BC(acq, AIL) = 0.0047, "
        "BC(tgt, AIL) = 0.0014, BC(synth, AIL) = **0.0321** — the synthetic is "
        "~7× more similar to AILTECHNOLOGIES than either constituent firm alone. "
        "This magnitude (target share 20%, ΔBC 0.027) is large enough that VI "
        "perturbation from adding 4,466 patents isn't a trivial explanation. "
        "**This is the strongest candidate for a real merger-effect signal** in "
        "the case study, pending the random-patent-injection calibration.\n"
        "\n"
        "**Pair 2 (001632 + 014256)** — target share 20.6%, pre-merger BC "
        "0.0108. Argmax |ΔBC| = 0.010 — much smaller than pair 1 despite a "
        "comparable target share. Likely explanation: the acquirer and target "
        "already overlap technologically (pre-merger BC is 8× higher than pair 1), "
        "so the merger doesn't add much new directionality. Nothing in this "
        "pair's distribution rises above what VI perturbation could plausibly "
        "produce; don't claim a signal here.\n"
        "\n"
        "**Pair 3 (007257 + 018510)** — target share 0.79%, pre-merger BC "
        "0.0002. Argmax ΔBC = +0.028 at `002403`. This is suspicious because the "
        "target is only 99 patents; the magnitude is comparable to pair 1 "
        "despite the target being ~45× smaller as a fraction of the portfolio. "
        "Most likely interpretation: VI perturbation floor is in the ~0.02 range "
        "for this acquirer, and the 99-patent perturbation is enough to hit it. "
        "Needs calibration.\n"
        "\n"
        "**Pair 4 (005606 + 008633)** — target share 0.023%. Argmax |ΔBC| = "
        "0.019 on 9 target patents. Mechanically impossible for the target's "
        "*content* to explain this magnitude. Best interpretation: the addition "
        "of any 9 patents to this acquirer's 39,719-patent portfolio pushes VI "
        "to a different local optimum, producing external-firm BC shifts on the "
        "order of 0.02 that are not merger effects. This pair's ΔBC distribution "
        "is effectively the **empirical VI-perturbation floor for large "
        "acquirers** in the pipeline.\n"
        "\n"
        "**Overall**: pair 1 is the only pair where the ΔBC magnitude "
        "plausibly exceeds what VI perturbation alone could produce, and even "
        "there the cleanest interpretation requires the random-patent-injection "
        "calibration. The other three pairs contribute mostly to characterizing "
        "the VI-perturbation floor itself — which is valuable methodology output "
        "for the Step 4 spec, just not itself a merger-effect finding.\n"
    ))

    # ----- Section 8: Methodology implications -----
    cells.append(md(
        "## Section 8: Methodology implications for Step 4\n"
        "\n"
        "Two concrete methodology outputs from this prototype, independent of "
        "whether any individual pair's ΔBC represents a merger effect:\n"
        "\n"
        "1. **VI refit stochasticity does NOT contaminate external-firm BC** "
        "under the production-faithful pipeline (same priors, same seed, same "
        "data → BC to any third firm is stable to 10⁻⁵). This is a useful "
        "invariant for the Step 4 spec: synthetic-portfolio ΔBC methodology "
        "does not need a refit-noise baseline.\n"
        "2. **VI-sensitivity to small data perturbations DOES affect ΔBC to "
        "external firms** on the order of ~0.02 for 9-patent perturbations to a "
        "40K-patent acquirer. This *is* a confound for the synthetic-portfolio "
        "pipeline: ΔBC of similar magnitude cannot be attributed to merger "
        "effect without a perturbation calibration. The Step 4 spec should "
        "require the random-patent-injection null distribution as a per-pair "
        "input, not just a global hyperparameter.\n"
        "\n"
        "Both of these shape what ADR-008 will recommend when it's written. "
        "They would not have been visible without running this prototype.\n"
    ))

    # ----- Section 9: Limitations -----
    cells.append(md(
        "## Section 9: Limitations\n"
        "\n"
        "1. **Prototype, not production.** No ADR, no spec, no unit tests, no "
        "Codex review. Production promotion requires ADR-008 for synthetic-"
        "portfolio methodology and a significance-testing framework.\n"
        "2. **Top-20 comparator set** is conditional on the acquirer. Universe-"
        "wide BC (all 7,485 firms) removes this restriction and is a prerequisite "
        "for any stat-sig claim.\n"
        "3. **No random-patent-injection calibration.** We cannot yet cleanly "
        "separate merger-effect signal from VI-perturbation noise for individual "
        "pairs. Section 4 motivates this; Section 10 lists it as the next step.\n"
        "4. **No significance testing.** All ΔBC values are point estimates.\n"
        "5. **Single-seed run.** One random_state=42 for synth and acq_refit. "
        "Multi-seed would quantify the seed-level uncertainty on top of the "
        "data-level uncertainty.\n"
        "6. **K_max=15 assumption.** Locked per ADR-004; validated by the "
        "convergence analysis in Notebook 03 for production BC matrices. "
        "Inherited here; not retested for synthetic fits.\n"
        "7. **Co-assignment assumption.** Zero shared patent_ids across the four "
        "pairs (asserted). `run_synthetic_merger` surfaces `n_shared_patents` so "
        "swapped pairs with overlap are caught, not silently deduped.\n"
    ))

    # ----- Section 10: Next steps -----
    cells.append(md(
        "## Section 10: Next steps\n"
        "\n"
        "Priority-ordered. The first item is the critical one for making sense "
        "of pair 1.\n"
        "\n"
        "1. **Random-patent-injection calibration** (~15 min per pair). For "
        "each pair, sample `n_target` patents uniformly from the acquirer's own "
        "portfolio, add them to the acquirer, refit, compute ΔBC_null. Repeat "
        "20–50 times to build a per-pair null distribution of |ΔBC|. The actual "
        "merger's |ΔBC| is interpretable only relative to this null. This is "
        "the single most-important experiment to validate pair 1's signal.\n"
        "2. **Universe-wide BC per pair** (~20 min per pair). Drop the top-20 "
        "restriction. Compute BC(synth, f) for all 7,485 deduplicated firms. "
        "Enables global ranking of the merger's pull and is a prerequisite for "
        "stat-sig claims.\n"
        "3. **Permutation / bootstrap significance** (~1 hour per pair). For "
        "each comparator, estimate the null distribution of ΔBC by randomly "
        "permuting patent-to-firm assignments and refitting. Yields p-values.\n"
        "4. **Formalize as Step 4** (~1 week). ADR-008 + spec + Codex review + "
        "TDD module. Required before this methodology can be cited in the "
        "paper as production.\n"
    ))

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def main() -> None:
    nb = build()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NB_PATH.open("w") as f:
        nbf.write(nb, f)
    print(f"Wrote {NB_PATH} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
