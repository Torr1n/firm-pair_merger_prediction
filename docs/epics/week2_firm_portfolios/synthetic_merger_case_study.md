# Synthetic Merger Case Study — Week 2 Prototype

**Status**: Prototype — NOT production methodology  |  **Date**: 2026-04-21
**Requested by**: Arthur (email 2026-04-19)  |  **Delivered by**: Torrin
**ADR dependencies**: ADR-004 (K_max=15 production), ADR-005 (empirical-Bayes priors), ADR-006 (diagonal covariance), ADR-007 (raw normalization)
**Blocks**: ADR-008 (directional complementarity + production Step 4 synthesis metric) — not yet opened.

---

## 1. Research question

From Arthur's email (quoted):

> To check the hypothesis that firms merge to become more similar to other firms in the same industry, I think we have to follow this particular pipeline:
>
> Take the patent portfolio of the acquirer and merge it with the patent portfolio of the target company (combine patent embeddings relating to both the acquirer and target gvkey). Then, fit a new GMM over this combined portfolio. With the new GMM, calculate the pairwise BC against the entire universe of firm patent portfolios. Then, calculate the change in BC between the acquirer's BC with a firm and the synthetic merged portfolio (acquirer + target)'s BC with a firm. My thoughts are that if the change in BC is large and positive for a specific firm, the acquirer might be trying to merge with the target in an attempt to become more similar to that specific firm. High BC should theoretically represent firms occupying the same technology space in their patents.

Torrin chose the computationally-manageable option Arthur offered (top-20 comparator set rather than universe-wide).

---

## 2. Scope and non-goals

**In scope**:
- Four hand-picked pairs: (001161, 022325), (001632, 014256), (007257, 018510), (005606, 008633).
- Synthetic portfolio via patent-vector concatenation + re-fit Bayesian GMM.
- **Acquirer-refit baseline** — each acquirer is also refit from scratch (alone, no target) with identical priors, random_state, and hyperparameters as the synthetic. This lets us report `ΔBC_clean = BC(synth, f) − BC(acquirer_refit, f)` which cancels variational-inference refit stochasticity at first order and isolates the merger effect from refit noise.
- BC against the **top-20** firms by BC(acquirer, *), excluding acquirer and target.
- Linear-weighted BC formula (bounded [0, 1]), same as production post-correction.
- K_max=15 (production lock per ADR-004).

**Explicitly NOT in scope**:
- Universe-wide BC computation.
- Permutation / bootstrap significance tests.
- Multi-seed baseline (single-seed refit at `random_state=42`; cross-seed envelope is a natural follow-up).
- A new ADR or spec.
- Unit tests or production TDD.
- New module in `src/`.
- Modification of any production scripts.

---

## 3. Methodology choices

### 3.1 Synthetic portfolio composition — vector concatenation
Patent vectors for acquirer and target are concatenated. Zero patent-id overlap was verified upstream across all four pairs (`n_shared_patents == 0` in the per-pair output); no de-duplication is performed. The script surfaces `n_shared_patents` in the per-pair parquet so that swapped pairs with overlap are caught, not silently absorbed.

### 3.2 GMM re-fit, not parameter-averaging
We refit a fresh Bayesian GMM over the combined patent vectors rather than computing a weighted average of the acquirer's and target's GMM parameters. Re-fitting preserves the Dirichlet-process prior on component weights and lets the effective K adapt to the combined distribution.

### 3.3 Acquirer-refit baseline for clean ΔBC
For each pair we also refit the acquirer alone (no target) with the same priors, random_state, n_init, and hyperparameters as the synthetic. The refit gvkey is `REFIT_{acquirer}`, stored in `synthetic_firm_gmm_parameters_k15.parquet`. We then report three related quantities per comparator:

- `delta_bc` (raw) = BC(synth, f) − BC(acquirer_production, f). Mixes merger effect with VI stochasticity.
- `delta_bc_clean` (merger-effect estimate) = BC(synth, f) − BC(acquirer_refit, f). The refit-noise component cancels at first order because synth and acquirer_refit share the same VI state.
- `refit_noise` = BC(acquirer_refit, f) − BC(acquirer_production, f). Empirical per-comparator estimate of the VI-stochasticity contribution.

`max_abs_refit_noise` per pair gives an empirical floor below which raw ΔBC is indistinguishable from refit stochasticity.

### 3.4 Comparator set — top-20 by BC(acquirer, *), excl. acquirer + target
- Acquirer is excluded because BC(acquirer, acquirer) = 1 by construction and carries no information.
- Target is excluded because the target is mechanically absorbed into the synthetic; its ΔBC would be dominated by "the synthetic contains the target's patents" rather than "the merger repositions the acquirer toward the target".
- **BC(synthetic, acquirer) and BC(synthetic, target) are reported separately as self-sanity rows** in the per-comparator parquet. The acquirer self-row is diagnostically important — it exposes refit non-identifiability.

### 3.5 Priors — global empirical Bayes, unchanged from production
We re-use `scripts/run_kmax_sweep.compute_global_priors(vectors)` to compute the global mean and variance **once** from the full unique-patent matrix (1,447,673 × 50). Using these as `mean_prior` and `covariance_prior` in every per-firm, per-synthetic, and per-refit fit matches production behavior.

### 3.6 BC formula — linear-weighted (`bc_mixture_linear`)
`BC(A, B) = Σᵢ Σⱼ πᵢᴬ · πⱼᴮ · BC(Nᵢᴬ, Nⱼᴮ)`, bounded in [0, 1] by Cauchy–Schwarz on the linear weights. Matches the production-corrected formula (see Notebook 04 Section 3 and Notebook 03 for the narrative on the earlier √-weighted bug).

---

## 4. Data lineage

| Input | Source |
|---|---|
| Per-firm GMM parameters (K=15) | `output/kmax_sweep/corrected/output/kmax_sweep/firm_gmm_parameters_k15.parquet` (S3: `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/`) |
| Pairwise BC matrix (K=15, dedup, linear-weighted) | Same run as above, `bc_matrix_all_k15_dedup_linear.npz` (7,485 × 7,485 float64) |
| 50D patent vectors | `output/week2_inputs/patent_vectors_50d.parquet` (1,447,673 × 50 float32) |
| Patent-to-firm mapping | `output/week2_inputs/gvkey_map.parquet` |

Priors, hyperparameters, random_state, n_init, reg_covar all match production (see `src/config/config.yaml` `portfolio:` section).

---

## 5. Pre-flight arithmetic (verified before the run)

| Pair | Acquirer patents | Target patents | Combined | Target share | Shared patent_ids | Pre-merger BC |
|---|---:|---:|---:|---:|---:|---:|
| 1 (001161 + 022325) | 17,347 | 4,466 | 21,813 | 20.5% | 0 | 0.0014 |
| 2 (001632 + 014256) | 4,320 | 1,123 | 5,443 | 20.6% | 0 | 0.0108 |
| 3 (007257 + 018510) | 12,415 | 99 | 12,514 | 0.79% | 0 | 0.0002 |
| 4 (005606 + 008633) | 39,719 | 9 | 39,728 | 0.023% | 0 | 0.0000 |

---

## 6. Limitations

### 6.1 Refit identifiability — tested by the baseline, NOT a confound on ΔBC
The baseline experiment (acquirer-refit with same seed/priors as synth) was set up as a falsification test for one hypothesis: "VI refit stochasticity contaminates ΔBC measurements to external firms." The test came back negative.

Empirically: `refit_noise = BC(acq_refit, f) − BC(acq_production, f)` across the top-20 comparators is in the 10⁻⁶ to 10⁻⁵ range per pair. Raw and clean ΔBC correlate at 1.000 with `|raw − clean|` bounded by ~4×10⁻⁵. Meaning the production acquirer GMM and a fresh same-seed refit disagree on their *self-BC* (0.08–0.11, see `bc_acquirer_refit_vs_production` in the pairs parquet) but produce indistinguishable BC values against any third firm.

This is a property of how the linear-weighted `bc_mixture_linear` treats mixture decompositions: two GMMs that represent the same underlying density can have low BC when compared against each other (if their components live at different centers) while giving the same BC against external firms (because external-firm BC depends on overall density, not component alignment). We document this invariant as a deliverable-level output for ADR-008.

### 6.2 VI-sensitivity to small data perturbations — the binding confound
Because the baseline removed one hypothesized confound, a different confound became visible: the synthetic and acquirer-refit GMMs share `random_state=42` but see different data (combined vs acquirer-alone). Even when the data difference is tiny, the VI optimization path is different, and the two fits land on substantially different local optima that *do* produce different BC values to external firms.

Pair 4 is the empirical tell. The target is 9 patents in a 39,728-patent merged portfolio (0.023%). The target's content is mathematically too small to account for `max |ΔBC| = 0.019`; the driver must be "adding these 9 patents to the acquirer tipped VI into a different optimum that has different BC to external firms." Pair 3 (99 target patents) has the same issue.

**This confound is not cancelled by the same-seed acquirer refit baseline.** The baseline cancels identical-data refit noise, not different-data perturbation sensitivity. Cleanly separating merger effect from VI-perturbation requires a different calibration — see §8 below and Notebook 05 Section 10 for the random-patent-injection null-distribution experiment that would do it.

### 6.3 No statistical significance
All ΔBC values are point estimates. No permutation, bootstrap, or posterior credible intervals are computed. A claim of significance requires either permutation over patent-to-firm assignment or a bootstrap over the underlying patents, combined with the VI-perturbation null in §6.2.

### 6.4 Top-20 comparator set is conditional on the acquirer
The comparator set is `{f : rank_{BC(acquirer, f)} ≤ 20}`. Under this restriction, ΔBC isn't comparable across pairs (different acquirers have different top-20s) and globally-strong candidates for "what the merger pulls toward" can be excluded. Universe-wide BC removes this restriction; it's the natural follow-up.

### 6.5 Single-seed baseline
The baseline is one refit at `random_state=42`. Even though we found refit noise on external-firm BCs to be negligible at that seed, a multi-seed sweep would quantify across-seed variation in both the synthetic and the refit — converting all ΔBC point estimates to intervals. Not required given §6.1, but useful for paper-quality numbers.

### 6.6 K_max=15 assumption
Locked per ADR-004; the convergence analysis (Notebook 03) demonstrated ranking stability across K_max ∈ {10, 15, 20, 25, 30} for the production BC matrix. We inherit that validation for the synthetic fit as well, though it has not been retested at other K_max values for synthetic firms.

---

## 7. Artifacts

| Artifact | Path |
|---|---|
| Script | `scripts/case_study_synthetic_mergers.py` |
| Notebook builder | `scripts/build_notebook_05.py` |
| Notebook | `notebooks/05_synthetic_merger_case_study.ipynb` |
| Bundle script | `scripts/build_case_study_bundle.sh` |
| Per-pair parquet | `output/case_studies/synthetic_mergers/synthetic_merger_pairs_k15.parquet` |
| Per-comparator parquet | `output/case_studies/synthetic_mergers/synthetic_merger_comparators_k15.parquet` |
| Synthetic firm GMMs | `output/case_studies/synthetic_mergers/synthetic_firm_gmm_parameters_k15.parquet` |
| Run metadata | `output/case_studies/synthetic_mergers/run_metadata.json` |
| Email draft | `docs/epics/week2_firm_portfolios/synthetic_merger_arthur_reply.md` |

The three output parquets are not committed to git; they ship via the case-study bundle.

---

## 8. Promotion path to production Step 4

This prototype does NOT authorize production synthetic-portfolio methodology. Findings and methodology observations from this run that shape the promotion path:

- **Refit noise is NOT a ΔBC contaminant** at the external-firm level (§6.1). The Step 4 spec can rely on this invariant — synthetic-portfolio ΔBC does not require a same-seed refit baseline to be meaningful.
- **VI-sensitivity to data perturbation IS a ΔBC contaminant** (§6.2). The Step 4 spec must require a per-pair null-distribution calibration — random-patent-injection over 20–50 seeds — before any ΔBC can be cited in a causal claim. Without this calibration, no per-pair ΔBC on a small-target merger can be distinguished from the VI-perturbation floor.

Remaining requirements before promoting:

1. **ADR-008** — synthetic-portfolio construction decision (concatenation vs weight-averaging; co-assignment handling; K_max for synthetic firms; which BC variant; random-patent-injection null as a spec-level input).
2. **Interface spec** (`docs/specs/synthesis_spec.md`) — `SyntheticPortfolioBuilder` contract, tests, round-trip serialization, null-calibration interface.
3. **Codex review** of both the ADR and the spec before any implementation.
4. **TDD module** in `src/synthesis/` (likely with `src/comparison/bhattacharyya.py` as a dependency once that module exists).
5. **Scaling decision** — universe-wide BC per synthetic firm, with SG-block caching analogous to the production BC pipeline.
6. **Significance framework** — null-distribution calibration via random-patent-injection (§6.2); bootstrap or permutation tests over patent-to-firm assignment for stronger causal claims.

Until the above steps are complete, the numbers in this prototype should be presented as "preliminary / methodology demonstration" rather than "findings".
