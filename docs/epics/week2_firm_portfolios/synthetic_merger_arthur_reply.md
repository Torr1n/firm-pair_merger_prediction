# Email Reply to Arthur — Synthetic Merger Case Study

**To**: Arthur Khamkhosy
**From**: Torrin Pataki
**Subject**: Re: Synthetic merger case study — results + a methodology finding

---

Hi Arthur,

Ran the case study for all four pairs with the top-20 comparator set and added a refit-baseline experiment on top of what you asked for (rationale in §3 below). Full results are in `notebooks/05_synthetic_merger_case_study.ipynb` and the accompanying parquets — short summary here.

## 1. Feasibility + runtime

All four pairs ran in about 12 minutes locally (most of that in the baseline refit for the 40K-patent acquirer in pair 4). Your top-20 restriction kept the BC calls trivial; the bottleneck is the GMM fitting. Universe-wide would be another ~20 min per pair if we want that for the paper — not needed right now.

## 2. Headline numbers

ΔBC = BC(synth, f) − BC(acquirer, f), top-20 firms by BC(acquirer, *), excluding acquirer and target.

| Pair | Acquirer + Target | Target share | Pre-merger BC | Max ΔBC | Argmax comparator |
|---|---|---:|---:|---:|---|
| 1 | 001161 + 022325 | 20.5% | 0.0014 | **+0.027** | PRIV_AILTECHNOLOGIES |
| 2 | 001632 + 014256 | 20.6% | 0.0108 | −0.010 | 133288 |
| 3 | 007257 + 018510 | 0.79% | 0.0002 | +0.028 | 002403 |
| 4 | 005606 + 008633 | 0.023% | 0.0000 | −0.019 | PRIV_INDUSTECHNOLOGY |

Zero shared patents across all four pairs; the synthetic is a clean concatenation.

Look at pair 1 specifically. BC(acq, AILTECHNOLOGIES) = 0.0047, BC(tgt, AILTECHNOLOGIES) = 0.0014, BC(synth, AILTECHNOLOGIES) = **0.0321**. The synthetic is ~7× more similar to AILTECHNOLOGIES than either constituent firm alone. That's the candidate finding in the whole run.

## 3. The methodology finding (this is the important part)

I added a second GMM fit per pair: the acquirer alone, refit from scratch with the same priors, seed, and `n_init=5` as the synthetic. This gives a same-seed baseline `ΔBC_clean = BC(synth, f) − BC(acquirer_refit, f)` that cancels VI refit stochasticity at first order. My worry going in was that BC(synth, acquirer_production) comes out only ~0.03–0.06 self-similarity, which looked like refit noise could be contaminating everything.

The baseline experiment answered this cleanly. **Refit noise on BC-to-external-firms is effectively zero** — `ΔBC_raw` and `ΔBC_clean` correlate at 1.000 with `|raw − clean|` bounded by ~4×10⁻⁵. The production acquirer GMM and a same-seed refit disagree on *self*-BC (component-permutation non-identifiability — two fits represent the same density with different component centers) but agree on BC to any third firm to ~5 decimal places. So the raw ΔBC numbers above are already clean merger-effect estimates at the refit-noise level.

**But** this also means a different confound is now the binding one: **VI-sensitivity to small data perturbations**. The synthetic and acq_refit GMMs share `random_state=42` but see different data (combined vs acquirer-alone). Even tiny data changes can tip VI into a different local optimum, producing ΔBC shifts to external firms that aren't merger effects. Pair 4 is the tell — the target is 9 patents in a 39,728 merged portfolio (0.023%), mathematically far too small for its *content* to produce a 0.019 BC shift. So pair 4's ΔBC is effectively the empirical VI-perturbation floor for large-acquirer synthetic fits, around ~0.02. Pair 3 (99 target patents, max ΔBC 0.028) is probably in the same regime — we can't distinguish "small real signal" from "VI perturbation" without calibration.

Pair 1's 0.027 is the outlier worth paying attention to, because 20.5% target share with 4,466 patents is substantial enough that VI perturbation from *that* quantity of added patents isn't a trivial explanation. But we can't confirm without calibrating.

## 4. What's in the deliverable

In the bundle (`output/case_study_bundle_20260421/`):

- `synthetic_merger_pairs_k15.parquet` — per-pair metadata, patent counts, `target_patent_share`, `bc_acquirer_refit_vs_production`, `max_abs_refit_noise`, `max_delta_bc` (raw), `max_delta_bc_clean`, argmaxes.
- `synthetic_merger_comparators_k15.parquet` — 88 rows. Columns: `bc_acquirer_to_comparator`, `bc_target_to_comparator`, `bc_synthetic_to_comparator`, `bc_acquirer_refit_to_comparator`, `delta_bc` (raw), `delta_bc_clean`, `refit_noise`, `rank_in_acquirer_top20`, `target_patent_share`, `comparator_n_patents`, `comparator_tier`.
- `synthetic_firm_gmm_parameters_k15.parquet` — 8 rows: 4 synthetic firms (`SYNTH_{acq}_{tgt}`) + 4 acquirer refits (`REFIT_{acq}`). Schema-compatible with production; loadable via `scripts/run_kmax_sweep.load_gmm_results`. Reusable for any future Step 4 work without re-running VI.
- `run_metadata.json` — git SHA, input SHAs, per-pair timings.
- `notebooks/05_synthetic_merger_case_study.ipynb` — the walkthrough. Section 4 is the baseline-experiment verdict; Section 7 is the per-pair honest reading; Section 8 is methodology implications for Step 4; Section 10 is next steps.
- `docs/epics/week2_firm_portfolios/synthetic_merger_case_study.md` — methodology record + promotion path.

## 5. Methodology choices I made

Two choices that deviate slightly from your email:

- **Comparator set excludes the acquirer** (trivially BC=1) **and the target** (mechanically absorbed — uninformative). `BC(synth, acquirer)` and `BC(synth, target)` are kept as self-sanity rows (`role` column) in the per-comparator parquet — filter `role == 'comparator'` for the 20 hypothesis-relevant rows per pair.
- **Acquirer-refit baseline added on top of your spec** — it was cheap (~10 min extra) and gave a cleaner methodology story than we'd have without it.

## 6. Offer + next step recommendation

The one experiment I'd recommend doing before committing to any case-study conclusion in the paper/presentation is a **random-patent-injection calibration**: for each pair, sample `n_target` random patents from the acquirer's own portfolio, add them to the acquirer, refit, and compute ΔBC_null. Repeat 20–50 times per pair to build a null distribution of |ΔBC| driven purely by VI-sensitivity. Compare pair 1's 0.027 against pair 1's null distribution — if it's an outlier, you have a merger-effect signal; if it's within the null distribution, you don't. ~15 min per pair.

If you'd rather skip the calibration and go straight to universe-wide BC (drops the top-20 restriction), that's also feasible — ~20 min per pair. It doesn't address the VI-perturbation confound but gives you global ranking of the synthetic's neighborhood.

On pair 4 specifically: the 9-patent target can't drive a meaningful merger-effect measurement at all. If you have another acquisition with target >5% of combined patents I can swap it in cheaply (<15 min total including the baseline for the new pair).

Let me know which you'd like to chase first.

Torrin

---

## Notes for Torrin (not part of the email)

1. **Delete this section before sending.**
2. **Headline numbers** in §2 match `synthetic_merger_pairs_k15.parquet` for git SHA `{fill in from run_metadata.json}`.
3. **Bundle size**: ~1 MB, email attachment viable.
4. **The narrative changed after running the baseline.** Before: "refit noise dominates, need baseline to clean." After: "refit noise is nil; VI-perturbation-floor is the real confound." This matters if Arthur or Jan ask why we spent extra time on the baseline — it's because running it was what turned up the actual confound structure. That's a methodology win even if it makes the per-pair story messier.
5. **If Arthur picks random-patent-injection** (recommended): about a half-day of work including plumbing the null-distribution code, re-running across 20+ seeds per pair, and updating notebook 05 with a new section and the null-calibrated p-value per comparator. Worth doing before any pair's ΔBC is cited in the paper.
6. **If Arthur picks universe-wide first**: cheap to add, but doesn't change the methodological situation — it just gives a bigger comparator set. The null-calibration question remains.
7. **Methodology doc §8 "Promotion path"** now includes the random-patent-injection calibration as a spec-level requirement for ADR-008. Worth surfacing to Jan.
