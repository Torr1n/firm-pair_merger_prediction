# Week 2 Handover Email — Draft

**To**: Ananya, Arthur, Amie, Duncan; cc Jan Bena
**From**: Torrin Pataki
**Subject**: Week 2 Handover — Patent Portfolio Distance Matrix Ready for Analysis

---

Hi all,

The Week 2 deliverable is ready. You now have a validated pairwise "technological distance" matrix over 7,485 deduplicated firms in the technology and biotech sectors — the first production input for our regression and hypothesis-testing work.

**Key numbers.**
- **7,485 firms** in the deduplicated set (464 aliases, subsidiaries, and predecessor records removed via a containment ≥ 0.95 rule).
- **K_max = 15** locked as production (ADR-004, 2026-04-14). Convergence was confirmed at K=10, and we chose K=15 to give mega-firms extra representational headroom without harming stability.
- **Spearman ρ = 0.991–0.993** and **top-50 pair overlap = 96–100%** across every K_max ∈ {10, 15, 20, 25, 30} transition. Rankings are stable.
- BC matrix computed with the corrected **linear-weighted formula** (bounded in [0, 1]); the earlier √-weighted variant was unbounded and caused the top-tail instability we caught on 2026-04-12.

**Bundle delivery.** The artifact bundle (~860 MB, 8 files plus `SHA256SUMS.txt`) is [attached / at link — Torrin to fill in]. Verify integrity when you extract it:
```
cd /path/to/bundle
sha256sum -c SHA256SUMS.txt
```
Then place the files per the instructions at the top of `README.md` or in Section 1 of Notebook 04.

**Notebook 04 is your entry point.** `notebooks/04_pipeline_output_overview.ipynb` walks you through loading the artifacts, a worked two-firm BC example (with a reproducibility assertion that recomputes BC from the formula and checks it against the matrix), finding top-k most-similar partners for any firm, distributional sanity plots, and the co-assignment caveat (important for regression design). Plan on 4–6 hours to feel comfortable with it. If you want the methodology defense story first, `notebooks/03_kmax_convergence_analysis.ipynb` is the convergence walkthrough.

**One caveat to flag up front.** We ran a co-assignment audit on the top-100 BC pairs: only 2 of 100 share more than 10% of their patents (both clear parent-subsidiary misses just below the 0.95 containment threshold), and 98 of 100 share zero patents. BC is substantially independent of co-assignment structure — the signal is genuinely distributional, not a re-discovery of existing joint ventures. Still, I recommend including `n_shared` or `jaccard` from `coassignment_audit.parquet` as a control covariate in your BC-based regressions to be defensive against the 2 outliers and the long tail we did not audit.

**What's still in progress** (1–4 weeks, staged delivery):
- Gaussian adequacy audit — checking whether the Bayesian GMM's Gaussian-component assumption holds empirically (Week 1-2)
- Pruning-threshold audit — sensitivity of effective K to the DP weight-threshold choice (Week 1-2)
- BC spec approval + BC module TDD (Week 1-3)
- Directional complementarity metric and v2 dataset (ADR-008 → spec → impl, Week 2-4)

None of these are blockers for you starting exploratory regressions on the BC matrix. If any of them turn out to be blocking, tell me and we can re-prioritize.

Happy to walk through Notebook 04 live in the meeting tomorrow if that's useful. Questions ahead of time are welcome — reach me directly.

Torrin

---

## Notes for Torrin (not part of the email above)

1. **Delete this notes section** before sending.
2. **Fill in the bundle delivery mechanism** in paragraph 3 (attachment vs. link).
3. **Optional pre-send check**: run `sha256sum -c SHA256SUMS.txt` yourself against the staging directory output from `scripts/build_handover_bundle.sh` before attaching/linking, so you're confident the checksums match.
4. **Meeting prep**: the caveats table in Notebook 04 Section 7 is the honest surface for "what's done vs what's in flight." Keep the live-walk of Notebook 04 in mind as a 20-minute option if teammates want a guided tour.
5. **If teammates ask "why K=15 and not K=10"**: both pass convergence thresholds; K=15 gives mega-firms (IBM, Intel, Qualcomm, Google, Cisco) extra representational headroom without harming ranking stability. See ADR-004's "Production K_max Decision (2026-04-14)" subsection.
