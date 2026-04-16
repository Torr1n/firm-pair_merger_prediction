# Week 2 Handover Email — Draft

**To**: Ananya, Arthur, Amie, Duncan; cc Jan Bena
**From**: Torrin Pataki
**Subject**: Week 2 Handover — Patent Portfolio Distance Matrix Ready for Analysis

---

Hi all,

The Week 2 deliverable is ready. You now have a validated pairwise "technological distance" matrix over 7,485 deduplicated firms in the technology and biotech sectors — the first production input for our regression and hypothesis-testing work.

**Jan — TL;DR for you.** The BC matrix is validated and ready for the team to begin analysis. Convergence is confirmed, dedup is applied, the formula bug from two weeks ago is fixed. Three methodology items (Gaussian adequacy audit, directional complementarity metric, production module TDD) are staged for delivery over the next 1–4 weeks. Full caveats table in Notebook 04 Section 7. Happy to sync separately on any of it.

**Key numbers.**
- **7,485 firms** in the deduplicated set (464 aliases, subsidiaries, and predecessor records removed via a containment ≥ 0.95 rule).
- **K_max = 15** locked as production (ADR-004, 2026-04-14). Convergence was confirmed at K=10, and we chose K=15 to give mega-firms extra representational headroom without harming stability.
- **Spearman ρ = 0.991–0.993** and **top-50 pair overlap = 96–100%** across every K_max ∈ {10, 15, 20, 25, 30} transition. Rankings are stable.
- BC matrix computed with the corrected **linear-weighted formula** (bounded in [0, 1]); the earlier √-weighted variant was unbounded and caused the top-tail instability we caught on 2026-04-12.

**Bundle delivery.** The artifact bundle is ~845 MB (9 files plus `SHA256SUMS.txt` and `MANIFEST.txt`), too large for most email attachments — I'm shipping it via [Google Drive link / UBC OneDrive link — Torrin to fill in]. The `MANIFEST.txt` at the top of the bundle records the git SHA, source S3 run, and ISO timestamp for versioning if we ever need to reproduce this exact cut. Verify integrity when you extract it:
```
# macOS / Linux / WSL:
cd /path/to/bundle
sha256sum -c SHA256SUMS.txt

# Windows PowerShell (no WSL): see README "Quickstart for Teammates" section for Get-FileHash equivalent.
```
Then place the files per the instructions at the top of `README.md` or in Section 1 of Notebook 04. Heads-up: the walkthrough reads from `output/kmax_sweep/corrected/output/kmax_sweep/` — the nested path is an S3-sync artifact we documented rather than flattened so that the bundle matches the cloud layout 1:1.

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
2. **Fill in the file-share link** in paragraph 3 (Google Drive, UBC OneDrive, Dropbox — teammates' choice). Email attachment will not work — 845 MB exceeds every major provider's limit. I verified the bundle at `output/handover_bundle_20260416/` end-to-end (8 content files + `SHA256SUMS.txt` + `MANIFEST.txt`, sha256 verified, Notebook 04 reproducibility anchor holds using bundle files alone).
3. **Optional pre-send check (already done once)**: run `sha256sum -c SHA256SUMS.txt` yourself in the staging directory to confirm checksums match after the upload round-trip.
4. **Meeting prep**: the caveats table in Notebook 04 Section 7 is the honest surface for "what's done vs what's in flight." A live 20-minute walkthrough of Notebook 04 is a good option if teammates want a guided tour. See `docs/epics/week2_firm_portfolios/week2_handover_morning_briefing.md` for anticipated questions + 1-sentence answers.
5. **If teammates ask "why K=15 and not K=10"**: both pass convergence thresholds; K=15 gives mega-firms (IBM, Intel, Qualcomm, Google, Cisco) extra representational headroom without harming ranking stability. See ADR-004's "Production K_max Decision (2026-04-14)" subsection.
6. **Jan paragraph is second** (right after the opening). It gives him a standalone TL;DR without forcing him to read the whole message. Delete or move if you prefer a different framing.
