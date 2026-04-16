# Morning Briefing — 2026-04-16 Team Meeting

**Audience**: Torrin (you), reading this 15 minutes before the meeting.
**Purpose**: quickly reload everything you need to run the handover well without re-reading the plan file, the instance summary, or the notebook.

---

## The 60-second version

The Week 2 deliverable is done. A validated 7,485×7,485 Bhattacharyya Coefficient matrix over deduplicated tech+biotech firms, K_max=15 production, corrected linear-weighted formula. Teammates (Ananya, Arthur, Amie, Duncan) receive the data; Jan gets the advisor summary; everyone can start exploratory regressions today and the paper/presentation track runs in parallel. Three methodology items (Gaussian adequacy, complementarity, BC module TDD) are in a staged 1–4 week post-handover delivery.

---

## Pre-meeting checklist (≤ 10 minutes)

- [ ] **Upload the bundle** from `output/handover_bundle_20260416/` (845 MB, 10 files) to a file-share link. Google Drive is the simplest — make a folder, upload all 10 files, share "anyone with the link can view," paste the link.
- [ ] **Open `docs/epics/week2_firm_portfolios/week2_handover_email.md`**, delete the "Notes for Torrin" section, paste the file-share link into paragraph 3, send to Ananya/Arthur/Amie/Duncan cc Jan.
- [ ] **Have `notebooks/04_pipeline_output_overview.ipynb` open** in Jupyter or in a browser tab on GitHub (the notebook renders with all 3 plots on GitHub directly). You may want to walk through it live if teammates ask.
- [ ] **Post-meeting follow-up**: if anyone flagged a blocker, update the roadmap priorities in Notebook 04 Section 7.

---

## Headline numbers (memorize these)

| What | Number | Why it matters |
|---|---|---|
| Firms in the deduplicated dataset | **7,485** | Rows/cols of the BC matrix |
| Duplicates removed | **464** | Aliases, subsidiaries, predecessors (containment ≥ 0.95) |
| K_max production | **15** | Locked 2026-04-14 per ADR-004 |
| K_max convergence floor | **10** | Reference artifact also shipped |
| Spearman ρ across K_max transitions | **0.991–0.993** | Rankings stable |
| Top-50 pair overlap across K_max | **96–100%** | Rankings stable |
| Top-100 pairs sharing zero patents | **98 / 100** | BC is distributional, not structural |
| Top-100 pairs with >10% patent overlap | **2 / 100** | Dedup misses (see below) |
| BC formula | **linear-weighted πᵢπⱼ** | Bounded in [0, 1]; fixes the √-weighted bug |
| Bundle size | **845 MB** | Too big for email; use a link |

---

## Two outlier pairs worth naming if the co-assignment audit comes up

If anyone asks "what are the 2 pairs with >10% overlap?":

| Rank in top-100 | Firm A | Firm B | BC | Shared / min | Our read |
|---|---|---|---|---|---|
| 37 | `060888` | `PRIV_OBLONGINDUSTRIES` | 0.595 | **94% overlap** | Parent + subsidiary; dedup missed because containment came in around 0.94 (just below our 0.95 rule) |
| 20 | `063083` | `PRIV_ENDOLOGIX` | 0.669 | **75% overlap** | Similar — private predecessor record with heavy patent overlap with the public entity |

These don't invalidate anything; they're an argument for including a `n_shared` control in BC-based regressions.

---

## Likely questions + 1-sentence answers

**"Why K=15 and not K=10?"**
Both pass the convergence thresholds identically; K=15 gives mega-firms (IBM, Intel, Qualcomm, Google, Cisco — all saturate at K=10) extra component headroom without harming rank stability for the rest of the dataset. See ADR-004's "Production K_max Decision (2026-04-14)" subsection.

**"Why linear-weighted BC and not √-weighted?"**
The √-weighted variant is an upper bound that can exceed 1.0 for multi-component mixtures (we saw values up to 5.39). That unboundedness was the root cause of the K_max=15→20 top-tail instability we caught on 2026-04-12. The linear-weighted formula is bounded in [0, 1] by Cauchy–Schwarz, and Notebook 04 Section 3 asserts matrix-vs-formula agreement as a reproducibility anchor.

**"Where are the top pairs — is IBM + Intel at the top?"**
No — the top-5 BC pairs are all private-firm pairs (`PRIV_*`) with small patent counts (5–83 patents each), because BC for small-K small-portfolio firms is very sensitive to cluster overlap. The economically-interesting pairs are further down the ranking; your regressions should use the full matrix rather than just the top-k.

**"Did you verify the Bayesian GMM's Gaussian assumption?"**
No — the Gaussian adequacy audit is explicitly deferred. This is the Week 1-2 follow-up item. If it finds the assumption is violated, the short-term fix is to mark the specific firms as failed-adequacy and flag them in regressions; the long-term fix is a Student-t mixture (contingent on audit outcome).

**"Can we run regressions on the BC matrix now?"**
Yes for exploratory. The caveats table in Notebook 04 Section 7 tells you which items would change your claimed effect sizes; Gaussian adequacy and pruning-threshold sensitivity are the main ones that might shift coefficients.

**"What if we need a different K_max for our analysis?"**
The K=10 convergence-floor artifact is shipped alongside K=15 for exactly this reason. Robustness checks should confirm your result holds at both K settings. Running further K_max values requires re-running `scripts/run_kmax_sweep.py` on EC2 — ask me.

**"Can we get the raw patent vectors?"**
They're on S3 at `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/` — ping me for AWS read access if you need them. For most hypothesis work the GMM parameters + BC matrix are enough.

**"What's the directional complementarity metric you mentioned in earlier messages?"**
That's a separate metric measuring "firm A fills the gaps in firm B's technology portfolio" (asymmetric). It needs an ADR (ADR-008) to pick between candidate formulas (asymmetric BC, KL-based, optimal transport, Bena–Li complement). v2 dataset in 2-4 weeks.

**"Who do we contact with questions?"**
Me (Torrin) for anything. Methodology questions → reference the ADRs. Blockers → ping directly, we'll re-prioritize.

---

## If you want to do a live walkthrough in the meeting (20 min)

Rough script:
1. (2 min) **Open Notebook 04, show Section 1**. "These are the files you've been handed; here's where they go." Reassure about the nested path.
2. (3 min) **Section 2 sanity checks**. Show the matrix shape, symmetry, diagonal=1. "This is where you confirm the files loaded correctly."
3. (5 min) **Section 3 worked example**. The IBM-Intel BC computed two ways with an assertion. "This is the reproducibility anchor — if your matrix ever doesn't match the formula, something has drifted."
4. (3 min) **Section 4 top-k partners**. Run `top_k_partners("006066", 20)` live. Point out that top pairs are small-PRIV firms so they should use the full matrix.
5. (3 min) **Section 6 co-assignment**. Load the parquet, show the 2 outliers, recommend `n_shared` as a regression control.
6. (3 min) **Section 7 caveats table**. Walk through what's done vs. what's coming, give the 1-4 week timeline.
7. (1 min) Close with "questions about any of this before you start?"

---

## What NOT to claim

- ❌ The BC module is TDD-production-grade. (It's ad-hoc script code today; TDD extraction is Week 2-3.)
- ❌ The Gaussian assumption is validated. (It's not yet; adequacy audit is Week 1-2.)
- ❌ Complementarity is available. (It's still ADR-pending; v2 dataset in Week 2-4.)
- ❌ The pruning threshold is audited for sensitivity. (Not yet; Week 1-2.)
- ❌ `comparison_spec.md` is approved. (It's reviewed, 5 findings addressed today, second Codex pass pending.)
- ❌ `src/comparison/bhattacharyya.py` exists. (It does not; production logic lives in `scripts/recompute_bc_corrected.py`.)

If teammates ask about any of these, the honest answer is in the caveats table.

---

## Meeting opening line options

- **Short & factual**: "The dataset is validated and ready. I've sent the bundle; notebook 04 is your walkthrough. Let's go through what's done, what's in flight, and how you'll use it."
- **With warmth**: "Thanks for being patient through the past week — there were two bugs to untangle and a K_max decision to think through, but we're in a good place. Here's what you've got."
- **If the dataset discussion is quick**: "Dataset is ready. Let's spend most of the time on your regression design — I want to make sure the matrix is useful for what you're actually going to do with it."

Pick whichever voice matches the room. The room — Jan, four students — will probably appreciate the second.

---

## One last thing

Nothing in the handover is load-bearing on perfect delivery today. The data has been validated; if we have to follow up on anything, we will. Don't rush the walkthrough if teammates are engaged. Good luck.
