# Week 2 Handover Planning — Instance Summary

**Session**: 2026-04-15 (planning, no execution)
**Author**: Claude Opus 4.6 (1M context)
**Target audience**: The next Claude instance executing the handover plan
**Companion documents**:
- `/home/torrin/.claude/plans/imperative-conjuring-wand.md` — the concrete execution plan (READ THIS)
- `docs/epics/week2_firm_portfolios/week2_handover_bootstrap_prompt.md` — your initialization message from Torrin

---

## What this document is

This is the context package for executing the Week 2 team handover. It answers three questions:

1. **What was done in this planning session?** (so you don't redo it)
2. **WHY was the scope narrowed to a minimum-viable handover?** (so you don't re-open settled debates)
3. **What are the open tensions, risks, and things I'm not confident about?** (so you bring judgment, not just compliance)

If you're the next instance, read this first, then the plan file, then start on Task #76.

---

## Project context (for grounding)

**Research question**: Can firms' patent portfolios be used as a predictor of M&A pairs in the Technology Sector?

**Team**: Ananya Ravichandran, Arthur Khamkhosy, Torrin Pataki (dev lead), Amie Le Hoang, Duncan Harrop. Advisor: Jan Bena (UBC Sauder).

**Torrin's role**: Sole developer on a team of econ/stats students and faculty. His job is to deliver a clean, defensible dataset + methodology so teammates can run regressions, write the paper, and build the presentation. He is Bayesian-statistics trained; tolerant of high latency in exchange for thoroughness; leverages Codex (an "impartial reviewer, pair-programmer, deployment lead" per CLAUDE.md) for spec/ADR review.

**Pipeline architecture** (four steps):
1. Vectorize patents (PatentSBERTa 768D title+abstract + 768D citations → 1536D concat → UMAP 50D) ✓ done (Week 1)
2. Firm patent portfolios (Bayesian GMM per firm over 50D patent vectors) ✓ done (K=15 production, 7,485 deduplicated firms)
3. Compare distributions (Bhattacharyya Coefficient ✓ computed via sweep; directional complementarity ✗ deferred)
4. Extensions (synthetic portfolio matching) — future

**Where we are in the 4-week pipeline**: End of Week 2 deliverable (firm portfolios + BC), transitioning to teammates for Week 3+ (economic analysis, paper, presentation). Production fitting + corrected BC recomputation ran 2026-04-12 on AWS c5.4xlarge.

---

## What I (this planning instance) did

### 1. Phase 0 contract correction (EXECUTED — file edits made, NOT yet committed)

**Files modified** (working tree only, awaiting commit):
- `src/config/config.yaml` — `k_max: 10 → 15`, added `k_max_reference: 10` + `k_max_sweep: [10,15,20,25,30]`, refreshed stale "pending EDA" comment on normalization, added `firm_gmm_parameters_reference` path
- `docs/adr/adr_004_k_selection_method.md` — updated status line ("K_max=15 production locked"), date line, `n_components=K_max,` comment ("Production default: 15"), `K_max=10 converged production default` → `K_max=15 (production locked, 2026-04-14)`, added a "Production K_max Decision (2026-04-14)" subsection explaining mega-firm K-ceiling rationale, updated Consequences section, updated function signature default to `K_max=15`
- `docs/specs/firm_portfolio_spec.md` — ADR Dependencies line, config block (k_max 10→15), Output File Inventory (K=15 primary, K=10 reference), Week 3 contract

**Why**: The original K_max=10 decision was a valid convergence-based choice, but all 5 mega-firms (IBM, Intel, Qualcomm, Google, Cisco) saturate at K=10. The K15→K20 transition also passes every convergence threshold (ρ=0.9925, top-50=100%), so K=15 is a "free upgrade" in representational headroom for large firms without harming the ranking stability guarantee. Torrin locked this on 2026-04-14 in conversation.

**Confidence**: High. The K=15 decision was direct user input; the contract corrections are mechanical (find/replace with verification via `grep -n "k_max: 10" …`).

**What's NOT done**: None of this is committed. Task #76 handles the commit.

### 2. `docs/specs/comparison_spec.md` drafted (EXECUTED — file exists, has 5 Codex review findings outstanding)

**What it is**: BC-only specification (directional complementarity split out per Codex's earlier guidance). Defines `BhattacharyyaComputer` class with `compute_bc`, `compute_sg_block`, `compute_bc_matrix`, `load_bc_matrix` methods. Documents the linear-weighted BC formula (πᵢπⱼ, bounded in [0,1]) explicitly and names the √-weighted variant as a known bad path.

**Why split from complementarity**: The directional complementarity formula does not exist yet — it needs ADR-008 to select among candidates (asymmetric BC, KL-based, optimal transport, Bena-Li complement). Bundling BC (implementable today) with complementarity (still-open design) would gate BC on an unfinished design review. Split lets BC proceed while complementarity goes through its own ADR→spec→impl cycle.

**Codex review findings (not yet fixed)**:
1. **Config key mismatch**: constructor docstring references `chunk_size`; config block uses `sg_block_chunk_size`
2. **Missing `comparison:` section in actual `src/config/config.yaml`**: the spec shows what it should look like, but the real config file has no such section
3. **Row ordering contract not explicit**: semantically-identical matrices with different row orders break regression tests
4. **Dedup provenance not enforced**: module accepts arbitrary `list[GMMResult]` with no guardrail that input is deduplicated → silent recreation of original top-pair bug possible
5. **Metadata schema inconsistency**: compute docstring lists one key set; storage section adds `source_gmm_path` not present in the compute docstring

**My judgment**: These findings are legitimate and must be fixed before the BC module is implemented. But fixing them is NOT gating for tomorrow's handover (the notebook inlines BC helpers, doesn't import from `src/comparison/`). I put these as Step 7 "if time remains" in the plan.

**What's NOT done**: The 5 edits. Task #82 handles them if time remains; they can slip to post-meeting.

### 3. Plan file fully rewritten for handover scope (EXECUTED)

`/home/torrin/.claude/plans/imperative-conjuring-wand.md` is now a ~700-line execution plan with:
- Context (what's done, what's needed)
- 8 sequenced steps
- Files-to-modify tables
- Critical implementation notes
- Verification criteria per step
- Appendix A: exact git commands
- Appendix B: paste-ready code snippets (BC formulas, loaders, helpers)
- Appendix C: expected output scales (so the next instance knows what's "right")
- Appendix D: troubleshooting (common failure modes)
- Appendix E: 10 explicit "do NOT do this" guardrails

**Why the plan was rewritten twice**: original plan had Phase 0/1/2/3 structure assuming 2-4 weeks of work. When Torrin announced a tomorrow handover, Codex pushed back with specific revisions: (a) artifact access via email, not S3/repo; (b) don't commit generated outputs; (c) reorder so bundle + notebook + email come before BC spec polish; (d) use "active branch" not "master". I absorbed all those.

### 4. Tasks reshaped for handover execution (EXECUTED)

Deleted tasks: #66, #68, #72, #73, #74, #75 (all Phase 3 work — out of scope).
Created tasks: #76 through #83 (handover Steps 1-8 mapped to plan sections).

**The next instance should claim tasks in ID order starting #76.**

### 5. Memory files (minor updates during session)

- `project_kmax_sweep_result.md` — updated to record K_max=15 production lock (2026-04-14 decision)
- `feedback_semantic_parity_over_byte_identical.md` — new feedback memory: prefer np.allclose + effective K + BC≈1 over byte-identical parquet hashing for module regression tests
- `feedback_gaussian_before_hyperparameter.md` — new feedback memory: test model-form adequacy before running comprehensive hyperparameter sweeps

These reflect preferences surfaced this session that should carry forward.

---

## WHY each major decision

### Why K_max=15 over K_max=10

User's explicit choice. Both pass convergence thresholds. K=10 was my initial recommendation because convergence is literally proven there; K=15 is a team judgment call that prioritizes mega-firm representational fidelity over parsimony. The production decision is the team's call, not the methodology's — methodology only certifies that both choices are valid. K=10 is retained as the convergence-floor reference artifact.

### Why handover tomorrow is feasible as MVP, not full completion

Codex's framing (which I endorsed): running a Gaussian adequacy diagnostic the day before a team meeting creates compound narrative risk. If it passes, we've added value. If it comes back ambiguous or fails, we have to explain a partial finding in 24 hours while also handing over the dataset. The expected value of shipping adequacy tomorrow is negative-conditional-on-unclear-outcome. Ship the dataset now with an honest "adequacy audit in progress, week 1-2" roadmap entry; run the diagnostic in the week following with time to think about results.

**Corollary**: the teammates don't actually need adequacy to start. Their first week of analysis is exploratory regressions on the BC matrix. Rework cost from a later adequacy finding is small and expected in research workflows.

### Why defer BC module TDD extraction from the handover

The validated dataset already exists (`firm_gmm_parameters_k15.parquet`, corrected BC matrix). Teammates consume the dataset, not the code that produced it. Module extraction is engineering-quality work: it matters for future regeneration, CI, and production scaling, but does not gate economic analysis. Codex pointed out this directly, and it's correct.

**Risk of extracting under deadline**: rushed TDD produces brittle tests; rushed implementation introduces regressions; rushed validation skips the semantic-parity check against the validated artifact. Better to lift it deliberately in Week 2-3.

### Why split BC and complementarity into two specs + two ADRs

Complementarity has no formula yet. Four plausible candidates (asymmetric BC, KL-based, optimal transport, Bena-Li-style complement) have different economic interpretations and different publishability risks. Picking one without the ADR-review cycle is cargo-cult spec-driven development — the whole point of the ADR gate is to evaluate alternatives. Splitting the specs lets BC ship while complementarity gets proper design.

### Why email the bundle instead of git-lfs / repo-hosted

Torrin explicitly said he can email files. The alternative (teaching teammates to configure AWS CLI with his credentials) is a day-one blocker we'd have to solve right now. Email-delivered bundle + SHA256 verification + documented local paths is simpler, faster, and entirely sufficient for a 4-person team. Git-LFS would also work but adds repo-complexity no one needs.

**Trade-off accepted**: regeneration requires re-running the sweep on EC2 or re-downloading from S3. That's fine — it's a week-1 teammate task at worst, and the ops scripts already exist.

### Why keep the nested `output/kmax_sweep/corrected/output/kmax_sweep/` path

It's a sync artifact from the aws s3 sync commands that landed the corrected run. Flattening would require (a) moving files, (b) updating every reference in scripts/regenerate_notebook_pngs.py + existing notebooks + this plan, (c) re-verifying nothing broke. The next instance's time is better spent on the notebook than on a low-value cleanup. Document the quirk; address in a future cleanup task.

### Why the co-assignment audit IS included but ~comprehensive caveats are NOT

30-minute audit with high interpretive value: it directly shapes how teammates read BC-based regression coefficients. Without it, they'd likely discover on their own that "top BC pairs share patents" and wonder why we didn't flag it. Including it is cheap and professional.

**Restrained interpretation is deliberate** (per Codex): overclaiming causality — "BC measures already-joint-ventured firms" — would overstate a correlation finding. The right language is "N of top-100 pairs share >10% patents, consider a shared-patent control."

### Why README keeps the academic sections

Teammates will want the research framing eventually (for the paper). Stripping it forces them to ask. Adding quickstart at the top gets them running today without losing the context. Layered document: actionable up top, foundational below.

---

## What I'm confident in

1. **K=15 production lock is correct and documented.** Three files updated, memory updated, verification passes.
2. **The linear-weighted BC formula is correct.** It's been validated at scale on c5.4xlarge, the notebook assertion will hold, and the formula in plan Appendix B.2 is lifted verbatim from the production run script.
3. **The handover scope is right.** Codex agreed, I agree, the user approved. Notebook 04 + README + email + bundle is the minimum-viable professional handover.
4. **The co-assignment audit interpretation should be restrained.** Causality claims here would embarrass the team later.
5. **Deferring BC module TDD is correct.** It's engineering polish, not science; not a teammate-gating item.
6. **The nested path quirk is documentable.** Fixing it is future work, not tomorrow's work.

## What I'm NOT confident in

1. **Exact co-assignment audit numbers.** I estimated (in plan Appendix C) that most top-100 pairs have 0 shared patents with 5-20 above 10% overlap. **This is a guess.** If the distribution is dramatically different — especially if >50 pairs have >25% overlap — that's a signal the dedup rule missed a duplicate class, and the next instance should flag rather than ship silently.

2. **Whether all 7,485 firms have gvkey_map entries.** I assume the `gvkey_map.parquet` is complete for the deduplicated set, but I haven't verified. The co-assignment script has a `firm_patents.get(gv, set())` fallback, which is safe, but if many firms have no patent list in the map, the audit results become meaningless. Next instance should spot-check: how many top-100 firms have `len(firm_patents[gv]) > 0`.

3. **Whether Torrin's teammates have the venv/Python setup already.** The README quickstart assumes they can `pip install -r requirements.txt` without friction. If they're new to Python virtualenv, the quickstart might need screenshots or a video. I don't know their technical background beyond "econ/stats students."

4. **Whether the notebook's "4-6 hours to ramp" estimate is right.** This is a guess based on Notebook 03's complexity and typical walkthrough time. The notebook itself should be clear enough that 4 hours is comfortable, but I haven't tested it with a non-developer.

5. **Whether Jupyter can render the notebook on teammates' platforms.** Assumed Mac/Linux; should probably mention Windows WSL too. Or just test on their setup.

6. **Whether Codex's 5 BC spec findings are the complete set.** These were findings from one review pass. A second review after edits might find more. The spec should not be labeled "approved" until a clean review pass. Step 7 in the plan is necessary but may not be sufficient — the spec may need another iteration cycle post-edits.

7. **Whether the `firm_gmm_parameters_k15.parquet` + `bc_matrix_all_k15_dedup_linear.npz` combination has the same firm ordering.** Should be the same (both produced from the same recompute run), but the notebook's sanity check (`shape == (7485, 7485)` and gvkey count match) is important.

8. **Whether the co-assignment script will load the 28M-pair BC matrix in reasonable time on WSL.** 407MB .npz file, 7485×7485 float64 = ~448MB uncompressed. Should fit in memory on any modern laptop. But if WSL2 has memory pressure, the script might swap. Next instance should verify the load succeeds before proceeding.

## What I didn't do and why

- **Did not commit anything.** The session was planning + spec drafting + contract editing. The user explicitly wanted to hand off to a fresh instance for execution. Commits are Step 1 of the plan.
- **Did not run the co-assignment audit.** Would've been trivial (30 min), but the plan puts the script creation + execution in Step 4 for the executing instance. Doing it here would fragment the commit history.
- **Did not build notebook 04.** Similar reasoning: the executing instance handles this in Step 3. Also, notebook building under plan-mode constraints would have been awkward.
- **Did not fix the comparison_spec.md Codex review findings.** Step 7, explicitly marked optional/if-time. The plan describes each edit concretely.
- **Did not run Gaussian adequacy or pruning-threshold audit.** Deferred to post-handover per scope decision.
- **Did not extract `src/comparison/` or `src/portfolio/` modules.** Deferred per Codex revision.
- **Did not update `.gitignore`.** `output/` is already gitignored per CLAUDE.md's data-files-not-in-git policy; no action needed.

---

## Open tensions and tradeoffs the next instance should be aware of

### Tension 1: "Done tomorrow" vs "done right"

The handover IS minimum-viable. Some pieces (module TDD, full Codex-approved BC spec, comprehensive robustness audits) are deferred for honest reasons. If the next instance feels pressured to do "just one more thing" to make the handover feel complete, resist. The caveats section is the honest surface; making the caveats shorter by rushing half-done work is worse than leaving them as-is.

### Tension 2: Codex's authority vs user's priorities

Codex had specific revisions (artifact access, don't commit outputs, narrow co-assignment interpretation) that the user implicitly accepted by approving the plan. If you encounter a situation where Codex's prior guidance seems to conflict with what the user is asking — ask. Don't unilaterally resolve in Codex's favor or the user's favor. The project has a specific Codex review culture (CLAUDE.md) that treats Codex as authoritative for spec/ADR/implementation review gates, but Codex isn't a substitute for user judgment on scope.

### Tension 3: Teammate ramp quality vs polish breadth

Notebook 04 could be 2 hours of skeleton or 8 hours of polished examples. The plan aims for ~5 hours. If you find yourself tempted to add a 6th or 7th deep-dive section, remember: Notebook 03 is 1.1MB and already provides the methodology story. Notebook 04 is the loader+explorer+caveats entry point. Overloading it creates decision fatigue for teammates.

### Tension 4: Accuracy of "what's shipped" claims in the caveats table

The Section 7 caveats table in notebook 04 and README "Open Items" must match reality exactly. If you finish Step 7 (BC spec fixes), update the status for that row. If you don't finish it, leave it as "reviewed, revisions in progress." The team email should match. **Any mismatch between what the email says, what the README says, and what the notebook says is a reputation risk** — teammates will notice and lose trust in the other claims.

### Tension 5: Co-assignment result might not be small

If the audit returns, say, 40 pairs with >25% overlap, the notebook's interpretive paragraph changes character — it's no longer "a small caveat to note" but "a structural finding that reshapes interpretation." In that case:
- Do NOT ship the handover with a mild-sounding caveat. The interpretation must match the data.
- Consider whether the deduplication rule needs strengthening (currently containment ≥ 0.95; could it miss long-tail near-duplicates?).
- Flag to the user. This is above your judgment line — it changes what the team is receiving.

The plan assumes the audit result is in the "small but non-zero" range. If it's dramatically different, halt and ask.

---

## Key references (read in this order if you need to catch up)

1. **The plan file**: `/home/torrin/.claude/plans/imperative-conjuring-wand.md` (your execution script)
2. **CLAUDE.md**: Project-level standards, Codex review checkpoints, design principles
3. **ADR-004**: K_max=15 production decision (just-edited)
4. **`firm_portfolio_spec.md`**: Module contracts for the (future, deferred) PortfolioBuilder and GMMFitter
5. **`comparison_spec.md`**: BC module spec draft (with 5 Codex findings unresolved)
6. **`notebooks/03_kmax_convergence_analysis.ipynb`**: Narrative convention template + `DEEP_DIVE_FIRMS` gvkey mapping to copy
7. **`scripts/recompute_bc_corrected.py`**: Source for `bc_mixture_linear` (linear πᵢπⱼ)
8. **`scripts/run_kmax_sweep.py`**: Source for `bc_component_matrix` (diagonal closed form)
9. **`scripts/identify_top_pairs.py`**: Pattern reference for top-k partner lookup
10. **`docs/epics/week2_firm_portfolios/kmax_sweep_executive_summary.md`**: Team doc style reference
11. **`docs/epics/week2_firm_portfolios/kmax_diagnostic_findings.md`**: Bug diagnosis story (dedup + BC formula)
12. **Memory files at `/home/torrin/.claude/projects/-mnt-c-Users-TPata-firm-pair-merger-prediction/memory/`**: User profile, feedback preferences, project state

---

## Operating norms for this project

The following are stable across sessions; internalize them before executing:

1. **Thoroughness over speed**. Torrin has high latency tolerance. Don't rush by skipping subagent exploration or verification.
2. **Leverage subagents** — `Explore` for codebase questions, `Plan` for design stress-tests, `api-docs-synthesizer` for unfamiliar libraries. The user views these as force multipliers.
3. **Spec-driven, TDD-required** (CLAUDE.md #3 and #4). Specs exist as contracts; tests exist before implementation. This is why we don't skip Codex review on BC spec even under time pressure.
4. **Codex review gates** halt implementation. Specs and ADRs need review before code. The next instance does NOT implement `src/comparison/bhattacharyya.py` — that's post-Codex-approval work.
5. **Over-document the "why"** (CLAUDE.md #2). ADRs, comments on rationale, not mechanics. If you introduce a non-obvious choice, write the WHY.
6. **Evidence-based engineering** (CLAUDE.md #7). Claims backed by artifacts, not assumptions. When saying "the dataset is validated," reference specific runs (20260412T043407Z) and specific metrics (ρ=0.991-0.993).
7. **Parsimony** (CLAUDE.md #10). Minimum complexity for the current task. The notebook inlines helpers rather than building `src/comparison/` early because that's what the current task needs.
8. **Memory hygiene**: update memory files when learnings emerge; don't write duplicate memories; reference the stable `/home/torrin/.claude/projects/-mnt-c-Users-TPata-firm-pair-merger-prediction/memory/` path.

---

## The single most important thing

**Ship the minimum-viable handover tomorrow with honest caveats.** Not more, not less. The next instance's job is execution fidelity against the plan, not plan re-design. If the plan feels wrong during execution, ask the user — don't silently deviate. If the plan feels right, execute it.

Torrin has been working on this project for weeks. The handover is the culmination of Week 2. It sets the tone for teammates' entire downstream work. A clean handover with honest caveats is worth more than a flashy one with hidden gaps.

Good luck.
