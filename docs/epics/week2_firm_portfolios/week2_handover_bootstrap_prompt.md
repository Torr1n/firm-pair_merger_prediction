# Bootstrap Prompt for Downstream Handover Execution Instance

**Purpose**: This is the message Torrin will send to the next Claude instance to bootstrap it into continuing our work. It establishes context, values, and the task at hand — the next instance starts with no knowledge of our prior sessions.

---

## The prompt

> We have made significant progress over the last few weeks on this project. The research question is: can firms' patent portfolios be used as a predictor of M&A pairs in the Technology Sector? Building on Bena & Li (2014), we redefine "technological overlap" from a static similarity score to a probabilistic measure of overlap in entire patent portfolio distributions — "in what way are patent portfolios interacting in the technology space?"
>
> The pipeline has four steps: vectorize patents via PatentSBERTa + UMAP to 50D; aggregate per-firm vectors into Bayesian Gaussian Mixture Models (GMMs); compare firm pairs via the Bhattacharyya Coefficient (BC); and eventually extend to synthetic portfolio matching. Steps 1 and 2 are complete at production scale (7,485 deduplicated firms, K_max=15 locked), and a full-scale BC matrix has been computed and validated. The validated artifacts live locally under `output/kmax_sweep/corrected/output/kmax_sweep/` and on S3 at `s3://ubc-torrin/firm-pair-merger/week2/kmax_sweep/runs/20260412T043407Z-dedup-linear/`.
>
> Along the way, we caught and fixed two bugs that were silently corrupting our convergence results: 464 duplicate firms (aliases, subsidiaries, predecessor records with near-identical patent sets) were inflating top-pair BC values, and the original BC mixture formula used `√(πᵢπⱼ)` weighting which is an unbounded upper bound on true BC (observed values up to 5.39). The deduplication rule (containment ≥ 0.95) and the corrected linear `πᵢπⱼ` formula are now the production defaults. K_max convergence was confirmed at scale: Spearman ρ=0.991-0.993 and top-50 pair overlap 96-100% across every adjacent K_max transition from K=10 onward.
>
> **The immediate task**: I am meeting with my team tomorrow (2026-04-16) to hand over the dataset for their post-analysis, paper writing, and presentation creation. The prior instance planned the handover thoroughly. Your job is to execute that plan.
>
> **Before you do anything**, read these two documents — in order:
>
> 1. `/home/torrin/.claude/plans/imperative-conjuring-wand.md` — the concrete execution plan with 8 sequenced steps, exact git commands, paste-ready code snippets, expected outputs, troubleshooting, and guardrails
> 2. `docs/epics/week2_firm_portfolios/week2_handover_planning_instance_summary.md` — the WHY behind each decision, open tensions, and what the planning instance was NOT confident in
>
> Then check `TaskList` — there are 8 pending tasks numbered #76-83 mirroring the plan's 8 steps. Claim and execute them in ID order.
>
> **A few things I want you to internalize before you start:**
>
> - **Thoroughness over speed, always.** I have abnormally high latency tolerance. I would rather you spend six hours building a careful handover than thirty minutes shipping a rushed one. If a subagent investigation deepens your understanding, run it. If a Plan-agent stress-test of your approach catches an error before I do, the time was well spent. You are expected to leverage subagents (Explore, Plan, api-docs-synthesizer, context-synthesizer) aggressively — not hoard context-window space.
>
> - **Partnership, not compliance.** If something in the plan feels wrong once you're executing, surface it. If a decision point comes up that isn't explicit in the plan, ask. If Codex's prior guidance appears to conflict with new information, flag it. I'd rather iterate on approach than discover problems in the committed result.
>
> - **Spec-driven, test-driven, no exceptions.** This project's culture is defined in `CLAUDE.md`. ADRs before code. Specs as contracts. Tests before implementation. Codex reviews gate spec/implementation transitions. The handover plan explicitly defers some items (module TDD, complementarity, Gaussian adequacy) that this culture would require — that deferral is intentional and blessed; do not reopen it under time pressure. But do NOT rush code that does land in the repo.
>
> - **Evidence over assumption.** When you claim "the dataset is validated," reference specific runs (`20260412T043407Z`) and specific metrics (Spearman ρ=0.991-0.993, top-50 overlap 96-100%). When you write a caveat, make it honest about what's been checked and what hasn't. The team will trust what they can verify; they will lose trust on anything that turns out to be overstated.
>
> - **Over-document the "why".** If you make a non-obvious choice, write the rationale — in the commit message, in an ADR, in a comment. "What" is visible in the diff; "why" decays from memory within days. CLAUDE.md makes this our second design principle.
>
> - **Parsimony. Minimum complexity for the current task.** The notebook inlines BC helpers rather than building `src/comparison/` because that's what tomorrow needs. Do not add features, abstractions, or defensive code beyond the plan's scope. Three similar lines is better than a premature abstraction. Half-finished implementations are worse than no implementation.
>
> - **The team receiving this are my teammates** — Ananya, Arthur, Amie, Duncan (econ/stats students) and Jan Bena (advisor). They are not developers. The README quickstart, the notebook 04 walkthrough, and my email should speak to their technical level. They know what Bayesian GMMs and Bhattacharyya coefficients are conceptually; they probably don't know UMAP details or spot the difference between linear and √-weighted mixture formulas at a glance. Explain mechanics where non-obvious; don't lecture on what they already know.
>
> - **Codex is in our review loop.** Codex has already reviewed the BC spec draft (`docs/specs/comparison_spec.md`) and returned 5 legitimate findings (see `comparison_spec.md` top of file + plan Step 7). We will not implement from that spec until the findings are addressed and a clean review pass returns. For this handover, that means: the notebook inlines BC helpers, and the spec is marked "reviewed — revisions in progress, not yet approved for implementation" rather than fabricating approval. The distinction matters.
>
> - **The artifact bundle is ~860 MB** and will be delivered to teammates by email-attachment-or-file-share (my choice — I have it covered). The repo does NOT solve artifact distribution. The notebook and README tell teammates where to place files after they receive them. Do not add `output/*.parquet` or `output/*.npz` to git — those live in the handoff bundle only.
>
> - **Tomorrow's handover is intentionally minimum-viable.** The full roadmap (BC module TDD, pruning-threshold audit, Gaussian adequacy MVP, ADR-008 → complementarity v2 dataset, PortfolioBuilder/GMMFitter extraction) spans 1-4 weeks of post-handover work. Shipping the dataset now with an honest caveats table is the right move; rushing the roadmap into tomorrow is not. If you feel pressured to do "just one more thing," resist — the caveats section is the honest surface.
>
> - **You will not be able to ask me questions in real-time during execution.** I'm preparing for the meeting. So: if you hit a decision point that isn't resolved in the plan or the Instance Summary, ask via `AskUserQuestion` before proceeding, and I'll answer when I'm free. Don't guess on ambiguity.
>
> - **Ground all recommendations in current code, not just memory.** Memory files in `/home/torrin/.claude/projects/-mnt-c-Users-TPata-firm-pair-merger-prediction/memory/` are point-in-time; verify against current code before asserting as fact. If a memory says "X exists at path Y," check `git log` and read the file.
>
> Your immediate next action: read the plan file and Instance Summary in full, then `TaskList` to see #76-83, then claim #76 and begin execution. Consult `CLAUDE.md` for standing project norms as needed.
>
> Let me know if there is anything I can clarify, any resources I can provide, any decisions you need my input on before proceeding. This is a partnership and partnership relies on collaboration.

---

## Notes for Torrin (not part of the prompt above)

1. **Deliver this message verbatim** to the next instance at session start. The content and ordering are deliberate — it mirrors your original bootstrap style so the instance recognizes the pattern and activates the right working mode.

2. **You may want to append, at the very end of the prompt, one line confirming your availability for questions during execution** (e.g., "I'll be periodically checking in — flag anything that needs my input"). The prompt above already tells them to use `AskUserQuestion` if you're not reachable, so either mode works.

3. **The plan file is the execution document; the Instance Summary is the context document; this bootstrap prompt is the framing document.** Three layers, each for a different purpose. Don't duplicate content between them — they reinforce each other via cross-reference.

4. **If you need to adjust scope before sending** (e.g., you decide to include Gaussian adequacy after all), update the plan file first, then update the Instance Summary's "what was done / not done" sections accordingly, then this prompt. Internal consistency between these three documents is the main reputation risk.

5. **If the next instance pushes back on the plan** — e.g., they have a better sequencing or catch a bug in the plan — take that seriously. The plan is a current-best-effort snapshot, not scripture. Their perspective is fresh.
