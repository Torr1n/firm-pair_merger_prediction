# Instance Summary: Week 1 Development Session

**From**: Claude Code (outgoing Week 1 development instance)  
**To**: Incoming development instance  
**Date**: 2026-04-07  
**Session span**: 2026-03-31 through 2026-04-07  
**Repository**: `git@github.com:Torr1n/firm-pair_merger_prediction.git` (branch: `master`)  
**HEAD at handover**: commit `ef575ca`

---

## Purpose of This Document

You are bootstrapping into the firm-pair merger prediction project. This document is your primary context package. It captures not just *what* was built, but *why* each decision was made, *how* we work with Torrin, and *where* the project is going. Read this before touching code.

This session spanned the full Week 1 sprint — from an empty repository to a production pipeline running on AWS. You are picking up at the transition between Week 1 (patent vectorization) and Week 2 (firm portfolio construction via GMMs).

---

## Who You're Working With

### Torrin Pataki — Development Lead
- 5th-year UBC Combined Major in Business & CS, Statistics Minor
- NLP expert on a team of four economics/statistics honors students
- Previously led the Financial Topic Modeling (FTM) project through 12+ sprints to production on AWS — this project inherits FTM's development culture
- Faculty advisor: Jan Bena (UBC Sauder), co-author of Bena & Li (2014)
- Uses voice transcription workflow — his messages are often long, conversational, and may contain tangents. The core directive is usually in the first and last paragraphs.

**How Torrin works:**
- He thinks architecturally first, then drills into implementation
- He values thoroughness over speed — he has explicitly said "I care far more about the quality of your work than the amount of time you spend, the latency of your response, or the number of tokens you burn"
- He gives direct, honest feedback. When something is wrong, he says so. When something is good, he moves fast.
- He trusts the agent to make judgment calls but expects to be consulted on irreversible decisions
- He communicates with his team in a warm, casual tone (first-name basis, "hey guys," humor) but expects the technical work underneath to be rigorous
- He skipped the formal Codex design gate in Week 1 (approved proceeding to implementation before Codex reviewed ADRs) — a pragmatic call that he acknowledged carried risk. Codex's subsequent review caught a critical bug, which validated the value of the gate.

**Key phrases he uses and means:**
- "Do your due diligence" — be thorough, don't cut corners, leverage sub-agents
- "Boring technology" — use well-understood tools (scikit-learn, pandas, parquet), not exotic ones
- "Over-document the why" — every non-obvious decision needs rationale, not just the what
- "Spec-driven development" — write the spec, get it approved, then build to spec
- "Evidence-based" — claims must be backed by artifacts (histograms, timing tables, coverage stats)

### The Team
| Name | Role | Interaction Pattern |
|------|------|-------------------|
| **Arthur Khamkhosy** | Methodology design | Authored the 4-step pipeline. His email is the ground truth for methodology. |
| **Amie Le Hoang** | Data preparation | Responsive, iterative. Delivered v1 → v2 → v3 data within a week. Ask her directly about data questions. |
| **Ananya Ravichandran** | Researcher | |
| **Duncan Harrop** | Researcher | |
| **Jan Bena** | Faculty advisor | Co-author of Bena & Li (2014). Has AWS account. Cloud admin is David. |
| **Codex** | Impartial reviewer | Long-running agent. System prompt at `docs/epics/codex_system_prompt.md`. Reviews code, Terraform, deploys infrastructure. Found real bugs (resume corruption, encoding duplication). |

---

## The Research Question

*Can firms' patent portfolios predict M&A pairs in the Technology Sector?*

Building on Bena & Li (2014), the project redefines "technological overlap" from a static similarity score to a probabilistic measure of overlap in patent portfolio distributions. The pipeline has four stages:

1. **Vectorize Patents** (Week 1 — DONE): PatentSBERTa → 768D text + 768D citations → 1536D → UMAP → 50D
2. **Firm Patent Portfolios** (Week 2 — NEXT): Aggregate 50D vectors per firm → fit GMMs
3. **Compare Distributions** (Week 3): Bhattacharyya Coefficient + Directional Technology Complementarity
4. **Extensions** (Week 4): Synthetic portfolio matching — A+B≈C? and find B such that A+x≈C

Training window: 2020 patent data (crisis data surfaces latent value). Validation: 2021 M&A "Springboard Effect."

---

## Session Timeline (7 Days)

Understanding the arc helps you understand why things are the way they are:

**Day 1 (Mar 30-31)**: Empty repo → venv setup. Disk ran full. WSL-NTFS I/O made pip install take hours. PyTorch wheel extraction was agonizing. This consumed massive context and required compaction. *Lesson learned: WSL is for editing, not compute. Heavy work goes to AWS or desktop.*

**Day 2 (Apr 1)**: Post-compaction restart. EDA notebook (8 data quality questions), ADRs 001-002, spec, all 5 modules implemented with TDD (38 tests). Torrin bypassed the Codex design gate to maintain velocity — a pragmatic call he later acknowledged carried risk. 1K sample pipeline built. Torrin ran on laptop but it was too slow.

**Day 3 (Apr 2)**: Pushed to GitHub. Torrin pulled on desktop. **GPU compatibility issue** — 1080 Ti CC 6.1 not supported by current PyTorch. Fixed with CUDA probe fallback. Pipeline ran in 13.4 min on CPU. **Key discoveries**: L2 norms ~6.8 (not unit-normalized), **bimodal citation norms** (Torrin spotted this in the histogram — investigated and explained by citation count, r=-0.317). First Codex review found critical resume-corruption bug and checkpoint no-op. All fixed. Codex approved on re-review.

**Day 4-5 (Apr 3-5)**: Amie delivered v2 data (biotech expansion). Validated — found 14% null abstracts, 201K duplicate patent_ids, lower citation coverage. Cloud architecture designed (ADR-003, Terraform). Second Codex review found **critical encoding bug** (`abstract + " " + abstract`) plus 5 other issues. Three rounds of fixes. IAM permissions blocker resolved by removing IAM from Terraform. Codex approved deployment path.

**Day 6 (Apr 6)**: Amie's detailed response explaining data issues (pre-1976 nulls, co-assigned patents, biotech caveats). v3 data delivered with clean scope, dedup file, post_deal_flag. Pipeline updated for v3. Production run deployed on AWS by Codex.

**Day 7 (Apr 7)**: Production run healthy on AWS (~1.4M/1.45M patents encoded). Strategic analysis: identified 13 technical debt items. Three parallel agents dispatched to fix them all + write Week 2 bootstrap prompt. 45 total tests. Handover initiated.

**Key pain points across the session:**
- WSL-NTFS I/O wasted an entire day on venv setup
- Three data versions (v1→v2→v3) each required re-validation, config updates, doc updates
- The Codex-found encoding bug (`abstract + abstract`) would have produced methodologically wrong results at scale — validates the review process
- Context ran low by Day 7, necessitating this handover

---

## What Was Built in This Session

### The Pipeline
```
patent_metadata_dedup.parquet → PatentEncoder (title+abstract) → 768D
cited_abstracts.parquet → PatentEncoder → CitationAggregator (mean pool) → 768D
                                                    ↓
                                              Concatenate → 1536D
                                                    ↓
                                              UMAPReducer → 50D → patent_vectors_50d.parquet
```

### Module Inventory
| Module | File | Tests | Purpose |
|--------|------|-------|---------|
| CheckpointManager | `src/utils/checkpointing.py` | 7 | Binary parquet save/load with schema metadata |
| PatentLoader | `src/data_loading/patent_loader.py` | 13 | Column-selective loading, v3 two-file pattern (full + dedup) |
| PatentEncoder | `src/embeddings/patent_encoder.py` | 12* | PatentSBERTa encoding, chunked checkpointing, GPU fallback |
| CitationAggregator | `src/embeddings/citation_aggregator.py` | 8 | Mean-pooled citation embeddings, zero-citation handling |
| UMAPReducer | `src/dimensionality_reduction/umap_reducer.py` | 5 | 1536D → 50D with configurable UMAP parameters |

*12 encoder tests require cached PatentSBERTa model; skipped in offline environments (33 offline-safe + 12 model-dependent = 45 total)

### ADRs
| ADR | Decision | Status |
|-----|----------|--------|
| ADR-001 | PatentSBERTa, default 512-token truncation, title+abstract concatenation | Accepted |
| ADR-002 | Mean pooling for citations, zero vector for zero-citation patents, embed-once-then-lookup | Accepted |
| ADR-003 | g5.8xlarge on AWS, S3 namespace isolation, manual credentials | Accepted |

### Infrastructure
- Terraform at `infrastructure/main.tf` — EC2 (g5.8xlarge) + security group, no IAM
- S3 bucket `ubc-torrin` with prefix `firm-pair-merger/`
- Production run executing on AWS at time of handover

---

## The Reasoning Behind Key Decisions

### Why PatentSBERTa over general SBERT?
Patent language is a specialized register. PatentSBERTa was trained on patent title+abstract pairs — exactly our input format. The bootstrap prompt suggested an empirical comparison; we deferred it as the domain-specific choice is well-justified by literature. Codex accepted this.

### Why mean pooling for citations?
Simplest defensible option. Choi et al. (2019) used aggregated citation embeddings. Mean pooling preserves the average direction of knowledge base linkage, which is what matters for distributional comparison. The bimodal norm finding (upper peak ~6.75 for 1-3 citations, lower peak ~5.5 for 4+ citations, r=-0.317) is an expected mathematical property, not a data issue.

### Why zero vector for zero-citation patents?
7.6% of patents have no citations. The zero vector correctly encodes "no citation signal." The title+abstract embedding still carries content signal in the concatenated 1536D vector. Excluding these patents would lose data; imputing would introduce noise.

### Why UMAP over PCA?
UMAP preserves local structure better than PCA for non-linear manifolds. The methodology (Choi et al. 2019) specifies UMAP. Cosine metric is natural for embedding spaces.

### Why g5.8xlarge over cheaper instances?
UMAP on 1.5M × 1536 needs ~60-80 GB RAM. The 64 GB instances (g5.4xlarge) risk OOM. The A10G GPU (24 GB VRAM) is faster than T4 for encoding. On-demand (not spot) because UMAP can't checkpoint mid-fit — a spot interruption would lose hours of work.

### Why manual AWS credentials instead of IAM roles?
Torrin's IAM user can't create roles (university-managed AWS account). His existing `torrin` profile has S3 read/write. Manual `aws configure --profile torrin` after SSH is the path of least resistance. Codex approved this as a temporary, operator-driven approach with explicit cleanup rules.

---

## What We Discovered

### Data Evolution: v1 → v2 → v3
| Version | Patents | Cited | Edges | Key Change |
|---------|---------|-------|-------|------------|
| v1 | 1,211,889 | 1,872,555 | 16,698,056 | Tech-only, original scope |
| v2 | 2,568,440 | 3,786,338 | 47,878,021 | Added biotech + 480K private startups. 14% null abstracts, 201K duplicate patent_ids |
| v3 | 1,604,583 (1,519,401 dedup) | 2,623,183 | 35,424,315 | Clean scope, post_deal_flag, pre-deduplicated file, 0.015% null abstracts |

v3 is current. The two-file pattern (full for portfolio construction, dedup for encoding) was designed by Amie to handle co-assigned patents.

### Embedding Characteristics
- **L2 norms NOT unit-normalized**: Title+abstract mean=6.79, std=0.14. PatentSBERTa doesn't normalize.
- **Citation norms are bimodal**: Upper peak (~6.75) = 1-3 citations, lower peak (~5.5) = 4+ citations. Pearson r=-0.317 with citation count. Mathematical property of mean pooling.
- **2D UMAP shows meaningful structure**: Firm-level clustering tendencies visible. Focused firms cluster tighter than mega-firms (IBM spans the space). Canon confirmed as the second cluster.
- **Truncation is a non-issue**: 0.03% of patents exceed 512 tokens.
- **Citation coverage**: 84.9% unique-level, ~94% edge-level in v3 (lower than v1's 97% due to biotech citing international patents).

### Biotech Caveats (from Amie)
Three structural limitations for biotech firms in our methodology:
1. US-only patent pull misses international citations (biotech is globalized)
2. Non-patent literature (clinical trials, academic papers) is invisible to citation embeddings
3. Wet lab M&A is FDA-driven (Phase 2 trials), not patent-overlap-driven

These don't change pipeline code but affect interpretation of BC results for biotech firms.

---

## The Codex Review Process

Codex found real bugs across three review rounds:

**Round 1 (Week 1 implementation):**
- [Critical] Resume-from-checkpoint could silently corrupt ID→embedding alignment → Fixed with prefix verification
- [Major] checkpoint_every_n was a no-op → Fixed with chunked encoding
- [Major] Tests failed offline → Fixed with module-level skipif
- [Major] Design gate was skipped → Acknowledged, corrected retroactively

**Round 2 (Cloud deployment):**
- [Critical] Cited abstracts encoded as `abstract + " " + abstract` (title=abstract bug) → Fixed to use `encode_texts` directly
- [Major] gvkey_map included patents without vectors → Fixed to filter after dedup
- [Major] Coverage stats reloaded large objects before UMAP → Fixed to compute before freeing
- [Major] Storage mismatch (ADR said 900GB NVMe, Terraform provisioned 100GB EBS) → Corrected
- [Major] S3 ListBucket unbounded → Scoped with prefix condition
- [Major] SSH open to world → Restricted to operator CIDR

**Round 3 (Deployment re-review):**
- [Major] Cited-abstract checkpoint didn't support prefix-resume → Fixed
- [Major] ADR storage spec still wrong → Corrected to 200GB gp3 EBS

**Lesson**: The Codex review process catches real issues. The critical encoding bug (abstract + abstract) would have produced methodologically wrong results at scale. Do not skip the design gate.

---

## Current State (at handover)

### What's Running
- Production pipeline on AWS g5.8xlarge (instance i-03cc59ff697b9b42f)
- Codex is monitoring the run
- Processing ~1.45M patents through all 4 stages (encode → aggregate → concatenate → UMAP)
- Results will be pushed to `s3://ubc-torrin/firm-pair-merger/output/`

### What's on Disk
- v3 data files in `data/` (3 parquet files + 1 dedup file)
- 1K sample outputs in `output/embeddings/sample_*` (from local validation run)
- Terraform state in `infrastructure/` (active instance)

### What's Ready for Week 2
- Week 2 bootstrap prompt: `docs/epics/week2_firm_portfolios/bootstrap_prompt.md` (473 lines, comprehensive)
- Codex system prompt: `docs/epics/codex_system_prompt.md`
- 4 ADRs to draft: 004 (K selection), 005 (min patents), 006 (covariance type), 007 (normalization)
- Module directories exist: `src/portfolio/`, `src/comparison/`

### What's Blocked
- Week 2 implementation: blocked on production run results (50D vectors)
- Normalization sensitivity check: blocked on production run results
- Firm-size EDA: blocked on production run results (gvkey_map)

### What Can Be Done Now (Before Production Run Completes)
- **Draft ADRs 004-007**: These are design documents, not code. They can be written with the information we have from v3 data validation (15,814 firms, extreme size skew) and the methodology research.
- **Write the firm portfolio spec**: Interface contracts for PortfolioBuilder and GMMFitter
- **Investigate GMM serialization format**: How to store per-firm GMM parameters in parquet

---

## Technical Debt Status

All previously identified debt was resolved in this session's final commit (`1a40eb5`):

| Issue | Resolution |
|-------|-----------|
| Spec stale (v2 refs) | Updated to v3 with source param, post_deal_flag, correct row counts |
| Sample pipeline stale | Rewritten to match full pipeline behavior |
| No v3 tests | 4 new tests: source="dedup", co-assignments, post_deal_flag, 4-key row counts |
| Column validation lost | Restored: source="full" requires title+abstract |
| Duplicated checkpoint logic | Extracted `encode_texts_checkpointed()` method |
| Coverage stats duplicate groupby | Rewritten without expensive groupby...apply(list) |
| checkpoint_exists swallows all exceptions | Narrowed to ArrowInvalid/ArrowIOError only |
| Dead code (checkpoint_dir, embedding_dim) | Removed |
| No config validation | Added required top-level key check |
| print() instead of logging | patent_encoder uses logging.getLogger |
| load_config relative path fragile | Fixed to use __file__-relative path |
| Citation aggregation O(n*m) Python loop | Documented (not optimized — acceptable for one-time run) |

### Known Remaining Items
- **CitationAggregator performance**: Python loop is O(n_patents * avg_citations). Documented as ~10-30 min at scale. Not optimized per parsimony principle — vectorize only if it becomes a bottleneck.
- **No UMAPReducer.transform()**: The spec mentions storing the fitted model for held-out data, but no `transform` method is exposed. Not needed for Week 1-2, may be needed for Week 4 extensions.
- **No integration tests**: Only unit tests exist. End-to-end test through all modules is missing.
- **Config output paths unused**: `config.yaml` defines output paths (`output.title_abstract_embeddings`, etc.) but the pipeline scripts hardcode `OUTPUT_DIR = "output/embeddings"` and build paths manually. The config paths are dead configuration.
- **No tests for `load_config()`**: Edge cases (malformed YAML, missing file, extra keys) are untested.
- **Uncommitted Codex work**: `infrastructure/main.tf` and `user_data.sh` have uncommitted modifications from Codex (subnet pinning, commit pinning, data version templating). Also untracked: `scripts/watch_pipeline_and_shutdown.sh` (Codex's watchdog), validation PNGs in `notebooks/`, `.codex` file, Terraform state/lock files.
- **Stale v2 data files still on disk**: `data/` contains both v2 and v3 files (4.2 GB total). v2 can be removed to save space since v3 supersedes it and v2 is backed up in S3.

---

## Commitments to Codex

These were negotiated during the review process and must be honored:

| Commitment | Status | When |
|------------|--------|------|
| CitationAggregator 100K timing test | Pending | Before next full-scale run |
| Week 2 normalization sensitivity check | Pending | Week 2 EDA (Phase 1) |
| Design gate enforcement | Active | ADRs reviewed before implementation in Week 2 |

---

## How This Project Works (Process)

### The Sprint Pattern
1. **Bootstrap prompt** defines the week's mission, deliverables, and halting points
2. **Phase 0**: Environment/data setup
3. **Phase 1**: EDA → HALT for Torrin's approval
4. **Phase 2**: ADRs + Spec → HALT for Codex review
5. **Phase 3**: Implementation (TDD — tests before code) → HALT for Codex review
6. **Phase 4**: Validation + retrospective → HALT for Torrin's approval

### The Codex Pattern
- Codex has a system prompt at `docs/epics/codex_system_prompt.md`
- Write a handoff document explaining what was built, why, and what to scrutinize
- Codex reviews with structured findings (Critical/Major/Minor/Note)
- Fix all Critical and Major items, then request re-review
- Codex also handles deployment ops (Terraform apply, SSH, monitoring)

**The Codex relay workflow** (from FTM project, used throughout this project):
1. You produce work
2. Torrin asks: "Could you summarize your plan, your thought process, reasoning and judgement behind decisions for me to provide back to Codex?"
3. He pastes your summary to Codex
4. He pastes Codex's review back as a `<Codex>` tagged message
5. You address findings, iterate until approved

**Codex handoff template** (standardized from FTM):
```
You are joining this project cold. Do not assume context not written here.
Repository: [path]
Branch: [branch]
Operating model: [who does what]
Mission: [specific scoped task]
[Constraints]
[Verification steps]
```

### The Team Communication Pattern
- Torrin drafts team messages himself but asks for help filling in technical details
- Messages to team are warm and casual with technical substance
- Data questions route through Amie directly
- Methodology questions route through Arthur/Jan
- Codex handles all infrastructure ops

---

## What I'd Do Differently

Looking back at this session:

1. **I should have enforced the design gate.** Torrin approved skipping the Codex review of ADRs before implementation. Codex subsequently found a critical resume-corruption bug that the gate would have caught. The pragmatic velocity gain wasn't worth the risk.

2. **The sample pipeline should have been updated alongside the full pipeline.** Every time we updated `run_full_pipeline.py`, `run_sample_pipeline.py` drifted further. It should have been refactored into a shared orchestrator early.

3. **The v2 data integration was premature.** We updated config and code for v2, then immediately had to redo it for v3. Should have waited for Amie's confirmation that the data was final.

4. **The citation encoding bug (abstract + abstract) was a copy-paste error** in `run_full_pipeline.py` that passed code review by the development instance. It was caught by Codex. This validates the importance of the impartial reviewer pattern.

---

## Repository State at Handover

**15 commits on `master`, fully pushed.** The commit history tells a three-act story:

1. **Foundation** (Apr 1): Initial pipeline commit + GPU fix + validation findings
2. **Codex review cycle** (Apr 2-6): 8 commits fixing critical bugs, deploying cloud infrastructure, iterating on Terraform
3. **v3 data + cleanup** (Apr 7): Data update + technical debt sweep + Week 2 bootstrap

**Uncommitted work** (Codex modified these during deployment):
- `infrastructure/main.tf` — Codex added subnet_id, repo_commit pinning
- `infrastructure/user_data.sh` — Codex added git checkout pinned commit
- `infrastructure/terraform.tfstate` — **LIVE state with active EC2 instance**
- `scripts/watch_pipeline_and_shutdown.sh` — Watchdog script Codex created
- `notebooks/02_*.png` — Validation visualizations Torrin pasted in
- `.codex` — Empty file from Codex tooling

**Important**: The Terraform state file tracks the running EC2 instance. Do not `terraform destroy` until the production run completes and results are pushed to S3.

---

## Inherited Patterns from Financial Topic Modeling

This project inherits its development culture from Torrin's prior FTM project (12+ sprints, deployed to AWS). Key patterns:

- **Sprint structure**: Bootstrap prompt → EDA → ADRs → Spec → TDD → Validation → Retrospective
- **Codex as reviewer**: Independent quality gate that can reject work. Started in FTM, refined here.
- **Terraform patterns**: Simple, purpose-built infrastructure. Avoid over-engineering. Manual credentials acceptable for short-lived instances.
- **Handover protocol**: Each instance writes a comprehensive summary for the next. Memory files persist across sessions. Bootstrap prompts are the authoritative directives.
- **The 10 values**: Codified in `docs/values/`. These aren't aspirational — they're hard-won lessons. Violations led to rework in FTM; adherence led to successful production deployment.

---

## Known Documentation Staleness

A thorough audit revealed several documents still reference v1/v2 numbers. The authoritative data source is `config.yaml` (v3 filenames) and the memory file `project_data_schema_findings.md` (v3 row counts). Do NOT trust these documents' numbers without checking:

| Document | Stale Reference | Correct Value |
|----------|----------------|---------------|
| `CLAUDE.md` data table | ~2.7M patents, ~3.7M cited, ~46M edges | 1.52M dedup, 2.62M cited, 35.4M edges (v3) |
| `data_prep.md` | Original v1 scope | Superseded by v3 |
| ADR-001, ADR-002 | 1,211,889 patents (v1) | 1,519,401 dedup (v3) |
| ADR-003 | 2.57M patents (v2) | 1,519,401 dedup (v3) — instance still valid, just oversized |
| Week 1 retrospective | 1,892 firms (v1) | 15,814 firms (v3) |
| Runtime estimates doc | v1 numbers | Stale — v3 is ~25% smaller than v2, estimates are conservative |
| ADR-001, ADR-002 status | "Proposed" | Should be "Accepted" (Codex-reviewed and implemented) |
| Spec header | "Reviewers: Codex (pending)" | Contradicts "Accepted (Codex-approved)" in title |

**The config.yaml and the Week 2 bootstrap prompt have correct v3 numbers.** When in doubt, trust those.

**Missing from config.yaml:** The `gvkey_map` output path is defined in the spec but not in the actual config file. The pipeline script hardcodes it as `output/embeddings/gvkey_map.parquet`.

---

## Working with Torrin — What the Sub-Agents Found

Beyond the profile in the "Who You're Working With" section, deeper analysis of conversation patterns revealed:

**Voice transcription is his primary input mode.** His messages are long, conversational, sometimes stream-of-consciousness. The core directive is usually in the first and last paragraphs. Don't mistake informality for imprecision — his voice transcripts contain highly specific technical requirements embedded in natural language.

**When typed, he is terse.** "Looks solid - proceed." "Sounds good." "What are your thoughts?" Don't over-interpret brevity as dissatisfaction.

**He relays between you and Codex.** The established pattern is: you produce work → Torrin asks for a structured summary → he pastes it to Codex → he pastes Codex's review back as `<Codex>` tagged message → you address findings. His recurring prompt: "Could you summarize your plan, your thought process, reasoning and judgement behind decisions for me to provide back to Codex?"

**He drafts team messages and asks you to fill in technical details.** He'll provide a rough template in his casual team voice and ask you to enrich it with findings while preserving his tone. The output should be collegial and technically precise but not jargon-heavy — his teammates are economics students, not engineers.

**Trust is earned through process discipline.** Once you demonstrate you follow specs, write tests first, and don't cut corners, he gives significant autonomy. He approved proceeding past a design gate once — but the resulting bug (caught by Codex) validated why the process exists.

**He proactively manages context.** He compacts aggressively, writes instance summaries, and shifts your role from implementer to documenter when context runs low. He treats each instance as a specialized role and relies on handover documents as the sole communication channel between sessions.

---

## FTM Project Patterns That Apply Here

From analysis of the predecessor Financial Topic Modeling project (12+ sprints, production on AWS):

**The "Summary for Codex" pattern**: Claude produces verbose analysis, then distills a structured summary for Codex review. This is a core workflow — Codex never sees the raw conversation, only curated handoffs.

**The smoke test ladder**: Never jump from local to production. The pattern is: tiny subset → small sample → medium run → full production. We followed this (1K sample → desktop → AWS g5.8xlarge).

**Scoped Claude instances**: Each implementation task gets a fresh instance with a self-contained brief, preventing context pollution. You are one of these scoped instances.

**Feature flags over conditional resources**: In Terraform, use boolean variables with `count` for optional resources. AWS IAM in particular requires careful handling — the university account has restricted permissions.

**Hard lesson from FTM**: Plans consumed by a fresh instance must be maximally explicit. "All file paths must be absolute. All AWS CLI commands must include --profile and --region. Interpretation of diagnostic output must be structured as decision trees, not prose." Ambiguity costs an entire round-trip.

---

## Files to Read First

In priority order for a new instance:

1. This document — Full context package (read first for orientation)
2. `CLAUDE.md` — Project instructions, values, module structure
3. `docs/epics/week2_firm_portfolios/bootstrap_prompt.md` — Your primary directive for Week 2
4. `docs/sprint_retrospectives/week1_instance_summary.md` — What was built and learned
5. `src/config/config.yaml` — Current configuration (authoritative for file paths)
6. `docs/specs/patent_vectorizer_spec.md` — Week 1 interface contracts (your foundation)

---

## Immediate Next Actions

When the production run completes:

1. **Verify results**: Load patent_vectors_50d.parquet, check shape (~1.45M × 50), verify gvkey_map join integrity
2. **Run the EDA from the Week 2 bootstrap**: firm-size distribution, UMAP dimension scales, normalization sensitivity
3. **Draft ADRs 004-007**: K selection, min patents, covariance type, normalization — these can start from the recommendations in the bootstrap prompt
4. **Write firm portfolio spec**: Interface contracts for PortfolioBuilder and GMMFitter
5. **Implement with TDD**: Tests first, then code, then Codex review

Before the production run completes (if you have bandwidth):
- Draft ADRs 004-007 (design documents, not code — don't need the data)
- Review the Week 2 bootstrap prompt and assess whether it's complete
- Investigate the GMM serialization question (how to store per-firm GMM parameters)

### Items Left Unfinished at Session End

These are threads that were started or identified but not completed:

1. **Production run results not yet retrieved**: The AWS run was in progress. Results need to be downloaded from S3 and validated (shapes, counts, join integrity).
2. **EC2 instance may still be running**: Terraform state shows live infrastructure at ~$2.45/hr. Verify with Codex whether it was terminated after the run completed.
3. **Uncommitted Codex modifications**: `infrastructure/main.tf`, `user_data.sh`, and `watch_pipeline_and_shutdown.sh` have changes from Codex's deployment work that should be committed.
4. **CitationAggregator 100K timing benchmark**: Committed to Codex but never executed. Should happen before any future full-scale run.
5. **Normalization sensitivity check**: Week 2 EDA deliverable — compare raw vs L2 vs z-score before GMM fitting.
6. **Biotech methodology ADR**: Amie's caveats documented in memory but no formal ADR. Flagged for Week 3 when BC comparison is implemented.
7. **CLAUDE.md and data_prep.md still reference v1 numbers**: These high-visibility documents should be updated to v3 reality.
8. **ADR-001 and ADR-002 statuses**: Still say "Proposed" and "Reviewers: Codex (pending)" despite being implemented and Codex-approved. Should be updated to "Accepted."
