# Over-Document the "Why"

## The Principle

Every non-obvious decision needs a written rationale. Code comments explain *why*, not *what*. ADRs capture the reasoning behind architectural choices so that future readers understand the constraints and tradeoffs that were active at the time of the decision — even if those constraints have since changed.

The "what" is in the code. The "how" is in the tests. The "why" is in the documentation. If the "why" is missing, the next person to encounter the code will either (a) be afraid to change it because they don't understand the reasoning, or (b) change it incorrectly because they don't know what constraint it was satisfying.

## Why It Matters

**Decisions have shelf lives.** A choice that was optimal six months ago may be suboptimal today. But you can only evaluate whether to change a decision if you know *why* it was made. Without the "why," changing the decision is a gamble — you might fix the code, or you might reintroduce the exact problem the original decision was solving.

**Teams change.** People join and leave projects. AI instances start fresh every session. The "why" is the only thing that survives context resets. It is the difference between "we do it this way because we always have" and "we do it this way because of constraint X, and if X changes, we should reconsider."

**From the Financial Topic Modeling project:** Every ADR (004-007) has a full "Context" section that explains the problem being solved, the constraints that shaped the decision, and the alternatives that were considered. ADR-006 (LLM Strategy) doesn't just say "use vLLM with Qwen3-8B" — it explains that the Grok API rate limit (500 req/min) would cause 429 errors at 5x parallelism, that self-hosting removes the rate limit entirely, that 8B fits on a single g5.xlarge GPU, and that the quality tradeoff is acceptable because scalability is the priority.

When the Q1 full-quarter run needed to switch from Spot to on-demand instances, the recommendation document (`2026-03-23_q1_map_rerun_recommendation.md`) explained *why*: "Retries are good for occasional transient noise. They are weak primary control when map jobs are long enough that Spot interruption becomes a normal operating hazard." This reasoning is what allows a future reader to know whether the decision still applies — if job duration shrinks, Spot may become viable again.

## How to Apply It

1. **ADRs for architectural decisions.** Any decision that constrains future work — technology choices, data schemas, processing strategies — gets an ADR with Status, Date, Context, Decision, Consequences, Alternatives Considered.
2. **Comments for non-obvious code.** Don't comment `x += 1  # increment x`. Do comment `# Zero vector for patents with no citations — represents "no knowledge base linkage" signal`.
3. **Commit messages for the "why."** The diff shows what changed. The commit message explains why it was changed.
4. **Sprint retrospectives for the "what we learned."** At the end of each sprint, document what worked, what didn't, and what the next person should know.

## When to Invoke This Value

- When writing an ADR: ensure the Context section explains the problem, not just the solution
- When adding a code comment: ask "would removing this comment leave the next reader confused about *why*?"
- When someone proposes changing an existing approach: first check whether an ADR explains the original reasoning
- When handing off work: ensure the handover document explains *why* the current state is the way it is
