# Institutional Memory

## The Principle

Lessons must survive across sessions, sprints, and team changes. ADRs, sprint retrospectives, handover documents, and memory files preserve hard-won knowledge so that it doesn't need to be rediscovered.

Every sprint summary is written for a fresh reader. The "Assume Total Amnesia" principle means that every handoff document includes: the full context, the decisions that were made, the reasoning behind them, and the immediate next step. No assumptions about what the reader already knows.

## Why It Matters

**Knowledge is expensive to acquire.** Debugging a Spot interruption issue, discovering that a model emits thinking tokens despite being told not to, learning that GPU resources in ECS use STRINGSET not INTEGER — these lessons took hours or days to learn. If they're not documented, the next person will spend hours or days learning them again.

**AI instances start fresh.** Every Claude Code session begins with no memory of previous sessions. The only continuity comes from written artifacts. If a lesson isn't written down, it doesn't exist for the next instance.

**Retrospectives compound.** A single retrospective is useful. Ten retrospectives spanning three months of development are a knowledge base that captures the project's entire learning curve — what worked, what didn't, what was surprising, and what to do differently next time.

**From the Financial Topic Modeling project:** The project maintained multiple layers of institutional memory:
- **ADRs** (docs/adr/): Captured architectural decisions with full reasoning
- **Sprint retrospectives** (docs/sprint_retrospectives/): Captured what was built, what was learned, confidence levels, and next steps — written as "Instance Summaries" for a fresh reader
- **Handover documents** (docs/handovers/): Captured specific bug fixes, recommendations, and knowledge transfers
- **Memory files** (MEMORY.md): Captured operational patterns, account restrictions, and feedback

Key lessons preserved in memory:
- "Thinking-Token Contamination": Qwen3-8B emits `<think>...</think>` blocks despite `enable_thinking: false` — fixed with defensive regex strip at the single LLM exit point
- "Reduce Config Bug": reduce entrypoint was using firm-level BERTopic defaults (6 min_cluster_size) instead of theme-level (20 min_cluster_size)
- "ECS GPU STRINGSET": GPU resources use STRINGSET not INTEGER; don't query integerValue

Without these memory records, each of these bugs would have been rediscovered — likely at the worst possible time.

## How to Apply It

1. **Write retrospectives after every sprint.** Include: Executive Summary, "The WHY", What Was Built, Key Decisions, Confidence Levels, What Was NOT Done, Critical Context for Next Instance, and Next Phase.
2. **Write handovers at every context boundary.** When ending a session, passing work to another instance, or reaching a stopping point. Use the Lego Brick pattern: Status, Pipeline_Position, Decisions, Wins, Artifacts, Issues, Next_Steps.
3. **Update memory when you learn something surprising.** If a bug takes more than 30 minutes to diagnose, it's worth documenting. If a deployment procedure has a non-obvious step, it's worth documenting.
4. **Read memory before starting work.** At the beginning of each session, read the retrospectives, handovers, and memory files to absorb the project's accumulated knowledge.

## When to Invoke This Value

- At the end of every sprint: "Have I written a retrospective?"
- When handing off work: "Have I written a self-contained handover?"
- When a bug is fixed: "Should this be documented so the next person doesn't hit the same issue?"
- At the start of every session: "Have I read the existing retrospectives and memory?"
