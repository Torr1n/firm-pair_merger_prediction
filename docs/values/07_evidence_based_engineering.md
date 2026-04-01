# Evidence-Based Engineering

## The Principle

Claims are backed by artifacts, not assumptions. Decisions are grounded in empirical data — timing tables, distribution histograms, coverage statistics, cost breakdowns. No "I think" without evidence to support it.

When you say "the embeddings look reasonable," you should be pointing at a histogram of L2 norms. When you say "UMAP preserved structure," you should be pointing at a 2D visualization. When you say "the pipeline is fast enough," you should be pointing at a timing breakdown table.

## Why It Matters

**Assumptions kill silently.** "I think the data is clean" leads to a pipeline that produces garbage from null values nobody checked. "I think 512 tokens is enough" leads to truncated abstracts that lose critical information. "I think mean pooling is fine" leads to citation embeddings that don't capture the intended signal. Every assumption is a bet — and bets should have evidence behind them.

**Evidence enables evaluation.** When you present results to your advisor (Jan), to your team, or to the Codex reviewer, they need evidence to evaluate your work. A claim without evidence is an opinion. A claim with evidence is a finding.

**Evidence survives context resets.** When the next AI instance picks up this work, it can read the EDA notebook, look at the distribution plots, and see the timing tables. It doesn't need to re-derive everything from first principles. The evidence is the institutional memory of what was measured and what was true at that point in time.

**From the Financial Topic Modeling project:** The Spot interruption fix was validated with actual CloudWatch logs showing `Host EC2*` termination events — not a guess that Spot was the problem. The thinking-token contamination issue was verified by reading exact line numbers in parquet files (`firm_topics_strings_excerpt.txt` lines 245-250), confirming that `<think>...</think>` blocks were present in the output.

Every smoke test and production run produced formal reports with: execution metadata, configuration, timing breakdowns, observations, artifacts, and categorized verdicts (Pass/Fail with category). These reports made it possible to compare runs against prior baselines and identify exactly what changed.

## How to Apply It

1. **EDA before implementation.** Before writing any pipeline code, run an EDA notebook that documents: schema verification, null rates, text length distributions, citation count distributions, firm distribution, coverage statistics.
2. **Visualize at each stage.** After embedding: L2 norm histograms. After UMAP: 2D projections. After GMM: cluster weight distributions. These are not optional — they are the evidence that the pipeline is working.
3. **Timing breakdowns.** Record how long each stage takes. This identifies bottlenecks and informs batching strategies.
4. **Coverage statistics.** After each stage: how many patents have valid outputs? What fraction were excluded and why?
5. **No anecdotal evidence.** "I looked at a few examples and they seem fine" is not evidence. "98.7% of patents have valid embeddings; the remaining 1.3% have null abstracts and were excluded" is evidence.

## When to Invoke This Value

- When making a claim about data quality: "Where is the evidence?"
- When presenting results: "Can I point to a specific artifact?"
- When debugging: "What does the data actually show vs. what do I expect?"
- When comparing approaches: "What are the measured tradeoffs, not the theoretical ones?"
