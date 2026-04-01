# Incremental Validation

## The Principle

Do not write 500 lines of code before testing. The pattern is:

1. Implement one method
2. Run tests
3. Fix issues
4. Move to next method

Smoke tests before full runs. Small samples before the full dataset. 10-firm validation before 3000-firm production. Every step should produce evidence that it works before the next step begins.

## Why It Matters

**Bugs compound.** If you write 500 lines and then test, you have 500 lines of potential bug surface. If the test fails, the bug could be anywhere. If you write 20 lines and test, the bug is in those 20 lines. The cost of finding and fixing a bug grows super-linearly with the amount of untested code.

**Assumptions are fragile.** Every step of a pipeline makes assumptions about its inputs. Step 2 assumes Step 1 produced valid output. Step 3 assumes Step 2 produced valid output. If you don't validate at each step, invalid assumptions propagate and corrupt downstream results silently. You might not discover the problem until the final output looks wrong — and then you have to debug the entire pipeline.

**From the Financial Topic Modeling project:** Sprint 10 ran a 10-firm smoke test before attempting a full quarter run. This smoke test discovered two separate issues (Batch AZ filter misconfiguration and Step Functions revision pinning) that would have caused the full run to fail. Each run produced formal reports with timing breakdowns, reliability signals, and cost metrics — evidence that the system worked, not just a hope.

The full-quarter ramp epic required each pilot run to be logged with a standardized tuning matrix. One variable was changed at a time to maintain interpretability. Sprint 12 deferred the Qwen3.5 model upgrade until after Pilot 1 completed, keeping all other variables constant. The principle: never change multiple things at once, because you won't know which change caused the result.

## How to Apply It

1. **Start with a sample.** Before running PatentSBERTa on 2.7M patents, run it on 100 patents. Inspect the output. Does it look right? Are the embeddings reasonable? Are there nulls?
2. **Validate each stage.** After title+abstract embeddings: check shapes, norms, null rates. After citation aggregation: check that zero-citation patents got zero vectors. After UMAP: check that the output dimensionality is correct and visualize a 2D projection.
3. **Produce evidence, not assertions.** "It works" is not evidence. A timing breakdown, a coverage table, a distribution histogram — these are evidence.
4. **Halting points.** After each phase, HALT and present findings. Do not proceed to the next phase until the current phase is validated and approved.

## When to Invoke This Value

- When starting a new pipeline stage: "What does valid output look like? How will I verify it?"
- When tempted to "just run the whole thing": "Have I validated each step independently?"
- When presenting results: "Am I showing evidence or making assertions?"
- When debugging: "What was the last validated checkpoint? Start debugging from there."
