# Checkpoint Everything

## The Principle

Save intermediate results at every pipeline stage. Fault tolerance comes from resumability, not from retries. If a pipeline stage fails, you should be able to resume from the last checkpoint — not re-run everything from scratch.

Checkpoints serve three purposes:
1. **Fault tolerance**: If the process crashes at Stage 3, restart from the Stage 2 checkpoint.
2. **Debugging**: If Stage 4's output looks wrong, load the Stage 3 checkpoint and inspect the inputs.
3. **Iteration speed**: If you're tuning Stage 4's parameters, you shouldn't need to re-run Stages 1-3 every time.

## Why It Matters

**Computation is expensive.** Embedding 2.7M patents with PatentSBERTa takes hours. UMAP on 2.7M vectors takes significant time and memory. If the process crashes after 90% of embedding is complete and there are no checkpoints, you lose all that work.

**Debugging requires access to intermediates.** When the final output is wrong, the question is: which stage introduced the error? Without checkpoints, you can only debug by re-running the entire pipeline with print statements. With checkpoints, you can load any stage's output and inspect it independently.

**From the Financial Topic Modeling project:** The FTM pipeline wrote per-firm progress to Parquet every ~50 firms. This meant that a Spot instance interruption (which happened during the Q1 full-quarter run) only lost the work on the current firm batch, not the entire run. The pipeline could resume from the last checkpoint without re-processing firms that had already completed.

The topic persistence feature cached firm-level results with a `topic_persistence.enabled` config flag. This allowed the team to iterate on the cross-firm theme clustering (reduce phase) without re-running the per-firm topic modeling (map phase) — a process that took hours per run.

## How to Apply It

1. **Checkpoint after each pipeline stage.** For the patent vectorization pipeline:
   - After title+abstract embedding → save `title_abstract_embeddings.parquet`
   - After citation embedding → save `citation_embeddings.parquet`
   - After concatenation → save `concatenated_1536d.parquet`
   - After UMAP → save `patent_vectors_50d.parquet`

2. **Checkpoint within long-running stages.** If embedding 2.7M patents takes hours, save progress every 100K patents. Include a resume mechanism that checks existing checkpoints before starting.

3. **Use standard formats.** Parquet for tabular data with metadata. NumPy `.npy` for large arrays that don't need column names. Avoid pickle (security concerns, version fragility).

4. **Include metadata.** Each checkpoint should record: timestamp, row count, configuration used, stage name. This makes it possible to verify that a checkpoint is consistent with the current configuration.

## When to Invoke This Value

- When designing any pipeline stage: "What is the checkpoint format for this stage's output?"
- When a long-running process starts: "If this crashes at 90%, will I lose all the work?"
- When debugging: "Can I load the intermediate checkpoint and inspect it?"
- When tuning hyperparameters: "Can I re-run just this stage from the previous stage's checkpoint?"
