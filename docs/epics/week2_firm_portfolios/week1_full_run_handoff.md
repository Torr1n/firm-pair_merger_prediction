# Full-Scale Run Handoff: Week 1 -> Week 2

**From**: Codex (review + ops)  
**To**: Claude Code (development instance)  
**Date**: 2026-04-08  
**Context**: Full-scale patent vectorization run completed successfully on AWS. Week 2 can now start from production outputs rather than the 1K sample.

---

## Outcome

The full Week 1 pipeline completed successfully on the v3 tech+biotech dataset.

- **Run ID**: `20260408T005013Z`
- **Repo commit used for the run**: `c590f029c143c99a5eba300b341cef8a3197e84c`
- **Status**: `success`
- **Runtime**: `25491.9s` = `424.9 min` = `7.1 hr`
- **AWS instance**: `g5.8xlarge`
- **Instance ID**: `i-03cc59ff697b9b42f`
- **Shutdown**: clean, instance-initiated shutdown after artifact sync

The critical result is now available:

- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/patent_vectors_50d.parquet`

---

## Final Metrics

### Data processed

- **Input deduplicated patents file**: `firm_patents_dedup_techbio_v3.parquet`
- **Final patents encoded**: `1,447,673`
- **Final 50D vectors**: shape `(1447673, 50)`

### Citation stage

- **Cited abstracts encoded**: `2,594,154`
- **Zero-citation patents**: `86,688` (`6.0%`)
- **Citation edge coverage**: `32,652,966 / 34,446,559` (`94.8%`)

### Notes

- This run used the **v3** data contract:
  - upstream dedup file for embedding generation
  - full metadata file for firm mapping / co-assignments
  - `post_deal_flag == 0` filter for clean pre-acquisition features
- The overnight run completed without needing the retry path, but the watcher did sync intermediates and shut the instance down correctly.

---

## Artifact Locations

All run outputs are stored under:

- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/`

### Week 2 inputs

- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/patent_vectors_50d.parquet`
- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/gvkey_map.parquet`

### Intermediate checkpoints

- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/title_abstract_embeddings.parquet`
- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/cited_abstract_embeddings.parquet`
- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/citation_embeddings.parquet`
- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/concatenated_1536d.parquet`

### Logs / status

- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/logs/pipeline.log`
- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/status/run_status.json`

---

## How To Access The Results

### Minimal Week 2 pull

```bash
mkdir -p output/week2_inputs

aws s3 cp \
  s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/patent_vectors_50d.parquet \
  output/week2_inputs/patent_vectors_50d.parquet \
  --profile torrin --region us-west-2

aws s3 cp \
  s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/output/embeddings/gvkey_map.parquet \
  output/week2_inputs/gvkey_map.parquet \
  --profile torrin --region us-west-2
```

### Full run artifact pull

```bash
aws s3 sync \
  s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/ \
  output/full_run_20260408T005013Z/ \
  --profile torrin --region us-west-2
```

---

## What Week 2 Should Use

Use these two files as the primary interface into Week 2:

1. `patent_vectors_50d.parquet`
Each row is one encoded patent after:
- deduplication for embedding generation
- `post_deal_flag == 0` filtering
- dropping patents with no title and no abstract

2. `gvkey_map.parquet`
This is the firm ownership map for the encoded patents and is the correct bridge into firm-level portfolio construction.

Important semantics:

- `patent_vectors_50d.parquet` is **one row per encoded patent**
- `gvkey_map.parquet` can contain **multiple rows per patent_id**
- that duplication is intentional and reflects legitimate co-assignment / multi-firm credit in v3
- therefore:
  - deduplicate for vector generation
  - preserve duplicate firm mappings for portfolio construction

This matches Amie's intended two-phase design.

---

## Remaining Source Data

The original v3 parquet files remain relevant:

- `data/firm_patents_text_metadata_techbio_v3.parquet`
- `data/firm_patents_dedup_techbio_v3.parquet`
- `data/cited_abstracts_techbio_v3.parquet`
- `data/citation_network_techbio_v3.parquet`

Use the full metadata file if Week 2 needs:

- `universal_id`
- `link_source`
- `link_method`
- `post_deal_flag`

But for core Week 2 portfolio fitting, the production outputs above should be the default inputs.

---

## File Format Reminder

The embedding checkpoints written by `CheckpointManager` use parquet with:

- column `patent_id`
- column `embedding` as raw `float32` bytes
- schema metadata including `embedding_dim` and `row_count`

So downstream code should continue using the existing checkpoint loader rather than assuming `50` separate numeric columns.

---

## Recommended Immediate Next Steps

1. Promote or copy `patent_vectors_50d.parquet` and `gvkey_map.parquet` into the canonical Week 2 working location.
2. Validate row counts after load:
   - vectors: `1,447,673`
   - unique patent IDs in vectors: `1,447,673`
3. Confirm `gvkey_map.parquet` joins cleanly to the vector file on `patent_id`.
4. Begin Week 2 design/implementation around:
   - firm-level portfolio assembly
   - minimum-patent threshold policy
   - GMM fitting strategy for small vs large firms

---

## Ops Notes

- The EC2 instance is **stopped**, not running.
- The Terraform apply previously hit an `ec2:DescribeTags` permission error after create, so infra state should be cleaned up deliberately rather than assumed tidy.
- The run itself is complete and the outputs are safe in S3. Week 2 work does **not** depend on reviving the instance.

---

## Bottom Line

Week 1 is complete at full scale. Claude should start Week 2 from:

- `patent_vectors_50d.parquet`
- `gvkey_map.parquet`

under run prefix:

- `s3://ubc-torrin/firm-pair-merger/runs/20260408T005013Z/`
