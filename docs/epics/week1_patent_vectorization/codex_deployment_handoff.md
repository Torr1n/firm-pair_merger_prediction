# Deployment Handoff: Development Instance → Codex

**From**: Claude Code (development instance)  
**To**: Codex (reviewer + deployment lead)  
**Date**: 2026-04-05  
**Context**: Transitioning from Week 1 implementation to full-scale cloud deployment

---

## Situation Update

Since your last review (approved at commit `2626c40`), the project has evolved:

1. **Amie delivered v2 data files** — expanded scope from tech-only to tech + biotech, plus ~480K recovered private startup patents. The dataset roughly doubled:

   | Metric | v1 | v2 | Factor |
   |--------|----|----|--------|
   | Patents | 1,211,889 | 2,568,440 | 2.1x |
   | Cited abstracts | 1,872,555 | 3,786,338 | 2.0x |
   | Citation edges | 16,698,056 | 47,878,021 | 2.9x |
   | Unique firms | 1,892 | 34,500 | 18.2x |

2. **Cloud deployment is now the primary goal.** The full pipeline needs to run on AWS because:
   - CPU encoding would take ~4 days
   - UMAP needs ~60-80 GB RAM (exceeds desktop capacity)
   - GPU encoding reduces total runtime to ~3-5 hours

3. **Three v2 data quality issues were identified** (see below). These affect pipeline behavior but do not block deployment.

---

## What's New in the Repo (commit `7eff5fd`)

| Artifact | Path | Purpose |
|----------|------|---------|
| ADR-003 | `docs/adr/adr_003_cloud_architecture.md` | Instance selection (g5.8xlarge), S3 namespace, cost estimates |
| Terraform | `infrastructure/main.tf` | EC2, security group, IAM role, instance profile |
| Bootstrap script | `infrastructure/user_data.sh` | User-data: clone repo, install deps, pull data from S3 |
| Full pipeline script | `scripts/run_full_pipeline.py` | Handles dedup, nulls, memory management, progress logging |
| Config update | `src/config/config.yaml` | v2 filenames |
| Loader update | `src/data_loading/patent_loader.py` | Warns on duplicate patent_ids instead of rejecting |

---

## V2 Data Quality Findings

### 1. 14% null titles / 13% null abstracts (was 0.02% in v1)

Likely from the recovered private startup patents. The full pipeline script drops patents with *both* null title and abstract, and embeds title-only or abstract-only for partial nulls. Amie has been asked whether the nulls are recoverable.

### 2. 201,682 duplicate patent_ids (0 in v1)

Same patent linked to multiple gvkeys — likely firm identity changes over time (Compustat gvkey reassignment) or subsidiary linkage. v2 includes provenance columns (`universal_id`, `link_source`, `link_method`) that may disambiguate. The full pipeline deduplicates by `patent_id` (keeps first) but preserves the full `patent_id → gvkey` mapping for downstream portfolio construction.

Pending team input on whether `universal_id` can serve as a persistent firm identifier for more intelligent dedup.

### 3. Citation coverage: 93.8% edge-level (was 97.1%)

Expected — biotech patents cite prior art outside the dataset's scope. Mean pooling handles this gracefully. Accepted as-is.

---

## What We Need Codex To Review

### 1. Terraform Configuration (`infrastructure/main.tf`)

Review for:
- **IAM scope**: The policy grants `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on `ubc-torrin/firm-pair-merger/*` only. Verify this is correctly scoped and doesn't grant access to Torrin's existing `financial-topic-modeling/` namespace.
- **Security group**: SSH from `0.0.0.0/0` — is this acceptable for a short-lived instance, or should we restrict to Torrin's IP?
- **AMI selection**: We're using `Deep Learning OSS Nvidia Driver AMI GPU PyTorch *Ubuntu 22.04*` via data source. Verify the filter is specific enough to get a stable, working AMI.
- **Instance type**: g5.8xlarge (A10G, 128GB RAM, $2.45/hr). ADR-003 documents the sizing rationale.

### 2. Bootstrap Script (`infrastructure/user_data.sh`)

Review for:
- **S3 data pull**: Does the `aws s3 cp --recursive` correctly pull from the right prefix?
- **Git clone**: Uses HTTPS (public repo). If the repo is private, this needs an auth token.
- **Venv setup**: Does the dependency install order work on the Deep Learning AMI?

### 3. Full Pipeline Script (`scripts/run_full_pipeline.py`)

Review for:
- **Memory management**: The script explicitly `del`s large arrays after use. Verify the lifecycle is correct — especially that the citation lookup table (~12 GB) is freed before UMAP starts.
- **Dedup logic**: `drop_duplicates(subset="patent_id", keep="first")` — is "first" the right choice, or should it be more deterministic?
- **Checkpoint consistency**: The script uses the same `encode_patents` method that Codex previously reviewed and approved. Verify the v2 data flow doesn't introduce new misalignment risks.
- **Coverage stats computation**: There's a potentially expensive re-load of citation data for stats computation. Flag if this is a concern.

### 4. AWS Permissions Verification

Torrin has IAM access to Jan Bena's AWS account via cloud administrator David. Existing permissions include AWS Batch, Step Functions, ECS, S3 (for the Financial Topic Modeling project). We need to verify:
- Can Torrin's IAM user/role create EC2 instances?
- Can Torrin's IAM user/role create IAM roles and instance profiles? (Terraform needs this)
- Can Torrin's IAM user/role create security groups?
- Is there a service quota for g5 instances in the account's region?
- Is the `ubc-torrin` S3 bucket accessible and does the prefix `firm-pair-merger/` not conflict with anything?

If Torrin lacks these permissions, we need to request them from David. Draft the minimal permission set needed.

---

## Deployment Plan

1. **Pre-flight** (Codex reviews Terraform + scripts, verifies permissions)
2. **Upload data to S3**: `aws s3 sync data/ s3://ubc-torrin/firm-pair-merger/data/v2/`
3. **Deploy**: `cd infrastructure && terraform init && terraform apply -var="key_name=torrin-key"`
4. **SSH in, run pipeline**: monitor via `tee output/pipeline.log`
5. **Push results to S3**: `aws s3 sync output/ s3://ubc-torrin/firm-pair-merger/output/`
6. **Terminate instance**: `terraform destroy`

### Codex's Role During the Run

- **Monitor**: Check CloudWatch or SSH logs for errors, OOM, or stalls
- **Post-run validation**: Compare output shapes and coverage stats against the 1K-sample baseline
- **Cost tracking**: Record actual runtime and cost for the retrospective
- **Results analysis**: Pull examples, generate summary statistics, run validation visualizations via Lambda or local analysis

---

## Commitments Carried Forward from Week 1 Review

| Commitment | Status | When |
|------------|--------|------|
| CitationAggregator 100K timing test | Pending | Before full-scale run |
| Week 2 normalization sensitivity check | Pending | Week 2 EDA |
| Design gate enforcement | Active | ADR-003 submitted for review before deployment |

---

## Questions for Codex

1. Should the security group restrict SSH to a specific IP range, or is `0.0.0.0/0` acceptable for a short-lived (<6 hr) instance?
2. The `run_full_pipeline.py` script reloads citation data for coverage stats after freeing the lookup table. Is this worth optimizing, or is the I/O cost negligible relative to the multi-hour encoding stages?
3. For the g5.8xlarge in us-west-2: are there known availability issues for this instance type? Should we set a fallback to g4dn.8xlarge in the Terraform?
4. Do we need a CloudWatch alarm to auto-terminate the instance after idle timeout, or is manual `terraform destroy` sufficient?
