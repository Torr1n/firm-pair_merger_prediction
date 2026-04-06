# ADR-003: Cloud Architecture for Full-Scale Pipeline Execution

**Status**: Proposed  
**Date**: 2026-04-05  
**Authors**: Torrin Pataki, Claude Code  
**Reviewers**: Codex (pending)

## Context

The patent vectorization pipeline needs to process 2.57M patents through PatentSBERTa encoding, citation aggregation, and UMAP reduction. Local execution on CPU would take ~4 days, which is impractical. We need a cloud deployment that can complete the run in a few hours for a reasonable cost.

Torrin has access to Jan Bena's AWS account through cloud administrator David. The account already runs an earnings call processing pipeline (AWS Batch, Step Functions, ECS, S3). We need to deploy alongside that work without interference.

### Workload Characteristics

| Stage                                         | Compute                       | Memory                | Duration (est.) |
| --------------------------------------------- | ----------------------------- | --------------------- | --------------- |
| PatentSBERTa encoding (2.57M texts)           | GPU (batch inference)         | ~4-6 GB               | ~20-30 min      |
| PatentSBERTa encoding (3.79M cited abstracts) | GPU (batch inference)         | ~4-6 GB               | ~25-40 min      |
| Citation aggregation                          | CPU (dict lookup + mean pool) | ~12 GB (lookup table) | ~10-20 min      |
| UMAP (2.57M × 1536 → 50D)                     | CPU (memory-bound)            | **~60-80 GB**         | ~1.5-3 hr       |
| **Total**                                     |                               | **Peak ~80 GB**       | **~2.5-4.5 hr** |

## Decision

### Instance: g5.8xlarge (on-demand)

| Spec            | Value                       |
| --------------- | --------------------------- |
| GPU             | 1x NVIDIA A10G (24 GB VRAM) |
| vCPUs           | 32                          |
| RAM             | 128 GB                      |
| Storage         | 200 GB gp3 EBS (root volume) |
| On-demand price | ~$2.45/hr (us-west-2)       |

**Why g5.8xlarge over alternatives:**

- **g4dn.4xlarge / g5.4xlarge (64 GB RAM)**: Too tight. UMAP on 2.57M × 1536 needs ~60-80 GB working memory. With OS overhead, citation lookup table (~12 GB), and Python interpreter, 64 GB risks OOM.
- **g4dn.8xlarge (128 GB, T4)**: Same RAM but T4 is slower than A10G for inference. Saves ~$0.27/hr but encoding takes ~1.5x longer. Net saving is negligible (~$1 total).
- **g4dn.12xlarge / g5.12xlarge (192 GB, 4 GPUs)**: Overkill. We only use 1 GPU. Paying for 4 GPUs wastes 75% of GPU spend.

**Why on-demand (not spot):**

- Pipeline runs once, takes 3-5 hours. Spot interruption would lose progress (even with checkpointing, UMAP can't checkpoint mid-fit).
- Cost difference is ~$3-5 total. Not worth the operational risk for a one-time run.

### Storage: S3 bucket `ubc-torrin` with namespace isolation

```
s3://ubc-torrin/
├── financial-topic-modeling/    # Existing earnings call work (DO NOT TOUCH)
└── firm-pair-merger/            # This project
    ├── data/v2/                 # Input parquet files
    ├── output/embeddings/       # Pipeline checkpoint outputs
    └── output/validation/       # Validation artifacts
```

**Why reuse existing bucket:**

- Torrin already has IAM access to `ubc-torrin`
- Prefix-based isolation is standard S3 practice
- No cross-project interference — different prefixes, no shared state

### AMI: AWS Deep Learning AMI (Ubuntu 22.04, PyTorch)

Pre-installed NVIDIA drivers, CUDA, and PyTorch. We install project-specific dependencies (sentence-transformers, umap-learn) on top.

### Deployment pattern: Single-instance batch job

1. Launch g5.8xlarge with bootstrap script (user-data)
2. Bootstrap installs project dependencies, pulls data from S3
3. Runs `scripts/run_full_pipeline.py`
4. Pushes results back to S3
5. Instance can be terminated manually or via CloudWatch alarm on idle

No orchestration (Step Functions, Batch) needed — this is a single sequential pipeline on one machine.

## Alternatives Considered

| Alternative                                             | Why Rejected                                                                                                                     |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| AWS Batch                                               | Adds orchestration complexity for a single-machine job. Good for the earnings call pipeline (many parallel jobs), overkill here. |
| SageMaker Processing                                    | Higher cost, vendor lock-in on job format, unnecessary abstractions.                                                             |
| Spot instance                                           | Risk of interruption during UMAP (non-checkpointable). Net savings ~$3-5.                                                        |
| Multi-instance (encode on GPU, UMAP on high-memory CPU) | Added data transfer complexity. Single g5.8xlarge handles both.                                                                  |
| Lambda / serverless                                     | UMAP needs 80+ GB RAM and hours of runtime. Not serverless-shaped.                                                               |

## Cost Estimate

| Scenario    | Hours  | Instance Cost | S3 Storage/mo | Total    |
| ----------- | ------ | ------------- | ------------- | -------- |
| Optimistic  | 2.5 hr | $6.12         | ~$0.50        | **~$7**  |
| Expected    | 3.5 hr | $8.57         | ~$0.50        | **~$9**  |
| Pessimistic | 6 hr   | $14.69        | ~$0.50        | **~$15** |

## Consequences

- Full pipeline run completes in 3-5 hours for ~$10
- All checkpoint parquet files stored in S3 for downstream Week 2 consumption
- Instance is ephemeral — no ongoing cost after termination
- Data is isolated from earnings call work via S3 prefix
- Codex can monitor the run via CloudWatch logs and verify results post-completion

## Validation

- Codex reviews Terraform and deployment script before launch
- Codex verifies IAM permissions are sufficient before deployment
- Pipeline outputs are validated against 1K-sample results (same code, larger dataset)
- Runtime and cost are logged for retrospective
