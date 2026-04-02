# Full-Scale Runtime Estimates

**Date**: 2026-04-01  
**Basis**: Extrapolated from 1K-sample pipeline run (13.4 min on CPU desktop)

## Dataset

| Input | Count |
|-------|-------|
| Patents to encode (title+abstract) | 1,211,889 |
| Unique cited abstracts to encode | 1,872,555 |
| UMAP input matrix | 1,211,889 x 1536 = **7.4 GB** (float32) |

## Measured Encoding Rate (CPU Desktop)

- Batch size: 32
- Title+abstract: 1.64 s/batch → ~19.5 patents/sec
- Cited abstracts: 1.73 s/batch → ~18.5 abstracts/sec

## Estimates by Compute Option

| Stage | CPU Desktop | 1080 Ti (if working) | AWS g4dn.4xlarge |
|-------|------------|---------------------|-----------------|
| Title+abstract encoding | **17.3 hours** | ~13 min | ~13 min |
| Citation abstract encoding | **28.1 hours** | ~21 min | ~21 min |
| Citation aggregation | ~10-30 min | ~10-30 min | ~10-30 min |
| UMAP (1536D → 50D) | **1-3 hours** | 1-3 hours | ~1-2 hours |
| **Total** | **~47 hours (2 days)** | **~3-4 hours** | **~2-3 hours** |

## Memory Requirements

| Stage | RAM Needed |
|-------|-----------|
| Encoding (batched) | ~4-6 GB |
| Citation aggregation (lookup table) | ~6 GB (1.9M x 768 x 4 bytes) |
| UMAP fit | **30-37 GB** (input matrix + internal structures) |

**UMAP is the memory bottleneck.** The 7.4 GB input matrix plus UMAP's approximate nearest neighbor structures require 30-37 GB RAM. If unavailable, use the subsampling fallback: fit on 500K patents, transform the rest.

## GPU Compatibility

The desktop's 1080 Ti (compute capability 6.1) is not supported by the current PyTorch build (requires CC >= 7.5). Options:

1. **Downgrade PyTorch**: `pip install torch==2.0.1+cu117` — last version supporting CC 6.1. Risk: may break other dependencies.
2. **CPU-only**: Works but ~80x slower for encoding.
3. **AWS**: No compatibility issues. A g4dn.4xlarge (T4 GPU, 64 GB RAM) costs ~$1.20/hr, total ~$3-4 for the run.

## Recommendation

AWS g4dn.4xlarge is the path of least resistance: $3-4, 2-3 hours, no driver issues, sufficient RAM for UMAP. The CPU desktop is viable as a 2-day overnight run if it has 32+ GB RAM.

## Checkpoint Recovery

With `checkpoint_every_n=100000` (default), encoding saves every ~85 min on CPU. A crash at hour 40 loses at most ~85 min of work. The pipeline resumes from the last checkpoint automatically.
