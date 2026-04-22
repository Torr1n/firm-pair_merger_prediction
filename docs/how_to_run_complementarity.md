# How to Run: Complementarity Script

**Script**: `scripts/compute_complementarity.py`
**Output**: `output/comparison/complementarity_matrix_global_k50.npz`
**Runtime**: ~15 minutes

---

## Required Data Files

You need two folders on your Desktop (contact Amie Le Hoang for access):

**`~/Desktop/output/week2_inputs/`**
| File | Description |
| --- | --- |
| `patent_vectors_50d.parquet` | 1.4M patents × 50D embeddings |
| `gvkey_map.parquet` | Links each patent to a firm (gvkey) |

**`~/Desktop/output/kmax_sweep/`**
| File | Description |
| --- | --- |
| `deduplication_decisions.csv` | List of 464 firms to exclude |

---

## Step 1 — Clone the repo (if you haven't already)

```bash
git clone https://github.com/Torr1n/firm-pair_merger_prediction.git
cd firm-pair_merger_prediction
git checkout feature/directional-complementarity
```

---

## Step 2 — Create the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas pyarrow scikit-learn
```

You only need to do this once. Next time just run `source venv/bin/activate`.

---

## Step 3 — Run the script

```bash
source venv/bin/activate
python scripts/compute_complementarity.py
```

Expected output:
```
Loading deduplication list...
  Excluding 464 firms
Loading patent embeddings...
  1,447,673 patents, shape (1447673, 50)
Loading gvkey map...
  1,531,922 patent-firm assignments
Fitting global K-means (K=50)...
  Fitting MiniBatchKMeans K=50 on 1,447,673 patents...
  Done. Cluster sizes: min=6,893  max=59,595
Building per-firm weight vectors...
  7485 firms, P shape: (7485, 50)
Computing BC, dissimilarity, and directional complementarity matrices...
Validating...
...
Saved: output/comparison/complementarity_matrix_global_k50.npz
```

---

## Step 4 — Load and use the output

```python
import numpy as np

data = np.load("output/comparison/complementarity_matrix_global_k50.npz", allow_pickle=True)

gvkeys     = data["gvkeys"].tolist()        # list of 7485 firm identifiers
bc         = data["bc_matrix"]              # (7485, 7485) symmetric similarity
diss       = data["dissimilarity_matrix"]   # (7485, 7485) symmetric, = 1 - BC
comp_dir   = data["comp_dir_matrix"]        # (7485, 7485) asymmetric directional score

gvkey_to_idx = {gv: i for i, gv in enumerate(gvkeys)}

# Example: IBM (006066) acquiring Intel (012141)
i = gvkey_to_idx["006066"]  # IBM
j = gvkey_to_idx["012141"]  # Intel

print(bc[i, j])          # how similar are IBM and Intel?
print(comp_dir[i, j])    # does Intel fill IBM's market-relative gaps?
print(comp_dir[j, i])    # does IBM fill Intel's market-relative gaps?
```

---

## What the three features mean

| Feature | Symmetric? | Interpretation |
| --- | --- | --- |
| `bc_matrix[i,j]` | Yes | How much do firms i and j overlap in the same technology zones? |
| `dissimilarity_matrix[i,j]` | Yes | How different are their portfolios? (`= 1 - BC`) |
| `comp_dir_matrix[i,j]` | **No** | Does firm j concentrate in zones where firm i is underweight relative to the global market? Use i as acquirer, j as target. |

For `comp_dir`, the direction matters:
- `comp_dir[i,j]` = does target j fill acquirer i's gaps?
- `comp_dir[j,i]` = does target i fill acquirer j's gaps?

---

## Troubleshooting

**`ModuleNotFoundError`** — you forgot to activate the venv. Run `source venv/bin/activate` first.

**`FileNotFoundError`** — check that the data files are in the exact folder paths listed above (`~/Desktop/output/week2_inputs/` and `~/Desktop/output/kmax_sweep/`).

**Script runs slowly past the K-means step** — normal, the embedding decode step loads ~290MB into memory. Should complete in under 15 minutes total.
