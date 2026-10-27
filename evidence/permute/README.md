# `tl.permute` and `tl.trans` of any rank

Library evidence (not a kernel's), for library issue L17. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Question.** What does `tl.permute(x, dims)` compute, and `tl.trans` with and without `dims`, for
ranks 2, 3 and 4?

**What the library reads.**
- `tl.permute(x, *dims)` and `tl.trans(x, *dims)` take `dims` as a tuple or as separate arguments
  (`primitives/triton-3.8.0-core.py:2070-2121`). Without `dims`, `tl.trans` swaps the last two axes:
  `dims = range(n - 2) + [n - 1, n - 2]`, and it rejects ranks 0 and 1 (`:2091-2097`).
- `semantic.permute` (`primitives/triton-3.8.0-semantic.py:701-709`) requires `dims` to be a
  permutation of `0, …, n-1`; the result's dimension `k` is the input's dimension `dims[k]`
  (`tt.trans` with `order = dims`).
- `x.T` is `permute(x, (1, 0))` (`primitives/triton-3.8.0-code-generator.py:1535-1537`).

The library's definition is NumPy's and torch's: `out[o] = x[i]` with `i[dims[k]] = o[k]`.

**Method.** One program loads a block whose elements are their own row-major positions, applies
the operation, and stores the result in row-major order; the script compares it with
`torch.permute`.

**Result.**

| Case | Result |
|---|---|
| rank 3, `[2, 4, 8]` and `[16, 32, 64]`: every permutation, through `tl.permute(x, dims)`, `tl.trans(x, dims)` and `tl.permute(x, *dims)` | 6 of 6 equal `torch.permute`, for each form and shape |
| rank 4, `[2, 2, 4, 8]` and `[4, 8, 16, 32]`: every permutation, the same three forms | 24 of 24 equal `torch.permute`, for each form and shape |
| `tl.trans(x)` on rank 3 and rank 4 | equals `dims = [0, 2, 1]` and `[0, 1, 3, 2]` |
| `tl.trans(x)` and `x.T` on rank 2, `[8, 16]` | equal `dims = [1, 0]` |

**Conclusions.**
1. `tl.permute(x, dims)` and `tl.trans(x, dims)` are the permutation of axes with output dimension
   `k` = input dimension `dims[k]`, for every permutation of ranks 3 and 4 tried, on small and
   multi-warp blocks.
2. `tl.trans(x)` without `dims` swaps the last two axes, for every rank from 2 up.
