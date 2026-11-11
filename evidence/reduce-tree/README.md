# The trees of a float reduction, and a reduction rewritten across loop iterations

Library evidence (not a kernel's), for library issue L149. Round 32.

**Question.** `tl.sum` and its axis and fusable forms combine their positions along a tree Triton chooses, and
until round 32 the library allowed every tree over the positions (`sum_err_covers`, `sumc_err_covers`: depth up to
`n − 1`). What trees does Triton 3.8.0 use on sm_90? Is their depth bounded whatever the layout? And does any
compiler pass combine a reduction's elements in another order than the source's?

**Sources** (Triton 3.8.0, pinned in `primitives/`).
- `triton-3.8.0-ReduceOpToLLVM.cpp` 33-105 (the lowering of `tt.reduce`): broadcast registers removed (43-51);
  the elements of the axis one thread holds reduced by `treeReduce` (237-322: the registers of the axis brought to
  the front, then pairs of adjacent values combined level by level, 119-142; for 16-bit and `float32` adds two
  values may be packed in one vector add, then combined once more, 266-315); a butterfly of `shuffleXor` over the
  lane bits along the axis (327-389: one combination per lane bit, `redux` only for some integer combines,
  372-374); then, while the axis is not reduced, a layout conversion through shared memory that puts the warps'
  partial values on lanes, and the same butterfly (66-92, at most 2 rounds). Each stage halves the positions
  left, so the tree over the `2^k` positions of a Triton axis is perfect, of depth `k`, whatever the layout.
- `triton-3.8.0-TargetInfoBase.h` 112-118: `getReductionTreeArity` is 2 unless a target overrides it; only AMD's
  does.
- `triton-3.8.0-ScanOpToLLVM.cpp` 29-49, 53-84, 133-244, 505-542 (scans): sequential within a thread,
  Hillis-Steele over the lanes, then sequential over the warps and repetitions of the layout along the axis. The
  depth of a position's tree depends on the layout (how many elements a thread holds along the axis, and how many
  warps and repetitions), which the source does not fix; no depth fact is stated for scans.
- `triton-3.8.0-OptimizeThreadLocality.cpp` 242-372 and `triton-3.8.0-nvidia-compiler.py` 276: a pass rewrites a
  reduction inside a `for` loop when its combine is one float `addf`, `mulf`, `maximumf`, `maxnumf`, `minimumf`
  or `minnumf`, its operand is a rank-2-or-more tile in a blocked layout produced directly by a `tl.load`, it
  reduces the last axis, the thread holds more than one element along it, its result has one use, an operation
  whose result has one use, the loop's yield, and the loop-carried value starts from a constant. The rewrite
  reduces each thread's elements in the loop, accumulates those partial values across the trips, and reduces them
  once after the loop: the trips' elements are reassociated when the update is the combine. (Round 33, library
  issue L149: the pass clones the update operation, so with another update the result is no association of the
  source, and it reads the IR after folding, so operands that are loads up to operations that fold away count; see
  [evidence/thread-locality](../thread-locality/README.md). Round 34: it reads the IR after inlining, folding,
  specialisation and dead-code removal, so the exclusion is now stated on the source, more broadly than this
  pattern: docs/exclusions.md, docs/SEMANTICS.md "Float `+` and `-`".)

**Method** ([run.py](run.py), output [output.txt](output.txt)). One process, an empty Triton cache in a temporary
directory, `CUDA_VISIBLE_DEVICES=0`.
- T, the tree's shape: a tuple `tl.reduce` of leaves `(0, 1)` with the combine `(d1, c1), (d2, c2) ->
  (max(d1, d2) + 1, c1 + c2)` returns the tree's depth and its number of leaves, in `int32`, `int16` and `int64`
  (the layouts of 32-, 16- and 64-bit loads). Expected: depth `log₂ n`, count `n`.
- F, `tl.sum` itself: every leaf `1 + 2^-p` (`p` the format's fraction bits) and, separately, every leaf
  `1 − 2^-(p+1)`. A partial sum of `m` such leaves is a float exactly when `m` is a power of two, so a sum of
  `n = 2^k` leaves is exact along a perfect tree and rounded at the first node of any other tree whose two
  subtrees differ in size. `float32`, `float16`, `bfloat16`, `float64`, and `tl.sum(x * y)` with `y = 1` (a sum of
  products, `enable_fp_fusion` on).
- Layouts: 1-D loads of `n = 1 … 16384` with `num_warps` 1, 2, 4, 8, 16, contiguous (vectorised) and strided by 3
  (one element per access); 2-D `[M, N]` loads, `M, N ∈ {1, …, 256}` with `M·N ≤ 16384`, along axis 0 and 1,
  row- and column-major, through `tl.trans`, `keep_dims` both ways, `num_warps` 1, 4, 8; the result of an `int8`
  and a `float16` `tl.dot` (`K = 32`) plus a loaded tile, `M, N ∈ {16, 32, 64, 128}`, `num_warps` 1, 2, 4, 8
  (`mma.sync` and `wgmma`, read from the PTX); a 3-D tile reduced along axis 2 and then along axis 1. Each
  reduction's operand layout is read from the TTGIR.
- L, the loop rewrite: `acc = acc + tl.sum(x, axis=1)` over two trips of a loaded `[M, N]` tile, from
  `tl.zeros`. Trip 0 holds `2^24` in column 0 and `1` in the last column, trip 1 `−2^24` and `1`: each trip's sum
  along any tree is `2^24` (`2^24 + 1` rounds to even) and `−2^24 + 1`, so every association of the source gives
  `1`. The same loop from a loaded accumulator, and on a 1-D tile.

**Result** (Triton 3.8.0, NVIDIA H200 NVL, 2026-11-08).
- L: the rewrite happens and changes the result. For `[M, N]` = `[16, 64]`, `[32, 128]`, `[64, 256]`, `[4, 1024]`,
  `[128, 32]` (4 or 8 warps) from `tl.zeros`, the TTGIR has a 3-D `tt.reshape` of the tile, a loop-carried
  `[M, N / e]` value (`e` the elements a thread holds along the axis) and a second `tt.reduce` after the loop, and
  every row stored `2.0`, where every association of the source gives `1.0`. From a loaded accumulator, and on a
  1-D tile, the loop is not rewritten and every row stored `1.0`.
- T and F: 7960 configurations, each with a tree probe and a float probe; in every one the tree had depth
  `log₂ n` over exactly `n` leaves, and the float sum was exact, for both leaf values (0 failures): 1800 1-D
  (15 sizes × 5 warp counts × 2 load patterns × 6 dtype pairs, with and without `x * y`, × 2 leaf values; the
  `float64` and the `float16`/`bfloat16` packed adds among them), 5616 2-D (both axes, row- and column-major,
  `tl.trans`, `keep_dims`, `float32`, `float16`, `bfloat16`), 256 on `tl.dot` results (12 `mma.sync` and 4
  `wgmma` configurations per the PTX, the reductions' operands in the `nvidia_mma` layout), and 288 reductions
  of reductions (the output's counts are cumulative, in the order L, dot, 3-D, 1-D, 2-D). 259 distinct reduction operand layouts were seen in the TTGIR (228 blocked, 31
  `nvidia_mma`).

**Conclusions.**
1. On sm_90, every float reduction of the tested forms combined the `n = 2^k` positions of its axis along a
   perfect binary tree of depth `k`, whatever the layout; with the sources above (each stage of the lowering
   halves the positions left) this is `Rules.Num.SumDepth`: the depth is at most `⌈log₂ n⌉`. Which positions
   pair up depends on the layout; the fact states only the depth.
2. Scans are not covered: their depth depends on the layout, and no depth fact is stated for them.
3. A reduction inside a `for` loop that `OptimizeThreadLocality.cpp` rewrites (above) is computed in
   another association than the source's, which no tree of each trip's reduction gives (L): such a launch is
   outside the library (`docs/exclusions.md`), and `SumDepth` does not speak of it. With an update other than the
   combine the rewrite computes no association of the source at all (a miscompile; round 33,
   [evidence/thread-locality](../thread-locality/README.md)). The exclusion covers every such loop and more (round
   34: stated on the source, whatever the update, the uses and the layout).
