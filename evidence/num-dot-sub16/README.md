# A subnormal 16-bit operand and `tl.dot` on sm_90

Evidence for library issue L126, written by the coordinator (2026-11-01). Triton 3.8.0 (TVA's
`.venv`), NVIDIA H200 NVL (sm_90), GPU 0. Scripts: [run.py](run.py) (output [out.txt](out.txt)) and
[probe.py](probe.py) (output [probe.out](probe.out)).

**Question.** Library issue L125 found that the fp8 `tl.dot` facts failed when a product with a
subnormal operand sets the alignment window: the tensor core reads a subnormal operand's exponent
field as the smallest normal's. Do `DotF16` and `DotBF16` fail the same way? `evidence/num-dot`'s
Part 2 had no such input.

**Method.**
- **`probe.py`.** Take `K = 16`, row 0 of `a` and column 0 of `b`, with one product made of a
  subnormal operand and a large partner (product `2^-9`). The other 15 products have normal operands,
  each with every fraction bit set, just below the unit of a window that follows the subnormal's field
  read as the smallest normal's. The same case is also run with normal operands of the same product.
  The large product is placed at `k = 0`, `7` and `15`, on `wgmma` (`64×64`, 4 warps) and `mma.sync`
  (`16×64`, 1 warp). Each result is compared with the exact sum (fractions) and with `DotF16`'s bound
  `(K+2)·2^-22·Σ|a·b| + (K+2)·2^-126` (also `DotBF16`'s).
- **`run.py`.** A randomized sweep of the same kind of input, at `K = 16, 32, 64`, on both paths, with
  the subnormal in `a` or in `b`, on the diagonal elements of a `tl.dot`. Its large partners stay below
  `2^15`, so for `bfloat16` its products are tiny and the absolute term dominates; `probe.py` covers
  `bfloat16` with a large partner.

**Result** (`probe.out`).

| Case | Path | error / bound |
|---|---|---|
| `float16`, subnormal `a = 2^-23`, `b = 2^14` | `wgmma` and `mma.sync`, every `k` | `53.18` |
| `float16`, normal `a = 2^-14`, `b = 2^5` (same product) | both | `0.007` |
| `bfloat16`, subnormal `a = 2^-133`, `b = 2^124` | both, every `k` | `13.12` |
| `bfloat16`, normal `a = 2^-126`, `b = 2^117` (same product) | both | `0.014` |

`run.py`'s sweep (`out.txt`) reaches `106` times the bound for `float16` (`wgmma`, `K = 16`), and
more than `24` times at every `K` and on both paths. For `bfloat16` it stays below `0.001`: its large
partners are too small, as said above.

**Conclusions.**
1. **`DotF16` and `DotBF16`, as stated before round 25** (relative to `|acc| + Σ|a·b|`), **do not
   hold** when a product with a subnormal operand has the largest exponent field of its chunk while
   being far smaller. The products below the window's unit lose
   their low bits, and the error is up to `53` (`float16`) and `13` (`bfloat16`) times the facts'
   bound on these inputs. With normal operands of the same products the error stays far inside the
   bound. It happens on both instruction paths, wherever the product sits in the chunk. (Round 26,
   library issue L132: the bound here and in the table's `error / bound` is the facts' as stated
   before round 25. Since round 25 the names denote the facts restated relative to lifted
   magnitudes (`DotLiftAcc`), which hold on these inputs, at `0.104` and `0.1025` of their bound:
   [evidence/num-dot-sub16-round25](../num-dot-sub16-round25/README.md), Part F.)
2. **What the data say about the window.** The `float16` losses fit a window of 25 bits below the
   field `2^0` that the subnormal product reads as: each of the 15 products loses its bits below
   `2^-25`. An earlier probe by the coordinator, with the large product `2^-24 · 2^15` and smaller
   products `1.5·2^-23`, showed no loss, because those products are whole multiples of that window's
   unit. The probe here is built to lose bits.
3. **The fix** is the library's (L126). It can follow L125's: count a subnormal operand at the format's
   smallest normal (`liftMag`) in the facts' magnitudes, with evidence for the window's width. The
   kernels that cite `DotF16` must then be re-proved.
