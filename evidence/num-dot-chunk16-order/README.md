# A 16-bit `tl.dot` on sm_90: which `k` each instruction adds, and the chunks of 64

Library evidence (not a kernel's), for library issue L150 (round 33). Triton 3.8.0 (TVA's `.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-11-08. Script: [run.py](run.py); output:
[output.txt](output.txt) (a few minutes). It builds on [evidence/num-dot16-model](../num-dot16-model/README.md)
(the bit-level model of one instruction) and [evidence/num-dot-chunk16](../num-dot-chunk16/README.md) (round 32).

**Question.** Round 32 stated `DotF16Chunk` and `DotBF16Chunk` per chunk of 16 consecutive `k`, one instruction per
chunk. Round 32's library review found them false on `mma.sync` when an operand of the 16-bit `tl.dot` was converted
in registers from an 8-bit load: the operands' layout then has `kWidth = 4` and each instruction adds the
`k ≡ 0, 1 (mod 4)` of a block of 32, then the `k ≡ 2, 3`; a `K = 32` launch has a device result that no partial
values of 16-chunks reach. Which `k` does each instruction add, for every operand origin, and do the facts hold when
restated per chunk of 64 consecutive `k` (`lean/TVA/Rules/Num/Dot16Chunk.lean`, round 33)?

**Source.** `primitives/triton-3.8.0-AccelerateMatmul.cpp`, lines 385-427
(`computeOrigBitWidth`: the narrowest element type of the loads in the operands' elementwise backward slice, halved
when a `tl.join` doubles the element width packing the last dimension; 4 for an `Fp4ToFp` conversion) and 561-568
(on `mma.sync`, both operands get that width, which starts from their own 16 bits);
`primitives/triton-3.8.0-TritonGPUAttrDefs.td` lines 1478-1488 (`kWidth = max(32 / bitwidth, 1)`).
On `wgmma` a register operand's `kWidth` comes from its own 16-bit type (2), and `b` is read from shared memory.

**Method** (`run.py`).
- **Part W.** For `float16` and `bfloat16` dots, operands loaded in the format, as `float32`, `float64`, `int16`,
  `int8`, `uint8`, `bool`, `float8e5`, `float8e4nv` and converted, or loaded as `float8e5` and unpacked into twice the
  `k` by `tl.join(lo, hi).reshape`, on 8 shapes (`16×16` with 1 warp to `128×128` with 8) and `K = 32, 128`: the path
  (from the PTX) and `kWidth` (from the TTGIR).
- **Part P.** The `k` of each instruction, found on the device: row values `a = 2^-14`, `b = 2^-13` (products
  `2^-27`) with one product `1` at `k = big`; the result is above `1` exactly when an instruction before the one
  holding `big` added 16 products `2^-27`, so scanning `big` finds the first instruction's `k`; they are zeroed and the
  scan repeats. 104 configurations (both formats; `a` in the format, from `float8e5`, unpacked by `tl.join`;
  `16×16`, `32×32`, `16×64`, `32×64` on `mma.sync` with 1, 2 or 4 warps, `64×64` on `wgmma`; `K = 16` to 256),
  against the rule the model uses, and whether each instruction's `k` lie in one aligned block of 64 with the blocks
  in increasing order.
- **Part X.** Round 32's counterexample (lr-32b): `K = 32`, accumulator `(1 − 2^-24)·2^-25`, products `1` at `k = 0`,
  `−1` at `k = 2`, `3·2^-25` at `k = 1` and `(1 − 2^-11)·2^-25` elsewhere, `a` from `float8e5` on `mma.sync`; whether
  partial values of 16-chunks meeting the round-32 bound exist (exact rationals at every kink of the piecewise-linear
  slack), and the round-33 step.
- **Part M.** The per-instruction model with the instructions' `k` (Part P's rule) against the device on every
  element, on both formats, `a` in the format, from `float8e5`, from `int8`, unpacked by `tl.join`, and `b` from
  `float8e5` (with `a` in the format or from `float8e5`), on `mma.sync` and `wgmma`, `K = 16` to 256, six kinds of
  inputs drawn twice each: `normal`, `spread`, `bits` (every bit pattern of the source dtype), `accdom` (an
  accumulator `2^20` with every product just below the window's unit, the second arm's worst case), `grow` (the
  accumulator outgrows `Mj` within the chunk: products `1` in the chunk's first instructions, the last one's just
  below its unit), `sub` (subnormal operands and lifted products far above the products). On sampled elements
  (32 per launch) each chunk of 64 is checked in exact rationals at the smallest `Mj`, with the partial values the
  model's chain gives at each chunk's end (its last value is the device's result); where the instructions are 16
  consecutive `k` the round-32 bound is checked too, and at `K = 32` with `kWidth > 2` whether partial values of
  16-chunks exist.

**Result** (`output.txt`).
- **Part W.** `mma.sync` has `kWidth = 2` for operands loaded in 16 bits or wider (`float16`, `bfloat16`,
  `float32`, `float64`, `int16`), `4` when either operand was loaded in 8 bits (`int8`, `uint8`, `bool`,
  `float8e5`, `float8e4nv`) and `8` for the `tl.join` unpacking on 1 warp (`4` on 2 or more warps for some shapes);
  `wgmma` has `kWidth = 2` for a converted register operand and none otherwise. No other value occurs.
- **Part P.** All 104 configurations match the rule: `kWidth = 2` and `wgmma`, 16 consecutive `k` per instruction in
  increasing order; `kWidth = 4`, the `k ≡ 0, 1 (mod 4)` then the `k ≡ 2, 3` of each block of 32; `kWidth = 8`, for
  `K ≥ 64` the `k ≡ 2i, 2i+1 (mod 8)` of each block of 64 (`i = 0, …, 3`), at `K = 32` the `k ≡ 0..3` then `4..7
  (mod 8)`; at `K = 16`, the 16 `k`. Every instruction's `k` lie in one aligned block of 64 (the widest spans 58 `k`)
  and the blocks come in increasing order.
- **Part X.** With `a` from `float8e5` (`kWidth = 4`) the device gives `0`, an error of `32.99·2^-25`; no partial
  values of 16-chunks meet the round-32 bound (the round-32 facts are false there); the round-33 step is `0.0081` of
  its bound. With `a` loaded in `float16` (`kWidth = 2`) the device gives `5.66·10^-7`, both bounds hold.
- **Part M.** The model equals the device on all `2605056` element results. Of `62976` elements checked exactly, none
  is outside the chunk-of-64 bound; the largest ratio is `0.381` (`float16`, `accdom`, second arm), `0.308` for
  `bfloat16`, and `0.091` for `grow` (first arm). Where the instructions are 16 consecutive `k`, none is outside the
  round-32 bound either. On these random and constructed inputs with `kWidth > 2` at `K = 32`, partial values of
  16-chunks existed: the round-32 facts fail on inputs built for it (Part X), not on these.

**Conclusions.**
1. The `k` an instruction adds depend on the operands' origin on `mma.sync` (`kWidth` 2, 4 or 8), but always lie in
   one aligned block of 64 taken in order, four instructions of 16 per block. So the chunk facts hold per chunk of 64
   consecutive `k`: with the per-instruction model (equal to the device on every element, every partition) the
   derivation in `Dot16Chunk.lean` bounds each chunk by `min(2^-13·Mj, 2·A_j + 2^-22·Mj) + 2^-20·|s j + c_j| +
   2^-143` whatever the partition, and every measured step is within `0.39` of it.
2. The round-32 facts (16 consecutive `k`) are false on `mma.sync` with an operand converted from 8 bits (Part X);
   `DotF16` and `DotBF16`, whose derivation needs only 16 products per instruction, are unaffected.
3. `kWidth = 8` occurs for a 16-bit dot: an 8-bit load unpacked by `tl.join` into twice as many `k` (the int4-weight
   pattern); its instructions span up to 58 `k` of a block of 64.
