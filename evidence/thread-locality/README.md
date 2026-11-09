# What Triton's `OptimizeThreadLocality` pass rewrites, and what the rewrite computes (library issue L149)

**Question.** Round 32 excluded the loops that Triton 3.8.0's `OptimizeThreadLocality` pass rewrites, describing
them by their source ("a tile produced directly by `tl.load`") and the rewrite as a reassociation. Round 32's
library review found both wrong: the pass reads the IR after the front end has inlined helpers and folded
constants, so operands that are loads up to operations that fold away are rewritten too; and it clones the
loop's update operation, so an update other than the reduction's combine gives a value that no association of the
source gives. Which loops does the pass rewrite on sm_90, and what do they compute? Round 33 restated the pattern on
the IR; round 33's library review (lr-33a) found rewritten loops that the source wording of the IR's conditions
missed (identities after the reduction, dead uses, compile-time `if`s, `tl.static_range(1)`, helpers, a specialised
runtime argument). Round 34 adds Part W for those forms and the edges of a broader exclusion stated on the source,
and a static scan of the claims.

**Source** (`primitives/triton-3.8.0-OptimizeThreadLocality.cpp`, on for NVIDIA,
`primitives/triton-3.8.0-nvidia-compiler.py:276`, after coalescing and layout propagation). Lines 252-372 match
a `tt.reduce` whose combine region is one float `addf`, `mulf`, `maximumf`, `maxnumf`, `minimumf` or `minnumf`;
whose operand is a tensor of rank 2 or more in a blocked layout, defined by a `tt.load`, reduced along its last
axis, with more than one element per thread along it; whose result has one use, an operation with one use, the
`scf.yield` of the `scf.for` whose body holds the reduction; whose loop-carried value starts from an
`arith.constant`. Lines 393-520 rewrite it: a new carried value per thread starts at the combine's neutral
element; each trip reduces the thread's own elements and applies a clone of the update operation `U` to the
carried value and that partial result (operand 0 and operand 1 of `U` taking them, whatever the source's order);
after the loop the original reduction runs on the carried values, and `U` is applied once more, to the initial
constant `c` and that result. The device computes `U(c, R(v))`.

**Method** (`run.py`). On an H200 NVL (sm_90), Triton 3.8.0 (torch 2.11.0+cu130), GPU 2, 2026-11-08: kernels of
the form `acc = <constant>; for t in range(T): acc = U(acc, tl.sum(<operand>, axis=1))` on `[64, 64]` tiles, 4
warps unless stated. For each: whether the TTGIR holds the rewrite (a `tt.reshape` to one more dimension), the
values stored, the source's value, and, in Part U, the value of `U(c, R(v))` where it does not depend on which
thread holds which element. Every value is exact in `float32` along every association within the trips, so a
stored value other than the source's is not a rounding.
- Part U (the update): `x` is 1 everywhere but 2 in column 1, four trips (row sum 65, max 2, min 1): `acc - sum`,
  `sum - acc`, `acc + max`, `maximum(acc, sum)`, `acc * sum` from 1, `acc + min`, `minimum(acc, sum)`,
  `maximum(acc, max)` and `acc + sum` (the combines), `acc / sum`, `acc * max` from 1; and `maximum(acc, sum)` with
  a 1 in column 0 on trip 0 and in column 63 on trip 1.
- Part F (the operand): two trips, `2^24` in column 0 and `1` in column 63, then `-2^24` and `1`. Each trip's sum
  is `2^24` and `-2^24 + 1` along any tree, so the source gives `1.0`; folding each thread's elements across the
  trips first gives `0 + 2 = 2.0`. Operands: a plain load; `.to(tl.float32)` of a `float32` load; `* 1.0`;
  `* SCALE` with `SCALE: tl.constexpr = 1.0`; `/ 1.0`; `- 0.0`; `+ (-0.0)`; a block-pointer load; a load in a
  helper; `tl.maximum(x, x)`; `tl.trans(tl.trans(x))`; a reshape to its own shape; `tl.where` on a compile-time
  condition; a load with an always-true mask, with a runtime mask, with a cache modifier. Controls: `+ 0.0`;
  `tl.sum(x * 2.0) * 0.5`; a `float16` load (`2048`, `1`; `-2048`, `1`); a host-descriptor load; the update in a
  runtime `if`; an initial value computed from a load; `tl.where(x == x, x, 0.0)`; the sum used twice; a product
  with another load.
- Part L (the loop and the layout): `tl.range(…, num_stages=3)`, `tl.static_range`, `while`; the last axis against
  axis 0; a 1-D tile; 1, 2 and 8 warps; tiles `[1, 64]`, `[64, 1]`, `[2, 16]`, `[128, 128]`.
- Part W (round 34, same data as Part F, so every association within the trips gives `1.0` unless stated; the source
  value is computed in `float32` one operation at a time): every form of lr-33a's `otl_wording33.py` — `.to(tl.float32)`
  of the sum, `* 1.0` on the sum, `(acc + s) * 1.0`, `-(-s)`; a dead `s * 3.0`, a dead `acc * 2.0`, a store under a
  constexpr-false `if`; the reduction under a constexpr-true `if`, in a nested `tl.static_range(1)` and
  `tl.static_range(2)`; the update or the reduction in a helper; initial values `tl.zeros + iv` with a runtime int `iv`
  of `0` and of `1`, `tl.zeros + fv` with a runtime float `0.0`, `tl.zeros * 1.0`; `tl.broadcast_to` its own shape; a
  runtime `if` — and the edges of the broader exclusion: `tl.sum(x * k, 1)` with a runtime int `k` of `1` and of `2`;
  `x.to(tl.float64).to(tl.float32)`; a `float16` tile summed into a `float16` accumulator (`2048`, `1`; `-2048`, `1`)
  and its `tl.reduce` with a maximum combine (`tl.max` itself converts `float16` to `float32` first,
  `primitives/triton-3.8.0-standard.py:185-187`; `tl.sum` does not, lines 270-293); a two-operand `tl.reduce` of two
  loads whose combine is one `+`; `acc + tl.sum(x, 1) * 2.0`; `tl.range(0, T, loop_unroll_factor=2)` with a
  compile-time and a runtime trip count of 2 and 3 (trips 2 and 3 zero). The compiles that may fail run in a child
  process of the script, which records the error.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/thread-locality/run.py` from TVA's root (under two minutes).
Round 34 re-ran the whole script on GPU 2 on 2026-11-08; Parts U, F and L gave the same values as in round 33.

The claims (`claims_scan.py`, output `claims_scan.txt`; run with `python3 -I evidence/thread-locality/claims_scan.py`,
no GPU): a static scan, with Python's `ast`, of every claim's source for each reduction anywhere inside a `for` loop
over `range` or `tl.range`, following the helpers of the same source that the loop's body calls, with a generous test
of whether the operand can be a load up to identities and whether the result flows into a loop-carried variable.

**Result** (`output.txt`).
- Part U: every form was rewritten. The stored value was `U(c, R(v))` in each case where that value does not
  depend on the layout: `acc - sum` and `sum - acc` stored `+260` (the source gives `-260` and `0`); `acc + max`
  stored `-inf` (source `8`); `acc + min` `+inf` (source `4`); `acc * sum` from 1 stored `0` (source
  `17850625`); `acc / sum` NaN (source `0`); `acc * max` from 1 `-inf` (source `16`); `maximum(acc, sum)` and
  `minimum(acc, sum)` stored the source's `65` and `0` on that data, and `2.0` where the source gives `1.0` when the
  two 1s are in different trips and columns; the combines stored the source's value on exact data.
- Part F: rewritten, storing `2.0` where the source gives `1.0`: the plain load, `.to(tl.float32)`, `* 1.0`,
  `* SCALE`, `/ 1.0`, `- 0.0`, the block pointer, the helper, `tl.maximum(x, x)`, the double transpose, the
  reshape, the compile-time `tl.where`, both masks, the cache modifier. Not rewritten, storing `1.0`: `+ (-0.0)`
  (not folded here) and every control.
- Part L: `tl.range` with stages was rewritten; `tl.static_range` (unrolled), `while`, axis 0, the 1-D tile and the
  tiles with one element per thread along the axis (`[1, 64]`, `[64, 1]`, `[2, 16]`) were not; 1, 2 and 8 warps
  and `[128, 128]` were.

- Part W: rewritten, storing `2.0` where the source gives `1.0`: `.to(tl.float32)`, `* 1.0`, `(acc + s) * 1.0`,
  `-(-s)`, both dead uses and the store under a constexpr-false `if`, the constexpr-true `if`, `tl.static_range(1)`,
  both helpers, `tl.zeros * 1.0`, `broadcast_to`, `x * k` with `k = 1` (specialised), `.to(tl.float64).to(tl.float32)`,
  the `float16` sum; the initial value `tl.zeros + iv` with `iv = 1` (specialised) stored `3.0` where the source gives
  `1.0` (`1 + 2^24` rounds to `2^24`); the `float16` maximum stored `-inf` for `4.0`. Not rewritten, storing the
  source's value: `iv = 0`, `fv = 0.0`, `k = 2` (`2.0`), `* 2.0` (`2.0`), the runtime `if`, `tl.static_range(2)` with
  `* 0.5` (`0.5`, the correct `float32` value: `2^23 + 0.5` rounds to `2^23`; lr-33a's label "1.0" there was not the
  source's value), and `loop_unroll_factor=2` with a compile-time trip count. With a runtime trip count and
  `loop_unroll_factor=2` the compile aborts in the pass (assertion `isa<BlockArgument>(accumOperand)`, line 327: after
  unrolling, the second copy's update takes the first copy's result); the two-operand `tl.reduce` fails the
  verifier after the pass (`'tt.reduce' op requires the same shape for all operands`). Both are errors on the host,
  with no outcome.
- The claims: 4 candidates, none in the exclusion when read: `extend_attn_bf16` lines 397 and 563 (`tl.max(qk, 1)`,
  `qk` a `tl.dot` result, scaled and masked by `tl.where` on a runtime mask, so not equal to a load); `moe_router_bf16`
  line 145 (a 1-D sum of a product with a mask, stored, not carried); `paged_attention_f16` line 68 (`tl.sum(p[:, None]
  * v, axis=0)`, axis 0 and a product with the softmax weights). The other reductions in loops have operands computed
  from `tl.dot`, products of two loads or converted loads (`paged_attention_f16` line 61: `k * q[None, :]`), or are in
  host-side reference code.

**Conclusions.**
1. The pass is a miscompile, not a reassociation: with an update other than the combine the device's value is no
   association of the source's terms (`+260` for `-260`, `0` for `17850625`, `-inf` for `8`).
2. Whether a loop is rewritten is decided on the IR after inlining, folding, specialisation, dead-code removal and
   unrolling, and by the layout (more than one element per thread along the axis): Part W shows it rewritten for
   every form lr-33a found, among them source forms whose reduction has other (dead) uses, sits under a
   compile-time `if` or in a `tl.static_range`, or starts from a runtime argument. A reviewer should not have to
   follow that. Round 34 therefore states the exclusion on the source and more broadly than the pass's pattern
   (docs/exclusions.md; docs/SEMANTICS.md, "Float `+` and `-`"): a float reduction along the last axis of a tile
   of rank 2 or more, an operand of which equals a load's tile up to NaN payload (on the launch's compile-time
   constants and specialised arguments, whatever operations stand between, folded or not, as `+ (-0.0)` here shows
   they need not be), anywhere in a `for` loop's body (helpers, `if`s and `tl.static_range` included), whose result
   flows into a value the loop carries, whatever the initial value, the update, the other uses, the float width and
   the layout. Every rewritten loop of Parts U, F, L and W is in it; so are some loops the pass leaves alone
   (`+ (-0.0)`, a runtime `if`, one element per thread, `* 2.0`), which the exclusion keeps out on purpose.
3. No claim's covered path has an excluded loop (`claims_scan.txt`, read above).
