# The accumulator of an int8 `tl.dot` (library issue L141, round 32)

**Question.** Round 31 stated that an int8 `tl.dot` with an accumulator other than `int32` is rejected on the
host. Round 31's library review (lr-31b) found a `uint32` accumulator accepted. Which accumulator dtypes does
Triton 3.8.0 accept for `tl.dot(a, b, acc)` with `int8` operands, and what does it compute with a `uint32` one?

**Source.** `semantic.dot` (`primitives/triton-3.8.0-semantic.py:1429-1520`): with no `out_dtype` it takes the
accumulator's dtype; it then requires the accumulator's dtype to equal `out_dtype` ("expected accumulator dtype
…; pass out_dtype=… to use this accumulator dtype") and the operands to be `int8`; the result's dtype is `int32`
whatever `out_dtype` says, and the accumulator's handle is passed to `tt.dot` as it is. The IR's integers are
signless, so a `uint32` accumulator is an `i32` tensor, the type of the `int32` result, and the verifier accepts
it; any other width, and a float accumulator, differs from the result's `i32` and is rejected.

**Method** (`run.py`).
- A1: `tl.dot(a, b, acc)` with `int8` operands and an accumulator loaded from a buffer of each of ten dtypes
  (`int32`, `uint32`, `int8`, `int16`, `int64`, `uint8`, `uint16`, `uint64`, `float16`, `float32`), with
  `out_dtype` omitted, equal to the accumulator's dtype, and `tl.int32`.
- A2: a `uint32` accumulator on both paths (`mma.sync` at `(16, 16)`, `wgmma` at `(64, 64)`, 4 warps), `K = 32`,
  `64`, `256`, operands in `[-128, 127]`, accumulator bits whose `int32` reading leaves room
  (`|reading| + Σ|a·b| ≤ 2^31 - 1`, the edge on two rows, half of the bits at or above `2^31`), `out_dtype`
  omitted and `tl.uint32`, 8 trials each: is the `int32` result the accumulator's bits read as `int32` plus the
  exact sum?
- A3: the result in the kernel: `c < 0` and `c.to(tl.float32)` for the accumulator `3000000000`.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/num-dot-int8-round32/run.py 2>/dev/null | grep -E
"^(Triton|A[0-9] )"` from TVA's root (MLIR prints the diagnostics of the rejected cases on stdout).

**Triton version.** 3.8.0 (torch 2.11.0+cu130). **GPU.** NVIDIA H200 NVL (sm_90), GPU 0. **Date.** 2026-11-08.

**Result** (`output.txt`).
- A1: accepted are an `int32` accumulator (with `out_dtype` omitted or `tl.int32`) and a `uint32` one (with
  `out_dtype` omitted or `tl.uint32`); the result is `int32` and equal to `5 +` the exact sum. Rejected on the
  host are a `uint32` accumulator with `out_dtype=tl.int32` (the front end's check) and every other accumulator
  dtype, with any `out_dtype` (the front end's check when `out_dtype` differs, the IR's verifier otherwise).
- A2: 4096 of 4096 results on `mma.sync` and 65536 of 65536 on `wgmma` at each `K` equal the accumulator's bits
  read as `int32` plus the exact sum, the edge of the room included.
- A3: `3000000000 + 32` is stored as `-1294967264`, `c < 0` holds, and `c.to(tl.float32)` is `-1294967296.0`:
  the result is an `int32` in the kernel.

**Conclusion.** An int8 `tl.dot` accepts an accumulator of `int32` or of `uint32` (never `out_dtype=tl.int32`
with the latter), and nothing else. With a `uint32` accumulator it computes what it computes with the
accumulator's bits read as `int32` (`tl.bitcast .u32 .i32 acc`, `TVA.tl.bitcast`), an `int32` result, so
`DotI8` covers it with that accumulator, on the room of the `int32` reading. This corrects evidence/num-dot-int8
(I5 and conclusion 1), which tested only `int32` and `float32` accumulators.
