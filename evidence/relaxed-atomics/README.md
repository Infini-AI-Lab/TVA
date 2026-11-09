# Atomics whose result is unused, with any `sem` (library issue L157)

Library evidence (not a kernel's), for library issue L157 (user decision 2026-11-08: a library rule
for relaxed read-modify-write atomics whose results are unused, rather than narrowing the claims that
use them). Triton 3.8.0 (TVA's `.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0,
2026-11-08. Script: [run.py](run.py); output: [out.txt](out.txt) (about 30 seconds); Part F (library issue L164):
[float_maxmin.py](float_maxmin.py), [float_maxmin.out.txt](float_maxmin.out.txt).

**Question.** Before L157, a `tl.atomic_*` with `sem="relaxed"` (or `"acquire"`, `"release"`) on
cells that other programs also update was a race in the semantics (`Event.Strong` exempts only
`acq_rel` pairs), so `Compiler.Faithful` said nothing of such a launch. vLLM's `_lora_shrink_kernel`
(`claims/lora_shrink_bf16`, source line 305) adds each split-K part into the output with
`tl.atomic_add(c_ptr, accumulator, mask=c_mask, sem="relaxed")`, its value unused. The rule stated in
L157 (`MemSem.unused`, `lean/TVA/Lang/Prog.lean`; `tl.atomic_add_unused` and the others,
`lean/TVA/Lang/Ops.lean`; docs/SEMANTICS.md, "Atomics") models such an atomic, at `gpu` or `sys`
scope, as the `acq_rel` one. Its argument is in `MemSem.unused`'s docstring: the semantics has no
synchronises-with and already performs a program's atomics after any later step that does not depend
on their result; the only further freedom a relaxed atomic has on the device is to be performed
before its thread's earlier accesses, which no other program can observe when its result is unused,
since its own bytes are reached by atomics only and the PTX memory model performs them with
single-copy atomicity (PTX ISA 9.5, §8.4 strong operations, §8.7 morally strong operations, §8.10.3
atomicity). This folder checks the device side: what Triton emits for these atomics, and that no
update is lost, doubled or torn under contention, with each `sem`.

**Method** (`run.py`).
- Part P, the instructions. For each `tl.atomic_*` (`add` on `int32`, `float32`, `float16`,
  `bfloat16`, `float64`; `max`, `min` on `int32` and `float32`; `and`, `or`, `xor`, `xchg` on
  `int32`; `xchg` on `float32`; `cas` on `int32`), as a tensor atomic of 128 lanes (4 warps) with
  each `sem` and scope `gpu` and `sys`, its result unused; with its result stored (`relaxed` and
  `acq_rel`, `gpu`); and as a scalar atomic, unused and used: the `atom`, `red` and `cp.reduce`
  instructions of the PTX, the strong loads, and whether a barrier was added. Also a scalar
  `tl.atomic_add(p, 0.0)` on a cell holding `-0.0`, with `relaxed`, `acquire` and `acq_rel`.
- Part A, atomicity under contention. `G` programs of 256 lanes each update one of 64 cells, lane
  `j` of program `p` the cell `(7p + 13j) mod 64` (scattered) or `(8p + j) mod 64` (contiguous runs,
  which Triton vectorises: `.v2.f32`, `.v2.f16`, `.v2.bf16`), with the result unused, for each op
  and each `sem` (`relaxed` also at `sys` scope), 50 launches per case. `G = 4096` (16384 updates
  per cell), except `float16` and `bfloat16`, `G = 32` (128 updates per cell). Each lane's operand
  is chosen so that the final memory is the same for every order of the updates: integers, and for
  floats small integers in `{-2, …, 2}` whose every partial sum is exact in the format; `max`, `min`
  of distinct values; `and`, `or`, `xor` of bit masks. Any lost, doubled or torn update gives
  another value. For `xchg` and `cas` (compare with `0`, from `0`), whose final value depends on the
  order, each cell must end as a value some lane wrote to it.
- Part M, other memory beside the atomics. 2048 programs each load their own input, store their
  own block before and after an 8-trip loop of two relaxed `atomic_add`s each into 256 shared
  `int32` cells, and the whole memory is checked after each of 50 launches, with `relaxed` and with
  `acq_rel`.
- Part L, the split-K pattern of `_lora_shrink_kernel`: `bfloat16` tiles of `32 × 16`, `tl.dot` per
  part of K, `accumulator *= scaling`, the relaxed masked `tl.atomic_add` into a zeroed `float32`
  output, at vLLM's two configurations (`BLOCK_K = 256`, `SPLIT_K = 64`; `BLOCK_K = 32`,
  `SPLIT_K = 8`), `K = 4096` and `14336`, `M` not a multiple of the tile (masked rows), 20 launches
  each. With integer data in `{-2, …, 2}` and `scaling = 1` every partial sum is exact, so the output
  must equal the exact product; with random data the error against the `float64` product is
  reported for `relaxed` and for `acq_rel`.
- Part F, which atomic a float `atomic_max`/`atomic_min` runs (library issue L164; script
  [float_maxmin.py](float_maxmin.py), output [float_maxmin.out.txt](float_maxmin.out.txt), GPU 0,
  2026-11-08, a few seconds). Triton 3.8.0 (`semantic.py:1292-1373`) predicates the two integer
  atomics on `mask & !signbit(val)` and `mask & signbit(val)`. For `float32` and `float64`,
  `max` and `min`, `relaxed`, one tensor call of 2048 lanes, each lane on its own cell: every
  (old, operand) pair of 40 bit patterns (zeros of both signs, `±1`, `2`, `-3`, `5`, `±inf`,
  subnormals, quiet and signalling NaNs of both signs, the rest random), every 17th lane masked
  off. Checked per lane: the new bits are those of the one integer atomic the operand's sign bit
  picks, applied to the old bits; the returned value is the old value; the new value is inside
  `fBound`; a masked-off lane's cell is unchanged.

**Result** (`out.txt`).
- Part P: every atomic of every op, dtype, `sem` and scope compiles to one atomic
  read-modify-write per enabled lane, with the source's order and scope (wording of library issue
  L164: a float `atomic_max` or `atomic_min` is two integer `atom`s predicated on the operand's
  sign bit, so exactly one runs per enabled lane, NaN operands included: `max.s32` (sign bit
  clear) or `min.u32` (set) for `atomic_max`, `min.s32` or `max.u32` for `atomic_min`, `.s64` and
  `.u64` for `float64`; see Part F; every other one is one `atom`, in the PTX: on sm_90 `ptxas`
  compiles a 16-bit `atom` to a 32-bit compare-and-swap loop on its word, and Triton makes the
  contiguous unmasked lanes that one thread holds of a 16-bit or `float32` add one vector `atom`,
  library issue L172, [evidence/half-atomics](../half-atomics/README.md)) (`atom.global.gpu.relaxed.add.f32`, `atom.global.sys.acquire.add.f32`,
  `atom.global.gpu.release.add.noftz.f16`, `atom.global.gpu.relaxed.add.noftz.bf16`,
  `atom.global.gpu.relaxed.add.f64`, `atom.global.gpu.relaxed.exch.b32`,
  `atom.global.relaxed.gpu.cas.b32`, integer `max`/`min` for the float ones, …), whether or not its
  result is used; none became a `red`, none was dropped, and no barrier was added except where a
  scalar atomic's result is used (its value goes through shared memory). The one exception is the
  scalar `atomic_add` of a constant `0` with `relaxed` or `acquire` order: a strong load
  (`ld.global.gpu.relaxed.b32`, `ld.global.gpu.acquire.b32`), no atomic, and `-0.0` stays `-0.0`;
  with `acq_rel` it is an `atom` and `-0.0` becomes `+0.0` (both are values `P.atomicAdd` allows:
  docs/rules/num.md, "Float `atomic_add`").
- Part A: 120 cases (12 ops and dtypes, two lane patterns, five `sem`/scope pairs), 3200 cells each
  over 50 launches: no cell wrong, with every `sem`.
- Part M: no shared cell and no private element wrong, with `relaxed` and with `acq_rel`
  (12800 cells, 26214400 elements each).
- Part L: the integer runs were exact at every configuration (428800 elements); on random data the
  largest error, relative to `scaling · Σ|a·b|`, was the same order with `relaxed` and with
  `acq_rel` (at most `1.3·10^-7`). The kernel compiles to `atom.global.gpu.relaxed.add.v4.f32`.
- Part F (`float_maxmin.out.txt`): the PTX holds the two predicated `atom`s (`max.s32` and
  `min.u32` for `float32` `max`, `min.s32` and `max.u32` for `min`, `.s64`/`.u64` for `float64`).
  In each of the four cases, of 1927 enabled lanes, 0 have a new value other than the picked
  atomic's, 0 return other than the old value, 0 are outside `fBound`, and 0 of 121 masked-off
  cells changed. Of 151 lanes with a NaN operand, 73 or 74 wrote the NaN (in `atomic_max` a NaN
  with its sign bit clear beats every non-NaN under `max.s32`, in `atomic_min` one with it set
  beats every non-NaN under `max.u32`), as
  `fBound` allows.

**Conclusions.**
1. Triton 3.8.0 gives a `relaxed`, `acquire` or `release` atomic the instruction it gives an
   `acq_rel` one, with only the order qualifier changed, and adds no other access: the device
   performs the atomic read-modify-write the model states, with the order the PTX memory model
   gives it. The rule (`MemSem.unused`) relies on that and on the PTX memory model's atomicity of
   strong operations at a common scope on the same bytes, not on these runs alone; the runs found
   no lost, doubled or torn update in 384000 contended cells and 428800 split-K outputs.
2. The rule covers pointer atomics at `gpu` or `sys` scope whose result the source never uses. It
   does not cover `cta` scope (atomic only within a program: docs/exclusions.md), atomics whose
   result is used, or descriptor reductions (`tl.desc_atomic`, TMA reductions in the async proxy,
   not tested here), which stay as before.
3. Vector forms (`.v2`, `.v4`) occur for contiguous lanes; programs with the same layout and
   alignment issue them on the same bytes, as here. A vector atomic of one program and a narrower
   one of another on part of its bytes is review-questions.md Q7, unchanged by this rule.
