# Differential testing of TVA v3

Draft for review with zhuominc (Yi Zhou, 2026-11-11). Nothing here is part of the library.

## Goal

Every one of the 41 claims has a proof (`claims/<kernel>/oracle/`), so inside the model there is
nothing left to check: every covered launch of each model is correct. What a reader of a proved
claim still trusts is listed in `TVA/Claim.lean`:

1. Lean and Mathlib;
2. **the transcription**: the model is the kernel;
3. **the compiler**: Triton compiles a direct launch so that every execution ends in one of the
   model's outcomes, with the meaning `TVA.Triton` gives each `tl.*` operation;
4. **the device facts** (`Device P`, 38 fields), measured in `evidence/`.

Items 2–4 are claims about the real GPU, so they can be tested against it. The question every
test asks is one direction of refinement:

> Is what the device did one of the outcomes the model allows?

A device behaviour outside the model's outcomes is a **soundness finding**: a proof covers a
program the device does not run. The other direction (the model allows more than the device does)
only makes proofs harder, and the proofs exist, so tightness is low priority here.

A mismatch is brought to zhuominc, not filed as a library issue.

## What exists

| Test | What it compares | What it does not see |
|---|---|---|
| `claims/<kernel>/check_gpu.py` (41) | the real kernel, launched by its own host code on the `SETUP.md` launches and on edges of `Inst.ok`, against `Post` as stated: values bit for bit or within tolerance, negative tests showing the tolerance rejects wrong outputs, inputs unchanged, sentinel buffers around outputs | the **model**: a mistranscribed model whose proved `Post` still holds on the inputs tried passes; the **well-behavedness** predictions (no invalid access, no race): a kernel can read out of bounds and still produce the right output; **where** a failure comes from |
| `evidence/num-*` (15) | the device's results for the platform operations (`exp`, `/`, `tl.dot`, fp8 conversion, float atomics, reduction trees) against the accuracy facts | the `tl.*` definitions themselves |
| `evidence/float-layer` | the library's float definitions, compiled to native code, against the device, bit for bit | operations other than scalar floats, conversions and bytes; it was run on the previous library's `TVA.Math.Defs`, whose definitions are the same text as v3's (see below) |

No run logs of `check_gpu.py` are kept in the repository; the claims' READMEs quote some results.

### Evidence that refers to the previous library

Evidence READMEs name modules that are not in v3 (`TVA.Math`, `Rules.Num`, `Sem/`). The
inventory ([TRUST.md](TRUST.md)) found this is a matter of names: v3's float definitions
(`Base/Float`), `Int`, `DType`, `Bytes`, `Literal`, `Promote`, `Random` and the hints are the same
text as the definitions the evidence ran on, so the evidence applies to v3. What is missing is
evidence itself: fifteen folders of commit `5ea2115` (`literals`, `promotion`, `cdiv-round17`,
`permute`, `int-reductions`, `itof-rtz`, `fp8-constants`, `python-numbers`, `scalar-args`, six
`hint-*`) are not in the v3 tree, though v3's definitions rest on them.

## Phases

Phases 0 and 3 need no GPU. Phases 1, 2 and 5 need a GPU but no change to the library, so they
do not wait on the decision below.

### 0. Trust inventory (no GPU): done, [TRUST.md](TRUST.md)

A table, `validation/TRUST.md`: for every `tl.*` definition in `TVA/Triton`, every field of
`Device`, and every claim, what validates it today: the Lean definition run on the device, a
Python reference, `check_gpu.py`, reading the compiler's source, or nothing. Includes the
mapping from each `Device` field to the evidence folder and the bound measured there. Its output
is the list of gaps, which orders the rest.

### 1. The float layer and the exact operations, against v3 (GPU)

The scalar float layer already has evidence (above); a rerun on v3's own code is a cheap
confirmation. The priority is what [TRUST.md](TRUST.md) lists with no device evidence. Compile
v3's definitions into a native executable (`lake exe`, in `validation/`) and run them against the
device with the `float-layer` harness (hash of each chunk of `2^20` outputs, NaN
canonicalised, full domains where they fit in `2^32`):

- `TVA/Base`: float formats, rounding, IEEE `+ - * fma div_rn sqrt_rn`, `max`/`min`,
  comparisons, conversions between float formats and between integers and floats, bytes;
- `TVA/Triton/Convert` and `Promote` (promotion also without a GPU, against
  `tl.static_assert` and TTIR);
- integer `//`, `%`, shifts (including where the model says undefined behaviour);
- `TVA/Triton/Random` (Philox): no choice in it, so bit for bit;
- the tensor operations with no choice in them: reshape, join, split, interleave, broadcasting,
  masked `load` with `other`.

Each test is checked against **mutated copies** of the definition under test, so that we know it
would catch a wrong definition.

### 2. Undefined-behaviour predictions, under `compute-sanitizer` (GPU)

Every covered launch is proved well behaved, so the model predicts: no invalid access, no race.
Run each claim's `check_gpu.py` unchanged under `compute-sanitizer --tool memcheck` and
`--tool initcheck` (and `racecheck`, which sees shared memory only, the compiler's own). Any
report on a covered launch is a finding. Add cases at the boundaries of `Inst.ok` where the
scripts do not already have them: a mask that ends exactly at the end of a buffer, a buffer of
exactly the required length, offsets at the edge of `int32`.

The cheapest phase with the most coverage: the 41 runners exist.

### 3. An executor for v3 models (no GPU)

`exec : Prog α → Mem → Oracle → Option (Res α)` runs one program on a concrete memory, and
`exec_sound` proves that whatever it reports (a final memory, or undefined behaviour), `Step`
can reach. Nothing in it is trusted. It needs a change to `Prog`, described in the next section;
this is the decision to make. A version for the previous library exists (`validation/Exec.lean`
on `validation-exec`, 223 lines, `exec_sound` proved) and ports directly.

The launch is run one program after another, in grid order. For a tile-parallel kernel this is
one of the launch's interleavings; for kernels with atomics the order is the oracle's.

### 4. Each kernel's model against the device (GPU, needs phase 3)

For each claim, on small inputs taken from its `check_gpu.py`: run the model through `exec`, run
the kernel on the device, compare the final memories bit for bit. This tests the transcription
and the `tl.*` semantics together, on the actual kernels.

The model's choices are made so that the comparison is exact:

- platform operations (`exp`, `/`, `tl.dot`, fp8 conversion, float atomics): a **platform
  calibrated on the device**, `P_dev`, a table of the device's results, measured by separate
  kernels of one operation on the operands the run needs (two passes: record the queries, measure
  them, run again). Unary operations on 16-bit formats are tabulated whole once;
- free choices (reduction trees, lane order, disabled lanes' values): inputs for which every
  choice gives the same result (floats with small integer values make sums and `tl.dot` exact in
  any order; no NaNs), or `enable_fp_fusion=False` where fusion is the only freedom.

Order: the kernels with no platform operation and no float reduction first (`vec_add_f32`,
`kv_indices_i32`, `kv_cache_write_bf16`, `rejection_greedy_i32`, `vocab_embedding_bf16`,
`seeded_dropout_f32`, `count_expert_tokens_i32`), then the float kernels.

**Mutation checks on transcriptions.** Seed typical transcription errors into copies of a model
(an off-by-one mask, a swapped stride, a missing conversion, the wrong axis of a reduction) and
check that phase 4 catches each. This measures how much the tests say about item 2 of the
trusted base, which today is checked by reading.

A mismatch in phase 4 that is not a bug: `P_dev` assumes the device computes an operation inside
the kernel as it does alone. If, for example, fusion inside the kernel changes `tl.exp`'s result,
the test fails while the model (a relation per element) allows it. Such a mismatch is localised
by instrumenting the kernel (extra stores), which is fine for localising and not for testing:
instrumenting changes what Triton compiles.

### 5. Atomics across programs (GPU)

Several claims use atomics (`streamk_matmul_f16`, `lora_shrink_bf16`, `route_histogram_i32`,
`count_expert_tokens_i32`). Litmus tests of what `TVA.Launch` allows (message passing, store
buffering, coherence on one cell), and of the argument of `MemSem.unused`: an atomic whose result
is unused behaves as `acq_rel` at `gpu` scope, whatever its `sem`.

## The change to `Prog` that phase 3 needs

### The problem

`Prog.pick P k` stores the predicate `P : β → Prop` and the continuation `k`. To run it, an
executor must produce some `b` with `P b` and continue with `k b`. In compiled code `P` is
erased (it is a proposition), and so is `β`: at run time the executor sees a `pick` node and a
closure, and cannot tell an `upToNaN` from a reduction tree or a bounds guard. It can neither
build a witness nor check one, and an oracle cannot answer a question it cannot see.

Recording the values from the device instead does not work: most `pick`s have no value on the
device (a guard picks `()`; the tree of a reduction is the shape of the computation, not a value;
the order in which a store's lanes land), recording a value that exists changes what Triton
compiles, and a recorded bit pattern still has to be given to `k` at an erased type.

### The proposal

Each `pick` carries a solver, which the semantics ignores:

```lean
| pick (P : β → Prop) (solve : Solve P) (k : β → Prog α) : Prog α

/-- A certified answer to "some `b` with `P b`". -/
inductive Ans (P : β → Prop) where
  | ok    (b : β) (h : P b)     -- a witness, with its proof
  | empty (h : ∀ b, ¬ P b)      -- no witness: undefined behaviour

structure Solve (P : β → Prop) where
  run : Hints → Option (Ans P)  -- `none`: no answer, the executor gives up
```

The solver is written next to `P`, by whoever writes the `pick`, in `TVA/Triton` (no claim
contains a `pick`). At each site it either computes the witness or asks the oracle a typed
question from a closed family that survives compilation, and returns the answer with its proof:

| Kind of site (examples) | Solver |
|---|---|
| guards: block pointers, descriptors, `tl.assume`, `static_assert`, conversions' conditions | decides the condition; `empty` is a predicted undefined behaviour |
| compiler hints: `multiple_of`, `max_contiguous`, `max_constancy` | the witness is the value itself, if the hint's condition holds |
| partial functions: integer `//`, `%`, shifts | the only candidate, or `empty` |
| `upToNaN`, IEEE operations, `maximum`/`minimum`, `addc`/`mul_add` without fusion | the IEEE result |
| free choices: reduction and scan trees, `sort`, `histogram`, `reshape(can_reorder=True)`, disabled lanes, padding | asks the oracle (`Query.tree n`, `Query.perm s`, `Query.val d`, …), then evaluates `P` on the answer |

Platform operations need one more constructor. They are written
`env fun e => pick (fun r => e.P.math .exp f x r) k`: `P` mentions the platform `e.P`, which is a
variable, so no solver can produce or check a result. The proposal replaces this pattern by
`plat q k`, where `q : PlatQ` is the operation on its operands (`math`, `div`, `cvt` into fp8,
`dot`, `dotScaled`, float `atomicAdd`); `Step.plat` takes any result `q.Rel E.P` allows, which is
what the `env`-then-`pick` meant. The executor's oracle is for one fixed platform (`P_dev`), and
answers `q` with a proof of `q.Rel P_dev`. `rmw` needs the same for its `upd` relation.

### What it costs and what it does not change

- **Meaning: unchanged.** `Step.pick` ignores `solve`; `Step.plat` is the old pattern's meaning.
  A reader of a claim can ignore solvers: a wrong solver can only make the executor give up,
  because every answer carries its proof, so `exec_sound` holds for every solver and every
  oracle.
- **Claims: no statement changes.** No claim file contains `pick`. Whether the claims' sha256
  change depends only on whether the claim files change, which they need not.
- **Proofs.** `Safe` and its rules gain a `plat` case; the rules for `math`, `div`, `cvt8`, `dot`,
  `dot_scaled` keep their statements and change their proofs. Every case analysis on `Step` gains
  `plat`/`platUB`. The oracle proofs need checking; on the previous library no kernel proof
  mentioned `pick`.
- **Size.** 68 `pick` sites in `TVA/Triton`, each given a solver (most are one word:
  `Solve.guard`, `Solve.ok v rfl`); about a dozen platform sites (in `Arith`, `Convert`, `Dot`, `Memory`) become `plat`.

### Alternatives considered

- **A copy of each model that the executor can run**, in `validation/`, with a proof that it
  equals the original. No library change, but 41 copies to keep in sync and to prove equal.
- **An executor over Lean's syntax trees** (`Expr`), recognising each `pick` by the shape of
  its `P`. No library change, but slow (models build tensors of thousands of elements) and tied
  to how each definition happens to be written.
- **No executor**: phases 0–2 and 5 only. The transcriptions stay checked by reading and by
  `check_gpu.py`.

## To decide with zhuominc

1. The change to `Prog` (solvers and `plat`), or one of the alternatives.
2. The evidence that names the previous library: is redoing `float-layer` for v3 (phase 1)
   wanted, and are the `num-*` folders' bounds the v3 facts' bounds?
3. `check_gpu.py` results: are runs recorded somewhere (date, driver, claim sha256)? Should
   they be?
4. GPU: the H200 NVL the evidence ran on, or an AWS P5e (H200); do results from an H100 count?
   Phases 1, 2, 4 and 5 need one, with `compute-sanitizer` and a driver like the evidence's
   (610.57).
5. Where the work lives: a `validation/` folder on a branch off `v3`, not merged into the library
   until reviewed.
