# Library batch, round 11: author's notes

Issues: L5, L37, L57, L58, L59, L60, L61, L71, L72, L73, L74, L75 ([ISSUES.md](../../ISSUES.md)),
2026-10-28. The semantics and rules batch, run beside rounds 12 (TVA's float layer) and 13 (tool and
gate) in a separate tree; the coordinator added L75 after the round started. `lake build TVA` is
green, and the new and changed theorems depend only on `propext`, `Classical.choice` and
`Quot.sound`. No float definition, numeric lemma or code of `lean/TVA/Lang/Scalar.lean` changed.

**Kernel that does not build or pass the gate:** `kernels/vec_add_f32`, as before this round
(expected; its prover re-proves it after these rounds). L71 adds a theorem every claim must prove
(`satisfiable : Claim.Satisfiable Inst.ok Pre`), so the kernel needs it too. SEMANTICS.md and
exclusions.md changed, so every item of every kernel that reaches the library re-opens. I did not
edit the kernel folder or run `tvr check`.

**Files of other rounds** (small changes, for the coordinator's merge): `docs/PROVER_GUIDE.md` (L57,
L61, L71), `docs/REVIEWER_GUIDE.md` (L72, L74: one sentence), `docs/REVIEW_PROTOCOL.md` §2 and §8
(L71), `docs/DESIGN.md` "No vacuous claims" (L71), `docs/prompts/kernel-prover.md` (L71: the theorems to
write), `tools/claim_reader.lean.in` and `tools/tvr.py`
(L71: the `shape` check and claim.md's claim), `docs/LIBRARY.md` (the new constants). New evidence
folders: `evidence/desc-zero-shape`, `hint-contig`, `hint-multi`, `python-numbers`, `if-names`,
`py-builtins`, `compiled-options`, `launch-frame`, `claim-satisfiable`; round 8's
`evidence/hint-loops` and `evidence/hint-rank2` READMEs have notes on the new hint readings (L58).

## L5: A descriptor built inside the kernel writes launch scratch memory

- **What changed.** `TensorDesc.Valid` (`lean/TVA/Lang/Desc.lean`) asks `0 < shape[k]` for every
  dimension (it asked only `< 2^31`), so an access (`desc_load`, `desc_store`, `desc_atomic`,
  `desc_gather`, `desc_scatter`, all through `Access` or `RowAccess`) through a descriptor with a
  zero shape entry is undefined behaviour. `tl.make_tensor_descriptor` still accepts a zero entry (it
  is pure, and building one is harmless). SEMANTICS "Tensor descriptors" says so, and adds that on
  the H200 `desc.gather` and `desc.scatter` do not compile at all (ptxas: `.tile::gather4` needs
  sm_100, `.tile::scatter4` is not on sm_90a), so a launch that uses one is rejected on the host and
  has no outcome; the module docstring and the docstrings of `make_tensor_descriptor`, `desc_gather`
  and `desc_scatter` say the same. The descriptor lemmas (`TVA.Lemmas.Desc`) build unchanged.
- **Why.** lr-8a: every load or store through an in-kernel descriptor with a zero dimension faults
  ("an illegal instruction was encountered"), while the model gave the padding or wrote nothing, so
  `Faithful`'s "never faults" was false for a well-behaved model.
- **Evidence.** [evidence/desc-zero-shape](../../../../evidence/desc-zero-shape/README.md): Z2, Z3,
  Z5-Z7, Z10, Z11 (load, store and reduction, either dimension zero, rank 1 and 2, any offsets)
  fault; Z1 and Z4 (`[1, 32]`) are as the model; Z12 (built, never accessed) is as the model; Z13 (a
  host-built one) is rejected on the host; Z8, Z9, Z14-Z16 (gather and scatter, in-kernel and
  host-built, any shape) are rejected by ptxas.
- **Constants whose meaning changed.** `TVA.TensorDesc.Valid`, and through it
  `TVA.TensorDesc.Access`, `TVA.TensorDesc.RowAccess`, `TVA.tl.desc_load`, `TVA.tl.desc_store`,
  `TVA.tl.desc_atomic`, `TVA.tl.desc_gather`, `TVA.tl.desc_scatter`; the docstrings of
  `TVA.tl.make_tensor_descriptor`, `TVA.tl.desc_gather` and `TVA.tl.desc_scatter`; the module
  docstring of `TVA.Lang.Desc`.

## L37: Cover every pinned tutorial kernel, except tutorial 05's `_layer_norm_bwd_dx_fused` and tutorial 11

- **What changed.** exclusions.md records the user's two decisions of 2026-10-28, citing them (L37):
  a row excluding tutorial 08's `grouped_matmul_kernel` (its `[16, 16]` hints are false by Triton's
  documentation and by the library, so it has no meaning on any launch; `grouped_matmul_tma_kernel`
  stays in scope), and a row covering tutorial 09's pointer kernels (`matmul_kernel`,
  `matmul_kernel_persistent`) on full tiles only (`Inst.ok` requires `M` and `N` to be multiples of
  the block sizes of every config the autotuner may pick). SEMANTICS "Compiler hints" and
  TRANSCRIPTION's hint pitfall point to the two rows.
- **Why.** lr-8b's reopen: tutorial 08's pointer kernel had no claim on any launch, which needed the
  user's decision; the user decided both cases.
- **Evidence.** The decisions in ISSUES.md (L37, "User decisions, 2026-10-28"); evidence/hint-rank2
  R1, R2 (tutorial 08's hint false) and T2 (tutorial 09 on a partial tile).
- **Constants whose meaning changed.** None (exclusions.md, SEMANTICS.md and TRANSCRIPTION.md).

## L57: PROVER_GUIDE still builds a comparison's constant in the operation's dtype

- **What changed.** `docs/PROVER_GUIDE.md` ("Write each Python number as the constant Triton
  builds") now says "with an arithmetic or bitwise operator, as a value of `tl.where`, and in
  `tl.full`, in the operation's dtype; elsewhere (a comparison, `other=`, a stored value, another
  `tl.*` function) as a tensor of its own type", as SEMANTICS "Constants" does, and points to L59's
  fact. Only that bullet changed.
- **Why.** The guide's "with an operator" included comparisons, for which Triton first makes the
  number a tensor of its own type (evidence/promotion K1, K3), so a prover following it wrote a model
  that differs from the device.
- **Evidence.** evidence/promotion K1-K4 (unchanged).
- **Constants whose meaning changed.** None (PROVER_GUIDE.md, round 13's file).

## L58: Two hint readings are stronger than Triton relies on

- **What changed.** SEMANTICS "Compiler hints" ("What it promises") and the hint section of
  `TVA.Lang.Ops`:
  - `tl.multiple_of(x, v)` alone is `FirstsDiv x k C v` with `C` a lower bound, as the library
    states it, of the contiguity Triton's analysis infers for `x`, and `1` (every element) where it
    states none. The hint replaces only the divisibility, which the compiler pairs with its inferred
    contiguity (`AxisInfo.cpp:1244-1259`); a smaller `C` is a stronger promise, so a lower bound
    covers the compiler's belief.
  - A new library fact, "The contiguity the analysis infers", per dimension: `tl.arange(a, b)` has
    `b - a`; `x + y`, `y + x` and `x - y` with `y` constant along the dimension (a scalar, a Python
    number, a size-1 dimension that is broadcast) keep `x`'s bound, for pointers too; `x[:, None]`,
    `tl.expand_dims` and broadcasts keep the other dimensions' bounds and give `1` to the new or
    broadcast one; integer conversions keep it; anything else has `1` (`AxisInfo.cpp:138-147,
    169-183, 242-279, 553-575, 607-668`).
  - The pair `tl.max_contiguous(tl.multiple_of(x, v), C)` is the new
    `tl.max_contiguous_multiple_of x C v` (and `…N`): `ContigAlong x 0 C ∧ FirstsDiv x 0 C v`, with
    `C` below the value's own contiguity allowed; `ContigIs` stays for `max_contiguous` alone. New
    lemmas `safe_max_contiguous_multiple_of` and `safe_max_contiguous_multiple_ofN`.

  The docstrings of `multiple_of`, `multiple_ofN`, `max_contiguous` and `max_contiguousN` say what
  `contig` may be and point to the pair; their definitions are unchanged (a model that wrote the pair
  as `multiple_of … (contig := C)` and `max_contiguous` still means what it did, a stronger check).
  TRANSCRIPTION's hint rows and pitfall use the new forms. evidence/hint-loops' column now says that
  it reads the hints as `FirstsDiv p 0 128 16`, and evidence/hint-rank2's conclusion 2 notes the
  refinement.
- **Why.** lr-8a, lr-8b: the "every element" reading made the common 1-D idiom false on every
  launch, and `ContigIs` for the pair excluded correct launches (lr-8a's W1, W3-W5, W7).
- **Evidence.** [evidence/hint-contig](../../../../evidence/hint-contig/README.md): for each listed
  shape (an arange plus or minus an argument or a loaded scalar, on either side; a pointer plus it;
  its `int64` conversion; a 2-D sum of a computed or loaded column and a row) a hint true under the
  new reading vectorised the loads and was correct, and one false under it faulted (H1-H16, against
  scalar loads without a hint, H20, H21); the pair with `C = 4` and `8`, below the contiguity 128, is
  correct when true and faults when false (H17-H19).
- **Constants whose meaning changed.** New: `TVA.tl.max_contiguous_multiple_of`,
  `TVA.tl.max_contiguous_multiple_ofN`, `TVA.safe_max_contiguous_multiple_of`,
  `TVA.safe_max_contiguous_multiple_ofN`. Docstrings: `TVA.tl.multiple_of`, `TVA.tl.multiple_ofN`,
  `TVA.tl.max_contiguous`, `TVA.tl.max_contiguousN`, and the hint section of
  `lean/TVA/Lang/Ops.lean`.

## L59: A name assigned a Python number holds a scalar tensor of its own dtype

- **What changed.** SEMANTICS "How the front end evaluates the source" has a new fact: a name that a
  statement assigns a Python `int`, `float` or `bool` (`=`, augmented assignments, `:=`, tuple
  targets; a literal, a `tl.constexpr`'s value, a compile-time expression) holds a scalar tensor of
  the number's own type, with the constant Triton builds for it, unless the target is annotated
  `: tl.constexpr`; `None` and dtypes are kept, and other values are rejected; a number passed
  directly to a helper stays a compile-time value ("Helpers are inlined"). "Order" no longer counts
  such a name among compile-time values, and "Constants" and "Operators and promotion" ("Python
  numbers") say that it is a tensor operand. The `Operand` docstring and the module docstring of
  `TVA.Lang.Promote` say the same. TRANSCRIPTION has two rows (`e = 0.1; x * e`, `i = 0`) and a
  pitfall.
- **Why.** lr-8b: the rules read as if `e = 1e-7`, `F = False` or `a = -7` still held Python
  values, and models that follow them differ from the device; tutorials 05, 06, 08 and 10 initialise
  loop variables from literals.
- **Evidence.** [evidence/python-numbers](../../../../evidence/python-numbers/README.md) A1-A14
  (`code_generator.py:727-760, 1358-1365`, `semantic.py:118-158, 601-613`): `e = 1e-7; x * e` is
  `float32(1e-7)·x` where the literal in place gives `0`; `a // b` after `a = -7` is `-3`, also for a
  tuple, an augmented assignment and a name assigned in `tl.static_range`; `F = False` runs the
  atomic in 8 of 8 programs and makes `if F: tl.static_assert(False)` reject the kernel; `B = BLOCK`
  makes `tl.arange(0, B)` rejected; an annotated target and a helper argument stay compile-time
  (`-4`).
- **Constants whose meaning changed.** The docstrings of `TVA.Operand` and of the module
  `TVA.Lang.Promote` (no definition).

## L60: A hint on one result of an operation with several results applies to every result

- **What changed.** SEMANTICS "Compiler hints: which value" and the `TVA.Lang.Ops` hint section:
  when the operation that defined the hinted value has several results, the hint promises the same
  about each result, and the model checks each, where the operation defines them. The operations
  are `tl.split`, `tl.reduce` and `tl.associative_scan` on a tuple, a `noinline=True` helper that
  returns a tuple, and a loop or a runtime `if` (a hint after it on a name it carries or merges).
  Also new: a hint on the value an inlined call returns (a helper, or a `tl.*` function written in
  Triton, listed: `tl.sum`, `tl.max` with or without `return_indices`, `tl.argmax`, …) has no
  effect, since the mark is on the call and inlining removes it, and the model drops it
  (TRANSCRIPTION's `no-op:hint`). TRANSCRIPTION has a `tl.split` row and the pitfall.
- **Why.** lr-8b: with `a, b = tl.split(z)`, a true hint on `a` made the loads through `b` vectorise
  and fault, while the prescribed model (a check on `a`) is well behaved. Of the issue's list,
  `tl.max(…, return_indices=True)` is a call that Triton inlines, so a hint on its results has no
  effect at all (M3); it falls under the inlined-call fact instead. Hints on loop and `if` results
  showed no effect (M7-M10), but a mark on those operations would be read for every result
  wherever the analysis took their entry state (`AxisInfo.cpp:1318-1342`), so they stay under
  "check each", which is sound either way.
- **Evidence.** [evidence/hint-multi](../../../../evidence/hint-multi/README.md): a true hint on one
  result made the loads through another fault for a tuple reduce (M4), a tuple scan (M5), a
  `noinline` helper (M6) and a `tl.split` of a computed value (M14); for a `tl.split` of a loaded
  block the mark was lost in the conversion to TTGIR (M1), so whether it applies depends on the
  passes; false hints on inlined calls' results (M12, M13) and on loop and `if` results (M8, M10)
  had no effect.
- **Constants whose meaning changed.** None by definition: the hint section of
  `lean/TVA/Lang/Ops.lean` (a module comment, hashed with the file) and SEMANTICS.md.

## L61: Wording: index arithmetic, one-branch names, builtin `min`/`max`, hooks and arguments, masked lanes, PROVER_GUIDE's hints

- **What changed.**
  - SEMANTICS "Index arithmetic": arithmetic stays `int32` "unless an operand promotes it (a cast,
    or an operand of a wider or unsigned type, such as a runtime `int64` or `uint64` scalar)", and
    an `int32` argument is "in `[-2^31, 2^31)`".
  - SEMANTICS "A runtime `if`": a name defined before the `if` keeps its dtype and shape in every
    branch that assigns it (both included), a new name used after it is defined in both with one
    type, and a new name that only one branch defines belongs to that branch (usable there; a use
    after the `if` is rejected).
  - SEMANTICS "Operators and promotion": Python's `min` and `max` of compile-time values reject NaN
    and `-0.0` (the kernel is rejected on the host); with a tensor argument they fold left to right.
  - SEMANTICS "One launch", `Faithful`'s "Which launch" and the rule page: the claimed launch has
    the arguments the last hook leaves; TRANSCRIPTION's host pitfall too.
  - TRANSCRIPTION "Masked-off lanes": "not necessarily `0`".
  - PROVER_GUIDE: the hint sentence no longer calls L1 and L2 open: a hint on a parameter, an
    induction variable or an inlined call's result is dropped, and one on a carried name checks the
    value before the loop.
- **Why.** lr-8a, lr-8b: each text said more, or other, than is true.
- **Evidence.** [evidence/if-names](../../../../evidence/if-names/README.md) N1-N6 (a one-branch name
  used only there runs, used after the `if` is rejected; a live-in retyped in both branches is
  rejected); [evidence/python-numbers](../../../../evidence/python-numbers/README.md) B1-B4
  (`max(1.0, nan)` and `min(1.0, -0.0)` rejected; `core.py:3882-3912`); for the hooks,
  `tutorials/06-fused-attention.py:112-124` (`_host_descriptor_pre_hook`) and
  `tutorials/09-persistent-matmul.py:183-194` (`matmul_tma_set_block_size_hook`) set the
  descriptors' `block_shape`.
- **Constants whose meaning changed.** The docstring of `TVA.Rules.Compiler.Faithful` ("Which
  launch").

## L71: The gate passes a claim whose conditions cover no instance

- **What changed.** New `TVA.Claim.Satisfiable ok Pre := ∃ d init, ok d ∧ Pre d init`. A claim
  proves `theorem satisfiable : Claim.Satisfiable Inst.ok Pre`: the claim reader
  (`tools/claim_reader.lean.in`) elaborates it beside `main` and `consistent` and reports its
  axioms; `tools/tvr.py`'s `shape` detail and claim.md's "The claim" name it, and the `axioms` check
  covers it (it checks every axiom list the reader prints). `TVA.Claim`'s module docstring explains
  why `Consistent` and `Satisfiable` together make a claim non-vacuous, and that coverage of real
  use stays the reviewer's judgment. Protocol §2 and §8, DESIGN.md "No vacuous claims", the rule
  page, PROVER_GUIDE and the kernel prover's prompt mention the theorem.
- **Why.** lr-10a: `Consistent` bounds only the assumptions; `Inst.ok _ := False` with a real
  `consistent` passed every check, and an unsatisfiable combination of conditions can be invisible
  to a reader. The mechanical check is complete for what it checks (one covered instance and
  memory) and costs a prover one witness.
- **Evidence.** [evidence/claim-satisfiable](../../../../evidence/claim-satisfiable/README.md): the
  reader, filled in by `tvr`'s own `claim_reader_text`, accepts a claim with a `satisfiable` proof
  (S1, no axioms) and does not elaborate one with `Inst.ok _ := False` and none (S2).
- **Constants whose meaning changed.** New `TVA.Claim.Satisfiable`; the module docstring of
  `TVA.Claim`; the claim's shape (every kernel must add `satisfiable`).

## L72: Faithful's scope 4 misses options given at `warmup` or `triton.compile`

- **What changed.** Scope 4 (`Faithful`'s docstring, the rule page, SEMANTICS "Launches",
  exclusions.md's options row): no option other than the eight is given "to the call that compiles
  the kernel (a JIT launch, `kernel.warmup(…)`, which takes the same keywords, or `triton.compile`'s
  `options`) or in an autotune `triton.Config`". The rule page's reviewer check, TRANSCRIPTION's
  host pitfall and REVIEWER_GUIDE's host check (one sentence) name `warmup` and `triton.compile`.
- **Why.** lr-10b: `k.warmup(…, enable_reflect_ftz=False)` and then a launch with no keyword changed
  `tl.math.exp2(-130)`, with scope 4 as written satisfied.
- **Evidence.** [evidence/compiled-options](../../../../evidence/compiled-options/README.md) O1-O4:
  `enable_reflect_ftz=False` at a JIT launch, at `warmup` and in `triton.compile`'s options each
  gave `2^-130`, against the default's `0` (O3).
- **Constants whose meaning changed.** The docstring of `TVA.Rules.Compiler.Faithful` (scope 4).

## L73: No claim can say the launch leaves memory outside `init` alone

- **What changed.** `Faithful` (its docstring's "Memory", the rule page's "What it says" and
  "Memory") states the frame: a valid launch within the scope, from an `init` from which it is well
  behaved, writes no byte of global memory other than those `init` gives, except the scratch of the
  descriptors the kernel builds (scope 5). `TVA.Claim` has a bullet "Other memory": a claim's `main`
  can only be proved when the launch is valid and well behaved on its covered instances (no other
  assumption constrains `K`), so a claim citing the rule implies, through the rule's text, that the
  launch leaves every other byte as it was; `Post` cannot state it, since `ok fin` holds only
  `init`'s bytes. SEMANTICS "Launches: Memory" says the same.
- **Why.** lr-10b: the round-8 "Memory" paragraph put what the device does to other bytes outside
  the outcome, so no certificate excluded a launch that writes past its buffers, which a host
  composing launches needs; trusting the compiler implies the frame. A formal frame would need
  outcomes over the whole memory (a change to `Behaviour`, `Faithful` and every claim), so the frame
  is part of the rule's trusted text.
- **Evidence.** [evidence/launch-frame](../../../../evidence/launch-frame/README.md): six kernels
  (a masked tile, a row softmax, a pipelined `tl.dot`, atomics, sort, scan and histogram, in-kernel
  descriptors), every buffer in one sentinel-filled pool, changed no byte outside their outputs (and
  K6's scratch), with two sentinels; `global_scratch_size` was `0` but for descriptors, and
  `profile_scratch_size` `0` (`nvidia-driver.py:280-333`: the launcher's only own buffers, with instrumentation's, which the scope excludes).
- **Constants whose meaning changed.** The docstring of `TVA.Rules.Compiler.Faithful`; the module
  docstring of `TVA.Claim`.

## L74: Wording: Faithful names no target, its grid sentence, scope 6's host descriptors

- **What changed.**
  - A scope point 7, "The target": the launch runs on an NVIDIA H200 (sm_90), with code Triton
    compiles for it (no `arch`, no `TRITON_OVERRIDE_ARCH`, no other `triton.compile` target); the
    model's defaults and the device facts (`Rules.Num.Sm90`) are the H200's. `Faithful`'s
    docstring, the rule page ("seven conditions"), SEMANTICS "Launches", `TVA.Claim` (its module and
    `Behaviour` docstrings) and exclusions.md's targets row name it.
  - `Faithful`'s grid sentence: `2^31` to `2^32` programs are skipped without an error, and a larger
    grid may run (`2^32 + 1` programs wrap to one), as protocol §6 says.
  - Scope 6 names host tensor descriptors (the warm-up's dtype and block shape, their
    specialisation, `specialize.cc:165-226`, since the launcher builds the TMA descriptor with the
    compiled block shape, `nvidia-driver.py:209-264`), and `triton.compile`'s signature and
    attributes as its specialisation.
- **Why.** lr-10b: a claim that cites only `Faithful` read as one for any GPU; the reason given for
  `2^31` was stale; host descriptors were missing from the argument classes.
- **Evidence.** [evidence/compiled-options](../../../../evidence/compiled-options/README.md) O5 (a
  `triton.compile` for sm_80 is rejected on the H200), O6 and O7 (a `[16, 16]` descriptor given to a
  kernel compiled for `[32, 32]` had 1024 elements copied; the JIT recompiles and copies 256).
- **Constants whose meaning changed.** The docstrings of `TVA.Rules.Compiler.Faithful` and
  `TVA.Claim.Behaviour`; the module docstring of `TVA.Claim`.

## L75: SEMANTICS never says what a call of one of Python's builtins in a kernel gives

- **What changed.** SEMANTICS "How the front end evaluates the source" has a new fact, "Python's
  builtins": a kernel sees only `len`, `list`, `range`, `float`, `int`, `isinstance`, `getattr`,
  `hasattr` (Python's) and `print`, `min`, `max` (Triton's); any other builtin is rejected on the
  host. A call of `float`, `int`, `isinstance`, `len`, `list`, `getattr` or `hasattr` is evaluated
  by Python at compile time on its arguments' values, after they are evaluated as usual, emits no
  operation, and gives a compile-time value; the fact works through the tutorials' cases
  (`-float('inf')` as `other=`, `tl.zeros(…) - float("inf")`, `isinstance(…, tl.tensor_descriptor)`
  choosing a branch). "Order" counts such a result among compile-time values, "Expressions with no
  effect" lists such calls, and "Constants" counts a builtin's number as a Python number.
  TRANSCRIPTION has two rows and a pitfall, and its `config` reason names an `isinstance` condition.
- **Why.** The round-10 coverage audit: tutorials 02 and 06 need these facts, and SEMANTICS counted
  every call as possibly effectful.
- **Evidence.** [evidence/py-builtins](../../../../evidence/py-builtins/README.md) F1-F10
  (`code_generator.py:357-365, 381-427, 1444-1448, 1638-1659`): `-float('inf')`, stored and as
  `other=`, is `0xff800000`, and so is `tl.zeros(…) - float("inf")`; `float`, `int`, `len`, `list`,
  `getattr` and `hasattr` in `tl.static_assert` compile; `isinstance` tells a host descriptor from a
  pointer, and only the branch it picks is emitted; an atomic inside `isinstance(…)` ran in 8 of 8
  programs; `float` and `int` of a runtime value, and `abs`, are rejected on the host.
- **Constants whose meaning changed.** None (SEMANTICS.md and TRANSCRIPTION.md).
