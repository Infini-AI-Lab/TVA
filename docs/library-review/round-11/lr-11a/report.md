# Library review, round 11, by lr-11a

## Verdicts

- **L5** ACCEPT — `TensorDesc.Valid` now asks `0 < shape[k]`, every access goes through `Access` or
  `RowAccess` (both include `Valid`), and `tl.make_tensor_descriptor` stays pure. evidence/desc-zero-shape
  re-ran identical: Z2-Z11 fault, Z1, Z4 and Z12 are as the model, Z13 is rejected on the host, and
  ptxas rejects gather and scatter. On sm_90 descriptors always go through the TMA lowering (the
  rewrite to pointers is only for capability below 9, `backends/nvidia/compiler.py:249-250`), so every
  gather or scatter needs `.tile::gather4`/`.tile::scatter4`. My own tests tried other in-kernel
  descriptors that `Valid` admits (`work/desc_edges.py`): a leading stride of 0, a stride of 2^40
  bytes, overlapping rows, a 12-byte innermost extent, a 512-row block, a 2^31 - 1 extent, and rank 3
  with a stride of 0. All eight were as the model, so I found no other admitted value that faults.
- **L37** ACCEPT — exclusions.md rows 46 and 47 record both decisions of 2026-10-28 and cite L37, and
  SEMANTICS "Compiler hints" points to them. On a full tile, tutorial 09's `tl.where(offs_am < M,
  offs_am, 0)` is the identity, so `offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)` with
  `BLOCK_SIZE_M | start_m`, and the pair reading holds (`09-persistent-matmul.py:130-136, 313-321`).
- **L57** ACCEPT — PROVER_GUIDE's bullet now matches SEMANTICS "Constants": the operation's dtype for
  arithmetic or bitwise operators, `tl.where` values and `tl.full`, and a tensor of its own type for a
  comparison, `other=`, a stored value or another `tl.*` function.
- **L58** REOPEN — The readings themselves are right. `multiple_of` alone is `FirstsDiv` with a lower
  bound of the analysis' contiguity, and the pair is `ContigAlong ∧ FirstsDiv`, also below the
  value's own contiguity. I checked them against `AxisInfo.cpp:1232-1259` and the visitors at
  169-183, 242-279 and 553-668, and evidence/hint-contig re-ran identical. But the new fact "the
  contiguity the analysis infers" is not a lower bound when a sub-expression carries a contiguity
  hint. The pair `tl.max_contiguous(tl.multiple_of(a, v), C)` may set `C` below `a`'s own contiguity,
  as the library allows. The analysis then gives `a` contiguity `C`, and `a + s` inherits it (the gcd
  at :265-279). The list instead gives `tl.arange(0, B) + s` the bound `B`. In
  `work/hint_bound_subexpr.py` (int8 loads of 512 elements, one warp), `y = a + s; y =
  tl.multiple_of(y, 16)` loads 16 bytes per access without the inner pair (B1) and 4 bytes with it
  (B2). So the compiler believes that 16 divides `y[4j]` for every `j`, while the model (`contig :=
  512`) checks only `y[0]`, and the argument "a lower bound covers the compiler's belief" fails. I
  found no wrong result, because the vectoriser takes min(divisibility, contiguity). Fix: say that a
  value carrying a `max_contiguous` (alone or in the pair) has the hinted `C` as its bound along each
  dimension, whatever operation defines it, and that the list applies to values without one.
- **L59** ACCEPT — `visit_Assign`/`_sanitize_value` (`code_generator.py:727-760`), `to_tensor_type`
  and `scalar_constant` (`semantic.py:118-158, 601-613`) are as stated: augmented assignments and `:=`
  go through `visit_Assign`, tuples are converted element by element, `None` and dtypes are kept, an
  annotated `: tl.constexpr` target and a helper's argument (`normalize_value`, defaults included,
  :1358-1365) stay compile-time, and a zero is `+0`. The `Operand` docstring agrees.
  evidence/python-numbers re-ran identical.
- **L60** ACCEPT — `tt.split`, `tt.reduce`, `tt.scan` and `tt.call` have no visitor
  (`AxisInfo.cpp:1179-1213`). An empty result goes to `setAllToEntryStates`, and the pessimistic
  state reads the defining operation's hint for every result (:1318-1342). `set_attr` on a function's
  block argument sets nothing (`ir.cc:411-425`). The inlined-call list matches the `@jit` functions of
  `standard.py`. evidence/hint-multi re-ran identical. Checking each result is sound where the mark is
  lost or never read. I filed one wording follow-up below.
- **L61** REOPEN — Items 1 and 3-6 are fixed. The rewritten "A runtime `if`" text (item 2) is false
  for a name whose value before the `if` is not a Triton value. `visit_then_else_blocks` skips such a
  live-in (`code_generator.py:858-861`; `None` and dtypes are not `base_value`s, and a `tl.constexpr`
  is one, `core.py:210`), and `enter_sub_region` restores the scope (:136-140).
  `work/if_compile_time_livein.py`, with `n = 2`, shows three cases. After `y = None; if n > 0: y =
  tl.load(p)`, `y` is still `None` after the `if` (I2: the `else` of `if y is not None` ran). After
  `d = tl.float16; if n > 0: d = tl.float32`, `d` is still `float16` (I6: `1 + 2^-12` was stored as
  `1.0`). And a pre-`if` `None` that both branches assign merges and runs (I3), where the text says
  the kernel is rejected. A model that follows the text merges these names and differs from the
  device, so this part is soundness-level and should not wait as wording. Fix: say that a name
  holding `None` or a dtype before a runtime `if` merges only when both branches define it (one type
  in both, as a new name), and otherwise keeps its value from before the `if`. A `tl.constexpr` is
  rejected as stated (I1, I4).
- **L71** ACCEPT — `Claim.Satisfiable ok Pre := ∃ d init, ok d ∧ Pre d init` is complete for what it
  checks. Only `Faithful` constrains `K`, so a behaviour that faults wherever the launch is not
  valid, `debug`-free and well behaved satisfies the assumptions. `main` therefore forces those
  conditions on the satisfiable instance, and with `consistent`'s total witness that instance has an
  `ok` outcome meeting `Post`. The reader elaborates `example : TVA.Claim.Satisfiable NS.Inst.ok
  NS.Pre := NS.satisfiable` (`claim_reader.lean.in:20`) and collects its axioms (:284).
  evidence/claim-satisfiable re-ran identical. `TVA.Claim`, DESIGN.md, the rule page and
  REVIEWER_GUIDE §2 leave coverage of real use to the reviewer.
- **L72** ACCEPT — Scope 4 (docstring, rule page, SEMANTICS, exclusions row 39) names the JIT launch,
  `warmup` (`jit.py:362-363`), `triton.compile`'s `options` (`compiler.py:226-243`) and a config.
  evidence/compiled-options O1-O4 re-ran identical. I filed a related gap below: `triton.compile` of
  an IR file.
- **L73** ACCEPT — The frame is in `Faithful`'s hashed docstring and on its page. `TVA.Claim` ("Other
  memory") argues correctly that `main` forces validity and good behaviour on the covered instances,
  and says that `Post` cannot state the frame. A prose frame is enough, since the rest of the scope
  is prose in the same trusted text. My sweep (`work/scratch_sweep.py`) found `global_scratch_size`
  `0` for `tl.sort`, `tl.topk`, `tl.histogram`, `tl.gather`, `tl.cumsum`, `tl.dot`, a pipelined dot,
  block pointers, `tl.device_print` and atomics, and `256` only with `tl.make_tensor_descriptor`,
  which is the frame's one exception. evidence/launch-frame re-ran with no changed byte outside the
  written buffers.
- **L74** ACCEPT — Scope 7 names the H200 and code compiled for it, the grid sentence says that
  `2^31` to `2^32` programs are skipped and a larger grid may run, and scope 6 names host descriptors
  by dtype and block shape (`specialize.cc:165-226`, `nvidia-driver.py:209-264`). O5-O7 re-ran
  identical.
- **L75** ACCEPT — The builtins a kernel sees (`code_generator.py:357-365`), the lookup order
  (:381-427), the unwrap and `normalize_value` in `call_Function` (:1444-1448) and the static executor
  for `int` and `len` (:1638-1659) are as stated. An argument's operations run, the call emits
  nothing, and any other builtin is a `NameError`. evidence/py-builtins re-ran identical.

## New issues

- **NEW** gap — `Compiler.Faithful` scopes 4 and 6 (`lean/TVA/Rules/Compiler.lean`, the rule page,
  SEMANTICS "Launches") treat a `CompiledKernel` from `triton.compile` as the kernel's code, but
  `triton.compile` also takes an IR file path (`compiler.py:235-239`: `IRSource`). That replaces the
  source as `ir_override` does, and the scope does not exclude it. Fix: say that `triton.compile`'s
  source is an `ASTSource` of the kernel's `@triton.jit` function, and add the IR-file route to
  exclusions row 39 and to the reviewer's host check.
- **NEW** wording — SEMANTICS "Compiler hints: an operation with several results" (and the hint
  section of `TVA.Lang.Ops`) says that the hint "promises the same about each result" and that the
  model checks each. But a result may be a float (a tuple reduce or a loop carrying an accumulator)
  or have another rank (a loop or an `if` carrying a scalar beside a block), and neither the promise
  nor `HintVal` is defined for those. Fix: say "each integer or pointer result of the hinted value's
  rank", and that a hint after a loop or a runtime `if` is never read (M7-M10).
- **NEW** wording — SEMANTICS "Index arithmetic" says that `int32` arithmetic is promoted by "an
  operand of a wider or unsigned type". An unsigned operand narrower than 32 bits (`uint8`, `uint16`)
  does not promote `int32`: the signed, wider type wins, by the library's own `computationType?`.
  Fix: say "a wider type, or an unsigned type at least 32 bits wide".
- **NEW** gap — SEMANTICS (Scans) states `tl.cumsum`'s promotions but not its `dtype=` argument.
  Triton first widens `bfloat16` to `float32`, then converts the input to `dtype`
  (`standard.py:334-343`). State it as for `tl.sum`. No pinned tutorial uses it.

## Re-ran

- In full, on GPU 0 (H200 NVL, Triton 3.8.0, 2026-10-28), output identical to the recorded one:
  evidence/desc-zero-shape, hint-contig, hint-multi, python-numbers, if-names, py-builtins,
  compiled-options and claim-satisfiable (`work/rerun-*.txt`, `work/rerun_all.sh`).
- In full: evidence/launch-frame. Same conclusions: no changed byte outside the written buffers or
  the scratch, and the same scratch sizes. The per-kernel count of changed bytes inside the written
  buffers differs by a few bytes from run to run (random outputs that equal the sentinel).
- Tests added: `work/desc_edges.py` (D1-D9), `work/if_compile_time_livein.py` (I1-I6),
  `work/hint_bound_subexpr.py` (B1-B3) and `work/scratch_sweep.py`, each with its `.out`.

## Not checked

- The `tools/tvr.py` and claim.md changes for L71, beyond the two claims of
  evidence/claim-satisfiable (the gate's area).
- TRANSCRIPTION.md as a whole: only the rows and pitfalls the batch added.
- `lean/TVA/Math` and the numeric lemmas (round 12 replaces them).
- The release check (the coordinator's run had not finished when I finished).
- `tl.range(warp_specialize=True)` with pointer loads aborts the compiler's process ("LLVM ERROR:
  unsupported load type for producer commit"), so no launch happens. I noted this but did not
  pursue what "rejected" means for a host abort.
