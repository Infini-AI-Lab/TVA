# Prover guide

Each kernel is one folder, `kernels/<kernel>/`, reviewed under the protocol in
[REVIEW_PROTOCOL.md](REVIEW_PROTOCOL.md). Your prompt is
[prompts/kernel-prover.md](prompts/kernel-prover.md).
- Start the folder with `python3 -I tools/tvr.py new <kernel> …`.
- Write the Lean bundle in its `lean/`, the manifest `kernel.json`, the model map and the
  write-up `README.md`.
- Make `tvr check <kernel>` pass.
- Answer each round's findings in `review/round-N/prover/`, and record them with
  `tvr respond <kernel>`.

You prove that a Triton kernel meets a specification on the device, under assumptions the
reviewers accept. You write four things in Lean, in one bundle. Lean checks the proof, and the
kernel reviewers judge the rest.

**Your scope.** The kernel is yours; the library is not. The library gives the meaning of
every operation and every fact about Triton a model relies on ([SEMANTICS.md](SEMANTICS.md)), the
compiler rule and the device facts. You cite them; you never write a device or compiler fact of
your own or choose accuracy constants. If the kernel needs something the library lacks, ask for
it as a library request: a fact, an operation, or the meaning of a construct. Kernel review
trusts the library, so a gap filled inside the kernel would go unchecked.
[TRANSCRIPTION.md](TRANSCRIPTION.md) is guidance on writing the model (worked patterns,
pitfalls, common reasons to drop a line); the reviewer judges your model by what it means, not by
whether it follows a pattern.

| Part | What it is | Who judges it |
|---|---|---|
| **Model** | the kernel transcribed into the `tl.*` vocabulary, line by line, meaning what the source does by SEMANTICS.md | kernel reviewer (`MODEL_MISMATCH`) |
| **Specification** | `Inst`, `Inst.ok`, `launch`, `Pre`, `Post`; `Spec K := Meets Inst.ok Pre Post K` | kernel reviewer (`SPEC_TOO_WEAK`, `CONDITION_TOO_NARROW`) |
| **Assumptions** | `structure Assumptions K P`: `[rule]` fields cite the library, `[kernel]` fields are facts about your own values (closed statements, which `consistent` proves) | kernel reviewer (`RULE_MISCITED`; `ASSUMPTION_*` for `[kernel]`) |
| **Proof** | `theorem main : ∀ K P, Assumptions K P → Spec K` | Lean |

`claims/vec_add_f32/Claim.lean` has the first three for tutorial 01 (the model, the specification,
the assumptions, and the `Statement` they make), with `consistent`; it holds no proof. In a claim task
the same claim is in the task's worked example, `examples/vec_add_f32/` beside these docs, with its
complete proof, `Proof.lean`: all four, `theorem main` included.

## 1. The model

Write the kernel as a `Kernel` whose `body : Ctx → Prog Unit` is a `do` block, keeping the
source's line structure and names, so that a reviewer can read the two side by side. What each
construct means is in [SEMANTICS.md](SEMANTICS.md); [TRANSCRIPTION.md](TRANSCRIPTION.md) has worked
patterns and the usual pitfalls. The points that matter most:

```lean
def add_kernel (x_ptr y_ptr output_ptr : Ptr .f32) (n_elements : I32) (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    let pid := tl.program_id ctx 0
    let block_start := pid * (IntV.ofNat BLOCK_SIZE : I32)
    let offsets := block_start + tl.arange 0 BLOCK_SIZE
    let mask := offsets.ltS n_elements
    let x ← tl.load (x_ptr + offsets) (some mask)
    ...
```

- **Stay inside the supported subset** (SEMANTICS.md, "The supported subset"; library issue L88).
  A kernel that uses another form of Python has no claim until the library adds the form: ask for
  it as a library request. A source that has `@triton.autotune` is outside the subset (library
  issue L116): ask the coordinator to set the kernel up again.
- **`tl.constexpr` arguments are Lean parameters** (`BLOCK_SIZE : Nat`).
- **Write every implicit conversion and broadcast explicitly**: `tl.cvt` for `.to(...)` and
  for dtype promotion, scalar–tensor arithmetic, `bc`, `.col`/`.row`. Use `tl.cvt8` into fp8,
  `tl.cvtFrom8` out of fp8, and `tl.itof` from integers. Write a narrowing with the default
  rounding of an integer-to-float value (an explicit chain, or the implicit conversion of a store
  into a narrower buffer) as one `tl.itof` to the final format; with
  `fp_downcast_rounding="rtz"`, as `tl.itof` to the intermediate format, then `tl.cvt … .rz`
  (SEMANTICS.md, `x.to(float)`).
- **Write each Python number as the constant Triton builds** (SEMANTICS.md, "Constants"): with an
  arithmetic or bitwise operator, as a value of `tl.where`, and in `tl.full`, in the operation's
  dtype, `FloatV.litF64`, `litF32`, `litF16` or `litBF16` of the number's double (`FloatV.ofPat` of
  its `float64` pattern), and `FloatV.litF8` in an fp8 dtype (not `tl.cvt8`; `Operand.constIn?`
  gives each); elsewhere (a comparison, `other=`, a stored value, another `tl.*`
  function) as a tensor of its own type, then the conversion. A zero, `-0.0` included, is `+0.0`. A
  name a statement assigned a number holds a tensor of its own type (SEMANTICS.md, "How the front
  end evaluates the source").
- **Pick the meaning of each float `+`/`-`.**
  - `tl.add`/`tl.sub` when no product feeds it (a `float32` quotient is a product: SEMANTICS.md
    "Float `+` and `-`").
  - `tl.mul_add a b c` for `a * b + c`.
  - `tl.addc` when you cannot tell.

  Say why in a comment. Reductions and scans work the same way: `tl.sum`/`tl.cumsum` when the
  input is not a product, `tl.sumc`/`tl.cumsumc` when it is. This is the most likely
  `MODEL_MISMATCH`.
- **Keep `mask` and `other` exactly as in the source.** A masked load without `other` gives
  unspecified values on the disabled lanes.
- **Map every line, and say why.** Each entry of the model map gives the model text, or drops the
  line with a reason and a note (the condition that makes the drop hold). A note that says why
  this operation (the IEEE `+` because no product feeds it), which conversion, or which condition
  of `Inst.ok`, saves the reviewer a question. `rules` is optional and unchecked.
- **Class every function of the source** (every `def` and `lambda`, from its first decorator to
  its last line): the kernel lines (`source.ranges` in `kernel.json`) hold the launched kernel and
  everything it runs, on the device or at compile time (helpers, however `jit` is spelled,
  `triton.constexpr_function`s); the host lines (`source.host`) hold the functions only the host
  runs; `source.unused` (a list of names) names a `def` nothing runs. The gate fails on a function
  in none of these, and the reviewers judge the classes (protocol §7, library issue L50). Class
  every code line outside a function too (imports, module-level assignments, a class's own lines,
  a `__main__` block), and the header of each unused function (its decorators and `def` lines: a
  `def` runs at import and evaluates them, defaults included, even if nothing calls the function): a
  line that binds a name the launch reads (the import of `triton.language as tl`, a JIT function's
  `def`) goes in the kernel lines, where it is mapped (modelled, or dropped with its reason), and the
  others in the host lines; the gate fails on a line in neither (L79). The only module-level names a
  launch may read are the imports of `triton` and of `triton.language as tl` and the module's
  `@triton.jit` functions, each bound only by its own `import` or `def`: a kernel that reads any
  other (a `tl.constexpr(…)` global, a dtype) is outside the supported subset, and reviewers file
  `UNSUPPORTED` (L88). Lines are counted at line feeds only; a source or Lean file with another line
  break (a form feed, U+2028, a lone carriage return) fails the gate (L51).
- **Integer reductions widen.** `tl.max`/`tl.min` of an integer narrower than 32 bits reduce
  `int32` values, whatever the signedness; `tl.sum` reduces `int32` or `uint32` (SEMANTICS.md,
  reductions). Write the conversion.
- **`tl.debug_barrier()` is a no-op.** Write a comment where it was. Prints and assertions
  (`tl.static_print`, `tl.device_print`, `print`, `tl.device_assert`, `assert`,
  `tl.static_assert`) may be dropped only when what the front end evaluates in their arguments has
  no effect; otherwise the operations it evaluates are model lines (SEMANTICS.md, "How the front
  end evaluates the source"; TRANSCRIPTION.md §4). Compiler hints are not no-ops: write
  `tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy` and `tl.assume`.
  Put each check where the hinted value is defined, before any use of it; a hint on a parameter, an
  induction variable or an inlined call's result has no effect and is dropped, and one on a loop's
  carried name checks the value before the loop (SEMANTICS.md, "Compiler hints").
- **Shifts:** `>>` is `tl.ashr` when the tensor operand Python dispatches on (the left one,
  or the right one when the left is a Python scalar) is signed before promotion, `tl.lshr`
  otherwise.
- **Block pointers, descriptors and `dot_scaled` operands are written as in the source.** Write
  `tl.make_block_ptr` with each `shape`/`strides`/`offsets` entry cast to `int64`, a
  `TensorDesc` with the source's `padding_option`, and `tl.dot_scaled` with the storage
  tensors as they are (packed fp4 as `uint8`). The library does the arithmetic and unpacking.
- **Scans take the combine function as a relation.** Write the `combine_fn` body as a
  relation (`AddRel false` for an IEEE `a + b`, `AddRel E.fusion` when a product may feed it,
  `fun a b c => c = a + b` for integers).
- **Declare everything in the kernel's namespace, and give no name another meaning.** No
  declaration in `TVA`, none with a name component `tl`, none whose name (as you would write it)
  would otherwise mean another constant (a library name through `TVA` or a namespace you `open`,
  or a root name such as `min`), names in plain ASCII only, no instance at all, no binder named
  like a library namespace (`tl`, `Tensor`, `Block`, …, or a look-alike), no notation,
  `open … renaming`, `export` or `set_option` beyond resource limits, printing and linters
  (library issue L20). The gate checks this as a backstop, and the kernel reviewer reads all of
  your Lean.
- **Float operations are `Prog`s** (`let y ← tl.neg x`), because a NaN result may be any NaN.
  State `Post` up to NaN payload (`FloatV.NaNEq`).
- **Loops:** `range(lo, hi, step)` is `tl.forRange`, and a fixed count is `tl.forN`. A loop's
  carried variables form a tuple state.
- If an operation you need is missing from `TVA.Lang.Ops`, **do not define it in the bundle**.
  Ask for it in the library, so its reading of the documentation is reviewed once.

## 2. Specification

- **`Inst`**: everything the host chooses, including sizes, block sizes, scalar arguments and
  buffers. Use `Buf d` for a tensor argument (`base`, `len`).
- **`Inst.ok`**: the instances covered. Typically this means:
  - positive block sizes;
  - sizes that keep `int32` index arithmetic from wrapping (`n + BLOCK ≤ 2^31`);
  - buffer lengths;
  - disjointness of buffers that are written.

  Keep it as wide as the kernel's real use. A runtime argument's value is part of the instance,
  and your conditions constrain it, for example `n ≤ x.len` for a length `n_elements` of the buffer
  `x` (the user's decision, 2026-10-29, library issue L117: "for the viriable n_elements, its value
  should be merged into the assumptions of the prover. e.g., x.len >= n_elements, etc."). Point 11
  of `Compiler.Faithful`'s scope assumes that the launch runs the code Triton compiles for the
  instance's own argument values, so the conditions, not Triton's caches, decide which values the
  claim covers.
- **The kernel and its launches stay within part (A) of `Compiler.Faithful`'s scope**
  ([rules/compiler-faithful.md](rules/compiler-faithful.md), "Scope", and its checklist in "How a
  bundle cites it"), point by point: (4) only the launch options `Launch` records plus
  `num_warps`, `num_stages`, `maxnreg`, `launch_cooperative_grid`, at the launch, at
  `kernel.warmup(…)`, in `triton.compile`'s options or in a `triton.Config`, and no
  `@triton.heuristics`, with the decorator's `debug` as `Launch.debug`; (5) no descriptor built in
  the kernel (no call of `tl.make_tensor_descriptor` the front end visits on a covered instance);
  (6) for a `CompiledKernel`, the kernel's own source (an `ASTSource` for
  `triton.compile`, not an IR file) and arguments in its specialisation (dtypes, pointer alignment,
  integer class, host descriptors' dtype and block shape); (8) no host `TensorDescriptor` with
  `round_f32_to_tf32=True`. Say so in the write-up's Conditions.
  Part (B), the environment (points 1, 2, 3, the environment's part of 4, 7, 10 and 11: other
  writers, knobs, hooks, `kernel.debug` and `preload`, the target and host (an H200; x86-64,
  CPython 3.13 without `-O`/`-OO` and with `PYTHONOPTIMIZE` unset, in the default floating-point
  environment), Triton as installed and other code, the kernel new: the code Triton compiles for the
  launch's own argument values and settings), is assumed and not checked:
  the write-up says only that the claim
  relies on it (the user's decisions, 2026-10-29: "To make the verifier easy, we do not verify
  triton cache.", "we assume the kernel is new." and "So basically, we want to verify the kernel
  itself, rather than triton installation or other code programmed by users.").
- **`launch d`**: the host's grid, the source's block shapes, and `enable_fp_fusion`, such as
  `Launch.grid1 (cdiv n BLOCK) [[BLOCK]]`. `Inst.ok` must make the launch valid
  (`Launch.Valid`): `BLOCK` a power of two of at most `2^20` elements, the grid within CUDA's
  limits, and fewer than `2^31` programs. `Launch.valid_grid1` proves it for a 1-D grid.
- **`Pre d init`**: the buffers are valid (`Buf.Valid`, which means aligned and present), plus
  conditions on input values if needed.
- **`Post d init fin`**: what the outputs are. Also say what does *not* change, for example
  `∀ x ∉ out.bytes, fin x = init x`.

## 3. Assumptions

Every bundle that reasons about the model cites `Compiler.Faithful`
([rule page](rules/compiler-faithful.md)):

```lean
  /-- [rule] `Compiler.Faithful` -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
```

A kernel that uses approximate operations also needs facts about `P`. Examples are `tl.exp`,
`/`, `tl.dot`, fp8 conversions and float atomics. Cite the library's device fact for the
operation, format and target (`[rule]`). If the library has none, the kernel waits: ask for it
as a library request, and do not write it yourself.

Every assumption is one of two kinds, and the gate checks both:
- **`[rule]`** cites a library rule or device fact as the library states it, by its name, applied
  to `P`: a fact [rules/num.md](rules/num.md)'s "Facts the library states" lists, as it lists it
  (`Rules.Num.Exp2F32 P`, `Rules.Num.DotF16 P`, `Rules.Num.DivFull P`, `Rules.Num.DivRn P
  .binary64`, a `…Total` rule), or `Compiler.Faithful` exactly as above, `Faithful P (model d)
  (launch d) (K d)` for one bound `d` (library issue L62), under `∀`/`→` binders if needed. The gate
  refuses any other constant of `TVA.Rules`. Never a shape with constants
  (`Rules.Num.MathAcc P .exp .binary32 …`, `DotAcc`, `DivAcc`, …), even the library's constants
  written out: the gate refuses it (library issue L62). Its statement may mention the library's
  constants, `P`, and your `Inst`, `Inst.ok`, `model` and `launch`, but no other constant of yours
  (a `private` or `_root_` one included, L63), and `K` only in `Compiler.Faithful`.
- **`[kernel]`** is a fact about your own values (for example, a property of an error constant).
  It mentions neither `K` nor `P`, so it is a closed statement, and `consistent` proves it (its
  witness must satisfy every field). The reviewer judges whether it is needed and clear, and
  whether the constants it constrains mean what your write-up says. A lemma proved on its own is
  clearer still: use one when you can.

Only `faithful` may mention `K`.

Then prove the assumptions consistent, and the conditions satisfiable
(`theorem satisfiable : Claim.Satisfiable Inst.ok Pre`, one covered instance and memory; the gate
checks it, library issue L71), so that the claim is not vacuous:

```lean
theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful P₀ model launch
  exact ⟨K, P₀, htot, ⟨fun d _ => hf d, …⟩⟩
```

Here `P₀` is a platform satisfying the facts the bundle cites: `Platform.any` if there are none,
or else `Platform.sm90ref` (`lean/TVA/Lemmas/Sm90.lean`), which satisfies every fact of
`TVA.Rules.Num.Sm90` (`Platform.sm90ref.expF32`, `.exp2F32`, …, `.dotF16`, …, `.atomicAddF32`),
`DivFull`, `DivRn` and every `…Total` rule together ([the rule page](rules/num.md),
"Satisfiability"). `Consistent` is mandatory. The facts are the library's, so a bundle needs no
witness of its own.

## 4. Proof

The common route, for kernels whose programs touch disjoint tiles, is
`TVA.meets_independent`. It takes:

- `hf`: the `faithful` assumption;
- `hdebug`, `hvalid`: each covered launch has `debug = false` and is `Launch.Valid`;
- `R d init i`, `W d init i`: program `i`'s read and write footprints;
- `Q d init i m`: what program `i` leaves in its own memory;
- **`hsafe`**: each program is `Safe` from `init`;
- **`hdisj`**: `W i` is disjoint from `R j ∪ W j` for `i ≠ j`;
- **`hRW`**: `R i` and `W i` are disjoint: a program does not read what it writes (library issue
  L15: two accesses of one program to the same bytes, one writing, are not well behaved);
- **`hpost`**: the programs' results give `Post`.

`Safe` writes each byte at most once: a store or atomic rule hands the rest of the program the
write footprint without the bytes it wrote (`W \ b.tile p B n` after `safe_store_block`). A
kernel that stores the same bytes twice, reads back what it wrote, or updates an input in place
cannot be proved, and is out of scope ([exclusions.md](exclusions.md)).

**Proving `Safe` for one program.** Unfold the model for program `i`; a `body_eq` lemma that
`rfl` proves keeps this tidy. Then apply one rule per line, in order:

| Line | Rule |
|---|---|
| `tl.load (b.ptr 0 + offs p B) (some (mask p B n))` | `safe_load_block` (gives each element) |
| `tl.store (b.ptr 0 + offs p B) v (some (mask …))` | `safe_store_block` (gives each element, the frame, presence) |
| any other `tl.load` / `tl.store` | `safe_load`, `safe_store` (distinct lanes), `safe_store_any` |
| `tl.add`, `sub`, `mul`, `fma`, `div_rn`, `sqrt_rn`, `neg`, `abs` | `safe_add`, … (the IEEE result up to NaN payload); `safe_exact` when no NaN arises |
| `x.to(float dtype)` | `safe_cvt` (standard formats, `FloatV.cvtWith rnd`), `safe_cvtFrom8` (out of fp8, exact when finite), `safe_itof` (from integers, `tl.ItofRel`) |
| `tl.rand(seed, offs)`, `tl.randint` | `safe_rand` (`tl.randint` is the exact function `Philox.randint4x`) |
| `tl.maximum`, `tl.minimum` | `safe_maximum`, `safe_minimum` (`MaxRel`) |
| `//`, `%`, `<<`, `>>` | `safe_idiv`, `safe_irem`, `safe_shl`, `safe_ashr`, `safe_lshr` (no bad lane) |
| `tl.multiple_of`, `max_contiguous`, `max_constancy`, `assume` | `safe_multiple_of`, … (prove the hint) |
| `tl.dot(a, b) + c` | `safe_dot_add` |
| `tl.dot_scaled(…) + c` | `safe_dot_scaled_add` |
| `tl.sum(x[:, :, None] * y[None, :, :], axis=1)` | `safe_sum_bcast_mul` |
| `desc.load(offs)`, `desc.store(offs, v)` (TMA) | `safe_desc_load`, `safe_desc_store` (descriptor over a `Buf`) |
| `desc.atomic_add(offs, v)` etc. | `safe_desc_atomic` (elements the program owns; `RedOp.upd_total_int`) |
| `desc.gather`, `desc.scatter` | `safe_desc_gather`, `safe_desc_scatter` |
| `tl.load(block_ptr, boundary_check, padding_option)`, `tl.store(block_ptr, …)` | `safe_load_block_ptr`, `safe_store_block_ptr`; `BlockPtr.elem_eq`, `inb_iff`, `advance_offsets` |
| a tensor atomic on memory the program owns | `safe_atomic_buf` |
| `tl.join`, `tl.split`, `tl.interleave`, `tl.reshape` | pure: `Tensor.get_join_snoc`, `split_join`, `get_interleave_snoc`, `get_reshape`, `reshape_reshape` |
| `tl.reshape(…, can_reorder=True)` | `safe_reshape_any` |
| `tl.associative_scan`, `tl.cumsum`, `tl.cumprod` | `safe_associative_scan`, `safe_cumsum(c)`, `safe_icumsum` (prefix sums), `safe_cumprod`, `safe_icumprod`; error: `sum_err_list` with `ScanLeaves.perm` |
| `tl.sort` | `safe_sort` (integers and floats: `SortKey.Lawful`) |
| `tl.histogram` | `safe_histogram` (prove every enabled value in range) |
| `tl.dot_scaled` | `safe_dot_scaled` (with `Rules.Num.DotScaledTotal`); accuracy: the named fact (`Rules.Num.DotScaledE4m3`, `DotScaledE4m3E2m1`; each is a `DotScaledAcc`) |
| hints on rank ≥ 2 | `safe_multiple_ofN`, `safe_max_contiguousN`, `safe_max_constancyN` |
| `tl.mul_add`, `tl.addc`, `tl.subc` | `safe_mul_add`, `safe_addc`, `safe_subc` (a relation on the result) |
| `tl.exp` …, `/`, `tl.cvt8`, `tl.dot` | `safe_math`, `safe_div`, `safe_cvt8`, `safe_dot` (need a result to exist); accuracy of `tl.dot`: the named fact (`Rules.Num.DotF16`, `DotBF16`: each a `DotLiftAcc`, relative to lifted magnitudes, which `DotLiftAcc.mono` weakens, one `K` step by `DotLift.dotAdd_step`; the fp8 chunk facts `DotE5m2Chunk`, …: a `DotChunkAcc`, a whole dot by `Dot8Chunk.DotF8Chunk.liftAcc`; the fp8 loss facts `DotE5m2ChunkLoss`, …: a `DotChunkLoss`, each chunk within the smaller of two bounds, from the exact inputs by `Dot8Chunk.DotF8ChunkLoss.err`, and giving the chunk fact by `DotF8ChunkLoss.chunk`; `DotF32Ieee`: a `DotAcc`, which `DotAcc.mono` weakens); `tl.cvt8`: the named fact (`Rules.Num.CvtF32E4m3`, … in range, a `CvtRn`; `CvtSatF32E4m3`, … on every input, a `CvtSat`, with `FloatV.cvtSat_rn_err`) |
| accuracy of `/` | `Rules.Num.DivFull` (fp32), `Rules.Num.DivRn P .binary64` + `DivRn.divAcc` (fp64) |
| `tl.sum`, `sum1`, `sum0` / `sumc`, `sum1c` / `max1`, `max0`, `maxAll` | `safe_sum`, …, `safe_max1`, `safe_max0`, `safe_maxAll` (a tree for each output); accuracy: `sum_err_covers` |
| accuracy of `tl.exp` … | the named fact (`Rules.Num.ExpF32`, `Exp2F32`, `LogF32`, …; each is a `MathAcc`, which `MathAcc.mono` weakens); special values (`exp(-∞) = +0`): `Rules.Num.ExpF32NegInf`, `Exp2F32NegInf` (a `MathAt`) |
| `x.to(int)` | `safe_ftoi` (each element in range) |
| `for` loop | `safe_forRange` / `safe_forN_bind` with an invariant and per-trip write footprints `Wt t`, disjoint across trips (`fun _ => ∅` when the body writes nothing; `iterW_empty`) |
| `while` loop | `safe_while` with an invariant, a decreasing measure `μ`, and write footprints `Wv` per value of `μ`, disjoint |
| the last line | `safe_last`, then that line's rule with `pure` as the rest |
| `return` | `safe_ret` |

**Programs that share memory through atomics** (split-K, histograms, counters): use `TVA.shared`
instead of `meets_independent`.
- Prove each program with `SafeA`.
- Private segments are ordinary `Safe` proofs, lifted by `SafeA.of_safe`.
- A scalar atomic on the shared region uses `SafeA.atomic1`, a tensor atomic (masked or not)
  `SafeA.atomic`: it may find any old value, and each step goes into the program's log.
- The postcondition `Q i log m` says what the log contains, for example "one `atomic_add` of
  my partial sum per cell". It must not depend on the log's order (`PermInv`): a program's
  atomics may be performed after later steps that do not depend on their result (`Step.defer`).
  A launch in which a program uses an atomic's result on a cell another program, or another
  atomic operation of the program, also accesses is excluded (SEMANTICS.md "Atomics", library
  issue L161).
- `replay_add` and `sum_proj` (`replay_add_any`, `sum_proj_any` at any width) then give each
  integer cell's final value.
- A logged step does not record its operand, so for float accumulation and `max`/`min` the
  postcondition pairs the log with the program's operands per position: `Paired R log ops`
  (`Paired.lanes` for a tensor atomic's lanes, `Paired.append`; `Paired.perm` makes it
  `PermInv`). `replay_err_shared` (float, with `accStep_atomicAdd` from `Rules.Num.AtomicAddF32`
  and `accStep_other`) and `replay_max`, `replay_min` (integers) then bound or give the cell
  against every program's operands.
- `shared` also needs `hRW` (disjoint read and write footprints) and `hQ` (`PermInv`).

`Safe.bind` (the two parts write disjoint footprints; `Safe.bind_ro` when the first writes
nothing) and `Safe.mono` compose and weaken. `TVA.Lemmas.Int` turns `int32` arithmetic into
integer arithmetic once you show it does not wrap.

**Tips.**
- Choose footprints as unions of `Buf.tile`s. `Buf.tile_disjoint` and `Buf.tile_disjoint_of`
  then prove `hdisj`.
- For the 1-D block pattern, `Block.fits_of_grid_pow2` gives `Block.Fits` for every program of
  the grid `cdiv(n, BLOCK)` when `n < 2^31` and `BLOCK` is a power of two.
- Validity of a buffer survives stores (`safe_store_block` gives presence).
- `#print axioms main` must show only `propext`, `Classical.choice` and `Quot.sound`.
- Where a lemma would help other kernels, add it to the library (`TVA/Lemmas/`), not to the
  bundle.

## 5. What kernel reviewers will ask

Read [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) before submitting. The questions most likely to
come back:

- Does each model line mean what its source line does (SEMANTICS.md), and each `+` the right
  meaning?
- Are the masked lanes handled?
- Does `Inst.ok` exclude real uses, and is every narrowing stated?
- Does `Post` say enough, and is its tolerance meaningful at the largest covered sizes?
- Is each `[rule]` the library's fact for the operation the model uses?
