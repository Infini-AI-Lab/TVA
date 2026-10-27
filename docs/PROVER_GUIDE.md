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
every operation, the transcription rules ([TRANSCRIPTION.md](TRANSCRIPTION.md)), the compiler
rule and the device facts. You cite them; you never write a device or compiler fact of your own
or choose accuracy constants. If the kernel needs something the library lacks, ask for it as a
library request. That covers a fact, an operation and a transcription rule. Kernel review
trusts the library, so a gap filled inside the kernel would go unchecked.

| Part | What it is | Who judges it |
|---|---|---|
| **Model** | the kernel transcribed into the `tl.*` vocabulary by the transcription rules, line by line | kernel reviewer (`MODEL_MISMATCH`) |
| **Specification** | `Inst`, `Inst.ok`, `launch`, `Pre`, `Post`; `Spec K := Meets Inst.ok Pre Post K` | kernel reviewer (`SPEC_TOO_WEAK`, `CONDITION_TOO_NARROW`) |
| **Assumptions** | `structure Assumptions K P`: `[rule]` fields cite the library, `[kernel]` fields are your own unproved facts | kernel reviewer (`RULE_MISCITED`; `ASSUMPTION_*` for `[kernel]`) |
| **Proof** | `theorem main : ∀ K P, Assumptions K P → Spec K` | Lean |

`kernels/vec_add_f32/lean/VecAdd.lean` has all four for tutorial 01.

## 1. The model

Write the kernel as a `Kernel` whose `body : Ctx → Prog Unit` is a `do` block, keeping the
source's line structure and names. [TRANSCRIPTION.md](TRANSCRIPTION.md) gives the rule for each
source construct; the main ones are repeated below.

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

Rules:
- **`tl.constexpr` arguments are Lean parameters** (`BLOCK_SIZE : Nat`).
- **Write every implicit conversion and broadcast explicitly**: `tl.cvt` for `.to(...)` and
  for dtype promotion, scalar–tensor arithmetic, `bc`, `.col`/`.row`. Use `tl.cvt8` into fp8,
  `tl.cvtFrom8` out of fp8, and `tl.itof` from integers. Write any narrowing of an integer-to-float value (an explicit chain, or the implicit
  conversion of a store into a narrower buffer) as one `tl.itof` to the final format. Write `-0.0` literals as `+0.0`, which is what Triton
  builds.
- **Pick the meaning of each float `+`/`-`.**
  - `tl.add`/`tl.sub` when no product feeds it.
  - `tl.mul_add a b c` for `a * b + c`.
  - `tl.addc` when you cannot tell.

  Say why in a comment. Reductions and scans work the same way: `tl.sum`/`tl.cumsum` when the
  input is not a product, `tl.sumc`/`tl.cumsumc` when it is. This is the most likely
  `MODEL_MISMATCH`.
- **Keep `mask` and `other` exactly as in the source.** A masked load without `other` gives
  unspecified values on the disabled lanes.
- **`tl.debug_barrier()` is a no-op.** Write a comment where it was. Compiler hints are not
  no-ops: write `tl.multiple_of`, `tl.max_contiguous`, `tl.max_constancy` and `tl.assume`.
  Put each check where the hinted value is defined, before any use of it. A hint inside a loop
  body on a loop-carried name, and the meaning of a hint on a rank ≥ 2 tensor, are open library
  issues (L1, L2 in [library-review/ISSUES.md](library-review/ISSUES.md)).
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
- **Do not declare anything in the `TVA` or `tl` namespaces, nor instances of library
  classes.** The reviewer rejects a bundle that does.
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

  Keep it as wide as the kernel's real use.
- **`launch d`**: the host's grid, the source's block shapes, and `enable_fp_fusion`, such as
  `Launch.grid1 (cdiv n BLOCK) [[BLOCK]]`. `Inst.ok` must make the launch valid
  (`Launch.Valid`): `BLOCK` a power of two of at most `2^20` elements, and the grid within
  CUDA's limits.
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
- **`[rule]`** cites a library rule or device fact as the library states it. Its statement may
  mention the library's constants, `K`, `P`, and your `Inst`, `Inst.ok`, `model` and `launch`,
  but no other constant of yours.
- **`[kernel]`** is an unproved fact about your own values (for example, a property of an error
  constant). It mentions neither `K` nor `P`. Prove it instead when you can.

Only `faithful` may mention `K`.

Then prove the assumptions consistent, so that the claim is not vacuous:

```lean
theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful P₀ model launch
  exact ⟨K, P₀, htot, ⟨fun d _ => hf d, …⟩⟩
```

Here `P₀` is a platform satisfying the rules about `P`: `Platform.any` if there are none, or
`Platform.ref rel abs sp`, the most accurate platform. It covers these rules for every bound
that allows one correct rounding ([the rule page](rules/num.md) gives the exact conditions):
- every `…Total` rule (`Platform.ref.mathTotal`, `divTotal`, `cvtTotal`, `atomicAddTotal`,
  `dotTotal`, `dotScaledTotal`);
- `MathAcc`:
  - `mathAcc_of_eta` for `abs ≥ η`;
  - `mathAcc_of_normal` for any `abs ≥ 0` where the real function's values are normal;
- `MathAt` at non-finite inputs (`mathAt`, from the table `sp`);
- division (`divFull`, `divRn`);
- `CvtAcc` (`cvtAcc`);
- `AtomicAddAcc` (`atomicAddAcc`);
- `DotAcc`: `dotAcc` (float accumulator), `dotAcc_int` (integer operands, signed accumulator);
- `DotScaledAcc` (`dotScaledAcc`).

`rel` and `abs` of `Platform.ref` are the bundle's `MathAcc` constants, and `sp` lists its
`MathAt` values. `Consistent` is mandatory. A bound tighter than one rounding is false of the
device anyway, so a bundle should not need its own witness.

## 4. Proof

The common route, for kernels whose programs touch disjoint tiles, is
`TVA.meets_independent`. It takes:

- `hf`: the `faithful` assumption;
- `hdebug`, `hvalid`: each covered launch has `debug = false` and is `Launch.Valid`;
- `R d init i`, `W d init i`: program `i`'s read and write footprints;
- `Q d init i m`: what program `i` leaves in its own memory;
- **`hsafe`**: each program is `Safe` from `init`;
- **`hdisj`**: `W i` is disjoint from `R j ∪ W j` for `i ≠ j`;
- **`hpost`**: the programs' results give `Post`.

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
| `tl.dot_scaled` | `safe_dot_scaled` (with `Rules.Num.DotScaledTotal`); accuracy: `DotScaledAcc` |
| hints on rank ≥ 2 | `safe_multiple_ofN`, `safe_max_contiguousN`, `safe_max_constancyN` |
| `tl.mul_add`, `tl.addc`, `tl.subc` | `safe_mul_add`, `safe_addc`, `safe_subc` (a relation on the result) |
| `tl.exp` …, `/`, `tl.cvt8`, `tl.dot` | `safe_math`, `safe_div`, `safe_cvt8`, `safe_dot` (need a result to exist); accuracy of `tl.dot`: `Rules.Num.DotAcc` |
| accuracy of `/` | `Rules.Num.DivFull` (fp32), `Rules.Num.DivRn` + `DivRn.divAcc` (fp64) |
| `tl.sum`, `sum1`, `sum0` / `sumc`, `sum1c` / `max1`, `max0`, `maxAll` | `safe_sum`, …, `safe_max1`, `safe_max0`, `safe_maxAll` (a tree for each output); accuracy: `sum_err_covers` |
| accuracy of `tl.exp` … | `Rules.Num.MathAcc`; special values (`exp(-∞) = +0`): `Rules.Num.MathAt` |
| `x.to(int)` | `safe_ftoi` (each element in range) |
| `for` loop | `safe_forRange` / `safe_forN_bind` with an invariant |
| `while` loop | `safe_while` with an invariant and a decreasing measure |
| the last line | `safe_last`, then that line's rule with `pure` as the rest |
| `return` | `safe_ret` |

**Programs that share memory through atomics** (split-K, histograms, counters): use `TVA.shared`
instead of `meets_independent`.
- Prove each program with `SafeA`.
- Private segments are ordinary `Safe` proofs, lifted by `SafeA.of_safe`.
- A scalar atomic on the shared region uses `SafeA.atomic1`: it may find any old value, and
  the step goes into the program's log.
- The postcondition `Q i log m` says what the log contains, for example "one `atomic_add` of
  my partial sum per cell".
- `replay_add` and `sum_proj` then give each integer cell's final value.

`Safe.bind` and `Safe.mono` compose and weaken. `TVA.Lemmas.Int` turns `int32` arithmetic into
integer arithmetic once you show it does not wrap.

**Tips.**
- Choose footprints as unions of `Buf.tile`s. `Buf.tile_disjoint` and `Buf.tile_disjoint_of`
  then prove `hdisj`.
- Validity of a buffer survives stores (`safe_store_block` gives presence).
- `#print axioms main` must show only `propext`, `Classical.choice` and `Quot.sound`.
- Where a lemma would help other kernels, add it to the library (`TVA/Lemmas/`), not to the
  bundle.

## 5. What kernel reviewers will ask

Read [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) before submitting. The questions most likely to
come back:

- Does each model line follow the transcription rule for its construct, and each `+` the right
  meaning?
- Are the masked lanes handled?
- Does `Inst.ok` exclude real uses, and is every narrowing stated?
- Does `Post` say enough, and is its tolerance meaningful at the largest covered sizes?
- Is each `[rule]` the library's fact for the operation the model uses?
