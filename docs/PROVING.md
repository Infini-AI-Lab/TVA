# Proving a claim

A claim (`claims/<kernel>/Claim.lean`) defines `claim : TVA.Claim` in its own namespace. Proving
it means writing `theorem main : claim.Holds` in that namespace, in a file `Proof.lean` that
imports `Claim` and `TVA.Proof` (and Mathlib), with lemmas in modules under `Proof/` if you like.
Lean checks the proof; nobody reads it. The checker (`sandbox/claim/tva-check`) accepts it when:

1. `Claim.lean` is unchanged (its sha256);
2. the proof files pass the hygiene list: no `sorry`, `admit`, `axiom`, `native_decide`, `unsafe`,
   `implemented_by`, `extern`, no meta code (`macro`, `syntax`, `elab`, `#eval`, …), no
   `notation`, no `local` or `scoped` instance, no `set_option` but `maxHeartbeats`,
   `maxRecDepth` and `synthInstance.maxHeartbeats`; imports only `TVA`, `Mathlib`, `Claim`,
   `Proof`;
3. `Claim` and `Proof` build;
4. `main` has exactly the type `claim.Holds`, and `#print axioms` shows only `propext`,
   `Classical.choice` and `Quot.sound`;
5. `leanchecker` re-checks the compiled files.

The claim is fixed. If it turns out false for the kernel, the proof cannot be finished; do not
look for a way around a condition.

Read first: `lean/TVA/Claim.lean` (what `claim.Holds` means), the claim's `README.md` and
`Claim.lean`, and [SEMANTICS.md](SEMANTICS.md) for what each model line means. The worked example
is `vec_add_f32`'s proof, `sandbox/claim/oracle/vec_add_f32/Proof.lean` (in a claim container,
`examples/vec_add_f32/` beside these docs); section 8 walks through it.

## 1. The shape of a proof

`claim.Holds` unfolds to: for every platform `P` with `Device P`, every instance `d` with
`d.ok`, and every initial memory with `Pre d init`, the launch is `Correct`. So `main` starts

```lean
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
```

and ends with the endpoint `correct_of_safe` (`TVA.Proof.Correct`), which reduces `Correct` to
four things:

```lean
  refine correct_of_safe (R d) (W d) (Q d init) hvalid (fun i => ?_) ?_ ?_ ?_
```

- `hvalid : L.Valid`. For a 1-D grid, `Launch.valid_grid1`: fewer than `2^31` programs, and every
  listed block shape a power of two with at most `2^20` elements (`Shape.TritonOk`).
- **each program is `Safe`** alone from `init`, with read footprint `R i`, write footprint `W i`
  and post-condition `Q i` ([section 2](#2-safe-one-program-at-a-time));
- **the footprints**: `∀ i j, i ≠ j → Disjoint (W i) (R j ∪ W j)` (no program writes what another
  reads or writes) and `∀ i, Disjoint (R i) (W i)` (no program reads what it writes);
- **the specification**: any final memory `fin` that agrees on each program's footprint with some
  memory satisfying `Q i`, and with `init` wherever no program writes, satisfies `Post`.

The docstring of `correct_of_safe` shows this skeleton. Its proof (`independent`,
`TVA.Proof.Compose`) shows that such a launch is well behaved, cannot hang, and ends as the
programs' own runs say, whatever the interleaving. Almost every kernel fits: each program reads
inputs and writes its own output tile. Kernels whose programs meet through atomics on shared cells
use `correct_of_safeA` instead ([section 6](#6-atomics)).

**Plan before walking.** Write down, before any rule: each program's footprints, what each output
element satisfies, each loop's invariant (with its error bound), and check the bound at the
largest covered instance. Then state the lemmas and `main`. A wrong invariant found after the walk
costs hours.

## 2. `Safe`: one program at a time

`Safe E R W Q m p` (`TVA.Proof.Safe`): the program `p`, run alone from memory `m`, cannot reach
undefined behaviour and always ends, along every choice and every order of its atomics; it reads
only bytes of `R`, writes only bytes of `W`, each byte at most once, uses no atomic's result, and
ends in a memory satisfying `Q`. It is an inductive predicate with one constructor per effect of
`Prog`.

A proof of `Safe` walks the model's `do` block from the top, one rule per line. Every rule has the
shape `Safe … (op >>= k)`: its premises are what the operation needs, and its last premise is the
rest of the program `k`, for every result the operation may give. So a walk is a chain of

```lean
  refine safe_load_buf b hA hin hv hR fun v hv' _ => ?_
  refine safe_add fun r hr => ?_
  …
```

each `?_` being the next line. Start by unfolding the model for program `i`
(`simp only [model, add_kernel]`, or a `body_eq` lemma that `rfl` proves). Points to know:

- **The last line.** A `do` block's last line is not followed by `>>=`; `safe_last` turns it into
  `op >>= pure`, so its rule applies, and `safe_ret` ends the walk with the post-condition.
- **Composition.** `Safe.bind` (or `Safe.seq`) splits a program in two parts with disjoint write
  footprints; `Safe.bind_ro` when the first part writes nothing; `Safe.mono` weakens the
  post-condition or enlarges the write footprint (the read footprint cannot grow).
- **Each byte is written once.** After a store, the rest of the program has the write footprint
  without the stored bytes. A kernel that stores the same bytes twice, reads back what it wrote, or
  updates an input in place cannot be proved: that is a conflict within the program, which a
  well-behaved launch does not have (SEMANTICS.md, "Launches").
- **Platform operations need a result.** A rule for `tl.exp`, `/`, `tl.dot`, `tl.cvt8`, … asks that
  the platform has a result (`hex`); the totality facts give it: `hP.mathTotal`, `hP.divTotal`,
  `hP.cvtTotal`, `hP.dotTotal`, `hP.atomicAddTotal`, `hP.dotScaledTotal` for `hP : Device P`. The
  rest of the walk then holds for every result the platform allows; how close that result is comes
  from the accuracy facts, later.
- **Choices are universally quantified.** A masked-out lane without `other`, a reduction's tree, an
  approximate result: the rule's continuation quantifies over them, so carry only what the rule
  tells you about them.

### Which rule for which line

| Source line | Rule | Module |
|---|---|---|
| `tl.load` of buffer elements | `safe_load_buf` (each enabled lane gets its element) | `Proof.Rules.Memory` |
| `tl.store` to buffer elements | `safe_store_buf` (the elements hold the values; the frame; presence) | `Proof.Rules.Memory` |
| any other load or store | `safe_load`, `safe_store` (distinct lanes), `safe_store_any` (lanes may collide) | `Proof.Rules.Memory` |
| scalar load or store | `safe_load1`, `safe_load1_buf`, `safe_store1` | `Proof.Rules.Memory` |
| 1-D block `pid * B + tl.arange(0, B)`, `mask = offs < n` | `safe_load_block`, `safe_store_block` | `Proof.Pattern.Block` |
| 2-D tile `base + rows[:, None] * s0 + cols[None, :] * s1` | `safe_load_tileAt`, `safe_store_tileAt` | `Proof.Pattern.Tile` |
| block pointer | `safe_load_block_ptr`, `safe_store_block_ptr` | `Proof.Pattern.BlockPtr` |
| descriptor | `safe_desc_load`, `safe_desc_store`, `safe_desc_atomic` | `Proof.Pattern.Desc` |
| block-pointer or descriptor padding | `safe_pad`, `Padding.exists_holds` | `Proof.Rules.Memory` |
| `tl.add`, `sub`, `mul`, `fma`, `div_rn`, `sqrt_rn`, `neg`, `abs` | `safe_add`, …; `safe_upToNaN`, `safe_exact` | `Proof.Rules.Arith` |
| `tl.mul_add`, `tl.addc`, `tl.subc` | `safe_mul_add`, `safe_addc`, `safe_subc` | `Proof.Rules.Arith` |
| `/`, `tl.fdiv`, `tl.exp` and the other math functions | `safe_div`, `safe_fdiv`, `safe_math` | `Proof.Rules.Arith` |
| `tl.maximum`, `tl.minimum` | `safe_maximum`, `safe_minimum` | `Proof.Rules.Arith` |
| integer `//`, `%`, `<<`, `>>` | `safe_idiv`, `safe_irem`, `safe_shl`, `safe_ashr`, `safe_lshr`; scalars `safe_idiv1`, `safe_irem1` | `Proof.Rules.Arith` |
| `tl.convert`, `tl.cvt`, `tl.cvt8`, `tl.cvtFrom8`, `tl.itof`, `tl.ftoi` | `safe_convert`, `safe_cvt`, `safe_cvt8`, `safe_cvtFrom8`, `safe_itof`, `safe_ftoi` | `Proof.Rules.Convert` |
| `tl.sum`, `sum1`, `sum0`, `sumc`, … | `safe_sum`, `safe_sum1`, `safe_sum0`, `safe_sumc`, … (a covering tree of bounded depth) | `Proof.Rules.Reduce` |
| `tl.max`, `tl.min` | `safe_maxAll`, `safe_max1`, `safe_max0` | `Proof.Rules.Reduce` |
| `tl.argmax`, `tl.argmin` | `safe_argmaxAll`, `safe_argmax1`, `safe_argmax0`; facts `tl.ArgRes.index_lt`, `tl.ArgRes.value`, `tl.ArgRes.tree_of_noNaN`, `Tree.eval_argCombine_first` | `Proof.Rules.ArgReduce` |
| `tl.dot`, `tl.dot(a, b) + c` | `safe_dot`, `safe_dot_add` | `Proof.Rules.Dot` |
| `tl.sum(x[:, :, None] * y[None, :, :], 1)` | `safe_sum_bcast_mul` (with `tl.MulRel.total`, `AccAdd.total`) | `Proof.Rules.Dot` |
| `tl.dot_scaled` | `safe_dot_scaled`, `safe_dot_scaled_add` | `Proof.Rules.Dot` |
| scans | `safe_associative_scan`, `safe_cumsum`, `safe_cumsumc`, `safe_cumprod`, `safe_icumsum`, `safe_icumprod` | `Proof.Scan` |
| `tl.sort`, `tl.histogram` | `safe_sort`, `safe_histogram` | `Proof.Scan` |
| `tl.rand` | `safe_rand`, `safe_uint_to_uniform_float` (`tl.randint` is an exact function) | `Proof.Random` |
| `tl.assume`, `tl.static_assert` | `safe_assume`, `safe_static_assert` | `Proof.Rules.Core` |
| hints | `safe_multiple_of`, `safe_max_contiguous`, `safe_max_constancy`, `safe_max_contiguous_multiple_of`, `safe_multiple_of1`, and the `…N` forms | `Proof.Rules.Core` |
| `tl.reshape_any` | `safe_reshape_any` | `Proof.Rules.Core` |
| loops | `safe_forRange`, `safe_forN`, `safe_forN_bind`, `safe_while` | `Proof.Rules.Loop` |
| an atomic on elements the program owns | `safe_atomic_buf` | `Proof.Rules.Memory` |
| an atomic on cells several programs update | `SafeA.atomic_add`, `SafeA.atomic`, `SafeA.atomic1` | `Proof.Shared.Rules` |
| exact operations (`let x := …`) | no rule: `simp` with the `get_*` lemmas of `TVA.Tensor` | `TVA.Tensor` |

To find a rule, look in the table of `lean/TVA/Proof/Basic.lean`, or `grep -rn "theorem NAME"
lean/TVA/Proof`. [LIBRARY.md](LIBRARY.md) maps every module.

## 3. Footprints and disjointness

State footprints with buffers (`Buf`, `TVA.Memory`): `b.elems S` is the bytes of the elements `S`
of `b`. A program's `R i` and `W i` are usually unions of `b.elems` of its tile of each buffer.

- **From lanes to elements.** Most accesses address elements of one buffer: lane `i` addresses
  element `g i` of `b` when enabled (`en i`), which `Buf.Addrs` states. Prove it once from the
  lanes' pointers with `Buf.addrs_of_get`; then `safe_load_buf`/`safe_store_buf` give each lane its
  element, and the access's bytes are `b.elems (Buf.img g en)` (`Buf.Addrs.accBytes`). A store also
  needs distinct elements per enabled lane (`hinj`).
- **Patterns** (`TVA.Proof.Pattern`) do that once per access shape and give footprints you can
  compare across programs: `Buf.tile p B n` for 1-D blocks (`Buf.tile_disjoint`,
  `Buf.tile_sub_elems`, `Block.tile_div`), `Buf.tileAt` and `Buf.tile2` for 2-D tiles
  (`Buf.tileAt_disjoint`, `Buf.tile2_disjoint`, `Buf.tile2_disjoint_grid`,
  `Buf.tileAt_sub_matrix`). A store through a 2-D tile needs a layout that does not alias
  (`Tile2.Inj`: `Tile2.inj_rowMajor`, `Tile2.inj_colMajor`).
- **Disjointness lemmas.** `Buf.elems_disjoint` (two element sets of one buffer),
  `Buf.elems_disjoint_of` (two buffers), `Buf.elems_mono`, and `Disjoint.mono` to reduce a claim's
  `Inst.ok` disjointness to the programs' tiles. `Disjoint.union_right` splits `R j ∪ W j`.
- **The last goal.** To read an output element in `fin`, take the program `i` that wrote it, get
  its memory `m` from the hypothesis on the footprints, and move the read with `Mem.read_congr`
  (the element's bytes lie in `W i`; `Buf.span_sub_elems`). Bytes no program writes equal `init`'s.
- **Memory facts** (`TVA.Proof.Mem`): `Mem.read_congr`, `Mem.write_frame`, `Mem.read_write_self`,
  `Mem.read_write_other`, `writeAll_read` (a store to distinct lanes reads back lane by lane).
  `Buf.valid_of_agree` carries a buffer's validity to a later memory.

## 4. Integers

Index arithmetic is `int32` (sometimes `int64`) and wraps. A proof shows no step wraps, then
computes with `Int` and `Nat` (`TVA.Proof.Int`; its module docstring states the conventions).

- **Track values.** Every lemma takes the operands' values as hypotheses and gives the result's:
  from `ha : a.toInt = x`, `hb : b.toInt = y` and a bound, `IntV.I32.add_nat`, `mul_nat`,
  `sub_nat` give `(a + b).toInt = x + y`, and so on. Carry each intermediate's value down the walk
  and discharge the bounds with `omega`.
- **Constants**: `IntV.I32.toInt_ofNat` (with the bound), `IntV.toInt_ofNat_lit` for a numeral.
- **Lanes**: `x.get1 i` is lane `i`; the `@[simp]` lemmas `Tensor.get1_add`, `get1_mul`,
  `get1_ltS`, … reduce a lane of an expression to the expression of lanes;
  `IntV.I32.toInt_sadd_arange` gives the lanes of `start + tl.arange(0, n)`.
- **Comparisons**: `IntV.lt_eq`, `IntV.le_eq`, `IntV.eq_eq` turn `<`, `<=`, `==` into comparisons of
  values; `IntV.toInt_max`, `IntV.toInt_min` do `tl.maximum`, `tl.minimum`.
- **`//` and `%`** of a nonnegative value by a positive one: `IntV.div?_of_nonneg`,
  `IntV.rem?_of_nonneg`; on lanes `idiv_ok`, `irem_ok`, `idiv_toInt`, `irem_toInt`.
- **Casts**: `IntV.toInt_cast_of_sfits`, `IntV.toInt_cast`; a `tl.convert` between integer dtypes
  is the cast (`ConvRel.iff_cast_i32_i64` and the other pairs).
- **Block sizes**: for the grid `cdiv(n, B)` with `B` a power of two and `n < 2^31`, every program's
  offsets fit (`Block.fits_of_grid_pow2`).

## 5. Floats and error bounds

Nothing about floats is trusted beyond `TVA.Base.Float` and the device facts; the float kit
(`TVA.Proof.Float`) proves the bounds.

- **Exact results.** A `tl.*` float operation returns the IEEE result up to NaN payload
  (`FloatV.NaNEq`). When the IEEE result is finite, `FloatV.NaNEq.exact_of_isFinite` gives
  equality. `vec_add`'s `Post` is stated up to NaN payload and needs nothing more.
- **One rounding.** For finite operands whose exact result `q` has `|q| ≤ maxVal`, the result is
  finite and within `|q|·u + η` of `q` (`FloatV.add_err`, `sub_err`, `mul_err`, `fma_err`,
  `divRn_err`, `sqrtRn_err`, `cvt_err`; the `_real` forms in `ℝ`). `u`, `η`, `maxVal` of each format
  are in `TVA.Proof.Float.Constants` (`binary32_u`, …), not `@[simp]`: rewrite with them by name.
  The format must be `Regular` (`Format.binary32_regular`, `Format.Std.regular`).
- **Fusable additions.** `tl.addc` and `tl.mul_add` are within `3u·(|x| + |y|) + 4η`
  (`FloatV.AddRel.err`, `FloatV.mulAdd_err`), relative to the operands' magnitudes, not the sum's.
- **Reductions.** Every tree of depth `D` over `n` leaves: `sum_err_depth`, `sum_err_covers`
  (IEEE), `sumc_err_depth`, `sumc_err_covers` (fusable), giving
  `errBound δ η D (n − 1) (Σ |xᵢ|)` (`Tree.errBound`). `Device.SumDepth` (`hP.sumDepth`) bounds the
  depth by `⌈log₂ n⌉`. Maxima: `Tree.eval_maxRel`, `FloatV.MaxRel.max_approx`.
- **Approximate functions and division.** Take the fact from `hP`: `hP.expF32 X hX`,
  `hP.divFull`, `hP.exp2F32`, … (each a `Device.MathAcc` or `Device.DivAcc`; weaken constants with
  `Device.MathAcc.mono`). Special values: `hP.expF32NegInf`, `hP.expF32Overflow`.
- **`tl.dot`** (`TVA.Proof.Dot`): one trip of `acc += tl.dot(a, b)` from a whole-dot fact
  (`DotLoop.dotAdd_step` from `Device.DotAcc`; `DotLift.dotAdd_step` from `hP.dotF16`,
  `hP.dotBF16`, or an fp8 fact as `hP.dotE5m2Lift`), then the loop by `DotLoop.accum` against the
  claim's tolerance `DotLoop.bound` (closed form `DotLoop.bound_le`). Claims whose tolerance follows
  the chunks use `TVA.Proof.Dot.Chunk8` and `Chunk16`. `Device.dotI8_eq`: an `int8` dot in range is
  exact. `tl.dot_scaled`: `Device.DotScaledAcc.addStep`.
- **Online softmax** (`TVA.Proof.Softmax`): the exact invariant (`Softmax.l_eq`, `Softmax.acc_eq`,
  `Softmax.ratio_eq`, `Softmax.wavg_blocks`), and with rounding the running bound (`Softmax.l_err`,
  `Softmax.acc_err`, one trip by `Softmax.step_err`) and the output (`Softmax.err_final_mono`).
  `Softmax.two_rpow_eq_exp` turns `exp2` into `exp`.
- **Decayed recurrences** (`TVA.Proof.Recurrence`): `S (t+1) = g t · S t + U t`; the closed form
  from a cumulative sum of gates (`Recurrence.closed_cumsum`) and the error of a computed state
  (`Recurrence.perturb_le_sum`, matrix forms `Recurrence.perturb_mat_le_sum`).

Check each tolerance at the largest covered instance before proving it: the bound must fit the
claim's constant there, and every intermediate must stay below `maxVal` ("room").

## 6. Atomics

- **Owned elements.** An atomic whose elements no other program touches, and whose result is
  unused (`let _ ← tl.atomic_add …`, or `tl.atomic_add_unused`), is a `Safe` step:
  `safe_atomic_buf`. For `tl.atomic_add`, unfold it first (it reads `enable_fp_fusion` through
  `.env`; the rule's docstring says how).
- **Shared cells** (split-K, stream-K, histograms; `TVA.Proof.Shared`). Use `correct_of_safeA`:
  1. prove each program `SafeA`: private parts with the `Safe` rules lifted by `SafeA.of_safe`, and
     each atomic on a shared cell with `SafeA.atomic_add` (or `SafeA.atomic`, `SafeA.atomic1`). The
     rule quantifies over every value the cell may hold and appends the steps to the program's log;
  2. the post-condition `Q i log m` says what the log holds (its increments, or `Paired` with the
     operands) without depending on its order (`PermInv`; `Paired.perm`), because a program's
     atomics may be performed in another order than the text's;
  3. in the last goal the cells hold the replay of all programs' steps in some interleaving:
     `replay_add` with `sum_proj` for integer additions, `replay_max`, `replay_min` for integer
     maxima, `replay_err_shared` (with `Paired.merge` and `accStep_atomicAdd` from
     `hP.atomicAddF32`) for float additions.
- An atomic whose result the program uses, on a cell anything else reaches, is outside every
  claim; neither `Safe` nor `SafeA` accepts a used result.

## 7. Loops

- `for i in range(lo, hi, step)` (`tl.forRange`): `safe_forRange`. Give an invariant `I t st m`
  over the trip number `t` (the induction variable is `lo + t * step`) and a write footprint
  `Wt t` per trip; trips write disjoint bytes (a conflict across iterations is a conflict within
  the program). The loop's footprint is `iterW Wt n`; a loop that writes nothing takes
  `Wt := fun _ => ∅` (`iterW_empty`). The bounds must satisfy `tl.RangeOk`, which `decide` or
  `omega` usually shows from `Inst.ok`.
- `tl.forN` (unrolled or fixed counts): `safe_forN`, `safe_forN_bind` (with the rest of the
  program after the loop).
- `while` (`tl.whileLoop`): `safe_while`, with an invariant and a variant that each iteration
  decreases.
- State the invariant with the loop's carried values *and* the memory, including what earlier
  trips stored (each trip's stores are in its `Wt t`, and later trips must not touch them).

## 8. A worked walk: `vec_add_f32`

The kernel (`claims/vec_add_f32/Claim.lean`): program `pid` loads `x` and `y` at
`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` masked by `offsets < n`, adds them with
`tl.add` and stores `output`. `Post` says that each `output[j]`, `j < n`, is `FloatV.add` of the
initial `x[j]`, `y[j]` up to NaN payload, and that every other byte is unchanged. The proof is
`sandbox/claim/oracle/vec_add_f32/Proof.lean`; its parts:

**The footprints.** Program `p` owns the elements `blk d p = {j | p·B ≤ j < p·B + B, j < n}`. It
reads `R d p = d.x.elems (blk d p) ∪ d.y.elems (blk d p)` and writes `W d p = d.output.elems (blk d
p)`; `Q d init p m` says each element of its block of `output` in `m` is the sum of the initial `x`
and `y`. These are decided before any rule.

**The integer bound** (`start_bound`). With `B = 2^e ≤ 2^20` and `n < 2^31`, a program `p` of the
grid `cdiv(n, B)` has `p·B < n` and `p·B + B ≤ 2^31`: the `int32` offsets do not wrap.

**One program** (`safe_program`). After `simp only [model, add_kernel]` the goal is the `do` block.
The proof first computes the lanes, with the tracked-value lemmas:

```lean
  have hpid : (tl.program_id c 0).toInt = (p : Int) := by
    rw [tl.program_id, hc]; exact IntV.I32.toInt_ofNat (by omega)
  have hstart : (tl.program_id c 0 * (IntV.ofNat d.BLOCK_SIZE : I32)).toInt = ((p * d.BLOCK_SIZE : Nat) : Int) :=
    IntV.I32.mul_nat hpid (IntV.I32.toInt_ofNat (by omega)) (by omega)
  have hoff : ∀ i, ((… + tl.arange 0 d.BLOCK_SIZE).get i).toInt = ((g i : Nat) : Int) := by
    rintro ⟨i, ⟨⟩⟩; exact IntV.I32.toInt_sadd_arange hstart (by omega) i
```

so lane `i` addresses element `g i = p·B + i`, enabled (`en i`) when `g i < n` (`hmask`, by
`Tensor.get_ltS` and `IntV.lt_eq`), and its pointer is `b.ptr (g i)` (`hptr`, by `Buf.ptr_offset`).
Then one rule per line:

```lean
  refine safe_load_buf d.x (Buf.addrs_of_get d.x hmask (fun i _ => hptr d.x i)) (hin d.x hd.x_len)
    hvx (fun z hz => Or.inl (d.x.elems_mono himg hz)) fun vx hx _ => ?_
  refine safe_load_buf d.y (…) fun vy hy _ => ?_
  refine safe_add fun vo hvadd => ?_
  refine safe_last ?_
  refine safe_store_buf d.output (…) ?_ hvo (d.output.elems_mono himg) fun m' hm' _ _ => safe_ret ?_
```

Each load's bytes lie in `R` (`himg`: the enabled lanes' elements are in `blk d p`), the store's in
`W`; the store needs distinct elements per lane (the first side goal, `g` is injective), and
`safe_ret`'s goal is `Q`: for `j ∈ blk d p`, take lane `j − p·B` and combine `hx`, `hy`, `hm'` and
`hvadd` (`tl.add`'s result is `NaNEq` to `FloatV.add`).

**The launch** (`main`). After `intro P _ d hd init hpre`, `correct_of_safe` with `R d`, `W d`,
`Q d init`:

- validity: `Launch.valid_grid1`, from `n < 2^31` and the block shape `[B]`;
- program `i`: `safe_program` at `p = i` (`Launch.ctx` of `grid1` gives `pid 0 = i`);
- footprints: blocks of distinct programs are disjoint (`blk_disjoint`, by `j / B`), so
  `Buf.elems_disjoint` separates the outputs, and `Inst.ok`'s `output_x`, `output_y` (with
  `Disjoint.mono`) separate output from inputs;
- `Post`: element `j < n` is in the block of program `j / B` (`mem_blk`); take that program's
  memory from the hypothesis and move the read with `Mem.read_congr` and `Buf.span_sub_elems`.
  Bytes outside every `W i` equal `init`'s directly.

The proof uses no device fact: the only float operation is an IEEE `+`.

## 9. Lessons

- **Search before you write.** Most of what a kernel needs is a lemma of the kit already: grep
  `lean/TVA/Proof` for the operation's name, and read the module docstrings, which say when to use
  each theorem.
- **Prove a `body_eq` once.** Unfolding a large model with `simp` can be slow and fragile; a lemma
  `(model d).body c = …` proved by `rfl` fixes the form you walk.
- **Split by mode, not by copy.** When the kernel branches on a flag (variable length, a mask, a
  dtype), walk the body once and split where the code differs.
- **Keep lemmas small.** State one lemma per program part (the loads, the loop, the epilogue), each
  with its own footprints, and compose them with `Safe.bind`. Long single `refine` chains time out.
- **Masked lanes.** A masked load without `other` gives arbitrary values on the disabled lanes; a
  proof may use nothing about them, and the specification must not depend on them.
- **NaN.** Keep every float statement up to NaN payload until finiteness is proved, then use
  `FloatV.NaNEq.exact_of_isFinite`.
- **`#print axioms main`** before stopping: only `propext`, `Classical.choice`, `Quot.sound`.
