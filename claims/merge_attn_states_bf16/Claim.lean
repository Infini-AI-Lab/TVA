import TVA

/-!
# `merge_attn_states_bf16`: the claim

Source: `source/merge_attn_states.py`. The kernel is
`merge_attn_states_kernel` (lines 65–216); it reads the module-level names `triton` and `tl`
(lines 61–62). The host wrapper `merge_attn_states` (lines 221–299) launches it on the grid
`(num_tokens, num_heads)` with Triton's default launch options.

The covered paths are the ones `SETUP.md` asks for: `OUTPUT_LSE` either value (`return_lse`
`True` or `False`, `Inst.outLse`) and `USE_FP8 = False` (no `output_scale`, a `bfloat16` output),
so the compile-time `if`s of lines 123 and 204 are decided, and those of lines 107 and 167 follow
`Inst.outLse`.

* The model: `merge_attn_states_kernel`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with the reference `refLse`, `refOut` and the tolerances
  `lseTol`, `bf16Round`, `outTol`, `absTol`.
* The assumptions: `Assumptions`.
* The claim: `Statement`.
* Checked here: `consistent`, `satisfiable`, and the coverage facts `setup_*_ok`.
-/

namespace Claims.MergeAttnStates

open TVA TVA.Math.Float

/-! ## The model -/

/-- `float("-inf")`, the `float32` constant Triton builds for the Python double `-∞` (`litF32` of
its pattern; SEMANTICS.md, "Constants"). -/
def negInfC : F32 := FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)

/-- `float("inf")`, the `float32` constant of the Python double `+∞`. -/
def posInfC : F32 := FloatV.litF32 (FloatV.ofPat 0x7FF0000000000000)

/-- `0.0` of line 202, built in `float32` beside the `float32` block `out`. -/
def zeroC : F32 := FloatV.litF32 (FloatV.ofPat 0)

set_option linter.unusedVariables false in
/-- `merge_attn_states_kernel` (source lines 65–216). The pointers are `*bfloat16` (`output`,
`prefix_output`, `suffix_output`) and `*float32` (`output_lse`, `prefix_lse`, `suffix_lse`); the
eight strides and `prefill_tokens_with_context` are `int32` arguments (an argument equal to `1`
is the compile-time constant `1`, which each use builds in `int32`: the same model). `HEAD_SIZE`
and `PADDED_HEAD_SIZE` are the `tl.constexpr` parameters. `output_lse` is `none` when the host
passes `None` (`return_lse=False`) and `some p` for a tensor; the host passes the constexpr
`OUTPUT_LSE = output_lse is not None` (line 294), so the model reads `OUTPUT_LSE` off
`output_lse`: `some` is `True`, `none` is `False`. `USE_FP8 = False` is decided, and
`output_scale`, `FP8_MIN`, `FP8_MAX` are read only in the untaken `USE_FP8` branches. -/
def merge_attn_states_kernel
    (output : Ptr .bf16) (output_lse : Option (Ptr .f32))
    (prefix_output : Ptr .bf16) (prefix_lse : Ptr .f32)
    (suffix_output : Ptr .bf16) (suffix_lse : Ptr .f32)
    (prefix_head_stride output_head_stride prefix_lse_head_stride prefix_lse_token_stride
      suffix_lse_head_stride suffix_lse_token_stride output_lse_head_stride
      output_lse_token_stride prefill_tokens_with_context : I32)
    (HEAD_SIZE PADDED_HEAD_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 90: axis 0 of the grid `(num_tokens, num_heads)`, an `int32`
    let token_idx := tl.program_id ctx 0
    -- line 91: axis 1
    let head_idx := tl.program_id ctx 1
    -- line 92: the grid's size along axis 1
    let num_heads := tl.num_programs ctx 1
    -- line 94: two `int32` scalars compared signed; an `int1` scalar
    let prefix_mask := ((bc token_idx).ltS prefill_tokens_with_context).item
    -- line 96
    let head_arange := tl.arange 0 PADDED_HEAD_SIZE
    -- line 97: the constexpr `HEAD_SIZE` built in `int32`, signed comparison
    let head_mask := head_arange.ltS (IntV.ofNat HEAD_SIZE : I32)
    -- line 101: a runtime `if` on the `int1` scalar (`not` of it); the branch ends at line 136's
    -- `return`, so the rest of the body is its `else`
    if !prefix_mask then
      -- lines 102–106: one `float32` element
      let s_lse ← tl.load1 (suffix_lse + head_idx * suffix_lse_head_stride
        + token_idx * suffix_lse_token_stride)
      -- lines 107–113: `if OUTPUT_LSE:`, a compile-time `if` (`OUTPUT_LSE` is
      -- `output_lse is not None`): with `None` not visited; with a tensor the element is stored,
      -- already the pointer's dtype
      match output_lse with
      | none => pure ()
      | some output_lse =>
        tl.store1 (output_lse + head_idx * output_lse_head_stride
          + token_idx * output_lse_token_stride) s_lse
      -- lines 115–121: masked lanes unspecified (no `other=`)
      let s_out : Tensor [PADDED_HEAD_SIZE] BF16 ←
        tl.load (suffix_output + token_idx * num_heads * prefix_head_stride
          + head_idx * prefix_head_stride + head_arange) (some head_mask)
      -- lines 123–126: `USE_FP8` is `False` (compile time): not visited
      -- lines 128–135: already `bfloat16`, the pointer's dtype
      tl.store (output + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride + head_arange) s_out (some head_mask)
      -- line 136: `return`
    else
      -- lines 140–144
      let p_lse ← tl.load1 (prefix_lse + head_idx * prefix_lse_head_stride
        + token_idx * prefix_lse_token_stride)
      -- lines 145–149
      let s_lse ← tl.load1 (suffix_lse + head_idx * suffix_lse_head_stride
        + token_idx * suffix_lse_token_stride)
      -- lines 156–157: conditional expressions on a runtime condition: a runtime `if` whose
      -- value is a `float32` scalar (SEMANTICS.md, "Order"); the comparison is with the
      -- `float32` `+∞`, the taken side is the `float32` `-∞`
      let p_lse := if ((bc p_lse).feq (bc posInfC)).item then negInfC else p_lse
      let s_lse := if ((bc s_lse).feq (bc posInfC)).item then negInfC else s_lse
      -- line 159: `tl.maximum` of two `float32` scalars (`MaxRel`)
      let max_lse := (← tl.maximum (bc p_lse) (bc s_lse)).item
      -- lines 160–161: the IEEE `-` (no product feeds it)
      let p_lse := (← tl.sub (bc p_lse) (bc max_lse)).item
      let s_lse := (← tl.sub (bc s_lse) (bc max_lse)).item
      -- lines 163–164: the platform's `tl.exp` on `float32`
      let p_se := (← tl.exp (bc p_lse)).item
      let s_se := (← tl.exp (bc s_lse)).item
      -- line 165: the IEEE `+` (its operands are `tl.exp` results, not products)
      let out_se := (← tl.add (bc p_se) (bc s_se)).item
      -- line 167: `if OUTPUT_LSE:`, a compile-time `if`: with `None` lines 168–177 are not
      -- visited
      match output_lse with
      | none => pure ()
      | some output_lse =>
        -- line 168: `tl.log` on `float32`, then the IEEE `+` (no product feeds it)
        let out_lse := (← tl.log (bc out_se)).item
        let out_lse := (← tl.add (bc out_lse) (bc max_lse)).item
        -- line 171: `tl.where` of two `float32` scalars; the condition compares with the
        -- `float32` `-∞`, the `-∞` side is the `float32` constant
        let out_lse := (Tensor.where_ ((bc max_lse).feq (bc negInfC)) (bc negInfC)
          (bc out_lse) : Tensor [] F32).item
        -- lines 172–177
        tl.store1 (output_lse + head_idx * output_lse_head_stride
          + token_idx * output_lse_token_stride) out_lse
      -- lines 179–185: masked lanes unspecified (no `other=`)
      let p_out : Tensor [PADDED_HEAD_SIZE] BF16 ←
        tl.load (prefix_output + token_idx * num_heads * prefix_head_stride
          + head_idx * prefix_head_stride + head_arange) (some head_mask)
      -- lines 186–192
      let s_out : Tensor [PADDED_HEAD_SIZE] BF16 ←
        tl.load (suffix_output + token_idx * num_heads * prefix_head_stride
          + head_idx * prefix_head_stride + head_arange) (some head_mask)
      -- lines 197–198: `float32 / float32`, the platform's division
      let p_scale := (← tl.div (bc p_se) (bc out_se)).item
      let s_scale := (← tl.div (bc s_se) (bc out_se)).item
      -- line 199: the `bfloat16` blocks promote to `float32` (exact), the IEEE products; the
      -- `+` has a product on each side: the fusable `tl.addc`
      let p32 : Tensor [PADDED_HEAD_SIZE] F32 ← tl.cvt p_out
      let s32 : Tensor [PADDED_HEAD_SIZE] F32 ← tl.cvt s_out
      let prod_p ← tl.mul p32 (bc p_scale)
      let prod_s ← tl.mul s32 (bc s_scale)
      let out32 ← tl.addc prod_p prod_s
      -- line 202: `tl.where` in `float32`; the condition, a scalar, broadcast to the block; the
      -- `0.0` built in `float32`
      let out32 := Tensor.where_ (bc ((bc max_lse).feq (bc negInfC)).item) (bc zeroC) out32
      -- lines 209–216: the store's conversion to the pointer's `bfloat16` (nearest even)
      let out16 : Tensor [PADDED_HEAD_SIZE] BF16 ← tl.cvt out32
      tl.store (output + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride + head_arange) out16 (some head_mask)

/-! ## The conditions -/

/-- What the host chooses: one launch of `merge_attn_states_kernel` through
`merge_attn_states(prefix_output, prefix_lse, suffix_output, suffix_lse, return_lse,
prefill_tokens_with_context)` (no `output_scale`). Each tensor is its storage (`Buf`) and its own
strides in elements. -/
structure Inst where
  /-- `return_lse`: the host allocates `output_lse` and passes it (`OUTPUT_LSE = True`), or
  passes `None` (`OUTPUT_LSE = False`), as the backends' calls without `output_lse` do. -/
  outLse : Bool
  /-- `num_tokens` (`output.shape[0]`): the grid's `x`. -/
  T : Nat
  /-- `num_query_heads` (`output.shape[1]`): the grid's `y`. -/
  H : Nat
  /-- `head_size` (`output.shape[2]`), the constexpr `HEAD_SIZE`. -/
  head_size : Nat
  /-- `padded_head_size` (`triton.next_power_of_2(head_size)`), the constexpr
  `PADDED_HEAD_SIZE`. -/
  padded : Nat
  /-- `prefill_tokens_with_context`, after the host's `None ↦ num_tokens` (lines 272–273). -/
  ptwc : Nat
  /-- `prefix_output.stride(0)`, `prefix_output.stride(1)` (the last dimension contiguous). -/
  pots : Nat
  phs : Nat
  /-- `suffix_output.stride(0)`, `suffix_output.stride(1)`. -/
  sots : Nat
  shs : Nat
  /-- `output.stride(0)`, `output.stride(1)`. -/
  ots : Nat
  ohs : Nat
  /-- `prefix_lse.stride(0)` (head), `prefix_lse.stride(1)` (token). -/
  plhs : Nat
  plts : Nat
  /-- `suffix_lse.stride(0)`, `suffix_lse.stride(1)`. -/
  slhs : Nat
  slts : Nat
  /-- `output_lse.stride(0)`, `output_lse.stride(1)` (the host passes `0`, `0` without
  `output_lse`, lines 268–269; the kernel then reads neither). -/
  olhs : Nat
  olts : Nat
  /-- `output`: `bfloat16`, `[num_tokens, num_heads, head_size]`. -/
  output : Buf .bf16
  /-- `output_lse`: `float32`, `[num_heads, num_tokens]`; with `outLse = false` the host passes
  `None` and this buffer has no role. -/
  output_lse : Buf .f32
  /-- `prefix_output`: `bfloat16`, `[num_tokens, num_heads, head_size]`. -/
  prefix_output : Buf .bf16
  /-- `prefix_lse`: `float32`, `[num_heads, num_tokens]`. -/
  prefix_lse : Buf .f32
  /-- `suffix_output`: `bfloat16`, `[num_tokens, num_heads, head_size]`. -/
  suffix_output : Buf .bf16
  /-- `suffix_lse`: `float32`, `[num_heads, num_tokens]`. -/
  suffix_lse : Buf .f32

/-- Element `[t, h, j]` of `prefix_output`, by its own strides. -/
def poIdx (d : Inst) (t h j : Nat) : Nat := t * d.pots + h * d.phs + j
/-- Element `[t, h, j]` of `suffix_output`. -/
def soIdx (d : Inst) (t h j : Nat) : Nat := t * d.sots + h * d.shs + j
/-- Element `[t, h, j]` of `output`. -/
def oIdx (d : Inst) (t h j : Nat) : Nat := t * d.ots + h * d.ohs + j
/-- Element `[h, t]` of `prefix_lse`. -/
def plIdx (d : Inst) (t h : Nat) : Nat := h * d.plhs + t * d.plts
/-- Element `[h, t]` of `suffix_lse`. -/
def slIdx (d : Inst) (t h : Nat) : Nat := h * d.slhs + t * d.slts
/-- Element `[h, t]` of `output_lse`. -/
def olIdx (d : Inst) (t h : Nat) : Nat := h * d.olhs + t * d.olts

/-- The `output` elements the launch writes: `[t, h, j]`, `j < head_size`. -/
def outSet (d : Inst) : Set Nat :=
  {k | ∃ t h j, t < d.T ∧ h < d.H ∧ j < d.head_size ∧ k = oIdx d t h j}

/-- The `output_lse` elements the launch writes. -/
def outLseSet (d : Inst) : Set Nat := {k | ∃ t h, t < d.T ∧ h < d.H ∧ k = olIdx d t h}

/-- The bytes the launch writes: `output`'s elements, and `output_lse`'s when it is passed. -/
def writeSet (d : Inst) : Set Int :=
  d.output.elems (outSet d) ∪ (if d.outLse then d.output_lse.elems (outLseSet d) else ∅)

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `PADDED_HEAD_SIZE` is a power of two (Triton requires it of `tl.arange`), at most `2^20`
  (Triton's largest block), and at least `HEAD_SIZE` (the host's `next_power_of_2`). -/
  padded_pow2 : ∃ e, d.padded = 2 ^ e
  padded_le : d.padded ≤ 2 ^ 20
  head_le : d.head_size ≤ d.padded
  /-- `prefill_tokens_with_context ∈ [0, num_tokens]` (SETUP.md; `None` is `num_tokens`). -/
  ptwc_le : d.ptwc ≤ d.T
  /-- The grid `(num_tokens, num_heads)` within CUDA's limits, fewer than `2^31` programs. -/
  T_lt : d.T < 2 ^ 31
  /-- `num_tokens`, the tokens of the batch, is a context length: at most `2^22`. -/
  T_ctx : d.T ≤ 2 ^ 22
  H_lt : d.H < 2 ^ 16
  grid_lt : d.T * d.H < 2 ^ 31
  /-- The layouts the kernel indexes (SETUP.md): `prefix_output` and `suffix_output` share one
  head stride (the host asserts it, lines 233–237), and each output-like tensor's token stride is
  `num_heads` times its head stride (the kernel computes `token_idx * num_heads * head_stride`). -/
  shs_eq : d.shs = d.phs
  pots_eq : d.pots = d.H * d.phs
  sots_eq : d.sots = d.H * d.shs
  ots_eq : d.ots = d.H * d.ohs
  /-- The output's head stride is at least `head_size`, so distinct `[t, h, j]` are distinct
  elements (the host's fresh `output` has `ohs = head_size`). -/
  ohs_ge : d.head_size ≤ d.ohs
  /-- `output_lse`, when passed, is laid out row-major in one order or the other, so distinct
  `[h, t]` are distinct elements (the host's fresh `torch.empty` has strides `(num_tokens, 1)`). -/
  olse_layout : d.outLse →
    (1 ≤ d.olts ∧ d.T * d.olts ≤ d.olhs) ∨ (1 ≤ d.olhs ∧ d.H * d.olhs ≤ d.olts)
  /-- The eight strides the kernel receives are `int32` arguments. -/
  strides_lt : d.phs < 2 ^ 31 ∧ d.ohs < 2 ^ 31 ∧ d.plhs < 2 ^ 31 ∧ d.plts < 2 ^ 31 ∧
    d.slhs < 2 ^ 31 ∧ d.slts < 2 ^ 31 ∧ d.olhs < 2 ^ 31 ∧ d.olts < 2 ^ 31
  /-- **A limit of the kernel**: its `int32` products `token_idx * num_heads * head_stride`,
  `head_idx * head_stride` (lines 117–118, 130–131, 181–182, 188–189, 211–212) and
  `head_idx * lse_head_stride`, `token_idx * lse_token_stride` (lines 104–105, 110–111, 142–143,
  147–148, 174–175) do not wrap. The prefix's are formed only by the merged tokens. -/
  out_i32 : (d.T - 1) * d.H * d.ohs < 2 ^ 31 ∧ (d.H - 1) * d.ohs < 2 ^ 31
  in_i32 : (d.T - 1) * d.H * d.phs < 2 ^ 31 ∧ (d.H - 1) * d.phs < 2 ^ 31
  lse_i32 : (0 < d.ptwc → (d.H - 1) * d.plhs < 2 ^ 31 ∧ (d.ptwc - 1) * d.plts < 2 ^ 31) ∧
    (d.H - 1) * d.slhs < 2 ^ 31 ∧ (d.T - 1) * d.slts < 2 ^ 31 ∧
    (d.H - 1) * d.olhs < 2 ^ 31 ∧ (d.T - 1) * d.olts < 2 ^ 31
  /-- Each buffer holds every element the launch touches (its last one), the prefix's only for
  the merged tokens `t < prefill_tokens_with_context`. -/
  output_len : 0 < d.T → 0 < d.H → 0 < d.head_size →
    (d.T - 1) * d.ots + (d.H - 1) * d.ohs + d.head_size ≤ d.output.len
  suffix_output_len : 0 < d.T → 0 < d.H → 0 < d.head_size →
    (d.T - 1) * d.sots + (d.H - 1) * d.shs + d.head_size ≤ d.suffix_output.len
  prefix_output_len : 0 < d.ptwc → 0 < d.H → 0 < d.head_size →
    (d.ptwc - 1) * d.pots + (d.H - 1) * d.phs + d.head_size ≤ d.prefix_output.len
  output_lse_len : d.outLse → 0 < d.T → 0 < d.H →
    (d.H - 1) * d.olhs + (d.T - 1) * d.olts < d.output_lse.len
  suffix_lse_len : 0 < d.T → 0 < d.H →
    (d.H - 1) * d.slhs + (d.T - 1) * d.slts < d.suffix_lse.len
  prefix_lse_len : 0 < d.ptwc → 0 < d.H →
    (d.H - 1) * d.plhs + (d.ptwc - 1) * d.plts < d.prefix_lse.len
  /-- The outputs share no byte with each other or with the four inputs (the host allocates
  them fresh, lines 240–245). The inputs may overlap each other. -/
  output_lse_disj : d.outLse → Disjoint d.output.bytes d.output_lse.bytes
  output_disj : Disjoint d.output.bytes
    (d.prefix_output.bytes ∪ d.suffix_output.bytes ∪ d.prefix_lse.bytes ∪ d.suffix_lse.bytes)
  output_lse_in_disj : d.outLse → Disjoint d.output_lse.bytes
    (d.prefix_output.bytes ∪ d.suffix_output.bytes ∪ d.prefix_lse.bytes ∪ d.suffix_lse.bytes)

/-- The model on an instance: the buffers' first elements (`output_lse` `none` without
`return_lse`), the eight strides and `prefill_tokens_with_context` as `int32`, the two
constexprs. -/
def model (d : Inst) : Kernel :=
  merge_attn_states_kernel (d.output.ptr 0) (if d.outLse then some (d.output_lse.ptr 0) else none)
    (d.prefix_output.ptr 0) (d.prefix_lse.ptr 0) (d.suffix_output.ptr 0) (d.suffix_lse.ptr 0)
    (IntV.ofNat d.phs) (IntV.ofNat d.ohs) (IntV.ofNat d.plhs) (IntV.ofNat d.plts)
    (IntV.ofNat d.slhs) (IntV.ofNat d.slts) (IntV.ofNat d.olhs) (IntV.ofNat d.olts)
    (IntV.ofNat d.ptwc) d.head_size d.padded

/-- The host's launch (line 275): the grid `(num_tokens, num_heads, 1)`; the source's only block
shape `[PADDED_HEAD_SIZE]`; `enable_fp_fusion` Triton's default (`true`); `num_ctas = 1`, no
`launch_pdl`, `debug = false` (`num_warps = 4`, `num_stages = 3` change no meaning). -/
def launch (d : Inst) : Launch :=
  { grid := ![d.T, d.H, 1], fusion := true, blocks := [[d.padded]] }

/-- The `float32` value at element `i` of a buffer, if present. -/
def rdF (m : Mem) (b : Buf .f32) (i : Nat) : Option F32 := m.read .f32 (b.ptr (i : Int)).addr

/-- The `bfloat16` value at element `i` of a buffer, if present. -/
def rdB (m : Mem) (b : Buf .bf16) (i : Nat) : Option BF16 := m.read .bf16 (b.ptr (i : Int)).addr

/-- An lse value read as the source reads it (lines 156–157): `+∞` (FlashAttention-2's empty
side) is `-∞`. -/
def corr (x : F32) : F32 := if x = posInfC then negInfC else x

/-- Both sides of a merged `(t, h)` are empty: both lse values, read as `corr` reads them, are
`-∞`. -/
def bothEmpty (p s : F32) : Prop := corr p = negInfC ∧ corr s = negInfC

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte
present; `output_lse` only when passed); for each merged `(t, h)` (`t < prefill_tokens_with_context`) the two lse values are not
NaN (finite, `+∞` or `-∞`), and, unless both sides are empty, the two `bfloat16` output rows are
finite. Nothing is required of the copied tokens' values. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.output.Valid init ∧ (d.outLse → d.output_lse.Valid init) ∧ d.prefix_output.Valid init ∧
    d.prefix_lse.Valid init ∧ d.suffix_output.Valid init ∧ d.suffix_lse.Valid init ∧
    ∀ t h, t < d.ptwc → h < d.H → ∃ p s : F32,
      rdF init d.prefix_lse (plIdx d t h) = some p ∧ rdF init d.suffix_lse (slIdx d t h) = some s ∧
      p.isNaN = false ∧ s.isNaN = false ∧
      (¬ bothEmpty p s → ∀ j, j < d.head_size → ∃ pv sv : BF16,
        rdB init d.prefix_output (poIdx d t h j) = some pv ∧
        rdB init d.suffix_output (soIdx d t h j) = some sv ∧
        pv.isFinite = true ∧ sv.isFinite = true)

/-! ## The specification -/

/-- The larger of the two lse values (`corr` applied), as a real; an empty side does not count.
Only used when not both sides are empty. -/
noncomputable def refMax (p s : F32) : ℝ :=
  if p = negInfC then s.toReal else if s = negInfC then p.toReal else max p.toReal s.toReal

/-- One side's weight `e^(x − m)`, `0` for an empty side. -/
noncomputable def side (x : F32) (m : ℝ) : ℝ := if x = negInfC then 0 else Real.exp (x.toReal - m)

/-- The exact normaliser `e^(p−m) + e^(s−m)`. -/
noncomputable def refSum (p s : F32) : ℝ := side p (refMax p s) + side s (refMax p s)

/-- The exact merged lse, `log(e^(p−m) + e^(s−m)) + m` (`log(e^p + e^s)`), of the `corr`ected
values. -/
noncomputable def refLse (p s : F32) : ℝ := Real.log (refSum p s) + refMax p s

/-- The exact merged output element, `(pv·e^(p−m) + sv·e^(s−m)) / (e^(p−m) + e^(s−m))`, of the
`corr`ected lse values and the two `bfloat16` inputs. -/
noncomputable def refOut (p s : F32) (pv sv : ℝ) : ℝ :=
  (pv * side p (refMax p s) + sv * side s (refMax p s)) / refSum p s

/-- The lse tolerance, relative to `1 + |lse|`: `2^-16`. -/
def lseTol : ℚ := 2 ^ (-16 : ℤ)

/-- One rounding to `bfloat16`, relative: `2^-8`. -/
def bf16Round : ℚ := 2 ^ (-8 : ℤ)

/-- The output tolerance relative to the inputs' magnitudes `|pv| + |sv|`: `2^-14`. -/
def outTol : ℚ := 2 ^ (-14 : ℤ)

/-- The output's absolute floor: `2^-133`, the smallest `bfloat16` subnormal. -/
def absTol : ℚ := 2 ^ (-133 : ℤ)

/-- The `bfloat16` `+0`. -/
def zeroB : BF16 := FloatV.ofPat 0

/-- What a merged `(t, h)` leaves in `fin`: with `p`, `s` the initial lse values,
* when `output_lse` is passed, the lse element: exactly `-∞` when both sides are empty, otherwise finite and within
  `lseTol·(1 + |refLse|)` of `refLse (corr p) (corr s)`;
* each output element `j < head_size`: exactly `+0` when both sides are empty, otherwise finite
  and within `bf16Round·|refOut| + outTol·(|pv| + |sv|) + absTol` of `refOut`, with `pv`, `sv` the
  initial `prefix_output[t, h, j]`, `suffix_output[t, h, j]`. -/
def MergeOk (d : Inst) (init fin : Mem) (t h : Nat) : Prop :=
  ∃ p s : F32,
    rdF init d.prefix_lse (plIdx d t h) = some p ∧ rdF init d.suffix_lse (slIdx d t h) = some s ∧
    (d.outLse → ∃ cl : F32, rdF fin d.output_lse (olIdx d t h) = some cl ∧
      (bothEmpty p s → cl = negInfC) ∧
      (¬ bothEmpty p s → cl.isFinite = true ∧
        |cl.toReal - refLse (corr p) (corr s)| ≤
          (lseTol : ℝ) * (1 + |refLse (corr p) (corr s)|))) ∧
    ∀ j, j < d.head_size → ∃ w : BF16, rdB fin d.output (oIdx d t h j) = some w ∧
      (bothEmpty p s → w = zeroB) ∧
      (¬ bothEmpty p s → ∃ pv sv : BF16,
        rdB init d.prefix_output (poIdx d t h j) = some pv ∧
        rdB init d.suffix_output (soIdx d t h j) = some sv ∧
        w.isFinite = true ∧
        |w.toReal - refOut (corr p) (corr s) pv.toReal sv.toReal| ≤
          (bf16Round : ℝ) * |refOut (corr p) (corr s) pv.toReal sv.toReal| +
            (outTol : ℝ) * (|pv.toReal| + |sv.toReal|) + (absTol : ℝ))

/-- What a copied `(t, h)` (`t ≥ prefill_tokens_with_context`) leaves: the suffix's lse element
(when `output_lse` is passed) and its `head_size` output elements, bit for bit. -/
def CopyOk (d : Inst) (init fin : Mem) (t h : Nat) : Prop :=
  (d.outLse → ∃ s : F32, rdF init d.suffix_lse (slIdx d t h) = some s ∧
    rdF fin d.output_lse (olIdx d t h) = some s) ∧
  ∀ j, j < d.head_size → ∃ v : BF16, rdB init d.suffix_output (soIdx d t h j) = some v ∧
    rdB fin d.output (oIdx d t h j) = some v

/-- What the launch leaves: the merge for each token with context, the copy for each token
without, and every byte outside the written elements unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t h, t < d.ptwc → h < d.H → MergeOk d init fin t h) ∧
  (∀ t h, d.ptwc ≤ t → t < d.T → h < d.H → CopyOk d init fin t h) ∧
  ∀ z, z ∉ writeSet d → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.MathTotal`: `tl.exp` on `float32` has a result for every input (lines
  163–164; a NaN input arises when both sides are empty, `-∞ − (−∞)`). -/
  exp_total : Rules.Num.MathTotal P .exp .binary32
  /-- [rule] `Num.ExpF32`: the accuracy of `tl.exp` on `float32` (lines 163–164). -/
  exp_acc : Rules.Num.ExpF32 P
  /-- [rule] `Num.ExpF32NegInf`: `tl.exp(-∞) = +0` on `float32` (an empty side). -/
  exp_ninf : Rules.Num.ExpF32NegInf P
  /-- [rule] `Num.MathTotal`: `tl.log` on `float32` has a result for every input (line 168). -/
  log_total : Rules.Num.MathTotal P .log .binary32
  /-- [rule] `Num.LogF32`: the accuracy of `tl.log` on `float32` (line 168). -/
  log_acc : Rules.Num.LogF32 P
  /-- [rule] `Num.DivTotal`: `/` on `float32` has a result for every pair of inputs (lines
  197–198). -/
  div_total : Rules.Num.DivTotal P .binary32
  /-- [rule] `Num.DivFull`: `/` on `float32` is `div.full.f32` (lines 197–198). -/
  div_full : Rules.Num.DivFull P

/-- **The claim to prove.** -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.ref.mathTotal,
    Platform.sm90ref.expF32, Platform.sm90ref.expF32NegInf, Platform.ref.mathTotal,
    Platform.sm90ref.logF32, Platform.ref.divTotal _, Platform.ref.divFull⟩⟩

/-! ## The host's layouts

`canon T H hs padded ptwc pad lseT outLse` is the launch the host makes for `num_tokens = T`,
`num_heads = H`, `head_size = hs` (`padded = next_power_of_2(hs)`), `prefill_tokens_with_context
= ptwc` (`None` is `T`), the inputs' heads padded by `pad` elements (the input tensors
`[T, H, hs + pad]` sliced to `[..., :hs]`) and the lse tensors either `[H, T]` contiguous or the
transposes of `[T, H]` tensors (`lseT`), and `return_lse = outLse`. The host allocates `output`
(`[T, H, hs]`) contiguous, and `output_lse` (`[H, T]`) contiguous when `return_lse`; without it
the host passes `None` and the strides `0`, `0` (the buffer at `2·2^36` then has no role). The six buffers are the tensors' storages, at the addresses
`2^36`, `2·2^36`, …, `6·2^36`. -/

def canon (T H hs padded ptwc pad : Nat) (lseT outLse : Bool) : Inst where
  outLse := outLse
  T := T
  H := H
  head_size := hs
  padded := padded
  ptwc := ptwc
  pots := H * (hs + pad)
  phs := hs + pad
  sots := H * (hs + pad)
  shs := hs + pad
  ots := H * hs
  ohs := hs
  plhs := if lseT then 1 else T
  plts := if lseT then H else 1
  slhs := if lseT then 1 else T
  slts := if lseT then H else 1
  olhs := if outLse then T else 0
  olts := if outLse then 1 else 0
  output := ⟨2 ^ 36, T * H * hs⟩
  output_lse := ⟨2 * 2 ^ 36, H * T⟩
  prefix_output := ⟨3 * 2 ^ 36, T * H * (hs + pad)⟩
  prefix_lse := ⟨4 * 2 ^ 36, H * T⟩
  suffix_output := ⟨5 * 2 ^ 36, T * H * (hs + pad)⟩
  suffix_lse := ⟨6 * 2 ^ 36, H * T⟩

/-- Two buffers whose byte ranges do not meet are disjoint. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (h : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem lt_of_le_lt {a b c : Nat} (h1 : a ≤ b) (h2 : b < c) : a < c := lt_of_le_of_lt h1 h2

/-- **Every host launch at the code base's sizes is covered**: up to `2^16` tokens, up to `64`
heads, head sizes up to `256` padded by up to `64`, any `prefill_tokens_with_context` in
`[0, num_tokens]`, either lse layout, with or without `output_lse`. -/
theorem canon_ok (T H hs padded ptwc pad e : Nat) (lseT outLse : Bool)
    (hT : T ≤ 2 ^ 16) (hH : H ≤ 64) (hhs : hs ≤ padded) (hp : padded = 2 ^ e)
    (hpl : padded ≤ 256) (hpad : pad ≤ 64) (hptwc : ptwc ≤ T) :
    (canon T H hs padded ptwc pad lseT outLse).ok := by
  have hs256 : hs ≤ 256 := le_trans hhs hpl
  have hTH : T * H ≤ 2 ^ 16 * 64 := Nat.mul_le_mul hT hH
  have hhp : hs + pad ≤ 320 := by omega
  have hTm : T - 1 ≤ 2 ^ 16 := le_trans (Nat.sub_le _ _) hT
  have hT1H : (T - 1) * H ≤ 2 ^ 16 * 64 := Nat.mul_le_mul hTm hH
  have hTHs : (T - 1) * H * (hs + pad) ≤ 2 ^ 16 * 64 * 320 := Nat.mul_le_mul hT1H hhp
  have hTHo : (T - 1) * H * hs ≤ 2 ^ 16 * 64 * 256 := Nat.mul_le_mul hT1H hs256
  have hHm : H - 1 ≤ 64 := le_trans (Nat.sub_le _ _) hH
  have hHp : (H - 1) * (hs + pad) ≤ 64 * 320 := Nat.mul_le_mul hHm hhp
  have hHo : (H - 1) * hs ≤ 64 * 256 := Nat.mul_le_mul hHm hs256
  have hHT : (H - 1) * T ≤ 64 * 2 ^ 16 := Nat.mul_le_mul hHm hT
  have hTH' : (T - 1) * H ≤ 2 ^ 16 * 64 := hT1H
  have hTlen : H * T ≤ 64 * 2 ^ 16 := Nat.mul_le_mul hH hT
  have hlenO : T * H * hs ≤ 2 ^ 16 * 64 * 256 := Nat.mul_le_mul hTH hs256
  have hlenP : T * H * (hs + pad) ≤ 2 ^ 16 * 64 * 320 := Nat.mul_le_mul hTH hhp
  refine
    { padded_pow2 := ⟨e, hp⟩
      padded_le := by simp only [canon]; omega
      head_le := hhs
      ptwc_le := hptwc
      T_lt := by simp only [canon]; omega
      T_ctx := by simp only [canon]; omega
      H_lt := by simp only [canon]; omega
      grid_lt := by simp only [canon]; omega
      shs_eq := rfl
      pots_eq := rfl
      sots_eq := rfl
      ots_eq := rfl
      ohs_ge := le_refl _
      olse_layout := fun ho => by
        simp only [canon] at ho ⊢
        simp only [ho, ite_true]
        exact Or.inl ⟨le_refl _, by simp⟩
      strides_lt := by
        simp only [canon]
        refine ⟨by omega, by omega, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> split <;> omega
      out_i32 := by simp only [canon]; constructor <;> omega
      in_i32 := by simp only [canon]; constructor <;> omega
      lse_i32 := by
        simp only [canon]
        have hpt : (ptwc - 1) * H ≤ 2 ^ 16 * 64 :=
          Nat.mul_le_mul (le_trans (Nat.sub_le _ _) (le_trans hptwc hT)) hH
        have hT1H2 : (T - 1) * H ≤ 2 ^ 16 * 64 := hT1H
        have hHT2 : (H - 1) * T ≤ 64 * 2 ^ 16 := hHT
        refine ⟨fun _ => ⟨?_, ?_⟩, ?_, ?_, ?_, ?_⟩ <;> split <;> omega
      output_len := by
        intro hT0 hH0 hs0
        simp only [canon] at hT0 hH0 hs0 ⊢
        have h1 : (T - 1) * (H * hs) + (H - 1) * hs + hs = T * H * hs := by
          obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by omega⟩
          obtain ⟨H', rfl⟩ : ∃ H', H = H' + 1 := ⟨H - 1, by omega⟩
          simp only [Nat.add_sub_cancel]; ring
        omega
      suffix_output_len := by
        intro hT0 hH0 hs0
        simp only [canon] at hT0 hH0 hs0 ⊢
        have h1 : (T - 1) * (H * (hs + pad)) + (H - 1) * (hs + pad) + hs ≤
            T * H * (hs + pad) := by
          obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by omega⟩
          obtain ⟨H', rfl⟩ : ∃ H', H = H' + 1 := ⟨H - 1, by omega⟩
          simp only [Nat.add_sub_cancel]; nlinarith
        omega
      prefix_output_len := by
        intro hT0 hH0 hs0
        simp only [canon] at hT0 hH0 hs0 ⊢
        have h1 : (ptwc - 1) * (H * (hs + pad)) + (H - 1) * (hs + pad) + hs ≤
            T * H * (hs + pad) := by
          obtain ⟨T', rfl⟩ : ∃ T', ptwc = T' + 1 := ⟨ptwc - 1, by omega⟩
          obtain ⟨H', rfl⟩ : ∃ H', H = H' + 1 := ⟨H - 1, by omega⟩
          simp only [Nat.add_sub_cancel]
          have : (T' + 1) * (H' + 1) * (hs + pad) ≤ T * (H' + 1) * (hs + pad) :=
            Nat.mul_le_mul (Nat.mul_le_mul hptwc (le_refl _)) (le_refl _)
          nlinarith
        omega
      output_lse_len := by
        intro ho hT0 hH0
        simp only [canon] at ho hT0 hH0 ⊢
        simp only [ho, ite_true]
        have : (H - 1) * T + (T - 1) * 1 < H * T := by
          obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by omega⟩
          obtain ⟨H', rfl⟩ : ∃ H', H = H' + 1 := ⟨H - 1, by omega⟩
          simp only [Nat.add_sub_cancel]; nlinarith
        exact this
      suffix_lse_len := by
        intro hT0 hH0
        simp only [canon] at hT0 hH0 ⊢
        obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := ⟨T - 1, by omega⟩
        obtain ⟨H', rfl⟩ : ∃ H', H = H' + 1 := ⟨H - 1, by omega⟩
        cases lseT <;> simp only [Nat.add_sub_cancel] <;> simp <;> nlinarith
      prefix_lse_len := by
        intro hp0 hH0
        simp only [canon] at hp0 hH0 ⊢
        obtain ⟨T', rfl⟩ : ∃ T', ptwc = T' + 1 := ⟨ptwc - 1, by omega⟩
        obtain ⟨H', rfl⟩ : ∃ H', H = H' + 1 := ⟨H - 1, by omega⟩
        have hT' : T' + 1 ≤ T := hptwc
        cases lseT <;> simp only [Nat.add_sub_cancel] <;> simp <;> nlinarith
      output_lse_disj := fun _ =>
        bytes_disj _ _ (Or.inl (by simp only [canon, DType.bytes]; omega))
      output_disj := by
        refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_ <;>
          exact bytes_disj _ _ (Or.inl (by simp only [canon, DType.bytes]; omega))
      output_lse_in_disj := fun _ => by
        refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_ <;>
          exact bytes_disj _ _ (Or.inl (by simp only [canon, DType.bytes]; omega)) }

/-! ## Coverage: the host's test cases (source lines 335–345)

`(num_tokens, num_heads, head_size, prefill_tokens_with_context, head padding, lse transposed)`. -/

/-- `(1, 8, 128, None, 0, False)`. -/
def setup_t1 : Inst := canon 1 8 128 128 1 0 false true
theorem setup_t1_ok : setup_t1.ok := canon_ok 1 8 128 128 1 0 7 false true
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) le_rfl

/-- `(6, 8, 128, None, 0, False)`, the both-empty test. -/
def setup_t6 : Inst := canon 6 8 128 128 6 0 false true
theorem setup_t6_ok : setup_t6.ok := canon_ok 6 8 128 128 6 0 7 false true
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) le_rfl

/-- `(256, 4, 32, None, 0, False)`. -/
def setup_t256 : Inst := canon 256 4 32 32 256 0 false true
theorem setup_t256_ok : setup_t256.ok := canon_ok 256 4 32 32 256 0 5 false true
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) le_rfl

/-- `(613, 16, 48, 128, 0, False)`. -/
def setup_t613 : Inst := canon 613 16 48 64 128 0 false true
theorem setup_t613_ok : setup_t613.ok := canon_ok 613 16 48 64 128 0 6 false true
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(512, 32, 96, 128, 0, True)`. -/
def setup_t512 : Inst := canon 512 32 96 128 128 0 true true
theorem setup_t512_ok : setup_t512.ok := canon_ok 512 32 96 128 128 0 7 true true
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(1024, 48, 128, None, 64, True)`. -/
def setup_t1024 : Inst := canon 1024 48 128 128 1024 64 true true
theorem setup_t1024_ok : setup_t1024.ok := canon_ok 1024 48 128 128 1024 64 7 true true
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) le_rfl

/-- `(1536, 64, 256, 1, 0, False)`. -/
def setup_t1536 : Inst := canon 1536 64 256 256 1 0 false true
theorem setup_t1536_ok : setup_t1536.ok := canon_ok 1536 64 256 256 1 0 8 false true
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(4096, 32, 128, 128, 0, False)`. -/
def setup_t4096 : Inst := canon 4096 32 128 128 128 0 false true
theorem setup_t4096_ok : setup_t4096.ok := canon_ok 4096 32 128 128 128 0 7 false true
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(300, 8, 64, 0, 8, True)`: every token copied. -/
def setup_t300 : Inst := canon 300 8 64 64 0 8 true true
theorem setup_t300_ok : setup_t300.ok := canon_ok 300 8 64 64 0 8 6 true true
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The code base's largest test shape (`tests/kernels/attention/test_merge_attn_states.py`):
4096 tokens, 64 heads, head size 256, `prefill_tokens_with_context = 128`. Every other
combination of its sizes is `canon_ok` with smaller arguments. -/
def setup_largest : Inst := canon 4096 64 256 256 128 0 false true
theorem setup_largest_ok : setup_largest.ok := canon_ok 4096 64 256 256 128 0 8 false true
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-! ## Coverage: without `output_lse`

The backends' calls that pass no `output_lse` (`return_lse=False`, `OUTPUT_LSE = False`): the
host's test runs each of its cases this way too (source lines 387–389). -/

/-- `(613, 16, 48, 128, 0, False)` without `output_lse`. -/
def setup_nolse_t613 : Inst := canon 613 16 48 64 128 0 false false
theorem setup_nolse_t613_ok : setup_nolse_t613.ok := canon_ok 613 16 48 64 128 0 6 false false
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `(1024, 48, 128, None, 64, True)` without `output_lse`. -/
def setup_nolse_t1024 : Inst := canon 1024 48 128 128 1024 64 true false
theorem setup_nolse_t1024_ok : setup_nolse_t1024.ok := canon_ok 1024 48 128 128 1024 64 7 true false
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) le_rfl

/-- `(300, 8, 64, 0, 8, True)` without `output_lse`: every token copied. -/
def setup_nolse_t300 : Inst := canon 300 8 64 64 0 8 true false
theorem setup_nolse_t300_ok : setup_nolse_t300.ok := canon_ok 300 8 64 64 0 8 6 true false
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- The code base's largest test shape without `output_lse`. -/
def setup_nolse_largest : Inst := canon 4096 64 256 256 128 0 false false
theorem setup_nolse_largest_ok : setup_nolse_largest.ok :=
  canon_ok 4096 64 256 256 128 0 8 false false
  (by norm_num) (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-! ## Satisfiable -/

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem testMem_read_f32 {a : Int} (ha : Mem.Aligned .f32 a) :
    testMem.read .f32 a = some (FloatV.ofPat 0 : F32) := by
  have hb : testMem.readBytes a 4 = some [0, 0, 0, 0] := by simp [Mem.readBytes, testMem]
  unfold Mem.read
  rw [show DType.f32.bytes = 4 from rfl]
  simp only [ha, ite_true, hb]
  decide

theorem testMem_read_bf16 {a : Int} (ha : Mem.Aligned .bf16 a) :
    testMem.read .bf16 a = some (FloatV.ofPat 0 : BF16) := by
  have hb : testMem.readBytes a 2 = some [0, 0] := by simp [Mem.readBytes, testMem]
  unfold Mem.read
  rw [show DType.bf16.bytes = 2 from rfl]
  simp only [ha, ite_true, hb]
  decide

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨setup_t1, testMem, setup_t1_ok, ?_⟩
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid testMem := fun b h1 h2 h3 =>
    ⟨h1, h2, h3, fun _ _ => by simp [testMem]⟩
  refine ⟨hv _ (by simp [setup_t1, canon]) (by simp [setup_t1, canon, DType.bytes])
      (by simp [setup_t1, canon, DType.bytes]),
    fun _ => hv _ (by simp [setup_t1, canon]) (by simp [setup_t1, canon, DType.bytes])
      (by simp [setup_t1, canon, DType.bytes]),
    hv _ (by simp [setup_t1, canon]) (by simp [setup_t1, canon, DType.bytes])
      (by simp [setup_t1, canon, DType.bytes]),
    hv _ (by simp [setup_t1, canon]) (by simp [setup_t1, canon, DType.bytes])
      (by simp [setup_t1, canon, DType.bytes]),
    hv _ (by simp [setup_t1, canon]) (by simp [setup_t1, canon, DType.bytes])
      (by simp [setup_t1, canon, DType.bytes]),
    hv _ (by simp [setup_t1, canon]) (by simp [setup_t1, canon, DType.bytes])
      (by simp [setup_t1, canon, DType.bytes]), ?_⟩
  intro t h ht hh
  have ht0 : t = 0 := by simp [setup_t1, canon] at ht; omega
  have hh8 : h < 8 := by simpa [setup_t1, canon] using hh
  subst ht0
  refine ⟨FloatV.ofPat 0, FloatV.ofPat 0, ?_, ?_, by decide, by decide, ?_⟩
  · refine testMem_read_f32 ⟨?_, ?_, ?_⟩ <;>
      simp only [Buf.ptr, setup_t1, canon, plIdx, DType.bytes, Bool.false_eq_true, ite_false] <;> omega
  · refine testMem_read_f32 ⟨?_, ?_, ?_⟩ <;>
      simp only [Buf.ptr, setup_t1, canon, slIdx, DType.bytes, Bool.false_eq_true, ite_false] <;> omega
  · intro _ j hj
    have hj' : j < 128 := by simpa [setup_t1, canon] using hj
    refine ⟨FloatV.ofPat 0, FloatV.ofPat 0, ?_, ?_, by decide, by decide⟩
    · refine testMem_read_bf16 ⟨?_, ?_, ?_⟩ <;>
        simp only [Buf.ptr, setup_t1, canon, poIdx, DType.bytes] <;> omega
    · refine testMem_read_bf16 ⟨?_, ?_, ?_⟩ <;>
        simp only [Buf.ptr, setup_t1, canon, soIdx, DType.bytes] <;> omega

end Claims.MergeAttnStates
