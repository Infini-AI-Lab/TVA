import TVA

/-!
# `moe_router_bf16`: MoE router with soft cap and top-2, SGLang

The kernel is `fused_moe_router_cudacore_kernel` of `source/moe_router.py` (lines 51–152; it reads
the module-level names `triton` and `tl`, lines 47–48). The host `fused_moe_router_cudacore`
(lines 155–195) launches it on the grid `(bs,)`, one token per program, with
`BLOCK_SIZE = next_power_of_2(hidden_dim)`. The covered launches are the ones `SETUP.md` asks for:
Grok's router, `num_experts = 8`, `topk = 2`, `moe_softcapping = 30.0`, no correction bias,
`moe_renormalize = False`, a `bfloat16` `x` and a `float32` gate, `hidden_dim` up to `8192`.

What is claimed: every covered launch ends normally, and for every token there are reals `C e`, one
per expert, each within `capEps = 2^-18·Σ_h |x·w| + 2^-14` of the exact soft-capped logit
`30·tanh(μ_e / 30)` (`μ_e = Σ_h x[t, h]·w[e, h]`), such that `topk_ids` holds the first index of
the largest `C` and the first index of the largest `C` among the other experts, and `topk_weights`
holds finite values within `2^-14` (relative) plus `2^-130` of the softmax of `C` at those two
experts; every other byte is unchanged.

The conditions: `hidden_dim` a positive multiple of `128`, at most `BLOCK_SIZE ≤ 2^13` (a power of
two); `1 ≤ bs ≤ 2^22` and the `int32` products and offsets do not wrap; the buffers hold their
tensors and the outputs share no byte with the inputs or with each other; every input is finite,
and for every token and expert `Σ_h |x·w| ≤ 2^120` and the logit plus its fine tolerance `capTol`
is at most `1320` (the soft cap's `exp` does not overflow).

The tolerances rest on the device facts for `tl.exp` and `/` on `float32` and on the depth of a
`tl.sum`'s tree (`Device`); README, Tolerances derives them.
-/

namespace Claims.MoeRouter

open TVA

/-! ## The model -/

/-- The `float32` constant Triton builds for the Python number whose `float64` pattern is `p`
(`FloatV.litF32`, `TVA.Triton.Promote`). -/
def c32 (p : Nat) : F32 := FloatV.litF32 (FloatV.ofPat p)

/-- `fused_moe_router_cudacore_kernel` (source lines 51–152) on the covered compile-time values:
`num_experts = 8`, `topk = 2`, `moe_softcapping = 30.0`, `is_correction_bias = False`,
`moe_renormalize = False` (the model writes them as constants and leaves out the branches
other values would take). `hidden_dim` and `BLOCK_SIZE` are `tl.constexpr` (Lean
parameters). `input_ptr : *bf16`, `moe_router_weight_ptr : *fp32`, `topk_weights_ptr : *fp32`,
`topk_ids_ptr : *i32`; `correction_bias_ptr` is the host's `None`, which no visited line reads, so
the model takes no parameter for it. -/
def moe_router_kernel (input_ptr : Ptr .bf16) (moe_router_weight_ptr topk_weights_ptr : Ptr .f32)
    (topk_ids_ptr : Ptr .i32) (hidden_dim BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 66
    let pid := tl.program_id ctx 0
    -- line 68
    let offsets := tl.arange 0 BLOCK_SIZE
    -- line 69: the constexpr `hidden_dim` is an `int32` constant; signed comparison
    let mask := offsets.ltS (IntV.ofNat hidden_dim : I32)
    -- line 72: `tl.arange(0, num_experts)[:, None]`, broadcast along the row
    let expert_offsets := (tl.arange 0 8).col BLOCK_SIZE
    -- line 73: `mask[None, :]`, broadcast along the column
    let router_mask := mask.row 8
    -- lines 74–78: `other=0.0` is the `float32` `+0`
    let w_router ← tl.load
      (moe_router_weight_ptr + expert_offsets * (IntV.ofNat hidden_dim : I32) + offsets.row 8)
      (some router_mask) (some (bc (c32 0)))
    -- line 80: `other=0.0`: the `float32` `+0` converted to `bfloat16`, `+0`
    let x ← tl.load (input_ptr + pid * (IntV.ofNat hidden_dim : I32) + offsets) (some mask)
      (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
    -- line 83: `w_router.to(tl.float32)` is the identity; `x[None, :].to(tl.float32)` the exact
    -- widening (written before the broadcast); the IEEE product; the row sum of a product, whose
    -- additions may be fused (`tl.sum1c`)
    let x32 : Tensor [BLOCK_SIZE] F32 ← tl.cvt x
    let prod ← tl.mul w_router (x32.row 8)
    let logits ← tl.sum1c prod
    -- line 86: `moe_softcapping == 0` is false (`30.0`): the `else` branch
    -- line 89: `/` by the constexpr `30.0` (the `float32` `30`): `tl.fdiv`, the platform's `/`
    let logits_scaled ← tl.div logits (bc (c32 0x403E000000000000))
    -- line 90: `2 *` builds the `float32` `2.0`; the IEEE product, then `tl.exp`
    let doubled ← tl.mul (bc (c32 0x4000000000000000)) logits_scaled
    let exped ← tl.exp doubled
    -- lines 91–92: the `float32` `1.0`; no product feeds either: IEEE
    let top ← tl.sub exped (bc (c32 0x3FF0000000000000))
    let bottom ← tl.add exped (bc (c32 0x3FF0000000000000))
    -- line 93: `top / bottom * moe_softcapping`
    let logits_softcapped ← tl.mul (← tl.div top bottom) (bc (c32 0x403E000000000000))
    -- lines 96–98: `is_correction_bias` is false: not visited
    -- line 105: `tl.argmax(…, axis=0)`, `tie_break_left=True` (the default)
    let top1 := (← tl.argmaxAll true true logits_softcapped).2
    -- line 106: `pid * topk + 0`, `int32`
    tl.store1 (topk_ids_ptr + (pid * (IntV.ofNat 2 : I32) + (IntV.ofInt 0 : I32))) top1
    -- line 108: `tl.max(…, axis=0)`
    let top1_v ← tl.maxAll true logits_softcapped
    -- line 109: `logits_softcapped` is a product (line 93), so the subtraction may be fused
    -- (`tl.subc`); `tl.exp`; the IEEE sum of the exponentials; `1.0 /` the `float32` `1.0`
    let diffs ← tl.subc logits_softcapped (bc top1_v)
    let exps ← tl.exp diffs
    let den ← tl.sum exps
    let invsumexp : F32 := (← tl.div (Tensor.scalar (c32 0x3FF0000000000000)) (Tensor.scalar den)).item
    -- lines 111–114
    tl.store1 (topk_weights_ptr + (pid * (IntV.ofNat 2 : I32) + (IntV.ofInt 0 : I32))) invsumexp
    -- line 116: `topk >= 2` is true: the branch is visited
    -- lines 117–122: `tl.where(arange != top1, logits_softcapped, float("-inf"))`, the `-inf`
    -- built in `float32`; then `tl.argmax`
    let masked := tl.where ((tl.arange 0 8).map (fun j => !(IntV.eq j top1))) logits_softcapped
      (bc (c32 0xFFF0000000000000))
    let top2 := (← tl.argmaxAll true true masked).2
    -- line 123
    tl.store1 (topk_ids_ptr + (pid * (IntV.ofNat 2 : I32) + (IntV.ofInt 1 : I32))) top2
    -- line 124: the `int1` mask times a `float32`: converted to `float32`; the IEEE product; the
    -- sum of a product (`tl.sumc`)
    let onehot32 : Tensor [8] F32 ← tl.convert .i1 .f32 ((tl.arange 0 8).map (fun j => IntV.eq j top2))
    let prod2 ← tl.mul logits_softcapped onehot32
    let top2_v ← tl.sumc prod2
    -- lines 125–128: `top2_v - top1_v` (a reduction's and a maximum's results: no product feeds
    -- it, IEEE); `tl.exp`; the IEEE product
    let wdiff ← tl.sub (Tensor.scalar top2_v) (Tensor.scalar top1_v)
    let w2e ← tl.exp wdiff
    let w1 : F32 := (← tl.mul w2e (Tensor.scalar invsumexp)).item
    tl.store1 (topk_weights_ptr + (pid * (IntV.ofNat 2 : I32) + (IntV.ofInt 1 : I32))) w1
    -- lines 131–151: `topk > 2` is false: not visited

/-! ## The conditions -/

/-- What the host chooses: one launch of `fused_moe_router_cudacore` on Grok's router. -/
structure Inst where
  /-- `bs`, the grid `(bs,)`. -/
  bs : Nat
  /-- `hidden_dim`. -/
  hidden : Nat
  /-- `BLOCK_SIZE` (the host passes `triton.next_power_of_2(hidden_dim)`). -/
  BLOCK : Nat
  /-- `x`: `bfloat16`, `[bs, hidden_dim]`, contiguous. -/
  x : Buf .bf16
  /-- `router_weight`: `float32`, `[8, hidden_dim]`, contiguous. -/
  w : Buf .f32
  /-- `topk_weights`: `float32`, `[bs, 2]`, allocated by the host. -/
  outw : Buf .f32
  /-- `topk_ids`: `int32`, `[bs, 2]`, allocated by the host. -/
  outi : Buf .i32

/-- The instances covered (README, Conditions). -/
structure Inst.ok (d : Inst) : Prop where
  /-- A token has a hidden state; the block covers it; `BLOCK_SIZE` a power of two (`tl.arange`)
  of at most `8192` (the host's `next_power_of_2(hidden_dim)` for Grok's 6144 and 8192). -/
  hidden_pos : 0 < d.hidden
  hidden_le : d.hidden ≤ d.BLOCK
  block_pow2 : ∃ k, d.BLOCK = 2 ^ k
  block_le : d.BLOCK ≤ 2 ^ 13
  /-- `hidden_dim` is a model dimension: a multiple of 128, at most `2^17`. -/
  hidden_mul : d.hidden % 128 = 0
  hidden_le17 : d.hidden ≤ 2 ^ 17
  /-- At least one token; the `int32` products `pid·hidden_dim` (line 80 adds it to the pointer
  before `offsets`, so only the product is `int32`) and the offset `pid·2 + 1` do not wrap (a limit
  of the kernel, README, Exclusions), and fewer than `2^31` programs. -/
  bs_pos : 0 < d.bs
  x_off : (d.bs - 1) * d.hidden < 2 ^ 31
  out_off : 2 * d.bs ≤ 2 ^ 31
  /-- `bs` is a number of tokens, a context length: at most `2^22`. -/
  bs_le : d.bs ≤ 2 ^ 22
  /-- The buffers hold their tensors. -/
  x_len : d.bs * d.hidden ≤ d.x.len
  w_len : 8 * d.hidden ≤ d.w.len
  outw_len : 2 * d.bs ≤ d.outw.len
  outi_len : 2 * d.bs ≤ d.outi.len
  /-- The outputs (allocated by the host) share no byte with the inputs or with each other. -/
  outw_x : Disjoint d.outw.bytes d.x.bytes
  outw_w : Disjoint d.outw.bytes d.w.bytes
  outw_outi : Disjoint d.outw.bytes d.outi.bytes
  outi_x : Disjoint d.outi.bytes d.x.bytes
  outi_w : Disjoint d.outi.bytes d.w.bytes

/-- The model on an instance. -/
def model (d : Inst) : Kernel :=
  moe_router_kernel (d.x.ptr 0) (d.w.ptr 0) (d.outw.ptr 0) (d.outi.ptr 0) d.hidden d.BLOCK

/-- The host's launch (lines 180–193): the grid `(bs,)`; the source's block shapes `[8, BLOCK_SIZE]`
(`w_router`, the product), `[BLOCK_SIZE]` (`offsets`, `mask`, `x`) and `[8]` (the logits, the
masks of lines 119 and 124); `enable_fp_fusion` at Triton's default; `num_warps` (the host's
formula) is performance-only; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 d.bs [[8, d.BLOCK], [d.BLOCK], [8]] (fusion := true)

/-! ### The host's inputs -/

/-- `x[t, h]`, as a real (`0` when absent). -/
noncomputable def xVal (m : Mem) (d : Inst) (t h : Nat) : ℝ :=
  match m.read .bf16 (d.x.ptr ((t * d.hidden + h : Nat) : Int)).addr with
  | some v => v.toReal
  | none => 0

/-- `router_weight[e, h]`, as a real (`0` when absent). -/
noncomputable def wVal (m : Mem) (d : Inst) (e h : Nat) : ℝ :=
  match m.read .f32 (d.w.ptr ((e * d.hidden + h : Nat) : Int)).addr with
  | some v => v.toReal
  | none => 0

/-- The exact logit of token `t` and expert `e`: `Σ_h x[t, h] · router_weight[e, h]` (the
`float64` reference's `x @ router_weight.T`). -/
noncomputable def mu (m : Mem) (d : Inst) (t e : Nat) : ℝ :=
  ∑ h ∈ Finset.range d.hidden, xVal m d t h * wVal m d e h

/-- `Σ_h |x[t, h] · router_weight[e, h]|`, the magnitude the reduction's error scales with. -/
noncomputable def absS (m : Mem) (d : Inst) (t e : Nat) : ℝ :=
  ∑ h ∈ Finset.range d.hidden, |xVal m d t h * wVal m d e h|

/-- The exact soft cap of a logit: `30 · tanh(l / 30)` (the reference's
`tanh(logits / c) * c`). -/
noncomputable def capRef (l : ℝ) : ℝ := 30 * Real.tanh (l / 30)

/-! ### The tolerances (README, Tolerances) -/

/-- One rounding of `float32`: `u = 2^-24`, `η = 2^-150`. -/
noncomputable def u32 : ℝ := 2 ^ (-24 : ℤ)
noncomputable def eta32 : ℝ := 2 ^ (-150 : ℤ)

/-- The computed logit's error: the products' roundings (`u·S + h·η`), and the fused reduction
over the `BLOCK` lanes along a tree of depth at most `⌈log₂ BLOCK⌉` (`Device.SumDepth`;
the fused tree sum's error bound `errBound (3u) (4η) ⌈log₂ B⌉ (B−1)` of the leaves' magnitudes,
at most `(1+u)·S + h·η`). -/
noncomputable def tolL (d : Inst) (S : ℝ) : ℝ :=
  ((1 + 3 * u32) ^ Nat.clog 2 d.BLOCK - 1) * ((1 + u32) * S + d.hidden * eta32) +
    4 * eta32 * (d.BLOCK - 1 : ℕ) * (1 + 3 * u32) ^ Nat.clog 2 d.BLOCK +
    u32 * S + d.hidden * eta32

/-- The fine bound of a computed capped logit's error against `capRef` of the exact logit (used
by `Pre`'s range condition; `Post` states `capEps`, which is at least it): the logit's error and
the scaling's (`/ 30`, `2 ·`) carried through the soft cap, which is `1`-Lipschitz in the logit
(`2^-21·(|μ| + tolL)` from the division and the doubling), plus the cap formula's own error
(`tl.exp`, `− 1`, `+ 1`, `/`, `· 30`), within `2^-14`. -/
noncomputable def capTol (d : Inst) (μ S : ℝ) : ℝ :=
  (1 + 2 ^ (-21 : ℤ)) * tolL d S + 2 ^ (-21 : ℤ) * |μ| + 2 ^ (-14 : ℤ)

/-- **The tolerance of a capped logit** in `Post`: `2^-18` relative to `absS` (the logit's
reduction, along a tree of depth at most `⌈log₂ BLOCK⌉ ≤ 13`, `Device.SumDepth`, and the scaling),
plus `2^-14` (the cap formula's own error). README, Tolerances, derives it; it is at least the fine
bound `capTol` up to `capTol`'s absolute terms below `2^-130`, which the formula's margin covers. -/
noncomputable def capEps (S : ℝ) : ℝ := 2 ^ (-18 : ℤ) * S + 2 ^ (-14 : ℤ)

/-- The relative tolerance of the first weight (the inverse sum of the exponentials). -/
noncomputable def wRel0 : ℝ := 2 ^ (-14 : ℤ)

/-- The relative tolerance of the second weight (also the one-hot reduction `top2_v`, along a
tree of depth `3` over the `8` leaves, and `exp(top2_v − top1_v)`). -/
noncomputable def wRel1 : ℝ := 2 ^ (-14 : ℤ)

/-- The weights' absolute floor (the `tl.exp`s' absolute errors). -/
noncomputable def wFloor : ℝ := 2 ^ (-130 : ℤ)

/-- The exact softmax over `C`, at expert `j`, normalised at `i0`:
`exp(C j − C i0) / Σ_k exp(C k − C i0)`. -/
noncomputable def softRef (C : ℕ → ℝ) (i0 j : ℕ) : ℝ :=
  Real.exp (C j - C i0) / ∑ k ∈ Finset.range 8, Real.exp (C k - C i0)

/-- The inputs covered: the four buffers are valid (aligned, present), and for every token the
inputs are finite and, limits of the kernel's `float32` arithmetic (README, Exclusions), every
expert's `Σ_h |x·w| ≤ 2^120` (the sum stays finite) and its logit, with its tolerance, at most
`1320 = 44·30` (above about `1331` the soft cap's `exp(2·l/30)` overflows and the kernel's
`inf/inf` is a NaN). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.x.Valid init ∧ d.w.Valid init ∧ d.outw.Valid init ∧ d.outi.Valid init ∧
  (∀ t h, t < d.bs → h < d.hidden →
    ∃ v, init.read .bf16 (d.x.ptr ((t * d.hidden + h : Nat) : Int)).addr = some v ∧ v.isFinite = true) ∧
  (∀ e h, e < 8 → h < d.hidden →
    ∃ v, init.read .f32 (d.w.ptr ((e * d.hidden + h : Nat) : Int)).addr = some v ∧ v.isFinite = true) ∧
  ∀ t e, t < d.bs → e < 8 →
    absS init d t e ≤ 2 ^ 120 ∧ mu init d t e + capTol d (mu init d t e) (absS init d t e) ≤ 1320

/-! ## The specification -/

/-- What the launch leaves for token `t`: there are reals `C e`, one per expert (the capped logits
the run computes), each within `capEps` of the exact capped logit `capRef (mu t e)`, such that
* `topk_ids[t, 0] = i0`, the first index of the largest `C` (`tie_break_left`), and
  `topk_ids[t, 1] = i1 ≠ i0`, the first index of the largest `C` among the other experts;
* `topk_weights[t, 0]` and `topk_weights[t, 1]` are finite and within `wRel0` and `wRel1`
  (relative) plus `wFloor` of the softmax of `C` at `i0` and `i1`. -/
def TokenPost (d : Inst) (init fin : Mem) (t : Nat) : Prop :=
  ∃ (C : ℕ → ℝ) (i0 i1 : ℕ),
    (∀ e, e < 8 → |C e - capRef (mu init d t e)| ≤ capEps (absS init d t e)) ∧
    i0 < 8 ∧ (∀ e, e < 8 → C e ≤ C i0) ∧ (∀ e, e < i0 → C e < C i0) ∧
    i1 < 8 ∧ i1 ≠ i0 ∧ (∀ e, e < 8 → e ≠ i0 → C e ≤ C i1) ∧
    (∀ e, e < i1 → e ≠ i0 → C e < C i1) ∧
    fin.read .i32 (d.outi.ptr ((2 * t : Nat) : Int)).addr = some (IntV.ofNat i0) ∧
    fin.read .i32 (d.outi.ptr ((2 * t + 1 : Nat) : Int)).addr = some (IntV.ofNat i1) ∧
    ∃ a0 a1 : F32,
      fin.read .f32 (d.outw.ptr ((2 * t : Nat) : Int)).addr = some a0 ∧
      fin.read .f32 (d.outw.ptr ((2 * t + 1 : Nat) : Int)).addr = some a1 ∧
      a0.isFinite = true ∧ a1.isFinite = true ∧
      |a0.toReal - softRef C i0 i0| ≤ wRel0 * softRef C i0 i0 + wFloor ∧
      |a1.toReal - softRef C i0 i1| ≤ wRel1 * softRef C i0 i1 + wFloor

/-- What the launch leaves: every token's outputs as `TokenPost` says, and every byte outside the
`2·bs` elements of `topk_weights` and of `topk_ids` unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ t, t < d.bs → TokenPost d init fin t) ∧
  ∀ z, z ∉ d.outw.elems {j | j < 2 * d.bs} ∪ d.outi.elems {j | j < 2 * d.bs} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `fused_moe_router_cudacore_kernel` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- An instance of the host's launch: `bs` tokens, `hidden_dim`, `BLOCK_SIZE`, and the four tensors
at the 16-byte-aligned addresses `k·2^34`, each exactly as long as its tensor. -/
def mk (bs hidden B : Nat) : Inst :=
  { bs := bs, hidden := hidden, BLOCK := B,
    x := ⟨1 * 17179869184, bs * hidden⟩, w := ⟨2 * 17179869184, 8 * hidden⟩,
    outw := ⟨3 * 17179869184, 2 * bs⟩, outi := ⟨4 * 17179869184, 2 * bs⟩ }

/-- Two buffers at the addresses `i·2^34` and `j·2^34`, `i ≠ j`, each at most `2^34` bytes long,
share no byte. -/
theorem disj_far {d1 d2 : DType} (b : Buf d1) (c : Buf d2) (i j : Nat)
    (hb : b.base = i * 17179869184) (hc : c.base = j * 17179869184)
    (hbl : b.len * d1.bytes ≤ 17179869184) (hcl : c.len * d2.bytes ≤ 17179869184) (hij : i ≠ j) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  rcases Nat.lt_or_gt_of_ne hij with h | h
  · have : i * 17179869184 + 17179869184 ≤ j * 17179869184 := by
      have := Nat.mul_le_mul_right 17179869184 (Nat.succ_le_of_lt h); simpa [Nat.succ_mul] using this
    rw [hb] at hx; rw [hc] at hy; omega
  · have : j * 17179869184 + 17179869184 ≤ i * 17179869184 := by
      have := Nat.mul_le_mul_right 17179869184 (Nat.succ_le_of_lt h); simpa [Nat.succ_mul] using this
    rw [hb] at hx; rw [hc] at hy; omega

theorem mk_ok (bs hidden B k : Nat) (hB : B = 2 ^ k)
    (h : 0 < hidden ∧ hidden ≤ B ∧ B ≤ 2 ^ 13 ∧ 0 < bs ∧ (bs - 1) * hidden < 2 ^ 31 ∧
      2 * bs ≤ 2 ^ 31 ∧ bs * hidden * 2 ≤ 17179869184 ∧ 2 * bs * 4 ≤ 17179869184)
    (h' : hidden % 128 = 0 ∧ hidden ≤ 2 ^ 17 ∧ bs ≤ 2 ^ 22) :
    (mk bs hidden B).ok := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  have hw : 8 * hidden * 4 ≤ 17179869184 := by omega
  exact
    { hidden_pos := h1, hidden_le := h2, block_pow2 := ⟨k, hB⟩, block_le := h3, bs_pos := h4,
      hidden_mul := h'.1, hidden_le17 := h'.2.1, bs_le := h'.2.2,
      x_off := h5, out_off := h6, x_len := le_refl _, w_len := le_refl _, outw_len := le_refl _,
      outi_len := le_refl _,
      outw_x := disj_far _ _ 3 1 rfl rfl h8 h7 (by decide),
      outw_w := disj_far _ _ 3 2 rfl rfl h8 hw (by decide),
      outw_outi := disj_far _ _ 3 4 rfl rfl h8 h8 (by decide),
      outi_x := disj_far _ _ 4 1 rfl rfl h8 h7 (by decide),
      outi_w := disj_far _ _ 4 2 rfl rfl h8 hw (by decide) }

/-- The source's Grok test cases (line 215): `bs` in `1, 7, 128, 511`, `hidden_dim` `6144` (Grok-1)
and `8192` (Grok-2), `BLOCK_SIZE = next_power_of_2(hidden_dim) = 8192`. -/
def setup_1_6144 : Inst := mk 1 6144 8192
def setup_1_8192 : Inst := mk 1 8192 8192
def setup_7_6144 : Inst := mk 7 6144 8192
def setup_7_8192 : Inst := mk 7 8192 8192
def setup_128_6144 : Inst := mk 128 6144 8192
def setup_128_8192 : Inst := mk 128 8192 8192
def setup_511_6144 : Inst := mk 511 6144 8192
def setup_511_8192 : Inst := mk 511 8192 8192

theorem setup_1_6144_ok : setup_1_6144.ok := mk_ok _ _ _ 13 rfl (by decide) (by decide)
theorem setup_1_8192_ok : setup_1_8192.ok := mk_ok _ _ _ 13 rfl (by decide) (by decide)
theorem setup_7_6144_ok : setup_7_6144.ok := mk_ok _ _ _ 13 rfl (by decide) (by decide)
theorem setup_7_8192_ok : setup_7_8192.ok := mk_ok _ _ _ 13 rfl (by decide) (by decide)
theorem setup_128_6144_ok : setup_128_6144.ok := mk_ok _ _ _ 13 rfl (by decide) (by decide)
theorem setup_128_8192_ok : setup_128_8192.ok := mk_ok _ _ _ 13 rfl (by decide) (by decide)
theorem setup_511_6144_ok : setup_511_6144.ok := mk_ok _ _ _ 13 rfl (by decide) (by decide)
theorem setup_511_8192_ok : setup_511_8192.ok := mk_ok _ _ _ 13 rfl (by decide) (by decide)

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem valid_zero {dt : DType} (b : Buf dt) (hb : 0 ≤ b.base)
    (h2 : b.base + b.len * dt.bytes ≤ 2 ^ 64) (ha : b.base % dt.bytes = 0) : b.Valid zeroMem :=
  ⟨hb, h2, ha, fun _ _ => by simp [zeroMem]⟩

/-- The reduction's growth factor is at most `3` on the covered blocks. -/
theorem growth_le {n : Nat} (hn : n ≤ 8192) : (1 + 3 * u32) ^ n ≤ 3 := by
  have hu : u32 = 2 ^ (-24 : ℤ) := rfl
  have h1 : 1 + 3 * u32 ≤ Real.exp (3 * u32) := by linarith [Real.add_one_le_exp (3 * u32)]
  have h0 : (0 : ℝ) ≤ 1 + 3 * u32 := by rw [hu]; positivity
  have hx0 : (0 : ℝ) ≤ n * (3 * u32) := by rw [hu]; positivity
  have hx1 : (n : ℝ) * (3 * u32) ≤ 1 / 2 := by
    have : (n : ℝ) ≤ 8192 := by exact_mod_cast hn
    rw [hu]; norm_num; nlinarith
  calc (1 + 3 * u32) ^ n ≤ Real.exp (3 * u32) ^ n := pow_le_pow_left₀ h0 h1 n
    _ = Real.exp (n * (3 * u32)) := (Real.exp_nat_mul _ n).symm
    _ ≤ 1 / (1 - n * (3 * u32)) := Real.exp_bound_div_one_sub_of_interval hx0 (by linarith)
    _ ≤ 3 := by rw [div_le_iff₀ (by linarith)]; linarith

/-- Every aligned `bfloat16` read of `zeroMem` gives `+0`. -/
theorem zero_bf16 {a : Int} (ha : Mem.Aligned .bf16 a) :
    zeroMem.read .bf16 a = some (FloatV.ofPat 0) := by
  have hb : zeroMem.readBytes a 2 = some [0, 0] := by simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  rw [show DType.bf16.bytes = 2 from rfl]
  simp only [ha, ↓reduceIte, hb, Option.map_some]
  rfl

/-- Every aligned `float32` read of `zeroMem` gives `+0`. -/
theorem zero_f32 {a : Int} (ha : Mem.Aligned .f32 a) :
    zeroMem.read .f32 a = some (FloatV.ofPat 0) := by
  have hb : zeroMem.readBytes a 4 = some [0, 0, 0, 0] := by simp [Mem.readBytes, zeroMem]
  unfold Mem.read
  rw [show DType.f32.bytes = 4 from rfl]
  simp only [ha, ↓reduceIte, hb, Option.map_some]
  rfl

/-- `tolL` is nonnegative. -/
theorem tolL_nonneg (d : Inst) (S : ℝ) (hS : 0 ≤ S) : 0 ≤ tolL d S := by
  have hu : 0 ≤ u32 := by unfold u32; positivity
  have he : 0 ≤ eta32 := by unfold eta32; positivity
  have hG1 : 1 ≤ (1 + 3 * u32) ^ Nat.clog 2 d.BLOCK := one_le_pow₀ (by linarith)
  unfold tolL
  generalize (1 + 3 * u32) ^ Nat.clog 2 d.BLOCK = G at hG1 ⊢
  have : 0 ≤ (1 + u32) * S + d.hidden * eta32 := by positivity
  have : 0 ≤ (G - 1) * ((1 + u32) * S + d.hidden * eta32) := mul_nonneg (by linarith) this
  have : 0 ≤ 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G := by positivity
  have : 0 ≤ u32 * S := mul_nonneg hu hS
  have : 0 ≤ (d.hidden : ℝ) * eta32 := by positivity
  linarith

/-- With `S = 0` the logit's tolerance is tiny. -/
theorem tolL_zero_le (d : Inst) (hB : d.BLOCK ≤ 2 ^ 13) (hh : d.hidden ≤ 2 ^ 13) :
    tolL d 0 ≤ 2 ^ (-100 : ℤ) := by
  have hu : 0 ≤ u32 := by unfold u32; positivity
  have he : 0 ≤ eta32 := by unfold eta32; positivity
  have hD : Nat.clog 2 d.BLOCK ≤ 13 := by
    have := Nat.clog_mono_right 2 hB
    rwa [Nat.clog_pow 2 13 (by norm_num)] at this
  have hG := growth_le (n := Nat.clog 2 d.BLOCK) (by omega)
  have hG1 : 1 ≤ (1 + 3 * u32) ^ Nat.clog 2 d.BLOCK := one_le_pow₀ (by linarith)
  have hB' : ((d.BLOCK - 1 : ℕ) : ℝ) ≤ 8192 := by exact_mod_cast (by omega : d.BLOCK - 1 ≤ 8192)
  have hh' : (d.hidden : ℝ) ≤ 8192 := by exact_mod_cast (by omega : d.hidden ≤ 8192)
  unfold tolL
  generalize (1 + 3 * u32) ^ Nat.clog 2 d.BLOCK = G at hG hG1 ⊢
  simp only [mul_zero, zero_add, add_zero]
  have h1 : (G - 1) * (d.hidden * eta32) ≤ 2 * (8192 * eta32) :=
    mul_le_mul (by linarith) (mul_le_mul_of_nonneg_right hh' he) (by positivity) (by norm_num)
  have h2 : 4 * eta32 * ((d.BLOCK - 1 : ℕ) : ℝ) * G ≤ 4 * eta32 * 8192 * 3 :=
    mul_le_mul (mul_le_mul_of_nonneg_left hB' (by positivity)) hG (by linarith) (by positivity)
  have h3 : (d.hidden : ℝ) * eta32 ≤ 8192 * eta32 := mul_le_mul_of_nonneg_right hh' he
  have h4 : eta32 = 2 ^ (-150 : ℤ) := rfl
  have h5 : (2 * (8192 * eta32) + 4 * eta32 * 8192 * 3 + 8192 * eta32 : ℝ) ≤ 2 ^ (-100 : ℤ) := by
    rw [h4]; norm_num
  linarith

theorem setup_x_aligned (t h : Nat) (ht : t < 1) (hh : h < 6144) :
    Mem.Aligned .bf16 (setup_1_6144.x.ptr ((t * setup_1_6144.hidden + h : Nat) : Int)).addr := by
  have : (setup_1_6144.x.ptr ((t * setup_1_6144.hidden + h : Nat) : Int)).addr =
      17179869184 + ((t * 6144 + h : Nat) : Int) * 2 := rfl
  rw [this]
  refine ⟨by omega, ?_, ?_⟩
  · show 17179869184 + ((t * 6144 + h : Nat) : Int) * 2 + 2 ≤ 2 ^ 64
    omega
  · show (17179869184 + ((t * 6144 + h : Nat) : Int) * 2) % 2 = 0
    omega

theorem setup_w_aligned (e h : Nat) (he : e < 8) (hh : h < 6144) :
    Mem.Aligned .f32 (setup_1_6144.w.ptr ((e * setup_1_6144.hidden + h : Nat) : Int)).addr := by
  have : (setup_1_6144.w.ptr ((e * setup_1_6144.hidden + h : Nat) : Int)).addr =
      34359738368 + ((e * 6144 + h : Nat) : Int) * 4 := rfl
  rw [this]
  refine ⟨by omega, ?_, ?_⟩
  · show 34359738368 + ((e * 6144 + h : Nat) : Int) * 4 + 4 ≤ 2 ^ 64
    omega
  · show (34359738368 + ((e * 6144 + h : Nat) : Int) * 4) % 4 = 0
    omega

theorem zero_bf16_rat : (FloatV.ofPat 0 : BF16).toRat = 0 := by decide +kernel
theorem zero_f32_rat : (FloatV.ofPat 0 : F32).toRat = 0 := by decide +kernel
theorem zero_bf16_val : (FloatV.ofPat 0 : BF16).toReal = 0 := by simp [FloatV.toReal, zero_bf16_rat]
theorem zero_f32_val : (FloatV.ofPat 0 : F32).toReal = 0 := by simp [FloatV.toReal, zero_f32_rat]
theorem zero_bf16_fin : (FloatV.ofPat 0 : BF16).isFinite = true := by decide +kernel
theorem zero_f32_fin : (FloatV.ofPat 0 : F32).isFinite = true := by decide +kernel

theorem setup_valid : setup_1_6144.x.Valid zeroMem ∧ setup_1_6144.w.Valid zeroMem ∧
    setup_1_6144.outw.Valid zeroMem ∧ setup_1_6144.outi.Valid zeroMem :=
  ⟨valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide),
    valid_zero _ (by decide) (by decide) (by decide), valid_zero _ (by decide) (by decide) (by decide)⟩

theorem xVal_eq {m : Mem} {d : Inst} {t h : Nat} {v : BF16}
    (hr : m.read .bf16 (d.x.ptr ((t * d.hidden + h : Nat) : Int)).addr = some v) :
    xVal m d t h = v.toReal := by
  unfold xVal; rw [hr]

theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨setup_1_6144, zeroMem, setup_1_6144_ok, ?_⟩
  have hxa := setup_x_aligned
  have hwa := setup_w_aligned
  have h0b := zero_bf16_val
  have h0f := zero_f32_val
  have hfb := zero_bf16_fin
  have hff := zero_f32_fin
  have hx : ∀ t h, t < 1 → h < 6144 → xVal zeroMem setup_1_6144 t h = 0 := fun t h ht hh =>
    (xVal_eq (zero_bf16 (hxa t h ht hh))).trans h0b
  have hhid : setup_1_6144.hidden = 6144 := rfl
  have hB : setup_1_6144.BLOCK = 8192 := rfl
  refine ⟨setup_valid.1, setup_valid.2.1, setup_valid.2.2.1, setup_valid.2.2.2,
    fun t h ht hh => ⟨_, zero_bf16 (hxa t h ht (by rwa [hhid] at hh)), hfb⟩,
    fun e h he hh => ⟨_, zero_f32 (hwa e h he (by rwa [hhid] at hh)), hff⟩, ?_⟩
  intro t e ht he
  have ht1 : t < 1 := ht
  have hlt : ∀ h, h ∈ Finset.range setup_1_6144.hidden → h < 6144 := fun h hh => by
    have := Finset.mem_range.mp hh; rwa [hhid] at this
  have hmu : mu zeroMem setup_1_6144 t e = 0 :=
    Finset.sum_eq_zero fun h hh => by rw [hx t h ht1 (hlt h hh)]; ring
  have hS : absS zeroMem setup_1_6144 t e = 0 :=
    Finset.sum_eq_zero fun h hh => by rw [hx t h ht1 (hlt h hh)]; simp
  rw [hmu, hS]
  refine ⟨by norm_num, ?_⟩
  have h := tolL_zero_le setup_1_6144 (by decide) (by decide)
  have h0 : 0 ≤ tolL setup_1_6144 0 := tolL_nonneg _ _ (le_refl _)
  unfold capTol
  rw [abs_zero, mul_zero, zero_add]
  have h1 : (1 + (2 : ℝ) ^ (-21 : ℤ)) * tolL setup_1_6144 0 ≤ (1 + 2 ^ (-21 : ℤ)) * 2 ^ (-100 : ℤ) :=
    mul_le_mul_of_nonneg_left h (by positivity)
  have h2 : (1 + (2 : ℝ) ^ (-21 : ℤ)) * 2 ^ (-100 : ℤ) + 2 ^ (-14 : ℤ) ≤ 1320 := by norm_num
  linarith

end Claims.MoeRouter
