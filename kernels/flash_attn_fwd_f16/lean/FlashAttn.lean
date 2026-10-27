import FlashAttn.Numeric
import TVA.Lemmas.Desc
import TVA.Lemmas.Claim
import TVA.Lemmas.Consistency
import TVA.Lemmas.Scalar

/-!
# Worked example: flash attention, forward (tutorial 06)

The source is `_attn_fwd` with `_attn_fwd_inner` from `tutorials/06-fused-attention.py`
(Triton 3.8.0), on the configuration this bundle covers:

* `STAGE = 1` (non-causal): `_attn_fwd_inner` runs once, with `STAGE = 3`, over
  `lo, hi = 0, N_CTX`;
* `FP8_OUTPUT = False`: `dtype = tl.float16`;
* `IS_HOPPER = True`: `acc = acc * alpha[:, None]` (no split of the accumulator);
* `N_CTX` a multiple of `BLOCK_M` and `BLOCK_N` (the kernel has no masks).

```python
for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    k = desc_k.load([offsetk_y, 0]).T
    qk = tl.dot(q, k)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
    qk = qk * qk_scale - m_ij[:, None]
    p = tl.math.exp2(qk)
    alpha = tl.math.exp2(m_i - m_ij)
    l_ij = tl.sum(p, 1)
    acc = acc * alpha[:, None]
    v = desc_v.load([offsetv_y, 0])
    p = p.to(dtype)
    acc = tl.dot(p, v, acc)
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    offsetk_y += BLOCK_N
    offsetv_y += BLOCK_N
```

`_attn_fwd` computes `start_m = program_id(0)`, `off_hz = program_id(1)`, the head's first row
`offset_y = off_z * (N_CTX * H) + off_h * N_CTX` of the flattened `[Z·H·N_CTX, HEAD_DIM]`
tensors, `m_i = -inf`, `l_i = 1`, `acc = 0`, `qk_scale = sm_scale * 1.44269504`, loads its `q`
tile, runs the loop, and stores `M = m_i + log2(l_i)` and `O = acc / l_i` (as `float16`). The
grid is `(cdiv(N_CTX, BLOCK_M), Z·H, 1)`.

The sections below are the four parts of the claim:

* **Model** (`attn_fwd`, `inner`): the source, line by line.
* **Specification** (`Post`): every output element against the exact softmax over the reals.
* **Conditions** (`Inst.ok`, `Pre`): the instances and inputs covered.
* **Assumptions** (`Assumptions`): `Compiler.Faithful` and the numeric rules, with constants.
* **Proof** (`main`, `consistent`).
-/
namespace TVAExamples.FlashAttn

open TVA TVA.Softmax TVA.Math.Float

/-! ## Model (prover-written, line by line) -/

/-- `tl.dot` of `float16` operands into `float32` (`input_precision` does not apply to
`float16`). -/
abbrev cfg : DotCfg := { a := .f16, b := .f16, acc := .f32 }

/-- `1.44269504` as Triton builds it: a Python float that fits is a `float32` constant
(`semantic.py` `to_tensor`), the nearest one, `0x3FB8AA3B`. -/
def log2e : F32 := FloatV.ofPat 0x3FB8AA3B
/-- `1.0` in `float32`. -/
def one32 : F32 := FloatV.ofPat 0x3F800000

/-- The loop-carried values of `_attn_fwd_inner`: `acc`, `l_i`, `m_i`, `offsetk_y`,
`offsetv_y`. -/
structure St (BM HD : Nat) where
  acc : Tensor [BM, HD] F32
  l : Tensor [BM] F32
  m : Tensor [BM] F32
  offk : I32
  offv : I32

/-- One iteration of the loop of `_attn_fwd_inner` (`STAGE = 3`, `float16`, Hopper). The
additions that consume a product are written as fusable: `qk * qk_scale - m_ij[:, None]` is
`tl.mul` then `tl.subc` (an `fma` when fused), `l_i * alpha + l_ij` is `tl.mul_add`. -/
def inner {BM HD : Nat} (desc_k desc_v : TensorDesc .f16) (q : Tensor [BM, HD] F16)
    (qk_scale : F32) (BN : Nat) (start_n : I32) (st : St BM HD) : Prog (St BM HD) := do
  let _start_n ← tl.multiple_of1 start_n BN
  let k : Tensor [BN, HD] F16 ← tl.desc_load desc_k [st.offk, 0]
  let qk ← tl.dot cfg q k.trans (tl.zeros [BM, BN] .f32)
  let rowmax ← tl.max1 true qk
  let mx ← tl.mul rowmax (tl.full [BM] qk_scale)
  let m_ij ← tl.maximum st.m mx
  let qks ← tl.mul qk (tl.full [BM, BN] qk_scale)
  let x ← tl.subc qks (m_ij.col BN)
  let p ← tl.exp2 x
  let d ← tl.sub st.m m_ij
  let alpha ← tl.exp2 d
  let l_ij ← tl.sum1 p
  let acc ← tl.mul st.acc (alpha.col HD)
  let v : Tensor [BN, HD] F16 ← tl.desc_load desc_v [st.offv, 0]
  let p16 : Tensor [BM, BN] F16 ← tl.cvt p
  let acc ← tl.dot cfg p16 v acc
  let l_i ← tl.mul_add st.l alpha l_ij
  return ⟨acc, l_i, m_ij, st.offk + IntV.ofNat BN, st.offv + IntV.ofNat BN⟩

/-- `_attn_fwd` with `STAGE = 1`. The descriptors are arguments: on Hopper the host builds
them (`TensorDescriptor`), or the kernel does (`tl.make_tensor_descriptor`, with
`warp_specialize`); both have shape `[Z·H·N_CTX, HEAD_DIM]`, strides `[HEAD_DIM, 1]`, the
blocks `[BLOCK_M, HEAD_DIM]` (`q`, `o`) and `[BLOCK_N, HEAD_DIM]` (`k`, `v`), and zero padding. -/
def attn_fwd (sm_scale : F32) (M : Ptr .f32) (H : I32) (desc_q desc_k desc_v desc_o : TensorDesc .f16)
    (N_CTX : I32) (HEAD_DIM BLOCK_M BLOCK_N : Nat) : Kernel where
  body ctx := do
    let start_m := tl.program_id ctx 0
    let off_hz := tl.program_id ctx 1
    let off_z := (← tl.idiv (Tensor.scalar off_hz) (Tensor.scalar H)).item
    let off_h := (← tl.irem (Tensor.scalar off_hz) (Tensor.scalar H)).item
    let offset_y := off_z * (N_CTX * H) + off_h * N_CTX
    let qo_offset_y := offset_y + start_m * (IntV.ofNat BLOCK_M : I32)
    let offs_m : Tensor [BLOCK_M] I32 := start_m * (IntV.ofNat BLOCK_M : I32) + tl.arange 0 BLOCK_M
    let m_i ← tl.sub (tl.zeros [BLOCK_M] .f32) (tl.full [BLOCK_M] posInf32)
    let l_i ← tl.add (tl.zeros [BLOCK_M] .f32) (tl.full [BLOCK_M] one32)
    let acc := tl.zeros [BLOCK_M, HEAD_DIM] .f32
    let qk_scale := (← tl.mul (Tensor.scalar sm_scale) (Tensor.scalar log2e)).item
    let q : Tensor [BLOCK_M, HEAD_DIM] F16 ← tl.desc_load desc_q [qo_offset_y, 0]
    let st ← tl.forRange (0 : I32) N_CTX (IntV.ofNat BLOCK_N)
      (⟨acc, l_i, m_i, offset_y + 0, offset_y + 0⟩ : St BLOCK_M HEAD_DIM)
      (inner desc_k desc_v q qk_scale BLOCK_N)
    let lg ← tl.log2 st.l
    let m_out ← tl.add st.m lg
    let acc ← tl.div st.acc (st.l.col HEAD_DIM)
    let m_ptrs : Tensor [BLOCK_M] (Ptr .f32) := (M + off_hz * N_CTX) + offs_m
    tl.store m_ptrs m_out
    let o16 : Tensor [BLOCK_M, HEAD_DIM] F16 ← tl.cvt acc
    tl.desc_store desc_o [qo_offset_y, 0] o16

/-! ## Specification -/

/-- An instance: the sizes (`Z`, `H`, `N_CTX`, `HEAD_DIM`, `BLOCK_M`, `BLOCK_N`), `sm_scale`,
the buffers (`q`, `k`, `v`, `o` of `float16`, `M` of `float32`, contiguous), and the bounds the
error constants use: `|q|, |k| ≤ R`, `|v| ≤ V`, scale `c ≤ C`. -/
structure Inst where
  Z : Nat
  H : Nat
  N : Nat
  HD : Nat
  BM : Nat
  BN : Nat
  sm_scale : F32
  q : Buf .f16
  k : Buf .f16
  v : Buf .f16
  o : Buf .f16
  M : Buf .f32
  R : ℝ
  V : ℝ
  C : ℝ

/-- The sizes and bounds of the error constants (`TVAExamples.FlashAttn.Err`). -/
def Inst.bnd (d : Inst) : Bnd := ⟨d.N, d.HD, d.BN, d.R, d.V, d.C⟩

/-- The device's scale `qk_scale = sm_scale ⊗ fl(1.44269504)` (one `float32` product). -/
def Inst.scale (d : Inst) : F32 := FloatV.mul d.sm_scale log2e

/-- The rows of the flattened tensors, `Z·H·N_CTX`. -/
def Inst.Y (d : Inst) : Nat := d.Z * d.H * d.N

/-- Row `(z·H + h)·N_CTX + i` holds query (key, value, output) `i` of head `(z, h)`. -/
def Inst.row (d : Inst) (z h i : Nat) : Nat := (z * d.H + h) * d.N + i

/-- The instances covered.
* The tutorial's configurations: `HEAD_DIM ∈ {16, …, 256}`, `BLOCK_M ∈ {64, 128}`,
  `BLOCK_N ∈ {32, 64, 128}`, `BLOCK_N ≤ HEAD_DIM` (`tl.static_assert`).
* `N_CTX` a multiple of `BLOCK_M` and `BLOCK_N` (no masks: otherwise a tile reads the next
  head's rows).
* The grid's `y` below `2^16`; the rows `Z·H·N_CTX`, plus one block of each kind, at most
  `2^31` (`int32` offsets, the loop bound and TMA coordinates).
* Contiguous buffers of the right lengths, the TMA ones 16-byte aligned; `o` and `M` disjoint
  from each other and from the inputs.
* The device scale `c` finite with `0 < c ≤ C`, and the bounds meet `Bnd.Ok`. -/
def Inst.ok (d : Inst) : Prop :=
  d.HD ∈ [16, 32, 64, 128, 256] ∧ d.BM ∈ [64, 128] ∧ d.BN ∈ [32, 64, 128] ∧ d.BN ≤ d.HD ∧
    1 ≤ d.Z ∧ 1 ≤ d.H ∧ d.Z * d.H < 2 ^ 16 ∧ 0 < d.N ∧ d.BM ∣ d.N ∧ d.BN ∣ d.N ∧
    d.Y + d.BM + d.BN ≤ 2 ^ 31 ∧
    d.q.len = d.Y * d.HD ∧ d.k.len = d.Y * d.HD ∧ d.v.len = d.Y * d.HD ∧ d.o.len = d.Y * d.HD ∧
    d.M.len = d.Y ∧
    d.q.base % 16 = 0 ∧ d.k.base % 16 = 0 ∧ d.v.base % 16 = 0 ∧ d.o.base % 16 = 0 ∧
    Disjoint d.o.bytes d.q.bytes ∧ Disjoint d.o.bytes d.k.bytes ∧ Disjoint d.o.bytes d.v.bytes ∧
    Disjoint d.o.bytes d.M.bytes ∧ Disjoint d.M.bytes d.q.bytes ∧ Disjoint d.M.bytes d.k.bytes ∧
    Disjoint d.M.bytes d.v.bytes ∧
    d.scale.isFinite = true ∧ 0 < d.scale.toReal ∧ d.scale.toReal ≤ d.C ∧ d.bnd.Ok

/-- The grid `(cdiv(N_CTX, BLOCK_M), Z·H, 1)`. -/
def grid (d : Inst) : Fin 3 → Nat := ![d.N / d.BM, d.Z * d.H, 1]

/-- The launch: the grid, and the source's block shapes (`offs_m`, `m_i`, `l_i`: `[BLOCK_M]`;
`offs_n`: `[BLOCK_N]`; `q`, `acc`, `o`: `[BLOCK_M, HEAD_DIM]`; `k`, `v`: `[BLOCK_N, HEAD_DIM]`;
`k.T`: `[HEAD_DIM, BLOCK_N]`; `qk`, `p`: `[BLOCK_M, BLOCK_N]`). -/
def launch (d : Inst) : Launch :=
  { grid := grid d, blocks := [[d.BM], [d.BN], [d.BM, d.HD], [d.BN, d.HD], [d.HD, d.BN], [d.BM, d.BN]] }

/-- A descriptor over a `[Y, HD]` row-major buffer with block `[B, HD]`. -/
def descOf (b : Buf .f16) (Y HD B : Nat) : TensorDesc .f16 :=
  { base := b.ptr 0, shape := [Y, HD], strides := [HD, 1], block := [B, HD] }

/-- The kernel as launched on instance `d`. -/
def model (d : Inst) : Kernel :=
  attn_fwd d.sm_scale (d.M.ptr 0) (IntV.ofNat d.H) (descOf d.q d.Y d.HD d.BM) (descOf d.k d.Y d.HD d.BN)
    (descOf d.v d.Y d.HD d.BN) (descOf d.o d.Y d.HD d.BM) (IntV.ofNat d.N) d.HD d.BM d.BN

/-- An `fp16` input element in range: finite, at most `B` in magnitude. -/
def InRange (init : Mem) (b : Buf .f16) (B : ℝ) : Prop :=
  ∀ j < b.len, ∃ x, init.read .f16 (b.ptr j).addr = some x ∧ x.isFinite = true ∧ |x.toReal| ≤ B

/-- The buffers are valid; the inputs are finite with `|q|, |k| ≤ R` and `|v| ≤ V`. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.q.Valid init ∧ d.k.Valid init ∧ d.v.Valid init ∧ d.o.Valid init ∧ d.M.Valid init ∧
    InRange init d.q d.R ∧ InRange init d.k d.R ∧ InRange init d.v d.V

/-- The value of element `j` of an `fp16` buffer in memory `m` (`0` if unreadable; `Pre` makes
every element readable). -/
noncomputable def val (m : Mem) (b : Buf .f16) (j : Nat) : ℝ :=
  ((m.read .f16 (b.ptr j).addr).map FloatV.toReal).getD 0

/-- The exact score of query row `y` against key row `y'`: `Σ_e q[y, e]·k[y', e]`. -/
noncomputable def score (d : Inst) (m : Mem) (y y' : Nat) : ℝ :=
  ∑ e ∈ Finset.range d.HD, val m d.q (y * d.HD + e) * val m d.k (y' * d.HD + e)

/-- The softmax weight `2^(c · score)` of key `j` for query `i` of head `(z, h)`. -/
noncomputable def weight (d : Inst) (m : Mem) (z h i j : Nat) : ℝ :=
  (2 : ℝ) ^ (d.scale.toReal * score d m (d.row z h i) (d.row z h j))

/-- `Σ_j 2^(c · score)`. -/
noncomputable def lse (d : Inst) (m : Mem) (z h i : Nat) : ℝ := ∑ j ∈ Finset.range d.N, weight d m z h i j

/-- **The reference output**: the softmax average of the values, over the reals. With
`c = sm_scale·log₂e`, `2^(c·s) = e^(sm_scale·s)`, so this is `softmax(sm_scale · q kᵀ) v`. -/
noncomputable def Oref (d : Inst) (m : Mem) (z h i e : Nat) : ℝ :=
  (∑ j ∈ Finset.range d.N, weight d m z h i j * val m d.v (d.row z h j * d.HD + e)) / lse d m z h i

/-- **The reference log-sum-exp** (base 2): `log₂ Σ_j 2^(c · score)`. -/
noncomputable def Mref (d : Inst) (m : Mem) (z h i : Nat) : ℝ := Real.logb 2 (lse d m z h i)

/-- **The specification.** For every head `(z, h)` and query `i`: `M[z, h, i]` is a finite
`float32` within `errM` of `Mref`, and every `O[z, h, i, e]` a finite `float16` within `errO`
of `Oref`; nothing outside `o` and `M` changes. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ z < d.Z, ∀ h < d.H, ∀ i < d.N,
    (∃ mm, fin.read .f32 (d.M.ptr (d.row z h i)).addr = some mm ∧ mm.isFinite = true ∧
      |mm.toReal - Mref d init z h i| ≤ Err.errM d.bnd) ∧
    ∀ e < d.HD, ∃ x, fin.read .f16 (d.o.ptr (d.row z h i * d.HD + e)).addr = some x ∧
      x.isFinite = true ∧ |x.toReal - Oref d init z h i e| ≤ Err.errO d.bnd) ∧
  ∀ x, x ∉ d.o.bytes → x ∉ d.M.bytes → fin x = init x

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## Assumptions (reviewed) -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on each covered instance the compiled kernel behaves as the
  model `model d` launched with `launch d`. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotTotal`: `tl.dot` of `float16` into `float32` has a result. -/
  dotTotal : Rules.Num.DotTotal P cfg
  /-- [rule] `Num.DotAcc`: a `K`-term `float16 × float16 → float32` dot product (`wgmma`) is
  within `(K + 2)·2^-22 · (|acc| + Σ|a·b|) + (K + 2)·2^-126` of `acc + Σ a·b`, for all finite
  operands. -/
  dotAcc : Rules.Num.DotAcc P cfg (fun _ => True) Err.relDot Err.absDot
  /-- [rule] `Num.MathTotal`: `exp2` of `float32` has a result. -/
  exp2Total : Rules.Num.MathTotal P .exp2 .binary32
  /-- [rule] `Num.MathAcc`: `tl.math.exp2` on `float32` is `ex2.approx.f32` (Triton 3.8.0:
  `math::Exp2Op` → `llvm.exp2.f32` → `EX2_APPROX_f32`, `llvm-NVPTXInstrInfo.td:1146-1151`,
  without `.ftz`): within `2^-22·2^x + 2^-149` of `2^x` (library issue L4). -/
  exp2Acc : Rules.Num.MathAcc P .exp2 .binary32 Set.univ Err.relE Err.absE
  /-- [rule] `Num.MathAt`: `ex2.approx.f32(-∞) = +0` (PTX ISA; library issue L4). -/
  exp2NegInf : Rules.Num.MathAt P .exp2 .binary32 negInf32 zero32
  /-- [rule] `Num.MathTotal`: `log2` of `float32` has a result. -/
  log2Total : Rules.Num.MathTotal P .log2 .binary32
  /-- [rule] `Num.MathAcc`: `tl.math.log2` on `float32` is `lg2.approx.f32`
  (`llvm-NVPTXIntrinsics.td:1645-1648`): within `2^-22·|log₂ x| + 2^-22` of `log₂ x` for
  positive finite `x` (library issue L4). -/
  log2Acc : Rules.Num.MathAcc P .log2 .binary32 Set.univ Err.relL Err.absL
  /-- [rule] `Num.DivTotal`: `float32` division has a result. -/
  divTotal : Rules.Num.DivTotal P .binary32
  /-- [rule] `Num.DivFull`: `float32` `/` is `div.full.f32`. -/
  divFull : Rules.Num.DivFull P

/-! ## Proof -/

/-! ### Descriptors over row-major buffers -/

section Desc
variable {b : Buf .f16} {Y HD B : Nat}

theorem descOf_valid (hY : Y < 2 ^ 31) (hHD : HD < 2 ^ 31) (h16 : HD * 2 % 16 = 0)
    (hb : b.base % 16 = 0) : (descOf b Y HD B).Valid := by
  refine ⟨by show 1 ≤ 2; norm_num, by show 2 ≤ 5; norm_num, rfl, rfl, fun k hk => ?_, ?_, rfl,
    fun k hk => ?_, by show (Padding.zero : Padding) ≠ .undef; decide, fun h => by cases h⟩
  · have hk' : k < 2 := hk
    interval_cases k
    · exact hY
    · exact hHD
  · show (b.ptr 0).addr % 16 = 0
    simpa using hb
  · have hk' : k = 0 := by have : k + 1 < 2 := hk; omega
    subst hk'
    exact h16

theorem I32.toInt_zero : IntV.toInt (0 : I32) = 0 := by decide

theorem descOf_access (hv : (descOf b Y HD B).Valid) (o : I32) :
    (descOf b Y HD B).Access ([o, 0].map IntV.toInt) [B, HD] :=
  ⟨hv, rfl, rfl, rfl, by simp [I32.toInt_zero]⟩

theorem descOf_coord0 {o : I32} {ro : Nat} (ho : o.toInt = ro) (l : Idx [B, HD]) :
    TensorDesc.coord ([o, 0].map IntV.toInt) l 0 = ro + l.1.1 := by
  obtain ⟨a, e, u⟩ := l
  simp [TensorDesc.coord, ho, Idx.coord]

theorem descOf_coord1 (o : I32) (l : Idx [B, HD]) :
    TensorDesc.coord ([o, 0].map IntV.toInt) l 1 = l.2.1.1 := by
  obtain ⟨a, e, u⟩ := l
  simp [TensorDesc.coord, I32.toInt_zero, Idx.coord]

/-- Lane `(a, e)` of the block at row `ro` is element `(ro + a)·HD + e`, inside the tensor. -/
theorem descOf_lane {o : I32} {ro : Nat} (ho : o.toInt = ro) (hfit : ro + B ≤ Y) (l : Idx [B, HD]) :
    (descOf b Y HD B).inbAt (TensorDesc.coord ([o, 0].map IntV.toInt) l) = true ∧
      ((descOf b Y HD B).elemAt (TensorDesc.coord ([o, 0].map IntV.toInt) l)).toNat =
        (ro + l.1.1) * HD + l.2.1.1 := by
  have h0 := descOf_coord0 (B := B) (HD := HD) ho l
  have h1 := descOf_coord1 (B := B) (HD := HD) o l
  have ha := l.1.2
  have he := l.2.1.2
  constructor
  · unfold TensorDesc.inbAt
    apply decide_eq_true
    intro k hk
    have hk' : k < 2 := hk
    interval_cases k
    · rw [h0]; show 0 ≤ (ro : Int) + l.1.1 ∧ (ro : Int) + l.1.1 < (Y : Int)
      constructor <;> omega
    · rw [h1]; show 0 ≤ (l.2.1.1 : Int) ∧ (l.2.1.1 : Int) < (HD : Int)
      constructor <;> omega
  · unfold TensorDesc.elemAt
    rw [show (descOf b Y HD B).rank = 2 from rfl, show List.range 2 = [0, 1] from rfl]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    have h0' : TensorDesc.coord [IntV.toInt o, IntV.toInt 0] l 0 = ro + l.1.1 := h0
    have h1' : TensorDesc.coord [IntV.toInt o, IntV.toInt 0] l 1 = l.2.1.1 := h1
    rw [h0', h1']
    show (((ro : Int) + l.1.1) * ((HD : Nat) : Int) + ((l.2.1.1 : Int) * ((1 : Nat) : Int) + 0)).toNat = _
    rw [show ((ro : Int) + l.1.1) * ((HD : Nat) : Int) + ((l.2.1.1 : Int) * ((1 : Nat) : Int) + 0) =
      (((ro + l.1.1) * HD + l.2.1.1 : Nat) : Int) by push_cast; ring]
    exact Int.toNat_natCast _

end Desc

theorem rowcol_lt {y Y e HD : Nat} (hy : y < Y) (he : e < HD) : y * HD + e < Y * HD :=
  calc y * HD + e < y * HD + HD := by omega
    _ = (y + 1) * HD := by ring
    _ ≤ Y * HD := Nat.mul_le_mul_right _ hy

section Tile
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
  {b : Buf .f16} {Y HD B : Nat}

/-- **`desc.load` of a full tile** of a row-major buffer: lane `(a, e)` holds element
`(ro + a)·HD + e`. -/
theorem safe_tile_load {o : I32} {ro : Nat} (ho : o.toInt = ro) (hfit : ro + B ≤ Y)
    (hlen : b.len = Y * HD) (hv : b.Valid m) (hvalid : (descOf b Y HD B).Valid) (hR : b.bytes ⊆ R)
    {k : Tensor [B, HD] F16 → Prog α}
    (hk : ∀ v : Tensor [B, HD] F16, (∀ (a : Fin B) (e : Fin HD),
      m.read .f16 (b.ptr (((ro + a) * HD + e : Nat) : Int)).addr = some (v.get2 a e)) → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.desc_load (descOf b Y HD B) [o, 0] >>= k) := by
  refine safe_desc_load b rfl (descOf_access hvalid o) (fun l _ => ?_) hv ?_ fun v h1 _ => hk v fun a e => ?_
  · rw [(descOf_lane ho hfit l).2, hlen]
    exact rowcol_lt (by have := l.1.2; omega) l.2.1.2
  · refine subset_trans (b.elems_sub_bytes ?_) hR
    rintro j ⟨l, -, rfl⟩
    beta_reduce
    rw [(descOf_lane (b := b) (HD := HD) ho hfit l).2, hlen]
    exact rowcol_lt (by have := l.1.2; omega) l.2.1.2
  · have hl := descOf_lane (b := b) (Y := Y) (HD := HD) (B := B) ho hfit (a, e, ())
    have := h1 (a, e, ()) hl.1
    rw [hl.2] at this; exact this

/-- The elements of a full tile at row `ro`. -/
def tileSet (ro B HD : Nat) : Set Nat := {j | ∃ a < B, ∃ e < HD, j = (ro + a) * HD + e}

/-- **`desc.store` of a full tile** to a row-major buffer. -/
theorem safe_tile_store {o : I32} {ro : Nat} (ho : o.toInt = ro) (hfit : ro + B ≤ Y)
    (hlen : b.len = Y * HD) (hv : b.Valid m) (hvalid : (descOf b Y HD B).Valid)
    {vals : Tensor [B, HD] F16} (hW : b.elems (tileSet ro B HD) ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m', (∀ (a : Fin B) (e : Fin HD),
        m'.read .f16 (b.ptr (((ro + a) * HD + e : Nat) : Int)).addr = some (vals.get2 a e)) →
      (∀ x, x ∉ b.elems (tileSet ro B HD) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R W Q m' (k ())) :
    Safe E R W Q m (tl.desc_store (descOf b Y HD B) [o, 0] vals >>= k) := by
  have himg : Buf.img (fun i : Idx [B, HD] =>
      ((descOf b Y HD B).elemAt (TensorDesc.coord ([o, 0].map IntV.toInt) i)).toNat)
      (fun i => (descOf b Y HD B).inbAt (TensorDesc.coord ([o, 0].map IntV.toInt) i)) ⊆ tileSet ro B HD := by
    rintro j ⟨l, -, rfl⟩
    exact ⟨l.1.1, l.1.2, l.2.1.1, l.2.1.2, (descOf_lane (b := b) (HD := HD) ho hfit l).2⟩
  refine safe_desc_store b rfl (descOf_access hvalid o) (fun l _ => ?_) (fun l l' _ _ h => ?_) hv
    (subset_trans (b.elems_mono himg) hW) fun m' h1 h2 h3 => hk m' (fun a e => ?_) (fun x hx => h2 x
      fun hx' => hx (b.elems_mono himg hx')) h3
  · rw [(descOf_lane (b := b) (HD := HD) ho hfit l).2, hlen]
    exact rowcol_lt (by have := l.1.2; omega) l.2.1.2
  · rw [(descOf_lane (b := b) (HD := HD) ho hfit l).2, (descOf_lane (b := b) (HD := HD) ho hfit l').2] at h
    obtain ⟨⟨a, ha⟩, ⟨e, he⟩, u⟩ := l
    obtain ⟨⟨a', ha'⟩, ⟨e', he'⟩, u'⟩ := l'
    have h1 : e = e' := by
      have k1 : ((ro + a) * HD + e) % HD = e := by
        rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt he
      have k2 : ((ro + a') * HD + e') % HD = e' := by
        rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt he'
      have h' : (ro + a) * HD + e = (ro + a') * HD + e' := h
      rw [← k1, ← k2, h']
    subst h1
    have h2 : ro + a = ro + a' := by
      have := Nat.add_right_cancel h
      exact Nat.eq_of_mul_eq_mul_right (by omega) this
    have : a = a' := by omega
    subst this
    rfl
  · have hl := descOf_lane (b := b) (Y := Y) (HD := HD) (B := B) ho hfit (a, e, ())
    have := h1 (a, e, ()) hl.1
    rw [hl.2] at this; exact this

end Tile

/-! ### Input values -/

theorem val_of_read {m : Mem} {b : Buf .f16} {j : Nat} {x : F16}
    (h : m.read .f16 (b.ptr (j : Int)).addr = some x) : val m b j = x.toReal := by
  unfold val; rw [h]; rfl

theorem inRange_get {init : Mem} {b : Buf .f16} {B : ℝ} (h : InRange init b B) {j : Nat}
    (hj : j < b.len) {x : F16} (hx : init.read .f16 (b.ptr (j : Int)).addr = some x) :
    x.isFinite = true ∧ |x.toReal| ≤ B := by
  obtain ⟨y, hy, hf, hb⟩ := h j hj
  rw [hx] at hy
  cases hy
  exact ⟨hf, hb⟩

theorem val_le {init : Mem} {b : Buf .f16} {B : ℝ} (h : InRange init b B) {j : Nat}
    (hj : j < b.len) : |val init b j| ≤ B := by
  obtain ⟨y, hy, -, hb⟩ := h j hj
  rw [val_of_read hy]; exact hb

theorem score_le {d : Inst} {init : Mem} (hq : InRange init d.q d.R) (hk : InRange init d.k d.R)
    (hql : d.q.len = d.Y * d.HD) (hkl : d.k.len = d.Y * d.HD) (hR : 0 ≤ d.R) {y y' : Nat}
    (hy : y < d.Y) (hy' : y' < d.Y) : |score d init y y'| ≤ d.HD * d.R ^ 2 := by
  unfold score
  calc |∑ e ∈ Finset.range d.HD, val init d.q (y * d.HD + e) * val init d.k (y' * d.HD + e)|
      ≤ ∑ e ∈ Finset.range d.HD, |val init d.q (y * d.HD + e) * val init d.k (y' * d.HD + e)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _e ∈ Finset.range d.HD, d.R ^ 2 := Finset.sum_le_sum fun e he => by
        rw [abs_mul, sq]
        have he' := Finset.mem_range.mp he
        exact mul_le_mul (val_le hq (by rw [hql]; exact rowcol_lt hy he'))
          (val_le hk (by rw [hkl]; exact rowcol_lt hy' he')) (abs_nonneg _) hR
    _ = d.HD * d.R ^ 2 := by simp

theorem zeros_get {s : Shape} (i : Idx s) : (tl.zeros s .f32).get i = zero32 := rfl
theorem zeros_get2 {M N : Nat} (i : Fin M) (j : Fin N) : (tl.zeros [M, N] .f32).get2 i j = zero32 := rfl

/-- **One `tl.dot` element**, `float16` into `float32`, under `DotAcc`, over the reals. -/
theorem dot_val {P : Platform} (hdot : Rules.Num.DotAcc P cfg (fun _ => True) Err.relDot Err.absDot)
    {M N K : Nat} {a : Tensor [M, K] F16} {bt : Tensor [K, N] F16} {c r : Tensor [M, N] F32}
    (h : P.dot cfg a bt c r) (i : Fin M) (n : Fin N)
    (ha : ∀ k, (a.get2 i k).isFinite = true) (hb : ∀ k, (bt.get2 k n).isFinite = true)
    (hc : (c.get2 i n).isFinite = true) (hK : K ≤ 2 ^ 20)
    (hroom : |(c.get2 i n).toReal| + ∑ k, |(a.get2 i k).toReal * (bt.get2 k n).toReal| ≤ 2 ^ 120) :
    (r.get2 i n).isFinite = true ∧
      |(r.get2 i n).toReal - ((c.get2 i n).toReal + ∑ k, (a.get2 i k).toReal * (bt.get2 k n).toReal)| ≤
        (Err.relDot K : ℝ) * (|(c.get2 i n).toReal| + ∑ k, |(a.get2 i k).toReal * (bt.get2 k n).toReal|) +
          (Err.absDot K : ℝ) := by
  have hX : ((|(c.get2 i n).toRat| + ∑ k, |(a.get2 i k).toRat * (bt.get2 k n).toRat| : ℚ) : ℝ) ≤ 2 ^ 120 := by
    push_cast; simpa only [toReal_def] using hroom
  have hX' : |(c.get2 i n).toRat| + ∑ k, |(a.get2 i k).toRat * (bt.get2 k n).toRat| ≤ 2 ^ 120 := by
    exact_mod_cast hX
  have hrel : Err.relDot K ≤ 1 := by
    unfold Err.relDot
    have : (K : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hK
    rw [zpow_neg]; norm_num; linarith
  have habs : Err.absDot K ≤ 1 := by
    unfold Err.absDot
    have : (K : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hK
    rw [zpow_neg]; norm_num; linarith
  have hrel0 : 0 ≤ Err.relDot K := by unfold Err.relDot; positivity
  have hX0 : 0 ≤ |(c.get2 i n).toRat| + ∑ k, |(a.get2 i k).toRat * (bt.get2 k n).toRat| :=
    add_nonneg (abs_nonneg _) (Finset.sum_nonneg fun _ _ => abs_nonneg _)
  have hroomQ : |(c.get2 i n).toRat| + ∑ k, |(a.get2 i k).toRat * (bt.get2 k n).toRat| +
      (Err.relDot K * (|(c.get2 i n).toRat| + ∑ k, |(a.get2 i k).toRat * (bt.get2 k n).toRat|) + Err.absDot K) ≤
      cfg.acc.bound := by
    show _ ≤ Format.binary32.maxVal
    have := mul_le_mul hrel hX' hX0 (by norm_num)
    have := maxVal32_ge
    linarith [show (2 : ℚ) ^ 120 + 1 * 2 ^ 120 + 1 ≤ 2 ^ 127 by norm_num]
  obtain ⟨vr, hvr, he⟩ := hdot a bt c r h i n (fun k => (a.get2 i k).toRat) (fun k => (bt.get2 k n).toRat)
    (c.get2 i n).toRat (fun k => val?_f16 (ha k)) (fun k => val?_f16 (hb k)) (fun _ => ⟨trivial, trivial⟩)
    (val?_f32 hc) hroomQ
  obtain ⟨hrf, hrv⟩ := of_val?_f32 hvr
  refine ⟨hrf, ?_⟩
  have he' : ((|vr - ((c.get2 i n).toRat + ∑ k, (a.get2 i k).toRat * (bt.get2 k n).toRat)| : ℚ) : ℝ) ≤
      ((Err.relDot K * (|(c.get2 i n).toRat| + ∑ k, |(a.get2 i k).toRat * (bt.get2 k n).toRat|) + Err.absDot K : ℚ) : ℝ) := by
    exact_mod_cast he
  push_cast at he'
  rw [← hrv] at he'
  simpa only [toReal_def] using he'

/-! ### Programs, footprints and the loop invariant -/

section Prog
variable (d : Inst)

/-- Program `p` handles the query block `start_m = sm p` of head `off_hz = hz p`. -/
def sm (p : Nat) : Nat := p % (d.N / d.BM)
def hz (p : Nat) : Nat := p / (d.N / d.BM) % (d.Z * d.H)
/-- The head's first row, and the program's first query row. -/
def hrow (p : Nat) : Nat := hz d p * d.N
def qrow (p : Nat) : Nat := hrow d p + sm d p * d.BM

/-- Every program reads the inputs … -/
def R : Set Int := d.q.bytes ∪ d.k.bytes ∪ d.v.bytes
/-- … and writes its tile of `o` and its rows of `M`. -/
def MT (p : Nat) : Set Nat := {j | ∃ r < d.BM, j = qrow d p + r}
def W (p : Nat) : Set Int := d.o.elems (tileSet (qrow d p) d.BM d.HD) ∪ d.M.elems (MT d p)

/-- The exact scores and values of row `r` of program `p`. -/
noncomputable def sref (init : Mem) (p r j : Nat) : ℝ := score d init (qrow d p + r) (hrow d p + j)
noncomputable def vref (init : Mem) (p j : Nat) (e : Fin d.HD) : ℝ :=
  val init d.v ((hrow d p + j) * d.HD + e)

/-- **The loop invariant** before iteration `t`: memory unchanged, the offsets at block `t`, and
each row either in its initial state (`t = 0`) or satisfying `RowInv` after `t` blocks. -/
def Inv (init : Mem) (p t : Nat) (st : St d.BM d.HD) (m : Mem) : Prop :=
  m = init ∧ st.offk.toInt = ((hrow d p + t * d.BN : Nat) : Int) ∧
    st.offv.toInt = ((hrow d p + t * d.BN : Nat) : Int) ∧
    ∀ r : Fin d.BM, (t = 0 ∧ st.m.get1 r = negInf32 ∧ (st.l.get1 r).isFinite = true ∧
        ∀ e, st.acc.get2 r e = zero32) ∨
      (0 < t ∧ RowInv d.bnd d.scale.toReal (sref d init p r) (vref d init p) t (st.m.get1 r) (st.l.get1 r)
        (fun e => st.acc.get2 r e))

/-- What program `p` leaves in its own memory: row `r`'s `M` and `O` within the bounds. -/
def Q (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ r < d.BM,
    (∃ mm, m.read .f32 (d.M.ptr (qrow d p + r : Nat)).addr = some mm ∧ mm.isFinite = true ∧
      |mm.toReal - Real.logb 2 (∑ j ∈ Finset.range d.N, (2 : ℝ) ^ (d.scale.toReal * sref d init p r j))| ≤
        Err.errM d.bnd) ∧
    ∀ e : Fin d.HD, ∃ x, m.read .f16 (d.o.ptr (((qrow d p + r) * d.HD + e : Nat) : Int)).addr = some x ∧
      x.isFinite = true ∧
      |x.toReal - (∑ j ∈ Finset.range d.N, (2 : ℝ) ^ (d.scale.toReal * sref d init p r j) * vref d init p j e) /
        (∑ j ∈ Finset.range d.N, (2 : ℝ) ^ (d.scale.toReal * sref d init p r j))| ≤ Err.errO d.bnd

end Prog

/-! ### Facts about covered instances -/

section Facts
variable {d : Inst} (hd : d.ok)
include hd

theorem HD_ge : 16 ≤ d.HD := by obtain ⟨h, -⟩ := hd; simp at h; omega
theorem HD_le : d.HD ≤ 256 := by obtain ⟨h, -⟩ := hd; simp at h; omega
theorem HD16 : d.HD * 2 % 16 = 0 := by obtain ⟨h, -⟩ := hd; simp at h; omega
theorem BM_pos : 0 < d.BM := by obtain ⟨-, h, -⟩ := hd; simp at h; omega
theorem BM_le : d.BM ≤ 128 := by obtain ⟨-, h, -⟩ := hd; simp at h; omega
theorem BN_pos : 0 < d.BN := by obtain ⟨-, -, h, -⟩ := hd; simp at h; omega
theorem BN_le : d.BN ≤ 128 := by obtain ⟨-, -, h, -⟩ := hd; simp at h; omega
theorem N_pos : 0 < d.N := hd.2.2.2.2.2.2.2.1
theorem BMN : d.BM ∣ d.N := hd.2.2.2.2.2.2.2.2.1
theorem BNN : d.BN ∣ d.N := hd.2.2.2.2.2.2.2.2.2.1
theorem ZH_lt : d.Z * d.H < 2 ^ 16 := hd.2.2.2.2.2.2.1
theorem H_pos : 0 < d.H := hd.2.2.2.2.2.1
theorem ZH_pos : 0 < d.Z * d.H := Nat.mul_pos hd.2.2.2.2.1 hd.2.2.2.2.2.1
theorem Y31 : d.Y + d.BM < 2 ^ 31 := by
  have := hd.2.2.2.2.2.2.2.2.2.2.1; have := BN_pos hd; omega
theorem YBN : d.Y + d.BN ≤ 2 ^ 31 := by have := hd.2.2.2.2.2.2.2.2.2.2.1; omega
theorem N_le_Y : d.N ≤ d.Y := by
  unfold Inst.Y; have := ZH_pos hd; exact Nat.le_mul_of_pos_left _ this
theorem bnd_ok : d.bnd.Ok := hd.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2

theorem g0_pos : 0 < d.N / d.BM := Nat.div_pos (Nat.le_of_dvd (N_pos hd) (BMN hd)) (BM_pos hd)

end Facts

theorem size_eq (d : Inst) : (launch d).size = d.N / d.BM * (d.Z * d.H) := by
  show d.N / d.BM * (d.Z * d.H) * 1 = _; ring

section Arith
variable {d : Inst} (hd : d.ok) {p : Nat} (hp : p < (launch d).size)
include hd hp

omit hd in
theorem div_lt : p / (d.N / d.BM) < d.Z * d.H := by
  rw [size_eq] at hp
  exact Nat.div_lt_of_lt_mul hp

omit hd in
theorem hz_eq : hz d p = p / (d.N / d.BM) := Nat.mod_eq_of_lt (div_lt hp)

omit hd in
theorem hz_lt : hz d p < d.Z * d.H := by rw [hz_eq hp]; exact div_lt hp

omit hp in
theorem sm_lt : sm d p < d.N / d.BM := Nat.mod_lt _ (g0_pos hd)

theorem qrow_eq : qrow d p = p * d.BM := by
  unfold qrow hrow sm
  rw [hz_eq hp]
  have hN : d.N = d.N / d.BM * d.BM := (Nat.div_mul_cancel (BMN hd)).symm
  have h := Nat.div_add_mod' p (d.N / d.BM)
  set g := d.N / d.BM
  calc p / g * d.N + p % g * d.BM = (p / g * g + p % g) * d.BM := by rw [hN]; ring
    _ = p * d.BM := by rw [h]

omit hd in
theorem hrow_fit : hrow d p + d.N ≤ d.Y := by
  unfold hrow Inst.Y
  have := hz_lt hp
  calc hz d p * d.N + d.N = (hz d p + 1) * d.N := by ring
    _ ≤ d.Z * d.H * d.N := Nat.mul_le_mul_right _ this

omit hp in
theorem qrow_fit : qrow d p + d.BM ≤ hrow d p + d.N := by
  unfold qrow
  have h1 := sm_lt (p := p) hd
  have h2 : (sm d p + 1) * d.BM ≤ d.N / d.BM * d.BM := Nat.mul_le_mul_right _ h1
  rw [Nat.div_mul_cancel (BMN hd)] at h2
  nlinarith

end Arith

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hg0 : d.N / d.BM < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.div_le_self _ _) (by have := N_le_Y hd; have := Y31 hd; omega)
  have hZH := ZH_lt hd
  obtain ⟨hHD, hBM, hBN, -⟩ := hd
  refine ⟨hg0, hZH, by simp [launch, grid], rfl, ?_⟩
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hHD hBM hBN
  intro s hs
  simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hHD with h1 | h1 | h1 | h1 | h1 <;> rcases hBM with h2 | h2 <;> rcases hBN with h3 | h3 | h3 <;>
    rw [h1, h2, h3] at hs <;>
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl <;>
    refine ⟨fun k hk => ?_, by decide⟩ <;>
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hk <;>
    rcases hk with rfl | rfl <;>
    first | exact ⟨4, rfl⟩ | exact ⟨5, rfl⟩ | exact ⟨6, rfl⟩ | exact ⟨7, rfl⟩ | exact ⟨8, rfl⟩

/-- **The scores**: `qk[r, k]` is within `ε₁` of the exact score of query row `qrow + r` against
key row `ro + k`. -/
theorem qk_val {P : Platform} (hdot : Rules.Num.DotAcc P cfg (fun _ => True) Err.relDot Err.absDot)
    {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat} (hp : p < (launch d).size)
    {qt : Tensor [d.BM, d.HD] F16}
    (hqt : ∀ (a : Fin d.BM) (e : Fin d.HD),
      init.read .f16 (d.q.ptr (((qrow d p + a) * d.HD + e : Nat) : Int)).addr = some (qt.get2 a e))
    {ro : Nat} (hfit : ro + d.BN ≤ d.Y) {kt : Tensor [d.BN, d.HD] F16}
    (hkt : ∀ (a : Fin d.BN) (e : Fin d.HD),
      init.read .f16 (d.k.ptr (((ro + a) * d.HD + e : Nat) : Int)).addr = some (kt.get2 a e))
    {qk : Tensor [d.BM, d.BN] F32} (h : P.dot cfg qt kt.trans (tl.zeros [d.BM, d.BN] .f32) qk)
    (r : Fin d.BM) (k : Fin d.BN) :
    (qk.get2 r k).isFinite = true ∧
      |(qk.get2 r k).toReal - score d init (qrow d p + r) (ro + k)| ≤ Err.ε₁ d.bnd := by
  have hb := bnd_ok hd
  have hHD := HD_le hd
  have hy : qrow d p + r < d.Y := by
    have := qrow_fit (p := p) hd; have := hrow_fit (d := d) hp; have := r.2; omega
  have hy' : ro + k < d.Y := by have := k.2; omega
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, hql, hkl, -⟩ := id hd
  obtain ⟨-, -, -, -, -, hRq, hRk, -⟩ := id hpre
  have hq : ∀ e : Fin d.HD, (qt.get2 r e).isFinite = true ∧ |(qt.get2 r e).toReal| ≤ d.R := fun e =>
    inRange_get hRq (by rw [hql]; exact rowcol_lt hy e.2) (hqt r e)
  have hk : ∀ e : Fin d.HD, (kt.get2 k e).isFinite = true ∧ |(kt.get2 k e).toReal| ≤ d.R := fun e =>
    inRange_get hRk (by rw [hkl]; exact rowcol_lt hy' e.2) (hkt k e)
  have hR0 : 0 ≤ d.R := hb.hR
  have hsum : ∑ e : Fin d.HD, |(qt.get2 r e).toReal * (kt.get2 k e).toReal| ≤ d.HD * d.R ^ 2 := by
    calc ∑ e : Fin d.HD, |(qt.get2 r e).toReal * (kt.get2 k e).toReal| ≤ ∑ _e : Fin d.HD, d.R ^ 2 :=
          Finset.sum_le_sum fun e _ => by
            rw [abs_mul, sq]; exact mul_le_mul (hq e).2 (hk e).2 (abs_nonneg _) hR0
      _ = d.HD * d.R ^ 2 := by simp
  have hS : (d.HD : ℝ) * d.R ^ 2 ≤ 2 ^ 41 := by
    have h1 : (d.HD : ℝ) ≤ 256 := by exact_mod_cast hHD
    have h2 : d.R ^ 2 ≤ 65504 ^ 2 := pow_le_pow_left₀ hR0 hb.hR16 2
    calc (d.HD : ℝ) * d.R ^ 2 ≤ 256 * 65504 ^ 2 := mul_le_mul h1 h2 (by positivity) (by norm_num)
      _ ≤ 2 ^ 41 := by norm_num
  obtain ⟨hf, he⟩ := dot_val hdot h r k (fun e => (hq e).1) (fun e => (hk e).1)
    (by rw [zeros_get2]; exact zero32_isFinite) (by omega) (by
      rw [zeros_get2, zero32_toReal, abs_zero, zero_add]
      exact le_trans hsum (le_trans hS (by norm_num)))
  refine ⟨hf, ?_⟩
  have hscore : score d init (qrow d p + r) (ro + k) =
      ∑ e : Fin d.HD, (qt.get2 r e).toReal * (kt.trans.get2 e k).toReal := by
    unfold score
    rw [Finset.sum_range]
    refine Finset.sum_congr rfl fun e _ => ?_
    rw [val_of_read (hqt r e), val_of_read (hkt k e)]
    rfl
  rw [hscore]
  rw [zeros_get2, zero32_toReal, zero_add, abs_zero, zero_add] at he
  have hsum' : ∑ e : Fin d.HD, |(qt.get2 r e).toReal * (kt.trans.get2 e k).toReal| ≤ Err.S d.bnd := hsum
  have hrel : (0 : ℝ) ≤ (Err.relDot d.HD : ℝ) := by exact_mod_cast Err.relDot_nonneg d.HD
  calc _ ≤ _ := he
    _ ≤ (Err.relDot d.HD : ℝ) * Err.S d.bnd + (Err.absDot d.HD : ℝ) := by
      have := mul_le_mul_of_nonneg_left hsum' hrel; linarith
    _ = Err.ε₁ d.bnd := rfl

/-! ### One iteration of the loop -/

theorem safe_step {P : Platform} {K : Claim.Behaviour Inst} (hA : Assumptions K P) {d : Inst}
    (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat} (hp : p < (launch d).size)
    {qt : Tensor [d.BM, d.HD] F16}
    (hqt : ∀ (a : Fin d.BM) (e : Fin d.HD),
      init.read .f16 (d.q.ptr (((qrow d p + a) * d.HD + e : Nat) : Int)).addr = some (qt.get2 a e))
    {t : Nat} (ht : t < Err.T d.bnd) {start_n : I32} (hsn : start_n.toInt = t * d.BN)
    {st : St d.BM d.HD} {m : Mem} (hI : Inv d init p t st m) :
    Safe ((launch d).env P) (R d) (W d p) (fun st' m' => Inv d init p (t + 1) st' m') m
      (inner (descOf d.k d.Y d.HD d.BN) (descOf d.v d.Y d.HD d.BN) qt d.scale d.BN start_n st) := by
  obtain ⟨hm, hk0, hv0, hrows⟩ := hI
  subst m
  have hb := bnd_ok hd
  have hTN : (t + 1) * d.BN ≤ d.N := (Nat.le_div_iff_mul_le (BN_pos hd)).mp ht
  have hfitk : hrow d p + t * d.BN + d.BN ≤ d.Y := by
    have := hrow_fit (d := d) hp; rw [Nat.succ_mul] at hTN; omega
  have hY := Y31 hd
  have hHD := HD_le hd
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, hql, hkl, hvl, -, -, -, hk16, hv16, -⟩ := id hd
  obtain ⟨hvq, hvk, hvv, -, -, hRq, hRk, hRv⟩ := id hpre
  unfold inner
  refine safe_multiple_of1 (by show (d.BN : Int) ∣ start_n.toInt; rw [hsn]; exact Dvd.intro_left _ rfl) ?_
  refine safe_tile_load (b := d.k) hk0 hfitk hkl hvk (descOf_valid (by omega) (by omega) (HD16 hd) hk16)
    (fun x hx => Or.inl (Or.inr hx)) fun kt hkt => ?_
  refine safe_dot (hA.dotTotal _ _ _) fun qk hqk => ?_
  refine safe_max1 (BN_pos hd) true fun rowmax hrm => ?_
  refine safe_mul fun mx hmx => ?_
  refine safe_maximum fun mij hmij => ?_
  refine safe_mul fun qks hqks => ?_
  refine safe_subc fun x hx => ?_
  refine safe_math (fun i => hA.exp2Total _) fun pp hpp => ?_
  refine safe_sub fun dd hdd => ?_
  refine safe_math (fun i => hA.exp2Total _) fun alpha halpha => ?_
  refine safe_sum1 (BN_pos hd) fun lsum hls => ?_
  refine safe_mul fun accs haccs => ?_
  refine safe_tile_load (b := d.v) hv0 hfitk hvl hvv (descOf_valid (by omega) (by omega) (HD16 hd) hv16)
    (fun x hx => Or.inr hx) fun vt hvt => ?_
  refine safe_cvt ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩ fun p16 hp16 => ?_
  refine safe_dot (hA.dotTotal _ _ _) fun acc' hacc' => ?_
  refine safe_mul_add fun l' hl' => ?_
  have hBN31 : d.BN < 2 ^ 31 := by have := BN_le hd; omega
  have hoff : ∀ o : I32, o.toInt = ((hrow d p + t * d.BN : Nat) : Int) →
      (o + IntV.ofNat d.BN).toInt = ((hrow d p + (t + 1) * d.BN : Nat) : Int) := fun o ho => by
    rw [IntV.I32.add_nat ho (IntV.I32.toInt_ofNat hBN31) (by omega)]
    push_cast; ring
  refine .ret ⟨rfl, hoff _ hk0, hoff _ hv0, fun r => Or.inr ⟨by omega, ?_⟩⟩
  -- one row
  have hy : qrow d p + r < d.Y := by
    have := qrow_fit (p := p) hd; have := hrow_fit (d := d) hp; have := r.2; omega
  have hs : ∀ j < d.N, |sref d init p r j| ≤ Err.S d.bnd := fun j hj =>
    score_le hRq hRk hql hkl hb.hR hy (by have := hrow_fit (d := d) hp; omega)
  have hv : ∀ j < d.N, ∀ e : Fin d.HD, |vref d init p j e| ≤ d.V := fun j hj e =>
    val_le hRv (by rw [hvl]; exact rowcol_lt (by have := hrow_fit (d := d) hp; omega) e.2)
  have hqk' : ∀ k : Fin d.BN, (qk.get2 r k).isFinite = true ∧
      |(qk.get2 r k).toReal - sref d init p r (t * d.BN + k)| ≤ Err.ε₁ d.bnd := fun k => by
    have := qk_val hA.dotAcc hd hpre hp hqt hfitk hkt hqk r k
    unfold sref; rwa [← Nat.add_assoc]
  have hvt' : ∀ (k : Fin d.BN) (e : Fin d.HD), (vt.get2 k e).isFinite = true ∧
      (vt.get2 k e).toReal = vref d init p (t * d.BN + k) e := fun k e => by
    have hk' : hrow d p + t * d.BN + k < d.Y := by have := k.2; omega
    refine ⟨(inRange_get hRv (by rw [hvl]; exact rowcol_lt hk' e.2) (hvt k e)).1, ?_⟩
    unfold vref; rw [← Nat.add_assoc, val_of_read (hvt k e)]
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hsf, hs0, hsC, -⟩ := id hd
  exact row_step hb hA.exp2Acc hA.exp2NegInf (c := d.scale.toReal) (s := sref d init p r)
    (vv := vref d init p) hs0 hsC hs hv ht (hrows r) (qk := fun k => qk.get2 r k) hqk' (hrm r) hsf rfl
    (hmx (r, ())) (hmij (r, ())) (fun k => hqks (r, k, ())) (fun k => hx (r, k, ()))
    (fun k => hpp (r, k, ())) (hdd (r, ())) (halpha (r, ())) (hls r) (fun e => haccs (r, e, ()))
    (fun k => hp16 (r, k, ())) hvt'
    (fun hp16f e vc hvc hroom => hA.dotAcc p16 vt accs acc' hacc' r e (fun k => (p16.get2 r k).toRat)
      (fun k => (vt.get2 k e).toRat) vc (fun k => val?_f16 (hp16f k)) (fun k => val?_f16 (hvt' k e).1)
      (fun _ => ⟨trivial, trivial⟩) hvc hroom)
    (hl' (r, ()))

/-! ### One program -/

theorem one32_isFinite : one32.isFinite = true := by decide

theorem ptr_offset2 {dt : DType} (b : Buf dt) (x y : Int) : ((b.ptr 0).offset x).offset y = b.ptr (x + y) := by
  simp only [Buf.ptr, Ptr.offset]; congr 1; ring

theorem safe_program {P : Platform} {K : Claim.Behaviour Inst} (hA : Assumptions K P) {d : Inst}
    (hd : d.ok) {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d) (W d i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hp : (i : Nat) < (launch d).size := i.isLt
  have hb := bnd_ok hd
  have hY := Y31 hd
  have hYBN := YBN hd
  have hNY := N_le_Y hd
  have hBM := BM_pos hd
  have hBN := BN_pos hd
  have hBN128 := BN_le hd
  have hHD := HD_le hd
  have hH := H_pos hd
  have hzlt := hz_lt hp
  have hsmlt := sm_lt (p := i) hd
  have hqf := qrow_fit (p := i) hd
  have hhf := hrow_fit (d := d) hp
  obtain ⟨-, -, -, -, -, -, hZH, -, -, hBNN, -, hql, hkl, hvl, hol, hMl, hq16, -, -, hob16, -⟩ := id hd
  obtain ⟨hvq, hvk, hvv, hvo, hvM, hRq, hRk, hRv⟩ := id hpre
  have hpid0 : ((launch d).ctx i).pid 0 = sm d i := rfl
  have hpid1 : ((launch d).ctx i).pid 1 = hz d i := rfl
  have hHZ : d.H ≤ d.Z * d.H := Nat.le_mul_of_pos_left _ (ZH_pos hd |> fun h => by
    exact Nat.pos_of_mul_pos_right h)
  have hNH : d.N * d.H ≤ d.Y := by unfold Inst.Y; nlinarith [hd.2.2.2.2.1]
  have hz31 : hz d i < 2 ^ 31 := by omega
  have hsm31 : sm d i < 2 ^ 31 := by
    have : d.N / d.BM ≤ d.N := Nat.div_le_self _ _; omega
  have hpz : (tl.program_id ((launch d).ctx i) 1).toInt = (hz d i : Int) := by
    unfold tl.program_id; rw [hpid1]; exact IntV.I32.toInt_ofNat hz31
  have hpm : (tl.program_id ((launch d).ctx i) 0).toInt = (sm d i : Int) := by
    unfold tl.program_id; rw [hpid0]; exact IntV.I32.toInt_ofNat hsm31
  have hHv : (IntV.ofNat d.H : I32).toInt = (d.H : Int) := IntV.I32.toInt_ofNat (by omega)
  have hNv : (IntV.ofNat d.N : I32).toInt = (d.N : Int) := IntV.I32.toInt_ofNat (by omega)
  have hBMv : (IntV.ofNat d.BM : I32).toInt = (d.BM : Int) := IntV.I32.toInt_ofNat (by omega)
  have hBNv : (IntV.ofNat d.BN : I32).toInt = (d.BN : Int) := IntV.I32.toInt_ofNat (by omega)
  have h0v : (0 : I32).toInt = 0 := I32.toInt_zero
  dsimp only [model, attn_fwd]
  refine safe_idiv1 (by rw [hpz]; positivity) (by rw [hHv]; exact_mod_cast hH) fun rz hrz => ?_
  refine safe_irem1 (by rw [hpz]; positivity) (by rw [hHv]; exact_mod_cast hH) fun rh hrh => ?_
  rw [hpz, hHv] at hrz hrh
  have hrz' : rz.item.toInt = ((hz d i / d.H : Nat) : Int) := by rw [hrz]; push_cast; rfl
  have hrh' : rh.item.toInt = ((hz d i % d.H : Nat) : Int) := by rw [hrh]; push_cast; rfl
  have hdm : hz d i / d.H * (d.N * d.H) + hz d i % d.H * d.N = hrow d i := by
    unfold hrow
    have := Nat.div_add_mod (hz d i) d.H
    calc hz d i / d.H * (d.N * d.H) + hz d i % d.H * d.N = (d.H * (hz d i / d.H) + hz d i % d.H) * d.N := by ring
      _ = hz d i * d.N := by rw [this]
  have hle1 : hz d i / d.H * (d.N * d.H) ≤ hrow d i := by rw [← hdm]; omega
  have hle2 : hz d i % d.H * d.N ≤ hrow d i := by rw [← hdm]; omega
  have hoy : (rz.item * (IntV.ofNat d.N * IntV.ofNat d.H) + rh.item * IntV.ofNat d.N).toInt =
      ((hrow d i : Nat) : Int) := by
    rw [IntV.I32.add_nat (IntV.I32.mul_nat hrz' (IntV.I32.mul_nat hNv hHv (by omega)) (by omega))
      (IntV.I32.mul_nat hrh' hNv (by omega)) (by omega), hdm]
  have hqo : (rz.item * (IntV.ofNat d.N * IntV.ofNat d.H) + rh.item * IntV.ofNat d.N +
      (tl.program_id ((launch d).ctx i) 0) * (IntV.ofNat d.BM : I32)).toInt = ((qrow d i : Nat) : Int) := by
    rw [IntV.I32.add_nat hoy (IntV.I32.mul_nat hpm hBMv (by unfold qrow at hqf; omega)) (by unfold qrow at hqf; omega)]
    rfl
  refine safe_sub fun mi hmi => ?_
  refine safe_add fun li hli => ?_
  refine safe_mul fun sc hsc => ?_
  have hsc' : sc = Tensor.scalar d.scale := by
    ext u
    exact (hsc u).eq (by
      have := hd.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
      exact isNaN_false this)
  subst hsc'
  refine safe_tile_load (b := d.q) hqo (by omega) hql hvq (descOf_valid (by omega) (by omega) (HD16 hd) hq16)
    (fun x hx => Or.inl (Or.inl hx)) fun qt hqt => ?_
  have hrange : tl.RangeOk (0 : I32) (IntV.ofNat d.N : I32) (IntV.ofNat d.BN) := by
    refine ⟨by rw [hBNv]; exact_mod_cast hBN, ?_, ?_, ?_⟩ <;>
      simp only [Format.intRange, ite_true, h0v, hNv, hBNv] <;> omega
  have htrip : tl.tripCount (0 : I32).toInt (IntV.ofNat d.N : I32).toInt (IntV.ofNat d.BN : I32).toInt =
      Err.T d.bnd := by
    rw [h0v, hNv, hBNv]; unfold tl.tripCount Err.T
    obtain ⟨T, hT⟩ := hBNN
    show _ = d.N / d.BN
    rw [hT, Nat.mul_div_cancel_left _ hBN]
    have : ((d.BN * T : Nat) : Int) - 0 + d.BN - 1 = ((d.BN : Int) - 1) + d.BN * T := by push_cast; ring
    rw [this, Int.add_mul_ediv_left _ _ (by positivity), Int.ediv_eq_zero_of_lt (by omega) (by omega),
      zero_add]
    simp
  refine safe_forRange hrange (Inv d init i) ?_ (fun t ht st m hI => ?_) fun st m hI => ?_
  · -- the initial state
    refine ⟨rfl, ?_, ?_, fun r => Or.inl ⟨rfl, ?_, ?_, fun e => rfl⟩⟩
    · rw [IntV.I32.add_nat (y := 0) hoy (by rw [h0v]; rfl) (by omega)]; simp
    · rw [IntV.I32.add_nat (y := 0) hoy (by rw [h0v]; rfl) (by omega)]; simp
    · have h := hmi (r, ())
      rw [show (tl.zeros [d.BM] .f32).get (r, ()) = FloatV.ofPat 0 from rfl,
        show (tl.full [d.BM] posInf32).get (r, ()) = posInf32 from rfl, sub_zero_posInf] at h
      exact h.eq (by decide)
    · have h := hli (r, ())
      rw [show (tl.zeros [d.BM] .f32).get (r, ()) = zero32 from rfl,
        show (tl.full [d.BM] one32).get (r, ()) = one32 from rfl] at h
      refine (add_val zero32_isFinite one32_isFinite h ?_).1
      rw [zero32_toReal, zero_add, toReal_def, show one32.toRat = 1 by decide]; norm_num
  · -- one iteration
    rw [htrip] at ht
    refine safe_step hA hd hpre hp hqt ht ?_ hI
    rw [IntV.I32.toInt_ofInt] <;> rw [h0v, hBNv]
    · ring
    · have : t * d.BN < d.N := by
        have ht' : t < d.N / d.BN := ht
        have := (Nat.le_div_iff_mul_le hBN).mp ht'; rw [Nat.succ_mul] at this; omega
      refine ⟨le_trans (show (-2 ^ 31 : Int) ≤ 0 by norm_num) (by positivity), ?_⟩
      have : t * d.BN < 2 ^ 31 := by omega
      rw [zero_add]; exact_mod_cast this
  · -- the epilogue
    rw [htrip] at hI
    obtain ⟨hm, -, -, hrows⟩ := hI
    subst m
    have hT0 : 0 < Err.T d.bnd := Nat.div_pos (Nat.le_of_dvd (N_pos hd) hBNN) hBN
    have hinv : ∀ r : Fin d.BM, RowInv d.bnd d.scale.toReal (sref d init i r) (vref d init i) (Err.T d.bnd)
        (st.m.get1 r) (st.l.get1 r) (fun e => st.acc.get2 r e) := fun r =>
      ((hrows r).resolve_left fun h => by omega).2
    refine safe_math (fun _ => hA.log2Total _) fun lg hlg => ?_
    refine safe_add fun mo hmo => ?_
    refine safe_div (fun _ => hA.divTotal _ _) fun o ho => ?_
    have hptr : ∀ u : Idx [d.BM], ((d.M.ptr 0 + tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.N) +
        (tl.program_id ((launch d).ctx i) 0 * (IntV.ofNat d.BM : I32) + tl.arange 0 d.BM)).get u =
        d.M.ptr ((qrow d i + u.1.1 : Nat) : Int) := by
      intro u
      have hul := u.1.2
      have hqr : qrow d i = hrow d i + sm d i * d.BM := rfl
      have hu : ((tl.arange 0 d.BM).get u).toInt = ((u.1.1 : Nat) : Int) := by
        show (IntV.ofInt (((0 : Nat) : Int) + u.1.1) : I32).toInt = _
        rw [IntV.I32.toInt_ofInt]
        · push_cast; ring
        · constructor <;> push_cast <;> omega
      have h1 := IntV.I32.mul_nat hpz hNv (show hz d i * d.N < 2 ^ 31 by unfold hrow at hhf; omega)
      have h2 := IntV.I32.add_nat (IntV.I32.mul_nat hpm hBMv (show sm d i * d.BM < 2 ^ 31 by omega)) hu
        (show sm d i * d.BM + u.1.1 < 2 ^ 31 by omega)
      rw [Tensor.get_ptrAdd, Ptr.add_int, ptr_offset2]
      show d.M.ptr ((tl.program_id ((launch d).ctx i) 1 * IntV.ofNat d.N).toInt +
        (tl.program_id ((launch d).ctx i) 0 * IntV.ofNat d.BM + (tl.arange 0 d.BM).get u).toInt) = _
      rw [h1, h2]; congr 1; rw [hqr]; unfold hrow; push_cast; ring
    refine safe_store_buf d.M (g := fun u : Idx [d.BM] => qrow d i + u.1.1) (en := fun _ => true)
      (Buf.addrs_of_get_all d.M hptr) (fun u _ => by rw [hMl]; have := u.1.2; omega)
      (fun u v _ _ h => by
        obtain ⟨⟨a, ha⟩, u2⟩ := u; obtain ⟨⟨b, hb⟩, v2⟩ := v
        simp only at h; have : a = b := by omega
        subst this; rfl) hvM
      (fun x ⟨j, ⟨u, _, hu⟩, hx⟩ => Or.inr ⟨j, ⟨u.1.1, u.1.2, hu.symm⟩, hx⟩) fun m1 hm1 hfr1 hnn1 => ?_
    refine safe_cvt ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩ fun o16 ho16 => ?_
    have hvo1 : d.o.Valid m1 := Buf.valid_of_agree hvo fun x hx => hnn1 x (hvo.2.2.2 x hx)
    refine safe_last (safe_tile_store (b := d.o) hqo (by omega) hol hvo1
      (descOf_valid (by omega) (by omega) (HD16 hd) hob16) (fun x hx => Or.inl hx) fun m2 hm2 hfr2 _ => .ret ?_)
    intro r hr
    have hrow' : qrow d i + r < d.Y := by omega
    have hv : ∀ j < d.bnd.N, ∀ e : Fin d.bnd.HD, |vref d init i j e| ≤ d.bnd.V := fun j hj e =>
      val_le hRv (by rw [hvl]; exact rowcol_lt (show hrow d i + j < d.Y by have : j < d.N := hj; omega) e.2)
    obtain ⟨hmmf, hmme, hoe⟩ := row_out hb hA.log2Acc hA.divFull hv (hinv ⟨r, hr⟩) (hlg (⟨r, hr⟩, ()))
      (hmo (⟨r, hr⟩, ())) (o := fun e => o.get2 ⟨r, hr⟩ e) (fun e => ho (⟨r, hr⟩, e, ()))
      (o16 := fun e => o16.get2 ⟨r, hr⟩ e) (fun e => ho16 (⟨r, hr⟩, e, ()))
    refine ⟨⟨mo.get1 ⟨r, hr⟩, ?_, hmmf, hmme⟩, fun e => ⟨o16.get2 ⟨r, hr⟩ e, hm2 ⟨r, hr⟩ e, (hoe e).1, (hoe e).2⟩⟩
    obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hoM, -⟩ := id hd
    have hspan : Mem.span .f32 (d.M.ptr ((qrow d i + r : Nat) : Int)).addr ⊆ d.M.bytes :=
      d.M.span_sub (by rw [hMl]; exact hrow')
    have htile : d.o.elems (tileSet (qrow d i) d.BM d.HD) ⊆ d.o.bytes :=
      d.o.elems_sub_bytes fun j ⟨a, ha, e, he, hj⟩ => by
        rw [hj, hol]; exact rowcol_lt (by omega) he
    rw [Mem.read_congr fun x hx => hfr2 x fun hx' => Set.disjoint_left.mp hoM (htile hx') (hspan hx)]
    exact hm1 (⟨r, hr⟩, ()) rfl

/-! ### Footprints and the claim -/

section Foot
variable {d : Inst} (hd : d.ok)
include hd

theorem tile_lt {p : Nat} (hp : p < (launch d).size) : ∀ j ∈ tileSet (qrow d p) d.BM d.HD, j < d.o.len := by
  rintro j ⟨a, ha, e, he, rfl⟩
  have := qrow_fit (p := p) hd; have := hrow_fit (d := d) hp
  rw [hd.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1]; exact rowcol_lt (by omega) he

theorem mt_lt {p : Nat} (hp : p < (launch d).size) : ∀ j ∈ MT d p, j < d.M.len := by
  rintro j ⟨r, hr, rfl⟩
  have := qrow_fit (p := p) hd; have := hrow_fit (d := d) hp
  rw [hd.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1]; omega

theorem W_sub {p : Nat} (hp : p < (launch d).size) : W d p ⊆ d.o.bytes ∪ d.M.bytes :=
  Set.union_subset_union (d.o.elems_sub_bytes (tile_lt hd hp)) (d.M.elems_sub_bytes (mt_lt hd hp))

theorem tileSet_disj {p p' : Nat} (hp : p < (launch d).size) (hp' : p' < (launch d).size) (hne : p ≠ p') :
    Disjoint (tileSet (qrow d p) d.BM d.HD) (tileSet (qrow d p') d.BM d.HD) := by
  rw [Set.disjoint_left]
  rintro j ⟨a, ha, e, he, rfl⟩ ⟨a', ha', e', he', h⟩
  rw [qrow_eq hd hp, qrow_eq hd hp'] at h
  have k1 : ((p * d.BM + a) * d.HD + e) % d.HD = e := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt he
  have k2 : ((p' * d.BM + a') * d.HD + e') % d.HD = e' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]; exact Nat.mod_eq_of_lt he'
  have hee : e = e' := by rw [← k1, ← k2, h]
  subst hee
  have hrow : p * d.BM + a = p' * d.BM + a' :=
    Nat.eq_of_mul_eq_mul_right (by omega) (Nat.add_right_cancel h)
  apply hne
  rcases Nat.lt_or_ge p p' with h1 | h1
  · have : (p + 1) * d.BM ≤ p' * d.BM := Nat.mul_le_mul_right _ h1
    rw [Nat.succ_mul] at this; omega
  · rcases Nat.lt_or_ge p' p with h2 | h2
    · have : (p' + 1) * d.BM ≤ p * d.BM := Nat.mul_le_mul_right _ h2
      rw [Nat.succ_mul] at this; omega
    · omega

theorem MT_disj {p p' : Nat} (hp : p < (launch d).size) (hp' : p' < (launch d).size) (hne : p ≠ p') :
    Disjoint (MT d p) (MT d p') := by
  rw [Set.disjoint_left]
  rintro j ⟨r, hr, rfl⟩ ⟨r', hr', h⟩
  rw [qrow_eq hd hp, qrow_eq hd hp'] at h
  apply hne
  rcases Nat.lt_or_ge p p' with h1 | h1
  · have : (p + 1) * d.BM ≤ p' * d.BM := Nat.mul_le_mul_right _ h1
    rw [Nat.succ_mul] at this; omega
  · rcases Nat.lt_or_ge p' p with h2 | h2
    · have : (p' + 1) * d.BM ≤ p * d.BM := Nat.mul_le_mul_right _ h2
      rw [Nat.succ_mul] at this; omega
    · omega

end Foot

/-- Query `i` of head `(z, h)` is row `i % BLOCK_M` of program `(z·H + h)·(N/BLOCK_M) + i / BLOCK_M`. -/
theorem prog_of {d : Inst} (hd : d.ok) {z h i : Nat} (hz0 : z < d.Z) (hh0 : h < d.H) (hi0 : i < d.N) :
    let p := (z * d.H + h) * (d.N / d.BM) + i / d.BM
    p < (launch d).size ∧ qrow d p + i % d.BM = d.row z h i ∧ hrow d p = (z * d.H + h) * d.N := by
  intro p
  have hg := g0_pos hd
  have hBM := BM_pos hd
  have hiB : i / d.BM < d.N / d.BM := Nat.div_lt_div_of_lt_of_dvd (BMN hd) hi0
  have hzh : z * d.H + h < d.Z * d.H := by
    calc z * d.H + h < z * d.H + d.H := by omega
      _ = (z + 1) * d.H := by ring
      _ ≤ d.Z * d.H := Nat.mul_le_mul_right _ hz0
  have hp : p < (launch d).size := by
    rw [size_eq]
    calc p < (z * d.H + h) * (d.N / d.BM) + d.N / d.BM := by omega
      _ = (z * d.H + h + 1) * (d.N / d.BM) := by ring
      _ ≤ d.Z * d.H * (d.N / d.BM) := Nat.mul_le_mul_right _ hzh
      _ = d.N / d.BM * (d.Z * d.H) := by ring
  have hdiv : p / (d.N / d.BM) = z * d.H + h := by
    show ((z * d.H + h) * (d.N / d.BM) + i / d.BM) / (d.N / d.BM) = _
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hg, Nat.div_eq_of_lt hiB, zero_add]
  have hmod : p % (d.N / d.BM) = i / d.BM := by
    show ((z * d.H + h) * (d.N / d.BM) + i / d.BM) % (d.N / d.BM) = _
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hiB]
  have hhz : hz d p = z * d.H + h := by rw [hz_eq hp, hdiv]
  refine ⟨hp, ?_, by unfold hrow; rw [hhz]⟩
  unfold qrow hrow sm Inst.row
  rw [hhz, hmod]
  have := Nat.div_add_mod i d.BM
  calc (z * d.H + h) * d.N + i / d.BM * d.BM + i % d.BM = (z * d.H + h) * d.N + (d.BM * (i / d.BM) + i % d.BM) := by ring
    _ = (z * d.H + h) * d.N + i := by rw [this]

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent (model := model) (launch := launch) hA.faithful (fun _ _ => rfl)
    (fun d hd => launch_valid hd) (fun d _ _ => R d) (fun d _ i => W d i) (fun d init i m => Q d init i m)
    (fun d hd init hpre i => safe_program hA hd hpre i) ?_ ?_
  · -- footprints: a program's writes miss the inputs and the other programs' writes
    intro d hd init _ i j hij
    have hne : (i : Nat) ≠ j := Fin.val_ne_of_ne hij
    obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hoq, hok, hov, hoM, hMq, hMk, hMv, -⟩ := id hd
    rw [Set.disjoint_union_right]
    constructor
    · refine Disjoint.mono_left (W_sub hd i.2) ?_
      unfold R
      refine Set.disjoint_union_left.mpr ⟨?_, ?_⟩ <;>
        refine Set.disjoint_union_right.mpr ⟨Set.disjoint_union_right.mpr ⟨?_, ?_⟩, ?_⟩
      exacts [hoq, hok, hov, hMq, hMk, hMv]
    · unfold W
      refine Set.disjoint_union_left.mpr ⟨Set.disjoint_union_right.mpr ⟨?_, ?_⟩,
        Set.disjoint_union_right.mpr ⟨?_, ?_⟩⟩
      · exact d.o.elems_disjoint (tileSet_disj hd i.2 j.2 hne)
      · exact d.o.elems_disjoint_of d.M hoM (tile_lt hd i.2) (mt_lt hd j.2)
      · exact d.M.elems_disjoint_of d.o hoM.symm (mt_lt hd i.2) (tile_lt hd j.2)
      · exact d.M.elems_disjoint (MT_disj hd i.2 j.2 hne)
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun z hz0 h hh0 i hi0 => ?_, fun x hxo hxM => hfr x fun p hx => ?_⟩
    · obtain ⟨hp, hq, hhr⟩ := prog_of hd hz0 hh0 hi0
      obtain ⟨m, hQ, hag⟩ := hloc ⟨_, hp⟩
      have hr : i % d.BM < d.BM := Nat.mod_lt _ (BM_pos hd)
      obtain ⟨⟨mm, hmm, hmmf, hmme⟩, hO⟩ := hQ (i % d.BM) hr
      have hsref : ∀ j, sref d init ((z * d.H + h) * (d.N / d.BM) + i / d.BM) (i % d.BM) j =
          score d init (d.row z h i) (d.row z h j) := fun j => by
        unfold sref; rw [hq, hhr]; rfl
      have hvref : ∀ j (e : Fin d.HD), vref d init ((z * d.H + h) * (d.N / d.BM) + i / d.BM) j e =
          val init d.v (d.row z h j * d.HD + e) := fun j e => by
        unfold vref; rw [hhr]; rfl
      simp only [hsref, hvref] at hmme hO
      rw [hq] at hmm hO
      refine ⟨⟨mm, ?_, hmmf, hmme⟩, fun e he => ?_⟩
      · rw [Mem.read_congr fun x hx => hag x (Or.inr (Or.inr (d.M.span_sub_elems
          (show qrow d _ + i % d.BM ∈ MT d _ from ⟨i % d.BM, hr, rfl⟩) (by rw [hq]; exact hx))))]
        exact hmm
      · obtain ⟨x, hx, hxf, hxe⟩ := hO ⟨e, he⟩
        refine ⟨x, ?_, hxf, hxe⟩
        rw [Mem.read_congr fun y hy => hag y (Or.inr (Or.inl (d.o.span_sub_elems
          (show (qrow d _ + i % d.BM) * d.HD + e ∈ tileSet (qrow d _) d.BM d.HD from
            ⟨i % d.BM, hr, e, he, rfl⟩) (by rw [hq]; exact hy))))]
        exact hx
    · rcases W_sub hd p.2 hx with h | h
      · exact hxo h
      · exact hxM h

/-! ### Consistency -/

/-- The witness platform's accuracies (`Platform.ref`): `2^-22` relative for every function,
`2^-149` absolute for `exp2` and `2^-22` for the others. -/
noncomputable def refRel : MathFn → Format → ℝ := fun _ _ => 2 ^ (-22 : ℤ)
noncomputable def refAbs : MathFn → Format → ℝ
  | .exp2, _ => Err.absE
  | _, _ => Err.absL
/-- The witness platform's special value: `exp2(-∞) = +0`. -/
def refSp : (fn : MathFn) → (f : Format) → FloatV f → Option (FloatV f)
  | .exp2, _, x => if x.pat = negInf32.pat then some (FloatV.ofPat 0) else none
  | _, _, _ => none

/-- **The assumptions are consistent**: the claim is not vacuous. -/
theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful (Platform.ref refRel refAbs refSp) model launch
  have hu : ((Format.binary32.u : ℚ) : ℝ) ≤ 2 ^ (-22 : ℤ) := by rw [u32_eq]; unfold Err.u; norm_num
  have heE : ((Format.binary32.eta : ℚ) : ℝ) ≤ Err.absE := by
    rw [eta32_eq]; unfold Err.η Err.absE; norm_num
  have heL : ((Format.binary32.eta : ℚ) : ℝ) ≤ Err.absL := by
    rw [eta32_eq]; unfold Err.η Err.absL; norm_num
  have hsp : refSp .exp2 .binary32 negInf32 = some zero32 := by
    show (if negInf32.pat = negInf32.pat then some (FloatV.ofPat 0) else none) = some zero32
    simp only [↓reduceIte]; rfl
  refine ⟨K, Platform.ref refRel refAbs refSp, htot, ⟨fun d _ => hf d, Platform.ref.dotTotal cfg,
    Platform.ref.dotAcc (g := Format.binary32) rfl _ (fun K _ => ?_) (fun K _ => ?_)
      (Err.relDot_nonneg 0) (Err.absDot_nonneg 0),
    Platform.ref.mathTotal,
    Platform.ref.mathAcc_of_eta rfl Format.binary32_regular hu heE Set.univ,
    Platform.ref.mathAt (by decide) hsp,
    Platform.ref.mathTotal,
    Platform.ref.mathAcc_of_eta rfl Format.binary32_regular hu heL Set.univ,
    Platform.ref.divTotal _, Platform.ref.divFull⟩⟩
  · rw [show Format.binary32.u = 2 ^ (-24 : ℤ) by unfold Format.u; rfl]; unfold Err.relDot
    have : (2 : ℚ) ≤ K + 2 := by linarith [(Nat.cast_nonneg K : (0 : ℚ) ≤ K)]
    have e : (2 : ℚ) ^ (-24 : ℤ) = 2 ^ (-22 : ℤ) / 4 := by norm_num
    rw [e]; nlinarith [show (0 : ℚ) < 2 ^ (-22 : ℤ) by positivity]
  · rw [show Format.binary32.eta = 2 ^ (-150 : ℤ) by
      unfold Format.eta; rw [show Format.binary32.emin - (Format.binary32.m : ℤ) - 1 = -150 by decide]]
    unfold Err.absDot
    have : (2 : ℚ) ≤ K + 2 := by linarith [(Nat.cast_nonneg K : (0 : ℚ) ≤ K)]
    have e : (2 : ℚ) ^ (-150 : ℤ) = 2 ^ (-126 : ℤ) / 2 ^ 24 := by norm_num
    rw [e]; nlinarith [show (0 : ℚ) < 2 ^ (-126 : ℤ) by positivity]


end TVAExamples.FlashAttn
