import BlockScaledMxfp8.Defs

/-!
# `block_scaled_matmul_mxfp8`: facts the per-program proof uses

* `int32` arithmetic on small values (`i32_nat`, `i32_zero`, `i32_one`, `i32_trip`);
* the lanes of the descriptors' blocks: rank 2 (`lane2`) and rank 5 (`lane5`), and their accesses
  (`access2`, `access5`);
* the scale blocks after the reshapes and the permutation of lines 226–227 (`scaleA_get`,
  `scaleB_get`), and where their lanes lie in the scale tensors (`sIdx`);
* what `tl.dot_scaled` reads of its operands (`unpackL_get`, `scalesL_get`, `unpackR_get`,
  `scalesR_get`, `scaledValue_eq`);
* the tolerance (`tol_nonneg`, `tol_mono`, `partSum_succ`, `partAbs`);
* one step of the accumulation (`acc_step`), from `Rules.Num.DotScaledE4m3`;
* the last rounding to `float16` (`out_f16`).
-/

namespace Kernels.BlockScaledMxfp8

open TVA

/-! ## `int32` arithmetic -/

theorem i32_nat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n := IntV.I32.toInt_ofNat h

theorem i32_int {n : Int} (h0 : -2 ^ 31 ≤ n) (h1 : n < 2 ^ 31) : (IntV.ofInt n : I32).toInt = n :=
  IntV.I32.toInt_ofInt ⟨h0, h1⟩

theorem i32_zero : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)

theorem i32_one : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)

theorem i32_zero' : (IntV.ofInt 0 : I32).toInt = ((0 : Nat) : Int) := i32_zero

/-- The induction variable of a loop from the `int32` `0` with step `1`, at trip `t`. -/
theorem i32_trip {t : Nat} (ht : t < 2 ^ 31) :
    (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt = t := by
  rw [i32_zero, i32_one, zero_add, mul_one]; exact i32_int (by omega) (by omega)

/-! ## Descriptor lanes -/

/-- A lane of a rank-2 block at the offsets `O0`, `O1` (the values of `o0`, `o1`) of a row-major
`[s0, s1]` descriptor with row stride `l`: inside the tensor when both coordinates are, and then
element `(O0 + i)·l + (O1 + j)`. -/
theorem lane2 {d : DType} (base : Ptr d) (s0 s1 l : Nat) (bl : Shape) (pad : Padding) {m n : Nat}
    {o0 o1 : I32} {O0 O1 : Nat} (h0 : o0.toInt = O0) (h1 : o1.toInt = O1) (i : Fin m) (j : Fin n) :
    (⟨base, [s0, s1], [l, 1], bl, pad⟩ : TensorDesc d).inbAt
        (TensorDesc.coord ([o0, o1].map IntV.toInt) (bs := [m, n]) (i, j, ())) =
      decide (O0 + i < s0 ∧ O1 + j < s1) ∧
    ((⟨base, [s0, s1], [l, 1], bl, pad⟩ : TensorDesc d).elemAt
        (TensorDesc.coord ([o0, o1].map IntV.toInt) (bs := [m, n]) (i, j, ()))).toNat =
      (O0 + i) * l + (O1 + j) := by
  constructor
  · unfold TensorDesc.inbAt TensorDesc.rank
    refine decide_eq_decide.mpr ⟨fun h => ?_, fun h k hk => ?_⟩
    · have e0 := h 0 (by simp)
      have e1 := h 1 (by simp)
      simp only [TensorDesc.coord, List.map_cons, List.map_nil, h0, h1, Idx.coord] at e0 e1
      simp at e0 e1
      omega
    · simp only [List.length_cons, List.length_nil] at hk
      rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl <;>
        simp [TensorDesc.coord, h0, h1, Idx.coord] <;> omega
  · unfold TensorDesc.elemAt TensorDesc.rank
    simp [List.range_succ, TensorDesc.coord, h0, h1, Idx.coord]
    rw [show ((O0 : Int) + (i : Nat)) * (l : Int) + ((O1 : Int) + (j : Nat)) =
      (((O0 + i) * l + (O1 + j) : Nat) : Int) by push_cast; ring, Int.toNat_natCast]

/-- A lane of a rank-5 block at the offsets `O0 … O4` of a `[s0, …, s4]` descriptor with strides
`[t0, …, t4]`: inside the tensor when every coordinate is, and then element
`Σ (Oₖ + lₖ)·tₖ`. -/
theorem lane5 {d : DType} (base : Ptr d) (s0 s1 s2 s3 s4 t0 t1 t2 t3 t4 : Nat) (bl : Shape)
    (pad : Padding) {b0 b1 b2 b3 b4 : Nat} {o0 o1 o2 o3 o4 : I32} {O0 O1 O2 O3 O4 : Nat}
    (h0 : o0.toInt = O0) (h1 : o1.toInt = O1) (h2 : o2.toInt = O2) (h3 : o3.toInt = O3)
    (h4 : o4.toInt = O4) (l0 : Fin b0) (l1 : Fin b1) (l2 : Fin b2) (l3 : Fin b3) (l4 : Fin b4) :
    (⟨base, [s0, s1, s2, s3, s4], [t0, t1, t2, t3, t4], bl, pad⟩ : TensorDesc d).inbAt
        (TensorDesc.coord ([o0, o1, o2, o3, o4].map IntV.toInt) (bs := [b0, b1, b2, b3, b4])
          (l0, l1, l2, l3, l4, ())) =
      decide (O0 + l0 < s0 ∧ O1 + l1 < s1 ∧ O2 + l2 < s2 ∧ O3 + l3 < s3 ∧ O4 + l4 < s4) ∧
    ((⟨base, [s0, s1, s2, s3, s4], [t0, t1, t2, t3, t4], bl, pad⟩ : TensorDesc d).elemAt
        (TensorDesc.coord ([o0, o1, o2, o3, o4].map IntV.toInt) (bs := [b0, b1, b2, b3, b4])
          (l0, l1, l2, l3, l4, ()))).toNat =
      (O0 + l0) * t0 + (O1 + l1) * t1 + (O2 + l2) * t2 + (O3 + l3) * t3 + (O4 + l4) * t4 := by
  constructor
  · unfold TensorDesc.inbAt TensorDesc.rank
    refine decide_eq_decide.mpr ⟨fun h => ?_, fun h k hk => ?_⟩
    · have e0 := h 0 (by simp)
      have e1 := h 1 (by simp)
      have e2 := h 2 (by simp)
      have e3 := h 3 (by simp)
      have e4 := h 4 (by simp)
      simp only [TensorDesc.coord, List.map_cons, List.map_nil, h0, h1, h2, h3, h4, Idx.coord] at e0 e1 e2 e3 e4
      simp at e0 e1 e2 e3 e4
      omega
    · simp only [List.length_cons, List.length_nil] at hk
      rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 by omega) with rfl | rfl | rfl | rfl | rfl <;>
        simp [TensorDesc.coord, h0, h1, h2, h3, h4, Idx.coord] <;> omega
  · unfold TensorDesc.elemAt TensorDesc.rank
    simp [List.range_succ, TensorDesc.coord, h0, h1, h2, h3, h4, Idx.coord]
    exact (congrArg Int.toNat (by push_cast; ring)).trans (Int.toNat_natCast _)

/-- A block access through a row-major 2-D descriptor meets Triton's requirements. -/
theorem access2 {d : DType} {base : Ptr d} {s0 s1 l : Nat} {bl : Shape} {o0 o1 : I32}
    (hs0 : 0 < s0 ∧ s0 < 2 ^ 31) (hs1 : 0 < s1 ∧ s1 < 2 ^ 31) (hb : base.addr % 16 = 0)
    (hl : (l * d.bytes) % 16 = 0) (hbl : bl.length = 2) (ho : (o1.toInt * d.bytes) % 16 = 0) :
    (⟨base, [s0, s1], [l, 1], bl, .zero⟩ : TensorDesc d).Access ([o0, o1].map IntV.toInt) bl := by
  refine ⟨⟨by simp [TensorDesc.rank], by simp [TensorDesc.rank], by simp [TensorDesc.rank],
    by simp [TensorDesc.rank, hbl], fun k hk => ?_, hb, rfl, fun k hk => ?_, by simp, by simp⟩,
    rfl, by simp [TensorDesc.rank, hbl], by simp [TensorDesc.rank], by simpa using ho⟩
  · simp only [TensorDesc.rank, List.length_cons, List.length_nil] at hk
    rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl
    · simpa using hs0
    · simpa using hs1
  · simp only [TensorDesc.rank, List.length_cons, List.length_nil] at hk
    rw [show k = 0 by omega]
    simpa using hl

/-- A block access through a 5-D `uint8` descriptor with last strides `256, 1` and innermost
offset `0` meets Triton's requirements. -/
theorem access5 {base : Ptr .u8} {s0 s1 s2 s3 s4 t0 t1 : Nat} {bl : Shape} {o0 o1 o2 o3 : I32}
    (hs : ∀ s ∈ [s0, s1, s2, s3, s4], 0 < s ∧ s < 2 ^ 31) (hb : base.addr % 16 = 0)
    (ht0 : t0 % 16 = 0) (ht1 : t1 % 16 = 0) (hbl : bl.length = 5) :
    (⟨base, [s0, s1, s2, s3, s4], [t0, t1, 512, 256, 1], bl, .zero⟩ : TensorDesc .u8).Access
      ([o0, o1, o2, o3, (IntV.ofInt 0 : I32)].map IntV.toInt) bl := by
  refine ⟨⟨by simp [TensorDesc.rank], by simp [TensorDesc.rank], by simp [TensorDesc.rank],
    by simp [TensorDesc.rank, hbl], fun k hk => ?_, hb, rfl, fun k hk => ?_, by simp, by simp⟩,
    rfl, by simp [TensorDesc.rank, hbl], by simp [TensorDesc.rank], by simp [i32_zero]⟩
  · simp only [TensorDesc.rank, List.length_cons, List.length_nil] at hk
    rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 by omega) with rfl | rfl | rfl | rfl | rfl <;>
      simpa using hs _ (by simp)
  · simp only [TensorDesc.rank, List.length_cons, List.length_nil] at hk
    rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 by omega) with rfl | rfl | rfl | rfl <;>
      simp [DType.bytes, ht0, ht1]

/-! ## The scale blocks -/

theorem ofPos_eq_of_pos {s : Shape} {p : Nat} (h : p < s.prod) (j : Idx s) (hj : Idx.pos s j = p) :
    Idx.ofPos s p h = j := by
  subst hj; exact Idx.ofPos_pos s j h

/-- Element `i` of a reshape is the element of the same row-major position. -/
theorem get_reshape_of_pos {s s' : Shape} {α : Type} (x : Tensor s α) (h : s'.prod = s.prod)
    (i : Idx s') (j : Idx s) (hj : Idx.pos s j = Idx.pos s' i) : (x.reshape s' h).get i = x.get j := by
  rw [Tensor.get_reshape, ofPos_eq_of_pos _ j hj]

/-- **`a`'s scales** (line 226, `rep_m = rep_k = 1`): element `(i, c)` of the `[128, 4]` scale
block is lane `(0, 0, 0, q / 256, q % 256)` of the loaded `[1, 1, 1, 2, 256]` block, with
`q = 16·(i % 32) + 4·(i / 32) + c`. -/
theorem scaleA_get (v : Tensor [1, 1, 1, 2, 256] U8) (h1 : [1, 1, 32, 4, 4].prod = [1, 1, 1, 2, 256].prod)
    (h2 : [128, 4].prod = [1, 4, 32, 1, 4].prod) (i : Fin 128) (c : Fin 4) :
    (((v.reshape [1, 1, 32, 4, 4] h1).permute [0, 3, 2, 1, 4]).reshape [128, 4] h2).get2 i c =
      v.get (0, 0, 0, ⟨(16 * (i % 32) + 4 * (i / 32) + c) / 256, by omega⟩,
        ⟨(16 * (i % 32) + 4 * (i / 32) + c) % 256, by omega⟩, ()) := by
  refine (get_reshape_of_pos _ h2 (i, c, ())
    ((0, ⟨i / 32, by omega⟩, ⟨i % 32, by omega⟩, 0, c, ()) : Idx [1, 4, 32, 1, 4])
    (by simp [Idx.pos]; omega)).trans ?_
  refine (Tensor.get_permute (s := [1, 1, 32, 4, 4]) _ _ _ _
    ((0, 0, ⟨i % 32, by omega⟩, ⟨i / 32, by omega⟩, c, ()) : Idx [1, 1, 32, 4, 4])
    (by
      intro k hk
      simp only [List.length_cons, List.length_nil] at hk
      rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 by omega) with rfl | rfl | rfl | rfl | rfl <;>
        rfl)).trans ?_
  exact get_reshape_of_pos _ h1 _ _ (by simp [Idx.pos]; omega)

/-- **`b`'s scales** (line 227, `rep_n = 2`, `rep_k = 1`): element `(j, c)` of the `[256, 4]`
scale block is lane `(0, j / 128, 0, q / 256, q % 256)` of the loaded `[1, 2, 1, 2, 256]` block,
with `q = 16·(j % 32) + 4·((j % 128) / 32) + c`. -/
theorem scaleB_get (v : Tensor [1, 2, 1, 2, 256] U8) (h1 : [2, 1, 32, 4, 4].prod = [1, 2, 1, 2, 256].prod)
    (h2 : [256, 4].prod = [2, 4, 32, 1, 4].prod) (j : Fin 256) (c : Fin 4) :
    (((v.reshape [2, 1, 32, 4, 4] h1).permute [0, 3, 2, 1, 4]).reshape [256, 4] h2).get2 j c =
      v.get (0, ⟨j / 128, by omega⟩, 0, ⟨(16 * (j % 32) + 4 * (j % 128 / 32) + c) / 256, by omega⟩,
        ⟨(16 * (j % 32) + 4 * (j % 128 / 32) + c) % 256, by omega⟩, ()) := by
  refine (get_reshape_of_pos _ h2 (j, c, ())
    ((⟨j / 128, by omega⟩, ⟨j % 128 / 32, by omega⟩, ⟨j % 32, by omega⟩, 0, c, ()) : Idx [2, 4, 32, 1, 4])
    (by simp [Idx.pos]; omega)).trans ?_
  refine (Tensor.get_permute (s := [2, 1, 32, 4, 4]) _ _ _ _
    ((⟨j / 128, by omega⟩, 0, ⟨j % 32, by omega⟩, ⟨j % 128 / 32, by omega⟩, c, ()) : Idx [2, 1, 32, 4, 4])
    (by
      intro k hk
      simp only [List.length_cons, List.length_nil] at hk
      rcases (show k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 by omega) with rfl | rfl | rfl | rfl | rfl <;>
        rfl)).trans ?_
  exact get_reshape_of_pos _ h1 _ _ (by simp [Idx.pos]; omega)

/-! ## What `tl.dot_scaled` reads -/

theorem unpackL_get (A : Tensor [128, 128] F8E4) (i k : Fin 128) :
    (cfgMx.unpackL 128 128 (d := .f8e4nv) A).get2 i k = DType.toBits .f8e4nv (A.get2 i k) := by
  simp [ScaledCfg.unpackL, ScaledCfg.lhsBits, cfgMx, Tensor.getD2]

theorem scalesL_get (sA : Tensor [128, 4] U8) (i k : Fin 128) :
    (cfgMx.scalesL 128 128 (ds := .u8) sA).get2 i k = DType.toBits .u8 (sA.get2 i ⟨k / 32, by omega⟩) := by
  simp only [ScaledCfg.scalesL, Tensor.get2_of2, Tensor.getD2]
  split_ifs with h
  · rfl
  · exact absurd ⟨i.2, show (k : Nat) / 32 < 4 by omega⟩ h

theorem unpackR_get (Bm : Tensor [256, 128] F8E4) (k : Fin 128) (j : Fin 256) :
    (cfgMx.unpackR 128 256 (d := .f8e4nv) Bm.trans).get2 k j = DType.toBits .f8e4nv (Bm.get2 j k) := by
  simp [ScaledCfg.unpackR, ScaledCfg.rhsBits, cfgMx, Tensor.getD2]

theorem scalesR_get (sB : Tensor [256, 4] U8) (k : Fin 128) (j : Fin 256) :
    (cfgMx.scalesR 128 256 (ds := .u8) sB).get2 k j = DType.toBits .u8 (sB.get2 j ⟨k / 32, by omega⟩) := by
  simp only [ScaledCfg.scalesR, Tensor.get2_of2, Tensor.getD2]
  split_ifs with h
  · rfl
  · exact absurd ⟨j.2, show (k : Nat) / 32 < 4 by omega⟩ h

/-- The exact value `tl.dot_scaled` gives an `e4m3` element with an `e8m0` scale: `scaled`. -/
theorem scaledValue_eq {x : F8E4} {s : U8} (hx : x.isFinite = true) (hs : s.bits.toNat ≠ 255) :
    scaledValue .e4m3 .e8m0 (DType.toBits .f8e4nv x) (some (DType.toBits .u8 s)) = some (scaled x s) := by
  have h1 : MXFmt.value .e4m3 (DType.toBits .f8e4nv x) = some x.toRat := by
    show (if Math.Float.Format.e4m3.isFinite x.pat = true then some (Math.Float.Format.e4m3.toRat x.pat)
      else none) = _
    rw [ite_eq_left (show Math.Float.Format.e4m3.isFinite x.pat = true from hx)]
    rfl
  have h2 : ScaleFmt.value .e8m0 (DType.toBits .u8 s) = some (scaleV s) := by
    show e8m0Value s.bits.toNat = _
    unfold e8m0Value scaleV
    rw [ite_eq_right hs]
  unfold scaledValue
  rw [h1]
  show Option.map (fun y => x.toRat * y) (ScaleFmt.value .e8m0 (DType.toBits .u8 s)) = _
  rw [h2]
  rfl

/-! ## The tolerance -/

/-- The magnitudes of the products before step `t` (`k < 128 t`). -/
def partAbs (p : Nat → ℚ) (t : Nat) : ℚ := ∑ e ∈ Finset.range (128 * t), |p e|

theorem rho_eq : rho = 130 / 2 ^ 22 := by unfold rho Rules.Num.relDot16; norm_num

theorem alpha_eq : alpha = 130 / 2 ^ 126 := by unfold alpha Rules.Num.absDot16; norm_num

theorem rho_nonneg : (0 : ℚ) ≤ rho := by rw [rho_eq]; norm_num

theorem alpha_nonneg : (0 : ℚ) ≤ alpha := by rw [alpha_eq]; norm_num

theorem tol_nonneg (p : Nat → ℚ) : ∀ t, 0 ≤ tol p t
  | 0 => le_rfl
  | t + 1 => by
    have h := tol_nonneg p t
    have h1 : 0 ≤ chunkAbs p t := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have h2 := rho_nonneg
    have h3 := alpha_nonneg
    simp only [tol]
    have : 0 ≤ rho * (|partSum p t| + tol p t + chunkAbs p t) :=
      mul_nonneg h2 (by have := abs_nonneg (partSum p t); linarith)
    linarith

theorem tol_le_succ (p : Nat → ℚ) (t : Nat) : tol p t ≤ tol p (t + 1) := by
  have h := tol_nonneg p t
  have h1 : 0 ≤ chunkAbs p t := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have : 0 ≤ rho * (|partSum p t| + tol p t + chunkAbs p t) :=
    mul_nonneg rho_nonneg (by have := abs_nonneg (partSum p t); linarith)
  simp only [tol]
  linarith [alpha_nonneg]

theorem tol_mono (p : Nat → ℚ) {t T : Nat} (h : t ≤ T) : tol p t ≤ tol p T :=
  monotone_nat_of_le_succ (tol_le_succ p) h

theorem partSum_succ (p : Nat → ℚ) (t : Nat) :
    partSum p (t + 1) = partSum p t + ∑ e ∈ Finset.range 128, p (128 * t + e) := by
  unfold partSum
  rw [show 128 * (t + 1) = 128 * t + 128 by ring, Finset.sum_range_add]

theorem partAbs_succ (p : Nat → ℚ) (t : Nat) : partAbs p (t + 1) = partAbs p t + chunkAbs p t := by
  unfold partAbs chunkAbs
  rw [show 128 * (t + 1) = 128 * t + 128 by ring, Finset.sum_range_add]

theorem abs_partSum_le (p : Nat → ℚ) (t : Nat) : |partSum p t| ≤ partAbs p t :=
  Finset.abs_sum_le_sum_abs _ _

theorem partAbs_mono (p : Nat → ℚ) {t T : Nat} (h : t ≤ T) : partAbs p t ≤ partAbs p T :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.mpr (by omega))
    fun _ _ _ => abs_nonneg _

/-- **The tolerance in closed form** (not used by the proof; README, Specification): after `t`
steps, `tol p t ≤ ((1 + rho)^t − 1) · Σ_{k < 128 t} |p k| + t · (1 + rho)^t · alpha`. -/
theorem tol_le_closed (p : Nat → ℚ) :
    ∀ t, tol p t ≤ ((1 + rho) ^ t - 1) * partAbs p t + t * (1 + rho) ^ t * alpha
  | 0 => by simp [tol, partAbs]
  | t + 1 => by
    have ih := tol_le_closed p t
    have hS := abs_partSum_le p t
    have hr := rho_nonneg
    have ha := alpha_nonneg
    have hc : 0 ≤ chunkAbs p t := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have hX : 1 ≤ (1 + rho) ^ t := one_le_pow₀ (by linarith)
    have h1 : (1 + rho) * tol p t ≤
        (1 + rho) * (((1 + rho) ^ t - 1) * partAbs p t + t * (1 + rho) ^ t * alpha) :=
      mul_le_mul_of_nonneg_left ih (by linarith)
    have h2 : 0 ≤ ((1 + rho) ^ t - 1) * (1 + rho) * chunkAbs p t :=
      mul_nonneg (mul_nonneg (by linarith) (by linarith)) hc
    have h3 : 0 ≤ ((1 + rho) ^ t * (1 + rho) - 1) * alpha := mul_nonneg (by nlinarith) ha
    have h4 : rho * |partSum p t| ≤ rho * partAbs p t := mul_le_mul_of_nonneg_left hS hr
    show tol p t + rho * (|partSum p t| + tol p t + chunkAbs p t) + alpha ≤ _
    rw [partAbs_succ, pow_succ]
    push_cast
    linarith

/-- At the largest covered `K = 2^14` (`128` steps), `(1 + rho)^128 − 1 ≤ 1/250` (`0.4%`). -/
theorem closed_rel_le : (1 + rho) ^ 128 - 1 ≤ 1 / 250 := by
  rw [rho_eq]; norm_num

/-- …and the absolute part `128 · (1 + rho)^128 · alpha` is below `2^-110`. -/
theorem closed_abs_le : (128 : ℚ) * (1 + rho) ^ 128 * alpha ≤ 1 / 2 ^ 110 := by
  rw [rho_eq, alpha_eq]; norm_num

/-! ## One step of the accumulation -/

/-- **One step of the loop** (line 234), for one output element `(i, j)` of the tile at step
`t < T`: the accumulator stays finite and within `tol` of the exact partial sum, by
`Rules.Num.DotScaledE4m3` (each step a `tl.dot_scaled` of `128` products into the accumulator,
within `rho·(|acc| + chunkAbs) + alpha`). `A`, `Bm` are the loaded blocks of `a` and `b`, `sA`,
`sB` the scale blocks after line 226–227, whose entries for `(i, j)` are the inputs' elements
and scales of step `t`. -/
theorem acc_step {P : Platform} (hdot : Rules.Num.DotScaledE4m3 P)
    {A : Tensor [128, 128] F8E4} {Bm : Tensor [256, 128] F8E4} {sA : Tensor [128, 4] U8}
    {sB : Tensor [256, 4] U8} {acc r : Tensor [128, 256] F32}
    (hr : P.dotScaled cfgMx (cfgMx.unpackL 128 128 (d := .f8e4nv) A) ((some sA).map (cfgMx.scalesL 128 128 (ds := .u8)))
      (cfgMx.unpackR 128 256 (d := .f8e4nv) Bm.trans) ((some sB).map (cfgMx.scalesR 128 256 (ds := .u8))) acc r)
    (i : Fin 128) (j : Fin 256) {xa xb : Nat → F8E4} {sa sb : Nat → U8} {T t : Nat} (ht : t < T)
    (hA : ∀ k : Fin 128, A.get2 i k = xa (128 * t + k))
    (hsA : ∀ k : Fin 128, sA.get2 i ⟨k / 32, by omega⟩ = sa (128 * t + k))
    (hB : ∀ k : Fin 128, Bm.get2 j k = xb (128 * t + k))
    (hsB : ∀ k : Fin 128, sB.get2 j ⟨k / 32, by omega⟩ = sb (128 * t + k))
    (hdom : ∀ k, k < 128 * T → InDom (xa k) (sa k) ∧ InDom (xb k) (sb k))
    (hroom : dotAbs (prodV xa xb sa sb) (128 * T) + tol (prodV xa xb sa sb) T ≤ 2 ^ 127)
    (hinv : (acc.get2 i j).isFinite = true ∧
      |(acc.get2 i j).toRat - partSum (prodV xa xb sa sb) t| ≤ tol (prodV xa xb sa sb) t) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toRat - partSum (prodV xa xb sa sb) (t + 1)| ≤ tol (prodV xa xb sa sb) (t + 1) := by
  set p := prodV xa xb sa sb with hp
  have hk : ∀ k : Fin 128, 128 * t + k < 128 * T := fun k => by
    have := k.2; have : t + 1 ≤ T := ht; nlinarith
  have hsum : ∑ k : Fin 128, scaled (xa (128 * t + k)) (sa (128 * t + k)) * scaled (xb (128 * t + k)) (sb (128 * t + k)) =
      ∑ e ∈ Finset.range 128, p (128 * t + e) :=
    Fin.sum_univ_eq_sum_range (fun e => p (128 * t + e)) 128
  have habs : ∑ k : Fin 128, |scaled (xa (128 * t + k)) (sa (128 * t + k)) * scaled (xb (128 * t + k)) (sb (128 * t + k))| =
      chunkAbs p t :=
    Fin.sum_univ_eq_sum_range (fun e => |p (128 * t + e)|) 128
  -- sizes
  have hT0 := tol_nonneg p t
  have hTm := tol_mono p ht.le
  have hpa := partAbs_mono p (show t + 1 ≤ T from ht)
  rw [partAbs_succ] at hpa
  have hSa := abs_partSum_le p t
  have hacc : |(acc.get2 i j).toRat| ≤ |partSum p t| + tol p t := by
    have := abs_sub_abs_le_abs_sub (acc.get2 i j).toRat (partSum p t)
    linarith [hinv.2]
  have hroom' : |(acc.get2 i j).toRat| + chunkAbs p t ≤ 2 ^ 127 := by
    have : partAbs p T = dotAbs p (128 * T) := rfl
    linarith
  have hc0 : 0 ≤ chunkAbs p t := Finset.sum_nonneg fun _ _ => abs_nonneg _
  -- the fact
  obtain ⟨hrfin, herr⟩ := hdot false _ _ _ _ _ _ hr i j (fun k => scaled (xa (128 * t + k)) (sa (128 * t + k)))
    (fun k => scaled (xb (128 * t + k)) (sb (128 * t + k)))
    (fun k => by
      have hd := (hdom _ (hk k)).1
      show scaledValue .e4m3 .e8m0 ((cfgMx.unpackL 128 128 (d := .f8e4nv) A).get2 i k)
        (some ((cfgMx.scalesL 128 128 (ds := .u8) sA).get2 i k)) = _
      rw [unpackL_get, scalesL_get, hA k, hsA k]
      exact scaledValue_eq hd.1 (by have := hd.2.1.2; omega))
    (fun k => by
      have hd := (hdom _ (hk k)).2
      show scaledValue .e4m3 .e8m0 ((cfgMx.unpackR 128 256 (d := .f8e4nv) Bm.trans).get2 k j)
        (some ((cfgMx.scalesR 128 256 (ds := .u8) sB).get2 k j)) = _
      rw [unpackR_get, scalesR_get, hB k, hsB k]
      exact scaledValue_eq hd.1 (by have := hd.2.1.2; omega))
    (fun k => ⟨(hdom _ (hk k)).1.2.2, (hdom _ (hk k)).2.2.2⟩)
    (fun t' ht' k => by
      cases ht'
      show Rules.Num.e8m0Dom ((cfgMx.scalesL 128 128 (ds := .u8) sA).get2 i k)
      rw [scalesL_get, hsA k]
      exact (hdom _ (hk k)).1.2.1)
    (fun t' ht' k => by
      cases ht'
      show Rules.Num.e8m0Dom ((cfgMx.scalesR 128 256 (ds := .u8) sB).get2 k j)
      rw [scalesR_get, hsB k]
      exact (hdom _ (hk k)).2.2.1)
    hinv.1
    (by
      rw [habs]
      have hr0 : Rules.Num.relDot16 128 = 130 / 2 ^ 22 := rho_eq
      have ha0 : Rules.Num.absDot16 128 = 130 / 2 ^ 126 := alpha_eq
      rw [hr0, ha0, Math.Float.Format.binary32_maxVal]
      have := abs_nonneg (acc.get2 i j).toRat
      nlinarith)
  refine ⟨hrfin, ?_⟩
  rw [hsum, habs] at herr
  have hr0 : Rules.Num.relDot16 128 = rho := rfl
  have ha0 : Rules.Num.absDot16 128 = alpha := rfl
  rw [hr0, ha0] at herr
  rw [partSum_succ]
  show _ ≤ tol p t + rho * (|partSum p t| + tol p t + chunkAbs p t) + alpha
  have htri : |(r.get2 i j).toRat - (partSum p t + ∑ e ∈ Finset.range 128, p (128 * t + e))| ≤
      |(r.get2 i j).toRat - ((acc.get2 i j).toRat + ∑ e ∈ Finset.range 128, p (128 * t + e))| +
        |(acc.get2 i j).toRat - partSum p t| := by
    have e : (r.get2 i j).toRat - (partSum p t + ∑ e ∈ Finset.range 128, p (128 * t + e)) =
        ((r.get2 i j).toRat - ((acc.get2 i j).toRat + ∑ e ∈ Finset.range 128, p (128 * t + e))) +
          ((acc.get2 i j).toRat - partSum p t) := by ring
    rw [e]; exact abs_add_le _ _
  have hm : rho * (|(acc.get2 i j).toRat| + chunkAbs p t) ≤ rho * (|partSum p t| + tol p t + chunkAbs p t) :=
    mul_le_mul_of_nonneg_left (by linarith) rho_nonneg
  linarith [hinv.2]

/-- After the last step (`K = 128 T`) the partial sum is the whole sum. -/
theorem partSum_end (p : Nat → ℚ) (T : Nat) : partSum p T = dotSum p (128 * T) := rfl

/-! ## The last rounding -/

/-- **The last rounding** (line 240): a finite `v` within `E` of `S`, rounded to `float16`
(`c` equal to it up to NaN payload), is finite and within `E` plus one rounding when
`|S| + E ≤ 65504`. -/
theorem out_f16 {v : F32} {c : F16} {S E : ℚ} (hv : v.isFinite = true) (hE : |v.toRat - S| ≤ E)
    (hc : FloatV.NaNEq (FloatV.cvt v) c) (hroom : |S| + E ≤ Math.Float.Format.binary16.maxVal) :
    c.isFinite = true ∧
      |c.toRat - S| ≤ E + (|S| + E) * Math.Float.Format.binary16.u + Math.Float.Format.binary16.eta := by
  have hvS : |v.toRat| ≤ |S| + E := by
    have := abs_sub_abs_le_abs_sub v.toRat S
    linarith
  obtain ⟨hcf, hce⟩ := FloatV.cvt_err (g := Math.Float.Format.binary16)
    Math.Float.Format.binary16_regular hv (hvS.trans hroom)
  have hn : (FloatV.cvt v : F16).isNaN = false := (FloatV.isFinite_iff.mp hcf).1
  rw [hc.eq hn]
  refine ⟨hcf, ?_⟩
  have hu : 0 ≤ Math.Float.Format.binary16.u := Math.Float.Format.u_nonneg
  have e : (FloatV.cvt v : F16).toRat - S = ((FloatV.cvt v : F16).toRat - v.toRat) + (v.toRat - S) := by ring
  rw [e]
  refine (abs_add_le _ _).trans ?_
  have : |v.toRat| * Math.Float.Format.binary16.u ≤ (|S| + E) * Math.Float.Format.binary16.u :=
    mul_le_mul_of_nonneg_right hvS hu
  linarith

end Kernels.BlockScaledMxfp8
