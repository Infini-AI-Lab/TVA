import GroupedMatmulTma.Sched

/-!
# `grouped_matmul_tma_f16`: facts the per-program proof uses

* `int32` arithmetic on small values (`i32_add`, `i32_sub`, `i32_mul`, `i32_lt`, `i32_le`, …);
* the addresses the pointer tables hold (`intToPtr_ofInt`) and a 2-D descriptor's lanes
  (`inbAt2`, `elemAt2`, `coord2_0`, `coord2_1`, `access2`);
* the descriptors' zero padding (`f16_zero`);
* the constants of the tolerance (`rho_le`, `rho_nonneg`, `alpha_nonneg`, `tol_rel_lt`);
* the accumulation of one step `acc += tl.dot(a, b.T)` (`acc_step`), from `Rules.Num.DotF16`;
* the last rounding to `float16` (`out_f16`).
-/

namespace Kernels.GroupedMatmulTma

open TVA

/-! ## `int32` arithmetic -/

theorem i32_add {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x + y) (h1 : x + y < 2 ^ 31) : (a + b).toInt = x + y := by
  rw [IntV.toInt_add (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_sub {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x - y) (h1 : x - y < 2 ^ 31) : (a - b).toInt = x - y := by
  rw [IntV.toInt_sub (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_mul {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h0 : -2 ^ 31 ≤ x * y) (h1 : x * y < 2 ^ 31) : (a * b).toInt = x * y := by
  rw [IntV.toInt_mul (by norm_num) (by rw [ha, hb]; constructor <;> norm_num <;> omega), ha, hb]

theorem i32_nat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n := IntV.I32.toInt_ofNat h

theorem i32_int {n : Int} (h0 : -2 ^ 31 ≤ n) (h1 : n < 2 ^ 31) : (IntV.ofInt n : I32).toInt = n :=
  IntV.I32.toInt_ofInt ⟨h0, h1⟩

theorem i32_lt {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    IntV.lt a b = decide (x < y) := by
  rw [IntV.lt_eq, ha, hb]

theorem i32_le {a b : I32} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y) :
    IntV.le a b = decide (x ≤ y) := by
  rw [IntV.le_eq, ha, hb]

theorem i32_zero : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)

theorem i32_one : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)

/-- The induction variable of a loop from the `int32` `0` with step `1`, at trip `t`. -/
theorem i32_trip {t : Nat} (ht : t < 2 ^ 31) :
    (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt = t := by
  rw [i32_zero, i32_one, zero_add, mul_one]; exact i32_int (by omega) (by omega)

/-! ## Pointers and descriptors -/

/-- A table element holding the address `b` (`IntV.ofInt b`, an `int64`), cast to a pointer, is the
pointer to `b`. -/
theorem intToPtr_ofInt {b : Int} (h0 : 0 ≤ b) (h1 : b < 2 ^ 64) :
    tl.intToPtr .f16 (IntV.ofInt b : I64) = (⟨b⟩ : Ptr .f16) := by
  unfold tl.intToPtr IntV.ofInt
  congr 1
  rw [BitVec.toNat_ofInt, Int.emod_eq_of_lt h0 (by exact_mod_cast h1), Int.toNat_of_nonneg h0]

/-- A buffer's pointer to element `0` is its base. -/
theorem buf_ptr0 {d : DType} (b : Buf d) : b.ptr 0 = (⟨b.base⟩ : Ptr d) := by
  simp [Buf.ptr]

/-- A lane of a rank-2 block is inside a `[s0, s1]` tensor when both coordinates are. -/
theorem inbAt2 {d : DType} (base : Ptr d) (s0 s1 l : Nat) (bl : Shape) (pad : Padding) (c : Nat → Int) :
    (⟨base, [s0, s1], [l, 1], bl, pad⟩ : TensorDesc d).inbAt c =
      decide ((0 ≤ c 0 ∧ c 0 < s0) ∧ (0 ≤ c 1 ∧ c 1 < s1)) := by
  unfold TensorDesc.inbAt TensorDesc.rank
  refine decide_eq_decide.mpr ⟨fun h => ⟨h 0 (by simp), h 1 (by simp)⟩, fun h k hk => ?_⟩
  simp only [List.length_cons, List.length_nil] at hk
  rcases (show k = 0 ∨ k = 1 by omega) with rfl | rfl
  · exact h.1
  · exact h.2

/-- The element of a row-major `[s0, s1]` tensor (strides `[l, 1]`) at coordinates `c`. -/
theorem elemAt2 {d : DType} (base : Ptr d) (s0 s1 l : Nat) (bl : Shape) (pad : Padding) (c : Nat → Int) :
    (⟨base, [s0, s1], [l, 1], bl, pad⟩ : TensorDesc d).elemAt c = c 0 * l + c 1 := by
  unfold TensorDesc.elemAt TensorDesc.rank
  simp [List.range_succ]

theorem coord2_0 (o0 o1 : Int) {m n : Nat} (i : Fin m) (j : Fin n) :
    TensorDesc.coord [o0, o1] (bs := [m, n]) (i, j, ()) 0 = o0 + i.val := by
  simp [TensorDesc.coord, Idx.coord]

theorem coord2_1 (o0 o1 : Int) {m n : Nat} (i : Fin m) (j : Fin n) :
    TensorDesc.coord [o0, o1] (bs := [m, n]) (i, j, ()) 1 = o1 + j.val := by
  simp [TensorDesc.coord, Idx.coord]

/-- A block access through a row-major 2-D descriptor meets Triton's requirements: positive shape
entries below `2^31`, a 16-byte-aligned base, a row stride of a multiple of 16 bytes, the
descriptor's block, and an innermost offset of a multiple of 16 bytes. -/
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

/-- The descriptors' padding `"zero"` is the `float16` `+0.0`: finite, of value `0`. -/
theorem f16_zero : (FloatV.ofPat 0 : F16).isFinite = true ∧ (FloatV.ofPat 0 : F16).toRat = 0 := by
  have hlt : (0 : ℕ) < 2 ^ Math.Float.Format.binary16.width := Nat.two_pow_pos _
  rw [FloatV.isFinite_ofPat hlt, FloatV.toRat_ofPat hlt]
  exact ⟨Math.Float.Format.isFinite_zero Math.Float.Format.binary16_regular,
    Math.Float.Format.toRat_of_isZero (by decide)⟩

/-! ## The tolerance's constants -/

theorem relDot16_64 : Rules.Num.relDot16 64 = 66 / 2 ^ 22 := by
  unfold Rules.Num.relDot16; norm_num

theorem absDot16_64 : Rules.Num.absDot16 64 = 66 / 2 ^ 126 := by
  unfold Rules.Num.absDot16; norm_num

theorem rho_eq : rho = 66 / 2 ^ 22 + 3 / 2 ^ 24 * (1 + 66 / 2 ^ 22) := by
  unfold rho DotLoop.stepRel
  rw [relDot16_64, DotLoop.binary32_u]; norm_num

theorem alpha_eq : alpha = 66 / 2 ^ 126 * (1 + 3 / 2 ^ 24) + 4 / 2 ^ 150 := by
  unfold alpha DotLoop.stepAbs
  rw [absDot16_64, DotLoop.binary32_u, Math.Float.Format.binary32_eta]; norm_num

theorem rho_nonneg : (0 : ℚ) ≤ rho := by rw [rho_eq]; norm_num

theorem alpha_nonneg : (0 : ℚ) ≤ alpha := by rw [alpha_eq]; norm_num

theorem rho_le : rho ≤ 16 / 10 ^ 6 := by rw [rho_eq]; norm_num

theorem alpha_le : alpha ≤ 1 / 10 ^ 36 := by rw [alpha_eq]; norm_num

/-- The relative part of `tol` on covered instances (`K ≤ 2^13`, `Inst.ok.K_le`): below `0.21%`,
about half the largest relative error of a rounding to `bfloat16` (`2^-8/(1 + 2^-8)`). -/
theorem tol_rel_lt {K : Nat} (hK : K ≤ 2 ^ 13) : (1 + rho) ^ steps K - 1 < 21 / 10000 := by
  have hT : steps K ≤ 128 := by unfold steps; omega
  have h1 : (1 : ℚ) ≤ 1 + rho := by have := rho_nonneg; linarith
  have h2 := pow_le_pow_right₀ h1 hT
  have h3 : (1 + rho) ^ 128 ≤ (1 + 16 / 10 ^ 6) ^ 128 :=
    pow_le_pow_left₀ (by have := rho_nonneg; linarith) (by have := rho_le; linarith) 128
  have h4 : ((1 : ℚ) + 16 / 10 ^ 6) ^ 128 < 1 + 21 / 10000 := by norm_num
  linarith

/-! ## The accumulation -/

/-- Product `e` of the reduction, `0` from `K` on (the lanes the descriptors pad). -/
def prodV (va vb : Nat → F16) (K e : Nat) : ℚ := if e < K then (va e).toRat * (vb e).toRat else 0

/-- The exact partial sum after `t` steps: the products `e < 64·t`. -/
def partS (va vb : Nat → F16) (K t : Nat) : ℚ := ∑ e ∈ Finset.range (64 * t), prodV va vb K e

/-- The partial sum of the magnitudes. -/
def partA (va vb : Nat → F16) (K t : Nat) : ℚ := ∑ e ∈ Finset.range (64 * t), |prodV va vb K e|

theorem f16_abs_le {x : F16} (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have := FloatV.abs_toRat_le_maxVal Math.Float.Format.binary16_regular hx
  rwa [Math.Float.Format.binary16_maxVal] at this

theorem prodV_le {va vb : Nat → F16} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (e : Nat) : |prodV va vb K e| ≤ 2 ^ 32 := by
  unfold prodV
  split_ifs with h
  · rw [abs_mul]
    have h1 := f16_abs_le (hfa e h)
    have h2 := f16_abs_le (hfb e h)
    have h3 : |(va e).toRat| * |(vb e).toRat| ≤ 65504 * 65504 :=
      mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
    linarith
  · simp

theorem partA_le {va vb : Nat → F16} {K : Nat} (hfa : ∀ e, e < K → (va e).isFinite = true)
    (hfb : ∀ e, e < K → (vb e).isFinite = true) (t : Nat) : partA va vb K t ≤ 64 * t * 2 ^ 32 := by
  unfold partA
  calc ∑ e ∈ Finset.range (64 * t), |prodV va vb K e| ≤ ∑ _e ∈ Finset.range (64 * t), (2 : ℚ) ^ 32 :=
        Finset.sum_le_sum fun e _ => prodV_le hfa hfb e
    _ = 64 * t * 2 ^ 32 := by simp

theorem partS_succ (va vb : Nat → F16) (K t : Nat) :
    partS va vb K (t + 1) = partS va vb K t + ∑ k : Fin 64, prodV va vb K (64 * t + k) := by
  unfold partS
  rw [show 64 * (t + 1) = 64 * t + 64 by ring, Finset.sum_range_add,
    Fin.sum_univ_eq_sum_range (fun k => prodV va vb K (64 * t + k)) 64]

theorem partA_succ (va vb : Nat → F16) (K t : Nat) :
    partA va vb K (t + 1) = partA va vb K t + ∑ k : Fin 64, |prodV va vb K (64 * t + k)| := by
  unfold partA
  rw [show 64 * (t + 1) = 64 * t + 64 by ring, Finset.sum_range_add,
    Fin.sum_univ_eq_sum_range (fun k => |prodV va vb K (64 * t + k)|) 64]

theorem val_f16 {x : F16} {v : ℚ} (hx : x.isFinite = true) (hv : x.toRat = v) :
    DType.val? .f16 x = some v := by
  show (if x.isFinite = true then some x.toRat else none) = some v
  rw [ite_eq_left hx, hv]

/-- The accumulator before the loop (line 262): `+0.0`, exact. -/
theorem acc0_val (va vb : Nat → F16) (K : Nat) (i j : Fin 128) :
    ((tl.zeros [128, 128] .f32).get2 i j).isFinite = true ∧
      |((tl.zeros [128, 128] .f32).get2 i j).toRat - partS va vb K 0| ≤ DotLoop.bound rho alpha 0 (partA va vb K 0) := by
  have h := DotLoop.val?_f32.mp DotLoop.zero_val
  refine ⟨h.1, ?_⟩
  show |(FloatV.ofPat 0 : F32).toRat - partS va vb K 0| ≤ _
  rw [h.2]
  simp [partS, DotLoop.bound]

/-- **One step of the loop** (line 266), for one output element `(i, j)` of the tile: the
accumulator stays finite and within `DotLoop.bound rho alpha` of the exact partial sum, whichever
form of `tl.dot_add` the device runs, by `Rules.Num.DotF16` (`DotLoop.dotAdd_step`) and
`DotLoop.accum_step`. `A` and `Bm` are the loaded blocks of `a` and `b`: their lanes in bounds hold
the matrix elements, the others the padding `+0.0`. -/
theorem acc_step {P : Platform} (hdot : Rules.Num.DotF16 P) {fusion : Bool}
    {A Bm : Tensor [128, 64] F16} {acc r : Tensor [128, 128] F32}
    (hr : P.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 } A Bm.trans acc r ∨
      ∃ dd, P.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 } A Bm.trans
        (Tensor.full (FloatV.ofPat 0)) dd ∧ ∀ ix, AccAdd fusion .f32 (dd.get ix) (acc.get ix) (r.get ix))
    (i j : Fin 128) {va vb : Nat → F16} {K t : Nat} (hK : K ≤ 2 ^ 16) (ht : 64 * t < K)
    (hfa : ∀ e, e < K → (va e).isFinite = true) (hfb : ∀ e, e < K → (vb e).isFinite = true)
    (hA : ∀ k : Fin 64, (64 * t + k < K → A.get2 i k = va (64 * t + k)) ∧
      (K ≤ 64 * t + k → (A.get2 i k).isFinite = true ∧ (A.get2 i k).toRat = 0))
    (hB : ∀ k : Fin 64, (64 * t + k < K → Bm.get2 j k = vb (64 * t + k)) ∧
      (K ≤ 64 * t + k → (Bm.get2 j k).isFinite = true ∧ (Bm.get2 j k).toRat = 0))
    (hinv : (acc.get2 i j).isFinite = true ∧
      |(acc.get2 i j).toRat - partS va vb K t| ≤ DotLoop.bound rho alpha t (partA va vb K t)) :
    (r.get2 i j).isFinite = true ∧
      |(r.get2 i j).toRat - partS va vb K (t + 1)| ≤ DotLoop.bound rho alpha (t + 1) (partA va vb K (t + 1)) := by
  -- the lanes' values
  have hfA : ∀ k : Fin 64, (A.get2 i k).isFinite = true := fun k => by
    by_cases h : 64 * t + k < K
    · rw [(hA k).1 h]; exact hfa _ h
    · exact ((hA k).2 (by omega)).1
  have hfB : ∀ k : Fin 64, (Bm.get2 j k).isFinite = true := fun k => by
    by_cases h : 64 * t + k < K
    · rw [(hB k).1 h]; exact hfb _ h
    · exact ((hB k).2 (by omega)).1
  have hprod : ∀ k : Fin 64, (A.get2 i k).toRat * (Bm.get2 j k).toRat = prodV va vb K (64 * t + k) := by
    intro k
    unfold prodV
    by_cases h : 64 * t + k < K
    · rw [ite_eq_left h, (hA k).1 h, (hB k).1 h]
    · rw [ite_eq_right h, ((hA k).2 (by omega)).2, zero_mul]
  have hchunk : ∑ k : Fin 64, (A.get2 i k).toRat * (Bm.get2 j k).toRat =
      ∑ k : Fin 64, prodV va vb K (64 * t + k) := Finset.sum_congr rfl fun k _ => hprod k
  have hchunkA : ∑ k : Fin 64, |(A.get2 i k).toRat * (Bm.get2 j k).toRat| =
      ∑ k : Fin 64, |prodV va vb K (64 * t + k)| := Finset.sum_congr rfl fun k _ => by rw [hprod k]
  -- sizes
  have ht1 : (t : ℚ) ≤ 1024 := by
    have : t ≤ 1024 := by omega
    exact_mod_cast this
  have hB0 : partA va vb K t ≤ 2 ^ 48 := by
    have h1 := partA_le hfa hfb t
    have h2 : (64 * t : ℚ) ≤ 2 ^ 16 := by
      have : 64 * t ≤ 2 ^ 16 := by omega
      exact_mod_cast this
    nlinarith
  have hBn : 0 ≤ partA va vb K t := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hcA : ∑ k : Fin 64, |prodV va vb K (64 * t + k)| ≤ 2 ^ 38 := by
    calc ∑ k : Fin 64, |prodV va vb K (64 * t + k)| ≤ ∑ _k : Fin 64, (2 : ℚ) ^ 32 :=
          Finset.sum_le_sum fun k _ => prodV_le hfa hfb _
      _ = 2 ^ 38 := by simp; norm_num
  have hcA0 : 0 ≤ ∑ k : Fin 64, |prodV va vb K (64 * t + k)| := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hρ0 := rho_nonneg
  have hα0 := alpha_nonneg
  have hρ := rho_le
  have hα := alpha_le
  have hbound : DotLoop.bound rho alpha t (partA va vb K t) ≤ 2 ^ 44 := by
    have htρ : (t : ℚ) * rho ≤ 1 / 2 := by nlinarith
    refine (DotLoop.bound_le hρ0 hα0 htρ hBn).trans ?_
    have ht0 : (0 : ℚ) ≤ t := Nat.cast_nonneg t
    have e1 : rho * partA va vb K t ≤ 16 / 10 ^ 6 * 2 ^ 48 := mul_le_mul hρ hB0 hBn (by norm_num)
    have e2 : rho * partA va vb K t + alpha ≤ 16 / 10 ^ 6 * 2 ^ 48 + 1 / 10 ^ 36 := by linarith
    have e3 : 2 * (t : ℚ) * (rho * partA va vb K t + alpha) ≤ 2 * 1024 * (16 / 10 ^ 6 * 2 ^ 48 + 1 / 10 ^ 36) :=
      mul_le_mul (by linarith) e2 (by positivity) (by norm_num)
    linarith [show (2 : ℚ) * 1024 * (16 / 10 ^ 6 * 2 ^ 48 + 1 / 10 ^ 36) ≤ 2 ^ 44 by norm_num]
  have hSt : |partS va vb K t| ≤ partA va vb K t := Finset.abs_sum_le_sum_abs _ _
  have hvc : |(acc.get2 i j).toRat| ≤ 2 ^ 49 := by
    have := abs_sub_abs_le_abs_sub (acc.get2 i j).toRat (partS va vb K t)
    linarith [hinv.2]
  -- the step, by `DotF16` and `DotLoop.dotAdd_step`
  have hrel0 : (0 : ℚ) ≤ Rules.Num.relDot16 64 := by rw [relDot16_64]; norm_num
  have habs0 : (0 : ℚ) ≤ Rules.Num.absDot16 64 := by rw [absDot16_64]; norm_num
  obtain ⟨hrfin, herr⟩ := DotLoop.dotAdd_step (hdot .tf32) hrel0 habs0 hr i j
    (va := fun k => (A.get2 i k).toRat) (vb := fun k => (Bm.get2 j k).toRat)
    (fun k => val_f16 (hfA k) rfl) (fun k => val_f16 (hfB k) rfl) (fun _ => ⟨trivial, trivial⟩) hinv.1 (by
      rw [hchunkA, relDot16_64, absDot16_64, Math.Float.Format.binary32_maxVal,
        Math.Float.Format.binary32_eta]
      have h0 := abs_nonneg (acc.get2 i j).toRat
      norm_num
      nlinarith)
  refine ⟨hrfin, ?_⟩
  rw [partS_succ, partA_succ]
  rw [hchunk, hchunkA] at herr
  exact DotLoop.accum_step hρ0 hα0 hSt (Finset.abs_sum_le_sum_abs _ _) hinv.2 herr

/-- After the last step (`64·T ≥ K`) the partial sums are the whole sums. -/
theorem partS_end (va vb : Nat → F16) {K T : Nat} (hT : K ≤ 64 * T) :
    partS va vb K T = dotSum va vb K ∧ partA va vb K T = dotAbs va vb K := by
  unfold partS partA dotSum dotAbs
  obtain ⟨x, hx⟩ : ∃ x, 64 * T = K + x := ⟨64 * T - K, by omega⟩
  rw [hx, Finset.sum_range_add, Finset.sum_range_add]
  have h1 : ∀ e ∈ Finset.range K, prodV va vb K e = (va e).toRat * (vb e).toRat := fun e he => by
    unfold prodV; rw [ite_eq_left (Finset.mem_range.mp he)]
  have h2 : ∀ y, prodV va vb K (K + y) = 0 := fun y => by
    unfold prodV; rw [ite_eq_right (by omega)]
  constructor
  · rw [Finset.sum_congr rfl h1]; simp [h2]
  · rw [Finset.sum_congr rfl fun e he => by rw [h1 e he]]; simp [h2]

/-- **The last rounding** (line 271): a finite `v` within `E` of `S`, rounded to `float16`
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

end Kernels.GroupedMatmulTma
