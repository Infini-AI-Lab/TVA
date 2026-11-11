import Proof.Post

/-!
# Float operations as statements about reals

One lemma per operation of a trip: what the kit's rounding lemmas and the device facts say about
the `float32` (or `float16`) values of one lane, in reals and with the constants of `Proof.Tol`.
-/

namespace Claims.AttnFwd

open TVA Finset

noncomputable section

/-- The binary32 format is regular. -/
theorem binary32_reg : Format.binary32.Regular := Format.Std.regular Format.binary32_std

/-- The power `2^(-24)` equals `u32`. -/
theorem u32_eq : (2 : ℝ) ^ (-24 : ℤ) = u32 := by unfold u32; norm_num
/-- The power `2^(-150)` equals `1 / 2^150`. -/
theorem eta32_eq : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
/-- The power `2^(-22)` equals `q22`. -/
theorem q22_eq : (2 : ℝ) ^ (-22 : ℤ) = q22 := by unfold q22; norm_num

/-- `tl.mul_add a b c` (a product then a sum, or an `fma`) of finite `float32` values with room. -/
theorem mulAdd_real {fusion : Bool} {a b c r : F32}
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul a b) c) r ∨ (fusion = true ∧ FloatV.NaNEq (FloatV.fma a b c) r))
    (ha : a.isFinite = true) (hb : b.isFinite = true) (hc : c.isFinite = true)
    (hM : |a.toReal * b.toReal| + |c.toReal| ≤ 2 ^ 100) :
    r.isFinite = true ∧
      |r.toReal - (a.toReal * b.toReal + c.toReal)| ≤
        3 * u32 * (|a.toReal * b.toReal| + |c.toReal|) + 4 / 2 ^ 150 := by
  have hM' : |a.toRat * b.toRat| + |c.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have : ((|a.toRat * b.toRat| + |c.toRat| : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      rw [Format.binary32_maxVal_real, Format.binary32_eta_real]
      unfold FloatV.toReal at hM
      have : (2 : ℝ) ^ 100 ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * 2 ^ (-150 : ℤ) := by norm_num
      linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.mulAdd_err binary32_reg (by rw [Format.binary32_u]; norm_num) h ha hb hc hM'
  refine ⟨hf, ?_⟩
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  rw [Format.binary32_u_real, Format.binary32_eta_real, u32_eq, eta32_eq] at this
  unfold FloatV.toReal
  linarith

/-- A fusable addition `AddRel` of finite `float32` values with room. -/
theorem addrel_real {fusion : Bool} {x y r : F32} (h : AddRel fusion x y r)
    (hx : x.isFinite = true) (hy : y.isFinite = true) (hM : |x.toReal| + |y.toReal| ≤ 2 ^ 100) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal + y.toReal)| ≤ 3 * u32 * (|x.toReal| + |y.toReal|) + 4 / 2 ^ 150 := by
  have hM' : |x.toRat| + |y.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have : ((|x.toRat| + |y.toRat| : ℚ) : ℝ) ≤
        ((Format.binary32.maxVal / 2 - 2 * Format.binary32.eta : ℚ) : ℝ) := by
      push_cast
      rw [Format.binary32_maxVal_real, Format.binary32_eta_real]
      unfold FloatV.toReal at hM
      have : (2 : ℝ) ^ 100 ≤ (2 ^ 24 - 1) * 2 ^ 104 / 2 - 2 * 2 ^ (-150 : ℤ) := by norm_num
      linarith
    exact_mod_cast this
  obtain ⟨hf, he⟩ := FloatV.AddRel.err binary32_reg (by decide) (by rw [Format.binary32_u]; norm_num) h hx hy hM'
  refine ⟨hf, ?_⟩
  have := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at this
  rw [Format.binary32_u_real, Format.binary32_eta_real, u32_eq, eta32_eq] at this
  unfold FloatV.toReal
  linarith

/-- `tl.mul` (IEEE) of finite `float32` values with room. -/
theorem mul_real {x y r : F32} (h : FloatV.NaNEq (FloatV.mul x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hM : |x.toReal * y.toReal| ≤ 2 ^ 100) :
    r.isFinite = true ∧ |r.toReal - x.toReal * y.toReal| ≤ u32 * |x.toReal * y.toReal| + 1 / 2 ^ 150 := by
  obtain ⟨hf, he⟩ := FloatV.mul_err_real binary32_reg hx hy (by
    rw [Format.binary32_maxVal_real]; refine hM.trans ?_; norm_num)
  have hr : r = FloatV.mul x y := h.eq_of_isFinite hf
  rw [hr, Format.binary32_u_real, Format.binary32_eta_real, u32_eq, eta32_eq] at *
  exact ⟨hf, by rw [hr]; linarith [mul_comm |x.toReal * y.toReal| u32]⟩

/-- `tl.sub` (IEEE) of finite `float32` values with room. -/
theorem sub_real {x y r : F32} (h : FloatV.NaNEq (FloatV.sub x y) r) (hx : x.isFinite = true)
    (hy : y.isFinite = true) (hM : |x.toReal - y.toReal| ≤ 2 ^ 100) :
    r.isFinite = true ∧ |r.toReal - (x.toReal - y.toReal)| ≤ u32 * |x.toReal - y.toReal| + 1 / 2 ^ 150 := by
  obtain ⟨hf, he⟩ := FloatV.sub_err_real binary32_reg hx hy (by
    rw [Format.binary32_maxVal_real]; refine hM.trans ?_; norm_num)
  have hr : r = FloatV.sub x y := h.eq_of_isFinite hf
  rw [hr, Format.binary32_u_real, Format.binary32_eta_real, u32_eq, eta32_eq] at *
  exact ⟨hf, by rw [hr]; linarith [mul_comm |x.toReal - y.toReal| u32]⟩

/-- `tl.math.exp2` on a finite `float32` (device fact): finite, within `q22·2^x + 2^-126` of `2^x`. -/
theorem exp2_real {P : Platform} (hP : Device P) {x r : F32} (h : P.math .exp2 .binary32 x r)
    (hx : x.isFinite = true) (hup : x.toReal ≤ 100) :
    r.isFinite = true ∧ |r.toReal - (2 : ℝ) ^ x.toReal| ≤ q22 * (2 : ℝ) ^ x.toReal + 1 / 2 ^ 126 := by
  obtain ⟨ref, href, hacc⟩ := hP.exp2F32
  have hr : ref = fun y => (2 : ℝ) ^ y := by
    simp only [MathFn.ref, Option.some.injEq] at href; exact href.symm
  subst hr
  have h2 : (2 : ℝ) ^ x.toReal ≤ 2 ^ (100 : ℝ) := Real.rpow_le_rpow_of_exponent_le (by norm_num) hup
  have h3 : (2 : ℝ) ^ (100 : ℝ) = 2 ^ 100 := by norm_num
  have h4 : (0 : ℝ) < (2 : ℝ) ^ x.toReal := Real.rpow_pos_of_pos (by norm_num) _
  obtain ⟨hf, he⟩ := hacc x r h hx (Set.mem_univ _) (by simp [MathFn.natDom]) (by
    rw [Format.binary32_maxVal_real]
    have : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
    have : (2 : ℝ) ^ (-126 : ℤ) ≤ 1 := by norm_num
    rw [abs_of_pos h4]
    nlinarith)
  refine ⟨hf, ?_⟩
  rw [abs_of_pos h4, q22_eq] at he
  rw [show (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 by norm_num] at he
  exact he

/-- `x.to(float16)` of a `float32` of magnitude at most `2`. -/
theorem cvt16_real {x : F32} {r : F16} (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r)
    (hx : x.isFinite = true) (hM : |x.toReal| ≤ 2) :
    r.isFinite = true ∧ |r.toReal - x.toReal| ≤ u16 * |x.toReal| + e16 := by
  obtain ⟨hf, he⟩ := FloatV.cvt_err_real (g := Format.binary16) (Format.Std.regular Format.binary16_std) hx (by
    rw [Format.binary16_maxVal_real]; refine hM.trans ?_; norm_num)
  have hr : r = FloatV.cvt x := by
    have := h.eq_of_isFinite (by rw [FloatV.cvtWith_rn]; exact hf)
    rw [FloatV.cvtWith_rn] at this
    exact this
  subst hr
  refine ⟨hf, ?_⟩
  rw [Format.binary16_u_real, Format.binary16_eta_real] at he
  have e11 : (2 : ℝ) ^ (-11 : ℤ) = u16 := by unfold u16; norm_num
  have e25 : (2 : ℝ) ^ (-25 : ℤ) = e16 := by unfold e16; norm_num
  rw [e11, e25] at he
  linarith [mul_comm |x.toReal| u16]

/-- Summing a mapped list over `range n` equals the finset sum over `range n`. -/
theorem sumList_eq (f : ℕ → ℚ) (n : ℕ) : ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := rfl

/-- **`tl.sum(p, 1)`** of a row of `64` finite `float32` lanes (any tree of the device's depth): finite,
within `rhoS·Σ|p| + 2^-143` of `Σ p`. -/
theorem sum_real {P : Platform} (hP : Device P) {p : Fin 64 → F32} {s : F32}
    (hp : ∀ j, (p j).isFinite = true) (hbig : ∑ j, |(p j).toReal| ≤ 2 ^ 10)
    (ht : ∃ t : Tree, t.Covers 64 ∧ t.depth ≤ tl.sumBound P 64 ∧
      t.Eval (AddRel false) (fun j => if h : j < 64 then p ⟨j, h⟩ else default) s) :
    s.isFinite = true ∧ |s.toReal - ∑ j, (p j).toReal| ≤ rhoS * ∑ j, |(p j).toReal| + 1 / 2 ^ 143 := by
  obtain ⟨t, hcov, hdep, hev⟩ := ht
  have hD : tl.sumBound P 64 ≤ 6 := by
    unfold tl.sumBound
    have h1 := hP.sumDepth 64
    have h2 : Nat.clog 2 64 = 6 := by decide
    omega
  set x : ℕ → F32 := fun j => if h : j < 64 then p ⟨j, h⟩ else default with hx
  have hxf : ∀ i, i < 64 → (x i).isFinite = true := fun i hi => by simp [hx, hi, hp]
  have hsum_abs : ((List.range 64).map fun i => |(x i).toRat|).sum = ∑ j : Fin 64, |(p j).toRat| := by
    rw [sumList_eq]
    rw [Finset.sum_range]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp [hx]
  have hsum : ((List.range 64).map fun i => (x i).toRat).sum = ∑ j : Fin 64, (p j).toRat := by
    rw [sumList_eq]
    rw [Finset.sum_range]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp [hx]
  have hbigq : ((∑ j : Fin 64, |(p j).toRat| : ℚ) : ℝ) ≤ 2 ^ 10 := by
    push_cast
    simpa [FloatV.toReal] using hbig
  obtain ⟨hf, he⟩ := sum_err_depth binary32_reg x hcov (hdep.trans hD) hev hxf (by
    rw [hsum_abs]
    have hbq : (∑ j : Fin 64, |(p j).toRat| : ℚ) ≤ 2 ^ 10 := by exact_mod_cast hbigq
    have : Tree.errBound Format.binary32.u Format.binary32.eta 6 (64 - 1) (∑ j : Fin 64, |(p j).toRat|) ≤ 2 ^ 20 := by
      unfold Tree.errBound
      rw [Format.binary32_u, Format.binary32_eta]
      have h0 : (0 : ℚ) ≤ ∑ j : Fin 64, |(p j).toRat| := Finset.sum_nonneg fun j _ => abs_nonneg _
      have hc : ((1 + (2 : ℚ) ^ (-24 : ℤ)) ^ 6 - 1) ≤ 1 / 2 ^ 10 := by norm_num
      have h1 : ((1 + (2 : ℚ) ^ (-24 : ℤ)) ^ 6 - 1) * ∑ j : Fin 64, |(p j).toRat| ≤ 1 / 2 ^ 10 * 2 ^ 10 :=
        mul_le_mul hc hbq h0 (by norm_num)
      have h2 : (2 : ℚ) ^ (-150 : ℤ) * (64 - 1 : ℕ) * (1 + 2 ^ (-24 : ℤ)) ^ 6 ≤ 1 := by norm_num
      norm_num at h1 h2 ⊢
      linarith
    rw [Format.binary32_maxVal]
    linarith [hbq])
  refine ⟨hf, ?_⟩
  rw [hsum] at he
  rw [hsum_abs] at he
  have he' := (Rat.cast_le (K := ℝ)).mpr he
  unfold Tree.errBound at he'
  rw [Format.binary32_u, Format.binary32_eta] at he'
  push_cast at he'
  unfold FloatV.toReal
  have e1 : (((1 + (2 : ℝ) ^ (-24 : ℤ)) ^ 6 - 1)) = rhoS := by unfold rhoS u32; norm_num
  have e2 : (2 : ℝ) ^ (-150 : ℤ) * ((64 - 1 : ℕ) : ℝ) * (1 + (2 : ℝ) ^ (-24 : ℤ)) ^ 6 ≤ 1 / 2 ^ 143 := by
    norm_num
  rw [e1] at he'
  norm_num at he' e2 ⊢
  linarith

/-- The reads of a `float16` tile whose elements are finite give their values (`val16`). -/
theorem val16_eq_toRat {m : Mem} {b : Buf .f16} {i : Nat} {x : F16}
    (h : m.read .f16 (b.ptr (i : Int)).addr = some x) : val16 m b i = x.toRat := by
  unfold val16; rw [h]

/-- **The score `dot`**: lane `(r, j)` of `tl.dot(q, k^T)` against the exact score `q · k`. -/
theorem lane_dot1 {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem} {bh sm τ : Nat}
    (h : HeadOk d init bh) (hτ : 64 * τ + 64 ≤ d.N_CTX) (hsm : sm < G d)
    {q : Tensor [128, d.HEAD_DIM] F16} {kt : Tensor [64, d.HEAD_DIM] F16} {qk : Tensor [128, 64] F32}
    (hq : ∀ (r : Fin 128) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.q.ptr (idx d bh (sm * 128 + r) e : Int)).addr = some (q.get2 r e))
    (hk : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.k.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (kt.get2 kk e))
    (hdot : P.dot cfg16 q kt.trans (tl.zeros [128, 64] DType.f32) qk) (r : Fin 128) (j : Fin 64) :
    (qk.get2 r j).isFinite = true ∧
      |(qk.get2 r j).toReal - (score d init bh (sm * 128 + r) (64 * τ + j) : ℝ)| ≤
        rD d.HEAD_DIM * (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ) +
          ((d.HEAD_DIM : ℝ) + 2) / 2 ^ 126 := by
  have hn : sm * 128 + (r : Nat) < d.N_CTX := sm_row_lt hd hsm r.isLt
  have hkN : 64 * τ + (j : Nat) < d.N_CTX := by have := j.isLt; omega
  have hfq : ∀ e : Fin d.HEAD_DIM, (q.get2 r e).isFinite = true := by
    intro e
    obtain ⟨x, hx, hxf⟩ := (h.1 (sm * 128 + r) e hn e.isLt).1
    have := hq r e
    rw [hx] at this
    cases this; exact hxf
  have hfk : ∀ e : Fin d.HEAD_DIM, (kt.get2 j e).isFinite = true := by
    intro e
    obtain ⟨x, hx, hxf⟩ := (h.1 (64 * τ + j) e hkN e.isLt).2.1
    have := hk j e
    rw [hx] at this
    cases this; exact hxf
  have hva : ∀ e : Fin d.HEAD_DIM, DType.val? .f16 (q.get2 r e) = some (val16 init d.q (idx d bh (sm * 128 + r) e)) := by
    intro e
    rw [val16_eq_toRat (hq r e)]
    exact DType.val?_f16.mpr ⟨hfq e, rfl⟩
  have hvb : ∀ e : Fin d.HEAD_DIM, DType.val? .f16 (kt.trans.get2 e j) =
      some (val16 init d.k (idx d bh (64 * τ + j) e)) := by
    intro e
    rw [val16_eq_toRat (hk j e)]
    simpa using DType.val?_f16.mpr ⟨hfk e, rfl⟩
  have hsum : ∑ e : Fin d.HEAD_DIM, val16 init d.q (idx d bh (sm * 128 + r) e) *
      val16 init d.k (idx d bh (64 * τ + j) e) = score d init bh (sm * 128 + r) (64 * τ + j) := by
    unfold score
    exact Fin.sum_univ_eq_sum_range (fun e => val16 init d.q (idx d bh (sm * 128 + r) e) *
      val16 init d.k (idx d bh (64 * τ + j) e)) d.HEAD_DIM
  have hlift : ∑ e : Fin d.HEAD_DIM, Device.liftMag .f16 (val16 init d.q (idx d bh (sm * 128 + r) e)) *
      Device.liftMag .f16 (val16 init d.k (idx d bh (64 * τ + j) e)) = scoreMag d init bh (sm * 128 + r) (64 * τ + j) := by
    unfold scoreMag
    exact Fin.sum_univ_eq_sum_range (fun e => Device.liftMag .f16 (val16 init d.q (idx d bh (sm * 128 + r) e)) *
      Device.liftMag .f16 (val16 init d.k (idx d bh (64 * τ + j) e))) d.HEAD_DIM
  have hmag : scoreMag d init bh (sm * 128 + r) (64 * τ + j) ≤ 2 ^ 74 := by
    have h3 := h.2.2 (sm * 128 + r) (64 * τ + j) hn hkN
    have hs' : (1 : ℚ) / 2 ^ 64 ≤ sigma d := hd.scale.2.1
    have hm0 := scoreMag_nonneg (d := d) init bh (sm * 128 + r) (64 * τ + j)
    have h4 : (1 / 2 ^ 64 : ℚ) * scoreMag d init bh (sm * 128 + r) (64 * τ + j) ≤
        sigma d * scoreMag d init bh (sm * 128 + r) (64 * τ + j) := mul_le_mul_of_nonneg_right hs' hm0
    linarith
  have hroom : |(0 : ℚ)| + ∑ e : Fin d.HEAD_DIM, Device.liftMag .f16 (val16 init d.q (idx d bh (sm * 128 + r) e)) *
      Device.liftMag .f16 (val16 init d.k (idx d bh (64 * τ + j) e)) +
      (Device.relDot16 d.HEAD_DIM * (|(0 : ℚ)| + ∑ e : Fin d.HEAD_DIM, Device.liftMag .f16 (val16 init d.q (idx d bh (sm * 128 + r) e)) *
      Device.liftMag .f16 (val16 init d.k (idx d bh (64 * τ + j) e))) + Device.absDot16 d.HEAD_DIM) ≤
      (⟨DType.f32, DType.f32, DType.f32, InputPrecision.tf32, 0⟩ : DotCfg).acc.bound := by
    rw [hlift]
    show _ ≤ Format.binary32.maxVal
    rw [Format.binary32_maxVal]
    have hD := hd.head_dim
    have hrel : Device.relDot16 d.HEAD_DIM ≤ 1 := by
      unfold Device.relDot16
      rcases hD with h | h | h <;> rw [h] <;> norm_num
    have habs : Device.absDot16 d.HEAD_DIM ≤ 1 := by
      unfold Device.absDot16
      rcases hD with h | h | h <;> rw [h] <;> norm_num
    have hm0 := scoreMag_nonneg (d := d) init bh (sm * 128 + r) (64 * τ + j)
    have hrel0 : 0 ≤ Device.relDot16 d.HEAD_DIM := by unfold Device.relDot16; positivity
    simp only [abs_zero, zero_add]
    nlinarith
  obtain ⟨vr, hvr, herr⟩ := hP.dotF16 .tf32 q kt.trans (tl.zeros [128, 64] DType.f32) qk hdot r j
    (fun e => val16 init d.q (idx d bh (sm * 128 + r) e)) (fun e => val16 init d.k (idx d bh (64 * τ + j) e))
    0 hva hvb DType.val?_zero_f32 hroom
  obtain ⟨hfin, hvr'⟩ := DType.val?_f32.mp hvr
  refine ⟨hfin, ?_⟩
  rw [hsum, hlift] at herr
  have h1 := (Rat.cast_le (K := ℝ)).mpr herr
  have hx : (qk.get2 r j).toReal = vr := by unfold FloatV.toReal; rw [hvr']
  rw [hx]
  push_cast at h1
  simp only [abs_zero, zero_add] at h1
  have e1 : ((Device.relDot16 d.HEAD_DIM : ℚ) : ℝ) = rD d.HEAD_DIM := by
    unfold Device.relDot16 rD; push_cast; norm_num; ring
  have e2 : ((Device.absDot16 d.HEAD_DIM : ℚ) : ℝ) = ((d.HEAD_DIM : ℝ) + 2) / 2 ^ 126 := by
    unfold Device.absDot16; push_cast; norm_num; ring
  rw [e1, e2] at h1
  simpa using h1

set_option maxHeartbeats 1600000 in
/-- **The score part of a lane's exponent** (base 2): the score `dot` within `rD·scoreMag`, the
scale within `3.1·u32`: `|c·d̂|` is at most `1.4429·X` and `c·d̂` is within `ξs` of
`σ·score/ln 2`, for scaled score magnitudes at most `X`. -/
theorem expo_score {σ Λ sc dh c X : ℝ} (D : ℕ) (hD : D = 64 ∨ D = 128 ∨ D = 256)
    (hX : 0 ≤ X ∧ X ≤ 1024) (hσ : 1 / 2 ^ 64 ≤ σ ∧ σ ≤ 2 ^ 64) (hΛ0 : 0 ≤ Λ) (hΛ : σ * Λ ≤ X)
    (hsc : |sc| ≤ Λ) (hd : |dh - sc| ≤ rD D * Λ + ((D : ℝ) + 2) / 2 ^ 126)
    (hc : |c - σ / Real.log 2| ≤ 31 / 10 * u32 * (σ / Real.log 2)) :
    |c * dh| ≤ 14429 / 10000 * X + 1 / 2 ^ 44 ∧
      |c * dh - σ * sc / Real.log 2| ≤ 14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43 := by
  have hlog0 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hK : 1 / Real.log 2 ≤ 14427 / 10000 := by
    rw [div_le_iff₀ hlog0]; linarith [Real.log_two_gt_d9]
  have hK0 : 0 < 1 / Real.log 2 := by positivity
  set K := 1 / Real.log 2 with hKdef
  set c0 := σ / Real.log 2 with hc0def
  have hc0 : c0 = σ * K := by rw [hc0def, hKdef]; ring
  have hσ0 : 0 < σ := lt_of_lt_of_le (by norm_num) hσ.1
  have hc00 : 0 < c0 := by rw [hc0]; positivity
  have hP : c0 * Λ ≤ 14427 / 10000 * X := by
    have : c0 * Λ = K * (σ * Λ) := by rw [hc0]; ring
    rw [this]
    calc K * (σ * Λ) ≤ 14427 / 10000 * (σ * Λ) := mul_le_mul_of_nonneg_right hK (by positivity)
      _ ≤ 14427 / 10000 * X := by linarith
  have hP0 : 0 ≤ c0 * Λ := by positivity
  have hc0big : c0 ≤ 2 ^ 65 := by
    rw [hc0]
    have : K ≤ 2 := by linarith
    calc σ * K ≤ 2 ^ 64 * 2 := mul_le_mul hσ.2 this hK0.le (by positivity)
      _ = 2 ^ 65 := by norm_num
  have hrD : rD D ≤ 1 / 10 ^ 4 := by
    unfold rD; rcases hD with h | h | h <;> rw [h] <;> norm_num
  have hrD0 : 0 ≤ rD D := by unfold rD; positivity
  have habs : ((D : ℝ) + 2) / 2 ^ 126 ≤ 1 / 2 ^ 110 := by
    rcases hD with h | h | h <;> rw [h] <;> norm_num
  have hu : u32 = 1 / 2 ^ 24 := rfl
  have hu0 : 0 < u32 := by rw [hu]; positivity
  -- |dh|
  have hdh : |dh| ≤ Λ * (1 + rD D) + 1 / 2 ^ 110 := by
    have := abs_sub_abs_le_abs_sub dh sc
    nlinarith
  -- |c|
  have hcabs : |c| ≤ c0 * (1 + 31 / 10 * u32) := by
    have := abs_sub_abs_le_abs_sub c c0
    rw [abs_of_pos hc00] at this
    nlinarith
  have hu' : u32 = 1 / 2 ^ 24 := rfl
  have hr1 : 1 + rD D ≤ 1 + 1 / 10 ^ 4 := by linarith
  -- the product c * dh
  have hcdh : |c * dh| ≤ (1 + 31 / 10 * u32) * (1 + rD D) * (c0 * Λ) + (1 + 31 / 10 * u32) * (c0 * (1 / 2 ^ 110)) := by
    rw [abs_mul]
    calc |c| * |dh| ≤ (c0 * (1 + 31 / 10 * u32)) * (Λ * (1 + rD D) + 1 / 2 ^ 110) :=
          mul_le_mul hcabs hdh (abs_nonneg _) (by positivity)
      _ = _ := by ring
  have hcdh1 : (1 + 31 / 10 * u32) * (1 + rD D) * (c0 * Λ) ≤ 14429 / 10000 * X := by
    have h1 : (1 + 31 / 10 * u32) * (1 + rD D) ≤ (1 + 31 / 10 * u32) * (1 + 1 / 10 ^ 4) :=
      mul_le_mul_of_nonneg_left hr1 (by positivity)
    calc (1 + 31 / 10 * u32) * (1 + rD D) * (c0 * Λ) ≤ (1 + 31 / 10 * u32) * (1 + 1 / 10 ^ 4) * (c0 * Λ) :=
          mul_le_mul_of_nonneg_right h1 hP0
      _ ≤ (1 + 31 / 10 * u32) * (1 + 1 / 10 ^ 4) * (14427 / 10000 * X) :=
          mul_le_mul_of_nonneg_left hP (by positivity)
      _ ≤ 14429 / 10000 * X := by
          have hk : (1 + 31 / 10 * u32) * (1 + 1 / 10 ^ 4) * (14427 / 10000) ≤ 14429 / 10000 := by
            rw [hu']; norm_num
          have := mul_le_mul_of_nonneg_right hk hX.1
          linarith
  have hcdh2 : (1 + 31 / 10 * u32) * (c0 * (1 / 2 ^ 110)) ≤ 1 / 2 ^ 44 := by
    have : c0 * (1 / 2 ^ 110) ≤ 2 ^ 65 * (1 / 2 ^ 110) := mul_le_mul_of_nonneg_right hc0big (by positivity)
    rw [hu']; nlinarith
  have hBc : |c * dh| ≤ 14429 / 10000 * X + 1 / 2 ^ 44 := by linarith
  -- the score part
  have hscore : |c * dh - c0 * sc| ≤ 31 / 10 * u32 * (1 + rD D) * (c0 * Λ) + rD D * (c0 * Λ) + 2 * (c0 * (1 / 2 ^ 110)) := by
    have e : c * dh - c0 * sc = (c - c0) * dh + c0 * (dh - sc) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_mul, abs_of_pos hc00]
    have t1 : |c - c0| * |dh| ≤ (31 / 10 * u32 * c0) * (Λ * (1 + rD D) + 1 / 2 ^ 110) :=
      mul_le_mul hc hdh (abs_nonneg _) (by positivity)
    have t2 : c0 * |dh - sc| ≤ c0 * (rD D * Λ + 1 / 2 ^ 110) := mul_le_mul_of_nonneg_left (hd.trans (by linarith)) hc00.le
    have t3 : 31 / 10 * u32 * c0 * (1 / 2 ^ 110) ≤ c0 * (1 / 2 ^ 110) := by
      rw [hu']; nlinarith [hc00]
    nlinarith
  have hA : 31 / 10 * u32 * (1 + rD D) * (c0 * Λ) ≤ 31 / 10 * u32 * (1 + 1 / 10 ^ 4) * (14427 / 10000 * X) :=
    mul_le_mul (mul_le_mul_of_nonneg_left hr1 (by positivity)) hP hP0 (by positivity)
  have hB : rD D * (c0 * Λ) ≤ rD D * (14427 / 10000 * X) := mul_le_mul_of_nonneg_left hP hrD0
  have hC : 2 * (c0 * (1 / 2 ^ 110)) ≤ 1 / 2 ^ 43 := by
    have : c0 * (1 / 2 ^ 110) ≤ 2 ^ 65 * (1 / 2 ^ 110) := mul_le_mul_of_nonneg_right hc0big (by positivity)
    linarith [show (2 : ℝ) * (2 ^ 65 * (1 / 2 ^ 110)) = 1 / 2 ^ 44 by norm_num]
  have hA : 31 / 10 * u32 * (1 + rD D) * (c0 * Λ) ≤ 31 / 10 * u32 * (1 + 1 / 10 ^ 4) * (14427 / 10000 * X) :=
    mul_le_mul (mul_le_mul_of_nonneg_left hr1 (by positivity)) hP hP0 (by positivity)
  have hB : rD D * (c0 * Λ) ≤ rD D * (14427 / 10000 * X) := mul_le_mul_of_nonneg_left hP hrD0
  have hC : 2 * (c0 * (1 / 2 ^ 110)) ≤ 1 / 2 ^ 43 := by
    have : c0 * (1 / 2 ^ 110) ≤ 2 ^ 65 * (1 / 2 ^ 110) := mul_le_mul_of_nonneg_right hc0big (by positivity)
    linarith [show (2 : ℝ) * (2 ^ 65 * (1 / 2 ^ 110)) = 1 / 2 ^ 44 by norm_num]
  have hS : σ * sc / Real.log 2 = c0 * sc := by rw [hc0def]; ring
  refine ⟨hBc, ?_⟩
  rw [hS]
  refine hscore.trans ?_
  have hX5 : 31 / 10 * u32 * (1 + 1 / 10 ^ 4) * (14427 / 10000 * X) ≤ 5 * u32 * X := by
    rw [hu']; nlinarith [hX.1]
  have hrr : rD D * (14427 / 10000 * X) = 14427 / 10000 * rD D * X := by ring
  linarith

set_option maxHeartbeats 1600000 in
/-- **The rounding part of a lane's exponent**: given the score part, the product `c·d̂` and the
subtraction of the reference `m1` (each rounded once) leave `y` within `xi D X` of `σ·score/ln 2 − m1`. -/
theorem expo_round {σ sc dh c m1 prod y X : ℝ} (D : ℕ) (hX : 0 ≤ X ∧ X ≤ 1024) (hrD0 : 0 ≤ rD D)
    (hBc : |c * dh| ≤ 14429 / 10000 * X + 1 / 2 ^ 44)
    (hscore : |c * dh - σ * sc / Real.log 2| ≤ 14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43)
    (hm1 : |m1| ≤ Bm X) (hp : |prod - c * dh| ≤ u32 * |c * dh| + 1 / 2 ^ 150)
    (hy : |y - (prod - m1)| ≤ 3 * u32 * (|prod| + |m1|) + 4 / 2 ^ 150) :
    |y - (σ * sc / Real.log 2 - m1)| ≤ xi D X := by
  have hu : u32 = 1 / 2 ^ 24 := rfl
  have hu0 : 0 < u32 := by rw [hu]; positivity
  have hcdh' : |c * dh| ≤ 14429 / 10000 * X + 1 / 2 ^ 40 := hBc.trans (by norm_num)
  have hprod : |prod| ≤ |c * dh| * (1 + u32) + 1 / 2 ^ 150 := by
    have := abs_sub_abs_le_abs_sub prod (c * dh)
    nlinarith [abs_nonneg (c * dh)]
  have hBm : |m1| ≤ 1443 / 1000 * X + 1 / 2 ^ 40 := hm1
  have h3 : 3 * u32 * (|prod| + |m1|) ≤ 3 * u32 * ((14429 / 10000 * X + 1 / 2 ^ 40) * (1 + u32) + 1 / 2 ^ 150 +
      (1443 / 1000 * X + 1 / 2 ^ 40)) := by
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    have : |c * dh| * (1 + u32) ≤ (14429 / 10000 * X + 1 / 2 ^ 40) * (1 + u32) :=
      mul_le_mul_of_nonneg_right hcdh' (by positivity)
    linarith
  have h4 : u32 * |c * dh| ≤ u32 * (14429 / 10000 * X + 1 / 2 ^ 40) := mul_le_mul_of_nonneg_left hcdh' hu0.le
  have hround : |y - (c * dh - m1)| ≤ 3 * u32 * ((14429 / 10000 * X + 1 / 2 ^ 40) * (1 + u32) + 1 / 2 ^ 150 +
      (1443 / 1000 * X + 1 / 2 ^ 40)) + 4 / 2 ^ 150 + u32 * (14429 / 10000 * X + 1 / 2 ^ 40) + 1 / 2 ^ 150 := by
    have e : y - (c * dh - m1) = (y - (prod - m1)) + (prod - c * dh) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    linarith
  have e : y - (σ * sc / Real.log 2 - m1) = (y - (c * dh - m1)) + (c * dh - σ * sc / Real.log 2) := by ring
  rw [e]
  refine (abs_add_le _ _).trans ?_
  have hxi : xi D X = X * (14428 / 10000 * rD D + 1 / 2 ^ 20) + 1 / 2 ^ 40 := rfl
  rw [hxi]
  have hrX := mul_nonneg hrD0 hX.1
  rw [hu] at hround hscore
  nlinarith [hround, hscore, hrX, hX.1]

/-- A lane's weight is at most `2^ξ`: `σ·score/ln 2 − m1 ≤ ξ` when `c·d̂ − m1` is at most the
rounding of the largest. -/
theorem expo_up {σ sc dh c m1 X : ℝ} (D : ℕ) (hX : 0 ≤ X ∧ X ≤ 1024) (hrD0 : 0 ≤ rD D)
    (_hBc : |c * dh| ≤ 14429 / 10000 * X + 1 / 2 ^ 44)
    (hscore : |c * dh - σ * sc / Real.log 2| ≤ 14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43)
    (hmax : c * dh - m1 ≤ u32 * (14429 / 10000 * X + 1 / 2 ^ 44) + 1 / 2 ^ 150) :
    σ * sc / Real.log 2 - m1 ≤ xi D X := by
  have hxi : xi D X = X * (14428 / 10000 * rD D + 1 / 2 ^ 20) + 1 / 2 ^ 40 := rfl
  rw [hxi]
  have hrX := mul_nonneg hrD0 hX.1
  have hu : u32 = 1 / 2 ^ 24 := rfl
  rw [hu] at hmax hscore
  have := (abs_le.mp hscore).1
  nlinarith [hrX, hX.1]

set_option maxHeartbeats 800000 in
/-- **The rescaling factor** `α̂ = exp2(m0 − m1)` for references `m0 ≤ m1` of magnitude at most
`1500`: finite, within `ρ·α + 2^-125` of `α = 2^(m0−m1)`, with `ρ ≤ Qα + CA·(m1 − m0)`. -/
theorem alpha_real {P : Platform} (hP : Device P) {m0 m1 s a : F32} (hm0 : m0.isFinite = true)
    (hm1 : m1.isFinite = true) (hle : m0.toReal ≤ m1.toReal) (h0 : |m0.toReal| ≤ 1500)
    (h1 : |m1.toReal| ≤ 1500) (hs : FloatV.NaNEq (FloatV.sub m0 m1) s)
    (ha : P.math .exp2 .binary32 s a) :
    a.isFinite = true ∧ ∃ ρ : ℝ, 0 ≤ ρ ∧ ρ ≤ Qα + CA * (m1.toReal - m0.toReal) ∧
      |a.toReal - Real.exp ((m0.toReal - m1.toReal) * Real.log 2)| ≤
        ρ * Real.exp ((m0.toReal - m1.toReal) * Real.log 2) + 1 / 2 ^ 125 := by
  have hlog0 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hlog1 : Real.log 2 < 7 / 10 := by linarith [Real.log_two_lt_d9]
  obtain ⟨hsf, hse⟩ := sub_real hs hm0 hm1 (by
    have := abs_sub (m0.toReal) (m1.toReal)
    have h3 : (3000 : ℝ) ≤ 2 ^ 100 := by norm_num
    linarith)
  set Δ := m0.toReal - m1.toReal with hΔ
  have hΔ0 : Δ ≤ 0 := by linarith
  have hΔb : -Δ ≤ 3000 := by
    have := abs_le.mp h0; have := abs_le.mp h1; linarith
  have hs_le : s.toReal ≤ 100 := by
    have := (abs_le.mp hse).2
    have hu : u32 = 1 / 2 ^ 24 := rfl
    rw [abs_of_nonpos hΔ0, hu] at this
    norm_num at this ⊢
    linarith
  obtain ⟨haf, hae⟩ := exp2_real hP ha hsf hs_le
  refine ⟨haf, ?_⟩
  -- the exponent error
  set δ := u32 * (-Δ) + 1 / 2 ^ 150 with hδ
  have hsδ : |s.toReal - Δ| ≤ δ := by
    have := hse; rw [abs_of_nonpos hΔ0] at this; linarith
  have hδ0 : 0 ≤ δ := by rw [hδ]; have : (0:ℝ) ≤ u32 := by unfold u32; norm_num
                         nlinarith
  set δ' := Real.log 2 * δ with hδ'
  have hδ'0 : 0 ≤ δ' := by positivity
  have hδ'1 : δ' ≤ 1 / 2 ^ 11 := by
    have hu : u32 = 1 / 2 ^ 24 := rfl
    have : δ ≤ 1 / 2 ^ 24 * 3000 + 1 / 2 ^ 150 := by
      rw [hδ, hu]; nlinarith
    calc δ' = Real.log 2 * δ := rfl
      _ ≤ 7 / 10 * (1 / 2 ^ 24 * 3000 + 1 / 2 ^ 150) := mul_le_mul hlog1.le this hδ0 (by norm_num)
      _ ≤ 1 / 2 ^ 11 := by norm_num
  set α := Real.exp (Δ * Real.log 2) with hα
  have hαpos : 0 < α := Real.exp_pos _
  have h2s : (2 : ℝ) ^ s.toReal = Real.exp (s.toReal * Real.log 2) := Softmax.two_rpow_eq_exp _
  have hexp : |Real.exp (s.toReal * Real.log 2) - α| ≤ δ' * (1 + δ') * α := by
    have := Softmax.abs_exp_sub_exp_le_mul (x := Δ * Real.log 2) (y := s.toReal * Real.log 2) (δ := δ') (by
      rw [← sub_mul, abs_mul, abs_of_pos hlog0, mul_comm]
      exact mul_le_mul_of_nonneg_left hsδ hlog0.le) (by linarith)
    exact this
  rw [h2s] at hae
  set ρ := q22 * (1 + δ' * (1 + δ')) + δ' * (1 + δ') with hρ
  have hq0 : 0 ≤ q22 := by unfold q22; norm_num
  refine ⟨ρ, by rw [hρ]; positivity, ?_, ?_⟩
  · have hu : u32 = 1 / 2 ^ 24 := rfl
    have hCA : CA = 6936 / 10000 * u32 := rfl
    have hQ : Qα = q22 + 1 / 2 ^ 100 := rfl
    have hq1 : q22 = 1 / 2 ^ 22 := rfl
    have hfac : δ' * (1 + δ') * (1 + q22) ≤ δ' * (10005 / 10000) := by
      have h1 : (1 + δ') * (1 + q22) ≤ 10005 / 10000 := by
        rw [hq1]; nlinarith [hδ'0, hδ'1]
      calc δ' * (1 + δ') * (1 + q22) = δ' * ((1 + δ') * (1 + q22)) := by ring
        _ ≤ δ' * (10005 / 10000) := mul_le_mul_of_nonneg_left h1 hδ'0
    have hρ' : ρ ≤ q22 + δ' * (10005 / 10000) := by
      rw [hρ]; nlinarith [hfac]
    have hδ'' : δ' * (10005 / 10000) ≤ 6936 / 10000 * u32 * (-Δ) + 1 / 2 ^ 100 := by
      rw [hδ', hδ]
      have h2 : Real.log 2 * (10005 / 10000) ≤ 6936 / 10000 := by linarith [Real.log_two_lt_d9]
      have h3 : Real.log 2 * (10005 / 10000) * (u32 * (-Δ)) ≤ 6936 / 10000 * (u32 * (-Δ)) :=
        mul_le_mul_of_nonneg_right h2 (by rw [hu]; nlinarith)
      have h4 : Real.log 2 * (10005 / 10000) * (1 / 2 ^ 150) ≤ 1 / 2 ^ 100 := by
        have : Real.log 2 * (10005 / 10000) ≤ 1 := by linarith [Real.log_two_lt_d9]
        calc Real.log 2 * (10005 / 10000) * (1 / 2 ^ 150) ≤ 1 * (1 / 2 ^ 150) :=
              mul_le_mul_of_nonneg_right this (by positivity)
          _ ≤ 1 / 2 ^ 100 := by norm_num
      have e : Real.log 2 * (u32 * (-Δ) + 1 / 2 ^ 150) * (10005 / 10000) =
          Real.log 2 * (10005 / 10000) * (u32 * (-Δ)) + Real.log 2 * (10005 / 10000) * (1 / 2 ^ 150) := by ring
      rw [e]
      have e2 : 6936 / 10000 * u32 * (-Δ) = 6936 / 10000 * (u32 * (-Δ)) := by ring
      rw [e2]
      linarith
    rw [hQ, hCA]
    have : m1.toReal - m0.toReal = -Δ := by rw [hΔ]; ring
    rw [this]
    linarith
  · have h2s_le : Real.exp (s.toReal * Real.log 2) ≤ α + δ' * (1 + δ') * α := by
      have := (abs_le.mp hexp).2
      linarith
    have hq : q22 * Real.exp (s.toReal * Real.log 2) ≤ q22 * (α + δ' * (1 + δ') * α) :=
      mul_le_mul_of_nonneg_left h2s_le hq0
    have : |a.toReal - α| ≤ |a.toReal - Real.exp (s.toReal * Real.log 2)| + |Real.exp (s.toReal * Real.log 2) - α| := by
      have := abs_add_le (a.toReal - Real.exp (s.toReal * Real.log 2)) (Real.exp (s.toReal * Real.log 2) - α)
      rwa [sub_add_sub_cancel] at this
    have e : ρ * α = q22 * (α + δ' * (1 + δ') * α) + δ' * (1 + δ') * α := by rw [hρ]; ring
    have h126 : (1 : ℝ) / 2 ^ 126 ≤ 1 / 2 ^ 125 := by norm_num
    linarith

/-- **A weight**: `p̂ = exp2(y)` for an exponent `y` within `ξ` of `S − m1` (and `S − m1 ≤ ξ`):
finite, within `rhoP·pw + 2^-125` of the exact weight `pw = 2^(S − m1)`, and at most `2`. -/
theorem p_real {P : Platform} (hP : Device P) {y p : F32} (hy : y.isFinite = true)
    (hp : P.math .exp2 .binary32 y p) {S m1 ξ : ℝ} (hξ0 : 0 ≤ ξ) (hξ1 : ξ ≤ 1 / 10)
    (hyS : |y.toReal - (S - m1)| ≤ ξ) (hSm : S - m1 ≤ ξ) :
    p.isFinite = true ∧
      |p.toReal - Real.exp ((S - m1) * Real.log 2)| ≤
        (ξ * (1 + ξ) + q22 * (1 + ξ * (1 + ξ))) * Real.exp ((S - m1) * Real.log 2) + 1 / 2 ^ 125 ∧
      |p.toReal| ≤ 2 := by
  have hlog0 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hlog1 : Real.log 2 < 7 / 10 := by linarith [Real.log_two_lt_d9]
  have hy_le : y.toReal ≤ 100 := by
    have := (abs_le.mp hyS).2; linarith
  obtain ⟨hpf, hpe⟩ := exp2_real hP hp hy hy_le
  set pw := Real.exp ((S - m1) * Real.log 2) with hpw
  have hpw0 : 0 < pw := Real.exp_pos _
  have h2s : (2 : ℝ) ^ y.toReal = Real.exp (y.toReal * Real.log 2) := Softmax.two_rpow_eq_exp _
  set δ' := ξ * Real.log 2 with hδ'
  have hδ'0 : 0 ≤ δ' := by positivity
  have hδ'ξ : δ' ≤ ξ := by nlinarith
  have hexp : |Real.exp (y.toReal * Real.log 2) - pw| ≤ δ' * (1 + δ') * pw := by
    refine Softmax.abs_exp_sub_exp_le_mul (x := (S - m1) * Real.log 2) (y := y.toReal * Real.log 2) (δ := δ') ?_ (by linarith)
    rw [← sub_mul, abs_mul, abs_of_pos hlog0, mul_comm]
    exact (mul_le_mul_of_nonneg_left hyS hlog0.le).trans_eq (by rw [hδ', mul_comm])
  rw [h2s] at hpe
  have hq0 : 0 ≤ q22 := by unfold q22; norm_num
  have hmono : δ' * (1 + δ') ≤ ξ * (1 + ξ) := by nlinarith
  have hexp_le : Real.exp (y.toReal * Real.log 2) ≤ pw + δ' * (1 + δ') * pw := by
    have := (abs_le.mp hexp).2; linarith
  have hq : q22 * Real.exp (y.toReal * Real.log 2) ≤ q22 * (pw + δ' * (1 + δ') * pw) :=
    mul_le_mul_of_nonneg_left hexp_le hq0
  have htri : |p.toReal - pw| ≤ |p.toReal - Real.exp (y.toReal * Real.log 2)| +
      |Real.exp (y.toReal * Real.log 2) - pw| := by
    have := abs_add_le (p.toReal - Real.exp (y.toReal * Real.log 2)) (Real.exp (y.toReal * Real.log 2) - pw)
    rwa [sub_add_sub_cancel] at this
  have h126 : (1 : ℝ) / 2 ^ 126 ≤ 1 / 2 ^ 125 := by norm_num
  have hA : δ' * (1 + δ') * pw ≤ ξ * (1 + ξ) * pw := mul_le_mul_of_nonneg_right hmono hpw0.le
  have hB : q22 * (δ' * (1 + δ') * pw) ≤ q22 * (ξ * (1 + ξ) * pw) := mul_le_mul_of_nonneg_left hA hq0
  have hpe' : |p.toReal - pw| ≤ (ξ * (1 + ξ) + q22 * (1 + ξ * (1 + ξ))) * pw + 1 / 2 ^ 125 := by
    have e : (ξ * (1 + ξ) + q22 * (1 + ξ * (1 + ξ))) * pw =
        q22 * pw + q22 * (ξ * (1 + ξ) * pw) + ξ * (1 + ξ) * pw := by ring
    have e2 : q22 * (pw + δ' * (1 + δ') * pw) = q22 * pw + q22 * (δ' * (1 + δ') * pw) := by ring
    linarith [htri, hpe, hexp, hq, hA, hB, h126]
  refine ⟨hpf, hpe', ?_⟩
  have hpw1 : pw ≤ 3 / 2 := by
    have h1 : (S - m1) * Real.log 2 ≤ ξ * Real.log 2 := mul_le_mul_of_nonneg_right hSm hlog0.le
    have h2 : Real.exp ((S - m1) * Real.log 2) ≤ Real.exp (ξ * Real.log 2) := Real.exp_le_exp.mpr h1
    have h3 := Softmax.exp_le_one_add_two_mul (d := ξ * Real.log 2) (by positivity) (by nlinarith)
    nlinarith
  have habs := abs_sub_abs_le_abs_sub p.toReal pw
  rw [abs_of_pos hpw0] at habs
  have hr : ξ * (1 + ξ) + q22 * (1 + ξ * (1 + ξ)) ≤ 1 / 5 := by
    have : q22 = 1 / 2 ^ 22 := rfl
    rw [this]; nlinarith
  have hrp := mul_le_mul_of_nonneg_right hr hpw0.le
  nlinarith

/-- The reference is within `ξ` of the score of the lane it comes from. -/
theorem expo_down {σ sc dh c m1 X : ℝ} (D : ℕ) (hX : 0 ≤ X ∧ X ≤ 1024) (hrD0 : 0 ≤ rD D)
    (hscore : |c * dh - σ * sc / Real.log 2| ≤ 14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43)
    (hmax : m1 - c * dh ≤ u32 * (14429 / 10000 * X + 1 / 2 ^ 44) + 1 / 2 ^ 150) :
    m1 - xi D X ≤ σ * sc / Real.log 2 := by
  have hxi : xi D X = X * (14428 / 10000 * rD D + 1 / 2 ^ 20) + 1 / 2 ^ 40 := rfl
  rw [hxi]
  have hrX := mul_nonneg hrD0 hX.1
  have hu : u32 = 1 / 2 ^ 24 := rfl
  rw [hu] at hmax hscore
  have := (abs_le.mp hscore).2
  nlinarith [hrX, hX.1]

/-- **The reference and the exponents of a trip** (unmasked, one row). -/
theorem trip_expo {fusion : Bool} {D : ℕ} (hD : D = 64 ∨ D = 128 ∨ D = 256) {X σ : ℝ}
    (hX : 0 ≤ X ∧ X ≤ 1024) (hσ : 1 / 2 ^ 64 ≤ σ ∧ σ ≤ 2 ^ 64)
    {qkr prodr yr : Fin 64 → F32} {sc Λ : Fin 64 → ℝ} {c mx mc m0 m1 : F32}
    (hqk : ∀ j, (qkr j).isFinite = true ∧
      |(qkr j).toReal - sc j| ≤ rD D * Λ j + ((D : ℝ) + 2) / 2 ^ 126)
    (hΛ0 : ∀ j, 0 ≤ Λ j) (hΛ : ∀ j, σ * Λ j ≤ X) (hsc : ∀ j, |sc j| ≤ Λ j)
    (hcf : c.isFinite = true) (hcpos : 0 < c.toReal)
    (hc : |c.toReal - σ / Real.log 2| ≤ 31 / 10 * u32 * (σ / Real.log 2))
    (hmx : ∃ t : Tree, t.Covers 64 ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < 64 then qkr ⟨j, h⟩ else default) mx)
    (hmc : FloatV.NaNEq (FloatV.mul mx c) mc) (hm1 : FloatV.MaxRel true m0 mc m1)
    (hm0 : m0 = FloatV.negInf ∨ (m0.isFinite = true ∧ |m0.toReal| ≤ Bm X))
    (hprod : ∀ j, FloatV.NaNEq (FloatV.mul (qkr j) c) (prodr j))
    (hy : ∀ j, AddRel fusion (prodr j) (FloatV.neg m1) (yr j)) :
    m1.isFinite = true ∧ |m1.toReal| ≤ Bm X ∧ (m0.isFinite = true → m0.toReal ≤ m1.toReal) ∧
      (∀ j, (yr j).isFinite = true ∧ |(yr j).toReal - (σ * sc j / Real.log 2 - m1.toReal)| ≤ xi D X ∧
        σ * sc j / Real.log 2 - m1.toReal ≤ xi D X) ∧
      ((m0.isFinite = true ∧ m1.toReal = m0.toReal) ∨
        ∃ j, m1.toReal - xi D X ≤ σ * sc j / Real.log 2) := by
  have hnanq : ∀ i, i < 64 → (if h : i < 64 then qkr ⟨i, h⟩ else (default : F32)).isNaN = false := by
    intro i hi
    simp only [dite_eq_left hi]
    exact (FloatV.isFinite_iff.mp (hqk ⟨i, hi⟩).1).1
  -- the lane bounds
  have hsco : ∀ j, |(qkr j).toReal * c.toReal| ≤ 14429 / 10000 * X + 1 / 2 ^ 44 ∧
      |c.toReal * (qkr j).toReal - σ * sc j / Real.log 2| ≤ 14427 / 10000 * rD D * X + 5 * u32 * X + 1 / 2 ^ 43 := by
    intro j
    obtain ⟨h1, h2⟩ := expo_score D hD hX hσ (hΛ0 j) (hΛ j) (hsc j) (hqk j).2 hc
    exact ⟨by rw [mul_comm]; exact h1, h2⟩
  have hrD0 : 0 ≤ rD D := by unfold rD; positivity
  -- the maximum of the row
  obtain ⟨t, hcov, hev⟩ := hmx
  have hlv : ∀ i, i ∈ t.leaves ↔ i < 64 := fun i => by rw [hcov.mem_iff, List.mem_range]
  obtain ⟨⟨i0, hi0, hmxeq⟩, hle⟩ := Tree.eval_maxRel_toReal hev (fun i hi => hnanq i ((hlv i).mp hi))
  have hi0' : i0 < 64 := (hlv i0).mp hi0
  set j0 : Fin 64 := ⟨i0, hi0'⟩ with hj0
  have hmxj : mx = qkr j0 := by rw [hmxeq]; simp [hj0, hi0']
  have hmxf : mx.isFinite = true := by rw [hmxj]; exact (hqk j0).1
  have hqle : ∀ j, (qkr j).toReal ≤ mx.toReal := by
    intro j
    have := hle j.1 ((hlv j.1).mpr j.2)
    simpa [dite_eq_left j.2] using this
  have hmxc : |mx.toReal * c.toReal| ≤ 14429 / 10000 * X + 1 / 2 ^ 44 := by rw [hmxj]; exact (hsco j0).1
  have hBcle : (14429 / 10000 * X + 1 / 2 ^ 44 : ℝ) ≤ 2 ^ 100 := by nlinarith [hX.2]
  obtain ⟨hmcf, hmce⟩ := mul_real hmc hmxf hcf (hmxc.trans hBcle)
  have hu : u32 = 1 / 2 ^ 24 := rfl
  have hmcabs : |mc.toReal| ≤ 1443 / 1000 * X + 1 / 2 ^ 40 := by
    have h1 := abs_sub_abs_le_abs_sub mc.toReal (mx.toReal * c.toReal)
    have h2 : u32 * |mx.toReal * c.toReal| ≤ u32 * (14429 / 10000 * X + 1 / 2 ^ 44) :=
      mul_le_mul_of_nonneg_left hmxc (by rw [hu]; norm_num)
    rw [hu] at h2 hmce
    nlinarith [hX.1]
  -- c * mx is at most mc plus a rounding
  have hcmx : c.toReal * mx.toReal ≤ mc.toReal + (u32 * (14429 / 10000 * X + 1 / 2 ^ 44) + 1 / 2 ^ 150) := by
    have h2 : u32 * |mx.toReal * c.toReal| ≤ u32 * (14429 / 10000 * X + 1 / 2 ^ 44) :=
      mul_le_mul_of_nonneg_left hmxc (by rw [hu]; norm_num)
    have := (abs_le.mp hmce).1
    rw [mul_comm c.toReal mx.toReal]
    linarith
  have hmcnan : mc.isNaN = false := (FloatV.isFinite_iff.mp hmcf).1
  -- the new reference
  have hcases : (m1 = mc ∧ mc.toReal ≤ m1.toReal) ∨
      (m0.isFinite = true ∧ m1.toReal = m0.toReal ∧ mc.toReal ≤ m1.toReal) := by
    rcases hm0 with h0 | ⟨h0f, -⟩
    · left
      have : m1 = mc := FloatV.MaxRel.negInf_left binary32_reg (Format.Std.ieee Format.binary32_std)
        (h0 ▸ hm1) hmcnan
      exact ⟨this, by rw [this]⟩
    · have hm0nan : m0.isNaN = false := (FloatV.isFinite_iff.mp h0f).1
      have hmax := hm1.toReal_eq_max hm0nan hmcnan
      rcases hm1.mem hm0nan hmcnan with h | h
      · right; exact ⟨h0f, by rw [h], by rw [hmax]; exact le_max_right _ _⟩
      · left; exact ⟨h, by rw [hmax]; exact le_max_right _ _⟩
  have hm1f : m1.isFinite = true := by
    rcases hcases with ⟨h, -⟩ | ⟨h0f, h, -⟩
    · rw [h]; exact hmcf
    · have hm0nan : m0.isNaN = false := (FloatV.isFinite_iff.mp h0f).1
      rcases hm1.mem hm0nan hmcnan with h' | h'
      · rw [h']; exact h0f
      · rw [h']; exact hmcf
  have hm1B : |m1.toReal| ≤ Bm X := by
    rcases hcases with ⟨h, -⟩ | ⟨h0f, h, -⟩
    · rw [h]; exact hmcabs
    · rw [h]; rcases hm0 with h0 | ⟨-, h0b⟩
      · exfalso
        rw [h0] at h0f
        have hninf : (FloatV.negInf : F32).isFinite = false := by decide +kernel
        rw [hninf] at h0f; exact absurd h0f (by simp)
      · exact h0b
  have hmono : m0.isFinite = true → m0.toReal ≤ m1.toReal := by
    intro h0f
    have hm0nan : m0.isNaN = false := (FloatV.isFinite_iff.mp h0f).1
    have hmax := hm1.toReal_eq_max hm0nan hmcnan
    rw [hmax]; exact le_max_left _ _
  have hge : mc.toReal ≤ m1.toReal := by rcases hcases with ⟨-, h⟩ | ⟨-, -, h⟩ <;> exact h
  have hBm0 : 0 ≤ Bm X := by unfold Bm; have := hX.1; positivity
  refine ⟨hm1f, hm1B, hmono, fun j => ?_, ?_⟩
  · obtain ⟨hs1, hs2⟩ := hsco j
    have hcd : |c.toReal * (qkr j).toReal| ≤ 14429 / 10000 * X + 1 / 2 ^ 44 := by rw [mul_comm]; exact hs1
    obtain ⟨hpf, hpe⟩ := mul_real (hprod j) (hqk j).1 hcf (hs1.trans hBcle)
    have hpabs : |(prodr j).toReal| ≤ 14429 / 10000 * X + 1 / 2 ^ 44 + 1 := by
      have h1 := abs_sub_abs_le_abs_sub (prodr j).toReal ((qkr j).toReal * c.toReal)
      have h2 : u32 * |(qkr j).toReal * c.toReal| ≤ u32 * (14429 / 10000 * X + 1 / 2 ^ 44) :=
        mul_le_mul_of_nonneg_left hs1 (by rw [hu]; norm_num)
      rw [hu] at h2 hpe
      nlinarith [hX.1]
    obtain ⟨hyf, hye⟩ := addrel_real (hy j) hpf (by rw [FloatV.isFinite_neg]; exact hm1f) (by
      rw [FloatV.toReal_neg, abs_neg]
      have : (1443 / 1000 * X + 1 / 2 ^ 40 : ℝ) ≤ 2 ^ 100 := by nlinarith [hX.2]
      have := hm1B; unfold Bm at this
      nlinarith [hX.2])
    rw [FloatV.toReal_neg] at hye
    refine ⟨hyf, ?_, ?_⟩
    · refine expo_round D hX hrD0 (c := c.toReal) (dh := (qkr j).toReal) (prod := (prodr j).toReal)
        (y := (yr j).toReal) hcd hs2 hm1B (by rw [mul_comm]; simpa [mul_comm] using hpe) ?_
      simpa [sub_eq_add_neg] using hye
    · refine expo_up D hX hrD0 hcd hs2 ?_
      have h1 : c.toReal * (qkr j).toReal ≤ c.toReal * mx.toReal :=
        mul_le_mul_of_nonneg_left (hqle j) hcpos.le
      linarith
  · rcases hcases with ⟨hEq, hge'⟩ | ⟨h0f, h, -⟩
    · right
      refine ⟨j0, ?_⟩
      obtain ⟨hs1, hs2⟩ := hsco j0
      have hmc_le : mc.toReal ≤ mx.toReal * c.toReal + (u32 * (14429 / 10000 * X + 1 / 2 ^ 44) + 1 / 2 ^ 150) := by
        have h2 : u32 * |mx.toReal * c.toReal| ≤ u32 * (14429 / 10000 * X + 1 / 2 ^ 44) :=
          mul_le_mul_of_nonneg_left hmxc (by rw [hu]; norm_num)
        have := (abs_le.mp hmce).2
        linarith
      refine expo_down D hX hrD0 hs2 ?_
      have : m1.toReal = mc.toReal := by rw [hEq]
      rw [this]
      have e : c.toReal * (qkr j0).toReal = mx.toReal * c.toReal := by rw [hmxj, mul_comm]
      rw [e]
      linarith
    · left; exact ⟨h0f, h⟩

end

end Claims.AttnFwd
