import Claim
import TVA.Proof
import Proof.Load

/-!
# One turn of the loop

Turn `t` of the kernel's main loop: it reads the next `BK`-wide slab of `a` and `b` (masked to
the rows, columns and `k`-lanes inside the chunk), accumulates it into the tile with the fused
`dot_add`, and advances the two pointer tiles by `BK·stride`. The turn leaves the invariant
`inv` true at `t + 1`: the loads only read, the pointer lanes move by exactly `BK` elements, and
the accumulator gains the slab's exact dot product, within the step's error bound.
-/

open TVA
namespace Claims.SsdBmm

set_option maxHeartbeats 800000

/-! ## Small helpers -/

/-- `DType.val?` of a `bfloat16`. -/
private theorem val?_of_bf16 {x : BF16} {v : ℚ} :
    DType.val? .bf16 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  show (if x.isFinite = true then some x.toRat else none) = some v ↔ _
  split_ifs with h
  · simp [h]
  · simp [h]

/-- The standard formats. -/
private theorem std_binary32 : Format.Std .binary32 := Or.inr (Or.inr (Or.inl rfl))

/-- `bfloat16` is a standard format. -/
private theorem std_bfloat16 : Format.Std .bfloat16 := Or.inr (Or.inl rfl)

/-- Every `float32` converts to `bfloat16`. -/
private theorem exists_conv_f32_bf16 {P : Platform} (x : F32) : ∃ r, ConvRel P .f32 .bf16 x r := by
  refine ⟨FloatV.cvt x, ?_⟩
  show FloatRel P .binary32 .bfloat16 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide)]
  exact Or.inl ⟨std_binary32, std_bfloat16, FloatV.NaNEq.refl _⟩

/-- The `k` mask bound of the model's step is `kMask`. -/
private theorem kMask_eq (d : Inst) (t : Nat) :
    (IntV.ofNat d.K -
        IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : ℤ) * (IntV.ofInt 1 : I32).toInt) *
          IntV.ofNat BK : I32) = kMask d t := by
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := by decide
  have h1 : (IntV.ofInt 1 : I32).toInt = 1 := by decide
  rw [h0, h1]
  have hz : (0 : ℤ) + (t : ℤ) * 1 = (t : ℤ) := by ring
  rw [hz]
  rfl

/-- The mask of the model's `a` load is `maskA`. -/
private theorem maskA_eq (d : Inst) (i : Fin (launch d).size) (t : Nat) :
    ((((conv64 (offsM d i)).col BK).ltS
          (tl.iminimum (Tensor.scalar (IntV.ofInt (d.C : ℤ)))
              (Tensor.scalar
                (IntV.ofInt (d.T : ℤ) -
                  (Tensor.scalar (IntV.ofInt (↑(↑i / gx d / d.G) : ℤ))).item *
                    IntV.ofInt (d.C : ℤ)))).item).and
      (((tl.arange 0 BK).row BM).ltS
        (IntV.ofNat d.K -
          IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : ℤ) * (IntV.ofInt 1 : I32).toInt) *
            IntV.ofNat BK))) = maskA d i t := by
  rw [kMask_eq d t]
  rfl

/-- The mask of the model's `b` load is `maskB`. -/
private theorem maskB_eq (d : Inst) (i : Fin (launch d).size) (t : Nat) :
    ((((tl.arange 0 BK).col BN).ltS
          (IntV.ofNat d.K -
            IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : ℤ) * (IntV.ofInt 1 : I32).toInt) *
              IntV.ofNat BK)).and
      ((((conv64 (offsN d i)).row BK).ltS
        (tl.iminimum (Tensor.scalar (IntV.ofInt (d.C : ℤ)))
            (Tensor.scalar
              (IntV.ofInt (d.T : ℤ) -
                (Tensor.scalar (IntV.ofInt (↑(↑i / gx d / d.G) : ℤ))).item *
                  IntV.ofInt (d.C : ℤ)))).item))) = maskB d i t := by
  rw [kMask_eq d t]
  rfl

/-- Sign-extending a column-broadcast vector is broadcasting the sign-extended vector. -/
private theorem map_col_map {m : Nat} (x : Tensor [m] I32) (n : Nat) :
    ((x.col n).map (fun v => IntV.ofInt v.toInt)) = (conv64 x).col n := rfl

/-- Sign-extending a row-broadcast vector is broadcasting the sign-extended vector. -/
private theorem map_row_map {n : Nat} (x : Tensor [n] I32) (m : Nat) :
    ((x.row m).map (fun v => IntV.ofInt v.toInt)) = (conv64 x).row m := rfl

/-- The step `BLOCK_SIZE_K · stride_ak` of `a_ptrs` (line 162) has value `32 · aK`, without wrapping. -/
private theorem aStep_toInt {d : Inst} (hd : d.ok) :
    ((IntV.ofNat BK : I32) * IntV.ofNat d.aK).toInt = ((BK * d.aK : Nat) : ℤ) := by
  have hBKval : (IntV.ofNat BK : I32).toInt = (BK : ℤ) := by
    have h := IntV.I32.toInt_ofNat (n := BK) (by norm_num [BK])
    simpa using h
  have haKval : (IntV.ofNat d.aK : I32).toInt = (d.aK : ℤ) := IntV.I32.toInt_ofNat hd.aK_lt
  have hfits : IntV.SFits 32 ((BK : ℤ) * (d.aK : ℤ)) := by
    rw [← Nat.cast_mul]
    refine IntV.sfits_of_nonneg (Int.natCast_nonneg _) ?_
    have h2 : ((BK * d.aK : Nat) : ℤ) < (2 : ℤ) ^ 31 := by exact_mod_cast hd.aK_step
    have h4 : (2 : ℤ) ^ (32 - 1) = (2 : ℤ) ^ 31 := by norm_num
    rw [h4]; exact h2
  rw [IntV.toInt_mul hBKval haKval hfits]
  push_cast
  ring

/-- The step `BLOCK_SIZE_K · stride_bk` of `b_ptrs` (line 163) has value `32 · bK`, without wrapping. -/
private theorem bStep_toInt {d : Inst} (hd : d.ok) :
    ((IntV.ofNat BK : I32) * IntV.ofNat d.bK).toInt = ((BK * d.bK : Nat) : ℤ) := by
  have hBKval : (IntV.ofNat BK : I32).toInt = (BK : ℤ) := by
    have h := IntV.I32.toInt_ofNat (n := BK) (by norm_num [BK])
    simpa using h
  have hbKval : (IntV.ofNat d.bK : I32).toInt = (d.bK : ℤ) := IntV.I32.toInt_ofNat hd.bK_lt
  have hfits : IntV.SFits 32 ((BK : ℤ) * (d.bK : ℤ)) := by
    rw [← Nat.cast_mul]
    refine IntV.sfits_of_nonneg (Int.natCast_nonneg _) ?_
    have h2 : ((BK * d.bK : Nat) : ℤ) < (2 : ℤ) ^ 31 := by exact_mod_cast hd.bK_step
    have h4 : (2 : ℤ) ^ (32 - 1) = (2 : ℤ) ^ 31 := by norm_num
    rw [h4]; exact h2
  rw [IntV.toInt_mul hBKval hbKval hfits]
  push_cast
  ring

/-- Moving an `a` pointer by one step lands on the same lane one block later. -/
private theorem aPtr_step {d : Inst} (hd : d.ok) (r h kk t : Nat) :
    (d.a.ptr (aIdx d r h (t * BK + kk) : ℤ)).offset
        ((IntV.ofNat BK : I32) * IntV.ofNat d.aK).toInt
      = d.a.ptr (aIdx d r h ((t + 1) * BK + kk) : ℤ) := by
  rw [Buf.ptr_offset, aStep_toInt hd]
  congr 1
  unfold aIdx
  push_cast
  ring

/-- Moving a `b` pointer by one step lands on the same lane one block later. -/
private theorem bPtr_step {d : Inst} (hd : d.ok) (r h kk t : Nat) :
    (d.b.ptr (bIdx d r h (t * BK + kk) : ℤ)).offset
        ((IntV.ofNat BK : I32) * IntV.ofNat d.bK).toInt
      = d.b.ptr (bIdx d r h ((t + 1) * BK + kk) : ℤ) := by
  rw [Buf.ptr_offset, bStep_toInt hd]
  congr 1
  unfold bIdx
  push_cast
  ring

/-- A sum over `Fin n` is `dotSum`. -/
private theorem sum_fin_eq_dotSum (va vb : Nat → BF16) (n : Nat) :
    (∑ k : Fin n, (va ↑k).toRat * (vb ↑k).toRat) = dotSum va vb n := by
  unfold dotSum
  exact Fin.sum_univ_eq_sum_range (fun k => (va k).toRat * (vb k).toRat) n

/-- A sum over `Fin n` of lifted magnitudes is `dotMag`. -/
private theorem sum_fin_eq_dotMag (va vb : Nat → BF16) (n : Nat) :
    (∑ k : Fin n, Device.liftMag .bf16 (va ↑k).toRat *
        Device.liftMag .bf16 (vb ↑k).toRat) = dotMag va vb n := by
  unfold dotMag
  exact Fin.sum_univ_eq_sum_range
    (fun k => Device.liftMag .bf16 (va k).toRat * Device.liftMag .bf16 (vb k).toRat) n

/-! ## The step -/

/-- One turn of the loop, from the invariant at `t` to the invariant at `t + 1`. -/
theorem step_safe {P : Platform} {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (t : Nat)
    (ht : t < steps d.K) {st : StT} {m₁ init : Mem} (hinv : inv d init i t st m₁)
    (hpre : Pre d init) (hP : Device P) :
    Safe ((launch d).env P) (R d init i) ∅ (fun st' m₂ => inv d init i (t + 1) st' m₂) m₁
      (do
        let o1 ←
          tl.convert .i32 .i64
            (((Tensor.scalar (IntV.ofInt (↑(↑i % gx d / ((d.C + (BN - 1)) / BN)) : ℤ) : I32)).item *
                  (IntV.ofNat BM : I32) + tl.arange 0 BM).col BK)
        let z1 ← tl.convert .f32 .bf16 (tl.full [BM, BK] zf32)
        let a ←
          tl.load st.1
            (some
              ((o1.ltS
                    (tl.iminimum (Tensor.scalar (IntV.ofInt (d.C : ℤ)))
                        (Tensor.scalar
                          (IntV.ofInt (d.T : ℤ) -
                            (Tensor.scalar (IntV.ofInt (↑(↑i / gx d / d.G) : ℤ))).item *
                              IntV.ofInt (d.C : ℤ)))).item).and
                (((tl.arange 0 BK).row BM).ltS
                  (IntV.ofNat d.K -
                    IntV.ofInt ((IntV.ofInt 0 : I32).toInt + ↑t * (IntV.ofInt 1 : I32).toInt) *
                      IntV.ofNat BK))))
            (some z1)
        let o2 ←
          tl.convert .i32 .i64
            (((Tensor.scalar (IntV.ofInt (↑(↑i % gx d % ((d.C + (BN - 1)) / BN)) : ℤ) : I32)).item *
                  (IntV.ofNat BN : I32) + tl.arange 0 BN).row BK)
        let z2 ← tl.convert .f32 .bf16 (tl.full [BK, BN] zf32)
        let b ←
          tl.load st.2.1
            (some
              ((((tl.arange 0 BK).col BN).ltS
                    (IntV.ofNat d.K -
                      IntV.ofInt ((IntV.ofInt 0 : I32).toInt + ↑t * (IntV.ofInt 1 : I32).toInt) *
                        IntV.ofNat BK)).and
                (o2.ltS
                  (tl.iminimum (Tensor.scalar (IntV.ofInt (d.C : ℤ)))
                      (Tensor.scalar
                        (IntV.ofInt (d.T : ℤ) -
                          (Tensor.scalar (IntV.ofInt (↑(↑i / gx d / d.G) : ℤ))).item *
                            IntV.ofInt (d.C : ℤ)))).item)))
            (some z2)
        let acc ← tl.dot_add cfgBF16 a b st.2.2
        pure (st.1 + (bc ((IntV.ofNat BK : I32) * IntV.ofNat d.aK) : Tensor [BM, BK] I32),
          st.2.1 + (bc ((IntV.ofNat BK : I32) * IntV.ofNat d.bK) : Tensor [BK, BN] I32), acc)) := by
  obtain ⟨rfl, hptrA, hptrB, hnum⟩ := hinv
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro idx
    exact ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩
  · intro o1 ho1
    have ho1e : o1 = (conv64 (offsM d i)).col BK := by
      rw [← map_col_map]
      refine Tensor.ext (funext fun idx => ?_)
      exact (ConvRel.iff_cast_i32_i64.mp (ho1 idx)).trans (IntV.cast_eq_ofInt _)
    rw [ho1e, maskA_eq d i t]
    refine safe_convert (d := .f32) (d' := .bf16) ?_ ?_
    · intro idx
      simpa only [tl.full, Tensor.get_full, zf32_eq] using
        exists_conv_f32_bf16 (P := ((launch d).env P).P) (FloatV.ofPat 0 : F32)
    · intro z1 hz1
      have hz1e : ∀ idx : Idx [BM, BK], z1.get idx = (FloatV.ofPat 0 : BF16) := by
        intro idx
        refine conv_bf16_zero (P := ((launch d).env P).P) ?_
        simpa only [tl.full, Tensor.get_full, zf32_eq] using hz1 idx
      refine safe_load_buf d.a (aLoad_addrs hd i t ht ⟨rfl, hptrA, hptrB, hnum⟩)
        (fun idx hen => aLoad_lt hd i t idx hen) hpre.1 (aLoad_R hd i t) ?_
      intro ta hta htao
      refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
      · intro idx
        exact ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩
      · intro o2 ho2
        have ho2e : o2 = (conv64 (offsN d i)).row BK := by
          rw [← map_row_map]
          refine Tensor.ext (funext fun idx => ?_)
          exact (ConvRel.iff_cast_i32_i64.mp (ho2 idx)).trans (IntV.cast_eq_ofInt _)
        rw [ho2e, maskB_eq d i t]
        refine safe_convert (d := .f32) (d' := .bf16) ?_ ?_
        · intro idx
          simpa only [tl.full, Tensor.get_full, zf32_eq] using
            exists_conv_f32_bf16 (P := ((launch d).env P).P) (FloatV.ofPat 0 : F32)
        · intro z2 hz2
          have hz2e : ∀ idx : Idx [BK, BN], z2.get idx = (FloatV.ofPat 0 : BF16) := by
            intro idx
            refine conv_bf16_zero (P := ((launch d).env P).P) ?_
            simpa only [tl.full, Tensor.get_full, zf32_eq] using hz2 idx
          refine safe_load_buf d.b (bLoad_addrs hd i t ht ⟨rfl, hptrA, hptrB, hnum⟩)
            (fun idx hen => bLoad_lt hd i t idx hen) hpre.2.1 (bLoad_R hd i t) ?_
          intro tb htb htbo
          refine safe_dot_add (cfg := cfgBF16) (M := BM) (N := BN) (K := BK) ?_ ?_
          · exact hP.dotTotal cfgBF16 ta tb st.2.2
          · intro acc hacc
            refine Safe.ret ?_
            refine ⟨rfl, ?_, ?_, ?_⟩
            · intro mll kk hC
              show (st.1 + (bc ((IntV.ofNat BK : I32) * IntV.ofNat d.aK) :
                  Tensor [BM, BK] I32)).get (mll, kk, ()) =
                d.a.ptr (aIdx d (rowA d i mll.val) (hOf d i) ((t + 1) * BK + kk.val) : ℤ)
              refine Eq.trans (Tensor.get_ptrAddT st.1
                (bc ((IntV.ofNat BK : I32) * IntV.ofNat d.aK)) (mll, kk, ())) ?_
              rw [hptrA mll kk hC]
              exact aPtr_step hd (rowA d i mll.val) (hOf d i) kk.val t
            · intro kk nn hC
              show (st.2.1 + (bc ((IntV.ofNat BK : I32) * IntV.ofNat d.bK) :
                  Tensor [BK, BN] I32)).get (kk, nn, ()) =
                d.b.ptr (bIdx d (colB d i nn.val) (hOf d i) ((t + 1) * BK + kk.val) : ℤ)
              refine Eq.trans (Tensor.get_ptrAddT st.2.1
                (bc ((IntV.ofNat BK : I32) * IntV.ofNat d.bK)) (kk, nn, ())) ?_
              rw [hptrB kk nn hC]
              exact bPtr_step hd (colB d i nn.val) (hOf d i) kk.val t
            · intro mll nn henM henN va vb hva hvb hmag
              have hmrow : mOf d i * BM + mll.val < chunkNat d i := enM_row henM
              have hncol : nOf d i * BN + nn.val < chunkNat d i := enN_col henN
              have hmag' : dotMag va vb (min (t * BK) d.K) ≤ 2 ^ 120 :=
                le_trans (dotMag_mono
                  (min_le_min_right d.K (Nat.mul_le_mul_right BK (Nat.le_succ t)))) hmag
              have hvaS : ∀ k : Fin BK,
                  DType.val? .bf16 (ta.get2 mll k) = some ((kex d t va k.val).toRat) := by
                intro k
                by_cases henA : enA d i t mll.val k.val = true
                · have hk : t * BK + k.val < d.K := enA_k henA
                  have hread : m₁.read .bf16
                      (d.a.ptr (aIdx d (rowA d i mll.val) (hOf d i) (t * BK + k.val) : ℤ)).addr
                        = some (ta.get2 mll k) := hta (mll, k, ()) henA
                  have hval : ta.get2 mll k = va (t * BK + k.val) :=
                    Option.some.inj (hread.symm.trans (hva (t * BK + k.val) hk).1)
                  rw [kex_pos hk, val?_of_bf16]
                  refine ⟨?_, ?_⟩
                  · rw [hval]; exact (hva (t * BK + k.val) hk).2
                  · rw [hval]
                · have henAf : enA d i t mll.val k.val = false := by
                    cases h : enA d i t mll.val k.val with
                    | true => exact absurd h henA
                    | false => rfl
                  have hk : ¬ t * BK + k.val < d.K := by
                    intro hlt
                    apply henA
                    unfold enA
                    rw [decide_eq_true_eq.mpr hmrow, decide_eq_true_eq.mpr hlt]
                    rfl
                  have hval : ta.get2 mll k = FloatV.ofPat 0 := by
                    have h1 : ta.get2 mll k = z1.get (mll, k, ()) :=
                      htao (mll, k, ()) henAf z1 rfl
                    exact h1.trans (hz1e (mll, k, ()))
                  rw [hval, kex_neg hk]
                  exact bf16_zero_val
              have hvbS : ∀ k : Fin BK,
                  DType.val? .bf16 (tb.get2 k nn) = some ((kex d t vb k.val).toRat) := by
                intro k
                by_cases henB : enB d i t k.val nn.val = true
                · have hk : t * BK + k.val < d.K := enB_k henB
                  have hread : m₁.read .bf16
                      (d.b.ptr (bIdx d (colB d i nn.val) (hOf d i) (t * BK + k.val) : ℤ)).addr
                        = some (tb.get2 k nn) := htb (k, nn, ()) henB
                  have hval : tb.get2 k nn = vb (t * BK + k.val) :=
                    Option.some.inj (hread.symm.trans (hvb (t * BK + k.val) hk).1)
                  rw [kex_pos hk, val?_of_bf16]
                  refine ⟨?_, ?_⟩
                  · rw [hval]; exact (hvb (t * BK + k.val) hk).2
                  · rw [hval]
                · have henBf : enB d i t k.val nn.val = false := by
                    cases h : enB d i t k.val nn.val with
                    | true => exact absurd h henB
                    | false => rfl
                  have hk : ¬ t * BK + k.val < d.K := by
                    intro hlt
                    apply henB
                    unfold enB
                    rw [decide_eq_true_eq.mpr hlt, decide_eq_true_eq.mpr hncol]
                    rfl
                  have hval : tb.get2 k nn = FloatV.ofPat 0 := by
                    have h1 : tb.get2 k nn = z2.get (k, nn, ()) :=
                      htbo (k, nn, ()) henBf z2 rfl
                    exact h1.trans (hz2e (k, nn, ()))
                  rw [hval, kex_neg hk]
                  exact bf16_zero_val
              have hprev := hnum mll nn henM henN va vb hva hvb hmag'
              have hc : (st.2.2.get2 mll nn).isFinite = true := hprev.1
              have he0 : |(st.2.2.get2 mll nn).toRat - dotSum va vb (t * BK)| ≤
                  DotLoop.bound rho alpha t (dotMag va vb (t * BK)) := by
                have h := hprev.2.1
                rwa [min_eq_left (le_of_lt (t_mul_BK_lt ht))] at h
              have hA : (∑ k : Fin BK, Device.liftMag .bf16 ((kex d t va k.val).toRat) *
                    Device.liftMag .bf16 ((kex d t vb k.val).toRat)) ≤ 2 ^ 120 := by
                rw [sum_fin_eq_dotMag]
                have hsplit := dotMag_split (t_mul_BK_lt ht) va vb
                have h0 := dotMag_nonneg va vb (t * BK)
                linarith
              have hroom : 2 * ((1 + Device.relDot16 BK) *
                    (|(st.2.2.get2 mll nn).toRat| +
                      ∑ k : Fin BK, Device.liftMag .bf16 ((kex d t va k.val).toRat) *
                        Device.liftMag .bf16 ((kex d t vb k.val).toRat)) +
                    Device.absDot16 BK + 2 * Format.binary32.eta) ≤
                  Format.binary32.maxVal :=
                room_le (vc := (st.2.2.get2 mll nn).toRat) hprev.2.2 hA
              have hrel : 0 ≤ Device.relDot16 BK := by
                unfold Device.relDot16; positivity
              have habs : 0 ≤ Device.absDot16 BK := by
                unfold Device.absDot16; positivity
              have hz0 : DType.val? .f32 ((Tensor.full (cfgBF16.acc.ofBits 0) :
                  Tensor [BM, BN] F32).get2 mll nn) = some 0 := by
                simp only [Tensor.get2, cfgBF16, f32_ofBits_zero]
                exact f32_zero_val
              have hstep := DotLift.dotAdd_step (hP.dotBF16 .tf32) hrel habs hacc mll nn hz0 hvaS hvbS hc hroom
              have he : |(acc.get2 mll nn).toRat -
                    ((st.2.2.get2 mll nn).toRat + dotSum (kex d t va) (kex d t vb) BK)| ≤
                  rho * (|(st.2.2.get2 mll nn).toRat| +
                    dotMag (kex d t va) (kex d t vb) BK) + alpha := by
                have h := hstep.2
                rw [sum_fin_eq_dotSum, sum_fin_eq_dotMag] at h
                exact h
              have h2 : |(acc.get2 mll nn).toRat -
                    dotSum va vb (min ((t + 1) * BK) d.K)| ≤
                  DotLoop.bound rho alpha (t + 1) (dotMag va vb (min ((t + 1) * BK) d.K)) := by
                have hmain := DotLoop.accum_step rho_nonneg' alpha_nonneg
                  (abs_dotSum_le_dotMag va vb (t * BK))
                  (abs_dotSum_le_dotMag (kex d t va) (kex d t vb) BK) he0 he
                rwa [← dotSum_split (t_mul_BK_lt ht) va vb,
                  ← dotMag_split (t_mul_BK_lt ht) va vb] at hmain
              have hT : t + 1 ≤ 32 := by
                have h2' : steps d.K ≤ 32 := steps_le hd.K_le
                omega
              have hble : DotLoop.bound rho alpha (t + 1)
                    (dotMag va vb (min ((t + 1) * BK) d.K)) ≤ 2 ^ 120 := by
                have hb : DotLoop.bound rho alpha (t + 1)
                      (dotMag va vb (min ((t + 1) * BK) d.K)) ≤
                    ((t + 1 : Nat) : ℚ) / 2 ^ 16 *
                        dotMag va vb (min ((t + 1) * BK) d.K) + 1 / 2 ^ 112 :=
                  bound_le_tol (T := t + 1) hT
                    (B := dotMag va vb (min ((t + 1) * BK) d.K)) (dotMag_nonneg va vb _)
                have hTq : ((t + 1 : Nat) : ℚ) ≤ 32 := by exact_mod_cast hT
                have hB0 : (0 : ℚ) ≤ dotMag va vb (min ((t + 1) * BK) d.K) :=
                  dotMag_nonneg va vb _
                have hstep1 : ((t + 1 : Nat) : ℚ) / 2 ^ 16 ≤ 32 / 2 ^ 16 := by
                  rw [div_eq_mul_inv, div_eq_mul_inv]
                  exact mul_le_mul_of_nonneg_right hTq (by norm_num)
                have h1 : ((t + 1 : Nat) : ℚ) / 2 ^ 16 * dotMag va vb (min ((t + 1) * BK) d.K)
                    ≤ 32 / 2 ^ 16 * 2 ^ 120 := by
                  calc ((t + 1 : Nat) : ℚ) / 2 ^ 16 * dotMag va vb (min ((t + 1) * BK) d.K)
                      ≤ 32 / 2 ^ 16 * dotMag va vb (min ((t + 1) * BK) d.K) :=
                        mul_le_mul_of_nonneg_right hstep1 hB0
                    _ ≤ 32 / 2 ^ 16 * 2 ^ 120 := mul_le_mul_of_nonneg_left hmag (by norm_num)
                have h2' : (32 : ℚ) / 2 ^ 16 * 2 ^ 120 + 1 / 2 ^ 112 ≤ 2 ^ 120 := by norm_num
                linarith
              have h3 : |(acc.get2 mll nn).toRat| ≤ 2 ^ 121 := by
                have hds : |dotSum va vb (min ((t + 1) * BK) d.K)| ≤ 2 ^ 120 :=
                  abs_dotSum_le va vb hmag
                have htri : |(acc.get2 mll nn).toRat| ≤
                    |(acc.get2 mll nn).toRat - dotSum va vb (min ((t + 1) * BK) d.K)| +
                      |dotSum va vb (min ((t + 1) * BK) d.K)| := by
                  have h := abs_sub_abs_le_abs_sub ((acc.get2 mll nn).toRat)
                    (dotSum va vb (min ((t + 1) * BK) d.K))
                  linarith
                linarith
              exact ⟨hstep.1, h2, h3⟩

end Claims.SsdBmm



