import Proof.TripAux

/-!
# `chunk_delta_h_bf16`: the main loop

`loopVar` (variable length: the `int64` loop variable) and `loopFix` (fixed length: the `int32`
variable and arguments) run `range(NT)` with `NT = nch` trips; trip `t` calls the step with
`i_t_int64 = t` and the offset of the chunk's last token (`g_last_off`).
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The number of tokens through the chunk `t` of sequence `n`: `min(64(t+1), len) = 64t + cl`. -/
theorem min_chunk {d : Inst} {n t : Nat} (ht : t < nch d n) :
    min (BT * (t + 1)) (lenOf d n) = BT * t + cl d n t := by
  simp only [cl, BT, nch] at *; omega

/-- **The `int64` loop** (`loopVar`) is a loop of `nch` trips. -/
theorem safe_loopVar {d : Inst} (hd : d.ok) {E : Env} {R : Set Int} {σ α : Type} {n hv : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) {bos i_h T NT : I64} (hbos : bos.toInt = (bosOf d n : Int))
    (hih : i_h.toInt = (hv : Int)) (hT : T.toInt = (lenOf d n : Int)) (hNT : NT.toInt = (nch d n : Int))
    {I : Nat → σ → Mem → Prop} {Wt : Nat → Set Int} {Wk : Set Int} {Qf : α → Mem → Prop} {m : Mem}
    (step : I64 → I64 → σ → Prog σ) {st0 : σ} {k : σ → Prog α}
    (hWt : ∀ t u, t < u → u < nch d n → Disjoint (Wt t) (Wt u))
    (hdis : Disjoint (iterW Wt (nch d n)) Wk) (h0 : I 0 st0 m)
    (hstep : ∀ t, t < nch d n → ∀ iT goff : I64, iT.toInt = (t : Int) →
      (d.useG = true → goff.toInt = (gIdx d (rowOf d n) (startOf d n + BT * t + cl d n t - 1) hv : Int)) →
      ∀ st m₁, I t st m₁ → Safe E R (Wt t) (fun st' m₂ => I (t + 1) st' m₂) m₁ (step iT goff st))
    (hk : ∀ st m₁, I (nch d n) st m₁ → Safe E R Wk Qf m₁ (k st)) :
    Safe E R (iterW Wt (nch d n) ∪ Wk) Qf m (loopVar d.HV BT NT T bos i_h step st0 >>= k) := by
  have hT22 := hd.T_ctx
  have hlen : lenOf d n ≤ d.T := by have := start_add_len_le hd hn; omega
  have hbos35 := bos_bound hd hn
  have hHVl := hd.HV_lt
  have hnch : nch d n ≤ 2 ^ 16 := by simp only [nch, BT]; omega
  have hlo : (IntV.ofInt 0 : I64).toInt = 0 := IntV.toInt_ofInt_zero
  have hstp : (IntV.ofInt 1 : I64).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
  have hrange : tl.RangeOk (IntV.ofInt 0 : I64) NT (IntV.ofInt 1 : I64) := by
    unfold tl.RangeOk
    rw [hlo, hstp, hNT]
    simp only [IntV.maxInt, ite_true]
    refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
    have : (nch d n : Int) ≤ 2 ^ 16 := by exact_mod_cast hnch
    norm_num; linarith
  have hc : tl.tripCount (IntV.ofInt 0 : I64).toInt NT.toInt (IntV.ofInt 1 : I64).toInt = nch d n := by
    rw [hlo, hstp, hNT]; simp [tl.tripCount]
  have key := safe_forRange hrange I Wt (Wk := Wk) (m := m) (k := k) (init := st0)
    (body := fun i_t st => do
      let i_t_int64 := i_t
      let last_idx : I64 := (tl.iminimum (Tensor.scalar ((i_t + IntV.ofInt 1) * (IntV.ofNat BT : I64)))
        (Tensor.scalar T)).item - IntV.ofInt 1
      step i_t_int64 (bos * (IntV.ofNat d.HV : I64) + last_idx * (IntV.ofNat d.HV : I64) + i_h) st)
    (by rw [hc]; exact hWt) (by rw [hc]; exact hdis) h0 (fun t ht st m₁ hI => ?_) (by rw [hc]; exact hk)
  · rw [hc] at key; exact key
  · rw [hc] at ht
    have hiT : (IntV.ofInt ((IntV.ofInt 0 : I64).toInt + t * (IntV.ofInt 1 : I64).toInt) : I64).toInt = (t : Int) := by
      rw [hlo, hstp, zero_add, mul_one]
      exact IntV.I64.toInt_ofInt ⟨by omega, by omega⟩
    refine hstep t ht _ _ hiT (fun hG => ?_) st m₁ hI
    have hcl : 0 < cl d n t := by simp only [cl, BT, nch] at *; omega
    have hchunk := min_chunk (d := d) (n := n) ht
    have hHVi : (IntV.ofNat d.HV : I64).toInt = (d.HV : Int) := IntV.I64.toInt_ofNat (by omega)
    have hBTi : (IntV.ofNat BT : I64).toInt = (BT : Int) := IntV.I64.toInt_ofNat (by unfold BT; norm_num)
    have hi1 : (IntV.ofInt ((IntV.ofInt 0 : I64).toInt + t * (IntV.ofInt 1 : I64).toInt) + IntV.ofInt 1 : I64).toInt =
        ((t + 1 : Nat) : Int) := by
      have := IntV.add_nat (a := (IntV.ofInt ((IntV.ofInt 0 : I64).toInt + t * (IntV.ofInt 1 : I64).toInt) : I64))
        (b := (IntV.ofInt 1 : I64)) (x := t) (y := 1) hiT hstp (by omega)
      simpa using this
    have hm1 : ((IntV.ofInt ((IntV.ofInt 0 : I64).toInt + t * (IntV.ofInt 1 : I64).toInt) + IntV.ofInt 1 : I64) *
        IntV.ofNat BT).toInt = (((t + 1) * BT : Nat) : Int) :=
      IntV.mul_nat hi1 hBTi (by unfold BT; omega)
    have hmin : ((tl.iminimum (Tensor.scalar ((IntV.ofInt ((IntV.ofInt 0 : I64).toInt + t * (IntV.ofInt 1 : I64).toInt) +
        IntV.ofInt 1 : I64) * IntV.ofNat BT)) (Tensor.scalar T)).item).toInt = ((BT * t + cl d n t : Nat) : Int) := by
      show (IntV.min _ T).toInt = _
      rw [IntV.toInt_min]
      dsimp only [Tensor.scalar]
      rw [hm1, hT, ← hchunk]
      push_cast
      rw [mul_comm ((t : Int) + 1) (BT : Int)]
    have hlast : (((tl.iminimum (Tensor.scalar ((IntV.ofInt ((IntV.ofInt 0 : I64).toInt + t * (IntV.ofInt 1 : I64).toInt) +
        IntV.ofInt 1 : I64) * IntV.ofNat BT)) (Tensor.scalar T)).item) - IntV.ofInt 1 : I64).toInt =
        ((BT * t + cl d n t - 1 : Nat) : Int) :=
      IntV.sub_nat hmin hstp (by omega) (by simp only [cl, BT] at *; omega)
    have hl2 : ((((tl.iminimum (Tensor.scalar ((IntV.ofInt ((IntV.ofInt 0 : I64).toInt + t * (IntV.ofInt 1 : I64).toInt) +
        IntV.ofInt 1 : I64) * IntV.ofNat BT)) (Tensor.scalar T)).item) - IntV.ofInt 1 : I64) *
        IntV.ofNat d.HV).toInt = (((BT * t + cl d n t - 1) * d.HV : Nat) : Int) :=
      IntV.mul_nat hlast hHVi (by
        have : (BT * t + cl d n t - 1) * d.HV ≤ 2 ^ 22 * 2 ^ 16 :=
          Nat.mul_le_mul (by simp only [cl, BT, nch] at *; omega) hHVl.le
        exact lt_of_le_of_lt this (by norm_num))
    have hb2 : (bos * IntV.ofNat d.HV : I64).toInt = ((bosOf d n * d.HV : Nat) : Int) :=
      IntV.mul_nat hbos hHVi (by
        have : bosOf d n * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) hHVl.le
        exact lt_of_le_of_lt this (by norm_num))
    have hsum := IntV.add_nat hb2 hl2 (by
      have h1 : bosOf d n * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) hHVl.le
      have h2 : (BT * t + cl d n t - 1) * d.HV ≤ 2 ^ 22 * 2 ^ 16 :=
        Nat.mul_le_mul (by simp only [cl, BT, nch] at *; omega) hHVl.le
      calc _ ≤ 2 ^ 35 * 2 ^ 16 + 2 ^ 22 * 2 ^ 16 := by omega
        _ < 2 ^ 63 := by norm_num)
    have htot := IntV.add_nat hsum hih (by
      have h1 : bosOf d n * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) hHVl.le
      have h2 : (BT * t + cl d n t - 1) * d.HV ≤ 2 ^ 22 * 2 ^ 16 :=
        Nat.mul_le_mul (by simp only [cl, BT, nch] at *; omega) hHVl.le
      have h3 : hv < 2 ^ 16 := by omega
      calc _ ≤ 2 ^ 35 * 2 ^ 16 + 2 ^ 22 * 2 ^ 16 + 2 ^ 16 := by omega
        _ < 2 ^ 63 := by norm_num)
    refine htot.trans ?_
    congr 1
    obtain ⟨y, hy⟩ : ∃ y, BT * t + cl d n t = y + 1 := ⟨BT * t + cl d n t - 1, by omega⟩
    simp only [gIdx, bosOf, hy, Nat.add_sub_cancel]
    have : startOf d n + BT * t + cl d n t - 1 = startOf d n + y := by omega
    rw [this]; ring

/-- **The `int32` loop** (`loopFix`) is a loop of `nch` trips. -/
theorem safe_loopFix {d : Inst} (hd : d.ok) {E : Env} {R : Set Int} {σ α : Type} {n hv : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) (hfix : d.varlen = false) {bos i_h : I64} {T NT : I32}
    (hbos : bos.toInt = (bosOf d n : Int)) (hih : i_h.toInt = (hv : Int)) (hT : T.toInt = (d.T : Int))
    (hNT : NT.toInt = (nch d n : Int))
    {I : Nat → σ → Mem → Prop} {Wt : Nat → Set Int} {Wk : Set Int} {Qf : α → Mem → Prop} {m : Mem}
    (step : I64 → I64 → σ → Prog σ) {st0 : σ} {k : σ → Prog α}
    (hWt : ∀ t u, t < u → u < nch d n → Disjoint (Wt t) (Wt u))
    (hdis : Disjoint (iterW Wt (nch d n)) Wk) (h0 : I 0 st0 m)
    (hstep : ∀ t, t < nch d n → ∀ iT goff : I64, iT.toInt = (t : Int) →
      (d.useG = true → goff.toInt = (gIdx d (rowOf d n) (startOf d n + BT * t + cl d n t - 1) hv : Int)) →
      ∀ st m₁, I t st m₁ → Safe E R (Wt t) (fun st' m₂ => I (t + 1) st' m₂) m₁ (step iT goff st))
    (hk : ∀ st m₁, I (nch d n) st m₁ → Safe E R Wk Qf m₁ (k st)) :
    Safe E R (iterW Wt (nch d n) ∪ Wk) Qf m (loopFix d.HV BT NT T bos i_h step st0 >>= k) := by
  have hT22 := hd.T_ctx
  have hlenT : lenOf d n = d.T := by simp [lenOf, hfix]
  have hbos35 := bos_bound hd hn
  have hHVl := hd.HV_lt
  have hTl := hd.T_lt
  have hnch : nch d n ≤ 2 ^ 16 := by simp only [nch, BT]; omega
  have hlo : (IntV.ofInt 0 : I32).toInt = 0 := IntV.toInt_ofInt_zero
  have hstp : (IntV.ofInt 1 : I32).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) NT (IntV.ofInt 1 : I32) := by
    unfold tl.RangeOk
    rw [hlo, hstp, hNT]
    simp only [IntV.maxInt, ite_true]
    refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
    have : (nch d n : Int) ≤ 2 ^ 16 := by exact_mod_cast hnch
    norm_num; linarith
  have hc : tl.tripCount (IntV.ofInt 0 : I32).toInt NT.toInt (IntV.ofInt 1 : I32).toInt = nch d n := by
    rw [hlo, hstp, hNT]; simp [tl.tripCount]
  unfold loopFix
  have key := safe_forRange hrange I Wt (Wk := Wk) (m := m) (k := k) (init := st0)
    (body := fun i_t st => do
      let i_t_int64 := (← tl.convert .i32 .i64 (Tensor.scalar i_t)).item
      let last_idx : I32 := (tl.iminimum (Tensor.scalar ((i_t + IntV.ofInt 1) * (IntV.ofNat BT : I32)))
        (Tensor.scalar T)).item - IntV.ofInt 1
      let lh := (← tl.convert .i32 .i64 (Tensor.scalar (last_idx * (IntV.ofNat d.HV : I32)))).item
      step i_t_int64 (bos * (IntV.ofNat d.HV : I64) + lh + i_h) st)
    (by rw [hc]; exact hWt) (by rw [hc]; exact hdis) h0 (fun t ht st m₁ hI => ?_) (by rw [hc]; exact hk)
  · rw [hc] at key; exact key
  · rw [hc] at ht
    have hiT32 : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + t * (IntV.ofInt 1 : I32).toInt) : I32).toInt = (t : Int) := by
      rw [hlo, hstp, zero_add, mul_one]
      exact IntV.I32.toInt_ofInt ⟨by omega, by omega⟩
    refine safe_convert (fun i => ConvRel.exists_i32_i64 _) fun r1 hr1 => ?_
    refine safe_convert (fun i => ConvRel.exists_i32_i64 _) fun r2 hr2 => ?_
    have hr1' : r1.item.toInt = (t : Int) := (ConvRel.toInt_i32_i64 (hr1 ())).trans hiT32
    refine hstep t ht _ _ hr1' (fun hG => ?_) st m₁ hI
    have hcl : 0 < cl d n t := by simp only [cl, BT, nch] at *; omega
    have hchunk := min_chunk (d := d) (n := n) ht
    have hlen_le : BT * t + cl d n t ≤ d.T := by simp only [cl, BT, nch] at *; omega
    have hTHV := hd.THV_lt hG
    have hHVi32 : (IntV.ofNat d.HV : I32).toInt = (d.HV : Int) := IntV.I32.toInt_ofNat (by omega)
    have hHVi : (IntV.ofNat d.HV : I64).toInt = (d.HV : Int) := IntV.I64.toInt_ofNat (by omega)
    have hBTi : (IntV.ofNat BT : I32).toInt = (BT : Int) := IntV.I32.toInt_ofNat (by unfold BT; norm_num)
    have hi1 : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + t * (IntV.ofInt 1 : I32).toInt) + IntV.ofInt 1 : I32).toInt =
        ((t + 1 : Nat) : Int) := by
      have := IntV.add_nat (a := (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + t * (IntV.ofInt 1 : I32).toInt) : I32))
        (b := (IntV.ofInt 1 : I32)) (x := t) (y := 1) hiT32 hstp (by omega)
      simpa using this
    have hm1 : ((IntV.ofInt ((IntV.ofInt 0 : I32).toInt + t * (IntV.ofInt 1 : I32).toInt) + IntV.ofInt 1 : I32) *
        IntV.ofNat BT).toInt = (((t + 1) * BT : Nat) : Int) :=
      IntV.mul_nat hi1 hBTi (by unfold BT; simp only [cl, BT, nch] at *; omega)
    have hmin : ((tl.iminimum (Tensor.scalar ((IntV.ofInt ((IntV.ofInt 0 : I32).toInt + t * (IntV.ofInt 1 : I32).toInt) +
        IntV.ofInt 1 : I32) * IntV.ofNat BT)) (Tensor.scalar T)).item).toInt = ((BT * t + cl d n t : Nat) : Int) := by
      show (IntV.min _ T).toInt = _
      rw [IntV.toInt_min]
      dsimp only [Tensor.scalar]
      rw [hm1, hT, ← hlenT, ← hchunk]
      push_cast
      rw [mul_comm ((t : Int) + 1) (BT : Int)]
    have hlast : (((tl.iminimum (Tensor.scalar ((IntV.ofInt ((IntV.ofInt 0 : I32).toInt + t * (IntV.ofInt 1 : I32).toInt) +
        IntV.ofInt 1 : I32) * IntV.ofNat BT)) (Tensor.scalar T)).item) - IntV.ofInt 1 : I32).toInt =
        ((BT * t + cl d n t - 1 : Nat) : Int) :=
      IntV.sub_nat hmin hstp (by omega) (by simp only [cl, BT] at *; omega)
    have hl2 : ((((tl.iminimum (Tensor.scalar ((IntV.ofInt ((IntV.ofInt 0 : I32).toInt + t * (IntV.ofInt 1 : I32).toInt) +
        IntV.ofInt 1 : I32) * IntV.ofNat BT)) (Tensor.scalar T)).item) - IntV.ofInt 1 : I32) *
        IntV.ofNat d.HV).toInt = (((BT * t + cl d n t - 1) * d.HV : Nat) : Int) :=
      IntV.mul_nat hlast hHVi32 (by
        have : (BT * t + cl d n t - 1) * d.HV ≤ d.T * d.HV := Nat.mul_le_mul (by omega) le_rfl
        omega)
    have hr2' : r2.item.toInt = (((BT * t + cl d n t - 1) * d.HV : Nat) : Int) :=
      (ConvRel.toInt_i32_i64 (hr2 ())).trans hl2
    have hb2 : (bos * IntV.ofNat d.HV : I64).toInt = ((bosOf d n * d.HV : Nat) : Int) :=
      IntV.mul_nat hbos hHVi (by
        have : bosOf d n * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) hHVl.le
        exact lt_of_le_of_lt this (by norm_num))
    have h1 : bosOf d n * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) hHVl.le
    have h2 : (BT * t + cl d n t - 1) * d.HV ≤ 2 ^ 22 * 2 ^ 16 :=
      Nat.mul_le_mul (by simp only [cl, BT, nch] at *; omega) hHVl.le
    have hsum := IntV.add_nat hb2 hr2' (by
      calc _ ≤ 2 ^ 35 * 2 ^ 16 + 2 ^ 22 * 2 ^ 16 := by omega
        _ < 2 ^ 63 := by norm_num)
    have htot := IntV.add_nat hsum hih (by
      have h3 : hv < 2 ^ 16 := by omega
      calc _ ≤ 2 ^ 35 * 2 ^ 16 + 2 ^ 22 * 2 ^ 16 + 2 ^ 16 := by omega
        _ < 2 ^ 63 := by norm_num)
    refine htot.trans ?_
    congr 1
    obtain ⟨y, hy⟩ : ∃ y, BT * t + cl d n t = y + 1 := ⟨BT * t + cl d n t - 1, by omega⟩
    simp only [gIdx, bosOf, hy, Nat.add_sub_cancel]
    have : startOf d n + BT * t + cl d n t - 1 = startOf d n + y := by omega
    rw [this]; ring

end Claims.ChunkDeltaH
