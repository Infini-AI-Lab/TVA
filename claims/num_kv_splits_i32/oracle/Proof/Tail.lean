import Proof.Loads

/-!
# Source lines 81-103: the chunk sizes and the split counts

Each stage is a lemma with an arbitrary continuation: `safe_chunk1` (lines 81-82), `safe_core`
(lines 85-88, the `float32` lines, through `core_chain`), `safe_chunk2` (lines 89-99) and
`safe_splits` (lines 101-103). None touches memory.
-/

namespace Claims.NumKvSplits

open TVA

variable {α : Type} {E : Env} {R W : Set Int} {Q : α → Mem → Prop} {m : Mem}

/-- The lengths' width is at least `2`. -/
theorem w_two {d : Inst} (hd : d.ok) : 2 ≤ wOf d.dL := by have := w_ge hd.dL_ok; omega

/-- An `int32` argument below `2^31` has its value. -/
theorem ofNat_val {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n :=
  IntV.I32.toInt_ofNat h

/-- `max_kv_splits` promoted to the lengths' width. -/
theorem K_cast_val {d : Inst} (hd : d.ok) :
    (IntV.cast (IntV.ofNat d.K : I32) : IntV true (wOf d.dL)).toInt = d.K := by
  rw [IntV.toInt_cast _ (w_ge hd.dL_ok), ofNat_val (by have := hd.K_le; omega)]

/-- **Lines 81-82**: `kv_chunk_size_1` is `chunk1`. -/
theorem safe_chunk1 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {mxl mnl : IntV true (wOf d.dL)} (hmx : mxl.toInt = maxLen d init)
    (hmn : mnl.toInt = minLen d init) {k : Tensor [] (IntV true (wOf d.dL)) → Prog α}
    (hk : ∀ c : Tensor [] (IntV true (wOf d.dL)), c.item.toInt = chunk1 d init →
      Safe E R W Q m (k c)) :
    Safe E R W Q m (tl.cdiv (Tensor.scalar mxl) (Tensor.scalar mnl) >>= fun q1 =>
      tl.cdiv (Tensor.scalar mxl)
        (Tensor.scalar (q1.item.min (IntV.cast (IntV.ofNat d.K : I32)))) >>= k) := by
  have hw := w_two hd
  have hp := pow_wZ hd.dL_ok
  obtain ⟨hM1, hM2, -⟩ := maxLen_bounds hd hpre
  obtain ⟨hm1, hm2⟩ := minLen_bounds hd hpre
  have hK1 : (1 : ℤ) ≤ d.K := by exact_mod_cast hd.K_pos
  have hK2 : (d.K : ℤ) ≤ 2 ^ 22 := by exact_mod_cast hd.K_le
  refine safe_cdiv hw (fun _ => ?_) fun q1 hq1 => ?_
  · show 0 ≤ mxl.toInt ∧ 0 < mnl.toInt ∧ mxl.toInt + mnl.toInt - 1 < 2 ^ (wOf d.dL - 1)
    rw [hmx, hmn]; omega
  have hq : q1.item.toInt = cdivZ (maxLen d init) (minLen d init) := by
    rw [Tensor.item, hq1 ()]; exact congrArg₂ cdivZ hmx hmn
  have hmin : (q1.item.min (IntV.cast (IntV.ofNat d.K : I32)) : IntV true (wOf d.dL)).toInt =
      min (cdivZ (maxLen d init) (minLen d init)) d.K := by
    rw [IntV.toInt_min, hq, K_cast_val hd]
  have hq1p := cdivZ_pos hM1 hm1
  refine safe_cdiv hw (fun _ => ?_) fun c hc => hk c ?_
  · show 0 ≤ mxl.toInt ∧ 0 < (q1.item.min (IntV.cast (IntV.ofNat d.K : I32))).toInt ∧
      mxl.toInt + (q1.item.min (IntV.cast (IntV.ofNat d.K : I32))).toInt - 1 < 2 ^ (wOf d.dL - 1)
    rw [hmx, hmin]
    refine ⟨by omega, lt_of_lt_of_le (by norm_num) (le_min hq1p hK1), ?_⟩
    have := min_le_right (cdivZ (maxLen d init) (minLen d init)) (d.K : ℤ)
    omega
  · rw [Tensor.item, hc ()]
    show cdivZ mxl.toInt (q1.item.min (IntV.cast (IntV.ofNat d.K : I32))).toInt = _
    rw [hmx, hmin]; rfl

/-- **Lines 85-88**: `ext_device_core_count` satisfies `CoreOk` and is at most `2^21`. -/
theorem safe_core {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (hP : Device E.P)
    {mxl : IntV true (wOf d.dL)} (hmx : mxl.toInt = maxLen d init)
    {k : Tensor [] I32 → Prog α}
    (hk : ∀ e : Tensor [] I32, CoreOk d init e.item.toInt → e.item.toInt ≤ 2 ^ 21 →
      Safe E R W Q m (k e)) :
    Safe E R W Q m (tl.itof (f := .binary32) (Tensor.scalar mxl) >>= fun msl =>
      tl.div msl (Tensor.scalar c64) >>= fun ext => tl.log2 ext >>= fun lg =>
      tl.maximum lg (Tensor.scalar c1) >>= fun mx =>
      tl.itof (f := .binary32) (Tensor.scalar (IntV.ofNat d.C : I32)) >>= fun dcc =>
      tl.mul dcc mx >>= fun prod => tl.ftoi (sg := true) (w := 32) prod >>= k) := by
  obtain ⟨hM1, hM2, -⟩ := maxLen_bounds hd hpre
  set L : ℕ := (maxLen d init).toNat with hL
  have hLz : (L : ℤ) = maxLen d init := Int.toNat_of_nonneg (by omega)
  have hL1 : 1 ≤ L := by omega
  have hL2 : L ≤ 2 ^ 22 := by omega
  have hC1 := hd.C_pos
  have hC2 := hd.C_le
  have hstd : Format.binary32.Std := by decide
  refine safe_itof hstd fun msl hmsl => ?_
  refine safe_div (fun i => hP.divTotal _ _ _) fun ext hext => ?_
  refine safe_math (fn := .log2) (fun i => hP.mathTotal _ _ _) fun lg hlg => ?_
  refine safe_maximum fun mx hmx' => ?_
  refine safe_itof hstd fun dcc hdcc => ?_
  refine safe_mul fun prod hprod => ?_
  have hmsl0 : tl.ItofRel .binary32 (L : ℤ) (msl.get ()) := by
    have := hmsl (); rw [hLz, ← hmx]; exact this
  have hdcc0 : tl.ItofRel .binary32 (d.C : ℤ) (dcc.get ()) := by
    have := hdcc (); rwa [show ((Tensor.scalar (IntV.ofNat d.C : I32)).get ()).toInt = d.C from
      ofNat_val (by omega)] at this
  obtain ⟨hpf, hpC, hpv⟩ := core_chain hP hL1 hL2 hC1 hC2 hmsl0 (hext ()) (hlg ()) (hmx' ())
    hdcc0 (hprod ())
  have hlogb : max (Real.logb 2 ((L : ℝ) / 64)) 1 ≤ 16 := by
    have ha1 : (2 : ℝ) ^ (-6 : ℤ) ≤ (L : ℝ) / 64 := by
      have : (1 : ℝ) ≤ L := by exact_mod_cast hL1
      have e : (2 : ℝ) ^ (-6 : ℤ) = 1 / 64 := by norm_num [zpow_neg]
      rw [e]; linarith
    have ha2 : (L : ℝ) / 64 ≤ 2 ^ (16 : ℤ) := by
      have : (L : ℝ) ≤ 2 ^ 22 := by exact_mod_cast hL2
      have e : (2 : ℝ) ^ (16 : ℤ) = 2 ^ 22 / 64 := by norm_num
      rw [e]; linarith
    exact max_le (logb_bounds ha1 ha2).2 (by norm_num)
  have hCr : (d.C : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hC2
  have hCr1 : (1 : ℝ) ≤ d.C := by exact_mod_cast hC1
  have hτ : (d.C : ℝ) * 2 ^ (-17 : ℤ) ≤ 1 / 2 := by
    have : (2 : ℝ) ^ (-17 : ℤ) = 1 / 2 ^ 17 := by norm_num [zpow_neg]
    rw [this]; nlinarith
  have hv16 : (d.C : ℝ) * max (Real.logb 2 ((L : ℝ) / 64)) 1 ≤ d.C * 16 :=
    mul_le_mul_of_nonneg_left hlogb (by positivity)
  have hpup : (prod.get ()).toReal ≤ d.C * 16 + 1 / 2 := by
    have := (abs_le.mp hpv).2; linarith
  obtain ⟨hex, hall⟩ := trunc_spec hpf (by linarith) (by nlinarith)
  refine safe_ftoi hstd (fun _ => hex) fun e he => hk e ?_ ?_
  · obtain ⟨hlo, hhi⟩ := hall _ (he ())
    change ((e.item.toInt : ℤ) : ℝ) ≤ _ at hlo
    change _ < ((e.item.toInt : ℤ) : ℝ) + 1 at hhi
    have hcore : coreExact d init = (d.C : ℝ) * max (Real.logb 2 ((L : ℝ) / 64)) 1 := by
      unfold coreExact; rw [← hLz, Int.cast_natCast]
    have hv := abs_le.mp hpv
    refine ⟨?_, ?_, ?_⟩
    · have : ((d.C : ℤ) : ℝ) < ((e.item.toInt + 1 : ℤ) : ℝ) := by
        push_cast; exact lt_of_le_of_lt hpC hhi
      have := (Int.cast_lt (R := ℝ)).mp this
      omega
    · rw [hcore]; exact le_trans hlo (by linarith)
    · rw [hcore]; linarith
  · obtain ⟨hlo, -⟩ := hall _ (he ())
    change ((e.item.toInt : ℤ) : ℝ) ≤ _ at hlo
    have : (e.item.toInt : ℝ) ≤ 2 ^ 21 := by nlinarith
    exact_mod_cast this

/-- `num_head // num_kv_head` as an `int32` and the condition of line 90. -/
theorem group_val {d : Inst} (hd : d.ok) {g : I32}
    (hg : (IntV.ofNat d.H : I32).div? (IntV.ofNat d.Hkv) = some g) :
    g.toInt = (d.H : ℤ) / d.Hkv := by
  have hH : d.H < 2 ^ 31 := by
    have := hd.grid_le
    have h1 : d.H ≤ d.S * d.G * d.H :=
      Nat.le_mul_of_pos_left _ (Nat.mul_pos hd.S_pos hd.G_pos)
    omega
  have hHkv : d.Hkv < 2 ^ 31 := by have := hd.Hkv_le; omega
  rw [IntV.toInt_of_div? hg (by rw [ofNat_val hH]; positivity)
    (by rw [ofNat_val hHkv]; exact_mod_cast hd.Hkv_pos), ofNat_val hH, ofNat_val hHkv]

/-- **Lines 89-99**: `kv_chunk_size_2` is `chunk2` of `E`. -/
theorem safe_chunk2 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {mxl : IntV true (wOf d.dL)} (hmx : mxl.toInt = maxLen d init) {e : I32}
    (he1 : 1 ≤ e.toInt) (he2 : e.toInt ≤ 2 ^ 21) {k : Tensor [] (IntV true (wOf d.dL)) → Prog α}
    (hk : ∀ c : Tensor [] (IntV true (wOf d.dL)), c.item.toInt = chunk2 d init e.toInt →
      Safe E R W Q m (k c)) :
    Safe E R W Q m (tl.idiv (Tensor.scalar (IntV.ofNat d.H : I32))
        (Tensor.scalar (IntV.ofNat d.Hkv : I32)) >>= fun g =>
      (if g.item.eq (IntV.ofInt 1) = true then
          pure (IntV.ofNat d.S * IntV.ofNat d.G * IntV.ofNat d.H : I32)
        else do
          let c ← tl.cdiv (Tensor.scalar (IntV.ofNat d.H : I32))
            (Tensor.scalar ((IntV.ofInt 16 : I32).min g.item))
          pure (IntV.ofNat d.S * IntV.ofNat d.G * c.item : I32) : Prog I32) >>= fun tg =>
      tl.cdiv (Tensor.scalar e) (Tensor.scalar tg) >>= fun q2 =>
      tl.cdiv (Tensor.scalar mxl)
        (Tensor.scalar (IntV.cast (q2.item.min (IntV.ofNat d.K : I32)))) >>= k) := by
  have hw := w_two hd
  have hp := pow_wZ hd.dL_ok
  obtain ⟨hM1, hM2, -⟩ := maxLen_bounds hd hpre
  obtain ⟨hT1, hT2⟩ := tokenGrid_bounds hd
  have hgrid : ((d.S * d.G * d.H : ℕ) : ℤ) ≤ 2 ^ 30 := by exact_mod_cast hd.grid_le
  push_cast at hgrid
  have hH : (d.H : ℤ) < 2 ^ 31 := by
    have : (d.H : ℤ) ≤ d.S * d.G * d.H := by
      have h1 : (1 : ℤ) ≤ (d.S : ℤ) * d.G := by
        have : 1 ≤ d.S * d.G := Nat.one_le_iff_ne_zero.mpr (by have := hd.S_pos; have := hd.G_pos; positivity)
        exact_mod_cast this
      nlinarith
    omega
  have hH0 : (1 : ℤ) ≤ d.H := by have := hd.Hkv_pos; have := hd.Hkv_le; omega
  have hHv : (IntV.ofNat d.H : I32).toInt = d.H := ofNat_val (by omega)
  have hSv : (IntV.ofNat d.S : I32).toInt = d.S := ofNat_val (by have := hd.S_le; omega)
  have hGv : (IntV.ofNat d.G : I32).toInt = d.G := by
    have : (d.G : ℤ) < 2 ^ 31 := by
      have : (d.G : ℤ) ≤ d.S * d.G := by have := hd.S_pos; nlinarith
      nlinarith
    exact ofNat_val (by omega)
  have hSG : (IntV.ofNat d.S * IntV.ofNat d.G : I32).toInt = (d.S : ℤ) * d.G :=
    IntV.toInt_mul hSv hGv (by
      rw [IntV.I32.sfits_iff]; constructor
      · have : (0 : ℤ) ≤ (d.S : ℤ) * d.G := by positivity
        omega
      · have : (d.S : ℤ) * d.G ≤ d.S * d.G * d.H := by nlinarith
        omega)
  have hKv : (IntV.ofNat d.K : I32).toInt = d.K := ofNat_val (by have := hd.K_le; omega)
  have hK1 : (1 : ℤ) ≤ d.K := by exact_mod_cast hd.K_pos
  have hK2 : (d.K : ℤ) ≤ 2 ^ 22 := by exact_mod_cast hd.K_le
  refine safe_idiv (fun _ => ?_) fun g hg => ?_
  · have hHkv : (IntV.ofNat d.Hkv : I32).toInt = d.Hkv := ofNat_val (by have := hd.Hkv_le; omega)
    obtain ⟨q, hq, -⟩ := IntV.div?_of_nonneg (a := (IntV.ofNat d.H : I32))
      (b := (IntV.ofNat d.Hkv : I32)) (by rw [hHv]; omega)
      (by rw [hHkv]; exact_mod_cast hd.Hkv_pos)
    exact ⟨q, hq⟩
  have hgv : g.item.toInt = (d.H : ℤ) / d.Hkv := group_val hd (hg ())
  -- the token grid, in both branches, then the rest
  suffices hrest : ∀ tg : I32, tg.toInt = tokenGrid d →
      Safe E R W Q m (tl.cdiv (Tensor.scalar e) (Tensor.scalar tg) >>= fun q2 =>
        tl.cdiv (Tensor.scalar mxl)
          (Tensor.scalar (IntV.cast (q2.item.min (IntV.ofNat d.K : I32)))) >>= k) by
    by_cases h1 : (d.H : ℤ) / d.Hkv = 1
    · have hc : g.item.eq (IntV.ofInt 1) = true := by
        rw [IntV.eq_eq, hgv, h1, IntV.I32.toInt_ofInt (by norm_num)]; simp
      rw [if_pos hc, pure_bind]
      refine hrest _ ?_
      have hp0 : (0 : ℤ) ≤ (d.S : ℤ) * d.G * d.H := by positivity
      rw [IntV.toInt_mul hSG hHv (by rw [IntV.I32.sfits_iff]; constructor <;> omega)]
      unfold tokenGrid; rw [if_pos h1]
    · have hc : g.item.eq (IntV.ofInt 1) = false := by
        rw [IntV.eq_eq, hgv, IntV.I32.toInt_ofInt (by norm_num)]; simp [h1]
      rw [if_neg (by rw [hc]; simp), bind_assoc]
      have hg1 : (1 : ℤ) ≤ (d.H : ℤ) / d.Hkv := by
        rw [Int.le_ediv_iff_mul_le (by exact_mod_cast hd.Hkv_pos)]
        have := hd.Hkv_le; omega
      have hbh : ((IntV.ofInt 16 : I32).min g.item).toInt = min 16 ((d.H : ℤ) / d.Hkv) := by
        rw [IntV.toInt_min, hgv, IntV.I32.toInt_ofInt (by norm_num)]
      have hb1 : (1 : ℤ) ≤ min 16 ((d.H : ℤ) / d.Hkv) := le_min (by norm_num) hg1
      refine safe_cdiv (by norm_num) (fun _ => ?_) fun c hc' => ?_
      · show 0 ≤ (IntV.ofNat d.H : I32).toInt ∧ 0 < ((IntV.ofInt 16 : I32).min g.item).toInt ∧
          (IntV.ofNat d.H : I32).toInt + ((IntV.ofInt 16 : I32).min g.item).toInt - 1 < 2 ^ (32 - 1)
        rw [hHv, hbh]
        have h16 := min_le_left 16 ((d.H : ℤ) / d.Hkv)
        have hH30 : (d.H : ℤ) ≤ 2 ^ 30 := by
          have h1 : (1 : ℤ) ≤ (d.S : ℤ) * d.G := by
            have : 1 ≤ d.S * d.G := Nat.one_le_iff_ne_zero.mpr
              (by have := hd.S_pos; have := hd.G_pos; positivity)
            exact_mod_cast this
          nlinarith
        refine ⟨by omega, lt_of_lt_of_le one_pos hb1, ?_⟩
        norm_num; omega
      rw [pure_bind]
      refine hrest _ ?_
      have hcv : c.item.toInt = cdivZ d.H (min 16 ((d.H : ℤ) / d.Hkv)) := by
        rw [Tensor.item, hc' ()]
        show cdivZ (IntV.ofNat d.H : I32).toInt ((IntV.ofInt 16 : I32).min g.item).toInt = _
        rw [hHv, hbh]
      have hc1 := cdivZ_le (a := (d.H : ℤ)) hH0 hb1
      have hc0 := cdivZ_pos (a := (d.H : ℤ)) hH0 hb1
      rw [IntV.toInt_mul hSG hcv (by
        rw [IntV.I32.sfits_iff]; constructor
        · have : (0 : ℤ) ≤ (d.S : ℤ) * d.G := by positivity
          nlinarith
        · have : (d.S : ℤ) * d.G * cdivZ d.H (min 16 ((d.H : ℤ) / d.Hkv)) ≤ d.S * d.G * d.H :=
            mul_le_mul_of_nonneg_left hc1 (by positivity)
          omega)]
      unfold tokenGrid; rw [if_neg h1]
  intro tg htg
  refine safe_cdiv (by norm_num) (fun _ => ?_) fun q2 hq2 => ?_
  · show 0 ≤ e.toInt ∧ 0 < tg.toInt ∧ e.toInt + tg.toInt - 1 < 2 ^ (32 - 1)
    rw [htg]; norm_num; omega
  have hq : q2.item.toInt = cdivZ e.toInt (tokenGrid d) := by
    rw [Tensor.item, hq2 ()]; show cdivZ e.toInt tg.toInt = _; rw [htg]
  have hmin : (q2.item.min (IntV.ofNat d.K : I32)).toInt = min (cdivZ e.toInt (tokenGrid d)) d.K := by
    rw [IntV.toInt_min, hq, hKv]
  have hq0 := cdivZ_pos he1 hT1
  have hcast : (IntV.cast (q2.item.min (IntV.ofNat d.K : I32)) : IntV true (wOf d.dL)).toInt =
      min (cdivZ e.toInt (tokenGrid d)) d.K := by
    rw [IntV.toInt_cast _ (w_ge hd.dL_ok), hmin]
  refine safe_cdiv hw (fun _ => ?_) fun c hc => hk c ?_
  · show 0 ≤ mxl.toInt ∧ 0 < (IntV.cast (q2.item.min (IntV.ofNat d.K : I32)) :
        IntV true (wOf d.dL)).toInt ∧ mxl.toInt + (IntV.cast (q2.item.min (IntV.ofNat d.K : I32)) :
        IntV true (wOf d.dL)).toInt - 1 < 2 ^ (wOf d.dL - 1)
    rw [hmx, hcast]
    refine ⟨by omega, lt_of_lt_of_le (by norm_num) (le_min hq0 hK1), ?_⟩
    have := min_le_right (cdivZ e.toInt (tokenGrid d)) (d.K : ℤ)
    omega
  · rw [Tensor.item, hc ()]
    show cdivZ mxl.toInt (IntV.cast (q2.item.min (IntV.ofNat d.K : I32)) :
      IntV true (wOf d.dL)).toInt = _
    rw [hmx, hcast]; rfl

/-- **Lines 101-103**: the split counts of the batch's lanes are `numSplits`. -/
theorem safe_splits {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {Ev : ℤ}
    (hE : 1 ≤ Ev) {sl : Tensor [d.MAX - 0] (IntV true (wOf d.dL))}
    (hsl : ∀ i : Idx [d.MAX - 0], (sl.get i).toInt =
      if i.1.val < d.S then len d init i.1.val else maxLen d init)
    {c1 c2 : IntV true (wOf d.dL)} (hc1 : c1.toInt = chunk1 d init)
    (hc2 : c2.toInt = chunk2 d init Ev)
    {k : Tensor [d.MAX - 0] (IntV true (wOf d.dL)) → Tensor [d.MAX - 0] (IntV true (wOf d.dL)) →
      Prog α}
    (hk : ∀ n1 n2 : Tensor [d.MAX - 0] (IntV true (wOf d.dL)),
      (∀ i : Idx [d.MAX - 0], i.1.val < d.S →
        ((tl.imaximum n1 n2).get i).toInt = numSplits d init Ev i.1.val) →
      Safe E R W Q m (k n1 n2)) :
    Safe E R W Q m (tl.cdiv sl (Tensor.full c1) >>= fun n1 =>
      tl.cdiv sl (Tensor.full c2) >>= fun n2 => k n1 n2) := by
  have hw := w_two hd
  have hp := pow_wZ hd.dL_ok
  obtain ⟨hM1, hM2, -⟩ := maxLen_bounds hd hpre
  obtain ⟨h11, h12⟩ := chunk1_bounds hd hpre
  obtain ⟨h21, h22⟩ := chunk2_bounds hd hpre hE
  have hlane : ∀ i : Idx [d.MAX - 0], 0 ≤ (sl.get i).toInt ∧ (sl.get i).toInt ≤ 2 ^ 22 := by
    intro i
    rw [hsl i]
    split_ifs with hi
    · obtain ⟨a, b⟩ := hpre.2.2.2 _ hi; omega
    · omega
  refine safe_cdiv hw (fun i => ?_) fun n1 hn1 => ?_
  · obtain ⟨a, b⟩ := hlane i
    show 0 ≤ (sl.get i).toInt ∧ 0 < c1.toInt ∧ (sl.get i).toInt + c1.toInt - 1 < 2 ^ (wOf d.dL - 1)
    rw [hc1]; omega
  refine safe_cdiv hw (fun i => ?_) fun n2 hn2 => hk n1 n2 fun i hi => ?_
  · obtain ⟨a, b⟩ := hlane i
    show 0 ≤ (sl.get i).toInt ∧ 0 < c2.toInt ∧ (sl.get i).toInt + c2.toInt - 1 < 2 ^ (wOf d.dL - 1)
    rw [hc2]; omega
  show (IntV.max (n1.get i) (n2.get i)).toInt = _
  rw [IntV.toInt_max, hn1 i, hn2 i]
  show max (cdivZ (sl.get i).toInt c1.toInt) (cdivZ (sl.get i).toInt c2.toInt) = _
  rw [hsl i, if_pos hi, hc1, hc2]
  rfl

end Claims.NumKvSplits
