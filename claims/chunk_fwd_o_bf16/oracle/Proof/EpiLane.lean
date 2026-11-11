import Claim
import TVA.Proof
import Proof.Gate
import Proof.NumEp
import Proof.FloatLane

/-!
# One output element of the epilogue

`epi_lane`: given the facts of the gated terms (`OGate`, `AGate`) at one lane, the rounding of the
scores, the last dot, the scale, the products, the sum and the store's conversion, the converted
value is finite and within `tol` of `ref`.
-/

open TVA
namespace Claims.ChunkFwdO

variable {d : Inst} {g : Geo} {r : Nat}

/-- The sum over the keys `s ≤ t` of the chunk is a sum over the local keys `k ≤ i`. -/
theorem sum_upTo (hg : Geo.Ok d g r) (F : Nat → ℝ) (i : Nat) :
    ∑ s ∈ upTo d r (g.lo + i), F s = ∑ k ∈ Finset.range (i + 1), F (g.lo + k) := by
  have : upTo d r (g.lo + i) = Finset.Ico g.lo (g.lo + i + 1) := by
    ext s; simp only [upTo, Finset.mem_Icc, Finset.mem_Ico, hg.lo_eq]; omega
  rw [this, Finset.sum_Ico_eq_sum_range]
  have : g.lo + i + 1 - g.lo = i + 1 := by omega
  rw [this]

/-- The gate of token `g.lo + i` is in `[0, 2^64]`. -/
theorem gate_bounds (hg : Geo.Ok d g r) {i : Nat} (hi : i < BT) (hr : RowIn d r (g.lo + i) g.ih) :
    0 ≤ gate d r (g.lo + i) g.ih ∧ gate d r (g.lo + i) g.ih ≤ 2 ^ 64 := by
  unfold gate
  by_cases hG : d.useG = true
  · simp only [hG, ↓reduceIte]
    refine ⟨by positivity, ?_⟩
    calc (2:ℝ) ^ ((gAt d (g.lo + i) g.ih : ℚ) : ℝ) ≤ (2:ℝ) ^ ((64:ℕ):ℝ) :=
          Real.rpow_le_rpow_of_exponent_le (by norm_num) (by
            have := (hr.2.2.2.1 hG).1
            push_cast; exact_mod_cast this)
      _ = 2 ^ 64 := by norm_num
  · by_cases hGm : d.useGamma = true
    · simp only [hG, hGm, Bool.false_eq_true, ↓reduceIte]
      refine ⟨by positivity, ?_⟩
      obtain ⟨-, h64', h1⟩ := hr.2.2.1 hGm
      have hn : ((g.lo + i - rowLo d r : Nat) : ℚ) + 1 ≤ 64 := by
        rw [hg.lo_eq, Nat.add_sub_cancel_left]; have : i + 1 ≤ 64 := hi; exact_mod_cast this
      have hn0 : (0:ℚ) ≤ ((g.lo + i - rowLo d r : Nat) : ℚ) + 1 := by positivity
      calc (2:ℝ) ^ (((gam d g.ih * (((g.lo + i - rowLo d r : Nat) : ℚ) + 1) : ℚ)) : ℝ)
          ≤ (2:ℝ) ^ ((64:ℕ):ℝ) :=
            Real.rpow_le_rpow_of_exponent_le (by norm_num) (by
              push_cast
              have hq : gam d g.ih * (((g.lo + i - rowLo d r : Nat) : ℚ) + 1) ≤ 64 := by
                nlinarith [mul_nonneg (sub_nonneg.2 h1) hn0]
              exact_mod_cast hq)
        _ = 2 ^ 64 := by norm_num
    · simp [hG, hGm]; norm_num
  
/-- The weight of key `g.lo + k ≤ g.lo + i` is in `[0, 2^64]`. -/
theorem wgt_bounds (hg : Geo.Ok d g r) {i k : Nat} (hik : k ≤ i) (hi : i < BT)
    (hr : RowIn d r (g.lo + i) g.ih) :
    0 ≤ wgt d r (g.lo + i) (g.lo + k) g.ih ∧ wgt d r (g.lo + i) (g.lo + k) g.ih ≤ 2 ^ 64 := by
  have hs : g.lo + k ∈ upTo d r (g.lo + i) := by
    simp only [upTo, Finset.mem_Icc, hg.lo_eq]; omega
  unfold wgt
  by_cases hG : d.useG = true
  · simp only [hG, ↓reduceIte]
    refine ⟨by positivity, ?_⟩
    obtain ⟨-, hall⟩ := hr.2.2.2.1 hG
    have := (hall _ hs).2.2
    calc (2:ℝ) ^ ((gAt d (g.lo + i) g.ih - gAt d (g.lo + k) g.ih : ℚ) : ℝ)
        ≤ (2:ℝ) ^ ((64:ℕ):ℝ) :=
          Real.rpow_le_rpow_of_exponent_le (by norm_num) (by push_cast; exact_mod_cast this)
      _ = 2 ^ 64 := by norm_num
  · by_cases hGm : d.useGamma = true
    · simp only [hG, hGm, Bool.false_eq_true, ↓reduceIte]
      refine ⟨by positivity, ?_⟩
      obtain ⟨-, h64', h1⟩ := hr.2.2.1 hGm
      have hn : ((g.lo + i : Nat) : ℚ) - ((g.lo + k : Nat) : ℚ) ≤ 64 := by
        have : i ≤ 64 := by have : i < 64 := hi; omega
        push_cast
        have : (k:ℚ) ≥ 0 := Nat.cast_nonneg _
        have : (i:ℚ) ≤ 64 := by exact_mod_cast ‹i ≤ 64›
        linarith
      have hn0 : (0:ℚ) ≤ ((g.lo + i : Nat) : ℚ) - ((g.lo + k : Nat) : ℚ) := by
        push_cast; have : (k:ℚ) ≤ i := by exact_mod_cast hik
        linarith
      calc (2:ℝ) ^ (((gam d g.ih * (((g.lo + i : Nat) : ℚ) - ((g.lo + k : Nat) : ℚ)) : ℚ)) : ℝ)
          ≤ (2:ℝ) ^ ((64:ℕ):ℝ) :=
            Real.rpow_le_rpow_of_exponent_le (by norm_num) (by
              push_cast
              have hq : gam d g.ih * (((g.lo + i : Nat) : ℚ) - ((g.lo + k : Nat) : ℚ)) ≤ 64 := by
                nlinarith [mul_nonneg (sub_nonneg.2 h1) hn0]
              have hq' : ((gam d g.ih * (((g.lo + i : Nat) : ℚ) - ((g.lo + k : Nat) : ℚ)) : ℚ) : ℝ) ≤ 64 := by
                exact_mod_cast hq
              push_cast at hq'
              exact hq')
        _ = 2 ^ 64 := by norm_num
    · simp [hG, hGm]; norm_num

/-- Row `k`, column `c` of the `v` tile (`b_v`): the value, or `+0` past the sequence or the
value columns. -/
def vTile (d : Inst) (g : Geo) (k c : Nat) : BF16 :=
  if g.rowOk k ∧ g.colOk d c then d.vv (((g.lo + k) * d.HV + g.ih) * d.V + (g.iv * d.bk + c))
  else FloatV.ofPat 0

/-- The epilogue magnitude sum over the first `i + 1` rows equals the proof's `Mag` for row `g.lo + i`. -/
theorem epMag_eq (hg : Geo.Ok d g r) (i : Nat) (e : Nat) :
    epMag (i + 1) (gate d r (g.lo + i) g.ih) (PQS d r (g.lo + i) g.ih e : ℝ)
      (fun k => wgt d r (g.lo + i) (g.lo + k) g.ih)
      (fun k => (PA d (g.lo + i) (g.lo + k) g.ih : ℝ))
      (fun k => (L (vAt d (g.lo + k) g.ih e) : ℝ)) = Mag d r (g.lo + i) g.ih e := by
  unfold epMag Mag
  rw [sum_upTo hg (fun s => wgt d r (g.lo + i) s g.ih * (PA d (g.lo + i) s g.ih : ℝ) *
    (L (vAt d s g.ih e) : ℝ)) i]

/-- The epilogue floor sum over the first `i + 1` rows equals the proof's `Flo` for row `g.lo + i`. -/
theorem epFlo_eq (hg : Geo.Ok d g r) (i : Nat) (e : Nat) :
    epFlo (i + 1) (gate d r (g.lo + i) g.ih) (PQS d r (g.lo + i) g.ih e : ℝ)
      (fun k => wgt d r (g.lo + i) (g.lo + k) g.ih)
      (fun k => (PA d (g.lo + i) (g.lo + k) g.ih : ℝ))
      (fun k => (L (vAt d (g.lo + k) g.ih e) : ℝ)) = Flo d r (g.lo + i) g.ih e := by
  unfold epFlo Flo
  rw [sum_upTo hg (fun s => (1 + wgt d r (g.lo + i) s g.ih + (PA d (g.lo + i) s g.ih : ℝ)) *
    (L (vAt d s g.ih e) : ℝ)) i]

/-- The epilogue absolute-product sum over the first `i + 1` rows equals the proof's `Mabs` for row `g.lo + i`. -/
theorem epMabs_eq (hg : Geo.Ok d g r) (i : Nat) (e : Nat) :
    epMabs (i + 1) (fun k => (Aqk d (g.lo + i) (g.lo + k) g.ih : ℝ))
      (fun k => wgt d r (g.lo + i) (g.lo + k) g.ih)
      (fun k => (vAt d (g.lo + k) g.ih e : ℝ)) = Mabs d r (g.lo + i) g.ih e := by
  unfold epMabs Mabs
  rw [sum_upTo hg (fun s => |(Aqk d (g.lo + i) s g.ih : ℝ)| * wgt d r (g.lo + i) s g.ih *
    |(vAt d s g.ih e : ℝ)|) i]

/-- The reference value at row `g.lo + i` is the gated query-state term plus the sum, over rows up to `i`, of `Aqk * wgt * v`, scaled by `sig`. -/
theorem ref_eq (hg : Geo.Ok d g r) (i : Nat) (e : Nat) :
    ref d r (g.lo + i) g.ih e = sig d * (gate d r (g.lo + i) g.ih * (QS d r (g.lo + i) g.ih e : ℝ) +
      ∑ k ∈ Finset.range (i + 1), (Aqk d (g.lo + i) (g.lo + k) g.ih : ℝ) *
        wgt d r (g.lo + i) (g.lo + k) g.ih * (vAt d (g.lo + k) g.ih e : ℝ)) := by
  unfold ref
  rw [sum_upTo hg (fun s => (Aqk d (g.lo + i) s g.ih : ℝ) * wgt d r (g.lo + i) s g.ih *
    (vAt d s g.ih e : ℝ)) i]

/-- Within range, the loaded `v` tile entry is a finite value. -/
theorem vTile_fin (hg : Geo.Ok d g r) {i : Nat} {c : Nat} (_hcol : g.colOk d c)
    (hin : InRange d r (g.lo + i) g.ih (g.iv * d.bk + c)) {k : Nat} (hkB : k < BT) :
    (vTile d g k c).isFinite = true := by
  unfold vTile
  by_cases h : g.rowOk k ∧ g.colOk d c
  · simp only [h]
    refine hin.2.2.1 (g.lo + k) ?_ ?_
    · rw [hg.lo_eq]; omega
    · rw [hg.hi_eq]; have := h.1; unfold Geo.rowOk at this; unfold Geo.lo; simp only [BT] at *; omega
  · simp only [h, ↓reduceIte]; exact bf16_zero_isFinite

/-- For a valid row and column, the `v` tile entry's real value is `vAt` at the matching position. -/
theorem vTile_val (_hg : Geo.Ok d g r) {k c : Nat} (hrow : g.rowOk k) (hcol : g.colOk d c) :
    (vTile d g k c).toReal = (vAt d (g.lo + k) g.ih (g.iv * d.bk + c) : ℝ) := by
  unfold vTile vAt
  simp only [hrow, hcol, and_self, ↓reduceIte]; rfl

/-- A rounded product of magnitude below `2^102` is below `2^103`. -/
theorem m_abs {m p : ℝ} (hp : |p| ≤ 2 ^ 102) (h : |m - p| ≤ 2 ^ (-24 : ℤ) * |p| + 2 ^ (-150 : ℤ)) :
    |m| ≤ 2 ^ 103 := by
  have h1 := abs_sub_abs_le_abs_sub m p
  have h2 : (2 : ℝ) ^ (-24 : ℤ) ≤ 1 / 1000 := by norm_num
  have h3 : (2 : ℝ) ^ (-150 : ℤ) ≤ 1 / 1000 := by norm_num
  have h4 := mul_le_mul_of_nonneg_right h2 (abs_nonneg p)
  generalize (2 : ℝ) ^ (-24 : ℤ) = A at *
  generalize (2 : ℝ) ^ (-150 : ℤ) = B at *
  have : (2:ℝ) ^ 103 = 2 * 2 ^ 102 := by norm_num
  nlinarith [abs_nonneg p]

/-- **One output element of the epilogue.** -/
theorem epi_lane {P : Platform} (hP : Device P) (hd : d.ok) (hg : Geo.Ok d g r)
    {i : Fin BT} {c : Fin d.bk} (hrow : g.rowOk i) (hcol : g.colOk d c)
    (hin : InRange d r (g.lo + i) g.ih (g.iv * d.bk + c))
    {bo : Tensor [BT, d.bk] F32} (hOG : OGate d g r bo)
    {bA2 : Tensor [BT, BT] F32} (hAG : AGate d g r bA2)
    {a16 : Tensor [BT, BT] BF16}
    (hzero : ∀ k : Fin BT, ¬ k ≤ i → bA2.get2 i k = FloatV.ofPat 0)
    (hcv : ∀ k : Fin BT, FloatV.NaNEq (FloatV.cvtWith .rn (bA2.get2 i k)) (a16.get2 i k))
    {bv : Tensor [BT, d.bk] BF16} (hbv : ∀ k : Fin BT, bv.get2 k c = vTile d g k c)
    {Dt : Tensor [BT, d.bk] F32} (hdot : P.dot cfgBF16 a16 bv (tl.zeros [BT, d.bk] .f32) Dt)
    {m1 m2 x : F32} {o16 : BF16}
    (hm1 : FloatV.NaNEq (FloatV.mul (bo.get2 i c) (FloatV.hostF32 d.scale)) m1)
    (hm2 : FloatV.NaNEq (FloatV.mul (Dt.get2 i c) (FloatV.hostF32 d.scale)) m2)
    {fusion : Bool} (hx : AddRel fusion m1 m2 x)
    (hc : FloatV.NaNEq (FloatV.cvtWith .rn x) o16) :
    o16.isFinite = true ∧
      |o16.toReal - ref d r (g.lo + i) g.ih (g.iv * d.bk + c)| ≤
        tol d r (g.lo + i) g.ih (g.iv * d.bk + c) := by
  set t := g.lo + (i : Nat) with ht
  set e := g.iv * d.bk + (c : Nat) with he
  have hrin := hin.rowIn
  obtain ⟨hot0, hot⟩ := hOG i c hrow hcol hin
  -- the scale
  obtain ⟨hsc0, hsc1, hsc2⟩ := hd.scale_ok
  have hσ64 : |d.scale.toReal| ≤ 2 ^ 64 := by
    have : d.scale.toRat ≤ 2 ^ 64 := hsc2
    have h0 : 0 ≤ d.scale.toRat := le_trans (by positivity) hsc1
    rw [FloatV.toReal, abs_of_nonneg (by exact_mod_cast h0)]
    exact_mod_cast this
  obtain ⟨hσhF, hσh⟩ := scale_lane hsc0 hσ64
  have hsig : sig d = (d.scale.toRat : ℝ) := rfl
  have hn : (i : Nat) + 1 ≤ BT := i.isLt
  have hrk : ∀ k : Nat, k < (i : Nat) + 1 → k < BT ∧ k ≤ (i : Nat) := fun k hk => by omega
  -- the families of the chain
  obtain ⟨At, hAt0⟩ : ∃ f : ℕ → ℝ, f = fun k => if h : k < BT then (bA2.get2 i ⟨k, h⟩).toReal else 0 :=
    ⟨_, rfl⟩
  obtain ⟨aa, haa0⟩ : ∃ f : ℕ → ℝ, f = fun k => if h : k < BT then (a16.get2 i ⟨k, h⟩).toReal else 0 :=
    ⟨_, rfl⟩
  have hAt1 : ∀ k : Fin BT, At k = (bA2.get2 i k).toReal := fun k => by simp [hAt0, k.isLt]
  have haa1 : ∀ k : Fin BT, aa k = (a16.get2 i k).toReal := fun k => by simp [haa0, k.isLt]
  have hmk : ∀ a' : ℕ → ℝ, (∀ k < (i : Nat) + 1, |a' k - At k| ≤ 2 ^ (-8 : ℤ) * |At k| + 2 ^ (-134 : ℤ)) →
      EpIn (i + 1) (sig d) (FloatV.hostF32 d.scale).toReal (gate d r t g.ih)
        (gamErr d g.ih) (QS d r t g.ih e : ℝ) (PQS d r t g.ih e : ℝ) (bo.get2 i c).toReal
        (fun k => (Aqk d t (g.lo + k) g.ih : ℝ)) (fun k => wgt d r t (g.lo + k) g.ih)
        (fun k => (PA d t (g.lo + k) g.ih : ℝ)) (fun k => (vAt d (g.lo + k) g.ih e : ℝ))
        (fun k => (L (vAt d (g.lo + k) g.ih e) : ℝ)) At a' := by
    intro a' ha'
    have hmem : ∀ k, k < (i : Nat) + 1 → g.lo + k ∈ upTo d r t := fun k hk => by
      simp only [upTo, Finset.mem_Icc, hg.lo_eq, ht]; omega
    have hΓd : 0 ≤ gamErr d g.ih ∧ gamErr d g.ih ≤ 2 ^ (-9 : ℤ) := by
      unfold gamErr
      by_cases hGm : d.useGamma = true
      · obtain ⟨-, h64, -⟩ := hrin.2.2.1 hGm
        have h64' : |((gam d g.ih : ℚ) : ℝ)| ≤ 64 := by exact_mod_cast h64
        simp only [hGm, ↓reduceIte]
        refine ⟨by positivity, ?_⟩
        have : (1 / 2 ^ 15 : ℝ) * 64 = 2 ^ (-9 : ℤ) := by norm_num
        nlinarith
      · simp [hGm]
    obtain ⟨hGa, hGb⟩ := gate_bounds hg i.isLt hrin
    have hP0 : (0:ℝ) ≤ (PQS d r t g.ih e : ℝ) := by exact_mod_cast PQS_nonneg d _ _ _ _
    refine
      { hσ := ?_, hσh := ?_, hG := hGa, hG64 := hGb, hΓ := hΓd.1, hΓ' := hΓd.2, hPQS := hP0,
        hQS := by exact_mod_cast abs_QS_le d r t g.ih e,
        hw := fun k hk => (wgt_bounds hg (hrk k hk).2 i.isLt hrin).1,
        hw64 := fun k hk => (wgt_bounds hg (hrk k hk).2 i.isLt hrin).2,
        hPA := fun k _ => by exact_mod_cast PA_nonneg d _ _ _,
        hAPA := fun k _ => by exact_mod_cast abs_Aqk_le d _ _ _,
        hLv := fun k _ => by exact_mod_cast L_nonneg _,
        hv := fun k _ => by exact_mod_cast Device.abs_le_liftMag .bf16 _,
        hGPQS := hin.2.2.2.2.2.2.1, hPQS100 := by exact_mod_cast hin.2.2.2.2.2.1,
        hwPA := fun k hk => (hrin.2.2.2.2 _ (hmem k hk)).2,
        hPA100 := fun k hk => by exact_mod_cast (hrin.2.2.2.2 _ (hmem k hk)).1,
        hroom := ?_, hroomσ := ?_, hot := hot, hAt := ?_, ha := ha' }
    · constructor
      · have : (1 / 2 ^ 64 : ℚ) ≤ d.scale.toRat := hsc1
        have : ((1 / 2 ^ 64 : ℚ) : ℝ) ≤ (d.scale.toRat : ℝ) := by exact_mod_cast this
        rw [hsig]; push_cast at this; rw [show (2:ℝ) ^ (-64 : ℤ) = 1 / 2 ^ 64 by norm_num]; exact this
      · rw [hsig]; exact_mod_cast hsc2
    · have h0 : 0 ≤ d.scale.toReal := by
        have : 0 ≤ d.scale.toRat := le_trans (by positivity) hsc1
        unfold FloatV.toReal; exact_mod_cast this
      rw [abs_of_nonneg h0] at hσ64
      have := hσh
      rw [FloatV.toReal, abs_of_nonneg h0] at this
      simpa [hsig, FloatV.toReal] using this
    · rw [epMag_eq hg, epFlo_eq hg]; exact hin.2.2.2.2.2.2.2.2.1
    · rw [epMag_eq hg, epFlo_eq hg]; exact hin.2.2.2.2.2.2.2.2.2
    · intro k hk
      have h := hAG i ⟨k, (hrk k hk).1⟩ hrow (by exact (hrk k hk).2) hrin
      rw [hAt1 ⟨k, (hrk k hk).1⟩]
      exact h.2
  have hAt_room : ∀ k < (i : Nat) + 1, |At k| ≤ 2 ^ 101 := fun k hk =>
    ep_room_a (hmk At (fun k _ => by simp; positivity)) hk
  have hAfin : ∀ k : Fin BT, k ≤ i → (bA2.get2 i k).isFinite = true :=
    fun k hk => (hAG i k hrow hk hrin).1
  have hcvt : ∀ k : Fin BT, k ≤ i → (a16.get2 i k).isFinite = true ∧
      |(a16.get2 i k).toReal - (bA2.get2 i k).toReal| ≤
        2 ^ (-8 : ℤ) * |(bA2.get2 i k).toReal| + 2 ^ (-134 : ℤ) := by
    intro k hk
    have h1 := hAt_room k (by have : (k : Nat) ≤ i := hk; omega)
    rw [hAt1 k] at h1
    exact cvtbf_lane (hAfin k hk) (h1.trans (by norm_num)) (hcv k)
  have hazero : ∀ k : Fin BT, ¬ k ≤ i → a16.get2 i k = FloatV.ofPat 0 := by
    intro k hk
    have h := hcv k
    rw [hzero k hk] at h
    have h2 : (FloatV.cvtWith .rn (FloatV.ofPat 0 : F32) : BF16) = FloatV.ofPat 0 := bf16_cvt_zero
    rw [h2] at h
    exact h.eq (by rfl)
  have ha : ∀ k < (i : Nat) + 1, |aa k - At k| ≤ 2 ^ (-8 : ℤ) * |At k| + 2 ^ (-134 : ℤ) := by
    intro k hk
    have hk' := hrk k hk
    rw [haa1 ⟨k, hk'.1⟩, hAt1 ⟨k, hk'.1⟩]
    exact (hcvt ⟨k, hk'.1⟩ hk'.2).2
  have hE := hmk aa ha
  -- the last dot
  obtain ⟨vaN, hvaN⟩ : ∃ f : ℕ → ℚ, f = fun k =>
      if h : k < BT then (a16.get2 i ⟨k, h⟩).toRat else 0 := ⟨_, rfl⟩
  obtain ⟨vbN, hvbN⟩ : ∃ f : ℕ → ℚ, f = fun k =>
      if h : k < BT then (bv.get2 ⟨k, h⟩ c).toRat else 0 := ⟨_, rfl⟩
  have hvaN1 : ∀ k : Fin BT, vaN k = (a16.get2 i k).toRat := fun k => by simp [hvaN, k.isLt]
  have hvbN1 : ∀ k : Fin BT, vbN k = (bv.get2 k c).toRat := fun k => by simp [hvbN, k.isLt]
  have hafin : ∀ k : Fin BT, (a16.get2 i k).isFinite = true := by
    intro k
    by_cases hk : k ≤ i
    · exact (hcvt k hk).1
    · rw [hazero k hk]; exact bf16_zero_isFinite
  have hbfin : ∀ k : Fin BT, (bv.get2 k c).isFinite = true := fun k => by
    rw [hbv k]; exact vTile_fin hg hcol hin k.isLt
  have hva : ∀ k : Fin BT, DType.val? .bf16 (a16.get2 i k) = some (vaN k) := fun k => by
    rw [hvaN1]; exact val?_bf16.mpr ⟨hafin k, rfl⟩
  have hvb : ∀ k : Fin BT, DType.val? .bf16 (bv.get2 k c) = some (vbN k) := fun k => by
    rw [hvbN1]; exact val?_bf16.mpr ⟨hbfin k, rfl⟩
  have hvaz : ∀ k, (i : Nat) + 1 ≤ k → k < BT → vaN k = 0 := by
    intro k hk hkB
    have := hazero ⟨k, hkB⟩ (by simp only [Fin.le_def]; omega)
    simp [hvaN, hkB, this]
  have hsub : Finset.range ((i : Nat) + 1) ⊆ Finset.range BT := Finset.range_subset_range.mpr hn
  have hsumF : ∀ F : ℚ → ℚ → ℚ, (∀ x, F 0 x = 0) →
      ∑ k : Fin BT, F (vaN k) (vbN k) = ∑ k ∈ Finset.range ((i : Nat) + 1), F (vaN k) (vbN k) := by
    intro F hF
    rw [Fin.sum_univ_eq_sum_range (fun k => F (vaN k) (vbN k)) BT]
    refine (Finset.sum_subset hsub fun k hk hkn => ?_).symm
    have : k < BT := Finset.mem_range.mp hk
    rw [hvaz k (by simpa using hkn) this]; exact hF _
  have hvbR : ∀ k < (i : Nat) + 1, (vbN k : ℝ) = (vAt d (g.lo + k) g.ih e : ℝ) := by
    intro k hk
    have hk' := hrk k hk
    have h1 : vbN k = (bv.get2 ⟨k, hk'.1⟩ c).toRat := hvbN1 ⟨k, hk'.1⟩
    have h2 := hbv ⟨k, hk'.1⟩
    have hrk' : g.rowOk k := by
      have := hrow; unfold Geo.rowOk at *; omega
    rw [h1, h2]
    have := vTile_val hg hrk' hcol
    unfold FloatV.toReal at this
    exact this
  have haaR : ∀ k < (i : Nat) + 1, aa k = (vaN k : ℝ) := by
    intro k hk
    have h1 : vaN k = (a16.get2 i ⟨k, (hrk k hk).1⟩).toRat := hvaN1 ⟨k, (hrk k hk).1⟩
    rw [haa1 ⟨k, (hrk k hk).1⟩, h1]; rfl
  -- magnitude of the dot's products
  have hvbQ : ∀ k < (i : Nat) + 1, vbN k = vAt d (g.lo + k) g.ih e := fun k hk => by
    exact_mod_cast hvbR k hk
  have hLL : (((∑ k : Fin BT, Device.liftMag .bf16 (vaN k) * Device.liftMag .bf16 (vbN k) : ℚ)) : ℝ) ≤
      ∑ k ∈ Finset.range ((i : Nat) + 1), (|aa k| + 2 ^ (-126 : ℤ)) *
        (L (vAt d (g.lo + k) g.ih e) : ℝ) := by
    rw [hsumF (fun a b => Device.liftMag .bf16 a * Device.liftMag .bf16 b) (fun x => by simp)]
    push_cast
    refine Finset.sum_le_sum fun k hk => ?_
    have hk' := Finset.mem_range.mp hk
    have h1 := Device.liftMag_le .bf16 (vaN k)
    rw [Device.liftFloor_bf16] at h1
    have h2 : ((Device.liftMag .bf16 (vaN k) : ℚ) : ℝ) ≤ |aa k| + 2 ^ (-126 : ℤ) := by
      rw [haaR k hk']
      have : ((Device.liftMag .bf16 (vaN k) : ℚ) : ℝ) ≤ ((|vaN k| + 2 ^ (-126 : ℤ) : ℚ) : ℝ) := by
        exact_mod_cast h1
      push_cast at this
      exact this
    rw [hvbQ k hk']
    exact mul_le_mul_of_nonneg_right h2 (by exact_mod_cast L_nonneg _)
  have hroomD := ep_room_D hE
  have hLL101 : (((∑ k : Fin BT, Device.liftMag .bf16 (vaN k) * Device.liftMag .bf16 (vbN k) : ℚ)) : ℝ)
      ≤ 2 ^ 101 := hLL.trans hroomD
  have hLLq : (∑ k : Fin BT, Device.liftMag .bf16 (vaN k) * Device.liftMag .bf16 (vbN k) : ℚ) ≤ 2 ^ 101 := by
    exact_mod_cast hLL101
  have hz : DType.val? .f32 ((tl.zeros [BT, d.bk] .f32).get2 i c) = some 0 := by
    rw [zeros_get2]; exact DType.val?_f32.mpr ⟨f32_zero_isFinite, f32_zero_toRat⟩
  have hrel : Device.relDot16 BT = 66 * 2 ^ (-22 : ℤ) := by unfold Device.relDot16 BT; norm_num
  have habs : Device.absDot16 BT = 66 * 2 ^ (-126 : ℤ) := by unfold Device.absDot16 BT; norm_num
  obtain ⟨vr, hvr, herr⟩ := hP.dotBF16 .tf32 a16 bv _ Dt hdot i c (fun k => vaN k) (fun k => vbN k)
    0 hva hvb hz (by
      simp only [DType.bound, abs_zero, zero_add]
      rw [Format.binary32_maxVal, hrel, habs]
      have h1 : (66 * 2 ^ (-22 : ℤ) : ℚ) ≤ 1 := by norm_num
      have h2 : (66 * 2 ^ (-126 : ℤ) : ℚ) ≤ 1 := by norm_num
      have h3 : (0:ℚ) ≤ ∑ k : Fin BT, Device.liftMag .bf16 (vaN k) * Device.liftMag .bf16 (vbN k) :=
        Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
      have h4 : (2:ℚ) ^ 101 * 4 ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
      nlinarith)
  obtain ⟨hDF, hDv⟩ := DType.val?_f32.mp hvr
  rw [hrel, habs] at herr
  have herr2 : |vr - (0 + ∑ k : Fin BT, vaN k * vbN k)| ≤
      66 * 2 ^ (-22 : ℤ) * (|(0:ℚ)| + ∑ k : Fin BT, Device.liftMag .bf16 (vaN k) *
        Device.liftMag .bf16 (vbN k)) + 66 * 2 ^ (-126 : ℤ) := herr
  rw [hsumF (fun a b => a * b) (fun x => by simp)] at herr2
  have hD : |(Dt.get2 i c).toReal - ∑ k ∈ Finset.range ((i : Nat) + 1), aa k * (vAt d (g.lo + k) g.ih e : ℝ)| ≤
      66 * 2 ^ (-22 : ℤ) * ∑ k ∈ Finset.range ((i : Nat) + 1), (|aa k| + 2 ^ (-126 : ℤ)) *
        (L (vAt d (g.lo + k) g.ih e) : ℝ) + 66 * 2 ^ (-126 : ℤ) := by
    have h1 : ((|vr - (0 + ∑ k ∈ Finset.range ((i : Nat) + 1), vaN k * vbN k)| : ℚ) : ℝ) ≤
        ((66 * 2 ^ (-22 : ℤ) * (|(0:ℚ)| + ∑ k : Fin BT, Device.liftMag .bf16 (vaN k) *
          Device.liftMag .bf16 (vbN k)) + 66 * 2 ^ (-126 : ℤ) : ℚ) : ℝ) := by
      exact_mod_cast herr2
    push_cast at h1
    have hS : (∑ k ∈ Finset.range ((i : Nat) + 1), (vaN k : ℝ) * (vbN k : ℝ)) =
        ∑ k ∈ Finset.range ((i : Nat) + 1), aa k * (vAt d (g.lo + k) g.ih e : ℝ) :=
      Finset.sum_congr rfl fun k hk => by
        have hk' := Finset.mem_range.mp hk
        rw [haaR k hk', hvbR k hk']
    have hDr : (Dt.get2 i c).toReal = (vr : ℝ) := by unfold FloatV.toReal; rw [hDv]
    rw [hS, ← hDr] at h1
    have hLLr : (((∑ k : Fin BT, Device.liftMag .bf16 (vaN k) * Device.liftMag .bf16 (vbN k) : ℚ)) : ℝ) =
        ∑ k : Fin BT, ((Device.liftMag .bf16 (vaN k) : ℚ) : ℝ) * ((Device.liftMag .bf16 (vbN k) : ℚ) : ℝ) := by
      push_cast; rfl
    have hLL' := hLL
    rw [hLLr] at hLL'
    simp only [abs_zero, zero_add] at h1
    have h66 : (0:ℝ) ≤ 66 * 2 ^ (-22 : ℤ) := by positivity
    refine h1.trans ?_
    have := mul_le_mul_of_nonneg_left hLL' h66
    linarith
  -- the chain
  have hroot := ep_room_ot hE
  have hroD := ep_room_Dm hE hD
  obtain ⟨hm1F, hm1b⟩ := mul32_lane hot0 hσhF (hroot.trans (by norm_num)) hm1
  obtain ⟨hm2F, hm2b⟩ := mul32_lane hDF hσhF (hroD.trans (by norm_num)) hm2
  have hxsum : |m1.toReal| + |m2.toReal| ≤ 2 ^ 105 := by
    have := m_abs hroot hm1b
    have := m_abs hroD hm2b
    have : (2:ℝ) ^ 103 + 2 ^ 103 ≤ 2 ^ 105 := by norm_num
    linarith
  obtain ⟨hxF, hxb⟩ := addc32_lane hm1F hm2F hxsum hx
  have hx2 := (ep_room_x hroot hroD hm1b hm2b hxb).2
  obtain ⟨hoF, hob⟩ := cvtbf_lane hxF hx2 hc
  refine ⟨hoF, ?_⟩
  have hfin := ep_final hE hD hm1b hm2b hxb hob
  rw [epMabs_eq hg, epMag_eq hg, epFlo_eq hg] at hfin
  have hr1 : ref d r t g.ih e = _ := ref_eq hg i e
  have hr2 : ref d r (g.lo + i) g.ih e = _ := ref_eq hg i e
  unfold tol
  simp only [hr1]
  have e8 : (1 / 2 ^ 8 : ℝ) = 2 ^ (-8 : ℤ) := by norm_num
  have e13 : (1 / 2 ^ 13 : ℝ) = 2 ^ (-13 : ℤ) := by norm_num
  have e100 : (1 / 2 ^ 100 : ℝ) = 2 ^ (-100 : ℤ) := by norm_num
  have e130 : (1 / 2 ^ 130 : ℝ) = 2 ^ (-130 : ℤ) := by norm_num
  rw [e8, e13, e100, e130]
  exact hfin

end Claims.ChunkFwdO
