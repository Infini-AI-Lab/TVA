import Proof.Trips

/-!
# The unmasked trip

`expo_plain`: the exponents of an unmasked trip, one row (`ExpoOK`). `safe_plain_trip`: the walk of
the trip's body (source lines 70-109 with `STAGE ≠ 2`).
-/

namespace Claims.AttnFwd

open TVA

noncomputable section

/-- **The exponents of an unmasked trip**, one row. -/
theorem expo_plain {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    {bh sm τ : Nat} (hsm : sm < G d) (hτ : 64 * τ + 64 ≤ d.N_CTX)
    (h : HeadOk d init bh) (r : Fin 128) {X : ℝ} (hPar : Par d.HEAD_DIM X)
    (hkeys : ∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X)
    (hall : ∀ j : Fin 64, 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r))
    {c : F32} (hc : c.isFinite = true ∧ 0 < c.toReal ∧
      |c.toReal - (sigma d : ℝ) / Real.log 2| ≤ 31 / 10 * u32 * ((sigma d : ℝ) / Real.log 2))
    {q : Tensor [128, d.HEAD_DIM] F16} {kt : Tensor [64, d.HEAD_DIM] F16}
    (hq : ∀ (r : Fin 128) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.q.ptr (idx d bh (sm * 128 + r) e : Int)).addr = some (q.get2 r e))
    (hk : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.k.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (kt.get2 kk e))
    {fusion : Bool} {mt : Tensor [128] F32} {qk prod y : Tensor [128, 64] F32}
    {mx mc m1 : Tensor [128] F32}
    (hdot : P.dot cfg16 q kt.trans (tl.zeros [128, 64] DType.f32) qk)
    (hm0 : mt.get1 r = FloatV.negInf ∨ ((mt.get1 r).isFinite = true ∧ |(mt.get1 r).toReal| ≤ Bm X))
    (hmx : ∃ t : Tree, t.Covers 64 ∧
      t.Eval (FloatV.MaxRel true) (fun j => if h : j < 64 then qk.get2 r ⟨j, h⟩ else default) (mx.get1 r))
    (hmc : FloatV.NaNEq (FloatV.mul (mx.get1 r) c) (mc.get1 r))
    (hm1 : FloatV.MaxRel true (mt.get1 r) (mc.get1 r) (m1.get1 r))
    (hprod : ∀ j : Fin 64, FloatV.NaNEq (FloatV.mul (qk.get2 r j) c) (prod.get2 r j))
    (hy : ∀ j : Fin 64, AddRel fusion (prod.get2 r j) (FloatV.neg (m1.get1 r)) (y.get2 r j)) :
    ExpoOK d init bh (sm * 128 + r) τ X (mt.get1 r) (m1.get1 r) (fun j => y.get2 r j) := by
  have hσ : 1 / 2 ^ 64 ≤ (sigma d : ℝ) ∧ (sigma d : ℝ) ≤ 2 ^ 64 := by
    obtain ⟨-, hlo, hhi⟩ := hd.scale
    refine ⟨?_, ?_⟩
    · have : ((1 / 2 ^ 64 : ℚ) : ℝ) ≤ (sigma d : ℝ) := by exact_mod_cast hlo
      simpa using this
    · have : (sigma d : ℝ) ≤ ((2 ^ 64 : ℚ) : ℝ) := by exact_mod_cast hhi
      simpa using this
  have hlane := fun j : Fin 64 => lane_dot1 hd hP h hτ hsm hq hk hdot r j
  have hΛ0 : ∀ j : Fin 64, 0 ≤ (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ) := fun j => by
    exact_mod_cast scoreMag_nonneg (d := d) init bh (sm * 128 + r) (64 * τ + j)
  have hΛ : ∀ j : Fin 64, (sigma d : ℝ) * (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ) ≤ X := fun j => by
    have := hkeys _ (hall j)
    exact_mod_cast this
  have hsc : ∀ j : Fin 64, |(score d init bh (sm * 128 + r) (64 * τ + j) : ℝ)| ≤
      (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ) := fun j => by
    have := abs_score_le (d := d) init bh (sm * 128 + r) (64 * τ + j)
    exact_mod_cast this
  obtain ⟨hm1f, hm1B, hmono, hlanes, hwit⟩ := trip_expo (fusion := fusion) (D := d.HEAD_DIM) hd.head_dim
    hPar.hX hσ (qkr := fun j => qk.get2 r j) (prodr := fun j => prod.get2 r j) (yr := fun j => y.get2 r j)
    (sc := fun j => (score d init bh (sm * 128 + r) (64 * τ + j) : ℝ))
    (Λ := fun j => (scoreMag d init bh (sm * 128 + r) (64 * τ + j) : ℝ)) (c := c) (mx := mx.get1 r)
    (mc := mc.get1 r) (m0 := mt.get1 r) (m1 := m1.get1 r)
    (fun j => hlane j) hΛ0 hΛ hsc hc.1 hc.2.1 hc.2.2 hmx hmc hm1 hm0 hprod hy
  refine ⟨hm1f, hm1B, hmono, fun j _ => hlanes j, ?_⟩
  rcases hwit with hw | ⟨j, hj⟩
  · exact Or.inl hw
  · exact Or.inr ⟨j, hall j, hj⟩

/-- **The unmasked trip**: from the state after `τ` blocks (offsets at row `bh·N + 64τ`), the body
reads the next block of `k` and `v`, writes nothing, and leaves the state after `τ + 1` blocks. -/
theorem safe_plain_trip {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d)
    {q : Tensor [128, d.HEAD_DIM] F16} {c : F32}
    (hq : ∀ (r : Fin 128) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.q.ptr (idx d bh (sm * 128 + r) e : Int)).addr = some (q.get2 r e))
    (hc : c.isFinite = true ∧ 0 < c.toReal ∧
      |c.toReal - (sigma d : ℝ) / Real.log 2| ≤ 31 / 10 * u32 * ((sigma d : ℝ) / Real.log 2))
    (τ : Nat) (hτ : 64 * τ + 64 ≤ d.N_CTX) (hτ1 : τ < 1024)
    (hall : ∀ (r : Fin 128) (j : Fin 64), 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r))
    (st : St d) (ho : st.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * τ : Nat) : Int))
    (ho2 : st.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * τ : Nat) : Int))
    (hst : StateOK d init bh sm τ st.2.2.1 st.2.1 st.1) :
    Safe ((launch d).env P) (Rset d) ∅
      (fun x m' => m' = init ∧ StateOK d init bh sm (τ + 1) x.2.2.1 x.2.1 x.1 ∧
        x.2.2.2.1.toInt = ((bh * d.N_CTX + 64 * (τ + 1) : Nat) : Int) ∧
        x.2.2.2.2.toInt = ((bh * d.N_CTX + 64 * (τ + 1) : Nat) : Int)) init
      (plainTrip d q (hostDesc d d.k 64) (hostDesc d d.v 64) c st) := by
  have hbN := row_bound' hd hbh
  have hN32 := hd.n_le
  have hrl := hd.rows_lt
  unfold plainTrip
  refine safe_tile_load hd (b := d.k) (by simpa [Buf.ptr] using hd.align.2.1) hd.lens.2.1 hpre.2.1 64 _ ?_ ?_ ?_ fun kt hkt => ?_
  · rw [ho]; positivity
  · rw [ho]; push_cast; omega
  · exact Set.subset_union_right.trans (Set.subset_union_left)
  refine safe_dot (hP.dotTotal _ _ _ _) fun qk hqk => ?_
  refine safe_max1 (by norm_num) true fun mx hmx => ?_
  refine safe_mul fun mc hmc => ?_
  refine safe_maximum fun m1 hm1 => ?_
  refine safe_mul fun prod hprod => ?_
  refine safe_subc fun y hy => ?_
  refine safe_math (fn := .exp2) (fun i => hP.mathTotal _ _ _) fun p hp => ?_
  refine safe_sub fun s hs => ?_
  refine safe_math (fn := .exp2) (fun i => hP.mathTotal _ _ _) fun al hal => ?_
  refine safe_sum1 (by norm_num) fun lij hl => ?_
  refine safe_mul fun acc1 ha1 => ?_
  refine safe_tile_load hd (b := d.v) (by simpa [Buf.ptr] using hd.align.2.2.1) hd.lens.2.2.1 hpre.2.2.1 64 _ ?_ ?_ ?_ fun vt hvt => ?_
  · rw [ho2]; positivity
  · rw [ho2]; push_cast; omega
  · exact Set.subset_union_right
  refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩ fun p16 hp16 => ?_
  refine safe_dot (hP.dotTotal _ _ _ _) fun acc' hacc' => ?_
  refine safe_mul_add fun l' hl' => ?_
  refine safe_ret ?_
  have hk' : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.k.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (kt.get2 kk e) := by
    intro kk e
    have := hkt kk e
    rw [ho, Int.toNat_natCast] at this
    rw [← this]; unfold idx; congr 3; ring_nf
  have hv' : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.v.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (vt.get2 kk e) := by
    intro kk e
    have := hvt kk e
    rw [ho2, Int.toNat_natCast] at this
    rw [← this]; unfold idx; congr 3; ring_nf
  refine ⟨rfl, ?_, ?_, ?_⟩
  · refine trip_state hd hP hsm hτ hτ1 hv' (fun h r X hX hk => ?_) (fun h r j hj => absurd (hall r j) hj)
      ⟨hp, hs, hal, hl, ha1, hp16, hacc', hl'⟩ hst
    exact expo_plain hd hP hsm hτ h r hX hk (hall r) hc hq hk' hqk (state_m0 hst h r hX hk) (hmx r)
      (hmc (r, ())) (hm1 (r, ())) (fun j => hprod (r, j, ())) (fun j => hy (r, j, ()))
  · exact (IntV.I32.add_nat ho (IntV.I32.toInt_ofNat (by norm_num)) (by omega)).trans (by push_cast; ring_nf)
  · exact (IntV.I32.add_nat ho2 (IntV.I32.toInt_ofNat (by norm_num)) (by omega)).trans (by push_cast; ring_nf)

end

end Claims.AttnFwd
