import Proof.Trip

/-!
# One trip, all rows

`trip_state`: if the exponents and the rest of a trip are as `ExpoOK` and `TripOps` say for every
row, the new state of the tile satisfies `StateOK` one trip later.
-/

namespace Claims.AttnFwd

open TVA

noncomputable section

/-- The running maximum before a trip is `−∞` or bounded by `Bm X`. -/
theorem state_m0 {d : Inst} {init : Mem} {bh sm τ : Nat} {mt lt : Tensor [128] F32}
    {acc : Tensor [128, d.HEAD_DIM] F32} (hst : StateOK d init bh sm τ mt lt acc)
    (h : HeadOk d init bh) (r : Fin 128) {X : ℝ} (hPar : Par d.HEAD_DIM X)
    (hk : ∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) :
    mt.get1 r = FloatV.negInf ∨ ((mt.get1 r).isFinite = true ∧ |(mt.get1 r).toReal| ≤ Bm X) := by
  rcases hst h r with ⟨-, h0, -⟩ | ⟨-, hf, -, -, hX⟩
  · exact Or.inl h0
  · obtain ⟨mf, hM, -⟩ := hX X hPar hk
    exact Or.inr ⟨hf, hM.hm⟩

/-- The scaled score bound `2^10` of a covered head holds for every attended key. -/
theorem keys_1024 {d : Inst} {init : Mem} {bh n : Nat} (h : HeadOk d init bh) (hn : n < d.N_CTX) :
    ∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ (1024 : ℝ) := by
  intro j hj
  have hjN : j < d.N_CTX := by
    simp only [keys, Finset.mem_filter, Finset.mem_range] at hj; exact hj.1
  have := h.2.2 n j hn hjN
  exact_mod_cast this

/-- **One trip, all rows**: the state after the trip is `StateOK` at `τ + 1`. -/
theorem trip_state {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    {bh sm τ : Nat} (hsm : sm < G d) (hτ : 64 * τ + 64 ≤ d.N_CTX) (hτ1 : τ < 1024)
    {vt : Tensor [64, d.HEAD_DIM] F16}
    (hv : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.v.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (vt.get2 kk e))
    {fusion : Bool} {mt lt : Tensor [128] F32} {acc : Tensor [128, d.HEAD_DIM] F32}
    {m1 s al lij l' : Tensor [128] F32}
    {y p : Tensor [128, 64] F32} {acc1 acc' : Tensor [128, d.HEAD_DIM] F32}
    {p16 : Tensor [128, 64] F16}
    (hE : ∀ (_h : HeadOk d init bh) (r : Fin 128) (X : ℝ), Par d.HEAD_DIM X →
      (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
      ExpoOK d init bh (sm * 128 + r) τ X (mt.get1 r) (m1.get1 r) (fun j => y.get2 r j))
    (hmask : ∀ (_h : HeadOk d init bh) (r : Fin 128) (j : Fin 64), 64 * τ + (j : Nat) ∉ keys d (sm * 128 + r) →
      (p.get2 r j).isFinite = true ∧ |(p.get2 r j).toReal| ≤ 1 / 2 ^ 125 ∧ (p16.get2 r j).toRat = 0)
    (ops : TripOps P fusion d mt lt acc vt m1 s al lij l' y p acc1 acc' p16)
    (hst : StateOK d init bh sm τ mt lt acc) :
    StateOK d init bh sm (τ + 1) m1 l' acc' := by
  intro h r
  have hprev : ∀ X : ℝ, Par d.HEAD_DIM X →
      (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
      ((τ = 0 ∧ mt.get1 r = FloatV.negInf ∧
        lt.get1 r = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) ∧
        ∀ e : Fin d.HEAD_DIM, acc.get2 r e = FloatV.litF32 (FloatV.ofPat 0)) ∨
      (1 ≤ τ ∧ (mt.get1 r).isFinite = true ∧ (lt.get1 r).isFinite = true ∧
        (∀ e : Fin d.HEAD_DIM, (acc.get2 r e).isFinite = true) ∧
        ∃ mf : ℝ, InvM (rowCtx d init bh (sm * 128 + r) h) X τ (mt.get1 r).toReal mf ∧
          InvL (rowCtx d init bh (sm * 128 + r) h) X τ (mt.get1 r).toReal mf (lt.get1 r).toReal ∧
          InvA (rowCtx d init bh (sm * 128 + r) h) X τ (mt.get1 r).toReal mf
            (fun e => if h : e < d.HEAD_DIM then (acc.get2 r ⟨e, h⟩).toReal else 0))) := by
    intro X hX hk
    rcases hst h r with h0 | ⟨h1, hm, hl, ha, hX'⟩
    · exact Or.inl h0
    · exact Or.inr ⟨h1, hm, hl, ha, hX' X hX hk⟩
  have hcore : ∀ X : ℝ, Par d.HEAD_DIM X →
      (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
      (m1.get1 r).isFinite = true ∧ (l'.get1 r).isFinite = true ∧
      (∀ e : Fin d.HEAD_DIM, (acc'.get2 r e).isFinite = true) ∧
      ∃ mf : ℝ, InvM (rowCtx d init bh (sm * 128 + r) h) X (τ + 1) (m1.get1 r).toReal mf ∧
        InvL (rowCtx d init bh (sm * 128 + r) h) X (τ + 1) (m1.get1 r).toReal mf (l'.get1 r).toReal ∧
        InvA (rowCtx d init bh (sm * 128 + r) h) X (τ + 1) (m1.get1 r).toReal mf
          (fun e => if h : e < d.HEAD_DIM then (acc'.get2 r ⟨e, h⟩).toReal else 0) :=
    fun X hX hk => trip_core hd hP hsm hτ hτ1 h r hX hv (hE h r X hX hk) (hmask h r) ops (hprev X hX hk)
  have hn : sm * 128 + (r : Nat) < d.N_CTX := sm_row_lt hd hsm r.isLt
  have hX1024 : Par d.HEAD_DIM (1024 : ℝ) := ⟨hd.head_dim, by norm_num, le_rfl⟩
  have hk1024 : ∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ (1024 : ℝ) := by
    intro j hj
    have hjN : j < d.N_CTX := by
      simp only [keys, Finset.mem_filter, Finset.mem_range] at hj; exact hj.1
    have := h.2.2 (sm * 128 + r) j hn hjN
    exact_mod_cast this
  obtain ⟨hm, hl, ha, -⟩ := hcore 1024 hX1024 hk1024
  exact Or.inr ⟨by omega, hm, hl, ha, fun X hX hk => (hcore X hX hk).2.2.2⟩

end

end Claims.AttnFwd
