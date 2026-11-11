import Proof.Lane
import Proof.Tile

/-!
# One trip of the key loop
-/

namespace Claims.AttnFwd

open TVA

noncomputable section

/-- The loop state: `acc`, `l_i`, `m_i` and the two row offsets of the next key and value blocks. -/
abbrev St (d : Inst) :=
  Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32 × I32 × I32

/-- The body of the key loop for `STAGE ≠ 2` (source lines 70–109), as it unfolds in the model. -/
def plainTrip (d : Inst) (q : Tensor [128, d.HEAD_DIM] F16) (desc_k desc_v : TensorDesc .f16)
    (c : F32) (x : St d) : Prog (St d) := do
  let kt ← tl.desc_load (bs := [64, d.HEAD_DIM]) desc_k [x.2.2.2.1, IntV.ofInt 0]
  let qk ← tl.dot cfg16 q kt.trans (tl.zeros [128, 64] DType.f32)
  let mx ← tl.max1 true qk
  let mc ← tl.mul mx (bc c)
  let m1 ← tl.maximum x.2.2.1 mc
  let prod ← tl.mul qk (bc c)
  let y ← tl.subc prod (m1.col 64)
  let p ← tl.exp2 y
  let s ← tl.sub x.2.2.1 m1
  let al ← tl.exp2 s
  let lij ← tl.sum1 p
  let acc1 ← tl.mul x.1 (al.col d.HEAD_DIM)
  let vt ← tl.desc_load (bs := [64, d.HEAD_DIM]) desc_v [x.2.2.2.2, IntV.ofInt 0]
  let p16 : Tensor [128, 64] F16 ← tl.cvt p
  let acc' ← tl.dot cfg16 p16 vt acc1
  let l' ← tl.mul_add x.2.1 al lij
  pure (acc', l', m1, x.2.2.2.1 + IntV.ofNat 64, x.2.2.2.2 + IntV.ofNat 64)

/-- Everything a trip computes after the exponents `y`, as the program's steps give it: the weights,
the rescaling factor, the sum, the rescaled accumulator, the weights in `float16`, the second `dot`
and the new row sum. -/
structure TripOps (P : Platform) (fusion : Bool) (d : Inst)
    (mt lt : Tensor [128] F32) (acc : Tensor [128, d.HEAD_DIM] F32)
    (vt : Tensor [64, d.HEAD_DIM] F16) (m1 s al lij l' : Tensor [128] F32)
    (y p : Tensor [128, 64] F32) (acc1 acc' : Tensor [128, d.HEAD_DIM] F32)
    (p16 : Tensor [128, 64] F16) : Prop where
  hp : ∀ i : Idx [128, 64], P.math .exp2 .binary32 (y.get i) (p.get i)
  hs : ∀ i : Idx [128], FloatV.NaNEq (FloatV.sub (mt.get i) (m1.get i)) (s.get i)
  hal : ∀ i : Idx [128], P.math .exp2 .binary32 (s.get i) (al.get i)
  hl : ∀ i : Fin 128, ∃ t : Tree, t.Covers 64 ∧ t.depth ≤ tl.sumBound P 64 ∧
    t.Eval (AddRel false) (fun j => if h : j < 64 then p.get2 i ⟨j, h⟩ else default) (lij.get1 i)
  ha1 : ∀ i : Idx [128, d.HEAD_DIM],
    FloatV.NaNEq (FloatV.mul (acc.get i) ((al.col d.HEAD_DIM).get i)) (acc1.get i)
  hp16 : ∀ i : Idx [128, 64], FloatV.NaNEq (FloatV.cvtWith .rn (p.get i)) (p16.get i)
  hd2 : P.dot cfg16 p16 vt acc1 acc'
  hl' : ∀ i : Idx [128],
    FloatV.NaNEq (FloatV.add (FloatV.mul (lt.get i) (al.get i)) (lij.get i)) (l'.get i) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (lt.get i) (al.get i) (lij.get i)) (l'.get i))

/-- What the exponents of a trip give for one row: the new maximum `m1` is finite and bounded, at
least the old one, the exponent `y` of every attended lane is within `ξ` of the exact one, which
is at most `ξ`, and `m1` is the old maximum or within `ξ` of an attended lane. -/
def ExpoOK (d : Inst) (init : Mem) (bh n τ : Nat) (X : ℝ) (m0 m1 : F32) (y : Fin 64 → F32) : Prop :=
  m1.isFinite = true ∧ |m1.toReal| ≤ Bm X ∧ (m0.isFinite = true → m0.toReal ≤ m1.toReal) ∧
  (∀ j : Fin 64, 64 * τ + (j : Nat) ∈ keys d n →
    (y j).isFinite = true ∧
    |(y j).toReal - ((sigma d : ℝ) * (score d init bh n (64 * τ + j) : ℝ) / Real.log 2 - m1.toReal)| ≤
      xi d.HEAD_DIM X ∧
    (sigma d : ℝ) * (score d init bh n (64 * τ + j) : ℝ) / Real.log 2 - m1.toReal ≤ xi d.HEAD_DIM X) ∧
  ((m0.isFinite = true ∧ m1.toReal = m0.toReal) ∨
    ∃ j : Fin 64, 64 * τ + (j : Nat) ∈ keys d n ∧
      m1.toReal - xi d.HEAD_DIM X ≤ (sigma d : ℝ) * (score d init bh n (64 * τ + j) : ℝ) / Real.log 2)

/-- A score is at most its magnitude. -/
theorem abs_score_le {d : Inst} (init : Mem) (bh n j : Nat) :
    |score d init bh n j| ≤ scoreMag d init bh n j := by
  unfold score scoreMag
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun e _ => ?_)
  rw [abs_mul]
  exact mul_le_mul (Device.abs_le_liftMag _ _) (Device.abs_le_liftMag _ _) (abs_nonneg _)
    (Device.liftMag_nonneg _ _)

/-- One trip of one row (masked or not): if the row's state before is initial or satisfies the
invariants at `τ`, the exponents are as `ExpoOK` says, the lanes that are not attended have a
negligible weight that rounds to `0` in `float16`, and the rest of the trip is as `TripOps` says,
then the new state is finite and satisfies the invariants at `τ + 1`. -/
theorem trip_core {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    {bh sm τ : Nat} (hsm : sm < G d) (hτ : 64 * τ + 64 ≤ d.N_CTX) (hτ1 : τ < 1024)
    (h : HeadOk d init bh) (r : Fin 128) {X : ℝ} (hPar : Par d.HEAD_DIM X)
    {vt : Tensor [64, d.HEAD_DIM] F16}
    (hv : ∀ (kk : Fin 64) (e : Fin d.HEAD_DIM),
      init.read .f16 (d.v.ptr (idx d bh (64 * τ + kk) e : Int)).addr = some (vt.get2 kk e))
    {fusion : Bool} {mt lt : Tensor [128] F32} {acc : Tensor [128, d.HEAD_DIM] F32}
    {m1 s al lij l' : Tensor [128] F32}
    {y p : Tensor [128, 64] F32} {acc1 acc' : Tensor [128, d.HEAD_DIM] F32}
    {p16 : Tensor [128, 64] F16}
    (hE : ExpoOK d init bh (sm * 128 + r) τ X (mt.get1 r) (m1.get1 r) (fun j => y.get2 r j))
    (hmask : ∀ j : Fin 64, 64 * τ + (j : Nat) ∉ keys d (sm * 128 + r) →
      (p.get2 r j).isFinite = true ∧ |(p.get2 r j).toReal| ≤ 1 / 2 ^ 125 ∧ (p16.get2 r j).toRat = 0)
    (ops : TripOps P fusion d mt lt acc vt m1 s al lij l' y p acc1 acc' p16)
    (hprev : (τ = 0 ∧ mt.get1 r = FloatV.negInf ∧
        lt.get1 r = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) ∧
        ∀ e : Fin d.HEAD_DIM, acc.get2 r e = FloatV.litF32 (FloatV.ofPat 0)) ∨
      (1 ≤ τ ∧ (mt.get1 r).isFinite = true ∧ (lt.get1 r).isFinite = true ∧
        (∀ e : Fin d.HEAD_DIM, (acc.get2 r e).isFinite = true) ∧
        ∃ mf : ℝ, InvM (rowCtx d init bh (sm * 128 + r) h) X τ (mt.get1 r).toReal mf ∧
          InvL (rowCtx d init bh (sm * 128 + r) h) X τ (mt.get1 r).toReal mf (lt.get1 r).toReal ∧
          InvA (rowCtx d init bh (sm * 128 + r) h) X τ (mt.get1 r).toReal mf
            (fun e => if h : e < d.HEAD_DIM then (acc.get2 r ⟨e, h⟩).toReal else 0))) :
    (m1.get1 r).isFinite = true ∧ (l'.get1 r).isFinite = true ∧
      (∀ e : Fin d.HEAD_DIM, (acc'.get2 r e).isFinite = true) ∧
      ∃ mf : ℝ, InvM (rowCtx d init bh (sm * 128 + r) h) X (τ + 1) (m1.get1 r).toReal mf ∧
        InvL (rowCtx d init bh (sm * 128 + r) h) X (τ + 1) (m1.get1 r).toReal mf (l'.get1 r).toReal ∧
        InvA (rowCtx d init bh (sm * 128 + r) h) X (τ + 1) (m1.get1 r).toReal mf
          (fun e => if h : e < d.HEAD_DIM then (acc'.get2 r ⟨e, h⟩).toReal else 0) := by
  have hn : sm * 128 + (r : Nat) < d.N_CTX := sm_row_lt hd hsm r.isLt
  set R := rowCtx d init bh (sm * 128 + r) h with hR
  obtain ⟨hm1f, hm1B, hmono, hlanes, hwit⟩ := hE
  have hninf : (FloatV.negInf : F32).isFinite = false := by decide +kernel
  have hBm1500 : Bm X ≤ 1500 := by have := hPar.hX.2; unfold Bm; nlinarith
  -- the rescaling factor
  have hαall : (al.get1 r).isFinite = true ∧ ∃ ρα : ℝ, 0 ≤ ρα ∧
      ρα ≤ Qα + CA * ((m1.get1 r).toReal - (if τ = 0 then (m1.get1 r).toReal else (mt.get1 r).toReal)) ∧
      |(al.get1 r).toReal - (if τ = 0 then 0 else Real.exp (((mt.get1 r).toReal - (m1.get1 r).toReal) * Real.log 2))| ≤
        ρα * (if τ = 0 then 0 else Real.exp (((mt.get1 r).toReal - (m1.get1 r).toReal) * Real.log 2)) + 1 / 2 ^ 125 ∧
      (τ = 0 → (al.get1 r).toReal = 0) := by
    rcases hprev with ⟨hτ0, h0, -, -⟩ | ⟨hτ1', hm0f, -, -, mf, hM, -, -⟩
    · have hs0 := ops.hs (r, ())
      have hsn : FloatV.sub (mt.get (r, ())) (m1.get (r, ())) = FloatV.negInf := by
        have : mt.get (r, ()) = FloatV.negInf := h0
        rw [this]; exact FloatV.sub_negInf_left (Format.Std.ieee Format.binary32_std) hm1f
      rw [hsn] at hs0
      have hnn : (FloatV.negInf : F32).isNaN = false := by decide +kernel
      have hsn' : s.get (r, ()) = FloatV.negInf := hs0.eq hnn
      have hal0 := ops.hal (r, ())
      rw [hsn'] at hal0
      have : al.get (r, ()) = FloatV.ofPat 0 := hP.exp2F32NegInf _ hal0
      have hzero : (al.get1 r) = FloatV.ofPat 0 := this
      refine ⟨by rw [hzero]; exact FloatV.isFinite_zero_f32, 0, le_rfl, ?_, ?_, fun _ => ?_⟩
      · simp [hτ0]; unfold Qα q22; positivity
      · simp [hτ0, hzero]
      · rw [hzero]; simp
    · have hτne : τ ≠ 0 := by omega
      have hm0le := hmono hm0f
      obtain ⟨hlf, ρα, hρ0, hρq, hρe⟩ := alpha_real hP hm0f hm1f hm0le
        (hM.hm.trans hBm1500) (hm1B.trans hBm1500) (ops.hs (r, ())) (ops.hal (r, ()))
      refine ⟨hlf, ρα, hρ0, ?_, ?_, fun h => absurd h hτne⟩
      · simp [hτne]; exact hρq
      · simpa [hτne, Tensor.get1] using hρe
  obtain ⟨hαf, ρα, hρα0, hραq, hα', hαz⟩ := hαall
  -- the exact weights of the block
  have hxi0 := hPar.xi_nonneg
  have hxi1 := hPar.xi_le
  have hokiff : ∀ j : Fin 64, R.ok (τ * 64 + j) = true ↔ 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r) := by
    intro j
    rw [hR, rowCtx_ok_iff, show τ * 64 + (j : Nat) = 64 * τ + j by ring]
  have hpwj : ∀ j : Fin 64, 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r) →
      R.pw τ (m1.get1 r).toReal j =
      Real.exp (((sigma d : ℝ) * (score d init bh (sm * 128 + r) (64 * τ + j) : ℝ) / Real.log 2 -
        (m1.get1 r).toReal) * Real.log 2) := by
    intro j hj
    have hok : R.ok (τ * 64 + j) = true := (hokiff j).mpr hj
    unfold RowCtx.pw
    rw [ite_eq_left hok]
    have hlog0 : Real.log 2 ≠ 0 := by positivity
    congr 1
    rw [hR]; unfold rowCtx
    simp only
    rw [show τ * 64 + (j : Nat) = 64 * τ + j by ring]
    field_simp
  have hpl : ∀ j : Fin 64, (p.get2 r j).isFinite = true ∧
      |(p.get2 r j).toReal - R.pw τ (m1.get1 r).toReal j| ≤ rhoP d.HEAD_DIM X * R.pw τ (m1.get1 r).toReal j + 1 / 2 ^ 125 ∧
      |(p.get2 r j).toReal| ≤ 2 := by
    intro j
    by_cases hj : 64 * τ + (j : Nat) ∈ keys d (sm * 128 + r)
    · obtain ⟨hyf, hye, hup⟩ := hlanes j hj
      obtain ⟨a, b, c'⟩ := p_real hP hyf (ops.hp (r, j, ())) hxi0 hxi1 hye hup
      rw [hpwj j hj]
      exact ⟨a, b, c'⟩
    · obtain ⟨hf, hb, -⟩ := hmask j hj
      have h0 : R.pw τ (m1.get1 r).toReal j = 0 := by
        unfold RowCtx.pw
        rw [ite_eq_right (by rw [hokiff]; exact hj)]
      rw [h0]
      refine ⟨hf, ?_, hb.trans (by norm_num)⟩
      simpa using hb
  -- the previous state, virtual at the first trip
  set m1r := (m1.get1 r).toReal with hm1r
  have hone : (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) : F32).toRat = 1 := by decide +kernel
  have honef : (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) : F32).isFinite = true := by decide +kernel
  have hone' : (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) : F32).toReal = 1 := by
    unfold FloatV.toReal; rw [hone]; simp
  have hm0v : ∃ (m0v l0v : ℝ) (a0v : ℕ → ℝ) (mf : ℝ), m0v ≤ m1r ∧ InvM R X τ m0v mf ∧
      InvL R X τ m0v mf l0v ∧ InvA R X τ m0v mf a0v ∧
      m0v = (if τ = 0 then m1r else (mt.get1 r).toReal) ∧
      (lt.get1 r).isFinite = true ∧ |(lt.get1 r).toReal| ≤ 2 ^ 18 ∧
      (τ = 0 → l0v = 0 ∧ ∀ e : Fin d.HEAD_DIM, a0v e = 0) ∧
      (1 ≤ τ → l0v = (lt.get1 r).toReal ∧ ∀ e : Fin d.HEAD_DIM, a0v e = (acc.get2 r e).toReal) ∧
      (∀ e : Fin d.HEAD_DIM, (acc.get2 r e).isFinite = true ∧ |(acc.get2 r e).toReal| ≤ 2 ^ 34 ∧ |a0v e| ≤ 2 ^ 34) := by
    rcases hprev with ⟨hτ0, -, hl0, ha0⟩ | ⟨hτ1', hmf0, hlf, haf, mf, hM, hL, hA⟩
    · subst hτ0
      refine ⟨m1r, 0, fun _ => 0, m1r, le_rfl, InvM.zero R hm1B, InvL.zero R X m1r, InvA.zero R X m1r,
        by simp, ?_, ?_, fun _ => ⟨rfl, fun _ => rfl⟩, fun h => absurd h (by omega), fun e => ⟨?_, ?_, by norm_num⟩⟩
      · rw [hl0]; exact honef
      · rw [hl0, hone']; norm_num
      · rw [ha0 e]; decide +kernel
      · rw [ha0 e]
        have : (FloatV.litF32 (FloatV.ofPat 0) : F32).toReal = 0 := by simp
        rw [this]; norm_num
    · refine ⟨(mt.get1 r).toReal, (lt.get1 r).toReal,
        fun e => if h : e < d.HEAD_DIM then (acc.get2 r ⟨e, h⟩).toReal else 0, mf, hmono hmf0, hM, hL, hA,
        by simp; omega, hlf, InvL.abs_le hPar hM hL, fun h => absurd h (by omega), fun _ => ⟨rfl, fun e => ?_⟩,
        fun e => ⟨haf e, ?_, ?_⟩⟩
      · simp [e.isLt]
      · have := InvA.abs_le hPar hM hA (e := e) e.isLt
        simpa [e.isLt] using this
      · simpa [e.isLt] using InvA.abs_le hPar hM hA (e := e) e.isLt
  obtain ⟨m0v, l0v, a0v, mf0, hm0le, hM0, hL0, hA0, hm0veq, hltf, hlt18, hv0, hv1, haccb⟩ := hm0v
  -- the exact factor `α` and the recurrences
  set α : ℝ := (if τ = 0 then 0 else Real.exp (((mt.get1 r).toReal - m1r) * Real.log 2)) with hαdef
  have hα01 : 0 ≤ α ∧ α ≤ 1 := by
    by_cases hτ0 : τ = 0
    · simp [hαdef, hτ0]
    · have hmf0 : (mt.get1 r).isFinite = true := by
        rcases hprev with ⟨h0, -⟩ | ⟨-, hmf0, -⟩
        · exact absurd h0 hτ0
        · exact hmf0
      refine ⟨by simp [hαdef, hτ0]; positivity, ?_⟩
      simp only [hαdef, hτ0, ite_false]
      exact Real.exp_le_one_iff.mpr (by
        have := hmono hmf0
        have hl : 0 < Real.log 2 := Real.log_pos (by norm_num)
        nlinarith)
  have hexpEq : Real.exp ((m0v - m1r) * Real.log 2) * R.Ls m0v τ = α * R.Ls m0v τ := by
    by_cases hτ0 : τ = 0
    · subst hτ0; simp [R.Ls_zero]
    · simp [hαdef, hτ0, hm0veq]
  have hLs : R.Ls m1r (τ + 1) = α * R.Ls m0v τ + ∑ j, R.pw τ m1r j := by
    rw [R.Ls_succ m0v m1r τ, hexpEq]
  have hAs : ∀ e, R.As e m1r (τ + 1) = α * R.As e m0v τ + ∑ j, R.pw τ m1r j * R.v (τ * 64 + j) e := by
    intro e
    rw [R.As_succ e m0v m1r τ]
    by_cases hτ0 : τ = 0
    · subst hτ0; simp [R.As_zero]
    · simp [hαdef, hτ0, hm0veq]
  have hMs : ∀ e, R.Ms e m1r (τ + 1) = α * R.Ms e m0v τ + ∑ j, R.pw τ m1r j * |R.v (τ * 64 + j) e| := by
    intro e
    rw [R.Ms_succ e m0v m1r τ]
    by_cases hτ0 : τ = 0
    · subst hτ0; simp [R.Ms_zero]
    · simp [hαdef, hτ0, hm0veq]
  have hραq' : ρα ≤ Qα + CA * (m1r - m0v) := by rw [hm0veq]; exact hραq
  set αh := (al.get1 r).toReal with hαh
  have hαh1 : |αh| ≤ 2 := by
    have h1 := abs_sub_abs_le_abs_sub αh α
    have h2 : ρα ≤ 1 / 100 := by
      have : m1r - m0v ≤ 3000 := by
        have h3 := abs_le.mp hm1B; have h4 := abs_le.mp hM0.hm; linarith
      have hCA : CA * (m1r - m0v) ≤ CA * 3000 := mul_le_mul_of_nonneg_left this (by unfold CA u32; norm_num)
      have : Qα + CA * 3000 ≤ 1 / 100 := by unfold Qα q22 CA u32; norm_num
      linarith
    have h3 : ρα * α ≤ 1 / 100 := by nlinarith [hα01.1, hα01.2, hρα0]
    rw [abs_of_nonneg hα01.1] at h1
    have : (1 : ℝ) / 2 ^ 125 ≤ 1 / 2 := by norm_num
    linarith
  -- the sum of the weights
  have hpsum : ∑ j : Fin 64, |(p.get2 r j).toReal| ≤ 2 ^ 10 := by
    calc ∑ j : Fin 64, |(p.get2 r j).toReal| ≤ ∑ _j : Fin 64, (2 : ℝ) := Finset.sum_le_sum fun j _ => (hpl j).2.2
      _ = 128 := by simp; norm_num
      _ ≤ 2 ^ 10 := by norm_num
  obtain ⟨hlijf, hlije⟩ := sum_real hP (p := fun j => p.get2 r j) (fun j => (hpl j).1) hpsum (ops.hl r)
  have hlijb : |(lij.get1 r).toReal| ≤ 2 ^ 10 + 1 := by
    have := abs_sub_abs_le_abs_sub (lij.get1 r).toReal (∑ j, (p.get2 r j).toReal)
    have h2 : |∑ j, (p.get2 r j).toReal| ≤ ∑ j, |(p.get2 r j).toReal| := Finset.abs_sum_le_sum_abs _ _
    have h3 : rhoS * ∑ j, |(p.get2 r j).toReal| ≤ 1 / 2 ^ 20 * 2 ^ 10 := by
      refine mul_le_mul ?_ hpsum (Finset.sum_nonneg fun _ _ => abs_nonneg _) (by norm_num)
      unfold rhoS u32; norm_num
    have h4 : (1 : ℝ) / 2 ^ 143 ≤ 1 / 2 := by norm_num
    have h5 : (1 : ℝ) / 2 ^ 20 * 2 ^ 10 ≤ 1 / 2 := by norm_num
    linarith [hpsum, h2, h3, hlije, this, h4, h5]
  -- the new row sum
  have hl0b : |l0v| ≤ 2 ^ 18 := by
    by_cases hτ0 : τ = 0
    · rw [(hv0 hτ0).1]; norm_num
    · rw [(hv1 (by omega)).1]; exact hlt18
  obtain ⟨hl'f, hl'e⟩ := mulAdd_real (fusion := fusion) (ops.hl' (r, ())) hltf hαf hlijf (by
    have h1 : |(lt.get (r, ())).toReal * (al.get (r, ())).toReal| ≤ 2 ^ 19 := by
      rw [abs_mul]
      have h2 : |(lt.get (r, ())).toReal| ≤ 2 ^ 18 := hlt18
      have h3 : |(al.get (r, ())).toReal| ≤ 2 := hαh1
      calc |(lt.get (r, ())).toReal| * |(al.get (r, ())).toReal| ≤ 2 ^ 18 * 2 :=
            mul_le_mul h2 h3 (abs_nonneg _) (by norm_num)
        _ = 2 ^ 19 := by norm_num
    have h4 : |(lij.get (r, ())).toReal| ≤ 2 ^ 10 + 1 := hlijb
    have h5 : (2 : ℝ) ^ 19 + (2 ^ 10 + 1) ≤ 2 ^ 100 := by norm_num
    linarith)
  -- the rescaled accumulator
  have ha1 : ∀ e : Fin d.HEAD_DIM, (acc1.get2 r e).isFinite = true ∧
      |(acc1.get2 r e).toReal - αh * a0v e| ≤ u32 * |αh * a0v e| + 1 / 2 ^ 150 := by
    intro e
    obtain ⟨hf, he⟩ := mul_real (ops.ha1 (r, e, ())) (haccb e).1 hαf (by
      rw [abs_mul]
      have h1 : |(acc.get2 r e).toReal| ≤ 2 ^ 34 := (haccb e).2.1
      calc |FloatV.toReal (acc.get (r, e, ()))| * |FloatV.toReal (al.get1 r)| ≤ 2 ^ 34 * 2 :=
            mul_le_mul h1 hαh1 (abs_nonneg _) (by norm_num)
        _ ≤ 2 ^ 100 := by norm_num)
    refine ⟨hf, ?_⟩
    have hprodEq : (acc.get2 r e).toReal * αh = αh * a0v e := by
      by_cases hτ0 : τ = 0
      · have hz := (hv0 hτ0).2 e
        have hz' : (acc.get2 r e).toReal = 0 := by
          rcases hprev with ⟨-, -, -, ha0⟩ | ⟨h1, -⟩
          · rw [ha0 e]; simp
          · omega
        rw [hz, hz']; ring
      · rw [(hv1 (by omega)).2 e]; ring
    have he2 : |(acc1.get2 r e).toReal - (acc.get2 r e).toReal * αh| ≤
        u32 * |(acc.get2 r e).toReal * αh| + 1 / 2 ^ 150 := he
    rw [hprodEq] at he2
    exact he2
  -- the weights in `float16`
  have hp16 : ∀ j : Fin 64, (p16.get2 r j).isFinite = true ∧
      |(p16.get2 r j).toReal - R.pw τ m1r j| ≤ rho16 d.HEAD_DIM X * R.pw τ m1r j + eta16 ∧
      |(p16.get2 r j).toReal| ≤ 3 := by
    intro j
    obtain ⟨hpf, hpe, hpb⟩ := hpl j
    obtain ⟨hcf16, hce⟩ := cvt16_real (ops.hp16 (r, j, ())) hpf hpb
    refine ⟨hcf16, ?_, ?_⟩
    swap
    · have hce' : |(p16.get2 r j).toReal - (p.get2 r j).toReal| ≤ u16 * |(p.get2 r j).toReal| + e16 := hce
      have h1 := abs_sub_abs_le_abs_sub (p16.get2 r j).toReal (p.get2 r j).toReal
      have h2 : u16 * |(p.get2 r j).toReal| ≤ u16 * 2 := mul_le_mul_of_nonneg_left hpb (by unfold u16; norm_num)
      have h3 : u16 * (2 : ℝ) + e16 ≤ 1 := by unfold u16 e16; norm_num
      linarith
    have hpw0 : 0 ≤ R.pw τ m1r j := R.pw_nonneg τ m1r j
    have h1 := abs_sub_abs_le_abs_sub (p.get2 r j).toReal (R.pw τ m1r j)
    rw [abs_of_nonneg hpw0] at h1
    have hu0 : 0 ≤ u16 := by unfold u16; norm_num
    have h2 : u16 * |(p.get2 r j).toReal| ≤ u16 * ((1 + rhoP d.HEAD_DIM X) * R.pw τ m1r j + 1 / 2 ^ 125) :=
      mul_le_mul_of_nonneg_left (by linarith) hu0
    have h3 : (1 + u16) * (1 / 2 ^ 125 : ℝ) ≤ 1 / 2 ^ 124 := by unfold u16; norm_num
    have h4 : |(p16.get2 r j).toReal - R.pw τ m1r j| ≤
        |(p16.get2 r j).toReal - (p.get2 r j).toReal| + |(p.get2 r j).toReal - R.pw τ m1r j| := by
      have := abs_add_le ((p16.get2 r j).toReal - (p.get2 r j).toReal) ((p.get2 r j).toReal - R.pw τ m1r j)
      rwa [sub_add_sub_cancel] at this
    have e1 : rho16 d.HEAD_DIM X * R.pw τ m1r j = (rhoP d.HEAD_DIM X + u16 * (1 + rhoP d.HEAD_DIM X)) * R.pw τ m1r j := rfl
    have e2 : eta16 = 1 / 2 ^ 25 + 1 / 2 ^ 124 := rfl
    have e3 : e16 = 1 / 2 ^ 25 := rfl
    rw [e1, e2]
    have hce' : |(p16.get2 r j).toReal - (p.get2 r j).toReal| ≤ u16 * |(p.get2 r j).toReal| + e16 := hce
    rw [e3] at hce'
    linarith [hce', hpe, h2, h3, h4]
  -- the lifted magnitudes
  have hvR : ∀ (j : Fin 64) (e : Fin d.HEAD_DIM), R.v (τ * 64 + j) e = ((vt.get2 j e).toRat : ℝ) := by
    intro j e
    have := val16_eq_toRat (hv j e)
    show (val16 init d.v (idx d bh (τ * 64 + j) e) : ℝ) = _
    rw [show τ * 64 + (j : Nat) = 64 * τ + j by ring, this]
  have hvf : ∀ (j : Fin 64) (e : Fin d.HEAD_DIM), (vt.get2 j e).isFinite = true := by
    intro j e
    obtain ⟨x, hx, hxf⟩ := (h.1 (64 * τ + j) e (by have := j.isLt; omega) e.isLt).2.2
    have := hv j e
    rw [hx] at this
    cases this; exact hxf
  have hvb : ∀ (j : Fin 64) (e : Fin d.HEAD_DIM), |R.v (τ * 64 + j) e| ≤ 2 ^ 15 := by
    intro j e
    have := h.2.1 (64 * τ + j) e (by have := j.isLt; omega) e.isLt
    rw [show τ * 64 + (j : Nat) = 64 * τ + j by ring]
    show |(val16 init d.v (idx d bh (64 * τ + j) e) : ℝ)| ≤ _
    exact_mod_cast this
  -- the lifted magnitudes of the weights and of the values
  have hlpb : ∀ j : Fin 64, |(p16.get2 r j).toReal| ≤ ((Device.liftMag .f16 (p16.get2 r j).toRat : ℚ) : ℝ) ∧
      ((Device.liftMag .f16 (p16.get2 r j).toRat : ℚ) : ℝ) ≤ |(p16.get2 r j).toReal| + 1 / 2 ^ 14 := by
    intro j
    have h1 := Device.abs_le_liftMag .f16 (p16.get2 r j).toRat
    have h2 := Device.liftMag_le .f16 (p16.get2 r j).toRat
    rw [Device.liftFloor_f16] at h2
    refine ⟨?_, ?_⟩
    · have := (Rat.cast_le (K := ℝ)).mpr h1
      push_cast at this; simpa [FloatV.toReal] using this
    · have := (Rat.cast_le (K := ℝ)).mpr h2
      push_cast at this; simpa [FloatV.toReal] using this
  have hlvb : ∀ (j : Fin 64) (e : Fin d.HEAD_DIM), |R.v (τ * 64 + j) e| ≤
      ((Device.liftMag .f16 (vt.get2 j e).toRat : ℚ) : ℝ) ∧
      ((Device.liftMag .f16 (vt.get2 j e).toRat : ℚ) : ℝ) ≤ |R.v (τ * 64 + j) e| + 1 / 2 ^ 14 := by
    intro j e
    have h1 := Device.abs_le_liftMag .f16 (vt.get2 j e).toRat
    have h2 := Device.liftMag_le .f16 (vt.get2 j e).toRat
    rw [Device.liftFloor_f16] at h2
    rw [hvR j e]
    refine ⟨?_, ?_⟩
    · have := (Rat.cast_le (K := ℝ)).mpr h1
      push_cast at this; exact this
    · have := (Rat.cast_le (K := ℝ)).mpr h2
      push_cast at this; simpa using this
  have hacc1b : ∀ e : Fin d.HEAD_DIM, |(acc1.get2 r e).toReal| ≤ 2 ^ 36 := by
    intro e
    obtain ⟨-, he⟩ := ha1 e
    have h1 : |αh * a0v e| ≤ 2 ^ 35 := by
      rw [abs_mul]
      calc |αh| * |a0v e| ≤ 2 * 2 ^ 34 := mul_le_mul hαh1 (haccb e).2.2 (abs_nonneg _) (by norm_num)
        _ = 2 ^ 35 := by norm_num
    have h2 := abs_sub_abs_le_abs_sub (acc1.get2 r e).toReal (αh * a0v e)
    have h3 : u32 * |αh * a0v e| ≤ u32 * 2 ^ 35 := mul_le_mul_of_nonneg_left h1 (by unfold u32; norm_num)
    have h4 : u32 * (2 : ℝ) ^ 35 + 1 / 2 ^ 150 ≤ 2 ^ 35 := by unfold u32; norm_num
    linarith
  -- the second `dot`
  have hdot2 : ∀ e : Fin d.HEAD_DIM, (acc'.get2 r e).isFinite = true ∧
      |(acc'.get2 r e).toReal - ((acc1.get2 r e).toReal +
        ∑ j : Fin 64, (p16.get2 r j).toReal * R.v (τ * 64 + j) e)| ≤
        r64 * (|(acc1.get2 r e).toReal| + ∑ j : Fin 64,
          ((Device.liftMag .f16 (p16.get2 r j).toRat : ℚ) : ℝ) *
            ((Device.liftMag .f16 (vt.get2 j e).toRat : ℚ) : ℝ)) + 66 / 2 ^ 126 := by
    intro e
    have hva : ∀ k : Fin 64, DType.val? .f16 (p16.get2 r k) = some (p16.get2 r k).toRat :=
      fun k => DType.val?_f16.mpr ⟨(hp16 k).1, rfl⟩
    have hvb' : ∀ k : Fin 64, DType.val? .f16 (vt.get2 k e) = some (vt.get2 k e).toRat :=
      fun k => DType.val?_f16.mpr ⟨hvf k e, rfl⟩
    have hvc : DType.val? .f32 (acc1.get2 r e) = some (acc1.get2 r e).toRat :=
      DType.val?_f32.mpr ⟨(ha1 e).1, rfl⟩
    have hLR : ((∑ k : Fin 64, Device.liftMag .f16 (p16.get2 r k).toRat *
        Device.liftMag .f16 (vt.get2 k e).toRat : ℚ) : ℝ) ≤ 64 * 4 * 2 ^ 16 := by
      push_cast
      calc ∑ k : Fin 64, ((Device.liftMag .f16 (p16.get2 r k).toRat : ℚ) : ℝ) *
            ((Device.liftMag .f16 (vt.get2 k e).toRat : ℚ) : ℝ) ≤ ∑ _k : Fin 64, (4 : ℝ) * 2 ^ 16 := by
            refine Finset.sum_le_sum fun k _ => ?_
            have h1 := (hlpb k).2
            have h2 := (hlvb k e).2
            have h3 := (hp16 k).2.2
            have h4 := hvb k e
            have h5 := (hlpb k).1
            have h6 := (hlvb k e).1
            have h7 : ((Device.liftMag .f16 (p16.get2 r k).toRat : ℚ) : ℝ) ≤ 4 := by
              have : (1 : ℝ) / 2 ^ 14 ≤ 1 := by norm_num
              linarith
            have h8 : ((Device.liftMag .f16 (vt.get2 k e).toRat : ℚ) : ℝ) ≤ 2 ^ 16 := by
              have : (1 : ℝ) / 2 ^ 14 + 2 ^ 15 ≤ 2 ^ 16 := by norm_num
              linarith
            exact mul_le_mul h7 h8 (by exact_mod_cast Device.liftMag_nonneg .f16 (vt.get2 k e).toRat) (by norm_num)
        _ = 64 * 4 * 2 ^ 16 := by simp; norm_num
    have hLQ : (∑ k : Fin 64, Device.liftMag .f16 (p16.get2 r k).toRat *
        Device.liftMag .f16 (vt.get2 k e).toRat : ℚ) ≤ 64 * 4 * 2 ^ 16 := by exact_mod_cast hLR
    have hvcQ : |(acc1.get2 r e).toRat| ≤ 2 ^ 36 := by
      have := hacc1b e
      have h2 : ((|(acc1.get2 r e).toRat| : ℚ) : ℝ) ≤ ((2 ^ 36 : ℚ) : ℝ) := by
        push_cast; simpa [FloatV.toReal] using this
      exact_mod_cast h2
    have hLQ0 : 0 ≤ (∑ k : Fin 64, Device.liftMag .f16 (p16.get2 r k).toRat *
        Device.liftMag .f16 (vt.get2 k e).toRat : ℚ) :=
      Finset.sum_nonneg fun k _ => mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
    have hrel : Device.relDot16 64 ≤ 1 := by unfold Device.relDot16; norm_num
    have hrel0 : 0 ≤ Device.relDot16 64 := by unfold Device.relDot16; norm_num
    have habs : Device.absDot16 64 ≤ 1 := by unfold Device.absDot16; norm_num
    have hroom : |(acc1.get2 r e).toRat| + (∑ k : Fin 64, Device.liftMag .f16 (p16.get2 r k).toRat *
        Device.liftMag .f16 (vt.get2 k e).toRat) + (Device.relDot16 64 * (|(acc1.get2 r e).toRat| +
        ∑ k : Fin 64, Device.liftMag .f16 (p16.get2 r k).toRat * Device.liftMag .f16 (vt.get2 k e).toRat) +
        Device.absDot16 64) ≤ (⟨DType.f16, DType.f16, DType.f32, InputPrecision.tf32, 0⟩ : DotCfg).acc.bound := by
      show _ ≤ Format.binary32.maxVal
      rw [Format.binary32_maxVal]
      have hS0 : 0 ≤ |(acc1.get2 r e).toRat| + ∑ k : Fin 64, Device.liftMag .f16 (p16.get2 r k).toRat *
          Device.liftMag .f16 (vt.get2 k e).toRat := by positivity
      have hm := mul_le_mul_of_nonneg_right hrel hS0
      nlinarith
    obtain ⟨vr, hvr, herr⟩ := hP.dotF16 .tf32 p16 vt acc1 acc' ops.hd2 r e (fun k => (p16.get2 r k).toRat)
      (fun k => (vt.get2 k e).toRat) (acc1.get2 r e).toRat hva hvb' hvc hroom
    obtain ⟨hfin, hvr'⟩ := DType.val?_f32.mp hvr
    refine ⟨hfin, ?_⟩
    have h1 := (Rat.cast_le (K := ℝ)).mpr herr
    have hx : (acc'.get2 r e).toReal = vr := by unfold FloatV.toReal; rw [hvr']
    rw [hx]
    push_cast at h1
    have e1 : ((Device.relDot16 64 : ℚ) : ℝ) = r64 := by unfold Device.relDot16 r64; push_cast; norm_num
    have e2 : ((Device.absDot16 64 : ℚ) : ℝ) = 66 / 2 ^ 126 := by unfold Device.absDot16; push_cast; norm_num
    rw [e1, e2] at h1
    have e3 : ∑ k : Fin 64, ((p16.get2 r k).toRat : ℝ) * ((vt.get2 k e).toRat : ℝ) =
        ∑ j : Fin 64, (p16.get2 r j).toReal * R.v (τ * 64 + j) e := by
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [hvR k e]; rfl
    rw [e3] at h1
    have e4 : ((acc1.get2 r e).toRat : ℝ) = (acc1.get2 r e).toReal := rfl
    rw [e4] at h1
    simpa [FloatV.toReal] using h1
  -- assembly
  have hsj : ∀ j : Fin 64, R.s (τ * 64 + j) = (sigma d : ℝ) * (score d init bh (sm * 128 + r) (64 * τ + j) : ℝ) := by
    intro j
    rw [hR]; unfold rowCtx; simp only
    rw [show τ * 64 + (j : Nat) = 64 * τ + j by ring]
  have hlog0 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hMnew : InvM R X (τ + 1) m1r mf0 := by
    refine trip_m hPar hτ1 hM0 hm0le hm1B (fun j hj => ?_) ?_
    · rw [hsj j]
      have := (hlanes j ((hokiff j).mp hj)).2.2
      have h2 : (sigma d : ℝ) * (score d init bh (sm * 128 + r) (64 * τ + j) : ℝ) / Real.log 2 ≤ m1r + xi d.HEAD_DIM X := by linarith
      have h3 := mul_le_mul_of_nonneg_right h2 hlog0.le
      rwa [div_mul_cancel₀ _ hlog0.ne'] at h3
    · by_cases hτ0 : τ = 0
      · right
        rcases hwit with ⟨hf0, -⟩ | ⟨j, hjk, hj⟩
        · exfalso
          rcases hprev with ⟨-, h0, -⟩ | ⟨h1, -⟩
          · rw [h0] at hf0; exact absurd hf0 (by decide +kernel)
          · omega
        · refine ⟨j, (hokiff j).mpr hjk, ?_⟩
          rw [hsj j]
          have h3 := mul_le_mul_of_nonneg_right hj hlog0.le
          rwa [div_mul_cancel₀ _ hlog0.ne'] at h3
      · rcases hwit with ⟨hf0, heq⟩ | ⟨j, hjk, hj⟩
        · left
          refine ⟨by omega, ?_⟩
          rw [hm0veq]; simp [hτ0, heq]
        · right
          refine ⟨j, (hokiff j).mpr hjk, ?_⟩
          rw [hsj j]
          have h3 := mul_le_mul_of_nonneg_right hj hlog0.le
          rwa [div_mul_cancel₀ _ hlog0.ne'] at h3
  have hprodl : (lt.get (r, ())).toReal * (al.get (r, ())).toReal = αh * l0v := by
    by_cases hτ0 : τ = 0
    · have hz := hαz hτ0
      have : (al.get (r, ())).toReal = 0 := hz
      rw [this, (hv0 hτ0).1]; ring
    · rw [(hv1 (by omega)).1]; show (lt.get1 r).toReal * αh = _; ring
  refine ⟨hm1f, hl'f, fun e => (hdot2 e).1, mf0, hMnew, ?_, ?_⟩
  · refine trip_l hPar hτ1 hM0 hL0 hα01.1 hα01.2 hLs hm0le hρα0 hραq' hα' (p' := fun j => (p.get2 r j).toReal)
      (fun j => (hpl j).2.1) hlije ?_
    have := hl'e
    simp only [hprodl] at this
    exact this
  · refine trip_a hPar hτ1 hM0 hA0 hα01.1 hα01.2 hm0le hLs hAs hMs hρα0 hραq' hα'
      (a0 := a0v) (a1 := fun e => if h : e < d.HEAD_DIM then (acc1.get2 r ⟨e, h⟩).toReal else 0)
      (a' := fun e => if h : e < d.HEAD_DIM then (acc'.get2 r ⟨e, h⟩).toReal else 0)
      (p16 := fun j => (p16.get2 r j).toReal)
      (lp := fun j => ((Device.liftMag .f16 (p16.get2 r j).toRat : ℚ) : ℝ))
      (lv := fun e j => if h : e < d.HEAD_DIM then
        ((Device.liftMag .f16 (vt.get2 j ⟨e, h⟩).toRat : ℚ) : ℝ) else 0)
      ?_ (fun j => (hp16 j).2.1) (fun j hj => by
        have hj' : 64 * τ + (j : Nat) ∉ keys d (sm * 128 + r) := fun hk => by rw [← hokiff j, hj] at hk; cases hk
        have h0 := (hmask j hj').2.2
        refine ⟨by simp [FloatV.toReal, h0], ?_⟩
        rw [h0]; simp) (fun j => hlpb j) ?_ ?_
    · intro e he
      have he' : e < d.HEAD_DIM := he
      have := (ha1 ⟨e, he'⟩).2
      rw [dite_eq_left he']
      simpa using this
    · intro e he j
      have he' : e < d.HEAD_DIM := he
      rw [dite_eq_left he']
      exact hlvb j ⟨e, he'⟩
    · intro e he
      have he' : e < d.HEAD_DIM := he
      have := (hdot2 ⟨e, he'⟩).2
      simp only [dite_eq_left he']
      simpa using this

end

end Claims.AttnFwd
