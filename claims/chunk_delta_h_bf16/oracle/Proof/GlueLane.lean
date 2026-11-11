import Proof.Lane
import Proof.GlueFam
import Proof.GlueIdx
import Proof.NumV

/-!
# `chunk_delta_h_bf16`: blocks as rows

A family of `64`-row blocks read by the row `c = 64·b + a` (`flat`), and of `64`-column blocks by
the column (`flatC`): the `ℕ`-indexed data of `lane_core`.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- Row `c` (lane `j`) of a family of blocks of `64` rows. -/
def flat {α : Type} [Inhabited α] {n : Nat} (F : ℕ → Tensor [64, n] α) (c j : ℕ) : α :=
  at2 (F (c / 64)) (c % 64) j

/-- Column `c` (row `i`) of a family of blocks of `64` columns. -/
def flatC {α : Type} [Inhabited α] {m : Nat} (F : ℕ → Tensor [m, 64] α) (i c : ℕ) : α :=
  at2 (F (c / 64)) i (c % 64)

/-- Row `64b + a` of a family of blocks is row `a` of block `b`. -/
theorem flat_blk {α : Type} [Inhabited α] {n : Nat} (F : ℕ → Tensor [64, n] α) (b : ℕ) (a : Fin 64)
    (j : Fin n) : flat F (64 * b + a) j = (F b).get2 a j := by
  unfold flat
  rw [show (64 * b + (a : ℕ)) / 64 = b by omega, show (64 * b + (a : ℕ)) % 64 = a by omega]
  exact at2_eq _ a j

/-- Column `64b + a` of a family of blocks is column `a` of block `b`. -/
theorem flatC_blk {α : Type} [Inhabited α] {m : Nat} (F : ℕ → Tensor [m, 64] α) (b : ℕ) (i : Fin m)
    (a : Fin 64) : flatC F i (64 * b + a) = (F b).get2 i a := by
  unfold flatC
  rw [show (64 * b + (a : ℕ)) / 64 = b by omega, show (64 * b + (a : ℕ)) % 64 = a by omega]
  exact at2_eq _ i a

/-- Every row is in a block. -/
theorem split64 (c : ℕ) : ∃ (b : ℕ) (a : Fin 64), c = 64 * b + a :=
  ⟨c / 64, ⟨c % 64, Nat.mod_lt _ (by norm_num)⟩, by simp only; omega⟩

/-- Lane `i` of a vector, by a natural-number index (the default outside the vector). -/
def at1 {n : Nat} {α : Type} [Inhabited α] (T : Tensor [n] α) (i : Nat) : α :=
  if h : i < n then T.get1 ⟨i, h⟩ else default

/-- `at1` on a lane of the vector is the lane. -/
theorem at1_eq {n : Nat} {α : Type} [Inhabited α] (T : Tensor [n] α) (i : Fin n) :
    at1 T i = T.get1 i := by
  simp [at1, i.isLt]

/-- The constants of the update step of the `[K, V]` layout, in the form `lane_core` takes. -/
theorem rD_facts : (0 ≤ rD ∧ rD ≤ 16 / 10 ^ 6) ∧ (0 ≤ aD ∧ aD ≤ 2 ^ (-110 : ℤ)) ∧
    rD ≤ rD + 3 * u32 ∧ aD ≤ (1 + rD + 3 * u32) * aD + 4 * e32 := by
  rw [rD_eq, aD_eq]
  unfold u32 e32
  refine ⟨⟨by positivity, by norm_num⟩, ⟨by positivity, by norm_num⟩, by norm_num, ?_⟩
  have : (0 : ℝ) ≤ 66 * 2 ^ (-126 : ℤ) := by positivity
  nlinarith [show (0 : ℝ) ≤ 66 * 2 ^ (-22 : ℤ) + 3 * 2 ^ (-24 : ℤ) from by positivity,
    show (0 : ℝ) ≤ 4 * 2 ^ (-150 : ℤ) from by positivity]

/-- The constants of the update step of the `[V, K]` layout. -/
theorem vf_consts : (0 ≤ rD + 4 * u32 ∧ rD + 4 * u32 ≤ 16 / 10 ^ 6) ∧
    (0 ≤ (1 + 3 * u32) * aD + 4 * e32 ∧ (1 + 3 * u32) * aD + 4 * e32 ≤ 2 ^ (-110 : ℤ)) ∧
    (0 ≤ 3 * u32 ∧ 3 * u32 ≤ rD + 3 * u32) ∧
    (1 + 3 * u32) * aD + 4 * e32 ≤ (1 + rD + 3 * u32) * aD + 4 * e32 := by
  rw [rD_eq, aD_eq]
  unfold u32 e32
  refine ⟨⟨by positivity, by norm_num⟩, ⟨by positivity, by norm_num⟩, ⟨by positivity, by norm_num⟩, ?_⟩
  have : (0 : ℝ) ≤ 66 * 2 ^ (-126 : ℤ) := by positivity
  nlinarith [show (0 : ℝ) ≤ 66 * 2 ^ (-22 : ℤ) from by positivity]

/-- A lane of `tl.zeros` is a finite zero. -/
theorem zeros_lane {m n : Nat} (i : Fin m) (jj : Fin n) : ((tl.zeros [m, n] .f32).get2 i jj).isFinite = true ∧
    ((tl.zeros [m, n] .f32).get2 i jj).toReal = 0 := by
  obtain ⟨hf, hr⟩ := DType.val?_f32.1 (DType.val?_zero_f32)
  have e : (tl.zeros [m, n] .f32).get2 i jj = FloatV.ofPat 0 := rfl
  rw [e]
  exact ⟨hf, by unfold FloatV.toReal; rw [hr]; simp⟩

/-- **One column of a trip (`[K, V]` layout), from the blocks' tiles.** `lane_core` at lane `jj`. -/
theorem lane_nonvf_fam {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {init m5 : Mem}
    {n hv jv t : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n) (jj : Fin BV)
    (hin : inBlk d jv (BV * jv + jj)) (hV : Vals init d) (hSt : StOk m5 d n hv (BV * jv + jj) t)
    {YY : Nat → Nat → Nat → F32}
    (Bst : ℕ → Tensor [64, BV] F32) (Xh Xs : ℕ → Tensor [64, BV] BF16)
    (hrep : ∀ b, b < nbOf d → ∀ a jj', (Bst b).get2 a jj' = YY jj' t (64 * b + a))
    (hYf : ∀ c, c < d.K → (YY jj t c).isFinite = true)
    (hpad : ∀ c, d.K ≤ c → c < 64 * nbOf d → (YY jj t c).isFinite = true ∧
      |(YY jj t c).toReal| ≤ padB init d n hv t)
    (hpb : padB init d n hv t ≤ 2 ^ 100)
    (hstore : ∀ b, b < nbOf d → ∀ a jj', FloatV.NaNEq (FloatV.cvt ((Bst b).get2 a jj')) ((Xh b).get2 a jj'))
    (hsread : ∀ b, b < nbOf d → ∀ (a : Fin 64) (jj' : Fin BV), 64 * b + a < d.K → BV * jv + jj' < d.V →
      rdB m5 d.h (hIdx d (cbase d n + t) hv (64 * b + a) (BV * jv + jj')) = some ((Xh b).get2 a jj'))
    (Bw : ℕ → Tensor [BT, 64] BF16) (Rr : ℕ → Tensor [BT, BV] F32)
    (hW : ∀ b, b < nbOf d → WTile d init n hv t (64 * b) (Bw b))
    (hX : ∀ b, b < nbOf d → ∀ a jj', FloatV.NaNEq (FloatV.cvt ((Bst b).get2 a jj')) ((Xs b).get2 a jj'))
    (hdotw : ∀ b, b < nbOf d → E.P.dot cfgBF16 (Bw b) (Xs b) (Rr b) (Rr (b + 1)))
    (hR0 : Rr 0 = tl.zeros [BT, BV] .f32)
    {bu : Tensor [BT, BV] BF16} {uf bv5 : Tensor [BT, BV] F32} {xvt : Tensor [BT, BV] BF16}
    (hbu : UTile d init n hv jv t bu)
    (hconv : ∀ i jj', ConvRel E.P .bf16 .f32 (bu.get2 i jj') (uf.get2 i jj'))
    (hsub : ∀ i jj', AddRel E.fusion (uf.get2 i jj') (FloatV.neg ((Rr (nbOf d)).get2 i jj')) (bv5.get2 i jj'))
    (hvc : ∀ i jj', FloatV.NaNEq (FloatV.cvt (bv5.get2 i jj')) (xvt.get2 i jj'))
    (hvr : ∀ (i : Fin BT) (jj' : Fin BV), BT * t + i < lenOf d n → BV * jv + jj' < d.V →
      rdB m5 d.vn (uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv (BV * jv + jj')) = some (xvt.get2 i jj'))
    {bv6 : Tensor [BT, BV] F32} {vb : Tensor [BT, BV] BF16} {gt : Tensor [] F32}
    {Sb : ℕ → Tensor [64, BV] F32}
    (hvbc : ∀ i jj', FloatV.NaNEq (FloatV.cvt (bv6.get2 i jj')) (vb.get2 i jj'))
    (hgate : (d.useG = true ∧ GFacts d E.P init n hv t bv5 bv6 gt ∧
        ∀ b, b < nbOf d → ∀ i, FloatV.NaNEq (FloatV.mul ((Bst b).get i) gt.item) ((Sb b).get i)) ∨
      (d.useG = false ∧ bv6 = bv5 ∧ ∀ b, b < nbOf d → Sb b = Bst b))
    (Bk : ℕ → Tensor [64, BT] BF16) (Yb : ℕ → Tensor [64, BV] F32)
    (hK : ∀ b, b < nbOf d → KTile d init n hv t (64 * b) (Bk b) ∧
      E.P.dot cfgBF16 (Bk b) vb (Sb b) (Yb b)) :
    (∀ i, i < cl d n t →
      |(at2 xvt i jj).toReal - vrefS init m5 d n hv (BV * jv + jj) t i| ≤
          tolV init m5 d n hv (BV * jv + jj) t i ∧ (at2 xvt i jj).isFinite = true) ∧
    ((∀ c, c < d.K → (YY jj t c).isFinite = true ∧ (flat Yb c jj).isFinite = true) ∧
      colNorm d.K (fun c => (flat Yb c jj).toReal - Step init m5 d n hv (BV * jv + jj) t
        (fun c' => (YY jj t c').toReal) c) ≤ errStep init m5 d n hv (BV * jv + jj) t) ∧
    (∀ c, d.K ≤ c → c < 64 * nbOf d → (flat Yb c jj).isFinite = true ∧
      |(flat Yb c jj).toReal| ≤ padB init d n hv (t + 1)) := by
  obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
  obtain ⟨⟨hρ0, hρ1⟩, ⟨hα0, hα1⟩, hρ2', hα2'⟩ := rD_facts
  have hjV : BV * jv + jj < d.V := hin.2.2
  have hb_of : ∀ (b : ℕ) (a : Fin 64) (c : ℕ), c < 64 * nbOf d → c = 64 * b + a → b < nbOf d := by
    intro b a c hc hcb; omega
  refine lane_core (jv := jv) (fusion := E.fusion) hd hP.exp2F32 hn hhv ht hin hV hSt hnb hnb'
    (Y := fun c => YY jj t c) (xh := fun c => flat Xh c jj) (xs := fun c => flat Xs c jj)
    (hxh := ?hxh) (hxhc := ?hxhc) (hxsc := ?hxsc) (hYf := hYf) (hpad := hpad) (hpb := hpb)
    (wt := flatC Bw) (bu := fun i => at2 bu i jj) (xv := fun i => at2 xvt i jj)
    (uf := fun i => at2 uf i jj) (bv5 := fun i => at2 bv5 i jj) (rr := fun b i => at2 (Rr b) i jj)
    (hwr := ?hwr) (hwz := ?hwz) (hbu := ?hbu) (huz := ?huz) (huf := ?huf) (hr0 := ?hr0)
    (hdot := ?hdot) (hsub := ?hsub) (hxvc := ?hxvc) (_hxvr := ?hxvr)
    (bv6 := fun i => at2 bv6 i jj) (vb := fun i => at2 vb i jj) (gt := gt.item)
    (S := fun c => flat Sb c jj) (hvbc := ?hvbc) (hgate := ?hgate)
    (kk := fun i c => flat Bk c i) (Yn := fun c => flat Yb c jj) (hkr := ?hkr) (hkz := ?hkz)
    (ρ2 := rD) (α2 := aD) (hρ2 := ⟨hρ0, hρ1⟩) (hα2 := ⟨hα0, hα1⟩) (ρP := rD) (αP := aD)
    (hρP := ⟨hρ0, hρ2'⟩) (hαP := hα2') (hupdP := ?hupdP) (hupd := ?hupd)
  case hxh =>
    intro c hc
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ (by omega) rfl
    rw [flat_blk]
    exact hsread b hb a jj hc hjV
  case hxhc =>
    intro c hc
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ (by omega) rfl
    rw [flat_blk, ← hrep b hb a jj]
    exact hstore b hb a jj
  case hxsc =>
    intro c hc
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ hc rfl
    rw [flat_blk, ← hrep b hb a jj]
    exact hX b hb a jj
  case hwr =>
    intro i hi c hc
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ (by omega) rfl
    have hi64 : i < BT := by simp only [cl, BT] at hi ⊢; omega
    rw [show flatC Bw i (64 * b + a) = (Bw b).get2 ⟨i, hi64⟩ a from flatC_blk Bw b ⟨i, hi64⟩ a]
    exact ((hW b hb) ⟨i, hi64⟩ a).1 ⟨cl_lt hi, hc⟩
  case hwz =>
    intro i hi c hc hcond
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ hc rfl
    rw [show flatC Bw i (64 * b + a) = (Bw b).get2 ⟨i, hi⟩ a from flatC_blk Bw b ⟨i, hi⟩ a]
    refine ((hW b hb) ⟨i, hi⟩ a).2 fun ⟨h1, h2⟩ => ?_
    simp only [cl, BT] at hcond h1
    omega
  case hbu =>
    intro i hi
    have hi64 : i < BT := by simp only [cl, BT] at hi ⊢; omega
    rw [show at2 bu i jj = bu.get2 ⟨i, hi64⟩ jj from at2_eq bu ⟨i, hi64⟩ jj]
    exact (hbu ⟨i, hi64⟩ jj).1 ⟨cl_lt hi, hjV⟩
  case huz =>
    intro i hi hi64
    rw [show at2 bu i jj = bu.get2 ⟨i, hi64⟩ jj from at2_eq bu ⟨i, hi64⟩ jj]
    refine (hbu ⟨i, hi64⟩ jj).2 fun ⟨h1, h2⟩ => ?_
    simp only [cl, BT] at hi h1
    omega
  case huf =>
    intro i hi
    rw [show at2 bu i jj = bu.get2 ⟨i, hi⟩ jj from at2_eq bu ⟨i, hi⟩ jj,
      show at2 uf i jj = uf.get2 ⟨i, hi⟩ jj from at2_eq uf ⟨i, hi⟩ jj]
    exact hconv _ _
  case hr0 =>
    intro i hi
    rw [hR0, show at2 (tl.zeros [BT, BV] .f32) i jj = (tl.zeros [BT, BV] .f32).get2 ⟨i, hi⟩ jj from
      at2_eq _ ⟨i, hi⟩ jj]
    exact zeros_lane _ _
  case hsub =>
    intro i hi
    rw [show at2 uf i jj = uf.get2 ⟨i, hi⟩ jj from at2_eq uf ⟨i, hi⟩ jj,
      show at2 (Rr (nbOf d)) i jj = (Rr (nbOf d)).get2 ⟨i, hi⟩ jj from at2_eq _ ⟨i, hi⟩ jj,
      show at2 bv5 i jj = bv5.get2 ⟨i, hi⟩ jj from at2_eq bv5 ⟨i, hi⟩ jj]
    exact hsub _ _
  case hxvc =>
    intro i hi
    rw [show at2 bv5 i jj = bv5.get2 ⟨i, hi⟩ jj from at2_eq bv5 ⟨i, hi⟩ jj,
      show at2 xvt i jj = xvt.get2 ⟨i, hi⟩ jj from at2_eq xvt ⟨i, hi⟩ jj]
    exact hvc _ _
  case hxvr =>
    intro i hi
    have hi64 : i < BT := by simp only [cl, BT] at hi ⊢; omega
    rw [show at2 xvt i jj = xvt.get2 ⟨i, hi64⟩ jj from at2_eq xvt ⟨i, hi64⟩ jj]
    exact hvr ⟨i, hi64⟩ jj (cl_lt hi) hjV
  case hvbc =>
    intro i hi
    rw [show at2 bv6 i jj = bv6.get2 ⟨i, hi⟩ jj from at2_eq bv6 ⟨i, hi⟩ jj,
      show at2 vb i jj = vb.get2 ⟨i, hi⟩ jj from at2_eq vb ⟨i, hi⟩ jj]
    exact hvbc _ _
  case hkr =>
    intro i hi c hc
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ (by omega) rfl
    have hi64 : i < BT := by simp only [cl, BT] at hi ⊢; omega
    rw [show flat Bk (64 * b + a) i = (Bk b).get2 a ⟨i, hi64⟩ from flat_blk Bk b a ⟨i, hi64⟩]
    exact ((hK b hb).1 a ⟨i, hi64⟩).1 ⟨hc, cl_lt hi⟩
  case hkz =>
    intro i hi c hc hcond
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ hc rfl
    rw [show flat Bk (64 * b + a) i = (Bk b).get2 a ⟨i, hi⟩ from flat_blk Bk b a ⟨i, hi⟩]
    refine ((hK b hb).1 a ⟨i, hi⟩).2 fun ⟨h1, h2⟩ => ?_
    simp only [cl, BT] at hcond h2
    omega
  case hdot =>
    intro i hi b hb hfin hacc hroom
    have e1 : at2 (Rr b) i jj = (Rr b).get2 ⟨i, hi⟩ jj := at2_eq _ ⟨i, hi⟩ jj
    have e2 : at2 (Rr (b + 1)) i jj = (Rr (b + 1)).get2 ⟨i, hi⟩ jj := at2_eq _ ⟨i, hi⟩ jj
    simp only [e1] at hacc hroom
    simp only [e1, e2]
    exact dot_lane_nat hP.dotBF16 (hdotw b hb) ⟨i, hi⟩ jj (fun k => flatC Bw i (64 * b + k))
      (fun k => flat Xs (64 * b + k) jj) (fun k => flatC_blk Bw b ⟨i, hi⟩ k)
      (fun k => flat_blk Xs b k jj) hfin hacc hroom
  case hupdP =>
    intro c hcK hc hSf hvbf hroom
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ hc rfl
    have hz : ∀ k : Fin 64, ZB ((Bk b).get2 a ⟨k, by have := k.isLt; unfold BT; omega⟩) := fun k =>
      ((hK b hb).1 a _).2 fun ⟨h, _⟩ => by omega
    have e1 : flat Sb (64 * b + a) jj = (Sb b).get2 a jj := flat_blk Sb b a jj
    have e2 : flat Yb (64 * b + a) jj = (Yb b).get2 a jj := flat_blk Yb b a jj
    simp only [e1] at hSf hroom
    simp only [e1, e2]
    have hs0 : ∑ k ∈ Finset.range 64, lmB (flat Bk (64 * b + a) k) * lmB (at2 vb k jj) = 0 :=
      Finset.sum_eq_zero fun k hk => by
        have hk64 : k < BT := by have := Finset.mem_range.1 hk; unfold BT; omega
        rw [show flat Bk (64 * b + a) k = (Bk b).get2 a ⟨k, hk64⟩ from flat_blk Bk b a ⟨k, hk64⟩,
          lmB_zero (hz ⟨k, Finset.mem_range.1 hk⟩).2, zero_mul]
    have hs1 : ∑ k ∈ Finset.range 64, (flat Bk (64 * b + a) k).toReal * (at2 vb k jj).toReal = 0 :=
      Finset.sum_eq_zero fun k hk => by
        have hk64 : k < BT := by have := Finset.mem_range.1 hk; unfold BT; omega
        rw [show flat Bk (64 * b + a) k = (Bk b).get2 a ⟨k, hk64⟩ from flat_blk Bk b a ⟨k, hk64⟩,
          (hz ⟨k, Finset.mem_range.1 hk⟩).2, zero_mul]
    have hdl := dot_lane_nat hP.dotBF16 (hK b hb).2 a jj (fun k => flat Bk (64 * b + a) k)
      (fun k => at2 vb k jj) (fun k => flat_blk Bk b a ⟨k, by have := k.isLt; unfold BT; omega⟩)
      (fun k => at2_eq vb ⟨k, by have := k.isLt; unfold BT; omega⟩ jj)
      (fun k hk => ⟨(hz ⟨k, hk⟩).1 |> fun h => by
        have hk64 : k < BT := by unfold BT; omega
        rw [show flat Bk (64 * b + a) k = (Bk b).get2 a ⟨k, hk64⟩ from flat_blk Bk b a ⟨k, hk64⟩]; exact h,
        hvbf k hk⟩) hSf (by rw [hs0, add_zero]; exact hroom)
    rw [hs0, hs1, add_zero, add_zero] at hdl
    refine ⟨hdl.1, ?_⟩
    have h1 := abs_sub_abs_le_abs_sub (Yb b |>.get2 a jj).toReal (Sb b |>.get2 a jj).toReal
    have h2 := hdl.2
    rw [add_mul, one_mul]
    linarith
  case hupd =>
    intro c hc hSf hkv hroom
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ hc rfl
    have e1 : flat Sb (64 * b + a) jj = (Sb b).get2 a jj := flat_blk Sb b a jj
    have e2 : flat Yb (64 * b + a) jj = (Yb b).get2 a jj := flat_blk Yb b a jj
    simp only [e1] at hSf hroom
    simp only [e1, e2]
    exact dot_lane_nat hP.dotBF16 (hK b hb).2 a jj (fun k => flat Bk (64 * b + a) k)
      (fun k => at2 vb k jj) (fun k => flat_blk Bk b a ⟨k, by have := k.isLt; unfold BT; omega⟩)
      (fun k => at2_eq vb ⟨k, by have := k.isLt; unfold BT; omega⟩ jj) hkv hSf hroom
  case hgate =>
    rcases hgate with ⟨hG, ⟨bgl, bg, xsg, e, hbgl, hbg, hsubg, hexp, hmulg, hgtexp⟩, hSn⟩ | ⟨hG, hbv6, hSb⟩
    · refine Or.inl ⟨hG, bgl, hbgl, hgtexp, ⟨fun i => at1 bg i, fun i => at1 xsg i, fun i => at1 e i, ?_, ?_, ?_, ?_⟩, ?_⟩
      · intro i hi
        beta_reduce
        have hi64 : i < BT := by simp only [cl, BT] at hi ⊢; omega
        rw [show at1 bg i = bg.get1 ⟨i, hi64⟩ from at1_eq bg ⟨i, hi64⟩]
        exact ((hbg ⟨i, hi64⟩).1 (cl_lt hi))
      · intro i hi
        beta_reduce
        have hiB : i < BT := hi
        rw [show at1 bg i = bg.get1 ⟨i, hiB⟩ from at1_eq bg ⟨i, hiB⟩,
          show at1 xsg i = xsg.get1 ⟨i, hiB⟩ from at1_eq xsg ⟨i, hiB⟩]
        exact hsubg ⟨i, hiB⟩
      · intro i hi
        beta_reduce
        have hiB : i < BT := hi
        rw [show at1 xsg i = xsg.get1 ⟨i, hiB⟩ from at1_eq xsg ⟨i, hiB⟩,
          show at1 e i = e.get1 ⟨i, hiB⟩ from at1_eq e ⟨i, hiB⟩]
        exact hexp ⟨i, hiB⟩
      · intro i hi
        beta_reduce
        have hiB : i < BT := hi
        rw [show at2 bv5 i jj = bv5.get2 ⟨i, hiB⟩ jj from at2_eq bv5 ⟨i, hiB⟩ jj,
          show at2 bv6 i jj = bv6.get2 ⟨i, hiB⟩ jj from at2_eq bv6 ⟨i, hiB⟩ jj]
        have hm := hmulg ⟨i, hiB⟩ jj
        have hiff : (BT * t + i < lenOf d n) ↔ i < cl d n t := by simp only [cl, BT]; omega
        by_cases hc : i < cl d n t
        · rw [ite_eq_left hc, show at1 e i = e.get1 ⟨i, hiB⟩ from at1_eq e ⟨i, hiB⟩]
          rw [ite_eq_left (hiff.2 hc)] at hm; exact hm
        · rw [ite_eq_right hc]
          rw [ite_eq_right (fun h => hc (hiff.1 h))] at hm; exact hm
      · intro c hc
        obtain ⟨b, a, rfl⟩ := split64 c
        have hb := hb_of b a _ hc rfl
        rw [flat_blk, ← hrep b hb a jj]
        exact hSn b hb (a, jj, ())
    · refine Or.inr ⟨hG, fun i hi => ?_, fun c hc => ?_⟩
      · rw [hbv6]
      · obtain ⟨b, a, rfl⟩ := split64 c
        have hb := hb_of b a _ hc rfl
        rw [flat_blk, hSb b hb, hrep b hb a jj]

end Claims.ChunkDeltaH
