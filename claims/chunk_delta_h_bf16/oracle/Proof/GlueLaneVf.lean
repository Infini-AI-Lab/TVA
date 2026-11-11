import Proof.GlueLane
import Proof.GlueScale

/-!
# `chunk_delta_h_bf16`: one column of a trip (`[V, K]` layout)

As `GlueLane`, for the state in the `[V, K]` layout: the rows of a block are the lanes of the tile's
first axis, the update is a dot into zeros and a fusable sum.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- Row `c` (lane `j`) of a family of `[BV, 64]` blocks. -/
def flatV {α : Type} [Inhabited α] (F : ℕ → Tensor [BV, 64] α) (c j : ℕ) : α :=
  at2 (F (c / 64)) j (c % 64)

/-- Row `64b + a` of a family of `[BV, 64]` blocks is element `(j, a)` of block `b`. -/
theorem flatV_blk {α : Type} [Inhabited α] (F : ℕ → Tensor [BV, 64] α) (b : ℕ) (a : Fin 64)
    (j : Fin BV) : flatV F (64 * b + a) j = (F b).get2 j a := by
  unfold flatV
  rw [show (64 * b + (a : ℕ)) / 64 = b by omega, show (64 * b + (a : ℕ)) % 64 = a by omega]
  exact at2_eq _ j a

/-- **One column of a trip (`[V, K]` layout), from the blocks' tiles.** `lane_core` at lane `jj`. -/
theorem lane_vf_fam {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {init m5 : Mem}
    {n hv jv t : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n) (jj : Fin BV)
    (hin : inBlk d jv (BV * jv + jj)) (hV : Vals init d) (hSt : StOk m5 d n hv (BV * jv + jj) t)
    {YY : Nat → Nat → Nat → F32}
    (Bst : ℕ → Tensor [BV, 64] F32) (Xh : ℕ → Tensor [BV, 64] BF16) (Xs : ℕ → Tensor [64, BV] BF16)
    (hrep : ∀ b, b < nbOf d → ∀ jj' a, (Bst b).get2 jj' a = YY jj' t (64 * b + a))
    (hYf : ∀ c, c < d.K → (YY jj t c).isFinite = true)
    (hpad : ∀ c, d.K ≤ c → c < 64 * nbOf d → (YY jj t c).isFinite = true ∧
      |(YY jj t c).toReal| ≤ padB init d n hv t)
    (hpb : padB init d n hv t ≤ 2 ^ 100)
    (hstore : ∀ b, b < nbOf d → ∀ jj' a, FloatV.NaNEq (FloatV.cvt ((Bst b).get2 jj' a)) ((Xh b).get2 jj' a))
    (hsread : ∀ b, b < nbOf d → ∀ (a : Fin 64) (jj' : Fin BV), 64 * b + a < d.K → BV * jv + jj' < d.V →
      rdB m5 d.h (hIdx d (cbase d n + t) hv (64 * b + a) (BV * jv + jj')) = some ((Xh b).get2 jj' a))
    (Bw : ℕ → Tensor [BT, 64] BF16) (Rr : ℕ → Tensor [BT, BV] F32)
    (hW : ∀ b, b < nbOf d → WTile d init n hv t (64 * b) (Bw b))
    (hX : ∀ b, b < nbOf d → ∀ jj' a, FloatV.NaNEq (FloatV.cvt ((Bst b).get2 jj' a)) ((Xs b).get2 a jj'))
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
    {Sb : ℕ → Tensor [BV, 64] F32}
    (hvbc : ∀ i jj', FloatV.NaNEq (FloatV.cvt (bv6.get2 i jj')) (vb.get2 i jj'))
    (hgate : (d.useG = true ∧ GFacts d E.P init n hv t bv5 bv6 gt ∧
        ∀ b, b < nbOf d → ∀ i, FloatV.NaNEq (FloatV.mul ((Bst b).get i) gt.item) ((Sb b).get i)) ∨
      (d.useG = false ∧ bv6 = bv5 ∧ ∀ b, b < nbOf d → Sb b = Bst b))
    (Bk : ℕ → Tensor [64, BT] BF16) (Yb : ℕ → Tensor [BV, 64] F32) (Dd : ℕ → Tensor [64, BV] F32)
    (hK : ∀ b, b < nbOf d → KTile d init n hv t (64 * b) (Bk b) ∧
      E.P.dot cfgBF16 (Bk b) vb (tl.zeros [64, BV] .f32) (Dd b) ∧
      ∀ (jj' : Fin BV) (a : Fin 64), AddRel E.fusion ((Sb b).get2 jj' a) ((Dd b).get2 a jj') ((Yb b).get2 jj' a)) :
    (∀ i, i < cl d n t →
      |(at2 xvt i jj).toReal - vrefS init m5 d n hv (BV * jv + jj) t i| ≤
          tolV init m5 d n hv (BV * jv + jj) t i ∧ (at2 xvt i jj).isFinite = true) ∧
    ((∀ c, c < d.K → (YY jj t c).isFinite = true ∧ (flatV Yb c jj).isFinite = true) ∧
      colNorm d.K (fun c => (flatV Yb c jj).toReal - Step init m5 d n hv (BV * jv + jj) t
        (fun c' => (YY jj t c').toReal) c) ≤ errStep init m5 d n hv (BV * jv + jj) t) ∧
    (∀ c, d.K ≤ c → c < 64 * nbOf d → (flatV Yb c jj).isFinite = true ∧
      |(flatV Yb c jj).toReal| ≤ padB init d n hv (t + 1)) := by
  obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
  obtain ⟨c1, c2, c3, c4⟩ := vf_consts
  have hjV : BV * jv + jj < d.V := hin.2.2
  have hb_of : ∀ (b : ℕ) (a : Fin 64) (c : ℕ), c < 64 * nbOf d → c = 64 * b + a → b < nbOf d := by
    intro b a c hc hcb; omega
  refine lane_core (jv := jv) (fusion := E.fusion) hd hP.exp2F32 hn hhv ht hin hV hSt hnb hnb'
    (Y := fun c => YY jj t c) (xh := fun c => flatV Xh c jj) (xs := fun c => flat Xs c jj)
    (hxh := ?hxh) (hxhc := ?hxhc) (hxsc := ?hxsc) (hYf := hYf) (hpad := hpad) (hpb := hpb)
    (wt := flatC Bw) (bu := fun i => at2 bu i jj) (xv := fun i => at2 xvt i jj)
    (uf := fun i => at2 uf i jj) (bv5 := fun i => at2 bv5 i jj) (rr := fun b i => at2 (Rr b) i jj)
    (hwr := ?hwr) (hwz := ?hwz) (hbu := ?hbu) (huz := ?huz) (huf := ?huf) (hr0 := ?hr0)
    (hdot := ?hdot) (hsub := ?hsub) (hxvc := ?hxvc) (_hxvr := ?hxvr)
    (bv6 := fun i => at2 bv6 i jj) (vb := fun i => at2 vb i jj) (gt := gt.item)
    (S := fun c => flatV Sb c jj) (hvbc := ?hvbc) (hgate := ?hgate)
    (kk := fun i c => flat Bk c i) (Yn := fun c => flatV Yb c jj) (hkr := ?hkr) (hkz := ?hkz)
    (ρ2 := rD + 4 * u32) (α2 := (1 + 3 * u32) * aD + 4 * e32) (hρ2 := c1) (hα2 := c2)
    (ρP := 3 * u32) (αP := (1 + 3 * u32) * aD + 4 * e32) (hρP := c3) (hαP := c4)
    (hupdP := ?hupdP) (hupd := ?hupd)
  case hxh =>
    intro c hc
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ (by omega) rfl
    rw [flatV_blk]
    exact hsread b hb a jj hc hjV
  case hxhc =>
    intro c hc
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ (by omega) rfl
    rw [flatV_blk, ← hrep b hb jj a]
    exact hstore b hb jj a
  case hxsc =>
    intro c hc
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ hc rfl
    rw [flat_blk, ← hrep b hb jj a]
    exact hX b hb jj a
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
    have e1 : flatV Sb (64 * b + a) jj = (Sb b).get2 jj a := flatV_blk Sb b a jj
    have e2 : flatV Yb (64 * b + a) jj = (Yb b).get2 jj a := flatV_blk Yb b a jj
    simp only [e1] at hSf hroom
    simp only [e1, e2]
    have hk64 : ∀ k, k < 64 → k < BT := fun k hk => by unfold BT; omega
    have hs0 : ∑ k ∈ Finset.range 64, lmB (flat Bk (64 * b + a) k) * lmB (at2 vb k jj) = 0 :=
      Finset.sum_eq_zero fun k hk => by
        have hk' := hk64 k (Finset.mem_range.1 hk)
        rw [show flat Bk (64 * b + a) k = (Bk b).get2 a ⟨k, hk'⟩ from flat_blk Bk b a ⟨k, hk'⟩,
          lmB_zero (hz ⟨k, Finset.mem_range.1 hk⟩).2, zero_mul]
    have hs1 : ∑ k ∈ Finset.range 64, (flat Bk (64 * b + a) k).toReal * (at2 vb k jj).toReal = 0 :=
      Finset.sum_eq_zero fun k hk => by
        have hk' := hk64 k (Finset.mem_range.1 hk)
        rw [show flat Bk (64 * b + a) k = (Bk b).get2 a ⟨k, hk'⟩ from flat_blk Bk b a ⟨k, hk'⟩,
          (hz ⟨k, Finset.mem_range.1 hk⟩).2, zero_mul]
    obtain ⟨hz0f, hz0r⟩ := zeros_lane (m := 64) (n := BV) a jj
    have hdl := dot_lane_nat hP.dotBF16 (hK b hb).2.1 a jj (fun k => flat Bk (64 * b + a) k)
      (fun k => at2 vb k jj) (fun k => flat_blk Bk b a ⟨k, hk64 k k.isLt⟩)
      (fun k => at2_eq vb ⟨k, hk64 k k.isLt⟩ jj)
      (fun k hk => ⟨by
        rw [show flat Bk (64 * b + a) k = (Bk b).get2 a ⟨k, hk64 k hk⟩ from flat_blk Bk b a ⟨k, hk64 k hk⟩]
        exact (hz ⟨k, hk⟩).1, hvbf k hk⟩) hz0f
      (by rw [hs0, hz0r]; norm_num)
    rw [hs0, hs1, hz0r] at hdl
    simp only [abs_zero, add_zero, mul_zero, zero_add, sub_zero] at hdl
    exact upd_vf_pad hSf hdl.1 (hroom.trans (by norm_num)) (by simpa using hdl.2) ((hK b hb).2.2 jj a)
  case hupd =>
    intro c hc hSf hkv hroom
    obtain ⟨b, a, rfl⟩ := split64 c
    have hb := hb_of b a _ hc rfl
    have e1 : flatV Sb (64 * b + a) jj = (Sb b).get2 jj a := flatV_blk Sb b a jj
    have e2 : flatV Yb (64 * b + a) jj = (Yb b).get2 jj a := flatV_blk Yb b a jj
    simp only [e1] at hSf hroom
    simp only [e1, e2]
    have hk64 : ∀ k : Fin 64, (k : ℕ) < BT := fun k => by have := k.isLt; unfold BT; omega
    obtain ⟨hz0f, hz0r⟩ := zeros_lane (m := 64) (n := BV) a jj
    have hL0 : 0 ≤ ∑ k ∈ Finset.range 64, lmB (flat Bk (64 * b + a) k) * lmB (at2 vb k jj) :=
      Finset.sum_nonneg fun k _ => mul_nonneg (lmB_nonneg _) (lmB_nonneg _)
    have hdl := dot_lane_nat hP.dotBF16 (hK b hb).2.1 a jj (fun k => flat Bk (64 * b + a) k)
      (fun k => at2 vb k jj) (fun k => flat_blk Bk b a ⟨k, hk64 k⟩)
      (fun k => at2_eq vb ⟨k, hk64 k⟩ jj) hkv hz0f
      (by rw [hz0r, abs_zero, zero_add]; linarith [abs_nonneg (((Sb b).get2 jj a).toReal)])
    rw [hz0r] at hdl
    simp only [abs_zero, zero_add] at hdl
    exact upd_vf hSf hdl.1 hroom hL0 (abs_sum_le_lm 64 (fun k => flat Bk (64 * b + a) k)
      (fun k => at2 vb k jj)) hdl.2 ((hK b hb).2.2 jj a)
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
        rw [flatV_blk, ← hrep b hb jj a]
        exact hSn b hb (jj, a, ())
    · refine Or.inr ⟨hG, fun i hi => ?_, fun c hc => ?_⟩
      · rw [hbv6]
      · obtain ⟨b, a, rfl⟩ := split64 c
        have hb := hb_of b a _ hc rfl
        rw [flatV_blk, hSb b hb, hrep b hb jj a]

end Claims.ChunkDeltaH
