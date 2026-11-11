import Proof.GlueNonvf
import Proof.GlueLaneVf

/-!
# `chunk_delta_h_bf16`: the invariant after a trip (`[V, K]` layout)

As `GlueNonvf`: the facts the walk of `chunk_step_vf` collects give the loop invariant after the trip.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The state blocks of a tuple in the `[V, K]` layout, as a family. -/
theorem blkV_fam4 (y1 y2 y3 y4 : Tensor [BV, 64] F32) (b : ℕ) :
    blkV (y1, y2, y3, y4) b = fam4 y1 y2 y3 y4 b := by
  rcases b with _ | _ | _ | _ <;> rfl

/-- **The invariant after a trip.** -/
theorem glue_vf {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {init m m1 m2 m3 m4 m5 : Mem}
    {n hv jv t : Nat} (hn : n < d.N) (hhv : hv < d.HV) (_hjv : jv < NV d) (ht : t < nch d n)
    (_hmem : MemOk d init m) {YY : Nat → Nat → Nat → F32}
    {b1 b2 b3 b4 : Tensor [BV, 64] F32} (hrep : StRepV d (b1, b2, b3, b4) YY t)
    (hlane : LaneInv d init m n hv jv t YY) (hmem5 : MemOk d init m5)
    -- the stores of the state blocks and of the new values
    {x1 x2 x3 x4 : Tensor [BV, 64] BF16}
    (hx1 : ∀ jj a, FloatV.NaNEq (FloatV.cvt (b1.get2 jj a)) (x1.get2 jj a))
    (hx2 : ∀ jj a, FloatV.NaNEq (FloatV.cvt (b2.get2 jj a)) (x2.get2 jj a))
    (hx3 : ∀ jj a, FloatV.NaNEq (FloatV.cvt (b3.get2 jj a)) (x3.get2 jj a))
    (hx4 : ∀ jj a, FloatV.NaNEq (FloatV.cvt (b4.get2 jj a)) (x4.get2 jj a))
    (hr1 : ∀ (jj : Fin BV) (a : Fin 64), 0 + a < d.K → BV * jv + jj < d.V →
      m1.read .bf16 (d.h.ptr ((hIdx d (cbase d n + t) hv (0 + a) (BV * jv + jj) : Nat) : Int)).addr
        = some (x1.get2 jj a))
    (hr2 : ∀ (jj : Fin BV) (a : Fin 64), 64 + a < d.K → BV * jv + jj < d.V →
      m2.read .bf16 (d.h.ptr ((hIdx d (cbase d n + t) hv (64 + a) (BV * jv + jj) : Nat) : Int)).addr
        = some (x2.get2 jj a))
    (hr3 : ∀ (jj : Fin BV) (a : Fin 64), 128 + a < d.K → BV * jv + jj < d.V →
      m3.read .bf16 (d.h.ptr ((hIdx d (cbase d n + t) hv (128 + a) (BV * jv + jj) : Nat) : Int)).addr
        = some (x3.get2 jj a))
    (hr4 : ∀ (jj : Fin BV) (a : Fin 64), 192 + a < d.K → BV * jv + jj < d.V →
      m4.read .bf16 (d.h.ptr ((hIdx d (cbase d n + t) hv (192 + a) (BV * jv + jj) : Nat) : Int)).addr
        = some (x4.get2 jj a))
    (hf1 : ∀ y, y ∉ d.h.elems (hBlk d n hv jv t 0) → m1 y = m y)
    (hf2 : ∀ y, y ∉ d.h.elems (hBlk d n hv jv t 64) → m2 y = m1 y)
    (hf3 : ∀ y, y ∉ d.h.elems (hBlk d n hv jv t 128) → m3 y = m2 y)
    (hf4 : ∀ y, y ∉ d.h.elems (hBlk d n hv jv t 192) → m4 y = m3 y)
    (hf5 : ∀ y, y ∉ d.vn.elems (vCells d n hv jv t) → m5 y = m4 y)
    -- the `w` dots
    {r1 r2 r3 r4 : Tensor [BT, BV] F32}
    (hw1 : WFacts d E.P init n hv t 0 b1.trans (tl.zeros [BT, BV] .f32) r1)
    (hw2 : (64 < d.K ∧ WFacts d E.P init n hv t 64 b2.trans r1 r2) ∨ (¬ 64 < d.K ∧ r2 = r1))
    (hw3 : (128 < d.K ∧ WFacts d E.P init n hv t 128 b3.trans r2 r3) ∨ (¬ 128 < d.K ∧ r3 = r2))
    (hw4 : (192 < d.K ∧ WFacts d E.P init n hv t 192 b4.trans r3 r4) ∨ (¬ 192 < d.K ∧ r4 = r3))
    -- the difference and the store of `v_new`
    {bv5 : Tensor [BT, BV] F32} (hv5 : VFacts d E.P E.fusion init m5 n hv jv t r4 bv5)
    -- the gate, the rounded gated values and the decayed state
    {bv6 : Tensor [BT, BV] F32} {gt : Tensor [] F32} {s1 s2 s3 s4 : Tensor [BV, 64] F32}
    (hgate : (d.useG = true ∧ GFacts d E.P init n hv t bv5 bv6 gt ∧
        (∀ i, FloatV.NaNEq (FloatV.mul (b1.get i) gt.item) (s1.get i)) ∧
        ((64 < d.K ∧ ∀ i, FloatV.NaNEq (FloatV.mul (b2.get i) gt.item) (s2.get i)) ∨ (¬ 64 < d.K ∧ s2 = b2)) ∧
        ((128 < d.K ∧ ∀ i, FloatV.NaNEq (FloatV.mul (b3.get i) gt.item) (s3.get i)) ∨ (¬ 128 < d.K ∧ s3 = b3)) ∧
        ((192 < d.K ∧ ∀ i, FloatV.NaNEq (FloatV.mul (b4.get i) gt.item) (s4.get i)) ∨ (¬ 192 < d.K ∧ s4 = b4))) ∨
      (d.useG = false ∧ bv6 = bv5 ∧ s1 = b1 ∧ s2 = b2 ∧ s3 = b3 ∧ s4 = b4))
    {vb : Tensor [BT, BV] BF16}
    (hcvb : ∀ i jj, FloatV.NaNEq (FloatV.cvt (bv6.get2 i jj)) (vb.get2 i jj))
    -- the `k` dots
    {y1 y2 y3 y4 : Tensor [BV, 64] F32} {d1 : Tensor [64, BV] F32}
    (hk1 : KFacts d E.P init n hv t 0 vb (tl.zeros [64, BV] .f32) d1)
    (hy1 : ∀ (jj : Fin BV) (a : Fin 64), AddRel E.fusion (s1.get2 jj a) (d1.get2 a jj) (y1.get2 jj a))
    (hk2 : (64 < d.K ∧ KUpdFacts d E.P E.fusion init n hv t 64 vb s2 y2) ∨ (¬ 64 < d.K ∧ y2 = s2))
    (hk3 : (128 < d.K ∧ KUpdFacts d E.P E.fusion init n hv t 128 vb s3 y3) ∨ (¬ 128 < d.K ∧ y3 = s3))
    (hk4 : (192 < d.K ∧ KUpdFacts d E.P E.fusion init n hv t 192 vb s4 y4) ∨ (¬ 192 < d.K ∧ y4 = s4)) :
    InvV d init n hv jv (t + 1) (y1, y2, y3, y4) m5 := by
  have hK0 := hd.K_pos
  obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
  obtain ⟨Bw, Xs, Rr, hR0, hRn, hW⟩ := chain_w hd hw1 hw2 hw3 hw4
  obtain ⟨Bk, Dd, hKf⟩ := chain_kv hd hk1 hy1 hk2 hk3 hk4
  obtain ⟨bu, hbu, uf, xvt, hconv, hsub, hvc, hvr⟩ := hv5
  have hfr : ∀ y, y ∉ Wt d n hv jv t → m5 y = m y := by
    intro y hy
    rw [hf5 y fun h => hy (vCells_sub_Wt h), hf4 y fun h => hy (hBlk_sub_Wt h),
      hf3 y fun h => hy (hBlk_sub_Wt h), hf2 y fun h => hy (hBlk_sub_Wt h),
      hf1 y fun h => hy (hBlk_sub_Wt h)]
  -- the stored values, read in the final memory
  have hsread : ∀ b, b < nbOf d → ∀ (a : Fin 64) (jj : Fin BV), 64 * b + a < d.K → BV * jv + jj < d.V →
      rdB m5 d.h (hIdx d (cbase d n + t) hv (64 * b + a) (BV * jv + jj)) =
        some ((fam4 x1 x2 x3 x4 b).get2 jj a) := by
    intro b hb a jj hcK hjV
    have hmemS : ∀ b0, hIdx d (cbase d n + t) hv (64 * b + a) (BV * jv + jj) ∈
        (hBlk d n hv jv t (64 * b0)) ∨ b0 ≠ b := by
      intro b0
      by_cases h : b0 = b
      · subst h; exact Or.inl ⟨a, jj, a.isLt, hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, rfl⟩
      · exact Or.inr h
    have hd5 : ∀ b0, Disjoint (d.h.elems (hBlk d n hv jv t b0)) (d.vn.elems (vCells d n hv jv t)) :=
      fun b0 => hBlk_vn_disj hd hn hhv ht
    unfold rdB
    have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by
      have := hd.K_le; unfold nbOf at hb; omega
    rcases hb3 with rfl | rfl | rfl | rfl
    · have hc : hIdx d (cbase d n + t) hv (64 * 0 + a) (BV * jv + jj) ∈ hBlk d n hv jv t 0 :=
        ⟨a, jj, a.isLt, hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, by simp⟩
      rw [read_keep hc (hd5 0) hf5, read_keep hc (hBlk_disj hhv (by omega : 0 + 64 ≤ 192)) hf4,
        read_keep hc (hBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hf3,
        read_keep hc (hBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hf2]
      exact hr1 jj a (by simpa using hcK) hjV
    · have hc : hIdx d (cbase d n + t) hv (64 * 1 + a) (BV * jv + jj) ∈ hBlk d n hv jv t 64 :=
        ⟨a, jj, a.isLt, hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, by simp⟩
      rw [read_keep hc (hd5 64) hf5, read_keep hc (hBlk_disj hhv (by omega : 64 + 64 ≤ 192)) hf4,
        read_keep hc (hBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hf3]
      exact hr2 jj a (by omega) hjV
    · have hc : hIdx d (cbase d n + t) hv (64 * 2 + a) (BV * jv + jj) ∈ hBlk d n hv jv t 128 :=
        ⟨a, jj, a.isLt, hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, by simp⟩
      rw [read_keep hc (hd5 128) hf5, read_keep hc (hBlk_disj hhv (by omega : 128 + 64 ≤ 192)) hf4]
      exact hr3 jj a (by omega) hjV
    · have hc : hIdx d (cbase d n + t) hv (64 * 3 + a) (BV * jv + jj) ∈ hBlk d n hv jv t 192 :=
        ⟨a, jj, a.isLt, hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, by simp⟩
      rw [read_keep hc (hd5 192) hf5]
      exact hr4 jj a (by omega) hjV
  have hrep' : ∀ b, b < nbOf d → ∀ jj a, (fam4 b1 b2 b3 b4 b).get2 jj a = YY jj t (64 * b + a) := by
    intro b hb jj a; rw [← blkV_fam4]; exact hrep b hb a jj
  have hb4 : ∀ b, b < nbOf d → b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by
    intro b hb; have := hd.K_le; unfold nbOf at hb; omega
  have hstore : ∀ b, b < nbOf d → ∀ jj a, FloatV.NaNEq (FloatV.cvt ((fam4 b1 b2 b3 b4 b).get2 jj a))
      ((fam4 x1 x2 x3 x4 b).get2 jj a) := by
    intro b hb jj a
    rcases hb4 b hb with rfl | rfl | rfl | rfl
    exacts [hx1 jj a, hx2 jj a, hx3 jj a, hx4 jj a]
  have hXv : ∀ b, b < nbOf d → ∀ jj a, FloatV.NaNEq (FloatV.cvt ((fam4 b1 b2 b3 b4 b).get2 jj a))
      ((Xs b).get2 a jj) := by
    intro b hb jj a
    rcases hb4 b hb with rfl | rfl | rfl | rfl
    exacts [(hW 0 hb).2.1 a jj, (hW 1 hb).2.1 a jj, (hW 2 hb).2.1 a jj, (hW 3 hb).2.1 a jj]
  have hgateF : (d.useG = true ∧ GFacts d E.P init n hv t bv5 bv6 gt ∧
        ∀ b, b < nbOf d → ∀ i, FloatV.NaNEq (FloatV.mul ((fam4 b1 b2 b3 b4 b).get i) gt.item)
          ((fam4 s1 s2 s3 s4 b).get i)) ∨
      (d.useG = false ∧ bv6 = bv5 ∧ ∀ b, b < nbOf d → fam4 s1 s2 s3 s4 b = fam4 b1 b2 b3 b4 b) := by
    rcases hgate with ⟨hG, hgf, hs1, hs2, hs3, hs4⟩ | ⟨hG, hbv, h1, h2, h3, h4⟩
    · exact Or.inl ⟨hG, hgf, chain_s hd hs1 hs2 hs3 hs4⟩
    · exact Or.inr ⟨hG, hbv, fun b hb => by rw [h1, h2, h3, h4]⟩
  have hpbT : Vals init d → padB init d n hv t ≤ 2 ^ 100 := fun hV => hV.2.2.2.2.2.2 n hv t hn hhv ht.le
  refine ⟨hmem5, fun jj t' c => if t' = t + 1 then flatV (fam4 y1 y2 y3 y4) c jj else YY jj t' c, ?_, ?_⟩
  · intro b hb a jj
    simp only [↓reduceIte]
    rw [blkV_fam4]
    exact (flatV_blk _ b a jj).symm
  · intro jj hjj hin
    obtain ⟨hold, hfin⟩ := hlane jj hjj hin
    have hag : ∀ x ∈ colCells d n hv (BV * jv + jj) t, m5 x = m x := fun x hx =>
      hfr x (fun hw => Set.disjoint_left.mp (colCells_disjoint_Wt hd (jv := jv) hn hhv hin ht) hx hw)
    have hSH : ∀ t', t' < t → SameH d m m5 n hv (BV * jv + jj) t' := by
      intro t' ht' c hc
      exact rdB_congr fun x hx => hag x (Or.inl ⟨hIdx d (cbase d n + t') hv c (BV * jv + jj),
        ⟨t', c, ht', hc, rfl⟩, hx⟩)
    have hcore : Vals init d → (∀ t', t' ≤ t → StOk m5 d n hv (BV * jv + jj) t') →
        (∀ i, i < cl d n t → |(at2 xvt i jj).toReal - vrefS init m5 d n hv (BV * jv + jj) t i| ≤
            tolV init m5 d n hv (BV * jv + jj) t i ∧ (at2 xvt i jj).isFinite = true) ∧
        ((∀ c, c < d.K → (YY jj t c).isFinite = true ∧ (flatV (fam4 y1 y2 y3 y4) c jj).isFinite = true) ∧
          colNorm d.K (fun c => (flatV (fam4 y1 y2 y3 y4) c jj).toReal -
            Step init m5 d n hv (BV * jv + jj) t (fun c' => (YY jj t c').toReal) c) ≤
              errStep init m5 d n hv (BV * jv + jj) t) ∧
        (∀ c, d.K ≤ c → c < 64 * nbOf d → (flatV (fam4 y1 y2 y3 y4) c jj).isFinite = true ∧
          |(flatV (fam4 y1 y2 y3 y4) c jj).toReal| ≤ padB init d n hv (t + 1)) := by
      intro hV hSt
      have hfin' := hfin hV fun t' ht' => (StOk_congr (hSH t' ht')).mp (hSt t' (by omega))
      exact lane_vf_fam hd hP hn hhv ht ⟨jj, hjj⟩ hin hV (hSt t le_rfl) (Bst := fam4 b1 b2 b3 b4)
        (Xh := fam4 x1 x2 x3 x4) (Xs := Xs) hrep' hfin'.1 hfin'.2 (hpbT hV) hstore hsread Bw Rr
        (fun b hb => (hW b hb).1) hXv (fun b hb => (hW b hb).2.2) hR0
        hbu hconv (by rw [hRn]; exact hsub) hvc hvr (fun i jj' => hcvb i jj') hgateF Bk _ Dd
        (fun b hb => hKf b hb)
    have hjV : BV * jv + jj < d.V := hin.2.2
    refine ⟨?_, ?_⟩
    · refine colPre_extend hd hn hhv hin ht hold hfr
        (Yn := fun t' c => if t' = t + 1 then flatV (fam4 y1 y2 y3 y4) c jj else YY jj t' c) ?_ ?_ ?_ ?_ ?_
      · intro t' ht'; funext c; simp [show t' ≠ t + 1 by omega]
      · intro i hi
        have hi64 : i < BT := by simp only [cl, BT] at hi ⊢; omega
        refine ⟨at2 xvt i jj, ?_, fun hV hSt => ?_⟩
        · rw [show at2 xvt i jj = xvt.get2 ⟨i, hi64⟩ ⟨jj, hjj⟩ from at2_eq xvt ⟨i, hi64⟩ ⟨jj, hjj⟩]
          exact hvr ⟨i, hi64⟩ ⟨jj, hjj⟩ (cl_lt hi) hjV
        · obtain ⟨c1, -, -⟩ := hcore hV hSt
          exact ⟨(c1 i hi).2, (c1 i hi).1⟩
      · intro c hc
        obtain ⟨b, a, rfl⟩ := split64 c
        have hb : b < nbOf d := by omega
        exact ⟨_, hsread b hb a ⟨jj, hjj⟩ hc hjV⟩
      · intro c hc hYfin
        obtain ⟨b, a, rfl⟩ := split64 c
        have hb : b < nbOf d := by omega
        rw [hsread b hb a ⟨jj, hjj⟩ hc hjV]
        have hyfin : (YY jj t (64 * b + a)).isFinite = true := hYfin
        rw [← hrep' b hb ⟨jj, hjj⟩ a] at hyfin ⊢
        exact congrArg some (((hstore b hb ⟨jj, hjj⟩ a).eq (cvt_f32_bf16_notNaN hyfin)))
      · intro hV hSt
        obtain ⟨-, ⟨hfa, hnorm⟩, -⟩ := hcore hV hSt
        refine ⟨fun c hc => ?_, ?_⟩
        · simpa using hfa c hc
        · simpa using hnorm
    · intro hV hSt
      obtain ⟨-, ⟨hfa, -⟩, hpd⟩ := hcore hV (fun t' ht' => hSt t' (by omega))
      refine ⟨fun c hc => ?_, fun c hc1 hc2 => ?_⟩
      · simpa using (hfa c hc).2
      · simpa using hpd c hc1 hc2

end Claims.ChunkDeltaH
