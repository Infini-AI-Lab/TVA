import Proof.Init
import Proof.Foot

/-!
# `chunk_delta_h_bf16`: the final state

After the loop (`nch` trips), `ColPre` holds, and with `STORE_FINAL_STATE` the state blocks are
stored to `ht`; `Col` follows (`col_of_pre`).
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The post-condition of a program: `Col` of each of its columns. -/
def QCol (d : Inst) (init : Mem) (n hv jv : Nat) (m : Mem) : Prop :=
  ∀ jj, jj < BV → inBlk d jv (BV * jv + jj) → Col d init m n hv (BV * jv + jj)

/-- Stores to `ht` do not change what `ColPre` reads: the cells of `h` and `v_new`. -/
theorem colCells_not_ht {d : Inst} (hd : d.ok) (hst : d.storeHt = true) {n hv jv j t : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) (hj : inBlk d jv j) (ht : t ≤ nch d n) :
    Disjoint (colCells d n hv j t) (d.ht.elems (tCells d n hv jv)) := by
  have hA : ∀ y ∈ {x | ∃ t' c, t' < t ∧ c < d.K ∧ x = hIdx d (cbase d n + t') hv c j},
      y < d.B * NTt d * d.HV * d.K * d.V := by
    rintro y ⟨t', c, ht', hc, rfl⟩
    exact hIdx_lt (chunk_lt hd hn (lt_of_lt_of_le ht' ht)) hhv hc hj.2.2
  have hB : ∀ y ∈ {x | ∃ s, s < min (lenOf d n) (BT * t) ∧
      x = uIdx d (rowOf d n) (startOf d n + s) hv j}, y < d.B * d.T * d.HV * d.V := by
    rintro y ⟨s, hs, rfl⟩
    exact uIdx_lt (row_lt hd hn) (tok_lt hd hn (lt_of_lt_of_le hs (min_le_left _ _))) hhv hj.2.2
  unfold colCells
  exact Set.disjoint_union_left.mpr ⟨hht_disjoint hd hst hA (tCells_lt hn hhv),
    vnht_disjoint hd hst hB (tCells_lt hn hhv)⟩

/-- The tiles of two different blocks of `ht` are disjoint. -/
theorem tBlk_disj {d : Inst} {n hv jv b0 b0' : Nat} (hhv : hv < d.HV) (hb : b0 + 64 ≤ b0') :
    Disjoint (d.ht.elems (tBlk d n hv jv b0)) (d.ht.elems (tBlk d n hv jv b0')) := by
  refine d.ht.elems_disjoint ?_
  rw [Set.disjoint_left]
  rintro x ⟨a, jj, ha, haK, hjj, hin, rfl⟩ ⟨a', jj', ha', haK', hjj', hin', h⟩
  have := (sIdx_inj hhv hhv haK haK' hin.2.2 hin'.2.2 h).2.2.1
  omega

/-- A block's tile of `ht` is among the cells the program writes at the end. -/
theorem tBlk_sub_Wend {d : Inst} (hst : d.storeHt = true) {n hv jv base0 : Nat} :
    d.ht.elems (tBlk d n hv jv base0) ⊆ Wend d n hv jv := by
  intro x hx
  unfold Wend
  rw [ite_eq_left hst]
  exact d.ht.elems_mono tBlk_sub_tCells hx

/-- With the final state stored, a program's memory is as before outside `ht`'s tiles, so `ColPre`
is unchanged. -/
theorem colPre_of_ht {d : Inst} (hd : d.ok) (hst : d.storeHt = true) {init m m' : Mem}
    {n hv jv j : Nat} {Y : Nat → Nat → F32} (hn : n < d.N) (hhv : hv < d.HV) (hj : inBlk d jv j)
    (hfr : ∀ y, y ∉ d.ht.elems (tCells d n hv jv) → m' y = m y)
    (h : ColPre d init m n hv j (nch d n) Y) : ColPre d init m' n hv j (nch d n) Y :=
  ColPre_of_agree (fun x hx => hfr x fun hxt =>
    Set.disjoint_left.mp (colCells_not_ht hd hst hn hhv hj le_rfl) hx hxt) h

/-- A buffer stays valid when every present byte stays present. -/
theorem valid_step {dt : DType} {b : Buf dt} {m m' : Mem} (hv : b.Valid m)
    (hp : ∀ y, m y ≠ none → m' y ≠ none) : b.Valid m' :=
  Buf.valid_of_agree hv fun x hx => hp x (hv.2.2.2 x hx)

/-- The `ht` tensor has room for sequence `n`, head `hv`. -/
theorem hlen_ht {d : Inst} (hd : d.ok) (hst : d.storeHt = true) {n hv : Nat} (hn : n < d.N)
    (hhv : hv < d.HV) : (n * d.HV + hv + 1) * d.K * d.V ≤ d.ht.len := by
  have h1 := hd.ht_len hst
  have h2 : n * d.HV + hv + 1 ≤ d.N * d.HV := by
    have : (n + 1) * d.HV ≤ d.N * d.HV := Nat.mul_le_mul_right _ hn
    have : (n + 1) * d.HV = n * d.HV + d.HV := by ring
    omega
  exact le_trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ h2)) h1

/-- **`Col` from `ColPre` and the final state's stores to `ht`** (`[K, V]` layout). `mf` agrees with
`m1` outside `ht`'s cells and holds the blocks' lanes. -/
theorem qcol_ht {d : Inst} (hd : d.ok) (hst : d.storeHt = true) {init m1 mf : Mem} {n hv jv : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) {YY : Nat → Nat → Nat → F32} (b1 b2 b3 b4 : Tensor [64, BV] F32)
    (hrep : StRep d (b1, b2, b3, b4) YY (nch d n))
    (hlane : LaneInv d init m1 n hv jv (nch d n) YY)
    (hfr : ∀ y, y ∉ d.ht.elems (tCells d n hv jv) → mf y = m1 y)
    (hread : ∀ b, b < nbOf d → ∀ (a : Fin 64) (jj : Fin BV), 64 * b + a < d.K →
      BV * jv + jj < d.V →
      mf.read .f32 (d.ht.ptr ((sIdx d n hv (64 * b + a) (BV * jv + jj) : Nat) : Int)).addr =
        some ((fam4 b1 b2 b3 b4 b).get2 a jj)) :
    QCol d init n hv jv mf := by
  intro jj hjj hin
  obtain ⟨hcp, -⟩ := hlane jj hjj hin
  refine col_of_pre (colPre_of_ht hd hst hn hhv hin hfr hcp) (fun _ c hc => ?_)
  obtain ⟨b, a, rfl⟩ := split64 c
  obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
  have hb : b < nbOf d := by omega
  have := hread b hb a ⟨jj, hjj⟩ hc hin.2.2
  unfold rdF
  rw [this]
  congr 1
  rw [← hrep b hb a ⟨jj, hjj⟩, blk_fam4]

/-- **The end of a program** (`[K, V]` layout): after `nch` trips, with the final state stored to
`ht` when asked, every column satisfies `Col`. -/
theorem safe_final_nonvf {d : Inst} (hd : d.ok) {E : Env} {R : Set Int} {init m1 : Mem}
    {n hv jv : Nat} (hvf : d.vf = false) (hn : n < d.N) (hhv : hv < d.HV)
    {hp : Ptr .f32} (hhp : d.storeHt = true → hp = d.ht.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int))
    {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool} {o_k1 o_k2 o_k3 o_k4 : Tensor [64] I32}
    {m_k1 m_k2 m_k3 m_k4 : Tensor [64] Bool}
    (hL : Lanes d jv o_v m_v o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4)
    (hvt : d.storeHt = true → d.ht.Valid init) {st : St BV}
    (hinv : Inv d init n hv jv (nch d n) st m1) :
    Safe E R (Wend d n hv jv) (fun (_ : Unit) m => QCol d init n hv jv m) m1
      (match st with
        | (b_h1, b_h2, b_h3, b_h4) =>
          if d.storeHt then do
            tl.store (hp + o_k1.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64) b_h1
              (some ((m_k1.col BV).and (m_v.row 64)))
            if 64 < d.K then
              tl.store (hp + o_k2.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64) b_h2
                (some ((m_k2.col BV).and (m_v.row 64)))
            else pure ()
            if 128 < d.K then
              tl.store (hp + o_k3.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64) b_h3
                (some ((m_k3.col BV).and (m_v.row 64)))
            else pure ()
            if 192 < d.K then
              tl.store (hp + o_k4.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64) b_h4
                (some ((m_k4.col BV).and (m_v.row 64)))
            else pure ()
          else pure ()) := by
  obtain ⟨hmem1, YY, hrep, hlane⟩ := hinv
  rcases st with ⟨b1, b2, b3, b4⟩
  by_cases hst : d.storeHt = true
  · simp only [hst, ite_true]
    have hK0 := hd.K_pos
    have hlen := hlen_ht hd hst hn hhv
    obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
    have hv1 : d.ht.Valid m1 := valid_of_memOk (hvt hst) hmem1
    refine safe_tstore hd hvf hn hhv hK0 (by omega) (hhp hst) hL.ok1 hL.mk1 hL.ov hL.mv hv1
      (tBlk_sub_Wend hst) hlen ?_
    intro m2 hr1 hf1 hp1
    have hv2 := valid_step hv1 hp1
    refine safe_tstore_opt hd hvf hn hhv (by omega) (hhp hst) hL.ok2 hL.mk2 hL.ov hL.mv hv2 ?_ hlen ?_
    · intro y hy
      exact ⟨tBlk_sub_Wend hst hy, fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hy' hy⟩
    intro m3 hr2 hf2 hp2
    have hv3 := valid_step hv2 hp2
    refine safe_tstore_opt hd hvf hn hhv (by omega) (hhp hst) hL.ok3 hL.mk3 hL.ov hL.mv hv3 ?_ hlen ?_
    · intro y hy
      exact ⟨⟨tBlk_sub_Wend hst hy, fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hy' hy⟩,
        fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hy' hy⟩
    intro m4 hr3 hf3 hp3
    have hv4 := valid_step hv3 hp3
    by_cases h192 : 192 < d.K
    · simp only [h192, ite_true]
      refine safe_last (safe_tstore hd hvf hn hhv h192 (le_refl 192) (hhp hst) hL.ok4 hL.mk4 hL.ov hL.mv hv4 ?_ hlen
        (k := fun _ => pure ()) ?_)
      · intro y hy
        exact ⟨⟨⟨tBlk_sub_Wend hst hy, fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 0 + 64 ≤ 192)) hy' hy⟩,
          fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 64 + 64 ≤ 192)) hy' hy⟩,
          fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 128 + 64 ≤ 192)) hy' hy⟩
      intro m5 hr4 hf4 hp4
      refine safe_ret ?_
      have hsub : ∀ b0, d.ht.elems (tBlk d n hv jv b0) ⊆ d.ht.elems (tCells d n hv jv) :=
        fun b0 => d.ht.elems_mono tBlk_sub_tCells
      refine qcol_ht hd hst hn hhv b1 b2 b3 b4 hrep hlane ?_ ?_
      · intro y hy
        rw [hf4 y fun h => hy (hsub _ h), hf3 y fun h => hy (hsub _ h), hf2 y fun h => hy (hsub _ h),
          hf1 y fun h => hy (hsub _ h)]
      · intro b hb a jj hcK hjV
        have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by have := hd.K_le; unfold nbOf at hb; omega
        have hmem : ∀ b0, b0 = 64 * b → sIdx d n hv (64 * b + a) (BV * jv + jj) ∈ tBlk d n hv jv b0 :=
          fun b0 h => ⟨a, jj, a.isLt, by rw [h]; exact hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, by rw [h]⟩
        rcases hb3 with rfl | rfl | rfl | rfl
        · rw [read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 192)) hf4,
            read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hf3,
            read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hf2]
          simpa [fam4] using hr1 a jj (by simpa using hcK) hjV
        · rw [read_keep (hmem 64 rfl) (tBlk_disj hhv (by omega : 64 + 64 ≤ 192)) hf4,
            read_keep (hmem 64 rfl) (tBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hf3]
          simpa [fam4] using hr2 a jj (by omega) hjV
        · rw [read_keep (hmem 128 rfl) (tBlk_disj hhv (by omega : 128 + 64 ≤ 192)) hf4]
          simpa [fam4] using hr3 a jj (by omega) hjV
        · simpa [fam4] using hr4 a jj (by omega) hjV
    · simp only [h192, ite_false]
      refine safe_ret ?_
      have hsub : ∀ b0, d.ht.elems (tBlk d n hv jv b0) ⊆ d.ht.elems (tCells d n hv jv) :=
        fun b0 => d.ht.elems_mono tBlk_sub_tCells
      refine qcol_ht hd hst hn hhv b1 b2 b3 b4 hrep hlane ?_ ?_
      · intro y hy
        rw [hf3 y fun h => hy (hsub _ h), hf2 y fun h => hy (hsub _ h), hf1 y fun h => hy (hsub _ h)]
      · intro b hb a jj hcK hjV
        have hb3 : b = 0 ∨ b = 1 ∨ b = 2 := by have := hd.K_le; unfold nbOf at hb; omega
        have hmem : ∀ b0, b0 = 64 * b → sIdx d n hv (64 * b + a) (BV * jv + jj) ∈ tBlk d n hv jv b0 :=
          fun b0 h => ⟨a, jj, a.isLt, by rw [h]; exact hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, by rw [h]⟩
        rcases hb3 with rfl | rfl | rfl
        · rw [read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hf3,
            read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hf2]
          simpa [fam4] using hr1 a jj (by simpa using hcK) hjV
        · rw [read_keep (hmem 64 rfl) (tBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hf3]
          simpa [fam4] using hr2 a jj (by omega) hjV
        · simpa [fam4] using hr3 a jj (by omega) hjV
  · have hst' : d.storeHt = false := by simpa using hst
    simp only [hst', Bool.false_eq_true, ite_false]
    refine safe_ret ?_
    intro jj hjj hin
    exact col_of_pre (hlane jj hjj hin).1 (fun h => absurd h hst)


/-- **`Col` from `ColPre` and the final state's stores to `ht`** (`[V, K]` layout). `mf` agrees with
`m1` outside `ht`'s cells and holds the blocks' lanes. -/
theorem qcol_ht_vf {d : Inst} (hd : d.ok) (hst : d.storeHt = true) {init m1 mf : Mem} {n hv jv : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) {YY : Nat → Nat → Nat → F32} (b1 b2 b3 b4 : Tensor [BV, 64] F32)
    (hrep : StRepV d (b1, b2, b3, b4) YY (nch d n))
    (hlane : LaneInv d init m1 n hv jv (nch d n) YY)
    (hfr : ∀ y, y ∉ d.ht.elems (tCells d n hv jv) → mf y = m1 y)
    (hread : ∀ b, b < nbOf d → ∀ (jj : Fin BV) (a : Fin 64), 64 * b + a < d.K →
      BV * jv + jj < d.V →
      mf.read .f32 (d.ht.ptr ((sIdx d n hv (64 * b + a) (BV * jv + jj) : Nat) : Int)).addr =
        some ((fam4 b1 b2 b3 b4 b).get2 jj a)) :
    QCol d init n hv jv mf := by
  intro jj hjj hin
  obtain ⟨hcp, -⟩ := hlane jj hjj hin
  refine col_of_pre (colPre_of_ht hd hst hn hhv hin hfr hcp) (fun _ c hc => ?_)
  obtain ⟨b, a, rfl⟩ := split64 c
  obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
  have hb : b < nbOf d := by omega
  have := hread b hb ⟨jj, hjj⟩ a hc hin.2.2
  unfold rdF
  rw [this]
  congr 1
  rw [← hrep b hb a ⟨jj, hjj⟩, blkV_fam4]

/-- **The end of a program** (`[V, K]` layout): after `nch` trips, with the final state stored to
`ht` when asked, every column satisfies `Col`. -/
theorem safe_final_vf {d : Inst} (hd : d.ok) {E : Env} {R : Set Int} {init m1 : Mem}
    {n hv jv : Nat} (hvf : d.vf = true) (hjv : jv < NV d) (hn : n < d.N) (hhv : hv < d.HV)
    {hp : Ptr .f32} (hhp : d.storeHt = true → hp = d.ht.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int))
    {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool} {o_k1 o_k2 o_k3 o_k4 : Tensor [64] I32}
    {m_k1 m_k2 m_k3 m_k4 : Tensor [64] Bool}
    (hL : Lanes d jv o_v m_v o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4)
    (hvt : d.storeHt = true → d.ht.Valid init) {st : StV BV}
    (hinv : InvV d init n hv jv (nch d n) st m1) :
    Safe E R (Wend d n hv jv) (fun (_ : Unit) m => QCol d init n hv jv m) m1
      (match st with
        | (b_h1, b_h2, b_h3, b_h4) =>
          if d.storeHt then do
            tl.store (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k1.row BV) b_h1
              (some ((m_v.col 64).and (m_k1.row BV)))
            if 64 < d.K then
              tl.store (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k2.row BV) b_h2
                (some ((m_v.col 64).and (m_k2.row BV)))
            else pure ()
            if 128 < d.K then
              tl.store (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k3.row BV) b_h3
                (some ((m_v.col 64).and (m_k3.row BV)))
            else pure ()
            if 192 < d.K then
              tl.store (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k4.row BV) b_h4
                (some ((m_v.col 64).and (m_k4.row BV)))
            else pure ()
          else pure ()) := by
  obtain ⟨hmem1, YY, hrep, hlane⟩ := hinv
  rcases st with ⟨b1, b2, b3, b4⟩
  by_cases hst : d.storeHt = true
  · simp only [hst, ite_true]
    have hK0 := hd.K_pos
    have hlen := hlen_ht hd hst hn hhv
    obtain ⟨hnb, hnb'⟩ := nbOf_bounds (d := d)
    have hv1 : d.ht.Valid m1 := valid_of_memOk (hvt hst) hmem1
    refine safe_tstore_vf hd hvf hn hhv hjv (hhp hst) hL.ok1 hL.mk1 hL.ov hL.mv hv1
      (tBlk_sub_Wend hst) hlen ?_
    intro m2 hr1 hf1 hp1
    have hv2 := valid_step hv1 hp1
    refine safe_tstore_vf_opt hd hvf hn hhv hjv (hhp hst) hL.ok2 hL.mk2 hL.ov hL.mv hv2 ?_ hlen ?_
    · intro y hy
      exact ⟨tBlk_sub_Wend hst hy, fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hy' hy⟩
    intro m3 hr2 hf2 hp2
    have hv3 := valid_step hv2 hp2
    refine safe_tstore_vf_opt hd hvf hn hhv hjv (hhp hst) hL.ok3 hL.mk3 hL.ov hL.mv hv3 ?_ hlen ?_
    · intro y hy
      exact ⟨⟨tBlk_sub_Wend hst hy, fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hy' hy⟩,
        fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hy' hy⟩
    intro m4 hr3 hf3 hp3
    have hv4 := valid_step hv3 hp3
    by_cases h192 : 192 < d.K
    · simp only [h192, ite_true]
      refine safe_last (safe_tstore_vf hd hvf hn hhv hjv (hhp hst) hL.ok4 hL.mk4 hL.ov hL.mv hv4 ?_ hlen
        (k := fun _ => pure ()) ?_)
      · intro y hy
        exact ⟨⟨⟨tBlk_sub_Wend hst hy, fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 0 + 64 ≤ 192)) hy' hy⟩,
          fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 64 + 64 ≤ 192)) hy' hy⟩,
          fun hy' => Set.disjoint_left.mp (tBlk_disj hhv (by omega : 128 + 64 ≤ 192)) hy' hy⟩
      intro m5 hr4 hf4 hp4
      refine safe_ret ?_
      have hsub : ∀ b0, d.ht.elems (tBlk d n hv jv b0) ⊆ d.ht.elems (tCells d n hv jv) :=
        fun b0 => d.ht.elems_mono tBlk_sub_tCells
      refine qcol_ht_vf hd hst hn hhv b1 b2 b3 b4 hrep hlane ?_ ?_
      · intro y hy
        rw [hf4 y fun h => hy (hsub _ h), hf3 y fun h => hy (hsub _ h), hf2 y fun h => hy (hsub _ h),
          hf1 y fun h => hy (hsub _ h)]
      · intro b hb jj a hcK hjV
        have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by have := hd.K_le; unfold nbOf at hb; omega
        have hmem : ∀ b0, b0 = 64 * b → sIdx d n hv (64 * b + a) (BV * jv + jj) ∈ tBlk d n hv jv b0 :=
          fun b0 h => ⟨a, jj, a.isLt, by rw [h]; exact hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, by rw [h]⟩
        rcases hb3 with rfl | rfl | rfl | rfl
        · rw [read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 192)) hf4,
            read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hf3,
            read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hf2]
          simpa [fam4] using hr1 jj a (by simpa using hcK) hjV
        · rw [read_keep (hmem 64 rfl) (tBlk_disj hhv (by omega : 64 + 64 ≤ 192)) hf4,
            read_keep (hmem 64 rfl) (tBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hf3]
          simpa [fam4] using hr2 jj a (by omega) hjV
        · rw [read_keep (hmem 128 rfl) (tBlk_disj hhv (by omega : 128 + 64 ≤ 192)) hf4]
          simpa [fam4] using hr3 jj a (by omega) hjV
        · simpa [fam4] using hr4 jj a (by omega) hjV
    · simp only [h192, ite_false]
      refine safe_ret ?_
      have hsub : ∀ b0, d.ht.elems (tBlk d n hv jv b0) ⊆ d.ht.elems (tCells d n hv jv) :=
        fun b0 => d.ht.elems_mono tBlk_sub_tCells
      refine qcol_ht_vf hd hst hn hhv b1 b2 b3 b4 hrep hlane ?_ ?_
      · intro y hy
        rw [hf3 y fun h => hy (hsub _ h), hf2 y fun h => hy (hsub _ h), hf1 y fun h => hy (hsub _ h)]
      · intro b hb jj a hcK hjV
        have hb3 : b = 0 ∨ b = 1 ∨ b = 2 := by have := hd.K_le; unfold nbOf at hb; omega
        have hmem : ∀ b0, b0 = 64 * b → sIdx d n hv (64 * b + a) (BV * jv + jj) ∈ tBlk d n hv jv b0 :=
          fun b0 h => ⟨a, jj, a.isLt, by rw [h]; exact hcK, jj.isLt, ⟨by omega, by omega, hjV⟩, by rw [h]⟩
        rcases hb3 with rfl | rfl | rfl
        · rw [read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 128)) hf3,
            read_keep (hmem 0 rfl) (tBlk_disj hhv (by omega : 0 + 64 ≤ 64)) hf2]
          simpa [fam4] using hr1 jj a (by simpa using hcK) hjV
        · rw [read_keep (hmem 64 rfl) (tBlk_disj hhv (by omega : 64 + 64 ≤ 128)) hf3]
          simpa [fam4] using hr2 jj a (by omega) hjV
        · simpa [fam4] using hr3 jj a (by omega) hjV
  · have hst' : d.storeHt = false := by simpa using hst
    simp only [hst', Bool.false_eq_true, ite_false]
    refine safe_ret ?_
    intro jj hjj hin
    exact col_of_pre (hlane jj hjj hin).1 (fun h => absurd h hst)


end Claims.ChunkDeltaH
