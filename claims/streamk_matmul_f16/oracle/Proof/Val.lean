import Claim
import TVA.Proof
import Proof.Step

/-!
# `Proof.Val`

The atomic invariant, and the decomposition of the final memory: `post_of_decomp` gives `Post`.
-/

/-!
# The invariant across one atomic run

The `atomic_add` branch of one iteration appends one atomic step to the program's log.  This file
shows that `Qg` is preserved: the pairing list grows by one entry whose value function is supported
on the output elements of the tile being run, and the finished-tile clauses are unaffected because
the tile is not owned (so it is not already finished) and every other gated tile was gated before.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-- The length of the filter of a list with one extra element appended. -/
private theorem filter_append_single (L : List (AtomicStep × (Nat × Nat → Option ℚ)))
    (s : AtomicStep) (vals : Nat × Nat → Option ℚ) (q : Nat × Nat) :
    ((L ++ [(⟨s, vals⟩ : AtomicStep × (Nat × Nat → Option ℚ))]).filter
        (fun p => (p.2 q).isSome)).length
      = (L.filter (fun p => (p.2 q).isSome)).length + (if (vals q).isSome then 1 else 0) := by
  rw [List.filter_append, List.length_append]
  have hsingle : (List.filter (fun p => (p.2 q).isSome)
      [(⟨s, vals⟩ : AtomicStep × (Nat × Nat → Option ℚ))]).length
      = if (vals q).isSome then 1 else 0 := by
    by_cases hv : (vals q).isSome = true
    · simp [hv]
    · simp [hv]
  rw [hsingle]

/-- **`Qg` across one atomic run.**  `log` grows by the step `s`, and `vals` is the value of each
output element the run added (supported exactly on the output elements of the tile `u`). -/
theorem Qg_atomic {init : Mem} {i it it' : Nat} {log : List AtomicStep} {m : Mem}
    {u : Nat} {s : AtomicStep} {vals : Nat × Nat → Option ℚ}
    (hQg : Qg d init i it log m)
    (hS : 0 < steps d.K) (hu : u ∈ tiles d) (hit : it < hi d i)
    (hule : u * steps d.K ≤ it) (hult : it < (u + 1) * steps d.K)
    (hnex : it' = min ((u + 1) * steps d.K) (hi d i))
    (hne : c0 d i u < c1 d i u) (hns : ¬ owns d i u) (hsp : splitTile d u)
    (haccs : ∀ q ∈ splitOuts d, AccStep .f32 (cAddr d q.1 q.2) FloatV.toRat good aRho aAlpha
      Format.binary32.maxVal s (vals q))
    (hrun : ∀ q ∈ splitOuts d, ∀ w : ℚ, vals q = some w → RunVal d init i u q w)
    (hsupp : ∀ q ∈ splitOuts d, (vals q = none ↔ q ∉ tileOut d u))
    (hgate : ∀ w ∈ tiles d, w ≠ u → c0 d i w < c1 d i w → gate d i it' w → gate d i it w) :
    Qg d init i it' (log ++ [s]) m := by
  obtain ⟨L, hL, hacc, hframe, hval, hrun_old, hcnt⟩ := hQg
  refine ⟨L ++ [⟨s, vals⟩], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [List.map_append, hL]
    rfl
  · intro p hp q hq va vb hva hvb
    rcases List.mem_append.mp hp with hp | hp
    · exact hacc p hp q hq va vb hva hvb
    · rw [List.mem_singleton] at hp
      subst hp
      exact haccs q hq
  · intro x hx
    exact hframe x hx
  · intro u0 hu0 ho hg q hq
    obtain ⟨h0, h1⟩ := ho
    have hne0 : u0 ≠ u := by
      rintro rfl
      exact hns ⟨h0, h1⟩
    have hc0 : c0 d i u0 < c1 d i u0 := by
      rw [h0, h1]
      exact hS
    exact hval u0 hu0 ⟨h0, h1⟩ (hgate u0 hu0 hne0 hc0 hg) q hq
  · intro p hp q hq w hw
    rcases List.mem_append.mp hp with hp | hp
    · exact hrun_old p hp q hq w hw
    · rw [List.mem_singleton] at hp
      subst hp
      have hw' : vals q = some w := hw
      have hmem : q ∈ tileOut d u := by
        by_contra hc
        have hnone : vals q = none := (hsupp q hq).mpr hc
        rw [hnone] at hw'
        simp at hw'
      exact ⟨u, hu, hsp, hmem, hns, hrun q hq w hw'⟩
  · intro q hq u0 hu0 hqu
    have hlen := filter_append_single L s vals q
    by_cases hwu : u0 = u
    · have hne' : c0 d i u0 < c1 d i u0 := by rw [hwu]; exact hne
      have hqu' : q ∈ tileOut d u := by rw [← hwu]; exact hqu
      have hgate_old : ¬ gate d i it u0 := by
        rw [hwu]
        show ¬ (min (u * steps d.K + steps d.K) (hi d i) ≤ it)
        intro hg
        rcases min_le_iff.mp hg with h | h
        · rw [Nat.succ_mul] at hult
          omega
        · omega
      have hold : (L.filter (fun p => (p.2 q).isSome)).length = 0 := by
        have h := hcnt q hq u0 hu0 hqu
        rw [ite_eq_right (fun hh => hgate_old hh.2)] at h
        exact h
      have hvne : vals q ≠ none := fun hv => ((hsupp q hq).mp hv) hqu'
      have hvq : (vals q).isSome = true := by
        cases hv : vals q with
        | none => exact absurd hv hvne
        | some w => rfl
      have hgate_new : gate d i it' u0 := by
        rw [hwu]
        show min (u * steps d.K + steps d.K) (hi d i) ≤ it'
        rw [hnex, Nat.succ_mul]
      rw [hlen, hold, hvq]
      simp [hne', hgate_new]
    · have hnotmem : q ∉ tileOut d u := fun hq' =>
        Set.disjoint_left.mp (tileOut_disjoint hu0 hu hwu) hqu hq'
      have hvq : vals q = none := (hsupp q hq).mpr hnotmem
      have hle : it ≤ it' := by
        rw [hnex]
        exact le_min hult.le hit.le
      have hiff : (c0 d i u0 < c1 d i u0 ∧ gate d i it u0)
          ↔ (c0 d i u0 < c1 d i u0 ∧ gate d i it' u0) := by
        constructor
        · rintro ⟨h1, h2⟩
          exact ⟨h1, gate_mono hle h2⟩
        · rintro ⟨h1, h2⟩
          exact ⟨h1, hgate u0 hu0 hwu h1 h2⟩
      have hz : (if (vals q).isSome then 1 else 0) = 0 := by
        simp [hvq]
      rw [hlen, hcnt q hq u0 hu0 hqu, hz, add_zero]
      by_cases hA : (c0 d i u0 < c1 d i u0 ∧ gate d i it u0)
      · rw [ite_eq_left hA, ite_eq_left (hiff.mp hA)]
      · rw [ite_eq_right hA, ite_eq_right (fun hh => hA (hiff.mpr hh))]

end Claims.StreamKMatmul


set_option maxHeartbeats 1000000

/-!
# From the shared-memory decomposition to `Post`

`TVA.shared` hands back, for every outcome, the final memory `fin`, an interleaving `G` of the
programs' atomic steps, the memory each program left in its own footprint, and the replay of `G`.
From those this file derives the claim's `Post`: each output element holds the dot product of its
row and column to the tolerance `tol`, and the rest of the memory is unchanged.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst} {P : Platform} {init : Mem}

/-! ## Small helpers -/

/-- A replay preserves writability, hence every writable cell can still be read. -/
theorem Replay.exists_read {m m' : Mem} {l : List AtomicStep} (h : Replay m l m')
    {dd : DType} {a : Int} (hw : m.Writable dd a) : ∃ v, m'.read dd a = some v := by
  induction h with
  | nil => exact Mem.read_some_of_writable hw
  | cons _ _ ih => exact ih (Mem.writable_write (d := _) (a := _) _ hw)

/-- The number of recorded values at `q` equals the number of steps at `q`. -/
theorem length_filterMap_eq_filter (l : List (AtomicStep × (Nat × Nat → Option ℚ))) (q : Nat × Nat) :
    (l.filterMap (fun p => p.2 q)).length = (l.filter (fun p => (p.2 q).isSome)).length := by
  rw [List.length_filterMap_eq_countP, List.countP_eq_length_filter]

/-- The first chunk of a tile run is within `steps K`. -/
theorem c0_le {d : Inst} {p u : Nat} : c0 d p u ≤ steps d.K := min_le_left _ _

/-- The exact value of one program's run on a tile is at most `2 ^ 45` in magnitude. -/
theorem abs_pre_sub_le_2_45 {d : Inst} {va vb : Nat → F16} (hd : d.ok)
    (hva : ∀ k, k < d.K → |(va k).toRat| ≤ 65504)
    (hvb : ∀ k, k < d.K → |(vb k).toRat| ≤ 65504) (i u : Nat) :
    |pre va vb d.K (c1 d i u) - pre va vb d.K (c0 d i u)| ≤ 2 ^ 45 := by
  have hdev : dev va vb d.K (max (c0 d i u) (c1 d i u)) ≤ 2 ^ 45 := by
    have h := dev_abs_le (va := va) (vb := vb) (K := d.K)
      (t := max (c0 d i u) (c1 d i u)) hd.K_le hva hvb
    rwa [abs_of_nonneg (dev_nonneg va vb d.K _)] at h
  rcases le_total (c0 d i u) (c1 d i u) with h | h
  · calc |pre va vb d.K (c1 d i u) - pre va vb d.K (c0 d i u)|
        ≤ dev va vb d.K (c1 d i u) := le_dev h
      _ = dev va vb d.K (max (c0 d i u) (c1 d i u)) := by rw [max_eq_right h]
      _ ≤ 2 ^ 45 := hdev
  · rw [abs_sub_comm]
    calc |pre va vb d.K (c0 d i u) - pre va vb d.K (c1 d i u)|
        ≤ dev va vb d.K (c0 d i u) := le_dev h
      _ = dev va vb d.K (max (c0 d i u) (c1 d i u)) := by rw [max_eq_left h]
      _ ≤ 2 ^ 45 := hdev

/-- The room hypothesis of `replay_err_shared` when the cell starts at zero and the accumulated
magnitudes are at most `2 ^ 53`. -/
theorem room_le_maxVal {A : ℚ} {n : Nat} (hA0 : 0 ≤ A) (hA : A ≤ 2 ^ 53) (hn : n ≤ 128) :
    A + DotLoop.bound aRho aAlpha n A ≤ Format.binary32.maxVal := by
  have hn128 : (n : ℚ) ≤ 128 := by exact_mod_cast hn
  have hb := bound_atomic_le (na := n) hn128 hA0
  have hb' : DotLoop.bound aRho aAlpha n A ≤
      (2 ^ (-16 : ℤ) + 2 ^ (-31 : ℤ)) * A + 2 ^ (-116 : ℤ) := by
    simpa only [aRho, aAlpha] using hb
  have hcoef : (2 ^ (-16 : ℤ) + 2 ^ (-31 : ℤ) : ℚ) ≤ 1 := by norm_num
  have hsmall : (2 ^ (-116 : ℤ) : ℚ) ≤ 1 := by norm_num
  have h1 := mul_le_mul_of_nonneg_right hcoef hA0
  have hb2 : DotLoop.bound aRho aAlpha n A ≤ A + 1 := by nlinarith
  have hmax : (2 : ℚ) ^ 55 ≤ Format.binary32.maxVal := by
    rw [Format.binary32_maxVal]
    norm_num
  nlinarith

/-! ## The theorem -/

/-- **The postcondition from the decomposition.** -/
theorem post_of_decomp (_hP : Device P) (hd : d.ok) (hpre : Pre d init) {fin : Mem}
    {G : List (Fin (launch d).size × AtomicStep)}
    (hprogs : ∀ i : Fin (launch d).size, ∃ m, Q d init (i : Nat) (proj G i) m ∧
      ∀ x ∈ inElems d ∪ Wset d (i : Nat), fin x = m x)
    (hreplay : ∃ mA, Replay init (G.map Prod.snd) mA ∧ ∀ x ∈ Cbytes d, fin x = mA x)
    (hframe : ∀ x, x ∉ Cbytes d → (∀ i : Fin (launch d).size, x ∉ Wset d (i : Nat)) →
      fin x = init x) :
    Post d init fin := by
  classical
  obtain ⟨mA, hrep, hAgreeC⟩ := hreplay
  refine ⟨?_, ?_⟩
  · intro i j hi hj
    have hq : (i, j) ∈ allOuts d := ⟨hi, hj⟩
    obtain ⟨u, hu, hqu⟩ := mem_tiles_cover (d := d) hq
    by_cases hsplit : splitTile d u
    · -- the tile is combined with atomics
      have hqs : (i, j) ∈ splitOuts d := ⟨u, hu, hsplit, hqu⟩
      obtain ⟨z, hz0, hzfin, hzrat⟩ := hpre.2.2.2 i j hi hj
      obtain ⟨c, hc⟩ := Replay.exists_read hrep (Mem.writable_of_read hz0)
      have hspanC : ∀ x ∈ Mem.span .f32 (cAddr d i j), fin x = mA x := fun x hx =>
        hAgreeC x ⟨(i, j), hqs, hx⟩
      have hfc : fin.read .f32 (cAddr d i j) = some c :=
        (read_congr hspanC).trans hc
      refine ⟨c, hfc, ?_⟩
      intro va vb hva hvb
      have hva' : ∀ k, k < d.K → |(va k).toRat| ≤ 65504 := fun k hk => f16_toRat_abs_le (hva k hk).2
      have hvb' : ∀ k, k < d.K → |(vb k).toRat| ≤ 65504 := fun k hk => f16_toRat_abs_le (hvb k hk).2
      -- the programs' memories and logs
      choose m hm using hprogs
      choose L hL using fun i' => (hm i').1
      have hLmap : ∀ i', (L i').map Prod.fst = proj G i' := fun i' => (hL i').1
      have hAccQ : ∀ i' (p : AtomicStep × (Nat × Nat → Option ℚ)), p ∈ L i' →
          ∀ q ∈ splitOuts d, ∀ va vb : Nat → F16,
            ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
            AccStep .f32 (cAddr d q.1 q.2) FloatV.toRat good aRho aAlpha
              Format.binary32.maxVal p.1 (p.2 q) :=
        fun i' p hp q hq va vb hva hvb => (hL i').2.1 p hp q hq va vb hva hvb
      have hRunQ : ∀ i' (p : AtomicStep × (Nat × Nat → Option ℚ)), p ∈ L i' →
          ∀ q ∈ splitOuts d, ∀ w : ℚ, p.2 q = some w →
            ∃ u' ∈ tiles d, splitTile d u' ∧ q ∈ tileOut d u' ∧ ¬ owns d (i' : Nat) u' ∧
              RunVal d init (i' : Nat) u' q w :=
        fun i' p hp q hq w hw => (hL i').2.2.2.2.1 p hp q hq w hw
      have hCntQ : ∀ i' q, q ∈ splitOuts d → ∀ u' ∈ tiles d, q ∈ tileOut d u' →
          ((L i').filter (fun p => (p.2 q).isSome)).length =
            if c0 d (i' : Nat) u' < c1 d (i' : Nat) u' then 1 else 0 :=
        fun i' q hq u' hu' hqu' => (hL i').2.2.2.2.2 q hq u' hu' hqu'
      have hAgreeW : ∀ (i' : Fin (launch d).size) (x : Int),
          x ∈ inElems d ∪ Wset d (i' : Nat) → fin x = m i' x :=
        fun i' x => (hm i').2 x
      set ops : Fin (launch d).size → Multiset ℚ :=
        fun i' => ((L i').filterMap (fun p => p.2 (i, j)) : List ℚ) with hops
      have hops_apply : ∀ i', ops i' = ((L i').filterMap (fun p => p.2 (i, j)) : Multiset ℚ) :=
        fun i' => by simp only [hops]
      -- the pairing of each program's log with its operands
      have hPaired : ∀ i' : Fin (launch d).size,
          Paired (AccStep .f32 (cAddr d i j) FloatV.toRat good aRho aAlpha Format.binary32.maxVal)
            (proj G i') (ops i') := by
        intro i'
        refine ⟨(L i').map (fun p => (p.1, p.2 (i, j))), ?_, ?_, ?_⟩
        · have hmap : ((L i').map (fun p => (p.1, p.2 (i, j)))).map Prod.fst
              = (L i').map Prod.fst := by
            rw [List.map_map]
            congr 1
          rw [hmap]
          exact hLmap i'
        · rw [hops_apply i', List.filterMap_map]
          rfl
        · intro p hp
          obtain ⟨p0, hp0, rfl⟩ := List.mem_map.mp hp
          exact hAccQ i' p0 hp0 (i, j) hqs va vb hva hvb
      -- the exact value of each program's run on the tile
      set Pp : Fin (launch d).size → ℚ :=
        fun i' => pre va vb d.K (c1 d (i' : Nat) u) - pre va vb d.K (c0 d (i' : Nat) u) with hPp
      set Ff : Fin (launch d).size → ℚ :=
        fun i' => fineOne va vb d.K (c0 d (i' : Nat) u) (c1 d (i' : Nat) u - c0 d (i' : Nat) u)
        with hFf
      have hFf0 : ∀ i' : Fin (launch d).size, 0 ≤ Ff i' := by
        intro i'
        simpa only [hFf] using
          fineOne_nonneg (va := va) (vb := vb) (K := d.K)
            (c0 := c0 d (i' : Nat) u) (n := c1 d (i' : Nat) u - c0 d (i' : Nat) u)
      have hempty : ∀ (i' : Fin (launch d).size),
          ¬ (c0 d (i' : Nat) u < c1 d (i' : Nat) u) → ops i' = 0 := by
        intro i' hh
        have hcard : (ops i').card = 0 := by
          rw [hops_apply i', Multiset.coe_card, length_filterMap_eq_filter,
            hCntQ i' (i, j) hqs u hu hqu]
          simp [hh]
        exact Multiset.card_eq_zero.mp hcard
      have hsingle : ∀ (i' : Fin (launch d).size) (hh : c0 d (i' : Nat) u < c1 d (i' : Nat) u),
          ∃ w : ℚ, ops i' = {w} ∧ |w - Pp i'| ≤ Ff i' := by
        intro i' hh
        have hcard : (ops i').card = 1 := by
          rw [hops_apply i', Multiset.coe_card, length_filterMap_eq_filter,
            hCntQ i' (i, j) hqs u hu hqu]
          simp [hh]
        obtain ⟨w, hw⟩ := Multiset.card_eq_one.mp hcard
        refine ⟨w, hw, ?_⟩
        have hmem : w ∈ ops i' := by rw [hw]; exact Multiset.mem_singleton_self w
        have hmem2 : w ∈ ((L i').filterMap (fun p => p.2 (i, j)) : Multiset ℚ) := by
          simpa only [hops] using hmem
        rw [Multiset.mem_coe, List.mem_filterMap] at hmem2
        obtain ⟨p, hpL, hpq⟩ := hmem2
        obtain ⟨u', hu', _hsplit', hqu', _hnown, v, hwv, hboundv⟩ :=
          hRunQ i' p hpL (i, j) hqs w hpq
        have huu : u' = u := by
          by_contra hne
          exact (Set.disjoint_left.mp (tileOut_disjoint hu hu' (fun h => hne h.symm)) hqu hqu')
        subst huu
        have hb := (hboundv va vb hva hvb).2
        simpa only [hwv, hPp, hFf] using hb
      -- the per-program magnitude and value bounds
      have hA_i : ∀ i' : Fin (launch d).size, (Multiset.map (fun w => |w|) (ops i')).sum ≤
          if c0 d (i' : Nat) u < c1 d (i' : Nat) u then |Pp i'| + Ff i' else 0 := by
        intro i'
        by_cases hh : c0 d (i' : Nat) u < c1 d (i' : Nat) u
        · obtain ⟨w, hw, hb⟩ := hsingle i' hh
          simp only [hw, Multiset.map_singleton, Multiset.sum_singleton, ite_eq_left hh]
          have h1 : |w| ≤ |w - Pp i'| + |Pp i'| := by
            have e : w = (w - Pp i') + Pp i' := by ring
            calc |w| = |(w - Pp i') + Pp i'| := congrArg abs e
              _ ≤ |w - Pp i'| + |Pp i'| := abs_add_le _ _
          linarith
        · rw [ite_eq_right hh, hempty i' hh]
          simp
      have hV_i : ∀ i' : Fin (launch d).size, |(ops i').sum -
          (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Pp i' else 0)| ≤
          (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Ff i' else 0) := by
        intro i'
        by_cases hh : c0 d (i' : Nat) u < c1 d (i' : Nat) u
        · obtain ⟨w, hw, hb⟩ := hsingle i' hh
          simp only [hw, Multiset.sum_singleton, ite_eq_left hh]
          simpa using hb
        · rw [ite_eq_right hh, ite_eq_right hh, hempty i' hh]
          simp
      -- the global sums
      set na : Nat := ∑ i' : Fin (launch d).size, (ops i').card with hna
      set B : ℚ := ∑ i' : Fin (launch d).size, (Multiset.map (fun w => |w|) (ops i')).sum with hB
      set Ws : ℚ := ∑ i' : Fin (launch d).size, (ops i').sum with hWs
      set S : ℚ := ∑ i' : Fin (launch d).size,
        (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Ff i' else 0) with hS
      have hna_le : na ≤ steps d.K := by
        rw [hna]
        calc ∑ i' : Fin (launch d).size, (ops i').card
            = ∑ i' : Fin (launch d).size,
                (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then 1 else 0) := by
              refine Finset.sum_congr rfl fun i' _ => ?_
              rw [hops_apply i', Multiset.coe_card, length_filterMap_eq_filter,
                hCntQ i' (i, j) hqs u hu hqu]
          _ ≤ steps d.K := sum_count_le_steps hd hu
      have hB0 : 0 ≤ B := by
        rw [hB]
        refine Finset.sum_nonneg fun i' _ => ?_
        exact Multiset.sum_nonneg (s := (ops i').map (fun w => |w|)) fun x hx => by
          obtain ⟨w, -, rfl⟩ := Multiset.mem_map.mp hx
          exact abs_nonneg w
      have hS0 : 0 ≤ S := by
        rw [hS]
        refine Finset.sum_nonneg fun i' _ => ?_
        by_cases hh : c0 d (i' : Nat) u < c1 d (i' : Nat) u
        · rw [ite_eq_left hh]; exact hFf0 i'
        · rw [ite_eq_right hh]
      have hSle : S ≤ tolDotR va vb d.K := by
        rw [hS]
        calc ∑ i' : Fin (launch d).size,
              (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Ff i' else 0)
            ≤ ∑ i' : Fin (launch d).size, Ff i' := by
              refine Finset.sum_le_sum fun i' _ => ?_
              by_cases hh : c0 d (i' : Nat) u < c1 d (i' : Nat) u
              · rw [ite_eq_left hh]
              · rw [ite_eq_right hh]; exact hFf0 i'
          _ ≤ tolDotR va vb d.K := by
              simpa only [hFf] using
                sum_fineOne_tile (d := d) hd hu (va := va) (vb := vb)
      have hB_le_S : B ≤ (∑ t ∈ Finset.range (steps d.K), |chunkSum va vb d.K t|) + S := by
        have h1 : B ≤ ∑ i' : Fin (launch d).size,
            (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then |Pp i'| + Ff i' else 0) := by
          rw [hB]
          exact Finset.sum_le_sum fun i' _ => hA_i i'
        have h2 : (∑ i' : Fin (launch d).size,
              (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then |Pp i'| + Ff i' else 0))
            = (∑ i' : Fin (launch d).size,
                (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then |Pp i'| else 0))
              + (∑ i' : Fin (launch d).size,
                (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Ff i' else 0)) := by
          rw [← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun i' _ => ?_
          by_cases hh : c0 d (i' : Nat) u < c1 d (i' : Nat) u <;> simp [hh]
        have h3 : (∑ i' : Fin (launch d).size,
              (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then |Pp i'| else 0))
            ≤ ∑ t ∈ Finset.range (steps d.K), |chunkSum va vb d.K t| := by
          have h := sum_abs_pre_sub_split (d := d) hd hu (va := va) (vb := vb)
          simpa only [hPp] using h
        have h4 : (∑ i' : Fin (launch d).size,
              (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Ff i' else 0)) = S := hS.symm
        linarith
      have hWs_le : |Ws - dotSum va vb d.K| ≤ S := by
        have hdot : dotSum va vb d.K = ∑ i' : Fin (launch d).size,
            (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Pp i' else 0) := by
          rw [sum_pre_sub_split (d := d) hd hu (va := va) (vb := vb)]
        rw [hdot, hWs, hS, ← Finset.sum_sub_distrib]
        calc |∑ i' : Fin (launch d).size,
              ((ops i').sum - (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Pp i' else 0))|
            ≤ ∑ i' : Fin (launch d).size,
              |(ops i').sum - (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Pp i' else 0)| :=
              Finset.abs_sum_le_sum_abs _ _
          _ ≤ ∑ i' : Fin (launch d).size,
              (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then Ff i' else 0) :=
              Finset.sum_le_sum fun i' _ => hV_i i'
      -- the room for the atomic bound
      have hroom : |z.toRat| +
          (∑ i' : Fin (launch d).size, (Multiset.map (fun w => |w|) (ops i')).sum) +
          DotLoop.bound aRho aAlpha (∑ i' : Fin (launch d).size, (ops i').card)
            (|z.toRat| + ∑ i' : Fin (launch d).size,
              (Multiset.map (fun w => |w|) (ops i')).sum) ≤ Format.binary32.maxVal := by
        have hA0 : (0 : ℚ) ≤ ∑ i' : Fin (launch d).size,
            (Multiset.map (fun w => |w|) (ops i')).sum := by
          rw [← hB]; exact hB0
        have hA : (∑ i' : Fin (launch d).size,
            (Multiset.map (fun w => |w|) (ops i')).sum) ≤ 2 ^ 53 := by
          calc (∑ i' : Fin (launch d).size, (Multiset.map (fun w => |w|) (ops i')).sum)
              ≤ ∑ i' : Fin (launch d).size,
                  (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then |Pp i'| + Ff i' else 0) :=
                Finset.sum_le_sum fun i' _ => hA_i i'
            _ ≤ ∑ i' : Fin (launch d).size,
                  (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then (2 : ℚ) ^ 46 else 0) := by
                refine Finset.sum_le_sum fun i' _ => ?_
                by_cases hh : c0 d (i' : Nat) u < c1 d (i' : Nat) u
                · rw [ite_eq_left hh, ite_eq_left hh]
                  have hP : |Pp i'| ≤ 2 ^ 45 := by
                    simpa only [hPp] using
                      abs_pre_sub_le_2_45 hd hva' hvb' (i' : Nat) u
                  have hF : Ff i' ≤ 2 ^ 38 := by
                    have hn : c1 d (i' : Nat) u - c0 d (i' : Nat) u ≤ 128 :=
                      le_trans (Nat.sub_le _ _) (le_trans c1_le (steps_le_128 hd.K_le))
                    simpa only [hFf] using
                      fineOne_le (va := va) (vb := vb) (K := d.K)
                        (c0 := c0 d (i' : Nat) u)
                        (n := c1 d (i' : Nat) u - c0 d (i' : Nat) u) hd.K_le hn hva' hvb'
                  linarith
                · rw [ite_eq_right hh, ite_eq_right hh]
            _ ≤ 2 ^ 53 := by
                have hcount : (∑ i' : Fin (launch d).size,
                    (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then (1 : ℚ) else 0)) ≤ 128 := by
                  have hnat := sum_count_le_steps (d := d) hd hu
                  have hcast : ((∑ i' : Fin (launch d).size,
                      (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then (1 : Nat) else 0) : Nat) : ℚ)
                      = ∑ i' : Fin (launch d).size,
                        (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then (1 : ℚ) else 0) := by
                    rw [Nat.cast_sum]
                    refine Finset.sum_congr rfl fun i' _ => ?_
                    split_ifs <;> simp
                  rw [← hcast]
                  have h2 : ((∑ i' : Fin (launch d).size,
                      (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then (1 : Nat) else 0) : Nat) : ℚ)
                      ≤ ((steps d.K : Nat) : ℚ) := by exact_mod_cast hnat
                  have h3 : ((steps d.K : Nat) : ℚ) ≤ 128 := by
                    exact_mod_cast steps_le_128 hd.K_le
                  linarith
                calc ∑ i' : Fin (launch d).size,
                      (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then (2 : ℚ) ^ 46 else 0)
                    = (2 : ℚ) ^ 46 * ∑ i' : Fin (launch d).size,
                        (if c0 d (i' : Nat) u < c1 d (i' : Nat) u then (1 : ℚ) else 0) := by
                      rw [Finset.mul_sum]
                      refine Finset.sum_congr rfl fun i' _ => ?_
                      by_cases hh : c0 d (i' : Nat) u < c1 d (i' : Nat) u <;> simp [hh]
                  _ ≤ 2 ^ 46 * 128 := by
                      have h0 : (0 : ℚ) ≤ 2 ^ 46 := by positivity
                      exact mul_le_mul_of_nonneg_left hcount h0
                  _ = 2 ^ 53 := by norm_num
        have hn128 : (∑ i' : Fin (launch d).size, (ops i').card) ≤ 128 :=
          le_trans hna_le (steps_le_128 hd.K_le)
        have key := room_le_maxVal
          (A := ∑ i' : Fin (launch d).size, (Multiset.map (fun w => |w|) (ops i')).sum)
          (n := ∑ i' : Fin (launch d).size, (ops i').card) hA0 hA hn128
        rw [hzrat]
        simpa using key
      -- the atomic replay of the cell
      obtain ⟨y, hyread, hygood, hyb⟩ :=
        replay_err_shared (d := .f32) (a := cAddr d i j) (val := FloatV.toRat) (good := good)
          (ρ := aRho) (α := aAlpha) (B := Format.binary32.maxVal)
          (N := (launch d).size) (G := G) (ops := ops) (m := init) (m' := mA) (x := z)
          aRho_nonneg aAlpha_nonneg hrep hPaired hz0 (by simpa [good] using hzfin) hroom
      have hcy : c = y := by
        have h1 : fin.read .f32 (cAddr d i j) = mA.read .f32 (cAddr d i j) := read_congr hspanC
        rw [hfc, hyread] at h1
        exact Option.some.inj h1
      have hmain : |y.toRat - dotSum va vb d.K| ≤ tol va vb d.K := by
        have h1 : |y.toRat - dotSum va vb d.K| ≤
            |y.toRat - Ws| + |Ws - dotSum va vb d.K| := by
          have e : y.toRat - dotSum va vb d.K =
              (y.toRat - Ws) + (Ws - dotSum va vb d.K) := by ring
          rw [e]
          exact abs_add_le _ _
        have h2 : |y.toRat - Ws| ≤ DotLoop.bound aRho aAlpha na B := by
          rw [hna, hB, hWs]
          simpa [hzrat] using hyb
        have h3 : DotLoop.bound aRho aAlpha na B ≤
            DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) na
              ((∑ t ∈ Finset.range (steps d.K), |chunkSum va vb d.K t|) + S) := by
          have h := bound_mono_B (ρ := aRho) (α := aAlpha) aRho_nonneg (t := na) hB_le_S
          simpa only [aRho, aAlpha] using h
        have h4 := tol_close2 (va := va) (vb := vb) (K := d.K) (na := na)
          hd.K_pos hd.K_le hna_le (e := |Ws - dotSum va vb d.K|) (S := S) hWs_le hS0 hSle
        linarith
      refine ⟨?_, ?_⟩
      · rw [hcy]; exact hygood
      · rw [hcy]; exact hmain
    · -- the tile is stored by a program
      have hex : ∃ p, p < d.num_programs ∧ owns d p u := by
        by_contra h
        exact hsplit fun p hp how => h ⟨p, hp, how⟩
      obtain ⟨p, hp, how⟩ := hex
      have hp' : p < (launch d).size := by rwa [launch_size]
      let pF : Fin (launch d).size := ⟨p, hp'⟩
      obtain ⟨m, hQm, hAgree⟩ := hprogs pF
      obtain ⟨L, _hLmap, _hAccL, _hFrameL, hValL, _hRunL, _hCntL⟩ := hQm
      obtain ⟨v, hreadv, hboundv⟩ := hValL u hu how (i, j) hqu
      have hspanW : ∀ x ∈ Mem.span .f32 (cAddr d i j), fin x = m x := by
        intro x hx
        exact hAgree x (Or.inr ⟨(i, j), ⟨u, hu, how, hqu⟩, hx⟩)
      refine ⟨v, (read_congr hspanW).trans hreadv, ?_⟩
      intro va vb hva hvb
      refine ⟨?_, ?_⟩
      · simpa [good] using (hboundv va vb hva hvb).1
      · have hb := (hboundv va vb hva hvb).2
        rw [fineOne_zero] at hb
        exact hb.trans (tolDotR_le_tol hd.K_pos hd.K_le)
  · -- the frame
    intro z hz
    refine hframe z ?_ ?_
    · intro hc
      exact hz (Cbytes_sub_cElems hc)
    · intro i hw
      exact hz (Wset_sub_cElems (i : Nat) hw)

end Claims.StreamKMatmul

