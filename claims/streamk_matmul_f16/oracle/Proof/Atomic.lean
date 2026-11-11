import Claim
import TVA.Proof
import Proof.Tail

/-!
# `Proof.Atomic`

The end of a run when the tile is split: the shared `tl.atomic_add`, and the safety of the tail.
-/

/-!
# The invariant across a list of atomic runs

This is the list version of `Qg_atomic` (`Proof/QgAtomic.lean`): `log` grows by every step of
`newL`, each entry pairing an atomic step with the value of each output element the run added.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-- **`Qg` across a list of atomic runs.**  `log` grows by the steps of `newL`, and each entry's
value function is supported exactly on the output elements of the tile `u`. -/
theorem Qg_atomics {init : Mem} {i it it' : Nat} {log : List AtomicStep} {m : Mem}
    {u : Nat} {newL : List (AtomicStep × (Nat × Nat → Option ℚ))}
    (hQg : Qg d init i it log m)
    (hS : 0 < steps d.K) (hu : u ∈ tiles d) (hit : it < hi d i)
    (hule : u * steps d.K ≤ it) (hult : it < (u + 1) * steps d.K)
    (hnex : it' = min ((u + 1) * steps d.K) (hi d i))
    (hne : c0 d i u < c1 d i u) (hns : ¬ owns d i u) (hsp : splitTile d u)
    (haccs : ∀ p ∈ newL, ∀ q ∈ splitOuts d, ∀ va vb : Nat → F16,
      ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
      AccStep .f32 (cAddr d q.1 q.2) FloatV.toRat good aRho aAlpha
        Format.binary32.maxVal p.1 (p.2 q))
    (hrun : ∀ p ∈ newL, ∀ q ∈ splitOuts d, ∀ w : ℚ, p.2 q = some w →
      RunVal d init i u q w)
    (hsupp : ∀ p ∈ newL, ∀ q ∈ splitOuts d, q ∉ tileOut d u → p.2 q = none)
    (hcnt1 : ∀ q ∈ splitOuts d, q ∈ tileOut d u →
      (newL.filter (fun p => (p.2 q).isSome)).length = 1)
    (hgate : ∀ w ∈ tiles d, w ≠ u → c0 d i w < c1 d i w → gate d i it' w → gate d i it w) :
    Qg d init i it' (log ++ newL.map Prod.fst) m := by
  obtain ⟨L, hL, hacc, hframe, hval, hrun_old, hcnt⟩ := hQg
  refine ⟨L ++ newL, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [List.map_append, hL]
  · intro p hp q hq va vb hva hvb
    rcases List.mem_append.mp hp with hp | hp
    · exact hacc p hp q hq va vb hva hvb
    · exact haccs p hp q hq va vb hva hvb
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
    · have hw' : p.2 q = some w := hw
      have hmem : q ∈ tileOut d u := by
        by_contra hc
        have hnone : p.2 q = none := hsupp p hp q hq hc
        rw [hnone] at hw'
        simp at hw'
      exact ⟨u, hu, hsp, hmem, hns, hrun p hp q hq w hw'⟩
  · intro q hq u0 hu0 hqu
    have hLf : ((L ++ newL).filter (fun p => (p.2 q).isSome)).length
        = (L.filter (fun p => (p.2 q).isSome)).length
          + (newL.filter (fun p => (p.2 q).isSome)).length := by
      rw [List.filter_append, List.length_append]
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
      have hnewlen : (newL.filter (fun p => (p.2 q).isSome)).length = 1 :=
        hcnt1 q hq hqu'
      have hgate_new : gate d i it' u0 := by
        rw [hwu]
        show min (u * steps d.K + steps d.K) (hi d i) ≤ it'
        rw [hnex, Nat.succ_mul]
      rw [hLf, hold, hnewlen]
      simp [hne', hgate_new]
    · have hnotmem : q ∉ tileOut d u := fun hq' =>
        Set.disjoint_left.mp (tileOut_disjoint hu0 hu hwu) hqu hq'
      have hnewlen : (newL.filter (fun p => (p.2 q).isSome)).length = 0 := by
        have hempty : newL.filter (fun p => (p.2 q).isSome) = [] := by
          apply List.filter_eq_nil_iff.mpr
          intro p hp
          have hvq : p.2 q = none := hsupp p hp q hq hnotmem
          simp [hvq]
        rw [hempty]
        rfl
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
      rw [hLf, hcnt q hq u0 hu0 hqu, hnewlen, add_zero]
      by_cases hA : (c0 d i u0 < c1 d i u0 ∧ gate d i it u0)
      · rw [ite_eq_left hA, ite_eq_left (hiff.mp hA)]
      · rw [ite_eq_right hA, ite_eq_right (fun hh => hA (hiff.mpr hh))]

end Claims.StreamKMatmul


/-!
# The atomic branch of one loop iteration

When program `i` does *not* own the tile `u = tileOf d it` of the iteration `it`, the tail of the
loop body adds the accumulated tile into `c` with one shared `atomic_add` per output element.  The
cells written are exactly the split tiles' cells (`Ccells`), the footprint is `Wv`, and the gated
invariant holds at the next counter `min ((u+1) · steps K) (hi d i)`.

This file should also record that the atomic run's old values read as the dot products of the
programs that stored the cells before (that is what `Qg_atomics` asks for).
-/

namespace Claims.StreamKMatmul

open TVA

set_option maxHeartbeats 800000

variable {d : Inst} {P : Platform}

/-! ## Output elements of a tile and the lanes that write them -/

/-- The output element `(i, j)` that the lane `k` of tile `u` writes. -/
private def qiOf (d : Inst) (u : Nat) (k : Idx [128, 128]) : Nat × Nat :=
  (pidMOf d u * 128 + k.1.val, pidNOf d u * 128 + k.2.1.val)

/-- The lane of the tile that writes the output element `q`. -/
private def laneOf (q : Nat × Nat) : Idx [128, 128] :=
  (⟨q.1 % 128, Nat.mod_lt _ (by norm_num)⟩, ⟨q.2 % 128, Nat.mod_lt _ (by norm_num)⟩, ())

/-- `laneOf` undoes `qiOf`. -/
private theorem laneOf_qiOf (d : Inst) (u : Nat) (k : Idx [128, 128]) :
    laneOf (qiOf d u k) = k := by
  obtain ⟨i, j, t⟩ := k
  cases t
  have h1 : (pidMOf d u * 128 + i.val) % 128 = i.val := by
    rw [Nat.mul_comm (pidMOf d u) 128, Nat.mul_add_mod, Nat.mod_eq_of_lt i.isLt]
  have h2 : (pidNOf d u * 128 + j.val) % 128 = j.val := by
    rw [Nat.mul_comm (pidNOf d u) 128, Nat.mul_add_mod, Nat.mod_eq_of_lt j.isLt]
  dsimp only [laneOf, qiOf]
  refine Prod.ext (Fin.ext ?_) (Prod.ext (Fin.ext ?_) rfl)
  · exact h1
  · exact h2

/-- `qiOf` undoes `laneOf`. -/
private theorem qiOf_laneOf {u : Nat} {q : Nat × Nat} (hq : q ∈ tileOut d u) :
    qiOf d u (laneOf q) = q := by
  have hqm : q.1 / 128 = pidMOf d u := hq.2.1
  have hqn : q.2 / 128 = pidNOf d u := hq.2.2
  have hm : pidMOf d u * 128 + q.1 % 128 = q.1 := by
    rw [Nat.mul_comm, ← hqm]
    exact Nat.div_add_mod q.1 128
  have hn : pidNOf d u * 128 + q.2 % 128 = q.2 := by
    rw [Nat.mul_comm, ← hqn]
    exact Nat.div_add_mod q.2 128
  dsimp only [qiOf, laneOf]
  exact Prod.ext hm hn

/-- Different lanes name different output elements. -/
private theorem qiOf_ne_of_laneOf_ne {u : Nat} {k : Idx [128, 128]} {q : Nat × Nat}
    (h : k ≠ laneOf q) : qiOf d u k ≠ q := by
  intro hq
  exact h (by rw [← hq]; exact (laneOf_qiOf d u k).symm)

/-- An enabled lane names an element of the tile. -/
private theorem qiOf_mem_tileOut {u : Nat} {k : Idx [128, 128]}
    (he : (fun i : Idx [128, 128] => decide (pidMOf d u * 128 + i.1.val < d.M) &&
        decide (pidNOf d u * 128 + i.2.1.val < d.N)) k = true) :
    pidMOf d u * 128 + k.1.val < d.M ∧ pidNOf d u * 128 + k.2.1.val < d.N ∧
      qiOf d u k ∈ tileOut d u := by
  have hq1 : pidMOf d u * 128 + k.1.val < d.M := by
    simp only [Bool.and_eq_true, decide_eq_true_eq] at he
    exact he.1
  have hq2 : pidNOf d u * 128 + k.2.1.val < d.N := by
    simp only [Bool.and_eq_true, decide_eq_true_eq] at he
    exact he.2
  have hd1 : (pidMOf d u * 128 + k.1.val) / 128 = pidMOf d u := by
    rw [Nat.mul_comm (pidMOf d u) 128, Nat.mul_add_div (by norm_num : 0 < 128),
      Nat.div_eq_of_lt k.1.isLt, Nat.add_zero]
  have hd2 : (pidNOf d u * 128 + k.2.1.val) / 128 = pidNOf d u := by
    rw [Nat.mul_comm (pidNOf d u) 128, Nat.mul_add_div (by norm_num : 0 < 128),
      Nat.div_eq_of_lt k.2.1.isLt, Nat.add_zero]
  exact ⟨hq1, hq2, ⟨⟨hq1, hq2⟩, hd1, hd2⟩⟩

/-- The mask of the `c` atomic add, as the arithmetic side of the addresses. -/
private def enC (d : Inst) (u : Nat) : Idx [128, 128] → Bool :=
  fun i => decide (pidMOf d u * 128 + i.1.val < d.M) &&
    decide (pidNOf d u * 128 + i.2.1.val < d.N)

/-- An enabled lane of the `c` atomic add has its arithmetic mask true. -/
private theorem lane_en {pm pn : I32} {u : Nat} {k : Idx [128, 128]}
    (hd : d.ok) (hpm : pm.toInt = (pidMOf d u : Int)) (hpn : pn.toInt = (pidNOf d u : Int))
    (hu : u < tilesM d * tilesN d)
    (hk : (tl.addrs
        (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
          (IntV.ofNat d.stride_cn))
        (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N))) k).isSome
      = true) :
    enC d u k = true := by
  by_contra hne
  simp only [enC] at hne
  rw [(cAddrs (u := u) (d := d) hd hpm hpn hu) k, ite_eq_right hne] at hk
  exact absurd hk (by simp)

/-- The address of an enabled lane of the `c` atomic add. -/
private theorem lane_addr {pm pn : I32} {u : Nat} {k : Idx [128, 128]}
    (hd : d.ok) (hpm : pm.toInt = (pidMOf d u : Int)) (hpn : pn.toInt = (pidNOf d u : Int))
    (hu : u < tilesM d * tilesN d) (he : enC d u k = true) :
    ((ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
      (IntV.ofNat d.stride_cn)).get k).addr = cAddr d (qiOf d u k).1 (qiOf d u k).2 := by
  have hc := (cAddrs (u := u) (d := d) hd hpm hpn hu) k
  simp only [enC] at he
  rw [ite_eq_left he] at hc
  simp only [tl.addrs] at hc
  by_cases hm : (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)).get k
      = true
  · rw [ite_eq_left hm, Option.some.injEq] at hc
    exact hc
  · rw [ite_eq_right hm] at hc
    exact absurd hc (by simp)

/-- The value this run adds at each output element: the accumulator element of its lane. -/
private def valsiOf (d : Inst) (u : Nat) (acc : Tensor [128, 128] F32) (k : Idx [128, 128]) :
    Nat × Nat → Option ℚ :=
  fun q => if q = qiOf d u k then some (acc.get k).toRat else none

/-- On an element the atomic updates, `valsiOf` is the lane's operand. -/
private theorem valsiOf_some {u : Nat} {acc : Tensor [128, 128] F32} {k : Idx [128, 128]}
    {q : Nat × Nat} (h : q = qiOf d u k) :
    valsiOf d u acc k q = some (acc.get k).toRat := by
  simp only [valsiOf, ite_eq_left h]

/-- On any other element, `valsiOf` is `none`. -/
private theorem valsiOf_none {u : Nat} {acc : Tensor [128, 128] F32} {k : Idx [128, 128]}
    {q : Nat × Nat} (h : q ≠ qiOf d u k) : valsiOf d u acc k q = none := by
  simp only [valsiOf, ite_eq_right h]

/-- A list with exactly one element satisfying `p`, namely `a`. -/
private theorem filter_eq_singleton_of {α : Type} {l : List α} {p : α → Bool} {a : α}
    (hnd : l.Nodup) (ha : a ∈ l) (hpa : p a = true)
    (hother : ∀ x ∈ l, x ≠ a → p x = false) : l.filter p = [a] := by
  induction l with
  | nil => exact absurd ha (by simp)
  | cons b t ih =>
    rw [List.nodup_cons] at hnd
    obtain ⟨hbt, hndt⟩ := hnd
    by_cases hba : b = a
    · subst b
      rw [List.filter_cons, ite_eq_left hpa]
      have hempty : t.filter p = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro x hx hpx
        have hxa : x ≠ a := by
          rintro rfl
          exact hbt hx
        have hfx : p x = false := hother x (List.mem_cons_of_mem a hx) hxa
        rw [hfx] at hpx
        exact Bool.false_ne_true hpx
      rw [hempty]
    · have hpb : p b = false := hother b List.mem_cons_self hba
      have hpbne : ¬ (p b = true) := by rw [hpb]; exact Bool.false_ne_true
      rw [List.filter_cons, ite_eq_right hpbne]
      have hat : a ∈ t := by
        rcases List.mem_cons.mp ha with h | h
        · exact absurd h.symm hba
        · exact h
      exact ih hndt hat (fun x hx hxa => hother x (List.mem_cons_of_mem b hx) hxa)

/-- A list with one element satisfying the predicate has a one-element filter. -/
private theorem length_filter_eq_one_of {α : Type} {l : List α} {p : α → Bool} {a : α}
    (hnd : l.Nodup) (ha : a ∈ l) (hpa : p a = true)
    (hother : ∀ x ∈ l, x ≠ a → p x = false) : (l.filter p).length = 1 := by
  rw [filter_eq_singleton_of hnd ha hpa hother, List.length_singleton]

/-- **The atomic branch.**  Program `i` does not own tile `u`: it adds the tile's partial sums into
the split cells of `c`, and the invariant advances to the next counter. -/
theorem tail_atomic (hP : Device P) (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i it : Nat} (hpi : i < d.num_programs) (hlo : lo d i ≤ it) (hit : it < hi d i)
    {stv kf kl pm pn : I32} {acc : Tensor [128, 128] F32} {m1 : Mem} {log1 : List AtomicStep}
    (hstv : stv.toInt = (it : Int))
    (hkf : kf.toInt = (c0 d i (tileOf d it) : Int))
    (hkl : kl.toInt = (c1 d i (tileOf d it) : Int))
    (hpm : pm.toInt = (pidMOf d (tileOf d it) : Int))
    (hpn : pn.toInt = (pidNOf d (tileOf d it) : Int))
    (hQg : Qg d init i it log1 m1) (hpres : Pres d m1)
    (hacc : AccI d init i (tileOf d it) (c1 d i (tileOf d it) - c0 d i (tileOf d it)) acc)
    (hns : ¬ owns d i (tileOf d it))
    (hsp : splitTile d (tileOf d it))
    (hrc : it = lo d i ∨ it % steps d.K = 0) :
    SafeA ((launch d).env P) (inElems d) (Wv d i (mu d i stv)) (Ccells d)
      (fun st' log m => Inv d init i st' log m ∧ mu d i st' < mu d i stv) log1 m1
      (tl.atomic_add
          (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
            (IntV.ofNat d.stride_cn))
          acc
          (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))
        >>= fun _ => pure (stv + (kl - kf))) := by
  have hS : 0 < steps d.K := T_pos hd
  set u : Nat := tileOf d it with hu
  have huT : u < T d := by rw [hu]; exact tileOf_lt_T hd hit
  have hut : u < tilesM d * tilesN d := by simpa only [T] using huT
  have hutiles : u ∈ tiles d := by rw [mem_tiles]; exact hut
  have hkg : kFirstOf d it = c0 d i u := by
    simpa only [hu] using kFirstOf_eq_c0 (d := d) (i := i) (it := it) hd hlo hit hrc
  have hklg : kLastOf d it (hi d i) = c1 d i u := by
    simpa only [hu] using kLastOf_eq_c1 (d := d) (i := i) (it := it) hit
  have hne : c0 d i u < c1 d i u := by
    rw [← hkg, ← hklg]
    show kFirstOf d it < min (steps d.K) (kFirstOf d it + (hi d i - it))
    exact lt_min (by unfold kFirstOf; exact Nat.mod_lt _ hS) (by omega)
  have hcle : c0 d i u ≤ c1 d i u := le_of_lt hne
  have haccu : AccI d init i u (c1 d i u - c0 d i u) acc := by simpa only [hu] using hacc
  have hnsu : ¬ owns d i u := by simpa only [hu] using hns
  have hspu : splitTile d u := by simpa only [hu] using hsp
  have hpmu : pm.toInt = (pidMOf d u : Int) := by rw [hu]; exact hpm
  have hpnu : pn.toInt = (pidNOf d u : Int) := by rw [hu]; exact hpn
  have hget : ∀ k : Idx [128, 128], acc.get k = acc.get2 k.1 k.2.1 := by
    intro k
    obtain ⟨ii, jj, t⟩ := k
    cases t
    rfl
  -- the next counter
  have hle : u * steps d.K ≤ it := by
    have hdec := tile_decomp (d := d) (it := it)
    rw [← hu] at hdec
    omega
  have hult : it < (u + 1) * steps d.K := by
    have hdec := tile_decomp (d := d) (it := it)
    rw [← hu] at hdec
    have hkf : kFirstOf d it < steps d.K := by unfold kFirstOf; exact Nat.mod_lt _ hS
    rw [Nat.succ_mul]
    omega
  have hc1lt : c1 d i u < 2 ^ 31 := by
    have h1 := c1_le (d := d) (p := i) (u := u)
    have h2 := steps_le_128 hd.K_le
    omega
  have hit' : it + (c1 d i u - c0 d i u) = min ((u + 1) * steps d.K) (hi d i) := by
    have h := it_next (d := d) (i := i) (it := it) hd hit
    rw [hklg, hkg, ← hu] at h
    exact h
  have hit31 : it + (c1 d i u - c0 d i u) < 2 ^ 31 := by
    have h1 : min ((u + 1) * steps d.K) (hi d i) ≤ lo d i + d.iters_per_program := by
      refine le_trans (min_le_right _ _) ?_
      rw [← succ_mul_ipp d i]
      exact hi_le_succ d i
    have h2 := lo_add_ipp_lt (d := d) (p := i) hd hpi
    omega
  have hsub : (kl - kf).toInt = ((c1 d i u - c0 d i u : Nat) : Int) :=
    kLast_sub_kFirst (kl := kl) (kf := kf) (r := kl - kf) hkl hkf hcle hc1lt rfl
  have hadd : (stv + (kl - kf)).toInt = ((it + (c1 d i u - c0 d i u) : Nat) : Int) :=
    IntV.I32.add_nat hstv hsub hit31
  have hitN : itN (stv + (kl - kf)) = it + (c1 d i u - c0 d i u) := by
    rw [itN, hadd, Int.toNat_natCast]
  have hnextU : itN (stv + (kl - kf)) = min ((u + 1) * steps d.K) (hi d i) := by
    rw [hitN, hit']
  have hnextT : itN (stv + (kl - kf)) = min ((tileOf d it + 1) * steps d.K) (hi d i) := by
    rw [hitN]; exact hit'
  have hstIt : it = itN stv := by
    rw [itN, hstv, Int.toNat_natCast]
  have hvalid : d.c.Valid m1 :=
    valid_of_buf (dd := .f32) (b := d.c) hpre.2.2.1 hpres (cBytes_sub (d := d))
  -- the atomic
  refine SafeA.env ?_
  refine SafeA.atomic (d := .f32)
    (ptrs := ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
      (IntV.ofNat d.stride_cn))
    (upd := fun k => tl.addUpd ((launch d).env P) .f32 (acc.get k))
    (mask := some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))
    (sem := .acqRel) (sc := .gpu) (k := .ret (stv + (kl - kf)))
    ?_ rfl (by decide) ?_ ?_
  · -- every enabled lane is an aligned cell of `Ccells`
    intro k hk
    have hen : enC d u k = true :=
      lane_en (d := d) (pm := pm) (pn := pn) (u := u) hd hpmu hpnu hut hk
    obtain ⟨hq1, hq2, hqT⟩ := qiOf_mem_tileOut (d := d) (u := u) (k := k) hen
    have haddr := lane_addr (d := d) (pm := pm) (pn := pn) (u := u) hd hpmu hpnu hut hen
    refine ⟨?_, ?_⟩
    · rw [haddr]
      exact cAligned (d := d) (m := m1) hd hvalid (tileOut_sub_allOuts u hqT)
    · rw [haddr]
      exact cCells_mem (d := d) ⟨u, hutiles, hspu, hqT⟩
  · -- the update is total
    intro k old
    exact hP.atomicAddTotal _ old (acc.get k)
  · -- the run
    intro order horder olds news hup
    let stepOf : Idx [128, 128] → AtomicStep :=
      fun k => (⟨.f32, ((ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128)
        (IntV.ofNat d.stride_cm) (IntV.ofNat d.stride_cn)).get k).addr, olds k, news k⟩ : AtomicStep)
    let newL : List (AtomicStep × (Nat × Nat → Option ℚ)) :=
      order.map fun k => (stepOf k, valsiOf d u acc k)
    have hnewL : newL = order.map (fun k => (stepOf k, valsiOf d u acc k)) := rfl
    have hlog : newL.map Prod.fst = order.map stepOf := by
      rw [hnewL, List.map_map]
      exact List.map_congr_left (fun a _ => rfl)
    have hnd : order.Nodup := horder.nodup_iff.mpr (enabled_nodup (tl.addrs
      (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
        (IntV.ofNat d.stride_cn))
      (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))))
    have hEn : ∀ k ∈ order, (tl.addrs
        (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
          (IntV.ofNat d.stride_cn))
        (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M)
          (IntV.ofNat d.N))) k).isSome = true := by
      intro k hk
      have h1 : k ∈ enabled (tl.addrs
          (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
            (IntV.ofNat d.stride_cn))
          (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M)
            (IntV.ofNat d.N)))) := horder.mem_iff.mp hk
      exact (List.mem_filter.mp h1).2
    have haccs : ∀ p ∈ newL, ∀ q ∈ splitOuts d, ∀ va vb : Nat → F16,
        ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
        AccStep .f32 (cAddr d q.1 q.2) FloatV.toRat good aRho aAlpha
          Format.binary32.maxVal p.1 (p.2 q) := by
      intro p hp q hq va vb hva hvb
      rw [hnewL] at hp
      obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hp
      dsimp only [stepOf]
      have hen : enC d u k = true :=
        lane_en (d := d) (pm := pm) (pn := pn) (u := u) hd hpmu hpnu hut (hEn k hk)
      obtain ⟨hq1, hq2, hqT⟩ := qiOf_mem_tileOut (d := d) (u := u) (k := k) hen
      by_cases hqk : q = qiOf d u k
      · rw [valsiOf_some hqk, lane_addr (d := d) (pm := pm) (pn := pn) (u := u) hd hpmu
          hpnu hut hen, hqk]
        obtain ⟨v, hv, hgood, hb⟩ := haccu k.1 k.2.1 hq1 hq2 va vb
          (by simpa only [hqk, qiOf] using hva) (by simpa only [hqk, qiOf] using hvb)
        have hacc_k : acc.get k = v := by rw [hget k]; exact hv
        have hatomic : Device.AtomicAddAcc P Format.binary32
            (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) := by
          simpa only [Device.AtomicAddF32, aRho, aAlpha] using hP.atomicAddF32
        have hnew : P.atomicAdd Format.binary32 (olds k) v (news k) := by
          rw [← hacc_k]
          exact hup k hk
        have hvfin : FloatV.isFinite v = true := by
          simpa only [good] using hgood
        have hstep := accStep_atomicAdd (rel := 2 ^ (-23 : ℤ)) (abs := 2 ^ (-124 : ℤ))
          (a := cAddr d (qiOf d u k).1 (qiOf d u k).2) (old := olds k) (v := v)
          (new := news k) hatomic hnew hvfin
        rw [hacc_k]
        exact hstep
      · rw [valsiOf_none hqk, lane_addr (d := d) (pm := pm) (pn := pn) (u := u) hd hpmu
          hpnu hut hen]
        refine accStep_other ?_
        exact (span_disjoint_of_ne (d := d) (q := q) (q' := qiOf d u k) hd
          (splitOuts_sub_allOuts hq) (tileOut_sub_allOuts u hqT) hqk).symm
    have hrun : ∀ p ∈ newL, ∀ q ∈ splitOuts d, ∀ w : ℚ, p.2 q = some w →
        RunVal d init i u q w := by
      intro p hp q hq w hw
      rw [hnewL] at hp
      obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hp
      change valsiOf d u acc k q = some w at hw
      have hen : enC d u k = true :=
        lane_en (d := d) (pm := pm) (pn := pn) (u := u) hd hpmu hpnu hut (hEn k hk)
      obtain ⟨hq1, hq2, hqT⟩ := qiOf_mem_tileOut (d := d) (u := u) (k := k) hen
      by_cases hqk : q = qiOf d u k
      · rw [valsiOf_some hqk] at hw
        have hsome : (acc.get k).toRat = w := Option.some.inj hw
        have h := accI_runVal (d := d) (init := init) (i := i) (u := u) (acc := acc)
          haccu hcle (ii := k.1) (jj := k.2.1) hq1 hq2
        rw [hqk, ← hsome, hget k]
        exact h
      · rw [valsiOf_none hqk] at hw
        exact absurd hw (by simp)
    have hsupp : ∀ p ∈ newL, ∀ q ∈ splitOuts d, q ∉ tileOut d u → p.2 q = none := by
      intro p hp q hq hqnot
      rw [hnewL] at hp
      obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hp
      change valsiOf d u acc k q = none
      have hen : enC d u k = true :=
        lane_en (d := d) (pm := pm) (pn := pn) (u := u) hd hpmu hpnu hut (hEn k hk)
      exact valsiOf_none (fun hqk => hqnot (by
        rw [hqk]
        exact (qiOf_mem_tileOut (d := d) (u := u) (k := k) hen).2.2))
    have hcnt1 : ∀ q ∈ splitOuts d, q ∈ tileOut d u →
        (newL.filter (fun p => (p.2 q).isSome)).length = 1 := by
      intro q hq hqt
      have hq1 : q.1 < d.M := hqt.1.1
      have hq2 : q.2 < d.N := hqt.1.2
      have hqm : q.1 / 128 = pidMOf d u := hqt.2.1
      have hqn : q.2 / 128 = pidNOf d u := hqt.2.2
      have h1 : (laneOf q).1.val = q.1 % 128 := rfl
      have h2 : (laneOf q).2.1.val = q.2 % 128 := rfl
      have hm : pidMOf d u * 128 + q.1 % 128 = q.1 := by
        rw [Nat.mul_comm, ← hqm]
        exact Nat.div_add_mod q.1 128
      have hn : pidNOf d u * 128 + q.2 % 128 = q.2 := by
        rw [Nat.mul_comm, ← hqn]
        exact Nat.div_add_mod q.2 128
      have henq : enC d u (laneOf q) = true := by
        simp only [enC, h1, h2, hm, hn, Bool.and_eq_true, decide_eq_true_eq]
        exact ⟨hq1, hq2⟩
      have hAq : (tl.addrs
          (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
            (IntV.ofNat d.stride_cn))
          (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M)
            (IntV.ofNat d.N)))) (laneOf q) = some (cAddr d q.1 q.2) := by
        have h := cAddr_of_lane (d := d) (pm := pm) (pn := pn) (u := u) hd hpmu hpnu hut
          (i := laneOf q) henq
        rw [h]
        simp only [h1, h2, hm, hn]
      have hmem : laneOf q ∈ order := horder.mem_iff.mpr (mem_enabled hAq)
      have hcnt1q : (newL.filter (fun p : AtomicStep × (Nat × Nat → Option ℚ) =>
            (p.2 q).isSome)).length
          = (order.filter (fun k => (valsiOf d u acc k q).isSome)).length := by
        rw [hnewL, List.filter_map, List.length_map]
        congr 1
      rw [hcnt1q]
      exact length_filter_eq_one_of (l := order) (p := fun k => (valsiOf d u acc k q).isSome)
        (a := laneOf q) hnd hmem
        (by rw [valsiOf_some (qiOf_laneOf (d := d) (u := u) (q := q) hqt).symm]; rfl)
        (fun x hx hxne => by
          rw [valsiOf_none (fun hqk => qiOf_ne_of_laneOf_ne (d := d) (u := u) (k := x)
            (q := q) hxne hqk.symm)]
          rfl)
    have hgate : ∀ w ∈ tiles d, w ≠ u → c0 d i w < c1 d i w →
        gate d i (itN (stv + (kl - kf))) w → gate d i it w := by
      intro w hw hwu hlt hg
      exact gate_prev (d := d) (u := u) hd hit hle hult hnextU hlt hwu hg
    have hQg' : Qg d init i (itN (stv + (kl - kf))) (log1 ++ newL.map Prod.fst) m1 :=
      Qg_atomics (u := u) (newL := newL) (it' := itN (stv + (kl - kf)))
        hQg hS hutiles hit hle hult hnextU hne hnsu hspu haccs hrun hsupp hcnt1 hgate
    rw [hlog] at hQg'
    refine SafeA.ret ?_
    refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
    · rw [hadd]
      exact Int.natCast_nonneg _
    · rw [hnextT]
      exact lo_le_next (d := d) (i := i) (it := it) hd hlo hit
    · exact hQg'
    · exact hpres
    · rw [hnextT]
      intro hlt
      rcases reach_step (d := d) (i := i) (it := it) hit with h | h | h
      · exact Or.inl h
      · exact Or.inr h
      · omega
    · exact mu_step (d := d) (i := i) (it := it) (st := stv) (st' := stv + (kl - kf))
        (it' := itN (stv + (kl - kf))) hd hit hstIt rfl hnextT

end Claims.StreamKMatmul


/-!
# The tail of one loop iteration

The tail of the loop body branches on a runtime `if`: when `k_first == 0` and
`k_last == iters_per_tile` the program owns the tile `u = tileOf d it` and stores it; otherwise it
adds the tile's partial sums into the split tiles with an `atomic_add` per output element.  The
model's `Prog` has no `ite` constructor, so the branch is taken on the decidable equality of the
condition with `true` and the respective branch lemma applies.

This is the `body` of the outer loop, after the inner `forRange` over `k`.
-/

namespace Claims.StreamKMatmul

open TVA

set_option maxHeartbeats 800000

variable {d : Inst} {P : Platform}

/-- **The tail of one iteration.**  One `if` chooses between the store of the owned tile and the
atomic add into the split cells; either way the invariant advances past `it`. -/
theorem tail_safe (hP : Device P) (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i it : Nat} (hpi : i < d.num_programs) (hlo : lo d i ≤ it) (hit : it < hi d i)
    {stv kf kl pm pn itp : I32} {acc : Tensor [128, 128] F32} {m1 : Mem} {log1 : List AtomicStep}
    (hstv : stv.toInt = (it : Int))
    (hkf : kf.toInt = (c0 d i (tileOf d it) : Int))
    (hkl : kl.toInt = (c1 d i (tileOf d it) : Int))
    (hpm : pm.toInt = (pidMOf d (tileOf d it) : Int))
    (hpn : pn.toInt = (pidNOf d (tileOf d it) : Int))
    (hitp : itp.toInt = (steps d.K : Int))
    (hQg : Qg d init i it log1 m1) (hpres : Pres d m1)
    (hacc : AccI d init i (tileOf d it) (c1 d i (tileOf d it) - c0 d i (tileOf d it)) acc)
    (hrc : it = lo d i ∨ it % steps d.K = 0) :
    SafeA ((launch d).env P) (inElems d) (Wv d i (mu d i stv)) (Ccells d)
      (fun st' log m => Inv d init i st' log m ∧ mu d i st' < mu d i stv) log1 m1
      (if (IntV.eq kf (IntV.ofInt 0 : I32) && IntV.eq kl itp) = true then
          tl.store
            (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
              (IntV.ofNat d.stride_cn))
            acc
            (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))
          >>= fun _ => pure (stv + (kl - kf))
        else
          tl.atomic_add
            (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
              (IntV.ofNat d.stride_cn))
            acc
            (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))
          >>= fun _ => pure (stv + (kl - kf))) := by
  classical
  by_cases hcond : (IntV.eq kf (IntV.ofInt 0 : I32) && IntV.eq kl itp) = true
  · rw [ite_eq_left hcond]
    exact tail_store hP hd hpre hpi hlo hit hstv hkf hkl hpm hpn hQg hpres hacc
      ((eq_and_eq_iff hkf hkl hitp rfl).mp hcond) hrc
  · rw [ite_eq_right hcond]
    have hns : ¬ owns d i (tileOf d it) :=
      fun ho => hcond ((eq_and_eq_iff hkf hkl hitp rfl).mpr ho)
    have hc : c0 d i (tileOf d it) < c1 d i (tileOf d it) := by
      have h1 : kFirstOf d it = c0 d i (tileOf d it) := kFirstOf_eq_c0 hd hlo hit hrc
      have h2 : kLastOf d it (hi d i) = c1 d i (tileOf d it) := kLastOf_eq_c1 hit
      rw [← h1, ← h2]
      have hlt : kFirstOf d it < steps d.K := by
        unfold kFirstOf
        exact Nat.mod_lt _ (T_pos hd)
      unfold kLastOf
      omega
    have hu : tileOf d it ∈ tiles d := by
      rw [mem_tiles]
      exact tileOf_lt_T hd hit
    exact tail_atomic hP hd hpre hpi hlo hit hstv hkf hkl hpm hpn hQg hpres hacc hns
      (splitTile_of_not_owns hd hu hpi hns hc) hrc

end Claims.StreamKMatmul

