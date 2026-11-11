import Claim
import TVA.Proof
import Proof.Iter

/-!
# `Proof.Tail`

The end of a run when the program owns the whole tile: its `tl.store`, and the uniqueness of the owner.
-/

/-!
# The invariant across one store

When the program owns the tile `u` it is running, the run ends with a store of the accumulator to
the tile's output elements.  This file shows that `Qg` is preserved: the log is unchanged and the
memory gains the tile's values, while every other tile's cells read as before.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-- **`Qg` across the store of an owned tile.** -/
theorem Qg_store {init : Mem} {i it it' : Nat} {log : List AtomicStep} {m m' : Mem} {u : Nat}
    (hQg : Qg d init i it log m)
    (hS : 0 < steps d.K) (hu : u ∈ tiles d) (hit : it < hi d i)
    (hule : u * steps d.K ≤ it) (hult : it < (u + 1) * steps d.K)
    (hnex : it' = min ((u + 1) * steps d.K) (hi d i))
    (hne : c0 d i u < c1 d i u) (ho : owns d i u)
    (hframe : ∀ x, x ∉ touchedBytes d i → m' x = init x)
    (hnew : ∀ q ∈ tileOut d u, ∃ v : F32, m'.read .f32 (cAddr d q.1 q.2) = some v ∧
      ∀ va vb : Nat → F16, ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
        good v ∧ |v.toRat - dotSum va vb d.K| ≤ fineOne va vb d.K 0 (steps d.K))
    (hkeep : ∀ w ∈ tiles d, w ≠ u → ∀ q ∈ tileOut d w,
      m'.read .f32 (cAddr d q.1 q.2) = m.read .f32 (cAddr d q.1 q.2))
    (hgate : ∀ w ∈ tiles d, w ≠ u → c0 d i w < c1 d i w → gate d i it' w → gate d i it w)
    (hsplit : ∀ q ∈ splitOuts d, q ∈ tileOut d u → False) :
    Qg d init i it' log m' := by
  obtain ⟨L, hm, hacc, _hframe, hval, hrun, hcnt⟩ := hQg
  have hle : it ≤ it' := by
    rw [hnex]
    exact le_min (le_of_lt hult) (le_of_lt hit)
  refine ⟨L, hm, hacc, hframe, ?_, hrun, ?_⟩
  · intro w hw ho hg q hq
    by_cases hwu : w = u
    · subst hwu
      exact hnew q hq
    · have hlt : c0 d i w < c1 d i w := by
        obtain ⟨h0, h1⟩ := ho
        rw [h0, h1]
        exact hS
      obtain ⟨v, hv, hspec⟩ := hval w hw ho (hgate w hw hwu hlt hg) q hq
      exact ⟨v, by rw [hkeep w hw hwu q hq]; exact hv, hspec⟩
  · intro q hq w hw hqw
    by_cases hwu : w = u
    · subst hwu
      exact (hsplit q hq hqw).elim
    · rw [hcnt q hq w hw hqw]
      by_cases hlt : c0 d i w < c1 d i w
      · have hgiff : gate d i it w ↔ gate d i it' w :=
          ⟨fun h => gate_mono hle h, fun h => hgate w hw hwu hlt h⟩
        exact if_congr (and_congr_right (fun _ => hgiff)) rfl rfl
      · rw [ite_eq_right (fun h : c0 d i w < c1 d i w ∧ gate d i it w => hlt h.1),
          ite_eq_right (fun h : c0 d i w < c1 d i w ∧ gate d i it' w => hlt h.1)]

end Claims.StreamKMatmul


/-!
# Helpers for the tail of the loop body

The tail of the loop body is where a program hands its tile to `c`: either it stores the tile it
owns, or it adds the tile's partial sums atomically into the split tiles.  Both cases need

* the `c` lanes of the tile to be aligned and to lie in the atomic cells `Ccells` (the store branch
  needs `StoreOk`, the atomic branch needs the `hcell` hypothesis of `SafeA.atomic`);
* the lanes of a tile to be pairwise non-overlapping (`DistinctLanes`), which makes the store /
  the atomic run read back the lane's own value;
* a frame law: a read from a `c` cell is unchanged by writes to lanes whose spans avoid it.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

set_option maxHeartbeats 1000000

/-- The address of an output element is aligned. -/
theorem cAligned {m : Mem} (hd : d.ok) (hv : d.c.Valid m) {q : Nat × Nat} (hq : q ∈ allOuts d) :
    Mem.Aligned .f32 (cAddr d q.1 q.2) := by
  have h := Buf.aligned d.c hv (hd.c_len q.1 q.2 hq.1 hq.2)
  simpa only [cAddr, Buf.ptr] using h

/-- The cell of an output element that is not owned by anyone is an atomic cell. -/
theorem cCells_mem {q : Nat × Nat} (hq : q ∈ splitOuts d) :
    (.f32, cAddr d q.1 q.2) ∈ Ccells d :=
  ⟨q, hq, rfl⟩

/-- The enabled `c` lanes of the tile `u` write their own output element. -/
theorem cAddr_of_lane (hd : d.ok) {pm pn : I32} {u : Nat}
    (hpm : pm.toInt = (pidMOf d u : Int)) (hpn : pn.toInt = (pidNOf d u : Int))
    (hu : u < tilesM d * tilesN d) {i : Idx [128, 128]}
    (he : (fun i : Idx [128, 128] =>
        decide (pidMOf d u * 128 + i.1.val < d.M) &&
          decide (pidNOf d u * 128 + i.2.1.val < d.N)) i = true) :
    tl.addrs
        (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
          (IntV.ofNat d.stride_cn))
        (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N))) i =
      some (cAddr d (pidMOf d u * 128 + i.1.val) (pidNOf d u * 128 + i.2.1.val)) := by
  rw [(cAddrs (u := u) hd hpm hpn hu) i, ite_eq_left he]
  rfl

/-- **The `c` lanes of one tile are distinct lanes.** -/
theorem cDistinct (hd : d.ok) {pm pn : I32} {u : Nat}
    (hpm : pm.toInt = (pidMOf d u : Int)) (hpn : pn.toInt = (pidNOf d u : Int))
    (hu : u < tilesM d * tilesN d) :
    DistinctLanes .f32
      (tl.addrs
          (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
            (IntV.ofNat d.stride_cn))
        (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))) := by
  intro i j a b hij ha hb
  by_cases hei : (fun i : Idx [128, 128] =>
      decide (pidMOf d u * 128 + i.1.val < d.M) &&
        decide (pidNOf d u * 128 + i.2.1.val < d.N)) i = true
  · by_cases hej : (fun i : Idx [128, 128] =>
        decide (pidMOf d u * 128 + i.1.val < d.M) &&
          decide (pidNOf d u * 128 + i.2.1.val < d.N)) j = true
    · rw [cAddr_of_lane hd hpm hpn hu hei] at ha
      rw [cAddr_of_lane hd hpm hpn hu hej] at hb
      rw [Option.some.injEq] at ha hb
      rw [← ha, ← hb]
      have hq : (pidMOf d u * 128 + i.1.val, pidNOf d u * 128 + i.2.1.val) ∈ allOuts d := by
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hei
        exact hei
      have hq' : (pidMOf d u * 128 + j.1.val, pidNOf d u * 128 + j.2.1.val) ∈ allOuts d := by
        simp only [Bool.and_eq_true, decide_eq_true_eq] at hej
        exact hej
      have hne : (pidMOf d u * 128 + i.1.val, pidNOf d u * 128 + i.2.1.val)
          ≠ (pidMOf d u * 128 + j.1.val, pidNOf d u * 128 + j.2.1.val) := by
        intro he
        apply hij
        have h1 : i.1.val = j.1.val := by
          have := congrArg (fun p : Nat × Nat => p.1) he
          simp only at this
          omega
        have h2 : i.2.1.val = j.2.1.val := by
          have := congrArg (fun p : Nat × Nat => p.2) he
          simp only at this
          omega
        exact Prod.ext (Fin.ext h1) (Prod.ext (Fin.ext h2) (by cases i.2.2; cases j.2.2; rfl))
      exact span_disjoint_of_ne (q := (pidMOf d u * 128 + i.1.val, pidNOf d u * 128 + i.2.1.val))
        (q' := (pidMOf d u * 128 + j.1.val, pidNOf d u * 128 + j.2.1.val)) hd hq hq' hne
    · rw [(cAddrs (u := u) hd hpm hpn hu) j] at hb
      rw [ite_eq_right hej] at hb
      exact absurd hb (by simp)
  · rw [(cAddrs (u := u) hd hpm hpn hu) i] at ha
    rw [ite_eq_right hei] at ha
    exact absurd ha (by simp)

/-- A read of `a` is unchanged by writes to lanes whose spans avoid `Mem.span dd' a`. -/
theorem writeAll_read_of_avoid {dd dd' : DType} {m : Mem} {s : Shape} {addrs : Idx s → Option Int}
    {vals : Tensor s dd.Val} {order : List (Idx s)} {S : Set Int} {a : Int}
    (havoid : ∀ k ∈ order, ∀ b, addrs k = some b → Disjoint (Mem.span dd b) S)
    (hS : Mem.span dd' a ⊆ S) :
    (writeAll m dd addrs vals order).read dd' a = m.read dd' a := by
  refine Mem.read_congr ?_
  intro x hx
  exact writeAll_frame_of havoid x (hS hx)

end Claims.StreamKMatmul


/-!
# Which tiles a program may find split, and when they become available

The kernel's iteration range of program `i` is contiguous, so at most one tile is *partly*
handled: all earlier tiles are finished (they are not owned only if another program also handled
them), and all later tiles have not started.  This file collects the structural facts the loop
needs:

* a tile partly handled by `i` is split unless `i` owns it entirely (`splitTile_of_not_owns`);
* a split tile's output elements are never the output elements of a tile `i` owns
  (`splitOuts_not_owns`);
* a tile other than the current one that is available at the next iteration value was already
  available before (`gate_prev`).
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-- **A tile partly handled by a program that does not own it is split.** -/
theorem splitTile_of_not_owns (hd : d.ok) {i u : Nat} (hu : u ∈ tiles d)
    (hi : i < d.num_programs) (hns : ¬ owns d i u) (hc : c0 d i u < c1 d i u) :
    splitTile d u := by
  have hT := T_pos hd
  have hv : c0 d i u < steps d.K := lt_of_lt_of_le hc c1_le
  obtain ⟨r, ⟨hr1, hr2, hr3⟩, huniq⟩ := chunk_partition hd hu hv
  have hir : i = r := huniq i ⟨hi, le_rfl, hc⟩
  intro p hp ho
  obtain ⟨h0, h1⟩ := ho
  have hpr : p = r := huniq p ⟨hp, by rw [h0]; exact Nat.zero_le _, by rw [h1]; exact hv⟩
  exact hns (by rw [hir, ← hpr]; exact ⟨h0, h1⟩)

/-- A split tile's output elements are not the output elements of a tile `i` owns. -/
theorem splitOuts_not_owns (_hd : d.ok) {i u : Nat} (hu : u ∈ tiles d)
    (hi : i < d.num_programs) {q : Nat × Nat} (hq : q ∈ splitOuts d)
    (hqu : q ∈ tileOut d u) (ho : owns d i u) : False := by
  obtain ⟨v, hv, hsp, hqv⟩ := hq
  have hvu : v = u := by
    by_contra hne
    exact Set.disjoint_left.mp (tileOut_disjoint hv hu hne) hqv hqu
  subst hvu
  exact hsp i hi ho

/-- An owned tile has a nonempty chunk range. -/
theorem owns_c0_lt_c1 (hd : d.ok) {i u : Nat} (ho : owns d i u) : c0 d i u < c1 d i u := by
  have hT := T_pos hd
  obtain ⟨h0, h1⟩ := ho
  omega

/-- **Availability is inherited.**  If `u` is available at the next iteration value
`it' = min ((u+1)·S) hi` and `u` is not the tile currently being run, then it was already
available at `it`.  (The only tile that becomes available at `it'` is the current one.) -/
theorem gate_prev (hd : d.ok) {i it it' u w : Nat}
    (_hit : it < hi d i)
    (hule : u * steps d.K ≤ it) (_hult : it < (u + 1) * steps d.K)
    (hnex : it' = min ((u + 1) * steps d.K) (hi d i))
    (hne : c0 d i w < c1 d i w) (hw : w ≠ u) (hg : gate d i it' w) :
    gate d i it w := by
  have hT := T_pos hd
  unfold gate at hg ⊢
  rcases Nat.lt_or_gt_of_ne hw with hlt | hgt
  · -- the tile is before the current one
    have hw1 : (w + 1) * steps d.K ≤ u * steps d.K := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.succ_mul] at hw1
    exact le_trans (min_le_left _ _) (by omega)
  · -- the tile is after the current one: an owned chunk range there cannot be nonempty
    have hw1 : (u + 1) * steps d.K ≤ w * steps d.K := Nat.mul_le_mul_right _ (by omega)
    have hWS : hi d i ≤ w * steps d.K := by
      by_cases hcase : (u + 1) * steps d.K ≤ hi d i
      · rw [hnex, min_eq_left hcase] at hg
        rcases min_le_iff.mp hg with h | h
        · rw [Nat.succ_mul] at hw1 h
          omega
        · omega
      · omega
    have hc1 : c1 d i w = 0 := by
      unfold c1
      rw [Nat.sub_eq_zero_of_le hWS, max_eq_left (Nat.zero_le _), min_eq_right (by omega)]
    omega

end Claims.StreamKMatmul


/-!
# The store branch of one loop iteration

When program `i` owns the tile `u = tileOf d it` of the iteration `it`, the tail of the loop body
stores the accumulated tile to `c`.  The tile's lanes write exactly `Wv d i (mu d i stv)` (the
private write footprint of the tile), the accumulated value is the tile's dot product
(`accI_val`), and after the store the gated invariant holds at the next counter `it'`, which is
`min ((u+1) · steps K) (hi d i)`.

The statement is used with `stv`, `kf`, `kl`, `pm`, `pn` the locals of the loop body.
-/

namespace Claims.StreamKMatmul

open TVA

set_option maxHeartbeats 800000

variable {d : Inst} {P : Platform}

/-- An enabled `c` lane of the tile `u` sits at the output element its coordinates describe. -/
private theorem lane_info {d : Inst} {u : Nat} {pm pn : I32}
    {A : Idx [128, 128] → Option Int}
    (hA : A = tl.addrs
      (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
        (IntV.ofNat d.stride_cn))
      (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N))))
    (hd : d.ok) (hpm : pm.toInt = (pidMOf d u : Int))
    (hpn : pn.toInt = (pidNOf d u : Int)) (hu : u < tilesM d * tilesN d)
    {k : Idx [128, 128]} {b : Int} (hk : A k = some b) :
    b = cAddr d (pidMOf d u * 128 + k.1.val) (pidNOf d u * 128 + k.2.1.val) ∧
      (pidMOf d u * 128 + k.1.val, pidNOf d u * 128 + k.2.1.val) ∈ tileOut d u := by
  rw [hA] at hk
  have he : (fun i : Idx [128, 128] =>
      decide (pidMOf d u * 128 + i.1.val < d.M) &&
        decide (pidNOf d u * 128 + i.2.1.val < d.N)) k = true := by
    by_contra hne
    rw [(cAddrs (u := u) hd hpm hpn hu) k, ite_eq_right hne] at hk
    exact absurd hk (by simp)
  have h := cAddr_of_lane (d := d) (pm := pm) (pn := pn) (u := u) hd hpm hpn hu (i := k) he
  rw [hk, Option.some.injEq] at h
  have hii : (⟨k.1.val, k.1.isLt⟩ : Fin 128).val = k.1.val := rfl
  have hjj : (⟨k.2.1.val, k.2.1.isLt⟩ : Fin 128).val = k.2.1.val := rfl
  refine ⟨h, ?_⟩
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
  exact ⟨⟨hq1, hq2⟩, hd1, hd2⟩

/-- **The store branch.**  Program `i` owns tile `u`: it stores the accumulator, which holds the
full dot product of the tile, and the invariant advances to the next counter. -/
theorem tail_store (_hP : Device P) (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i it : Nat} (hpi : i < d.num_programs) (hlo : lo d i ≤ it) (hit : it < hi d i)
    {stv kf kl pm pn : I32} {acc : Tensor [128, 128] F32} {m1 : Mem} {log1 : List AtomicStep}
    (hstv : stv.toInt = (it : Int))
    (hkf : kf.toInt = (c0 d i (tileOf d it) : Int))
    (hkl : kl.toInt = (c1 d i (tileOf d it) : Int))
    (hpm : pm.toInt = (pidMOf d (tileOf d it) : Int))
    (hpn : pn.toInt = (pidNOf d (tileOf d it) : Int))
    (hQg : Qg d init i it log1 m1) (hpres : Pres d m1)
    (hacc : AccI d init i (tileOf d it) (c1 d i (tileOf d it) - c0 d i (tileOf d it)) acc)
    (ho : owns d i (tileOf d it))
    (hrc : it = lo d i ∨ it % steps d.K = 0) :
    SafeA ((launch d).env P) (inElems d) (Wv d i (mu d i stv)) (Ccells d)
      (fun st' log m => Inv d init i st' log m ∧ mu d i st' < mu d i stv) log1 m1
      (tl.store
          (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
            (IntV.ofNat d.stride_cn))
          acc
          (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))
        >>= fun _ => pure (stv + (kl - kf))) := by
  have hS : 0 < steps d.K := T_pos hd
  set u : Nat := tileOf d it with hu
  let A : Idx [128, 128] → Option Int :=
    tl.addrs
      (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
        (IntV.ofNat d.stride_cn))
      (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))
  have hA : A = tl.addrs
      (ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
        (IntV.ofNat d.stride_cn))
      (some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N))) := rfl
  have huT : u < T d := by rw [hu]; exact tileOf_lt_T hd hit
  have hut : u < tilesM d * tilesN d := by simpa only [T] using huT
  have hutiles : u ∈ tiles d := by rw [mem_tiles]; exact hut
  have hne : c0 d i u < c1 d i u := owns_c0_lt_c1 hd ho
  have hcle : c0 d i u ≤ c1 d i u := le_of_lt hne
  have haccS : AccI d init i u (steps d.K) acc := by
    rw [ho.1, ho.2] at hacc
    simpa using hacc
  have hkg : kFirstOf d it = c0 d i u := by
    simpa only [hu] using kFirstOf_eq_c0 (d := d) (i := i) (it := it) hd hlo hit hrc
  have hklg : kLastOf d it (hi d i) = c1 d i u := by
    simpa only [hu] using kLastOf_eq_c1 (d := d) (i := i) (it := it) hit
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
  have hmu : mu d i stv = T d - u := by
    simp only [mu, itN, hstv, Int.toNat_natCast, ite_eq_left hit, ← hu]
  have hacc_bytes : accBytes .f32 A = outBytes d (tileOut d u) := by
    rw [hA]
    exact cAccBytes (d := d) (u := u) hd hpm hpn hut
  have hWs : accBytes .f32 A ⊆ Wv d i (mu d i stv) := by
    rw [hacc_bytes, hmu, Wv_of_owns huT ho]
  have hok : StoreOk m1 .f32 A := by
    have h := storeOk_of_addrs (d := d) (dd := .f32) (b := d.c) (init := init) (m := m1)
      hpre.2.2.1 hpres (cBytes_sub (d := d))
      (s := [128, 128])
      (ptrs := ptrsC (d.c.ptr 0) (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.stride_cm)
        (IntV.ofNat d.stride_cn))
      (mask := some (maskC (offs1 pm 128) (offs1 pn 128) (IntV.ofNat d.M) (IntV.ofNat d.N)))
      (g := fun i => (pidMOf d u * 128 + i.1.val) * d.stride_cm +
        (pidNOf d u * 128 + i.2.1.val) * d.stride_cn)
      (en := fun i => decide (pidMOf d u * 128 + i.1.val < d.M) &&
        decide (pidNOf d u * 128 + i.2.1.val < d.N))
      (cAddrs (u := u) (d := d) hd hpm hpn hut)
      (fun i he => cLt_len hd hut i he)
    simpa only [A] using h
  -- the next counter
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
  -- the `Qg` components
  have hQgs := hQg
  obtain ⟨L, hLmap, hLacc, hLframe, hLval, hLrun, hLcnt⟩ := hQgs
  simp only [tl.store, Prog.bind_eq, Prog.bind, Prog.pure_eq]
  refine SafeA.store (d := .f32) (addrs := A) (vals := acc) (k := .ret (stv + (kl - kf))) hok hWs ?_
  intro order hperm
  let m' : Mem := writeAll m1 .f32 A acc order
  have hframe : ∀ x, x ∉ touchedBytes d i → m' x = init x := by
    intro x hx
    have hx' : x ∉ accBytes .f32 A := by
      rw [hacc_bytes]
      exact fun hc => hx (outBytes_tileOut_sub_touched (d := d) hd huT hne hc)
    rw [show m' = writeAll m1 .f32 A acc order from rfl,
      writeAll_frame (d := .f32) (x := x) m1 A acc order hx']
    exact hLframe x hx
  have hkeep : ∀ w ∈ tiles d, w ≠ u → ∀ q ∈ tileOut d w,
      m'.read .f32 (cAddr d q.1 q.2) = m1.read .f32 (cAddr d q.1 q.2) := by
    intro w hw hwu q hq
    have hqA : q ∈ allOuts d := tileOut_sub_allOuts w hq
    have hnot : ∀ x ∈ Mem.span .f32 (cAddr d q.1 q.2), x ∉ accBytes .f32 A := by
      intro x hx hxacc
      simp only [accBytes, Set.mem_ofPred_eq] at hxacc
      obtain ⟨i', a', hAi', hxa'⟩ := hxacc
      obtain ⟨haeq, hmem⟩ := lane_info hA hd hpm hpn hut hAi'
      have hq''A : (pidMOf d u * 128 + i'.1.val, pidNOf d u * 128 + i'.2.1.val) ∈ allOuts d :=
        tileOut_sub_allOuts u hmem
      have hne' : q ≠ (pidMOf d u * 128 + i'.1.val, pidNOf d u * 128 + i'.2.1.val) := by
        intro he
        have hdisj := Set.disjoint_left.mp
          (tileOut_disjoint (u := w) (v := u) hw hutiles hwu)
        exact hdisj hq (by simpa only [he] using hmem)
      have hsp := span_disjoint_of_ne (d := d) (q := q)
        (q' := (pidMOf d u * 128 + i'.1.val, pidNOf d u * 128 + i'.2.1.val)) hd hqA hq''A hne'
      exact Set.disjoint_left.mp hsp hx (by rw [← haeq]; exact hxa')
    rw [show m' = writeAll m1 .f32 A acc order from rfl]
    exact Mem.read_congr (fun x hx =>
      writeAll_frame (d := .f32) (x := x) m1 A acc order (hnot x hx))
  have hnew : ∀ q ∈ tileOut d u, ∃ v : F32, m'.read .f32 (cAddr d q.1 q.2) = some v ∧
      ∀ va vb : Nat → F16, ReadsA d init q.1 q.2 va → ReadsB d init q.1 q.2 vb →
        good v ∧ |v.toRat - dotSum va vb d.K| ≤ fineOne va vb d.K 0 (steps d.K) := by
    intro q hq
    have hq1 : q.1 < d.M := hq.1.1
    have hq2 : q.2 < d.N := hq.1.2
    have hqm : q.1 / 128 = pidMOf d u := hq.2.1
    have hqn : q.2 / 128 = pidNOf d u := hq.2.2
    have hq1' : q.1 % 128 < 128 := Nat.mod_lt _ (by norm_num)
    have hq2' : q.2 % 128 < 128 := Nat.mod_lt _ (by norm_num)
    have hm : pidMOf d u * 128 + q.1 % 128 = q.1 := by
      rw [Nat.mul_comm, ← hqm]
      exact Nat.div_add_mod q.1 128
    have hn : pidNOf d u * 128 + q.2 % 128 = q.2 := by
      rw [Nat.mul_comm, ← hqn]
      exact Nat.div_add_mod q.2 128
    have hii : (⟨q.1 % 128, hq1'⟩ : Fin 128).val = q.1 % 128 := rfl
    have hjj : (⟨q.2 % 128, hq2'⟩ : Fin 128).val = q.2 % 128 := rfl
    have hAq : A (⟨q.1 % 128, hq1'⟩, ⟨q.2 % 128, hq2'⟩, ()) = some (cAddr d q.1 q.2) := by
      rw [hA]
      have hen : (fun i : Idx [128, 128] =>
          decide (pidMOf d u * 128 + i.1.val < d.M) &&
            decide (pidNOf d u * 128 + i.2.1.val < d.N))
          (⟨q.1 % 128, hq1'⟩, ⟨q.2 % 128, hq2'⟩, ()) = true := by
        simp only [hm, hn, Bool.and_eq_true, decide_eq_true_eq]
        exact ⟨hq1, hq2⟩
      have h := cAddr_of_lane (d := d) (pm := pm) (pn := pn) (u := u) hd hpm hpn hut
        (i := (⟨q.1 % 128, hq1'⟩, ⟨q.2 % 128, hq2'⟩, ())) hen
      rw [h]
      simp only [hm, hn]
    have hmem : (⟨q.1 % 128, hq1'⟩, ⟨q.2 % 128, hq2'⟩, ()) ∈ order :=
      hperm.mem_iff.mpr (mem_enabled hAq)
    have hnodup : order.Nodup := hperm.nodup_iff.mpr (enabled_nodup A)
    have hread : m'.read .f32 (cAddr d q.1 q.2) =
        some (acc.get2 ⟨q.1 % 128, hq1'⟩ ⟨q.2 % 128, hq2'⟩) := by
      rw [show m' = writeAll m1 .f32 A acc order from rfl]
      exact writeAll_read (hdist := cDistinct (d := d) hd hpm hpn hut) hok order hnodup
        (⟨q.1 % 128, hq1'⟩, ⟨q.2 % 128, hq2'⟩, ()) (cAddr d q.1 q.2) hmem hAq
    refine ⟨acc.get2 ⟨q.1 % 128, hq1'⟩ ⟨q.2 % 128, hq2'⟩, hread, ?_⟩
    intro va vb hva hvb
    obtain ⟨v, hv, hg, hb⟩ := accI_val (d := d) (u := u) hd ho haccS
      (ii := ⟨q.1 % 128, hq1'⟩) (jj := ⟨q.2 % 128, hq2'⟩)
      (by rw [hii, hm]; exact hq1) (by rw [hjj, hn]; exact hq2)
      (by simpa only [hii, hjj, hm, hn] using hva)
      (by simpa only [hii, hjj, hm, hn] using hvb)
    rw [hv]
    exact ⟨hg, hb⟩
  have hgate : ∀ w ∈ tiles d, w ≠ u → c0 d i w < c1 d i w → gate d i (itN (stv + (kl - kf))) w →
      gate d i it w := by
    intro w hw hwu hlt hg
    exact gate_prev (d := d) (u := u) hd hit hle hult hnextU hlt hwu hg
  have hsplit : ∀ q ∈ splitOuts d, q ∈ tileOut d u → False := by
    intro q hq hqu
    exact splitOuts_not_owns (d := d) (u := u) hd hutiles hpi hq hqu ho
  have hQg' : Qg d init i (itN (stv + (kl - kf))) log1 m' :=
    Qg_store (u := u) (it := it) (it' := itN (stv + (kl - kf))) (m := m1) (m' := m')
      hQg hS hutiles hit hle hult hnextU hne ho hframe hnew hkeep hgate hsplit
  refine SafeA.ret ?_
  refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · rw [hadd]
    exact Int.natCast_nonneg _
  · rw [hnextT]
    exact lo_le_next (d := d) (i := i) (it := it) hd hlo hit
  · exact hQg'
  · intro x hx
    exact writeAll_present (d := .f32) (m := m1) (x := x) A acc order (hpres x hx)
  · rw [hnextT]
    intro hlt
    rcases reach_step (d := d) (i := i) (it := it) hit with h | h | h
    · exact Or.inl h
    · exact Or.inr h
    · omega
  · exact mu_step (d := d) (i := i) (it := it) (st := stv) (st' := stv + (kl - kf))
      (it' := itN (stv + (kl - kf))) hd hit hstIt rfl hnextT

end Claims.StreamKMatmul

