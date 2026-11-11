import Proof.Count

/-!
# `route_histogram_i32`: `_add_counts` is `SafeA`

`add_counts` only shares the cells of `counts` through its two `tl.atomic_add`s, and each of
them logs exactly the program's own pairs (`Qset_steps_inc`, `Qset_steps_hist`). So the whole
procedure is `SafeA` with an empty private write footprint.
-/

namespace Claims.RouteHistogram

open TVA
open TVA.Block

/-- A lane the mask cuts takes bin `BINS - 1`. -/
theorem where_get_false {d : Inst} {p : Nat} {buckets : Tensor [d.block] I32}
    (j : Fin d.block) (hm : (mask p d.block d.n).get (j, ()) = false) :
    (tl.where (mask p d.block d.n) buckets
        (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get (j, ())
      = (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32)).get (j, ()) := by
  show (if (mask p d.block d.n).get (j, ()) = true then buckets.get (j, ())
      else (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32)).get (j, ())) = _
  rw [hm]
  rfl

/-- `hist_card` with the histogram's own mask (`none`, every lane counts). -/
theorem hist_card_none {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) (hbins : d.bins ≠ 0)
    {buckets : Tensor [d.block] I32}
    (hbuckets : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (buckets.get (l, ())).toInt = (bucketOf d (p * d.block + (l : Nat)) : Int))
    (b : Nat) (hb : b < d.numBuckets) :
    (Finset.univ.filter fun j : Fin d.block =>
      tl.enabledBy none j = true ∧
        ((tl.where (mask p d.block d.n) buckets
          (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt
          = ((b : Nat) : Int)).card
      = progCount d p b := by
  have hfilter : (Finset.univ.filter fun j : Fin d.block =>
        tl.enabledBy none j = true ∧
          ((tl.where (mask p d.block d.n) buckets
            (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt
            = ((b : Nat) : Int))
      = (Finset.univ.filter fun j : Fin d.block =>
        tl.enabledBy (some (mask p d.block d.n)) j = true ∧
          ((tl.where (mask p d.block d.n) buckets
            (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt
            = ((b : Nat) : Int)) := by
    refine Finset.filter_congr ?_
    intro j _
    have htr : tl.enabledBy none j = true := rfl
    simp only [true_and, tl.enabledBy]
    constructor
    · intro hval
      by_cases hm : (mask p d.block d.n).get (j, ()) = true
      · show (mask p d.block d.n).get1 j = true ∧ _
        exact ⟨hm, hval⟩
      · exfalso
        have hmf : (mask p d.block d.n).get (j, ()) = false := Bool.eq_false_iff.mpr hm
        have hval0 : ((tl.where (mask p d.block d.n) buckets
            (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt
            = ((d.bins - 1 : Nat) : Int) := by
          show ((tl.where (mask p d.block d.n) buckets
            (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get (j, ())).toInt = _
          rw [where_get_false j hmf]
          exact IntV.I32.toInt_ofNat (by
            have hle : d.bins ≤ 2 ^ 20 := by
              rcases hd.bins_ok with h0 | ⟨-, hle, -⟩
              · exact absurd h0 hbins
              · exact hle
            have hpos : 0 < d.bins := Nat.pos_of_ne_zero hbins
            omega)
        have hblt : b < d.bins - 1 := by
          have hgt : d.numBuckets < d.bins := bins_gt_buckets hd hbins
          omega
        have hcast : ((b : Nat) : Int) < ((d.bins - 1 : Nat) : Int) := by
          exact_mod_cast hblt
        rw [hval0] at hval
        exact absurd hval (ne_of_gt hcast)
    · rintro ⟨-, hval⟩
      exact hval
  rw [hfilter]
  exact hist_card hd hp hbins hbuckets b hb

/-- **`add_counts`** (both paths, `BINS = 0` and `tl.histogram`) is `SafeA` and ends with `Qset`, given the lane values of the bucket tensor. -/
theorem safe_add_counts {d : Inst} (hd : d.ok) {E : Env} {m : Mem} {p : Nat}
    (hp : p < grid d) (hvc : d.counts.Valid m) {bid : Tensor [d.block] I32}
    (hval : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (bid.get (l, ())).toInt = (bucketOf d (p * d.block + (l : Nat)) : Int)) :
    SafeA E (Rpriv d) ∅ (cells d)
      (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Qset d p log m) [] m
      (add_counts (d.counts.ptr 0) bid (mask p d.block d.n) d.numBuckets d.bins) := by
  have hfit : Fits p d.block := fits hd hp
  have hn : d.n < 2 ^ 31 := by have := n_max_lt hd; omega
  have hnb : d.numBuckets ≤ d.counts.len := hd.counts_len
  simp only [add_counts]
  set bw : Tensor [d.block] I32 :=
    tl.where (bid.ltS (IntV.ofInt 0 : I32))
      (tl.full [d.block] (IntV.ofNat (d.numBuckets - 1) : I32)) bid with hbwdef
  have hbwv : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (bw.get (l, ())).toInt = (bucketOf d (p * d.block + (l : Nat)) : Int) := by
    intro l hl
    rw [hbwdef]
    exact laneOf_raw hd (fun l hl => Or.inl (hval l hl)) l hl
  have hmaskL : ∀ l : Fin d.block, (mask p d.block d.n).get (l, ()) = true →
      p * d.block + (l : Nat) < d.n := by
    intro l h
    have hm := mask_get hfit hn l
    rw [hm] at h
    exact of_decide_eq_true h
  have hptrs0 : ∀ i : Idx [d.block],
      (tl.addrs (d.counts.ptr 0 + bw) (some (mask p d.block d.n)) i).isSome = true →
      ((d.counts.ptr 0 + bw).get i).addr
        = (d.counts.ptr ((bucketOf d (p * d.block + (i.1 : Nat)) : Nat) : Int)).addr := by
    intro i hi
    obtain ⟨l, u⟩ := i
    have hu : u = () := by cases u; rfl
    rw [hu] at hi ⊢
    have hm : (mask p d.block d.n).get (l, ()) = true := by
      rw [addrs_isSome (d.counts.ptr 0 + bw) (mask p d.block d.n) (l, ())] at hi
      exact hi
    have hl := hmaskL l hm
    show ((d.counts.ptr 0).offset (bw.get (l, ())).toInt).addr = _
    rw [hbwv l hl]
    simp [Buf.ptr]
  by_cases hbins : d.bins = 0
  · rw [ite_eq_left hbins]
    simp only [tl.atomic_add, Prog.bind_eq, Prog.bind]
    refine SafeA.env ?_
    have hcell0 : ∀ i, (tl.addrs (d.counts.ptr 0 + bw) (some (mask p d.block d.n)) i).isSome = true →
        Mem.Aligned DType.i32 ((d.counts.ptr 0 + bw).get i).addr ∧
          (DType.i32, ((d.counts.ptr 0 + bw).get i).addr) ∈ cells d := by
      intro i hi
      have hlt := bucketOf_lt hd (p * d.block + (i.1 : Nat))
      refine ⟨?_, ?_⟩
      · rw [hptrs0 i hi]
        exact Buf.aligned d.counts hvc (by omega)
      · rw [hptrs0 i hi]
        exact ⟨bucketOf d (p * d.block + (i.1 : Nat)), hlt, rfl⟩
    have htot0 : ∀ i old, ∃ new,
        tl.addUpd E DType.i32 ((tl.full [d.block] (IntV.ofInt 1 : I32)).get i) old new :=
      fun i old => ⟨old + IntV.ofInt 1, by simp only [tl.addUpd, tl.full, Tensor.get_full]⟩
    have hk0 : ∀ order : List (Idx [d.block]),
        order.Perm ((Idx.all [d.block]).filter
          fun i => (tl.addrs (d.counts.ptr 0 + bw) (some (mask p d.block d.n)) i).isSome) →
        ∀ olds news : Idx [d.block] → I32,
          (∀ i ∈ order, tl.addUpd E DType.i32
            ((tl.full [d.block] (IntV.ofInt 1 : I32)).get i) (olds i) (news i)) →
          SafeA E (Rpriv d) ∅ (cells d) (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Qset d p log m)
            ([] ++ order.map fun i => ⟨DType.i32, ((d.counts.ptr 0 + bw).get i).addr, olds i, news i⟩) m
            (Prog.ret ()) := by
      intro order horder olds news hup
      refine SafeA.ret ?_
      simp only [List.nil_append]
      have hmap : (order.map fun i =>
            (⟨DType.i32, ((d.counts.ptr 0 + bw).get i).addr, olds i, news i⟩ : AtomicStep))
          = (order.map fun i =>
            (⟨DType.i32, ((d.counts.ptr 0 + bw).get i).addr, olds i,
              olds i + (IntV.ofInt 1 : I32)⟩ : AtomicStep)) := by
        refine List.map_congr_left ?_
        intro i hi
        have h := hup i hi
        simp only [tl.addUpd, tl.full, Tensor.get_full] at h
        rw [h]
      rw [hmap]
      exact Qset_steps_inc hd hp horder hptrs0 (fun i => rfl)
    exact SafeA.atomic (C := cells d) (upd := fun i => tl.addUpd E DType.i32
        ((tl.full [d.block] (IntV.ofInt 1 : I32)).get i))
      (k := Prog.ret ()) hcell0 rfl (by decide) htot0 hk0

  · rw [ite_eq_right hbins]
    have hbinsle : d.bins ≤ 2 ^ 20 := by
      rcases hd.bins_ok with h0 | ⟨-, hle, -⟩
      · exact absurd h0 hbins
      · exact hle
    have hxval : ∀ j : Fin d.block, tl.enabledBy none j = true →
        0 ≤ ((tl.where (mask p d.block d.n) bw (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt ∧ ((tl.where (mask p d.block d.n) bw (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt < d.bins := by
      intro j _
      by_cases hm : (mask p d.block d.n).get (j, ()) = true
      · have hl := hmaskL j hm
        have hget : (tl.where (mask p d.block d.n) bw (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j = bw.get1 j := where_get j hm
        have hb : (bw.get1 j).toInt = (bucketOf d (p * d.block + (j : Nat)) : Int) := hbwv j hl
        constructor
        · rw [hget, hb]
          exact bucketOf_nonneg d _
        · rw [hget, hb]
          have h3 : ((bucketOf d (p * d.block + (j : Nat)) : Nat) : Int) < (d.bins : Int) := by
            have h1 := bucketOf_lt hd (p * d.block + (j : Nat))
            have h2 := bins_gt_buckets hd hbins
            exact_mod_cast (show bucketOf d (p * d.block + (j : Nat)) < d.bins by omega)
          exact h3
      · have hmf : (mask p d.block d.n).get (j, ()) = false := by
          cases h : (mask p d.block d.n).get (j, ())
          · rfl
          · exact absurd h hm
        have hget : (tl.where (mask p d.block d.n) bw (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j
            = (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32)).get1 j := where_get_false j hmf
        have hval0 : ((tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32)).get1 j).toInt
            = ((d.bins - 1 : Nat) : Int) := by
          have hpos : 0 < d.bins := Nat.pos_of_ne_zero hbins
          have h31 : d.bins - 1 < 2 ^ 31 := by omega
          show (IntV.ofNat (d.bins - 1) : I32).toInt = ((d.bins - 1 : Nat) : Int)
          exact IntV.I32.toInt_ofNat h31
        constructor
        · rw [hget, hval0]
          exact_mod_cast Nat.zero_le (d.bins - 1)
        · rw [hget, hval0]
          have hpos : 0 < d.bins := Nat.pos_of_ne_zero hbins
          exact_mod_cast (show d.bins - 1 < d.bins by omega)
    let w : Tensor [d.bins] I32 := Tensor.of1 fun b : Fin d.bins =>
      IntV.ofNat (Finset.univ.filter (fun j : Fin d.block =>
        tl.enabledBy none j = true ∧
          ((tl.where (mask p d.block d.n) bw (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt = (b : Int))).card
    have hwv : ∀ b : Fin d.bins, w.get1 b = IntV.ofNat (Finset.univ.filter (fun j : Fin d.block =>
        tl.enabledBy none j = true ∧ ((tl.where (mask p d.block d.n) bw (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt = (b : Int))).card :=
      fun b => rfl
    simp only [tl.histogram, tl.atomic_add, Prog.bind_eq, Prog.bind]
    refine SafeA.pick ⟨w, hxval, hwv⟩ ?_
    intro r hr
    refine SafeA.env ?_
    have hmem : ∀ i, (tl.addrs (d.counts.ptr 0 + tl.arange 0 d.bins) (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt r))) i).isSome = true →
        (i.1 : Nat) < d.numBuckets := by
      intro i hi
      have hmask : (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt r)).get i = true := by
        rw [← addrs_isSome (d.counts.ptr 0 + tl.arange 0 d.bins) (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt r)) i]
        exact hi
      exact ((hist_mask_get_idx hd hbins (mine := r) i).mp hmask).1
    have haddrH : ∀ i, (tl.addrs (d.counts.ptr 0 + tl.arange 0 d.bins) (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt r))) i).isSome = true →
        ((d.counts.ptr 0 + tl.arange 0 d.bins).get i).addr = (d.counts.ptr (i.1 : Nat)).addr := by
      intro i hi
      obtain ⟨l, u⟩ := i
      have hu : u = () := by cases u; rfl
      rw [hu] at hi ⊢
      have hmask : (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt r)).get (l, ()) = true := by
        rw [← addrs_isSome (d.counts.ptr 0 + tl.arange 0 d.bins) (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt r)) (l, ())]
        exact hi
      have hone : ((tl.arange 0 d.bins).get (l, ())).toInt = (l : Int) := by
        have h31 : ((l : Nat) : Int) < 2 ^ 31 := by
          have hlt : (l : Nat) < 2 ^ 31 := by
            have hl1 := l.isLt
            omega
          exact_mod_cast hlt
        show (IntV.ofInt ((0 : Int) + ((l : Nat) : Int)) : I32).toInt = ((l : Nat) : Int)
        rw [IntV.I32.toInt_ofInt (show -(2 ^ 31 : Int) ≤ (0 : Int) + (l : Nat) ∧
            (0 : Int) + (l : Nat) < 2 ^ 31 by omega)]
        ring
      show ((d.counts.ptr 0).offset ((tl.arange 0 d.bins).get (l, ())).toInt).addr = _
      rw [hone]
      simp [Buf.ptr]
    have hcellH : ∀ i, (tl.addrs (d.counts.ptr 0 + tl.arange 0 d.bins) (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt r))) i).isSome = true →
        Mem.Aligned DType.i32 ((d.counts.ptr 0 + tl.arange 0 d.bins).get i).addr ∧
          (DType.i32, ((d.counts.ptr 0 + tl.arange 0 d.bins).get i).addr) ∈ cells d := by
      intro i hi
      have hlt := hmem i hi
      refine ⟨?_, ?_⟩
      · rw [haddrH i hi]
        exact Buf.aligned d.counts hvc (by omega)
      · rw [haddrH i hi]
        exact ⟨(i.1 : Nat), hlt, rfl⟩
    have htotH : ∀ i old, ∃ new, tl.addUpd E DType.i32 (r.get i) old new :=
      fun i old => ⟨old + r.get i, rfl⟩
    have hcount : ∀ (b : Fin d.bins), (b : Nat) < d.numBuckets →
        r.get1 b = IntV.ofNat (progCount d p (b : Nat)) := by
      intro b hb
      rw [hr.2 b, hist_card_none hd hp hbins hbwv (b : Nat) hb]
    have hkH : ∀ order : List (Idx [d.bins]),
        order.Perm ((Idx.all [d.bins]).filter
          fun i => (tl.addrs (d.counts.ptr 0 + tl.arange 0 d.bins) (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt r))) i).isSome) →
        ∀ olds news : Idx [d.bins] → I32,
          (∀ i ∈ order, tl.addUpd E DType.i32 (r.get i) (olds i) (news i)) →
          SafeA E (Rpriv d) ∅ (cells d) (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Qset d p log m)
            ([] ++ order.map fun i => (⟨DType.i32,
              ((d.counts.ptr 0 + tl.arange 0 d.bins).get i).addr, olds i, news i⟩ : AtomicStep)) m
            (Prog.ret ()) := by
      intro order horder olds news hup
      refine SafeA.ret ?_
      simp only [List.nil_append]
      have hmap : (order.map fun i => (⟨DType.i32,
            ((d.counts.ptr 0 + tl.arange 0 d.bins).get i).addr, olds i, news i⟩ : AtomicStep))
          = (order.map fun i => (⟨DType.i32,
            ((d.counts.ptr 0 + tl.arange 0 d.bins).get i).addr, olds i,
            olds i + r.get1 i.1⟩ : AtomicStep)) := by
        refine List.map_congr_left ?_
        intro i hi
        have h := hup i hi
        simp only [tl.addUpd] at h
        rw [h]
        congr 1
      rw [hmap]
      exact Qset_steps_hist hd hbins
        (horder.nodup_iff.mpr ((Idx.nodup_all [d.bins]).filter _)) horder haddrH hcount
    exact SafeA.atomic (C := cells d) (upd := fun i => tl.addUpd E DType.i32 (r.get i))
      (k := Prog.ret ()) hcellH rfl (by decide) htotH hkH

end Claims.RouteHistogram
