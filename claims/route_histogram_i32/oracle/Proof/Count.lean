import Proof.Lane

/-!
# `route_histogram_i32`: what one `tl.atomic_add` logs

`Qset` (`Defs.lean`) says what a program's log holds: only cell steps, and the increments at
each bucket sum to its own pairs. These lemmas show that the two `tl.atomic_add`s of
`add_counts` log exactly that: one step of `1` per enabled lane (the `BINS = 0` path), and one
step of `mine b` per bin `b < NUM_BUCKETS` (the `tl.histogram` path).
-/

namespace Claims.RouteHistogram

open TVA
open TVA.Block

/-- With `BINS ≠ 0`, the histogram's bins cover the buckets. -/
theorem bins_gt_buckets {d : Inst} (hd : d.ok) (h : d.bins ≠ 0) : d.numBuckets < d.bins := by
  rcases hd.bins_ok with h0 | ⟨-, -, h2⟩
  · exact absurd h0 h
  · exact h2

/-- A program has at most `BLOCK` lanes in a bucket. -/
theorem progCount_le_block (d : Inst) (p b : Nat) : progCount d p b ≤ d.block := by
  unfold progCount
  calc ((Finset.range d.block).filter
        (fun l => p * d.block + l < d.n ∧ bucketOf d (p * d.block + l) = b)).card
      ≤ (Finset.range d.block).card := Finset.card_filter_le _ _
    _ = d.block := Finset.card_range d.block

/-- `progCount` fits `int32`. -/
theorem progCount_lt (d : Inst) (hd : d.ok) (p b : Nat) : progCount d p b < 2 ^ 31 := by
  have h1 := progCount_le_block d p b
  have h2 := hd.block_le
  have h3 : (2 : Nat) ^ 20 < 2 ^ 31 := by norm_num
  omega

/-! ## Helpers -/

/-- The address of a masked access is present exactly at the mask's lanes. -/
theorem addrs_isSome {d : DType} {s : Shape} (ptrs : Tensor s (Ptr d))
    (m : Tensor s Bool) (i : Idx s) : (tl.addrs ptrs (some m) i).isSome = m.get i := by
  by_cases h : m.get i <;> simp [tl.addrs, h]

/-- The histogram mask at a lane: the lane is below `NUM_BUCKETS` and its bin count is positive. -/
theorem hist_mask_get {d : Inst} (hd : d.ok) (hbins : d.bins ≠ 0)
    {mine : Tensor [d.bins] I32} (i : Fin d.bins) :
    ((((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
        ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine)).get (i, ()) = true)
      ↔ ((i : Nat) < d.numBuckets ∧ 0 < (mine.get (i, ())).toInt) := by
  have hbins_le : d.bins ≤ 2 ^ 20 := by
    rcases hd.bins_ok with h0 | ⟨-, hle, -⟩
    · exact absurd h0 hbins
    · exact hle
  have hiv : -(2 ^ 31 : Int) ≤ (0 : Int) + (i : Nat) ∧ (0 : Int) + (i : Nat) < 2 ^ 31 := by
    have h31 : ((i : Nat) : Int) < 2 ^ 31 := by
      have hilt : (i : Nat) < 2 ^ 20 := lt_of_lt_of_le i.isLt hbins_le
      have h2 : (2 : Nat) ^ 20 ≤ 2 ^ 31 := by norm_num
      omega
    omega
  show (IntV.lt (IntV.ofInt ((0 : Int) + (i : Nat))) (IntV.ofNat d.numBuckets : I32) &&
        IntV.lt (IntV.ofInt (0 : Int)) (mine.get (i, ()))) = true ↔
      ((i : Nat) < d.numBuckets ∧ 0 < (mine.get (i, ())).toInt)
  rw [IntV.lt_eq, IntV.lt_eq, IntV.I32.toInt_ofInt hiv, IntV.I32.toInt_ofNat hd.buckets_lt,
    IntV.I32.toInt_ofInt (by norm_num : -(2 ^ 31 : Int) ≤ (0 : Int) ∧ (0 : Int) < 2 ^ 31),
    Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨by omega, h2⟩
  · rintro ⟨h1, h2⟩
    exact ⟨by omega, h2⟩

/-- The histogram mask at an arbitrary index of the block shape. -/
theorem hist_mask_get_idx {d : Inst} (hd : d.ok) (hbins : d.bins ≠ 0)
    {mine : Tensor [d.bins] I32} (i : Idx [d.bins]) :
    ((((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
        ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine)).get i = true)
      ↔ ((i.1 : Nat) < d.numBuckets ∧ 0 < (mine.get1 i.1).toInt) := by
  obtain ⟨iv, u⟩ := i
  cases u
  exact hist_mask_get hd hbins iv

/-- A lane the mask keeps takes its bin from `x`. -/
theorem where_get {d : Inst} {p : Nat} {buckets : Tensor [d.block] I32}
    (j : Fin d.block) (hm : (mask p d.block d.n).get (j, ()) = true) :
    (tl.where (mask p d.block d.n) buckets
        (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get (j, ())
      = buckets.get (j, ()) := by
  show (if (mask p d.block d.n).get (j, ()) = true then buckets.get (j, ())
      else (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32)).get (j, ())) = buckets.get (j, ())
  rw [hm, ite_eq_left rfl]

/-- `int32` `1` is the literal `1`. -/
theorem ofInt_one_eq {s : Bool} {w : Nat} : (IntV.ofInt 1 : IntV s w) = 1 := by
  apply IntV.ext
  rfl

/-- `card_enabled_bucket` with the two conditions of the filter in the other order (the order
`List.filter_filter` produces). -/
theorem card_enabled_bucket' {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d)
    (b : Nat) :
    (((Idx.all [d.block]).filter (fun i =>
        decide (bucketOf d (p * d.block + (i.1 : Nat)) = b) &&
          (mask p d.block d.n).get i)).length)
      = progCount d p b := by
  have hfit : Fits p d.block := fits hd hp
  have hn : d.n < 2 ^ 31 := by have := n_max_lt hd; omega
  set c : Nat → Bool := fun l =>
    decide (p * d.block + l < d.n ∧ bucketOf d (p * d.block + l) = b) with hcdef
  have hcongr : (Idx.all [d.block]).filter (fun i =>
        decide (bucketOf d (p * d.block + (i.1 : Nat)) = b) &&
          (mask p d.block d.n).get i)
      = (Idx.all [d.block]).filter (fun i => c (i.1 : Nat)) := by
    refine List.filter_congr ?_
    intro i _
    obtain ⟨l, u⟩ := i
    cases u
    simp only [hcdef, mask_get hfit hn, Bool.decide_and]
    exact Bool.and_comm _ _
  rw [hcongr, length_filter_idx d.block c]
  unfold progCount
  congr 1

/-! ## The atomics -/

/-- The `BINS = 0` atomic: the log adds the program's own pairs to each bucket. -/
theorem Qset_steps_inc {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) {m : Mem}
    {order : List (Idx [d.block])} {ptrs : Tensor [d.block] (Ptr DType.i32)}
    {olds news : Idx [d.block] → I32}
    (hperm : order.Perm (enabled (tl.addrs ptrs (some (mask p d.block d.n)))))
    (haddr : ∀ i, (tl.addrs ptrs (some (mask p d.block d.n)) i).isSome = true →
      (ptrs.get i).addr
        = (d.counts.ptr ((bucketOf d (p * d.block + (i.1 : Nat)) : Nat) : Int)).addr)
    (hupd : ∀ i, news i = olds i + (IntV.ofInt 1 : I32)) :
    Qset d p (order.map (fun i => ⟨DType.i32, (ptrs.get i).addr, olds i, news i⟩)) m := by
  have hmem : ∀ i ∈ order, (tl.addrs ptrs (some (mask p d.block d.n)) i).isSome = true := by
    intro i hi
    have hien : i ∈ enabled (tl.addrs ptrs (some (mask p d.block d.n))) :=
      (List.Perm.mem_iff hperm).mp hi
    simp only [enabled, List.mem_filter] at hien
    exact hien.2
  constructor
  · intro s hs
    rw [List.mem_map] at hs
    obtain ⟨i, hi, rfl⟩ := hs
    exact ⟨(bucketOf d (p * d.block + (i.1 : Nat)) : Nat), bucketOf_lt hd _, by
      rw [haddr i (hmem i hi)]⟩
  · intro b hb
    have hstep : ∀ i ∈ order,
        stepInc (d.counts.ptr (b : Int)).addr
          ⟨DType.i32, (ptrs.get i).addr, olds i, news i⟩
        = (if decide (bucketOf d (p * d.block + (i.1 : Nat)) = b) then (1 : I32) else 0) := by
      intro i hi
      rw [haddr i (hmem i hi)]
      by_cases hkb : bucketOf d (p * d.block + (i.1 : Nat)) = b
      · have hcast : (((bucketOf d (p * d.block + (i.1 : Nat)) : Nat)) : Int) = (b : Int) := by
          rw [hkb]
        rw [hcast, stepInc_self, hupd i, add_sub_cancel, ite_eq_left (by simp [hkb])]
        exact ofInt_one_eq
      · have hne : (((bucketOf d (p * d.block + (i.1 : Nat)) : Nat)) : Int) ≠ (b : Int) := by
          intro h
          exact hkb (by exact_mod_cast h)
        rw [stepInc_counts_ne hne (olds i) (news i), ite_eq_right (by simp [hkb])]
    have hmap_eq : ((order.map (fun i =>
          ⟨DType.i32, (ptrs.get i).addr, olds i, news i⟩)).map
        (stepInc (d.counts.ptr (b : Int)).addr))
        = order.map (fun i =>
            if decide (bucketOf d (p * d.block + (i.1 : Nat)) = b) then (1 : I32) else 0) := by
      rw [List.map_map]
      exact List.map_congr_left hstep
    have hcnt : order.countP
        (fun i => decide (bucketOf d (p * d.block + (i.1 : Nat)) = b)) = progCount d p b := by
      rw [List.Perm.countP_eq _ hperm, List.countP_eq_length_filter]
      have hen : enabled (tl.addrs ptrs (some (mask p d.block d.n)))
          = (Idx.all [d.block]).filter (fun i => (mask p d.block d.n).get i) := by
        simp only [enabled]
        refine List.filter_congr ?_
        intro i _
        exact addrs_isSome ptrs (mask p d.block d.n) i
      rw [hen, List.filter_filter]
      exact card_enabled_bucket' hd hp b
    rw [hmap_eq, sum_map_ite_one, hcnt]

/-- A masked-off lane of the histogram goes to bin `BINS - 1`, so the lanes counted in bin
`b < NUM_BUCKETS` are exactly the enabled lanes whose bucket is `b`. -/
theorem hist_card {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) (_hbins : d.bins ≠ 0)
    {buckets : Tensor [d.block] I32}
    (hbuckets : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (buckets.get (l, ())).toInt = (bucketOf d (p * d.block + (l : Nat)) : Int))
    (b : Nat) (_hb : b < d.numBuckets) :
    (Finset.univ.filter fun j : Fin d.block =>
      tl.enabledBy (some (mask p d.block d.n)) j = true ∧
        ((tl.where (mask p d.block d.n) buckets
          (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get1 j).toInt
          = ((b : Nat) : Int)).card
      = progCount d p b := by
  have hfit : Fits p d.block := fits hd hp
  have hn : d.n < 2 ^ 31 := by have := n_max_lt hd; omega
  unfold progCount
  refine Finset.card_nbij (fun j : Fin d.block => (j : Nat)) ?_ ?_ ?_
  · intro j hj
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_range] at hj ⊢
    have hm : (mask p d.block d.n).get (j, ()) = true := by
      have h1 : (mask p d.block d.n).get1 j = true := by simpa [tl.enabledBy] using hj.1
      exact h1
    have hlt : p * d.block + (j : Nat) < d.n := by
      have h := mask_get hfit hn j
      rw [h] at hm
      exact of_decide_eq_true hm
    have hval : ((tl.where (mask p d.block d.n) buckets
        (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get (j, ())).toInt
        = ((b : Nat) : Int) := hj.2
    rw [where_get j hm] at hval
    rw [hbuckets j hlt] at hval
    refine ⟨j.isLt, hlt, ?_⟩
    exact_mod_cast hval
  · intro j₁ _ j₂ _ h
    exact Fin.val_injective h
  · intro l hl
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range] at hl
    obtain ⟨hlt, hln, hlb⟩ := hl
    refine ⟨⟨l, hlt⟩, ?_, rfl⟩
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and]
    have hm : (mask p d.block d.n).get ((⟨l, hlt⟩ : Fin d.block), ()) = true := by
      rw [mask_get hfit hn ⟨l, hlt⟩, decide_eq_true_eq]
      exact hln
    refine ⟨?_, ?_⟩
    · show (mask p d.block d.n).get1 ⟨l, hlt⟩ = true
      exact hm
    · show ((tl.where (mask p d.block d.n) buckets
        (tl.full [d.block] (IntV.ofNat (d.bins - 1) : I32))).get ((⟨l, hlt⟩ : Fin d.block), ())).toInt
        = ((b : Nat) : Int)
      rw [where_get ⟨l, hlt⟩ hm, hbuckets ⟨l, hlt⟩ hln, hlb]

/-- The `tl.histogram` atomic: the log adds the program's own pairs to each bucket. -/
theorem Qset_steps_hist {d : Inst} (hd : d.ok) (hbins : d.bins ≠ 0) {p : Nat} {m : Mem}
    {order : List (Idx [d.bins])} {ptrs : Tensor [d.bins] (Ptr DType.i32)}
    {olds : Idx [d.bins] → I32} {mine : Tensor [d.bins] I32}
    (hnd : order.Nodup)
    (hperm : order.Perm (enabled (tl.addrs ptrs
      (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
        ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine))))))
    (haddr : ∀ i, (tl.addrs ptrs
      (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
        ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine))) i).isSome = true →
      (ptrs.get i).addr = (d.counts.ptr (i.1 : Nat)).addr)
    (hcount : ∀ (b : Fin d.bins), (b : Nat) < d.numBuckets →
      mine.get1 b = IntV.ofNat (progCount d p (b : Nat))) :
    Qset d p (order.map (fun i =>
      ⟨DType.i32, (ptrs.get i).addr, olds i, olds i + mine.get1 i.1⟩)) m := by
  have hmem : ∀ i ∈ order,
      (tl.addrs ptrs
        (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
          ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine))) i).isSome = true := by
    intro i hi
    have hien : i ∈ enabled (tl.addrs ptrs
        (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
          ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine)))) :=
      (List.Perm.mem_iff hperm).mp hi
    simp only [enabled, List.mem_filter] at hien
    exact hien.2
  have hmask : ∀ i, (tl.addrs ptrs
      (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
        ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine))) i).isSome
      = (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
        ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine)).get i :=
    fun i => addrs_isSome ptrs _ i
  constructor
  · intro s hs
    rw [List.mem_map] at hs
    obtain ⟨i, hi, rfl⟩ := hs
    have his := hmem i hi
    have hmi : ((((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
        ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine)).get i = true) := by
      rw [← hmask i]
      exact his
    exact ⟨(i.1 : Nat), (hist_mask_get_idx hd hbins i).mp hmi |>.1, by
      rw [haddr i his]⟩
  · intro b hb
    have hblt : b < d.bins := by
      have hnb := bins_gt_buckets hd hbins
      omega
    let b' : Fin d.bins := ⟨b, hblt⟩
    let bI : Idx [d.bins] := (b', ())
    have hstep : ∀ i ∈ order,
        stepInc (d.counts.ptr (b : Int)).addr
          ⟨DType.i32, (ptrs.get i).addr, olds i, olds i + mine.get1 i.1⟩
        = (if i = bI then mine.get1 i.1 else 0) := by
      intro i hi
      have his := hmem i hi
      rw [haddr i his]
      by_cases hik : (i.1 : Nat) = b
      · have hibi : i = bI := by
          have hfin : i.1 = b' := Fin.ext hik
          obtain ⟨iv, u⟩ := i
          cases u
          exact Prod.ext hfin rfl
        have hcast : (((i.1 : Nat)) : Int) = (b : Int) := by rw [hik]
        rw [hcast, ite_eq_left hibi, stepInc_self, add_sub_cancel]
      · have hne : (((i.1 : Nat)) : Int) ≠ (b : Int) := by
          intro h
          exact hik (by exact_mod_cast h)
        have hneI : i ≠ bI := by
          intro h
          apply hik
          rw [h]
        rw [stepInc_counts_ne hne (olds i) (olds i + mine.get1 i.1), ite_eq_right hneI]
    have hmap_eq : ((order.map (fun i =>
          ⟨DType.i32, (ptrs.get i).addr, olds i, olds i + mine.get1 i.1⟩)).map
        (stepInc (d.counts.ptr (b : Int)).addr))
        = order.map (fun i => if i = bI then mine.get1 i.1 else 0) := by
      rw [List.map_map]
      exact List.map_congr_left hstep
    rw [hmap_eq, sum_map_single order (fun i => if i = bI then mine.get1 i.1 else 0) hnd bI]
    · by_cases hin : bI ∈ order
      · rw [ite_eq_left hin, ite_eq_left rfl, hcount b' hb]
      · rw [ite_eq_right hin]
        have hzero : progCount d p b = 0 := by
          by_contra hne
          have hpos : 0 < progCount d p b := Nat.pos_of_ne_zero hne
          apply hin
          rw [List.Perm.mem_iff hperm]
          show bI ∈ (Idx.all [d.bins]).filter (fun i => (tl.addrs ptrs
            (some (((tl.arange 0 d.bins).ltS (IntV.ofNat d.numBuckets : I32)).and
              ((tl.full [d.bins] (IntV.ofInt 0 : I32)).lt mine))) i).isSome)
          rw [List.mem_filter]
          refine ⟨Idx.mem_all _ _, ?_⟩
          rw [hmask bI, hist_mask_get_idx hd hbins bI]
          refine ⟨hb, ?_⟩
          rw [show mine.get1 bI.1 = mine.get1 b' from rfl, hcount b' hb,
            IntV.I32.toInt_ofNat (progCount_lt d hd p b)]
          exact_mod_cast hpos
        rw [hzero, ofNat_zero]
    · intro i hi hne
      rw [ite_eq_right hne]

end Claims.RouteHistogram
