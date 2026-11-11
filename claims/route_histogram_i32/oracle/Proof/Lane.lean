import Proof.Arith

/-!
# `route_histogram_i32`: the lanes

The value the model computes for a lane: `route_bucket_ids` gives the bucket, or `int32` `-1`
when the lane's pair is not live, and `add_counts` turns `-1` into `NUM_BUCKETS - 1`
(`bucketOf`'s value for a pair that is not live). This file states what `bucketOf` is on live and
dead pairs, the range and no-wrap facts that make the `int32` arithmetic exact, and the
`int32`-sum lemmas the atomics need.
-/

namespace Claims.RouteHistogram

open TVA
open TVA.Block

/-! ## `bucketOf`: the bucket of a pair -/

/-- The "live" condition of the kernel: the slot is in `[0, MAX_LORAS)` and, with groups, the
group is in `[0, GROUPS_PER_SLOT)`. -/
def liveCond (d : Inst) (sl g : Int) : Prop :=
  0 ≤ sl ∧ sl < d.maxLoras ∧ (d.hasGroups = true → 0 ≤ g ∧ (d.gps = 1 ∨ g < d.gps))

/-- There is at least one bucket. -/
theorem numBuckets_pos {d : Inst} (hd : d.ok) : 0 < d.numBuckets := by
  have h1 := hd.buckets_eq
  have h2 := hd.gps_pos
  omega

/-- `MAX_LORAS * GROUPS_PER_SLOT` is below `NUM_BUCKETS`. -/
theorem maxLoras_mul_gps_lt {d : Inst} (hd : d.ok) : d.maxLoras * d.gps < d.numBuckets := by
  have h1 := hd.buckets_eq
  have h2 := hd.gps_pos
  have h3 : d.maxLoras * d.gps = d.gps * d.maxLoras := Nat.mul_comm _ _
  omega

/-- A live slot is below `NUM_BUCKETS`. -/
theorem live_slot_lt {d : Inst} (hd : d.ok) {sl g : Int} (h : liveCond d sl g) :
    sl.toNat < d.numBuckets := by
  have h0 : 0 ≤ sl := h.1
  have h1 : sl < (d.maxLoras : Int) := h.2.1
  have h2 : (sl.toNat : Int) = sl := Int.toNat_of_nonneg h0
  have h3 : sl.toNat < d.maxLoras := by omega
  have h4 : d.maxLoras < d.numBuckets := by
    have h5 := maxLoras_mul_gps_lt hd
    have h6 : d.maxLoras ≤ d.maxLoras * d.gps := Nat.le_mul_of_pos_right _ hd.gps_pos
    omega
  omega

/-- `sl * GROUPS_PER_SLOT + g` of a live pair is below `NUM_BUCKETS`. -/
theorem live_group_lt {d : Inst} (hd : d.ok) {sl g : Int} (h : liveCond d sl g)
    (hg : d.hasGroups = true) (hgps : d.gps ≠ 1) :
    (sl * (d.gps : Int) + g).toNat < d.numBuckets := by
  have h0 : 0 ≤ sl := h.1
  have h1 : sl < (d.maxLoras : Int) := h.2.1
  obtain ⟨hgg0, hgg1⟩ := h.2.2 hg
  have hglt : g < (d.gps : Int) := by
    rcases hgg1 with h1' | h1'
    · exact absurd h1' hgps
    · exact h1'
  have h2 : (sl.toNat : Int) = sl := Int.toNat_of_nonneg h0
  have h3 : (g.toNat : Int) = g := Int.toNat_of_nonneg hgg0
  have h4 : sl.toNat < d.maxLoras := by omega
  have h5 : g.toNat < d.gps := by omega
  have hgp : (0 : Int) ≤ (d.gps : Int) := by exact_mod_cast Nat.zero_le d.gps
  have hnn : 0 ≤ sl * (d.gps : Int) + g := by
    have hm := mul_nonneg h0 hgp
    linarith
  have h6 : sl * (d.gps : Int) + g = ((sl.toNat * d.gps + g.toNat : Nat) : Int) := by
    push_cast
    rw [h2, h3]
  have htv : (sl * (d.gps : Int) + g).toNat = sl.toNat * d.gps + g.toNat := by
    have h7 : ((sl * (d.gps : Int) + g).toNat : Int) = ((sl.toNat * d.gps + g.toNat : Nat) : Int) := by
      rw [Int.toNat_of_nonneg hnn]
      exact h6
    omega
  rw [htv]
  have h7 : sl.toNat * d.gps + g.toNat < (sl.toNat + 1) * d.gps := by
    have h8 : (sl.toNat + 1) * d.gps = sl.toNat * d.gps + d.gps := by ring
    omega
  have h9 : (sl.toNat + 1) * d.gps ≤ d.maxLoras * d.gps :=
    Nat.mul_le_mul_right _ (by omega)
  have h10 := maxLoras_mul_gps_lt hd
  omega

/-- A live slot fits `int32`. -/
theorem live_slot_sfits {d : Inst} (hd : d.ok) {sl g : Int} (h : liveCond d sl g) :
    IntV.SFits 32 sl := by
  have h0 : 0 ≤ sl := h.1
  have h1 : sl < (d.maxLoras : Int) := h.2.1
  have hml : (d.maxLoras : Int) < 2 ^ 31 := by
    have hb := hd.buckets_lt
    have hgp : 0 < d.gps := hd.gps_pos
    have he := hd.buckets_eq
    have hle : d.maxLoras ≤ d.gps * d.maxLoras := Nat.le_mul_of_pos_left _ hgp
    omega
  have h2p : ((2 : Int) ^ (32 - 1)) = 2 ^ 31 := by norm_num
  simp only [IntV.SFits, h2p]
  constructor <;> omega

/-- A live pair's `slot * GROUPS_PER_SLOT + group` fits `int32`. -/
theorem live_group_sfits {d : Inst} (hd : d.ok) {sl g : Int} (h : liveCond d sl g)
    (hg : d.hasGroups = true) (hgps : d.gps ≠ 1) :
    IntV.SFits 32 (sl * (d.gps : Int) + g) := by
  have hlt := live_group_lt hd h hg hgps
  have h0 : 0 ≤ sl := h.1
  obtain ⟨hgg0, -⟩ := h.2.2 hg
  have hgp : (0 : Int) ≤ (d.gps : Int) := by exact_mod_cast Nat.zero_le d.gps
  have hnn : 0 ≤ sl * (d.gps : Int) + g := by
    have hm := mul_nonneg h0 hgp
    linarith
  have hcast : ((sl * (d.gps : Int) + g).toNat : Int) = sl * (d.gps : Int) + g :=
    Int.toNat_of_nonneg hnn
  have hb : d.numBuckets < 2 ^ 31 := hd.buckets_lt
  have hnb : sl * (d.gps : Int) + g < (d.numBuckets : Int) := by omega
  have h2p : ((2 : Int) ^ (32 - 1)) = 2 ^ 31 := by norm_num
  simp only [IntV.SFits, h2p]
  constructor <;> omega

/-- The bucket of a pair whose lane is not live. -/
theorem bucketOf_dead {d : Inst} {p : Nat} {sl g : Int}
    (hsl : (d.tsVal (p / d.width)).toInt = sl) (hgi : (d.giVal p).toInt = g)
    (h : ¬ liveCond d sl g) : bucketOf d p = d.numBuckets - 1 := by
  have hc : ¬ (0 ≤ sl ∧ sl < (d.maxLoras : Int) ∧
      (d.hasGroups = true → 0 ≤ g ∧ (d.gps = 1 ∨ g < (d.gps : Int)))) := by
    simpa only [liveCond] using h
  simp only [bucketOf, hsl, hgi, ite_eq_right hc]

/-- The bucket of a live pair, without groups. -/
theorem bucketOf_live_slot {d : Inst} {p : Nat} {sl g : Int}
    (hsl : (d.tsVal (p / d.width)).toInt = sl) (hgi : (d.giVal p).toInt = g)
    (h : liveCond d sl g) : bucketOf d p = (if d.hasGroups = true ∧ d.gps ≠ 1 then sl * (d.gps : Int) + g else sl).toNat := by
  have hc : 0 ≤ sl ∧ sl < (d.maxLoras : Int) ∧
      (d.hasGroups = true → 0 ≤ g ∧ (d.gps = 1 ∨ g < (d.gps : Int))) := by
    simpa only [liveCond] using h
  simp only [bucketOf, hsl, hgi, ite_eq_left hc]

/-- The bucket of a live pair, with one group per slot. -/
theorem bucketOf_live_g1 {d : Inst} {p : Nat} {sl g : Int}
    (hsl : (d.tsVal (p / d.width)).toInt = sl) (hgi : (d.giVal p).toInt = g)
    (h : liveCond d sl g) (hgps : d.gps = 1) :
    bucketOf d p = sl.toNat := by
  have hc : ¬ (d.hasGroups = true ∧ d.gps ≠ 1) := by
    rintro ⟨-, hne⟩
    exact hne hgps
  rw [bucketOf_live_slot hsl hgi h, ite_eq_right hc]

/-- The bucket of a live pair, with groups and more than one group per slot. -/
theorem bucketOf_live_gN {d : Inst} {p : Nat} {sl g : Int}
    (hsl : (d.tsVal (p / d.width)).toInt = sl) (hgi : (d.giVal p).toInt = g)
    (h : liveCond d sl g) (hg : d.hasGroups = true) (hgps : d.gps ≠ 1) :
    bucketOf d p = (sl * (d.gps : Int) + g).toNat := by
  rw [bucketOf_live_slot hsl hgi h, ite_eq_left ⟨hg, hgps⟩]

/-- Every bucket is below `NUM_BUCKETS`. -/
theorem bucketOf_lt {d : Inst} (hd : d.ok) (p : Nat) : bucketOf d p < d.numBuckets := by
  by_cases hlive : liveCond d (d.tsVal (p / d.width)).toInt (d.giVal p).toInt
  · by_cases h2 : d.hasGroups = true ∧ d.gps ≠ 1
    · rw [bucketOf_live_gN rfl rfl hlive h2.1 h2.2]
      exact live_group_lt hd hlive h2.1 h2.2
    · rw [bucketOf_live_slot rfl rfl hlive, ite_eq_right h2]
      exact live_slot_lt hd hlive
  · rw [bucketOf_dead rfl rfl hlive]
    have := numBuckets_pos hd
    omega

/-- A bucket as an integer is nonnegative. -/
theorem bucketOf_nonneg (d : Inst) (p : Nat) : 0 ≤ (bucketOf d p : Int) := by
  simp only [bucketOf]
  split_ifs <;> exact Int.natCast_nonneg _

/-! ## The lane value of `route_bucket_ids` -/

/-- Each live lane of `route_bucket_ids` gets its pair's bucket, a dead one `-1`. -/
def RawOk (d : Inst) (p : Nat) (t : Tensor [d.block] I32) : Prop :=
  ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
    (t.get (l, ())).toInt = (bucketOf d (p * d.block + (l : Nat)) : Int) ∨
      ((t.get (l, ())).toInt = -1 ∧ bucketOf d (p * d.block + (l : Nat)) = d.numBuckets - 1)

/-- `add_counts` turns the `-1` of a dead lane into `NUM_BUCKETS - 1`: after it, every enabled
lane's value is its pair's bucket. -/
theorem laneOf_raw {d : Inst} (hd : d.ok) {p : Nat} {t : Tensor [d.block] I32}
    (h : RawOk d p t) :
    ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      ((tl.where (t.ltS (IntV.ofInt 0 : I32))
          (tl.full [d.block] (IntV.ofNat (d.numBuckets - 1) : I32)) t).get (l, ())).toInt
        = (bucketOf d (p * d.block + (l : Nat)) : Int) := by
  intro l hl
  obtain h1 | ⟨h1, h2⟩ := h l hl
  · have hneg : ¬ ((t.get (l, ())).toInt < (0 : Int)) := by
      rw [h1]
      have := bucketOf_nonneg d (p * d.block + (l : Nat))
      omega
    have hlt : IntV.lt (t.get (l, ())) (IntV.ofInt 0 : I32) = false := by
      rw [IntV.lt_eq, IntV.I32.toInt_ofInt (by norm_num)]
      exact decide_eq_false hneg
    simp only [tl.where, Tensor.where_, Tensor.ltS, Tensor.map, tl.full, Tensor.get_full,
      hlt, Bool.false_eq_true, ite_false]
    exact h1
  · have hneg : (t.get (l, ())).toInt < (0 : Int) := by
      rw [h1]
      norm_num
    have hlt : IntV.lt (t.get (l, ())) (IntV.ofInt 0 : I32) = true := by
      rw [IntV.lt_eq, IntV.I32.toInt_ofInt (by norm_num)]
      exact decide_eq_true hneg
    have hm1 : d.numBuckets - 1 < 2 ^ 31 := by
      have := hd.buckets_lt
      omega
    simp only [tl.where, Tensor.where_, Tensor.ltS, Tensor.map, tl.full, Tensor.get_full,
      hlt, ite_true]
    rw [IntV.I32.toInt_ofNat hm1, h2]

/-! ## `int32` sums -/

/-- A sum of `int32` values where only one index can contribute. -/
theorem sum_map_single {w : Nat} {ι : Type} [DecidableEq ι] (l : List ι) (f : ι → IntV true w)
    (hl : l.Nodup) (b : ι) (hf : ∀ i ∈ l, i ≠ b → f i = 0) :
    (l.map f).sum = if b ∈ l then f b else 0 := by
  classical
  induction l with
  | nil => simp
  | cons a t ih =>
    obtain ⟨hat, htn⟩ := List.nodup_cons.mp hl
    rw [List.map_cons, List.sum_cons,
      ih htn (fun i hi hne => hf i (List.mem_cons_of_mem _ hi) hne)]
    by_cases hab : a = b
    · subst hab
      simp [hat]
    · have hfa : f a = 0 := hf a List.mem_cons_self hab
      have hba : ¬ (b = a) := fun h => hab h.symm
      simp [hfa, hba]

/-! ## `counts` cells -/

/-- Element `k` of `counts` and element `b` of it are the same cell only when `k = b`. -/
theorem counts_ptr_inj {d : Inst} {k b : Int} (h : (d.counts.ptr k).addr = (d.counts.ptr b).addr) :
    k = b := by
  simp only [Buf.ptr] at h
  have hb : DType.i32.bytes = 4 := rfl
  rw [hb] at h
  omega

/-- The increment `stepInc` at cell `b` of a step on element `k` of `counts`. -/
theorem stepInc_counts_self {d : Inst} {b : Int} {o n : I32} :
    stepInc (d.counts.ptr (b : Int)).addr ⟨DType.i32, (d.counts.ptr (b : Int)).addr, o, n⟩
      = n - o :=
  stepInc_self o n

/-- The increment `stepInc` at cell `b` of a step on a different element of `counts` is `0`. -/
theorem stepInc_counts_ne {d : Inst} {k b : Int} (h : k ≠ b) (o n : I32) :
    stepInc (d.counts.ptr b).addr ⟨DType.i32, (d.counts.ptr k).addr, o, n⟩ = 0 := by
  refine stepInc_ne o n ?_
  intro hc
  exact h (counts_ptr_inj hc)


/-! ## The value of `route_bucket_ids` on a lane

`route_bucket_ids` selects, per lane, the bucket of its pair, or `int32` `-1` when the pair is dead.
Given the values the two loads read, the resulting tensor is `RawOk`: every lane below `n` holds
its pair's bucket, or `-1` where `bucketOf` is `NUM_BUCKETS - 1`. -/

private theorem castInt_toInt_eq {s : Shape} (x : Tensor s I32) (i : Idx s) :
    ((x.castInt : Tensor s I32).get i).toInt = (x.get i).toInt := by
  simp only [Tensor.get_castInt]
  exact IntV.toInt_cast _ (le_refl _)

/-- A lane the condition keeps takes the first operand of `tl.where`. -/
private theorem where_true {s : Shape} {α : Type} {c : Tensor s Bool} {x y : Tensor s α}
    (i : Idx s) (h : c.get i = true) : (tl.where c x y).get i = x.get i := by
  simp only [tl.where, Tensor.get_where, h, ite_true]

/-- A lane the condition cuts takes the second operand of `tl.where`. -/
private theorem where_false {s : Shape} {α : Type} {c : Tensor s Bool} {x y : Tensor s α}
    (i : Idx s) (h : c.get i = false) : (tl.where c x y).get i = y.get i := by
  simp only [tl.where, Tensor.get_where, h, Bool.false_eq_true, ite_false]

/-- `MAX_LORAS` fits `int32`. -/
private theorem maxLoras_lt_i32 {d : Inst} (hd : d.ok) : d.maxLoras < 2 ^ 31 := by
  have hbe := hd.buckets_eq
  have hb := hd.buckets_lt
  have hgp := hd.gps_pos
  have h1 : d.maxLoras ≤ d.gps * d.maxLoras := Nat.le_mul_of_pos_left _ hgp
  omega

/-- The `int32` constant `0`. -/
private theorem ofInt_zero_toInt : ((IntV.ofInt 0 : I32)).toInt = 0 :=
  IntV.I32.toInt_ofInt (by norm_num)

/-- The `int32` constant `-1`. -/
private theorem ofInt_neg_one_toInt : ((IntV.ofInt (-1) : I32)).toInt = -1 :=
  IntV.I32.toInt_ofInt (by norm_num)

/-- A live slot times `GROUPS_PER_SLOT` fits `int32`. -/
theorem live_mul_sfits {d : Inst} (hd : d.ok) {sl g : Int} (h : liveCond d sl g) :
    IntV.SFits 32 (sl * (d.gps : Int)) := by
  have h0 : 0 ≤ sl := h.1
  have h1 : sl < (d.maxLoras : Int) := h.2.1
  have hml := maxLoras_mul_gps_lt hd
  have hml' : ((d.maxLoras * d.gps : Nat) : Int) < (d.numBuckets : Int) := by
    exact_mod_cast hml
  have hcast : ((d.maxLoras * d.gps : Nat) : Int) = (d.maxLoras : Int) * (d.gps : Int) := by
    push_cast; ring
  have hb := hd.buckets_lt
  have hgp : (0 : Int) < (d.gps : Int) := by exact_mod_cast hd.gps_pos
  have h2p : ((2 : Int) ^ (32 - 1)) = 2 ^ 31 := by norm_num
  simp only [IntV.SFits, h2p]
  constructor
  · have : 0 ≤ sl * (d.gps : Int) := mul_nonneg h0 (le_of_lt hgp)
    linarith
  · have hlt : sl * (d.gps : Int) < (d.maxLoras : Int) * (d.gps : Int) :=
      mul_lt_mul_of_pos_right h1 hgp
    omega

/-- Without groups, `route_bucket_ids` gives every lane below `n` its pair's bucket, or `-1`. -/
theorem raw_nogroups {d : Inst} (hd : d.ok) {p : Nat} {slots : Tensor [d.block] I32}
    (hn : d.hasGroups = false)
    (hsl : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (slots.get (l, ())).toInt = (d.tsVal ((p * d.block + (l : Nat)) / d.width)).toInt) :
    RawOk d p
      (tl.where
        (((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
          (slots.ltS (IntV.ofNat d.maxLoras)))
        (slots.castInt : Tensor [d.block] I32)
        (tl.full [d.block] (IntV.ofInt (-1) : I32))) := by
  have hml := maxLoras_lt_i32 hd
  intro l hl
  set i : Idx [d.block] := (l, ()) with hi_def
  set sl := (slots.get i).toInt with hsl_def
  set g := (d.giVal (p * d.block + (l : Nat))).toInt with hg_def
  have hslv : (d.tsVal ((p * d.block + (l : Nat)) / d.width)).toInt = sl := by
    rw [hsl_def]; exact (hsl l hl).symm
  have hgiv : (d.giVal (p * d.block + (l : Nat))).toInt = g := rfl
  have hif : ¬ (d.hasGroups = true ∧ d.gps ≠ 1) := by rw [hn]; simp
  by_cases hlive : liveCond d sl g
  · have hz : 0 ≤ sl := hlive.1
    have hlt : sl < (d.maxLoras : Int) := hlive.2.1
    have hcond : ((((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
        (slots.ltS (IntV.ofNat d.maxLoras))).get i) = true := by
      simp only [Tensor.get_and, Tensor.get_le, Tensor.get_ltS, tl.full, Tensor.get_full,
        IntV.le_eq, IntV.lt_eq, ofInt_zero_toInt, IntV.I32.toInt_ofNat hml]
      rw [decide_eq_true hz, decide_eq_true hlt]
      rfl
    rw [where_true i hcond]
    left
    rw [castInt_toInt_eq, ← hsl_def]
    have hb : bucketOf d (p * d.block + (l : Nat)) = sl.toNat := by
      rw [bucketOf_live_slot hslv hgiv hlive, ite_eq_right hif]
    rw [hb, Int.toNat_of_nonneg hz]
  · have hc : ¬ (0 ≤ sl ∧ sl < (d.maxLoras : Int)) := by
      rintro ⟨a, b⟩
      exact hlive ⟨a, b, by rw [hn]; simp⟩
    have hcond : ((((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
        (slots.ltS (IntV.ofNat d.maxLoras))).get i) = false := by
      simp only [Tensor.get_and, Tensor.get_le, Tensor.get_ltS, tl.full, Tensor.get_full,
        IntV.le_eq, IntV.lt_eq, ofInt_zero_toInt, IntV.I32.toInt_ofNat hml]
      by_cases h1 : 0 ≤ sl
      · have h2 : ¬ sl < (d.maxLoras : Int) := fun h2 => hc ⟨h1, h2⟩
        rw [decide_eq_true h1, decide_eq_false h2]
        rfl
      · rw [decide_eq_false h1]
        rfl
    rw [where_false i hcond]
    right
    exact ⟨by simp only [tl.full, Tensor.get_full, ofInt_neg_one_toInt],
      bucketOf_dead hslv hgiv hlive⟩

/-- With groups and one group per slot, `route_bucket_ids` gives every lane below `n` its pair's
bucket, or `-1`. -/
theorem raw_g1 {d : Inst} (hd : d.ok) {p : Nat} {slots groups : Tensor [d.block] I32}
    (hg : d.hasGroups = true) (hgps : d.gps = 1)
    (hsl : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (slots.get (l, ())).toInt = (d.tsVal ((p * d.block + (l : Nat)) / d.width)).toInt)
    (hgr : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (groups.get (l, ())).toInt = (d.giVal (p * d.block + (l : Nat))).toInt) :
    RawOk d p
      (tl.where
        ((((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
            (slots.ltS (IntV.ofNat d.maxLoras))).and
          ((tl.full [d.block] (IntV.ofInt 0 : I32)).le groups))
        (slots.castInt : Tensor [d.block] I32)
        (tl.full [d.block] (IntV.ofInt (-1) : I32))) := by
  have hml := maxLoras_lt_i32 hd
  intro l hl
  set i : Idx [d.block] := (l, ()) with hi_def
  set sl := (slots.get i).toInt with hsl_def
  set g := (groups.get i).toInt with hg_def
  have hslv : (d.tsVal ((p * d.block + (l : Nat)) / d.width)).toInt = sl := by
    rw [hsl_def]; exact (hsl l hl).symm
  have hgiv : (d.giVal (p * d.block + (l : Nat))).toInt = g := by
    rw [hg_def]; exact (hgr l hl).symm
  by_cases hlive : liveCond d sl g
  · have hz : 0 ≤ sl := hlive.1
    have hlt : sl < (d.maxLoras : Int) := hlive.2.1
    have hgz : 0 ≤ g := (hlive.2.2 hg).1
    have hcond : (((((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
        (slots.ltS (IntV.ofNat d.maxLoras))).and
          ((tl.full [d.block] (IntV.ofInt 0 : I32)).le groups)).get i) = true := by
      simp only [Tensor.get_and, Tensor.get_le, Tensor.get_ltS, tl.full, Tensor.get_full,
        IntV.le_eq, IntV.lt_eq, ofInt_zero_toInt, IntV.I32.toInt_ofNat hml]
      rw [decide_eq_true hz, decide_eq_true hlt, decide_eq_true hgz]
      rfl
    rw [where_true i hcond]
    left
    rw [castInt_toInt_eq, ← hsl_def]
    have hb : bucketOf d (p * d.block + (l : Nat)) = sl.toNat :=
      bucketOf_live_g1 hslv hgiv hlive hgps
    rw [hb, Int.toNat_of_nonneg hz]
  · have hc : ¬ (0 ≤ sl ∧ sl < (d.maxLoras : Int) ∧ 0 ≤ g) := by
      rintro ⟨a, b, c⟩
      exact hlive ⟨a, b, fun _ => ⟨c, Or.inl hgps⟩⟩
    have hcond : (((((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
        (slots.ltS (IntV.ofNat d.maxLoras))).and
          ((tl.full [d.block] (IntV.ofInt 0 : I32)).le groups)).get i) = false := by
      simp only [Tensor.get_and, Tensor.get_le, Tensor.get_ltS, tl.full, Tensor.get_full,
        IntV.le_eq, IntV.lt_eq, ofInt_zero_toInt, IntV.I32.toInt_ofNat hml]
      by_cases h1 : 0 ≤ sl
      · by_cases h2 : sl < (d.maxLoras : Int)
        · have h3 : ¬ 0 ≤ g := fun h3 => hc ⟨h1, h2, h3⟩
          rw [decide_eq_true h1, decide_eq_true h2, decide_eq_false h3]
          rfl
        · rw [decide_eq_true h1, decide_eq_false h2]
          rfl
      · rw [decide_eq_false h1]
        rfl
    rw [where_false i hcond]
    right
    exact ⟨by simp only [tl.full, Tensor.get_full, ofInt_neg_one_toInt],
      bucketOf_dead hslv hgiv hlive⟩

/-- With groups and more than one group per slot, `route_bucket_ids` gives every lane below `n`
its pair's bucket, or `-1`.

The `where` condition is the model's `live` (which includes `0 <= groups`), followed by
`groups < GROUPS_PER_SLOT`; without the `0 <= groups` conjunct the statement is false (a lane with
a negative group id would pass the condition while its bucket is dead). -/
theorem raw_gN {d : Inst} (hd : d.ok) {p : Nat} {slots groups : Tensor [d.block] I32}
    (hg : d.hasGroups = true) (hgps : d.gps ≠ 1)
    (hsl : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (slots.get (l, ())).toInt = (d.tsVal ((p * d.block + (l : Nat)) / d.width)).toInt)
    (hgr : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
      (groups.get (l, ())).toInt = (d.giVal (p * d.block + (l : Nat))).toInt) :
    RawOk d p
      (tl.where
        (((((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
            (slots.ltS (IntV.ofNat d.maxLoras))).and
          ((tl.full [d.block] (IntV.ofInt 0 : I32)).le groups)).and
          (groups.ltS (IntV.ofNat d.gps)))
        ((slots.castInt : Tensor [d.block] I32) * (IntV.ofNat d.gps : I32) +
          (groups.castInt : Tensor [d.block] I32))
        (tl.full [d.block] (IntV.ofInt (-1) : I32))) := by
  have hml := maxLoras_lt_i32 hd
  have hgpslt := hd.gps_lt
  intro l hl
  set i : Idx [d.block] := (l, ()) with hi_def
  set sl := (slots.get i).toInt with hsl_def
  set g := (groups.get i).toInt with hg_def
  have hslv : (d.tsVal ((p * d.block + (l : Nat)) / d.width)).toInt = sl := by
    rw [hsl_def]; exact (hsl l hl).symm
  have hgiv : (d.giVal (p * d.block + (l : Nat))).toInt = g := by
    rw [hg_def]; exact (hgr l hl).symm
  have hgp : (0 : Int) < (d.gps : Int) := by exact_mod_cast hd.gps_pos
  by_cases hlive : liveCond d sl g
  · have hz : 0 ≤ sl := hlive.1
    have hlt : sl < (d.maxLoras : Int) := hlive.2.1
    have hgz : 0 ≤ g := (hlive.2.2 hg).1
    have hglt : g < (d.gps : Int) := by
      rcases (hlive.2.2 hg).2 with h1 | h1
      · exact absurd h1 hgps
      · exact h1
    have hcond : (((((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
        (slots.ltS (IntV.ofNat d.maxLoras))).and
          ((tl.full [d.block] (IntV.ofInt 0 : I32)).le groups)).and
          (groups.ltS (IntV.ofNat d.gps))).get i = true := by
      simp only [Tensor.get_and, Tensor.get_le, Tensor.get_ltS, tl.full, Tensor.get_full,
        IntV.le_eq, IntV.lt_eq, ofInt_zero_toInt, IntV.I32.toInt_ofNat hml,
        IntV.I32.toInt_ofNat hgpslt]
      rw [decide_eq_true hz, decide_eq_true hlt, decide_eq_true hgz, decide_eq_true hglt]
      rfl
    rw [where_true i hcond]
    left
    have ha : (IntV.cast (slots.get i) : I32).toInt = (sl.toNat : Int) := by
      rw [IntV.toInt_cast (slots.get i) (le_refl 32), ← hsl_def]
      exact (Int.toNat_of_nonneg hz).symm
    have hb : (IntV.ofNat d.gps : I32).toInt = (d.gps : Int) := IntV.I32.toInt_ofNat hgpslt
    have hc : (IntV.cast (groups.get i) : I32).toInt = (g.toNat : Int) := by
      rw [IntV.toInt_cast (groups.get i) (le_refl 32), ← hg_def]
      exact (Int.toNat_of_nonneg hgz).symm
    have hbound : sl.toNat * d.gps + g.toNat < 2 ^ 31 := by
      have hs : (sl.toNat : Int) < (d.maxLoras : Int) := by
        rw [Int.toNat_of_nonneg hz]; exact hlt
      have hgs : (g.toNat : Int) < (d.gps : Int) := by
        rw [Int.toNat_of_nonneg hgz]; exact hglt
      have hx : (sl.toNat : Int) ≤ (d.maxLoras : Int) - 1 := by omega
      have hy : (g.toNat : Int) ≤ (d.gps : Int) - 1 := by omega
      have h1 : (sl.toNat : Int) * (d.gps : Int) ≤ ((d.maxLoras : Int) - 1) * (d.gps : Int) :=
        mul_le_mul_of_nonneg_right hx (le_of_lt hgp)
      have hml' : ((d.maxLoras * d.gps : Nat) : Int) < (d.numBuckets : Int) := by
        exact_mod_cast maxLoras_mul_gps_lt hd
      have hcast : ((d.maxLoras * d.gps : Nat) : Int) =
          (d.maxLoras : Int) * (d.gps : Int) := by push_cast; ring
      have hb2 : (d.numBuckets : Int) ≤ 2 ^ 31 := by
        exact_mod_cast Nat.le_of_lt hd.buckets_lt
      have key : (sl.toNat : Int) * (d.gps : Int) + (g.toNat : Int) < 2 ^ 31 := by
        nlinarith [h1, hy, hml', hcast, hb2]
      have h2 : ((sl.toNat * d.gps + g.toNat : Nat) : Int) < 2 ^ 31 := by
        push_cast
        exact key
      exact_mod_cast h2
    have hmul : ((IntV.cast (slots.get i) : I32) * (IntV.ofNat d.gps : I32)).toInt =
        ((sl.toNat * d.gps : Nat) : Int) := by
      refine IntV.I32.mul_nat ha hb ?_
      omega
    have hadd : (((IntV.cast (slots.get i) : I32) * (IntV.ofNat d.gps : I32)) +
        (IntV.cast (groups.get i) : I32)).toInt =
        ((sl.toNat * d.gps + g.toNat : Nat) : Int) := by
      refine IntV.I32.add_nat hmul hc ?_
      omega
    have hrot : ((sl.toNat * d.gps + g.toNat : Nat) : Int) = sl * (d.gps : Int) + g := by
      push_cast
      rw [Int.toNat_of_nonneg hz, Int.toNat_of_nonneg hgz]
    have hval : ((((slots.castInt : Tensor [d.block] I32) * (IntV.ofNat d.gps : I32) +
        (groups.castInt : Tensor [d.block] I32)).get i)).toInt =
        ((sl * (d.gps : Int) + g).toNat : Int) := by
      simp only [Tensor.get_add, Tensor.get_muls, Tensor.get_castInt]
      rw [hadd, hrot]
      exact (Int.toNat_of_nonneg (add_nonneg (mul_nonneg hz (le_of_lt hgp)) hgz)).symm
    rw [bucketOf_live_gN hslv hgiv hlive hg hgps]
    exact hval
  · have hc : ¬ (0 ≤ sl ∧ sl < (d.maxLoras : Int) ∧ 0 ≤ g ∧ g < (d.gps : Int)) := by
      rintro ⟨a, b, c, e⟩
      exact hlive ⟨a, b, fun _ => ⟨c, Or.inr e⟩⟩
    have hcond : (((((tl.full [d.block] (IntV.ofInt 0 : I32)).le slots).and
        (slots.ltS (IntV.ofNat d.maxLoras))).and
          ((tl.full [d.block] (IntV.ofInt 0 : I32)).le groups)).and
          (groups.ltS (IntV.ofNat d.gps))).get i = false := by
      simp only [Tensor.get_and, Tensor.get_le, Tensor.get_ltS, tl.full, Tensor.get_full,
        IntV.le_eq, IntV.lt_eq, ofInt_zero_toInt, IntV.I32.toInt_ofNat hml,
        IntV.I32.toInt_ofNat hgpslt]
      by_cases h1 : 0 ≤ sl
      · by_cases h2 : sl < (d.maxLoras : Int)
        · by_cases h3 : 0 ≤ g
          · have h4 : ¬ g < (d.gps : Int) := fun h4 => hc ⟨h1, h2, h3, h4⟩
            rw [decide_eq_true h1, decide_eq_true h2, decide_eq_true h3, decide_eq_false h4]
            rfl
          · rw [decide_eq_true h1, decide_eq_true h2, decide_eq_false h3]
            rfl
        · rw [decide_eq_true h1, decide_eq_false h2]
          rfl
      · rw [decide_eq_false h1]
        rfl
    rw [where_false i hcond]
    right
    exact ⟨by simp only [tl.full, Tensor.get_full, ofInt_neg_one_toInt],
      bucketOf_dead hslv hgiv hlive⟩

end Claims.RouteHistogram
