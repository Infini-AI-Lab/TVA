import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Num
import Proof.Trees
import Proof.Lane

/-!
# One output lane is good

`lane_good`: at trip `t`, lane `l` (position `p = base + l`) adds the running sum to its scan;
the output's tree is the lane's scan tree (renumbered to positions) joined with the running sum's
tree. Its leaves are the positions of `rng p` (and `+0`s outside the sequence), its depth is the
claim's `depth`, so `good_of_tree` applies.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

variable {d : Inst} {i : Nat}

/-- The scan's leaves at lane `l`. -/
def scanL (rev : Bool) (BT l : Nat) : List Nat :=
  if rev then List.range' l (BT - l) else List.range (l + 1)

/-- A lane's scan sum over positions. -/
theorem scan_sum (f : Nat → ℚ) (rev : Bool) {BT l : Nat} (hl : l < BT) (c : Nat) :
    ((scanL rev BT l).map (fun j => f (j + c))).sum =
      if rev then ∑ q ∈ Finset.Ico (c + l) (c + BT), f q
      else ∑ q ∈ Finset.Ico c (c + l + 1), f q := by
  unfold scanL
  cases rev
  · simp only [Bool.false_eq_true, ite_false]
    rw [list_range_sum (fun j => f (j + c)), range_shift_sum]
    congr 1
  · simp only [ite_true]
    rw [list_range'_sum (fun j => f (j + c)), Ico_shift_sum]
    rw [show l + (BT - l) + c = c + BT by omega, Nat.add_comm l c]

/-- The geometry of lane `l` of trip `t`. -/
structure LaneGeo (d : Inst) (i t l : Nat) : Prop where
  p_lo : loOf d i ≤ base d i t + l
  p_hi : base d i t + l < hiOf d i
  blk_eq : blk d (seqOf d i) (base d i t + l) = bt d i t
  start_eq : blkStart d (seqOf d i) (base d i t + l) = base d i t
  before_eq : before d (seqOf d i) (base d i t + l) = t
  lane_eq : laneDepth d (seqOf d i) (base d i t + l) = if d.reverse then d.BT - 1 - l else l

theorem laneGeo (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d) {t l : Nat}
    (ht : t < nbOf d i) (hl : l < d.BT) (hen : bt d i t * d.BT + l < lenOf d i) :
    LaneGeo d i t l := by
  have hb := bt_pos hd
  have hla := lo_add_len hT hi
  have hblk : blk d (seqOf d i) (base d i t + l) = bt d i t := by
    unfold blk base
    show (loOf d i + bt d i t * d.BT + l - loOf d i) / d.BT = bt d i t
    rw [show loOf d i + bt d i t * d.BT + l - loOf d i = l + bt d i t * d.BT by omega,
      Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hl, Nat.zero_add]
  have hstart : blkStart d (seqOf d i) (base d i t + l) = base d i t := by
    unfold blkStart; rw [hblk]; rfl
  refine ⟨by unfold base; omega, by unfold base; omega, hblk, hstart, ?_, ?_⟩
  · unfold before; rw [hblk]
    show (if d.reverse then nbOf d i - 1 - bt d i t else bt d i t) = t
    unfold bt; split <;> omega
  · unfold laneDepth; rw [hstart]
    split <;> omega

/-- The positions summed at lane `l`'s output: the scan's and the running sum's. -/
theorem total_sum (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d)
    {t l : Nat} (ht : t < nbOf d i) (hl : l < d.BT) (hen : bt d i t * d.BT + l < lenOf d i)
    (f : Nat → ℚ) (hf : ∀ q, hiOf d i ≤ q → f q = 0) :
    (if d.reverse then ∑ q ∈ Finset.Ico (base d i t + l) (base d i t + d.BT), f q
      else ∑ q ∈ Finset.Ico (base d i t) (base d i t + l + 1), f q) +
      ∑ q ∈ Pset d i t, f q = ∑ q ∈ rng d (seqOf d i) (base d i t + l), f q := by
  have g := laneGeo hd hT hi ht hl hen
  have hla := lo_add_len hT hi
  have hlen := len_le_nb (i := i) hd
  unfold rng Pset
  cases hr : d.reverse
  · simp only [Bool.false_eq_true, ite_false]
    have hbt : bt d i t = t := by unfold bt; rw [hr]; rfl
    have hbase : base d i t = loOf d i + t * d.BT := by unfold base; rw [hbt]
    rw [add_comm, hbase, Finset.sum_Ico_consecutive f (by omega) (by omega),
      Finset.Ico_add_one_right_eq_Icc]
    rfl
  · simp only [ite_true]
    have hbt : bt d i t = nbOf d i - 1 - t := by unfold bt; rw [hr]; rfl
    have hbase : base d i t + d.BT = loOf d i + (nbOf d i - t) * d.BT := by
      unfold base; rw [hbt]
      have : nbOf d i - t = (nbOf d i - 1 - t) + 1 := by omega
      rw [this]; ring
    have e3 : (nbOf d i - t) * d.BT ≤ nbOf d i * d.BT := Nat.mul_le_mul_right _ (by omega)
    have hp := g.p_hi
    have h1 : base d i t + l ≤ loOf d i + (nbOf d i - t) * d.BT := by omega
    rw [hbase, Finset.sum_Ico_consecutive f h1 (by omega)]
    show ∑ q ∈ Finset.Ico (base d i t + l) (loOf d i + nbOf d i * d.BT), f q =
      ∑ q ∈ Finset.Ico (base d i t + l) (hiOf d i), f q
    rw [← Finset.sum_Ico_consecutive f (n := hiOf d i) (by omega) (by omega)]
    rw [Finset.sum_eq_zero (s := Finset.Ico (hiOf d i) (loOf d i + nbOf d i * d.BT))
      (fun q hq => hf q (Finset.mem_Ico.mp hq).1), add_zero]

/-- The positions of lane `l`'s scan and of the running sum before it lie in `rng p`, when they
are inside the sequence. -/
theorem leaves_in_rng (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d)
    {t l : Nat} (ht : t < nbOf d i) (hl : l < d.BT) (hen : bt d i t * d.BT + l < lenOf d i)
    {q : Nat} (hq : (∃ j ∈ scanL d.reverse d.BT l, q = j + base d i t) ∨ q ∈ Pset d i t)
    (hin : loOf d i ≤ q ∧ q < hiOf d i) : q ∈ rng d (seqOf d i) (base d i t + l) := by
  have g := laneGeo hd hT hi ht hl hen
  unfold rng
  cases hr : d.reverse
  · simp only [Bool.false_eq_true, ite_false, Finset.mem_Icc]
    have hbt : bt d i t = t := by unfold bt; rw [hr]; rfl
    rcases hq with ⟨j, hj, rfl⟩ | hq
    · simp only [scanL, hr, Bool.false_eq_true, ite_false, List.mem_range] at hj
      refine ⟨hin.1, by omega⟩
    · simp only [Pset, hr, Bool.false_eq_true, ite_false, Finset.mem_Ico] at hq
      unfold base; rw [hbt]
      exact ⟨hin.1, by omega⟩
  · simp only [ite_true, Finset.mem_Ico]
    have hbt : bt d i t = nbOf d i - 1 - t := by unfold bt; rw [hr]; rfl
    rcases hq with ⟨j, hj, rfl⟩ | hq
    · simp only [scanL, hr, ite_true, List.mem_range'_1] at hj
      exact ⟨by omega, hin.2⟩
    · simp only [Pset, hr, ite_true, Finset.mem_Ico] at hq
      have hbase : base d i t + d.BT = loOf d i + (nbOf d i - t) * d.BT := by
        unfold base; rw [hbt]
        have : nbOf d i - t = (nbOf d i - 1 - t) + 1 := by omega
        rw [this]; ring
      exact ⟨by omega, hin.2⟩

/-- `float32` leaves outside the sequence are `+0`, whose value is `0`. -/
theorem X_toRat_out {q : Nat} (hq : hiOf d i ≤ q) : (X d i q).toRat = 0 := by
  rw [X_out (by omega), ofPat_zero_toRat]

/-- **Lane `l` of trip `t` is good**: its scan tree (renumbered to positions) joined with the
running sum by the `b_o + b_z` addition, then scaled. -/
theorem lane_good (hd : d.ok) (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d)
    {t l : Nat} (ht : t < nbOf d i) (hl : l < d.BT) (hen : bt d i t * d.BT + l < lenOf d i)
    {z rs ra va : F32} (hz : ZTree d i t z) {tl : Tree}
    (hsl : tl.leaves.Perm (scanL d.reverse d.BT l))
    (hev : tl.Eval (AddRel false) (fun j => X d i (j + base d i t)) rs)
    (hadd : AddRel false rs z ra)
    (hv : if d.hasScale then FloatV.NaNEq (FloatV.mul ra (sigma32 d)) va else va = ra) :
    Good d (seqOf d i) (base d i t + l) (headOf d i) va := by
  obtain ⟨tz, hE, hD, hN, hL, hS, hA⟩ := hz
  have g := laneGeo hd hT hi ht hl hen
  have hmap := eval_mapLeaves (g := fun j => j + base d i t) hev
  have hperm : (tl.mapLeaves (fun j => j + base d i t)).leaves.Perm
      ((scanL d.reverse d.BT l).map (fun j => j + base d i t)) := by
    rw [Tree.leaves_mapLeaves]; exact hsl.map _
  have hlenL : (scanL d.reverse d.BT l).length = if d.reverse then d.BT - l else l + 1 := by
    unfold scanL; split <;> simp
  have hnodes : tl.nodes + 1 = (scanL d.reverse d.BT l).length := by
    rw [Tree.nodes_add_one, hsl.length_eq]
  have hdepth := Tree.depth_le_nodes tl
  have hb := bt_pos hd
  have hnb := nb_lt_len (i := i) hd
  have hlenT := lenOf_le hT hi
  have hclog : Nat.clog 2 d.BT ≤ 20 := Nat.clog_le_of_le_pow hd.bt_le
  have hT22 := hd.T_le
  have hnbl := nb_le_len (i := i) hd
  refine good_of_tree (x := X d i) (t := .node (tl.mapLeaves (fun j => j + base d i t)) tz)
    (r := ra) (.node hmap hE hadd) ?_ ?_ ?_ ?_ ?_ ?_ hv
  · -- the leaves are finite in range
    intro hIR q hq
    simp only [Tree.leaves, List.mem_append] at hq
    by_cases hin : loOf d i ≤ q ∧ q < hiOf d i
    · rw [X_in hin]
      refine hIR.1 q (leaves_in_rng hd hT hi ht hl hen ?_ hin)
      rcases hq with hq | hq
      · rw [Tree.leaves_mapLeaves, List.mem_map] at hq
        obtain ⟨j, hj, rfl⟩ := hq
        exact Or.inl ⟨j, hsl.subset hj, rfl⟩
      · rcases hL q hq with h | h
        · exact Or.inr h
        · exact absurd hin (by omega)
    · rw [X_out hin]; exact ofPat_zero_isFinite
  · -- the exact sum
    rw [Tree.sumOf_node, hS, sumOf_perm FloatV.toRat (X d i) hperm, List.map_map]
    rw [show ((fun i_1 => (X d i i_1).toRat) ∘ fun j => j + base d i t) =
      (fun j => (fun q => (X d i q).toRat) (j + base d i t)) from rfl]
    rw [scan_sum (fun q => (X d i q).toRat) d.reverse hl (base d i t)]
    rw [show base d i t + l + 1 = base d i t + l + 1 from rfl]
    rw [total_sum hd hT hi ht hl hen _ (fun q hq => X_toRat_out hq)]
    unfold refSum
    refine Finset.sum_congr rfl (fun q hq => ?_)
    have hin : loOf d i ≤ q ∧ q < hiOf d i := by
      unfold rng at hq
      have := g.p_lo; have := g.p_hi
      split at hq
      · rw [Finset.mem_Ico] at hq; exact ⟨by omega, hq.2⟩
      · rw [Finset.mem_Icc] at hq; exact ⟨hq.1, by omega⟩
    rw [X_in hin]
  · -- the absolute sum
    rw [Tree.absSumOf_node, hA, absSumOf_perm FloatV.toRat (X d i) hperm, List.map_map]
    rw [show ((fun i_1 => |(X d i i_1).toRat|) ∘ fun j => j + base d i t) =
      (fun j => (fun q => |(X d i q).toRat|) (j + base d i t)) from rfl]
    rw [scan_sum (fun q => |(X d i q).toRat|) d.reverse hl (base d i t)]
    rw [total_sum hd hT hi ht hl hen _ (fun q hq => by simp [X_toRat_out hq])]
    unfold absSum
    refine Finset.sum_congr rfl (fun q hq => ?_)
    have hin : loOf d i ≤ q ∧ q < hiOf d i := by
      unfold rng at hq
      have := g.p_lo; have := g.p_hi
      split at hq
      · rw [Finset.mem_Ico] at hq; exact ⟨by omega, hq.2⟩
      · rw [Finset.mem_Icc] at hq; exact ⟨hq.1, by omega⟩
    rw [X_in hin]
  · -- the depth
    simp only [Tree.depth, depth_mapLeaves]
    unfold depth carryDepth
    rw [g.lane_eq, g.before_eq]
    have hz : zD d t = if t = 0 then 0 else Nat.clog 2 d.BT + t := rfl
    rw [hz] at hD
    rw [hlenL] at hnodes
    split at hnodes <;> split at hD <;> simp_all <;> omega
  · -- the depth is at most `2^23`
    unfold depth carryDepth
    rw [g.lane_eq, g.before_eq]
    have hbt20 : d.BT ≤ 2 ^ 20 := hd.bt_le
    split <;> split <;> omega
  · -- the node count
    simp only [Tree.nodes, nodes_mapLeaves]
    rw [hlenL] at hnodes
    have h1 : tl.nodes ≤ d.BT - 1 := by split at hnodes <;> omega
    have h2 : t * d.BT + d.BT ≤ nbOf d i * d.BT := by
      have : (t + 1) * d.BT ≤ nbOf d i * d.BT := Nat.mul_le_mul_right _ ht
      have e : (t + 1) * d.BT = t * d.BT + d.BT := by ring
      omega
    have hbt20 : d.BT ≤ 2 ^ 20 := hd.bt_le
    omega

end Claims.ChunkGlobalCumsum
