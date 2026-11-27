import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Num
import Proof.Trees

/-!
# The running sum and one output lane, as trees

`ZTree d i t z`: after `t` trips the running sum `z` is a tree of IEEE additions over the
positions `Pset d i t` of the blocks done (and the `+0` it starts from, at position `hi`), of
depth at most `zD t` and at most `t·BT` nodes. `ztree_zero` starts it, `ztree_step` adds a
block's `tl.sum`; `lane_good` joins a lane's scan tree with it into the output's tree and gives
`Good` (`good_of_tree`).
-/

namespace Claims.ChunkGlobalCumsum

open TVA

/-- The running sum after `t` trips, as a tree over the positions done. -/
def ZTree (d : Inst) (i t : Nat) (z : F32) : Prop :=
  ∃ tz : Tree, tz.Eval (AddRel false) (X d i) z ∧ tz.depth ≤ zD d t ∧ tz.nodes ≤ t * d.BT ∧
    (∀ q ∈ tz.leaves, q ∈ Pset d i t ∨ q = hiOf d i) ∧
    Tree.sumOf FloatV.toRat (X d i) tz = ∑ q ∈ Pset d i t, (X d i q).toRat ∧
    Tree.absSumOf FloatV.toRat (X d i) tz = ∑ q ∈ Pset d i t, |(X d i q).toRat|

/-- The first position of trip `t`'s block. -/
def base (d : Inst) (i t : Nat) : Nat := loOf d i + bt d i t * d.BT

variable {d : Inst} {i : Nat}

/-- Outside the sequence the leaf value is `+0`. -/
theorem X_out {q : Nat} (h : ¬ (loOf d i ≤ q ∧ q < hiOf d i)) : X d i q = FloatV.ofPat 0 := by
  unfold X; rw [ite_eq_right h]

theorem X_in {q : Nat} (h : loOf d i ≤ q ∧ q < hiOf d i) : X d i q = sAt d q (headOf d i) := by
  unfold X; rw [ite_eq_left h]

theorem X_hi : X d i (hiOf d i) = FloatV.ofPat 0 := X_out (by omega)

/-- **Before the loop** the running sum is the `+0` leaf at `hi`. -/
theorem ztree_zero : ZTree d i 0 (FloatV.ofPat 0) := by
  refine ⟨.leaf (hiOf d i), ?_, by simp [Tree.depth, zD], by simp [Tree.nodes], ?_, ?_, ?_⟩
  · rw [← X_hi (d := d) (i := i)]; exact .leaf _
  · intro q hq; simp [Tree.leaves] at hq; exact Or.inr hq
  · simp only [Tree.sumOf_leaf, X_hi, ofPat_zero_toRat, Pset]
    split <;> simp
  · simp only [Tree.absSumOf_leaf, X_hi, ofPat_zero_toRat, Pset]
    split <;> simp

/-- The positions done after `t + 1` trips: those after `t`, and trip `t`'s block. -/
theorem pset_succ (f : Nat → ℚ) {t : Nat} (ht : t < nbOf d i) :
    ∑ q ∈ Pset d i (t + 1), f q =
      ∑ q ∈ Pset d i t, f q + ∑ q ∈ Finset.Ico (base d i t) (base d i t + d.BT), f q := by
  unfold Pset base bt
  cases hr : d.reverse
  · simp only [Bool.false_eq_true, ite_false]
    have e : loOf d i + (t + 1) * d.BT = loOf d i + t * d.BT + d.BT := by ring
    rw [e, Finset.sum_Ico_consecutive f (by omega) (by omega)]
  · simp only [ite_true]
    have e1 : nbOf d i - (t + 1) = nbOf d i - 1 - t := by omega
    have e2 : loOf d i + (nbOf d i - 1 - t) * d.BT + d.BT = loOf d i + (nbOf d i - t) * d.BT := by
      have : nbOf d i - t = (nbOf d i - 1 - t) + 1 := by omega
      rw [this]; ring
    have e3 : (nbOf d i - t) * d.BT ≤ nbOf d i * d.BT := Nat.mul_le_mul_right _ (by omega)
    have e4 : (nbOf d i - 1 - t) * d.BT ≤ (nbOf d i - t) * d.BT := Nat.mul_le_mul_right _ (by omega)
    rw [e1, e2, add_comm (∑ q ∈ Finset.Ico (loOf d i + (nbOf d i - t) * d.BT)
      (loOf d i + nbOf d i * d.BT), f q)]
    rw [Finset.sum_Ico_consecutive f (by omega) (by omega)]

/-- The positions done after `t` trips are done after `t + 1`. -/
theorem pset_mono {t q : Nat} (ht : t < nbOf d i) (hq : q ∈ Pset d i t) : q ∈ Pset d i (t + 1) := by
  unfold Pset at *
  split at hq
  · rename_i hr
    simp only [hr, ite_true, Finset.mem_Ico] at hq ⊢
    have : (nbOf d i - (t + 1)) * d.BT ≤ (nbOf d i - t) * d.BT :=
      Nat.mul_le_mul_right _ (by omega)
    omega
  · rename_i hr
    simp only [hr, Bool.false_eq_true, ite_false, Finset.mem_Ico] at hq ⊢
    have : t * d.BT ≤ (t + 1) * d.BT := Nat.mul_le_mul_right _ (by omega)
    omega

/-- Trip `t`'s block is done after `t + 1` trips. -/
theorem block_mem_pset {t j : Nat} (ht : t < nbOf d i) (hj : j < d.BT) :
    j + base d i t ∈ Pset d i (t + 1) := by
  unfold Pset base bt
  split
  · rename_i hr
    simp only [Finset.mem_Ico]
    have e1 : nbOf d i - (t + 1) = nbOf d i - 1 - t := by omega
    have e2 : (nbOf d i - 1 - t) * d.BT + d.BT ≤ nbOf d i * d.BT := by
      have : (nbOf d i - 1 - t + 1) * d.BT ≤ nbOf d i * d.BT := Nat.mul_le_mul_right _ (by omega)
      have e : (nbOf d i - 1 - t + 1) * d.BT = (nbOf d i - 1 - t) * d.BT + d.BT := by ring
      omega
    rw [e1]
    omega
  · simp only [Finset.mem_Ico]
    have : (t + 1) * d.BT = t * d.BT + d.BT := by ring
    omega

/-- **One trip** adds its block's `tl.sum` (a tree over the block's lanes, of depth at most
`⌈log₂ BT⌉`) to the running sum. -/
theorem ztree_step {t : Nat} (ht : t < nbOf d i) {z b z' : F32} (hz : ZTree d i t z)
    {ts : Tree} (hcov : ts.Covers d.BT) (hdep : ts.depth ≤ Nat.clog 2 d.BT)
    (hev : ts.Eval (AddRel false) (fun j => X d i (j + base d i t)) b)
    (hadd : AddRel false z b z') : ZTree d i (t + 1) z' := by
  obtain ⟨tz, hE, hD, hN, hL, hS, hA⟩ := hz
  have hmap := eval_mapLeaves (g := fun j => j + base d i t) hev
  have hcovN := Tree.nodes_of_covers hcov
  have hleaf : ∀ j ∈ ts.leaves, j < d.BT := fun j hj => List.mem_range.mp (hcov.subset hj)
  refine ⟨.node tz (ts.mapLeaves (fun j => j + base d i t)), .node hE hmap hadd, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [Tree.depth, depth_mapLeaves, zD] at hD ⊢
    split at hD <;> simp <;> omega
  · simp only [Tree.nodes, nodes_mapLeaves]
    have : (t + 1) * d.BT = t * d.BT + d.BT := by ring
    omega
  · intro q hq
    simp only [Tree.leaves, List.mem_append, Tree.leaves_mapLeaves, List.mem_map] at hq
    rcases hq with hq | ⟨j, hj, rfl⟩
    · rcases hL q hq with h | h
      · exact Or.inl (pset_mono ht h)
      · exact Or.inr h
    · exact Or.inl (block_mem_pset ht (hleaf j hj))
  · rw [Tree.sumOf_node, hS, pset_succ _ ht]
    congr 1
    rw [sumOf_perm FloatV.toRat (X d i) (L := (List.range d.BT).map (fun j => j + base d i t))
      (by rw [Tree.leaves_mapLeaves]; exact hcov.map _)]
    rw [List.map_map, list_range_sum]
    exact (range_shift_sum (fun q => (X d i q).toRat) d.BT (base d i t)).trans (by rw [Nat.add_comm])
  · rw [Tree.absSumOf_node, hA, pset_succ _ ht]
    congr 1
    rw [absSumOf_perm FloatV.toRat (X d i) (L := (List.range d.BT).map (fun j => j + base d i t))
      (by rw [Tree.leaves_mapLeaves]; exact hcov.map _)]
    rw [List.map_map, list_range_sum]
    exact (range_shift_sum (fun q => |(X d i q).toRat|) d.BT (base d i t)).trans (by rw [Nat.add_comm])

end Claims.ChunkGlobalCumsum
