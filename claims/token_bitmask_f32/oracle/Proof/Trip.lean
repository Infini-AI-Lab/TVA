import Proof.Tail

/-!
# `token_bitmask_f32`: one trip of the persistent loop

Trip `t` of program `p` serves work item `w = p + t * NUM_SMS`: row `w / nb`, block `w % nb`. It
computes the row and the block offset, reads `indices[k]` when the host passes indices, and runs
the block (`Proof/Tail.lean`).
-/

namespace Claims.TokenBitmask

open TVA

/-- **One trip is safe**: item `p + t * NUM_SMS` (below `items`) writes its cells and extends the
invariant by one trip. -/
theorem safe_trip {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {p t : Nat} {nbv w : I32} (hnb : nbv.toInt = (nb d : Int))
    (hw : w.toInt = ((p + t * d.NUM_SMS : Nat) : Int)) (hlt : p + t * d.NUM_SMS < items d)
    {m : Mem} (hI : Inv d init p t m) :
    Safe E (R d) (d.logits.elems (itemElems d init (p + t * d.NUM_SMS)))
      (fun _ m' => Inv d init p (t + 1) m') m (tripProg d nbv w) := by
  have hitems : items d = d.num_rows * nb d := rfl
  have hnbpos : 0 < nb d := by
    rcases Nat.eq_zero_or_pos (nb d) with h | h
    · rw [hitems, h, Nat.mul_zero] at hlt; omega
    · exact h
  have hk : (p + t * d.NUM_SMS) / nb d < d.num_rows :=
    Nat.div_lt_of_lt_mul (by rw [hitems, Nat.mul_comm d.num_rows] at hlt; exact hlt)
  have hj : (p + t * d.NUM_SMS) % nb d < nb d := Nat.mod_lt _ hnbpos
  have hitem : (p + t * d.NUM_SMS) / nb d * nb d + (p + t * d.NUM_SMS) % nb d =
      p + t * d.NUM_SMS := Nat.div_add_mod' _ _
  have hV := hd.vocab_lt
  have hj4096 : (p + t * d.NUM_SMS) % nb d * 4096 + 4096 ≤ d.vocab_size + 4095 := by
    have h1 : ((p + t * d.NUM_SMS) % nb d + 1) * 4096 ≤ nb d * 4096 :=
      Nat.mul_le_mul_right 4096 (Nat.succ_le_of_lt hj)
    have h2 : nb d * 4096 ≤ d.vocab_size + 4095 := Nat.div_mul_le_self _ _
    calc (p + t * d.NUM_SMS) % nb d * 4096 + 4096 = ((p + t * d.NUM_SMS) % nb d + 1) * 4096 := by
          ring
      _ ≤ _ := h1.trans h2
  simp only [tripProg]
  -- line 106: `row_id`
  refine safe_idiv1 (x := w) (y := nbv) (by rw [hw]; positivity)
    (by rw [hnb]; exact_mod_cast hnbpos) fun rq hrq => ?_
  have hrow : rq.item.toInt = (((p + t * d.NUM_SMS) / nb d : Nat) : Int) := by
    rw [hrq, hw, hnb]; norm_cast
  -- line 107: `block_offset`
  refine safe_irem1 (x := w) (y := nbv) (by rw [hw]; positivity)
    (by rw [hnb]; exact_mod_cast hnbpos) fun mq hmq => ?_
  have hmod : mq.item.toInt = (((p + t * d.NUM_SMS) % nb d : Nat) : Int) := by
    rw [hmq, hw, hnb]; norm_cast
  have hbo : (mq.item * (IntV.ofNat 4096 : I32)).toInt =
      (((p + t * d.NUM_SMS) % nb d * 4096 : Nat) : Int) :=
    IntV.I32.mul_nat hmod (IntV.I32.toInt_ofNat (by norm_num)) (by omega)
  -- line 108: `batch_id`
  cases hI' : d.hasIdx
  · simp only [Bool.false_eq_true, ↓reduceIte]
    have hr : (((p + t * d.NUM_SMS) / nb d : Nat) : Int) =
        rowOf d init ((p + t * d.NUM_SMS) / nb d) := by
      simp only [rowOf, hI', Bool.false_eq_true, ↓reduceIte]
    exact safe_tail hd hpre E hk hj hr hitem hbo hrow hI
  · simp only [↓reduceIte]
    have hkl : (p + t * d.NUM_SMS) / nb d < d.indices.len := by
      have := hd.idx_len hI'; omega
    have hk31 : (p + t * d.NUM_SMS) / nb d < 2 ^ 31 := by have := hd.rows_lt; omega
    obtain ⟨v, hv⟩ := Buf.read_some d.indices (hpre.2.2.1 hI') hkl
    have hptr : (d.indices.ptr 0 + rq.item).addr =
        (d.indices.ptr (((p + t * d.NUM_SMS) / nb d : Nat) : Int)).addr := by
      simp only [Ptr.add_int, Buf.ptr_offset, zero_add, hrow]
    have hspan := Buf.span_sub d.indices hkl
    have hag : ∀ z ∈ d.indices.bytes, m z = init z :=
      fun z hz => hI.2.1 z (fun hz' => Set.disjoint_left.mp (hd.disj_idx hI') hz' hz)
    have hvm : m.read .i32 (d.indices.ptr 0 + rq.item).addr = some v := by
      rw [hptr, Mem.read_congr fun x hx => hag x (hspan hx)]
      exact hv
    refine safe_load1 hvm ?_ ?_
    · rw [hptr]
      intro z hz
      simp only [R, hI', ↓reduceIte]
      exact Or.inr (hspan hz)
    · have hval : v.toInt = rowOf d init ((p + t * d.NUM_SMS) / nb d) := by
        simp only [rowOf, hI', ↓reduceIte, idxVal, hv]
      obtain ⟨h0, -, -⟩ := rowOf_bounds hd hpre hk
      exact safe_tail hd hpre E (r := (rowOf d init ((p + t * d.NUM_SMS) / nb d)).toNat) hk hj
        (Int.toNat_of_nonneg h0) hitem hbo (by rw [hval, Int.toNat_of_nonneg h0]) hI

end Claims.TokenBitmask
