import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Num
import Proof.Trees
import Proof.Lane
import Proof.LaneGood
import Proof.TripLanes
import Proof.Trip

/-!
# The walk of one trip

`safe_trip`: trip `t` of the loop is safe from `Inv t`, writes `Wt t`, and ends in `Inv (t + 1)`.
-/

namespace Claims.ChunkGlobalCumsum

open TVA

variable {d : Inst} {i : Nat}

/-- Re-associate two binds. -/
theorem safe_bind_assoc {E : Env} {R W : Set Int} {α β γ : Type} {Q : γ → Mem → Prop} {m : Mem}
    {p : Prog α} {f : α → Prog β} {g : β → Prog γ}
    (h : Safe E R W Q m (p >>= fun a => f a >>= g)) : Safe E R W Q m ((p >>= f) >>= g) := by
  simp only [Prog.bind_eq] at h ⊢
  rw [Prog.bind_assoc]; exact h

/-- A `pure` followed by the rest is the rest. -/
theorem safe_pure_bind {E : Env} {R W : Set Int} {α β : Type} {Q : β → Mem → Prop} {m : Mem}
    {a : α} {k : α → Prog β} (h : Safe E R W Q m (k a)) : Safe E R W Q m (pure a >>= k) := h

/-- **One trip** of the loop. -/
theorem safe_trip (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform} (hP : Device P)
    (hT : d.varlen = true → TablesOk d) (hi : i < gridSize d)
    {NT TT bos ih : I64} (hNT : NT.toInt = (nbOf d i : Int)) (hTT : TT.toInt = (lenOf d i : Int))
    (hbos : bos.toInt = (loOf d i : Int)) (hih : ih.toInt = (headOf d i : Int))
    {t : Nat} (ht : t < nbOf d i) {ic : I64} (hic : ic.toInt = (t : Int)) {z : F32} {m : Mem}
    (hI : Inv d init i t z m) :
    Safe ((launch d).env P) (R d) (Wt d i t) (fun z' m' => Inv d init i (t + 1) z' m') m
      (tripBody d NT TT bos ih ic z) := by
  obtain ⟨hfr, hpres, hz, hdone⟩ := hI
  have hsv : d.s.Valid m :=
    Buf.valid_of_agree hpre.1 (fun x hx => hpres x (hpre.1.2.2.2 x hx))
  have hit := it_toInt hd hNT hic ht hT hi
  have hb := bt_pos hd
  unfold tripBody
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j; exact ⟨IntV.ofInt ((tl.arange 0 d.BT).get j).toInt, rfl⟩
  intro c hc
  have hc' : ∀ j, (c.get j).toInt = (j.1.val : Int) :=
    cvt_arange_toInt (BT := d.BT) hd.bt_le hc
  have ho := o_t_toInt hd hT hi ht hit hc'
  have hmask := mask_eq (TT := TT) hTT ho
  have hps := ptr_eq hd hT hi ht hbos hih ho d.s
  have hpo := ptr_eq hd hT hi ht hbos hih ho d.o
  have hin_s : ∀ j, enE d i t j = true → gE d i t j < d.s.len := by
    intro j hj
    have hen : bt d i t * d.BT + j.1.val < lenOf d i := by unfold enE at hj; simpa using hj
    have h1 := elem_lt hd hT hi (lane_hi hT hi hen)
    have h2 := hd.s_len
    unfold gE; omega
  refine safe_load_buf (m := m) d.s (Buf.addrs_of_get d.s hmask (fun j _ => hps j)) hin_s hsv
    (fun x hx => Or.inl (Buf.elems_sub_bytes d.s
      (fun e he => by obtain ⟨j, hj, rfl⟩ := he; exact hin_s j hj) hx)) ?_
  intro v hread hdis
  have hleaf := fun {j : Nat} (hj : j < d.BT) => leaf_X hd hpre hT hi hfr hread hdis hj
  refine safe_cumsum (x := v) (axis := 0) (reverse := d.reverse) (f := .binary32)
    (s := [d.BT]) (by simp) ?_
  intro r hscan
  refine safe_sum (E := (launch d).env P) hb ?_
  rintro b_ss ⟨ts, hcov, hdep, hev⟩
  -- the block's `tl.sum`, renumbered to positions
  have hev' : ts.Eval (AddRel false) (fun j => X d i (j + base d i t)) b_ss := by
    refine eval_congr hev (fun j hj => ?_)
    have hjl : j < d.BT := List.mem_range.mp (hcov.subset hj)
    simp only [hjl, dite_true]
    exact hleaf hjl
  have hdep' : ts.depth ≤ Nat.clog 2 d.BT := by
    have h1 : tl.sumBound ((launch d).env P).P d.BT = max (P.sumDepth d.BT) (Nat.clog 2 d.BT) := rfl
    have h2 := hP.sumDepth d.BT
    omega
  refine safe_add (x := r) (y := bc z) ?_
  intro ra hra
  have hle : IntV.le (IntV.ofInt 0 : I64) ic = true := by
    rw [IntV.le_eq, hic, IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; norm_num)]
    simp
  rw [ite_eq_left hle]
  refine safe_add (x := Tensor.scalar z) (y := Tensor.scalar b_ss) ?_
  intro zz hzz
  have hz' : ZTree d i (t + 1) zz.item :=
    ztree_step ht hz hcov hdep' hev' (Or.inl (hzz ()))
  refine safe_pure_bind ?_
  -- each lane's output before the scale
  have hlane : ∀ l (hl : l < d.BT), bt d i t * d.BT + l < lenOf d i → ∀ va : F32,
      (if d.hasScale then FloatV.NaNEq (FloatV.mul (ra.get (⟨l, hl⟩, ())) (sigma32 d)) va
        else va = ra.get (⟨l, hl⟩, ())) →
      Good d (seqOf d i) (base d i t + l) (headOf d i) va := by
    intro l hl hen va hva
    obtain ⟨tl', hSC, hEval⟩ := hscan (⟨l, hl⟩, ())
    have hperm : tl'.leaves.Perm (scanL d.reverse d.BT l) := ScanLeaves.perm hSC
    have hEval' : tl'.Eval (AddRel false) (fun j => X d i (j + base d i t)) (r.get (⟨l, hl⟩, ())) := by
      refine eval_congr hEval (fun j hj => ?_)
      have hjl : j < d.BT := by
        have := hperm.subset hj
        unfold scanL at this
        split at this
        · rw [List.mem_range'_1] at this; omega
        · rw [List.mem_range] at this; omega
      rw [get_set0 v hl hjl]
      exact hleaf hjl
    exact lane_good hd hT hi ht hl hen hz hperm hEval' (Or.inl (hra (⟨l, hl⟩, ()))) hva
  dsimp only
  by_cases hs : d.hasScale = true
  · rw [ite_eq_left hs]
    refine safe_mul (x := ra) (y := bc (sigma32 d)) ?_
    intro vm hvm
    exact store_step hd hpre hT hi ht hfr hpres hdone hz' hmask (fun j _ => hpo j)
      (fun l hl hen => hlane l hl hen _ (by rw [ite_eq_left hs]; exact hvm (⟨l, hl⟩, ())))
  · rw [ite_eq_right hs]
    refine safe_pure_bind ?_
    exact store_step hd hpre hT hi ht hfr hpres hdone hz' hmask (fun j _ => hpo j)
      (fun l hl hen => hlane l hl hen _ (by rw [ite_eq_right hs]))

end Claims.ChunkGlobalCumsum
