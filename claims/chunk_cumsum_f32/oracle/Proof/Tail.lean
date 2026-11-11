import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.GeoBound
import Proof.GeoSum
import Proof.LaneArith
import Proof.NumScan
import Proof.NumArith
import Proof.Util

/-!
# After the scalars: offsets, mask, load, scan, scale, store

`tail_safe` walks the body from line 108 on: the lane offsets and mask, the masked load, the
`tl.cumsum` (each lane's value is a tree over its chunk's leaves, bounded by `tol_of_scan`), the
scale, and the store; the program ends with `Q`.
-/

namespace Claims.ChunkCumsum

open TVA

/-! ## The converted arange -/

/-- `tl.arange` converted to `int64` keeps its values. -/
theorem cvt_arange_toInt {BT : Nat} (hBT : BT ≤ 2 ^ 20) {P : Platform}
    {c : Tensor [BT] I64}
    (hc : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 BT).get i) (c.get i)) :
    ∀ i, (c.get i).toInt = (i.1.val : Int) := by
  intro i
  have h1 : c.get i = IntV.ofInt ((tl.arange 0 BT).get i).toInt := hc i
  rw [h1]
  have h2 : ((tl.arange 0 BT).get i).toInt = (i.1.val : Int) := by
    show (IntV.ofInt (0 + (i.1.val : Int)) : I32).toInt = (i.1.val : Int)
    rw [IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; constructor <;> omega)]
    omega
  rw [h2]
  rw [IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; constructor <;> omega)]

/-! ## Trees and their leaves -/

/-- A tree evaluates to the same result when the leaves it uses are unchanged. -/
theorem eval_congr {α : Type} {op : α → α → α → Prop} {x y : Nat → α} {t : Tree} {r : α}
    (h : t.Eval op x r) : (∀ j ∈ t.leaves, x j = y j) → t.Eval op y r := by
  induction h with
  | leaf i =>
      intro hy
      rw [hy i (by simp [Tree.leaves])]
      exact .leaf i
  | node hl hr hop ihl ihr =>
      intro hy
      exact .node
        (ihl (fun j hj => hy j (by
          simp only [Tree.leaves, List.mem_append]; exact Or.inl hj)))
        (ihr (fun j hj => hy j (by
          simp only [Tree.leaves, List.mem_append]; exact Or.inr hj)))
        hop

/-- A scan's leaves are lanes below `BT`. -/
theorem leavesOf_lt {rev : Bool} {BT l j : Nat} (hl : l < BT) (hj : j ∈ leavesOf rev BT l) :
    j < BT := by
  unfold leavesOf at hj
  by_cases hr : rev
  · rw [ite_eq_left hr] at hj
    rw [List.mem_range'_1] at hj
    omega
  · rw [ite_eq_right hr] at hj
    rw [List.mem_range] at hj
    omega

/-! ## The lane values -/

/-- The `int64` constant `H` has value `H`. -/
theorem ofNatH_toInt {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.H : I64).toInt = (d.H : Int) :=
  IntV.I64.toInt_ofNat (by have := H_lt_two_31 hd; omega)

/-- The lane offsets `i_t·BT + arange` have value `chunk·BT + l`. -/
theorem o_t_get_toInt {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {i : Nat} (hi : i < gridSize d) {it : I64} (hit : it.toInt = (chkOf d i : Int))
    {c : Tensor [d.BT] I64} (hc : ∀ j, (c.get j).toInt = (j.1.val : Int)) (j : Idx [d.BT]) :
    ((bc (it * (IntV.ofNat d.BT : I64)) + c).get j).toInt
      = ((chkOf d i * d.BT + j.1.val : Nat) : Int) := by
  have hb : chkOf d i * d.BT + j.1.val < 2 ^ 63 := by
    have := chkBT_lt hd hT hi j.1.isLt
    omega
  have hmul : (it * (IntV.ofNat d.BT : I64)).toInt = ((chkOf d i * d.BT : Nat) : Int) :=
    IntV.I64.mul_nat hit (IntV.I64.toInt_ofNat (bt_lt_two_63 hd)) (by omega)
  rw [Tensor.get_add, Tensor.get_full]
  exact IntV.I64.add_nat hmul (hc j) hb

/-- The mask `o_t < T` is the enabled-lane predicate. -/
theorem ltS_get_eq {d : Inst} {i : Nat} {TT : I64} (hTT : TT.toInt = (lenOf d i : Int))
    {o : Tensor [d.BT] I64}
    (ho : ∀ j, (o.get j).toInt = ((chkOf d i * d.BT + j.1.val : Nat) : Int)) (j : Idx [d.BT]) :
    (o.ltS TT).get j = enF d i j := by
  rw [Tensor.get_ltS, IntV.lt_eq, hTT, ho j]
  show decide (((chkOf d i * d.BT + j.1.val : Nat) : Int) < ((lenOf d i : Nat) : Int)) = _
  unfold enF en
  rw [decide_eq_decide]
  constructor <;> omega

/-- The pointer tensor addresses element `elem` of lane `l`. -/
theorem p_get_eq {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < gridSize d) {bos ih : I64}
    (hbos : bos.toInt = (baseOf d i : Int)) (hih : ih.toInt = (headOf d i : Int))
    {o : Tensor [d.BT] I64}
    (ho : ∀ j, (o.get j).toInt = ((chkOf d i * d.BT + j.1.val : Nat) : Int))
    (b : Buf .f32) (j : Idx [d.BT]) :
    (b.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih + o * (IntV.ofNat d.H : I64)).get j
      = b.ptr (elem d i j.1.val : Int) := by
  have hH : (IntV.ofNat d.H : I64).toInt = (d.H : Int) := ofNatH_toInt hd
  have hbosH : (bos * (IntV.ofNat d.H : I64)).toInt = ((baseOf d i * d.H : Nat) : Int) :=
    IntV.I64.mul_nat hbos hH (baseOf_mul_H_lt hd hT hi hpre)
  have hoH : ((o * (IntV.ofNat d.H : I64)).get j).toInt
      = (((chkOf d i * d.BT + j.1.val) * d.H : Nat) : Int) := by
    rw [Tensor.get_muls]
    exact IntV.I64.mul_nat (ho j) hH (chkBT_mul_H_lt hd hT hi j.1.isLt)
  have key : (((b.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih
      + o * (IntV.ofNat d.H : I64)).get j).addr)
      = (b.ptr (elem d i j.1.val : Int)).addr := by
    simp only [Tensor.get_ptrAdd, Ptr.add_int, Ptr.offset_addr, Buf.ptr_addr]
    rw [hbosH, hih, hoH]
    unfold elem pos
    push_cast
    ring
  calc (b.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih + o * (IntV.ofNat d.H : I64)).get j
      = ⟨((b.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih
          + o * (IntV.ofNat d.H : I64)).get j).addr⟩ := rfl
    _ = ⟨(b.ptr (elem d i j.1.val : Int)).addr⟩ := by rw [key]
    _ = b.ptr (elem d i j.1.val : Int) := rfl

/-! ## The loaded lane values are the scan leaves -/

/-- The masked load gives the scan's leaves: the gate values at enabled lanes, `+0` elsewhere. -/
theorem leaf_val {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < gridSize d)
    {v : Tensor [d.BT] F32}
    (hread : ∀ j, enF d i j = true →
      init.read .f32 (d.s.ptr (elemF d i j : Int)).addr = some (v.get j))
    (hdis : ∀ j, enF d i j = false → ∀ o : Tensor [d.BT] F32,
      some (bc (FloatV.litF32 (FloatV.ofPat 0))) = some o → v.get j = o.get j)
    {j : Nat} (hj : j < d.BT) :
    v.get (⟨j, hj⟩, ()) = sLeaf d i (headOf d i) j := by
  have hsHold : d.s.Holds init d.sv := hpre.2.2.1
  unfold sLeaf
  by_cases he : en d i j = true
  · rw [ite_eq_left he]
    have h1 : init.read .f32 (d.s.ptr (elem d i j)).addr
        = some (v.get (⟨j, hj⟩, ())) := hread (⟨j, hj⟩, ()) he
    have hin : elem d i j < d.s.len := by
      have h2 : elem d i j < d.B * d.T * d.H := elem_lt hd hT (nOf_lt hd hT hi) he
      have h3 := hd.s_len
      omega
    have h4 : init.read .f32 (d.s.ptr (elem d i j)).addr
        = some (d.sv (elem d i j)) := hsHold (elem d i j) hin
    have h5 : v.get (⟨j, hj⟩, ()) = d.sv (elem d i j) := by
      have h6 : some (v.get (⟨j, hj⟩, ())) = some (d.sv (elem d i j)) := by
        rw [← h1, h4]
      exact Option.some.inj h6
    show v.get (⟨j, hj⟩, ()) = sAt d (pos d i j) (headOf d i)
    rw [h5]
    rfl
  · rw [ite_eq_right he]
    have hef : en d i j = false := by cases h : en d i j <;> simp_all
    have h1 := hdis (⟨j, hj⟩, ()) hef (bc (FloatV.litF32 (FloatV.ofPat 0))) rfl
    rw [h1]
    simp only [bc, litF32_zero]
    rfl

/-! ## The tail -/

/-- **The rest of the body** after the scalars are computed (offsets, mask, load, scan, scale, store) is safe and ends with `Q`. -/
theorem tail_safe {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (hT : d.varlen = true → TablesOk d) {i : Nat} (hi : i < gridSize d) {it bos TT ih : I64}
    (hit : it.toInt = (chkOf d i : Int)) (hbos : bos.toInt = (baseOf d i : Int))
    (hTT : TT.toInt = (lenOf d i : Int)) (hih : ih.toInt = (headOf d i : Int)) :
    Safe E (R d i) (W d i) (fun _ m => Q d i m) init
      (do
        let o_t := bc (it * (IntV.ofNat d.BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.BT))
        let m_t := o_t.ltS TT
        let p_s := d.s.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih + o_t * (IntV.ofNat d.H : I64)
        let p_o := d.o.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih + o_t * (IntV.ofNat d.H : I64)
        let b_s ← tl.load p_s (some m_t) (some (bc (FloatV.litF32 (FloatV.ofPat 0))))
        let b_o ← tl.cumsum b_s 0 d.reverse
        let b_o ← if d.hasScale then tl.mul b_o (bc (sigma32 d)) else pure b_o
        tl.store p_o b_o (some m_t)) := by
  have hsv : d.s.Valid init := hpre.1
  have hov : d.o.Valid init := hpre.2.1
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j
    exact ⟨IntV.ofInt ((tl.arange 0 d.BT).get j).toInt, rfl⟩
  · intro c hc
    dsimp only
    set o_t : Tensor [d.BT] I64 := bc (it * (IntV.ofNat d.BT : I64)) + c with ho_t
    set m_t : Tensor [d.BT] Bool := o_t.ltS TT with hm_t
    set p_s : Tensor [d.BT] (Ptr .f32) :=
      d.s.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih + o_t * (IntV.ofNat d.H : I64) with hp_s
    set p_o : Tensor [d.BT] (Ptr .f32) :=
      d.o.ptr 0 + bos * (IntV.ofNat d.H : I64) + ih + o_t * (IntV.ofNat d.H : I64) with hp_o
    have hc' : ∀ j, (c.get j).toInt = (j.1.val : Int) :=
      cvt_arange_toInt (BT := d.BT) (by have := hd.bt_le; omega) hc
    have ho : ∀ j, (o_t.get j).toInt = ((chkOf d i * d.BT + j.1.val : Nat) : Int) := by
      intro j
      rw [ho_t]
      exact o_t_get_toInt hd hT hi hit hc' j
    have hmask : ∀ j, m_t.get j = enF d i j := by
      intro j
      rw [hm_t]
      exact ltS_get_eq hTT ho j
    have hp_s' : ∀ j, enF d i j = true → p_s.get j = d.s.ptr (elemF d i j : Int) := by
      intro j _
      rw [hp_s]
      exact p_get_eq hd hT hpre hi hbos hih ho d.s j
    have hp_o' : ∀ j, enF d i j = true → p_o.get j = d.o.ptr (elemF d i j : Int) := by
      intro j _
      rw [hp_o]
      exact p_get_eq hd hT hpre hi hbos hih ho d.o j
    have hin_s : ∀ j, enF d i j = true → elemF d i j < d.s.len := by
      intro j hj
      show elem d i j.1.val < d.s.len
      have h1 : elem d i j.1.val < d.B * d.T * d.H :=
        elem_lt hd hT (nOf_lt hd hT hi) (by simpa [enF] using hj)
      have h2 := hd.s_len
      omega
    have hin_o : ∀ j, enF d i j = true → elemF d i j < d.o.len := by
      intro j hj
      show elem d i j.1.val < d.o.len
      have h1 : elem d i j.1.val < d.B * d.T * d.H :=
        elem_lt hd hT (nOf_lt hd hT hi) (by simpa [enF] using hj)
      have h2 := hd.o_len
      omega
    have hstore : ∀ (vals : Tensor [d.BT] F32),
        (∀ (l : Nat) (hl : l < d.BT), en d i l = true →
          InRange d (nOf d i) (pos d i l) (headOf d i) →
          (vals.get (⟨l, hl⟩, ())).isFinite = true ∧
          |(vals.get (⟨l, hl⟩, ())).toRat
              - sig d * refSum d (nOf d i) (pos d i l) (headOf d i)|
            ≤ tol d (nOf d i) (pos d i l) (headOf d i)) →
        Safe E (R d i) (W d i) (fun _ m => Q d i m) init
          (tl.store p_o vals (some m_t)) := by
      intro vals hval
      refine safe_last (safe_store_buf (b := d.o) (ptrs := p_o) (mask := some m_t)
        (vals := vals) (g := elemF d i) (en := enF d i) ?_ hin_o ?_ hov ?_ ?_)
      · exact Buf.addrs_of_get d.o hmask hp_o'
      · intro x y _ _ hg
        have h1 : x.1 = y.1 := Fin.ext (elem_inj hd hg)
        exact Prod.ext h1 (by cases x.2; cases y.2; rfl)
      · intro x hx
        exact hx
      · intro m' hwrite _ _
        refine safe_ret ?_
        intro l hl hen
        exact ⟨vals.get (⟨l, hl⟩, ()), hwrite (⟨l, hl⟩, ()) hen, fun hinr => hval l hl hen hinr⟩
    refine safe_load_buf (m := init) (b := d.s) (ptrs := p_s) (mask := some m_t)
      (other := some (bc (FloatV.litF32 (FloatV.ofPat 0)))) (g := elemF d i) (en := enF d i)
      (Buf.addrs_of_get d.s hmask hp_s') hin_s hsv ?_ ?_
    · intro x hx
      exact Or.inl hx
    · intro v hread hdis
      refine safe_cumsum (x := v) (axis := 0) (reverse := d.reverse) (f := .binary32)
        (s := [d.BT]) (by simp) ?_
      intro r hscan
      have hleaf : ∀ (l : Nat) (hl : l < d.BT) (j : Nat) (hj : j ∈ leavesOf d.reverse d.BT l),
          v.get (Idx.set (⟨l, hl⟩, ()) 0 j) = sLeaf d i (headOf d i) j := by
        intro l hl j hj
        have hlt : j < d.BT := leavesOf_lt hl hj
        rw [get_set0 v hl hlt]
        exact leaf_val hd hT hpre hi hread hdis hlt
      have hA0 : ∀ l, 0 ≤ absSum d (nOf d i) (pos d i l) (headOf d i) := by
        intro l
        unfold absSum
        exact Finset.sum_nonneg (fun q _ => abs_nonneg _)
      by_cases hs : d.hasScale = true
      · rw [ite_eq_left hs]
        refine safe_mul (x := r) (y := bc (sigma32 d)) ?_
        intro vm hmul
        refine hstore vm ?_
        intro l hl hen hinr
        obtain ⟨t, hSC, hEval⟩ := hscan (⟨l, hl⟩, ())
        have hPerm : t.leaves.Perm (leavesOf d.reverse d.BT l) := ScanLeaves.perm hSC
        have hEval' : t.Eval (AddRel false) (fun j => sLeaf d i (headOf d i) j)
            (r.get (⟨l, hl⟩, ())) :=
          eval_congr hEval (fun j hj => hleaf l hl j (hPerm.subset hj))
        have hn : nOf d i < nSeq d := nOf_lt hd hT hi
        have hσf : (sigma32 d).isFinite = true := (hinr.2.2 hs).2.1
        have hσA : |(sigma32 d).toRat| * absSum d (nOf d i) (pos d i l) (headOf d i) ≤ 2 ^ 126 :=
          (hinr.2.2 hs).2.2
        have hσget : ((bc (sigma32 d) : Tensor [d.BT] F32).get (⟨l, hl⟩, ())) = sigma32 d := rfl
        have hvm : FloatV.NaNEq (FloatV.mul (r.get (⟨l, hl⟩, ())) (sigma32 d))
            (vm.get (⟨l, hl⟩, ())) := by
          rw [← hσget]
          exact hmul (⟨l, hl⟩, ())
        have hb := scan_scale_bound (t := t) (L := leavesOf d.reverse d.BT l)
          (x := fun j => sLeaf d i (headOf d i) j) (r := r.get (⟨l, hl⟩, ()))
          (v := vm.get (⟨l, hl⟩, ())) (σ := sigma32 d)
          (S := refSum d (nOf d i) (pos d i l) (headOf d i))
          (A := absSum d (nOf d i) (pos d i l) (headOf d i)) (sig := sig d)
          hPerm hEval' (fun j hj => leaves_finite hT hd hn hl hj hinr)
          (leaves_sum hT hd hn hl hen) (leaves_absSum hT hd hn hl hen)
          (hA0 l) hinr.2.1 (leaves_length_le hd d.reverse hl) hσf hσA hvm
        have hs32 : sig32 d = (sigma32 d).toRat := by rw [sig32, ite_eq_left hs]
        have hlen' : ((depth d (nOf d i) (pos d i l) : ℚ) + 1)
            = ((leavesOf d.reverse d.BT l).length : ℚ) := by
          rw [leaves_length hd hl]
          push_cast
          ring
        have htol : tol d (nOf d i) (pos d i l) (headOf d i)
            = ((leavesOf d.reverse d.BT l).length : ℚ) * 2 ^ (-23 : ℤ)
                * |(sigma32 d).toRat| * absSum d (nOf d i) (pos d i l) (headOf d i)
              + |sig d - (sigma32 d).toRat| * absSum d (nOf d i) (pos d i l) (headOf d i)
              + 2 ^ (-128 : ℤ) * (1 + |(sigma32 d).toRat|) := by
          unfold tol
          rw [hs32, hlen']
          ring
        exact ⟨hb.1, by rw [htol]; exact hb.2⟩
      · rw [ite_eq_right hs]
        show Safe E (R d i) (W d i) (fun _ m => Q d i m) init
          (tl.store p_o r (some m_t))
        refine hstore r ?_
        intro l hl hen hinr
        obtain ⟨t, hSC, hEval⟩ := hscan (⟨l, hl⟩, ())
        have hPerm : t.leaves.Perm (leavesOf d.reverse d.BT l) := ScanLeaves.perm hSC
        have hEval' : t.Eval (AddRel false) (fun j => sLeaf d i (headOf d i) j)
            (r.get (⟨l, hl⟩, ())) :=
          eval_congr hEval (fun j hj => hleaf l hl j (hPerm.subset hj))
        have hn : nOf d i < nSeq d := nOf_lt hd hT hi
        have hb := scan_bound (t := t) (L := leavesOf d.reverse d.BT l)
          (x := fun j => sLeaf d i (headOf d i) j) (r := r.get (⟨l, hl⟩, ()))
          (S := refSum d (nOf d i) (pos d i l) (headOf d i))
          (A := absSum d (nOf d i) (pos d i l) (headOf d i))
          hPerm hEval' (fun j hj => leaves_finite hT hd hn hl hj hinr)
          (leaves_sum hT hd hn hl hen) (leaves_absSum hT hd hn hl hen)
          (hA0 l) hinr.2.1 (leaves_length_le hd d.reverse hl)
        have hs' : sig d = 1 := by rw [sig, ite_eq_right hs]
        have hs32 : sig32 d = 1 := by rw [sig32, ite_eq_right hs]
        have hlen' : ((depth d (nOf d i) (pos d i l) : ℚ) + 1)
            = ((leavesOf d.reverse d.BT l).length : ℚ) := by
          rw [leaves_length hd hl]
          push_cast
          ring
        have htol : tol d (nOf d i) (pos d i l) (headOf d i)
            = ((leavesOf d.reverse d.BT l).length : ℚ) * 2 ^ (-23 : ℤ)
                * absSum d (nOf d i) (pos d i l) (headOf d i) + 2 ^ (-128 : ℤ) * 2 := by
          unfold tol
          rw [hs', hs32, hlen']
          norm_num
        refine ⟨hb.1, ?_⟩
        rw [hs', one_mul, htol]
        have h2 : (0 : ℚ) < 2 ^ (-128 : ℤ) := by norm_num
        linarith [hb.2]

end Claims.ChunkCumsum
