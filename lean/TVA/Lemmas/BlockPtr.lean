import TVA.Lang.BlockPtr
import TVA.Lemmas.Access
import TVA.Lemmas.Int

/-!
# Proof rules for block pointers

A block pointer over a buffer `b` (`p.base = b.ptr 0`) addresses, on each lane that passes the
boundary check, element `p.elem i` of `b` (`BlockPtr.addrs_buf`). So the `Buf.Addrs` kit
applies:

* `safe_load_block_ptr`: lanes that pass the check get their element, the others the padding;
* `safe_store_block_ptr`: lanes that pass the check are written (to distinct elements), nothing
  else changes.

`p.elem` is computed in wrapping `int64` arithmetic, as Triton does. `elem_eq` turns it into
`Σₖ (offsets[k] + i_k) · strides[k]` when nothing overflows, and `inb_iff` reads the check.
`advance_offsets` gives the offsets after `tl.advance`.
-/
namespace TVA

namespace BlockPtr

variable {d : DType}

theorem inb_iff (p : BlockPtr d) (bc : List Nat) (i : Idx p.block) :
    p.inb bc i = true ↔
      ∀ k ∈ bc, 0 ≤ (p.coord i k).toInt ∧ (p.coord i k).toInt < (p.shape.getD k 0).toInt := by
  simp [inb, List.all_eq_true]

/-- Without overflow, a coordinate is `offsets[k] + i_k`. -/
theorem coord_toInt (p : BlockPtr d) (i : Idx p.block) (k : Nat)
    (h : IntV.SFits 64 ((p.offsets.getD k 0).toInt + i.coord k)) (hi : i.coord k < 2 ^ 63) :
    (p.coord i k).toInt = (p.offsets.getD k 0).toInt + i.coord k := by
  unfold coord
  have hc : (IntV.ofNat (i.coord k) : I64).toInt = i.coord k :=
    IntV.toInt_ofNat (by norm_num) (by exact_mod_cast hi)
  rw [IntV.toInt_add (by norm_num) (by rw [hc]; exact h), hc]

/-- Without overflow, the element of lane `i` is `Σₖ (offsets[k] + i_k) · strides[k]`. -/
theorem elem_eq (p : BlockPtr d) (i : Idx p.block)
    (h : ∀ k, k < p.rank → IntV.SFits 64 ((p.offsets.getD k 0).toInt + i.coord k) ∧ i.coord k < 2 ^ 63 ∧
      IntV.SFits 64 (((p.offsets.getD k 0).toInt + i.coord k) * (p.strides.getD k 0).toInt)) :
    p.elem i = ((List.range p.rank).map fun k =>
      ((p.offsets.getD k 0).toInt + i.coord k) * (p.strides.getD k 0).toInt).sum := by
  unfold elem
  congr 1
  apply List.map_congr_left
  intro k hk
  have hk' := List.mem_range.mp hk
  obtain ⟨h1, h2, h3⟩ := h k hk'
  rw [IntV.toInt_mul (by norm_num) (by rw [coord_toInt p i k h1 h2]; exact h3), coord_toInt p i k h1 h2]

/-- A block pointer whose base is buffer `b`'s start addresses element `elem i` of `b` on each
lane that passes the check. -/
theorem addrs_buf (p : BlockPtr d) (b : Buf d) (hbase : p.base = b.ptr 0) (bc : List Nat)
    (hnn : ∀ i, p.inb bc i = true → 0 ≤ p.elem i) :
    b.Addrs (tl.addrs p.ptrs (some (p.mask bc))) (fun i => (p.elem i).toNat) (p.inb bc) := by
  intro i
  show (if p.inb bc i then some (p.base.offset (p.elem i)).addr else none) = _
  by_cases h : p.inb bc i = true
  · rw [if_pos h, if_pos h, hbase, Buf.ptr_offset, zero_add, Int.toNat_of_nonneg (hnn i h)]
  · rw [if_neg h, if_neg h]

end BlockPtr

/-- `tl.advance` adds the deltas to the offsets. -/
theorem BlockPtr.advance_offsets {d : DType} (p : BlockPtr d) (deltas : List I64) (k : Nat)
    (h : deltas.length = p.offsets.length) (hk : k < p.offsets.length) :
    (tl.advance p deltas).offsets.getD k 0 = p.offsets.getD k 0 + deltas.getD k 0 := by
  simp only [tl.advance, List.getD_eq_getElem?_getD]
  rw [List.getElem?_zipWith]
  rw [List.getElem?_eq_getElem hk, List.getElem?_eq_getElem (h ▸ hk)]
  rfl

section Rules
variable {d : DType} {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **`tl.load(p, boundary_check=bc, padding_option=pad)`** of a block pointer into buffer `b`. -/
theorem safe_load_block_ptr (b : Buf d) {p : BlockPtr d} {bc : List Nat} {pad : Padding}
    (hbase : p.base = b.ptr 0) (hok : p.Valid ∧ p.CheckOk bc)
    (hin : ∀ i, p.inb bc i = true → 0 ≤ p.elem i ∧ (p.elem i).toNat < b.len)
    (hpad : ∃ z : d.Val, pad.Holds z) (hv : b.Valid m)
    (hR : b.elems (Buf.img (fun i => (p.elem i).toNat) (p.inb bc)) ⊆ R)
    {k : Tensor p.block d.Val → Prog α}
    (hk : ∀ v : Tensor p.block d.Val,
      (∀ i, p.inb bc i = true → m.read d (b.ptr ((p.elem i).toNat : Int)).addr = some (v.get i)) →
      (∀ i, p.inb bc i = false → pad.Holds (v.get i)) → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load_block p bc pad >>= k) := by
  simp only [tl.load_block, Prog.bind_eq, Prog.bind, Prog.bind_assoc]
  refine .pick ⟨(), hok⟩ fun _ _ => ?_
  refine safe_load_buf b (p.addrs_buf b hbase bc fun i h => (hin i h).1) (fun i h => (hin i h).2) hv hR
    fun v hv1 _ => safe_pad hpad fun r hr hr' => hk r (fun i hi => (hr i hi).symm ▸ hv1 i hi) hr'

/-- **`tl.store(p, vals, boundary_check=bc)`** of a block pointer into buffer `b`, the lanes
that pass the check on distinct elements. -/
theorem safe_store_block_ptr (b : Buf d) {p : BlockPtr d} {bc : List Nat} {vals : Tensor p.block d.Val}
    (hbase : p.base = b.ptr 0) (hok : p.Valid ∧ p.CheckOk bc)
    (hin : ∀ i, p.inb bc i = true → 0 ≤ p.elem i ∧ (p.elem i).toNat < b.len)
    (hinj : ∀ i j, p.inb bc i = true → p.inb bc j = true → (p.elem i).toNat = (p.elem j).toNat → i = j)
    (hv : b.Valid m) (hW : b.elems (Buf.img (fun i => (p.elem i).toNat) (p.inb bc)) ⊆ W)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ i, p.inb bc i = true →
        m'.read d (b.ptr ((p.elem i).toNat : Int)).addr = some (vals.get i)) →
      (∀ x, x ∉ b.elems (Buf.img (fun i => (p.elem i).toNat) (p.inb bc)) → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) → Safe E R W Q m' (k ())) :
    Safe E R W Q m (tl.store_block p vals bc >>= k) := by
  simp only [tl.store_block, Prog.bind_eq, Prog.bind]
  refine .pick ⟨(), hok⟩ fun _ _ => ?_
  exact safe_store_buf b (p.addrs_buf b hbase bc fun i h => (hin i h).1) (fun i h => (hin i h).2) hinj hv hW hk

end Rules

end TVA
