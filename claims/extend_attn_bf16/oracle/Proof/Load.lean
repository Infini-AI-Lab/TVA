import Proof.Setup

/-!
# Loads and stores by lane

`safe_load_en` and `safe_store_en`: a masked load or store whose enabled lanes `en` address the
elements `g` of one buffer. The lane facts (the mask, the pointers, the extent) are the premises.
-/

namespace Claims.ExtendAttn

open TVA

section
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}
  {d : DType}

/-- **A masked load**: lane `i` is enabled exactly when `en i`, and then points at element `g i`
of `b`, below `b.len`. Each enabled lane gets its element; a masked-off lane gets `other`. -/
theorem safe_load_en (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Tensor s Bool}
    {other : Option (Tensor s d.Val)} (en : Idx s → Bool) (g : Idx s → Nat)
    (hm : ∀ i, mask.get i = en i) (hp : ∀ i, en i = true → ptrs.get i = b.ptr (g i : Int))
    (hin : ∀ i, en i = true → g i < b.len) (hv : b.Valid m) (hR : b.bytes ⊆ R)
    {k : Tensor s d.Val → Prog α}
    (hk : ∀ v : Tensor s d.Val,
      (∀ i, en i = true → m.read d (b.ptr (g i : Int)).addr = some (v.get i)) →
      (∀ i, en i = false → ∀ o, other = some o → v.get i = o.get i) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs (some mask) other >>= k) :=
  safe_load_buf b (Buf.addrs_of_get b hm hp) hin hv
    ((b.elems_sub_bytes (fun j (hj : j ∈ Buf.img g en) => by
      obtain ⟨i, hi, rfl⟩ := hj; exact hin i hi)).trans hR) hk

/-- **A masked store** to distinct elements. -/
theorem safe_store_en (b : Buf d) {ptrs : Tensor s (Ptr d)} {mask : Tensor s Bool}
    {vals : Tensor s d.Val} (en : Idx s → Bool) (g : Idx s → Nat)
    (hm : ∀ i, mask.get i = en i) (hp : ∀ i, en i = true → ptrs.get i = b.ptr (g i : Int))
    (hin : ∀ i, en i = true → g i < b.len)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j) (hv : b.Valid m)
    (hW : b.elems (Buf.img g en) ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m', (∀ i, en i = true → m'.read d (b.ptr (g i : Int)).addr = some (vals.get i)) →
      (∀ x, x ∉ b.elems (Buf.img g en) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (Buf.img g en)) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals (some mask) >>= k) :=
  safe_store_buf b (Buf.addrs_of_get b hm hp) hin hinj hv hW hk

end

end Claims.ExtendAttn
