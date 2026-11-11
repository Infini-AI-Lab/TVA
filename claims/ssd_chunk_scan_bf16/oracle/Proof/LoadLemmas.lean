import Proof.Decode

/-!
# Masked loads of one element, and of a vector

The scalar loads of the set-ups (`tl.load(ptr, mask=cond, other=o)` on a scalar pointer) and the
`[BM]` vector loads (`ptr + offs * 1`, `mask = offs < n`), as rules whose hypotheses are the lane
facts the walk has.
-/

namespace Claims.SsdChunkScan

open TVA

/-- **A masked scalar load.** The mask `mk` (a `Bool`) enables the element `idx` of `b`; a disabled
load returns `o`. -/
theorem safe_load_scalar {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {δ : DType} (b : Buf δ) {ptr : Ptr δ} {mk : Bool} {o : δ.Val} {idx : Nat}
    (hp : mk = true → ptr = b.ptr (idx : Int)) (hin : mk = true → idx < b.len)
    (hv : b.Valid m) (hR : b.bytes ⊆ R) {k : Tensor [] δ.Val → Prog α}
    (hk : ∀ v : Tensor [] δ.Val, (mk = true → m.read δ (b.ptr (idx : Int)).addr = some v.item) →
      (mk = false → v.item = o) → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load (Tensor.scalar ptr) (some (Tensor.scalar mk)) (some (Tensor.scalar o))
      >>= k) := by
  refine safe_load_buf b (g := fun _ : Idx [] => idx) (en := fun _ : Idx [] => mk) ?_ (fun _ h => hin h) hv ?_
    (fun v hv1 hv2 => hk v (fun h => hv1 () h) (fun h => hv2 () h _ rfl))
  · exact Buf.addrs_of_get b (fun _ => rfl) (fun _ h => hp h)
  · exact (b.elems_sub_bytes (fun x hx => by obtain ⟨_, he, rfl⟩ := hx; exact hin he)).trans hR

/-- **A masked vector load** at consecutive elements: lane `l` addresses element `g l` of `b` when
`en l`; a disabled lane returns `o`. (A thin form of `safe_load_buf` for `[n]` vectors.) -/
theorem safe_load_vec {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {δ : DType} {n : Nat} (b : Buf δ) {ptrs : Tensor [n] (Ptr δ)} {mask : Tensor [n] Bool}
    {o : δ.Val} {g : Fin n → Nat} {en : Fin n → Bool}
    (hm : ∀ l : Fin n, mask.get1 l = en l)
    (hp : ∀ l : Fin n, en l = true → ptrs.get1 l = b.ptr (g l : Int))
    (hin : ∀ l, en l = true → g l < b.len) (hv : b.Valid m) (hR : b.bytes ⊆ R)
    {k : Tensor [n] δ.Val → Prog α}
    (hk : ∀ v : Tensor [n] δ.Val, (∀ l, en l = true → m.read δ (b.ptr (g l : Int)).addr = some (v.get1 l)) →
      (∀ l, en l = false → v.get1 l = o) → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs (some mask) (some (bc o)) >>= k) := by
  refine safe_load_buf b (g := fun i : Idx [n] => g i.1) (en := fun i : Idx [n] => en i.1) ?_ (fun i h => hin i.1 h) hv ?_
    (fun v hv1 hv2 => hk v (fun l h => hv1 (l, ()) h) (fun l h => hv2 (l, ()) h _ rfl))
  · exact Buf.addrs_of_get b (fun i => hm i.1) (fun i h => hp i.1 h)
  · exact (b.elems_sub_bytes (fun x hx => by obtain ⟨i, he, rfl⟩ := hx; exact hin i.1 he)).trans hR

/-- Lane `l` of `p + offs * 1`, `p` pointing at element `e` of `b` and lane `l` of `offs` of value
`f l`: element `e + f l`. -/
theorem ptr_vec_lane {δ : DType} {n : Nat} (b : Buf δ) {p : Ptr δ} {e : Nat}
    (hp : p = b.ptr (e : Int)) {offs : Tensor [n] I32} {f : Fin n → Nat}
    (hoff : ∀ l, (offs.get1 l).toInt = (f l : Int)) (hf : ∀ l, f l < 2 ^ 31) (l : Fin n) :
    (p + offs * (IntV.ofNat 1 : I32)).get1 l = b.ptr ((e + f l : Nat) : Int) := by
  show p.offset ((offs * (IntV.ofNat 1 : I32)).get1 l).toInt = _
  rw [Tensor.get1_muls]
  have := IntV.mul_nat (hoff l) (IntV.I32.toInt_ofNat (by norm_num : 1 < 2 ^ 31))
    (show f l * 1 < 2 ^ (32 - 1) by have := hf l; omega)
  rw [this, hp, Buf.ptr_offset]; congr 1; push_cast; ring

end Claims.SsdChunkScan
