import TVA.Lang.Desc
import TVA.Lemmas.Access

/-!
# Proof rules for tensor descriptors (TMA)

A descriptor over a buffer `b` (`D.base = b.ptr 0`) addresses, on each lane `i` inside the
tensor, element `D.elemAt (c i)` of `b`, where `c i` are the lane's coordinates
(`TensorDesc.addrs_buf`). So the `Buf.Addrs` kit applies:

* `safe_desc_load`: in-bounds lanes get their element, the others the descriptor's padding;
* `safe_desc_store`: in-bounds lanes are written (to distinct elements), nothing else changes;
* `safe_desc_atomic`: each in-bounds lane reduces its own element once (owned elements only:
  shared descriptor reductions race, see `TVA.Lang.Desc`);
* `safe_desc_gather`, `safe_desc_scatter`: the row-indexed forms;
* `safe_make_tensor_descriptor`: a descriptor built in the kernel (no memory access).
-/
namespace TVA

namespace TensorDesc

variable {d : DType}

theorem elemAt_nonneg (D : TensorDesc d) {c : Nat → Int} (h : D.inbAt c = true) : 0 ≤ D.elemAt c := by
  unfold inbAt at h
  have h' := of_decide_eq_true h
  unfold elemAt
  apply List.sum_nonneg
  simp only [List.mem_map, List.mem_range]
  rintro _ ⟨k, hk, rfl⟩
  exact mul_nonneg (h' k hk).1 (Int.natCast_nonneg _)

/-- A descriptor whose base is buffer `b`'s start addresses element `elemAt (c i)` of `b` on
each lane inside the tensor. -/
theorem addrs_buf (D : TensorDesc d) (b : Buf d) (hbase : D.base = b.ptr 0) {bs : Shape}
    (c : Idx bs → Nat → Int) :
    b.Addrs (tl.addrs (D.ptrsAt c) (some (D.maskAt c))) (fun i => (D.elemAt (c i)).toNat)
      (fun i => D.inbAt (c i)) := by
  intro i
  show (if D.inbAt (c i) then some (D.base.offset (D.elemAt (c i))).addr else none) = _
  by_cases h : D.inbAt (c i) = true
  · rw [if_pos h, if_pos h, hbase, Buf.ptr_offset, zero_add, Int.toNat_of_nonneg (D.elemAt_nonneg h)]
  · rw [if_neg h, if_neg h]

/-- The same without a mask, when every lane is inside the tensor. -/
theorem addrs_buf_all (D : TensorDesc d) (b : Buf d) (hbase : D.base = b.ptr 0) {bs : Shape}
    (c : Idx bs → Nat → Int) (hall : ∀ i, D.inbAt (c i) = true) :
    b.Addrs (tl.addrs (D.ptrsAt c) none) (fun i => (D.elemAt (c i)).toNat) (fun _ => true) := by
  intro i
  show some (D.base.offset (D.elemAt (c i))).addr = _
  rw [if_pos rfl, hbase, Buf.ptr_offset, zero_add, Int.toNat_of_nonneg (D.elemAt_nonneg (hall i))]

end TensorDesc

namespace RedOp

/-- Every reduction of an integer dtype has a result. -/
theorem upd_total_int (e : Env) (op : RedOp) {d : DType} (hd : d ∈ [DType.u32, .i32, .u64, .i64]) :
    ∀ v o, ∃ n, op.Upd e d v o n := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hd
  rcases hd with rfl | rfl | rfl | rfl <;> cases op <;> exact fun _ _ => ⟨_, rfl⟩

/-- A `min`/`max` reduction of a 16-bit float dtype has a result. -/
theorem upd_total_minmax (e : Env) {op : RedOp} (hop : op = .min ∨ op = .max) {d : DType}
    (hd : d = .f16 ∨ d = .bf16) : ∀ v o, ∃ n, op.Upd e d v o n := by
  rcases hop with rfl | rfl <;> rcases hd with rfl | rfl <;>
    exact fun _ _ => ⟨_, Or.inl (Or.inl (FloatV.NaNEq.refl _))⟩

end RedOp

section Rules
variable {d : DType} {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **`tl.make_tensor_descriptor`** in the kernel: no memory access; the descriptor of the
arguments' values when no shape entry or stride is negative. -/
theorem safe_make_tensor_descriptor {base : Ptr d} {shape : List I32} {strides : List I64}
    {block : Shape} {padding : Padding} (hs : ∀ x ∈ shape, 0 ≤ x.toInt)
    (ht : ∀ x ∈ strides, 0 ≤ x.toInt) {k : TensorDesc d → Prog α}
    (hk : Safe E R W Q m (k ⟨base, shape.map (fun x => x.toInt.toNat),
      strides.map (fun x => x.toInt.toNat), block, padding⟩)) :
    Safe E R W Q m (tl.make_tensor_descriptor base shape strides block padding >>= k) := by
  simp only [tl.make_tensor_descriptor, Prog.bind_eq, Prog.bind]
  exact .pick ⟨_, hs, ht, rfl⟩ fun D hD => by rw [hD.2.2]; exact hk

/-- **`desc.load`** from a buffer. -/
theorem safe_desc_load (b : Buf d) {bs : Shape} {D : TensorDesc d} {offsets : List I32}
    (hbase : D.base = b.ptr 0) (hacc : D.Access (offsets.map IntV.toInt) bs)
    (hin : ∀ i : Idx bs, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = true →
      (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat < b.len)
    (hv : b.Valid m)
    (hR : b.elems (Buf.img (fun i : Idx bs => (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat)
      (fun i => D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i))) ⊆ R)
    {k : Tensor bs d.Val → Prog α}
    (hk : ∀ v : Tensor bs d.Val,
      (∀ i, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = true →
        m.read d (b.ptr ((D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat : Int)).addr =
          some (v.get i)) →
      (∀ i, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = false → D.padding.Holds (v.get i)) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.desc_load D offsets >>= k) := by
  simp only [tl.desc_load, Prog.bind_eq, Prog.bind, Prog.bind_assoc]
  refine .pick ⟨(), hacc⟩ fun _ _ => ?_
  have hpad := D.padding.exists_holds (d := d) hacc.1.2.2.2.2.2.2.2.2.2
  refine safe_load_buf b (D.addrs_buf b hbase _) hin hv hR fun v hv1 _ =>
    safe_pad hpad fun r hr hr' => hk r (fun i hi => (hr i hi).symm ▸ hv1 i hi) hr'

/-- **`desc.store`** to a buffer, in-bounds lanes on distinct elements. -/
theorem safe_desc_store (b : Buf d) {bs : Shape} {D : TensorDesc d} {offsets : List I32}
    {vals : Tensor bs d.Val}
    (hbase : D.base = b.ptr 0) (hacc : D.Access (offsets.map IntV.toInt) bs)
    (hin : ∀ i : Idx bs, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = true →
      (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat < b.len)
    (hinj : ∀ i j : Idx bs, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = true →
      D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) j) = true →
      (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat =
        (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) j)).toNat → i = j)
    (hv : b.Valid m)
    (hW : b.elems (Buf.img (fun i : Idx bs => (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat)
      (fun i => D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i))) ⊆ W)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ i, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = true →
        m'.read d (b.ptr ((D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat : Int)).addr =
          some (vals.get i)) →
      (∀ x, x ∉ b.elems (Buf.img (fun i : Idx bs => (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat)
        (fun i => D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i))) → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) → Safe E R (W \ b.elems (Buf.img (fun i : Idx bs => (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat)
        (fun i => D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i)))) Q m' (k ())) :
    Safe E R W Q m (tl.desc_store D offsets vals >>= k) := by
  simp only [tl.desc_store, Prog.bind_eq, Prog.bind]
  refine .pick ⟨(), hacc⟩ fun _ _ => ?_
  exact safe_store_buf b (D.addrs_buf b hbase _) hin hinj hv hW hk

/-- **`desc.atomic_*`** on owned elements of a buffer: each in-bounds lane reduces its own
element once. The elements are outside the read footprint, and the rest of the program may not
touch them again. -/
theorem safe_desc_atomic (b : Buf d) {bs : Shape} {op : RedOp} {D : TensorDesc d}
    {offsets : List I32} {vals : Tensor bs d.Val}
    (hbase : D.base = b.ptr 0) (hacc : D.Access (offsets.map IntV.toInt) bs) (hop : op.Allowed d)
    (htot : ∀ v o, ∃ n, op.Upd E d v o n)
    (hin : ∀ i : Idx bs, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = true →
      (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat < b.len)
    (hinj : ∀ i j : Idx bs, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = true →
      D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) j) = true →
      (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat =
        (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) j)).toNat → i = j)
    (hv : b.Valid m)
    (hW : b.elems (Buf.img (fun i : Idx bs => (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat)
      (fun i => D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i))) ⊆ W)
    (hRd : Disjoint (b.elems (Buf.img (fun i : Idx bs => (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat)
      (fun i => D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i)))) R)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ i, D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i) = true →
        ∃ o n, m.read d (b.ptr ((D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat : Int)).addr =
          some o ∧ op.Upd E d (vals.get i) o n ∧
          m'.read d (b.ptr ((D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat : Int)).addr = some n) →
      (∀ x, x ∉ b.elems (Buf.img (fun i : Idx bs => (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat)
        (fun i => D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i))) → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) → Safe E R (W \ b.elems (Buf.img (fun i : Idx bs => (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat)
        (fun i => D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i)))) Q m' (k ())) :
    Safe E R W Q m (tl.desc_atomic op D offsets vals >>= k) := by
  simp only [tl.desc_atomic, Prog.bind_eq, Prog.bind, Prog.bind_assoc]
  refine .pick ⟨(), hacc, hop⟩ fun _ _ => .env ?_
  refine safe_atomic_buf b (D.addrs_buf b hbase _) hin hinj (fun i => htot (vals.get i)) hv hW hRd
    fun old m' hdone hframe hpres => ?_
  simp only [Prog.pure_eq, Prog.ret_bind]
  exact hk m' (fun i hi => by obtain ⟨n, h1, h2, h3⟩ := hdone i hi; exact ⟨_, n, h1, h2, h3⟩) hframe hpres

/-- **`desc.gather`** from a buffer: in-bounds lanes get their element. -/
theorem safe_desc_gather (b : Buf d) {M BN w : Nat} {D : TensorDesc d} {xs : Tensor [M] (IntV true w)}
    {y : I32} (hbase : D.base = b.ptr 0) (hacc : D.RowAccess BN y)
    (hin : ∀ l : Idx [M, BN], D.inbAt (TensorDesc.rowCoord xs y l) = true →
      (D.elemAt (TensorDesc.rowCoord xs y l)).toNat < b.len)
    (hv : b.Valid m)
    (hR : b.elems (Buf.img (fun l : Idx [M, BN] => (D.elemAt (TensorDesc.rowCoord xs y l)).toNat)
      (fun l => D.inbAt (TensorDesc.rowCoord xs y l))) ⊆ R)
    {k : Tensor [M, BN] d.Val → Prog α}
    (hk : ∀ v : Tensor [M, BN] d.Val,
      (∀ l, D.inbAt (TensorDesc.rowCoord xs y l) = true →
        m.read d (b.ptr ((D.elemAt (TensorDesc.rowCoord xs y l)).toNat : Int)).addr = some (v.get l)) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.desc_gather (BN := BN) D xs y >>= k) := by
  simp only [tl.desc_gather, Prog.bind_eq, Prog.bind]
  refine .pick ⟨(), hacc⟩ fun _ _ => ?_
  exact safe_load_buf b (D.addrs_buf b hbase _) hin hv hR fun v hv1 _ => hk v hv1

/-- **`desc.scatter`** to a buffer: every lane in bounds, on distinct elements. -/
theorem safe_desc_scatter (b : Buf d) {M BN w : Nat} {D : TensorDesc d} {vals : Tensor [M, BN] d.Val}
    {xs : Tensor [M] (IntV true w)} {y : I32} (hbase : D.base = b.ptr 0) (hacc : D.RowAccess BN y)
    (hall : ∀ l : Idx [M, BN], D.inbAt (TensorDesc.rowCoord xs y l) = true)
    (hin : ∀ l : Idx [M, BN], (D.elemAt (TensorDesc.rowCoord xs y l)).toNat < b.len)
    (hinj : ∀ l l' : Idx [M, BN], (D.elemAt (TensorDesc.rowCoord xs y l)).toNat =
      (D.elemAt (TensorDesc.rowCoord xs y l')).toNat → l = l')
    (hv : b.Valid m)
    (hW : b.elems (Buf.img (fun l : Idx [M, BN] => (D.elemAt (TensorDesc.rowCoord xs y l)).toNat)
      (fun _ => true)) ⊆ W)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ l, m'.read d (b.ptr ((D.elemAt (TensorDesc.rowCoord xs y l)).toNat : Int)).addr =
        some (vals.get l)) →
      (∀ x, x ∉ b.elems (Buf.img (fun l : Idx [M, BN] => (D.elemAt (TensorDesc.rowCoord xs y l)).toNat)
        (fun _ => true)) → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) → Safe E R (W \ b.elems (Buf.img (fun l : Idx [M, BN] => (D.elemAt (TensorDesc.rowCoord xs y l)).toNat)
        (fun _ => true))) Q m' (k ())) :
    Safe E R W Q m (tl.desc_scatter D vals xs y >>= k) := by
  simp only [tl.desc_scatter, Prog.bind_eq, Prog.bind]
  refine .pick ⟨(), hacc, hall⟩ fun _ _ => ?_
  exact safe_store_buf b (D.addrs_buf_all b hbase _ hall) (fun l _ => hin l)
    (fun l l' _ _ h => hinj l l' h) hv hW fun m' h1 h2 h3 => hk m' (fun l => h1 l rfl) h2 h3

end Rules

end TVA
