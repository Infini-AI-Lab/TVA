import TVA.Proof.Basic

/-!
# Tensor descriptors

The rules for `TVA.Triton.Desc` (TMA accesses through a host-built descriptor):

```python
x = desc.load([r0, c0])            # a [BM, BN] block, padding outside the tensor
desc.store([r0, c0], x)
desc.atomic_add([r0, c0], x)
```

A descriptor over a buffer `b` (`D.base = b.ptr 0`) addresses, on each lane `i` inside the tensor
(`D.laneIn offsets i`), element `D.lane offsets i` of `b` (`TensorDesc.addrs_buf`), so the buffer
rules apply:

* `safe_desc_load`: in-bounds lanes get their element, the others the descriptor's padding;
* `safe_desc_store`: in-bounds lanes are written (to distinct elements), nothing else changes;
* `safe_desc_atomic`: each in-bounds lane reduces its own element once (elements the program owns;
  reductions from several programs into one element race);
* `safe_desc_gather`, `safe_desc_scatter`: the row-indexed forms (`D.rowLane`, `D.rowLaneIn`);
* `safe_make_tensor_descriptor`: a descriptor built in the kernel (no memory access).

`RedOp.upd_total_int` and `RedOp.upd_total_minmax` give the `htot` premise of
`safe_desc_atomic`.
-/
namespace TVA

namespace TensorDesc

variable {d : DType}

/-- The element (relative to the base) that lane `i` of the block at `offsets` addresses. -/
abbrev lane (D : TensorDesc d) (offsets : List I32) {bs : Shape} (i : Idx bs) : Nat :=
  (D.elemAt (TensorDesc.coord (offsets.map IntV.toInt) i)).toNat

/-- Lane `i` of the block at `offsets` lies inside the tensor. -/
abbrev laneIn (D : TensorDesc d) (offsets : List I32) {bs : Shape} (i : Idx bs) : Bool :=
  D.inbAt (TensorDesc.coord (offsets.map IntV.toInt) i)

/-- The element that lane `l` of a gather or scatter at rows `xs`, column `y` addresses. -/
abbrev rowLane {M BN w : Nat} (D : TensorDesc d) (xs : Tensor [M] (IntV true w)) (y : I32)
    (l : Idx [M, BN]) : Nat :=
  (D.elemAt (TensorDesc.rowCoord xs y l)).toNat

/-- Lane `l` of a gather or scatter at rows `xs`, column `y` lies inside the tensor. -/
abbrev rowLaneIn {M BN w : Nat} (D : TensorDesc d) (xs : Tensor [M] (IntV true w)) (y : I32)
    (l : Idx [M, BN]) : Bool :=
  D.inbAt (TensorDesc.rowCoord xs y l)

/-- A position inside the tensor has a nonnegative element. -/
theorem elemAt_nonneg (D : TensorDesc d) {c : Nat → Int} (h : D.inbAt c = true) :
    0 ≤ D.elemAt c := by
  unfold inbAt at h
  have h' := of_decide_eq_true h
  unfold elemAt
  apply List.sum_nonneg
  simp only [List.mem_map, List.mem_range]
  rintro _ ⟨k, hk, rfl⟩
  exact mul_nonneg (h' k hk).1 (Int.natCast_nonneg _)

/-- **The lanes of a descriptor access, as buffer elements**: with `D.base = b.ptr 0`, lane `i`
with coordinates `c i` addresses element `D.elemAt (c i)` of `b` when it lies inside the tensor. -/
theorem addrs_buf (D : TensorDesc d) (b : Buf d) (hbase : D.base = b.ptr 0) {bs : Shape}
    (c : Idx bs → Nat → Int) :
    b.Addrs (tl.addrs (D.ptrsAt c) (some (D.maskAt c))) (fun i => (D.elemAt (c i)).toNat)
      (fun i => D.inbAt (c i)) := by
  intro i
  show (if D.inbAt (c i) then some (D.base.offset (D.elemAt (c i))).addr else none) = _
  by_cases h : D.inbAt (c i) = true
  · rw [ite_eq_left h, ite_eq_left h, hbase, Buf.ptr_offset, zero_add,
      Int.toNat_of_nonneg (D.elemAt_nonneg h)]
  · rw [ite_eq_right h, ite_eq_right h]

/-- `addrs_buf` without a mask, when every lane lies inside the tensor. -/
theorem addrs_buf_all (D : TensorDesc d) (b : Buf d) (hbase : D.base = b.ptr 0) {bs : Shape}
    (c : Idx bs → Nat → Int) (hall : ∀ i, D.inbAt (c i) = true) :
    b.Addrs (tl.addrs (D.ptrsAt c) none) (fun i => (D.elemAt (c i)).toNat) (fun _ => true) := by
  intro i
  show some (D.base.offset (D.elemAt (c i))).addr = _
  rw [ite_eq_left rfl, hbase, Buf.ptr_offset, zero_add,
    Int.toNat_of_nonneg (D.elemAt_nonneg (hall i))]

end TensorDesc

namespace RedOp

/-- **Every integer reduction has a result** (`u32`, `i32`, `u64`, `i64`): the `htot` of
`safe_desc_atomic`. -/
theorem upd_total_int (e : Env) (op : RedOp) {d : DType}
    (hd : d ∈ [DType.u32, .i32, .u64, .i64]) : ∀ v o, ∃ n, op.Upd e d v o n := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hd
  rcases hd with rfl | rfl | rfl | rfl <;> cases op <;> exact fun _ _ => ⟨_, rfl⟩

/-- **A `min` or `max` reduction of a 16-bit float dtype has a result**: the `htot` of
`safe_desc_atomic`. -/
theorem upd_total_minmax (e : Env) {op : RedOp} (hop : op = .min ∨ op = .max) {d : DType}
    (hd : d = .f16 ∨ d = .bf16) : ∀ v o, ∃ n, op.Upd e d v o n := by
  rcases hop with rfl | rfl <;> rcases hd with rfl | rfl <;>
    exact fun v o => (maxRel_total _ o v).imp fun _ h => Or.inl h

end RedOp

section Rules
variable {d : DType} {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

open TensorDesc

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

/-- **`desc.load(offsets)`** from a buffer: in-bounds lanes get their element, the others hold
the descriptor's padding. The access must meet Triton's requirements (`D.Access`). -/
theorem safe_desc_load (b : Buf d) {bs : Shape} {D : TensorDesc d} {offsets : List I32}
    (hbase : D.base = b.ptr 0) (hacc : D.Access (offsets.map IntV.toInt) bs)
    (hin : ∀ i : Idx bs, D.laneIn offsets i = true → D.lane offsets i < b.len)
    (hv : b.Valid m) (hR : b.elems (Buf.img (D.lane (bs := bs) offsets) (D.laneIn offsets)) ⊆ R)
    {k : Tensor bs d.Val → Prog α}
    (hk : ∀ v : Tensor bs d.Val,
      (∀ i, D.laneIn offsets i = true →
        m.read d (b.ptr (D.lane offsets i : Int)).addr = some (v.get i)) →
      (∀ i, D.laneIn offsets i = false → D.padding.Holds (v.get i)) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.desc_load D offsets >>= k) := by
  simp only [tl.desc_load, Prog.bind_eq, Prog.bind, Prog.bind_assoc]
  refine .pick ⟨(), hacc⟩ fun _ _ => ?_
  have hpad := D.padding.exists_holds (d := d) hacc.1.2.2.2.2.2.2.2.2.2
  exact safe_load_buf b (D.addrs_buf b hbase _) hin hv hR fun v hv1 _ =>
    safe_pad hpad fun r hr hr' => hk r (fun i hi => (hr i hi).symm ▸ hv1 i hi) hr'

/-- **`desc.store(offsets, vals)`** to a buffer: in-bounds lanes, on distinct elements (`hinj`),
are written; nothing else changes. -/
theorem safe_desc_store (b : Buf d) {bs : Shape} {D : TensorDesc d} {offsets : List I32}
    {vals : Tensor bs d.Val}
    (hbase : D.base = b.ptr 0) (hacc : D.Access (offsets.map IntV.toInt) bs)
    (hin : ∀ i : Idx bs, D.laneIn offsets i = true → D.lane offsets i < b.len)
    (hinj : ∀ i j : Idx bs, D.laneIn offsets i = true → D.laneIn offsets j = true →
      D.lane offsets i = D.lane offsets j → i = j)
    (hv : b.Valid m) (hW : b.elems (Buf.img (D.lane (bs := bs) offsets) (D.laneIn offsets)) ⊆ W)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ i, D.laneIn offsets i = true →
        m'.read d (b.ptr (D.lane offsets i : Int)).addr = some (vals.get i)) →
      (∀ x, x ∉ b.elems (Buf.img (D.lane (bs := bs) offsets) (D.laneIn offsets)) → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (Buf.img (D.lane (bs := bs) offsets) (D.laneIn offsets))) Q m' (k ())) :
    Safe E R W Q m (tl.desc_store D offsets vals >>= k) := by
  simp only [tl.desc_store, Prog.bind_eq, Prog.bind]
  refine .pick ⟨(), hacc⟩ fun _ _ => ?_
  exact safe_store_buf b (D.addrs_buf b hbase _) hin hinj hv hW hk

/-- **`desc.atomic_add(offsets, vals)`** and the other reductions, on elements the program owns:
each in-bounds lane reduces its own element once (`hinj`). The elements are outside the read
footprint, and the rest of the program may not touch them again. `htot`: every reduction has a
result (`RedOp.upd_total_int`, `RedOp.upd_total_minmax`, the device's `atomicAdd` for floats). -/
theorem safe_desc_atomic (b : Buf d) {bs : Shape} {op : RedOp} {D : TensorDesc d}
    {offsets : List I32} {vals : Tensor bs d.Val}
    (hbase : D.base = b.ptr 0) (hacc : D.Access (offsets.map IntV.toInt) bs) (hop : op.Allowed d)
    (htot : ∀ v o, ∃ n, op.Upd E d v o n)
    (hin : ∀ i : Idx bs, D.laneIn offsets i = true → D.lane offsets i < b.len)
    (hinj : ∀ i j : Idx bs, D.laneIn offsets i = true → D.laneIn offsets j = true →
      D.lane offsets i = D.lane offsets j → i = j)
    (hv : b.Valid m) (hW : b.elems (Buf.img (D.lane (bs := bs) offsets) (D.laneIn offsets)) ⊆ W)
    (hRd : Disjoint (b.elems (Buf.img (D.lane (bs := bs) offsets) (D.laneIn offsets))) R)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ i, D.laneIn offsets i = true →
        ∃ o n, m.read d (b.ptr (D.lane offsets i : Int)).addr = some o ∧
          op.Upd E d (vals.get i) o n ∧ m'.read d (b.ptr (D.lane offsets i : Int)).addr = some n) →
      (∀ x, x ∉ b.elems (Buf.img (D.lane (bs := bs) offsets) (D.laneIn offsets)) → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (Buf.img (D.lane (bs := bs) offsets) (D.laneIn offsets))) Q m' (k ())) :
    Safe E R W Q m (tl.desc_atomic op D offsets vals >>= k) := by
  simp only [tl.desc_atomic, Prog.bind_eq, Prog.bind, Prog.bind_assoc, Prog.pure_eq]
  refine .pick ⟨(), hacc, hop⟩ fun _ _ => .env ?_
  exact safe_atomic_buf b (D.addrs_buf b hbase _) hin hinj (fun i => htot (vals.get i)) hv hW hRd
    hk

/-- **`desc.gather(xs, y)`** from a buffer: in-bounds lanes get their element (the others are
unspecified). -/
theorem safe_desc_gather (b : Buf d) {M BN w : Nat} {D : TensorDesc d}
    {xs : Tensor [M] (IntV true w)} {y : I32} (hbase : D.base = b.ptr 0)
    (hacc : D.RowAccess BN y)
    (hin : ∀ l : Idx [M, BN], D.rowLaneIn xs y l = true → D.rowLane xs y l < b.len)
    (hv : b.Valid m)
    (hR : b.elems (Buf.img (D.rowLane (BN := BN) xs y) (D.rowLaneIn xs y)) ⊆ R)
    {k : Tensor [M, BN] d.Val → Prog α}
    (hk : ∀ v : Tensor [M, BN] d.Val,
      (∀ l, D.rowLaneIn xs y l = true →
        m.read d (b.ptr (D.rowLane xs y l : Int)).addr = some (v.get l)) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.desc_gather (BN := BN) D xs y >>= k) := by
  simp only [tl.desc_gather, Prog.bind_eq, Prog.bind]
  refine .pick ⟨(), hacc⟩ fun _ _ => ?_
  exact safe_load_buf b (D.addrs_buf b hbase _) hin hv hR fun v hv1 _ => hk v hv1

/-- **`desc.scatter(vals, xs, y)`** to a buffer: every lane lies inside the tensor (`hall`), on
distinct elements (`hinj`); nothing else changes. -/
theorem safe_desc_scatter (b : Buf d) {M BN w : Nat} {D : TensorDesc d}
    {vals : Tensor [M, BN] d.Val} {xs : Tensor [M] (IntV true w)} {y : I32}
    (hbase : D.base = b.ptr 0) (hacc : D.RowAccess BN y)
    (hall : ∀ l : Idx [M, BN], D.rowLaneIn xs y l = true)
    (hin : ∀ l : Idx [M, BN], D.rowLane xs y l < b.len)
    (hinj : ∀ l l' : Idx [M, BN], D.rowLane xs y l = D.rowLane xs y l' → l = l')
    (hv : b.Valid m)
    (hW : b.elems (Buf.img (D.rowLane (BN := BN) xs y) (fun _ => true)) ⊆ W)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ l, m'.read d (b.ptr (D.rowLane xs y l : Int)).addr = some (vals.get l)) →
      (∀ x, x ∉ b.elems (Buf.img (D.rowLane (BN := BN) xs y) (fun _ => true)) → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (Buf.img (D.rowLane (BN := BN) xs y) (fun _ => true))) Q m' (k ())) :
    Safe E R W Q m (tl.desc_scatter D vals xs y >>= k) := by
  simp only [tl.desc_scatter, Prog.bind_eq, Prog.bind]
  refine .pick ⟨(), hacc, hall⟩ fun _ _ => ?_
  exact safe_store_buf b (D.addrs_buf_all b hbase _ hall) (fun l _ => hin l)
    (fun l l' _ _ h => hinj l l' h) hv hW fun m' h1 h2 h3 => hk m' (fun l => h1 l rfl) h2 h3

end Rules

end TVA
