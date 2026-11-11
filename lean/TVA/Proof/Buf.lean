import TVA.Proof.Mem

/-!
# Buffers and the accesses that address their elements

A claim states its footprints and specification with buffers (`Buf`, `TVA.Memory`): `b.elems S`
is the bytes of the elements `S` of `b`, `b.Valid m` says every element can be read and written.
Most loads and stores address elements of one buffer: lane `i` addresses element `g i` of `b` when
its mask enables it (`en i`), which `b.Addrs (tl.addrs ptrs mask) g en` states
(`TVA.Triton.Memory`). From that:

* the access's bytes are `b.elems (Buf.img g en)` (`Buf.Addrs.accBytes`);
* lanes on distinct elements touch distinct bytes (`Buf.Addrs.distinct`);
* a valid buffer's elements can be read and written (`Buf.read_some`, `Buf.writable`).

How this fits: the load and store rules for buffers (`safe_load_buf`, `safe_store_buf`,
`TVA.Proof.Rules.Memory`) take `Buf.Addrs` and give each lane its element, so a per-program
`Safe` proof never unfolds `accBytes`. An access pattern (a 1-D block, a 2-D tile, a block pointer)
only proves `Buf.Addrs` once, from its lanes' pointers (`Buf.addrs_of_get`).
-/
namespace TVA

namespace Buf

variable {d : DType} {s : Shape}

/-- An element below `len` of a valid buffer can be written. -/
theorem writable (b : Buf d) {m : Mem} (hv : b.Valid m) {i : Nat} (hi : i < b.len) :
    m.Writable d (b.ptr i).addr :=
  ⟨b.aligned hv hi, fun _ hk => hv.2.2.2 _ (b.span_sub hi (Mem.mem_span hk))⟩

/-- An element below `len` of a valid buffer can be read. -/
theorem read_some (b : Buf d) {m : Mem} (hv : b.Valid m) {i : Nat} (hi : i < b.len) :
    ∃ v, m.read d (b.ptr i).addr = some v :=
  Mem.read_some_of_writable (b.writable hv hi)

/-- Validity needs only the buffer's bytes to stay present: it survives stores and atomics
(`writeAll_present`, `Mem.write_present`). -/
theorem valid_of_agree {b : Buf d} {m m' : Mem} (hv : b.Valid m)
    (h : ∀ x ∈ b.bytes, m' x ≠ none) : b.Valid m' := ⟨hv.1, hv.2.1, hv.2.2.1, h⟩

/-- **The bytes of an access to buffer elements** are those of the elements its enabled lanes
address. Use it to state a load's or store's footprint as `b.elems (img g en)`. -/
theorem Addrs.accBytes {b : Buf d} {addrs : Idx s → Option Int} {g : Idx s → Nat}
    {en : Idx s → Bool} (h : b.Addrs addrs g en) : TVA.accBytes d addrs = b.elems (img g en) := by
  ext x
  constructor
  · rintro ⟨i, a, ha, hx⟩
    rw [h i] at ha
    split_ifs at ha with he
    · cases ha; exact ⟨_, ⟨i, he, rfl⟩, hx⟩
  · rintro ⟨_, ⟨i, he, rfl⟩, hx⟩
    exact ⟨i, _, by rw [h i, ite_eq_left he], hx⟩

/-- **Lanes on distinct elements touch distinct bytes**, when no two enabled lanes address the
same element. -/
theorem Addrs.distinct {b : Buf d} {addrs : Idx s → Option Int} {g : Idx s → Nat}
    {en : Idx s → Bool} (h : b.Addrs addrs g en)
    (hinj : ∀ i j, en i = true → en j = true → g i = g j → i = j) : DistinctLanes d addrs := by
  intro i j a a' hne ha ha'
  rw [h i] at ha; rw [h j] at ha'
  split_ifs at ha ha' with hi hj
  cases ha; cases ha'
  exact b.span_disjoint (fun e => hne (hinj i j hi hj (by exact_mod_cast e)))

/-- An enabled lane of `b.Addrs` addresses element `g i`. -/
theorem Addrs.some {b : Buf d} {addrs : Idx s → Option Int} {g : Idx s → Nat}
    {en : Idx s → Bool} (h : b.Addrs addrs g en) {i : Idx s} (he : en i = true) :
    addrs i = some (b.ptr (g i : Int)).addr := by
  rw [h i, ite_eq_left he]

end Buf

/-- An enabled lane's address is its pointer's. -/
theorem tl.addrs_some {d : DType} {s : Shape} {ptrs : Tensor s (Ptr d)}
    {mask : Option (Tensor s Bool)} {i : Idx s} {a : Int}
    (h : tl.addrs ptrs mask i = some a) : a = (ptrs.get i).addr := by
  unfold tl.addrs at h
  cases mask with
  | none => simp only [Option.some.injEq] at h; exact h.symm
  | some mk =>
    by_cases hm : mk.get i = true
    · simp only [hm, ite_true, Option.some.injEq] at h; exact h.symm
    · simp [hm] at h

end TVA
