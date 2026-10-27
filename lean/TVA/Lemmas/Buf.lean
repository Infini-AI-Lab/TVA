import TVA.Lemmas.Mem

/-!
# Buffers

`Buf d` is a contiguous array of `len` elements of dtype `d` at byte address `base`: a tensor
argument as the host passes it (`x_ptr` with `n` elements). Specifications and footprints are
stated with buffers:

* `b.ptr i`: the pointer to element `i` (`x_ptr + i`);
* `b.bytes`: the bytes it occupies;
* `b.Holds m v`: memory `m` holds the values `v` in it;
* `b.Valid m`: it is aligned and all its bytes are present.

Distinct elements do not overlap (`Buf.span_disjoint`), so a store of a tile to distinct
elements reads back elementwise.
-/
namespace TVA

/-- A contiguous buffer of `len` elements of dtype `d` at byte address `base`. -/
structure Buf (d : DType) where
  base : Int
  len : Nat
  deriving DecidableEq, Repr

namespace Buf

variable {d : DType}

/-- The pointer to element `i` (the kernel argument is `b.ptr 0`). -/
def ptr (b : Buf d) (i : Int) : Ptr d := ⟨b.base + i * d.bytes⟩

/-- The bytes of the buffer. -/
def bytes (b : Buf d) : Set Int := {x | b.base ≤ x ∧ x < b.base + b.len * d.bytes}

/-- The buffer is aligned, inside `[0, 2^64)`, and all its bytes are present in `m`. -/
def Valid (b : Buf d) (m : Mem) : Prop :=
  0 ≤ b.base ∧ b.base + b.len * d.bytes ≤ 2 ^ 64 ∧ b.base % d.bytes = 0 ∧ ∀ x ∈ b.bytes, m x ≠ none

/-- Memory `m` holds `v i` at each element `i < len`. -/
def Holds (b : Buf d) (m : Mem) (v : Nat → d.Val) : Prop :=
  ∀ i : Nat, i < b.len → m.read d (b.ptr i).addr = some (v i)

/-- Two buffers do not overlap. -/
def Disjoint' (b : Buf d) {d' : DType} (c : Buf d') : Prop := Disjoint b.bytes c.bytes

@[simp] theorem ptr_addr (b : Buf d) (i : Int) : (b.ptr i).addr = b.base + i * d.bytes := rfl

theorem span_sub (b : Buf d) {i : Nat} (hi : i < b.len) : Mem.span d (b.ptr i).addr ⊆ b.bytes := by
  intro x hx
  simp only [Mem.span, Set.mem_ofPred_eq, ptr_addr] at hx
  simp only [bytes, Set.mem_ofPred_eq]
  have hb := d.bytes_pos
  constructor
  · nlinarith
  · have : (i : Int) + 1 ≤ b.len := by exact_mod_cast hi
    nlinarith

theorem span_disjoint (b : Buf d) {i j : Int} (hij : i ≠ j) :
    Disjoint (Mem.span d (b.ptr i).addr) (Mem.span d (b.ptr j).addr) := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Mem.span, Set.mem_ofPred_eq, ptr_addr] at hx hy
  have hb := d.bytes_pos
  rcases lt_or_gt_of_ne hij with h | h
  · have : i + 1 ≤ j := h
    nlinarith
  · have : j + 1 ≤ i := h
    nlinarith

theorem aligned (b : Buf d) {m : Mem} (hv : b.Valid m) {i : Nat} (hi : i < b.len) :
    Mem.Aligned d (b.ptr i).addr := by
  obtain ⟨h0, h1, h2, -⟩ := hv
  have hb := d.bytes_pos
  refine ⟨by simp; positivity, ?_, ?_⟩
  · simp only [ptr_addr]
    have : (i : Int) + 1 ≤ b.len := by exact_mod_cast hi
    nlinarith
  · simp only [ptr_addr]; rw [Int.add_mul_emod_self_right]; exact h2

theorem writable (b : Buf d) {m : Mem} (hv : b.Valid m) {i : Nat} (hi : i < b.len) :
    m.Writable d (b.ptr i).addr :=
  ⟨b.aligned hv hi, fun k hk => hv.2.2.2 _ (b.span_sub hi (Mem.mem_span hk))⟩

/-- A valid buffer element can be read. -/
theorem read_some (b : Buf d) {m : Mem} (hv : b.Valid m) {i : Nat} (hi : i < b.len) :
    ∃ v, m.read d (b.ptr i).addr = some v := by
  have hw := b.writable hv hi
  -- reading after writing the same bytes back: the bytes are present
  unfold Mem.read
  rw [if_pos hw.1]
  have : ∃ bs, m.readBytes (b.ptr i).addr d.bytes = some bs := by
    have key : ∀ (n : Nat) (a : Int), (∀ k : Nat, k < n → m (a + k) ≠ none) → ∃ bs, m.readBytes a n = some bs := by
      intro n
      induction n with
      | zero => intro a _; exact ⟨[], rfl⟩
      | succ n ih =>
        intro a ha
        obtain ⟨b0, hb0⟩ := Option.ne_none_iff_exists'.mp (by simpa using ha 0 (by omega))
        obtain ⟨bs, hbs⟩ := ih (a + 1) (fun k hk => by
          have := ha (k + 1) (by omega); push_cast at this; rwa [show a + 1 + k = a + (k + 1) by ring])
        exact ⟨b0 :: bs, by simp [Mem.readBytes, hb0, hbs]⟩
    exact key _ _ hw.2
  obtain ⟨bs, hbs⟩ := this
  exact ⟨_, by rw [hbs]; rfl⟩

/-- Validity depends only on the buffer's bytes being present. -/
theorem valid_of_agree {b : Buf d} {m m' : Mem} (hv : b.Valid m) (h : ∀ x ∈ b.bytes, m' x ≠ none) :
    b.Valid m' := ⟨hv.1, hv.2.1, hv.2.2.1, h⟩

end Buf

/-! ## Stores to distinct elements -/

theorem Mem.write_present {m : Mem} {d : DType} {a : Int} (v : d.Val) {x : Int} (hx : m x ≠ none) :
    m.write d a v x ≠ none := by
  unfold Mem.write Mem.writeBytes
  rw [Mem.leBytes_len]
  split_ifs with h
  · have hlt : (x - a).toNat < (TVA.Math.Bytes.leBytes d.bytes (d.toBits v)).length := by
      rw [Mem.leBytes_len]; omega
    rw [List.getElem?_eq_getElem hlt]; simp
  · exact hx

theorem Mem.writable_write {m : Mem} {d d' : DType} {a b : Int} (v : d.Val) (h : m.Writable d' b) :
    (m.write d a v).Writable d' b :=
  ⟨h.1, fun i hi => Mem.write_present v (h.2 i hi)⟩

section
variable {d : DType} {s : Shape}

/-- The enabled lanes of an access touch pairwise disjoint bytes. -/
def DistinctLanes (d : DType) (addrs : Idx s → Option Int) : Prop :=
  ∀ i j a b, i ≠ j → addrs i = some a → addrs j = some b → Disjoint (Mem.span d a) (Mem.span d b)

theorem writeAll_writable {m : Mem} (addrs : Idx s → Option Int) (vals : Tensor s d.Val)
    (order : List (Idx s)) {d' : DType} {b : Int} (h : m.Writable d' b) :
    (writeAll m d addrs vals order).Writable d' b := by
  induction order generalizing m with
  | nil => exact h
  | cons i rest ih =>
    simp only [writeAll]
    cases addrs i with
    | some a => exact ih (Mem.writable_write _ h)
    | none => exact ih h

/-- A store keeps every present byte present. -/
theorem writeAll_present {m : Mem} (addrs : Idx s → Option Int) (vals : Tensor s d.Val)
    (order : List (Idx s)) {x : Int} (h : m x ≠ none) : writeAll m d addrs vals order x ≠ none := by
  induction order generalizing m with
  | nil => exact h
  | cons i rest ih =>
    simp only [writeAll]
    cases addrs i with
    | some a => exact ih (Mem.write_present _ h)
    | none => exact ih h

/-- Writes of lanes whose bytes avoid `x` leave `x` alone. -/
theorem writeAll_frame_of {m : Mem} {addrs : Idx s → Option Int} {vals : Tensor s d.Val}
    {S : Set Int} : ∀ {rest : List (Idx s)},
    (∀ k, k ∈ rest → ∀ b, addrs k = some b → Disjoint (Mem.span d b) S) →
    ∀ x ∈ S, writeAll m d addrs vals rest x = m x
  | [], _, _, _ => rfl
  | k :: rest, havoid, x, hx => by
    simp only [writeAll]
    cases hk : addrs k with
    | some b =>
      simp only
      rw [writeAll_frame_of (m := m.write d b (vals.get k)) (fun k' hk' b' hb' => havoid k'
        (List.mem_cons_of_mem _ hk') b' hb') x hx]
      exact Mem.write_frame _ _ _ _ (fun hxb => Set.disjoint_left.mp
        (havoid k (List.mem_cons_self) b hk) hxb hx)
    | none =>
      exact writeAll_frame_of (fun k' hk' b' hb' => havoid k' (List.mem_cons_of_mem _ hk') b' hb') x hx

/-- After a store to distinct elements, in any order, each enabled lane reads back its value. -/
theorem writeAll_read {m : Mem} {addrs : Idx s → Option Int} {vals : Tensor s d.Val}
    (hdist : DistinctLanes d addrs) (hok : StoreOk m d addrs) :
    ∀ (order : List (Idx s)), order.Nodup → ∀ i a, i ∈ order → addrs i = some a →
      (writeAll m d addrs vals order).read d a = some (vals.get i) := by
  intro order
  induction order generalizing m with
  | nil => intro _ i a hi; cases hi
  | cons j rest ih =>
    intro hnd i a hi ha
    have hnd' := List.nodup_cons.mp hnd
    simp only [writeAll]
    by_cases hij : j = i
    · subst hij
      rw [ha]
      simp only
      have hrest : ∀ x ∈ Mem.span d a,
          writeAll (m.write d a (vals.get j)) d addrs vals rest x = m.write d a (vals.get j) x :=
        writeAll_frame_of (fun k hk b hb => hdist k j b a (fun h => hnd'.1 (h ▸ hk)) hb ha)
      rw [Mem.read_congr hrest]
      exact Mem.read_write_self (hok j a ha) _
    · have hi' : i ∈ rest := by
        rcases List.mem_cons.mp hi with h | h
        · exact absurd h.symm hij
        · exact h
      cases hj : addrs j with
      | some b =>
        simp only
        exact ih (fun k c hc => Mem.writable_write _ (hok k c hc)) hnd'.2 i a hi' ha
      | none => exact ih hok hnd'.2 i a hi' ha

end

end TVA
