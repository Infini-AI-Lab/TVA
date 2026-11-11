import TVA.Launch
import TVA.Triton

/-!
# Memory lemmas: reads, writes and the accesses of one step

The facts about `Mem` (`TVA.Memory`) and about the accesses of a `tl.load` or `tl.store`
(`TVA.Launch.Step`) that every per-program proof uses:

* **Locality.** A read depends only on the bytes it touches (`Mem.read_congr`); a write changes
  only the bytes it touches (`Mem.write_frame`); memories that agree on some bytes still agree
  there after the same write (`Mem.write_agree`). The same for a whole tensor access:
  `loadOk_congr`, `loadBad_congr`, `storeOk_congr`, `writeAll_frame`, `writeAll_agree`.
* **Read after write.** `Mem.read_write_self` (a write is read back) and `Mem.read_write_other`
  (a write does not disturb an element it does not overlap); a writable element can be read
  (`Mem.read_some_of_writable`).
* **Presence.** A write keeps every present byte present (`Mem.write_present`,
  `Mem.writable_write`, `writeAll_present`, `writeAll_writable`).
* **Stores to distinct elements.** When no two enabled lanes overlap (`DistinctLanes`), a store
  reads back lane by lane in any order of its lanes (`writeAll_read`).

How this fits: a model is a `Prog`; a proof shows each program `Safe` (`TVA.Proof.Safe`) by the
rules of `TVA.Proof.Rules`, and composes the programs with `correct_of_safe`
(`TVA.Proof.Compose`). The lemmas here are what those steps use to move facts between memories.
Buffers (`Buf`) are in `TVA.Proof.Buf`.
-/
namespace TVA

open Bytes

/-! ## Encoding -/

/-- A bit pattern is below `256 ^ d.bytes`: it fits in the element's bytes. Use it with
`DType.ofBits_toBits` to read back a written value. -/
theorem DType.toBits_lt (d : DType) (v : d.Val) : d.toBits v < 256 ^ d.bytes := by
  cases d <;> simp only [DType.toBits, DType.bytes]
  case i1 => cases v <;> simp
  all_goals
    (have := v.bits.isLt
     simpa [FloatV.pat, Format.width, Format.binary16, Format.bfloat16, Format.binary32,
       Format.binary64, Format.e5m2, Format.e4m3] using this)

/-! ## Locality -/

namespace Mem

/-- `readBytes` depends only on the bytes it reads. -/
theorem readBytes_congr {m₁ m₂ : Mem} : ∀ {a : Int} {n : Nat},
    (∀ i : Nat, i < n → m₁ (a + i) = m₂ (a + i)) → m₁.readBytes a n = m₂.readBytes a n
  | _, 0, _ => rfl
  | a, n + 1, h => by
    have h0 : m₁ a = m₂ a := by simpa using h 0 (by omega)
    have hr : m₁.readBytes (a + 1) n = m₂.readBytes (a + 1) n :=
      readBytes_congr (fun i hi => by
        have := h (i + 1) (by omega); push_cast at this
        rwa [show a + 1 + i = a + (i + 1) by ring])
    simp only [readBytes, h0, hr]

/-- Byte `a + i` of an element at `a` is in its span when `i < d.bytes`. -/
theorem mem_span {d : DType} {a : Int} {i : Nat} (hi : i < d.bytes) : a + i ∈ span d a := by
  simp only [span, Set.mem_ofPred_eq]; omega

/-- **A read depends only on the bytes it touches.** Use it to carry a read from one memory to
another that agrees on the element's span. -/
theorem read_congr {m₁ m₂ : Mem} {d : DType} {a : Int} (h : ∀ x ∈ span d a, m₁ x = m₂ x) :
    m₁.read d a = m₂.read d a := by
  unfold read
  rw [readBytes_congr (fun i hi => h _ (mem_span hi))]

/-- Whether an element can be written depends only on its bytes. -/
theorem writable_congr {m₁ m₂ : Mem} {d : DType} {a : Int} (h : ∀ x ∈ span d a, m₁ x = m₂ x) :
    m₁.Writable d a ↔ m₂.Writable d a := by
  unfold Writable
  constructor
  · rintro ⟨ha, hb⟩; exact ⟨ha, fun i hi => by rw [← h _ (mem_span hi)]; exact hb i hi⟩
  · rintro ⟨ha, hb⟩; exact ⟨ha, fun i hi => by rw [h _ (mem_span hi)]; exact hb i hi⟩

/-- **A write changes only the bytes it touches.** -/
theorem write_frame (m : Mem) (d : DType) (a : Int) (v : d.Val) {x : Int} (hx : x ∉ span d a) :
    m.write d a v x = m x := by
  unfold write writeBytes
  rw [leBytes_length]
  simp only [span, Set.mem_ofPred_eq] at hx
  simp [hx]

/-- On its own bytes, a write does not depend on the memory written. -/
theorem write_span_eq (m m' : Mem) (d : DType) (a : Int) (v : d.Val) {x : Int}
    (hx : x ∈ span d a) : m.write d a v x = m'.write d a v x := by
  unfold write writeBytes
  rw [leBytes_length]
  simp only [span, Set.mem_ofPred_eq] at hx
  simp [hx]

/-- Memories that agree on a set of bytes agree on it after the same write. -/
theorem write_agree {m₁ m₂ : Mem} {S : Set Int} (h : ∀ x ∈ S, m₁ x = m₂ x) (d : DType) (a : Int)
    (v : d.Val) : ∀ x ∈ S, m₁.write d a v x = m₂.write d a v x := by
  intro x hx
  unfold write writeBytes
  split_ifs
  · rfl
  · exact h x hx

/-- Reading back the bytes just written. -/
theorem readBytes_writeBytes (m : Mem) : ∀ (a : Int) (bs : List Byte),
    (m.writeBytes a bs).readBytes a bs.length = some bs
  | _, [] => rfl
  | a, b :: bs => by
    have h0 : m.writeBytes a (b :: bs) a = some b := by simp [writeBytes]
    have hr : (m.writeBytes a (b :: bs)).readBytes (a + 1) bs.length =
        (m.writeBytes (a + 1) bs).readBytes (a + 1) bs.length := by
      apply readBytes_congr
      intro i hi
      simp only [writeBytes, List.length_cons]
      rw [ite_eq_left (by omega), ite_eq_left (by omega)]
      rw [show (a + 1 + i - a).toNat = i + 1 by omega, show (a + 1 + i - (a + 1)).toNat = i by omega]
      simp
    simp only [List.length_cons, readBytes, h0, hr, readBytes_writeBytes m (a + 1) bs]

/-- **A write is read back** at its own address, when the element is writable. -/
theorem read_write_self {m : Mem} {d : DType} {a : Int} (hw : m.Writable d a) (v : d.Val) :
    (m.write d a v).read d a = some v := by
  obtain ⟨hal, -⟩ := hw
  unfold read
  rw [ite_eq_left hal]
  have hbytes : (m.write d a v).readBytes a d.bytes = some (leBytes d.bytes (d.toBits v)) := by
    have := readBytes_writeBytes m a (leBytes d.bytes (d.toBits v))
    rwa [leBytes_length] at this
  rw [hbytes]
  simp only [Option.map_some, leValue_leBytes]
  rw [Nat.mod_eq_of_lt (DType.toBits_lt d v), DType.ofBits_toBits]

/-- **A write does not disturb a read** of an element it does not overlap. -/
theorem read_write_other {m : Mem} {d d' : DType} {a b : Int} (v : d.Val)
    (h : Disjoint (span d a) (span d' b)) : (m.write d a v).read d' b = m.read d' b := by
  apply read_congr
  intro x hx
  exact write_frame m d a v (Set.disjoint_right.mp h hx)

/-- An element whose bytes are present at an aligned address can be read. -/
theorem read_some_of_writable {m : Mem} {d : DType} {a : Int} (hw : m.Writable d a) :
    ∃ v, m.read d a = some v := by
  unfold read
  rw [ite_eq_left hw.1]
  have key : ∀ (n : Nat) (a : Int), (∀ k : Nat, k < n → m (a + k) ≠ none) →
      ∃ bs, m.readBytes a n = some bs := by
    intro n
    induction n with
    | zero => intro a _; exact ⟨[], rfl⟩
    | succ n ih =>
      intro a ha
      obtain ⟨b0, hb0⟩ := Option.ne_none_iff_exists'.mp (by simpa using ha 0 (by omega))
      obtain ⟨bs, hbs⟩ := ih (a + 1) (fun k hk => by
        have := ha (k + 1) (by omega); push_cast at this
        rwa [show a + 1 + k = a + (k + 1) by ring])
      exact ⟨b0 :: bs, by simp [readBytes, hb0, hbs]⟩
  obtain ⟨bs, hbs⟩ := key _ _ hw.2
  exact ⟨_, by rw [hbs]; rfl⟩

/-- A write keeps a present byte present. -/
theorem write_present {m : Mem} {d : DType} {a : Int} (v : d.Val) {x : Int} (hx : m x ≠ none) :
    m.write d a v x ≠ none := by
  unfold write writeBytes
  rw [leBytes_length]
  split_ifs with h
  · have hlt : (x - a).toNat < (leBytes d.bytes (d.toBits v)).length := by
      rw [leBytes_length]; omega
    rw [List.getElem?_eq_getElem hlt]; simp
  · exact hx

/-- A write keeps a writable element writable. -/
theorem writable_write {m : Mem} {d d' : DType} {a b : Int} (v : d.Val) (h : m.Writable d' b) :
    (m.write d a v).Writable d' b :=
  ⟨h.1, fun i hi => write_present v (h.2 i hi)⟩

end Mem

/-! ## Tensor accesses -/

section Access
variable {d : DType} {s : Shape}

/-- An enabled lane's element lies in the access's bytes. -/
theorem span_sub_accBytes {addrs : Idx s → Option Int} {i : Idx s} {a : Int}
    (h : addrs i = some a) : Mem.span d a ⊆ accBytes d addrs := fun _ hx => ⟨i, a, h, hx⟩

/-- What a load may return depends only on the bytes it reads. -/
theorem loadOk_congr {m₁ m₂ : Mem} {addrs : Idx s → Option Int} {v : Tensor s d.Val}
    (h : ∀ x ∈ accBytes d addrs, m₁ x = m₂ x) : LoadOk m₁ d addrs v ↔ LoadOk m₂ d addrs v := by
  unfold LoadOk
  have key : ∀ i a, addrs i = some a → m₁.read d a = m₂.read d a :=
    fun i a ha => Mem.read_congr (fun x hx => h x (span_sub_accBytes ha hx))
  constructor
  · intro H i a ha; rw [← key i a ha]; exact H i a ha
  · intro H i a ha; rw [key i a ha]; exact H i a ha

/-- Whether a load is undefined depends only on the bytes it reads. -/
theorem loadBad_congr {m₁ m₂ : Mem} {addrs : Idx s → Option Int}
    (h : ∀ x ∈ accBytes d addrs, m₁ x = m₂ x) : LoadBad m₁ d addrs ↔ LoadBad m₂ d addrs := by
  unfold LoadBad
  have key : ∀ i a, addrs i = some a → m₁.read d a = m₂.read d a :=
    fun i a ha => Mem.read_congr (fun x hx => h x (span_sub_accBytes ha hx))
  constructor
  · rintro ⟨i, a, ha, hr⟩; exact ⟨i, a, ha, by rw [← key i a ha]; exact hr⟩
  · rintro ⟨i, a, ha, hr⟩; exact ⟨i, a, ha, by rw [key i a ha]; exact hr⟩

/-- Whether a store can write depends only on the bytes it writes. -/
theorem storeOk_congr {m₁ m₂ : Mem} {addrs : Idx s → Option Int}
    (h : ∀ x ∈ accBytes d addrs, m₁ x = m₂ x) : StoreOk m₁ d addrs ↔ StoreOk m₂ d addrs := by
  unfold StoreOk
  have key : ∀ i a, addrs i = some a → (m₁.Writable d a ↔ m₂.Writable d a) :=
    fun i a ha => Mem.writable_congr (fun x hx => h x (span_sub_accBytes ha hx))
  constructor
  · intro H i a ha; exact (key i a ha).mp (H i a ha)
  · intro H i a ha; exact (key i a ha).mpr (H i a ha)

/-- Memories that agree on a set agree on it after the same store. -/
theorem writeAll_agree {m₁ m₂ : Mem} {S : Set Int} (h : ∀ x ∈ S, m₁ x = m₂ x)
    (addrs : Idx s → Option Int) (vals : Tensor s d.Val) (order : List (Idx s)) :
    ∀ x ∈ S, writeAll m₁ d addrs vals order x = writeAll m₂ d addrs vals order x := by
  induction order generalizing m₁ m₂ with
  | nil => exact h
  | cons i rest ih =>
    simp only [writeAll]
    cases addrs i with
    | some a => exact ih (Mem.write_agree h d _ _)
    | none => exact ih h

/-- **A store changes only the bytes of its enabled lanes.** -/
theorem writeAll_frame (m : Mem) (addrs : Idx s → Option Int) (vals : Tensor s d.Val)
    (order : List (Idx s)) {x : Int} (hx : x ∉ accBytes d addrs) :
    writeAll m d addrs vals order x = m x := by
  induction order generalizing m with
  | nil => rfl
  | cons i rest ih =>
    simp only [writeAll]
    cases ha : addrs i with
    | some a =>
      simp only
      rw [ih]
      exact Mem.write_frame m d a _ (fun hm => hx (span_sub_accBytes ha hm))
    | none => exact ih m

/-- A store keeps a writable element writable. -/
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

/-- Writes of lanes whose elements avoid `S` leave `S` alone. -/
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
        (havoid k List.mem_cons_self b hk) hxb hx)
    | none =>
      exact writeAll_frame_of (fun k' hk' b' hb' => havoid k' (List.mem_cons_of_mem _ hk') b' hb') x hx

/-- The enabled lanes of an access touch pairwise disjoint bytes: a store's lanes then land in
any order with the same result. -/
def DistinctLanes (d : DType) (addrs : Idx s → Option Int) : Prop :=
  ∀ i j a b, i ≠ j → addrs i = some a → addrs j = some b → Disjoint (Mem.span d a) (Mem.span d b)

/-- The list of enabled lanes has no duplicate. -/
theorem enabled_nodup (addrs : Idx s → Option Int) : (enabled addrs).Nodup :=
  (Idx.nodup_all s).filter _

/-- A lane with an address is enabled. -/
theorem mem_enabled {addrs : Idx s → Option Int} {i : Idx s} {a : Int} (h : addrs i = some a) :
    i ∈ enabled addrs := by
  simp [enabled, Idx.mem_all, h]

/-- **A store to distinct elements reads back**: in any order of its lanes, each enabled lane's
element holds its value afterwards. -/
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

end Access

end TVA
