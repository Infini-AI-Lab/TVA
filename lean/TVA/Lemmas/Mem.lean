import TVA.Sem.Step
import Mathlib.Tactic

/-!
# Memory lemmas

* Encoding: every dtype's value survives the trip through bytes (`DType.ofBits_toBits`,
  `DType.toBits_lt`).
* Locality: a read depends only on the bytes it touches (`Mem.read_congr`); a write changes
  only the bytes it touches (`Mem.write_frame`); agreement on a set of bytes survives the same
  write (`Mem.write_agree`, `writeAll_agree`).
* Read after write: `Mem.read_write_self`, and for elements that do not overlap
  `Mem.read_write_other`.
-/
namespace TVA

open TVA.Math.Bytes TVA.Math.Float

/-! ## Encoding -/

theorem FloatV.ofPat_pat {f : Format} (x : FloatV f) : FloatV.ofPat x.pat = x := by
  cases x; simp [FloatV.ofPat, FloatV.pat]

theorem IntV.ofNat_toNat {s : Bool} {w : Nat} (x : IntV s w) : (IntV.ofNat x.bits.toNat : IntV s w) = x := by
  cases x; simp [IntV.ofNat]

theorem DType.ofBits_toBits (d : DType) (v : d.Val) : d.ofBits (d.toBits v) = v := by
  cases d <;> first
    | (cases v <;> rfl)
    | exact IntV.ofNat_toNat v
    | exact FloatV.ofPat_pat v

theorem DType.toBits_lt (d : DType) (v : d.Val) : d.toBits v < 256 ^ d.bytes := by
  cases d <;> simp only [DType.toBits, DType.bytes]
  case i1 => cases v <;> simp
  all_goals
    (have := v.bits.isLt; simpa [FloatV.pat, Format.width, Format.binary16, Format.bfloat16,
        Format.binary32, Format.binary64, Format.e5m2, Format.e4m3] using this)

/-! ## Locality -/

namespace Mem

theorem readBytes_congr {m₁ m₂ : Mem} : ∀ {a : Int} {n : Nat},
    (∀ i : Nat, i < n → m₁ (a + i) = m₂ (a + i)) → m₁.readBytes a n = m₂.readBytes a n
  | _, 0, _ => rfl
  | a, n + 1, h => by
    have h0 : m₁ a = m₂ a := by simpa using h 0 (by omega)
    have hr : m₁.readBytes (a + 1) n = m₂.readBytes (a + 1) n :=
      readBytes_congr (fun i hi => by
        have := h (i + 1) (by omega); push_cast at this; rwa [show a + 1 + i = a + (i + 1) by ring])
    simp only [readBytes, h0, hr]

theorem mem_span {d : DType} {a : Int} {i : Nat} (hi : i < d.bytes) : a + i ∈ span d a := by
  simp only [span, Set.mem_ofPred_eq]; omega

/-- A read depends only on the bytes it touches. -/
theorem read_congr {m₁ m₂ : Mem} {d : DType} {a : Int} (h : ∀ x ∈ span d a, m₁ x = m₂ x) :
    m₁.read d a = m₂.read d a := by
  unfold read
  rw [readBytes_congr (fun i hi => h _ (mem_span hi))]

theorem writable_congr {m₁ m₂ : Mem} {d : DType} {a : Int} (h : ∀ x ∈ span d a, m₁ x = m₂ x) :
    m₁.Writable d a ↔ m₂.Writable d a := by
  unfold Writable
  constructor
  · rintro ⟨ha, hb⟩; exact ⟨ha, fun i hi => by rw [← h _ (mem_span hi)]; exact hb i hi⟩
  · rintro ⟨ha, hb⟩; exact ⟨ha, fun i hi => by rw [h _ (mem_span hi)]; exact hb i hi⟩

theorem leBytes_len (d : DType) (v : d.Val) : (leBytes d.bytes (d.toBits v)).length = d.bytes :=
  leBytes_length _ _

/-- A write changes only the bytes it touches. -/
theorem write_frame (m : Mem) (d : DType) (a : Int) (v : d.Val) {x : Int} (hx : x ∉ span d a) :
    m.write d a v x = m x := by
  unfold write writeBytes
  rw [leBytes_len]
  simp only [span, Set.mem_ofPred_eq] at hx
  simp [hx]

/-- Memories that agree on a set agree on it after the same write. -/
theorem write_agree {m₁ m₂ : Mem} {S : Set Int} (h : ∀ x ∈ S, m₁ x = m₂ x) (d : DType) (a : Int)
    (v : d.Val) : ∀ x ∈ S, m₁.write d a v x = m₂.write d a v x := by
  intro x hx
  unfold write writeBytes
  split_ifs
  · rfl
  · exact h x hx

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
      rw [if_pos (by omega), if_pos (by omega)]
      rw [show (a + 1 + i - a).toNat = i + 1 by omega, show (a + 1 + i - (a + 1)).toNat = i by omega]
      simp
    simp only [List.length_cons, readBytes, h0, hr, readBytes_writeBytes m (a + 1) bs]

/-- A write is visible at its own address. -/
theorem read_write_self {m : Mem} {d : DType} {a : Int} (hw : m.Writable d a) (v : d.Val) :
    (m.write d a v).read d a = some v := by
  obtain ⟨hal, -⟩ := hw
  unfold read
  rw [if_pos hal]
  have hbytes : (m.write d a v).readBytes a d.bytes = some (leBytes d.bytes (d.toBits v)) := by
    have := readBytes_writeBytes m a (leBytes d.bytes (d.toBits v))
    rwa [leBytes_len] at this
  rw [hbytes]
  simp only [Option.map_some, leValue_leBytes]
  rw [Nat.mod_eq_of_lt (DType.toBits_lt d v), DType.ofBits_toBits]

/-- A write does not disturb a read of bytes it does not touch. -/
theorem read_write_other {m : Mem} {d d' : DType} {a b : Int} (v : d.Val)
    (h : Disjoint (span d a) (span d' b)) : (m.write d a v).read d' b = m.read d' b := by
  apply read_congr
  intro x hx
  exact write_frame m d a v (Set.disjoint_right.mp h hx)

end Mem

/-! ## Tensor accesses -/

section Access
variable {d : DType} {s : Shape}

theorem span_sub_accBytes {addrs : Idx s → Option Int} {i : Idx s} {a : Int} (h : addrs i = some a) :
    Mem.span d a ⊆ accBytes d addrs := fun x hx => ⟨i, a, h, hx⟩

theorem loadOk_congr {m₁ m₂ : Mem} {addrs : Idx s → Option Int} {v : Tensor s d.Val}
    (h : ∀ x ∈ accBytes d addrs, m₁ x = m₂ x) : LoadOk m₁ d addrs v ↔ LoadOk m₂ d addrs v := by
  unfold LoadOk
  have key : ∀ i a, addrs i = some a → m₁.read d a = m₂.read d a :=
    fun i a ha => Mem.read_congr (fun x hx => h x (span_sub_accBytes ha hx))
  constructor
  · intro H i a ha; rw [← key i a ha]; exact H i a ha
  · intro H i a ha; rw [key i a ha]; exact H i a ha

theorem loadBad_congr {m₁ m₂ : Mem} {addrs : Idx s → Option Int}
    (h : ∀ x ∈ accBytes d addrs, m₁ x = m₂ x) : LoadBad m₁ d addrs ↔ LoadBad m₂ d addrs := by
  unfold LoadBad
  have key : ∀ i a, addrs i = some a → m₁.read d a = m₂.read d a :=
    fun i a ha => Mem.read_congr (fun x hx => h x (span_sub_accBytes ha hx))
  constructor
  · rintro ⟨i, a, ha, hr⟩; exact ⟨i, a, ha, by rw [← key i a ha]; exact hr⟩
  · rintro ⟨i, a, ha, hr⟩; exact ⟨i, a, ha, by rw [key i a ha]; exact hr⟩

theorem storeOk_congr {m₁ m₂ : Mem} {addrs : Idx s → Option Int}
    (h : ∀ x ∈ accBytes d addrs, m₁ x = m₂ x) : StoreOk m₁ d addrs ↔ StoreOk m₂ d addrs := by
  unfold StoreOk
  have key : ∀ i a, addrs i = some a → (m₁.Writable d a ↔ m₂.Writable d a) :=
    fun i a ha => Mem.writable_congr (fun x hx => h x (span_sub_accBytes ha hx))
  constructor
  · intro H i a ha; exact (key i a ha).mp (H i a ha)
  · intro H i a ha; exact (key i a ha).mpr (H i a ha)

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

/-- Writing enabled lanes changes only the bytes they touch. -/
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

end Access

end TVA
