import TVA.Tensor

/-!
# Memory, pointers and buffers

Global memory is a partial map from byte addresses to bytes (`Mem`). A byte is `none` when the
launch was not given it: the kernel may not access it. Every byte the host supplies, an output
buffer's initial contents included, is `some`.

An element of dtype `d` occupies `d.bytes` bytes at an address that is a multiple of `d.bytes`,
least significant byte first (`TVA.Bytes`). An access is valid when it is aligned, lies in
`[0, 2^64)` and every byte it touches is present (`Mem.Aligned`, `Mem.read`, `Mem.Writable`); an
invalid access is undefined behaviour (it may fault or return garbage on the device).

A pointer `Ptr d` (`tl.pointer_type(d)`) is a byte address together with its element type.
Triton's pointer arithmetic is written as in the source: `p + offs` and `p - k` move by whole
elements, and each integer offset is taken at its value (sign-extended for a signed dtype,
zero-extended for an unsigned one). Addresses are mathematical integers: the device's 64-bit
wrap-around only happens at addresses that are invalid here anyway.

A buffer `Buf d` is a contiguous array as the host passes it (`x_ptr` with `n` elements);
specifications and footprints are stated with buffers (`Buf.ptr`, `Buf.elems`, `Buf.Valid`,
`Buf.Holds`).

Used by `TVA.Program` (loads and stores name byte addresses), `TVA.Triton.Memory` (`tl.load`,
`tl.store`, atomics) and every claim.
-/
namespace TVA

open Bytes

/-- Global memory: the byte at each address, or `none` where the launch was given no memory. -/
abbrev Mem := Int → Option Byte

/-- A pointer to elements of dtype `d` (`tl.pointer_type(d)`): a byte address. -/
structure Ptr (d : DType) where
  /-- The byte address. -/
  addr : Int
  deriving DecidableEq, Repr

instance {d : DType} : Inhabited (Ptr d) := ⟨⟨0⟩⟩

namespace Ptr

variable {d : DType}

/-- `p + i`: the pointer `i` elements further (`i * d.bytes` bytes). -/
def offset (p : Ptr d) (i : Int) : Ptr d := ⟨p.addr + i * d.bytes⟩

@[simp] theorem offset_addr (p : Ptr d) (i : Int) :
    (p.offset i).addr = p.addr + i * d.bytes := rfl

@[simp] theorem offset_offset (p : Ptr d) (i j : Int) :
    (p.offset i).offset j = p.offset (i + j) := by
  simp only [offset]; congr 1; rw [Int.add_mul, Int.add_assoc]

end Ptr

/-! ### Pointer arithmetic -/

/-- `ptr + offsets`: a scalar pointer plus an integer tensor, lane by lane, each offset taken at
its value (`semantic.py:244-250`). -/
def Tensor.ptrAdd {s : Shape} {d : DType} {sg : Bool} {w : Nat} (p : Ptr d)
    (offs : Tensor s (IntV sg w)) : Tensor s (Ptr d) :=
  offs.map (fun o => p.offset o.toInt)

/-- `ptrs + offsets`: a pointer tensor plus an integer tensor, lane by lane. -/
def Tensor.ptrAddT {s : Shape} {d : DType} {sg : Bool} {w : Nat} (ps : Tensor s (Ptr d))
    (offs : Tensor s (IntV sg w)) : Tensor s (Ptr d) :=
  Tensor.map2 (fun p o => p.offset o.toInt) ps offs

/-- `ptr + offsets` in source syntax. -/
instance {s : Shape} {d : DType} {sg : Bool} {w : Nat} :
    HAdd (Ptr d) (Tensor s (IntV sg w)) (Tensor s (Ptr d)) := ⟨Tensor.ptrAdd⟩

/-- `ptrs + offsets` in source syntax. -/
instance {s : Shape} {d : DType} {sg : Bool} {w : Nat} :
    HAdd (Tensor s (Ptr d)) (Tensor s (IntV sg w)) (Tensor s (Ptr d)) := ⟨Tensor.ptrAddT⟩

/-- `ptr + k` for a scalar pointer and a scalar integer (`x_ptr + row * stride`). -/
instance {d : DType} {sg : Bool} {w : Nat} : HAdd (Ptr d) (IntV sg w) (Ptr d) :=
  ⟨fun p k => p.offset k.toInt⟩

/-! Triton builds `p - k` as `p + (0 - k)`: the negation is computed in `k`'s own integer type,
wrapping (`semantic.py:261-267, 478-485`), and that value is then added as `+` does. So for a
signed `k` above the type's minimum, `p - k` is `k` elements back (`Ptr.sub_int_of_signed`), and
for an unsigned `k ≠ 0` it is `2^w − k` elements forward. -/

/-- `ptr - k` for a scalar pointer and a scalar integer: `p + (-k)`, `-k` wrapping in `k`'s
type. -/
instance {d : DType} {sg : Bool} {w : Nat} : HSub (Ptr d) (IntV sg w) (Ptr d) :=
  ⟨fun p k => p + (-k)⟩

/-- `ptr - offsets` (a scalar pointer minus an integer tensor), lane by lane. -/
instance {s : Shape} {d : DType} {sg : Bool} {w : Nat} :
    HSub (Ptr d) (Tensor s (IntV sg w)) (Tensor s (Ptr d)) :=
  ⟨fun p offs => p + offs.map (fun o => -o)⟩

/-- `ptrs - offsets` (a pointer tensor minus an integer tensor), lane by lane. -/
instance {s : Shape} {d : DType} {sg : Bool} {w : Nat} :
    HSub (Tensor s (Ptr d)) (Tensor s (IntV sg w)) (Tensor s (Ptr d)) :=
  ⟨fun ps offs => ps + offs.map (fun o => -o)⟩

@[simp] theorem Ptr.add_int {d : DType} {sg : Bool} {w : Nat} (p : Ptr d) (k : IntV sg w) :
    p + k = p.offset k.toInt := rfl

@[simp] theorem Ptr.sub_int {d : DType} {sg : Bool} {w : Nat} (p : Ptr d) (k : IntV sg w) :
    p - k = p.offset (-k).toInt := rfl

/-- For a signed `k` above the type's minimum, `p - k` is `k` elements back. -/
theorem Ptr.sub_int_of_signed {d : DType} {w : Nat} (p : Ptr d) (k : IntV true w)
    (hk : -(2 ^ (w - 1) : Int) < k.toInt) : p - k = p.offset (-k.toInt) := by
  rw [Ptr.sub_int]
  congr 1
  simp only [IntV.toInt, ite_true]
  show (-k.bits).toInt = -k.bits.toInt
  rcases Nat.eq_zero_or_pos w with hw | hw
  · subst hw; simp [BitVec.toInt_zero_length]
  · rw [BitVec.toInt_neg_of_ne_intMin]
    intro h
    have : k.bits.toInt = -(2 ^ (w - 1) : Int) := by
      rw [h, BitVec.toInt_intMin_of_pos hw]
    simp only [IntV.toInt, ite_true] at hk
    omega

@[simp] theorem Tensor.get_ptrAdd {s : Shape} {d : DType} {sg : Bool} {w : Nat} (p : Ptr d)
    (offs : Tensor s (IntV sg w)) (i : Idx s) :
    (p + offs).get i = p.offset (offs.get i).toInt := rfl

@[simp] theorem Tensor.get_ptrAddT {s : Shape} {d : DType} {sg : Bool} {w : Nat}
    (ps : Tensor s (Ptr d)) (offs : Tensor s (IntV sg w)) (i : Idx s) :
    (ps + offs).get i = (ps.get i).offset (offs.get i).toInt := rfl

/-! ### Reading and writing memory -/

namespace Mem

/-- The `n` bytes at `a`, `a + 1`, …, if all are present. -/
def readBytes (m : Mem) (a : Int) : Nat → Option (List Byte)
  | 0 => some []
  | n + 1 => match m a, readBytes m (a + 1) n with
    | some b, some bs => some (b :: bs)
    | _, _ => none

/-- An access to an element of dtype `d` at byte address `a` is aligned to `d.bytes` and lies
inside `[0, 2^64)`. -/
def Aligned (d : DType) (a : Int) : Prop :=
  0 ≤ a ∧ a + d.bytes ≤ 2 ^ 64 ∧ a % d.bytes = 0

instance (d : DType) (a : Int) : Decidable (Aligned d a) := by unfold Aligned; infer_instance

/-- The element of dtype `d` at `a`, or `none` when the access is invalid (unaligned, out of
range, or touching a byte that is not present). -/
def read (m : Mem) (d : DType) (a : Int) : Option d.Val :=
  if Aligned d a then (m.readBytes a d.bytes).map (fun bs => d.ofBits (leValue bs)) else none

/-- An element of dtype `d` can be written at `a`: the access is aligned and every byte it
touches is present. -/
def Writable (m : Mem) (d : DType) (a : Int) : Prop :=
  Aligned d a ∧ ∀ i : Nat, i < d.bytes → m (a + i) ≠ none

/-- The memory with the bytes at `a`, `a + 1`, … replaced by `bs`. -/
def writeBytes (m : Mem) (a : Int) (bs : List Byte) : Mem :=
  fun x => if a ≤ x ∧ x < a + bs.length then bs[(x - a).toNat]? else m x

/-- The memory with the element `v` of dtype `d` written at `a`. -/
def write (m : Mem) (d : DType) (a : Int) (v : d.Val) : Mem :=
  m.writeBytes a (leBytes d.bytes (d.toBits v))

/-- The bytes an access to an element of dtype `d` at `a` touches. -/
def span (d : DType) (a : Int) : Set Int := {x | a ≤ x ∧ x < a + d.bytes}

end Mem

/-! ### Buffers -/

/-- A contiguous buffer of `len` elements of dtype `d` starting at byte address `base`: a tensor
argument as the host passes it. The kernel argument is `b.ptr 0`. -/
structure Buf (d : DType) where
  /-- The address of element `0`. -/
  base : Int
  /-- The number of elements. -/
  len : Nat
  deriving DecidableEq, Repr

namespace Buf

variable {d : DType}

/-- The pointer to element `j` (`x_ptr + j`). -/
def ptr (b : Buf d) (j : Int) : Ptr d := ⟨b.base + j * d.bytes⟩

/-- The bytes of the whole buffer. -/
def bytes (b : Buf d) : Set Int := {x | b.base ≤ x ∧ x < b.base + b.len * d.bytes}

/-- The bytes of the elements `S` of the buffer (a footprint). -/
def elems (b : Buf d) (S : Set Nat) : Set Int :=
  {x | ∃ j ∈ S, x ∈ Mem.span d (b.ptr (j : Int)).addr}

/-- The buffer is aligned to its element size, lies inside `[0, 2^64)`, and all its bytes are
present in `m`. -/
def Valid (b : Buf d) (m : Mem) : Prop :=
  0 ≤ b.base ∧ b.base + b.len * d.bytes ≤ 2 ^ 64 ∧ b.base % d.bytes = 0 ∧
    ∀ x ∈ b.bytes, m x ≠ none

/-- Memory `m` holds `v j` at each element `j < len`. -/
def Holds (b : Buf d) (m : Mem) (v : Nat → d.Val) : Prop :=
  ∀ j : Nat, j < b.len → m.read d (b.ptr j).addr = some (v j)

/-- Two buffers, possibly of different dtypes, do not overlap. -/
def Disjoint' (b : Buf d) {d' : DType} (c : Buf d') : Prop := Disjoint b.bytes c.bytes

@[simp] theorem ptr_addr (b : Buf d) (j : Int) : (b.ptr j).addr = b.base + j * d.bytes := rfl

@[simp] theorem ptr_offset (b : Buf d) (i k : Int) : (b.ptr i).offset k = b.ptr (i + k) := by
  simp only [ptr, Ptr.offset]; congr 1; ring

/-- An element below `len` lies inside the buffer. -/
theorem span_sub (b : Buf d) {j : Nat} (hj : j < b.len) :
    Mem.span d (b.ptr j).addr ⊆ b.bytes := by
  intro x hx
  simp only [Mem.span, Set.mem_ofPred_eq, ptr_addr] at hx
  simp only [bytes, Set.mem_ofPred_eq]
  have hb := d.bytes_pos
  constructor
  · nlinarith
  · have : (j : Int) + 1 ≤ b.len := by exact_mod_cast hj
    nlinarith

/-- Distinct elements of a buffer do not overlap. -/
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

/-- Each element of a valid buffer is an aligned access. -/
theorem aligned (b : Buf d) {m : Mem} (hv : b.Valid m) {j : Nat} (hj : j < b.len) :
    Mem.Aligned d (b.ptr j).addr := by
  obtain ⟨h0, h1, h2, -⟩ := hv
  have hb := d.bytes_pos
  refine ⟨by simp; positivity, ?_, ?_⟩
  · simp only [ptr_addr]
    have : (j : Int) + 1 ≤ b.len := by exact_mod_cast hj
    nlinarith
  · simp only [ptr_addr]; rw [Int.add_mul_emod_self_right]; exact h2

theorem span_sub_elems (b : Buf d) {S : Set Nat} {j : Nat} (hj : j ∈ S) :
    Mem.span d (b.ptr (j : Int)).addr ⊆ b.elems S := fun _ hx => ⟨j, hj, hx⟩

theorem elems_mono (b : Buf d) {S T : Set Nat} (h : S ⊆ T) : b.elems S ⊆ b.elems T := by
  rintro x ⟨j, hj, hx⟩; exact ⟨j, h hj, hx⟩

theorem elems_union (b : Buf d) (S T : Set Nat) :
    b.elems (S ∪ T) = b.elems S ∪ b.elems T := by
  ext x; constructor
  · rintro ⟨j, hj | hj, hx⟩
    · exact Or.inl ⟨j, hj, hx⟩
    · exact Or.inr ⟨j, hj, hx⟩
  · rintro (⟨j, hj, hx⟩ | ⟨j, hj, hx⟩)
    · exact ⟨j, Or.inl hj, hx⟩
    · exact ⟨j, Or.inr hj, hx⟩

/-- Elements below `len` lie inside the buffer. -/
theorem elems_sub_bytes (b : Buf d) {S : Set Nat} (hS : ∀ j ∈ S, j < b.len) :
    b.elems S ⊆ b.bytes := by
  rintro x ⟨j, hj, hx⟩; exact b.span_sub (hS j hj) hx

/-- Disjoint sets of elements of one buffer occupy disjoint bytes. -/
theorem elems_disjoint (b : Buf d) {S T : Set Nat} (hST : Disjoint S T) :
    Disjoint (b.elems S) (b.elems T) := by
  rw [Set.disjoint_left]
  rintro x ⟨j, hj, hx⟩ ⟨j', hj', hx'⟩
  by_cases h : j = j'
  · subst h; exact Set.disjoint_left.mp hST hj hj'
  · exact Set.disjoint_left.mp (b.span_disjoint (by exact_mod_cast h)) hx hx'

/-- Elements of two non-overlapping buffers occupy disjoint bytes. -/
theorem elems_disjoint_of {d' : DType} (b : Buf d) (c : Buf d') (hbc : Disjoint b.bytes c.bytes)
    {S T : Set Nat} (hS : ∀ j ∈ S, j < b.len) (hT : ∀ j ∈ T, j < c.len) :
    Disjoint (b.elems S) (c.elems T) :=
  hbc.mono (b.elems_sub_bytes hS) (c.elems_sub_bytes hT)

end Buf

end TVA
