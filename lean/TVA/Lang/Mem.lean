import TVA.Lang.Tensor

/-!
# Memory and pointers

Global memory is a partial map from byte addresses to bytes. A byte is `none` when the launch
was not given it: the kernel may not access it. Every byte the caller supplies, including an
output buffer's initial contents, is `some`.

An element of dtype `d` occupies `d.bytes` bytes at an address that is a multiple of `d.bytes`,
in little-endian order (`TVA.Math.Bytes`). An access is valid when it is aligned, lies in
`[0, 2^64)`, and every byte it touches is present. An invalid access is undefined behaviour (it
may fault or return garbage on the device).

A pointer `Ptr d` is a byte address together with its element type. `p + i` (Triton's pointer
arithmetic, `x_ptr + offsets`) advances by `i` elements. The address is a mathematical integer:
the device's 64-bit wrap-around only happens for addresses that are invalid here anyway.
-/
namespace TVA

open TVA.Math.Bytes

/-- Global memory. -/
abbrev Mem := Int → Option Byte

/-- A pointer to elements of dtype `d` (`tl.pointer_type(d)`). -/
structure Ptr (d : DType) where
  addr : Int
  deriving DecidableEq, Repr

instance {d : DType} : Inhabited (Ptr d) := ⟨⟨0⟩⟩

namespace Ptr

variable {d : DType}

/-- `p + i`: `i` elements further. -/
def offset (p : Ptr d) (i : Int) : Ptr d := ⟨p.addr + i * d.bytes⟩

/-- The bytes an element at `p` occupies. -/
def Bytes (p : Ptr d) : Set Int := {a | p.addr ≤ a ∧ a < p.addr + d.bytes}

@[simp] theorem offset_addr (p : Ptr d) (i : Int) : (p.offset i).addr = p.addr + i * d.bytes := rfl

end Ptr

/-- `ptr + offsets` for a scalar pointer and an integer offset tensor: each offset is
sign-extended (signed types) or zero-extended (unsigned) to 64 bits, i.e. taken at its value. -/
def Tensor.ptrAdd {s : Shape} {d : DType} {sg : Bool} {w : Nat} (p : Ptr d)
    (offs : Tensor s (IntV sg w)) : Tensor s (Ptr d) :=
  offs.map (fun o => p.offset o.toInt)

/-- `ptrs + offsets` for a pointer tensor. -/
def Tensor.ptrAddT {s : Shape} {d : DType} {sg : Bool} {w : Nat} (ps : Tensor s (Ptr d))
    (offs : Tensor s (IntV sg w)) : Tensor s (Ptr d) :=
  Tensor.map2 (fun p o => p.offset o.toInt) ps offs

namespace Mem

/-- The `n` bytes at `a`, `a+1`, …, if all are present. -/
def readBytes (m : Mem) (a : Int) : Nat → Option (List Byte)
  | 0 => some []
  | n + 1 => match m a, readBytes m (a + 1) n with
    | some b, some bs => some (b :: bs)
    | _, _ => none

/-- An access of an element of dtype `d` at byte address `a` is aligned and inside `[0, 2^64)`. -/
def Aligned (d : DType) (a : Int) : Prop :=
  0 ≤ a ∧ a + d.bytes ≤ 2 ^ 64 ∧ a % d.bytes = 0

instance (d : DType) (a : Int) : Decidable (Aligned d a) := by unfold Aligned; infer_instance

/-- Read the element of dtype `d` at `a`: `none` when the access is invalid. -/
def read (m : Mem) (d : DType) (a : Int) : Option d.Val :=
  if Aligned d a then (m.readBytes a d.bytes).map (fun bs => d.ofBits (leValue bs)) else none

/-- An element of dtype `d` can be written at `a`: the access is valid. -/
def Writable (m : Mem) (d : DType) (a : Int) : Prop :=
  Aligned d a ∧ ∀ i : Nat, i < d.bytes → m (a + i) ≠ none

/-- Replace the bytes at `a`, `a+1`, … by `bs`. -/
def writeBytes (m : Mem) (a : Int) (bs : List Byte) : Mem :=
  fun x => if a ≤ x ∧ x < a + bs.length then bs[(x - a).toNat]? else m x

/-- Write `v` (dtype `d`) at `a`. -/
def write (m : Mem) (d : DType) (a : Int) (v : d.Val) : Mem :=
  m.writeBytes a (leBytes d.bytes (d.toBits v))

/-- The bytes an access of dtype `d` at `a` touches. -/
def span (d : DType) (a : Int) : Set Int := {x | a ≤ x ∧ x < a + d.bytes}

end Mem

end TVA

namespace TVA

/-- `ptr + offsets` (a scalar pointer plus an integer tensor) in source syntax. -/
instance {s : Shape} {d : DType} {sg : Bool} {w : Nat} :
    HAdd (Ptr d) (Tensor s (IntV sg w)) (Tensor s (Ptr d)) := ⟨Tensor.ptrAdd⟩

/-- `ptrs + offsets` (a pointer tensor plus an integer tensor). -/
instance {s : Shape} {d : DType} {sg : Bool} {w : Nat} :
    HAdd (Tensor s (Ptr d)) (Tensor s (IntV sg w)) (Tensor s (Ptr d)) := ⟨Tensor.ptrAddT⟩

/-- `ptr + k` for a scalar pointer and a scalar integer (`x_ptr + row * stride`). -/
instance {d : DType} {sg : Bool} {w : Nat} : HAdd (Ptr d) (IntV sg w) (Ptr d) := ⟨fun p k => p.offset k.toInt⟩

@[simp] theorem Ptr.add_int {d : DType} {sg : Bool} {w : Nat} (p : Ptr d) (k : IntV sg w) :
    p + k = p.offset k.toInt := rfl

@[simp] theorem Ptr.offset_offset {d : DType} (p : Ptr d) (i j : Int) :
    (p.offset i).offset j = p.offset (i + j) := by
  simp only [Ptr.offset]; congr 1; rw [Int.add_mul, Int.add_assoc]

@[simp] theorem Tensor.get_ptrAddT {s : Shape} {d : DType} {sg : Bool} {w : Nat} (ps : Tensor s (Ptr d))
    (offs : Tensor s (IntV sg w)) (i : Idx s) : (ps + offs).get i = (ps.get i).offset (offs.get i).toInt := rfl

@[simp] theorem Tensor.get_ptrAdd {s : Shape} {d : DType} {sg : Bool} {w : Nat} (p : Ptr d)
    (offs : Tensor s (IntV sg w)) (i : Idx s) : (p + offs).get i = p.offset (offs.get i).toInt := rfl

end TVA
