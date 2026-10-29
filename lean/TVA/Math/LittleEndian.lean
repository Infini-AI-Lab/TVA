import Mathlib.Tactic

/-!
# Bytes and the little-endian encoding

Global memory holds bytes (`TVA.Lang.Mem`). An element of `k` bytes is its bit pattern `v`
(a natural number below `256^k`) split into `k` bytes, least significant first: byte `i` is
`⌊v / 256^i⌋ mod 256`. That is how NVIDIA GPUs and the host lay out an element in memory
(`evidence/float-layer`, part E).

* `Byte`: a byte, a natural number below `256`.
* `leBytes k v`: the `k` bytes of `v`, least significant first (`v mod 256^k`; higher bits are
  dropped).
* `leValue bs`: the number the bytes `bs` stand for, the first one least significant.
* `leBytes_length`, `leValue_leBytes`: `k` bytes, and they read back as `v mod 256^k`.
-/
namespace TVA.Math.Bytes

/-- A byte. -/
abbrev Byte := Fin 256

/-- The `k` bytes of `v`, least significant first. -/
def leBytes : Nat → Nat → List Byte
  | 0, _ => []
  | k + 1, v => ⟨v % 256, Nat.mod_lt _ (by norm_num)⟩ :: leBytes k (v / 256)

/-- The number a list of bytes stands for, the first byte least significant. -/
def leValue : List Byte → Nat
  | [] => 0
  | b :: bs => b.val + 256 * leValue bs

@[simp] theorem leBytes_length : ∀ (k v : Nat), (leBytes k v).length = k
  | 0, _ => rfl
  | k + 1, v => by simp [leBytes, leBytes_length k (v / 256)]

/-- The bytes of `v` read back as `v mod 256^k`. -/
theorem leValue_leBytes : ∀ (k v : Nat), leValue (leBytes k v) = v % 256 ^ k
  | 0, v => by simp [leBytes, leValue, Nat.mod_one]
  | k + 1, v => by
    simp only [leBytes, leValue, leValue_leBytes k (v / 256)]
    rw [pow_succ, mul_comm (256 ^ k) 256, Nat.mod_mul]

/-- Byte `i` of the encoding is `⌊v / 256^i⌋ mod 256`. -/
theorem leBytes_get : ∀ (k v i : Nat) (h : i < (leBytes k v).length),
    ((leBytes k v)[i]'h).val = v / 256 ^ i % 256
  | 0, _, _, h => by simp at h
  | k + 1, v, 0, _ => by simp [leBytes]
  | k + 1, v, i + 1, h => by
    simp only [leBytes, List.getElem_cons_succ]
    rw [leBytes_get k (v / 256) i (by simpa using h), Nat.div_div_eq_div_mul, pow_succ, mul_comm]

end TVA.Math.Bytes
