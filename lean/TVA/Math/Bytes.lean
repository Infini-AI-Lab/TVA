/-!
# Little-endian byte strings

The number format of memory: a byte is an 8-bit pattern (`Byte`), and a natural number is held
in `n` bytes as its little-endian encoding (`leBytes n v`, the `n` bytes of `v mod 256^n`, least
significant first); `leValue` reads a byte string back as a number. The memory model
(`TVA.Lang.Mem`) holds values as these bytes, and every level reads multi-byte values out of
memory through them.

Facts:

* `leBytes_length`, `leValue_leBytes`, `leValue_lt`: `n` bytes, which read back as `v mod 256^n`;
  a string of `k` bytes reads as a number below `256^k`;
* `leBytes_length_leValue`, `leBytes_leValue_of_length`: a byte string is the encoding of its own
  value;
* `leBytes_add_eq`, `leBytes_concat`: the encoding of `m + n` bytes splits into the low `m` and
  the high `n` bytes (two halves packed into one word, e.g. two bf16 in a `uint32_t`:
  `leBytes_pair16`);
* `leBytes_mod_pow`: only the value modulo `256^n` matters;
* `leBytes_toNat_ofNat_leValue`, `ofNat_leValue_leBytes_toNat`: four bytes as a 32-bit pattern
  (an `f32`) and back.
-/
namespace TVA.Math.Bytes

/-- A byte: an 8-bit pattern. -/
abbrev Byte := BitVec 8

/-- The `n` little-endian bytes of `v mod 256^n`. -/
def leBytes : Nat → Nat → List Byte
  | 0, _ => []
  | n + 1, v => BitVec.ofNat 8 v :: leBytes n (v / 256)

/-- The natural number with little-endian bytes `bs`. -/
def leValue : List Byte → Nat
  | [] => 0
  | b :: bs => b.toNat + 256 * leValue bs

theorem leBytes_length (n v : Nat) : (leBytes n v).length = n := by
  induction n generalizing v with
  | zero => rfl
  | succ n ih => simp [leBytes, ih]

theorem leValue_leBytes (n v : Nat) : leValue (leBytes n v) = v % 256 ^ n := by
  induction n generalizing v with
  | zero => simp [leBytes, leValue, Nat.mod_one]
  | succ n ih =>
    simp only [leBytes, leValue, ih, BitVec.toNat_ofNat]
    rw [show 256 ^ (n + 1) = 256 * 256 ^ n by rw [Nat.pow_succ, Nat.mul_comm], Nat.mod_mul]

theorem leValue_lt (bs : List Byte) : leValue bs < 256 ^ bs.length := by
  induction bs with
  | nil => simp [leValue]
  | cons b bs ih =>
    simp only [leValue, List.length_cons, Nat.pow_succ]
    have := b.isLt
    simp only [Nat.reducePow] at this
    omega

/-- A byte string is the little-endian encoding of its own value. -/
theorem leBytes_length_leValue : ∀ bs : List Byte, leBytes bs.length (leValue bs) = bs
  | [] => rfl
  | b :: bs => by
    simp only [List.length_cons, leBytes, leValue]
    have hb := b.isLt
    have h1 : (b.toNat + 256 * leValue bs) / 256 = leValue bs := by omega
    rw [h1, leBytes_length_leValue bs]
    congr 1
    apply BitVec.eq_of_toNat_eq
    simp only [BitVec.toNat_ofNat]
    omega

/-- `leBytes_length_leValue` with the length given. -/
theorem leBytes_leValue_of_length {bs : List Byte} {n : Nat} (h : bs.length = n) :
    leBytes n (leValue bs) = bs := by
  subst h; exact leBytes_length_leValue bs

/-- The encoding of `m + n` bytes: the low `m` bytes, then the high `n` bytes. -/
theorem leBytes_add_eq (m n v : Nat) :
    leBytes (m + n) v = leBytes m v ++ leBytes n (v / 256 ^ m) := by
  induction m generalizing v with
  | zero => simp [leBytes]
  | succ m ih =>
    rw [show m + 1 + n = (m + n) + 1 by omega]
    simp only [leBytes, ih, List.cons_append, Nat.pow_succ, Nat.div_div_eq_div_mul]
    rw [Nat.mul_comm (256 ^ m) 256]

/-- Only the value modulo `256^n` is encoded. -/
theorem leBytes_mod_pow (n v : Nat) : leBytes n (v % 256 ^ n) = leBytes n v := by
  induction n generalizing v with
  | zero => rfl
  | succ n ih =>
    simp only [leBytes]
    congr 1
    · apply BitVec.eq_of_toNat_eq
      simp only [BitVec.toNat_ofNat]
      rw [Nat.pow_succ, Nat.mod_mul_left_mod]
    · rw [← ih (v / 256), ← ih ((v % 256 ^ (n + 1)) / 256)]
      congr 1
      rw [Nat.pow_succ, Nat.mul_comm, Nat.mod_mul_right_div_self, Nat.mod_mod]

/-- Two encodings side by side are the encoding of the packed value. -/
theorem leBytes_concat {n m x y : Nat} (hx : x < 256 ^ n) :
    leBytes (n + m) (x + 256 ^ n * y) = leBytes n x ++ leBytes m y := by
  rw [leBytes_add_eq, ← leBytes_mod_pow n (x + 256 ^ n * y), Nat.add_mul_mod_self_left,
    Nat.mod_eq_of_lt hx, Nat.add_mul_div_left _ _ (Nat.pow_pos (by decide)), Nat.div_eq_of_lt hx, Nat.zero_add]

/-- Two 16-bit halves, the low one first, as one 32-bit word. -/
theorem leBytes_pair16 {x y : Nat} (hx : x < 65536) :
    leBytes 2 x ++ leBytes 2 y = leBytes 4 (x + 65536 * y) :=
  (leBytes_concat (n := 2) (m := 2) (y := y) (show x < 256 ^ 2 by omega)).symm

/-- Four bytes read as a 32-bit pattern encode back to themselves. -/
theorem leBytes_toNat_ofNat_leValue {bs : List Byte} (h : bs.length = 4) :
    leBytes 4 (BitVec.ofNat 32 (leValue bs)).toNat = bs := by
  have hl := leValue_lt bs
  rw [h] at hl
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by simpa using hl)]
  exact leBytes_leValue_of_length h

/-- A 32-bit pattern encoded in four bytes reads back as itself. -/
theorem ofNat_leValue_leBytes_toNat (x : BitVec 32) :
    BitVec.ofNat 32 (leValue (leBytes 4 x.toNat)) = x := by
  rw [leValue_leBytes]
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat]
  have := x.isLt
  omega

end TVA.Math.Bytes
