import TVA.Lang.Ops

/-!
# Random numbers (`tl.randint`, `tl.rand`)

Triton's generator is Philox 4x32 with 10 rounds, written in Triton itself
(`primitives/triton-3.8.0-random.py`). It is pure integer arithmetic, so the library defines it
exactly; only `uint_to_uniform_float` rounds (an integer-to-float conversion and an IEEE
product).

* `Philox.impl`: `philox_impl` on `uint32` state;
* `Philox.randint4x`: `randint4x(seed, offset)` for one element; the seed is converted to
  `uint64` and split into the key, the offset into the first two counter words;
* `tl.randint`, `tl.randint4x`: element-wise over an offset tensor;
* `tl.uint_to_uniform_float`, `tl.rand`: `x` bitcast to `int32`, `~x` when negative, times the
  `float32` constant `4.6566127342e-10`. The result is a *product*: the functions are inlined
  (`add_inliner`), so an addition that consumes `tl.rand(…)` may be fused into an `fma` and
  must be modelled `tl.addc`/`tl.subc`, not `tl.add`/`tl.sub`.

The 64-bit variant (`uint64` counters) and `randn` are not covered.
-/
namespace TVA

namespace Philox

/-- The `uint32` constants of `philox_impl`. -/
def keyA : BitVec 32 := 0x9E3779B9
def keyB : BitVec 32 := 0xBB67AE85
def roundA : BitVec 32 := 0xD2511F53
def roundB : BitVec 32 := 0xCD9E8D57

/-- `math.umulhi(a, b)`: the high 32 bits of the 64-bit product. -/
def umulhi (a b : BitVec 32) : BitVec 32 := BitVec.ofNat 32 (a.toNat * b.toNat / 2 ^ 32)

/-- `philox_impl(c0, c1, c2, c3, k0, k1, n)` on `uint32` state. -/
def impl : Nat → BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32 →
    BitVec 32 × BitVec 32 × BitVec 32 × BitVec 32
  | 0, c0, c1, c2, c3, _, _ => (c0, c1, c2, c3)
  | n + 1, c0, c1, c2, c3, k0, k1 =>
    impl n (umulhi roundB c2 ^^^ c1 ^^^ k0) (roundB * c2) (umulhi roundA c0 ^^^ c3 ^^^ k1)
      (roundA * c0) (k0 + keyA) (k1 + keyB)

/-- `randint4x(seed, offset, n_rounds)` for one offset. `seed.to(uint64)` gives the key
`(seed & 0xffffffff, seed >> 32)`; the counter is `(offset.to(uint32), hi, 0, 0)`, where `hi` is
`(offset >> 32).to(uint32)` for offsets wider than 32 bits and `0` otherwise. -/
def randint4x {sg : Bool} {w : Nat} {sg' : Bool} {w' : Nat} (seed : IntV sg w)
    (offset : IntV sg' w') (nRounds : Nat := 10) : U32 × U32 × U32 × U32 :=
  let s : U64 := seed.cast
  let k0 : BitVec 32 := s.bits.setWidth 32
  let k1 : BitVec 32 := (s.bits >>> 32).setWidth 32
  let lo : BitVec 32 := (offset.cast : U32).bits
  let hi : BitVec 32 :=
    if 32 < w' then (if sg' then offset.bits.sshiftRight 32 else offset.bits >>> 32).setWidth 32 else 0
  let r := impl nRounds lo hi 0 0 k0 k1
  (⟨r.1⟩, ⟨r.2.1⟩, ⟨r.2.2.1⟩, ⟨r.2.2.2⟩)

/-- The `float32` constant `4.6566127342e-10` of `uint_to_uniform_float`: the Python float (a
double) converted to `float32`. -/
def scale32 : F32 := FloatV.cvt (FloatV.ofRat (46566127342 / 10 ^ 20 : ℚ) : F64)

/-- The integer `uint_to_uniform_float` scales: the bits as `int32`, complemented when
negative (so in `[0, 2^31)`). -/
def toNonneg (a : U32) : I32 := let b : I32 := ⟨a.bits⟩; if b.lt 0 then b.not else b

end Philox

namespace tl

variable {s : Shape}

/-- `tl.randint(seed, offset, n_rounds)`: element-wise, the first word of `randint4x`. -/
def randint {sg : Bool} {w : Nat} {sg' : Bool} {w' : Nat} (seed : IntV sg w)
    (offset : Tensor s (IntV sg' w')) (nRounds : Nat := 10) : Tensor s U32 :=
  offset.map fun o => (Philox.randint4x seed o nRounds).1

/-- `tl.randint4x(seed, offset, n_rounds)`: the four words, element-wise. -/
def randint4x {sg : Bool} {w : Nat} {sg' : Bool} {w' : Nat} (seed : IntV sg w)
    (offset : Tensor s (IntV sg' w')) (nRounds : Nat := 10) :
    Tensor s U32 × Tensor s U32 × Tensor s U32 × Tensor s U32 :=
  (offset.map fun o => (Philox.randint4x seed o nRounds).1,
   offset.map fun o => (Philox.randint4x seed o nRounds).2.1,
   offset.map fun o => (Philox.randint4x seed o nRounds).2.2.1,
   offset.map fun o => (Philox.randint4x seed o nRounds).2.2.2)

/-- `uint_to_uniform_float(x)` for `uint32`: `Philox.toNonneg`, converted to `float32`
(`tl.itof`), times `Philox.scale32` (IEEE). -/
def uint_to_uniform_float (x : Tensor s U32) : Prog (Tensor s F32) := do
  let xf ← tl.itof (x.map Philox.toNonneg)
  tl.mul xf (Tensor.full Philox.scale32)

/-- `tl.rand(seed, offset, n_rounds)`: `uint_to_uniform_float(randint(seed, offset))`. -/
def rand {sg : Bool} {w : Nat} {sg' : Bool} {w' : Nat} (seed : IntV sg w)
    (offset : Tensor s (IntV sg' w')) (nRounds : Nat := 10) : Prog (Tensor s F32) :=
  uint_to_uniform_float (randint seed offset nRounds)

end tl

end TVA
