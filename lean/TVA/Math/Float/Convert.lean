import TVA.Math.Float.IEEE

/-!
# Conversions between binary formats on bit patterns

`bf16 → f32` and `f16 → f32` are exact on numbers; `f32 → bf16` and `f32 → f16` round to
nearest even. NaN inputs: `f32 → bf16` and `f32 → f16` give the canonical NaN `0x7fff`, and
`f16 → f32` gives the canonical NaN `0x7fffffff`, whatever the sign and payload of the input;
`bf16 → f32` is a 16-bit shift and keeps both.

Evidence: the conformance records `docs/evidence/conformance/cvt.md` (`cvt.bf16.f32`,
`cvt.f16.f32`, `cvt.f32.bf16`, `cvt.f32.f16`; every input, H200 NVL, nvcc 13.2, 2026-10-25). The
f16 NaN rules were corrected after those records (findings F1, F2: the earlier model kept the
sign and payload of a NaN).
-/
namespace TVA.Math.Float

/-- Round-to-nearest-even conversion `f32 → bf16` on bits (canonical NaN `0x7fff`). -/
def f32ToBf16RNE (bits : Nat) : Nat :=
  let bits := bits % 2 ^ 32
  if f32IsNaN bits then 0x7fff
  else
    let lsb := (bits >>> 16) % 2
    let rounded := (bits + 0x7FFF + lsb) >>> 16
    rounded % 2 ^ 16

/-- Round-to-nearest-even `f32 → f16` on bits, including subnormals and overflow to inf; every
NaN gives the canonical NaN `0x7fff` (`cvt.rn.f16.f32`, `__float2half_rn`). -/
def f32ToF16RNE (bits : Nat) : Nat :=
  let f := f32Fields bits
  let sign := if f.sign then 0x8000 else 0
  if f.isNaN 8 then 0x7fff
  else if f.isInf 8 then sign ||| 0x7C00
  else
    let e : Int := (f.exp : Int) - 127
    if e > 15 then sign ||| 0x7C00
    else if e ≥ -14 then
      let m := f.mant
      let shift := 13
      let lsb := (m >>> shift) % 2
      let half := 1 <<< (shift - 1)
      let r := m + half - 1 + lsb
      let m16 := r >>> shift
      let e16 := (e + 15).toNat
      if m16 ≥ 1024 then
        if e16 + 1 ≥ 31 then sign ||| 0x7C00 else sign ||| ((e16 + 1) <<< 10)
      else sign ||| (e16 <<< 10) ||| m16
    else
      let full := (1 <<< 23) ||| f.mant
      let shift := (-14 - e).toNat + 13
      if shift ≥ 64 then sign
      else
        let lsb := (full >>> shift) % 2
        let half := 1 <<< (shift - 1)
        let rem := full % (1 <<< shift)
        let base := full >>> shift
        let m16 := if rem > half || (rem == half && lsb == 1) then base + 1 else base
        sign ||| m16

/-- `f16 → f32` is exact on numbers and infinities; every NaN gives the canonical NaN
`0x7fffffff` (`cvt.f32.f16`, `__half2float`). -/
def f16ToF32Bits (bits : Nat) : Nat :=
  let f := f16Fields bits
  let sign := if f.sign then 1 <<< 31 else 0
  if f.isNaN 5 then 0x7fffffff
  else if f.isInf 5 then sign ||| 0x7F800000
  else if f.exp == 0 then
    if f.mant == 0 then sign
    else
      let rec norm (m e : Nat) (fuel : Nat) : Nat × Nat :=
        match fuel with
        | 0 => (m, e)
        | fuel + 1 => if m ≥ 1024 then (m, e) else norm (m <<< 1) (e - 1) fuel
      let (m, e) := norm f.mant (127 - 15 + 1) 24
      sign ||| (e <<< 23) ||| ((m % 1024) <<< 13)
  else sign ||| ((f.exp + 127 - 15) <<< 23) ||| (f.mant <<< 13)

end TVA.Math.Float
