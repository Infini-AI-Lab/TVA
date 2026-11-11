import TVA.Device.Shapes

/-!
# The device facts (H200, sm_90, Triton 3.8.0)

Each definition here is one fact about the platform `P`: one shape of `TVA.Device.Shapes` at one
operation, format and configuration, with its constants written in. Its docstring says what the
device does (the instructions Triton 3.8.0 lowers the operation to), the bound, and the folder of
`evidence/` where TVA measured it on an NVIDIA H200 NVL. The fp8 and 16-bit `tl.dot` facts stated
chunk by chunk are in `TVA.Device.Dot8Chunk` and `TVA.Device.Dot16Chunk`.

The constants a claim's tolerance may use are here under their own names: `relDot16`,
`absDot16` (a 16-bit `tl.dot`), `scaledDomBF16`, `e8m0Dom` (the domain of `tl.dot_scaled`).

`TVA.Device` gathers these facts into `Device P`; `TVA.Device.Ref.Sm90` proves each of them of
the reference platform.
-/
namespace TVA.Device

variable (P : Platform)

/-! ## Approximate functions on `float32` -/

/-- **`tl.exp` on `float32`.** Triton computes `x · log2 e` with `mul.f32` (the constant rounded
to `float32`), then `ex2.approx.f32` without `.ftz`, so subnormal results are kept
(`ElementwiseOpToLLVM.cpp:634-658`). Rounding the product moves the exponent by up to about
`2^-24·|x|`, so the relative error grows with `|x|`: for every `X ≥ 0` and finite `x` in
`[-X, X]` (with room in `float32`), the result is finite and within
`(2^-22 + 2^-23·X)·e^x + 2^-140` of `e^x`. Evidence: `evidence/num-math-f32` (every `float32`
input). -/
def ExpF32 : Prop :=
  ∀ X : ℝ, 0 ≤ X →
    MathAcc P .exp .binary32 (Set.Icc (-X) X) (2 ^ (-22 : ℤ) + 2 ^ (-23 : ℤ) * X) (2 ^ (-140 : ℤ))

/-- **`tl.math.exp2` on `float32`.** Triton calls libdevice's `__nv_exp2f`, which with Triton's
default flush-to-zero setting is `ex2.approx.ftz.f32`: results below `2^-126` are flushed to `0`.
Every finite input (with room in `float32`) gives a finite result within `2^-22·2^x + 2^-126` of
`2^x`. Evidence: `evidence/num-math-f32` (every `float32` input). -/
def Exp2F32 : Prop := MathAcc P .exp2 .binary32 Set.univ (2 ^ (-22 : ℤ)) (2 ^ (-126 : ℤ))

/-- **`tl.log` on `float32`** (libdevice's `__nv_logf`): for finite `x > 0`, within
`2^-22·|ln x| + 2^-149` of `ln x`. Evidence: `evidence/num-math-f32` (every `float32` input). -/
def LogF32 : Prop := MathAcc P .log .binary32 Set.univ (2 ^ (-22 : ℤ)) (2 ^ (-149 : ℤ))

/-- **`tl.math.log2` on `float32`** (libdevice's `__nv_log2f`, a polynomial with `fma.rn.ftz`
that scales subnormal inputs first): for finite `x > 0`, within `2^-22·|log₂ x| + 2^-149` of
`log₂ x`. Evidence: `evidence/num-math-f32` (every `float32` input). -/
def Log2F32 : Prop := MathAcc P .log2 .binary32 Set.univ (2 ^ (-22 : ℤ)) (2 ^ (-149 : ℤ))

/-- The inputs on which `tl.sqrt` is accurate: `0` and the normal positive floats. -/
def sqrtDom : Set ℝ := {x | x = 0 ∨ 2 ^ (-126 : ℤ) ≤ x}

/-- **`tl.sqrt` on `float32`** (`sqrt.approx.ftz.f32`): for `x = 0` or a normal `x > 0`, within
`2^-22·√x + 2^-149` of `√x`. A subnormal input is flushed to `0` and gives `0`, so subnormal
inputs are outside the domain. Evidence: `evidence/num-math-f32` (every `float32` input). -/
def SqrtF32 : Prop := MathAcc P .sqrt .binary32 sqrtDom (2 ^ (-22 : ℤ)) (2 ^ (-149 : ℤ))

/-- **`tl.math.rsqrt` on `float32`** (`rsqrt.approx.ftz.f32`): for a normal `x > 0`, within
`2^-22/√x + 2^-149` of `1/√x`. A subnormal input is flushed and gives `+∞`. Evidence:
`evidence/num-math-f32` (every `float32` input). -/
def RsqrtF32 : Prop :=
  MathAcc P .rsqrt .binary32 (Set.Ici (2 ^ (-126 : ℤ))) (2 ^ (-22 : ℤ)) (2 ^ (-149 : ℤ))

/-- **`tl.exp(-∞) = +0`** on `float32` (`ex2.approx.f32` of `-∞`). Softmax over masked lanes
loaded as `-inf` needs it. Evidence: `evidence/num-math-f32` (special values). -/
def ExpF32NegInf : Prop := MathAt P .exp .binary32 FloatV.negInf (FloatV.ofPat 0)

/-- **`tl.math.exp2(-∞) = +0`** on `float32` (`ex2.approx.ftz.f32` of `-∞`). An online softmax
needs it when the running maximum starts at `-inf`. Evidence: `evidence/num-math-f32`. -/
def Exp2F32NegInf : Prop := MathAt P .exp2 .binary32 FloatV.negInf (FloatV.ofPat 0)

/-- **`tl.exp` overflows to `+∞` on `float32`**: every input from `88.72283935546875` (pattern
`0x42b17218`, `1453635/16384`) up, and `+∞`, gives `+∞`. Triton computes `x · log2 e` with
`mul.f32`, then `ex2.approx.f32` (`ElementwiseOpToLLVM.cpp:634-658`), which is `+∞` once the
product reaches `128`; the largest input with a finite result is `0x42b17217`
(`88.72283172607422`, result `0x7f7fff4f`). `ExpF32` says nothing there (its room fails).
Evidence: `evidence/num-special-f32`, part E1 (every `float32` pattern from `+64` to `+∞`). -/
def ExpF32Overflow : Prop :=
  ∀ x : F32, ((x.isFinite = true ∧ (1453635 / 16384 : ℚ) ≤ x.toRat) ∨ x = FloatV.posInf) →
    ∀ r, P.math .exp .binary32 x r → r = FloatV.posInf

/-! ## Division -/

/-- **`/` on `float32` is `div.full.f32`**, within `2^-22·|x/y| + 2^-148` of `x/y` for
`|x/y| ≤ 2^126`. Triton 3.8.0 lowers `arith.divf` on `f32` to `llvm.nvvm.div.full`
(`ElementwiseOpToLLVM.cpp:553-580`), which LLVM selects as `div.full.f32`
(`NVPTXIntrinsics.td:1536-1537`). The PTX ISA documents it as a full-range approximation with a
maximum error of 2 ulp. The fact is the relative bound, not "2 ulp": near the top of a binade,
where the bound is almost 4 ulp of the quotient's binade, errors reach 2.11 ulp. Evidence:
`evidence/num-div-f32` (about `3.1·10^10` pairs: every divisor mantissa at many exponents,
subnormal operands, random pairs; all within the bound, at most `0.651` of it). -/
def DivFull : Prop := DivAcc P .binary32 (2 ^ 126) (2 ^ (-22 : ℤ)) (2 ^ (-148 : ℤ))

/-- **`x / ±∞ = ±0`** on `float32` (`div.full.f32`): for every finite `x` and infinite `y`, the
result is the zero whose sign is `sign x xor sign y` (`1 / ∞ = +0`), as IEEE division gives it.
`DivFull` says nothing at an infinite divisor. Evidence: `evidence/num-special-f32`, part D1
(every finite `float32` `x`, by `+∞` and by `-∞`). -/
def DivF32InfDivisor : Prop :=
  ∀ x y : F32, x.isFinite = true → y.isInf = true → ∀ r, P.div .binary32 x y r →
    r = FloatV.ofPat (Format.binary32.zeroPat (x.sign != y.sign))

/-- **`/` on `float64` is IEEE division**, up to NaN payload. Triton 3.8.0 lowers `arith.divf` on
`f64` to `llvm.nvvm.div.rn.d` (`ElementwiseOpToLLVM.cpp:569-571`), which is `div.rn.f64` or,
after LLVM's `instcombine`, an IEEE `fdiv`. Evidence: `evidence/num-div-f64` (`x / y` and
`tl.fdiv(x, y)` on `float64` against `FloatV.divRn` on about `4.5·10^8` pairs: random patterns,
subnormal dividends, quotients and divisors, divisors near `1`, quotients next to midpoints,
constant divisors; all equal up to NaN payload). -/
def DivF64 : Prop := DivRn P .binary64

/-! ## Conversions into fp8

Triton converts into fp8 with `cvt.rn.satfinite.e4m3x2` / `e5m2x2` (from `float32` or `float16`;
a `bfloat16` source is first widened exactly to `float32`). `.satfinite` sends a finite value
beyond `maxVal` (`448`, `57344`) and `±∞` to `±maxVal`, and a NaN to the NaN `0x7f`. Each fact is
`CvtSat`: the result is `FloatV.cvtSat .rn x`, up to NaN payload. The round-to-nearest facts in
range (`CvtF32E4m3`, …) follow (`Device.cvtF32E4m3`, …). Evidence: `evidence/num-cvt8` (every
input of the three source formats, bit for bit: `run.py` in range and above `maxVal`,
`specials.py` every `±∞` and NaN pattern) and `evidence/float-layer`, part A (`FloatV.cvtSat`
against the device on every `float16`, `bfloat16` and `float32` input). -/

/-- **`float32 → float8e4nv`**, every input (`cvt.rn.satfinite.e4m3x2.f32`): `CvtSat`. -/
def CvtSatF32E4m3 : Prop := CvtSat P .binary32 .e4m3

/-- **`float32 → float8e5`**, every input (`cvt.rn.satfinite.e5m2x2.f32`): `CvtSat`. -/
def CvtSatF32E5m2 : Prop := CvtSat P .binary32 .e5m2

/-- **`float16 → float8e4nv`**, every input (`cvt.rn.satfinite.e4m3x2.f16x2`): `CvtSat`. -/
def CvtSatF16E4m3 : Prop := CvtSat P .binary16 .e4m3

/-- **`float16 → float8e5`**, every input (`cvt.rn.satfinite.e5m2x2.f16x2`): `CvtSat`. -/
def CvtSatF16E5m2 : Prop := CvtSat P .binary16 .e5m2

/-- **`bfloat16 → float8e4nv`**, every input (exact widening to `float32`, then
`cvt.rn.satfinite.e4m3x2.f32`): `CvtSat`. -/
def CvtSatBF16E4m3 : Prop := CvtSat P .bfloat16 .e4m3

/-- **`bfloat16 → float8e5`**, every input (exact widening, then `cvt.rn.satfinite.e5m2x2.f32`):
`CvtSat`. -/
def CvtSatBF16E5m2 : Prop := CvtSat P .bfloat16 .e5m2

/-- **`float32 → float8e4nv` in range**: a finite value with `|x| ≤ 448` rounds to nearest even
(`CvtRn`). It follows from `CvtSatF32E4m3`. -/
def CvtF32E4m3 : Prop := CvtRn P .binary32 .e4m3

/-- **`float32 → float8e5` in range** (`|x| ≤ 57344`): `CvtRn`. It follows from
`CvtSatF32E5m2`. -/
def CvtF32E5m2 : Prop := CvtRn P .binary32 .e5m2

/-- **`float16 → float8e4nv` in range**: `CvtRn`. It follows from `CvtSatF16E4m3`. -/
def CvtF16E4m3 : Prop := CvtRn P .binary16 .e4m3

/-- **`float16 → float8e5` in range**: `CvtRn`. It follows from `CvtSatF16E5m2`. -/
def CvtF16E5m2 : Prop := CvtRn P .binary16 .e5m2

/-- **`bfloat16 → float8e4nv` in range**: `CvtRn`. It follows from `CvtSatBF16E4m3`. -/
def CvtBF16E4m3 : Prop := CvtRn P .bfloat16 .e4m3

/-- **`bfloat16 → float8e5` in range**: `CvtRn`. It follows from `CvtSatBF16E5m2`. -/
def CvtBF16E5m2 : Prop := CvtRn P .bfloat16 .e5m2

/-! ## `tl.dot` -/

/-- The relative constant of a 16-bit `tl.dot` with `K` products: `(K + 2)·2^-22`. -/
def relDot16 (K : ℕ) : ℚ := (K + 2) * 2 ^ (-22 : ℤ)

/-- The absolute constant of a 16-bit `tl.dot` with `K` products: `(K + 2)·2^-126`. -/
def absDot16 (K : ℕ) : ℚ := (K + 2) * 2 ^ (-126 : ℤ)

/-- **`tl.dot` of `float16` operands into `float32`** (`max_num_imprecise_acc = 0`, any
`input_precision`, which Triton ignores for these operands), relative to lifted magnitudes
(`DotLiftAcc`). Triton chains `mma.sync` or `wgmma` instructions of `16` products each along `K`.
Each instruction aligns the accumulator and its products to the largest exponent field `E` (a
product's field is its operands' fields added, a subnormal operand's field read as the smallest
normal's, `2^-14`), truncates every term toward zero to a multiple of `2^(E-25)`, sums them
exactly and truncates the sum toward zero to `float32`. So an instruction's error is below
`5.25·2^-23·(|s| + A) + 5.25·2^-149`, with `s` the accumulator before it (never larger than
`|acc| + Σ|a·b|`) and `A` its sum of `liftMag a · liftMag b`; over the `⌈K/16⌉` instructions this
stays below the constants. Per output element, with `L = |acc| + Σ liftMag a · liftMag b` and room
in `float32`: within `(K+2)·2^-22·L + (K+2)·2^-126` of `acc + Σ a·b`. Evidence:
`evidence/num-dot16-model` (a bit-exact model of both instruction paths, and adversarial inputs
at every `K` that compiles), `evidence/num-dot`. -/
def DotF16 : Prop :=
  ∀ p : InputPrecision, DotLiftAcc P ⟨.f16, .f16, .f32, p, 0⟩ relDot16 absDot16

/-- **`tl.dot` of `bfloat16` operands into `float32`**: as `DotF16`, with the same constants, a
subnormal operand counted at `2^-126` (the absolute term covers products below `float32`'s normal
range). Evidence: `evidence/num-dot16-model`, `evidence/num-dot`. -/
def DotBF16 : Prop :=
  ∀ p : InputPrecision, DotLiftAcc P ⟨.bf16, .bf16, .f32, p, 0⟩ relDot16 absDot16

/-- **`tl.dot` of `float32` operands with `input_precision="ieee"`** (no tensor cores): each
output element is a sequence of `K` `fma.rn.f32`, `k = 0, …, K-1`, from the accumulator. Per
element: within `(K+1)·2^-23·(|acc| + Σ|a·b|) + (K+1)·2^-149` of `acc + Σ a·b`, with room.
Evidence: `evidence/num-dot` (bit for bit against the sequential `fma`, and adversarial
inputs). -/
def DotF32Ieee : Prop :=
  DotAcc P ⟨.f32, .f32, .f32, .ieee, 0⟩ (fun _ => True)
    (fun K => (K + 1) * 2 ^ (-23 : ℤ)) (fun K => (K + 1) * 2 ^ (-149 : ℤ))

/-- **`tl.dot` of `int8` operands into `int32`** (with or without an accumulator, any
`input_precision`, `max_num_imprecise_acc = 0`, the default for operands that are not fp8,
`semantic.py:1516-1520`): exact. Triton 3.8.0's only integer `tl.dot` is this one
(`semantic.py:1483-1486`: both operands `int8`, an `int32` result whatever `out_dtype` says). It
chains `mma.sync …satfinite.s32.s8.s8.s32` (`m16n8k32`) or `wgmma …s32.s8.s8` (`k32`)
instructions along `K`, which sum in integers: the first saturates an `int32` partial sum, the
second wraps it. When `|acc| + Σ|a·b|` is at most `2^31 - 1`, no partial sum in any order leaves
the `int32` range, so the result is `acc + Σ a·b` exactly. Beyond that the two paths differ, and
the fact says nothing. Evidence: `evidence/num-dot-int8` (random and extreme operands on both
paths and every precision, the room at its edge, and the paths' behaviour beyond it). -/
def DotI8 : Prop :=
  ∀ p : InputPrecision, DotAcc P ⟨.i8, .i8, .i32, p, 0⟩ (fun _ => True) (fun _ => 0) (fun _ => 0)

/-! ## `tl.dot_scaled` -/

/-- The scaled values the sm_90 emulation multiplies exactly in `bfloat16`: `0`, or a magnitude
in `bfloat16`'s normal range. -/
def scaledDomBF16 (q : ℚ) : Prop :=
  q = 0 ∨ ((2 : ℚ) ^ (-126 : ℤ) ≤ |q| ∧ |q| ≤ Format.bfloat16.maxVal)

/-- The `e8m0` scale bytes the emulation decodes as their OCP value: `1 … 254` (`0x00` is read
as `0`, `0xFF` is NaN). -/
def e8m0Dom (b : ℕ) : Prop := 1 ≤ b ∧ b ≤ 254

/-- `tl.dot_scaled` with `e8m0` scales on both sides, groups of `32`, `K`-packed operands and
either `fast_math`, as sm_90 runs it: each element and its scale are converted to `bfloat16` and
multiplied there (exact on `scaledDomBF16`), then a `bfloat16` `tl.dot` into `float32`
(`DotBF16`'s constants). On `scaledDomBF16` no operand of that dot is subnormal, so its lifted
magnitudes are `|a·b|` and the bound is relative to `|acc| + Σ|a·b|`. -/
def DotScaledMx (lhs rhs : MXFmt) : Prop :=
  ∀ fm : Bool, DotScaledAcc P ⟨lhs, rhs, .e8m0, .e8m0, 32, true, true, fm⟩ scaledDomBF16 e8m0Dom
    relDot16 absDot16

/-- **`tl.dot_scaled(…, "e4m3", …, "e4m3", …)`** (MXFP8) on sm_90: Triton's emulation
(`DecomposeScaledBlocked.cpp:191-228`), as `DotScaledMx`. Per output element, for scaled values
in `scaledDomBF16` and scale bytes in `e8m0Dom`: within `(K+2)·2^-22·mag + (K+2)·2^-126`.
Evidence: `evidence/num-dot-scaled`. -/
def DotScaledE4m3 : Prop := DotScaledMx P .e4m3 .e4m3

/-- **`tl.dot_scaled(…, "e4m3", …, "e2m1", …)`** (MXFP8 by MXFP4) on sm_90, as `DotScaledMx`.
Evidence: `evidence/num-dot-scaled`. -/
def DotScaledE4m3E2m1 : Prop := DotScaledMx P .e4m3 .e2m1

/-! ## Float `atomic_add` -/

/-- **`tl.atomic_add` on `float32`**: `AtomicAddAcc` with `rel = 2^-23`, `abs = 2^-124`. It
covers the pointer atomic (`atom.global.<scope>.<sem>.add.f32` and its `.v4` form, every `sem`
and `scope`, all bit-identical) and a descriptor's `atomic_add` (the TMA reduction
`cp.reduce.async.bulk.tensor ….add`). `atom.add.f32` rounds to nearest and flushes subnormal
inputs and results to zero: two operands just below the smallest normal sum to almost `2^-125`
and give `0`. That stays within `2^-24·|o + v| + 2^-125`; the fact keeps a factor `2` on both.
Evidence: `evidence/num-atomic-add` (`run.py`; `sem_scope.py` for the 12 `sem`/`scope`
forms). -/
def AtomicAddF32 : Prop := AtomicAddAcc P .binary32 (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ))

/-! ## The depth of a float reduction's tree -/

/-- **The trees of a float reduction on sm_90 are balanced**: `P.sumDepth n ≤ ⌈log₂ n⌉`, so a
float reduction (`tl.sum` and the other float reductions, on any axis) over the `n` positions of
its axis combines them along a tree of depth at most `⌈log₂ n⌉` (`log₂ n` for the power of two
`n` of a Triton axis). Triton 3.8.0 lowers a reduction (`ReduceOpToLLVM.cpp`) as a balanced
binary tree over the elements one thread holds along the axis (`treeReduce`), then a butterfly of
`shfl.sync.bfly` over the lanes along the axis, then, through shared memory, the same butterfly
over the warps along the axis: each stage halves the positions left, whatever the layout. It is
not stated for a reduction the compiler rewrites across loop iterations
(`OptimizeThreadLocality.cpp`). Evidence: `evidence/reduce-tree`. -/
def SumDepth : Prop := ∀ n, P.sumDepth n ≤ Nat.clog 2 n

end TVA.Device
