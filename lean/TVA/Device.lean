import TVA.Device.H200
import TVA.Device.L40

/-!
# The device facts: `Device P`

Some Triton operations have no exact meaning: the compiler lowers them to instructions whose
results the hardware documents only up to an accuracy, or not at all (`exp`, the default `/`,
`tl.dot`, conversions into fp8, float `atomic_add`, `tl.dot_scaled`, the tree of a float
reduction). TVA leaves them to a platform `P` (`TVA.Platform`), a relation per operation, and
every claim quantifies over every `P` that satisfies `Device P`: the conjunction of all the
facts TVA states about the target device running code compiled by Triton 3.8.0.

**The target is a module that can be swapped.** Each supported device has its own module of
facts, `TVA.Device.<target>` with a structure `Facts P` (one field per fact) and a proof
`ref_ok` that the reference platform satisfies them all:

| target | GPU | module | evidence |
|---|---|---|---|
| `H200` | NVIDIA H200 NVL, sm_90 | `TVA.Device.H200` | `evidence/<fact>/` |
| `L40` | NVIDIA L40, sm_89 | `TVA.Device.L40` | `evidence/l40/<fact>/` |

`Device` below names the target's `Facts`: it is the one line that chooses the device every
claim is about (and `ref_ok` the line after it). Both modules are always built and checked; the
fields have the same names on every device, so a proof written against one target is checked
against another by changing that line and building again. A field whose statement differs on
the other device (another constant, another shape) makes the proofs that use it fail there,
which is the point: a claim proved for one GPU is not thereby proved for another.

**What a fact says.** Each fact is one shape of `TVA.Device.Shapes` (accuracy on a domain, an
exact value, totality) at one operation, format and configuration, with its constants written in
(`TVA.Device.Sm90`, `TVA.Device.Dot8Chunk`, `TVA.Device.Dot16Chunk`; a device whose
measurements need other constants defines its own in its module). Its docstring names the
instructions Triton emits and the bound.

**Why they are trusted.** A fact is not proved: it is a statement about hardware, which Lean
cannot see. It is part of what a reader of a claim must trust (`TVA.Claim`), with Lean, Mathlib,
the claim's transcription of the kernel and the compiler. Each fact is kept as narrow as the
claims need, and each is stated with slack over what was observed (typically a factor of 2 on
each constant), so that it is not an exact model of one instruction sequence but a bound that
the instructions Triton may choose all meet.

**How they were measured.** Each fact cites a folder of `evidence/` (`evidence/num-math-f32`,
`evidence/num-dot`, …). It holds the script that ran the operation through Triton 3.8.0 on an
NVIDIA H200 NVL, its output, and a README saying what it shows. Where the input space is small
the measurement is exhaustive (every `float32` input of `exp`, every `float16` input of a
conversion); for `tl.dot` it combines a bit-level model of the tensor-core instructions, checked
against the device on every element tested, with adversarial and random inputs at every size
that compiles; and the bound is derived from the model, then checked against the device.

**Why `Device.ref_ok` matters.** Every claim assumes `Device P`. If the facts contradicted each
other, no platform would satisfy them, and every claim would hold vacuously, whatever its kernel
does. `Device.ref_ok` proves that one platform, the target's `Device.refPlatform`, satisfies
every fact at once: `Platform.ref` (`TVA.Device.Ref.Platform`: IEEE arithmetic rounded once, as
accurate as a float result can be) on the H200; on the L40 the same platform except for its
conversion from `float32` into fp8, which rounds twice as the device does. So the facts are
consistent, and a claim's proof cannot draw anything from them that a real platform as accurate
as the reference would not give.

A fact implied by others is a theorem rather than a field: the round-to-nearest conversions in range
from `float16` and `bfloat16` (`Device.cvtF16E4m3`, …, from the saturating ones, on every device),
from `float32` on the H200 only (`H200.Facts.cvtF32E4m3`, `cvtF32E5m2`; a proof writes
`h.cvtF32E4m3`), and the first arm of the fp8 chunk facts (`Device.dotE4m3Chunk`, …). Weaker forms
of any fact follow by `TVA.Device.Weaken`. -/ namespace TVA

open Device

/-- **The device facts** every claim assumes (`TVA.Claim`): those of the target device. **To
change the target, change `H200` here and in `Device.refPlatform`, `Device.ref_ok` below** (to
`L40`, …). -/
abbrev Device (P : Platform) : Prop := Device.H200.Facts P

namespace Device

variable {P : Platform}

/-- The target's reference platform: one that satisfies every fact of the target. -/
noncomputable abbrev refPlatform : Platform := H200.refPlatform

/-- **The target's facts are jointly satisfiable**: the target's reference platform satisfies every
one, so no claim holds vacuously through its assumption `Device P`. -/
theorem ref_ok : Device refPlatform := H200.ref_ok

/-! ## Facts implied by others -/

/-- `float16 → float8e4nv` rounds to nearest even in range (`CvtF16E4m3`). -/
theorem cvtF16E4m3 (h : Device P) : CvtF16E4m3 P := h.cvtSatF16E4m3.cvtRn Format.e4m3_regular

/-- `float16 → float8e5` rounds to nearest even in range (`CvtF16E5m2`). -/
theorem cvtF16E5m2 (h : Device P) : CvtF16E5m2 P := h.cvtSatF16E5m2.cvtRn Format.e5m2_regular

/-- `bfloat16 → float8e4nv` rounds to nearest even in range (`CvtBF16E4m3`). -/
theorem cvtBF16E4m3 (h : Device P) : CvtBF16E4m3 P :=
  h.cvtSatBF16E4m3.cvtRn Format.e4m3_regular

/-- `bfloat16 → float8e5` rounds to nearest even in range (`CvtBF16E5m2`). -/
theorem cvtBF16E5m2 (h : Device P) : CvtBF16E5m2 P :=
  h.cvtSatBF16E5m2.cvtRn Format.e5m2_regular

/-- `tl.dot` of `float8e4nv`, chunk by chunk, first arm (`DotE4m3Chunk`). -/
theorem dotE4m3Chunk (h : Device P) : DotE4m3Chunk P := DotF8ChunkLoss.chunk h.dotE4m3ChunkLoss

/-- `tl.dot` of `float8e5`, chunk by chunk, first arm (`DotE5m2Chunk`). -/
theorem dotE5m2Chunk (h : Device P) : DotE5m2Chunk P := DotF8ChunkLoss.chunk h.dotE5m2ChunkLoss

/-- `tl.dot` of `float8e4nv` by `float8e5`, chunk by chunk, first arm (`DotE4m3E5m2Chunk`). -/
theorem dotE4m3E5m2Chunk (h : Device P) : DotE4m3E5m2Chunk P :=
  DotF8ChunkLoss.chunk h.dotE4m3E5m2ChunkLoss

/-- `tl.dot` of `float8e5` by `float8e4nv`, chunk by chunk, first arm (`DotE5m2E4m3Chunk`). -/
theorem dotE5m2E4m3Chunk (h : Device P) : DotE5m2E4m3Chunk P :=
  DotF8ChunkLoss.chunk h.dotE5m2E4m3ChunkLoss

end Device

end TVA
