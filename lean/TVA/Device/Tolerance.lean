import TVA.Core

/-!
# Tolerance formulas that specifications use

Some claims bound a computed value's error by a formula that follows how the kernel accumulates:
a loop of `acc += tl.dot(a, b)` steps, or a `tl.dot` that accumulates fp8 products in chunks. A
claim's `Post` then names the formula. These are only arithmetic on rationals: they say how large
an error is allowed, not that any computation stays within it (that is what a proof shows, with
`TVA.Proof`). They live here, next to the device facts' constants, because a reader of a claim
reads them as part of its specification.

* `DotLoop.bound ρ α t B`: the error allowed after `t` steps of an accumulation, each with relative
  error `ρ` and absolute error `α`, over steps whose magnitudes sum to `B`:
  `((1 + ρ)^t − 1)·B + t·(1 + ρ)^t·α`.
* `DotLoop.stepRel rel K`, `DotLoop.stepAbs abs K`: the per-step errors of one
  `acc += tl.dot(a, b)` with a `K`-long dot whose own errors are `rel K` and `abs K`, the
  addition into the `float32` accumulator included.
* `Dot8Chunk.psum`, `Dot8Chunk.errB`, `Dot8Chunk.errM`: the error after `j` chunks of an fp8
  `tl.dot` that rounds its running sum once per chunk (`TVA.Device.Dot8Chunk` states the per-chunk
  facts).
-/
namespace TVA

namespace DotLoop

/-- The error allowed after `t` accumulation steps with per-step relative error `ρ` and absolute
error `α`, over steps whose magnitudes sum to `B`: `((1 + ρ)^t − 1)·B + t·(1 + ρ)^t·α`. -/
def bound (ρ α : ℚ) (t : ℕ) (B : ℚ) : ℚ := ((1 + ρ) ^ t - 1) * B + t * (1 + ρ) ^ t * α

/-- The relative error of one `acc += tl.dot(a, b)` step: the dot's own `rel K`, plus one
`float32` addition into the accumulator (`3u` covers the rounding and its interaction). -/
def stepRel (rel : ℕ → ℚ) (K : ℕ) : ℚ := rel K + 3 * Format.binary32.u * (1 + rel K)

/-- The absolute error of one `acc += tl.dot(a, b)` step: the dot's own `abs K`, grown by the
addition, plus the addition's absolute error below `float32`'s normal range. -/
def stepAbs (abs : ℕ → ℚ) (K : ℕ) : ℚ := abs K * (1 + 3 * Format.binary32.u) + 4 * Format.binary32.eta

end DotLoop

namespace Dot8Chunk

/-- The exact partial sums of an accumulation: the start `v` plus the first `j` chunk sums. -/
def psum (v : ℚ) (c : ℕ → ℚ) (j : ℕ) : ℚ := v + ∑ i ∈ Finset.range j, c i

/-- The error allowed after `j` chunks when each chunk's rounding has relative error `ρ` on the
running value, `σ` on the new partial sum and absolute error `α`: `errB 0 = 0` and
`errB (j+1) = (1+ρ+σ)·errB j + ρ·max |psum v c j| (m j) + σ·|psum v c (j+1)| + α`
(`m j` bounds the magnitude the chunk's terms reach). -/
def errB (ρ σ α v : ℚ) (c m : ℕ → ℚ) : ℕ → ℚ
  | 0 => 0
  | j + 1 => (1 + ρ + σ) * errB ρ σ α v c m j + ρ * max |psum v c j| (m j) +
      σ * |psum v c (j + 1)| + α

/-- A sharper chunk bound that takes, at each chunk, the smaller of the relative charge and a
charge `κ·a j + ρ'·(…)` proportional to the chunk's own size `a j`: with `X j = max |psum v c j| (m j)`,
`errM 0 = 0` and
`errM (j+1) = (1+σ)·errM j + min (ρ·(X j + errM j)) (κ·a j + ρ'·(X j + errM j)) + σ·|psum v c (j+1)| + α`.
It is at most `errB`. -/
def errM (ρ σ α κ ρ' v : ℚ) (c a m : ℕ → ℚ) : ℕ → ℚ
  | 0 => 0
  | j + 1 => (1 + σ) * errM ρ σ α κ ρ' v c a m j +
      min (ρ * (max |psum v c j| (m j) + errM ρ σ α κ ρ' v c a m j))
        (κ * a j + ρ' * (max |psum v c j| (m j) + errM ρ σ α κ ρ' v c a m j)) +
      σ * |psum v c (j + 1)| + α

end Dot8Chunk

end TVA
