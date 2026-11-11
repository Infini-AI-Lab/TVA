import TVA.Triton.Core

/-!
# Loops

How a model writes the kernel's loops; a loop's carried variables are its state (a tuple).

* `for i in range(lo, hi, step)` and `for i in tl.range(lo, hi, step, …)` are `tl.forRange`
  (`range(hi)` has `lo = 0`, `range(lo, hi)` has `step = 1`). The induction variable has the
  integer type Triton gives it, the promotion of the three bounds (`code_generator.py:1264-1265`):
  `int32` usually, `int64` when a bound is.
* `tl.static_range` is unrolled by the front end, with a Python integer as the variable
  (`code_generator.py:1208-1216`): the model writes the unrolled body, or `tl.forN` over a Lean
  count.
* `while cond: …` is `tl.whileLoop`, with the source's condition (`Prog.loop`: the trip count is
  whatever the condition decides at run time).

The scheduling keywords of `tl.range` (`num_stages`, `loop_unroll_factor`, `warp_specialize`,
`flatten`, …) change no outcome of a well-behaved model: pipelining can make an iteration miss what
an earlier one stored, but that is a program accessing its own bytes twice in a conflicting way,
which a well-behaved launch never does (`TVA.Program`).
-/
namespace TVA

namespace tl

/-- Run `body t` for `t = 0, …, n-1`, threading the state: a loop with a fixed trip count. -/
def forN {σ : Type} : Nat → (Nat → σ → Prog σ) → σ → Prog σ
  | 0, _, init => .ret init
  | n + 1, body, init => (forN n body init).bind (body n)

/-- The number of iterations of `range(lo, hi, step)` for `step > 0`: `⌈(hi - lo) / step⌉`, or `0`
when `hi ≤ lo`. -/
def tripCount (lo hi step : Int) : Nat := ((hi - lo + step - 1) / step).toNat

/-- The bounds `tl.forRange` covers: a positive step, and `lo`, `step` and `hi + step − 1` within
the *signed* range of the width, so that the last increment does not overflow. `scf.for` compares
signed even for an unsigned induction variable (`code_generator.py:1267-1288` passes no
signedness), so unsigned bounds at or above `2^(w−1)` would run a different number of times. -/
def RangeOk {sg : Bool} {w : Nat} (lo hi step : IntV sg w) : Prop :=
  0 < step.toInt ∧ hi.toInt + step.toInt - 1 ≤ IntV.maxInt true w ∧
    lo.toInt ≤ IntV.maxInt true w ∧ step.toInt ≤ IntV.maxInt true w

instance {sg : Bool} {w : Nat} (lo hi step : IntV sg w) : Decidable (RangeOk lo hi step) := by
  unfold RangeOk; infer_instance

/-- `for i in range(lo, hi, step)` (and `tl.range`) with a loop-carried state: the body runs with
`i = lo, lo + step, …` below `hi`. Bounds outside `RangeOk` are undefined behaviour here. -/
def forRange {σ : Type} {sg : Bool} {w : Nat} (lo hi step : IntV sg w) (init : σ)
    (body : IntV sg w → σ → Prog σ) : Prog σ :=
  if RangeOk lo hi step then
    forN (tripCount lo.toInt hi.toInt step.toInt)
      (fun t => body (IntV.ofInt (lo.toInt + t * step.toInt))) init
  else .ub

/-- `while cond(s): s = body(s)`: evaluate the condition on the state, run the body while it
holds, and end with the state that fails it. -/
def whileLoop {σ : Type} (init : σ) (cond : σ → Prog Bool) (body : σ → Prog σ) : Prog σ :=
  .loop init (fun st => (cond st).bind fun c =>
    if c then (body st).bind (fun st' => .ret (.inl st')) else .ret (.inr st)) .ret

end tl

end TVA
