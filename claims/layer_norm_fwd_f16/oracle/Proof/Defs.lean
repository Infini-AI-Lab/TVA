import Proof.Bridge

/-!
# Footprints of one program

Program `t` (the grid `(M,)`) owns row `t`: it reads the row of `x` and all of `weight` and `bias`,
and writes the row of `y`, `mean[t]` and `rstd[t]`. `MemOk` says a memory still holds the launch's
inputs: no store of the program touches them.
-/

namespace Claims.LayerNormFwdF16

open TVA

/-- The elements of row `t` of an `[M, N]` tensor. -/
def rowEls (d : Inst) (t : Nat) : Set Nat := {k | ∃ j, j < d.N ∧ k = idx d t j}

/-- The bytes program `t` reads. -/
def R (d : Inst) (t : Nat) : Set ℤ :=
  d.x.elems (rowEls d t) ∪ d.w.elems (vecSet d) ∪ d.b.elems (vecSet d)

/-- The bytes program `t` writes. -/
def W (d : Inst) (t : Nat) : Set ℤ :=
  d.y.elems (rowEls d t) ∪ d.mean.elems {t} ∪ d.rstd.elems {t}

/-- A memory that still holds the inputs of the launch: the bytes read are `init`'s, and every byte
present in `init` is present. -/
def MemOk (d : Inst) (init m : Mem) : Prop :=
  (∀ z ∈ inBytes d, m z = init z) ∧ ∀ z, init z ≠ none → m z ≠ none

/-- Element `(t, j)` of a covered launch is an element of the `[M, N]` tensor. -/
theorem idx_lt {d : Inst} {t j : Nat} (ht : t < d.M) (hj : j < d.N) : idx d t j < d.M * d.N := by
  unfold idx
  have : (t + 1) * d.N ≤ d.M * d.N := Nat.mul_le_mul_right _ ht
  nlinarith

end Claims.LayerNormFwdF16
