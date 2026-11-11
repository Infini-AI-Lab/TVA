import Claim
import TVA.Proof

/-!
# `attn_bwd_f16`: the shared definitions of the proof

What a computed `P` or `dS` element is allowed to be (`PGood`, `DSGood`), and the invariant of an
accumulator lane that sums products over a set of processed indices (`Fam`, `AccInv`).
-/

namespace Claims.AttnBwd

open TVA

/-- The computed `float16` `P[n, m]` (line 302) is finite, within `p16Err` of the exact `P[n, m]`,
and its lifted magnitude is at most `p16L`. -/
def PGood (d : Inst) (hv : HeadV) (n m : Nat) (x : F16) : Prop :=
  x.isFinite = true ∧ |x.toReal - pE d hv n m| ≤ p16Err d hv n m ∧ lm x ≤ p16L d hv n m

/-- The computed `float16` `dS[n, m]` (line 309) is finite, within `ds16Err` of the exact
`dS[n, m]`, and its lifted magnitude is at most `ds16L`. -/
def DSGood (d : Inst) (hv : HeadV) (n m : Nat) (x : F16) : Prop :=
  x.isFinite = true ∧ |x.toReal - dsE d hv n m| ≤ ds16Err d hv n m ∧ lm x ≤ ds16L d hv n m

/-- A family of products `coef(r, s) · u(s, j)` summed over `s`: the exact coefficient `wt`, the
error `err` of a computed coefficient, a bound `lb` on its lifted magnitude, and the column values
`u`. The three accumulators of the kernel are `dV` (`P`, `dO`), `dK` (`dS`, `Q`) and `dQ`
(`dS`, `K`). -/
structure Fam where
  /-- The exact coefficient of row `r` and summation index `s`. -/
  wt : Nat → Nat → ℝ
  /-- The error bound of the computed coefficient. -/
  err : Nat → Nat → ℝ
  /-- The bound on the computed coefficient's lifted magnitude. -/
  lb : Nat → Nat → ℝ
  /-- The `float16` value at summation index `s`, column `j`. -/
  u : Nat → Nat → F16

/-- **The invariant of an accumulator** `acc` (rows `rows r`, columns `cols j`) after `t`
accumulation steps over the summation indices `S`: each row has computed coefficients `cv`, finite,
within `err` of `wt` with lifted magnitude within `lb`, and each lane's value is within
`accB t (Σ lb·lm u)` of `Σ cv·u`. -/
def AccInv {M N : Nat} (fam : Fam) (rows : Fin M → Nat) (cols : Fin N → Nat) (S : Finset Nat)
    (t : Nat) (acc : Tensor [M, N] F32) : Prop :=
  ∀ r : Fin M, ∃ cv : Nat → F16,
    (∀ s ∈ S, (cv s).isFinite = true ∧ |(cv s).toReal - fam.wt (rows r) s| ≤ fam.err (rows r) s ∧
      lm (cv s) ≤ fam.lb (rows r) s) ∧
    ∀ j : Fin N, ∃ v : ℚ, DType.val? .f32 (acc.get2 r j) = some v ∧
      |(v : ℝ) - ∑ s ∈ S, (cv s).toReal * (fam.u s (cols j)).toReal| ≤
        accB t (∑ s ∈ S, fam.lb (rows r) s * lm (fam.u s (cols j)))

/-- The family of `dV[n, j] = Σ_m P[n, m]·dO[m, j]`. -/
noncomputable def famDV (d : Inst) (hv : HeadV) : Fam :=
  ⟨pE d hv, p16Err d hv, p16L d hv, fun s j => hv.dO s j⟩

/-- The family of `Σ_m dS[n, m]·Q[m, j]`, the accumulation of `dK`. -/
noncomputable def famDK (d : Inst) (hv : HeadV) : Fam :=
  ⟨dsE d hv, ds16Err d hv, ds16L d hv, fun s j => hv.q s j⟩

/-- The family of `Σ_n dS[n, m]·K[n, j]`, the accumulation of `dQ` (row `m`, summation index `n`). -/
noncomputable def famDQ (d : Inst) (hv : HeadV) : Fam :=
  ⟨fun m n => dsE d hv n m, fun m n => ds16Err d hv n m, fun m n => ds16L d hv n m,
    fun s j => hv.k s j⟩

/-- The `dV` and `dK` accumulators of the key rows `n0, …, n0 + 127` of head `b`, after `t`
steps over the query rows `S`, for every head whose inputs are in memory `init` and covered. -/
def DkdvInv (d : Inst) (init : Mem) (b n0 : Nat) (S : Finset Nat) (t : Nat)
    (dk dv : Tensor [128, d.HEAD_DIM] F32) : Prop :=
  ∀ hv : HeadV, hv.Holds d init b → hv.Ok d →
    AccInv (famDV d hv) (fun i : Fin 128 => n0 + i) (fun j : Fin d.HEAD_DIM => j.val) S t dv ∧
    AccInv (famDK d hv) (fun i : Fin 128 => n0 + i) (fun j : Fin d.HEAD_DIM => j.val) S t dk

/-- The `dQ` accumulator of the query rows `m0, …, m0 + 127` of head `b`, after `t` steps over
the key rows `S`. -/
def DqInv (d : Inst) (init : Mem) (b m0 : Nat) (S : Finset Nat) (t : Nat)
    (dq : Tensor [128, d.HEAD_DIM] F32) : Prop :=
  ∀ hv : HeadV, hv.Holds d init b → hv.Ok d →
    AccInv (famDQ d hv) (fun i : Fin 128 => m0 + i) (fun j : Fin d.HEAD_DIM => j.val) S t dq

end Claims.AttnBwd
