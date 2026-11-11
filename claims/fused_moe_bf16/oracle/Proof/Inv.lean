import TVA.Proof
import Claim
import Proof.Pid
import Proof.Foot
import Proof.Defs
import Proof.Tol
import Proof.Store

/-!
# The loop's invariant and the two pieces around it

`InInv` is the invariant of the `K` loop after `t` trips: the memory is the initial one, the tile
pointers are at trip `t`, and every accumulator lane of a routed row is within `accErr` of the
exact partial sum of its row of `A` and its column of `B`. `trip_safe` keeps it for one trip
(`tripBody`); `tail_safe` turns the final accumulator into the stored tile (`tailOf`).
-/

namespace Claims.FusedMoe

open TVA

/-- The row of `A` that lane `l` of block `pmN` reads: its flat index divided by `top_k`. -/
noncomputable def arow (d : Inst) (init : Mem) (pmN l : Nat) : Nat :=
  (entryOf d init (pmN * d.BM + l)).toNat / d.top_k

/-- The column of `B` that lane `j` of column tile `pnN` reads: `(pid_n * BN + j) % N`. -/
def bcol (d : Inst) (pnN j : Nat) : Nat := (pnN * d.BN + j) % d.N

/-- Every accumulator lane of a routed row, whose operands' values along `K` are finite with
`dotMag ≤ 2^120`, is finite and within `accErr` of the exact partial sum after `t` trips. -/
def AccInv (d : Inst) (init : Mem) (pmN pnN e t : Nat) (acc : Tensor [d.BM, d.BN] F32) : Prop :=
  ∀ (l : Fin d.BM) (j : Fin d.BN), entryOf d init (pmN * d.BM + l) < (nv d : Int) →
    ∀ va vb : Nat → BF16,
      (∀ k, k < d.K → init.read .bf16 (d.a.ptr ((arow d init pmN l * d.K + k : Nat) : Int)).addr =
          some (va k) ∧ (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .bf16
          (d.b.ptr ((e * d.stride_be + bcol d pnN j * d.stride_bn + k : Nat) : Int)).addr =
          some (vb k) ∧ (vb k).isFinite = true) →
      dotMag va vb d.K ≤ 2 ^ 120 →
      ∃ v : ℚ, DType.val? .f32 (acc.get2 l j) = some v ∧
        |v - preSum va vb d.K (t * (d.BK / 64))| ≤ accErr d.BK d.K va vb t

/-- The loop's invariant after `t` trips. -/
def InInv (d : Inst) (init : Mem) (pmN pnN e t : Nat) (st : St d) (m : Mem) : Prop :=
  m = init ∧
  (∀ (l : Fin d.BM) (kk : Fin d.BK),
    st.1.get2 l kk = d.a.ptr ((arow d init pmN l * d.K + (t * d.BK + kk) : Nat) : Int)) ∧
  (∀ (kk : Fin d.BK) (j : Fin d.BN),
    st.2.1.get2 kk j =
      d.b.ptr ((e * d.stride_be + bcol d pnN j * d.stride_bn + (t * d.BK + kk) : Nat) : Int)) ∧
  AccInv d init pmN pnN e t st.2.2

end Claims.FusedMoe
