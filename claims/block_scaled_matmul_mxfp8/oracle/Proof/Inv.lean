import Claim
import TVA.Proof
import Proof.Basic

/-!
# The loop's invariant

After step `t` of the `K`-loop, memory is the initial one, the accumulator's in-bounds lanes hold
finite values within the loop's tolerance of the partial sum over the first `128 · t` terms, and
the three counters have advanced by `128 · t` (bytes of `a`), `(128 / ELEM_PER_BYTE_B) · t`
(bytes of `b`) and `t` (scale blocks).
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- The state the loop carries: the accumulator and the three offsets. -/
def LoopState : Type := Tensor [128, 256] F32 × I32 × I32 × I32

/-- The loop's invariant at step `t`. -/
def Inv (d : Inst) (init : Mem) (i : Fin (launch d).size) (t : Nat)
    (st : LoopState) (m : Mem) : Prop :=
  m = init ∧
    (∀ l : Idx [128, 256], laneMask d i l = true →
      (st.1.get l).isFinite = true ∧
        |(st.1.get l).toRat - partSum (prods d init (laneRow d i l) (laneCol d i l)) t| ≤
          tol (prods d init (laneRow d i l) (laneCol d i l)) t) ∧
    st.2.1.toInt = ((128 * t : Nat) : Int) ∧
    st.2.2.1.toInt = (((128 / epbB d.mixed) * t : Nat) : Int) ∧
    st.2.2.2.toInt = ((t : Nat) : Int)

end Claims.BlockScaledMxfp8
