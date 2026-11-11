import Proof.PhaseA

/-!
# Phase B: the loop over the chunk's tokens

Walk of `phaseB`: the invariant `BInv` of the loop of line 397 (pointers of the four operands after
`n` trips, and `LoopInv` of every accumulator lane), one trip (`bstep_safe`), and the loop itself.
It leaves `BPost`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- **What phase B leaves**: when the inputs' values are covered, every valid lane of the
accumulator is finite, within `errSig + errW + errDot` of `exp(dSt)·Σ C·S + Σ wTS·x`, and of
magnitude at most `aAcc`. -/
def BPost (d : Inst) (init : Mem) (k pm pn h : Nat) (acc : Tensor [BM, BN] F32) : Prop :=
  Vals init d → ∀ ii jj : Fin _, RowOk d k pm ii → ColOk d pn jj →
    (acc.get2 ii jj).isFinite = true ∧
    |((acc.get2 ii jj).toRat : ℝ) - (Real.exp (dSt init d (tokOf d k pm ii) h) *
        sig init d (tokOf d k pm ii) h (colOf pn jj) +
        ∑ s ∈ toks d (tokOf d k pm ii), wTS init d (tokOf d k pm ii) h s *
          xv init d s h (colOf pn jj))| ≤
      errSig init d (tokOf d k pm ii) h (colOf pn jj) + errW init d (tokOf d k pm ii) h (colOf pn jj) +
        errDot init d (tokOf d k pm ii) h (colOf pn jj) ∧
    |((acc.get2 ii jj).toRat : ℝ)| ≤ aAcc init d (tokOf d k pm ii) h (colOf pn jj)

/-- **The loop invariant** after `n` trips: the memory is the initial one; the pointer tiles are at
columns `lo + 32·n + l` of the chunk's rows; every valid accumulator lane satisfies `LoopInv`. -/
def BInv (d : Inst) (init : Mem) (k pm pn h n : Nat) (st : LoopSt) (m : Mem) : Prop :=
  m = init ∧
  (∀ (ii : Fin BM) (l : Fin BK), rowOf d k pm ii < d.cs →
    st.2.1.get2 ii l = d.cb.ptr ((cbIdx d (lc d k) (grp d h) 0 0 + rowOf d k pm ii * d.cs +
      (lo d k + BK * n + l) * 1 : Nat) : Int)) ∧
  (∀ (l : Fin BK) (jj : Fin BN), colOf pn jj < d.P →
    st.2.2.1.get2 l jj = d.x.ptr ((xIdx d (lc d k * d.cs) h 0 + (lo d k + BK * n + l) * d.xT +
      colOf pn jj * 1 : Nat) : Int)) ∧
  (∀ l : Fin BK, st.2.2.2.1.get1 l =
    d.dt.ptr ((dtIdx d h (lc d k) 0 + (lo d k + BK * n + l) : Nat) : Int)) ∧
  (∀ l : Fin BK, st.2.2.2.2.get1 l =
    d.dA.ptr ((dtIdx d h (lc d k) 0 + (lo d k + BK * n + l) : Nat) : Int)) ∧
  (Vals init d → ∀ ii jj : Fin _, RowOk d k pm ii → ColOk d pn jj →
    LoopInv init d (lc d k) (lo d k) (tokOf d k pm ii) h (colOf pn jj) n (st.1.get2 ii jj))

end Claims.SsdChunkScan
