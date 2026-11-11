import Proof.Core
import Proof.Zero
import Proof.NumState
import Proof.NumLoop
import Proof.NumFinal

/-!
# What the walk of a program carries

The device facts the walk uses, the standing hypotheses on a program `(k, pm, pn, h)`, and the
statements the phases leave: `APost` (the state term `acc₀`) and `BPost` (the accumulator after the
loop).
-/

namespace Claims.SsdChunkScan

open TVA

/-- The device facts the walk uses: `tl.exp` and `tl.dot` have results, their accuracy, and the
launch fuses additions. -/
structure Dev (E : Env) : Prop where
  expTotal : Device.MathTotal E.P .exp .binary32
  dotTotal : Device.DotTotal E.P cfgBF16
  expAcc : Device.ExpF32 E.P
  dotAcc : Device.DotBF16 E.P
  fusion : E.fusion = true

/-- The standing hypotheses on program `(k, pm, pn, h)`: logical chunk `k`, row tile `pm`, column
tile `pn`, head `h`. -/
structure Std (d : Inst) (init : Mem) (E : Env) (k pm pn h : Nat) : Prop where
  hd : d.ok
  hpre : Pre d init
  dev : Dev E
  hk : k < d.nmeta
  hpm : pm < nmt d
  hpn : pn < npn d
  hh : h < d.H

/-- **What phase A leaves**: when the inputs' values are covered, every valid lane of `acc₀` is
finite, within `errSig` of `exp(dSt)·Σ_n C·S` and of magnitude at most `aSig`. -/
def APost (d : Inst) (init : Mem) (k pm pn h : Nat) (acc : Tensor [BM, BN] F32) : Prop :=
  Vals init d → ∀ ii jj : Fin _, RowOk d k pm ii → ColOk d pn jj →
    (acc.get2 ii jj).isFinite = true ∧
    |((acc.get2 ii jj).toRat : ℝ) - Real.exp (dSt init d (tokOf d k pm ii) h) *
        sig init d (tokOf d k pm ii) h (colOf pn jj)| ≤
      errSig init d (tokOf d k pm ii) h (colOf pn jj) ∧
    |((acc.get2 ii jj).toRat : ℝ)| ≤ aSig init d (tokOf d k pm ii) h (colOf pn jj)

end Claims.SsdChunkScan
