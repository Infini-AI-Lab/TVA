import Claim
import TVA.Proof
import Proof.Basic
import Proof.Idx
import Proof.Spec
import Proof.Safe

/-!
# One program is safe

`safe_program` is `safe_body` (`Proof/Safe.lean`) at the launch's environment and a platform `P` satisfying the device facts.
-/

set_option maxHeartbeats 1000000

namespace Claims.GdnGating

open TVA

/-- Every program is safe on its own, for the platform `P`. -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (Rset d i) (Wset d i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) :=
  safe_body hd hpre _ hP.expF32 hP.logF32 hP.divFull (hP.mathTotal _ _) (hP.mathTotal _ _)
    (hP.divTotal _) hP.expF32NegInf i

end Claims.GdnGating
