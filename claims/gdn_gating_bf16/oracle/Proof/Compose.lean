import Claim
import TVA.Proof
import Proof.Basic
import Proof.Idx
import Proof.Spec
import Proof.Safe
import Proof.Program

/-!
# The launch is correct

`correct` composes the programs of the launch with `correct_of_safe`: each program is safe (`safe_program`), the footprints are disjoint (`hdisj_spec`, `hRW_spec`) and the local postconditions give `Post` (`hpost_spec`). `correct_claim` states it in the words of `claim.Holds`.
-/

set_option maxHeartbeats 1000000

namespace Claims.GdnGating

open TVA

/-- The launch is correct: the programs' footprints are disjoint and their postconditions give `Post`. -/
theorem correct {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) : Correct ((launch d).env P) (model d) (launch d) init (Post d init) :=
  correct_of_safe (fun i => Rset d i) (fun i => Wset d i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => safe_program hP hd hpre i) (fun _ _ hij => hdisj_spec hd hij)
    (fun i => hRW_spec hd i) (fun _ hloc hfr => hpost_spec hloc hfr)

/-- The launch is correct, in the words of `claim.Holds`. -/
theorem correct_claim {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) :
    Correct ((claim.launch d).env P) (claim.kernel d) (claim.launch d) init (claim.Post d init) :=
  correct hP hd hpre

end Claims.GdnGating
