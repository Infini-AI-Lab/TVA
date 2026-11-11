import Proof.Defs
import Proof.Numeric
import Proof.Pid
import Proof.Prog
import Proof.Spec

/-!
# `lora_shrink_bf16`: the launch

Every program is `SafeA` (`safe_program`) with the private reads `Rall`, no private writes and the
output elements as shared cells; `correct_of_safeA` composes them, and its last goal is `Post`
(`post_of_decomp`).
-/

namespace Claims.LoraShrink

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safeA (fun _ => Rall d) (fun _ => (∅ : Set Int)) (Cells d)
    (fun i log m => Q d init i log m) (launch_valid hd) (fun i => safe_program hP hd hpre i)
    (fun i => Q_permInv d init i) (fun i j _ => Set.empty_disjoint _)
    (fun i => Set.disjoint_empty _) (fun i => by simpa using cells_disjoint_Rall d hd)
    (cells_distinct d) (cells_present d hd hpre) ?_
  intro fin G hQ hrep hfr
  exact post_of_decomp hd hpre fin G (fun i => (hQ i).imp fun m h => h.1) hrep
    (fun x hx => hfr x hx fun i => Set.notMem_empty x)

end Claims.LoraShrink
