import Claim
import TVA.Proof
import Proof.Setup
import Proof.Spine

/-!
# The claim from the programs

`correct_of_safe` turns the per-program `safe_program` into `Correct` for the launch.
-/

namespace Claims.MoeRouter

open TVA

/-- **The claim**, from `safe_program`, the disjoint footprints and `hpost`. -/
theorem main_holds : claim.Holds := by
  intro P hP d hd init hpre
  have hsafe : ∀ i : Fin (launch d).size, Safe ((launch d).env P) (R d init i) (W d init i)
      (fun _ m => Q d init i m) init ((model d).body ((launch d).ctx i)) := fun i =>
    safe_program hd (hP.mathTotal _ _) hP.expF32 (hP.divTotal _) hP.divFull hP.sumDepth hpre i
  have hpost' : ∀ fin, (∀ i, ∃ m, Q d init i m ∧ ∀ x ∈ R d init i ∪ W d init i, fin x = m x) →
      (∀ x, (∀ i, x ∉ W d init i) → fin x = init x) → Post d init fin :=
    fun fin hloc hfr => hpost hd init hpre fin hloc hfr
  have h := correct_of_safe (R d init) (W d init) (Q d init) (launch_valid hd) hsafe
    (hdisj hd init hpre) (hRW hd init hpre) hpost'
  exact h
end Claims.MoeRouter
