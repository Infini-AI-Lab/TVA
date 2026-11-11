import Proof.Private
import Proof.Post

/-!
# `route_histogram_i32`: the launch, and the claim

Every program of the launch is `SafeA` (`safe_private`) with the private reads `Rpriv`, no private
writes and the shared cells `cells d`. `correct_of_safeA` composes them; its last goal is `Post`,
from the per-program `Qset` facts and the replay of all logs (`post_of_shared`).
-/

namespace Claims.RouteHistogram

open TVA
open TVA.Block

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hbody : ∀ i : Fin (launch d).size,
      SafeA ((launch d).env P) (Rpriv d) ∅ (cells d)
        (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Qset d (i : Nat) log m) [] init
        ((model d).body ((launch d).ctx i)) := by
    intro i
    rw [body_pid (d : Inst) i]
    exact safe_private hd (launch_size (d : Inst) ▸ i.isLt) hpre
  refine correct_of_safeA (fun _ => Rpriv d) (fun _ => (∅ : Set Int)) (cells d)
    (fun i log m => Qset d (i : Nat) log m) (launch_valid hd) hbody
    (fun i => Qset_perm d (i : Nat)) (fun i j _ => Set.empty_disjoint _)
    (fun i => Set.disjoint_empty _) (fun i => by simpa using disjoint_cellBytes_Rpriv hd)
    (cells_distinct d) (cells_present hd hpre) ?_
  intro fin G hQ hrep hframe
  obtain ⟨mA, hrep, hAgree⟩ := hrep
  refine post_of_shared hd (fun i => ?_) hrep hAgree ?_ ?_
  · obtain ⟨m, hm, hx⟩ := hQ i
    exact ⟨m, hm, fun x hx' => hx x (Or.inl hx')⟩
  · intro x hx
    exact hframe x hx (fun i h => h)
  · intro b hb
    exact hpre.2.2.2.1 b (Nat.lt_of_lt_of_le hb hd.counts_len)

end Claims.RouteHistogram
