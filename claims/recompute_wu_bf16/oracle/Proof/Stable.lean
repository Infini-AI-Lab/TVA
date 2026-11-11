import Claim
import TVA.Proof

/-!
# Facts that survive a program's writes

`safe_and_inv`: if a program writes only inside `W ⊆ S`, any property of the memory that is stable
under changes confined to `S` (bytes outside `S` unchanged, present bytes stay present) holds at
the end as well as at the start.
-/

namespace Claims.RecomputeWU

open TVA

/-- **A stable property holds at the end of a safe program.** `P` holds in `m` and is preserved by
any change of the memory that leaves the bytes outside `S` alone and keeps present bytes present;
the program writes only inside `W ⊆ S`; then `P` holds in the final memory, beside `Q`. -/
theorem safe_and_inv {E : Env} {R W S : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {p : Prog α} (h : Safe E R W Q m p) (P : Mem → Prop)
    (hP : ∀ m m' : Mem, (∀ x, x ∉ S → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      P m → P m') (hm : P m) (hWS : W ⊆ S) :
    Safe E R W (fun a m' => Q a m' ∧ P m') m p := by
  induction h with
  | ret hq => exact .ret ⟨hq, hm⟩
  | load hb hRs _ ih => exact .load hb hRs fun v hv => ih v hv hm hWS
  | store hok hWs _ ih =>
    refine .store hok hWs fun order hperm => ih order hperm ?_ (Set.sdiff_subset.trans hWS)
    refine hP _ _ (fun x hx => ?_) (fun x hx => writeAll_present _ _ _ hx) hm
    exact writeAll_frame _ _ _ _ (fun hxa => hx (hWS (hWs hxa)))
  | rmw hex hS hSR hc _ ih =>
    refine .rmw hex hS hSR hc fun old new ho hn => ih old new ho hn ?_ (Set.sdiff_subset.trans hWS)
    refine hP _ _ (fun x hx => ?_) (fun x hx => Mem.write_present _ hx) hm
    exact Mem.write_frame _ _ _ _ (fun hxa => hx (hWS (hS hxa)))
  | pick hex _ ih => exact .pick hex fun b hb => ih b hb hm hWS
  | loop _ ih => exact .loop (ih hm hWS)
  | env _ ih => exact .env (ih hm hWS)

end Claims.RecomputeWU
