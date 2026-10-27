import TVA.Rules.Compiler
import TVA.Lemmas.Independent

/-!
# From the semantics to the claim

`meets_of_faithful`: if every launch of the model, from memory satisfying `Pre`, is well
behaved and has only normal outcomes satisfying `Post`, then `Compiler.Faithful` gives the
claim about the device.

`meets_independent`: the same claim from per-program proofs (`Safe`) with footprints that do
not conflict across programs (`TVA.Lemmas.Independent`). Most kernel proofs end with this
lemma.

`Launch.valid_grid1`: a one-dimensional grid of fewer than `2^31` programs, with block shapes
Triton accepts (and the defaults `num_ctas = 1`, no programmatic dependent launch), is
`Launch.Valid` (both lemmas need validity).
-/
namespace TVA

open TVA.Claim

/-- A one-dimensional grid of `n` programs has `n` programs. -/
theorem Launch.size_grid1 (n : Nat) (blocks : List Shape) (fusion : Bool) :
    (Launch.grid1 n blocks fusion).size = n := by
  simp [Launch.size, Launch.grid1]

/-- A one-dimensional launch is valid when it has fewer than `2^31` programs and Triton accepts
its block shapes: the bound on `x` is also the bound on the program count (`grid1` keeps the
defaults `num_ctas = 1` and `launch_pdl = False`). -/
theorem Launch.valid_grid1 {n : Nat} {blocks : List Shape} {fusion : Bool} (hn : n < 2 ^ 31)
    (hb : ∀ s ∈ blocks, s.TritonOk) : (Launch.grid1 n blocks fusion).Valid :=
  ⟨by simpa [Launch.grid1] using hn, by simp [Launch.grid1], by simp [Launch.grid1],
    by rw [Launch.size_grid1]; exact hn, rfl, rfl, hb⟩

theorem meets_of_faithful {Inst : Type} {ok : Inst → Prop} {Pre : Inst → Mem → Prop}
    {Post : Inst → Mem → Mem → Prop} {model : Inst → Kernel} {launch : Inst → Launch}
    {K : Behaviour Inst} {P : Platform}
    (hf : ∀ d, ok d → Rules.Compiler.Faithful P (model d) (launch d) (K d))
    (hdebug : ∀ d, ok d → (launch d).debug = false)
    (hvalid : ∀ d, ok d → (launch d).Valid)
    (h : ∀ d, ok d → ∀ init, Pre d init →
      WellBehaved ((launch d).env P) (model d) (launch d) init ∧
      ∀ o, Outcomes ((launch d).env P) (model d) (launch d) init o →
        ∃ fin, o = .ok fin ∧ Post d init fin) :
    Meets ok Pre Post K := by
  intro d hd init hpre o hK
  obtain ⟨hwb, hout⟩ := h d hd init hpre
  have := hf d hd (hdebug d hd) (hvalid d hd) init hwb o hK
  cases o with
  | ok fin =>
    obtain ⟨fin', heq, hpost⟩ := hout _ this
    cases heq
    exact ⟨fin, rfl, hpost⟩
  | fault => exact this.elim
  | hang =>
    obtain ⟨_, heq, -⟩ := hout _ this
    cases heq

/-- **The claim from independent programs.** For each covered instance and initial memory:
program `i` is safe alone with footprint `R i`/`W i` and leaves `Q i` in its own memory; no
program writes into another's footprint; and the programs' results together give `Post`. -/
theorem meets_independent {Inst : Type} {ok : Inst → Prop} {Pre : Inst → Mem → Prop}
    {Post : Inst → Mem → Mem → Prop} {model : Inst → Kernel} {launch : Inst → Launch}
    {K : Behaviour Inst} {P : Platform}
    (hf : ∀ d, ok d → Rules.Compiler.Faithful P (model d) (launch d) (K d))
    (hdebug : ∀ d, ok d → (launch d).debug = false)
    (hvalid : ∀ d, ok d → (launch d).Valid)
    (R W : (d : Inst) → Mem → Fin (launch d).size → Set Int)
    (Q : (d : Inst) → Mem → Fin (launch d).size → Mem → Prop)
    (hsafe : ∀ d, ok d → ∀ init, Pre d init → ∀ i,
      Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
        ((model d).body ((launch d).ctx i)))
    (hdisj : ∀ d, ok d → ∀ init, Pre d init → ∀ i j, i ≠ j →
      Disjoint (W d init i) (R d init j ∪ W d init j))
    (hpost : ∀ d, ok d → ∀ init, Pre d init → ∀ fin,
      (∀ i, ∃ m, Q d init i m ∧ ∀ x ∈ R d init i ∪ W d init i, fin x = m x) →
      (∀ x, (∀ i, x ∉ W d init i) → fin x = init x) → Post d init fin) :
    Meets ok Pre Post K := by
  refine meets_of_faithful hf hdebug hvalid fun d hd init hpre => ?_
  obtain ⟨hwb, hout⟩ := independent (hsafe d hd init hpre) (hdisj d hd init hpre)
  refine ⟨hwb, fun o ho => ?_⟩
  obtain ⟨fin, rfl, hloc, hfr⟩ := hout o ho
  exact ⟨fin, rfl, hpost d hd init hpre fin hloc hfr⟩

end TVA
