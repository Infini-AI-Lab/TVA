import Proof.Basic

/-!
# `write_req_to_token_pool_i32`: the programs' results give `Post`

Each written table element is in its program's write footprint, where the final memory agrees
with that program's memory; every other byte is written by no program and keeps its initial
value.
-/

namespace Claims.WriteReqToTokenPool

open TVA

/-- A table element of a request's row window is in that program's write footprint. -/
theorem span_sub_W {d : Inst} {init : Mem} {b : Nat} {j : Nat} (hj0 : (j : Int) < seq d init b)
    (hr0 : 0 ≤ rpi d init b * (d.max_context_len : Int)) :
    Mem.span .i32 (d.req_to_token.ptr (rpi d init b * (d.max_context_len : Int) + j)).addr ⊆
      W d init b := by
  have e : rpi d init b * (d.max_context_len : Int) + j =
      (((rpi d init b * (d.max_context_len : Int) + j).toNat : Nat) : Int) :=
    (Int.toNat_of_nonneg (by positivity)).symm
  rw [e]
  refine d.req_to_token.span_sub_elems ⟨j, hj0, ?_⟩
  rw [← e]

/-- **The specification** from the programs' results. -/
theorem post_of {d : Inst} {init : Mem} (hpre : Pre d init) (fin : Mem)
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d init (i : Nat) m ∧
      ∀ x ∈ R d init (i : Nat) ∪ W d init (i : Nat), fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d init (i : Nat)) → fin x = init x) :
    Post d init fin := by
  have hfin : ∀ b, b < d.bs → ∃ m, Q d init b m ∧ ∀ x ∈ W d init b, fin x = m x := by
    intro b hb
    obtain ⟨m, hQ, hag⟩ := hloc ⟨b, by rw [launch_size]; exact hb⟩
    exact ⟨m, hQ, fun x hx => hag x (Or.inr hx)⟩
  refine ⟨?_, ?_, ?_⟩
  · intro b hb j hj
    obtain ⟨hp0, hps, -, hr0, -⟩ := hpre.req b hb
    obtain ⟨m, ⟨hQ1, -⟩, hag⟩ := hfin b hb
    obtain ⟨v, hv1, hv2⟩ := hQ1 j hj
    have hrow : 0 ≤ rpi d init b * (d.max_context_len : Int) := by positivity
    refine ⟨v, by simpa using hv1, ?_⟩
    rw [Mem.read_congr fun x hx => hag x (span_sub_W (by linarith) hrow hx)]
    exact hv2
  · intro b hb j hj1 hj2
    obtain ⟨hp0, hps, -, hr0, -⟩ := hpre.req b hb
    obtain ⟨m, ⟨-, hQ2⟩, hag⟩ := hfin b hb
    have hrow : 0 ≤ rpi d init b * (d.max_context_len : Int) := by positivity
    have ho : (((j : Int) - pre d init b).toNat : Int) = (j : Int) - pre d init b :=
      Int.toNat_of_nonneg (by linarith)
    obtain ⟨v, hv1, hv2⟩ := hQ2 ((j : Int) - pre d init b).toNat (by rw [ho]; linarith)
    rw [ho] at hv1 hv2
    refine ⟨v, hv1, ?_⟩
    rw [Mem.read_congr fun x hx => hag x (span_sub_W hj2 hrow hx)]
    rw [show rpi d init b * (d.max_context_len : Int) + pre d init b + ((j : Int) - pre d init b) =
      rpi d init b * (d.max_context_len : Int) + j by ring] at hv2
    exact hv2
  · intro z hz
    refine hfr z fun i hi => hz ?_
    obtain ⟨e, ⟨o, ho, he⟩, hx⟩ := hi
    have hb : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
    exact ⟨e, ⟨i, hb, o, ho, he⟩, hx⟩

end Claims.WriteReqToTokenPool
