import Claim
import TVA.Proof
import Proof.Grid

/-!
# The specification from the programs' results

When every program ends with all its slots done, and the final memory agrees with each program's
memory on its footprint and with the initial memory elsewhere, `Post` holds.
-/
namespace Claims.SsdStatePassing

open TVA

/-- The program owning state element `(h, m)`. -/
theorem owner {d : Inst} (hd : d.ok) {h m : Nat} (hh : h < d.nheads) (hm : m < d.dim) :
    ∃ i : Fin (launch d).size, (i : Nat) % gx d = m / BL ∧ (i : Nat) / gx d = h := by
  have hg := gx_pos hd
  have hpm : m / BL < gx d := by
    unfold gx
    rw [BL_eq]
    omega
  have hlt : m / BL + gx d * h < (launch d).size := by
    rw [launch_size_eq]
    have : (h + 1) * gx d ≤ d.nheads * gx d := Nat.mul_le_mul_right _ hh
    rw [Nat.succ_mul] at this
    rw [Nat.mul_comm (gx d) h]
    nlinarith
  refine ⟨⟨m / BL + gx d * h, hlt⟩, ?_, ?_⟩
  · show (m / BL + gx d * h) % gx d = _
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hpm]
  · show (m / BL + gx d * h) / gx d = _
    rw [Nat.add_mul_div_left _ _ hg, Nat.div_eq_of_lt hpm]
    simp

/-- A lane belongs to its program's block. -/
theorem mem_lanes {d : Inst} {m : Nat} (hm : m < d.dim) : m ∈ lanes d (m / BL) := by
  refine ⟨?_, ?_, hm⟩ <;> rw [BL_eq] <;> omega

/-- **The specification**: the slots done by every program give `Post`. -/
theorem post_of_done {d : Inst} (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ mem, Done d init ((i : Nat) % gx d) ((i : Nat) / gx d) d.nchunks mem ∧
      ∀ x ∈ Rin d ∪ Wp d ((i : Nat) / gx d) ((i : Nat) % gx d), fin x = mem x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ Wp d ((i : Nat) / gx d) ((i : Nat) % gx d)) →
      fin x = init x) : Post d init fin := by
  have hn0 := nchunks_pos hd
  -- the slots of lane `m` of head `h`, read in the final memory
  have key : ∀ {h m : Nat}, h < d.nheads → m < d.dim → ∃ mem, Done d init (m / BL) h d.nchunks mem ∧
      ∀ c, c ≤ d.nchunks → slotRead d h c m fin = slotRead d h c m mem := by
    intro h m hh hm
    obtain ⟨i, hi1, hi2⟩ := owner hd hh hm
    obtain ⟨mem, hDone, hagree⟩ := hloc i
    rw [hi1, hi2] at hDone hagree
    refine ⟨mem, hDone, fun c hc => ?_⟩
    have hs := slot_span_sub (h := h) (c := c) (mem_lanes hm)
    unfold slotRead
    by_cases h1 : c < d.nchunks
    · simp only [h1, ite_true] at hs ⊢
      exact Mem.read_congr fun x hx => hagree x (Or.inr ⟨c, by omega, hs hx⟩)
    · simp only [h1, ite_false] at hs ⊢
      exact Mem.read_congr fun x hx => hagree x (Or.inr ⟨c, by omega, hs hx⟩)
  refine ⟨fun h m hh hm => ?_, fun h m hh hm => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
  · obtain ⟨mem, hDone, hr⟩ := key hh hm
    refine ⟨?_, fun c hc1 hc2 => ?_⟩
    · obtain ⟨v, hv, hz⟩ := hDone.1 m (mem_lanes hm)
      refine ⟨v, ?_, hz⟩
      have := hr 0 (by omega)
      rw [hv] at this
      simpa [slotRead, hn0] using this
    · obtain ⟨v, hv, hq⟩ := hDone.2 c hc1 (by omega) m (mem_lanes hm)
      refine ⟨v, ?_, hq⟩
      have := hr c (by omega)
      rw [hv] at this
      simpa [slotRead, hc2] using this
  · obtain ⟨mem, hDone, hr⟩ := key hh hm
    obtain ⟨v, hv, hq⟩ := hDone.2 d.nchunks (by omega) le_rfl m (mem_lanes hm)
    refine ⟨v, ?_, hq⟩
    have := hr d.nchunks le_rfl
    rw [hv] at this
    simpa [slotRead] using this
  · exact Wp_sub (grid_h_lt i) hzi

end Claims.SsdStatePassing
