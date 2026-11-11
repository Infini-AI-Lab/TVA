import Claim
import TVA.Proof
import Proof.GeoBound

/-!
# The launch

The launch is a valid grid of `nChunks · nRows` programs, and program number `i` has
`program_id(0) = i % nChunks` and `program_id(1) = i / nChunks`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The grid has `nChunks · nRows` programs. -/
theorem launch_size (d : Inst) : (launch d).size = nChunks d * nRows d := by
  simp [Launch.size, launch]

/-- Program `i`'s first grid coordinate. -/
theorem launch_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = i % nChunks d := by
  simp [Launch.ctx, launch]

/-- Program `i`'s second grid coordinate. -/
theorem launch_pid1 (d : Inst) (i : Nat) :
    ((launch d).ctx i).pid 1 = i / nChunks d % nRows d := by
  simp [Launch.ctx, launch]

/-- A power of two `≤ 128` and a second one: a block shape of the launch is within Triton's limits. -/
private theorem tritonOk_of_pow2 {s : Shape} (h : ∀ k ∈ s, ∃ e, k = 2 ^ e ∧ e ≤ 7) (hl : s.length ≤ 2) :
    s.TritonOk := by
  refine ⟨fun k hk => (h k hk).imp fun e he => he.1, ?_⟩
  have hk : ∀ k ∈ s, k ≤ 2 ^ 7 := fun k hk => by
    obtain ⟨e, rfl, he⟩ := h k hk
    exact Nat.pow_le_pow_right (by norm_num) he
  match s, hl with
  | [], _ => simp
  | [a], _ => have := hk a (by simp); simp; omega
  | [a, b], _ =>
    have ha := hk a (by simp)
    have hb := hk b (by simp)
    simp only [List.prod_cons, List.prod_nil, Nat.mul_one]
    calc a * b ≤ 2 ^ 7 * 2 ^ 7 := Nat.mul_le_mul ha hb
      _ ≤ 2 ^ 20 := by norm_num

/-- The launch is valid. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  obtain ⟨a, ha⟩ := hd.bk_pow2
  obtain ⟨b, hb⟩ := hd.bv_pow2
  have hae : a ≤ 7 := by
    by_contra h
    have : 2 ^ 8 ≤ 2 ^ a := Nat.pow_le_pow_right (by norm_num) (by omega)
    have := hd.bk_le
    omega
  have hbe : b ≤ 7 := by
    by_contra h
    have : 2 ^ 8 ≤ 2 ^ b := Nat.pow_le_pow_right (by norm_num) (by omega)
    have := hd.bv_le
    omega
  have hc : ∃ c, d.BT = 2 ^ c ∧ c ≤ 7 := by
    rcases hd.bt_ok with h | h | h
    · exact ⟨4, by omega, by omega⟩
    · exact ⟨5, by omega, by omega⟩
    · exact ⟨6, by omega, by omega⟩
  obtain ⟨c, hc, hce⟩ := hc
  have h1 : ∃ e, (1 : Nat) = 2 ^ e ∧ e ≤ 7 := ⟨0, rfl, by norm_num⟩
  have hT : ∃ e, d.BT = 2 ^ e ∧ e ≤ 7 := ⟨c, hc, hce⟩
  have hK : ∃ e, d.BK = 2 ^ e ∧ e ≤ 7 := ⟨a, ha, hae⟩
  have hV : ∃ e, d.BV = 2 ^ e ∧ e ≤ 7 := ⟨b, hb, hbe⟩
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, ?_⟩
  · simpa [launch] using hd.grid0_lt
  · simpa [launch, nRows] using hd.grid1_lt
  · simp [launch]
  · rw [launch_size]; exact hd.grid_lt
  · intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    refine tritonOk_of_pow2 ?_ ?_ <;>
      rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      first
        | (intro k hk; simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
           rcases hk with rfl | rfl <;> assumption)
        | simp
end Claims.ChunkBwdDvLocal
