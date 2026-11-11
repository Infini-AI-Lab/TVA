import Proof.Basic

/-!
# The launch is valid
-/

namespace Claims.RotaryVarlen

open TVA

/-- A one-dimensional block shape is Triton's when its dimension is a power of two and at most
`2^20`. -/
private theorem tritonOk_one {a : Nat} (ha : ∃ e, a = 2 ^ e) (h : a ≤ 2 ^ 20) :
    Shape.TritonOk [a] := by
  refine ⟨?_, by simpa using h⟩
  intro k hk
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk
  subst hk
  exact ha

/-- A two-dimensional block shape is Triton's when both dimensions are powers of two and their
product is at most `2^20`. -/
private theorem tritonOk_pair {a b : Nat} (ha : ∃ e, a = 2 ^ e) (hb : ∃ e, b = 2 ^ e)
    (h : a * b ≤ 2 ^ 20) : Shape.TritonOk [a, b] := by
  refine ⟨?_, by simpa using h⟩
  intro k hk
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk
  rcases hk with rfl | rfl
  · exact ha
  · exact hb

/-- Every block shape the kernel uses is one Triton accepts. -/
theorem blockShapes_triton {d : Inst} (hd : d.ok) :
    ∀ s ∈ blockShapes d, Shape.TritonOk s := by
  obtain ⟨eb, hb⟩ := hd.bt_pow2
  obtain ⟨ed, hbd⟩ := hd.bd_pow2
  have hbt1 : 1 ≤ d.BT := hb ▸ Nat.one_le_two_pow
  have hbd2 : 2 ≤ d.BD := by
    have h1 := hd.r_bd
    have h2 := hd.r_pos
    omega
  have hbd1 : 1 ≤ d.BD := by omega
  have hed1 : 1 ≤ ed := by
    rcases Nat.eq_zero_or_pos ed with h | h
    · subst h
      simp at hbd
      omega
    · exact h
  have hhalf : d.BD / 2 = 2 ^ (ed - 1) := by
    have he : 2 ^ ed = 2 ^ (ed - 1) * 2 := by
      conv_lhs => rw [show ed = (ed - 1) + 1 from (Nat.sub_add_cancel hed1).symm]
      rw [pow_succ]
    rw [hbd, he, Nat.mul_div_cancel _ (by norm_num)]
  have hhalf_le : d.BD / 2 ≤ d.BD := Nat.div_le_self _ _
  have hbl := hd.block_le
  have hBT : d.BT ≤ 2 ^ 20 := by
    have h : d.BT ≤ d.BT * d.BD := by simpa using Nat.mul_le_mul_left d.BT hbd1
    omega
  have hBD : d.BD ≤ 2 ^ 20 := by
    have h : d.BD ≤ d.BT * d.BD := by simpa using Nat.mul_le_mul_right d.BD hbt1
    omega
  have hHalf : d.BD / 2 ≤ 2 ^ 20 := le_trans hhalf_le hBD
  intro s hs
  simp only [blockShapes] at hs
  split at hs
  · simp only [List.mem_cons, List.mem_nil_iff, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl
    · exact tritonOk_one ⟨eb, hb⟩ hBT
    · exact tritonOk_pair ⟨eb, hb⟩ ⟨0, by norm_num⟩ (by simpa using hBT)
    · exact tritonOk_one ⟨ed, hbd⟩ hBD
    · exact tritonOk_pair ⟨0, by norm_num⟩ ⟨ed, hbd⟩ (by simpa using hBD)
    · exact tritonOk_pair ⟨eb, hb⟩ ⟨ed, hbd⟩ hd.block_le
  · simp only [List.mem_cons, List.mem_nil_iff, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl
    · exact tritonOk_one ⟨eb, hb⟩ hBT
    · exact tritonOk_pair ⟨eb, hb⟩ ⟨0, by norm_num⟩ (by simpa using hBT)
    · exact tritonOk_one ⟨ed - 1, hhalf⟩ hHalf
    · exact tritonOk_pair ⟨0, by norm_num⟩ ⟨ed - 1, hhalf⟩ (by simpa using hHalf)
    · exact tritonOk_pair ⟨eb, hb⟩ ⟨ed - 1, hhalf⟩ (le_trans (Nat.mul_le_mul_left _ hhalf_le) hd.block_le)

/-- Every covered launch is valid: its grid fits the limits and its block shapes are Triton's. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, blockShapes_triton hd⟩
  · simpa [launch] using hd.grid_t
  · simpa [launch] using hd.grid_b
  · simpa [launch] using hd.grid_h
  · simpa [launch, Launch.size] using hd.grid_size

end Claims.RotaryVarlen
