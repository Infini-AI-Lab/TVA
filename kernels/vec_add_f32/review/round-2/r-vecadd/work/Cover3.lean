import VecAdd

/-! r-vecadd, round 2, third filing: `Inst.ok` against the host's use, at gate 990243026f1289fb.

The host's launch: three fresh contiguous `float32` buffers of `n` elements (`torch.rand`,
`torch.rand`, `torch.empty_like`), `BLOCK_SIZE = 1024`, `n` from `98432` (line 85) and
`2^12 … 2^27` (line 110). Checked: `Inst.ok` holds there, at the edges `n = 0`, `n = 1` and
`n = 2^31 - 1` (with `BLOCK_SIZE = 1` and `2^20`), and for `add(x, x)`; it fails at `n = 2^31` and
for an output that is an input (in place), as the README says. The grid formula gives the host's
`triton.cdiv(n, 1024)` at its sizes. -/

open TVA Kernels.VecAdd

/-- Three buffers of `n` elements at `2^40`, `2^41`, `2^42` (any 4-aligned, disjoint ones do). -/
def hostInst (n B : Nat) : Inst where
  n := n
  BLOCK_SIZE := B
  x := ⟨2 ^ 40, n⟩
  y := ⟨2 ^ 41, n⟩
  output := ⟨2 ^ 42, n⟩

theorem host_ok (n e : Nat) (hn : n < 2 ^ 31) (he : e ≤ 20) : (hostInst n (2 ^ e)).ok := by
  have hB : 2 ^ e ≤ 2 ^ 20 := Nat.pow_le_pow_right (by norm_num) he
  refine ⟨⟨e, rfl⟩, hB, hn, le_refl _, le_refl _, le_refl _, ?_, ?_⟩
  · exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by simp [hostInst]; omega)
  · exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by simp [hostInst]; omega)

-- the host's sizes, with BLOCK_SIZE = 1024 = 2^10
example : (hostInst 98432 1024).ok := host_ok 98432 10 (by norm_num) (by norm_num)
example : (hostInst (2 ^ 12) 1024).ok := host_ok _ 10 (by norm_num) (by norm_num)
example : (hostInst (2 ^ 27) 1024).ok := host_ok _ 10 (by norm_num) (by norm_num)
-- the edges
example : (hostInst 0 1024).ok := host_ok 0 10 (by norm_num) (by norm_num)
example : (hostInst 1 1024).ok := host_ok 1 10 (by norm_num) (by norm_num)
example : (hostInst (2 ^ 31 - 1) 1).ok := host_ok _ 0 (by norm_num) (by norm_num)
example : (hostInst (2 ^ 31 - 1) (2 ^ 20)).ok := host_ok _ 20 (by norm_num) (by norm_num)

/-- `add(x, x)`: both inputs one buffer. -/
example : ({ hostInst 98432 1024 with y := ⟨2 ^ 40, 98432⟩ } : Inst).ok := by
  refine ⟨⟨10, rfl⟩, by simp [hostInst], by simp [hostInst], le_refl _, le_refl _, le_refl _, ?_, ?_⟩
  · exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by simp [hostInst])
  · exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by simp [hostInst])

/-- `n = 2^31` is outside (`n_elements` would be an `int64`). -/
example : ¬ (hostInst (2 ^ 31) 1024).ok := fun h => absurd h.n_lt (by simp [hostInst])

/-- An in-place call (`output` is `x`) is outside. -/
example : ¬ ({ hostInst 98432 1024 with output := ⟨2 ^ 40, 98432⟩ } : Inst).ok := by
  intro h
  have := h.output_x
  rw [Set.disjoint_left] at this
  refine this (a := 2 ^ 40) ⟨0, by simp [hostInst], ?_⟩ ⟨0, by simp [hostInst], ?_⟩ <;>
    simp [hostInst, Mem.span, DType.bytes]

-- the grid: triton.cdiv(98432, 1024) = 97, cdiv(2^27, 1024) = 2^17, cdiv(0, 1024) = 0,
-- cdiv(1, 1024) = 1, cdiv(2^31 - 1, 2^20) = 2^11, cdiv(2^31 - 1, 1) = 2^31 - 1
example : (launch (hostInst 98432 1024)).grid 0 = 97 := by decide
example : (launch (hostInst (2 ^ 27) 1024)).grid 0 = 2 ^ 17 := by decide
example : (launch (hostInst 0 1024)).grid 0 = 0 := by decide
example : (launch (hostInst 1 1024)).grid 0 = 1 := by decide
example : (launch (hostInst (2 ^ 31 - 1) (2 ^ 20))).grid 0 = 2 ^ 11 := by decide
example : (launch (hostInst (2 ^ 31 - 1) 1)).grid 0 = 2 ^ 31 - 1 := by decide
example : (launch (hostInst 98432 1024)).grid 1 = 1 ∧ (launch (hostInst 98432 1024)).grid 2 = 1 := by
  decide
example (d : Inst) : (launch d).debug = false ∧ (launch d).fusion = true ∧ (launch d).numCtas = 1 ∧
    (launch d).pdl = false ∧ (launch d).blocks = [[d.BLOCK_SIZE]] := ⟨rfl, rfl, rfl, rfl, rfl⟩

-- the last program's largest offset at the extremes: (grid - 1) * B + (B - 1) < 2^31
#eval ((2 ^ 31 - 1 + 2 ^ 20 - 1) / 2 ^ 20 - 1) * 2 ^ 20 + (2 ^ 20 - 1) < 2 ^ 31
#eval ((2 ^ 31 - 1 + 1024 - 1) / 1024 - 1) * 1024 + 1023 < 2 ^ 31
