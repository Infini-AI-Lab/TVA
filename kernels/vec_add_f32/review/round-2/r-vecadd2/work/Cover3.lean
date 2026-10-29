import VecAdd

/-! r-vecadd2, round 2, at gate 990243026f1289fb: `Inst.ok` covers every launch the script makes.

The host's `add(x, y)` launches with `BLOCK_SIZE = 1024`, `n = output.numel()`, and three
`float32` tensors of `n` elements: `x`, `y` and the fresh `output = torch.empty_like(x)`, whose
bytes meet neither input. The script's sizes are `98432` and `2^12 … 2^27`. Below: for every
`n < 2^31` and every three buffers of at least `n` elements whose byte ranges `output` shares with
neither input, the instance with `BLOCK_SIZE = 1024` is covered; also `add(x, x)` (`y` the same
tensor as `x`). And the script's sizes are below `2^31`. -/

open TVA

theorem host_covered (n : Nat) (hn : n < 2 ^ 31) (x y o : Buf .f32)
    (hx : n ≤ x.len) (hy : n ≤ y.len) (ho : n ≤ o.len)
    (hox : Disjoint o.bytes x.bytes) (hoy : Disjoint o.bytes y.bytes) :
    Kernels.VecAdd.Inst.ok ⟨n, 1024, x, y, o⟩ := by
  refine ⟨⟨10, by norm_num⟩, by norm_num, hn, hx, hy, ho, ?_, ?_⟩
  · exact hox.mono (Buf.elems_sub_bytes o fun j hj => by
      simp only [Set.mem_ofPred_eq] at hj; omega)
      (Buf.elems_sub_bytes x fun j hj => by simp only [Set.mem_ofPred_eq] at hj; omega)
  · exact hoy.mono (Buf.elems_sub_bytes o fun j hj => by
      simp only [Set.mem_ofPred_eq] at hj; omega)
      (Buf.elems_sub_bytes y fun j hj => by simp only [Set.mem_ofPred_eq] at hj; omega)

/-- `add(x, x)`: the same tensor for both inputs. -/
example (n : Nat) (hn : n < 2 ^ 31) (x o : Buf .f32) (hx : n ≤ x.len) (ho : n ≤ o.len)
    (hox : Disjoint o.bytes x.bytes) : Kernels.VecAdd.Inst.ok ⟨n, 1024, x, x, o⟩ :=
  host_covered n hn x x o hx hx ho hox hox

-- The script's sizes are all below `2^31`.
#eval (decide (98432 < 2 ^ 31), (List.range 16).all fun i => decide (2 ^ (12 + i) < 2 ^ 31))
