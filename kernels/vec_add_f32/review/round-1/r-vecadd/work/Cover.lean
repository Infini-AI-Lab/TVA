import VecAdd

/-! Reviewer r-vecadd, round 1: the conditions cover the host's instances and are satisfiable.

* `host d n`: the host's launch `add(x, y)` on fresh, disjoint tensors of `n` elements, with
  `BLOCK_SIZE = 1024` (x at `0`, y at `4n`, output at `8n`).
* `Inst.ok` holds at the script's sizes `98432` and `2^27`, at the largest covered size with the
  host's block (`2^31 - 1024`), and in place (`output = x`).
* `Pre` holds with every byte present.
* `Inst.ok` fails at `2^31 - 1023` (the L8 narrowing). -/

open TVA Kernels.VecAdd

def host (n : Nat) : Inst :=
  ⟨n, 1024, ⟨0, n⟩, ⟨4 * n, n⟩, ⟨8 * n, n⟩⟩

def inPlace (n : Nat) : Inst :=
  ⟨n, 1024, ⟨0, n⟩, ⟨4 * n, n⟩, ⟨0, n⟩⟩

theorem disj {a b : Int} {la lb : Nat} (h : a + la * 4 ≤ b) :
    Disjoint (Buf.bytes (⟨b, lb⟩ : Buf .f32)) (Buf.bytes (⟨a, la⟩ : Buf .f32)) := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq, DType.bytes] at hx hy
  push_cast at hx hy
  omega

theorem host_ok {n : Nat} (hn : n + 1024 ≤ 2 ^ 31) : (host n).ok where
  block_pow2 := ⟨10, rfl⟩
  block_le := show 1024 ≤ 2 ^ 20 by decide
  n_fits := hn
  x_len := le_refl _
  y_len := le_refl _
  output_len := le_refl _
  output_x := Or.inl (disj (by omega))
  output_y := Or.inl (disj (by omega))

example : (host 98432).ok := host_ok (by norm_num)
example : (host (2 ^ 27)).ok := host_ok (by norm_num)
example : (host (2 ^ 31 - 1024)).ok := host_ok (by norm_num)

theorem inPlace_ok {n : Nat} (hn : n + 1024 ≤ 2 ^ 31) : (inPlace n).ok where
  block_pow2 := ⟨10, rfl⟩
  block_le := show 1024 ≤ 2 ^ 20 by decide
  n_fits := hn
  x_len := le_refl _
  y_len := le_refl _
  output_len := le_refl _
  output_x := Or.inr rfl
  output_y := Or.inl (by
    rw [Set.disjoint_left]
    intro x hx hy
    simp only [inPlace, Buf.bytes, Set.mem_ofPred_eq, DType.bytes] at hx hy
    push_cast at hx hy
    omega)

example : (inPlace 98432).ok := inPlace_ok (by norm_num)

/-- Every byte present, all zero. -/
def full : Mem := fun _ => some 0

example : Pre (host 98432) full := by
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩⟩ <;>
    first | (intro x _; simp [full]) | simp [host, DType.bytes]

/-- The L8 narrowing: the host's block at `n = 2^31 - 1023` is not covered. -/
example : ¬ (host (2 ^ 31 - 1023)).ok := fun h => by
  have := h.n_fits; simp [host] at this

#eval (2 ^ 31 - 1024 : Nat)
#eval ((2 ^ 31 - 1024 + 1024 - 1) / 1024 : Nat)
