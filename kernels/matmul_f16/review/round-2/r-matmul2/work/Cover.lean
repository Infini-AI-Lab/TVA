import Matmul

/-! Reviewer r-matmul2 (rounds 1 and 2): the host's own launches meet `Inst.ok`. The benchmark's largest launch
(`M = N = K = 4096`) and a partial-tile one (`M = N = K = 384`, `N` not a multiple of `256`), with
row-major `a`, `b`, `c` as `matmul` passes them (strides `(K, 1)`, `(N, 1)`, `(N, 1)`), in buffers
of exactly `M·K`, `K·N` and `M·N` elements at 16-byte-aligned addresses that do not overlap. -/

namespace Kernels.Matmul

open TVA

/-- A square launch of size `n` as `matmul` makes it. -/
def sq (n : Nat) : Inst where
  M := n
  N := n
  K := n
  stride_am := n
  stride_ak := 1
  stride_bk := n
  stride_bn := 1
  stride_cm := n
  stride_cn := 1
  a := ⟨2 ^ 30, n * n⟩
  b := ⟨2 * 2 ^ 30, n * n⟩
  c := ⟨3 * 2 ^ 30, n * n⟩

theorem sq_ok (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 4096) : (sq n).ok := by
  have hn2 : n * n ≤ 4096 * 4096 := Nat.mul_le_mul h2 h2
  refine ⟨by simp only [sq]; omega, by simp only [sq]; omega, by simp only [sq]; omega,
    by simp only [sq]; omega, by simp only [sq]; omega, by simp only [sq]; omega,
    by simp only [sq]; omega, by simp only [sq]; omega, by simp only [sq]; omega,
    by simp only [sq]; omega, by simp only [sq]; omega, by simp only [sq]; omega,
    by simp only [sq]; omega,
    fun i k hi hk => ?_, fun k j hk hj => ?_, fun i j hi hj => ?_, ?_, ?_⟩
  · simp only [sq] at hi hk ⊢; nlinarith
  · simp only [sq] at hk hj ⊢; nlinarith
  · simp only [sq] at hi hj ⊢; nlinarith
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro x hx hy
      simp only [Buf.bytes, sq, Set.mem_ofPred_eq, DType.bytes] at hx hy
      omega
    · rintro _ ⟨i, j, hi, hj, rfl⟩
      simp only [sq] at hi hj ⊢; nlinarith
    · rintro _ ⟨i, k, hi, hk, rfl⟩
      simp only [sq] at hi hk ⊢; nlinarith
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro x hx hy
      simp only [Buf.bytes, sq, Set.mem_ofPred_eq, DType.bytes] at hx hy
      omega
    · rintro _ ⟨i, j, hi, hj, rfl⟩
      simp only [sq] at hi hj ⊢; nlinarith
    · rintro _ ⟨k, j, hk, hj, rfl⟩
      simp only [sq] at hk hj ⊢; nlinarith

/-- The benchmark's largest launch and a partial-tile one are covered. -/
example : (sq 4096).ok := sq_ok 4096 (by norm_num) le_rfl
example : (sq 384).ok := sq_ok 384 (by norm_num) (by norm_num)
example : (sq 256).ok := sq_ok 256 (by norm_num) (by norm_num)

/-- The tolerance at the test's and the benchmark's largest `K`, against `A = Σ|a·b|`: at most
`1.3e-4` and `1.01e-3` times `A` (for `A = 1`). At `K = 2^16` it is `1.624e-2`, computed exactly
with Python's fractions (norm_num does not expand the 1024th power). -/
example : tol 512 1 ≤ 13 / 100000 := by
  unfold tol steps DotLoop.bound Rules.Num.relDot16 Rules.Num.absDot16; norm_num
example : tol 4096 1 ≤ 101 / 100000 := by
  unfold tol steps DotLoop.bound Rules.Num.relDot16 Rules.Num.absDot16; norm_num

/-- Round 2: at the largest covered `K = 2^13` the tolerance is between `2.01e-3` and `2.03e-3`
times `A` (plus the absolute part, below `10^-34`). -/
example : tol (2 ^ 13) 1 ≤ 203 / 100000 := by
  unfold tol steps DotLoop.bound Rules.Num.relDot16 Rules.Num.absDot16; norm_num
example : 201 / 100000 ≤ tol (2 ^ 13) 1 := by
  unfold tol steps DotLoop.bound Rules.Num.relDot16 Rules.Num.absDot16; norm_num
example : tol (2 ^ 13) 0 < 1 / 10 ^ 34 := by
  unfold tol steps DotLoop.bound Rules.Num.relDot16 Rules.Num.absDot16; norm_num

/-- Round 2: the write-up's claim that `Post` still rejects a `bfloat16`-rounded result at
`K = 2^13`. With one nonzero product `S = 1 + 2^-8 - 2^-10` (a `float16` value, `A = S`), a kernel
that rounds to `bfloat16` stores `1`, which is farther from `S` than `OutSpec`'s finite-case bound
allows. -/
example : let S : ℚ := 1 + 1 / 2 ^ 8 - 1 / 2 ^ 10
    tol (2 ^ 13) S + (|S| + tol (2 ^ 13) S) * Math.Float.Format.binary16.u +
      Math.Float.Format.binary16.eta < |(1 : ℚ) - S| := by
  intro S
  rw [Math.Float.Format.binary16_u, Math.Float.Format.binary16_eta]
  simp only [S]
  unfold tol steps DotLoop.bound Rules.Num.relDot16 Rules.Num.absDot16
  norm_num [abs_of_pos, abs_of_neg]

end Kernels.Matmul
