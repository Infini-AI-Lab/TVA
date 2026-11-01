import MatmulE5m2.Program
import MatmulE5m2.Reject

/-!
# `matmul_e5m2`: tutorial 03, `matmul_kernel` on `float8e5`

Source: `source/03-matrix-multiplication.py`. The kernel is lines 228–316 and it reads the
module-level names `triton` and `tl` (lines 154–155). Its host function `matmul` (lines 330–350)
launches it, in the script's `float8e5` run, on `float8e5` matrices `a` (`M × K`, contiguous) and
`b` (`K × N`, the transpose of a contiguous matrix) into a new `float16` matrix `c` (`M × N`): the
test (`512 × 512 × 512`, lines 372–388) and the benchmark's `fp8` line (square sizes
`256, 384, …, 4096`, lines 400–440).

The claim: on every covered instance, every launch that runs ends normally; each element `(i, j)`
of `c` is the `float16` rounding of a `float32` accumulation within an explicit tolerance (the
library's fp8 `tl.dot` fact applied chunk by chunk, from the exact partial sums) of the exact dot
product of row `i` of `a` and column `j` of `b` (when these are finite); every byte outside the
matrix `c` is unchanged.

* `MatmulE5m2.Defs`: the model (`matmul_kernel`, `model`, `launch`), the conditions (`Inst`,
  `Inst.ok`, `Pre`), the specification (`errAt`, `tol`, `OutSpec`, `Post`, `Spec`), the assumptions.
* `MatmulE5m2.Arith`: the grouped program order (lines 253–260) is a bijection onto the tiles.
* `MatmulE5m2.Basic`: `int32` arithmetic, `other=0.0`, one accumulation step, the last rounding.
* `MatmulE5m2.Program`: one program is safe (`safe_program`).
* `MatmulE5m2.Reject`: what the tolerance rejects at every covered size (`zero_rejected`,
  `drop_rejected`, `tol_spike`).
* Here: `main` by `TVA.meets_independent`, `consistent` and `satisfiable`.
-/

namespace Kernels.MatmulE5m2

open TVA

/-! ## The programs together -/

/-- Program `p` writes only elements of the matrix `c`. -/
theorem W_sub (d : Inst) (p : Nat) : W d p ⊆ d.c.elems (cIdx d) := by
  refine d.c.elems_mono ?_
  rintro _ ⟨⟨i, j, ⟨⟩⟩, he, rfl⟩
  simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he
  exact ⟨pm d p * 128 + i, pn d p * 256 + j, he.1, he.2, rfl⟩

/-- Two programs of the grid write different elements of `c`: their tiles differ (`pid_inj`). -/
theorem W_disj {d : Inst} (hd : d.ok) {p q : Nat} (hp : p < grid d) (hq : q < grid d) (hpq : p ≠ q) :
    Disjoint (W d p) (W d q) := by
  refine d.c.elems_disjoint (Set.disjoint_left.mpr ?_)
  rintro _ ⟨⟨i, j, ⟨⟩⟩, he, rfl⟩ ⟨⟨i', j', ⟨⟩⟩, he', hg⟩
  simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he he'
  dsimp only [cG] at hg
  obtain ⟨h1, h2⟩ := c_inj hd.cn.1 hd.c_rows he'.2 he.2 hg
  have hm : pm d q = pm d p := by have := i.2; have := i'.2; omega
  have hn : pn d q = pn d p := by have := j.2; have := j'.2; omega
  exact hpq (pid_inj hp hq hm.symm hn.symm)

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hg := grid_lt hd
  unfold grid tilesM tilesN at hg
  refine Launch.valid_grid1 (by
    rw [show d.M + 128 - 1 = d.M + 127 by omega, show d.N + 256 - 1 = d.N + 255 by omega]; exact hg) ?_
  have hpow : ∀ k : Nat, k = 1 ∨ k = 64 ∨ k = 128 ∨ k = 256 → ∃ e, k = 2 ^ e := by
    rintro k (rfl | rfl | rfl | rfl)
    exacts [⟨0, rfl⟩, ⟨6, rfl⟩, ⟨7, rfl⟩, ⟨8, rfl⟩]
  intro s hs
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨fun k hk => hpow k (by simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk; omega),
      by simp⟩

/-! ## The claim -/

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ _ => R d) (fun d _ i => W d i) (fun d init i m => Q d init i m) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    exact safe_program hd hpre hA.dot hA.dot_total hA.cvt hA.cvt_total (lt_of_lt_of_eq i.isLt (launch_size d))
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
    have hj : (j : Nat) < grid d := lt_of_lt_of_eq j.isLt (launch_size d)
    refine Disjoint.union_right ?_ (W_disj hd hi hj hij')
    exact (Disjoint.union_right hd.c_a hd.c_b).mono_left (W_sub d i)
  · -- no program reads what it writes
    intro d hd init _ i
    exact ((Disjoint.union_right hd.c_a hd.c_b).mono_left (W_sub d i)).symm
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun i j hi hj => ?_, fun z hz => hfr z fun p hzp => hz (W_sub d p hzp)⟩
    have hmt : i / 128 < tilesM d := by unfold tilesM; omega
    have hnt : j / 256 < tilesN d := by unfold tilesN; omega
    obtain ⟨p, hp, hpm, hpn⟩ := pid_surj hmt hnt
    have hpL : p < (launch d).size := by rw [launch_size]; exact hp
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨p, hpL⟩
    have hrow : pm d p * 128 + i % 128 = i := by
      show pmOf (tilesM d) (tilesN d) p * 128 + i % 128 = i
      rw [hpm]; omega
    have hcol : pn d p * 256 + j % 256 = j := by
      show pnOf (tilesM d) (tilesN d) p * 256 + j % 256 = j
      rw [hpn]; omega
    have hQ' := hQ ⟨i % 128, Nat.mod_lt _ (by norm_num)⟩ ⟨j % 256, Nat.mod_lt _ (by norm_num)⟩
      (by simp only; omega) (by simp only; omega)
    simp only [hrow, hcol] at hQ'
    obtain ⟨c, hc, hrest⟩ := hQ'
    refine ⟨c, ?_, hrest⟩
    rw [← hc]
    refine Mem.read_congr fun x hx => hagree x (Or.inr (d.c.span_sub_elems ?_ hx))
    refine ⟨(⟨i % 128, Nat.mod_lt _ (by norm_num)⟩, ⟨j % 256, Nat.mod_lt _ (by norm_num)⟩, ()), ?_, ?_⟩
    · simp only [cEn, hrow, hcol, hi, hj, decide_true, Bool.and_self]
    · simp only [cG, hrow, hcol]

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotE5m2ChunkLoss, Platform.ref.dotTotal _,
    Platform.sm90ref.cvtF32E5m2, Platform.ref.cvtTotal _ _ _⟩⟩

/-! ## A covered instance -/

/-- The source's `float8e5` test instance (lines 374–381): `M = N = K = 512`, row-major `a` and `c`
(strides `512` and `1`) and the transposed `b` (strides `1` and `512`), each of `512 · 512`
elements, at the 16-byte-aligned addresses `2^21`, `2·2^21` and `3·2^21`. -/
def testInst : Inst where
  M := 512
  N := 512
  K := 512
  stride_am := 512
  stride_ak := 1
  stride_bk := 1
  stride_bn := 512
  stride_cm := 512
  stride_cn := 1
  a := ⟨2097152, 262144⟩
  b := ⟨4194304, 262144⟩
  c := ⟨6291456, 262144⟩

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨testInst, testMem, ⟨by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst],
    fun i k hi hk => by simp only [testInst] at hi hk ⊢; omega,
    fun k j hk hj => by simp only [testInst] at hk hj ⊢; omega,
    fun i j hi hj => by simp only [testInst] at hi hj ⊢; omega, ?_, ?_⟩, ?_, ?_, ?_⟩
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro x hx hy
      simp only [Buf.bytes, testInst, Set.mem_ofPred_eq, DType.bytes] at hx hy
      omega
    · rintro _ ⟨i, j, hi, hj, rfl⟩
      simp only [testInst] at hi hj ⊢; omega
    · rintro _ ⟨i, k, hi, hk, rfl⟩
      simp only [testInst] at hi hk ⊢; omega
  · refine Buf.elems_disjoint_of _ _ ?_ ?_ ?_
    · rw [Set.disjoint_left]
      intro x hx hy
      simp only [Buf.bytes, testInst, Set.mem_ofPred_eq, DType.bytes] at hx hy
      omega
    · rintro _ ⟨i, j, hi, hj, rfl⟩
      simp only [testInst] at hi hj ⊢; omega
    · rintro _ ⟨k, j, hk, hj, rfl⟩
      simp only [testInst] at hk hj ⊢; omega
  all_goals
    refine ⟨by norm_num [testInst], by norm_num [testInst, DType.bytes], by norm_num [testInst, DType.bytes],
      fun _ _ => by simp [testMem]⟩

/-! ## The host's use -/

/-- Row `r < n` and column `s < n` of an `n × n` matrix stored with strides `n` and `1` give an
element below `n·n`. -/
theorem sq_lt {n r s : Nat} (hr : r < n) (hs : s < n) : r * n + s < n * n := by
  have h := Nat.mul_le_mul_right n (show r + 1 ≤ n by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- **The script's `float8e5` run is covered.** Every launch `matmul(a, b)` makes there (the test,
`n = 512`, lines 374–381; the benchmark's `fp8` line, `n = 256, 384, …, 4096`, lines 423–435) is
an `n × n × n` instance with `a` contiguous (strides `(n, 1)`), `b` the transposed view (strides
`(1, n)`) and `c` from `torch.empty` (strides `(n, 1)`), each buffer holding (at least) the `n·n`
elements of its matrix, and `c` a fresh buffer that shares no byte with `a` or `b`. Each such
instance with `1 ≤ n ≤ 4096` meets `Inst.ok`. -/
theorem fp8_run_ok {n : Nat} (h1 : 1 ≤ n) (hn : n ≤ 4096) (a b : Buf .f8e5) (c : Buf .f16)
    (ha : n * n ≤ a.len) (hb : n * n ≤ b.len) (hc : n * n ≤ c.len)
    (hca : Disjoint c.bytes a.bytes) (hcb : Disjoint c.bytes b.bytes) :
    Inst.ok ⟨n, n, n, n, 1, 1, n, n, 1, a, b, c⟩ := by
  have hsq : n * n ≤ 4096 * 4096 := Nat.mul_le_mul hn hn
  have hcl : ∀ i j, i < n → j < n → i * n + j * 1 < c.len := fun i j hi hj => by
    have := sq_lt hi hj; omega
  refine ⟨?_, ?_, ?_, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ?_, ?_, ?_, ?_,
    fun i k hi hk => ?_, fun k j hk hj => ?_, hcl, ?_, ?_⟩
  all_goals dsimp only
  any_goals omega
  · have := sq_lt (n := n) hi hk; omega
  · have := sq_lt (n := n) hj hk; omega
  · refine Buf.elems_disjoint_of _ _ hca ?_ ?_
    · rintro _ ⟨i, j, hi, hj, rfl⟩
      exact hcl i j hi hj
    · rintro _ ⟨i, k, hi, hk, rfl⟩
      have := sq_lt (n := n) hi hk
      show i * n + k * 1 < a.len
      omega
  · refine Buf.elems_disjoint_of _ _ hcb ?_ ?_
    · rintro _ ⟨i, j, hi, hj, rfl⟩
      exact hcl i j hi hj
    · rintro _ ⟨k, j, hk, hj, rfl⟩
      have := sq_lt (n := n) hj hk
      show k * 1 + j * n < b.len
      omega

end Kernels.MatmulE5m2
