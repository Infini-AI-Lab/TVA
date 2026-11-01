import BlockScaledMxfp8.Program

/-!
# `block_scaled_matmul_mxfp8`: tutorial 10, `block_scaled_matmul_kernel` on `mxfp8`

Source: `source/10-block-scaled-matmul.py`. The kernel is lines 174–240 and it reads the
module-level names `triton` and `tl` (lines 123–124). The host function `block_scaled_matmul`
(lines 243–281), with the config `initialize_block_scaled` (lines 330–449) builds for `"mxfp8"`,
launches it on `float8e4nv` matrices `a` (`M × K`) and `b` (`N × K`) with `e8m0` scales in groups
of `32` along `K`: a grid of `cdiv(M, 128) · cdiv(N, 256)` programs, program `p` computing the
`128 × 256` tile `(p % cdiv(M, 128), p / cdiv(M, 128))` of the `float16` output `c`.

The claim: on every covered instance, every launch that runs ends normally; each element `(i, j)`
of `c` is the `float16` rounding of a `float32` accumulation within an explicit tolerance of the
exact dot product of row `i` of `a` and row `j` of `b`, each element times `2^(s − 127)` for its
scale byte `s` (when these are in the dot's domain); every byte outside `c` is unchanged.

* `BlockScaledMxfp8.Defs`: the model, the conditions, the specification, the assumptions.
* `BlockScaledMxfp8.Basic`: `int32` arithmetic, descriptor lanes, the scales' reshape and permute,
  the dot's operands, the tolerance, one accumulation step, the last rounding.
* `BlockScaledMxfp8.Program`: the schedule, the footprints, one program (`safe_program`).
* Here: `main` by `TVA.meets_independent`, `consistent` and `satisfiable`.
-/

namespace Kernels.BlockScaledMxfp8

open TVA

/-! ## The programs together -/

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (by have := hd.grid_lt; omega) ?_
  have hpow : ∀ k : Nat, k = 1 ∨ k = 2 ∨ k = 4 ∨ k = 32 ∨ k = 128 ∨ k = 256 → ∃ e, k = 2 ^ e := by
    rintro k (rfl | rfl | rfl | rfl | rfl | rfl)
    exacts [⟨0, rfl⟩, ⟨1, rfl⟩, ⟨2, rfl⟩, ⟨5, rfl⟩, ⟨7, rfl⟩, ⟨8, rfl⟩]
  intro s hs
  simp only [List.mem_cons, List.mem_nil_iff, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    exact ⟨fun k hk => hpow k (by simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk; omega),
      by simp⟩

/-- Element `(i, j)` of `c` lies in the tile of program `(j / 256)·cdiv(M, 128) + i / 128`, at
lane `(i % 128, j % 256)`. -/
theorem tile_of {d : Inst} (hd : d.ok) {i j : Nat} (hi : i < d.M) (hj : j < d.N) :
    j / 256 * nM d + i / 128 < nM d * nN d ∧ pm d (j / 256 * nM d + i / 128) = i / 128 ∧
      pn d (j / 256 * nM d + i / 128) = j / 256 := by
  have hi' : i / 128 < nM d := by unfold nM; omega
  have hj' : j / 256 < nN d := by unfold nN; omega
  refine ⟨?_, ?_, ?_⟩
  · calc j / 256 * nM d + i / 128 < j / 256 * nM d + nM d := by omega
      _ = (j / 256 + 1) * nM d := by ring
      _ ≤ nN d * nM d := Nat.mul_le_mul_right _ hj'
      _ = nM d * nN d := Nat.mul_comm _ _
  · unfold pm
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi']
  · unfold pn
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (nM_pos hd), Nat.div_eq_of_lt hi', zero_add]

/-! ## The claim -/

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ _ => R d) (fun d _ i => W d i) (fun d init i m => Q d init i m) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    exact safe_program hA.dot hA.dot_total hd hpre (lt_of_lt_of_eq i.isLt (launch_size d))
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (R_W hd i).symm (W_disj d hij')
  · -- no program reads what it writes
    intro d hd init _ i
    exact R_W hd i
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun i j hi hj => ?_, fun z hz => hfr z fun p hzp => hz (W_sub hd p hzp)⟩
    obtain ⟨hp, hpm, hpn⟩ := tile_of hd hi hj
    have hpL : j / 256 * nM d + i / 128 < (launch d).size := by rw [launch_size]; exact hp
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨_, hpL⟩
    have hjN : pn d (j / 256 * nM d + i / 128) * 256 + j % 256 < d.N := by rw [hpn]; omega
    have hO := hQ (i % 128) (j % 256) (Nat.mod_lt _ (by norm_num)) (Nat.mod_lt _ (by norm_num)) hjN
    rw [hpm, hpn, show i / 128 * 128 + i % 128 = i by omega, show j / 256 * 256 + j % 256 = j by omega] at hO
    obtain ⟨c, hc, hrest⟩ := hO
    refine ⟨c, ?_, hrest⟩
    rw [← hc]
    refine Mem.read_congr fun x hx => hagree x (Or.inr ?_)
    refine d.c.span_sub_elems (S := tileIdx d _) ⟨i % 128, j % 256, Nat.mod_lt _ (by norm_num),
      Nat.mod_lt _ (by norm_num), hjN, ?_⟩ hx
    rw [hpm, hpn, show i / 128 * 128 + i % 128 = i by omega, show j / 256 * 256 + j % 256 = j by omega]

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot,
    ⟨fun d _ => hf d, Platform.sm90ref.dotScaledE4m3, Platform.ref.dotScaledTotal _⟩⟩

/-! ## A covered instance -/

/-- The smallest covered sizes, `M = N = K = 128`: `a`, `b` of `128 · 128` bytes, the scales of
`512` bytes each, `c` of `128 · 128` halves, back to back at 16-byte-aligned addresses. -/
def testInst : Inst where
  M := 128
  N := 128
  K := 128
  a := ⟨0, 16384⟩
  b := ⟨16384, 16384⟩
  a_scale := ⟨32768, 512⟩
  b_scale := ⟨33280, 512⟩
  c := ⟨33792, 16384⟩

/-- A memory in which every byte is present. -/
def testMem : Mem := fun _ => some 0

/-- The first `n` elements of a buffer lie in its bytes. -/
theorem elems_lt_sub {dt : DType} (b : Buf dt) {n : Nat} (hn : n ≤ b.len) :
    b.elems {e | e < n} ⊆ {x | b.base ≤ x ∧ x < b.base + b.len * dt.bytes} :=
  (b.elems_sub_bytes fun _ hj => lt_of_lt_of_le hj hn).trans fun _ hx => hx

theorem test_ok : testInst.ok := by
  refine ⟨by norm_num [testInst], by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], ?_⟩
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  have h1 := elems_lt_sub testInst.c (by norm_num [testInst]) hx
  simp only [testInst, DType.bytes, Set.mem_ofPred_eq] at h1
  rcases hy with ((hy | hy) | hy) | hy
  · have h2 := elems_lt_sub testInst.a (by norm_num [testInst]) hy
    simp only [testInst, DType.bytes, Set.mem_ofPred_eq] at h2
    omega
  · have h2 := elems_lt_sub testInst.b (by norm_num [testInst]) hy
    simp only [testInst, DType.bytes, Set.mem_ofPred_eq] at h2
    omega
  · have h2 := elems_lt_sub testInst.a_scale (by norm_num [testInst]) hy
    simp only [testInst, DType.bytes, Set.mem_ofPred_eq] at h2
    omega
  · have h2 := elems_lt_sub testInst.b_scale (by norm_num [testInst]) hy
    simp only [testInst, DType.bytes, Set.mem_ofPred_eq] at h2
    omega

theorem test_pre : Pre testInst testMem := by
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid testMem := fun b h0 h1 h2 =>
    ⟨h0, h1, h2, fun _ _ => Option.some_ne_none 0⟩
  exact ⟨hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes]),
    hv _ (by norm_num [testInst]) (by norm_num [testInst, DType.bytes]) (by norm_num [testInst, DType.bytes])⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := ⟨testInst, testMem, test_ok, test_pre⟩

end Kernels.BlockScaledMxfp8
