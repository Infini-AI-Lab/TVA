import AttnBwd.Program

/-!
# `attn_bwd_f16`: tutorial 06, `_attn_bwd` on `float16`

Source: `source/06-fused-attention.py`. The kernel is `_attn_bwd` (lines 366–500) with its helpers
`_attn_bwd_dkdv` (lines 266–315) and `_attn_bwd_dq` (lines 319–363); it reads the module-level
names `triton` and `tl` (lines 20–21). `_attention.backward` (lines 582–623) launches it on the
grid `(N_CTX // 128, 1, BATCH * N_HEAD)`.

The claim: on every covered instance, every launch that runs ends normally; each element of
`dv`, `dk` and `dq` is the `float16` rounding of a `float32` value within an explicit tolerance of
the exact gradient of attention, computed from the values in `q`, `arg_k`, `v`, `do`, `M` and
`delta` (when these are finite and the head's sizes fit, `HeadV.Ok`); every byte outside the three
outputs is unchanged.

* `AttnBwd.Model`: the model, the conditions, the launch.
* `AttnBwd.Spec`: the specification and the assumptions.
* `AttnBwd.Lane`, `AttnBwd.Acc`, `AttnBwd.Basic`, `AttnBwd.Step`: one element of `pT`/`dsT`, one
  accumulation step, the invariants of the loops.
* `AttnBwd.DkDv`, `AttnBwd.Dq`: the helpers' loops (`safe_dkdv`, `safe_dq`).
* `AttnBwd.Out`, `AttnBwd.Program`: the stored values; one program is safe (`safe_program`).
* Here: `main` by `TVA.meets_independent`, `consistent` and `satisfiable`.
-/

namespace Kernels.AttnBwd

open TVA

/-! ## The grid -/

theorem launch_size (d : Inst) : (launch d).size = d.N_CTX / 128 * 1 * G d := rfl

theorem ctx_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = i % (d.N_CTX / 128) := rfl

theorem ctx_pid2 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 2 = i / (d.N_CTX / 128) := by
  show i / (d.N_CTX / 128 * 1) = _
  rw [Nat.mul_one]

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hg := hd.g_lt
  have hN := hd.n_le
  have hHD := hd_le hd
  have hpow : ∀ k : Nat, k = 1 ∨ k = 16 ∨ k = 32 ∨ k = 128 ∨ k = d.HEAD_DIM → ∃ e, k = 2 ^ e := by
    rintro k (rfl | rfl | rfl | rfl | rfl)
    · exact ⟨0, rfl⟩
    · exact ⟨4, rfl⟩
    · exact ⟨5, rfl⟩
    · exact ⟨7, rfl⟩
    · rcases hd.hd with h | h | h | h | h <;> rw [h]
      exacts [⟨4, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨7, rfl⟩, ⟨8, rfl⟩]
  refine ⟨show d.N_CTX / 128 < 2 ^ 31 by omega, show 1 < 2 ^ 16 by norm_num, show G d < 2 ^ 16 from hg,
    ?_, rfl, rfl, ?_⟩
  · rw [launch_size]
    have : d.N_CTX / 128 * 1 * G d ≤ 128 * G d := by
      rw [Nat.mul_one]; exact Nat.mul_le_mul_right _ (by omega)
    omega
  · intro s hs
    simp only [launch, blocks, List.mem_cons, List.mem_nil_iff, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl <;>
    exact ⟨fun k hk => hpow k (by simp only [List.mem_cons, List.mem_nil_iff, or_false] at hk; omega),
      by simp only [List.prod_cons, List.prod_nil]; omega⟩

/-! ## The programs together -/

/-- Program `(p, b)` writes only bytes of the outputs. -/
theorem W_sub {d : Inst} {b p : Nat} (hb : b < G d) (hp : 128 * p + 128 ≤ d.N_CTX) :
    W d b p ⊆ outBytes d := by
  have hT := tileSet_sub hb hp
  unfold W outBytes
  exact Set.union_subset (Set.union_subset ((d.DV.elems_mono hT).trans Set.subset_union_right)
    ((d.DK.elems_mono hT).trans (Set.subset_union_right.trans Set.subset_union_left)))
    ((d.DQ.elems_mono hT).trans (Set.subset_union_left.trans Set.subset_union_left))

/-- Different programs have different tiles. -/
theorem tile_disj {d : Inst} {b p b' p' : Nat} (hp : 128 * p + 128 ≤ d.N_CTX)
    (hp' : 128 * p' + 128 ≤ d.N_CTX) (hne : ¬ (b = b' ∧ p = p')) :
    Disjoint (tileSet d b p) (tileSet d b' p') := by
  rw [Set.disjoint_left]
  rintro _ ⟨ix, -, rfl⟩ ⟨ix', -, h⟩
  obtain ⟨h1, h2, -⟩ := tileG_inj hp' hp ix' ix h
  exact hne ⟨h1.symm, h2.symm⟩

/-- Different programs write different bytes. -/
theorem W_disj {d : Inst} (hd : d.ok) {b p b' p' : Nat} (hb : b < G d) (hb' : b' < G d)
    (hp : 128 * p + 128 ≤ d.N_CTX) (hp' : 128 * p' + 128 ≤ d.N_CTX) (hne : ¬ (b = b' ∧ p = p')) :
    Disjoint (W d b p) (W d b' p') := by
  have hT := tileSet_sub hb hp
  have hT' := tileSet_sub hb' hp'
  have ht := tile_disj hp hp' hne
  unfold W
  refine Disjoint.union_left (Disjoint.union_left ?_ ?_) ?_ <;>
    refine Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_
  · exact d.DV.elems_disjoint ht
  · exact hd.dk_dv.symm.mono (d.DV.elems_mono hT) (d.DK.elems_mono hT')
  · exact hd.dq_dv.symm.mono (d.DV.elems_mono hT) (d.DQ.elems_mono hT')
  · exact hd.dk_dv.mono (d.DK.elems_mono hT) (d.DV.elems_mono hT')
  · exact d.DK.elems_disjoint ht
  · exact hd.dq_dk.symm.mono (d.DK.elems_mono hT) (d.DQ.elems_mono hT')
  · exact hd.dq_dv.mono (d.DQ.elems_mono hT) (d.DV.elems_mono hT')
  · exact hd.dq_dk.mono (d.DQ.elems_mono hT) (d.DK.elems_mono hT')
  · exact d.DQ.elems_disjoint ht

/-- `OutSpec` depends on the final memory only through the element's bytes. -/
theorem outSpec_congr {d : Inst} {init m fin : Mem} {B : Buf .f16} {b r j : Nat} {E T : HeadV → ℝ}
    (h : ∀ x ∈ Mem.span .f16 (B.ptr ↑(eIdx d b r j)).addr, fin x = m x)
    (hs : OutSpec d init m B b r j E T) : OutSpec d init fin B b r j E T := by
  obtain ⟨c, hc, hrest⟩ := hs
  exact ⟨c, (Mem.read_congr h).trans hc, hrest⟩

/-- Element `(b, r, j)` is in the tile of program `(r / 128, b)`. -/
theorem mem_tile {d : Inst} {b r j : Nat} (hj : j < d.HEAD_DIM) :
    eIdx d b r j ∈ tileSet d b (r / 128) :=
  ⟨(⟨r % 128, Nat.mod_lt _ (by norm_num)⟩, ⟨j, hj⟩, ()), rfl, by
    rw [tileG_eq]; simp only [Nat.div_add_mod]⟩

/-! ## The claim -/

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ _ => inBytes d) (fun d _ i => W d (i / (d.N_CTX / 128)) (i % (d.N_CTX / 128)))
    (fun d init i m => Qp d init (i / (d.N_CTX / 128)) (i % (d.N_CTX / 128)) m) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    have hi : (i : Nat) < d.N_CTX / 128 * 1 * G d := lt_of_lt_of_eq i.isLt (launch_size d)
    have h0 : 0 < d.N_CTX / 128 := by
      rcases Nat.eq_zero_or_pos (d.N_CTX / 128) with h | h
      · rw [h] at hi; simp at hi
      · exact h
    rw [Nat.mul_one] at hi
    exact safe_program (E := (launch d).env P) hA.dot hA.dot_total hA.exp2 hA.exp2_total hd hpre
      ((Nat.div_lt_iff_lt_mul h0).mpr (by rw [Nat.mul_comm]; exact hi)) (Nat.mod_lt _ h0) _
      (ctx_pid0 d i) (ctx_pid2 d i)
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hi : (i : Nat) < d.N_CTX / 128 * 1 * G d := lt_of_lt_of_eq i.isLt (launch_size d)
    have hj : (j : Nat) < d.N_CTX / 128 * 1 * G d := lt_of_lt_of_eq j.isLt (launch_size d)
    have h0 : 0 < d.N_CTX / 128 := by
      rcases Nat.eq_zero_or_pos (d.N_CTX / 128) with h | h
      · rw [h] at hi; simp at hi
      · exact h
    rw [Nat.mul_one] at hi hj
    have hn := hd.n_mod
    have hbi : (i : Nat) / (d.N_CTX / 128) < G d :=
      (Nat.div_lt_iff_lt_mul h0).mpr (by rw [Nat.mul_comm]; exact hi)
    have hbj : (j : Nat) / (d.N_CTX / 128) < G d :=
      (Nat.div_lt_iff_lt_mul h0).mpr (by rw [Nat.mul_comm]; exact hj)
    have hpi := Nat.mod_lt (i : Nat) h0
    have hpj := Nat.mod_lt (j : Nat) h0
    have hpi' : 128 * ((i : Nat) % (d.N_CTX / 128)) + 128 ≤ d.N_CTX := by omega
    have hpj' : 128 * ((j : Nat) % (d.N_CTX / 128)) + 128 ≤ d.N_CTX := by omega
    refine Disjoint.union_right ((hd.out_in.symm.mono_right (W_sub hbi hpi')).symm) ?_
    refine W_disj hd hbi hbj hpi' hpj' fun ⟨h1, h2⟩ => hij (Fin.ext ?_)
    rw [← Nat.div_add_mod (i : Nat) (d.N_CTX / 128), ← Nat.div_add_mod (j : Nat) (d.N_CTX / 128), h1, h2]
  · -- no program reads what it writes
    intro d hd init _ i
    have hi : (i : Nat) < d.N_CTX / 128 * 1 * G d := lt_of_lt_of_eq i.isLt (launch_size d)
    have h0 : 0 < d.N_CTX / 128 := by
      rcases Nat.eq_zero_or_pos (d.N_CTX / 128) with h | h
      · rw [h] at hi; simp at hi
      · exact h
    rw [Nat.mul_one] at hi
    have hn := hd.n_mod
    have hbi : (i : Nat) / (d.N_CTX / 128) < G d :=
      (Nat.div_lt_iff_lt_mul h0).mpr (by rw [Nat.mul_comm]; exact hi)
    have hpi := Nat.mod_lt (i : Nat) h0
    exact hd.out_in.symm.mono_right (W_sub hbi (by omega))
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    have hn := hd.n_mod
    refine ⟨fun b r j hb hr hj => ?_, fun z hz => hfr z fun i hzi => ?_⟩
    · have h0 : 0 < d.N_CTX / 128 := by omega
      have hp : r / 128 < d.N_CTX / 128 := by omega
      have hiL : b * (d.N_CTX / 128) + r / 128 < (launch d).size := by
        rw [launch_size, Nat.mul_one]
        have : b * (d.N_CTX / 128) + d.N_CTX / 128 ≤ G d * (d.N_CTX / 128) := by
          rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hb
        rw [Nat.mul_comm (d.N_CTX / 128)]; omega
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨_, hiL⟩
      have hdiv : (b * (d.N_CTX / 128) + r / 128) / (d.N_CTX / 128) = b := by
        rw [Nat.add_comm, Nat.add_mul_div_right _ _ h0, Nat.div_eq_of_lt hp, zero_add]
      have hmod : (b * (d.N_CTX / 128) + r / 128) % (d.N_CTX / 128) = r / 128 := by
        rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hp]
      simp only [hdiv, hmod] at hQ hagree
      have hrow : 128 * (r / 128) + r % 128 = r := Nat.div_add_mod r 128
      have hQ' := hQ ⟨r % 128, Nat.mod_lt _ (by norm_num)⟩ ⟨j, hj⟩
      simp only [hrow] at hQ'
      have hmem := mem_tile (d := d) (b := b) (r := r) hj
      obtain ⟨h1, h2, h3⟩ := hQ'
      refine ⟨outSpec_congr (fun x hx => hagree x (Or.inr ?_)) h1,
        outSpec_congr (fun x hx => hagree x (Or.inr ?_)) h2,
        outSpec_congr (fun x hx => hagree x (Or.inr ?_)) h3⟩
      · exact Or.inl (Or.inl (d.DV.span_sub_elems hmem hx))
      · exact Or.inl (Or.inr (d.DK.span_sub_elems hmem hx))
      · exact Or.inr (d.DQ.span_sub_elems hmem hx)
    · have hi : (i : Nat) < d.N_CTX / 128 * 1 * G d := lt_of_lt_of_eq i.isLt (launch_size d)
      have h0 : 0 < d.N_CTX / 128 := by
        rcases Nat.eq_zero_or_pos (d.N_CTX / 128) with h | h
        · rw [h] at hi; simp at hi
        · exact h
      rw [Nat.mul_one] at hi
      have hbi : (i : Nat) / (d.N_CTX / 128) < G d :=
        (Nat.div_lt_iff_lt_mul h0).mpr (by rw [Nat.mul_comm]; exact hi)
      have hpi := Nat.mod_lt (i : Nat) h0
      exact hz (W_sub hbi (by omega) hzi)

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotF16, Platform.ref.dotTotal _,
    Platform.sm90ref.exp2F32, Platform.ref.mathTotal⟩⟩

/-! ## A covered instance -/

/-- An instance of the source's test (lines 631–695): `Z = 1`, `H = 2`, `N_CTX = 128`,
`HEAD_DIM = 64`, not causal, `sm_scale = 0.5`, the nine tensors contiguous at the aligned
addresses `2^21, 2·2^21, …, 9·2^21` (each 4-D tensor of `2·128·64` elements, `M` and `delta` of
`2·128`). -/
def testInst : Inst where
  Z := 1
  H := 2
  N_CTX := 128
  HEAD_DIM := 64
  CAUSAL := false
  sm_scale := FloatV.ofPat 0x3FE0000000000000
  Q := ⟨2097152, 16384⟩
  K := ⟨4194304, 16384⟩
  V := ⟨6291456, 16384⟩
  DO := ⟨8388608, 16384⟩
  M := ⟨10485760, 256⟩
  D := ⟨12582912, 256⟩
  DQ := ⟨14680064, 16384⟩
  DK := ⟨16777216, 16384⟩
  DV := ⟨18874368, 16384⟩

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem test_sm : (smF testInst).isFinite = true ∧ |(smF testInst).toRat| ≤ 2 ^ 16 := by
  decide +kernel

theorem test_bytes {dt : DType} {B : Buf dt} {S : Set Nat} {x : Int} (hx : x ∈ B.elems S)
    (hS : ∀ j ∈ S, j < B.len) : B.base ≤ x ∧ x < B.base + B.len * dt.bytes :=
  B.elems_sub_bytes hS hx

theorem test_t (j : Nat) (hj : j ∈ tSet testInst) : j < 16384 := by
  simp only [tSet, G, testInst, Set.mem_ofPred_eq] at hj; omega

theorem test_r (j : Nat) (hj : j ∈ rSet testInst) : j < 256 := by
  simp only [rSet, G, testInst, Set.mem_ofPred_eq] at hj; omega

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hout : ∀ x ∈ outBytes testInst, 14680064 ≤ x := by
    intro x hx
    simp only [outBytes, Set.mem_union] at hx
    rcases hx with (h | h) | h
    · have := test_bytes h test_t; simp only [testInst] at this; omega
    · have := test_bytes h test_t; simp only [testInst] at this; omega
    · have := test_bytes h test_t; simp only [testInst] at this; omega
  have hin : ∀ x ∈ inBytes testInst, x < 14680064 := by
    intro x hx
    simp only [inBytes, Set.mem_union] at hx
    rcases hx with ((((h | h) | h) | h) | h) | h
    · have := test_bytes h test_t; simp only [testInst, DType.bytes] at this; omega
    · have := test_bytes h test_t; simp only [testInst, DType.bytes] at this; omega
    · have := test_bytes h test_t; simp only [testInst, DType.bytes] at this; omega
    · have := test_bytes h test_t; simp only [testInst, DType.bytes] at this; omega
    · have := test_bytes h test_r; simp only [testInst, DType.bytes] at this; omega
    · have := test_bytes h test_r; simp only [testInst, DType.bytes] at this; omega
  refine ⟨testInst, testMem, ⟨Or.inr (Or.inr (Or.inl rfl)), by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst, G], by norm_num [testInst, G], test_sm, by norm_num [testInst, G],
    by norm_num [testInst, G], by norm_num [testInst, G], by norm_num [testInst, G],
    by norm_num [testInst, G], by norm_num [testInst, G], by norm_num [testInst, G],
    by norm_num [testInst, G], by norm_num [testInst, G],
    Set.disjoint_left.mpr fun x hx hy => absurd (hin x hy) (not_lt.mpr (hout x hx)), ?_, ?_, ?_⟩,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine Set.disjoint_left.mpr fun x hx hy => ?_
    have h1 := test_bytes hx test_t
    have h2 := test_bytes hy test_t
    simp only [testInst, DType.bytes] at h1 h2; omega
  · refine Set.disjoint_left.mpr fun x hx hy => ?_
    have h1 := test_bytes hx test_t
    have h2 := test_bytes hy test_t
    simp only [testInst, DType.bytes] at h1 h2; omega
  · refine Set.disjoint_left.mpr fun x hx hy => ?_
    have h1 := test_bytes hx test_t
    have h2 := test_bytes hy test_t
    simp only [testInst, DType.bytes] at h1 h2; omega
  all_goals
    refine ⟨by norm_num [testInst], by norm_num [testInst, DType.bytes], by norm_num [testInst, DType.bytes],
      fun _ _ => by simp [testMem]⟩

end Kernels.AttnBwd
