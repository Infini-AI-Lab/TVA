import AttnFwdE5m2.Program

/-!
# `attn_fwd_e5m2`: tutorial 06, `_attn_fwd` on `float8e5`

As given, the kernel reads `v` for the first `⌈Z·H/HEAD_DIM⌉` heads only: for every other head
(`bh·HEAD_DIM ≥ Z·H`) line 98 reads the zero padding of `v`'s descriptor, and the head's output is
`±0`. The claim compares `o` with the attention of the values the kernel uses (`vEff`), so for those
heads it certifies `±0`.

Source: `source/06-fused-attention.py`. The kernel is `_attn_fwd` (lines 176–245) with its helpers
`_attn_fwd_inner` (lines 47–110) and `_maybe_make_tensor_desc` (lines 168–173), in the `float8e5`
run (`FP8_OUTPUT = True`); it reads the module-level names `triton` and `tl` (lines 20–21). The
host (`_attention.forward`, lines 505–580) launches it on `float8e5` tensors `q`, `k` (contiguous
`[Z, H, N_CTX, HEAD_DIM]`) and `v` (the `[Z, H, N_CTX, HEAD_DIM]` view of a contiguous
`[Z, H, HEAD_DIM, N_CTX]` tensor) into the `float8e5` output `o` and the `float32` log-sum-exp `M`:
the test (`test_op`, lines 631–695, its `float8e5` cases) and the benchmark (lines 698–780, its
`fp8` lines).

The claim: on every covered instance, every launch that runs ends normally; every element of `o`
and `M` holds a value, and for a head whose values are covered (`HeadOk`) the output is within two
explicit tolerances (`tolO`, `tolV`) of the exact attention of the values the kernel uses
(`attnRef`, with `vEff`) and `M` of the exact log-sum-exp (base 2); every byte outside `o` and `M`
is unchanged. On the host's data the first tolerance is a few times the fp8 floor, the error the
fp8 rounding of `p` alone allows (README, Specification).

* `AttnFwdE5m2.Model`: the model, the conditions, the specification, the assumptions.
* `AttnFwdE5m2.Real`, `AttnFwdE5m2.Lane`, `AttnFwdE5m2.RowMax`, `AttnFwdE5m2.RowAcc`,
  `AttnFwdE5m2.Step`: one row's running maximum, `l_i` and `acc` over the loop (lines 72–109), as
  real-number bounds.
* `AttnFwdE5m2.Final`: `l_i` after the loop, the log-sum-exp (line 241) and the tolerances' numeric
  bounds.
* `AttnFwdE5m2.Mag`: the second bound on the output (`tolV`), over the computed weights.
* `AttnFwdE5m2.Cen`: the first bound's centred step (`acc − r·l_i` against `kap` times the centred
  sum of the exact weights), the `p·v` `tl.dot`'s chunks.
* `AttnFwdE5m2.Facts`: the grid, `int32` arithmetic, the descriptors, the constants, `qk_scale`.
* `AttnFwdE5m2.Row`, `AttnFwdE5m2.Tile`, `AttnFwdE5m2.Inner`: a row's state, the tiles' loads and
  stores, the loop of `_attn_fwd_inner` and its three calls.
* `AttnFwdE5m2.Program`: one program is safe (`safe_program`).
* Here: `main` by `TVA.meets_independent`, `consistent` and `satisfiable`.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The programs together -/

/-- `o` and `M` share no byte with `q`, `k` or `v`. -/
theorem outBytes_disj {d : Inst} (hd : d.ok) : Disjoint (outBytes d) (Rd d) := by
  obtain ⟨hoq, hok, hov, -, hMq, hMk, hMv⟩ := hd.disj
  exact Disjoint.union_left (Disjoint.union_right (Disjoint.union_right hoq hok) hov)
    (Disjoint.union_right (Disjoint.union_right hMq hMk) hMv)

/-! ## The claim -/

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ _ => Rd d) (fun d _ i => Wr d i) (fun d init i m => Qp d init (pm d i) (pbh d i) m)
    ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    exact safe_program hd hpre hA.dot hA.dot_total hA.exp2 hA.exp2_ninf hA.exp2_total hA.cvt
      hA.cvt_total hA.log2 hA.log2_total hA.div_full hA.div_total i.isLt
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hij' : (i : ℕ) ≠ j := fun h => hij (Fin.ext h)
    exact Disjoint.union_right ((outBytes_disj hd).mono_left (Wr_sub hd i.isLt))
      (Wr_disj hd i.isLt j.isLt hij')
  · -- no program reads what it writes
    intro d hd init _ i
    exact ((outBytes_disj hd).mono_left (Wr_sub hd i.isLt)).symm
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    have hn' : ∀ n : ℕ, 128 * (n / 128) + n % 128 = n := fun n => Nat.div_add_mod n 128
    have hsm : ∀ n : ℕ, n < d.N_CTX → n / 128 < g0 d := fun n hn => by
      rw [g0_eq hd]; have := hd.n_mul; omega
    refine ⟨fun bh n e hbh hn he => ?_, fun bh n hbh hn => ?_,
      fun z hz => hfr z fun i hzi => hz (Wr_sub hd i.isLt hzi)⟩
    · obtain ⟨hi, hpm, hpbh⟩ := prog_of hd hbh (hsm n hn)
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨bh * g0 d + n / 128, hi⟩
      have hrow := row0 hd (bh * g0 d + n / 128)
      simp only at hQ hagree
      rw [hpm, hpbh] at hQ hrow
      have hr : n % 128 < 128 := Nat.mod_lt _ (by norm_num)
      obtain ⟨c, hc, hout⟩ := (hQ ⟨n % 128, hr⟩).1 ⟨e, he⟩
      simp only [hn'] at hc hout
      refine ⟨c, ?_, hout⟩
      rw [← hc]
      refine Mem.read_congr fun x hx => hagree x (Or.inr (Or.inl (d.o.span_sub_elems ?_ hx)))
      refine ⟨n % 128, hr, e, he, ?_⟩
      unfold idx
      rw [show bh * d.N_CTX + n = 128 * (bh * g0 d + n / 128) + n % 128 by have := hn' n; omega]
    · obtain ⟨hi, hpm, hpbh⟩ := prog_of hd hbh (hsm n hn)
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨bh * g0 d + n / 128, hi⟩
      have hrow := row0 hd (bh * g0 d + n / 128)
      simp only at hQ hagree
      rw [hpm, hpbh] at hQ hrow
      have hr : n % 128 < 128 := Nat.mod_lt _ (by norm_num)
      obtain ⟨c, hc, hout⟩ := (hQ ⟨n % 128, hr⟩).2
      simp only [hn'] at hc hout
      refine ⟨c, ?_, hout⟩
      rw [← hc]
      refine Mem.read_congr fun x hx => hagree x (Or.inr (Or.inr (d.M.span_sub_elems ?_ hx)))
      exact ⟨n % 128, hr, by have := hn' n; omega⟩

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotE5m2Chunk, Platform.ref.dotTotal _,
    Platform.sm90ref.cvtSatF32E5m2, Platform.ref.cvtTotal _ _ _, Platform.sm90ref.exp2F32,
    Platform.sm90ref.exp2F32NegInf, Platform.ref.mathTotal, Platform.sm90ref.log2F32, Platform.ref.mathTotal,
    Platform.sm90ref.divFull, Platform.ref.divTotal _⟩⟩

/-! ## A covered instance -/

/-- A covered instance with the test's smallest `N_CTX` and `HEAD_DIM` and its `sm_scale`, one head
(`Z = H = 1`, which the test does not use: its `H` is `2` or `48`), `N_CTX = 128`, `HEAD_DIM = 64`,
`sm_scale = 0.5`, not causal, host descriptors; the tensors at the 16-byte-aligned addresses `2^21`,
`2·2^21`, `3·2^21`, `4·2^21` and `M` at `5·2^21`. -/
def testInst : Inst where
  Z := 1
  H := 1
  N_CTX := 128
  HEAD_DIM := 64
  sm_scale := FloatV.ofPat 0x3FE0000000000000
  causal := false
  warp_specialize := false
  q := ⟨2097152, 8192⟩
  k := ⟨4194304, 8192⟩
  v := ⟨6291456, 8192⟩
  o := ⟨8388608, 8192⟩
  M := ⟨10485760, 128⟩

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem half_facts : (FloatV.ofPat 0x3FE0000000000000 : F64).isFinite = true ∧
    (FloatV.ofPat 0x3FE0000000000000 : F64).toRat = 1 / 2 :=
  ⟨by decide +kernel, by decide +kernel⟩

/-- Disjoint byte ranges give disjoint elements. -/
theorem disj_of_le {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (hbc : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base)
    {S T : ℕ} (hS : S ≤ b.len) (hT : T ≤ c.len) : Disjoint (b.elems (below S)) (c.elems (below T)) := by
  refine Buf.elems_disjoint_of b c (Set.disjoint_left.mpr fun x hx hy => ?_)
    (fun j hj => lt_of_lt_of_le hj hS) (fun j hj => lt_of_lt_of_le hj hT)
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid testMem := fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [testMem]⟩
  refine ⟨testInst, testMem, ⟨Or.inl rfl, by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst, rows], ?_,
    by norm_num [testInst, rows], by norm_num [testInst], ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  · show (FloatV.ofPat 0x3FE0000000000000 : F64).isFinite = true ∧
      1 / 2 ^ 64 ≤ (FloatV.ofPat 0x3FE0000000000000 : F64).toRat ∧
      (FloatV.ofPat 0x3FE0000000000000 : F64).toRat ≤ 2 ^ 64
    rw [half_facts.2]
    exact ⟨half_facts.1, by norm_num, by norm_num⟩
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      exact disj_of_le _ _ (by simp [testInst, DType.bytes]) (by simp [testInst, rows])
        (by simp [testInst, rows])
  all_goals exact hv _ (by simp [testInst]) (by simp [testInst, DType.bytes]) (by simp [testInst, DType.bytes])

end Kernels.AttnFwdE5m2
