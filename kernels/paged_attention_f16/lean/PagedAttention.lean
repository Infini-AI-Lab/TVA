import PagedAttention.Program

/-!
# `paged_attention_f16`: paged attention, decode

Source: `source/paged_attention.py`. The kernel is `paged_attention_kernel` (lines 30–72); it reads
the module-level names `triton` and `tl` (lines 26–27). The host (`paged_attention`, lines 75–97)
launches it with one program per sequence and query head on `float16` `q`, `k_cache`, `v_cache`
and `int32` `block_tables`, `seq_lens`, into the `float16` output `out`.

The claim: on every covered instance, every launch that runs ends normally; every element of
`out` holds a value, and for a sequence and head whose values are covered (`SeqVals`) the value is
finite and within an explicit tolerance of the exact attention over the sequence's tokens, read
through the block table as memory holds it at launch; every byte outside `out`'s elements is
unchanged.

* `PagedAttention.Model`: the model, the conditions, the specification, the assumptions.
* `PagedAttention.Real`, `.Lane`, `.RowMax`, `.RowAcc`, `.Step`: one row's running maximum, `l_i`
  and `acc` over the loop (lines 54–69), as real-number bounds.
* `PagedAttention.Final`: the end of a row (lines 71–72) and the tolerances' numeric bounds.
* `PagedAttention.Facts`: integers, constants, the row's state, one page (`row_step`).
* `PagedAttention.Program`: one program is safe (`safe_program`).
* Here: `main` by `TVA.meets_independent`, `consistent` and `satisfiable`.
-/

namespace Kernels.PagedAttention

open TVA TVA.Math.Float

/-! ## The launch -/

theorem pow2_ok {n : ℕ} (h : ∃ a, a ≤ 8 ∧ n = 2 ^ a) : ∃ e, n = 2 ^ e := by
  obtain ⟨a, -, ha⟩ := h; exact ⟨a, ha⟩

/-- **The launch is valid**: the grid `(num_seqs, num_heads, 1)` within CUDA's limits, fewer than
`2^31` programs, and block shapes of powers of two with at most `2^16` elements. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hSH := hd.grid_lt
  have hH1 := hd.heads_pos
  have hS31 : d.num_seqs < 2 ^ 31 := lt_of_le_of_lt (Nat.le_mul_of_pos_right _ hH1) hSH
  have hD := headDim_le hd
  have hP := pageSize_le hd
  obtain ⟨eD, heD⟩ := pow2_ok hd.head_dim
  obtain ⟨eP, -, -, heP⟩ := hd.page_size
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, ?_⟩
  · show d.num_seqs < 2 ^ 31; exact hS31
  · show d.num_heads < 2 ^ 16; exact hd.heads_lt
  · show 1 < 2 ^ 16; norm_num
  · rw [launch_size]; exact hSH
  · intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    have hPD : d.PAGE_SIZE * d.HEAD_DIM ≤ 2 ^ 20 := by
      have := Nat.mul_le_mul hP hD; exact this.trans (by norm_num)
    rcases hs with rfl | rfl | rfl | rfl | rfl
    · exact ⟨by simp; exact ⟨eP, heP⟩, by simp; omega⟩
    · exact ⟨by simp; exact ⟨eD, heD⟩, by simp; omega⟩
    · exact ⟨by simp; exact ⟨⟨eP, heP⟩, ⟨0, rfl⟩⟩, by simp; omega⟩
    · exact ⟨by simp; exact ⟨⟨0, rfl⟩, ⟨eD, heD⟩⟩, by simp; omega⟩
    · exact ⟨by simp; exact ⟨⟨eP, heP⟩, ⟨eD, heD⟩⟩, by simpa using hPD⟩

/-! ## The programs together -/

theorem outRow_sub {d : Inst} {s h : ℕ} (hs : s < d.num_seqs) (hh : h < d.num_heads) :
    outRow d s h ⊆ outElems d := by
  rintro _ ⟨e, he, rfl⟩; exact ⟨s, h, e, hs, hh, he, rfl⟩

theorem outElems_lt {d : Inst} (hd : d.ok) : ∀ j ∈ outElems d, j < d.out.len := by
  rintro _ ⟨s, h, e, hs, hh, he, rfl⟩; exact hd.out_len s h e hs hh he

theorem Wr_sub {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) :
    Wr d i ⊆ d.out.elems (outElems d) :=
  d.out.elems_mono (outRow_sub (pS_lt hd i) (pH_lt hi))

/-- `out` shares no byte with the inputs. -/
theorem out_Rd {d : Inst} (hd : d.ok) : Disjoint d.out.bytes (Rd d) := by
  obtain ⟨hq, hk, hv, hb, hs⟩ := hd.disj
  exact Disjoint.union_right (Disjoint.union_right (Disjoint.union_right (Disjoint.union_right hq hk) hv) hb) hs

theorem Wr_Rd {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) : Disjoint (Wr d i) (Rd d) :=
  (out_Rd hd).mono_left ((Wr_sub hd hi).trans (d.out.elems_sub_bytes (outElems_lt hd)))

theorem prog_eq (d : Inst) (i : ℕ) : pH d i * d.num_seqs + pS d i = i := by
  unfold pH pS; rw [mul_comm]; exact Nat.div_add_mod i d.num_seqs

/-- Different programs write different elements. -/
theorem Wr_disj {d : Inst} (hd : d.ok) {i j : ℕ} (hi : i < (launch d).size) (hj : j < (launch d).size)
    (hij : i ≠ j) : Disjoint (Wr d i) (Wr d j) := by
  refine d.out.elems_disjoint (Set.disjoint_left.mpr ?_)
  rintro _ ⟨e, he, rfl⟩ ⟨e', he', h⟩
  obtain ⟨h1, h2, -⟩ := hd.out_inj _ _ _ _ _ _ (pS_lt hd i) (pH_lt hi) he (pS_lt hd j) (pH_lt hj) he' h
  have := prog_eq d i
  have := prog_eq d j
  rw [h1, h2] at *
  omega

/-- Sequence `s`, head `h` is program `h·num_seqs + s`. -/
theorem prog_of {d : Inst} {s h : ℕ} (hs : s < d.num_seqs) (hh : h < d.num_heads) :
    h * d.num_seqs + s < (launch d).size ∧ pS d (h * d.num_seqs + s) = s ∧ pH d (h * d.num_seqs + s) = h := by
  have hS0 : 0 < d.num_seqs := by omega
  refine ⟨?_, ?_, ?_⟩
  · rw [launch_size]
    calc h * d.num_seqs + s < h * d.num_seqs + d.num_seqs := by omega
      _ = (h + 1) * d.num_seqs := by ring
      _ ≤ d.num_heads * d.num_seqs := Nat.mul_le_mul_right _ hh
      _ = d.num_seqs * d.num_heads := by ring
  · unfold pS; rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hs]
  · unfold pH; rw [Nat.add_comm, Nat.add_mul_div_right _ _ hS0, Nat.div_eq_of_lt hs, zero_add]

/-! ## The claim -/

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ _ => Rd d) (fun d _ i => Wr d i) (fun d init i m => Qp d init i m) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    exact safe_program hd hpre hA.exp_acc hA.exp_ninf hA.exp_total hA.div_full hA.div_total i.isLt
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hij' : (i : ℕ) ≠ j := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (Wr_Rd hd i.isLt) (Wr_disj hd i.isLt j.isLt hij')
  · -- no program reads what it writes
    intro d hd init _ i
    exact (Wr_Rd hd i.isLt).symm
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun s h e hs hh he => ?_, fun z hz => hfr z fun i hzi => hz (Wr_sub hd i.isLt hzi)⟩
    obtain ⟨hi, hps, hph⟩ := prog_of hs hh
    obtain ⟨m, hQ, hagree⟩ := hloc ⟨h * d.num_seqs + s, hi⟩
    simp only at hQ hagree
    obtain ⟨c, hc, hout⟩ := hQ ⟨e, he⟩
    rw [hps, hph] at hc hout
    refine ⟨c, ?_, hout⟩
    rw [← hc]
    refine Mem.read_congr fun x hx => hagree x (Or.inr (d.out.span_sub_elems ?_ hx))
    show outIdx d s h e ∈ outRow d (pS d (h * d.num_seqs + s)) (pH d (h * d.num_seqs + s))
    rw [hps, hph]
    exact ⟨e, he, rfl⟩

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.expF32,
    Platform.sm90ref.expF32NegInf, Platform.ref.mathTotal, Platform.ref.divFull, Platform.ref.divTotal _⟩⟩

/-! ## A covered instance -/

/-- A covered instance at the test's `head_dim` and `page_size` and its `scale`: one sequence,
one head (`num_kv_heads = 1`), `HEAD_DIM = 128`, `PAGE_SIZE = 16`, one page, `scale = 128^-0.5`
rounded to a double; contiguous tensors at the addresses `2^20`, `2·2^20`, …, `6·2^20`. -/
def testInst : Inst where
  num_seqs := 1
  num_heads := 1
  num_kv_heads := 1
  HEAD_DIM := 128
  PAGE_SIZE := 16
  num_pages := 1
  max_pages := 1
  scale := FloatV.ofPat 0x3FB6A09E667F3BCD
  stride_qs := 128
  stride_qh := 128
  stride_kp := 2048
  stride_kt := 128
  stride_kh := 128
  stride_bs := 1
  stride_os := 128
  stride_oh := 128
  q := ⟨1048576, 128⟩
  k_cache := ⟨2097152, 2048⟩
  v_cache := ⟨3145728, 2048⟩
  out := ⟨4194304, 128⟩
  block_tables := ⟨5242880, 1⟩
  seq_lens := ⟨6291456, 1⟩

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem scale_facts_test : (FloatV.ofPat 0x3FB6A09E667F3BCD : F64).isFinite = true ∧
    1 / 2 ^ 64 ≤ (FloatV.ofPat 0x3FB6A09E667F3BCD : F64).toRat ∧
    (FloatV.ofPat 0x3FB6A09E667F3BCD : F64).toRat ≤ 2 ^ 64 := by
  refine ⟨by decide +kernel, ?_, ?_⟩
  · have : (1 / 2 ^ 64 : ℚ) ≤ (FloatV.ofPat 0x3FB6A09E667F3BCD : F64).toRat := by decide +kernel
    exact this
  · have : (FloatV.ofPat 0x3FB6A09E667F3BCD : F64).toRat ≤ 2 ^ 64 := by decide +kernel
    exact this

/-- Every aligned `int32` read of `testMem` gives `0`. -/
theorem testMem_read {a : Int} (ha : Mem.Aligned .i32 a) :
    ∃ v : I32, testMem.read .i32 a = some v ∧ v.toInt = 0 := by
  have hb : testMem.readBytes a 4 = some [0, 0, 0, 0] := by
    simp [Mem.readBytes, testMem]
  unfold Mem.read
  rw [show DType.i32.bytes = 4 from rfl]
  simp only [ha, ite_true, hb, Option.map_some]
  exact ⟨_, rfl, by decide⟩

/-- Disjoint byte ranges. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (hbc : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_setOf_eq] at hx hy
  omega

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid testMem := fun b h1 h2 h3 => ⟨h1, h2, h3, fun _ _ => by simp [testMem]⟩
  refine ⟨testInst, testMem, ⟨by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    by norm_num [testInst], by norm_num [testInst], by norm_num [testInst], ⟨7, by norm_num, rfl⟩,
    ⟨4, by norm_num, by norm_num, rfl⟩, by norm_num [testInst], by norm_num [testInst], by norm_num [testInst],
    ?_, ?_, ?_, ?_, by norm_num [testInst], ?_, ?_, scale_facts_test⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro s h e hs hh he
    simp only [testInst, qIdx] at hs hh he ⊢; omega
  · intro s h e hs hh he
    simp only [testInst, outIdx] at hs hh he ⊢; omega
  · intro p r g e hp hr hg he
    simp only [testInst, kvIdx] at hp hr hg he ⊢; omega
  · intro s j hs hj
    simp only [testInst, btIdx] at hs hj ⊢; omega
  · intro s h e s' h' e' hs hh he hs' hh' he' heq
    simp only [testInst, outIdx] at hs hh he hs' hh' he' heq ⊢; omega
  · refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> exact bytes_disj _ _ (by simp [testInst, DType.bytes])
  all_goals try exact hv _ (by simp [testInst]) (by simp [testInst, DType.bytes]) (by simp [testInst, DType.bytes])
  intro s hs
  have hs0 : s = 0 := by simp only [testInst] at hs; omega
  subst hs0
  obtain ⟨v, hrd, hv0⟩ := testMem_read (a := (testInst.seq_lens.ptr ((0 : ℕ) : ℤ)).addr)
    ⟨by simp [testInst, Buf.ptr, DType.bytes], by simp [testInst, Buf.ptr, DType.bytes],
      by simp [testInst, Buf.ptr, DType.bytes]⟩
  refine ⟨v, hrd, by omega, by rw [hv0]; positivity, fun j hj => ?_⟩
  rw [hv0] at hj
  omega

end Kernels.PagedAttention
