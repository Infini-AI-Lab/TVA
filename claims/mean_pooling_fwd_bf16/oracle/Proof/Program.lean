import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Lanes
import Proof.Walk

/-!
# The scalars of one program

`safe_program` walks lines 94–106: the two program ids converted to `int64`, `i_d`, `i_t`, `i_b`,
`i_h` by `%` and `//`, then either the four table reads, `T = eos − bos` and the unused `tl.cdiv`
(variable length), or `NT = tl.cdiv(T, BT)`, `i_tg = i_b·NT + i_t` and `bos = i_b·T` (fixed
length); both end in `tail_safe` with the scalars' values.
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- An `int32` of natural value `n` converted to `int64` (`ConvRel`) keeps it. -/
theorem conv_i64_val {P : Platform} {a : I32} {n : Nat} (ha : a.toInt = n) {r : Tensor [] I64}
    (hr : ∀ j, ConvRel P .i32 .i64 ((Tensor.scalar a).get j) (r.get j)) : r.item.toInt = n := by
  have h1 : r.item = IntV.ofInt a.toInt := hr ()
  rw [h1]
  refine (IntV.toInt_ofInt (by norm_num) ?_).trans ha
  have hs := IntV.sfits_toInt a
  simp only [IntV.SFits] at hs ⊢
  omega

/-- The bytes of a table entry read by the program are in its read footprint. -/
theorem R_mem_extra {d : Inst} (hv : d.varlen = true) (i : Nat) {p : Ptr d.idxD}
    (h : p.addr = (d.ci.ptr (2 * rG d i : Nat)).addr ∨ p.addr = (d.ci.ptr (2 * rG d i + 1 : Nat)).addr ∨
      p.addr = (d.cu.ptr (seqN d i : Nat)).addr ∨ p.addr = (d.cu.ptr (seqN d i + 1 : Nat)).addr) :
    Mem.span d.idxD p.addr ⊆ R d i := by
  intro z hz
  refine Or.inr ?_
  unfold extraReads
  simp only [hv, ite_true, Set.mem_union]
  rcases h with h | h | h | h <;> rw [h] at hz
  · exact Or.inl (Or.inl (Or.inl (by exact_mod_cast hz)))
  · exact Or.inl (Or.inl (Or.inr (by exact_mod_cast hz)))
  · exact Or.inl (Or.inr (by exact_mod_cast hz))
  · exact Or.inr (by exact_mod_cast hz)

/-- **One program** is safe: it reads within `R`, writes within `W` and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    (hP : Device E.P) {i : Nat} (hi : i < gx d * gy d) {c : Ctx}
    (hc0 : c.pid 0 = p0 d i) (hc1 : c.pid 1 = p1 d i) :
    Safe E (R d i) (W d i) (fun _ m => Q d i m) init ((model d).body c) := by
  have hT : d.varlen = true → TablesOk d := fun hv => (hpre.2.2.2 hv).2.2.2.2
  have hciH : d.varlen = true → d.ci.Holds init d.ciVal := fun hv => (hpre.2.2.2 hv).2.2.2.1
  have hcuH : d.varlen = true → d.cu.Holds init d.cuVal := fun hv => (hpre.2.2.2 hv).2.2.1
  have hsz := size_lt hd
  obtain ⟨hp0lt, hdivlt, hp1eq⟩ := coords_lt hi
  have hgy : gy d < 2 ^ 16 := hd.BH_lt
  have hp0 : p0 d i < 2 ^ 31 := by
    have : gx d ≤ gx d * gy d := Nat.le_mul_of_pos_right _ (by omega)
    omega
  have hp1 : p1 d i < 2 ^ 31 := by omega
  have hnD := nD_le hd
  have hnD0 := nD_pos hd
  have hH16 : d.H < 2 ^ 16 := lt_of_le_of_lt (Nat.le_mul_of_pos_left _ hd.B_pos) hd.BH_lt
  have hcI : ((IntV.ofNat ((d.D + d.BD - 1) / d.BD) : I64)).toInt = (nD d : Int) :=
    i64_ofNat (by unfold nD at hnD; omega)
  have hHI : (IntV.ofNat d.H : I64).toInt = (d.H : Int) := i64_ofNat (by omega)
  have hbt := bt_le hd
  have hBTI : (IntV.ofNat d.BT : I64).toInt = (d.BT : Int) := i64_ofNat (by omega)
  rw [body_eq]
  have hpid0 : tl.program_id c 0 = (IntV.ofNat (p0 d i) : I32) := by simp only [tl.program_id, hc0]
  have hpid1 : tl.program_id c 1 = (IntV.ofNat (p1 d i) : I32) := by simp only [tl.program_id, hc1]
  rw [hpid0, hpid1]
  -- line 94: `program_id(0).to(tl.int64)`, `% X`, `// X`
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j; exact ⟨IntV.ofInt ((IntV.ofNat (p0 d i) : I32).toInt), rfl⟩
  intro r1 hr1
  have hv1 : r1.item.toInt = (p0 d i : Int) := conv_i64_val (IntV.I32.toInt_ofNat hp0) hr1
  refine safe_irem1 (x := r1.item) (y := (IntV.ofNat ((d.D + d.BD - 1) / d.BD) : I64)) ?_ ?_ ?_
  · rw [hv1]; omega
  · rw [hcI]; exact_mod_cast hnD0
  intro r2 hr2
  have hv2 : r2.item.toInt = (iD d i : Int) := by
    rw [hr2, hv1, hcI]; unfold iD; push_cast; rfl
  refine safe_idiv1 (x := r1.item) (y := (IntV.ofNat ((d.D + d.BD - 1) / d.BD) : I64)) ?_ ?_ ?_
  · rw [hv1]; omega
  · rw [hcI]; exact_mod_cast hnD0
  intro r3 hr3
  have hv3 : r3.item.toInt = (rG d i : Int) := by
    rw [hr3, hv1, hcI]; unfold rG; push_cast; rfl
  -- line 95: `program_id(1).to(tl.int64)`
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j; exact ⟨IntV.ofInt ((IntV.ofNat (p1 d i) : I32).toInt), rfl⟩
  intro r4 hr4
  have hv4 : r4.item.toInt = (p1 d i : Int) := conv_i64_val (IntV.I32.toInt_ofNat hp1) hr4
  -- line 96: `// H`, `% H`
  refine safe_idiv1 (x := r4.item) (y := (IntV.ofNat d.H : I64)) ?_ ?_ ?_
  · rw [hv4]; omega
  · rw [hHI]; exact_mod_cast hd.H_pos
  intro r5 hr5
  have hv5 : r5.item.toInt = (bOf d i : Int) := by
    rw [hr5, hv4, hHI]; unfold bOf; push_cast; rfl
  refine safe_irem1 (x := r4.item) (y := (IntV.ofNat d.H : I64)) ?_ ?_ ?_
  · rw [hv4]; omega
  · rw [hHI]; exact_mod_cast hd.H_pos
  intro r6 hr6
  have hv6 : r6.item.toInt = (hOf d i : Int) := by
    rw [hr6, hv4, hHI]; unfold hOf; push_cast; rfl
  by_cases hv : d.varlen = true
  · -- variable length: four table reads
    simp only [hv, ite_true]
    obtain ⟨hrNT, hsN, hc0', hmono, hcuT, hk0, hklt⟩ := row_facts hd hv (hT hv) hi
    have hT22 := hd.T_le
    have hz1 : IntV.SFits 32 (entryInt d.idxD (d.ciVal (2 * rG d i))) := by
      show IntV.SFits 32 (seqAt d (rG d i))
      have hN31 : ((d.N : Nat) : Int) < 2 ^ 31 := by exact_mod_cast hd.N_lt hv
      obtain ⟨hs0, hsN', -, -⟩ := (hT hv).2.2.2 (rG d i) hrNT
      simp only [IntV.SFits]
      omega
    have hrG31 : rG d i < 2 ^ 31 := by
      have : rG d i < ntG d := (rG_lt hd hi).1
      have h2 : ntG d ≤ gx d := by unfold gx; exact Nat.le_mul_of_pos_left _ hnD0
      have h3 : gx d ≤ gx d * gy d := Nat.le_mul_of_pos_right _ (Nat.mul_pos hd.B_pos hd.H_pos)
      omega
    have hk1 : (r3.item * (IntV.ofNat 2 : I64)).toInt = ((2 * rG d i : Nat) : Int) := by
      have hm := IntV.I64.mul_nat (x := rG d i) (y := 2) hv3 (i64_ofNat (by norm_num)) (by omega)
      rw [hm]; push_cast; ring
    have hk2 : (r3.item * (IntV.ofNat 2 : I64)).toInt + (IntV.ofNat 1 : I32).toInt
        = ((2 * rG d i + 1 : Nat) : Int) := by
      rw [hk1, IntV.I32.toInt_ofNat (by norm_num)]; push_cast; ring
    have hcil : 2 * rG d i + 1 < d.ci.len := by have := hd.ci_len hv; omega
    have hcul : seqN d i + 1 < d.cu.len := by have := hd.cu_len hv; omega
    refine load_conv_i32 (dt := d.idxD) (b := d.ci) (v := d.ciVal) (n := 2 * rG d i)
      (hd.idx_ok hv) (hciH hv) (by omega) (addr_ptr_add d.ci hk1)
      (R_mem_extra hv i (Or.inl (by rw [addr_ptr_add d.ci hk1]))) hz1 ?_
    intro a1 ha1
    have hn1 : a1.item.toInt = ((seqN d i : Nat) : Int) := by
      rw [ha1]; unfold seqN
      obtain ⟨hs0, -, -, -⟩ := (hT hv).2.2.2 (rG d i) hrNT
      rw [Int.toNat_of_nonneg hs0]; rfl
    refine load_conv_i64 (dt := d.idxD) (b := d.ci) (v := d.ciVal) (n := 2 * rG d i + 1)
      (hd.idx_ok hv) (hciH hv) hcil (addr_ptr_add2 d.ci hk2)
      (R_mem_extra hv i (Or.inr (Or.inl (by rw [addr_ptr_add2 d.ci hk2])))) ?_
    intro t2 ht2
    have hchk : t2.item.toInt = (chk d i : Int) := by
      rw [ht2]; simp only [chk, hv, ite_true]; rw [Int.toNat_of_nonneg hk0]; rfl
    refine load_conv_i64 (dt := d.idxD) (b := d.cu) (v := d.cuVal) (n := seqN d i)
      (hd.idx_ok hv) (hcuH hv) (by omega) (addr_ptr_add d.cu hn1)
      (R_mem_extra hv i (Or.inr (Or.inr (Or.inl (by rw [addr_ptr_add d.cu hn1]))))) ?_
    intro b3 hb3
    have hbs : b3.item.toInt = (bos d i : Int) := by
      rw [hb3]; simp only [bos, hv, ite_true]; rw [Int.toNat_of_nonneg hc0']; rfl
    have hn2 : a1.item.toInt + (IntV.ofNat 1 : I32).toInt = ((seqN d i + 1 : Nat) : Int) := by
      rw [hn1, IntV.I32.toInt_ofNat (by norm_num)]; push_cast; ring
    refine load_conv_i64 (dt := d.idxD) (b := d.cu) (v := d.cuVal) (n := seqN d i + 1)
      (hd.idx_ok hv) (hcuH hv) hcul (addr_ptr_add2 d.cu hn2)
      (R_mem_extra hv i (Or.inr (Or.inr (Or.inr (by rw [addr_ptr_add2 d.cu hn2]))))) ?_
    intro e4 he4
    have hes : e4.item.toInt = (((cuAt d (seqN d i + 1)).toNat : Nat) : Int) := by
      rw [he4, Int.toNat_of_nonneg (by omega)]; rfl
    have hb3' : b3.item.toInt = (((cuAt d (seqN d i)).toNat : Nat) : Int) := by
      rw [hb3, Int.toNat_of_nonneg hc0']; rfl
    have hlenS : (e4.item - b3.item).toInt = (lenS d i : Int) := by
      rw [IntV.I64.sub_nat hes hb3' (by omega) (by omega)]
      simp only [lenS, hv, ite_true]
    have hlen22 := lenS_le hd hT hi
    refine safe_cdiv1 hlenS hBTI (bt_pos hd) (by decide) (by omega) ?_
    intro _ _
    exact tail_safe hd hpre hP hT hi hv2 hv6 (itg := r3.item) (by simp only [outRow, hv, ite_true]; exact hv3)
      hchk hbs hlenS
  · -- fixed length
    have hvf : d.varlen = false := by cases h : d.varlen <;> simp_all
    simp only [hvf, Bool.false_eq_true, ite_false]
    have hTI : (IntV.ofNat d.T : I32).toInt = (d.T : Int) := IntV.I32.toInt_ofNat hd.T_lt
    have hBT32 : (IntV.ofNat d.BT : I32).toInt = (d.BT : Int) := IntV.I32.toInt_ofNat (by omega)
    have hT22 := hd.T_le
    refine safe_cdiv1 hTI hBT32 (bt_pos hd) (by decide) (by omega) ?_
    intro nt hnt
    refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
    · intro j; exact ⟨IntV.ofInt (nt.item.toInt), rfl⟩
    intro nt64 hnt64
    have hnt' : nt64.item.toInt = (ntFix d : Int) := conv_i64_val (by rw [hnt]; rfl) hnt64
    refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
    · intro j; exact ⟨IntV.ofInt ((IntV.ofNat d.T : I32).toInt), rfl⟩
    intro t64 ht64
    have hT64 : t64.item.toInt = (d.T : Int) := conv_i64_val hTI ht64
    have hb := (bOf_lt hd hi).1
    have hrg : rG d i < ntFix d := by have := (rG_lt hd hi).1; simpa [ntG, hvf] using this
    have hntf := (ntFix_mul d hd).2
    have hbnt : bOf d i * ntFix d + rG d i < 2 ^ 63 := by
      have h1 : bOf d i * ntFix d + rG d i < d.B * ntFix d := by
        have := Nat.mul_le_mul_right (ntFix d) (show bOf d i + 1 ≤ d.B by omega)
        rw [Nat.add_mul, Nat.one_mul] at this; omega
      have h2 : ntFix d ≤ 2 ^ 23 := by
        have : ntFix d ≤ ntFix d * d.BT := Nat.le_mul_of_pos_right _ (bt_pos hd)
        omega
      have h4 : d.B * ntFix d ≤ 8192 * 2 ^ 23 := Nat.mul_le_mul hd.B_le h2
      omega
    have hitg : (r5.item * nt64.item + r3.item).toInt = (outRow d i : Int) := by
      have h1 := IntV.I64.mul_nat hv5 hnt' (by omega)
      have h2 := IntV.I64.add_nat h1 hv3 hbnt
      rw [h2]; simp only [outRow, hvf, Bool.false_eq_true, ite_false]
    have hbs : (r5.item * t64.item).toInt = (bos d i : Int) := by
      have h4 : bOf d i * d.T ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul (by have := hd.B_le; omega) hT22
      rw [IntV.I64.mul_nat hv5 hT64 (by omega)]
      simp only [bos, hvf, Bool.false_eq_true, ite_false]
    exact tail_safe hd hpre hP hT hi hv2 hv6 hitg (by simp only [chk, hvf, Bool.false_eq_true, ite_false]; exact hv3)
      hbs (by rw [hT64]; simp only [lenS, hvf, Bool.false_eq_true, ite_false])

end Claims.MeanPoolingFwd
