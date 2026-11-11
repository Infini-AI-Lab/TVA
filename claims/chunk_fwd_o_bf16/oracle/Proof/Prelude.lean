import Claim
import TVA.Proof
import Proof.Tail
import Proof.GeoOf
import Proof.ConvIdx

/-!
# The scalars of lines 108–121

`safe_body`: program `i`'s body is safe with the claim's post-condition for its chunk: the program
ids give `i_v`, `i_t`, `i_b`, `i_h`; the fixed-length branch computes `NT`, the variable-length
one reads the tables; `safe_tail1` does the rest.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {d : Inst} {init : Mem}

/-- Axis 0 of the grid. -/
theorem launch_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = pid0 d i := by
  simp [Launch.ctx, launch, pid0]

/-- Axis 1 of the grid. -/
theorem launch_pid1 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 1 = pid1 d i := by
  simp [Launch.ctx, launch, pid1]

/-- **One program** of the launch is safe from the initial memory. -/
theorem safe_body {P : Platform} (hP : Device P) (hd : d.ok) (hpre : Pre d init) {i : Nat}
    (hi : i < grid0 d * (d.B * d.HV)) :
    Safe ((launch d).env P) (Rs d) (Wg d (geoOf d i))
      (fun _ m' => Qg d (geoOf d i) (geoOf d i).itg m') init
      ((model d).body ((launch d).ctx i)) := by
  have hbk := bk_pos hd
  have hX := nVB_pos hd
  have hXeq := nVB_eq d hbk
  have hgrid := hd.grid0_lt
  have hp0lt : pid0 d i < grid0 d := by
    unfold pid0; exact Nat.mod_lt _ (by
      rcases Nat.eq_zero_or_pos (grid0 d) with h | h
      · rw [h] at hi; simp at hi
      · exact h)
  have hp1lt : pid1 d i < d.B * d.HV := Nat.mod_lt _ (Nat.mul_pos hd.B_pos hd.HV_pos)
  have hp0_31 : pid0 d i < 2 ^ 31 := lt_trans hp0lt hgrid
  have hp1_31 : pid1 d i < 2 ^ 31 := by
    have := hd.grid1_lt; have : pid1 d i < 2 ^ 16 := lt_trans hp1lt this
    omega
  have hXlt : (d.V + (d.bk - 1)) / d.bk < 2 ^ 31 := by
    have : nVB d ≤ grid0 d := by
      rw [grid0_eq]
      refine Nat.le_mul_of_pos_right _ ?_
      rcases Nat.eq_zero_or_pos (nCh d) with h | h
      · have := hp0lt; rw [grid0_eq, h] at this; simp at this
      · exact h
    rw [← hXeq]; omega
  have hHV31 : d.HV < 2 ^ 31 := by have := HV_lt16 hd; omega
  rw [body_eq d]
  simp only [tl.program_id, launch_pid0, launch_pid1, Prog.bind_eq, Prog.pure_eq]
  rw [← hXeq]
  refine safe_conv_i32_i64_scalar (IntV.ofNat (pid0 d i)) ?_
  rw [IntV.I32.toInt_ofNat hp0_31]
  refine safe_irem_val (w := 64) (pid0 d i) (nVB d) (by norm_num)
    (ofInt64 (by omega)) (ofNat64 (by omega)) hX
    (by have := Nat.mod_lt (pid0 d i) hX
        omega) ?_
  refine safe_idiv_val (w := 64) (pid0 d i) (nVB d) (by norm_num)
    (ofInt64 (by omega)) (ofNat64 (by omega)) hX
    (by have := Nat.div_le_self (pid0 d i) (nVB d); omega) ?_
  refine safe_conv_i32_i64_scalar (IntV.ofNat (pid1 d i)) ?_
  rw [IntV.I32.toInt_ofNat hp1_31]
  refine safe_idiv_val (w := 64) (pid1 d i) d.HV (by norm_num)
    (ofInt64 (by omega)) (ofNat64 (by omega)) hd.HV_pos
    (by have := Nat.div_le_self (pid1 d i) d.HV; omega) ?_
  refine safe_irem_val (w := 64) (pid1 d i) d.HV (by norm_num)
    (ofInt64 (by omega)) (ofNat64 (by omega)) hd.HV_pos
    (by have := Nat.mod_le (pid1 d i) d.HV; omega) ?_
  by_cases hv : d.varlen = true
  · simp only [hv, ↓reduceIte, Prog.ret_bind]
    have htab := hpre.tables hv
    obtain ⟨hcuV, hciV, hcuH, hciH, htok⟩ := htab
    have hidx := hd.idx_ok hv
    have hcNT : pid0 d i / nVB d < d.NT := by
      have := hp0lt; rw [grid0_eq] at this
      exact (Nat.div_lt_iff_lt_mul hX).mpr (by simpa [nCh, hv, Nat.mul_comm] using this)
    obtain ⟨hs0, hsN, hk0, hkl⟩ := htok.2.2.2.1 _ hcNT
    have hN := hd.N_lt hv
    have hB1 := hd.varlen_B hv
    have hTT := hd.T_le
    have hcibd := hd.ci_len hv
    have hcubd := hd.cu_len hv
    set c := pid0 d i / nVB d with hc
    have hgeo : geoOf d i = Geo.mk (pid0 d i % nVB d) (pid1 d i % d.HV) c (chkAt d c).toNat
        (cuAt d (seqOf d c)).toNat (cuAt d (seqOf d c + 1) - cuAt d (seqOf d c)).toNat := by
      simp [geoOf, hv, hc]
    have hg := geoOf_ok hd (fun h => (hpre.tables h).2.2.2.2) hi
    have hcuN : ∀ a, a ≤ d.N → 0 ≤ cuAt d a ∧ cuAt d a ≤ d.T := fun a ha =>
      ⟨htok.1.trans (cu_mono d htok (Nat.zero_le a) ha), (cu_mono d htok ha le_rfl).trans htok.2.2.1⟩
    have hseq : seqOf d c < d.N := by unfold seqOf; omega
    have hc31 : c < 2 ^ 31 := lt_of_le_of_lt (Nat.div_le_self _ _) hp0_31
    -- the pointer into `chunk_indices`
    have hc2 : ((Tensor.scalar (IntV.ofInt (c : Int)) : Tensor [] I64).item * (IntV.ofNat 2 : I64)).toInt =
        ((c * 2 : Nat) : Int) :=
      IntV.mul_nat (ofInt64 (by omega)) (ofNat64 (by omega)) (by omega)
    have hRs_ci : ∀ x, x ∈ d.ci.bytes → x ∈ Rs d := fun x hx =>
      Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨hv, Or.inr hx⟩)))))
    have hRs_cu : ∀ x, x ∈ d.cu.bytes → x ∈ Rs d := fun x hx =>
      Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨hv, Or.inl hx⟩)))))
    have hclt : 2 * c < d.ci.len := by omega
    have hclt1 : 2 * c + 1 < d.ci.len := by omega
    have hciptr : d.ci.ptr 0 + (Tensor.scalar (IntV.ofInt (c : Int)) : Tensor [] I64).item *
        (IntV.ofNat 2 : I64) = d.ci.ptr ((2 * c : Nat) : Int) :=
      ptr_add_val d.ci (by rw [hc2]; push_cast; ring)
    rw [hciptr]
    refine safe_load1_buf d.ci hclt hciV ((d.ci.span_sub_elems (S := {2 * c})
      (Set.mem_singleton _)).trans ((d.ci.elems_sub_bytes (fun j hj => by
        rw [Set.mem_singleton_iff.mp hj]; exact hclt)).trans hRs_ci)) fun x1 hx1 => ?_
    have hx1' : x1 = d.ciVal (2 * c) := by
      rw [hciH _ hclt] at hx1; exact (Option.some.inj hx1).symm
    have hseqfit : IntV.SFits 32 (seqAt d c) := ⟨by omega, by omega⟩
    refine safe_conv_idx32 hidx (n := seqAt d c) (by rw [hx1']; rfl) hseqfit ?_
    have hciptr1 : d.ci.ptr ((2 * c : Nat) : Int) + (IntV.ofNat 1 : I32) =
        d.ci.ptr ((2 * c + 1 : Nat) : Int) := by
      show (d.ci.ptr ((2 * c : Nat) : Int)).offset _ = _
      rw [Buf.ptr_offset, IntV.I32.toInt_ofNat (by norm_num)]
      push_cast; rfl
    rw [hciptr1]
    refine safe_load1_buf d.ci hclt1 hciV ((d.ci.span_sub_elems (S := {2 * c + 1})
      (Set.mem_singleton _)).trans ((d.ci.elems_sub_bytes (fun j hj => by
        rw [Set.mem_singleton_iff.mp hj]; exact hclt1)).trans hRs_ci)) fun y1 hy1 => ?_
    have hy1' : y1 = d.ciVal (2 * c + 1) := by
      rw [hciH _ hclt1] at hy1; exact (Option.some.inj hy1).symm
    refine safe_conv_idx64 hidx (n := chkAt d c) (by rw [hy1']; rfl) ?_
    -- the pointers into `cu_seqlens`
    have hseqv : (IntV.ofInt (seqAt d c) : I32).toInt = ((seqOf d c : Nat) : Int) := by
      rw [IntV.toInt_ofInt (w := 32) (by norm_num) hseqfit]; unfold seqOf; omega
    have hcult : seqOf d c < d.cu.len := by omega
    have hcult1 : seqOf d c + 1 < d.cu.len := by omega
    simp only [Tensor.item, Tensor.scalar]
    rw [ptr_add_val d.cu hseqv]
    refine safe_load1_buf d.cu hcult hcuV ((d.cu.span_sub_elems (S := {seqOf d c})
      (Set.mem_singleton _)).trans ((d.cu.elems_sub_bytes (fun j hj => by
        rw [Set.mem_singleton_iff.mp hj]; exact hcult)).trans hRs_cu)) fun z hz => ?_
    have hz' : z = d.cuVal (seqOf d c) := by
      rw [hcuH _ hcult] at hz; exact (Option.some.inj hz).symm
    refine safe_conv_idx64 hidx (n := cuAt d (seqOf d c)) (by rw [hz']; rfl) ?_
    have hcuptr1 : d.cu.ptr ((seqOf d c : Nat) : Int) + (IntV.ofNat 1 : I32) =
        d.cu.ptr ((seqOf d c + 1 : Nat) : Int) := by
      show (d.cu.ptr ((seqOf d c : Nat) : Int)).offset _ = _
      rw [Buf.ptr_offset, IntV.I32.toInt_ofNat (by norm_num)]
      push_cast; rfl
    rw [hcuptr1]
    refine safe_load1_buf d.cu hcult1 hcuV ((d.cu.span_sub_elems (S := {seqOf d c + 1})
      (Set.mem_singleton _)).trans ((d.cu.elems_sub_bytes (fun j hj => by
        rw [Set.mem_singleton_iff.mp hj]; exact hcult1)).trans hRs_cu)) fun w hw => ?_
    have hw' : w = d.cuVal (seqOf d c + 1) := by
      rw [hcuH _ hcult1] at hw; exact (Option.some.inj hw).symm
    refine safe_conv_idx64 hidx (n := cuAt d (seqOf d c + 1)) (by rw [hw']; rfl) ?_
    obtain ⟨hl0, hlT⟩ := hcuN (seqOf d c) (by omega)
    obtain ⟨hh0, hhT⟩ := hcuN (seqOf d c + 1) (by omega)
    have hmono' := cu_mono d htok (show seqOf d c ≤ seqOf d c + 1 by omega) (by omega)
    have hfit64 : ∀ n : Int, 0 ≤ n → n ≤ 2 ^ 22 → IntV.SFits 64 n := fun n h0 h1 =>
      ⟨by omega, by omega⟩
    refine safe_tail1 hP hd hpre (hgeo ▸ hg) rfl ?_ ?_ ?_ ?_ ?_ ?_
    · rw [hgeo]; exact ofInt64 (by have := Nat.mod_le (pid0 d i) (nVB d); omega)
    · rw [hgeo]; exact ofInt64 (by have := Nat.mod_le (pid1 d i) d.HV; omega)
    · rw [hgeo]; exact ofInt64 (by omega)
    · rw [hgeo]
      show (IntV.ofInt (chkAt d c) : I64).toInt = _
      rw [IntV.toInt_ofInt (w := 64) (by norm_num) (hfit64 _ hk0 (by
        have : chkAt d c * BT < cuAt d (seqOf d c + 1) - cuAt d (seqOf d c) := by
          unfold seqOf; exact hkl
        simp only [BT] at this; omega))]
      show chkAt d c = ((chkAt d c).toNat : Int)
      omega
    · rw [hgeo]
      show (IntV.ofInt (cuAt d (seqOf d c)) : I64).toInt = _
      rw [IntV.toInt_ofInt (w := 64) (by norm_num) (hfit64 _ hl0 (by omega))]
      show cuAt d (seqOf d c) = ((cuAt d (seqOf d c)).toNat : Int)
      omega
    · rw [hgeo]
      show ((IntV.ofInt (cuAt d (seqOf d c + 1)) : I64) - (IntV.ofInt (cuAt d (seqOf d c)) : I64)).toInt = _
      rw [IntV.toInt_sub (IntV.toInt_ofInt (w := 64) (by norm_num) (hfit64 _ hh0 (by omega)))
        (IntV.toInt_ofInt (w := 64) (by norm_num) (hfit64 _ hl0 (by omega))) ⟨by omega, by omega⟩]
      show cuAt d (seqOf d c + 1) - cuAt d (seqOf d c) =
        ((cuAt d (seqOf d c + 1) - cuAt d (seqOf d c)).toNat : Int)
      omega
  · have hvf : d.varlen = false := by simpa using hv
    simp only [hvf, Bool.false_eq_true, ↓reduceIte,
      Prog.ret_bind]
    have hT22 := hd.T_le
    have hB13 := hd.B_le
    have hib : pid1 d i / d.HV < d.B := (Nat.div_lt_iff_lt_mul hd.HV_pos).mpr hp1lt
    have hBT64 : BT = 64 := rfl
    have hY : (d.T + (BT - 1)) / BT ≤ 2 ^ 16 := by rw [hBT64]; omega
    have hcN : pid0 d i / nVB d < NTpad d := (Nat.div_lt_iff_lt_mul hX).mpr (by
      have := hp0lt; rw [grid0_eq] at this
      simpa [nCh, hvf, Nat.mul_comm] using this)
    refine safe_idiv_val (w := 32) (d.T + (BT - 1)) BT (by norm_num)
      (IntV.I32.add_nat (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_ofNat (by norm_num [BT]))
        (by omega))
      (IntV.I32.toInt_ofNat (by norm_num [BT])) (by norm_num [BT]) (by omega) ?_
    refine safe_conv_i32_i64_scalar _ ?_
    simp only [Tensor.item, Tensor.scalar]
    rw [IntV.toInt_ofInt (w := 32) (by norm_num) (sfits_nat (by norm_num) (by omega))]
    refine safe_conv_i32_i64_scalar (IntV.ofNat d.T) ?_
    rw [IntV.I32.toInt_ofNat (by omega)]
    have hNT : NTpad d = (d.T + (BT - 1)) / BT := by
      unfold NTpad
      have : d.T + BT - 1 = d.T + (BT - 1) := by simp [BT]
      rw [this]
    have hg := geoOf_ok hd (fun h => (hpre.tables h).2.2.2.2) hi
    have hgeo : geoOf d i = Geo.mk (pid0 d i % nVB d) (pid1 d i % d.HV)
        (pid1 d i / d.HV * NTpad d + pid0 d i / nVB d) (pid0 d i / nVB d)
        (pid1 d i / d.HV * d.T) d.T := by simp [geoOf, hvf]
    have hbT : (IntV.ofInt (((pid1 d i / d.HV : Nat)) : Int) * IntV.ofInt (d.T : Int) : I64).toInt =
        ((pid1 d i / d.HV * d.T : Nat) : Int) :=
      IntV.mul_nat (ofInt64 (by omega)) (ofInt64 (by omega)) (by
        have : pid1 d i / d.HV * d.T ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul (by omega) (by omega)
        omega)
    have hiT : (IntV.ofInt (((pid1 d i / d.HV : Nat)) : Int) *
        IntV.ofInt (((d.T + (BT - 1)) / BT : Nat) : Int) : I64).toInt =
        ((pid1 d i / d.HV * NTpad d : Nat) : Int) := by
      rw [hNT]
      exact IntV.mul_nat (ofInt64 (by omega)) (ofInt64 (by omega)) (by
        have : pid1 d i / d.HV * ((d.T + (BT - 1)) / BT) ≤ 8192 * 2 ^ 16 :=
          Nat.mul_le_mul (by omega) hY
        omega)
    refine safe_tail1 hP hd hpre (hgeo ▸ hg) rfl ?_ ?_ ?_ ?_ ?_ ?_
    · rw [hgeo]; exact ofInt64 (by have := Nat.mod_le (pid0 d i) (nVB d); omega)
    · rw [hgeo]; exact ofInt64 (by have := Nat.mod_le (pid1 d i) d.HV; omega)
    · rw [hgeo]
      exact IntV.add_nat hiT (ofInt64 (by have := Nat.div_le_self (pid0 d i) (nVB d); omega))
        (by
          have h1 : pid1 d i / d.HV * NTpad d ≤ 8192 * 2 ^ 16 := by
            rw [hNT]; exact Nat.mul_le_mul (by omega) hY
          have h2 : pid0 d i / nVB d < NTpad d := hcN
          rw [hNT] at h2
          have := Nat.div_le_self (pid0 d i) (nVB d)
          omega)
    · rw [hgeo]; exact ofInt64 (by have := Nat.div_le_self (pid0 d i) (nVB d); omega)
    · rw [hgeo]; exact hbT
    · rw [hgeo]; exact ofInt64 (by omega)

end Claims.ChunkFwdO
