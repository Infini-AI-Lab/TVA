import Proof.BodyRest

/-!
# `chunk_delta_h_bf16`: one program of the launch

The program with `program_id(0) = i` decodes its sequence, value head and block of value columns,
reads its sequence's bounds (`cu_seqlens`, variable length) or computes them (fixed length), and
runs `body_rest` (`BodyRest`).
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- **One program** of the launch is safe from `init`: it reads outside the outputs, writes inside
`W d i` and leaves `Q d init i`. -/
theorem safe_program {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {init : Mem}
    (hpre : Pre d init) (hT1 : d.vf = false → TripNonvf d E init)
    (hT2 : d.vf = true → TripVf d E init) {i : Nat}
    (hi : i < gridN d) (c : Ctx) (hc : c.pid 0 = i) :
    Safe E (Rset d) (W d i) (fun _ m => Q d init i m) init ((model d).body c) := by
  obtain ⟨hn, hhv, hjv, -⟩ := decode hd hi
  unfold model fwd_kernel_h
  simp (config := {zeta := false}) only []
  have hNVp : 0 < NV d := by have := hd.V_pos; unfold NV BV; omega
  have hHVp := hd.HV_pos
  have hgrid := hd.grid_lt
  have hNVb : NV d < 2 ^ 16 := by have := hd.V_lt; unfold NV BV; omega
  have hNVi : (IntV.ofNat ((d.V + (BV - 1)) / BV) : I64).toInt = (NV d : Int) := by
    rw [IntV.I64.toInt_ofNat (by have := hd.V_lt; unfold BV; omega)]
    unfold NV; congr 2
  have hHVi : (IntV.ofNat d.HV : I64).toInt = (d.HV : Int) := IntV.I64.toInt_ofNat (by have := hd.HV_lt; omega)
  refine safe_convert (fun _ => ConvRel.exists_i32_i64 _) fun r0 hr0 => ?_
  have hpid : r0.item.toInt = (i : Int) := by
    refine (ConvRel.toInt_i32_i64 (hr0 ())).trans ?_
    show (tl.program_id c 0).toInt = _
    rw [tl.program_id, hc]
    exact IntV.I32.toInt_ofNat (by omega)
  refine safe_irem1 (by rw [hpid]; positivity) (by rw [hNVi]; exact_mod_cast hNVp) fun r1 hr1 => ?_
  refine safe_idiv1 (by rw [hpid]; positivity) (by rw [hNVi]; exact_mod_cast hNVp) fun r2 hr2 => ?_
  have hnh0 : 0 ≤ r2.item.toInt := by rw [hr2, hpid, hNVi]; positivity
  refine safe_idiv1 hnh0 (by rw [hHVi]; exact_mod_cast hHVp) fun r3 hr3 => ?_
  refine safe_irem1 hnh0 (by rw [hHVi]; exact_mod_cast hHVp) fun r4 hr4 => ?_
  have hiv : r1.item.toInt = (jvOf d i : Int) := by rw [hr1, hpid, hNVi]; rfl
  have hinh : r2.item.toInt = ((i / NV d : Nat) : Int) := by rw [hr2, hpid, hNVi]; rfl
  have hin : r3.item.toInt = (nOf d i : Int) := by rw [hr3, hr2, hpid, hNVi, hHVi]; rfl
  have hih : r4.item.toInt = (hvOf d i : Int) := by rw [hr4, hr2, hpid, hNVi, hHVi]; rfl
  have hinh' : r2.item.toInt = ((nOf d i * d.HV + hvOf d i : Nat) : Int) := by
    rw [hinh]; congr 1
    unfold nOf hvOf
    have := Nat.div_add_mod (i / NV d) d.HV
    rw [mul_comm]; exact this.symm
  by_cases hvl : d.varlen = true
  · simp only [hvl, ite_true]
    have hvar : d.varlen = true := hvl
    obtain ⟨hvcu, hvco, hcuH, hcoR⟩ := hpre.2.2.2.2.2.2.2.2 hvar
    have hcuD := hd.var_cuD hvar
    have hcu_len := hd.cu_len hvar
    have hco_len := hd.co_len hvar
    have hnN : nOf d i < d.N := hn
    have hcuRs : d.cu.bytes ⊆ Rset d := bytes_sub_Rset d.cu hd.disj_h.2.2.2.2.2.2.2.1.symm
      hd.disj_vn.2.2.2.2.2.2.1.symm hd.disj_ht.2.2.2.2.2.1.symm
    have hcoRs : d.co.bytes ⊆ Rset d := bytes_sub_Rset d.co hd.disj_h.2.2.2.2.2.2.2.2.symm
      hd.disj_vn.2.2.2.2.2.2.2.symm hd.disj_ht.2.2.2.2.2.2.symm
    have hP1 : d.cu.ptr 0 + r3.item = d.cu.ptr ((nOf d i : Nat) : Int) := ptr_off d.cu rfl hin
    have hP2 : d.cu.ptr 0 + r3.item + (IntV.ofInt 1 : I32) = d.cu.ptr ((nOf d i + 1 : Nat) : Int) := by
      show (d.cu.ptr 0 + r3.item).offset _ = _
      rw [hP1, IntV.toInt_ofInt_one (by norm_num), Buf.ptr_offset]; push_cast; rfl
    have hP3 : d.co.ptr 0 + r3.item = d.co.ptr ((nOf d i : Nat) : Int) := ptr_off d.co rfl hin
    rw [hP2, hP1, hP3]
    refine safe_load1_buf d.cu (i := nOf d i) (by omega) hvcu
      (fun x hx => hcuRs (d.cu.span_sub (by omega) hx)) fun v1 hv1 => ?_
    have hv1' : v1 = d.cuVal (nOf d i) := by
      have := hcuH (nOf d i) (by omega)
      rw [hv1] at this; exact Option.some.inj this
    refine safe_convert (fun _ => (conv_cu d.cuD hcuD).1 _) fun r5 hr5 => ?_
    have hbos : r5.item.toInt = (cuAt d (nOf d i) : Int) := by
      have := (conv_cu d.cuD hcuD).2 _ _ (hr5 ())
      rw [hv1'] at this
      exact this.trans (entry_eq hd hvar (by omega))
    refine safe_load1_buf d.cu (i := nOf d i + 1) (by omega) hvcu
      (fun x hx => hcuRs (d.cu.span_sub (by omega) hx)) fun v2 hv2 => ?_
    have hv2' : v2 = d.cuVal (nOf d i + 1) := by
      have := hcuH (nOf d i + 1) (by omega)
      rw [hv2] at this; exact Option.some.inj this
    refine safe_convert (fun _ => (conv_cu d.cuD hcuD).1 _) fun r6 hr6 => ?_
    have heos : r6.item.toInt = (cuAt d (nOf d i + 1) : Int) := by
      have := (conv_cu d.cuD hcuD).2 _ _ (hr6 ())
      rw [hv2'] at this
      exact this.trans (entry_eq hd hvar (by omega))
    have hmono := cuAt_mono hd hvar (m := nOf d i) (n := nOf d i + 1) (by omega) (by omega)
    have hend := cuAt_mono hd hvar (m := nOf d i + 1) (n := d.N) (by omega) le_rfl
    have hNe := cuAt_N hd hvar
    have hT22 := hd.T_ctx
    have hlenv : lenOf d (nOf d i) = cuAt d (nOf d i + 1) - cuAt d (nOf d i) := by simp [lenOf, hvar]
    have hTv : (r6.item - r5.item).toInt = (lenOf d (nOf d i) : Int) := by
      rw [hlenv]
      exact IntV.sub_nat heos hbos hmono (by omega)
    have hbosv : bosOf d (nOf d i) = cuAt d (nOf d i) := by simp [bosOf, rowOf, startOf, hvar]
    have hBTi : (IntV.ofNat BT : I64).toInt = (BT : Int) := IntV.I64.toInt_ofNat (by unfold BT; norm_num)
    have hB1 : (IntV.ofNat (BT - 1) : I64).toInt = ((BT - 1 : Nat) : Int) := IntV.I64.toInt_ofNat (by unfold BT; norm_num)
    have hsum : (r6.item - r5.item + IntV.ofNat (BT - 1) : I64).toInt = ((lenOf d (nOf d i) + (BT - 1) : Nat) : Int) :=
      IntV.add_nat hTv hB1 (by unfold BT; omega)
    refine safe_idiv1 (by rw [hsum]; positivity) (by rw [hBTi]; unfold BT; norm_num) fun rN hrN => ?_
    have hNT : rN.item.toInt = (nch d (nOf d i) : Int) := by
      rw [hrN, hsum, hBTi]
      have e : lenOf d (nOf d i) + (BT - 1) = lenOf d (nOf d i) + BT - 1 := by unfold BT; omega
      unfold nch; rw [e]; exact (Int.natCast_div _ _).symm
    refine safe_load1_buf d.co (i := nOf d i) (by omega) hvco
      (fun x hx => hcoRs (d.co.span_sub (by omega) hx)) fun boh hboh0 => ?_
    have hboh' : boh = IntV.ofNat (cbase d (nOf d i)) := by
      have := hcoR (nOf d i) (by omega)
      rw [hboh0] at this; exact Option.some.inj this
    have hcb := cbase_le hd hvar (n := nOf d i) (by omega)
    have hcuT : cuAt d (nOf d i) ≤ d.T := by
      have := cuAt_mono hd hvar (m := nOf d i) (n := d.N) (by omega) le_rfl
      omega
    have hNb := hd.N_batch.1
    have hbohv : boh.toInt = (cbase d (nOf d i) : Int) := by
      rw [hboh']; exact IntV.I64.toInt_ofNat (by omega)
    have hTl : (r6.item - r5.item).toInt = (lenOf d (nOf d i) : Int) := hTv
    have hbosl : r5.item.toInt = (bosOf d (nOf d i) : Int) := by rw [hbosv]; exact hbos
    by_cases hvf : d.vf = true
    · simp only [hvf, ite_true]
      exact safe_body_vf hd hP hpre hvf (hT2 hvf) hn hhv hjv hiv hinh' hih hbosl hbohv hTl
        (runLoop_var hd hn hhv hbosl hih hTl hNT)
    · have hvf' : d.vf = false := by simpa using hvf
      simp only [hvf, Bool.false_eq_true, ite_false]
      exact safe_body_nonvf hd hP hpre hvf' (hT1 hvf') hn hhv hjv hiv hinh' hih hbosl hbohv hTl
        (runLoop_var hd hn hhv hbosl hih hTl hNT)
  · have hfix : d.varlen = false := by simpa using hvl
    simp only [hvl, Bool.false_eq_true, ite_false]
    have hT31 := hd.T_lt
    have hT22 := hd.T_ctx
    have hBT31 : (BT - 1) < 2 ^ 31 := by unfold BT; norm_num
    refine safe_convert (fun _ => ConvRel.exists_i32_i64 _) fun rT hrT => ?_
    have hT64 : rT.item.toInt = (d.T : Int) :=
      (ConvRel.toInt_i32_i64 (hrT ())).trans (IntV.I32.toInt_ofNat (by omega))
    have hTi : (IntV.ofNat d.T : I32).toInt = (d.T : Int) := IntV.I32.toInt_ofNat (by omega)
    have hBTi : (IntV.ofNat BT : I32).toInt = (BT : Int) := IntV.I32.toInt_ofNat (by unfold BT; norm_num)
    have hB1 : (IntV.ofNat (BT - 1) : I32).toInt = ((BT - 1 : Nat) : Int) := IntV.I32.toInt_ofNat hBT31
    have hsum : ((IntV.ofNat d.T : I32) + IntV.ofNat (BT - 1)).toInt = ((d.T + (BT - 1) : Nat) : Int) :=
      IntV.add_nat hTi hB1 (by unfold BT at *; omega)
    refine safe_idiv1 (w := 32) (by rw [hsum]; positivity) (by rw [hBTi]; unfold BT; norm_num) fun rN hrN => ?_
    have hNT : rN.item.toInt = (nch d (nOf d i) : Int) := by
      rw [hrN, hsum, hBTi]
      have hl : lenOf d (nOf d i) = d.T := by simp [lenOf, hfix]
      unfold nch; rw [hl]
      have e : d.T + (BT - 1) = d.T + BT - 1 := by unfold BT; omega
      rw [e]; exact (Int.natCast_div _ _).symm
    refine safe_convert (fun _ => ConvRel.exists_i32_i64 _) fun rB hrB => ?_
    have hNT64 : rB.item.toInt = (nch d (nOf d i) : Int) := (ConvRel.toInt_i32_i64 (hrB ())).trans hNT
    have hbosv : bosOf d (nOf d i) = nOf d i * d.T := by simp [bosOf, rowOf, startOf, hfix]
    have hcbv : cbase d (nOf d i) = nOf d i * nch d (nOf d i) := by
      simp only [cbase, hfix, Bool.false_eq_true, ite_false, nch, BT]
      have hl : lenOf d (nOf d i) = d.T := by simp [lenOf, hfix]
      rw [hl]
    have hnN : nOf d i ≤ 8192 := by have := hd.N_batch.1; omega
    have hbos : (r3.item * rT.item).toInt = (bosOf d (nOf d i) : Int) := by
      rw [hbosv]
      exact IntV.mul_nat hin hT64 (by
        have : nOf d i * d.T ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul hnN hT22
        exact lt_of_le_of_lt this (by norm_num))
    have hlT : lenOf d (nOf d i) = d.T := by simp [lenOf, hfix]
    have hnchb : nch d (nOf d i) ≤ 2 ^ 16 := by
      simp only [nch, BT]; rw [hlT]; omega
    have hboh : (r3.item * rB.item).toInt = (cbase d (nOf d i) : Int) := by
      rw [hcbv]
      exact IntV.mul_nat hin hNT64 (by
        have : nOf d i * nch d (nOf d i) ≤ 8192 * 2 ^ 16 := Nat.mul_le_mul hnN hnchb
        exact lt_of_le_of_lt this (by norm_num))
    have hTl : (IntV.ofNat d.T : I32).toInt = (lenOf d (nOf d i) : Int) := by
      rw [hTi]; simp [lenOf, hfix]
    have hT64l : rT.item.toInt = (lenOf d (nOf d i) : Int) := by rw [hT64]; simp [lenOf, hfix]
    by_cases hvf : d.vf = true
    · simp only [hvf, ite_true]
      exact safe_body_vf hd hP hpre hvf (hT2 hvf) hn hhv hjv hiv hinh' hih hbos hboh hT64l
        (runLoop_fix hd hn hhv hfix hbos hih hTi hNT)
    · have hvf' : d.vf = false := by simpa using hvf
      simp only [hvf, Bool.false_eq_true, ite_false]
      exact safe_body_nonvf hd hP hpre hvf' (hT1 hvf') hn hhv hjv hiv hinh' hih hbos hboh hT64l
        (runLoop_fix hd hn hhv hfix hbos hih hTi hNT)

end Claims.ChunkDeltaH
