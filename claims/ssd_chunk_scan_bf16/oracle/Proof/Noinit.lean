import Proof.SetupDefs
import Proof.PtrLemmas
import Proof.LoadLemmas

/-!
# The set-up of the variant without initial states

Walk of `setupN` (`chunk_scan_kernel_noinit` up to the core, lines 172–340): program coordinates,
the pointers (all `int64` offsets), `chunk_size_limit`, `seq_idx_prev`, `dA_cs_m`, `seq_idx_m` and
the scale, then `core_safe`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- **The variant without initial states**: walk of `setupN`, ending in `core_safe`. -/
theorem noinit_tile {E : Env} {d : Inst} {init : Mem} (hd : d.ok) (hpre : Pre d init) (dev : Dev E)
    (hN : d.hasInit = false) {i k pm pn h : Nat} (hi : i < gx d * d.nmeta * d.H)
    (hkd : kOf d i = k) (hpmd : pmOf d i = pm) (hpnd : pnOf d i = pn) (hhd : hOf d i = h) :
    Safe E (R d) (Wt d k pm pn h) (fun _ m => Qt d init k pm pn h m) init
      ((model d).body ((launch d).ctx i)) := by
  rw [body_eq_noinit d hN]
  obtain ⟨hk, hh, hpm, hpn, ha⟩ := decode_lt hd hi
  rw [hkd] at hk; rw [hhd] at hh; rw [hpmd] at hpm; rw [hpnd] at hpn
  have S : Std d init E k pm pn h := ⟨hd, hpre, dev, hk, hpm, hpn, hh⟩
  have pid1 : tl.program_id ((launch d).ctx i) 1 = IntV.ofNat k := by rw [pid1_eq, hkd]
  have pid2 : tl.program_id ((launch d).ctx i) 2 = IntV.ofNat h := by rw [pid2_eq, hhd]
  obtain ⟨hvcb, hvx, hvdt, hvdA, hvseq, hvC, hvst, hvD, hvin, hvci, hvco, hvout, hmeta, hcb0⟩ := hpre
  unfold setupN
  have hk31 : k < 2 ^ 31 := by have := hd.nmeta_lt; omega
  have hh31 : h < 2 ^ 31 := by have := hd.H_lt; omega
  have hcs := cs_pos hd
  have hcs14 := hd.cs_le
  have hcs31 : d.cs < 2 ^ 31 := by have := hd.cs_le; omega
  have hnpn := npn_pos hd
  have hlc : lc d (k) = k := lc_noinit hN _
  have hlo : lo d (k) = 0 := lo_noinit hN _
  -- `pid_c = pid_bc // 1`
  have hpbc : (IntV.cast (tl.program_id ((launch d).ctx i) 1) : I64).toInt = (k : Int) := by
    rw [pid1]; exact cast_val (IntV.I32.toInt_ofNat hk31)
  have hone64 : (IntV.ofNat 1 : I64).toInt = 1 := by simp
  refine safe_idiv1 (by rw [hpbc]; positivity) (by rw [hone64]; norm_num) (fun r1 h1 => ?_)
  have hpc : r1.item.toInt = (k : Int) := by rw [h1, hpbc, hone64]; simp
  -- `num_pid_n = (hdim + 63) // 64`
  have hP31 := P_small hd
  have hnum : (IntV.ofNat d.P + (IntV.ofNat (BN - 1) : I32)).toInt = ((d.P + (BN - 1) : Nat) : Int) :=
    IntV.add_nat (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_ofNat (by norm_num [BN]))
      (by simp only [BN]; omega)
  have hcsv : (IntV.ofNat d.cs : I32).toInt = (d.cs : Int) := IntV.I32.toInt_ofNat hcs31
  have hbn : (IntV.ofNat BN : I32).toInt = (BN : Int) := IntV.I32.toInt_ofNat (by norm_num [BN])
  refine safe_idiv1 (by rw [hnum]; positivity) (by rw [hbn]; norm_num [BN]) (fun r2 h2 => ?_)
  have hnp : r2.item.toInt = (npn d : Int) := by
    rw [h2, hnum, hbn]; unfold npn; simp only [BN]; omega
  -- `pid_m`, `pid_n`
  have hp0 : (tl.program_id ((launch d).ctx i) 0).toInt = ((i % gx d : Nat) : Int) := by
    rw [pid0_eq]; exact IntV.I32.toInt_ofNat (by have := hd.grid_lt; have := Nat.mod_le i (gx d); omega)
  refine safe_idiv1 (by rw [hp0]; positivity) (by rw [hnp]; exact_mod_cast hnpn) (fun r3 h3 => ?_)
  have hpmv : r3.item.toInt = (pm : Int) := by rw [h3, hp0, hnp, ← hpmd]; unfold pmOf; exact (Int.natCast_ediv _ _).symm
  refine safe_irem1 (by rw [hp0]; positivity) (by rw [hnp]; exact_mod_cast hnpn) (fun r4 h4 => ?_)
  have hpnv : r4.item.toInt = (pn : Int) := by rw [h4, hp0, hnp, ← hpnd]; unfold pnOf; exact (Int.natCast_emod _ _).symm
  -- the group of the head, twice
  have hph : (tl.program_id ((launch d).ctx i) 2).toInt = (h : Int) := by
    rw [pid2]; exact IntV.I32.toInt_ofNat hh31
  have hrat : ratio d < 2 ^ 31 := by
    have := hd.H_lt; unfold ratio; have := Nat.div_le_self d.H d.G; omega
  have hratv : (IntV.ofNat (ratio d) : I32).toInt = (ratio d : Int) := IntV.I32.toInt_ofNat hrat
  have hratp := ratio_pos hd
  refine safe_idiv1 (by rw [hph]; positivity) (by rw [hratv]; exact_mod_cast hratp) (fun r5 h5 => ?_)
  have hg5 : r5.item.toInt = (grp d (h) : Int) := by
    rw [h5, hph, hratv]; exact (Int.natCast_ediv _ _).symm
  refine safe_idiv1 (by rw [hph]; positivity) (by rw [hratv]; exact_mod_cast hratp) (fun r6 h6 => ?_)
  have hg6 : r6.item.toInt = (grp d (h) : Int) := by
    rw [h6, hph, hratv]; exact (Int.natCast_ediv _ _).symm
  -- `pid_b = pid_bc - pid_c · 1 = 0`
  have hone64n : (IntV.ofNat 1 : I64).toInt = ((1 : Nat) : Int) := I64_ofNat (by norm_num)
  have hpb : (IntV.cast (tl.program_id ((launch d).ctx i) 1) - r1.item * (IntV.ofNat 1 : I64)).toInt
      = 0 := by
    have h := mul64 hpc hone64n (by omega : k * 1 < 2 ^ 62)
    have := IntV.toInt_sub hpbc h (by simp only [IntV.SFits]; push_cast; constructor <;> omega)
    rw [this]; push_cast; ring
  -- `seq_idx_prev`
  have hT31 : d.T < 2 ^ 31 := by have := hd.T_ctx; omega
  have hkcs : k * d.cs < d.T := noinit_start_lt hd hN hk
  have hsip : d.seqIdx.ptr 0 + ((IntV.cast (tl.program_id ((launch d).ctx i) 1) - r1.item * IntV.ofNat 1) *
      (IntV.ofNat d.T : I32).cast + r1.item * (IntV.ofNat d.cs : I32).cast *
        (IntV.ofNat 1 : I32).cast) = d.seqIdx.ptr ((k * d.cs * 1 : Nat) : Int) :=
    ptr_c d.seqIdx hpb hpc hT31 hcs31 (by norm_num) (by omega) (by omega)
  have hneg : (-(IntV.ofNat 1 : I32)).toInt = -1 := by decide
  refine safe_load_scalar d.seqIdx (idx := k * d.cs - 1) ?_ ?_ hvseq
    (bytes_sub_R hd.disj.2.2.2.2.1) (fun v hv1 hv2 => ?_)
  · intro hmk
    rw [IntV.le_eq, IntV.toInt_ofInt_one (by norm_num), hpc, decide_eq_true_eq] at hmk
    have hk1 : 1 ≤ k * d.cs := by
      have : 1 ≤ k := by exact_mod_cast hmk
      exact Nat.mul_pos this hcs
    show (_ : Ptr .i32).offset _ = _
    rw [hsip, Buf.ptr_offset, hneg]
    congr 1
    push_cast [Nat.cast_sub hk1]; ring
  · intro hmk
    have := hd.seq_len
    omega
  have hprev : v.item = (IntV.ofNat (prevReq d (k)) : I32) := by
    by_cases hle : IntV.le (IntV.ofInt 1 : I64) r1.item = true
    · have h1 := hv1 hle
      have hle' := hle
      rw [IntV.le_eq, IntV.toInt_ofInt_one (by norm_num), hpc, decide_eq_true_eq] at hle'
      have hk1 : 1 ≤ k := by exact_mod_cast hle'
      have hk2 : 1 ≤ k * d.cs := Nat.mul_pos hk1 hcs
      have := hmeta.1 (k * d.cs - 1) (by omega)
      rw [h1] at this
      rw [Option.some.inj this]
      unfold prevReq; rw [ite_eq_left hk1]
    · have h2 := hv2 (by simpa using hle)
      have hk0 : k = 0 := by
        rw [IntV.le_eq, IntV.toInt_ofInt_one (by norm_num), hpc] at hle
        simp at hle; omega
      rw [h2]; unfold prevReq; rw [hk0]; rfl
  -- the rows of the tile
  have hco0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.toInt_ofInt_zero
  have hpmB : BM * pm < d.cs := by unfold nmt at hpm; simp only [BM] at *; omega
  have hbm : (r3.item * (IntV.ofNat BM : I32) + IntV.ofInt 0).toInt = ((pm * BM + 0 : Nat) : Int) := by
    have h1 : (r3.item * (IntV.ofNat BM : I32)).toInt = ((pm * BM : Nat) : Int) :=
      IntV.mul_nat hpmv (IntV.I32.toInt_ofNat (by norm_num [BM])) (by simp only [BM] at *; omega)
    exact IntV.add_nat h1 (by simpa using hco0) (by simp only [BM] at *; omega)
  have hom : ∀ ii : Fin BM, ((r3.item * (IntV.ofNat BM : I32) + IntV.ofInt 0 + tl.arange 0 BM :
      Tensor [BM] I32).get1 ii).toInt = (rowOf d k pm ii : Int) := fun ii => by
    have := IntV.I32.toInt_sadd_arange (n := BM) hbm (by simp only [BM] at *; omega) ii
    rw [this]; unfold rowOf; rw [hlo]; push_cast; ring
  -- the pointers and the `dA_cumsum` of the tile's rows
  have hHnc : d.H * (nc d * d.cs) < 2 ^ 31 := hd.dt_i32
  have hnccs : nc d * d.cs < 2 ^ 31 := by
    have : nc d * d.cs ≤ d.H * (nc d * d.cs) := Nat.le_mul_of_pos_left _ hd.H_pos
    omega
  have hphv : (tl.program_id ((launch d).ctx i) 2).toInt = (h : Int) := by
    rw [pid2]; exact IntV.I32.toInt_ofNat hh31
  have hh_nc : h * (nc d * d.cs) < 2 ^ 31 := by
    have : h * (nc d * d.cs) ≤ d.H * (nc d * d.cs) := Nat.mul_le_mul_right _ hh.le
    omega
  have hdAp : d.dA.ptr 0 + ((IntV.cast (tl.program_id ((launch d).ctx i) 1) - r1.item * IntV.ofNat 1) *
      (IntV.ofNat (d.H * (nc d * d.cs)) : I32).cast + r1.item * (IntV.ofNat d.cs : I32).cast +
        IntV.cast (tl.program_id ((launch d).ctx i) 2 * (IntV.ofNat (nc d * d.cs) : I32))) =
      d.dA.ptr (dtIdx d h k 0 : Int) := by
    have := ptr_a d.dA hpb hpc hphv hHnc hcs31 hnccs hh_nc (by have := noinit_start_lt hd hN hk; omega)
    rw [this]; congr 1; unfold dtIdx; push_cast; ring
  have hrowlt : ∀ l : Fin BM, rowOf d k pm l < 2 ^ 31 := fun l => by
    have := rowOf_lt hd hk hpm l.isLt; have := hd.cs_le; omega
  refine safe_load_vec d.dA (g := fun l : Fin BM => dtIdx d h k 0 + rowOf d k pm l)
    (en := fun l : Fin BM => decide (rowOf d k pm l < d.cs)) ?_ ?_ ?_ hvdA
    (bytes_sub_R hd.disj.2.2.2.1) (fun dAm hd1 hd2 => ?_)
  · intro l
    rw [Tensor.get1_ltS, IntV.lt_eq, hom l, hcsv]
    simp
  · intro l hl
    exact ptr_vec_lane d.dA hdAp hom hrowlt l
  · intro l hl
    have hl' : rowOf d k pm l < d.cs := by simpa using hl
    have := dtIdx_lt hd hk hh (u := rowOf d k pm l) hl'
    rw [hlc] at this
    have e : dtIdx d h k 0 + rowOf d k pm l = dtIdx d h k (rowOf d k pm l) := by unfold dtIdx; ring
    show dtIdx d h k 0 + rowOf d k pm l < d.dA.len
    rw [e]; exact lt_of_lt_of_le this hd.dA_len
  have hdAm : ∀ ii : Fin BM, RowOk d k pm ii → ∃ v : F32,
      init.read .f32 (d.dA.ptr (dtIdx d h (lc d k) (rowOf d k pm ii) : Int)).addr = some v ∧
        dAm.get1 ii = v := by
    intro ii hri
    have hl : rowOf d k pm ii < d.cs := by have := lend_le hd hk; unfold RowOk at hri; omega
    have e : dtIdx d h k 0 + rowOf d k pm ii = dtIdx d h k (rowOf d k pm ii) := by unfold dtIdx; ring
    refine ⟨dAm.get1 ii, ?_, rfl⟩
    have := hd1 ii (by simpa using hl)
    rw [hlc, ← e]; exact this
  -- `chunk_size_limit` and `seq_idx_m`
  have hlimv : ((tl.iminimum (Tensor.scalar (IntV.ofNat d.cs : I32).cast)
      (Tensor.scalar ((IntV.ofNat d.T : I32).cast - r1.item * (IntV.ofNat d.cs : I32).cast))).item
      ).toInt = (lend d k : Int) := by
    have h1 : (r1.item * (IntV.ofNat d.cs : I32).cast).toInt = ((k * d.cs : Nat) : Int) :=
      mul64 hpc (cast_ofNat hcs31) (by have := hd.T_ctx; omega)
    have h2 : ((IntV.ofNat d.T : I32).cast - r1.item * (IntV.ofNat d.cs : I32).cast).toInt =
        ((d.T - k * d.cs : Nat) : Int) :=
      IntV.sub_nat (cast_ofNat hT31) h1 (by omega) (by omega)
    have e : ∀ a b : I64, (tl.iminimum (Tensor.scalar a) (Tensor.scalar b)).item = IntV.min a b :=
      fun _ _ => rfl
    rw [e, IntV.toInt_min, cast_ofNat hcs31, h2, lend_noinit hN]
    push_cast; rfl
  refine safe_load_vec d.seqIdx (g := fun l : Fin BM => k * d.cs * 1 + rowOf d k pm l)
    (en := fun l : Fin BM => decide (rowOf d k pm l < lend d k)) ?_ ?_ ?_ hvseq
    (bytes_sub_R hd.disj.2.2.2.2.1) (fun sm hs1 hs2 => ?_)
  · intro l
    rw [Tensor.get1_ltS, IntV.lt_eq, Tensor.get1_castInt, hlimv, cast_val (hom l)]
    simp
  · intro l hl
    exact ptr_vec_lane d.seqIdx hsip hom hrowlt l
  · intro l hl
    have hl' : rowOf d k pm l < lend d k := by simpa using hl
    have := lend_le hd hk
    rw [hlc] at this
    have := hd.seq_len
    show k * d.cs * 1 + rowOf d k pm l < d.seqIdx.len
    omega
  have hsm : ∀ ii : Fin BM, RowOk d k pm ii →
      sm.get1 ii = (IntV.ofNat (reqOf d (tokOf d k pm ii)) : I32) := by
    intro ii hri
    have h1 := hs1 ii (decide_eq_true hri)
    have := lend_le hd hk
    rw [hlc] at this
    have hT := (tok_of_row hd hk ⟨by unfold rowOf; omega, hri⟩).1
    have h2 := hmeta.1 (tokOf d k pm ii) hT
    have e : k * d.cs * 1 + rowOf d k pm ii = tokOf d k pm ii := by unfold tokOf; rw [hlc]; ring
    rw [e] at h1
    rw [h1] at h2
    exact Option.some.inj h2
  refine safe_math (fun i => dev.expTotal _) (fun ex hex => ?_)
  -- sizes used by the pointer evaluations
  have hpnB : BN * pn < d.P + 64 := by unfold npn at hpn; simp only [BN] at *; omega
  have hnc1 := nc_pos hd
  have hTcT : d.T * d.cT < 2 ^ 31 := hd.c_arg
  have hT1 := T_pos hd
  have hcT31 : d.cT < 2 ^ 31 := by
    have : d.cT ≤ d.T * d.cT := Nat.le_mul_of_pos_left _ hT1
    omega
  have hgN : grp d h * d.N < 2 ^ 31 := by
    have h1 : (grp d h) * d.N ≤ d.G * d.N := Nat.mul_le_mul_right _ (grp_lt hd hh).le
    have := hd.cT_ge; omega
  have hN31 : d.N < 2 ^ 31 := by have := hd.N_le; have := hd.bsd_le; omega
  have hkcs' : k * d.cs ≤ d.T := hkcs.le
  have hkcsT : k * d.cs * d.cT < 2 ^ 31 := by
    have : k * d.cs * d.cT ≤ d.T * d.cT := Nat.mul_le_mul_right _ hkcs'
    omega
  have hPN31 : PN d < 2 ^ 31 := by
    unfold PN
    have h1 : d.P * d.N ≤ nc d * (d.H * (d.P * d.N)) := by
      calc d.P * d.N = 1 * (d.P * d.N) := (Nat.one_mul _).symm
        _ ≤ (nc d * d.H) * (d.P * d.N) := Nat.mul_le_mul_right _ (Nat.mul_pos hnc1 hd.H_pos)
        _ = _ := by ring
    have := hd.st_i32; unfold PN at this; omega
  have hHPN : d.H * PN d < 2 ^ 31 := by
    have : d.H * PN d ≤ nc d * (d.H * PN d) := Nat.le_mul_of_pos_left _ hnc1
    have := hd.st_i32; omega
  have hkHPN : k * (d.H * PN d) < 2 ^ 31 := by
    have h1 : k * (d.H * PN d) ≤ nc d * (d.H * PN d) := Nat.mul_le_mul_right _ (by
      have := lc_lt hd hk; rw [hlc] at this; omega)
    have := hd.st_i32; omega
  have hhPN : h * PN d < 2 ^ 31 := by
    have h1 : h * PN d ≤ d.H * PN d := Nat.mul_le_mul_right _ hh.le
    omega
  refine core_safe S (u := false) (r := 0) (by norm_num) (fun h => absurd h (by simp))
    ?hC ?hS (by simp [hN]) (fun h => absurd h (by simp)) rfl rfl ?hcb ?hx ?hdt ?hdA ?hout rfl
    ?hph ?hpm ?hpn ?hco hom ?hon ?hwiden ?hcsw ?hlim ?hkm hdAm ?hsc
  case hC =>
    rw [hlc]
    exact (ptr_b d.C hpb hpc hg6 hTcT hcs31 hcT31 hN31 hgN (by omega) hkcsT)
  case hS =>
    simp only [stBuf, stBase, Bool.false_eq_true, ite_false]
    rw [hlc]
    exact ptr_s d.states hpb hpc hphv (by have := hd.st_i32; omega) hHPN hPN31 hhPN hkHPN
  case hcb =>
    have hcbA : nc d * (d.G * (d.cs * d.cs)) < 2 ^ 31 := hd.cb_i32
    have hGcs : d.G * (d.cs * d.cs) < 2 ^ 31 := by
      have : d.G * (d.cs * d.cs) ≤ nc d * (d.G * (d.cs * d.cs)) := Nat.le_mul_of_pos_left _ hnc1
      omega
    have hcs2 : d.cs * d.cs < 2 ^ 31 := by
      have : d.cs * d.cs ≤ d.G * (d.cs * d.cs) := Nat.le_mul_of_pos_left _ hd.G_pos
      omega
    have hgc : grp d h * (d.cs * d.cs) < 2 ^ 31 := by
      have : grp d h * (d.cs * d.cs) ≤ d.G * (d.cs * d.cs) :=
        Nat.mul_le_mul_right _ (grp_lt hd hh).le
      omega
    have hkc : k * (d.G * (d.cs * d.cs)) < 2 ^ 31 := by
      have : k * (d.G * (d.cs * d.cs)) ≤ nc d * (d.G * (d.cs * d.cs)) :=
        Nat.mul_le_mul_right _ (by have := lc_lt hd hk; rw [hlc] at this; omega)
      omega
    rw [ptr_a d.cb hpb hpc hg5 hcbA hGcs hcs2 hgc hkc]
    congr 1; unfold cbIdx; rw [hlc]; push_cast; ring
  case hx =>
    have hxT31 : d.T * d.xT < 2 ^ 31 := hd.x_arg
    have hxT : d.xT < 2 ^ 31 := by
      have : d.xT ≤ d.T * d.xT := Nat.le_mul_of_pos_left _ hT1
      omega
    have hhP : h * d.P < 2 ^ 31 := by
      have := hd.xT_ge; have : h * d.P ≤ d.H * d.P := Nat.mul_le_mul_right _ hh.le
      have := hd.x_off
      have : d.xT ≤ (2 * d.cs + 64) * d.xT := Nat.le_mul_of_pos_left _ (by omega)
      omega
    have hkx : k * d.cs * d.xT < 2 ^ 31 := by
      have : k * d.cs * d.xT ≤ d.T * d.xT := Nat.mul_le_mul_right _ hkcs'
      omega
    rw [ptr_b d.x hpb hpc hphv hxT31 hcs31 hxT (P_lt31 hd) hhP (by omega) hkx]
    congr 1; unfold xIdx; rw [hlc]; push_cast; ring
  case hdt =>
    have hkc : k * d.cs < 2 ^ 31 := by omega
    rw [ptr_a d.dt hpb hpc hphv hHnc hcs31 hnccs hh_nc hkc]
    congr 1; unfold dtIdx; rw [hlc]; push_cast; ring
  case hdA =>
    have hkc : k * d.cs < 2 ^ 31 := by omega
    rw [ptr_a d.dA hpb hpc hphv hHnc hcs31 hnccs hh_nc hkc]
    congr 1; unfold dtIdx; rw [hlc]; push_cast; ring
  case hout =>
    have hout31 : d.T * (d.H * d.P) < 2 ^ 31 := hd.out_arg
    have hHP : d.H * d.P < 2 ^ 31 := by
      have : d.H * d.P ≤ d.T * (d.H * d.P) := Nat.le_mul_of_pos_left _ hT1
      omega
    have hhP : h * d.P < 2 ^ 31 := by
      have : h * d.P ≤ d.H * d.P := Nat.mul_le_mul_right _ hh.le
      omega
    have hko : k * d.cs * (d.H * d.P) < 2 ^ 31 := by
      have : k * d.cs * (d.H * d.P) ≤ d.T * (d.H * d.P) := Nat.mul_le_mul_right _ hkcs'
      omega
    rw [ptr_b d.out hpb hpc hphv hout31 hcs31 hHP (P_lt31 hd) hhP (by omega) hko]
    congr 1; rw [hlc]
  case hph => exact hphv
  case hpm => exact hpmv
  case hpn => exact hpnv
  case hco => rw [hlo, hco0]; rfl
  case hon =>
    have hbn' : (r4.item * (IntV.ofNat BN : I32)).toInt = ((pn * BN : Nat) : Int) :=
      IntV.mul_nat hpnv (IntV.I32.toInt_ofNat (by norm_num [BN])) (by simp only [BN] at *; omega)
    intro jj
    have := IntV.I32.toInt_sadd_arange (n := BN) hbn' (by simp only [BN] at *; omega) jj
    rw [this]; unfold colOf; push_cast; ring
  case hwiden =>
    intro n x l
    rw [Tensor.get1_castInt]; exact IntV.toInt_cast _ (by norm_num)
  case hcsw => exact cast_ofNat hcs31
  case hlim => exact hlimv
  case hkm =>
    have h1 : (r3.item + IntV.ofInt 1).toInt = ((pm + 1 : Nat) : Int) :=
      IntV.add_nat hpmv (by simpa using IntV.toInt_ofInt_one (w := 32) (by norm_num))
        (by simp only [BM] at *; omega)
    have h2 := IntV.mul_nat h1 (IntV.I32.toInt_ofNat (n := BM) (by norm_num [BM]))
      (show (pm + 1) * BM < 2 ^ (32 - 1) by simp only [BM] at *; omega)
    rw [cast_val h2]; push_cast; ring
  case hsc =>
    intro hV ii hri
    have hrow : lo d k ≤ rowOf d k pm ii ∧ rowOf d k pm ii < lend d k :=
      ⟨by unfold rowOf; omega, hri⟩
    obtain ⟨hT, hchk, hmod, -⟩ := tok_of_row hd hk hrow
    have hT' : tokOf d k pm ii < d.T := hT
    have hchk' : chk d (tokOf d k pm ii) = k := by
      have : chk d (tokOf d k pm ii) = lc d k := hchk
      rwa [hlc] at this
    have hmod' : tokOf d k pm ii % d.cs = rowOf d k pm ii := hmod
    obtain ⟨a, hra, hav⟩ := hdAm ii hri
    have hcond : (sm.eq (bc v.item)).get1 ii =
        decide (reqOf d (tokOf d k pm ii) = prevReq d k) := by
      show IntV.eq (sm.get1 ii) v.item = _
      rw [IntV.eq_eq, hsm ii hri, hprev, IntV.I32.toInt_ofNat (reqOf_lt hd _),
        IntV.I32.toInt_ofNat (by unfold prevReq; split_ifs; exact reqOf_lt hd _; norm_num)]
      simp
    refine scale_noinit (P := E.P) dev.expAcc hd hV hN hT' hh (p := 0) hd.P_pos
      (a := a) (e0 := ex.get1 ii) ?_ ?_ ?_
    · rw [hchk', hmod']; rw [hlc] at hra; exact hra
    · have := hex (ii, ()); rw [← hav]; exact this
    · show (if (sm.eq (bc v.item)).get (ii, ()) = true then ex.get (ii, ()) else
        (bc zf32 : Tensor [BM] F32).get (ii, ())) = _
      have hc2 : (sm.eq (bc v.item)).get (ii, ()) = decide (reqOf d (tokOf d k pm ii) = prevReq d k) :=
        hcond
      rw [hc2]
      have hf := fromInit_noinit hd hT'
      rw [hchk'] at hf
      by_cases hq : reqOf d (tokOf d k pm ii) = prevReq d k
      · have : ¬ fromInit d (tokOf d k pm ii) = true := fun h' => (hf.mp h') hq.symm
        simp [hq, this]
        rfl
      · have : fromInit d (tokOf d k pm ii) = true := hf.mpr (fun h' => hq h'.symm)
        simp [hq, this]
        rfl

end Claims.SsdChunkScan
