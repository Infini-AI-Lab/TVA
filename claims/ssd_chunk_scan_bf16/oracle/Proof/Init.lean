import Proof.SetupInitDefs
import Proof.PtrLemmas
import Proof.LoadLemmas

/-!
# The set-up of the variant with initial states

Walk of `setupI`, `tail1I` and `tail2I` (`chunk_scan_kernel` lines 172–344): the metadata loads, the
pointers, the two runtime `if`s, and the scale, then `core_safe`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- **The scale and the core** (`tail2I`): the decay boundary and the limit decided, the scale of
the entering state is `exp(dA_cs_m − boundary)`, and `core_safe` finishes. -/
theorem tail2I_safe {E : Env} {d : Inst} {init : Mem} {k pm pn h : Nat}
    (S : Std d init E k pm pn h) {c_off pid_h pid_m pid_n : I32} {prev_ptr : Ptr .f32}
    {hdimS dstS : I32} {cb_ptr : Ptr .f32} {x_ptr : Ptr .bf16} {dt_ptr dA_ptr : Ptr .f32}
    {C_ptr out_ptr : Ptr .bf16} {offs_m : Tensor [BM] I32} {offs_n : Tensor [BN] I32}
    {dA_cs_m : Tensor [BM] F32} {limit : I32} {boundary : F32} {u : Bool} {r : Nat}
    (hN : d.hasInit = true)
    (hui : u = true → d.hasInit = true ∧ r < d.nreq)
    (hC : C_ptr = d.C.ptr ((lc d k * d.cs * d.cT + grp d h * d.N : Nat) : Int))
    (hS : prev_ptr = (stBuf d u).ptr (stBase d u (lc d k) r h : Nat))
    (hu : ∀ ii, RowOk d k pm ii → (d.hasInit && fromInit d (tokOf d k pm ii)) = u)
    (hr : u = true → ∀ ii, RowOk d k pm ii → reqOf d (tokOf d k pm ii) = r)
    (hhd : hdimS = IntV.ofNat d.N) (hds : dstS = IntV.ofNat 1)
    (hcb : cb_ptr = d.cb.ptr (cbIdx d (lc d k) (grp d h) 0 0 : Int))
    (hx : x_ptr = d.x.ptr (xIdx d (lc d k * d.cs) h 0 : Int))
    (hdt : dt_ptr = d.dt.ptr (dtIdx d h (lc d k) 0 : Int))
    (hdA : dA_ptr = d.dA.ptr (dtIdx d h (lc d k) 0 : Int))
    (hout : out_ptr = d.out.ptr ((lc d k * d.cs * (d.H * d.P) + h * d.P : Nat) : Int))
    (hph : pid_h.toInt = (h : Int)) (hpm : pid_m.toInt = (pm : Int))
    (hpn : pid_n.toInt = (pn : Int)) (hco : c_off.toInt = (lo d k : Int))
    (hom : ∀ ii : Fin BM, (offs_m.get1 ii).toInt = (rowOf d k pm ii : Int))
    (hon : ∀ jj : Fin BN, (offs_n.get1 jj).toInt = (colOf pn jj : Int))
    (hlim : limit.toInt = (lend d k : Int))
    (hdAm : ∀ ii : Fin BM, RowOk d k pm ii → ∃ v : F32,
      init.read .f32 (d.dA.ptr (dtIdx d h (lc d k) (rowOf d k pm ii) : Int)).addr = some v ∧
        dA_cs_m.get1 ii = v)
    (hbd : ((0 < lo d k ∧ ∃ b : F32, init.read .f32
        (d.dA.ptr (dtIdx d h (lc d k) (lo d k - 1) : Int)).addr = some b ∧ boundary = b) ∨
      (lo d k = 0 ∧ boundary = zf32))) :
    Safe E (R d) (Wt d k pm pn h) (fun _ m => Qt d init k pm pn h m) init
      (tail2I d c_off pid_h pid_m pid_n prev_ptr hdimS dstS cb_ptr x_ptr dt_ptr dA_ptr C_ptr out_ptr
        offs_m offs_n dA_cs_m limit boundary) := by
  obtain ⟨hd, hpre, dev, hk, hpm', hpn', hh⟩ := S
  have S : Std d init E k pm pn h := ⟨hd, hpre, dev, hk, hpm', hpn', hh⟩
  unfold tail2I
  refine safe_sub (fun xt hxt => ?_)
  refine safe_math (fun i => dev.expTotal _) (fun ex hex => ?_)
  have hcs31 : d.cs < 2 ^ 31 := by have := hd.cs_le; omega
  have hpmB : BM * pm < d.cs := by unfold nmt at hpm'; simp only [BM] at *; have := cs_pos hd; omega
  refine core_safe S (w := 32) (u := u) (r := r) (by norm_num) hui hC hS hu hr hhd hds hcb hx hdt hdA
    hout rfl hph hpm hpn hco hom hon (fun x i => rfl) (IntV.I32.toInt_ofNat hcs31) hlim ?hkm hdAm ?hsc
  case hkm =>
    have h1 : (pid_m + IntV.ofInt 1).toInt = ((pm + 1 : Nat) : Int) :=
      IntV.add_nat hpm (by simpa using IntV.toInt_ofInt_one (w := 32) (by norm_num))
        (by simp only [BM] at *; omega)
    rw [IntV.mul_nat h1 (IntV.I32.toInt_ofNat (n := BM) (by norm_num [BM]))
      (show (pm + 1) * BM < 2 ^ (32 - 1) by have := hd.cs_le; simp only [BM] at *; omega),
      Nat.mul_comm]
  case hsc =>
    intro hV ii hri
    have hrow : lo d k ≤ rowOf d k pm ii ∧ rowOf d k pm ii < lend d k :=
      ⟨by unfold rowOf; omega, hri⟩
    obtain ⟨hT, hchk, hmod, hlOff⟩ := tok_of_row hd hk hrow
    replace hlOff := hlOff hN
    have hT' : tokOf d k pm ii < d.T := hT
    have hchk' : chk d (tokOf d k pm ii) = lc d k := hchk
    have hmod' : tokOf d k pm ii % d.cs = rowOf d k pm ii := hmod
    have hlOff' : lOff d (tokOf d k pm ii) = lo d k := hlOff
    obtain ⟨a, hra, hav⟩ := hdAm ii hri
    refine scale_init (P := E.P) dev.expAcc hd hV hN hT' hh (p := 0) hd.P_pos (a := a)
      (b := boundary) (xt := xt.get1 ii) ?_ ?_ ?_ ?_
    · rw [hchk', hmod']; exact hra
    · rw [hlOff', hchk']
      rcases hbd with ⟨h0, b, hrb, hbb⟩ | ⟨h0, hbb⟩
      · exact Or.inl ⟨h0, by rw [hbb]; exact hrb⟩
      · exact Or.inr ⟨h0, hbb⟩
    · have h1 := hxt (ii, ())
      have e1 : dA_cs_m.get (ii, ()) = a := hav
      rw [e1] at h1
      exact h1
    · exact hex (ii, ())

/-- **Lines 261–310** (`tail1I`): `offs_n`, `dA_cs_m`, `c_idx_n`, and the second runtime `if`
(the limit cut at the next logical chunk, the decay boundary), then `tail2I_safe`. -/
theorem tail1I_safe {E : Env} {d : Inst} {init : Mem} {k pm pn h : Nat}
    (S : Std d init E k pm pn h) {pid_c : I64} {c_idx c_off pid_h pid_m pid_n limit : I32}
    {prev_ptr : Ptr .f32} {hdimS dstS : I32} {cb_ptr : Ptr .f32} {x_ptr : Ptr .bf16}
    {dt_ptr dA_ptr : Ptr .f32} {C_ptr out_ptr : Ptr .bf16} {offs_m : Tensor [BM] I32}
    {u : Bool} {r : Nat} (hN : d.hasInit = true)
    (hpc : pid_c.toInt = (k : Int)) (hci : c_idx = IntV.ofNat (lc d k))
    (hco' : c_off = IntV.ofNat (lo d k)) (hlimm : limit.toInt = (lm d k : Int))
    (hui : u = true → d.hasInit = true ∧ r < d.nreq)
    (hC : C_ptr = d.C.ptr ((lc d k * d.cs * d.cT + grp d h * d.N : Nat) : Int))
    (hS : prev_ptr = (stBuf d u).ptr (stBase d u (lc d k) r h : Nat))
    (hu : ∀ ii, RowOk d k pm ii → (d.hasInit && fromInit d (tokOf d k pm ii)) = u)
    (hr : u = true → ∀ ii, RowOk d k pm ii → reqOf d (tokOf d k pm ii) = r)
    (hhd : hdimS = IntV.ofNat d.N) (hds : dstS = IntV.ofNat 1)
    (hcb : cb_ptr = d.cb.ptr (cbIdx d (lc d k) (grp d h) 0 0 : Int))
    (hx : x_ptr = d.x.ptr (xIdx d (lc d k * d.cs) h 0 : Int))
    (hdt : dt_ptr = d.dt.ptr (dtIdx d h (lc d k) 0 : Int))
    (hdA : dA_ptr = d.dA.ptr (dtIdx d h (lc d k) 0 : Int))
    (hout : out_ptr = d.out.ptr ((lc d k * d.cs * (d.H * d.P) + h * d.P : Nat) : Int))
    (hph : pid_h.toInt = (h : Int)) (hpm : pid_m.toInt = (pm : Int))
    (hpn : pid_n.toInt = (pn : Int))
    (hom : ∀ ii : Fin BM, (offs_m.get1 ii).toInt = (rowOf d k pm ii : Int)) :
    Safe E (R d) (Wt d k pm pn h) (fun _ m => Qt d init k pm pn h m) init
      (tail1I d pid_c c_idx c_off pid_h pid_m pid_n limit prev_ptr hdimS dstS cb_ptr x_ptr dt_ptr
        dA_ptr C_ptr out_ptr offs_m) := by
  have S' := S
  obtain ⟨hd, hpre, dev, hk, hpm', hpn', hh⟩ := S
  obtain ⟨hvcb, hvx, hvdt, hvdA, hvseq, hvC, hvst, hvD, hvin, hvci, hvco, hvout, hmeta, hcb0⟩ := hpre
  unfold tail1I
  have hcs31 : d.cs < 2 ^ 31 := by have := hd.cs_le; omega
  have hcsv : (IntV.ofNat d.cs : I32).toInt = (d.cs : Int) := IntV.I32.toInt_ofNat hcs31
  have hlc31 := meta_lt hd hk
  have hlend := lend_le hd hk
  have hlo := lo_lt hd hk
  -- the columns
  have hP256 := P_small hd
  have hpnB : BN * pn < d.P + 64 := by unfold npn at hpn'; simp only [BN] at *; omega
  have hbn : (pid_n * (IntV.ofNat BN : I32)).toInt = ((pn * BN : Nat) : Int) :=
    IntV.mul_nat hpn (IntV.I32.toInt_ofNat (by norm_num [BN])) (by simp only [BN] at *; omega)
  have hon : ∀ jj : Fin BN, ((pid_n * (IntV.ofNat BN : I32) + tl.arange 0 BN : Tensor [BN] I32).get1 jj).toInt
      = (colOf pn jj : Int) := fun jj => by
    have := IntV.I32.toInt_sadd_arange (n := BN) hbn (by simp only [BN] at *; omega) jj
    rw [this]; unfold colOf; push_cast; ring
  -- `dA_cs_m`
  have hrowlt : ∀ l : Fin BM, rowOf d k pm l < 2 ^ 31 := fun l => by
    have := rowOf_lt hd hk hpm' l.isLt; have := hd.cs_le; omega
  refine safe_load_vec d.dA (g := fun l : Fin BM => dtIdx d h (lc d k) 0 + rowOf d k pm l)
    (en := fun l : Fin BM => decide (rowOf d k pm l < d.cs)) ?_ ?_ ?_ hvdA
    (bytes_sub_R hd.disj.2.2.2.1) (fun dAm hd1 hd2 => ?_)
  · intro l
    rw [Tensor.get1_ltS, IntV.lt_eq, hom l, hcsv]
    simp
  · intro l hl
    exact ptr_vec_lane d.dA hdA hom hrowlt l
  · intro l hl
    have hl' : rowOf d k pm l < d.cs := by simpa using hl
    have := dtIdx_lt hd hk hh (u := rowOf d k pm l) hl'
    have e : dtIdx d h (lc d k) 0 + rowOf d k pm l = dtIdx d h (lc d k) (rowOf d k pm l) := by
      unfold dtIdx; ring
    show dtIdx d h (lc d k) 0 + rowOf d k pm l < d.dA.len
    rw [e]; exact lt_of_lt_of_le this hd.dA_len
  have hdAm : ∀ ii : Fin BM, RowOk d k pm ii → ∃ v : F32,
      init.read .f32 (d.dA.ptr (dtIdx d h (lc d k) (rowOf d k pm ii) : Int)).addr = some v ∧
        dAm.get1 ii = v := by
    intro ii hri
    have hl : rowOf d k pm ii < d.cs := by unfold RowOk at hri; omega
    have e : dtIdx d h (lc d k) 0 + rowOf d k pm ii = dtIdx d h (lc d k) (rowOf d k pm ii) := by
      unfold dtIdx; ring
    refine ⟨dAm.get1 ii, ?_, rfl⟩
    have := hd1 ii (by simpa using hl)
    rw [← e]; exact this
  -- `c_idx_n`
  have hone64 : (IntV.ofInt 1 : I64).toInt = ((1 : Nat) : Int) := by
    simpa using IntV.toInt_ofInt_one (w := 64) (by norm_num)
  have hk1 : (pid_c + (IntV.ofInt 1 : I64)).toInt = ((k + 1 : Nat) : Int) :=
    IntV.add_nat hpc hone64 (by have := hd.nmeta_lt; show k + 1 < 2 ^ (64 - 1); omega)
  have hnm : (IntV.cast (IntV.ofNat d.nmeta : I32) : I64).toInt = (d.nmeta : Int) :=
    cast_ofNat (by have := hd.nmeta_lt; omega)
  have hmk : (IntV.lt (IntV.ofInt (-1) : I64) pid_c &&
      IntV.lt (pid_c + (IntV.ofInt 1 : I64)) (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)) =
      decide (k + 1 < d.nmeta) := by
    rw [IntV.lt_eq, IntV.lt_eq, IntV.toInt_ofInt_neg_one (by norm_num), hpc, hk1, hnm]
    have : (-1 : Int) < (k : Int) := by omega
    simp [this]
    omega
  refine safe_load_scalar d.chunkIdx (idx := k + 1) (mk := IntV.lt (IntV.ofInt (-1) : I64) pid_c &&
      IntV.lt (pid_c + (IntV.ofInt 1 : I64)) (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)) ?_ ?_ hvci
    (bytes_sub_R hd.disj.2.2.2.2.2.2.2.2.2.1) (fun v hv1 hv2 => ?_)
  · intro hmk'
    exact ptr_zero_add d.chunkIdx hk1
  · intro hmk'
    rw [hmk] at hmk'
    have := hd.idx_len hN
    have := of_decide_eq_true hmk'
    omega
  have hhi : lc d k = hostIdx d k := by simp [lc, hN]
  have hlo' : lo d k = hostOff d k := by simp [lo, hN]
  have hcn : v.item = if k + 1 < d.nmeta then (IntV.ofNat (hostIdx d (k + 1)) : I32)
      else IntV.ofInt (-1) := by
    by_cases hlt : k + 1 < d.nmeta
    · rw [ite_eq_left hlt]
      have hm : (IntV.lt (IntV.ofInt (-1) : I64) pid_c &&
          IntV.lt (pid_c + (IntV.ofInt 1 : I64)) (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)) = true := by
        rw [hmk]; simpa using hlt
      have h1 := hv1 hm
      have h2 := hmeta.2.1 hN (k + 1) hlt
      rw [h1] at h2
      exact Option.some.inj h2
    · rw [ite_eq_right hlt]
      have hm : (IntV.lt (IntV.ofInt (-1) : I64) pid_c &&
          IntV.lt (pid_c + (IntV.ofInt 1 : I64)) (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)) = false := by
        rw [hmk]; simpa using hlt
      exact hv2 hm
  subst hci hco'
  have hlcv : (IntV.ofNat (lc d k) : I32).toInt = (lc d k : Int) := IntV.I32.toInt_ofNat hlc31.1
  have hlov : (IntV.ofNat (lo d k) : I32).toInt = (lo d k : Int) := IntV.I32.toInt_ofNat hlc31.2
  have hceq : IntV.eq (IntV.ofNat (lc d k) : I32) v.item =
      decide (k + 1 < d.nmeta ∧ hostIdx d (k + 1) = hostIdx d k) := by
    rw [IntV.eq_eq, hcn, hlcv]
    by_cases hlt : k + 1 < d.nmeta
    · have hm := meta_lt hd hlt
      rw [ite_eq_left hlt, IntV.I32.toInt_ofNat (by simpa [lc, hN] using hm.1)]
      simp only [hlt, true_and, decide_eq_decide]; rw [hhi]; omega
    · rw [ite_eq_right hlt, IntV.toInt_ofInt_neg_one (by norm_num)]
      simp only [hlt, false_and, decide_false, decide_eq_false_iff_not]; omega
  have hlt0 : IntV.lt (IntV.ofInt 0 : I32) (IntV.ofNat (lo d k)) = decide (0 < lo d k) := by
    rw [IntV.lt_eq, IntV.toInt_ofInt_zero, hlov]; simp
  dsimp only
  by_cases hc : (IntV.eq (IntV.ofNat (lc d k) : I32) v.item || IntV.lt (IntV.ofInt 0 : I32)
      (IntV.ofNat (lo d k))) = true
  · rw [ite_eq_left hc]
    refine safe_load_scalar d.chunkOffs (idx := k + 1) (mk := IntV.lt (IntV.ofInt (-1) : I64) pid_c &&
      IntV.lt (pid_c + (IntV.ofInt 1 : I64)) (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)) ?_ ?_ hvco
      (bytes_sub_R hd.disj.2.2.2.2.2.2.2.2.2.2) (fun v2 hv1' hv2' => ?_)
    · intro _; exact ptr_zero_add d.chunkOffs hk1
    · intro hmk'
      rw [hmk] at hmk'
      have := hd.offs_len hN
      have := of_decide_eq_true hmk'
      omega
    have hcoffn : v2.item = if k + 1 < d.nmeta then (IntV.ofNat (hostOff d (k + 1)) : I32)
        else IntV.ofNat d.cs := by
      by_cases hlt : k + 1 < d.nmeta
      · rw [ite_eq_left hlt]
        have hm : (IntV.lt (IntV.ofInt (-1) : I64) pid_c &&
            IntV.lt (pid_c + (IntV.ofInt 1 : I64)) (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)) =
            true := by rw [hmk]; simpa using hlt
        have h2 := hmeta.2.2 hN (k + 1) hlt
        rw [hv1' hm] at h2
        exact Option.some.inj h2
      · rw [ite_eq_right hlt]
        have hm : (IntV.lt (IntV.ofInt (-1) : I64) pid_c &&
            IntV.lt (pid_c + (IntV.ofInt 1 : I64)) (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)) =
            false := by rw [hmk]; simpa using hlt
        exact hv2' hm
    -- the decay boundary
    have hone32 : (IntV.ofInt 1 : I32).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
    have hco1 : ((IntV.ofNat (lo d k) : I32) - IntV.ofInt 1).toInt = (lo d k : Int) - 1 :=
      IntV.toInt_sub hlov hone32 (by
        simp only [IntV.SFits]; have := hlc31.2; constructor <;> omega)
    have hmkb : (IntV.lt (IntV.ofInt (-1) : I32) ((IntV.ofNat (lo d k) : I32) - IntV.ofInt 1) &&
        IntV.lt (IntV.ofNat (lo d k) : I32) (IntV.ofNat d.cs)) = decide (0 < lo d k) := by
      rw [IntV.lt_eq, IntV.lt_eq, IntV.toInt_ofInt_neg_one (by norm_num), hco1, hlov, hcsv]
      have := lo_lt hd hk
      by_cases h0 : 0 < lo d k
      · simp [h0, this]
      · have : lo d k = 0 := by omega
        simp [this]
    refine safe_load_scalar d.dA (idx := dtIdx d h (lc d k) (lo d k - 1))
      (mk := IntV.lt (IntV.ofInt (-1) : I32) ((IntV.ofNat (lo d k) : I32) - IntV.ofInt 1) &&
        IntV.lt (IntV.ofNat (lo d k) : I32) (IntV.ofNat d.cs)) ?_ ?_ hvdA
      (bytes_sub_R hd.disj.2.2.2.1) (fun vb hb1 hb2 => ?_)
    · intro hm
      rw [hmkb] at hm
      have h0 : 0 < lo d k := of_decide_eq_true hm
      show (dA_ptr + ((IntV.ofNat (lo d k) : I32) - IntV.ofInt 1) * (IntV.ofNat 1 : I32)) = _
      show dA_ptr.offset _ = _
      have h1 : (((IntV.ofNat (lo d k) : I32) - IntV.ofInt 1) * (IntV.ofNat 1 : I32)).toInt =
          ((lo d k - 1 : Nat) : Int) := by
        have := IntV.mul_nat (a := ((IntV.ofNat (lo d k) : I32) - IntV.ofInt 1)) (b := (IntV.ofNat 1 : I32))
          (x := lo d k - 1) (y := 1) (by rw [hco1]; push_cast [Nat.cast_sub h0]; ring)
          (IntV.I32.toInt_ofNat (by norm_num)) (show (lo d k - 1) * 1 < 2 ^ (32 - 1) by omega)
        rw [this]; simp
      rw [hdA, Buf.ptr_offset, h1]
      congr 1
    · intro hm
      rw [hmkb] at hm
      have h0 : 0 < lo d k := of_decide_eq_true hm
      have := dtIdx_lt hd hk hh (u := lo d k - 1) (by have := lo_lt hd hk; omega)
      exact lt_of_lt_of_le this hd.dA_len
    have hbd : ((0 < lo d k ∧ ∃ b : F32, init.read .f32
        (d.dA.ptr (dtIdx d h (lc d k) (lo d k - 1) : Int)).addr = some b ∧ vb.item = b) ∨
        (lo d k = 0 ∧ vb.item = zf32)) := by
      by_cases h0 : 0 < lo d k
      · have hm : (IntV.lt (IntV.ofInt (-1) : I32) ((IntV.ofNat (lo d k) : I32) - IntV.ofInt 1) &&
            IntV.lt (IntV.ofNat (lo d k) : I32) (IntV.ofNat d.cs)) = true := by
          rw [hmkb]; simpa using h0
        exact Or.inl ⟨h0, vb.item, hb1 hm, rfl⟩
      · have hm : (IntV.lt (IntV.ofInt (-1) : I32) ((IntV.ofNat (lo d k) : I32) - IntV.ofInt 1) &&
            IntV.lt (IntV.ofNat (lo d k) : I32) (IntV.ofNat d.cs)) = false := by
          rw [hmkb]; simpa using h0
        exact Or.inr ⟨by omega, hb2 hm⟩
    refine tail2I_safe S' hN hui hC hS hu hr hhd hds hcb hx hdt hdA hout hph hpm hpn hlov hom hon
      ?_ hdAm hbd
    dsimp only
    have hlt' := lend_eq_init hd hN hk
    have e32 : ∀ a b : I32, (tl.iminimum (Tensor.scalar a) (Tensor.scalar b)).item = IntV.min a b :=
      fun _ _ => rfl
    by_cases hq : (IntV.ofNat (lc d k) : I32).eq v.item = true
    · rw [ite_eq_left hq]
      rw [hceq] at hq
      have hq' := of_decide_eq_true hq
      have hoff : v2.item = IntV.ofNat (hostOff d (k + 1)) := by rw [hcoffn, ite_eq_left hq'.1]
      have hoffv : (IntV.ofNat (hostOff d (k + 1)) : I32).toInt = (hostOff d (k + 1) : Int) :=
        IntV.I32.toInt_ofNat (by have := (meta_lt hd hq'.1).2; simpa [lo, hN] using this)
      rw [e32, IntV.toInt_min, hoff, hoffv, hlimm, hlt'.1 hq']
      push_cast; rfl
    · rw [ite_eq_right hq]
      rw [hceq] at hq
      rw [hlimm, (hlt'.2 (by simpa using hq))]
  · rw [ite_eq_right hc]
    have hc' : ¬ ((k + 1 < d.nmeta ∧ hostIdx d (k + 1) = hostIdx d k) ∨ 0 < lo d k) := by
      intro h'
      apply hc
      rw [Bool.or_eq_true, hceq, hlt0]
      simpa using h'
    push Not at hc'
    have hlend : lend d k = lm d k := by
      unfold lend; rw [ite_eq_right]; intro hh'; exact (hc'.1 hh'.2.1 hh'.2.2) |>.elim
    have hlo0 : lo d k = 0 := by omega
    refine tail2I_safe S' hN hui hC hS hu hr hhd hds hcb hx hdt hdA hout hph hpm hpn hlov hom hon
      ?_ hdAm (Or.inr ⟨hlo0, rfl⟩)
    rw [hlend]; exact hlimm

/-- **The variant with initial states**: walk of `setupI`. -/
theorem init_tile {E : Env} {d : Inst} {init : Mem} (hd : d.ok) (hpre : Pre d init) (dev : Dev E)
    (hN : d.hasInit = true) {i k pm pn h : Nat} (hi : i < gx d * d.nmeta * d.H)
    (hkd : kOf d i = k) (hpmd : pmOf d i = pm) (hpnd : pnOf d i = pn) (hhd : hOf d i = h) :
    Safe E (R d) (Wt d k pm pn h) (fun _ m => Qt d init k pm pn h m) init
      ((model d).body ((launch d).ctx i)) := by
  rw [body_eq_init d hN]
  obtain ⟨hk, hh, hpm, hpn, ha⟩ := decode_lt hd hi
  rw [hkd] at hk; rw [hhd] at hh; rw [hpmd] at hpm; rw [hpnd] at hpn
  have S : Std d init E k pm pn h := ⟨hd, hpre, dev, hk, hpm, hpn, hh⟩
  have pid1 : tl.program_id ((launch d).ctx i) 1 = IntV.ofNat k := by rw [pid1_eq, hkd]
  have pid2 : tl.program_id ((launch d).ctx i) 2 = IntV.ofNat h := by rw [pid2_eq, hhd]
  obtain ⟨hvcb, hvx, hvdt, hvdA, hvseq, hvC, hvst, hvD, hvin, hvci, hvco, hvout, hmeta, hcb0⟩ := hpre
  unfold setupI
  have hk31 : k < 2 ^ 31 := by have := hd.nmeta_lt; omega
  have hh31 : h < 2 ^ 31 := by have := hd.H_lt; omega
  have hcs := cs_pos hd
  have hcs31 : d.cs < 2 ^ 31 := by have := hd.cs_le; omega
  have hcs14 := hd.cs_le
  have hnpn := npn_pos hd
  have hhi : lc d k = hostIdx d k := by simp [lc, hN]
  have hlo' : lo d k = hostOff d k := by simp [lo, hN]
  -- `pid_c = pid_bc // 1`
  have hpbc : (IntV.cast (tl.program_id ((launch d).ctx i) 1) : I64).toInt = (k : Int) := by
    rw [pid1]; exact cast_val (IntV.I32.toInt_ofNat hk31)
  have hone64 : (IntV.ofNat 1 : I64).toInt = 1 := by simp
  refine safe_idiv1 (by rw [hpbc]; positivity) (by rw [hone64]; norm_num) (fun r1 h1 => ?_)
  have hpc : r1.item.toInt = (k : Int) := by rw [h1, hpbc, hone64]; simp
  have hone64n : (IntV.ofNat 1 : I64).toInt = ((1 : Nat) : Int) := I64_ofNat (by norm_num)
  have hpb : (IntV.cast (tl.program_id ((launch d).ctx i) 1) - r1.item * (IntV.ofNat 1 : I64)).toInt
      = 0 := by
    have h := mul64 hpc hone64n (by omega : k * 1 < 2 ^ 62)
    have := IntV.toInt_sub hpbc h (by simp only [IntV.SFits]; push_cast; constructor <;> omega)
    rw [this]; push_cast; ring
  -- the metadata of the logical chunk
  have hneg1 : (IntV.ofInt (-1) : I64).toInt = -1 := IntV.toInt_ofInt_neg_one (by norm_num)
  have hmk1 : IntV.lt (IntV.ofInt (-1) : I64) r1.item = true := by
    rw [IntV.lt_eq, hneg1, hpc, decide_eq_true_eq]; omega
  refine safe_load_scalar d.chunkIdx (idx := k) (mk := IntV.lt (IntV.ofInt (-1) : I64) r1.item)
    (fun _ => ptr_zero_add d.chunkIdx hpc) (fun _ => by have := hd.idx_len hN; omega) hvci
    (bytes_sub_R hd.disj.2.2.2.2.2.2.2.2.2.1) (fun vci hci1 hci2 => ?_)
  have hci : vci.item = (IntV.ofNat (lc d k) : I32) := by
    have := hmeta.2.1 hN k hk
    rw [hci1 hmk1] at this
    rw [hhi]; exact Option.some.inj this
  refine safe_load_scalar d.chunkOffs (idx := k) (mk := IntV.lt (IntV.ofInt (-1) : I64) r1.item)
    (fun _ => ptr_zero_add d.chunkOffs hpc) (fun _ => by have := hd.offs_len hN; omega) hvco
    (bytes_sub_R hd.disj.2.2.2.2.2.2.2.2.2.2) (fun vco hco1 hco2 => ?_)
  have hco' : vco.item = (IntV.ofNat (lo d k) : I32) := by
    have := hmeta.2.2 hN k hk
    rw [hco1 hmk1] at this
    rw [hlo']; exact Option.some.inj this
  -- `num_pid_n`, `pid_m`, `pid_n`
  have hP31 := P_small hd
  have hnum : (IntV.ofNat d.P + (IntV.ofNat (BN - 1) : I32)).toInt = ((d.P + (BN - 1) : Nat) : Int) :=
    IntV.add_nat (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_ofNat (by norm_num [BN]))
      (by simp only [BN]; omega)
  have hbn : (IntV.ofNat BN : I32).toInt = (BN : Int) := IntV.I32.toInt_ofNat (by norm_num [BN])
  refine safe_idiv1 (by rw [hnum]; positivity) (by rw [hbn]; norm_num [BN]) (fun r2 h2 => ?_)
  have hnp : r2.item.toInt = (npn d : Int) := by
    rw [h2, hnum, hbn]; unfold npn; simp only [BN]; omega
  have hp0 : (tl.program_id ((launch d).ctx i) 0).toInt = ((i % gx d : Nat) : Int) := by
    rw [pid0_eq]; exact IntV.I32.toInt_ofNat (by have := hd.grid_lt; have := Nat.mod_le i (gx d); omega)
  refine safe_idiv1 (by rw [hp0]; positivity) (by rw [hnp]; exact_mod_cast hnpn) (fun r3 h3 => ?_)
  have hpmv : r3.item.toInt = (pm : Int) := by
    rw [h3, hp0, hnp, ← hpmd]; unfold pmOf; exact (Int.natCast_ediv _ _).symm
  refine safe_irem1 (by rw [hp0]; positivity) (by rw [hnp]; exact_mod_cast hnpn) (fun r4 h4 => ?_)
  have hpnv : r4.item.toInt = (pn : Int) := by
    rw [h4, hp0, hnp, ← hpnd]; unfold pnOf; exact (Int.natCast_emod _ _).symm
  -- the group of the head, twice
  have hph : (tl.program_id ((launch d).ctx i) 2).toInt = (h : Int) := by
    rw [pid2]; exact IntV.I32.toInt_ofNat hh31
  have hrat : ratio d < 2 ^ 31 := by
    have := hd.H_lt; unfold ratio; have := Nat.div_le_self d.H d.G; omega
  have hratv : (IntV.ofNat (ratio d) : I32).toInt = (ratio d : Int) := IntV.I32.toInt_ofNat hrat
  have hratp := ratio_pos hd
  refine safe_idiv1 (by rw [hph]; positivity) (by rw [hratv]; exact_mod_cast hratp) (fun r5 h5 => ?_)
  have hg5 : r5.item.toInt = (grp d h : Int) := by
    rw [h5, hph, hratv]; exact (Int.natCast_ediv _ _).symm
  refine safe_idiv1 (by rw [hph]; positivity) (by rw [hratv]; exact_mod_cast hratp) (fun r6 h6 => ?_)
  have hg6 : r6.item.toInt = (grp d h : Int) := by
    rw [h6, hph, hratv]; exact (Int.natCast_ediv _ _).symm
  -- name the scalars
  generalize (IntV.cast (tl.program_id ((launch d).ctx i) 1) - r1.item * (IntV.ofNat 1 : I64)) = pb at hpb ⊢
  generalize r1.item = pc at hpc ⊢
  generalize vci.item = ci at hci ⊢
  generalize vco.item = co at hco' ⊢
  generalize r3.item = pmv at hpmv ⊢
  generalize r4.item = pnv at hpnv ⊢
  generalize r5.item = gcb at hg5 ⊢
  generalize r6.item = gC at hg6 ⊢
  generalize tl.program_id ((launch d).ctx i) 2 = phv at hph ⊢
  -- sizes
  have hlend := lend_le hd hk
  have hlc31 := meta_lt hd hk
  have hlcs := lc_start_lt hd hk
  have hT31 : d.T < 2 ^ 31 := by have := hd.T_ctx; omega
  have hcit : ci.toInt = (lc d k : Int) := by rw [hci]; exact IntV.I32.toInt_ofNat hlc31.1
  have hcsv : (IntV.ofNat d.cs : I32).toInt = (d.cs : Int) := IntV.I32.toInt_ofNat hcs31
  have hTv : (IntV.ofNat d.T : I32).toInt = (d.T : Int) := IntV.I32.toInt_ofNat hT31
  have hcics : (ci * (IntV.ofNat d.cs : I32)).toInt = ((lc d k * d.cs : Nat) : Int) :=
    IntV.mul_nat hcit hcsv (show lc d k * d.cs < 2 ^ (32 - 1) by omega)
  have hlimv : ((tl.iminimum (Tensor.scalar (IntV.ofNat d.cs : I32))
      (Tensor.scalar ((IntV.ofNat d.T : I32) - ci * (IntV.ofNat d.cs : I32)))).item).toInt =
      (lm d k : Int) := by
    have h2 : ((IntV.ofNat d.T : I32) - ci * (IntV.ofNat d.cs : I32)).toInt =
        ((d.T - lc d k * d.cs : Nat) : Int) :=
      IntV.sub_nat (x := d.T) hTv hcics (by omega) (by omega)
    have e : ∀ a b : I32, (tl.iminimum (Tensor.scalar a) (Tensor.scalar b)).item = IntV.min a b :=
      fun _ _ => rfl
    rw [e, IntV.toInt_min, hcsv, h2]
    unfold lm; push_cast; rfl
  -- the pointers
  have hnc1 := nc_pos hd
  have hT1 := T_pos hd
  have hnc : lc d k < nc d := lc_lt hd hk
  have hlcs' : lc d k * d.cs ≤ d.T := hlcs.le
  have hcT31 : d.cT < 2 ^ 31 := by
    have : d.cT ≤ d.T * d.cT := Nat.le_mul_of_pos_left _ hT1
    have := hd.c_arg; omega
  have hgN : grp d h * d.N < 2 ^ 31 := by
    have h1 : (grp d h) * d.N ≤ d.G * d.N := Nat.mul_le_mul_right _ (grp_lt hd hh).le
    have := hd.cT_ge; omega
  have hN31 : d.N < 2 ^ 31 := by have := hd.N_le; have := hd.bsd_le; omega
  have hHnc : d.H * (nc d * d.cs) < 2 ^ 31 := hd.dt_i32
  have hnccs : nc d * d.cs < 2 ^ 31 := by
    have : nc d * d.cs ≤ d.H * (nc d * d.cs) := Nat.le_mul_of_pos_left _ hd.H_pos
    omega
  have hh_nc : h * (nc d * d.cs) < 2 ^ 31 := by
    have : h * (nc d * d.cs) ≤ d.H * (nc d * d.cs) := Nat.mul_le_mul_right _ hh.le
    omega
  have hlc_cs : lc d k * d.cs < 2 ^ 31 := by omega
  have hCP : d.C.ptr 0 + (pb * (IntV.ofNat (d.T * d.cT) : I32).cast +
      IntV.cast (ci * (IntV.ofNat d.cs : I32) * (IntV.ofNat d.cT : I32)) +
      IntV.cast (gC * (IntV.ofNat d.N : I32))) =
      d.C.ptr ((lc d k * d.cs * d.cT + grp d h * d.N : Nat) : Int) :=
    ptr_b' d.C hpb hcit hg6 hd.c_arg hcs31 hcT31 hN31 hgN hlc_cs
      (by have : lc d k * d.cs * d.cT ≤ d.T * d.cT := Nat.mul_le_mul_right _ hlcs'
          have := hd.c_arg; omega)
  have hHPN : d.H * PN d < 2 ^ 31 := by
    have : d.H * PN d ≤ nc d * (d.H * PN d) := Nat.le_mul_of_pos_left _ hnc1
    have := hd.st_i32; omega
  have hPN31 : PN d < 2 ^ 31 := by
    have : PN d ≤ d.H * PN d := Nat.le_mul_of_pos_left _ hd.H_pos
    omega
  have hhPN : h * PN d < 2 ^ 31 := by
    have h1 : h * PN d ≤ d.H * PN d := Nat.mul_le_mul_right _ hh.le
    omega
  have hkHPN : lc d k * (d.H * PN d) < 2 ^ 31 := by
    have h1 : lc d k * (d.H * PN d) ≤ nc d * (d.H * PN d) := Nat.mul_le_mul_right _ hnc.le
    have := hd.st_i32; omega
  have hSP : d.states.ptr 0 + pb * (IntV.ofNat (nc d * (d.H * PN d)) : I32).cast +
      ci * (IntV.ofNat (d.H * PN d) : I32) + phv * (IntV.ofNat (PN d) : I32) =
      d.states.ptr ((lc d k * (d.H * PN d) + h * PN d : Nat) : Int) :=
    ptr_s' d.states hpb hcit hph (by have := hd.st_i32; omega) hHPN hPN31 hhPN hkHPN
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
  have hkc : lc d k * (d.G * (d.cs * d.cs)) < 2 ^ 31 := by
    have : lc d k * (d.G * (d.cs * d.cs)) ≤ nc d * (d.G * (d.cs * d.cs)) :=
      Nat.mul_le_mul_right _ hnc.le
    omega
  have hCBP : d.cb.ptr 0 + (pb * (IntV.ofNat (nc d * (d.G * (d.cs * d.cs))) : I32).cast +
      IntV.cast (ci * (IntV.ofNat (d.G * (d.cs * d.cs)) : I32)) +
      IntV.cast (gcb * (IntV.ofNat (d.cs * d.cs) : I32))) =
      d.cb.ptr (cbIdx d (lc d k) (grp d h) 0 0 : Int) := by
    rw [ptr_a' d.cb hpb hcit hg5 hcbA hGcs hcs2 hgc hkc]
    congr 1; unfold cbIdx; push_cast; ring
  have hxT31 : d.T * d.xT < 2 ^ 31 := hd.x_arg
  have hxT : d.xT < 2 ^ 31 := by
    have : d.xT ≤ d.T * d.xT := Nat.le_mul_of_pos_left _ hT1
    omega
  have hhP : h * d.P < 2 ^ 31 := by
    have := hd.xT_ge; have : h * d.P ≤ d.H * d.P := Nat.mul_le_mul_right _ hh.le
    have := hd.x_off
    have : d.xT ≤ (2 * d.cs + 64) * d.xT := Nat.le_mul_of_pos_left _ (by omega)
    omega
  have hXP : d.x.ptr 0 + (pb * (IntV.ofNat (d.T * d.xT) : I32).cast +
      IntV.cast (ci * (IntV.ofNat d.cs : I32) * (IntV.ofNat d.xT : I32)) +
      IntV.cast (phv * (IntV.ofNat d.P : I32))) =
      d.x.ptr (xIdx d (lc d k * d.cs) h 0 : Int) := by
    rw [ptr_b' d.x hpb hcit hph hxT31 hcs31 hxT (P_lt31 hd) hhP hlc_cs
      (by have : lc d k * d.cs * d.xT ≤ d.T * d.xT := Nat.mul_le_mul_right _ hlcs'; omega)]
    congr 1
  have hDTP : d.dt.ptr 0 + (pb * (IntV.ofNat (d.H * (nc d * d.cs)) : I32).cast +
      IntV.cast (ci * (IntV.ofNat d.cs : I32)) +
      IntV.cast (phv * (IntV.ofNat (nc d * d.cs) : I32))) =
      d.dt.ptr (dtIdx d h (lc d k) 0 : Int) := by
    rw [ptr_a' d.dt hpb hcit hph hHnc hcs31 hnccs hh_nc hlc_cs]
    congr 1; unfold dtIdx; push_cast; ring
  have hDAP : d.dA.ptr 0 + (pb * (IntV.ofNat (d.H * (nc d * d.cs)) : I32).cast +
      IntV.cast (ci * (IntV.ofNat d.cs : I32)) +
      IntV.cast (phv * (IntV.ofNat (nc d * d.cs) : I32))) =
      d.dA.ptr (dtIdx d h (lc d k) 0 : Int) := by
    rw [ptr_a' d.dA hpb hcit hph hHnc hcs31 hnccs hh_nc hlc_cs]
    congr 1; unfold dtIdx; push_cast; ring
  have hout31 : d.T * (d.H * d.P) < 2 ^ 31 := hd.out_arg
  have hHP : d.H * d.P < 2 ^ 31 := by
    have : d.H * d.P ≤ d.T * (d.H * d.P) := Nat.le_mul_of_pos_left _ hT1
    omega
  have hhP' : h * d.P < 2 ^ 31 := hhP
  have hOP : d.out.ptr 0 + (pb * (IntV.ofNat (d.T * (d.H * d.P)) : I32).cast +
      IntV.cast (ci * (IntV.ofNat d.cs : I32) * (IntV.ofNat (d.H * d.P) : I32)) +
      IntV.cast (phv * (IntV.ofNat d.P : I32))) =
      d.out.ptr ((lc d k * d.cs * (d.H * d.P) + h * d.P : Nat) : Int) :=
    ptr_b' d.out hpb hcit hph hout31 hcs31 hHP (P_lt31 hd) hhP' hlc_cs
      (by have : lc d k * d.cs * (d.H * d.P) ≤ d.T * (d.H * d.P) := Nat.mul_le_mul_right _ hlcs'
          omega)
  -- `seq_idx_prev`
  have hsip : d.seqIdx.ptr 0 + (pb * (IntV.ofNat d.T : I32).cast +
      IntV.cast (ci * (IntV.ofNat d.cs : I32) * (IntV.ofNat 1 : I32))) =
      d.seqIdx.ptr ((lc d k * d.cs * 1 : Nat) : Int) :=
    ptr_c' d.seqIdx hpb hcit hT31 hcs31 (by norm_num) hlc_cs (by omega)
  have hneg : (-(IntV.ofNat 1 : I32)).toInt = -1 := by decide
  have hone32 : (IntV.ofInt 1 : I32).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
  have hle1 : IntV.le (IntV.ofInt 1 : I32) ci = decide (1 ≤ lc d k) := by
    rw [IntV.le_eq, hone32, hcit]; simp
  refine safe_load_scalar d.seqIdx (idx := lc d k * d.cs - 1) (mk := IntV.le (IntV.ofInt 1 : I32) ci)
    ?_ ?_ hvseq (bytes_sub_R hd.disj.2.2.2.2.1) (fun vsp hsp1 hsp2 => ?_)
  · intro hmk
    rw [hle1] at hmk
    have hk1 : 1 ≤ lc d k * d.cs := Nat.mul_pos (of_decide_eq_true hmk) hcs
    show (_ : Ptr .i32).offset _ = _
    rw [hsip, Buf.ptr_offset, hneg]
    congr 1
    push_cast [Nat.cast_sub hk1]; ring
  · intro hmk
    have := hd.seq_len
    omega
  have hprev : vsp.item = (IntV.ofNat (prevReq d (lc d k)) : I32) := by
    by_cases hle : IntV.le (IntV.ofInt 1 : I32) ci = true
    · have h1 := hsp1 hle
      rw [hle1] at hle
      have hk1 : 1 ≤ lc d k := of_decide_eq_true hle
      have hk2 : 1 ≤ lc d k * d.cs := Nat.mul_pos hk1 hcs
      have := hmeta.1 (lc d k * d.cs - 1) (by omega)
      rw [h1] at this
      rw [Option.some.inj this]
      unfold prevReq; rw [ite_eq_left hk1]
    · have h2 := hsp2 (by simpa using hle)
      have hk0 : lc d k = 0 := by
        rw [hle1] at hle; simp at hle; omega
      rw [h2]; unfold prevReq; rw [hk0]; rfl
  generalize vsp.item = sp at hprev ⊢
  generalize (tl.iminimum (Tensor.scalar (IntV.ofNat d.cs : I32))
    (Tensor.scalar ((IntV.ofNat d.T : I32) - ci * (IntV.ofNat d.cs : I32)))).item = lim at hlimv ⊢
  -- the first runtime `if`
  have hlo31 : lo d k < 2 ^ 31 := hlc31.2
  have hpmB : BM * pm < d.cs := by unfold nmt at hpm; simp only [BM] at *; omega
  have hlolt := lo_lt hd hk
  have hmc : (pmv * (IntV.ofNat BM : I32) + co).toInt = ((pm * BM + lo d k : Nat) : Int) := by
    have h1 : (pmv * (IntV.ofNat BM : I32)).toInt = ((pm * BM : Nat) : Int) :=
      IntV.mul_nat hpmv (IntV.I32.toInt_ofNat (by norm_num [BM]))
        (show pm * BM < 2 ^ (32 - 1) by simp only [BM] at *; omega)
    have hcov : co.toInt = (lo d k : Int) := by rw [hco']; exact IntV.I32.toInt_ofNat hlo31
    exact IntV.add_nat h1 hcov (show pm * BM + lo d k < 2 ^ (32 - 1) by simp only [BM] at *; omega)
  have hcond1 : IntV.lt (pmv * (IntV.ofNat BM : I32) + co) lim = decide (pm * BM + lo d k < lm d k) := by
    rw [IntV.lt_eq, hmc, hlimv]; simp only [decide_eq_decide]; push_cast; omega
  have hom : ∀ ii : Fin BM, ((pmv * (IntV.ofNat BM : I32) + co + tl.arange 0 BM : Tensor [BM] I32).get1 ii).toInt
      = (rowOf d k pm ii : Int) := fun ii => by
    have := IntV.I32.toInt_sadd_arange (n := BM) hmc (by simp only [BM] at *; omega) ii
    rw [this]; unfold rowOf; push_cast; ring
  have hd2 := fun (ii : Nat) (hri : RowOk d k pm ii) => (show rowOf d k pm ii < lm d k by
    unfold RowOk at hri; have : lend d k ≤ lm d k := by unfold lend; split_ifs <;> simp
    omega)
  dsimp only
  by_cases hc1 : IntV.lt (pmv * (IntV.ofNat BM : I32) + co) lim = true
  · rw [ite_eq_left hc1]
    rw [hcond1] at hc1
    have hlt1 : pm * BM + lo d k < lm d k := of_decide_eq_true hc1
    have hlmT : lm d k ≤ d.T - lc d k * d.cs := by unfold lm; exact min_le_right _ _
    have hT0 : lc d k * d.cs * 1 + (pm * BM + lo d k) * 1 < d.T := by omega
    have hmc1 : ((pmv * (IntV.ofNat BM : I32) + co) * (IntV.ofNat 1 : I32)).toInt =
        (((pm * BM + lo d k) * 1 : Nat) : Int) :=
      IntV.mul_nat hmc (IntV.I32.toInt_ofNat (by norm_num))
        (show (pm * BM + lo d k) * 1 < 2 ^ (32 - 1) by simp only [BM] at *; omega)
    have hseqm : d.seqIdx.ptr 0 + (pb * (IntV.ofNat d.T : I32).cast +
        IntV.cast (ci * (IntV.ofNat d.cs : I32) * (IntV.ofNat 1 : I32))) +
        (pmv * (IntV.ofNat BM : I32) + co) * (IntV.ofNat 1 : I32) =
        d.seqIdx.ptr ((lc d k * d.cs * 1 + (pm * BM + lo d k) * 1 : Nat) : Int) := by
      show (_ : Ptr .i32).offset _ = _
      rw [hsip, Buf.ptr_offset, hmc1]
      congr 1
    have hidx : lc d k * d.cs * 1 + (pm * BM + lo d k) * 1 < d.seqIdx.len :=
      lt_of_lt_of_le hT0 hd.seq_len
    obtain ⟨vm, hvm⟩ := d.seqIdx.read_some hvseq hidx
    refine safe_load1 (v0 := vm) ?_ ?_ ?_
    · rw [hseqm]; exact hvm
    · rw [hseqm]; exact (d.seqIdx.span_sub hidx).trans (bytes_sub_R hd.disj.2.2.2.2.1)
    have hvmv : vm = (IntV.ofNat (reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1)) : I32) := by
      have := hmeta.1 _ hT0
      rw [hvm] at this
      exact Option.some.inj this
    have hcov : co.toInt = (lo d k : Int) := by rw [hco']; exact IntV.I32.toInt_ofNat hlo31
    have hreq31 := reqOf_lt hd (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1)
    have hcoeq : IntV.eq co (IntV.ofInt 0 : I32) = decide (lo d k = 0) := by
      rw [IntV.eq_eq, hcov, IntV.toInt_ofInt_zero]; simp
    have hltc : IntV.lt (IntV.ofInt 0 : I32) co = decide (0 < lo d k) := by
      rw [IntV.lt_eq, hcov, IntV.toInt_ofInt_zero]; simp
    have hspv : IntV.eq sp vm = decide (prevReq d (lc d k) =
        reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1)) := by
      rw [IntV.eq_eq, hprev, hvmv, IntV.I32.toInt_ofNat (prevReq_lt hd _),
        IntV.I32.toInt_ofNat hreq31]; simp
    have hrw : ∀ ii, RowOk d k pm ii →
        lc d k * d.cs * 1 + (pm * BM + lo d k) * 1 = lc d k * d.cs + rowOf d k pm 0 := by
      intro ii hri; unfold rowOf; ring
    have hreqs : ∀ ii, RowOk d k pm ii →
        reqOf d (tokOf d k pm ii) = reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1) := by
      intro ii hri
      have hrow : lo d k ≤ rowOf d k pm ii ∧ rowOf d k pm ii < lend d k :=
        ⟨by unfold rowOf; omega, hri⟩
      have hrow0 : lo d k ≤ rowOf d k pm 0 ∧ rowOf d k pm 0 < lend d k :=
        ⟨by unfold rowOf; omega, by unfold RowOk at hri; unfold rowOf at *; omega⟩
      rw [hrw ii hri]
      exact req_same hd hN hk hrow hrow0
    have hcnd : (IntV.eq co (IntV.ofInt 0 : I32) && !IntV.eq sp vm ||
        IntV.lt (IntV.ofInt 0 : I32) co) = true ↔
        (0 < lo d k ∨ prevReq d (lc d k) ≠
          reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1)) := by
      rw [hcoeq, hspv, hltc]
      by_cases h0 : 0 < lo d k
      · simp [h0]
      · have : lo d k = 0 := by omega
        simp [this]
    have hfi : ∀ ii, RowOk d k pm ii → (fromInit d (tokOf d k pm ii) = true ↔
        (0 < lo d k ∨ prevReq d (lc d k) ≠
          reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1))) := by
      intro ii hri
      have hrow : lo d k ≤ rowOf d k pm ii ∧ rowOf d k pm ii < lend d k :=
        ⟨by unfold rowOf; omega, hri⟩
      have := fromInit_init hd hN hk hrow
      rw [← hreqs ii hri]
      exact this
    by_cases hc2 : (IntV.eq co (IntV.ofInt 0 : I32) && !IntV.eq sp vm ||
        IntV.lt (IntV.ofInt 0 : I32) co) = true
    · rw [ite_eq_left hc2]
      have hrlt : reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1) < d.nreq :=
        (reqOf_spec hd hT0).1
      refine tail1I_safe S hN hpc hci hco' hlimv (u := true)
        (r := reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1)) (fun _ => ⟨hN, hrlt⟩)
        ?_ ?_ ?_ ?_ rfl rfl hCBP hXP hDTP hDAP hOP hph hpmv hpnv hom
      · exact hCP
      · show d.initst.ptr 0 + vm * (IntV.ofNat (d.H * PN d) : I32) + phv * (IntV.ofNat (PN d) : I32) = _
        have hvmt : vm.toInt = (reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1) : Int) := by
          rw [hvmv]; exact IntV.I32.toInt_ofNat hreq31
        have hr1 : reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1) * (d.H * PN d) < 2 ^ 31 := by
          have h1 : reqOf d (lc d k * d.cs * 1 + (pm * BM + lo d k) * 1) ≤ d.nreq - 1 := by omega
          have h2 := hd.init_i32 hN
          have := Nat.mul_le_mul_right (d.H * PN d) h1
          omega
        exact ptr_i d.initst hvmt hph hHPN hPN31 hr1 hhPN
      · intro ii hri
        have := (hfi ii hri).mpr (hcnd.mp hc2)
        simp [hN, this]
      · intro _ ii hri; exact hreqs ii hri
    · rw [ite_eq_right hc2]
      refine tail1I_safe S hN hpc hci hco' hlimv (u := false) (r := 0) (fun h' => absurd h' (by simp))
        ?_ ?_ ?_ (fun h' => absurd h' (by simp)) rfl rfl hCBP hXP hDTP hDAP hOP hph hpmv hpnv hom
      · exact hCP
      · simp only [stBuf, stBase, Bool.false_eq_true, ite_false]; exact hSP
      · intro ii hri
        have : ¬ fromInit d (tokOf d k pm ii) = true := fun h' => hc2 (hcnd.mpr ((hfi ii hri).mp h'))
        simp [this]
  · rw [ite_eq_right hc1]
    have hnone : ∀ ii, RowOk d k pm ii → False := by
      intro ii hri
      apply hc1
      rw [hcond1]
      have h1 : lo d k + 64 * pm + ii < lm d k := by
        have := hd2 ii hri; unfold rowOf at this; simpa [BM] using this
      exact decide_eq_true (by simp only [BM]; omega)
    refine tail1I_safe S hN hpc hci hco' hlimv (u := false) (r := 0) (fun h' => absurd h' (by simp))
      ?_ ?_ (fun ii hri => (hnone ii hri).elim) (fun h' => absurd h' (by simp)) rfl rfl hCBP hXP hDTP
      hDAP hOP hph hpmv hpnv hom
    · exact hCP
    · simp only [stBuf, stBase, Bool.false_eq_true, ite_false]; exact hSP

end Claims.SsdChunkScan