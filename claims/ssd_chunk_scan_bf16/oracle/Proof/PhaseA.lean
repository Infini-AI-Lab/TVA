import Proof.Walk
import Proof.Bounds
import Proof.Lane

/-!
# Phase A: the state term

Walk of `phaseA`: the two loads (the `[BM, BSD]` tile of `C` and the `[BSD, BN]` tile of the state),
the conversion, the `tl.dot` and the product with the scale. It leaves `APost`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- **Phase A is safe**: it leaves `APost`. -/
theorem phaseA_safe {w : Nat} {E : Env} {d : Inst} {init : Mem} {k pm pn h : Nat}
    (S : Std d init E k pm pn h) {α : Type} {Q : α → Mem → Prop} {Wf : Set Int}
    {C_ptr : Ptr .bf16} {prev_ptr : Ptr .f32} {hdimS dstS : I32} {offs_m : Tensor [BM] I32}
    {offs_n : Tensor [BN] I32} {offs_mw : Tensor [BM] (IntV true w)} {limit : IntV true w}
    {scale_m : Tensor [BM] F32} {u : Bool} {r : Nat}
    (hui : u = true → d.hasInit = true ∧ r < d.nreq)
    (hC : C_ptr = d.C.ptr ((lc d k * d.cs * d.cT + grp d h * d.N : Nat) : Int))
    (hS : prev_ptr = (stBuf d u).ptr (stBase d u (lc d k) r h : Nat))
    (hu : ∀ ii, RowOk d k pm ii → (d.hasInit && fromInit d (tokOf d k pm ii)) = u)
    (hr : u = true → ∀ ii, RowOk d k pm ii → reqOf d (tokOf d k pm ii) = r)
    (hhd : hdimS = IntV.ofNat d.N) (hds : dstS = IntV.ofNat 1)
    (hom : ∀ ii : Fin BM, (offs_m.get1 ii).toInt = (rowOf d k pm ii : Int))
    (hon : ∀ jj : Fin BN, (offs_n.get1 jj).toInt = (colOf pn jj : Int))
    (homw : ∀ ii : Fin BM, (offs_mw.get1 ii).toInt = (rowOf d k pm ii : Int))
    (hlim : limit.toInt = (lend d k : Int))
    (hsc : Vals init d → ∀ ii : Fin BM, RowOk d k pm ii →
      ScaleOK init d (tokOf d k pm ii) h (scale_m.get1 ii))
    {kont : Tensor [BM, BN] F32 → Prog α}
    (hkont : ∀ acc₀, APost d init k pm pn h acc₀ → Safe E (R d) Wf Q init (kont acc₀)) :
    Safe E (R d) Wf Q init
      (phaseA d C_ptr prev_ptr hdimS dstS offs_m offs_n offs_mw limit scale_m kont) := by
  obtain ⟨hd, hpre, dev, hk, hpm, hpn, hh⟩ := S
  obtain ⟨hvcb, hvx, hvdt, hvdA, hvseq, hvC, hvst, hvD, hvin, hvci, hvco, hvout, hmeta, hcb0⟩ := hpre
  have hpre : Pre d init := ⟨hvcb, hvx, hvdt, hvdA, hvseq, hvC, hvst, hvD, hvin, hvci, hvco, hvout,
    hmeta, hcb0⟩
  unfold phaseA
  have hlend := lend_le hd hk
  have hcsp := cs_pos hd
  have hg := grp_lt hd hh
  have hN31 : d.N < 2 ^ 31 := by have := hd.N_le; have := hd.bsd_le; omega
  have hcT31 : d.cT < 2 ^ 31 := by
    have := hd.c_off; nlinarith [hd.cT_ge, hd.N_pos, hd.G_pos, Nat.zero_le d.BSD]
  have hcT : (IntV.ofNat d.cT : I32).toInt = (d.cT : Int) := IntV.I32.toInt_ofNat hcT31
  have hone : (IntV.ofNat 1 : I32).toInt = ((1 : Nat) : Int) := IntV.I32.toInt_ofNat (by norm_num)
  have hNv : (IntV.ofNat d.N : I32).toInt = (d.N : Int) := IntV.I32.toInt_ofNat hN31
  have hkd : ∀ n : Fin d.BSD, ((tl.arange 0 d.BSD : Tensor [d.BSD] I32).get1 n).toInt = (n.val : Int) :=
    fun n => IntV.I32.toInt_tl_arange (n := d.BSD) (by have := hd.bsd_le; omega) n
  -- the `[BM, BSD]` tile of `C`
  have hCfit : (lend d k - 1) * d.cT + (d.N - 1) * 1 < 2 ^ (32 - 1) := by
    have h1 : (lend d k - 1) * d.cT ≤ (2 * d.cs + 64) * d.cT :=
      Nat.mul_le_mul_right _ (by omega)
    have := hd.c_off; have := hd.N_le; omega
  have hCp := Tile2.ptr_lane_on (d := .bf16) d.C hC (rows := offs_m) (cols := tl.arange 0 d.BSD)
    (ρ := fun i : Fin BM => rowOf d k pm i) (κ := fun n : Fin d.BSD => n.val) hom hkd hcT hone hCfit
  have hClen : lend d k = 0 ∨ d.N = 0 ∨
      lc d k * d.cs * d.cT + grp d h * d.N + (lend d k - 1) * d.cT + (d.N - 1) * 1 < d.C.len := by
    rcases Nat.eq_zero_or_pos (lend d k) with h0 | h0
    · exact Or.inl h0
    refine Or.inr (Or.inr ?_)
    have e1 : (lc d k * d.cs + (lend d k - 1)) * d.cT ≤ (d.T - 1) * d.cT :=
      Nat.mul_le_mul_right _ (by omega)
    have e1' : (lc d k * d.cs + (lend d k - 1)) * d.cT =
        lc d k * d.cs * d.cT + (lend d k - 1) * d.cT := Nat.add_mul _ _ _
    have e2 : (grp d h + 1) * d.N ≤ d.G * d.N := grp_mul_le hd hh d.N
    have e2' : (grp d h + 1) * d.N = grp d h * d.N + d.N := by rw [Nat.add_mul, Nat.one_mul]
    have := hd.C_len
    have := hd.N_pos
    omega
  -- the `[BSD, BN]` tile of the state
  have hdsv : dstS.toInt = ((1 : Nat) : Int) := by rw [hds]; exact hone
  have hdhv : hdimS.toInt = (d.N : Int) := by rw [hhd]; exact hNv
  have hP31 : d.P < 2 ^ 31 := P_lt31 hd
  have hPv : (IntV.ofNat d.P : I32).toInt = (d.P : Int) := IntV.I32.toInt_ofNat hP31
  have hPN31 : d.P * d.N < 2 ^ 31 := by
    have h1 := hd.st_i32
    have h2 : d.P * d.N ≤ nc d * (d.H * PN d) := by
      have h3 : 1 ≤ nc d * d.H := Nat.mul_pos (nc_pos hd) hd.H_pos
      calc d.P * d.N = 1 * (d.P * d.N) := (Nat.one_mul _).symm
        _ ≤ (nc d * d.H) * (d.P * d.N) := Nat.mul_le_mul_right _ h3
        _ = nc d * (d.H * PN d) := by unfold PN; ring
    omega
  have hSfit0 : (d.N - 1) * 1 < 2 ^ (32 - 1) := by omega
  have hSfit1 : (d.P - 1) * d.N < 2 ^ (32 - 1) := by
    have : (d.P - 1) * d.N ≤ d.P * d.N := Nat.mul_le_mul_right _ (by omega)
    omega
  have hSfit : (d.N - 1) * 1 + (d.P - 1) * d.N < 2 ^ (32 - 1) := by
    have : (d.P - 1) * d.N + d.N ≤ d.P * d.N := by
      have := hd.P_pos
      exact le_of_eq (by rw [← Nat.succ_mul]; congr 1; omega)
    have := hd.N_pos
    omega
  have huc : u = false → lc d k < nc d := fun _ => lc_lt hd hk
  have hSlen : d.N = 0 ∨ d.P = 0 ∨ stBase d u (lc d k) r h + (d.N - 1) * 1 + (d.P - 1) * d.N <
      (stBuf d u).len := Or.inr (Or.inr (st_len hd hh huc hui))
  refine safe_convert (fun _ => exists_conv_f32_bf16 _ _) (fun z hz => ?_)
  have hz0 : ∀ i j, z.get2 i j = FloatV.ofPat 0 := fun i j =>
    convRel_zf32 (by have := hz (i, j, ()); simp only [tl.full] at this; exact this)
  refine safe_load_tileAt d.C (e := lc d k * d.cs * d.cT + grp d h * d.N) (s0 := d.cT) (s1 := 1)
    (ρ := fun i : Fin BM => rowOf d k pm i) (κ := fun n : Fin d.BSD => n.val) (M := lend d k)
    (N := d.N) ?_ ?_ ?_ hvC ?_ fun v hv1 hv2 => ?_
  · exact hCp
  · exact maskAt_and2 homw hkd hlim hNv
  · exact hClen
  · exact (Buf.tileAt_sub_bytes d.C hClen).trans (bytes_sub_R hd.disj.2.2.2.2.2.1)
  refine safe_load_tileAt (stBuf d u) (e := stBase d u (lc d k) r h) (s0 := 1) (s1 := d.N)
    (ρ := fun n : Fin d.BSD => n.val) (κ := fun jj : Fin BN => colOf pn jj) (M := d.N)
    (N := d.P) ?_ ?_ ?_ (stBuf_valid hpre u) ?_ fun vS hS1 hS2 => ?_
  · exact Tile2.ptr_lane_sum_swap_of_on (stBuf d u) hS
      (fun i j hi => Tile2.colMul_toInt_on hkd hdsv hSfit0 i j hi)
      (fun i j hj => Tile2.rowMul_toInt_on hon hdhv hSfit1 i j hj) hSfit
  · exact Tile2.maskAt_and hkd hon hNv hPv
  · exact hSlen
  · exact (Buf.tileAt_sub_bytes _ hSlen).trans (bytes_sub_R (stBuf_disj hd u))
  refine safe_cvt ⟨by simp [Format.Std], by simp [Format.Std], Or.inl rfl⟩ (fun b1 hb1 => ?_)
  refine safe_dot (dev.dotTotal _ _ _) (fun rr hrr => ?_)
  refine safe_mul (fun acc hacc => ?_)
  refine hkont acc ?_
  intro hV ii jj hri hcj
  have hrow : lo d k ≤ rowOf d k pm ii ∧ rowOf d k pm ii < lend d k := ⟨by unfold rowOf; omega, hri⟩
  obtain ⟨ht, hchk, -, -⟩ := tok_of_row hd hk hrow
  have ht' : tokOf d k pm ii < d.T := ht
  have hchk' : chk d (tokOf d k pm ii) = lc d k := hchk
  have hrow' : rowOf d k pm ii < lend d k := hri
  refine accA_fact (P := E.P) dev.dotAcc hd hV ht' hh hcj (hsc hV ii hri) hrr ?_ ?_ ?_
  · intro n
    have e : cIdx d (tokOf d k pm ii) (grp d h) n =
        lc d k * d.cs * d.cT + grp d h * d.N + rowOf d k pm ii * d.cT + n * 1 := by
      unfold cIdx tokOf; ring
    refine ⟨fun hn => ?_, fun hn => ?_⟩
    · rw [e]; exact hv1 ii n hrow' hn
    · exact (hv2 ii n (by omega) z rfl).trans (hz0 ii n)
  · intro n
    refine ⟨vS.get2 n jj, ?_, fun hn => ?_, fun hn => ?_⟩
    · exact hb1 (n, jj, ())
    · have e : stSrc d (tokOf d k pm ii) h (colOf pn jj) n =
          (stBuf d u).ptr ((stBase d u (lc d k) r h + n * 1 + colOf pn jj * d.N : Nat) : Int) := by
        unfold stSrc
        rw [hu ii hri, hchk']
        cases u
        · simp only [stBase, Bool.false_eq_true, ite_false]; congr 2; ring
        · rw [hr rfl ii hri]; congr 2; ring
      rw [e]; exact hS1 n jj hn hcj
    · exact hS2 n jj (by omega) (bc zf32) rfl
  · exact hacc (ii, jj, ())

end Claims.SsdChunkScan
