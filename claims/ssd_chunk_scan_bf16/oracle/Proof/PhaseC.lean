import Proof.PhaseBLoop

/-!
# Phase C: the epilogue

Walk of `phaseC`: `D` and the residual `x` tile, `acc + x·D`, the conversion to `bfloat16` and the
store. It leaves `Qt` (the tile of `out` holds the values `final_fact` bounds).
-/

namespace Claims.SsdChunkScan

open TVA

/-- **Phase C is safe**: the epilogue leaves the tile's values (`Qt`). -/
theorem phaseC_safe {w : Nat} {E : Env} {d : Inst} {init : Mem} {k pm pn h : Nat}
    (S : Std d init E k pm pn h) {x_ptr out_ptr D_ptr : Ptr .bf16} {pid_h pid_m pid_n c_off : I32}
    {offs_m : Tensor [BM] I32} {offs_n : Tensor [BN] I32}
    {widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w)} {limit : IntV true w}
    {acc : Tensor [BM, BN] F32} (hw : 32 ≤ w)
    (hx : x_ptr = d.x.ptr (xIdx d (lc d k * d.cs) h 0 : Int))
    (hout : out_ptr = d.out.ptr ((lc d k * d.cs * (d.H * d.P) + h * d.P : Nat) : Int))
    (hD : D_ptr = d.D.ptr 0) (hph : pid_h.toInt = (h : Int)) (hpm : pid_m.toInt = (pm : Int))
    (hpn : pid_n.toInt = (pn : Int)) (hco : c_off.toInt = (lo d k : Int))
    (hom : ∀ ii : Fin BM, (offs_m.get1 ii).toInt = (rowOf d k pm ii : Int))
    (hon : ∀ jj : Fin BN, (offs_n.get1 jj).toInt = (colOf pn jj : Int))
    (hwiden : ∀ {n : Nat} (x : Tensor [n] I32) (i : Fin n),
      ((widen x).get1 i).toInt = (x.get1 i).toInt)
    (hlim : limit.toInt = (lend d k : Int)) (hacc : BPost d init k pm pn h acc) :
    Safe E (R d) (Wt d k pm pn h) (fun _ m => Qt d init k pm pn h m) init
      (phaseC d x_ptr out_ptr D_ptr pid_h pid_m pid_n c_off offs_m offs_n widen limit acc) := by
  obtain ⟨hd, hpre, dev, hk, hpm', hpn', hh⟩ := S
  obtain ⟨hvcb, hvx, hvdt, hvdA, hvseq, hvC, hvst, hvD, hvin, hvci, hvco, hvout, hmeta, hcb0⟩ := hpre
  unfold phaseC
  have hlend := lend_le hd hk
  have hcsp := cs_pos hd
  have hcs14 := hd.cs_le
  have hlo := lo_lt hd hk
  have hP256 := P_small hd
  have hpow : 2 ^ 31 ≤ 2 ^ (w - 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hpmB : BM * pm < d.cs := by
    unfold nmt at hpm'; simp only [BM] at *; omega
  have hpnB : BN * pn < d.P + 64 := by
    unfold npn at hpn'; simp only [BN] at *; omega
  have hone : (IntV.ofNat 1 : I32).toInt = ((1 : Nat) : Int) := IntV.I32.toInt_ofNat (by norm_num)
  have hbm : (pid_m * (IntV.ofNat BM : I32) + c_off).toInt = ((pm * BM + lo d k : Nat) : Int) := by
    have h1 : (pid_m * (IntV.ofNat BM : I32)).toInt = ((pm * BM : Nat) : Int) :=
      IntV.mul_nat hpm (IntV.I32.toInt_ofNat (by norm_num [BM])) (by simp only [BM] at *; omega)
    exact IntV.add_nat h1 hco (by simp only [BM] at *; omega)
  have hoo : ∀ ii : Fin BM, (((pid_m * (IntV.ofNat BM : I32) + c_off) + tl.arange 0 BM :
      Tensor [BM] I32).get1 ii).toInt = (rowOf d k pm ii : Int) := fun ii => by
    have := IntV.I32.toInt_sadd_arange (n := BM) hbm (by simp only [BM] at *; omega) ii
    rw [this]; unfold rowOf; push_cast; ring
  have hbn : (pid_n * (IntV.ofNat BN : I32)).toInt = ((pn * BN : Nat) : Int) :=
    IntV.mul_nat hpn (IntV.I32.toInt_ofNat (by norm_num [BN])) (by simp only [BN] at *; omega)
  have hnn : ∀ jj : Fin BN, ((pid_n * (IntV.ofNat BN : I32) + tl.arange 0 BN :
      Tensor [BN] I32).get1 jj).toInt = (colOf pn jj : Int) := fun jj => by
    have := IntV.I32.toInt_sadd_arange (n := BN) hbn (by simp only [BN] at *; omega) jj
    rw [this]; unfold colOf; push_cast; ring
  -- `D[h]`
  have hhD : h < d.D.len := lt_of_lt_of_le hh hd.D_len
  have hDp : D_ptr + pid_h * (IntV.ofNat 1 : I32) = d.D.ptr (h : Int) := by
    have : (pid_h * (IntV.ofNat 1 : I32)).toInt = ((h * 1 : Nat) : Int) :=
      IntV.mul_nat hph hone (show h * 1 < 2 ^ (32 - 1) by have := hd.H_lt; omega)
    show D_ptr.offset _ = _
    rw [hD, Buf.ptr_offset, this]; congr 1; push_cast; ring
  obtain ⟨D0, hD0⟩ := d.D.read_some hvD hhD
  refine safe_load1 (p := D_ptr + pid_h * (IntV.ofNat 1 : I32)) (v0 := D0) (by rw [hDp]; exact hD0)
    (by rw [hDp]; exact (d.D.span_sub hhD).trans (bytes_sub_R hd.disj.2.2.2.2.2.2.2.1)) ?_
  refine safe_cvt ⟨by simp [Format.Std], by simp [Format.Std], Or.inl rfl⟩ (fun Dc hDc => ?_)
  -- the residual `x` tile
  refine safe_convert (fun _ => exists_conv_f32_bf16 _ _) (fun z hz => ?_)
  have hz0 : ∀ i j, z.get2 i j = FloatV.ofPat 0 := fun i j =>
    convRel_zf32 (by have := hz (i, j, ()); simp only [tl.full] at this; exact this)
  have hxTv : (IntV.ofNat d.xT : I32).toInt = (d.xT : Int) := by
    refine IntV.I32.toInt_ofNat ?_
    have := hd.x_off; have : d.xT ≤ (2 * d.cs + 64) * d.xT := Nat.le_mul_of_pos_left _ (by omega)
    omega
  have hPv : (IntV.ofNat d.P : I32).toInt = (d.P : Int) := IntV.I32.toInt_ofNat (by omega)
  have hxfit : (lend d k - 1) * d.xT + (d.P - 1) * 1 < 2 ^ (32 - 1) := by
    rcases Nat.eq_zero_or_pos (lend d k) with h0 | h0
    · rw [h0]; simp only [Nat.zero_sub, Nat.zero_mul]; omega
    have h1 : lend d k * d.xT ≤ (2 * d.cs + 64) * d.xT := Nat.mul_le_mul_right _ (by omega)
    have e : (lend d k - 1) * d.xT + d.xT = lend d k * d.xT := by
      rw [← Nat.succ_mul]; congr 1; omega
    have := hd.x_off; have := hd.xT_ge
    have h3 : d.P ≤ d.H * d.P := Nat.le_mul_of_pos_left _ hd.H_pos
    have := hd.P_pos
    omega
  have hxp := Tile2.ptr_lane_on (d := .bf16) d.x hx (rows := offs_m) (cols := offs_n)
    (a0 := (IntV.ofNat d.xT : I32)) (a1 := (IntV.ofNat 1 : I32))
    (ρ := fun i : Fin BM => rowOf d k pm i) (κ := fun j : Fin BN => colOf pn j)
    (M := lend d k) (N := d.P) hom hon hxTv hone hxfit
  have hxlen : lend d k = 0 ∨ d.P = 0 ∨
      xIdx d (lc d k * d.cs) h 0 + (lend d k - 1) * d.xT + (d.P - 1) * 1 < d.x.len := by
    rcases Nat.eq_zero_or_pos (lend d k) with h0 | h0
    · exact Or.inl h0
    · exact Or.inr (Or.inr (x_tile_len hd hk hh h0))
  refine safe_load_tileAt d.x (e := xIdx d (lc d k * d.cs) h 0) (s0 := d.xT) (s1 := 1)
    (ρ := fun i : Fin BM => rowOf d k pm i) (κ := fun j : Fin BN => colOf pn j)
    (M := lend d k) (N := d.P) ?_ ?_ ?_ hvx ?_ fun xr0 hx1 hx2 => ?_
  · exact hxp
  · exact maskAt_and2 (fun ii => (hwiden offs_m ii).trans (hom ii)) hon hlim hPv
  · exact hxlen
  · exact (Buf.tileAt_sub_bytes d.x hxlen).trans (bytes_sub_R hd.disj.2.1)
  refine safe_cvt ⟨by simp [Format.Std], by simp [Format.Std], Or.inl rfl⟩ (fun xr hxr => ?_)
  refine safe_mul_add (fun accF hma => ?_)
  refine safe_cvt ⟨by simp [Format.Std], by simp [Format.Std], Or.inl rfl⟩ (fun out16 ho => ?_)
  refine safe_last ?_
  have hHP31 : d.H * d.P < 2 ^ 31 := by
    have := hd.out_off
    have : d.H * d.P ≤ (2 * d.cs + 64) * (d.H * d.P) := Nat.le_mul_of_pos_left _ (by omega)
    omega
  have hHPv : (IntV.ofNat (d.H * d.P) : I32).toInt = ((d.H * d.P : Nat) : Int) :=
    IntV.I32.toInt_ofNat hHP31
  have hPHP : d.P ≤ d.H * d.P := Nat.le_mul_of_pos_left _ hd.H_pos
  have hofit0 : (lend d k - 1) * (d.H * d.P) + (d.P - 1) * 1 < 2 ^ (32 - 1) := by
    rcases Nat.eq_zero_or_pos (lend d k) with h0 | h0
    · rw [h0]; simp only [Nat.zero_sub, Nat.zero_mul]; omega
    have h1 : lend d k * (d.H * d.P) ≤ (2 * d.cs + 64) * (d.H * d.P) :=
      Nat.mul_le_mul_right _ (by omega)
    have e : (lend d k - 1) * (d.H * d.P) + d.H * d.P = lend d k * (d.H * d.P) := by
      rw [← Nat.succ_mul]; congr 1; omega
    have := hd.out_off; have := hd.P_pos
    omega
  have hofit1 : (lend d k - 1) * (d.H * d.P) < 2 ^ (32 - 1) := by omega
  have hofit2 : (d.P - 1) * 1 < 2 ^ (32 - 1) := by omega
  have hop := Tile2.ptr_lane_sum_of_on (d := .bf16) d.out hout
    (x := (IntV.ofNat (d.H * d.P) : I32) * ((pid_m * (IntV.ofNat BM : I32) + c_off) +
      tl.arange 0 BM : Tensor [BM] I32).col BN)
    (y := (pid_n * (IntV.ofNat BN : I32) + tl.arange 0 BN : Tensor [BN] I32).row BM *
      (IntV.ofNat 1 : I32))
    (ρ := fun i : Fin BM => rowOf d k pm i) (κ := fun j : Fin BN => colOf pn j)
    (M := lend d k) (N := d.P) (s0 := d.H * d.P) (s1 := 1)
    (fun i j hi => Tile2.mulCol_toInt_on hoo hHPv hofit1 i j hi)
    (fun i j hj => Tile2.rowMul_toInt_on hnn hone hofit2 i j hj) hofit0
  have houtlen : lend d k = 0 ∨ d.P = 0 ∨ lc d k * d.cs * (d.H * d.P) + h * d.P +
      (lend d k - 1) * (d.H * d.P) + (d.P - 1) * 1 < d.out.len := by
    rcases Nat.eq_zero_or_pos (lend d k) with h0 | h0
    · exact Or.inl h0
    refine Or.inr (Or.inr ?_)
    have e1 : (lc d k * d.cs + (lend d k - 1)) * (d.H * d.P) ≤ (d.T - 1) * (d.H * d.P) := 
      Nat.mul_le_mul_right _ (by omega)
    have e1' : (lc d k * d.cs + (lend d k - 1)) * (d.H * d.P) =
        lc d k * d.cs * (d.H * d.P) + (lend d k - 1) * (d.H * d.P) := Nat.add_mul _ _ _
    have e2 : (h + 1) * d.P ≤ d.H * d.P := Nat.mul_le_mul_right _ hh
    have e2' : (h + 1) * d.P = h * d.P + d.P := by rw [Nat.add_mul, Nat.one_mul]
    have e3 : (d.T - 1) * (d.H * d.P) + d.H * d.P = d.T * (d.H * d.P) := by
      rw [← Nat.succ_mul]; congr 1; have := T_pos hd; omega
    have := hd.out_len; have := hd.P_pos
    omega
  refine safe_store_tileAt d.out (e := lc d k * d.cs * (d.H * d.P) + h * d.P) (s0 := d.H * d.P)
    (s1 := 1) (ρ := fun i : Fin BM => rowOf d k pm i) (κ := fun j : Fin BN => colOf pn j)
    (M := lend d k) (N := d.P) ?_ ?_ ?_ ?_ ?_ ?_ hvout (subset_refl _) fun m' hread hframe hpres => ?_
  · exact hop
  · exact maskAt_and2 (fun ii => (hwiden _ ii).trans (hoo ii)) hnn hlim hPv
  · exact houtlen
  · exact Tile2.inj_rowMajor (by norm_num) (by omega)
  · intro i i' _ _ h'; exact Fin.ext (by unfold rowOf at h'; omega)
  · intro j j' _ _ h'; exact Fin.ext (by unfold colOf at h'; omega)
  refine safe_ret ?_
  intro ii jj hii hjj hri hcj
  have hrow : lo d k ≤ rowOf d k pm ii ∧ rowOf d k pm ii < lend d k :=
    ⟨by unfold rowOf; omega, hri⟩
  have ht' : tokOf d k pm ii < d.T := (tok_of_row hd hk hrow).1
  have hidx : lc d k * d.cs * (d.H * d.P) + h * d.P + rowOf d k pm ii * (d.H * d.P) +
      colOf pn jj * 1 = outIdx d (tokOf d k pm ii) h (colOf pn jj) := by
    unfold outIdx tokOf; ring
  refine ⟨out16.get2 ⟨ii, hii⟩ ⟨jj, hjj⟩, ?_, fun hV => ?_⟩
  · unfold rdB
    rw [← hidx]
    exact hread ⟨ii, hii⟩ ⟨jj, hjj⟩ hri hcj
  · have hx0 : init.read .bf16 (d.x.ptr (xIdx d (tokOf d k pm ii) h (colOf pn jj) : Int)).addr =
        some (xr0.get2 ⟨ii, hii⟩ ⟨jj, hjj⟩) := by
      have e : xIdx d (lc d k * d.cs) h 0 + rowOf d k pm ii * d.xT + colOf pn jj * 1 =
          xIdx d (tokOf d k pm ii) h (colOf pn jj) := by unfold xIdx tokOf; ring
      rw [← e]; exact hx1 ⟨ii, hii⟩ ⟨jj, hjj⟩ hri hcj
    refine final_fact hd hV ht' hh hcj (acc := acc.get2 ⟨ii, hii⟩ ⟨jj, hjj⟩) (D := Dc.item)
      (xr := xr.get2 ⟨ii, hii⟩ ⟨jj, hjj⟩) (r := accF.get2 ⟨ii, hii⟩ ⟨jj, hjj⟩)
      (hacc hV ⟨ii, hii⟩ ⟨jj, hjj⟩ hri hcj) hD0 (hDc ()) hx0 (hxr (⟨ii, hii⟩, ⟨jj, hjj⟩, ())) ?_
      (ho (⟨ii, hii⟩, ⟨jj, hjj⟩, ()))
    rcases hma (⟨ii, hii⟩, ⟨jj, hjj⟩, ()) with h1 | ⟨-, h2⟩
    · exact Or.inl h1
    · exact Or.inr h2
