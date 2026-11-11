import Proof.PhaseB

/-!
# One trip of the loop

Walk of `bBody`: the four loads (`cb`, `dA_cumsum`, `dt`, `x`), the weight chain
`cb · exp(dA_m − dA_k) · dt_k` rounded to `bfloat16`, the `tl.dot_add`, and the advance of the four
pointer tiles. It maps `BInv` at `n` to `BInv` at `n + 1`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- The hypotheses on the loop's constants: the index tensors, the width, the limits and the
`dA_cumsum` of the tile's rows. -/
structure BCtx {w : Nat} (d : Inst) (init : Mem) (k pm pn h : Nat) (offs_m : Tensor [BM] I32)
    (offs_n : Tensor [BN] I32) (offs_k : Tensor [BK] I32)
    (widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w)) (csw limit : IntV true w)
    (dA_cs_m : Tensor [BM] F32) : Prop where
  hw : 32 ≤ w
  hom : ∀ ii : Fin BM, (offs_m.get1 ii).toInt = (rowOf d k pm ii : Int)
  hon : ∀ jj : Fin BN, (offs_n.get1 jj).toInt = (colOf pn jj : Int)
  hok : ∀ l : Fin BK, (offs_k.get1 l).toInt = ((lo d k + l : Nat) : Int)
  hwiden : ∀ {n : Nat} (x : Tensor [n] I32) (i : Fin n), ((widen x).get1 i).toInt = (x.get1 i).toInt
  hcsw : csw.toInt = (d.cs : Int)
  hlim : limit.toInt = (lend d k : Int)
  hdAm : ∀ ii : Fin BM, RowOk d k pm ii → ∃ v : F32,
    init.read .f32 (d.dA.ptr (dtIdx d h (lc d k) (rowOf d k pm ii) : Int)).addr = some v ∧
      dA_cs_m.get1 ii = v

/-- **One trip of the loop** maps `BInv` at `n` to `BInv` at `n + 1`. -/
theorem bstep_safe {w : Nat} {E : Env} {d : Inst} {init : Mem} {k pm pn h : Nat}
    (S : Std d init E k pm pn h) {offs_m : Tensor [BM] I32} {offs_n : Tensor [BN] I32}
    {offs_k : Tensor [BK] I32} {widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w)}
    {csw limit : IntV true w} {dA_cs_m : Tensor [BM] F32}
    (C : BCtx d init k pm pn h offs_m offs_n offs_k widen csw limit dA_cs_m) {n : Nat}
    (hn : BK * n < lend d k) {kk : IntV true w} (hkk : kk.toInt = ((BK * n : Nat) : Int))
    {st : LoopSt} {m : Mem} (hI : BInv d init k pm pn h n st m) :
    Safe E (R d) ∅ (fun st' m' => BInv d init k pm pn h (n + 1) st' m') m
      (bBody d offs_m offs_n offs_k widen csw limit dA_cs_m kk st) := by
  obtain ⟨hd, hpre, dev, hk, hpm, hpn, hh⟩ := S
  obtain ⟨hvcb, hvx, hvdt, hvdA, hvseq, hvC, hvst, hvD, hvin, hvci, hvco, hvout, hmeta, hcb0⟩ := hpre
  obtain ⟨acc, cbp, xp, dtp, dAp⟩ := st
  obtain ⟨rfl, hcbp, hxp, hdtp, hdAp, hacc⟩ := hI
  have hlend := lend_le hd hk
  have hcsp := cs_pos hd
  have hcs14 : d.cs ≤ 2 ^ 14 := hd.cs_le
  unfold bBody
  have hw := C.hw
  have hpow : 2 ^ 31 ≤ 2 ^ (w - 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hnc : BK * n < d.cs := by omega
  have hcsK : (csw - kk).toInt = (d.cs : Int) - ((BK * n : Nat) : Int) := by
    have := IntV.sub_nat C.hcsw hkk (by omega) (by omega)
    rw [this]; push_cast [Nat.cast_sub (by omega : BK * n ≤ d.cs)]; ring
  have hlimK : (limit - kk).toInt = (lend d k : Int) - ((BK * n : Nat) : Int) := by
    have := IntV.sub_nat C.hlim hkk (by omega) (by omega)
    rw [this]; push_cast [Nat.cast_sub (by omega : BK * n ≤ lend d k)]; ring
  have hcsv : (IntV.ofNat d.cs : I32).toInt = (d.cs : Int) := IntV.I32.toInt_ofNat (by omega)
  have hlen_cb := cb_tile_len hd hk hh
  refine safe_load_tileAt d.cb (e := cbIdx d (lc d k) (grp d h) 0 0) (s0 := d.cs) (s1 := 1)
    (ρ := fun ii : Fin BM => rowOf d k pm ii) (κ := fun l : Fin BK => lo d k + BK * n + l)
    (M := d.cs) (N := d.cs) ?_ ?_ ?_ hvcb ?_ fun cbT hc1 hc2 => ?_
  · exact fun i j hi hj => hcbp i j hi
  · refine maskAt_and_off 0 ((BK * n : Nat) : Int) ?_ ?_ ?_ ?_
    · intro i; simp [C.hom i]
    · intro j
      rw [C.hwiden, C.hok]; push_cast; ring
    · simpa using hcsv
    · exact hcsK
  · exact Or.inr (Or.inr hlen_cb)
  · exact (Buf.tileAt_sub_bytes d.cb (Or.inr (Or.inr hlen_cb))).trans
      (bytes_sub_R hd.disj.1)
  -- the `dA_cumsum` entries of the trip
  have hmaskK : ∀ l : Fin BK,
      ((widen offs_k).ltS (csw - kk)).get1 l = decide (lo d k + BK * n + l < d.cs) := by
    intro l
    have h1 : ((widen offs_k).get1 l).toInt = ((lo d k + l : Nat) : Int) :=
      (C.hwiden offs_k l).trans (C.hok l)
    rw [Tensor.get1_ltS, IntV.lt_eq, h1, hcsK]
    simp only [decide_eq_decide]; push_cast; omega
  have hinK : ∀ l : Fin BK, lo d k + BK * n + l < d.cs →
      dtIdx d h (lc d k) (lo d k + BK * n + l) < d.H * (nc d * d.cs) :=
    fun l hl => dtIdx_lt hd hk hh hl
  refine safe_load_buf d.dA (g := fun i : Idx [BK] => dtIdx d h (lc d k) (lo d k + BK * n + i.1.val))
    (en := fun i : Idx [BK] => decide (lo d k + BK * n + i.1.val < d.cs)) ?_ ?_ hvdA ?_
    fun dAk hk1 hk2 => ?_
  · refine Buf.addrs_of_get d.dA (fun i => hmaskK i.1) (fun i hi => ?_)
    have e : dtIdx d h (lc d k) 0 + (lo d k + BK * n + i.1.val) =
        dtIdx d h (lc d k) (lo d k + BK * n + i.1.val) := by unfold dtIdx; ring
    exact (hdAp i.1).trans (by rw [e])
  · intro i hi
    exact lt_of_lt_of_le (hinK i.1 (by simpa using hi)) hd.dA_len
  · refine (d.dA.elems_sub_bytes ?_).trans (bytes_sub_R hd.disj.2.2.2.1)
    rintro x ⟨i, hi, rfl⟩
    exact lt_of_lt_of_le (hinK i.1 (by simpa using hi)) hd.dA_len
  refine safe_sub (fun xt hxt => ?_)
  refine safe_math (fun i => dev.expTotal _) (fun ex hex => ?_)
  refine safe_mul (fun q1 hq1 => ?_)
  -- the `dt` entries of the trip
  refine safe_load_buf d.dt (g := fun i : Idx [BK] => dtIdx d h (lc d k) (lo d k + BK * n + i.1.val))
    (en := fun i : Idx [BK] => decide (lo d k + BK * n + i.1.val < d.cs)) ?_ ?_ hvdt ?_
    fun dtk hk3 hk4 => ?_
  · refine Buf.addrs_of_get d.dt (fun i => hmaskK i.1) (fun i hi => ?_)
    have e : dtIdx d h (lc d k) 0 + (lo d k + BK * n + i.1.val) =
        dtIdx d h (lc d k) (lo d k + BK * n + i.1.val) := by unfold dtIdx; ring
    exact (hdtp i.1).trans (by rw [e])
  · intro i hi
    exact lt_of_lt_of_le (hinK i.1 (by simpa using hi)) hd.dt_len
  · refine (d.dt.elems_sub_bytes ?_).trans (bytes_sub_R hd.disj.2.2.1)
    rintro x ⟨i, hi, rfl⟩
    exact lt_of_lt_of_le (hinK i.1 (by simpa using hi)) hd.dt_len
  refine safe_mul (fun q2 hq2 => ?_)
  refine safe_cvt ⟨by simp [Format.Std], by simp [Format.Std], Or.inl rfl⟩ (fun cb4 hcb4 => ?_)
  refine safe_convert (fun _ => exists_conv_f32_bf16 _ _) (fun z hz => ?_)
  have hz0 : ∀ i j, z.get2 i j = FloatV.ofPat 0 := fun i j =>
    convRel_zf32 (by have := hz (i, j, ()); simp only [tl.full] at this; exact this)
  have hlend0 : 0 < lend d k := by omega
  have hlenx : lend d k = 0 ∨ d.P = 0 ∨
      xIdx d (lc d k * d.cs) h 0 + (lend d k - 1) * d.xT + (d.P - 1) * 1 < d.x.len :=
    Or.inr (Or.inr (x_tile_len hd hk hh hlend0))
  have hPv : (IntV.ofNat d.P : I32).toInt = (d.P : Int) := IntV.I32.toInt_ofNat (P_lt31 hd)
  refine safe_load_tileAt d.x (e := xIdx d (lc d k * d.cs) h 0) (s0 := d.xT) (s1 := 1)
    (ρ := fun l : Fin BK => lo d k + BK * n + l) (κ := fun jj : Fin BN => colOf pn jj)
    (M := lend d k) (N := d.P) ?_ ?_ ?_ hvx ?_ fun xT hx1 hx2 => ?_
  · exact fun l jj hl hj => hxp l jj hj
  · refine maskAt_and_off ((BK * n : Nat) : Int) 0 ?_ ?_ ?_ ?_
    · intro l
      rw [C.hwiden, C.hok]; push_cast; ring
    · intro jj; simp [C.hon jj]
    · exact hlimK
    · simpa using hPv
  · exact hlenx
  · exact (Buf.tileAt_sub_bytes d.x hlenx).trans (bytes_sub_R hd.disj.2.1)
  refine safe_dot_add (dev.dotTotal _ _ _) (fun r hr => ?_)
  refine safe_ret ?_
  have hs1 : ((IntV.ofNat BK : I32) * IntV.ofNat 1).toInt = (BK : Int) := by
    have := IntV.mul_nat (a := (IntV.ofNat BK : I32)) (b := (IntV.ofNat 1 : I32)) (x := BK) (y := 1)
      (IntV.I32.toInt_ofNat (by norm_num [BK])) (IntV.I32.toInt_ofNat (by norm_num))
      (by norm_num [BK])
    simpa using this
  have hsx : ((IntV.ofNat BK : I32) * IntV.ofNat d.xT).toInt = ((BK * d.xT : Nat) : Int) := by
    have hxT : d.xT < 2 ^ 31 := by
      have := hd.x_off; have : d.xT ≤ (2 * d.cs + 64) * d.xT := Nat.le_mul_of_pos_left _ (by omega)
      omega
    refine IntV.mul_nat (IntV.I32.toInt_ofNat (by norm_num [BK])) (IntV.I32.toInt_ofNat hxT) ?_
    have : BK * d.xT ≤ (2 * d.cs + 64) * d.xT := Nat.mul_le_mul_right _ (by unfold BK; omega)
    have := hd.x_off; show BK * d.xT < 2 ^ (32 - 1); omega
  refine ⟨rfl, ?_, ?_, ?_, ?_, ?_⟩
  · intro ii l hrow
    show (cbp.get2 ii l).offset _ = _
    rw [hcbp ii l hrow, Buf.ptr_offset, hs1]; congr 1; push_cast; ring
  · intro l jj hj
    show (xp.get2 l jj).offset _ = _
    rw [hxp l jj hj, Buf.ptr_offset, hsx]; congr 1; push_cast; ring
  · intro l
    show (dtp.get1 l).offset _ = _
    rw [hdtp l, Buf.ptr_offset, hs1]; congr 1; push_cast; ring
  · intro l
    show (dAp.get1 l).offset _ = _
    rw [hdAp l, Buf.ptr_offset, hs1]; congr 1; push_cast; ring
  · intro hV ii jj hri hcj
    have hrow : lo d k ≤ rowOf d k pm ii ∧ rowOf d k pm ii < lend d k :=
      ⟨by unfold rowOf; omega, hri⟩
    obtain ⟨ht, hchk, hmod, hlOff⟩ := tok_of_row hd hk hrow
    have ht' : tokOf d k pm ii < d.T := ht
    have hchk' : chk d (tokOf d k pm ii) = lc d k := hchk
    have hmod' : tokOf d k pm ii % d.cs = rowOf d k pm ii := hmod
    have hcs' : rowOf d k pm ii < d.cs := by omega
    have hlT : lc d k * d.cs + lo d k ≤ lTok d (tokOf d k pm ii) := lTok_tile hd hk hri
    have hL : tokOf d k pm ii < lc d k * d.cs + lend d k := by unfold tokOf; omega
    -- the weights of the trip
    have hw : ∀ l : Fin BK, TokW m d (tokOf d k pm ii) h (lc d k * d.cs + lo d k + BK * n + l)
        (cb4.get2 ii l) := by
      intro l
      obtain ⟨dAm, hdAm1, hdAm2⟩ := C.hdAm ii hri
      have hkT : kToks d (tokOf d k pm ii) =
          Finset.Ico (lc d k * d.cs + lo d k) (lc d k * d.cs + rowOf d k pm ii + 1) :=
        kToks_eq hd hk hrow
      have hs1' : chk d (tokOf d k pm ii) * d.cs ≤ lc d k * d.cs + lo d k + BK * n + l := by
        rw [hchk']; omega
      have hu : ∀ P : Prop, (lo d k + BK * n + l ≤ rowOf d k pm ii → P) →
          (lc d k * d.cs + lo d k + BK * n + l ≤ tokOf d k pm ii → P) := by
        intro P hP hle; apply hP; unfold tokOf at hle; omega
      refine weight_chain (P := E.P) dev.expAcc hd hV hcb0 ht' hh hcj hs1'
        (cbL := cbT.get2 ii l) (dAm := dAm) (dAk := dAk.get1 l) (dtk := dtk.get1 l)
        (xt := xt.get2 ii l) (ex := ex.get2 ii l) (q1 := q1.get2 ii l) (q2 := q2.get2 ii l)
        (q3 := (tl.where (Tensor.le ((kk + widen offs_k).row BM) ((widen offs_m).col BK))
          q2 (bc zf32)).get2 ii l) (w16 := cb4.get2 ii l) ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
      · intro hst
        rw [hkT, Finset.mem_Ico]; unfold tokOf at hst; omega
      · intro hst
        have hu' : lo d k + BK * n + l < d.cs := by unfold tokOf at hst; omega
        have e : cbIdx d (chk d (tokOf d k pm ii)) (grp d h) (tokOf d k pm ii % d.cs)
            (lc d k * d.cs + lo d k + BK * n + l - chk d (tokOf d k pm ii) * d.cs) =
            cbIdx d (lc d k) (grp d h) 0 0 + rowOf d k pm ii * d.cs + (lo d k + BK * n + l) * 1 := by
          rw [hchk', hmod']; unfold cbIdx
          have : lc d k * d.cs + lo d k + BK * n + l - lc d k * d.cs = lo d k + BK * n + l := by omega
          rw [this]; ring
        rw [e]; exact hc1 ii l hcs' hu'
      · rw [hchk', hmod']; exact hdAm1
      · intro hst
        have hu' : lo d k + BK * n + l < d.cs := by unfold tokOf at hst; omega
        have e : lc d k * d.cs + lo d k + BK * n + l - chk d (tokOf d k pm ii) * d.cs =
            lo d k + BK * n + l := by rw [hchk']; omega
        rw [e, hchk']; exact hk1 (l, ()) (by simpa using hu')
      · intro hst
        have hu' : lo d k + BK * n + l < d.cs := by unfold tokOf at hst; omega
        have e : lc d k * d.cs + lo d k + BK * n + l - chk d (tokOf d k pm ii) * d.cs =
            lo d k + BK * n + l := by rw [hchk']; omega
        rw [e, hchk']; exact hk3 (l, ()) (by simpa using hu')
      · have h := hxt (ii, l, ())
        have e1 : ((dA_cs_m.col BK).get (ii, l, ())) = dAm := hdAm2
        rw [e1] at h; exact h
      · exact hex (ii, l, ())
      · exact hq1 (ii, l, ())
      · exact hq2 (ii, l, ())
      · have hmask : (Tensor.le ((kk + widen offs_k).row BM) ((widen offs_m).col BK)).get
            (ii, l, ()) = decide (lo d k + BK * n + l ≤ rowOf d k pm ii) := by
          have hlt31 : BK * n + (lo d k + l) < 2 ^ 31 := by
            have := lo_lt hd hk; have := hd.cs_le; have := l.isLt
            unfold BK at *; omega
          have h1 : (kk + (widen offs_k).get1 l).toInt = ((BK * n + (lo d k + l) : Nat) : Int) :=
            IntV.add_nat hkk ((C.hwiden offs_k l).trans (C.hok l)) (lt_of_lt_of_le hlt31 hpow)
          show IntV.le (kk + (widen offs_k).get1 l) ((widen offs_m).get1 ii) = _
          rw [IntV.le_eq, h1, C.hwiden, C.hom]
          simp only [decide_eq_decide]; push_cast; omega
        show (tl.where (Tensor.le ((kk + widen offs_k).row BM) ((widen offs_m).col BK)) q2
          (bc zf32)).get (ii, l, ()) = _
        have hiff : lc d k * d.cs + lo d k + BK * n + l ≤ tokOf d k pm ii ↔
            lo d k + BK * n + l ≤ rowOf d k pm ii := by unfold tokOf; omega
        unfold tl.where
        show (if (Tensor.le ((kk + widen offs_k).row BM) ((widen offs_m).col BK)).get (ii, l, ()) =
          true then q2.get (ii, l, ()) else (bc zf32 : Tensor [BM, BK] F32).get (ii, l, ())) = _
        rw [hmask]
        by_cases hle : lo d k + BK * n + l ≤ rowOf d k pm ii
        · rw [ite_eq_left (hiff.mpr hle)]; simp only [hle, decide_true, ite_true]; rfl
        · rw [ite_eq_right (fun h' => hle (hiff.mp h'))]; simp only [hle, decide_false, Bool.false_eq_true, ite_false]; rfl
      · exact hcb4 (ii, l, ())
    have hxl : ∀ l : Fin BK,
        (lo d k + BK * n + l < lend d k →
          m.read .bf16 (d.x.ptr (xIdx d (lc d k * d.cs + (lo d k + BK * n + l)) h (colOf pn jj) : Int)).addr
            = some (xT.get2 l jj)) ∧
        (¬ lo d k + BK * n + l < lend d k → xT.get2 l jj = FloatV.ofPat 0) := by
      intro l
      refine ⟨fun hl => ?_, fun hl => ?_⟩
      · have e : xIdx d (lc d k * d.cs) h 0 + (lo d k + BK * n + l) * d.xT + colOf pn jj * 1 =
            xIdx d (lc d k * d.cs + (lo d k + BK * n + l)) h (colOf pn jj) := by
          unfold xIdx; ring
        have := hx1 l jj hl hcj
        rw [e] at this
        exact this
      · exact (hx2 l jj (fun h' => hl h'.1) z rfl).trans (hz0 l jj)
    refine loop_step (P := E.P) dev.dotAcc hd hV ht' hh hcj hchk' hlT hL hlend.1 (by omega)
      (show n < (d.cs + 32 - 1) / 32 by have := hn; simp only [BK] at this; omega)
      (z := (Tensor.full (cfgBF16.acc.ofBits 0) : Tensor [BM, BN] F32)) ?_ ?_ hw hxl (hacc hV ii jj hri hcj)
    · rcases hr with h1 | ⟨dd, h1, h2⟩
      · exact Or.inl h1
      · exact Or.inr ⟨dd, h1, by rw [dev.fusion] at h2; exact h2⟩
    · exact DType.val?_zero_f32

end Claims.SsdChunkScan
