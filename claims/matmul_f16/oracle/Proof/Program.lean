import TVA.Proof
import Claim
import Proof.Tiles

/-!
# `safe_program`

`safe_program`: one program of the launch is safe, walked line by line.
-/


namespace Claims.Matmul

open TVA

set_option maxHeartbeats 2000000

/-- **One program** of the launch is safe from `init`: it reads inside `R`, writes inside `W`, and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) (P : Platform) (init : Mem) (hpre : Pre d init)
    (i : Fin (launch d).size)
    (hdot : Device.DotF16 P)
    (htot : Device.DotTotal P
      { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  rw [body_eq]
  simp only [tl.program_id, launch_pid]
  refine safe_idiv1 (w := 32) ?_ ?_ ?_
  · rw [toInt_add_ofNat (by have := hd.M_le; omega)]
    show (0 : Int) ≤ ((d.M : Nat) : Int) + ((128 - 1 : Nat) : Int)
    have h127 : ((128 - 1 : Nat) : Int) = 127 := by norm_num
    rw [h127]; omega
  · rw [i32_128]; norm_num
  · intro r1 h1
    refine safe_idiv1 (w := 32) ?_ ?_ ?_
    · rw [toInt_add_ofNat (by have := hd.N_le; omega)]
      show (0 : Int) ≤ ((d.N : Nat) : Int) + ((256 - 1 : Nat) : Int)
      have h255 : ((256 - 1 : Nat) : Int) = 255 := by norm_num
      rw [h255]; omega
    · rw [i32_256]; norm_num
    · intro r2 h2
      have hn : r2.item.toInt = (npn d : Int) := num_pid_n_val hd h2
      have hy3 : 0 < ((IntV.ofNat 8 : I32) * r2.item).toInt := eight_mul_pos hd hn
      refine safe_idiv1 (w := 32) ?_ ?_ ?_
      · have hp := i32_pid d hd i
        rw [hp]; exact_mod_cast Nat.zero_le (i : Nat)
      · exact hy3
      · intro r3 h3
        have hm : r1.item.toInt = (npm d : Int) := num_pid_m_val hd h1
        have hg : r3.item.toInt = (grp d i : Int) := group_val hd i hn h3
        refine safe_irem1 (w := 32) ?_ ?_ ?_
        · have hp := i32_pid d hd i
          rw [hp]; exact_mod_cast Nat.zero_le (i : Nat)
        · exact hy3
        · intro r4 h4
          have hx5 : 0 ≤ r4.item.toInt := rem_pos hd i hn h4
          have hy5 : 0 < (tl.iminimum (Tensor.scalar (r1.item - r3.item * (IntV.ofNat 8 : I32)))
              (Tensor.scalar (IntV.ofNat 8 : I32))).item.toInt := gsize_pos' hd i hm hg
          refine safe_irem1 (w := 32) ?_ ?_ ?_
          · exact hx5
          · exact hy5
          · intro r5 h5
            refine safe_irem1 (w := 32) ?_ ?_ ?_
            · have hp := i32_pid d hd i
              rw [hp]; exact_mod_cast Nat.zero_le (i : Nat)
            · exact hy3
            · intro r6 h6
              have hx7 : 0 ≤ r6.item.toInt := rem_pos hd i hn h6
              refine safe_idiv1 (w := 32) ?_ ?_ ?_
              · exact hx7
              · exact hy5
              · intro r7 h7
                have ho : r5.item.toInt = ((pidM d i - fpm d i : Nat) : Int) :=
                  off_val hd i hn hm hg h4 h5
                have hpn : r7.item.toInt = ((pidN d i : Nat) : Int) :=
                  pidn_val hd i hn hm hg h6 h7
                refine safe_assume (le_pidm hd i hg ho) ?_
                refine safe_assume (le_pidn hd i hpn) ?_
                refine safe_assume (lt_stride d.stride_am hd.am.1 hd.am.2) ?_
                refine safe_assume (lt_stride d.stride_ak hd.ak.1 hd.ak.2) ?_
                refine safe_assume (lt_stride d.stride_bn hd.bn.1 hd.bn.2) ?_
                refine safe_assume (lt_stride d.stride_bk hd.bk.1 hd.bk.2) ?_
                refine safe_assume (lt_stride d.stride_cm hd.cm.1 hd.cm.2) ?_
                refine safe_assume (lt_stride d.stride_cn hd.cn.1 hd.cn.2) ?_
                refine safe_irem (s := [128]) ?_ ?_
                · intro idx
                  obtain ⟨r, u⟩ := idx
                  cases u
                  obtain ⟨q, hq, _⟩ := IntV.rem?_of_nonneg
                    (a := (((r3.item * IntV.ofNat 8 + r5.item) : I32) * (IntV.ofNat 128 : I32)
                      + tl.arange 0 128).get (r, ()))
                    (b := (IntV.ofNat d.M : I32))
                    (by rw [offs_row_val hd i hg ho r]; omega)
                    (by show 0 < (IntV.ofNat d.M : I32).toInt; rw [i32_M d hd]
                        exact_mod_cast M_pos i)
                  exact ⟨q, hq⟩
                · intro res hres
                  refine safe_irem (s := [256]) ?_ ?_
                  · intro idx
                    obtain ⟨c, u⟩ := idx
                    cases u
                    obtain ⟨q, hq, _⟩ := IntV.rem?_of_nonneg
                      (a := ((r7.item * (IntV.ofNat 256 : I32) + tl.arange 0 256).get (c, ())))
                      (b := (IntV.ofNat d.N : I32))
                      (by rw [offs_col_val hd i hpn c]; omega)
                      (by show 0 < (IntV.ofNat d.N : I32).toInt; rw [i32_N d hd]
                          exact_mod_cast N_pos i)
                    exact ⟨q, hq⟩
                  · intro res2 hres2
                    refine safe_idiv1 (w := 32) ?_ ?_ ?_
                    · rw [toInt_add_ofNat (by have := hd.K_le; omega)]
                      show (0 : Int) ≤ ((d.K : Nat) : Int) + ((64 - 1 : Nat) : Int)
                      have h63 : ((64 - 1 : Nat) : Int) = 63 := by norm_num
                      rw [h63]; omega
                    · show 0 < (IntV.ofNat 64 : I32).toInt
                      rw [IntV.I32.toInt_ofNat (by norm_num)]; norm_num
                    · intro rt ht
                      have hrt : rt.item.toInt = (steps d.K : Int) := steps_val hd ht
                      have htz : (IntV.ofInt 0 : I32).toInt = 0 := ofInt_zero
                      have hto : (IntV.ofInt 1 : I32).toInt = 1 := by
                        rw [IntV.I32.toInt_ofInt (by norm_num)]
                      have hT : tl.tripCount (IntV.ofInt 0 : I32).toInt rt.item.toInt
                          (IntV.ofInt 1 : I32).toInt = steps d.K := by
                        show ((rt.item.toInt - 0 + 1 - 1) / 1).toNat = steps d.K
                        rw [hrt]
                        have h2 : ((steps d.K : Int) - 0 + 1 - 1) = (steps d.K : Int) := by ring
                        rw [h2]
                        simp
                      rw [show W d init i =
                        (iterW (fun _ => (∅ : Set Int)) (tl.tripCount (IntV.ofInt 0 : I32).toInt
                          rt.item.toInt (IntV.ofInt 1 : I32).toInt) ∪ W d init i) from by
                          rw [hT]; simp [iterW]]
                      refine safe_forRange (I := InInv d i init) (Wt := fun _ => (∅ : Set Int))
                        (Wk := W d init i) ?hrange ?hWt ?hWk ?h0 ?hstep ?hk
                      case hrange =>
                        unfold tl.RangeOk
                        rw [htz, hrt, hto]
                        have hsteple : (steps d.K : Int) ≤ 128 := by
                          exact_mod_cast steps_le d hd
                        have hbnd : IntV.maxInt true 32 = 2 ^ 31 - 1 := by
                          simp [IntV.maxInt]
                        rw [hbnd]
                        refine ⟨by norm_num, ?_, ?_, ?_⟩ <;> omega
                      case hWt =>
                        intro t u _ _
                        simp
                      case hWk =>
                        rw [iterW_empty]
                        simp
                      case h0 =>
                        refine ⟨rfl, ?_, ?_, ?_⟩
                        · intro r kk
                          have hr : ∀ r : Fin 128, (res.get (r, ())).toInt = (aRow d i r : Int) :=
                            fun r => lane_am hd i hg ho hres r
                          rw [ptrAdd_get2, Buf_ptr_zero_offset, aOff_val hd i hr r kk]
                        · intro kk c
                          have hrc : ∀ c : Fin 256, (res2.get (c, ())).toInt = (bCol d i c : Int) :=
                            fun c => lane_bn hd i hpn hres2 c
                          rw [ptrAdd_get2, Buf_ptr_zero_offset, bOff_val hd i hrc kk c]
                        · intro r c _ _ va vb _ _
                          refine ⟨0, ?_, ?_⟩
                          · simp only [tl.zeros, Tensor.get2]
                            exact DType.val?_zero_f32
                          · rw [partSum_zero]
                            unfold AccB
                            simp
                      case hstep =>
                        intro t htc st m1 hinv
                        obtain ⟨hm1, hpA, hpB, hacc⟩ := hinv
                        subst hm1
                        have ht : t < steps d.K := by rw [← hT]; exact htc
                        refine safe_convert (d := .f32) (d' := .f16)
                          (x := tl.full [128, 64] (FloatV.ofPat 0).litF32) ?_ ?_
                        · intro idx
                          exact ⟨_, convRel_cvt16 P _⟩
                        · intro za _
                          -- load `a`
                          refine safe_load_buf d.a
                            (g := fun idx : Idx [128, 64] => aAdr d i t idx.1 idx.2.1)
                            (en := fun _ => true) ?_ ?_ ?_ ?_ ?_
                          · refine Buf.addrs_of_get d.a ?_ ?_
                            · intro idx
                              obtain ⟨r, kk, u⟩ := idx
                              cases u
                              simpa only [Tensor.get2] using a_mask_get2 hd ht r kk
                            · intro idx _
                              obtain ⟨r, kk, u⟩ := idx
                              cases u
                              simpa only [Tensor.get2] using hpA r kk
                          · intro idx _
                            obtain ⟨r, kk, u⟩ := idx
                            cases u
                            exact aAdr_lt hd i ht kk.isLt
                          · exact hpre.1
                          · refine Set.Subset.trans (Buf.elems_mono d.a (T := aSet d) ?_) (by simp [R])
                            rintro j ⟨idx, _, rfl⟩
                            obtain ⟨r, kk, u⟩ := idx
                            cases u
                            exact aAdr_mem hd i ht kk.isLt
                          · intro av hreadA _
                            -- convert the `b` tile to float16
                            refine safe_convert (d := .f32) (d' := .f16)
                              (x := tl.full [64, 256] (FloatV.ofPat 0).litF32) ?_ ?_
                            · intro idx
                              exact ⟨_, convRel_cvt16 P _⟩
                            · intro zb _
                              -- load `b`
                              refine safe_load_buf d.b
                                (g := fun idx : Idx [64, 256] => bAdr d i t idx.1 idx.2.1)
                                (en := fun _ => true) ?_ ?_ ?_ ?_ ?_
                              · refine Buf.addrs_of_get d.b ?_ ?_
                                · intro idx
                                  obtain ⟨kk, c, u⟩ := idx
                                  cases u
                                  simpa only [Tensor.get2] using b_mask_get2 hd ht kk c
                                · intro idx _
                                  obtain ⟨kk, c, u⟩ := idx
                                  cases u
                                  simpa only [Tensor.get2] using hpB kk c
                              · intro idx _
                                obtain ⟨kk, c, u⟩ := idx
                                cases u
                                exact bAdr_lt hd i ht kk.isLt
                              · exact hpre.2.1
                              · refine Set.Subset.trans (Buf.elems_mono d.b (T := bSet d) ?_) (by simp [R])
                                rintro j ⟨idx, _, rfl⟩
                                obtain ⟨kk, c, u⟩ := idx
                                cases u
                                exact bAdr_mem hd i ht kk.isLt
                              · intro bt hreadB _
                                refine safe_dot
                                  (cfg := (⟨.f16, .f16, .f32, .tf32, 0⟩ : DotCfg))
                                  (hex := htot av bt st.2.2) ?_
                                intro accust hdotres
                                refine safe_ret ?_
                                refine ⟨rfl, ?_, ?_, ?_⟩
                                · intro r kk
                                  exact aStep_val hd i (hpA r kk)
                                · intro kk c
                                  exact bStep_val hd i (hpB kk c)
                                · intro r c hM hN va vb ha hb
                                  obtain ⟨v0, hval0, hprev⟩ := hacc r c hM hN va vb ha hb
                                  have hvat : ∀ kk : Fin 64,
                                      DType.val? .f16 (av.get2 r kk)
                                        = some ((va (64 * t + kk)).toRat) := by
                                    intro kk
                                    have h1 : m1.read .f16 (d.a.ptr ↑(aAdr d i t r kk)).addr
                                        = some (av.get2 r kk) := by
                                      have h := hreadA (r, kk, ()) rfl
                                      simpa only [Tensor.get2] using h
                                    have h2 : m1.read .f16 (d.a.ptr ↑(aAdr d i t r kk)).addr
                                        = some (va (64 * t + kk)) := by
                                      have h3 : aAdr d i t r kk = aRow d i r * d.stride_am
                                          + (64 * t + kk) * d.stride_ak := rfl
                                      rw [h3]
                                      exact (ha (64 * t + kk) (step_lt hd ht kk.isLt)).1
                                    rw [Option.some.inj (h1.symm.trans h2)]
                                    exact DType.val?_f16.mpr
                                      ⟨(ha (64 * t + kk) (step_lt hd ht kk.isLt)).2, rfl⟩
                                  have hvbt : ∀ kk : Fin 64,
                                      DType.val? .f16 (bt.get2 kk c)
                                        = some ((vb (64 * t + kk)).toRat) := by
                                    intro kk
                                    have h1 : m1.read .f16 (d.b.ptr ↑(bAdr d i t kk c)).addr
                                        = some (bt.get2 kk c) := by
                                      have h := hreadB (kk, c, ()) rfl
                                      simpa only [Tensor.get2, Prod.mk.injEq] using h
                                    have h2 : m1.read .f16 (d.b.ptr ↑(bAdr d i t kk c)).addr
                                        = some (vb (64 * t + kk)) := by
                                      have h3 : bAdr d i t kk c = (64 * t + kk) * d.stride_bk
                                          + bCol d i c * d.stride_bn := rfl
                                      rw [h3]
                                      exact (hb (64 * t + kk) (step_lt hd ht kk.isLt)).1
                                    rw [Option.some.inj (h1.symm.trans h2)]
                                    exact DType.val?_f16.mpr
                                      ⟨(hb (64 * t + kk) (step_lt hd ht kk.isLt)).2, rfl⟩
                                  exact lane_step hd hdot ht (fun e he => (ha e he).2) (fun e he => (hb e he).2)
                                    hvat hvbt hval0 hprev hdotres
                      case hk =>
                        intro st m1 hinv
                        obtain ⟨hm1, hpA, hpB, hacc⟩ := hinv
                        rw [hT] at hacc
                        refine safe_cvt (g := .binary16) ?_ ?_
                        · exact ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩
                        · intro cc hcc
                          refine safe_last (safe_store_buf d.c
                            (g := fun idx : Idx [128, 256] => cAdr d i (idx.1 : Nat) (idx.2.1 : Nat))
                            (en := fun idx : Idx [128, 256] =>
                              decide (cRow d i (idx.1 : Nat) < d.M) &&
                                decide (cCol d i (idx.2.1 : Nat) < d.N))
                            ?_ ?_ ?_ ?_ ?_ ?_)
                          · refine Buf.addrs_of_get d.c ?_ ?_
                            · intro idx
                              obtain ⟨r, c, u⟩ := idx
                              cases u
                              simpa only [Tensor.get2] using c_mask_get2 hd i hg ho hpn r c
                            · intro idx hen
                              obtain ⟨r, c, u⟩ := idx
                              cases u
                              have hM : cRow d i r < d.M := by
                                by_contra h
                                simp [h] at hen
                              have hN : cCol d i c < d.N := by
                                by_contra h
                                simp [h] at hen
                              simpa only [Tensor.get2] using cptr_get2_fit hd i hg ho hpn r c hM hN
                          · intro idx hen
                            obtain ⟨r, c, u⟩ := idx
                            cases u
                            have hM : cRow d i r < d.M := by
                              by_contra h
                              simp [h] at hen
                            have hN : cCol d i c < d.N := by
                              by_contra h
                              simp [h] at hen
                            exact cAdr_lt hd hM hN
                          · intro idx idx' hen hen' heq0
                            obtain ⟨r, c, u⟩ := idx
                            obtain ⟨r', c', u'⟩ := idx'
                            cases u
                            cases u'
                            have hM : cRow d i r < d.M := by
                              by_contra h
                              simp [h] at hen
                            have hN : cCol d i c < d.N := by
                              by_contra h
                              simp [h] at hen
                            have hM' : cRow d i r' < d.M := by
                              by_contra h
                              simp [h] at hen'
                            have hN' : cCol d i c' < d.N := by
                              by_contra h
                              simp [h] at hen'
                            have heq : cRow d i r * d.stride_cm + cCol d i c * d.stride_cn
                                = cRow d i r' * d.stride_cm + cCol d i c' * d.stride_cn := heq0
                            obtain ⟨h1, h2⟩ :=
                              hd.c_inj (cRow d i r) (cCol d i c) (cRow d i r') (cCol d i c')
                                hM hN hM' hN' heq
                            have hr : (r : Nat) = (r' : Nat) := by
                              unfold cRow at h1
                              omega
                            have hc : (c : Nat) = (c' : Nat) := by
                              unfold cCol at h2
                              omega
                            have hrr : r = r' := Fin.ext hr
                            have hcc : c = c' := Fin.ext hc
                            subst hrr
                            subst hcc
                            rfl
                          · rw [hm1]
                            exact hpre.2.2
                          · refine Set.Subset.trans
                              (Buf.elems_mono d.c (T := tileSet d (i : Nat)) ?_) ?_
                            · rintro e ⟨idx, hen, rfl⟩
                              obtain ⟨r, c, u⟩ := idx
                              cases u
                              have hM : cRow d i r < d.M := by
                                by_contra h
                                simp [h] at hen
                              have hN : cCol d i c < d.N := by
                                by_contra h
                                simp [h] at hen
                              exact cAdr_mem (i : Nat) r.isLt c.isLt hM hN
                            · intro e he
                              exact he
                          · intro m' hm' _ _
                            refine safe_ret ?_
                            intro r c hr hc hM hN
                            let rf : Fin 128 := ⟨r, hr⟩
                            let cf : Fin 256 := ⟨c, hc⟩
                            have haRow : aRow d i r = cRow d i r := by
                              unfold aRow cRow
                              exact Nat.mod_eq_of_lt hM
                            have hbCol : bCol d i c = cCol d i c := by
                              unfold bCol cCol
                              exact Nat.mod_eq_of_lt hN
                            refine ⟨cc.get2 rf cf, ?_, ?_⟩
                            · have hen : (decide (cRow d i r < d.M) &&
                                  decide (cCol d i c < d.N)) = true := by
                                simp [hM, hN]
                              simpa only [Tensor.get2, cAdr] using hm' (rf, cf, ()) hen
                            · intro va vb ha hb
                              have ha' : ∀ kk, kk < d.K →
                                  init.read .f16
                                      (d.a.ptr ↑(aRow d i r * d.stride_am + kk * d.stride_ak)).addr =
                                    some (va kk) ∧ (va kk).isFinite = true := by
                                intro kk hk
                                rw [haRow]
                                exact ha kk hk
                              have hb' : ∀ kk, kk < d.K →
                                  init.read .f16
                                      (d.b.ptr ↑(kk * d.stride_bk + bCol d i c * d.stride_bn)).addr =
                                    some (vb kk) ∧ (vb kk).isFinite = true := by
                                intro kk hk
                                rw [hbCol]
                                exact hb kk hk
                              obtain ⟨v0, hval0, hprev⟩ := hacc rf cf hM hN va vb ha' hb'
                              have hK64 : d.K % 64 = 0 := by
                                have := hd.K_dim.2.1
                                omega
                              have hsteps : steps d.K = d.K / 64 := steps_eq hK64
                              have hds : dotSum va vb d.K = partSum (prodV va vb d.K) (steps d.K) := by
                                rw [hsteps]
                                exact dotSum_eq hK64
                              have hfinA : (st.2.2.get2 rf cf).isFinite = true :=
                                (DType.val?_f32.mp hval0).1
                              have hvalA : (st.2.2.get2 rf cf).toRat = v0 :=
                                (DType.val?_f32.mp hval0).2
                              have htol0 : |v0 - dotSum va vb d.K| ≤ tol va vb d.K := by
                                rw [hds]
                                exact hprev.trans (AccB_le_tol hd (fun e he => (ha e he).2)
                                  (fun e he => (hb e he).2) (T := steps d.K) (le_refl _))
                              have htolA : |(st.2.2.get2 rf cf).toRat - dotSum va vb d.K| ≤
                                  tol va vb d.K := by
                                rw [hvalA]
                                exact htol0
                              have hv : |(st.2.2.get2 rf cf).toRat - dotSum va vb d.K| ≤
                                  (3 / 2) * rho * pathMag va vb d.K +
                                    (3 / 2) * (steps d.K : ℚ) * alp := by
                                rw [hvalA, hds]
                                exact hprev.trans (AccB_le_fine hd (T := steps d.K) (le_refl _))
                              have hnanA : FloatV.NaNEq (FloatV.cvt (st.2.2.get2 rf cf))
                                  (cc.get2 rf cf) := by
                                simpa only [Tensor.get2, FloatV.cvtWith_rn] using hcc (rf, cf, ())
                              refine ⟨st.2.2.get2 rf cf, hfinA, htolA, hnanA, ?_⟩
                              intro hroom
                              have hab : |v0| ≤ |dotSum va vb d.K| + tol va vb d.K := by
                                have h := abs_add_le (v0 - dotSum va vb d.K) (dotSum va vb d.K)
                                have e : (v0 - dotSum va vb d.K) + dotSum va vb d.K = v0 := by ring
                                rw [e] at h
                                linarith
                              have hmax : |(st.2.2.get2 rf cf).toRat| ≤
                                  Format.binary16.maxVal := by
                                rw [hvalA]
                                linarith
                              obtain ⟨hcvtFin, hcvtErr⟩ :=
                                FloatV.cvt_err (g := .binary16) (x := st.2.2.get2 rf cf)
                                  Format.binary16_regular hfinA hmax
                              have hcv : |(FloatV.cvt (g := .binary16) (st.2.2.get2 rf cf)).toRat -
                                    (st.2.2.get2 rf cf).toRat| ≤
                                  |(st.2.2.get2 rf cf).toRat| * 2 ^ (-11 : ℤ) +
                                    2 ^ (-25 : ℤ) := by
                                have h := hcvtErr
                                rw [Format.binary16_u,
                                  Format.binary16_eta] at h
                                exact h
                              have hcvnan : (FloatV.cvt (st.2.2.get2 rf cf)).isNaN = false :=
                                (FloatV.isFinite_iff.mp hcvtFin).1
                              have hceq : cc.get2 rf cf = FloatV.cvt (st.2.2.get2 rf cf) :=
                                FloatV.NaNEq.eq hnanA hcvnan
                              constructor
                              · rw [hceq]
                                exact hcvtFin
                              · rw [hceq]
                                exact cvt_bound (pathMag_nonneg va vb d.K) (steps_le d hd) hcv hv

end Claims.Matmul
