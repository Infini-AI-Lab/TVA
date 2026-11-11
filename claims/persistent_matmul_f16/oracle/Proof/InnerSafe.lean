import Claim
import TVA.Proof
import Proof.Invariant
import Proof.Lanes
import Proof.Compute
import Proof.Numeric

/-!
# One step of the inner `K` loop is safe and keeps the accumulator's bound.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-- **One inner step is safe**: loading the `a` and `b` tiles, the `tl.dot`, and the accumulator update keep the cell invariant. -/
theorem safe_inner {d : Inst} (hd : d.ok) {E : Env} {init m : Mem} (hpre : Pre d init)
    (hm : MemInv d init m)
    (hacc : Device.DotLiftAcc E.P ⟨.f16, .f16, .f32, .tf32, 0⟩
      Device.relDot16 Device.absDot16)
    (htot : Device.DotTotal E.P ⟨.f16, .f16, .f32, .tf32, 0⟩)
    {T t : Nat} (hT : T < nT d) (ht : t < steps d.K)
    {offs_am : Tensor [128] I32} {offs_bn : Tensor [256] I32}
    {offs_k_for_mask : Tensor [64] I32} {ki : I32}
    (ham : ∀ i : Fin 128, (offs_am.get1 i).toInt = ((pm d T * 128 + i.val : Nat) : Int))
    (hbn : ∀ j : Fin 256, (offs_bn.get1 j).toInt = ((pn d T * 256 + j.val : Nat) : Int))
    (hkm : ∀ j : Fin 64, (offs_k_for_mask.get1 j).toInt = (j.val : Int))
    (hki : ki.toInt = (t : Int))
    {acc : Tensor [128, 256] F32} (hcell : CellInv d init T t acc) :
    Safe E (abSet d) ∅
      (fun acc' (m' : Mem) => CellInv d init T (t + 1) acc' ∧ MemInv d init m' ∧ m' = m) m
      (innerLoop d offs_am offs_bn offs_k_for_mask ki acc) := by
  -- the value of `offs_k`
  have hoffk : ∀ j : Fin 64,
      ((ki * (IntV.ofNat 64 : I32) + tl.arange 0 64).get1 j).toInt =
        ((t * 64 + j.val : Nat) : Int) := by
    intro j
    have hmul : (ki * (IntV.ofNat 64 : I32)).toInt = ((t * 64 : Nat) : Int) :=
      IntV.I32.mul_nat (a := ki) (b := (IntV.ofNat 64 : I32)) (x := t) (y := 64) hki
        (IntV.I32.toInt_ofNat (by norm_num))
        (by have h1 := kk_mul_lt hd ht; have h2 := hd.K_le; omega)
    have har : ((tl.arange 0 64).get1 j).toInt = (j.val : Int) :=
      arange_get1_toInt (by norm_num) j
    rw [show (ki * (IntV.ofNat 64 : I32) + tl.arange 0 64).get1 j
        = (ki * (IntV.ofNat 64 : I32)) + (tl.arange 0 64).get1 j from rfl]
    rw [IntV.I32.add_nat hmul har]
    have h1 := kk_bound hd ht j.isLt
    have h2 := hd.K_le
    omega
  simp only [innerLoop]
  refine safe_convert ?_ ?_
  · intro i; exact convRel_f32_f16 E.P _
  · intro az _
    have hspanA : ∀ x : Idx [128, 64],
        Mem.span .f16 (d.a.ptr (aGs d T t x : Int)).addr ⊆ abSet d := by
      intro x
      exact (Buf.span_sub_elems d.a (aG_mem hd hT ht x.1.isLt x.2.1.isLt)).trans
        Set.subset_union_left
    have hA : d.a.Addrs
        (tl.addrs (d.a.ptr 0 + (offs_am.col 64 * (IntV.ofNat d.stride_am : I32) +
          (ki * (IntV.ofNat 64 : I32) + tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32)))
          (some ((offs_k_for_mask.row 128).ltS ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32)))))
        (aGs d T t) (fun _ => true) := by
      refine Buf.addrs_of_get d.a ?_ ?_
      · intro x
        exact amask_get hd hkm hki ht x
      · intro x _
        exact congrArg Ptr.mk (aptrs_lane hd hT ht ham hoffk x)
    have hvA : d.a.Valid m :=
      Buf.valid_of_agree hpre.1 (fun x hx => hm.2 x (hpre.1.2.2.2 x hx))
    have hRA : d.a.elems (Buf.img (aGs d T t) (fun _ => true)) ⊆ abSet d :=
      (Buf.elems_mono d.a (aGs_mem hd hT ht)).trans Set.subset_union_left
    refine safe_load_buf (b := d.a) (g := aGs d T t) (en := fun _ => true) hA ?_ hvA hRA ?_
    · intro x _
      exact aG_len hd hT ht x.1.isLt x.2.1.isLt
    · intro v hread hother
      refine safe_convert ?_ ?_
      · intro i; exact convRel_f32_f16 E.P _
      · intro bz _
        have hspanB : ∀ x : Idx [64, 256],
            Mem.span .f16 (d.b.ptr (bGs d T t x : Int)).addr ⊆ abSet d := by
          intro x
          exact (Buf.span_sub_elems d.b (bG_mem hd hT ht x.1.isLt x.2.1.isLt)).trans
            Set.subset_union_right
        have hA2 : d.b.Addrs
            (tl.addrs (d.b.ptr 0 +
              ((ki * (IntV.ofNat 64 : I32) + tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
                offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32)))
              (some ((offs_k_for_mask.col 256).ltS ((IntV.ofNat d.K : I32) - ki * (IntV.ofNat 64 : I32)))))
            (bGs d T t) (fun _ => true) := by
          refine Buf.addrs_of_get d.b ?_ ?_
          · intro x
            exact bmask_get hd hkm hki ht x
          · intro x _
            exact congrArg Ptr.mk (bptrs_lane hd hT ht hbn hoffk x)
        have hvB : d.b.Valid m :=
          Buf.valid_of_agree hpre.2.1 (fun x hx => hm.2 x (hpre.2.1.2.2.2 x hx))
        have hRB : d.b.elems (Buf.img (bGs d T t) (fun _ => true)) ⊆ abSet d :=
          (Buf.elems_mono d.b (bGs_mem hd hT ht)).trans Set.subset_union_right
        refine safe_load_buf (b := d.b) (g := bGs d T t) (en := fun _ => true) hA2 ?_ hvB hRB ?_
        · intro x _
          exact bG_len hd hT ht x.1.isLt x.2.1.isLt
        · intro w hreadb hotherb
          refine safe_last ?_
          refine safe_dot (htot v w acc) ?_
          intro r hdot
          refine safe_ret ?_
          refine ⟨?_, hm, rfl⟩
          -- the cell invariant at t+1
          intro iX nX va vb hva hvb
          obtain ⟨hcfin, hcbound⟩ := hcell iX nX va vb hva hvb
          -- the loaded rows' values
          have hvaK : ∀ k : Fin 64,
              DType.val? .f16 (v.get2 iX k) = some ((va (t * 64 + k.val)).toRat) := by
            intro k
            have hk : t * 64 + k.val < d.K := kk_bound hd ht k.isLt
            have hreadx := hread (iX, k, ()) rfl
            have hrc : m.read .f16 (d.a.ptr (aGs d T t (iX, k, ()) : Int)).addr =
                init.read .f16 (d.a.ptr (aGs d T t (iX, k, ()) : Int)).addr :=
              Mem.read_congr (fun x hx => hm.1 x (hspanA (iX, k, ()) hx))
            rw [hrc] at hreadx
            rw [show (d.a.ptr (aGs d T t (iX, k, ()) : Int)).addr
                = (d.a.ptr ↑((pm d T * 128 + iX.val) * d.stride_am +
                    (t * 64 + k.val) * d.stride_ak)).addr from rfl] at hreadx
            rw [(hva (t * 64 + k.val) hk).1] at hreadx
            have heq : v.get2 iX k = va (t * 64 + k.val) := by
              have := Option.some.inj hreadx
              exact this.symm
            rw [heq, DType.val?_f16]
            exact ⟨(hva _ hk).2, rfl⟩
          have hvbK : ∀ k : Fin 64,
              DType.val? .f16 (w.get2 k nX) = some ((vb (t * 64 + k.val)).toRat) := by
            intro k
            have hk : t * 64 + k.val < d.K := kk_bound hd ht k.isLt
            have hreadx := hreadb (k, nX, ()) rfl
            have hrc : m.read .f16 (d.b.ptr (bGs d T t (k, nX, ()) : Int)).addr =
                init.read .f16 (d.b.ptr (bGs d T t (k, nX, ()) : Int)).addr :=
              Mem.read_congr (fun x hx => hm.1 x (hspanB (k, nX, ()) hx))
            rw [hrc] at hreadx
            rw [show (d.b.ptr (bGs d T t (k, nX, ()) : Int)).addr
                = (d.b.ptr ↑((t * 64 + k.val) * d.stride_bk +
                    (pn d T * 256 + nX.val) * d.stride_bn)).addr from rfl] at hreadx
            rw [(hvb (t * 64 + k.val) hk).1] at hreadx
            have heq : w.get2 k nX = vb (t * 64 + k.val) := by
              have := Option.some.inj hreadx
              exact this.symm
            rw [heq, DType.val?_f16]
            exact ⟨(hvb _ hk).2, rfl⟩
          -- `hc`: the accumulator element is bounded by 2^54
          have hSb : |partSum (prodV va vb d.K) t| ≤ accMag va vb d.K := partSum_le_accMag ht
          have hEm : errB rho' alpha' (partMag va vb d.K t) t ≤
              errB rho' alpha' (accMag va vb d.K) (steps d.K) := by
            refine errB_mono rho'_nonneg alpha'_nonneg ?_ (accMag_nonneg va vb d.K) (Nat.le_of_lt ht)
            rw [← partMag_steps]
            exact partMag_mono (Nat.le_of_lt ht)
          have hEtol : errB rho' alpha' (accMag va vb d.K) (steps d.K) ≤ tol va vb d.K :=
            errB_le_tol hd.K_le (fun k hk => (hva k hk).2) (fun k hk => (hvb k hk).2)
          have hta : tol va vb d.K + accMag va vb d.K ≤ 2 ^ 54 :=
            tol_add_accMag_le hd.K_le (fun k hk => (hva k hk).2) (fun k hk => (hvb k hk).2)
          have hc : |(acc.get2 iX nX).toRat| ≤ 2 ^ 54 := by
            have h1 : |(acc.get2 iX nX).toRat| ≤
                |(acc.get2 iX nX).toRat - partSum (prodV va vb d.K) t| +
                  |partSum (prodV va vb d.K) t| := by
              have h := abs_add_le ((acc.get2 iX nX).toRat - partSum (prodV va vb d.K) t)
                (partSum (prodV va vb d.K) t)
              rwa [sub_add_cancel] at h
            linarith
          have hvaAbs : ∀ k : Fin 64, |(va (t * 64 + k.val)).toRat| ≤ 65504 := by
            intro k
            exact f16_abs_le (hva (t * 64 + k.val) (kk_bound hd ht k.isLt)).2
          have hvbAbs : ∀ k : Fin 64, |(vb (t * 64 + k.val)).toRat| ≤ 65504 := by
            intro k
            exact f16_abs_le (hvb (t * 64 + k.val) (kk_bound hd ht k.isLt)).2
          have hroom := room hvaAbs hvbAbs (c := (acc.get2 iX nX).toRat) hc
          have hrel : (0 : ℚ) ≤ Device.relDot16 64 := by
            unfold Device.relDot16; positivity
          have habs : (0 : ℚ) ≤ Device.absDot16 64 := by
            unfold Device.absDot16; positivity
          have hdotres := DotLift.dotAdd_step (P := E.P) (fusion := true)
            (da := .f16) (db := .f16) (pr := .tf32) (mi := 0)
            (rel := Device.relDot16) (abs := Device.absDot16)
            hacc hrel habs (z := Tensor.full (FloatV.ofPat 0)) (Or.inl hdot) iX nX
            (DType.val?_f32.mpr ⟨(FloatV.isFinite_ofPat_zero Format.binary32_regular), FloatV.toRat_ofPat_zero⟩) hvaK hvbK hcfin hroom
          -- identify the sums with the step magnitudes
          have ht64 : 64 * (t + 1) ≤ d.K := by
            have h1 : t + 1 ≤ steps d.K := Nat.succ_le_of_lt ht
            have h2 : steps d.K = d.K / 64 := steps_eq hd
            have h3 : d.K / 64 * 64 = d.K := Nat.div_mul_cancel (K_dvd hd)
            have h4 := Nat.mul_le_mul_right 64 h1
            omega
          have hD : (∑ k : Fin 64, (va (t * 64 + k.val)).toRat * (vb (t * 64 + k.val)).toRat)
              = stepMag (prodV va vb d.K) t := by
            rw [stepMag_prodV ht64]
            simp only [Nat.mul_comm 64 t]
          have hL : (∑ k : Fin 64, Device.liftMag .f16 ((va (t * 64 + k.val)).toRat) *
                Device.liftMag .f16 ((vb (t * 64 + k.val)).toRat))
              = stepMag (liftV va vb d.K) t := by
            rw [stepMag_liftV ht64]
            simp only [Nat.mul_comm 64 t]
          have hs : |(r.get2 iX nX).toRat - ((acc.get2 iX nX).toRat +
                stepMag (prodV va vb d.K) t)| ≤
              rho' * (|(acc.get2 iX nX).toRat| + stepMag (liftV va vb d.K) t) + alpha' := by
            simpa [rho', alpha', hD, hL] using hdotres.2
          have hPS : partSum (prodV va vb d.K) (t + 1) =
              partSum (prodV va vb d.K) t + stepMag (prodV va vb d.K) t := by
            rw [partSum_succ, stepMag_prodV ht64]
            congr 1
            refine Finset.sum_congr rfl (fun k _ => ?_)
            unfold prodV
            have hk : 64 * t + (k : Nat) < d.K := by
              have := kk_bound hd ht k.isLt
              omega
            rw [ite_eq_left hk]
          have hPM := partMag_succ va vb d.K t
          have hstep := errB_step (ρ := rho') (α := alpha') (B := partMag va vb d.K t)
            (t := t) (c := (acc.get2 iX nX).toRat) (S := partSum (prodV va vb d.K) t)
            (D := stepMag (prodV va vb d.K) t) (L := stepMag (liftV va vb d.K) t)
            (r := (r.get2 iX nX).toRat) rho'_nonneg alpha'_nonneg
            (stepMag_liftV_nonneg va vb d.K t) hcbound hs
          refine ⟨hdotres.1, ?_⟩
          rw [hPS, hPM]
          exact hstep

end Claims.PersistentMatmul
