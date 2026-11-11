import Claim
import TVA.Proof
import Proof.Basic
import Proof.Inv
import Proof.Arith
import Proof.Off
import Proof.Step
import Proof.Foot
import Proof.Fill
import Proof.ScaleOff
import Proof.Glue

/-!
# The `tl.dot_scaled` of one trip: it is safe, and the accumulator after it is within the tolerance of the partial sum.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- The rest of a loop trip after the loads: the dot of the trip is safe, and the accumulator after it keeps the invariant. -/
theorem dot_tail_safe (P : Platform) {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) {t : Nat} (ht : t < d.K / 128) {m : Mem} (hm : m = init)
    {acc : Tensor [128, 256] F32} {ka kb ks : I32}
    (hacc : ∀ l : Idx [128, 256], laneMask d i l = true →
      (acc.get l).isFinite = true ∧
        |(acc.get l).toRat - partSum (prods d init (laneRow d i l) (laneCol d i l)) t| ≤
          tol (prods d init (laneRow d i l) (laneCol d i l)) t)
    (hka : ka.toInt = ((128 * t : Nat) : Int))
    (hkb : kb.toInt = (((128 / epbB d.mixed) * t : Nat) : Int))
    (hks : ks.toInt = ((t : Nat) : Int))
    {cfg : ScaledCfg} {dbt : DType}
    (hshape : cfg.ShapeOk 128 256 128 128 128 (128 / epbB d.mixed) 256 4)
    (htypes : cfg.TypesOk .f8e4nv dbt .u8 .u8)
    (htot : Device.DotScaledTotal P cfg)
    (hrule : Device.DotScaledAcc P cfg Device.scaledDomBF16 Device.e8m0Dom
      Device.relDot16 Device.absDot16)
    {av : Tensor [128, 128] (DType.Val .f8e4nv)}
    {bv : Tensor [256, 128 / epbB d.mixed] (DType.Val dbt)}
    {sca : Tensor [128, 4] (DType.Val .u8)} {scb : Tensor [256, 4] (DType.Val .u8)}
    (hsa : ∀ (r : Fin 128) (k : Fin 128),
      scaledValue cfg.lhs cfg.lhsScale ((cfg.unpackL (d := .f8e4nv) 128 128 av).get2 r k)
        (some ((cfg.scalesL 128 128 sca).get2 r k)) =
        some (scaled (xa d init (((i : Nat) % gridM d) * 128 + (r : Nat))
          (128 * t + (k : Nat))).toRat
          (sa d init (((i : Nat) % gridM d) * 128 + (r : Nat)) (128 * t + (k : Nat)))))
    (hsb : ∀ (k : Fin 128) (n : Fin 256),
      ((i : Nat) / gridM d) * 256 + (n : Nat) < d.N →
      scaledValue cfg.rhs cfg.rhsScale ((cfg.unpackR (d := dbt) 128 256 bv.trans).get2 k n)
        (some ((cfg.scalesR 128 256 scb).get2 k n)) =
        some (scaled (bVal d init (((i : Nat) / gridM d) * 256 + (n : Nat))
          (128 * t + (k : Nat)))
          (sb d init (((i : Nat) / gridM d) * 256 + (n : Nat)) (128 * t + (k : Nat)))))
    (hsca : ∀ (r : Fin 128) (k : Fin 128),
      (cfg.scalesL 128 128 sca).get2 r k =
        DType.toBits .u8 (sa d init (((i : Nat) % gridM d) * 128 + (r : Nat))
          (128 * t + (k : Nat))))
    (hscb : ∀ (k : Fin 128) (n : Fin 256),
      ((i : Nat) / gridM d) * 256 + (n : Nat) < d.N →
      (cfg.scalesR 128 256 scb).get2 k n =
        DType.toBits .u8 (sb d init (((i : Nat) / gridM d) * 256 + (n : Nat))
          (128 * t + (k : Nat)))) :
    Safe ((launch d).env P) (R d i) ∅ (fun st' m₂ => Inv d init i (t + 1) st' m₂) m
      (do
        let accumulator ← tl.dot_scaled cfg 128 256 128 av (some sca) bv.trans (some scb) acc
        let ka' := ka + (IntV.ofNat (128 / 1) : I32)
        let kb' := kb + (IntV.ofNat (128 / epbB d.mixed) : I32)
        let ks' := ks + (IntV.ofNat 1 : I32)
        return (accumulator, ka', kb', ks')) := by
  obtain ⟨_, _, _, _, _, hvals⟩ := hpre
  refine safe_dot_scaled (cfg := cfg) (M := 128) (N := 256) (K := 128)
    (dl := .f8e4nv) (dr := dbt) (dsl := .u8) (dsr := .u8)
    (Ra := 128) (Ca := 128) (Rb := 128 / epbB d.mixed) (Cb := 256) (G := 4)
    (lhs := av) (lhs_scale := some sca) (rhs := bv.trans) (rhs_scale := some scb)
    (acc := acc) hshape htypes ?_ ?_
  · exact htot (a := cfg.unpackL (d := .f8e4nv) 128 128 av)
      (sa := some (cfg.scalesL 128 128 sca))
      (b := cfg.unpackR (d := dbt) 128 256 bv.trans)
      (sb := some (cfg.scalesR 128 256 scb)) (acc := acc)
  · intro res hpd
    refine Safe.ret ?_
    have ht1024 : t ≤ 1023 := by
      have h : d.K / 128 ≤ 1024 := by
        have h1 : d.K ≤ 131072 := by
          have hK := hd.K_ok.2.2
          have h2 : (2 : Nat) ^ 17 = 131072 := by norm_num
          omega
        have h2 : d.K / 128 ≤ 131072 / 128 := Nat.div_le_div_right h1
        have h3 : (131072 : Nat) / 128 = 1024 := by norm_num
        omega
      omega
    refine ⟨hm, ?_, ?_, ?_, ?_⟩
    · intro l hl
      obtain ⟨hfin, hinv⟩ := hacc l hl
      obtain ⟨hMrow, hNcol⟩ := laneMask_bounds hl
      have hl0 : l.coord 0 < 128 := l.coord_lt (by simp)
      have hl1 : l.coord 1 < 256 := l.coord_lt (by simp)
      have hkK : ∀ k : Fin 128, 128 * t + (k : Nat) < d.K := fun k => by
        have h := off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := d.K) (l := (k : Nat))
          ht k.isLt
        simpa only [Nat.mul_comm] using h
      set r0 : Fin 128 := ⟨l.coord 0, hl0⟩ with hr0
      set n0 : Fin 256 := ⟨l.coord 1, hl1⟩ with hn0
      set va : Fin 128 → ℚ := fun k =>
        scaled (xa d init (laneRow d i l) (128 * t + (k : Nat))).toRat
          (sa d init (laneRow d i l) (128 * t + (k : Nat))) with hva
      set vb : Fin 128 → ℚ := fun k =>
        scaled (bVal d init (laneCol d i l) (128 * t + (k : Nat)))
          (sb d init (laneCol d i l) (128 * t + (k : Nat))) with hvb
      have hsumAbs : (∑ k : Fin 128, |va k * vb k|) =
          chunkAbs (prods d init (laneRow d i l) (laneCol d i l)) t := by
        rw [hva, hvb]
        rw [show (∑ k : Fin 128,
            |scaled (xa d init (laneRow d i l) (128 * t + (k : Nat))).toRat
                (sa d init (laneRow d i l) (128 * t + (k : Nat))) *
              scaled (bVal d init (laneCol d i l) (128 * t + (k : Nat)))
                (sb d init (laneCol d i l) (128 * t + (k : Nat)))|) =
          ∑ k : Fin 128,
            |prods d init (laneRow d i l) (laneCol d i l) (128 * t + (k : Nat))| from rfl]
        exact sum_abs_chunk t
      have hsumProd : (∑ k : Fin 128, va k * vb k) =
          chunkSum (prods d init (laneRow d i l) (laneCol d i l)) t := by
        rw [hva, hvb]
        rw [show (∑ k : Fin 128,
            scaled (xa d init (laneRow d i l) (128 * t + (k : Nat))).toRat
                (sa d init (laneRow d i l) (128 * t + (k : Nat))) *
              scaled (bVal d init (laneCol d i l) (128 * t + (k : Nat)))
                (sb d init (laneCol d i l) (128 * t + (k : Nat)))) =
          ∑ k : Fin 128,
            prods d init (laneRow d i l) (laneCol d i l) (128 * t + (k : Nat)) from rfl]
        exact sum_chunk t
      have hroomIn : dotAbs (prods d init (laneRow d i l) (laneCol d i l)) d.K +
          tol (prods d init (laneRow d i l) (laneCol d i l)) (d.K / 128) ≤
          2 ^ (127 : ℤ) := by
        have h := hvals.room (laneRow d i l) (laneCol d i l) hMrow hNcol
        have h2 : (2 : ℚ) ^ (127 : ℤ) = 2 ^ 127 := by norm_num
        rw [h2]
        simpa [tolOut] using h
      have htT : t ≤ d.K / 128 := by omega
      have htK : 128 * (t + 1) ≤ d.K := by
        have h := K_div d hd
        omega
      have hroom : |(acc.get2 r0 n0).toRat| + (∑ k : Fin 128, |va k * vb k|) +
          (Device.relDot16 128 * (|(acc.get2 r0 n0).toRat| +
            ∑ k : Fin 128, |va k * vb k|) + Device.absDot16 128) ≤
          Format.binary32.maxVal := by
        rw [hsumAbs]
        have h := room_step (p := prods d init (laneRow d i l) (laneCol d i l)) (t := t)
          (K := d.K) (acc := (acc.get2 r0 n0).toRat) hinv htK htT hroomIn
        simpa [rho, alpha] using h
      have hres := hrule (M := 128) (N := 256) (K := 128)
        (a := cfg.unpackL (d := .f8e4nv) 128 128 av)
        (sa := some (cfg.scalesL 128 128 sca))
        (b := cfg.unpackR (d := dbt) 128 256 bv.trans)
        (sb := some (cfg.scalesR 128 256 scb))
        (acc := acc) res hpd r0 n0 va vb
        (fun k => hsa r0 k)
        (fun k => hsb k n0 hNcol)
        (fun k => ⟨(hvals.a_dom (laneRow d i l) (128 * t + (k : Nat)) hMrow (hkK k)).2.2,
          (hvals.b_dom (laneCol d i l) (128 * t + (k : Nat)) hNcol (hkK k)).2.2⟩)
        (by
          intro q hq k
          have hq' : q = cfg.scalesL 128 128 sca := (Option.some.inj hq).symm
          rw [hq', hsca r0 k]
          exact (hvals.a_dom (laneRow d i l) (128 * t + (k : Nat)) hMrow (hkK k)).2.1)
        (by
          intro q hq k
          have hq' : q = cfg.scalesR 128 256 scb := (Option.some.inj hq).symm
          rw [hq', hscb k n0 hNcol]
          exact (hvals.b_dom (laneCol d i l) (128 * t + (k : Nat)) hNcol (hkK k)).2.1)
        hfin hroom
      obtain ⟨hresFin, hresBnd⟩ := hres
      refine ⟨hresFin, ?_⟩
      have hstep : |(res.get2 r0 n0).toRat -
          ((acc.get2 r0 n0).toRat + (∑ k : Fin 128, va k * vb k))| ≤
          rho * (|(acc.get2 r0 n0).toRat| +
            chunkAbs (prods d init (laneRow d i l) (laneCol d i l)) t) + alpha := by
        rw [hsumAbs] at hresBnd
        simpa [rho, alpha] using hresBnd
      exact step_inv (p := prods d init (laneRow d i l) (laneCol d i l)) (t := t)
        (acc := (acc.get2 r0 n0).toRat) (acc' := (res.get2 r0 n0).toRat)
        (cs := ∑ k : Fin 128, va k * vb k) hinv hsumProd hstep
    · have hbnd : 128 * t + 128 < 2 ^ 31 := by
        have h2 : (2 : Nat) ^ 31 = 2147483648 := by norm_num
        omega
      have h128 : 128 / 1 = 128 := by norm_num
      rw [h128]
      exact (addN_toInt (a := ka) (x := 128 * t) (N := 128) hka hbnd).trans (by
        push_cast
        ring)
    · have hbl : 128 / epbB d.mixed ≤ 128 := by
        unfold epbB
        split <;> norm_num
      have hbnd : 128 / epbB d.mixed * t + 128 / epbB d.mixed < 2 ^ 31 := by
        have h1 : 128 / epbB d.mixed * t ≤ 128 * t := Nat.mul_le_mul_right t hbl
        have h2 : 128 * t ≤ 128 * 1023 := Nat.mul_le_mul_left 128 ht1024
        have h3 : (2 : Nat) ^ 31 = 2147483648 := by norm_num
        omega
      exact (addN_toInt (a := kb) (x := 128 / epbB d.mixed * t) (N := 128 / epbB d.mixed)
        hkb hbnd).trans (by push_cast; ring)
    · have hbnd : t + 1 < 2 ^ 31 := by
        have h2 : (2 : Nat) ^ 31 = 2147483648 := by norm_num
        omega
      exact addN_toInt (a := ks) (x := t) (N := 1) hks hbnd

end Claims.BlockScaledMxfp8
