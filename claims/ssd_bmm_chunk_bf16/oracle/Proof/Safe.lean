import Claim
import TVA.Proof
import Proof.PostSafe

/-!
# The body of one program

Program `i`'s body: the scalar index arithmetic, the two pointer tiles, the main loop over the
`BK`-wide slabs of `K` (`Proof/Step.lean`), and the store of the masked tile
(`Proof/PostSafe.lean`). This file wires them together into the body's safety.
-/

open TVA
namespace Claims.SsdBmm

set_option maxHeartbeats 1600000


/-- Program `i`'s body leaves `Q` and writes only the tile `W`. -/
theorem body_safe {P : Platform} {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (hP : Device P)
    (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hGpos : 0 < d.G := hd.G_pos
  have hTpos : 0 < d.T := hd.T_pos
  have hKpos : 0 < d.K := hd.K_pos
  have hCge : 2 ≤ d.C := hd.C_ge
  have hG31 : d.G < 2 ^ 31 := G_lt_31 hd
  have hi31 : (i : Nat) < 2 ^ 31 := prog_lt hd i
  have hiG : (i : Nat) / gx d < 2 ^ 31 :=
    Nat.lt_of_le_of_lt (Nat.div_le_self _ _) hi31
  have hC31 : d.C < 2 ^ 31 := by
    have h := hd.oC_lt
    unfold oC at h
    have hG : 0 < d.G := hd.G_pos
    have hCge : 2 ≤ d.C := hd.C_ge
    have hcc : d.C ≤ d.C * d.C := Nat.le_mul_of_pos_right d.C (by omega)
    have h4 : d.C * d.C ≤ d.G * (d.C * d.C) := Nat.le_mul_of_pos_left _ hG
    omega
  have hCBN : d.C + BN < 2 ^ 31 := by
    have h := hd.oC_lt
    unfold oC at h
    have hle : d.C * d.C ≤ d.G * (d.C * d.C) := Nat.le_mul_of_pos_left _ hGpos
    have h2 : d.C * 2 ≤ d.C * d.C := Nat.mul_le_mul_left d.C hCge
    have hBN : BN = 128 := rfl
    omega
  rw [body_eq d ((launch d).ctx i)]
  simp only [tl.program_id, launch_pid0, launch_pid1, launch_pid2]
  -- peel 1: convert .i32 .i64 (program_id c 2)
  refine safe_conv_i32_i64_scalar (n := (IntV.ofNat ((i : Nat) / gx d) : I32)) ?_
  rw [IntV.I32.toInt_ofNat hiG]
  -- peel 2: wide d.G
  refine safe_wide_nat (n := d.G) hG31 ?_
  -- peel 3: idiv pid_ch d.G
  refine safe_idiv_val (w := 64) (a := (i : Nat) / gx d) (b := d.G) (hw := by norm_num) ?_ ?_
    (by omega) ?_ ?_
  · exact I64_toInt_ofNat hiG
  · exact I64_toInt_ofNat hG31
  · have h1 : ((i : Nat) / gx d / d.G : Nat) ≤ i :=
      le_trans (Nat.div_le_self _ _) (Nat.div_le_self _ _)
    have h2 : ((i : Nat) / gx d / d.G : Nat) < 2 ^ 31 := Nat.lt_of_le_of_lt h1 hi31
    have : (2 : ℤ) ^ 31 ≤ (2 : ℤ) ^ (64 - 1) := by norm_num
    have : (((i : Nat) / gx d / d.G : Nat) : ℤ) < (2 ^ 31 : ℤ) := by exact_mod_cast h2
    linarith
  · -- peel 4: wide d.G for pid_h
    refine safe_wide_nat (n := d.G) hG31 ?_
    have hph : (Tensor.scalar (IntV.ofInt (((i : Nat) / gx d : Nat) : ℤ) : I64)).item
        - (Tensor.scalar (IntV.ofInt (((i : Nat) / gx d / d.G : Nat) : ℤ) : I64)).item
          * (IntV.ofInt ((d.G : Nat) : ℤ) : I64)
        = (IntV.ofInt ((hOf d i : Nat) : ℤ) : I64) := by
      show ((IntV.ofInt (((i : Nat) / gx d : Nat) : ℤ) : I64)
          - (IntV.ofInt (((i : Nat) / gx d / d.G : Nat) : ℤ) : I64)
            * (IntV.ofInt ((d.G : Nat) : ℤ) : I64))
        = (IntV.ofInt ((hOf d i : Nat) : ℤ) : I64)
      rw [IntV.ofInt_mul_ofInt, IntV.ofInt_sub_ofInt]
      congr 1
      have h1 : (((i : Nat) / gx d : Nat) : ℤ) =
          d.G * (((i : Nat) / gx d / d.G : Nat) : ℤ) + (((i : Nat) / gx d % d.G : Nat) : ℤ) := by
        exact_mod_cast (Nat.div_add_mod ((i : Nat) / gx d) d.G).symm
      unfold hOf
      push_cast at h1 ⊢
      linarith
    simp only [hph]
    -- peel 5: num_pid_n
    refine safe_idiv_val (w := 32) (a := d.C + (BN - 1)) (b := BN) (hw := by norm_num) ?_ ?_
      (by norm_num [BN]) ?_ ?_
    · show ((IntV.ofNat d.C : I32) + (IntV.ofNat (BN - 1) : I32)).toInt = (((d.C + (BN - 1) : Nat)) : ℤ)
      refine IntV.I32.add_nat ?_ ?_ ?_
      · exact IntV.I32.toInt_ofNat hC31
      · exact IntV.I32.toInt_ofNat (by norm_num [BN])
      · have hBN : BN = 128 := rfl
        omega
    · show (IntV.ofNat BN : I32).toInt = ((BN : Nat) : ℤ)
      exact IntV.I32.toInt_ofNat (by norm_num [BN])
    · have h1 : (d.C + (BN - 1)) / BN ≤ d.C + (BN - 1) := Nat.div_le_self _ _
      have h2 : d.C + (BN - 1) < 2 ^ 31 := by
        have hBN : BN = 128 := rfl
        omega
      have h3 : ((d.C + (BN - 1)) / BN : Nat) < 2 ^ 31 := lt_of_le_of_lt h1 h2
      have h4 : (((d.C + (BN - 1)) / BN : Nat) : ℤ) < (2 : ℤ) ^ 31 := by exact_mod_cast h3
      have h5 : (2 : ℤ) ^ (32 - 1) = (2 : ℤ) ^ 31 := by norm_num
      rw [h5]; exact h4
    · -- peel 6: pid_m
      have hq32 : ((d.C + (BN - 1)) / BN : Nat) < 2 ^ 31 := by
        have h1 : (d.C + (BN - 1)) / BN ≤ d.C + (BN - 1) := Nat.div_le_self _ _
        have h2 : d.C + (BN - 1) < 2 ^ 31 := by
          have hBN : BN = 128 := rfl
          omega
        exact lt_of_le_of_lt h1 h2
      have hmod31 : ((i : Nat) % gx d) < 2 ^ 31 :=
        lt_of_le_of_lt (Nat.mod_le _ _) hi31
      refine safe_idiv_val (w := 32) (a := (i : Nat) % gx d) (b := (d.C + (BN - 1)) / BN)
        (hw := by norm_num) ?_ ?_ ?_ ?_ ?_
      · show (IntV.ofNat ((i : Nat) % gx d) : I32).toInt = (((i : Nat) % gx d : Nat) : ℤ)
        exact IntV.I32.toInt_ofNat hmod31
      · show (IntV.ofInt (((d.C + (BN - 1)) / BN : Nat) : ℤ) : I32).toInt = ((((d.C + (BN - 1)) / BN : Nat)) : ℤ)
        refine IntV.toInt_ofInt (w := 32) (by norm_num) ?_
        refine IntV.sfits_of_nonneg (Int.natCast_nonneg _) ?_
        have h4 : ((((d.C + (BN - 1)) / BN : Nat)) : ℤ) < (2 : ℤ) ^ 31 := by exact_mod_cast hq32
        have h5 : (2 : ℤ) ^ (32 - 1) = (2 : ℤ) ^ 31 := by norm_num
        rw [h5]; exact h4
      · apply Nat.div_pos _ (by norm_num [BN])
        have h := hCge
        norm_num [BN]
        omega
      · have h1 : ((i : Nat) % gx d) / ((d.C + (BN - 1)) / BN) ≤ (i : Nat) % gx d :=
          Nat.div_le_self _ _
        have h3 : ((i : Nat) % gx d) / ((d.C + (BN - 1)) / BN) < 2 ^ 31 := lt_of_le_of_lt h1 hmod31
        have h4 : ((((i : Nat) % gx d) / ((d.C + (BN - 1)) / BN) : Nat) : ℤ) < (2 : ℤ) ^ 31 := by
          exact_mod_cast h3
        have h5 : (2 : ℤ) ^ (32 - 1) = (2 : ℤ) ^ 31 := by norm_num
        rw [h5]; exact h4
      · -- peel 7: pid_n
        refine safe_irem_val (w := 32) (a := (i : Nat) % gx d) (b := (d.C + (BN - 1)) / BN)
          (hw := by norm_num) ?_ ?_ ?_ ?_ ?_
        · show (IntV.ofNat ((i : Nat) % gx d) : I32).toInt = (((i : Nat) % gx d : Nat) : ℤ)
          exact IntV.I32.toInt_ofNat hmod31
        · show (IntV.ofInt (((d.C + (BN - 1)) / BN : Nat) : ℤ) : I32).toInt = ((((d.C + (BN - 1)) / BN : Nat)) : ℤ)
          refine IntV.toInt_ofInt (w := 32) (by norm_num) ?_
          refine IntV.sfits_of_nonneg (Int.natCast_nonneg _) ?_
          have h4 : ((((d.C + (BN - 1)) / BN : Nat)) : ℤ) < (2 : ℤ) ^ 31 := by exact_mod_cast hq32
          have h5 : (2 : ℤ) ^ (32 - 1) = (2 : ℤ) ^ 31 := by norm_num
          rw [h5]; exact h4
        · apply Nat.div_pos _ (by norm_num [BN])
          have h := hCge
          norm_num [BN]
          omega
        · have h1 : ((i : Nat) % gx d) % ((d.C + (BN - 1)) / BN) < 2 ^ 31 :=
            lt_of_le_of_lt (Nat.mod_le _ _) hmod31
          have h4 : ((((i : Nat) % gx d) % ((d.C + (BN - 1)) / BN) : Nat) : ℤ) < (2 : ℤ) ^ 31 := by
            exact_mod_cast h1
          have h5 : (2 : ℤ) ^ (32 - 1) = (2 : ℤ) ^ 31 := by norm_num
          rw [h5]; exact h4
        · -- peel 8: batchOff aB
          refine safe_batchOff_zero (s := d.aB) ?_
          -- peel 9: wide d.C
          refine safe_wide_nat (n := d.C) hC31 ?_
          -- peel 10: wide d.aS
          refine safe_wide_nat (n := d.aS) hd.aS_lt ?_
          -- peel 11: wide d.aH
          refine safe_wide_nat (n := d.aH) hd.aH_lt ?_
          -- peel 12: batchOff bB
          refine safe_batchOff_zero (s := d.bB) ?_
          -- peel 13: wide d.C
          refine safe_wide_nat (n := d.C) hC31 ?_
          -- peel 14: wide d.bS
          refine safe_wide_nat (n := d.bS) hd.bS_lt ?_
          -- peel 15: wide d.bH
          refine safe_wide_nat (n := d.bH) hd.bH_lt ?_
          -- peel 16: wide (0 * T)
          simp only [IntV.ofNat_zero_mul]
          refine safe_wide_nat (n := 0) (by norm_num) ?_
          -- peel 17: wide d.C
          refine safe_wide_nat (n := d.C) hC31 ?_
          -- peel 18: wide 1
          refine safe_wide_nat (n := 1) (by norm_num) ?_
          -- peel 19: wide d.C
          refine safe_wide_nat (n := d.C) hC31 ?_
          -- peel 20: wide d.T
          refine safe_wide_nat (n := d.T) hd.T_lt ?_
          -- peel 21: wide d.C
          refine safe_wide_nat (n := d.C) hC31 ?_
          -- peel 22: trip count idiv
          refine safe_idiv_val (w := 32) (a := d.K + (BK - 1)) (b := BK) (hw := by norm_num) ?_ ?_
            (by norm_num [BK]) ?_ ?_
          · show ((IntV.ofNat d.K : I32) + (IntV.ofNat (BK - 1) : I32)).toInt = ((d.K + (BK - 1) : Nat) : ℤ)
            refine IntV.I32.add_nat ?_ ?_ ?_
            · exact IntV.I32.toInt_ofNat (by have h := hd.K_le; omega)
            · exact IntV.I32.toInt_ofNat (by norm_num [BK])
            · norm_num [BK]; have h := hd.K_le; omega
          · show (IntV.ofNat BK : I32).toInt = ((BK : Nat) : ℤ)
            exact IntV.I32.toInt_ofNat (by norm_num [BK])
          · have h1 : d.K + (BK - 1) ≤ 1024 + 31 := by
              have h := hd.K_le
              norm_num [BK]
              omega
            have h2 : (d.K + (BK - 1)) / BK ≤ (1024 + 31) / 32 := by
              have hBK : BK = 32 := rfl
              rw [hBK]
              exact Nat.div_le_div_right h1
            have h3 : (d.K + (BK - 1)) / BK ≤ 32 := by
              have := h2
              norm_num at this
              exact this
            have h4 : (((d.K + (BK - 1)) / BK : Nat) : ℤ) ≤ (32 : ℤ) := by exact_mod_cast h3
            have h5 : (2 : ℤ) ^ (32 - 1) = (2 : ℤ) ^ 31 := by norm_num
            rw [h5]
            linarith
          · -- the `for k in range(0, cdiv(K, BK))` loop
            refine safe_forRange_ne (E := (launch d).env P) (R := R d init i)
              (Q := fun _ m => Q d init i m) (m := init)
              (I := inv d init i) (Wk := W d init i)
              (lo := (IntV.ofInt 0 : I32))
              (hi := (IntV.ofInt ((d.K + (BK - 1)) / BK : ℕ) : I32))
              (step := (IntV.ofInt 1 : I32)) ?_ ?_ ?_ ?_
            · exact rangeOk hd
            · exact inv_zero hd i
            · intro t ht st m₁ hinv
              rw [hsteps hd] at ht
              exact step_safe hd i t ht hinv hpre hP
            · intro st m₁ hinv
              rw [hsteps hd] at hinv
              exact post_safe hd i hpre hinv

end Claims.SsdBmm


