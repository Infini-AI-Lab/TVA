import Proof.LaneG

/-!
# One lane's GELU, the float chain (lines 86–90)

`lane_gelu` walks the lane's operations with the float lemmas of `Proof.Glue`: finiteness and room
at each step, the argument of `tl.sigmoid` by `x2_core`, the logistic by `sig_chain` and
`sig_core`, and `activated` by `act_core`.
-/

namespace Claims.GeluAndMulSparse

open TVA

set_option maxHeartbeats 1600000

/-- The reciprocal and the division of `tl.sigmoid`: `1 + exp(-x2)` rounded (`den`), then the
platform's `1 / den` (`sg`), against `sig X`. -/
theorem sig_core {X ex den sg : ℝ}
    (hc : 1 / 2 ≤ 1 + ex ∧ 1 + ex ≤ 3 ∧ |1 / (1 + ex) - sig X| ≤ 2 ^ (-19 : ℤ))
    (hden : |den - (1 + ex)| ≤ u32 * |1 + ex| + eta32)
    (hsg : |sg - 1 / den| ≤ 2 ^ (-22 : ℤ) * |1 / den| + 2 ^ (-148 : ℤ)) :
    |sg - sig X| ≤ 2 ^ (-18 : ℤ) ∧ 49 / 100 ≤ den ∧ den ≤ 31 / 10 := by
  obtain ⟨h1, h2, h3⟩ := hc
  have hu : u32 = 1 / 16777216 := by unfold u32; norm_num
  have he : eta32 ≤ 1 / 10 ^ 45 := by unfold eta32; norm_num
  have he0 : 0 ≤ eta32 := by unfold eta32; positivity
  rw [hu, abs_of_pos (by linarith : (0 : ℝ) < 1 + ex)] at hden
  have hd1 : 49 / 100 ≤ den := by have := (abs_le.mp hden).1; nlinarith
  have hd2 : den ≤ 31 / 10 := by have := (abs_le.mp hden).2; nlinarith
  refine ⟨?_, hd1, hd2⟩
  have hdpos : 0 < den := by linarith
  have hepos : 0 < 1 + ex := by linarith
  have hinv : |1 / den| ≤ 100 / 49 := by
    rw [abs_of_pos (by positivity), div_le_div_iff₀ hdpos (by norm_num)]; linarith
  have hdiff : |1 / den - 1 / (1 + ex)| ≤ (1 / 16777216 * 3 + eta32) * (200 / 49) := by
    have : 1 / den - 1 / (1 + ex) = ((1 + ex) - den) / (den * (1 + ex)) := by
      field_simp
    rw [this, abs_div, abs_of_pos (by positivity : 0 < den * (1 + ex)), div_le_iff₀ (by positivity)]
    have hnum : |(1 + ex) - den| ≤ 1 / 16777216 * 3 + eta32 := by
      rw [abs_sub_comm]; have := (abs_le.mp hden); rw [abs_le]; constructor <;> nlinarith
    have hprod : 49 / 200 ≤ den * (1 + ex) := by nlinarith
    have h0 : 0 ≤ (1 / 16777216 * 3 + eta32) * (200 / 49) := by positivity
    calc |(1 + ex) - den| ≤ 1 / 16777216 * 3 + eta32 := hnum
      _ = (1 / 16777216 * 3 + eta32) * (200 / 49) * (49 / 200) := by ring
      _ ≤ (1 / 16777216 * 3 + eta32) * (200 / 49) * (den * (1 + ex)) :=
          mul_le_mul_of_nonneg_left hprod h0
  have hsplit : sg - sig X = (sg - 1 / den) + (1 / den - 1 / (1 + ex)) + (1 / (1 + ex) - sig X) := by
    ring
  rw [hsplit]
  have t1 := abs_add_le ((sg - 1 / den) + (1 / den - 1 / (1 + ex))) (1 / (1 + ex) - sig X)
  have t2 := abs_add_le (sg - 1 / den) (1 / den - 1 / (1 + ex))
  have hsg' : |sg - 1 / den| ≤ 2 ^ (-22 : ℤ) * (100 / 49) + 2 ^ (-148 : ℤ) := by
    have : 2 ^ (-22 : ℤ) * |1 / den| ≤ 2 ^ (-22 : ℤ) * (100 / 49) := by gcongr
    linarith
  have hn : (2 : ℝ) ^ (-22 : ℤ) * (100 / 49) + 2 ^ (-148 : ℤ) + (1 / 16777216 * 3 + 1 / 10 ^ 45) * (200 / 49) +
      2 ^ (-19 : ℤ) ≤ 2 ^ (-18 : ℤ) := by norm_num
  have hdiff' : (1 / 16777216 * 3 + eta32) * (200 / 49) ≤ (1 / 16777216 * 3 + 1 / 10 ^ 45) * (200 / 49) := by
    gcongr
  linarith

/-- The last steps of the GELU (lines 89–90): `2·sg`, `− 1` (fusable), `1 + …`, `0.5·S` and their
product, against `S · sig X`. -/
theorem act_core {S X sg tw th ot h act : ℝ} (hS0 : 0 ≤ S) (hS1 : S ≤ 2 ^ 41) (hX0 : 0 ≤ X)
    (hsg : |sg - sig X| ≤ 2 ^ (-18 : ℤ))
    (htw : |tw - 2 * sg| ≤ u32 * |2 * sg| + eta32)
    (hth : |th - (tw + -1)| ≤ 3 * u32 * (|tw| + |(-1 : ℝ)|) + 4 * eta32)
    (hot : |ot - (1 + th)| ≤ u32 * |1 + th| + eta32)
    (hh : |h - 1 / 2 * S| ≤ u32 * |1 / 2 * S| + eta32)
    (hact : |act - h * ot| ≤ u32 * |h * ot| + eta32) :
    |act - S * sig X| ≤ 2 ^ (-15 : ℤ) * (S * sig X) + 2 ^ (-145 : ℤ) ∧ |ot| ≤ 3 := by
  have hσ1 := sig_le_one X
  have hσ2 := sig_ge_half hX0
  have hu : u32 = 1 / 16777216 := by unfold u32; norm_num
  have he : eta32 ≤ 1 / 10 ^ 45 := by unfold eta32; norm_num
  have he0 : 0 ≤ eta32 := by unfold eta32; positivity
  rw [hu] at htw hth hot hh hact
  have h18 : (2 : ℝ) ^ (-18 : ℤ) = 1 / 262144 := by norm_num
  rw [h18] at hsg
  have hsg1 : sg ≤ 1 + 1 / 262144 := by have := (abs_le.mp hsg).2; linarith
  have hsg0 : 1 / 2 - 1 / 262144 ≤ sg := by have := (abs_le.mp hsg).1; linarith
  rw [abs_of_pos (by linarith : (0 : ℝ) < 2 * sg)] at htw
  have htw1 : |tw| ≤ 2 * sg * (1 + 1 / 16777216) + eta32 := by
    have := abs_le_of_err htw; rw [abs_of_pos (by linarith : (0 : ℝ) < 2 * sg)] at this; linarith
  rw [abs_neg, abs_one] at hth
  -- ot against 2 sg
  have hot2 : |ot - 2 * sg| ≤ 14 / 16777216 := by
    have hth1 : |1 + th - 2 * sg| ≤ 1 / 16777216 * (2 * sg) + eta32 +
        3 * (1 / 16777216) * (|tw| + 1) + 4 * eta32 := by
      have : 1 + th - 2 * sg = (th - (tw + -1)) + (tw - 2 * sg) := by ring
      rw [this]; have := abs_add_le (th - (tw + -1)) (tw - 2 * sg); linarith
    have h1th : |1 + th| ≤ 2 * sg + 1 / 1000 := by
      have := abs_le_of_err hth1
      rw [abs_of_pos (by linarith : (0 : ℝ) < 2 * sg)] at this
      nlinarith
    have : ot - 2 * sg = (ot - (1 + th)) + (1 + th - 2 * sg) := by ring
    rw [this]
    have t := abs_add_le (ot - (1 + th)) (1 + th - 2 * sg)
    have : 1 / 16777216 * |1 + th| ≤ 1 / 16777216 * (2 * sg + 1 / 1000) := by gcongr
    nlinarith
  have hotσ : |ot - 2 * sig X| ≤ 1 / 131072 + 14 / 16777216 := by
    have : ot - 2 * sig X = (ot - 2 * sg) + 2 * (sg - sig X) := by ring
    rw [this]
    have t := abs_add_le (ot - 2 * sg) (2 * (sg - sig X))
    rw [abs_mul, abs_two] at t
    linarith
  have hotb : |ot| ≤ 3 := by
    have := abs_le_of_err hotσ
    rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ 2 * sig X)] at this
    linarith
  refine ⟨?_, hotb⟩
  rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ 1 / 2 * S)] at hh
  have hha : |h| ≤ 1 / 2 * S * (1 + 1 / 16777216) + eta32 := by
    have := abs_le_of_err hh; rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ 1 / 2 * S)] at this
    linarith
  have hsplit : act - S * sig X = (act - h * ot) + (h - 1 / 2 * S) * ot +
      1 / 2 * S * (ot - 2 * sig X) := by ring
  rw [hsplit]
  have t1 := abs_add_le ((act - h * ot) + (h - 1 / 2 * S) * ot) (1 / 2 * S * (ot - 2 * sig X))
  have t2 := abs_add_le (act - h * ot) ((h - 1 / 2 * S) * ot)
  have q1 : |h * ot| ≤ (1 / 2 * S * (1 + 1 / 16777216) + eta32) * 3 := by
    rw [abs_mul]; exact mul_le_mul hha hotb (abs_nonneg _) (by positivity)
  have q2 : |(h - 1 / 2 * S) * ot| ≤ (1 / 16777216 * (1 / 2 * S) + eta32) * 3 := by
    rw [abs_mul]; exact mul_le_mul hh hotb (abs_nonneg _) (by positivity)
  have q3 : |1 / 2 * S * (ot - 2 * sig X)| ≤ 1 / 2 * S * (1 / 131072 + 14 / 16777216) := by
    rw [abs_mul, abs_of_nonneg (by linarith : (0 : ℝ) ≤ 1 / 2 * S)]
    exact mul_le_mul_of_nonneg_left hotσ (by linarith)
  have hSσ : S ≤ 2 * (S * sig X) := by nlinarith
  have h15 : (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 := by norm_num
  have h145 : (1 / 10 ^ 45 : ℝ) * 10 ≤ 2 ^ (-145 : ℤ) := by norm_num
  rw [h15]
  generalize (2 : ℝ) ^ (-145 : ℤ) = Q at *
  nlinarith

/-- **One lane's GELU** (lines 86–90): finite, within `2^-15·geluRef s + 2^-145` of `geluRef s`,
and a zero when `s` is. -/
theorem lane_gelu {P : Platform} (hP : Device P) {fusion : Bool}
    {s p1 p2 z inn x2 nx ex den sg tw th h ot act : F32}
    (hs : s.isFinite = true) (hs0 : 0 ≤ s.toReal) (hs1 : s.toReal ≤ 2 ^ 41)
    (h1 : FloatV.NaNEq (FloatV.mul (c32 0x3FA6E4E26D4801F7) s) p1)
    (h2 : FloatV.NaNEq (FloatV.mul p1 s) p2)
    (h3 : FloatV.NaNEq (FloatV.add (FloatV.mul p2 s) s) z ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma p2 s s) z))
    (h4 : FloatV.NaNEq (FloatV.mul (c32 0x3FE9884533D43651) z) inn)
    (h5 : FloatV.NaNEq (FloatV.mul (c32 0x4000000000000000) inn) x2)
    (h6 : FloatV.NaNEq (FloatV.neg x2) nx)
    (h7 : P.math .exp .binary32 nx ex)
    (h8 : FloatV.NaNEq (FloatV.add (FloatV.ofRat 1) ex) den)
    (h9 : P.div .binary32 (FloatV.ofRat 1) den sg)
    (h10 : FloatV.NaNEq (FloatV.mul (c32 0x4000000000000000) sg) tw)
    (h11 : AddRel fusion tw (FloatV.neg (c32 0x3FF0000000000000)) th)
    (h12 : FloatV.NaNEq (FloatV.mul (c32 0x3FE0000000000000) s) h)
    (h13 : FloatV.NaNEq (FloatV.add (c32 0x3FF0000000000000) th) ot)
    (h14 : FloatV.NaNEq (FloatV.mul h ot) act) :
    act.isFinite = true ∧
      |act.toReal - geluRef s.toReal| ≤ 2 ^ (-15 : ℤ) * geluRef s.toReal + 2 ^ (-145 : ℤ) ∧
      (s.toRat = 0 → act.toRat = 0) := by
  have hu : u32 = 1 / 16777216 := by unfold u32; norm_num
  have he : eta32 ≤ 1 / 10 ^ 45 := by unfold eta32; norm_num
  have he0 : 0 ≤ eta32 := by unfold eta32; positivity
  obtain ⟨haf, hav⟩ := c_a
  obtain ⟨hkf, hkv⟩ := c_k
  obtain ⟨h2f, h2v⟩ := c_two
  obtain ⟨h1f, h1v⟩ := c_one
  obtain ⟨hhf, hhv⟩ := c_half
  obtain ⟨hsf, hsv⟩ := c_sig1
  have haF : ((12003091 / 268435456 : ℚ) : ℝ) = 12003091 / 268435456 := by norm_num
  have hkF : ((6693141 / 8388608 : ℚ) : ℝ) = 6693141 / 8388608 := by norm_num
  have h2F : ((2 : ℚ) : ℝ) = 2 := by norm_num
  have h1F : ((1 : ℚ) : ℝ) = 1 := by norm_num
  have hhF : ((1 / 2 : ℚ) : ℝ) = 1 / 2 := by norm_num
  rw [haF] at hav; rw [hkF] at hkv; rw [h2F] at h2v; rw [h1F] at h1v hsv; rw [hhF] at hhv
  generalize hS : s.toReal = S at hs0 hs1
  -- p1
  have hA0 : 0 ≤ 12003091 / 268435456 * S := mul_nonneg (by norm_num) hs0
  have r1 : |(c32 0x3FA6E4E26D4801F7).toReal * s.toReal| ≤ 2 ^ 120 := by
    rw [hav, hS, abs_of_nonneg hA0]; linarith
  obtain ⟨p1f, p1e⟩ := mul_lane haf hs r1 h1
  rw [hav, hS, hu, abs_of_nonneg hA0] at p1e
  -- p2
  have hp1 : |p1.toReal| ≤ S / 22 + 1 / 10 ^ 45 := by
    have := abs_le_of_err p1e; rw [abs_of_nonneg hA0] at this; linarith
  have hp1S : |p1.toReal * S| ≤ S * S / 22 + S / 10 ^ 45 := by
    rw [abs_mul, abs_of_nonneg hs0]
    have := mul_le_mul_of_nonneg_right hp1 hs0
    have e : (S / 22 + 1 / 10 ^ 45) * S = S * S / 22 + S / 10 ^ 45 := by ring
    linarith
  have hSS : S * S ≤ 2 ^ 82 := by nlinarith
  have hp1S' : |p1.toReal * S| ≤ 2 ^ 78 := by linarith
  obtain ⟨p2f, p2e⟩ := mul_lane p1f hs (by rw [hS]; linarith) h2
  rw [hS, hu] at p2e
  have hp2 : |p2.toReal| ≤ 2 ^ 78 * (1 + 1 / 16777216) + 1 / 10 ^ 45 := by
    have := abs_le_of_err p2e; linarith
  have hp2S : |p2.toReal * S| ≤ (2 ^ 78 * (1 + 1 / 16777216) + 1 / 10 ^ 45) * 2 ^ 41 := by
    rw [abs_mul, abs_of_nonneg hs0]; exact mul_le_mul hp2 hs1 hs0 (by positivity)
  have hp2S' : |p2.toReal * S| ≤ 2 ^ 119 + 2 ^ 100 := by
    have : ((2 : ℝ) ^ 78 * (1 + 1 / 16777216) + 1 / 10 ^ 45) * 2 ^ 41 ≤ 2 ^ 119 + 2 ^ 100 := by norm_num
    linarith
  -- z
  obtain ⟨zf, ze⟩ := muladd_lane p2f hs hs (by rw [hS, abs_of_nonneg hs0]; linarith) h3
  rw [hS, hu] at ze
  have hz : |z.toReal| ≤ 2 ^ 119 + 2 ^ 101 := by
    have := abs_le_of_err ze
    have h8 : |p2.toReal * S + S| ≤ 2 ^ 119 + 2 ^ 100 + 2 ^ 41 := by
      have := abs_add_le (p2.toReal * S) S; rw [abs_of_nonneg hs0] at this; linarith
    rw [abs_of_nonneg hs0] at this
    linarith
  -- inner, x2
  have hk0 : (0 : ℝ) < 6693141 / 8388608 := by norm_num
  have r4 : |(c32 0x3FE9884533D43651).toReal * z.toReal| ≤ 2 ^ 120 := by
    rw [hkv, abs_mul, abs_of_pos hk0]
    have := mul_le_mul_of_nonneg_left hz hk0.le
    have : (6693141 / 8388608 : ℝ) * (2 ^ 119 + 2 ^ 101) ≤ 2 ^ 120 := by norm_num
    linarith
  obtain ⟨inf, ine⟩ := mul_lane hkf zf r4 h4
  rw [hkv, hu] at ine
  have hin : |inn.toReal| ≤ 2 ^ 119 := by
    have := abs_le_of_err ine
    have : |(6693141 / 8388608 : ℝ) * z.toReal| ≤ 6693141 / 8388608 * (2 ^ 119 + 2 ^ 101) := by
      rw [abs_mul, abs_of_pos hk0]; exact mul_le_mul_of_nonneg_left hz hk0.le
    have : (6693141 / 8388608 : ℝ) * (2 ^ 119 + 2 ^ 101) * (1 + 1 / 16777216) + 1 / 10 ^ 45 ≤ 2 ^ 119 := by
      norm_num
    nlinarith [abs_nonneg ((6693141 / 8388608 : ℝ) * z.toReal)]
  have r5 : |(c32 0x4000000000000000).toReal * inn.toReal| ≤ 2 ^ 120 := by
    rw [h2v, abs_mul, abs_two]; linarith
  obtain ⟨x2f, x2e⟩ := mul_lane h2f inf r5 h5
  rw [h2v, hu] at x2e
  have hX := x2_core (aF := 12003091 / 268435456) (kF := 6693141 / 8388608) hs0 hs1 (by norm_num)
    (by norm_num) (by rw [hu, abs_of_nonneg hA0]; exact p1e) (by rw [hu]; exact p2e)
    (by rw [hu, abs_of_nonneg hs0] at *; exact ze) (by rw [hu]; exact ine) (by rw [hu]; exact x2e)
  have hX0 : 0 ≤ Xof S := Xof_nonneg hs0
  -- the logistic
  obtain ⟨nxf, nxv⟩ := neg_lane x2f h6
  have hnx1 : nx.toReal ≤ 1 := by
    rw [nxv]
    have h9 := (abs_le.mp hX).1
    have h10 : (2 : ℝ) ^ (-20 : ℤ) * Xof S ≤ Xof S := by
      have : (2 : ℝ) ^ (-20 : ℤ) ≤ 1 := by norm_num
      have := mul_le_mul_of_nonneg_right this hX0
      linarith
    have : (2 : ℝ) ^ (-120 : ℤ) ≤ 1 := by norm_num
    linarith
  obtain ⟨exf, exe⟩ := exp_lane hP nxf (exp_room hnx1) h7
  rw [nxv, abs_neg] at exe
  have hsc := sig_chain hX0 hX exe
  have hex1 := hsc.1
  have r8 : |(FloatV.ofRat 1 : F32).toReal + ex.toReal| ≤ 2 ^ 120 := by
    rw [hsv, abs_of_pos (by linarith)]; linarith
  obtain ⟨denf, dene⟩ := add_lane hsf exf r8 h8
  rw [hsv] at dene
  have hden0 : 49 / 100 ≤ den.toReal := by
    have := (abs_le.mp dene).1
    rw [hu, abs_of_pos (by linarith : (0 : ℝ) < 1 + ex.toReal)] at this
    nlinarith
  have hdv0 : den.toReal ≠ 0 := by linarith
  have r9 : |(FloatV.ofRat 1 : F32).toReal / den.toReal| ≤ 2 ^ 126 := by
    rw [hsv, abs_of_pos (div_pos one_pos (by linarith))]
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  obtain ⟨sgf, sge⟩ := div_real hP hsf denf hdv0 r9 h9
  rw [hsv] at sge
  obtain ⟨hsg, -, -⟩ := sig_core (X := Xof S) hsc dene sge
  have hσ1 := sig_le_one (Xof S)
  have hσ2 := sig_ge_half hX0
  have hsgb : |sg.toReal| ≤ 2 := by
    have := abs_le_of_err hsg
    rw [abs_of_nonneg (show (0 : ℝ) ≤ sig (Xof S) by linarith)] at this
    have : (2 : ℝ) ^ (-18 : ℤ) ≤ 1 := by norm_num
    linarith
  -- 2 sg - 1, 1 + tanh
  have r10 : |(c32 0x4000000000000000).toReal * sg.toReal| ≤ 2 ^ 120 := by
    rw [h2v, abs_mul, abs_two]; linarith
  obtain ⟨twf, twe⟩ := mul_lane h2f sgf r10 h10
  rw [h2v] at twe
  have twb : |tw.toReal| ≤ 5 := by
    have := abs_le_of_err twe; rw [hu, abs_mul, abs_two] at this; linarith
  have hn1f : (FloatV.neg (c32 0x3FF0000000000000)).isFinite = true :=
    (FloatV.isFinite_neg _).trans h1f
  have hn1v : (FloatV.neg (c32 0x3FF0000000000000)).toReal = -1 := by
    rw [FloatV.toReal_neg, h1v]
  obtain ⟨thf, the⟩ := addc_real twf hn1f (by rw [hn1v, abs_neg, abs_one]; linarith) h11
  rw [hn1v] at the
  have thb : |th.toReal| ≤ 7 := by
    have := abs_le_of_err the; rw [hu, abs_neg, abs_one] at this
    have : |tw.toReal + -1| ≤ 6 := by
      have := abs_add_le tw.toReal (-1); rw [abs_neg, abs_one] at this; linarith
    nlinarith [abs_nonneg tw.toReal]
  have r13 : |(c32 0x3FF0000000000000).toReal + th.toReal| ≤ 2 ^ 120 := by
    rw [h1v]
    have := abs_add_le 1 th.toReal; rw [abs_one] at this; linarith
  obtain ⟨otf, ote⟩ := add_lane h1f thf r13 h13
  rw [h1v] at ote
  -- 0.5 s and the product
  have r12 : |(c32 0x3FE0000000000000).toReal * s.toReal| ≤ 2 ^ 120 := by
    rw [hhv, hS, abs_of_nonneg (by linarith)]; linarith
  obtain ⟨hf, he12⟩ := mul_lane hhf hs r12 h12
  rw [hhv, hS] at he12
  have hb12 : |h.toReal| ≤ 2 ^ 41 := by
    have := abs_le_of_err he12; rw [hu, abs_of_nonneg (by linarith : (0 : ℝ) ≤ 1 / 2 * S)] at this
    linarith
  have otb : |ot.toReal| ≤ 9 := by
    have := abs_le_of_err ote; rw [hu] at this
    have : |1 + th.toReal| ≤ 8 := by
      have := abs_add_le 1 th.toReal; rw [abs_one] at this; linarith
    nlinarith [abs_nonneg (1 + th.toReal)]
  have r14 : |h.toReal * ot.toReal| ≤ 2 ^ 120 := by
    rw [abs_mul]
    calc |h.toReal| * |ot.toReal| ≤ 2 ^ 41 * 9 := mul_le_mul hb12 otb (abs_nonneg _) (by positivity)
      _ ≤ 2 ^ 120 := by norm_num
  obtain ⟨actf, acte⟩ := mul_lane hf otf r14 h14
  obtain ⟨hcore, -⟩ := act_core hs0 hs1 hX0 hsg twe the ote he12 acte
  refine ⟨actf, ?_, fun h0 => ?_⟩
  · rw [geluRef_eq]; exact hcore
  · obtain ⟨hf', hh0⟩ := mul_zero_right hhf hs h0 h12
    exact (mul_zero_left hf' hh0 otf h14).2

end Claims.GeluAndMulSparse
