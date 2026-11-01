import AttnFwdE5m2.Mag

/-!
# `attn_fwd_e5m2`: the output's accuracy, centred on the reference

`acc` and `l_i` are rescaled by the same `alpha` (lines 95 and 106), and each attended key's fp8
`p` lies between `a8` and `b8` times its exact weight in the running maximum's units (up to `eps8`).
So for a fixed `r` (the reference `attnRef`) and `kap`, the middle of `[a8, b8]`, `acc − r·l_i`
stays near `kap` times the scaled centred sum `Σ_j w_j·(v_j − r)`: the score's error enters only
as the spread `hw` of the ratios, times `Σ w_j |v_j − r|`. After the last block the centred sum is
`0` for `r = attnRef`.

* `cstep_gen`: one block of `|acc − r·l_i − kap·2^-m·C| ≤ 2^-m·ε_C + A·l_i`, in real numbers, for
  either arm of the `p·v` `tl.dot` (`Model.errStep`, `Model.accStep`);
* `lane_cen`, `block_cen`, `block_qv`: one lane's and one block's `q·v − r·p − kap·w·(v − r)` and
  `|q·v|`;
* `cl2_cen`: the `p·v` `tl.dot`'s chunks' largest lifted products against the weights;
* `Cx`, `bC`, `bA`, `bLW`, `bVL`, `bV`, `bVa`, `stR`, `acR`, `ecR`: the row's centred sums, block
  sums and the recursion of `Model.stC` (`Model.accA`, `Model.errC`) in the row's terms, at any
  choice `W` of `Model.stepC`; `CInv`: the centred bound, for every `r` and `W`.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The constants -/

theorem a8_pos {pe : ℝ} (h : 1 ≤ pe) : 0 < a8 pe := by
  unfold a8; have : 0 < pe := by linarith
  positivity

theorem a8_le {pe : ℝ} (h : 1 ≤ pe) : a8 pe ≤ 7 / 8 := by
  unfold a8
  rw [div_le_iff₀ (by linarith)]
  nlinarith

theorem b8_ge {pe : ℝ} (h : 1 ≤ pe) : 9 / 8 ≤ b8 pe := by unfold b8; nlinarith

theorem kap_nonneg {pe : ℝ} (h : 1 ≤ pe) : 0 ≤ kap pe := by
  unfold kap; have := a8_pos h; have := b8_ge h; linarith

theorem hw_nonneg {pe : ℝ} (h : 1 ≤ pe) : 0 ≤ hw pe := by
  unfold hw; have := a8_le h; have := b8_ge h; linarith

theorem etaA_nonneg {dm : ℝ} (h : 0 ≤ dm) : 0 ≤ etaA dm := by unfold etaA; nlinarith

theorem etaA_le {dm : ℝ} (h0 : 0 ≤ dm) (h1 : dm ≤ 1 / 2 ^ 13) : etaA dm ≤ 1 / 2 ^ 12 := by
  unfold etaA; nlinarith

theorem gD_eq (cs dm : ℝ) :
    gD cs dm = (1 + cs) * ((1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24))) - 1 := by
  unfold gD; ring

theorem gD_nonneg {cs dm : ℝ} (hc : 0 ≤ cs) (h : 0 ≤ dm) : 0 ≤ gD cs dm := by
  rw [gD_eq]
  have : (1 : ℝ) ≤ (1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) := by nlinarith
  nlinarith

theorem gD_le {cs dm : ℝ} (hc0 : 0 ≤ cs) (hc1 : cs ≤ 1 / 50) (h0 : 0 ≤ dm) (h1 : dm ≤ 1 / 2 ^ 13) :
    gD cs dm ≤ 1 / 32 := by
  rw [gD_eq]
  have k1 : (1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) ≤ (1 + 1 / 2 ^ 13) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) :=
    mul_le_mul_of_nonneg_right (by linarith) (by norm_num)
  have k2 : 0 ≤ (1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) := by positivity
  have k3 := mul_le_mul (by linarith : 1 + cs ≤ 1 + 1 / 50) k1 k2 (by norm_num)
  have k4 : (1 + 1 / 50 : ℝ) * ((1 + 1 / 2 ^ 13) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24))) - 1 ≤ 1 / 32 := by norm_num
  linarith

theorem denX_ge {dm : ℝ} (h0 : 0 ≤ dm) (h1 : dm ≤ 1 / 2 ^ 13) : 1 / 2 ≤ denX dm := by
  unfold denX; have := etaA_le h0 h1; have := etaA_nonneg h0; linarith

theorem denS_ge : 1 / 2 ≤ denS := by
  unfold denS; rw [gam63_r]; have := gamr_le; have := gamr_nonneg; linarith

theorem coefD_nonneg {cr : ℝ} (h : 0 ≤ cr) : 0 ≤ coefD cr := by
  unfold coefD; rw [gam63_r]; have := gamr_nonneg; positivity

theorem accStep_nonneg {dm cc cp cr R A Ba Bq Bv : ℝ} (h0 : 0 ≤ dm) (h1 : dm ≤ 1 / 2 ^ 13) (hcc : 0 ≤ cc)
    (hcp : 0 ≤ cp) (hcr : 0 ≤ cr) (hR : 0 ≤ R) (hA : 0 ≤ A) (hBa : 0 ≤ Ba) (hBq : 0 ≤ Bq) (hBv : 0 ≤ Bv) :
    0 ≤ accStep dm cc cp cr R A Ba Bq Bv := by
  unfold accStep
  have hd := denS_ge
  have hc := coefD_nonneg hcr
  have hM : 0 ≤ max ((A + 3 * (1 + etaA dm) * R / 2 ^ 24) / denX dm) (R * coefD cr / denS) :=
    le_max_of_le_right (div_nonneg (mul_nonneg hR hc) (by linarith))
  rw [eps8_r]
  positivity

theorem errStep_nonneg {pe dm cs cc cp EC C R A Lw Bc Bq Bl : ℝ} (hpe : 1 ≤ pe) (h0 : 0 ≤ dm)
    (hcs : 0 ≤ cs) (hcc : 0 ≤ cc) (hcp : 0 ≤ cp) (hEC : 0 ≤ EC) (hR : 0 ≤ R) (hA : 0 ≤ A) (hLw : 0 ≤ Lw)
    (hBc : 0 ≤ Bc) (hBq : 0 ≤ Bq) (hBl : 0 ≤ Bl) :
    0 ≤ errStep pe dm cs cc cp EC C R A Lw Bc Bq Bl := by
  unfold errStep
  have := gD_nonneg hcs h0
  have := etaA_nonneg h0
  have := kap_nonneg hpe
  have := hw_nonneg hpe
  have := b8_ge hpe
  positivity

/-- `errStep` scales with the units of its centred terms. -/
theorem errStep_scale (pe dm cs cc cp EC C R A Lw Bc Bq Bl : ℝ) {k : ℝ} (hk : 0 ≤ k) :
    errStep pe dm cs cc cp (k * EC) (k * C) R A (k * Lw) (k * Bc) (k * Bq) (k * Bl) =
      k * errStep pe dm cs cc cp EC C R A Lw Bc Bq Bl := by
  unfold errStep
  rw [abs_mul, abs_of_nonneg hk]
  ring

/-- The two arms' constants. -/
theorem arm_consts :
    (0 : ℝ) ≤ (cS8 : ℝ) ∧ (cS8 : ℝ) ≤ 1 / 50 ∧ (0 : ℝ) ≤ (cC8 : ℝ) ∧ (0 : ℝ) ≤ (cP8 : ℝ) ∧
      (0 : ℝ) ≤ (cA8 : ℝ) ∧ (cA8 : ℝ) ≤ 1 / 2 ^ 120 ∧
      (0 : ℝ) ≤ (lS8 : ℝ) ∧ (lS8 : ℝ) ≤ 1 / 50 ∧ (0 : ℝ) ≤ (lC8 : ℝ) ∧ (0 : ℝ) ≤ (lP8 : ℝ) ∧
      (0 : ℝ) ≤ (lA8 : ℝ) ∧ (lA8 : ℝ) ≤ 1 / 2 ^ 120 := by
  rw [cS8_r, cC8_r, cP8_r, cA8_r, lS8_r, lC8_r, lP8_r, lA8_r]
  norm_num

/-! ## One block, in real numbers -/

set_option maxHeartbeats 1600000 in
/-- **One block of the centred bound** (lines 95, 101–103, 106), in real numbers, for either arm
of the `p·v` `tl.dot` (constants `cs`, `cc`, `cp`, `ca`; `cr` the share of `cc·Σ|q·v|` charged
against `|r|`). `a`, `l` are the old element of `acc` and `l_i`, `α` is `alpha` (within
`etaA·λ + 2^-126` of `λ`), `a1` the element of `acc * alpha`, `a'` that of `tl.dot(p, v, acc)`
(`hdot`: its error over two chunks, with `QA` the lanes' `Σ|q·v|`, split by `hQA`, and `QL` the
chunks' largest lifted products), `lij` the block's `tl.sum(p)` (`P`, `S` the lanes' `Σ p` and
`Σ|p|`), `l'` the new `l_i`. The old bound is scaled to the new maximum (`C0`, `EC0`,
`λ·l ≤ Lu + Lab`); the block brings `Q = Σ q·v` within `hw·Bc + …` of `r·P + kap·Cb` (`Cb` its
scaled centred sum, `Bc` its `Σ w|v − r|`). -/
theorem cstep_gen {pe dm cs cc cp ca cr : ℝ} (hpe : 1 ≤ pe) (hdm0 : 0 ≤ dm) (hdm1 : dm ≤ 1 / 2 ^ 13)
    (hcs0 : 0 ≤ cs) (hcs1 : cs ≤ 1 / 50) (hcc0 : 0 ≤ cc) (hcp0 : 0 ≤ cp) (hca0 : 0 ≤ ca)
    (hca1 : ca ≤ 1 / 2 ^ 120) (hcr0 : 0 ≤ cr)
    {a a1 a' l l' α lij r lam C0 Cb EC0 A Q P S QA QL Bc Bq Bl Ba Bqa Bv Lu Lab tau : ℝ}
    (hlam0 : 0 ≤ lam) (hl0 : 0 ≤ l) (hl1 : l ≤ 2 ^ 44)
    (hα : |α - lam| ≤ etaA dm * lam + 1 / 2 ^ 126)
    (htau0 : 0 ≤ tau) (htau : tau ≤ 1 / 2 ^ 66)
    (ha1 : |a1 - lam * a| ≤ ((1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) - 1) * (lam * |a|) + tau)
    (hA0 : 0 ≤ A) (hEC0 : 0 ≤ EC0)
    (hI : |lam * a - r * (lam * l) - kap pe * C0| ≤ EC0 + A * (lam * l))
    (hLu0 : 0 ≤ Lu) (hLab0 : 0 ≤ Lab) (hLab : Lab ≤ 1 / 2 ^ 56) (hLu : lam * l ≤ Lu + Lab)
    (hdot : |a' - (a1 + Q)| ≤ cs * |a1| + cc * QA + cp * QL + ca)
    (hBc : 0 ≤ Bc) (hBq : 0 ≤ Bq) (hBl : 0 ≤ Bl) (hBa : 0 ≤ Ba) (hBqa : 0 ≤ Bqa) (hBv : 0 ≤ Bv)
    (hQA : cc * QA ≤ cc * b8 pe * Bq + cc * (eps8 : ℝ) * Bqa + cr * |r| * (9 / 8 * S + 1 / 2 ^ 11))
    (hQL : QL ≤ b8 pe * Bl + ((eps8 : ℝ) + 1 / 2 ^ 14) * Bv)
    (hT : |Q - r * P - kap pe * Cb| ≤ hw pe * Bc + (eps8 : ℝ) * Ba + |r| * S / 8 + |r| / 2 ^ 11)
    (hS0 : 0 ≤ S) (hP1 : S - 1 / 2 ^ 118 ≤ P) (hP2 : |P| ≤ S)
    (hlij : |lij - P| ≤ gamr * S + 1 / 2 ^ 142)
    (hl' : |l' - (l * α + lij)| ≤ 3 * (|l * α| + |lij|) / 2 ^ 24 + 4 / 2 ^ 150)
    (hl'h : 1 / 2 ≤ l') :
    |a' - r * l' - kap pe * (C0 + Cb)| ≤
      errStep pe dm cs cc cp EC0 C0 |r| A Lu Bc Bq Bl + accStep dm cc cp cr |r| A Ba Bqa Bv * l' := by
  -- the constants
  have hgA0 := gD_nonneg hcs0 hdm0
  have hgA1 := gD_le hcs0 hcs1 hdm0 hdm1
  have hgAe := gD_eq cs dm
  have hη0 := etaA_nonneg hdm0
  have hη1 := etaA_le hdm0 hdm1
  have hK0 := kap_nonneg hpe
  have hH0 := hw_nonneg hpe
  have hB0 : 0 ≤ b8 pe := by have := b8_ge hpe; linarith
  have hdX := denX_ge hdm0 hdm1
  have hdS := denS_ge
  have hcS := coefD_nonneg hcr0
  have hγ0 := gamr_nonneg
  have hγ1 := gamr_le
  have he8 : (eps8 : ℝ) = 1 / 2 ^ 17 + 1 / 2 ^ 125 := eps8_r
  have hdXd : denX dm = 1 - etaA dm - 3 * (1 + etaA dm) / 2 ^ 24 := rfl
  have hdSd : denS = 1 - gamr - 3 * (1 + gamr) / 2 ^ 24 := by unfold denS; rw [gam63_r]
  have hcSd : coefD cr = 1 / 8 + 9 / 8 * cr + gamr + 3 * (1 + gamr) / 2 ^ 24 := by
    unfold coefD; rw [gam63_r]
  have hR0 : 0 ≤ |r| := abs_nonneg r
  have hX0 : 0 ≤ lam * l := mul_nonneg hlam0 hl0
  have hΦ1 : 1 ≤ (1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) := by nlinarith
  have hcS0 : 0 ≤ cs := hcs0
  have hcC0 : 0 ≤ cc := hcc0
  have hcP0 : 0 ≤ cp := hcp0
  have he80 : 0 ≤ (eps8 : ℝ) := by rw [he8]; norm_num
  unfold errStep accStep
  set M := max ((A + 3 * (1 + etaA dm) * |r| / 2 ^ 24) / denX dm) (|r| * coefD cr / denS) with hMd
  -- `M` against its two parts
  have hM1 : A + 3 * (1 + etaA dm) * |r| / 2 ^ 24 ≤ M * denX dm := by
    have := le_max_left ((A + 3 * (1 + etaA dm) * |r| / 2 ^ 24) / denX dm) (|r| * coefD cr / denS)
    rw [← hMd, div_le_iff₀ (by linarith)] at this; exact this
  have hM2 : |r| * coefD cr ≤ M * denS := by
    have := le_max_right ((A + 3 * (1 + etaA dm) * |r| / 2 ^ 24) / denX dm) (|r| * coefD cr / denS)
    rw [← hMd, div_le_iff₀ (by linarith)] at this; exact this
  have hM0 : 0 ≤ M := by
    rw [hMd]; exact le_max_of_le_right (div_nonneg (mul_nonneg hR0 hcS) (by linarith))
  -- the old element, scaled
  have hla : |lam * a| ≤ |r| * (lam * l) + kap pe * |C0| + EC0 + A * (lam * l) := by
    have e : lam * a = (lam * a - r * (lam * l) - kap pe * C0) + r * (lam * l) + kap pe * C0 := by ring
    have h1 : |lam * a| ≤ |lam * a - r * (lam * l) - kap pe * C0| + |r * (lam * l)| + |kap pe * C0| := by
      calc |lam * a| = |(lam * a - r * (lam * l) - kap pe * C0) + r * (lam * l) + kap pe * C0| := by rw [← e]
        _ ≤ |(lam * a - r * (lam * l) - kap pe * C0) + r * (lam * l)| + |kap pe * C0| := abs_add_le _ _
        _ ≤ |lam * a - r * (lam * l) - kap pe * C0| + |r * (lam * l)| + |kap pe * C0| := by
          linarith [abs_add_le (lam * a - r * (lam * l) - kap pe * C0) (r * (lam * l))]
    rw [abs_mul r (lam * l), abs_of_nonneg hX0, abs_mul (kap pe) C0, abs_of_nonneg hK0] at h1
    linarith
  have hla' : lam * |a| = |lam * a| := by rw [abs_mul, abs_of_nonneg hlam0]
  rw [hla'] at ha1
  -- the dot's and the rescaling's errors
  have ha1b : |a1| ≤ (1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) * |lam * a| + tau := by
    have := abs_sub_abs_le_abs_sub a1 (lam * a)
    linarith
  have hT12 : |a' - (a1 + Q)| + |a1 - lam * a| ≤ gD cs dm * |lam * a| + (1 + cs) * tau +
      cc * QA + cp * QL + ca := by
    have k1 : cs * |a1| ≤ cs * ((1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) * |lam * a| + tau) :=
      mul_le_mul_of_nonneg_left ha1b hcS0
    have e : cs * ((1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) * |lam * a| + tau) +
        (((1 + dm) * ((1 + 1 / 2 ^ 22) * (1 + 1 / 2 ^ 24)) - 1) * |lam * a| + tau) =
        gD cs dm * |lam * a| + (1 + cs) * tau := by
      rw [hgAe]; ring
    linarith
  have hgla : gD cs dm * |lam * a| ≤ gD cs dm * (|r| * (lam * l) + kap pe * |C0| + EC0 + A * (lam * l)) :=
    mul_le_mul_of_nonneg_left hla hgA0
  have hgX : gD cs dm * ((|r| + A) * (lam * l)) ≤ gD cs dm * ((|r| + A) * (Lu + Lab)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hLu (by linarith)) hgA0
  have hQA' := hQA
  have hQL' : cp * QL ≤ cp * (b8 pe * Bl + ((eps8 : ℝ) + 1 / 2 ^ 14) * Bv) :=
    mul_le_mul_of_nonneg_left hQL hcP0
  -- `alpha` on `l_i`
  have hαl : |α| ≤ (1 + etaA dm) * lam + 1 / 2 ^ 126 := by
    have := abs_sub_abs_le_abs_sub α lam
    rw [abs_of_nonneg hlam0] at this; linarith
  have hl126 : l * (1 / 2 ^ 126) ≤ 1 / 2 ^ 82 := by
    have := mul_le_mul_of_nonneg_right hl1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 126); norm_num at this ⊢; linarith
  have hlα : |l * α| ≤ (1 + etaA dm) * (lam * l) + 1 / 2 ^ 82 := by
    rw [abs_mul, abs_of_nonneg hl0]
    have h1 := mul_le_mul_of_nonneg_left hαl hl0
    have e : l * ((1 + etaA dm) * lam + 1 / 2 ^ 126) = (1 + etaA dm) * (lam * l) + l * (1 / 2 ^ 126) := by ring
    linarith
  have hT5 : |r * (lam * l - l * α)| ≤ etaA dm * |r| * (Lu + Lab) + |r| / 2 ^ 82 := by
    rw [abs_mul, show lam * l - l * α = l * (lam - α) by ring, abs_mul, abs_of_nonneg hl0, abs_sub_comm]
    have h1 := mul_le_mul_of_nonneg_left hα hl0
    have h3 : l * (etaA dm * lam + 1 / 2 ^ 126) ≤ etaA dm * (Lu + Lab) + 1 / 2 ^ 82 := by
      have := mul_le_mul_of_nonneg_left hLu hη0
      have e : l * (etaA dm * lam + 1 / 2 ^ 126) = etaA dm * (lam * l) + l * (1 / 2 ^ 126) := by ring
      linarith
    have h4 := mul_le_mul_of_nonneg_left (h1.trans h3) hR0
    have e2 : |r| * (etaA dm * (Lu + Lab) + 1 / 2 ^ 82) = etaA dm * |r| * (Lu + Lab) + |r| / 2 ^ 82 := by ring
    linarith
  -- `l_ij`
  have hlijabs : |lij| ≤ (1 + gamr) * S + 1 / 2 ^ 142 := by
    have := abs_sub_abs_le_abs_sub lij P; linarith
  have hT6 : |r * (P - lij)| ≤ |r| * (gamr * S) + |r| / 2 ^ 142 := by
    rw [abs_mul, abs_sub_comm]
    have := mul_le_mul_of_nonneg_left hlij hR0
    linarith [show |r| * (gamr * S + 1 / 2 ^ 142) = |r| * (gamr * S) + |r| / 2 ^ 142 by ring]
  -- `l'`
  have hl'r : |l' - (l * α + lij)| ≤ 3 * (1 + etaA dm) * (lam * l) / 2 ^ 24 + 3 * (1 + gamr) * S / 2 ^ 24 +
      1 / 2 ^ 100 := by
    have h1 : 3 * (|l * α| + |lij|) / 2 ^ 24 ≤
        3 * ((1 + etaA dm) * (lam * l) + 1 / 2 ^ 82 + ((1 + gamr) * S + 1 / 2 ^ 142)) / 2 ^ 24 :=
      div_le_div_of_nonneg_right (by linarith) (by norm_num)
    have e : 3 * ((1 + etaA dm) * (lam * l) + 1 / 2 ^ 82 + ((1 + gamr) * S + 1 / 2 ^ 142)) / 2 ^ 24 =
        3 * (1 + etaA dm) * (lam * l) / 2 ^ 24 + 3 * (1 + gamr) * S / 2 ^ 24 +
          3 * (1 / 2 ^ 82 + 1 / 2 ^ 142) / 2 ^ 24 := by ring
    have h2 : 3 * (1 / 2 ^ 82 + 1 / 2 ^ 142 : ℝ) / 2 ^ 24 + 4 / 2 ^ 150 ≤ 1 / 2 ^ 100 := by norm_num
    linarith
  have hT7 : |r * (l * α + lij - l')| ≤ |r| * (3 * (1 + etaA dm) * (lam * l) / 2 ^ 24) +
      |r| * (3 * (1 + gamr) * S / 2 ^ 24) + |r| / 2 ^ 100 := by
    rw [abs_mul, show l * α + lij - l' = -(l' - (l * α + lij)) by ring, abs_neg]
    have := mul_le_mul_of_nonneg_left hl'r hR0
    linarith [show |r| * (3 * (1 + etaA dm) * (lam * l) / 2 ^ 24 + 3 * (1 + gamr) * S / 2 ^ 24 + 1 / 2 ^ 100) =
      |r| * (3 * (1 + etaA dm) * (lam * l) / 2 ^ 24) + |r| * (3 * (1 + gamr) * S / 2 ^ 24) + |r| / 2 ^ 100 by ring]
  -- `l'` from below
  have hl'low : denX dm * (lam * l) + denS * S ≤ l' + 1 / 2 ^ 80 := by
    have h1 := (abs_le.mp hl').1
    have hlα0 : (1 - etaA dm) * (lam * l) - 1 / 2 ^ 82 ≤ l * α := by
      have hα1 : lam - (etaA dm * lam + 1 / 2 ^ 126) ≤ α := by have := (abs_le.mp hα).1; linarith
      have := mul_le_mul_of_nonneg_left hα1 hl0
      have e : l * (lam - (etaA dm * lam + 1 / 2 ^ 126)) = (1 - etaA dm) * (lam * l) - l * (1 / 2 ^ 126) := by ring
      linarith
    have hlij0 : (1 - gamr) * S - 1 / 2 ^ 117 ≤ lij := by
      have := (abs_le.mp hlij).1
      have : (1 : ℝ) / 2 ^ 142 + 1 / 2 ^ 118 ≤ 1 / 2 ^ 117 := by norm_num
      linarith
    have h3 : 3 * (|l * α| + |lij|) / 2 ^ 24 ≤
        3 * ((1 + etaA dm) * (lam * l) + 1 / 2 ^ 82 + ((1 + gamr) * S + 1 / 2 ^ 142)) / 2 ^ 24 :=
      div_le_div_of_nonneg_right (by linarith) (by norm_num)
    rw [hdXd, hdSd]
    have h4 : (1 / 2 ^ 82 : ℝ) + 1 / 2 ^ 117 + 3 * (1 / 2 ^ 82 + 1 / 2 ^ 142) / 2 ^ 24 + 4 / 2 ^ 150 ≤
        1 / 2 ^ 80 := by norm_num
    linarith
  have hfin : (A + 3 * (1 + etaA dm) * |r| / 2 ^ 24) * (lam * l) + |r| * coefD cr * S ≤ M * l' + M / 2 ^ 80 := by
    have k1 := mul_le_mul_of_nonneg_right hM1 hX0
    have k2 := mul_le_mul_of_nonneg_right hM2 hS0
    have k3 := mul_le_mul_of_nonneg_left hl'low hM0
    linarith only [k1, k2, k3]
  rw [hcSd] at hfin
  have hMl : M / 2 ^ 80 ≤ 1 / 2 ^ 50 * M * l' := by
    have := mul_le_mul_of_nonneg_left hl'h hM0
    linarith only [this, hM0]
  -- the block's absolute terms, against `l' ≥ 1/2`
  set Ab := (eps8 : ℝ) * Ba + cc * (eps8 : ℝ) * Bqa + cp * ((eps8 : ℝ) + 1 / 2 ^ 14) * Bv +
    |r| * (1 + cr) / 2 ^ 11 + (1 + |r| + A) / 2 ^ 50 with hAbd
  have hAb0 : 0 ≤ Ab := by rw [hAbd]; positivity
  have hAb : Ab ≤ 2 * Ab * l' := by
    have := mul_le_mul_of_nonneg_left hl'h hAb0
    linarith only [this]
  -- the small terms
  have htiny : (1 + cs) * tau + ca + gD cs dm * ((|r| + A) * Lab) + etaA dm * |r| * Lab +
      |r| / 2 ^ 82 + |r| / 2 ^ 142 + |r| / 2 ^ 100 ≤ (1 + |r| + A) / 2 ^ 50 := by
    have k1 : (1 + cs) * tau ≤ 2 * (1 / 2 ^ 66) := mul_le_mul (by linarith) htau htau0 (by norm_num)
    have k2 : gD cs dm * ((|r| + A) * Lab) ≤ 1 / 32 * ((|r| + A) * (1 / 2 ^ 56)) :=
      mul_le_mul hgA1 (mul_le_mul_of_nonneg_left hLab (by linarith)) (by positivity) (by norm_num)
    have k3 : etaA dm * |r| * Lab ≤ 1 / 2 ^ 12 * |r| * (1 / 2 ^ 56) :=
      mul_le_mul (mul_le_mul_of_nonneg_right hη1 hR0) hLab hLab0 (by positivity)
    linarith
  have hηA : 0 ≤ etaA dm * A * Lu := by positivity
  -- the parts of `a' − r·l' − kap·(C0 + Cb)`
  have e : a' - r * l' - kap pe * (C0 + Cb) = (a' - (a1 + Q)) + (a1 - lam * a) +
      (lam * a - r * (lam * l) - kap pe * C0) + (Q - r * P - kap pe * Cb) + r * (lam * l - l * α) +
      r * (P - lij) + r * (l * α + lij - l') := by ring
  rw [e]
  have t1 := abs_add_le ((a' - (a1 + Q)) + (a1 - lam * a) + (lam * a - r * (lam * l) - kap pe * C0) +
      (Q - r * P - kap pe * Cb) + r * (lam * l - l * α) + r * (P - lij)) (r * (l * α + lij - l'))
  have t2 := abs_add_le ((a' - (a1 + Q)) + (a1 - lam * a) + (lam * a - r * (lam * l) - kap pe * C0) +
      (Q - r * P - kap pe * Cb) + r * (lam * l - l * α)) (r * (P - lij))
  have t3 := abs_add_le ((a' - (a1 + Q)) + (a1 - lam * a) + (lam * a - r * (lam * l) - kap pe * C0) +
      (Q - r * P - kap pe * Cb)) (r * (lam * l - l * α))
  have t4 := abs_add_le ((a' - (a1 + Q)) + (a1 - lam * a) + (lam * a - r * (lam * l) - kap pe * C0))
    (Q - r * P - kap pe * Cb)
  have t5 := abs_add_le ((a' - (a1 + Q)) + (a1 - lam * a)) (lam * a - r * (lam * l) - kap pe * C0)
  have t6 := abs_add_le (a' - (a1 + Q)) (a1 - lam * a)
  have hsum : |(a' - (a1 + Q)) + (a1 - lam * a) + (lam * a - r * (lam * l) - kap pe * C0) +
      (Q - r * P - kap pe * Cb) + r * (lam * l - l * α) + r * (P - lij) + r * (l * α + lij - l')| ≤
      (|a' - (a1 + Q)| + |a1 - lam * a|) + |lam * a - r * (lam * l) - kap pe * C0| +
      |Q - r * P - kap pe * Cb| + |r * (lam * l - l * α)| + |r * (P - lij)| + |r * (l * α + lij - l')| := by
    linarith only [t1, t2, t3, t4, t5, t6]
  linarith only [hsum, hT12, hI, hT, hT5, hT6, hT7, hgla, hgX, hQA', hQL', hfin, hMl, hAb, hAbd, htiny, hηA]

/-! ## One lane and one block -/

/-- **One lane** of the centred bound: `q` the fp8 `p` (between `a8·w − eps` and `b8·w + eps` for
an attended key of scaled weight `w`, within one rounding of `p`; `0` for a masked key, whose `p` is
below `2^-125`). With `kap`, `hw` the middle and half-width of `[a8, b8]`: `q·v − r·p` is within
`(hw·w + eps)·|v − r| + |r|·(|p|/8 + 2^-17)` of `kap·w·(v − r)` (`0` for a masked key), and `|q·v|`
at most `(b8·w + eps)·|v − r| + |r|·(9/8·|p| + 2^-17)`. -/
theorem lane_cen {q p v w r a8 b8 eps : ℝ} {al : Bool} (hw0 : 0 ≤ w) (heps : 0 ≤ eps) (ha8 : 0 ≤ a8)
    (hab : a8 ≤ b8)
    (hatt : al = true → a8 * w - eps ≤ q ∧ q ≤ b8 * w + eps ∧ |q - p| ≤ |p| / 8 + 1 / 2 ^ 17)
    (hmask : al = false → q = 0 ∧ |p| ≤ 1 / 2 ^ 125) :
    |q * v - r * p - (a8 + b8) / 2 * (if al = true then w * (v - r) else 0)| ≤
        (b8 - a8) / 2 * (if al = true then w * |v - r| else 0) + eps * (if al = true then |v - r| else 0) +
          |r| * |p| / 8 + |r| / 2 ^ 17 ∧
      |q * v| ≤ b8 * (if al = true then w * |v - r| else 0) + eps * (if al = true then |v - r| else 0) +
          |r| * (9 / 8 * |p| + 1 / 2 ^ 17) := by
  have hr := abs_nonneg r
  have hpa := abs_nonneg p
  cases al
  · obtain ⟨rfl, hp⟩ := hmask rfl
    simp only [Bool.false_eq_true, ↓reduceIte, mul_zero, zero_mul, zero_sub, sub_zero, abs_neg, abs_zero,
      add_zero, zero_add]
    constructor
    · rw [abs_mul]
      have h1 : |r| * |p| ≤ |r| * (1 / 2 ^ 125) := mul_le_mul_of_nonneg_left hp hr
      have h2 : |r| * (1 / 2 ^ 125) ≤ |r| / 2 ^ 17 := by
        rw [mul_one_div]; exact div_le_div_of_nonneg_left hr (by norm_num) (by norm_num)
      have h3 : 0 ≤ |r| * |p| / 8 := by positivity
      linarith
    · positivity
  · obtain ⟨hq1, hq2, hqp⟩ := hatt rfl
    simp only [↓reduceIte]
    have hqb : |q| ≤ 9 / 8 * |p| + 1 / 2 ^ 17 := by
      have := abs_sub_abs_le_abs_sub q p; linarith
    have hvr := abs_nonneg (v - r)
    constructor
    · have e : q * v - r * p - (a8 + b8) / 2 * (w * (v - r)) = (q - (a8 + b8) / 2 * w) * (v - r) + r * (q - p) := by
        ring
      rw [e]
      have h1 : |q - (a8 + b8) / 2 * w| ≤ (b8 - a8) / 2 * w + eps := by
        rw [abs_le]; constructor <;> nlinarith
      have h2 : |(q - (a8 + b8) / 2 * w) * (v - r)| ≤ ((b8 - a8) / 2 * w + eps) * |v - r| := by
        rw [abs_mul]; exact mul_le_mul_of_nonneg_right h1 hvr
      have h3 : |r * (q - p)| ≤ |r| * (|p| / 8 + 1 / 2 ^ 17) := by
        rw [abs_mul]; exact mul_le_mul_of_nonneg_left hqp hr
      have h4 := abs_add_le ((q - (a8 + b8) / 2 * w) * (v - r)) (r * (q - p))
      have e2 : ((b8 - a8) / 2 * w + eps) * |v - r| + |r| * (|p| / 8 + 1 / 2 ^ 17) =
          (b8 - a8) / 2 * (w * |v - r|) + eps * |v - r| + |r| * |p| / 8 + |r| / 2 ^ 17 := by ring
      linarith
    · have hq0 : |q| ≤ b8 * w + eps := by
        rw [abs_le]; constructor
        · have h1 : 0 ≤ a8 * w := mul_nonneg ha8 hw0
          have h2 : 0 ≤ b8 * w := mul_nonneg (ha8.trans hab) hw0
          linarith
        · linarith
      have hv : |v| ≤ |v - r| + |r| := by have := abs_sub_abs_le_abs_sub v r; linarith
      rw [abs_mul]
      have h1 : |q| * |v| ≤ |q| * |v - r| + |q| * |r| := by
        have := mul_le_mul_of_nonneg_left hv (abs_nonneg q); linarith
      have h2 : |q| * |v - r| ≤ (b8 * w + eps) * |v - r| := mul_le_mul_of_nonneg_right hq0 hvr
      have h3 : |q| * |r| ≤ (9 / 8 * |p| + 1 / 2 ^ 17) * |r| := mul_le_mul_of_nonneg_right hqb hr
      have e2 : (b8 * w + eps) * |v - r| + (9 / 8 * |p| + 1 / 2 ^ 17) * |r| =
          b8 * (w * |v - r|) + eps * |v - r| + |r| * (9 / 8 * |p| + 1 / 2 ^ 17) := by ring
      linarith

/-- **One block** of the centred bound: `lane_cen` summed over the 64 lanes. -/
theorem block_cen {q p v w : Fin 64 → ℝ} {al : Fin 64 → Bool} {r a8 b8 eps : ℝ} (heps : 0 ≤ eps)
    (ha8 : 0 ≤ a8) (hab : a8 ≤ b8) (hw0 : ∀ jj, 0 ≤ w jj)
    (hatt : ∀ jj, al jj = true →
      a8 * w jj - eps ≤ q jj ∧ q jj ≤ b8 * w jj + eps ∧ |q jj - p jj| ≤ |p jj| / 8 + 1 / 2 ^ 17)
    (hmask : ∀ jj, al jj = false → q jj = 0 ∧ |p jj| ≤ 1 / 2 ^ 125) :
    |∑ jj, q jj * v jj - r * ∑ jj, p jj -
        (a8 + b8) / 2 * ∑ jj, (if al jj = true then w jj * (v jj - r) else 0)| ≤
        (b8 - a8) / 2 * ∑ jj, (if al jj = true then w jj * |v jj - r| else 0) +
          eps * ∑ jj, (if al jj = true then |v jj - r| else 0) + |r| * (∑ jj, |p jj|) / 8 + |r| / 2 ^ 11 ∧
      ∑ jj, |q jj * v jj| ≤ b8 * ∑ jj, (if al jj = true then w jj * |v jj - r| else 0) +
          eps * ∑ jj, (if al jj = true then |v jj - r| else 0) + |r| * (9 / 8 * ∑ jj, |p jj| + 1 / 2 ^ 11) := by
  have hl := fun jj => lane_cen (v := v jj) (r := r) (hw0 jj) heps ha8 hab (hatt jj) (hmask jj)
  constructor
  · have e : ∑ jj, q jj * v jj - r * ∑ jj, p jj -
        (a8 + b8) / 2 * ∑ jj, (if al jj = true then w jj * (v jj - r) else 0) =
        ∑ jj, (q jj * v jj - r * p jj - (a8 + b8) / 2 * (if al jj = true then w jj * (v jj - r) else 0)) := by
      rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, Finset.mul_sum, Finset.mul_sum]
    rw [e]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ((Finset.sum_le_sum fun jj _ => (hl jj).1).trans (le_of_eq ?_))
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_div, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    ring
  · refine (Finset.sum_le_sum fun jj _ => (hl jj).2).trans (le_of_eq ?_)
    simp only [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    ring

/-- **One block's `Σ|q·v|`** for the loss arm: each attended lane's fp8 `p` is at most `b8·w + eps`
in magnitude, a masked lane's is `0`, so `Σ|q·v| ≤ b8·Σ w|v| + eps·Σ|v|` over the attended lanes. -/
theorem block_qv {q v w : Fin 64 → ℝ} {al : Fin 64 → Bool} {a8 b8 eps : ℝ} (heps : 0 ≤ eps) (ha8 : 0 ≤ a8)
    (hb8 : 0 ≤ b8) (hw0 : ∀ jj, 0 ≤ w jj)
    (hatt : ∀ jj, al jj = true → a8 * w jj - eps ≤ q jj ∧ q jj ≤ b8 * w jj + eps)
    (hmask : ∀ jj, al jj = false → q jj = 0) :
    ∑ jj, |q jj * v jj| ≤ b8 * ∑ jj, (if al jj = true then w jj * |v jj| else 0) +
      eps * ∑ jj, (if al jj = true then |v jj| else 0) := by
  have hl : ∀ jj, |q jj * v jj| ≤ b8 * (if al jj = true then w jj * |v jj| else 0) +
      eps * (if al jj = true then |v jj| else 0) := fun jj => by
    cases ha : al jj
    · rw [hmask jj ha]; simp
    · obtain ⟨h1, h2⟩ := hatt jj ha
      have k1 := mul_nonneg ha8 (hw0 jj)
      have k2 := mul_nonneg hb8 (hw0 jj)
      have hq : |q jj| ≤ b8 * w jj + eps := by rw [abs_le]; constructor <;> linarith
      simp only [if_true]
      rw [abs_mul]
      have := mul_le_mul_of_nonneg_right hq (abs_nonneg (v jj))
      linarith
  refine (Finset.sum_le_sum fun jj _ => hl jj).trans (le_of_eq ?_)
  rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]

/-- `fold max 0` of `c · f` for `c ≥ 0`. -/
theorem fold_max_mul {ι : Type} [DecidableEq ι] (s : Finset ι) (f : ι → ℝ) {c : ℝ} (hc : 0 ≤ c) :
    s.fold max 0 (fun k => c * f k) = c * s.fold max 0 f := by
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih => rw [Finset.fold_insert ha, Finset.fold_insert ha, ih, mul_max_of_nonneg _ _ hc]

theorem fold_max_nonneg {ι : Type} [DecidableEq ι] (s : Finset ι) (f : ι → ℝ) : 0 ≤ s.fold max 0 f :=
  (Finset.le_fold_max _).mpr (Or.inl le_rfl)

theorem le_fold_max' {ι : Type} [DecidableEq ι] {s : Finset ι} (f : ι → ℝ) {k : ι} (hk : k ∈ s) :
    f k ≤ s.fold max 0 f :=
  (Finset.le_fold_max _).mpr (Or.inr ⟨k, hk, le_rfl⟩)

/-- **The `p·v` `tl.dot`'s chunks' largest lifted products** against the scaled weights `w` and the
values' lifted magnitudes `vl`: when each lane's lifted product is at most `(b8·w + eps + 2^-14)·vl`
(attended) or `0` (masked), `cl2` is at most `b8` times the chunks' largest `w·vl` plus
`eps + 2^-14` times their largest `vl`. -/
theorem cl2_cen {qr vr : Fin 64 → ℚ} {w vl : Fin 64 → ℝ} {al : Fin 64 → Bool} {b8 eps : ℝ} (hb : 0 ≤ b8)
    (heps : 0 ≤ eps) (hw0 : ∀ jj, 0 ≤ w jj) (hvl : ∀ jj, 0 ≤ vl jj)
    (hl : ∀ jj, ((Rules.Num.liftMag .f8e5 (qr jj) * Rules.Num.liftMag .f8e5 (vr jj) : ℚ) : ℝ) ≤
      if al jj = true then (b8 * w jj + eps + 1 / 2 ^ 14) * vl jj else 0) :
    cl2 qr vr ≤
      b8 * ∑ c ∈ Finset.range 2, (Rules.Num.chunk8 64 c).fold max 0 (fun k => if al k = true then w k * vl k else 0) +
        (eps + 1 / 2 ^ 14) * ∑ c ∈ Finset.range 2, (Rules.Num.chunk8 64 c).fold max 0
          (fun k => if al k = true then vl k else 0) := by
  have hc : ∀ c : ℕ, ((Dot8Chunk.chunkLift8 .f8e5 .f8e5 qr vr c : ℚ) : ℝ) ≤
      b8 * (Rules.Num.chunk8 64 c).fold max 0 (fun k => if al k = true then w k * vl k else 0) +
        (eps + 1 / 2 ^ 14) * (Rules.Num.chunk8 64 c).fold max 0 (fun k => if al k = true then vl k else 0) := by
    intro c
    have h1 := fold_max_nonneg (Rules.Num.chunk8 64 c) (fun k => if al k = true then w k * vl k else 0)
    have h2 := fold_max_nonneg (Rules.Num.chunk8 64 c) (fun k => if al k = true then vl k else 0)
    unfold Dot8Chunk.chunkLift8
    refine fold_max_le_real _ _ (by positivity) fun k hk => ?_
    refine (hl k).trans ?_
    have k1 := le_fold_max' (fun k => if al k = true then w k * vl k else 0) hk
    have k2 := le_fold_max' (fun k => if al k = true then vl k else 0) hk
    split_ifs at k1 k2 ⊢ with ha
    · have e : (b8 * w k + eps + 1 / 2 ^ 14) * vl k = b8 * (w k * vl k) + (eps + 1 / 2 ^ 14) * vl k := by ring
      rw [e]
      exact add_le_add (mul_le_mul_of_nonneg_left k1 hb) (mul_le_mul_of_nonneg_left k2 (by positivity))
    · positivity
  unfold cl2
  push_cast
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  have := hc 0
  have := hc 1
  linarith

/-! ## The row's centred sums and recursions -/

/-- `Σ ew_j·(ve_j − r)` over the attended keys of the first `t` blocks. -/
noncomputable def Cx (R : RowCtx) (ve : ℕ → ℝ) (r : ℝ) (t : ℕ) : ℝ := psum R.al t (fun j => ew R j * (ve j - r))

/-- `Σ ew_j·|ve_j − r|` over the attended keys of block `t`. -/
noncomputable def bC (R : RowCtx) (ve : ℕ → ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  ∑ jj : Fin 64, if R.al (64 * t + jj) = true then ew R (64 * t + jj) * |ve (64 * t + jj) - r| else 0

/-- `Σ |ve_j − r|` over the attended keys of block `t`. -/
noncomputable def bA (R : RowCtx) (ve : ℕ → ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  ∑ jj : Fin 64, if R.al (64 * t + jj) = true then |ve (64 * t + jj) - r| else 0

/-- Over the two chunks of block `t`, each chunk's largest `ew_j·vl_j` over its attended keys. -/
noncomputable def bLW (R : RowCtx) (vl : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ c ∈ Finset.range 2, (Rules.Num.chunk8 64 c).fold max 0
    (fun k : Fin 64 => if R.al (64 * t + k) = true then ew R (64 * t + k) * vl (64 * t + k) else 0)

/-- Over the two chunks of block `t`, each chunk's largest `vl_j` over its attended keys. -/
noncomputable def bVL (R : RowCtx) (vl : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ c ∈ Finset.range 2, (Rules.Num.chunk8 64 c).fold max 0
    (fun k : Fin 64 => if R.al (64 * t + k) = true then vl (64 * t + k) else 0)

/-- `Σ ew_j·|ve_j|` over the attended keys of block `t`. -/
noncomputable def bV (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ jj : Fin 64, if R.al (64 * t + jj) = true then ew R (64 * t + jj) * |ve (64 * t + jj)| else 0

/-- `Σ |ve_j|` over the attended keys of block `t`. -/
noncomputable def bVa (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ jj : Fin 64, if R.al (64 * t + jj) = true then |ve (64 * t + jj)| else 0

/-- `Model.stC` in the row's terms (`stepC`'s choice at `W`). -/
noncomputable def stR (R : RowCtx) (ve vl : ℕ → ℝ) (r W : ℝ) : ℕ → ℝ × ℝ
  | 0 => (0, 0)
  | t + 1 => stepC W (pER R) (dmaxR R) |r| (stR R ve vl r W t) (Cx R ve r t) (FhR R t * psum R.al t (ew R))
      (bC R ve r t) (bV R ve t) (bLW R vl t) (bA R ve r t) (bVa R ve t) (bVL R vl t)

/-- `Model.errC` in the row's terms. -/
noncomputable def ecR (R : RowCtx) (ve vl : ℕ → ℝ) (r W : ℝ) (t : ℕ) : ℝ := (stR R ve vl r W t).1

/-- `Model.accA` in the row's terms. -/
noncomputable def acR (R : RowCtx) (ve vl : ℕ → ℝ) (r W : ℝ) (t : ℕ) : ℝ := (stR R ve vl r W t).2

theorem bV_nonneg (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : 0 ≤ bV R ve t :=
  Finset.sum_nonneg fun jj _ => by
    split_ifs
    · exact mul_nonneg (ew_pos _ _).le (abs_nonneg _)
    · exact le_rfl

theorem bVa_nonneg (R : RowCtx) (ve : ℕ → ℝ) (t : ℕ) : 0 ≤ bVa R ve t :=
  Finset.sum_nonneg fun jj _ => by
    split_ifs
    · exact abs_nonneg _
    · exact le_rfl

theorem bC_nonneg (R : RowCtx) (ve : ℕ → ℝ) (r : ℝ) (t : ℕ) : 0 ≤ bC R ve r t :=
  Finset.sum_nonneg fun jj _ => by
    split_ifs
    · exact mul_nonneg (ew_pos _ _).le (abs_nonneg _)
    · exact le_rfl

theorem bA_nonneg (R : RowCtx) (ve : ℕ → ℝ) (r : ℝ) (t : ℕ) : 0 ≤ bA R ve r t :=
  Finset.sum_nonneg fun jj _ => by
    split_ifs
    · exact abs_nonneg _
    · exact le_rfl

theorem bLW_nonneg (R : RowCtx) (vl : ℕ → ℝ) (t : ℕ) : 0 ≤ bLW R vl t :=
  Finset.sum_nonneg fun _ _ => fold_max_nonneg _ _

theorem bVL_nonneg (R : RowCtx) (vl : ℕ → ℝ) (t : ℕ) : 0 ≤ bVL R vl t :=
  Finset.sum_nonneg fun _ _ => fold_max_nonneg _ _

/-- `stepC` keeps both components nonnegative. -/
theorem stepC_nonneg {W pe dm R : ℝ} {st : ℝ × ℝ} {C Lw Bc BV Bl Ba BVa Bv : ℝ} (hpe : 1 ≤ pe) (h0 : 0 ≤ dm)
    (h1 : dm ≤ 1 / 2 ^ 13) (hR : 0 ≤ R) (hs1 : 0 ≤ st.1) (hs2 : 0 ≤ st.2) (hLw : 0 ≤ Lw) (hBc : 0 ≤ Bc)
    (hBV : 0 ≤ BV) (hBl : 0 ≤ Bl) (hBa : 0 ≤ Ba) (hBVa : 0 ≤ BVa) (hBv : 0 ≤ Bv) :
    0 ≤ (stepC W pe dm R st C Lw Bc BV Bl Ba BVa Bv).1 ∧ 0 ≤ (stepC W pe dm R st C Lw Bc BV Bl Ba BVa Bv).2 := by
  obtain ⟨hcs0, -, hcc0, hcp0, -, -, hls0, -, hlc0, hlp0, -, -⟩ := arm_consts
  unfold stepC
  split_ifs
  · exact ⟨errStep_nonneg hpe h0 hls0 hlc0 hlp0 hs1 hR hs2 hLw hBc hBV hBl,
      accStep_nonneg h0 h1 hlc0 hlp0 le_rfl hR hs2 hBa hBVa hBv⟩
  · exact ⟨errStep_nonneg hpe h0 hcs0 hcc0 hcp0 hs1 hR hs2 hLw hBc hBc hBl,
      accStep_nonneg h0 h1 hcc0 hcp0 hcc0 hR hs2 hBa hBa hBv⟩

theorem stR_nonneg {R : RowCtx} (hR : R.Ok) (ve vl : ℕ → ℝ) (r W : ℝ) :
    ∀ t, 0 ≤ (stR R ve vl r W t).1 ∧ 0 ≤ (stR R ve vl r W t).2
  | 0 => ⟨le_rfl, le_rfl⟩
  | t + 1 => by
    have ih := stR_nonneg hR ve vl r W t
    exact stepC_nonneg (pER_ge hR) (dmaxR_nonneg hR) (dmaxR_le hR) (abs_nonneg r) ih.1 ih.2
      (mul_nonneg ((zero_le_one).trans (FhR_ge hR t)) (psum_nonneg fun j => (ew_pos R j).le))
      (bC_nonneg R ve r t) (bV_nonneg R ve t) (bLW_nonneg R vl t) (bA_nonneg R ve r t) (bVa_nonneg R ve t)
      (bVL_nonneg R vl t)

theorem acR_nonneg {R : RowCtx} (hR : R.Ok) (ve vl : ℕ → ℝ) (r W : ℝ) (t : ℕ) : 0 ≤ acR R ve vl r W t :=
  (stR_nonneg hR ve vl r W t).2

theorem ecR_nonneg {R : RowCtx} (hR : R.Ok) (ve vl : ℕ → ℝ) (r W : ℝ) (t : ℕ) : 0 ≤ ecR R ve vl r W t :=
  (stR_nonneg hR ve vl r W t).1

/-- **The centred bound** on an element `a` of `acc` after `t` blocks with running maximum `m` and
`l_i = l`: for every `r` and every choice `W` of `stepC`, `a − r·l` is within `2^-m·ecR + acR·l` of
`kap·2^-m·Cx`. -/
def CInv (R : RowCtx) (ve vl : ℕ → ℝ) (t : ℕ) (m a l : F32) : Prop :=
  ∀ r W : ℝ, |a.toReal - r * l.toReal - kap (pER R) * ((2 : ℝ) ^ (-m.toReal) * Cx R ve r t)| ≤
    (2 : ℝ) ^ (-m.toReal) * ecR R ve vl r W t + acR R ve vl r W t * l.toReal

end Kernels.AttnFwdE5m2
