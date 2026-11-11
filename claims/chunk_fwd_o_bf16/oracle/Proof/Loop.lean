import Claim
import TVA.Proof
import Proof.DotStep
import Proof.Sums
import Proof.Load

/-!
# The loop over `K`: invariant and the two dots of a trip

After `n` trips of the loop (lines 135–158), `b_o[i, c]` is within `DotLoop.bound` of the partial
inter-chunk dot `SQ` and `b_A[i, i']` within `DotLoop.bound` of the partial score `SA`, for the
lanes the claim's `InRange` makes finite (`OInv`, `AInv`). `dots_safe` is the two `tl.dot`s of a
trip from the lane values of the three loaded tiles (`qTile`, `kTile`, `hTile`).
-/

open TVA
namespace Claims.ChunkFwdO

/-- After `n` trips `b_o` follows the inter-chunk dot, on the lanes whose inputs are finite. -/
def OInv (d : Inst) (g : Geo) (r n : Nat) (bo : Tensor [BT, d.bk] F32) : Prop :=
  ∀ (i : Fin BT) (c : Fin d.bk), g.rowOk i → g.colOk d c →
    OkO d r (g.lo + i) g.ih (g.iv * d.bk + c) →
    (bo.get2 i c).isFinite = true ∧
      |(bo.get2 i c).toRat - SQ d r (g.lo + i) g.ih (g.iv * d.bk + c) (n * d.bk)| ≤
        DotLoop.bound (rhoK d) (alphaK d) n (MQ d r (g.lo + i) g.ih (g.iv * d.bk + c) (n * d.bk))

/-- After `n` trips `b_A` follows the scores `q[t]·k[s]`, `s ≤ t`, on the lanes whose inputs are
finite. -/
def AInv (d : Inst) (g : Geo) (n : Nat) (bA : Tensor [BT, BT] F32) : Prop :=
  ∀ (i i' : Fin BT), g.rowOk i → i' ≤ i → OkA d (g.lo + i) (g.lo + i') g.ih →
    (bA.get2 i i').isFinite = true ∧
      |(bA.get2 i i').toRat - SA d (g.lo + i) (g.lo + i') g.ih (n * d.bk)| ≤
        DotLoop.bound (rhoK d) (alphaK d) n (MA d (g.lo + i) (g.lo + i') g.ih (n * d.bk))

/-- Every entry of a `float32` zeros tile is the `+0` pattern. -/
theorem zeros_get2 {M N : Nat} (i : Fin M) (j : Fin N) :
    (tl.zeros [M, N] .f32).get2 i j = (FloatV.ofPat 0 : F32) := rfl

/-- The invariant holds before the first trip (the accumulators are `+0`). -/
theorem inv_zero (d : Inst) (g : Geo) (r : Nat) :
    OInv d g r 0 (tl.zeros [BT, d.bk] .f32) ∧ AInv d g 0 (tl.zeros [BT, BT] .f32) := by
  constructor
  · intro i c _ _ _
    rw [zeros_get2, f32_zero_toRat]
    simp [DotLoop.bound, SQ]
  · intro i i' _ _ _
    rw [zeros_get2, f32_zero_toRat]
    simp [DotLoop.bound, SA]

/-! ## The loaded tiles -/

/-- Lane `(i, j)` of the `b_q` tile of trip `t`: `q` of token row `i` at `x = t·BK + j`, or `+0`
(row past the sequence, or `x ≥ K`). -/
def qTile (d : Inst) (g : Geo) (t i j : Nat) : BF16 :=
  if g.rowOk i ∧ t * d.bk + j < d.K then
    d.qv (((g.lo + i) * d.H + qh d g.ih) * d.K + (t * d.bk + j))
  else FloatV.ofPat 0

/-- Lane `(j, i')` of the `b_k` tile of trip `t`: `k` of token row `i'` at `x = t·BK + j`, or `+0`. -/
def kTile (d : Inst) (g : Geo) (t j i' : Nat) : BF16 :=
  if g.rowOk i' ∧ t * d.bk + j < d.K then
    d.kv (((g.lo + i') * d.H + qh d g.ih) * d.K + (t * d.bk + j))
  else FloatV.ofPat 0

/-- Lane `(k, c)` of the state tile of trip `t` (in the `[BK, BV]` orientation of the dot): the
state element `S_r[ih, t·BK + k, iv·BV + c]`, or `+0`. -/
def hTile (d : Inst) (g : Geo) (r t k c : Nat) : BF16 :=
  if t * d.bk + k < d.K ∧ g.colOk d c then d.hv (hIdx d r g.ih (t * d.bk + k) (g.iv * d.bk + c))
  else FloatV.ofPat 0

/-- The `bfloat16` `+0` pattern has value `some 0`. -/
theorem val?_zero_bf16 : DType.val? .bf16 (FloatV.ofPat 0 : BF16) = some 0 :=
  val?_bf16.mpr ⟨bf16_zero_isFinite, bf16_zero_toRat⟩

/-- For a valid row, a `q` tile entry has the value of `qPad` at the matching position. -/
theorem val?_qTile (d : Inst) (g : Geo) (t i k : Nat) (hrow : g.rowOk i)
    (hq : ∀ x, x < d.K → (d.qv (((g.lo + i) * d.H + qh d g.ih) * d.K + x)).isFinite = true) :
    DType.val? .bf16 (qTile d g t i k) = some (qPad d (g.lo + i) g.ih (t * d.bk + k)) := by
  unfold qTile qPad
  by_cases hx : t * d.bk + k < d.K
  · simp only [hrow, hx, and_self, ite_true]
    exact val?_bf16.mpr ⟨hq _ hx, rfl⟩
  · simp only [hx, and_false, ite_false]
    exact val?_zero_bf16

/-- For a valid row, a `k` tile entry has the value of `kPad` at the matching position. -/
theorem val?_kTile (d : Inst) (g : Geo) (t j i' : Nat) (hrow : g.rowOk i')
    (hk : ∀ x, x < d.K → (d.kv (((g.lo + i') * d.H + qh d g.ih) * d.K + x)).isFinite = true) :
    DType.val? .bf16 (kTile d g t j i') = some (kPad d (g.lo + i') g.ih (t * d.bk + j)) := by
  unfold kTile kPad
  by_cases hx : t * d.bk + j < d.K
  · simp only [hrow, hx, and_self, ite_true]
    exact val?_bf16.mpr ⟨hk _ hx, rfl⟩
  · simp only [hx, and_false, ite_false]
    exact val?_zero_bf16

/-- For a valid column, an `h` tile entry has the value of `hPad` at the matching position. -/
theorem val?_hTile (d : Inst) (g : Geo) (r t k c : Nat) (hcol : g.colOk d c)
    (hh : ∀ x, x < d.K → (d.hv (hIdx d r g.ih x (g.iv * d.bk + c))).isFinite = true) :
    DType.val? .bf16 (hTile d g r t k c) = some (hPad d r g.ih (g.iv * d.bk + c) (t * d.bk + k)) := by
  unfold hTile hPad
  by_cases hx : t * d.bk + k < d.K
  · simp only [hcol, hx, and_self, ite_true]
    exact val?_bf16.mpr ⟨hh _ hx, rfl⟩
  · simp only [hx, false_and, ite_false]
    exact val?_zero_bf16

/-! ## Facts about the configuration -/

theorem bk_le {d : Inst} (hd : d.ok) : d.bk ≤ 128 := by
  have := hd.bk_ok; omega

/-- For a valid instance, `bk` is positive. -/
theorem bk_pos {d : Inst} (hd : d.ok) : 0 < d.bk := by
  have := hd.bk_ok; omega

/-- For a valid instance, `K ≤ nK d * bk`. -/
theorem K_le_nK {d : Inst} (hd : d.ok) : d.K ≤ nK d * d.bk := by
  have hb := bk_pos hd
  unfold nK
  have h1 := Nat.div_add_mod (d.K + (d.bk - 1)) d.bk
  have h2 := Nat.mod_lt (d.K + (d.bk - 1)) hb
  rw [Nat.mul_comm]
  generalize d.bk * ((d.K + (d.bk - 1)) / d.bk) = X at h1 ⊢
  omega

/-- For a valid instance, the trip count `nK d` is at most `256`. -/
theorem nK_le {d : Inst} (hd : d.ok) : nK d ≤ 256 := by
  have hb := bk_pos hd
  unfold nK
  have h1 : (d.K + (d.bk - 1)) / d.bk ≤ d.K + (d.bk - 1) := Nat.div_le_self _ _
  have h2 := hd.K_le
  have : (d.K + (d.bk - 1)) / d.bk ≤ (256 + (d.bk - 1)) / d.bk :=
    Nat.div_le_div_right (by omega)
  have h3 : (256 + (d.bk - 1)) / d.bk ≤ 256 := by
    rcases hd.bk_ok with h | h | h <;> rw [h] <;> norm_num
  omega

/-- For a trip index `t < nK d`, `t * relDot16 bk ≤ 1/2`. -/
theorem trip_rel {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nK d) :
    (t : ℚ) * Device.relDot16 d.bk ≤ 1 / 2 := by
  have h1 : (t : ℚ) ≤ 256 := by exact_mod_cast (le_of_lt (lt_of_lt_of_le ht (nK_le hd)))
  have h2 : Device.relDot16 d.bk ≤ 130 * 2 ^ (-22 : ℤ) := by
    unfold Device.relDot16
    have : (d.bk : ℚ) ≤ 128 := by exact_mod_cast bk_le hd
    exact mul_le_mul_of_nonneg_right (by linarith) (by positivity)
  have h3 : (0 : ℚ) ≤ Device.relDot16 d.bk := by unfold Device.relDot16; positivity
  calc (t : ℚ) * Device.relDot16 d.bk ≤ 256 * (130 * 2 ^ (-22 : ℤ)) :=
        mul_le_mul h1 h2 h3 (by norm_num)
    _ ≤ 1 / 2 := by norm_num

end Claims.ChunkFwdO
