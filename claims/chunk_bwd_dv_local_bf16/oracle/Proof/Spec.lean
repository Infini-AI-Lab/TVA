import Claim
import TVA.Proof
import Proof.Setup
import Proof.NumEntry

/-!
# The specifications of the pieces of the tail

What the tail's pieces leave, in terms of the program's lanes `a`, `b` (tokens `pos a`, `pos b`):

* `GOk`: the gate block holds `g[pos l, hv]` at every enabled lane;
* `KFinal`: the `K` loop's accumulator is within `σ·(θ₀·kqMag + 2^-110)` of `σ·kq` at every
  enabled pair `a ≤ b`;
* `AOk`: the rounded matrix is, at every enabled pair `a ≤ b`, an `EntryOk` approximation of the
  exact entry `Aex`, and exactly `+0` elsewhere.

`sum_lanes` turns the claim's sum over the tokens `[p, chunkEnd)` into a sum over lanes.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- Entry `(p, j)` of `A`, as a `bfloat16` `x`: finite, within `2^-8·|Aex| + κ·aMag` of the exact
entry, of lifted magnitude at most `(1 + 2^-7)·aMag`. -/
def EntryOk (d : Inst) (m : Mem) (p j hv : Nat) (x : BF16) : Prop :=
  x.isFinite = true ∧
    |x.toReal - Aex d m p j hv| ≤ 1 / 2 ^ 8 * |Aex d m p j hv| + kap * aMag d m p j hv ∧
    ((Device.liftMag .bf16 x.toRat : ℚ) : ℝ) ≤ (1 + 1 / 2 ^ 7) * aMag d m p j hv

/-- The gate block: with a gate, each enabled lane holds `g[pos l, hv]`, a finite `float32`. -/
def GOk (d : Inst) (init : Mem) (r bh : Nat) (b_g : Tensor [d.BT] F32) : Prop :=
  d.useG = true → ∀ l : Fin d.BT, en d r bh l = true →
    (b_g.get1 l).isFinite = true ∧
      (b_g.get1 l).toRat = valF init d.g (gIdx d (pos d r bh l) (hvOf d bh))

/-- The `K` loop's accumulator at every enabled pair `a ≤ b`. -/
def KFinal (d : Inst) (init : Mem) (r bh : Nat) (b_A : Tensor [d.BT, d.BT] F32) : Prop :=
  ∀ a b : Fin d.BT, en d r bh a = true → en d r bh b = true → a ≤ b →
    (b_A.get2 a b).isFinite = true ∧
      |(b_A.get2 a b).toReal - sig d * kq d init (pos d r bh a) (pos d r bh b) (hvOf d bh)| ≤
        sig d * (theta0 * kqMag d init (pos d r bh a) (pos d r bh b) (hvOf d bh) + 1 / 2 ^ 110)

/-- The rounded matrix: `EntryOk` at every enabled pair `a ≤ b`, `+0` at every other lane. -/
def AOk (d : Inst) (init : Mem) (r bh : Nat) (A : Tensor [d.BT, d.BT] BF16) : Prop :=
  ∀ a b : Fin d.BT,
    (en d r bh a = true ∧ en d r bh b = true ∧ a ≤ b →
      EntryOk d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (A.get2 a b)) ∧
    (¬ (en d r bh a = true ∧ en d r bh b = true ∧ a ≤ b) → A.get2 a b = FloatV.ofPat 0)

/-! ## The claim's sums are sums over lanes -/

/-- The start of the chunk of token `pos l` is the program's first token `pos 0`. -/
theorem chunkStart_pos {d : Inst} (hd : d.ok) (r bh : Nat) {l : Nat} (hl : l < d.BT) :
    chunkStart d (nOf d r bh) (pos d r bh l) = baseOf d r bh + chkOf d r bh * d.BT := by
  have hBT := bt_pos hd
  unfold chunkStart pos baseOf
  have h1 : lo d (nOf d r bh) + chkOf d r bh * d.BT + l - lo d (nOf d r bh) =
      chkOf d r bh * d.BT + l := by omega
  rw [h1]
  have h2 : (chkOf d r bh * d.BT + l) / d.BT = chkOf d r bh := by
    rw [Nat.mul_comm, Nat.mul_add_div hBT, Nat.div_eq_of_lt hl, Nat.add_zero]
  rw [h2]

/-- **The tokens `[p, chunkEnd)` of the claim's sums are the lanes `l' ≥ l` that are enabled.** -/
theorem sum_lanes {d : Inst} (hd : d.ok) (r bh : Nat) (l : Fin d.BT) (F : Nat → ℝ) :
    ∑ j ∈ Finset.Ico (pos d r bh l) (chunkEnd d (nOf d r bh) (pos d r bh l)), F j =
      ∑ l' : Fin d.BT, if (l ≤ l' ∧ en d r bh l' = true) then F (pos d r bh l') else 0 := by
  rw [← Finset.sum_filter]
  have hcs := chunkStart_pos hd r bh l.isLt
  have hend : chunkEnd d (nOf d r bh) (pos d r bh l) =
      min (baseOf d r bh + chkOf d r bh * d.BT + d.BT) (hi d (nOf d r bh)) := by
    unfold chunkEnd; rw [hcs]
  have hinj : ∀ l₁ l₂ : Fin d.BT, pos d r bh l₁ = pos d r bh l₂ → l₁ = l₂ := by
    intro l₁ l₂ h
    unfold pos at h
    exact Fin.ext (by omega)
  have himg : Finset.Ico (pos d r bh l) (chunkEnd d (nOf d r bh) (pos d r bh l)) =
      (Finset.univ.filter (fun l' : Fin d.BT => l ≤ l' ∧ en d r bh l' = true)).image
        (fun l' : Fin d.BT => pos d r bh l') := by
    ext j
    rw [Finset.mem_image, Finset.mem_Ico, hend]
    constructor
    · rintro ⟨h1, h2⟩
      have hBT := l.isLt
      obtain ⟨hlt1, hlt2⟩ := lt_min_iff.mp h2
      have hj1 : j - (baseOf d r bh + chkOf d r bh * d.BT) < d.BT := by
        unfold pos at h1; omega
      refine ⟨⟨j - (baseOf d r bh + chkOf d r bh * d.BT), hj1⟩, ?_, ?_⟩
      · rw [Finset.mem_filter]
        refine ⟨Finset.mem_univ _, ?_, ?_⟩
        · show l.val ≤ j - (baseOf d r bh + chkOf d r bh * d.BT)
          unfold pos at h1; omega
        · rw [en_iff]
          show chkOf d r bh * d.BT + (j - (baseOf d r bh + chkOf d r bh * d.BT)) <
            lenOf d r bh
          unfold lenOf baseOf at *
          unfold pos baseOf at h1
          omega
      · show pos d r bh (j - (baseOf d r bh + chkOf d r bh * d.BT)) = j
        unfold pos at *
        omega
    · rintro ⟨l', hl', rfl⟩
      rw [Finset.mem_filter] at hl'
      obtain ⟨-, h1, h2⟩ := hl'
      rw [en_iff] at h2
      have hBT := l'.isLt
      have h1' : l.val ≤ l'.val := h1
      unfold lenOf at h2
      unfold pos baseOf at *
      refine ⟨by omega, lt_min (by omega) (by omega)⟩
  rw [himg, Finset.sum_image]
  intro a _ b _ h
  exact hinj a b h

end Claims.ChunkBwdDvLocal
