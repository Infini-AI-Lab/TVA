import Claim
import TVA.Proof
import Proof.Steps
import Proof.Prog

/-!
# The bound on one stored `u`

`u_lane_bound`: with the lanes `b_A` and `b_vb` the dot sees, the `bfloat16` rounding of its
element meets `CellU`'s bound (`uRef`, `uMag`, `tolU`).
-/

namespace Claims.RecomputeWU

open TVA

/-- Row `i < 64` of chunk `it` is in chunk `it`. -/
theorem div64 (it i : Nat) (hi : i < 64) : (it * 64 + i) / 64 = it := by omega

/-- Column `k` of the row, as a function on `Nat`. -/
def colOf (b : Tensor [64, 64] BF16) (n : Fin 64) (k : Nat) : BF16 :=
  if hk : k < 64 then b.get2 ⟨k, hk⟩ n else FloatV.ofPat 0

/-- **The stored `u`**: with the lanes `b_A` and `b_vb` the dot sees, the `bfloat16` rounding of its element is finite and within `tolU` of `uRef`. -/
theorem u_lane_bound {P : Platform} (hP : Device P) {d : Inst} {init : Mem} (hpre : Pre d init)
    {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV)
    {b_A b_vb : Tensor [64, 64] BF16} {r : Tensor [64, 64] F32}
    (hr : P.dot cfgU b_A b_vb (tl.zeros [64, 64] .f32) r)
    (hA : ALane d init s0 len it h b_A) (hVb : VbLane d init s0 len it h t b_vb)
    (i n : Fin 64) (hi : rowEn it len i = true) (hn : colEn t d.V n = true) {o : BF16}
    (ho : (FloatV.cvt (r.get2 i n)).NaNEq o) :
    o.isFinite = true ∧
      |o.toRat - uRef d init s0 len (it * 64 + i) h (t * 64 + n)| ≤
        tolU (uRef d init s0 len (it * 64 + i) h (t * 64 + n))
          (uMag d init s0 len (it * 64 + i) h (t * 64 + n)) := by
  have hrow : it * 64 + i < len := by simpa [rowEn] using hi
  have htok : tokOf s0 it i < nTok d := by unfold tokOf; omega
  have key := u_cell hP .ieee hr i n (fun k => bfV d.A init (aIdx d (tokOf s0 it i) h k))
    (colOf b_vb n) (rowEn it len) (fun k => ((hA i k).1 hi).symm ▸ rfl)
    (fun k => by simp only [colOf, k.isLt, ↓reduceDIte])
    (fun k hk => (A_ok hpre htok hh hk).1) (fun k hk => (A_ok hpre htok hh hk).2)
    (fun k hk => by simp only [colOf, hk, ↓reduceDIte]; exact (hVb ⟨k, hk⟩ n hn).1)
    (fun k hk => by simp only [colOf, hk, ↓reduceDIte]; exact (hVb ⟨k, hk⟩ n hn).2.1)
    (fun k hk he => by simp only [colOf, hk, ↓reduceDIte]; exact (hVb ⟨k, hk⟩ n hn).2.2.2 he) rfl
  have hS : uRef d init s0 len (it * 64 + i) h (t * 64 + n) =
      ∑ k ∈ Finset.range 64, (if rowEn it len k = true then
        (bfV d.A init (aIdx d (tokOf s0 it i) h k)).toRat * (colOf b_vb n k).toRat else 0) := by
    unfold uRef
    refine Finset.sum_congr rfl fun k hk => ?_
    have hk' := Finset.mem_range.mp hk
    have e1 : inSeq len (it * 64 + i) k = rowEn it len k := by
      unfold inSeq rowEn; rw [div64 it i i.isLt]
    rw [e1]
    by_cases he : rowEn it len k = true
    · simp only [he, ite_true]
      have e2 : aV d init (s0 + (it * 64 + i)) h k = bfV d.A init (aIdx d (tokOf s0 it i) h k) := by
        unfold aV tokOf; congr 2; omega
      have e3 : colTok s0 (it * 64 + i) k = tokOf s0 it k := by
        unfold colTok tokOf; rw [div64 it i i.isLt]
      rw [e2, e3]
      simp only [colOf, hk', ↓reduceDIte]
      rw [(hVb ⟨k, hk'⟩ n hn).2.2.1 he]
    · simp [he]
  have hM : uMag d init s0 len (it * 64 + i) h (t * 64 + n) =
      ∑ k ∈ Finset.range 64, (if rowEn it len k = true then
        Device.liftMag .bf16 (bfV d.A init (aIdx d (tokOf s0 it i) h k)).toRat *
          Device.liftMag .bf16 (colOf b_vb n k).toRat else 0) := by
    unfold uMag
    refine Finset.sum_congr rfl fun k hk => ?_
    have hk' := Finset.mem_range.mp hk
    have e1 : inSeq len (it * 64 + i) k = rowEn it len k := by
      unfold inSeq rowEn; rw [div64 it i i.isLt]
    rw [e1]
    by_cases he : rowEn it len k = true
    · simp only [he, ite_true]
      have e2 : aV d init (s0 + (it * 64 + i)) h k = bfV d.A init (aIdx d (tokOf s0 it i) h k) := by
        unfold aV tokOf; congr 2; omega
      have e3 : colTok s0 (it * 64 + i) k = tokOf s0 it k := by
        unfold colTok tokOf; rw [div64 it i i.isLt]
      rw [e2, e3]
      simp only [colOf, hk', ↓reduceDIte]
      rw [(hVb ⟨k, hk'⟩ n hn).2.2.1 he]
    · simp [he]
  rw [hS, hM] at *
  rw [ho.eq_of_isFinite key.1]
  exact key


/-- `|k · beta| ≤ 2^40` for covered inputs. -/
theorem kbX_le {d : Inst} {init : Mem} (hpre : Pre d init) {f h c : Nat} (hf : f < nTok d)
    (hh : h < d.HV) (hk : kHead d h < d.H) (hc : c < d.K) : |kbX d init f h c| ≤ 2 ^ 40 := by
  have hk0 := k_ok hpre hf hk hc
  have hkb : |(bfV d.k init (kIdx d f (kHead d h) c)).toReal| ≤ 2 ^ 32 := by
    unfold FloatV.toReal; exact_mod_cast hk0.2
  unfold kbX betaR
  by_cases hbb : d.bbf = true
  · have hb := betab_ok hpre hbb hf hh
    have hbb' : |(bfV d.betab init (hIdx d f h)).toReal| ≤ 2 ^ 8 := by
      unfold FloatV.toReal; exact_mod_cast hb.2
    simp only [hbb, ite_true]
    exact abs_kb_le hkb hbb'
  · have hbb0 : d.bbf = false := by simpa using hbb
    have hb := beta_ok hpre hbb0 hf hh
    have hbb' : |(f32V d.beta init (hIdx d f h)).toReal| ≤ 2 ^ 8 := by
      unfold FloatV.toReal; exact_mod_cast hb.2
    simp only [hbb, Bool.false_eq_true, ite_false]
    exact abs_kb_le hkb hbb'

/-- `2^g` is positive and at most `2^8` for a covered `g`. -/
theorem gE_bounds {d : Inst} {init : Mem} (hpre : Pre d init) {f h : Nat} (hf : f < nTok d)
    (hh : h < d.HV) : 0 < gE d init f h ∧ gE d init f h ≤ 2 ^ 8 := by
  have hg := g_ok hpre hf hh
  have h8 : (f32V d.g init (hIdx d f h)).toReal ≤ 8 := by
    unfold FloatV.toReal; exact_mod_cast hg.2
  exact ⟨two_rpow_pos _, two_rpow_le h8⟩

/-- **The stored `w`**: the `bfloat16` rounding of the dot's element is finite and within `tolW` of `wRef`. -/
theorem w_lane_bound {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d) (hh : h < d.HV)
    {b_A b_kbh : Tensor [64, 64] BF16} {r : Tensor [64, 64] F32}
    (hr : P.dot cfgW b_A b_kbh (tl.zeros [64, 64] .f32) r)
    (hA : ALane d init s0 len it h b_A) (hKb : KbLane d init s0 len it h t b_kbh)
    (i n : Fin 64) (hi : rowEn it len i = true) (hn : colEn t d.K n = true) {o : BF16}
    (ho : (FloatV.cvt (r.get2 i n)).NaNEq o) :
    o.isFinite = true ∧
      |o.toReal - wRef d init s0 len (it * 64 + i) h (t * 64 + n)| ≤
        tolW (wMag d init s0 len (it * 64 + i) h (t * 64 + n))
          (wFloor d init s0 len (it * 64 + i) h (t * 64 + n)) := by
  have hrow : it * 64 + i < len := by simpa [rowEn] using hi
  have htok : tokOf s0 it i < nTok d := by unfold tokOf; omega
  have hkc : t * 64 + n < d.K := by simpa [colEn] using hn
  have htk : ∀ k : Nat, rowEn it len k = true → k < 64 → tokOf s0 it k < nTok d := by
    intro k hk hk64
    have : it * 64 + k < len := by simpa [rowEn] using hk
    unfold tokOf; omega
  have key := w_cell hP .tf32 hr i n (fun k => bfV d.A init (aIdx d (tokOf s0 it i) h k))
    (colOf b_kbh n) (rowEn it len) (fun k => kbX d init (tokOf s0 it k) h (t * 64 + n))
    (fun k => gE d init (tokOf s0 it k) h) (fun k => ((hA i k).1 hi).symm ▸ rfl)
    (fun k => by simp only [colOf, k.isLt, ↓reduceDIte])
    (fun k hk => (A_ok hpre htok hh hk).1) (fun k hk => (A_ok hpre htok hh hk).2)
    (fun k hk => by simp only [colOf, hk, ↓reduceDIte]; exact (hKb ⟨k, hk⟩ n hn).1)
    (fun k hk he => by
      simp only [colOf, hk, ↓reduceDIte]; exact (hKb ⟨k, hk⟩ n hn).2.1 he)
    (fun k hk he => kbX_le hpre (htk k he hk) hh (kHead_lt hd hh) hkc)
    (fun k hk he => gE_bounds hpre (htk k he hk) hh)
    (fun k hk he => by simp only [colOf, hk, ↓reduceDIte]; exact (hKb ⟨k, hk⟩ n hn).2.2 he) rfl
  have e1 : ∀ k, inSeq len (it * 64 + i) k = rowEn it len k := fun k => by
    unfold inSeq rowEn; rw [div64 it i i.isLt]
  have e2 : ∀ k, aV d init (s0 + (it * 64 + i)) h k = bfV d.A init (aIdx d (tokOf s0 it i) h k) :=
    fun k => by unfold aV tokOf; congr 2; omega
  have e3 : ∀ k, colTok s0 (it * 64 + i) k = tokOf s0 it k := fun k => by
    unfold colTok tokOf; rw [div64 it i i.isLt]
  have hS : wRef d init s0 len (it * 64 + i) h (t * 64 + n) = ∑ k ∈ Finset.range 64,
      (if rowEn it len k = true then (bfV d.A init (aIdx d (tokOf s0 it i) h k)).toReal *
        kbX d init (tokOf s0 it k) h (t * 64 + n) * gE d init (tokOf s0 it k) h else 0) := by
    unfold wRef
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [e1, e2, e3]
  have hM : wMag d init s0 len (it * 64 + i) h (t * 64 + n) = ∑ k ∈ Finset.range 64,
      (if rowEn it len k = true then
        (Device.liftMag .bf16 (bfV d.A init (aIdx d (tokOf s0 it i) h k)).toRat : ℝ) *
          |kbX d init (tokOf s0 it k) h (t * 64 + n)| * gE d init (tokOf s0 it k) h else 0) := by
    unfold wMag
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [e1, e2, e3]
  have hF : wFloor d init s0 len (it * 64 + i) h (t * 64 + n) = ∑ k ∈ Finset.range 64,
      (if rowEn it len k = true then
        (Device.liftMag .bf16 (bfV d.A init (aIdx d (tokOf s0 it i) h k)).toRat : ℝ) *
          (|kbX d init (tokOf s0 it k) h (t * 64 + n)| + 1) else 0) := by
    unfold wFloor
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [e1, e2, e3]
  rw [hS, hM, hF] at *
  rw [ho.eq_of_isFinite key.1]
  exact key

end Claims.RecomputeWU
