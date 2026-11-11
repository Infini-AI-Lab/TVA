import Proof.Seg
import Proof.Inv

/-!
# `chunk_delta_h_bf16`: the four blocks as a family

The four state blocks of a trip, with the optional ones resolved by `K`, as families indexed by the
block number `b < nbOf d`.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The four blocks, by number (`b ≥ 3` is the fourth). -/
def fam4 {α : Type} (x1 x2 x3 x4 : α) : ℕ → α
  | 0 => x1
  | 1 => x2
  | 2 => x3
  | _ => x4

/-- The five stages of a chain (the accumulator before block `0`, after block `0`, …). -/
def fam5 {α : Type} (x0 x1 x2 x3 x4 : α) : ℕ → α
  | 0 => x0
  | 1 => x1
  | 2 => x2
  | 3 => x3
  | _ => x4

/-- The number of blocks that exist, by `K`. -/
theorem nb_cases {d : Inst} (hd : d.ok) :
    (nbOf d = 1 ∧ d.K ≤ 64) ∨ (nbOf d = 2 ∧ 64 < d.K ∧ d.K ≤ 128) ∨
      (nbOf d = 3 ∧ 128 < d.K ∧ d.K ≤ 192) ∨ (nbOf d = 4 ∧ 192 < d.K) := by
  have := hd.K_pos; have := hd.K_le
  unfold nbOf; omega

/-- The blocks that exist cover the rows of the state. -/
theorem nbOf_bounds {d : Inst} : d.K ≤ 64 * nbOf d ∧ 64 * nbOf d < d.K + 64 := by
  unfold nbOf; omega

/-- **The `w` dots as a chain over the blocks that exist**: tiles `Bw b`, roundings `Xs b` of the
state blocks, and results `Rr b` (`Rr 0` the zeros, `Rr nb` the last result). -/
theorem chain_w {d : Inst} (hd : d.ok) {P : Platform} {init : Mem} {n hv t : Nat}
    {b1 b2 b3 b4 : Tensor [64, BV] F32} {r1 r2 r3 r4 : Tensor [BT, BV] F32}
    (hw1 : WFacts d P init n hv t 0 b1 (tl.zeros [BT, BV] .f32) r1)
    (hw2 : (64 < d.K ∧ WFacts d P init n hv t 64 b2 r1 r2) ∨ (¬ 64 < d.K ∧ r2 = r1))
    (hw3 : (128 < d.K ∧ WFacts d P init n hv t 128 b3 r2 r3) ∨ (¬ 128 < d.K ∧ r3 = r2))
    (hw4 : (192 < d.K ∧ WFacts d P init n hv t 192 b4 r3 r4) ∨ (¬ 192 < d.K ∧ r4 = r3)) :
    ∃ (Bw : ℕ → Tensor [BT, 64] BF16) (Xs : ℕ → Tensor [64, BV] BF16) (Rr : ℕ → Tensor [BT, BV] F32),
      Rr 0 = tl.zeros [BT, BV] .f32 ∧ Rr (nbOf d) = r4 ∧
      ∀ b, b < nbOf d → WTile d init n hv t (64 * b) (Bw b) ∧
        (∀ a jj, FloatV.NaNEq (FloatV.cvt ((fam4 b1 b2 b3 b4 b).get2 a jj)) ((Xs b).get2 a jj)) ∧
        P.dot cfgBF16 (Bw b) (Xs b) (Rr b) (Rr (b + 1)) := by
  obtain ⟨bw1, hbw1, x1, hx1, hd1⟩ := hw1
  have key2 : ∃ (bw : Tensor [BT, 64] BF16) (x : Tensor [64, BV] BF16), (64 < d.K →
      WTile d init n hv t 64 bw ∧ (∀ a jj, FloatV.NaNEq (FloatV.cvt (b2.get2 a jj)) (x.get2 a jj)) ∧
        P.dot cfgBF16 bw x r1 r2) ∧ (¬ 64 < d.K → r2 = r1) := by
    rcases hw2 with ⟨h, bw, hb, x, hx, hdt⟩ | ⟨h, he⟩
    · exact ⟨bw, x, fun _ => ⟨hb, hx, hdt⟩, fun h' => absurd h h'⟩
    · exact ⟨default, default, fun h' => absurd h' h, fun _ => he⟩
  have key3 : ∃ (bw : Tensor [BT, 64] BF16) (x : Tensor [64, BV] BF16), (128 < d.K →
      WTile d init n hv t 128 bw ∧ (∀ a jj, FloatV.NaNEq (FloatV.cvt (b3.get2 a jj)) (x.get2 a jj)) ∧
        P.dot cfgBF16 bw x r2 r3) ∧ (¬ 128 < d.K → r3 = r2) := by
    rcases hw3 with ⟨h, bw, hb, x, hx, hdt⟩ | ⟨h, he⟩
    · exact ⟨bw, x, fun _ => ⟨hb, hx, hdt⟩, fun h' => absurd h h'⟩
    · exact ⟨default, default, fun h' => absurd h' h, fun _ => he⟩
  have key4 : ∃ (bw : Tensor [BT, 64] BF16) (x : Tensor [64, BV] BF16), (192 < d.K →
      WTile d init n hv t 192 bw ∧ (∀ a jj, FloatV.NaNEq (FloatV.cvt (b4.get2 a jj)) (x.get2 a jj)) ∧
        P.dot cfgBF16 bw x r3 r4) ∧ (¬ 192 < d.K → r4 = r3) := by
    rcases hw4 with ⟨h, bw, hb, x, hx, hdt⟩ | ⟨h, he⟩
    · exact ⟨bw, x, fun _ => ⟨hb, hx, hdt⟩, fun h' => absurd h h'⟩
    · exact ⟨default, default, fun h' => absurd h' h, fun _ => he⟩
  obtain ⟨bw2, x2, H2, E2⟩ := key2
  obtain ⟨bw3, x3, H3, E3⟩ := key3
  obtain ⟨bw4, x4, H4, E4⟩ := key4
  refine ⟨fam4 bw1 bw2 bw3 bw4, fam4 x1 x2 x3 x4, fam5 (tl.zeros [BT, BV] .f32) r1 r2 r3 r4, rfl, ?_, ?_⟩
  · rcases nb_cases hd with ⟨h, hk⟩ | ⟨h, h1, hk⟩ | ⟨h, h1, hk⟩ | ⟨h, h1⟩
    · rw [h]
      show r1 = r4
      rw [E4 (by omega), E3 (by omega), E2 (by omega)]
    · rw [h]
      show r2 = r4
      rw [E4 (by omega), E3 (by omega)]
    · rw [h]
      show r3 = r4
      rw [E4 (by omega)]
    · rw [h]; rfl
  · intro b hb
    have hnb4 : nbOf d ≤ 4 := by have := hd.K_le; unfold nbOf; omega
    have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by omega
    rcases hb3 with rfl | rfl | rfl | rfl
    · exact ⟨hbw1, hx1, hd1⟩
    · have h : 64 < d.K := by unfold nbOf at hb; omega
      exact H2 h
    · have h : 128 < d.K := by unfold nbOf at hb; omega
      exact H3 h
    · have h : 192 < d.K := by unfold nbOf at hb; omega
      exact H4 h

/-- **The `k` dots over the blocks that exist**: tiles `Bk b`, into the decayed block `s_b`. -/
theorem chain_k {d : Inst} (hd : d.ok) {P : Platform} {init : Mem} {n hv t : Nat}
    {vb : Tensor [BT, BV] BF16} {s1 s2 s3 s4 y1 y2 y3 y4 : Tensor [64, BV] F32}
    (hk1 : KFacts d P init n hv t 0 vb s1 y1)
    (hk2 : (64 < d.K ∧ KFacts d P init n hv t 64 vb s2 y2) ∨ (¬ 64 < d.K ∧ y2 = s2))
    (hk3 : (128 < d.K ∧ KFacts d P init n hv t 128 vb s3 y3) ∨ (¬ 128 < d.K ∧ y3 = s3))
    (hk4 : (192 < d.K ∧ KFacts d P init n hv t 192 vb s4 y4) ∨ (¬ 192 < d.K ∧ y4 = s4)) :
    ∃ Bk : ℕ → Tensor [64, BT] BF16, ∀ b, b < nbOf d → KTile d init n hv t (64 * b) (Bk b) ∧
      P.dot cfgBF16 (Bk b) vb (fam4 s1 s2 s3 s4 b) (fam4 y1 y2 y3 y4 b) := by
  obtain ⟨bk1, hbk1, hd1⟩ := hk1
  have key : ∀ (base0 : Nat) (s y : Tensor [64, BV] F32),
      ((base0 < d.K ∧ KFacts d P init n hv t base0 vb s y) ∨ (¬ base0 < d.K ∧ y = s)) →
      ∃ bk : Tensor [64, BT] BF16, base0 < d.K → KTile d init n hv t base0 bk ∧ P.dot cfgBF16 bk vb s y := by
    intro base0 s y h
    rcases h with ⟨hc, bk, hb, hdt⟩ | ⟨hc, -⟩
    · exact ⟨bk, fun _ => ⟨hb, hdt⟩⟩
    · exact ⟨default, fun h' => absurd h' hc⟩
  obtain ⟨bk2, H2⟩ := key 64 s2 y2 hk2
  obtain ⟨bk3, H3⟩ := key 128 s3 y3 hk3
  obtain ⟨bk4, H4⟩ := key 192 s4 y4 hk4
  refine ⟨fam4 bk1 bk2 bk3 bk4, fun b hb => ?_⟩
  have hnb4 : nbOf d ≤ 4 := by have := hd.K_le; unfold nbOf; omega
  have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by omega
  rcases hb3 with rfl | rfl | rfl | rfl
  · exact ⟨hbk1, hd1⟩
  · exact H2 (by unfold nbOf at hb; omega)
  · exact H3 (by unfold nbOf at hb; omega)
  · exact H4 (by unfold nbOf at hb; omega)

/-- **The decay of the blocks that exist**, with the gate. -/
theorem chain_s {d : Inst} (hd : d.ok) {sh : Shape} {b1 b2 b3 b4 s1 s2 s3 s4 : Tensor sh F32} {g : F32}
    (hs1 : ∀ i, FloatV.NaNEq (FloatV.mul (b1.get i) g) (s1.get i))
    (hs2 : (64 < d.K ∧ ∀ i, FloatV.NaNEq (FloatV.mul (b2.get i) g) (s2.get i)) ∨ (¬ 64 < d.K ∧ s2 = b2))
    (hs3 : (128 < d.K ∧ ∀ i, FloatV.NaNEq (FloatV.mul (b3.get i) g) (s3.get i)) ∨ (¬ 128 < d.K ∧ s3 = b3))
    (hs4 : (192 < d.K ∧ ∀ i, FloatV.NaNEq (FloatV.mul (b4.get i) g) (s4.get i)) ∨ (¬ 192 < d.K ∧ s4 = b4)) :
    ∀ b, b < nbOf d → ∀ i, FloatV.NaNEq (FloatV.mul ((fam4 b1 b2 b3 b4 b).get i) g)
      ((fam4 s1 s2 s3 s4 b).get i) := by
  intro b hb
  have hnb4 : nbOf d ≤ 4 := by have := hd.K_le; unfold nbOf; omega
  have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by omega
  rcases hb3 with rfl | rfl | rfl | rfl
  · exact hs1
  · rcases hs2 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h
  · rcases hs3 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h
  · rcases hs4 with ⟨-, h⟩ | ⟨h, -⟩
    · exact h
    · exact absurd (by unfold nbOf at hb; omega) h

/-- **The `[V, K]` updates over the blocks that exist**: tiles `Bk b`, dots `Dd b` into zeros, and the
fusable sums into the decayed block `s_b`. -/
theorem chain_kv {d : Inst} (hd : d.ok) {P : Platform} {fusion : Bool} {init : Mem} {n hv t : Nat}
    {vb : Tensor [BT, BV] BF16} {s1 s2 s3 s4 y1 y2 y3 y4 : Tensor [BV, 64] F32}
    {d1 : Tensor [64, BV] F32}
    (hk1 : KFacts d P init n hv t 0 vb (tl.zeros [64, BV] .f32) d1)
    (hy1 : ∀ (jj : Fin BV) (a : Fin 64), AddRel fusion (s1.get2 jj a) (d1.get2 a jj) (y1.get2 jj a))
    (hk2 : (64 < d.K ∧ KUpdFacts d P fusion init n hv t 64 vb s2 y2) ∨ (¬ 64 < d.K ∧ y2 = s2))
    (hk3 : (128 < d.K ∧ KUpdFacts d P fusion init n hv t 128 vb s3 y3) ∨ (¬ 128 < d.K ∧ y3 = s3))
    (hk4 : (192 < d.K ∧ KUpdFacts d P fusion init n hv t 192 vb s4 y4) ∨ (¬ 192 < d.K ∧ y4 = s4)) :
    ∃ (Bk : ℕ → Tensor [64, BT] BF16) (Dd : ℕ → Tensor [64, BV] F32),
      ∀ b, b < nbOf d → KTile d init n hv t (64 * b) (Bk b) ∧
        P.dot cfgBF16 (Bk b) vb (tl.zeros [64, BV] .f32) (Dd b) ∧
        ∀ (jj : Fin BV) (a : Fin 64), AddRel fusion ((fam4 s1 s2 s3 s4 b).get2 jj a) ((Dd b).get2 a jj)
          ((fam4 y1 y2 y3 y4 b).get2 jj a) := by
  obtain ⟨bk1, hbk1, hd1⟩ := hk1
  have key : ∀ (base0 : Nat) (s y : Tensor [BV, 64] F32),
      ((base0 < d.K ∧ KUpdFacts d P fusion init n hv t base0 vb s y) ∨ (¬ base0 < d.K ∧ y = s)) →
      ∃ (bk : Tensor [64, BT] BF16) (dd : Tensor [64, BV] F32), base0 < d.K → KTile d init n hv t base0 bk ∧
        P.dot cfgBF16 bk vb (tl.zeros [64, BV] .f32) dd ∧
        ∀ (jj : Fin BV) (a : Fin 64), AddRel fusion (s.get2 jj a) (dd.get2 a jj) (y.get2 jj a) := by
    intro base0 s y h
    rcases h with ⟨hc, bk, hb, dd, hdt, hadd⟩ | ⟨hc, -⟩
    · exact ⟨bk, dd, fun _ => ⟨hb, hdt, hadd⟩⟩
    · exact ⟨default, default, fun h' => absurd h' hc⟩
  obtain ⟨bk2, dd2, H2⟩ := key 64 s2 y2 hk2
  obtain ⟨bk3, dd3, H3⟩ := key 128 s3 y3 hk3
  obtain ⟨bk4, dd4, H4⟩ := key 192 s4 y4 hk4
  refine ⟨fam4 bk1 bk2 bk3 bk4, fam4 d1 dd2 dd3 dd4, fun b hb => ?_⟩
  have hnb4 : nbOf d ≤ 4 := by have := hd.K_le; unfold nbOf; omega
  have hb3 : b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 := by omega
  rcases hb3 with rfl | rfl | rfl | rfl
  · exact ⟨hbk1, hd1, hy1⟩
  · exact H2 (by unfold nbOf at hb; omega)
  · exact H3 (by unfold nbOf at hb; omega)
  · exact H4 (by unfold nbOf at hb; omega)

end Claims.ChunkDeltaH
