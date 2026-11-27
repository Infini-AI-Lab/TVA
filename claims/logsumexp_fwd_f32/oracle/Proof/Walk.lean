import Proof.Addr

/-!
# The `Safe` walk of one program

`tail_noscale` and `tail_scale` walk the program after the load (the optional product, `tl.where`,
the max, the subtraction, `tl.exp`, the sum, `tl.log`, the last addition and the store) and derive
`Qf` from `final_noscale` / `final_scale`. `safe_program` walks the index arithmetic and the load,
then splits on `HAS_SCALE`.
-/

set_option maxHeartbeats 1000000

namespace Claims.LogsumexpFwd

open TVA

/-- `-float('inf')` in `float32` is `−∞`. -/
theorem negInfC_eq : negInfC = (FloatV.negInf : F32) := by decide

/-- The number of valid lanes of block `i`. -/
def nb (d : Inst) (i : Nat) : Nat := min d.D ((i + 1) * d.BD) - i * d.BD

/-- Block `i < ND` has at least one valid lane, at most `BD`. -/
theorem nb_pos {d : Inst} (hd : d.ok) {i : Nat} (hi : i < ND d) : 0 < nb d i ∧ nb d i ≤ d.BD := by
  have hb := bd_pos hd
  have h1 : i * d.BD < d.D := by
    have hi' : i < (d.D + d.BD - 1) / d.BD := hi
    by_contra h
    push Not at h
    have : (d.D + d.BD - 1) / d.BD < i + 1 :=
      (Nat.div_lt_iff_lt_mul hb).mpr (by rw [Nat.succ_mul]; omega)
    omega
  have h2 : (i + 1) * d.BD = i * d.BD + d.BD := Nat.succ_mul _ _
  unfold nb
  constructor <;> omega

/-- A lane is valid exactly when its column is below `D`. -/
theorem lt_nb_iff {d : Inst} {i l : Nat} (hl : l < d.BD) : l < nb d i ↔ i * d.BD + l < d.D := by
  have h2 : (i + 1) * d.BD = i * d.BD + d.BD := Nat.succ_mul _ _
  unfold nb; omega

/-- The block's columns are `i·BD + l`, `l < nb`. -/
theorem sum_blk {d : Inst} (i : Nat) (f : Nat → ℝ) :
    ∑ j ∈ blk d i, f j = ∑ l ∈ Finset.range (nb d i), f (i * d.BD + l) := by
  unfold blk nb
  rw [Finset.sum_Ico_eq_sum_range]

/-- A bound on the block's columns is a bound on its lanes. -/
theorem forall_blk {d : Inst} {i : Nat} {q : Nat → Prop} :
    (∀ j ∈ blk d i, q j) ↔ ∀ l, l < nb d i → q (i * d.BD + l) := by
  constructor
  · intro h l hl
    refine h _ ?_
    simp only [blk, Finset.mem_Ico]
    unfold nb at hl; omega
  · intro h j hj
    simp only [blk, Finset.mem_Ico] at hj
    have := h (j - i * d.BD) (by unfold nb; omega)
    rwa [show i * d.BD + (j - i * d.BD) = j by omega] at this

/-- The lanes after `tl.where`: the loaded (and scaled) value on a valid lane, `−∞` elsewhere. -/
theorem where_lane {d : Inst} {i : Nat} {m_d : Tensor [d.BD] Bool} {b : Tensor [d.BD] F32}
    (hmd : ∀ l : Idx [d.BD], m_d.get l = decide (i * d.BD + l.1.val < d.D)) :
    (∀ (l : Nat) (hl : l < d.BD), l < nb d i →
      (tl.where m_d b (bc negInfC)).get1 ⟨l, hl⟩ = b.get1 ⟨l, hl⟩) ∧
    (∀ (l : Nat) (hl : l < d.BD), ¬ l < nb d i →
      (tl.where m_d b (bc negInfC)).get1 ⟨l, hl⟩ = FloatV.negInf) := by
  constructor
  · intro l hl hln
    have e : (tl.where m_d b (bc negInfC)).get1 ⟨l, hl⟩ =
        if m_d.get1 ⟨l, hl⟩ then b.get1 ⟨l, hl⟩ else negInfC := rfl
    have hm : m_d.get1 ⟨l, hl⟩ = decide (i * d.BD + l < d.D) := hmd _
    rw [e, hm, decide_eq_true ((lt_nb_iff hl).mp hln)]
    rfl
  · intro l hl hln
    have e : (tl.where m_d b (bc negInfC)).get1 ⟨l, hl⟩ =
        if m_d.get1 ⟨l, hl⟩ then b.get1 ⟨l, hl⟩ else negInfC := rfl
    have hm : m_d.get1 ⟨l, hl⟩ = decide (i * d.BD + l < d.D) := hmd _
    rw [e, hm, decide_eq_false (fun h => hln ((lt_nb_iff hl).mpr h))]
    exact negInfC_eq

/-- **The program after the load, without a scale.** -/
theorem tail_noscale {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) (hs : d.hasScale = false)
    {init : Mem} (hpre : Pre d init) {n i : Nat} (hn : n < d.N) (hi : i < ND d)
    {m_d : Tensor [d.BD] Bool} {b_x : Tensor [d.BD] F32} {pz : Ptr .f32}
    (hmd : ∀ l : Idx [d.BD], m_d.get l = decide (i * d.BD + l.1.val < d.D))
    (hbx : ∀ l : Idx [d.BD], i * d.BD + l.1.val < d.D →
      b_x.get l = xsOf d init n (i * d.BD + l.1.val))
    (hpz : pz = d.z.ptr ((zIdx d n i : Nat) : Int))
    {R : Set Int} :
    Safe ((launch d).env P) R (Wf d n i) (fun _ m => Qf d init n i m) init
      (tl.maxAll true (tl.where m_d b_x (bc negInfC)) >>= fun b_m =>
       tl.sub (tl.where m_d b_x (bc negInfC)) (bc b_m) >>= fun b_t =>
       tl.exp b_t >>= fun b_e =>
       tl.sum b_e >>= fun b_s =>
       tl.log (bc b_s : Tensor [] F32) >>= fun L =>
       tl.add (bc L.item : Tensor [] F32) (bc b_m) >>= fun c =>
       tl.store1 pz c.item) := by
  obtain ⟨hnb0, hnbB⟩ := nb_pos hd hi
  have hB := bd_pos hd
  have hBle := hd.bd_le
  refine safe_maxAll hB true fun mx hmx => ?_
  refine safe_sub fun t ht => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun e he => ?_
  refine safe_sum hB fun S hS => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun L hL => ?_
  refine safe_add fun c hc => ?_
  refine safe_last ?_
  have hzlen : zIdx d n i < d.z.len := lt_of_lt_of_le (zIdx_lt hn hi) hd.z_len
  rw [hpz]
  refine safe_store1 (Buf.writable d.z hpre.2 hzlen) (fun x hx => ⟨_, rfl, hx⟩) fun m' hm' _ => ?_
  refine safe_ret ⟨c.item, hm', fun hok => ?_⟩
  -- the block's relations
  obtain ⟨hw1, hw2⟩ := where_lane (b := b_x) hmd
  set v := tl.where m_d b_x (bc negInfC) with hvdef
  have hvx : ∀ (l : Nat) (hl : l < d.BD), l < nb d i →
      v.get1 ⟨l, hl⟩ = xsOf d init n (i * d.BD + l) := by
    intro l hl hln
    rw [hw1 l hl hln]
    exact hbx (⟨l, hl⟩, ()) ((lt_nb_iff hl).mp hln)
  have hrel : BlockRel P d.BD (nb d i) v t e mx S L.item := by
    refine ⟨hnb0, hnbB, hBle, fun l hl hln => ?_, hw2, hmx, fun l hl => he (⟨l, hl⟩, ()), hS,
      hL ()⟩
    rw [hvx l hl hln, forall_blk.mp (fun j hj => (hok j hj).1) l hln]
  have hsig : sig d = 1 := by simp [sig, hs]
  have hymax : yMax d = 2 ^ 126 := by simp [yMax, hs]
  obtain ⟨hcf, hcb⟩ := final_noscale hP hrel
    (fun l hl hln => by
      have := (forall_blk.mp (fun j hj => (hok j hj).2) l hln)
      rw [hsig, hymax, mul_one] at this
      rw [hvx l hl hln]; exact this)
    (fun l hl => by
      have := ht (⟨l, hl⟩, ())
      exact this)
    (hc ())
  refine ⟨hcf, fun M hM => ?_⟩
  have hb := hcb M (fun l hl hln => by
    rw [hvx l hl hln]
    have := (forall_blk.mp hM) l hln
    rwa [hsig, mul_one] at this)
  unfold lseRef
  rw [sum_blk, hsig]
  have hsum : ∑ l ∈ Finset.range (nb d i),
      Real.exp (if hl : l < d.BD then (v.get1 ⟨l, hl⟩).toReal else 0) =
      ∑ l ∈ Finset.range (nb d i), Real.exp ((xsOf d init n (i * d.BD + l)).toReal * 1) := by
    refine Finset.sum_congr rfl fun l hl' => ?_
    have hln := Finset.mem_range.mp hl'
    have hl := lt_of_lt_of_le hln hnbB
    rw [dite_eq_left hl, hvx l hl hln, mul_one]
  rw [← hsum]
  exact hb

/-- **The program after the load, with a scale.** -/
theorem tail_scale {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) (hs : d.hasScale = true)
    {init : Mem} (hpre : Pre d init) {n i : Nat} (hn : n < d.N) (hi : i < ND d)
    {m_d : Tensor [d.BD] Bool} {b_x : Tensor [d.BD] F32} {pz : Ptr .f32}
    (hmd : ∀ l : Idx [d.BD], m_d.get l = decide (i * d.BD + l.1.val < d.D))
    (hbx : ∀ l : Idx [d.BD], i * d.BD + l.1.val < d.D →
      b_x.get l = xsOf d init n (i * d.BD + l.1.val))
    (hpz : pz = d.z.ptr ((zIdx d n i : Nat) : Int))
    {R : Set Int} :
    Safe ((launch d).env P) R (Wf d n i) (fun _ m => Qf d init n i m) init
      (tl.mul b_x (bc (FloatV.hostF32 d.scale)) >>= fun b_x =>
       tl.maxAll true (tl.where m_d b_x (bc negInfC)) >>= fun b_m =>
       tl.subc (tl.where m_d b_x (bc negInfC)) (bc b_m) >>= fun b_t =>
       tl.exp b_t >>= fun b_e =>
       tl.sum b_e >>= fun b_s =>
       tl.log (bc b_s : Tensor [] F32) >>= fun L =>
       tl.addc (bc L.item : Tensor [] F32) (bc b_m) >>= fun c =>
       tl.store1 pz c.item) := by
  obtain ⟨hnb0, hnbB⟩ := nb_pos hd hi
  have hB := bd_pos hd
  have hBle := hd.bd_le
  refine safe_mul fun b_y hby => ?_
  refine safe_maxAll hB true fun mx hmx => ?_
  refine safe_subc fun t ht => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun e he => ?_
  refine safe_sum hB fun S hS => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun L hL => ?_
  refine safe_addc fun c hc => ?_
  refine safe_last ?_
  have hzlen : zIdx d n i < d.z.len := lt_of_lt_of_le (zIdx_lt hn hi) hd.z_len
  rw [hpz]
  refine safe_store1 (Buf.writable d.z hpre.2 hzlen) (fun x hx => ⟨_, rfl, hx⟩) fun m' hm' _ => ?_
  refine safe_ret ⟨c.item, hm', fun hok => ?_⟩
  obtain ⟨hsf, hsQ⟩ := hd.scale_ok hs
  have hsig : sig d = d.scale.toReal := by simp [sig, hs, FloatV.toReal_eq]
  have hymax : yMax d = 2 ^ 20 := by simp [yMax, hs]
  have hs126 : |d.scale.toReal| ≤ 2 ^ 126 := by
    rw [FloatV.toReal_eq, ← Rat.cast_abs]; exact_mod_cast hsQ
  obtain ⟨hw1, hw2⟩ := where_lane (b := b_y) hmd
  set v := tl.where m_d b_y (bc negInfC) with hvdef
  set y : Nat → ℝ := fun l => (xsOf d init n (i * d.BD + l)).toReal * d.scale.toReal with hydef
  have hlane : ∀ (l : Nat) (hl : l < d.BD), l < nb d i →
      (v.get1 ⟨l, hl⟩).isFinite = true ∧ |(v.get1 ⟨l, hl⟩).toReal - y l| ≤
        3 * uR * |y l| + 5 * uR := by
    intro l hl hln
    rw [hw1 l hl hln]
    have hxl := hbx (⟨l, hl⟩, ()) ((lt_nb_iff hl).mp hln)
    have hfin := forall_blk.mp (fun j hj => (hok j hj).1) l hln
    have hyb := forall_blk.mp (fun j hj => (hok j hj).2) l hln
    rw [hsig, hymax] at hyb
    have hm := hby (⟨l, hl⟩, ())
    rw [hxl] at hm
    exact scale_lane hfin hsf hs126 hyb hm
  have hrel : BlockRel P d.BD (nb d i) v t e mx S L.item :=
    ⟨hnb0, hnbB, hBle, fun l hl hln => (hlane l hl hln).1, hw2, hmx, fun l hl => he (⟨l, hl⟩, ()),
      hS, hL ()⟩
  obtain ⟨hcf, hcb⟩ := final_scale hP (b := true) hrel (y := y)
    (fun l hln => by
      have hyb := forall_blk.mp (fun j hj => (hok j hj).2) l hln
      rw [hsig, hymax] at hyb
      exact hyb)
    (fun l hl hln => (hlane l hl hln).2)
    (fun l hl => ht (⟨l, hl⟩, ()))
    (hc ())
  refine ⟨hcf, fun M hM => ?_⟩
  have hb := hcb M (fun l hln => by
    have := (forall_blk.mp hM) l hln
    rwa [hsig] at this)
  unfold lseRef
  rw [sum_blk, hsig]
  exact hb

end Claims.LogsumexpFwd
