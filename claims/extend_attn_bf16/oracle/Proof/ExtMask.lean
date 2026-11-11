import Proof.Spec
import Proof.Lanes
import Proof.Load

/-!
# The masks of one new-token trip
-/

namespace Claims.ExtendAttn

open TVA

/-- `extBody` up to the window mask, then the continuation `rest`. -/
def extBodyG (d : Inst) (s : Sc d) (cm : Option (I64 × Option I64)) (mask_m : Tensor [d.BM] Bool)
    (cur_block_m_end : QI d.cm) (start_n : QI d.cm)
    (rest : Tensor [BN] Bool → Tensor [d.BM, BN] Bool → Prog (LoopSt d)) :
    Prog (LoopSt d) := do
  let offs_m : Tensor [d.BM] I32 := tl.arange 0 d.BM
  let offs_n : Tensor [BN] I32 := tl.arange 0 BN
  let mask_n : Tensor [BN] Bool :=
    (start_n + (← toQ d.cm offs_n)).ltS cur_block_m_end
  let final_mask : Tensor [d.BM, BN] Bool :=
    (mask_m.col BN).and (mask_n.row d.BM)
  let final_mask : Tensor [d.BM, BN] Bool ← match cm with
    | some (cur_seq_mask_start_idx, window_kv_offset) => do
      let ptrs : Tensor [d.BM, BN] (Ptr .i1) ← match window_kv_offset with
        | none => do
          let wko : I32 := IntV.ofInt 0
          let row_len : QI d.cm := s.cur_seq_len + (← toQ d.cm (Tensor.scalar wko)).item
          let row_off : Tensor [d.BM, BN] (QI d.cm) :=
            (← toQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m.col BN)) *
              row_len
          pure ((d.cmask.ptr 0) + cur_seq_mask_start_idx + row_off +
            (bc wko : Tensor [d.BM, BN] _) +
            (bc s.cur_seq_len_prefix : Tensor [d.BM, BN] _) +
            (bc start_n : Tensor [d.BM, BN] _) + offs_n.row d.BM)
        | some wko => do
          let row_len : I64 := (← qToI64 d.cm (Tensor.scalar s.cur_seq_len)).item + wko
          let row_off : Tensor [d.BM, BN] I64 :=
            (← tl.convert .i32 .i64
              (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m.col BN)) * row_len
          pure ((d.cmask.ptr 0) + cur_seq_mask_start_idx + row_off +
            (bc wko : Tensor [d.BM, BN] _) +
            (bc s.cur_seq_len_prefix : Tensor [d.BM, BN] _) +
            (bc start_n : Tensor [d.BM, BN] _) + offs_n.row d.BM)
      let custom_mask : Tensor [d.BM, BN] Bool ←
        tl.load ptrs (some ((mask_m.col BN).and (mask_n.row d.BM)))
          (some (← tl.convert .i32 .i1 (tl.full [d.BM, BN] (IntV.ofInt 0 : I32))))
      let cm32 : Tensor [d.BM, BN] I32 ← tl.convert .i1 .i32 custom_mask
      let custom_mask : Tensor [d.BM, BN] Bool :=
        ((cm32.map (fun x => !(IntV.eq x (IntV.ofInt 0)))).and (mask_m.col BN)).and
          (mask_n.row d.BM)
      pure (final_mask.and custom_mask)
    | none => do
      let mask_causual : Tensor [d.BM, BN] Bool :=
        Tensor.le (start_n + (← toQ d.cm (offs_n.row d.BM)))
          (← toQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m.col BN))
      let mask_causual := mask_causual.and ((mask_m.col BN).and (mask_n.row d.BM))
      pure (final_mask.and mask_causual)
  let final_mask : Tensor [d.BM, BN] Bool ←
    if 0 < d.sw then do
      let window_mask : Tensor [d.BM, BN] Bool :=
        Tensor.le
          (← toQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + offs_m.col BN))
          (start_n + (← toQ d.cm (offs_n.row d.BM)) +
            (IntV.ofNat d.sw : QI d.cm))
      pure (final_mask.and window_mask)
    else pure final_mask
  rest mask_n final_mask

/-- The trip's work (lines 548-595): the loads of `k` and `v`, the score `tl.dot`, the online
softmax and `tl.dot_add`. -/
def extTripP (d : Inst) (s : Sc d) (q : Tensor [d.BM, d.BD] BF16)
    (start_n : QI d.cm) (mask_n : Tensor [BN] Bool)
    (acc : Tensor [d.BM, d.BD] F32) (deno e_max : Tensor [d.BM] F32)
    (final_mask : Tensor [d.BM, BN] Bool) : Prog (LoopSt d) := do
  let offs_d : Tensor [d.BD] I32 := tl.arange 0 d.BD
  let offs_dv : Tensor [d.BD] I32 := tl.arange 0 d.BD
  let mask_d : Tensor [d.BD] Bool := offs_d.ltS (IntV.ofNat d.D : I32)
  let mask_dv : Tensor [d.BD] Bool := offs_dv.ltS (IntV.ofNat d.D : I32)
  let offs_n : Tensor [BN] I32 := tl.arange 0 BN
  let offs_k : Tensor [d.BD, BN] (QI d.cm) :=
    ((s.cur_seq_extend_start_idx + start_n) +
          (← toQ d.cm (offs_n.row d.BD))) *
        (← toQ d.cm (Tensor.scalar (IntV.ofNat d.skt : I32))).item +
      (← toQ d.cm (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.skh : I32)))).item +
      (← toQ d.cm (offs_d.col BN))
  let k : Tensor [d.BD, BN] BF16 ←
    tl.load ((d.k.ptr 0) + offs_k) (some ((mask_n.row d.BD).and (mask_d.col BN)))
      (some (← tl.convert .f32 .bf16 (tl.full [d.BD, BN] zf32)))
  let qk : Tensor [d.BM, BN] F32 ←
    tl.dot cfgBF16 q k (tl.zeros [d.BM, BN] .f32)
  let qk ← tl.mul qk (bc (FloatV.hostF32 d.sm_scale))
  let qk := tl.where final_mask qk (bc ninf32)
  let row_max : Tensor [d.BM] F32 ← tl.max1 true qk
  let row_max_fixed := tl.where (row_max.feq (bc ninf32)) (bc neg1e20) row_max
  let n_e_max : Tensor [d.BM] F32 ← tl.maximum row_max_fixed e_max
  let re_scale : Tensor [d.BM] F32 ← tl.exp (← tl.sub e_max n_e_max)
  let p : Tensor [d.BM, BN] F32 ← tl.exp (← tl.sub qk (n_e_max.col BN))
  let deno ← tl.mul_add deno re_scale (← tl.sum1 p)
  let offs_v : Tensor [BN, d.BD] (QI d.cm) :=
    ((s.cur_seq_extend_start_idx + start_n) +
          (← toQ d.cm (offs_n.col d.BD))) *
        (← toQ d.cm (Tensor.scalar (IntV.ofNat d.svt : I32))).item +
      (← toQ d.cm (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.svh : I32)))).item +
      (← toQ d.cm (offs_dv.row BN))
  let v : Tensor [BN, d.BD] BF16 ←
    tl.load ((d.v.ptr 0) + offs_v) (some ((mask_n.col d.BD).and (mask_dv.row BN)))
      (some (← tl.convert .f32 .bf16 (tl.full [BN, d.BD] zf32)))
  let p16 : Tensor [d.BM, BN] BF16 ← tl.cvt p
  let acc ← tl.dot_add cfgBF16 p16 v (← tl.mul acc (re_scale.col d.BD))
  let e_max := n_e_max
  pure (acc, deno, e_max)

/-- The rest of a trip (lines 545-596): the tile skip, then the trip's work. -/
def extRest (d : Inst) (s : Sc d) (q : Tensor [d.BM, d.BD] BF16)
    (start_n : QI d.cm) (mask_n : Tensor [BN] Bool)
    (acc : Tensor [d.BM, d.BD] F32) (deno e_max : Tensor [d.BM] F32)
    (final_mask : Tensor [d.BM, BN] Bool) : Prog (LoopSt d) := do
  let SKIP_TILE : Tensor [] Bool ←
    if d.cm || decide (0 < d.sw) then do
      let fm : Tensor [d.BM, BN] I32 ← tl.convert .i1 .i32 final_mask
      pure (Tensor.scalar (IntV.eq
        (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 fm))
        (IntV.ofInt 0)))
    else pure (Tensor.scalar false)
  if (Tensor.not SKIP_TILE).item then extTripP d s q start_n mask_n acc deno e_max final_mask
  else pure (acc, deno, e_max)


/-- The tree/causal lane of the claim for new token `i` against new token `j`. -/
def mpLane (d : Inst) (init : Mem) (i : Nat) (r : Nat) (j : Nat) : Prop :=
  if d.cm = true then maskBit d init (bOf d i) (mOf d i * d.BM + r) j = true
  else j ≤ mOf d i * d.BM + r

/-- The window lane. -/
def winLane (d : Inst) (i : Nat) (r : Nat) (j : Nat) : Prop :=
  0 < d.sw → mOf d i * d.BM + r ≤ j + d.sw

instance (d : Inst) (init : Mem) (i r j : Nat) : Decidable (mpLane d init i r j) := by
  unfold mpLane; infer_instance

instance (d : Inst) (i r j : Nat) : Decidable (winLane d i r j) := by
  unfold winLane; infer_instance

/-- The extend-phase validity flag of a valid row at offset `S + c` holds exactly when the position is inside the block, passes the mask-pointer test, and lies in the window. -/
theorem vE_iff {d : Inst} {init : Mem} (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size)
    {r : Fin d.BM} (hr : validRow d init i r) (S c : Nat) :
    vE d init i r S c = true ↔
      (S + c < blockEnd d init i ∧ mpLane d init i r (S + c) ∧ winLane d i r (S + c)) := by
  obtain ⟨hb, -, -⟩ := prog_range hi
  unfold vE keys attends blockEnd mpLane winLane
  have hr' : r.val < d.BM := r.isLt
  have hrb : (mOf d i + 1) * d.BM = mOf d i * d.BM + d.BM := by ring
  rw [hrb]
  have hv : mOf d i * d.BM + r.val < extOf d init (bOf d i) := hr
  obtain ⟨M, hMdef⟩ : ∃ M, M = mOf d i * d.BM := ⟨_, rfl⟩
  rw [← hMdef] at hv ⊢
  simp only [decide_eq_true_eq, Finset.mem_filter, Finset.mem_range, Bool.and_eq_true,
    Bool.or_eq_true]
  by_cases hcm : d.cm = true
  · have hM := (hpre.2.2.2.2.2.2.2.2.2.2.2.1 hcm (bOf d i) hb).2.2.2
    simp only [hcm, ite_true]
    have hb2 := (hM _ hv).2
    have e1 : preOf d init (bOf d i) + (S + c) - preOf d init (bOf d i) = S + c := by omega
    rw [e1]
    constructor
    · rintro ⟨h1, h2, h3⟩
      have h2' : maskBit d init (bOf d i) (M + ↑r) (S + c) = true := by
        rcases h2 with h | h
        · omega
        · exact h
      refine ⟨?_, h2', ?_⟩
      · have : S + c ≤ M + r.val := by
          by_contra hn
          have := hb2 (S + c) (by omega) (by omega)
          rw [this] at h2'; exact absurd h2' (by simp)
        omega
      · intro hw; omega
    · rintro ⟨h1, h2, h3⟩
      refine ⟨by omega, Or.inr h2, ?_⟩
      rcases Nat.eq_zero_or_pos d.sw with h | h
      · exact Or.inl h
      · exact Or.inr (by have := h3 h; omega)
  · simp only [hcm, Bool.false_eq_true, ite_false, decide_eq_true_eq]
    constructor
    · rintro ⟨h1, h2, h3⟩
      refine ⟨by omega, by omega, ?_⟩
      intro hw; omega
    · rintro ⟨h1, h2, h3⟩
      refine ⟨by omega, Or.inr (by omega), ?_⟩
      rcases Nat.eq_zero_or_pos d.sw with h | h
      · exact Or.inl h
      · exact Or.inr (by have := h3 h; omega)

/-- The extend length of the block is at most 2^18. -/
theorem ext_le_pow {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) : extOf d init (bOf d i) ≤ 2 ^ 18 := by
  obtain ⟨hb, -, -⟩ := prog_range hi
  have := req_sizes hpre hb
  have := hd.maxExt_le
  omega

/-- The block end never exceeds the extend length of its batch. -/
theorem blockEnd_le_ext (d : Inst) (init : Mem) (i : Nat) : blockEnd d init i ≤ extOf d init (bOf d i) :=
  Nat.min_le_left _ _

/-- Lane `c` of `arange 0 BN` has integer value `c`. -/
theorem ar_toInt (c : Fin BN) : ((tl.arange 0 BN : Tensor [BN] I32).get1 c).toInt = c :=
  IntV.I32.toInt_tl_arange (by norm_num [BN]) c

/-- Lane `r` of `arange 0 BM` has integer value `r`. -/
theorem arM_toInt (d : Inst) (hd : d.ok) (r : Fin d.BM) :
    ((tl.arange 0 d.BM : Tensor [d.BM] I32).get1 r).toInt = r :=
  IntV.I32.toInt_tl_arange (le_trans (bm_le hd) (by norm_num)) r

/-- Lane `c` of the query-width cast of `arange 0 BN` has integer value `c`. -/
theorem arQ_toInt (qo32 : Bool) (c : Fin BN) :
    ((toQv qo32 (tl.arange 0 BN : Tensor [BN] I32)).get1 c).toInt = c := by
  have e : (toQv qo32 (tl.arange 0 BN : Tensor [BN] I32)).get1 c =
      castQ qo32 ((tl.arange 0 BN : Tensor [BN] I32).get1 c) := toQv_get _ _ _
  rw [e, castQ_toInt]; exact ar_toInt c

/-- `mask_n`'s lanes. -/
theorem mn_lane {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {cbe sn : QI d.cm} {S : Nat}
    (hcbe : cbe.toInt = (blockEnd d init i : Int)) (hsn : sn.toInt = (S : Int))
    (hS : S < blockEnd d init i) (c : Fin BN) :
    ((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).get1 c = decide (S + c < blockEnd d init i) := by
  have h1 := ext_le_pow hd hpre hi
  have h2 := blockEnd_le_ext d init i
  have hc := c.isLt
  have e : ((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).get1 c).toInt = ((S + c : Nat) : Int) := by
    simp only [Tensor.get1_sadd]
    exact IntV.add_nat hsn (arQ_toInt d.cm c) (lt_pow_qoW d.cm (by simp [BN] at hc; omega))
  rw [Tensor.get1_ltS, IntV.lt_eq, e, hcbe]
  simp only [decide_eq_decide]; omega


/-- Row `r` of the block-row offset `cur_block_m * BM + arange` has integer value `mOf * BM + r`. -/
theorem mrow_lane {d : Inst} (hd : d.ok) {init : Mem} {i : Nat} (hi : i < (launch d).size)
    {s : Sc d} (hs : ScOk d init i s) (r : Fin d.BM) :
    ((s.cur_block_m * (IntV.ofNat d.BM : I32) + (tl.arange 0 d.BM : Tensor [d.BM] I32)).get1 r).toInt =
      ((mOf d i * d.BM + r : Nat) : Int) := by
  obtain ⟨-, -, hm⟩ := prog_range hi
  have hnb := hd.nb_lt
  have hbm := bm_le hd
  have hB : (IntV.ofNat d.BM : I32).toInt = d.BM := IntV.I32.toInt_ofNat (by omega)
  have hmb : mOf d i * d.BM ≤ 2 ^ 16 * 128 := Nat.mul_le_mul (by omega) hbm
  have h1 : (s.cur_block_m * (IntV.ofNat d.BM : I32)).toInt = ((mOf d i * d.BM : Nat) : Int) :=
    IntV.mul_ofNat hs.blk (by omega)
  exact IntV.I32.toInt_sadd_arange (n := d.BM) h1 (by omega) r

/-- Entry `(r, c)` of the broadcast, query-width-cast block-row offset has the same integer value as `mrow_lane` gives. -/
theorem mrowQ_lane {d : Inst} (hd : d.ok) {init : Mem} {i : Nat} (hi : i < (launch d).size)
    {s : Sc d} (hs : ScOk d init i s) (r : Fin d.BM) (c : Fin BN) :
    ((toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) +
      (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)).get2 r c).toInt =
      ((mOf d i * d.BM + r : Nat) : Int) := by
  have e : (toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) +
      (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)).get2 r c = castQ d.cm
      ((s.cur_block_m * (IntV.ofNat d.BM : I32) +
      (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN : Tensor [d.BM, BN] I32).get2 r c) :=
    toQv_get (s := [d.BM, BN]) d.cm _ (r, c, ())
  rw [e, castQ_toInt]
  exact mrow_lane hd hi hs r

/-- Entry `(r, c)` of the row-broadcast query-width cast of `arange 0 BN` has integer value `c`. -/
theorem arQ_row_lane (d : Inst) (r : Fin d.BM) (c : Fin BN) :
    ((toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM)).get2 r c).toInt = c := by
  rw [toQv_row_get, castQ_toInt]; exact ar_toInt c


/-- The sliding-window mask computation is safe and yields a mask that agrees lane-wise with `winLane`. -/
theorem win_safe {P : Platform} {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    {sn : QI d.cm} {S : Nat} (hsn : sn.toInt = (S : Int)) (hS : S < blockEnd d init i)
    {Q : LoopSt d → Mem → Prop} {m : Mem} (mn : Tensor [BN] Bool) (fm0 : Tensor [d.BM, BN] Bool)
    (F : Fin d.BM → Fin BN → Bool) (hF : ∀ r c, fm0.get2 r c = F r c)
    (rest : Tensor [BN] Bool → Tensor [d.BM, BN] Bool → Prog (LoopSt d))
    (hrest : ∀ fm : Tensor [d.BM, BN] Bool,
      (∀ r c, fm.get2 r c = (F r c && decide (winLane d i r (S + c)))) →
      Safe ((launch d).env P) (Rf d) ∅ Q m (rest mn fm)) :
    Safe ((launch d).env P) (Rf d) ∅ Q m
      (if 0 < d.sw then do
          let a : Tensor [d.BM, BN] (QI d.cm) ← toQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)
          let b : Tensor [d.BM, BN] (QI d.cm) ← toQ d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM)
          rest mn (fm0.and (a.le (sn + b + (IntV.ofNat d.sw : QI d.cm))))
        else rest mn fm0) := by
  have h1 := ext_le_pow hd hpre hi
  have h2 := blockEnd_le_ext d init i
  have h3 := hd.sw_le
  split_ifs with hsw
  · simp only [safe_toQ]
    refine hrest _ fun r c => ?_
    have hc := c.isLt
    have hA := mrowQ_lane hd hi hs r c
    have hb := arQ_row_lane d r c
    have hB : ((sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM) +
        (IntV.ofNat d.sw : QI d.cm)).get2 r c).toInt = ((S + c + d.sw : Nat) : Int) := by
      have e : (sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM) +
        (IntV.ofNat d.sw : QI d.cm)).get2 r c = sn + (toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM)).get2 r c + (IntV.ofNat d.sw : QI d.cm) := rfl
      rw [e]
      exact IntV.add_nat (IntV.add_nat hsn hb (lt_pow_qoW d.cm (by simp [BN] at hc; omega)))
        (IntV.toInt_ofNat (lt_pow_qoW d.cm (by omega))) (lt_pow_qoW d.cm (by simp [BN] at hc; omega))
    have e : (fm0.and ((toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) +
      (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)).le (sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM) + (IntV.ofNat d.sw : QI d.cm)))).get2 r c =
      (fm0.get2 r c && IntV.le ((toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) +
      (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)).get2 r c) ((sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM) + (IntV.ofNat d.sw : QI d.cm)).get2 r c)) := rfl
    rw [e, hF, IntV.le_eq, hA, hB]
    simp only [winLane, hsw, true_implies, Nat.cast_le]
  · refine hrest _ fun r c => ?_
    simp [winLane, hsw, hF]

/-- Combining the row-valid flag, the in-block flag and a mask flag that is correct when both hold gives the lane predicate. -/
theorem lane_core {d : Inst} {init : Mem} (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size)
    (r : Fin d.BM) (S c : Nat) (A B M : Bool) (hA : A = decide (validRow d init i r))
    (hB : B = decide (S + c < blockEnd d init i))
    (hM : (A && B) = true → (M = true ↔ mpLane d init i r (S + c))) :
    (((A && B) && M) && decide (winLane d i r (S + c))) =
      (decide (validRow d init i r) && vE d init i r S c) := by
  by_cases hv : validRow d init i r
  · have hA' : A = true := by rw [hA]; exact decide_eq_true hv
    have hvE := vE_iff hpre hi hv S c
    have hBb : B = true ↔ S + c < blockEnd d init i := by rw [hB]; exact decide_eq_true_iff
    subst hA'
    rw [decide_eq_true hv, Bool.true_and, Bool.true_and]
    apply Bool.eq_iff_iff.mpr
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    rw [hvE, hBb]
    constructor
    · rintro ⟨⟨hb, hm⟩, hw⟩
      exact ⟨hb, (hM (by simpa using hBb.mpr hb)).mp hm, hw⟩
    · rintro ⟨hb, hm, hw⟩
      exact ⟨⟨hb, (hM (by simpa using hBb.mpr hb)).mpr hm⟩, hw⟩
  · have hA' : A = false := by rw [hA]; exact decide_eq_false hv
    subst hA'
    rw [decide_eq_false hv]
    simp

/-- The causal comparison of the position `S + c` against the row index agrees lane-wise with the causal predicate. -/
theorem causal_lane {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s) {sn : QI d.cm} {S : Nat}
    (hsn : sn.toInt = (S : Int)) (hS : S < blockEnd d init i) (r : Fin d.BM) (c : Fin BN) :
    IntV.le ((sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM)).get (r, c, ()))
      ((toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) +
        (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)).get (r, c, ())) =
      decide (S + c ≤ mOf d i * d.BM + r) := by
  have h1 := ext_le_pow hd hpre hi
  have h2 := blockEnd_le_ext d init i
  have hc := c.isLt
  have hA := mrowQ_lane hd hi hs r c
  have hb := arQ_row_lane d r c
  have hB : ((sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BM)).get (r, c, ())).toInt =
      ((S + c : Nat) : Int) :=
    IntV.add_nat hsn hb (lt_pow_qoW d.cm (by simp [BN] at hc; omega))
  rw [IntV.le_eq]
  have hA' : ((toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) +
        (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)).get (r, c, ())).toInt = ((mOf d i * d.BM + r : Nat) : Int) := hA
  rw [hB, hA']
  simp only [Nat.cast_le]

section
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}

/-- `tl.convert(0, int1)` (the `other` of the mask's load) is a tile of `false`. -/
theorem safe_conv_zero_i1 {k : Tensor s Bool → Prog α} :
    Safe E R W Q m (tl.convert .i32 .i1 (tl.full s (IntV.ofInt 0 : I32)) >>= k) ↔
      Safe E R W Q m (k (tl.full s false)) := by
  refine safe_convert_iff fun i r => ?_
  simp [ConvRel, ofIntRel, tl.full, IntV.toInt_ofInt_zero]
end

/-- When the custom-mask flag is set, the mask buffer lies in the readable region. -/
theorem cmask_sub_Rf (d : Inst) (hcm : d.cm = true) : d.cmask.bytes ⊆ Rf d := by
  intro x hx
  simp only [Rf, Set.mem_union, hcm, ite_true]
  tauto

/-- A mask element of a row of the request and a new token inside it is in `custom_mask`. -/
theorem maskIdx_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) (hcm : d.cm = true) {r j : Nat}
    (hr : validRow d init i r) (hj : j < extOf d init (bOf d i)) :
    maskIdx d init (bOf d i) (mOf d i * d.BM + r) j < d.cmask.len := by
  obtain ⟨hb, -, -⟩ := prog_range hi
  have hM := (hpre.2.2.2.2.2.2.2.2.2.2.2.1 hcm (bOf d i) hb).2.2.1
  have hlen := hd.cmask_len hcm
  have hv : mOf d i * d.BM + r < extOf d init (bOf d i) := hr
  unfold maskIdx
  generalize mOf d i * d.BM + r = I at *
  have h1 : (I + 1) * (preOf d init (bOf d i) + extOf d init (bOf d i) + wOff d init (bOf d i)) ≤
      extOf d init (bOf d i) * (preOf d init (bOf d i) + extOf d init (bOf d i) + wOff d init (bOf d i)) :=
    Nat.mul_le_mul_right _ hv
  have h2 : (I + 1) * (preOf d init (bOf d i) + extOf d init (bOf d i) + wOff d init (bOf d i)) =
      I * (preOf d init (bOf d i) + extOf d init (bOf d i) + wOff d init (bOf d i)) +
      (preOf d init (bOf d i) + extOf d init (bOf d i) + wOff d init (bOf d i)) := by ring
  omega

/-- The tree-mask load is safe and its result agrees lane-wise with the mask-pointer predicate. -/
theorem tree_load_safe {P : Platform} {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    (hcm : d.cm = true) {cbe sn : QI d.cm} {S : Nat}
    (hcbe : cbe.toInt = (blockEnd d init i : Int)) (hsn : sn.toInt = (S : Int))
    (hS : S < blockEnd d init i) {Q : LoopSt d → Mem → Prop}
    (ptrs : Tensor [d.BM, BN] (Ptr .i1))
    (hptr : ∀ (r : Fin d.BM) (c : Fin BN), validRow d init i r → S + c < blockEnd d init i →
      ptrs.get2 r c = d.cmask.ptr (maskIdx d init (bOf d i) (mOf d i * d.BM + r) (S + c) : Nat))
    {K : Tensor [d.BM, BN] Bool → Prog (LoopSt d)}
    (hK : ∀ cmk : Tensor [d.BM, BN] Bool,
      (∀ (r : Fin d.BM) (c : Fin BN), cmk.get2 r c =
        ((decide (validRow d init i r) && decide (S + c < blockEnd d init i)) &&
          maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c))) →
      Safe ((launch d).env P) (Rf d) ∅ Q init (K cmk)) :
    Safe ((launch d).env P) (Rf d) ∅ Q init
      (tl.convert .i32 .i1 (tl.full [d.BM, BN] (IntV.ofInt 0 : I32)) >>= fun other =>
        tl.load ptrs (some (((maskM d s).col BN).and
          (((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).row d.BM))) (some other) >>= K) := by
  have mmL := maskM_lane hd hi hs
  have mnL := mn_lane hd hpre hi hcbe hsn hS
  rw [safe_conv_zero_i1]
  have hext := blockEnd_le_ext d init i
  refine safe_load_en d.cmask (fun ix : Idx [d.BM, BN] => decide (validRow d init i ix.1) &&
      decide (S + ix.2.1 < blockEnd d init i))
    (fun ix : Idx [d.BM, BN] => maskIdx d init (bOf d i) (mOf d i * d.BM + ix.1) (S + ix.2.1)) ?_ ?_ ?_
    (hpre.2.2.2.2.2.2.2.2.2.1 hcm).1 (cmask_sub_Rf d hcm) fun v hv1 hv2 => hK v fun r c => ?_
  · rintro ⟨r, c, ⟨⟩⟩
    show ((maskM d s).get1 r && ((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).get1 c) =
      (decide (validRow d init i r) && decide (S + c < blockEnd d init i))
    rw [mmL r, mnL c]
  · rintro ⟨r, c, ⟨⟩⟩ hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact hptr r c hen.1 hen.2
  · rintro ⟨r, c, ⟨⟩⟩ hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact maskIdx_lt (j := S + c) hd hpre hi hcm hen.1 (by omega)
  · by_cases hen : (decide (validRow d init i r) && decide (S + c < blockEnd d init i)) = true
    · have h := hv1 (r, c, ()) hen
      unfold maskBit
      have h' : init.read .i1 (d.cmask.ptr (maskIdx d init (bOf d i) (mOf d i * d.BM + r) (S + c) : Nat)).addr = some (v.get2 r c) := h
      rw [hen, Bool.true_and]
      rw [h']
      rfl
    · have hf : (decide (validRow d init i r) && decide (S + c < blockEnd d init i)) = false :=
        Bool.eq_false_iff.mpr hen
      rw [hf, Bool.false_and]
      have := hv2 (r, c, ()) hf (tl.full [d.BM, BN] false) rfl
      exact this

/-- `qo_indptr`'s type to `int64`, as a value. -/
def qToI64v {s : Shape} : (qo32 : Bool) → Tensor s (QI qo32) → Tensor s I64
  | true, x => x.castInt
  | false, x => x

section
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}

/-- Safety of `qToI64` followed by a continuation equals safety of the continuation applied to the widened tensor `qToI64v`. -/
theorem safe_qToI64 (qo32 : Bool) {x : Tensor s (QI qo32)} {k : Tensor s I64 → Prog α} :
    Safe E R W Q m (qToI64 qo32 x >>= k) ↔ Safe E R W Q m (k (qToI64v qo32 x)) := by
  cases qo32
  · exact Iff.rfl
  · exact safe_conv64
end

/-- Widening a scalar query-width integer to 64 bits preserves its integer value. -/
theorem qToI64v_scalar_toInt (qo32 : Bool) (x : QI qo32) :
    ((qToI64v qo32 (Tensor.scalar x)).item).toInt = x.toInt := by
  cases qo32
  · rfl
  · exact IntV.toInt_cast x (by simp [qoW])

/-- With the custom mask on, the mask-pointer offset of batch `b` is at most 2^40. -/
theorem wOff_le {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.batch)
    (hcm : d.cm = true) : wOff d init b ≤ 2 ^ 40 := by
  unfold wOff
  split_ifs with hsw
  · obtain ⟨w', hw', -, hw2⟩ := (hpre.2.2.2.2.2.2.2.2.2.2.2.1 hcm b hb).2.1 hsw
    unfold val64
    rw [hw']
    simp only
    omega
  · omega

/-- Comparing `b2i x` with 0 for inequality recovers the boolean `x`. -/
theorem b2i_ne (x : Bool) : (!IntV.eq (b2i x) (IntV.ofInt 0)) = x := by
  cases x <;> decide


/-- **The masks of a trip**: the continuation `rest` sees `mask_n` and a `final_mask` whose lanes are
`valid row ∧ vE` (a key the row attends). -/
theorem extBodyG_safe {P : Platform} {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    {cm : Option (I64 × Option I64)}
    (hcm : CmOk d init i cm) {cbe sn : QI d.cm} {S : Nat}
    (hcbe : cbe.toInt = (blockEnd d init i : Int)) (hsn : sn.toInt = (S : Int))
    (hS : S < blockEnd d init i) {Q : LoopSt d → Mem → Prop}
    {rest : Tensor [BN] Bool → Tensor [d.BM, BN] Bool → Prog (LoopSt d)}
    (hrest : ∀ fm : Tensor [d.BM, BN] Bool,
      (∀ (r : Fin d.BM) (c : Fin BN), fm.get2 r c = (decide (validRow d init i r) && vE d init i r S c)) →
      Safe ((launch d).env P) (Rf d) ∅ Q init
        (rest ((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe) fm)) :
    Safe ((launch d).env P) (Rf d) ∅ Q init
      (extBodyG d s cm (maskM d s) cbe sn rest) := by
  have mmL := maskM_lane hd hi hs
  have mnL := mn_lane hd hpre hi hcbe hsn hS
  unfold extBodyG
  simp only [safe_toQ]
  rcases hcm with ⟨hcm0, rfl⟩ | ⟨hcm1, msi, wo, rfl, hmsi, hwo⟩
  · simp only [safe_toQ, pure_bind]
    refine win_safe (P := P) hd hpre hi hs hsn hS _ _ (fun r c =>
      ((decide (validRow d init i r) && decide (S + c < blockEnd d init i)) &&
        decide (S + c ≤ mOf d i * d.BM + r))) ?_ rest ?_
    · intro r c
      dsimp only [Tensor.and, Tensor.le, Tensor.get2, Tensor.map2, Tensor.get]
      have hA : ((maskM d s).col BN).get (r, c, ()) = decide (validRow d init i r) := mmL r
      have hB : (((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).row d.BM).get (r, c, ())
        = decide (S + c < blockEnd d init i) := mnL c
      rw [hA, hB, causal_lane hd hpre hi hs hsn hS r c]
      cases decide (validRow d init i r) <;> cases decide (S + c < blockEnd d init i) <;>
        cases decide (S + c ≤ mOf d i * d.BM + r) <;> rfl
    · intro fm hfm
      refine hrest fm fun r c => ?_
      rw [hfm r c]
      refine lane_core hpre hi r S c _ _ (decide (S + c ≤ mOf d i * d.BM + r)) rfl rfl ?_
      intro _
      simp only [decide_eq_true_eq, mpLane, hcm0, Bool.false_eq_true, ite_false]
  · rcases hwo with ⟨hsw0, rfl⟩ | ⟨hswp, w, rfl, hw⟩
    · simp only [safe_toQ, pure_bind]
      refine tree_load_safe hd hpre hi hs hcm1 hcbe hsn hS _ ?_ ?_
      · intro r c hr hb
        simp only [Tensor.get2_ptrAdd, Tensor.get2_ptrAddT, Ptr.add_int, Buf.ptr_offset]
        obtain ⟨hb0, -, -⟩ := prog_range hi
        have hrb : mOf d i * d.BM + r < extOf d init (bOf d i) := hr
        have hext := ext_le_pow hd hpre hi
        have hext12 : extOf d init (bOf d i) ≤ 2 ^ 12 := by
          have := (req_sizes hpre hb0).2.1
          have := hd.cm_maxExt hcm1
          omega
        have hsz := (req_sizes hpre hb0).2.2.2
        have hlen : s.cur_seq_len.toInt = ((preOf d init (bOf d i) + extOf d init (bOf d i) : Nat) : Int) := by
          rw [hs.len]; push_cast; rfl
        have hzero : ((toQv d.cm (Tensor.scalar (IntV.ofInt 0 : I32))).item).toInt = ((0 : Nat) : Int) := by
          rw [toQv_scalar_item, castQ_toInt]; exact IntV.toInt_ofInt_zero
        have hrl : (s.cur_seq_len + (toQv d.cm (Tensor.scalar (IntV.ofInt 0 : I32))).item).toInt =
            ((preOf d init (bOf d i) + extOf d init (bOf d i) + 0 : Nat) : Int) :=
          IntV.add_nat hlen hzero (lt_pow_qoW d.cm (by omega))
        have hroff : ((toQv d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32) + (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN) *
            (s.cur_seq_len + (toQv d.cm (Tensor.scalar (IntV.ofInt 0 : I32))).item)).get2 r c).toInt =
            (((mOf d i * d.BM + r) * (preOf d init (bOf d i) + extOf d init (bOf d i) + 0) : Nat) : Int) := by
          refine IntV.mul_nat (mrowQ_lane hd hi hs r c) hrl (lt_pow_qoW d.cm ?_)
          have : (mOf d i * d.BM + r) * (preOf d init (bOf d i) + extOf d init (bOf d i) + 0) ≤
              2 ^ 12 * 2 ^ 18 := Nat.mul_le_mul (by omega) (by omega)
          norm_num at this ⊢; omega
        have e0 : ((bc (IntV.ofInt 0 : I32) : Tensor [d.BM, BN] I32).get2 r c).toInt = 0 := IntV.toInt_ofInt_zero
        have e1 : ((bc s.cur_seq_len_prefix : Tensor [d.BM, BN] I32).get2 r c).toInt = preOf d init (bOf d i) := hs.pre
        have e2 : ((bc sn : Tensor [d.BM, BN] (QI d.cm)).get2 r c).toInt = S := hsn
        have e3 : ((((tl.arange 0 BN : Tensor [BN] I32)).row d.BM).get2 r c).toInt = c := ar_toInt c
        have hw0 : wOff d init (bOf d i) = 0 := by simp [wOff, hsw0]
        rw [hroff, e0, e1, e2, e3, hmsi]
        unfold maskIdx
        rw [hw0]
        congr 1
        push_cast
        ring
      · intro cmk hcmk
        rw [safe_convB]
        refine win_safe (P := P) hd hpre hi hs hsn hS _ _ (fun r c =>
          ((decide (validRow d init i r) && decide (S + c < blockEnd d init i)) &&
            maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c))) ?_ rest ?_
        · intro r c
          dsimp only [Tensor.and, Tensor.map2, Tensor.map, Tensor.get2, Tensor.get]
          have hA : ((maskM d s).col BN).get (r, c, ()) = decide (validRow d init i r) := mmL r
          have hB : (((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).row d.BM).get (r, c, ())
            = decide (S + c < blockEnd d init i) := mnL c
          have hv : cmk.get (r, c, ()) = ((decide (validRow d init i r) && decide (S + c < blockEnd d init i)) &&
            maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c)) := hcmk r c
          rw [hA, hB]
          rw [b2i_ne, hv]
          cases decide (validRow d init i r) <;> cases decide (S + c < blockEnd d init i) <;>
            cases maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c) <;> rfl
        · intro fm hfm
          refine hrest fm fun r c => ?_
          rw [hfm r c]
          refine lane_core hpre hi r S c _ _ (maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c)) rfl rfl ?_
          intro _
          simp only [mpLane, hcm1, ite_true]
    · simp only [pure_bind, safe_qToI64]
      rw [safe_conv64]
      refine tree_load_safe hd hpre hi hs hcm1 hcbe hsn hS _ ?_ ?_
      · intro r c hr hb
        simp only [Tensor.get2_ptrAdd, Tensor.get2_ptrAddT, Ptr.add_int, Buf.ptr_offset]
        obtain ⟨hb0, -, -⟩ := prog_range hi
        have hrb : mOf d i * d.BM + r < extOf d init (bOf d i) := hr
        have hext := ext_le_pow hd hpre hi
        have hext12 : extOf d init (bOf d i) ≤ 2 ^ 12 := by
          have := (req_sizes hpre hb0).2.1
          have := hd.cm_maxExt hcm1
          omega
        have hsz := (req_sizes hpre hb0).2.2.2
        have hwle := wOff_le hpre hb0 hcm1
        have hlen : s.cur_seq_len.toInt = ((preOf d init (bOf d i) + extOf d init (bOf d i) : Nat) : Int) := by
          rw [hs.len]; push_cast; rfl
        have hit : ((qToI64v d.cm (Tensor.scalar s.cur_seq_len)).item).toInt =
            ((preOf d init (bOf d i) + extOf d init (bOf d i) : Nat) : Int) := by
          rw [qToI64v_scalar_toInt, hlen]
        have hrl : ((qToI64v d.cm (Tensor.scalar s.cur_seq_len)).item + w).toInt =
            ((preOf d init (bOf d i) + extOf d init (bOf d i) + wOff d init (bOf d i) : Nat) : Int) := by
          refine IntV.add_nat hit hw ?_
          norm_num; omega
        have hmr : (((s.cur_block_m * (IntV.ofNat d.BM : I32) + (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN).castInt :
            Tensor [d.BM, BN] I64).get2 r c).toInt = ((mOf d i * d.BM + r : Nat) : Int) := by
          have e1 : ((s.cur_block_m * (IntV.ofNat d.BM : I32) + (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN :
            Tensor [d.BM, BN] I32).castInt : Tensor [d.BM, BN] I64).get2 r c =
            IntV.cast ((s.cur_block_m * (IntV.ofNat d.BM : I32) + (tl.arange 0 d.BM : Tensor [d.BM] I32)).get1 r) := rfl
          rw [e1, IntV.toInt_cast _ (by norm_num)]
          exact mrow_lane hd hi hs r
        have hroff : ((((s.cur_block_m * (IntV.ofNat d.BM : I32) + (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN :
            Tensor [d.BM, BN] I32).castInt : Tensor [d.BM, BN] I64) *
            ((qToI64v d.cm (Tensor.scalar s.cur_seq_len)).item + w)).get2 r c).toInt =
            (((mOf d i * d.BM + r) * (preOf d init (bOf d i) + extOf d init (bOf d i) + wOff d init (bOf d i)) : Nat) : Int) := by
          refine IntV.mul_nat hmr hrl ?_
          have : (mOf d i * d.BM + r) * (preOf d init (bOf d i) + extOf d init (bOf d i) + wOff d init (bOf d i)) ≤
              2 ^ 12 * (2 ^ 18 + 2 ^ 40) := Nat.mul_le_mul (by omega) (by omega)
          norm_num at this ⊢; omega
        have e1 : ((bc s.cur_seq_len_prefix : Tensor [d.BM, BN] I32).get2 r c).toInt = preOf d init (bOf d i) := hs.pre
        have e2 : ((bc sn : Tensor [d.BM, BN] (QI d.cm)).get2 r c).toInt = S := hsn
        have e3 : ((((tl.arange 0 BN : Tensor [BN] I32)).row d.BM).get2 r c).toInt = c := ar_toInt c
        have e4 : ((bc w : Tensor [d.BM, BN] I64).get2 r c).toInt = wOff d init (bOf d i) := hw
        rw [hroff, e1, e2, e3, e4, hmsi]
        unfold maskIdx
        congr 1
        push_cast
        ring
      · intro cmk hcmk
        rw [safe_convB]
        refine win_safe (P := P) hd hpre hi hs hsn hS _ _ (fun r c =>
          ((decide (validRow d init i r) && decide (S + c < blockEnd d init i)) &&
            maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c))) ?_ rest ?_
        · intro r c
          dsimp only [Tensor.and, Tensor.map2, Tensor.map, Tensor.get2, Tensor.get]
          have hA : ((maskM d s).col BN).get (r, c, ()) = decide (validRow d init i r) := mmL r
          have hB : (((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).row d.BM).get (r, c, ())
            = decide (S + c < blockEnd d init i) := mnL c
          have hv : cmk.get (r, c, ()) = ((decide (validRow d init i r) && decide (S + c < blockEnd d init i)) &&
            maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c)) := hcmk r c
          rw [hA, hB]
          rw [b2i_ne, hv]
          cases decide (validRow d init i r) <;> cases decide (S + c < blockEnd d init i) <;>
            cases maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c) <;> rfl
        · intro fm hfm
          refine hrest fm fun r c => ?_
          rw [hfm r c]
          refine lane_core hpre hi r S c _ _ (maskBit d init (bOf d i) (mOf d i * d.BM + r) (S + c)) rfl rfl ?_
          intro _
          simp only [mpLane, hcm1, ite_true]

end Claims.ExtendAttn
