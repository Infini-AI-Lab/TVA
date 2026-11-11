import Proof.Spec
import Proof.Lanes
import Proof.Facts

/-!
# The load of `q`

The `q` tile (lines 238-246): the pointers, the mask and the extent of the load, and the tile it
hands over (`QFacts`).
-/

namespace Claims.ExtendAttn

open TVA

/-- The offsets `offs_q` of the `q` load (lines 238-243), as the model builds them from `Sc`. -/
def offsQ (d : Inst) (s : Sc d) : Tensor [d.BM, d.BD] (QI d.cm) :=
  (s.cur_seq_extend_start_idx + (toQv d.cm (Tensor.scalar (s.cur_block_m * (IntV.ofNat d.BM : I32)))).item +
        toQv d.cm ((tl.arange 0 d.BM : Tensor [d.BM] I32).col d.BD)) *
      (toQv d.cm (Tensor.scalar (IntV.ofNat d.sqt : I32))).item +
    (toQv d.cm (Tensor.scalar (s.cur_head * (IntV.ofNat d.sqh : I32)))).item +
    toQv d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).row d.BM)

/-- **The pointer lane of the `q` offsets**: on an enabled lane `(r, c)` (row inside the request's
new tokens, `c < D`) the offset `offs_q` has the natural value of the element
`[qs + m*BM + r, h, c]` of `q_extend`, with no wrap in `qo_indptr`'s type. -/
theorem offsQ_lane {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s) (r : Fin d.BM) (c : Fin d.BD)
    (hr : validRow d init i r) (hc : c.val < d.D) :
    ((offsQ d s).get2 r c).toInt =
      ((qIdx d (qsOf d init (bOf d i) + (mOf d i * d.BM + r)) (hOf d i) c : Nat) : Int) := by
  obtain ⟨hb, hh, hm⟩ := prog_range hi
  have hnb := hd.nb_lt
  have hbm := bm_le hd
  have hbd := hd.bd_le
  obtain ⟨hsqt, hsqh, -⟩ := hd.strides_lt
  obtain ⟨hq1, -, -, -, -, -⟩ := hd.heads_i32
  obtain ⟨hT, -, -, -⟩ := req_sizes hpre hb
  have hTle := hd.T_le
  have hr' : mOf d i * d.BM + r < extOf d init (bOf d i) := hr
  have hTpos : 0 < d.T := by omega
  have hBM : (IntV.ofNat d.BM : I32).toInt = d.BM := IntV.I32.toInt_ofNat (by omega)
  have hmb : mOf d i * d.BM ≤ 2 ^ 16 * 128 := Nat.mul_le_mul (by omega) hbm
  have h1 : (s.cur_block_m * (IntV.ofNat d.BM : I32)).toInt = ((mOf d i * d.BM : Nat) : Int) :=
    IntV.mul_ofNat hs.blk (by omega)
  have hH : (s.cur_head * (IntV.ofNat d.sqh : I32)).toInt = ((hOf d i * d.sqh : Nat) : Int) :=
    IntV.I32.mul_nat hs.head (IntV.I32.toInt_ofNat hsqh) (by
      have : hOf d i * d.sqh ≤ (d.H - 1) * d.sqh := Nat.mul_le_mul_right _ (by omega)
      omega)
  have hx : ((tl.arange 0 d.BM : Tensor [d.BM] I32).get1 r).toInt = (r : Nat) :=
    IntV.I32.toInt_tl_arange (n := d.BM) (by omega) r
  have hy : ((tl.arange 0 d.BD : Tensor [d.BD] I32).get1 c).toInt = (c : Nat) :=
    IntV.I32.toInt_tl_arange (n := d.BD) (by omega) c
  have hfit := idx_fits hd (t := qsOf d init (bOf d i) + (mOf d i * d.BM + r)) (g := hOf d i)
    (e := c) (T' := d.T) (G := d.H) (s0 := d.sqt) (s1 := d.sqh) (by omega) hh hc hTle
    (by have := hd.H_lt; omega) hsqt hsqh (fun hcm => by
      have := (hd.cm_i32 hcm hTpos).1; omega)
  have hA : (qsOf d init (bOf d i) + mOf d i * d.BM + r) < 2 ^ (qoW d.cm - 1) :=
    lt_pow_qoW d.cm (by omega)
  have hpP : (castQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32))).toInt = ((mOf d i * d.BM : Nat) : Int) := by
    rw [castQ_toInt, h1]
  have key := qoff_val d.cm (a := s.cur_seq_extend_start_idx)
    (p := castQ d.cm (s.cur_block_m * (IntV.ofNat d.BM : I32)))
    (x := (tl.arange 0 d.BM : Tensor [d.BM] I32).get1 r) (s0 := (IntV.ofNat d.sqt : I32))
    (c := s.cur_head * (IntV.ofNat d.sqh : I32)) (y := (tl.arange 0 d.BD : Tensor [d.BD] I32).get1 c)
    hs.qs hpP hx (IntV.I32.toInt_ofNat hsqt) hH hy
    hA
    (by
      have e : (qsOf d init (bOf d i) + mOf d i * d.BM + r) * d.sqt + hOf d i * d.sqh + c =
          (qsOf d init (bOf d i) + (mOf d i * d.BM + r)) * d.sqt + hOf d i * d.sqh + c := by ring
      rw [e]; exact hfit)
  have e : (offsQ d s).get2 r c =
      (s.cur_seq_extend_start_idx + (toQv d.cm (Tensor.scalar (s.cur_block_m * (IntV.ofNat d.BM : I32)))).item +
        (toQv d.cm ((tl.arange 0 d.BM : Tensor [d.BM] I32).col d.BD)).get2 r c) *
      (toQv d.cm (Tensor.scalar (IntV.ofNat d.sqt : I32))).item +
      (toQv d.cm (Tensor.scalar (s.cur_head * (IntV.ofNat d.sqh : I32)))).item +
      (toQv d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).row d.BM)).get2 r c := rfl
  rw [e, toQv_scalar_item, toQv_scalar_item, toQv_scalar_item, toQv_col_get, toQv_row_get]
  rw [key]
  simp only [qIdx]
  push_cast
  ring_nf

/-- **The load of `q`** is safe and gives `QFacts`: the rest, `afterQ`, runs with it. -/
theorem afterCm_safe {P : Platform} (_hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    (cm : Option (I64 × Option I64))
    (hrest : ∀ q : Tensor [d.BM, d.BD] BF16, QFacts d init i q →
      Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
        (afterQ d s cm (maskM d s) q)) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      (afterCm d s cm) := by
  obtain ⟨hb, hh, hm⟩ := prog_range hi
  obtain ⟨hqv, -, -, -, -, -, -, -, -, -, -, -, hvals⟩ := id hpre
  obtain ⟨hT, -, -, -⟩ := req_sizes hpre hb
  have hbd := hd.bd_le
  have hbm := bm_le hd
  unfold afterCm
  simp only [safe_toQ]
  refine convert_zero_safe fun z hz => ?_
  refine safe_load_en d.q (en := fun l : Idx [d.BM, d.BD] => decide (validRow d init i l.1 ∧ l.2.1.val < d.D))
    (g := fun l : Idx [d.BM, d.BD] => qIdx d (qsOf d init (bOf d i) + (mOf d i * d.BM + l.1)) (hOf d i) l.2.1)
    ?_ ?_ ?_ hqv (by intro x hx; simp only [Rf, Set.mem_union]; tauto) ?_
  · intro l
    obtain ⟨r, c, ⟨⟩⟩ := l
    show ((maskM d s).get1 r && (maskD d).get1 c) = _
    have hmd : (maskD d).get1 c = decide (c.val < d.D) := by
      show IntV.lt ((tl.arange 0 d.BD : Tensor [d.BD] I32).get1 c) (IntV.ofNat d.D : I32) = _
      rw [IntV.lt_eq, IntV.I32.toInt_tl_arange (n := d.BD) (by omega) c,
        IntV.I32.toInt_ofNat (by have := hd.D_le; omega)]
      simp only [Nat.cast_lt]
    rw [maskM_lane hd hi hs r, hmd]
    simp only [Bool.decide_and]
  · intro l hl
    obtain ⟨r, c, ⟨⟩⟩ := l
    simp only [decide_eq_true_eq] at hl
    exact ptr0_add d.q (offsQ d s) (r, c, ()) (offsQ_lane hd hpre hi hs r c hl.1 hl.2)
  · intro l hl
    obtain ⟨r, c, ⟨⟩⟩ := l
    simp only [decide_eq_true_eq] at hl
    have hr : mOf d i * d.BM + r < extOf d init (bOf d i) := hl.1
    have hTpos : 0 < d.T := by omega
    have hql := hd.q_len hTpos
    have h1 : (qsOf d init (bOf d i) + (mOf d i * d.BM + r)) ≤ d.T - 1 := by omega
    have h2 : hOf d i ≤ d.H - 1 := by omega
    have h3 := Nat.mul_le_mul_right d.sqt h1
    have h4 := Nat.mul_le_mul_right d.sqh h2
    have h5 := hl.2
    simp only [qIdx]
    omega
  · intro v hv1 hv2 
    refine hrest v ?_
    intro r c
    by_cases hl : validRow d init i r ∧ c.val < d.D
    · have hr : mOf d i * d.BM + r < extOf d init (bOf d i) := hl.1
      have hv := hv1 (r, c, ()) (by simpa using hl)
      obtain ⟨hq, -⟩ := hvals (bOf d i) hb
      obtain ⟨x, hx, hfin⟩ := hq (mOf d i * d.BM + r) (hOf d i) c hr hh hl.2
      have hv' : v.get2 r c = x := by
        have : some (v.get (r, c, ())) = some x := by
          rw [← hv]; exact hx
        exact Option.some.inj this |>.symm ▸ rfl
      simp only [hl, and_self, ↓reduceIte, hv']
      refine DType.val?_bf16.mpr ⟨hfin, ?_⟩
      simp only [qv, valB, hx]
    · have hv := hv2 (r, c, ()) (by simpa using hl) z rfl
      have : v.get2 r c = FloatV.ofPat 0 := by
        show v.get (r, c, ()) = _
        rw [hv]; exact hz _
      simp only [hl, ↓reduceIte, this]
      exact DType.val?_zero_bf16

end Claims.ExtendAttn
