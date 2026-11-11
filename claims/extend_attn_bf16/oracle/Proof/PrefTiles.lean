import Proof.Spec
import Proof.Facts
import Proof.Scalars
import Proof.Qi
import Proof.RowFacts

/-!
# The tensors common to the tiles of a cached-part trip

The lanes of `mask_n` (`maskNP`) and of `final_mask` (`finalMaskP`) of the cached-part trip `n`, and the
skip test.
-/

namespace Claims.ExtendAttn

open TVA

/-- The skip test of a cached-part trip holds (only with a window). -/
def PrefSkip (d : Inst) (s : Sc d) (sn : I32) : Prop :=
  0 < d.sw ∧ (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32
    (Tensor.map b2i (finalMaskP d s (maskM d s) sn)))).eq (IntV.ofInt 0) = true

/-- The cached-part trip `n` is inside the cached-part. -/
theorem pref_trip_lt {d : Inst} {init : Mem} {b n : Nat} (hn : n < tripsPre d init b) :
    BN * n < preOf d init b := by
  unfold tripsPre BN at hn
  unfold BN
  omega

/-- `mask_n`'s lanes: key `64 n + c` is inside the cached-part. -/
theorem maskNP_lane {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s) {n : Nat}
    (hn : n < tripsPre d init (bOf d i)) {sn : I32} (hsn : sn.toInt = ((BN * n : Nat) : Int))
    (c : Fin BN) :
    (maskNP d s sn).get1 c = decide (BN * n + c < preOf d init (bOf d i)) := by
  obtain ⟨hb, -, -⟩ := prog_range hi
  have hrs := req_sizes hpre hb
  have h1 := pref_trip_lt hn
  have h2 := IntV.I32.toInt_sadd_arange (n := BN) hsn (by have := c.isLt; unfold BN at *; omega) c
  show IntV.lt ((sn + tl.arange 0 BN).get1 c) s.cur_seq_len_prefix = _
  rw [IntV.lt_eq]
  have h3 : ((sn + tl.arange 0 BN).get1 c).toInt = ((BN * n + c : Nat) : Int) := h2
  rw [h3, hs.pre]
  simp only [Nat.cast_lt]

/-- Whether lane `c` of the cached-part trip `n` is a key row `i'` attends, as a comparison. -/
theorem vP_iff {d : Inst} {init : Mem} {i r n c : Nat} :
    vP d init i r n c = true ↔ BN * n + c < preOf d init (bOf d i) ∧
      (d.sw = 0 ∨ preOf d init (bOf d i) + (mOf d i * d.BM + r) ≤ BN * n + c + d.sw) := by
  unfold vP keys attends
  simp only [decide_eq_true_eq, Finset.mem_filter, Finset.mem_range, Bool.and_eq_true,
    Bool.or_eq_true]
  constructor
  · rintro ⟨⟨-, -, h2⟩, h3⟩
    exact ⟨h3, by rcases h2 with h | h <;> simp_all⟩
  · rintro ⟨h3, h2⟩
    exact ⟨⟨by omega, Or.inl h3, by rcases h2 with h | h <;> simp_all⟩, h3⟩

/-- `final_mask`'s lanes: row `r` is inside the request and key `64 n + c` is one it attends. -/
theorem finalMaskP_lane {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s) {n : Nat}
    (hn : n < tripsPre d init (bOf d i)) {sn : I32} (hsn : sn.toInt = ((BN * n : Nat) : Int))
    (r : Fin d.BM) (c : Fin BN) :
    (finalMaskP d s (maskM d s) sn).get2 r c = (decide (validRow d init i r) && vP d init i r n c) := by
  obtain ⟨hb, -, hm⟩ := prog_range hi
  have hrs := req_sizes hpre hb
  have h1 := pref_trip_lt hn
  have hbm := bm_le hd
  have hsw := hd.sw_le
  have hnb := hd.nb_lt
  have hmb : mOf d i * d.BM ≤ 2 ^ 16 * 128 := Nat.mul_le_mul (by omega) hbm
  have e1 : ((maskM d s).col BN).get2 r c = decide (validRow d init i r) := by
    rw [Tensor.get2_col]; exact maskM_lane hd hi hs r
  have e2 : (((maskNP d s sn).row d.BM)).get2 r c = decide (BN * n + c < preOf d init (bOf d i)) := by
    rw [Tensor.get2_row]; exact maskNP_lane hd hpre hi hs hn hsn c
  unfold finalMaskP
  by_cases hsw0 : 0 < d.sw
  · simp only [hsw0, ite_true]
    have hB : (IntV.ofNat d.BM : I32).toInt = d.BM := IntV.I32.toInt_ofNat (by omega)
    have hcb : (s.cur_block_m * (IntV.ofNat d.BM : I32)).toInt = ((mOf d i * d.BM : Nat) : Int) :=
      IntV.mul_ofNat hs.blk (by omega)
    have hA1 := IntV.add_nat hs.pre hcb (by omega)
    have hA2 := IntV.add_nat hA1 (IntV.I32.toInt_tl_arange (n := d.BM) (by omega) r) (by omega)
    have hB1 : (sn + (tl.arange 0 BN : Tensor [BN] I32).get1 c).toInt = ((BN * n + c : Nat) : Int) :=
      IntV.I32.toInt_sadd_arange (n := BN) hsn (by have := c.isLt; unfold BN at *; omega) c
    have hB2 := IntV.add_nat hB1 (IntV.I32.toInt_ofNat (n := d.sw) (by omega)) (by
      have := c.isLt; unfold BN at *; omega)
    have e3 : (Tensor.le (s.cur_seq_len_prefix + s.cur_block_m * (IntV.ofNat d.BM : I32) +
        (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)
        (sn + (tl.arange 0 BN : Tensor [BN] I32).row d.BM + (IntV.ofNat d.sw : I32))).get2 r c =
        decide (preOf d init (bOf d i) + (mOf d i * d.BM + r) ≤ BN * n + c + d.sw) := by
      show IntV.le (s.cur_seq_len_prefix + s.cur_block_m * (IntV.ofNat d.BM : I32) +
        (tl.arange 0 d.BM : Tensor [d.BM] I32).get1 r)
        (sn + (tl.arange 0 BN : Tensor [BN] I32).get1 c + (IntV.ofNat d.sw : I32)) = _
      rw [IntV.le_eq, hA2, hB2]
      simp only [Nat.cast_le, Nat.add_assoc]
    show ((((maskM d s).col BN).get2 r c && ((maskNP d s sn).row d.BM).get2 r c) &&
      (Tensor.le (s.cur_seq_len_prefix + s.cur_block_m * (IntV.ofNat d.BM : I32) +
        (tl.arange 0 d.BM : Tensor [d.BM] I32).col BN)
        (sn + (tl.arange 0 BN : Tensor [BN] I32).row d.BM + (IntV.ofNat d.sw : I32))).get2 r c) = _
    rw [e1, e2, e3]
    rw [Bool.and_assoc]
    congr 1
    refine Bool.eq_iff_iff.mpr ?_
    simp only [Bool.and_eq_true, decide_eq_true_eq, vP_iff]
    omega
  · simp only [hsw0, ite_false]
    show (((maskM d s).col BN).get2 r c && ((maskNP d s sn).row d.BM).get2 r c) = _
    rw [e1, e2]
    have : d.sw = 0 := by omega
    congr 1
    refine Bool.eq_iff_iff.mpr ?_
    simp only [decide_eq_true_eq, vP_iff]
    omega

/-- A fold of `max` is at least its start and each element. -/
theorem foldl_max_ge {w : Nat} (l : List (IntV true w)) (a : IntV true w) :
    a.toInt ≤ (l.foldl IntV.max a).toInt ∧ ∀ x ∈ l, x.toInt ≤ (l.foldl IntV.max a).toInt := by
  induction l generalizing a with
  | nil => exact ⟨le_rfl, fun x hx => absurd hx (by simp)⟩
  | cons y l ih =>
    obtain ⟨h1, h2⟩ := ih (a.max y)
    rw [IntV.toInt_max] at h1
    refine ⟨le_trans (le_max_left _ _) h1, fun x hx => ?_⟩
    rcases List.mem_cons.mp hx with rfl | hx
    · exact le_trans (le_max_right _ _) h1
    · exact h2 x hx

/-- `b2i b` has integer value 1 if `b` holds and 0 otherwise. -/
theorem b2i_toInt (b : Bool) : (b2i b).toInt = if b then 1 else 0 := by
  cases b
  · exact IntV.toInt_ofInt_zero
  · exact IntV.toInt_ofInt_one (by norm_num)

/-- **The skip test is true only when no lane of `final_mask` is on.** -/
theorem pref_skip_lanes {d : Inst} {s : Sc d} {sn : I32} (hsk : PrefSkip d s sn) (r : Fin d.BM)
    (c : Fin BN) : (finalMaskP d s (maskM d s) sn).get2 r c = false := by
  by_contra hne
  have ht : (finalMaskP d s (maskM d s) sn).get2 r c = true := by simpa using hne
  have h0 := hsk.2
  rw [IntV.eq_eq, decide_eq_true_eq, IntV.toInt_ofInt_zero] at h0
  set F : Tensor [d.BM, BN] I32 := Tensor.map b2i (finalMaskP d s (maskM d s) sn) with hF
  have hFc : (F.get2 r c).toInt = 1 := by
    show (b2i ((finalMaskP d s (maskM d s) sn).get2 r c)).toInt = 1
    rw [ht, b2i_toInt]; rfl
  have hrow : 1 ≤ ((Tensor.foldRows IntV.max imin32 F).get1 r).toInt := by
    show 1 ≤ ((List.finRange BN).foldl (fun acc j => IntV.max acc (F.get2 r j)) imin32).toInt
    have := (foldl_max_ge ((List.finRange BN).map (F.get2 r)) imin32).2 (F.get2 r c)
      (List.mem_map.mpr ⟨c, List.mem_finRange c, rfl⟩)
    rw [List.foldl_map] at this
    rw [hFc] at this
    exact this
  have hall := (foldl_max_ge (Tensor.foldRows IntV.max imin32 F).toList imin32).2
    ((Tensor.foldRows IntV.max imin32 F).get1 r)
    (List.mem_map.mpr ⟨(r, ()), Idx.mem_all _ _, rfl⟩)
  have : 1 ≤ (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 F)).toInt :=
    le_trans hrow hall
  omega

/-- **A skipped tile keeps the invariant**: no valid row sees a new key, so each row's state is
the same `RowFl` of the same keys, with more trips. -/
theorem pref_skip_inv {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s) {n : Nat}
    (hn : n < tripsPre d init (bOf d i)) {sn : I32} (hsn : sn.toInt = ((BN * n : Nat) : Int))
    (hsk : PrefSkip d s sn) {st : LoopSt d}
    (hI : LoopInvP d init i n st init) : LoopInvP d init i (n + 1) st init := by
  refine ⟨hI.1, fun r hr => ?_⟩
  have hold := hI.2 r hr
  obtain ⟨hX0, hX, -, hcard, -, -, hT, -⟩ := row_facts hd hpre hi hr
  have hbd := hd.bd_le
  have hBD' : (d.BD : ℝ) ≤ 256 := by exact_mod_cast hbd
  have hd1 : dlt d.BD (rowX d init i r) ≤ 1 / 2 := by
    unfold dlt
    have h1 : ((d.BD : ℝ) + 8) * (1 / 2 ^ 22) * rowX d init i r ≤ (256 + 8) * (1 / 2 ^ 22) * 2 ^ 12 := by
      apply mul_le_mul (mul_le_mul_of_nonneg_right (by linarith) (by norm_num)) hX hX0 (by norm_num)
    norm_num at h1 ⊢
    linarith
  have hseen : seenP d init (bOf d i) (mOf d i * d.BM + r) (n + 1) =
      seenP d init (bOf d i) (mOf d i * d.BM + r) n := by
    ext j
    simp only [seenP, Finset.mem_filter]
    constructor
    · rintro ⟨hj, hlt⟩
      refine ⟨hj, ?_⟩
      by_contra hnot
      have hc : j - BN * n < BN := by unfold BN at *; omega
      have hl := finalMaskP_lane hd hpre hi hs hn hsn r ⟨j - BN * n, hc⟩
      rw [pref_skip_lanes hsk r ⟨j - BN * n, hc⟩] at hl
      have hv : vP d init i r n (j - BN * n) = true := by
        unfold vP
        have : BN * n + (j - BN * n) = j := by omega
        simp only [this, decide_eq_true_eq]
        exact ⟨hj, by omega⟩
      rw [hv] at hl
      simp [hr] at hl
    · rintro ⟨hj, hlt⟩
      exact ⟨hj, lt_of_lt_of_le hlt (min_le_min (Nat.mul_le_mul_left _ (Nat.le_succ n)) le_rfl)⟩
  rw [hseen]
  have hγ : g2 + (n : ℝ) * g1p ≤ g2 + ((n + 1 : Nat) : ℝ) * g1p := by
    have : (0 : ℝ) ≤ g1p := by unfold g1p; norm_num
    push_cast
    nlinarith
  have hcard' : (seenP d init (bOf d i) (mOf d i * d.BM + r) n).card ≤ nmax :=
    le_trans (Finset.card_le_card (Finset.filter_subset _ _)) hcard
  exact RowFl.mono (by norm_num at hX ⊢; linarith) hcard' (Nat.le_succ n)
    (by unfold tmax; omega) hγ hold

/-- Lane `c` of the pointers of the `kv_indices` load. -/
theorem kvi_ptr_lane {d : Inst} {init : Mem} {i : Nat} {s : Sc d} (hs : ScOk d init i s) {n : Nat}
    {sn : I32} (hsn : sn.toInt = ((BN * n : Nat) : Int)) (c : Fin BN) :
    (d.kvi.ptr 0 + s.cur_seq_kv_start_idx + sn + tl.arange 0 BN).get1 c =
      d.kvi.ptr ((ksOf d init (bOf d i) + (BN * n + c) : Nat) : Int) := by
  have hc : ((tl.arange 0 BN : Tensor [BN] I32).get1 c).toInt = c := 
    IntV.I32.toInt_tl_arange (by decide) c
  show (d.kvi.ptr 0 + s.cur_seq_kv_start_idx + sn).offset ((tl.arange 0 BN : Tensor [BN] I32).get1 c).toInt = _
  rw [hc, Ptr.add_int, Ptr.add_int, hs.ks, hsn]
  simp only [Buf.ptr_offset]
  congr 1; push_cast; ring

/-- The key/value index buffer lies in the readable region. -/
theorem rf_kvi {d : Inst} : d.kvi.bytes ⊆ Rf d := by intro x hx; simp [Rf, hx]
/-- The key block-table buffer lies in the readable region. -/
theorem rf_kb {d : Inst} : d.kb.bytes ⊆ Rf d := by intro x hx; simp [Rf, hx]
/-- The value block-table buffer lies in the readable region. -/
theorem rf_vb {d : Inst} : d.vb.bytes ⊆ Rf d := by intro x hx; simp [Rf, hx]

/-- The slot of a cached-part token: `kv_indices` holds it, a slot of the cache. -/
theorem slot_read {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.batch)
    {j : Nat} (hj : j < preOf d init b) :
    ∃ sl : I64, init.read .i64 (d.kvi.ptr ((ksOf d init b + j : Nat) : Int)).addr = some sl ∧
      0 ≤ sl.toInt ∧ sl.toInt < d.S ∧ (slotOf d init b j : Int) = sl.toInt := by
  obtain ⟨q0, q1, k0, k1, -, -, -, -, -, -, -, -, -, -, -, -, hs⟩ :=
    hpre.2.2.2.2.2.2.2.2.2.2.1 b hb
  obtain ⟨sl, hr, h0, h1⟩ := hs j hj
  refine ⟨sl, hr, h0, h1, ?_⟩
  unfold slotOf val64
  rw [hr]
  exact Int.toNat_of_nonneg h0

/-- The loaded `kv_indices` tile has the claim's slots. -/
theorem kvloc_facts {d : Inst} {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {n : Nat} {kv : Tensor [BN] I64}
    (h1 : ∀ c : Idx [BN], decide (BN * n + c.1.val < preOf d init (bOf d i)) = true →
      init.read .i64 (d.kvi.ptr ((ksOf d init (bOf d i) + (BN * n + c.1.val) : Nat) : Int)).addr =
        some (kv.get c))
    (h2 : ∀ c : Idx [BN], decide (BN * n + c.1.val < preOf d init (bOf d i)) = false →
      ∀ o, some ((tl.full [BN] (IntV.ofInt 0 : I32)).castInt : Tensor [BN] I64) = some o →
        kv.get c = o.get c) :
    KvLocFacts d init i n kv := by
  intro c
  by_cases hc : BN * n + c.val < preOf d init (bOf d i)
  · obtain ⟨sl, hr, -, -, he⟩ := slot_read hpre (prog_range hi).1 hc
    have := h1 (c, ()) (by simpa using hc)
    dsimp only at this
    rw [hr] at this
    have e : sl = kv.get1 c := Option.some.inj this
    simp only [hc, ite_true]
    rw [he, e]
  · simp only [hc, ite_false]
    have := h2 (c, ()) (by simpa using hc) _ rfl
    show (kv.get (c, ())).toInt = 0
    rw [this]
    show (IntV.cast (IntV.ofInt 0 : I32) : I64).toInt = 0
    rw [IntV.toInt_cast _ (by decide)]
    exact IntV.toInt_ofInt_zero

/-- The I64 offset `slot * st + G + e` of a gather, when it does not wrap. -/
theorem gather_val {kv a b c : I64} {slot G e st : Nat} (hkv : kv.toInt = slot)
    (ha : a.toInt = st) (hb : b.toInt = G) (hc : c.toInt = e)
    (h : slot * st + G + e < 2 ^ 63) :
    (kv * a + b + c).toInt = ((slot * st + G + e : Nat) : Int) := by
  have e1 := IntV.mul_nat hkv ha (by
    have : slot * st ≤ slot * st + G + e := by omega
    exact lt_of_le_of_lt this h)
  have e2 := IntV.add_nat e1 hb (by show slot * st + G < 2 ^ 63; omega)
  exact IntV.add_nat e2 hc h

/-- The widened `st` and `kv_head * sh` (`int32` to `int64`). -/
theorem cast_ofNat_toInt {st : Nat} (h : st < 2 ^ 31) :
    (IntV.cast (IntV.ofNat st : I32) : I64).toInt = st := by
  rw [IntV.toInt_cast _ (by decide)]; exact IntV.toInt_ofNat h

/-- Casting `kvhv * sh` to 64 bits keeps the product `g * sh` when it is below 2^31. -/
theorem cast_kvmul_toInt {kvhv : I32} {g sh : Nat} (hg : kvhv.toInt = g) (h : g * sh < 2 ^ 31) :
    (IntV.cast (kvhv * IntV.ofNat sh : I32) : I64).toInt = (g * sh : Nat) := by
  rw [IntV.toInt_cast _ (by decide)]; exact IntV.mul_ofNat hg h

/-- `mask_d`'s lanes. -/
theorem maskD_lane' {d : Inst} (hd : d.ok) (e : Fin d.BD) :
    ((tl.arange 0 d.BD : Tensor [d.BD] I32).ltS (IntV.ofNat d.D : I32)).get1 e =
      decide (e.val < d.D) := by
  have hbd := hd.bd_le
  have hDl := hd.D_le
  show IntV.lt ((tl.arange 0 d.BD : Tensor [d.BD] I32).get1 e) (IntV.ofNat d.D : I32) = _
  rw [IntV.lt_eq, IntV.I32.toInt_tl_arange (n := d.BD) (by omega) e, IntV.toInt_ofNat (by norm_num; omega)]
  simp only [Nat.cast_lt]

/-- A valid `bfloat16` buffer has at most `2^63` elements. -/
theorem buf_len_le {b : Buf .bf16} {m : Mem} (hv : b.Valid m) : b.len ≤ 2 ^ 63 := by
  have h := hv.2.1
  have h0 := hv.1
  have : (DType.bf16.bytes : Int) = 2 := rfl
  rw [this] at h
  have h2 : (b.len : Int) * 2 ≤ 2 ^ 64 := by omega
  have : (b.len : Int) ≤ 2 ^ 63 := by omega
  exact_mod_cast this

/-- The key/value head index `kvh d h` is below the number of key/value heads. -/
theorem kvh_lt' {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.H) : kvh d h < d.HKV := by
  have hpos := hd.kv_pos
  have hkv : 0 < kvg d := Nat.div_pos (Nat.le_of_dvd (by omega) hd.kv_dvd) hpos
  unfold kvh
  rw [Nat.div_lt_iff_lt_mul hkv]
  have : d.HKV * kvg d = d.H := Nat.mul_div_cancel' hd.kv_dvd
  omega

/-- The element `slot * st + g * sh + e` of a buffer whose last element the claim bounds. -/
theorem gather_idx_lt {S HKV D len st sh slot g e : Nat} (hS : slot < S) (hg : g < HKV) (he : e < D)
    (hlen : (S - 1) * st + (HKV - 1) * sh + D ≤ len) :
    slot * st + g * sh + e < len := by
  have h1 : slot * st ≤ (S - 1) * st := Nat.mul_le_mul_right _ (by omega)
  have h2 : g * sh ≤ (HKV - 1) * sh := Nat.mul_le_mul_right _ (by omega)
  omega

/-- **The offset of a gather lane**: `slot * st + kv_head * sh + e` in `int64`, no wrap. -/
theorem gather_lane_toInt {d : Inst} (hd : d.ok) {init : Mem} {i : Nat} (hi : i < (launch d).size)
    {s : Sc d} (hs : ScOk d init i s) {kv : I64} {slot : Nat} (hkv : kv.toInt = slot)
    (hslot : slot < d.S) {st sh len : Nat} (hst : st < 2 ^ 31) (_hsh : sh < 2 ^ 31)
    (hgsh : (d.HKV - 1) * sh < 2 ^ 31)
    (hlen : (d.S - 1) * st + (d.HKV - 1) * sh + d.D ≤ len) (hlen63 : len ≤ 2 ^ 63) {ev : I32}
    {e : Nat} (he : e < d.D) (hev : ev.toInt = e) :
    (kv * (IntV.cast (IntV.ofNat st : I32) : I64) +
      (IntV.cast (s.cur_kv_head * (IntV.ofNat sh : I32)) : I64) + (IntV.cast ev : I64)).toInt =
      ((slot * st + kvh d (hOf d i) * sh + e : Nat) : Int) := by
  have hg := kvh_lt' hd (prog_range hi).2.1
  have hgs : kvh d (hOf d i) * sh < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (by omega)) hgsh
  refine gather_val hkv (cast_ofNat_toInt hst) (cast_kvmul_toInt hs.kvh hgs) ?_ ?_
  · rw [IntV.toInt_cast _ (by decide)]; exact hev
  · exact lt_of_lt_of_le (gather_idx_lt hslot hg he hlen) hlen63

/-- A bf16 read from a finite entry yields the value `valB` of that entry. -/
theorem val?_of_read {m : Mem} {b : Buf .bf16} {idx : Nat} {x : BF16}
    (hr : m.read .bf16 (b.ptr (idx : Int)).addr = some x) (hf : finB m b idx) :
    DType.val? .bf16 x = some (valB m b idx) := by
  obtain ⟨y, hy, hfin⟩ := hf
  unfold rdB at hy
  rw [hr] at hy
  have : x = y := Option.some.inj hy
  subst this
  rw [DType.val?_bf16]
  refine ⟨hfin, ?_⟩
  unfold valB rdB
  rw [hr]

/-- The bf16 pattern 0 reads back as the rational 0. -/
theorem val?_zero_bf16 : DType.val? .bf16 (FloatV.ofPat 0 : BF16) = some 0 := by
  rw [DType.val?_bf16]
  exact ⟨FloatV.isFinite_ofPat_zero Format.bfloat16_regular, FloatV.toRat_ofPat_zero⟩

/-- The `k` tile of the load of trip `n`. -/
theorem kFacts_of_load {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size)
    {n : Nat} {k ok : Tensor [d.BD, BN] BF16}
    (h1 : ∀ x : Idx [d.BD, BN],
      decide (BN * n + x.2.1.val < preOf d init (bOf d i) ∧ x.1.val < d.D) = true →
      init.read .bf16 (d.kb.ptr ((kbIdx d (slotOf d init (bOf d i) (BN * n + x.2.1.val))
        (kvh d (hOf d i)) x.1.val : Nat) : Int)).addr = some (k.get x))
    (h2 : ∀ x : Idx [d.BD, BN],
      decide (BN * n + x.2.1.val < preOf d init (bOf d i) ∧ x.1.val < d.D) = false →
      ∀ o, some ok = some o → k.get x = o.get x)
    (hok : ∀ ix, ok.get ix = FloatV.ofPat 0) : KFactsP d init i n k := by
  intro e c
  by_cases hc : BN * n + c.val < preOf d init (bOf d i) ∧ e.val < d.D
  · simp only [hc]
    have hr := h1 (e, c, ()) (by simpa using hc)
    have hg := kvh_lt' hd (prog_range hi).2.1
    have hvals := (hpre.2.2.2.2.2.2.2.2.2.2.2.2 (bOf d i) (prog_range hi).1).2.2.1
    have hkb := (hvals (BN * n + c.val) (kvh d (hOf d i)) e.val hc.1 hg hc.2).1
    show DType.val? .bf16 (k.get (e, c, ())) = _
    unfold keyK
    simp only [hc.1, ite_true]
    exact val?_of_read hr hkb
  · simp only [hc, ite_false]
    have := h2 (e, c, ()) (by simpa using hc) ok rfl
    show DType.val? .bf16 (k.get (e, c, ())) = _
    rw [this]
    exact (congrArg (DType.val? .bf16) (hok (e, c, ()))).trans val?_zero_bf16

/-- `tl.cvt` of `bfloat16` to `bfloat16` is exact on the finite lanes of `q`. -/
theorem qFacts_cvt {d : Inst} {init : Mem} {i : Nat} {q q16 : Tensor [d.BM, d.BD] BF16}
    (hq : QFacts d init i q)
    (h : ∀ x, FloatV.NaNEq (FloatV.cvtWith .rn (q.get x)) (q16.get x)) : QFacts d init i q16 := by
  intro r e
  have h1 := hq r e
  rw [DType.val?_bf16] at h1 ⊢
  obtain ⟨hf, hv⟩ := h1
  have hw : Format.Wider Format.bfloat16 Format.bfloat16 := ⟨le_rfl, le_rfl, le_rfl⟩
  obtain ⟨hf2, hr2, -⟩ := FloatV.cvt_of_wider Format.bfloat16_regular Format.bfloat16_regular hw hf
  have hn : (FloatV.cvtWith .rn (q.get2 r e) : BF16).isNaN = false := by
    rw [FloatV.cvtWith_rn]
    exact (FloatV.isFinite_iff.mp hf2).1
  have := FloatV.NaNEq.eq (h (r, e, ())) hn
  have e1 : q16.get2 r e = FloatV.cvt (q.get2 r e) := by
    show q16.get (r, e, ()) = _
    rw [this, FloatV.cvtWith_rn]
    rfl
  rw [e1]
  exact ⟨hf2, hr2.trans hv⟩

/-- The `v` tile of the load of trip `n`. -/
theorem vFacts_of_load {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {n : Nat} {v ok : Tensor [BN, d.BD] BF16}
    (h1 : ∀ x : Idx [BN, d.BD],
      decide (BN * n + x.1.val < preOf d init (bOf d i) ∧ x.2.1.val < d.D) = true →
      init.read .bf16 (d.vb.ptr ((vbIdx d (slotOf d init (bOf d i) (BN * n + x.1.val))
        (kvh d (hOf d i)) x.2.1.val : Nat) : Int)).addr = some (v.get x))
    (h2 : ∀ x : Idx [BN, d.BD],
      decide (BN * n + x.1.val < preOf d init (bOf d i) ∧ x.2.1.val < d.D) = false →
      ∀ o, some ok = some o → v.get x = o.get x)
    (hok : ∀ ix, ok.get ix = FloatV.ofPat 0) : VFactsP d init i n v := by
  intro c e
  by_cases hc : BN * n + c.val < preOf d init (bOf d i) ∧ e.val < d.D
  · simp only [hc]
    have hr := h1 (c, e, ()) (by simpa using hc)
    have hg := kvh_lt' hd (prog_range hi).2.1
    have hvals := (hpre.2.2.2.2.2.2.2.2.2.2.2.2 (bOf d i) (prog_range hi).1).2.2.1
    have hvb := (hvals (BN * n + c.val) (kvh d (hOf d i)) e.val hc.1 hg hc.2).2.1
    show DType.val? .bf16 (v.get (c, e, ())) = _
    unfold keyV
    simp only [hc.1, ite_true]
    exact val?_of_read hr hvb
  · simp only [hc, ite_false]
    have := h2 (c, e, ()) (by simpa using hc) ok rfl
    show DType.val? .bf16 (v.get (c, e, ())) = _
    rw [this]
    exact (congrArg (DType.val? .bf16) (hok (c, e, ()))).trans val?_zero_bf16

end Claims.ExtendAttn
