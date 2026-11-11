import Proof.ExtMask
import Proof.ExtGlue
import Proof.RowFacts
import Proof.TileQ

/-!
# One new-token trip: the tile skip, the loads and the update

`ext_rest_safe` walks `extRest`: the tile skip (`SKIP_TILE`), the loads of `k` and `v`, the score
`tl.dot`, the online-softmax update and `tl.dot_add`, given the lanes of the `final_mask`.
-/
namespace Claims.ExtendAttn
open TVA

/-- The integer value of the 32-bit max is the max of the integer values. -/
theorem imax_toInt (a b : I32) : (IntV.max a b).toInt = max a.toInt b.toInt := by
  unfold IntV.max
  rw [apply_ite IntV.toInt, IntV.lt_eq]
  by_cases h : a.toInt < b.toInt
  · simp [h, max_eq_right h.le]
  · simp [h, max_eq_left (not_lt.mp h)]

/-- A left fold of 32-bit max is at least its start and each term, and equals the start or one of the terms. -/
theorem fold_imax {α : Type} (f : α → I32) : ∀ (L : List α) (a : I32),
    a.toInt ≤ (L.foldl (fun x j => IntV.max x (f j)) a).toInt ∧
    (∀ j ∈ L, (f j).toInt ≤ (L.foldl (fun x j => IntV.max x (f j)) a).toInt) ∧
    (L.foldl (fun x j => IntV.max x (f j)) a = a ∨
      ∃ j ∈ L, L.foldl (fun x j => IntV.max x (f j)) a = f j)
  | [], a => by simp
  | j :: L, a => by
    obtain ⟨h1, h2, h3⟩ := fold_imax f L (IntV.max a (f j))
    have hm := imax_toInt a (f j)
    have hc : IntV.max a (f j) = a ∨ IntV.max a (f j) = f j := by
      unfold IntV.max; split_ifs <;> simp
    simp only [List.foldl_cons]
    refine ⟨le_trans (by rw [hm]; exact le_max_left _ _) h1, ?_, ?_⟩
    · intro x hx
      rcases List.mem_cons.mp hx with rfl | hx
      · exact le_trans (by rw [hm]; exact le_max_right _ _) h1
      · exact h2 x hx
    · rcases h3 with h | ⟨x, hx, h⟩
      · rcases hc with hc | hc
        · exact Or.inl (h.trans hc)
        · exact Or.inr ⟨j, List.mem_cons_self, h.trans hc⟩
      · exact Or.inr ⟨x, List.mem_cons_of_mem _ hx, h⟩


/-- `b2i b` has integer value 1 if `b` holds and 0 otherwise. -/
theorem b2i_toInt (b : Bool) : (b2i b).toInt = if b then 1 else 0 := by
  cases b <;> simp [b2i, IntV.I32.toInt_ofInt]

/-- `imin32` has integer value -2^31. -/
theorem imin32_toInt : imin32.toInt = -(2 ^ 31) := by
  unfold imin32; rw [IntV.I32.toInt_ofInt (by norm_num)]

/-- A max-fold over a nonempty list of 0/1 values, started from `imin32`, is 1 iff some term is 1, else 0. -/
theorem fold_01 {α : Type} (f : α → I32) (hf : ∀ j, (f j).toInt = 0 ∨ (f j).toInt = 1)
    (L : List α) (hL : L ≠ []) :
    (L.foldl (fun x j => IntV.max x (f j)) imin32).toInt = if ∃ j ∈ L, (f j).toInt = 1 then 1 else 0 := by
  obtain ⟨h1, h2, h3⟩ := fold_imax f L imin32
  have hmin := imin32_toInt
  split_ifs with hex
  · obtain ⟨j, hj, hj1⟩ := hex
    have := h2 j hj
    rw [hj1] at this
    rcases h3 with h | ⟨j', -, h⟩
    · rw [h, hmin] at this; norm_num at this
    · rw [h] at this ⊢
      rcases hf j' with h' | h' <;> omega
  · obtain ⟨j0, hj0⟩ := List.exists_mem_of_ne_nil L hL
    have := h2 j0 hj0
    rcases h3 with h | ⟨j', hj', h⟩
    · rw [h, hmin] at this
      rcases hf j0 with h' | h' <;> omega
    · rw [h]
      rcases hf j' with h' | h'
      · exact h'
      · exact absurd ⟨j', hj', h'⟩ hex

/-- The tile skip's value: `max` of the mask is `0` exactly when no lane is on. -/
theorem skip_iff {M : Nat} (hM : 0 < M) (FM : Tensor [M, BN] Bool) :
    IntV.eq (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 (FM.map b2i)))
      (IntV.ofInt 0) = true ↔ ∀ r c, FM.get2 r c = false := by
  have hb : ∀ b : Bool, (b2i b).toInt = 0 ∨ (b2i b).toInt = 1 := by
    intro b; rw [b2i_toInt]; split_ifs <;> simp
  have hrow : ∀ r : Fin M, ((Tensor.foldRows IntV.max imin32 (FM.map b2i)).get1 r).toInt =
      if ∃ c, FM.get2 r c = true then 1 else 0 := by
    intro r
    have hg : (Tensor.foldRows IntV.max imin32 (FM.map b2i)).get1 r =
        (List.finRange BN).foldl (fun x j => IntV.max x (b2i (FM.get2 r j))) imin32 := rfl
    rw [hg, fold_01 (fun j : Fin BN => b2i (FM.get2 r j)) (fun j => hb _) _
      (List.ne_nil_of_mem (List.mem_finRange ⟨0, by norm_num [BN]⟩))]
    simp only [b2i_toInt, List.mem_finRange, true_and]
    congr 1
    simp
  set rows := Tensor.foldRows IntV.max imin32 (FM.map b2i) with hrows
  have hg : Tensor.foldAll IntV.max imin32 rows =
      (Idx.all [M]).foldl (fun x j => IntV.max x (rows.get j)) imin32 := by
    simp [Tensor.foldAll, Tensor.toList, List.foldl_map]
  have hrb : ∀ j : Idx [M], (rows.get j).toInt = 0 ∨ (rows.get j).toInt = 1 := by
    rintro ⟨r, ⟨⟩⟩
    have := hrow r
    simp only [Tensor.get1] at this
    rw [this]; split_ifs <;> simp
  have hA := fold_01 (fun j : Idx [M] => rows.get j) hrb (Idx.all [M])
    (List.ne_nil_of_mem (Idx.mem_all [M] ((⟨0, hM⟩ : Fin M), ())))
  rw [← hg] at hA
  rw [IntV.eq_eq, IntV.toInt_ofInt_zero, hA]
  simp only [decide_eq_true_eq]
  constructor
  · intro h r c
    by_contra hc
    have hc' : FM.get2 r c = true := by simpa using hc
    have : ∃ j ∈ Idx.all [M], (rows.get j).toInt = 1 := by
      refine ⟨(r, ()), Idx.mem_all _ _, ?_⟩
      have := hrow r
      simp only [Tensor.get1] at this
      rw [this]
      simp only [show (∃ c, FM.get2 r c = true) from ⟨c, hc'⟩, ite_true]
    simp only [this, ite_true] at h
    exact absurd h (by norm_num)
  · intro h
    have hn : ¬ (∃ j ∈ Idx.all [M], (rows.get j).toInt = 1) := by
      rintro ⟨⟨r, ⟨⟩⟩, -, hj⟩
      have := hrow r
      simp only [Tensor.get1] at this
      rw [this] at hj
      by_cases hh : ∃ c, FM.get2 r c = true
      · obtain ⟨c, hc⟩ := hh
        rw [h r c] at hc; exact absurd hc (by simp)
      · simp only [hh, ite_false] at hj
        norm_num at hj
    simp only [hn, ite_false]

/-- The key/value head index `kvh d h` is below the number of key/value heads. -/
theorem kvh_lt' {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.H) : kvh d h < d.HKV := by
  have hpos := hd.kv_pos
  have hkv : 0 < kvg d := Nat.div_pos (Nat.le_of_dvd (by omega) hd.kv_dvd) hpos
  unfold kvh
  rw [Nat.div_lt_iff_lt_mul hkv]
  have : d.HKV * kvg d = d.H := Nat.mul_div_cancel' hd.kv_dvd
  omega

/-- The key buffer lies in the readable region. -/
theorem k_sub_Rf (d : Inst) : d.k.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf, Set.mem_union]; tauto

/-- The value buffer lies in the readable region. -/
theorem v_sub_Rf (d : Inst) : d.v.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf, Set.mem_union]; tauto

/-- The element `[qs + j, g, e]` of `k_extend` is inside its buffer. -/
theorem k_idx_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {j e : Nat} (hj : j < extOf d init (bOf d i)) (he : e < d.D) :
    kIdx d (qsOf d init (bOf d i) + j) (kvh d (hOf d i)) e < d.k.len := by
  obtain ⟨hb, hh, -⟩ := prog_range hi
  have hsz := (req_sizes hpre hb).1
  have hT : 0 < d.T := by omega
  have hl := hd.k_len hT
  have hg := kvh_lt' hd hh
  have h1 : (qsOf d init (bOf d i) + j) * d.skt ≤ (d.T - 1) * d.skt := Nat.mul_le_mul_right _ (by omega)
  have h2 : kvh d (hOf d i) * d.skh ≤ (d.HKV - 1) * d.skh := Nat.mul_le_mul_right _ (by omega)
  unfold kIdx
  omega

/-- The value-buffer index of any extend position and in-range dimension is within the buffer length. -/
theorem v_idx_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {j e : Nat} (hj : j < extOf d init (bOf d i)) (he : e < d.D) :
    vIdx d (qsOf d init (bOf d i) + j) (kvh d (hOf d i)) e < d.v.len := by
  obtain ⟨hb, hh, -⟩ := prog_range hi
  have hsz := (req_sizes hpre hb).1
  have hT : 0 < d.T := by omega
  have hl := hd.v_len hT
  have hg := kvh_lt' hd hh
  have h1 : (qsOf d init (bOf d i) + j) * d.svt ≤ (d.T - 1) * d.svt := Nat.mul_le_mul_right _ (by omega)
  have h2 : kvh d (hOf d i) * d.svh ≤ (d.HKV - 1) * d.svh := Nat.mul_le_mul_right _ (by omega)
  unfold vIdx
  omega


/-- **The load of the new tokens' keys** gives `KFactsE`. -/
theorem k_load_safe {P : Platform} {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {S : Nat}
    {Q : LoopSt d → Mem → Prop} (ptrs : Tensor [d.BD, BN] (Ptr .bf16)) (mask : Tensor [d.BD, BN] Bool)
    (hmask : ∀ (e : Fin d.BD) (c : Fin BN), mask.get2 e c =
      (decide (S + c < blockEnd d init i) && decide (e.val < d.D)))
    (hptr : ∀ (e : Fin d.BD) (c : Fin BN), S + c < blockEnd d init i → e.val < d.D →
      ptrs.get2 e c = d.k.ptr (kIdx d (qsOf d init (bOf d i) + (S + c)) (kvh d (hOf d i)) e : Nat))
    {K : Tensor [d.BD, BN] BF16 → Prog (LoopSt d)}
    (hK : ∀ k : Tensor [d.BD, BN] BF16, KFactsE d init i S (blockEnd d init i) k →
      Safe ((launch d).env P) (Rf d) ∅ Q init (K k)) :
    Safe ((launch d).env P) (Rf d) ∅ Q init
      (tl.convert .f32 .bf16 (tl.full [d.BD, BN] zf32) >>= fun other =>
        tl.load ptrs (some mask) (some other) >>= K) := by
  obtain ⟨hb, hh, -⟩ := prog_range hi
  have hext := blockEnd_le_ext d init i
  refine convert_zero_safe fun ok hok => ?_
  refine safe_load_en d.k (fun ix : Idx [d.BD, BN] => decide (S + ix.2.1 < blockEnd d init i) &&
      decide (ix.1.val < d.D))
    (fun ix : Idx [d.BD, BN] => kIdx d (qsOf d init (bOf d i) + (S + ix.2.1)) (kvh d (hOf d i)) ix.1)
    ?_ ?_ ?_ hpre.2.1 (k_sub_Rf d) fun k hk1 hk2 => hK k fun e c => ?_
  · rintro ⟨e, c, ⟨⟩⟩
    exact hmask e c
  · rintro ⟨e, c, ⟨⟩⟩ hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact hptr e c hen.1 hen.2
  · rintro ⟨e, c, ⟨⟩⟩ hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact k_idx_lt hd hpre hi (j := S + c) (by omega) hen.2
  · by_cases hen : S + c < blockEnd d init i ∧ e.val < d.D
    · have h := hk1 (e, c, ()) (by simpa using hen)
      have hfin := ((hpre.2.2.2.2.2.2.2.2.2.2.2.2 (bOf d i) hb).2.1 (S + c) (kvh d (hOf d i)) e
        (by omega) (kvh_lt' hd hh) hen.2).1
      obtain ⟨x, hx, hxf⟩ := hfin
      simp only [hen, and_self, ite_true]
      rw [DType.val?_bf16]
      have h' : init.read .bf16 (d.k.ptr (kIdx d (qsOf d init (bOf d i) + (S + c)) (kvh d (hOf d i)) e : Nat)).addr = some (k.get2 e c) := h
      unfold rdB at hx
      rw [h'] at hx
      have hxe : k.get2 e c = x := (Option.some.inj hx)
      refine ⟨by rw [hxe]; exact hxf, ?_⟩
      unfold keyK valB rdB
      have hnl : ¬ (preOf d init (bOf d i) + (S + c) < preOf d init (bOf d i)) := by omega
      simp only [hnl, ite_false, Nat.add_sub_cancel_left, h']
    · have hf : (decide (S + c < blockEnd d init i) && decide (e.val < d.D)) = false := by
        simpa using hen
      have := hk2 (e, c, ()) hf ok rfl
      have h0 : k.get2 e c = FloatV.ofPat 0 := this.trans (hok _)
      rw [h0]
      simp only [hen, ite_false]
      simp


/-- **The load of the new tokens' values** gives `VFactsE`. -/
theorem v_load_safe {P : Platform} {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {S : Nat}
    {Q : LoopSt d → Mem → Prop} (ptrs : Tensor [BN, d.BD] (Ptr .bf16)) (mask : Tensor [BN, d.BD] Bool)
    (hmask : ∀ (c : Fin BN) (e : Fin d.BD), mask.get2 c e =
      (decide (S + c < blockEnd d init i) && decide (e.val < d.D)))
    (hptr : ∀ (c : Fin BN) (e : Fin d.BD), S + c < blockEnd d init i → e.val < d.D →
      ptrs.get2 c e = d.v.ptr (vIdx d (qsOf d init (bOf d i) + (S + c)) (kvh d (hOf d i)) e : Nat))
    {K : Tensor [BN, d.BD] BF16 → Prog (LoopSt d)}
    (hK : ∀ k : Tensor [BN, d.BD] BF16, VFactsE d init i S (blockEnd d init i) k →
      Safe ((launch d).env P) (Rf d) ∅ Q init (K k)) :
    Safe ((launch d).env P) (Rf d) ∅ Q init
      (tl.convert .f32 .bf16 (tl.full [BN, d.BD] zf32) >>= fun other =>
        tl.load ptrs (some mask) (some other) >>= K) := by
  obtain ⟨hb, hh, -⟩ := prog_range hi
  have hext := blockEnd_le_ext d init i
  refine convert_zero_safe fun ok hok => ?_
  refine safe_load_en d.v (fun ix : Idx [BN, d.BD] => decide (S + ix.1 < blockEnd d init i) &&
      decide (ix.2.1.val < d.D))
    (fun ix : Idx [BN, d.BD] => vIdx d (qsOf d init (bOf d i) + (S + ix.1)) (kvh d (hOf d i)) ix.2.1)
    ?_ ?_ ?_ hpre.2.2.1 (v_sub_Rf d) fun k hk1 hk2 => hK k fun c e => ?_
  · rintro ⟨c, e, ⟨⟩⟩
    exact hmask c e
  · rintro ⟨c, e, ⟨⟩⟩ hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact hptr c e hen.1 hen.2
  · rintro ⟨c, e, ⟨⟩⟩ hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact v_idx_lt hd hpre hi (j := S + c) (by omega) hen.2
  · by_cases hen : S + c < blockEnd d init i ∧ e.val < d.D
    · have h := hk1 (c, e, ()) (by simpa using hen)
      have hfin := ((hpre.2.2.2.2.2.2.2.2.2.2.2.2 (bOf d i) hb).2.1 (S + c) (kvh d (hOf d i)) e
        (by omega) (kvh_lt' hd hh) hen.2).2.1
      obtain ⟨x, hx, hxf⟩ := hfin
      simp only [hen, and_self, ite_true]
      rw [DType.val?_bf16]
      have h' : init.read .bf16 (d.v.ptr (vIdx d (qsOf d init (bOf d i) + (S + c)) (kvh d (hOf d i)) e : Nat)).addr = some (k.get2 c e) := h
      unfold rdB at hx
      rw [h'] at hx
      have hxe : k.get2 c e = x := (Option.some.inj hx)
      refine ⟨by rw [hxe]; exact hxf, ?_⟩
      unfold keyV valB rdB
      have hnl : ¬ (preOf d init (bOf d i) + (S + c) < preOf d init (bOf d i)) := by omega
      simp only [hnl, ite_false, Nat.add_sub_cancel_left, h']
    · have hf : (decide (S + c < blockEnd d init i) && decide (e.val < d.D)) = false := by
        simpa using hen
      have := hk2 (c, e, ()) hf ok rfl
      have h0 : k.get2 c e = FloatV.ofPat 0 := this.trans (hok _)
      rw [h0]
      simp only [hen, ite_false]
      simp



/-- Lane `e` of `arange 0 BD` has integer value `e`. -/
theorem arD_toInt {d : Inst} (hd : d.ok) (e : Fin d.BD) :
    ((tl.arange 0 d.BD : Tensor [d.BD] I32).get1 e).toInt = e :=
  IntV.I32.toInt_tl_arange (le_trans hd.bd_le (by norm_num)) e

/-- `mask_d`'s lanes. -/
theorem extD_lane {d : Inst} (hd : d.ok) (e : Fin d.BD) :
    ((tl.arange 0 d.BD : Tensor [d.BD] I32).ltS (IntV.ofNat d.D : I32)).get1 e =
      decide (e.val < d.D) := by
  have hD : d.D < 2 ^ 31 := lt_of_le_of_lt (le_trans hd.D_le hd.bd_le) (by norm_num)
  rw [Tensor.get1_ltS, IntV.lt_eq, arD_toInt hd, IntV.I32.toInt_ofNat hD]
  simp

/-- The mask of the `k` tile: the trip's keys inside the block, the head dimension's lanes. -/
theorem kmask_lane {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {cbe sn : QI d.cm} {S : Nat}
    (hcbe : cbe.toInt = (blockEnd d init i : Int)) (hsn : sn.toInt = (S : Int))
    (hS : S < blockEnd d init i) (e : Fin d.BD) (c : Fin BN) :
    ((((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).row d.BD).and
      (((tl.arange 0 d.BD : Tensor [d.BD] I32).ltS (IntV.ofNat d.D : I32)).col BN)).get2 e c =
      (decide (S + c < blockEnd d init i) && decide (e.val < d.D)) := by
  show ((((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).get1 c) &&
    ((tl.arange 0 d.BD : Tensor [d.BD] I32).ltS (IntV.ofNat d.D : I32)).get1 e) = _
  rw [mn_lane hd hpre hi hcbe hsn hS c, extD_lane hd e]

/-- The mask of the `v` tile. -/
theorem vmask_lane {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {cbe sn : QI d.cm} {S : Nat}
    (hcbe : cbe.toInt = (blockEnd d init i : Int)) (hsn : sn.toInt = (S : Int))
    (hS : S < blockEnd d init i) (c : Fin BN) (e : Fin d.BD) :
    ((((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).col d.BD).and
      (((tl.arange 0 d.BD : Tensor [d.BD] I32).ltS (IntV.ofNat d.D : I32)).row BN)).get2 c e =
      (decide (S + c < blockEnd d init i) && decide (e.val < d.D)) := by
  show ((((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe).get1 c) &&
    ((tl.arange 0 d.BD : Tensor [d.BD] I32).ltS (IntV.ofNat d.D : I32)).get1 e) = _
  rw [mn_lane hd hpre hi hcbe hsn hS c, extD_lane hd e]

/-- The key-pointer offset at lane `(e, c)` has the integer value of the key-buffer index of the corresponding position and dimension. -/
theorem koff_lane {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s) {sn : QI d.cm} {S : Nat}
    (hsn : sn.toInt = (S : Int)) (e : Fin d.BD) (c : Fin BN)
    (hSc : S + c < blockEnd d init i) (heD : e.val < d.D) :
    (((s.cur_seq_extend_start_idx + sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BD)) *
        (toQv d.cm (Tensor.scalar (IntV.ofNat d.skt : I32))).item +
      (toQv d.cm (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.skh : I32)))).item +
      toQv d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).col BN)).get2 e c).toInt =
      ((kIdx d (qsOf d init (bOf d i) + (S + c)) (kvh d (hOf d i)) e : Nat) : Int) := by
  obtain ⟨hb, hh, -⟩ := prog_range hi
  have hext := blockEnd_le_ext d init i
  have hsz := (req_sizes hpre hb).1
  have hc := c.isLt
  have he := heD
  have hg := kvh_lt' hd hh
  have hT := hd.T_le
  have hsl := hd.strides_lt
  have hHl : d.HKV ≤ 2 ^ 16 := by
    have : d.HKV ≤ d.H := Nat.le_of_dvd (by omega) hd.kv_dvd
    have := hd.H_lt
    omega
  have hs0 : (IntV.ofNat d.skt : I32).toInt = d.skt := IntV.I32.toInt_ofNat hsl.2.2.1
  have hcc : (s.cur_kv_head * (IntV.ofNat d.skh : I32)).toInt = ((kvh d (hOf d i) * d.skh : Nat) : Int) :=
    IntV.mul_ofNat (by rw [hs.kvh]) (by
      have := hd.heads_i32.2.1
      have h2 : kvh d (hOf d i) * d.skh ≤ (d.HKV - 1) * d.skh := Nat.mul_le_mul_right _ (by omega)
      omega)
  have h1 : qsOf d init (bOf d i) + S + c < 2 ^ (qoW d.cm - 1) :=
    lt_pow_qoW d.cm (by omega)
  have h2 : (qsOf d init (bOf d i) + S + c) * d.skt + kvh d (hOf d i) * d.skh + e <
      2 ^ (qoW d.cm - 1) :=
    idx_fits hd (t := qsOf d init (bOf d i) + S + c) (g := kvh d (hOf d i)) (e := e)
      (T' := d.T) (G := d.HKV) (by omega) hg he hT hHl hsl.2.2.1 hsl.2.2.2.1
      (fun hcm => by
        have hT0 : 0 < d.T := by omega
        exact (hd.cm_i32 hcm hT0).2.1)
  have key := qoff_val d.cm (a := s.cur_seq_extend_start_idx) (p := sn)
    (x := (tl.arange 0 BN : Tensor [BN] I32).get1 c) (s0 := IntV.ofNat d.skt)
    (c := s.cur_kv_head * (IntV.ofNat d.skh : I32)) (y := (tl.arange 0 d.BD : Tensor [d.BD] I32).get1 e)
    hs.qs hsn (ar_toInt c) hs0 hcc (arD_toInt hd e) h1 h2
  have e1 : (((s.cur_seq_extend_start_idx + sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BD)) *
        (toQv d.cm (Tensor.scalar (IntV.ofNat d.skt : I32))).item +
      (toQv d.cm (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.skh : I32)))).item +
      toQv d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).col BN)).get2 e c) =
      (s.cur_seq_extend_start_idx + sn + castQ d.cm ((tl.arange 0 BN : Tensor [BN] I32).get1 c)) *
        castQ d.cm (IntV.ofNat d.skt : I32) +
        castQ d.cm (s.cur_kv_head * (IntV.ofNat d.skh : I32)) +
        castQ d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).get1 e) := by
    rw [toQv_scalar_item, toQv_scalar_item]
    show (s.cur_seq_extend_start_idx + sn + (toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).row d.BD)).get2 e c) * _ + _ +
      (toQv d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).col BN)).get2 e c = _
    rw [toQv_row_get, toQv_col_get]
  rw [e1, key]
  unfold kIdx
  push_cast
  ring


/-- The value-pointer offset at lane `(c, e)` has the integer value of the value-buffer index of the corresponding position and dimension. -/
theorem voff_lane {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s) {sn : QI d.cm} {S : Nat}
    (hsn : sn.toInt = (S : Int)) (c : Fin BN) (e : Fin d.BD)
    (hSc : S + c < blockEnd d init i) (heD : e.val < d.D) :
    (((s.cur_seq_extend_start_idx + sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).col d.BD)) *
        (toQv d.cm (Tensor.scalar (IntV.ofNat d.svt : I32))).item +
      (toQv d.cm (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.svh : I32)))).item +
      toQv d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).row BN)).get2 c e).toInt =
      ((vIdx d (qsOf d init (bOf d i) + (S + c)) (kvh d (hOf d i)) e : Nat) : Int) := by
  obtain ⟨hb, hh, -⟩ := prog_range hi
  have hext := blockEnd_le_ext d init i
  have hsz := (req_sizes hpre hb).1
  have hc := c.isLt
  have he := heD
  have hg := kvh_lt' hd hh
  have hT := hd.T_le
  have hsl := hd.strides_lt
  have hHl : d.HKV ≤ 2 ^ 16 := by
    have : d.HKV ≤ d.H := Nat.le_of_dvd (by omega) hd.kv_dvd
    have := hd.H_lt
    omega
  have hs0 : (IntV.ofNat d.svt : I32).toInt = d.svt := IntV.I32.toInt_ofNat hsl.2.2.2.2.1
  have hcc : (s.cur_kv_head * (IntV.ofNat d.svh : I32)).toInt = ((kvh d (hOf d i) * d.svh : Nat) : Int) :=
    IntV.mul_ofNat (by rw [hs.kvh]) (by
      have := hd.heads_i32.2.2.1
      have h2 : kvh d (hOf d i) * d.svh ≤ (d.HKV - 1) * d.svh := Nat.mul_le_mul_right _ (by omega)
      omega)
  have h1 : qsOf d init (bOf d i) + S + c < 2 ^ (qoW d.cm - 1) :=
    lt_pow_qoW d.cm (by omega)
  have h2 : (qsOf d init (bOf d i) + S + c) * d.svt + kvh d (hOf d i) * d.svh + e <
      2 ^ (qoW d.cm - 1) :=
    idx_fits hd (t := qsOf d init (bOf d i) + S + c) (g := kvh d (hOf d i)) (e := e)
      (T' := d.T) (G := d.HKV) (by omega) hg he hT hHl hsl.2.2.2.2.1 hsl.2.2.2.2.2.1
      (fun hcm => by
        have hT0 : 0 < d.T := by omega
        exact (hd.cm_i32 hcm hT0).2.2.1)
  have key := qoff_val d.cm (a := s.cur_seq_extend_start_idx) (p := sn)
    (x := (tl.arange 0 BN : Tensor [BN] I32).get1 c) (s0 := IntV.ofNat d.svt)
    (c := s.cur_kv_head * (IntV.ofNat d.svh : I32)) (y := (tl.arange 0 d.BD : Tensor [d.BD] I32).get1 e)
    hs.qs hsn (ar_toInt c) hs0 hcc (arD_toInt hd e) h1 h2
  have e1 : (((s.cur_seq_extend_start_idx + sn + toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).col d.BD)) *
        (toQv d.cm (Tensor.scalar (IntV.ofNat d.svt : I32))).item +
      (toQv d.cm (Tensor.scalar (s.cur_kv_head * (IntV.ofNat d.svh : I32)))).item +
      toQv d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).row BN)).get2 c e) =
      (s.cur_seq_extend_start_idx + sn + castQ d.cm ((tl.arange 0 BN : Tensor [BN] I32).get1 c)) *
        castQ d.cm (IntV.ofNat d.svt : I32) +
        castQ d.cm (s.cur_kv_head * (IntV.ofNat d.svh : I32)) +
        castQ d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).get1 e) := by
    rw [toQv_scalar_item, toQv_scalar_item]
    show (s.cur_seq_extend_start_idx + sn + (toQv d.cm ((tl.arange 0 BN : Tensor [BN] I32).col d.BD)).get2 c e) * _ + _ +
      (toQv d.cm ((tl.arange 0 d.BD : Tensor [d.BD] I32).row BN)).get2 c e = _
    rw [toQv_col_get, toQv_row_get]
  rw [e1, key]
  unfold vIdx
  push_cast
  ring



/-- A full extend tile iteration (all positions in the block) is safe and preserves the extend-loop invariant. -/
theorem ext_full_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    {q : Tensor [d.BM, d.BD] BF16} (hq : QFacts d init i q) {k : Nat}
    (hk : extStart d i + BN * k < blockEnd d init i)
    {cbe : QI d.cm} (hcbe : cbe.toInt = (blockEnd d init i : Int)) {sn : QI d.cm}
    (hsn : sn.toInt = ((extStart d i + BN * k : Nat) : Int))
    (acc : Tensor [d.BM, d.BD] F32) (deno e_max : Tensor [d.BM] F32)
    (hI : LoopInvE d init i (extStart d i) k (acc, deno, e_max) init)
    (FM : Tensor [d.BM, BN] Bool)
    (hFM : ∀ (r : Fin d.BM) (c : Fin BN), FM.get2 r c =
      (decide (validRow d init i r) && vE d init i r (extStart d i + BN * k) c)) :
    Safe ((launch d).env P) (Rf d) ∅ (fun st' m' => LoopInvE d init i (extStart d i) (k + 1) st' m') init
      (extTripP d s q sn ((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe) acc deno e_max FM) := by
  unfold extTripP
  simp only [safe_toQ]
  refine k_load_safe (S := extStart d i + BN * k) hd hpre hi _ _
    (kmask_lane hd hpre hi hcbe hsn hk) ?_ fun kt hKF => ?_
  · intro e c hc he
    simp only [Tensor.get2_ptrAdd, Buf.ptr_offset]
    rw [koff_lane hd hpre hi hs hsn e c hc he]
    simp
  refine safe_dot (hP.dotTotal _ _ _ _) fun qk0 hqk0 => ?_
  refine safe_mul fun qk1 hqk1 => ?_
  refine safe_max1 (by decide) true fun row_max hrm => ?_
  refine safe_maximum fun n_e_max hnm => ?_
  refine safe_sub fun x1 hx1 => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun re hre => ?_
  refine safe_sub fun x2 hx2 => ?_
  refine safe_math (fun _ => hP.mathTotal _ _ _) fun p hp => ?_
  refine safe_sum1 (by decide) fun sm hsm => ?_
  refine safe_mul_add fun deno' hden => ?_
  simp only [safe_toQ]
  refine v_load_safe (S := extStart d i + BN * k) hd hpre hi _ _
    (vmask_lane hd hpre hi hcbe hsn hk) ?_ fun vt hVF => ?_
  · intro c e hc he
    simp only [Tensor.get2_ptrAdd, Buf.ptr_offset]
    rw [voff_lane hd hpre hi hs hsn c e hc he]
    simp
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun p16 hp16 => ?_
  refine safe_mul fun a1 ha1 => ?_
  refine safe_dot_add (hP.dotTotal _ _ _ _) fun acc' hacc' => ?_
  refine safe_ret ⟨rfl, ?_⟩
  exact ext_trip hP hd hpre hi hk rfl
    { q := q, k := kt, v := vt, FM := FM, acc := acc, deno := deno, e_max := e_max, qk0 := qk0,
      qk1 := qk1, row_max := row_max, n_e_max := n_e_max, x1 := x1, re := re, x2 := x2, p := p,
      sm := sm, deno' := deno', p16 := p16, a1 := a1, acc' := acc',
      hdot1 := hqk0, hmul1 := hqk1, hmax := hrm, hnm := hnm, hsub1 := hx1, hexp1 := hre,
      hsub2 := hx2, hexp2 := hp, hsum := hsm, hden := hden, hcvt := hp16, hmul2 := ha1,
      hdot2 := hacc' } hq hKF hVF hFM hI.2


/-- A trip whose lanes attend no key of the row sees nothing new. -/
theorem seenE_skip {d : Inst} {init : Mem} {b p S : Nat}
    (hno : ∀ c : Nat, c < BN → preOf d init b + (S + c) ∉ keys d init b p) :
    seenE d init b p (S + BN) = seenE d init b p S := by
  ext j
  simp only [seenE, Finset.mem_filter]
  constructor
  · rintro ⟨hj, hlt⟩
    refine ⟨hj, ?_⟩
    by_contra hge
    exact hno (j - preOf d init b - S) (by unfold BN at *; omega)
      (by have : preOf d init b + (S + (j - preOf d init b - S)) = j := by omega
          rw [this]; exact hj)
  · rintro ⟨hj, hlt⟩
    exact ⟨hj, by omega⟩

/-- For `0 <= X <= 2^12`, `X + dlt BD X` is at most 2^13. -/
theorem dlt_le {d : Inst} (hd : d.ok) {X : ℝ} (h0 : 0 ≤ X) (h1 : X ≤ 2 ^ 12) :
    X + dlt d.BD X ≤ 2 ^ 13 := by
  have hBD : (d.BD : ℝ) ≤ 256 := by exact_mod_cast hd.bd_le
  have hBD0 : (0 : ℝ) ≤ d.BD := by positivity
  unfold dlt
  have h2 : ((d.BD : ℝ) + 8) * (1 / 2 ^ 22) * X ≤ (256 + 8) * (1 / 2 ^ 22) * 2 ^ 12 := by
    apply mul_le_mul (by nlinarith) h1 h0 (by positivity)
  norm_num at h2 ⊢
  linarith


/-- An extend iteration starting at or beyond the block end leaves the loop invariant unchanged. -/
theorem ext_skip_inv {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {i : Nat}
    (hi : i < (launch d).size) {k : Nat} (hk : extStart d i + BN * k < blockEnd d init i)
    (st : LoopSt d)
    (hI : LoopInvE d init i (extStart d i) k st init)
    (FM : Tensor [d.BM, BN] Bool)
    (hFM : ∀ (r : Fin d.BM) (c : Fin BN), FM.get2 r c =
      (decide (validRow d init i r) && vE d init i r (extStart d i + BN * k) c))
    (hnone : ∀ r c, FM.get2 r c = false) :
    LoopInvE d init i (extStart d i) (k + 1) st init := by
  refine ⟨rfl, fun r hr => ?_⟩
  have h := hI.2 r hr
  obtain ⟨hX0, hX12, hne, hcard, hmag, hv, htr, htr1⟩ := row_facts hd hpre hi hr
  have hseen : seenE d init (bOf d i) (mOf d i * d.BM + r) (extStart d i + BN * (k + 1)) =
      seenE d init (bOf d i) (mOf d i * d.BM + r) (extStart d i + BN * k) := by
    have e : extStart d i + BN * (k + 1) = (extStart d i + BN * k) + BN := by ring
    rw [e]
    refine seenE_skip fun c hc hmem => ?_
    have h1 := hnone r ⟨c, hc⟩
    rw [hFM r ⟨c, hc⟩] at h1
    have h2 : vE d init i r (extStart d i + BN * k) c = true := by
      unfold vE; exact decide_eq_true hmem
    simp [hr, h2] at h1
  rw [hseen]
  have hdiv : (mOf d i * d.BM + r) / d.BM = mOf d i := by
    have := r.isLt
    exact Nat.div_eq_of_lt_le (by omega) (by rw [Nat.succ_mul]; omega)
  have hk1 : k + 1 ≤ tripsExt d init (bOf d i) (mOf d i * d.BM + r) := by
    unfold tripsExt
    rw [hdiv]
    have : min (extOf d init (bOf d i)) ((mOf d i + 1) * d.BM) = blockEnd d init i := rfl
    rw [this]
    have hBN : BN = 64 := rfl
    rw [hBN] at hk ⊢
    omega
  refine RowFl.mono (dlt_le hd hX0 hX12) ((Finset.card_le_card (Finset.filter_subset _ _)).trans hcard)
    (by omega) (by unfold tmax; omega) ?_ h
  unfold g1e
  push_cast
  nlinarith


/-- The item of the negated scalar tensor is the negated boolean. -/
theorem not_item (b : Bool) : (Tensor.not (Tensor.scalar b)).item = !b := rfl

/-- The remaining (partial) extend tile iteration is safe and preserves the extend-loop invariant. -/
theorem ext_rest_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    {q : Tensor [d.BM, d.BD] BF16} (hq : QFacts d init i q) {k : Nat}
    (hk : extStart d i + BN * k < blockEnd d init i)
    {cbe : QI d.cm} (hcbe : cbe.toInt = (blockEnd d init i : Int)) {sn : QI d.cm}
    (hsn : sn.toInt = ((extStart d i + BN * k : Nat) : Int))
    (acc : Tensor [d.BM, d.BD] F32) (deno e_max : Tensor [d.BM] F32)
    (hI : LoopInvE d init i (extStart d i) k (acc, deno, e_max) init)
    (FM : Tensor [d.BM, BN] Bool)
    (hFM : ∀ (r : Fin d.BM) (c : Fin BN), FM.get2 r c =
      (decide (validRow d init i r) && vE d init i r (extStart d i + BN * k) c)) :
    Safe ((launch d).env P) (Rf d) ∅ (fun st' m' => LoopInvE d init i (extStart d i) (k + 1) st' m') init
      (extRest d s q sn ((sn + toQv d.cm (tl.arange 0 BN : Tensor [BN] I32)).ltS cbe) acc deno e_max FM) := by
  unfold extRest
  by_cases hc : (d.cm || decide (0 < d.sw)) = true
  · simp only [hc, ite_true]
    rw [safe_convB]
    simp only [pure_bind]
    rw [not_item]
    by_cases hsk : IntV.eq (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 (FM.map b2i)))
        (IntV.ofInt 0) = true
    · have hnone := (skip_iff (by rw [hd.bm]; split_ifs <;> omega) FM).mp hsk
      rw [hsk]
      simp only [Bool.not_true, Bool.false_eq_true, ite_false]
      exact safe_ret (ext_skip_inv hd hpre hi hk _ hI FM hFM hnone)
    · have hsk' := Bool.eq_false_iff.mpr hsk
      rw [hsk']
      simp only [Bool.not_false, ite_true]
      exact ext_full_safe hP hd hpre hi hs hq hk hcbe hsn acc deno e_max hI FM hFM
  · simp only [hc, Bool.false_eq_true, ite_false, pure_bind]
    rw [not_item]
    simp only [Bool.not_false, ite_true]
    exact ext_full_safe hP hd hpre hi hs hq hk hcbe hsn acc deno e_max hI FM hFM

end Claims.ExtendAttn
