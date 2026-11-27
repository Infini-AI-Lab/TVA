import Claim
import TVA.Proof

/-!
# Definitions and arithmetic shared by the proof of `logprob_ranks_i64`

`hit d r q` is whether position `q` of row `r`'s padded walk counts: a vocabulary entry
(`q < V`) counts when the token's logit is at most it, a padding lane (`q ≥ V`, holding `-inf`)
when the token's logit is at most `-inf`. `cnt d r t` is the count after `t` trips (the positions
`q < t * BLOCK`); after the last trip it is `rank d r` (`cnt_final`). `Rp`/`Wp` are a program's
read and write footprints; `Qp` is what a program leaves in memory.
-/

namespace Claims.LogprobRanks

open TVA

/-! ## Counting -/

/-- Whether position `q` of row `r`'s walk counts (`1`) or not (`0`). -/
def hit (d : Inst) (r q : Nat) : Nat :=
  if (if q < d.V then geB d.dt (xOf d r) (logitAt d r q) else geB d.dt (xOf d r) (negInf d.dt))
    = true then 1 else 0

/-- The count after `t` trips: the positions `q < t * BLOCK` that count. -/
def cnt (d : Inst) (r t : Nat) : Nat := ∑ q ∈ Finset.range (t * d.BLOCK), hit d r q

theorem hit_le (d : Inst) (r q : Nat) : hit d r q ≤ 1 := by
  unfold hit; split_ifs <;> omega

/-- Trip `t` adds the positions `t * BLOCK + l`, `l < BLOCK`. -/
theorem cnt_succ (d : Inst) (r t : Nat) :
    cnt d r (t + 1) = cnt d r t + ∑ l ∈ Finset.range d.BLOCK, hit d r (t * d.BLOCK + l) := by
  rw [cnt, cnt, Nat.add_mul, one_mul, Finset.sum_range_add]

theorem cnt_zero (d : Inst) (r : Nat) : cnt d r 0 = 0 := by simp [cnt]

/-- A count is at most the positions walked. -/
theorem cnt_le (d : Inst) (r t : Nat) : cnt d r t ≤ t * d.BLOCK := by
  calc cnt d r t ≤ ∑ _q ∈ Finset.range (t * d.BLOCK), 1 :=
        Finset.sum_le_sum fun q _ => hit_le d r q
    _ = t * d.BLOCK := by simp

/-! ## Bounds from `Inst.ok` -/

theorem blk_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK := by
  obtain ⟨k, hk⟩ := hd.blk_pow2
  rw [hk]
  exact Nat.two_pow_pos k

/-- Below the trip count, the trip's first position is inside the vocabulary. -/
theorem trip_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < numTrips d) : t * d.BLOCK < d.V := by
  have hB := blk_pos hd
  have h1 : numTrips d * d.BLOCK ≤ d.V + d.BLOCK - 1 := Nat.div_mul_le_self _ _
  have h4 : (t + 1) * d.BLOCK ≤ numTrips d * d.BLOCK := Nat.mul_le_mul_right _ ht
  rw [Nat.add_mul, one_mul] at h4
  omega

/-- The trips cover the vocabulary. -/
theorem V_le_trips {d : Inst} (hd : d.ok) : d.V ≤ numTrips d * d.BLOCK := by
  have hB := blk_pos hd
  have h3 : d.V + d.BLOCK - 1 = d.V + (d.BLOCK - 1) := by omega
  have h := Nat.div_add_mod' (d.V + (d.BLOCK - 1)) d.BLOCK
  have hr := Nat.mod_lt (d.V + (d.BLOCK - 1)) hB
  simp only [numTrips, h3]
  omega

/-- The trips walk fewer than `2^31` positions. -/
theorem trips_mul_lt {d : Inst} (hd : d.ok) : numTrips d * d.BLOCK < 2 ^ 31 := by
  have h1 : numTrips d * d.BLOCK ≤ d.V + d.BLOCK - 1 := Nat.div_mul_le_self _ _
  have h2 := hd.V_blk
  have hB := blk_pos hd
  omega

/-- After the last trip the count is the rank. -/
theorem cnt_final {d : Inst} (hd : d.ok) (r : Nat) : cnt d r (numTrips d) = rank d r := by
  have hV := V_le_trips hd
  have hsplit : numTrips d * d.BLOCK = d.V + padLanes d := by
    simp only [padLanes]; omega
  rw [cnt, hsplit, Finset.sum_range_add]
  have h1 : ∑ q ∈ Finset.range d.V, hit d r q = countGe d r := by
    rw [countGe, Finset.card_filter]
    refine Finset.sum_congr rfl fun q hq => ?_
    rw [Finset.mem_range] at hq
    simp only [hit, ite_eq_left hq]
  have h2 : ∑ k ∈ Finset.range (padLanes d), hit d r (d.V + k) =
      if geB d.dt (xOf d r) (negInf d.dt) = true then padLanes d else 0 := by
    have : ∀ k ∈ Finset.range (padLanes d), hit d r (d.V + k) =
        if geB d.dt (xOf d r) (negInf d.dt) = true then 1 else 0 := by
      intro k _
      have : ¬ (d.V + k < d.V) := by omega
      simp only [hit, ite_eq_right this]
    rw [Finset.sum_congr rfl this, Finset.sum_const, Finset.card_range, smul_eq_mul]
    split <;> simp
  rw [h1, h2, rank]

/-! ## Footprints and the post-condition of one program -/

/-- The elements of row `r` the kernel reads: `r * stride + j`, `j < V`. -/
def rowSet (d : Inst) (r : Nat) : Set Nat := {k | ∃ j, j < d.V ∧ k = r * d.stride + j}

/-- The bytes program `r` may read: its token id and its row of logits. -/
def Rp (d : Inst) (r : Nat) : Set Int := d.tok.elems {r} ∪ d.logits.elems (rowSet d r)

/-- The bytes program `r` may write: element `r` of the output. -/
def Wp (d : Inst) (r : Nat) : Set Int := d.out.elems {r}

/-- What program `r` leaves in memory: element `r` of the output holds the rank, and nothing
outside it changed. -/
def Qp (d : Inst) (init : Mem) (r : Nat) (m : Mem) : Prop :=
  ∃ v : I64, v.toInt = (rank d r : Int) ∧ m.read .i64 (d.out.ptr r).addr = some v ∧
    (∀ x, x ∉ Mem.span .i64 (d.out.ptr r).addr → m x = init x)

/-! ## Small value lemmas -/

theorem toInt_zero_i32 : (0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)

theorem intRange_32 : IntV.maxInt true 32 = 2 ^ 31 - 1 := by norm_num [IntV.maxInt]

/-- Every lane of a broadcast scalar is the scalar. -/
theorem get1_full {n : Nat} {α : Type} (a : α) (i : Fin n) :
    (Tensor.full a : Tensor [n] α).get1 i = a := rfl

/-- `get1` is `get` at `(i, ())`. -/
theorem get1_get {n : Nat} {α : Type} (x : Tensor [n] α) (i : Fin n) :
    x.get1 i = x.get (i, ()) := rfl

/-- Lane `i` of `p + offs` moves `p` by the lane of `offs`. -/
theorem get1_ptrAdd {n : Nat} {d : DType} (p : Ptr d) (o : Tensor [n] I32) (i : Fin n) :
    (p + o).get1 i = p.offset (o.get1 i).toInt := rfl

/-- A lane of the comparison `logits >= x` is `geB` of the token's logit and the lane. -/
theorem geX_get1 {n : Nat} (dt : DType) (v : Tensor [n] dt.Val) (x : dt.Val) (l : Fin n) :
    (geX dt v x).get1 l = geB dt x (v.get1 l) := by
  cases dt <;> rfl

/-- Two floats with the same pattern are equal. -/
theorem floatV_eq_of_pat {f : Format} {x y : FloatV f} (h : x.pat = y.pat) : x = y := by
  cases x; cases y
  simp only [FloatV.pat] at h
  congr
  exact BitVec.eq_of_toNat_eq h

/-- The `float32` constant of `float("-inf")` is `-inf`. -/
theorem litF32_negInf : (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000) : F32) = FloatV.negInf := by
  decide

/-- A value up to NaN payload of a value that is not NaN is that value. -/
theorem eq_of_nanEq {f : Format} {x r : FloatV f} (h : FloatV.NaNEq x r) (hx : x.isNaN = false) :
    r = x := by
  rcases h with h | ⟨h, -⟩
  · exact h.symm
  · rw [hx] at h; exact absurd h (by decide)

/-- `x.to(dt)` of the `float32` `-inf`, for the three covered dtypes, is `-inf` of `dt`. -/
theorem other_eq {P : Platform} {dt : DType} (hdt : dt = .f32 ∨ dt = .bf16 ∨ dt = .f16)
    {r : dt.Val} (h : ConvRel P .f32 dt (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)) r) :
    r = negInf dt := by
  rcases hdt with rfl | rfl | rfl
  · have h' : r.pat = (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000) : F32).pat := by
      simpa [ConvRel, ofFloatRel, FloatRel] using h
    rw [litF32_negInf] at h'
    exact floatV_eq_of_pat h'
  · have h' := (ConvRel.f32_bf16_iff (P := P)).mp h
    rw [litF32_negInf] at h'
    have hc : (FloatV.cvt (FloatV.negInf : F32) : BF16) = FloatV.negInf := by decide
    rw [hc] at h'
    exact eq_of_nanEq h' (by decide)
  · have h' := (ConvRel.f32_f16_iff (P := P)).mp h
    rw [litF32_negInf] at h'
    have hc : (FloatV.cvt (FloatV.negInf : F32) : F16) = FloatV.negInf := by decide
    rw [hc] at h'
    exact eq_of_nanEq h' (by decide)

/-! ## The `int32` sum of the converted block -/

/-- A folding `int32` addition is the sum of the `toInt`s, when nothing overflows. -/
theorem foldl_add_toInt (l : List I32) (a : I32) (ha : 0 ≤ a.toInt)
    (hpos : ∀ x ∈ l, 0 ≤ x.toInt)
    (hbound : a.toInt + (l.map IntV.toInt).sum < 2 ^ 31) :
    (l.foldl (· + ·) a).toInt = a.toInt + (l.map IntV.toInt).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x t ih =>
    have hx : 0 ≤ x.toInt := hpos x (List.mem_cons_self ..)
    have ht : ∀ y ∈ t, 0 ≤ y.toInt := fun y hy => hpos y (List.mem_cons_of_mem _ hy)
    have htsum : 0 ≤ (t.map IntV.toInt).sum := by
      apply List.sum_nonneg
      intro y hy
      obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hy
      exact ht z hz
    simp only [List.foldl_cons, List.map_cons, List.sum_cons] at hbound ⊢
    have hax : (a + x).toInt = a.toInt + x.toInt := by
      refine IntV.toInt_add rfl rfl ?_
      simp only [IntV.SFits]
      constructor <;> norm_num <;> omega
    rw [ih (a + x) (by rw [hax]; omega) ht (by rw [hax]; omega), hax]
    omega

/-- The wrapping `int32` sum of a block whose lanes have natural values `f l` is their sum, when
it is below `2^31`. -/
theorem foldAll_toInt {n : Nat} (c : Tensor [n] I32) (f : Nat → Nat)
    (hc : ∀ l : Fin n, (c.get1 l).toInt = (f l : Int))
    (hb : ∑ l ∈ Finset.range n, f l < 2 ^ 31) :
    (Tensor.foldAll (· + ·) (0 : I32) c).toInt = ((∑ l ∈ Finset.range n, f l : Nat) : Int) := by
  have hmap : (Idx.all [n]).map c.get = (List.finRange n).map c.get1 := by
    let g : Fin n → Idx [n] := fun i => (i, ())
    have hfun : (fun i => (Idx.all ([] : Shape)).map (fun j => ((i, j) : Idx [n])))
        = (fun i => ([g i] : List (Idx [n]))) := by
      funext i
      rfl
    have h2 : (List.finRange n).flatMap
          (fun i => (Idx.all ([] : Shape)).map (fun j => ((i, j) : Idx [n])))
        = (List.finRange n).flatMap (fun i => ([g i] : List (Idx [n]))) :=
      congrArg (fun f => List.flatMap f (List.finRange n)) hfun
    have h3 : (List.finRange n).flatMap (fun i => ([g i] : List (Idx [n])))
        = (List.finRange n).map g := List.map_eq_flatMap.symm
    have hidx : Idx.all [n] = (List.finRange n).map g :=
      Eq.trans (rfl : Idx.all [n] = (List.finRange n).flatMap
          (fun i => (Idx.all ([] : Shape)).map (fun j => ((i, j) : Idx [n])))) (Eq.trans h2 h3)
    exact Eq.trans (congrArg (fun l => l.map c.get) hidx)
      (Eq.trans List.map_map
        (List.map_congr_left (fun i _ => (rfl : c.get (g i) = c.get1 i))))
  have hlist : Tensor.foldAll (· + ·) (0 : I32) c
      = ((List.finRange n).map c.get1).foldl (· + ·) 0 :=
    congrArg (fun l => l.foldl (· + ·) (0 : I32)) hmap
  have hsum_eq : (((List.finRange n).map c.get1).map IntV.toInt).sum
      = ((∑ l ∈ Finset.range n, f l : Nat) : Int) := by
    rw [List.map_map]
    simp only [Function.comp_def]
    rw [List.map_congr_left (fun i _ => hc i)]
    rw [show ((List.finRange n).map (fun i : Fin n => (f (i : Nat) : Int))).sum
        = ∑ l ∈ Finset.range n, (f l : Int) from by
      rw [Finset.sum_range]
      rw [← List.sum_toFinset _ (List.nodup_finRange n), List.toFinset_finRange]]
    rw [Nat.cast_sum]
  have hpos : ∀ x ∈ (List.finRange n).map c.get1, 0 ≤ x.toInt := by
    intro x hx
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp hx
    rw [hc i]
    exact Int.natCast_nonneg _
  have hbound : (0 : I32).toInt
      + (((List.finRange n).map c.get1).map IntV.toInt).sum < 2 ^ 31 := by
    rw [toInt_zero_i32, hsum_eq, Int.zero_add]
    exact_mod_cast hb
  rw [hlist, foldl_add_toInt _ _ (by rw [toInt_zero_i32]) hpos hbound, toInt_zero_i32, hsum_eq]
  omega

end Claims.LogprobRanks
