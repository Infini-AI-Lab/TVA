import Proof.Defs

/-!
# One trip of the loop

Trip `t` (induction variable `i = t * BLOCK`) converts `other`, loads the lanes
`t * BLOCK + l` of the row (masked by `< V`, the others `-inf`), compares them with the token's
logit, converts the comparison to `int32` and adds its sum to `n`. It reads only the row and
writes nothing; it advances the invariant `n = cnt d r t`.
-/

namespace Claims.LogprobRanks

open TVA

/-- The body of one trip, with the row pointer and the token's logit as parameters. -/
def tripBody (d : Inst) (row_ptr : Ptr d.dt) (x : d.dt.Val) (i : I32) (n : I32) : Prog I32 :=
  tl.convert .f32 d.dt negInfLit >>= fun other =>
  tl.load (row_ptr + (i + tl.arange 0 d.BLOCK))
      (some ((i + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.V))) (some other) >>= fun logits =>
  tl.convert .i1 .i32 (geX d.dt logits x) >>= fun c =>
  pure (n + Tensor.foldAll (· + ·) (0 : I32) c)

/-- `x.to(dt)` of the `float32` `-inf` has a result, for the three covered dtypes. -/
theorem other_exists {P : Platform} {dt : DType} (hdt : dt = .f32 ∨ dt = .bf16 ∨ dt = .f16)
    (x : F32) : ∃ r, ConvRel P .f32 dt x r := by
  rcases hdt with rfl | rfl | rfl
  · exact ⟨x, by simp [ConvRel, ofFloatRel, FloatRel]⟩
  · exact ⟨_, ConvRel.cvt_f32_bf16 x⟩
  · exact ⟨_, ConvRel.cvt_f32_f16 x⟩

/-- `x.to(tl.int32)` of an `int1` has a result. -/
theorem i1_exists {P : Platform} (b : Bool) : ∃ r : I32, ConvRel P .i1 .i32 b r :=
  ⟨IntV.ofInt (if b then 1 else 0), rfl⟩

/-- The lanes of `i + tl.arange(0, BLOCK)` at trip `t`. -/
theorem block_lane {d : Inst} (hd : d.ok) {t : Nat} (ht : t < numTrips d) {i : I32}
    (hi : i.toInt = ((t * d.BLOCK : Nat) : Int)) (l : Fin d.BLOCK) :
    ((i + tl.arange 0 d.BLOCK).get1 l).toInt = ((t * d.BLOCK + l : Nat) : Int) := by
  have h1 := trip_lt hd ht
  have h2 := hd.V_blk
  have h3 : t * d.BLOCK + d.BLOCK ≤ 2 ^ 31 := by omega
  exact IntV.I32.toInt_sadd_arange (n := d.BLOCK) hi h3 l

/-- **One trip** is safe, reads inside row `r`, writes nothing, and adds the trip's lanes to the
count. -/
theorem safe_trip {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {r : Nat} (hr : r < d.B) {t : Nat} (ht : t < numTrips d) {i : I32}
    (hi : i.toInt = ((t * d.BLOCK : Nat) : Int)) {n : I32}
    (hn : n.toInt = (cnt d r t : Int)) :
    Safe E (Rp d r) ∅ (fun n' m => m = init ∧ n'.toInt = (cnt d r (t + 1) : Int)) init
      (tripBody d (d.logits.ptr ((r * d.stride : Nat) : Int)) (xOf d r) i n) := by
  unfold tripBody
  have hV : (IntV.ofNat d.V : I32).toInt = (d.V : Int) :=
    IntV.I32.toInt_ofNat (by have := hd.V_blk; have := blk_pos hd; omega)
  refine safe_convert (fun _ => other_exists hd.dt_ok _) fun other hother => ?_
  have hoth : ∀ j, other.get j = negInf d.dt := fun j => other_eq hd.dt_ok (hother j)
  -- the lanes: position `t * B + l`, element `r * stride + t * B + l`
  let g : Idx [d.BLOCK] → Nat := fun j => r * d.stride + (t * d.BLOCK + j.1)
  let en : Idx [d.BLOCK] → Bool := fun j => decide (t * d.BLOCK + j.1 < d.V)
  have hmask : ∀ j, ((i + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.V)).get j = en j := by
    rintro ⟨l, ⟨⟩⟩
    show ((i + tl.arange 0 d.BLOCK).get1 l).lt (IntV.ofNat d.V) = _
    rw [IntV.lt_eq, block_lane hd ht hi l, hV]
    simp only [en, Nat.cast_lt]
  have hptr : ∀ j, en j = true →
      (d.logits.ptr ((r * d.stride : Nat) : Int) + (i + tl.arange 0 d.BLOCK)).get j =
        d.logits.ptr (g j : Int) := by
    rintro ⟨l, ⟨⟩⟩ _
    show (d.logits.ptr ((r * d.stride : Nat) : Int) + (i + tl.arange 0 d.BLOCK)).get1 l = _
    rw [get1_ptrAdd, block_lane hd ht hi l, Buf.ptr_offset]
    simp only [g]
    push_cast
    ring_nf
  have hin : ∀ j, en j = true → g j < d.logits.len := by
    intro j hj
    have hj' : t * d.BLOCK + j.1 < d.V := of_decide_eq_true hj
    have := hd.logits_len r hr
    simp only [g]
    omega
  have hR : d.logits.elems (Buf.img g en) ⊆ Rp d r := by
    intro z hz
    obtain ⟨k, ⟨j, hj, rfl⟩, hzk⟩ := hz
    exact Or.inr ⟨g j, ⟨t * d.BLOCK + j.1, of_decide_eq_true hj, rfl⟩, hzk⟩
  refine safe_load_buf (g := g) (en := en) d.logits (Buf.addrs_of_get d.logits hmask hptr) hin
    hpre.1 hR fun v hv hvo => ?_
  -- each lane of `v`: the row's entry in range, `-inf` past it
  have hlane : ∀ l : Fin d.BLOCK, v.get1 l =
      if t * d.BLOCK + l < d.V then logitAt d r (t * d.BLOCK + l) else negInf d.dt := by
    intro l
    by_cases hl : t * d.BLOCK + l < d.V
    · rw [ite_eq_left hl]
      have hen : en (l, ()) = true := decide_eq_true hl
      have h1 := hv (l, ()) hen
      have h2 := hpre.2.2.2.1 (g (l, ())) (hin (l, ()) hen)
      rw [h2] at h1
      exact (Option.some.inj h1).symm
    · rw [ite_eq_right hl]
      have hen : en (l, ()) = false := decide_eq_false hl
      rw [get1_get, hvo (l, ()) hen other rfl]
      exact hoth _
  refine safe_convert (fun _ => i1_exists _) fun c hc => ?_
  have hcl : ∀ l : Fin d.BLOCK, (c.get1 l).toInt = (hit d r (t * d.BLOCK + l) : Int) := by
    intro l
    show (c.get (l, ())).toInt = _
    rw [ConvRel.toInt_i1_i32 (hc (l, ()))]
    show (if (geX d.dt v (xOf d r)).get1 l = true then (1 : Int) else 0) = _
    rw [geX_get1, hlane l]
    simp only [hit]
    by_cases hl : t * d.BLOCK + l < d.V
    · simp only [hl, ite_true]; split <;> simp
    · simp only [hl, ite_false]; split <;> simp
  have hs : ∑ l ∈ Finset.range d.BLOCK, hit d r (t * d.BLOCK + l) ≤ d.BLOCK := by
    calc ∑ l ∈ Finset.range d.BLOCK, hit d r (t * d.BLOCK + l)
        ≤ ∑ _l ∈ Finset.range d.BLOCK, 1 := Finset.sum_le_sum fun l _ => hit_le d r _
      _ = d.BLOCK := by simp
  have hsum := foldAll_toInt c (fun l => hit d r (t * d.BLOCK + l)) hcl
    (lt_of_le_of_lt hs (by have := hd.blk_le; omega))
  refine safe_ret ⟨rfl, ?_⟩
  have hb : cnt d r t + ∑ l ∈ Finset.range d.BLOCK, hit d r (t * d.BLOCK + l) < 2 ^ 31 := by
    rw [← cnt_succ]
    have h1 := cnt_le d r (t + 1)
    have h2 : (t + 1) * d.BLOCK ≤ numTrips d * d.BLOCK := Nat.mul_le_mul_right _ ht
    have h3 := trips_mul_lt hd
    omega
  rw [IntV.I32.add_nat hn hsum hb, cnt_succ]
  rfl

end Claims.LogprobRanks
