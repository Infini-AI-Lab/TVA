import TVA
open TVA Tile2

/-! lr-40a, round 40: usability probes of L180, L181 and L183 in the forms a transcription
reaches, checked with `lake env lean` from `lean/`. -/

/-! L180: the last partial tile of `pid * BM + tl.arange(0, BM)` (int32), the row term and the
column term added one at a time (`base + rows[:, None] * s0 + cols[None, :] * s1`), from the
matrix bound only; then the store rule. Numbers of masked_wrap.py (`M = 1000`, `BM = 64`,
`s0 = 2149000`, `N = BN = 16`, `s1 = 1`), with `pid = 15`. -/
example {d : DType} (b : Buf d) {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {a0 a1 mv nv : I32} {vals : Tensor [64, 16] d.Val}
    (ha0 : a0.toInt = (2149000 : Nat)) (ha1 : a1.toInt = (1 : Nat))
    (hmv : mv.toInt = (1000 : Nat)) (hnv : nv.toInt = (16 : Nat))
    (hlen : 0 + (1000 - 1) * 2149000 + (16 - 1) * 1 < b.len) (hv : b.Valid m)
    (hW : b.tile2 0 2149000 1 960 0 64 16 1000 16 ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m', Safe E R (W \ b.tile2 0 2149000 1 960 0 64 16 1000 16) Q m' (k ())) :
    let rows : Tensor [64] I32 := (IntV.ofNat 15 * IntV.ofNat 64 : I32) + tl.arange 0 64
    let cols : Tensor [16] I32 := tl.arange 0 16
    Safe E R W Q m (tl.store (b.ptr 0 + rows.col 16 * a0 + cols.row 64 * a1) vals
      (some (((rows.col 16).ltS mv).and ((cols.row 64).ltS nv))) >>= k) := by
  intro rows cols
  have hr : ∀ i : Fin 64, (rows.get1 i).toInt = ((960 + i : Nat) : Int) := by
    intro i
    have hp : ((IntV.ofNat 15 * IntV.ofNat 64 : I32)).toInt = ((960 : Nat) : Int) := by
      simpa using IntV.mul_ofNat (w := 32) (a := (IntV.ofNat 15 : I32)) (x := 15) (n := 64)
        (by simp) (by norm_num)
    exact IntV.I32.toInt_sadd_arange hp (by norm_num) i
  have hc : ∀ j : Fin 16, (cols.get1 j).toInt = ((0 + j : Nat) : Int) := by
    intro j; simpa using IntV.I32.toInt_tl_arange (n := 16) (by norm_num) j
  exact safe_store_tile2 (e := 0) b
    (ptr_lane_seq_on b (by simp) hr hc ha0 ha1 (by norm_num) (by norm_num))
    (maskOk_and hr hc hmv hnv) (.inr (.inr hlen)) (inj_rowMajor1 (by norm_num)) hv hW
    fun m' _ _ _ => hk m'

/-! L180, 1-D: the last block of `n = 2^31 - 1` with `B = 1024` fits (`p = (n - 1) / 1024`). -/
example : Block.Fits ((2 ^ 31 - 2) / 1024) 1024 :=
  Block.fits_of_lt_pow2 (e := 10) (by norm_num) (by norm_num) (by norm_num) (n := 2 ^ 31 - 1)
    (by norm_num)

/-! L181: an empty column range (`N = 0`) of a tile placed past the end of its buffer
(`e = b.len + 5`): load and footprint go through with `.inr (.inl rfl)`. -/
example {d : DType} (b : Buf d) {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
    {m : Mem} {BM BN r0 M s0 s1 : Nat} {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} (hm : MaskOk mk r0 0 M 0) (hv : b.Valid m)
    {k : Tensor [BM, BN] d.Val → Prog α} (hk : ∀ v, Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs mk none >>= k) := by
  have hsub : b.tile2 (b.len + 5) s0 s1 r0 0 BM BN M 0 ⊆ R := by
    intro x hx
    obtain ⟨y, ⟨i, j, -, -, -, hj, -⟩, -⟩ := hx
    omega
  exact safe_load_tile2 (e := b.len + 5) b (fun _ _ _ h => absurd h (by omega)) hm
    (.inr (.inl rfl)) hv hsub fun v _ _ => hk v

/-! L183 and the castInt wording: for an integer widening, `tl.convert`'s relation is the cast
`castInt` computes, so the two transcriptions of `x.to(tl.int64)` agree. -/
example {P : Platform} (a : I32) : ConvRel P .i32 .i64 a (IntV.cast a) := by
  show IntV.cast a = IntV.ofInt a.toInt
  apply IntV.ext_toInt
  rw [IntV.toInt_cast _ (by norm_num), IntV.toInt_ofInt_toInt a (by norm_num)]
