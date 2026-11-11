import Proof.Bounds

/-!
# Lane lemmas

Masks and offsets of tiles whose two index vectors have different integer widths (the comparison
with the limit is in the variant's width, the comparison of a column with `dstate` or `headdim`
in `int32`).
-/

namespace Claims.SsdChunkScan

open TVA

/-- A lane compared with a limit, both of known value. -/
theorem lt_lane' {w : Nat} {x mv : IntV true w} {k K : Nat} (hx : x.toInt = (k : Int))
    (hm : mv.toInt = (K : Int)) : IntV.lt x mv = decide (k < K) := by
  rw [IntV.lt_eq, hx, hm]; simp only [Nat.cast_lt]

/-- `(rows < M)[:, None] & (cols < N)[None, :]` with `rows` and `cols` of different widths. -/
theorem maskAt_and2 {w w' BM BN M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w')} {mv : IntV true w}
    {nv : IntV true w'} (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int)) (hmv : mv.toInt = M)
    (hnv : nv.toInt = N) :
    Tile2.MaskAt (some (((rows.col BN).ltS mv).and ((cols.row BM).ltS nv))) ρ κ M N := by
  intro i j
  show (IntV.lt (rows.get1 i) mv && IntV.lt (cols.get1 j) nv) = _
  rw [lt_lane' (hr i) hmv, lt_lane' (hc j) hnv]

/-- The tile mask with shifted index vectors, in two widths: `rows` is `ρ - sr`, `mv` is `M - sr`;
`cols` is `κ - sc`, `nv` is `N - sc` (a loop's `offs_k < limit - k`). -/
theorem maskAt_and_off {w w' BM BN M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}
    {rows : Tensor [BM] (IntV true w)} {cols : Tensor [BN] (IntV true w')} {mv : IntV true w}
    {nv : IntV true w'} (sr sc : Int)
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((ρ i : Nat) : Int) - sr)
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((κ j : Nat) : Int) - sc)
    (hmv : mv.toInt = (M : Int) - sr) (hnv : nv.toInt = (N : Int) - sc) :
    Tile2.MaskAt (some (((rows.col BN).ltS mv).and ((cols.row BM).ltS nv))) ρ κ M N := by
  intro i j
  show (IntV.lt (rows.get1 i) mv && IntV.lt (cols.get1 j) nv) = _
  rw [IntV.lt_eq, IntV.lt_eq, hr i, hc j, hmv, hnv]
  congr 1 <;> simp only [decide_eq_decide] <;> omega

end Claims.SsdChunkScan
