import TVA
open TVA Tile2

/-! Round 40 probes (library author): the reviewers' cases of L180, L181 and L183, checked with
`lake env lean` from `lean/`. -/

/-! L180 (lr-39b): `M = 1000`, `BM = 64`, `s0 = 2149000`, `N = BN = 1`, `s1 = 1`, `int32`. The matrix
bound `(M - 1) * s0 + (N - 1) * s1 < 2^31` holds, the every-lane bound fails on the last tile
(`r0 = 960`), and the enabled-lane rule gives the pointers of that tile. -/
example : (1000 - 1) * 2149000 + (1 - 1) * 1 < 2 ^ (32 - 1) := by norm_num
example : ¬ ((960 + 64 - 1) * 2149000 + (0 + 1 - 1) * 1 < 2 ^ (32 - 1)) := by norm_num

example {d : DType} (b : Buf d) {rows : Tensor [64] I32} {cols : Tensor [1] I32} {a0 a1 : I32}
    (hr : ∀ i : Fin 64, (rows.get1 i).toInt = ((960 + i : Nat) : Int))
    (hc : ∀ j : Fin 1, (cols.get1 j).toInt = ((0 + j : Nat) : Int))
    (ha0 : a0.toInt = (2149000 : Nat)) (ha1 : a1.toInt = (1 : Nat)) :
    ∀ (i : Fin 64) (j : Fin 1), 960 + (i : Nat) < 1000 → 0 + (j : Nat) < 1 →
      (b.ptr 0 + (rows.col 1 * a0 + cols.row 64 * a1)).get2 i j =
        b.ptr ((0 + (960 + i) * 2149000 + (0 + j) * 1 : Nat) : Int) :=
  ptr_lane_on (M := 1000) (N := 1) (e := 0) b (by simp) hr hc ha0 ha1 (by norm_num)

-- ... and the load rule takes it, with the mask rule.
example {d : DType} (b : Buf d) {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {rows : Tensor [64] I32} {cols : Tensor [1] I32} {a0 a1 mv nv : I32}
    (hr : ∀ i : Fin 64, (rows.get1 i).toInt = ((960 + i : Nat) : Int))
    (hc : ∀ j : Fin 1, (cols.get1 j).toInt = ((0 + j : Nat) : Int))
    (ha0 : a0.toInt = (2149000 : Nat)) (ha1 : a1.toInt = (1 : Nat))
    (hmv : mv.toInt = (1000 : Nat)) (hnv : nv.toInt = (1 : Nat))
    (hlen : 0 + (1000 - 1) * 2149000 + (1 - 1) * 1 < b.len) (hv : b.Valid m)
    (hR : b.tile2 0 2149000 1 960 0 64 1 1000 1 ⊆ R) {k : Tensor [64, 1] d.Val → Prog α}
    (hk : ∀ v, Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load (b.ptr 0 + (rows.col 1 * a0 + cols.row 64 * a1))
      (some (((rows.col 1).ltS mv).and ((cols.row 64).ltS nv))) none >>= k) :=
  safe_load_tile2 b (ptr_lane_on (e := 0) b (by simp) hr hc ha0 ha1 (by norm_num))
    (maskOk_and hr hc hmv hnv) (.inr (.inr hlen)) hv hR fun v _ _ => hk v

/-! L180, 1-D: every program with an element below `n ≤ 2^31` fits, for a power-of-two block. -/
example {p n : Nat} (hp : p * 1024 < n) (hn : n ≤ 2 ^ 31) : Block.Fits p 1024 :=
  Block.fits_of_lt_pow2 (e := 10) (by norm_num) (by norm_num) hp hn

/-! L181 (lr-39a `work/tile2_empty.lean`): an empty matrix (`M = 0`) at the end of its buffer
(`e = b.len`) now meets `safe_load_tile2` and `safe_store_tile2`; every lane is masked off, so
`hp` holds vacuously. -/
example {d : DType} (b : Buf d) {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
    {m : Mem} {BM BN c0 N s0 s1 : Nat} {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} (hm : MaskOk mk 0 c0 0 N) (hv : b.Valid m)
    (hR : b.tile2 b.len s0 s1 0 c0 BM BN 0 N ⊆ R) {k : Tensor [BM, BN] d.Val → Prog α}
    (hk : ∀ v, Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load ptrs mk none >>= k) :=
  safe_load_tile2 (e := b.len) (s0 := s0) (s1 := s1) b (fun _ _ h _ => absurd h (by omega)) hm
    (.inl rfl) hv hR fun v _ _ => hk v

example {d : DType} (b : Buf d) {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
    {m : Mem} {BM BN c0 N s0 s1 : Nat} {ptrs : Tensor [BM, BN] (Ptr d)}
    {mk : Option (Tensor [BM, BN] Bool)} {vals : Tensor [BM, BN] d.Val}
    (hm : MaskOk mk 0 c0 0 N) (hinj : Inj s0 s1 0 N) (hv : b.Valid m)
    (hW : b.tile2 b.len s0 s1 0 c0 BM BN 0 N ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m', Safe E R (W \ b.tile2 b.len s0 s1 0 c0 BM BN 0 N) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals mk >>= k) :=
  safe_store_tile2 (e := b.len) b (fun _ _ h _ => absurd h (by omega)) hm (.inl rfl) hinj hv hW
    fun m' _ _ _ => hk m'

/-! L183: `pid.to(tl.int64) * BM + tl.arange(0, BM)` as the model writes it, the `arange` converted
by `tl.convert .i32 .i64`, through `safe_convert`. -/
example {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {a : I64} {r0 : Nat}
    (ha : a.toInt = r0) (hfit : r0 + 64 ≤ 2 ^ 63) {k : Tensor [64 - 0] I64 → Prog α}
    (hk : ∀ x : Tensor [64 - 0] I64,
      (∀ i : Fin (64 - 0), ((a + x).get1 i).toInt = ((r0 + i : Nat) : Int)) → Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.convert .i32 .i64 (tl.arange 0 64) >>= k) :=
  safe_convert (fun i => ⟨_, ConvRel.exists_i32_i64 _ |>.choose_spec⟩)
    fun x hx => hk x (start_arange64_toInt ha hx hfit (by norm_num))
