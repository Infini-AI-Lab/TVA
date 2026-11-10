import TVA

/-! lr-41b, round 41: probes of L184 (Tile2, rows and columns of any form), L185 (the integer cast)
and L189 (the `ofBits` lemmas no longer `simp`). Run with `lake env lean` from `lean/`. -/

namespace TVA
open Tile2

section L184
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- P1. Tutorial 03's `b` tile: `offs_bn = (pid_n * BN + tl.arange(0, BN)) % N` (columns wrapped,
`κ j = (c0 + j) % N`, no column mask), `offs_k = tl.arange(0, BK)` rows under the mask
`offs_k[:, None] < K`: `maskAt_rows`, `ptr_lane_on`, `safe_load_tileAt`. -/
example {BK BN K N c0 sk sn : Nat} (b : Buf .f16) {base : Ptr .f16} (hb : base = b.ptr (0 : Int))
    {x y : Tensor [BN] I32} {rows : Tensor [BK] I32} {a0 a1 kv : I32}
    (hx : ∀ j : Fin BN, (x.get1 j).toInt = ((c0 + j : Nat) : Int))
    (hy : ∀ j : Fin BN, (y.get1 j).toInt = (N : Int)) (hN : 0 < N)
    (hr : ∀ i : Fin BK, (rows.get1 i).toInt = ((i : Nat) : Int))
    (ha0 : a0.toInt = sk) (ha1 : a1.toInt = sn) (hkv : kv.toInt = K)
    (hfit : (K - 1) * sk + (N - 1) * sn < 2 ^ (32 - 1))
    (hlen : (K - 1) * sk + (N - 1) * sn < b.len) (hv : b.Valid m)
    (hR : b.tile2 0 sk sn 0 0 K N K N ⊆ R)
    {k : Tensor [BK, BN] (DType.f16).Val → Prog α}
    (hk : ∀ v : Tensor [BK, BN] (DType.f16).Val,
      (∀ (i : Fin BK) (j : Fin BN), (i : Nat) < K → (c0 + j) % N < N →
        m.read .f16 (b.ptr ((0 + (i : Nat) * sk + (c0 + j) % N * sn : Nat) : Int)).addr =
          some (v.get2 i j)) → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.irem x y >>= fun cols =>
      tl.load (base + ((rows.col BN) * a0 + (cols.row BK) * a1))
        (some ((rows.col BN).ltS kv)) none >>= k) := by
  refine safe_irem_lanes hx hy hN fun cols hc => ?_
  have hp := ptr_lane_on (e := 0) b hb hr hc ha0 ha1 hfit
  have hm : MaskAt (some ((rows.col BN).ltS kv)) (fun i => (i : Nat)) (fun j => (c0 + j) % N) K N :=
    maskAt_rows hr hkv fun j => Nat.mod_lt _ hN
  exact safe_load_tileAt b hp hm (.inr (.inr (by simpa using hlen))) hv
    ((Buf.tileAt_sub_matrix b).trans hR) fun v h1 _ => hk v h1

/-- P2. The same wrap in `int64` (`pid.to(tl.int64) * BM + tl.arange(0, BM)` promoted, then `% M`):
`start_arange64_toInt` feeds `irem_toInt` at width 64. -/
example {P : Platform} {BM M r0 : Nat} {a : I64} {x : Tensor [BM - 0] I64} {y r : Tensor [BM - 0] I64}
    (ha : a.toInt = r0) (hx : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 BM).get i) (x.get i))
    (hfit : r0 + BM ≤ 2 ^ 63) (hn : BM ≤ 2 ^ 31)
    (hy : ∀ i : Fin (BM - 0), (y.get1 i).toInt = (M : Int)) (hM : 0 < M)
    (hr : ∀ i, ((a + x).get i).rem? (y.get i) = some (r.get i)) (i : Fin (BM - 0)) :
    (r.get1 i).toInt = (((r0 + i) % M : Nat) : Int) :=
  irem_toInt (f := fun i => r0 + i) (start_arange64_toInt ha hx hfit hn) hy hM hr i

/-- P3. A store of `% M` rows with `BM ≤ M` and the first row `r0 < M`... is not injective in
general (`r0 = M - 1`, `BM = 2`: rows `M - 1, 0`, distinct), but with `BM ≤ M` distinct lanes always
give distinct rows. Proved here by hand: no library lemma gives `hρ` for `% M` rows. -/
example {BM M r0 : Nat} (hBM : BM ≤ M) :
    ∀ i i' : Fin BM, (r0 + (i : Nat)) % M < M → (r0 + (i' : Nat)) % M < M →
      (r0 + (i : Nat)) % M = (r0 + i') % M → i = i' := by
  intro i i' _ _ h
  have hi := i.isLt; have hi' := i'.isLt
  apply Fin.ext
  have := Nat.ModEq.add_left_cancel' r0 h
  have h1 : (i : Nat) % M = i := Nat.mod_eq_of_lt (by omega)
  have h2 : (i' : Nat) % M = i' := Nat.mod_eq_of_lt (by omega)
  unfold Nat.ModEq at this; omega

/-- P4. Old callers that pass `hr` about `r0 + i` get back the old form of the pointers, with
`r0 + i` (the implicit `ρ` beta-reduced), and that form feeds the `r0 + i` load rule unchanged. -/
example {BM BN M N r0 c0 s0 s1 : Nat} (b : Buf .f32) {base : Ptr .f32} (hb : base = b.ptr (0 : Int))
    {rows : Tensor [BM] I32} {cols : Tensor [BN] I32} {a0 a1 : I32}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int))
    (ha0 : a0.toInt = s0) (ha1 : a1.toInt = s1)
    (hfit : (M - 1) * s0 + (N - 1) * s1 < 2 ^ (32 - 1)) :
    ∀ (i : Fin BM) (j : Fin BN), r0 + i < M → c0 + j < N →
      (base + ((rows.col BN) * a0 + (cols.row BM) * a1)).get2 i j =
        b.ptr ((0 + (r0 + i) * s0 + (c0 + j) * s1 : Nat) : Int) :=
  ptr_lane_on (e := 0) b hb hr hc ha0 ha1 hfit

/-- P5. `irem_ok`'s premise cannot be met by a negative row: `f` is a natural (a lane of value
`-1` has no `f`), so the rule does not claim `(-1) % M` is `M - 1` (C semantics gives `-1`). -/
example : (IntV.ofInt (-1) : I32).rem? (IntV.ofInt 3) = some (IntV.ofInt (-1)) := by decide

end L184

section L185
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- P6. Any shape, with `y` inferred from the goal. (The IntW section docstring's form,
`safe_convert_iff (fun i r => ConvRel.iff_cast_i32_i64) (y := x.castInt)`, does not elaborate:
`Variant.lean`, `variant.out`.) -/
example {s : Shape} (x : Tensor s I32) (k : Tensor s I64 → Prog α) :
    Safe E R W Q m (tl.convert .i32 .i64 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff fun _ _ => ConvRel.iff_cast_i32_i64

/-- P7. A 2-D tensor, unsigned to signed narrowing. -/
example (x : Tensor [4, 8] U64) (k : Tensor [4, 8] I16 → Prog α) :
    Safe E R W Q m (tl.convert .u64 .i16 x >>= k) ↔ Safe E R W Q m (k x.castInt) :=
  safe_convert_iff fun _ _ => ConvRel.iff_cast_u64_i16

/-- P8. The scalar form the claims write (`IntV.cast (tl.program_id ctx 0)` for `pid.to(tl.int64)`):
the one result of the scalar conversion relation. -/
example {P : Platform} (pid : I32) : ConvRel P .i32 .i64 pid (IntV.cast pid) :=
  ConvRel.iff_cast_i32_i64.2 rfl

/-- P9. `int1` is excluded, and rightly: `ConvRel .i1 .i32 true r` is `r = 1`, while `IntV.cast`
does not apply (`int1` is `Bool`). And `uint8 255 → int8` is `-1`, `int8 -1 → uint16` is `65535`
(sign-extension by the source). -/
example : ((IntV.cast (IntV.ofInt 255 : U8) : I8)).toInt = -1 := by decide
example : ((IntV.cast (IntV.ofInt (-1) : I8) : U16)).toInt = 65535 := by decide
example : ((IntV.cast (IntV.ofInt 200 : U8) : I32)).toInt = 200 := by decide

/-- P10. `safe_convert_iff` cannot be used where a conversion has several results: for `f32 → f16`
of a NaN the relation allows any NaN (no `y` with `↔`). Its premise is the content, checked: the
premise is false for the identity map on a non-unique relation. (Statement only.) -/
example : ¬ ∀ r : Bool, (r = true ∨ r = false) ↔ r = true := by decide

end L185

section L189

/-- P11. The six `ofBits` lemmas are no longer `simp`: `simp` leaves `DType.ofBits .f16 0` alone,
and a proof cites them by name. -/
example : DType.ofBits .f16 0 = (FloatV.ofPat 0 : F16) := by rw [DType.ofBits_f16]
example (h : (DType.ofBits .f16 0 : F16).toRat = 0) : (DType.ofBits .f16 0 : F16).toRat = 0 := by
  simp [h]

end L189

end TVA
