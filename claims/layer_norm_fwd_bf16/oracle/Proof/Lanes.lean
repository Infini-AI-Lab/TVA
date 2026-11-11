import Claim
import TVA.Proof

namespace Claims.LayerNormFwd

open TVA


/-! Integer lane arithmetic: the offsets `(32p + r)·D + j` of a block's lanes and the group index
`(32p + r) mod G` neither wrap nor leave their buffers. -/

/-! ## Integer lane arithmetic -/

theorem conv_i32_i64 {P : Platform} {x : I32} {r : I64} (h : ConvRel P .i32 .i64 x r) :
    r = IntV.ofInt x.toInt := by simpa [ConvRel, ofIntRel] using h

/-- A natural number below `2^31` converts to a 32-bit integer whose signed value is that number. -/
theorem toInt_ofNat32' {n : Nat} (hn : n < 2^31) :
    (IntV.ofNat n : I32).toInt = (n : Int) :=
  IntV.toInt_ofNat (w := 32) (by omega)

/-- An integer in the signed 32-bit range converts to a 32-bit integer with that signed value. -/
theorem toInt_ofInt32' {x : Int} (h1 : -(2^31 : Int) ≤ x) (h2 : x < 2^31) :
    (IntV.ofInt x : I32).toInt = x :=
  IntV.toInt_ofInt (w := 32) (by norm_num) (by simpa [IntV.SFits] using ⟨h1, h2⟩)

/-- A natural number below `2^62` converts to a 64-bit integer whose signed value is that number. -/
theorem toInt_ofNat64' {n : Nat} (hn : n < 2^62) :
    (IntV.ofNat n : I64).toInt = (n : Int) :=
  IntV.toInt_ofNat (w := 64) (by omega)

/-- An integer in the signed 64-bit range converts to a 64-bit integer with that signed value. -/
theorem toInt_ofInt64' {x : Int} (h1 : -(2^63 : Int) ≤ x) (h2 : x < 2^63) :
    (IntV.ofInt x : I64).toInt = x :=
  IntV.toInt_ofInt (w := 64) (by norm_num) (by simpa [IntV.SFits] using ⟨h1, h2⟩)

/-- If the exact sum of two 64-bit values fits the signed range, 64-bit addition gives that sum. -/
theorem toInt_add64' {a b : I64} (h1 : -(2^63 : Int) ≤ a.toInt + b.toInt)
    (h2 : a.toInt + b.toInt < 2^63) : (a + b).toInt = a.toInt + b.toInt :=
  IntV.toInt_add rfl rfl (by simpa [IntV.SFits] using ⟨h1, h2⟩)

/-- If the exact product of two 64-bit values fits the signed range, 64-bit multiplication gives that product. -/
theorem toInt_mul64' {a b : I64} (h1 : -(2^63 : Int) ≤ a.toInt * b.toInt)
    (h2 : a.toInt * b.toInt < 2^63) : (a * b).toInt = a.toInt * b.toInt :=
  IntV.toInt_mul rfl rfl (by simpa [IntV.SFits] using ⟨h1, h2⟩)

/-- Addition of 64-bit values with known signed values `x`, `y` yields `x + y` when it does not overflow. -/
theorem I64.add_toInt {a b : I64} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h1 : -(2^63 : Int) ≤ x + y) (h2 : x + y < 2^63) : (a + b).toInt = x + y := by
  rw [toInt_add64' (by rw [ha, hb]; exact h1) (by rw [ha, hb]; exact h2), ha, hb]

/-- Multiplying a 64-bit value of signed value `x` by the literal `n` yields `x * n` when it does not overflow. -/
theorem I64.mul_ofNat_toInt {a : I64} {n : Nat} {x : Int} (ha : a.toInt = x) (hn : n < 2^62)
    (h1 : -(2^63 : Int) ≤ x * (n : Int)) (h2 : x * (n : Int) < 2^63) :
    (a * (IntV.ofNat n : I64)).toInt = x * (n : Int) := by
  have hn' : (IntV.ofNat n : I64).toInt = (n : Int) := toInt_ofNat64' hn
  rw [toInt_mul64' (by rw [ha, hn']; exact h1) (by rw [ha, hn']; exact h2), ha, hn']

/-- Multiplying a 32-bit value of signed value `x` by the literal `n` yields `x * n` when it does not overflow. -/
theorem I32.mul_ofNat_toInt {a : I32} {n : Nat} {x : Int} (ha : a.toInt = x) (hn : n < 2^31)
    (h1 : -(2^31 : Int) ≤ x * (n : Int)) (h2 : x * (n : Int) < 2^31) :
    (a * (IntV.ofNat n : I32)).toInt = x * (n : Int) := by
  have hn' : (IntV.ofNat n : I32).toInt = (n : Int) := toInt_ofNat32' hn
  rw [IntV.toInt_mul ha hn' (by exact ⟨h1, h2⟩)]

/-- Adding a scalar to a tensor acts lane-wise: lane `i` of `b + x` is `b + x.get i`. -/
theorem get1_sadd {sg : Bool} {w : Nat} {s : Shape} (b : IntV sg w) (x : Tensor s (IntV sg w))
    (i : Idx s) : (b + x).get i = b + x.get i := rfl

/-- Integer-lane accessor lemmas, stated with `get1`/`get2` so that they rewrite. -/
theorem get2_eq_get {m n : Nat} {α : Type} (x : Tensor [m, n] α) (i : Fin m) (j : Fin n) :
    x.get2 i j = x.get (i, j, ()) := rfl

/-- Lane `(i, j)` of a column-broadcast tensor times a scalar is lane `i` of the tensor times the scalar. -/
theorem get2_muls {sg : Bool} {w m n : Nat} (x : Tensor [m] (IntV sg w)) (a : IntV sg w)
    (i : Fin m) (j : Fin n) : ((x.col n) * a).get2 i j = x.get1 i * a := rfl

/-- Lane `i` of the comparison with a scalar is the comparison of lane `i` with that scalar. -/
theorem get1_ltS {sg : Bool} {w n : Nat} (x : Tensor [n] (IntV sg w)) (a : IntV sg w)
    (i : Fin n) : (x.ltS a).get1 i = IntV.lt (x.get1 i) a := rfl

/-- Lane `(i, j)` of the conjunction of two boolean tensors is the conjunction of their lanes. -/
theorem get2_and {m n : Nat} (x y : Tensor [m, n] Bool) (i : Fin m) (j : Fin n) :
    (x.and y).get2 i j = (x.get2 i j && y.get2 i j) := rfl

/-- Lane `(i, j)` of a column broadcast is lane `i` of the source tensor. -/
theorem get2_col {m n : Nat} {α : Type} (x : Tensor [m] α) (i : Fin m) (j : Fin n) :
    (x.col n).get2 i j = x.get1 i := rfl

/-- Lane `(i, j)` of a row broadcast is lane `j` of the source tensor. -/
theorem get2_row {m n : Nat} {α : Type} (x : Tensor [n] α) (i : Fin m) (j : Fin n) :
    (x.row m).get2 i j = x.get1 j := rfl

/-- Lane `(i, j)` of a pointer tensor plus an offset tensor is the pointer lane offset by the integer offset lane. -/
theorem get2_ptrAddT {m n : Nat} {d : DType} {sg : Bool} {w : Nat}
    (ps : Tensor [m, n] (Ptr d)) (o : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    ((ps + o).get2 i j) = (ps.get2 i j).offset (o.get2 i j).toInt := rfl

/-- Lane `(i, j)` of a scalar pointer plus an offset tensor is the pointer offset by the integer offset lane. -/
theorem get2_ptrAdd {m n : Nat} {d : DType} {sg : Bool} {w : Nat}
    (p : Ptr d) (o : Tensor [m, n] (IntV sg w)) (i : Fin m) (j : Fin n) :
    ((p + o).get2 i j) = p.offset (o.get2 i j).toInt := rfl

/-- The `o_t` lane `32p + r`, as an integer. -/
theorem oT_toInt {p : Nat} (hp : p < 2^31) {r : Fin 32} {it : I64} {ar : Tensor [32] I64}
    (hit : it = IntV.ofInt (p : Int)) (har : ar.get1 r = IntV.ofInt (r : Int)) :
    ((it * (IntV.ofNat 32 : I64) + ar).get1 r).toInt = 32 * (p : Int) + (r : Int) := by
  have hrI : (0:Int) ≤ (r : Int) ∧ (r : Int) < 32 := by have := r.isLt; omega
  have h32 : (IntV.ofNat 32 : I64).toInt = (32 : Int) := toInt_ofNat64' (by norm_num)
  have hit' : it.toInt = (p : Int) := by rw [hit, toInt_ofInt64' (by omega) (by omega)]
  have har' : (ar.get1 r).toInt = (r : Int) := by
    rw [har, toInt_ofInt64' (by omega) (by omega)]
  have hmul : (it * (IntV.ofNat 32 : I64)).toInt = 32 * (p : Int) := by
    rw [toInt_mul64' (by rw [hit', h32]; omega) (by rw [hit', h32]; omega), hit', h32]
    ring
  have hsum : ((it * (IntV.ofNat 32 : I64)) + ar.get1 r).toInt
      = (it * (IntV.ofNat 32 : I64)).toInt + (ar.get1 r).toInt :=
    toInt_add64' (by rw [hmul, har']; omega) (by rw [hmul, har']; omega)
  show (((it * (IntV.ofNat 32 : I64)) + ar.get1 r).toInt) = _
  rw [hsum, hmul, har']

/-- The converted `arange` lane `r` is the integer `r`. -/
theorem ar_eq {P : Platform} {ar : Tensor [32] I64} (r : Fin 32)
    (har : ConvRel P .i32 .i64 (IntV.ofInt (0 + (r : Nat))) (ar.get1 r)) :
    ar.get1 r = IntV.ofInt ((r : Int)) := by
  rw [conv_i32_i64 har, toInt_ofInt32' (by omega) (by have := r.isLt; omega)]
  simp

/-- If a 32-bit value `p` converts to a 64-bit value `it`, then `it` is the 64-bit integer `p`. -/
theorem it_eq {P : Platform} {p : Nat} (hp : p < 2^31) {it : I64}
    (hit : ConvRel P .i32 .i64 (IntV.ofNat p) it) : it = IntV.ofInt (p : Int) := by
  rw [conv_i32_i64 hit, toInt_ofNat32' hp]

/-- `rem?` of a nonnegative value by a positive one. -/
theorem rem_toInt {a b q : I64} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt)
    (h : a.rem? b = some q) : q.toInt = a.toInt % b.toInt := by
  obtain ⟨q', hq', hv⟩ := IntV.rem?_of_nonneg (w := 64) ha hb
  rw [h] at hq'
  have hq : q = q' := (Option.some.injEq _ _).mp hq'
  rw [hq]; exact hv

/-- The `o_g` lane `(32p + r) % G`. -/
theorem oG_toInt {p G : Nat} (hG : 0 < G) (hG2 : G < 2 ^ 62) {r : Fin 32} {ot og : I64}
    (hot : ot.toInt = 32 * (p : Int) + (r : Int))
    (hrem : ot.rem? (IntV.ofNat G : I64) = some og) :
    og.toInt = (((32 * p + r) % G : Nat) : Int) := by
  have hg' : (IntV.ofNat G : I64).toInt = (G : Int) := toInt_ofNat64' (by omega)
  have ha : 0 ≤ ot.toInt := by rw [hot]; have := r.isLt; omega
  have hb : 0 < (IntV.ofNat G : I64).toInt := by rw [hg']; omega
  rw [rem_toInt ha hb hrem, hot, hg', Int.natCast_mod]
  push_cast
  ring

/-- The `o_d` lane `0 + j`, as an integer. -/
theorem oD_toInt {BD : Nat} (hBD : BD < 2^31) {od : Tensor [BD] I32} (j : Fin BD)
    (hod : od.get1 j = IntV.ofInt (0 + (j : Nat))) : (od.get1 j).toInt = (j : Int) := by
  rw [hod, toInt_ofInt32' (by omega) (by have := j.isLt; omega)]
  simp

/-- The `m_t` lane: `o_t < T`. -/
theorem mT_get {T p : Nat} (_hT : T < 2^31) {ot : Tensor [32] I64} (r : Fin 32) {c : I64}
    (hot : (ot.get1 r).toInt = 32 * (p : Int) + (r : Int)) (hc : c.toInt = (T : Int)) :
    (ot.ltS c).get1 r = decide (32 * p + r < T) := by
  rw [get1_ltS, IntV.lt_eq, hot, hc]
  exact decide_eq_decide.mpr (by omega)

/-- The `m_d` lane: `o_d < D`. -/
theorem mD_get {D BD : Nat} (hD : D < 2^31) {od : Tensor [BD] I32} (j : Fin BD)
    (hod : (od.get1 j).toInt = (j : Int)) :
    (od.ltS (IntV.ofNat D : I32)).get1 j = decide (j < D) := by
  rw [get1_ltS, IntV.lt_eq, hod, toInt_ofNat32' hD]
  exact decide_eq_decide.mpr (by omega)

/-- The `m_x` lane: both masks held. -/
theorem mX_get {T D p : Nat} {BD : Nat} {mt : Tensor [32] Bool} {md : Tensor [BD] Bool}
    (r : Fin 32) (j : Fin BD)
    (hmt : mt.get1 r = decide (32 * p + r < T)) (hmd : md.get1 j = decide (j < D)) :
    ((mt.col BD).and (md.row 32)).get2 r j = decide (32 * p + r < T ∧ j < D) := by
  rw [get2_and, get2_col, get2_row, hmt, hmd]
  by_cases h1 : 32 * p + r < T <;> by_cases h2 : j < D <;> simp [h1, h2]

/-! ## Pointer blocks -/

theorem get1_ptrAdd {n : Nat} {dy : DType} {sg : Bool} {w : Nat} (p : Ptr dy)
    (o : Tensor [n] (IntV sg w)) (i : Fin n) :
    ((p + o).get1 i) = p.offset (o.get1 i).toInt := rfl

/-- `(32p + r)·D` fits in `int64`. -/
theorem lane_mul_bound {d : Inst} (hd : d.ok) {p : Nat} (hp : p < 2^31) (r : Fin 32) :
    -(2^63 : Int) ≤ (32 * (p:Int) + (r:Int)) * (d.D:Int) ∧
      (32 * (p:Int) + (r:Int)) * (d.D:Int) < 2^63 := by
  have hle : d.D ≤ 512 := hd.d_le
  have hpos : 0 < d.D := hd.d_pos
  have hp1 : (p:Int) < 2^31 := by omega
  have hr1 : (r:Int) < 32 := by omega
  have hD1 : (d.D:Int) ≤ 512 := by omega
  have hp0 : (0:Int) ≤ (p:Int) := by omega
  have hr0 : (0:Int) ≤ (r:Int) := by omega
  have hD0 : (0:Int) ≤ (d.D:Int) := by omega
  constructor <;> nlinarith

/-- `((32p + r) mod G)·D` fits in `int64`. -/
theorem gidx_mul_bound {d : Inst} (hd : d.ok) {p : Nat} (_hp : p < 2^31) (r : Fin 32) :
    -(2^63 : Int) ≤ (((32 * p + r) % d.G : Nat) : Int) * (d.D:Int) ∧
      (((32 * p + r) % d.G : Nat) : Int) * (d.D:Int) < 2^63 := by
  have hle : d.D ≤ 512 := hd.d_le
  have hg : 0 < d.G := hd.g_pos
  have hgm : (32 * p + r) % d.G < d.G := Nat.mod_lt _ hg
  have hgle : d.G ≤ 2 ^ 16 := hd.g_le
  have hD0 : (0:Int) ≤ (d.D:Int) := by omega
  have hD1 : (d.D:Int) ≤ 512 := by omega
  have hG0 : (0:Int) ≤ (((32 * p + r) % d.G : Nat) : Int) := by
    exact_mod_cast Nat.zero_le ((32 * p + r) % d.G)
  have hG1 : (((32 * p + r) % d.G : Nat) : Int) ≤ 2 ^ 16 := by omega
  constructor <;> nlinarith

/-- The `[T, D]` block pointer `x.ptr 0 + o_t.col BD * D + o_d.row 32`, lane `(r, j)`. -/
theorem pmat_get {d : Inst} (hd : d.ok) {dy : DType} (B : Buf dy) (p : Nat)
    (hp : p < 2 ^ 31)
    {ot : Tensor [32] I64} {od : Tensor [d.BD] I32}
    (hot : ∀ r : Fin 32, (ot.get1 r).toInt = 32 * (p : Int) + (r : Int))
    (hod : ∀ j : Fin d.BD, (od.get1 j).toInt = (j : Int))
    (r : Fin 32) (j : Fin d.BD) :
    ((B.ptr 0 + ot.col d.BD * (IntV.ofNat d.D : I64)) + od.row 32).get2 r j
      = B.ptr (idx d (32 * p + r) j) := by
  have hpos : 0 < d.D := hd.d_pos
  have hle : d.D ≤ 512 := hd.d_le
  have hDv : (IntV.ofNat d.D : I64).toInt = (d.D : Int) := toInt_ofNat64' (by omega)
  have h32r : (ot.get1 r).toInt = 32 * (p : Int) + (r : Int) := hot r
  have hrng : 0 ≤ (r : Int) ∧ (r : Int) < 32 := by have := r.isLt; omega
  have hmul : (ot.get1 r * (IntV.ofNat d.D : I64)).toInt
      = (32 * (p : Int) + (r : Int)) * (d.D : Int) := by
    have hb := lane_mul_bound hd hp r
    rw [toInt_mul64' (by rw [h32r, hDv]; exact hb.1) (by rw [h32r, hDv]; exact hb.2), h32r, hDv]
  rw [get2_ptrAddT, get2_ptrAdd, get2_muls, get2_row, Ptr.offset_offset, Buf.ptr_offset]
  have hk : (0 : Int) + ((ot.get1 r * (IntV.ofNat d.D : I64)).toInt + (od.get1 j).toInt)
      = (idx d (32 * p + r) j : Int) := by
    rw [hmul, hod j]
    simp only [idx]
    push_cast
    ring
  rw [hk]

/-- The `[G·D]` block pointer `w.ptr 0 + o_g.col BD * D + o_d.row 32`, lane `(r, j)`. -/
theorem pvec_get {d : Inst} (hd : d.ok) {dy : DType} (B : Buf dy) (p : Nat)
    (hp : p < 2 ^ 31)
    {og : Tensor [32] I64} {od : Tensor [d.BD] I32}
    (hog : ∀ r : Fin 32, (og.get1 r).toInt = (((32 * p + r) % d.G : Nat) : Int))
    (hod : ∀ j : Fin d.BD, (od.get1 j).toInt = (j : Int))
    (r : Fin 32) (j : Fin d.BD) :
    ((B.ptr 0 + og.col d.BD * (IntV.ofNat d.D : I64)) + od.row 32).get2 r j
      = B.ptr (gidx d (32 * p + r) j) := by
  have hpos : 0 < d.D := hd.d_pos
  have hle : d.D ≤ 512 := hd.d_le
  have hG : 0 < d.G := hd.g_pos
  have hGle : d.G ≤ 2 ^ 16 := hd.g_le
  have hDv : (IntV.ofNat d.D : I64).toInt = (d.D : Int) := toInt_ofNat64' (by omega)
  have hgr : (og.get1 r).toInt = (((32 * p + r) % d.G : Nat) : Int) := hog r
  have hmul : (og.get1 r * (IntV.ofNat d.D : I64)).toInt
      = (((32 * p + r) % d.G : Nat) : Int) * (d.D : Int) := by
    have hb := gidx_mul_bound hd hp r
    rw [toInt_mul64' (by rw [hgr, hDv]; exact hb.1) (by rw [hgr, hDv]; exact hb.2), hgr, hDv]
  rw [get2_ptrAddT, get2_ptrAdd, get2_muls, get2_row, Ptr.offset_offset, Buf.ptr_offset]
  have hk : (0 : Int) + ((og.get1 r * (IntV.ofNat d.D : I64)).toInt + (od.get1 j).toInt)
      = (gidx d (32 * p + r) j : Int) := by
    rw [hmul, hod j]
    simp only [gidx]
    push_cast
    ring
  rw [hk]

/-- The `[T]` block pointer `mean.ptr 0 + o_t`, lane `r`. -/
theorem prow_get {dy : DType} (B : Buf dy) (p : Nat) (_hp : p < 2 ^ 31)
    {ot : Tensor [32] I64}
    (hot : ∀ r : Fin 32, (ot.get1 r).toInt = 32 * (p : Int) + (r : Int)) (r : Fin 32) :
    ((B.ptr 0 + ot).get1 r) = B.ptr (32 * p + r) := by
  have h32r : (ot.get1 r).toInt = 32 * (p : Int) + (r : Int) := hot r
  rw [get1_ptrAdd, Buf.ptr_offset]
  have hk : (0 : Int) + (ot.get1 r).toInt = (32 * (p : Int) + (r : Int)) := by
    rw [h32r]; ring
  rw [hk]
