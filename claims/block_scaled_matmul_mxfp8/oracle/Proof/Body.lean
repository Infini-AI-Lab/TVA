import Claim
import TVA.Proof
import Proof.Basic
import Proof.Inv
import Proof.Arith
import Proof.Num
import Proof.Off
import Proof.ScaleOff
import Proof.ScaleIdx
import Proof.Tile
import Proof.ScaledVal
import Proof.Step
import Proof.Lane
import Proof.Fill
import Proof.Glue
import Proof.Tail

/-!
# One step of the `K`-loop

The body of the kernel's `for` loop, at step `t` (`t < K / 128`), starting from a state that
satisfies `Inv` at `t`: it loads the `a` block, the `b` block and the two scale blocks, does the
scaled dot product into the accumulator, and advances the three offsets.

The step reads only (its write set is empty) and re-establishes `Inv` at `t + 1`.
-/

/-!
# The loads of one trip: the `a`, `b` and scale tiles read through their descriptors are the scaled values of the inputs, and the scales after the kernel's reshape and permute are the ones of the tile.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- `tl.dot_scaled` unpacks the left operand as the identity for a `K`-packed `e4m3` operand. -/
theorem unpackL_get2 {cfg : ScaledCfg} {M K R C : Nat} {d : DType}
    (raw : Tensor [R, C] d.Val) (i : Fin M) (k : Fin K) :
    (cfg.unpackL M K raw).get2 i k = cfg.lhsBits raw (i : Nat) (k : Nat) := by
  simp only [ScaledCfg.unpackL, Tensor.get2_of2]

/-- `tl.dot_scaled` unpacks the right operand as the identity for a `K`-packed `e4m3` operand. -/
theorem unpackR_get2 {cfg : ScaledCfg} {K N R C : Nat} {d : DType}
    (raw : Tensor [R, C] d.Val) (k : Fin K) (n : Fin N) :
    (cfg.unpackR K N raw).get2 k n = cfg.rhsBits raw (k : Nat) (n : Nat) := by
  simp only [ScaledCfg.unpackR, Tensor.get2_of2]

/-- The loaded `a` tile, unpacked as the dot does, holds the scaled values `xa` of the `mxfp8` operand. -/
theorem aTile_mx {d : Inst} {m : Mem} {pm t : Nat} {av : Tensor [128, 128] F8E4}
    (hav : ∀ (r c : Fin 128), av.get2 r c = xa d m (pm * 128 + (r : Nat)) (128 * t + (c : Nat)))
    (i k : Nat) (hi : i < 128) (hk : k < 128) :
    ScaledCfg.lhsBits (d := .f8e4nv) cfgMx av i k = (xa d m (pm * 128 + i) (128 * t + k)).pat := by
  simp only [ScaledCfg.lhsBits, cfgMx, Tensor.getD2, hi, hk, hav]
  rfl

/-- The left scales after the kernel's reshape and permute are the scale bytes of `a_scale` of rows and groups of the tile. -/
theorem scalesL_gen {d : Inst} (hd : d.ok) {m : Mem} {pm t : Nat} (hpm : pm < d.M / 128)
    (ht : t < d.K / 128) {sc : Tensor [128, 4] U8} {av : Tensor [1, 1, 1, 2, 256] U8}
    (hsc : ∀ (r : Fin 128) (g : Fin 4),
      sc.get2 r g = av.get (Idx.ofPos [1, 1, 1, 2, 256] (aU (r : Nat) (g : Nat)) (aU_lt r.isLt g.isLt)))
    (hav : ∀ l : Idx [1, 1, 1, 2, 256],
      (aScaleDesc d).inbAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
      m.read .u8 (d.a_scale.ptr (((aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
        = some (av.get l))
    {cfg : ScaledCfg} (hg : cfg.group = 32) :
    ∀ (i : Fin 128) (k : Fin 128),
      (ScaledCfg.scalesL (ds := .u8) cfg 128 128 sc).get2 i k =
        DType.toBits .u8 (sa d m (pm * 128 + (i : Nat)) (128 * t + (k : Nat))) := by
  intro i k
  have hk : (k : Nat) / 32 < 4 := by have := k.isLt; omega
  have h1 : sc.getD2 (i : Nat) ((k : Nat) / 32) = sc.get2 i ⟨(k : Nat) / 32, hk⟩ :=
    getD2_eq sc i.isLt hk
  refine Eq.trans (b := DType.toBits .u8 (sc.getD2 (i : Nat) ((k : Nat) / 32))) ?_ ?_
  · show DType.toBits .u8 (sc.getD2 (i : Nat) ((k : Nat) / cfg.group)) =
      DType.toBits .u8 (sc.getD2 (i : Nat) ((k : Nat) / 32))
    rw [hg]
  rw [h1, hsc i ⟨(k : Nat) / 32, hk⟩]
  refine Eq.trans
    (b := DType.toBits .u8 (sa d m (pm * 128 + (i : Nat)) (128 * t + 32 * ((k : Nat) / 32))))
    (aScale_val_at hd hpm ht hav (i : Nat) ((k : Nat) / 32) i.isLt hk) ?_
  exact congrArg (DType.toBits .u8)
    (sa_shift (d := d) (m := m) (i := pm * 128 + (i : Nat)) (t := t) (k := (k : Nat)) k.isLt)

/-- The right scales after the kernel's reshape and permute are the scale bytes of `b_scale` of columns and groups of the tile. -/
theorem scalesR_gen {d : Inst} (hd : d.ok) {m : Mem} {pn t : Nat} (ht : t < d.K / 128)
    {sc : Tensor [256, 4] U8} {bv : Tensor [1, 2, 1, 2, 256] U8}
    (hsc : ∀ (n : Fin 256) (g : Fin 4),
      sc.get2 n g = bv.get (Idx.ofPos [1, 2, 1, 2, 256] (bPos (n : Nat) (g : Nat)) (bPos_lt n.isLt g.isLt)))
    (hbv : ∀ l : Idx [1, 2, 1, 2, 256],
      (bScaleDesc d).inbAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
      m.read .u8 (d.b_scale.ptr (((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
        = some (bv.get l))
    {cfg : ScaledCfg} (hg : cfg.group = 32) (n : Fin 256)
    (hr : 2 * pn + (n : Nat) / 128 < d.N / 128) :
    ∀ (k : Fin 128),
      (ScaledCfg.scalesR (ds := .u8) cfg 128 256 sc).get2 k n =
        DType.toBits .u8 (sb d m (pn * 256 + (n : Nat)) (128 * t + (k : Nat))) := by
  intro k
  have hk : (k : Nat) / 32 < 4 := by have := k.isLt; omega
  have h1 : sc.getD2 (n : Nat) ((k : Nat) / 32) = sc.get2 n ⟨(k : Nat) / 32, hk⟩ :=
    getD2_eq sc n.isLt hk
  refine Eq.trans (b := DType.toBits .u8 (sc.getD2 (n : Nat) ((k : Nat) / 32))) ?_ ?_
  · show DType.toBits .u8 (sc.getD2 (n : Nat) ((k : Nat) / cfg.group)) =
      DType.toBits .u8 (sc.getD2 (n : Nat) ((k : Nat) / 32))
    rw [hg]
  rw [h1, hsc n ⟨(k : Nat) / 32, hk⟩]
  refine Eq.trans
    (b := DType.toBits .u8 (sb d m (pn * 256 + (n : Nat)) (128 * t + 32 * ((k : Nat) / 32))))
    (bScale_val_at hd ht hbv (n : Nat) ((k : Nat) / 32) n.isLt hk hr) ?_
  exact congrArg (DType.toBits .u8)
    (sb_shift (d := d) (m := m) (j := pn * 256 + (n : Nat)) (t := t) (k := (k : Nat)) k.isLt)


/-- The loaded `a` tile, in the `mixed` launch, holds the values `xa` of the `e4m3` operand. -/
theorem aTile_mixed {d : Inst} {m : Mem} {pm t : Nat} {av : Tensor [128, 128] F8E4}
    (hav : ∀ (r c : Fin 128), av.get2 r c = xa d m (pm * 128 + (r : Nat)) (128 * t + (c : Nat)))
    (i k : Nat) (hi : i < 128) (hk : k < 128) :
    ScaledCfg.lhsBits (d := .f8e4nv) cfgMixed av i k =
      (xa d m (pm * 128 + i) (128 * t + k)).pat := by
  simp only [ScaledCfg.lhsBits, cfgMixed, Tensor.getD2, hi, hk, hav]
  rfl

/-! ## The `b` descriptor at a symbolic column count `C` -/

theorem bDesc_elemAt_mixed_gen {C : Nat} {d : Inst} (hm : d.mixed = true) (pn t : Nat)
    (l : Idx [256, C]) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat =
      (pn * 256 + l.coord 0) * (d.K / 2) + (64 * t + l.coord 1) := by
  have hep : epbB true = 2 := by simp [epbB]
  have key : ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l))
      = (((pn * 256 + l.coord 0) * (d.K / 2) + (64 * t + l.coord 1) : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bDesc, hm, hep, List.length_cons,
      List.length_nil, List.range_succ, List.range_zero, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, TensorDesc.coord, List.getD_cons_zero, List.getD_cons_succ,
      List.nil_append, List.cons_append]
    push_cast
    ring
  rw [key, Int.toNat_natCast]

/-- A lane of `b`'s block in bounds (`mixed`, any column count) has its row and column below the tensor's. -/
theorem bDesc_inbAt_bnds_mixed_gen {C : Nat} {d : Inst} (hm : d.mixed = true) {pn t : Nat}
    (l : Idx [256, C])
    (hl : (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) = true) :
    pn * 256 + l.coord 0 < d.N ∧ 64 * t + l.coord 1 < d.K / 2 := by
  have hep : epbB true = 2 := by simp [epbB]
  rw [TensorDesc.inbAt, decide_eq_true_iff] at hl
  have h0 := hl 0 (by simp [TensorDesc.rank, bDesc, hm, hep])
  have h1 := hl 1 (by simp [TensorDesc.rank, bDesc, hm, hep])
  simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ] at h0 h1
  push_cast at h0 h1
  omega

/-- A lane of `b`'s block with row and column below the tensor's is in bounds (`mixed`). -/
theorem bDesc_inbAt_of_lt_mixed_gen {C : Nat} {d : Inst} (hm : d.mixed = true) {pn t : Nat}
    {l : Idx [256, C]}
    (hrow : pn * 256 + l.coord 0 < d.N) (hcol : 64 * t + l.coord 1 < d.K / 2) :
    (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) = true := by
  have hep : epbB true = 2 := by simp [epbB]
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk2 : k < 2 := by simpa [TensorDesc.rank, bDesc, hm, hep] using hk
  interval_cases k
  all_goals
    simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ]
    constructor <;> push_cast <;> omega

/-- An in-bounds lane of `b`'s block addresses an element below `N * K / 2` (`mixed`). -/
theorem bDesc_elemAt_lt_mixed_inb_gen {C : Nat} {d : Inst} (hm : d.mixed = true) {pn t : Nat}
    (l : Idx [256, C])
    (hl : (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) = true) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat <
      d.N * (d.K / 2) := by
  rw [bDesc_elemAt_mixed_gen hm pn t l]
  obtain ⟨hA, hB⟩ := bDesc_inbAt_bnds_mixed_gen hm l hl
  exact off_mul_add_lt_mul hA hB

/-- The element of `b` that a lane of its block addresses (`mxfp8`, any column count). -/
theorem bDesc_elemAt_mx_gen {C : Nat} {d : Inst} (hm : d.mixed = false) (pn t : Nat)
    (l : Idx [256, C]) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat =
      (pn * 256 + l.coord 0) * d.K + (128 * t + l.coord 1) := by
  have hep : epbB false = 1 := by simp [epbB]
  have key : ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l))
      = (((pn * 256 + l.coord 0) * d.K + (128 * t + l.coord 1) : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bDesc, hm, hep, List.length_cons,
      List.length_nil, List.range_succ, List.range_zero, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, TensorDesc.coord, List.getD_cons_zero, List.getD_cons_succ,
      List.nil_append, List.cons_append, Nat.div_one]
    push_cast
    ring
  rw [key, Int.toNat_natCast]

/-- A lane of `b`'s block in bounds (`mxfp8`) has its row and column below the tensor's. -/
theorem bDesc_inbAt_bnds_mx_gen {C : Nat} {d : Inst} (hm : d.mixed = false) {pn t : Nat}
    (l : Idx [256, C])
    (hl : (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true) :
    pn * 256 + l.coord 0 < d.N ∧ 128 * t + l.coord 1 < d.K := by
  have hep : epbB false = 1 := by simp [epbB]
  rw [TensorDesc.inbAt, decide_eq_true_iff] at hl
  have h0 := hl 0 (by simp [TensorDesc.rank, bDesc, hm, hep])
  have h1 := hl 1 (by simp [TensorDesc.rank, bDesc, hm, hep])
  simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ] at h0 h1
  push_cast at h0 h1
  omega

/-- A lane of `b`'s block with row and column below the tensor's is in bounds (`mxfp8`). -/
theorem bDesc_inbAt_of_lt_mx_gen {C : Nat} {d : Inst} (hm : d.mixed = false) {pn t : Nat}
    {l : Idx [256, C]}
    (hrow : pn * 256 + l.coord 0 < d.N) (hcol : 128 * t + l.coord 1 < d.K) :
    (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
  have hep : epbB false = 1 := by simp [epbB]
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk2 : k < 2 := by simpa [TensorDesc.rank, bDesc, hm, hep] using hk
  interval_cases k
  all_goals
    simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ,
      Nat.div_one]
    constructor <;> push_cast <;> omega

/-- An in-bounds lane of `b`'s block addresses an element below `N * K` (`mxfp8`). -/
theorem bDesc_elemAt_lt_mx_inb_gen {C : Nat} {d : Inst} (hm : d.mixed = false) {pn t : Nat}
    (l : Idx [256, C])
    (hl : (bDesc d).inbAt
      (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat <
      d.N * d.K := by
  rw [bDesc_elemAt_mx_gen hm pn t l]
  obtain ⟨hA, hB⟩ := bDesc_inbAt_bnds_mx_gen hm l hl
  exact off_mul_add_lt_mul hA hB

/-! ## Read-backs at a symbolic column count `C` -/

theorem bTile_read_mixed_gen {d : Inst} (hd : d.ok) (hm : d.mixed = true) {m : Mem} {pn t : Nat}
    (ht : t < d.K / 128) {C : Nat} (hC : C = 64) {bv : Tensor [256, C] ((bDT d.mixed).Val)}
    (hbv : ∀ i : Idx [256, C],
      (bDesc d).inbAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] i) = true →
      m.read (bDT d.mixed) (d.b.ptr (((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] i)).toNat : Int)).addr
        = some (bv.get i)) :
    ∀ (n : Fin 256), pn * 256 + (n : Nat) < d.N → ∀ j : Fin C,
      DType.toBits (bDT d.mixed) (bv.get2 n j) =
        rdBits m d.b ((pn * 256 + (n : Nat)) * (d.K / 2) + (64 * t + (j : Nat))) := by
  intro n hrow j
  have hj : (j : Nat) < 64 := hC ▸ j.isLt
  have hrow' : pn * 256 + Idx.coord (s := [256, C]) ((n, j, ()) : Idx [256, C]) 0 < d.N := by
    simpa only [Idx.coord] using hrow
  have hcol : 64 * t + Idx.coord (s := [256, C]) ((n, j, ()) : Idx [256, C]) 1 < d.K / 2 := by
    simpa only [Idx.coord] using bcol_lt hd ht hj
  have hinb := bDesc_inbAt_of_lt_mixed_gen (d := d) hm (l := ((n, j, ()) : Idx [256, C])) hrow' hcol
  have hread := hbv ((n, j, ()) : Idx [256, C]) hinb
  have helem : ((bDesc d).elemAt
        (TensorDesc.coord (bs := [256, C]) [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)]
          ((n, j, ()) : Idx [256, C]))).toNat =
      (pn * 256 + (n : Nat)) * (d.K / 2) + (64 * t + (j : Nat)) := by
    simpa only [Idx.coord] using
      bDesc_elemAt_mixed_gen (d := d) (C := C) hm pn t ((n, j, ()) : Idx [256, C])
  rw [helem] at hread
  unfold rdBits
  rw [hread]
  rfl

/-- The loaded `b` tile (`mxfp8`) holds the values of `b` at the tile's rows and columns. -/
theorem bTile_read_mx_gen {d : Inst} (_hd : d.ok) (hm : d.mixed = false) {m : Mem} {pn t : Nat}
    (ht : t < d.K / 128) {C : Nat} (hC : C = 128) {bv : Tensor [256, C] ((bDT d.mixed).Val)}
    (hbv : ∀ i : Idx [256, C],
      (bDesc d).inbAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] i) = true →
      m.read (bDT d.mixed) (d.b.ptr (((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] i)).toNat : Int)).addr
        = some (bv.get i)) :
    ∀ (n : Fin 256), pn * 256 + (n : Nat) < d.N → ∀ j : Fin C,
      DType.toBits (bDT d.mixed) (bv.get2 n j) =
        rdBits m d.b ((pn * 256 + (n : Nat)) * d.K + (128 * t + (j : Nat))) := by
  intro n hrow j
  have hj : (j : Nat) < 128 := hC ▸ j.isLt
  have hrow' : pn * 256 + Idx.coord (s := [256, C]) ((n, j, ()) : Idx [256, C]) 0 < d.N := by
    simpa only [Idx.coord] using hrow
  have hcol : 128 * t + Idx.coord (s := [256, C]) ((n, j, ()) : Idx [256, C]) 1 < d.K := by
    rw [Nat.mul_comm 128 t]
    simpa only [Idx.coord] using
      off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := d.K) (l := (j : Nat)) ht hj
  have hinb := bDesc_inbAt_of_lt_mx_gen (d := d) hm (l := ((n, j, ()) : Idx [256, C])) hrow' hcol
  have hread := hbv ((n, j, ()) : Idx [256, C]) hinb
  have helem : ((bDesc d).elemAt
        (TensorDesc.coord (bs := [256, C]) [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)]
          ((n, j, ()) : Idx [256, C]))).toNat =
      (pn * 256 + (n : Nat)) * d.K + (128 * t + (j : Nat)) := by
    simpa only [Idx.coord] using
      bDesc_elemAt_mx_gen (d := d) (C := C) hm pn t ((n, j, ()) : Idx [256, C])
  rw [helem] at hread
  unfold rdBits
  rw [hread]
  rfl

/-- Column `n` of the loaded `b` tile (`mixed`), unpacked, holds the `e2m1` values `xb` of `b`. -/
theorem bTile_mixed_col_gen {C : Nat} {d : Inst} (hm : d.mixed = true) {m : Mem} {pn t : Nat}
    (hC : C = 64) {bv : Tensor [256, C] U8} (n : Fin 256)
    (hbv : ∀ (j : Fin C), DType.toBits .u8 (bv.get2 n j) =
        rdBits m d.b ((pn * 256 + (n : Nat)) * (d.K / 2) + (64 * t + (j : Nat))))
    (kk : Nat) (hkk : kk < 128) :
    ScaledCfg.rhsBits (d := .u8) cfgMixed bv.trans kk (n : Nat) =
      xbBits d m (pn * 256 + (n : Nat)) (128 * t + kk) := by
  have h1 : kk / 2 < C := by rw [hC]; omega
  have h2 : (n : Nat) < 256 := n.isLt
  have hdiv : (128 * t + kk) / 2 = 64 * t + kk / 2 := by omega
  have hmod : (128 * t + kk) % 2 = kk % 2 := by omega
  simp [ScaledCfg.rhsBits, cfgMixed, Tensor.getD2, h1, h2, hbv, xbBits, hm, hdiv, hmod]

/-- Column `n` of the loaded `b` tile (`mxfp8`) holds the values `xb` of `b`. -/
theorem bTile_mx_col_gen {C : Nat} {d : Inst} (hm : d.mixed = false) {m : Mem} {pn t : Nat}
    (hC : C = 128) {bv : Tensor [256, C] F8E4} (n : Fin 256)
    (hbv : ∀ (j : Fin C), (bv.get2 n j).pat =
        rdBits m d.b ((pn * 256 + (n : Nat)) * d.K + (128 * t + (j : Nat))))
    (kk : Nat) (hkk : kk < 128) :
    ScaledCfg.rhsBits (d := .f8e4nv) cfgMx bv.trans kk (n : Nat) =
      xbBits d m (pn * 256 + (n : Nat)) (128 * t + kk) := by
  have h2 : (n : Nat) < 256 := n.isLt
  have hbit : ∀ x : F8E4, DType.toBits .f8e4nv x = x.pat := fun _ => rfl
  simp [ScaledCfg.rhsBits, cfgMx, Tensor.getD2, hC, hkk, h2, hbit, hbv, xbBits, hm]



/-- The right scaled value of a finite `mxfp8` element with an in-domain scale is `scaled x s`. -/
theorem scaledValue_rhs_mx {d : Inst} {m : Mem} {j k : Nat} {s : U8} (hm : d.mixed = false)
    (hfin : bFin d m j k = true) (hd : Device.e8m0Dom s.bits.toNat) :
    scaledValue cfgMx.rhs cfgMx.rhsScale (xbBits d m j k) (some (DType.toBits .u8 s)) =
      some (scaled (bVal d m j k) s) := by
  have h := scaledValue_rhs (d := d) (m := m) (j := j) (k := k) (s := s) hfin hd
  simpa [cfgMx, hm] using h

/-- The right scaled value of a finite `e2m1` element with an in-domain scale is `scaled x s`. -/
theorem scaledValue_rhs_mixed {d : Inst} {m : Mem} {j k : Nat} {s : U8} (hm : d.mixed = true)
    (hfin : bFin d m j k = true) (hd : Device.e8m0Dom s.bits.toNat) :
    scaledValue cfgMixed.rhs cfgMixed.rhsScale (xbBits d m j k) (some (DType.toBits .u8 s)) =
      some (scaled (bVal d m j k) s) := by
  have h := scaledValue_rhs (d := d) (m := m) (j := j) (k := k) (s := s) hfin hd
  simpa [cfgMixed, hm] using h

/-- **One step of the loop.** -/
theorem body_safe (P : Platform) {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) (hdot : Device.DotScaledE4m3 P)
    (hdotm : Device.DotScaledE4m3E2m1 P) (htot : Device.DotScaledTotal P cfgMx)
    (htotm : Device.DotScaledTotal P cfgMixed) {t : Nat} (ht : t < d.K / 128) {m : Mem}
    (hm : m = init) {acc : Tensor [128, 256] F32} {ka kb ks am bn sm sn : I32}
    (hacc : ∀ l : Idx [128, 256], laneMask d i l = true →
      (acc.get l).isFinite = true ∧
        |(acc.get l).toRat - partSum (prods d init (laneRow d i l) (laneCol d i l)) t| ≤
          tol (prods d init (laneRow d i l) (laneCol d i l)) t)
    (ham : am.toInt = (((i : Nat) % gridM d) * 128 : Nat))
    (hbn : bn.toInt = (((i : Nat) / gridM d) * 256 : Nat))
    (hsm : sm.toInt = (((i : Nat) % gridM d : Nat) : Int))
    (hsn : sn.toInt = ((2 * ((i : Nat) / gridM d) : Nat) : Int))
    (hka : ka.toInt = ((128 * t : Nat) : Int))
    (hkb : kb.toInt = (((128 / epbB d.mixed) * t : Nat) : Int))
    (hks : ks.toInt = ((t : Nat) : Int)) :
    Safe ((launch d).env P) (R d i) ∅ (fun st' m₂ => Inv d init i (t + 1) st' m₂) m
      (do
        let a ← tl.desc_load (bs := [128, 128 / 1]) (aDesc d) [am, ka]
        let b ← tl.desc_load (bs := [256, 128 / epbB d.mixed]) (bDesc d) [bn, kb]
        let scale_a ← tl.desc_load (bs := [1, 1, 1, 2, 256]) (aScaleDesc d)
          [(IntV.ofInt 0 : I32), sm, ks, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)]
        let scale_b ← tl.desc_load (bs := [1, 2, 1, 2, 256]) (bScaleDesc d)
          [(IntV.ofInt 0 : I32), sn, ks, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)]
        let scale_a := ((scale_a.reshape [1, 1, 32, 4, 4]
          (by simp only [List.prod_cons, List.prod_nil]; ring)).permute [0, 3, 2, 1, 4]).reshape
            [128, 128 / 32] rfl
        let scale_b := ((scale_b.reshape [2, 1, 32, 4, 4]
          (by simp only [List.prod_cons, List.prod_nil]; ring)).permute [0, 3, 2, 1, 4]).reshape
            [256, 128 / 32] rfl
        let accumulator ←
          if 1 = 1 ∧ epbB d.mixed = 2 then
            tl.dot_scaled cfgMixed 128 256 128 a (some scale_a) b.trans (some scale_b) acc
          else if 1 = 2 ∧ epbB d.mixed = 2 then
            tl.dot_scaled cfgFp4 128 256 128 a (some scale_a) b.trans (some scale_b) acc
          else
            tl.dot_scaled cfgMx 128 256 128 a (some scale_a) b.trans (some scale_b) acc
        let ka' := ka + (IntV.ofNat (128 / 1) : I32)
        let kb' := kb + (IntV.ofNat (128 / epbB d.mixed) : I32)
        let ks' := ks + (IntV.ofNat 1 : I32)
        return (accumulator, ka', kb', ks')) := by
  rw [hm]
  rcases d with ⟨mixed, M, N, K, a, b, a_scale, b_scale, c⟩
  cases mixed
  case false =>
    let dd : Inst := ⟨false, M, N, K, a, b, a_scale, b_scale, c⟩
    have hd' : dd.ok := hd
    have hpre' : Pre dd init := hpre
    have ht' : t < dd.K / 128 := ht
    have hmix : dd.mixed = false := rfl
    have hep : epbB dd.mixed = 1 := rfl
    have hbs : (128 : Nat) / epbB dd.mixed = 128 := by rw [hep]
    have heK : dd.K / epbB dd.mixed = dd.K := by rw [hep, Nat.div_one]
    let pm : Nat := (i : Nat) % gridM dd
    let pn : Nat := (i : Nat) / gridM dd
    have hpm : pm < dd.M / 128 := by
      have h := Nat.mod_lt (i : Nat) (gridM_pos hd')
      have hg : gridM dd = dd.M / 128 := gridM_eq hd'
      omega
    have e0 : IntV.toInt (IntV.ofInt 0 : I32) = (0 : Int) := ofInt_toInt (by norm_num)
    have e1 : IntV.toInt am = ((pm * 128 : Nat) : Int) := by rw [ham]
    have e2 : IntV.toInt ka = ((128 * t : Nat) : Int) := hka
    have e3 : IntV.toInt bn = ((pn * 256 : Nat) : Int) := by rw [hbn]
    have e4 : IntV.toInt kb = ((128 * t : Nat) : Int) := by rw [hkb, hbs]
    have e5 : IntV.toInt sm = ((pm : Nat) : Int) := by rw [hsm]
    have e6 : IntV.toInt sn = ((2 * pn : Nat) : Int) := by rw [hsn]
    have e7 : IntV.toInt ks = ((t : Nat) : Int) := hks
    have hvals : ValsOk dd init := hpre'.2.2.2.2.2
    refine safe_desc_load (b := dd.a) (bs := [128, 128]) (D := aDesc dd) (offsets := [am, ka])
      rfl ?_ ?_ ?_ ?_ ?_
    · simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using aDesc_access hd' pm t
    · intro l hl
      have hl' : (aDesc dd).inbAt
          (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hl
      simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using aDesc_hin hd' pm t l hpm ht'
    · exact hpre'.1
    · rintro x ⟨j, ⟨l, hl, rfl⟩, hspan⟩
      have hl' : (aDesc dd).inbAt
          (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hl
      have hs' : x ∈ Mem.span .f8e4nv
          (dd.a.ptr (((aDesc dd).elemAt (TensorDesc.coord
            [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat : Int)).addr := by
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hspan
      show x ∈ dd.a.elems {e | e < dd.M * dd.K} ∪
        dd.b.elems {e | e < dd.N * (dd.K / epbB dd.mixed)} ∪
        dd.a_scale.elems {e | e < dd.M / 128 * (dd.K / 128) * 512} ∪
        dd.b_scale.elems {e | e < dd.N / 128 * (dd.K / 128) * 512}
      exact Or.inl (Or.inl (Or.inl ⟨_, aDesc_elemAt_lt hd' hpm ht' l, hs'⟩))
    · intro ta hread hpad
      have havr : ∀ (r cc : Fin 128), ta.get2 r cc =
          xa dd init (pm * 128 + (r : Nat)) (128 * t + (cc : Nat)) := by
        refine aTile_read (d := dd) hd' hpm ht' ?_
        intro l hl
        have hl' : (aDesc dd).inbAt
            (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hl
        have hr := hread l (by simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hl)
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hr
      refine safe_desc_load (b := dd.b) (bs := [256, 128 / epbB dd.mixed]) (D := bDesc dd)
        (offsets := [bn, kb]) rfl ?_ ?_ ?_ ?_ ?_
      · refine ⟨(bDesc_access_mx (d := dd) hd' hmix pn t).1, rfl, rfl, rfl, ?_⟩
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using
          (bDesc_access_mx (d := dd) hd' hmix pn t).2.2.2.2
      · intro l hl
        have hl' : (bDesc dd).inbAt
            (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hl
        have hlt : ((bDesc dd).elemAt (TensorDesc.coord
            [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat <
            dd.N * (dd.K / epbB dd.mixed) := by
          have h := bDesc_elemAt_lt_mx_inb_gen (d := dd) (C := 128 / epbB dd.mixed) hmix l hl'
          rwa [← heK] at h
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using Nat.lt_of_lt_of_le hlt hd'.b_len
      · exact hpre'.2.1
      · rintro x ⟨j, ⟨l, hl, rfl⟩, hspan⟩
        have hl' : (bDesc dd).inbAt
            (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hl
        have hlt : ((bDesc dd).elemAt (TensorDesc.coord
            [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat <
            dd.N * (dd.K / epbB dd.mixed) := by
          have h := bDesc_elemAt_lt_mx_inb_gen (d := dd) (C := 128 / epbB dd.mixed) hmix l hl'
          rwa [← heK] at h
        have hs' : x ∈ Mem.span (bDT dd.mixed)
            (dd.b.ptr (((bDesc dd).elemAt (TensorDesc.coord
              [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat : Int)).addr := by
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hspan
        show x ∈ dd.a.elems {e | e < dd.M * dd.K} ∪
          dd.b.elems {e | e < dd.N * (dd.K / epbB dd.mixed)} ∪
          dd.a_scale.elems {e | e < dd.M / 128 * (dd.K / 128) * 512} ∪
          dd.b_scale.elems {e | e < dd.N / 128 * (dd.K / 128) * 512}
        exact Or.inl (Or.inl (Or.inr ⟨_, hlt, hs'⟩))
      · intro tb hreadb hpadb
        have hbvr : ∀ (n : Fin 256), pn * 256 + (n : Nat) < dd.N →
            ∀ j : Fin (128 / epbB dd.mixed),
              DType.toBits (bDT dd.mixed) (tb.get2 n j) =
                rdBits init dd.b ((pn * 256 + (n : Nat)) * dd.K + (128 * t + (j : Nat))) :=
          bTile_read_mx_gen (d := dd) hd' hmix ht' (m := init) (C := 128 / epbB dd.mixed) hbs
            (bv := tb) (fun i hi => by
              have hr := hreadb i (by
                simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hi)
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hr)
        refine safe_desc_load (b := dd.a_scale) (bs := [1, 1, 1, 2, 256]) (D := aScaleDesc dd)
          (offsets := [(IntV.ofInt 0 : I32), sm, ks, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)])
          rfl ?_ ?_ ?_ ?_ ?_
        · simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using
            aScaleDesc_access (d := dd) hd' pm t
        · intro l hl
          have hl' : (aScaleDesc dd).inbAt
              (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true := by
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hl
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using
            aScaleDesc_hin (d := dd) hd' pm t l hpm ht'
        · exact hpre'.2.2.1
        · rintro x ⟨j, ⟨l, hl, rfl⟩, hspan⟩
          have hl' : (aScaleDesc dd).inbAt
              (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true := by
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hl
          have hs' : x ∈ Mem.span .u8
              (dd.a_scale.ptr (((aScaleDesc dd).elemAt (TensorDesc.coord
                [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr := by
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hspan
          show x ∈ dd.a.elems {e | e < dd.M * dd.K} ∪
            dd.b.elems {e | e < dd.N * (dd.K / epbB dd.mixed)} ∪
            dd.a_scale.elems {e | e < dd.M / 128 * (dd.K / 128) * 512} ∪
            dd.b_scale.elems {e | e < dd.N / 128 * (dd.K / 128) * 512}
          exact Or.inl (Or.inr ⟨_, aScaleDesc_elemAt_lt hd' hpm ht' l, hs'⟩)
        · intro tsa hreada hpada
          have havrS : ∀ l : Idx [1, 1, 1, 2, 256],
              (aScaleDesc dd).inbAt
                (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
              init.read .u8 (dd.a_scale.ptr (((aScaleDesc dd).elemAt (TensorDesc.coord
                [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
                = some (tsa.get l) := by
            intro l hl
            have hr := hreada l (by
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hl)
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hr
          refine safe_desc_load (b := dd.b_scale) (bs := [1, 2, 1, 2, 256]) (D := bScaleDesc dd)
            (offsets := [(IntV.ofInt 0 : I32), sn, ks, (IntV.ofInt 0 : I32),
              (IntV.ofInt 0 : I32)]) rfl ?_ ?_ ?_ ?_ ?_
          · simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using
              bScaleDesc_access (d := dd) hd' pn t
          · intro l hl
            have hl' : (bScaleDesc dd).inbAt
                (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) =
                true := by
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hl
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using Nat.lt_of_lt_of_le
              (bScaleDesc_elemAt_lt_inb (d := dd) hd' (pn := pn) (t := t) l hl') hd'.bs_len
          · exact hpre'.2.2.2.1
          · rintro x ⟨j, ⟨l, hl, rfl⟩, hspan⟩
            have hl' : (bScaleDesc dd).inbAt
                (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) =
                true := by
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hl
            have hs' : x ∈ Mem.span .u8
                (dd.b_scale.ptr (((bScaleDesc dd).elemAt (TensorDesc.coord
                  [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr := by
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hspan
            show x ∈ dd.a.elems {e | e < dd.M * dd.K} ∪
              dd.b.elems {e | e < dd.N * (dd.K / epbB dd.mixed)} ∪
              dd.a_scale.elems {e | e < dd.M / 128 * (dd.K / 128) * 512} ∪
              dd.b_scale.elems {e | e < dd.N / 128 * (dd.K / 128) * 512}
            exact Or.inr ⟨_, bScaleDesc_elemAt_lt_inb (d := dd) hd' (pn := pn) (t := t) l hl',
              hs'⟩
          · intro tsb hreadb hpaddb
            have hbvrS : ∀ l : Idx [1, 2, 1, 2, 256],
                (bScaleDesc dd).inbAt
                  (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
                init.read .u8 (dd.b_scale.ptr (((bScaleDesc dd).elemAt (TensorDesc.coord
                  [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
                  = some (tsb.get l) := by
              intro l hl
              have hr := hreadb l (by
                simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hl)
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hr
            dsimp only
            rw [ite_eq_right (by rw [hep]; simp), ite_eq_right (by simp)]
            refine dot_tail_safe (d := dd) P hd' hpre' i ht' (m := init) rfl hacc hka hkb hks
              ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
            · simpa only [hbs] using shapeOk_mx
            · simpa only [bDT, hmix] using typesOk_mx
            · exact htot
            · exact hdot false
            · intro r k
              have hrow : pm * 128 + (r : Nat) < dd.M :=
                off_mul_add_lt_of_lt_div (p := pm) (c := 128) (M := dd.M) (l := (r : Nat))
                  hpm r.isLt
              have hcol : 128 * t + (k : Nat) < dd.K := by
                rw [Nat.mul_comm 128 t]
                exact off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := dd.K) (l := (k : Nat))
                  ht' k.isLt
              rw [unpackL_get2 (cfg := cfgMx) (M := 128) (K := 128) (d := .f8e4nv) ta r k]
              rw [aTile_mx (d := dd) (m := init) (pm := pm) (t := t) (av := ta) havr r k
                r.isLt k.isLt]
              rw [scalesL_gen (d := dd) hd' hpm ht'
                (hsc := aScale_reshape (by simp only [List.prod_cons, List.prod_nil]; ring) rfl)
                (hav := havrS) (cfg := cfgMx) (by rfl) r k]
              exact scaledValue_lhs (x := xa dd init (pm * 128 + (r : Nat))
                (128 * t + (k : Nat))) (hvals.a_dom _ _ hrow hcol).1
                (hvals.a_dom _ _ hrow hcol).2.1
            · intro k n hn
              have hrow : pn * 256 + (n : Nat) < dd.N := hn
              have hcol : 128 * t + (k : Nat) < dd.K := by
                rw [Nat.mul_comm 128 t]
                exact off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := dd.K) (l := (k : Nat))
                  ht' k.isLt
              rw [unpackR_get2 (cfg := cfgMx) (K := 128) (N := 256) (d := bDT dd.mixed)
                tb.trans k n]
              have hbmx : cfgMx.rhsBits (d := .f8e4nv) tb.trans (k : Nat) (n : Nat) =
                  xbBits dd init (pn * 256 + (n : Nat)) (128 * t + (k : Nat)) :=
                bTile_mx_col_gen (d := dd) hmix (m := init) (C := 128 / epbB dd.mixed) hbs
                  (bv := tb) n (fun j => hbvr n hn j) (k : Nat) k.isLt
              rw [show cfgMx.rhsBits (d := bDT dd.mixed) tb.trans (k : Nat) (n : Nat) =
                  xbBits dd init (pn * 256 + (n : Nat)) (128 * t + (k : Nat)) from hbmx]
              rw [scalesR_gen (d := dd) hd' ht'
                (hsc := bScale_reshape (by simp only [List.prod_cons, List.prod_nil]; ring) rfl)
                (hbv := hbvrS) (cfg := cfgMx) (by rfl) n (bRow_lt hd' hn) k]
              exact scaledValue_rhs_mx (d := dd) (m := init) hmix
                (hvals.b_dom _ _ hrow hcol).1 (hvals.b_dom _ _ hrow hcol).2.1
            · intro r k
              exact scalesL_gen (d := dd) hd' hpm ht'
                (hsc := aScale_reshape (by simp only [List.prod_cons, List.prod_nil]; ring) rfl)
                (hav := havrS) (cfg := cfgMx) (by rfl) r k
            · intro k n hn
              exact scalesR_gen (d := dd) hd' ht'
                (hsc := bScale_reshape (by simp only [List.prod_cons, List.prod_nil]; ring) rfl)
                (hbv := hbvrS) (cfg := cfgMx) (by rfl) n (bRow_lt hd' hn) k
  case true =>
    let dd : Inst := ⟨true, M, N, K, a, b, a_scale, b_scale, c⟩
    have hd' : dd.ok := hd
    have hpre' : Pre dd init := hpre
    have ht' : t < dd.K / 128 := ht
    have hmix : dd.mixed = true := rfl
    have hep : epbB dd.mixed = 2 := rfl
    have hbs : (128 : Nat) / epbB dd.mixed = 64 := by rw [hep]
    have heK : dd.K / epbB dd.mixed = dd.K / 2 := by rw [hep]
    let pm : Nat := (i : Nat) % gridM dd
    let pn : Nat := (i : Nat) / gridM dd
    have hpm : pm < dd.M / 128 := by
      have h := Nat.mod_lt (i : Nat) (gridM_pos hd')
      have hg : gridM dd = dd.M / 128 := gridM_eq hd'
      omega
    have e0 : IntV.toInt (IntV.ofInt 0 : I32) = (0 : Int) := ofInt_toInt (by norm_num)
    have e1 : IntV.toInt am = ((pm * 128 : Nat) : Int) := by rw [ham]
    have e2 : IntV.toInt ka = ((128 * t : Nat) : Int) := hka
    have e3 : IntV.toInt bn = ((pn * 256 : Nat) : Int) := by rw [hbn]
    have e4 : IntV.toInt kb = ((64 * t : Nat) : Int) := by rw [hkb, hbs]
    have e5 : IntV.toInt sm = ((pm : Nat) : Int) := by rw [hsm]
    have e6 : IntV.toInt sn = ((2 * pn : Nat) : Int) := by rw [hsn]
    have e7 : IntV.toInt ks = ((t : Nat) : Int) := hks
    have hvals : ValsOk dd init := hpre'.2.2.2.2.2
    refine safe_desc_load (b := dd.a) (bs := [128, 128]) (D := aDesc dd) (offsets := [am, ka])
      rfl ?_ ?_ ?_ ?_ ?_
    · simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using aDesc_access hd' pm t
    · intro l hl
      have hl' : (aDesc dd).inbAt
          (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hl
      simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using aDesc_hin hd' pm t l hpm ht'
    · exact hpre'.1
    · rintro x ⟨j, ⟨l, hl, rfl⟩, hspan⟩
      have hl' : (aDesc dd).inbAt
          (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hl
      have hs' : x ∈ Mem.span .f8e4nv
          (dd.a.ptr (((aDesc dd).elemAt (TensorDesc.coord
            [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat : Int)).addr := by
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hspan
      show x ∈ dd.a.elems {e | e < dd.M * dd.K} ∪
        dd.b.elems {e | e < dd.N * (dd.K / epbB dd.mixed)} ∪
        dd.a_scale.elems {e | e < dd.M / 128 * (dd.K / 128) * 512} ∪
        dd.b_scale.elems {e | e < dd.N / 128 * (dd.K / 128) * 512}
      exact Or.inl (Or.inl (Or.inl ⟨_, aDesc_elemAt_lt hd' hpm ht' l, hs'⟩))
    · intro ta hread hpad
      have havr : ∀ (r cc : Fin 128), ta.get2 r cc =
          xa dd init (pm * 128 + (r : Nat)) (128 * t + (cc : Nat)) := by
        refine aTile_read (d := dd) hd' hpm ht' ?_
        intro l hl
        have hl' : (aDesc dd).inbAt
            (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l) = true := by
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hl
        have hr := hread l (by simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hl)
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e1, e2] using hr
      refine safe_desc_load (b := dd.b) (bs := [256, 128 / epbB dd.mixed]) (D := bDesc dd)
        (offsets := [bn, kb]) rfl ?_ ?_ ?_ ?_ ?_
      · refine ⟨(bDesc_access_mixed (d := dd) hd' hmix pn t).1, rfl, rfl, rfl, ?_⟩
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using
          (bDesc_access_mixed (d := dd) hd' hmix pn t).2.2.2.2
      · intro l hl
        have hl' : (bDesc dd).inbAt
            (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) = true := by
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hl
        have hlt : ((bDesc dd).elemAt (TensorDesc.coord
            [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat <
            dd.N * (dd.K / epbB dd.mixed) := by
          have h := bDesc_elemAt_lt_mixed_inb_gen (d := dd) (C := 128 / epbB dd.mixed) hmix l hl'
          rwa [← heK] at h
        simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using Nat.lt_of_lt_of_le hlt hd'.b_len
      · exact hpre'.2.1
      · rintro x ⟨j, ⟨l, hl, rfl⟩, hspan⟩
        have hl' : (bDesc dd).inbAt
            (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) = true := by
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hl
        have hlt : ((bDesc dd).elemAt (TensorDesc.coord
            [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat <
            dd.N * (dd.K / epbB dd.mixed) := by
          have h := bDesc_elemAt_lt_mixed_inb_gen (d := dd) (C := 128 / epbB dd.mixed) hmix l hl'
          rwa [← heK] at h
        have hs' : x ∈ Mem.span (bDT dd.mixed)
            (dd.b.ptr (((bDesc dd).elemAt (TensorDesc.coord
              [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat : Int)).addr := by
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hspan
        show x ∈ dd.a.elems {e | e < dd.M * dd.K} ∪
          dd.b.elems {e | e < dd.N * (dd.K / epbB dd.mixed)} ∪
          dd.a_scale.elems {e | e < dd.M / 128 * (dd.K / 128) * 512} ∪
          dd.b_scale.elems {e | e < dd.N / 128 * (dd.K / 128) * 512}
        exact Or.inl (Or.inl (Or.inr ⟨_, hlt, hs'⟩))
      · intro tb hreadb hpadb
        have hbvr : ∀ (n : Fin 256), pn * 256 + (n : Nat) < dd.N →
            ∀ j : Fin (128 / epbB dd.mixed),
              DType.toBits (bDT dd.mixed) (tb.get2 n j) =
                rdBits init dd.b ((pn * 256 + (n : Nat)) * (dd.K / 2) + (64 * t + (j : Nat))) :=
          bTile_read_mixed_gen (d := dd) hd' hmix ht' (m := init) (C := 128 / epbB dd.mixed) hbs
            (bv := tb) (fun i hi => by
              have hr := hreadb i (by
                simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hi)
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e3, e4] using hr)
        refine safe_desc_load (b := dd.a_scale) (bs := [1, 1, 1, 2, 256]) (D := aScaleDesc dd)
          (offsets := [(IntV.ofInt 0 : I32), sm, ks, (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32)])
          rfl ?_ ?_ ?_ ?_ ?_
        · simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using
            aScaleDesc_access (d := dd) hd' pm t
        · intro l hl
          have hl' : (aScaleDesc dd).inbAt
              (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true := by
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hl
          simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using
            aScaleDesc_hin (d := dd) hd' pm t l hpm ht'
        · exact hpre'.2.2.1
        · rintro x ⟨j, ⟨l, hl, rfl⟩, hspan⟩
          have hl' : (aScaleDesc dd).inbAt
              (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true := by
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hl
          have hs' : x ∈ Mem.span .u8
              (dd.a_scale.ptr (((aScaleDesc dd).elemAt (TensorDesc.coord
                [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr := by
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hspan
          show x ∈ dd.a.elems {e | e < dd.M * dd.K} ∪
            dd.b.elems {e | e < dd.N * (dd.K / epbB dd.mixed)} ∪
            dd.a_scale.elems {e | e < dd.M / 128 * (dd.K / 128) * 512} ∪
            dd.b_scale.elems {e | e < dd.N / 128 * (dd.K / 128) * 512}
          exact Or.inl (Or.inr ⟨_, aScaleDesc_elemAt_lt hd' hpm ht' l, hs'⟩)
        · intro tsa hreada hpada
          have havrS : ∀ l : Idx [1, 1, 1, 2, 256],
              (aScaleDesc dd).inbAt
                (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
              init.read .u8 (dd.a_scale.ptr (((aScaleDesc dd).elemAt (TensorDesc.coord
                [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
                = some (tsa.get l) := by
            intro l hl
            have hr := hreada l (by
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hl)
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e5, e7] using hr
          refine safe_desc_load (b := dd.b_scale) (bs := [1, 2, 1, 2, 256]) (D := bScaleDesc dd)
            (offsets := [(IntV.ofInt 0 : I32), sn, ks, (IntV.ofInt 0 : I32),
              (IntV.ofInt 0 : I32)]) rfl ?_ ?_ ?_ ?_ ?_
          · simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using
              bScaleDesc_access (d := dd) hd' pn t
          · intro l hl
            have hl' : (bScaleDesc dd).inbAt
                (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) =
                true := by
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hl
            simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using Nat.lt_of_lt_of_le
              (bScaleDesc_elemAt_lt_inb (d := dd) hd' (pn := pn) (t := t) l hl') hd'.bs_len
          · exact hpre'.2.2.2.1
          · rintro x ⟨j, ⟨l, hl, rfl⟩, hspan⟩
            have hl' : (bScaleDesc dd).inbAt
                (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) =
                true := by
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hl
            have hs' : x ∈ Mem.span .u8
                (dd.b_scale.ptr (((bScaleDesc dd).elemAt (TensorDesc.coord
                  [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr := by
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hspan
            show x ∈ dd.a.elems {e | e < dd.M * dd.K} ∪
              dd.b.elems {e | e < dd.N * (dd.K / epbB dd.mixed)} ∪
              dd.a_scale.elems {e | e < dd.M / 128 * (dd.K / 128) * 512} ∪
              dd.b_scale.elems {e | e < dd.N / 128 * (dd.K / 128) * 512}
            exact Or.inr ⟨_, bScaleDesc_elemAt_lt_inb (d := dd) hd' (pn := pn) (t := t) l hl',
              hs'⟩
          · intro tsb hreadb hpaddb
            have hbvrS : ∀ l : Idx [1, 2, 1, 2, 256],
                (bScaleDesc dd).inbAt
                  (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l) = true →
                init.read .u8 (dd.b_scale.ptr (((bScaleDesc dd).elemAt (TensorDesc.coord
                  [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat : Int)).addr
                  = some (tsb.get l) := by
              intro l hl
              have hr := hreadb l (by
                simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hl)
              simpa only [TensorDesc.laneIn, TensorDesc.lane, List.map_cons, List.map_nil, e0, e6, e7] using hr
            dsimp only
            rw [ite_eq_left (by rw [hep]; simp)]
            refine dot_tail_safe (d := dd) P hd' hpre' i ht' (m := init) rfl hacc hka hkb hks
              ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
            · simpa only [hbs] using shapeOk_mixed
            · simpa only [bDT, hmix] using typesOk_mixed
            · exact htotm
            · exact hdotm false
            · intro r k
              have hrow : pm * 128 + (r : Nat) < dd.M :=
                off_mul_add_lt_of_lt_div (p := pm) (c := 128) (M := dd.M) (l := (r : Nat))
                  hpm r.isLt
              have hcol : 128 * t + (k : Nat) < dd.K := by
                rw [Nat.mul_comm 128 t]
                exact off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := dd.K) (l := (k : Nat))
                  ht' k.isLt
              rw [unpackL_get2 (cfg := cfgMixed) (M := 128) (K := 128) (d := .f8e4nv) ta r k]
              rw [aTile_mixed (d := dd) (m := init) (pm := pm) (t := t) (av := ta) havr r k
                r.isLt k.isLt]
              rw [scalesL_gen (d := dd) hd' hpm ht'
                (hsc := aScale_reshape (by simp only [List.prod_cons, List.prod_nil]; ring) rfl)
                (hav := havrS) (cfg := cfgMixed) (by rfl) r k]
              exact scaledValue_lhs (x := xa dd init (pm * 128 + (r : Nat))
                (128 * t + (k : Nat))) (hvals.a_dom _ _ hrow hcol).1
                (hvals.a_dom _ _ hrow hcol).2.1
            · intro k n hn
              have hrow : pn * 256 + (n : Nat) < dd.N := hn
              have hcol : 128 * t + (k : Nat) < dd.K := by
                rw [Nat.mul_comm 128 t]
                exact off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := dd.K) (l := (k : Nat))
                  ht' k.isLt
              rw [unpackR_get2 (cfg := cfgMixed) (K := 128) (N := 256) (d := bDT dd.mixed)
                tb.trans k n]
              have hbmx : cfgMixed.rhsBits (d := .u8) tb.trans (k : Nat) (n : Nat) =
                  xbBits dd init (pn * 256 + (n : Nat)) (128 * t + (k : Nat)) :=
                bTile_mixed_col_gen (d := dd) hmix (m := init) (C := 128 / epbB dd.mixed) hbs
                  (bv := tb) n (fun j => hbvr n hn j) (k : Nat) k.isLt
              rw [show cfgMixed.rhsBits (d := bDT dd.mixed) tb.trans (k : Nat) (n : Nat) =
                  xbBits dd init (pn * 256 + (n : Nat)) (128 * t + (k : Nat)) from hbmx]
              rw [scalesR_gen (d := dd) hd' ht'
                (hsc := bScale_reshape (by simp only [List.prod_cons, List.prod_nil]; ring) rfl)
                (hbv := hbvrS) (cfg := cfgMixed) (by rfl) n (bRow_lt hd' hn) k]
              exact scaledValue_rhs_mixed (d := dd) (m := init) hmix
                (hvals.b_dom _ _ hrow hcol).1 (hvals.b_dom _ _ hrow hcol).2.1
            · intro r k
              exact scalesL_gen (d := dd) hd' hpm ht'
                (hsc := aScale_reshape (by simp only [List.prod_cons, List.prod_nil]; ring) rfl)
                (hav := havrS) (cfg := cfgMixed) (by rfl) r k
            · intro k n hn
              exact scalesR_gen (d := dd) hd' ht'
                (hsc := bScale_reshape (by simp only [List.prod_cons, List.prod_nil]; ring) rfl)
                (hbv := hbvrS) (cfg := cfgMixed) (by rfl) n (bRow_lt hd' hn) k

end Claims.BlockScaledMxfp8
