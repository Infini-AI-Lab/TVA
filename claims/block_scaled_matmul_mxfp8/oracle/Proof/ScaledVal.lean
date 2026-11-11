import Claim
import TVA.Proof
import Proof.Basic

/-!
# `scaledValue` of an element and its `e8m0` scale

The library's `scaledValue` (the exact value of an element with its scale) for a finite element
and an `e8m0` scale byte in `e8m0Dom`, and the shape/dtype checks of the two configurations the
kernel uses.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- The `e8m0` value of a byte in `e8m0Dom` is `scaleV`. -/
theorem e8m0Value_eq {s : U8} (h : Device.e8m0Dom s.bits.toNat) :
    e8m0Value s.bits.toNat = some (scaleV s) := by
  have hb : s.bits.toNat ≠ 255 := by
    unfold Device.e8m0Dom at h
    omega
  simp [e8m0Value, scaleV, hb]

/-- **`scaledValue` of a finite element with an `e8m0` scale**: the value is the element's value
times `scaleV`. -/
theorem scaledValue_e8m0 {fmt : MXFmt} {bits : Nat} {s : U8}
    (hfin : fmt.format.isFinite bits = true) (hd : Device.e8m0Dom s.bits.toNat) :
    scaledValue fmt ScaleFmt.e8m0 bits (some (DType.toBits .u8 s)) =
      some (fmt.format.toRat bits * scaleV s) := by
  have hs : e8m0Value (DType.toBits .u8 s) = some (scaleV s) := by
    simpa [DType.toBits] using e8m0Value_eq hd
  simp only [scaledValue, MXFmt.value, ScaleFmt.value, hfin, ite_true, hs, Option.map_some]

/-- The `lhs` of the `"mixed"` dot: the value of a finite `e4m3` element with its scale. -/
theorem scaledValue_lhs {x : F8E4} {s : U8} (hfin : x.isFinite = true)
    (hd : Device.e8m0Dom s.bits.toNat) :
    scaledValue MXFmt.e4m3 ScaleFmt.e8m0 (DType.toBits .f8e4nv x) (some (DType.toBits .u8 s)) =
      some (scaled x.toRat s) := by
  have h := scaledValue_e8m0 (fmt := MXFmt.e4m3) (bits := DType.toBits .f8e4nv x) hfin hd
  simpa [scaled, MXFmt.format, DType.toBits, FloatV.toRat] using h

/-- The `rhs` of either dot: the value of a finite element of `b` with its scale. -/
theorem scaledValue_rhs {d : Inst} {m : Mem} {j k : Nat} {s : U8}
    (hfin : bFin d m j k = true) (hd : Device.e8m0Dom s.bits.toNat) :
    scaledValue (if d.mixed then MXFmt.e2m1 else MXFmt.e4m3) ScaleFmt.e8m0 (xbBits d m j k)
        (some (DType.toBits .u8 s)) =
      some (scaled (bVal d m j k) s) := by
  by_cases hm : d.mixed
  · have hfin' : MXFmt.e2m1.format.isFinite (xbBits d m j k) = true := by
      simpa [bFin, bFmt, hm, MXFmt.format] using hfin
    have h := scaledValue_e8m0 (fmt := MXFmt.e2m1) (bits := xbBits d m j k) hfin' hd
    simpa [scaled, bVal, bFmt, hm, MXFmt.format] using h
  · have hfin' : MXFmt.e4m3.format.isFinite (xbBits d m j k) = true := by
      simpa [bFin, bFmt, hm, MXFmt.format] using hfin
    have h := scaledValue_e8m0 (fmt := MXFmt.e4m3) (bits := xbBits d m j k) hfin' hd
    simpa [scaled, bVal, bFmt, hm, MXFmt.format] using h

/-! ## The two configurations' checks -/

theorem shapeOk_mixed : cfgMixed.ShapeOk 128 256 128 128 128 64 256 4 := by
  unfold ScaledCfg.ShapeOk ScaledCfg.StorageOk cfgMixed
  decide

/-- The `mxfp8` dot's shapes are accepted. -/
theorem shapeOk_mx : cfgMx.ShapeOk 128 256 128 128 128 128 256 4 := by
  unfold ScaledCfg.ShapeOk ScaledCfg.StorageOk cfgMx
  decide

/-- The `mixed` dot's operand types are accepted. -/
theorem typesOk_mixed : cfgMixed.TypesOk .f8e4nv .u8 .u8 .u8 := by
  unfold ScaledCfg.TypesOk MXFmt.StoredIn ScaleFmt.Stores cfgMixed
  decide

/-- The `mxfp8` dot's operand types are accepted. -/
theorem typesOk_mx : cfgMx.TypesOk .f8e4nv .f8e4nv .u8 .u8 := by
  unfold ScaledCfg.TypesOk MXFmt.StoredIn ScaleFmt.Stores cfgMx
  decide

end Claims.BlockScaledMxfp8
