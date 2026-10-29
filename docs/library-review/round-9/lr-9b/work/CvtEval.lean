import TVA

/-!
lr-9b, library review round 9 (L4): the Lean rounding `Format.cvt f g .rn false .canonical`, which
`Rules.Num.CvtRn` (and so `CvtF16E4m3`, …) names, evaluated on every `float16` and `bfloat16`
pattern and on a list of `float32` patterns, for comparison with the device
(`cvt_compare.py`). Prints one hex byte per input.
-/

open TVA TVA.Math.Float

def hex2 (n : Nat) : String :=
  let d := "0123456789abcdef".toList
  String.mk [d.getD (n / 16 % 16) '0', d.getD (n % 16) '0']

def table (f g : Format) (pats : List Nat) : String :=
  String.join (pats.map fun p => hex2 (Format.cvt f g .rn false .canonical p))

/-- `float32` patterns: every 65521st pattern from 0, plus edge cases near the fp8 ranges. -/
def f32pats : List Nat :=
  (List.range 65551).map (fun i => (i * 65521) % 4294967296)

#eval IO.println ("F16E4 " ++ table .binary16 .e4m3 (List.range 65536))
#eval IO.println ("F16E5 " ++ table .binary16 .e5m2 (List.range 65536))
#eval IO.println ("BF16E4 " ++ table .bfloat16 .e4m3 (List.range 65536))
#eval IO.println ("BF16E5 " ++ table .bfloat16 .e5m2 (List.range 65536))
#eval IO.println ("F32E4 " ++ table .binary32 .e4m3 f32pats)
#eval IO.println ("F32E5 " ++ table .binary32 .e5m2 f32pats)
