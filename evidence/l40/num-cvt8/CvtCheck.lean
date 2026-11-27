import TVA

/-! The library's statement of the L40's float32 → fp8 conversion, against the device.

`lake env lean --run CvtCheck.lean SAMPLE.txt` (from `lean/`), with `SAMPLE.txt` from `dump.py`:
one line `pattern e4m3 e5m2` per float32 input. For each, the library computes
`FloatV.cvtSat .rn (FloatV.cvtWith (g := .binary16) .rz x)` into `e4m3` and `e5m2` (the statement of
`Device.L40.CvtSatF32E4m3`, `…E5m2`) and compares the bits with the device's, a NaN result up to
payload (`FloatV.NaNEq`). It also counts where one rounding (`FloatV.cvtSat .rn x`, the H200's
statement) differs from the device. -/

open TVA

def twoRoundings (g : Format) (x : F32) : FloatV g :=
  FloatV.cvtSat .rn (FloatV.cvtWith (g := .binary16) .rz x)

def agree (g : Format) (lib : FloatV g) (dev : Nat) : Bool :=
  lib.pat == dev || (g.isNaN lib.pat && g.isNaN dev)

def main (args : List String) : IO UInt32 := do
  let lines ← IO.FS.lines (args.headD "cvt_sample.txt")
  let mut n := 0
  let mut bad2 := 0
  let mut bad1 := 0
  let mut ex : List String := []
  for l in lines do
    match (l.splitOn " ").map String.toNat? with
    | [some p, some d4, some d5] =>
      let x : F32 := FloatV.ofPat p
      n := n + 1
      let ok4 := agree .e4m3 (twoRoundings .e4m3 x) d4
      let ok5 := agree .e5m2 (twoRoundings .e5m2 x) d5
      if !(ok4 && ok5) then
        bad2 := bad2 + 1
        if ex.length < 5 then ex := ex ++ [s!"{p}: device {d4} {d5}, library {(twoRoundings .e4m3 x).pat} {(twoRoundings .e5m2 x).pat}"]
      if !(agree .e4m3 (FloatV.cvtSat (g := .e4m3) .rn x) d4 && agree .e5m2 (FloatV.cvtSat (g := .e5m2) .rn x) d5) then
        bad1 := bad1 + 1
    | _ => pure ()
  IO.println s!"{n} float32 inputs; the library's two roundings (cvtSat .rn of cvtWith .rz into binary16) differ from the device on {bad2}; one rounding (cvtSat .rn x) differs on {bad1}"
  for e in ex do IO.println e
  return if bad2 == 0 then 0 else 1
