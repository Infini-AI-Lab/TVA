import TVA.Math.Float.Format

/-!
lr-9a: the library's own rounding (`Format.cvt src dst .rn false .canonical`, which
`FloatV.cvtWith .rn` is, and which `Rules.Num.CvtRn` says the device's conversion equals) on
input patterns; `cvt_lean.py` compares the output with the device bit for bit.

- `f16`/`bf16`: every pattern, into e4m3 and e5m2;
- `f32`: every pattern whose high 16 bits are anything and whose low 16 bits are one of
  `0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff` (ties and their neighbours for both fp8
  formats sit in the high bits), into e4m3 and e5m2;
- `s32`: the same `f32` patterns into float16 and bfloat16 with `.rn` and `.rz` (`tl.cvt`,
  `FloatV.cvtWith`; the low halves above hold the bfloat16 ties and their neighbours).

One line per high half (or per 16-bit pattern): the destination bytes in hex, or `--` where the
input is not finite (the fact says nothing there; the script skips them).
-/
open TVA.Math.Float

def hex2 (n : Nat) : String :=
  let d := "0123456789abcdef".toList
  String.ofList [d[(n / 16) % 16]!, d[n % 16]!]

def conv (src : Format) (x : Nat) : String :=
  if src.isFinite x then
    hex2 (Format.cvt src Format.e4m3 .rn false .canonical x) ++
      hex2 (Format.cvt src Format.e5m2 .rn false .canonical x)
  else "----"

def hex4 (n : Nat) : String := hex2 (n / 256) ++ hex2 (n % 256)

def conv16 (x : Nat) : String :=
  if Format.binary32.isFinite x || Format.binary32.isInf x then
    hex4 (Format.cvt Format.binary32 Format.binary16 .rn false .canonical x) ++
      hex4 (Format.cvt Format.binary32 Format.binary16 .rz false .canonical x) ++
      hex4 (Format.cvt Format.binary32 Format.bfloat16 .rn false .canonical x) ++
      hex4 (Format.cvt Format.binary32 Format.bfloat16 .rz false .canonical x)
  else "----------------"

def lows : List Nat := [0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff]

def main : IO Unit := do
  for i in [0:65536] do
    IO.println s!"f16 {i} {conv Format.binary16 i}"
  for i in [0:65536] do
    IO.println s!"bf16 {i} {conv Format.bfloat16 i}"
  for hi in [0:65536] do
    let s := lows.foldl (fun acc lo => acc ++ conv Format.binary32 (hi * 65536 + lo)) ""
    IO.println s!"f32 {hi} {s}"
  for hi in [0:65536] do
    let s := lows.foldl (fun acc lo => acc ++ conv16 (hi * 65536 + lo)) ""
    IO.println s!"s32 {hi} {s}"
