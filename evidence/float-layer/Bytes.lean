import TVA.Lang.Scalar

/-!
Part E, the library side: the memory encoding of `TVA.Lang.Scalar` (`DType.bytes`,
`DType.toBits`, `DType.ofBits`) and `TVA.Math.Bytes.leBytes` / `leValue`, which `TVA.Lang.Mem`
uses to write and read an element. Reads lines `w <dtype> <pattern>` (write: print the bytes
`leBytes d.bytes (d.toBits (d.ofBits pattern))`) and `r <dtype> <bytes, hex, least significant
first>` (read: print `d.toBits (d.ofBits (leValue bytes))`) from the file named by the first
argument. Run with `lake env lean --run Bytes.lean <file>` from `lean/`.
-/
open TVA TVA.Math.Bytes

def dtypeOf : String → Option DType
  | "i1" => some .i1 | "i8" => some .i8 | "i16" => some .i16 | "i32" => some .i32
  | "i64" => some .i64 | "u8" => some .u8 | "u16" => some .u16 | "u32" => some .u32
  | "u64" => some .u64 | "f16" => some .f16 | "bf16" => some .bf16 | "f32" => some .f32
  | "f64" => some .f64 | "f8e5" => some .f8e5 | "f8e4nv" => some .f8e4nv
  | _ => none

def hexByte (b : Byte) : String :=
  let s := String.ofList (Nat.toDigits 16 b.val)
  if s.length == 1 then "0" ++ s else s

def parseHex (s : String) : Nat :=
  s.foldl (fun acc c => acc * 16 + (if c.isDigit then c.toNat - '0'.toNat else c.toLower.toNat - 'a'.toNat + 10)) 0

def main (args : List String) : IO Unit := do
  let lines ← IO.FS.lines args.head!
  let stdout ← IO.getStdout
  for l in lines do
    match l.splitOn " " with
    | ["w", t, p] =>
        let some d := dtypeOf t | throw (IO.userError s!"dtype {t}")
        let bs := leBytes d.bytes (d.toBits (d.ofBits (parseHex p)))
        stdout.putStrLn (" ".intercalate (bs.map hexByte))
    | ["r", t, b] =>
        let some d := dtypeOf t | throw (IO.userError s!"dtype {t}")
        let n := b.length / 2
        let bs : List Byte := (List.range n).map fun i =>
          ⟨parseHex ((b.drop (2 * i)).take 2).toString % 256, Nat.mod_lt _ (by decide)⟩
        stdout.putStrLn (String.ofList (Nat.toDigits 16 (d.toBits (d.ofBits (leValue bs)))))
    | _ => throw (IO.userError s!"line {l}")
