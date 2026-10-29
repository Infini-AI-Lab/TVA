import TVA.Math.Defs

/-!
The library's float definitions (`TVA.Math.Defs`, compiled to native code by `build.sh`) on a
range of inputs. `run.py` gives the same inputs to the device and compares.

  driver <spec> all <lo> <hi> <cb>       inputs from the index `i` in `[lo, hi)` (`inputsOf`)
  driver <spec> file <path> <base> <cb>  inputs from a file of records of three 64-bit
                                         little-endian words; record `j` has index `base + j`
  driver <spec> dump-all <lo> <hi>       one line `i out` per index, `out` in hex
  driver <spec> dump-file <path> <base>

For `all` and `file`, the output is one line `c h` per chunk `c` of `2^cb` indices meeting
`[lo, hi)`, with `h = Σ out(i)·(2·i·K + 1) mod 2^64` over the chunk (`K = 0x9E3779B97F4A7C15`).
A NaN output is first replaced by the format's `nanMag` (the library's operations are up to NaN
payload). A spec is `op:args` (see `specOf`).
-/
open TVA.Math.Float TVA.Math.Float.Format

def fmtOf : String → Option Format
  | "f16" => some binary16 | "bf16" => some bfloat16 | "f32" => some binary32
  | "f64" => some binary64 | "e5m2" => some e5m2 | "e4m3" => some e4m3 | "e2m1" => some e2m1
  | _ => none

def canon (g : Format) (r : Nat) : Nat := if g.isNaN r then g.nanMag else r

/-- An operation under test: its arity, the width of an input pattern, and its output. -/
structure Spec where
  arity : Nat
  inW : Nat
  out : Nat → Nat → Nat → Nat

def bool01 (b : Bool) : Nat := if b then 1 else 0

/-- The integer an `sg`/`w` pattern encodes. -/
def intOf (sg : Bool) (w : Nat) (n : Nat) : Int :=
  if sg && n ≥ 2 ^ (w - 1) then (n : Int) - 2 ^ w else (n : Int)

/-- Truncation toward zero of a finite pattern, as an integer pattern; `2^63 + 12345` (no result
of a float input) when not finite or out of the range of the integer type. -/
def ftoiPat (f : Format) (sg : Bool) (w : Nat) (x : Nat) : Nat :=
  if f.isFinite x then
    let q := f.toRat x
    let t : Int := q.num.tdiv q.den
    let r := intRange sg w
    if r.1 ≤ t ∧ t ≤ r.2 then (t % (2 ^ w : Int)).toNat else 2 ^ 63 + 12345
  else 2 ^ 63 + 12345

def intTy (s : String) : Option (Bool × Nat) :=
  match s.toList with
  | 's' :: ds => (String.ofList ds).toNat?.map fun w => (true, w)
  | 'u' :: ds => (String.ofList ds).toNat?.map fun w => (false, w)
  | _ => none

def rndOf : String → RoundMode
  | "rz" => .rz
  | _ => .rn

def specOf (s : String) : Option Spec :=
  match s.splitOn ":" with
  | ["cvt", a, b, r, sat] => do
      let f ← fmtOf a; let g ← fmtOf b
      pure ⟨1, f.width, fun x _ _ => canon g (f.cvt g (rndOf r) (sat == "1") x)⟩
  | ["cvtfin", a, b] => do
      -- a conversion of the finite patterns; `2^width` (no pattern) for the others
      let f ← fmtOf a; let g ← fmtOf b
      pure ⟨1, f.width, fun x _ _ =>
        if f.isFinite x then canon g (f.cvt g .rn false x) else 2 ^ g.width⟩
  | ["sqrt", a] => do let f ← fmtOf a; pure ⟨1, f.width, fun x _ _ => canon f (f.sqrt .rn x)⟩
  | ["neg", a] => do let f ← fmtOf a; pure ⟨1, f.width, fun x _ _ => canon f (f.neg x)⟩
  | ["abs", a] => do let f ← fmtOf a; pure ⟨1, f.width, fun x _ _ => canon f (f.absPat x)⟩
  | ["add", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => canon f (f.add .rn x y)⟩
  | ["sub", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => canon f (f.sub .rn x y)⟩
  | ["mul", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => canon f (f.mul .rn x y)⟩
  | ["div", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => canon f (f.div .rn x y)⟩
  | ["fma", a] => do let f ← fmtOf a; pure ⟨3, f.width, fun x y z => canon f (f.fma .rn x y z)⟩
  | ["max", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => canon f (f.maxNum x y)⟩
  | ["min", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => canon f (f.minNum x y)⟩
  | ["maxnan", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => canon f (f.maxNaN x y)⟩
  | ["minnan", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => canon f (f.minNaN x y)⟩
  | ["lt", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => bool01 (f.cmp x y == some .lt)⟩
  | ["le", a] => do
      let f ← fmtOf a
      pure ⟨2, f.width, fun x y _ => bool01 (f.cmp x y == some .lt || f.cmp x y == some .eq)⟩
  | ["eq", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => bool01 (f.cmp x y == some .eq)⟩
  | ["ne", a] => do let f ← fmtOf a; pure ⟨2, f.width, fun x y _ => bool01 (f.cmp x y != some .eq)⟩
  | ["itof", t, b] => do
      let (sg, w) ← intTy t; let g ← fmtOf b
      pure ⟨1, w, fun x _ _ => canon g (g.roundRat .rn false false (intOf sg w x))⟩
  | ["itof2", t, v, b] => do
      -- rounded into `v` first, then converted into `b`
      let (sg, w) ← intTy t; let h ← fmtOf v; let g ← fmtOf b
      pure ⟨1, w, fun x _ _ => canon g (h.cvt g .rn false (h.roundRat .rn false false (intOf sg w x)))⟩
  | ["ftoi", a, t] => do
      let f ← fmtOf a; let (sg, w) ← intTy t
      pure ⟨1, f.width, fun x _ _ => ftoiPat f sg w x⟩
  | _ => none

/-- The inputs of index `i` of an `all` domain: one pattern, or `x = i / 2^w`, `y = i mod 2^w`. -/
def inputsOf (sp : Spec) (i : Nat) : Nat × Nat × Nat :=
  if sp.arity == 1 then (i, 0, 0) else (i / 2 ^ sp.inW, i % 2 ^ sp.inW, 0)

def K : UInt64 := 0x9E3779B97F4A7C15

def word (b : ByteArray) (o : Nat) : Nat := Id.run do
  let mut v := 0
  for k in [0:8] do
    v := v + (b.get! (o + k)).toNat * 2 ^ (8 * k)
  return v

def recordOf (b : ByteArray) (i : Nat) : Nat × Nat × Nat :=
  (word b (24 * i), word b (24 * i + 8), word b (24 * i + 16))

def hashRange (sp : Spec) (get : Nat → Nat × Nat × Nat) (lo hi cb : Nat) (base : Nat := 0) :
    IO Unit := do
  let stdout ← IO.getStdout
  let mut c := lo / 2 ^ cb
  let mut h : UInt64 := 0
  for i in [lo:hi] do
    if i / 2 ^ cb != c then
      stdout.putStrLn s!"{c} {h}"
      c := i / 2 ^ cb
      h := 0
    let (x, y, z) := get (i - base)
    let o := sp.out x y z
    h := h + (UInt64.ofNat o) * (2 * (UInt64.ofNat i) * K + 1)
  if lo < hi then stdout.putStrLn s!"{c} {h}"
  stdout.flush

def hex (n : Nat) : String := String.ofList (Nat.toDigits 16 n)

def dumpRange (sp : Spec) (get : Nat → Nat × Nat × Nat) (lo hi : Nat) (base : Nat := 0) :
    IO Unit := do
  let stdout ← IO.getStdout
  for i in [lo:hi] do
    let (x, y, z) := get (i - base)
    stdout.putStrLn s!"{i} {hex (sp.out x y z)}"
  stdout.flush

def main (args : List String) : IO UInt32 := do
  match args with
  | [s, "all", lo, hi, cb] =>
      let some sp := specOf s | IO.eprintln s!"bad spec {s}"; return 2
      hashRange sp (inputsOf sp) lo.toNat! hi.toNat! cb.toNat!; return 0
  | [s, "file", p, base, cb] =>
      let some sp := specOf s | IO.eprintln s!"bad spec {s}"; return 2
      let b ← IO.FS.readBinFile p
      let n := b.size / 24
      hashRange sp (recordOf b) base.toNat! (base.toNat! + n) cb.toNat! base.toNat!; return 0
  | [s, "dump-all", lo, hi] =>
      let some sp := specOf s | IO.eprintln s!"bad spec {s}"; return 2
      dumpRange sp (inputsOf sp) lo.toNat! hi.toNat!; return 0
  | [s, "dump-file", p, base] =>
      let some sp := specOf s | IO.eprintln s!"bad spec {s}"; return 2
      let b ← IO.FS.readBinFile p
      dumpRange sp (recordOf b) base.toNat! (base.toNat! + b.size / 24) base.toNat!; return 0
  | _ => IO.eprintln "usage: see Driver.lean"; return 2
