import Lean
/-! lr-7a, library review round 7: can a change to a kernel definition escape the item digests?

The claim reader (`tools/claim_reader.lean.in`, `constJson`) gives each constant the hash
`s!"{th}-{vh}"` with `th := mixHash (mixHash ci.type.hash (hash (toString ci.type))) <doc hash>` and,
for a definition, `vh := mixHash v.value.hash (hash (toString v.value))`; `tvr` puts `name hash` of
every cone constant into the item's sha256 digest, and hashes no kernel file whole. `Expr.hash` is 32
bits (`Expr.Data.hash := c.toUInt32.toUInt64`), `String.hash` and `mixHash` are 64-bit non-cryptographic
functions, so `vh` is a 64-bit non-cryptographic value: two definitions of the same type whose `vh`
agree get the same item digest.

This file searches for two values `fun x : Nat => x + a` and `fun x : Nat => x + b`, `a ≠ b`, with the
same `vh`, by parallel collision search with distinguished points (van Oorschot–Wiener): about 2^32
evaluations of `vh`. Compiled to a native program, without lake (an interpreted `#eval` of `search` runs
at about the same speed per thread but scales worse):
  lean --root=<its folder> -c Coll.c DigestCollision.lean
  leanc -O3 -o coll Coll.c -L<toolchain>/lib/lean -lleanshared -Wl,-rpath,<toolchain>/lib/lean
  coll 64 4      (64 threads, 4 trails per task and round)
The output is DigestCollision.out; work/digest_check.py then declares the two definitions and prints the
hash `constJson` gives each. -/
open Lean

/-- `fun x : Nat => x + n`, as Lean elaborates `def m : Nat → Nat := fun x => x + n`. -/
def val (n : Nat) : Expr :=
  mkLambda `x .default (mkConst ``Nat)
    (mkAppN (mkConst ``HAdd.hAdd [0, 0, 0]) #[mkConst ``Nat, mkConst ``Nat, mkConst ``Nat,
      mkAppN (mkConst ``instHAdd [0]) #[mkConst ``Nat, mkConst ``instAddNat], mkBVar 0,
      mkAppN (mkConst ``OfNat.ofNat [0]) #[mkConst ``Nat, mkRawNatLit n, mkApp (mkConst ``instOfNatNat) (mkRawNatLit n)]])

/-- `constJson`'s value hash. -/
def vh (v : Expr) : UInt64 := mixHash v.hash (hash (toString v))

/-- The step of the search: a 64-bit value names a candidate, whose `vh` is the next value. The numeral is
kept below 2^62 so that it stays a small literal. -/
def step (h : UInt64) : UInt64 := vh (val (h >>> 2).toNat)

def dpBits : UInt64 := 14
def isDP (h : UInt64) : Bool := h >>> (64 - dpBits) == 0

/-- A trail from `s` to its first distinguished point: `(start, dp, length)`, or none if too long. -/
def trail (s : UInt64) : Option (UInt64 × UInt64 × Nat) := Id.run do
  let mut x := s
  for i in [0:(20 <<< dpBits.toNat)] do
    x := step x
    if isDP x then return some (s, x, i + 1)
  return none

/-- Walk `n` steps. -/
def walk (x : UInt64) : Nat → UInt64
  | 0 => x
  | n + 1 => walk (step x) n

/-- Two trails that end at the same point: the two different inputs whose steps first agree. -/
def locate (a : UInt64 × Nat) (b : UInt64 × Nat) : Option (UInt64 × UInt64) := Id.run do
  let ((s1, l1), (s2, l2)) := if a.2 ≥ b.2 then (a, b) else (b, a)
  let mut x := walk s1 (l1 - l2)
  let mut y := s2
  if x == y then return none
  for _ in [0:l2] do
    let x' := step x
    let y' := step y
    if x' == y' then return some (x, y)
    x := x'
    y := y'
  return none

def search (threads perTask : Nat) (seed : UInt64) : IO Unit := do
  let mut table : Std.HashMap UInt64 (UInt64 × Nat) := {}
  let mut rnd : UInt64 := seed
  let mut evals := 0
  let t0 ← IO.monoMsNow
  for r in [0:100000] do
    let mut tasks := #[]
    for _ in [0:threads] do
      let mut starts := #[]
      for _ in [0:perTask] do
        rnd := rnd * 6364136223846793005 + 1442695040888963407
        starts := starts.push rnd
      tasks := tasks.push (Task.spawn (prio := .dedicated) fun _ => starts.filterMap trail)
    for t in tasks do
      for (s, dp, l) in t.get do
        evals := evals + l
        match table[dp]? with
        | some (s', l') =>
          if s' != s then
            if let some (x, y) := locate (s, l) (s', l') then
              let a := (x >>> 2).toNat
              let b := (y >>> 2).toNat
              if a != b then
                let t1 ← IO.monoMsNow
                IO.println s!"collision after {evals} evaluations, {(t1 - t0) / 1000} s, round {r}:"
                IO.println s!"  a = {a}, b = {b}"
                IO.println s!"  vh (fun x => x + a) = {vh (val a)}, vh (fun x => x + b) = {vh (val b)}"
                IO.println s!"  Expr.hash: {(val a).hash}, {(val b).hash}; the values are equal: {val a == val b}"
                return
        | none => table := table.insert dp (s, l)
    if r % 20 == 0 then
      let t1 ← IO.monoMsNow
      IO.println s!"round {r}: {evals} evaluations, {table.size} distinguished points, {(t1 - t0) / 1000} s"
      (← IO.getStdout).flush



def main (args : List String) : IO Unit := do
  let threads := (args.getD 0 "32").toNat!
  let per := (args.getD 1 "32").toNat!
  -- a quick throughput check, then the search
  let t0 ← IO.monoMsNow
  let mut acc : UInt64 := 1
  for _ in [0:200000] do acc := step acc
  let t1 ← IO.monoMsNow
  IO.println s!"one thread: 200000 steps in {t1 - t0} ms ({acc % 3})"
  search threads per 0x5eed
