import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `seeded_dropout_f32` run by `execLaunch` on small instances: the
four ways Triton passes the seed (a nonnegative and a negative `int32`, an `int64`, a `uint64`),
`p = 0.5` and `p = 0.75`, grids of one or several programs with and without a masked last block.
Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and
that the final memory is what `Post` says, both halves: each output element `j < n` (`+0` when
dropped; when kept, `x[j] / (1 - p)`, exact on these inputs), and every other byte unchanged (the
input `x`, and sentinels in the output elements past `n` that the masked lanes must not write).

The keep decisions use the claim's own random numbers, `Claims.SeededDropout.uRef` (Philox on the
seed's key words, the reference `Post` names, not the kernel's `tl.rand`); the kept values are
computed here from the inputs as rationals (`x[j] / (1 - p)` with `p` the case's own rational).

`seeded_dropout_f32` has one platform operation: `x / (1 - p)` is `tl.div` (`div.full.f32`,
`Platform.div`). Its answers come from the oracle (`divOracle`: the IEEE quotient, which is exact
here, the divisor being `1/2` or `1/4`), so each run rests on one assumed answer per program and
its verdict is `conditional` with that count, not `certified`. -/

open TVA TVA.Validation Claims.SeededDropout

/-- A platform whose division is the IEEE quotient; nothing else is related. (`exec` does not
consult it: the division's answers come from `divOracle`; it says what those answers assume.) -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ x y r := r = FloatV.divRn x y
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- The oracle: the IEEE quotient, lane by lane, for every division; nothing else. -/
def divOracle : Oracle := fun _ _ q => match q with
  | .plat (.div _ x y) => some (Tensor.map2 FloatV.divRn x y)
  | _ => none

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- A sentinel in the output elements past `n`. -/
def sentinel : F32 := FloatV.ofRat (-7)

/-- One case: `n`, `BLOCK_SIZE = 2^e`, the seed, and `p` as a double (bit pattern) and as a
rational. -/
structure Case where
  n : Nat
  e : Nat
  seed : Int
  pBits : Nat
  pRat : Rat

def Case.inst (c : Case) : Inst := { Claims.SeededDropout.mk c.n c.e c.seed with p := FloatV.ofPat c.pBits }
def Case.programs (c : Case) : Nat := (c.n + 2 ^ c.e - 1) / 2 ^ c.e
/-- The output elements no program should write: up to the end of the last block. -/
def Case.padded (c : Case) : Nat := c.programs * 2 ^ c.e

/-- `x[j]`: `±(j + 1) * 3/2`, alternating in sign (exact in `float32`, as are its quotients by
`1/2` and `1/4`). -/
def xv (j : Nat) : Rat := (if j % 2 = 0 then 1 else -1) * ((j + 1 : Nat) : Rat) * 3 / 2

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .f32 ((List.range c.n).map fun j : Nat => ((d.x.ptr j).addr, (FloatV.ofRat (xv j) : F32)))
  CMem.ofElems m .f32 ((List.range c.padded).map fun j : Nat => ((d.output.ptr j).addr, sentinel))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) divOracle fill0 1000 (model c.inst) (launch c.inst) c.init

/-- Element `j` is kept: the claim's random number `uRef seed j` is above the kernel's `p`. -/
def Case.kept (c : Case) (j : Nat) : Bool := FloatV.lt (p32 c.inst) (uRef c.seed j)

/-- The final memory against the case `e`'s expectation (`e = c` for the test): each output
element `j < n` (`+0` bits when dropped; a finite value equal to `x[j] / (1 - p)` when kept);
then every byte of the initial memory outside the output's first `n` elements unchanged (`x`, the
sentinels past `n`), and no byte added. -/
def Case.agreeE (c e : Case) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let bad := (List.range c.n).filter fun j : Nat =>
    match m.read .f32 (d.output.ptr j).addr with
    | none => true
    | some v =>
      if e.kept j then !(v.isFinite && v.toRat == xv j / (1 - e.pRat))
      else DType.toBits .f32 v != 0
  let lo := d.output.base
  let hi := d.output.base + (c.n * 4 : Nat)
  let changed := c.init.toList.filter fun (a, b) => !(lo ≤ a && a < hi) && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"output elements {bad} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside output[0, n) changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c

/-- The doubles `0.5` and `0.75`. -/
def half : Nat := 0x3FE0000000000000
def threeQuarters : Nat := 0x3FE8000000000000

/-- The file's launch shape, smaller: seed `123` (an `int32`), `p = 0.5`, `n = 10` in blocks of
`4`: three programs, the last one masked (lanes `10, 11`). -/
def i32Seed : Case := { n := 10, e := 2, seed := 123, pBits := half, pRat := 1 / 2 }
/-- A negative `int32` seed (`-5`: the key's high word is `0xFFFFFFFF`, from the sign extension
to `uint64`), `p = 0.75`, `n = 6` in one block of `8` (lanes `6, 7` masked). -/
def negSeed : Case := { n := 6, e := 3, seed := -5, pBits := threeQuarters, pRat := 3 / 4 }
/-- An `int64` seed (`2^40 + 5`), `p = 0.5`, `n = 5` in blocks of `2` (three programs). -/
def i64Seed : Case := { n := 5, e := 1, seed := 2 ^ 40 + 5, pBits := half, pRat := 1 / 2 }
/-- A `uint64` seed (`2^63 + 7`), `p = 0.75`, `n = 8` in one unmasked block of `8`. -/
def u64Seed : Case := { n := 8, e := 3, seed := 2 ^ 63 + 7, pBits := threeQuarters, pRat := 3 / 4 }

def cases : List (String × Case) :=
  [("i32_seed", i32Seed), ("neg_seed", negSeed), ("i64_seed", i64Seed), ("u64_seed", u64Seed)]

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch c.agree c.run

#eval cases.forM fun (name, c) => IO.println (c.verdict.line s!"seeded_dropout_f32/{name}")
-- An agreement resting on the oracle's quotients only: one per program.
#guard cases.all fun (_, c) => c.verdict == .conditional c.programs

-- The instances are covered (`Inst.ok`): `BLOCK_SIZE = 2^e ≤ 2^20`, `n < 2^31`, `p32` finite in
-- `[0, 1]` and equal to the case's rational, the seed in `[-2^63, 2^64)`; `mk` gives the buffers'
-- lengths and their disjointness (`mk_ok`). `Pre`: both buffers 4-aligned, inside `[0, 2^64)`,
-- every byte of their first `n` elements written by `init`.
#guard cases.all fun (_, c) =>
  c.e ≤ 20 && c.n < 2 ^ 31 && (p32 c.inst).isFinite && (p32 c.inst).toRat == c.pRat &&
    0 ≤ c.pRat && c.pRat ≤ 1 && -2 ^ 63 ≤ c.seed && c.seed < 2 ^ 64 &&
    c.inst.x.base % 4 == 0 && c.inst.output.base % 4 == 0
-- The seeds take the three types Triton gives a Python `int` (`model`'s three branches).
#guard (-2 ^ 31 ≤ negSeed.seed ∧ i64Seed.seed ≥ 2 ^ 31 ∧ i64Seed.seed < 2 ^ 63 ∧ u64Seed.seed ≥ 2 ^ 63 : Bool)
-- Every case keeps some elements and drops others.
#guard cases.all fun (_, c) => (List.range c.n).any c.kept && (List.range c.n).any (!c.kept ·)

-- Not vacuous: a wrong expectation is caught. Another seed's keep decisions, and another `p`'s
-- quotients, against the real runs.
#guard (Verdict.ofLaunch (i32Seed.agreeE { i32Seed with seed := 124 }) i32Seed.run).isDiscrepancy
#guard (Verdict.ofLaunch (u64Seed.agreeE { u64Seed with seed := 7 }) u64Seed.run).isDiscrepancy
#guard (Verdict.ofLaunch (negSeed.agreeE { negSeed with pRat := 1 / 2 }) negSeed.run).isDiscrepancy
