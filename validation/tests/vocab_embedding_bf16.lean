import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `vocab_embedding_bf16` run by `execLaunch` on three small instances
(`int64` ids with an added range and padding rows, two column blocks per row; `int32` ids on a
column slice of a wider weight, with `BLOCK_H` wider than `hidden_dim` and the `int32` extremes;
`int64` ids with overlapping original and added ranges, a negative `added_offset`, four column
blocks). Not a test against the device (phase 4): it checks that `exec` runs the model end to end,
and that the final memory is what `Post` says: each output row the weight's local row of its id,
bit for bit (`+0` for an id outside both ranges), and every other element (the inputs, the
elements just around the output) unchanged, with no byte added. -/

open TVA TVA.Validation Claims.VocabEmbedding

/-- A platform that relates nothing: `vocab_embedding_bf16` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- `hidden_dim`: the smallest covered (a positive multiple of `128`). -/
def hid : Nat := 128

/-- The bits of the weight's storage element `k`: distinct finite `bfloat16` values. -/
def wBits (k : Nat) : Nat := 0x3000 + k

/-- A sentinel in the output (a fresh allocation: any values) and in `pad` elements on each side
of it, which no program may write. -/
def sentinel : Nat := 0xBEEF
def pad : Nat := 16

/-- One case: the id dtype, `BLOCK_H`, the weight's row stride and rows, the vocabulary values,
and the ids. -/
structure Case where
  dI : DType
  BLOCK : Nat
  stride : Nat
  rows : Nat
  os : Int
  oe : Int
  npad : Int
  as : Int
  ae : Int
  ids : List Int

def Case.n (c : Case) : Nat := c.ids.length
def Case.inst (c : Case) : Inst :=
  Claims.VocabEmbedding.mk c.dI c.n hid c.BLOCK c.stride c.rows c.os c.oe c.npad c.as c.ae

/-- An integer element of an id dtype (`int64` or `int32`). -/
def ival (d : DType) (z : Int) : d.Val := d.ofBits (z % (2 ^ (8 * d.bytes) : Int)).toNat

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ d.dI ((List.range c.n).map fun r : Nat =>
    ((d.input_.ptr r).addr, ival d.dI (c.ids.getD r 0)))
  let m := CMem.ofElems m .bf16 ((List.range (c.rows * c.stride)).map fun k : Nat =>
    ((d.weight.ptr k).addr, DType.bf16.ofBits (wBits k)))
  CMem.ofElems m .bf16 ((List.range (c.n * hid + 2 * pad)).map fun j : Nat =>
    ((d.output.ptr ((j : Int) - pad)).addr, DType.bf16.ofBits sentinel))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) Oracle.empty fill0 1000 (model c.inst) (launch c.inst) c.init

/-- The weight row id `v` names, if any: `v - os` in the original range (which wins on an
overlap), `v - (as - (oe - os) - npad)` in the added range. -/
def Case.row (c : Case) (v : Int) : Option Int :=
  if c.os ≤ v ∧ v < c.oe then some (v - c.os)
  else if c.as ≤ v ∧ v < c.ae then some (v - (c.as - (c.oe - c.os) - c.npad))
  else none

/-- The bits `Post` says output element `j` (`-pad ≤ j < n * hidden_dim + pad`) holds: column
`j % hidden_dim` of the row of token `j / hidden_dim`'s id, or `+0`; the sentinel outside the
output. -/
def Case.expect (c : Case) (j : Int) : Nat :=
  if j < 0 ∨ (c.n * hid : Nat) ≤ j then sentinel else
  match c.row (c.ids.getD (j.toNat / hid) 0) with
  | some w => wBits (w * c.stride + (j.toNat % hid : Nat)).toNat
  | none => 0

/-- The final memory against `Post`: the output and the `pad` elements around it, the inputs
unchanged, and no byte added. -/
def Case.agree (c : Case) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m0 := CMem.toMem c.init
  let m := CMem.toMem cf.mem
  let badOut := (List.range (c.n * hid + 2 * pad)).filter fun j : Nat =>
    let i : Int := (j : Int) - pad
    (m.read .bf16 (d.output.ptr i).addr).map DType.bf16.toBits != some (c.expect i)
  let same {dt : DType} (b : Buf dt) (n : Nat) : Bool :=
    (List.range n).all fun j : Nat =>
      (m.read dt (b.ptr j).addr).map dt.toBits == (m0.read dt (b.ptr j).addr).map dt.toBits
  let inputsSame := same d.input_ c.n && same d.weight (c.rows * c.stride)
  if !badOut.isEmpty then some s!"output elements (from -{pad}) {badOut} differ from Post"
  else if !inputsSame then some "an input changed"
  else if cf.mem.size != c.init.size then some "bytes outside the buffers were written"
  else none

/-- `TokOk` on every id (an id of a range names a row inside the weight), the vocabulary values
in `[0, 2^63)`, and the ids representable in the id dtype. -/
def Case.covered (c : Case) : Bool :=
  [c.os, c.oe, c.npad, c.as, c.ae].all (fun v => 0 ≤ v && v < 2 ^ 63) &&
  c.ids.all fun v =>
    (-(2 ^ (8 * c.dI.bytes - 1) : Int) ≤ v && v < 2 ^ (8 * c.dI.bytes - 1)) &&
    match c.row v with
    | some w => 0 ≤ w && w * c.stride + hid ≤ (c.rows * c.stride : Nat)
    | none => true

/-- `int64` ids; original range `[10, 13)`, one padding row, added range `[20, 22)` (local rows
`4, 5`); ids in each range, at the original range's end, `-1`; `BLOCK_H = 64`: two column blocks. -/
def added : Case :=
  { dI := .i64, BLOCK := 64, stride := 128, rows := 6, os := 10, oe := 13, npad := 1,
    as := 20, ae := 22, ids := [11, 21, 13, 20, -1, 10] }

/-- `int32` ids on a column slice (row stride 256, `hidden_dim = 128`); no added range;
`BLOCK_H = 256 > hidden_dim` (half of each program's lanes masked); the `int32` extremes. -/
def slice32 : Case :=
  { dI := .i32, BLOCK := 256, stride := 256, rows := 4, os := 4, oe := 8, npad := 0,
    as := 8, ae := 8, ids := [7, 4, 2147483647, 8, -2147483648, 5] }

/-- `int64` ids; original range `[0, 4)` overlapping the added range `[2, 6)` (`added_offset =
-2`: ids `4, 5` read rows `6, 7`; ids `2, 3` read the original range's rows); `BLOCK_H = 32`:
four column blocks. -/
def overlap : Case :=
  { dI := .i64, BLOCK := 32, stride := 128, rows := 8, os := 0, oe := 4, npad := 0,
    as := 2, ae := 6, ids := [3, 5, 4, 6] }

#guard added.covered && slice32.covered && overlap.covered

def verdictAdded : Verdict := Verdict.ofLaunch added.agree added.run
def verdictSlice : Verdict := Verdict.ofLaunch slice32.agree slice32.run
def verdictOverlap : Verdict := Verdict.ofLaunch overlap.agree overlap.run

#eval IO.println (verdictAdded.line "vocab_embedding_bf16/i64_added")
#eval IO.println (verdictSlice.line "vocab_embedding_bf16/i32_slice")
#eval IO.println (verdictOverlap.line "vocab_embedding_bf16/i64_overlap")
#guard verdictAdded == .certified
#guard verdictSlice == .certified
#guard verdictOverlap == .certified

-- A wrong expectation is caught: `agree` is not vacuous (the padding row forgotten; the added
-- range taking precedence on the overlap).
def noPad : Case := { added with npad := 0 }
#guard !(Verdict.ofLaunch noPad.agree added.run == .certified)
def addedFirst : Case := { overlap with os := 2, oe := 6, as := 0, ae := 4, npad := -2 }
#guard !(Verdict.ofLaunch addedFirst.agree overlap.run == .certified)
