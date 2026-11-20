import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `kv_indices_i32` run by `execLaunch` on two small instances (with
`kv_start_idx` and `int64` indices; without it and `int32` indices). Not a test against the device
(phase 4): it checks that `exec` runs the model end to end, and that the final memory is what
`Post` says: each request's window of its table row copied to its slice of `kv_indices`, and every
other element (the inputs, and the output elements no request owns) unchanged. -/

open TVA TVA.Validation Claims.KvIndices

/-- A platform that relates nothing: `kv_indices_i32` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- The table: `slots × ctx`, entry `(r, c)` holds `1000 * r + c`. -/
def slots : Nat := 4
def ctx : Nat := 8
def entry (k : Nat) : Int := 1000 * (k / ctx) + k % ctx

/-- A sentinel in every output element: those no request owns must keep it. -/
def sentinel : Int := -7

/-- One case: the requests' rows, lengths and starts, and the index dtypes. -/
structure Case where
  rows : List Nat
  lens : List Nat
  starts : Option (List Nat)
  dp : DType
  dl : DType
  ds : DType
  /-- The output's length: the lengths' sum plus some unowned elements. -/
  total : Nat

def Case.bs (c : Case) : Nat := c.rows.length
def Case.inst (c : Case) : Inst := Claims.KvIndices.mk slots ctx c.bs c.dp c.dl c.ds c.starts.isSome c.total
/-- `kv_indptr`: the exclusive running sums of the lengths, from `0`. -/
def Case.indptr (c : Case) : List Nat := (List.range (c.bs + 1)).map fun b => (c.lens.take b).sum
def Case.start (c : Case) (b : Nat) : Nat := (c.starts.getD []).getD b 0

/-- An integer element of an index dtype (`int64` or `int32`). -/
def ival (d : DType) (z : Int) : d.Val := d.ofBits (z % (2 ^ (8 * d.bytes) : Int)).toNat

def writeInts (m : CMem) (d : DType) (b : Buf d) (xs : List Int) : CMem :=
  CMem.ofElems m d ((List.range xs.length).map fun j : Nat => ((b.ptr j).addr, ival d (xs.getD j 0)))

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .i32 ((List.range (slots * ctx)).map fun k : Nat =>
    ((d.req_to_token.ptr k).addr, (IntV.ofInt (entry k) : I32)))
  let m := writeInts m d.dP d.req_pool_indices (c.rows.map (↑))
  let m := writeInts m d.dL d.seq_lens (c.lens.map (↑))
  let m := writeInts m .i32 d.kv_indptr (c.indptr.map (↑))
  let m := if c.starts.isSome then writeInts m d.dS d.kv_start_idx ((List.range c.bs).map fun b => (c.start b : Int)) else m
  writeInts m .i32 d.kv_indices (List.replicate c.total sentinel)

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) Oracle.empty fill0 1000 (model c.inst) (launch c.inst) c.init

/-- What `Post` says element `j` of `kv_indices` holds: request `b`'s `p`-th entry of its window
when `j = kv_indptr[b] + p`, `p < seq_lens[b]`; the sentinel otherwise. -/
def Case.expect (c : Case) (j : Nat) : Int :=
  match (List.range c.bs).find? (fun b => c.indptr.getD b 0 ≤ j ∧ j < c.indptr.getD b 0 + c.lens.getD b 0) with
  | some b => entry (c.rows.getD b 0 * ctx + c.start b + (j - c.indptr.getD b 0))
  | none => sentinel

/-- The final memory against `Post`: every output element, and the inputs unchanged. -/
def Case.agree (c : Case) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m0 := CMem.toMem c.init
  let m := CMem.toMem cf.mem
  let badOut := (List.range c.total).filter fun j : Nat =>
    (m.read .i32 (d.kv_indices.ptr j).addr).map (·.toInt) != some (c.expect j)
  let same {dt : DType} (b : Buf dt) (n : Nat) : Bool :=
    (List.range n).all fun j : Nat => (m.read dt (b.ptr j).addr).map dt.toBits == (m0.read dt (b.ptr j).addr).map dt.toBits
  let inputsSame := same d.req_to_token (slots * ctx) && same d.req_pool_indices c.bs &&
    same d.seq_lens c.bs && same d.kv_indptr (c.bs + 1) &&
    (!c.starts.isSome || same d.kv_start_idx c.bs)
  if !badOut.isEmpty then some s!"kv_indices elements {badOut} differ from Post"
  else if !inputsSame then some "an input changed"
  else none

/-- With `kv_start_idx` (`int64` indices); request 1 is empty; two output elements unowned. -/
def withStart : Case :=
  { rows := [2, 0, 3], lens := [3, 0, 5], starts := some [1, 4, 2],
    dp := .i64, dl := .i64, ds := .i64, total := 10 }

/-- Without `kv_start_idx` (`int32` indices); a request reading a whole row. -/
def noStart : Case :=
  { rows := [1, 3], lens := [8, 2], starts := none,
    dp := .i32, dl := .i32, ds := .i32, total := 11 }

def verdictWith : Verdict := Verdict.ofLaunch withStart.agree withStart.run
def verdictNo : Verdict := Verdict.ofLaunch noStart.agree noStart.run

#eval IO.println (verdictWith.line "kv_indices_i32/with_start")
#eval IO.println (verdictNo.line "kv_indices_i32/no_start")
#guard verdictWith == .certified
#guard verdictNo == .certified

-- The instances are covered ones (`Inst.ok`'s decidable parts; `mk_ok`-style facts aside).
#guard withStart.indptr == [0, 3, 3, 8] && noStart.indptr == [0, 8, 10]

-- A wrong expectation is caught: `agree` is not vacuous.
def shifted : Case := { withStart with starts := some [0, 4, 2] }
#guard (Verdict.ofLaunch (shifted.agree) withStart.run).isDiscrepancy ||
  !((Verdict.ofLaunch (shifted.agree) withStart.run) == .certified)
