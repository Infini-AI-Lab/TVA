import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `attn_bwd_pre_f16` run by `execLaunch` on two covered instances. Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says, both halves: (a) every row `r < Z·H·N_CTX` of `Delta` holds a
`float32` value, and for a row whose inputs are finite that value is finite and within
`tol (Σ_k |o_k·do_k|)` of the exact `Σ_k o_k·do_k` (computed here from the inputs as rationals;
`tol` is the function `Post` names); (b) every other byte is unchanged (`O`, `DO`, sentinels in
`Delta`'s tail and on both sides of it), and no byte is added.

The inputs are multiples of `1/2` of magnitude at most `4` (so every product and every partial
sum is an exact `float32` whatever the order and the fusion), except one row of each case holding
a NaN and one an `∞` (in `o` or `do`): there `Post` only asks for a value.

`attn_bwd_pre_f16` uses no platform operation (the products, the `float16` widenings and the
`tl.sum` tree are solved): the verdicts are `certified`, with the empty oracle. -/

open TVA TVA.Validation Claims.AttnBwdPre

/-- A platform that relates nothing: `attn_bwd_pre_f16` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits (no load of this kernel is masked). -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- A sentinel in every `Delta` element before the launch, in the buffer's tail past
`Z·H·N_CTX`, and in `pad` elements on each side of the buffer. -/
def sentinel : F32 := FloatV.ofRat (-12345)
def pad : Nat := 4

/-- The `float16` patterns of NaN and `+∞`. -/
def nan16 : F16 := DType.f16.ofBits 0x7E00
def inf16 : F16 := DType.f16.ofBits 0x7C00

/-- One case: an instance, the rows holding a NaN / an `∞`, and the inputs' values (element `e`
of `O` and of `DO`, as multiples of `1/2`). -/
structure Case where
  inst : Inst
  nanRow : Nat
  infRow : Nat
  oVal : Nat → Rat
  doVal : Nat → Rat

def Case.rows (c : Case) : Nat := c.inst.Z * c.inst.H * c.inst.N_CTX
def Case.elems (c : Case) : Nat := c.rows * c.inst.HEAD_DIM

/-- Element `e` of `O`: the value, except the first element of the NaN row (a NaN). -/
def Case.oElem (c : Case) (e : Nat) : F16 :=
  if e = c.nanRow * c.inst.HEAD_DIM then nan16 else FloatV.ofRat (c.oVal e)
/-- Element `e` of `DO`: the value, except the last element of the `∞` row (`+∞`). -/
def Case.doElem (c : Case) (e : Nat) : F16 :=
  if e = c.infRow * c.inst.HEAD_DIM + c.inst.HEAD_DIM - 1 then inf16 else FloatV.ofRat (c.doVal e)

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .f16 ((List.range c.elems).map fun e : Nat => ((d.O.ptr e).addr, c.oElem e))
  let m := CMem.ofElems m .f16 ((List.range c.elems).map fun e : Nat => ((d.DO.ptr e).addr, c.doElem e))
  CMem.ofElems m .f32 ((List.range (d.Delta.len + 2 * pad)).map fun j : Nat =>
    ((d.Delta.ptr ((j : Int) - pad)).addr, sentinel))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) Oracle.empty fill0 100000 (model c.inst) (launch c.inst) c.init

/-- Row `r`'s inputs are finite (no NaN, no `∞`). -/
def Case.finiteRow (c : Case) (r : Nat) : Bool :=
  (List.range c.inst.HEAD_DIM).all fun k : Nat =>
    (c.oElem (r * c.inst.HEAD_DIM + k)).isFinite && (c.doElem (r * c.inst.HEAD_DIM + k)).isFinite

/-- `Σ_k o[r, k]·do[r, k]` and `Σ_k |o[r, k]·do[r, k]|`, exact, from the case's values. -/
def Case.dot (c : Case) (r : Nat) : Rat :=
  ((List.range c.inst.HEAD_DIM).map fun k => c.oVal (r * c.inst.HEAD_DIM + k) * c.doVal (r * c.inst.HEAD_DIM + k)).sum
def Case.dotAbs (c : Case) (r : Nat) : Rat :=
  ((List.range c.inst.HEAD_DIM).map fun k => |c.oVal (r * c.inst.HEAD_DIM + k) * c.doVal (r * c.inst.HEAD_DIM + k)|).sum

/-- `Delta`'s addresses written: elements `r < Z·H·N_CTX`. -/
def Case.inOut (c : Case) (a : Int) : Bool :=
  c.inst.Delta.base ≤ a && a < c.inst.Delta.base + 4 * (c.rows : Nat)

/-- The final memory against `Post` with the expectation `ex` (the case's `dot` for the test):
(a) every row of `Delta` holds a value, finite and within `tol` of `ex r` when the row is finite;
(b) every byte outside the rows of `Delta` unchanged, none added. -/
def Case.agreeE (c : Case) (ex : Nat → Rat) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let badOut := (List.range c.rows).filter fun r : Nat =>
    match m.read .f32 (d.Delta.ptr r).addr with
    | none => true
    | some v => c.finiteRow r && !(v.isFinite && |v.toRat - ex r| ≤ tol (c.dotAbs r))
  let badElse := c.init.toList.filter fun (a, b) => !c.inOut a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"Delta rows {badOut.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside Delta's rows changed"
  else if cf.mem.size != c.init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch (c.agreeE c.dot) c.run

/-- A multiple of `1/2` in `[-4, 4]` from an index. -/
def halfVal (z : Nat) : Rat := (((z % 17 : Nat) : Int) - 8 : Int) / 2

/-! ## The cases -/

/-- The host's launch `canon 1 1 128 16` (`BLOCK_M = PRE_BLOCK = 128`, `HEAD_DIM = 16`, one
program: the whole `128 × 16` tile); row `5` holds a NaN in `o`, row `77` an `∞` in `do`. -/
def host : Case :=
  { inst := canon 1 1 128 16, nanRow := 5, infRow := 77,
    oVal := fun e => halfVal (3 * e + 1), doVal := fun e => halfVal (5 * e + 2) }

theorem host_ok : host.inst.ok :=
  canon_ok 1 1 128 16 4 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- A grid of `2 × 6` programs: `Z = 2`, `H = 3`, `N_CTX = 4`, `BLOCK_M = 2`, `HEAD_DIM = 8`
(the head offsets `off_hz·HEAD_DIM·N_CTX` and `off_hz·N_CTX` at six heads), `Delta` with `3`
spare elements past the `24` rows (they keep the sentinel); the buffers apart, not at the host's
spacing. Row `13` holds a NaN in `o`, row `22` an `∞` in `do`. -/
def gridInst : Inst :=
  { Z := 2, H := 3, N_CTX := 4, HEAD_DIM := 8, BLOCK_M := 2,
    O := ⟨4096, 192⟩, DO := ⟨8192, 192⟩, Delta := ⟨1024, 27⟩ }

def grid : Case :=
  { inst := gridInst, nanRow := 13, infRow := 22,
    oVal := fun e => halfVal (7 * e + 3), doVal := fun e => halfVal (2 * e + 11) }

theorem grid_ok : grid.inst.ok := by
  refine
    { block_pow2 := ⟨1, rfl⟩
      head_pow2 := ⟨3, rfl⟩
      head_le := by decide
      block_le := by decide
      ctx_dvd := by decide
      ctx_lt := by decide
      ctx_le := by decide
      batch_le := by decide
      bh_lt := by decide
      grid_lt := by decide
      row_fit := by decide
      head_fit := by decide
      O_len := by decide
      DO_len := by decide
      Delta_len := by decide
      Delta_O := ?_
      Delta_DO := ?_ }
  · exact elems_disjoint_of_lt (fun j hj => by simp only [outSet, Set.mem_ofPred_eq, grid, gridInst] at hj ⊢; omega)
      (fun j hj => by simp only [inSet, Set.mem_ofPred_eq, grid, gridInst] at hj ⊢; omega)
      (by simp only [grid, gridInst, DType.bytes]; omega)
  · exact elems_disjoint_of_lt (fun j hj => by simp only [outSet, Set.mem_ofPred_eq, grid, gridInst] at hj ⊢; omega)
      (fun j hj => by simp only [inSet, Set.mem_ofPred_eq, grid, gridInst] at hj ⊢; omega)
      (by simp only [grid, gridInst, DType.bytes]; omega)

def cases : List (String × Case) := [("host", host), ("grid", grid)]

-- `Pre`: each buffer aligned, inside `[0, 2^64)`, every byte present in `init` (checked element by
-- element).
#guard cases.all fun (_, c) =>
  let d := c.inst
  let m := CMem.toMem c.init
  d.O.base % 2 == 0 && d.DO.base % 2 == 0 && d.Delta.base % 4 == 0 &&
  ((List.range d.O.len).all fun e : Nat => (m.read .f16 (d.O.ptr e).addr).isSome) &&
  ((List.range d.DO.len).all fun e : Nat => (m.read .f16 (d.DO.ptr e).addr).isSome) &&
  ((List.range d.Delta.len).all fun e : Nat => (m.read .f32 (d.Delta.ptr e).addr).isSome)

-- The inputs are encoded exactly; each case has exactly two non-finite rows; the expectations of
-- the finite rows are not the sentinel, and some are nonzero.
#guard cases.all fun (_, c) =>
  ((List.range c.elems).all fun e : Nat =>
    (e == c.nanRow * c.inst.HEAD_DIM || (c.oElem e).toRat == c.oVal e) &&
    (e == c.infRow * c.inst.HEAD_DIM + c.inst.HEAD_DIM - 1 || (c.doElem e).toRat == c.doVal e)) &&
  ((List.range c.rows).filter fun r => !c.finiteRow r) == [min c.nanRow c.infRow, max c.nanRow c.infRow] &&
  ((List.range c.rows).all fun r => c.dot r != sentinel.toRat) &&
  ((List.range c.rows).any fun r => c.finiteRow r && c.dot r != 0)

def verdictHost : Verdict := host.verdict
def verdictGrid : Verdict := grid.verdict

#eval IO.println (verdictHost.line "attn_bwd_pre_f16/host")
#eval IO.println (verdictGrid.line "attn_bwd_pre_f16/grid")
#guard verdictHost == .certified
#guard verdictGrid == .certified

-- A wrong expectation is caught: `agree` is not vacuous (one row off by `1/2`; every row the dot
-- with the next row of `do`).
#guard
  let r := grid.run
  (Verdict.ofLaunch (grid.agreeE fun i => if i = 6 then grid.dot i + 1 / 2 else grid.dot i) r).isDiscrepancy &&
  (Verdict.ofLaunch (grid.agreeE fun i => ((List.range 8).map fun k =>
    grid.oVal (i * 8 + k) * grid.doVal ((i + 1) * 8 + k)).sum) r).isDiscrepancy
