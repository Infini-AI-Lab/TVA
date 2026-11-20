import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `moe_sum_reduce_bf16` run by `execLaunch` on three small instances.
Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and that
the final memory is what `Post` says on these inputs. The inputs are small integers and the scales
`2`, `1/2`, `1`, so every `float32` partial sum, the scaled sum and its `bfloat16` rounding are
exact whatever the order: each output element is exactly `σ · Σ_k input[t, k, j]` (which `Post`
allows, at distance `0`), and every other byte is unchanged (the input, sentinels on both sides of
the output, and the bytes the masked lanes of the ragged edge would reach).

`moe_sum_reduce_bf16` uses no platform operation: the verdicts are `certified`, with the empty
oracle. -/

open TVA TVA.Validation Claims.MoeSumReduce

/-- A platform that relates nothing: `moe_sum_reduce_bf16` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- A sentinel in every output element before the launch, and in the elements just outside the
output: the outside ones must keep it; every inside one must be overwritten (no expectation below
equals it). -/
def sentinel : Int := -99

/-- Sentinel elements on each side of the output buffer. -/
def pad : Nat := 4

/-- One case: the sizes, the block exponents, the scale (its double's pattern and its value), and
the inputs as integers. -/
structure Case where
  T : Nat
  K : Nat
  D : Nat
  em : Nat
  ed : Nat
  sbits : Nat
  /-- The scale's value, written independently of `sbits` (checked against it below). -/
  sigma : Rat
  /-- Input `(t, k, j)`. -/
  x : Nat → Nat → Nat → Int

def Case.vals (c : Case) (e : Nat) : Int :=
  c.x (e / (c.K * c.D)) (e / c.D % c.K) (e % c.D)

/-- The claim's instance, with `inVal` the case's inputs (as `Pre` requires `input` to hold). -/
def Case.inst (c : Case) : Inst :=
  { Claims.MoeSumReduce.mk c.T c.K c.D c.sbits c.em c.ed with
    inVal := fun e => FloatV.ofRat (c.vals e : Rat) }

def bfv (z : Int) : BF16 := FloatV.ofRat (z : Rat)

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .bf16 ((List.range (c.T * c.K * c.D)).map fun e : Nat =>
    ((d.input.ptr e).addr, bfv (c.vals e)))
  -- the output and `pad` elements on each side of it, all the sentinel
  CMem.ofElems m .bf16 ((List.range (c.T * c.D + 2 * pad)).map fun e : Nat =>
    ((d.output.ptr ((e : Int) - pad)).addr, bfv sentinel))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) Oracle.empty fill0 1000 (model c.inst) (launch c.inst) c.init

/-- What `Post` says output `(t, j)` holds on these inputs: `σ · Σ_k x[t, k, j]`, exact. -/
def Case.expect (c : Case) (t j : Nat) : Rat :=
  c.sigma * (((List.range c.K).map fun k => c.x t k j).sum : Int)

/-- The output's addresses (`T·D` elements of `2` bytes from the buffer's base). -/
def Case.inOut (c : Case) (a : Int) : Bool :=
  c.inst.output.base ≤ a && a < c.inst.output.base + 2 * (c.T * c.D : Nat)

/-- The final memory against `Post`: every output element is the expectation; every byte outside
the output (the input, the sentinels) keeps its initial value, and no byte is added. -/
def Case.agree (c : Case) (expect : Nat → Nat → Rat) (cf : CConfig (launch c.inst).size) :
    Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let badOut := (List.range (c.T * c.D)).filter fun e : Nat =>
    (m.read .bf16 (d.output.ptr e).addr).map (·.toRat) != some (expect (e / c.D) (e % c.D))
  let badElse := c.init.toList.filter fun (a, b) => !c.inOut a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"output elements {badOut.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside the output changed"
  else if cf.mem.size != c.init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch (c.agree c.expect) c.run

/-- `T = 3`, `K = 4`, `D = 10`, `BLOCK_M = 2`, `BLOCK_DIM = 4`, `σ = 2`: a `2 × 3` grid, ragged in
both axes (the last token block has one masked row, whose lanes would reach the sentinels after
the output; the last column block has two masked columns). Signed inputs in `[-5, 5]`. -/
def ragged : Case :=
  { T := 3, K := 4, D := 10, em := 1, ed := 2, sbits := 4611686018427387904, sigma := 2,
    x := fun t k j => (((7 * t + 3 * k + 5 * j) % 11 : Nat) : Int) - 5 }

/-- `T = 2`, `K = 9` (the host's largest top-k), `D = 5`, `BLOCK_M = 1`, `BLOCK_DIM = 8`,
`σ = 1/2`: a nine-step loop, odd sums halved (exact in `bfloat16`), one masked column block. -/
def topk9 : Case :=
  { T := 2, K := 9, D := 5, em := 0, ed := 3, sbits := 4602678819172646912, sigma := 1 / 2,
    x := fun t k j => (((5 * t + 2 * k + 3 * j + 1) % 13 : Nat) : Int) - 6 }

/-- The host's block sizes `BLOCK_M = 1`, `BLOCK_DIM = 2048`, `σ = 1`: `T = 2`, `K = 2`, `D = 3`,
two programs each with `2045` masked columns; one element sums to zero (`3 + -3`). -/
def host : Case :=
  { T := 2, K := 2, D := 3, em := 0, ed := 11, sbits := 4607182418800017408, sigma := 1,
    x := fun t k j => if t = 1 ∧ j = 0 then (if k = 0 then 3 else -3) else (t + 1 : Int) * (k + 2) - j }

-- The instances are covered (`Inst.ok`, by the claim's `mk_ok`; `Pre` holds of `init`: `input`
-- holds `inVal`, and every byte of both buffers is present). `Inst.ok` does not read `inVal`.
theorem Case.ok_of (c : Case)
    (h : (Claims.MoeSumReduce.mk c.T c.K c.D c.sbits c.em c.ed).ok) : c.inst.ok where
  bm_pow2 := h.bm_pow2
  bd_pow2 := h.bd_pow2
  block_le := h.block_le
  T_pos := h.T_pos
  K_pos := h.K_pos
  D_pos := h.D_pos
  T_lt := h.T_lt
  K_lt := h.K_lt
  K_le := h.K_le
  D_lt := h.D_lt
  D_dim := h.D_dim
  T_ctx := h.T_ctx
  stride_lt := h.stride_lt
  grid1_lt := h.grid1_lt
  grid_size_lt := h.grid_size_lt
  sbits_lt := h.sbits_lt
  input_len := h.input_len
  output_len := h.output_len
  disj := h.disj

macro "covered" : tactic => `(tactic| exact Case.ok_of _ (mk_ok (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)))

theorem ragged_ok : ragged.inst.ok := by unfold ragged; covered
theorem topk9_ok : topk9.inst.ok := by unfold topk9; covered
theorem host_ok : host.inst.ok := by unfold host; covered

-- The scales' patterns are the values the expectations use.
#guard [ragged, topk9, host].all fun c => (sigma c.inst).toRat == c.sigma && (sigma32 c.inst).toRat == c.sigma

-- The inputs are encoded exactly, and `input` holds `inVal` (`Pre`'s third part).
#guard [ragged, topk9, host].all fun c => (List.range (c.T * c.K * c.D)).all fun e : Nat =>
  (bfv (c.vals e)).toRat == (c.vals e : Rat) &&
  ((CMem.toMem c.init).read .bf16 (c.inst.input.ptr e).addr).map DType.bf16.toBits ==
    some (DType.bf16.toBits (c.inst.inVal e))

-- The inputs are signed (the casts give negative values).
#guard ragged.x 0 0 0 == -5 && topk9.x 0 0 0 == -5 && host.x 1 1 0 == -3

-- No expectation equals the sentinel (an unwritten element would be caught).
#guard [ragged, topk9, host].all fun c => (List.range (c.T * c.D)).all fun e : Nat =>
  c.expect (e / c.D) (e % c.D) != (sentinel : Rat)

def verdictRagged : Verdict := ragged.verdict
def verdictTopk9 : Verdict := topk9.verdict
def verdictHost : Verdict := host.verdict

#eval IO.println (verdictRagged.line "moe_sum_reduce_bf16/ragged")
#eval IO.println (verdictTopk9.line "moe_sum_reduce_bf16/topk9")
#eval IO.println (verdictHost.line "moe_sum_reduce_bf16/host")
#guard verdictRagged == .certified
#guard verdictTopk9 == .certified
#guard verdictHost == .certified

-- A wrong expectation is caught: `agree` is not vacuous (the scale forgotten; one element off).
#guard (Verdict.ofLaunch (ragged.agree fun t j => ragged.expect t j / 2) ragged.run).isDiscrepancy
#guard (Verdict.ofLaunch (topk9.agree fun t j => if t = 1 ∧ j = 4 then topk9.expect t j + 1 / 2
  else topk9.expect t j) topk9.run).isDiscrepancy
