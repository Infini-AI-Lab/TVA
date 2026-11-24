import Solvers
import Verdict
import Lean.Data.Json

/-!
# Single-operation tests: one `tl.*` definition against the device

Phase 1 of `validation/PLAN.md`, through the executor. A case (`cases.json`, written by
`cases.py`) is one `tl.*` operation on one dtype and its inputs, as bit patterns; the device's
outputs (`device.json`, written by `device.py` from a one-operation Triton kernel) are compared
with what `exec` makes of the model's definition on the same inputs.

* **What is compared.** Lane by lane, with the relation the definition allows: equal bits, or,
  where the definition is `upToNaN`, NaN against NaN (`FloatV.NaNEq`); for `maximum`/`minimum`
  the library's own `MaxRel` decided on the device's value. Never looser than the definition.
* **Undefined behaviour.** When `exec` predicts undefined behaviour for a case, the case is run
  again lane by lane: the lanes the model calls undefined are counted, not compared, and the
  others are compared.
* **Platform operations** (`/`, `exp`, `tl.dot`, conversions into fp8, `dot_add`): the oracle
  answers with the device's outputs, as `Ans.assumed`, so the comparison holds by construction;
  what such a case tests is everything around the platform value (shapes, the order of
  elements, promotions). The values themselves are the device facts' (`evidence/num-*`); the
  verdict is `conditional`, with the number of values assumed.
* **Reductions** (`sum`): the oracle offers the trees in `candidates` (each checked by the
  solver, so a match is certified); a case agrees when one of them gives the device's value. No
  candidate agreeing is `inconclusive`, not a finding.
* **Scans** (`cumsum`, `cumprod`, `cummax`, `cummin`, with `_ax1` and `_rev`): on an axis of at most 8 elements the
  oracle searches every tree a position allows for the device's value (`scanTrees`), so a
  position no tree explains is a discrepancy; longer axes are run on exact inputs.
* **Verdicts** (`Verdict`, `validation/Verdict.lean`): each case's `status` is its verdict,
  `certified`, `conditional`, `discrepancy` (a candidate finding, with up to 8 example lanes),
  `inconclusive` or `unsupported`; or, for a case the harness could not run, `bad_case`,
  `no_device`, `unknown_op`, or `device_error` (`device.py` raised, usually because Triton
  rejects the kernel; `model` is the model's own outcome on the case, `modelOutcome`: `done`
  means the model accepts what Triton rejects). `summary` counts them.
* **Mutants** (`mutants`): wrong copies of a definition, run on the same cases. A mutant is
  caught when its verdict is a discrepancy; one that agrees with the device everywhere is a gap
  in the inputs. With `--self`, the original model's own outputs stand in for the device (no
  GPU): a mutant is caught if the inputs tell it from the original.

  lean --run Ops.lean CASES.json (DEVICE.json | --self) RESULTS.json
  (or `run_test.py --ops …`, which compiles what it imports and summarises the results)
-/

open Lean TVA TVA.Validation

namespace TVA.Validation.Ops

/-! ## Values from bits -/

def dtype? : String → Option DType
  | "i1" => some .i1 | "i8" => some .i8 | "i16" => some .i16 | "i32" => some .i32
  | "i64" => some .i64 | "u8" => some .u8 | "u16" => some .u16 | "u32" => some .u32
  | "u64" => some .u64 | "f16" => some .f16 | "bf16" => some .bf16 | "f32" => some .f32
  | "f64" => some .f64 | "f8e5" => some .f8e5 | "f8e4nv" => some .f8e4nv
  | _ => none

/-- A rank-1 tensor of `n` elements of dtype `d` from bit patterns. -/
def vec (d : DType) (n : Nat) (bs : Array Nat) : Tensor [n] d.Val :=
  Tensor.of1 fun i => d.ofBits (bs.getD i.1 0)

/-- A rank-2 tensor `[m, n]`, row-major. -/
def mat (d : DType) (m n : Nat) (bs : Array Nat) : Tensor [m, n] d.Val :=
  Tensor.of2 fun i j => d.ofBits (bs.getD (i.1 * n + j.1) 0)

def bitsOf {s : Shape} (d : DType) (t : Tensor s d.Val) : Array Nat :=
  (t.toList.map d.toBits).toArray

/-- The position of an index in row-major order (as `Tensor.toList`). -/
def flat : (s : Shape) → Idx s → Nat
  | [], _ => 0
  | _ :: s, (i, j) => i.1 * s.foldr (· * ·) 1 + flat s j

/-- The device's answer to a platform question, read from the case's outputs. -/
def platOut (dev : Array Nat) : (q : PlatQ) → Option q.Out
  | .math _ f _ => ofFloat f
  | .div f _ _ => ofFloat f
  | .cvt _ g _ _ => ofFloat g
  | @PlatQ.dot cfg _ N _ _ _ _ => some (Tensor.of2 fun i j => cfg.acc.ofBits (dev.getD (i.1 * N + j.1) 0))
  | @PlatQ.dotScaled _ _ N _ _ _ _ _ _ => some (Tensor.of2 fun i j => FloatV.ofPat (dev.getD (i.1 * N + j.1) 0))
where
  ofFloat {s : Shape} (f : Format) : Option (Tensor s (FloatV f)) :=
    some ⟨fun i => FloatV.ofPat (dev.getD (flat s i) 0)⟩

/-! ## The operations -/

/-- How a lane's model value and device value are compared, given the lane's inputs. -/
abbrev Cmp := (ins : List Nat) → (model dev : Nat) → Bool

def exact : Cmp := fun _ a b => a == b

/-- `FloatV.NaNEq`, the relation of `upToNaN`. -/
def nanEq (d : DType) : Cmp := fun _ a b =>
  a == b || (match d.format? with
    | some f => f.isNaN a && f.isNaN b
    | none => false)

instance {f : Format} (isMax : Bool) (x y r : FloatV f) : Decidable (FloatV.MaxRel isMax x y r) := by
  unfold FloatV.MaxRel FloatV.NaNEq; infer_instance

/-- `FloatV.MaxRel isMax` decided on the device's value. -/
def maxRel (isMax : Bool) (d : DType) : Cmp := fun ins _ b =>
  match d.format?, ins with
  | some f, [x, y] => decide (FloatV.MaxRel isMax (FloatV.ofPat (f := f) x) (FloatV.ofPat y) (FloatV.ofPat b))
  | _, _ => false

/-- A case's program: from its dtypes, sizes and inputs, the model's outputs as bits. -/
structure Op where
  /-- The program, or `none` if the case does not fit (dtype, arity). -/
  prog : (dts : List DType) → (dims : List Nat) → (ins : List (Array Nat)) → Option (Prog (Array Nat))
  /-- Lane comparison, for the output dtype. -/
  cmp : (dts : List DType) → Cmp := fun dts => nanEq (dts.getLastD .i32)
  /-- The number of elements of input `j`, from the case's dims. -/
  inLen : (dims : List Nat) → (j : Nat) → Nat := fun dims _ => dims.headD 0
  /-- The number of output elements. -/
  outLen : (dims : List Nat) → Nat := fun dims => dims.headD 0
  /-- Elementwise: undefined behaviour can be localised by running lanes alone. -/
  lanewise : Bool := true
  /-- For a scan, the oracle's tree for the output element at each place, from the case's dtypes,
  dims, inputs and the device's outputs (`TQuery.scanTree`); `none` for the solver's default. -/
  trees : (dts : List DType) → (dims : List Nat) → (ins : List (Array Nat)) → (dev : Array Nat) →
    Nat → Option Tree := fun _ _ _ _ _ => none

/-- An elementwise operation on float tensors. -/
def floatN (k : Nat) (g : ∀ {f : Format} {n : Nat}, List (Tensor [n] (FloatV f)) → Option (Prog (Tensor [n] (FloatV f)))) :
    Op where
  prog
    | [d], [n], ins => match d.format? with
      | some f =>
        if ins.length != k then none else
        match g (f := f) (ins.map fun bs => Tensor.of1 fun i => FloatV.ofPat (bs.getD i.1 0)) with
        | some p => some ((fun t : Tensor [n] (FloatV f) => (t.toList.map (·.pat)).toArray) <$> p)
        | none => none
      | none => none
    | _, _, _ => none

def float1 (g : ∀ {f : Format} {n : Nat}, Tensor [n] (FloatV f) → Prog (Tensor [n] (FloatV f))) : Op :=
  floatN 1 fun | [x] => some (g x) | _ => none
def float2 (g : ∀ {f : Format} {n : Nat}, Tensor [n] (FloatV f) → Tensor [n] (FloatV f) → Prog (Tensor [n] (FloatV f))) : Op :=
  floatN 2 fun | [x, y] => some (g x y) | _ => none
def float3 (g : ∀ {f : Format} {n : Nat}, Tensor [n] (FloatV f) → Tensor [n] (FloatV f) → Tensor [n] (FloatV f) → Prog (Tensor [n] (FloatV f))) : Op :=
  floatN 3 fun | [x, y, z] => some (g x y z) | _ => none

/-- The signedness and width of an integer dtype. -/
def intT : DType → Option (Bool × Nat)
  | .i8 => some (true, 8) | .i16 => some (true, 16) | .i32 => some (true, 32) | .i64 => some (true, 64)
  | .u8 => some (false, 8) | .u16 => some (false, 16) | .u32 => some (false, 32) | .u64 => some (false, 64)
  | _ => none

/-- An elementwise operation on two integer tensors. -/
def int2 (g : ∀ {sg : Bool} {w n : Nat}, Tensor [n] (IntV sg w) → Tensor [n] (IntV sg w) → Prog (Tensor [n] (IntV sg w))) :
    Op where
  prog
    | [d], [n], [xs, ys] => match intT d with
      | some (sg, w) =>
        let v (bs : Array Nat) : Tensor [n] (IntV sg w) := Tensor.of1 fun i => IntV.ofNat (bs.getD i.1 0)
        some ((fun t : Tensor [n] (IntV sg w) => (t.toList.map (·.bits.toNat)).toArray) <$> g (v xs) (v ys))
      | none => none
    | _, _, _ => none
  cmp _ := exact

/-- `ConvRel` on the device's value, where the platform plays no part: the same bits within a
dtype; up to NaN payload between standard float formats; from fp8, exact on finite inputs. -/
def convCmp : List DType → Cmp
  | [d, d'] => fun ins a b =>
    if d == d' then a == b else
    match d.format?, d'.format? with
    | some f, some g =>
      if f.F8 then !(FloatV.ofPat (f := f) (ins.headD 0)).isFinite || a == b
      else if g.F8 then a == b
      else nanEq d' ins a b
    | _, _ => a == b
  | _ => fun _ _ _ => false

/-- `x.to(d')`. -/
def convertOp : Op where
  prog
    | [d, d'], [n], [xs] => some (bitsOf d' <$> tl.convert d d' (vec d n xs))
    | _, _, _ => none
  cmp := convCmp

/-- `tl.sum(x)` of a rank-1 float tensor. -/
def sumOp : Op where
  prog
    | [d], [n], [xs] => match d.format? with
      | some f => some ((fun r : FloatV f => #[r.pat]) <$> tl.sum (Tensor.of1 fun i : Fin n => FloatV.ofPat (xs.getD i.1 0)))
      | none => none
    | _, _, _ => none
  lanewise := false
  outLen _ := 1

/-- The sizes of `a`, `b` and `acc` for dims `[M, N, K]`. -/
def dotInLen : List Nat → Nat → Nat
  | [M, N, K], j => [M * K, K * N, M * N].getD j 0
  | _, _ => 0

def dotOutLen : List Nat → Nat
  | [M, N, _] => M * N
  | _ => 0

/-- `tl.dot(a, b, acc)` with `a : [M, K]`, `b : [K, N]`, `acc : [M, N]`; dtypes `a`, `b`, `acc`. -/
def dotOp (precision : InputPrecision) : Op where
  prog
    | [da, db, dc], [M, N, K], [as, bs, cs] =>
      let cfg : DotCfg := { a := da, b := db, acc := dc, precision := precision }
      some (bitsOf dc <$> tl.dot cfg (mat da M K as) (mat db K N bs) (mat dc M N cs))
    | _, _, _ => none
  lanewise := false
  inLen := dotInLen
  outLen := dotOutLen

/-- `tl.dot(a, b) + c`. -/
def dotAddOp (precision : InputPrecision) : Op where
  prog
    | [da, db, dc], [M, N, K], [as, bs, cs] =>
      let cfg : DotCfg := { a := da, b := db, acc := dc, precision := precision }
      some (bitsOf dc <$> tl.dot_add cfg (mat da M K as) (mat db K N bs) (mat dc M N cs))
    | _, _, _ => none
  lanewise := false
  inLen := dotInLen
  outLen := dotOutLen

/-! ### Scans: `tl.cumsum`, `tl.cumprod`, running maxima and minima

A case `[d, d']` scans an input of dtype `d` after Triton's promotion to `d'` (`standard.py`:
`bfloat16` to `float32`; for `tl.cumsum`, integers narrower than 32 bits to 32 bits of the same
signedness), along `axis` of a rank-1 (`[n]`) or rank-2 (`[m, n]`, row-major) block.

**The trees.** The model allows, at each output position, any tree over its leaves in order
(`ScanLeaves`). On an axis of at most `scanSearchMax` elements the oracle tries every such tree
(every bracketing of the leaves) for the device's value at that position and supplies the first
that gives it (`TQuery.scanTree`); the solver checks it. The search is exhaustive, so a position no
tree explains is a discrepancy, not a missing candidate. Longer axes get the solver's sequential
tree, and their cases use inputs whose partial results are exact (`cases.py`), on which every tree
gives the same value. -/

/-- Every tree over the leaves `l` in order (Catalan many). -/
partial def bracketings : List Nat → List Tree
  | [] => []
  | [a] => [.leaf a]
  | l => (List.range (l.length - 1)).flatMap fun k =>
      (bracketings (l.take (k + 1))).flatMap fun a => (bracketings (l.drop (k + 1))).map (.node a)

/-- The longest axis on which the oracle searches every tree. -/
def scanSearchMax : Nat := 8

/-- For the element at place `L` of a block of dims `[n]` or `[m, n]` scanned along `axis`: the
axis' size, the element's coordinate along it, and the place of coordinate `k` on its line. -/
def scanLine : List Nat → Nat → Nat → Option (Nat × Nat × (Nat → Nat))
  | [n], 0, L => some (n, L, id)
  | [_, n], 1, L => some (n, L % n, fun k => L / n * n + k)
  | [m, n], 0, L => some (m, L / n, fun k => k * n + L % n)
  | _, _, _ => none

/-- A scan's combine: `+` (`tl.cumsum`), `*` (`tl.cumprod`), or `tl.maximum`/`tl.minimum`
(`tl.associative_scan(x, axis, combine)` with a `@triton.jit` combine calling it). -/
inductive ScanKind | sum | prod | max | min
  deriving BEq

/-- The float combine of a scan kind (`Combine`), and its function. -/
def ScanKind.fcombine {f : Format} : ScanKind → Combine (FloatV f)
  | .sum => Combine.addRel false | .prod => Combine.mulRn
  | .max => Combine.maxRel true | .min => Combine.maxRel false

/-- The scan of kind `k` along `axis`, reversed with `rev`, of an input of dtype `d` promoted to
`d'`: `tl.cumsum`, `tl.cumprod`, or `tl.associative_scan` with `Combine.maxRel`. Integer inputs
for `+` and `*` only. -/
def scanProg (k : ScanKind) (axis : Nat) (rev : Bool) (d d' : DType) (dims : List Nat)
    (xs : Array Nat) : Option (Prog (Array Nat)) :=
  if !(dims.length == 1 && axis == 0 || dims.length == 2 && axis < 2) then none else
  match d.format?, d'.format?, intT d, intT d' with
  | some f, some g, _, _ =>
    let x : Tensor dims (FloatV g) := ⟨fun i => FloatV.cvt (FloatV.ofPat (f := f) (xs.getD (flat dims i) 0))⟩
    let p := match k with
      | .sum => tl.cumsum x axis rev | .prod => tl.cumprod x axis rev
      | _ => tl.associative_scan x axis k.fcombine rev
    some ((fun t : Tensor dims (FloatV g) => (t.toList.map (·.pat)).toArray) <$> p)
  | _, _, some (sg, w), some (sg', w') =>
    if k == .max || k == .min then none else
    let x : Tensor dims (IntV sg' w') :=
      ⟨fun i => IntV.ofInt (IntV.ofNat (s := sg) (w := w) (xs.getD (flat dims i) 0)).toInt⟩
    let p := if k == .prod then tl.icumprod x axis rev else tl.icumsum x axis rev
    some ((fun t : Tensor dims (IntV sg' w') => (t.toList.map (·.bits.toNat)).toArray) <$> p)
  | _, _, _, _ => none

/-- The oracle's trees for a float scan: at each place, the first bracketing of the position's
leaves whose IEEE evaluation is the device's value (up to NaN payload), when the axis has at most
`scanSearchMax` elements. Integer scans need none (every tree gives the same value). -/
def scanTrees (k : ScanKind) (axis : Nat) (rev : Bool) (dts : List DType) (dims : List Nat)
    (ins : List (Array Nat)) (dev : Array Nat) (L : Nat) : Option Tree :=
  match dts, ins with
  | [d, d'], [xs] => match d.format?, d'.format?, scanLine dims axis L with
    | some f, some g, some (n, j, place) =>
      if n > scanSearchMax then none else
      let leaf (k : Nat) : FloatV g := FloatV.cvt (FloatV.ofPat (f := f) (xs.getD (place k) 0))
      (bracketings (scanLeaves rev n j)).find? fun t =>
        nanEq d' [] (treeEval (k.fcombine (f := g)).fn leaf t).pat (dev.getD L 0)
    | _, _, _ => none
  | _, _ => none

def scanOp (k : ScanKind) (axis : Nat) (rev : Bool) : Op where
  prog
    | [d, d'], dims, [xs] => scanProg k axis rev d d' dims xs
    | _, _, _ => none
  inLen dims _ := dims.foldr (· * ·) 1
  outLen dims := dims.foldr (· * ·) 1
  lanewise := false
  trees := scanTrees k axis rev

def ops : List (String × Op) := [
  ("add", float2 tl.add), ("sub", float2 tl.sub), ("mul", float2 tl.mul),
  ("div_rn", float2 tl.div_rn), ("div", float2 tl.div), ("fma", float3 tl.fma),
  ("maximum", { float2 tl.maximum with cmp := fun dts => maxRel true (dts.getLastD .i32) }),
  ("minimum", { float2 tl.minimum with cmp := fun dts => maxRel false (dts.getLastD .i32) }),
  ("neg", float1 tl.neg), ("abs", float1 tl.abs), ("sqrt_rn", float1 tl.sqrt_rn),
  ("exp", float1 tl.exp), ("exp2", float1 tl.exp2), ("log", float1 tl.log),
  ("log2", float1 tl.log2), ("sqrt", float1 tl.sqrt), ("rsqrt", float1 tl.rsqrt),
  ("idiv", int2 tl.idiv), ("irem", int2 tl.irem),
  ("shl", int2 tl.shl), ("ashr", int2 tl.ashr), ("lshr", int2 tl.lshr),
  ("convert", convertOp), ("sum", sumOp),
  ("dot_ieee", dotOp .ieee), ("dot_tf32", dotOp .tf32),
  ("dot_add_ieee", dotAddOp .ieee), ("dot_add_tf32", dotAddOp .tf32),
  ("cumsum", scanOp .sum 0 false), ("cumsum_rev", scanOp .sum 0 true),
  ("cumsum_ax1", scanOp .sum 1 false), ("cumsum_ax1_rev", scanOp .sum 1 true),
  ("cumprod", scanOp .prod 0 false), ("cumprod_rev", scanOp .prod 0 true),
  ("cumprod_ax1", scanOp .prod 1 false), ("cumprod_ax1_rev", scanOp .prod 1 true),
  ("cummax", scanOp .max 0 false), ("cummax_rev", scanOp .max 0 true),
  ("cummax_ax1", scanOp .max 1 false), ("cummin", scanOp .min 0 false),
  ("cummin_rev", scanOp .min 0 true), ("cummin_ax1", scanOp .min 1 false)]

/-! ## Mutants: wrong copies of definitions, as a transcription might have them

A mutant is compared by its own value (`nanEq`), never by the original's relation. A mutant that
does not apply to a case (`bad_case`) is not counted. Some mutants are equivalent on some dtypes
(floor and truncating division on unsigned integers; rounding of an exact conversion). -/

/-- `x // y` rounding toward negative infinity (Python's), not toward zero. -/
def floorDiv {sg : Bool} {w n : Nat} (x y : Tensor [n] (IntV sg w)) : Prog (Tensor [n] (IntV sg w)) :=
  .pick (.lanes [n] _ fun i => if (y.get i).toInt = 0 then none
      else some (IntV.ofInt ((x.get i).toInt.fdiv (y.get i).toInt))) .ret

/-- `x % y` with the sign of the divisor (Python's), not of the dividend. -/
def floorRem {sg : Bool} {w n : Nat} (x y : Tensor [n] (IntV sg w)) : Prog (Tensor [n] (IntV sg w)) :=
  .pick (.lanes [n] _ fun i => if (y.get i).toInt = 0 then none
      else some (IntV.ofInt ((x.get i).toInt.fmod (y.get i).toInt))) .ret

/-- A result that is subnormal flushed to a zero of its sign. -/
def ftz {f : Format} (x : FloatV f) : FloatV f :=
  if f.expField x.pat == 0 && f.fracField x.pat != 0 then FloatV.ofPat (x.pat / 2 ^ (f.e + f.m) * 2 ^ (f.e + f.m)) else x

def mutants : List (String × String × Op) := [
  ("add", "flushes subnormal results", float2 fun x y => tl.upToNaN (Tensor.map2 (fun a b => ftz (FloatV.add a b)) x y)),
  ("mul", "flushes subnormal results", float2 fun x y => tl.upToNaN (Tensor.map2 (fun a b => ftz (FloatV.mul a b)) x y)),
  ("fma", "two roundings, not one", float3 fun x y z => tl.upToNaN (Tensor.map3 (fun a b c => FloatV.add (FloatV.mul a b) c) x y z)),
  ("maximum", "propagates NaN", float2 fun x y => tl.upToNaN (Tensor.map2 FloatV.maxNaN x y)),
  ("minimum", "the maximum", float2 fun x y => tl.upToNaN (Tensor.map2 FloatV.max x y)),
  ("sub", "operands swapped", float2 fun x y => tl.sub y x),
  ("idiv", "rounds toward -∞", int2 floorDiv),
  ("irem", "sign of the divisor", int2 floorRem),
  ("ashr", "logical shift", int2 fun x n => tl.lshr x n),
  ("lshr", "arithmetic shift", int2 fun x n => tl.ashr x n),
  ("convert", "rounds toward zero between floats", { convertOp with prog := fun
    | [d, d'], [n], [xs] => match d.format?, d'.format? with
      | some f, some g => some (pure (bitsOf d' (vec d' n ((vec d n xs).toList.map fun v =>
          (Format.cvt f g .rz false (d.toBits v))).toArray)))
      | _, _ => none
    | _, _, _ => none }),
  ("cumsum_rev", "reverse ignored", scanOp .sum 0 false),
  ("cumsum_ax1", "the other axis", scanOp .sum 0 false),
  ("cumprod", "the sum, not the product", scanOp .sum 0 false),
  ("cummax", "the minimum", scanOp .min 0 false)]

/-! ## Running a case -/

/-- A platform that relates everything: platform values are not checked here (they are the
device facts'); only `sumDepth` matters, and it is the device fact's (`⌈log₂ n⌉`). -/
def Pany : Platform where
  math _ _ _ _ := True
  div _ _ _ _ := True
  dot := @fun _ _ _ _ _ _ _ _ => True
  cvt _ _ _ _ _ := True
  atomicAdd _ _ _ _ := True
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => True
  sumDepth _ := 0

def E : Env := { P := Pany, fusion := true }

instance : Inhabited Tree := ⟨.leaf 0⟩

/-- Candidate trees for a reduction of `n` (a power of two) leaves: halves, and strides (the
order of a warp's butterfly), with a chunk of `c` contiguous leaves reduced first by halves. -/
def halves (base size : Nat) : Tree :=
  if size ≤ 1 then .leaf base else .node (halves base (size / 2)) (halves (base + size / 2) (size / 2))
termination_by size
decreasing_by all_goals omega

def strided (start step count : Nat) (leafT : Nat → Tree) : Tree :=
  if count ≤ 1 then leafT start
  else .node (strided start (2 * step) (count / 2) leafT) (strided (start + step) (2 * step) (count / 2) leafT)
termination_by count
decreasing_by all_goals omega

def candidates (n : Nat) : List Tree :=
  let cs := (List.range 6).map (2 ^ ·) |>.filter (· ≤ n)
  [halves 0 n] ++ cs.map fun c => strided 0 1 (n / c) fun t => halves (t * c) c

/-- The oracle: trees from `tree`, a scan position's tree from `scanT` (by the element's place),
the device's values for platform questions, at every request. -/
def oracle (dev : Array Nat) (conv : Nat → Option Nat) (tree : Option Tree)
    (scanT : Nat → Option Tree := fun _ => none) : Oracle :=
  fun _ _ q => match q with
    | .tree _ => tree
    | .scanTree _ _ _ lane => scanT lane
    | .plat q => platOut dev q
    | .convert _ d' x => (conv (DType.toBits _ x)).map d'.ofBits
    | _ => none

/-- Disabled lanes read zero bits (no case masks a load). -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

def runProg (O : Oracle) (p : Prog (Array Nat)) : Res (Array Nat) × List Record :=
  exec E O fill0 100 0 ∅ p

structure Case where
  id : String
  op : String
  dts : List DType
  dims : List Nat
  ins : List (Array Nat)

def Case.lanes (c : Case) : Nat := c.dims.headD 0

/-- The lanes where the model's outputs `out` and the device's `dev` are not related by `cmp`. -/
def badLanes (cmp : Cmp) (ins : List (Array Nat)) (out dev : Array Nat) : List Nat :=
  (List.range (max out.size dev.size)).filter fun i => match out[i]?, dev[i]? with
    | some a, some b => !cmp (ins.map (·.getD i 0)) a b
    | _, _ => true

/-- Up to 8 differing lanes, `lane` counting from `base`, as JSON. -/
def examples (ins : List (Array Nat)) (out dev : Array Nat) (bad : List Nat) (base : Nat := 0) : List Json :=
  (bad.take 8).map fun i => Json.mkObj [("lane", toJson (base + i)),
    ("in", toJson (ins.map (·.getD i 0))), ("model", toJson (out.getD i 0)), ("device", toJson (dev.getD i 0))]

/-- The verdict of a run against `dev`. -/
def verdictOf (cmp : Cmp) (ins : List (Array Nat)) (dev : Array Nat) (x : Res (Array Nat) × List Record) : Verdict :=
  Verdict.ofRes (fun out _ =>
    let bad := badLanes cmp ins out dev
    if bad.isEmpty then none else some s!"{bad.length} of {out.size} lanes differ from the device") x

/-- A case's report: its verdict (`status`, and `result` with its details), and `fields`. -/
def report (c : Case) (v : Verdict) (fields : List (String × Json)) : Json :=
  Json.mkObj ([("id", toJson c.id), ("op", toJson c.op), ("status", toJson v.name),
    ("result", toJson v)] ++ fields)

/-- A case the harness could not run (not a verdict): `bad_case`, `no_device`, `unknown_op`,
`device_error` (with the error in `what`, and other `fields`). -/
def harness (c : Case) (status : String) (what : Option String := none)
    (fields : List (String × Json) := []) : Json :=
  Json.mkObj ([("id", toJson c.id), ("op", toJson c.op), ("status", toJson status)] ++
    ((what.map fun w => [("what", toJson w)]).getD []) ++ fields)

/-- The verdict of a case against `dev`, as JSON (`Verdict`, `validation/Verdict.lean`):

* one run, with the oracle answering platform questions with `dev`: its verdict;
* a reduction: one run per candidate tree; the first that agrees gives the verdict; if none
  does, `inconclusive` (the device's tree is not among the candidates), never a discrepancy;
* a scan: one run, with each position's tree found by the oracle (`scanTrees`);
* a predicted undefined behaviour, for an elementwise operation: each lane run alone; the lanes
  the model calls undefined are counted (`ub_lanes`), not compared, and the verdict joins the
  others' (`Verdict.join`). For any other operation, `inconclusive`: the device's output on
  inputs the model calls undefined is not evidence either way. -/
def runCase (o : Op) (c : Case) (dev : Array Nat) : Json := Id.run do
  let conv (x : Nat) : Option Nat := match (c.ins.headD #[]).findIdx? (· == x) with
    | some i => dev[i]?
    | none => none
  let cmp := o.cmp c.dts
  let badIn := (List.range c.ins.length).any fun j => (c.ins.getD j #[]).size != o.inLen c.dims j
  if c.ins.isEmpty || badIn then
    return harness c "bad_case"
  if dev.size != o.outLen c.dims then return harness c "bad_device"
  let some p := o.prog c.dts c.dims c.ins | return harness c "bad_case"
  let trees : List (Option Tree) := if c.op == "sum" then (candidates c.lanes).map some else [none]
  let mut last : Option (Verdict × List Json) := none
  let mut sawDiff := false
  for t in trees do
    let x := runProg (oracle dev conv t (o.trees c.dts c.dims c.ins dev)) p
    if let .ub _ := x.1 then
      if !o.lanewise then
        return report c (.inconclusive "the model predicts undefined behaviour (not localised)") []
      return localise o c conv cmp dev
    let v := verdictOf cmp c.ins dev x
    if v.agrees then
      return report c v [("lanes", toJson (match x.1 with | .done out _ => out.size | _ => 0))]
    let ex := match x.1 with
      | .done out _ => examples c.ins out dev (badLanes cmp c.ins out dev)
      | _ => []
    sawDiff := sawDiff || v.isDiscrepancy
    last := some (v, ex)
  let (v, ex) := last.getD (.inconclusive "no candidate", [])
  let v := if trees.length > 1 && sawDiff then
    Verdict.inconclusive s!"no candidate tree of {trees.length} gives the device's value" else v
  return report c v [("examples", Json.arr ex.toArray)]
where
  localise (o : Op) (c : Case) (conv : Nat → Option Nat) (cmp : Cmp) (dev : Array Nat) : Json := Id.run do
    let mut v := Verdict.certified
    let mut ubLanes : Nat := 0
    let mut nbad : Nat := 0
    let mut ex : Array Json := #[]
    for i in List.range c.lanes do
      let ins1 := c.ins.map fun bs => #[bs.getD i 0]
      let dev1 := #[dev.getD i 0]
      let some p1 := o.prog c.dts (1 :: c.dims.tail) ins1 | return harness c "bad_case"
      let x := runProg (oracle dev1 conv none) p1
      if let .ub _ := x.1 then ubLanes := ubLanes + 1; continue
      let v1 := verdictOf cmp ins1 dev1 x
      if v1.isDiscrepancy then
        nbad := nbad + 1
        if let .done out _ := x.1 then
          if ex.size < 8 then ex := ex ++ (examples ins1 out dev1 [0] i).toArray
      v := v.join v1
    let vf := if ubLanes == c.lanes then .inconclusive "the model calls every lane undefined" else
      match v with
      | .discrepancy n _ => .discrepancy n s!"{nbad} of {c.lanes} lanes differ from the device"
      | w => w
    return report c vf [("lanes", toJson c.lanes), ("ub_lanes", toJson ubLanes),
      ("compared", toJson (c.lanes - ubLanes)), ("bad", toJson nbad),
      ("examples", Json.arr ex)]

/-- The model's own outputs on a case (for `--self`): lanes the model calls undefined give `0`. -/
def selfOut (o : Op) (c : Case) : Option (Array Nat) :=
  match o.prog c.dts c.dims c.ins with
  | none => none
  | some p =>
  match (runProg (oracle #[] (fun _ => none) ((candidates c.lanes).head?)) p).1 with
  | .done out _ => some out
  | .ub _ =>
    if !o.lanewise then none else
    some <| (List.range c.lanes).toArray.map fun i =>
      match o.prog c.dts (1 :: c.dims.tail) (c.ins.map fun bs => #[bs.getD i 0]) with
      | some p1 => match (runProg (oracle #[] (fun _ => none) none) p1).1 with
        | .done out _ => out.getD 0 0
        | _ => 0
      | none => 0
  | _ => none

/-- The model's outcome on a case, without a device: `done` (it gives a result), `ub` (undefined
behaviour: the model rejects the case too), `inconclusive` (the run stopped without either, for
example a platform value no oracle supplied), or `bad_case`. From the run itself, not `selfOut`,
which stands in zeros for undefined lanes. -/
def modelOutcome (o : Op) (c : Case) : String :=
  match o.prog c.dts c.dims c.ins with
  | none => "bad_case"
  | some p => match (runProg (oracle #[] (fun _ => none) ((candidates c.lanes).head?)) p).1 with
    | .done _ _ => "done"
    | .ub _ => "ub"
    | .stuck _ _ | .outOfFuel => "inconclusive"

/-- `modelOutcome` of `op` on two lanes of `dt` with inputs `x` (one operand per element). -/
def outcomeOf (op : String) (dt : DType) (x : List Nat) : String :=
  match ops.lookup op with
  | none => "unknown_op"
  | some o => modelOutcome o ⟨"t", op, [dt], [2], x.map fun v => #[v, v]⟩

-- `tl.div_rn` and `tl.sqrt_rn` are undefined off `float32` (as Triton rejects them), defined on
-- it; `+` is defined on `float16`; `tl.exp` is `done` on an assumed platform value.
#guard outcomeOf "div_rn" .f16 [0x3c00, 0x3c00] == "ub"
#guard outcomeOf "div_rn" .bf16 [0x3f80, 0x3f80] == "ub"
#guard outcomeOf "sqrt_rn" .f16 [0x3c00] == "ub"
#guard outcomeOf "sqrt_rn" .bf16 [0x3f80] == "ub"
#guard outcomeOf "div_rn" .f32 [0x3f800000, 0x3f800000] == "done"
#guard outcomeOf "sqrt_rn" .f32 [0x3f800000] == "done"
#guard outcomeOf "add" .f16 [0x3c00, 0x3c00] == "done"
#guard outcomeOf "exp" .f32 [0x3f800000] == "done"
-- `selfOut` is no measure of acceptance: it fills the undefined lanes with zeros.
#guard ((ops.lookup "div_rn").bind fun o =>
  selfOut o ⟨"t", "div_rn", [.f16], [2], [#[0x3c00, 0x3c00], #[0x3c00, 0x3c00]]⟩).isSome

def parseCase (j : Json) : Except String Case := do
  let id ← j.getObjValAs? String "id"
  let op ← j.getObjValAs? String "op"
  let dts ← (← j.getObjValAs? (List String) "dtypes").mapM fun s =>
    (dtype? s).elim (.error s!"dtype {s}") pure
  let dims ← j.getObjValAs? (List Nat) "dims"
  let ins ← j.getObjValAs? (List (Array Nat)) "inputs"
  pure { id, op, dts, dims, ins }

end TVA.Validation.Ops

open TVA.Validation.Ops in
def main (args : List String) : IO UInt32 := do
  let [casesF, devF, outF] := args
    | IO.eprintln "usage: lean --run Ops.lean CASES.json (DEVICE.json | --self) RESULTS.json"; return 2
  let cj ← IO.ofExcept (Json.parse (← IO.FS.readFile casesF))
  let cases ← IO.ofExcept ((← IO.ofExcept (cj.getArr?)).toList.mapM parseCase)
  let self := devF == "--self"
  let devJ ← if self then pure Json.null else IO.ofExcept (Json.parse (← IO.FS.readFile devF))
  let device (o : Op) (c : Case) : Option (Array Nat) :=
    if self then selfOut o c else (devJ.getObjValAs? (Array Nat) c.id).toOption
  let devError (c : Case) : Option String :=
    if self then none else (devJ.getObjValAs? String (c.id ++ ".error")).toOption
  let mut results : Array Json := #[]
  let mut muts : Array Json := #[]
  let mut counts : Std.HashMap String Nat := {}
  for c in cases do
    let r := match ops.lookup c.op with
      | none => harness c "unknown_op"
      | some o => match device o c with
        | none => match devError c with
          | some e => harness c "device_error" e [("model", toJson (modelOutcome o c))]
          | none => harness c "no_device"
        | some dev => runCase o c dev
    let st := (r.getObjValAs? String "status").toOption.getD "?"
    IO.println s!"{c.id}: {st}"
    counts := counts.insert st (counts.getD st 0 + 1)
    results := results.push r
    let some o := ops.lookup c.op | continue
    let some dev := device o c | continue
    for (name, why, m) in mutants do
      if name == c.op then
        let s := (runCase m c dev).getObjValAs? String "status" |>.toOption.getD "?"
        if s != "bad_case" && s != "bad_device" then
          muts := muts.push (Json.mkObj [("id", c.id), ("op", name), ("mutant", why),
            ("caught", s == "discrepancy"), ("status", s)])
  let summary := Json.mkObj (counts.toList.map fun (k, n) => (k, toJson n))
  IO.FS.writeFile outF (Json.mkObj [("summary", summary), ("cases", Json.arr results),
    ("mutants", Json.arr muts)]).pretty
  return 0
