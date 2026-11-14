import Exec
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
  answers with the device's outputs, as `Ans.assume`, so the comparison holds by construction;
  what such a case tests is everything around the platform value (shapes, the order of
  elements, promotions). The values themselves are the device facts' (`evidence/num-*`); the
  assumption log says how many were taken.
* **Reductions** (`sum`): the oracle offers the trees in `candidates`; a case matches when one of
  them gives the device's value. No candidate matching is inconclusive, not a finding.
* **Mutants** (`mutants`): wrong copies of a definition, run on the same cases. A mutant that
  matches the device everywhere is a gap in the inputs. With `--self`, the original model's own
  outputs stand in for the device (no GPU): a mutant is caught if the inputs tell it from the
  original.

  lean --run Ops.lean CASES.json (DEVICE.json | --self) RESULTS.json
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

/-- A tensor of shape `s` from bit patterns in row-major order. -/
def ofFlat {s : Shape} (d : DType) (bs : Array Nat) : Tensor s d.Val :=
  ⟨fun i => d.ofBits (bs.getD (flat s i) 0)⟩

/-- A float format's dtype. -/
def fmtDType : Format → Option DType
  | f => [DType.f16, .bf16, .f32, .f64, .f8e5, .f8e4nv].find? (·.format? == some f)

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
  cmp : DType → Cmp := nanEq
  /-- Elementwise: undefined behaviour can be localised by running lanes alone. -/
  lanewise : Bool := true

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

/-- `x.to(d')`. -/
def convertOp : Op where
  prog
    | [d, d'], [n], [xs] => some (bitsOf d' <$> tl.convert d d' (vec d n xs))
    | _, _, _ => none

/-- `tl.sum(x)` of a rank-1 float tensor. -/
def sumOp : Op where
  prog
    | [d], [n], [xs] => match d.format? with
      | some f => some ((fun r : FloatV f => #[r.pat]) <$> tl.sum (Tensor.of1 fun i : Fin n => FloatV.ofPat (xs.getD i.1 0)))
      | none => none
    | _, _, _ => none
  lanewise := false

/-- `tl.dot(a, b, acc)` with `a : [M, K]`, `b : [K, N]`, `acc : [M, N]`; dtypes `a`, `b`, `acc`. -/
def dotOp (precision : InputPrecision) : Op where
  prog
    | [da, db, dc], [M, N, K], [as, bs, cs] =>
      let cfg : DotCfg := { a := da, b := db, acc := dc, precision := precision }
      some (bitsOf dc <$> tl.dot cfg (mat da M K as) (mat db K N bs) (mat dc M N cs))
    | _, _, _ => none
  lanewise := false

/-- `tl.dot(a, b) + c`. -/
def dotAddOp (precision : InputPrecision) : Op where
  prog
    | [da, db, dc], [M, N, K], [as, bs, cs] =>
      let cfg : DotCfg := { a := da, b := db, acc := dc, precision := precision }
      some (bitsOf dc <$> tl.dot_add cfg (mat da M K as) (mat db K N bs) (mat dc M N cs))
    | _, _, _ => none
  lanewise := false

def ops : List (String × Op) := [
  ("add", float2 tl.add), ("sub", float2 tl.sub), ("mul", float2 tl.mul),
  ("div_rn", float2 tl.div_rn), ("div", float2 tl.div), ("fma", float3 tl.fma),
  ("maximum", { float2 tl.maximum with cmp := maxRel true }),
  ("minimum", { float2 tl.minimum with cmp := maxRel false }),
  ("neg", float1 tl.neg), ("abs", float1 tl.abs), ("sqrt_rn", float1 tl.sqrt_rn),
  ("exp", float1 tl.exp), ("exp2", float1 tl.exp2), ("log", float1 tl.log),
  ("log2", float1 tl.log2), ("sqrt", float1 tl.sqrt), ("rsqrt", float1 tl.rsqrt),
  ("idiv", int2 tl.idiv), ("irem", int2 tl.irem),
  ("shl", int2 tl.shl), ("ashr", int2 tl.ashr), ("lshr", int2 tl.lshr),
  ("convert", convertOp), ("sum", sumOp),
  ("dot_ieee", dotOp .ieee), ("dot_tf32", dotOp .tf32),
  ("dot_add_ieee", dotAddOp .ieee), ("dot_add_tf32", dotAddOp .tf32)]

/-! ## Mutants: wrong copies of definitions, as a transcription might have them

A mutant is compared by its own value (`nanEq`), never by the original's relation. A mutant that
does not apply to a case (`bad_case`) is not counted. Some mutants are equivalent on some dtypes
(floor and truncating division on unsigned integers; rounding of an exact conversion). -/

/-- `x // y` rounding toward negative infinity (Python's), not toward zero. -/
def floorDiv {sg : Bool} {w n : Nat} (x y : Tensor [n] (IntV sg w)) : Prog (Tensor [n] (IntV sg w)) :=
  .pick (fun r : Tensor [n] (IntV sg w) => ∀ i, (if (y.get i).toInt = 0 then none
      else some (IntV.ofInt ((x.get i).toInt.fdiv (y.get i).toInt))) = some (r.get i))
    (.lanes fun i => if (y.get i).toInt = 0 then none
      else some (IntV.ofInt ((x.get i).toInt.fdiv (y.get i).toInt))) .ret

/-- `x % y` with the sign of the divisor (Python's), not of the dividend. -/
def floorRem {sg : Bool} {w n : Nat} (x y : Tensor [n] (IntV sg w)) : Prog (Tensor [n] (IntV sg w)) :=
  .pick (fun r : Tensor [n] (IntV sg w) => ∀ i, (if (y.get i).toInt = 0 then none
      else some (IntV.ofInt ((x.get i).toInt.fmod (y.get i).toInt))) = some (r.get i))
    (.lanes fun i => if (y.get i).toInt = 0 then none
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
    | _, _, _ => none })]

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
partial def halves (base size : Nat) : Tree :=
  if size ≤ 1 then .leaf base else .node (halves base (size / 2)) (halves (base + size / 2) (size / 2))

partial def strided (start step count : Nat) (leafT : Nat → Tree) : Tree :=
  if count ≤ 1 then leafT start
  else .node (strided start (2 * step) (count / 2) leafT) (strided (start + step) (2 * step) (count / 2) leafT)

def candidates (n : Nat) : List Tree :=
  let cs := (List.range 6).map (2 ^ ·) |>.filter (· ≤ n)
  [halves 0 n] ++ cs.map fun c => strided 0 1 (n / c) fun t => halves (t * c) c

/-- The oracle: trees from `tree`, the device's values for platform questions. -/
def oracle (dev : Array Nat) (conv : Nat → Option Nat) (tree : Option Tree) : Oracle E where
  hints := fun
    | .tree _ => tree
    | .plat q => platOut dev q
    | .convert _ d' x => (conv (DType.toBits _ x)).map d'.ofBits
    | _ => none
  fill d _ _ := d.ofBits 0

inductive Outcome where
  | done (out : Array Nat) (assumed : Nat)
  | ub
  | gaveUp

def runProg (O : Oracle E) (p : Prog (Array Nat)) : Outcome :=
  match exec E O 100 ∅ p with
  | some (.done out _, l) => .done out l.length
  | some (.ub, _) => .ub
  | none => .gaveUp

structure Case where
  id : String
  op : String
  dts : List DType
  dims : List Nat
  ins : List (Array Nat)

def Case.lanes (c : Case) : Nat := c.dims.headD 0

/-- The result of a case against `dev`, as JSON. -/
def runCase (o : Op) (c : Case) (dev : Array Nat) : Json := Id.run do
  let outD := c.dts.getLastD .i32
  let conv (x : Nat) : Option Nat := match (c.ins.headD #[]).findIdx? (· == x) with
    | some i => dev[i]?
    | none => none
  let cmp := o.cmp outD
  let summary (status : String) (fields : List (String × Json)) : Json :=
    Json.mkObj ([("id", toJson c.id), ("op", toJson c.op), ("status", toJson status)] ++ fields)
  let some p := o.prog c.dts c.dims c.ins | return summary "bad_case" []
  let trees : List (Option Tree) := if c.op == "sum" then (candidates c.lanes).map some else [none]
  let mut last := Outcome.gaveUp
  for t in trees do
    last := runProg (oracle dev conv t) p
    match last with
    | .done out a =>
      let bad := (List.range out.size).filter fun i =>
        !cmp (c.ins.map (·.getD i 0)) (out.getD i 0) (dev.getD i 0)
      if bad.isEmpty then
        return summary "match" [("lanes", toJson (out.size)), ("assumed", toJson (a))]
      if t.isNone || some t == trees.getLast? then
        let ex := (bad.take 8).map fun i => Json.mkObj [("lane", toJson (i)),
          ("in", toJson (c.ins.map (·.getD i 0))), ("model", toJson (out.getD i 0)), ("device", toJson (dev.getD i 0))]
        return summary (if c.op == "sum" then "no_tree" else "mismatch")
          [("lanes", toJson (out.size)), ("bad", toJson (bad.length)), ("examples", Json.arr ex.toArray)]
    | _ => pure ()
  match last with
  | .gaveUp => return summary "gave_up" []
  | .done .. => return summary "no_tree" []
  | .ub =>
    if !o.lanewise then return summary "ub" []
    -- localise: each lane alone
    let mut ubLanes : Nat := 0
    let mut bad : Array Json := #[]
    let mut nbad : Nat := 0
    for i in List.range c.lanes do
      let ins1 := c.ins.map fun bs => #[bs.getD i 0]
      let some p1 := o.prog c.dts (1 :: c.dims.tail) ins1 | return summary "bad_case" []
      match runProg (oracle #[dev.getD i 0] conv none) p1 with
      | .ub => ubLanes := ubLanes + 1
      | .done out _ =>
        if !cmp (ins1.map (·.getD 0 0)) (out.getD 0 0) (dev.getD i 0) then
          nbad := nbad + 1
          if bad.size < 8 then
            bad := bad.push (Json.mkObj [("lane", toJson (i)), ("in", toJson (ins1.map (·.getD 0 0))),
              ("model", toJson (out.getD 0 0)), ("device", toJson (dev.getD i 0))])
      | .gaveUp => return summary "gave_up" []
    return summary (if nbad == 0 then "match" else "mismatch")
      [("lanes", toJson (c.lanes)), ("ub_lanes", toJson (ubLanes)), ("bad", toJson (nbad)), ("examples", Json.arr bad)]

/-- The model's own outputs on a case (for `--self`): lanes the model calls undefined give `0`. -/
def selfOut (o : Op) (c : Case) : Option (Array Nat) :=
  match o.prog c.dts c.dims c.ins with
  | none => none
  | some p =>
  let dev0 : Array Nat := #[]
  match runProg (oracle dev0 (fun _ => none) ((candidates c.lanes).head?)) p with
  | .done out _ => some out
  | .gaveUp => none
  | .ub =>
    if !o.lanewise then none else
    some <| (List.range c.lanes).toArray.map fun i =>
      match o.prog c.dts (1 :: c.dims.tail) (c.ins.map fun bs => #[bs.getD i 0]) with
      | some p1 => match runProg (oracle #[] (fun _ => none) none) p1 with
        | .done out _ => out.getD 0 0
        | _ => 0
      | none => 0

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
  let mut results : Array Json := #[]
  let mut muts : Array Json := #[]
  for c in cases do
    let some o := ops.lookup c.op | results := results.push (Json.mkObj [("id", c.id), ("status", "unknown_op")]); continue
    let some dev := device o c | results := results.push (Json.mkObj [("id", c.id), ("op", c.op), ("status", "no_device")]); continue
    let r := runCase o c dev
    IO.println s!"{c.id}: {(r.getObjValAs? String "status").toOption.getD "?"}"
    results := results.push r
    for (name, why, m) in mutants do
      if name == c.op then
        let s := (runCase m c dev).getObjValAs? String "status" |>.toOption.getD "?"
        if s != "bad_case" then
          muts := muts.push (Json.mkObj [("id", c.id), ("op", name), ("mutant", why),
            ("caught", s == "mismatch" || s == "no_tree"), ("status", s)])
  IO.FS.writeFile outF (Json.mkObj [("cases", Json.arr results), ("mutants", Json.arr muts)]).pretty
  return 0
