import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `rotary_varlen_bf16` run by `execLaunch` on three small covered
instances:
* `neox_padded_offt_i32`: GPT-NeoX, a padded batch, a tensor of `int32` offsets with a negative
  one (left padding: the token at position `-1` is skipped), partial rotary (`2R = 4` of `D = 8`),
  a ragged last chunk (`T = 3`, `BT = 2`), and `+∞` in the channels from `2R` on, which the kernel
  must not read;
* `gptj_packed_i64`: GPT-J (interleaved), two packed sequences (`int64` `cu_seqlens`,
  `chunk_indices` in a non-sorted order), one `int` offset `1`;
* `neox_packed_i32_offt_i64`: GPT-NeoX, three packed sequences (`int32` `cu_seqlens`), one of them
  empty, a tensor of `int64` offsets with a negative one (two skipped tokens), `R = 1`.

Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and that
the final memory is what `Post` says, all of it: every rotated channel `c < 2R` of every head of
every token at a position `p ≥ 0` holds a finite `bfloat16` within `Post`'s tolerance (`tol`, the
function `Post` names, used here) of the exact rotation `a·cs ∓ b·sn`, computed here from the
inputs with the test's own pairing (`c ± R`, or `c ± 1` by parity); and every other byte is
unchanged: the skipped tokens' channels, `y`'s channels from `2R` on, the inputs, sentinels after
`y`, and no byte added. On these inputs (`x` integers in `[-8, 8]`, table entries multiples of `1/4`
in `[-1, 1]`) every product, sum and `bfloat16` rounding is exact, along every fused or unfused
order, so the outputs are the exact rotations.

`rotary_varlen_bf16` uses no platform operation: the verdicts are `certified`, with the empty
oracle. -/

open TVA TVA.Validation Claims.RotaryVarlen

/-- A platform that relates nothing: `rotary_varlen_bf16` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits (every masked load here has `other`). -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The instances -/

/-- GPT-NeoX, padded: `B = 2`, `T = 3`, `H = 2`, `D = 8`, `R = 2`, `TR = 5`, `BT = 2`, `BD = 4`;
offsets `[-1, 2]` (`int32`). -/
def neoxPadded : Inst where
  idx := .i32
  B := 2
  T := 3
  H := 2
  D := 8
  R := 2
  TR := 5
  BT := 2
  BD := 4
  N := 2
  NT := 2
  off := 0
  isVarlen := false
  interleaved := false
  x := ⟨1099511627776, 96⟩
  cos := ⟨2199023255552, 10⟩
  sin := ⟨3298534883328, 10⟩
  y := ⟨4398046511104, 96⟩
  cu := ⟨5497558138880, 0⟩
  ci := ⟨6597069766656, 0⟩
  offT := true
  offD := .i32
  offs := ⟨7696581394432, 2⟩

/-- GPT-J, packed: sequences of `3` and `2` tokens (`T = 5`), `H = 1`, `D = 4`, `R = 2`, `TR = 6`,
`BT = 2`, `BD = 4`, three chunks; `int64` indices; offset `1`. -/
def gptjPacked : Inst where
  idx := .i64
  B := 1
  T := 5
  H := 1
  D := 4
  R := 2
  TR := 6
  BT := 2
  BD := 4
  N := 2
  NT := 3
  off := 1
  isVarlen := true
  interleaved := true
  x := ⟨1099511627776, 20⟩
  cos := ⟨2199023255552, 12⟩
  sin := ⟨3298534883328, 12⟩
  y := ⟨4398046511104, 20⟩
  cu := ⟨5497558138880, 3⟩
  ci := ⟨6597069766656, 6⟩
  offT := false
  offD := .i32
  offs := ⟨7696581394432, 0⟩

/-- GPT-NeoX, packed: sequences of `2`, `0` and `3` tokens (`T = 5`), `H = 2`, `D = 4`, `R = 1`,
`TR = 7`, `BT = 2`, `BD = 2`, three chunks; `int32` indices; offsets `[0, 7, -2]` (`int64`). -/
def neoxPacked : Inst where
  idx := .i32
  B := 1
  T := 5
  H := 2
  D := 4
  R := 1
  TR := 7
  BT := 2
  BD := 2
  N := 3
  NT := 3
  off := 0
  isVarlen := true
  interleaved := false
  x := ⟨1099511627776, 40⟩
  cos := ⟨2199023255552, 7⟩
  sin := ⟨3298534883328, 7⟩
  y := ⟨4398046511104, 40⟩
  cu := ⟨5497558138880, 4⟩
  ci := ⟨6597069766656, 6⟩
  offT := true
  offD := .i64
  offs := ⟨7696581394432, 3⟩

theorem neoxPadded_ok : neoxPadded.ok :=
  ⟨Or.inl rfl, ⟨1, rfl⟩, ⟨2, rfl⟩,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    bytes_disj (by decide), bytes_disj (by decide), bytes_disj (by decide),
    bytes_disj (by decide), bytes_disj (by decide),
    by decide, fun _ => Or.inl rfl, by decide, bytes_disj (by decide), by decide, by decide⟩

theorem gptjPacked_ok : gptjPacked.ok :=
  ⟨Or.inr rfl, ⟨1, rfl⟩, ⟨2, rfl⟩,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    bytes_disj (by decide), bytes_disj (by decide), bytes_disj (by decide),
    bytes_disj (by decide), bytes_disj (by decide),
    by decide, fun h => absurd h (by decide), by decide, bytes_disj (by decide), by decide, by decide⟩

theorem neoxPacked_ok : neoxPacked.ok :=
  ⟨Or.inl rfl, ⟨1, rfl⟩, ⟨1, rfl⟩,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    bytes_disj (by decide), bytes_disj (by decide), bytes_disj (by decide),
    bytes_disj (by decide), bytes_disj (by decide),
    by decide, fun _ => Or.inr rfl, by decide, bytes_disj (by decide), by decide, by decide⟩

/-! ## The cases -/

/-- One case: the instance, the packed metadata (`cu_seqlens`, `chunk_indices` as `(sequence,
chunk)` pairs), the tensor of offsets, and the inputs. -/
structure Case where
  d : Inst
  cuv : List Int
  civ : List (Nat × Nat)
  offv : List Int
  /-- `x`'s element `k`, as a `bfloat16` (`+∞` in the channels from `2R` on, never read). -/
  xbits : Nat → BF16
  /-- `cos[r, k]` and `sin[r, k]`. -/
  cosv : Nat → Nat → Rat
  sinv : Nat → Nat → Rat

def bf (q : Rat) : BF16 := FloatV.ofRat q

/-- The activations: integers in `[-8, 8]` in the rotated channels, `+∞` in the others. -/
def xOf (D R : Nat) (k : Nat) : BF16 :=
  if k % D < 2 * R then bf ((((k * 7 + 3) % 17 : Nat) : Int) - 8) else FloatV.posInf

def cosT (r k : Nat) : Rat := [1, 1 / 2, -1 / 2, 0, 3 / 4, -1, 1 / 4, -3 / 4].getD ((r * 3 + k) % 8) 0
def sinT (r k : Nat) : Rat := [0, 1 / 2, 3 / 4, -1, 1 / 4, 1, -1 / 4, 1 / 2].getD ((r * 5 + k * 3 + 1) % 8) 0

def neoxPaddedC : Case :=
  { d := neoxPadded, cuv := [], civ := [], offv := [-1, 2], xbits := xOf 8 2, cosv := cosT, sinv := sinT }
def gptjPackedC : Case :=
  { d := gptjPacked, cuv := [0, 3, 5], civ := [(1, 0), (0, 1), (0, 0)], offv := [],
    xbits := xOf 4 2, cosv := cosT, sinv := sinT }
def neoxPackedC : Case :=
  { d := neoxPacked, cuv := [0, 2, 2, 5], civ := [(2, 1), (0, 0), (2, 0)], offv := [0, 7, -2],
    xbits := xOf 4 1, cosv := cosT, sinv := sinT }

/-- An integer element of an index dtype (`int32` or `int64`). -/
def ival (dt : DType) (z : Int) : dt.Val := dt.ofBits (z % (2 ^ (8 * dt.bytes) : Int)).toNat

def ints {dt : DType} (b : Buf dt) (zs : List Int) : List (Int × dt.Val) :=
  (List.range zs.length).map fun j : Nat => ((b.ptr j).addr, ival dt (zs.getD j 0))

/-- A sentinel in every element of `y` and `pad` elements after it. -/
def sentinel : Rat := -99
def pad : Nat := 16

def Case.init (c : Case) : CMem :=
  let d := c.d
  let m := CMem.ofElems ∅ .bf16 ((List.range d.x.len).map fun k : Nat => ((d.x.ptr k).addr, c.xbits k))
  let m := CMem.ofElems m .bf16 ((List.range (d.TR * d.R)).map fun k : Nat =>
    ((d.cos.ptr k).addr, bf (c.cosv (k / d.R) (k % d.R))))
  let m := CMem.ofElems m .bf16 ((List.range (d.TR * d.R)).map fun k : Nat =>
    ((d.sin.ptr k).addr, bf (c.sinv (k / d.R) (k % d.R))))
  let m := CMem.ofElems m .bf16 ((List.range (d.y.len + pad)).map fun k : Nat =>
    ((d.y.ptr k).addr, bf sentinel))
  let m := CMem.ofElems m d.idx (ints d.cu c.cuv)
  let m := CMem.ofElems m d.idx (ints d.ci (c.civ.flatMap fun (n, j) => [(n : Int), j]))
  CMem.ofElems m d.offD (ints d.offs c.offv)

def Case.run (c : Case) : LRes (launch c.d).size × List Record :=
  execLaunch ((launch c.d).env P0) Oracle.empty fill0 1000 (model c.d) (launch c.d) c.init

/-- The sequences: `(first flat token, length, offset)`, from the case's own lists. -/
def Case.seqs (c : Case) : List (Nat × Nat × Int) :=
  if c.d.isVarlen then
    (List.range c.d.N).map fun n =>
      ((c.cuv.getD n 0).toNat, (c.cuv.getD (n + 1) 0 - c.cuv.getD n 0).toNat,
        if c.d.offT then c.offv.getD n 0 else c.d.off)
  else
    (List.range c.d.B).map fun n => (n * c.d.T, c.d.T, if c.d.offT then c.offv.getD n 0 else c.d.off)

/-- The rotated cells `(t, h, ch, p)` (flat token, head, channel `< 2R`, position), every one. -/
def Case.cells (c : Case) : List (Nat × Nat × Nat × Int) :=
  c.seqs.flatMap fun (bos, len, off) => (List.range len).flatMap fun l =>
    (List.range c.d.H).flatMap fun h => (List.range (2 * c.d.R)).map fun ch =>
      (bos + l, h, ch, (l : Int) + off)

/-- The exact rotation of cell `(t, h, ch)` at position `p` (`conj` turns the other way: a wrong
expectation). GPT-NeoX pairs `ch` with `ch ± R`, GPT-J with `ch ± 1`; the first of a pair is
`a·cs − b·sn`, the second `a·cs + b·sn`. Also the tolerance `Post` gives it (`tol`). -/
def Case.expect (c : Case) (conj : Bool) (t h ch : Nat) (p : Nat) : Rat × Rat :=
  let d := c.d
  let (first, other, k) :=
    if d.interleaved then (ch % 2 == 0, if ch % 2 == 0 then ch + 1 else ch - 1, ch / 2)
    else if ch < d.R then (true, ch + d.R, ch) else (false, ch - d.R, ch - d.R)
  let a := (c.xbits ((t * d.H + h) * d.D + ch)).toRat
  let b := (c.xbits ((t * d.H + h) * d.D + other)).toRat
  let cs := c.cosv p k
  let sn := c.sinv p k
  let ref := if first != conj then a * cs - b * sn else a * cs + b * sn
  (ref, tol first a b cs sn)

/-- The final memory against `Post`: every cell at `p ≥ 0` finite within `tol` of the rotation;
every byte outside those cells unchanged (the skipped cells, `y`'s other channels, the inputs, the
sentinels after `y`); no byte added. -/
def Case.agreeC (c : Case) (conj : Bool) (cf : CConfig (launch c.d).size) : Option String :=
  let d := c.d
  let m := CMem.toMem cf.mem
  let live := c.cells.filter fun (_, _, _, p) => 0 ≤ p
  let bad := live.filter fun (t, h, ch, p) =>
    let (ref, tl) := c.expect conj t h ch p.toNat
    match m.read .bf16 (d.y.ptr ((t * d.H + h) * d.D + ch : Nat)).addr with
    | some v => !(v.isFinite && |v.toRat - ref| ≤ tl)
    | none => true
  let cellBytes : Std.HashSet Int := live.foldl (fun s (t, h, ch, _) =>
    let a := (d.y.ptr ((t * d.H + h) * d.D + ch : Nat)).addr
    (s.insert a).insert (a + 1)) {}
  let changed := c.init.toList.filter fun (a, b) => !cellBytes.contains a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"cells {(bad.take 8).map fun (t, h, ch, _) => (t, h, ch)} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the rotated cells changed"
  else if cf.mem.size != c.init.size then some "the launch wrote outside the initial memory"
  else none

def Case.agree (c : Case) := c.agreeC false

def cases : List (String × Case) :=
  [("neox_padded_offt_i32", neoxPaddedC), ("gptj_packed_i64", gptjPackedC),
   ("neox_packed_i32_offt_i64", neoxPackedC)]

/-! ## Covered: `Pre`'s decidable parts on each case's initial memory -/

def validB {dt : DType} (m : Mem) (b : Buf dt) : Bool :=
  0 ≤ b.base && b.base + b.len * dt.bytes ≤ 2 ^ 64 && b.base % dt.bytes == 0 &&
    (List.range b.len).all fun k : Nat => (m.read dt (b.ptr k).addr).isSome

#guard cases.all fun (_, c) =>
  let d := c.d
  let m := CMem.toMem c.init
  let xok (v : BF16) := v.isFinite && |v.toRat| ≤ 2 ^ 125
  let tabok (v : BF16) := v.isFinite && |v.toRat| ≤ 1
  -- `xv`, `yv`, `cosv`, `sinv`
  validB m d.x && validB m d.y && validB m d.cos && validB m d.sin &&
  -- `x_vals`: the rotated channels finite, `|·| ≤ 2^125`
  ((List.range (nTok d * d.H * d.D)).all fun j : Nat =>
    j % d.D ≥ 2 * d.R || ((bfAt d.x m j).map xok == some true)) &&
  -- `cos_vals`, `sin_vals`: every row read (`p ≥ 0`), from the claim's own `offAt`, `lenAt`
  ((List.range (nSeq d)).all fun n => (List.range (lenAt d m n)).all fun l =>
    let p : Int := (l : Int) + offAt d m n
    p < 0 || (List.range d.R).all fun k =>
      (bfAt d.cos m (p.toNat * d.R + k)).map tabok == some true &&
      (bfAt d.sin m (p.toNat * d.R + k)).map tabok == some true) &&
  -- `seqs`: `SeqOk` when packed
  (!d.isVarlen || (validB m d.cu && validB m d.ci &&
    ((List.range (d.N + 1)).all fun n => (idxAt d.cu m n).any (0 ≤ ·)) &&
    cuN d m 0 == 0 && cuN d m d.N == d.T &&
    ((List.range d.N).all fun n => cuN d m n ≤ cuN d m (n + 1)) &&
    ((List.range (2 * d.NT)).all fun i => (idxAt d.ci m i).any (0 ≤ ·)) &&
    ((List.range d.NT).all fun i => chSeq d m i < d.N && chChunk d m i * d.BT < seqLen d m (chSeq d m i)) &&
    ((List.range d.N).all fun n => (List.range (d.T + 1)).all fun j =>
      j * d.BT ≥ seqLen d m n || (List.range d.NT).any fun i => chSeq d m i == n && chChunk d m i == j) &&
    ((List.range d.NT).all fun i => (List.range d.NT).all fun i' =>
      !(chSeq d m i == chSeq d m i' && chChunk d m i == chChunk d m i') || i == i'))) &&
  -- `offs_vals`, `offs_ctx`
  (!d.offT || (validB m d.offs && (List.range (nSeq d)).all fun n =>
    match idxAt d.offs m n with
    | some v => -2 ^ 31 ≤ v && (lenAt d m n : Int) + v ≤ d.TR && v ≤ 2 ^ 22
    | none => false))

-- The case's own lists agree with the memory the claim reads (`cuN`, `offAt`): the test's
-- sequences are the claim's.
#guard cases.all fun (_, c) =>
  let m := CMem.toMem c.init
  c.seqs == (List.range (nSeq c.d)).map fun n => (bosAt c.d m n, lenAt c.d m n, offAt c.d m n)

-- What the cases cover: skipped tokens (`p < 0`) in two cases, an empty sequence, `+∞` in
-- unread channels, and every expectation exact in `bfloat16` (an unwritten cell, holding the
-- sentinel, would be caught).
#guard (neoxPaddedC.cells.any fun (_, _, _, p) => p < 0) && (neoxPackedC.cells.any fun (_, _, _, p) => p < 0)
#guard neoxPackedC.seqs.any fun (_, len, _) => len == 0
#guard (xOf 8 2 4).isInf
#guard cases.all fun (_, c) => c.cells.all fun (t, h, ch, p) =>
  p < 0 || let (ref, _) := c.expect false t h ch p.toNat; (bf ref).toRat == ref && ref != sentinel

/-- Each case's verdict, and the wrong expectation's (the rotation the other way) on the same
run. -/
def results : List (String × Verdict × Verdict) := cases.map fun (n, c) =>
  let r := c.run
  (n, Verdict.ofLaunch c.agree r, Verdict.ofLaunch (c.agreeC true) r)

#eval results.forM fun (n, v, _) => IO.println (v.line s!"rotary_varlen_bf16/{n}")
-- `certified` (no platform operation), and not vacuous: the wrong expectation is a discrepancy.
#guard results.all fun (_, v, w) => v == .certified && w.isDiscrepancy
