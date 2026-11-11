import Claim
import TVA.Proof

/-!
# `lora_shrink_bf16`: the definitions of the proof

* the program of the launch: its adapter group, slice, part of `K` and tile of the output, from
  the program number (`pidSk`, `pidM`, `pidN`, `Active`, `Tgt`);
* the shared cells (the output elements) and the footprints of the programs;
* what one program leaves in the log of its atomic additions (`Q`);
* the facts about one program's operand (`PartOk`).
-/

namespace Claims.LoraShrink

open TVA

/-! ## The program number -/

/-- The number of rank tiles, `cdiv(R, 16)` (line 347). -/
def cn (d : Inst) : Nat := (d.R + 15) / 16

/-- The number of row tiles, `cdiv(M, 32)` (line 348). -/
def cm (d : Inst) : Nat := (d.M + 31) / 32

/-- The part of `K` of axis-0 program `x` (line 351). -/
def pidSk (d : Inst) (x : Nat) : Nat := x % sk d

/-- `pid_sk_m_n // SPLIT_K` (line 353). -/
def pidMN (d : Inst) (x : Nat) : Nat := x / sk d

/-- `group_id` of the grouped ordering (line 355). -/
def grp8 (d : Inst) (x : Nat) : Nat := pidMN d x / (8 * cn d)

/-- `first_pid_m` (line 356). -/
def fpm (d : Inst) (x : Nat) : Nat := grp8 d x * 8

/-- `group_size_m` (line 357). -/
def gsm (d : Inst) (x : Nat) : Nat := min (cm d - fpm d x) 8

/-- The row tile `pid_m` of axis-0 program `x` (line 360). -/
def pidM (d : Inst) (x : Nat) : Nat := fpm d x + (pidMN d x % (8 * cn d)) % gsm d x

/-- The rank tile `pid_n` of axis-0 program `x` (line 361). -/
def pidN (d : Inst) (x : Nat) : Nat := (pidMN d x % (8 * cn d)) / gsm d x

/-- Axis 0 of program number `i`. -/
def ax0 (d : Inst) (i : Nat) : Nat := i % grid0 d

/-- The slice of program number `i` (axis 1). -/
def slice (d : Inst) (i : Nat) : Nat := i / grid0 d % d.S

/-- The adapter group of program number `i` (axis 2). -/
def grpOf (d : Inst) (i : Nat) : Nat := i / (grid0 d * d.S)

/-- The part of `K` program `i` accumulates. -/
def part (d : Inst) (i : Nat) : Nat := pidSk d (ax0 d i)

/-- The row tile of program `i`. -/
def tileM (d : Inst) (i : Nat) : Nat := pidM d (ax0 d i)

/-- The rank tile of program `i`. -/
def tileN (d : Inst) (i : Nat) : Nat := pidN d (ax0 d i)

/-- `cta_m_len` of program `i`: its rows in the group, at most 32 (line 379). -/
def mlen (d : Inst) (i : Nat) : Nat := min 32 (cntAt d (grpOf d i) - tileM d i * 32)

/-- Program `i` does work: its group has an adapter (`lora_id ≠ -1`, line 367) and its row tile
starts inside the group (line 374). -/
def Active (d : Inst) (i : Nat) : Prop :=
  idAt d (grpOf d i) ≠ -1 ∧ tileM d i * 32 < cntAt d (grpOf d i)

/-- The token of row `m` of program `i`'s tile: entry `start + pid_m·32 + m` of the sorted token
indices. -/
def tokOf (d : Inst) (i m : Nat) : Nat := d.tIdx (startAt d (grpOf d i) + tileM d i * 32 + m)

/-- Program `i` adds into the output element `(s, t, j)`: it is active, `s` is its slice, `t` the
token of one of its rows and `j` one of its 16 columns. -/
def Tgt (d : Inst) (i s t j : Nat) : Prop :=
  Active d i ∧ s = slice d i ∧ (∃ m, m < mlen d i ∧ t = tokOf d i m) ∧
    16 * tileN d i ≤ j ∧ j < 16 * tileN d i + 16

/-! ## The shared cells and the footprints -/

/-- The byte address of output element `(s, t, j)`. -/
def cellAddr (d : Inst) (s t j : Nat) : Int := (d.out.ptr (outAt d s t j)).addr

/-- The shared cells: the elements of the output. -/
def Cells (d : Inst) : Set (DType × Int) :=
  {c | ∃ s t j, s < d.S ∧ t < d.M ∧ j < d.R ∧ c = (DType.f32, cellAddr d s t j)}

/-- The bytes every program reads: all the buffers it loads from (the address table only with
several slices). -/
def Rall (d : Inst) : Set Int :=
  d.inp.bytes ∪ (⋃ s ∈ {s | s < d.S}, (d.lora s).bytes) ∪ {x | 1 < d.S ∧ x ∈ d.tab.bytes} ∪
    d.tIdxBuf.bytes ∪ d.cntBuf.bytes ∪ d.startBuf.bytes ∪ d.idBuf.bytes

/-! ## What one operand satisfies -/

/-- The kernel's `float32` `scaling`, as a rational. -/
def s32 (d : Inst) : ℚ := (scaling32 d).toRat

/-- The error of rounding one scaled part (`float32` multiplication by `scaling`): at most
`u·|s32|·(|partSum p| + errPart p) + η`. -/
def mulErr (d : Inst) (va vb : Nat → BF16) (p : Nat) : ℚ :=
  Format.binary32.u * |s32 d| * (|partSum d va vb p| + errPart d va vb p) + Format.binary32.eta

/-- The operand a program adds for part `p`: a finite `float32` within `|s32|·errPart + mulErr` of
`s32 · partSum p`, of magnitude at most `|s32|·(|partSum p| + errPart p) + mulErr`. -/
def PartOk (d : Inst) (va vb : Nat → BF16) (p : Nat) (v : F32) : Prop :=
  v.isFinite = true ∧
    |v.toRat - s32 d * partSum d va vb p| ≤ |s32 d| * errPart d va vb p + mulErr d va vb p ∧
    |v.toRat| ≤ |s32 d| * (|partSum d va vb p| + errPart d va vb p) + mulErr d va vb p

/-- The hypotheses of `OutSpec` for the output element `(s, t, j)`: token `t` has an adapter, the
`K` values of its row of `inputs` are `va` and of row `j` of its adapter's weights are `vb`, all
finite, and `dotMag` and the room hold. -/
def Dat (d : Inst) (init : Mem) (s t j : Nat) (va vb : Nat → BF16) : Prop :=
  0 ≤ d.adapter t ∧
  (∀ k, k < d.K → init.read .bf16 (d.inp.ptr (t * d.K + k)).addr = some (va k) ∧
    (va k).isFinite = true) ∧
  (∀ k, k < d.K → init.read .bf16
      ((d.lora s).ptr ((d.adapter t).toNat * (d.R * d.K) + j * d.K + k)).addr = some (vb k) ∧
    (vb k).isFinite = true) ∧
  dotMag d va vb ≤ 2 ^ 120 ∧
  vMag d va vb + atomTol d va vb ≤ Format.binary32.maxVal

/-- The relation each logged atomic step on output element `(s, t, j)` satisfies, with its
operand: a `float32` addition of the operand, within `2^-23·|x + w| + 2^-124`. -/
abbrev AccE (d : Inst) (s t j : Nat) : AtomicStep → Option ℚ → Prop :=
  AccStep .f32 (cellAddr d s t j) FloatV.toRat (fun x : F32 => x.isFinite = true)
    (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) Format.binary32.maxVal

/-- **What program `i` leaves.** Each step of its log is on an element it targets; and there are
operands `ops s t j` (the multiset of the operands it added to element `(s, t, j)`): none for an
element it does not target; for one it targets, the single value of a `float32` that is `PartOk`
for the program's part, whenever the data are `Dat`; and, whenever the data are `Dat`, its log is
paired with them by `AccE`. -/
def Q (d : Inst) (init : Mem) (i : Nat) (log : List AtomicStep) (_ : Mem) : Prop :=
  (∀ st ∈ log, ∃ s t j, Tgt d i s t j ∧ st.d = DType.f32 ∧ st.a = cellAddr d s t j) ∧
  ∃ ops : Nat → Nat → Nat → Multiset ℚ,
    (∀ s t j, s < d.S → t < d.M → j < d.R → ∀ va vb, Dat d init s t j va vb →
      Paired (AccE d s t j) log (ops s t j)) ∧
    (∀ s t j, ¬ Tgt d i s t j → ops s t j = 0) ∧
    (∀ s t j, Tgt d i s t j → ∃ v : F32, ops s t j = {v.toRat} ∧
      ∀ va vb, Dat d init s t j va vb → PartOk d va vb (part d i) v)

/-! ## The accumulator of one lane after some trips -/

/-- The lifted magnitude of the first `cpb · t` chunks of part `p`: what `t` trips of the loop
have accumulated (`partLift p` is the value at `t = stepsOf p`). -/
def pLift (d : Inst) (va vb : Nat → BF16) (p t : Nat) : ℚ :=
  ∑ q ∈ Finset.range (cpb d * t), ∑ k ∈ chunkK d p q,
    Device.liftMag .bf16 (va k).toRat * Device.liftMag .bf16 (vb k).toRat

/-- **The loop invariant of one accumulator lane** after `t` trips (`t ≤ stepsOf p`): the lane is a
finite `float32` within `accErr` (the chunk-by-chunk bound) and within `DotLoop.bound` of the
step-by-step bound, of the exact sum of the first `cpb · t` chunks of part `p`. -/
def LaneInv (d : Inst) (va vb : Nat → BF16) (p t : Nat) (c : F32) : Prop :=
  c.isFinite = true ∧
    |c.toRat - pSum d va vb p (cpb d * t)| ≤ accErr d va vb p t ∧
    |c.toRat - pSum d va vb p (cpb d * t)| ≤ DotLoop.bound (rho d) (alpha d) t (pLift d va vb p t)

end Claims.LoraShrink
