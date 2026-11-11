import Claim
import TVA.Proof

/-!
# Footprints and definitions for the `rotary_varlen_bf16` proof

Program `i` of the launch is `(ax0, ax1, ax2)` (`i` splits fastest along the grid's axis 0).
Its metadata (`chunkI`, `seqI`, `bosI`, `lenI`, `offI`) comes from `init`, and its lanes are the
`l < BT` of its chunk; a lane is on when its token and position are in range (`laneOn`).

`R` is every input byte the launch can read, `W` the rotated channels of the program's head of
its on lanes, and `Q` what the program leaves there.
-/

namespace Claims.RotaryVarlen

open TVA

/-- A natural number below `2^(w-1)` (stated in `Int`) is the value of its signed constant. -/
theorem ofNat_toInt {w n : Nat} (h : (n : Int) < 2 ^ (w - 1)) :
    (IntV.ofNat n : IntV true w).toInt = n :=
  IntV.toInt_ofNat (by exact_mod_cast h)

/-! ## The launch -/

/-- The number of programs is the product of the three grid axes. -/
theorem launch_size (d : Inst) : (launch d).size = g0 d * d.B * d.H := by
  simp [Launch.size, launch]

/-- Program `i`'s grid axis 0. -/
def ax0 (d : Inst) (i : Nat) : Nat := i % g0 d

/-- Program `i`'s grid axis 1. -/
def ax1 (d : Inst) (i : Nat) : Nat := i / g0 d % d.B

/-- Program `i`'s grid axis 2. -/
def ax2 (d : Inst) (i : Nat) : Nat := i / (g0 d * d.B)

/-- Program `i`'s `tl.program_id(0)` is `ax0`. -/
theorem ctx_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = ax0 d i := by
  simp [Launch.ctx, launch, ax0]

/-- Program `i`'s `tl.program_id(1)` is `ax1`. -/
theorem ctx_pid1 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 1 = ax1 d i := by
  simp [Launch.ctx, launch, ax1]

/-- Program `i`'s `tl.program_id(2)` is `ax2`. -/
theorem ctx_pid2 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 2 = ax2 d i := by
  simp [Launch.ctx, launch, ax2]

/-! ## The program's metadata -/

/-- The sequence (or, padded, the batch entry) program `i` works on. -/
def seqI (d : Inst) (init : Mem) (i : Nat) : Nat :=
  if d.isVarlen then chSeq d init (ax0 d i) else ax1 d i

/-- The chunk of `BT` tokens program `i` works on. -/
def chunkI (d : Inst) (init : Mem) (i : Nat) : Nat :=
  if d.isVarlen then chChunk d init (ax0 d i) else ax0 d i

/-- In a packed launch, program `i`'s sequence is the one `chunk_indices` names for its chunk. -/
theorem seqI_true {d : Inst} (h : d.isVarlen = true) (init : Mem) (i : Nat) :
    seqI d init i = chSeq d init (ax0 d i) := by simp [seqI, h]

/-- In a padded launch, program `i`'s sequence is its batch index `ax1`. -/
theorem seqI_false {d : Inst} (h : d.isVarlen = false) (init : Mem) (i : Nat) :
    seqI d init i = ax1 d i := by simp [seqI, h]

/-- In a packed launch, program `i`'s chunk is the one `chunk_indices` names. -/
theorem chunkI_true {d : Inst} (h : d.isVarlen = true) (init : Mem) (i : Nat) :
    chunkI d init i = chChunk d init (ax0 d i) := by simp [chunkI, h]

/-- In a padded launch, program `i`'s chunk is its grid axis 0. -/
theorem chunkI_false {d : Inst} (h : d.isVarlen = false) (init : Mem) (i : Nat) :
    chunkI d init i = ax0 d i := by simp [chunkI, h]

/-- The program's sequence's first flat token (`cu_seqlens[n]`, padded `n · T`). -/
def bosI (d : Inst) (init : Mem) (i : Nat) : Nat := bosAt d init (seqI d init i)

/-- The program's sequence's length. -/
def lenI (d : Inst) (init : Mem) (i : Nat) : Nat := lenAt d init (seqI d init i)

/-- The program's sequence's offset. -/
def offI (d : Inst) (init : Mem) (i : Nat) : Int := offAt d init (seqI d init i)

/-- Lane `l`'s token index inside the program's sequence. -/
def locI (d : Inst) (init : Mem) (i : Nat) (l : Nat) : Nat := chunkI d init i * d.BT + l

/-- Lane `l`'s flat token. -/
def tokI (d : Inst) (init : Mem) (i : Nat) (l : Nat) : Nat := bosI d init i + locI d init i l

/-- Lane `l`'s table position. -/
def posI (d : Inst) (init : Mem) (i : Nat) (l : Nat) : Int := (locI d init i l : Int) + offI d init i

/-- Lane `l` is enabled: its token is in its sequence, its position is not negative and is in the
tables. -/
def laneOn (d : Inst) (init : Mem) (i : Nat) (l : Nat) : Prop :=
  (locI d init i l : Int) < (lenI d init i : Int) ∧
    0 ≤ posI d init i l ∧ posI d init i l < (d.TR : Int)

/-! ## Footprints -/

/-- What a program may read: every byte of the launch's inputs. -/
def R (d : Inst) (_init : Mem) (_i : Nat) : Set Int :=
  d.x.bytes ∪ d.cos.bytes ∪ d.sin.bytes ∪ d.cu.bytes ∪ d.ci.bytes ∪ d.offs.bytes

/-- What a program may write: the rotated channels `c < 2 R` of its head of its on lanes. -/
def W (d : Inst) (init : Mem) (i : Nat) : Set Int :=
  d.y.elems {k | ∃ l c, l < d.BT ∧ c < 2 * d.R ∧ laneOn d init i l ∧
    k = tokIdx d (tokI d init i l) (ax2 d i) c}

/-- What a program leaves: each rotated channel of its head of its on lanes is `CellOk`
at the lane's position. -/
def Q (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  ∀ l c, l < d.BT → c < 2 * d.R → laneOn d init i l →
    CellOk d init m (tokI d init i l) (ax2 d i) c (posI d init i l).toNat

end Claims.RotaryVarlen
