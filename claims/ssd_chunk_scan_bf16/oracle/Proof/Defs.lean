import Claim
import TVA.Proof

/-!
# Definitions shared by the proof

The program decode (`pmOf`, `pnOf`, `kOf`, `hOf`), the logical chunks (`lc`, `lo`, `lend`: the
physical chunk, first row and end row of the `k`-th logical chunk, as the kernel computes them),
a program's footprints `R`, `W` and post-condition `Q`, and the error terms of the derivation in
`README.md` (Tolerances) that `Claim.lean` leaves unnamed (`errSig`, `ewS`, `errW`, `errMA`,
`errPre`, `fine`).
-/

namespace Claims.SsdChunkScan

open TVA

/-! ## The program index -/

/-- `num_pid_n = cdiv(headdim, 64)`, line 183. -/
def npn (d : Inst) : Nat := (d.P + BN - 1) / BN

/-- `cdiv(chunk_size, 64)`, the number of `64`-row tiles of a chunk. -/
def nmt (d : Inst) : Nat := (d.cs + BM - 1) / BM

/-- The grid's `x` is the number of row tiles times the number of column tiles. -/
theorem gx_eq (d : Inst) : gx d = nmt d * npn d := rfl

/-- `pid_m` of program `i` (grid index `i = pid0 + gx·(pid1 + nmeta·pid2)`). -/
def pmOf (d : Inst) (i : Nat) : Nat := i % gx d / npn d
/-- `pid_n` of program `i`. -/
def pnOf (d : Inst) (i : Nat) : Nat := i % gx d % npn d
/-- The logical chunk (`program_id(1)`) of program `i`. -/
def kOf (d : Inst) (i : Nat) : Nat := i / gx d % d.nmeta
/-- The head (`program_id(2)`) of program `i`. -/
def hOf (d : Inst) (i : Nat) : Nat := i / (gx d * d.nmeta)

/-! ## The logical chunks of the launch -/

/-- The physical chunk of logical chunk `k` (`c_idx`): the host's `chunk_indices[k]` with initial
states, `k` without. -/
def lc (d : Inst) (k : Nat) : Nat := if d.hasInit then hostIdx d k else k

/-- Its first row in the physical chunk (`c_off`): `chunk_offsets[k]`, or `0`. -/
def lo (d : Inst) (k : Nat) : Nat := if d.hasInit then hostOff d k else 0

/-- The rows of physical chunk `lc k` that exist: `min(chunk_size, seqlen - c_idx·chunk_size)`
(line 220). -/
def lm (d : Inst) (k : Nat) : Nat := min d.cs (d.T - lc d k * d.cs)

/-- Its end row (`chunk_size_limit`, lines 220, 287–297): `lm`, cut at the next logical chunk's
first row when that one lies in the same physical chunk. -/
def lend (d : Inst) (k : Nat) : Nat :=
  if d.hasInit = true ∧ k + 1 < d.nmeta ∧ hostIdx d (k + 1) = hostIdx d k
  then min (hostOff d (k + 1)) (lm d k) else lm d k

/-- The request of the token before physical chunk `c` (`seq_idx_prev`, lines 227–229): `0` in
chunk `0`. -/
def prevReq (d : Inst) (c : Nat) : Nat := if 1 ≤ c then reqOf d (c * d.cs - 1) else 0

/-- The buffer the kernel reads the entering state from: the initial states when the logical chunk
starts its request (`u`), else the chunk states. -/
def stBuf (d : Inst) (u : Bool) : Buf .f32 := if u then d.initst else d.states

/-- The element of that buffer holding state `(h, 0, 0)` of request `r` (initial states) or of
physical chunk `c` (chunk states). -/
def stBase (d : Inst) (u : Bool) (c r h : Nat) : Nat :=
  if u then r * (d.H * PN d) + h * PN d else c * (d.H * PN d) + h * PN d

/-! ## A program's tile -/

/-- Row `ii` of the tile of program `(k, pm)`, as a row of the physical chunk. -/
def rowOf (d : Inst) (k pm ii : Nat) : Nat := lo d k + BM * pm + ii
/-- The token of row `ii` of the tile of program `(k, pm)`. -/
def tokOf (d : Inst) (k pm ii : Nat) : Nat := lc d k * d.cs + rowOf d k pm ii
/-- Row `ii` is inside the logical chunk (the store's mask, line 487). -/
def RowOk (d : Inst) (k pm ii : Nat) : Prop := rowOf d k pm ii < lend d k
/-- The channel of column `jj` of the tile of program `pn`. -/
def colOf (pn jj : Nat) : Nat := BN * pn + jj
/-- Column `jj` is inside `headdim`. -/
def ColOk (d : Inst) (pn jj : Nat) : Prop := colOf pn jj < d.P

/-! ## Footprints and post-condition -/

/-- Every program may read every byte but `out`'s. -/
def R (d : Inst) : Set Int := d.out.bytesᶜ

/-- The bytes of the tile `(k, pm, pn)` of `out`, head `h`. -/
def Wt (d : Inst) (k pm pn h : Nat) : Set Int :=
  d.out.tileAt (lc d k * d.cs * (d.H * d.P) + h * d.P) (d.H * d.P) 1
    (fun ii : Fin BM => rowOf d k pm ii) (fun jj : Fin BN => colOf pn jj) (lend d k) d.P

/-- **What a tile leaves**: each element of it holds a `bfloat16` value, which, when the inputs'
values are covered, is finite and within `tol` of `yRef`. -/
def Qt (d : Inst) (init : Mem) (k pm pn h : Nat) (m : Mem) : Prop :=
  ∀ ii jj : Nat, ii < BM → jj < BN → RowOk d k pm ii → ColOk d pn jj →
    ∃ v : BF16, rdB m d.out (outIdx d (tokOf d k pm ii) h (colOf pn jj)) = some v ∧
      (Vals init d → v.isFinite = true ∧
        |v.toReal - yRef init d (tokOf d k pm ii) h (colOf pn jj)| ≤
          tol init d (tokOf d k pm ii) h (colOf pn jj))

/-- The bytes of program `i`'s tile of `out`. -/
def W (d : Inst) (i : Nat) : Set Int := Wt d (kOf d i) (pmOf d i) (pnOf d i) (hOf d i)

/-- What program `i` leaves. -/
def Q (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  Qt d init (kOf d i) (pmOf d i) (pnOf d i) (hOf d i) m

/-! ## The derivation's error terms (README, Tolerances) -/

/-- The error of the state term `acc₀` (README step 2): the dot's error `eSig` scaled by the
computed decay, the product's rounding, and the decay's own error. -/
noncomputable def errSig (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  (1 + uF) * eSig m₀ d t h p * hSt m₀ d t h + uF * mSig m₀ d t h p * hSt m₀ d t h +
    mSig m₀ d t h p * eSt m₀ d t h + etaF

/-- The error of the computed weight `w̃` of token `s` against `wTS` (README step 3). -/
noncomputable def ewS (m₀ : Mem) (d : Inst) (t h s : Nat) : ℝ :=
  |cbTS m₀ d t h s| * |dtS m₀ d t h s| * errE (dS m₀ d t h s) (dErrS m₀ d t h s) +
    (uB + 3 * uF) * wMag m₀ d t h s + wAbs m₀ d t h s

/-- `Σ_s ewS·|x_s|`, the weights' error in the output. -/
noncomputable def errW (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  ∑ s ∈ toks d t, ewS m₀ d t h s * |xv m₀ d s h p|

/-- The error of line 442's `tl.mul_add` (README step 5). -/
noncomputable def errMA (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  (2 ^ (-23 : ℤ) + 2 ^ (-47 : ℤ)) * (|xv m₀ d t h p * Dv m₀ d h| + aAcc m₀ d t h p) +
    3 * 2 ^ (-150 : ℤ)

/-- The `float32` value before the store is within `errPre` of `yRef` (README step 6). -/
noncomputable def errPre (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  errSig m₀ d t h p + errW m₀ d t h p + errDot m₀ d t h p + errMA m₀ d t h p

/-- The output is within `fine` of `yRef` (README step 6), and `fine ≤ tol` (step 8). -/
noncomputable def fine (m₀ : Mem) (d : Inst) (t h p : Nat) : ℝ :=
  (1 + uB) * errPre m₀ d t h p + uB * |yRef m₀ d t h p| + 2 ^ (-134 : ℤ)

end Claims.SsdChunkScan
