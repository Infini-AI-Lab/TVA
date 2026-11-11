import Proof.Pid

/-!
# The tile footprint and post-state of one program

Program `p` of the launch handles the block `(pid_m p, pid_n p)` of `128 × 256`
elements.  This file names the global row/column of a block element, the set of
matrix indices the program stores, the read footprint (both input matrices) and
the per-program post-state predicate `Q`.
-/

namespace Claims.MatmulE5m2

open TVA

/-- Global row of block row `x` of program `p` (not reduced modulo `M`). -/
def ocm (d : Inst) (p x : Nat) : Nat := pidm d p * 128 + x

/-- Global column of block column `y` of program `p` (not reduced modulo `N`). -/
def ocn (d : Inst) (p y : Nat) : Nat := pidn d p * 256 + y

/-- The matrix indices program `p` stores: its block, restricted to the matrix. -/
def cBlk (d : Inst) (p : Nat) : Set Nat :=
  {e | ∃ x y, x < 128 ∧ y < 256 ∧ ocm d p x < d.M ∧ ocn d p y < d.N ∧
        e = ocm d p x * d.stride_cm + ocn d p y * d.stride_cn}

/-- Everything program `p` reads: the whole `a` and `b` matrices. -/
def R (d : Inst) (_init : Mem) (_p : Nat) : Set Int :=
  d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)

/-- Everything program `p` writes: its block of `c`. -/
def W (d : Inst) (_init : Mem) (p : Nat) : Set Int := d.c.elems (cBlk d p)

/-- Program `p`'s post-state: every in-matrix element of its block holds a
correct result. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ x y, x < 128 → y < 256 → ocm d p x < d.M → ocn d p y < d.N →
    OutSpec d init m (ocm d p x) (ocn d p y)

end Claims.MatmulE5m2
