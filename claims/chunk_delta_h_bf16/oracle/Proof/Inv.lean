import Proof.Defs

/-!
# `chunk_delta_h_bf16`: the loop invariant

After `t` trips of the main loop, the four state blocks `st` hold the states `YY jj t` (lane `jj`
of the block of value columns), and for each valid column `ColPre` holds of the memory: the first
`t` chunks' states are stored and the first `t` chunks' new values and steps are as `Post`
says. `FinInv` carries what the next trip needs to apply the device facts: the state entering the
trip is finite, and its padded rows are within `padB`.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The number of blocks of `64` state rows that exist: `cdiv(K, 64)`. -/
def nbOf (d : Inst) : Nat := (d.K + 63) / 64

/-- State block `b` (`b < 4`) of a state in the `[K, V]` layout. -/
def blk (st : St BV) : Nat → Tensor [64, BV] F32
  | 0 => st.1
  | 1 => st.2.1
  | 2 => st.2.2.1
  | _ => st.2.2.2

/-- State block `b` of a state in the `[V, K]` layout. -/
def blkV (st : StV BV) : Nat → Tensor [BV, 64] F32
  | 0 => st.1
  | 1 => st.2.1
  | 2 => st.2.2.1
  | _ => st.2.2.2

/-- The state blocks hold the states `YY jj t`: row `64·b + a` of lane `jj` is element `(a, jj)`
of block `b`. -/
def StRep (d : Inst) (st : St BV) (YY : Nat → Nat → Nat → F32) (t : Nat) : Prop :=
  ∀ b, b < nbOf d → ∀ (a : Fin 64) (jj : Fin BV), (blk st b).get2 a jj = YY jj t (64 * b + a)

/-- As `StRep`, in the `[V, K]` layout: element `(jj, a)` of block `b`. -/
def StRepV (d : Inst) (st : StV BV) (YY : Nat → Nat → Nat → F32) (t : Nat) : Prop :=
  ∀ b, b < nbOf d → ∀ (a : Fin 64) (jj : Fin BV), (blkV st b).get2 jj a = YY jj t (64 * b + a)

/-- The state entering chunk `t` is finite, and its padded rows are within `padB t`, when the
inputs are in range and the stored states of the chunks before `t` are. -/
def FinInv (d : Inst) (init m : Mem) (n hv j t : Nat) (Y : Nat → Nat → F32) : Prop :=
  Vals init d → (∀ t', t' < t → StOk m d n hv j t') →
    (∀ c, c < d.K → (Y t c).isFinite = true) ∧
    (∀ c, d.K ≤ c → c < 64 * nbOf d →
      (Y t c).isFinite = true ∧ |(Y t c).toReal| ≤ padB init d n hv t)

/-- The invariant of every lane (valid column) of the block. -/
def LaneInv (d : Inst) (init m : Mem) (n hv jv t : Nat) (YY : Nat → Nat → Nat → F32) : Prop :=
  ∀ jj, jj < BV → inBlk d jv (BV * jv + jj) →
    ColPre d init m n hv (BV * jv + jj) t (YY jj) ∧ FinInv d init m n hv (BV * jv + jj) t (YY jj)

/-- **The loop invariant** after `t` trips (`[K, V]` layout). -/
def Inv (d : Inst) (init : Mem) (n hv jv t : Nat) (st : St BV) (m : Mem) : Prop :=
  MemOk d init m ∧ ∃ YY, StRep d st YY t ∧ LaneInv d init m n hv jv t YY

/-- **The loop invariant** after `t` trips (`[V, K]` layout). -/
def InvV (d : Inst) (init : Mem) (n hv jv t : Nat) (st : StV BV) (m : Mem) : Prop :=
  MemOk d init m ∧ ∃ YY, StRepV d st YY t ∧ LaneInv d init m n hv jv t YY

end Claims.ChunkDeltaH
