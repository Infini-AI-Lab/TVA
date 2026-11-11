import Proof.Lanes

/-!
# `chunk_delta_h_bf16`: what one loop trip does

`TripNonvf` (`TripVf`) says: from the loop invariant after `t` trips, the model's `chunk_step`
(`chunk_step_vf`) on the program's pointers and lanes is safe, writes inside chunk `t`'s cells, and
ends with the invariant after `t + 1` trips. The prologue (`Program`) establishes the hypotheses
about the pointers and lanes; the walk of the trip (`Trip*`) proves the statements.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The first token of sequence `n` as a flat row-major token index `row·T + start`. -/
def bosOf (d : Inst) (n : Nat) : Nat := rowOf d n * d.T + startOf d n

/-- The lane values the program computes before its loop (source lines 131–140): lane `jj` of
`o_v`, `m_v` is column `32·jv + jj` and whether it exists; lane `a` of `o_k_b`, `m_k_b` is state row
`64·(b−1) + a` and whether it exists. -/
structure Lanes (d : Inst) (jv : Nat) (o_v : Tensor [BV] I64) (m_v : Tensor [BV] Bool)
    (o_k1 o_k2 o_k3 o_k4 : Tensor [64] I32) (m_k1 m_k2 m_k3 m_k4 : Tensor [64] Bool) : Prop where
  ov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int)
  mv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V)
  ok1 : ∀ a : Fin 64, (o_k1.get1 a).toInt = ((0 + a : Nat) : Int)
  ok2 : ∀ a : Fin 64, (o_k2.get1 a).toInt = ((64 + a : Nat) : Int)
  ok3 : ∀ a : Fin 64, (o_k3.get1 a).toInt = ((128 + a : Nat) : Int)
  ok4 : ∀ a : Fin 64, (o_k4.get1 a).toInt = ((192 + a : Nat) : Int)
  mk1 : ∀ a : Fin 64, m_k1.get1 a = decide (0 + a < d.K)
  mk2 : ∀ a : Fin 64, m_k2.get1 a = decide (64 + a < d.K)
  mk3 : ∀ a : Fin 64, m_k3.get1 a = decide (128 + a < d.K)
  mk4 : ∀ a : Fin 64, m_k4.get1 a = decide (192 + a < d.K)

/-- **One trip of the main loop** (`[K, V]` layout): for the program of sequence `n`, value head
`hv`, block `jv`, with the pointers of source lines 118–123 (`kp`, `wp`, `vp`, `vnp`, `hp`), the
sequence's `bos`, `i_h` and length `T`, and the lanes; trip `t` (`iT = t`, and the gate's offset
`goff` the index of the chunk's last token) takes the invariant after `t` trips to the invariant
after `t + 1`, reading outside the outputs and writing chunk `t`'s cells. -/
def TripNonvf (d : Inst) (E : Env) (init : Mem) : Prop :=
  ∀ (n hv jv : Nat), n < d.N → hv < d.HV → jv < NV d →
  ∀ (kp wp vp vnp hp : Ptr .bf16) (gp : Ptr .f32) (bos i_h T : I64)
    (o_v : Tensor [BV] I64) (m_v : Tensor [BV] Bool)
    (o_k1 o_k2 o_k3 o_k4 : Tensor [64] I32) (m_k1 m_k2 m_k3 m_k4 : Tensor [64] Bool),
  kp = d.k.ptr (((bosOf d n * d.H + kHead d hv) * d.K : Nat) : Int) →
  wp = d.w.ptr (((bosOf d n * d.HV + hv) * d.K : Nat) : Int) →
  vp = d.u.ptr (((bosOf d n * d.HV + hv) * d.V : Nat) : Int) →
  vnp = d.vn.ptr (((bosOf d n * d.HV + hv) * d.V : Nat) : Int) →
  hp = d.h.ptr (((cbase d n * d.HV + hv) * d.K * d.V : Nat) : Int) →
  gp = d.g.ptr 0 →
  bos.toInt = (bosOf d n : Int) → i_h.toInt = (hv : Int) → T.toInt = (lenOf d n : Int) →
  Lanes d jv o_v m_v o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4 →
  ∀ (t : Nat), t < nch d n → ∀ (iT goff : I64), iT.toInt = (t : Int) →
  (d.useG = true →
    goff.toInt = (gIdx d (rowOf d n) (startOf d n + BT * t + cl d n t - 1) hv : Int)) →
  ∀ (st : St BV) (m : Mem), Inv d init n hv jv t st m →
    Safe E (Rset d) (Wt d n hv jv t) (fun st' m' => Inv d init n hv jv (t + 1) st' m') m
      (chunk_step kp wp vp vnp hp gp d.H d.HV d.K d.V BT BV d.useG bos i_h T o_v m_v
        o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4 iT goff st)

/-- **One trip of the main loop** (`[V, K]` layout): as `TripNonvf`. -/
def TripVf (d : Inst) (E : Env) (init : Mem) : Prop :=
  ∀ (n hv jv : Nat), n < d.N → hv < d.HV → jv < NV d →
  ∀ (kp wp vp vnp hp : Ptr .bf16) (gp : Ptr .f32) (bos i_h T : I64)
    (o_v : Tensor [BV] I64) (m_v : Tensor [BV] Bool)
    (o_k1 o_k2 o_k3 o_k4 : Tensor [64] I32) (m_k1 m_k2 m_k3 m_k4 : Tensor [64] Bool),
  kp = d.k.ptr (((bosOf d n * d.H + kHead d hv) * d.K : Nat) : Int) →
  wp = d.w.ptr (((bosOf d n * d.HV + hv) * d.K : Nat) : Int) →
  vp = d.u.ptr (((bosOf d n * d.HV + hv) * d.V : Nat) : Int) →
  vnp = d.vn.ptr (((bosOf d n * d.HV + hv) * d.V : Nat) : Int) →
  hp = d.h.ptr (((cbase d n * d.HV + hv) * d.K * d.V : Nat) : Int) →
  gp = d.g.ptr 0 →
  bos.toInt = (bosOf d n : Int) → i_h.toInt = (hv : Int) → T.toInt = (lenOf d n : Int) →
  Lanes d jv o_v m_v o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4 →
  ∀ (t : Nat), t < nch d n → ∀ (iT goff : I64), iT.toInt = (t : Int) →
  (d.useG = true →
    goff.toInt = (gIdx d (rowOf d n) (startOf d n + BT * t + cl d n t - 1) hv : Int)) →
  ∀ (st : StV BV) (m : Mem), InvV d init n hv jv t st m →
    Safe E (Rset d) (Wt d n hv jv t) (fun st' m' => InvV d init n hv jv (t + 1) st' m') m
      (chunk_step_vf kp wp vp vnp hp gp d.H d.HV d.K d.V BT BV d.useG bos i_h T o_v m_v
        o_k1 o_k2 o_k3 o_k4 m_k1 m_k2 m_k3 m_k4 iT goff st)

end Claims.ChunkDeltaH
