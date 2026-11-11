import Proof.Defs
import Proof.NumUtil

/-!
# `chunk_delta_h_bf16`: the quantities of a column read only the cells they name

If two memories agree on the elements of `h` that a chunk's state occupies, the chunk's `Hs`,
`StOk`, `vrefS`, `tolV`, `Step` and `errStep` are the same in both.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- Two memories read the same `bfloat16` elements of `h` in column `j` of chunk `t` (all `K`
rows) of sequence `n` and head `hv`. -/
def SameH (d : Inst) (m m' : Mem) (n hv j t : Nat) : Prop :=
  ∀ c, c < d.K → rdB m' d.h (hIdx d (cbase d n + t) hv c j) = rdB m d.h (hIdx d (cbase d n + t) hv c j)

/-- A read of a `bfloat16` element depends only on the element's bytes. -/
theorem rdB_congr {m m' : Mem} {b : Buf .bf16} {i : Nat}
    (h : ∀ x ∈ Mem.span .bf16 (b.ptr (i : Int)).addr, m' x = m x) : rdB m' b i = rdB m b i :=
  Mem.read_congr h

/-- A read of a `float32` element depends only on the element's bytes. -/
theorem rdF_congr {m m' : Mem} {b : Buf .f32} {i : Nat}
    (h : ∀ x ∈ Mem.span .f32 (b.ptr (i : Int)).addr, m' x = m x) : rdF m' b i = rdF m b i :=
  Mem.read_congr h

variable {d : Inst} {init m m' : Mem} {n hv j t : Nat}

/-- The stored state's entries are the same. -/
theorem Hs_congr (h : SameH d m m' n hv j t) {c : Nat} (hc : c < d.K) :
    Hs m' d n hv j t c = Hs m d n hv j t c := by
  unfold Hs valB
  rw [h c hc]

/-- `‖h[t]‖` is the same. -/
theorem hn_congr (h : SameH d m m' n hv j t) : hn m' d n hv j t = hn m d n hv j t :=
  colNorm_congr fun _ hc => Hs_congr h hc

/-- `‖liftMag h[t]‖` is the same. -/
theorem hl_congr (h : SameH d m m' n hv j t) : hl m' d n hv j t = hl m d n hv j t := by
  refine colNorm_congr fun c hc => ?_
  unfold valB
  rw [h c hc]

/-- `StOk` is the same. -/
theorem StOk_congr (h : SameH d m m' n hv j t) : StOk m' d n hv j t ↔ StOk m d n hv j t := by
  unfold StOk
  refine and_congr (forall₂_congr fun c hc => by rw [h c hc]) ?_
  rw [colNorm_congr fun _ hc => Hs_congr h hc]

/-- The reference of the new values is the same. -/
theorem vrefS_congr (h : SameH d m m' n hv j t) (i : Nat) :
    vrefS init m' d n hv j t i = vrefS init m d n hv j t i := by
  unfold vrefS
  rw [Finset.sum_congr rfl fun c hc => by rw [Hs_congr h (Finset.mem_range.mp hc)]]

/-- The magnitude of a new value is the same. -/
theorem vAbs_congr (h : SameH d m m' n hv j t) (i : Nat) :
    vAbs init m' d n hv j t i = vAbs init m d n hv j t i := by
  unfold vAbs
  rw [vrefS_congr h, hl_congr h]

/-- The tolerance of a new value is the same. -/
theorem tolV_congr (h : SameH d m m' n hv j t) (i : Nat) :
    tolV init m' d n hv j t i = tolV init m d n hv j t i := by
  unfold tolV
  rw [vAbs_congr h]

/-- One exact step is the same. -/
theorem Step_congr (h : SameH d m m' n hv j t) (y : Nat → ℝ) (c : Nat) :
    Step init m' d n hv j t y c = Step init m d n hv j t y c := by
  unfold Step
  rw [Finset.sum_congr rfl fun i _ => by rw [vrefS_congr h]]

/-- The tolerance of a step is the same. -/
theorem errStep_congr (h : SameH d m m' n hv j t) :
    errStep init m' d n hv j t = errStep init m d n hv j t := by
  unfold errStep
  simp only [vAbs_congr h, hn_congr h]

end Claims.ChunkDeltaH
