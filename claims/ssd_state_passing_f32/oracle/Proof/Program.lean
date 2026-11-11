import Claim
import TVA.Proof
import Proof.Trip
import Proof.Lanes

/-!
# The model's body, in the shape the proof walks

For each launch (with and without initial states), the model's program for a context `c` is the
loads and stores before the loop, the loop `tl.forRange` over `tripI` / `tripN`, and `pure ()`.
-/
namespace Claims.SsdStatePassing

open TVA

/-- `pid_m · BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` (source line 143). -/
def offsM (c : Ctx) : Tensor [BL - 0] I32 := tl.program_id c 0 * (IntV.ofNat BL : I32) + tl.arange 0 BL

/-- `offs_m < dim`. -/
def maskM (d : Inst) (c : Ctx) : Tensor [BL - 0] Bool := (offsM c).ltS (IntV.ofNat d.dim)

/-- The pointer block into `initial_states` (lines 135-137, 152-153). -/
def initP (d : Inst) (c : Ctx) : Tensor [BL - 0] (Ptr .f32) :=
  d.initst.ptr 0 + tl.program_id c 2 * (IntV.ofNat (sSh d) : I32) + offsM c * (IntV.ofNat 1 : I32)

/-- The pointer block into `states` (lines 125, 144). -/
def statesP (d : Inst) (c : Ctx) : Tensor [BL - 0] (Ptr .f32) :=
  d.states.ptr 0 + (tl.program_id c 1 * (IntV.ofNat (sSb d) : I32) +
    tl.program_id c 2 * (IntV.ofNat (sSh d) : I32)) + offsM c * (IntV.ofNat 1 : I32)

/-- The pointer into `dA_cumsum` at the chunk's last entry (lines 126-130). -/
def dAP (d : Inst) (c : Ctx) : Ptr .f32 :=
  d.dA.ptr 0 + (tl.program_id c 1 * (IntV.ofNat (sDb d) : I32) +
    tl.program_id c 2 * (IntV.ofNat (sDh d) : I32) +
    ((IntV.ofNat d.chunk : I32) - IntV.ofInt 1) * (IntV.ofNat 1 : I32))

/-- The pointer block into `out` (lines 131, 145). -/
def outP (d : Inst) (c : Ctx) : Tensor [BL - 0] (Ptr .f32) :=
  d.out.ptr 0 + (tl.program_id c 1 * (IntV.ofNat (sSb d) : I32) +
    tl.program_id c 2 * (IntV.ofNat (sSh d) : I32)) + offsM c * (IntV.ofNat 1 : I32)

/-- The pointer block into `final_states` (lines 132-134, 146). -/
def finP (d : Inst) (c : Ctx) : Tensor [BL - 0] (Ptr .f32) :=
  d.finalst.ptr 0 + (tl.program_id c 1 * (IntV.ofNat (sSc d) : I32) +
    tl.program_id c 2 * (IntV.ofNat (sSh d) : I32)) + offsM c * (IntV.ofNat 1 : I32)

/-- The pointer into `seq_idx` (lines 140-141). -/
def seqP (d : Inst) (c : Ctx) : Ptr .i32 :=
  d.seqIdx.ptr 0 + tl.program_id c 1 * (IntV.ofNat (sQb d) : I32)

/-- **The model with initial states**: the load and store before the loop, the loop, `pure ()`. -/
theorem bodyI_eq (d : Inst) (hh : d.hasInit = true) (c : Ctx) :
    (model d).body c =
      (tl.load (initP d c) (some (maskM d c)) (some (bc (FloatV.litF32 (FloatV.ofPat 0)))) >>=
        fun states => tl.store (outP d c) states (some (maskM d c)) >>= fun _ =>
        tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.nchunks : I32) (IntV.ofInt 1 : I32)
          (states, initP d c, statesP d c, dAP d c,
            (outP d c).map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32)),
            (IntV.ofInt 0 : I32), (IntV.ofInt 0 : I32))
          (tripI (IntV.ofNat d.nmeta) (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen)
            (IntV.ofNat d.chunk) (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sDc d))
            (IntV.ofNat 1) (IntV.ofNat (sSc d)) (IntV.ofNat 1) BL (offsM c) (seqP d c)
            (d.chunkOffs.ptr 0) (initP d c) (finP d c)) >>= fun _ => pure ()) := by
  rw [model, ite_eq_left hh]
  rfl

/-- **The model without initial states**. -/
theorem bodyN_eq (d : Inst) (hh : d.hasInit = false) (c : Ctx) :
    (model d).body c =
      (tl.store (outP d c) (tl.zeros [BL] .f32) (some (maskM d c)) >>= fun _ =>
        tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.nchunks : I32) (IntV.ofInt 1 : I32)
          (tl.zeros [BL] .f32, statesP d c, dAP d c,
            (outP d c).map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32)),
            (IntV.ofInt 0 : I32))
          (tripN (IntV.ofNat d.dim) (IntV.ofNat d.nchunks) (IntV.ofNat d.seqlen)
            (IntV.ofNat d.chunk) (IntV.ofNat (sSc d)) (IntV.ofNat (sSc d)) (IntV.ofNat (sDc d))
            (IntV.ofNat 1) BL (offsM c) (seqP d c) (finP d c)) >>= fun _ => pure ()) := by
  rw [model, ite_eq_right (by simp [hh])]
  rfl

end Claims.SsdStatePassing
