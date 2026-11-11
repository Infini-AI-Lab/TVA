import Claim
import TVA.Proof
import Proof.Defs

/-!
# The two branches of the kernel, as programs

The kernel body is a runtime `if` on `token_idx < prefill_tokens_with_context`. This file
names the two branches, so that the safety proofs can be stated about them separately.

The definitions copy the source text (and so the model's transcription) verbatim, with the
buffer parameters replaced by the buffers' first elements and the scalar parameters by the
instance's fields.
-/

namespace Claims.MergeAttnStates

open TVA

/-- The program of a token before `prefill_tokens_with_context`: copy the suffix row. -/
def copyProg (d : Inst) (c : Ctx) (ol : Option (Ptr .f32)) : Prog Unit := do
  let s_lse ← tl.load1 (d.suffix_lse.ptr 0 + tl.program_id c 1 * IntV.ofNat d.slhs
    + tl.program_id c 0 * IntV.ofNat d.slts)
  match ol with
  | none => pure ()
  | some output_lse =>
    tl.store1 (output_lse + tl.program_id c 1 * IntV.ofNat d.olhs
      + tl.program_id c 0 * IntV.ofNat d.olts) s_lse
  let head_arange := tl.arange 0 d.padded
  let head_mask := head_arange.ltS (IntV.ofNat d.head_size : I32)
  let s_out : Tensor [d.padded] BF16 ←
    tl.load (d.suffix_output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.phs
      + tl.program_id c 1 * IntV.ofNat d.phs + head_arange : Tensor [d.padded] (Ptr .bf16)) (some head_mask)
  tl.store (d.output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.ohs
    + tl.program_id c 1 * IntV.ofNat d.ohs + head_arange) s_out (some head_mask)

/-- The program of a token the context covers: merge the two rows by their lse. -/
def mergeProg (d : Inst) (c : Ctx) (ol : Option (Ptr .f32)) : Prog Unit := do
  let p_lse ← tl.load1 (d.prefix_lse.ptr 0 + tl.program_id c 1 * IntV.ofNat d.plhs
    + tl.program_id c 0 * IntV.ofNat d.plts)
  let s_lse ← tl.load1 (d.suffix_lse.ptr 0 + tl.program_id c 1 * IntV.ofNat d.slhs
    + tl.program_id c 0 * IntV.ofNat d.slts)
  let p_lse := if ((bc p_lse).feq (bc posInfC)).item then negInfC else p_lse
  let s_lse := if ((bc s_lse).feq (bc posInfC)).item then negInfC else s_lse
  let max_lse := (← tl.maximum (bc p_lse) (bc s_lse)).item
  let p_lse := (← tl.sub (bc p_lse) (bc max_lse)).item
  let s_lse := (← tl.sub (bc s_lse) (bc max_lse)).item
  let p_se := (← tl.exp (bc p_lse)).item
  let s_se := (← tl.exp (bc s_lse)).item
  let out_se := (← tl.add (bc p_se) (bc s_se)).item
  match ol with
  | none => pure ()
  | some output_lse =>
    let out_lse := (← tl.log (bc out_se)).item
    let out_lse := (← tl.add (bc out_lse) (bc max_lse)).item
    let out_lse := (Tensor.where_ ((bc max_lse).feq (bc negInfC)) (bc negInfC)
      (bc out_lse) : Tensor [] F32).item
    tl.store1 (output_lse + tl.program_id c 1 * IntV.ofNat d.olhs
      + tl.program_id c 0 * IntV.ofNat d.olts) out_lse
  let head_arange := tl.arange 0 d.padded
  let head_mask := head_arange.ltS (IntV.ofNat d.head_size : I32)
  let p_out : Tensor [d.padded] BF16 ←
    tl.load (d.prefix_output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.phs
      + tl.program_id c 1 * IntV.ofNat d.phs + head_arange : Tensor [d.padded] (Ptr .bf16)) (some head_mask)
  let s_out : Tensor [d.padded] BF16 ←
    tl.load (d.suffix_output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.phs
      + tl.program_id c 1 * IntV.ofNat d.phs + head_arange : Tensor [d.padded] (Ptr .bf16)) (some head_mask)
  let p_scale := (← tl.div (bc p_se) (bc out_se)).item
  let s_scale := (← tl.div (bc s_se) (bc out_se)).item
  let p32 : Tensor [d.padded] F32 ← tl.cvt p_out
  let s32 : Tensor [d.padded] F32 ← tl.cvt s_out
  let prod_p ← tl.mul p32 (bc p_scale)
  let prod_s ← tl.mul s32 (bc s_scale)
  let out32 ← tl.addc prod_p prod_s
  let out32 := Tensor.where_ (bc ((bc max_lse).feq (bc negInfC)).item) (bc zeroC) out32
  let out16 : Tensor [d.padded] BF16 ← tl.cvt out32
  tl.store (d.output.ptr 0 + tl.program_id c 0 * tl.num_programs c 1 * IntV.ofNat d.ohs
    + tl.program_id c 1 * IntV.ofNat d.ohs + head_arange) out16 (some head_mask)

/-- The lse pointer the kernel is given: the merged lse's first element, or `none`. -/
def olOf (d : Inst) : Option (Ptr .f32) := if d.outLse then some (d.output_lse.ptr 0) else none

/-- The model's body is the branch selection. -/
theorem body_split (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let token_idx := tl.program_id c 0
        let prefix_mask := ((bc token_idx).ltS (IntV.ofNat d.ptwc : I32)).item
        if !prefix_mask then copyProg d c (olOf d) else mergeProg d c (olOf d)) := rfl

end Claims.MergeAttnStates
