import AttnBwdPre.Model
import AttnBwdPre.Err
import AttnBwdPre.Proof

/-!
# `attn_bwd_pre_f16`

`_attn_bwd_preprocess` of tutorial 06 (fused attention), as `_attention.backward` launches it:
`O` and `DO` are `float16`, `Delta` is `float32`. The claim's parts:

* `AttnBwdPre.Model`: the model (`_attn_bwd_preprocess`, `model`, `launch`), the conditions
  (`Inst`, `Inst.ok`, `Pre`), the specification (`Post`, `Spec`) and the assumptions
  (`Assumptions`);
* `AttnBwdPre.Err`: the error of one output (`row_bound`) and the tolerance (`tol_eq`, `tol_le`);
* `AttnBwdPre.Proof`: the proof, `main`, `consistent` and `satisfiable`.
-/
