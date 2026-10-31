import Softmax.Model
import Softmax.Err
import Softmax.Proof

/-!
# `softmax_f32`

`softmax_kernel` of tutorial 02 (fused softmax) on `float32`. The claim's parts:

* `Softmax.Model`: the model (`softmax_kernel`, `model`, `launch`), the conditions (`Inst`,
  `Inst.ok`, `Pre`), the specification (`Post`, `Spec`) and the assumptions (`Assumptions`);
* `Softmax.Err`: the error of one row;
* `Softmax.Proof`: the proof, `main`, `consistent` and `satisfiable`.
-/
