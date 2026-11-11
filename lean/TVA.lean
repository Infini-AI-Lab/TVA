import TVA.Core
import TVA.Triton
import TVA.Launch
import TVA.Device
import TVA.Device.Tolerance
import TVA.Claim

/-!
# TVA: the trusted part

Everything a claim is stated in, and everything a reader of a proved claim trusts:

* `TVA.Core`: values, tensors, memory, the platform and the program monad (`Base/`, `Tensor`,
  `Memory`, `Platform`, `Program`);
* `TVA.Triton`: the meaning of each `tl.*` operation;
* `TVA.Launch`: a launch of a grid of programs, its outcomes, and `Correct`;
* `TVA.Device`: the device facts (`Device P`), shown satisfiable by `Device.ref_ok`, and the
  tolerance formulas specifications use (`TVA.Device.Tolerance`);
* `TVA.Claim`: what a claim is and what `claim.Holds` means.

Claims import `TVA`. Proofs also import `TVA.Proof`, the proof kit, which is proved, not trusted.
-/
