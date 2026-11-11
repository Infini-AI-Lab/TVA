import Claim
import TVA.Proof
import Proof.Prologue
import Proof.SafeTail

/-!
# One program of the launch

The prologue gives the chunk, the sequence's first token and length and the value head
(`safe_prologue`); the tail does the rest (`safe_tail`).
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- **One program** of the launch is safe: it reads within `R`, writes within `W`, and ends with
`Q`. The prologue gives the chunk, the sequence's start and length and the value head; the tail
(`safe_tail`) does the rest. -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d) {c : Ctx}
    (hc0 : c.pid 0 = r) (hc1 : c.pid 1 = bh) :
    Safe ((launch d).env P) (R d r bh) (W d r bh) (fun _ m => Q d init r bh m) init
      ((model d).body c) :=
  safe_prologue hd hpre _ hr hbh hc0 hc1 fun _ _ _ _ h1 h2 h3 h4 =>
    safe_tail hP hd hpre hr hbh h1 h2 h3 h4

end Claims.ChunkBwdDvLocal
