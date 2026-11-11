import Proof.BodyBf16
import Proof.BodyFp8
import Proof.BodyFp8in

/-!
# `kv_cache_write_bf16`: the model's program is `Safe` in every mode

The three mode-specific `Safe` proofs, dispatched on the instance's mode.
-/

namespace Claims.KvCacheWrite

open TVA

/-- Program `i` is `Safe` in every mode, with footprints `R d`, `W d init i` and post-condition
`Q d init i`. -/
theorem safe_body {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (hdivT : Device.DivTotal E.P .binary32)
    (hdivF : Device.DivFull E.P)
    (hcvtT : Device.CvtTotal E.P .binary32 .e4m3)
    (hcvtS : Device.CvtSatF32E4m3 E.P)
    (i : Fin (launch d).size) :
    Safe E (R d) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m) init
      ((model d).body ((launch d).ctx i)) := by
  rcases d with ⟨m, nt, nh, hs, bs, nb, ts, kst, vst, bst, pst, hsd, key, value, kc, vc, sm,
    ksc, vsc⟩
  cases m
  · exact safe_bf16 hd hpre E i rfl
  · exact safe_fp8 hd hpre E hdivT hdivF hcvtT hcvtS i rfl
  · exact safe_fp8in hd hpre E i rfl

end Claims.KvCacheWrite
