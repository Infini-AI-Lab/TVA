import Proof.Inv
import Proof.Scalars

/-!
# What the loads hand over

The facts the walk takes from each load, in the form the float glue consumes: a loaded tile holds
the claim's value at each enabled lane and `+0` elsewhere.
-/

namespace Claims.ExtendAttn

open TVA

/-- The loaded `q` tile: lane `(r, e)` holds `q[qs + m*BM + r, h, e]` inside the request's new
tokens and the head dimension, and `+0` outside. -/
def QFacts (d : Inst) (init : Mem) (i : Nat) (q : Tensor [d.BM, d.BD] BF16) : Prop :=
  ∀ (r : Fin d.BM) (e : Fin d.BD), DType.val? .bf16 (q.get2 r e) =
    some (if validRow d init i r ∧ e.val < d.D then
      qv d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e else 0)

/-- The cached-part trip `n`'s slots (`kv_indices[kv_indptr[b] + 64 n + c]`, `0` beyond the cached-part). -/
def KvLocFacts (d : Inst) (init : Mem) (i n : Nat) (kvloc : Tensor [BN] I64) : Prop :=
  ∀ c : Fin BN, (kvloc.get1 c).toInt =
    if BN * n + c < preOf d init (bOf d i) then (slotOf d init (bOf d i) (BN * n + c) : Int) else 0

/-- The cached-part trip `n`'s `k` tile: lane `(e, c)` holds component `e` of the key at position
`64 n + c` of the cache when that is inside the cached-part and `e < D`, else `+0`. -/
def KFactsP (d : Inst) (init : Mem) (i n : Nat) (k : Tensor [d.BD, BN] BF16) : Prop :=
  ∀ (e : Fin d.BD) (c : Fin BN), DType.val? .bf16 (k.get2 e c) =
    some (if BN * n + c < preOf d init (bOf d i) ∧ e.val < d.D then
      keyK d init (bOf d i) (hOf d i) (BN * n + c) e else 0)

/-- The cached-part trip `n`'s `v` tile: lane `(c, e)`. -/
def VFactsP (d : Inst) (init : Mem) (i n : Nat) (v : Tensor [BN, d.BD] BF16) : Prop :=
  ∀ (c : Fin BN) (e : Fin d.BD), DType.val? .bf16 (v.get2 c e) =
    some (if BN * n + c < preOf d init (bOf d i) ∧ e.val < d.D then
      keyV d init (bOf d i) (hOf d i) (BN * n + c) e else 0)

/-- Whether lane `c` of the cached-part trip `n` is a key row `r` attends (a `final_mask` lane). -/
def vP (d : Inst) (init : Mem) (i r n c : Nat) : Bool :=
  decide (BN * n + c ∈ keys d init (bOf d i) (mOf d i * d.BM + r) ∧ BN * n + c < preOf d init (bOf d i))

/-- The new-token trip starting at the new token `sn` (a multiple of `64`): its `k` tile holds
component `e` of the key at sequence position `pre + sn + c` (the new token `sn + c`), when that
is below `cur_block_m_end = cbe` and `e < D`, else `+0`. -/
def KFactsE (d : Inst) (init : Mem) (i sn cbe : Nat) (k : Tensor [d.BD, BN] BF16) : Prop :=
  ∀ (e : Fin d.BD) (c : Fin BN), DType.val? .bf16 (k.get2 e c) =
    some (if sn + c < cbe ∧ e.val < d.D then
      keyK d init (bOf d i) (hOf d i) (preOf d init (bOf d i) + (sn + c)) e else 0)

/-- The new-token trip's `v` tile. -/
def VFactsE (d : Inst) (init : Mem) (i sn cbe : Nat) (v : Tensor [BN, d.BD] BF16) : Prop :=
  ∀ (c : Fin BN) (e : Fin d.BD), DType.val? .bf16 (v.get2 c e) =
    some (if sn + c < cbe ∧ e.val < d.D then
      keyV d init (bOf d i) (hOf d i) (preOf d init (bOf d i) + (sn + c)) e else 0)

/-- Whether lane `c` of the new-token trip at `sn` is a key row `r` attends. -/
def vE (d : Inst) (init : Mem) (i r sn c : Nat) : Bool :=
  decide (preOf d init (bOf d i) + (sn + c) ∈ keys d init (bOf d i) (mOf d i * d.BM + r))

/-- `cur_block_m_end`: the end of the block's new tokens, `min(ext, (m + 1) * BM)`. -/
def blockEnd (d : Inst) (init : Mem) (i : Nat) : Nat :=
  min (extOf d init (bOf d i)) ((mOf d i + 1) * d.BM)

/-- The first new-token trip's start `extend_start` (`0` without a window). -/
def extStart (d : Inst) (i : Nat) : Nat :=
  if 0 < d.sw then (mOf d i * d.BM - d.sw) / BN * BN else 0

end Claims.ExtendAttn
