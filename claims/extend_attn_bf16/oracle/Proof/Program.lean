import Proof.Preamble
import Proof.TileQ
import Proof.PrefStep
import Proof.ExtStep
import Proof.Epilogue
import Proof.LoopRange

/-!
# The loops and the whole program

`afterQ_safe`: the cached-part loop (`safe_forRange`, invariant `LoopInvP`); `afterPref_safe`: the
new-token loop (invariant `LoopInvE`) and the epilogue; `safe_program`: the body.
-/

namespace Claims.ExtendAttn

open TVA

/-- Nothing is seen before the first cached-part trip. -/
theorem seenP_zero (d : Inst) (m : Mem) (b i : Nat) : seenP d m b i 0 = ∅ := by
  simp [seenP]

/-- **The cached-part loop** starts from `e_max = -∞`, `deno = 0`, `acc = 0`, keeps `LoopInvP` over its
`cdiv(pre, 64)` trips, and the rest runs from the final state. -/
theorem afterQ_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    (hm : mOf d i * d.BM < extOf d init (bOf d i)) {cm : Option (I64 × Option I64)}
    {q : Tensor [d.BM, d.BD] BF16} (hq : QFacts d init i q)
    (hafter : ∀ st : LoopSt d, LoopInvP d init i (tripsPre d init (bOf d i)) st init →
      Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
        (afterPref d s cm (maskM d s) q st)) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      (afterQ d s cm (maskM d s) q) := by
  obtain ⟨hb, -, -⟩ := prog_range hi
  obtain ⟨-, -, -, hpe⟩ := req_sizes hpre hb
  have hpre2 : preOf d init (bOf d i) < 2 ^ 18 + 1 := by omega
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.toInt_ofInt_zero
  have hB : (IntV.ofNat BN : I32).toInt = BN := IntV.I32.toInt_ofNat (by norm_num [BN])
  obtain ⟨hrange, htrip⟩ := range_ok (lo := (IntV.ofInt 0 : I32)) (hi := s.cur_seq_len_prefix)
    (L := 0) (H := preOf d init (bOf d i)) h0 hs.pre (Nat.zero_le _) (by norm_num; omega)
  unfold afterQ
  refine safe_sub (fun e_max he => ?_)
  have hr0 : ∀ j, e_max.get j = FloatV.negInf := by
    intro j
    have h := he j
    have e : FloatV.sub (FloatV.ofPat 0 : F32) pinf32 = FloatV.negInf := by decide
    rw [tl.zeros_get_f32] at h
    simp only [Tensor.get_full] at h
    rw [e] at h
    rcases h with h | ⟨h, -⟩
    · exact h.symm
    · exact absurd h (by decide)
  have key := safe_forRange (E := (launch d).env P) (R := Rf d)
    (Q := fun _ m => Q d init i m) (m := init)
    (init := ((tl.zeros [d.BM, d.BD] .f32, tl.zeros [d.BM] .f32, e_max) : LoopSt d))
    (body := fun start_n st => prefBody d s (maskM d s) q start_n st)
    (k := fun st => afterPref d s cm (maskM d s) q st) hrange
    (fun n st m => LoopInvP d init i n st m) (fun _ => ∅) (Wk := Wf d init i)
    (fun _ _ _ _ => by simp) (by simp)
    ⟨rfl, fun r hr => ?_⟩
    (fun t ht st m hI => ?_)
    (fun st m hI => ?_)
  · simpa [BN] using key
  · rw [seenP_zero]
    exact RowFl.zero _ _ _ _ _ _ _ (hr0 _) (tl.zeros_get_f32 _) (fun e => tl.zeros_get_f32 _)
  · have h64 : (IntV.ofNat 64 : I32).toInt = 64 := IntV.I32.toInt_ofNat (by norm_num)
    have ht' : t < tripsPre d init (bOf d i) := by
      rw [htrip] at ht; unfold tripsPre BN; omega
    have hsn : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofNat 64 : I32).toInt) : I32).toInt
        = ((BN * t : Nat) : Int) := by
      rw [h0, h64, IntV.I32.toInt_ofInt (by unfold tripsPre BN at ht'; push_cast; omega)]
      push_cast [BN]; ring
    exact pref_trip_safe hP hd hpre hi hs hm hq ht' hsn hI
  · have hm0 := hI.1
    subst hm0
    have e : (preOf d m (bOf d i) - 0 + 63) / 64 = tripsPre d m (bOf d i) := by
      unfold tripsPre BN; omega
    rw [htrip, e] at hI
    exact hafter st hI

/-- `cur_block_m_end`'s value: `min(ext, (m + 1) * BLOCK_M)`. -/
theorem cbe_val {d : Inst} (hd : d.ok) {init : Mem} {i : Nat} (hi : i < (launch d).size)
    {s : Sc d} (hs : ScOk d init i s) :
    ((tl.iminimum (Tensor.scalar s.cur_seq_len_extend)
        (toQv d.cm (Tensor.scalar ((s.cur_block_m + IntV.ofInt 1) *
          IntV.ofNat d.BM)))).item).toInt = (blockEnd d init i : Int) := by
  obtain ⟨-, -, hm⟩ := prog_range hi
  have hnb := hd.nb_lt
  have hbm := bm_le hd
  have hB : (IntV.ofNat d.BM : I32).toInt = d.BM := IntV.I32.toInt_ofNat (by omega)
  have h1 : (IntV.ofInt 1 : I32).toInt = ((1 : Nat) : Int) := by
    simpa using (IntV.I32.toInt_ofInt (x := 1) (by norm_num))
  have hm1 : (s.cur_block_m + IntV.ofInt 1).toInt = ((mOf d i + 1 : Nat) : Int) :=
    IntV.I32.add_nat hs.blk h1 (by omega)
  have hmb : (mOf d i + 1) * d.BM ≤ 2 ^ 16 * 128 := Nat.mul_le_mul (by omega) hbm
  have h2 : ((s.cur_block_m + IntV.ofInt 1) * IntV.ofNat d.BM).toInt =
      (((mOf d i + 1) * d.BM : Nat) : Int) := IntV.I32.mul_nat hm1 hB (by omega)
  show (IntV.min s.cur_seq_len_extend
    ((toQv d.cm (Tensor.scalar ((s.cur_block_m + IntV.ofInt 1) * IntV.ofNat d.BM))).item)).toInt = _
  rw [IntV.toInt_min, toQv_scalar_item, castQ_toInt, h2, hs.ext]
  unfold blockEnd
  push_cast; rfl

/-- **The new-token loop and the epilogue**, from the start `es` of the loop (`extStart`) and the
end `cbe` of the block, given the cached-part loop's final state. -/
theorem ext_loop_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    (hm : mOf d i * d.BM < extOf d init (bOf d i)) {cm : Option (I64 × Option I64)}
    (hcm : CmOk d init i cm) {q : Tensor [d.BM, d.BD] BF16} (hq : QFacts d init i q)
    {es : I32} (hes : es.toInt = (extStart d i : Int)) {cbe : QI d.cm}
    (hcbe : cbe.toInt = (blockEnd d init i : Int)) {st : LoopSt d}
    (hI : LoopInvP d init i (tripsPre d init (bOf d i)) st init) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      (tl.forRange (toQv d.cm (Tensor.scalar es)).item cbe (IntV.ofNat BN) st
        (fun start_n st => extBody d s cm (maskM d s) q cbe start_n st) >>= fun x =>
        epilogue d s (maskM d s) x.1 x.2.1) := by
  obtain ⟨hb, -, hmb⟩ := prog_range hi
  have hbm := bm_le hd
  have hblk : blockEnd d init i ≤ 2 ^ 18 := by
    unfold blockEnd
    have := (req_sizes hpre hb).2.1
    have := hd.maxExt_le
    omega
  have hesle : extStart d i < blockEnd d init i := by
    unfold blockEnd extStart
    have hmm : (mOf d i + 1) * d.BM = mOf d i * d.BM + d.BM := by ring
    have hbp : 0 < d.BM := by rw [hd.bm]; split_ifs <;> omega
    split_ifs
    · have := Nat.div_mul_le_self (mOf d i * d.BM - d.sw) BN
      simp only [BN] at this ⊢
      omega
    · omega
  have hlo : (toQv d.cm (Tensor.scalar es)).item.toInt = (extStart d i : Int) := by
    rw [toQv_scalar_item, castQ_toInt, hes]
  obtain ⟨hrange, htrip⟩ := range_ok (lo := (toQv d.cm (Tensor.scalar es)).item) (hi := cbe)
    (L := extStart d i) (H := blockEnd d init i) hlo hcbe hesle.le
    (lt_of_le_of_lt (by omega : blockEnd d init i + 64 ≤ 2 ^ 31 - 1) (by
      have := lt_pow_qoW d.cm (n := 2 ^ 31 - 1) (by norm_num)
      omega))
  have key := safe_forRange (E := (launch d).env P) (R := Rf d)
    (Q := fun _ m => Q d init i m) (m := init) (init := st)
    (body := fun start_n st => extBody d s cm (maskM d s) q cbe start_n st)
    (k := fun x => epilogue d s (maskM d s) x.1 x.2.1) hrange
    (fun k st m => LoopInvE d init i (extStart d i) k st m) (fun _ => ∅) (Wk := Wf d init i)
    (fun _ _ _ _ => by simp) (by simp)
    ⟨rfl, fun r hr => ?_⟩
    (fun t ht st m hI => ?_)
    (fun st m hI => ?_)
  · simpa [BN] using key
  · have := hI.2 r hr
    rw [← seen_start hd hpre hi hr] at this
    simpa using this
  · have hm0 : init = m := hI.1.symm
    subst hm0
    rw [htrip] at ht
    have hk : extStart d i + BN * t < blockEnd d init i := by unfold BN; omega
    have h64 : (IntV.ofNat 64 : IntV true (qoW d.cm)).toInt = 64 :=
      IntV.toInt_ofNat (lt_pow_qoW d.cm (by norm_num))
    have hsn : (IntV.ofInt ((toQv d.cm (Tensor.scalar es)).item.toInt + (t : Int) *
        (IntV.ofNat 64 : IntV true (qoW d.cm)).toInt) : QI d.cm).toInt =
        ((extStart d i + BN * t : Nat) : Int) := by
      rw [hlo, h64, IntV.toInt_ofInt (by have := qoW_ge d.cm; omega)]
      · push_cast [BN]; ring
      · have : extStart d i + BN * t < 2 ^ (qoW d.cm - 1) :=
          lt_pow_qoW d.cm (by omega)
        have h2 := IntV.sfits_natCast (w := qoW d.cm) (n := extStart d i + BN * t) this
        push_cast [BN] at h2 ⊢
        convert h2 using 2; ring
    exact ext_trip_safe hP hd hpre hi hs hm hcm hq hk hcbe hsn hI
  · have hm0 : init = m := hI.1.symm
    subst hm0
    rw [htrip] at hI
    refine epilogue_safe hP hd hpre hi hs hI ?_ ?_
    · unfold BN; omega
    · unfold BN; omega

/-- **Everything after the cached-part loop** is safe. -/
theorem afterPref_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    (hm : mOf d i * d.BM < extOf d init (bOf d i)) {cm : Option (I64 × Option I64)}
    (hcm : CmOk d init i cm) {q : Tensor [d.BM, d.BD] BF16} (hq : QFacts d init i q)
    (st : LoopSt d) (hI : LoopInvP d init i (tripsPre d init (bOf d i)) st init) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      (afterPref d s cm (maskM d s) q st) := by
  obtain ⟨acc, deno, e_max⟩ := st
  obtain ⟨-, -, hmb⟩ := prog_range hi
  have hnb := hd.nb_lt
  have hbm := bm_le hd
  have hcbe := cbe_val hd hi hs
  unfold afterPref
  simp only [safe_toQ]
  by_cases hsw : 0 < d.sw
  · simp only [hsw, ite_true]
    have hB : (IntV.ofNat d.BM : I32).toInt = d.BM := IntV.I32.toInt_ofNat (by omega)
    have hsw31 : d.sw ≤ 2 ^ 18 := hd.sw_le
    have hS : (IntV.ofNat d.sw : I32).toInt = d.sw := IntV.I32.toInt_ofNat (by omega)
    have hmb' : mOf d i * d.BM ≤ 2 ^ 16 * 128 := Nat.mul_le_mul (by omega) hbm
    have h1 : (s.cur_block_m * IntV.ofNat d.BM : I32).toInt = ((mOf d i * d.BM : Nat) : Int) :=
      IntV.mul_ofNat hs.blk (by omega)
    have h2 : (s.cur_block_m * IntV.ofNat d.BM - IntV.ofNat d.sw : I32).toInt =
        ((mOf d i * d.BM : Nat) : Int) - d.sw :=
      IntV.toInt_sub h1 hS (IntV.I32.sfits (by omega) (by omega))
    have h0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.toInt_ofInt_zero
    have hmax : (IntV.max (s.cur_block_m * IntV.ofNat d.BM - IntV.ofNat d.sw : I32)
        (IntV.ofInt 0)).toInt = (((mOf d i * d.BM - d.sw : Nat)) : Int) := by
      rw [IntV.toInt_max, h2, h0]; omega
    simp only [show ∀ a b : I32, tl.imaximum (Tensor.scalar a) (Tensor.scalar b) =
      Tensor.scalar (IntV.max a b) from fun _ _ => rfl]
    have h64 : (IntV.ofNat BN : I32).toInt = BN := IntV.I32.toInt_ofNat (by norm_num [BN])
    refine safe_idiv1 (by rw [hmax]; positivity) (by rw [h64]; norm_num [BN]) fun r hr => ?_
    simp only [pure_bind, safe_toQ]
    refine ext_loop_safe hP hd hpre hi hs hm hcm hq (es := r.item * IntV.ofNat BN) ?_ hcbe hI
    have := IntV.I32.mul_nat (a := r.item) (b := IntV.ofNat BN)
      (x := (mOf d i * d.BM - d.sw) / BN) (y := BN) (by rw [hr, hmax, h64]; norm_cast) h64
      (by simp only [BN]; omega)
    rw [this]; unfold extStart; simp only [hsw, ite_true]
  · simp only [hsw, ite_false]
    simp only [pure_bind, safe_toQ]
    refine ext_loop_safe hP hd hpre hi hs hm hcm hq (es := IntV.ofInt 0) ?_ hcbe hI
    rw [IntV.toInt_ofInt_zero]; unfold extStart; simp [hsw]

/-- **One program** of the launch is safe from `init`: it reads inputs only, writes its tile of
`o_extend`, and ends with `Q`. -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) :=
  body_safe hP hd hpre i.isLt fun _s hs hm =>
    elseProg_safe hP hd hpre i.isLt hs fun cm hcm =>
      afterCm_safe hP hd hpre i.isLt hs cm fun _q hq =>
        afterQ_safe hP hd hpre i.isLt hs hm hq fun st hst =>
          afterPref_safe hP hd hpre i.isLt hs hm hcm hq st hst

end Claims.ExtendAttn
