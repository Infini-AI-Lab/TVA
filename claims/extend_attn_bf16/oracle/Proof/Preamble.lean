import Proof.Spec
import Proof.Lanes

/-!
# The scalar steps before the loops

The body loads `qo_indptr` and `kv_indptr`, returns early for a block beyond the request, and
otherwise loads the custom-mask data. Each is `Safe`, and the rest runs with the scalars of the
program (`ScOk`) and its mask data (`CmOk`).
-/

namespace Claims.ExtendAttn

open TVA

section
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {dt : DType}

/-- A base pointer plus an index with integer value `n` is the pointer at offset `n`. -/
theorem pre_ptr1 (b : Buf dt) {sg : Bool} {w : Nat} {x : IntV sg w} {n : Nat} (hx : x.toInt = n) :
    (b.ptr 0 + x) = b.ptr (n : Int) := by
  show (b.ptr 0).offset x.toInt = _
  rw [hx, Buf.ptr_offset]; simp

/-- A base pointer plus two indices with integer values `n` and `n'` is the pointer at offset `n + n'`. -/
theorem pre_ptr2 (b : Buf dt) {sg : Bool} {w : Nat} {x : IntV sg w} {n : Nat} (hx : x.toInt = n)
    {sg' : Bool} {w' : Nat} {y : IntV sg' w'} {n' : Nat} (hy : y.toInt = n') :
    (b.ptr 0 + x + y) = b.ptr ((n + n' : Nat) : Int) := by
  rw [pre_ptr1 b hx]
  show (b.ptr (n : Int)).offset y.toInt = _
  rw [hy, Buf.ptr_offset]; push_cast; rfl

/-- A one-element load at an in-bounds index of a valid readable buffer is safe if the continuation is safe for the value read. -/
theorem pre_load1 (b : Buf dt) {sg : Bool} {w : Nat} {x : IntV sg w} {n : Nat} (hx : x.toInt = n)
    (hn : n < b.len) (hv : b.Valid m) (hR : b.bytes ⊆ R) {k : dt.Val → Prog α}
    (hk : ∀ v, m.read dt (b.ptr (n : Int)).addr = some v → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load1 (b.ptr 0 + x) >>= k) := by
  rw [pre_ptr1 b hx]
  exact safe_load1_buf b hn hv ((b.span_sub hn).trans hR) hk

/-- A one-element load at the sum of two in-bounds indices of a valid readable buffer is safe if the continuation is safe for the value read. -/
theorem pre_load1' (b : Buf dt) {sg : Bool} {w : Nat} {x : IntV sg w} {n : Nat} (hx : x.toInt = n)
    {sg' : Bool} {w' : Nat} {y : IntV sg' w'} {n' : Nat} (hy : y.toInt = n')
    (hn : n + n' < b.len) (hv : b.Valid m) (hR : b.bytes ⊆ R) {k : dt.Val → Prog α}
    (hk : ∀ v, m.read dt (b.ptr ((n + n' : Nat) : Int)).addr = some v → Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load1 (b.ptr 0 + x + y) >>= k) := by
  rw [pre_ptr2 b hx hy]
  exact safe_load1_buf b hn hv ((b.span_sub hn).trans hR) hk
end

/-- The query/output buffer lies in the readable region. -/
theorem rf_qo (d : Inst) : d.qo.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf]; simp [hx]
/-- The key/value pointer buffer lies in the readable region. -/
theorem rf_kvp (d : Inst) : d.kvp.bytes ⊆ Rf d := by
  intro x hx; simp only [Rf]; simp [hx]

/-- A scalar load of `qo_indptr` at pointer `p = b.ptr n`: the rest runs with the loaded value. Generic in
`qo32` (case split inside); instantiate at `d.cm`. -/
theorem pre_safe_loadQo {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    (qo32 : Bool) {b : Buf (qoD qo32)} {p : Ptr (qoD qo32)} {n : Nat} (hp : p = b.ptr (n : Int))
    (hn : n < b.len) (hv : b.Valid m) (hR : b.bytes ⊆ R) {k : QI qo32 → Prog α}
    (hk : ∀ v : QI qo32,
      (m.read (qoD qo32) (b.ptr (n : Int)).addr).map (qoInt qo32) = some v.toInt →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (loadQo qo32 p >>= k) := by
  subst hp
  cases qo32 <;> simp only [loadQo] <;>
    exact safe_load1_buf b hn hv ((b.span_sub hn).trans hR) (fun v hv' => hk v (by rw [hv']; rfl))

/-- Signed subtraction of `b` with `0 <= b <= a` does not wrap: its integer value is the difference. -/
theorem pre_sub {w : Nat} {a b : IntV true w} (h0 : 0 ≤ b.toInt) (h : b.toInt ≤ a.toInt) :
    (a - b).toInt = a.toInt - b.toInt := by
  have ha := (IntV.sfits_toInt a).2
  have hp : (0 : Int) < 2 ^ (w - 1) := by positivity
  exact IntV.toInt_sub rfl rfl ⟨by linarith, by linarith⟩

/-- A natural number below 2^31 converts to a 32-bit integer of the same value. -/
theorem pre_toInt_i32 {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n :=
  IntV.toInt_ofNat (by simpa using h)

/-- The mask data loads (lines 213-219) are safe, and the rest runs with `CmOk`. -/
theorem elseProg_safe {P : Platform} (_hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    (hrest : ∀ cm, CmOk d init i cm →
      Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
        (afterCm d s cm)) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init (elseProg d s) := by
  obtain ⟨hb, -, -⟩ := prog_range hi
  obtain ⟨-, -, -, -, -, -, -, -, -, hcm, -, hmask, -⟩ := hpre
  unfold elseProg
  by_cases h : d.cm = true
  · obtain ⟨-, hmip, hwko⟩ := hcm h
    obtain ⟨⟨ms, hms, hms0⟩, hw, -⟩ := hmask h _ hb
    simp only [h, ↓reduceIte]
    refine pre_load1 d.mip hs.seq (by have := hd.mip_len h; omega) hmip (rf_mip h) fun v hv => ?_
    have hv' : v = ms := by rw [hms] at hv; exact (Option.some.inj hv).symm
    subst hv'
    have hmv : v.toInt = mStart d init (bOf d i) := by
      have e : val64 init d.mip (bOf d i) = v.toInt := by unfold val64; rw [hms]
      simp only [mStart, e]; exact (Int.toNat_of_nonneg hms0).symm
    by_cases hsw : 0 < d.sw
    · obtain ⟨w, hw1, hw0, -⟩ := hw hsw
      simp only [hsw, ↓reduceIte]
      refine pre_load1 d.wko hs.seq (by have := hd.wko_len h hsw; omega) (hwko hsw) (rf_wko h hsw) fun u hu => ?_
      have hu' : u = w := by rw [hw1] at hu; exact (Option.some.inj hu).symm
      subst hu'
      refine hrest _ (Or.inr ⟨h, _, _, rfl, hmv, Or.inr ⟨hsw, _, rfl, ?_⟩⟩)
      have e : val64 init d.wko (bOf d i) = u.toInt := by unfold val64; rw [hw1]
      simp only [wOff, hsw, ↓reduceIte, e]; exact (Int.toNat_of_nonneg hw0).symm
    · simp only [hsw, ↓reduceIte]
      exact hrest _ (Or.inr ⟨h, _, _, rfl, hmv, Or.inl ⟨by omega, rfl⟩⟩)
  · simp only [h]
    exact hrest _ (Or.inl ⟨by simpa using h, rfl⟩)

/-- **The body**: the scalar loads, the early exit for a block beyond the request (nothing to
write: `Q` holds vacuously), and the rest with the scalars `ScOk` for a block inside it. -/
theorem body_safe {P : Platform} (_hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size)
    (hrest : ∀ s : Sc d, ScOk d init i s → mOf d i * d.BM < extOf d init (bOf d i) →
      Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init (elseProg d s)) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  obtain ⟨hb, hh, hm⟩ := prog_range hi
  have hpre' := hpre
  obtain ⟨-, -, -, -, -, -, hqo, hkvp, -, -, hreq, -⟩ := hpre
  obtain ⟨q0, q1, k0, k1, hq0, hq1, hk0, hk1, h0, h01, h1T, h1m, hk0', hk01, hkNI, hsum, -⟩ :=
    hreq _ hb
  have hsz := req_sizes hpre' hb
  have hHpos : 0 < d.H := by
    have : i < d.batch * d.H * nBlocks d := hi
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h] at this; simp at this
    · exact h
  have hkvgpos : 0 < kvg d := Nat.div_pos (Nat.le_of_dvd hHpos hd.kv_dvd) hd.kv_pos
  have hkvglt : kvg d < 2 ^ 31 := lt_of_le_of_lt (Nat.div_le_self _ _) (by have := hd.H_lt; omega)
  have hbat := hd.batch_le
  have hHl := hd.H_lt
  have hnb := hd.nb_lt
  have e0 : tl.program_id ((launch d).ctx i) 0 = IntV.ofNat (bOf d i) := rfl
  have e1 : tl.program_id ((launch d).ctx i) 1 = IntV.ofNat (hOf d i) := rfl
  have e2 : tl.program_id ((launch d).ctx i) 2 = IntV.ofNat (mOf d i) := rfl
  have t0 : (IntV.ofNat (bOf d i) : I32).toInt = bOf d i := pre_toInt_i32 (by omega)
  have t1 : (IntV.ofNat (hOf d i) : I32).toInt = hOf d i := pre_toInt_i32 (by omega)
  have t2 : (IntV.ofNat (mOf d i) : I32).toInt = mOf d i := pre_toInt_i32 (by omega)
  rw [body_eq]
  simp only [e0, e1, e2]
  have t_one : (IntV.ofInt 1 : I32).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
  have hqv0 : valQo d init (bOf d i) = q0 := by simp [valQo, hq0]
  have hqv1 : valQo d init (bOf d i + 1) = q1 := by simp [valQo, hq1]
  have fv0 : val32 init d.kvp (bOf d i) = k0.toInt := by unfold val32; rw [hk0]
  have fv1 : val32 init d.kvp (bOf d i + 1) = k1.toInt := by unfold val32; rw [hk1]
  have eExt : (extOf d init (bOf d i) : Int) = q1 - q0 := by
    unfold extOf; rw [hqv0, hqv1]; omega
  have ePre : (preOf d init (bOf d i) : Int) = k1.toInt - k0.toInt := by
    unfold preOf; rw [fv0, fv1]; omega
  have eQs : (qsOf d init (bOf d i) : Int) = q0 := by unfold qsOf; rw [hqv0]; omega
  have eKs : (ksOf d init (bOf d i) : Int) = k0.toInt := by unfold ksOf; rw [fv0]; omega
  refine safe_idiv1 (x := IntV.ofNat (hOf d i)) (y := IntV.ofNat (kvg d))
    (by rw [t1]; positivity) (by rw [pre_toInt_i32 hkvglt]; exact_mod_cast hkvgpos) fun r hr => ?_
  refine pre_safe_loadQo d.cm (n := bOf d i) (pre_ptr1 d.qo t0) (by have := hd.qo_len; omega) hqo
    (rf_qo d) fun v0 hv0 => ?_
  have hv0' : v0.toInt = q0 := by have := hv0.symm.trans hq0; simpa using this
  refine pre_safe_loadQo d.cm (n := bOf d i + 1) (pre_ptr2 d.qo t0 (n' := 1) (by simpa using t_one)) (by have := hd.qo_len; omega)
    hqo (rf_qo d) fun v1 hv1 => ?_
  have hv1' : v1.toInt = q1 := by have := hv1.symm.trans hq1; simpa using this
  refine pre_load1 d.kvp t0 (by have := hd.kvp_len; omega) hkvp (rf_kvp d) fun w0 hw0 => ?_
  have hw0' : w0 = k0 := by rw [hk0] at hw0; exact (Option.some.inj hw0).symm
  subst hw0'
  refine pre_load1' d.kvp (n' := 1) t0 (by simpa using t_one) (by have := hd.kvp_len; omega) hkvp (rf_kvp d) fun w1 hw1 => ?_
  have hw1' : w1 = k1 := by rw [hk1] at hw1; exact (Option.some.inj hw1).symm
  subst hw1'
  refine (safe_toQ d.cm).2 ?_
  refine (safe_toQ d.cm).2 ?_
  simp only [toQv_scalar_item]
  have hext : (v1 - v0).toInt = q1 - q0 := by rw [pre_sub (by omega) (by omega), hv0', hv1']
  have hpr : (w1 - w0).toInt = w1.toInt - w0.toInt := pre_sub hk0' hk01
  have hbm : d.BM ≤ 128 := bm_le hd
  have hBM : (IntV.ofNat d.BM : I32).toInt = d.BM := pre_toInt_i32 (by omega)
  have hblk : (IntV.ofNat (mOf d i) * IntV.ofNat d.BM : I32).toInt = ((mOf d i * d.BM : Nat) : Int) :=
    IntV.mul_nat t2 hBM (by
      have : mOf d i * d.BM ≤ 2 ^ 16 * 128 := Nat.mul_le_mul (by omega) hbm
      omega)
  have hcb : (castQ d.cm (IntV.ofNat (mOf d i) * IntV.ofNat d.BM : I32)).toInt = ((mOf d i * d.BM : Nat) : Int) := by
    rw [castQ_toInt, hblk]
  have hsfit : IntV.SFits (qoW d.cm) ((q1 - q0) + (w1.toInt - w0.toInt)) := by
    have h31 := lt_pow_qoW d.cm (n := 2 ^ 18) (by norm_num)
    have h31' : ((2 ^ 18 : Nat) : Int) < 2 ^ (qoW d.cm - 1) := by exact_mod_cast h31
    have hp : (0 : Int) < 2 ^ (qoW d.cm - 1) := by positivity
    push_cast at h31'
    exact ⟨by omega, by omega⟩
  have hlen : (castQ d.cm (w1 - w0) + (v1 - v0)).toInt =
      (preOf d init (bOf d i) : Int) + extOf d init (bOf d i) := by
    rw [IntV.toInt_add (x := w1.toInt - w0.toInt) (y := q1 - q0) (by rw [castQ_toInt, hpr]) hext
      (by simpa [add_comm] using hsfit)]
    omega
  by_cases hlt : mOf d i * d.BM < extOf d init (bOf d i)
  · have hF : IntV.le (v1 - v0) (castQ d.cm (IntV.ofNat (mOf d i) * IntV.ofNat d.BM : I32)) = false := by
      rw [IntV.le_eq, hext, hcb]; simp; omega
    simp only [hF, Bool.false_eq_true, ↓reduceIte]
    refine hrest _ ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ hlt
    · exact t0
    · exact t1
    · exact t2
    · show (r.item).toInt = _
      rw [hr, t1]
      simp only [kvh]
      have : ((kvg d : Nat) : Int) = (IntV.ofNat (kvg d) : I32).toInt := (pre_toInt_i32 hkvglt).symm
      rw [← this]; push_cast; rfl
    · rw [hv0']; omega
    · rw [hext]; omega
    · show w0.toInt = _; omega
    · rw [hpr]; omega
    · exact hlen
  · have hT : IntV.le (v1 - v0) (castQ d.cm (IntV.ofNat (mOf d i) * IntV.ofNat d.BM : I32)) = true := by
      rw [IntV.le_eq, hext, hcb]; simp; omega
    simp only [hT, ↓reduceIte]
    refine safe_ret ?_
    intro r e hr he hlt'
    omega
end Claims.ExtendAttn
