import Claim
import TVA.Proof
import Proof.Tail

/-!
# Lines 91–98: the program's head, chunk and sequence

`pre_pad`, `pre_var`: `preProg` is safe, and hands the rest of the body the head `h`, the chunk
`it`, the sequence start `s0` and the length `len`, whether the batch is padded (line 98) or
packed (lines 94–96, from `chunk_indices` and `cu_seqlens`).
-/

namespace Claims.RecomputeWU

open TVA

variable {E : Env} {R W : Set Int} {Q : Unit → Mem → Prop} {m : Mem}

/-- A program id below `2^31`, converted to `int64`, keeps its value. -/
theorem cast_pid (c : Nat) (hc : c < 2 ^ 31) :
    (IntV.cast (IntV.ofNat c : I32) : I64).toInt = c := by
  rw [IntV.toInt_cast_of_sfits _ (by norm_num) (by
    rw [IntV.I32.toInt_ofNat hc]; exact IntV.I64.sfits_of_nonneg (by positivity) (by
      have : (c : Int) < 2 ^ 31 := by exact_mod_cast hc
      linarith))]
  exact IntV.I32.toInt_ofNat hc

/-- **The padded head** (line 98). -/
theorem pre_pad {sD : DType} (cu ci : Ptr sD) (T : I32) (HV : Nat) (ctx : Ctx)
    {rest : I64 → I64 → I64 → I64 → Prog Unit} (p0 bh Tn : Nat)
    (hp0 : ctx.pid 0 = p0) (hbh : ctx.pid 1 = bh) (hp0lt : p0 < 2 ^ 31) (hbhlt : bh < 2 ^ 31)
    (hHV0 : 0 < HV) (hHV : HV < 2 ^ 62) (hT : T.toInt = Tn) (hTn : Tn < 2 ^ 31)
    (hb : (bh / HV) * Tn + Tn < 2 ^ 62)
    (hrest : ∀ i_h i_t bos T_n : I64, i_h.toInt = (bh % HV : Nat) → i_t.toInt = p0 →
      bos.toInt = (((bh / HV) * Tn : Nat) : Int) → T_n.toInt = Tn →
      Safe E R W Q m (rest i_h i_t bos T_n)) :
    Safe E R W Q m (preProg cu ci T HV false ctx rest) := by
  unfold preProg
  have hpid0 : tl.program_id ctx 0 = IntV.ofNat p0 := by simp [tl.program_id, hp0]
  have hpid1 : tl.program_id ctx 1 = IntV.ofNat bh := by simp [tl.program_id, hbh]
  rw [hpid0, hpid1]
  have e0 := cast_pid p0 hp0lt
  have e1 := cast_pid bh hbhlt
  refine safe_idiv1 (x := (IntV.cast (IntV.ofNat bh : I32) : I64)) (y := (IntV.ofNat HV : I64))
    (by rw [e1]; positivity) (by rw [IntV.I64.toInt_ofNat (by omega)]; exact_mod_cast hHV0) ?_
  intro i_b hib
  rw [e1, IntV.I64.toInt_ofNat (by omega)] at hib
  refine safe_irem1 (x := (IntV.cast (IntV.ofNat bh : I32) : I64)) (y := (IntV.ofNat HV : I64))
    (by rw [e1]; positivity) (by rw [IntV.I64.toInt_ofNat (by omega)]; exact_mod_cast hHV0) ?_
  intro i_h hih
  rw [e1, IntV.I64.toInt_ofNat (by omega)] at hih
  have hTc : (IntV.cast T : I64).toInt = Tn := by
    rw [IntV.toInt_cast_of_sfits _ (by norm_num) (by
      rw [hT]; exact IntV.I64.sfits_of_nonneg (by positivity) (by
        have : (Tn : Int) < 2 ^ 31 := by exact_mod_cast hTn
        linarith))]
    exact hT
  have hib' : i_b.item.toInt = ((bh / HV : Nat) : Int) := by rw [hib]; rfl
  have hbos : (i_b.item * (IntV.cast T : I64)).toInt = (((bh / HV) * Tn : Nat) : Int) :=
    IntV.mul_nat hib' hTc (by norm_num; omega)
  simp only []
  exact hrest _ _ _ _ (by rw [hih]; rfl) (by rw [e0]) hbos hTc


/-- **A scalar table entry loaded and converted to `int64`** (`tl.load1` then
`tl.convert sD .i64`): the rest sees the entry's value. -/
theorem safe_load_conv64 {d : Inst} {init : Mem} (hag : Ag d init m) {sD : DType}
    (hsD : sD = .i32 ∨ sD = .i64) (b : Buf sD) (hbR : b.bytes ⊆ Rall d) (hv : b.Valid init)
    {j : Nat} (hj : j < b.len) {x : Int} (hx : idxAt b init j = some x) {ptr : Ptr sD}
    (hp : ptr = b.ptr (j : Int)) {k : Tensor [] I64 → Prog Unit}
    (hk : ∀ r : Tensor [] I64, r.item.toInt = x → Safe E (Rall d) W Q m (k r)) :
    Safe E (Rall d) W Q m
      (tl.load1 ptr >>= fun v => tl.convert sD .i64 (Tensor.scalar v) >>= k) := by
  subst hp
  have hspan : Mem.span sD (b.ptr (j : Int)).addr ⊆ Rall d :=
    fun y hy => hbR (b.span_sub hj hy)
  unfold idxAt at hx
  rw [← read_ag hag b hbR hj] at hx
  rcases hsD with rfl | rfl
  · refine safe_load1_buf b hj (valid_ag hag hbR hv) hspan fun v hrd => ?_
    rw [hrd] at hx
    have hxv : v.toInt = x := Option.some.inj hx
    refine safe_convert (d := .i32) (d' := .i64) (fun _ => ⟨IntV.ofInt v.toInt, rfl⟩) ?_
    intro r hr
    refine hk r ?_
    have h1 : r.item = IntV.ofInt v.toInt := hr ()
    rw [h1, IntV.toInt_ofInt (by norm_num) (by
      have := IntV.sfits_toInt v
      simp only [IntV.SFits] at this ⊢; omega), hxv]
  · refine safe_load1_buf b hj (valid_ag hag hbR hv) hspan fun v hrd => ?_
    rw [hrd] at hx
    have hxv : v.toInt = x := Option.some.inj hx
    refine safe_convert (d := .i64) (d' := .i64) (fun _ => ⟨IntV.ofInt v.toInt, rfl⟩) ?_
    intro r hr
    refine hk r ?_
    have h1 : r.item = IntV.ofInt v.toInt := hr ()
    rw [h1, IntV.toInt_ofInt (by norm_num) (by
      have := IntV.sfits_toInt v
      simp only [IntV.SFits] at this ⊢; omega), hxv]

/-- The same converted to `int32`, for an entry that fits. -/
theorem safe_load_conv32 {d : Inst} {init : Mem} (hag : Ag d init m) {sD : DType}
    (hsD : sD = .i32 ∨ sD = .i64) (b : Buf sD) (hbR : b.bytes ⊆ Rall d) (hv : b.Valid init)
    {j : Nat} (hj : j < b.len) {x : Int} (hx : idxAt b init j = some x)
    (hfit : IntV.SFits 32 x) {ptr : Ptr sD}
    (hp : ptr = b.ptr (j : Int)) {k : Tensor [] I32 → Prog Unit}
    (hk : ∀ r : Tensor [] I32, r.item.toInt = x → Safe E (Rall d) W Q m (k r)) :
    Safe E (Rall d) W Q m
      (tl.load1 ptr >>= fun v => tl.convert sD .i32 (Tensor.scalar v) >>= k) := by
  subst hp
  have hspan : Mem.span sD (b.ptr (j : Int)).addr ⊆ Rall d :=
    fun y hy => hbR (b.span_sub hj hy)
  unfold idxAt at hx
  rw [← read_ag hag b hbR hj] at hx
  rcases hsD with rfl | rfl
  · refine safe_load1_buf b hj (valid_ag hag hbR hv) hspan fun v hrd => ?_
    rw [hrd] at hx
    have hxv : v.toInt = x := Option.some.inj hx
    refine safe_convert (d := .i32) (d' := .i32) (fun _ => ⟨IntV.ofInt v.toInt, rfl⟩) ?_
    intro r hr
    refine hk r ?_
    have h1 : r.item = IntV.ofInt v.toInt := hr ()
    rw [h1, IntV.toInt_ofInt (by norm_num) (by rw [hxv]; exact hfit), hxv]
  · refine safe_load1_buf b hj (valid_ag hag hbR hv) hspan fun v hrd => ?_
    rw [hrd] at hx
    have hxv : v.toInt = x := Option.some.inj hx
    refine safe_convert (d := .i64) (d' := .i32) (fun _ => ⟨IntV.ofInt v.toInt, rfl⟩) ?_
    intro r hr
    refine hk r ?_
    have h1 : r.item = IntV.ofInt v.toInt := hr ()
    rw [h1, IntV.toInt_ofInt (by norm_num) (by rw [hxv]; exact hfit), hxv]


/-- **The packed head** (lines 94–96): the sequence and chunk come from the tables. -/
theorem pre_var {d : Inst} {init : Mem} (hag : Ag d init m) {sD : DType}
    (hsD : sD = .i32 ∨ sD = .i64) (cu ci : Buf sD) (hcuR : cu.bytes ⊆ Rall d)
    (hciR : ci.bytes ⊆ Rall d) (hcuv : cu.Valid init) (hciv : ci.Valid init) (T : I32) (HV : Nat)
    (ctx : Ctx) {rest : I64 → I64 → I64 → I64 → Prog Unit} (p0 bh n chunk c0 c1 : Nat)
    (hp0 : ctx.pid 0 = p0) (hbh : ctx.pid 1 = bh) (hp0lt : p0 < 2 ^ 31) (hbhlt : bh < 2 ^ 31)
    (hHV0 : 0 < HV) (hHV : HV < 2 ^ 62)
    (hl1 : 2 * p0 + 1 < ci.len) (hl3 : n + 1 < cu.len)
    (hx1 : idxAt ci init (2 * p0) = some (n : Int))
    (hx2 : idxAt ci init (2 * p0 + 1) = some (chunk : Int))
    (hx3 : idxAt cu init n = some (c0 : Int)) (hx4 : idxAt cu init (n + 1) = some (c1 : Int))
    (hn : n < 2 ^ 31) (hc1 : c1 < 2 ^ 62) (hc01 : c0 ≤ c1)
    (hrest : ∀ i_h i_t bos T_n : I64, i_h.toInt = (bh % HV : Nat) → i_t.toInt = chunk →
      bos.toInt = c0 → T_n.toInt = ((c1 - c0 : Nat) : Int) →
      Safe E (Rall d) W Q m (rest i_h i_t bos T_n)) :
    Safe E (Rall d) W Q m (preProg (cu.ptr 0) (ci.ptr 0) T HV true ctx rest) := by
  unfold preProg
  have hpid0 : tl.program_id ctx 0 = IntV.ofNat p0 := by simp [tl.program_id, hp0]
  have hpid1 : tl.program_id ctx 1 = IntV.ofNat bh := by simp [tl.program_id, hbh]
  rw [hpid0, hpid1]
  have e0 := cast_pid p0 (by omega)
  have e1 := cast_pid bh hbhlt
  refine safe_idiv1 (x := (IntV.cast (IntV.ofNat bh : I32) : I64)) (y := (IntV.ofNat HV : I64))
    (by rw [e1]; positivity) (by rw [IntV.I64.toInt_ofNat (by omega)]; exact_mod_cast hHV0) ?_
  intro i_b hib
  refine safe_irem1 (x := (IntV.cast (IntV.ofNat bh : I32) : I64)) (y := (IntV.ofNat HV : I64))
    (by rw [e1]; positivity) (by rw [IntV.I64.toInt_ofNat (by omega)]; exact_mod_cast hHV0) ?_
  intro i_h hih
  rw [e1, IntV.I64.toInt_ofNat (by omega)] at hih
  simp only [ite_true]
  simp only [bind_assoc, pure_bind]
  have hk0 : ((IntV.cast (IntV.ofNat p0 : I32) : I64) * (IntV.ofNat 2 : I64)).toInt =
      ((2 * p0 : Nat) : Int) := by
    have := IntV.mul_ofNat (n := 2) e0 (by norm_num; omega)
    rw [this]; push_cast; ring
  have hone : (IntV.ofNat 1 : I32).toInt = ((1 : Nat) : Int) := IntV.I32.toInt_ofNat (by norm_num)
  refine safe_load_conv32 hag hsD ci hciR hciv (j := 2 * p0) (by omega) hx1
    (IntV.I32.sfits_natCast (by omega)) (ptr_zero_add ci hk0 |>.trans (by push_cast; rfl)) ?_
  intro i_n hin
  refine safe_load_conv64 hag hsD ci hciR hciv (j := 2 * p0 + 1) hl1 hx2
    ((ptr_zero_add2 ci hk0 hone).trans (by push_cast; rfl)) ?_
  intro i_t hit
  refine safe_load_conv64 hag hsD cu hcuR hcuv (j := n) (by omega) hx3
    ((ptr_zero_add cu hin).trans rfl) ?_
  intro bos hbos
  refine safe_load_conv64 hag hsD cu hcuR hcuv (j := n + 1) hl3 hx4
    ((ptr_zero_add2 cu hin hone).trans (by push_cast; rfl)) ?_
  intro eos heos
  have hT : (eos.item - bos.item).toInt = ((c1 - c0 : Nat) : Int) :=
    IntV.sub_nat (a := eos.item) (b := bos.item) (x := c1) (y := c0) heos hbos hc01
      (by norm_num; omega)
  exact hrest _ _ _ _ (by rw [hih]; rfl) hit hbos hT

end Claims.RecomputeWU
