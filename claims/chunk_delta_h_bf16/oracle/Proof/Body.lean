import Proof.Loop
import Proof.Foot
import Proof.GlueAux
import Proof.Trip

/-!
# `chunk_delta_h_bf16`: the program after its sequence is chosen

`body_rest` (source lines 100–175 and 318–349): the state blocks, the offset pointers, the lanes,
the initial state, the main loop and the final state. Given the program's sequence, head and
block, as `int64` values, and a loop (`runLoop`) that runs `nch` trips, it leaves `Q`.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- **The rule of the main loop**, as the body needs it: a loop of `nch` trips, trip `t` calling the
step with `i_t_int64 = t` and the offset of the chunk's last token. -/
def RunLoopSpec {σ : Type} (d : Inst) (E : Env) (R : Set Int) (n hv : Nat)
    (runLoop : (I64 → I64 → σ → Prog σ) → σ → Prog σ) : Prop :=
  ∀ {α : Type} (I : Nat → σ → Mem → Prop) (Wt : Nat → Set Int) (Wk : Set Int) (Qf : α → Mem → Prop)
    (m : Mem) (step : I64 → I64 → σ → Prog σ) (st0 : σ) (k : σ → Prog α),
    (∀ t u, t < u → u < nch d n → Disjoint (Wt t) (Wt u)) →
    Disjoint (iterW Wt (nch d n)) Wk → I 0 st0 m →
    (∀ t, t < nch d n → ∀ iT goff : I64, iT.toInt = (t : Int) →
      (d.useG = true → goff.toInt = (gIdx d (rowOf d n) (startOf d n + BT * t + cl d n t - 1) hv : Int)) →
      ∀ st m₁, I t st m₁ → Safe E R (Wt t) (fun st' m₂ => I (t + 1) st' m₂) m₁ (step iT goff st)) →
    (∀ st m₁, I (nch d n) st m₁ → Safe E R Wk Qf m₁ (k st)) →
    Safe E R (iterW Wt (nch d n) ∪ Wk) Qf m (runLoop step st0 >>= k)

/-- The `int64` loop satisfies the rule. -/
theorem runLoop_var {σ : Type} {d : Inst} (hd : d.ok) {E : Env} {R : Set Int} {n hv : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) {bos i_h T NT : I64} (hbos : bos.toInt = (bosOf d n : Int))
    (hih : i_h.toInt = (hv : Int)) (hT : T.toInt = (lenOf d n : Int)) (hNT : NT.toInt = (nch d n : Int)) :
    RunLoopSpec (σ := σ) d E R n hv (loopVar d.HV BT NT T bos i_h) :=
  fun _I _Wt _Wk _Qf _m step _st0 _k hWt hdis h0 hstep hk =>
    safe_loopVar hd hn hhv hbos hih hT hNT step hWt hdis h0 hstep hk

/-- The `int32` loop satisfies the rule. -/
theorem runLoop_fix {σ : Type} {d : Inst} (hd : d.ok) {E : Env} {R : Set Int} {n hv : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) (hfix : d.varlen = false) {bos i_h : I64} {T NT : I32}
    (hbos : bos.toInt = (bosOf d n : Int)) (hih : i_h.toInt = (hv : Int)) (hT : T.toInt = (d.T : Int))
    (hNT : NT.toInt = (nch d n : Int)) :
    RunLoopSpec (σ := σ) d E R n hv (loopFix d.HV BT NT T bos i_h) :=
  fun _I _Wt _Wk _Qf _m step _st0 _k hWt hdis h0 hstep hk =>
    safe_loopFix hd hn hhv hfix hbos hih hT hNT step hWt hdis h0 hstep hk

/-- A valid `bfloat16` buffer has at most `2^63` elements. -/
theorem len_le_bf16 {b : Buf .bf16} {m : Mem} (hv : b.Valid m) : b.len ≤ 2 ^ 63 := by
  have := hv.2.1
  have h2 : DType.bf16.bytes = 2 := rfl
  rw [h2] at this
  have h3 : (2 : Int) ^ 64 = 2 * 2 ^ 63 := by norm_num
  have := hv.1
  omega

/-- A valid `float32` buffer has at most `2^63` elements. -/
theorem len_le_f32 {b : Buf .f32} {m : Mem} (hv : b.Valid m) : b.len ≤ 2 ^ 63 := by
  have := hv.2.1
  have h2 : DType.f32.bytes = 4 := rfl
  rw [h2] at this
  have h3 : (2 : Int) ^ 64 = 4 * 2 ^ 62 := by norm_num
  have := hv.1
  omega

/-- The `h0` of a program is inside its tensor. -/
theorem hlen_h0 {d : Inst} (hd : d.ok) (hU : d.useH0 = true) {n hv : Nat} (hn : n < d.N)
    (hhv : hv < d.HV) : (n * d.HV + hv + 1) * d.K * d.V ≤ d.h0.len := by
  have h1 := hd.h0_len hU
  have h2 : n * d.HV + hv + 1 ≤ d.N * d.HV := by
    have : (n + 1) * d.HV ≤ d.N * d.HV := Nat.mul_le_mul_right _ hn
    have : (n + 1) * d.HV = n * d.HV + d.HV := by ring
    omega
  exact le_trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ h2)) h1

/-- The `int64` offset `i_nh · K · V` of the `h0`/`ht` pointers. -/
theorem nv_ptr {d : Inst} (hd : d.ok) {n hv : Nat} (_hn : n < d.N) (_hhv : hv < d.HV) {i_nh : I64}
    (hinh : i_nh.toInt = ((n * d.HV + hv : Nat) : Int)) {len : Nat}
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ len) (hl : len ≤ 2 ^ 63) :
    (i_nh * (IntV.ofNat d.K : I64) * (IntV.ofNat d.V : I64)).toInt = (((n * d.HV + hv) * d.K * d.V : Nat) : Int) := by
  have hKp := hd.K_pos; have hVp := hd.V_pos
  have hKl := hd.K_le; have hVl := hd.V_lt
  have h1 : (n * d.HV + hv) * d.K * d.V < 2 ^ 63 := by
    have : (n * d.HV + hv) * d.K * d.V < (n * d.HV + hv + 1) * d.K * d.V := by
      apply Nat.mul_lt_mul_of_pos_right; apply Nat.mul_lt_mul_of_pos_right; omega; exact hKp; exact hVp
    omega
  have h2 : (n * d.HV + hv) * d.K ≤ (n * d.HV + hv) * d.K * d.V := Nat.le_mul_of_pos_right _ hVp
  have hK' : (IntV.ofNat d.K : I64).toInt = (d.K : Int) := IntV.I64.toInt_ofNat (by omega)
  have hV' : (IntV.ofNat d.V : I64).toInt = (d.V : Int) := IntV.I64.toInt_ofNat (by omega)
  have e1 := IntV.mul_nat hinh hK' (by show (n * d.HV + hv) * d.K < 2 ^ 63; omega)
  exact IntV.mul_nat e1 hV' h1

/-- A pointer offset by an `int64` of known value. -/
theorem ptr_off {dt : DType} (b : Buf dt) {base : Ptr dt} (hb : base = b.ptr 0) {x : I64} {e : Nat}
    (hx : x.toInt = (e : Int)) : base + x = b.ptr (e : Int) := by
  show base.offset x.toInt = _
  rw [hb, hx, Buf.ptr_offset]; simp

/-- `(a · c + b) · k` in `int64`, when the result is below `2^63`. -/
theorem lin2 {a b : I64} {A B c k : Nat} (ha : a.toInt = (A : Int)) (hb : b.toInt = (B : Int))
    (hc : c < 2 ^ 63) (hk : k < 2 ^ 63) (hlt : (A * c + B) * k < 2 ^ 63) (hk1 : 1 ≤ k) :
    ((a * (IntV.ofNat c : I64) + b) * (IntV.ofNat k : I64)).toInt = (((A * c + B) * k : Nat) : Int) := by
  have hx : A * c + B ≤ (A * c + B) * k := Nat.le_mul_of_pos_right _ hk1
  have hAB : A * c + B < 2 ^ 63 := lt_of_le_of_lt hx hlt
  have h1 : (a * (IntV.ofNat c : I64)).toInt = ((A * c : Nat) : Int) :=
    IntV.mul_nat ha (IntV.I64.toInt_ofNat hc) (by show A * c < 2 ^ 63; omega)
  have h2 : (a * (IntV.ofNat c : I64) + b).toInt = ((A * c + B : Nat) : Int) :=
    IntV.add_nat h1 hb (by show A * c + B < 2 ^ 63; omega)
  exact IntV.mul_nat h2 (IntV.I64.toInt_ofNat hk) hlt

/-- `(a · c + b) · k · l` in `int64`, when the result is below `2^63`. -/
theorem lin3 {a b : I64} {A B c k l : Nat} (ha : a.toInt = (A : Int)) (hb : b.toInt = (B : Int))
    (hc : c < 2 ^ 63) (hk : k < 2 ^ 63) (hl : l < 2 ^ 63) (hlt : (A * c + B) * k * l < 2 ^ 63)
    (hk1 : 1 ≤ k) (hl1 : 1 ≤ l) :
    ((a * (IntV.ofNat c : I64) + b) * (IntV.ofNat k : I64) * (IntV.ofNat l : I64)).toInt =
      (((A * c + B) * k * l : Nat) : Int) := by
  have h12 : (A * c + B) * k ≤ (A * c + B) * k * l := Nat.le_mul_of_pos_right _ hl1
  have := lin2 ha hb hc hk (lt_of_le_of_lt h12 hlt) hk1
  exact IntV.mul_nat this (IntV.I64.toInt_ofNat hl) hlt

/-- **The lanes of the prologue** (source lines 131–140): `o_v = i_v · 32 + arange(0, 32)`, `m_v`,
and the state rows `o_k_b = 64(b-1) + arange(0, 64)`, `m_k_b`. -/
theorem lanes_of {d : Inst} (hd : d.ok) {jv : Nat} (hjv : jv < NV d) {i_v : I64}
    (hiv : i_v.toInt = (jv : Int)) {ot : Tensor [BV - 0] I64}
    (hot : ∀ i : Fin BV, (ot.get1 i).toInt = (i : Nat)) :
    Lanes d jv (i_v * (IntV.ofNat BV : I64) + ot) ((i_v * (IntV.ofNat BV : I64) + ot).ltS (IntV.ofNat d.V))
      (tl.arange 0 64) ((IntV.ofNat 64 : I32) + tl.arange 0 64) ((IntV.ofNat 128 : I32) + tl.arange 0 64)
      ((IntV.ofNat 192 : I32) + tl.arange 0 64) ((tl.arange 0 64).ltS (IntV.ofNat d.K))
      (((IntV.ofNat 64 : I32) + tl.arange 0 64).ltS (IntV.ofNat d.K))
      (((IntV.ofNat 128 : I32) + tl.arange 0 64).ltS (IntV.ofNat d.K))
      (((IntV.ofNat 192 : I32) + tl.arange 0 64).ltS (IntV.ofNat d.K)) := by
  have hVl := hd.V_lt
  have hKl := hd.K_le
  have hjb : jv < 2 ^ 16 := by have := hjv; simp only [NV, BV] at this; omega
  have hBV : (IntV.ofNat BV : I64).toInt = (BV : Int) := IntV.I64.toInt_ofNat (by unfold BV; norm_num)
  have hVi : (IntV.ofNat d.V : I64).toInt = (d.V : Int) := IntV.I64.toInt_ofNat (by omega)
  have hKi : (IntV.ofNat d.K : I32).toInt = (d.K : Int) := IntV.I32.toInt_ofNat (by omega)
  have hov : ∀ jj : Fin BV, ((i_v * (IntV.ofNat BV : I64) + ot).get1 jj).toInt = ((BV * jv + jj : Nat) : Int) := by
    intro jj
    have hg : (i_v * (IntV.ofNat BV : I64) + ot).get1 jj = i_v * (IntV.ofNat BV : I64) + ot.get1 jj := by simp
    rw [hg]
    have h1 : (i_v * (IntV.ofNat BV : I64)).toInt = ((jv * BV : Nat) : Int) :=
      IntV.mul_nat hiv hBV (by unfold BV; omega)
    have := IntV.add_nat h1 (hot jj) (by have := jj.isLt; unfold BV at *; omega)
    rw [this]; congr 1; ring
  have hk : ∀ (off : Nat), off ≤ 192 → ∀ a : Fin 64,
      (((IntV.ofNat off : I32) + tl.arange 0 64).get1 a).toInt = ((off + a : Nat) : Int) := by
    intro off hoff a
    have hg : ((IntV.ofNat off : I32) + tl.arange 0 64).get1 a = (IntV.ofNat off : I32) + (tl.arange 0 64).get1 a := by simp
    rw [hg]
    exact IntV.add_nat (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_tl_arange (n := 64) (by norm_num) a)
      (by have := a.isLt; omega)
  have hm : ∀ (off : Nat) (x : Tensor [64] I32), (∀ a : Fin 64, (x.get1 a).toInt = ((off + a : Nat) : Int)) →
      ∀ a : Fin 64, (x.ltS (IntV.ofNat d.K)).get1 a = decide (off + a < d.K) := by
    intro off x hx a
    show (x.get1 a).lt _ = _
    rw [IntV.lt_eq, hx a, hKi]; simp only [Nat.cast_lt]
  have hk0 : ∀ a : Fin 64, ((tl.arange 0 64).get1 a).toInt = ((0 + a : Nat) : Int) := fun a =>
    (IntV.I32.toInt_tl_arange (n := 64) (by norm_num) a).trans (by simp)
  refine ⟨hov, fun jj => ?_, hk0, hk 64 (by norm_num), hk 128 (by norm_num), hk 192 (by norm_num),
    hm 0 _ hk0, hm 64 _ (hk 64 (by norm_num)), hm 128 _ (hk 128 (by norm_num)), hm 192 _ (hk 192 (by norm_num))⟩
  show ((i_v * (IntV.ofNat BV : I64) + ot).get1 jj).lt (IntV.ofNat d.V) = _
  rw [IntV.lt_eq, hov jj, hVi]; simp only [Nat.cast_lt]

/-- Lane `i` of `tl.convert(tl.arange(0, BV), int64)` is `i`. -/
theorem ot_laneV {P : Platform} {ot : Tensor [BV - 0] I64}
    (h : ∀ i, ConvRel P .i32 .i64 ((tl.arange 0 BV).get i) (ot.get i)) (i : Fin BV) :
    (ot.get1 i).toInt = (i : Nat) := by
  have h1 := ConvRel.toInt_i32_i64 (h (i, ()))
  have := IntV.I32.toInt_tl_arange (n := BV) (by unfold BV; norm_num) i
  exact h1.trans this

/-- **The offset pointers of the program** (source lines 118–123) when the sequence has a chunk. -/
theorem ptr_facts {d : Inst} (hd : d.ok) {n hv : Nat} (hn : n < d.N) (hhv : hv < d.HV)
    (hnch : 0 < nch d n) {bos boh i_h : I64} {kr : Tensor [] I64}
    (hbos : bos.toInt = (bosOf d n : Int)) (hboh : boh.toInt = (cbase d n : Int))
    (hih : i_h.toInt = (hv : Int)) (hkr : kr.item.toInt = (kHead d hv : Int))
    (hkl : d.k.len ≤ 2 ^ 63) (hwl : d.w.len ≤ 2 ^ 63) (hul : d.u.len ≤ 2 ^ 63) (hvl : d.vn.len ≤ 2 ^ 63)
    (hhl : d.h.len ≤ 2 ^ 63) :
    (d.k.ptr 0 + (bos * (IntV.ofNat d.H : I64) + kr.item) * (IntV.ofNat d.K : I64) =
        d.k.ptr (((bosOf d n * d.H + kHead d hv) * d.K : Nat) : Int)) ∧
    (d.w.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.K : I64) =
        d.w.ptr (((bosOf d n * d.HV + hv) * d.K : Nat) : Int)) ∧
    (d.u.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64) =
        d.u.ptr (((bosOf d n * d.HV + hv) * d.V : Nat) : Int)) ∧
    (d.vn.ptr 0 + (bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64) =
        d.vn.ptr (((bosOf d n * d.HV + hv) * d.V : Nat) : Int)) ∧
    (d.h.ptr 0 + (boh * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.K : I64) * (IntV.ofNat d.V : I64) =
        d.h.ptr (((cbase d n * d.HV + hv) * d.K * d.V : Nat) : Int)) := by
  have hHle : d.H ≤ d.HV := Nat.le_of_dvd hd.HV_pos hd.H_dvd
  have hKp := hd.K_pos; have hVp := hd.V_pos; have hHp := hd.H_pos; have hHVp := hd.HV_pos
  have hKl := hd.K_le; have hVl := hd.V_lt; have hHVl := hd.HV_lt
  have hlen1 : 0 < lenOf d n := by simp only [nch, BT] at hnch; omega
  have hbl := bos_le hd hn
  have hkh := kHead_lt hd hhv
  have hbos' : bosOf d n + 1 ≤ bosOf d n + lenOf d n := by omega
  refine ⟨ptr_off d.k rfl (lin2 hbos hkr (by omega) (by omega) ?_ hKp), ptr_off d.w rfl (lin2 hbos hih
    (by omega) (by omega) ?_ hKp), ptr_off d.u rfl (lin2 hbos hih (by omega) (by omega) ?_ hVp),
    ptr_off d.vn rfl (lin2 hbos hih (by omega) (by omega) ?_ hVp),
    ptr_off d.h rfl (lin3 hboh hih (by omega) (by omega) (by omega) ?_ hKp hVp)⟩
  · have h1 : (bosOf d n * d.H + kHead d hv + 1) * d.K ≤ (bosOf d n + lenOf d n) * d.H * d.K := by
      apply Nat.mul_le_mul_right
      nlinarith
    have h2 := len_k hd hn
    nlinarith
  · have h1 : (bosOf d n * d.HV + hv + 1) * d.K ≤ (bosOf d n + lenOf d n) * d.HV * d.K := by
      apply Nat.mul_le_mul_right
      nlinarith
    have h2 := len_w hd hn
    nlinarith
  · have h1 : (bosOf d n * d.HV + hv + 1) * d.V ≤ (bosOf d n + lenOf d n) * d.HV * d.V := by
      apply Nat.mul_le_mul_right
      nlinarith
    have h2 := len_u hd hn
    nlinarith
  · have h1 : (bosOf d n * d.HV + hv + 1) * d.V ≤ (bosOf d n + lenOf d n) * d.HV * d.V := by
      apply Nat.mul_le_mul_right
      nlinarith
    have h2 := len_vn hd hn
    nlinarith
  · have hc := chunk_lt hd hn hnch
    have h1 : (cbase d n * d.HV + hv + 1) * d.K * d.V ≤ d.B * NTt d * d.HV * d.K * d.V := by
      apply Nat.mul_le_mul_right
      apply Nat.mul_le_mul_right
      nlinarith
    have h2 := hd.h_len
    nlinarith

/-- `cu_seqlens` entries are nonnegative up to `N`. -/
theorem entry_nonneg {d : Inst} (hd : d.ok) (hv : d.varlen = true) {j : Nat} (hj : j ≤ d.N) :
    0 ≤ entryInt d.cuD (d.cuVal j) := by
  induction j with
  | zero => rw [hd.var_cu0 hv]
  | succ j ih => exact le_trans (ih (by omega)) (hd.var_mono hv j (by omega))

/-- An entry of `cu_seqlens` as a natural number: `cuAt` is its value. -/
theorem entry_eq {d : Inst} (hd : d.ok) (hv : d.varlen = true) {j : Nat} (hj : j ≤ d.N) :
    entryInt d.cuD (d.cuVal j) = (cuAt d j : Int) := by
  unfold cuAt; rw [Int.toNat_of_nonneg (entry_nonneg hd hv hj)]

/-- Converting an entry of `cu_seqlens` to `int64` keeps its value. -/
theorem conv_cu {P : Platform} (D : DType) (hD : D = .i32 ∨ D = .i64) :
    (∀ x : D.Val, ∃ r, ConvRel P D .i64 x r) ∧
      ∀ (x : D.Val) (r : I64), ConvRel P D .i64 x r → r.toInt = entryInt D x := by
  rcases hD with rfl | rfl
  · exact ⟨fun x => ConvRel.exists_i32_i64 x, fun x r h => ConvRel.toInt_i32_i64 h⟩
  · exact ⟨fun x => ConvRel.exists_i64_i64 x, fun x r h => by
      have : r = IntV.cast x := ConvRel.iff_cast_i64_i64.1 h
      rw [this]; exact IntV.toInt_cast x (by norm_num)⟩

/-- The first chunk of sequence `n` of a variable-length launch is below `cuAt n + n`. -/
theorem cbase_le {d : Inst} (hd : d.ok) (hv : d.varlen = true) {n : Nat} (hn : n ≤ d.N) :
    cbase d n ≤ cuAt d n + n := by
  induction n with
  | zero => simp [cbase, hv]
  | succ n ih =>
    rw [cbase_succ]
    have h1 := ih (by omega)
    have h2 := cuAt_mono hd hv (m := n) (n := n + 1) (by omega) hn
    have h3 : lenOf d n = cuAt d (n + 1) - cuAt d n := by simp [lenOf, hv]
    have h4 : nch d n ≤ lenOf d n + 1 := by simp only [nch, BT]; omega
    omega

end Claims.ChunkDeltaH
