import Proof.TripSpec
import Proof.FootAux1

/-!
# `chunk_delta_h_bf16`: segments of a loop trip

Each lemma is `Safe` of a fragment of `chunk_step` (or `chunk_step_vf`) followed by an arbitrary
rest, in the style of `safe_hstore`; the continuation receives the raw facts the fragment
established (which cell a lane was read from or written to, the platform's relations), and the
numeric glue turns them into error bounds.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA TVA.Tile2

/-- A sequence's tokens, as flat row-major token indices, are below `2^35`. -/
theorem bos_bound {d : Inst} (hd : d.ok) {n : Nat} (hn : n < d.N) :
    bosOf d n + lenOf d n ≤ 2 ^ 35 := by
  have h1 := row_lt hd hn
  have h2 := start_add_len_le hd hn
  have hB := hd.N_batch.2
  have hT := hd.T_ctx
  have h3 : (rowOf d n + 1) * d.T ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul (by omega) hT
  have h4 : (rowOf d n + 1) * d.T = rowOf d n * d.T + d.T := by ring
  unfold bosOf
  omega

/-! ## The `other = 0.0` constant -/

/-- A `bfloat16` zero: finite, value `0`. -/
def ZB (x : BF16) : Prop := x.isFinite = true ∧ x.toReal = 0

/-- The conversion of the `float32` constant `+0` to `bfloat16` (the `other=0.0` of a masked load)
gives a zero. -/
theorem zb_of_conv {P : Platform} {x : BF16} (h : ConvRel P .f32 .bf16 zf32 x) : ZB x := by
  rw [ConvRel.f32_bf16_iff] at h
  have hz : zf32 = FloatV.ofPat 0 := FloatV.litF32_of_isZero (c := FloatV.ofPat 0) (by simp)
  rw [hz, FloatV.cvt_ofPat_zero Format.binary32_regular] at h
  have := h.eq_of_isFinite (by simp)
  subst this
  simp [ZB]

/-- **`tl.convert(tl.full(s, 0.0), bfloat16)`** (the `other` of the loads): a tensor of zeros. -/
theorem safe_other {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}
    {k : Tensor s BF16 → Prog α} (hk : ∀ z : Tensor s BF16, (∀ i, ZB (z.get i)) → Safe E R W Q m (k z)) :
    Safe E R W Q m ((tl.convert .f32 .bf16 (tl.full s zf32)).bind k) := by
  refine safe_convert (fun i => ⟨_, ConvRel.cvt_f32_bf16 _⟩) fun z hz => hk z fun i => ?_
  exact zb_of_conv (by simpa [tl.full] using hz i)

/-- A read of an element inside a footprint on which `m` and `init` agree gives the same value. -/
theorem read_init {dt : DType} {b : Buf dt} {R : Set Int} {m init : Mem}
    (hmi : ∀ x ∈ R, m x = init x) {S : Set Nat} (hS : b.elems S ⊆ R) {j : Nat} (hj : j ∈ S) :
    m.read dt (b.ptr (j : Int)).addr = init.read dt (b.ptr (j : Int)).addr :=
  Mem.read_congr fun _ hx => hmi _ (hS ⟨j, hj, hx⟩)

/-! ## The `w` load and the dot -/

/-- **The loaded `w` tile** of block `base0` in chunk `t`: lane `(i, a)` is `w[token 64t+i, state row
base0+a]` read from `init` when the token is in the sequence and the row below `K`, and a zero
otherwise. -/
def WTile (d : Inst) (init : Mem) (n hv t base0 : Nat) (bw : Tensor [BT, 64] BF16) : Prop :=
  ∀ (i : Fin BT) (a : Fin 64),
    (BT * t + i < lenOf d n ∧ base0 + a < d.K →
      rdB init d.w (wIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv (base0 + a)) =
        some (bw.get2 i a)) ∧
    (¬ (BT * t + i < lenOf d n ∧ base0 + a < d.K) → ZB (bw.get2 i a))

/-- What the `w` load of block `base0` and its dot establish: the loaded tile `bw`, the `bfloat16`
rounding `x` of the state block `xb`, and the dot's result `r`. -/
def WFacts (d : Inst) (P : Platform) (init : Mem) (n hv t base0 : Nat) (xb : Tensor [64, BV] F32)
    (acc r : Tensor [BT, BV] F32) : Prop :=
  ∃ bw, WTile d init n hv t base0 bw ∧ ∃ x : Tensor [64, BV] BF16,
    (∀ a jj, FloatV.NaNEq (FloatV.cvt (xb.get2 a jj)) (x.get2 a jj)) ∧ P.dot cfgBF16 bw x acc r

/-- **The `w` load of block `base0` and its dot** (source lines 211–216 and 217–237 for an existing
block): `b_w` is the tile of `w`, `x` the `bfloat16` rounding of the state block `xb`, and the
dot's result `r` any the platform allows. -/
theorem safe_wblock {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRw : d.w.bytes ⊆ R)
    (hvw : d.w.Valid m) {n hv t base0 : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    {wp : Ptr .bf16} (hwp : wp = d.w.ptr (((bosOf d n * d.HV + hv) * d.K : Nat) : Int))
    {o_t : Tensor [BT] I64} (hot : ∀ i : Fin BT, (o_t.get1 i).toInt = ((BT * t + i : Nat) : Int))
    {T : I64} (hT : T.toInt = (lenOf d n : Int)) {o_k : Tensor [64] I32} {m_k : Tensor [64] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hlen : (bosOf d n + lenOf d n) * d.HV * d.K ≤ d.w.len)
    (xb : Tensor [64, BV] F32) (acc : Tensor [BT, BV] F32) {k : Tensor [BT, BV] F32 → Prog α}
    (hk : ∀ r, WFacts d E.P init n hv t base0 xb acc r → Safe E R W Q m (k r)) :
    Safe E R W Q m ((tl.convert .f32 .bf16 (tl.full [BT, 64] zf32)).bind fun z =>
      (tl.load (wp + o_t.col 64 * bc (IntV.ofNat (d.HV * d.K) : I64) + o_k.row BT)
        (some (((o_t.ltS T).col 64).and (m_k.row BT))) (some z)).bind fun b_w =>
      (tl.cvt xb).bind fun x => (tl.dot cfgBF16 b_w x acc).bind k) := by
  refine safe_other fun z hz => ?_
  have hn0 := hd.HV_lt; have hK := hd.K_le; have hKp := hd.K_pos; have hHVp := hd.HV_pos
  have hT31 := hd.T_lt
  have hlen0 : BT * t < lenOf d n := by simp only [nch, BT] at ht ⊢; omega
  have hHVK : d.HV * d.K ≤ 2 ^ 24 := by
    have : d.HV * d.K ≤ 2 ^ 16 * 256 := Nat.mul_le_mul hn0.le hK
    omega
  have hSi : (IntV.ofNat (d.HV * d.K) : I64).toInt = ((d.HV * d.K : Nat) : Int) :=
    IntV.I64.toInt_ofNat (by omega)
  have hlenT : lenOf d n ≤ d.T := by
    have := start_add_len_le hd hn; omega
  have hT31' : d.T < 2 ^ 31 := by have : BT = 64 := rfl; omega
  set e : Nat := (bosOf d n * d.HV + hv) * d.K with he
  have hx : ∀ (i : Fin BT) (a : Fin 64), ((o_t.col 64 * bc (IntV.ofNat (d.HV * d.K) : I64)).get2 i a).toInt =
      (((BT * t + i) * (d.HV * d.K) : Nat) : Int) := by
    intro i a
    show (o_t.get1 i * (IntV.ofNat (d.HV * d.K) : I64)).toInt = _
    refine IntV.mul_nat (hot i) hSi ?_
    have h1 : (BT * t + i) * (d.HV * d.K) ≤ (2 ^ 31) * (2 ^ 24) :=
      Nat.mul_le_mul (by have := i.isLt; have : BT = 64 := rfl; omega) hHVK
    exact lt_of_le_of_lt h1 (by norm_num)
  have hy : ∀ (i : Fin BT) (a : Fin 64), ((o_k.row BT).get2 i a).toInt = (((base0 + a) * 1 : Nat) : Int) := by
    intro i a
    show (o_k.get1 a).toInt = _
    rw [hok a]; simp
  have hptr : ∀ (i : Fin BT) (a : Fin 64), BT * t + i < lenOf d n → base0 + a < d.K →
      (wp + o_t.col 64 * bc (IntV.ofNat (d.HV * d.K) : I64) + o_k.row BT).get2 i a =
        d.w.ptr ((e + (BT * t + i) * (d.HV * d.K) + (base0 + a) * 1 : Nat) : Int) := fun i a _ _ =>
    ptr_lane_seq_of d.w hwp hx hy i a
  have hmask : MaskAt (some (((o_t.ltS T).col 64).and (m_k.row BT))) (fun i : Fin BT => BT * t + i)
      (fun a : Fin 64 => base0 + a) (lenOf d n) d.K := by
    refine maskAt_of_get fun i a => ?_
    have hlt : (o_t.ltS T).get1 i = decide (BT * t + i < lenOf d n) := by
      show (o_t.get1 i).lt T = _
      rw [IntV.lt_eq, hot i, hT]; norm_cast
    show ((o_t.ltS T).get1 i && m_k.get1 a) = _
    rw [hlt, hmk a]
  have hlen' : lenOf d n = 0 ∨ d.K = 0 ∨ e + (lenOf d n - 1) * (d.HV * d.K) + (d.K - 1) * 1 < d.w.len := by
    right; right
    obtain ⟨L, hL⟩ : ∃ L, lenOf d n = L + 1 := ⟨lenOf d n - 1, by omega⟩
    rw [hL] at hlen
    have h1 : (bosOf d n + (L + 1)) * d.HV * d.K = bosOf d n * d.HV * d.K + L * (d.HV * d.K) + d.HV * d.K := by ring
    have h2 : e = bosOf d n * d.HV * d.K + hv * d.K := by simp only [he]; ring
    have h3 : (hv + 1) * d.K ≤ d.HV * d.K := Nat.mul_le_mul_right _ hhv
    have h4 : (hv + 1) * d.K = hv * d.K + d.K := by ring
    rw [hL]; simp only [Nat.add_sub_cancel, mul_one]
    omega
  have hR : d.w.tileAt e (d.HV * d.K) 1 (fun i : Fin BT => BT * t + i) (fun a : Fin 64 => base0 + a)
      (lenOf d n) d.K ⊆ R := (d.w.tileAt_sub_bytes hlen').trans hRw
  refine safe_load_tileAt d.w (ρ := fun i : Fin BT => BT * t + i) (κ := fun a : Fin 64 => base0 + a)
    hptr hmask hlen' hvw hR fun bw hr ho => ?_
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun x hcv => ?_
  refine safe_dot (hP.dotTotal cfgBF16 bw x acc) fun r hr' => ?_
  refine hk r ⟨bw, fun i a => ⟨fun ⟨hi, ha⟩ => ?_, fun hno => ?_⟩, x, fun a jj => hcv (a, jj, ()), hr'⟩
  · have hrd := hr i a hi ha
    have hidx : wIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv (base0 + a) =
        e + (BT * t + i) * (d.HV * d.K) + (base0 + a) * 1 := by
      simp only [wIdx, bosOf, he]; ring
    unfold rdB
    rw [hidx, ← read_init hmi (b := d.w) (S := {x | ElemAt e (d.HV * d.K) 1 (fun i : Fin BT => BT * t + i)
      (fun a : Fin 64 => base0 + a) (lenOf d n) d.K x}) hR (j := e + (BT * t + i) * (d.HV * d.K) + (base0 + a) * 1)
      ⟨i, a, hi, ha, rfl⟩]
    exact hrd
  · have := ho i a hno z rfl
    rw [this]; exact hz (i, a, ())

/-- **The loaded `u` tile** of chunk `t` and block `jv`: lane `(i, jj)` is `u[token 64t+i, column
32jv+jj]` read from `init` when the token is in the sequence and the column below `V`, and a zero
otherwise. -/
def UTile (d : Inst) (init : Mem) (n hv jv t : Nat) (bu : Tensor [BT, BV] BF16) : Prop :=
  ∀ (i : Fin BT) (jj : Fin BV),
    (BT * t + i < lenOf d n ∧ BV * jv + jj < d.V →
      rdB init d.u (uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv (BV * jv + jj)) =
        some (bu.get2 i jj)) ∧
    (¬ (BT * t + i < lenOf d n ∧ BV * jv + jj < d.V) → ZB (bu.get2 i jj))

/-- What the `u` load, the widening, the fusable `-` and the `bfloat16` rounding establish, `bv'`
the difference `u - b_v` and `x` its rounding, and the stored values: the `float32` `u` tile `uf`
is the widened `bu`, and every enabled lane of the stored `v_new` tile holds `x`'s lane in `m'`. -/
def VFacts (d : Inst) (P : Platform) (fusion : Bool) (init m' : Mem) (n hv jv t : Nat)
    (b_v bv' : Tensor [BT, BV] F32) : Prop :=
  ∃ bu, UTile d init n hv jv t bu ∧ ∃ (uf : Tensor [BT, BV] F32) (x : Tensor [BT, BV] BF16),
    (∀ i jj, ConvRel P .bf16 .f32 (bu.get2 i jj) (uf.get2 i jj)) ∧
    (∀ i jj, AddRel fusion (uf.get2 i jj) (FloatV.neg (b_v.get2 i jj)) (bv'.get2 i jj)) ∧
    (∀ i jj, FloatV.NaNEq (FloatV.cvt (bv'.get2 i jj)) (x.get2 i jj)) ∧
    (∀ (i : Fin BT) (jj : Fin BV), BT * t + i < lenOf d n → BV * jv + jj < d.V →
      rdB m' d.vn (uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv (BV * jv + jj)) = some (x.get2 i jj))

/-- **The `u` load, the difference and the `v_new` store** (source lines 238–243). -/
theorem safe_vblock {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRu : d.u.bytes ⊆ R)
    (hvu : d.u.Valid m) (hvv : d.vn.Valid m) {n hv jv t : Nat} (hn : n < d.N) (hhv : hv < d.HV)
    (ht : t < nch d n)
    {vp vnp : Ptr .bf16} (hvp : vp = d.u.ptr (((bosOf d n * d.HV + hv) * d.V : Nat) : Int))
    (hvnp : vnp = d.vn.ptr (((bosOf d n * d.HV + hv) * d.V : Nat) : Int))
    {o_t : Tensor [BT] I64} (hot : ∀ i : Fin BT, (o_t.get1 i).toInt = ((BT * t + i : Nat) : Int))
    {T : I64} (hT : T.toInt = (lenOf d n : Int)) {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hlenu : (bosOf d n + lenOf d n) * d.HV * d.V ≤ d.u.len)
    (hlenv : (bosOf d n + lenOf d n) * d.HV * d.V ≤ d.vn.len)
    (hW : d.vn.elems (vCells d n hv jv t) ⊆ W) (b_v : Tensor [BT, BV] F32)
    {k : Tensor [BT, BV] F32 → Unit → Prog α}
    (hk : ∀ bv' m', VFacts d E.P E.fusion init m' n hv jv t b_v bv' →
      (∀ y, y ∉ d.vn.elems (vCells d n hv jv t) → m' y = m y) → (∀ y, m y ≠ none → m' y ≠ none) →
      Safe E R (W \ d.vn.elems (vCells d n hv jv t)) Q m' (k bv' ())) :
    Safe E R W Q m ((tl.convert .f32 .bf16 (tl.full [BT, BV] zf32)).bind fun z =>
      (tl.load (vp + o_t.col BV * bc (IntV.ofNat (d.HV * d.V) : I64) + o_v.row BT)
        (some (((o_t.ltS T).col BV).and (m_v.row BT))) (some z)).bind fun b_u =>
      (tl.convert .bf16 .f32 b_u).bind fun uf => (tl.subc uf b_v).bind fun bv' =>
      (tl.cvt bv').bind fun x =>
      (tl.store (vnp + o_t.col BV * bc (IntV.ofNat (d.HV * d.V) : I64) + o_v.row BT) x
        (some (((o_t.ltS T).col BV).and (m_v.row BT)))).bind (k bv')) := by
  refine safe_other fun z hz => ?_
  have hVp := hd.V_pos; have hHVp := hd.HV_pos
  have hHVV := hd.HVV_lt
  have hT31 := hd.T_lt
  have hlen0 : BT * t < lenOf d n := by simp only [nch, BT] at ht ⊢; omega
  have hSi : (IntV.ofNat (d.HV * d.V) : I64).toInt = ((d.HV * d.V : Nat) : Int) :=
    IntV.I64.toInt_ofNat (by omega)
  have hlenT : lenOf d n ≤ d.T := by
    have := start_add_len_le hd hn; omega
  set e : Nat := (bosOf d n * d.HV + hv) * d.V with he
  have hx : ∀ (i : Fin BT) (jj : Fin BV), ((o_t.col BV * bc (IntV.ofNat (d.HV * d.V) : I64)).get2 i jj).toInt =
      (((BT * t + i) * (d.HV * d.V) : Nat) : Int) := by
    intro i jj
    show (o_t.get1 i * (IntV.ofNat (d.HV * d.V) : I64)).toInt = _
    refine IntV.mul_nat (hot i) hSi ?_
    have h1 : (BT * t + i) * (d.HV * d.V) ≤ (2 ^ 31) * (2 ^ 31) :=
      Nat.mul_le_mul (by have := i.isLt; have : BT = 64 := rfl; omega) hHVV.le
    exact lt_of_le_of_lt h1 (by norm_num)
  have hy : ∀ (i : Fin BT) (jj : Fin BV), ((o_v.row BT).get2 i jj).toInt = (((BV * jv + jj) * 1 : Nat) : Int) := by
    intro i jj
    show (o_v.get1 jj).toInt = _
    rw [hov jj]; simp
  have hptr : ∀ (b : Buf .bf16) {p : Ptr .bf16}, p = b.ptr (e : Int) → ∀ (i : Fin BT) (jj : Fin BV),
      (p + o_t.col BV * bc (IntV.ofNat (d.HV * d.V) : I64) + o_v.row BT).get2 i jj =
        b.ptr ((e + (BT * t + i) * (d.HV * d.V) + (BV * jv + jj) * 1 : Nat) : Int) := fun b p hp i jj =>
    ptr_lane_seq_of b hp hx hy i jj
  have hmask : MaskAt (some (((o_t.ltS T).col BV).and (m_v.row BT))) (fun i : Fin BT => BT * t + i)
      (fun jj : Fin BV => BV * jv + jj) (lenOf d n) d.V := by
    refine maskAt_of_get fun i jj => ?_
    have hlt : (o_t.ltS T).get1 i = decide (BT * t + i < lenOf d n) := by
      show (o_t.get1 i).lt T = _
      rw [IntV.lt_eq, hot i, hT]; norm_cast
    show ((o_t.ltS T).get1 i && m_v.get1 jj) = _
    rw [hlt, hmv jj]
  have hlen' : ∀ (len : Nat), (bosOf d n + lenOf d n) * d.HV * d.V ≤ len →
      lenOf d n = 0 ∨ d.V = 0 ∨ e + (lenOf d n - 1) * (d.HV * d.V) + (d.V - 1) * 1 < len := by
    intro len hl
    right; right
    obtain ⟨L, hL⟩ : ∃ L, lenOf d n = L + 1 := ⟨lenOf d n - 1, by omega⟩
    rw [hL] at hl
    have h1 : (bosOf d n + (L + 1)) * d.HV * d.V = bosOf d n * d.HV * d.V + L * (d.HV * d.V) + d.HV * d.V := by ring
    have h2 : e = bosOf d n * d.HV * d.V + hv * d.V := by simp only [he]; ring
    have h3 : (hv + 1) * d.V ≤ d.HV * d.V := Nat.mul_le_mul_right _ hhv
    have h4 : (hv + 1) * d.V = hv * d.V + d.V := by ring
    rw [hL]; simp only [Nat.add_sub_cancel, mul_one]
    omega
  have hR : d.u.tileAt e (d.HV * d.V) 1 (fun i : Fin BT => BT * t + i) (fun jj : Fin BV => BV * jv + jj)
      (lenOf d n) d.V ⊆ R := (d.u.tileAt_sub_bytes (hlen' _ hlenu)).trans hRu
  have hidx : ∀ (i jj : Nat), uIdx d (rowOf d n) (startOf d n + BT * t + i) hv jj =
      e + (BT * t + i) * (d.HV * d.V) + jj * 1 := by
    intro i jj; simp only [uIdx, bosOf, he]; ring
  have hset : {x | ElemAt e (d.HV * d.V) 1 (fun i : Fin BT => BT * t + i) (fun jj : Fin BV => BV * jv + jj)
      (lenOf d n) d.V x} = vCells d n hv jv t := by
    ext x
    constructor
    · rintro ⟨i, jj, hi, hj, rfl⟩
      refine ⟨i, BV * jv + jj, ?_, ⟨by omega, by omega, hj⟩, (hidx _ _).symm⟩
      have : i.1 < BT := i.2
      have hi : BT * t + (i : Nat) < lenOf d n := hi
      show (i : Nat) < min BT (lenOf d n - BT * t)
      omega
    · rintro ⟨i, j, hi, ⟨hj1, hj2, hj3⟩, rfl⟩
      have hi' : i < BT := lt_of_lt_of_le hi (min_le_left _ _)
      have hi'' : i < lenOf d n - BT * t := lt_of_lt_of_le hi (min_le_right _ _)
      refine ⟨⟨i, hi'⟩, ⟨j - BV * jv, by unfold BV at *; omega⟩, by show BT * t + i < _; omega,
        by show BV * jv + (j - BV * jv) < _; omega, ?_⟩
      rw [← hidx]
      show uIdx d (rowOf d n) (startOf d n + BT * t + i) hv j = _
      have : BV * jv + (j - BV * jv) = j := by omega
      rw [hidx, hidx]; simp only [this]
  have hTile : d.vn.tileAt e (d.HV * d.V) 1 (fun i : Fin BT => BT * t + i)
      (fun jj : Fin BV => BV * jv + jj) (lenOf d n) d.V = d.vn.elems (vCells d n hv jv t) := by
    unfold Buf.tileAt; rw [hset]
  refine safe_load_tileAt d.u (ρ := fun i : Fin BT => BT * t + i) (κ := fun jj : Fin BV => BV * jv + jj)
    (fun i jj _ _ => hptr d.u hvp i jj) hmask (hlen' _ hlenu) hvu hR fun bu hrl hol => ?_
  refine safe_convert (fun i => ⟨_, ConvRel.cvt_bf16_f32 _⟩) fun uf huf => ?_
  refine safe_subc fun bv' hsub => ?_
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun x hcv => ?_
  refine safe_store_tileAt d.vn (ρ := fun i : Fin BT => BT * t + i) (κ := fun jj : Fin BV => BV * jv + jj)
    (e := e) (s0 := d.HV * d.V) (s1 := 1) (M := lenOf d n) (N := d.V)
    (fun i jj _ _ => hptr d.vn hvnp i jj) hmask (hlen' _ hlenv) ?_
    (fun i i' _ _ h => Fin.ext (by simpa using h)) ?_ hvv (by rw [hTile]; exact hW) ?_
  · exact inj_rowMajor1 (Nat.le_mul_of_pos_left _ hHVp)
  · intro j j' _ _ h; exact Fin.ext (by simpa using h)
  · intro m' hr hf hp'
    rw [hTile] at hf ⊢
    refine hk bv' m' ⟨bu, fun i jj => ⟨fun ⟨hi, hj⟩ => ?_, fun hno => ?_⟩, uf, x,
      fun i jj => huf (i, jj, ()), fun i jj => hsub (i, jj, ()), fun i jj => hcv (i, jj, ()), ?_⟩ hf hp'
    · have hrd := hrl i jj hi hj
      have hidx' : uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv (BV * jv + jj) =
          e + (BT * t + i) * (d.HV * d.V) + (BV * jv + jj) * 1 := by
        rw [← add_assoc, hidx]
      unfold rdB
      rw [hidx', ← read_init hmi (b := d.u) (S := {x | ElemAt e (d.HV * d.V) 1 (fun i : Fin BT => BT * t + i)
        (fun jj : Fin BV => BV * jv + jj) (lenOf d n) d.V x}) hR (j := e + (BT * t + i) * (d.HV * d.V) + (BV * jv + jj) * 1)
        ⟨i, jj, hi, hj, rfl⟩]
      exact hrd
    · have := hol i jj hno z rfl
      rw [this]; exact hz (i, jj, ())
    · intro i jj hi hj
      have hidx' : uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv (BV * jv + jj) =
          e + (BT * t + i) * (d.HV * d.V) + (BV * jv + jj) * 1 := by
        rw [← add_assoc, hidx]
      unfold rdB
      rw [hidx']
      exact hr i jj hi hj

/-! ## The gate -/

/-- A `float32` zero: finite, value `0`. -/
def ZF (x : F32) : Prop := x.isFinite = true ∧ x.toReal = 0

/-- The `float32` constant `+0` is a zero. -/
theorem zf_zf32 : ZF zf32 := by
  have hz : zf32 = FloatV.ofPat 0 := FloatV.litF32_of_isZero (c := FloatV.ofPat 0) (by simp)
  rw [hz]; simp [ZF]

/-- The index in `g` of token `s` of sequence `n`'s row, value head `hv`. -/
def gTok (d : Inst) (n hv s : Nat) : Nat := gIdx d (rowOf d n) (startOf d n + s) hv

/-- What the gate's loads and arithmetic establish: `bgl` is `g_last` (read from `init`), `bg` the
tile of `g` (a zero for a token past the sequence's end), `xs = g_last - g`, `e` the device's
`exp2` of it, `bv2` the gated values `b_v · (e or 0)`, and `gt` the device's `exp2(g_last)`. -/
def GFacts (d : Inst) (P : Platform) (init : Mem) (n hv t : Nat) (b_v bv2 : Tensor [BT, BV] F32)
    (gt : Tensor [] F32) : Prop :=
  ∃ (bgl : F32) (bg xs e : Tensor [BT] F32),
    rdF init d.g (gTok d n hv (BT * t + cl d n t - 1)) = some bgl ∧
    (∀ i : Fin BT, (BT * t + i < lenOf d n →
        rdF init d.g (gTok d n hv (BT * t + i)) = some (bg.get1 i)) ∧
      (¬ BT * t + i < lenOf d n → ZF (bg.get1 i))) ∧
    (∀ i : Fin BT, FloatV.NaNEq (FloatV.sub bgl (bg.get1 i)) (xs.get1 i)) ∧
    (∀ i : Fin BT, P.math .exp2 .binary32 (xs.get1 i) (e.get1 i)) ∧
    (∀ (i : Fin BT) (jj : Fin BV), FloatV.NaNEq
      (FloatV.mul (b_v.get2 i jj) (if BT * t + i < lenOf d n then e.get1 i else zf32))
      (bv2.get2 i jj)) ∧
    P.math .exp2 .binary32 bgl gt.item

/-- **The gate** (source lines 247–251): `g_last` and the tile of `g`, the decay `exp2(g_last - g)`
of every token (zero past the sequence's end) applied to the new values, and `exp2(g_last)`. -/
theorem safe_gate {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRg : d.g.bytes ⊆ R)
    (hvg : d.g.Valid m) {n hv t : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    {gp : Ptr .f32} (hgp : gp = d.g.ptr 0) {bos i_h : I64} (hbos : bos.toInt = (bosOf d n : Int))
    (hih : i_h.toInt = (hv : Int))
    {o_t : Tensor [BT] I64} (hot : ∀ i : Fin BT, (o_t.get1 i).toInt = ((BT * t + i : Nat) : Int))
    {T : I64} (hT : T.toInt = (lenOf d n : Int)) {goff : I64}
    (hgoff : goff.toInt = (gTok d n hv (BT * t + cl d n t - 1) : Int))
    (hlen : (bosOf d n + lenOf d n) * d.HV ≤ d.g.len) (b_v : Tensor [BT, BV] F32)
    {k : Tensor [BT, BV] F32 → Tensor [] F32 → Prog α}
    (hk : ∀ bv2 gt, GFacts d E.P init n hv t b_v bv2 gt → Safe E R W Q m (k bv2 gt)) :
    Safe E R W Q m ((tl.load1 (gp + goff)).bind fun bgl =>
      (tl.load (gp + (bos * IntV.ofNat d.HV + i_h) + o_t * bc (IntV.ofNat d.HV : I64))
        (some (o_t.ltS T)) (some (bc zf32))).bind fun bg =>
      (tl.sub (bc bgl) bg).bind fun xs => (tl.exp2 xs).bind fun e_g =>
      (tl.mul b_v ((tl.where (o_t.ltS T) e_g (bc zf32)).col BV)).bind fun bv2 =>
      (tl.exp2 (Tensor.scalar bgl)).bind (k bv2)) := by
  have hHVp := hd.HV_pos; have hHVl := hd.HV_lt
  have hT31 := hd.T_lt
  have hlen0 : BT * t < lenOf d n := by simp only [nch, BT] at ht ⊢; omega
  have hlenT : lenOf d n ≤ d.T := by
    have := start_add_len_le hd hn; omega
  have hcl : 0 < cl d n t := by simp only [cl, BT] at *; omega
  have hcl' : BT * t + cl d n t ≤ lenOf d n := by simp only [cl, BT] at *; omega
  have hSi : (IntV.ofNat d.HV : I64).toInt = (d.HV : Int) := IntV.I64.toInt_ofNat (by omega)
  have hidx : ∀ s, gTok d n hv s = (bosOf d n + s) * d.HV + hv := by
    intro s; simp only [gTok, gIdx, bosOf]; ring
  have hin : ∀ s, s < lenOf d n → gTok d n hv s < d.g.len := by
    intro s hs
    rw [hidx]
    have h1 : (bosOf d n + s + 1) * d.HV ≤ (bosOf d n + lenOf d n) * d.HV :=
      Nat.mul_le_mul_right _ (by omega)
    have h2 : (bosOf d n + s + 1) * d.HV = (bosOf d n + s) * d.HV + d.HV := by ring
    omega
  have hbosb := bos_bound hd hn
  have hA : (bos * IntV.ofNat d.HV + i_h).toInt = ((bosOf d n * d.HV + hv : Nat) : Int) := by
    have hb1 : (bos * IntV.ofNat d.HV).toInt = ((bosOf d n * d.HV : Nat) : Int) :=
      IntV.mul_nat hbos hSi (by
        have : bosOf d n * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) hHVl.le
        exact lt_of_le_of_lt this (by norm_num))
    refine IntV.add_nat hb1 hih ?_
    have : bosOf d n * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) hHVl.le
    have : hv < 2 ^ 16 := by omega
    show _ < 2 ^ 63
    calc _ ≤ 2 ^ 35 * 2 ^ 16 + 2 ^ 16 := by omega
      _ < 2 ^ 63 := by norm_num
  have hx : ∀ i : Fin BT, ((o_t * bc (IntV.ofNat d.HV : I64)).get1 i).toInt = (((BT * t + i) * d.HV : Nat) : Int) := by
    intro i
    show (o_t.get1 i * (IntV.ofNat d.HV : I64)).toInt = _
    refine IntV.mul_nat (hot i) hSi ?_
    have h1 : (BT * t + i) * d.HV ≤ 2 ^ 31 * 2 ^ 16 :=
      Nat.mul_le_mul (by have := i.isLt; have : BT = 64 := rfl; omega) hHVl.le
    exact lt_of_le_of_lt h1 (by norm_num)
  have hptr : ∀ i : Fin BT, (gp + (bos * IntV.ofNat d.HV + i_h) + o_t * bc (IntV.ofNat d.HV : I64)).get1 i =
      d.g.ptr ((gTok d n hv (BT * t + i) : Nat) : Int) := by
    intro i
    show (gp.offset (bos * IntV.ofNat d.HV + i_h).toInt).offset ((o_t * bc (IntV.ofNat d.HV : I64)).get1 i).toInt = _
    rw [hA, hx i, hgp, Buf.ptr_offset, Buf.ptr_offset, hidx]
    push_cast; ring_nf
  have hmask : ∀ i : Fin BT, (o_t.ltS T).get1 i = decide (BT * t + i < lenOf d n) := by
    intro i
    show (o_t.get1 i).lt T = _
    rw [IntV.lt_eq, hot i, hT]; norm_cast
  have hgl : gp + goff = d.g.ptr ((gTok d n hv (BT * t + cl d n t - 1) : Nat) : Int) := by
    show gp.offset goff.toInt = _
    rw [hgoff, hgp, Buf.ptr_offset]; simp
  have hglt : gTok d n hv (BT * t + cl d n t - 1) < d.g.len := hin _ (by omega)
  have hRs : ∀ S : Set Nat, (∀ j ∈ S, j < d.g.len) → d.g.elems S ⊆ R :=
    fun S hS => (d.g.elems_sub_bytes hS).trans hRg
  rw [hgl]
  refine safe_load1_buf d.g hglt hvg (fun x hx => hRg (d.g.span_sub hglt hx)) fun bgl hbgl => ?_
  have hbgl' : rdF init d.g (gTok d n hv (BT * t + cl d n t - 1)) = some bgl := by
    unfold rdF
    rw [← read_init hmi (b := d.g) (S := {gTok d n hv (BT * t + cl d n t - 1)})
      (hRs _ (fun j hj => by rw [hj]; exact hglt)) rfl]
    exact hbgl
  refine safe_load_buf d.g (g := fun i : Idx [BT] => gTok d n hv (BT * t + i.1.1))
    (en := fun i => decide (BT * t + i.1.1 < lenOf d n))
    (Buf.addrs_of_get d.g (fun ⟨i, ⟨⟩⟩ => hmask i) (fun ⟨i, ⟨⟩⟩ _ => hptr i))
    (fun ⟨i, ⟨⟩⟩ hi => hin _ (by simpa using hi)) hvg
    (hRs _ ?_) fun bg hr ho => ?_
  · rintro _ ⟨⟨i, ⟨⟩⟩, hi, rfl⟩
    exact hin _ (by simpa using hi)
  refine safe_sub fun xs hxs => ?_
  refine safe_math (fun i => hP.mathTotal .exp2 .binary32 _) fun e he => ?_
  refine safe_mul fun bv2 hmul => ?_
  refine safe_math (fun i => hP.mathTotal .exp2 .binary32 _) fun gt hgt => ?_
  refine hk bv2 gt ⟨bgl, bg, xs, e, hbgl', fun i => ⟨fun hi => ?_, fun hno => ?_⟩, fun i => ?_,
    fun i => he (i, ()), fun i jj => ?_, hgt ()⟩
  · have hrd := hr (i, ()) (by simpa using hi)
    unfold rdF
    rw [← read_init hmi (b := d.g) (S := {gTok d n hv (BT * t + i)})
      (hRs _ (fun j hj => by rw [hj]; exact hin _ hi)) rfl]
    exact hrd
  · have := ho (i, ()) (by simpa using hno) _ rfl
    rw [Tensor.get1, this]
    exact zf_zf32
  · exact hxs (i, ())
  · have := hmul (i, jj, ())
    have hw : ((tl.where (o_t.ltS T) e (bc zf32)).col BV).get (i, jj, ()) =
        if BT * t + i < lenOf d n then e.get1 i else zf32 := by
      show (if (o_t.ltS T).get1 i = true then e.get1 i else zf32) = _
      rw [hmask i]
      by_cases h : BT * t + i < lenOf d n <;> simp [h]
    rw [hw] at this
    exact this

/-! ## The state scaling, the `k` load and the update dot -/

/-- **`b_h * g`** (the decay of a state block by the scalar `g`): the IEEE product lane by lane. -/
theorem safe_scale {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}
    (b : Tensor s F32) (g : F32) {k : Tensor s F32 → Prog α}
    (hk : ∀ r : Tensor s F32, (∀ i, FloatV.NaNEq (FloatV.mul (b.get i) g) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m ((tl.mul b (bc g)).bind k) :=
  safe_mul fun r hr => hk r fun i => by simpa [bc] using hr i

/-- **The loaded `k` tile** of block `base0` in chunk `t`: lane `(a, i)` is `k[token 64t+i, state row
base0+a]` of the key head of `hv`, read from `init`, when the row is below `K` and the token in the
sequence, and a zero otherwise. -/
def KTile (d : Inst) (init : Mem) (n hv t base0 : Nat) (bk : Tensor [64, BT] BF16) : Prop :=
  ∀ (a : Fin 64) (i : Fin BT),
    (base0 + a < d.K ∧ BT * t + i < lenOf d n →
      rdB init d.k (kIdx d (rowOf d n) (startOf d n + (BT * t + i)) (kHead d hv) (base0 + a)) =
        some (bk.get2 a i)) ∧
    (¬ (base0 + a < d.K ∧ BT * t + i < lenOf d n) → ZB (bk.get2 a i))

/-- What the `k` load and the dot establish: the loaded tile `bk` and the dot's result `r` from the
accumulator `acc` and the rounded gated values `vb`. -/
def KFacts (d : Inst) (P : Platform) (init : Mem) (n hv t base0 : Nat) (vb : Tensor [BT, BV] BF16)
    (acc r : Tensor [64, BV] F32) : Prop :=
  ∃ bk, KTile d init n hv t base0 bk ∧ P.dot cfgBF16 bk vb acc r

/-- **The `k` load of block `base0` and its dot** (source lines 290–295 and 296–316 for an existing
block). -/
theorem safe_kdot {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRk : d.k.bytes ⊆ R)
    (hvk : d.k.Valid m) {n hv t base0 : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    {kp : Ptr .bf16} (hkp : kp = d.k.ptr (((bosOf d n * d.H + kHead d hv) * d.K : Nat) : Int))
    {o_t : Tensor [BT] I64} (hot : ∀ i : Fin BT, (o_t.get1 i).toInt = ((BT * t + i : Nat) : Int))
    {T : I64} (hT : T.toInt = (lenOf d n : Int)) {o_k : Tensor [64] I32} {m_k : Tensor [64] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hlen : (bosOf d n + lenOf d n) * d.H * d.K ≤ d.k.len)
    (vb : Tensor [BT, BV] BF16) (acc : Tensor [64, BV] F32) {k : Tensor [64, BV] F32 → Prog α}
    (hk : ∀ r, KFacts d E.P init n hv t base0 vb acc r → Safe E R W Q m (k r)) :
    Safe E R W Q m ((tl.convert .f32 .bf16 (tl.full [64, BT] zf32)).bind fun z =>
      (tl.load (kp + o_k.col BT + o_t.row 64 * bc (IntV.ofNat (d.H * d.K) : I64))
        (some ((m_k.col BT).and ((o_t.ltS T).row 64))) (some z)).bind fun b_k =>
      (tl.dot cfgBF16 b_k vb acc).bind k) := by
  refine safe_other fun z hz => ?_
  have hHle : d.H ≤ d.HV := Nat.le_of_dvd hd.HV_pos hd.H_dvd
  have hHl := hd.HV_lt
  have hK := hd.K_le; have hKp := hd.K_pos; have hHp := hd.H_pos
  have hT31 := hd.T_lt
  have hlen0 : BT * t < lenOf d n := by simp only [nch, BT] at ht ⊢; omega
  have hHK : d.H * d.K ≤ 2 ^ 24 := by
    have : d.H * d.K ≤ 2 ^ 16 * 256 := Nat.mul_le_mul (by omega) hK
    omega
  have hSi : (IntV.ofNat (d.H * d.K) : I64).toInt = ((d.H * d.K : Nat) : Int) :=
    IntV.I64.toInt_ofNat (by omega)
  have hlenT : lenOf d n ≤ d.T := by
    have := start_add_len_le hd hn; omega
  set e : Nat := (bosOf d n * d.H + kHead d hv) * d.K with he
  have hkh : kHead d hv < d.H := by
    unfold kHead
    have hdv : 0 < d.HV / d.H := Nat.div_pos hHle hHp
    rw [Nat.div_lt_iff_lt_mul hdv]
    obtain ⟨c, hc⟩ := hd.H_dvd
    have : d.HV / d.H = c := by rw [hc]; exact Nat.mul_div_cancel_left _ hHp
    rw [this]; rw [hc] at hhv; linarith
  have hx : ∀ (a : Fin 64) (i : Fin BT), ((o_k.col BT).get2 a i).toInt = (((base0 + a) * 1 : Nat) : Int) := by
    intro a i
    show (o_k.get1 a).toInt = _
    rw [hok a]; simp
  have hy : ∀ (a : Fin 64) (i : Fin BT), ((o_t.row 64 * bc (IntV.ofNat (d.H * d.K) : I64)).get2 a i).toInt =
      (((BT * t + i) * (d.H * d.K) : Nat) : Int) := by
    intro a i
    show (o_t.get1 i * (IntV.ofNat (d.H * d.K) : I64)).toInt = _
    refine IntV.mul_nat (hot i) hSi ?_
    have h1 : (BT * t + i) * (d.H * d.K) ≤ (2 ^ 31) * (2 ^ 24) :=
      Nat.mul_le_mul (by have := i.isLt; have : BT = 64 := rfl; omega) hHK
    exact lt_of_le_of_lt h1 (by norm_num)
  have hptr : ∀ (a : Fin 64) (i : Fin BT), base0 + a < d.K → BT * t + i < lenOf d n →
      (kp + o_k.col BT + o_t.row 64 * bc (IntV.ofNat (d.H * d.K) : I64)).get2 a i =
        d.k.ptr ((e + (base0 + a) * 1 + (BT * t + i) * (d.H * d.K) : Nat) : Int) := fun a i _ _ =>
    ptr_lane_seq_of d.k hkp hx hy a i
  have hmask : MaskAt (some ((m_k.col BT).and ((o_t.ltS T).row 64))) (fun a : Fin 64 => base0 + a)
      (fun i : Fin BT => BT * t + i) d.K (lenOf d n) := by
    refine maskAt_of_get fun a i => ?_
    have hlt : (o_t.ltS T).get1 i = decide (BT * t + i < lenOf d n) := by
      show (o_t.get1 i).lt T = _
      rw [IntV.lt_eq, hot i, hT]; norm_cast
    show (m_k.get1 a && (o_t.ltS T).get1 i) = _
    rw [hlt, hmk a]
  have hlen' : d.K = 0 ∨ lenOf d n = 0 ∨ e + (d.K - 1) * 1 + (lenOf d n - 1) * (d.H * d.K) < d.k.len := by
    right; right
    obtain ⟨L, hL⟩ : ∃ L, lenOf d n = L + 1 := ⟨lenOf d n - 1, by omega⟩
    rw [hL] at hlen
    have h1 : (bosOf d n + (L + 1)) * d.H * d.K = bosOf d n * d.H * d.K + L * (d.H * d.K) + d.H * d.K := by ring
    have h2 : e = bosOf d n * d.H * d.K + kHead d hv * d.K := by simp only [he]; ring
    have h3 : (kHead d hv + 1) * d.K ≤ d.H * d.K := Nat.mul_le_mul_right _ hkh
    have h4 : (kHead d hv + 1) * d.K = kHead d hv * d.K + d.K := by ring
    rw [hL]; simp only [Nat.add_sub_cancel, mul_one]
    omega
  have hR : d.k.tileAt e 1 (d.H * d.K) (fun a : Fin 64 => base0 + a) (fun i : Fin BT => BT * t + i)
      d.K (lenOf d n) ⊆ R := (d.k.tileAt_sub_bytes hlen').trans hRk
  refine safe_load_tileAt d.k (ρ := fun a : Fin 64 => base0 + a) (κ := fun i : Fin BT => BT * t + i)
    hptr hmask hlen' hvk hR fun bk hr ho => ?_
  refine safe_dot (hP.dotTotal cfgBF16 bk vb acc) fun r hr' => ?_
  refine hk r ⟨bk, fun a i => ⟨fun ⟨ha, hi⟩ => ?_, fun hno => ?_⟩, hr'⟩
  · have hrd := hr a i ha hi
    have hidx : kIdx d (rowOf d n) (startOf d n + (BT * t + i)) (kHead d hv) (base0 + a) =
        e + (base0 + a) * 1 + (BT * t + i) * (d.H * d.K) := by
      simp only [kIdx, bosOf, he]; ring
    unfold rdB
    rw [hidx, ← read_init hmi (b := d.k) (S := {x | ElemAt e 1 (d.H * d.K) (fun a : Fin 64 => base0 + a)
      (fun i : Fin BT => BT * t + i) d.K (lenOf d n) x}) hR
      (j := e + (base0 + a) * 1 + (BT * t + i) * (d.H * d.K)) ⟨a, i, ha, hi, rfl⟩]
    exact hrd
  · have := ho a i hno z rfl
    rw [this]; exact hz (a, i, ())

/-! ## The optional state stores -/

/-- No element is stored by a block that does not exist. -/
theorem hBlk_empty {d : Inst} {n hv jv t base0 : Nat} (h : ¬ base0 < d.K) :
    d.h.elems (hBlk d n hv jv t base0) = ∅ := by
  have : hBlk d n hv jv t base0 = ∅ := by
    ext x
    simp only [hBlk, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨a, jj, -, ha, -⟩
    omega
  rw [this]
  ext x
  simp [Buf.elems]

/-- **The store of an optional state block** (`if base0 < K: tl.store(...)`, source lines 186–209):
as `safe_hstore` when the block exists, and nothing happens otherwise. -/
theorem safe_hstore_opt {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {n hh jv t base0 : Nat}
    (hvf : d.vf = false) (hhh : hh < d.HV) (hb64 : base0 % 64 = 0)
    (hbase : base0 ≤ 192) {hp : Ptr .bf16} {sc : I64} {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hhp : hp = d.h.ptr (((cbase d n * d.HV + hh) * d.K * d.V : Nat) : Int))
    (hsc : sc.toInt = ((t * d.HV * d.K * d.V : Nat) : Int))
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hvalid : d.h.Valid m) (hW : d.h.elems (hBlk d n hh jv t base0) ⊆ W)
    (hlen : (cbase d n + t + 1) * d.HV * d.K * d.V ≤ d.h.len)
    {b : Tensor [64, BV] F32} {k : Unit → Prog α}
    (hk : ∀ x : Tensor [64, BV] BF16,
      (∀ a jj, FloatV.NaNEq (FloatV.cvt (b.get2 a jj)) (x.get2 a jj)) →
      ∀ m' : Mem,
        (∀ (a : Fin 64) (jj : Fin BV), base0 + a < d.K → BV * jv + jj < d.V →
          m'.read .bf16 (d.h.ptr ((hIdx d (cbase d n + t) hh (base0 + a) (BV * jv + jj) : Nat) : Int)).addr
            = some (x.get2 a jj)) →
        (∀ y, y ∉ d.h.elems (hBlk d n hh jv t base0) → m' y = m y) →
        (∀ y, m y ≠ none → m' y ≠ none) →
        Safe E R (W \ d.h.elems (hBlk d n hh jv t base0)) Q m' (k ())) :
    Safe E R W Q m (if base0 < d.K then (tl.cvt b).bind fun x =>
      (tl.store (hp + sc + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64) x
        (some ((m_k.col BV).and (m_v.row 64)))).bind k else k ()) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    have := safe_hstore hd hvf hhh hc hb64 hbase hhp hsc hok hmk hov hmv hvalid hW hlen (b := b)
      (k := k) hk
    exact this
  · simp only [hc, ite_false]
    have hE := hBlk_empty (d := d) (n := n) (hv := hh) (jv := jv) (t := t) hc
    have := hk (Tensor.map FloatV.cvt b) (fun a jj => FloatV.NaNEq.refl _) m
      (fun a jj ha => absurd (by omega) hc) (fun y _ => rfl) (fun y hy => hy)
    rw [hE, Set.sdiff_empty] at this
    exact this

/-- **The store of a state block** (`[V, K]` layout, source lines 179–185 with `STATE_V_FIRST`): the
rounded block, lanes `(jj, a)`, is stored to `h[chunk t][base0 + a, 32·jv + jj]`. -/
theorem safe_hstore_vf {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {n hh jv t base0 : Nat}
    (hvf : d.vf = true) (hhh : hh < d.HV) (hjv : jv < NV d) {hp : Ptr .bf16} {sc : I64}
    {o_k : Tensor [64] I32} {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hhp : hp = d.h.ptr (((cbase d n * d.HV + hh) * d.K * d.V : Nat) : Int))
    (hsc : sc.toInt = ((t * d.HV * d.K * d.V : Nat) : Int))
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hvalid : d.h.Valid m) (hW : d.h.elems (hBlk d n hh jv t base0) ⊆ W)
    (hlen : (cbase d n + t + 1) * d.HV * d.K * d.V ≤ d.h.len)
    {b : Tensor [BV, 64] F32} {k : Unit → Prog α}
    (hk : ∀ x : Tensor [BV, 64] BF16,
      (∀ jj a, FloatV.NaNEq (FloatV.cvt (b.get2 jj a)) (x.get2 jj a)) →
      ∀ m' : Mem,
        (∀ (jj : Fin BV) (a : Fin 64), base0 + a < d.K → BV * jv + jj < d.V →
          m'.read .bf16 (d.h.ptr ((hIdx d (cbase d n + t) hh (base0 + a) (BV * jv + jj) : Nat) : Int)).addr
            = some (x.get2 jj a)) →
        (∀ y, y ∉ d.h.elems (hBlk d n hh jv t base0) → m' y = m y) →
        (∀ y, m y ≠ none → m' y ≠ none) →
        Safe E R (W \ d.h.elems (hBlk d n hh jv t base0)) Q m' (k ())) :
    Safe E R W Q m ((tl.cvt b).bind fun x =>
      (tl.store (hp + sc + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV) x
        (some ((m_v.col 64).and (m_k.row BV)))).bind k) := by
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun x hcv => ?_
  have hK := hd.K_le; have hKp := hd.K_pos; have hVp := hd.V_pos
  have hKi : (IntV.ofNat d.K : I64).toInt = (d.K : Int) := IntV.I64.toInt_ofNat (by omega)
  set e : Nat := (cbase d n * d.HV + hh) * d.K * d.V + t * d.HV * d.K * d.V with he
  have hsp : hp + sc = d.h.ptr (e : Int) := by
    show hp.offset sc.toInt = _
    rw [hhp, hsc, Buf.ptr_offset, he]; push_cast; ring_nf
  have hjb : ∀ j : Fin BV, BV * jv + j < d.V + BV := by
    intro j; have := j.isLt; have := hjv; simp only [NV, BV] at *; omega
  have hx : ∀ (j : Fin BV) (a : Fin 64),
      ((o_v.col 64 * bc (IntV.ofNat d.K : I64)).get2 j a).toInt = (((BV * jv + j) * d.K : Nat) : Int) := by
    intro j a
    show (o_v.get1 j * (IntV.ofNat d.K : I64)).toInt = _
    refine IntV.mul_nat (hov j) hKi ?_
    have h1 : (BV * jv + j) * d.K ≤ (2 ^ 17) * 256 :=
      Nat.mul_le_mul (by have := hjb j; have := hd.V_lt; unfold BV at *; omega) hK
    exact lt_of_le_of_lt h1 (by norm_num)
  have hy : ∀ (j : Fin BV) (a : Fin 64), ((o_k.row BV).get2 j a).toInt = (((base0 + a) * 1 : Nat) : Int) := by
    intro j a
    show (o_k.get1 a).toInt = _
    rw [hok a]; simp
  have hptr : ∀ (j : Fin BV) (a : Fin 64), BV * jv + j < d.V → base0 + a < d.K →
      (hp + sc + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV).get2 j a =
        d.h.ptr ((e + (BV * jv + j) * d.K + (base0 + a) * 1 : Nat) : Int) := fun j a _ _ => by
    rw [hsp.symm.symm] at *
    exact ptr_lane_seq_of d.h hsp hx hy j a
  have hmask : MaskAt (some ((m_v.col 64).and (m_k.row BV))) (fun j : Fin BV => BV * jv + j)
      (fun a : Fin 64 => base0 + a) d.V d.K := by
    refine maskAt_of_get fun j a => ?_
    show (m_v.get1 j && m_k.get1 a) = _
    rw [hmv, hmk]
  have hset : {x | ElemAt e d.K 1 (fun j : Fin BV => BV * jv + j) (fun a : Fin 64 => base0 + a)
      d.V d.K x} = hBlk d n hh jv t base0 := by
    ext x
    constructor
    · rintro ⟨j, a, hj, ha, rfl⟩
      refine ⟨a, j, a.isLt, ha, j.isLt, ⟨by omega, by omega, hj⟩, ?_⟩
      simp only [hIdx, hvf, he, ite_true]
      ring
    · rintro ⟨a, j, ha64, ha, hj, ⟨-, -, hjv'⟩, rfl⟩
      refine ⟨⟨j, hj⟩, ⟨a, ha64⟩, hjv', ha, ?_⟩
      simp only [hIdx, hvf, he, ite_true]
      ring
  have hTile : d.h.tileAt e d.K 1 (fun j : Fin BV => BV * jv + j) (fun a : Fin 64 => base0 + a)
      d.V d.K = d.h.elems (hBlk d n hh jv t base0) := by
    unfold Buf.tileAt; rw [hset]
  refine safe_store_tileAt d.h (e := e) (s0 := d.K) (s1 := 1) (ρ := fun j : Fin BV => BV * jv + j)
    (κ := fun a : Fin 64 => base0 + a) (M := d.V) (N := d.K) hptr hmask ?_
    (inj_rowMajor1 le_rfl) ?_ ?_ hvalid (by rw [hTile]; exact hW) ?_
  · right; right
    have h1 : (cbase d n + t + 1) * d.HV * d.K * d.V = ((cbase d n + t) * d.HV + d.HV) * (d.V * d.K) := by ring
    have h2 : e = ((cbase d n + t) * d.HV + hh) * (d.V * d.K) := by simp only [he]; ring
    have h3 : ((cbase d n + t) * d.HV + hh + 1) * (d.V * d.K) ≤ ((cbase d n + t) * d.HV + d.HV) * (d.V * d.K) :=
      Nat.mul_le_mul_right _ (by omega)
    have h4 : (d.V - 1) * d.K + (d.K - 1) * 1 < d.V * d.K := by
      have : (d.V - 1) * d.K + d.K = d.V * d.K := by
        obtain ⟨k', hk'⟩ : ∃ k', d.V = k' + 1 := ⟨d.V - 1, by omega⟩
        rw [hk']; simp [Nat.succ_mul]
      omega
    nlinarith
  · intro i i' _ _ h; exact Fin.ext (by simpa using h)
  · intro j j' _ _ h; exact Fin.ext (by simpa using h)
  · intro m' hr hf hp'
    rw [hTile] at hf ⊢
    refine hk x (fun j a => hcv (j, a, ())) m' ?_ hf hp'
    intro j a ha hjj
    have := hr j a hjj ha
    convert this using 4
    simp only [hIdx, hvf, he, ite_true]
    congr 1
    ring

/-- **The store of an optional state block**, `[V, K]` layout. -/
theorem safe_hstore_vf_opt {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {n hh jv t base0 : Nat}
    (hvf : d.vf = true) (hhh : hh < d.HV) (hjv : jv < NV d) {hp : Ptr .bf16} {sc : I64}
    {o_k : Tensor [64] I32} {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hhp : hp = d.h.ptr (((cbase d n * d.HV + hh) * d.K * d.V : Nat) : Int))
    (hsc : sc.toInt = ((t * d.HV * d.K * d.V : Nat) : Int))
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hvalid : d.h.Valid m) (hW : d.h.elems (hBlk d n hh jv t base0) ⊆ W)
    (hlen : (cbase d n + t + 1) * d.HV * d.K * d.V ≤ d.h.len)
    {b : Tensor [BV, 64] F32} {k : Unit → Prog α}
    (hk : ∀ x : Tensor [BV, 64] BF16,
      (∀ jj a, FloatV.NaNEq (FloatV.cvt (b.get2 jj a)) (x.get2 jj a)) →
      ∀ m' : Mem,
        (∀ (jj : Fin BV) (a : Fin 64), base0 + a < d.K → BV * jv + jj < d.V →
          m'.read .bf16 (d.h.ptr ((hIdx d (cbase d n + t) hh (base0 + a) (BV * jv + jj) : Nat) : Int)).addr
            = some (x.get2 jj a)) →
        (∀ y, y ∉ d.h.elems (hBlk d n hh jv t base0) → m' y = m y) →
        (∀ y, m y ≠ none → m' y ≠ none) →
        Safe E R (W \ d.h.elems (hBlk d n hh jv t base0)) Q m' (k ())) :
    Safe E R W Q m (if base0 < d.K then (tl.cvt b).bind fun x =>
      (tl.store (hp + sc + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV) x
        (some ((m_v.col 64).and (m_k.row BV)))).bind k else k ()) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    have := safe_hstore_vf hd hvf hhh hjv hhp hsc hok hmk hov hmv hvalid hW hlen (b := b)
      (k := k) hk
    exact this
  · simp only [hc, ite_false]
    have hE := hBlk_empty (d := d) (n := n) (hv := hh) (jv := jv) (t := t) hc
    have := hk (Tensor.map FloatV.cvt b) (fun a jj => FloatV.NaNEq.refl _) m
      (fun a jj ha => absurd (by omega) hc) (fun y _ => rfl) (fun y hy => hy)
    rw [hE, Set.sdiff_empty] at this
    exact this

/-- **An optional `w` block** (`if base0 < K:` around the load and the dot): as `safe_wblock`, or the accumulator unchanged. -/
theorem safe_wblock_opt {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRw : d.w.bytes ⊆ R)
    (hvw : d.w.Valid m) {n hv t base0 : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    {wp : Ptr .bf16} (hwp : wp = d.w.ptr (((bosOf d n * d.HV + hv) * d.K : Nat) : Int))
    {o_t : Tensor [BT] I64} (hot : ∀ i : Fin BT, (o_t.get1 i).toInt = ((BT * t + i : Nat) : Int))
    {T : I64} (hT : T.toInt = (lenOf d n : Int)) {o_k : Tensor [64] I32} {m_k : Tensor [64] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hlen : (bosOf d n + lenOf d n) * d.HV * d.K ≤ d.w.len)
    (xb : Tensor [64, BV] F32) (acc : Tensor [BT, BV] F32) {k : Tensor [BT, BV] F32 → Prog α}
    (hk : ∀ r, (base0 < d.K ∧ WFacts d E.P init n hv t base0 xb acc r) ∨ (¬ base0 < d.K ∧ r = acc) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (if base0 < d.K then (tl.convert .f32 .bf16 (tl.full [BT, 64] zf32)).bind fun z =>
      (tl.load (wp + o_t.col 64 * bc (IntV.ofNat (d.HV * d.K) : I64) + o_k.row BT)
        (some (((o_t.ltS T).col 64).and (m_k.row BT))) (some z)).bind fun b_w =>
      (tl.cvt xb).bind fun x => (tl.dot cfgBF16 b_w x acc).bind k else k acc) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    exact safe_wblock hd hP hmi hRw hvw hn hhv ht hwp hot hT hok hmk hlen xb acc fun r hr => hk r (Or.inl ⟨hc, hr⟩)
  · simp only [hc, ite_false]
    exact hk acc (Or.inr ⟨hc, rfl⟩)

/-- **An optional `k` block**: as `safe_kdot`, or the accumulator unchanged. -/
theorem safe_kdot_opt {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRk : d.k.bytes ⊆ R)
    (hvk : d.k.Valid m) {n hv t base0 : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    {kp : Ptr .bf16} (hkp : kp = d.k.ptr (((bosOf d n * d.H + kHead d hv) * d.K : Nat) : Int))
    {o_t : Tensor [BT] I64} (hot : ∀ i : Fin BT, (o_t.get1 i).toInt = ((BT * t + i : Nat) : Int))
    {T : I64} (hT : T.toInt = (lenOf d n : Int)) {o_k : Tensor [64] I32} {m_k : Tensor [64] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hlen : (bosOf d n + lenOf d n) * d.H * d.K ≤ d.k.len)
    (vb : Tensor [BT, BV] BF16) (acc : Tensor [64, BV] F32) {k : Tensor [64, BV] F32 → Prog α}
    (hk : ∀ r, (base0 < d.K ∧ KFacts d E.P init n hv t base0 vb acc r) ∨ (¬ base0 < d.K ∧ r = acc) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (if base0 < d.K then (tl.convert .f32 .bf16 (tl.full [64, BT] zf32)).bind fun z =>
      (tl.load (kp + o_k.col BT + o_t.row 64 * bc (IntV.ofNat (d.H * d.K) : I64))
        (some ((m_k.col BT).and ((o_t.ltS T).row 64))) (some z)).bind fun b_k =>
      (tl.dot cfgBF16 b_k vb acc).bind k else k acc) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    exact safe_kdot hd hP hmi hRk hvk hn hhv ht hkp hot hT hok hmk hlen vb acc fun r hr => hk r (Or.inl ⟨hc, hr⟩)
  · simp only [hc, ite_false]
    exact hk acc (Or.inr ⟨hc, rfl⟩)

/-- **An optional state block's decay** (`if 64 < K: b_h2 *= g`): as `safe_scale`, or the block unchanged. -/
theorem safe_scale_opt {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}
    {c : Prop} [Decidable c] (b : Tensor s F32) (g : F32) {k : Tensor s F32 → Prog α}
    (hk : ∀ r : Tensor s F32, (c ∧ ∀ i, FloatV.NaNEq (FloatV.mul (b.get i) g) (r.get i)) ∨
      (¬ c ∧ r = b) → Safe E R W Q m (k r)) :
    Safe E R W Q m (if c then (tl.mul b (bc g)).bind k else k b) := by
  by_cases hc : c
  · simp only [hc, ite_true]
    exact safe_scale b g fun r hr => hk r (Or.inl ⟨hc, hr⟩)
  · simp only [hc, ite_false]
    exact hk b (Or.inr ⟨hc, rfl⟩)

/-- What the `[V, K]` update of a block establishes: the loaded tile `bk`, the dot `d_h` of it with the
rounded gated values into zeros, and the fusable sum `r = b_h + d_hᵀ`. -/
def KUpdFacts (d : Inst) (P : Platform) (fusion : Bool) (init : Mem) (n hv t base0 : Nat)
    (vb : Tensor [BT, BV] BF16) (bh r : Tensor [BV, 64] F32) : Prop :=
  ∃ bk, KTile d init n hv t base0 bk ∧ ∃ d_h : Tensor [64, BV] F32,
    P.dot cfgBF16 bk vb (tl.zeros [64, BV] .f32) d_h ∧
    ∀ (jj : Fin BV) (a : Fin 64), AddRel fusion (bh.get2 jj a) (d_h.get2 a jj) (r.get2 jj a)

/-- **An optional `[V, K]` update**: the `k` load, the dot into zeros and the fusable sum into the
state block, or the block unchanged. -/
theorem safe_kupd_vf_opt {d : Inst} (hd : d.ok) {E : Env} (hP : Device E.P) {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRk : d.k.bytes ⊆ R)
    (hvk : d.k.Valid m) {n hv t base0 : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    {kp : Ptr .bf16} (hkp : kp = d.k.ptr (((bosOf d n * d.H + kHead d hv) * d.K : Nat) : Int))
    {o_t : Tensor [BT] I64} (hot : ∀ i : Fin BT, (o_t.get1 i).toInt = ((BT * t + i : Nat) : Int))
    {T : I64} (hT : T.toInt = (lenOf d n : Int)) {o_k : Tensor [64] I32} {m_k : Tensor [64] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hlen : (bosOf d n + lenOf d n) * d.H * d.K ≤ d.k.len)
    (vb : Tensor [BT, BV] BF16) (bh : Tensor [BV, 64] F32) {k : Tensor [BV, 64] F32 → Prog α}
    (hk : ∀ r, (base0 < d.K ∧ KUpdFacts d E.P E.fusion init n hv t base0 vb bh r) ∨ (¬ base0 < d.K ∧ r = bh) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (if base0 < d.K then (tl.convert .f32 .bf16 (tl.full [64, BT] zf32)).bind fun z =>
      (tl.load (kp + o_k.col BT + o_t.row 64 * bc (IntV.ofNat (d.H * d.K) : I64))
        (some ((m_k.col BT).and ((o_t.ltS T).row 64))) (some z)).bind fun b_k =>
      (tl.dot cfgBF16 b_k vb (tl.zeros [64, BV] .f32)).bind fun d_h =>
        (tl.addc bh d_h.trans).bind k else k bh) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    refine safe_kdot hd hP hmi hRk hvk hn hhv ht hkp hot hT hok hmk hlen vb _ fun d_h hdh => ?_
    obtain ⟨bk, hbk, hdot⟩ := hdh
    refine safe_addc fun r hr => hk r (Or.inl ⟨hc, bk, hbk, d_h, hdot, fun jj a => ?_⟩)
    exact hr (jj, a, ())
  · simp only [hc, ite_false]
    exact hk bh (Or.inr ⟨hc, rfl⟩)

end Claims.ChunkDeltaH
