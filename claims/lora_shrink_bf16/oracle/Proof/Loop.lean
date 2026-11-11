import Proof.Defs
import Proof.Facts
import Proof.Steps
import Proof.Trip
import Proof.Wrap

/-!
# `lora_shrink_bf16`: the loop over `K` (lines 136-180)

The loop state is the accumulator and the two pointer tiles. `LoopInv` says, after `t` trips, where
the pointer tiles point and that every accumulator lane of a row of the program is a `LaneInv`
value. Each of the three kinds of trip (a whole block, a partial block with masked loads, a
skipped block) preserves it.
-/

namespace Claims.LoraShrink

open TVA

/-- The loop state: the accumulator, the pointers of the `inputs` tile and of the weights tile. -/
abbrev LState (d : Inst) : Type :=
  Tensor [32, 16] F32 × Tensor [32, bk d] (Ptr .bf16) × Tensor [bk d, 16] (Ptr .bf16)

/-- The element of `inputs` that lane `(r, kk)` of the `inputs` tile points at in trip `t`:
row `r % cta_m_len` of the program's tile, `k = t · STEP_K + pid_sk · BLOCK_K + kk`. -/
def aIdx (d : Inst) (i r kk t : Nat) : Nat :=
  tokOf d i (r % mlen d i) * d.K + (t * (bk d * sk d) + part d i * bk d + kk)

/-- The element of the weights that lane `(kk, n)` of the weights tile points at in trip `t`. -/
def bIdx (d : Inst) (i kk n t : Nat) : Nat :=
  (idAt d (grpOf d i)).toNat * (d.R * d.K) + (16 * tileN d i + n) * d.K +
    (t * (bk d * sk d) + part d i * bk d + kk)

/-- **The loop invariant** after `t` trips: the memory is the initial one; the pointer tiles point
at the elements of trip `t`; every accumulator lane `(r, n)` of a row `r < cta_m_len` is a
`LaneInv` value of part `p` after `min t (stepsOf p)` trips, whenever the data are `Dat`. -/
def LoopInv (d : Inst) (init : Mem) (i t : Nat) (st : LState d) (m : Mem) : Prop :=
  m = init ∧
    (∀ (r : Fin 32) (kk : Fin (bk d)),
      st.2.1.get2 r kk = d.inp.ptr ((aIdx d i r kk t : Nat) : Int)) ∧
    (∀ (kk : Fin (bk d)) (n : Fin 16),
      st.2.2.get2 kk n = (d.lora (slice d i)).ptr ((bIdx d i kk n t : Nat) : Int)) ∧
    ∀ (r : Fin 32) (n : Fin 16), (r : Nat) < mlen d i → ∀ va vb : Nat → BF16,
      Dat d init (slice d i) (tokOf d i r) (16 * tileN d i + n) va vb →
        LaneInv d va vb (part d i) (min t (stepsOf d (part d i))) (st.1.get2 r n)

/-- The end of a trip: the new accumulator and both pointer tiles moved on by one stride
(lines 179-180). -/
def advance (d : Inst) (acc : Tensor [32, 16] F32) (st : LState d) : LState d :=
  (acc, st.2.1 + (bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [32, bk d] I32),
    st.2.2 + (bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [bk d, 16] I32))

/-- A trip over a whole block: unmasked loads and `acc += tl.dot(a, b)` (lines 146-152). -/
def blockFull (d : Inst) (st : LState d) : Prog (LState d) := do
  let tiled_b ← tl.load st.2.2
  let tiled_a ← tl.load st.2.1
  let accumulator ← tl.dot_add dotCfg tiled_a tiled_b st.1 (Tensor.full (DType.f32.ofBits 0))
  pure (advance d accumulator st)

/-- A trip over a partial block: masked loads of the `k` below `K`, the rest `+0.0` (lines 169-177);
`base` is `iter_k`. -/
def blockMasked (d : Inst) (base : I32) (st : LState d) : Prog (LState d) := do
  let o1 ← tl.convert DType.f32 DType.bf16
    (tl.full [bk d, 16] ((FloatV.ofPat 0 : F64)).litF32)
  let tiled_b ← tl.load st.2.2
    (some (((base + tl.arange 0 (bk d)).ltS (IntV.ofNat d.K)).col 16)) (some o1)
  let o2 ← tl.convert DType.f32 DType.bf16
    (tl.full [32, bk d] ((FloatV.ofPat 0 : F64)).litF32)
  let tiled_a ← tl.load st.2.1
    (some (((base + tl.arange 0 (bk d)).ltS (IntV.ofNat d.K)).row 32)) (some o2)
  let accumulator ← tl.dot_add dotCfg tiled_a tiled_b st.1 (Tensor.full (DType.f32.ofBits 0))
  pure (advance d accumulator st)

/-- The constant stride added to a pointer tile fits `int32`. -/
theorem stride_lt (d : Inst) : bk d * sk d * 1 < 2 ^ 31 := by
  rcases bk_sk_cases d with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2] <;> norm_num

/-- The `inputs` pointer tile one stride on points at the next trip's elements. -/
theorem advance_a {d : Inst} {i t : Nat} {st : LState d}
    (h : ∀ (r : Fin 32) (kk : Fin (bk d)), st.2.1.get2 r kk = d.inp.ptr ((aIdx d i r kk t : Nat) : Int))
    (acc : Tensor [32, 16] F32) :
    ∀ (r : Fin 32) (kk : Fin (bk d)),
      (advance d acc st).2.1.get2 r kk = d.inp.ptr ((aIdx d i r kk (t + 1) : Nat) : Int) := by
  intro r kk
  have e : ((bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [32, bk d] I32).get2 r kk).toInt =
      ((bk d * sk d * 1 : Nat) : Int) := IntV.I32.toInt_ofNat (stride_lt d)
  show (st.2.1 + (bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [32, bk d] I32)).get2 r kk = _
  rw [Tensor.get2_ptrAddT, h r kk, e, Buf.ptr_offset]
  congr 1
  unfold aIdx
  push_cast
  ring

/-- The weights pointer tile one stride on points at the next trip's elements. -/
theorem advance_b {d : Inst} {i t : Nat} {st : LState d}
    (h : ∀ (kk : Fin (bk d)) (n : Fin 16),
      st.2.2.get2 kk n = (d.lora (slice d i)).ptr ((bIdx d i kk n t : Nat) : Int))
    (acc : Tensor [32, 16] F32) :
    ∀ (kk : Fin (bk d)) (n : Fin 16),
      (advance d acc st).2.2.get2 kk n = (d.lora (slice d i)).ptr ((bIdx d i kk n (t + 1) : Nat) : Int) := by
  intro kk n
  have e : ((bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [bk d, 16] I32).get2 kk n).toInt =
      ((bk d * sk d * 1 : Nat) : Int) := IntV.I32.toInt_ofNat (stride_lt d)
  show (st.2.2 + (bc (IntV.ofNat (bk d * sk d * 1) : I32) : Tensor [bk d, 16] I32)).get2 kk n = _
  rw [Tensor.get2_ptrAddT, h kk n, e, Buf.ptr_offset]
  congr 1
  unfold bIdx
  push_cast
  ring

section Steps

variable {P : Platform} {d : Inst} {init : Mem} {i : Nat}

/-- **A skipped trip** (`iter_k ≥ K`, lines 155-157): the accumulator stays, the pointers advance.
The invariant holds one trip later because `stepsOf p` trips are already done. -/
theorem step_skip {t : Nat}
    (ht : stepsOf d (part d i) ≤ t) {st : LState d} (hI : LoopInv d init i t st init) :
    Safe ((launch d).env P) (Rall d) ∅ (fun st' m' => LoopInv d init i (t + 1) st' m') init
      (pure (advance d st.1 st)) := by
  obtain ⟨-, hpa, hpb, hacc⟩ := hI
  refine safe_ret ⟨rfl, advance_a hpa _, advance_b hpb _, ?_⟩
  intro r n hr va vb hdat
  have h1 : min (t + 1) (stepsOf d (part d i)) = min t (stepsOf d (part d i)) := by omega
  rw [h1]
  exact hacc r n hr va vb hdat

/-- Row `r` of the tile, for `r < cta_m_len`, reads its own token. -/
theorem aIdx_row {d : Inst} {i : Nat} {r : Nat} (hr : r < mlen d i) (kk t : Nat) :
    aIdx d i r kk t = tokOf d i r * d.K + (t * (bk d * sk d) + part d i * bk d + kk) := by
  unfold aIdx
  rw [Nat.mod_eq_of_lt hr]

/-- **The invariant after a trip**, from the two loaded tiles and the dot: the pointers moved on, and
each accumulator lane of a row of the program is a `LaneInv` value one trip later (`lane_trip`).
Lane `kk` of a tile row/column holds the value read from the initial memory when its `k` is below
`K`, and `+0.0` otherwise. -/
theorem trip_inv (hP : Device P) (hd : d.ok) (hact : Active d i) (hp : part d i < sk d) {t : Nat}
    (ht : t < stepsOf d (part d i)) {st : LState d} (hI : LoopInv d init i t st init)
    {ta : Tensor [32, bk d] BF16} {tb : Tensor [bk d, 16] BF16} {r : Tensor [32, 16] F32}
    (hr : P.dot dotCfg ta tb st.1 r ∨ ∃ dd, P.dot dotCfg ta tb (Tensor.full (DType.f32.ofBits 0)) dd ∧
      ∀ ix, AccAdd (launch d).fusion .f32 (dd.get ix) (st.1.get ix) (r.get ix))
    (hA : ∀ (r' : Fin 32) (kk : Fin (bk d)),
      ((part d i + sk d * t) * bk d + kk < d.K →
        init.read .bf16 (d.inp.ptr ((aIdx d i r' kk t : Nat) : Int)).addr = some (ta.get2 r' kk)) ∧
      (¬ ((part d i + sk d * t) * bk d + kk < d.K) → ta.get2 r' kk = FloatV.ofPat 0))
    (hB : ∀ (kk : Fin (bk d)) (n : Fin 16),
      ((part d i + sk d * t) * bk d + kk < d.K →
        init.read .bf16 ((d.lora (slice d i)).ptr ((bIdx d i kk n t : Nat) : Int)).addr =
          some (tb.get2 kk n)) ∧
      (¬ ((part d i + sk d * t) * bk d + kk < d.K) → tb.get2 kk n = FloatV.ofPat 0)) :
    LoopInv d init i (t + 1) (advance d r st) init := by
  obtain ⟨-, hpa, hpb, hacc⟩ := hI
  refine ⟨rfl, advance_a hpa _, advance_b hpb _, ?_⟩
  intro r' n' hr' va vb hdat
  obtain ⟨hadapt, hra, hrb, hdm, hroom⟩ := hdat
  have hdat : Dat d init (slice d i) (tokOf d i r') (16 * tileN d i + n') va vb :=
    ⟨hadapt, hra, hrb, hdm, hroom⟩
  have hrow := hact.row hd hr'
  have hacc' := hacc r' n' hr' va vb hdat
  have hmin1 : min t (stepsOf d (part d i)) = t := by omega
  rw [hmin1] at hacc'
  have hmin2 : min (t + 1) (stepsOf d (part d i)) = t + 1 := by omega
  show LaneInv d va vb (part d i) (min (t + 1) (stepsOf d (part d i))) (r.get2 r' n')
  rw [hmin2]
  refine lane_trip hP (fusion := (launch d).fusion) d hd va vb (fun k hk => (hra k hk).2)
    (fun k hk => (hrb k hk).2) hdm (part d i) t hp ht ta tb st.1 (Tensor.full (DType.f32.ofBits 0)) r
    r' n' hr DType.val?_zero_f32 ?_ ?_ hacc'
  · intro kk
    by_cases hlt : (part d i + sk d * t) * bk d + kk < d.K
    · have hidx : aIdx d i r' kk t = tokOf d i r' * d.K + ((part d i + sk d * t) * bk d + kk) := by
        rw [aIdx_row hr']; ring
      have h1 := (hA r' kk).1 hlt
      rw [hidx] at h1
      have h2 := (hra ((part d i + sk d * t) * bk d + kk) hlt).1
      simp only [hlt, ↓reduceIte]
      exact Option.some.inj (h1.symm.trans h2)
    · simp only [hlt, ↓reduceIte]
      exact (hA r' kk).2 hlt
  · intro kk
    by_cases hlt : (part d i + sk d * t) * bk d + kk < d.K
    · have hidx : bIdx d i kk n' t = (d.adapter (tokOf d i r')).toNat * (d.R * d.K) +
          (16 * tileN d i + n') * d.K + ((part d i + sk d * t) * bk d + kk) := by
        unfold bIdx; rw [hrow.2.1]; ring
      have h1 := (hB kk n').1 hlt
      rw [hidx] at h1
      have h2 := (hrb ((part d i + sk d * t) * bk d + kk) hlt).1
      simp only [hlt, ↓reduceIte]
      exact Option.some.inj (h1.symm.trans h2)
    · simp only [hlt, ↓reduceIte]
      exact (hB kk n').2 hlt

/-- **A trip over a whole block** (`EVEN_K`, or the block ends by `K`): the invariant holds one
trip later. -/
theorem step_full (hP : Device P) (hd : d.ok) (hpre : Pre d init) (hi : i < (launch d).size)
    (hact : Active d i) (hs : slice d i < d.S) (hp : part d i < sk d) {t : Nat}
    (ht : t < stepsOf d (part d i)) (hfull : (part d i + sk d * t + 1) * bk d ≤ d.K)
    {st : LState d} (hI : LoopInv d init i t st init) :
    Safe ((launch d).env P) (Rall d) ∅ (fun st' m' => LoopInv d init i (t + 1) st' m') init
      (blockFull d st) := by
  have hI' := hI
  obtain ⟨-, hpa, hpb, -⟩ := hI
  obtain ⟨hinpV, hloraV, -⟩ := hpre
  have hmlen := hact.mlen_pos
  have hbk := bk_pos d
  have hkk : ∀ kk : Nat, kk < bk d → t * (bk d * sk d) + part d i * bk d + kk < d.K := by
    intro kk hk
    have h2 : (part d i + sk d * t) * bk d + kk < d.K := by nlinarith
    have h3 : (part d i + sk d * t) * bk d = t * (bk d * sk d) + part d i * bk d := by ring
    omega
  have hlidx := hact.lora_idx hd
  have hlt : ∀ kk : Nat, kk < bk d → (part d i + sk d * t) * bk d + kk < d.K := by
    intro kk hk
    have := hkk kk hk
    have h3 : (part d i + sk d * t) * bk d = t * (bk d * sk d) + part d i * bk d := by ring
    omega
  unfold blockFull
  refine safe_load_buf (d.lora (slice d i))
    (g := fun idx : Idx [bk d, 16] => bIdx d i idx.1 idx.2.1 t) (en := fun _ => true)
    (Buf.addrs_of_get_all _ ?_) ?_ (hloraV _ hs) ?_ fun tb hb1 _ => ?_
  · rintro ⟨kk, n, u⟩
    exact hpb kk n
  · rintro ⟨kk, n, u⟩ _
    exact hd.lora_len _ _ _ _ hs hlidx.1 (tileN_lt hd hi n.isLt) (hkk kk kk.isLt)
  · refine lora_elems_sub hs ?_
    rintro j ⟨⟨kk, n, u⟩, _, rfl⟩
    exact hd.lora_len _ _ _ _ hs hlidx.1 (tileN_lt hd hi n.isLt) (hkk kk kk.isLt)
  refine safe_load_buf d.inp (g := fun idx : Idx [32, bk d] => aIdx d i idx.1 idx.2.1 t)
    (en := fun _ => true) (Buf.addrs_of_get_all _ ?_) ?_ hinpV ?_ fun ta ha1 _ => ?_
  · rintro ⟨r, kk, u⟩
    exact hpa r kk
  · rintro ⟨r, kk, u⟩ _
    have hrow := (hact.row hd (Nat.mod_lt (r : Nat) hmlen)).1
    have := hd.inp_len _ _ hrow (hkk kk kk.isLt)
    exact this
  · refine inp_elems_sub ?_
    rintro j ⟨⟨r, kk, u⟩, _, rfl⟩
    have hrow := (hact.row hd (Nat.mod_lt (r : Nat) hmlen)).1
    exact hd.inp_len _ _ hrow (hkk kk kk.isLt)
  refine safe_dot_add (hP.dotTotal _ _ _ _) fun r hr => safe_ret ?_
  exact trip_inv hP hd hact hp ht hI' hr
    (fun r' kk => ⟨fun _ => ha1 (r', kk, ()) rfl, fun h => absurd (hlt kk kk.isLt) h⟩)
    (fun kk n => ⟨fun _ => hb1 (kk, n, ()) rfl, fun h => absurd (hlt kk kk.isLt) h⟩)

/-- The converted `+0.0` filler of a masked load (line 170) is `+0.0` in `bfloat16`. -/
theorem conv_zero_bf16 {P : Platform} {r : BF16}
    (h : ConvRel P .f32 .bf16 ((FloatV.ofPat 0 : F64)).litF32 r) : r = FloatV.ofPat 0 := by
  rw [FloatV.litF32_ofPat_zero] at h
  have h1 := ConvRel.f32_bf16_iff.mp h
  rw [FloatV.cvt_ofPat_zero Format.binary32_regular] at h1
  exact FloatV.NaNEq.eq h1 (FloatV.isNaN_ofPat_zero Format.bfloat16_regular)

/-- The filler conversion has a result. -/
theorem conv_zero_ex {P : Platform} :
    ∃ r : BF16, ConvRel P .f32 .bf16 ((FloatV.ofPat 0 : F64)).litF32 r :=
  ⟨FloatV.cvt _, ConvRel.f32_bf16_iff.mpr (FloatV.NaNEq.refl _)⟩

/-- **A trip over a partial block** (`iter_k < K < block_end`): the invariant holds one trip later.
`base` is `iter_k`. -/
theorem step_masked (hP : Device P) (hd : d.ok) (hpre : Pre d init) (hi : i < (launch d).size)
    (hact : Active d i) (hs : slice d i < d.S) (hp : part d i < sk d) {t : Nat}
    (ht : t < stepsOf d (part d i)) (_hpart : d.K < (part d i + sk d * t + 1) * bk d)
    {base : I32} (hbase : base.toInt = ((t * (bk d * sk d) + part d i * bk d : Nat) : Int))
    {st : LState d} (hI : LoopInv d init i t st init) :
    Safe ((launch d).env P) (Rall d) ∅ (fun st' m' => LoopInv d init i (t + 1) st' m') init
      (blockMasked d base st) := by
  have hI' := hI
  obtain ⟨-, hpa, hpb, -⟩ := hI
  obtain ⟨hinpV, hloraV, -⟩ := hpre
  have hmlen := hact.mlen_pos
  have hbk := bk_pos d
  have hlidx := hact.lora_idx hd
  have hK := hd.K_top
  have hblk : (part d i + sk d * t) * bk d = t * (bk d * sk d) + part d i * bk d := by ring
  have hN : t * (bk d * sk d) + part d i * bk d < d.K := by
    have h := (lt_blocks_iff d _).mp ((lt_stepsOf_iff d hp t).mp ht)
    omega
  have hbk31 : bk d ≤ 256 := by rcases bk_sk_cases d with h | h <;> omega
  have hmask : ∀ kk : Fin (bk d),
      ((base + tl.arange 0 (bk d)).ltS (IntV.ofNat d.K : I32)).get1 kk =
        decide ((part d i + sk d * t) * bk d + kk < d.K) := by
    intro kk
    have hkk := kk.isLt
    have hv : (base + (tl.arange 0 (bk d)).get1 kk).toInt =
        ((t * (bk d * sk d) + part d i * bk d + kk : Nat) : Int) :=
      IntV.I32.add_nat hbase (IntV.I32.toInt_tl_arange (n := bk d) (by omega) kk) (by omega)
    rw [Tensor.get1_ltS, Tensor.get1_sadd, IntV.lt_eq, hv, IntV.I32.toInt_ofNat (by omega)]
    simp only [hblk]
    exact decide_eq_decide.mpr Int.ofNat_lt
  unfold blockMasked
  refine safe_convert (d := .f32) (d' := .bf16) (fun _ => conv_zero_ex) fun o1 ho1 => ?_
  have ho1' : ∀ idx, o1.get idx = FloatV.ofPat 0 := fun idx => conv_zero_bf16 (ho1 idx)
  refine safe_load_buf (d.lora (slice d i))
    (g := fun idx : Idx [bk d, 16] => bIdx d i idx.1 idx.2.1 t)
    (en := fun idx : Idx [bk d, 16] => decide ((part d i + sk d * t) * bk d + idx.1 < d.K))
    (Buf.addrs_of_get _ ?_ ?_) ?_ (hloraV _ hs) ?_ fun tb hb1 hb2 => ?_
  · rintro ⟨kk, n, u⟩
    exact hmask kk
  · rintro ⟨kk, n, u⟩ _
    exact hpb kk n
  · rintro ⟨kk, n, u⟩ hen
    have hlt := of_decide_eq_true hen
    exact hd.lora_len _ _ _ _ hs hlidx.1 (tileN_lt hd hi n.isLt) (by omega)
  · refine lora_elems_sub hs ?_
    rintro j ⟨⟨kk, n, u⟩, hen, rfl⟩
    have hlt := of_decide_eq_true hen
    exact hd.lora_len _ _ _ _ hs hlidx.1 (tileN_lt hd hi n.isLt) (by omega)
  refine safe_convert (d := .f32) (d' := .bf16) (fun _ => conv_zero_ex) fun o2 ho2 => ?_
  have ho2' : ∀ idx, o2.get idx = FloatV.ofPat 0 := fun idx => conv_zero_bf16 (ho2 idx)
  refine safe_load_buf d.inp (g := fun idx : Idx [32, bk d] => aIdx d i idx.1 idx.2.1 t)
    (en := fun idx : Idx [32, bk d] => decide ((part d i + sk d * t) * bk d + idx.2.1 < d.K))
    (Buf.addrs_of_get _ ?_ ?_) ?_ hinpV ?_ fun ta ha1 ha2 => ?_
  · rintro ⟨r, kk, u⟩
    exact hmask kk
  · rintro ⟨r, kk, u⟩ _
    exact hpa r kk
  · rintro ⟨r, kk, u⟩ hen
    have hlt := of_decide_eq_true hen
    have hrow := (hact.row hd (Nat.mod_lt (r : Nat) hmlen)).1
    exact hd.inp_len _ _ hrow (by omega)
  · refine inp_elems_sub ?_
    rintro j ⟨⟨r, kk, u⟩, hen, rfl⟩
    have hlt := of_decide_eq_true hen
    have hrow := (hact.row hd (Nat.mod_lt (r : Nat) hmlen)).1
    exact hd.inp_len _ _ hrow (by omega)
  refine safe_dot_add (hP.dotTotal _ _ _ _) fun r hr => safe_ret ?_
  refine trip_inv hP hd hact hp ht hI' hr ?_ ?_
  · intro r' kk
    refine ⟨fun hlt => ha1 (r', kk, ()) (by simp [hlt]), fun hnlt => ?_⟩
    have := ha2 (r', kk, ()) (by simp [hnlt]) o2 rfl
    exact this.trans (ho2' _)
  · intro kk n
    refine ⟨fun hlt => hb1 (kk, n, ()) (by simp [hlt]), fun hnlt => ?_⟩
    have := hb2 (kk, n, ()) (by simp [hnlt]) o1 rfl
    exact this.trans (ho1' _)

end Steps

end Claims.LoraShrink
