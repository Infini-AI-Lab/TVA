import Proof.Inv

/-!
# `chunk_delta_h_bf16`: lanes, pointers and masks of the tiles

Rules for the segments of a loop trip, one per source line group. Each is `Safe` of a fragment of
`chunk_step` followed by an arbitrary rest `k`; its continuation hypothesis receives what the
fragment established. The tile pointers are `base + (row term) + (column term)`; the lane
hypotheses (`o_k`, `m_k`, `o_v`, `m_v`, ...) are the values the program computed before the trip.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA TVA.Tile2

/-- The elements of `h` that block `base0` (state rows `base0 + a`, `a < 64`, `a + base0 < K`) of
chunk `t` stores, in the columns of block `jv`. -/
def hBlk (d : Inst) (n hv jv t base0 : Nat) : Set Nat :=
  {x | ∃ a jj, a < 64 ∧ base0 + a < d.K ∧ jj < BV ∧ inBlk d jv (BV * jv + jj) ∧
    x = hIdx d (cbase d n + t) hv (base0 + a) (BV * jv + jj)}

/-- The elements of `h` stored by block `base0` are cells of chunk `t`. -/
theorem hBlk_sub_hCells {d : Inst} {n hv jv t base0 : Nat} :
    hBlk d n hv jv t base0 ⊆ hCells d n hv jv t := by
  rintro x ⟨a, jj, -, ha, -, hj, rfl⟩
  exact ⟨base0 + a, BV * jv + jj, ha, hj, rfl⟩

/-- **The store of a state block** (`[K, V]` layout, source lines 179–185): the rounded block is
stored to the tile `h[chunk t][base0 + a, 32·jv + jj]`. Afterwards every enabled lane's cell holds
its lane of the rounded tensor `x`, nothing else changed and every present byte is present. -/
theorem safe_hstore {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {n hh jv t base0 : Nat}
    (hvf : d.vf = false) (hhh : hh < d.HV) (_hb0 : base0 < d.K) (hb64 : base0 % 64 = 0)
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
    Safe E R W Q m (tl.cvt b >>= fun x =>
      tl.store (hp + sc + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64) x
        (some ((m_k.col BV).and (m_v.row 64))) >>= k) := by
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun x hcv => ?_
  have hV31 : d.V < 2 ^ 31 := lt_of_lt_of_le hd.V_lt (by norm_num)
  have hK := hd.K_le
  have hVi : (IntV.ofNat d.V : I32).toInt = (d.V : Int) := IntV.I32.toInt_ofNat hV31
  set e : Nat := (cbase d n * d.HV + hh) * d.K * d.V + t * d.HV * d.K * d.V with he
  have hsp : hp + sc = d.h.ptr (e : Int) := by
    show hp.offset sc.toInt = _
    rw [hhp, hsc, Buf.ptr_offset, he]; push_cast; ring_nf
  have hx : ∀ (i : Fin 64) (j : Fin BV),
      ((o_k.col BV * bc (IntV.ofNat d.V : I32)).get2 i j).toInt = (((base0 + i) * d.V : Nat) : Int) := by
    intro i j
    show (o_k.get1 i * (IntV.ofNat d.V : I32)).toInt = _
    refine IntV.I32.mul_nat (hok i) hVi ?_
    have := i.isLt
    have h1 : (base0 + i) * d.V ≤ 255 * d.V := Nat.mul_le_mul_right _ (by omega)
    have h2 : d.V < 2 ^ 16 := hd.V_lt
    omega
  have hy : ∀ (i : Fin 64) (j : Fin BV),
      ((o_v.row 64).get2 i j).toInt = (((BV * jv + j) * 1 : Nat) : Int) := by
    intro i j
    show (o_v.get1 j).toInt = _
    rw [hov j]; simp
  have hptr : ∀ (i : Fin 64) (j : Fin BV), base0 + i < d.K → BV * jv + j < d.V →
      (hp + sc + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64).get2 i j =
        d.h.ptr ((e + (base0 + i) * d.V + (BV * jv + j) * 1 : Nat) : Int) := fun i j _ _ =>
    ptr_lane_seq_of d.h hsp hx hy i j
  have hmask : MaskAt (some ((m_k.col BV).and (m_v.row 64))) (fun a : Fin 64 => base0 + a)
      (fun jj : Fin BV => BV * jv + jj) d.K d.V := by
    refine maskAt_of_get fun i j => ?_
    show (m_k.get1 i && m_v.get1 j) = _
    rw [hmk, hmv]
  have hset : {x | ElemAt e d.V 1 (fun a : Fin 64 => base0 + a) (fun jj : Fin BV => BV * jv + jj)
      d.K d.V x} = hBlk d n hh jv t base0 := by
    ext x
    constructor
    · rintro ⟨i, j, hi, hj, rfl⟩
      refine ⟨i, j, i.isLt, hi, j.isLt, ⟨by omega, by omega, hj⟩, ?_⟩
      simp only [hIdx, hvf, he, Bool.false_eq_true, ite_false]
      ring
    · rintro ⟨i, j, hi, ha, hj, ⟨-, -, hjv⟩, rfl⟩
      refine ⟨⟨i, hi⟩, ⟨j, hj⟩, ha, hjv, ?_⟩
      simp only [hIdx, hvf, he, Bool.false_eq_true, ite_false]
      ring
  have hTile : d.h.tileAt e d.V 1 (fun a : Fin 64 => base0 + a) (fun jj : Fin BV => BV * jv + jj)
      d.K d.V = d.h.elems (hBlk d n hh jv t base0) := by
    unfold Buf.tileAt; rw [hset]
  refine safe_store_tileAt d.h (e := e) (s0 := d.V) (s1 := 1) (ρ := fun a : Fin 64 => base0 + a)
    (κ := fun jj : Fin BV => BV * jv + jj) (M := d.K) (N := d.V) hptr hmask ?_
    (inj_rowMajor1 le_rfl) ?_ ?_ hvalid (by rw [hTile]; exact hW) ?_
  · right; right
    have h1 : (cbase d n + t + 1) * d.HV * d.K * d.V = ((cbase d n + t) * d.HV + d.HV) * (d.K * d.V) := by ring
    have h2 : e = ((cbase d n + t) * d.HV + hh) * (d.K * d.V) := by simp only [he]; ring
    have h3 : ((cbase d n + t) * d.HV + hh + 1) * (d.K * d.V) ≤ ((cbase d n + t) * d.HV + d.HV) * (d.K * d.V) :=
      Nat.mul_le_mul_right _ (by omega)
    have hKp := hd.K_pos
    have hVp := hd.V_pos
    have h4 : (d.K - 1) * d.V + (d.V - 1) * 1 < d.K * d.V := by
      have : (d.K - 1) * d.V + d.V = d.K * d.V := by
        obtain ⟨k', hk'⟩ : ∃ k', d.K = k' + 1 := ⟨d.K - 1, by omega⟩
        rw [hk']; simp [Nat.succ_mul]
      omega
    nlinarith
  · intro i i' _ _ h; exact Fin.ext (by simpa using h)
  · intro j j' _ _ h; exact Fin.ext (by simpa using h)
  · intro m' hr hf hp'
    rw [hTile] at hf ⊢
    refine hk x (fun a jj => hcv (a, jj, ())) m' ?_ hf hp'
    intro a jj ha hjj
    have := hr a jj ha hjj
    convert this using 4
    simp only [hIdx, hvf, he, Bool.false_eq_true, ite_false]
    ring

end Claims.ChunkDeltaH
