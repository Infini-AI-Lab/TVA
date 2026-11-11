import Proof.Seg
import Proof.Foot

/-!
# `chunk_delta_h_bf16`: the initial and final state (segments of the prologue and epilogue)

The loads of `h0` (added to the zero blocks) and the stores of the final state to `ht`, in both
layouts, with the optional blocks.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA TVA.Tile2

/-- The elements of `ht` that block `base0` stores, in the columns of block `jv` (`[K, V]` or
`[V, K]` layout alike). -/
def tBlk (d : Inst) (n hv jv base0 : Nat) : Set Nat :=
  {x | ∃ a jj, a < 64 ∧ base0 + a < d.K ∧ jj < BV ∧ inBlk d jv (BV * jv + jj) ∧
    x = sIdx d n hv (base0 + a) (BV * jv + jj)}

/-- The elements of `ht` stored by a block are among the cells the program writes. -/
theorem tBlk_sub_tCells {d : Inst} {n hv jv base0 : Nat} : tBlk d n hv jv base0 ⊆ tCells d n hv jv := by
  rintro x ⟨a, jj, -, ha, -, hj, rfl⟩
  exact ⟨base0 + a, BV * jv + jj, ha, hj, rfl⟩

/-- **The store of a state block to `ht`** (`[K, V]` layout, source lines 320–349): lane `(a, jj)` is
stored to `ht[n, hv, base0 + a, 32·jv + jj]`. -/
theorem safe_tstore {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {n hv jv base0 : Nat} (hvf : d.vf = false) (_hn : n < d.N)
    (_hhv : hv < d.HV) (_hb0 : base0 < d.K) (hbase : base0 ≤ 192) {hp : Ptr .f32}
    (hhp : hp = d.ht.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int)) {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hvalid : d.ht.Valid m) (hW : d.ht.elems (tBlk d n hv jv base0) ⊆ W)
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ d.ht.len) {b : Tensor [64, BV] F32}
    {k : Unit → Prog α}
    (hk : ∀ m' : Mem,
      (∀ (a : Fin 64) (jj : Fin BV), base0 + a < d.K → BV * jv + jj < d.V →
        m'.read .f32 (d.ht.ptr ((sIdx d n hv (base0 + a) (BV * jv + jj) : Nat) : Int)).addr
          = some (b.get2 a jj)) →
      (∀ y, y ∉ d.ht.elems (tBlk d n hv jv base0) → m' y = m y) →
      (∀ y, m y ≠ none → m' y ≠ none) →
      Safe E R (W \ d.ht.elems (tBlk d n hv jv base0)) Q m' (k ())) :
    Safe E R W Q m ((tl.store (hp + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64) b
      (some ((m_k.col BV).and (m_v.row 64)))).bind k) := by
  have hV31 : d.V < 2 ^ 31 := lt_of_lt_of_le hd.V_lt (by norm_num)
  have hK := hd.K_le
  have hVi : (IntV.ofNat d.V : I32).toInt = (d.V : Int) := IntV.I32.toInt_ofNat hV31
  set e : Nat := (n * d.HV + hv) * d.K * d.V with he
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
      (hp + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64).get2 i j =
        d.ht.ptr ((e + (base0 + i) * d.V + (BV * jv + j) * 1 : Nat) : Int) := fun i j _ _ =>
    ptr_lane_seq_of d.ht hhp hx hy i j
  have hmask : MaskAt (some ((m_k.col BV).and (m_v.row 64))) (fun a : Fin 64 => base0 + a)
      (fun jj : Fin BV => BV * jv + jj) d.K d.V := by
    refine maskAt_of_get fun i j => ?_
    show (m_k.get1 i && m_v.get1 j) = _
    rw [hmk, hmv]
  have hidx : ∀ (c j : Nat), sIdx d n hv c j = e + c * d.V + j * 1 := by
    intro c j; simp only [sIdx, hvf, he, Bool.false_eq_true, ite_false]; ring
  have hset : {x | ElemAt e d.V 1 (fun a : Fin 64 => base0 + a) (fun jj : Fin BV => BV * jv + jj)
      d.K d.V x} = tBlk d n hv jv base0 := by
    ext x
    constructor
    · rintro ⟨i, j, hi, hj, rfl⟩
      exact ⟨i, j, i.isLt, hi, j.isLt, ⟨by omega, by omega, hj⟩, (hidx _ _).symm⟩
    · rintro ⟨i, j, hi, ha, hj, ⟨-, -, hjv⟩, rfl⟩
      exact ⟨⟨i, hi⟩, ⟨j, hj⟩, ha, hjv, hidx _ _⟩
  have hTile : d.ht.tileAt e d.V 1 (fun a : Fin 64 => base0 + a) (fun jj : Fin BV => BV * jv + jj)
      d.K d.V = d.ht.elems (tBlk d n hv jv base0) := by
    unfold Buf.tileAt; rw [hset]
  refine safe_store_tileAt d.ht (e := e) (s0 := d.V) (s1 := 1) (ρ := fun a : Fin 64 => base0 + a)
    (κ := fun jj : Fin BV => BV * jv + jj) (M := d.K) (N := d.V) hptr hmask ?_
    (inj_rowMajor1 le_rfl) ?_ ?_ hvalid (by rw [hTile]; exact hW) ?_
  · right; right
    have h1 : (n * d.HV + hv + 1) * d.K * d.V = (n * d.HV + hv) * (d.K * d.V) + d.K * d.V := by ring
    have h2 : e = (n * d.HV + hv) * (d.K * d.V) := by simp only [he]; ring
    have hKp := hd.K_pos
    have hVp := hd.V_pos
    have h4 : (d.K - 1) * d.V + (d.V - 1) * 1 < d.K * d.V := by
      have : (d.K - 1) * d.V + d.V = d.K * d.V := by
        obtain ⟨k', hk'⟩ : ∃ k', d.K = k' + 1 := ⟨d.K - 1, by omega⟩
        rw [hk']; simp [Nat.succ_mul]
      omega
    omega
  · intro i i' _ _ h; exact Fin.ext (by simpa using h)
  · intro j j' _ _ h; exact Fin.ext (by simpa using h)
  · intro m' hr hf hp'
    rw [hTile] at hf ⊢
    refine hk m' ?_ hf hp'
    intro a jj ha hjj
    have := hr a jj ha hjj
    rw [hidx]
    exact this

/-- No element is stored by a block that does not exist. -/
theorem tBlk_empty {d : Inst} {n hv jv base0 : Nat} (h : ¬ base0 < d.K) :
    d.ht.elems (tBlk d n hv jv base0) = ∅ := by
  have : tBlk d n hv jv base0 = ∅ := by
    ext x
    simp only [tBlk, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨a, jj, -, ha, -⟩
    omega
  rw [this]
  ext x
  simp [Buf.elems]

/-- **The store of an optional state block to `ht`** (`[K, V]` layout). -/
theorem safe_tstore_opt {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {n hv jv base0 : Nat} (hvf : d.vf = false) (hn : n < d.N)
    (hhv : hv < d.HV) (hbase : base0 ≤ 192) {hp : Ptr .f32}
    (hhp : hp = d.ht.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int)) {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hvalid : d.ht.Valid m) (hW : d.ht.elems (tBlk d n hv jv base0) ⊆ W)
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ d.ht.len) {b : Tensor [64, BV] F32}
    {k : Unit → Prog α}
    (hk : ∀ m' : Mem,
      (∀ (a : Fin 64) (jj : Fin BV), base0 + a < d.K → BV * jv + jj < d.V →
        m'.read .f32 (d.ht.ptr ((sIdx d n hv (base0 + a) (BV * jv + jj) : Nat) : Int)).addr
          = some (b.get2 a jj)) →
      (∀ y, y ∉ d.ht.elems (tBlk d n hv jv base0) → m' y = m y) →
      (∀ y, m y ≠ none → m' y ≠ none) →
      Safe E R (W \ d.ht.elems (tBlk d n hv jv base0)) Q m' (k ())) :
    Safe E R W Q m (if base0 < d.K then (tl.store (hp + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64) b
      (some ((m_k.col BV).and (m_v.row 64)))).bind k else k ()) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    exact safe_tstore hd hvf hn hhv hc hbase hhp hok hmk hov hmv hvalid hW hlen hk
  · simp only [hc, ite_false]
    have hE := tBlk_empty (d := d) (n := n) (hv := hv) (jv := jv) hc
    have := hk m (fun a jj ha => absurd (by omega) hc) (fun y _ => rfl) (fun y hy => hy)
    rw [hE, Set.sdiff_empty] at this
    exact this

/-- **The store of a state block to `ht`** (`[V, K]` layout): lane `(jj, a)` is stored to
`ht[n, hv, 32·jv + jj, base0 + a]`. -/
theorem safe_tstore_vf {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {n hv jv base0 : Nat} (hvf : d.vf = true) (_hn : n < d.N)
    (_hhv : hv < d.HV) (hjv : jv < NV d) {hp : Ptr .f32}
    (hhp : hp = d.ht.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int)) {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hvalid : d.ht.Valid m) (hW : d.ht.elems (tBlk d n hv jv base0) ⊆ W)
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ d.ht.len) {b : Tensor [BV, 64] F32}
    {k : Unit → Prog α}
    (hk : ∀ m' : Mem,
      (∀ (jj : Fin BV) (a : Fin 64), base0 + a < d.K → BV * jv + jj < d.V →
        m'.read .f32 (d.ht.ptr ((sIdx d n hv (base0 + a) (BV * jv + jj) : Nat) : Int)).addr
          = some (b.get2 jj a)) →
      (∀ y, y ∉ d.ht.elems (tBlk d n hv jv base0) → m' y = m y) →
      (∀ y, m y ≠ none → m' y ≠ none) →
      Safe E R (W \ d.ht.elems (tBlk d n hv jv base0)) Q m' (k ())) :
    Safe E R W Q m ((tl.store (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV) b
      (some ((m_v.col 64).and (m_k.row BV)))).bind k) := by
  have hK := hd.K_le; have hKp := hd.K_pos; have hVp := hd.V_pos
  have hKi : (IntV.ofNat d.K : I64).toInt = (d.K : Int) := IntV.I64.toInt_ofNat (by omega)
  set e : Nat := (n * d.HV + hv) * d.K * d.V with he
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
      (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV).get2 j a =
        d.ht.ptr ((e + (BV * jv + j) * d.K + (base0 + a) * 1 : Nat) : Int) := fun j a _ _ =>
    ptr_lane_seq_of d.ht hhp hx hy j a
  have hmask : MaskAt (some ((m_v.col 64).and (m_k.row BV))) (fun j : Fin BV => BV * jv + j)
      (fun a : Fin 64 => base0 + a) d.V d.K := by
    refine maskAt_of_get fun j a => ?_
    show (m_v.get1 j && m_k.get1 a) = _
    rw [hmv, hmk]
  have hidx : ∀ (c j : Nat), sIdx d n hv c j = e + j * d.K + c * 1 := by
    intro c j; simp only [sIdx, hvf, he, ite_true]; ring
  have hset : {x | ElemAt e d.K 1 (fun j : Fin BV => BV * jv + j) (fun a : Fin 64 => base0 + a)
      d.V d.K x} = tBlk d n hv jv base0 := by
    ext x
    constructor
    · rintro ⟨j, a, hj, ha, rfl⟩
      exact ⟨a, j, a.isLt, ha, j.isLt, ⟨by omega, by omega, hj⟩, (hidx _ _).symm⟩
    · rintro ⟨a, j, ha64, ha, hj, ⟨-, -, hjv'⟩, rfl⟩
      exact ⟨⟨j, hj⟩, ⟨a, ha64⟩, hjv', ha, hidx _ _⟩
  have hTile : d.ht.tileAt e d.K 1 (fun j : Fin BV => BV * jv + j) (fun a : Fin 64 => base0 + a)
      d.V d.K = d.ht.elems (tBlk d n hv jv base0) := by
    unfold Buf.tileAt; rw [hset]
  refine safe_store_tileAt d.ht (e := e) (s0 := d.K) (s1 := 1) (ρ := fun j : Fin BV => BV * jv + j)
    (κ := fun a : Fin 64 => base0 + a) (M := d.V) (N := d.K) hptr hmask ?_
    (inj_rowMajor1 le_rfl) ?_ ?_ hvalid (by rw [hTile]; exact hW) ?_
  · right; right
    have h1 : (n * d.HV + hv + 1) * d.K * d.V = (n * d.HV + hv) * (d.V * d.K) + d.V * d.K := by ring
    have h2 : e = (n * d.HV + hv) * (d.V * d.K) := by simp only [he]; ring
    have h4 : (d.V - 1) * d.K + (d.K - 1) * 1 < d.V * d.K := by
      have : (d.V - 1) * d.K + d.K = d.V * d.K := by
        obtain ⟨k', hk'⟩ : ∃ k', d.V = k' + 1 := ⟨d.V - 1, by omega⟩
        rw [hk']; simp [Nat.succ_mul]
      omega
    omega
  · intro i i' _ _ h; exact Fin.ext (by simpa using h)
  · intro j j' _ _ h; exact Fin.ext (by simpa using h)
  · intro m' hr hf hp'
    rw [hTile] at hf ⊢
    refine hk m' ?_ hf hp'
    intro jj a hjj ha
    have := hr jj a ha hjj
    rw [hidx]
    exact this

/-- **The store of an optional state block to `ht`** (`[V, K]` layout). -/
theorem safe_tstore_vf_opt {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {n hv jv base0 : Nat} (hvf : d.vf = true) (hn : n < d.N)
    (hhv : hv < d.HV) (hjv : jv < NV d) {hp : Ptr .f32}
    (hhp : hp = d.ht.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int)) {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hvalid : d.ht.Valid m) (hW : d.ht.elems (tBlk d n hv jv base0) ⊆ W)
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ d.ht.len) {b : Tensor [BV, 64] F32}
    {k : Unit → Prog α}
    (hk : ∀ m' : Mem,
      (∀ (jj : Fin BV) (a : Fin 64), base0 + a < d.K → BV * jv + jj < d.V →
        m'.read .f32 (d.ht.ptr ((sIdx d n hv (base0 + a) (BV * jv + jj) : Nat) : Int)).addr
          = some (b.get2 jj a)) →
      (∀ y, y ∉ d.ht.elems (tBlk d n hv jv base0) → m' y = m y) →
      (∀ y, m y ≠ none → m' y ≠ none) →
      Safe E R (W \ d.ht.elems (tBlk d n hv jv base0)) Q m' (k ())) :
    Safe E R W Q m (if base0 < d.K then (tl.store (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV) b
      (some ((m_v.col 64).and (m_k.row BV)))).bind k else k ()) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    exact safe_tstore_vf hd hvf hn hhv hjv hhp hok hmk hov hmv hvalid hW hlen hk
  · simp only [hc, ite_false]
    have hE := tBlk_empty (d := d) (n := n) (hv := hv) (jv := jv) hc
    have := hk m (fun jj a ha => absurd (by omega) hc) (fun y _ => rfl) (fun y hy => hy)
    rw [hE, Set.sdiff_empty] at this
    exact this

/-- What the load of an `h0` tile and its sum with the block `z` establish: the tile `v` (lane `(a, jj)`
is `h0[n, hv, base0 + a, 32·jv + jj]`, read from `init`, or `+0` outside) and the IEEE sum `r`. -/
def H0Facts (d : Inst) (init : Mem) (n hv jv base0 : Nat) (z r : Tensor [64, BV] F32) : Prop :=
  ∃ v : Tensor [64, BV] F32,
    (∀ (a : Fin 64) (jj : Fin BV), base0 + a < d.K → BV * jv + jj < d.V →
      rdF init d.h0 (sIdx d n hv (base0 + a) (BV * jv + jj)) = some (v.get2 a jj)) ∧
    (∀ (a : Fin 64) (jj : Fin BV), ¬ (base0 + a < d.K ∧ BV * jv + jj < d.V) → v.get2 a jj = zf32) ∧
    ∀ a jj, FloatV.NaNEq (FloatV.add (z.get2 a jj) (v.get2 a jj)) (r.get2 a jj)

/-- As `H0Facts`, for the `[V, K]` layout: the tile and the sum have lanes `(jj, a)`. -/
def H0FactsV (d : Inst) (init : Mem) (n hv jv base0 : Nat) (z r : Tensor [BV, 64] F32) : Prop :=
  ∃ v : Tensor [BV, 64] F32,
    (∀ (jj : Fin BV) (a : Fin 64), base0 + a < d.K → BV * jv + jj < d.V →
      rdF init d.h0 (sIdx d n hv (base0 + a) (BV * jv + jj)) = some (v.get2 jj a)) ∧
    (∀ (jj : Fin BV) (a : Fin 64), ¬ (base0 + a < d.K ∧ BV * jv + jj < d.V) → v.get2 jj a = zf32) ∧
    ∀ jj a, FloatV.NaNEq (FloatV.add (z.get2 jj a) (v.get2 jj a)) (r.get2 jj a)

/-- **The load of an `h0` tile added to a block** (`[K, V]` layout, source lines 141–172). -/
theorem safe_h0blk {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRh0 : d.h0.bytes ⊆ R)
    (hvh0 : d.h0.Valid m) {n hv jv base0 : Nat} (hvf : d.vf = false) (_hn : n < d.N)
    (_hhv : hv < d.HV) (hbase : base0 ≤ 192) {hp : Ptr .f32}
    (hhp : hp = d.h0.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int)) {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ d.h0.len) (z : Tensor [64, BV] F32)
    {k : Tensor [64, BV] F32 → Prog α}
    (hk : ∀ r, H0Facts d init n hv jv base0 z r → Safe E R W Q m (k r)) :
    Safe E R W Q m (((tl.load (d := .f32) (hp + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64)
      (some ((m_k.col BV).and (m_v.row 64))) (some (bc zf32))) : Prog (Tensor [64, BV] F32)).bind
        fun v => (tl.add z v).bind k) := by
  have hV31 : d.V < 2 ^ 31 := lt_of_lt_of_le hd.V_lt (by norm_num)
  have hKp := hd.K_pos; have hVp := hd.V_pos
  have hVi : (IntV.ofNat d.V : I32).toInt = (d.V : Int) := IntV.I32.toInt_ofNat hV31
  set e : Nat := (n * d.HV + hv) * d.K * d.V with he
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
      (hp + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64).get2 i j =
        d.h0.ptr ((e + (base0 + i) * d.V + (BV * jv + j) * 1 : Nat) : Int) := fun i j _ _ =>
    ptr_lane_seq_of d.h0 hhp hx hy i j
  have hmask : MaskAt (some ((m_k.col BV).and (m_v.row 64))) (fun a : Fin 64 => base0 + a)
      (fun jj : Fin BV => BV * jv + jj) d.K d.V := by
    refine maskAt_of_get fun i j => ?_
    show (m_k.get1 i && m_v.get1 j) = _
    rw [hmk, hmv]
  have hidx : ∀ (c j : Nat), sIdx d n hv c j = e + c * d.V + j * 1 := by
    intro c j; simp only [sIdx, hvf, he, Bool.false_eq_true, ite_false]; ring
  have hlen' : d.K = 0 ∨ d.V = 0 ∨ e + (d.K - 1) * d.V + (d.V - 1) * 1 < d.h0.len := by
    right; right
    have h1 : (n * d.HV + hv + 1) * d.K * d.V = (n * d.HV + hv) * (d.K * d.V) + d.K * d.V := by ring
    have h2 : e = (n * d.HV + hv) * (d.K * d.V) := by simp only [he]; ring
    have h4 : (d.K - 1) * d.V + (d.V - 1) * 1 < d.K * d.V := by
      have : (d.K - 1) * d.V + d.V = d.K * d.V := by
        obtain ⟨k', hk'⟩ : ∃ k', d.K = k' + 1 := ⟨d.K - 1, by omega⟩
        rw [hk']; simp [Nat.succ_mul]
      omega
    omega
  have hR : d.h0.tileAt e d.V 1 (fun a : Fin 64 => base0 + a) (fun jj : Fin BV => BV * jv + jj)
      d.K d.V ⊆ R := (d.h0.tileAt_sub_bytes hlen').trans hRh0
  refine safe_load_tileAt d.h0 (ρ := fun a : Fin 64 => base0 + a) (κ := fun jj : Fin BV => BV * jv + jj)
    hptr hmask hlen' hvh0 hR fun v hr ho => ?_
  refine safe_add fun r hr' => hk r ⟨v, fun a jj ha hjj => ?_, fun a jj hno => ?_, fun a jj => hr' (a, jj, ())⟩
  · have hrd := hr a jj ha hjj
    unfold rdF
    rw [hidx, ← read_init hmi (b := d.h0) (S := {x | ElemAt e d.V 1 (fun a : Fin 64 => base0 + a)
      (fun jj : Fin BV => BV * jv + jj) d.K d.V x}) hR (j := e + (base0 + a) * d.V + (BV * jv + jj) * 1)
      ⟨a, jj, ha, hjj, rfl⟩]
    exact hrd
  · have := ho a jj (by simpa using hno) _ rfl
    rw [this]; rfl

/-- **The load of an `h0` tile added to a block** (`[V, K]` layout). -/
theorem safe_h0blk_vf {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRh0 : d.h0.bytes ⊆ R)
    (hvh0 : d.h0.Valid m) {n hv jv base0 : Nat} (hvf : d.vf = true) (_hn : n < d.N)
    (_hhv : hv < d.HV) (hjv : jv < NV d) {hp : Ptr .f32}
    (hhp : hp = d.h0.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int)) {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ d.h0.len) (z : Tensor [BV, 64] F32)
    {k : Tensor [BV, 64] F32 → Prog α}
    (hk : ∀ r, H0FactsV d init n hv jv base0 z r → Safe E R W Q m (k r)) :
    Safe E R W Q m (((tl.load (d := .f32) (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV)
      (some ((m_v.col 64).and (m_k.row BV))) (some (bc zf32))) : Prog (Tensor [BV, 64] F32)).bind
        fun v => (tl.add z v).bind k) := by
  have hK := hd.K_le; have hKp := hd.K_pos; have hVp := hd.V_pos
  have hKi : (IntV.ofNat d.K : I64).toInt = (d.K : Int) := IntV.I64.toInt_ofNat (by omega)
  set e : Nat := (n * d.HV + hv) * d.K * d.V with he
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
      (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV).get2 j a =
        d.h0.ptr ((e + (BV * jv + j) * d.K + (base0 + a) * 1 : Nat) : Int) := fun j a _ _ =>
    ptr_lane_seq_of d.h0 hhp hx hy j a
  have hmask : MaskAt (some ((m_v.col 64).and (m_k.row BV))) (fun j : Fin BV => BV * jv + j)
      (fun a : Fin 64 => base0 + a) d.V d.K := by
    refine maskAt_of_get fun j a => ?_
    show (m_v.get1 j && m_k.get1 a) = _
    rw [hmv, hmk]
  have hidx : ∀ (c j : Nat), sIdx d n hv c j = e + j * d.K + c * 1 := by
    intro c j; simp only [sIdx, hvf, he, ite_true]; ring
  have hlen' : d.V = 0 ∨ d.K = 0 ∨ e + (d.V - 1) * d.K + (d.K - 1) * 1 < d.h0.len := by
    right; right
    have h1 : (n * d.HV + hv + 1) * d.K * d.V = (n * d.HV + hv) * (d.V * d.K) + d.V * d.K := by ring
    have h2 : e = (n * d.HV + hv) * (d.V * d.K) := by simp only [he]; ring
    have h4 : (d.V - 1) * d.K + (d.K - 1) * 1 < d.V * d.K := by
      have : (d.V - 1) * d.K + d.K = d.V * d.K := by
        obtain ⟨k', hk'⟩ : ∃ k', d.V = k' + 1 := ⟨d.V - 1, by omega⟩
        rw [hk']; simp [Nat.succ_mul]
      omega
    omega
  have hR : d.h0.tileAt e d.K 1 (fun j : Fin BV => BV * jv + j) (fun a : Fin 64 => base0 + a)
      d.V d.K ⊆ R := (d.h0.tileAt_sub_bytes hlen').trans hRh0
  refine safe_load_tileAt d.h0 (ρ := fun j : Fin BV => BV * jv + j) (κ := fun a : Fin 64 => base0 + a)
    hptr hmask hlen' hvh0 hR fun v hr ho => ?_
  refine safe_add fun r hr' => hk r ⟨v, fun jj a ha hjj => ?_, fun jj a hno => ?_, fun jj a => hr' (jj, a, ())⟩
  · have hrd := hr jj a hjj ha
    unfold rdF
    rw [hidx, ← read_init hmi (b := d.h0) (S := {x | ElemAt e d.K 1 (fun j : Fin BV => BV * jv + j)
      (fun a : Fin 64 => base0 + a) d.V d.K x}) hR (j := e + (BV * jv + jj) * d.K + (base0 + a) * 1)
      ⟨jj, a, hjj, ha, rfl⟩]
    exact hrd
  · have := ho jj a (by simpa [and_comm] using hno) _ rfl
    rw [this]; rfl

/-- **An optional `h0` block** (`if base0 < K`): the load added to the block, or the block unchanged. -/
theorem safe_h0blk_opt {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRh0 : d.h0.bytes ⊆ R)
    (hvh0 : d.h0.Valid m) {n hv jv base0 : Nat} (hvf : d.vf = false) (hn : n < d.N)
    (hhv : hv < d.HV) (hbase : base0 ≤ 192) {hp : Ptr .f32}
    (hhp : hp = d.h0.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int)) {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ d.h0.len) (z : Tensor [64, BV] F32)
    {k : Tensor [64, BV] F32 → Prog α}
    (hk : ∀ r, (base0 < d.K ∧ H0Facts d init n hv jv base0 z r) ∨ (¬ base0 < d.K ∧ r = z) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (if base0 < d.K then ((tl.load (d := .f32) (hp + o_k.col BV * bc (IntV.ofNat d.V : I32) + o_v.row 64)
      (some ((m_k.col BV).and (m_v.row 64))) (some (bc zf32))) : Prog (Tensor [64, BV] F32)).bind
        (fun v => (tl.add z v).bind k) else k z) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    exact safe_h0blk hd hmi hRh0 hvh0 hvf hn hhv hbase hhp hok hmk hov hmv hlen z fun r hr => hk r (Or.inl ⟨hc, hr⟩)
  · simp only [hc, ite_false]
    exact hk z (Or.inr ⟨hc, rfl⟩)

/-- **An optional `h0` block**, `[V, K]` layout. -/
theorem safe_h0blk_vf_opt {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m init : Mem} (hmi : ∀ x ∈ R, m x = init x) (hRh0 : d.h0.bytes ⊆ R)
    (hvh0 : d.h0.Valid m) {n hv jv base0 : Nat} (hvf : d.vf = true) (hn : n < d.N)
    (hhv : hv < d.HV) (hjv : jv < NV d) {hp : Ptr .f32}
    (hhp : hp = d.h0.ptr (((n * d.HV + hv) * d.K * d.V : Nat) : Int)) {o_k : Tensor [64] I32}
    {m_k : Tensor [64] Bool} {o_v : Tensor [BV] I64} {m_v : Tensor [BV] Bool}
    (hok : ∀ a : Fin 64, (o_k.get1 a).toInt = ((base0 + a : Nat) : Int))
    (hmk : ∀ a : Fin 64, m_k.get1 a = decide (base0 + a < d.K))
    (hov : ∀ jj : Fin BV, (o_v.get1 jj).toInt = ((BV * jv + jj : Nat) : Int))
    (hmv : ∀ jj : Fin BV, m_v.get1 jj = decide (BV * jv + jj < d.V))
    (hlen : (n * d.HV + hv + 1) * d.K * d.V ≤ d.h0.len) (z : Tensor [BV, 64] F32)
    {k : Tensor [BV, 64] F32 → Prog α}
    (hk : ∀ r, (base0 < d.K ∧ H0FactsV d init n hv jv base0 z r) ∨ (¬ base0 < d.K ∧ r = z) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (if base0 < d.K then ((tl.load (d := .f32) (hp + o_v.col 64 * bc (IntV.ofNat d.K : I64) + o_k.row BV)
      (some ((m_v.col 64).and (m_k.row BV))) (some (bc zf32))) : Prog (Tensor [BV, 64] F32)).bind
        (fun v => (tl.add z v).bind k) else k z) := by
  by_cases hc : base0 < d.K
  · simp only [hc, ite_true]
    exact safe_h0blk_vf hd hmi hRh0 hvh0 hvf hn hhv hjv hhp hok hmk hov hmv hlen z fun r hr => hk r (Or.inl ⟨hc, hr⟩)
  · simp only [hc, ite_false]
    exact hk z (Or.inr ⟨hc, rfl⟩)

end Claims.ChunkDeltaH
