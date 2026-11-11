import Claim
import TVA.Proof
import Proof.Load
import Proof.Loop
import Proof.Facts

/-!
# The loads of a trip

`TailLanes`: the pointers and index tensors the loop starts from (lines 123–134). The loads of
`b_q`, `b_k` and the state tile (lines 138–150) read the elements `qTile`, `kTile`, `hTile` name.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- What the scalar part of the kernel leaves: the five offset pointers (as elements of the
buffers), the token offsets `o_t`, the length `T` and the value-column offsets `o_v`. -/
structure TailLanes (d : Inst) (g : Geo) (q k v h o : Ptr .bf16) (o_t : Tensor [BT] I64)
    (T : I64) (o_v : Tensor [d.bk] I64) : Prop where
  hq : q = d.q.ptr (((g.bos * d.H + qh d g.ih) * d.K : Nat) : Int)
  hk : k = d.k.ptr (((g.bos * d.H + qh d g.ih) * d.K : Nat) : Int)
  hv : v = d.v.ptr (((g.bos * d.HV + g.ih) * d.V : Nat) : Int)
  ho : o = d.o.ptr (((g.bos * d.HV + g.ih) * d.V : Nat) : Int)
  hh : h = d.h.ptr (((g.itg * d.HV + g.ih) * d.K * d.V : Nat) : Int)
  ot : ∀ i : Fin BT, (o_t.get1 i).toInt = ((g.it * BT + i : Nat) : Int)
  hT : T.toInt = (g.Tn : Int)
  ov : ∀ c : Fin d.bk, (o_v.get1 c).toInt = ((g.iv * d.bk + c : Nat) : Int)

/-- The lane `i` of `o_t < T` is the token mask. -/
theorem mt_get {d : Inst} {g : Geo} {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64}
    {o_v : Tensor [d.bk] I64} (hl : TailLanes d g q k v h o o_t T o_v) (i : Fin BT) :
    (o_t.ltS T).get1 i = decide (g.it * BT + i < g.Tn) := by
  rw [Tensor.get1_ltS, IntV.lt_eq, hl.ot i, hl.hT]
  simp only [Nat.cast_lt]

/-- The mask `(a[:, None]) & (b[None, :])` of two lane masks is the tile mask of rows `ρ < M` and
columns `κ < N`. -/
theorem maskAt_colrow {BM BN M N : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}
    {a : Tensor [BM] Bool} {b : Tensor [BN] Bool}
    (ha : ∀ i, a.get1 i = decide (ρ i < M)) (hb : ∀ j, b.get1 j = decide (κ j < N)) :
    Tile2.MaskAt (some ((a.col BN).and (b.row BM))) ρ κ M N := by
  refine Tile2.maskAt_of_get fun i j => ?_
  show (a.get1 i && b.get1 j) = _
  rw [ha, hb]

/-- `x < n` for the lanes of `i_k * BK + tl.arange(0, BK)`: lane `j` is `t·BK + j`. -/
theorem ok_get {d : Inst} (hd : d.ok) {t : Nat} (ht : t < nK d) {i_k : I32}
    (hik : i_k.toInt = (t : Int)) (j : Fin d.bk) :
    ((i_k * (IntV.ofNat d.bk : I32) + tl.arange 0 d.bk).get1 j).toInt = ((t * d.bk + j : Nat) : Int) := by
  have hb := bk_le hd
  have hn := nK_le hd
  have hj := j.isLt
  have h1 : t * d.bk ≤ 256 * 128 := Nat.mul_le_mul (by omega) hb
  have hbk : (IntV.ofNat d.bk : I32).toInt = (d.bk : Int) :=
    IntV.I32.toInt_ofNat (by norm_num; omega)
  have hmul : (i_k * (IntV.ofNat d.bk : I32)).toInt = ((t * d.bk : Nat) : Int) :=
    IntV.I32.mul_nat hik hbk (by norm_num; omega)
  exact IntV.I32.toInt_sadd_arange (n := d.bk) hmul (by omega) j

/-! ## The last element of a token-major tensor -/

/-- The last element of the `Tn`-token slab starting at token `bos` of a `[tokens, H, K]` tensor
(head `qh`) is below the tensor's length. -/
theorem slab_last_lt {bos Tn B T H K qh len : Nat} (hTn : 1 ≤ Tn) (hqh : qh < H) (hK : 0 < K)
    (hend : bos + Tn ≤ B * T) (hlen : B * T * H * K ≤ len) :
    (bos * H + qh) * K + (Tn - 1) * (H * K) + (K - 1) < len := by
  obtain ⟨n, rfl⟩ : ∃ n, Tn = n + 1 := ⟨Tn - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  have h1 : (bos + n + 1) * (H * K) ≤ B * T * (H * K) := Nat.mul_le_mul_right _ (by omega)
  have h2 : (qh + 1) * K ≤ H * K := Nat.mul_le_mul_right _ hqh
  have e1 : (bos + n + 1) * (H * K) = bos * (H * K) + n * (H * K) + H * K := by ring
  have e2 : (bos * H + qh) * K = bos * (H * K) + qh * K := by ring
  have e3 : (qh + 1) * K = qh * K + K := by ring
  have e4 : B * T * H * K = B * T * (H * K) := by ring
  omega

/-! ## The loads -/

variable {d : Inst} {g : Geo} {r : Nat} {init : Mem}

/-- **The load of `b_q`** (line 147): `b_q[i, j] = q[token i, qh, t·BK + j]`, `+0` past the sequence
or past `K`. -/
theorem safe_load_q (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) (hm : m = init) {t : Nat} (_ht : t < nK d)
    {o_k : Tensor [d.bk] I32}
    (hok : ∀ j : Fin d.bk, (o_k.get1 j).toInt = ((t * d.bk + j : Nat) : Int))
    {k' : Tensor [BT, d.bk] BF16 → Prog α}
    (hk : ∀ vq : Tensor [BT, d.bk] BF16, (∀ i j : Fin _, vq.get2 i j = qTile d g t i j) →
      Safe E (Rs d) W Q m (k' vq)) :
    Safe E (Rs d) W Q m (tl.convert .f32 .bf16 (tl.full [BT, d.bk] zf32) >>= fun z =>
      tl.load (q + o_t.col d.bk * (IntV.ofNat (d.H * d.K) : I64) + o_k.row BT)
        (some (((o_t.ltS T).col d.bk).and ((o_k.ltS (IntV.ofNat d.K : I32)).row BT))) (some z)
        >>= k') := by
  subst hm
  have hb := bk_le hd
  have hK0 := hd.K_pos
  have hTn : 1 ≤ g.Tn := by have := hg.start_lt; omega
  have hqh := qh_lt hd hg.ih_lt
  have hHK : d.H * d.K ≤ 2 ^ 24 := by
    have := H_lt16 hd; have := hd.K_le
    calc d.H * d.K ≤ 2 ^ 16 * 256 := Nat.mul_le_mul (by omega) (by omega)
      _ = 2 ^ 24 := by norm_num
  have hTn35 : g.Tn ≤ 2 ^ 35 := le_trans (by have := hg.end_le; omega) (BT_le hd)
  have hlen : g.Tn = 0 ∨ d.K = 0 ∨ (g.bos * d.H + qh d g.ih) * d.K + (g.Tn - 1) * (d.H * d.K) +
      (d.K - 1) * 1 < d.q.len :=
    Or.inr (Or.inr (by
      rw [Nat.mul_one]
      exact slab_last_lt hTn hqh hK0 hg.end_le hd.q_len))
  have hHK63 : d.H * d.K < 2 ^ 63 := lt_of_le_of_lt hHK (by norm_num)
  have hfit : (g.Tn - 1) * (d.H * d.K) < 2 ^ (64 - 1) := by
    have h1 : (g.Tn - 1) * (d.H * d.K) ≤ 2 ^ 35 * 2 ^ 24 :=
      Nat.mul_le_mul (by omega : g.Tn - 1 ≤ 2 ^ 35) hHK
    exact lt_of_le_of_lt h1 (by norm_num)
  have hK31 : d.K < 2 ^ 31 := by have := hd.K_le; omega
  refine safe_load_tileV d.q d.qv hpre.q_holds rfl
    (ρ := fun i : Fin BT => g.it * BT + i) (κ := fun j : Fin d.bk => t * d.bk + j)
    (M := g.Tn) (N := d.K) (e := (g.bos * d.H + qh d g.ih) * d.K) (s0 := d.H * d.K) (s1 := 1)
    ?_ (maskAt_colrow (mt_get hl) ?_) hlen hpre.q_valid ?_ fun vq hvq => hk vq fun i j => ?_
  · refine Tile2.ptr_lane_seq_of_on d.q hl.hq
      (fun i j hi => Tile2.colMul_toInt_on hl.ot (IntV.toInt_ofNat (w := 64) hHK63) hfit i j hi)
      (fun i j _ => Tile2.row_toInt (κ := fun j : Fin d.bk => t * d.bk + j) hok i j)
  · intro j
    rw [Tensor.get1_ltS, IntV.lt_eq, hok j, IntV.toInt_ofNat (w := 32) hK31]
    simp only [Nat.cast_lt]
  · exact (Buf.tileAt_sub_bytes d.q hlen).trans (fun x hx => Or.inl hx)
  · have hidx : (g.bos * d.H + qh d g.ih) * d.K + (g.it * BT + i) * (d.H * d.K) +
        (t * d.bk + j) * 1 = ((g.lo + i) * d.H + qh d g.ih) * d.K + (t * d.bk + j) := by
      unfold Geo.lo; ring
    rw [hvq i j]
    unfold qTile Geo.rowOk
    rw [hidx]

/-- **The load of `b_k`** (line 148): `b_k[j, i'] = k[token i', qh, t·BK + j]`, `+0` past the
sequence or past `K`. -/
theorem safe_load_k (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) (hm : m = init) {t : Nat}
    {o_k : Tensor [d.bk] I32}
    (hok : ∀ j : Fin d.bk, (o_k.get1 j).toInt = ((t * d.bk + j : Nat) : Int))
    {k' : Tensor [d.bk, BT] BF16 → Prog α}
    (hk : ∀ vk : Tensor [d.bk, BT] BF16, (∀ (j : Fin d.bk) (i : Fin BT), vk.get2 j i = kTile d g t j i) →
      Safe E (Rs d) W Q m (k' vk)) :
    Safe E (Rs d) W Q m (tl.convert .f32 .bf16 (tl.full [d.bk, BT] zf32) >>= fun z =>
      tl.load (k + o_k.col BT + o_t.row d.bk * (IntV.ofNat (d.H * d.K) : I64))
        (some (((o_k.ltS (IntV.ofNat d.K : I32)).col BT).and ((o_t.ltS T).row d.bk))) (some z)
        >>= k') := by
  subst hm
  have hb := bk_le hd
  have hK0 := hd.K_pos
  have hTn : 1 ≤ g.Tn := by have := hg.start_lt; omega
  have hqh := qh_lt hd hg.ih_lt
  have hHK : d.H * d.K ≤ 2 ^ 24 := by
    have := H_lt16 hd; have := hd.K_le
    calc d.H * d.K ≤ 2 ^ 16 * 256 := Nat.mul_le_mul (by omega) (by omega)
      _ = 2 ^ 24 := by norm_num
  have hTn35 : g.Tn ≤ 2 ^ 35 := le_trans (by have := hg.end_le; omega) (BT_le hd)
  have hlen : d.K = 0 ∨ g.Tn = 0 ∨ (g.bos * d.H + qh d g.ih) * d.K + (d.K - 1) * 1 +
      (g.Tn - 1) * (d.H * d.K) < d.k.len :=
    Or.inr (Or.inr (by
      have := slab_last_lt hTn hqh hK0 hg.end_le hd.k_len
      rw [Nat.mul_one]; omega))
  have hHK63 : d.H * d.K < 2 ^ 63 := lt_of_le_of_lt hHK (by norm_num)
  have hfit : (g.Tn - 1) * (d.H * d.K) < 2 ^ (64 - 1) := by
    have h1 : (g.Tn - 1) * (d.H * d.K) ≤ 2 ^ 35 * 2 ^ 24 :=
      Nat.mul_le_mul (by omega : g.Tn - 1 ≤ 2 ^ 35) hHK
    exact lt_of_le_of_lt h1 (by norm_num)
  have hK31 : d.K < 2 ^ 31 := by have := hd.K_le; omega
  refine safe_load_tileV d.k d.kv hpre.k_holds rfl
    (ρ := fun j : Fin d.bk => t * d.bk + j) (κ := fun i : Fin BT => g.it * BT + i)
    (M := d.K) (N := g.Tn) (e := (g.bos * d.H + qh d g.ih) * d.K) (s0 := 1) (s1 := d.H * d.K)
    ?_ (maskAt_colrow ?_ (mt_get hl)) hlen hpre.k_valid ?_ fun vk hvk => hk vk fun j i => ?_
  · exact Tile2.ptr_lane_seq_of_on d.k hl.hk
      (fun i j _ => Tile2.col_toInt (ρ := fun j : Fin d.bk => t * d.bk + j) hok i j)
      (fun i j hj => Tile2.rowMul_toInt_on hl.ot (IntV.toInt_ofNat (w := 64) hHK63) hfit i j hj)
  · intro j
    rw [Tensor.get1_ltS, IntV.lt_eq, hok j, IntV.toInt_ofNat (w := 32) hK31]
    simp only [Nat.cast_lt]
  · exact (Buf.tileAt_sub_bytes d.k hlen).trans (fun x hx => Or.inr (Or.inl hx))
  · have hidx : (g.bos * d.H + qh d g.ih) * d.K + (t * d.bk + j) * 1 +
        (g.it * BT + i) * (d.H * d.K) = ((g.lo + i) * d.H + qh d g.ih) * d.K + (t * d.bk + j) := by
      unfold Geo.lo; ring
    rw [hvk j i]
    unfold kTile Geo.rowOk
    rw [hidx]
    by_cases h1 : g.it * BT + i < g.Tn <;> by_cases h2 : t * d.bk + j < d.K <;> simp [h1, h2]

/-! ## The state tile -/

/-- A `[V, K]`-tile of the state `(a, e', x)` ends below `len`: `a·K·V + e'·K + x`. -/
theorem state_last_lt_svf {a nR HV K V len : Nat} (hK : 0 < K) (hV : 0 < V) (ha : a + 1 ≤ nR * HV)
    (hlen : nR * HV * K * V ≤ len) : a * K * V + (V - 1) * K + (K - 1) * 1 < len := by
  obtain ⟨v, rfl⟩ : ∃ v, V = v + 1 := ⟨V - 1, by omega⟩
  obtain ⟨k, rfl⟩ : ∃ k, K = k + 1 := ⟨K - 1, by omega⟩
  simp only [Nat.add_sub_cancel, Nat.mul_one]
  have h1 : (a + 1) * (k + 1) * (v + 1) ≤ nR * HV * (k + 1) * (v + 1) :=
    Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ ha)
  have e1 : (a + 1) * (k + 1) * (v + 1) = a * (k + 1) * (v + 1) + v * (k + 1) + k + 1 := by ring
  omega

/-- The same for the `[K, V]` orientation: `a·K·V + x·V + e'`. -/
theorem state_last_lt {a nR HV K V len : Nat} (hK : 0 < K) (hV : 0 < V) (ha : a + 1 ≤ nR * HV)
    (hlen : nR * HV * K * V ≤ len) : a * K * V + (K - 1) * V + (V - 1) * 1 < len := by
  obtain ⟨v, rfl⟩ : ∃ v, V = v + 1 := ⟨V - 1, by omega⟩
  obtain ⟨k, rfl⟩ : ∃ k, K = k + 1 := ⟨K - 1, by omega⟩
  simp only [Nat.add_sub_cancel, Nat.mul_one]
  have h1 : (a + 1) * (k + 1) * (v + 1) ≤ nR * HV * (k + 1) * (v + 1) :=
    Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ ha)
  have e1 : (a + 1) * (k + 1) * (v + 1) = a * (k + 1) * (v + 1) + k * (v + 1) + v + 1 := by ring
  omega

/-- The state row `g.itg * HV + g.ih + 1` is at most `nRows d * HV`. -/
theorem state_row_lt {d : Inst} (hg : Geo.Ok d g r) : g.itg * d.HV + g.ih + 1 ≤ nRows d * d.HV := by
  have h1 := hg.r_lt
  have h2 := hg.ih_lt
  have h3 : (g.itg + 1) * d.HV ≤ nRows d * d.HV := by
    apply Nat.mul_le_mul_right; rw [hg.itg_eq]; omega
  have : (g.itg + 1) * d.HV = g.itg * d.HV + d.HV := by ring
  omega

/-- **The load of the state, `STATE_V_FIRST`** (lines 141–150): the tile is `[V, K]`; lane `(c, k)`
is the state element `(x = t·BK + k, e = iv·BV + c)`. -/
theorem safe_load_h_svf (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) (hm : m = init) (hs : d.svf = true) {t : Nat}
    {o_k : Tensor [d.bk] I32}
    (hok : ∀ j : Fin d.bk, (o_k.get1 j).toInt = ((t * d.bk + j : Nat) : Int))
    {k' : Tensor [d.bk, d.bk] BF16 → Prog α}
    (hk : ∀ vh : Tensor [d.bk, d.bk] BF16,
      (∀ (k c : Fin d.bk), vh.trans.get2 k c = hTile d g r t k c) → Safe E (Rs d) W Q m (k' vh)) :
    Safe E (Rs d) W Q m (tl.convert .f32 .bf16 (tl.full [d.bk, d.bk] zf32) >>= fun z =>
      tl.load (h + o_v.col d.bk * (IntV.ofNat d.K : I64) + o_k.row d.bk)
        (some (((o_v.col d.bk).ltS (IntV.ofNat d.V : I64)).and
          ((o_k.ltS (IntV.ofNat d.K : I32)).row d.bk))) (some z) >>= k') := by
  subst hm
  have hb := bk_le hd
  have hK0 := hd.K_pos
  have hV0 := hd.V_pos
  have hKV : d.K * d.V ≤ 2 ^ 17 := by
    have := hd.K_le; have := hd.V_le
    calc d.K * d.V ≤ 256 * 512 := Nat.mul_le_mul (by omega) (by omega)
      _ = 2 ^ 17 := by norm_num
  have hlen : d.V = 0 ∨ d.K = 0 ∨ (g.itg * d.HV + g.ih) * d.K * d.V + (d.V - 1) * d.K +
      (d.K - 1) * 1 < d.h.len :=
    Or.inr (Or.inr (state_last_lt_svf hK0 hV0 (state_row_lt hg) hd.h_len))
  have hfit : (d.V - 1) * d.K < 2 ^ (64 - 1) := by
    have : (d.V - 1) * d.K ≤ d.K * d.V := by
      rw [Nat.mul_comm d.K]; exact Nat.mul_le_mul_right _ (Nat.sub_le _ _)
    exact lt_of_le_of_lt (this.trans hKV) (by norm_num)
  have hK31 : d.K < 2 ^ 31 := by have := hd.K_le; omega
  have hV63 : d.V < 2 ^ 63 := by have := hd.V_le; omega
  have hK63 : d.K < 2 ^ 63 := by have := hd.K_le; omega
  refine safe_load_tileV d.h d.hv hpre.h_holds rfl
    (ρ := fun c : Fin d.bk => g.iv * d.bk + c) (κ := fun j : Fin d.bk => t * d.bk + j)
    (M := d.V) (N := d.K) (e := (g.itg * d.HV + g.ih) * d.K * d.V) (s0 := d.K) (s1 := 1)
    ?_ (maskAt_colrow (a := o_v.ltS (IntV.ofNat d.V : I64)) ?_ ?_) hlen hpre.h_valid ?_
    fun vh hvh => hk vh fun k c => ?_
  · exact Tile2.ptr_lane_seq_of_on d.h hl.hh
      (fun i j hi => Tile2.colMul_toInt_on hl.ov (IntV.toInt_ofNat (w := 64) hK63) hfit i j hi)
      (fun i j _ => Tile2.row_toInt (κ := fun j : Fin d.bk => t * d.bk + j) hok i j)
  · intro c
    rw [Tensor.get1_ltS, IntV.lt_eq, hl.ov c, IntV.toInt_ofNat (w := 64) hV63]
    simp only [Nat.cast_lt]
  · intro j
    rw [Tensor.get1_ltS, IntV.lt_eq, hok j, IntV.toInt_ofNat (w := 32) hK31]
    simp only [Nat.cast_lt]
  · exact (Buf.tileAt_sub_bytes d.h hlen).trans (fun x hx => Or.inr (Or.inr (Or.inr (Or.inl hx))))
  · have hidx : (g.itg * d.HV + g.ih) * d.K * d.V + (g.iv * d.bk + c) * d.K + (t * d.bk + k) * 1 =
        hIdx d r g.ih (t * d.bk + k) (g.iv * d.bk + c) := by
      unfold hIdx; simp only [hs, ↓reduceIte]; rw [← hg.itg_eq]; ring
    rw [Tensor.get2_trans, hvh c k]
    unfold hTile Geo.colOk
    rw [hidx]
    by_cases h1 : g.iv * d.bk + c < d.V <;> by_cases h2 : t * d.bk + k < d.K <;> simp [h1, h2]

/-- **The load of the state, otherwise** (lines 143–150): the tile is `[K, V]`; lane `(k, c)` is
the state element `(x = t·BK + k, e = iv·BV + c)`. -/
theorem safe_load_h (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) (hm : m = init) (hs : d.svf = false) {t : Nat}
    {o_k : Tensor [d.bk] I32}
    (hok : ∀ j : Fin d.bk, (o_k.get1 j).toInt = ((t * d.bk + j : Nat) : Int))
    {k' : Tensor [d.bk, d.bk] BF16 → Prog α}
    (hk : ∀ vh : Tensor [d.bk, d.bk] BF16,
      (∀ (k c : Fin d.bk), vh.get2 k c = hTile d g r t k c) → Safe E (Rs d) W Q m (k' vh)) :
    Safe E (Rs d) W Q m (tl.convert .f32 .bf16 (tl.full [d.bk, d.bk] zf32) >>= fun z =>
      tl.load (h + o_k.col d.bk * (IntV.ofNat d.V : I32) + o_v.row d.bk)
        (some (((o_k.ltS (IntV.ofNat d.K : I32)).col d.bk).and
          ((o_v.row d.bk).ltS (IntV.ofNat d.V : I64)))) (some z) >>= k') := by
  subst hm
  have hb := bk_le hd
  have hK0 := hd.K_pos
  have hV0 := hd.V_pos
  have hKV : d.K * d.V ≤ 2 ^ 17 := by
    have := hd.K_le; have := hd.V_le
    calc d.K * d.V ≤ 256 * 512 := Nat.mul_le_mul (by omega) (by omega)
      _ = 2 ^ 17 := by norm_num
  have hlen : d.K = 0 ∨ d.V = 0 ∨ (g.itg * d.HV + g.ih) * d.K * d.V + (d.K - 1) * d.V +
      (d.V - 1) * 1 < d.h.len :=
    Or.inr (Or.inr (state_last_lt hK0 hV0 (state_row_lt hg) hd.h_len))
  have hfit : (d.K - 1) * d.V < 2 ^ (32 - 1) := by
    have : (d.K - 1) * d.V ≤ d.K * d.V := Nat.mul_le_mul_right _ (Nat.sub_le _ _)
    exact lt_of_le_of_lt (this.trans hKV) (by norm_num)
  have hK31 : d.K < 2 ^ 31 := by have := hd.K_le; omega
  have hV31 : d.V < 2 ^ 31 := by have := hd.V_le; omega
  have hV63 : d.V < 2 ^ 63 := by have := hd.V_le; omega
  refine safe_load_tileV d.h d.hv hpre.h_holds rfl
    (ρ := fun j : Fin d.bk => t * d.bk + j) (κ := fun c : Fin d.bk => g.iv * d.bk + c)
    (M := d.K) (N := d.V) (e := (g.itg * d.HV + g.ih) * d.K * d.V) (s0 := d.V) (s1 := 1)
    ?_ (maskAt_colrow (b := o_v.ltS (IntV.ofNat d.V : I64)) ?_ ?_) hlen hpre.h_valid ?_
    fun vh hvh => hk vh fun k c => ?_
  · exact Tile2.ptr_lane_seq_of_on d.h hl.hh
      (fun i j hi => Tile2.colMul_toInt_on hok (IntV.toInt_ofNat (w := 32) hV31) hfit i j hi)
      (fun i j _ => Tile2.row_toInt (κ := fun c : Fin d.bk => g.iv * d.bk + c) hl.ov i j)
  · intro j
    rw [Tensor.get1_ltS, IntV.lt_eq, hok j, IntV.toInt_ofNat (w := 32) hK31]
    simp only [Nat.cast_lt]
  · intro c
    rw [Tensor.get1_ltS, IntV.lt_eq, hl.ov c, IntV.toInt_ofNat (w := 64) hV63]
    simp only [Nat.cast_lt]
  · exact (Buf.tileAt_sub_bytes d.h hlen).trans (fun x hx => Or.inr (Or.inr (Or.inr (Or.inl hx))))
  · have hidx : (g.itg * d.HV + g.ih) * d.K * d.V + (t * d.bk + k) * d.V + (g.iv * d.bk + c) * 1 =
        hIdx d r g.ih (t * d.bk + k) (g.iv * d.bk + c) := by
      unfold hIdx; simp only [hs, Bool.false_eq_true, ↓reduceIte]; rw [← hg.itg_eq]; ring
    rw [hvh k c]
    unfold hTile Geo.colOk
    rw [hidx]

end Claims.ChunkFwdO
