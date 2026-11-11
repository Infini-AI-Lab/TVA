import Proof.Basic

/-!
# `attn_bwd_f16`: loads and stores of tiles of one head

`tl.load` and `tl.store` of a tile of rows `r i` and columns `c j` of head `b` (element
`eIdx d b (r i) (c j)`), or of a vector of rows of `M` or `delta`, in the form the walks need: the
pointers' lanes in, the read (or written) values out, with the footprints inside the inputs (the
outputs' tile).
-/

namespace Claims.AttnBwd

open TVA

/-- Element `(b, r, c)` of a tensor of a launch lies below `G · N_CTX · HEAD_DIM`. -/
theorem eIdx_lt (d : Inst) {b r c : Nat} (hb : b < G d) (hr : r < d.N_CTX) (hc : c < d.HEAD_DIM) :
    eIdx d b r c < G d * d.N_CTX * d.HEAD_DIM := by
  unfold eIdx
  have h1 : (b * d.N_CTX + r + 1) * d.HEAD_DIM ≤ G d * d.N_CTX * d.HEAD_DIM :=
    Nat.mul_le_mul_right _ (by nlinarith)
  nlinarith

/-- Element `(b, r)` of `M` or `delta` lies below `G · N_CTX`. -/
theorem rIdx_lt (d : Inst) {b r : Nat} (hb : b < G d) (hr : r < d.N_CTX) :
    rIdx d b r < G d * d.N_CTX := by
  unfold rIdx
  nlinarith


section Loads

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **A tile of rows `r i` and columns `c j` of head `b`, rows along the first axis**
(`base + rows[:, None] * stride_tok + cols[None, :] * stride_d`): each lane reads element
`eIdx d b (r i) (c j)`. -/
theorem safe_load_rc {dt : DType} (d : Inst) (B : Buf dt) (hlen : G d * d.N_CTX * d.HEAD_DIM ≤ B.len)
    (hsub : B.elems (tSet d) ⊆ R) (hv : B.Valid m) {b : Nat} (hb : b < G d) {A C : Nat}
    {ptrs : Tensor [A, C] (Ptr dt)} (r : Fin A → Nat) (c : Fin C → Nat)
    (hr : ∀ i, r i < d.N_CTX) (hc : ∀ j, c j < d.HEAD_DIM)
    (hp : ∀ i j, ptrs.get2 i j = B.ptr ((eIdx d b (r i) (c j) : Nat) : Int))
    {k : Tensor [A, C] dt.Val → Prog α}
    (hk : ∀ x : Tensor [A, C] dt.Val,
      (∀ i j, m.read dt (B.ptr ((eIdx d b (r i) (c j) : Nat) : Int)).addr = some (x.get2 i j)) →
      Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.load ptrs none none >>= k) := by
  have hN : ∀ i, r i < d.N_CTX := hr
  refine safe_load_tileAt (e := b * d.N_CTX * d.HEAD_DIM) (s0 := d.HEAD_DIM) (s1 := 1)
    (ρ := r) (κ := c) (M := d.N_CTX) (N := d.HEAD_DIM) B ?_ (Tile2.maskAt_none hr hc) ?_ hv ?_
    (fun x hx _ => hk x ?_)
  · intro i j _ _
    rw [hp i j]; congr 2; unfold eIdx; ring
  · by_cases hN0 : d.N_CTX = 0
    · exact Or.inl hN0
    by_cases hH0 : d.HEAD_DIM = 0
    · exact Or.inr (Or.inl hH0)
    refine Or.inr (Or.inr (lt_of_lt_of_le ?_ hlen))
    have hN1 : 1 ≤ d.N_CTX := Nat.one_le_iff_ne_zero.mpr hN0
    have hH1 : 1 ≤ d.HEAD_DIM := Nat.one_le_iff_ne_zero.mpr hH0
    have := eIdx_lt d hb (show d.N_CTX - 1 < d.N_CTX by omega) (show d.HEAD_DIM - 1 < d.HEAD_DIM by omega)
    unfold eIdx at this
    have e : b * d.N_CTX * d.HEAD_DIM + (d.N_CTX - 1) * d.HEAD_DIM + (d.HEAD_DIM - 1) * 1 =
        (b * d.N_CTX + (d.N_CTX - 1)) * d.HEAD_DIM + (d.HEAD_DIM - 1) := by ring
    omega
  · refine subset_trans ?_ hsub
    refine B.elems_mono ?_
    rintro x ⟨i, j, hi, hj, rfl⟩
    have := eIdx_lt d hb hi hj
    unfold eIdx at this
    show (_ : Nat) < _
    have e : b * d.N_CTX * d.HEAD_DIM + r i * d.HEAD_DIM + c j * 1 =
        (b * d.N_CTX + r i) * d.HEAD_DIM + c j := by ring
    omega
  · intro i j
    have e : b * d.N_CTX * d.HEAD_DIM + r i * d.HEAD_DIM + c j * 1 = eIdx d b (r i) (c j) := by
      unfold eIdx; ring
    have := hx i j (hr i) (hc j)
    rwa [e] at this

/-- **A transposed tile of head `b`, columns along the first axis**
(`base + cols[:, None] * stride_d + rows[None, :] * stride_tok`): lane `(j, i)` reads element
`eIdx d b (r i) (c j)`. -/
theorem safe_load_cr {dt : DType} (d : Inst) (B : Buf dt) (hlen : G d * d.N_CTX * d.HEAD_DIM ≤ B.len)
    (hsub : B.elems (tSet d) ⊆ R) (hv : B.Valid m) {b : Nat} (hb : b < G d) {A C : Nat}
    {ptrs : Tensor [C, A] (Ptr dt)} (r : Fin A → Nat) (c : Fin C → Nat)
    (hr : ∀ i, r i < d.N_CTX) (hc : ∀ j, c j < d.HEAD_DIM)
    (hp : ∀ j i, ptrs.get2 j i = B.ptr ((eIdx d b (r i) (c j) : Nat) : Int))
    {k : Tensor [C, A] dt.Val → Prog α}
    (hk : ∀ x : Tensor [C, A] dt.Val,
      (∀ j i, m.read dt (B.ptr ((eIdx d b (r i) (c j) : Nat) : Int)).addr = some (x.get2 j i)) →
      Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.load ptrs none none >>= k) := by
  refine safe_load_tileAt (e := b * d.N_CTX * d.HEAD_DIM) (s0 := 1) (s1 := d.HEAD_DIM)
    (ρ := c) (κ := r) (M := d.HEAD_DIM) (N := d.N_CTX) B ?_ (Tile2.maskAt_none hc hr) ?_ hv ?_
    (fun x hx _ => hk x ?_)
  · intro j i _ _
    rw [hp j i]; congr 2; unfold eIdx; ring
  · by_cases hN0 : d.N_CTX = 0
    · exact Or.inr (Or.inl hN0)
    by_cases hH0 : d.HEAD_DIM = 0
    · exact Or.inl hH0
    refine Or.inr (Or.inr (lt_of_lt_of_le ?_ hlen))
    have hN1 : 1 ≤ d.N_CTX := Nat.one_le_iff_ne_zero.mpr hN0
    have hH1 : 1 ≤ d.HEAD_DIM := Nat.one_le_iff_ne_zero.mpr hH0
    have := eIdx_lt d hb (show d.N_CTX - 1 < d.N_CTX by omega) (show d.HEAD_DIM - 1 < d.HEAD_DIM by omega)
    unfold eIdx at this
    have e : b * d.N_CTX * d.HEAD_DIM + (d.HEAD_DIM - 1) * 1 + (d.N_CTX - 1) * d.HEAD_DIM =
        (b * d.N_CTX + (d.N_CTX - 1)) * d.HEAD_DIM + (d.HEAD_DIM - 1) := by ring
    omega
  · refine subset_trans ?_ hsub
    refine B.elems_mono ?_
    rintro x ⟨j, i, hj, hi, rfl⟩
    have := eIdx_lt d hb hi hj
    unfold eIdx at this
    show (_ : Nat) < _
    have e : b * d.N_CTX * d.HEAD_DIM + c j * 1 + r i * d.HEAD_DIM =
        (b * d.N_CTX + r i) * d.HEAD_DIM + c j := by ring
    omega
  · intro j i
    have e : b * d.N_CTX * d.HEAD_DIM + c j * 1 + r i * d.HEAD_DIM = eIdx d b (r i) (c j) := by
      unfold eIdx; ring
    have := hx j i (hc j) (hr i)
    rwa [e] at this

/-- **A vector of rows of head `b` of `M` or `delta`** (`base + offs`): lane `i` reads element
`rIdx d b (r i)`. -/
theorem safe_load_vec {dt : DType} (d : Inst) (B : Buf dt) (hlen : G d * d.N_CTX ≤ B.len)
    (hsub : B.elems (rSet d) ⊆ R) (hv : B.Valid m) {b : Nat} (hb : b < G d) {A : Nat}
    {ptrs : Tensor [A] (Ptr dt)} (r : Fin A → Nat) (hr : ∀ i, r i < d.N_CTX)
    (hp : ∀ i, ptrs.get1 i = B.ptr ((rIdx d b (r i) : Nat) : Int))
    {k : Tensor [A] dt.Val → Prog α}
    (hk : ∀ x : Tensor [A] dt.Val,
      (∀ i, m.read dt (B.ptr ((rIdx d b (r i) : Nat) : Int)).addr = some (x.get1 i)) →
      Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.load ptrs none none >>= k) := by
  refine safe_load_buf B (g := fun i : Idx [A] => rIdx d b (r i.1)) (en := fun _ => true)
    (Buf.addrs_of_get_all B (fun i : Idx [A] => hp i.1)) (fun i _ => lt_of_lt_of_le (rIdx_lt d hb (hr _)) hlen)
    hv ?_ (fun x hx _ => hk x ?_)
  · refine subset_trans (B.elems_mono ?_) hsub
    rintro _ ⟨i, -, rfl⟩
    exact rIdx_lt d hb (hr _)
  · intro i
    exact hx (i, ()) rfl

end Loads

/-! ## The pointer tiles -/

/-- **The pointers `base + (c0 + arange C)[None, :] * s1 + arange(R)[:, None] * s2`** (a transposed
tile, `R` rows of the head's columns by `C` query rows): lane `(c, i)` points at `e + c·s2 + (r0 + i)·s1`. -/
theorem ptrT_lane {dt : DType} (B : Buf dt) {base : Ptr dt} {e : Nat} (hb : base = B.ptr (e : Int))
    {R C r0 s1v s2v : Nat} {c0 s1 s2 : I32} (hc : c0.toInt = r0) (h1 : s1.toInt = s1v)
    (h2 : s2.toInt = s2v) (hfit0 : r0 + C ≤ 2 ^ 31) (hfit1 : (r0 + C - 1) * s1v < 2 ^ 31)
    (hR : R ≤ 2 ^ 31) (hfit2 : (R - 1) * s2v < 2 ^ 31) (c : Fin R) (i : Fin C) :
    (base + (c0 + tl.arange 0 C).row R * s1 + (tl.arange 0 R).col C * s2).get2 c i =
      B.ptr ((e + (0 + c) * s2v + (r0 + i) * s1v : Nat) : Int) := by
  refine Tile2.ptr_lane_seq_swap_of B hb (r0 := 0) (c0 := r0) (s0 := s2v) (s1 := s1v) ?_ ?_ c i
  · intro a b'
    exact Tile2.colMul_toInt (rows := tl.arange 0 R) (r0 := 0) (s0 := s2v)
      (fun a' => by simpa using IntV.I32.toInt_tl_arange hR a') h2 (by simpa using hfit2) a b'
  · intro a b'
    exact Tile2.rowMul_toInt (cols := c0 + tl.arange 0 C) (c0 := r0) (s1 := s1v)
      (fun j => IntV.I32.toInt_sadd_arange hc hfit0 j) h1 hfit1 a b'

/-- **The pointers `base + (r0 + arange R)[:, None] * s1 + arange(C)[None, :] * s2`**: lane `(i, j)`
points at `e + (r0 + i)·s1 + j·s2`. -/
theorem ptrN_lane {dt : DType} (B : Buf dt) {base : Ptr dt} {e : Nat} (hb : base = B.ptr (e : Int))
    {R C r0 s1v s2v : Nat} {c0 s1 s2 : I32} (hc : c0.toInt = r0) (h1 : s1.toInt = s1v)
    (h2 : s2.toInt = s2v) (hfit0 : r0 + R ≤ 2 ^ 31) (hfit1 : (r0 + R - 1) * s1v < 2 ^ 31)
    (hC : C ≤ 2 ^ 31) (hfit2 : (C - 1) * s2v < 2 ^ 31) (i : Fin R) (j : Fin C) :
    (base + (c0 + tl.arange 0 R).col C * s1 + (tl.arange 0 C).row R * s2).get2 i j =
      B.ptr ((e + (r0 + i) * s1v + (0 + j) * s2v : Nat) : Int) := by
  refine Tile2.ptr_lane_seq_of B hb (r0 := r0) (c0 := 0) (s0 := s1v) (s1 := s2v) ?_ ?_ i j
  · intro a b'
    exact Tile2.colMul_toInt (rows := c0 + tl.arange 0 R) (r0 := r0) (s0 := s1v)
      (fun a' => IntV.I32.toInt_sadd_arange hc hfit0 a') h1 hfit1 a b'
  · intro a b'
    exact Tile2.rowMul_toInt (cols := tl.arange 0 C) (c0 := 0) (s1 := s2v)
      (fun j => by simpa using IntV.I32.toInt_tl_arange hC j) h2 (by simpa using hfit2) a b'

/-! ## The values read are the head's -/

theorem Q_sub_in (d : Inst) : d.Q.elems (tSet d) ⊆ inBytes d := by
  intro x hx; simp only [inBytes, Set.mem_union]; tauto

/-- The tensor `K` is part of the inputs. -/
theorem K_sub_in (d : Inst) : d.K.elems (tSet d) ⊆ inBytes d := by
  intro x hx; simp only [inBytes, Set.mem_union]; tauto

/-- The tensor `V` is part of the inputs. -/
theorem V_sub_in (d : Inst) : d.V.elems (tSet d) ⊆ inBytes d := by
  intro x hx; simp only [inBytes, Set.mem_union]; tauto

/-- The tensor `dO` is part of the inputs. -/
theorem DO_sub_in (d : Inst) : d.DO.elems (tSet d) ⊆ inBytes d := by
  intro x hx; simp only [inBytes, Set.mem_union]; tauto

/-- The tensor `M` is part of the inputs. -/
theorem M_sub_in (d : Inst) : d.M.elems (rSet d) ⊆ inBytes d := by
  intro x hx; simp only [inBytes, Set.mem_union]; tauto

/-- The tensor `delta` is part of the inputs. -/
theorem D_sub_in (d : Inst) : d.D.elems (rSet d) ⊆ inBytes d := by
  intro x hx; simp only [inBytes, Set.mem_union]; tauto

/-- A memory that agrees with `init` on the inputs reads an input element as `init` does. -/
theorem read_agree {dt : DType} {B : Buf dt} {S : Set Nat} {d : Inst} (hsub : B.elems S ⊆ inBytes d)
    {m0 init : Mem} (hag : ∀ x ∈ inBytes d, m0 x = init x) {j : Nat} (hj : j ∈ S) :
    m0.read dt (B.ptr (j : Int)).addr = init.read dt (B.ptr (j : Int)).addr :=
  Mem.read_congr fun x hx => hag x (hsub (B.span_sub_elems hj hx))

section Values
variable {d : Inst} {init m0 : Mem} (hag : ∀ x ∈ inBytes d, m0 x = init x) {b r c : Nat}
  {hv : HeadV} (hH : hv.Holds d init b) (hb : b < G d) (hr : r < d.N_CTX) (hc : c < d.HEAD_DIM)
include hag hH hb hr hc

/-- A value read from `Q` where the head holds `hv` is the head's `Q`. -/
theorem q_eq {x : F16} (h : m0.read .f16 (d.Q.ptr ((eIdx d b r c : Nat) : Int)).addr = some x) :
    x = hv.q r c := by
  rw [read_agree (Q_sub_in d) hag (j := eIdx d b r c) (eIdx_lt d hb hr hc), (hH r hr).1 c hc |>.1] at h
  exact (Option.some.inj h).symm

/-- A value read from `K` where the head holds `hv` is the head's `K`. -/
theorem k_eq {x : F16} (h : m0.read .f16 (d.K.ptr ((eIdx d b r c : Nat) : Int)).addr = some x) :
    x = hv.k r c := by
  rw [read_agree (K_sub_in d) hag (j := eIdx d b r c) (eIdx_lt d hb hr hc), (hH r hr).1 c hc |>.2.1] at h
  exact (Option.some.inj h).symm

/-- A value read from `V` where the head holds `hv` is the head's `V`. -/
theorem v_eq {x : F16} (h : m0.read .f16 (d.V.ptr ((eIdx d b r c : Nat) : Int)).addr = some x) :
    x = hv.v r c := by
  rw [read_agree (V_sub_in d) hag (j := eIdx d b r c) (eIdx_lt d hb hr hc), (hH r hr).1 c hc |>.2.2.1] at h
  exact (Option.some.inj h).symm

/-- A value read from `DO` where the head holds `hv` is the head's `dO`. -/
theorem do_eq {x : F16} (h : m0.read .f16 (d.DO.ptr ((eIdx d b r c : Nat) : Int)).addr = some x) :
    x = hv.dO r c := by
  rw [read_agree (DO_sub_in d) hag (j := eIdx d b r c) (eIdx_lt d hb hr hc), (hH r hr).1 c hc |>.2.2.2] at h
  exact (Option.some.inj h).symm

end Values

section Rows
variable {d : Inst} {init m0 : Mem} (hag : ∀ x ∈ inBytes d, m0 x = init x) {b r : Nat}
  {hv : HeadV} (hH : hv.Holds d init b) (hb : b < G d) (hr : r < d.N_CTX)
include hag hH hb hr

/-- A value read from `M` where the head holds `hv` is the head's `L`. -/
theorem lse_eq {x : F32} (h : m0.read .f32 (d.M.ptr ((rIdx d b r : Nat) : Int)).addr = some x) :
    x = hv.lse r := by
  rw [read_agree (M_sub_in d) hag (j := rIdx d b r) (rIdx_lt d hb hr), (hH r hr).2.1] at h
  exact (Option.some.inj h).symm

/-- A value read from `delta` where the head holds `hv` is the head's `Δ`. -/
theorem dl_eq {x : F32} (h : m0.read .f32 (d.D.ptr ((rIdx d b r : Nat) : Int)).addr = some x) :
    x = hv.dl r := by
  rw [read_agree (D_sub_in d) hag (j := rIdx d b r) (rIdx_lt d hb hr), (hH r hr).2.2] at h
  exact (Option.some.inj h).symm

end Rows


/-! ## Stores of a tile of rows -/

/-- The elements of rows `n0, …, n0 + 127` of head `b`. -/
def tileEl (d : Inst) (b n0 : Nat) : Set Nat :=
  {e | ∃ r j, r < 128 ∧ j < d.HEAD_DIM ∧ e = eIdx d b (n0 + r) j}

/-- The rows of a tile are elements of the tensor. -/
theorem tileEl_sub (d : Inst) {b n0 : Nat} (hb : b < G d) (hn0 : n0 + 128 ≤ d.N_CTX) :
    tileEl d b n0 ⊆ tSet d := by
  rintro _ ⟨r, j, hr, hj, rfl⟩
  exact eIdx_lt d hb (by omega) hj

section Stores

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **A tile of `128` rows of head `b` stored** (`base + rows[:, None] * stride_tok +
cols[None, :] * stride_d`): afterwards lane `(i, j)`'s element holds its value, nothing outside the
tile changed, every present byte is still present, and the rest of the program has the write
footprint without the tile. -/
theorem safe_store_rc {dt : DType} (d : Inst) (B : Buf dt) (hlen : G d * d.N_CTX * d.HEAD_DIM ≤ B.len)
    (hv : B.Valid m) {b : Nat} (hb : b < G d) (n0 : Nat) (hn0 : n0 + 128 ≤ d.N_CTX)
    {ptrs : Tensor [128, d.HEAD_DIM] (Ptr dt)} {vals : Tensor [128, d.HEAD_DIM] dt.Val}
    (hp : ∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
      ptrs.get2 i j = B.ptr ((eIdx d b (n0 + i) j : Nat) : Int))
    (hW : B.elems (tileEl d b n0) ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m' : Mem,
      (∀ (i : Fin 128) (j : Fin d.HEAD_DIM),
        m'.read dt (B.ptr ((eIdx d b (n0 + i) j : Nat) : Int)).addr = some (vals.get2 i j)) →
      (∀ x, x ∉ B.elems (tileEl d b n0) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ B.elems (tileEl d b n0)) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals none >>= k) := by
  have hset : {x | Tile2.ElemAt (b * d.N_CTX * d.HEAD_DIM) d.HEAD_DIM 1
      (fun i : Fin 128 => n0 + i.val) (fun j : Fin d.HEAD_DIM => j.val) d.N_CTX d.HEAD_DIM x} =
      tileEl d b n0 := by
    ext x
    constructor
    · rintro ⟨i, j, hi, hj, rfl⟩
      exact ⟨i, j, i.isLt, hj, by unfold eIdx; ring⟩
    · rintro ⟨r, j, hr, hj, rfl⟩
      exact ⟨⟨r, hr⟩, ⟨j, hj⟩, by simp only; omega, hj, by unfold eIdx; ring⟩
  have htile : B.tileAt (b * d.N_CTX * d.HEAD_DIM) d.HEAD_DIM 1 (fun i : Fin 128 => n0 + i.val)
      (fun j : Fin d.HEAD_DIM => j.val) d.N_CTX d.HEAD_DIM = B.elems (tileEl d b n0) := by
    unfold Buf.tileAt; rw [hset]
  refine safe_store_tileAt (e := b * d.N_CTX * d.HEAD_DIM) (s0 := d.HEAD_DIM) (s1 := 1)
    (ρ := fun i : Fin 128 => n0 + i.val) (κ := fun j : Fin d.HEAD_DIM => j.val) (M := d.N_CTX)
    (N := d.HEAD_DIM) B ?_ (Tile2.maskAt_none (fun i => by have := i.isLt; omega) (fun j => j.isLt))
    ?_ (Tile2.inj_rowMajor1 le_rfl) ?_ ?_ hv (by rw [htile]; exact hW) ?_
  · intro i j _ _
    rw [hp i j]; congr 2; unfold eIdx; ring
  · by_cases hN0 : d.N_CTX = 0
    · exact Or.inl hN0
    by_cases hH0 : d.HEAD_DIM = 0
    · exact Or.inr (Or.inl hH0)
    refine Or.inr (Or.inr (lt_of_lt_of_le ?_ hlen))
    have hN1 : 1 ≤ d.N_CTX := Nat.one_le_iff_ne_zero.mpr hN0
    have hH1 : 1 ≤ d.HEAD_DIM := Nat.one_le_iff_ne_zero.mpr hH0
    have := eIdx_lt d hb (show d.N_CTX - 1 < d.N_CTX by omega) (show d.HEAD_DIM - 1 < d.HEAD_DIM by omega)
    unfold eIdx at this
    have e : b * d.N_CTX * d.HEAD_DIM + (d.N_CTX - 1) * d.HEAD_DIM + (d.HEAD_DIM - 1) * 1 =
        (b * d.N_CTX + (d.N_CTX - 1)) * d.HEAD_DIM + (d.HEAD_DIM - 1) := by ring
    omega
  · intro i i' _ _ h
    apply Fin.ext; simpa using h
  · intro j j' _ _ h
    exact Fin.ext h
  · intro m' h1 h2 h3
    rw [htile] at h2 ⊢
    refine hk m' (fun i j => ?_) h2 h3
    have e : b * d.N_CTX * d.HEAD_DIM + (n0 + i.val) * d.HEAD_DIM + j.val * 1 = eIdx d b (n0 + i) j := by
      unfold eIdx; ring
    have := h1 i j (by have := i.isLt; omega) j.isLt
    rwa [e] at this

end Stores

/-! ## The mask `n ≤ m` -/

/-- **`tl.where(offs_n[:, None] <= offs_m[None, :], p, 0.0)`** (rows `r0 + i`, columns `c0 + j`):
lane `(i, j)` is `p` where `r0 + i ≤ c0 + j`, and `+0.0` elsewhere. -/
theorem where_lane_N {A C : Nat} {ro : Tensor [A] I32} {co : Tensor [C] I32} {r0 c0 : Nat}
    (hro : ∀ i, (ro.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hco : ∀ j, (co.get1 j).toInt = ((c0 + j : Nat) : Int))
    (pT : Tensor [A, C] F32) (i : Fin A) (j : Fin C) :
    (tl.where ((ro.col C).le (co.row A)) pT (tl.full [A, C] (FloatV.ofPat 0).litF32)).get2 i j =
      if r0 + i ≤ c0 + j then pT.get2 i j else FloatV.ofPat 0 := by
  have h0 : (FloatV.ofPat 0 : F64).litF32 = FloatV.ofPat 0 := FloatV.litF32_of_isZero (by decide)
  show (if (IntV.le (ro.get1 i) (co.get1 j)) = true then pT.get (i, j, ()) else (FloatV.ofPat 0 : F64).litF32) = _
  rw [h0, IntV.le_eq, hro, hco]
  simp only [Nat.cast_le, decide_eq_true_eq]
  rfl

/-- **`tl.where(offs_n[None, :] <= offs_m[:, None], p, 0.0)`**: the same with the roles of rows and
columns exchanged: lane `(i, j)` is `p` where `c0 + j ≤ r0 + i`. -/
theorem where_lane_T {A C : Nat} {ro : Tensor [A] I32} {co : Tensor [C] I32} {r0 c0 : Nat}
    (hro : ∀ i, (ro.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hco : ∀ j, (co.get1 j).toInt = ((c0 + j : Nat) : Int))
    (pT : Tensor [A, C] F32) (i : Fin A) (j : Fin C) :
    (tl.where ((co.row A).le (ro.col C)) pT (tl.full [A, C] (FloatV.ofPat 0).litF32)).get2 i j =
      if c0 + j ≤ r0 + i then pT.get2 i j else FloatV.ofPat 0 := by
  have h0 : (FloatV.ofPat 0 : F64).litF32 = FloatV.ofPat 0 := FloatV.litF32_of_isZero (by decide)
  show (if (IntV.le (co.get1 j) (ro.get1 i)) = true then pT.get (i, j, ()) else (FloatV.ofPat 0 : F64).litF32) = _
  rw [h0, IntV.le_eq, hro, hco]
  simp only [Nat.cast_le, decide_eq_true_eq]
  rfl

end Claims.AttnBwd
