import GroupedGemm.KLoop

/-!
# Grouped GEMM: one tile

`safe_tile`: the `while` body on global tile `ti` of problem `g` loads the problem's leading
dimensions and addresses, finds the tile's coordinates, runs the `K` loop (`safe_kloop`),
converts the accumulator to `float16` and stores it. Afterwards every element of `C` in tile `ti`
is `Good`: the rounding of a finite `float32` value within `errC` of the exact product.
-/
namespace TVAExamples.GroupedGemm

open TVA TVA.Math.Float

/-! ## Footprints and results -/

/-- Everything the kernel reads: the metadata and every problem's `A` and `B`. -/
def inputs (d : Inst) : Set Int :=
  {x | x ∈ d.sizes.bytes ∨ x ∈ d.lds.bytes ∨ x ∈ d.aptrs.bytes ∨ x ∈ d.bptrs.bytes ∨ x ∈ d.cptrs.bytes ∨
    ∃ g, g < d.G ∧ (x ∈ (d.gemm g).a.bytes ∨ x ∈ (d.gemm g).b.bytes)}

/-- The bytes of the `C` elements in global tile `t`. -/
def tileSet (d : Inst) (t : Nat) : Set Int :=
  {x | ∃ g i j, g < d.G ∧ i < (d.gemm g).M ∧ j < (d.gemm g).N ∧ d.tileOf g i j = t ∧
    x ∈ Mem.span .f16 ((d.gemm g).cAt i j)}

/-- Element `(i, j)` of problem `g`'s `C` holds its specified value in `m`. -/
def Good (d : Inst) (init m : Mem) (g i j : Nat) : Prop :=
  ∃ (o : F32) (c : F16), m.read .f16 ((d.gemm g).cAt i j) = some c ∧ FloatV.NaNEq (FloatV.cvt o) c ∧
    o.isFinite = true ∧
    |o.toRat - ∑ k ∈ Finset.range (d.gemm g).K, (d.gemm g).A init i k * (d.gemm g).B init k j| ≤
      errC d (d.gemm g).K (∑ k ∈ Finset.range (d.gemm g).K, |(d.gemm g).A init i k * (d.gemm g).B init k j|)

/-! ## Arithmetic -/

theorem ok_gemm {d : Inst} (hd : d.ok) {g : Nat} (hg : g < d.G) : Gemm.ok d (d.gemm g) := hd.1 g hg

theorem BM_pos {d : Inst} (hd : d.ok) : 0 < d.BM := by
  obtain ⟨e, he⟩ := hd.2.1; rw [he]; exact Nat.two_pow_pos _
theorem BN_pos {d : Inst} (hd : d.ok) : 0 < d.BN := by
  obtain ⟨e, he⟩ := hd.2.2.1; rw [he]; exact Nat.two_pow_pos _
theorem BK_pos {d : Inst} (hd : d.ok) : 0 < d.BK := by
  obtain ⟨e, he⟩ := hd.2.2.2.1; rw [he]; exact Nat.two_pow_pos _

theorem BK_le {d : Inst} (hd : d.ok) : d.BK ≤ 2 ^ 20 :=
  (Nat.le_mul_of_pos_left _ (BM_pos hd)).trans hd.2.2.2.2.1

theorem BM_le {d : Inst} (hd : d.ok) : d.BM ≤ 2 ^ 20 :=
  (Nat.le_mul_of_pos_right _ (BK_pos hd)).trans hd.2.2.2.2.1

theorem BN_le {d : Inst} (hd : d.ok) : d.BN ≤ 2 ^ 20 :=
  (Nat.le_mul_of_pos_left _ (BK_pos hd)).trans hd.2.2.2.2.2.1

theorem Inst.start_le (d : Inst) {g g' : Nat} (h : g ≤ g') : d.start g ≤ d.start g' := by
  induction g' with
  | zero => rw [Nat.le_zero.mp h]
  | succ n ih =>
    rcases Nat.lt_or_eq_of_le h with h' | h'
    · exact (ih (Nat.le_of_lt_succ h')).trans (by simp only [Inst.start]; omega)
    · rw [h']

/-- `cdiv` of a multiple. -/
theorem cdiv_dvd {n b : Nat} (hb : 0 < b) (h : b ∣ n) : (n + (b - 1)) / b = n / b := by
  obtain ⟨q, rfl⟩ := h
  rw [Nat.mul_add_div hb, Nat.div_eq_of_lt (by omega), Nat.mul_div_cancel_left _ hb, Nat.add_zero]

/-- `(q·b + r) / b = q` and `% b = r` for `r < b`: positions in a row-major grid are unique. -/
theorem grid_unique {n a b a' b' : Nat} (hb : b < n) (hb' : b' < n) (h : a * n + b = a' * n + b') :
    a = a' ∧ b = b' := by
  have hn : 0 < n := by omega
  have e1 : (a * n + b) / n = a := by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hb]; simp
  have e2 : (a' * n + b') / n = a' := by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hb']; simp
  have ha : a = a' := by rw [← e1, ← e2, h]
  subst ha
  exact ⟨rfl, by omega⟩

/-- The value of `tl.arange(0, n)` at lane `r`. -/
theorem arange_toInt {n : Nat} (hn : n < 2 ^ 31) (r : Nat) (hr : r < n) :
    (IntV.ofInt (((0 : ℕ) : ℤ) + (r : ℤ)) : I32).toInt = (r : ℤ) := by
  rw [IntV.I32.toInt_ofInt (by constructor <;> push_cast <;> omega)]; simp

/-- A scalar pointer plus an `int32`: `x_ptr + i` is element `i` further. -/
theorem ptr_add {dt : DType} (b : Buf dt) (i : Int) (x : I32) : b.ptr i + x = b.ptr (i + x.toInt) := by
  simp only [Ptr.add_int, Buf.ptr_offset]

/-- A pointer tile `base + rows[:, None] * ld + cols[None, :]`, element by element. -/
theorem ptr2_get {M N : Nat} (base : Ptr .f16) (rowOff : Tensor [M] I32) (ld : I32) (colOff : Tensor [N] I32)
    (r : Fin M) (c : Fin N) :
    ((base + rowOff.unsqueeze1 * ld).bcastCols N + colOff.unsqueeze0.bcastRows M : Tensor [M, N] (Ptr .f16)).get2 r c =
      (base.offset (rowOff.get1 r * ld).toInt).offset (colOff.get1 c).toInt := rfl

/-- The same with the leading dimension on the left (`ldc * offs_cm[:, None]`). -/
theorem ptr2_get' {M N : Nat} (base : Ptr .f16) (rowOff : Tensor [M] I32) (ld : I32) (colOff : Tensor [N] I32)
    (r : Fin M) (c : Fin N) :
    ((base + ld * rowOff.unsqueeze1).bcastCols N + colOff.unsqueeze0.bcastRows M : Tensor [M, N] (Ptr .f16)).get2 r c =
      (base.offset (ld * rowOff.get1 r).toInt).offset (colOff.get1 c).toInt := rfl

/-- `t * B + tl.arange(0, B)` at lane `r` is `t·B + r`. -/
theorem offs_toInt {B : Nat} (hB : B < 2 ^ 31) {t : I32} {tn : Nat} (ht : t.toInt = (tn : ℤ)) (r : Fin B)
    (hfit : tn * B + B < 2 ^ 31) :
    ((t * (IntV.ofNat B : I32) + tl.arange 0 B : Tensor [B] I32).get1 r).toInt = ((tn * B + r : ℕ) : ℤ) := by
  show (t * (IntV.ofNat B : I32) + IntV.ofInt (((0 : ℕ) : ℤ) + ((r : ℕ) : ℤ))).toInt = _
  have hr := r.isLt
  exact IntV.I32.add_nat (IntV.I32.mul_nat ht (IntV.I32.ofNat_nat hB) (by nlinarith))
    (arange_toInt (n := B) (by omega) r hr) (by nlinarith)

theorem arange_get1 {B : Nat} (hB : B < 2 ^ 31) (r : Fin B) :
    ((tl.arange 0 B : Tensor [B] I32).get1 r).toInt = ((r : ℕ) : ℤ) :=
  arange_toInt (n := B) hB r r.isLt

/-! ## The tile -/

section Tile

variable {E : Env} {R W : Set Int} {m : Mem}

/-- The facts about memory a tile of problem `g` needs: it agrees with `init` on the inputs, and
`C` is still a valid buffer. -/
structure TileMem (d : Inst) (init m : Mem) (g : Nat) : Prop where
  agree : ∀ x ∈ inputs d, m x = init x
  cvalid : (d.gemm g).c.Valid m

theorem inputs_a {d : Inst} {g : Nat} (hg : g < d.G) : (d.gemm g).a.bytes ⊆ inputs d :=
  fun _ hx => Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨g, hg, Or.inl hx⟩))))
theorem inputs_b {d : Inst} {g : Nat} (hg : g < d.G) : (d.gemm g).b.bytes ⊆ inputs d :=
  fun _ hx => Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨g, hg, Or.inr hx⟩))))

/-- Reading an input in `m` is reading it in `init`. -/
theorem read_input {d : Inst} {init : Mem} (hm : ∀ x ∈ inputs d, m x = init x) {dt : DType} {a : Int}
    (h : Mem.span dt a ⊆ inputs d) : m.read dt a = init.read dt a :=
  Mem.read_congr fun x hx => hm x (h hx)

theorem span_sub_of {dt : DType} (b : Buf dt) {S : Set Int} (hS : b.bytes ⊆ S) {i : Nat} (hi : i < b.len) :
    Mem.span dt (b.ptr (i : ℤ)).addr ⊆ S := (b.span_sub hi).trans hS

/-- **`tl.cdiv(x, div)`** of a nonnegative `int32` by a positive `constexpr`. -/
theorem safe_cdiv {γ : Type} {Q : γ → Mem → Prop} {x : I32} {n div : Nat} (hx : x.toInt = (n : ℤ))
    (hdiv : 0 < div) (hfit : n + div < 2 ^ 31) {k : I32 → Prog γ}
    (hk : ∀ r : I32, r.toInt = (((n + (div - 1)) / div : ℕ) : ℤ) → Safe E R W Q m (k r)) :
    Safe E R W Q m (cdiv x div >>= k) := by
  unfold cdiv
  rw [bind_assoc]
  have hs : (x + (IntV.ofNat (div - 1) : I32)).toInt = ((n + (div - 1) : ℕ) : ℤ) :=
    IntV.I32.add_nat hx (IntV.I32.ofNat_nat (by omega)) (by omega)
  have hd : (IntV.ofNat div : I32).toInt = (div : ℤ) := IntV.I32.ofNat_nat (by omega)
  refine safe_idiv1 (by rw [hs]; positivity) (by rw [hd]; exact_mod_cast hdiv) fun r hr => ?_
  exact hk r.item (by rw [hr, hs, hd]; push_cast; rfl)

/-- **One tile.** Program state: `g` is problem `gv`, `ti` is global tile `tiv` of it. -/
theorem safe_tile {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (hdot : Rules.Num.DotTotal E.P cfg) (hacc : Rules.Num.DotAcc E.P cfg (fun _ => True) rel abs)
    {g ti : Nat} (hg : g < d.G) (hlo : d.start g ≤ ti) (hhi : ti < d.start g + d.tiles g)
    {gv gk nn last tiv : I32} (hgv : gv.toInt = (g : ℤ)) (hgk : gk.toInt = ((d.gemm g).K : ℤ))
    (hnn : nn.toInt = (((d.gemm g).N / d.BN : ℕ) : ℤ)) (hlast : last.toInt = (d.start g : ℤ))
    (htiv : tiv.toInt = (ti : ℤ))
    (hR : inputs d ⊆ R) (hW : tileSet d ti ⊆ W) (hm : TileMem d init m g)
    {Q' : I32 → Mem → Prop}
    (hQ : ∀ m', (∀ i j, i < (d.gemm g).M → j < (d.gemm g).N → d.tileOf g i j = ti → Good d init m' g i j) →
      (∀ x, x ∉ tileSet d ti → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Q' (tiv + (IntV.ofNat d.NUM_SM : I32)) m') :
    Safe E R W Q' m (tile gv gk nn last (d.aptrs.ptr 0) (d.bptrs.ptr 0) (d.cptrs.ptr 0) (d.lds.ptr 0)
      d.NUM_SM d.BM d.BN d.BK tiv) := by
  have hok := ok_gemm hd hg
  obtain ⟨hal, hbl, hcl, hKa, hNb, hNc, hBMd, hBNd, hBKd, hMfit, hNfit, hKfit, hlda31, hldb31, hldc31,
    hMlda, hBKldb, hMldc, -⟩ := hok
  obtain ⟨hM, hN, hK, hlda, hldb, hldc, ⟨xa, hxa, hxab⟩, ⟨xb, hxb, hxbb⟩, ⟨xc, hxc, hxcb⟩, hva, hvb, hvc,
    hfa, hfb⟩ := hpre g hg
  obtain ⟨-, -, -, -, -, -, -, -, h3G, hstartG, haptrs, hbptrs, hcptrs, hsizes, hlds, -⟩ := id hd
  set p := d.gemm g with hp
  have hBM := BM_pos hd
  have hBN := BN_pos hd
  have hBK := BK_pos hd
  -- the tile's coordinates
  obtain ⟨nmt, hnmt⟩ : ∃ x, x = p.M / d.BM := ⟨_, rfl⟩
  obtain ⟨nnt, hnnt⟩ : ∃ x, x = p.N / d.BN := ⟨_, rfl⟩
  have htiles : d.tiles g = nmt * nnt := by rw [hnmt, hnnt]; rfl
  obtain ⟨ℓ, hℓ⟩ : ∃ x, x = ti - d.start g := ⟨_, rfl⟩
  have hℓlt : ℓ < nmt * nnt := by rw [← htiles]; omega
  have hnn0 : 0 < nnt := by
    rcases Nat.eq_zero_or_pos nnt with h | h
    · rw [h, Nat.mul_zero] at hℓlt; omega
    · exact h
  obtain ⟨tm, htmd⟩ : ∃ x, x = ℓ / nnt := ⟨_, rfl⟩
  obtain ⟨tn, htnd⟩ : ∃ x, x = ℓ % nnt := ⟨_, rfl⟩
  have htm : tm < nmt := by
    rw [htmd, Nat.div_lt_iff_lt_mul hnn0]; exact hℓlt
  have htn : tn < nnt := by rw [htnd]; exact Nat.mod_lt _ hnn0
  have hℓeq : tm * nnt + tn = ℓ := by rw [htmd, htnd, Nat.mul_comm]; exact Nat.div_add_mod ℓ nnt
  have hrowM : ∀ r : Fin d.BM, tm * d.BM + r < p.M := fun r => by
    have h1 : (tm + 1) * d.BM ≤ nmt * d.BM := Nat.mul_le_mul_right _ htm
    have h2 : nmt * d.BM = p.M := by rw [hnmt]; exact Nat.div_mul_cancel hBMd
    have := r.isLt
    calc tm * d.BM + r < tm * d.BM + d.BM := by omega
      _ = (tm + 1) * d.BM := by ring
      _ ≤ p.M := h2 ▸ h1
  have hcolN : ∀ c : Fin d.BN, tn * d.BN + c < p.N := fun c => by
    have h1 : (tn + 1) * d.BN ≤ nnt * d.BN := Nat.mul_le_mul_right _ htn
    have h2 : nnt * d.BN = p.N := by rw [hnnt]; exact Nat.div_mul_cancel hBNd
    have := c.isLt
    calc tn * d.BN + c < tn * d.BN + d.BN := by omega
      _ = (tn + 1) * d.BN := by ring
      _ ≤ p.N := h2 ▸ h1
  -- memory facts
  have hagree := hm.agree
  have hlds_in : d.lds.bytes ⊆ inputs d := fun _ hx => Or.inr (Or.inl hx)
  have haptrs_in : d.aptrs.bytes ⊆ inputs d := fun _ hx => Or.inr (Or.inr (Or.inl hx))
  have hbptrs_in : d.bptrs.bytes ⊆ inputs d := fun _ hx => Or.inr (Or.inr (Or.inr (Or.inl hx)))
  have hcptrs_in : d.cptrs.bytes ⊆ inputs d := fun _ hx => Or.inr (Or.inr (Or.inr (Or.inr (Or.inl hx))))
  have h3 : (3 : I32).toInt = 3 := IntV.I32.toInt_ofNat (by norm_num)
  have h1 : (1 : I32).toInt = 1 := IntV.I32.toInt_ofNat (by norm_num)
  have h2 : (2 : I32).toInt = 2 := IntV.I32.toInt_ofNat (by norm_num)
  have hg3 : (gv * 3).toInt = ((g * 3 : ℕ) : ℤ) := IntV.I32.mul_nat hgv h3 (by omega)
  have hp0 : ∀ q : Nat, q < 3 → d.lds.ptr 0 + gv * 3 + (IntV.ofNat q : I32) = d.lds.ptr ((3 * g + q : ℕ) : ℤ) := by
    intro q hq
    rw [ptr_add, ptr_add, hg3, IntV.I32.ofNat_nat (by omega)]; congr 1; push_cast; ring
  unfold tile
  -- `lda`, `ldb`, `ldc`
  have e0 : d.lds.ptr 0 + gv * 3 = d.lds.ptr ((3 * g : ℕ) : ℤ) := by
    rw [ptr_add, hg3]; congr 1; push_cast; ring
  refine safe_load1 (v0 := IntV.ofNat p.lda) (by
      rw [e0, read_input hagree (span_sub_of _ hlds_in (by omega))]; exact hlda)
    (by rw [e0]; exact (span_sub_of (i := 3 * g) _ hlds_in (by omega)).trans hR) ?_
  refine safe_load1 (v0 := IntV.ofNat p.ldb) (by
      rw [show (1 : I32) = IntV.ofNat 1 from rfl, hp0 1 (by norm_num),
        read_input hagree (span_sub_of _ hlds_in (by omega))]; exact hldb)
    (by rw [show (1 : I32) = IntV.ofNat 1 from rfl, hp0 1 (by norm_num)]
        exact (span_sub_of (i := 3 * g + 1) _ hlds_in (by omega)).trans hR) ?_
  refine safe_load1 (v0 := IntV.ofNat p.ldc) (by
      rw [show (2 : I32) = IntV.ofNat 2 from rfl, hp0 2 (by norm_num),
        read_input hagree (span_sub_of _ hlds_in (by omega))]; exact hldc)
    (by rw [show (2 : I32) = IntV.ofNat 2 from rfl, hp0 2 (by norm_num)]
        exact (span_sub_of (i := 3 * g + 2) _ hlds_in (by omega)).trans hR) ?_
  -- the three addresses
  have eg : ∀ {dt : DType} (b : Buf dt), b.ptr 0 + gv = b.ptr (g : ℤ) := fun b => by
    rw [ptr_add, hgv, zero_add]
  refine safe_load1 (v0 := xa) (by
      rw [eg, read_input hagree (span_sub_of _ haptrs_in (by omega))]; exact hxa)
    (by rw [eg]; exact (span_sub_of (i := g) _ haptrs_in (by omega)).trans hR) ?_
  refine safe_load1 (v0 := xb) (by
      rw [eg, read_input hagree (span_sub_of _ hbptrs_in (by omega))]; exact hxb)
    (by rw [eg]; exact (span_sub_of (i := g) _ hbptrs_in (by omega)).trans hR) ?_
  refine safe_load1 (v0 := xc) (by
      rw [eg, read_input hagree (span_sub_of _ hcptrs_in (by omega))]; exact hxc)
    (by rw [eg]; exact (span_sub_of (i := g) _ hcptrs_in (by omega)).trans hR) ?_
  have hap : tl.intToPtr .f16 xa = p.a.ptr 0 := by
    show (⟨(xa.bits.toNat : Int)⟩ : Ptr .f16) = ⟨p.a.base + 0 * DType.f16.bytes⟩; rw [hxab]; simp [hp]
  have hbp : tl.intToPtr .f16 xb = p.b.ptr 0 := by
    show (⟨(xb.bits.toNat : Int)⟩ : Ptr .f16) = ⟨p.b.base + 0 * DType.f16.bytes⟩; rw [hxbb]; simp [hp]
  have hcp : tl.intToPtr .f16 xc = p.c.ptr 0 := by
    show (⟨(xc.bits.toNat : Int)⟩ : Ptr .f16) = ⟨p.c.base + 0 * DType.f16.bytes⟩; rw [hxcb]; simp [hp]
  -- `tile_m_idx`, `tile_n_idx`
  have hti31 : ti < 2 ^ 31 := by
    have := d.start_le (Nat.succ_le_of_lt hg); simp only [Inst.start] at this; omega
  have hsub : (tiv - last).toInt = (ℓ : ℤ) := by rw [hℓ]; exact IntV.I32.sub_nat htiv hlast hlo hti31
  have hnn' : 0 < nn.toInt := by rw [hnn, ← hnnt]; exact_mod_cast hnn0
  refine safe_idiv1 (by rw [hsub]; positivity) hnn' fun qm hqm => ?_
  refine safe_irem1 (by rw [hsub]; positivity) hnn' fun qn hqn => ?_
  have htm' : qm.item.toInt = (tm : ℤ) := by rw [hqm, hsub, hnn, ← hnnt, htmd]; push_cast; rfl
  have htn' : qn.item.toInt = (tn : ℤ) := by rw [hqn, hsub, hnn, ← hnnt, htnd]; push_cast; rfl
  -- the tile's rows and columns
  obtain ⟨row, hrow⟩ : ∃ f : Fin d.BM → ℕ, f = fun (r : Fin d.BM) => tm * d.BM + (r : ℕ) := ⟨_, rfl⟩
  obtain ⟨col, hcol⟩ : ∃ f : Fin d.BN → ℕ, f = fun (c : Fin d.BN) => tn * d.BN + (c : ℕ) := ⟨_, rfl⟩
  have hrowlt : ∀ r, row r < p.M := fun r => by rw [hrow]; exact hrowM r
  have hcollt : ∀ c, col c < p.N := fun c => by rw [hcol]; exact hcolN c
  have hBM31 : d.BM < 2 ^ 31 := by have := BM_le hd; omega
  have hBN31 : d.BN < 2 ^ 31 := by have := BN_le hd; omega
  have hBK31 : d.BK < 2 ^ 31 := by have := BK_le hd; omega
  have hamfit : tm * d.BM + d.BM < 2 ^ 31 := by
    have := hrowM ⟨d.BM - 1, by omega⟩; simp only at this; omega
  have hbnfit : tn * d.BN + d.BN < 2 ^ 31 := by
    have := hcolN ⟨d.BN - 1, by omega⟩; simp only at this; omega
  have hoffa : ∀ r : Fin d.BM, ((qm.item * (IntV.ofNat d.BM : I32) + tl.arange 0 d.BM : Tensor [d.BM] I32).get1 r).toInt
      = ((row r : ℕ) : ℤ) := fun r => by rw [hrow]; exact offs_toInt hBM31 htm' r hamfit
  have hoffb : ∀ c : Fin d.BN, ((qn.item * (IntV.ofNat d.BN : I32) + tl.arange 0 d.BN : Tensor [d.BN] I32).get1 c).toInt
      = ((col c : ℕ) : ℤ) := fun c => by rw [hcol]; exact offs_toInt hBN31 htn' c hbnfit
  -- memory in `m`
  have hAm : ∀ i k, i < p.M → k < p.K → p.A m i k = p.A init i k := fun i k hi hk => by
    unfold Gemm.A val16
    rw [read_input hagree (span_sub_of _ (inputs_a hg) (by
      rw [hal]
      calc i * p.lda + k < i * p.lda + p.lda := by omega
        _ = (i + 1) * p.lda := by ring
        _ ≤ p.M * p.lda := Nat.mul_le_mul_right _ hi))]
  have hBm : ∀ k j, k < p.K → j < p.N → p.B m k j = p.B init k j := fun k j hk hj => by
    unfold Gemm.B val16
    rw [read_input hagree (span_sub_of _ (inputs_b hg) (by
      rw [hbl]
      calc k * p.ldb + j < k * p.ldb + p.ldb := by omega
        _ = (k + 1) * p.ldb := by ring
        _ ≤ p.K * p.ldb := Nat.mul_le_mul_right _ hk))]
  have hkc : KCtx d p m row col R := by
    refine ⟨ok_gemm hd hg, BK_le hd, hrowlt, hcollt, ⟨fun i k hi hk => ?_, fun k j hk hj => ?_⟩,
      Buf.valid_of_agree hva fun x hx => ?_, Buf.valid_of_agree hvb fun x hx => ?_,
      (inputs_a hg).trans hR, (inputs_b hg).trans hR⟩
    · rw [read_input hagree (span_sub_of _ (inputs_a hg) (by
        rw [hal]
        calc i * p.lda + k < i * p.lda + p.lda := by omega
          _ = (i + 1) * p.lda := by ring
          _ ≤ p.M * p.lda := Nat.mul_le_mul_right _ hi))]
      exact hfa i k hi hk
    · rw [read_input hagree (span_sub_of _ (inputs_b hg) (by
        rw [hbl]
        calc k * p.ldb + j < k * p.ldb + p.ldb := by omega
          _ = (k + 1) * p.ldb := by ring
          _ ≤ p.K * p.ldb := Nat.mul_le_mul_right _ hk))]
      exact hfb k j hk hj
    · rw [hagree x (inputs_a hg hx)]; exact hva.2.2.2 x hx
    · rw [hagree x (inputs_b hg hx)]; exact hvb.2.2.2 x hx
  -- `tl.cdiv(k, BLOCK_SIZE_K)`
  simp only []
  refine safe_cdiv hgk hBK (by omega) fun kt hkt => ?_
  rw [cdiv_dvd hBK hBKd] at hkt
  -- the `K` loop
  have hlda' : (IntV.ofNat p.lda : I32).toInt = (p.lda : ℤ) := IntV.I32.ofNat_nat hlda31
  have hldb' : (IntV.ofNat p.ldb : I32).toInt = (p.ldb : ℤ) := IntV.I32.ofNat_nat hldb31
  have hldc' : (IntV.ofNat p.ldc : I32).toInt = (p.ldc : ℤ) := IntV.I32.ofNat_nat hldc31
  refine safe_kloop hdot hacc hkc hldb' hkt (fun r k => ?_) (fun k c => ?_) fun st hst => ?_
  · rw [ptr2_get, hap, IntV.I32.mul_nat (hoffa r) hlda' (by
      have := hrowlt r
      exact lt_of_le_of_lt (Nat.mul_le_mul_right _ this.le) hMlda), arange_get1 hBK31,
      Buf.ptr_offset, Buf.ptr_offset]
    congr 1; push_cast; ring
  · rw [ptr2_get, hbp, IntV.I32.mul_nat (arange_get1 hBK31 k) hldb' (by
      have := k.isLt
      exact lt_of_le_of_lt (Nat.mul_le_mul_right _ this.le) hBKldb), hoffb, Buf.ptr_offset, Buf.ptr_offset]
    congr 1; push_cast; ring
  obtain ⟨acc, fa, fb⟩ := st
  simp only []
  -- `c = accumulator.to(tl.float16)`
  refine safe_cvt ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩ fun cv hcv => ?_
  -- `tl.store(c_ptrs, c)`
  have hoffa' : ∀ r : Fin d.BM,
      (qm.item * (IntV.ofNat d.BM : I32) + IntV.ofInt (((0 : ℕ) : ℤ) + ((r : ℕ) : ℤ))).toInt = ((row r : ℕ) : ℤ) :=
    fun r => hoffa r
  have hoffb' : ∀ c : Fin d.BN,
      (qn.item * (IntV.ofNat d.BN : I32) + IntV.ofInt (((0 : ℕ) : ℤ) + ((c : ℕ) : ℤ))).toInt = ((col c : ℕ) : ℤ) :=
    fun c => hoffb c
  have hcptr : ∀ (r : Fin d.BM) (c : Fin d.BN),
      ((tl.intToPtr DType.f16 xc).offset ((IntV.ofNat p.ldc : I32) *
        (qm.item * (IntV.ofNat d.BM : I32) + IntV.ofInt (((0 : ℕ) : ℤ) + ((r : ℕ) : ℤ)))).toInt).offset
          (qn.item * (IntV.ofNat d.BN : I32) + IntV.ofInt (((0 : ℕ) : ℤ) + ((c : ℕ) : ℤ))).toInt =
        p.c.ptr ((row r * p.ldc + col c : ℕ) : ℤ) := by
    intro r c
    rw [hcp, IntV.I32.mul_nat hldc' (hoffa' r) (by
      have := hrowlt r
      calc p.ldc * row r ≤ p.ldc * p.M := Nat.mul_le_mul_left _ this.le
        _ = p.M * p.ldc := Nat.mul_comm _ _
        _ < 2 ^ 31 := hMldc), hoffb', Buf.ptr_offset, Buf.ptr_offset]
    congr 1; push_cast; ring
  have hcidx : ∀ i : Fin d.BM × Fin d.BN × Unit, row i.1 * p.ldc + col i.2.1 < p.c.len := by
    rintro ⟨r, c, ⟨⟩⟩
    show row r * p.ldc + col c < p.c.len
    have hr := hrowlt r
    have hc := hcollt c
    rw [hcl]
    calc row r * p.ldc + col c < row r * p.ldc + p.ldc := by omega
      _ = (row r + 1) * p.ldc := by ring
      _ ≤ p.M * p.ldc := Nat.mul_le_mul_right _ hr
  have hdivr : ∀ r : Fin d.BM, row r / d.BM = tm := fun r => by
    rw [hrow]; show (tm * d.BM + r) / d.BM = tm
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hBM, Nat.div_eq_of_lt r.isLt, zero_add]
  have hdivc : ∀ c : Fin d.BN, col c / d.BN = tn := fun c => by
    rw [hcol]; show (tn * d.BN + c) / d.BN = tn
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hBN, Nat.div_eq_of_lt c.isLt, zero_add]
  have htileOf : ∀ r c, d.tileOf g (row r) (col c) = ti := fun r c => by
    unfold Inst.tileOf
    rw [hdivr, hdivc, ← hp, ← hnnt]
    omega
  have hsubT : p.c.elems (Buf.img (s := [d.BM, d.BN]) (fun i => row i.1 * p.ldc + col i.2.1) fun _ => true) ⊆
      tileSet d ti := by
    rintro x ⟨_, ⟨⟨r, c, ⟨⟩⟩, -, rfl⟩, hx⟩
    exact ⟨g, row r, col c, hg, hrowlt r, hcollt c, htileOf r c, hx⟩
  refine safe_store_buf p.c (Buf.addrs_of_get_all (s := [d.BM, d.BN]) _ fun ⟨r, c, ()⟩ => hcptr r c)
    (fun i _ => hcidx i) (fun ⟨r, c, ⟨⟩⟩ ⟨r', c', ⟨⟩⟩ _ _ h => ?_) hm.cvalid (hsubT.trans hW)
    fun m' hm' hfr hpres => .ret (hQ m' (fun i j hi hj hij => ?_) (fun x hx => hfr x fun h => hx (hsubT h)) hpres)
  · -- distinct lanes store distinct elements
    have hc := hcollt c
    have hc' := hcollt c'
    have h' : row r * p.ldc + col c = row r' * p.ldc + col c' := h
    obtain ⟨e1, e2⟩ := grid_unique (n := p.ldc) (by omega) (by omega) h'
    have er : (r : ℕ) = r' := by
      have h1 : row r = tm * d.BM + r := by rw [hrow]
      have h2 : row r' = tm * d.BM + r' := by rw [hrow]
      omega
    have ec : (c : ℕ) = c' := by
      have h1 : col c = tn * d.BN + c := by rw [hcol]
      have h2 : col c' = tn * d.BN + c' := by rw [hcol]
      omega
    rw [Fin.ext er, Fin.ext ec]
  · -- every element of the tile is `Good`
    -- `(i, j)` is lane `(i % BM, j % BN)`
    have hjN : j / d.BN < nnt := by rw [hnnt]; exact Nat.div_lt_div_of_lt_of_dvd hBNd hj
    have hij' : i / d.BM * nnt + j / d.BN = tm * nnt + tn := by
      unfold Inst.tileOf at hij
      rw [← hp, ← hnnt] at hij
      omega
    obtain ⟨e1, e2⟩ := grid_unique hjN htn hij'
    set r : Fin d.BM := ⟨i % d.BM, Nat.mod_lt _ hBM⟩
    set c : Fin d.BN := ⟨j % d.BN, Nat.mod_lt _ hBN⟩
    have hri : row r = i := by
      rw [hrow]; show tm * d.BM + i % d.BM = i
      rw [← e1, Nat.mul_comm]; exact Nat.div_add_mod i d.BM
    have hcj : col c = j := by
      rw [hcol]; show tn * d.BN + j % d.BN = j
      rw [← e2, Nat.mul_comm]; exact Nat.div_add_mod j d.BN
    obtain ⟨hfin, herr⟩ := hst r c
    refine ⟨acc.get2 r c, cv.get (r, c, ()), ?_, hcv (r, c, ()), hfin, ?_⟩
    · have := hm' (r, c, ()) rfl
      unfold Gemm.cAt
      rw [← hri, ← hcj]
      exact this
    · rw [hri, hcj] at herr
      have hS : p.S m i j p.K = ∑ k ∈ Finset.range p.K, p.A init i k * p.B init k j :=
        Finset.sum_congr rfl fun k hk => by
          rw [hAm i k hi (Finset.mem_range.mp hk), hBm k j (Finset.mem_range.mp hk) hj]
      have hSa : p.Sabs m i j p.K = ∑ k ∈ Finset.range p.K, |p.A init i k * p.B init k j| :=
        Finset.sum_congr rfl fun k hk => by
          rw [hAm i k hi (Finset.mem_range.mp hk), hBm k j (Finset.mem_range.mp hk) hj]
      rw [hS, hSa] at herr
      exact herr

end Tile

end TVAExamples.GroupedGemm
