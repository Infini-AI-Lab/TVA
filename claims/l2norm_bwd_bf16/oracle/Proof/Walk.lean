import Proof.Leaf

/-!
# `l2norm_bwd_bf16`: the `Safe` walk of one program

The loads of `y` and `dy` (`safe_loadT16`, `safe_loadT32`: the tile, widened to `float32` when it
is `bfloat16`), the load of `rstd` (`safe_loadR`) and the store of `dx` (`safe_store16`,
`safe_store32`) are stated once for a buffer of either dtype. `safe_program` peels the index
arithmetic (`program_id`, `arange`, the `T64` conversion), splits on the three dtypes, and walks
the rest one `tl.*` operation at a time, ending in `leaf`.
-/

set_option maxHeartbeats 1000000

namespace Claims.L2NormBwd

open TVA

section Access

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {d : Inst} {init : Mem}
  {p : Nat} {o_t : Tensor [d.BT] I64} {T64 : I64}

/-- The row block of program `p` ends within the offset range the `int64` addresses can hold. -/
theorem p_bound (hd : d.ok) (hp : p < grid d) : p * d.BT + d.BT ≤ 2 ^ 51 := by
  have hg : grid d < 2 ^ 31 := hd.grid_lt
  have h31 : p + 1 ≤ 2 ^ 31 := by omega
  have hBT := bt_le hd
  have h1 : p * d.BT + d.BT = (p + 1) * d.BT := by ring
  rw [h1]
  calc (p + 1) * d.BT ≤ 2 ^ 31 * 2 ^ 20 := Nat.mul_le_mul h31 hBT
    _ = 2 ^ 51 := by norm_num

/-- The lanes of the `[BT, BD]` accesses address the tile's elements of a buffer. -/
theorem addrs_of {d' : DType} (b : Buf d') (hd : d.ok) (hp : p * d.BT + d.BT ≤ 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int)) :
    b.Addrs (tl.addrs (pTile (b.ptr 0) d o_t) (some (mX d o_t T64))) (gx d p) (enx d p) :=
  addrs_tile b (by have := hd.d_le; omega) hd.d_le (by have := bd_le20 hd; omega)
    (hrow_of_hot hp hot) hot hT64

/-- **The load of a `bfloat16` tile**, widened to `float32` (source lines 63 and 65). -/
theorem safe_loadT16 (hd : d.ok) (hp : p * d.BT + d.BT ≤ 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int))
    {B : Sum (Buf .bf16) (Buf .f32)} {b : Buf .bf16} (hB : B = .inl b)
    (hlen : d.T * d.D ≤ lenOf B) (hv : validOf B init) (hR : elemsOf B (tileIdx d p) ⊆ R)
    (hfin : ∀ i, enx d p i = true → ∃ v, ValAt B init (gx d p i) v)
    {k : Tensor [d.BT, d.BD] F32 → Prog α}
    (hk : ∀ v, LoadOk B init d p v → Safe E R W Q init (k v)) :
    Safe E R W Q init
      (tl.load (pTile (b.ptr 0) d o_t) (some (mX d o_t T64))
          (some (bc (FloatV.litBF16 (FloatV.ofPat 0)))) >>= fun b16 => tl.cvt b16 >>= k) := by
  subst hB
  refine safe_load_buf b (addrs_of b hd hp hot hT64) (fun i hi => gx_lt_len hd hlen hi) hv
    (by rw [img_x hd]; exact hR) fun v hv1 hv0 => ?_
  refine safe_cvt (g := .binary32) (rnd := .rn)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ fun w hw => hk w ⟨?_, ?_⟩
  · intro i hi
    obtain ⟨_, c, hc, hcf, -⟩ := hfin i hi
    have hvc : v.get i = c := by rw [hv1 i hi] at hc; exact Option.some.inj hc
    obtain ⟨hcvf, hcvs⟩ := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
      Format.bfloat16_regular Format.binary32_regular Format.wider_bfloat16_binary32 .rn (hvc ▸ hcf)
    have hweq := (hw i).eq (FloatV.isFinite_iff.mp hcvf).1
    refine ⟨⟨c, hv1 i hi ▸ hvc ▸ rfl, hcf, ?_⟩, by rw [hweq]; exact hcvf⟩
    simp only [FloatV.toReal]
    rw [hweq, hcvs, hvc]
  · intro i hi
    have hz : v.get i = FloatV.litBF16 (FloatV.ofPat 0) := by
      have h := hv0 i hi (bc (FloatV.litBF16 (FloatV.ofPat 0))) rfl
      simpa only [bc, Tensor.get_full] using h
    have hz' : (FloatV.litBF16 (FloatV.ofPat 0) : BF16) = FloatV.ofPat 0 := by decide
    have hzf : (v.get i).isFinite = true := by rw [hz, hz']; decide
    obtain ⟨hcvf, hcvs⟩ := FloatV.toRat_cvt_of_wider (f := Format.bfloat16) (g := Format.binary32)
      Format.bfloat16_regular Format.binary32_regular Format.wider_bfloat16_binary32 .rn hzf
    have hweq := (hw i).eq (FloatV.isFinite_iff.mp hcvf).1
    refine ⟨by rw [hweq]; exact hcvf, ?_⟩
    have h0 := bf16_zero_toReal
    simp only [FloatV.toReal] at h0 ⊢
    rw [hweq, hcvs, hz, hz', h0]

/-- **The load of a `float32` tile** (source lines 63 and 65; `.to(tl.float32)` is the tile
itself). -/
theorem safe_loadT32 (hd : d.ok) (hp : p * d.BT + d.BT ≤ 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int))
    {B : Sum (Buf .bf16) (Buf .f32)} {b : Buf .f32} (hB : B = .inr b)
    (hlen : d.T * d.D ≤ lenOf B) (hv : validOf B init) (hR : elemsOf B (tileIdx d p) ⊆ R)
    (hfin : ∀ i, enx d p i = true → ∃ v, ValAt B init (gx d p i) v)
    {k : Tensor [d.BT, d.BD] F32 → Prog α}
    (hk : ∀ v, LoadOk B init d p v → Safe E R W Q init (k v)) :
    Safe E R W Q init
      (tl.load (pTile (b.ptr 0) d o_t) (some (mX d o_t T64))
          (some (bc (FloatV.litF32 (FloatV.ofPat 0)))) >>= k) := by
  subst hB
  refine safe_load_buf b (addrs_of b hd hp hot hT64) (fun i hi => gx_lt_len hd hlen hi) hv
    (by rw [img_x hd]; exact hR) fun v hv1 hv0 => hk v ⟨?_, ?_⟩
  · intro i hi
    obtain ⟨_, c, hc, hcf, -⟩ := hfin i hi
    have hvc : v.get i = c := by rw [hv1 i hi] at hc; exact Option.some.inj hc
    exact ⟨⟨c, hv1 i hi ▸ hvc ▸ rfl, hcf, by rw [hvc]⟩, hvc ▸ hcf⟩
  · intro i hi
    have hz : v.get i = FloatV.litF32 (FloatV.ofPat 0) := by
      have h := hv0 i hi (bc (FloatV.litF32 (FloatV.ofPat 0))) rfl
      simpa only [bc, Tensor.get_full] using h
    have hz' : (FloatV.litF32 (FloatV.ofPat 0) : F32) = FloatV.ofPat 0 := by decide
    rw [hz, hz']
    exact ⟨by decide, f32_zero_toReal⟩

/-- **The load of the block's `rstd`** (source line 64). -/
theorem safe_loadR (hd : d.ok)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int))
    (hv : d.rstd.Valid init) (hR : d.rstd.elems (tileRow d p) ⊆ R)
    {k : Tensor [d.BT] F32 → Prog α}
    (hk : ∀ v : Tensor [d.BT] F32, (∀ i, enT d p i = true →
      init.read .f32 (d.rstd.ptr (gT d p i)).addr = some (v.get i)) → Safe E R W Q init (k v)) :
    Safe E R W Q init
      (tl.load (d.rstd.ptr 0 + o_t) (some (o_t.ltS T64))
          (some (bc (FloatV.litF32 (FloatV.ofPat 0)))) >>= k) :=
  safe_load_buf d.rstd (addrs_rstd (T64 := T64) hot hT64) (fun i hi => gT_lt_rLen hd hi) hv
    (by rw [img_T]; exact hR) fun v hv1 _ => hk v hv1

/-- **The store of a `bfloat16` `dx`**, after the rounding (source line 67). -/
theorem safe_store16 (hd : d.ok) (hp : p * d.BT + d.BT ≤ 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int))
    {B : Sum (Buf .bf16) (Buf .f32)} {b : Buf .bf16} (hB : B = .inl b)
    (hlen : d.T * d.D ≤ lenOf B) (hv : validOf B init) (hW : elemsOf B (tileIdx d p) ⊆ W)
    {Q : Unit → Mem → Prop} {x : Tensor [d.BT, d.BD] F32}
    (hQ : ∀ m', (∀ i, enx d p i = true → Stored B m' (gx d p i) (x.get i)) → Q () m') :
    Safe E R W Q init
      (tl.cvt x >>= fun c : Tensor [d.BT, d.BD] BF16 =>
        tl.store (pTile (b.ptr 0) d o_t) c (some (mX d o_t T64))) := by
  subst hB
  refine safe_cvt (g := .bfloat16) (rnd := .rn)
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩ fun c hc => ?_
  refine safe_last ?_
  refine safe_store_buf b (addrs_of b hd hp hot hT64) (fun i hi => gx_lt_len hd hlen hi)
    (fun i j hi hj h => gx_inj hd hi hj h) hv (by rw [img_x hd]; exact hW)
    fun m' hm' _ _ => safe_ret (hQ m' fun i hi => ⟨c.get i, hm' i hi, hc i⟩)

/-- **The store of a `float32` `dx`** (source line 67; the conversion is `b_dx` itself). -/
theorem safe_store32 (hd : d.ok) (hp : p * d.BT + d.BT ≤ 2 ^ 51)
    (hot : ∀ i : Idx [d.BT], IntV.toInt (o_t.get i) = ((p * d.BT + i.1.val : Nat) : Int))
    (hT64 : IntV.toInt T64 = (d.T : Int))
    {B : Sum (Buf .bf16) (Buf .f32)} {b : Buf .f32} (hB : B = .inr b)
    (hlen : d.T * d.D ≤ lenOf B) (hv : validOf B init) (hW : elemsOf B (tileIdx d p) ⊆ W)
    {Q : Unit → Mem → Prop} {x : Tensor [d.BT, d.BD] F32}
    (hQ : ∀ m', (∀ i, enx d p i = true → Stored B m' (gx d p i) (x.get i)) → Q () m') :
    Safe E R W Q init (tl.store (pTile (b.ptr 0) d o_t) x (some (mX d o_t T64))) := by
  subst hB
  refine safe_last ?_
  exact safe_store_buf b (addrs_of b hd hp hot hT64) (fun i hi => gx_lt_len hd hlen hi)
    (fun i j hi hj h => gx_inj hd hi hj h) hv (by rw [img_x hd]; exact hW)
    fun m' hm' _ _ => safe_ret (hQ m' fun i hi => hm' i hi)

end Access

/-- The inputs `Pre` gives on the enabled lanes of a tile: a finite value. -/
theorem fin_y {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat} (i : Idx [d.BT, d.BD])
    (hi : enx d p i = true) :
    (∃ v, ValAt d.y init (gx d p i) v) ∧ (∃ v, ValAt d.dy init (gx d p i) v) := by
  have h := (hpre.2.2.2.2.1 (p * d.BT + i.1.val) (enx_row hi)).2 i.2.1.val (enx_col hi)
  simpa only [xIdx, gx] using h

/-- **One program is `Safe`**: the index arithmetic, then the walk for each dtype of `y`, `dy`,
`dx`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (P : Platform)
    (hsd : Device.SumDepth P) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hlt : i.val < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hp : i.val * d.BT + d.BT ≤ 2 ^ 51 := p_bound hd hlt
  have hi_lt : i.val < 2 ^ 31 := by
    have hg : grid d < 2 ^ 31 := hd.grid_lt
    omega
  have hBT31 : d.BT < 2 ^ 31 := by have := bt_le hd; omega
  have hRy : elemsOf d.y (tileIdx d i.val) ⊆ R d init i := fun x hx => Or.inl (Or.inl hx)
  have hRg : elemsOf d.dy (tileIdx d i.val) ⊆ R d init i := fun x hx => Or.inl (Or.inr hx)
  have hRr : d.rstd.elems (tileRow d i.val) ⊆ R d init i := fun x hx => Or.inr hx
  have hWx : elemsOf d.dx (tileIdx d i.val) ⊆ W d init i := fun x hx => hx
  have hfy : ∀ j, enx d i.val j = true → ∃ v, ValAt d.y init (gx d i.val j) v :=
    fun j hj => (fin_y hd hpre j hj).1
  have hfg : ∀ j, enx d i.val j = true → ∃ v, ValAt d.dy init (gx d i.val j) v :=
    fun j hj => (fin_y hd hpre j hj).2
  unfold model l2norm_bwd_kernel
  dsimp only
  refine safe_convert (d := .i32) (d' := .i64)
    (x := Tensor.scalar (tl.program_id ((launch d).ctx i) 0)) (hex := fun _ => ⟨_, rfl⟩) ?_
  intro r hr
  refine safe_convert (d := .i32) (d' := .i64) (x := tl.arange 0 d.BT)
    (hex := fun _ => ⟨_, rfl⟩) ?_
  intro o_a hoa
  have hi32 : IntV.toInt (tl.program_id ((launch d).ctx i) 0) = (i.val : Int) := by
    simp only [tl.program_id, launch_pid]
    exact IntV.I32.toInt_ofNat hi_lt
  have hi_t : IntV.toInt r.item = (i.val : Int) := by
    have h1 : r.item = IntV.ofInt (IntV.toInt (tl.program_id ((launch d).ctx i) 0)) := hr ()
    rw [h1, hi32]
    exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
  have hoa' : ∀ j : Idx [d.BT], IntV.toInt (o_a.get j) = (j.1.val : Int) := by
    intro j
    have h1 : o_a.get j = IntV.ofInt (IntV.toInt ((tl.arange 0 d.BT).get j)) := hoa j
    have h2 : IntV.toInt ((tl.arange 0 d.BT).get j) = (j.1.val : Int) :=
      arange_toInt d.BT hBT31 j.1
    rw [h1, h2]
    exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
  have hot := get_o_t (p := i.val) hi_t hoa' hp
  -- the runtime `T`, as an `int64`
  by_cases hT : d.T < 2 ^ 31
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hT)]
    dsimp only
    refine safe_convert (d := .i32) (d' := .i64) (x := Tensor.scalar (IntV.ofNat d.T))
      (hex := fun _ => ⟨_, rfl⟩) ?_
    intro r1 hr1
    have hT64 : IntV.toInt r1.item = (d.T : Int) := by
      have h1 : r1.item = IntV.ofInt (IntV.toInt ((Tensor.scalar (IntV.ofNat d.T)).get ())) :=
        hr1 ()
      have h2 : IntV.toInt ((Tensor.scalar (IntV.ofNat d.T : I32)).get ()) = (d.T : Int) :=
        IntV.I32.toInt_ofNat hT
      rw [h1, h2]
      exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by have := t_lt hd; omega⟩
    simp only [Prog.pure_eq, Prog.bind_eq, Prog.ret_bind]
    simp only [← Prog.bind_eq]
    all_goals
      cases hy : d.y <;> cases hg : d.dy <;> cases hx : d.dx <;> simp only [ptrOf]
    all_goals
      first
        | refine safe_loadT16 (B := d.y) hd hp hot hT64 hy hd.y_len hpre.1 hRy hfy ?_
        | refine safe_loadT32 (B := d.y) hd hp hot hT64 hy hd.y_len hpre.1 hRy hfy ?_
      intro b_y hyl
      refine safe_loadR hd hot hT64 hpre.2.2.2.1 hRr ?_
      intro b_rstd hrl
      first
        | refine safe_loadT16 (B := d.dy) hd hp hot hT64 hg hd.dy_len hpre.2.1 hRg hfg ?_
        | refine safe_loadT32 (B := d.dy) hd hp hot hT64 hg hd.dy_len hpre.2.1 hRg hfg ?_
      intro b_dy hgl
      refine safe_mul fun t1 h1 => ?_
      refine safe_mul fun pr hpr => ?_
      refine safe_sum1c (bd_pos hd) fun s hs => ?_
      refine safe_mul fun t2 h2 => ?_
      refine safe_mul fun t3 h3 => ?_
      refine safe_subc fun b_dx hsub => ?_
      first
        | refine safe_store16 (B := d.dx) hd hp hot hT64 hx hd.dx_len hpre.2.2.1 hWx ?_
        | refine safe_store32 (B := d.dx) hd hp hot hT64 hx hd.dx_len hpre.2.2.1 hWx ?_
      intro m hst
      exact leaf hd hpre hsd b_y b_dy t1 pr t2 t3 b_dx b_rstd s hyl hgl hrl h1 hpr hs h2 h3 hsub hst
  · rw [ite_eq_right_of_eq_false _ _ (eq_false hT)]
    dsimp only
    have hT64 : IntV.toInt (IntV.ofNat d.T : I64) = (d.T : Int) :=
      IntV.toInt_ofNat (by have := t_lt hd; omega)
    simp only [Prog.pure_eq, Prog.bind_eq, Prog.ret_bind]
    simp only [← Prog.bind_eq]
    all_goals
      cases hy : d.y <;> cases hg : d.dy <;> cases hx : d.dx <;> simp only [ptrOf]
    all_goals
      first
        | refine safe_loadT16 (B := d.y) hd hp hot hT64 hy hd.y_len hpre.1 hRy hfy ?_
        | refine safe_loadT32 (B := d.y) hd hp hot hT64 hy hd.y_len hpre.1 hRy hfy ?_
      intro b_y hyl
      refine safe_loadR hd hot hT64 hpre.2.2.2.1 hRr ?_
      intro b_rstd hrl
      first
        | refine safe_loadT16 (B := d.dy) hd hp hot hT64 hg hd.dy_len hpre.2.1 hRg hfg ?_
        | refine safe_loadT32 (B := d.dy) hd hp hot hT64 hg hd.dy_len hpre.2.1 hRg hfg ?_
      intro b_dy hgl
      refine safe_mul fun t1 h1 => ?_
      refine safe_mul fun pr hpr => ?_
      refine safe_sum1c (bd_pos hd) fun s hs => ?_
      refine safe_mul fun t2 h2 => ?_
      refine safe_mul fun t3 h3 => ?_
      refine safe_subc fun b_dx hsub => ?_
      first
        | refine safe_store16 (B := d.dx) hd hp hot hT64 hx hd.dx_len hpre.2.2.1 hWx ?_
        | refine safe_store32 (B := d.dx) hd hp hot hT64 hx hd.dx_len hpre.2.2.1 hWx ?_
      intro m hst
      exact leaf hd hpre hsd b_y b_dy t1 pr t2 t3 b_dx b_rstd s hyl hgl hrl h1 hpr hs h2 h3 hsub hst

end Claims.L2NormBwd
