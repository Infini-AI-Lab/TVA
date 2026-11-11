import Proof.Program

/-!
# The walk of one program

`rset`/`Rin` and `wtile`/`Wout` are the footprints of a program, `Qout` what it leaves. The
loop invariant (`loopI`) says each enabled lane holds the running sum; the loop, the final
product, the conversion and the store are walked one rule at a time, ending in `safe_prog`.
-/

namespace Claims.MoeSumReduce

open TVA

set_option maxHeartbeats 1000000

/-- The input elements a program reads, over all its iterations. -/
def rset (d : Inst) (p0 p1 : Nat) : Set Nat :=
  {j | ∃ t, t < d.K ∧ ∃ i : Lane d, enLane d p0 p1 i = true ∧ inIdx d p0 p1 t i = j}

/-- The bytes of the input a program reads. -/
def Rin (d : Inst) (p0 p1 : Nat) : Set Int := d.input.elems (rset d p0 p1)

/-- The bytes of the output a program writes. -/
def Wout (d : Inst) (p0 p1 : Nat) : Set Int := d.output.elems (wtile d p0 p1)

/-- The elements one iteration reads lie in `rset`. -/
theorem img_rset (d : Inst) {p0 p1 : Nat} {k : Nat} (hk : k < d.K) :
    Buf.img (inIdx d p0 p1 k) (enLane d p0 p1) ⊆ rset d p0 p1 := by
  intro e he
  unfold Buf.img at he
  obtain ⟨i, hi, hie⟩ := he
  exact ⟨k, hk, i, hi, hie⟩

/-- The value of the input one lane reads at iteration `k`, widened to `f32`. -/
def laneX (d : Inst) (p0 p1 : Nat) (i : Lane d) (k : Nat) : F32 :=
  FloatV.cvt (xv d (laneT d p0 i) k (laneJ d p1 i))

/-- What the store of a block must achieve: each element of its output tile holds a value within
the claim's bound of `sigma * refSum`. -/
def Qout (d : Inst) (p0 p1 : Nat) (m : Mem) : Prop :=
  ∀ t j, t < d.T → j < d.D → t / d.BLOCK_M = p0 → j / d.BLOCK_DIM = p1 →
    ∃ c : BF16, m.read .bf16 (d.output.ptr ((t * d.D + j : Nat) : Int)).addr = some c ∧
      (InRange d t j → c.isFinite = true ∧
        |c.toRat - (sigma d).toRat * refSum d t j| ≤
          tol d.K (refSum d t j) (absSum d t j) (sigma d).toRat)

/-- The loop invariant: the accumulator of each enabled lane is `foldF` at the current trip. -/
def loopI (d : Inst) (p0 p1 : Nat) (m : Mem) (t : Nat)
    (acc : Tensor [d.BLOCK_M, d.BLOCK_DIM] F32) (m' : Mem) : Prop :=
  m' = m ∧ ∀ i : Lane d, enLane d p0 p1 i = true →
    FloatV.NaNEq (foldF (laneX d p0 p1 i) t) (acc.get i)

/-- One trip of the accumulation loop is safe: it loads the plane, widens it and adds it. -/
theorem safe_step (d : Inst) (hd : d.ok) {E : Env} {R W : Set Int} {m : Mem}
    (hpre : Pre d m) {p0 p1 : Nat} {k : Nat} (hk : k < d.K)
    {base_ptrs : Tensor [d.BLOCK_M, d.BLOCK_DIM] (Ptr .bf16)} {s1 : I64}
    (hptrs : ∀ (j : I64), j.toInt = (k : Int) → ∀ i : Lane d,
      ((base_ptrs + (bc (j * s1) : Tensor [d.BLOCK_M, d.BLOCK_DIM] I64)).get i).addr
        = (d.input.ptr ((inIdx d p0 p1 k i : Nat) : Int)).addr)
    {mt : Tensor [d.BLOCK_M] Bool} {md : Tensor [d.BLOCK_DIM] Bool}
    (hmask : ∀ i : Lane d,
      ((mt.col d.BLOCK_DIM).and (md.row d.BLOCK_M)).get i = enLane d p0 p1 i)
    (hR : d.input.elems (rset d p0 p1) ⊆ R)
    {acc : Tensor [d.BLOCK_M, d.BLOCK_DIM] F32}
    (hacc : ∀ i : Lane d, enLane d p0 p1 i = true →
      FloatV.NaNEq (foldF (laneX d p0 p1 i) k) (acc.get i)) :
    Safe E R W (fun r m' => m' = m ∧ ∀ i : Lane d, enLane d p0 p1 i = true →
      FloatV.NaNEq (foldF (laneX d p0 p1 i) (k + 1)) (r.get i)) m
      (stepBody d base_ptrs s1 mt md (IntV.ofInt (k : Int)) acc) := by
  have hk31 : (k : Int) < (2 : Int) ^ 31 := by
    have h1 : (k : Int) < (d.K : Int) := by exact_mod_cast hk
    have h2 : (d.K : Int) < (2 : Int) ^ 31 := by exact_mod_cast hd.K_lt
    linarith
  have hKi : ((IntV.ofInt (k : Int) : I32).toInt) = (k : Int) := by
    refine IntV.I32.toInt_ofInt ⟨?_, hk31⟩
    have h0 : (0 : Int) ≤ (k : Int) := by exact_mod_cast Nat.zero_le k
    linarith
  unfold stepBody
  refine safe_convert (hex := ?_) (hk := ?_)
  · intro i
    exact ⟨IntV.ofInt ((IntV.ofInt (k : Int) : I32).toInt), rfl⟩
  · intro r hr
    have hi64 : r.item.toInt = (k : Int) := by
      have h0 : r.get () = (IntV.ofInt ((IntV.ofInt (k : Int) : I32).toInt) : I64) := hr ()
      have h1 : r.item = (IntV.ofInt (k : Int) : I64) := by
        rw [hKi] at h0
        exact h0
      rw [h1]
      refine IntV.toInt_ofInt (w := 64) (by norm_num) ?_
      have h3 : (2 : Int) ^ (64 - 1) = (2 : Int) ^ 63 := by norm_num
      unfold IntV.SFits
      rw [h3]
      constructor
      · have h0 : (0 : Int) ≤ (k : Int) := by exact_mod_cast Nat.zero_le k
        linarith
      · have h2 : (2 : Int) ^ 31 ≤ (2 : Int) ^ 63 := by norm_num
        linarith
    refine safe_convert (hex := ?_) (hk := ?_)
    · intro i
      exact ⟨FloatV.cvt ((tl.full [d.BLOCK_M, d.BLOCK_DIM] (FloatV.litF32 (FloatV.ofPat 0))).get i),
        Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), FloatV.NaNEq.refl _⟩⟩
    · intro pad hpad
      refine safe_load_buf d.input (g := inIdx d p0 p1 k) (en := enLane d p0 p1) ?_ ?_ hpre.1 ?_ ?_
      · refine Buf.addrs_of_get d.input hmask ?_
        intro i he
        calc ((base_ptrs + (bc (r.item * s1) : Tensor [d.BLOCK_M, d.BLOCK_DIM] I64)).get i)
            = ⟨((base_ptrs + (bc (r.item * s1) : Tensor [d.BLOCK_M, d.BLOCK_DIM] I64)).get i).addr⟩ :=
              rfl
          _ = ⟨(d.input.ptr ((inIdx d p0 p1 k i : Nat) : Int)).addr⟩ := by
              rw [hptrs r.item hi64 i]
          _ = d.input.ptr ((inIdx d p0 p1 k i : Nat) : Int) := rfl
      · intro i he
        exact inIdx_lt hd (k := k) i he hk
      · exact (Buf.elems_mono d.input (img_rset d hk)).trans hR
      · intro v hv1 hv2
        refine safe_cvt (f := Format.bfloat16) (g := Format.binary32)
          (hok := ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩) ?_
        intro t32 ht32
        refine safe_add (x := acc) (y := t32) ?_
        intro a ha
        refine safe_ret ⟨rfl, ?_⟩
        intro i he
        refine FloatV.NaNEq.trans ?_ (ha i)
        refine Or.inl ?_
        calc FloatV.add (foldF (laneX d p0 p1 i) k) (laneX d p0 p1 i k)
            = FloatV.add (acc.get i) (laneX d p0 p1 i k) := add_congr_left (hacc i he)
          _ = FloatV.add (acc.get i) (t32.get i) := by
              refine add_congr_right ?_
              have h1 := hv1 i he
              rw [hpre.2.2 (inIdx d p0 p1 k i) (inIdx_lt hd (k := k) i he hk)] at h1
              have hxv : v.get i = xv d (laneT d p0 i) k (laneJ d p1 i) :=
                (Option.some.inj h1).symm
              have h := ht32 i
              rw [hxv] at h
              exact h

/-- The accumulation loop is safe and leaves each enabled lane holding the running sum. -/
theorem safe_loop (d : Inst) (hd : d.ok) {E : Env} {R : Set Int} {m : Mem}
    (hpre : Pre d m) {p0 p1 : Nat}
    {base_ptrs : Tensor [d.BLOCK_M, d.BLOCK_DIM] (Ptr .bf16)} {s1 : I64}
    {mt : Tensor [d.BLOCK_M] Bool} {md : Tensor [d.BLOCK_DIM] Bool}
    (hptrs : ∀ (k : Nat), k < d.K → ∀ (j : I64), j.toInt = (k : Int) → ∀ i : Lane d,
      ((base_ptrs + (bc (j * s1) : Tensor [d.BLOCK_M, d.BLOCK_DIM] I64)).get i).addr
        = (d.input.ptr ((inIdx d p0 p1 k i : Nat) : Int)).addr)
    (hmask : ∀ i : Lane d,
      ((mt.col d.BLOCK_DIM).and (md.row d.BLOCK_M)).get i = enLane d p0 p1 i)
    (hR : d.input.elems (rset d p0 p1) ⊆ R)
    (acc : Tensor [d.BLOCK_M, d.BLOCK_DIM] F32)
    (hacc0 : ∀ i : Lane d, enLane d p0 p1 i = true →
      FloatV.NaNEq (foldF (laneX d p0 p1 i) 0) (acc.get i)) :
    Safe E R ∅ (fun acc' m' => m' = m ∧ ∀ i : Lane d, enLane d p0 p1 i = true →
      FloatV.NaNEq (foldF (laneX d p0 p1 i) d.K) (acc'.get i)) m
      (tl.forN d.K (fun t => stepBody d base_ptrs s1 mt md (IntV.ofInt (t : Int))) acc) := by
  have hWt : ∀ t u, t < u → u < d.K → Disjoint (∅ : Set Int) ∅ := by
    intro t u _ _
    simp
  have h0 : loopI d p0 p1 m 0 acc m := ⟨rfl, hacc0⟩
  have hstep : ∀ t, t < d.K → ∀ st m₁, loopI d p0 p1 m t st m₁ →
      Safe E R ∅ (fun st' m₂ => loopI d p0 p1 m (t + 1) st' m₂) m₁
        (stepBody d base_ptrs s1 mt md (IntV.ofInt (t : Int)) st) := by
    intro t ht st m₁ hI
    obtain ⟨hm1, hst⟩ := hI
    subst m₁
    exact safe_step d hd hpre ht (hptrs t ht) hmask hR hst
  have h := safe_forN d.K (fun t => stepBody d base_ptrs s1 mt md (IntV.ofInt (t : Int)))
    acc (loopI d p0 p1 m) (fun _ => ∅) hWt h0 hstep
  simpa only [iterW_empty, loopI] using h

/-- An enabled lane's program index on the token axis is inside the grid. -/
theorem out_block_lt {d : Inst} (hd : d.ok) {p0 p1 : Nat} {i : Lane d}
    (hi : enLane d p0 p1 i = true) : p0 < gT d := by
  have h1 : laneT d p0 i < d.T := laneT_lt_T hi
  have h2 : d.T ≤ gT d * d.BLOCK_M := gT_mul_lo hd
  have hBM : 0 < d.BLOCK_M := block_m_pos hd
  have h3 : p0 * d.BLOCK_M < gT d * d.BLOCK_M := by
    have h4 : p0 * d.BLOCK_M ≤ laneT d p0 i := by unfold laneT; omega
    omega
  exact (Nat.mul_lt_mul_right hBM).mp h3

/-- An enabled lane's program index on the hidden axis is inside the grid. -/
theorem out_col_lt {d : Inst} (hd : d.ok) {p0 p1 : Nat} {i : Lane d}
    (hi : enLane d p0 p1 i = true) : p1 < gD d := by
  have h1 : laneJ d p1 i < d.D := laneJ_lt_D hi
  have h2 : d.D ≤ gD d * d.BLOCK_DIM := gD_mul_lo hd
  have hBD : 0 < d.BLOCK_DIM := block_d_pos hd
  have h3 : p1 * d.BLOCK_DIM < gD d * d.BLOCK_DIM := by
    have h4 : p1 * d.BLOCK_DIM ≤ laneJ d p1 i := by unfold laneJ; omega
    omega
  exact (Nat.mul_lt_mul_right hBD).mp h3

/-- Equal addresses of `bf16` pointers make equal pointers. -/
theorem out_ptr_eq {d : Inst} {p0 p1 : Nat} {i : Lane d}
    {ptrs : Tensor [d.BLOCK_M, d.BLOCK_DIM] (Ptr .bf16)}
    (h : (ptrs.get i).addr = (d.output.ptr ((outIdx d p0 p1 i : Nat) : Int)).addr) :
    ptrs.get i = d.output.ptr (outIdx d p0 p1 i : Int) := by
  exact (Ptr.mk.injEq (ptrs.get i).addr (d.output.ptr (outIdx d p0 p1 i : Int)).addr).mpr h

/-- The multiply, conversion and store are safe and leave the claim's bound in the output tile. -/
theorem safe_out (d : Inst) (hd : d.ok) {E : Env} {R W : Set Int} {m : Mem}
    (hpre : Pre d m) {p0 p1 : Nat}
    {ptrs : Tensor [d.BLOCK_M, d.BLOCK_DIM] (Ptr .bf16)}
    {mt : Tensor [d.BLOCK_M] Bool} {md : Tensor [d.BLOCK_DIM] Bool}
    (hptrs : ∀ i : Lane d,
      (ptrs.get i).addr = (d.output.ptr ((outIdx d p0 p1 i : Nat) : Int)).addr)
    (hmask : ∀ i : Lane d,
      ((mt.col d.BLOCK_DIM).and (md.row d.BLOCK_M)).get i = enLane d p0 p1 i)
    (hW : Wout d p0 p1 ⊆ W)
    {acc : Tensor [d.BLOCK_M, d.BLOCK_DIM] F32}
    (hacc : ∀ i : Lane d, enLane d p0 p1 i = true →
      FloatV.NaNEq (foldF (laneX d p0 p1 i) d.K) (acc.get i)) :
    Safe E R W (fun _ => Qout d p0 p1) m
      (tl.mul acc (bc (sigma32 d)) >>= fun acc2 =>
        tl.cvt (g := .bfloat16) acc2 >>= fun out =>
          tl.store ptrs out (some ((mt.col d.BLOCK_DIM).and (md.row d.BLOCK_M)))) := by
  have hvout : d.output.Valid m := hpre.2.1
  refine safe_mul (x := acc) (y := bc (sigma32 d)) (f := .binary32) ?_
  intro acc2 hacc2
  refine safe_cvt (f := .binary32) (g := .bfloat16) (rnd := .rn) ?_ ?_
  · exact ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩
  intro out hout
  refine safe_last ?_
  refine safe_store_buf d.output (g := outIdx d p0 p1) (en := enLane d p0 p1)
    ?_ ?_ ?_ hvout ?_ (k := pure) ?_
  · exact Buf.addrs_of_get d.output hmask
      (fun i hi => out_ptr_eq (hptrs i))
  · exact fun i hi => outIdx_lt hd i hi
  · intro i j hi hj heq
    exact outIdx_inj hd (out_block_lt hd hi) (out_col_lt hd hi) hi hj heq
  · simpa only [Wout, wtile] using hW
  · intro m' hread hframe hpres
    refine Safe.ret (a := ()) ?_
    intro t j ht hj htdiv hjdiv
    let i0 : Lane d := ownerLane d t j (block_m_pos hd) (block_d_pos hd)
    have hT0 : laneT d p0 i0 = t := by
      dsimp only [i0]
      rw [← htdiv]
      exact owner_T (d := d) hd (t := t) (j := j)
    have hJ0 : laneJ d p1 i0 = j := by
      dsimp only [i0]
      rw [← hjdiv]
      exact owner_J (d := d) hd (t := t) (j := j)
    have hen : enLane d p0 p1 i0 = true := by
      dsimp only [i0]
      rw [← htdiv, ← hjdiv]
      exact owner_en (d := d) hd ht hj
    have hout0 : outIdx d p0 p1 i0 = t * d.D + j := by
      dsimp only [i0]
      rw [← htdiv, ← hjdiv]
      exact owner_out (d := d) hd ht hj
    refine ⟨out.get i0, ?_, ?_⟩
    · have h := hread i0 hen
      rwa [hout0] at h
    · intro hIR
      have hφ : ∀ k, k < d.K → (xv d t k j).isFinite = true := hIR.1
      have hσ64 : (sigma d).isFinite = true := hIR.2.1
      have hσ32 : (sigma32 d).isFinite = true := hIR.2.2.1
      have hA0 : (0 : ℚ) ≤ absSum d t j := by
        unfold absSum
        exact Finset.sum_nonneg (fun k _ => abs_nonneg _)
      have hS : |refSum d t j| ≤ absSum d t j := by
        unfold refSum
        exact absSum_partial d t j d.K le_rfl
      have hσrel := sigma32_rel d hσ64 hσ32
      have hxf : ∀ k, k < d.K → (laneX d p0 p1 i0 k).isFinite = true := by
        intro k hk
        simpa only [laneX, hT0, hJ0] using (cvt_widen (hφ k hk)).1
      have hsum : ∀ t', t' ≤ d.K →
          (∑ k ∈ Finset.range t', (laneX d p0 p1 i0 k).toRat)
            = ∑ k ∈ Finset.range t', (xv d t k j).toRat := by
        intro t' ht'
        refine Finset.sum_congr rfl ?_
        intro k hk
        have hk' : k < d.K := lt_of_lt_of_le (Finset.mem_range.mp hk) ht'
        have h := (cvt_widen (d := d) (t := t) (k := k) (j := j) (hφ k hk')).2
        show (laneX d p0 p1 i0 k).toRat = (xv d t k j).toRat
        rw [← h]
        simp only [laneX, hT0, hJ0]
      have hSe : refSum d t j = ∑ k ∈ Finset.range d.K, (laneX d p0 p1 i0 k).toRat := by
        unfold refSum
        exact (hsum d.K le_rfl).symm
      have hxS : ∀ t', t' ≤ d.K →
          |∑ k ∈ Finset.range t', (laneX d p0 p1 i0 k).toRat| ≤ absSum d t j := by
        intro t' ht'
        rw [hsum t' ht']
        exact absSum_partial d t j t' ht'
      have hvb := value_bound d.K (refSum d t j) (absSum d t j) ((sigma d).toRat)
        ((sigma32 d).toRat) hd.K_pos hd.K_le hA0 hS hσrel (laneX d p0 p1 i0) hxf hSe hxS
        (room d hd hIR)
      have hmul : FloatV.mul (foldF (laneX d p0 p1 i0) d.K) (sigma32 d)
          = FloatV.mul (acc.get i0) (sigma32 d) := by
        simpa only [Tensor.get_full] using
          mul_congr_left (f := .binary32) (hacc i0 hen)
      have h2 : FloatV.NaNEq (FloatV.mul (foldF (laneX d p0 p1 i0) d.K) (sigma32 d))
          (acc2.get i0) := by
        rw [hmul]
        simpa only [Tensor.get_full] using hacc2 i0
      have h3 : FloatV.NaNEq
          (FloatV.cvt (FloatV.mul (foldF (laneX d p0 p1 i0) d.K) (sigma32 d)))
          (out.get i0) := by
        rw [cvt_congr (f := .binary32) (g := .bfloat16) h2]
        exact hout i0
      exact (hvb.2 (sigma32 d) rfl hσ32).2 (out.get i0) h3

set_option maxHeartbeats 1000000 in
/-- **One program** is safe: it reads its input tile, writes its output tile, and ends with `Qout`. -/
theorem safe_prog (d : Inst) (hd : d.ok) {E : Env} {m : Mem} (hpre : Pre d m) (c : Ctx)
    (hp0 : c.pid 0 < gT d) (hp1 : c.pid 1 < gD d) :
    Safe E (Rin d (c.pid 0) (c.pid 1)) (Wout d (c.pid 0) (c.pid 1))
      (fun _ => Qout d (c.pid 0) (c.pid 1)) m (progBody d c) := by
  unfold progBody
  -- `s0`: the row stride `K * D`, as an `int64`
  refine safe_convert ?_ ?_
  · intro i
    exact ⟨_, rfl⟩
  intro t0 ht0
  dsimp only
  have hs0 : (Tensor.item t0).toInt = ((d.K * d.D : Nat) : Int) := by
    have h1 : t0.get () = IntV.ofInt ((IntV.ofNat (d.K * d.D) : I32).toInt) := ht0 ()
    have h2 : (IntV.ofNat (d.K * d.D) : I32).toInt = ((d.K * d.D : Nat) : Int) :=
      IntV.I32.toInt_ofNat hd.stride_lt
    have h3 : IntV.SFits 64 ((d.K * d.D : Nat) : Int) := by
      refine sfits64_of_lt (by positivity) ?_
      have hlt : d.K * d.D < 2 ^ 63 := by
        have h1 := hd.stride_lt
        have h2 : (2 : Nat) ^ 31 ≤ 2 ^ 63 := by norm_num
        omega
      exact_mod_cast hlt
    show (t0.get ()).toInt = ((d.K * d.D : Nat) : Int)
    rw [h1, h2]
    exact IntV.toInt_ofInt (by norm_num) h3
  -- `s1`: the column stride `D`, as an `int64`
  refine safe_convert ?_ ?_
  · intro i
    exact ⟨_, rfl⟩
  intro t1 ht1
  have hs1 : (Tensor.item t1).toInt = ((d.D : Nat) : Int) := by
    have h1 : t1.get () = IntV.ofInt ((IntV.ofNat d.D : I32).toInt) := ht1 ()
    have h2 : (IntV.ofNat d.D : I32).toInt = ((d.D : Nat) : Int) :=
      IntV.I32.toInt_ofNat hd.D_lt
    have h3 : IntV.SFits 64 ((d.D : Nat) : Int) := by
      refine sfits64_of_lt (by positivity) ?_
      have hlt : d.D < 2 ^ 63 := by
        have h1 := hd.D_lt
        have h2 : (2 : Nat) ^ 31 ≤ 2 ^ 63 := by norm_num
        omega
      exact_mod_cast hlt
    show (t1.get ()).toInt = ((d.D : Nat) : Int)
    rw [h1, h2]
    exact IntV.toInt_ofInt (by norm_num) h3
  -- `os0`: the output column stride `D`, as an `int64`
  refine safe_convert ?_ ?_
  · intro i
    exact ⟨_, rfl⟩
  intro t2 ht2
  have hos0 : (Tensor.item t2).toInt = ((d.D : Nat) : Int) := by
    have h1 : t2.get () = IntV.ofInt ((IntV.ofNat d.D : I32).toInt) := ht2 ()
    have h2 : (IntV.ofNat d.D : I32).toInt = ((d.D : Nat) : Int) :=
      IntV.I32.toInt_ofNat hd.D_lt
    have h3 : IntV.SFits 64 ((d.D : Nat) : Int) := by
      refine sfits64_of_lt (by positivity) ?_
      have hlt : d.D < 2 ^ 63 := by
        have h1 := hd.D_lt
        have h2 : (2 : Nat) ^ 31 ≤ 2 ^ 63 := by norm_num
        omega
      exact_mod_cast hlt
    show (t2.get ()).toInt = ((d.D : Nat) : Int)
    rw [h1, h2]
    exact IntV.toInt_ofInt (by norm_num) h3
  -- `offs0`: the token offsets, as `int64`
  refine safe_convert ?_ ?_
  · intro i
    exact ⟨_, rfl⟩
  intro o0 ho0
  have ho0' : ∀ a : Fin d.BLOCK_M,
      (o0.get1 a).toInt = ((c.pid 0 * d.BLOCK_M + a.val : Nat) : Int) := by
    intro a
    have h1 : o0.get (a, ()) = IntV.ofInt
        (((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32)
          + tl.arange 0 d.BLOCK_M).get (a, ())).toInt) := ho0 (a, ())
    have hT := offsT_toInt d hd c hp0 (a, ⟨0, block_d_pos hd⟩, ())
    have hT' : ((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32)
          + tl.arange 0 d.BLOCK_M).get (a, ())).toInt
        = ((c.pid 0 * d.BLOCK_M + a.val : Nat) : Int) := hT
    have h3 : IntV.SFits 64 ((c.pid 0 * d.BLOCK_M + a.val : Nat) : Int) := by
      refine sfits64_of_lt (by positivity) ?_
      have hlt : c.pid 0 * d.BLOCK_M + a.val < 2 ^ 23 :=
        laneT_lt hd hp0 (a, ⟨0, block_d_pos hd⟩, ())
      have hlt' : c.pid 0 * d.BLOCK_M + a.val < 2 ^ 63 := by
        have h2 : (2 : Nat) ^ 23 ≤ 2 ^ 63 := by norm_num
        omega
      exact_mod_cast hlt'
    show (o0.get (a, ())).toInt = ((c.pid 0 * d.BLOCK_M + a.val : Nat) : Int)
    rw [h1, hT']
    exact IntV.toInt_ofInt (by norm_num) h3
  -- `offs1`: the column offsets, as `int64`
  refine safe_convert ?_ ?_
  · intro i
    exact ⟨_, rfl⟩
  intro o1 ho1
  have ho1' : ∀ b : Fin d.BLOCK_DIM,
      (o1.get1 b).toInt = ((c.pid 1 * d.BLOCK_DIM + b.val : Nat) : Int) := by
    intro b
    have h1 : o1.get (b, ()) = IntV.ofInt
        (((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32)
          + tl.arange 0 d.BLOCK_DIM).get (b, ())).toInt) := ho1 (b, ())
    have hJ := offsJ_toInt d hd c hp1 (⟨0, block_m_pos hd⟩, b, ())
    have hJ' : ((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32)
          + tl.arange 0 d.BLOCK_DIM).get (b, ())).toInt
        = ((c.pid 1 * d.BLOCK_DIM + b.val : Nat) : Int) := hJ
    have h3 : IntV.SFits 64 ((c.pid 1 * d.BLOCK_DIM + b.val : Nat) : Int) := by
      refine sfits64_of_lt (by positivity) ?_
      have hlt : c.pid 1 * d.BLOCK_DIM + b.val < 2 ^ 23 :=
        laneJ_lt hd hp1 (⟨0, block_m_pos hd⟩, b, ())
      have hlt' : c.pid 1 * d.BLOCK_DIM + b.val < 2 ^ 63 := by
        have h2 : (2 : Nat) ^ 23 ≤ 2 ^ 63 := by norm_num
        omega
      exact_mod_cast hlt'
    show (o1.get (b, ())).toInt = ((c.pid 1 * d.BLOCK_DIM + b.val : Nat) : Int)
    rw [h1, hJ']
    exact IntV.toInt_ofInt (by norm_num) h3
  -- the accumulation loop reads the input tile
  have hzeros : ∀ i : Lane d, enLane d (c.pid 0) (c.pid 1) i = true →
      FloatV.NaNEq (foldF (laneX d (c.pid 0) (c.pid 1) i) 0)
        ((tl.zeros [d.BLOCK_M, d.BLOCK_DIM] .f32).get i) := by
    intro i _
    exact FloatV.NaNEq.refl _
  refine Safe.bind_ro ?_
  rw [forRange_K d hd]
  refine (safe_loop d hd hpre ?_ ?_ le_rfl (tl.zeros [d.BLOCK_M, d.BLOCK_DIM] .f32) hzeros).mono
    ?_ (Set.empty_subset _)
  · intro k hk j hj i
    exact congrArg Ptr.addr
      (load_ptrs_bounded d hd hp0 hp1 o0 (Tensor.item t0) o1 j (Tensor.item t1) k
        ho0' hs0 ho1' hj hs1 hk i)
  · intro i
    exact mask_eq d hd c hp0 hp1 i
  -- the epilogue stores the output tile
  · intro acc' m' hpost
    obtain ⟨hm', hacc⟩ := hpost
    subst hm'
    have hptrs : ∀ i : Lane d,
        ((d.output.ptr 0 + (o0.col d.BLOCK_DIM * (Tensor.item t2)
          + o1.row d.BLOCK_M)).get i).addr
          = (d.output.ptr ((outIdx d (c.pid 0) (c.pid 1) i : Nat) : Int)).addr := by
      intro i
      exact congrArg Ptr.addr
        (store_ptrs d hd hp0 hp1 o0 (Tensor.item t2) o1 ho0' ho1' hos0 i)
    have hmask : ∀ i : Lane d,
        ((((tl.program_id c 0 * (IntV.ofNat d.BLOCK_M : I32)
              + tl.arange 0 d.BLOCK_M).ltS (IntV.ofNat d.T : I32)).col d.BLOCK_DIM).and
            (((tl.program_id c 1 * (IntV.ofNat d.BLOCK_DIM : I32)
              + tl.arange 0 d.BLOCK_DIM).ltS (IntV.ofNat d.D : I32)).row d.BLOCK_M)).get i
          = enLane d (c.pid 0) (c.pid 1) i := by
      intro i
      exact mask_eq d hd c hp0 hp1 i
    exact safe_out d hd hpre hptrs hmask le_rfl hacc

end Claims.MoeSumReduce
