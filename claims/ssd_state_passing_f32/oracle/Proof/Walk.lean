import Claim
import TVA.Proof
import Proof.Program
import Proof.Loop

/-!
# One program: the loads and stores before the loop

Program `(pm, h)` of the grid first loads the initial states (zeros without), stores them into
slot `0` of `out`, and runs the loop. This file computes the lanes' pointers from the program ids
and walks those lines, so that `safe_program_I` / `safe_program_N` give the program's `Safe`.
-/
namespace Claims.SsdStatePassing

open TVA



/-- A program id below `2^31` is its own `int32` value. -/
theorem program_id_toInt {c : Ctx} {a : Fin 3} {n : Nat} (hc : c.pid a = n) (hn : n < 2 ^ 31) :
    (tl.program_id c a).toInt = (n : Int) := by
  unfold tl.program_id
  rw [hc]
  exact IntV.I32.toInt_ofNat hn

/-- The lanes of `offs_m` of program `pm`. -/
theorem offsM_lane {d : Inst} (hd : d.ok) {c : Ctx} {pm : Nat} (hc0 : c.pid 0 = pm)
    (hpm : pm * BL < d.dim) (k : Fin BL) :
    ((offsM c).get1 k).toInt = ((pm * BL + k : Nat) : Int) := by
  have hfit : Block.Fits pm BL := by
    have := hd.dim_lt
    rw [BL_eq] at *
    unfold Block.Fits
    omega
  have : tl.program_id c 0 = IntV.ofNat pm := by unfold tl.program_id; rw [hc0]
  unfold offsM
  rw [this]
  exact Block.lane_toInt hfit k

/-- The lanes of a pointer block `b.ptr 0 + X + offs_m * 1`. -/
theorem ptr_lane {dt : DType} (b : Buf dt) {X : I32} {x : Nat} (hX : X.toInt = x)
    {offs : Tensor [BL - 0] I32} {pm : Nat}
    (hoff : ∀ k : Fin BL, (offs.get1 k).toInt = ((pm * BL + k : Nat) : Int))
    (hb : pm * BL + BL < 2 ^ 31) (k : Fin BL) :
    ((b.ptr 0 + X) + offs * (IntV.ofNat 1 : I32)).get1 k = b.ptr ((x + (pm * BL + k) : Nat) : Int) := by
  have hk := k.isLt
  have h1 : (offs.get1 k * (IntV.ofNat 1 : I32)).toInt = (((pm * BL + k) * 1 : Nat) : Int) :=
    IntV.I32.mul_nat (hoff k) toInt_ofNat1 (by omega)
  show ((b.ptr 0 + X).offset (offs.get1 k * (IntV.ofNat 1 : I32)).toInt) = _
  rw [h1, Ptr.add_int, hX, Buf.ptr_offset, Buf.ptr_offset]
  congr 1
  push_cast
  ring

/-- A block of the grid starts below `dim`. -/
theorem block_lt {d : Inst} (hd : d.ok) {pm : Nat} (hpm : pm < gx d) : pm * BL < d.dim := by
  have h := chunk_pos hd
  unfold gx at hpm
  rw [BL_eq] at *
  have h1 := (Nat.le_div_iff_mul_le (by norm_num : 0 < 1024)).mp (show pm + 1 ≤ (d.dim + 1024 - 1) / 1024 by omega)
  omega

/-- The head offset `h · dim` is an `int32` value. -/
theorem h_dim_lt {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.nheads) : h * sSh d < 2 ^ 31 := by
  have h1 := hd.st_lt
  have h2 := nchunks_pos hd
  have h3 : (h + 1) * d.dim ≤ d.nheads * d.dim := Nat.mul_le_mul_right _ hh
  have h4 : d.nheads * d.dim ≤ d.nchunks * (d.nheads * d.dim) := Nat.le_mul_of_pos_left _ h2
  unfold sSh
  rw [Nat.succ_mul] at h3
  omega

/-- `pid_b · A + pid_h · stride_head` for `pid_b = 0`. -/
theorem xHead {d : Inst} (hd : d.ok) {c : Ctx} {h : Nat} (hc1 : c.pid 1 = 0) (hc2 : c.pid 2 = h)
    (hh : h < d.nheads) (A : Nat) (hA : A < 2 ^ 31) :
    (tl.program_id c 1 * (IntV.ofNat A : I32) + tl.program_id c 2 * (IntV.ofNat (sSh d) : I32)).toInt =
      ((h * sSh d : Nat) : Int) := by
  have hlt := h_dim_lt hd hh
  have hsh : sSh d < 2 ^ 31 := by
    have : sSh d ≤ h * sSh d + sSh d := by omega
    have h3 : (h + 1) * d.dim ≤ d.nheads * d.dim := Nat.mul_le_mul_right _ hh
    have h1 := hd.st_lt
    have h4 : d.nheads * d.dim ≤ d.nchunks * (d.nheads * d.dim) :=
      Nat.le_mul_of_pos_left _ (nchunks_pos hd)
    unfold sSh at *
    rw [Nat.succ_mul] at h3
    omega
  have e1 : (tl.program_id c 1).toInt = ((0 : Nat) : Int) := program_id_toInt hc1 (by norm_num)
  have e2 : (tl.program_id c 2).toInt = (h : Int) :=
    program_id_toInt hc2 (by have := hd.heads_lt; omega)
  have m1 : (tl.program_id c 1 * (IntV.ofNat A : I32)).toInt = ((0 * A : Nat) : Int) :=
    IntV.I32.mul_nat e1 (IntV.I32.toInt_ofNat hA) (by simp)
  have m2 : (tl.program_id c 2 * (IntV.ofNat (sSh d) : I32)).toInt = ((h * sSh d : Nat) : Int) :=
    IntV.I32.mul_nat e2 (IntV.I32.toInt_ofNat hsh) hlt
  rw [IntV.I32.add_nat m1 m2 (by simpa using hlt)]
  simp

/-- `pid_h · stride_head`. -/
theorem xHead1 {d : Inst} (hd : d.ok) {c : Ctx} {h : Nat} (hc2 : c.pid 2 = h)
    (hh : h < d.nheads) :
    (tl.program_id c 2 * (IntV.ofNat (sSh d) : I32)).toInt = ((h * sSh d : Nat) : Int) := by
  have hlt := h_dim_lt hd hh
  have hsh : sSh d < 2 ^ 31 := by
    have h3 : (h + 1) * d.dim ≤ d.nheads * d.dim := Nat.mul_le_mul_right _ hh
    have h1 := hd.st_lt
    have h4 : d.nheads * d.dim ≤ d.nchunks * (d.nheads * d.dim) :=
      Nat.le_mul_of_pos_left _ (nchunks_pos hd)
    unfold sSh at *
    rw [Nat.succ_mul] at h3
    omega
  have e2 : (tl.program_id c 2).toInt = (h : Int) :=
    program_id_toInt hc2 (by have := hd.heads_lt; omega)
  exact IntV.I32.mul_nat e2 (IntV.I32.toInt_ofNat hsh) hlt

/-- A block's lanes are `int32` values. -/
theorem block_fits {d : Inst} (hd : d.ok) {pm : Nat} (hpm : pm * BL < d.dim) :
    pm * BL + BL < 2 ^ 31 := by
  have := hd.dim_lt
  have h2 : BL = 1024 := BL_eq
  omega

/-- The batch stride is an `int32` value. -/
theorem sSb_lt {d : Inst} (hd : d.ok) : sSb d < 2 ^ 31 := hd.st_lt

/-- Lane `k` of the `initial_states` pointer block is element `(0, h, m)`. -/
theorem initP_lane {d : Inst} (hd : d.ok) {c : Ctx} {pm h : Nat} (hc0 : c.pid 0 = pm)
    (hc2 : c.pid 2 = h) (hh : h < d.nheads) (hpm : pm * BL < d.dim) (k : Fin BL) :
    (initP d c).get1 k = d.initst.ptr (inIdx d 0 h (pm * BL + k) : Int) := by
  have hb := block_fits hd hpm
  unfold initP
  rw [ptr_lane d.initst (xHead1 hd hc2 hh) (offsM_lane hd hc0 hpm) hb k]
  congr 1
  unfold inIdx sSh
  push_cast
  ring

/-- Lane `k` of the `states` pointer block is element `(0, 0, h, m)`. -/
theorem statesP_lane {d : Inst} (hd : d.ok) {c : Ctx} {pm h : Nat} (hc0 : c.pid 0 = pm)
    (hc1 : c.pid 1 = 0) (hc2 : c.pid 2 = h) (hh : h < d.nheads) (hpm : pm * BL < d.dim)
    (k : Fin BL) : (statesP d c).get1 k = d.states.ptr (stIdx d 0 h (pm * BL + k) : Int) := by
  have hb := block_fits hd hpm
  unfold statesP
  rw [ptr_lane d.states (xHead hd hc1 hc2 hh (sSb d) (sSb_lt hd)) (offsM_lane hd hc0 hpm) hb k]
  congr 1
  unfold stIdx sSh
  push_cast
  ring

/-- Lane `k` of the `out` pointer block is element `(0, 0, h, m)`. -/
theorem outP_lane {d : Inst} (hd : d.ok) {c : Ctx} {pm h : Nat} (hc0 : c.pid 0 = pm)
    (hc1 : c.pid 1 = 0) (hc2 : c.pid 2 = h) (hh : h < d.nheads) (hpm : pm * BL < d.dim)
    (k : Fin BL) : (outP d c).get1 k = d.out.ptr (stIdx d 0 h (pm * BL + k) : Int) := by
  have hb := block_fits hd hpm
  unfold outP
  rw [ptr_lane d.out (xHead hd hc1 hc2 hh (sSb d) (sSb_lt hd)) (offsM_lane hd hc0 hpm) hb k]
  congr 1
  unfold stIdx sSh
  push_cast
  ring

/-- Lane `k` of the `final_states` pointer block is element `(0, h, m)`. -/
theorem finP_lane {d : Inst} (hd : d.ok) {c : Ctx} {pm h : Nat} (hc0 : c.pid 0 = pm)
    (hc1 : c.pid 1 = 0) (hc2 : c.pid 2 = h) (hh : h < d.nheads) (hpm : pm * BL < d.dim)
    (k : Fin BL) : (finP d c).get1 k = d.finalst.ptr (finIdx d h (pm * BL + k) : Int) := by
  have hb := block_fits hd hpm
  unfold finP
  rw [ptr_lane d.finalst (xHead hd hc1 hc2 hh (sSc d) (sSc_lt hd)) (offsM_lane hd hc0 hpm) hb k]
  congr 1

/-- The `seq_idx` pointer is the start of the buffer. -/
theorem seqP_eq {d : Inst} (hd : d.ok) {c : Ctx} (hc1 : c.pid 1 = 0) :
    seqP d c = d.seqIdx.ptr 0 := by
  have e1 : (tl.program_id c 1).toInt = ((0 : Nat) : Int) := program_id_toInt hc1 (by norm_num)
  have m1 : (tl.program_id c 1 * (IntV.ofNat (sQb d) : I32)).toInt = ((0 * sQb d : Nat) : Int) :=
    IntV.I32.mul_nat e1 (IntV.I32.toInt_ofNat (seqlen_lt hd)) (by simp)
  unfold seqP
  rw [Ptr.add_int, m1, Buf.ptr_offset]
  simp

/-- The `dA_cumsum` pointer is the last entry of chunk `0` of head `h`. -/
theorem dAP_eq {d : Inst} (hd : d.ok) {c : Ctx} {h : Nat} (hc1 : c.pid 1 = 0) (hc2 : c.pid 2 = h)
    (hh : h < d.nheads) : dAP d c = d.dA.ptr (dAIdx d h 0 (d.chunk - 1) : Int) := by
  have hdA := hd.dA_lt
  have hch := chunk_pos hd
  have hlt : h * sDh d < 2 ^ 31 := by
    have h3 : (h + 1) * (d.nchunks * d.chunk) ≤ d.nheads * (d.nchunks * d.chunk) :=
      Nat.mul_le_mul_right _ hh
    rw [Nat.succ_mul] at h3
    unfold sDh
    omega
  have e1 : (tl.program_id c 1).toInt = ((0 : Nat) : Int) := program_id_toInt hc1 (by norm_num)
  have e2 : (tl.program_id c 2).toInt = (h : Int) :=
    program_id_toInt hc2 (by have := hd.heads_lt; omega)
  have hsDh : sDh d < 2 ^ 31 := by
    have : sDh d ≤ d.nheads * (d.nchunks * d.chunk) := by
      unfold sDh
      exact Nat.le_mul_of_pos_left _ (by omega)
    omega
  have m1 : (tl.program_id c 1 * (IntV.ofNat (sDb d) : I32)).toInt = ((0 * sDb d : Nat) : Int) :=
    IntV.I32.mul_nat e1 (IntV.I32.toInt_ofNat hdA) (by simp)
  have m2 : (tl.program_id c 2 * (IntV.ofNat (sDh d) : I32)).toInt = ((h * sDh d : Nat) : Int) :=
    IntV.I32.mul_nat e2 (IntV.I32.toInt_ofNat hsDh) hlt
  have a1 : (tl.program_id c 1 * (IntV.ofNat (sDb d) : I32) +
      tl.program_id c 2 * (IntV.ofNat (sDh d) : I32)).toInt = ((0 * sDb d + h * sDh d : Nat) : Int) :=
    IntV.I32.add_nat m1 m2 (by simpa using hlt)
  have c1 : ((IntV.ofNat d.chunk : I32) - IntV.ofInt 1).toInt = ((d.chunk - 1 : Nat) : Int) :=
    IntV.I32.sub_nat (IntV.I32.toInt_ofNat (chunk_lt hd)) toInt_one (by omega) (chunk_lt hd)
  have c2 : (((IntV.ofNat d.chunk : I32) - IntV.ofInt 1) * (IntV.ofNat 1 : I32)).toInt =
      (((d.chunk - 1) * 1 : Nat) : Int) :=
    IntV.I32.mul_nat c1 toInt_ofNat1 (by have := chunk_lt hd; omega)
  have hsum : (0 * sDb d + h * sDh d) + (d.chunk - 1) * 1 < 2 ^ 31 := by
    have h3 : (h + 1) * (d.nchunks * d.chunk) ≤ d.nheads * (d.nchunks * d.chunk) :=
      Nat.mul_le_mul_right _ hh
    have h4 : d.chunk ≤ d.nchunks * d.chunk := Nat.le_mul_of_pos_left _ (nchunks_pos hd)
    rw [Nat.succ_mul] at h3
    unfold sDh
    omega
  unfold dAP
  rw [Ptr.add_int, IntV.I32.add_nat a1 c2 hsum, Buf.ptr_offset]
  congr 1
  unfold dAIdx sDh
  push_cast
  ring

/-- The elements the first store writes are slot `0`. -/
theorem laneSet_sub_slot0 {d : Inst} {h pm : Nat} (hn0 : 0 < d.nchunks) :
    d.out.elems (laneSet (fun m => stIdx d 0 h m) pm d.dim) ⊆ slotW d h pm 0 := by
  rw [slotW_out hn0]
  refine d.out.elems_mono ?_
  rintro j ⟨k, hk, rfl⟩
  exact ⟨pm * BL + k, ⟨by omega, by have := k.isLt; omega, hk⟩, rfl⟩

/-- The slots after the first are outside the store into slot `0`. -/
theorem iterW_sub {d : Inst} (hd : d.ok) {h pm : Nat} (hh : h < d.nheads) (hn0 : 0 < d.nchunks) :
    iterW (fun t => slotW d h pm (t + 1)) d.nchunks ⊆
      Wp d h pm \ d.out.elems (laneSet (fun m => stIdx d 0 h m) pm d.dim) := by
  rintro x ⟨t, ht, hx⟩
  refine ⟨⟨t + 1, by omega, hx⟩, fun hxo => ?_⟩
  exact Set.disjoint_left.mp (slotW_disjoint hd hh (by omega) (by omega) (by omega))
    hx (laneSet_sub_slot0 hn0 hxo)

/-- **A program with initial states** is safe: it reads only inputs, writes only its slots, and
every slot ends done. -/
theorem safe_program_I {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (hinit : d.hasInit = true) {pm h : Nat} (hpm : pm < gx d)
    (hh : h < d.nheads) {c : Ctx} (hc0 : c.pid 0 = pm) (hc1 : c.pid 1 = 0) (hc2 : c.pid 2 = h) :
    Safe ((launch d).env P) (Rin d) (Wp d h pm) (fun _ m' => Done d init pm h d.nchunks m') init
      ((model d).body c) := by
  have hpmd := block_lt hd hpm
  have hoff := offsM_lane hd hc0 hpmd
  have hdim31 : d.dim < 2 ^ 31 := by have := hd.dim_lt; omega
  have hmask := mask_lane' (offs_m := offsM c) (pm := pm) hdim31 hoff
  have hn0 : 0 < d.nchunks := nchunks_pos hd
  have hnreq : 0 < d.nreq := hd.nreq_pos
  have hlenI : ∀ k : Fin BL, pm * BL + k < d.dim →
      inIdx d 0 h (pm * BL + k) < d.initst.len := fun k hk =>
    lt_of_lt_of_le (inIdx_lt hnreq hh hk) (hd.init_len hinit)
  rw [bodyI_eq d hinit c]
  refine safe_load_buf d.initst (g := fun i : Idx [BL - 0] => inIdx d 0 h (pm * BL + (i.1 : Nat)))
    (en := fun i : Idx [BL - 0] => decide (pm * BL + (i.1 : Nat) < d.dim))
    (addrs_lane d.initst hmask (initP_lane hd hc0 hc2 hh hpmd)) ?_ hpre.2.2.1 ?_
    fun st0 hst0 hoth => ?_
  · intro i hi
    exact hlenI i.1 (of_decide_eq_true hi)
  · refine sub_Rin_initst ?_
    rintro j ⟨i, hi, rfl⟩
    exact hlenI i.1 (of_decide_eq_true hi)
  have hW0 : d.out.elems (laneSet (fun m => stIdx d 0 h m) pm d.dim) ⊆ Wp d h pm :=
    (laneSet_sub_slot0 hn0).trans (fun x hx => ⟨0, by omega, hx⟩)
  refine safe_store_lanes d.out (fun m => stIdx d 0 h m) (n := d.dim) (pm := pm) ?_ ?_ hmask
    (outP_lane hd hc0 hc1 hc2 hh hpmd) hpre.2.2.2.2.2.1 hW0 fun m' hread hframe hpres => ?_
  · intro k k' hk hk' e
    exact Fin.ext (by have := (stIdx_inj hh hh hk hk' e).2.2; omega)
  · intro k hk
    exact lt_of_lt_of_le (stIdx_lt hn0 hh hk) hd.out_len
  have hagree : ∀ x ∈ Rin d, m' x = init x := fun x hx =>
    hframe x (fun hxo => Set.disjoint_left.mp (slotW_disjoint_Rin hd hh (le_refl 0 |>.trans (by omega)))
      ((laneSet_sub_slot0 hn0) hxo) hx)
  refine (safe_loop_I hd hP hpre hinit hh hoff (seqP_eq hd hc1) rfl
    (initP_lane hd hc0 hc2 hh hpmd) (finP_lane hd hc0 hc1 hc2 hh hpmd)
    (st0 := (st0, initP d c, statesP d c, dAP d c,
      (outP d c).map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32)), IntV.ofInt 0,
      IntV.ofInt 0)) (m := m') ?_).mono (fun _ _ hq => hq) (iterW_sub hd hh hn0)
  refine ⟨hagree, hpres, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk hV
    have hr := hst0 (k, ()) (decide_eq_true hk)
    obtain ⟨hf, hv⟩ := init_read hV hinit hnreq hh hk hr
    refine ⟨hf, ?_⟩
    have hv' : (st0.get1 k).toReal = startV init d 0 h (pm * BL + k) := hv
    show |(st0.get1 k).toReal - startV init d 0 h (pm * BL + k)| ≤ 0
    rw [hv', sub_self, abs_zero]
  · exact statesP_lane hd hc0 hc1 hc2 hh hpmd
  · exact dAP_eq hd hc1 hc2 hh
  · intro k
    show ((outP d c).map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32))).get1 k = _
    rw [get1_map_add, outP_lane hd hc0 hc1 hc2 hh hpmd k, sSc_toInt hd, Buf.ptr_offset]
    congr 1
    unfold stIdx sSc
    push_cast
    ring
  · show (IntV.ofInt 0 : I32).toInt = _
    rw [toInt_zero]
    simp [prevReq]
  · show (IntV.ofInt 0 : I32).toInt = _
    rw [toInt_zero]
    simp [prevReq, pc_zero]
  · refine ⟨fun m hm => ?_, fun c' hc1' hc2' => absurd hc2' (by omega)⟩
    obtain ⟨hm1, hm2, hm3⟩ := hm
    have hk : m - pm * BL < BL := by omega
    have hmk : pm * BL + (m - pm * BL) = m := by omega
    refine ⟨st0.get1 ⟨m - pm * BL, hk⟩, ?_, ?_⟩
    · unfold slotRead
      rw [ite_eq_left hn0]
      have := hread ⟨m - pm * BL, hk⟩ (by show pm * BL + (m - pm * BL) < d.dim; omega)
      simpa [hmk] using this
    · rw [ite_eq_left hinit]
      refine ⟨st0.get1 ⟨m - pm * BL, hk⟩, ?_, FloatV.NaNEq.refl _⟩
      have := hst0 (⟨m - pm * BL, hk⟩, ()) (decide_eq_true (by show pm * BL + (m - pm * BL) < d.dim; omega))
      simpa [hmk, Tensor.get1] using this

/-- **A program without initial states** is safe. -/
theorem safe_program_N {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (hnoinit : d.hasInit = false) {pm h : Nat} (hpm : pm < gx d)
    (hh : h < d.nheads) {c : Ctx} (hc0 : c.pid 0 = pm) (hc1 : c.pid 1 = 0) (hc2 : c.pid 2 = h) :
    Safe ((launch d).env P) (Rin d) (Wp d h pm) (fun _ m' => Done d init pm h d.nchunks m') init
      ((model d).body c) := by
  have hpmd := block_lt hd hpm
  have hoff := offsM_lane hd hc0 hpmd
  have hdim31 : d.dim < 2 ^ 31 := by have := hd.dim_lt; omega
  have hmask := mask_lane' (offs_m := offsM c) (pm := pm) hdim31 hoff
  have hn0 : 0 < d.nchunks := nchunks_pos hd
  have hW0 : d.out.elems (laneSet (fun m => stIdx d 0 h m) pm d.dim) ⊆ Wp d h pm :=
    (laneSet_sub_slot0 hn0).trans (fun x hx => ⟨0, by omega, hx⟩)
  rw [bodyN_eq d hnoinit c]
  refine safe_store_lanes d.out (fun m => stIdx d 0 h m) (n := d.dim) (pm := pm) ?_ ?_ hmask
    (outP_lane hd hc0 hc1 hc2 hh hpmd) hpre.2.2.2.2.2.1 hW0 fun m' hread hframe hpres => ?_
  · intro k k' hk hk' e
    exact Fin.ext (by have := (stIdx_inj hh hh hk hk' e).2.2; omega)
  · intro k hk
    exact lt_of_lt_of_le (stIdx_lt hn0 hh hk) hd.out_len
  have hagree : ∀ x ∈ Rin d, m' x = init x := fun x hx =>
    hframe x (fun hxo => Set.disjoint_left.mp (slotW_disjoint_Rin hd hh (le_refl 0 |>.trans (by omega)))
      ((laneSet_sub_slot0 hn0) hxo) hx)
  have hz : ∀ k : Fin BL, (tl.zeros [BL] .f32).get1 k = FloatV.ofPat 0 := fun _ => rfl
  refine (safe_loop_N hd hP hpre hnoinit hh hoff (seqP_eq hd hc1)
    (finP_lane hd hc0 hc1 hc2 hh hpmd)
    (st0 := (tl.zeros [BL] .f32, statesP d c, dAP d c,
      (outP d c).map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32)), IntV.ofInt 0))
    (m := m') ?_).mono (fun _ _ hq => hq) (iterW_sub hd hh hn0)
  refine ⟨hagree, hpres, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk hV
    have hzf : (FloatV.ofPat 0 : F32).isFinite = true := FloatV.isFinite_ofPat_zero Format.binary32_regular
    refine ⟨by rw [hz k]; exact hzf, ?_⟩
    show |((tl.zeros [BL] .f32).get1 k).toReal - startV init d 0 h (pm * BL + k)| ≤ 0
    rw [hz k]
    simp [startV, hnoinit]
  · exact statesP_lane hd hc0 hc1 hc2 hh hpmd
  · exact dAP_eq hd hc1 hc2 hh
  · intro k
    show ((outP d c).map (fun (q : Ptr .f32) => q + (IntV.ofNat (sSc d) : I32))).get1 k = _
    rw [get1_map_add, outP_lane hd hc0 hc1 hc2 hh hpmd k, sSc_toInt hd, Buf.ptr_offset]
    congr 1
    unfold stIdx sSc
    push_cast
    ring
  · show (IntV.ofInt 0 : I32).toInt = _
    rw [toInt_zero]
    simp [prevReq]
  · refine ⟨fun m hm => ?_, fun c' hc1' hc2' => absurd hc2' (by omega)⟩
    obtain ⟨hm1, hm2, hm3⟩ := hm
    have hk : m - pm * BL < BL := by omega
    have hmk : pm * BL + (m - pm * BL) = m := by omega
    refine ⟨(tl.zeros [BL] .f32).get1 ⟨m - pm * BL, hk⟩, ?_, ?_⟩
    · unfold slotRead
      rw [ite_eq_left hn0]
      have := hread ⟨m - pm * BL, hk⟩ (by show pm * BL + (m - pm * BL) < d.dim; omega)
      simpa [hmk] using this
    · rw [ite_eq_right (by simp [hnoinit]), hz]

end Claims.SsdStatePassing
