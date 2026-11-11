import Proof.Setup
import Proof.Reduction
import Proof.Epilogue
import Proof.InnerLoop

/-!
# `tma_persistent_matmul_f16`: One program of the launch, and the launch.
-/

namespace Claims.TmaPersistentMatmul

open TVA

/-!
## The epilogue of one tile is `Safe`

After the reduction loop, the kernel recomputes the tile coordinates with the cell id, converts both
halves of the accumulator from `float32` to `float16`, and stores them to the two halves of the
tile of `c`. This file shows that sequence `Safe`, with the tile's elements as write footprint, and
that it leaves `Qfact` for that tile: every in-bounds lane is present, and a finite row and column
give the specified value.
-/


/-- The epilogue of one tile: recompute the tile coordinates, convert both halves and store them. -/
def epiProg (d : Inst) (tile_id_c num_pid_in_group num_pid_m : I32)
    (accumulator : Tensor [128, 256] F32) : Prog I32 :=
  do
    let (pid_m, pid_n) ← compute_pid tile_id_c num_pid_in_group num_pid_m 8
    let offs_am_c := pid_m * (IntV.ofNat 128 : I32)
    let offs_bn_c := pid_n * (IntV.ofNat 256 : I32)
    let acc := accumulator.reshape [128, 2, 128] (by norm_num)
    let acc : Tensor [128, 128, 2] F32 := acc.permute [0, 2, 1]
    let (acc0, acc1) := Tensor.split (s := [128, 128]) acc
    let c0 : Tensor [128, 128] F16 ← tl.cvt acc0
    tl.desc_store (cDesc d) [offs_am_c, offs_bn_c] c0
    let c1 : Tensor [128, 128] F16 ← tl.cvt acc1
    tl.desc_store (cDesc d) [offs_am_c, offs_bn_c + (IntV.ofNat 128 : I32)] c1
    return tile_id_c

/-- A read of an element outside the written set is unchanged. -/
theorem read_preserved {δ : DType} {b : Buf δ} {S S' : Set Nat} (hS : Disjoint S S')
    {j : Nat} (hj : j ∈ S') {m m' : Mem} (hfr : ∀ x, x ∉ b.elems S → m' x = m x) :
    m'.read δ (b.ptr (j : Int)).addr = m.read δ (b.ptr (j : Int)).addr :=
  Mem.read_congr fun x hx =>
    hfr x fun hximg =>
      (Set.disjoint_left.mp (Buf.elems_disjoint b hS) hximg) (Buf.span_sub_elems b hj hx)

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 4000 in
/-- The epilogue of one tile is `Safe`. -/
theorem safe_epi {d : Inst} (hd : d.ok) {E : Env} {init m : Mem} {T : Nat}
    {acc : Tensor [128, 256] F32} {tile_id_c num_pid_in_group num_pid_m : I32}
    (htilec : tile_id_c.toInt = (T : Int))
    (hG : num_pid_in_group.toInt = ((8 * numPidN d : Nat) : Int))
    (hM : num_pid_m.toInt = ((numPidM d : Nat) : Int))
    (hT : T < tiles d)
    (hinv : AccInv d init T (steps d.K) acc)
    (hvc : d.c.Valid m) :
    Safe E (R d) (d.c.elems (tileIdx d T))
      (fun a m' => a.toInt = (T : Int) ∧ Qfact d init T m' ∧ d.c.Valid m' ∧
        (∀ x, x ∉ d.c.elems (tileIdx d T) → m' x = m x) ∧
        (∀ x, m x ≠ none → m' x ≠ none)) m
      (epiProg d tile_id_c num_pid_in_group num_pid_m acc) := by
  have h128i : (IntV.ofNat 128 : I32).toInt = ((128 : Nat) : Int) :=
    IntV.I32.toInt_ofNat (by norm_num)
  have h256i : (IntV.ofNat 256 : I32).toInt = ((256 : Nat) : Int) :=
    IntV.I32.toInt_ofNat (by norm_num)
  have hpid128 : pidM d T * 128 < 2 ^ 31 := by
    have h1 : 128 * pidM d T ≤ 128 * numPidM d :=
      Nat.mul_le_mul_left _ (le_of_lt (pidM_lt hd hT))
    have h2 := mul128_numPidM_le d
    have h3 := hd.M_le
    omega
  have hpid256 : pidN d T * 256 + 128 < 2 ^ 31 := by
    have h1 : 256 * pidN d T ≤ 256 * numPidN d :=
      Nat.mul_le_mul_left _ (le_of_lt (pidN_lt hd hT))
    have h2 := mul256_numPidN_le d
    have h3 := hd.N_le
    omega
  unfold epiProg
  refine safe_compute_pid hd htilec hG hM hT ?_
  rintro ⟨pm, pn⟩ hpm hpn
  dsimp only
  have hoffA : (pm * (IntV.ofNat 128 : I32)).toInt = ((128 * pidM d T : Nat) : Int) := by
    have h := IntV.I32.mul_nat (a := pm) (b := (IntV.ofNat 128 : I32)) (x := pidM d T) (y := 128)
      hpm h128i (by omega)
    rw [h, Nat.mul_comm]
  have hoffB : (pn * (IntV.ofNat 256 : I32)).toInt = ((256 * pidN d T : Nat) : Int) := by
    have h := IntV.I32.mul_nat (a := pn) (b := (IntV.ofNat 256 : I32)) (x := pidN d T) (y := 256)
      hpn h256i (by omega)
    rw [h, Nat.mul_comm]
  have hoffB2 : (pn * (IntV.ofNat 256 : I32) + (IntV.ofNat 128 : I32)).toInt
      = ((256 * pidN d T : Nat) : Int) + 128 := by
    have h := IntV.I32.add_nat (a := pn * (IntV.ofNat 256 : I32)) (b := (IntV.ofNat 128 : I32))
      (x := 256 * pidN d T) (y := 128) hoffB h128i (by omega)
    rw [h]
    push_cast
    ring
  let G1 : Idx [128, 128] → Nat := fun i => ((cDesc d).elemAt (TensorDesc.coord
    [(pm * (IntV.ofNat 128 : I32)).toInt, (pn * (IntV.ofNat 256 : I32)).toInt] i)).toNat
  let E1 : Idx [128, 128] → Bool := fun i => (cDesc d).inbAt (TensorDesc.coord
    [(pm * (IntV.ofNat 128 : I32)).toInt, (pn * (IntV.ofNat 256 : I32)).toInt] i)
  let G2 : Idx [128, 128] → Nat := fun i => ((cDesc d).elemAt (TensorDesc.coord
    [(pm * (IntV.ofNat 128 : I32)).toInt,
      (pn * (IntV.ofNat 256 : I32) + (IntV.ofNat 128 : I32)).toInt] i)).toNat
  let E2 : Idx [128, 128] → Bool := fun i => (cDesc d).inbAt (TensorDesc.coord
    [(pm * (IntV.ofNat 128 : I32)).toInt,
      (pn * (IntV.ofNat 256 : I32) + (IntV.ofNat 128 : I32)).toInt] i)
  have hW1 : Buf.elems d.c (Buf.img G1 E1) ⊆ Buf.elems d.c (tileIdx d T) :=
    Buf.elems_mono d.c (by
      simpa only [G1, E1, hoffA, hoffB] using cDesc_img_tile1 d T)
  have hW2 : Buf.elems d.c (Buf.img G2 E2) ⊆ Buf.elems d.c (tileIdx d T) :=
    Buf.elems_mono d.c (by
      simpa only [G2, E2, hoffA, hoffB2] using cDesc_img_tile2 d T)
  have hdisj : Disjoint (Buf.img G1 E1) (Buf.img G2 E2) := by
    simpa only [G1, E1, G2, E2, hoffA, hoffB, hoffB2] using cDesc_img12_disjoint hd T
  have hacc1 : (cDesc d).Access
      ([pm * (IntV.ofNat 128 : I32), pn * (IntV.ofNat 256 : I32)].map IntV.toInt) [128, 128] :=
    cDesc_access hd _ _ (by rw [hoffB]; push_cast; omega)
  have hacc2 : (cDesc d).Access
      ([pm * (IntV.ofNat 128 : I32),
        pn * (IntV.ofNat 256 : I32) + (IntV.ofNat 128 : I32)].map IntV.toInt) [128, 128] :=
    cDesc_access hd _ _ (by rw [hoffB2]; push_cast; omega)
  refine safe_cvt (x := (Tensor.split (s := [128, 128])
    ((acc.reshape [128, 2, 128] (by norm_num)).permute [0, 2, 1])).1)
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩ ?_
  intro c0 hne0
  refine safe_desc_store (b := d.c) (bs := [128, 128]) (D := cDesc d)
    (offsets := [pm * (IntV.ofNat 128 : I32), pn * (IntV.ofNat 256 : I32)]) (vals := c0) rfl
    hacc1 ?_ ?_ hvc hW1 ?_
  · intro i hi
    exact cDesc_len hd _ _ i hi
  · intro i j hi hj he
    exact cDesc_toNat_inj d hd _ _ i j hi hj he
  intro m1 hread1 hfr1 hpres1
  refine safe_cvt (x := (Tensor.split (s := [128, 128])
    ((acc.reshape [128, 2, 128] (by norm_num)).permute [0, 2, 1])).2)
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩ ?_
  intro c1 hne1
  refine safe_desc_store (b := d.c) (bs := [128, 128]) (D := cDesc d)
    (offsets := [pm * (IntV.ofNat 128 : I32),
      pn * (IntV.ofNat 256 : I32) + (IntV.ofNat 128 : I32)]) (vals := c1) rfl hacc2 ?_ ?_
    (d.c.valid_of_agree hvc fun x hx => hpres1 x (hvc.2.2.2 x hx)) ?_ ?_
  · intro i hi
    exact cDesc_len hd _ _ i hi
  · intro i j hi hj he
    exact cDesc_toNat_inj d hd _ _ i j hi hj he
  · intro x hx
    exact ⟨hW2 hx, fun hx1 => (Set.disjoint_left.mp (Buf.elems_disjoint d.c hdisj) hx1) hx⟩
  intro m2 hread2 hfr2 hpres2
  refine .ret ⟨htilec, ?_, ?_, ?_, ?_⟩
  · intro r n hr hn hrM hnN
    let rj : Fin 128 := ⟨r, hr⟩
    by_cases hn128 : n < 128
    · -- the first half: stored first, and preserved by the second store
      let col : Fin 128 := ⟨n, hn128⟩
      let hi : Idx [128, 128] := (rj, (col, ()))
      have hi0 : (hi.coord 0 : Int) = (r : Int) := by simp [hi, rj, Idx.coord]
      have hi1 : (hi.coord 1 : Int) = (n : Int) := by simp [hi, col, Idx.coord]
      have hinb : E1 hi = true := by
        refine (cDesc_inbAt d _ _ hi).mpr ⟨?_, ?_, ?_, ?_⟩
        · rw [hi0]; omega
        · rw [hi0]; omega
        · rw [hi1]; omega
        · rw [hi1]; omega
      have helem : ((cDesc d).elemAt (TensorDesc.coord (List.map IntV.toInt
          [pm * (IntV.ofNat 128 : I32), pn * (IntV.ofNat 256 : I32)]) hi)).toNat
          = (128 * pidM d T + r) * d.ldc + (256 * pidN d T + n) := by
        simp only [List.map_cons, List.map_nil]
        rw [cDesc_elemAt_toNat d _ _ hi (by rw [hi0]; omega)
          (by rw [hi1]; omega)]
        rw [show (IntV.toInt (pm * IntV.ofNat 128) + (hi.coord 0 : Int))
              = ((128 * pidM d T + r : Nat) : Int) by
              rw [hi0, hoffA]; push_cast; ring,
            show (IntV.toInt (pn * IntV.ofNat 256) + (hi.coord 1 : Int))
              = ((256 * pidN d T + n : Nat) : Int) by
              rw [hi1, hoffB]; push_cast; ring,
            Int.toNat_natCast, Int.toNat_natCast]
      have hjmem : (128 * pidM d T + r) * d.ldc + (256 * pidN d T + n) ∈ Buf.img G1 E1 := by
        rw [← helem]
        exact ⟨hi, hinb, rfl⟩
      have hr1 : m1.read .f16 (d.c.ptr (↑((128 * pidM d T + r) * d.ldc
          + (256 * pidN d T + n)))).addr = some (c0.get hi) := by
        have h := hread1 hi hinb
        simp only [TensorDesc.lane] at h
        rw [helem] at h
        exact h
      have hr2 : m2.read .f16 (d.c.ptr (↑((128 * pidM d T + r) * d.ldc
          + (256 * pidN d T + n)))).addr = some (c0.get hi) := by
        rw [read_preserved (b := d.c) (S := Buf.img G2 E2) (S' := Buf.img G1 E1) hdisj.symm
          hjmem hfr2]
        exact hr1
      refine ⟨⟨c0.get hi, hr2⟩, ?_⟩
      intro hRow hCol
      obtain ⟨hfin, herr⟩ := AccInv_final hd hinv rj ⟨n, hn⟩ hrM hnN hRow hCol
      have herr' : |(acc.get2 rj ⟨n, hn⟩).toRat
            - dSum d init (128 * pidM d T + r) (256 * pidN d T + n)|
          ≤ dTol d init (128 * pidM d T + r) (256 * pidN d T + n) := herr
      have hacc0 : ((acc.reshape [128, 2, 128] (by norm_num)).permute [0, 2, 1]).split.1.get hi
          = acc.get2 rj ⟨n, hn⟩ := by
        show ((acc.reshape [128, 2, 128] (by norm_num)).permute [0, 2, 1]).split.1.get2 rj col
          = acc.get2 rj ⟨n, hn⟩
        rw [acc0_get2]
      have hne : FloatV.NaNEq (FloatV.cvt (acc.get2 rj ⟨n, hn⟩)) (c0.get hi) := by
        have h := hne0 hi
        rw [hacc0] at h
        exact h
      refine ⟨acc.get2 rj ⟨n, hn⟩, hfin, herr', c0.get hi, hr2, hne, ?_⟩
      intro hroom
      have hle : |(acc.get2 rj ⟨n, hn⟩).toRat| ≤ Format.binary16.maxVal := by
        have h2 : |(acc.get2 rj ⟨n, hn⟩).toRat
              - dSum d init (128 * pidM d T + r) (256 * pidN d T + n)|
            + |dSum d init (128 * pidM d T + r) (256 * pidN d T + n)|
            ≤ Format.binary16.maxVal := by linarith
        calc |(acc.get2 rj ⟨n, hn⟩).toRat|
            = |((acc.get2 rj ⟨n, hn⟩).toRat
                - dSum d init (128 * pidM d T + r) (256 * pidN d T + n))
                + dSum d init (128 * pidM d T + r) (256 * pidN d T + n)| := by
              rw [sub_add_cancel]
          _ ≤ |(acc.get2 rj ⟨n, hn⟩).toRat
                - dSum d init (128 * pidM d T + r) (256 * pidN d T + n)|
              + |dSum d init (128 * pidM d T + r) (256 * pidN d T + n)| := abs_add_le _ _
          _ ≤ _ := h2
      have hfv := cvt_f16_finite hfin hle
      have hnn : (FloatV.cvt (acc.get2 rj ⟨n, hn⟩) : F16).isNaN = false :=
        (FloatV.isFinite_iff.mp hfv).1
      rw [FloatV.NaNEq.eq hne hnn]
      exact hfv
    · -- the second half: stored last
      have hn128' : 128 ≤ n := by omega
      let col : Fin 128 := ⟨n - 128, by omega⟩
      let hi : Idx [128, 128] := (rj, (col, ()))
      have hi0 : (hi.coord 0 : Int) = (r : Int) := by simp [hi, rj, Idx.coord]
      have hi1 : (hi.coord 1 : Int) = ((n : Int) - 128) := by
        simp [hi, col, Idx.coord]
        omega
      have hinb : E2 hi = true := by
        refine (cDesc_inbAt d _ _ hi).mpr ⟨?_, ?_, ?_, ?_⟩
        · rw [hi0]; omega
        · rw [hi0]; omega
        · rw [hi1]; omega
        · rw [hi1]; omega
      have helem : ((cDesc d).elemAt (TensorDesc.coord (List.map IntV.toInt
          [pm * (IntV.ofNat 128 : I32),
            pn * (IntV.ofNat 256 : I32) + (IntV.ofNat 128 : I32)]) hi)).toNat
          = (128 * pidM d T + r) * d.ldc + (256 * pidN d T + n) := by
        simp only [List.map_cons, List.map_nil]
        rw [cDesc_elemAt_toNat d _ _ hi (by rw [hi0]; omega)
          (by rw [hi1]; omega)]
        rw [show (IntV.toInt (pm * IntV.ofNat 128) + (hi.coord 0 : Int))
              = ((128 * pidM d T + r : Nat) : Int) by
              rw [hi0, hoffA]; push_cast; ring,
            show (IntV.toInt (pn * IntV.ofNat 256 + IntV.ofNat 128) + (hi.coord 1 : Int))
              = ((256 * pidN d T + n : Nat) : Int) by
              rw [hi1, hoffB2]; push_cast; ring,
            Int.toNat_natCast, Int.toNat_natCast]
      have hr2 : m2.read .f16 (d.c.ptr (↑((128 * pidM d T + r) * d.ldc
          + (256 * pidN d T + n)))).addr = some (c1.get hi) := by
        have h := hread2 hi hinb
        simp only [TensorDesc.lane] at h
        rw [helem] at h
        exact h
      refine ⟨⟨c1.get hi, hr2⟩, ?_⟩
      intro hRow hCol
      obtain ⟨hfin, herr⟩ := AccInv_final hd hinv rj ⟨n, hn⟩ hrM hnN hRow hCol
      have herr' : |(acc.get2 rj ⟨n, hn⟩).toRat
            - dSum d init (128 * pidM d T + r) (256 * pidN d T + n)|
          ≤ dTol d init (128 * pidM d T + r) (256 * pidN d T + n) := herr
      have hacc1 : ((acc.reshape [128, 2, 128] (by norm_num)).permute [0, 2, 1]).split.2.get hi
          = acc.get2 rj ⟨n, hn⟩ := by
        show ((acc.reshape [128, 2, 128] (by norm_num)).permute [0, 2, 1]).split.2.get2 rj col
          = acc.get2 rj ⟨n, hn⟩
        rw [acc1_get2]
        congr 1
        exact Fin.ext (by simp [col]; omega)
      have hne : FloatV.NaNEq (FloatV.cvt (acc.get2 rj ⟨n, hn⟩)) (c1.get hi) := by
        have h := hne1 hi
        rw [hacc1] at h
        exact h
      refine ⟨acc.get2 rj ⟨n, hn⟩, hfin, herr', c1.get hi, hr2, hne, ?_⟩
      intro hroom
      have hle : |(acc.get2 rj ⟨n, hn⟩).toRat| ≤ Format.binary16.maxVal := by
        have h2 : |(acc.get2 rj ⟨n, hn⟩).toRat
              - dSum d init (128 * pidM d T + r) (256 * pidN d T + n)|
            + |dSum d init (128 * pidM d T + r) (256 * pidN d T + n)|
            ≤ Format.binary16.maxVal := by linarith
        calc |(acc.get2 rj ⟨n, hn⟩).toRat|
            = |((acc.get2 rj ⟨n, hn⟩).toRat
                - dSum d init (128 * pidM d T + r) (256 * pidN d T + n))
                + dSum d init (128 * pidM d T + r) (256 * pidN d T + n)| := by
              rw [sub_add_cancel]
          _ ≤ |(acc.get2 rj ⟨n, hn⟩).toRat
                - dSum d init (128 * pidM d T + r) (256 * pidN d T + n)|
              + |dSum d init (128 * pidM d T + r) (256 * pidN d T + n)| := abs_add_le _ _
          _ ≤ _ := h2
      have hfv := cvt_f16_finite hfin hle
      have hnn : (FloatV.cvt (acc.get2 rj ⟨n, hn⟩) : F16).isNaN = false :=
        (FloatV.isFinite_iff.mp hfv).1
      rw [FloatV.NaNEq.eq hne hnn]
      exact hfv
  · exact d.c.valid_of_agree hvc fun x hx => hpres2 x (hpres1 x (hvc.2.2.2 x hx))
  · intro x hx
    rw [hfr2 x (fun hximg => hx (hW2 hximg)), hfr1 x (fun hximg => hx (hW1 hximg))]
  · intro x hx
    exact hpres2 x (hpres1 x hx)

/-!
## The kernel body is `Safe`

One tile: recompute the tile coordinates, run the inner reduction loop, and run the epilogue. The
persistent loop: the prologue divides to get the tile counts, then every program runs its tiles
with a fixed stride `NUM_SMS`, carrying the validity of `a`, `b`, `c` and the per-tile facts of the
tiles already written.
-/


/-- The first program axis of a context is its program id. -/
theorem launch_pid_toInt {d : Inst} (hd : d.ok) {i : Fin (launch d).size} :
    (tl.program_id ((launch d).ctx i) 0).toInt = ((i : Nat) : Int) := by
  have hlt : (i : Nat) < min d.NUM_SMS (tiles d) := by
    rw [← launch_size d]; exact i.isLt
  have h31 : (i : Nat) < 2 ^ 31 := by
    have h1 : min d.NUM_SMS (tiles d) ≤ d.NUM_SMS := min_le_left _ _
    have h2 := hd.sms.2
    omega
  show (IntV.ofNat (((launch d).ctx i).pid 0)).toInt = ((i : Nat) : Int)
  rw [launch_pid d i]
  exact IntV.I32.toInt_ofNat h31

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4000 in
/-- One tile of a program is `Safe`: it writes the tile's elements of `c` and leaves the tile fact
for it, keeping `a`, `b` and `c` valid. -/
theorem safe_one_tile {d : Inst} (hd : d.ok) {E : Env}
    (hacc : Device.DotLiftAcc E.P ⟨.f16, .f16, .f32, .tf32, 0⟩
      Device.relDot16 Device.absDot16)
    (htot : Device.DotTotal E.P ⟨.f16, .f16, .f32, .tf32, 0⟩)
    {init : Mem} {T : Nat} (hT : T < tiles d)
    {tid num_pid_in_group num_pid_m k_tiles : I32}
    (htid : tid.toInt = (T : Int))
    (hG : num_pid_in_group.toInt = ((8 * numPidN d : Nat) : Int))
    (hM : num_pid_m.toInt = ((numPidM d : Nat) : Int))
    (hkt : k_tiles.toInt = ((steps d.K : Nat) : Int))
    {st : I32} (hst : (st + (IntV.ofNat d.NUM_SMS : I32)).toInt = (T : Int))
    {m : Mem} (hva : d.a.Valid m) (hvb : d.b.Valid m) (hvc : d.c.Valid m)
    (hRA : ∀ x ∈ d.a.elems (aIdx d), m x = init x)
    (hRB : ∀ x ∈ d.b.elems (bIdx d), m x = init x) :
    Safe E (R d) (d.c.elems (tileIdx d T))
      (fun st' m' => st'.toInt = (T : Int) ∧ d.a.Valid m' ∧ d.b.Valid m' ∧ d.c.Valid m' ∧
        Qfact d init T m' ∧
        (∀ x, x ∉ d.c.elems (tileIdx d T) → m' x = m x) ∧
        (∀ x, m x ≠ none → m' x ≠ none)) m
      (do
        let (pid_m, pid_n) ← compute_pid tid num_pid_in_group num_pid_m 8
        let offs_am := pid_m * (IntV.ofNat 128 : I32)
        let offs_bn := pid_n * (IntV.ofNat 256 : I32)
        let accumulator := tl.zeros [128, 256] .f32
        let accumulator ← tl.forRange (IntV.ofInt 0 : I32) k_tiles (IntV.ofInt 1 : I32) accumulator
            (fun ki accumulator => do
          let offs_k := ki * (IntV.ofNat 64 : I32)
          let a ← tl.desc_load (bs := [128, 64]) (aDesc d) [offs_am, offs_k]
          let b ← tl.desc_load (bs := [256, 64]) (bDesc d) [offs_bn, offs_k]
          tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
            a b.trans accumulator)
        let tile_id_c := st + (IntV.ofNat d.NUM_SMS : I32)
        epiProg d tile_id_c num_pid_in_group num_pid_m accumulator) := by
  have h128i : (IntV.ofNat 128 : I32).toInt = ((128 : Nat) : Int) :=
    IntV.I32.toInt_ofNat (by norm_num)
  have h256i : (IntV.ofNat 256 : I32).toInt = ((256 : Nat) : Int) :=
    IntV.I32.toInt_ofNat (by norm_num)
  have hpid128 : pidM d T * 128 < 2 ^ 31 := by
    have h1 : 128 * pidM d T ≤ 128 * numPidM d :=
      Nat.mul_le_mul_left _ (le_of_lt (pidM_lt hd hT))
    have h2 := mul128_numPidM_le d
    have h3 := hd.M_le
    omega
  have hpid256 : pidN d T * 256 + 128 < 2 ^ 31 := by
    have h1 : 256 * pidN d T ≤ 256 * numPidN d :=
      Nat.mul_le_mul_left _ (le_of_lt (pidN_lt hd hT))
    have h2 := mul256_numPidN_le d
    have h3 := hd.N_le
    omega
  refine safe_compute_pid hd htid hG hM hT ?_
  rintro ⟨pm, pn⟩ hpm hpn
  dsimp only
  have hoffA : (pm * (IntV.ofNat 128 : I32)).toInt = ((128 * pidM d T : Nat) : Int) := by
    have h := IntV.I32.mul_nat (a := pm) (b := (IntV.ofNat 128 : I32)) (x := pidM d T) (y := 128)
      hpm h128i (by omega)
    rw [h, Nat.mul_comm]
  have hoffB : (pn * (IntV.ofNat 256 : I32)).toInt = ((256 * pidN d T : Nat) : Int) := by
    have h := IntV.I32.mul_nat (a := pn) (b := (IntV.ofNat 256 : I32)) (x := pidN d T) (y := 256)
      hpn h256i (by omega)
    rw [h, Nat.mul_comm]
  refine (Safe.bind (W₁ := ∅) (W₂ := d.c.elems (tileIdx d T)) ?_
    (by simp)).mono (fun _ _ h => h) ?_
  · refine (safe_inner (d := d) (E := E) (T := T) (m := m)
      (acc := tl.zeros [128, 256] .f32) (kt := k_tiles)
      (offs_am := pm * (IntV.ofNat 128 : I32))
      (offs_bn := pn * (IntV.ofNat 256 : I32)) hd hacc htot hkt hoffA hoffB
      (AccInv_zero d init T) hva hvb hRA hRB).mono ?_ (Set.Subset.refl _)
    intro acc m' h
    obtain ⟨hinv, hmem⟩ := h
    rw [hmem]
    refine (safe_epi (d := d) (E := E) (init := init) (T := T) (m := m)
      (acc := acc) (tile_id_c := st + (IntV.ofNat d.NUM_SMS : I32))
      hd hst hG hM hT hinv hvc).mono ?_ (Set.Subset.refl _)
    intro a m'' hq
    obtain ⟨haT, hqf, hvc', hfr, hpres⟩ := hq
    exact ⟨haT, Buf.valid_of_agree hva (fun x hx => hpres x (hva.2.2.2 x hx)),
      Buf.valid_of_agree hvb (fun x hx => hpres x (hvb.2.2.2 x hx)), hvc', hqf, hfr, hpres⟩
  · intro x hx
    simpa only [Set.empty_union] using hx

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4000 in
/-- The kernel body, with the epilogue of one tile packaged as `epiProg`. -/
theorem body_eq_epi (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let start_pid := tl.program_id c 0
        let num_pid_m := (← tl.idiv (Tensor.scalar (IntV.ofNat d.M + (IntV.ofNat 127 : I32)))
          (Tensor.scalar (IntV.ofNat 128 : I32))).item
        let num_pid_n := (← tl.idiv (Tensor.scalar (IntV.ofNat d.N + (IntV.ofNat 255 : I32)))
          (Tensor.scalar (IntV.ofNat 256 : I32))).item
        let k_tiles := (← tl.idiv (Tensor.scalar (IntV.ofNat d.K + (IntV.ofNat 63 : I32)))
          (Tensor.scalar (IntV.ofNat 64 : I32))).item
        let num_tiles := num_pid_m * num_pid_n
        let tile_id_c := start_pid - (IntV.ofNat d.NUM_SMS : I32)
        let num_pid_in_group := (IntV.ofNat 8 : I32) * num_pid_n
        let _tile_id_c ← tl.forRange start_pid num_tiles (IntV.ofNat d.NUM_SMS : I32) tile_id_c
            (fun tile_id tile_id_c => do
          let (pid_m, pid_n) ← compute_pid tile_id num_pid_in_group num_pid_m 8
          let offs_am := pid_m * (IntV.ofNat 128 : I32)
          let offs_bn := pid_n * (IntV.ofNat 256 : I32)
          let accumulator := tl.zeros [128, 256] .f32
          let accumulator ← tl.forRange (IntV.ofInt 0 : I32) k_tiles (IntV.ofInt 1 : I32) accumulator
              (fun ki accumulator => do
            let offs_k := ki * (IntV.ofNat 64 : I32)
            let a ← tl.desc_load (bs := [128, 64]) (aDesc d) [offs_am, offs_k]
            let b ← tl.desc_load (bs := [256, 64]) (bDesc d) [offs_bn, offs_k]
            tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
              a b.trans accumulator)
          let tile_id_c := tile_id_c + (IntV.ofNat d.NUM_SMS : I32)
          epiProg d tile_id_c num_pid_in_group num_pid_m accumulator)
        pure ()) := by
  rw [body_eq]
  unfold epiProg
  rfl

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4000 in
/-- The body of a kernel instance is `Safe`. -/
theorem safe_body {d : Inst} (hd : d.ok) {P : Platform}
    (hacc : Device.DotLiftAcc P ⟨.f16, .f16, .f32, .tf32, 0⟩
      Device.relDot16 Device.absDot16)
    (htot : Device.DotTotal P ⟨.f16, .f16, .f32, .tf32, 0⟩)
    {init : Mem} {i : Fin (launch d).size} (hPre : Pre d init) :
    Safe ((launch d).env P) (R d) (W d i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  obtain ⟨hva0, hvb0, hvc0⟩ := hPre
  have hS : 0 < d.NUM_SMS := hd.sms.1
  have hlt : (i : Nat) < min d.NUM_SMS (tiles d) := by
    rw [← launch_size d]; exact i.isLt
  have hiS : (i : Nat) < d.NUM_SMS := lt_of_lt_of_le hlt (min_le_left _ _)
  have hile : (i : Nat) ≤ tiles d := le_of_lt (lt_of_lt_of_le hlt (min_le_right _ _))
  have hpid : (tl.program_id ((launch d).ctx i) 0).toInt = ((i : Nat) : Int) := launch_pid_toInt hd
  have hmax : IntV.maxInt true 32 = 2 ^ 31 - 1 := by
    norm_num [IntV.maxInt]
  have hstepv : (IntV.ofNat d.NUM_SMS : I32).toInt = ((d.NUM_SMS : Nat) : Int) :=
    IntV.I32.toInt_ofNat hd.sms.2
  have hM31 : d.M < 2 ^ 31 := by have := hd.M_ctx; omega
  have hN31 : d.N < 2 ^ 31 := by have := hd.N_le; omega
  have hK31 : d.K < 2 ^ 31 := by have := hd.K_le; omega
  have h127 : (127 : Nat) < 2 ^ 31 := by norm_num
  have h255 : (255 : Nat) < 2 ^ 31 := by norm_num
  have h63 : (63 : Nat) < 2 ^ 31 := by norm_num
  have h64v : (IntV.ofNat 64 : I32).toInt = ((64 : Nat) : Int) :=
    IntV.I32.toInt_ofNat (by norm_num)
  have h256v : (IntV.ofNat 256 : I32).toInt = ((256 : Nat) : Int) :=
    IntV.I32.toInt_ofNat (by norm_num)
  have hMv : (IntV.ofNat d.M + (IntV.ofNat 127 : I32)).toInt = ((d.M + 127 : Nat) : Int) :=
    IntV.I32.add_nat (IntV.I32.toInt_ofNat hM31) (IntV.I32.toInt_ofNat h127)
      (by have h := hd.M_le; omega)
  have hNv : (IntV.ofNat d.N + (IntV.ofNat 255 : I32)).toInt = ((d.N + 255 : Nat) : Int) :=
    IntV.I32.add_nat (IntV.I32.toInt_ofNat hN31) (IntV.I32.toInt_ofNat h255)
      (by have h := hd.N_le; omega)
  have hKv : (IntV.ofNat d.K + (IntV.ofNat 63 : I32)).toInt = ((d.K + 63 : Nat) : Int) :=
    IntV.I32.add_nat (IntV.I32.toInt_ofNat hK31) (IntV.I32.toInt_ofNat h63)
      (by have h := hd.K_le; omega)
  have h128v : (IntV.ofNat 128 : I32).toInt = ((128 : Nat) : Int) :=
    IntV.I32.toInt_ofNat (by norm_num)
  rw [body_eq_epi]
  refine safe_idiv1 (E := (launch d).env P) (R := R d) (W := W d i)
    (Q := fun _ m => Q d init i m) (m := init)
    (x := IntV.ofNat d.M + (IntV.ofNat 127 : I32)) (y := (IntV.ofNat 128 : I32)) ?_ ?_ ?_
  · rw [hMv]; push_cast; omega
  · rw [h128v]; norm_num
  · intro r1 h1
    have h1M : r1.item.toInt = ((numPidM d : Nat) : Int) := by
      rw [h1, hMv, h128v, ← Int.natCast_ediv]
      rfl
    refine safe_idiv1 (E := (launch d).env P) (R := R d) (W := W d i)
      (Q := fun _ m => Q d init i m) (m := init)
      (x := IntV.ofNat d.N + (IntV.ofNat 255 : I32)) (y := (IntV.ofNat 256 : I32)) ?_ ?_ ?_
    · rw [hNv]; push_cast; omega
    · rw [h256v]; norm_num
    · intro r2 h2
      have h2N : r2.item.toInt = ((numPidN d : Nat) : Int) := by
        rw [h2, hNv, h256v, ← Int.natCast_ediv]
        rfl
      refine safe_idiv1 (E := (launch d).env P) (R := R d) (W := W d i)
        (Q := fun _ m => Q d init i m) (m := init)
        (x := IntV.ofNat d.K + (IntV.ofNat 63 : I32)) (y := (IntV.ofNat 64 : I32)) ?_ ?_ ?_
      · rw [hKv]; push_cast; omega
      · rw [h64v]; norm_num
      · intro r3 h3
        have h3K : r3.item.toInt = ((steps d.K : Nat) : Int) := by
          rw [h3, hKv, h64v, ← Int.natCast_ediv]
          rfl
        have hnt : (r1.item * r2.item).toInt = ((tiles d : Nat) : Int) := by
          have h := IntV.I32.mul_nat (a := r1.item) (b := r2.item) (x := numPidM d)
            (y := numPidN d) h1M h2N (by
              have h := hd.tiles_le
              rw [tiles_eq_numPid] at h
              omega)
          rw [h, ← tiles_eq_numPid]
        have hrange : tl.RangeOk (tl.program_id ((launch d).ctx i) 0) (r1.item * r2.item)
            (IntV.ofNat d.NUM_SMS : I32) := by
          unfold tl.RangeOk
          rw [hpid, hnt, hstepv]
          rw [hmax]
          refine ⟨by omega, ?_, ?_, ?_⟩
          · have h := hd.tiles_le; omega
          · have h1 := hlt; have h2 := hd.sms.2; omega
          · have h := hd.sms.2; omega
        have htc2 : tl.tripCount ((i : Nat) : Int) (tiles d : Int) (d.NUM_SMS : Int) = trips d i :=
          tripCount_eq hS hile
        have htc : tl.tripCount (tl.program_id ((launch d).ctx i) 0).toInt
            (r1.item * r2.item).toInt (IntV.ofNat d.NUM_SMS : I32).toInt = trips d i := by
          rw [hpid, hnt, hstepv]
          exact htc2
        have hiter : iterW (fun t => d.c.elems (tileIdx d (i + t * d.NUM_SMS)))
            (tl.tripCount (tl.program_id ((launch d).ctx i) 0).toInt
              (r1.item * r2.item).toInt (IntV.ofNat d.NUM_SMS : I32).toInt) = W d i := by
          rw [htc, ← htc2]
          exact iterW_tileIdx_eq hd hlt
        have hmain := safe_forRange (E := (launch d).env P) (R := R d) (m := init)
          (Q := fun _ m => Q d init i m)
          (lo := tl.program_id ((launch d).ctx i) 0) (hi := r1.item * r2.item)
          (step := (IntV.ofNat d.NUM_SMS : I32))
          (init := tl.program_id ((launch d).ctx i) 0 - (IntV.ofNat d.NUM_SMS : I32))
          (body := fun tile_id tile_id_c => do
            let (pid_m, pid_n) ← compute_pid tile_id ((IntV.ofNat 8 : I32) * r2.item) r1.item 8
            let offs_am := pid_m * (IntV.ofNat 128 : I32)
            let offs_bn := pid_n * (IntV.ofNat 256 : I32)
            let accumulator := tl.zeros [128, 256] .f32
            let accumulator ← tl.forRange (IntV.ofInt 0 : I32) r3.item (IntV.ofInt 1 : I32)
                accumulator (fun ki accumulator => do
              let offs_k := ki * (IntV.ofNat 64 : I32)
              let a ← tl.desc_load (bs := [128, 64]) (aDesc d) [offs_am, offs_k]
              let b ← tl.desc_load (bs := [256, 64]) (bDesc d) [offs_bn, offs_k]
              tl.dot { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
                a b.trans accumulator)
            let tile_id_c := tile_id_c + (IntV.ofNat d.NUM_SMS : I32)
            epiProg d tile_id_c ((IntV.ofNat 8 : I32) * r2.item) r1.item accumulator)
          (k := fun _ => pure ())
          hrange
          (fun t st m₁ => st.toInt = ((i : Nat) : Int) + (t : Int) * (d.NUM_SMS : Int)
              - (d.NUM_SMS : Int) ∧
            d.a.Valid m₁ ∧ d.b.Valid m₁ ∧ d.c.Valid m₁ ∧
            (∀ x ∈ d.a.elems (aIdx d), m₁ x = init x) ∧
            (∀ x ∈ d.b.elems (bIdx d), m₁ x = init x) ∧
            (∀ u, u < t → Qfact d init (i + u * d.NUM_SMS) m₁))
          (fun t => d.c.elems (tileIdx d (i + t * d.NUM_SMS))) (Wk := ∅) ?_ ?_ ?_ ?_ ?_
        simpa only [Set.union_empty, hiter] using hmain
        · intro t u htu hu
          have hu' : u < tl.tripCount ((i : Nat) : Int) (tiles d : Int) (d.NUM_SMS : Int) := by
            rw [htc2, ← htc]
            exact hu
          exact Wt_disjoint hd hlt t u htu hu'
        · simp
        · refine ⟨?_, hva0, hvb0, hvc0, ?_, ?_, ?_⟩
          · have hsub := IntV.toInt_sub (w := 32) (a := tl.program_id ((launch d).ctx i) 0)
              (b := (IntV.ofNat d.NUM_SMS : I32)) hpid hstepv ?_
            · rw [hsub]; push_cast; ring
            · constructor
              · have h1 : (0 : Int) ≤ ((i : Nat) : Int) := Int.natCast_nonneg _
                have h2 : ((d.NUM_SMS : Nat) : Int) ≤ 2147483647 := by
                  have h := hd.sms.2
                  omega
                omega
              · have h1 : ((i : Nat) : Int) < ((d.NUM_SMS : Nat) : Int) := by
                  exact_mod_cast hiS
                omega
          · intro x hx; rfl
          · intro x hx; rfl
          · intro u hu; omega
        · intro t ht st m₁ hI
          obtain ⟨hstI, hvaM, hvbM, hvcM, hRAM, hRBM, hQf⟩ := hI
          have ht' : t < trips d i := by rw [← htc]; exact ht
          have hTlt : i + t * d.NUM_SMS < tiles d := trip_lt hS hile ht'
          have hT31 : i + t * d.NUM_SMS < 2 ^ 31 := by
            have h := hd.tiles_le; omega
          have hTc : ((i : Nat) : Int) + (t : Int) * (d.NUM_SMS : Int)
              = ((i + t * d.NUM_SMS : Nat) : Int) := by push_cast; ring
          have htid : (IntV.ofInt ((tl.program_id ((launch d).ctx i) 0).toInt
              + (t : Int) * (IntV.ofNat d.NUM_SMS : I32).toInt) : I32).toInt
              = ((i + t * d.NUM_SMS : Nat) : Int) := by
            rw [hpid, hstepv]
            rw [show ((i : Nat) : Int) + (t : Int) * (d.NUM_SMS : Int)
                = ((i + t * d.NUM_SMS : Nat) : Int) from hTc]
            exact IntV.I32.toInt_ofInt ⟨by omega, by exact_mod_cast hT31⟩
          have hst' : (st + (IntV.ofNat d.NUM_SMS : I32)).toInt
              = ((i + t * d.NUM_SMS : Nat) : Int) := by
            have hadd := IntV.toInt_add (w := 32) (a := st) (b := (IntV.ofNat d.NUM_SMS : I32))
              hstI hstepv ?_
            · rw [hadd]; push_cast; ring
            · constructor
              · rw [show ((i : Nat) : Int) + (t : Int) * (d.NUM_SMS : Int) - (d.NUM_SMS : Int)
                    + (d.NUM_SMS : Int) = ((i + t * d.NUM_SMS : Nat) : Int) from by
                  rw [← hTc]; ring]
                omega
              · rw [show ((i : Nat) : Int) + (t : Int) * (d.NUM_SMS : Int) - (d.NUM_SMS : Int)
                    + (d.NUM_SMS : Int) = ((i + t * d.NUM_SMS : Nat) : Int) from by
                  rw [← hTc]; ring]
                exact_mod_cast hT31
          have hG8 : ((IntV.ofNat 8 : I32) * r2.item).toInt = ((8 * numPidN d : Nat) : Int) := by
            have h8 : (IntV.ofNat 8 : I32).toInt = ((8 : Nat) : Int) :=
              IntV.I32.toInt_ofNat (by norm_num)
            have h := IntV.I32.mul_nat (a := (IntV.ofNat 8 : I32)) (b := r2.item)
              (x := 8) (y := numPidN d) h8 h2N (by
                have h1 := mul256_numPidN_le d
                have h2 := hd.N_le
                omega)
            rw [h]
          refine (safe_one_tile (d := d) (E := (launch d).env P) (init := init)
            (T := i + t * d.NUM_SMS) (m := m₁)
            (tid := IntV.ofInt ((tl.program_id ((launch d).ctx i) 0).toInt
              + (t : Int) * (IntV.ofNat d.NUM_SMS : I32).toInt))
            (num_pid_in_group := (IntV.ofNat 8 : I32) * r2.item)
            (num_pid_m := r1.item) (k_tiles := r3.item) (st := st)
            hd hacc htot hTlt htid hG8 h1M h3K hst' hvaM hvbM hvcM hRAM hRBM).mono ?_
            (Set.Subset.refl _)
          · intro st' m' hq
            obtain ⟨hstT, hva', hvb', hvc', hqf, hfr, hpres⟩ := hq
            refine ⟨?_, hva', hvb', hvc', ?_, ?_, ?_⟩
            · rw [hstT]; push_cast; ring
            · intro x hx
              have hfx : m' x = m₁ x := hfr x fun hximg =>
                (Set.disjoint_left.mp hd.c_a
                  (Buf.elems_mono d.c (tileIdx_subset_cIdx d (i + t * d.NUM_SMS)) hximg)) hx
              rw [hfx]
              exact hRAM x hx
            · intro x hx
              have hfx : m' x = m₁ x := hfr x fun hximg =>
                (Set.disjoint_left.mp hd.c_b
                  (Buf.elems_mono d.c (tileIdx_subset_cIdx d (i + t * d.NUM_SMS)) hximg)) hx
              rw [hfx]
              exact hRBM x hx
            · intro u hu
              by_cases hu' : u < t
              · have huT : i + u * d.NUM_SMS < tiles d :=
                  trip_lt hS hile (trips_mono hS hu' ht')
                exact Qfact_transport hd hTlt huT (Ne.symm (tile_id_ne hS hu'))
                  (img := tileIdx d (i + t * d.NUM_SMS)) (m := m₁) (m' := m')
                  (Set.Subset.refl _) hfr (hQf u hu')
              · have hut : u = t := by omega
                subst hut
                exact hqf
        · intro st' m₁ hI
          obtain ⟨-, -, -, -, -, -, hQf⟩ := hI
          refine .ret ?_
          rw [Q_iff_Qfact]
          intro T hT
          rw [progTiles_eq hS hiS] at hT
          obtain ⟨t, ht, rfl⟩ := hT
          exact hQf t (by rw [htc]; exact ht)

/-!

-/


/-- The launch of every covered instance and memory is correct. -/
theorem launch_correct {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hPre : Pre d init) :
    Correct ((launch d).env P) (model d) (launch d) init (Post d init) := by
  refine correct_of_safe (fun _ => R d) (fun i => W d i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · exact safe_body hd (hP.dotF16 .tf32) (hP.dotTotal _) hPre
  · intro i j hij
    exact Disjoint.union_right (W_disjoint_reads hd i) (W_disjoint_W hd fun h => hij (Fin.ext h))
  · intro i
    exact reads_disjoint_W hd i
  · intro fin hloc hfr
    exact post_of_locs hd hloc fun x hx => hfr x fun i => hx (i : Nat)

end Claims.TmaPersistentMatmul
