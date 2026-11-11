import Claim
import TVA.Proof

/-!
# Defs

The footprints of a program (`R`, `W`), its post-condition `Q`, the grid arithmetic, and the facts about loads and tiles that every later module uses.
-/

namespace Claims.SsdChunkState

open TVA

set_option maxHeartbeats 4000000

/-!
# `ssd_chunk_state_bf16`: the launch, the footprints and the assembly of `Post`

This module fixes the objects the proof works with: the launch's validity, the map from a
program index to the tile it owns, the read and write footprints, and the per-program property
`Q` that the programs' results give `Post`.
-/

set_option maxHeartbeats 800000


open TVA

/-! ## The launch -/

/-- The number of programs of the launch: `gx · nchunks · nheads`. -/
theorem launch_size (d : Inst) : (launch d).size = gx d * nc d * d.nheads := rfl

/-- `program_id(0)` of program `i`: its tile (`i mod gx`). -/
theorem launch_pid0 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = (i : Nat) % gx d := rfl

/-- `program_id(1)` of program `i`: its chunk. -/
theorem launch_pid1 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 1 = (i : Nat) / gx d % nc d := rfl

/-- `program_id(2)` of program `i`: its head. -/
theorem launch_pid2 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 2 = (i : Nat) / (gx d * nc d) := rfl

/-- The chunk size is positive. -/
theorem chunk_pos {d : Inst} (hd : d.ok) : 0 < d.chunk := by
  obtain ⟨k, hk⟩ := hd.chunk_pow2
  rw [hk]
  exact Nat.two_pow_pos k

/-- There is at least one chunk. -/
theorem nc_pos {d : Inst} (hd : d.ok) : 0 < nc d := by
  have hc := chunk_pos hd
  have hs := hd.seqlen_pos
  unfold nc
  apply Nat.div_pos
  · omega
  · omega

/-- The tile count `gx` is below `2^31`. -/
theorem gx_lt {d : Inst} (hd : d.ok) : gx d < 2 ^ 31 := by
  have h1 := hd.grid_lt
  have hnc := nc_pos hd
  have hn := hd.heads_pos
  have hmul : 0 < nc d * d.nheads := Nat.mul_pos hnc hn
  have hle : gx d ≤ gx d * nc d * d.nheads := by
    have h2 : 1 ≤ nc d * d.nheads := hmul
    calc gx d = gx d * 1 := (Nat.mul_one _).symm
      _ ≤ gx d * (nc d * d.nheads) := Nat.mul_le_mul_left _ h2
      _ = gx d * nc d * d.nheads := by ring
  omega

/-- The launch is valid: the grid is within CUDA's limits and the block shapes are Triton-ok. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hy := hd.grid_y
  have hz := hd.grid_z
  have hg := hd.grid_lt
  have hx := gx_lt hd
  refine ⟨hx, hy, hz, hg, rfl, rfl, rfl, ?_⟩
  intro s hs
  simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl
  · refine ⟨?_, by norm_num [List.prod, BM]⟩
    intro k hk
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨6, by norm_num [BM]⟩
    cases hk
  · refine ⟨?_, by norm_num [List.prod, BN]⟩
    intro k hk
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨7, by norm_num [BN]⟩
    cases hk
  · refine ⟨?_, by norm_num [List.prod, BK]⟩
    intro k hk
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨6, by norm_num [BK]⟩
    cases hk
  · refine ⟨?_, by norm_num [List.prod, BM, BK]⟩
    intro k hk
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨6, by norm_num [BM]⟩
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨6, by norm_num [BK]⟩
    cases hk
  · refine ⟨?_, by norm_num [List.prod, BK, BN]⟩
    intro k hk
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨6, by norm_num [BK]⟩
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨7, by norm_num [BN]⟩
    cases hk
  · refine ⟨?_, by norm_num [List.prod, BM, BN]⟩
    intro k hk
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨6, by norm_num [BM]⟩
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ⟨7, by norm_num [BN]⟩
    cases hk

/-! ## The tile of a program -/

/-- The chunk `pid_c` of program `i`. -/
def chunkOf (d : Inst) (i : Nat) : Nat := i / gx d % nc d

/-- The head `pid_h` of program `i`. -/
def headOf (d : Inst) (i : Nat) : Nat := i / (gx d * nc d)

/-- `pid_m · num_pid_n + pid_n`, the part of the index that selects the tile. -/
def qOf (d : Inst) (i : Nat) : Nat := i % gx d

/-- `cdiv(dstate, 128)`: the number of tiles along `n`. -/
def npmOf (d : Inst) : Nat := (d.dstate + BN - 1) / BN

/-- The row block `pid_m` of program `i`. -/
def pmOf (d : Inst) (i : Nat) : Nat := qOf d i / npmOf d

/-- The column block `pid_n` of program `i`. -/
def pnOf (d : Inst) (i : Nat) : Nat := qOf d i % npmOf d

/-- The rows of the tile of program `i`: `p < headdim` in `[pid_m·64, (pid_m+1)·64)`. -/
def pSet (d : Inst) (i : Nat) : Set Nat :=
  {p | pmOf d i * BM ≤ p ∧ p < (pmOf d i + 1) * BM ∧ p < d.headdim}

/-- The columns of the tile of program `i`: `n < dstate` in `[pid_n·128, (pid_n+1)·128)`. -/
def nSet (d : Inst) (i : Nat) : Set Nat :=
  {n | pnOf d i * BN ≤ n ∧ n < (pnOf d i + 1) * BN ∧ n < d.dstate}

/-- The elements of `states` the tile of program `i` holds. -/
def stSet (d : Inst) (i : Nat) : Set Nat :=
  {j | ∃ p ∈ pSet d i, ∃ n ∈ nSet d i, j = stIdx d (chunkOf d i) (headOf d i) p n}

/-- The program that owns the element `(c, h, p, n)`. -/
def progOf (d : Inst) (c h p n : Nat) : Nat :=
  (p / BM * npmOf d + n / BN) + gx d * (c + nc d * h)

/-- The model's `tl.dot` configuration (line 204). -/
def dotCfg : DotCfg := { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-! ## Footprints -/

/-- The read footprint: every byte outside `states`. The programs only load from the five input
buffers, which `disj` separates from `states`, so this holds them all. -/
def R (d : Inst) (_ : Mem) (_ : Fin (launch d).size) : Set Int := {x | x ∉ d.states.bytes}

/-- The write footprint: the tile of `states` the program holds. -/
def W (d : Inst) (_ : Mem) (i : Fin (launch d).size) : Set Int := d.states.elems (stSet d i)

/-- What program `i` leaves in its own memory: its tile of `states` holds the chunk state. -/
noncomputable def Q (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop :=
  ∀ p n : Nat, p ∈ pSet d i → n ∈ nSet d i → ∃ v : F32,
    m.read .f32 (d.states.ptr (stIdx d (chunkOf d i) (headOf d i) p n)).addr = some v ∧
      (Vals init d → v.isFinite = true ∧
        |v.toReal - sRef init d (chunkOf d i) (headOf d i) p n| ≤
          tol init d (chunkOf d i) (headOf d i) p n)


/-! ## Idx -/

/-!
# `ssd_chunk_state_bf16`: the arithmetic of the grid index

A program index `i < gx·nc·nheads` decodes as `pid0 + gx·(pid1 + nc·pid2)`; the tile it owns is
the set of `(p, n)` with `p / 64 = pid_m` and `n / 128 = pid_n`. This module proves that the
`states` index of every element of a tile is below `sSb`, that different programs own disjoint
tiles, and that the program owning a given element is `progOf`.
-/

set_option maxHeartbeats 800000


open TVA

/-! ## Positivity -/

/-- The size of one head's state: `headdim · dstate`. -/
theorem sSh_eq (d : Inst) : sSh d = d.headdim * d.dstate := rfl

/-- The size of one chunk's states: `nheads · headdim · dstate`. -/
theorem sSc_eq (d : Inst) : sSc d = d.nheads * (d.headdim * d.dstate) := rfl

/-- The size of `states`: `nchunks · nheads · headdim · dstate`. -/
theorem sSb_eq (d : Inst) : sSb d = nc d * sSc d := rfl

/-- One chunk's states are not empty. -/
theorem sSc_pos {d : Inst} (hd : d.ok) : 0 < sSc d := by
  rw [sSc_eq]
  exact Nat.mul_pos hd.heads_pos (Nat.mul_pos hd.headdim_pos hd.dstate_pos)

/-- There is at least one `dstate` tile. -/
theorem npmOf_pos {d : Inst} (hd : d.ok) : 0 < npmOf d := by
  have hp := hd.dstate_pos
  unfold npmOf
  apply Nat.div_pos
  · simp only [BN]; omega
  · simp only [BN]; omega

/-- `gx` written out: `cdiv(headdim, 64) · cdiv(dstate, 128)`. -/
theorem gx_eq (d : Inst) : gx d = (d.headdim + BM - 1) / BM * npmOf d := rfl

/-- `gx` is positive. -/
theorem gx_pos {d : Inst} (hd : d.ok) : 0 < gx d := by
  have hp := hd.headdim_pos
  rw [gx_eq]
  apply Nat.mul_pos
  · apply Nat.div_pos
    · simp only [BM]; omega
    · simp only [BM]; omega
  · exact npmOf_pos hd

/-! ## The `states` index -/

/-- `p·dstate + n` is below `headdim·dstate`. -/
theorem row_lt {d : Inst} {p n : Nat} (hp : p < d.headdim) (hn : n < d.dstate) :
    p * d.dstate + n < sSh d := by
  have h1 : p * d.dstate + n < p * d.dstate + d.dstate := by omega
  have h2 : p * d.dstate + d.dstate = (p + 1) * d.dstate := by ring
  have h3 : (p + 1) * d.dstate ≤ d.headdim * d.dstate :=
    Nat.mul_le_mul_right _ (by omega)
  rw [sSh_eq]
  omega

/-- `h·(headdim·dstate) + (p·dstate + n)` is below `nheads·headdim·dstate`. -/
theorem inner_lt {d : Inst} {h p n : Nat}
    (hh : h < d.nheads) (hp : p < d.headdim) (hn : n < d.dstate) :
    h * sSh d + (p * d.dstate + n) < sSc d := by
  have hrow := row_lt hp hn
  have h2 : h * sSh d + (p * d.dstate + n) < (h + 1) * sSh d := by
    have e : (h + 1) * sSh d = h * sSh d + sSh d := by ring
    omega
  have h3 : (h + 1) * sSh d ≤ d.nheads * sSh d := Nat.mul_le_mul_right _ (by omega)
  have h4 : d.nheads * sSh d = sSc d := by rw [sSh_eq, sSc_eq]
  omega

/-- The `states` index of an element of a tile is below `sSb`. -/
theorem stIdx_lt {d : Inst} {c h p n : Nat}
    (hc : c < nc d) (hh : h < d.nheads) (hp : p < d.headdim) (hn : n < d.dstate) :
    stIdx d c h p n < sSb d := by
  have hin := inner_lt hh hp hn
  have e : stIdx d c h p n = c * sSc d + (h * sSh d + (p * d.dstate + n)) := by
    unfold stIdx sSh sSc
    ring
  rw [e]
  have h2 : c * sSc d + (h * sSh d + (p * d.dstate + n)) < (c + 1) * sSc d := by
    have ee : (c + 1) * sSc d = c * sSc d + sSc d := by ring
    omega
  have h3 : (c + 1) * sSc d ≤ nc d * sSc d := Nat.mul_le_mul_right _ (by omega)
  have h4 : nc d * sSc d = sSb d := (sSb_eq d).symm
  omega

/-- The four components of a `states` index are determined by it. -/
theorem stIdx_inj {d : Inst} (_hd : d.ok) {c c' h h' p p' n n' : Nat}
    (_hc : c < nc d) (hc' : c' < nc d) (hh : h < d.nheads) (hh' : h' < d.nheads)
    (hp : p < d.headdim) (hp' : p' < d.headdim) (hn : n < d.dstate) (hn' : n' < d.dstate)
    (he : stIdx d c h p n = stIdx d c' h' p' n') :
    c = c' ∧ h = h' ∧ p = p' ∧ n = n' := by
  have hin := inner_lt hh hp hn
  have hin' := inner_lt hh' hp' hn'
  have hrow := row_lt hp hn
  have hrow' := row_lt hp' hn'
  have e : ∀ (c h p n : Nat),
      stIdx d c h p n = c * sSc d + (h * sSh d + (p * d.dstate + n)) := by
    intro c h p n
    unfold stIdx sSh sSc
    ring
  rw [e, e] at he
  have h1 : (c * sSc d + (h * sSh d + (p * d.dstate + n))) / sSc d = c := by
    apply Nat.div_eq_of_lt_le (Nat.le_add_right _ _)
    have ee : (c + 1) * sSc d = c * sSc d + sSc d := by ring
    omega
  have h2 : (c' * sSc d + (h' * sSh d + (p' * d.dstate + n'))) / sSc d = c' := by
    apply Nat.div_eq_of_lt_le (Nat.le_add_right _ _)
    have ee : (c' + 1) * sSc d = c' * sSc d + sSc d := by ring
    omega
  have hcc : c = c' := by rw [← h1, he, h2]
  subst hcc
  have hRR : h * sSh d + (p * d.dstate + n) = h' * sSh d + (p' * d.dstate + n') :=
    Nat.add_left_cancel he
  have h3 : (h * sSh d + (p * d.dstate + n)) / sSh d = h := by
    apply Nat.div_eq_of_lt_le (Nat.le_add_right _ _)
    have ee : (h + 1) * sSh d = h * sSh d + sSh d := by ring
    omega
  have h4 : (h' * sSh d + (p' * d.dstate + n')) / sSh d = h' := by
    apply Nat.div_eq_of_lt_le (Nat.le_add_right _ _)
    have ee : (h' + 1) * sSh d = h' * sSh d + sSh d := by ring
    omega
  have hhh : h = h' := by rw [← h3, hRR, h4]
  subst hhh
  have hnn : p * d.dstate + n = p' * d.dstate + n' := Nat.add_left_cancel hRR
  have h5 : (p * d.dstate + n) / d.dstate = p := by
    apply Nat.div_eq_of_lt_le (Nat.le_add_right _ _)
    have ee : (p + 1) * d.dstate = p * d.dstate + d.dstate := by ring
    omega
  have h6 : (p' * d.dstate + n') / d.dstate = p' := by
    apply Nat.div_eq_of_lt_le (Nat.le_add_right _ _)
    have ee : (p' + 1) * d.dstate = p' * d.dstate + d.dstate := by ring
    omega
  have hpp : p = p' := by rw [← h5, hnn, h6]
  subst hpp
  exact ⟨rfl, rfl, rfl, by omega⟩

/-! ## Decoding a program index -/

/-- Every program's chunk is below `nchunks`. -/
theorem chunkOf_lt {d : Inst} (hd : d.ok) (i : Nat) : chunkOf d i < nc d := by
  unfold chunkOf
  exact Nat.mod_lt _ (nc_pos hd)

/-- Every program's head is below `nheads`. -/
theorem headOf_lt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < (launch d).size) :
    headOf d i < d.nheads := by
  have hpos : 0 < gx d * nc d := Nat.mul_pos (gx_pos hd) (nc_pos hd)
  rw [launch_size] at hi
  unfold headOf
  rw [Nat.div_lt_iff_lt_mul hpos]
  calc i < gx d * nc d * d.nheads := hi
    _ = d.nheads * (gx d * nc d) := by ring

/-- Program `i` is rebuilt from its tile, chunk and head. -/
theorem i_eq_parts (d : Inst) (i : Nat) :
    i = qOf d i + gx d * (chunkOf d i + nc d * headOf d i) := by
  have hdiv : (i / gx d) / nc d = headOf d i := by
    unfold headOf
    exact Nat.div_div_eq_div_mul _ _ _
  have h2 : i / gx d = chunkOf d i + nc d * headOf d i := by
    have h := Nat.mod_add_div (i / gx d) (nc d)
    rw [hdiv] at h
    unfold chunkOf
    exact h.symm
  have h1 := Nat.mod_add_div i (gx d)
  rw [h2] at h1
  exact h1.symm

/-! ## The tile of a program -/

/-- A row in program `i`'s rows lies in its `headdim` tile. -/
theorem pmOf_of_mem_pSet {d : Inst} {i p : Nat} (hp : p ∈ pSet d i) :
    pmOf d i = p / BM :=
  (Nat.div_eq_of_lt_le hp.1 hp.2.1).symm

/-- A column in program `i`'s columns lies in its `dstate` tile. -/
theorem pnOf_of_mem_nSet {d : Inst} {i n : Nat} (hn : n ∈ nSet d i) :
    pnOf d i = n / BN :=
  (Nat.div_eq_of_lt_le hn.1 hn.2.1).symm

/-- The tile index of a program in terms of its row tile and column tile. -/
theorem qOf_eq (d : Inst) (i : Nat) : qOf d i = pmOf d i * npmOf d + pnOf d i := by
  have h := Nat.div_add_mod (qOf d i) (npmOf d)
  have h' : (qOf d i / npmOf d) * npmOf d + qOf d i % npmOf d = qOf d i := by
    rw [Nat.mul_comm]
    exact h
  unfold pmOf pnOf
  exact h'.symm

/-- A program index is determined by its tile and its coordinates in the tile. -/
theorem eq_progOf {d : Inst} {i p n : Nat} (hp : p ∈ pSet d i) (hn : n ∈ nSet d i) :
    i = progOf d (chunkOf d i) (headOf d i) p n := by
  have hpm := pmOf_of_mem_pSet hp
  have hpn := pnOf_of_mem_nSet hn
  have hq := qOf_eq d i
  have hparts := i_eq_parts d i
  nth_rewrite 1 [hparts]
  unfold progOf
  rw [hq, hpm, hpn]

/-! ## Every element of a tile is inside `states` -/

/-- A program's tile of `states` lies inside `states`. -/
theorem stSet_sub {d : Inst} (hd : d.ok) {i : Nat} (hi : i < (launch d).size) :
    stSet d i ⊆ {j | j < sSb d} := by
  intro j hj
  obtain ⟨p, hp, n, hn, rfl⟩ := hj
  exact stIdx_lt (chunkOf_lt hd i) (headOf_lt hd hi) hp.2.2 hn.2.2

/-! ## Which program owns an element -/

/-- The tile part of the index of `progOf d c h p n` is the tile of `(p, n)`. -/
theorem qOf_progOf {d : Inst} (_hd : d.ok) {c h p n : Nat}
    (hp : p < d.headdim) (hn : n < d.dstate) :
    qOf d (progOf d c h p n) = p / BM * npmOf d + n / BN := by
  have hM : p / BM < (d.headdim + BM - 1) / BM := by
    simp only [BM]
    omega
  have hN : n / BN < npmOf d := by
    unfold npmOf
    simp only [BN]
    omega
  have hlt : p / BM * npmOf d + n / BN < gx d := by
    rw [gx_eq]
    have h1 : p / BM * npmOf d + n / BN < (p / BM + 1) * npmOf d := by
      have e : (p / BM + 1) * npmOf d = p / BM * npmOf d + npmOf d := by ring
      omega
    have h2 : (p / BM + 1) * npmOf d ≤ ((d.headdim + BM - 1) / BM) * npmOf d :=
      Nat.mul_le_mul_right _ (by omega)
    omega
  unfold qOf progOf
  rw [Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt hlt

/-- The program that owns `(c, h, p, n)` has row tile `p / 64`. -/
theorem pmOf_progOf {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hp : p < d.headdim) (hn : n < d.dstate) :
    pmOf d (progOf d c h p n) = p / BM := by
  have hN : n / BN < npmOf d := by
    unfold npmOf
    simp only [BN]
    omega
  have hq := qOf_progOf hd (c := c) (h := h) hp hn
  have hpos := npmOf_pos hd
  unfold pmOf
  rw [hq]
  have hsplit : p / BM * npmOf d + n / BN = npmOf d * (p / BM) + n / BN := by ring
  rw [hsplit, Nat.mul_add_div hpos, Nat.div_eq_of_lt hN]
  omega

/-- The program that owns `(c, h, p, n)` has column tile `n / 128`. -/
theorem pnOf_progOf {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hp : p < d.headdim) (hn : n < d.dstate) :
    pnOf d (progOf d c h p n) = n / BN := by
  have hq := qOf_progOf hd (c := c) (h := h) hp hn
  have hN : n / BN < npmOf d := by
    unfold npmOf
    simp only [BN]
    omega
  unfold pnOf
  rw [hq]
  have hsplit : p / BM * npmOf d + n / BN = npmOf d * (p / BM) + n / BN := by ring
  rw [hsplit, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hN]

/-- The program that owns `(c, h, p, n)` has chunk `c`. -/
theorem chunkOf_progOf {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hp : p < d.headdim) (hn : n < d.dstate) (hc : c < nc d) :
    chunkOf d (progOf d c h p n) = c := by
  have hN : n / BN < npmOf d := by
    unfold npmOf
    simp only [BN]
    omega
  have hlt : p / BM * npmOf d + n / BN < gx d := by
    have hq := qOf_progOf hd (c := c) (h := h) hp hn
    unfold qOf at hq
    rw [← hq]
    exact Nat.mod_lt _ (gx_pos hd)
  have hgi : progOf d c h p n / gx d = c + nc d * h := by
    unfold progOf
    rw [Nat.add_mul_div_left _ _ (gx_pos hd)]
    rw [Nat.div_eq_of_lt hlt]
    omega
  unfold chunkOf
  rw [hgi, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hc]

/-- The program that owns `(c, h, p, n)` has head `h`. -/
theorem headOf_progOf {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hp : p < d.headdim) (hn : n < d.dstate) (hc : c < nc d) :
    headOf d (progOf d c h p n) = h := by
  have hlt : p / BM * npmOf d + n / BN < gx d := by
    have hq := qOf_progOf hd (c := c) (h := h) hp hn
    unfold qOf at hq
    rw [← hq]
    exact Nat.mod_lt _ (gx_pos hd)
  have hpos : 0 < gx d * nc d := Nat.mul_pos (gx_pos hd) (nc_pos hd)
  have he : progOf d c h p n
      = (p / BM * npmOf d + n / BN + gx d * c) + (gx d * nc d) * h := by
    unfold progOf
    ring
  unfold headOf
  rw [he, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt (by
    have h1 : p / BM * npmOf d + n / BN + gx d * c < gx d + gx d * c := by omega
    have h2 : gx d + gx d * c = gx d * (c + 1) := by ring
    have h3 : gx d * (c + 1) ≤ gx d * nc d := Nat.mul_le_mul_left _ (by omega)
    omega)]
  omega

/-- Row `p` is in the rows of the program that owns `(c, h, p, n)`. -/
theorem mem_pSet_progOf {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hp : p < d.headdim) (hn : n < d.dstate) :
    p ∈ pSet d (progOf d c h p n) := by
  have hpm := pmOf_progOf hd (c := c) (h := h) (p := p) (n := n) hp hn
  refine ⟨?_, ?_, hp⟩
  · rw [hpm]
    exact Nat.div_mul_le_self p BM
  · rw [hpm]
    have h1 : p = BM * (p / BM) + p % BM := (Nat.div_add_mod p BM).symm
    have h2 : p % BM < BM := Nat.mod_lt _ (by simp only [BM]; omega)
    have h3 : (p / BM + 1) * BM = BM * (p / BM) + BM := by ring
    omega

/-- Column `n` is in the columns of the program that owns `(c, h, p, n)`. -/
theorem mem_nSet_progOf {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hp : p < d.headdim) (hn : n < d.dstate) :
    n ∈ nSet d (progOf d c h p n) := by
  have hpn := pnOf_progOf hd (c := c) (h := h) (p := p) (n := n) hp hn
  refine ⟨?_, ?_, hn⟩
  · rw [hpn]
    exact Nat.div_mul_le_self n BN
  · rw [hpn]
    have h1 : n = BN * (n / BN) + n % BN := (Nat.div_add_mod n BN).symm
    have h2 : n % BN < BN := Nat.mod_lt _ (by simp only [BN]; omega)
    have h3 : (n / BN + 1) * BN = BN * (n / BN) + BN := by ring
    omega

/-- `progOf` is a program index. -/
theorem progOf_lt {d : Inst} (hd : d.ok) {c h p n : Nat}
    (hc : c < nc d) (hh : h < d.nheads) (hp : p < d.headdim) (hn : n < d.dstate) :
    progOf d c h p n < (launch d).size := by
  have hlt : p / BM * npmOf d + n / BN < gx d := by
    have hq := qOf_progOf hd (c := c) (h := h) hp hn
    unfold qOf at hq
    rw [← hq]
    exact Nat.mod_lt _ (gx_pos hd)
  have ha : c + nc d * h < nc d * d.nheads := by
    calc c + nc d * h < nc d + nc d * h := by omega
      _ = nc d * (h + 1) := by ring
      _ ≤ nc d * d.nheads := Nat.mul_le_mul_left _ (by omega)
  rw [launch_size]
  unfold progOf
  have h1 : p / BM * npmOf d + n / BN + gx d * (c + nc d * h)
      < gx d + gx d * (c + nc d * h) := by omega
  have h2 : gx d + gx d * (c + nc d * h) = gx d * (c + nc d * h + 1) := by ring
  have h3 : gx d * (c + nc d * h + 1) ≤ gx d * (nc d * d.nheads) :=
    Nat.mul_le_mul_left _ (by omega)
  have h4 : gx d * (nc d * d.nheads) = gx d * nc d * d.nheads := by ring
  omega

/-! ## Different programs own disjoint tiles -/

/-- The tiles of two distinct programs are disjoint. -/
theorem stSet_disjoint {d : Inst} (hd : d.ok) {i i' : Nat} (hii : i ≠ i')
    (hi : i < (launch d).size) (hi' : i' < (launch d).size) :
    Disjoint (stSet d i) (stSet d i') := by
  rw [Set.disjoint_left]
  intro j hj hj'
  obtain ⟨p, hp, n, hn, rfl⟩ := hj
  obtain ⟨p', hp', n', hn', he⟩ := hj'
  obtain ⟨hce, hhe, hpe, hne⟩ := stIdx_inj hd (chunkOf_lt hd i) (chunkOf_lt hd i')
    (headOf_lt hd hi) (headOf_lt hd hi') hp.2.2 hp'.2.2 hn.2.2 hn'.2.2 he
  apply hii
  rw [eq_progOf hp hn, eq_progOf hp' hn', hce, hhe, hpe, hne]


/-! ## Len -/

/-!
# `ssd_chunk_state_bf16`: the indices are inside the buffers

Every element index the kernel computes lies inside its tensor, and the chunk's token range is
the loop's trip range.
-/

set_option maxHeartbeats 800000


open TVA

/-! ## Arithmetic of the number of chunks -/

/-- `min` and truncated subtraction. -/
theorem min_sub (a s b : Nat) (hba : b ≤ a) (hbs : b ≤ s) :
    min a s - b = min (a - b) (s - b) := by
  by_cases h : a ≤ s
  · rw [Nat.min_eq_left h, Nat.min_eq_left (by omega)]
  · rw [Nat.min_eq_right (by omega), Nat.min_eq_right (by omega)]

/-- An `x` index inside the tensor. -/
theorem xIdx_lt {d : Inst} (hd : d.ok) {t h p : Nat}
    (ht : t < d.seqlen) (hh : h < d.nheads) (hp : p < d.headdim) :
    xIdx d t h p < d.x.len := by
  have h1 : t * d.sXs ≤ (d.seqlen - 1) * d.sXs := Nat.mul_le_mul_right _ (by omega)
  have h2 : h * d.sXh ≤ (d.nheads - 1) * d.sXh := Nat.mul_le_mul_right _ (by omega)
  have h3 : p * d.sXp ≤ (d.headdim - 1) * d.sXp := Nat.mul_le_mul_right _ (by omega)
  have hle : xIdx d t h p ≤ xIdx d (d.seqlen - 1) (d.nheads - 1) (d.headdim - 1) := by
    unfold xIdx
    omega
  exact lt_of_le_of_lt hle hd.x_len

/-- A `B` index inside the tensor. -/
theorem bIdx_lt {d : Inst} (hd : d.ok) {t g n : Nat}
    (ht : t < d.seqlen) (hg : g < d.ngroups) (hn : n < d.dstate) :
    bIdx d t g n < d.b.len := by
  have h1 : t * d.sBs ≤ (d.seqlen - 1) * d.sBs := Nat.mul_le_mul_right _ (by omega)
  have h2 : g * d.sBg ≤ (d.ngroups - 1) * d.sBg := Nat.mul_le_mul_right _ (by omega)
  have h3 : n * d.sBn ≤ (d.dstate - 1) * d.sBn := Nat.mul_le_mul_right _ (by omega)
  have hle : bIdx d t g n ≤ bIdx d (d.seqlen - 1) (d.ngroups - 1) (d.dstate - 1) := by
    unfold bIdx
    omega
  exact lt_of_le_of_lt hle hd.b_len

/-- A `dt`/`dA_cumsum` index inside the per-batch block. -/
theorem dIdx_lt {d : Inst} {h c i : Nat}
    (hh : h < d.nheads) (hc : c < nc d) (hi : i < d.chunk) :
    dIdx d h c i < sDb d := by
  have hck : 0 < d.chunk := by omega
  have hnc : 0 < nc d := by omega
  have hK : 0 < nc d * d.chunk := Nat.mul_pos hnc hck
  have h1 : c * d.chunk + d.chunk ≤ nc d * d.chunk := by
    have e : (c + 1) * d.chunk = c * d.chunk + d.chunk := by ring
    simpa [e] using Nat.mul_le_mul_right d.chunk (by omega : c + 1 ≤ nc d)
  have h2 : h * (nc d * d.chunk) + nc d * d.chunk ≤ d.nheads * (nc d * d.chunk) := by
    have e : (h + 1) * (nc d * d.chunk) = h * (nc d * d.chunk) + nc d * d.chunk := by ring
    simpa [e] using Nat.mul_le_mul_right (nc d * d.chunk) (by omega : h + 1 ≤ d.nheads)
  unfold dIdx sDb
  omega

/-- A `seq_idx` index inside the tensor. -/
theorem seqIdx_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.seqlen) :
    t < d.seqIdx.len := by
  exact lt_of_lt_of_le ht hd.seq_len

/-- The chunk starts before the sequence ends. -/
theorem c_chunk_lt {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    c * d.chunk < d.seqlen := by
  have hck := chunk_pos hd
  have h2 : nc d * d.chunk ≤ d.seqlen + d.chunk - 1 := by
    unfold nc
    exact Nat.div_mul_le_self _ _
  have h3 : c * d.chunk + d.chunk ≤ nc d * d.chunk := by
    have e : (c + 1) * d.chunk = c * d.chunk + d.chunk := by ring
    simpa [e] using Nat.mul_le_mul_right d.chunk (by omega : c + 1 ≤ nc d)
  omega

/-- The chunk's token count: `min chunk (seqlen - c·chunk)`. -/
theorem chunk_size_eq {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    endTok d c + 1 - c * d.chunk = min d.chunk (d.seqlen - c * d.chunk) := by
  have hck := chunk_pos hd
  have hlt := c_chunk_lt hd hc
  have h1 : 1 ≤ (c + 1) * d.chunk := by
    have h2 : 1 ≤ d.chunk := hck
    calc 1 = 1 * 1 := by norm_num
      _ ≤ (c + 1) * d.chunk := Nat.mul_le_mul (by omega) h2
  have hm1 : 1 ≤ min ((c + 1) * d.chunk) d.seqlen := le_min h1 hd.seqlen_pos
  unfold endTok
  rw [Nat.sub_add_cancel hm1]
  rw [min_sub ((c + 1) * d.chunk) d.seqlen (c * d.chunk)
    (Nat.mul_le_mul_right d.chunk (Nat.le_succ c)) (le_of_lt hlt)]
  have e : (c + 1) * d.chunk - c * d.chunk = d.chunk := by
    have e1 : (c + 1) * d.chunk = c * d.chunk + d.chunk := by ring
    omega
  rw [e]

/-- The loop's trip count. -/
theorem tripCount_eq {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) :
    tl.tripCount 0 (endTok d c + 1 - c * d.chunk : Int) (BK : Int) = steps d c := by
  have hlt := c_chunk_lt hd hc
  have hcs := chunk_size_eq hd hc
  have hck := chunk_pos hd
  have hge : c * d.chunk ≤ endTok d c + 1 := by
    have h1 : 1 ≤ min d.chunk (d.seqlen - c * d.chunk) :=
      le_min hck (by omega)
    omega
  have hnum : ((↑(endTok d c) + 1 - ↑c * ↑d.chunk - 0 + (BK : Int) - 1) : Int)
      = ((endTok d c + 1 - c * d.chunk + BK - 1 : Nat) : Int) := by
    have h1 : ((endTok d c + 1 - c * d.chunk : Nat) : Int)
        = ↑(endTok d c) + 1 - ↑c * ↑d.chunk := by
      rw [Nat.cast_sub hge, Nat.cast_add, Nat.cast_one, Nat.cast_mul]
    have h2 : 1 ≤ (endTok d c + 1 - c * d.chunk) + BK := by
      have hb : 0 < BK := by norm_num [BK]
      omega
    rw [Nat.cast_sub h2, Nat.cast_add, Nat.cast_one, h1]
    ring
  unfold tl.tripCount steps
  rw [hnum]
  rfl

/-- The loop runs at most `1024` times. -/
theorem steps_le {d : Inst} (hd : d.ok) {c : Nat} (hc : c < nc d) : steps d c ≤ 1024 := by
  have hcs := chunk_size_eq hd hc
  have hchunk : d.chunk ≤ 2 ^ 16 := hd.chunk_le
  have h2 : (2 : Nat) ^ 16 = 65536 := by norm_num
  have hle : endTok d c + 1 - c * d.chunk ≤ 65536 := by
    rw [hcs]
    have h := Nat.min_le_left d.chunk (d.seqlen - c * d.chunk)
    omega
  unfold steps
  have hN : endTok d c + 1 - c * d.chunk + BK - 1 ≤ 65599 := by
    simp only [BK]
    omega
  have h := Nat.div_add_mod (endTok d c + 1 - c * d.chunk + BK - 1) BK
  have hm := Nat.mod_lt (endTok d c + 1 - c * d.chunk + BK - 1)
    (show 0 < BK by norm_num [BK])
  simp only [BK] at h hm hN ⊢
  omega

/-- A token of the loop's `t`-th step is a token of the chunk. -/
theorem tok_lt {d : Inst} (hd : d.ok) {c t k : Nat} (hc : c < nc d)
    (h : t * BK + k < endTok d c + 1 - c * d.chunk) :
    c * d.chunk + t * BK + k < d.seqlen := by
  have hcs := chunk_size_eq hd hc
  have h1 : endTok d c + 1 - c * d.chunk ≤ d.seqlen - c * d.chunk := by
    rw [hcs]
    exact Nat.min_le_right _ _
  have h2 : c * d.chunk < d.seqlen := c_chunk_lt hd hc
  omega


/-! ## Arith -/

/-!
# `ssd_chunk_state_bf16`: the footprints are disjoint

The programs read the five input buffers and write disjoint tiles of `states`, which `disj`
separates from the inputs. `R` is "every byte outside `states`", so every input byte is in it
and the write footprint is outside it.
-/

set_option maxHeartbeats 800000


open TVA

/-- A program writes only bytes of `states`. -/
theorem W_sub {d : Inst} (hd : d.ok) (i : Fin (launch d).size) (init : Mem) :
    W d init i ⊆ d.states.bytes := by
  intro x hx
  refine Buf.elems_sub_bytes d.states (S := stSet d i) (fun k hk => ?_) hx
  have h1 : k < sSb d := stSet_sub hd i.isLt hk
  have h2 := hd.states_len
  omega

/-! ## Each input buffer lies in the read footprint -/

/-! ## The footprints are disjoint -/

/-- A program's writes are disjoint from every program's reads. -/
theorem W_disj_R {d : Inst} (hd : d.ok) (init : Mem) (i j : Fin (launch d).size) :
    Disjoint (W d init i) (R d init j) := by
  rw [Set.disjoint_left]
  intro z hz hz'
  exact hz' (W_sub hd i init hz)

/-- A program's writes are disjoint from another's reads and writes. -/
theorem disjoint_W {d : Inst} (hd : d.ok) {init : Mem} (_hpre : Pre d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (W d init i) (R d init j ∪ W d init j) := by
  have hii : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
  refine Set.disjoint_union_right.mpr ⟨W_disj_R hd init i j, ?_⟩
  exact Buf.elems_disjoint d.states (stSet_disjoint hd hii i.isLt j.isLt)

/-- A program reads nothing it writes. -/
theorem disjoint_RW {d : Inst} (hd : d.ok) {init : Mem} (_hpre : Pre d init)
    (i : Fin (launch d).size) :
    Disjoint (R d init i) (W d init i) :=
  (W_disj_R hd init i i).symm


/-! ## Zero -/

/-!
# The zero constant

The loop of the kernel fills masked-off lanes from a `float32` zero tile converted to `bfloat16`,
and multiplies by a `float32` zero where the sequence index does not match.  This file collects
the arithmetic facts about those zeros: their value is `0`, they are finite and not NaN, and the
conversions between `float32` and `bfloat16` and the multiplication by them keep the value `0`.
-/

open TVA

set_option maxHeartbeats 800000

/-- The `float32` zero pattern has value `0`. -/
theorem F32_zero_toRat : (FloatV.ofPat (0 : Nat) : F32).toRat = 0 := by
  rw [FloatV.toRat_ofPat (by decide)]
  exact (Format.toRat_eq_zero_iff).mpr Format.isZero_zero

/-- The `float32` zero pattern is finite. -/
theorem F32_zero_finite : (FloatV.ofPat (0 : Nat) : F32).isFinite = true := by
  rw [FloatV.isFinite_ofPat (by decide)]
  exact Format.isFinite_zero Format.binary32_regular

/-- The `float32` zero pattern is not a NaN. -/
theorem F32_zero_notNaN : (FloatV.ofPat (0 : Nat) : F32).isNaN = false := by
  rw [FloatV.isNaN_ofPat (by decide)]
  decide

/-- The `bfloat16` zero pattern has value `0`. -/
theorem BF16_zero_toRat : (FloatV.ofPat (0 : Nat) : BF16).toRat = 0 := by
  rw [FloatV.toRat_ofPat (by decide)]
  exact (Format.toRat_eq_zero_iff).mpr Format.isZero_zero

/-- The `bfloat16` zero pattern is finite. -/
theorem BF16_zero_finite : (FloatV.ofPat (0 : Nat) : BF16).isFinite = true := by
  rw [FloatV.isFinite_ofPat (by decide)]
  exact Format.isFinite_zero Format.bfloat16_regular

/-- The `bfloat16` zero pattern is not a NaN. -/
theorem BF16_zero_notNaN : (FloatV.ofPat (0 : Nat) : BF16).isNaN = false := by
  rw [FloatV.isNaN_ofPat (by decide)]
  decide

/-- The host literal `0.0` used by `tl.full` is the `float32` zero pattern. -/
theorem litF32_zero : (FloatV.ofPat (0 : Nat) : F64).litF32 = (FloatV.ofPat (0 : Nat) : F32) := by
  rw [FloatV.litF32, FloatV.isZero_ofPat (by decide), Format.isZero_zero]
  rfl

/-- Converting the `float32` zero to `bfloat16` gives the `bfloat16` zero pattern. -/
theorem cvt_f32_bf16_zero :
    (FloatV.cvt (FloatV.ofPat (0 : Nat) : F32) : BF16) = (FloatV.ofPat (0 : Nat) : BF16) := by
  have h : Format.binary32.cvt Format.bfloat16 .rn false 0 = 0 := by
    rw [Format.cvt_of_finite (g := Format.bfloat16) (rnd := .rn) (sat := false)
      (by rw [Format.isFinite_zero Format.binary32_regular])]
    rw [show Format.binary32.toRat 0 = 0 from F32_zero_toRat,
      Format.roundRat_zero]
    simp [Format.setSign, Format.sign, Format.mag]
  unfold FloatV.cvt
  rw [FloatV.pat_ofPat (by decide), h]

/-- Converting the `bfloat16` zero to `float32` gives the `float32` zero pattern. -/
theorem cvt_bf16_f32_zero :
    (FloatV.cvt (FloatV.ofPat (0 : Nat) : BF16) : F32) = (FloatV.ofPat (0 : Nat) : F32) := by
  have h : Format.bfloat16.cvt Format.binary32 .rn false 0 = 0 := by
    rw [Format.cvt_of_finite (f := Format.bfloat16)
      (g := Format.binary32) (rnd := .rn) (sat := false)
      (by rw [Format.isFinite_zero Format.bfloat16_regular])]
    rw [show Format.bfloat16.toRat 0 = 0 from BF16_zero_toRat,
      Format.roundRat_zero]
    simp [Format.setSign, Format.sign, Format.mag]
  unfold FloatV.cvt
  rw [FloatV.pat_ofPat (by decide), h]

/-- A platform conversion `float32 → bfloat16` of the zero pattern is the zero pattern (up to
NaN, which it is not). -/
theorem convRel_f32_bf16_zero {P : Platform} {r : BF16}
    (h : ConvRel P .f32 .bf16 (FloatV.ofPat (0 : Nat) : F32) r) :
    r = (FloatV.ofPat (0 : Nat) : BF16) := by
  simp only [ConvRel, ofFloatRel, FloatRel] at h
  simp only [show Format.binary32 ≠ Format.bfloat16 from by decide] at h
  rcases h with ⟨-, -, h | h⟩ | ⟨-, hg, -⟩ | ⟨hf, -⟩
  · rw [cvt_f32_bf16_zero] at h
    exact h.symm
  · rw [cvt_f32_bf16_zero, BF16_zero_notNaN] at h
    exact absurd h.1 (by simp)
  · rw [Format.F8] at hg
    rcases hg with h' | h' <;> exact absurd h' (by decide)
  · rw [Format.F8] at hf
    rcases hf with h' | h' <;> exact absurd h' (by decide)

/-- Multiplying a finite `float32` by the zero pattern gives a zero value, and stays finite. -/
theorem mul_zero_right {x : F32} (hx : x.isFinite = true) :
    (FloatV.mul x (FloatV.ofPat (0 : Nat))).toRat = 0 ∧
      (FloatV.mul x (FloatV.ofPat (0 : Nat))).isFinite = true := by
  have hp : (FloatV.ofPat (0 : Nat) : F32).pat = 0 := FloatV.pat_ofPat (by decide)
  have hfin0 : Format.binary32.isFinite 0 = true :=
    Format.isFinite_zero Format.binary32_regular
  have hto0 : Format.binary32.toRat 0 = 0 := F32_zero_toRat
  have hmul : Format.binary32.mul .rn x.pat (FloatV.ofPat (0 : Nat) : F32).pat
      = Format.binary32.roundRat .rn false
          (Format.binary32.sign x.pat != Format.binary32.sign 0) 0 := by
    rw [hp, Format.mul_of_finite (rnd := .rn) hx hfin0, hto0, mul_zero]
  obtain ⟨hlt, hfin, hto⟩ :=
    Format.roundRat_zero_spec Format.binary32_regular (rnd := .rn) (sat := false)
      (zs := Format.binary32.sign x.pat != Format.binary32.sign 0)
  unfold FloatV.mul
  rw [hmul]
  exact ⟨by rw [FloatV.toRat_ofPat hlt, hto], by rw [FloatV.isFinite_ofPat hlt, hfin]⟩

/-- (The same, with the zero on the left.) -/
theorem mul_zero_left {y : F32} (hy : y.isFinite = true) :
    (FloatV.mul (FloatV.ofPat (0 : Nat)) y).toRat = 0 ∧
      (FloatV.mul (FloatV.ofPat (0 : Nat)) y).isFinite = true := by
  have hp : (FloatV.ofPat (0 : Nat) : F32).pat = 0 := FloatV.pat_ofPat (by decide)
  have hfin0 : Format.binary32.isFinite 0 = true :=
    Format.isFinite_zero Format.binary32_regular
  have hto0 : Format.binary32.toRat 0 = 0 := F32_zero_toRat
  have hmul : Format.binary32.mul .rn (FloatV.ofPat (0 : Nat) : F32).pat y.pat
      = Format.binary32.roundRat .rn false
          (Format.binary32.sign 0 != Format.binary32.sign y.pat) 0 := by
    rw [hp, Format.mul_of_finite (rnd := .rn) hfin0 hy, hto0, zero_mul]
  obtain ⟨hlt, hfin, hto⟩ :=
    Format.roundRat_zero_spec Format.binary32_regular (rnd := .rn) (sat := false)
      (zs := Format.binary32.sign 0 != Format.binary32.sign y.pat)
  unfold FloatV.mul
  rw [hmul]
  exact ⟨by rw [FloatV.toRat_ofPat hlt, hto], by rw [FloatV.isFinite_ofPat hlt, hfin]⟩


/-- A finite `float32` of value `0`, of any sign, converts to a `bfloat16` of value `0`. -/
theorem cvt_f32_bf16_toRat_zero {x : F32} (hx : x.isFinite = true) (h0 : x.toRat = 0) :
    (FloatV.cvt x : BF16).toRat = 0 := by
  have hfin : Format.binary32.isFinite x.pat = true := hx
  have h0' : Format.binary32.toRat x.pat = 0 := h0
  have hval : Format.binary32.cvt Format.bfloat16 .rn false x.pat =
      Format.bfloat16.roundRat .rn false
        (Format.binary32.sign x.pat) 0 := by
    rw [Format.cvt_of_finite (g := Format.bfloat16) (rnd := .rn)
      (sat := false) hfin, h0']
  obtain ⟨hlt, -, hto⟩ := Format.roundRat_zero_spec
    Format.bfloat16_regular (rnd := .rn) (sat := false)
    (zs := Format.binary32.sign x.pat)
  unfold FloatV.cvt
  rw [hval, FloatV.toRat_ofPat hlt]
  exact hto


/-! ## RCmp -/

/-!
# Comparisons and conversions used by the chunk-state rules

Bit comparisons of `int32` values, the `int32` inequality against the `-1` sentinel, the
platform conversion `bfloat16 → float32` of finite values, the conversion of a zero `float32`
value into `bfloat16`, and the bound on `reqOf`.
-/

open TVA
set_option maxHeartbeats 800000

/-- The reference `==` of `IntV` compares bit patterns, which is equality of the values. -/
theorem IntV.eq_eq_decide {s : Bool} {w : Nat} (a b : IntV s w) :
    IntV.eq a b = decide (a = b) := by
  by_cases h : a.bits = b.bits
  · have hab : a = b := by
      cases a
      cases b
      simp_all
    subst hab
    simp [IntV.eq]
  · have hne : a ≠ b := fun hab => h (by rw [hab])
    simp [IntV.eq, hne, h]

/-- An `int32` literal below `2 ^ 31` is not the `-1` pattern. -/
theorem i32_ofNat_ne_negOne {r : Nat} (hr : r < 2 ^ 31) :
    (IntV.ofNat r : I32) ≠ IntV.ofInt (-1) := by
  intro h
  have h2 := congrArg IntV.toInt h
  rw [IntV.I32.toInt_ofNat hr, IntV.I32.toInt_ofInt (by norm_num)] at h2
  omega

/-- The widening `bfloat16 → float32` of a finite value is finite. -/
theorem cvt_bf16_f32_finite {x : BF16} (hx : x.isFinite = true) :
    (FloatV.cvt x : F32).isFinite = true := by
  have hfin : Format.bfloat16.isFinite x.pat = true := hx
  have hxr : |Format.bfloat16.toRat x.pat| ≤ Format.binary32.maxVal :=
    (Format.abs_toRat_le_maxVal Format.bfloat16_regular hfin).trans
      Format.wider_bfloat16_binary32.2.2
  obtain ⟨hlt, hf, -⟩ := Format.roundRat_rn_err
    (f := Format.binary32) Format.binary32_regular (sat := false)
    (zs := Format.bfloat16.sign x.pat) hxr
  unfold FloatV.cvt
  rw [Format.cvt_of_finite (g := Format.binary32) (rnd := .rn)
    (sat := false) hfin]
  rw [FloatV.isFinite_ofPat hlt]
  exact hf

/-- The request index is below `2 ^ 32`. -/
theorem reqOf_lt {d : Inst} (hd : d.ok) (t : Nat) : reqOf d t < 2 ^ 32 := by
  have h1 := List.length_filter_le (fun r => decide (d.qsl r ≤ t)) (laterReqs d)
  rw [show (laterReqs d).length = d.nreq - 1 from by simp [laterReqs]] at h1
  have h3 := hd.nreq_le
  unfold reqOf
  omega


/-! ## BC -/

/-!
# Small facts for the `b` value clause

The `b` clause of the loop invariant needs the `bfloat16`-valued `DType.val?` unfolding, the bound
`reqOf < 2^31` on a `bfloat16` chunk's request index, and the finiteness of the `float32 → bfloat16`
conversion of a finite zero.
-/

open TVA

set_option maxHeartbeats 800000

/-- `DType.val?` of a `bfloat16` value. -/
theorem DType.val?_bf16 {x : BF16} {v : ℚ} :
    DType.val? .bf16 x = some v ↔ x.isFinite = true ∧ x.toRat = v := by
  show (if x.isFinite = true then some x.toRat else none) = some v ↔ _
  split_ifs with h
  · simp [h]
  · simp [h]

/-- The request index is below `2 ^ 31`. -/
theorem reqOf_lt_2_31 {d : Inst} (hd : d.ok) (t : Nat) : reqOf d t < 2 ^ 31 := by
  have h1 := List.length_filter_le (fun r => decide (d.qsl r ≤ t)) (laterReqs d)
  rw [show (laterReqs d).length = d.nreq - 1 from by simp [laterReqs]] at h1
  have h3 := hd.nreq_le
  unfold reqOf
  omega

/-- A finite `float32` of value `0` converts to a finite `bfloat16`. -/
theorem cvt_f32_bf16_zero_finite {x : F32} (hx : x.isFinite = true) (h0 : x.toRat = 0) :
    (FloatV.cvt x : BF16).isFinite = true := by
  have hfin : Format.binary32.isFinite x.pat = true := hx
  have h0' : Format.binary32.toRat x.pat = 0 := h0
  have hval : Format.binary32.cvt Format.bfloat16 .rn false x.pat =
      Format.bfloat16.roundRat .rn false
        (Format.binary32.sign x.pat) 0 := by
    rw [Format.cvt_of_finite (g := Format.bfloat16) (rnd := .rn)
      (sat := false) hfin, h0']
  obtain ⟨hlt, hf, -⟩ := Format.roundRat_zero_spec
    Format.bfloat16_regular (rnd := .rn) (sat := false)
    (zs := Format.binary32.sign x.pat)
  unfold FloatV.cvt
  rw [hval, FloatV.isFinite_ofPat hlt]
  exact hf


/-! ## LRead -/
open TVA
set_option maxHeartbeats 800000

/-!
# Reading a masked tile lane by lane

Masked variants of `read_some_of_lane` and `accBytes_R_of_lane`: the lane bound is only required
on the lanes the mask enables.
-/

/-- A masked lane of a valid buffer reads a value. -/
theorem read_some_masked {d' : DType} {b : Buf d'} {m : Mem} (hv : b.Valid m) {s : Shape}
    {ptrs : Tensor s (Ptr d')} {mask : Tensor s Bool} {g : Idx s → Nat}
    (haddr : ∀ ix, (ptrs.get ix).addr = (b.ptr (g ix : Int)).addr)
    (hlen : ∀ ix, mask.get ix = true → g ix < b.len) (i : Idx s) (a : Int)
    (ha : tl.addrs ptrs (some mask) i = some a) : ∃ v, m.read d' a = some v := by
  simp only [tl.addrs] at ha
  by_cases hm : mask.get i = true
  · rw [ite_eq_left hm] at ha
    obtain rfl : (ptrs.get i).addr = a := by simpa using ha
    rw [haddr i]
    exact b.read_some hv (hlen i hm)
  · rw [ite_eq_right hm] at ha
    exact absurd ha (by simp)

/-- An enabled lane of a masked load reads its value from memory. -/
theorem load_read {d' : DType} {s : Shape} {ptrs : Tensor s (Ptr d')} {mask : Tensor s Bool}
    {o : Tensor s d'.Val} {m : Mem} {v : Tensor s d'.Val}
    (h : LoadRes m ptrs (some mask) (some o) v) {i : Idx s} (hi : mask.get i = true) :
    m.read d' (ptrs.get i).addr = some (v.get i) := by
  refine (h i).1 _ ?_
  simp [tl.addrs, hi]

/-- A disabled lane of a masked load takes the `other` value. -/
theorem load_mask_false {d' : DType} {s : Shape} {ptrs : Tensor s (Ptr d')} {mask : Tensor s Bool}
    {o : Tensor s d'.Val} {m : Mem} {v : Tensor s d'.Val}
    (h : LoadRes m ptrs (some mask) (some o) v) {i : Idx s} (hi : mask.get i = false) :
    v.get i = o.get i := by
  refine (h i).2 ?_ o rfl
  simp [tl.addrs, hi]

/-- The bytes a masked load touches in an input buffer are in the read footprint. -/
theorem accBytes_R_of_masked {d : Inst} {init : Mem} (i : Fin (launch d).size) {d' : DType}
    {b : Buf d'} (hdisj : Disjoint d.states.bytes b.bytes) {s : Shape}
    {ptrs : Tensor s (Ptr d')} {mask : Tensor s Bool} {g : Idx s → Nat}
    (haddr : ∀ ix, (ptrs.get ix).addr = (b.ptr (g ix : Int)).addr)
    (hlen : ∀ ix, mask.get ix = true → g ix < b.len) :
    accBytes d' (tl.addrs ptrs (some mask)) ⊆ R d init i := by
  intro x hx
  simp only [accBytes, Set.mem_ofPred_eq] at hx
  obtain ⟨j, a, ha, hxspan⟩ := hx
  simp only [tl.addrs] at ha
  simp only [R, Set.mem_ofPred_eq]
  by_cases hm : mask.get j = true
  · rw [ite_eq_left hm] at ha
    obtain rfl : (ptrs.get j).addr = a := by simpa using ha
    have hle : Mem.span d' (ptrs.get j).addr ⊆ b.bytes := by
      rw [haddr j]
      exact b.span_sub (hlen j hm)
    intro hxs
    exact (Set.disjoint_left.mp hdisj hxs) (hle hxspan)
  · rw [ite_eq_right hm] at ha
    exact absurd ha (by simp)


/-! ## LaneRead -/

/-!
# Reading a tile lane by lane

A masked `tl.load` of a pointer tile whose lanes address elements of one buffer reads only
elements of that buffer, so every enabled lane is readable, and the loaded bytes are outside the
`states` buffer.
-/

open TVA

set_option maxHeartbeats 800000

/-- A lane of a valid buffer reads a value. -/
theorem read_some_of_lane {d' : DType} {b : Buf d'} {m : Mem} (hv : b.Valid m) {s : Shape}
    {ptrs : Tensor s (Ptr d')} {mask : Tensor s Bool} {g : Idx s → Nat}
    (haddr : ∀ ix, (ptrs.get ix).addr = (b.ptr (g ix : Int)).addr)
    (hlen : ∀ ix, g ix < b.len) (i : Idx s) (a : Int)
    (ha : tl.addrs ptrs (some mask) i = some a) : ∃ v, m.read d' a = some v := by
  simp only [tl.addrs] at ha
  split at ha
  · obtain rfl : (ptrs.get i).addr = a := by simpa using ha
    rw [haddr i]
    exact b.read_some hv (hlen i)
  · exact absurd ha (by simp)

/-- The bytes an unmasked load touches in an input buffer are in the read footprint. -/
theorem accBytes_R_of_lane {d : Inst} {init : Mem} (i : Fin (launch d).size) {d' : DType}
    {b : Buf d'} (hdisj : Disjoint d.states.bytes b.bytes) {s : Shape}
    {ptrs : Tensor s (Ptr d')} {mask : Tensor s Bool} {g : Idx s → Nat}
    (haddr : ∀ ix, (ptrs.get ix).addr = (b.ptr (g ix : Int)).addr)
    (hlen : ∀ ix, g ix < b.len) :
    accBytes d' (tl.addrs ptrs (some mask)) ⊆ R d init i := by
  intro x hx
  simp only [accBytes, Set.mem_ofPred_eq] at hx
  obtain ⟨j, a, ha, hxspan⟩ := hx
  simp only [tl.addrs] at ha
  simp only [R, Set.mem_ofPred_eq]
  split at ha
  · obtain rfl : (ptrs.get j).addr = a := by simpa using ha
    have hle : Mem.span d' (ptrs.get j).addr ⊆ b.bytes := by
      rw [haddr j]
      exact b.span_sub (hlen j)
    intro hxs
    exact (Set.disjoint_left.mp hdisj hxs) (hle hxspan)
  · exact absurd ha (by simp)


end Claims.SsdChunkState
