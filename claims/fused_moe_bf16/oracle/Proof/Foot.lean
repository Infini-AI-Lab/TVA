import TVA.Proof
import Claim
import Proof.Pid

/-!
# Footprints and the assembly of `Post`

Program `i` owns the tile `(pidM i, pidN i)`: the lanes `l < BM` of block `pidM i`, whose flat
index `entryOf (pidM i * BM + l)` is routed, and the columns `pidN i * BN + j < N`. It reads the
input buffers (`Rfp`), writes its tile of `C` (`Wfp`) and leaves `outElem` in every element of it
(`Qfp`). The tiles are disjoint (distinct flat indices, distinct column tiles), `C` shares no byte
with the inputs, and every `(f, c)` is covered by the tile of the position that holds `f`.
-/

namespace Claims.FusedMoe

open TVA

/-- The bytes every program may read: the six input buffers. -/
def Rfp (d : Inst) : Set Int :=
  d.a.bytes ∪ d.b.bytes ∪ d.w.bytes ∪ d.s.bytes ∪ d.eid.bytes ∪ d.ntpb.bytes

/-- The elements of `C` that block `pm`, column tile `pn` write: lane `l`, column `j`, when the
position is below the padded total, its flat index is routed and the column is inside `N`. -/
def Wset (d : Inst) (init : Mem) (pm pn : Nat) : Set Nat :=
  {x | ∃ l j, l < d.BM ∧ j < d.BN ∧ pm * d.BM + l < ntpOf d init ∧
    entryOf d init (pm * d.BM + l) < (nv d : Int) ∧ pn * d.BN + j < d.N ∧
    x = (entryOf d init (pm * d.BM + l)).toNat * stride_cm d + (pn * d.BN + j)}

/-- The write footprint of program `i`. -/
def Wfp (d : Inst) (init : Mem) (i : Fin (launch d).size) : Set Int :=
  d.c.elems (Wset d init (pidM d i) (pidN d i))

/-- The post-condition of the program of tile `(pm, pn)`: every element of its tile satisfies
`outElem`. -/
def QT (d : Inst) (init : Mem) (pm pn : Nat) (m : Mem) : Prop :=
  ∀ l j, l < d.BM → j < d.BN → pm * d.BM + l < ntpOf d init →
    entryOf d init (pm * d.BM + l) < (nv d : Int) → pn * d.BN + j < d.N →
    outElem d init m (pm * d.BM + l) (entryOf d init (pm * d.BM + l)).toNat (pn * d.BN + j)

/-- The post-condition of program `i`. -/
def Qfp (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop :=
  QT d init (pidM d i) (pidN d i) m

/-- The tile's elements are elements of the matrix `C`. -/
theorem Wset_sub_cIdx {d : Inst} {init : Mem} (hpre : Pre d init) (pm pn : Nat) :
    Wset d init pm pn ⊆ cIdx d := by
  rintro x ⟨l, j, hl, hj, hpos, hent, hc, rfl⟩
  have h0 := hpre.s1 _ hpos
  refine ⟨(entryOf d init (pm * d.BM + l)).toNat, pn * d.BN + j, ?_, hc, rfl⟩
  unfold nv at hent ⊢
  omega

/-- The tile of program `i` is inside the matrix `C`. -/
theorem Wfp_sub {d : Inst} {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size) :
    Wfp d init i ⊆ d.c.elems (cIdx d) :=
  d.c.elems_mono (Wset_sub_cIdx hpre _ _)

/-- No program reads what it writes. -/
theorem R_W_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) : Disjoint (Rfp d) (Wfp d init i) := by
  have h := Wfp_sub hpre i
  unfold Rfp
  refine Disjoint.union_left (Disjoint.union_left (Disjoint.union_left (Disjoint.union_left
    (Disjoint.union_left ?_ ?_) ?_) ?_) ?_) ?_
  · exact (hd.c_a.mono h le_rfl).symm
  · exact (hd.c_b.mono h le_rfl).symm
  · exact (hd.c_w.mono h le_rfl).symm
  · exact (hd.c_s.mono h le_rfl).symm
  · exact (hd.c_e.mono h le_rfl).symm
  · exact (hd.c_n.mono h le_rfl).symm

/-- Two elements of `C` with overlapping bytes are the same element. -/
theorem mem_elems_single {b : Buf .bf16} {S : Set Nat} {x : Nat} {z : Int}
    (h1 : z ∈ b.elems {x}) (h2 : z ∈ b.elems S) : x ∈ S := by
  by_contra hx
  have hdis : Disjoint ({x} : Set Nat) S := Set.disjoint_singleton_left.mpr hx
  exact Set.disjoint_left.mp (b.elems_disjoint hdis) h1 h2

/-- A row-major index `f * N + c` with `c < N` determines `f` and `c`. -/
theorem mul_add_inj {f c f' c' N : Nat} (hc : c < N) (hc' : c' < N) (h : f * N + c = f' * N + c') :
    f = f' ∧ c = c' := by
  have hN : 0 < N := by omega
  have d1 : (c + f * N) / N = f := by
    rw [Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hc, Nat.zero_add]
  have d2 : (c' + f' * N) / N = f' := by
    rw [Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hc', Nat.zero_add]
  have m1 : (c + f * N) % N = c := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc]
  have m2 : (c' + f' * N) % N = c' := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc']
  have e : c + f * N = c' + f' * N := by omega
  constructor
  · rw [← d1, ← d2, e]
  · rw [← m1, ← m2, e]

/-- The bytes of an element in a set of elements. -/
theorem mem_elems_of_span {d : DType} (b : Buf d) {S : Set Nat} {x : Nat} {z : Int}
    (hx : x ∈ S) (hz : z ∈ Mem.span d (b.ptr (x : Int)).addr) : z ∈ b.elems S := ⟨x, hx, hz⟩

/-- The tiles of two programs are disjoint. -/
theorem W_disjoint_W {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) : Disjoint (Wfp d init i) (Wfp d init j) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hj : (j : Nat) < grid d := lt_of_lt_of_eq j.isLt (launch_size d)
  rw [Wfp, Wfp]
  refine d.c.elems_disjoint ?_
  rw [Set.disjoint_left]
  rintro x ⟨l, k, hl, hk, hpos, hent, hc, rfl⟩ ⟨l', k', hl', hk', hpos', hent', hc', hx⟩
  have h0 := hpre.s1 _ hpos
  have h0' := hpre.s1 _ hpos'
  obtain ⟨hf, hcc⟩ := mul_add_inj hc hc' hx
  have hff : entryOf d init (pidM d i * d.BM + l) = entryOf d init (pidM d j * d.BM + l') := by omega
  have hpp := hpre.s2 _ _ hpos hpos' hent hff.symm
  have hBM := (cfg_facts hd).1
  have hBN := (cfg_facts hd).2.1
  have hm : pidM d i = pidM d j := by
    have h1 : (pidM d j * d.BM + l') / d.BM = (pidM d i * d.BM + l) / d.BM := by rw [hpp]
    have e1 : (pidM d j * d.BM + l') / d.BM = pidM d j := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hl', Nat.zero_add]
    have e2 : (pidM d i * d.BM + l) / d.BM = pidM d i := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hl, Nat.zero_add]
    omega
  have hn : pidN d i = pidN d j := by
    have e1 : (pidN d j * d.BN + k') / d.BN = pidN d j := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hk', Nat.zero_add]
    have e2 : (pidN d i * d.BN + k) / d.BN = pidN d i := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hk, Nat.zero_add]
    have : (pidN d i * d.BN + k) / d.BN = (pidN d j * d.BN + k') / d.BN := by rw [hcc]
    omega
  exact hij (Fin.ext (pid_inj hd hi hj hm hn))

/-- `outElem` depends on the final memory only through the element's bytes. -/
theorem outElem_congr {d : Inst} {init m fin : Mem} {pos f c : Nat}
    (h : ∀ x ∈ Mem.span .bf16 (d.c.ptr ((f * stride_cm d + c : Nat) : Int)).addr, fin x = m x)
    (hm : outElem d init m pos f c) : outElem d init fin pos f c := by
  obtain ⟨cv, hr, rest⟩ := hm
  exact ⟨cv, by rw [Mem.read_congr h]; exact hr, rest⟩

/-- An index below `n` lies in one of the `cdiv(n, b)` tiles of size `b`. -/
theorem div_lt_cdiv {x n b : Nat} (hb : 0 < b) (hx : x < n) : x / b < (n + b - 1) / b := by
  have h1 : x / b * b ≤ x := Nat.div_mul_le_self x b
  have : x / b + 1 ≤ (n + b - 1) / b := by
    rw [Nat.le_div_iff_mul_le hb, Nat.add_mul, Nat.one_mul]
    omega
  omega

/-- The programs' results assemble into `Post`. -/
theorem post_of_tiles {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (fin : Mem)
    (hQ : ∀ i : Fin (launch d).size, ∃ m, Qfp d init i m ∧ ∀ x ∈ Rfp d ∪ Wfp d init i, fin x = m x)
    (hfr : ∀ x, (∀ i, x ∉ Wfp d init i) → fin x = init x) : Post d init fin := by
  refine ⟨fun f c hf hc => ⟨fun hno z hz => hfr z fun i hzi => ?_, fun pos hpos hent => ?_⟩,
    fun z hz => hfr z fun i hzi => hz (Wfp_sub hpre i hzi)⟩
  · obtain ⟨l, j, hl, hj, hp, he, hcj, hx⟩ := mem_elems_single hz hzi
    obtain ⟨h1, _⟩ := mul_add_inj hc hcj hx
    have h0 := hpre.s1 _ hp
    exact hno _ hp (by omega)
  · have hBM := (cfg_facts hd).1
    have hBN := (cfg_facts hd).2.1
    have hpos' := hpre.ntpEM
    obtain ⟨p, hp, hpm, hpn⟩ := pid_surj hd (a := pos / d.BM) (b := c / d.BN)
      (div_lt_cdiv (by omega) (by omega)) (div_lt_cdiv (by omega) hc)
    obtain ⟨m, hQm, hagree⟩ := hQ ⟨p, (launch_size d).symm ▸ hp⟩
    have hl : pos % d.BM < d.BM := Nat.mod_lt _ (by omega)
    have hj : c % d.BN < d.BN := Nat.mod_lt _ (by omega)
    have hposE : pidM d p * d.BM + pos % d.BM = pos := by
      rw [hpm, Nat.mul_comm]; exact Nat.div_add_mod pos d.BM
    have hcE : pidN d p * d.BN + c % d.BN = c := by
      rw [hpn, Nat.mul_comm]; exact Nat.div_add_mod c d.BN
    have h0 := hpre.s1 _ hpos
    have hfE : (entryOf d init pos).toNat = f := by omega
    have hQ' := hQm (pos % d.BM) (c % d.BN) hl hj (by rw [hposE]; exact hpos)
      (by rw [hposE, hent]; exact_mod_cast hf) (by rw [hcE]; exact hc)
    rw [hposE, hcE, hfE] at hQ'
    refine outElem_congr (m := m) (fun x hx => hagree x (Or.inr ?_)) hQ'
    refine mem_elems_of_span (d.c) ?_ hx
    show _ ∈ Wset d init (pidM d p) (pidN d p)
    refine ⟨pos % d.BM, c % d.BN, hl, hj, by rw [hposE]; exact hpos, by rw [hposE, hent]; exact_mod_cast hf,
      by rw [hcE]; exact hc, ?_⟩
    rw [hposE, hcE, hfE]

end Claims.FusedMoe
