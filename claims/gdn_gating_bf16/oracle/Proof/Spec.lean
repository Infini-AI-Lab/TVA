import Claim
import TVA.Proof
import Proof.Basic
import Proof.Idx

/-!
# The footprints of one program, and the specification from them

Program `i` reads the eight-or-fewer elements of `A_log`, `a`, `b`, `dt_bias` of its lanes and
writes those of `g`, `beta_output`.  `Rset`/`Wset` are these two sets of bytes, `Q` is what the
program leaves in its own memory, and this file proves the three `meets_independent` side
conditions: no two programs' write footprints meet, no program reads what it writes, and the
programs' results give `Post`.
-/

namespace Claims.GdnGating

open TVA

/-! ## The footprints -/

/-- The `A_log`/`dt_bias` elements of program `i`'s lanes. -/
abbrev HSet (d : Inst) (i : Fin (launch d).size) : Set Nat :=
  Buf.img (gH d i.val) (en8 d i.val)

/-- The `a` elements of program `i`'s lanes. -/
abbrev aSet (d : Inst) (i : Fin (launch d).size) : Set Nat :=
  Buf.img (gR d i.val d.stride_a) (en8 d i.val)

/-- The `b` elements of program `i`'s lanes. -/
abbrev bSet (d : Inst) (i : Fin (launch d).size) : Set Nat :=
  Buf.img (gR d i.val d.stride_b) (en8 d i.val)

/-- The `g`/`beta_output` elements of program `i`'s lanes. -/
abbrev oSet (d : Inst) (i : Fin (launch d).size) : Set Nat :=
  Buf.img (gO d i.val) (en8 d i.val)

/-- The read footprint of program `i`. -/
abbrev Rset (d : Inst) (i : Fin (launch d).size) : Set Int :=
  d.A_log.elems (HSet d i) ∪ d.a.elems (aSet d i) ∪ d.b.elems (bSet d i) ∪
    d.dt_bias.elems (HSet d i)

/-- The write footprint of program `i`. -/
abbrev Wset (d : Inst) (i : Fin (launch d).size) : Set Int :=
  d.g.elems (oSet d i) ∪ d.bout.elems (oSet d i)

/-! ## What one program leaves -/

/-- **The local postcondition.** Every enabled lane of the program finds `A`, `av`, `bv`, `bb`
in the initial memory and leaves in its element of `g` a `float32` within `gTol` of
`gRef A (av + bv)` (or `-∞` when the exact value is at the edge of `float32`), and in its element
of `beta_output` a `float32` within `bTol` of `sigRef bb`. -/
def Q (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) : Prop :=
  (∀ l : Fin 8, en d i.val l.val = true →
    ∃ A : F32, init.read .f32 (d.A_log.ptr (gH d i.val (lane l))).addr = some A ∧
    ∃ av : BF16, init.read .bf16 (d.a.ptr (gR d i.val d.stride_a (lane l))).addr = some av ∧
    ∃ bv : BF16, init.read .bf16 (d.dt_bias.ptr (gH d i.val (lane l))).addr = some bv ∧
    ∃ c : F32, m.read .f32 (d.g.ptr (gO d i.val (lane l))).addr = some c ∧
      ((c.isFinite = true ∧
          |c.toReal - gRef A.toReal (av.toReal + bv.toReal)| ≤
            gTol A.toReal (av.toReal + bv.toReal)) ∨
        (c = FloatV.negInf ∧
          (Format.binary32.maxVal : ℝ) ≤
            |gRef A.toReal (av.toReal + bv.toReal)| + gTol A.toReal (av.toReal + bv.toReal)))) ∧
  (∀ l : Fin 8, en d i.val l.val = true →
    ∃ bb : BF16, init.read .bf16 (d.b.ptr (gR d i.val d.stride_b (lane l))).addr = some bb ∧
    ∃ c : F32, m.read .f32 (d.bout.ptr (gO d i.val (lane l))).addr = some c ∧
      c.isFinite = true ∧ |c.toReal - sigRef bb.toReal| ≤ bTol bb.toReal)

/-! ## The bytes of the buffers are disjoint -/

theorem g_bytes_disj {d : Inst} (hd : d.ok) :
    Disjoint d.g.bytes d.A_log.bytes ∧ Disjoint d.g.bytes d.a.bytes ∧
    Disjoint d.g.bytes d.b.bytes ∧ Disjoint d.g.bytes d.dt_bias.bytes ∧
    Disjoint d.g.bytes d.bout.bytes := by
  have h := hd.g_disj
  rw [Set.disjoint_union_right] at h
  obtain ⟨h1, h5⟩ := h
  rw [Set.disjoint_union_right] at h1
  obtain ⟨h1, h4⟩ := h1
  rw [Set.disjoint_union_right] at h1
  obtain ⟨h1, h3⟩ := h1
  rw [Set.disjoint_union_right] at h1
  obtain ⟨h1, h2⟩ := h1
  exact ⟨h1, h2, h3, h4, h5⟩

/-- `beta_output` shares no byte with the inputs. -/
theorem bout_bytes_disj {d : Inst} (hd : d.ok) :
    Disjoint d.bout.bytes d.A_log.bytes ∧ Disjoint d.bout.bytes d.a.bytes ∧
    Disjoint d.bout.bytes d.b.bytes ∧ Disjoint d.bout.bytes d.dt_bias.bytes := by
  have h := hd.bout_disj
  rw [Set.disjoint_union_right] at h
  obtain ⟨h1, h4⟩ := h
  rw [Set.disjoint_union_right] at h1
  obtain ⟨h1, h3⟩ := h1
  rw [Set.disjoint_union_right] at h1
  obtain ⟨h1, h2⟩ := h1
  exact ⟨h1, h2, h3, h4⟩

/-! ## The elements of the lanes are inside their buffers -/

theorem len_g {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ∀ j ∈ oSet d i, j < d.g.len := by
  rintro j ⟨p, hp, rfl⟩
  exact gO_len_g hd i p.1 (hidx_of_en hp)

/-- A program's elements of `beta_output` are inside the buffer. -/
theorem len_bout {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ∀ j ∈ oSet d i, j < d.bout.len := by
  rintro j ⟨p, hp, rfl⟩
  exact gO_len_bout hd i p.1 (hidx_of_en hp)

/-- A program's elements of `A_log` are inside the buffer. -/
theorem len_A {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ∀ j ∈ HSet d i, j < d.A_log.len := by
  rintro j ⟨p, hp, rfl⟩
  exact gH_len_A hd i p.1 (hidx_of_en hp)

/-- A program's elements of `dt_bias` are inside the buffer. -/
theorem len_bias {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ∀ j ∈ HSet d i, j < d.dt_bias.len := by
  rintro j ⟨p, hp, rfl⟩
  exact gH_len_bias hd i p.1 (hidx_of_en hp)

/-- A program's elements of `a` are inside the buffer. -/
theorem len_a {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ∀ j ∈ aSet d i, j < d.a.len := by
  rintro j ⟨p, hp, rfl⟩
  exact gR_len_a hd i p.1 (hidx_of_en hp)

/-- A program's elements of `b` are inside the buffer. -/
theorem len_b {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ∀ j ∈ bSet d i, j < d.b.len := by
  rintro j ⟨p, hp, rfl⟩
  exact gR_len_b hd i p.1 (hidx_of_en hp)

/-! ## The write elements of two programs are disjoint -/

theorem gO_inj_ge {d : Inst} {i j : Fin (launch d).size} {p q : Idx [8]}
    (hpi : en8 d i.val p = true) (hpj : en8 d j.val q = true)
    (h : gO d i.val p = gO d j.val q) : i = j := by
  have hi : hidx d i.val p.1.val < d.H := hidx_of_en hpi
  have hj : hidx d j.val q.1.val < d.H := hidx_of_en hpj
  have hH : 0 < d.H := by omega
  have hmod : hidx d i.val p.1.val = hidx d j.val q.1.val := by
    have h1 : (tok d i.val * d.H + hidx d i.val p.1.val) % d.H =
        (tok d j.val * d.H + hidx d j.val q.1.val) % d.H := congrArg (fun n => n % d.H) h
    rw [Nat.mul_comm (tok d i.val) d.H, Nat.mul_comm (tok d j.val) d.H,
      Nat.mul_add_mod, Nat.mul_add_mod, Nat.mod_eq_of_lt hi, Nat.mod_eq_of_lt hj] at h1
    exact h1
  have hblk : blk d i.val = blk d j.val := by
    have h1 := hmod
    simp only [hidx] at h1
    omega
  have htok : tok d i.val = tok d j.val := by
    have h1 : tok d i.val * d.H + hidx d i.val p.1.val =
        tok d j.val * d.H + hidx d j.val q.1.val := by
      simpa only [gO] using h
    rw [hmod] at h1
    exact Nat.mul_right_cancel hH (Nat.add_right_cancel h1)
  have h1 : i.val = j.val := by
    rw [← Nat.div_add_mod' i.val d.batch, ← Nat.div_add_mod' j.val d.batch]
    simp only [tok, blk] at htok hblk
    rw [hblk, htok]
  exact Fin.ext h1

/-- The output elements of distinct programs are distinct. -/
theorem img_gO_disj {d : Inst} {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (oSet d i) (oSet d j) := by
  rw [Set.disjoint_left]
  intro a ha ha'
  obtain ⟨p, hp, hpa⟩ := ha
  obtain ⟨q, hq, hqa⟩ := ha'
  exact hij (gO_inj_ge hp hq (by rw [hpa, hqa]))

/-! ## Disjointness of the footprints -/

theorem disj_g_A {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.g.elems (oSet d i)) (d.A_log.elems (HSet d j)) :=
  Buf.elems_disjoint_of d.g d.A_log (g_bytes_disj hd).1 (len_g hd i) (len_A hd j)

/-- No program's `g` elements meet any program's `a` elements. -/
theorem disj_g_a {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.g.elems (oSet d i)) (d.a.elems (aSet d j)) :=
  Buf.elems_disjoint_of d.g d.a (g_bytes_disj hd).2.1 (len_g hd i) (len_a hd j)

/-- No program's `g` elements meet any program's `b` elements. -/
theorem disj_g_b {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.g.elems (oSet d i)) (d.b.elems (bSet d j)) :=
  Buf.elems_disjoint_of d.g d.b (g_bytes_disj hd).2.2.1 (len_g hd i) (len_b hd j)

/-- No program's `g` elements meet any program's `dt_bias` elements. -/
theorem disj_g_dt {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.g.elems (oSet d i)) (d.dt_bias.elems (HSet d j)) :=
  Buf.elems_disjoint_of d.g d.dt_bias (g_bytes_disj hd).2.2.2.1 (len_g hd i) (len_bias hd j)

/-- No program's `g` elements meet any program's `beta_output` elements. -/
theorem disj_g_bout {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.g.elems (oSet d i)) (d.bout.elems (oSet d j)) :=
  Buf.elems_disjoint_of d.g d.bout (g_bytes_disj hd).2.2.2.2 (len_g hd i) (len_bout hd j)

/-- No program's `beta_output` elements meet any program's `A_log` elements. -/
theorem disj_bout_A {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.bout.elems (oSet d i)) (d.A_log.elems (HSet d j)) :=
  Buf.elems_disjoint_of d.bout d.A_log (bout_bytes_disj hd).1 (len_bout hd i) (len_A hd j)

/-- No program's `beta_output` elements meet any program's `a` elements. -/
theorem disj_bout_a {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.bout.elems (oSet d i)) (d.a.elems (aSet d j)) :=
  Buf.elems_disjoint_of d.bout d.a (bout_bytes_disj hd).2.1 (len_bout hd i) (len_a hd j)

/-- No program's `beta_output` elements meet any program's `b` elements. -/
theorem disj_bout_b {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.bout.elems (oSet d i)) (d.b.elems (bSet d j)) :=
  Buf.elems_disjoint_of d.bout d.b (bout_bytes_disj hd).2.2.1 (len_bout hd i) (len_b hd j)

/-- No program's `beta_output` elements meet any program's `dt_bias` elements. -/
theorem disj_bout_dt {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) :
    Disjoint (d.bout.elems (oSet d i)) (d.dt_bias.elems (HSet d j)) :=
  Buf.elems_disjoint_of d.bout d.dt_bias (bout_bytes_disj hd).2.2.2 (len_bout hd i)
    (len_bias hd j)

/-- No program writes another program's footprint. -/
theorem hdisj_spec {d : Inst} (hd : d.ok) {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (Wset d i) (Rset d j ∪ Wset d j) := by
  refine Disjoint.union_right ?_ ?_
  · refine Disjoint.union_right ?_ ?_
    · refine Disjoint.union_right ?_ ?_
      · refine Disjoint.union_right ?_ ?_
        · exact Disjoint.union_left (disj_g_A hd i j) (disj_bout_A hd i j)
        · exact Disjoint.union_left (disj_g_a hd i j) (disj_bout_a hd i j)
      · exact Disjoint.union_left (disj_g_b hd i j) (disj_bout_b hd i j)
    · exact Disjoint.union_left (disj_g_dt hd i j) (disj_bout_dt hd i j)
  · refine Disjoint.union_right ?_ ?_
    · refine Disjoint.union_left ?_ ?_
      · exact Buf.elems_disjoint d.g (img_gO_disj hij)
      · exact (disj_g_bout hd j i).symm
    · refine Disjoint.union_left ?_ ?_
      · exact disj_g_bout hd i j
      · exact Buf.elems_disjoint d.bout (img_gO_disj hij)

/-- No program reads what it writes. -/
theorem hRW_spec {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    Disjoint (Rset d i) (Wset d i) := by
  refine Disjoint.union_left (Disjoint.union_left (Disjoint.union_left ?_ ?_) ?_) ?_
  · exact Disjoint.union_right (disj_g_A hd i i).symm (disj_bout_A hd i i).symm
  · exact Disjoint.union_right (disj_g_a hd i i).symm (disj_bout_a hd i i).symm
  · exact Disjoint.union_right (disj_g_b hd i i).symm (disj_bout_b hd i i).symm
  · exact Disjoint.union_right (disj_g_dt hd i i).symm (disj_bout_dt hd i i).symm

/-! ## From the local postconditions to `Post` -/

theorem gO_lt_flat {d : Inst} (i : Fin (launch d).size) {p : Idx [8]}
    (h : en8 d i.val p = true) : gO d i.val p < d.batch * d.H := by
  have hp : hidx d i.val p.1.val < d.H := hidx_of_en h
  have ht : tok d i.val + 1 ≤ d.batch := by
    have := tok_lt i
    omega
  have h3 : (tok d i.val + 1) * d.H ≤ d.batch * d.H := Nat.mul_le_mul_right _ ht
  have heq : (tok d i.val + 1) * d.H = tok d i.val * d.H + d.H := by ring
  simp only [gO]
  omega

/-- The first `batch * num_heads` elements of a buffer. -/
theorem elems_mono_flat {d : Inst} (i : Fin (launch d).size) :
    oSet d i ⊆ {j | j < d.batch * d.H} := by
  rintro j ⟨p, hp, rfl⟩
  exact gO_lt_flat i hp

/-- The program and the lane that handle token `t` and head `h`. -/
theorem post_prog {d : Inst} {t h : Nat} (ht : t < d.batch) (hh : h < d.H) :
    d.batch * (h / 8) + t < (launch d).size := by
  rw [launch_size, Bn]
  have hq : h / 8 + 1 ≤ (d.H + 7) / 8 := by omega
  have h1 : d.batch * (h / 8) + t < d.batch * (h / 8) + d.batch := Nat.add_lt_add_left ht _
  have h2 : d.batch * (h / 8) + d.batch = d.batch * (h / 8 + 1) := by
    rw [Nat.mul_add, Nat.mul_one]
  have h3 : d.batch * (h / 8 + 1) ≤ d.batch * ((d.H + 7) / 8) := Nat.mul_le_mul_left _ hq
  exact lt_of_lt_of_le (h1.trans_eq h2) h3

/-- Program `batch * b + t` handles token `t`. -/
theorem post_tok {d : Inst} {t b : Nat} (ht : t < d.batch) :
    (d.batch * b + t) % d.batch = t := by
  rw [Nat.mul_add_mod, Nat.mod_eq_of_lt ht]

/-- Program `batch * b + t` handles head block `b`. -/
theorem post_blk {d : Inst} {t b : Nat} (ht : t < d.batch) (hb : 0 < d.batch) :
    (d.batch * b + t) / d.batch = b := by
  rw [Nat.mul_add_div hb, Nat.div_eq_of_lt ht, Nat.add_zero]

/-- Lane `h % 8` of program `batch * (h / 8) + t` is head `h`. -/
theorem post_hidx {d : Inst} {t h : Nat} (ht : t < d.batch) (hb : 0 < d.batch) :
    hidx d (d.batch * (h / 8) + t) (h % 8) = h := by
  simp only [hidx, post_blk ht hb]
  exact Nat.div_add_mod' h 8

/-- That lane is enabled when `h < num_heads`. -/
theorem post_en {d : Inst} {t h : Nat} (ht : t < d.batch) (hb : 0 < d.batch) (hh : h < d.H) :
    en d (d.batch * (h / 8) + t) (h % 8) = true := by
  simp only [en, post_hidx ht hb]
  exact decide_eq_true hh

/-- **From the local postcondition to one lane's `g` clause of `Post`.** -/
theorem post_head {d : Inst} {init fin : Mem} {t h : Nat} (ht : t < d.batch) (hh : h < d.H)
    (hloc : ∀ i : Fin (launch d).size, ∃ m,
      Q d init i m ∧ ∀ x ∈ Rset d i ∪ Wset d i, fin x = m x) :
    ∃ A : F32, init.read .f32 (d.A_log.ptr h).addr = some A ∧
    ∃ av : BF16, init.read .bf16 (d.a.ptr (t * d.stride_a + h)).addr = some av ∧
    ∃ bv : BF16, init.read .bf16 (d.dt_bias.ptr h).addr = some bv ∧
    ∃ c : F32, fin.read .f32 (d.g.ptr (t * d.H + h)).addr = some c ∧
      ((c.isFinite = true ∧
          |c.toReal - gRef A.toReal (av.toReal + bv.toReal)| ≤
            gTol A.toReal (av.toReal + bv.toReal)) ∨
        (c = FloatV.negInf ∧
          (Format.binary32.maxVal : ℝ) ≤
            |gRef A.toReal (av.toReal + bv.toReal)| + gTol A.toReal (av.toReal + bv.toReal))) := by
  have hb : 0 < d.batch := by omega
  let p : Fin (launch d).size := ⟨d.batch * (h / 8) + t, post_prog ht hh⟩
  let l : Fin 8 := ⟨h % 8, by omega⟩
  have hev : en d p.val l.val = true := post_en ht hb hh
  have htok : tok d p.val = t := post_tok ht
  have hid : hidx d p.val l.val = h := post_hidx ht hb
  obtain ⟨m, hmQ, hag⟩ := hloc p
  obtain ⟨A, hA, av, hav, bv, hbv, c, hc, hcl⟩ := hmQ.1 l hev
  have hgH : gH d p.val (lane l) = h := hid
  have hgR : gR d p.val d.stride_a (lane l) = t * d.stride_a + h := by
    show tok d p.val * d.stride_a + hidx d p.val l.val = t * d.stride_a + h
    rw [htok, hid]
  have hgO : gO d p.val (lane l) = t * d.H + h := by
    show tok d p.val * d.H + hidx d p.val l.val = t * d.H + h
    rw [htok, hid]
  refine ⟨A, ?_, av, ?_, bv, ?_, c, ?_, hcl⟩
  · rw [hgH] at hA
    exact hA
  · rw [hgR] at hav
    exact hav
  · rw [hgH] at hbv
    exact hbv
  · have hptr : d.g.ptr (t * d.H + h) = d.g.ptr (gO d p.val (lane l)) := by
      rw [hgO]
      congr 1
    rw [hptr]
    have hread : fin.read .f32 (d.g.ptr (gO d p.val (lane l))).addr =
        m.read .f32 (d.g.ptr (gO d p.val (lane l))).addr :=
      Mem.read_congr (fun x hx =>
        hag x (Or.inr (Or.inl (Buf.span_sub_elems d.g (S := oSet d p)
          (j := gO d p.val (lane l)) ⟨lane l, hev, rfl⟩ hx))))
    exact hread.trans hc

/-- **From the local postcondition to one lane's `beta_output` clause of `Post`.** -/
theorem post_beta {d : Inst} {init fin : Mem} {t h : Nat} (ht : t < d.batch) (hh : h < d.H)
    (hloc : ∀ i : Fin (launch d).size, ∃ m,
      Q d init i m ∧ ∀ x ∈ Rset d i ∪ Wset d i, fin x = m x) :
    ∃ bb : BF16, init.read .bf16 (d.b.ptr (t * d.stride_b + h)).addr = some bb ∧
    ∃ c : F32, fin.read .f32 (d.bout.ptr (t * d.H + h)).addr = some c ∧
      c.isFinite = true ∧ |c.toReal - sigRef bb.toReal| ≤ bTol bb.toReal := by
  have hb : 0 < d.batch := by omega
  let p : Fin (launch d).size := ⟨d.batch * (h / 8) + t, post_prog ht hh⟩
  let l : Fin 8 := ⟨h % 8, by omega⟩
  have hev : en d p.val l.val = true := post_en ht hb hh
  have htok : tok d p.val = t := post_tok ht
  have hid : hidx d p.val l.val = h := post_hidx ht hb
  obtain ⟨m, hmQ, hag⟩ := hloc p
  obtain ⟨bb, hbb, c, hc, hfin, hcl⟩ := hmQ.2 l hev
  have hgR : gR d p.val d.stride_b (lane l) = t * d.stride_b + h := by
    show tok d p.val * d.stride_b + hidx d p.val l.val = t * d.stride_b + h
    rw [htok, hid]
  have hgO : gO d p.val (lane l) = t * d.H + h := by
    show tok d p.val * d.H + hidx d p.val l.val = t * d.H + h
    rw [htok, hid]
  refine ⟨bb, ?_, c, ?_, hfin, hcl⟩
  · rw [hgR] at hbb
    exact hbb
  · have hptr : d.bout.ptr (t * d.H + h) = d.bout.ptr (gO d p.val (lane l)) := by
      rw [hgO]
      congr 1
    rw [hptr]
    have hread : fin.read .f32 (d.bout.ptr (gO d p.val (lane l))).addr =
        m.read .f32 (d.bout.ptr (gO d p.val (lane l))).addr :=
      Mem.read_congr (fun x hx =>
        hag x (Or.inr (Or.inr (Buf.span_sub_elems d.bout (S := oSet d p)
          (j := gO d p.val (lane l)) ⟨lane l, hev, rfl⟩ hx))))
    exact hread.trans hc

/-- **From the local postconditions to `Post`.** -/
theorem hpost_spec {d : Inst} {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m,
      Q d init i m ∧ ∀ x ∈ Rset d i ∪ Wset d i, fin x = m x)
    (hfr : ∀ x, (∀ i, x ∉ Wset d i) → fin x = init x) :
    Post d init fin := by
  refine ⟨?_, ?_, ?_⟩
  · intro t ht h hh
    exact post_head ht hh hloc
  · intro t ht h hh
    exact post_beta ht hh hloc
  · intro z hz
    apply hfr
    intro i hzi
    apply hz
    rcases hzi with h | h
    · exact Or.inl (Buf.elems_mono d.g (elems_mono_flat i) h)
    · exact Or.inr (Buf.elems_mono d.bout (elems_mono_flat i) h)

end Claims.GdnGating
