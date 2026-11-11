import Claim
import TVA.Proof

/-!
# Programs, footprints and the post-condition

Program `i` of the grid `(g0, B·HV, 1)` is the chunk `it` of a sequence (first flat token `s0`,
`len` tokens) and the value head `h`: it reads the input buffers, and writes the elements of `u`
and `w` of the tokens `s0 + it·64 + r` (`r < 64`, inside the sequence) of head `h`. `Q` says every
such token meets `TokOk`.
-/

namespace Claims.RecomputeWU

open TVA

/-! ## The programs -/

/-- Program `i`'s axis-0 coordinate: the chunk of the padded grid, the row of `chunk_indices` of
the packed one. -/
def p0Of (d : Inst) (i : Nat) : Nat := i % g0 d

/-- Program `i`'s axis-1 coordinate `b·HV + h`. -/
def bhOf (d : Inst) (i : Nat) : Nat := i / g0 d % (d.B * d.HV)

/-- The value head of program `i`. -/
def hOf (d : Inst) (i : Nat) : Nat := bhOf d i % d.HV

/-- The batch row of program `i` (`0` when packed). -/
def bOf (d : Inst) (i : Nat) : Nat := bhOf d i / d.HV

/-- The chunk of program `i` within its sequence. -/
def itOf (d : Inst) (init : Mem) (i : Nat) : Nat :=
  if d.isVarlen then chChunk d init (p0Of d i) else p0Of d i

/-- The first flat token of program `i`'s sequence. -/
def s0Of (d : Inst) (init : Mem) (i : Nat) : Nat :=
  if d.isVarlen then cuN d init (chSeq d init (p0Of d i)) else bOf d i * d.T

/-- The length of program `i`'s sequence. -/
def lenOf (d : Inst) (init : Mem) (i : Nat) : Nat :=
  if d.isVarlen then seqLen d init (chSeq d init (p0Of d i)) else d.T

/-- Whether row `r` of chunk `it` is a token of the sequence (of length `len`). -/
def rowEn (it len r : Nat) : Bool := decide (it * 64 + r < len)

/-- Whether column `c` of block `t` (of width `64`) is below `n`. -/
def colEn (t n c : Nat) : Bool := decide (t * 64 + c < n)

/-! ## Footprints -/

/-- Every program reads (within) the input buffers. -/
def Rall (d : Inst) : Set Int :=
  d.k.bytes ∪ d.v.bytes ∪ d.beta.bytes ∪ d.betab.bytes ∪ d.A.bytes ∪ d.g.bytes ∪ d.cu.bytes ∪
    d.ci.bytes

/-- The elements of `u` that the chunk `it` of the sequence at `s0` of length `len` writes, head
`h`. -/
def UE (d : Inst) (s0 len it h : Nat) : Set Nat :=
  {x | ∃ r c, r < 64 ∧ it * 64 + r < len ∧ c < d.V ∧ x = vIdx d (s0 + it * 64 + r) h c}

/-- The elements of `w` that the chunk writes. -/
def WE (d : Inst) (s0 len it h : Nat) : Set Nat :=
  {x | ∃ r c, r < 64 ∧ it * 64 + r < len ∧ c < d.K ∧ x = wIdx d (s0 + it * 64 + r) h c}

/-- The bytes program `i` writes. -/
def Wf (d : Inst) (init : Mem) (i : Nat) : Set Int :=
  d.u.elems (UE d (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i)) ∪
    d.w.elems (WE d (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i))

/-- What program `i` leaves: every token of its chunk meets `TokOk`. -/
def Qf (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  ∀ r, r < 64 → itOf d init i * 64 + r < lenOf d init i →
    TokOk d init m (s0Of d init i) (lenOf d init i) (itOf d init i * 64 + r) (hOf d i)

/-- The memory agrees with `init` on the inputs. -/
def Ag (d : Inst) (init m : Mem) : Prop := ∀ x ∈ Rall d, m x = init x

/-- The output buffers are present in `m`. -/
def Pres (d : Inst) (m : Mem) : Prop := d.u.Valid m ∧ d.w.Valid m

/-! ## Element indices -/

/-- `(f·X + h)·Y + c` is below `n·X·Y` when `f < n`, `h < X`, `c < Y`. -/
theorem idx_lt {f X h Y c n : Nat} (hf : f < n) (hh : h < X) (hc : c < Y) :
    (f * X + h) * Y + c < n * X * Y := by
  have h1 : f * X + h < n * X := by
    have : (f + 1) * X ≤ n * X := Nat.mul_le_mul_right X hf
    nlinarith
  have h2 : (f * X + h + 1) * Y ≤ n * X * Y := Nat.mul_le_mul_right Y h1
  nlinarith

/-- `(f·X + h)·Y + c` determines `f`, `h`, `c` for `h < X`, `c < Y`. -/
theorem idx_inj {f f' X h h' Y c c' : Nat} (hh : h < X) (hh' : h' < X) (hc : c < Y) (hc' : c' < Y)
    (e : (f * X + h) * Y + c = (f' * X + h') * Y + c') : f = f' ∧ h = h' ∧ c = c' := by
  have hY : 0 < Y := by omega
  have hX : 0 < X := by omega
  have e1 : ((f * X + h) * Y + c) / Y = f * X + h := by
    rw [Nat.mul_comm (f * X + h) Y, Nat.mul_add_div hY, Nat.div_eq_of_lt hc]; simp
  have e2 : ((f' * X + h') * Y + c') / Y = f' * X + h' := by
    rw [Nat.mul_comm (f' * X + h') Y, Nat.mul_add_div hY, Nat.div_eq_of_lt hc']; simp
  have e3 : ((f * X + h) * Y + c) % Y = c := by
    rw [Nat.mul_comm (f * X + h) Y, Nat.mul_add_mod]; exact Nat.mod_eq_of_lt hc
  have e4 : ((f' * X + h') * Y + c') % Y = c' := by
    rw [Nat.mul_comm (f' * X + h') Y, Nat.mul_add_mod]; exact Nat.mod_eq_of_lt hc'
  rw [e] at e1 e3
  have hc2 : c = c' := by rw [← e3, e4]
  have hfh : f * X + h = f' * X + h' := by rw [← e1, e2]
  have e5 : (f * X + h) / X = f := by
    rw [Nat.mul_comm f X, Nat.mul_add_div hX, Nat.div_eq_of_lt hh]; simp
  have e6 : (f' * X + h') / X = f' := by
    rw [Nat.mul_comm f' X, Nat.mul_add_div hX, Nat.div_eq_of_lt hh']; simp
  have e7 : (f * X + h) % X = h := by
    rw [Nat.mul_comm f X, Nat.mul_add_mod]; exact Nat.mod_eq_of_lt hh
  have e8 : (f' * X + h') % X = h' := by
    rw [Nat.mul_comm f' X, Nat.mul_add_mod]; exact Nat.mod_eq_of_lt hh'
  rw [hfh] at e5 e7
  exact ⟨by rw [← e5, e6], by rw [← e7, e8], hc2⟩


/-! ## The trips' tiles -/

/-- The element of `u` at the base of the sequence and head: `(s0·HV + h)·V`. -/
def eU (d : Inst) (s0 h : Nat) : Nat := (s0 * d.HV + h) * d.V

/-- The element of `w` at the base of the sequence and head: `(s0·HV + h)·K`. -/
def eW (d : Inst) (s0 h : Nat) : Nat := (s0 * d.HV + h) * d.K

/-- The bytes of `u` that trip `t` of the `u` loop writes. -/
def tileU (d : Inst) (s0 len it h t : Nat) : Set Int :=
  d.u.tileAt (eU d s0 h) (d.HV * d.V) 1 (fun i : Fin 64 => it * 64 + i.val)
    (fun j : Fin 64 => t * 64 + j.val) len d.V

/-- The bytes of `w` that trip `t` of the `w` loop writes. -/
def tileW (d : Inst) (s0 len it h t : Nat) : Set Int :=
  d.w.tileAt (eW d s0 h) (d.HV * d.K) 1 (fun i : Fin 64 => it * 64 + i.val)
    (fun j : Fin 64 => t * 64 + j.val) len d.K

/-- What trip `t` of the `u` loop leaves: the cells of its columns. -/
def UNew (d : Inst) (init : Mem) (s0 len it h t : Nat) (m : Mem) : Prop :=
  ∀ i j : Fin 64, rowEn it len i = true → colEn t d.V j = true →
    CellU d init m s0 len (it * 64 + i) h (t * 64 + j)

/-- What trip `t` of the `w` loop leaves. -/
def WNew (d : Inst) (init : Mem) (s0 len it h t : Nat) (m : Mem) : Prop :=
  ∀ i j : Fin 64, rowEn it len i = true → colEn t d.K j = true →
    CellW d init m s0 len (it * 64 + i) h (t * 64 + j)

end Claims.RecomputeWU
