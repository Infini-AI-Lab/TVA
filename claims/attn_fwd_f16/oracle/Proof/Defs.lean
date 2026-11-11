import Claim
import Proof.Final

/-!
# What a program holds: the row data, the state invariant, the footprints
-/

namespace Claims.AttnFwd

open TVA

noncomputable section

/-- The row `n` of head `bh` as a `RowCtx`: natural-log scores `sm_scale · q · k`, values, and the
attended keys (`keys d n`). The values of attended keys are at most `2^15` when the head is
covered (`HeadOk`). -/
def rowCtx (d : Inst) (init : Mem) (bh n : Nat) (h : HeadOk d init bh) : RowCtx where
  D := d.HEAD_DIM
  s := fun k => (sigma d : ℝ) * (score d init bh n k : ℝ)
  v := fun k e => (val16 init d.v (idx d bh k e) : ℝ)
  ok := fun k => decide (k ∈ keys d n)
  hv := fun k e hk he => by
    have hk' : k ∈ keys d n := of_decide_eq_true hk
    have hkN : k < d.N_CTX := by
      simp only [keys, Finset.mem_filter, Finset.mem_range] at hk'; exact hk'.1
    have := h.2.1 k e hkN he
    exact_mod_cast this

/-- The state `(m_i, l_i, acc)` of all the rows of tile `sm` of head `bh` after `t` blocks of keys:
before any block it is the kernel's start (`m_i = −∞`, `l_i = 1`, `acc = 0`); after, every value is
finite and, for every bound `X` on the row's scaled score magnitudes, the invariant of the row. -/
def StateOK (d : Inst) (init : Mem) (bh sm t : Nat) (m_i l_i : Tensor [128] F32)
    (acc : Tensor [128, d.HEAD_DIM] F32) : Prop :=
  ∀ h : HeadOk d init bh, ∀ r : Fin 128,
    (t = 0 ∧ m_i.get1 r = FloatV.negInf ∧ l_i.get1 r = FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) ∧
      ∀ e : Fin d.HEAD_DIM, acc.get2 r e = FloatV.litF32 (FloatV.ofPat 0)) ∨
    (1 ≤ t ∧ (m_i.get1 r).isFinite = true ∧ (l_i.get1 r).isFinite = true ∧
      (∀ e : Fin d.HEAD_DIM, (acc.get2 r e).isFinite = true) ∧
      ∀ X : ℝ, Par d.HEAD_DIM X → (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
        ∃ mf : ℝ, InvM (rowCtx d init bh (sm * 128 + r) h) X t (m_i.get1 r).toReal mf ∧
          InvL (rowCtx d init bh (sm * 128 + r) h) X t (m_i.get1 r).toReal mf (l_i.get1 r).toReal ∧
          InvA (rowCtx d init bh (sm * 128 + r) h) X t (m_i.get1 r).toReal mf
            (fun e => if h : e < d.HEAD_DIM then (acc.get2 r ⟨e, h⟩).toReal else 0))

/-! ## The footprints and the post-condition of a program -/

/-- The number of query tiles of a head, the grid's `x` extent. -/
def G (d : Inst) : Nat := (d.N_CTX + 127) / 128

/-- What every program may read: the elements of `q`, `k`, `v`. -/
def Rset (d : Inst) : Set Int :=
  d.q.elems (below (rows d * d.HEAD_DIM)) ∪ d.k.elems (below (rows d * d.HEAD_DIM)) ∪
    d.v.elems (below (rows d * d.HEAD_DIM))

/-- The elements of `o` in tile `sm` of head `bh`: `128` rows, all columns. -/
def oTile (d : Inst) (bh sm : Nat) : Set Nat :=
  {x | ∃ r, r < 128 ∧ ∃ e, e < d.HEAD_DIM ∧ x = idx d bh (sm * 128 + r) e}

/-- The elements of `M` in tile `sm` of head `bh`. -/
def mTile (d : Inst) (bh sm : Nat) : Set Nat :=
  {x | ∃ r, r < 128 ∧ x = bh * d.N_CTX + (sm * 128 + r)}

/-- What program `(sm, bh)` may write: its tile of `o` and of `M`. -/
def Wset (d : Inst) (bh sm : Nat) : Set Int := d.o.elems (oTile d bh sm) ∪ d.M.elems (mTile d bh sm)

/-- What program `(sm, bh)` leaves in its tile: `Post`'s statement for its elements. -/
def Qpost (d : Inst) (init : Mem) (bh sm : Nat) (m : Mem) : Prop :=
  ∀ r, r < 128 →
    (∀ e, e < d.HEAD_DIM →
      ∃ c : F16, m.read .f16 (d.o.ptr (idx d bh (sm * 128 + r) e : Int)).addr = some c ∧
        (HeadOk d init bh → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
          (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
          |c.toReal - attnRef d init bh (sm * 128 + r) e| ≤
            (epsO d.HEAD_DIM d.N_CTX X : ℝ) * attnMag d init bh (sm * 128 + r) e + 1 / 2 ^ 24)) ∧
    ∃ c : F32, m.read .f32 (d.M.ptr ((bh * d.N_CTX + (sm * 128 + r) : Nat) : Int)).addr = some c ∧
      (HeadOk d init bh → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
        (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
        |c.toReal - lseRef d init bh (sm * 128 + r)| ≤ (epsM d.HEAD_DIM X : ℝ))

end

end Claims.AttnFwd
