import Claim
import TVA.Proof
import Proof.Body

/-!
# `chunk_fwd_o_bf16`: what one program owns, reads and leaves

* `Geo`: the scalars a program computes in lines 108–121 (its value block, head, chunk, flat start
  of the sequence and its length), and `Geo.Ok`, how they relate to the chunk `r` of the claim;
* the padded sums of the dots (`qPad`, `kPad`, `hPad`, `SQ`, `MQ`, `SA`, `MA`) the loop's
  invariant follows (a lane past `K` is the masked `+0`);
* `OkO`, `OkA`: the parts of `InRange` one dot needs;
* the footprints `Rs` (every input buffer) and `Wg` (the program's tile of `o`) and the
  post-condition `Qg`.
-/

namespace Claims.ChunkFwdO

open TVA

/-! ## The scalars of a program -/

/-- The scalars of lines 108–121 as naturals: `i_v`, `i_h`, `i_tg`, `i_t`, `bos`, `T`. -/
structure Geo where
  iv : Nat
  ih : Nat
  itg : Nat
  it : Nat
  bos : Nat
  Tn : Nat

namespace Geo

/-- The flat index of the program's first token, `bos + 64·i_t`. -/
def lo (g : Geo) : Nat := g.bos + BT * g.it

/-- Local row `i` of the program's tile is a token of its sequence (`m_t`). -/
def rowOk (g : Geo) (i : Nat) : Prop := g.it * BT + i < g.Tn

/-- Local column `c` of the tile is a value column (`o_v < V`). -/
def colOk (d : Inst) (g : Geo) (c : Nat) : Prop := g.iv * d.bk + c < d.V

instance (g : Geo) (i : Nat) : Decidable (g.rowOk i) :=
  inferInstanceAs (Decidable (g.it * BT + i < g.Tn))

instance (d : Inst) (g : Geo) (c : Nat) : Decidable (g.colOk d c) :=
  inferInstanceAs (Decidable (g.iv * d.bk + c < d.V))

end Geo

/-- The scalars of a program of chunk `r`: `r` is a chunk of the batch, `i_tg = r`, `bos + 64·i_t`
is the chunk's first token, and its tokens end where the sequence does; the program's value block
and head exist; the sequence lies in the batch. -/
structure Geo.Ok (d : Inst) (g : Geo) (r : Nat) : Prop where
  r_lt : r < nRows d
  itg_eq : g.itg = r
  lo_eq : rowLo d r = g.lo
  hi_eq : rowHi d r = min (g.lo + BT) (g.bos + g.Tn)
  start_lt : g.it * BT < g.Tn
  ih_lt : g.ih < d.HV
  iv_lt : g.iv * d.bk < d.V
  end_le : g.bos + g.Tn ≤ d.B * d.T

/-! ## Padded sums -/

/-- `q[t, qh j, x]` for `x < K`, and the masked lane's `+0` past `K`. -/
def qPad (d : Inst) (t j x : Nat) : ℚ := if x < d.K then qAt d t (qh d j) x else 0

/-- `k[s, qh j, x]` for `x < K`, and `+0` past `K`. -/
def kPad (d : Inst) (s j x : Nat) : ℚ := if x < d.K then kAt d s (qh d j) x else 0

/-- `S_r[j, x, e]` for `x < K`, and `+0` past `K`. -/
def hPad (d : Inst) (r j e x : Nat) : ℚ := if x < d.K then hAt d r j x e else 0

/-- The first `n` terms of the inter-chunk dot. -/
def SQ (d : Inst) (r t j e n : Nat) : ℚ :=
  ∑ x ∈ Finset.range n, qPad d t j x * hPad d r j e x

/-- The first `n` terms' lifted magnitude. -/
def MQ (d : Inst) (r t j e n : Nat) : ℚ :=
  ∑ x ∈ Finset.range n, L (qPad d t j x) * L (hPad d r j e x)

/-- The first `n` terms of the score `q[t]·k[s]`. -/
def SA (d : Inst) (t s j n : Nat) : ℚ :=
  ∑ x ∈ Finset.range n, qPad d t j x * kPad d s j x

/-- The first `n` terms' lifted magnitude. -/
def MA (d : Inst) (t s j n : Nat) : ℚ :=
  ∑ x ∈ Finset.range n, L (qPad d t j x) * L (kPad d s j x)

/-- The per-trip relative constant of the loop's dots: `relDot16 BK`. -/
def rhoK (d : Inst) : ℚ := Device.relDot16 d.bk

/-- The per-trip absolute constant of the loop's dots: `absDot16 BK`. -/
def alphaK (d : Inst) : ℚ := Device.absDot16 d.bk

/-- The number of trips of the loop, `cdiv(K, BK)`. -/
def nK (d : Inst) : Nat := (d.K + (d.bk - 1)) / d.bk

/-! ## The parts of `InRange` a dot needs -/

/-- The inter-chunk dot of element `(t, j, e)` reads finite values and has room. -/
def OkO (d : Inst) (r t j e : Nat) : Prop :=
  (∀ x, x < d.K → (d.qv ((t * d.H + qh d j) * d.K + x)).isFinite = true ∧
      (d.hv (hIdx d r j x e)).isFinite = true) ∧ PQS d r t j e ≤ 2 ^ 100

/-- The score of tokens `t ≥ s` reads finite values and has room. -/
def OkA (d : Inst) (t s j : Nat) : Prop :=
  (∀ x, x < d.K → (d.qv ((t * d.H + qh d j) * d.K + x)).isFinite = true ∧
      (d.kv ((s * d.H + qh d j) * d.K + x)).isFinite = true) ∧ PA d t s j ≤ 2 ^ 100

/-! ## Footprints and post-condition -/

/-- Every program reads (at most) the input buffers. -/
def Rs (d : Inst) : Set Int :=
  {x | x ∈ d.q.bytes ∨ x ∈ d.k.bytes ∨ x ∈ d.v.bytes ∨ x ∈ d.h.bytes ∨
    (d.useG = true ∧ x ∈ d.g.bytes) ∨ (d.useGamma = true ∧ x ∈ d.gg.bytes) ∨
    (d.varlen = true ∧ (x ∈ d.cu.bytes ∨ x ∈ d.ci.bytes))}

/-- The elements of `o` a program stores: its valid rows and value columns. -/
def tileS (d : Inst) (g : Geo) : Set Nat :=
  {x | ∃ i c, i < BT ∧ c < d.bk ∧ g.rowOk i ∧ g.colOk d c ∧
    x = oIdx d (g.lo + i) g.ih (g.iv * d.bk + c)}

/-- The bytes a program writes. -/
def Wg (d : Inst) (g : Geo) : Set Int := d.o.elems (tileS d g)

/-- What a program of chunk `r` leaves: each element of its tile holds a `bfloat16` value that is
finite and within `tol` of `ref` when the element is `InRange` (the claim's `Post`). -/
def Qg (d : Inst) (g : Geo) (r : Nat) (m : Mem) : Prop :=
  ∀ i c : Nat, i < BT → c < d.bk → g.rowOk i → g.colOk d c →
    ∃ cv : BF16, m.read .bf16 (d.o.ptr (oIdx d (g.lo + i) g.ih (g.iv * d.bk + c))).addr = some cv ∧
      (InRange d r (g.lo + i) g.ih (g.iv * d.bk + c) → cv.isFinite = true ∧
        |cv.toReal - ref d r (g.lo + i) g.ih (g.iv * d.bk + c)| ≤
          tol d r (g.lo + i) g.ih (g.iv * d.bk + c))

/-! ## The parts of `InRange` that do not depend on the column -/

/-- The conditions on row `t` (value head `j`) that `InRange` makes for every column: the query row
and the keys up to `t` are finite, the gates are in range, and the scores have room. -/
def RowIn (d : Inst) (r t j : Nat) : Prop :=
  (∀ x, x < d.K → (d.qv ((t * d.H + qh d j) * d.K + x)).isFinite = true) ∧
    (∀ s ∈ upTo d r t, ∀ x, x < d.K → (d.kv ((s * d.H + qh d j) * d.K + x)).isFinite = true) ∧
    (d.useGamma = true → (d.ggv j).isFinite = true ∧ |gam d j| ≤ 64 ∧ gam d j ≤ 1) ∧
    (d.useG = true → gAt d t j ≤ 64 ∧ ∀ s ∈ upTo d r t,
      (d.gv (s * d.HV + j)).isFinite = true ∧ |gAt d s j| ≤ 2 ^ 64 ∧ gAt d t j - gAt d s j ≤ 64) ∧
    (∀ s ∈ upTo d r t, PA d t s j ≤ 2 ^ 100 ∧ wgt d r t s j * (PA d t s j : ℝ) ≤ 2 ^ 100)

/-- `InRange` gives the column-independent conditions. -/
theorem InRange.rowIn {d : Inst} {r t j e : Nat} (h : InRange d r t j e) : RowIn d r t j :=
  ⟨fun x hx => (h.1 x hx).1, h.2.1, h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2.2.2.1⟩

/-- `InRange` gives the inter-chunk dot's conditions. -/
theorem InRange.okO {d : Inst} {r t j e : Nat} (h : InRange d r t j e) : OkO d r t j e :=
  ⟨h.1, h.2.2.2.2.2.1⟩

/-- A row's conditions give the score of any earlier key its conditions. -/
theorem RowIn.okA {d : Inst} {r t j s : Nat} (h : RowIn d r t j) (hs : s ∈ upTo d r t) :
    OkA d t s j :=
  ⟨fun x hx => ⟨h.1 x hx, h.2.1 s hs x hx⟩, (h.2.2.2.2 s hs).1⟩

end Claims.ChunkFwdO
