import Claim
import Proof.Flt1
import Proof.Final

/-!
# The program's footprints, its post-condition, and the body of the kernel

Program `(s, h)` of the grid reads the query row `q[s, h, :]`, `seq_lens[s]`, the block-table
entries of sequence `s` it visits and the cache elements of the sequence's tokens at key/value
head `h // QUERIES_PER_KV`, and writes `out[s, h, :]`.
-/
namespace Claims.PagedAttention

open TVA

/-- The elements of `q` that program `(s, h)` reads: its query row. -/
def qSet (d : Inst) (s h : Nat) : Set Nat := {x | ∃ e, e < d.HEAD_DIM ∧ x = qIdx d s h e}

/-- The elements of `block_tables` it reads: the entries of the pages the sequence spans. -/
def btSet (d : Inst) (m : Mem) (s : Nat) : Set Nat :=
  {x | ∃ j, j * d.PAGE_SIZE < seqLen d m s ∧ x = btIdx d s j}

/-- The elements of `k_cache` (and of `v_cache`) it reads: those of the sequence's tokens. -/
def tokSet (d : Inst) (m : Mem) (s h : Nat) : Set Nat :=
  {x | ∃ t e, t < seqLen d m s ∧ e < d.HEAD_DIM ∧ x = tokElem d m s h t e}

/-- The elements of `out` it writes: its output row. -/
def oSet (d : Inst) (s h : Nat) : Set Nat := {x | ∃ e, e < d.HEAD_DIM ∧ x = outIdx d s h e}

/-- **The read footprint** of program `(s, h)`. -/
def R (d : Inst) (init : Mem) (s h : Nat) : Set Int :=
  d.q.elems (qSet d s h) ∪ d.seq_lens.elems {s} ∪ d.block_tables.elems (btSet d init s) ∪
    d.k_cache.elems (tokSet d init s h) ∪ d.v_cache.elems (tokSet d init s h)

/-- **The write footprint** of program `(s, h)`. -/
def W (d : Inst) (s h : Nat) : Set Int := d.out.elems (oSet d s h)

/-- **What program `(s, h)` leaves**: each element of its output row holds a `float16` value,
and, when the sequence's values are covered, one within the claim's tolerance of the exact
attention. -/
def Q (d : Inst) (init : Mem) (s h : Nat) (m : Mem) : Prop :=
  ∀ e, e < d.HEAD_DIM → ∃ c : F16, m.read .f16 (d.out.ptr (outIdx d s h e : Int)).addr = some c ∧
    (SeqVals d init s h → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
      (∀ t, t < seqLen d init s → sigma d * scoreAbs d init s h t ≤ X) →
      |c.toReal - attnRef d init s h e| ≤
        (tolA (nblocks d init s) X : ℝ) * (attnAbs d init s h e + |attnRef d init s h e|) +
          1 / 2 ^ 11 * |attnRef d init s h e| + 1 / 2 ^ 24)

/-- The loop body of the kernel (lines 55–69): one page. -/
def trip (d : Inst) (seq kv_head seq_len : I32) (q : Tensor [d.HEAD_DIM] F32) (j : I32)
    (st : Tensor [d.HEAD_DIM] F32 × F32 × F32) :
    Prog (Tensor [d.HEAD_DIM] F32 × F32 × F32) :=
  match st with
  | (acc, l_i, m_i) => do
    let offs_t : Tensor [d.PAGE_SIZE] I32 := tl.arange 0 d.PAGE_SIZE
    let offs_d : Tensor [d.HEAD_DIM] I32 := tl.arange 0 d.HEAD_DIM
    let page := (← tl.convert .i32 .i64 (Tensor.scalar (← tl.load1 (d.block_tables.ptr 0 + seq * (IntV.ofNat d.stride_bs : I32) + j)))).item
    let mask := (j * (IntV.ofNat d.PAGE_SIZE : I32) + offs_t).ltS seq_len
    let kv_offs : Tensor [d.PAGE_SIZE, d.HEAD_DIM] I64 :=
      page * (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.stride_kp : I32))).item
        + (← tl.convert .i32 .i64 (offs_t.col d.HEAD_DIM * (IntV.ofNat d.stride_kt : I32)))
        + (← tl.convert .i32 .i64 (Tensor.scalar (kv_head * (IntV.ofNat d.stride_kh : I32)))).item
        + (← tl.convert .i32 .i64 (offs_d.row d.PAGE_SIZE))
    let k : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F32 ← tl.cvt (← tl.load (d.k_cache.ptr 0 + kv_offs) (some (mask.col d.HEAD_DIM))
      (some (← tl.convert .f32 .f16 (tl.full [d.PAGE_SIZE, d.HEAD_DIM] (FloatV.litF32 (FloatV.ofPat 0))))))
    let s ← tl.mul (← tl.sum1c (← tl.mul k (q.row d.PAGE_SIZE))) (bc (FloatV.hostF32 d.scale))
    let s := tl.where mask s (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)))
    let m_new := (← tl.maximum (Tensor.scalar m_i) (Tensor.scalar (← tl.maxAll true s))).item
    let alpha := (← tl.exp (← tl.sub (Tensor.scalar m_i) (Tensor.scalar m_new))).item
    let p ← tl.exp (← tl.sub s (bc m_new))
    let l_i := (← tl.mul_add (Tensor.scalar l_i) (Tensor.scalar alpha) (Tensor.scalar (← tl.sum p))).item
    let v : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F32 ← tl.cvt (← tl.load (d.v_cache.ptr 0 + kv_offs) (some (mask.col d.HEAD_DIM))
      (some (← tl.convert .f32 .f16 (tl.full [d.PAGE_SIZE, d.HEAD_DIM] (FloatV.litF32 (FloatV.ofPat 0))))))
    let pv ← tl.sum0c (← tl.mul (p.col d.HEAD_DIM) v)
    let acc ← if d.PAGE_SIZE = 1 then do tl.addc (← tl.mul acc (bc alpha)) pv
      else tl.mul_add acc (bc alpha) pv
    let m_i := m_new
    pure (acc, l_i, m_i)

/-- The kernel's body: the setup, the loop over pages, the division, the store. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c = (do
      let seq := tl.program_id c 0
      let head := tl.program_id c 1
      let kv_head := (← tl.idiv (Tensor.scalar head) (Tensor.scalar (IntV.ofNat (qpk d) : I32))).item
      let seq_len ← tl.load1 (d.seq_lens.ptr 0 + seq)
      let offs_d : Tensor [d.HEAD_DIM] I32 := tl.arange 0 d.HEAD_DIM
      let q : Tensor [d.HEAD_DIM] F32 ← tl.cvt (← tl.load (d.q.ptr 0 + seq * (IntV.ofNat d.stride_qs : I32) + head * (IntV.ofNat d.stride_qh : I32) + offs_d))
      let (acc, l_i, _m_i) ← tl.forRange (IntV.ofInt 0 : I32)
        (← tl.idiv (Tensor.scalar (seq_len + (IntV.ofNat (d.PAGE_SIZE - 1) : I32)))
          (Tensor.scalar (IntV.ofNat d.PAGE_SIZE : I32))).item
        (IntV.ofInt 1 : I32) (tl.zeros [d.HEAD_DIM] .f32, FloatV.litF32 (FloatV.ofPat 0), FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)) (trip d seq kv_head seq_len q)
      let out ← tl.div acc (bc l_i)
      let o16 : Tensor [d.HEAD_DIM] F16 ← tl.cvt out
      tl.store (d.out.ptr 0 + seq * (IntV.ofNat d.stride_os : I32) + head * (IntV.ofNat d.stride_oh : I32) + offs_d) o16) := rfl

/-! ## The loop invariant -/

/-- The loop's initial state `(acc, l_i, m_i)`. -/
def st0 (d : Inst) : Tensor [d.HEAD_DIM] F32 × F32 × F32 :=
  (tl.zeros [d.HEAD_DIM] .f32, FloatV.litF32 (FloatV.ofPat 0),
    FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000))

/-- The exact scaled score `scale · q·k` of token `k` of sequence `s` at head `h`. -/
noncomputable def Sx (d : Inst) (m : Mem) (s h k : Nat) : ℝ := (sigma d : ℝ) * (score d m s h k : ℝ)

/-- Component `e` of the value of token `k`. -/
noncomputable def Vx (d : Inst) (m : Mem) (s h e k : Nat) : ℝ :=
  (val16 m d.v_cache (tokElem d m s h k e) : ℝ)

/-- **The state after `t ≥ 1` pages**, for scores bounded by `X`: the running maximum is a finite
value in `[-(X+1), X+1]`, at most the scores seen plus their error and attained up to it; `l` and
every component of `acc` are finite and within the invariant's error of the exact sums at that
reference. -/
def StPos (d : Inst) (m : Mem) (s h : Nat) (X : ℝ) (t : Nat)
    (st : Tensor [d.HEAD_DIM] F32 × F32 × F32) : Prop :=
  st.2.2.isFinite = true ∧ st.2.1.isFinite = true ∧
    (∀ e (he : e < d.HEAD_DIM), (st.1.get1 ⟨e, he⟩).isFinite = true) ∧
    -(Num.Kx X) ≤ st.2.2.toReal ∧ st.2.2.toReal ≤ Num.Kx X ∧
    (∀ k, k < min (t * d.PAGE_SIZE) (seqLen d m s) → Sx d m s h k ≤ st.2.2.toReal + Num.dS X) ∧
    (∃ k, k < min (t * d.PAGE_SIZE) (seqLen d m s) ∧ st.2.2.toReal ≤ Sx d m s h k + Num.dS X) ∧
    Num.InvL X (Sx d m s h) (min (t * d.PAGE_SIZE) (seqLen d m s)) t st.2.2.toReal st.2.1.toReal ∧
    ∀ e (he : e < d.HEAD_DIM), Num.InvA X (Sx d m s h) (Vx d m s h e)
      (min (t * d.PAGE_SIZE) (seqLen d m s)) t st.2.2.toReal (st.1.get1 ⟨e, he⟩).toReal

/-- **The loop invariant** after `t` pages: the memory is still `init`, and, when the sequence's
values are covered, for every bound `X ≤ 2^10` on the scaled score magnitudes the state is the
initial one (`t = 0`) or satisfies `StPos`. -/
def Inv (d : Inst) (init : Mem) (s h : Nat) (t : Nat)
    (st : Tensor [d.HEAD_DIM] F32 × F32 × F32) (m : Mem) : Prop :=
  m = init ∧ (SeqVals d init s h → ∀ X : ℝ, 0 ≤ X → X ≤ 1024 →
    (∀ k, k < seqLen d init s → (sigma d : ℝ) * (scoreAbs d init s h k : ℝ) ≤ X) →
      (t = 0 → st = st0 d) ∧ (0 < t → StPos d init s h X t st))

end Claims.PagedAttention
