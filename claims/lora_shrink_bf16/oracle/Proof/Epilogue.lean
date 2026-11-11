import Proof.Defs
import Proof.Facts
import Proof.Foot
import Proof.Lanes
import Proof.Wrap
import Proof.Trip
import Proof.Numeric

/-!
# `lora_shrink_bf16`: the end of a program (lines 293-305)

The accumulator is scaled by `scaling` and added into the output with a relaxed `tl.atomic_add`,
masked to the program's rows. This file shows the program `SafeA` there, leaving the log `Q`.
-/

namespace Claims.LoraShrink

open TVA

/-- The end of the kernel (lines 300-305), for the output pointer `curOut` of the slice (line 286),
the token indices `ram` of the tile (line 388) and the accumulator `acc`; `SPLIT_K ≠ 1` takes the
atomic branch. -/
def epilogue (d : Inst) (i : Nat) (curOut : Ptr .f32) (ram : Tensor [32 - 0] I32)
    (acc : Tensor [32, 16] F32) : Prog Unit := do
  let accumulator ← tl.mul acc (bc (scaling32 d))
  if sk d = 1 then do
    tl.store
      (curOut + ram.col 16 * (IntV.ofNat d.R : I32) +
        (tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).row 32 * (IntV.ofNat 1 : I32))
      accumulator
      (some ((((tl.arange 0 32).col 16).ltS (IntV.ofNat (mlen d i) : I32)).and
        (((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).row 32).ltS (IntV.ofNat d.R : I32))))
    Prog.ret ()
  else do
    tl.atomic_add_unused
      (curOut + ram.col 16 * (IntV.ofNat d.R : I32) +
        (tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).row 32 * (IntV.ofNat 1 : I32))
      accumulator
      (some ((((tl.arange 0 32).col 16).ltS (IntV.ofNat (mlen d i) : I32)).and
        (((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).row 32).ltS (IntV.ofNat d.R : I32))))
      MemSem.relaxed
    Prog.ret ()

/-- The filter-map of a guarded value is the map of the filter. -/
theorem filterMap_guard {α β : Type} (p : α → Prop) [DecidablePred p] (w : α → β) (l : List α) :
    l.filterMap (fun a => if p a then some (w a) else none) = (l.filter (fun a => decide (p a))).map w := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    by_cases h : p a <;> simp [h, ih]

/-- A duplicate-free list in which exactly `x` satisfies `p` filters to `[x]`. -/
theorem filter_single {α : Type} (p : α → Bool) (l : List α) (hl : l.Nodup) (x : α) (hx : x ∈ l)
    (hpx : p x = true) (huniq : ∀ y ∈ l, p y = true → y = x) : l.filter p = [x] := by
  induction l with
  | nil => exact absurd hx (by simp)
  | cons a l ih =>
    rw [List.nodup_cons] at hl
    by_cases ha : p a = true
    · have hax : a = x := huniq a List.mem_cons_self ha
      subst hax
      simp only [List.filter_cons, ha, ite_true]
      have : l.filter p = [] := by
        rw [List.filter_eq_nil_iff]
        intro y hy hpy
        have := huniq y (List.mem_cons_of_mem _ hy) hpy
        exact hl.1 (this ▸ hy)
      rw [this]
    · have hax : a ≠ x := fun h => ha (h ▸ hpx)
      have hx' : x ∈ l := (List.mem_cons.mp hx).resolve_left (Ne.symm hax)
      simp only [List.filter_cons, ha, Bool.false_eq_true, ite_false]
      exact ih hl.2 hx' (fun y hy hpy => huniq y (List.mem_cons_of_mem _ hy) hpy)


/-- The cell of lane `(r, n)` of the output tile: the output element of its token and column. -/
def laneCell (d : Inst) (i : Nat) (r n : Nat) : Nat × Nat × Nat :=
  (slice d i, tokOf d i r, 16 * tileN d i + n)

section QAtomic

variable {P : Platform} {d : Inst} {init : Mem} {i : Nat}

/-- **The log after the atomic**: the lanes of the tile in any order add their values into their
cells. `hptr`/`hmask`: the tile's pointers and mask; `hvals`: each enabled lane's value is a
`PartOk` operand whenever the data are `Dat`. -/
theorem Q_atomic (hP : Device P) (hd : d.ok) (hact : Active d i)
    {ptrs : Tensor [32, 16] (Ptr .f32)} {mask : Tensor [32, 16] Bool} {vals : Tensor [32, 16] F32}
    (hptr : ∀ (r : Fin 32) (n : Fin 16), (r : Nat) < mlen d i →
      ptrs.get2 r n = d.out.ptr (outAt d (slice d i) (tokOf d i r) (16 * tileN d i + n) : Nat))
    (hmask : ∀ (r : Fin 32) (n : Fin 16), mask.get2 r n = true ↔ (r : Nat) < mlen d i)
    (hvals : ∀ (r : Fin 32) (n : Fin 16), (r : Nat) < mlen d i → ∀ va vb : Nat → BF16,
      Dat d init (slice d i) (tokOf d i r) (16 * tileN d i + n) va vb →
        PartOk d va vb (part d i) (vals.get2 r n))
    (_hs : slice d i < d.S) (hi : i < (launch d).size) {m : Mem}
    {order : List (Idx [32, 16])} (horder : order.Perm (enabled (tl.addrs ptrs (some mask))))
    {olds news : Idx [32, 16] → F32}
    (hupd : ∀ idx ∈ order, tl.addUpd ((launch d).env P) .f32 (vals.get idx) (olds idx) (news idx)) :
    Q d init i ([] ++ order.map fun idx => (⟨.f32, (ptrs.get idx).addr, olds idx, news idx⟩ : AtomicStep)) m := by
  classical
  have hmlen := hact.mlen_pos
  have hmlen32 := mlen_le d i
  have hmem' : ∀ idx : Idx [32, 16], idx ∈ order ↔ (idx.1 : Nat) < mlen d i := by
    intro idx
    rw [horder.mem_iff]
    constructor
    · intro h
      have h1 := (List.mem_filter.mp h).2
      unfold tl.addrs at h1
      by_cases hm : mask.get idx = true
      · exact (hmask idx.1 idx.2.1).mp hm
      · simp [hm] at h1
    · intro hr
      have hm : mask.get idx = true := (hmask idx.1 idx.2.1).mpr hr
      exact mem_enabled (a := (ptrs.get idx).addr) (by unfold tl.addrs; simp [hm])
  have hmem : ∀ (r : Fin 32) (n : Fin 16), ((r, n, ()) : Idx [32, 16]) ∈ order ↔ (r : Nat) < mlen d i :=
    fun r n => hmem' (r, n, ())
  have haddr : ∀ (r : Fin 32) (n : Fin 16), (r : Nat) < mlen d i →
      (ptrs.get2 r n).addr = cellAddr d (slice d i) (tokOf d i r) (16 * tileN d i + n) := by
    intro r n hr
    rw [hptr r n hr]
    rfl
  have hrange : ∀ (r : Fin 32) (n : Fin 16), (r : Nat) < mlen d i →
      tokOf d i r < d.M ∧ 16 * tileN d i + n < d.R := fun r n hr =>
    ⟨(hact.row hd hr).1, tileN_lt hd hi n.isLt⟩
  -- the operand of a lane, for a cell
  let match' : Nat → Nat → Nat → Idx [32, 16] → Prop := fun s t j idx =>
    (slice d i, tokOf d i idx.1, 16 * tileN d i + idx.2.1) = (s, t, j)
  let op : Nat → Nat → Nat → Idx [32, 16] → Option ℚ := fun s t j idx =>
    if match' s t j idx then some (vals.get idx).toRat else none
  have hnodup : order.Nodup := horder.nodup_iff.mpr (enabled_nodup _)
  refine ⟨?_, fun s t j => ((order.filterMap (op s t j) : List ℚ) : Multiset ℚ), ?_, ?_, ?_⟩
  · intro st hst
    simp only [List.nil_append, List.mem_map] at hst
    obtain ⟨⟨r, n, u⟩, hidx, rfl⟩ := hst
    cases u
    have hr := (hmem r n).mp hidx
    exact ⟨slice d i, tokOf d i r, 16 * tileN d i + n,
      ⟨hact, rfl, ⟨r, hr, rfl⟩, by omega, by omega⟩, rfl, haddr r n hr⟩
  · intro s t j hs ht hj va vb hD
    rw [List.nil_append]
    refine Paired.lanes (R := AccE d s t j) order _ (op s t j) ?_
    rintro ⟨r, n, u⟩ hidx
    cases u
    have hr := (hmem r n).mp hidx
    by_cases hmt : match' s t j (r, n, ())
    · have hop : op s t j (r, n, ()) = some (vals.get2 r n).toRat := by simp only [op, hmt, ite_true]; rfl
      rw [hop]
      have hmt' : slice d i = s ∧ tokOf d i r = t ∧ 16 * tileN d i + n = j := by
        have := Prod.mk.inj hmt
        exact ⟨this.1, (Prod.mk.inj this.2).1, (Prod.mk.inj this.2).2⟩
      obtain ⟨h1, h2, h3⟩ := hmt'
      have ha := haddr r n hr
      rw [h1, h2, h3] at ha
      have hv := (hvals r n hr va vb (by rw [h1, h2, h3]; exact hD)).1
      have hnew := hupd (r, n, ()) hidx
      have := accStep_atomicAdd (a := cellAddr d s t j) hP.atomicAddF32 hnew hv
      show AccStep .f32 (cellAddr d s t j) FloatV.toRat _ _ _ _ _ (some (vals.get2 r n).toRat)
      have e : (ptrs.get (r, n, ())).addr = cellAddr d s t j := ha
      simp only [e]
      exact this
    · have hop : op s t j (r, n, ()) = none := by simp only [op, hmt, ite_false]
      rw [hop]
      refine accStep_other ?_
      obtain ⟨hrt, hrj⟩ := hrange r n hr
      have ha := haddr r n hr
      have e : (ptrs.get (r, n, ())).addr = cellAddr d (slice d i) (tokOf d i r) (16 * tileN d i + n) := ha
      show Disjoint (Mem.span .f32 (ptrs.get (r, n, ())).addr) (Mem.span .f32 (cellAddr d s t j))
      rw [e]
      exact cell_disjoint d ht hj hrt hrj (fun h => hmt (by
        obtain ⟨h1, h2, h3⟩ := h
        show (slice d i, tokOf d i r, 16 * tileN d i + n) = (s, t, j)
        rw [h1, h2, h3]))
  · intro s t j hnt
    show ((order.filterMap (op s t j) : List ℚ) : Multiset ℚ) = 0
    have : order.filterMap (op s t j) = [] := by
      rw [List.filterMap_eq_nil_iff]
      rintro ⟨r, n, u⟩ hidx
      cases u
      have hr := (hmem r n).mp hidx
      by_cases hmt : match' s t j (r, n, ())
      · exfalso
        apply hnt
        have := Prod.mk.inj hmt
        obtain ⟨h1, h2, h3⟩ : slice d i = s ∧ tokOf d i r = t ∧ 16 * tileN d i + n = j :=
          ⟨this.1, (Prod.mk.inj this.2).1, (Prod.mk.inj this.2).2⟩
        exact ⟨hact, h1.symm, ⟨r, hr, h2.symm⟩, by omega, by omega⟩
      · simp only [op, hmt, ite_false]
    rw [this]
    rfl
  · intro s t j htgt
    obtain ⟨-, hsl, ⟨m, hm, htm⟩, hj1, hj2⟩ := htgt
    have hm32 : m < 32 := by omega
    have hn16 : j - 16 * tileN d i < 16 := by omega
    let r : Fin 32 := ⟨m, hm32⟩
    let n : Fin 16 := ⟨j - 16 * tileN d i, hn16⟩
    refine ⟨vals.get2 r n, ?_, fun va vb hD => ?_⟩
    · show ((order.filterMap (op s t j) : List ℚ) : Multiset ℚ) = {(vals.get2 r n).toRat}
      have hx : ((r, n, ()) : Idx [32, 16]) ∈ order := (hmem r n).mpr hm
      have hpx : (decide (match' s t j (r, n, ()))) = true := by
        apply decide_eq_true
        show (slice d i, tokOf d i m, 16 * tileN d i + (j - 16 * tileN d i)) = (s, t, j)
        rw [← hsl, ← htm]
        congr 2
        omega
      have hfil := filter_single (fun idx => decide (match' s t j idx)) order hnodup (r, n, ()) hx hpx
        (by
          rintro ⟨r', n', u'⟩ hy hpy
          cases u'
          have hr' := (hmem r' n').mp hy
          have hm' : (slice d i, tokOf d i r', 16 * tileN d i + n') = (s, t, j) :=
            of_decide_eq_true hpy
          have h2 : tokOf d i r' = t := (Prod.mk.inj (Prod.mk.inj hm').2).1
          have h3 : 16 * tileN d i + (n' : Nat) = j := (Prod.mk.inj (Prod.mk.inj hm').2).2
          have e1 : r' = r := by
            apply Fin.ext
            exact hact.tok_inj hd hr' hm (by rw [h2, htm])
          have e2 : n' = n := by
            apply Fin.ext
            show (n' : Nat) = j - 16 * tileN d i
            omega
          rw [e1, e2])
      have : order.filterMap (op s t j) = [(vals.get2 r n).toRat] := by
        have e : op s t j = fun idx => if match' s t j idx then some ((vals.get idx).toRat) else none := rfl
        rw [e, filterMap_guard (match' s t j) (fun idx => (vals.get idx).toRat) order, hfil]
        rfl
      rw [this]
      rfl
    · have hrm : (r : Nat) < mlen d i := hm
      have := hvals r n hrm va vb (by
        have e1 : tokOf d i r = t := htm.symm
        have e2 : 16 * tileN d i + n = j := by show 16 * tileN d i + (j - 16 * tileN d i) = j; omega
        rw [e1, e2, ← hsl] at *
        exact hD)
      exact this

/-- The atomic's pointers (line 288-292), lane by lane: row `r`'s token and column `n` of the tile. -/
theorem epi_ptr (hd : d.ok) (hact : Active d i) (hi : i < (launch d).size) {curOut : Ptr .f32}
    (hcur : curOut = d.out.ptr ((slice d i * (d.M * d.R) : Nat) : Int)) {ram : Tensor [32 - 0] I32}
    (hram : ∀ r : Fin (32 - 0), ram.get1 r = IntV.ofNat (tokOf d i (r.val % mlen d i)))
    (r : Fin 32) (n : Fin 16) (hr : (r : Nat) < mlen d i) :
    (curOut + ram.col 16 * (IntV.ofNat d.R : I32) +
      (tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).row 32 * (IntV.ofNat 1 : I32)).get2 r n =
      d.out.ptr (outAt d (slice d i) (tokOf d i r) (16 * tileN d i + n) : Nat) := by
  have hrow := hact.row hd hr
  have hMle := hd.M_le
  have hM := hd.out_wrap
  have hRK := hd.RK_lt
  have hRlt : d.R < 2 ^ 31 := by have := hd.K_pos; nlinarith
  have hn := tileN_lt hd hi n.isLt
  have hx : ∀ (r : Fin 32) (n : Fin 16), ((ram.col 16 * (IntV.ofNat d.R : I32)).get2 r n).toInt =
      ((tokOf d i (r.val % mlen d i) * d.R : Nat) : Int) := by
    intro r n
    have hrow := hact.row hd (Nat.mod_lt r.val hact.mlen_pos)
    have hlt : tokOf d i (r.val % mlen d i) * d.R < 2 ^ 31 := by
      have h1 : tokOf d i (r.val % mlen d i) ≤ d.M - 1 := by omega
      calc _ ≤ (d.M - 1) * d.R := Nat.mul_le_mul_right _ h1
        _ < 2 ^ 31 := hM
    show (ram.get1 r * (IntV.ofNat d.R : I32)).toInt = _
    rw [hram r]
    exact IntV.I32.mul_nat (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_ofNat hRlt) hlt
  have hy : ∀ (r : Fin 32) (n : Fin 16),
      (((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).row 32 * (IntV.ofNat 1 : I32)).get2 r n).toInt =
        (((n.val + tileN d i * 16) * 1 : Nat) : Int) := by
    intro r n
    have hn' := tileN_lt hd hi n.isLt
    show ((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).get1 n * (IntV.ofNat 1 : I32)).toInt = _
    refine IntV.I32.mul_nat (IntV.I32.add_nat (IntV.I32.toInt_tl_arange (n := 16) (by norm_num) n)
      (IntV.I32.toInt_ofNat (by omega)) (by omega)) (IntV.I32.toInt_ofNat (by norm_num)) (by omega)
  have := ptr_seq2 d.out (base := curOut) (e := slice d i * (d.M * d.R)) hcur hx hy r n
  rw [this]
  have hr0 : (r : Nat) % mlen d i = r := Nat.mod_eq_of_lt hr
  congr 2
  rw [hr0]
  unfold outAt
  ring

/-- The atomic's mask (line 293), lane by lane. -/
theorem epi_mask (hd : d.ok) (hi : i < (launch d).size) (r : Fin 32) (n : Fin 16) :
    ((((tl.arange 0 32).col 16).ltS (IntV.ofNat (mlen d i) : I32)).and
      (((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).row 32).ltS (IntV.ofNat d.R : I32))).get2 r n
      = true ↔ (r : Nat) < mlen d i := by
  have hn := tileN_lt hd hi n.isLt
  have hRK := hd.RK_lt
  have hRlt : d.R < 2 ^ 31 := by have := hd.K_pos; nlinarith
  have hm32 := mlen_le d i
  show (((tl.arange 0 32).get1 r).lt (IntV.ofNat (mlen d i) : I32) &&
    ((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).get1 n).lt (IntV.ofNat d.R : I32)) = true ↔ _
  rw [Bool.and_eq_true, IntV.lt_eq, IntV.lt_eq, IntV.I32.toInt_tl_arange (n := 32) (by norm_num) r,
    IntV.I32.toInt_ofNat (by omega)]
  have h2 : ((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).get1 n).toInt =
      ((n.val + tileN d i * 16 : Nat) : Int) :=
    IntV.I32.add_nat (IntV.I32.toInt_tl_arange (n := 16) (by norm_num) n)
      (IntV.I32.toInt_ofNat (by omega)) (by omega)
  rw [h2, IntV.I32.toInt_ofNat hRlt]
  simp only [decide_eq_true_eq]
  constructor
  · rintro ⟨h, -⟩
    exact_mod_cast h
  · intro h
    refine ⟨by exact_mod_cast h, ?_⟩
    have := n.isLt
    omega

/-- **The end of a program is `SafeA`**: the product with `scaling` of each accumulator lane of a row
is a `PartOk` operand; the atomic adds the tile into the output cells and leaves `Q`. -/
theorem safe_epilogue (hP : Device P) (hd : d.ok) (hpre : Pre d init) (hi : i < (launch d).size)
    (hact : Active d i) (hs : slice d i < d.S) {curOut : Ptr .f32}
    (hcur : curOut = d.out.ptr ((slice d i * (d.M * d.R) : Nat) : Int)) {ram : Tensor [32 - 0] I32}
    (hram : ∀ r : Fin (32 - 0), ram.get1 r = IntV.ofNat (tokOf d i (r.val % mlen d i)))
    {acc : Tensor [32, 16] F32}
    (hacc : ∀ (r : Fin 32) (n : Fin 16), (r : Nat) < mlen d i → ∀ va vb : Nat → BF16,
      Dat d init (slice d i) (tokOf d i r) (16 * tileN d i + n) va vb →
        LaneInv d va vb (part d i) (stepsOf d (part d i)) (acc.get2 r n)) :
    SafeA ((launch d).env P) (Rall d) ∅ (Cells d)
      (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Q d init i log m) [] init
      (epilogue d i curOut ram acc) := by
  obtain ⟨-, -, houtV, -⟩ := hpre
  have hsk1 : ¬ (sk d = 1) := by rcases bk_sk_cases d with ⟨-, h⟩ | ⟨-, h⟩ <;> omega
  have hp : part d i < sk d := Nat.mod_lt _ (sk_pos d)
  unfold epilogue
  simp only [hsk1, ite_false]
  refine SafeA.mul fun w hw => ?_
  have hvals : ∀ (r : Fin 32) (n : Fin 16), (r : Nat) < mlen d i → ∀ va vb : Nat → BF16,
      Dat d init (slice d i) (tokOf d i r) (16 * tileN d i + n) va vb →
        PartOk d va vb (part d i) (w.get2 r n) := by
    intro r n hr va vb hD
    obtain ⟨hc, herr⟩ := laneInv_final d va vb (part d i) (hacc r n hr va vb hD)
    exact partOk_of_mul d hd va vb (part d i) hp hD.2.2.2.2 _ _ hc herr (hw (r, n, ()))
  have hmem : ∀ (r : Fin 32) (n : Fin 16), (r : Nat) < mlen d i →
      (DType.f32, ((d.out.ptr (outAt d (slice d i) (tokOf d i r) (16 * tileN d i + n) : Nat) :
        Ptr .f32)).addr) ∈ Cells d ∧
      Mem.Aligned .f32 ((d.out.ptr (outAt d (slice d i) (tokOf d i r) (16 * tileN d i + n) : Nat) :
        Ptr .f32)).addr := by
    intro r n hr
    obtain ⟨hrt, hrj⟩ : tokOf d i r < d.M ∧ 16 * tileN d i + n < d.R :=
      ⟨(hact.row hd hr).1, tileN_lt hd hi n.isLt⟩
    exact ⟨⟨slice d i, tokOf d i r, 16 * tileN d i + n, hs, hrt, hrj, rfl⟩,
      d.out.aligned houtV (hd.out_len _ _ _ hs hrt hrj)⟩
  refine SafeA.atomic_add_unused (sc := .gpu) ?_ (by simp) ?_ ?_
  · rintro ⟨r, n, u⟩ hsome
    cases u
    unfold tl.addrs at hsome
    by_cases hm : ((((tl.arange 0 32).col 16).ltS (IntV.ofNat (mlen d i) : I32)).and
      (((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32)).row 32).ltS (IntV.ofNat d.R : I32))).get
        (r, n, ()) = true
    · have hr := (epi_mask hd hi r n).mp hm
      have hp' := epi_ptr hd hact hi hcur hram r n hr
      show Mem.Aligned .f32 (_ : Ptr .f32).addr ∧ _
      rw [show (_ : Tensor [32, 16] (Ptr .f32)).get (r, n, ()) = _ from hp']
      exact ⟨(hmem r n hr).2, (hmem r n hr).1⟩
    · simp [hm] at hsome
  · intro idx old
    exact hP.atomicAddTotal .binary32 old _
  · intro order horder olds news hupd
    refine SafeA.ret ?_
    exact Q_atomic hP hd hact (hptr := fun r n hr => epi_ptr hd hact hi hcur hram r n hr)
      (hmask := fun r n => epi_mask hd hi r n) hvals hs hi horder hupd

end QAtomic

end Claims.LoraShrink
