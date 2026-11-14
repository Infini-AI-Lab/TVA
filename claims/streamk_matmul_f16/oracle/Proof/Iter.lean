import Claim
import TVA.Proof
import Proof.Chunk

/-!
# `Proof.Iter`

Loop rules for `SafeA`, the inner-loop bounds, and the safety of the accumulator loop.
-/

/-!
# The integer arithmetic of one loop iteration

The body of the stream-K `while` loop recomputes, from the loop counter `st`, which tile it is on
and which chunks of it to run:

* `tile = st / steps K`, `pid_m = tile / tilesN`, `pid_n = tile % tilesN`;
* `k_first = st - tile * steps K = st % steps K`, `k_last = min (steps K) (k_first + (end - st))`.

This file proves that these `int32` computations agree with the `Nat` values the invariant of
`Proof/Inv.lean` is stated with (`kFirstOf`, `kLastOf`, `c0`, `c1`, `tileOf`), and the facts about
the loop step: the counter advances to `min ((tile + 1) · steps K) (hi p)`, the measure `mu`
decreases, and the `if` that chooses the store branch fires exactly on `owns`.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-! ## Small `int32` facts -/

/-- `tl.iminimum` on scalars is the minimum. -/
theorem iminimum_item (x y : I32) :
    (tl.iminimum (Tensor.scalar x) (Tensor.scalar y)).item = IntV.min x y := rfl

/-- Scalar `int32` addition. -/
theorem iadd_item (x y : I32) :
    (x + y : I32) = x + y := rfl

/-- The `if` of the loop body fires exactly when the program owns the tile. -/
theorem eq_and_eq_iff {kf kl b b0 : I32} {c0 c1 T : Nat}
    (hk : kf.toInt = (c0 : Int)) (hl : kl.toInt = (c1 : Int)) (hb : b.toInt = (T : Int))
    (h0 : b0 = IntV.ofInt 0) :
    ((IntV.eq kf b0 && IntV.eq kl b) = true) ↔ (c0 = 0 ∧ c1 = T) := by
  have hbz : b0.toInt = 0 := by
    rw [h0]
    exact IntV.I32.toInt_ofInt (x := 0) (by norm_num)
  constructor
  · intro h
    rw [Bool.and_eq_true] at h
    obtain ⟨h1, h2⟩ := h
    have e1 : kf.toInt = b0.toInt := by simpa [IntV.eq_eq] using h1
    have e2 : kl.toInt = b.toInt := by simpa [IntV.eq_eq] using h2
    refine ⟨?_, ?_⟩ <;> omega
  · rintro ⟨rfl, rfl⟩
    rw [Bool.and_eq_true]
    refine ⟨(IntV.eq_eq _ _).trans (decide_eq_true ?_), (IntV.eq_eq _ _).trans (decide_eq_true ?_)⟩
    · rw [hk, hbz]; norm_num
    · rw [hl, hb]

/-! ## Division and remainder -/

/-- `int32` division of a nonnegative value by `T` is the natural-number division. -/
theorem div_toNat {st b w : I32} {T : Nat} (hs : 0 ≤ st.toInt) (hb : b.toInt = (T : Int))
    (hT : 0 < T) (h : st.div? b = some w) :
    w.toInt = ((st.toInt.toNat / T : Nat) : Int) := by
  have hbpos : 0 < b.toInt := by rw [hb]; exact_mod_cast hT
  calc w.toInt = st.toInt / b.toInt := div?_toInt hs hbpos h
    _ = st.toInt / (T : Int) := by rw [hb]
    _ = ((st.toInt.toNat : Nat) : Int) / (T : Int) := by rw [Int.toNat_of_nonneg hs]
    _ = ((st.toInt.toNat / T : Nat) : Int) := (Int.natCast_div _ _).symm

/-- `int32` remainder of a nonnegative value by `T` is the natural-number remainder. -/
theorem rem_toNat {st b w : I32} {T : Nat} (hs : 0 ≤ st.toInt) (hb : b.toInt = (T : Int))
    (hT : 0 < T) (h : st.rem? b = some w) :
    w.toInt = ((st.toInt.toNat % T : Nat) : Int) := by
  have hbpos : 0 < b.toInt := by rw [hb]; exact_mod_cast hT
  calc w.toInt = st.toInt % b.toInt := rem?_toInt hs hbpos h
    _ = st.toInt % (T : Int) := by rw [hb]
    _ = ((st.toInt.toNat : Nat) : Int) % (T : Int) := by rw [Int.toNat_of_nonneg hs]
    _ = ((st.toInt.toNat % T : Nat) : Int) := (Int.natCast_emod _ _).symm

/-- The tile index: `st / steps K` is `tileOf`. -/
theorem tile_toInt {st b w : I32} (hs : 0 ≤ st.toInt) (hK : 0 < steps d.K)
    (hb : b.toInt = (steps d.K : Int)) (h : st.div? b = some w) :
    w.toInt.toNat = tileOf d (itN st) := by
  rw [div_toNat hs hb hK h, Int.toNat_natCast]
  rfl

/-- `k_first`: `st - (st / T) · T` is `st % T`. -/
theorem kFirst_toInt {st b w kf : I32} {T : Nat} (hs : 0 ≤ st.toInt) (hb : b.toInt = (T : Int))
    (hT : 0 < T) (hst : st.toInt < 2 ^ 31) (h : st.div? b = some w) (hk : kf = st - w * b) :
    kf.toInt = ((st.toInt.toNat % T : Nat) : Int) := by
  have hcast : ((st.toInt.toNat : Nat) : Int) = st.toInt := Int.toNat_of_nonneg hs
  have hnlt : st.toInt.toNat < 2 ^ 31 := by
    have h2 : ((st.toInt.toNat : Nat) : Int) < (2 ^ 31 : Int) := by rw [hcast]; exact hst
    exact_mod_cast h2
  have hw : w.toInt = ((st.toInt.toNat / T : Nat) : Int) := div_toNat hs hb hT h
  have hmul : (w * b).toInt = (((st.toInt.toNat / T) * T : Nat) : Int) :=
    IntV.I32.mul_nat hw hb (lt_of_le_of_lt (Nat.div_mul_le_self _ _) hnlt)
  have hsub : kf.toInt = ((st.toInt.toNat - (st.toInt.toNat / T) * T : Nat) : Int) := by
    rw [hk]
    exact IntV.I32.sub_nat hcast.symm hmul (Nat.div_mul_le_self _ _) hnlt
  rw [hsub]
  congr 1
  have hdiv : (st.toInt.toNat / T) * T + st.toInt.toNat % T = st.toInt.toNat := by
    rw [Nat.mul_comm]; exact Nat.div_add_mod _ _
  omega

/-- `k_first` against `kFirstOf`. -/
theorem kFirst_toInt' {st b w kf : I32} (hs : 0 ≤ st.toInt) (hK : 0 < steps d.K)
    (hb : b.toInt = (steps d.K : Int)) (hst : st.toInt < 2 ^ 31) (h : st.div? b = some w)
    (hk : kf = st - w * b) :
    kf.toInt = ((kFirstOf d (itN st) : Nat) : Int) := by
  exact kFirst_toInt (T := steps d.K) hs hb hK hst h hk

/-! ## The end of the program's range, and `k_last` -/

/-- `end - st` in `Nat`. -/
theorem endSub_toInt {st e r : I32} {it hi : Nat} (_hs : 0 ≤ st.toInt) (hit : st.toInt = (it : Int))
    (he : e.toInt = (hi : Int)) (hle : it ≤ hi) (hhi : hi < 2 ^ 31) (hr : r = e - st) :
    r.toInt = ((hi - it : Nat) : Int) := by
  rw [hr]
  exact IntV.I32.sub_nat he hit hle hhi

/-- `k_first + r` does not wrap in `int32` when `a + b < 2^31`. -/
theorem kfAdd_toInt {kf r s : I32} {a b : Nat} (hk : kf.toInt = (a : Int)) (hr : r.toInt = (b : Int))
    (hs : s = kf + r) (hab : a + b < 2 ^ 31) : s.toInt = ((a + b : Nat) : Int) := by
  rw [hs]
  exact IntV.I32.add_nat hk hr hab

/-- `k_last = min (steps K) (k_first + (end - st))`. -/
theorem kLast_toInt {s b kl : I32} {x T : Nat} (hs : s.toInt = (x : Int)) (hb : b.toInt = (T : Int))
    (hk : kl = IntV.min b s) : kl.toInt = ((min T x : Nat) : Int) := by
  rw [hk, IntV.toInt_min, hs, hb, ← Nat.cast_min]

/-- `k_last - k_first` is `c1 - c0` on the current tile. -/
theorem kLast_sub_kFirst {kl kf r : I32} {c0 c1 : Nat} (hl : kl.toInt = (c1 : Int))
    (hk : kf.toInt = (c0 : Int)) (hc : c0 ≤ c1) (hc1 : c1 < 2 ^ 31) (hr : r = kl - kf) :
    r.toInt = ((c1 - c0 : Nat) : Int) := by
  rw [hr]
  exact IntV.I32.sub_nat hl hk hc hc1

/-! ## The chunk range of the current tile -/

/-- `k_first` is the start `c0` of the run in the tile. -/
theorem kFirstOf_eq_c0 {i it : Nat} (hd : d.ok) (hlo : lo d i ≤ it) (_hit : it < hi d i)
    (hrc : it = lo d i ∨ it % steps d.K = 0) :
    kFirstOf d it = c0 d i (tileOf d it) := by
  have hS : 0 < steps d.K := T_pos hd
  have hdec := tile_decomp (d := d) (it := it)
  have hkf : kFirstOf d it < steps d.K := by unfold kFirstOf; exact Nat.mod_lt _ hS
  unfold c0 tileOf kFirstOf at *
  rcases hrc with hrc | hrc
  · rw [hrc]
    have hmod : lo d i - lo d i / steps d.K * steps d.K = lo d i % steps d.K := by
      have := Nat.div_add_mod (lo d i) (steps d.K)
      rw [Nat.mul_comm] at this
      omega
    rw [hmod, max_eq_right (Nat.zero_le _), min_eq_right (le_of_lt (Nat.mod_lt _ hS))]
  · rw [hrc]
    have hlt : lo d i ≤ it / steps d.K * steps d.K := by omega
    have hz : lo d i - it / steps d.K * steps d.K = 0 := by omega
    rw [hz, max_self, min_eq_right (Nat.zero_le _)]

/-- `k_last` is the end `c1` of the run in the tile. -/
theorem kLastOf_eq_c1 {i it : Nat} (hit : it < hi d i) :
    kLastOf d it (hi d i) = c1 d i (tileOf d it) := by
  unfold kLastOf c1
  by_cases hS : steps d.K = 0
  · simp [hS]
  · have hdec := tile_decomp (d := d) (it := it)
    have hle : tileOf d it * steps d.K ≤ hi d i := by omega
    have h1 : kFirstOf d it + (hi d i - it) = hi d i - tileOf d it * steps d.K := by omega
    rw [h1, max_eq_right (by omega)]

/-- The counter's next value. -/
theorem it_next {i it : Nat} (hd : d.ok) (hit : it < hi d i) :
    it + (kLastOf d it (hi d i) - kFirstOf d it) =
      min ((tileOf d it + 1) * steps d.K) (hi d i) := by
  have hS : 0 < steps d.K := T_pos hd
  have hdec := tile_decomp (d := d) (it := it)
  have hkf : kFirstOf d it < steps d.K := by unfold kFirstOf; exact Nat.mod_lt _ hS
  have hsub := kLastOf_sub (d := d) (it := it) (end_ := hi d i) (le_of_lt hit)
  rw [hsub, Nat.succ_mul]
  omega

/-- The tile index stays below the number of tiles. -/
theorem tileOf_lt_T {i it : Nat} (hd : d.ok) (hit : it < hi d i) : tileOf d it < T d := by
  rcases Nat.eq_zero_or_pos (steps d.K) with h0 | h0
  · exfalso
    have htot := hd.total
    rw [h0, Nat.mul_zero] at htot
    have hhi : hi d i ≤ d.total_iters := hi_le_total d i
    omega
  · have hle : it < d.total_iters := lt_of_lt_of_le hit (hi_le_total d i)
    unfold tileOf T
    rw [Nat.div_lt_iff_lt_mul h0, ← hd.total]
    exact hle

/-- One loop step strictly decreases the measure. -/
theorem mu_step {i : Nat} {st st' : I32} {it it' : Nat} (hd : d.ok)
    (hit : it < hi d i) (hst : it = itN st) (hst' : it' = itN st')
    (hnext : it' = min ((tileOf d it + 1) * steps d.K) (hi d i)) :
    mu d i st' < mu d i st := by
  have hS : 0 < steps d.K := by
    rcases Nat.eq_zero_or_pos (steps d.K) with h0 | h0
    · exfalso
      have htot := hd.total
      rw [h0, Nat.mul_zero] at htot
      have hhi : hi d i ≤ d.total_iters := hi_le_total d i
      omega
    · exact h0
  have hu : tileOf d it < T d := tileOf_lt_T hd hit
  have hstN : itN st = it := hst.symm
  have hstN' : itN st' = it' := hst'.symm
  have hmu : mu d i st = T d - tileOf d it := by
    unfold mu
    rw [hstN, ite_eq_left hit]
  rw [hmu]
  by_cases hlt : it' < hi d i
  · have hmu' : mu d i st' = T d - tileOf d it' := by
      unfold mu
      rw [hstN', ite_eq_left hlt]
    have hAle : (tileOf d it + 1) * steps d.K ≤ hi d i := by
      by_contra hc
      have hmin : min ((tileOf d it + 1) * steps d.K) (hi d i) = hi d i :=
        min_eq_right (le_of_lt (not_le.mp hc))
      rw [hnext, hmin] at hlt
      exact lt_irrefl _ hlt
    have heq : it' = (tileOf d it + 1) * steps d.K := by rw [hnext, min_eq_left hAle]
    have htil : tileOf d it' = tileOf d it + 1 := by
      rw [heq]
      unfold tileOf
      exact Nat.mul_div_cancel _ hS
    rw [hmu', htil]
    omega
  · have hmu' : mu d i st' = T d - T d := by
      unfold mu
      rw [hstN', ite_eq_right hlt]
    rw [hmu', Nat.sub_self]
    omega

/-- The loop invariant's reachability clause is preserved. -/
theorem reach_step {i : Nat} {it : Nat} (_hit : it < hi d i) :
    min ((tileOf d it + 1) * steps d.K) (hi d i) = lo d i ∨
      min ((tileOf d it + 1) * steps d.K) (hi d i) % steps d.K = 0 ∨
      min ((tileOf d it + 1) * steps d.K) (hi d i) = hi d i := by
  by_cases h : hi d i ≤ (tileOf d it + 1) * steps d.K
  · exact Or.inr (Or.inr (min_eq_right h))
  · refine Or.inr (Or.inl ?_)
    rw [min_eq_left (le_of_lt (not_le.mp h))]
    exact Nat.mul_mod_left _ _

/-- `lo p ≤ it` is preserved. -/
theorem lo_le_next {i it : Nat} (hd : d.ok) (hlo : lo d i ≤ it) (hit : it < hi d i) :
    lo d i ≤ min ((tileOf d it + 1) * steps d.K) (hi d i) := by
  have hS : 0 < steps d.K := T_pos hd
  have hdec := tile_decomp (d := d) (it := it)
  have hkf : kFirstOf d it < steps d.K := by unfold kFirstOf; exact Nat.mod_lt _ hS
  have hle : it ≤ (tileOf d it + 1) * steps d.K := by
    rw [Nat.succ_mul]
    omega
  exact le_min (le_trans hlo hle) (by omega)

/-! ## `tl.RangeOk` and the trip count of the inner loop -/

/-- The inner loop's bounds satisfy `tl.RangeOk`. -/
theorem rangeOk {kf kl : I32} {c0 c1 : Nat} (h0 : kf.toInt = (c0 : Int)) (h1 : kl.toInt = (c1 : Int))
    (hc : c0 ≤ c1) (h2 : c1 + 1 ≤ 2 ^ 31) :
    tl.RangeOk kf kl (IntV.ofInt 1 : I32) := by
  have hstep : (IntV.ofInt (1 : Int) : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)
  have hR : IntV.maxInt true 32 = ((2 ^ 31 - 1 : Nat) : Int) := by
    norm_num [IntV.maxInt]
  unfold tl.RangeOk
  rw [h0, h1, hstep, hR]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- The inner loop runs `c1 - c0` times. -/
theorem tripCount_eq {kf kl : I32} {c0 c1 : Nat} (h0 : kf.toInt = (c0 : Int))
    (h1 : kl.toInt = (c1 : Int)) (hc : c0 ≤ c1) :
    tl.tripCount kf.toInt kl.toInt (IntV.ofInt 1 : I32).toInt = c1 - c0 := by
  have hstep : (IntV.ofInt (1 : Int) : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)
  unfold tl.tripCount
  rw [h0, h1, hstep, Int.ediv_one]
  have hcast : ((c1 : Int) - (c0 : Int) + 1 - 1) = ((c1 - c0 : Nat) : Int) := by
    rw [Nat.cast_sub hc]
    ring
  rw [hcast, Int.toNat_natCast]

/-- The chunk run starts at `c0 + t`. -/
theorem chunk_val {kf : I32} {c0 : Nat} (h0 : kf.toInt = (c0 : Int)) (t : Nat)
    (ht : t < 2 ^ 31) (h : c0 + t < 2 ^ 31) :
    (IntV.ofInt (IntV.toInt kf + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt =
      ((c0 + t : Nat) : Int) := by
  have hstep : (IntV.ofInt (1 : Int) : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)
  rw [hstep, h0]
  rw [IntV.I32.toInt_ofInt (x := (c0 : Int) + (t : Int) * 1) (by constructor <;> omega)]
  push_cast
  ring


/-! ## The matrix coordinates of the current tile -/

/-- `pid_m`: `tile / tilesN` is `pidMOf`. -/
theorem pidM_toInt {tile b w : I32} {u : Nat} (htile : tile.toInt = (u : Int))
    (hb : b.toInt = (tilesN d : Int)) (hT : 0 < tilesN d) (h : tile.div? b = some w) :
    w.toInt = (pidMOf d u : Int) := by
  have ha : 0 ≤ tile.toInt := by rw [htile]; exact Int.natCast_nonneg _
  have hbpos : 0 < b.toInt := by rw [hb]; exact_mod_cast hT
  rw [div?_toInt ha hbpos h, htile, hb, ← Int.natCast_div]
  rfl

/-- `pid_n`: `tile % tilesN` is `pidNOf`. -/
theorem pidN_toInt {tile b w : I32} {u : Nat} (htile : tile.toInt = (u : Int))
    (hb : b.toInt = (tilesN d : Int)) (hT : 0 < tilesN d) (h : tile.rem? b = some w) :
    w.toInt = (pidNOf d u : Int) := by
  have ha : 0 ≤ tile.toInt := by rw [htile]; exact Int.natCast_nonneg _
  have hbpos : 0 < b.toInt := by rw [hb]; exact_mod_cast hT
  rw [rem?_toInt ha hbpos h, htile, hb, ← Int.natCast_emod]
  rfl

end Claims.StreamKMatmul


/-!
# From `Safe` to `SafeA`

`SafeA` is `Safe` with a shared-atomic rule and a log; a `Safe` program is a `SafeA` program that
performs no shared atomic.  The log stays empty and the postcondition is remembered.
-/

namespace Claims.StreamKMatmul

open TVA

/-- **A `Safe` program is a `SafeA` program** with an unchanged log. -/
theorem Safe.toSafeA {E : Env} {R W : Set Int} {C : Set (DType × Int)}
    {α : Type} {Q : α → Mem → Prop} {log : List AtomicStep} {m : Mem} {p : Prog α}
    (h : Safe E R W Q m p) :
    SafeA E R W C (fun a log' m' => log' = log ∧ Q a m') log m p := by
  induction h with
  | ret hq => exact SafeA.ret ⟨rfl, hq⟩
  | load hb hR _ ih => exact SafeA.load hb hR fun v hv => ih v hv
  | store hok hW _ ih => exact SafeA.store hok hW fun o hp => ih o hp
  | rmw hex hS hSR hc _ ih => exact SafeA.rmw hex hS hSR hc fun o n ho hn => ih o n ho hn
  | pick hex _ ih => exact SafeA.pick hex fun b hb => ih b hb
  | plat hex _ ih => exact SafeA.plat hex fun r hr => ih r hr
  | loop _ ih => exact SafeA.loop ih
  | env _ ih => exact SafeA.env ih

end Claims.StreamKMatmul


/-!
# Loop rules for `SafeA`

The `forN` / `forRange` loop rules of `TVA/Lemmas/Ops.lean` (`safe_forN`, `safe_forN_bind`,
`safe_forRange`), ported to `SafeA`: the atomic log is threaded through the invariant, and the
shared cell set `C` is carried along.
-/

namespace TVA

/-- **Loop invariant** for `forN`, for a program with shared cells: `I t st log m` holds before
iteration `t`; each iteration preserves it and writes only inside its own footprint `Wt t`, and
appends to the log through the invariant. -/
theorem SafeA.forN {E : Env} {R : Set Int} {C : Set (DType × Int)} {σ : Type}
    {log : List AtomicStep} {m : Mem}
    (n : Nat) (body : Nat → σ → Prog σ) (init : σ)
    (I : Nat → σ → List AtomicStep → Mem → Prop) (Wt : Nat → Set Int)
    (hWt : ∀ t u, t < u → u < n → Disjoint (Wt t) (Wt u)) (h0 : I 0 init log m)
    (hstep : ∀ t, t < n → ∀ st log₁ m₁, I t st log₁ m₁ →
      SafeA E R (Wt t) C (fun st' log₂ m₂ => I (t + 1) st' log₂ m₂) log₁ m₁ (body t st)) :
    SafeA E R (iterW Wt n) C (fun st log₁ m₁ => I n st log₁ m₁) log m
      (tl.forN n body init) := by
  induction n with
  | zero => exact .ret h0
  | succ n ih =>
    simp only [tl.forN]
    have hd : Disjoint (iterW Wt n) (Wt n) := Set.disjoint_left.mpr fun x ⟨t, ht, hx⟩ hx' =>
      Set.disjoint_left.mp (hWt t n ht (by omega)) hx hx'
    have h := SafeA.bind ((ih (fun t u htu hu => hWt t u htu (by omega))
      (fun t ht => hstep t (by omega))).mono
      (fun st log₁ m₁ h => hstep n (by omega) st log₁ m₁ h) le_rfl) hd
    refine h.mono (fun _ _ _ h => h) ?_
    rintro x (⟨t, ht, hx⟩ | hx)
    · exact ⟨t, by omega, hx⟩
    · exact ⟨n, by omega, hx⟩

/-- `forN` followed by the rest of the program, which writes outside the loop's footprint. -/
theorem SafeA.forN_bind {E : Env} {R : Set Int} {C : Set (DType × Int)} {σ α : Type}
    {n : Nat} {body : Nat → σ → Prog σ} {init : σ} {k : σ → Prog α}
    {Q : α → List AtomicStep → Mem → Prop} {log : List AtomicStep} {m : Mem}
    (I : Nat → σ → List AtomicStep → Mem → Prop) (Wt : Nat → Set Int) {Wk : Set Int}
    (hWt : ∀ t u, t < u → u < n → Disjoint (Wt t) (Wt u))
    (hWk : Disjoint (iterW Wt n) Wk) (h0 : I 0 init log m)
    (hstep : ∀ t, t < n → ∀ st log₁ m₁, I t st log₁ m₁ →
      SafeA E R (Wt t) C (fun st' log₂ m₂ => I (t + 1) st' log₂ m₂) log₁ m₁ (body t st))
    (hk : ∀ st log₁ m₁, I n st log₁ m₁ → SafeA E R Wk C Q log₁ m₁ (k st)) :
    SafeA E R (iterW Wt n ∪ Wk) C Q log m (tl.forN n body init >>= k) :=
  SafeA.bind ((SafeA.forN n body init I Wt hWt h0 hstep).mono hk le_rfl) hWk

/-- `for i in range(lo, hi, step)` with `step > 0` and no overflow of the last increment: an
invariant over the trip number `t`, where the induction variable is `lo + t * step`. -/
theorem SafeA.forRange {E : Env} {R : Set Int} {C : Set (DType × Int)} {σ : Type}
    {sg : Bool} {w : Nat} {lo hi step : IntV sg w} {init : σ} {body : IntV sg w → σ → Prog σ}
    {α : Type} {k : σ → Prog α} {Q : α → List AtomicStep → Mem → Prop}
    {log : List AtomicStep} {m : Mem}
    (hrange : tl.RangeOk lo hi step)
    (I : Nat → σ → List AtomicStep → Mem → Prop) (Wt : Nat → Set Int) {Wk : Set Int}
    (hWt : ∀ t u, t < u → u < tl.tripCount lo.toInt hi.toInt step.toInt → Disjoint (Wt t) (Wt u))
    (hWk : Disjoint (iterW Wt (tl.tripCount lo.toInt hi.toInt step.toInt)) Wk)
    (h0 : I 0 init log m)
    (hstep : ∀ t, t < tl.tripCount lo.toInt hi.toInt step.toInt → ∀ st log₁ m₁, I t st log₁ m₁ →
      SafeA E R (Wt t) C (fun st' log₂ m₂ => I (t + 1) st' log₂ m₂) log₁ m₁
        (body (IntV.ofInt (lo.toInt + (t : Int) * step.toInt)) st))
    (hk : ∀ st log₁ m₁, I (tl.tripCount lo.toInt hi.toInt step.toInt) st log₁ m₁ →
      SafeA E R Wk C Q log₁ m₁ (k st)) :
    SafeA E R (iterW Wt (tl.tripCount lo.toInt hi.toInt step.toInt) ∪ Wk) C Q log m
      (tl.forRange lo hi step init body >>= k) := by
  simp only [tl.forRange, ite_eq_left hrange]
  exact SafeA.forN_bind I Wt hWt hWk h0 hstep hk

/-- `forN` followed by a rest that writes nothing and leaves the state, log and memory
unchanged. -/
theorem SafeA.forN_bind' {E : Env} {R : Set Int} {C : Set (DType × Int)} {σ : Type}
    {n : Nat} {body : Nat → σ → Prog σ} {init : σ} {k : σ → Prog σ}
    {I : Nat → σ → List AtomicStep → Mem → Prop} {Wt : Nat → Set Int} {log : List AtomicStep} {m : Mem}
    (hWt : ∀ t u, t < u → u < n → Disjoint (Wt t) (Wt u)) (h0 : I 0 init log m)
    (hstep : ∀ t, t < n → ∀ st log₁ m₁, I t st log₁ m₁ →
      SafeA E R (Wt t) C (fun st' log₂ m₂ => I (t + 1) st' log₂ m₂) log₁ m₁ (body t st))
    (hk : ∀ st log₁ m₁, I n st log₁ m₁ →
      SafeA E R ∅ C (fun st' log₂ m₂ => st' = st ∧ log₂ = log₁ ∧ m₂ = m₁) log₁ m₁ (k st)) :
    SafeA E R (iterW Wt n) C (fun st log₁ m₁ => I n st log₁ m₁) log m
      (tl.forN n body init >>= k) := by
  have hk' : ∀ st log₁ m₁, I n st log₁ m₁ →
      SafeA E R ∅ C (fun st' log₂ m₂ => I n st' log₂ m₂) log₁ m₁ (k st) := by
    intro st log₁ m₁ hI
    refine (hk st log₁ m₁ hI).mono ?_ le_rfl
    rintro st' log₂ m₂ ⟨rfl, rfl, rfl⟩
    exact hI
  have h := SafeA.forN_bind (Wk := (∅ : Set Int)) I Wt hWt (Set.disjoint_empty _) h0 hstep hk'
  simpa [Set.union_empty] using h

end TVA


/-!
# The chunk range of one iteration

The `forRange` over the iteration's chunk range: starting from the zero accumulator, the program
runs one `ChunkBody` per 64-wide chunk of `K`, reading only.  The accumulator invariant `AccI`
runs from `0` to `c1 - c0`, and the log and the memory are unchanged.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst} {P : Platform}

/-- **The chunk range is safe**: it reads only `inElems`, leaves the memory and the log alone,
and its accumulator reaches `c1 - c0` chunk accumulations. -/
theorem acc_safe (hP : Device P) (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i u : Nat} {init0 : Mem} (hpres : Pres d init0)
    (hfr : ∀ x ∈ inElems d, init0 x = init x)
    {pm pn kf kl : I32} {acc0 : Tensor [128, 128] F32}
    (hpm : pm.toInt = (pidMOf d u : Int)) (hpn : pn.toInt = (pidNOf d u : Int))
    (hkf : kf.toInt = (c0 d i u : Int)) (hkl : kl.toInt = (c1 d i u : Int))
    (hc : c0 d i u ≤ c1 d i u) (hu : u < T d) (hacc0 : AccI d init i u 0 acc0)
    {log0 : List AtomicStep} {Wk : Set Int} {α : Type} {k : Tensor [128, 128] F32 → Prog α}
    {Q : α → List AtomicStep → Mem → Prop}
    (hk : ∀ acc log1 m1, log1 = log0 → m1 = init0 →
      AccI d init i u (c1 d i u - c0 d i u) acc →
      SafeA ((launch d).env P) (inElems d) Wk (Ccells d) Q log1 m1 (k acc)) :
    SafeA ((launch d).env P) (inElems d) Wk (Ccells d) Q log0 init0
      (tl.forRange kf kl (IntV.ofInt 1) acc0
          (fun kk acc => ChunkBody d (offs1 pm 128) (offs1 pn 128) kk acc) >>= k) := by
  have hS : 0 < steps d.K := T_pos hd
  have hc1 : c1 d i u ≤ steps d.K := c1_le
  have hS128 : steps d.K ≤ 128 := steps_le_128 hd.K_le
  refine (SafeA.forRange (hrange := rangeOk hkf hkl hc (by omega))
    (I := fun t acc log m => log = log0 ∧ m = init0 ∧ AccI d init i u t acc)
    (Wt := fun _ => ∅) (Wk := Wk) ?_ ?_ ?_ ?_ ?_).mono (fun a _ _ h => h) ?_
  · intro t s _ _; exact Set.disjoint_empty _
  · refine Set.disjoint_left.mpr ?_
    rintro x ⟨t, _, hx⟩ _
    exact hx
  · exact ⟨rfl, rfl, hacc0⟩
  · intro t ht acc log1 m1 hI
    obtain ⟨rfl, rfl, hacc⟩ := hI
    have htc : t < c1 d i u - c0 d i u := by
      rw [← tripCount_eq hkf hkl hc]; exact ht
    have ht128 : t < 128 := by omega
    have hks : c0 d i u + t < steps d.K := by omega
    have hkk : (IntV.ofInt (IntV.toInt kf + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32).toInt =
        ((c0 d i u + t : Nat) : Int) :=
      chunk_val hkf t (by omega) (by omega)
    exact Safe.toSafeA (log := log1)
      (chunkSafe hP hd hpre hpres hfr hpm hpn hkk hu ht128 hks hacc)
  · intro acc log1 m1 hI
    obtain ⟨hlog, hm, hacc⟩ := hI
    rw [tripCount_eq hkf hkl hc] at hacc
    exact hk acc log1 m1 hlog hm hacc
  · intro x hx
    rcases hx with ⟨t, _, hx⟩ | hx
    · simp at hx
    · exact hx

end Claims.StreamKMatmul


/-!
# The invariant of the outer loop

The state of the outer `while` loop of a program is the iteration counter `st`.  Besides the
facts about the run so far (`Qg`, the presence of the matrices), the invariant records where the
counter is: it starts at `lo`, and each step lands on a chunk boundary or at `hi`.
-/

namespace Claims.StreamKMatmul

open TVA

variable {d : Inst}

/-- **The outer-loop invariant**: the counter is non-negative and between `lo` and `hi`, the run's
`Qg` holds, the matrices are present, and every reached counter is either `lo`, a chunk boundary,
or `hi`. -/
def Inv (d : Inst) (init : Mem) (i : Nat) (st : I32) (log : List AtomicStep) (m : Mem) : Prop :=
  0 ≤ st.toInt ∧ lo d i ≤ itN st ∧ Qg d init i (itN st) log m ∧ Pres d m ∧
    (itN st < hi d i → (itN st = lo d i ∨ itN st % steps d.K = 0))

/-- The loop invariant holds at the start. -/
theorem inv_zero {init : Mem} {i : Nat} {st : I32} (hst : st.toInt = ((lo d i : Nat) : Int))
    (hQg : Qg d init i (lo d i) [] init) (hpre : Pre d init) : Inv d init i st [] init := by
  refine ⟨?_, ?_, ?_, pres_init hpre, ?_⟩
  · rw [hst]; exact Int.natCast_nonneg _
  · rw [itN, hst, Int.toNat_natCast]
  · rw [itN, hst, Int.toNat_natCast]; exact hQg
  · rw [itN, hst, Int.toNat_natCast]
    intro _; exact Or.inl rfl

end Claims.StreamKMatmul

