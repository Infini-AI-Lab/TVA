import TVA.Proof.Softmax.Exp

/-!
# The online softmax, exactly

**What is computed.** One output element of attention is the softmax-weighted sum
`wavg K S V = Σ_{k ∈ K} softmax K S k · V k` over the keys `K`, with scores `S` and values `V` (one
output column).

**Why any reference works.** `exp (S k − m) = exp (S k)·exp (−m)`, so subtracting a reference `m`
from every score multiplies the numerator `Σ exp (S k)·V k` and the denominator `Σ exp (S k)` by
the same positive factor, which cancels in the ratio (`wavg_eq_div_sub`). A kernel picks `m` to
keep the exponentials in range (the maximum score), but the ratio does not depend on it.

**The loop.** An attention kernel with a loop over key blocks keeps, for one query row, a running
reference `m` (the maximum so far), a running sum `l` and an accumulator `acc`. Trip `t` works on
lanes `j ∈ B t` with score `s t j` and value `v t j`; with the new reference `m (t+1)` and the
rescaling factor `α t = exp (m t − m (t+1))`,

  `l (t+1) = α t·l t + Σ_{j ∈ B t} exp (s t j − m (t+1))`,
  `acc (t+1) = α t·acc t + Σ_{j ∈ B t} exp (s t j − m (t+1))·v t j`.

Multiplying by `α t` moves what was summed so far from the old reference to the new one, so the
invariant is: after `t` trips, `l t` and `acc t` are the full sums over everything seen, at the
reference `m t` (`lsum`, `asum`; `l_eq`, `acc_eq`). Their ratio is then `wavg` over the pairs
(trip, lane) seen, whatever the references were (`asum_div_lsum`, `ratio_eq`).

The loop starts from `l 0 = acc 0 = 0`. The kernel's first reference is `−∞`, so its first `α` is
`exp (−∞) = 0`; here it multiplies `0`, so `α 0` is left free. A kernel that starts from `l = 1`
(Triton's tutorial) still has `l 0 = 0` in the reals, because its first `α` is `0`.

**How a kernel uses it.** A masked lane is left out of `B t` (`Finset.filter`), or kept with an
exact weight `0` (`Finset.sum_filter` turns one form into the other). A kernel using `exp2` takes
`s := s₂·log 2` and `m := m₂·log 2` (`two_rpow_eq_exp`). For keys in blocks of `b` (lane
`j : Fin b` of trip `u` holds the key `u·b + j`) under a mask `P` (a sequence bound, a causal
bound), `lsum_blocks`, `asum_blocks`, `ratio_blocks` and `wavg_blocks` give the flat sums over
`{k < n·b | P k}`, which is how a claim states its specification. With rounding, the computed
state follows the same recurrence up to per-trip errors: `TVA.Proof.Softmax.Error`.
-/

namespace TVA

namespace Softmax

open Finset

/-! ## The softmax-weighted sum -/

section Wavg

variable {κ : Type*}

/-- The softmax over a finite set of keys `K`: `exp (S k) / Σ_{k' ∈ K} exp (S k')`, the weight
`tl.softmax` (or an attention kernel) gives key `k`. -/
noncomputable def softmax (K : Finset κ) (S : κ → ℝ) (k : κ) : ℝ :=
  Real.exp (S k) / ∑ k' ∈ K, Real.exp (S k')

/-- The softmax-weighted sum `Σ_{k ∈ K} softmax K S k · V k`: one output element of attention,
for scores `S` and values `V` (one column). -/
noncomputable def wavg (K : Finset κ) (S V : κ → ℝ) : ℝ := ∑ k ∈ K, softmax K S k * V k

/-- Softmax weights are non-negative. -/
theorem softmax_nonneg (K : Finset κ) (S : κ → ℝ) (k : κ) : 0 ≤ softmax K S k :=
  div_nonneg (Real.exp_pos _).le (sum_nonneg fun _ _ => (Real.exp_pos _).le)

/-- The softmax weights of a nonempty set add up to `1`. -/
theorem sum_softmax {K : Finset κ} (hK : K.Nonempty) (S : κ → ℝ) : ∑ k ∈ K, softmax K S k = 1 := by
  unfold softmax
  rw [← Finset.sum_div]
  exact div_self (sum_pos (fun _ _ => Real.exp_pos _) hK).ne'

/-- `wavg` as one ratio: `(Σ exp (S k)·V k) / Σ exp (S k)`. -/
theorem wavg_eq_div (K : Finset κ) (S V : κ → ℝ) :
    wavg K S V = (∑ k ∈ K, Real.exp (S k) * V k) / ∑ k ∈ K, Real.exp (S k) := by
  unfold wavg softmax
  rw [Finset.sum_div]
  exact sum_congr rfl fun k _ => by ring

/-- **The reference does not matter:** for any `m`,
`wavg K S V = (Σ exp (S k − m)·V k) / Σ exp (S k − m)`. Use it to match a kernel's ratio, taken at
whatever reference it ended with, to the specification. -/
theorem wavg_eq_div_sub (K : Finset κ) (S V : κ → ℝ) (m : ℝ) :
    wavg K S V = (∑ k ∈ K, Real.exp (S k - m) * V k) / ∑ k ∈ K, Real.exp (S k - m) := by
  rw [wavg_eq_div]
  have hn : ∑ k ∈ K, Real.exp (S k - m) * V k = (∑ k ∈ K, Real.exp (S k) * V k) * Real.exp (-m) := by
    rw [Finset.sum_mul]
    exact sum_congr rfl fun k _ => by rw [sub_eq_add_neg, Real.exp_add]; ring
  have hd : ∑ k ∈ K, Real.exp (S k - m) = (∑ k ∈ K, Real.exp (S k)) * Real.exp (-m) := by
    rw [Finset.sum_mul]
    exact sum_congr rfl fun k _ => by rw [sub_eq_add_neg, Real.exp_add]
  rw [hn, hd, mul_div_mul_right _ _ (Real.exp_pos _).ne']

/-- `|wavg K S V| ≤ c` when every `|V k| ≤ c` (`k ∈ K`): the output is within the values' range,
which bounds it for the output format. -/
theorem abs_wavg_le {K : Finset κ} {S V : κ → ℝ} {c : ℝ} (hc : 0 ≤ c)
    (hV : ∀ k ∈ K, |V k| ≤ c) : |wavg K S V| ≤ c := by
  rcases K.eq_empty_or_nonempty with hK | hK
  · simp [wavg, hK, hc]
  · calc |wavg K S V| ≤ ∑ k ∈ K, |softmax K S k * V k| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ k ∈ K, softmax K S k * c := sum_le_sum fun k hk => by
          rw [abs_mul, abs_of_nonneg (softmax_nonneg K S k)]
          exact mul_le_mul_of_nonneg_left (hV k hk) (softmax_nonneg K S k)
      _ = c := by rw [← Finset.sum_mul, sum_softmax hK, one_mul]

end Wavg

/-! ## The invariant -/

section Invariant

variable {ι : Type*}

/-- **The running sum in closed form:** `Σ_{u<t} Σ_{j ∈ B u} exp (s u j − m)`, what the sum `l`
holds after `t` trips at the reference `m`. Trip `u` has lanes `B u` and scores `s u j`. -/
noncomputable def lsum (s : ℕ → ι → ℝ) (B : ℕ → Finset ι) (m : ℝ) (t : ℕ) : ℝ :=
  ∑ u ∈ range t, ∑ j ∈ B u, Real.exp (s u j - m)

/-- **The accumulator in closed form:** `Σ_{u<t} Σ_{j ∈ B u} exp (s u j − m)·v u j`, what the
accumulator `acc` (one output column, values `v u j`) holds after `t` trips at the reference `m`. -/
noncomputable def asum (s v : ℕ → ι → ℝ) (B : ℕ → Finset ι) (m : ℝ) (t : ℕ) : ℝ :=
  ∑ u ∈ range t, ∑ j ∈ B u, Real.exp (s u j - m) * v u j

variable {s v : ℕ → ι → ℝ} {B : ℕ → Finset ι}

/-- No trip, no sum. -/
theorem lsum_zero (m : ℝ) : lsum s B m 0 = 0 := by simp [lsum]

/-- No trip, no accumulator. -/
theorem asum_zero (m : ℝ) : asum s v B m 0 = 0 := by simp [asum]

/-- The running sum is non-negative. -/
theorem lsum_nonneg (m : ℝ) (t : ℕ) : 0 ≤ lsum s B m t :=
  sum_nonneg fun _ _ => sum_nonneg fun _ _ => (Real.exp_pos _).le

/-- **A change of reference** rescales the sum: `lsum m' = exp (m − m')·lsum m`. -/
theorem lsum_rescale (m m' : ℝ) (t : ℕ) : lsum s B m' t = Real.exp (m - m') * lsum s B m t := by
  unfold lsum
  rw [mul_sum]
  refine sum_congr rfl fun u _ => ?_
  rw [mul_sum]
  refine sum_congr rfl fun j _ => ?_
  rw [← Real.exp_add]; congr 1; ring

/-- **A change of reference** rescales the accumulator: `asum m' = exp (m − m')·asum m`. -/
theorem asum_rescale (m m' : ℝ) (t : ℕ) : asum s v B m' t = Real.exp (m - m') * asum s v B m t := by
  unfold asum
  rw [mul_sum]
  refine sum_congr rfl fun u _ => ?_
  rw [mul_sum]
  refine sum_congr rfl fun j _ => ?_
  rw [← mul_assoc, ← Real.exp_add]; congr 2; ring

/-- **One trip of the sum:** from the reference `m` to `m'`,
`lsum m' (t+1) = exp (m − m')·lsum m t + Σ_{j ∈ B t} exp (s t j − m')`. This is the kernel's
update of `l`, so the closed form is preserved by each trip. -/
theorem lsum_succ (m m' : ℝ) (t : ℕ) :
    lsum s B m' (t + 1) = Real.exp (m - m') * lsum s B m t + ∑ j ∈ B t, Real.exp (s t j - m') := by
  rw [← lsum_rescale]; unfold lsum; rw [sum_range_succ]

/-- **One trip of the accumulator:** from the reference `m` to `m'`,
`asum m' (t+1) = exp (m − m')·asum m t + Σ_{j ∈ B t} exp (s t j − m')·v t j`, the kernel's update
of `acc`. -/
theorem asum_succ (m m' : ℝ) (t : ℕ) :
    asum s v B m' (t + 1) =
      Real.exp (m - m') * asum s v B m t + ∑ j ∈ B t, Real.exp (s t j - m') * v t j := by
  rw [← asum_rescale]; unfold asum; rw [sum_range_succ]

/-- `|asum v| ≤ asum |v|`. -/
theorem abs_asum_le (m : ℝ) (t : ℕ) : |asum s v B m t| ≤ asum s (fun u j => |v u j|) B m t := by
  unfold asum
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun u _ =>
    (abs_sum_le_sum_abs _ _).trans (le_of_eq (sum_congr rfl fun j _ => ?_)))
  rw [abs_mul, abs_of_pos (Real.exp_pos _)]

/-- `asum |v| ≤ V·lsum` when every `|v u j| ≤ V` on the lanes: the accumulator's magnitude is
bounded by the values' range times the sum. -/
theorem asum_abs_le {m V : ℝ} {t : ℕ} (hV : ∀ u < t, ∀ j ∈ B u, |v u j| ≤ V) :
    asum s (fun u j => |v u j|) B m t ≤ V * lsum s B m t := by
  unfold asum lsum
  rw [mul_sum]
  refine sum_le_sum fun u hu => ?_
  rw [mul_sum]
  refine sum_le_sum fun j hj => ?_
  rw [mul_comm V]
  exact mul_le_mul_of_nonneg_left (hV u (mem_range.mp hu) j hj) (Real.exp_pos _).le

/-- One term is at most the sum: `exp (s u j − m) ≤ lsum m t` for `u < t`, `j ∈ B u`. -/
theorem exp_le_lsum {m : ℝ} {t u : ℕ} {j : ι} (hu : u < t) (hj : j ∈ B u) :
    Real.exp (s u j - m) ≤ lsum s B m t := by
  unfold lsum
  calc Real.exp (s u j - m) ≤ ∑ j' ∈ B u, Real.exp (s u j' - m) :=
        single_le_sum (f := fun j' => Real.exp (s u j' - m)) (fun _ _ => (Real.exp_pos _).le) hj
    _ ≤ ∑ u' ∈ range t, ∑ j' ∈ B u', Real.exp (s u' j' - m) :=
        single_le_sum (f := fun u' => ∑ j' ∈ B u', Real.exp (s u' j' - m))
          (fun _ _ => sum_nonneg fun _ _ => (Real.exp_pos _).le) (mem_range.mpr hu)

/-- `1 ≤ lsum m t` when some lane seen has a score at least `m` (for example, `m` is the maximum).
Use it to show the final sum is far from `0`, so dividing by it is well conditioned. -/
theorem one_le_lsum {m : ℝ} {t u : ℕ} {j : ι} (hu : u < t) (hj : j ∈ B u) (h : m ≤ s u j) :
    1 ≤ lsum s B m t :=
  (Real.one_le_exp_iff.mpr (sub_nonneg.mpr h)).trans (exp_le_lsum hu hj)

/-- `0 < lsum m t` when some trip before `t` has a lane. -/
theorem lsum_pos {m : ℝ} {t u : ℕ} (hu : u < t) (hB : (B u).Nonempty) : 0 < lsum s B m t := by
  obtain ⟨j, hj⟩ := hB
  exact (Real.exp_pos _).trans_le (exp_le_lsum hu hj)

/-- **Every term at most `1`:** when `m` is at least every score seen, `lsum m t` is at most the
number of lanes. Use it to keep the computed sum within the format's range. -/
theorem lsum_le_card {m : ℝ} {t : ℕ} (h : ∀ u < t, ∀ j ∈ B u, s u j ≤ m) :
    lsum s B m t ≤ ∑ u ∈ range t, ((B u).card : ℝ) := by
  unfold lsum
  refine sum_le_sum fun u hu => ?_
  calc ∑ j ∈ B u, Real.exp (s u j - m) ≤ ∑ j ∈ B u, (1 : ℝ) :=
        sum_le_sum fun j hj => Real.exp_le_one_iff.mpr (by linarith [h u (mem_range.mp hu) j hj])
    _ = (B u).card := by simp

/-- `lsum` as one sum over the pairs (trip, lane). -/
theorem lsum_eq_sum_sigma (m : ℝ) (t : ℕ) :
    lsum s B m t = ∑ x ∈ (range t).sigma B, Real.exp (s x.1 x.2 - m) := by
  unfold lsum; rw [sum_sigma]

/-- `asum` as one sum over the pairs (trip, lane). -/
theorem asum_eq_sum_sigma (m : ℝ) (t : ℕ) :
    asum s v B m t = ∑ x ∈ (range t).sigma B, Real.exp (s x.1 x.2 - m) * v x.1 x.2 := by
  unfold asum; rw [sum_sigma]

/-- **The ratio is the softmax-weighted sum**, whatever the reference `m`:
`asum m t / lsum m t = wavg` over the pairs (trip, lane) seen in `t` trips. -/
theorem asum_div_lsum (m : ℝ) (t : ℕ) :
    asum s v B m t / lsum s B m t =
      wavg ((range t).sigma B) (fun x => s x.1 x.2) (fun x => v x.1 x.2) := by
  rw [wavg_eq_div_sub _ _ _ m, asum_eq_sum_sigma, lsum_eq_sum_sigma]

/-- **The exact running sum.** A sequence `l` with `l 0 = 0` and
`l (t+1) = α t·l t + Σ_{j ∈ B t} exp (s t j − m (t+1))`, where `α t = exp (m t − m (t+1))` for
`t > 0` (`α 0` is free: it multiplies `l 0 = 0`), is `lsum s B (m t) t` for every `t ≤ n`, for any
references `m`. Use it for a kernel's `l` computed without rounding (or as the reference the
rounded one is compared with). -/
theorem l_eq {m α l : ℕ → ℝ} {n : ℕ} (h0 : l 0 = 0)
    (hα : ∀ t, 0 < t → t < n → α t = Real.exp (m t - m (t + 1)))
    (hl : ∀ t < n, l (t + 1) = α t * l t + ∑ j ∈ B t, Real.exp (s t j - m (t + 1))) :
    ∀ t ≤ n, l t = lsum s B (m t) t := by
  intro t
  induction t with
  | zero => intro _; simp [h0, lsum_zero]
  | succ t ih =>
    intro ht
    rw [hl t (by omega), lsum_succ (m t)]
    rcases Nat.eq_zero_or_pos t with rfl | htp
    · simp [h0, lsum_zero]
    · rw [hα t htp (by omega), ih (by omega)]

/-- **The exact accumulator.** A sequence `acc` with `acc 0 = 0` and
`acc (t+1) = α t·acc t + Σ_{j ∈ B t} exp (s t j − m (t+1))·v t j`, where
`α t = exp (m t − m (t+1))` for `t > 0`, is `asum s v B (m t) t` for every `t ≤ n`. -/
theorem acc_eq {m α acc : ℕ → ℝ} {n : ℕ} (h0 : acc 0 = 0)
    (hα : ∀ t, 0 < t → t < n → α t = Real.exp (m t - m (t + 1)))
    (ha : ∀ t < n, acc (t + 1) = α t * acc t + ∑ j ∈ B t, Real.exp (s t j - m (t + 1)) * v t j) :
    ∀ t ≤ n, acc t = asum s v B (m t) t := by
  intro t
  induction t with
  | zero => intro _; simp [h0, asum_zero]
  | succ t ih =>
    intro ht
    rw [ha t (by omega), asum_succ (m t)]
    rcases Nat.eq_zero_or_pos t with rfl | htp
    · simp [h0, asum_zero]
    · rw [hα t htp (by omega), ih (by omega)]

/-- **The online softmax computes the softmax-weighted sum.** With the recurrences of `l_eq` and
`acc_eq` (the same `α` and references `m`), `acc t / l t` is `wavg` over the pairs (trip, lane)
seen, for every `t ≤ n`. -/
theorem ratio_eq {m α l acc : ℕ → ℝ} {n : ℕ} (hl0 : l 0 = 0) (ha0 : acc 0 = 0)
    (hα : ∀ t, 0 < t → t < n → α t = Real.exp (m t - m (t + 1)))
    (hl : ∀ t < n, l (t + 1) = α t * l t + ∑ j ∈ B t, Real.exp (s t j - m (t + 1)))
    (ha : ∀ t < n, acc (t + 1) = α t * acc t + ∑ j ∈ B t, Real.exp (s t j - m (t + 1)) * v t j) :
    ∀ t ≤ n, acc t / l t = wavg ((range t).sigma B) (fun x => s x.1 x.2) (fun x => v x.1 x.2) := by
  intro t ht
  rw [l_eq hl0 hα hl t ht, acc_eq ha0 hα ha t ht, asum_div_lsum]

end Invariant

/-! ## Keys in blocks -/

/-- Consecutive blocks of `b` make one range: `Σ_{u<n} Σ_{j<b} g (u·b + j) = Σ_{k<n·b} g k`. -/
theorem sum_blocks {M : Type*} [AddCommMonoid M] (g : ℕ → M) (b : ℕ) : ∀ n : ℕ,
    ∑ u ∈ range n, ∑ j ∈ range b, g (u * b + j) = ∑ k ∈ range (n * b), g k
  | 0 => by simp
  | n + 1 => by
    rw [sum_range_succ, sum_blocks g b n, show (n + 1) * b = n * b + b by ring, sum_range_add]

/-- The same with a mask `P` on the key `u·b + j`:
`Σ_{u<n} Σ_{j<b, P (u·b+j)} g (u·b + j) = Σ_{k<n·b, P k} g k`. -/
theorem sum_blocks_filter {M : Type*} [AddCommMonoid M] (P : ℕ → Prop) [DecidablePred P]
    (g : ℕ → M) (b n : ℕ) :
    ∑ u ∈ range n, ∑ j ∈ (range b).filter (fun j => P (u * b + j)), g (u * b + j) =
      ∑ k ∈ (range (n * b)).filter P, g k := by
  have h := sum_blocks (fun k => if P k then g k else 0) b n
  rw [sum_filter, ← h]
  exact sum_congr rfl fun u _ => sum_filter _ _

/-- The same with lanes `j : Fin b` (a tensor's lanes):
`Σ_{u<n} Σ_{j : Fin b, P (u·b+j)} g (u·b + j) = Σ_{k<n·b, P k} g k`. -/
theorem sum_blocks_fin_filter {M : Type*} [AddCommMonoid M] (P : ℕ → Prop) [DecidablePred P]
    (g : ℕ → M) (b n : ℕ) :
    ∑ u ∈ range n, ∑ j ∈ univ.filter (fun j : Fin b => P (u * b + j)), g (u * b + j) =
      ∑ k ∈ (range (n * b)).filter P, g k := by
  rw [← sum_blocks_filter]
  refine sum_congr rfl fun u _ => ?_
  rw [sum_filter, sum_filter]
  exact Fin.sum_univ_eq_sum_range (fun j => if P (u * b + j) then g (u * b + j) else 0) b

/-- **`lsum` over keys in blocks of `b`:** trip `u`, lane `j : Fin b` holds the key `u·b + j`, kept
when `P` holds (`B u = {j | P (u·b + j)}`, scores `S (u·b + j)`); the sum is the flat
`Σ_{k < n·b, P k} exp (S k − m)`. -/
theorem lsum_blocks (P : ℕ → Prop) [DecidablePred P] (S : ℕ → ℝ) (b : ℕ) (m : ℝ) (n : ℕ) :
    lsum (fun u (j : Fin b) => S (u * b + j)) (fun u => univ.filter (fun j : Fin b => P (u * b + j)))
        m n =
      ∑ k ∈ (range (n * b)).filter P, Real.exp (S k - m) :=
  sum_blocks_fin_filter P (fun k => Real.exp (S k - m)) b n

/-- **`asum` over keys in blocks of `b`:** the flat sum `Σ_{k < n·b, P k} exp (S k − m)·V k`. -/
theorem asum_blocks (P : ℕ → Prop) [DecidablePred P] (S V : ℕ → ℝ) (b : ℕ) (m : ℝ) (n : ℕ) :
    asum (fun u (j : Fin b) => S (u * b + j)) (fun u (j : Fin b) => V (u * b + j))
        (fun u => univ.filter (fun j : Fin b => P (u * b + j))) m n =
      ∑ k ∈ (range (n * b)).filter P, Real.exp (S k - m) * V k :=
  sum_blocks_fin_filter P (fun k => Real.exp (S k - m) * V k) b n

/-- **The ratio over keys in blocks of `b`** is the softmax-weighted sum over the keys
`{k < n·b | P k}` (for example `P k := k < N`, or `k < N ∧ k ≤ q` for a causal row `q`). -/
theorem ratio_blocks (P : ℕ → Prop) [DecidablePred P] (S V : ℕ → ℝ) (b : ℕ) (m : ℝ) (n : ℕ) :
    asum (fun u (j : Fin b) => S (u * b + j)) (fun u (j : Fin b) => V (u * b + j))
        (fun u => univ.filter (fun j : Fin b => P (u * b + j))) m n /
      lsum (fun u (j : Fin b) => S (u * b + j))
        (fun u => univ.filter (fun j : Fin b => P (u * b + j))) m n =
      wavg ((range (n * b)).filter P) S V := by
  rw [asum_blocks, lsum_blocks, wavg_eq_div_sub _ _ _ m]

/-- **The softmax-weighted sum over the pairs (trip, lane), for keys in blocks of `b`,** is the
one over the flat keys `{k < n·b | P k}`: it turns the target of `ratio_eq`, `err_final` and
`err_final_mono` into the flat specification. Give `P` explicitly when rewriting
(`wavg_blocks (fun k => k < N ∧ k ≤ q)`): `rw` cannot infer it from the filter. -/
theorem wavg_blocks (P : ℕ → Prop) [DecidablePred P] (S V : ℕ → ℝ) (b n : ℕ) :
    wavg ((range n).sigma (fun u => univ.filter (fun j : Fin b => P (u * b + j))))
        (fun x => S (x.1 * b + x.2)) (fun x => V (x.1 * b + x.2)) =
      wavg ((range (n * b)).filter P) S V := by
  rw [← asum_div_lsum (s := fun u (j : Fin b) => S (u * b + j))
    (v := fun u (j : Fin b) => V (u * b + j)) 0 n, ratio_blocks]

end Softmax

end TVA
