import Proof.Trip

/-!
# `token_bitmask_f32`: one program

Program `p` computes `num_blocks = cdiv(vocab_size, 4096)` and runs the persistent loop
`for work_id in tl.range(p, items, NUM_SMS)`: trip `t` serves item `p + t * NUM_SMS`
(`Proof/Trip.lean`). After the loop, every cell of its items holds `-∞`.
-/

namespace Claims.TokenBitmask

open TVA

/-! ## The trip count -/

/-- A trip of `range(p, T, S)` serves an item below `T`. -/
theorem item_lt_of_trip {p T S t : Nat} (hS : 0 < S)
    (ht : t < tl.tripCount (p : Int) (T : Int) (S : Int)) : p + t * S < T := by
  unfold tl.tripCount at ht
  have hS' : (0 : Int) < S := by exact_mod_cast hS
  have hq : (t : Int) < ((T : Int) - p + S - 1) / S := by
    have h0 : (0 : Int) ≤ ((T : Int) - p + S - 1) / S := by
      by_contra hneg
      push Not at hneg
      rw [Int.toNat_of_nonpos hneg.le] at ht
      omega
    rw [← Int.toNat_of_nonneg h0]
    exact_mod_cast ht
  have hle : ((T : Int) - p + S - 1) / S * S ≤ (T : Int) - p + S - 1 := Int.ediv_mul_le _ (ne_of_gt hS')
  have h1 : ((t : Int) + 1) * S ≤ ((T : Int) - p + S - 1) / S * S :=
    mul_le_mul_of_nonneg_right (by omega) hS'.le
  have : (p : Int) + t * S < T := by nlinarith
  exact_mod_cast this

/-- Every item `w < T` of program `p < S` (`w % S = p`) is served by a trip of `range(p, T, S)`. -/
theorem lt_of_trip_count {p T S w : Nat} (hS : 0 < S) (hmod : w % S = p) (hw : w < T) :
    w < p + tl.tripCount (p : Int) (T : Int) (S : Int) * S := by
  unfold tl.tripCount
  have hS' : (0 : Int) < S := by exact_mod_cast hS
  have hdm := Nat.div_add_mod w S
  rw [hmod] at hdm
  generalize w / S = q at hdm
  have hwI : (w : Int) = S * (q : Int) + p := by
    have := congrArg (fun n : Nat => (n : Int)) hdm
    simp only [Nat.cast_add, Nat.cast_mul] at this
    linarith
  have hwT : (w : Int) < T := by exact_mod_cast hw
  have hq : (q : Int) + 1 ≤ ((T : Int) - p + S - 1) / S := by
    apply Int.le_ediv_of_mul_le hS'
    nlinarith
  have h0 : (0 : Int) ≤ ((T : Int) - p + S - 1) / S := le_trans (by positivity) hq
  have hc : ((((T : Int) - p + S - 1) / S).toNat : Int) = ((T : Int) - p + S - 1) / S :=
    Int.toNat_of_nonneg h0
  have : (w : Int) < p + ((((T : Int) - p + S - 1) / S).toNat : Int) * S := by
    rw [hc, hwI]
    nlinarith
  exact_mod_cast this

/-! ## One program -/

/-- **One program** of the grid is safe from `init`: it reads the bitmask (and the indices), writes
the cells of its items, and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size) :
    Safe E (R d) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m) init
      ((model d).body ((launch d).ctx i)) := by
  have hi : (i : Nat) < d.grid := lt_of_lt_of_eq i.isLt (launch_size d)
  have hiS : (i : Nat) < d.NUM_SMS := lt_of_lt_of_le hi hd.grid_le
  have hS : 0 < d.NUM_SMS := lt_of_le_of_lt (Nat.zero_le _) hiS
  have hS31 := hd.sms_lt
  have hV := hd.vocab_lt
  have hR := hd.range_le
  have hitems : items d = d.num_rows * nb d := rfl
  rw [body_eq]
  simp only [tl.program_id, launch_pid]
  -- line 104: `num_blocks = tl.cdiv(vocab_size, 4096)`
  have hx : ∀ u : Idx [], ((Tensor.scalar (IntV.ofNat d.vocab_size : I32) +
      (Tensor.scalar (IntV.ofNat 4096 : I32) - Tensor.full 1)).get u).toInt =
      ((d.vocab_size + 4095 : Nat) : Int) := by
    intro u
    show ((IntV.ofNat d.vocab_size : I32) + ((IntV.ofNat 4096 : I32) - 1)).toInt = _
    have h1 : ((IntV.ofNat 4096 : I32) - 1).toInt = ((4096 - 1 : Nat) : Int) :=
      IntV.I32.sub_nat (IntV.I32.toInt_ofNat (by norm_num)) (IntV.I32.toInt_ofNat (by norm_num))
        (by norm_num) (by norm_num)
    exact IntV.I32.add_nat (IntV.I32.toInt_ofNat (by omega)) h1 (by omega)
  have hy : ∀ u : Idx [], ((Tensor.scalar (IntV.ofNat 4096 : I32)).get u).toInt = ((4096 : Nat) : Int) :=
    fun _ => IntV.I32.toInt_ofNat (by norm_num)
  unfold tl.cdiv
  refine safe_idiv (fun u => ?_) fun nbT hnbT => ?_
  · obtain ⟨q, hq, -⟩ := IntV.div?_nat (hx u) (hy u) (by norm_num)
    exact ⟨q, hq⟩
  have hnb : nbT.item.toInt = (nb d : Int) :=
    IntV.toInt_of_div?_nat (hnbT ()) (hx ()) (hy ()) (by norm_num)
  -- line 105: the bounds of the loop
  have hT : ((IntV.ofNat d.num_rows : I32) * nbT.item).toInt = ((items d : Nat) : Int) := by
    rw [hitems]
    exact IntV.I32.mul_nat (IntV.I32.toInt_ofNat hd.rows_lt) hnb (by rw [← hitems]; omega)
  have hlo : (IntV.ofNat (i : Nat) : I32).toInt = ((i : Nat) : Int) := IntV.I32.toInt_ofNat (by omega)
  have hst : (IntV.ofNat d.NUM_SMS : I32).toInt = (d.NUM_SMS : Int) := IntV.I32.toInt_ofNat hS31
  have hmax : IntV.maxInt true 32 = 2 ^ 31 - 1 := by norm_num [IntV.maxInt]
  have hrange : tl.RangeOk (IntV.ofNat (i : Nat) : I32) ((IntV.ofNat d.num_rows : I32) * nbT.item)
      (IntV.ofNat d.NUM_SMS : I32) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [hst]; exact_mod_cast hS
    · rw [hT, hst, hmax]; push_cast; omega
    · rw [hlo, hmax]; push_cast; omega
    · rw [hst, hmax]; push_cast; omega
  let Wt : Nat → Set Int := fun t => d.logits.elems (itemElems d init ((i : Nat) + t * d.NUM_SMS))
  refine safe_last ?_
  refine Safe.mono (W := iterW Wt (tl.tripCount ((i : Nat) : Int) ((items d : Nat) : Int)
      (d.NUM_SMS : Int)) ∪ ∅) ?_ (fun _ _ h => h) ?_
  · have hstep := safe_forRange (E := E) (R := R d) (m := init) (Q := fun _ m => Q d init (i : Nat) m)
      (init := ()) (body := fun w _ => tripProg d nbT.item w) (k := pure) hrange
      (fun t _ m => Inv d init (i : Nat) t m) Wt (Wk := ∅)
    rw [hT, hlo, hst] at hstep
    refine hstep ?_ (Set.disjoint_empty _) ?_ ?_ ?_
    · intro t u htu _
      refine Buf.elems_disjoint d.logits (itemElems_disjoint hd hpre ?_)
      intro h
      have : t * d.NUM_SMS = u * d.NUM_SMS := by omega
      exact absurd (Nat.eq_of_mul_eq_mul_right hS this) (ne_of_lt htu)
    · refine ⟨?_, fun _ _ => rfl, hpre.1⟩
      intro k c _ hmod hlt
      have := Nat.mod_le (itemOf d k c) d.NUM_SMS
      omega
    · intro t ht st m₁ hI
      have hlt := item_lt_of_trip hS ht
      have hw : (IntV.ofInt (((i : Nat) : Int) + (t : Int) * (d.NUM_SMS : Int)) : I32).toInt =
          (((i : Nat) + t * d.NUM_SMS : Nat) : Int) := by
        push_cast
        have : ((i : Nat) + t * d.NUM_SMS : Nat) < 2 ^ 31 := by omega
        have h0 : (0 : Int) ≤ ((i : Nat) : Int) + (t : Int) * (d.NUM_SMS : Int) := by positivity
        refine IntV.I32.toInt_ofInt ⟨by linarith, ?_⟩
        exact_mod_cast this
      exact safe_trip hd hpre E hnb hw hlt hI
    · intro st m₁ hI
      refine safe_ret ?_
      intro k c hM hmod
      refine hI.1 k c hM hmod ?_
      exact lt_of_trip_count hS hmod (itemOf_lt hM.1 hM.2.1)
  · rintro x (⟨t, -, hx⟩ | hx)
    · refine Buf.elems_mono d.logits ?_ hx
      rintro e ⟨k, c, hM, hit, he⟩
      refine ⟨k, c, hM, ?_, he⟩
      rw [hit, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hiS]
    · exact absurd hx (Set.notMem_empty x)

end Claims.TokenBitmask
