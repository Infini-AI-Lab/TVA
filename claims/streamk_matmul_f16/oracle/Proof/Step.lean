import Claim
import TVA.Proof
import Proof.Atomic

/-!
# `Proof.Step`

One trip of the outer `while` loop, and the safety of a whole program (`prog_safe`).
-/

/-!
# One iteration of the outer loop

The body of the stream-K `while` loop: the program splits the counter into the tile index and the
chunk range, runs the `forRange` over the chunks of that tile (`acc_safe`), and finishes with the
store-or-atomic tail (`tail_safe`).  The three `tl.idiv`/`tl.irem` computations are deterministic
picks, so the invariant is preserved along them.
-/

namespace Claims.StreamKMatmul

open TVA

set_option maxHeartbeats 800000

variable {d : Inst} {P : Platform}


/-- One iteration of the `while` loop of the kernel, with the two values the loop computed before
it (`num_pid_n` and `iters_per_tile`) and the loop's end (`«end»`) as parameters. -/
def LoopBody (d : Inst) (qn qk endv : I32) : I32 → Prog I32 := fun it => do
  let __do_lift ← tl.idiv (Tensor.scalar it) (Tensor.scalar qk)
  have tile : I32 := __do_lift.item
  have k_first : I32 := it - tile * qk
  have k_last : I32 :=
    (tl.iminimum (Tensor.scalar qk) (Tensor.scalar (k_first + (endv - it)))).item
  let __do_lift ← tl.idiv (Tensor.scalar tile) (Tensor.scalar qn)
  have pid_m : I32 := __do_lift.item
  let __do_lift ← tl.irem (Tensor.scalar tile) (Tensor.scalar qn)
  have pid_n : I32 := __do_lift.item
  have _offs_m : Tensor [128] I32 := pid_m * (IntV.ofNat 128 : I32) + tl.arange 0 128
  have _offs_n : Tensor [128] I32 := pid_n * (IntV.ofNat 128 : I32) + tl.arange 0 128
  have acc0 : Tensor [128, 128] F32 := tl.zeros [128, 128] DType.f32
  let acc ← tl.forRange k_first k_last (IntV.ofInt 1) acc0
    (fun kk acc => ChunkBody d (offs1 pid_m 128) (offs1 pid_n 128) kk acc)
  have c_ptrs : Tensor [128, 128] (Ptr DType.f32) :=
    ptrsC (d.c.ptr 0) (offs1 pid_m 128) (offs1 pid_n 128) (IntV.ofNat d.stride_cm)
      (IntV.ofNat d.stride_cn)
  have c_mask : Tensor [128, 128] Bool :=
    maskC (offs1 pid_m 128) (offs1 pid_n 128) (IntV.ofNat d.M) (IntV.ofNat d.N)
  have __do_jp : Unit → Prog I32 := fun __r => pure (it + (k_last - k_first))
  if (IntV.eq k_first (IntV.ofInt 0 : I32) && IntV.eq k_last qk) = true then do
    let __r ← tl.store c_ptrs acc (some c_mask)
    __do_jp __r
  else do
    let _ ← tl.atomic_add c_ptrs acc (some c_mask)
    __do_jp ()

/-- **One iteration of the outer loop is safe.** -/
theorem step_safe (hP : Device P) (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hpi : i < d.num_programs) {stv qn qk endv : I32}
    (hqn : qn.toInt = (tilesN d : Int)) (hqk : qk.toInt = (steps d.K : Int))
    (hendv : endv.toInt = (hi d i : Int))
    {it : Nat} (hstv : stv.toInt = (it : Int))
    (hlo : lo d i ≤ it) (hit : it < hi d i)
    (hrc : it = lo d i ∨ it % steps d.K = 0)
    {log1 : List AtomicStep} {m1 : Mem}
    (hQg : Qg d init i it log1 m1) (hpres : Pres d m1)
    (hfr : ∀ x ∈ inElems d, m1 x = init x) :
    SafeA ((launch d).env P) (inElems d) (Wv d i (mu d i stv)) (Ccells d)
      (fun st' log m => Inv d init i st' log m ∧ mu d i st' < mu d i stv) log1 m1
      (LoopBody d qn qk endv stv) := by
  have hs0 : 0 ≤ stv.toInt := by rw [hstv]; exact Int.natCast_nonneg _
  have hK : 0 < steps d.K := T_pos hd
  have hk0 : 0 < qk.toInt := by rw [hqk]; exact_mod_cast hK
  have hn0 : 0 < qn.toInt := by rw [hqn]; exact_mod_cast tilesN_pos hd
  have hhi31 : hi d i < 2 ^ 31 := lt_of_le_of_lt (hi_le_total d i) (total_lt hd)
  have hit31 : it < 2 ^ 31 := lt_of_lt_of_le hit (le_of_lt hhi31)
  have hs31 : stv.toInt < 2 ^ 31 := by rw [hstv]; exact_mod_cast hit31
  unfold LoopBody
  -- the tile index
  obtain ⟨tv, htv, htvval⟩ := IntV.div?_of_nonneg (a := stv) (b := qk) hs0 hk0
  refine SafeA.pick ⟨Tensor.scalar tv, ?_⟩ ?_
  · intro idx
    show stv.div? qk = some tv
    exact htv
  · intro b hb
    rw [Prog.ret_bind]
    have hbtv : b.item = tv := by
      have h1 := hb ()
      simp only [Tensor.scalar] at h1
      rw [htv] at h1
      exact (Option.some.inj h1).symm
    rw [hbtv]
    have hs1 : 0 ≤ tv.toInt := by rw [htvval]; exact Int.ediv_nonneg hs0 hk0.le
    -- pid_m
    obtain ⟨pmv, hpmv, hpmvval⟩ := IntV.div?_of_nonneg (a := tv) (b := qn) hs1 hn0
    refine SafeA.pick ⟨Tensor.scalar pmv, ?_⟩ ?_
    · intro idx
      show tv.div? qn = some pmv
      exact hpmv
    · intro bm hbm
      rw [Prog.ret_bind]
      have hbpm : bm.item = pmv := by
        have h1 := hbm ()
        simp only [Tensor.scalar] at h1
        rw [hpmv] at h1
        exact (Option.some.inj h1).symm
      rw [hbpm]
      -- pid_n
      obtain ⟨pnv, hpnv, hpnvval⟩ := IntV.rem?_of_nonneg (a := tv) (b := qn) hs1 hn0
      refine SafeA.pick ⟨Tensor.scalar pnv, ?_⟩ ?_
      · intro idx
        show tv.rem? qn = some pnv
        exact hpnv
      · intro bn hbn
        rw [Prog.ret_bind]
        have hbpn : bn.item = pnv := by
          have h1 := hbn ()
          simp only [Tensor.scalar] at h1
          rw [hpnv] at h1
          exact (Option.some.inj h1).symm
        rw [hbpn]
        -- the tile index `stv / qk` is `tileOf d it`
        have htile : tv.toInt = (tileOf d it : Int) := by
          rw [htvval, hstv, hqk, ← Int.natCast_div]
          rfl
        have hpm : pmv.toInt = (pidMOf d (tileOf d it) : Int) :=
          pidM_toInt (u := tileOf d it) htile hqn (tilesN_pos hd) hpmv
        have hpn : pnv.toInt = (pidNOf d (tileOf d it) : Int) :=
          pidN_toInt (u := tileOf d it) htile hqn (tilesN_pos hd) hpnv
        -- the chunk range
        have hkf : (stv - tv * qk).toInt = (c0 d i (tileOf d it) : Int) := by
          have h := kFirst_toInt' (st := stv) (b := qk) (w := tv) (kf := stv - tv * qk)
            hs0 hK hqk hs31 htv rfl
          simp only [itN, hstv, Int.toNat_natCast] at h
          rw [h, kFirstOf_eq_c0 hd hlo hit hrc]
        have hsub : (endv - stv).toInt = ((hi d i - it : Nat) : Int) :=
          endSub_toInt (it := it) (hi := hi d i) hs0 hstv hendv (le_of_lt hit) hhi31 rfl
        have hkfAdd : (stv - tv * qk + (endv - stv)).toInt
            = ((c0 d i (tileOf d it) + (hi d i - it) : Nat) : Int) := by
          refine kfAdd_toInt (kf := stv - tv * qk) (r := endv - stv)
            (s := stv - tv * qk + (endv - stv)) hkf hsub rfl ?_
          have h1 : kFirstOf d it = c0 d i (tileOf d it) := kFirstOf_eq_c0 hd hlo hit hrc
          have hdec := tile_decomp (d := d) (it := it)
          omega
        have hkl : (tl.iminimum (Tensor.scalar qk)
              (Tensor.scalar (stv - tv * qk + (endv - stv)))).item.toInt
            = (c1 d i (tileOf d it) : Int) := by
          rw [iminimum_item]
          have h1 : (IntV.min qk (stv - tv * qk + (endv - stv))).toInt
              = ((min (steps d.K) (c0 d i (tileOf d it) + (hi d i - it)) : Nat) : Int) :=
            kLast_toInt (s := stv - tv * qk + (endv - stv)) (b := qk)
              (x := c0 d i (tileOf d it) + (hi d i - it)) (T := steps d.K)
              (kl := IntV.min qk (stv - tv * qk + (endv - stv))) hkfAdd hqk rfl
          have hnat : min (steps d.K) (c0 d i (tileOf d it) + (hi d i - it))
              = c1 d i (tileOf d it) := by
            rw [← kFirstOf_eq_c0 hd hlo hit hrc, ← kLastOf_eq_c1 (i := i) (it := it) hit]
            rfl
          rw [h1, hnat]
        have hu : tileOf d it < T d := tileOf_lt_T hd hit
        have hc : c0 d i (tileOf d it) ≤ c1 d i (tileOf d it) :=
          c0_le_c1 (le_trans hlo (le_of_lt hit))
        -- the inner `forRange` and the tail
        dsimp only
        refine acc_safe hP hd hpre hpres hfr (u := tileOf d it) (pm := pmv) (pn := pnv)
          (kf := stv - tv * qk)
          (kl := (tl.iminimum (Tensor.scalar qk)
            (Tensor.scalar (stv - tv * qk + (endv - stv)))).item)
          (acc0 := tl.zeros [128, 128] DType.f32)
          hpm hpn hkf hkl hc hu (accI_zero d init i (tileOf d it)) ?_
        intro acc2 log2 m2 hlog hm2 hacc2
        rw [hlog, hm2]
        exact tail_safe hP hd hpre hpi hlo hit hstv hkf hkl hpm hpn hqk hQg hpres hacc2 hrc

end Claims.StreamKMatmul


/-!
# A `while` loop rule for `SafeA`

`TVA.safe_while` is the loop rule for the private-memory rules `Safe`.  The Stream-K kernel's
outer loop lives in a program that shares cells with the other programs, so it needs the same
rule for `SafeA`: this file ports `safe_while` (`TVA/Lemmas/Ops.lean`) to `SafeA`, threading the
atomic log through the invariant.
-/

namespace TVA

/-- **While loop, for a program with shared cells.** Same as `safe_while`, with the log of shared
atomics carried by the invariant. -/
theorem SafeA.whileLoop {E : Env} {R : Set Int} {C : Set (DType × Int)} {σ : Type}
    {cond : σ → Prog Bool} {body : σ → Prog σ} {init : σ}
    {Q' : σ → List AtomicStep → Mem → Prop} {I : σ → List AtomicStep → Mem → Prop}
    {μ : σ → Nat} {Wv : Nat → Set Int} {log : List AtomicStep} {m : Mem}
    (hWv : ∀ u v, u < v → Disjoint (Wv u) (Wv v)) (h0 : I init log m)
    (hiter : ∀ st log₁ m₁, I st log₁ m₁ →
      SafeA E R ∅ C (fun c log₂ m₂ => if c then
          SafeA E R (Wv (μ st)) C (fun st' log₃ m₃ => I st' log₃ m₃ ∧ μ st' < μ st) log₂ m₂ (body st)
        else Q' st log₂ m₂) log₁ m₁ (cond st)) :
    SafeA E R (iterW Wv (μ init + 1)) C Q' log m (tl.whileLoop init cond body) := by
  unfold tl.whileLoop
  suffices H : ∀ n, ∀ st log₁ m₁, μ st = n → I st log₁ m₁ →
      SafeA E R (iterW Wv (n + 1)) C Q' log₁ m₁ (.loop st (fun st => (cond st).bind fun c =>
        if c then (body st).bind (fun st' => .ret (.inl st')) else .ret (.inr st)) .ret) from
    H _ init log m rfl h0
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro st log₁ m₁ hμ hI
    subst hμ
    apply SafeA.loop
    unfold Prog.loopStep
    have hd : Disjoint (Wv (μ st)) (iterW Wv (μ st)) := Set.disjoint_left.mpr
      fun x hx ⟨t, ht, hx'⟩ => Set.disjoint_left.mp (hWv t (μ st) ht) hx' hx
    refine (SafeA.bind (W₁ := Wv (μ st)) (W₂ := iterW Wv (μ st)) ?_ hd).mono
      (fun _ _ _ h => h) ?_
    · rw [show Wv (μ st) = ∅ ∪ Wv (μ st) by simp]
      refine SafeA.bind (W₁ := ∅) (W₂ := Wv (μ st)) ?_ (Set.empty_disjoint _)
      refine (hiter st log₁ m₁ hI).mono ?_ le_rfl
      intro c log₂ m₂ hc
      cases c with
      | true =>
        simp only [ite_true] at hc ⊢
        rw [show Wv (μ st) = Wv (μ st) ∪ ∅ by simp]
        refine SafeA.bind (W₁ := Wv (μ st)) (W₂ := ∅) ?_ (Set.disjoint_empty _)
        refine hc.mono ?_ le_rfl
        rintro st' log₃ m₃ ⟨hI', hlt⟩
        have hle : μ st' + 1 ≤ μ st := by omega
        exact SafeA.ret ((ih (μ st') hlt st' log₃ m₃ rfl hI').mono (fun _ _ _ h => h)
          (iterW_mono (Wt := Wv) hle))
      | false =>
        simp only [Bool.false_eq_true, ite_false] at hc ⊢
        exact SafeA.ret (SafeA.ret hc)
    · rintro x (hx | ⟨t, ht, hx⟩)
      · exact ⟨μ st, by omega, hx⟩
      · exact ⟨t, by omega, hx⟩

end TVA


/-!
# One program of the launch

Program `i` of the model is the two compile-time divisions of the kernel (which the `pick` rule
fixes to the values `tilesN` and `steps K`), then the outer `while` loop.  This file states that
the model's body is exactly the loop body of `Proof.StepSafe` and runs the loop with the
invariant `Inv`, from the initial memory to `Q`.
-/

namespace Claims.StreamKMatmul

open TVA

set_option maxHeartbeats 800000

variable {d : Inst} {P : Platform}

/-- The loop's start value of the model. -/
def It0 (d : Inst) (ctx : Ctx) : I32 :=
  tl.program_id ctx 0 * IntV.ofNat d.iters_per_program

/-- The loop's end (`«end»`) of the model. -/
def EndV (d : Inst) (ctx : Ctx) : I32 :=
  (tl.iminimum
    (Tensor.scalar (tl.program_id ctx 0 * IntV.ofNat d.iters_per_program +
      IntV.ofNat d.iters_per_program : I32))
    (Tensor.scalar (IntV.ofNat d.total_iters : I32))).item

/-- **The body of the model** at a context: the two `tl.cdiv` scalars, then the outer `while`
loop, whose body is `LoopBody` with the two scalars and the loop's end substituted.  Definitional,
so the rules apply to the model's program as written. -/
theorem model_body_eq (d : Inst) (ctx : Ctx) :
    (model d).body ctx =
      (tl.idiv (Tensor.scalar (IntV.ofNat d.N + (IntV.ofNat 127 : I32)))
          (Tensor.scalar (IntV.ofNat 128 : I32))) >>= fun b =>
      (tl.idiv (Tensor.scalar (IntV.ofNat d.K + (IntV.ofNat 63 : I32)))
          (Tensor.scalar (IntV.ofNat 64 : I32))) >>= fun b2 =>
      (tl.whileLoop (It0 d ctx) (fun it => pure (IntV.lt it (EndV d ctx)))
        (fun it => LoopBody d b.item b2.item (EndV d ctx) it)) >>= fun _ => pure () := rfl

/-- **Program `i` is safe alone**, with the `a`/`b` bytes as its read footprint, its stored output
elements as its write footprint, and the split output cells as the shared cells. -/
theorem prog_safe (hP : Device P) (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) :
    SafeA ((launch d).env P) (inElems d) (Wset d (i : Nat)) (Ccells d)
      (fun (_ : Unit) log m => Q d init (i : Nat) log m) [] init
      ((model d).body ((launch d).ctx i)) := by
  classical
  have hpi : (i : Nat) < d.num_programs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hpid : tl.program_id ((launch d).ctx i) 0 = IntV.ofNat (i : Nat) := by
    rw [tl.program_id, launch_pid d i]
  have hst0 : (It0 d ((launch d).ctx i)).toInt = ((lo d (i : Nat) : Nat) : Int) := by
    rw [It0, hpid]
    exact it_val hd hpi
  have hendv : (EndV d ((launch d).ctx i)).toInt = ((hi d (i : Nat) : Nat) : Int) := by
    rw [EndV, hpid]
    exact end_val hd hpi
  rw [model_body_eq]
  refine SafeA.mono
    (W := iterW (Wv d (i : Nat)) (mu d (i : Nat) (It0 d ((launch d).ctx i)) + 1) ∪ ∅)
    (Q := fun (_ : Unit) log m => Q d init (i : Nat) log m) ?_ (fun a log m h => h) ?_
  · obtain ⟨qn, hqn_eq, _⟩ := IntV.div?_of_nonneg
      (a := (IntV.ofNat d.N + (IntV.ofNat 127 : I32))) (b := (IntV.ofNat 128 : I32))
      (by rw [N_add127_toInt hd]; positivity) (by rw [lit128_toInt]; norm_num)
    obtain ⟨qk, hqk_eq, _⟩ := IntV.div?_of_nonneg
      (a := (IntV.ofNat d.K + (IntV.ofNat 63 : I32))) (b := (IntV.ofNat 64 : I32))
      (by rw [K_add63_toInt hd]; positivity) (by rw [lit64_toInt]; norm_num)
    refine SafeA.pick ⟨Tensor.scalar qn, ?_⟩ ?_
    · intro idx
      show (IntV.ofNat d.N + (IntV.ofNat 127 : I32)).div? (IntV.ofNat 128 : I32) = some qn
      exact hqn_eq
    · intro b hb
      rw [Prog.ret_bind]
      have hbn : b.item.toInt = ((tilesN d : Nat) : Int) := by
        have h1 := hb ()
        simp only [Tensor.scalar] at h1
        exact num_pid_n_toInt hd (q := b.item) (by simpa only [Tensor.item] using h1)
      refine SafeA.pick ⟨Tensor.scalar qk, ?_⟩ ?_
      · intro idx
        show (IntV.ofNat d.K + (IntV.ofNat 63 : I32)).div? (IntV.ofNat 64 : I32) = some qk
        exact hqk_eq
      · intro b2 hb2
        rw [Prog.ret_bind]
        have hbk : b2.item.toInt = ((steps d.K : Nat) : Int) := by
          have h1 := hb2 ()
          simp only [Tensor.scalar] at h1
          exact iters_per_tile_toInt hd (q := b2.item) (by simpa only [Tensor.item] using h1)
        refine SafeA.bind
          (W₁ := iterW (Wv d (i : Nat)) (mu d (i : Nat) (It0 d ((launch d).ctx i)) + 1))
          (W₂ := ∅) ?_ (Set.disjoint_empty _)
        refine SafeA.whileLoop (μ := fun st => mu d (i : Nat) st) (Wv := Wv d (i : Nat))
          (I := fun st log m => Inv d init (i : Nat) st log m)
          (Q' := fun (_ : I32) log m =>
            SafeA ((launch d).env P) (inElems d) ∅ (Ccells d)
              (fun (_ : Unit) log m => Q d init (i : Nat) log m) log m (pure ()))
          (fun u v h => Wv_disjoint hd h) ?_ ?_
        · exact inv_zero hst0 (Qg_init hd) hpre
        · intro st log1 m1 hI
          obtain ⟨hs0, hlole, hQg, hpres, hrc⟩ := hI
          have hstv : st.toInt = ((itN st : Nat) : Int) := by
            rw [itN, Int.toNat_of_nonneg hs0]
          refine SafeA.ret ?_
          by_cases hc : (IntV.lt st (EndV d ((launch d).ctx i))) = true
          · rw [ite_eq_left hc]
            have hlt : itN st < hi d (i : Nat) := by
              have h2 : st.toInt < (EndV d ((launch d).ctx i)).toInt := by
                have h := hc
                rw [IntV.lt_eq] at h
                exact of_decide_eq_true h
              rw [hendv] at h2
              omega
            obtain ⟨L, hL, hacc, hframe, hval, hrun, hcnt⟩ := hQg
            have hfr : ∀ x ∈ inElems d, m1 x = init x := fun x hx =>
              hframe x
                (fun hx' => Set.disjoint_left.mp (inElems_disjoint_touched hd (i : Nat)) hx hx')
            exact step_safe hP hd hpre hpi hbn hbk hendv hstv hlole hlt (hrc hlt)
              ⟨L, hL, hacc, hframe, hval, hrun, hcnt⟩ hpres hfr
          · rw [ite_eq_right hc]
            have hge : hi d (i : Nat) ≤ itN st := by
              have h2 : ¬ st.toInt < (EndV d ((launch d).ctx i)).toInt := by
                intro h3
                exact hc (by rw [IntV.lt_eq]; exact decide_eq_true h3)
              rw [hendv] at h2
              omega
            exact SafeA.ret (Qg_end hQg hge)
  · intro x hx
    rcases hx with hx | hx
    · exact iterW_Wv_sub_Wset hx
    · simp at hx

end Claims.StreamKMatmul

