import Claim
import Proof.Prog

/-!
# The proof of `paged_attention_f16`

Program `(s, h)` of the grid reads the query row `q[s, h, :]`, `seq_lens[s]`, the block-table
entries it visits and the cache elements of the sequence's tokens, and writes `out[s, h, :]`.

1. `Proof.Num`, `Proof.Final`: the accuracy of the online softmax in real numbers. The state `l` and
   each component of `acc` follow the exact sums at the running maximum up to a relative error
   `Fc` that grows by `(1 + e1 + γ·Δμ)` per page (`col_step`); the end of the loop gives the
   claim's tolerance (`final_core`).
2. `Proof.Flt1`, `Proof.Flt2`: each float operation of a page, from the device facts and the float
   kit (`scores_row`, `max_new`, `alpha_ok`, `lane_ok`, `lsum_block`, `acc_block`, ...).
3. `Proof.Stage`, `Proof.Stages`: one page's values, in numbers (`scores_stage`,
   `weights_stage`, `lcol_stage`, `acol_stage`).
4. `Proof.Trip`: one page of the loop is safe and keeps the invariant `Inv`.
5. `Proof.Out`, `Proof.Prog`: the division, the rounding, the store; one program `(s, h)` is `Safe`
   with footprints `R` and `W` and leaves `Q`.
6. Here: the footprints of different programs are disjoint, and the programs' results give `Post`.
-/
namespace Claims.PagedAttention

open TVA

/-- The sequence of program `i`: the grid's axis 0. -/
def sOf (d : Inst) (i : Nat) : Nat := i % d.num_seqs

/-- The head of program `i`: the grid's axis 1. -/
def hOf (d : Inst) (i : Nat) : Nat := i / d.num_seqs % d.num_heads

/-- The launch has `num_seqs · num_heads` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.num_seqs * d.num_heads := by
  simp [Launch.size, launch]

/-- Program `i` has `program_id(0) = sOf d i`. -/
theorem ctx_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = sOf d i := by
  simp [Launch.ctx, launch, sOf]

/-- Program `i` has `program_id(1) = hOf d i`. -/
theorem ctx_pid1 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 1 = hOf d i := by
  simp [Launch.ctx, launch, hOf]

/-- A program's sequence is below `num_seqs`. -/
theorem sOf_lt (d : Inst) {i : Nat} (hi : i < d.num_seqs * d.num_heads) : sOf d i < d.num_seqs := by
  have : 0 < d.num_seqs := by
    rcases Nat.eq_zero_or_pos d.num_seqs with h | h
    · rw [h] at hi; simp at hi
    · exact h
  exact Nat.mod_lt _ this

/-- A program's head is below `num_heads`. -/
theorem hOf_lt (d : Inst) {i : Nat} (hi : i < d.num_seqs * d.num_heads) : hOf d i < d.num_heads := by
  have : 0 < d.num_heads := by
    rcases Nat.eq_zero_or_pos d.num_heads with h | h
    · rw [h] at hi; simp at hi
    · exact h
  exact Nat.mod_lt _ this

/-- A program is determined by its sequence and head. -/
theorem idx_inj (d : Inst) {i j : Nat} (hi : i < d.num_seqs * d.num_heads)
    (hj : j < d.num_seqs * d.num_heads) (hs : sOf d i = sOf d j) (hh : hOf d i = hOf d j) : i = j := by
  have hS : 0 < d.num_seqs := by
    rcases Nat.eq_zero_or_pos d.num_seqs with h | h
    · rw [h] at hi; simp at hi
    · exact h
  have h1 : i / d.num_seqs < d.num_heads := (Nat.div_lt_iff_lt_mul hS).mpr (by rwa [mul_comm] at hi)
  have h2 : j / d.num_seqs < d.num_heads := (Nat.div_lt_iff_lt_mul hS).mpr (by rwa [mul_comm] at hj)
  unfold hOf at hh
  rw [Nat.mod_eq_of_lt h1, Nat.mod_eq_of_lt h2] at hh
  unfold sOf at hs
  have e1 := Nat.div_add_mod i d.num_seqs
  have e2 := Nat.div_add_mod j d.num_seqs
  rw [hh, hs] at e1
  omega

/-- Program `(s, h)`'s reads lie inside the buffers. -/
theorem R_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {s h : Nat} (hs : s < d.num_seqs)
    (hh : h < d.num_heads) :
    (∀ x ∈ qSet d s h, x < d.q.len) ∧ (∀ x ∈ btSet d init s, x < d.block_tables.len) ∧
      (∀ x ∈ tokSet d init s h, x < d.k_cache.len ∧ x < d.v_cache.len) := by
  obtain ⟨-, -, -, -, -, -, hsok⟩ := hpre
  obtain ⟨n, hn_read, hn0, hn_le, hn22, hbt⟩ := hsok s hs
  have hnl : seqLen d init s = n.toInt.toNat := by unfold seqLen val32; rw [hn_read]
  have hP0 : 0 < d.PAGE_SIZE := by obtain ⟨b, hb, he⟩ := hd.page_size; rw [he]; positivity
  refine ⟨?_, ?_, ?_⟩
  · rintro _ ⟨e, he, rfl⟩
    exact hd.q_len s h e hs hh he
  · rintro _ ⟨j, hj, rfl⟩
    have : j < d.max_pages := by
      have h1 : j * d.PAGE_SIZE < d.max_pages * d.PAGE_SIZE := by
        rw [hnl] at hj; exact lt_of_lt_of_le hj (Int.toNat_le.mpr hn_le)
      exact Nat.lt_of_mul_lt_mul_right h1
    exact hd.bt_len s j hs this
  · rintro _ ⟨t, e, ht, he, rfl⟩
    have hq : 0 < qpk d := by
      unfold qpk
      obtain ⟨k, hk⟩ := hd.kv_dvd
      have := hd.kv_pos
      rw [hk, Nat.mul_div_cancel_left _ this]
      rcases Nat.eq_zero_or_pos k with h0 | h0
      · rw [h0] at hk; omega
      · exact h0
    have hgk : h / qpk d < d.num_kv_heads := by
      have hH : d.num_heads = d.num_kv_heads * qpk d := by
        unfold qpk; exact (Nat.mul_div_cancel' hd.kv_dvd).symm
      rw [Nat.div_lt_iff_lt_mul hq]; rw [hH] at hh; exact hh
    have htP : t / d.PAGE_SIZE * d.PAGE_SIZE < seqLen d init s :=
      lt_of_le_of_lt (Nat.div_mul_le_self _ _) ht
    obtain ⟨b, hb_read, hb0, hbN⟩ := hbt (t / d.PAGE_SIZE) (by rw [hnl] at htP; omega)
    have hpg : pageOf d init s t < d.num_pages := by
      have h1 : b.toInt.toNat < d.num_pages := by omega
      unfold pageOf val32; rw [hb_read]; exact h1
    have hr : t % d.PAGE_SIZE < d.PAGE_SIZE := Nat.mod_lt _ hP0
    exact hd.kv_len _ _ _ _ hpg hr hgk he

/-- The elements of `out` a program writes lie inside `out`. -/
theorem W_lt {d : Inst} (hd : d.ok) {s h : Nat} (hs : s < d.num_seqs) (hh : h < d.num_heads) :
    ∀ x ∈ oSet d s h, x < d.out.len := by
  rintro _ ⟨e, he, rfl⟩
  exact hd.out_len s h e hs hh he

/-- The written row is disjoint from every program's reads. -/
theorem W_disj_R {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {s h s' h' : Nat}
    (hs : s < d.num_seqs) (hh : h < d.num_heads) (hs' : s' < d.num_seqs) (hh' : h' < d.num_heads) :
    Disjoint (W d s h) (R d init s' h') := by
  obtain ⟨hq, hbt, hkv⟩ := R_lt hd hpre hs' hh'
  have hW := W_lt hd hs hh
  obtain ⟨hdq, hdk, hdv, hdb, hdl⟩ := hd.disj
  unfold R W
  refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_) ?_
  · exact d.out.elems_disjoint_of d.q hdq hW hq
  · refine d.out.elems_disjoint_of d.seq_lens hdl hW ?_
    rintro j (rfl : j = s')
    exact lt_of_lt_of_le hs' hd.sl_len
  · exact d.out.elems_disjoint_of d.block_tables hdb hW hbt
  · exact d.out.elems_disjoint_of d.k_cache hdk hW (fun x hx => (hkv x hx).1)
  · exact d.out.elems_disjoint_of d.v_cache hdv hW (fun x hx => (hkv x hx).2)

/-- Different programs write different rows. -/
theorem W_disj_W {d : Inst} (hd : d.ok) {s h s' h' : Nat} (hs : s < d.num_seqs)
    (hh : h < d.num_heads) (hs' : s' < d.num_seqs) (hh' : h' < d.num_heads)
    (hne : ¬ (s = s' ∧ h = h')) : Disjoint (W d s h) (W d s' h') := by
  refine d.out.elems_disjoint ?_
  rw [Set.disjoint_left]
  rintro _ ⟨e, he, rfl⟩ ⟨e', he', hee⟩
  have := hd.out_inj s' h' e' s h e hs' hh' he' hs hh he hee.symm
  exact hne ⟨this.1.symm, this.2.1.symm⟩

/-- **The claim.** Each program `(s, h)` is safe (`safe_program`); different programs write
different rows and no program writes what any reads; and the stored rows give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hsize := launch_size d
  have hbs : ∀ sh ∈ (launch d).blocks, sh.TritonOk := by
    obtain ⟨a, ha, hDa⟩ := hd.head_dim
    obtain ⟨b, hb, hPb⟩ := hd.page_size
    have hDle : d.HEAD_DIM ≤ 2 ^ 8 := by rw [hDa]; exact Nat.pow_le_pow_right (by norm_num) ha
    have hPle : d.PAGE_SIZE ≤ 2 ^ 8 := by rw [hPb]; exact Nat.pow_le_pow_right (by norm_num) hb
    have hDP : d.PAGE_SIZE * d.HEAD_DIM ≤ 2 ^ 20 :=
      (Nat.mul_le_mul hPle hDle).trans (by norm_num)
    intro sh hsh
    simp only [launch, List.mem_cons, List.mem_nil_iff, or_false] at hsh
    rcases hsh with rfl | rfl | rfl | rfl | rfl
    · exact ⟨by simpa using ⟨b, hPb⟩, by simpa using hPle.trans (by norm_num)⟩
    · exact ⟨by simpa using ⟨a, hDa⟩, by simpa using hDle.trans (by norm_num)⟩
    · exact ⟨by simp; exact ⟨⟨b, hPb⟩, ⟨0, rfl⟩⟩, by simpa using hPle.trans (by norm_num)⟩
    · exact ⟨by simp; exact ⟨⟨0, rfl⟩, ⟨a, hDa⟩⟩, by simpa using hDle.trans (by norm_num)⟩
    · exact ⟨by simp; exact ⟨⟨b, hPb⟩, ⟨a, hDa⟩⟩, by simpa using hDP⟩
  have hvalid : (launch d).Valid := by
    refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, hbs⟩
    · simpa [launch] using hd.seqs_lt
    · simpa [launch] using hd.heads_lt
    · simp [launch]
    · rw [hsize]; exact hd.grid_lt
  refine correct_of_safe (fun i => R d init (sOf d i) (hOf d i)) (fun i => W d (sOf d i) (hOf d i))
    (fun i m => Q d init (sOf d i) (hOf d i) m) hvalid (fun i => ?_) ?_ ?_ ?_
  · -- program `i`
    have hi : (i : Nat) < d.num_seqs * d.num_heads := by rw [← hsize]; exact i.isLt
    exact safe_program hd hP hpre (sOf_lt d hi) (hOf_lt d hi) _ (ctx_pid0 d i) (ctx_pid1 d i)
  · -- no program writes what another reads or writes
    intro i j hij
    have hi : (i : Nat) < d.num_seqs * d.num_heads := by rw [← hsize]; exact i.isLt
    have hj : (j : Nat) < d.num_seqs * d.num_heads := by rw [← hsize]; exact j.isLt
    refine Disjoint.union_right (W_disj_R hd hpre (sOf_lt d hi) (hOf_lt d hi) (sOf_lt d hj) (hOf_lt d hj))
      (W_disj_W hd (sOf_lt d hi) (hOf_lt d hi) (sOf_lt d hj) (hOf_lt d hj) fun h' => ?_)
    exact hij (Fin.ext (idx_inj d hi hj h'.1 h'.2))
  · -- no program reads what it writes
    intro i
    have hi : (i : Nat) < d.num_seqs * d.num_heads := by rw [← hsize]; exact i.isLt
    exact (W_disj_R hd hpre (sOf_lt d hi) (hOf_lt d hi) (sOf_lt d hi) (hOf_lt d hi)).symm
  · -- the programs' results give `Post`
    intro fin hloc hfr
    refine ⟨fun s h e hs hh he => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
    · have hi : s + d.num_seqs * h < d.num_seqs * d.num_heads := by
        have : s + d.num_seqs * h < d.num_seqs * (h + 1) := by rw [Nat.mul_succ]; omega
        exact lt_of_lt_of_le this (Nat.mul_le_mul_left _ hh)
      have hs' : sOf d (s + d.num_seqs * h) = s := by
        unfold sOf; rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hs]
      have hh' : hOf d (s + d.num_seqs * h) = h := by
        unfold hOf
        rw [Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hs, zero_add, Nat.mod_eq_of_lt hh]
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨s + d.num_seqs * h, by rw [hsize]; exact hi⟩
      rw [hs', hh'] at hQ hagree
      obtain ⟨c, hc, hbd⟩ := hQ e he
      refine ⟨c, ?_, hbd⟩
      rw [Mem.read_congr fun x hx => hagree x (Or.inr (d.out.span_sub_elems (S := oSet d s h)
        (j := outIdx d s h e) ⟨e, he, rfl⟩ hx))]
      exact hc
    · have hi : (i : Nat) < d.num_seqs * d.num_heads := by rw [← hsize]; exact i.isLt
      refine d.out.elems_mono (fun x hx => ?_) hzi
      obtain ⟨e, he, rfl⟩ := hx
      exact ⟨sOf d i, hOf d i, e, sOf_lt d hi, hOf_lt d hi, he, rfl⟩

end Claims.PagedAttention
