import Claim
import TVA.Proof
import Proof.Walk

/-!
# The proof of `attn_fwd_f16`

One program computes one `128`-row query tile of one head: it loops over the keys in blocks of
`64` with an online softmax and stores the tile of `o` and of `M`.

1. `Proof.Online`, `Proof.Step`, `Proof.Row`, `Proof.Final`: the error analysis of a row, as pure
   real arithmetic. A row's state after `t` blocks is within a closed form of the exact one
   (`InvM`, `InvL`, `InvA`); each block preserves it (`trip_m`, `trip_l`, `trip_a`); at the end,
   `acc / l`, the rounding to `float16` and `m + log₂ l` are within the claim's tolerances
   (`ratio_facts`, `out_bound`, `lse_bound`, using `Proof.Relax`: the fine bound is below `epsO`
   and `epsM`).
2. `Proof.Lane`, `Proof.Trip`, `Proof.Trips`, `Proof.Plain`, `Proof.Mask`: one trip of the key loop.
   Float operations become real-number statements (`Proof.Lane`); a row's trip, masked or not,
   turns the exponents and the rest of the operations into the next invariants (`trip_core`);
   all rows give the next `StateOK` (`trip_state`); the unmasked and the masked (causal) bodies
   are walked (`safe_plain_trip`, `safe_masked_trip`; masked lanes get weight `0`).
3. `Proof.Loop`, `Proof.Walk`: the `tl.range` loop is walked by trips (`safe_keyloop`), and one
   program line by line (`safe_program`): it reads `q`, `k`, `v`, writes its tile of `o` and `M`,
   and leaves `Qpost` (`Proof.Post`, `Proof.Tile`, `Proof.Link` connect tiles and rows to the claim).
4. `Proof.Outer`: the launch is valid and the programs' footprints are disjoint.
5. `main`: `correct_of_safe` composes the programs; the tiles cover `o` and `M`.
-/

namespace Claims.AttnFwd

open TVA

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have hG := G_pos hd
  have hsize := launch_size d
  have hbh : ∀ i : Fin (launch d).size, (i : Nat) / G d < d.Z * d.H := fun i =>
    (Nat.div_lt_iff_lt_mul hG).mpr (by
      have := lt_of_lt_of_eq i.isLt hsize; rwa [Nat.mul_comm] at this)
  have hsm : ∀ i : Fin (launch d).size, (i : Nat) % G d < G d := fun i => Nat.mod_lt _ hG
  refine correct_of_safe (fun _ => Rset d) (fun i => Wset d (i / G d) (i % G d))
    (fun i m => Qpost d init (i / G d) (i % G d) m) (launch_valid hd)
    (fun i => safe_program hd hP hpre i) ?_ ?_ ?_
  · intro i j hij
    refine Disjoint.union_right (Wset_disjoint_R hd (hbh i) (hsm i)) (Wset_disjoint hd (hbh i) (hsm i)
      (hbh j) (hsm j) fun ⟨h1, h2⟩ => hij (Fin.ext ?_))
    have := Nat.div_add_mod (i : Nat) (G d)
    have := Nat.div_add_mod (j : Nat) (G d)
    rw [h1, h2] at *
    omega
  · intro i
    exact (Wset_disjoint_R hd (hbh i) (hsm i)).symm
  · intro fin hloc hfr
    have hindex : ∀ bh n, bh < d.Z * d.H → n < d.N_CTX →
        ∃ i : Fin (launch d).size, (i : Nat) / G d = bh ∧ (i : Nat) % G d = n / 128 := by
      intro bh n hbh hn
      have hsm : n / 128 < G d := by
        have := G_mul hd; omega
      refine ⟨⟨n / 128 + G d * bh, ?_⟩, ?_, ?_⟩
      · rw [hsize]; nlinarith
      · show (n / 128 + G d * bh) / G d = bh
        rw [Nat.add_mul_div_left _ _ hG, Nat.div_eq_of_lt hsm, zero_add]
      · show (n / 128 + G d * bh) % G d = n / 128
        rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hsm]
    refine ⟨fun bh n e hbh hn he => ?_, fun bh n hbh hn => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
    · obtain ⟨i, h1, h2⟩ := hindex bh n hbh hn
      obtain ⟨m, hQ, hagree⟩ := hloc i
      rw [h1, h2] at hQ
      have hnr : n / 128 * 128 + n % 128 = n := by omega
      obtain ⟨c, hc, hspec⟩ := (hQ (n % 128) (Nat.mod_lt _ (by norm_num))).1 e he
      rw [hnr] at hc hspec
      refine ⟨c, ?_, hspec⟩
      have hmem : idx d bh n e ∈ oTile d bh (n / 128) :=
        ⟨n % 128, Nat.mod_lt _ (by norm_num), e, he, by rw [hnr]⟩
      have hs : (i : Nat) / G d = bh := h1
      rw [Mem.read_congr (m₂ := m) (fun x hx => hagree x (Or.inr (by
        rw [h1, h2]; exact Or.inl (d.o.span_sub_elems hmem hx))))]
      exact hc
    · obtain ⟨i, h1, h2⟩ := hindex bh n hbh hn
      obtain ⟨m, hQ, hagree⟩ := hloc i
      rw [h1, h2] at hQ
      have hnr : n / 128 * 128 + n % 128 = n := by omega
      obtain ⟨c, hc, hspec⟩ := (hQ (n % 128) (Nat.mod_lt _ (by norm_num))).2
      rw [hnr] at hc hspec
      refine ⟨c, ?_, hspec⟩
      have hmem : bh * d.N_CTX + n ∈ mTile d bh (n / 128) :=
        ⟨n % 128, Nat.mod_lt _ (by norm_num), by rw [hnr]⟩
      rw [Mem.read_congr (m₂ := m) (fun x hx => hagree x (Or.inr (by
        rw [h1, h2]; exact Or.inr (d.M.span_sub_elems hmem hx))))]
      exact hc
    · -- the written bytes are among the output bytes
      have hbhi := hbh i
      have hsmi := hsm i
      unfold Wset at hzi
      unfold outBytes
      rcases hzi with hzi | hzi
      · exact Or.inl (d.o.elems_mono (oTile_sub hd hbhi hsmi) hzi)
      · exact Or.inr (d.M.elems_mono (mTile_sub hd hbhi hsmi) hzi)

end Claims.AttnFwd
