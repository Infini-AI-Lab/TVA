import Proof.Idx

/-!
# `kv_cache_write_bf16`: the postcondition from the programs' results
-/

namespace Claims.KvCacheWrite

open TVA

/-- `num_heads * head_size` fits in the tiles the grid launches. -/
theorem numVals_le_tiles (d : Inst) (hd : d.ok) : numVals d ≤ numTiles d * d.TILE_SIZE := by
  have hT := tile_pos' hd
  have hmod : d.TILE_SIZE * numTiles d + (numVals d + d.TILE_SIZE - 1) % d.TILE_SIZE =
      numVals d + d.TILE_SIZE - 1 := by
    have := Nat.div_add_mod (numVals d + d.TILE_SIZE - 1) d.TILE_SIZE
    simpa only [numTiles] using this
  have hr := Nat.mod_lt (numVals d + d.TILE_SIZE - 1) hT
  rw [Nat.mul_comm (numTiles d) d.TILE_SIZE]
  omega

/-- The destination element of an enabled lane is one the launch writes. -/
theorem img_subset_dstSet' {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d) :
    Buf.img (gi d init p) (en d init p) ⊆ dstSet d init := by
  intro k hk
  obtain ⟨l, hen, rfl⟩ := hk
  refine ⟨tk d p, pos d p l / d.head_size, pos d p l % d.head_size, tk_lt hp, en_slot hen, ?_, ?_, rfl⟩
  · have hhs : 0 < d.head_size := head_pos hd hp
    have h := pos_lt_vals hen
    simp only [numVals] at h
    rwa [Nat.div_lt_iff_lt_mul hhs]
  · exact Nat.mod_lt _ (head_pos hd hp)

/-- A component `(h, e)` of a token names a lane of the grid. -/
theorem lane_exists {d : Inst} (hd : d.ok) {init : Mem} {t h e : Nat}
    (ht : t < d.num_tokens) (h0 : 0 ≤ slotAt d init t)
    (hh : h < d.num_heads) (he : e < d.head_size) :
    ∃ (p : Nat) (l : Idx [d.TILE_SIZE]), p < d.num_tokens * numTiles d ∧ tk d p = t ∧
      pos d p l = h * d.head_size + e ∧ en d init p l = true ∧
      gi d init p l = dstIdx d init t h e := by
  have hT : 0 < d.TILE_SIZE := tile_pos' hd
  have hhs : 0 < d.head_size := Nat.lt_of_le_of_lt (Nat.zero_le e) he
  have hnt : 0 < d.num_tokens := Nat.lt_of_le_of_lt (Nat.zero_le t) ht
  set ppos := h * d.head_size + e with hpposdef
  set tit := ppos / d.TILE_SIZE with htitdef
  set p := tit * d.num_tokens + t with hpdef
  have hppos : ppos < numVals d := by
    have hsen : h + 1 ≤ d.num_heads := Nat.succ_le_of_lt hh
    have hm := Nat.mul_le_mul_right d.head_size hsen
    rw [Nat.succ_mul] at hm
    simp only [numVals]
    omega
  have hnvle : numVals d ≤ numTiles d * d.TILE_SIZE := numVals_le_tiles d hd
  have htit : tit < numTiles d :=
    (Nat.div_lt_iff_lt_mul hT).mpr (lt_of_lt_of_le hppos hnvle)
  have hp : p < d.num_tokens * numTiles d := by
    have h1 : tit + 1 ≤ numTiles d := Nat.succ_le_of_lt htit
    have h2 := Nat.mul_le_mul_right d.num_tokens h1
    rw [Nat.succ_mul] at h2
    rw [Nat.mul_comm (numTiles d) d.num_tokens] at h2
    omega
  have htk : tk d p = t := by
    simp only [tk]
    rw [hpdef, show tit * d.num_tokens + t = d.num_tokens * tit + t by rw [Nat.mul_comm]]
    rw [Nat.mul_add_mod, Nat.mod_eq_of_lt ht]
  have hti : ti d p = tit := by
    have hpdiv : p / d.num_tokens = tit := by
      rw [hpdef, show tit * d.num_tokens + t = d.num_tokens * tit + t by rw [Nat.mul_comm]]
      rw [Nat.mul_add_div hnt, Nat.div_eq_of_lt ht, Nat.add_zero]
    simp only [ti]
    rw [hpdiv, Nat.mod_eq_of_lt htit]
  have hpos : pos d p ((⟨ppos % d.TILE_SIZE, Nat.mod_lt _ hT⟩, ()) : Idx [d.TILE_SIZE]) = ppos := by
    show ti d p * d.TILE_SIZE + (ppos % d.TILE_SIZE) = ppos
    rw [hti, show tit * d.TILE_SIZE = d.TILE_SIZE * tit by rw [Nat.mul_comm]]
    exact Nat.div_add_mod ppos d.TILE_SIZE
  have hen : en d init p ((⟨ppos % d.TILE_SIZE, Nat.mod_lt _ hT⟩, ()) : Idx [d.TILE_SIZE]) = true := by
    simp only [en, hpos, htk, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨hppos, h0⟩
  have hdiv : pos d p ((⟨ppos % d.TILE_SIZE, Nat.mod_lt _ hT⟩, ()) : Idx [d.TILE_SIZE]) /
      d.head_size = h := by
    rw [hpos, hpposdef, show h * d.head_size + e = d.head_size * h + e by rw [Nat.mul_comm]]
    rw [Nat.mul_add_div hhs, Nat.div_eq_of_lt he, Nat.add_zero]
  have hmod : pos d p ((⟨ppos % d.TILE_SIZE, Nat.mod_lt _ hT⟩, ()) : Idx [d.TILE_SIZE]) %
      d.head_size = e := by
    rw [hpos, hpposdef, show h * d.head_size + e = d.head_size * h + e by rw [Nat.mul_comm],
      Nat.mul_add_mod, Nat.mod_eq_of_lt he]
  have hgi : gi d init p ((⟨ppos % d.TILE_SIZE, Nat.mod_lt _ hT⟩, ()) : Idx [d.TILE_SIZE]) =
      dstIdx d init t h e := by
    simp only [gi, hdiv, hmod, htk]
  exact ⟨p, ((⟨ppos % d.TILE_SIZE, Nat.mod_lt _ hT⟩, ()) : Idx [d.TILE_SIZE]), hp, htk, hpos,
    hen, hgi⟩

/-- Agreement on a program's write footprint turns a read of `m` into a read of `fin`. -/
theorem fin_read_eq {d : Inst} {init fin m : Mem} {p : Nat}
    (hagree : ∀ x ∈ R d ∪ W d init p, fin x = m x)
    (b : Buf d.mode.cd) {j : Nat}
    (hj : j ∈ Buf.img (gi d init p) (en d init p))
    (hsub : b.elems (Buf.img (gi d init p) (en d init p)) ⊆ W d init p) :
    fin.read d.mode.cd (b.ptr (j : Int)).addr = m.read d.mode.cd (b.ptr (j : Int)).addr := by
  apply Mem.read_congr
  intro x hx
  exact hagree x (Or.inr (hsub (b.span_sub_elems hj hx)))

/-- The programs' results give `Post`: each cache element is read back from the program that
wrote it, and every other byte is unchanged. -/
theorem post_of {d : Inst} (hd : d.ok) {init : Mem} {fin : Mem}
    (hQ : ∀ i : Fin (launch d).size, ∃ m : Mem, Q d init (i : Nat) m ∧
      ∀ x ∈ R d ∪ W d init (i : Nat), fin x = m x)
    (hframe : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d init (i : Nat)) → fin x = init x) :
    Post d init fin := by
  refine ⟨?_, ?_⟩
  · intro t ht h0 h e hh he
    obtain ⟨p, l, hp, htk, hpos, hen, hgi⟩ := lane_exists hd (init := init) ht h0 hh he
    obtain ⟨m, hQm, hagree⟩ := hQ ⟨p, by rw [launch_size]; exact hp⟩
    obtain ⟨⟨xk, vk, hkread, hkcache, hkstored⟩, ⟨xv, vv, hvread, hvcache, hvstored⟩⟩ :=
      hQm l hen
    have hj : gi d init p l ∈ Buf.img (gi d init p) (en d init p) := ⟨l, hen, rfl⟩
    constructor
    · refine ⟨xk, vk, ?_, ?_, hkstored⟩
      · simpa only [htk, hpos] using hkread
      · have hspan := fin_read_eq (init := init) (fin := fin) (m := m) (p := p)
          (b := d.key_cache) (j := gi d init p l) hagree hj (fun x hx => Or.inl hx)
        rw [hgi] at hspan
        rw [hspan]
        rw [hgi] at hkcache
        exact hkcache
    · refine ⟨xv, vv, ?_, ?_, hvstored⟩
      · simpa only [htk, hpos] using hvread
      · have hspan := fin_read_eq (init := init) (fin := fin) (m := m) (p := p)
          (b := d.value_cache) (j := gi d init p l) hagree hj (fun x hx => Or.inr hx)
        rw [hgi] at hspan
        rw [hspan]
        rw [hgi] at hvcache
        exact hvcache
  · intro z hz
    exact hframe z (fun i => by
      intro hmem
      apply hz
      have hp : (i : Nat) < d.num_tokens * numTiles d := by
        simpa [launch_size] using i.isLt
      simp only [W, writeSet] at hmem ⊢
      rcases hmem with h | h
      · exact Or.inl (Buf.elems_mono d.key_cache (img_subset_dstSet' hd hp) h)
      · exact Or.inr (Buf.elems_mono d.value_cache (img_subset_dstSet' hd hp) h))

end Claims.KvCacheWrite
