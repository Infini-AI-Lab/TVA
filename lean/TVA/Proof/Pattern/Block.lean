import TVA.Proof.Basic
import TVA.Proof.Int

/-!
# One-dimensional blocks

The most common Triton access pattern:

```python
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n
x = tl.load(x_ptr + offs, mask=mask)
```

In the model, `offs` is `Block.offs p B` (`tl.program_id ctx 0 * IntV.ofNat B + tl.arange 0 B`,
program `p = ctx.pid 0`), `mask` is `Block.mask p B n`, and the pointers are `b.ptr 0 + offs` for
the buffer `b` that `x_ptr` points into. Program `p` handles elements `p * B + l` for `l < B`, those
below `n` (`Block.Tile p B n j`); their bytes are `b.tile p B n`.

* **Lanes.** Lane `l` of `offs` has value `p * B + l` (`Block.lane_toInt`) and is enabled exactly
  when `p * B + l < n` (`Block.mask_lane`), when program `p`'s `int32` arithmetic does not wrap:
  `Block.Fits p B`, its largest offset `p * B + B - 1` is below `2^31`.
* **When it fits.** Every program of the grid `cdiv(n, B)` fits when `n + B ≤ 2^31`
  (`fits_of_grid`), or when `n < 2^31` and `B` divides `2^31` (`fits_of_grid_dvd`), as every power
  of two up to `2^31` does (`fits_of_grid_pow2`; Triton's block sizes are powers of two). A single
  program fits when its first offset is below `2^31` and `B` divides `2^31` (`fits_of_dvd`,
  `fits_of_pow2`, `fits_of_lt_pow2`).
* **Accesses.** `safe_load_block`, `safe_store_block` give each enabled lane its element; the
  footprint is `b.tile p B n` (`addrs_block`, `img_block`, `accBytes_block`).
* **Footprints.** Tiles of different programs are disjoint (`Buf.tile_disjoint`); every tile lies
  in the first `n` elements (`Buf.tile_sub_elems`), and element `j < n` is in the tile of program
  `j / B` (`Block.tile_div`).

The bound is needed on every lane, not only the enabled ones: the mask compares the offsets
themselves, so a lane whose offset wrapped to a negative value would be enabled. For a power-of-two
block this is no limit (`fits_of_lt_pow2`). The 2-D tiles of `TVA.Proof.Pattern.Tile` compute
their masks from rows and columns and need bounds on the enabled lanes only.
-/
namespace TVA

namespace Block

/-- `pid * BLOCK + tl.arange(0, BLOCK)` in `int32` for program `p`, block size `B`. -/
def offs (p B : Nat) : Tensor [B - 0] I32 := (IntV.ofNat p * IntV.ofNat B : I32) + tl.arange 0 B

/-- `offs < n` for the `int32` scalar `n`. -/
def mask (p B n : Nat) : Tensor [B - 0] Bool := (offs p B).ltS (IntV.ofNat n)

/-- Element `j` belongs to program `p`'s block and is below `n`. -/
def Tile (p B n j : Nat) : Prop := p * B ≤ j ∧ j < p * B + B ∧ j < n

/-- Program `p`'s block arithmetic fits `int32`: its largest offset `p * B + B - 1` is below
`2^31`. -/
def Fits (p B : Nat) : Prop := p * B + B ≤ 2 ^ 31

/-- **The lanes of `offs`.** Lane `l` has value `p * B + l` when the block fits. -/
theorem lane_toInt {p B : Nat} (hfit : Fits p B) (l : Fin (B - 0)) :
    ((offs p B).get1 l).toInt = ((p * B + l : Nat) : Int) := by
  have hpB : p * B < 2 ^ 31 := by
    unfold Fits at hfit
    rcases Nat.eq_zero_or_pos B with rfl | _
    · simp
    · omega
  unfold offs
  rw [IntV.ofNat_mul_ofNat]
  exact IntV.I32.toInt_sadd_arange (IntV.I32.toInt_ofNat hpB) hfit l

/-- **The lanes of `mask`.** Lane `l` is enabled exactly when `p * B + l < n`. -/
theorem mask_lane {p B n : Nat} (hfit : Fits p B) (hn : n < 2 ^ 31) (l : Fin (B - 0)) :
    (mask p B n).get1 l = decide (p * B + l < n) := by
  show IntV.lt ((offs p B).get1 l) (IntV.ofNat n) = _
  rw [IntV.lt_eq, lane_toInt hfit, IntV.I32.toInt_ofNat hn]
  simp only [Nat.cast_lt]

private theorem lane_tile {p B n : Nat} (l : Fin (B - 0)) (h : p * B + l < n) :
    Tile p B n (p * B + l) := by
  have := l.isLt; simp only [Nat.sub_zero] at this
  exact ⟨by omega, by omega, h⟩

private theorem tile_lane {p B n j : Nat} (h : Tile p B n j) :
    ∃ l : Fin (B - 0), j = p * B + l := by
  obtain ⟨h1, h2, -⟩ := h
  exact ⟨⟨j - p * B, by simp only [Nat.sub_zero]; omega⟩, by simp only; omega⟩

/-- Every element below `n` is in the block of program `j / B`. Use it to find the program that
wrote element `j` of the output. -/
theorem tile_div {B n j : Nat} (hB : 0 < B) (hj : j < n) : Tile (j / B) B n j :=
  ⟨Nat.div_mul_le_self j B, by have := Nat.lt_div_mul_add (a := j) hB; linarith, hj⟩

/-- An element is in the block of at most one program. -/
theorem tile_unique {p q B n j : Nat} (hp : Tile p B n j) (hq : Tile q B n j) : p = q := by
  obtain ⟨h1, h2, -⟩ := hp
  obtain ⟨h3, h4, -⟩ := hq
  rcases lt_trichotomy p q with h | h | h
  · have : (p + 1) * B ≤ q * B := Nat.mul_le_mul_right B h
    nlinarith
  · exact h
  · have : (q + 1) * B ≤ p * B := Nat.mul_le_mul_right B h
    nlinarith

/-- **Every program of the grid `cdiv(n, B)` fits** when `n + B ≤ 2^31`. -/
theorem fits_of_grid {p B n : Nat} (hnB : n + B ≤ 2 ^ 31) (hp : p < (n + B - 1) / B) :
    Fits p B := by
  have h1 : (p + 1) * B ≤ (n + B - 1) / B * B := Nat.mul_le_mul_right B hp
  have h2 : (n + B - 1) / B * B ≤ n + B - 1 := Nat.div_mul_le_self _ _
  unfold Fits; rw [Nat.succ_mul] at h1; omega

/-- **A program fits** when `B` divides `2^31` and its first offset `p * B` is below `2^31`: a
multiple of `B` below `2^31` is at most `2^31 - B`. -/
theorem fits_of_dvd {p B : Nat} (hB : B ∣ 2 ^ 31) (hp : p * B < 2 ^ 31) : Fits p B := by
  obtain ⟨c, hc⟩ := hB
  have hpc : p < c := by
    rw [hc, Nat.mul_comm B c] at hp
    exact Nat.lt_of_mul_lt_mul_right hp
  unfold Fits
  have : (p + 1) * B ≤ c * B := Nat.mul_le_mul_right B hpc
  rw [Nat.succ_mul] at this; rw [hc, Nat.mul_comm B c]; omega

/-- **Every program of the grid `cdiv(n, B)` fits** when `n < 2^31` and `B` divides `2^31`. -/
theorem fits_of_grid_dvd {p B n : Nat} (hB : B ∣ 2 ^ 31) (hn : n < 2 ^ 31)
    (hp : p < (n + B - 1) / B) : Fits p B := by
  have hB0 : 0 < B := Nat.pos_of_dvd_of_pos hB (by norm_num)
  have h1 : (p + 1) * B ≤ (n + B - 1) / B * B := Nat.mul_le_mul_right B hp
  have h2 : (n + B - 1) / B * B ≤ n + B - 1 := Nat.div_mul_le_self _ _
  rw [Nat.succ_mul] at h1
  exact fits_of_dvd hB (by omega)

/-- **Every program of the grid `cdiv(n, B)` fits** for a power-of-two block size `B = 2^e`,
`e ≤ 31`, when `n < 2^31`: the usual way to discharge `Fits` for a 1-D grid. -/
theorem fits_of_grid_pow2 {p B n e : Nat} (hB : B = 2 ^ e) (he : e ≤ 31) (hn : n < 2 ^ 31)
    (hp : p < (n + B - 1) / B) : Fits p B :=
  fits_of_grid_dvd (hB ▸ Nat.pow_dvd_pow 2 he) hn hp

/-- `fits_of_dvd` for a power-of-two block size `B = 2^e`, `e ≤ 31`: a program fits when its first
offset `p * B` is below `2^31`. -/
theorem fits_of_pow2 {p B e : Nat} (hB : B = 2 ^ e) (he : e ≤ 31) (hp : p * B < 2 ^ 31) :
    Fits p B :=
  fits_of_dvd (hB ▸ Nat.pow_dvd_pow 2 he) hp

/-- **A program with an element fits**: for a power-of-two block, a program with an element below
`n` (`p * B < n`, the last partial block included) fits when `n ≤ 2^31`. Use it when the grid is
not `cdiv(n, B)` (a persistent loop, a guard `if pid * B < n`). -/
theorem fits_of_lt_pow2 {p B n e : Nat} (hB : B = 2 ^ e) (he : e ≤ 31) (hp : p * B < n)
    (hn : n ≤ 2 ^ 31) : Fits p B :=
  fits_of_pow2 hB he (by omega)

end Block

open Block

namespace Buf

variable {d : DType}

/-- The bytes of the elements of `b` in program `p`'s block (`Block.Tile p B n`): the footprint of
a 1-D block access. -/
def tile (b : Buf d) (p B n : Nat) : Set Int := b.elems {j | Tile p B n j}

/-- A block's bytes lie in `b` when `n ≤ b.len`. -/
theorem tile_sub_bytes (b : Buf d) {p B n : Nat} (hlen : n ≤ b.len) : b.tile p B n ⊆ b.bytes :=
  b.elems_sub_bytes fun _ hj => by have := hj.2.2; omega

/-- A block's bytes lie in the first `n` elements of `b`, the set a claim's specification names.
Use it to move a footprint fact from the claim to a program. -/
theorem tile_sub_elems (b : Buf d) {p B n : Nat} : b.tile p B n ⊆ b.elems {j | j < n} :=
  b.elems_mono fun _ hj => hj.2.2

/-- **The blocks of two programs are disjoint**: the footprints of a 1-D grid do not overlap. -/
theorem tile_disjoint (b : Buf d) {p q B n : Nat} (hpq : p ≠ q) :
    Disjoint (b.tile p B n) (b.tile q B n) :=
  b.elems_disjoint (Set.disjoint_left.mpr fun _ hj hj' => hpq (tile_unique hj hj'))

/-- The bytes of an element of the block lie in the block's bytes. -/
theorem span_sub_tile (b : Buf d) {p B n j : Nat} (h : Tile p B n j) :
    Mem.span d (b.ptr (j : Int)).addr ⊆ b.tile p B n :=
  b.span_sub_elems h

end Buf

/-! ## Accesses -/

section Access
variable {d : DType} {p B n : Nat}

/-- **The lanes of a block access, as buffer elements**: lane `l` of `b.ptr 0 + offs` under
`mask` addresses element `p * B + l` when `p * B + l < n`. -/
theorem addrs_block (b : Buf d) (hfit : Fits p B) (hn : n < 2 ^ 31) :
    b.Addrs (tl.addrs (b.ptr 0 + offs p B) (some (mask p B n)))
      (fun i => p * B + i.1) (fun i => decide (p * B + i.1 < n)) := by
  refine Buf.addrs_of_get b (fun i => mask_lane hfit hn i.1) fun ⟨l, ⟨⟩⟩ _ => ?_
  show (b.ptr 0).offset ((offs p B).get1 l).toInt = _
  rw [lane_toInt hfit, Buf.ptr_offset, zero_add]

/-- The elements a block access touches are the block's (`Block.Tile`). -/
theorem img_block : Buf.img (s := [B - 0]) (fun i => p * B + i.1)
    (fun i => decide (p * B + i.1 < n)) = {j | Tile p B n j} := by
  ext j; constructor
  · rintro ⟨⟨l, ⟨⟩⟩, h, rfl⟩; exact lane_tile l (by simpa using h)
  · intro hj
    obtain ⟨l, rfl⟩ := tile_lane hj
    exact ⟨(l, ()), by simpa using hj.2.2, rfl⟩

/-- The bytes a block access touches: `b.tile p B n`. -/
theorem accBytes_block (b : Buf d) (hfit : Fits p B) (hn : n < 2 ^ 31) :
    accBytes d (tl.addrs (b.ptr 0 + offs p B) (some (mask p B n))) = b.tile p B n := by
  rw [(addrs_block b hfit hn).accBytes, img_block]; rfl

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **Block load** `tl.load(x_ptr + offs, mask=offs < n, other=…)`. Each enabled lane gets its
element; a masked-off lane gets `other` when it is given, and is unspecified otherwise. -/
theorem safe_load_block (b : Buf d) (hfit : Fits p B) (hn : n < 2 ^ 31) (hlen : n ≤ b.len)
    (hv : b.Valid m) (hR : b.tile p B n ⊆ R) {other : Option (Tensor [B - 0] d.Val)}
    {k : Tensor [B - 0] d.Val → Prog α}
    (hk : ∀ v : Tensor [B - 0] d.Val,
      (∀ l : Fin (B - 0), p * B + l < n →
        m.read d (b.ptr ((p * B + l : Nat) : Int)).addr = some (v.get1 l)) →
      (∀ l : Fin (B - 0), n ≤ p * B + l → ∀ o, other = some o → v.get1 l = o.get1 l) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load (b.ptr 0 + offs p B) (some (mask p B n)) other >>= k) :=
  safe_load_buf b (addrs_block b hfit hn) (fun _ h => by simp at h; omega) hv
    (by rw [img_block]; exact hR) fun v h1 h2 =>
      hk v (fun l hl => h1 (l, ()) (by simpa using hl))
        (fun l hl o ho => h2 (l, ()) (by simpa using hl) o ho)

/-- **Block store** `tl.store(out_ptr + offs, vals, mask=offs < n)`. Afterwards each enabled
lane's element holds its value, nothing outside the block changed, and every present byte is still
present. The rest of the program may not write the block again. -/
theorem safe_store_block (b : Buf d) (hfit : Fits p B) (hn : n < 2 ^ 31) (hlen : n ≤ b.len)
    (hv : b.Valid m) (hW : b.tile p B n ⊆ W) {vals : Tensor [B - 0] d.Val} {k : Unit → Prog α}
    (hk : ∀ m', (∀ l : Fin (B - 0), p * B + l < n →
        m'.read d (b.ptr ((p * B + l : Nat) : Int)).addr = some (vals.get1 l)) →
      (∀ x, x ∉ b.tile p B n → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.tile p B n) Q m' (k ())) :
    Safe E R W Q m (tl.store (b.ptr 0 + offs p B) vals (some (mask p B n)) >>= k) := by
  refine safe_store_buf b (addrs_block b hfit hn) (fun _ h => by simp at h; omega) ?_ hv
    (by rw [img_block]; exact hW) fun m' h1 h2 h3 => by
      rw [img_block]
      exact hk m' (fun l hl => h1 (l, ()) (by simpa using hl))
        (fun x hx => h2 x (by rw [img_block]; exact hx)) h3
  rintro ⟨l, ⟨⟩⟩ ⟨l', ⟨⟩⟩ - - h
  have : (l : Nat) = l' := by simp only at h; omega
  rw [Fin.ext this]

end Access

end TVA
