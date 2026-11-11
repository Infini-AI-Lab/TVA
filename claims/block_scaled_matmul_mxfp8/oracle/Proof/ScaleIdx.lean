import Claim
import TVA.Proof
import Proof.Basic
import Proof.ScaleOff

/-!
# The loaded scale tiles

The kernel loads one `512`-byte block of the scale tensor per step and reshapes it
(`reshape [1,1,32,4,4]`, `permute [0,3,2,1,4]`, `reshape [128,4]`). Entry `(r, g)` of the result
is the scale byte `sa`/`sb` uses at element `(pm · 128 + r, 128 · t + 32 · g)`.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- The position of lane `(r, g)` of `a`'s scale tile in its `512`-byte block. -/
def aU (r g : Nat) : Nat := 16 * (r % 32) + 4 * (r / 32) + g

/-- The position of lane `(n, g)` of `b`'s scale tile in its `512`-byte block. -/
def bU (n g : Nat) : Nat := 16 * (n % 32) + 4 * (n / 32 % 4) + g

/-- The position of lane `(n, g)` of `b`'s `[1,2,1,2,256]` scale tile. -/
def bPos (n g : Nat) : Nat := (n / 128) * 512 + bU n g

/-- The position `aU r g` is inside `a`'s scale block. -/
theorem aU_lt {r g : Nat} (hr : r < 128) (hg : g < 4) : aU r g < [1, 1, 1, 2, 256].prod := by
  unfold aU
  simp only [List.prod_cons, List.prod_nil]
  omega

/-- The position `bU n g` is below `512`. -/
theorem bU_lt {n g : Nat} (_hn : n < 256) (hg : g < 4) : bU n g < 512 := by
  unfold bU
  omega

/-- The position `bPos n g` is inside `b`'s scale block. -/
theorem bPos_lt {n g : Nat} (hn : n < 256) (hg : g < 4) : bPos n g < [1, 2, 1, 2, 256].prod := by
  unfold bPos
  simp only [List.prod_cons, List.prod_nil]
  have h := bU_lt hn hg
  omega

/-! ## The coordinates of the reshaped indices -/

theorem eA0 (p : Nat) (h : p < 512) : (Idx.ofPos [1,1,1,2,256] p h).coord 0 = 0 := by
  simp [Idx.ofPos, Idx.coord, Nat.div_eq_of_lt h]
/-- Coordinate 1 of position `p` in the scale block of a shape. -/
theorem eA1 (p : Nat) (h : p < 512) : (Idx.ofPos [1,1,1,2,256] p h).coord 1 = 0 := by
  simp [Idx.ofPos, Idx.coord, Nat.div_eq_of_lt h]
/-- Coordinate 2 of position `p` in the scale block of a shape. -/
theorem eA2 (p : Nat) (h : p < 512) : (Idx.ofPos [1,1,1,2,256] p h).coord 2 = 0 := by
  simp [Idx.ofPos, Idx.coord, Nat.div_eq_of_lt h]
/-- Coordinate 3 of position `p` in the scale block of a shape. -/
theorem eA3 (p : Nat) (h : p < 512) : (Idx.ofPos [1,1,1,2,256] p h).coord 3 = p / 256 := by
  simp [Idx.ofPos, Idx.coord, Nat.mod_eq_of_lt h]
/-- Coordinate 4 of position `p` in the scale block of a shape. -/
theorem eA4 (p : Nat) (h : p < 512) : (Idx.ofPos [1,1,1,2,256] p h).coord 4 = p % 256 := by
  simp [Idx.ofPos, Idx.coord, Nat.mod_eq_of_lt h]

/-- Coordinate 0 of position `p` in the scale block of b shape. -/
theorem eB0 (p : Nat) (h : p < 1024) : (Idx.ofPos [1,2,1,2,256] p h).coord 0 = 0 := by
  simp [Idx.ofPos, Idx.coord, Nat.div_eq_of_lt h]
/-- Coordinate 1 of position `p` in the scale block of b shape. -/
theorem eB1 (p : Nat) (h : p < 1024) : (Idx.ofPos [1,2,1,2,256] p h).coord 1 = p / 512 := by
  simp [Idx.ofPos, Idx.coord, Nat.mod_eq_of_lt h]
/-- Coordinate 2 of position `p` in the scale block of b shape. -/
theorem eB2 (p : Nat) (h : p < 1024) : (Idx.ofPos [1,2,1,2,256] p h).coord 2 = 0 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 3 of position `p` in the scale block of b shape. -/
theorem eB3 (p : Nat) (h : p < 1024) : (Idx.ofPos [1,2,1,2,256] p h).coord 3 = p % 512 / 256 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 4 of position `p` in the scale block of b shape. -/
theorem eB4 (p : Nat) (h : p < 1024) : (Idx.ofPos [1,2,1,2,256] p h).coord 4 = p % 256 := by
  simp [Idx.ofPos, Idx.coord]

/-- Coordinate 0 of position `p` in the reshaped scale of a shape. -/
theorem gA0 (p : Nat) (h : p < [1,1,32,4,4].prod) :
    (Idx.ofPos [1,1,32,4,4] p h).coord 0 = 0 := by
  simp [Idx.ofPos, Idx.coord, Nat.div_eq_of_lt (by simpa using h)]
/-- Coordinate 1 of position `p` in the reshaped scale of a shape. -/
theorem gA1 (p : Nat) (h : p < [1,1,32,4,4].prod) :
    (Idx.ofPos [1,1,32,4,4] p h).coord 1 = 0 := by
  simp [Idx.ofPos, Idx.coord, Nat.div_eq_of_lt (by simpa using h)]
/-- Coordinate 2 of position `p` in the reshaped scale of a shape. -/
theorem gA2 (p : Nat) (h : p < [1,1,32,4,4].prod) :
    (Idx.ofPos [1,1,32,4,4] p h).coord 2 = p / 16 := by
  simp [Idx.ofPos, Idx.coord, Nat.mod_eq_of_lt (by simpa using h)]
/-- Coordinate 3 of position `p` in the reshaped scale of a shape. -/
theorem gA3 (p : Nat) (h : p < [1,1,32,4,4].prod) :
    (Idx.ofPos [1,1,32,4,4] p h).coord 3 = p % 16 / 4 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 4 of position `p` in the reshaped scale of a shape. -/
theorem gA4 (p : Nat) (h : p < [1,1,32,4,4].prod) :
    (Idx.ofPos [1,1,32,4,4] p h).coord 4 = p % 4 := by
  simp [Idx.ofPos, Idx.coord]

/-- Coordinate 0 of position `p` in the permuted scale of a shape. -/
theorem hA0 (p : Nat) (h : p < [1,4,32,1,4].prod) :
    (Idx.ofPos [1,4,32,1,4] p h).coord 0 = 0 := by
  simp [Idx.ofPos, Idx.coord, Nat.div_eq_of_lt (by simpa using h)]
/-- Coordinate 1 of position `p` in the permuted scale of a shape. -/
theorem hA1 (p : Nat) (h : p < [1,4,32,1,4].prod) :
    (Idx.ofPos [1,4,32,1,4] p h).coord 1 = p / 128 := by
  simp [Idx.ofPos, Idx.coord, Nat.mod_eq_of_lt (by simpa using h)]
/-- Coordinate 2 of position `p` in the permuted scale of a shape. -/
theorem hA2 (p : Nat) (h : p < [1,4,32,1,4].prod) :
    (Idx.ofPos [1,4,32,1,4] p h).coord 2 = p % 128 / 4 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 3 of position `p` in the permuted scale of a shape. -/
theorem hA3 (p : Nat) (h : p < [1,4,32,1,4].prod) :
    (Idx.ofPos [1,4,32,1,4] p h).coord 3 = 0 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 4 of position `p` in the permuted scale of a shape. -/
theorem hA4 (p : Nat) (h : p < [1,4,32,1,4].prod) :
    (Idx.ofPos [1,4,32,1,4] p h).coord 4 = p % 4 := by
  simp [Idx.ofPos, Idx.coord]

/-- Coordinate 0 of position `p` in the reshaped scale of b shape. -/
theorem gB0 (p : Nat) (h : p < [2,1,32,4,4].prod) :
    (Idx.ofPos [2,1,32,4,4] p h).coord 0 = p / 512 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 1 of position `p` in the reshaped scale of b shape. -/
theorem gB1 (p : Nat) (h : p < [2,1,32,4,4].prod) :
    (Idx.ofPos [2,1,32,4,4] p h).coord 1 = 0 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 2 of position `p` in the reshaped scale of b shape. -/
theorem gB2 (p : Nat) (h : p < [2,1,32,4,4].prod) :
    (Idx.ofPos [2,1,32,4,4] p h).coord 2 = p % 512 / 16 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 3 of position `p` in the reshaped scale of b shape. -/
theorem gB3 (p : Nat) (h : p < [2,1,32,4,4].prod) :
    (Idx.ofPos [2,1,32,4,4] p h).coord 3 = p % 16 / 4 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 4 of position `p` in the reshaped scale of b shape. -/
theorem gB4 (p : Nat) (h : p < [2,1,32,4,4].prod) :
    (Idx.ofPos [2,1,32,4,4] p h).coord 4 = p % 4 := by
  simp [Idx.ofPos, Idx.coord]

/-- Coordinate 0 of position `p` in the permuted scale of b shape. -/
theorem hB0 (p : Nat) (h : p < [2,4,32,1,4].prod) :
    (Idx.ofPos [2,4,32,1,4] p h).coord 0 = p / 512 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 1 of position `p` in the permuted scale of b shape. -/
theorem hB1 (p : Nat) (h : p < [2,4,32,1,4].prod) :
    (Idx.ofPos [2,4,32,1,4] p h).coord 1 = p % 512 / 128 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 2 of position `p` in the permuted scale of b shape. -/
theorem hB2 (p : Nat) (h : p < [2,4,32,1,4].prod) :
    (Idx.ofPos [2,4,32,1,4] p h).coord 2 = p % 128 / 4 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 3 of position `p` in the permuted scale of b shape. -/
theorem hB3 (p : Nat) (h : p < [2,4,32,1,4].prod) :
    (Idx.ofPos [2,4,32,1,4] p h).coord 3 = 0 := by
  simp [Idx.ofPos, Idx.coord]
/-- Coordinate 4 of position `p` in the permuted scale of b shape. -/
theorem hB4 (p : Nat) (h : p < [2,4,32,1,4].prod) :
    (Idx.ofPos [2,4,32,1,4] p h).coord 4 = p % 4 := by
  simp [Idx.ofPos, Idx.coord]

/-- The flat position of `(r, g)` in a `128 x 4` tensor. -/
theorem posA (r : Fin 128) (g : Fin 4) : Idx.pos [128,4] (r,g,()) = 4 * (r:Nat) + (g:Nat) := by
  simp [Idx.pos]
  omega

/-- The flat position of `(n, g)` in a `256 x 4` tensor. -/
theorem posB (n : Fin 256) (g : Fin 4) : Idx.pos [256,4] (n,g,()) = 4 * (n:Nat) + (g:Nat) := by
  simp [Idx.pos]
  omega

/-! ## Which byte `sIdx` names -/

theorem sIdx_a {d : Inst} (_hd : d.ok) (pm t r g : Nat) (hr : r < 128) (hg : g < 4) :
    sIdx d.K (pm * 128 + r) (128 * t + 32 * g) = pm * (d.K / 128) * 512 + t * 512 + aU r g := by
  unfold sIdx aU
  have h1 : (pm * 128 + r) / 128 = pm := by omega
  have h2 : (pm * 128 + r) % 32 = r % 32 := by omega
  have h3 : (pm * 128 + r) % 128 / 32 = r / 32 := by omega
  have h4 : (128 * t + 32 * g) / 128 = t := by omega
  have h5 : (128 * t + 32 * g) % 128 / 32 = g := by omega
  rw [h1, h2, h3, h4, h5]
  ring

/-- The scale index of column `n` of a tile, group `g`, in the 5-D layout. -/
theorem sIdx_b {d : Inst} (_hd : d.ok) (pn t n g : Nat) (_hn : n < 256) (hg : g < 4) :
    sIdx d.K (pn * 256 + n) (128 * t + 32 * g) =
      (2 * pn + n / 128) * (d.K / 128) * 512 + t * 512 + bU n g := by
  unfold sIdx bU
  have h1 : (pn * 256 + n) / 128 = 2 * pn + n / 128 := by omega
  have h2 : (pn * 256 + n) % 32 = n % 32 := by omega
  have h3 : (pn * 256 + n) % 128 / 32 = n / 32 % 4 := by omega
  have h4 : (128 * t + 32 * g) / 128 = t := by
    have hg' := hg
    omega
  have h5 : (128 * t + 32 * g) % 128 / 32 = g := by omega
  rw [h1, h2, h3, h4, h5]
  ring

/-! ## The reshapes -/

theorem aScale_get {av : Tensor [1, 1, 1, 2, 256] U8}
    (h1 : [1, 1, 32, 4, 4].prod = [1, 1, 1, 2, 256].prod)
    (hp : [0, 3, 2, 1, 4].Perm (List.range 5))
    (h2 : [128, 4].prod = [1, 4, 32, 1, 4].prod) (r : Fin 128) (g : Fin 4) :
    (((av.reshape [1, 1, 32, 4, 4] h1).permute [0, 3, 2, 1, 4] hp).reshape [128, 4] h2).get
        (r, g, ()) =
      av.get (Idx.ofPos [1, 1, 1, 2, 256] (aU (r : Nat) (g : Nat)) (aU_lt r.2 g.2)) := by
  have hlt : aU (r:Nat) (g:Nat) < [1,1,32,4,4].prod := aU_lt r.2 g.2
  have ho : Idx.pos [128,4] (r,g,()) < [1,4,32,1,4].prod := h2 ▸ Idx.pos_lt [128,4] (r,g,())
  show ((av.reshape [1,1,32,4,4] h1).permute [0,3,2,1,4] hp).get
      (Idx.ofPos [1,4,32,1,4] (Idx.pos [128,4] (r,g,())) ho) =
    av.get (Idx.ofPos [1,1,1,2,256] (aU (r:Nat) (g:Nat)) (aU_lt r.2 g.2))
  rw [Tensor.get_permute (av.reshape [1,1,32,4,4] h1) [0,3,2,1,4] hp
      (Idx.ofPos [1,4,32,1,4] (Idx.pos [128,4] (r,g,())) ho)
      (Idx.ofPos [1,1,32,4,4] (aU (r:Nat) (g:Nat)) hlt) ?_]
  · rw [Tensor.get_reshape]
    exact congrArg av.get
      (Idx.ofPos_congr (Idx.pos_ofPos [1,1,32,4,4] (aU (r:Nat) (g:Nat)) hlt) _ _)
  · intro k hk
    have hk5 : k < 5 := by simpa using hk
    interval_cases k
    · show (Idx.ofPos [1,1,32,4,4] (aU (r:Nat) (g:Nat)) hlt).coord 0 =
        (Idx.ofPos [1,4,32,1,4] (Idx.pos [128,4] (r,g,())) ho).coord 0
      rw [gA0 (aU (r:Nat) (g:Nat)) hlt, hA0 (Idx.pos [128,4] (r,g,())) ho]
    · show (Idx.ofPos [1,1,32,4,4] (aU (r:Nat) (g:Nat)) hlt).coord 3 =
        (Idx.ofPos [1,4,32,1,4] (Idx.pos [128,4] (r,g,())) ho).coord 1
      rw [gA3 (aU (r:Nat) (g:Nat)) hlt, hA1 (Idx.pos [128,4] (r,g,())) ho, posA]
      have e1 : aU (r:Nat) (g:Nat) % 16 / 4 = (r:Nat) / 32 := by unfold aU; omega
      have e2 : (4 * (r:Nat) + (g:Nat)) / 128 = (r:Nat) / 32 := by omega
      rw [e1, e2]
    · show (Idx.ofPos [1,1,32,4,4] (aU (r:Nat) (g:Nat)) hlt).coord 2 =
        (Idx.ofPos [1,4,32,1,4] (Idx.pos [128,4] (r,g,())) ho).coord 2
      rw [gA2 (aU (r:Nat) (g:Nat)) hlt, hA2 (Idx.pos [128,4] (r,g,())) ho, posA]
      have e1 : aU (r:Nat) (g:Nat) / 16 = (r:Nat) % 32 := by unfold aU; omega
      have e2 : (4 * (r:Nat) + (g:Nat)) % 128 / 4 = (r:Nat) % 32 := by omega
      rw [e1, e2]
    · show (Idx.ofPos [1,1,32,4,4] (aU (r:Nat) (g:Nat)) hlt).coord 1 =
        (Idx.ofPos [1,4,32,1,4] (Idx.pos [128,4] (r,g,())) ho).coord 3
      rw [gA1 (aU (r:Nat) (g:Nat)) hlt, hA3 (Idx.pos [128,4] (r,g,())) ho]
    · show (Idx.ofPos [1,1,32,4,4] (aU (r:Nat) (g:Nat)) hlt).coord 4 =
        (Idx.ofPos [1,4,32,1,4] (Idx.pos [128,4] (r,g,())) ho).coord 4
      rw [gA4 (aU (r:Nat) (g:Nat)) hlt, hA4 (Idx.pos [128,4] (r,g,())) ho, posA]
      unfold aU
      omega

/-- Scale `(n, g)` of `b`'s block after the kernel's reshape and permute is the byte at `bPos n g`. -/
theorem bScale_get {bv : Tensor [1, 2, 1, 2, 256] U8}
    (h1 : [2, 1, 32, 4, 4].prod = [1, 2, 1, 2, 256].prod)
    (hp : [0, 3, 2, 1, 4].Perm (List.range 5))
    (h2 : [256, 4].prod = [2, 4, 32, 1, 4].prod) (n : Fin 256) (g : Fin 4) :
    (((bv.reshape [2, 1, 32, 4, 4] h1).permute [0, 3, 2, 1, 4] hp).reshape [256, 4] h2).get
        (n, g, ()) =
      bv.get (Idx.ofPos [1, 2, 1, 2, 256] (bPos (n : Nat) (g : Nat)) (bPos_lt n.2 g.2)) := by
  have hlt : bPos (n:Nat) (g:Nat) < [2,1,32,4,4].prod := bPos_lt n.2 g.2
  have ho : Idx.pos [256,4] (n,g,()) < [2,4,32,1,4].prod := h2 ▸ Idx.pos_lt [256,4] (n,g,())
  show ((bv.reshape [2,1,32,4,4] h1).permute [0,3,2,1,4] hp).get
      (Idx.ofPos [2,4,32,1,4] (Idx.pos [256,4] (n,g,())) ho) =
    bv.get (Idx.ofPos [1,2,1,2,256] (bPos (n:Nat) (g:Nat)) (bPos_lt n.2 g.2))
  rw [Tensor.get_permute (bv.reshape [2,1,32,4,4] h1) [0,3,2,1,4] hp
      (Idx.ofPos [2,4,32,1,4] (Idx.pos [256,4] (n,g,())) ho)
      (Idx.ofPos [2,1,32,4,4] (bPos (n:Nat) (g:Nat)) hlt) ?_]
  · rw [Tensor.get_reshape]
    exact congrArg bv.get
      (Idx.ofPos_congr (Idx.pos_ofPos [2,1,32,4,4] (bPos (n:Nat) (g:Nat)) hlt) _ _)
  · intro k hk
    have hk5 : k < 5 := by simpa using hk
    interval_cases k
    · show (Idx.ofPos [2,1,32,4,4] (bPos (n:Nat) (g:Nat)) hlt).coord 0 =
        (Idx.ofPos [2,4,32,1,4] (Idx.pos [256,4] (n,g,())) ho).coord 0
      rw [gB0 (bPos (n:Nat) (g:Nat)) hlt, hB0 (Idx.pos [256,4] (n,g,())) ho, posB]
      have hb := bU_lt n.2 g.2
      unfold bPos bU at *
      omega
    · show (Idx.ofPos [2,1,32,4,4] (bPos (n:Nat) (g:Nat)) hlt).coord 3 =
        (Idx.ofPos [2,4,32,1,4] (Idx.pos [256,4] (n,g,())) ho).coord 1
      rw [gB3 (bPos (n:Nat) (g:Nat)) hlt, hB1 (Idx.pos [256,4] (n,g,())) ho, posB]
      have e1 : bPos (n:Nat) (g:Nat) % 16 / 4 = (n:Nat) / 32 % 4 := by
        unfold bPos bU
        omega
      have e2 : (4 * (n:Nat) + (g:Nat)) % 512 / 128 = (n:Nat) / 32 % 4 := by omega
      rw [e1, e2]
    · show (Idx.ofPos [2,1,32,4,4] (bPos (n:Nat) (g:Nat)) hlt).coord 2 =
        (Idx.ofPos [2,4,32,1,4] (Idx.pos [256,4] (n,g,())) ho).coord 2
      rw [gB2 (bPos (n:Nat) (g:Nat)) hlt, hB2 (Idx.pos [256,4] (n,g,())) ho, posB]
      have hb := bU_lt n.2 g.2
      have e1 : bPos (n:Nat) (g:Nat) % 512 = bU (n:Nat) (g:Nat) := by
        unfold bPos
        omega
      have e2 : bU (n:Nat) (g:Nat) / 16 = (n:Nat) % 32 := by
        unfold bU
        omega
      have e3 : (4 * (n:Nat) + (g:Nat)) % 128 / 4 = (n:Nat) % 32 := by omega
      rw [e1, e2, e3]
    · show (Idx.ofPos [2,1,32,4,4] (bPos (n:Nat) (g:Nat)) hlt).coord 1 =
        (Idx.ofPos [2,4,32,1,4] (Idx.pos [256,4] (n,g,())) ho).coord 3
      rw [gB1 (bPos (n:Nat) (g:Nat)) hlt, hB3 (Idx.pos [256,4] (n,g,())) ho]
    · show (Idx.ofPos [2,1,32,4,4] (bPos (n:Nat) (g:Nat)) hlt).coord 4 =
        (Idx.ofPos [2,4,32,1,4] (Idx.pos [256,4] (n,g,())) ho).coord 4
      rw [gB4 (bPos (n:Nat) (g:Nat)) hlt, hB4 (Idx.pos [256,4] (n,g,())) ho, posB]
      unfold bPos bU
      omega

/-! ## The scale bytes -/

theorem aScale_elemAt {d : Inst} (pm t : Nat) {r g : Nat} (hr : r < 128) (hg : g < 4) :
    ((aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0]
          (Idx.ofPos [1, 1, 1, 2, 256] (aU r g) (aU_lt hr hg)))).toNat =
      pm * (d.K / 128) * 512 + t * 512 + aU r g := by
  have h : (aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0]
          (Idx.ofPos [1, 1, 1, 2, 256] (aU r g) (aU_lt hr hg))) =
      ((pm * (d.K / 128) * 512 + t * 512 + aU r g : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, aScaleDesc, List.length_cons, List.length_nil,
      List.range_succ, List.range_zero, List.map_append, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil, List.sum_append, List.nil_append, TensorDesc.coord, List.getD_cons_zero,
      List.getD_cons_succ, eA0 (aU r g) (aU_lt hr hg), eA1 (aU r g) (aU_lt hr hg),
      eA2 (aU r g) (aU_lt hr hg), eA3 (aU r g) (aU_lt hr hg), eA4 (aU r g) (aU_lt hr hg)]
    push_cast
    ring_nf
    omega
  rw [h, Int.toNat_natCast]

/-- The element of `b_scale` that the lane holding scale `(n, g)` addresses. -/
theorem bScale_elemAt {d : Inst} (pn t : Nat) {n g : Nat} (hn : n < 256) (hg : g < 4) :
    ((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0]
          (Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg)))).toNat =
      (2 * pn + n / 128) * (d.K / 128) * 512 + t * 512 + bU n g := by
  have hp : bPos n g < 1024 := by
    have := bPos_lt hn hg
    simpa using this
  have h : (bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0]
          (Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg))) =
      (((2 * pn + bPos n g / 512) * (d.K / 128) * 512 + t * 512 +
        (bPos n g % 512 / 256) * 256 + bPos n g % 256 : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bScaleDesc, TensorDesc.coord,
      List.length_cons, List.length_nil, List.range_succ, List.range_zero, List.map_append,
      List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.sum_append, List.nil_append,
      List.getD_cons_zero, List.getD_cons_succ,
      eB0 (bPos n g) hp, eB1 (bPos n g) hp, eB2 (bPos n g) hp, eB3 (bPos n g) hp,
      eB4 (bPos n g) hp]
    push_cast
    ring
  rw [h, Int.toNat_natCast]
  have hb := bU_lt hn hg
  have h1 : bPos n g / 512 = n / 128 := by unfold bPos; omega
  have h2 : bPos n g % 512 = bU n g := by unfold bPos; omega
  have h3 : bPos n g % 256 = bU n g % 256 := by unfold bPos; omega
  have h4 : bU n g / 256 * 256 + bU n g % 256 = bU n g := by omega
  rw [h1, h2, h3]
  rw [Nat.add_assoc, h4]

/-- **`a`'s scale of the lane**: the byte the loaded tile holds is `sa`'s byte. -/
theorem aScale_val {d : Inst} (hd : d.ok) {m : Mem} {pm t : Nat} {av : Tensor [1, 1, 1, 2, 256] U8}
    (hav : ∀ l : Idx [1, 1, 1, 2, 256],
      av.get l = rdU8 m d.a_scale
        ((aScaleDesc d).elemAt
          (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat)
    (r g : Nat) (hr : r < 128) (hg : g < 4) :
    DType.toBits .u8 (av.get (Idx.ofPos [1, 1, 1, 2, 256] (aU r g) (aU_lt hr hg))) =
      DType.toBits .u8 (sa d m (pm * 128 + r) (128 * t + 32 * g)) := by
  rw [hav (Idx.ofPos [1, 1, 1, 2, 256] (aU r g) (aU_lt hr hg))]
  rw [aScale_elemAt (d := d) pm t hr hg]
  show DType.toBits .u8 (rdU8 m d.a_scale (pm * (d.K / 128) * 512 + t * 512 + aU r g)) =
    DType.toBits .u8 (rdU8 m d.a_scale (sIdx d.K (pm * 128 + r) (128 * t + 32 * g)))
  rw [sIdx_a hd pm t r g hr hg]

/-- **`b`'s scale of the lane**: the byte the loaded tile holds is `sb`'s byte. -/
theorem bScale_val {d : Inst} (hd : d.ok) {m : Mem} {pn t : Nat} {bv : Tensor [1, 2, 1, 2, 256] U8}
    (hbv : ∀ l : Idx [1, 2, 1, 2, 256],
      bv.get l = rdU8 m d.b_scale
        ((bScaleDesc d).elemAt
          (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] l)).toNat)
    (n g : Nat) (hn : n < 256) (hg : g < 4) :
    DType.toBits .u8 (bv.get (Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg))) =
      DType.toBits .u8 (sb d m (pn * 256 + n) (128 * t + 32 * g)) := by
  rw [hbv (Idx.ofPos [1, 2, 1, 2, 256] (bPos n g) (bPos_lt hn hg))]
  rw [bScale_elemAt (d := d) pn t hn hg]
  show DType.toBits .u8 (rdU8 m d.b_scale ((2 * pn + n / 128) * (d.K / 128) * 512 + t * 512 + bU n g)) =
    DType.toBits .u8 (rdU8 m d.b_scale (sIdx d.K (pn * 256 + n) (128 * t + 32 * g)))
  rw [sIdx_b hd pn t n g hn hg]

end Claims.BlockScaledMxfp8
