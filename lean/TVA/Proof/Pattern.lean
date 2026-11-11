import TVA.Proof.Pattern.Block
import TVA.Proof.Pattern.TileOffset
import TVA.Proof.Pattern.Tile
import TVA.Proof.Pattern.BlockPtr
import TVA.Proof.Pattern.Desc

/-!
# Memory access patterns

A kernel's loads and stores address buffer elements through a few patterns. For each, this kit
proves, once, which element every lane addresses (`Buf.Addrs`, `TVA.Proof.Buf`), and wraps the
buffer rules `safe_load_buf`, `safe_store_buf` (`TVA.Proof.Rules.Memory`) into a rule stated in the
pattern's own terms, with a footprint a claim's proof can compare across programs.

* `Pattern/Block`: `offs = pid * BLOCK + tl.arange(0, BLOCK)`, `mask = offs < n`.
  `safe_load_block`, `safe_store_block`; footprint `Buf.tile`; `Block.Fits` (`fits_of_grid_pow2`).
* `Pattern/TileOffset`: `base + rows[:, None] * s0 + cols[None, :] * s1`. Lane offsets and
  pointers (`off_toInt`, `ptr_lane_on`, …); rows `r0 + tl.arange(0, BM)`, in `int64`, or `% M`
  (`safe_irem_lanes`).
* `Pattern/Tile`: the masked 2-D tile, `mask = (rows < M)[:, None] & (cols < N)[None, :]`.
  `safe_load_tileAt`, `safe_store_tileAt`; footprints `Buf.tileAt`, `Buf.tile2`, `Buf.matrix`;
  layouts that do not alias (`Tile2.Inj`) and disjoint tiles.
* `Pattern/BlockPtr`: `tl.make_block_ptr`, `tl.load(p, boundary_check=…)`, `tl.advance`.
  `safe_load_block_ptr`, `safe_store_block_ptr`; `BlockPtr.elem_eq`.
* `Pattern/Desc`: `desc.load`, `desc.store`, `desc.atomic_*`, `desc.gather`, `desc.scatter`.
  `safe_desc_load`, `safe_desc_store`, `safe_desc_atomic`, `safe_desc_gather`,
  `safe_desc_scatter`.

An access that fits none of them proves `Buf.Addrs` from its lanes' pointers
(`Buf.addrs_of_get`) and uses the buffer rules directly. The kit builds on `TVA.Proof.Basic` and
`TVA.Proof.Int`; claims' proofs use it, and nothing else in the library depends on it.
-/
