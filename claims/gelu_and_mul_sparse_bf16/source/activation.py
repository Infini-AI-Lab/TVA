"""
Gaussian-sparse GELU-and-multiply (Gemma 3n's activation sparsity)
==================================================================

For each row of ``x = [gate | up]`` (``2 * d`` elements), the kernel computes the mean and the
(biased) standard deviation of ``gate``, the cutoff ``mean + std * std_multiplier``, the sparse gate
``relu(gate - cutoff)``, and stores ``gelu_tanh(sparse_gate) * up`` (``d`` elements). Every
intermediate is rounded to the input dtype where PyTorch's native implementation rounds it.

One program per row; ``BLOCK_SIZE = next_power_of_2(d)`` covers the whole row.

Layouts:

* ``x``: ``[..., 2 * d]``, contiguous; row ``r`` holds ``gate`` at elements ``r * 2d .. r * 2d + d``
  and ``up`` at ``r * 2d + d .. r * 2d + 2d``;
* ``output``: ``[..., d]``, contiguous, of ``x``'s dtype, allocated fresh.

Provenance: vLLM commit ``4ba787ec99``, ``vllm/kernels/triton/activation.py``:

* ``_gelu_and_mul_sparse_kernel``: line 16 (its decorator) to line 52;
* ``_supports_gelu_and_mul_sparse``: line 55;
* the host ``_gelu_and_mul_sparse_triton_op``: line 73 (its decorator) to line 99;
* ``gelu_and_mul_sparse_triton``: line 102 (its decorator) to line 111.

Changes from the source:

* The imports (``activation.py:4-11``): ``triton`` and ``triton.language as tl`` are imported
  directly instead of through ``vllm.triton_utils``; ``torch.library``, ``vllm.ir`` and
  ``vllm.platforms`` are not imported (their uses are removed below).
* ``_gelu_and_mul_sparse_kernel`` (``activation.py:16-52``) and
  ``_supports_gelu_and_mul_sparse`` (``activation.py:55-70``) are copied verbatim, with the
  constants ``_MAX_INTERMEDIATE_SIZE`` and ``_SUPPORTED_DTYPES`` (``activation.py:12-13``).
* The host ``_gelu_and_mul_sparse_triton_op`` (``activation.py:73-99``) is copied without its
  ``@triton_op("vllm::gelu_and_mul_sparse_triton", mutates_args=())`` decorator (a registration
  with ``torch.library``), and the launch ``wrap_triton(_gelu_and_mul_sparse_kernel)[grid](...)``
  is written ``_gelu_and_mul_sparse_kernel[grid](...)`` (``wrap_triton`` makes the launch
  traceable by ``torch.compile``; eagerly it is this launch). Its body is otherwise verbatim.
* ``gelu_and_mul_sparse_triton`` (``activation.py:102-111``) is copied without its
  ``@ir.ops.gelu_and_mul_sparse.register_impl(...)`` decorator (vLLM's dispatch registry, which
  selects this implementation on CUDA when ``_supports_gelu_and_mul_sparse`` accepts the
  arguments); its body is verbatim.
* ``reference`` (PyTorch's native ``gelu_and_mul_sparse`` of ``vllm/ir/ops/activation.py:12-22``,
  in ``float64``) and the test under ``__main__`` are new.
"""

import torch
from torch import Tensor

import triton
import triton.language as tl

_MAX_INTERMEDIATE_SIZE = 32768
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _gelu_and_mul_sparse_kernel(
    x_ptr,
    output_ptr,
    d: tl.constexpr,
    std_multiplier,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d
    row_start = row * 2 * d

    gate = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
    # TODO: Investigate delaying this load to reduce spills for large reductions
    # without regressing smaller shapes.
    up = tl.load(x_ptr + row_start + d + offsets, mask=mask, other=0.0).to(tl.float32)

    input_dtype = x_ptr.dtype.element_ty
    mean = tl.sum(gate, axis=0) / d
    centered = tl.where(mask, gate - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / d
    # Match the input-dtype rounding at each native PyTorch op boundary.
    mean = mean.to(input_dtype).to(tl.float32)
    std = tl.sqrt(variance).to(input_dtype).to(tl.float32)
    scaled_std = (std * std_multiplier).to(input_dtype).to(tl.float32)
    cutoff = (mean + scaled_std).to(input_dtype).to(tl.float32)
    sparse_gate = (gate - cutoff).to(input_dtype).to(tl.float32)
    sparse_gate = tl.where(sparse_gate < 0.0, 0.0, sparse_gate)

    inner = 0.7978845608028654 * (
        sparse_gate + 0.044715 * sparse_gate * sparse_gate * sparse_gate
    )
    tanh = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    activated = 0.5 * sparse_gate * (1.0 + tanh)
    activated = activated.to(input_dtype).to(tl.float32)
    tl.store(output_ptr + row * d + offsets, activated * up, mask=mask)


def _supports_gelu_and_mul_sparse(
    x: Tensor, std_multiplier: float, approximate: str = "none"
) -> bool:
    del std_multiplier
    if (
        x.device.type != "cuda"
        or x.dtype not in _SUPPORTED_DTYPES
        or not x.is_contiguous()
        or x.ndim == 0
        or approximate != "tanh"
    ):
        return False
    last_dim = x.shape[-1]
    return (
        last_dim > 0 and last_dim % 2 == 0 and last_dim // 2 <= _MAX_INTERMEDIATE_SIZE
    )


def _gelu_and_mul_sparse_triton_op(x: Tensor, std_multiplier: float) -> Tensor:
    d = x.shape[-1] // 2
    output = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
    if output.numel() == 0:
        return output

    n_rows = x.numel() // x.shape[-1]

    def grid(meta):
        return (n_rows,)

    block_size = triton.next_power_of_2(d)
    num_warps = min(max(block_size // 256, 1), 8)
    _gelu_and_mul_sparse_kernel[grid](
        x,
        output,
        d=d,
        std_multiplier=std_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def gelu_and_mul_sparse_triton(
    x: Tensor, std_multiplier: float, approximate: str = "none"
) -> Tensor:
    assert approximate == "tanh"
    return _gelu_and_mul_sparse_triton_op(x, std_multiplier)


def reference(x: Tensor, std_multiplier: float) -> Tensor:
    """PyTorch's native ``gelu_and_mul_sparse`` (``approximate="tanh"``), in ``float64``."""
    xd = x.double()
    d = xd.shape[-1] // 2
    gate = xd[..., :d]
    mean = torch.mean(gate, dim=-1, keepdim=True)
    std = torch.std(gate, dim=-1, keepdim=True, unbiased=False)
    sparse_gate = torch.relu(gate - (mean + std * std_multiplier))
    return torch.nn.functional.gelu(sparse_gate, approximate="tanh") * xd[..., d:]


if __name__ == "__main__":
    torch.manual_seed(0)
    # Gemma 3n's multiplier: Normal(0, 1).icdf(0.95) in float32, as GeluAndMulSparse computes it.
    m = torch.distributions.normal.Normal(0, 1).icdf(torch.tensor(0.95)).item()
    for tokens, d in [(1, 8192), (7, 16384), (33, 4096), (5, 100)]:
        x = torch.randn(tokens, 2 * d, device="cuda", dtype=torch.bfloat16)
        assert _supports_gelu_and_mul_sparse(x, m, "tanh")
        y = gelu_and_mul_sparse_triton(x, m, "tanh")
        ref = reference(x, m)
        err = (y.double() - ref).abs().max().item()
        print(f"tokens {tokens:3d} d {d:5d}: max |y - ref| = {err:.3e}")
