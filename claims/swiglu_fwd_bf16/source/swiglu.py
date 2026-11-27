"""
SwiGLU, forward
===============

The Swish-gated linear unit, elementwise: ``z = x * sigmoid(x) * y`` (``swish(x) * y``), computed
in ``float32`` from two inputs of one shape and rounded to the output's dtype. The kernel walks the
inputs as a flat array of ``T = x.numel()`` elements, ``B`` per program; element ``k`` is column
``k % D`` of row ``k // D``, with ``D = x.shape[-1]``, and each input is addressed through its own
row stride, so an inner-contiguous view (one half of ``x.chunk(2, dim=-1)``) is read in place. fla
uses it in the gated MLP of its models (``down_proj(swiglu(gate, y))``) and as an output gate.

Layouts:

* ``x``, ``y``: ``[..., D]``, ``bfloat16``, unit stride along the last dimension and one row stride
  for the flattened leading dimensions (``prepare_input``; other layouts are copied to contiguous
  first);
* ``z``: ``[..., D]``, the dtype of ``x``, contiguous (row stride ``D``), freshly allocated.

One program per block of ``B`` consecutive elements: ``grid = (cdiv(T, B),)``.

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``, file
``fla/modules/activations/ops.py`` (the CUDA path; ``triton_ascend.py`` beside it is another
backend). Here, the kernel is lines 92-102 (its ``def`` at line 93), the host ``swiglu_fwd`` lines
105-124.

* ``prepare_input``, ``alloc_output``: ``ops.py:44-74``, verbatim;
* ``swiglu_fwd_kernel``: ``ops.py:531-542`` (the ``def`` at line 533);
* ``swiglu_fwd`` (the host): adapted from ``ops.py:587-605``;
* ``SwiGLUFunction``, ``swiglu``: adapted from ``ops.py:647-665, 708``;
* ``SwiGLULinearFunction``, ``swiglu_linear``: adapted from ``ops.py:668-713``.

Changes from the source:

* ``ops.py:17-25``: only the imports this file uses (``torch``, ``torch.nn.functional``,
  ``triton``, ``triton.language``).
* ``ops.py:531``: the decorator ``@triton.autotune(configs=_activation_autotune_configs(),
  key=['D'], **autotune_cache_kwargs)`` is removed (on NVIDIA GPUs its configurations are
  ``B`` in ``[512, 1024, 2048, 4096, 8192]`` times ``num_warps`` in ``[1, 2, 4, 8, 16, 32]``,
  ``ops.py:28-41``). ``@triton.jit(do_not_specialize=['T'])`` and the kernel's signature and body
  are verbatim.
* ``ops.py:587-605`` (``swiglu_fwd``): the decorator ``@dispatch`` (fla's backend registry, which
  falls back to this function on NVIDIA GPUs) is removed; two keyword parameters ``B`` and
  ``num_warps`` (defaults ``4096`` and ``8``, one configuration of the removed list) are added
  and passed to the launch, in place of the autotuner's choice. The rest (the shape check,
  ``prepare_input``, ``T``, ``D``, the output, the grid and the arguments) is the source's.
* ``ops.py:647-665``, ``668-706`` (``SwiGLUFunction``, ``SwiGLULinearFunction``): the decorators
  ``@input_guard(no_guard_contiguous=True)`` (which only selects the inputs' device here) and
  ``@autocast_custom_fwd``, ``@autocast_custom_bwd`` (no input casts) are removed; the backward
  passes, which launch another kernel (``swiglu_fwdbwd_kernel``), raise ``NotImplementedError``.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def prepare_input(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return an input and its row stride, copying only layouts the kernels cannot address."""
    if x.ndim == 0:
        return x, 1
    shape = x.shape
    D = shape[-1]
    # compiled kernels use dense rows to avoid symbolic stride arguments
    if torch.compiler.is_compiling():
        return x.contiguous(), D
    if x.is_contiguous():
        return x, D
    strides = x.stride()
    if D > 1 and strides[-1] != 1:
        return x.contiguous(), D
    row_stride, expected_stride = None, None
    for dim in range(len(shape) - 2, -1, -1):
        size, stride = shape[dim], strides[dim]
        if size == 1:
            continue
        if row_stride is None:
            row_stride = stride
        elif stride != expected_stride:
            return x.contiguous(), D
        expected_stride = stride * size
    return x, row_stride if row_stride is not None else D


def alloc_output(x: torch.Tensor, contiguous: bool = False) -> torch.Tensor:
    if contiguous:
        return x.new_empty(x.shape)
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@triton.jit(do_not_specialize=['T'])
def swiglu_fwd_kernel(x, y, z, stride_x_row, stride_y_row, stride_z_row, T, D: tl.constexpr, B: tl.constexpr):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_y = tl.load(y + row * stride_y_row + col, mask=mask, other=0.).to(tl.float32)
    b_z = b_x * tl.sigmoid(b_x) * b_y
    tl.store(z + row * stride_z_row + col, b_z.to(z.dtype.element_ty), mask=mask)


def swiglu_fwd(x: torch.Tensor, y: torch.Tensor, output_contiguous: bool = False,
               B: int = 4096, num_warps: int = 8) -> torch.Tensor:
    assert x.shape == y.shape, f"swiglu_fwd: shape mismatch x={x.shape} y={y.shape}"
    x, stride_x_row = prepare_input(x=x)
    y, stride_y_row = prepare_input(x=y)
    T, D = x.numel(), x.shape[-1]
    z = alloc_output(x=x, contiguous=output_contiguous)
    swiglu_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        z=z,
        stride_x_row=stride_x_row,
        stride_y_row=stride_y_row,
        stride_z_row=D,
        T=T,
        D=D,
        B=B,
        num_warps=num_warps,
    )
    return z


class SwiGLUFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function.

    .. math::
        \text{SwiGLU}(x, y) = swish(x) * y = \frac{x}{1 + \exp(-x)} * y
    """

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x=x, y=y)

    @staticmethod
    def backward(ctx, dout):
        # fla: swiglu_fwdbwd(x=x, y=y, g=dout), another kernel
        raise NotImplementedError("swiglu_fwdbwd_kernel is not part of this setup")


class SwiGLULinearFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function followed by a linear transformation.

    .. math::
        \text{SwiGLULinear}(x, y, W, b) = (swish(x) * y) W + b

    This simple wrap discards the intermediate results of SwiGLU(x, y) to save memory.
    """

    @staticmethod
    def forward(ctx, x, y, weight, bias):
        z = swiglu_fwd(x=x, y=y, output_contiguous=True)
        out = F.linear(z, weight, bias)
        ctx.save_for_backward(x, y, weight)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        # fla: swiglu_fwdbwd (another kernel), or swiglu_fwd(x=x, y=y, output_contiguous=True)
        # when only the weight's gradient is needed
        raise NotImplementedError("swiglu_fwdbwd_kernel is not part of this setup")


swiglu = SwiGLUFunction.apply


def swiglu_linear(x, y, weight, bias):
    return SwiGLULinearFunction.apply(x, y, weight, bias)


def reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``x * sigmoid(x) * y`` in ``float64`` from the exact inputs."""
    xd, yd = x.double(), y.double()
    return xd * torch.sigmoid(xd) * yd


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for shape in [(1, 1), (7, 33), (4, 1024, 512), (2, 2048, 2 * 1376)]:
        x = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        y = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        z = swiglu(x, y)
        err = (z.double() - reference(x, y)).abs().max().item()
        print(f"{shape}: max |z - ref| = {err:.3e}")
    # the gated MLP's split: both halves of one projection, read in place
    gy = torch.randn(4, 1024, 2 * 688, device=dev, dtype=torch.bfloat16)
    g, y = gy.chunk(2, dim=-1)
    z = swiglu(g, y)
    err = (z.double() - reference(g, y)).abs().max().item()
    print(f"chunked (4, 1024, 688): max |z - ref| = {err:.3e}")
