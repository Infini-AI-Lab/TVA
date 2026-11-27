"""
Log-sigmoid, forward
====================

``y = logsigmoid(x) / temperature``, elementwise, computed in ``float32`` as
``(min(0, x) - log(1 + exp(-|x|))) / temperature`` and stored in the dtype of ``x``. One program
handles ``B`` consecutive elements of the flattened tensor (``T = x.numel()`` elements in rows of
``D = x.shape[-1]``); element ``k`` is column ``k % D`` of row ``k // D``. fla uses it as the
``'logsigmoid'`` entry of its activation table ``ACT2FN`` (gate functions and feature maps of its
linear-attention layers), with a ``temperature`` dividing the result as GLA-style gate normalisers
do.

Layouts:

* ``x``: ``[..., D]``, unit stride along the last dimension and one row stride ``stride_x_row`` for
  the flattened rows (``prepare_input``: a contiguous tensor, or an inner-contiguous view such as
  one half of ``x.chunk(2, dim=-1)``; other layouts are copied first), ``bfloat16``;
* ``y``: a fresh contiguous tensor of the shape and dtype of ``x`` (row stride ``D``).

Provenance: flash-linear-attention (fla) v0.6.0, commit ``413ad2c655``. Here, the helpers ``exp``
and ``log`` are lines 53-56, ``prepare_input`` and ``alloc_output`` lines 59-89, the kernel lines
92-103, the host lines 106-121.

* ``exp``, ``log``: ``fla/ops/utils/op.py:44-49`` (the branch taken when the environment variable
  ``FLA_USE_FAST_OPS`` is not ``'1'``, its default), verbatim; ``exp2`` (lines 46-47), which the
  kernel does not call, is left out.
* ``prepare_input``, ``alloc_output``: ``fla/modules/activations/ops.py:44-74``, verbatim.
* ``logsigmoid_fwd_kernel``: ``fla/modules/activations/ops.py:248-260`` (the kernel's ``def`` at
  line 250).
* ``logsigmoid_fwd`` (the host): adapted from ``fla/modules/activations/ops.py:288-302``.

Changes from the source:

* ``fla/modules/activations/ops.py:248``: the decorator ``@triton.autotune(configs=
  _activation_autotune_configs(), key=['D'], **autotune_cache_kwargs)`` (over ``B`` in ``[512,
  1024, 2048, 4096, 8192]`` and ``num_warps`` in ``[1, 2, 4, 8, 16, 32]`` on NVIDIA GPUs,
  ``ops.py:28-41``) is removed. ``@triton.jit(do_not_specialize=['T'])`` and the kernel's
  signature and body are verbatim.
* ``fla/modules/activations/ops.py:288-302`` (``logsigmoid_fwd``): the decorator ``@dispatch``
  (fla's backend registry, which falls back to this function on NVIDIA GPUs) is removed; the launch
  passes one configuration of the autotune list, ``B=1024`` and ``num_warps=4``, as explicit
  arguments, so the grid is ``(triton.cdiv(T, 1024),)`` in place of the lambda over ``meta['B']``.
  The rest (``prepare_input``, the output allocation and the launch's keywords) is the wrapper's.
* ``reference`` and the test under ``__main__`` are new.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))
@triton.jit
def log(x): return tl.log(x.to(tl.float32))


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
def logsigmoid_fwd_kernel(x, y, stride_x_row, stride_y_row, temperature, T, D: tl.constexpr, B: tl.constexpr):
    i_n = tl.program_id(0).to(tl.int64)
    offs = i_n * B + tl.arange(0, B)
    mask = offs < T
    row = offs // D
    col = offs % D
    b_x = tl.load(x + row * stride_x_row + col, mask=mask, other=0.).to(tl.float32)
    b_m = tl.minimum(0., b_x)
    b_z = 1. + exp(-tl.abs(b_x))
    b_y = (b_m - log(b_z)) / temperature
    tl.store(y + row * stride_y_row + col, b_y.to(y.dtype.element_ty), mask=mask)


def logsigmoid_fwd(x: torch.Tensor, temperature: float = 1., output_contiguous: bool = False) -> torch.Tensor:
    x, stride_x_row = prepare_input(x=x)
    T, D = x.numel(), x.shape[-1]
    y = alloc_output(x=x, contiguous=output_contiguous)
    logsigmoid_fwd_kernel[(triton.cdiv(T, 1024),)](
        x=x,
        y=y,
        stride_x_row=stride_x_row,
        stride_y_row=D,
        temperature=temperature,
        T=T,
        D=D,
        B=1024,
        num_warps=4,
    )
    return y


def reference(x: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    """``logsigmoid(x) / temperature`` in ``float64``."""
    return torch.nn.functional.logsigmoid(x.double()) / temperature


if __name__ == "__main__":
    torch.manual_seed(42)
    for shape, temperature, noncontiguous in [((1, 1, 64), 1.0, False), ((2, 500, 128), 0.5, False),
                                              ((3, 2048, 1200), 2.0, False), ((2, 500, 128), 0.5, True),
                                              ((3, 2048, 1200), 2.0, True), ((63,), 1.0, False),
                                              ((4, 2048, 4096), 16.0, False)]:
        if noncontiguous:
            x = torch.randn(*shape[:-1], 2 * shape[-1], device='cuda', dtype=torch.bfloat16).chunk(2, dim=-1)[0]
        else:
            x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        y = logsigmoid_fwd(x, temperature)
        err = (y.double() - reference(x, temperature)).abs().max().item()
        print(shape, temperature, noncontiguous, f"max abs err {err:.3e}")
