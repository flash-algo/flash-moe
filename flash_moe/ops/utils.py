import functools
import importlib
import operator

from typing import Callable

import torch

from packaging.version import Version

try:
    import triton.language as tl
except ImportError:
    tl = None

try:
    import cuda.tile as ct
except ImportError:
    ct = None


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def infer_device():
    """
    Get current device name based on available devices
    """
    # Works for both NVIDIA and AMD
    if torch.cuda.is_available():
        return "cuda"
    # Intel XPU if available
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def get_amp_custom_fwd_bwd() -> Callable:
    device = infer_device()
    if compare_version("torch", operator.ge, "2.4.0"):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    if hasattr(torch, "npu") and getattr(torch.npu, "amp", None) is not None:
        return torch.npu.amp.custom_fwd, torch.npu.amp.custom_bwd
    return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd


amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()


torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


torch_to_cutile_dtype = {
    torch.float32: ct.float32,
    torch.float16: ct.float16,
    torch.bfloat16: ct.bfloat16,
}
