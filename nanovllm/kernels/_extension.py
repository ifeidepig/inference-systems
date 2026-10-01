from __future__ import annotations

import os
import site
import sys
import importlib
from functools import lru_cache
from pathlib import Path


def _select_cuda_home() -> None:
    """Prefer the CUDA toolkit bundled with the active Conda environment."""
    if "CUDA_HOME" in os.environ:
        return

    environment_root = Path(sys.executable).resolve().parent.parent
    if (environment_root / "bin" / "nvcc").is_file():
        os.environ["CUDA_HOME"] = str(environment_root)


def _cuda_include_paths() -> list[str]:
    """Collect headers from CUDA components installed as Python wheels."""
    include_paths: list[str] = []
    for site_packages in site.getsitepackages():
        nvidia_root = Path(site_packages) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        include_paths.extend(
            str(path)
            for path in sorted(nvidia_root.glob("*/include"))
            if path.is_dir()
        )
    return include_paths


@lru_cache(maxsize=1)
def load_cuda_ops() -> None:
    try:
        importlib.import_module("nanovllm._C")
        return
    except ImportError:
        pass

    _select_cuda_home()

    # Import after CUDA_HOME is selected: cpp_extension resolves CUDA_HOME at
    # module import time.
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).resolve().parents[1] / "csrc"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

    load(
        name="nanovllm_cuda_ops",
        sources=[
            str(csrc / "ops.cpp"),
            str(csrc / "add_kernel.cu"),
            str(csrc / "fused_add_rmsnorm_kernel.cu"),
            str(csrc / "gdn_decode_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        extra_include_paths=_cuda_include_paths(),
        with_cuda=True,
        is_python_module=True,
        verbose=False,
    )
