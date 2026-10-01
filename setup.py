from pathlib import Path
import site

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "nanovllm" / "csrc"


def cuda_include_paths() -> list[str]:
    paths: list[str] = []
    for site_packages in site.getsitepackages():
        nvidia_root = Path(site_packages) / "nvidia"
        if nvidia_root.is_dir():
            paths.extend(
                str(path)
                for path in sorted(nvidia_root.glob("*/include"))
                if path.is_dir()
            )
    return paths


setup(
    ext_modules=[
        CUDAExtension(
            name="nanovllm._C",
            sources=[
                str(CSRC / "ops.cpp"),
                str(CSRC / "add_kernel.cu"),
                str(CSRC / "fused_add_rmsnorm_kernel.cu"),
                str(CSRC / "gdn_decode_kernel.cu"),
            ],
            include_dirs=cuda_include_paths(),
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
)
