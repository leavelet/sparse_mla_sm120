import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

cuda_sources = [
    os.path.join(csrc_dir, "binding.cpp"),
    os.path.join(csrc_dir, "kernel", "decode", "decode_launch.cu"),
    os.path.join(csrc_dir, "kernel", "combine", "combine_kernel.cu"),
    os.path.join(csrc_dir, "kernel", "prefill", "prefill_stub.cu"),
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-gencode=arch=compute_120f,code=sm_120f",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v",
        "-DNDEBUG",
        "-lineinfo",
        "--threads",
        "8",
    ],
}

setup(
    name="flash_mla_sm120",
    version="0.1.0",
    packages=["flash_mla_sm120"],
    ext_modules=[
        CUDAExtension(
            name="flash_mla_sm120.cuda",
            sources=cuda_sources,
            include_dirs=[csrc_dir],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
