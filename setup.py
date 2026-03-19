import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

csrc_dir = os.path.join(os.path.dirname(__file__), "sparse_mla_sm120", "csrc")

cuda_sources = [
    os.path.join(csrc_dir, "binding.cpp"),
    os.path.join(csrc_dir, "sparse_mla_decode.cu"),
    os.path.join(csrc_dir, "sparse_mla_prefill.cu"),
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v",
        "-DNDEBUG",
        "-lineinfo",
        "--threads", "8",
    ],
}

setup(
    name="sparse_mla_sm120",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="sparse_mla_sm120._C",
            sources=cuda_sources,
            include_dirs=[csrc_dir],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
