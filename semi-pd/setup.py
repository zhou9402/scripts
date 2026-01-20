from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils import cpp_extension
import os

def get_cuda_extension():
    """创建 CUDA Green Context 扩展"""
    
    # 获取 CUDA 路径
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    if cuda_home is None:
        raise RuntimeError("CUDA_HOME 环境变量未设置")
    
    # 包含目录
    include_dirs = [
        f"{cuda_home}/include",
        torch.utils.cpp_extension.include_paths(),
        pybind11.get_include(),
    ]
    
    # 库目录
    library_dirs = [
        f"{cuda_home}/lib64",
        f"{cuda_home}/lib",
    ]
    
    # 链接库
    libraries = ["cuda", "cudart"]
    
    # 定义扩展
    ext = cpp_extension.CUDAExtension(
        name="green_context_lib",
        sources=["green_context_wrapper.cu"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "--expt-extended-lambda", "-std=c++17"]
        }
    )
    
    return ext

setup(
    name="green_context_lib",
    version="0.1.0",
    description="CUDA Green Context Python Library for FlashInfer",
    author="Your Name",
    ext_modules=[get_cuda_extension()],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "pybind11>=2.6.0",
    ],
) 