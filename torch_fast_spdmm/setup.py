from pathlib import Path

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


def find_cpp_files(directories):
    out = []
    for dir in directories:
        dir = Path(dir)
        # Cast the Path objects to strings
        out.extend(dir.rglob('*.cpp'))
        out.extend(dir.rglob('*.cu'))

    return out


cpp_files = find_cpp_files(['bindings', 'kernels', 'pbr_matrix'])
include_dirs = [os.path.join(os.getcwd())]

compile_cuda = bool(os.environ.get("CUDA_ENABLED", True))
if compile_cuda:
    print("Compiling with CUDA support.")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    extension = CUDAExtension(name='torch_fast_spdmm_core',
                              sources=cpp_files,
                              include_dirs=include_dirs,
                              extra_compile_args={'nvcc': ['-std=c++20', '-lineinfo'],
                                                  'cxx': ['-DCUDA_ENABLED=1', '-O3', '-std=c++20', '-w']})
else:
    print("Compiling without CUDA support.")
    cpp_files = [f for f in cpp_files if not f.suffix == '.cu']
    extension = CppExtension(name='torch_fast_spdmm_core',
                             sources=cpp_files,
                             include_dirs=include_dirs,
                             extra_compile_args={'cxx': ['-O3', '-std=c++20', '-w'], })

setup(name='torch_fast_spdmm_core',
      version='0.0.1',
      ext_modules=[extension],
      cmdclass={'build_ext': BuildExtension},
      packages=['torch_fast_spdmm'])
