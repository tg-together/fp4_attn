from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='nvfp4sim',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'nvfp4sim',
            ['wrapper.cpp', 'nvfp4sim.cu'],
            extra_compile_args={
                'cxx': ['-O0', '-g'],        # for wrapper.cpp
                'nvcc': ['-O0', '-g', '-G']  # for nvfp4sim.cu
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)