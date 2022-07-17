from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='sphere_conv',
      ext_modules=[
          CUDAExtension('sphere_conv_cuda',
                        [
                            'src/sphere_conv_cuda.cpp',
                            'src/sphere_conv_cuda_kernel.cu',
                        ]),
      ],
      cmdclass={'build_ext': BuildExtension})
