# setup file to compile C++ library

from setuptools import setup
import torch, os
from torch.utils.cpp_extension import CppExtension, BuildExtension

this_dir = os.path.dirname(os.path.realpath(__file__))
include_dir = this_dir + '/models/cpp/'
extra = {'cxx': ['-std=c++11']}

setup(name='ordercpp',
        #packages=['ordercpp'],
        ext_modules=[
                CppExtension('ordercpp',
                        ['models/cpp/order.cpp',
                        'models/cpp/pybind.cpp'],
                        include_dirs=[include_dir],
                        extra_compile_args=extra['cxx']
                        )
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False
)
