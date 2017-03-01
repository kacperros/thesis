from distutils.core import setup

from Cython.Build import cythonize

setup(
    ext_modules=cythonize("RotationAdapter.pyx"), requires=['Cython', 'scipy', 'numpy']
)
