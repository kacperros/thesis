from distutils.core import setup

from Cython.Build import cythonize

setup(
    ext_modules=cythonize("BSDSTrainingDTO.pyx"),
    requires=['Cython', 'cv2', 'scipy', 'numpy', 'os']
)
