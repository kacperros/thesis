from distutils.core import setup

from Cython.Build import cythonize

setup(
    ext_modules=cythonize("OrientedGradientLHandler.pyx"),
    requires=['Cython', 'cv2', 'scipy', 'multiprocessing', 'numpy']
)
