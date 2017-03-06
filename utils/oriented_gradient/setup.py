from distutils.core import setup

from Cython.Build import cythonize

setup(
    ext_modules=cythonize("OrientedGradientCalculator.py"),
    requires=['Cython', 'cv2', 'scipy', 'multiprocessing', 'numpy']
)
