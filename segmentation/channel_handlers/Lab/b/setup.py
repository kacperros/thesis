from distutils.core import setup

from Cython.Build import cythonize

setup(
    ext_modules=cythonize("OrientedGradientbHandler.pyx"),
    requires=['Cython', 'cv2', 'scipy', 'multiprocessing', 'numpy']
)
