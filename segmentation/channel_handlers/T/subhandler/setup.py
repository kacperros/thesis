from distutils.core import setup

from Cython.Build import cythonize

setup(
    ext_modules=cythonize("OrientedGradientTSubhandlers.pyx"),
    requires=['Cython', 'cv2', 'multiprocessing']
)
