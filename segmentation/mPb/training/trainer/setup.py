from distutils.core import setup

from Cython.Build import cythonize

setup(
    ext_modules=cythonize("AngledmPbTrainer.pyx"),
    requires=['Cython', 'cv2', 'scipy', 'numpy', 'os', 'time']
)
