from distutils.core import setup, Extension
import numpy

NAME = 'ciffy'
VERSION = '0.1.0'
LICENSE = 'CC BY-NC 4.0'
AUTHOR = 'Hamish M. Blair'
EMAIL = 'hmblair@stanford.edu'
URL = 'https://github.com/hmblair/ciffy'

MODULE = "_ciffy_c"
module = Extension(
    name=MODULE,
    sources=['ciffy/src/_ciffy_c.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3', '-march=native'],
)

setup(
    name=NAME,
    version=VERSION,
    packages=[NAME],
    ext_modules=[module],
    install_requires=['numpy'],
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
)
