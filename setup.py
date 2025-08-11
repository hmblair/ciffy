from distutils.core import setup, Extension
import numpy

NAME = 'ciffy'
VERSION = '0.1.0'
LICENSE = 'CC BY-NC 4.0'
AUTHOR = 'Hamish M. Blair'
EMAIL = 'hmblair@stanford.edu'
URL = 'https://github.com/hmblair/ciffy'

MODULE = "_ciffy_c"
SOURCES = [
    'ciffy/src/_ciffy_c.c',
    'ciffy/src/io.c',
    'ciffy/src/py.c',
    'ciffy/src/cif.c',
]
module = Extension(
    name=MODULE,
    sources=SOURCES,
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3'],
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
