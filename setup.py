from setuptools import setup, Extension
import numpy

NAME = 'ciffy'
VERSION = '0.1.1a'
LICENSE = 'CC BY-NC 4.0'
AUTHOR = 'Hamish M. Blair'
EMAIL = 'hmblair@stanford.edu'
URL = 'https://github.com/hmblair/ciffy'

EXT = "_c"
SOURCES = [
    'ciffy/src/_c.c',
    'ciffy/src/io.c',
    'ciffy/src/py.c',
    'ciffy/src/cif.c',
]
module = Extension(
    name=f"{NAME}.{EXT}",
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
    options={
        'bdist_wheel': {'universal': True},
    }
)
