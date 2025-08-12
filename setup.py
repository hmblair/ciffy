from setuptools import setup, Extension
import os
import re
import numpy

NAME = 'ciffy'


def _version() -> str:
    with open(os.path.join(os.path.dirname(__file__), NAME, '__init__.py')) as f:
        content = f.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Cannot find version information")


VERSION = _version()
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
    include_dirs=[numpy.get_include(), f'{NAME}/src'],
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
