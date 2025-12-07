from setuptools import setup, Extension
import os
import re
import sys
import numpy

NAME = 'ciffy'

# Cross-platform OpenMP configuration
if sys.platform == 'darwin':  # macOS
    # Requires: brew install libomp
    OMP_COMPILE_ARGS = ['-Xpreprocessor', '-fopenmp']
    OMP_LINK_ARGS = ['-lomp']
    # Homebrew libomp paths (Apple Silicon and Intel)
    HOMEBREW_PREFIX = os.environ.get('HOMEBREW_PREFIX', '/opt/homebrew')
    OMP_INCLUDE_DIRS = [f'{HOMEBREW_PREFIX}/opt/libomp/include']
    OMP_LIBRARY_DIRS = [f'{HOMEBREW_PREFIX}/opt/libomp/lib']
else:  # Linux
    OMP_COMPILE_ARGS = ['-fopenmp']
    OMP_LINK_ARGS = ['-fopenmp']
    OMP_INCLUDE_DIRS = []
    OMP_LIBRARY_DIRS = []


def _version() -> str:
    with open(os.path.join(os.path.dirname(__file__), NAME, '__init__.py')) as f:
        content = f.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Cannot find version information")


def _readme() -> str:
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        return f.read()


VERSION = _version()
DESCRIPTION = 'Fast CIF file parsing for molecular structures'
LONG_DESCRIPTION = _readme()
LICENSE = 'MIT'
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
    include_dirs=[numpy.get_include()] + OMP_INCLUDE_DIRS,
    library_dirs=OMP_LIBRARY_DIRS,
    extra_compile_args=['-O3'] + OMP_COMPILE_ARGS,
    extra_link_args=OMP_LINK_ARGS,
)

PACKAGES = [
    NAME,
    f'{NAME}.backend',
    f'{NAME}.utils',
    f'{NAME}.types',
    f'{NAME}.biochemistry',
    f'{NAME}.operations',
    f'{NAME}.io',
]

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Chemistry',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: C',
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=PACKAGES,
    ext_modules=[module],
    python_requires='>=3.9',
    install_requires=[
        'numpy',
    ],
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
)
