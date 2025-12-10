from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import re
import sys
import subprocess
import shutil
import numpy

NAME = 'ciffy'


class GenerateAndBuildExt(build_ext):
    """Custom build_ext that generates hash tables before compiling."""

    def run(self):
        self.generate_hash_tables()
        super().run()

    def generate_hash_tables(self):
        """Run the hash table generator before building."""
        generate_script = os.path.join(
            os.path.dirname(__file__),
            'ciffy', 'src', 'codegen', 'generate.py'
        )

        if not os.path.exists(generate_script):
            print("Warning: generate.py not found, skipping hash generation")
            return

        # Check if gperf is available (need 3.1+ for constants-prefix)
        # Check Homebrew paths first (they have newer versions)
        gperf_path = None
        for path in ["/opt/homebrew/bin/gperf", "/usr/local/bin/gperf"]:
            if os.path.exists(path):
                gperf_path = path
                break
        if gperf_path is None:
            gperf_path = shutil.which("gperf")

        if gperf_path is None:
            print("Warning: gperf not found, using pre-generated hash files")
            print("Install gperf to regenerate: brew install gperf (macOS) or apt install gperf (Linux)")
            # Still generate .gperf files and reverse.h (they don't need gperf)
            args = ["--skip-gperf"]
        else:
            args = ["--gperf-path", gperf_path]

        print("Generating hash lookup tables...")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(__file__)

        result = subprocess.run(
            [sys.executable, generate_script] + args,
            env=env,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: Hash generation failed: {result.stderr}")
        else:
            print(result.stdout)

# Cross-platform OpenMP configuration
# Set CIFFY_NO_OPENMP=1 to disable OpenMP (useful for PyTorch compatibility)
USE_OPENMP = os.environ.get('CIFFY_NO_OPENMP', '0') != '1'

if USE_OPENMP:
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
else:
    # Disable OpenMP - define NO_OPENMP to skip omp.h include
    OMP_COMPILE_ARGS = ['-DNO_OPENMP']
    OMP_LINK_ARGS = []
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
    'ciffy/src/module.c',
    'ciffy/src/io.c',
    'ciffy/src/python.c',
    'ciffy/src/parser.c',
    'ciffy/src/writer.c',
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
    cmdclass={'build_ext': GenerateAndBuildExt},
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
