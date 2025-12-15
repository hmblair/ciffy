"""
Setup script for ciffy C extension.

Metadata is defined in pyproject.toml. This file only handles:
1. C extension compilation
2. Hash table generation before build (downloads CCD if needed)

For CUDA support, see setup_cuda.py which must be run separately:
    pip install -e .                              # builds C extension
    python setup_cuda.py build_ext --inplace      # builds CUDA extension
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
import os
import sys
import subprocess
import shutil
import gzip
import numpy


# URL for the PDB Chemical Component Dictionary
CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"


def download_ccd(dest_path):
    """Download and decompress the CCD file."""
    import urllib.request

    print(f"Downloading CCD from {CCD_URL}...")
    gz_path = dest_path + ".gz"

    try:
        urllib.request.urlretrieve(CCD_URL, gz_path)
        print("Decompressing CCD...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
        print(f"CCD downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download CCD: {e}")
        if os.path.exists(gz_path):
            os.remove(gz_path)
        return False


def get_ccd_path():
    """Get path to CCD file, downloading if necessary."""
    # Check environment variable first
    ccd_path = os.environ.get("CIFFY_CCD_PATH")
    if ccd_path and os.path.exists(ccd_path):
        return ccd_path

    # Use centralized cache location
    cache_dir = os.path.expanduser("~/.cache/ciffy")
    ccd_path = os.path.join(cache_dir, "components.cif")

    if os.path.exists(ccd_path):
        return ccd_path

    # Download to cache directory
    os.makedirs(cache_dir, exist_ok=True)
    if download_ccd(ccd_path):
        return ccd_path

    return None


def generate_hash_tables(force=False):
    """Run the hash table generator.

    Args:
        force: If True, regenerate even if files exist (for sdist builds)
    """
    generate_script = os.path.join(
        os.path.dirname(__file__),
        'ciffy', 'src', 'codegen', 'generate.py'
    )
    hash_dir = os.path.join(os.path.dirname(__file__), 'ciffy', 'src', 'hash')

    if not os.path.exists(generate_script):
        print("Warning: generate.py not found, skipping hash generation")
        return

    # Check if hash files already exist (users installing from PyPI)
    atom_c = os.path.join(hash_dir, 'atom.c')
    if os.path.exists(atom_c) and not force:
        print("Using pre-generated hash files")
        return

    # Need to generate - get CCD file
    ccd_path = get_ccd_path()
    if not ccd_path:
        if os.path.exists(atom_c):
            print("Warning: CCD not available, using existing hash files")
            return
        else:
            print("ERROR: CCD file required but not found. Set CIFFY_CCD_PATH or download from:")
            print(f"  {CCD_URL}")
            return

    # Check if gperf is available (need 3.1+ for constants-prefix)
    gperf_path = None
    for path in ["/opt/homebrew/bin/gperf", "/usr/local/bin/gperf"]:
        if os.path.exists(path):
            gperf_path = path
            break
    if gperf_path is None:
        gperf_path = shutil.which("gperf")

    args = [ccd_path]
    if gperf_path is None:
        print("Warning: gperf not found, using pre-generated .c files if available")
        print("Install gperf to regenerate: brew install gperf (macOS) or apt install gperf (Linux)")
        args.append("--skip-gperf")
    else:
        args.extend(["--gperf-path", gperf_path])

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


class GenerateAndBuildExt(build_ext):
    """Custom build_ext that generates hash tables before compiling."""

    def run(self):
        generate_hash_tables(force=False)
        super().run()


class GenerateAndSdist(sdist):
    """Custom sdist that ensures hash tables are generated before packaging."""

    def run(self):
        # Force regeneration for sdist to ensure latest definitions
        generate_hash_tables(force=True)
        super().run()


# ============================================================================
# Compiler configuration
# ============================================================================

def check_openmp_available():
    """
    Check if OpenMP is available by trying to compile a simple test program.

    Returns:
        Tuple of (compile_args, link_args) or (None, None) if not available
    """
    import tempfile

    if sys.platform == 'darwin':
        # macOS: need libomp from Homebrew
        compile_args = ['-Xpreprocessor', '-fopenmp']
        link_args = ['-lomp']

        # Find libomp
        for libomp_path in ['/opt/homebrew/opt/libomp/lib', '/usr/local/opt/libomp/lib']:
            if os.path.exists(libomp_path):
                link_args.append(f'-L{libomp_path}')
                include_path = libomp_path.replace('/lib', '/include')
                if os.path.exists(include_path):
                    compile_args.append(f'-I{include_path}')
                break
        else:
            # libomp not found
            return None, None
    else:
        # Linux/Windows
        compile_args = ['-fopenmp']
        link_args = ['-fopenmp']

    # Try to compile a test program
    test_code = '#include <omp.h>\nint main() { return omp_get_num_threads(); }'
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(test_code)
            test_file = f.name

        import sysconfig
        cc = os.environ.get('CC', sysconfig.get_config_var('CC') or 'cc')
        # Handle cases where CC might be "cc -pthread" etc.
        cc = cc.split()[0]

        result = subprocess.run(
            [cc] + compile_args + [test_file, '-o', '/dev/null'] + link_args,
            capture_output=True,
            timeout=30
        )
        os.unlink(test_file)
        if result.returncode == 0:
            return compile_args, link_args
    except Exception:
        pass

    return None, None


# Build compile args
extra_compile_args = ['-O3']
extra_link_args = []

# Enable profiling if CIFFY_PROFILE environment variable is set
if os.environ.get('CIFFY_PROFILE', '').lower() in ('1', 'true', 'yes'):
    extra_compile_args.append('-DCIFFY_PROFILE')
    print("Profiling enabled: building with -DCIFFY_PROFILE")

# Enable OpenMP unless CIFFY_NO_OPENMP is set
if os.environ.get('CIFFY_NO_OPENMP', '').lower() not in ('1', 'true', 'yes'):
    omp_compile, omp_link = check_openmp_available()
    if omp_compile and omp_link:
        extra_compile_args.extend(omp_compile)
        extra_link_args.extend(omp_link)
        print("OpenMP enabled for parallel Z-matrix construction")
    else:
        print("OpenMP not available (install libomp on macOS: brew install libomp)")
        print("Building without OpenMP - Z-matrix construction will be single-threaded")
else:
    print("OpenMP disabled via CIFFY_NO_OPENMP")

# ============================================================================
# C extension module
# ============================================================================

c_sources = [
    'ciffy/src/module.c',
    'ciffy/src/pyutils.c',
    # CIF I/O module
    'ciffy/src/cif/io.c',
    'ciffy/src/cif/parser.c',
    'ciffy/src/cif/writer.c',
    'ciffy/src/cif/registry.c',
    # Internal coordinates C extension
    'ciffy/src/internal/geometry.c',
    'ciffy/src/internal/batch.c',
    'ciffy/src/internal/graph.c',
    'ciffy/src/internal/internal_module.c',
]

# Validate source files exist
missing_sources = [src for src in c_sources if not os.path.exists(src)]
if missing_sources:
    print(f"ERROR: Missing source files: {missing_sources}")
    print("Make sure you're building from the ciffy root directory.")
    sys.exit(1)

ext_module = Extension(
    name="ciffy._c",
    sources=c_sources,
    include_dirs=[numpy.get_include(), 'ciffy/src'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c',  # Explicitly specify C (not C++)
)

# ============================================================================
# Build extensions list
# ============================================================================

# The C extension is always built with standard setuptools
ext_modules = [ext_module]
cmdclass = {
    'build_ext': GenerateAndBuildExt,
    'sdist': GenerateAndSdist,
}

# CUDA extension is built separately via setup_cuda.py
# This avoids conflicts between setuptools C compiler and PyTorch's BuildExtension
# Users with CUDA should run:
#   pip install -e .                              # builds C extension
#   python setup_cuda.py build_ext --inplace      # builds CUDA extension

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
