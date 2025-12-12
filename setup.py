"""
Setup script for ciffy C extension.

Metadata is defined in pyproject.toml. This file only handles:
1. C extension compilation
2. Hash table generation before build (downloads CCD if needed)
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


# Build compile args
extra_compile_args = ['-O3']

# Enable profiling if CIFFY_PROFILE environment variable is set
if os.environ.get('CIFFY_PROFILE', '').lower() in ('1', 'true', 'yes'):
    extra_compile_args.append('-DCIFFY_PROFILE')
    print("Profiling enabled: building with -DCIFFY_PROFILE")

# C extension module
ext_module = Extension(
    name="ciffy._c",
    sources=[
        'ciffy/src/module.c',
        'ciffy/src/io.c',
        'ciffy/src/python.c',
        'ciffy/src/parser.c',
        'ciffy/src/writer.c',
        'ciffy/src/registry.c',
    ],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(
    ext_modules=[ext_module],
    cmdclass={
        'build_ext': GenerateAndBuildExt,
        'sdist': GenerateAndSdist,
    },
)
