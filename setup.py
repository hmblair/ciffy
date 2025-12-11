"""
Setup script for ciffy C extension.

Metadata is defined in pyproject.toml. This file only handles:
1. C extension compilation
2. Hash table generation before build
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess
import shutil
import numpy


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
    extra_compile_args=['-O3'],
)

setup(
    ext_modules=[ext_module],
    cmdclass={'build_ext': GenerateAndBuildExt},
)
