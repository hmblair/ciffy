## Overview

`ciffy` is a C program for rapidly reading `mmCIF` files into Python.

## Installation

### From PyPI

```bash
pip install ciffy
```

### From Source

```bash
git clone https://github.com/hmblair/ciffy.git
cd ciffy
pip install -r requirements.txt
pip install torch-scatter --no-build-isolation
pip install -e .
```

### Note on torch-scatter

`ciffy` requires `torch-scatter`, which may need to be installed separately depending on your platform:

**Linux/Windows with CUDA:**
```bash
# Replace CUDA with your version (e.g., cu118, cu121) or cpu
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+${CUDA}.html
```

**macOS or if pre-built wheels are unavailable:**
```bash
pip install torch-scatter --no-build-isolation
```
