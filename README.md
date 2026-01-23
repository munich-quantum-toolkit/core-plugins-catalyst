[![PyPI](https://img.shields.io/pypi/v/mqt.core?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.core/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/JOSS-10.21105/joss.07478-blue.svg?style=flat-square)](https://doi.org/10.21105/joss.07478)
[![CI](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/core-plugins-catalyst/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/munich-quantum-toolkit/core-plugins-catalyst/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/core-plugins-catalyst/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/munich-quantum-toolkit/core-plugins-catalyst/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/core-plugins-catalyst?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/core-plugins-catalyst)
[![codecov](https://img.shields.io/codecov/c/github/munich-quantum-toolkit/core-plugins-catalyst?style=flat-square&logo=codecov)](https://codecov.io/gh/munich-quantum-toolkit/core-plugins-catalyst)

> [!NOTE]
> This project is intended primarily as a demonstration and learning resource.
> It is provided for educational purposes and may not be suitable for production use.

<p align="center">
  <a href="https://mqt.readthedocs.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-dark.svg" width="60%">
      <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-light.svg" width="60%" alt="MQT Logo">
    </picture>
  </a>
</p>

# MLIR-Based MQT Core / Catalyst Plugin

This package provides a [Catalyst](https://github.com/PennyLaneAI/catalyst) plugin based on [MLIR](https://mlir.llvm.org/).
It allows you to use [MQT Core](https://github.com/munich-quantum-toolkit/core)'s MLIR dialects and transformations within Xanadu's [Catalyst](https://github.com/PennyLaneAI/catalyst) framework.

If you have any questions, feel free to create a [discussion](https://github.com/munich-quantum-toolkit/core-plugins-catalyst/discussions) or an [issue](https://github.com/munich-quantum-toolkit/core-plugins-catalyst/issues) on [GitHub](https://github.com/munich-quantum-toolkit/core-plugins-catalyst).

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make the MLIR-based MQT Core / Catalyst plugin a reality!

<p align="center">
  <a href="https://github.com/munich-quantum-toolkit/core-plugins-catalyst/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/core-plugins-catalyst" alt="Contributors to munich-quantum-toolkit/core-plugins-catalyst" />
  </a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: https://github.com/munich-quantum-toolkit
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see [References](https://mqt.readthedocs.io/projects/core-plugins-catalyst/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: https://github.com/sponsors/munich-quantum-toolkit

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

## Getting Started

`mqt.core.plugins.catalyst` is **NOT YET** available on [PyPI](https://pypi.org/project/mqt.core/).

Because `pennylane-catalyst` pins to a specific LLVM/MLIR revision, you must build that LLVM/MLIR locally and point CMake at it.

### 1) Build the exact LLVM/MLIR revision (locally)

```bash
# Pick a workspace (optional)
mkdir -p ~/dev && cd ~/dev

# Clone the exact LLVM revision Catalyst expects
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 113f01aa82d055410f22a9d03b3468fa68600589

# Configure & build MLIR (Release is recommended)
cmake -S llvm -B build_llvm -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_BUILD_TESTS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZLIB=FORCE_ON \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_VISIBILITY_PRESET=default

cmake --build build_llvm --config Release

# Export these for your shell/session
export MLIR_DIR="$PWD/build_llvm/lib/cmake/mlir"
export LLVM_DIR="$PWD/build_llvm/lib/cmake/llvm"
```

### 2) Clone the repository and create a local environment

```bash
# Clone the repository
cd ~/dev  # or wherever you want to work
git clone https://github.com/munich-quantum-toolkit/core-plugins-catalyst.git
cd core-plugins-catalyst

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install Catalyst first
uv pip install pennylane-catalyst==0.14.0
```

### 3) Build the plugin

Make sure `MLIR_DIR` and `LLVM_DIR` are still set from step 1. Then build the plugin:

```bash
# Build the plugin with uv (all on one line or with backslashes)
uv sync --verbose --active \
  --config-settings=cmake.define.CMAKE_BUILD_TYPE=Release \
  --config-settings=cmake.define.Python3_EXECUTABLE="$(which python)" \
  --config-settings=cmake.define.MLIR_DIR="$MLIR_DIR" \
  --config-settings=cmake.define.LLVM_DIR="$LLVM_DIR"
```

### 4) Use the MQT plugin and explore intermediate MLIR representations

The MQT plugin provides device configuration utilities to prevent Catalyst from decomposing gates into unitary matrices, enabling lossless roundtrip conversions.

You can inspect the intermediate MLIR representations during the roundtrip between `CatalystQuantum` and `MQTOpt` dialects.

#### Example: Create a test script

Create a file `test_example.py`:

```python
from __future__ import annotations
from pathlib import Path
from typing import Any

import pennylane as qml
from catalyst.passes import apply_pass
from mqt.core.plugins.catalyst import get_device

# Use get_device() to configure the device for MQT plugin compatibility
device = get_device("lightning.qubit", wires=2)


# Define your quantum circuit
@apply_pass("mqt.mqtopt-to-catalystquantum")
@apply_pass("mqt.catalystquantum-to-mqtopt")
@qml.qnode(device)
def circuit() -> None:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])


# Custom pipeline to capture intermediate MLIR
custom_pipeline = [
    ("Init", ["builtin.module(canonicalize)"]),  # Initial Catalyst MLIR
    ("ToMQTOpt", ["builtin.module(catalystquantum-to-mqtopt)"]),
    ("ToCatalystQuantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
]


# JIT compilation with intermediate MLIR files saved
@qml.qjit(target="mlir", autograph=True, keep_intermediate=2, pipelines=custom_pipeline)
def module() -> Any:
    return circuit()


# Trigger compilation and optimized MLIR generation
module.mlir_opt

# Catalyst writes intermediate MLIR files to the current working directory
mlir_dir = Path.cwd()
mlir_init = mlir_dir / "1_AfterInit.mlir"
mlir_to_mqtopt = mlir_dir / "2_AfterToMQTOpt.mlir"
mlir_to_catalyst = mlir_dir / "3_AfterToCatalystQuantum.mlir"

# Read MLIR files
print("=== Initial Catalyst MLIR ===")
if mlir_init.exists():
    print(mlir_init.read_text())

print("\n=== After CatalystQuantum → MQTOpt conversion ===")
if mlir_to_mqtopt.exists():
    print(mlir_to_mqtopt.read_text())

print("\n=== After MQTOpt → CatalystQuantum roundtrip ===")
if mlir_to_catalyst.exists():
    print(mlir_to_catalyst.read_text())
```

#### Run the example

Make sure your virtual environment is activated:

```bash
# Activate the virtual environment (if not already active)
source .venv/bin/activate  # On macOS/Linux

# Run the example
python test_example.py
```

You should see three MLIR representations showing the transformation through the MQT dialects and back.

#### Verify the installation

You can also run the test suite to verify everything is working:

```bash
# Make sure your venv is activated
source .venv/bin/activate

# Run the tests
pytest test/test_plugin.py -v
```

**Alternative:** You can also configure an existing device:

```python
from mqt.core.plugins.catalyst import configure_device_for_mqt

device = qml.device("lightning.qubit", wires=2)
device = configure_device_for_mqt(device)
```

## System Requirements

Building the MQT Core Catalyst Plugin requires a C++ compiler with support for C++20 and CMake 3.24 or newer.
Building (and running) is continuously tested under Linux and macOS using the [latest available system versions for GitHub Actions](https://github.com/actions/runner-images).
The MQT Core Catalyst Plugin is compatible with Python version 3.11 and newer.

The MQT Core Catalyst Plugin relies on some external dependencies:

- [llvm/llvm-project](https://github.com/llvm/llvm-project): A toolkit for the construction of highly optimized compilers, optimizers, and run-time environments (specific revision: `113f01aa82d055410f22a9d03b3468fa68600589`).
- [PennyLaneAI/catalyst](https://github.com/PennyLaneAI/catalyst): A package that enables just-in-time (JIT) compilation of hybrid quantum-classical programs implemented with PennyLane (version == 0.14.0).
- [MQT Core](https://github.com/munich-quantum-toolkit/core-plugins-catalyst): Provides the MQTOpt MLIR dialect and supporting infrastructure.

Note, both LLVM/MLIR and Catalyst are currently restricted to specific versions. You must build LLVM/MLIR locally from the exact revision specified above and configure CMake to use it (see installation instructions).

## Cite This

If you want to cite MQT Core Catalyst Plugin, please use the following BibTeX entry:

```bibtex
@inproceedings{Hopf_Integrating_Quantum_Software_2026,
author = {Hopf, Patrick and Ochoa Lopez, Erick and Stade, Yannick and Rovara, Damian and Quetschlich, Nils and Florea, Ioan Albert and Izaac, Josh and Wille, Robert and Burgholzer, Lukas},
booktitle = {SCA/HPCAsia 2026: Supercomputing Asia and International Conference on High Performance Computing in Asia Pacific Region},
doi = {10.1145/3773656.3773658},
month = jan,
publisher = {Association for Computing Machinery},
series = {SCA/HPCAsia 2026},
title = {{Integrating Quantum Software Tools with(in) MLIR}},
year = {2026}
}
```

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European
Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement
No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the
Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
