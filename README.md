[![PyPI](https://img.shields.io/pypi/v/mqt-core-plugins-catalyst?logo=pypi&style=flat-square)](https://pypi.org/project/mqt-core-plugins-catalyst/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/core-plugins-catalyst/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/munich-quantum-toolkit/core-plugins-catalyst/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/core-plugins-catalyst/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/munich-quantum-toolkit/core-plugins-catalyst/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/core-plugins-catalyst?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/core-plugins-catalyst)
[![codecov](https://img.shields.io/codecov/c/github/munich-quantum-toolkit/core-plugins-catalyst?style=flat-square&logo=codecov)](https://codecov.io/gh/munich-quantum-toolkit/core-plugins-catalyst)

> [!NOTE]
> This project is intended primarily as a demonstration and learning resource.
> It is provided for educational purposes and may not be suitable for production
> use.

<p align="center">
  <a href="https://mqt.readthedocs.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-dark.svg" width="60%">
      <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-light.svg" width="60%" alt="MQT Logo">
    </picture>
  </a>
</p>

# MLIR-Based MQT Core / Catalyst Plugin

This package provides a [Catalyst](https://github.com/PennyLaneAI/catalyst)
plugin based on [MLIR](https://mlir.llvm.org/). It allows you to use
[MQT Core](https://github.com/munich-quantum-toolkit/core)'s MLIR dialects and
transformations within Xanadu's
[Catalyst](https://github.com/PennyLaneAI/catalyst) framework.

If you have any questions, feel free to create a
[discussion](https://github.com/munich-quantum-toolkit/core-plugins-catalyst/discussions)
or an
[issue](https://github.com/munich-quantum-toolkit/core-plugins-catalyst/issues)
on [GitHub](https://github.com/munich-quantum-toolkit/core-plugins-catalyst).

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by
the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the
[Technical University of Munich](https://www.tum.de/) and supported by
[MQSC](https://mq.sc). Among others, it is part of the
[Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss)
ecosystem, which is being developed as part of the
[Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make the MLIR-based MQT Core /
Catalyst plugin a reality!

<p align="center">
  <a href="https://github.com/munich-quantum-toolkit/core-plugins-catalyst/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/core-plugins-catalyst" alt="Contributors to munich-quantum-toolkit/core-plugins-catalyst" />
  </a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the
future. We are firmly committed to keeping it open and actively maintained for
the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories:
  <https://github.com/munich-quantum-toolkit>
- Contributing code, documentation, tests, or examples via issues and pull
  requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see
  [References](https://mqt.readthedocs.io/projects/core-plugins-catalyst/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: <https://github.com/sponsors/munich-quantum-toolkit>

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

## Getting Started

Released versions of `mqt-core-plugins-catalyst` are available on
[PyPI](https://pypi.org/project/mqt.core.plugins.catalyst/).
This development baseline targets an unreleased Catalyst commit, so it must be
built from source until a compatible Catalyst release is published. The source
build uses one verified revision triple for Catalyst, MQT Core, and LLVM. Public
artifact publication is intentionally disabled for this baseline because its
exact nightly dependencies are not available from public PyPI. The manual CD
workflow produces verification artifacts only; publishing can resume once
compatible dependency releases are publicly installable.

### 1) Clone the project and bootstrap the exact toolchain

The bootstrap script installs a pre-built MLIR from
[setup-mlir](https://github.com/munich-quantum-software/setup-mlir), checks out
the exact Catalyst source, builds its pinned dependencies and local wheel
without building LLVM, and installs it into the project environment. Existing
artifacts under `.cache` are reused.

```bash
git clone https://github.com/munich-quantum-toolkit/core-plugins-catalyst.git
cd core-plugins-catalyst
./scripts/bootstrap.sh
```

### 2) Install the plugin from source

Install the exact Catalyst build and the plugin into the project environment:

```bash
uv sync --inexact --only-group build --only-group test \
  --no-install-project --no-install-package pennylane-catalyst
uv sync --inexact --no-dev \
  --no-build-isolation-package mqt-core-plugins-catalyst \
  --no-install-package pennylane-catalyst
```

In VS Code, the equivalent maintained sequence is **Bootstrap Toolchain**,
**Install Project Dependencies**, then **Install Debug Plugin**.

### 3) Use the MQT plugin and explore intermediate MLIR representations

The MQT plugin provides device configuration utilities to prevent Catalyst from
decomposing gates into unitary matrices, enabling lossless roundtrip
conversions.

You can inspect direct `CatalystQuantum → QCO → CatalystQuantum` round trips or
insert Core's `QCO → QC → QCO` bridge. The current conversion supports static
qnode circuit regions; arbitrary hybrid Catalyst control flow must be lowered
before conversion.

#### Example: Create a test script

Create a file `test_example.py`:

```python
from __future__ import annotations
from pathlib import Path
from typing import Any

import pennylane as qml
from catalyst.passes import apply_pass
from mqt.core.plugins.catalyst import get_catalyst_plugin_abs_path, get_device

# Use get_device() to configure the device for MQT plugin compatibility
device = get_device("lightning.qubit", wires=2)
plugin_path = str(get_catalyst_plugin_abs_path())


# Define your quantum circuit
@apply_pass("qco-to-catalystquantum")
@apply_pass("catalystquantum-to-qco")
@qml.qnode(device)
def circuit() -> None:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])


# Custom pipeline to capture intermediate MLIR
custom_pipeline = [
    ("Init", ["builtin.module(canonicalize)"]),  # Initial Catalyst MLIR
    ("ToQCO", ["builtin.module(catalystquantum-to-qco)"]),
    ("ToCatalystQuantum", ["builtin.module(qco-to-catalystquantum)"]),
]


# JIT compilation with intermediate MLIR files saved
@qml.qjit(
    target="mlir",
    autograph=True,
    keep_intermediate=2,
    pipelines=custom_pipeline,
    pass_plugins={plugin_path},
    dialect_plugins={plugin_path},
)
def module() -> Any:
    return circuit()


# Trigger compilation and optimized MLIR generation
module.mlir_opt

# Catalyst writes all intermediate stages into one module workspace
workspaces = [path for path in Path.cwd().glob("module*") if path.is_dir()]
if len(workspaces) != 1:
    raise RuntimeError(f"Expected one Catalyst workspace, found {workspaces}")
mlir_dir = workspaces[0]
mlir_init = mlir_dir / "1_AfterInit.mlir"
mlir_to_qco = mlir_dir / "2_AfterToQCO.mlir"
mlir_to_catalyst = mlir_dir / "3_AfterToCatalystQuantum.mlir"

# Read MLIR files
print("=== Initial Catalyst MLIR ===")
if mlir_init.exists():
    print(mlir_init.read_text())

print("\n=== After CatalystQuantum → QCO conversion ===")
if mlir_to_qco.exists():
    print(mlir_to_qco.read_text())

print("\n=== After QCO → CatalystQuantum roundtrip ===")
if mlir_to_catalyst.exists():
    print(mlir_to_catalyst.read_text())
```

**Alternative:** You can also configure an existing device:

```python
from mqt.core.plugins.catalyst import configure_device_for_mqt

device = qml.device("lightning.qubit", wires=2)
device = configure_device_for_mqt(device)
```

#### Run the example

```bash
uv run --no-sync test_example.py
```

You should see three MLIR representations showing the transformation through the
MQT dialects and back.

#### Verify the installation

You can run the test suite to verify everything is working:

```bash
# Run pytest using uv
uv run --no-sync pytest test -v
```

```bash
# Alternatively run the tests using nox (handles all dependencies automatically)
uvx nox -s tests
```

## System Requirements

Building the MQT Core Catalyst Plugin requires a C++ compiler with support for
C++20 and CMake 3.24 or newer. Building (and running) is continuously tested
under Linux and macOS using the
[latest available system versions for GitHub Actions](https://github.com/actions/runner-images).
The MQT Core Catalyst Plugin is compatible with Python versions 3.12 through
3.14.

The MQT Core Catalyst Plugin relies on some external dependencies:

- [llvm/llvm-project](https://github.com/llvm/llvm-project): A toolkit for the
  construction of highly optimized compilers, optimizers, and run-time
  environments (specific revision: `8f264586d7521b0e305ca7bb78825aa3382ffef7`).
- [PennyLaneAI/catalyst](https://github.com/PennyLaneAI/catalyst): Source
  revision `56a96d261c3ef70949967f6bcfa95ef1dec12d14` (`0.16.0-dev77`).
- [MQT Core](https://github.com/munich-quantum-toolkit/core): Source revision
  `756d3c17fec1ff478cae04622560da532ba61a02`, providing the QCO and QC dialects
  and their conversion passes.

The bootstrap verifies the LLVM and Catalyst revisions while building Catalyst
against the cached `setup-mlir` SDK. CMake fetches and verifies the exact MQT
Core revision when configuring the plugin.

## Cite This

If you want to cite MQT Core Catalyst Plugin, please use the following BibTeX
entry:

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

The Munich Quantum Toolkit has been supported by the European Research Council
(ERC) under the European Union's Horizon 2020 research and innovation program
(grant agreement No. 101001318), the Bavarian State Ministry for Science and
Arts through the Distinguished Professorship Program, as well as the Munich
Quantum Valley, which is supported by the Bavarian state government with funds
from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
